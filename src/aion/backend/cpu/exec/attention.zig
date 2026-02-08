const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");

const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");

const exec_utils = @import("utils.zig");
const fast_math = @import("../kernels/fast_math.zig");
const simd = @import("../kernels/simd.zig");
const attn_kernels = @import("../kernels/attention.zig");
const attention_registry = @import("../registry/attention_registry.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

const simd_lanes: usize = attn_kernels.simd_lanes;
const Vec = attn_kernels.Vec;
const vecLoad = attn_kernels.vecLoad;
const vecStore = attn_kernels.vecStore;

pub fn execAttentionTiled(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    kernels: attention_registry.Kernels,
    s: executable.StepAttentionTiled,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    if (!(s.scale > 0.0) or !std.math.isFinite(s.scale)) return BackendError.InvalidArgument;

    const out_meta = try store.meta(s.out);
    const q_meta = try store.meta(s.q);
    const k_meta = try store.meta(s.k);
    const v_meta = try store.meta(s.v);

    if (out_meta.rank != 2 or q_meta.rank != 2 or k_meta.rank != 2 or v_meta.rank != 2) return BackendError.InvalidArgument;
    // v1: attention is f32-only. (The legacy f16 path was slow and has been removed.)
    if (out_meta.dtype != .f32) return BackendError.Unsupported;
    if (q_meta.dtype != .f32 or k_meta.dtype != .f32 or v_meta.dtype != .f32) return BackendError.InvalidArgument;

    // Shapes: q:[m,dk], k:[n,dk], v:[n,dv], out:[m,dv]
    if (q_meta.shape[1] != k_meta.shape[1]) return BackendError.InvalidArgument;
    if (k_meta.shape[0] != v_meta.shape[0]) return BackendError.InvalidArgument;
    if (out_meta.shape[0] != q_meta.shape[0]) return BackendError.InvalidArgument;
    if (out_meta.shape[1] != v_meta.shape[1]) return BackendError.InvalidArgument;

    // Tiling contract as validated by graph/program.zig.
    const tm: usize = out_meta.tile_shape[0];
    const tv: usize = out_meta.tile_shape[1];
    const tn: usize = k_meta.tile_shape[0];

    // Validate V tiling matches K and output.
    if (v_meta.tile_shape[0] != tn) return BackendError.InvalidArgument;
    if (v_meta.tile_shape[1] != tv) return BackendError.InvalidArgument;
    if (v_meta.tile_counts[0] != k_meta.tile_counts[0]) return BackendError.InvalidArgument;
    if (v_meta.tile_counts[1] != out_meta.tile_counts[1]) return BackendError.InvalidArgument;

    if (tm == 0 or tm > 256) return BackendError.InvalidArgument;
    if (tn == 0 or tn > 128) return BackendError.InvalidArgument;
    if (tv == 0 or tv > 64) return BackendError.InvalidArgument;
    if (q_meta.tile_shape[1] == 0) return BackendError.InvalidArgument;

    const ti_q_total: usize = out_meta.tile_counts[0];
    const tile_bytes: usize = exec_utils.tileByteSize(out_meta);
    const min_total_bytes: usize = 256 * 1024;

    if (pool) |p| {
        if (exec_utils.shouldParallelTiles(thread_count, ti_q_total, tile_bytes, min_total_bytes) and ti_q_total >= 2) {
            const Task = struct {
                store: tensor_store.TensorStore,
                out_meta: tensor_store.TensorMeta,
                q_meta: tensor_store.TensorMeta,
                k_meta: tensor_store.TensorMeta,
                s: executable.StepAttentionTiled,
                kernels: attention_registry.Kernels,

                stop: std.atomic.Value(bool) = .init(false),
                err_mutex: std.Thread.Mutex = .{},
                err_any: ?anyerror = null,

                fn fail(t: *@This(), err: anyerror) void {
                    if (t.stop.swap(true, .acq_rel)) return;
                    t.err_mutex.lock();
                    defer t.err_mutex.unlock();
                    if (t.err_any == null) t.err_any = err;
                }

                fn runTiQ(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
                    _ = tid;
                    const t: *@This() = @ptrCast(@alignCast(ctx_any));
                    if (start >= end) return;
                    if (t.stop.load(.acquire)) return;
                    var ti_q: usize = start;
                    while (ti_q < end) : (ti_q += 1) {
                        if (t.stop.load(.acquire)) return;
                        attentionTileAllV_Stream_F32(t.store, t.out_meta, t.q_meta, t.k_meta, t.s, t.kernels, ti_q) catch |e| {
                            t.fail(e);
                            return;
                        };
                    }
                }
            };

            var task: Task = .{ .store = store, .out_meta = out_meta, .q_meta = q_meta, .k_meta = k_meta, .s = s, .kernels = kernels };
            p.parallelForAny(@ptrCast(&task), ti_q_total, 1, Task.runTiQ);
            if (task.err_any) |e| return @errorCast(e);
            return;
        }
    }

    // Sequential fallback.
    var ti_q: usize = 0;
    while (ti_q < ti_q_total) : (ti_q += 1) {
        try attentionTileAllV_Stream_F32(store, out_meta, q_meta, k_meta, s, kernels, ti_q);
    }
}

fn attentionTileAllV_Stream_F32(
    store: tensor_store.TensorStore,
    out_meta: tensor_store.TensorMeta,
    q_meta: tensor_store.TensorMeta,
    k_meta: tensor_store.TensorMeta,
    s: executable.StepAttentionTiled,
    kernels: attention_registry.Kernels,
    ti_q: usize,
) ExecuteProgramError!void {
    std.debug.assert(out_meta.dtype == .f32);

    const tm: usize = out_meta.tile_shape[0];
    const tn: usize = k_meta.tile_shape[0];

    const m_total: usize = out_meta.shape[0];
    const n_total: usize = k_meta.shape[0];

    const q_row0: usize = ti_q * tm;
    const m_tile: usize = if (q_row0 >= m_total) 0 else @min(tm, m_total - q_row0);
    if (m_tile == 0) return;

    // Per-row softmax stats.
    var m_state: [256]f32 = undefined;
    var l_state: [256]f32 = undefined;
    @memset(m_state[0..m_tile], -std.math.inf(f32));
    @memset(l_state[0..m_tile], 0.0);

    const tc_k: usize = k_meta.tile_counts[0];
    const tc_dk: usize = q_meta.tile_counts[1];

    // Scratch for one key tile: scores (tm x tn) and packed K/Q panels.
    var scores: [256 * 128]f32 = undefined;
    const scores_len: usize = tm * tn;
    var kt_scratch: [128 * 128]f32 = undefined;
    var qt_scratch: [256 * 128]f32 = undefined;

    const ti_v_total: usize = out_meta.tile_counts[1];
    var out_tiles: [256]tensor_store.TileRefMut = undefined;
    var out_views: [256]types.BufferViewMut = undefined;
    var out_strides: [256]usize = undefined;
    var out_len: usize = 0;

    if (ti_v_total > out_tiles.len) return BackendError.InvalidArgument;
    var ti_v: usize = 0;
    while (ti_v < ti_v_total) : (ti_v += 1) {
        var out_t = try store.acquireTileMut(s.out, ti_q, ti_v);
        const ov0 = out_t.bufferView();
        if (ov0.dtype != .f32 or ov0.layout.rank != 2) {
            store.releaseMut(out_t.token);
            return BackendError.InvalidArgument;
        }
        if ((@intFromPtr(ov0.bytes.ptr) & (@alignOf(f32) - 1)) != 0) {
            store.releaseMut(out_t.token);
            return BackendError.InvalidArgument;
        }
        const out_s0: []align(@alignOf(f32)) f32 = @alignCast(simd.bytesAsSliceMutUnaligned(f32, ov0.bytes));
        @memset(out_s0, 0.0);

        out_tiles[out_len] = out_t;
        out_views[out_len] = ov0;
        out_strides[out_len] = @as(usize, ov0.layout.shape[1]);
        out_len += 1;
    }
    defer {
        var i: usize = 0;
        while (i < out_len) : (i += 1) store.releaseMut(out_tiles[i].token);
    }

    // One-pass streaming softmax.
    var ti_k: usize = 0;
    while (ti_k < tc_k) : (ti_k += 1) {
        const key_row0: usize = ti_k * tn;
        if (s.causal) {
            const max_q_idx: usize = q_row0 + m_tile - 1;
            if (key_row0 > max_q_idx) continue;
        }

        @memset(scores[0..scores_len], 0.0);

        var ti_dk: usize = 0;
        while (ti_dk < tc_dk) : (ti_dk += 1) {
            const qt = try store.acquireTileConst(s.q, ti_q, ti_dk);
            defer store.releaseConst(qt.token);
            const kt = try store.acquireTileConst(s.k, ti_k, ti_dk);
            defer store.releaseConst(kt.token);

            const qv = qt.bufferView();
            const kv = kt.bufferView();
            if (qv.dtype != .f32 or kv.dtype != .f32) return BackendError.InvalidArgument;
            if (qv.layout.rank != 2 or kv.layout.rank != 2) return BackendError.InvalidArgument;

            const m_q: usize = qv.layout.shape[0];
            const k_q: usize = qv.layout.shape[1];
            const n_k: usize = kv.layout.shape[0];
            const k_k: usize = kv.layout.shape[1];
            if (m_q != m_tile) return BackendError.InvalidArgument;
            if (k_q != k_k) return BackendError.InvalidArgument;
            if (n_k > tn) return BackendError.InvalidArgument;

            const qs: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, qv.bytes);
            const ks: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, kv.bytes);

            kernels.pack_k_block(n_k, k_q, ks, k_k, &kt_scratch);
            kernels.calc_scores_f32(m_q, n_k, k_q, qs, k_q, ks, k_k, &kt_scratch, &qt_scratch, scores[0..scores_len], tn);
        }

        // Apply scale and causal mask; also clamp unused cols to -inf.
        for (0..m_tile) |r| {
            const q_idx: usize = q_row0 + r;
            const valid_cols: usize = @min(tn, n_total - key_row0);
            for (0..valid_cols) |c| {
                const k_idx: usize = key_row0 + c;
                const idx: usize = r * tn + c;
                var v0: f32 = scores[idx] * s.scale;
                if (s.causal and k_idx > q_idx) v0 = -std.math.inf(f32);
                scores[idx] = v0;
            }
            if (valid_cols < tn) {
                @memset(scores[r * tn + valid_cols .. r * tn + tn], -std.math.inf(f32));
            }
        }

        // Row max for this block.
        var row_max: [256]f32 = undefined;
        @memset(row_max[0..m_tile], -std.math.inf(f32));
        for (0..m_tile) |r| {
            var mx: f32 = -std.math.inf(f32);
            var c: usize = 0;
            if (tn >= simd_lanes) {
                var vmx: Vec = @splat(-std.math.inf(f32));
                while (c + simd_lanes <= tn) : (c += simd_lanes) {
                    vmx = @max(vmx, vecLoad(scores[r * tn + c ..].ptr));
                }
                mx = @reduce(.Max, vmx);
            }
            while (c < tn) : (c += 1) mx = @max(mx, scores[r * tn + c]);
            row_max[r] = mx;
        }

        // Compute m_new and rescale, then rescale l_state and ALL output tiles in place.
        var m_new: [256]f32 = undefined;
        var rescale: [256]f32 = undefined;
        for (0..m_tile) |r| {
            const mn: f32 = @max(m_state[r], row_max[r]);
            m_new[r] = mn;
            rescale[r] = if (m_state[r] == -std.math.inf(f32)) 0.0 else kernels.exp_softmax(m_state[r] - mn);
        }

        // Rescale previous accumulators (output tiles) and l_state.
        for (0..m_tile) |r| {
            const rs: f32 = rescale[r];
            l_state[r] *= rs;
            const v_rs: Vec = @splat(rs);

            var vi: usize = 0;
            while (vi < ti_v_total) : (vi += 1) {
                const ov = out_views[vi];
                if ((@intFromPtr(ov.bytes.ptr) & (@alignOf(f32) - 1)) != 0) return BackendError.InvalidArgument;
                const out_s: []align(@alignOf(f32)) f32 = @alignCast(simd.bytesAsSliceMutUnaligned(f32, ov.bytes));
                const out_stride: usize = out_strides[vi];

                const off: usize = r * out_stride;
                var j: usize = 0;
                const vec_end: usize = out_stride - (out_stride % simd_lanes);
                while (j < vec_end) : (j += simd_lanes) {
                    const ptr = out_s[off + j ..].ptr;
                    vecStore(ptr, vecLoad(ptr) * v_rs);
                }
                while (j < out_stride) : (j += 1) out_s[off + j] *= rs;
            }
        }

        // Compute P = exp(scores - m_new) and row_sum, overwrite scores with P.
        for (0..m_tile) |r| {
            const mn: f32 = m_new[r];
            var ssum: f32 = 0.0;

            var c: usize = 0;
            const v_mn: Vec = @splat(mn);
            var v_ssum: Vec = @splat(0.0);
            while (c + simd_lanes <= tn) : (c += simd_lanes) {
                const ptr = scores[r * tn + c ..].ptr;
                const v_s = vecLoad(ptr);
                const v_diff = fast_math.clampVecF32(simd_lanes, v_s - v_mn, -80.0, 0.0);
                const v_p = fast_math.expApproxVecF32(simd_lanes, v_diff);
                vecStore(ptr, v_p);
                v_ssum += v_p;
            }
            ssum += @reduce(.Add, v_ssum);
            while (c < tn) : (c += 1) {
                const p: f32 = kernels.exp_softmax(scores[r * tn + c] - mn);
                scores[r * tn + c] = p;
                ssum += p;
            }

            l_state[r] += ssum;
            m_state[r] = mn;
        }

        // Accumulate: out += P @ V for each V tile.
        ti_v = 0;
        while (ti_v < ti_v_total) : (ti_v += 1) {
            const vt = try store.acquireTileConst(s.v, ti_k, ti_v);
            defer store.releaseConst(vt.token);
            const vv = vt.bufferView();
            if (vv.dtype != .f32 or vv.layout.rank != 2) return BackendError.InvalidArgument;

            const n_v: usize = vv.layout.shape[0];
            const dv_v: usize = vv.layout.shape[1];
            if (n_v > tn) return BackendError.InvalidArgument;

            const vs: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, vv.bytes);

            const ov = out_views[ti_v];
            if ((@intFromPtr(ov.bytes.ptr) & (@alignOf(f32) - 1)) != 0) return BackendError.InvalidArgument;
            const out_s: []align(@alignOf(f32)) f32 = @alignCast(simd.bytesAsSliceMutUnaligned(f32, ov.bytes));
            const out_stride: usize = out_strides[ti_v];
            if (dv_v != out_stride) return BackendError.InvalidArgument;

            kernels.accumulate_values_f32(m_tile, n_v, dv_v, scores[0..scores_len], tn, vs, dv_v, out_s, out_stride);
        }
    }

    // Final normalize: out /= l_state.
    var vi2: usize = 0;
    while (vi2 < ti_v_total) : (vi2 += 1) {
        const ov = out_views[vi2];
        if ((@intFromPtr(ov.bytes.ptr) & (@alignOf(f32) - 1)) != 0) return BackendError.InvalidArgument;
        const out_s: []align(@alignOf(f32)) f32 = @alignCast(simd.bytesAsSliceMutUnaligned(f32, ov.bytes));
        const out_stride: usize = out_strides[vi2];

        for (0..m_tile) |r| {
            const denom: f32 = l_state[r];
            if (!(denom > 0.0) or !std.math.isFinite(denom)) return BackendError.InvalidArgument;
            const inv: f32 = 1.0 / denom;
            const v_inv: Vec = @splat(inv);
            const off: usize = r * out_stride;
            var j: usize = 0;
            const vec_end: usize = out_stride - (out_stride % simd_lanes);
            while (j < vec_end) : (j += simd_lanes) {
                const ptr = out_s[off + j ..].ptr;
                vecStore(ptr, vecLoad(ptr) * v_inv);
            }
            while (j < out_stride) : (j += 1) out_s[off + j] *= inv;
        }
    }
}

// NOTE: legacy grouped-V attention kernel removed.
