const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");

const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");

const exec_utils = @import("utils.zig");
const simd = @import("../kernels/simd.zig");
const attn_kernels = @import("../kernels/attention.zig");
const attention_registry = @import("../registry/attention_registry.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

const MAX_RANK: usize = 8;

const simd_lanes: usize = attn_kernels.simd_lanes;
const Vec = attn_kernels.Vec;
const vecLoad = attn_kernels.vecLoad;
const vecStore = attn_kernels.vecStore;

fn decodeLeadingCoords(
    row_idx: usize,
    row_dim_count: usize,
    row_dims: []const usize,
    row_strides: []const usize,
    coords: []usize,
) void {
    var rem: usize = row_idx;
    var rdi: usize = 0;
    while (rdi < row_dim_count) : (rdi += 1) {
        const stride0: usize = row_strides[rdi];
        const v: usize = rem / stride0;
        coords[row_dims[rdi]] = v;
        rem -= v * stride0;
    }
}

fn tileDim(shape: []const usize, tile_shape: []const usize, tile_index: usize, dim: usize) ExecuteProgramError!usize {
    if (dim >= shape.len or dim >= tile_shape.len) return BackendError.InvalidArgument;
    const start: usize = tile_index * tile_shape[dim];
    if (start >= shape[dim]) return BackendError.InvalidArgument;
    return @min(tile_shape[dim], shape[dim] - start);
}

pub fn execAttentionTiled(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    kernels: attention_registry.Kernels,
    s: executable.StepAttentionTiled,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    return execAttentionTiledND(pool, thread_count, kernels, s, store);
}

fn execAttentionTiledND(
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

    if (out_meta.rank < 2) return BackendError.InvalidArgument;
    if (q_meta.rank != out_meta.rank or k_meta.rank != out_meta.rank or v_meta.rank != out_meta.rank) return BackendError.InvalidArgument;
    // v1: attention is f32-only. (The legacy f16 path was slow and has been removed.)
    if (out_meta.dtype != .f32) return BackendError.Unsupported;
    if (q_meta.dtype != .f32 or k_meta.dtype != .f32 or v_meta.dtype != .f32) return BackendError.InvalidArgument;

    const rank: usize = @as(usize, out_meta.rank);
    const lead_dims: usize = rank - 2;

    // Shapes: q:[..., m, dk], k:[..., n, dk], v:[..., n, dv], out:[..., m, dv]
    var d: usize = 0;
    while (d < lead_dims) : (d += 1) {
        if (q_meta.shape[d] != k_meta.shape[d] or q_meta.shape[d] != v_meta.shape[d]) return BackendError.InvalidArgument;
        if (out_meta.shape[d] != q_meta.shape[d]) return BackendError.InvalidArgument;
    }
    if (q_meta.shape[rank - 1] != k_meta.shape[rank - 1]) return BackendError.InvalidArgument;
    if (k_meta.shape[rank - 2] != v_meta.shape[rank - 2]) return BackendError.InvalidArgument;
    if (out_meta.shape[rank - 2] != q_meta.shape[rank - 2]) return BackendError.InvalidArgument;
    if (out_meta.shape[rank - 1] != v_meta.shape[rank - 1]) return BackendError.InvalidArgument;

    // Batch/head dims must be tiled as size-1 so each tile is a single slice.
    d = 0;
    while (d < lead_dims) : (d += 1) {
        if (out_meta.tile_shape[d] != 1) return BackendError.InvalidArgument;
        if (q_meta.tile_shape[d] != 1 or k_meta.tile_shape[d] != 1 or v_meta.tile_shape[d] != 1) return BackendError.InvalidArgument;
        if (out_meta.tile_counts[d] != q_meta.tile_counts[d]) return BackendError.InvalidArgument;
        if (out_meta.tile_counts[d] != k_meta.tile_counts[d]) return BackendError.InvalidArgument;
        if (out_meta.tile_counts[d] != v_meta.tile_counts[d]) return BackendError.InvalidArgument;
    }

    // Tiling contract for last two dims.
    const tm: usize = out_meta.tile_shape[rank - 2];
    const tv: usize = out_meta.tile_shape[rank - 1];
    const tn: usize = k_meta.tile_shape[rank - 2];

    if (q_meta.tile_shape[rank - 2] != tm) return BackendError.InvalidArgument;
    if (q_meta.tile_shape[rank - 1] == 0) return BackendError.InvalidArgument;
    if (k_meta.tile_shape[rank - 1] != q_meta.tile_shape[rank - 1]) return BackendError.InvalidArgument;
    if (v_meta.tile_shape[rank - 2] != tn) return BackendError.InvalidArgument;
    if (v_meta.tile_shape[rank - 1] != tv) return BackendError.InvalidArgument;
    if (v_meta.tile_counts[rank - 2] != k_meta.tile_counts[rank - 2]) return BackendError.InvalidArgument;
    if (v_meta.tile_counts[rank - 1] != out_meta.tile_counts[rank - 1]) return BackendError.InvalidArgument;

    if (tm == 0 or tm > 256) return BackendError.InvalidArgument;
    if (tn == 0 or tn > 128) return BackendError.InvalidArgument;
    if (tv == 0 or tv > 64) return BackendError.InvalidArgument;

    const q_tiles_per_slice: usize = out_meta.tile_counts[rank - 2];
    const head_m: usize = out_meta.shape[rank - 2];
    const head_n: usize = k_meta.shape[rank - 2];
    const k_tile_start: usize = 0;
    const k_tile_count: usize = k_meta.tile_counts[rank - 2];

    var row_dims: [MAX_RANK]usize = undefined;
    var row_counts: [MAX_RANK]usize = undefined;
    var row_strides: [MAX_RANK]usize = undefined;
    const row_dim_count: usize = lead_dims;
    d = 0;
    while (d < lead_dims) : (d += 1) {
        row_dims[d] = d;
        row_counts[d] = out_meta.tile_counts[d];
    }

    var row_tile_total: usize = 1;
    if (lead_dims > 0) {
        var stride: usize = 1;
        var i: usize = lead_dims;
        while (i > 0) : (i -= 1) {
            const idx: usize = i - 1;
            row_strides[idx] = stride;
            stride = std.math.mul(usize, stride, row_counts[idx]) catch return BackendError.InvalidArgument;
        }
        row_tile_total = stride;
    }

    const tile_bytes: usize = exec_utils.tileByteSize(out_meta);
    const min_total_bytes: usize = 256 * 1024;
    var tile_total: usize = 1;
    d = 0;
    while (d < rank) : (d += 1) {
        tile_total *= out_meta.tile_counts[d];
    }

    const total_work: usize = row_tile_total * q_tiles_per_slice;
    if (total_work == 0) return BackendError.InvalidArgument;

    if (pool) |p| {
        if (exec_utils.shouldParallelTiles(thread_count, tile_total, tile_bytes, min_total_bytes) and total_work >= 2) {
            const Task = struct {
                store: tensor_store.TensorStore,
                out_meta: tensor_store.TensorMeta,
                q_meta: tensor_store.TensorMeta,
                k_meta: tensor_store.TensorMeta,
                v_meta: tensor_store.TensorMeta,
                s: executable.StepAttentionTiled,
                kernels: attention_registry.Kernels,
                head_m: usize,
                head_n: usize,
                k_tile_start: usize,
                k_tile_count: usize,
                row_dim_count: usize,
                row_dims: [MAX_RANK]usize,
                row_strides: [MAX_RANK]usize,
                q_tiles_per_slice: usize,

                stop: std.atomic.Value(bool) = .init(false),
                err_mutex: std.Io.Mutex = .init,
                err_any: ?anyerror = null,

                fn fail(t: *@This(), err: anyerror) void {
                    if (t.stop.swap(true, .acq_rel)) return;
                    std.Io.Threaded.mutexLock(&t.err_mutex);
                    defer std.Io.Threaded.mutexUnlock(&t.err_mutex);
                    if (t.err_any == null) t.err_any = err;
                }

                fn runWork(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
                    _ = tid;
                    const t: *@This() = @ptrCast(@alignCast(ctx_any));
                    if (start >= end) return;
                    if (t.stop.load(.acquire)) return;

                    var coords: [MAX_RANK]usize = .{0} ** MAX_RANK;
                    var idx: usize = start;
                    while (idx < end) : (idx += 1) {
                        if (t.stop.load(.acquire)) return;
                        const slice_idx: usize = idx / t.q_tiles_per_slice;
                        const ti_q: usize = idx - (slice_idx * t.q_tiles_per_slice);
                        if (t.row_dim_count > 0) {
                            decodeLeadingCoords(slice_idx, t.row_dim_count, t.row_dims[0..t.row_dim_count], t.row_strides[0..t.row_dim_count], coords[0..t.out_meta.tile_counts.len]);
                        }
                        attentionTileAllV_Stream_F32_SliceND(
                            t.store,
                            t.s.out,
                            t.s.q,
                            t.s.k,
                            t.s.v,
                            t.out_meta,
                            t.q_meta,
                            t.k_meta,
                            t.v_meta,
                            t.s.scale,
                            t.s.causal,
                            t.kernels,
                            coords[0..t.row_dim_count],
                            ti_q,
                            t.head_m,
                            t.head_n,
                            t.k_tile_start,
                            t.k_tile_count,
                        ) catch |e| {
                            t.fail(e);
                            return;
                        };
                    }
                }
            };

            var task: Task = .{
                .store = store,
                .out_meta = out_meta,
                .q_meta = q_meta,
                .k_meta = k_meta,
                .v_meta = v_meta,
                .s = s,
                .kernels = kernels,
                .head_m = head_m,
                .head_n = head_n,
                .k_tile_start = k_tile_start,
                .k_tile_count = k_tile_count,
                .row_dim_count = row_dim_count,
                .row_dims = row_dims,
                .row_strides = row_strides,
                .q_tiles_per_slice = q_tiles_per_slice,
            };
            p.parallelForAny(@ptrCast(&task), total_work, 1, Task.runWork);
            if (task.err_any) |e| return @errorCast(e);
            return;
        }
    }

    // Sequential fallback.
    var coords: [MAX_RANK]usize = .{0} ** MAX_RANK;
    var idx: usize = 0;
    while (idx < total_work) : (idx += 1) {
        const slice_idx: usize = idx / q_tiles_per_slice;
        const ti_q: usize = idx - (slice_idx * q_tiles_per_slice);
        if (row_dim_count > 0) {
            decodeLeadingCoords(slice_idx, row_dim_count, row_dims[0..row_dim_count], row_strides[0..row_dim_count], coords[0..out_meta.tile_counts.len]);
        }
        try attentionTileAllV_Stream_F32_SliceND(
            store,
            s.out,
            s.q,
            s.k,
            s.v,
            out_meta,
            q_meta,
            k_meta,
            v_meta,
            s.scale,
            s.causal,
            kernels,
            coords[0..row_dim_count],
            ti_q,
            head_m,
            head_n,
            k_tile_start,
            k_tile_count,
        );
    }
}

pub fn execMultiHeadAttentionTiled(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    kernels: attention_registry.Kernels,
    s: executable.StepMultiHeadAttentionTiled,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    if (!(s.scale > 0.0) or !std.math.isFinite(s.scale)) return BackendError.InvalidArgument;
    if (s.heads == 0) return BackendError.InvalidArgument;

    const out_meta = try store.meta(s.out);
    const q_meta = try store.meta(s.q);
    const k_meta = try store.meta(s.k);
    const v_meta = try store.meta(s.v);

    if (out_meta.rank < 3) return BackendError.InvalidArgument;
    const rank: usize = @as(usize, out_meta.rank);
    const head_dim: usize = rank - 3;
    if (out_meta.shape[head_dim] != s.heads) return BackendError.InvalidArgument;
    if (q_meta.shape[head_dim] != s.heads or k_meta.shape[head_dim] != s.heads or v_meta.shape[head_dim] != s.heads) return BackendError.InvalidArgument;

    const attn_step: executable.StepAttentionTiled = .{ .out = s.out, .q = s.q, .k = s.k, .v = s.v, .scale = s.scale, .causal = s.causal };
    return execAttentionTiledND(pool, thread_count, kernels, attn_step, store);
}

fn attentionTileAllV_Stream_F32_SliceND(
    store: tensor_store.TensorStore,
    out_id: executable.TensorId,
    q_id: executable.TensorId,
    k_id: executable.TensorId,
    v_id: executable.TensorId,
    out_meta: tensor_store.TensorMeta,
    q_meta: tensor_store.TensorMeta,
    k_meta: tensor_store.TensorMeta,
    v_meta: tensor_store.TensorMeta,
    scale: f32,
    causal: bool,
    kernels: attention_registry.Kernels,
    lead_coords: []const usize,
    ti_q: usize,
    head_m: usize,
    head_n: usize,
    k_tile_start: usize,
    k_tile_count: usize,
) ExecuteProgramError!void {
    std.debug.assert(out_meta.dtype == .f32);

    const rank: usize = @as(usize, out_meta.rank);
    const lead_dims: usize = rank - 2;
    if (lead_coords.len != lead_dims) return BackendError.InvalidArgument;

    const tm: usize = out_meta.tile_shape[rank - 2];
    const tn: usize = k_meta.tile_shape[rank - 2];

    const q_row0: usize = ti_q * tm;
    const local_q_row0: usize = q_row0;
    const m_tile: usize = if (local_q_row0 >= head_m) 0 else @min(tm, head_m - local_q_row0);
    if (m_tile == 0) return;

    var coords: [MAX_RANK]usize = .{0} ** MAX_RANK;
    var d: usize = 0;
    while (d < lead_dims) : (d += 1) {
        coords[d] = lead_coords[d];
    }

    // Per-row softmax stats.
    var m_state: [256]f32 = undefined;
    var l_state: [256]f32 = undefined;
    @memset(m_state[0..m_tile], -std.math.inf(f32));
    @memset(l_state[0..m_tile], 0.0);

    const tc_k: usize = k_tile_count;
    const tc_dk: usize = q_meta.tile_counts[rank - 1];

    // Scratch for one key tile: scores (tm x tn) and packed K/Q panels.
    var scores: [256 * 128]f32 = undefined;
    const scores_len: usize = tm * tn;
    var kt_scratch: [128 * 128]f32 = undefined;
    var qt_scratch: [256 * 128]f32 = undefined;

    const ti_v_total: usize = out_meta.tile_counts[rank - 1];
    var out_tiles: [256]tensor_store.TileRefMut = undefined;
    var out_strides: [256]usize = undefined;
    var out_len: usize = 0;

    if (ti_v_total > out_tiles.len) return BackendError.InvalidArgument;
    var ti_v: usize = 0;
    while (ti_v < ti_v_total) : (ti_v += 1) {
        coords[rank - 2] = ti_q;
        coords[rank - 1] = ti_v;
        const out_index: usize = try tensor_store.encodeTileIndex(out_meta, coords[0..out_meta.tile_counts.len]);
        var out_t = try store.acquireTileMutLinear(out_id, out_index);
        const ov0 = out_t.bufferView();
        if (ov0.dtype != .f32) {
            store.releaseMut(out_t.token);
            return BackendError.InvalidArgument;
        }
        if ((@intFromPtr(ov0.bytes.ptr) & (@alignOf(f32) - 1)) != 0) {
            store.releaseMut(out_t.token);
            return BackendError.InvalidArgument;
        }
        const dv_tile: usize = try tileDim(out_meta.shape, out_meta.tile_shape, ti_v, rank - 1);
        const out_s0: []align(@alignOf(f32)) f32 = @alignCast(simd.bytesAsSliceMutUnaligned(f32, ov0.bytes));
        @memset(out_s0[0 .. m_tile * dv_tile], 0.0);

        out_tiles[out_len] = out_t;
        out_strides[out_len] = dv_tile;
        out_len += 1;
    }
    defer {
        var i: usize = 0;
        while (i < out_len) : (i += 1) store.releaseMut(out_tiles[i].token);
    }

    // One-pass streaming softmax.
    var ti_k: usize = 0;
    while (ti_k < tc_k) : (ti_k += 1) {
        const ti_k_global: usize = k_tile_start + ti_k;
        const key_row0: usize = ti_k_global * tn;
        if (key_row0 >= head_n) continue;
        const local_key_row0: usize = key_row0;
        if (causal) {
            const max_q_idx: usize = local_q_row0 + m_tile - 1;
            if (local_key_row0 > max_q_idx) continue;
        }

        @memset(scores[0..scores_len], 0.0);

        var ti_dk: usize = 0;
        while (ti_dk < tc_dk) : (ti_dk += 1) {
            coords[rank - 2] = ti_q;
            coords[rank - 1] = ti_dk;
            const q_index: usize = try tensor_store.encodeTileIndex(q_meta, coords[0..q_meta.tile_counts.len]);
            const qt = try store.acquireTileConstLinear(q_id, q_index);
            defer store.releaseConst(qt.token);

            coords[rank - 2] = ti_k_global;
            coords[rank - 1] = ti_dk;
            const k_index: usize = try tensor_store.encodeTileIndex(k_meta, coords[0..k_meta.tile_counts.len]);
            const kt = try store.acquireTileConstLinear(k_id, k_index);
            defer store.releaseConst(kt.token);

            const qv = qt.bufferView();
            const kv = kt.bufferView();
            if (qv.dtype != .f32 or kv.dtype != .f32) return BackendError.InvalidArgument;

            const m_q: usize = m_tile;
            const k_q: usize = try tileDim(q_meta.shape, q_meta.tile_shape, ti_dk, rank - 1);
            const n_k: usize = try tileDim(k_meta.shape, k_meta.tile_shape, ti_k_global, rank - 2);
            const k_k: usize = k_q;
            if (n_k > tn) return BackendError.InvalidArgument;

            const qs: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, qv.bytes);
            const ks: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, kv.bytes);

            kernels.pack_k_block(n_k, k_q, ks, k_k, &kt_scratch);
            kernels.calc_scores_f32(m_q, n_k, k_q, qs, k_q, ks, k_k, &kt_scratch, &qt_scratch, scores[0..scores_len], tn);
        }

        attn_kernels.applyScaleMaskF32(scores[0..scores_len], m_tile, tn, head_n, local_key_row0, local_q_row0, scale, causal);

        var row_max: [256]f32 = undefined;
        @memset(row_max[0..m_tile], -std.math.inf(f32));
        attn_kernels.rowMaxF32(scores[0..scores_len], m_tile, tn, row_max[0..m_tile]);

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
                const ov = out_tiles[vi];
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

        attn_kernels.expNormalizeScoresF32(scores[0..scores_len], m_tile, tn, m_new[0..m_tile], l_state[0..m_tile]);
        for (0..m_tile) |r| m_state[r] = m_new[r];

        // Accumulate: out += P @ V for each V tile.
        ti_v = 0;
        while (ti_v < ti_v_total) : (ti_v += 1) {
            coords[rank - 2] = ti_k_global;
            coords[rank - 1] = ti_v;
            const v_index: usize = try tensor_store.encodeTileIndex(v_meta, coords[0..v_meta.tile_counts.len]);
            const vt = try store.acquireTileConstLinear(v_id, v_index);
            defer store.releaseConst(vt.token);
            const vv = vt.bufferView();
            if (vv.dtype != .f32) return BackendError.InvalidArgument;

            const n_v: usize = try tileDim(v_meta.shape, v_meta.tile_shape, ti_k_global, rank - 2);
            const dv_v: usize = try tileDim(v_meta.shape, v_meta.tile_shape, ti_v, rank - 1);
            if (n_v > tn) return BackendError.InvalidArgument;

            const vs: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, vv.bytes);

            const ov = out_tiles[ti_v];
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
        const ov = out_tiles[vi2];
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
