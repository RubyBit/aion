// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Relative-positional multi-head self-attention (Transformer-XL / Conformer style).
//
// Computes, per (batch, head) slice:
//   ac[i,j] = (q[i] + pos_bias_u) · k[j]                         (content score)
//   bd[i,p] = (q[i] + pos_bias_v) · pos_emb[p]                   (position score, P = 2*T_kv-1)
//   scores[i,j] = (ac[i,j] + bd[i, (T_q-1) - i + j]) * scale + mask[i,j]
//   out[i] = softmax_j(scores[i,:]) @ v
//
// The `(T_q-1) - i + j` index is the rel-shift written as direct index arithmetic:
// for a fixed query row i it is a *contiguous* band `bd[i, base_i .. base_i+T_kv]`
// with `base_i = (T_q-1) - i`, so no [T_q, P] matrix is materialized and the combine
// is a per-row vector add. This collapses to the usual self-attention rel-shift when
// T_q == T_kv (then base_i = (T-1) - i, i.e. p = (T-1) + (j-i)) and also handles the
// streaming case where a left-context cache is prepended to K/V (T_q < T_kv).
//
// Layout (head-third, matching how `reshape([B,T,H*D])` produces q/k/v in-graph,
// so no transpose op is needed before/after attention):
//   q,k,v,out : [B, T*, H, D]  (q rows T_q; k/v rows T_kv)   f32
//   pos_emb   : [H, P, D]      (P = 2*T_kv - 1)              f32
//   pos_bias_u/_v : [H, D]                                   f32
//   mask (optional) : [T_q, T_kv] additive                  f32
//
// Tiling: each (batch, head) slice is a single tile of shape [1, T, 1, D], which
// the tiled storage packs contiguously as [T, D] (row stride D) — so the kernel
// reads each slice as a contiguous [T*D] panel exactly as in the head-second case.
//
// The op does not own the streaming ring buffer: the converter concatenates the
// per-layer left-context cache onto K/V in-graph and slices the tail back out as the
// next cache. The chunked-limited attention window is supplied via the additive mask.
//
// Hot path: the `ac`/`bd` GEMMs and the V accumulation use the shared, SIMD-tuned
// attention kernels selected per lane width (see `relpos_mha_registry`). Slices are
// distributed across the thread pool. Lowering forces a single tile over the trailing
// [T, D] dims so each (batch, head) slice is one contiguous tile.

const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");

const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");

const simd = @import("../kernels/simd.zig");
const attn_kernels = @import("../kernels/attention.zig");
const relpos_registry = @import("../registry/relpos_mha_registry.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

const MAX_RANK: usize = 8;

const Dims = struct {
    b: usize,
    h: usize,
    t_q: usize,
    t_kv: usize,
    p: usize,
    d: usize,
};

const Scratch = struct {
    qu: []f32, // [T_q * D]
    qv: []f32, // [T_q * D]
    ac: []f32, // [T_q * T_kv]  (also holds the softmaxed scores)
    bd: []f32, // [T_q * P]
    ktp: []f32, // packed-K scratch, sized for the larger (bd) GEMM
    qtp: []f32, // packed-Q scratch
    row_max: []f32, // [T_q]
    l: []f32, // [T_q]

    fn perWorkerLen(dims: Dims, mr: usize, nr: usize) usize {
        const qu = dims.t_q * dims.d;
        const qv = dims.t_q * dims.d;
        const ac = dims.t_q * dims.t_kv;
        const bd = dims.t_q * dims.p;
        const ktp = alignUp(dims.p, nr) * dims.d;
        const qtp = alignUp(dims.t_q, mr) * dims.d;
        return qu + qv + ac + bd + ktp + qtp + dims.t_q + dims.t_q;
    }

    fn carve(buf: []f32, dims: Dims, mr: usize, nr: usize) Scratch {
        var off: usize = 0;
        const qu = take(buf, &off, dims.t_q * dims.d);
        const qv = take(buf, &off, dims.t_q * dims.d);
        const ac = take(buf, &off, dims.t_q * dims.t_kv);
        const bd = take(buf, &off, dims.t_q * dims.p);
        const ktp = take(buf, &off, alignUp(dims.p, nr) * dims.d);
        const qtp = take(buf, &off, alignUp(dims.t_q, mr) * dims.d);
        const row_max = take(buf, &off, dims.t_q);
        const l = take(buf, &off, dims.t_q);
        return .{ .qu = qu, .qv = qv, .ac = ac, .bd = bd, .ktp = ktp, .qtp = qtp, .row_max = row_max, .l = l };
    }

    fn take(buf: []f32, off: *usize, n: usize) []f32 {
        const s = buf[off.* .. off.* + n];
        off.* += n;
        return s;
    }
};

inline fn alignUp(x: usize, a: usize) usize {
    return ((x + a - 1) / a) * a;
}

fn tileIndex(meta: tensor_store.TensorMeta, coords: []const usize) ExecuteProgramError!usize {
    return tensor_store.encodeTileIndex(meta, coords[0..meta.tile_counts.len]);
}

pub fn execRelPosMHATiled(
    allocator: std.mem.Allocator,
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    kernels: relpos_registry.Kernels,
    s: executable.StepRelPosMHATiled,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    if (!(s.scale > 0.0) or !std.math.isFinite(s.scale)) return BackendError.InvalidArgument;
    if (s.heads == 0) return BackendError.InvalidArgument;

    const out_meta = try store.meta(s.out);
    const q_meta = try store.meta(s.q);
    const k_meta = try store.meta(s.k);
    const v_meta = try store.meta(s.v);
    const pe_meta = try store.meta(s.pos_emb);
    const bu_meta = try store.meta(s.pos_bias_u);
    const bv_meta = try store.meta(s.pos_bias_v);

    // Ranks: q/k/v/out are [B, T, H, D]; pos_emb [H, P, D]; biases [H, D].
    if (out_meta.rank != 4 or q_meta.rank != 4 or k_meta.rank != 4 or v_meta.rank != 4) return BackendError.InvalidArgument;
    if (pe_meta.rank != 3 or bu_meta.rank != 2 or bv_meta.rank != 2) return BackendError.InvalidArgument;

    if (out_meta.dtype != .f32 or q_meta.dtype != .f32 or k_meta.dtype != .f32 or v_meta.dtype != .f32) return BackendError.InvalidArgument;
    if (pe_meta.dtype != .f32 or bu_meta.dtype != .f32 or bv_meta.dtype != .f32) return BackendError.InvalidArgument;

    const B: usize = out_meta.shape[0];
    const T_q: usize = out_meta.shape[1];
    const H: usize = out_meta.shape[2];
    const D: usize = out_meta.shape[3];
    const T_kv: usize = k_meta.shape[1];
    const P: usize = pe_meta.shape[1];

    if (H != s.heads) return BackendError.InvalidArgument;
    if (q_meta.shape[0] != B or q_meta.shape[1] != T_q or q_meta.shape[2] != H or q_meta.shape[3] != D) return BackendError.InvalidArgument;
    if (k_meta.shape[0] != B or k_meta.shape[2] != H or k_meta.shape[3] != D) return BackendError.InvalidArgument;
    if (v_meta.shape[0] != B or v_meta.shape[1] != T_kv or v_meta.shape[2] != H or v_meta.shape[3] != D) return BackendError.InvalidArgument;
    if (pe_meta.shape[0] != H or pe_meta.shape[2] != D) return BackendError.InvalidArgument;
    if (bu_meta.shape[0] != H or bu_meta.shape[1] != D) return BackendError.InvalidArgument;
    if (bv_meta.shape[0] != H or bv_meta.shape[1] != D) return BackendError.InvalidArgument;
    if (P != 2 * T_kv - 1) return BackendError.InvalidArgument;
    if (T_q > T_kv) return BackendError.InvalidArgument;

    // Tiling contract: dims [T, D] (1 and 3) form a single tile per (batch, head);
    // dims [B, H] (0 and 2) are size-1 tiles. pos_emb [H,P,D] tiles [1,P,D]; biases single-tile.
    if (out_meta.tile_counts[0] != B or out_meta.tile_counts[2] != H) return BackendError.InvalidArgument;
    if (out_meta.tile_counts[1] != 1 or out_meta.tile_counts[3] != 1) return BackendError.InvalidArgument;
    if (q_meta.tile_counts[1] != 1 or q_meta.tile_counts[3] != 1) return BackendError.InvalidArgument;
    if (k_meta.tile_counts[1] != 1 or k_meta.tile_counts[3] != 1) return BackendError.InvalidArgument;
    if (v_meta.tile_counts[1] != 1 or v_meta.tile_counts[3] != 1) return BackendError.InvalidArgument;
    if (pe_meta.tile_counts[1] != 1 or pe_meta.tile_counts[2] != 1) return BackendError.InvalidArgument;

    if (s.mask) |mask_id| {
        const m_meta = try store.meta(mask_id);
        if (m_meta.rank != 2 or m_meta.dtype != .f32) return BackendError.InvalidArgument;
        if (m_meta.shape[0] != T_q or m_meta.shape[1] != T_kv) return BackendError.InvalidArgument;
        if (m_meta.tile_counts[0] != 1 or m_meta.tile_counts[1] != 1) return BackendError.InvalidArgument;
    }

    const dims: Dims = .{ .b = B, .h = H, .t_q = T_q, .t_kv = T_kv, .p = P, .d = D };
    const mr: usize = kernels.tuning.mr;
    const nr: usize = kernels.tuning.nr;
    const per_worker: usize = Scratch.perWorkerLen(dims, mr, nr);

    const total_slices: usize = B * H;
    if (total_slices == 0) return;

    const want_parallel: bool = pool != null and thread_count > 1 and total_slices >= 2;
    const worker_count: usize = if (want_parallel) thread_count else 1;

    const scratch_buf = allocator.alloc(f32, per_worker * worker_count) catch return BackendError.ExecutionFailed;
    defer allocator.free(scratch_buf);

    if (want_parallel) {
        if (pool) |p| {
            const Task = struct {
                store: tensor_store.TensorStore,
                s: executable.StepRelPosMHATiled,
                kernels: relpos_registry.Kernels,
                dims: Dims,
                mr: usize,
                nr: usize,
                per_worker: usize,
                scratch_buf: []f32,

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
                    const t: *@This() = @ptrCast(@alignCast(ctx_any));
                    if (start >= end) return;
                    if (t.stop.load(.acquire)) return;
                    const base: usize = (tid % (t.scratch_buf.len / t.per_worker)) * t.per_worker;
                    const scratch = Scratch.carve(t.scratch_buf[base .. base + t.per_worker], t.dims, t.mr, t.nr);
                    var idx: usize = start;
                    while (idx < end) : (idx += 1) {
                        if (t.stop.load(.acquire)) return;
                        const bb: usize = idx / t.dims.h;
                        const hh: usize = idx - bb * t.dims.h;
                        computeSlice(t.store, t.s, t.kernels, t.dims, bb, hh, scratch) catch |e| {
                            t.fail(e);
                            return;
                        };
                    }
                }
            };

            var task: Task = .{
                .store = store,
                .s = s,
                .kernels = kernels,
                .dims = dims,
                .mr = mr,
                .nr = nr,
                .per_worker = per_worker,
                .scratch_buf = scratch_buf,
            };
            p.parallelForAny(@ptrCast(&task), total_slices, 1, Task.runWork);
            if (task.err_any) |e| return @errorCast(e);
            return;
        }
    }

    const scratch = Scratch.carve(scratch_buf[0..per_worker], dims, mr, nr);
    var idx: usize = 0;
    while (idx < total_slices) : (idx += 1) {
        const bb: usize = idx / H;
        const hh: usize = idx - bb * H;
        try computeSlice(store, s, kernels, dims, bb, hh, scratch);
    }
}

fn computeSlice(
    store: tensor_store.TensorStore,
    s: executable.StepRelPosMHATiled,
    kernels: relpos_registry.Kernels,
    dims: Dims,
    b: usize,
    h: usize,
    scratch: Scratch,
) ExecuteProgramError!void {
    const T_q = dims.t_q;
    const T_kv = dims.t_kv;
    const P = dims.p;
    const D = dims.d;

    // Layout [B, T, H, D]: dims 1 (T) and 3 (D) are the single full tile; the slice
    // is selected by the size-1 tiles on dim 0 (B) and dim 2 (H).
    var coords4: [MAX_RANK]usize = .{0} ** MAX_RANK;
    coords4[0] = b;
    coords4[2] = h;

    // --- acquire input tiles (read) ---
    const q_t = try store.acquireTileConstLinear(s.q, try tileIndex((try store.meta(s.q)), &coords4));
    defer store.releaseConst(q_t.token);
    const k_t = try store.acquireTileConstLinear(s.k, try tileIndex((try store.meta(s.k)), &coords4));
    defer store.releaseConst(k_t.token);
    const v_t = try store.acquireTileConstLinear(s.v, try tileIndex((try store.meta(s.v)), &coords4));
    defer store.releaseConst(v_t.token);

    var pe_coords: [MAX_RANK]usize = .{0} ** MAX_RANK;
    pe_coords[0] = h;
    const pe_t = try store.acquireTileConstLinear(s.pos_emb, try tileIndex((try store.meta(s.pos_emb)), &pe_coords));
    defer store.releaseConst(pe_t.token);

    var bias_coords: [MAX_RANK]usize = .{0} ** MAX_RANK;
    const bu_t = try store.acquireTileConstLinear(s.pos_bias_u, try tileIndex((try store.meta(s.pos_bias_u)), &bias_coords));
    defer store.releaseConst(bu_t.token);
    const bv_t = try store.acquireTileConstLinear(s.pos_bias_v, try tileIndex((try store.meta(s.pos_bias_v)), &bias_coords));
    defer store.releaseConst(bv_t.token);

    const qbuf: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, q_t.bufferView().bytes);
    const kbuf: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, k_t.bufferView().bytes);
    const vbuf: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, v_t.bufferView().bytes);
    const pebuf: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, pe_t.bufferView().bytes);
    const ubuf: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, bu_t.bufferView().bytes);
    const vbias: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, bv_t.bufferView().bytes);

    if (qbuf.len < T_q * D or kbuf.len < T_kv * D or vbuf.len < T_kv * D) return BackendError.InvalidArgument;
    if (pebuf.len < P * D) return BackendError.InvalidArgument;

    const u_row: []align(1) const f32 = ubuf[h * D .. h * D + D];
    const v_row: []align(1) const f32 = vbias[h * D .. h * D + D];

    // --- qu = q + u ; qv = q + v_bias ---
    var i: usize = 0;
    while (i < T_q) : (i += 1) {
        const qr = qbuf[i * D .. i * D + D];
        const qu_r = scratch.qu[i * D .. i * D + D];
        const qv_r = scratch.qv[i * D .. i * D + D];
        var kk: usize = 0;
        while (kk < D) : (kk += 1) {
            const qval = qr[kk];
            qu_r[kk] = qval + u_row[kk];
            qv_r[kk] = qval + v_row[kk];
        }
    }

    // --- ac = (q+u) @ k^T  -> [T_q, T_kv] ---
    @memset(scratch.ac[0 .. T_q * T_kv], 0.0);
    kernels.pack_k_block(T_kv, D, kbuf, D, scratch.ktp);
    kernels.calc_scores_f32(T_q, T_kv, D, scratch.qu, D, kbuf, D, scratch.ktp, scratch.qtp, scratch.ac, T_kv);

    // --- bd = (q+v) @ pos_emb^T -> [T_q, P] ---
    @memset(scratch.bd[0 .. T_q * P], 0.0);
    kernels.pack_k_block(P, D, pebuf, D, scratch.ktp);
    kernels.calc_scores_f32(T_q, P, D, scratch.qv, D, pebuf, D, scratch.ktp, scratch.qtp, scratch.bd, P);

    // --- combine: scores[i,j] = (ac + bd[i, base_i + j]) * scale + mask[i,j] ---
    const scale = s.scale;
    var mask_buf: ?[]align(1) const f32 = null;
    var mask_tok: ?usize = null;
    if (s.mask) |mask_id| {
        var m_coords: [MAX_RANK]usize = .{0} ** MAX_RANK;
        const m_t = try store.acquireTileConstLinear(mask_id, try tileIndex((try store.meta(mask_id)), &m_coords));
        mask_tok = m_t.token;
        mask_buf = simd.bytesAsSliceConstUnaligned(f32, m_t.bufferView().bytes);
    }
    defer if (mask_tok) |tok| store.releaseConst(tok);

    i = 0;
    while (i < T_q) : (i += 1) {
        const base: usize = (T_q - 1) - i;
        const ac_row = scratch.ac[i * T_kv .. i * T_kv + T_kv];
        const bd_row = scratch.bd[i * P + base .. i * P + base + T_kv];
        if (mask_buf) |mb| {
            const mrow = mb[i * T_kv .. i * T_kv + T_kv];
            var j: usize = 0;
            while (j < T_kv) : (j += 1) ac_row[j] = (ac_row[j] + bd_row[j]) * scale + mrow[j];
        } else {
            var j: usize = 0;
            while (j < T_kv) : (j += 1) ac_row[j] = (ac_row[j] + bd_row[j]) * scale;
        }
    }

    // --- softmax over keys (j) ---
    @memset(scratch.row_max[0..T_q], -std.math.inf(f32));
    attn_kernels.rowMaxF32(scratch.ac[0 .. T_q * T_kv], T_q, T_kv, scratch.row_max[0..T_q]);
    @memset(scratch.l[0..T_q], 0.0);
    attn_kernels.expNormalizeScoresF32(scratch.ac[0 .. T_q * T_kv], T_q, T_kv, scratch.row_max[0..T_q], scratch.l[0..T_q]);
    i = 0;
    while (i < T_q) : (i += 1) {
        const denom = scratch.l[i];
        if (!(denom > 0.0) or !std.math.isFinite(denom)) return BackendError.InvalidArgument;
        const inv = 1.0 / denom;
        const row = scratch.ac[i * T_kv .. i * T_kv + T_kv];
        var j: usize = 0;
        while (j < T_kv) : (j += 1) row[j] *= inv;
    }

    // --- out = scores @ v ---
    const out_t = try store.acquireTileMutLinear(s.out, try tileIndex((try store.meta(s.out)), &coords4));
    defer store.releaseMut(out_t.token);
    const ov = out_t.bufferView();
    if (ov.dtype != .f32) return BackendError.InvalidArgument;
    const obuf: []align(@alignOf(f32)) f32 = @alignCast(simd.bytesAsSliceMutUnaligned(f32, ov.bytes));
    if (obuf.len < T_q * D) return BackendError.InvalidArgument;
    @memset(obuf[0 .. T_q * D], 0.0);
    kernels.accumulate_values_f32(T_q, T_kv, D, scratch.ac[0 .. T_q * T_kv], T_kv, vbuf, D, obuf, D);
}
