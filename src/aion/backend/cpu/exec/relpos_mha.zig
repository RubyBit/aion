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
// next cache.
//
// Chunked-limited attention is STRUCTURAL (`chunk_size`/`chunk_left`): every query
// row sees one contiguous key window, so the work is O(T_q * window) instead of
// scoring all T_kv keys and masking almost all of them away. The additive `mask`
// operand still applies inside the window, for what an interval cannot express
// (streaming padding, or any bespoke pattern).
//
// Hot path: rows are processed in panels that share a key window. The `ac`/`bd` GEMMs
// and the V accumulation use the shared, SIMD-tuned attention kernels selected per
// lane width (see `attention_registry`). A panel also bounds the `pos_emb` band it
// needs (consecutive rows shift by one), so `bd` costs `window + panel - 1` columns
// per row instead of all `P = 2*T_kv - 1`. Slices are distributed across the thread
// pool; lowering forces a single tile over the trailing [T, D] dims so each
// (batch, head) slice is one contiguous tile.

const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");

const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");

const simd = @import("../kernels/simd.zig");
const attention_registry = @import("../registry/attention_registry.zig");

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
    /// Chunked-limited window (`chunk_size == 0` => `t_kv`): the widest key range a
    /// single query row can see, which is what the score/pos_emb scratch is sized on.
    w_max: usize,
    /// Rows per panel, capped by `PANEL` and by the rows that share a key window.
    panel: usize,
};

const Scratch = struct {
    qu: []f32, // [T_q * D]
    qv: []f32, // [T_q * D]
    ac: []f32, // [panel, w_max] scores, softmaxed in place
    bd: []f32, // [panel, w_max + panel - 1] pos_emb band scores
    ktp: []f32, // packed-K scratch, sized for the larger (bd) GEMM
    qtp: []f32, // packed-Q scratch
    row_max: []f32, // [panel]
    l: []f32, // [panel]

    fn bdWidth(dims: Dims) usize {
        return dims.w_max + dims.panel - 1;
    }

    fn perWorkerLen(dims: Dims, mr: usize, nr: usize) usize {
        const qu = dims.t_q * dims.d;
        const qv = dims.t_q * dims.d;
        const ac = dims.panel * dims.w_max;
        const bd = dims.panel * bdWidth(dims);
        const ktp = alignUp(bdWidth(dims), nr) * dims.d;
        const qtp = alignUp(dims.panel, mr) * dims.d;
        return qu + qv + ac + bd + ktp + qtp + dims.panel + dims.panel;
    }

    fn carve(buf: []f32, dims: Dims, mr: usize, nr: usize) Scratch {
        var off: usize = 0;
        const qu = take(buf, &off, dims.t_q * dims.d);
        const qv = take(buf, &off, dims.t_q * dims.d);
        const ac = take(buf, &off, dims.panel * dims.w_max);
        const bd = take(buf, &off, dims.panel * bdWidth(dims));
        const ktp = take(buf, &off, alignUp(bdWidth(dims), nr) * dims.d);
        const qtp = take(buf, &off, alignUp(dims.panel, mr) * dims.d);
        const row_max = take(buf, &off, dims.panel);
        const l = take(buf, &off, dims.panel);
        return .{ .qu = qu, .qv = qv, .ac = ac, .bd = bd, .ktp = ktp, .qtp = qtp, .row_max = row_max, .l = l };
    }

    fn take(buf: []f32, off: *usize, n: usize) []f32 {
        const s = buf[off.* .. off.* + n];
        off.* += n;
        return s;
    }
};

/// Keys visible to query row `i` under chunked-limited attention. The row's absolute
/// index in the K/V window is `i + (t_kv - t_q)` (a left-context cache occupies the
/// front), it attends to its own chunk plus `chunk_left` frames before that chunk's
/// start, and the result is one contiguous interval — which is the whole reason this
/// can replace an additive [T_q, T_kv] mask.
fn keyWindow(dims: Dims, chunk_size: usize, chunk_left: usize, i: usize) struct { lo: usize, hi: usize } {
    if (chunk_size == 0) return .{ .lo = 0, .hi = dims.t_kv };
    const a: usize = i + (dims.t_kv - dims.t_q);
    const cs: usize = (a / chunk_size) * chunk_size;
    return .{
        .lo = cs - @min(cs, chunk_left),
        .hi = @min(cs + chunk_size, dims.t_kv),
    };
}

inline fn alignUp(x: usize, a: usize) usize {
    return ((x + a - 1) / a) * a;
}

/// `scores[rows, n_k] += qs @ ks^T`, splitting the row range between the two score
/// kernels: whole `mr` blocks go through the packed microkernel, the remainder through
/// the narrow one. `calc_scores_f32` degrades to a SCALAR loop for a partial row block
/// (`r0 + MR > m_q`), which a chunked window hits on every panel — a chunk of 14 rows
/// is one 8-row block plus 6 rows that would otherwise be scored one element at a time.
fn addScores(
    kernels: attention_registry.Kernels,
    rows: usize,
    n_k: usize,
    d: usize,
    qs: []align(1) const f32,
    q_stride: usize,
    ks: []align(1) const f32,
    k_stride: usize,
    kt: []f32,
    qt: []f32,
    scores: []f32,
    score_stride: usize,
) void {
    const mr: usize = kernels.tuning.mr;
    const aligned: usize = (rows / mr) * mr;
    if (aligned > 0) {
        kernels.pack_k_block(n_k, d, ks, k_stride, kt);
        kernels.calc_scores_f32(aligned, n_k, d, qs, q_stride, ks, k_stride, kt, qt, scores, score_stride);
    }
    if (aligned < rows) {
        kernels.calc_scores_narrow_f32(
            rows - aligned,
            n_k,
            d,
            qs[aligned * q_stride ..],
            q_stride,
            ks,
            k_stride,
            scores[aligned * score_stride ..],
            score_stride,
        );
    }
}

fn tileIndex(meta: tensor_store.TensorMeta, coords: []const usize) ExecuteProgramError!usize {
    return tensor_store.encodeTileIndex(meta, coords[0..meta.tile_counts.len]);
}

pub fn execRelPosMHATiled(
    allocator: std.mem.Allocator,
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    kernels: attention_registry.Kernels,
    s: executable.StepRelPosMHATiled,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    if (!(s.scale > 0.0) or !std.math.isFinite(s.scale)) return BackendError.InvalidArgument;

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

    if (H == 0) return BackendError.InvalidArgument;
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

    if (s.chunk_size == 0 and s.chunk_left != 0) return BackendError.InvalidArgument;
    // Widest window any row can see, and the most rows that can share one window.
    const w_max: usize = if (s.chunk_size == 0)
        T_kv
    else
        @min(T_kv, s.chunk_size + s.chunk_left);
    // Rows per panel: the tuned cap (see `attention_registry.Tuning.relpos_panel`),
    // then whatever can actually share one key window.
    const panel: usize = @min(kernels.tuning.relpos_panel, @min(T_q, if (s.chunk_size == 0) T_q else s.chunk_size));

    const dims: Dims = .{
        .b = B,
        .h = H,
        .t_q = T_q,
        .t_kv = T_kv,
        .p = P,
        .d = D,
        .w_max = w_max,
        .panel = @max(@as(usize, 1), panel),
    };
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
                kernels: attention_registry.Kernels,
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
    kernels: attention_registry.Kernels,
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
    var coords4: [MAX_RANK]usize = @splat(0);
    coords4[0] = b;
    coords4[2] = h;

    // --- acquire input tiles (read) ---
    const q_t = try store.acquireTileConstLinear(s.q, try tileIndex((try store.meta(s.q)), &coords4));
    defer store.releaseConst(q_t.token);
    const k_t = try store.acquireTileConstLinear(s.k, try tileIndex((try store.meta(s.k)), &coords4));
    defer store.releaseConst(k_t.token);
    const v_t = try store.acquireTileConstLinear(s.v, try tileIndex((try store.meta(s.v)), &coords4));
    defer store.releaseConst(v_t.token);

    var pe_coords: [MAX_RANK]usize = @splat(0);
    pe_coords[0] = h;
    const pe_t = try store.acquireTileConstLinear(s.pos_emb, try tileIndex((try store.meta(s.pos_emb)), &pe_coords));
    defer store.releaseConst(pe_t.token);

    var bias_coords: [MAX_RANK]usize = @splat(0);
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

    const scale = s.scale;
    var mask_buf: ?[]align(1) const f32 = null;
    var mask_tok: ?usize = null;
    if (s.mask) |mask_id| {
        var m_coords: [MAX_RANK]usize = @splat(0);
        const m_t = try store.acquireTileConstLinear(mask_id, try tileIndex((try store.meta(mask_id)), &m_coords));
        mask_tok = m_t.token;
        mask_buf = simd.bytesAsSliceConstUnaligned(f32, m_t.bufferView().bytes);
    }
    defer if (mask_tok) |tok| store.releaseConst(tok);

    const out_t = try store.acquireTileMutLinear(s.out, try tileIndex((try store.meta(s.out)), &coords4));
    defer store.releaseMut(out_t.token);
    const ov = out_t.bufferView();
    if (ov.dtype != .f32) return BackendError.InvalidArgument;
    const obuf: []align(@alignOf(f32)) f32 = @alignCast(simd.bytesAsSliceMutUnaligned(f32, ov.bytes));
    if (obuf.len < T_q * D) return BackendError.InvalidArgument;

    // --- row panels: rows sharing a key window, at most `panel` at a time ---
    var r0: usize = 0;
    while (r0 < T_q) {
        const win = keyWindow(dims, s.chunk_size, s.chunk_left, r0);
        // Extend the panel while the window and the cap allow it. With chunked
        // attention that stops at the chunk boundary; unchunked, every row shares
        // [0, T_kv) so it runs to the cap.
        var r1: usize = r0 + 1;
        while (r1 < T_q and (r1 - r0) < dims.panel) {
            const w = keyWindow(dims, s.chunk_size, s.chunk_left, r1);
            if (w.lo != win.lo or w.hi != win.hi) break;
            r1 += 1;
        }
        const rows: usize = r1 - r0;
        const n_k: usize = win.hi - win.lo;
        if (n_k == 0 or n_k > dims.w_max) return BackendError.InvalidArgument;

        // --- ac = (q+u) @ K[win]^T -> [rows, n_k] ---
        const ac = scratch.ac[0 .. rows * n_k];
        @memset(ac, 0.0);
        const k_panel = kbuf[win.lo * D ..];
        addScores(kernels, rows, n_k, D, scratch.qu[r0 * D ..], D, k_panel, D, scratch.ktp, scratch.qtp, ac, n_k);

        // --- bd = (q+v) @ pos_emb[band]^T -> [rows, n_k + rows - 1] ---
        // Row r reads pos_emb[base_r + win.lo ..][0..n_k] with base_r = (T_q-1) - r,
        // so the panel's rows span one contiguous band starting at the LAST row's base.
        const band: usize = n_k + rows - 1;
        const pe_lo: usize = ((T_q - 1) - (r1 - 1)) + win.lo;
        if (pe_lo + band > P) return BackendError.InvalidArgument;
        const bd = scratch.bd[0 .. rows * band];
        @memset(bd, 0.0);
        const pe_panel = pebuf[pe_lo * D ..];
        addScores(kernels, rows, band, D, scratch.qv[r0 * D ..], D, pe_panel, D, scratch.ktp, scratch.qtp, bd, band);

        // --- combine: scores[i,j] = (ac + bd[i, shift_i + j]) * scale + mask ---
        var ri: usize = 0;
        while (ri < rows) : (ri += 1) {
            const shift: usize = (r1 - 1) - (r0 + ri); // base_{r0+ri} - base_{r1-1}
            const ac_row = ac[ri * n_k ..][0..n_k];
            const bd_row = bd[ri * band + shift ..][0..n_k];
            if (mask_buf) |mb| {
                const mrow = mb[(r0 + ri) * T_kv + win.lo ..][0..n_k];
                var j: usize = 0;
                while (j < n_k) : (j += 1) ac_row[j] = (ac_row[j] + bd_row[j]) * scale + mrow[j];
            } else {
                var j: usize = 0;
                while (j < n_k) : (j += 1) ac_row[j] = (ac_row[j] + bd_row[j]) * scale;
            }
        }

        // --- softmax over the window ---
        @memset(scratch.row_max[0..rows], -std.math.inf(f32));
        kernels.row_max_f32(ac, rows, n_k, scratch.row_max[0..rows]);
        @memset(scratch.l[0..rows], 0.0);
        kernels.exp_normalize_scores_f32(ac, rows, n_k, scratch.row_max[0..rows], scratch.l[0..rows]);
        ri = 0;
        while (ri < rows) : (ri += 1) {
            const denom = scratch.l[ri];
            if (!(denom > 0.0) or !std.math.isFinite(denom)) return BackendError.InvalidArgument;
            const inv = 1.0 / denom;
            const row = ac[ri * n_k ..][0..n_k];
            var j: usize = 0;
            while (j < n_k) : (j += 1) row[j] *= inv;
        }

        // --- out[panel] = scores @ V[win] ---
        @memset(obuf[r0 * D .. r1 * D], 0.0);
        kernels.accumulate_values_f32(rows, n_k, D, ac, n_k, vbuf[win.lo * D ..], D, obuf[r0 * D ..], D);

        r0 = r1;
    }
}
