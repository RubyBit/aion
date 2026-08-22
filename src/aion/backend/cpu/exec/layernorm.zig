// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");

const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");

const exec_utils = @import("utils.zig");
const layernorm_kernels = @import("../kernels/layernorm.zig");
const elemwise = @import("../kernels/elemwise.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

const Mode = layernorm_kernels.Mode;

const MAX_RANK: usize = 8;
const ROW_CHUNK_MAX: usize = 256;

fn productUsize(vals: []const usize) ExecuteProgramError!usize {
    if (vals.len == 0) return BackendError.InvalidArgument;
    var acc: usize = 1;
    for (vals) |v| {
        acc = std.math.mul(usize, acc, v) catch return BackendError.InvalidArgument;
    }
    return acc;
}

fn tileDim(shape: []const usize, tile_shape: []const usize, tile_index: usize, dim: usize) ExecuteProgramError!usize {
    if (dim >= shape.len or dim >= tile_shape.len) return BackendError.InvalidArgument;
    const start: usize = tile_index * tile_shape[dim];
    if (start >= shape[dim]) return BackendError.InvalidArgument;
    return @min(tile_shape[dim], shape[dim] - start);
}

pub fn execLayerNormTiled(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepLayerNormTiled,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    return execNormTiled(pool, thread_count, .layernorm, s.out, s.x, s.gamma, s.beta, s.eps, store);
}

pub fn execRMSNormTiled(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepRMSNormTiled,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    try execNormTiled(pool, thread_count, .rmsnorm, s.out, s.x, s.gamma, s.beta, s.eps, store);
    if (s.residual) |res| try addResidualInPlace(s.out, res, store);
}

/// The residual half of `StepRMSNormTiled`, deliberately DECOMPOSED: the ordinary norm
/// has already run, and this adds the residual into `out` in place.
///
/// This arm exists for correctness, not speed, and a genuinely fused CPU kernel is not
/// worth writing — measured, not assumed. Profiling one CPU decode step of Gemma-4 E2B
/// (2026-08-19, `AION_PROFILE=summary`, 400.5 ms total):
///
///     MatMulTiled          369.3 ms   92.2%
///     MatMulNTTiled         27.1 ms    6.8%
///     ElemwiseBinaryTiled    0.763 ms  0.19%   <- ALL 182 of them
///     RMSNormTiled           0.495 ms  0.12%
///
/// So the entire elementwise category is under 0.2% of a CPU step, the 105 residual
/// adds are a subset of it, and fusing merges the add into the norm's apply pass rather
/// than removing its arithmetic — so the recoverable share is a fraction of 0.2%. CPU
/// decode is a q8 GEMV problem, full stop. (On GPU the same fusion is worth +1.1%,
/// because there the cost being removed is a dispatch launch, which has no CPU analogue.)
///
/// Keeping it decomposed has a second benefit: `program/fuse_steps.zig` fires only for GPU
/// targets, so a GPU-vs-CPU comparison pits the fused kernel against the unfused pair and
/// the test asserts "fusing changed nothing".
///
/// If this ever DOES need fusing, the work is to thread an optional residual through
/// `execNormTiledND`'s apply pass (and `normTi0`'s) and add a vectorized
/// `applyRMSNormAdd` beside `applyNorm` — not to reuse the loop below, which is serial
/// where `exec/elementwise.zig` is thread-pooled.
///
/// f32 addition commutes exactly, so the operand order the fusion pass picked cannot
/// change the result either.
fn addResidualInPlace(
    out_id: tensor_store.TensorId,
    residual: tensor_store.TensorId,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    const out_meta = try store.meta(out_id);
    var tile_total: usize = 1;
    var d: usize = 0;
    while (d < @as(usize, out_meta.rank)) : (d += 1) tile_total *= out_meta.tile_counts[d];

    var ti: usize = 0;
    while (ti < tile_total) : (ti += 1) {
        const out_tile = try store.acquireTileMutLinear(out_id, ti);
        defer store.releaseMut(out_tile.token);
        const a_tile = try store.acquireTileConstLinear(residual, ti);
        defer store.releaseConst(a_tile.token);

        const out_view = out_tile.bufferView();
        const a_view = a_tile.bufferView();
        if (out_view.dtype != .f32 or a_view.dtype != .f32) return BackendError.InvalidArgument;
        const n: usize = exec_utils.elemCountFromTileView(out_view);
        try elemwise.elemwiseBinaryF32(.add, out_view.bytes, out_view.bytes, a_view.bytes, n);
    }
}

fn execNormTiled(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    mode: Mode,
    out: tensor_store.TensorId,
    x: tensor_store.TensorId,
    gamma: tensor_store.TensorId,
    beta: tensor_store.TensorId,
    eps: f32,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    if (!(eps > 0.0) or !std.math.isFinite(eps)) return BackendError.InvalidArgument;
    const out_meta = try store.meta(out);
    const x_meta = try store.meta(x);
    const g_meta = try store.meta(gamma);
    const b_meta = try store.meta(beta);

    const rank: usize = @as(usize, out_meta.rank);
    if (rank == 0 or rank > MAX_RANK) return BackendError.InvalidArgument;
    if (out_meta.rank != x_meta.rank) return BackendError.InvalidArgument;
    if (!(out_meta.dtype == .f32 or out_meta.dtype == .f16)) return BackendError.InvalidArgument;
    if (out_meta.dtype != x_meta.dtype or out_meta.dtype != g_meta.dtype or out_meta.dtype != b_meta.dtype) return BackendError.InvalidArgument;

    const norm_rank: usize = @as(usize, g_meta.rank);
    if (norm_rank == 0 or norm_rank > rank) return BackendError.InvalidArgument;
    if (g_meta.rank != b_meta.rank) return BackendError.InvalidArgument;

    // Shapes must match trailing normalized dims.
    var d: usize = 0;
    while (d < rank) : (d += 1) {
        if (out_meta.shape[d] != x_meta.shape[d]) return BackendError.InvalidArgument;
    }
    d = 0;
    while (d < norm_rank) : (d += 1) {
        const od: usize = out_meta.shape[rank - norm_rank + d];
        if (g_meta.shape[d] != od or b_meta.shape[d] != od) return BackendError.InvalidArgument;
    }

    // Fast path: rank-2, normalized over last dim.
    if (rank == 2 and norm_rank == 1) {
        const tm: usize = out_meta.tile_shape[0];
        if (tm == 0) return BackendError.InvalidArgument;

        const ti0_total: usize = out_meta.tile_counts[0];
        const tile_bytes: usize = exec_utils.tileByteSize(out_meta);
        const min_total_bytes: usize = 256 * 1024;
        const tile_total: usize = out_meta.tile_counts[0] * out_meta.tile_counts[1];

        if (pool) |p| {
            if (exec_utils.shouldParallelTiles(thread_count, tile_total, tile_bytes, min_total_bytes) and ti0_total >= 2) {
                const Task = struct {
                    store: tensor_store.TensorStore,
                    out_meta: tensor_store.TensorMeta,
                    mode: Mode,
                    out: tensor_store.TensorId,
                    x: tensor_store.TensorId,
                    gamma: tensor_store.TensorId,
                    beta: tensor_store.TensorId,
                    eps: f32,

                    stop: std.atomic.Value(bool) = .init(false),
                    err_mutex: std.Io.Mutex = .init,
                    err_any: ?anyerror = null,

                    fn fail(t: *@This(), err: anyerror) void {
                        if (t.stop.swap(true, .acq_rel)) return;
                        std.Io.Threaded.mutexLock(&t.err_mutex);
                        defer std.Io.Threaded.mutexUnlock(&t.err_mutex);
                        if (t.err_any == null) t.err_any = err;
                    }

                    fn runTi0(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
                        _ = tid;
                        const t: *@This() = @ptrCast(@alignCast(ctx_any));
                        if (start >= end) return;
                        if (t.stop.load(.acquire)) return;
                        var ti0: usize = start;
                        while (ti0 < end) : (ti0 += 1) {
                            if (t.stop.load(.acquire)) return;
                            normTi0(t.store, t.out_meta, t.mode, t.out, t.x, t.gamma, t.beta, t.eps, ti0) catch |e| {
                                t.fail(e);
                                return;
                            };
                        }
                    }
                };

                var task: Task = .{ .store = store, .out_meta = out_meta, .mode = mode, .out = out, .x = x, .gamma = gamma, .beta = beta, .eps = eps };
                p.parallelForAny(@ptrCast(&task), ti0_total, 1, Task.runTi0);
                if (task.err_any) |e| return @errorCast(e);
                return;
            }
        }

        // Sequential fallback.
        var ti0: usize = 0;
        while (ti0 < ti0_total) : (ti0 += 1) {
            try normTi0(store, out_meta, mode, out, x, gamma, beta, eps, ti0);
        }
        return;
    }

    return execNormTiledND(pool, thread_count, mode, out, x, gamma, beta, eps, store, out_meta, g_meta, b_meta, norm_rank);
}

fn execNormTiledND(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    mode: Mode,
    out: tensor_store.TensorId,
    x: tensor_store.TensorId,
    gamma: tensor_store.TensorId,
    beta: tensor_store.TensorId,
    eps: f32,
    store: tensor_store.TensorStore,
    out_meta: tensor_store.TensorMeta,
    g_meta: tensor_store.TensorMeta,
    b_meta: tensor_store.TensorMeta,
    norm_rank: usize,
) ExecuteProgramError!void {
    const rank: usize = @as(usize, out_meta.rank);
    const row_dim_count: usize = rank - norm_rank;

    // Total normalized elements (per row).
    const norm_elems: usize = try productUsize(g_meta.shape);
    if (norm_elems == 0) return BackendError.InvalidArgument;
    const denom: f32 = @as(f32, @floatFromInt(norm_elems));
    const inv_denom: f32 = 1.0 / denom;

    var row_dims: [MAX_RANK]usize = undefined;
    var row_counts: [MAX_RANK]usize = undefined;
    var row_strides: [MAX_RANK]usize = undefined;
    var row_dim_idx: usize = 0;
    while (row_dim_idx < row_dim_count) : (row_dim_idx += 1) {
        row_dims[row_dim_idx] = row_dim_idx;
        row_counts[row_dim_idx] = out_meta.tile_counts[row_dim_idx];
    }

    var row_stride: usize = 1;
    var i: usize = row_dim_count;
    while (i > 0) : (i -= 1) {
        const idx: usize = i - 1;
        row_strides[idx] = row_stride;
        row_stride = std.math.mul(usize, row_stride, row_counts[idx]) catch return BackendError.InvalidArgument;
    }
    const row_tile_total: usize = row_stride;

    var norm_dims: [MAX_RANK]usize = undefined;
    var norm_counts: [MAX_RANK]usize = undefined;
    var norm_strides: [MAX_RANK]usize = undefined;
    var nd: usize = 0;
    while (nd < norm_rank) : (nd += 1) {
        norm_dims[nd] = row_dim_count + nd;
        norm_counts[nd] = out_meta.tile_counts[row_dim_count + nd];
    }

    var norm_stride: usize = 1;
    i = norm_rank;
    while (i > 0) : (i -= 1) {
        const idx: usize = i - 1;
        norm_strides[idx] = norm_stride;
        norm_stride = std.math.mul(usize, norm_stride, norm_counts[idx]) catch return BackendError.InvalidArgument;
    }
    const norm_tile_total: usize = norm_stride;

    const tile_bytes: usize = exec_utils.tileByteSize(out_meta);
    const min_total_bytes: usize = 256 * 1024;
    var tile_total: usize = 1;
    var d: usize = 0;
    while (d < rank) : (d += 1) {
        tile_total *= out_meta.tile_counts[d];
    }

    if (pool) |p| {
        if (exec_utils.shouldParallelTiles(thread_count, tile_total, tile_bytes, min_total_bytes) and row_tile_total >= 2) {
            const Task = struct {
                store: tensor_store.TensorStore,
                out_meta: tensor_store.TensorMeta,
                g_meta: tensor_store.TensorMeta,
                b_meta: tensor_store.TensorMeta,
                mode: Mode,
                out: tensor_store.TensorId,
                x: tensor_store.TensorId,
                gamma: tensor_store.TensorId,
                beta: tensor_store.TensorId,
                eps: f32,
                norm_rank: usize,
                row_dim_count: usize,
                row_dims: [MAX_RANK]usize,
                row_counts: [MAX_RANK]usize,
                row_strides: [MAX_RANK]usize,
                norm_dims: [MAX_RANK]usize,
                norm_counts: [MAX_RANK]usize,
                norm_strides: [MAX_RANK]usize,
                norm_tile_total: usize,
                inv_denom: f32,

                stop: std.atomic.Value(bool) = .init(false),
                err_mutex: std.Io.Mutex = .init,
                err_any: ?anyerror = null,

                fn fail(t: *@This(), err: anyerror) void {
                    if (t.stop.swap(true, .acq_rel)) return;
                    std.Io.Threaded.mutexLock(&t.err_mutex);
                    defer std.Io.Threaded.mutexUnlock(&t.err_mutex);
                    if (t.err_any == null) t.err_any = err;
                }

                fn runRowTile(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
                    _ = tid;
                    const t: *@This() = @ptrCast(@alignCast(ctx_any));
                    if (start >= end) return;
                    if (t.stop.load(.acquire)) return;

                    var coords: [MAX_RANK]usize = @splat(0);
                    var norm_coords: [MAX_RANK]usize = @splat(0);

                    var row_idx: usize = start;
                    while (row_idx < end) : (row_idx += 1) {
                        if (t.stop.load(.acquire)) return;
                        var rem: usize = row_idx;
                        var rdi: usize = 0;
                        while (rdi < t.row_dim_count) : (rdi += 1) {
                            const stride0: usize = t.row_strides[rdi];
                            const v: usize = rem / stride0;
                            coords[t.row_dims[rdi]] = v;
                            rem -= v * stride0;
                        }

                        var rows: usize = 1;
                        rdi = 0;
                        while (rdi < t.row_dim_count) : (rdi += 1) {
                            const dim: usize = t.row_dims[rdi];
                            const dim_len: usize = tileDim(t.out_meta.shape, t.out_meta.tile_shape, coords[dim], dim) catch |e| {
                                t.fail(e);
                                return;
                            };
                            rows = std.math.mul(usize, rows, dim_len) catch {
                                t.fail(BackendError.InvalidArgument);
                                return;
                            };
                        }
                        if (rows == 0) {
                            t.fail(BackendError.InvalidArgument);
                            return;
                        }

                        const elem_bytes: usize = switch (t.out_meta.dtype) {
                            .f32 => 4,
                            .f16 => 2,
                            else => {
                                t.fail(BackendError.InvalidArgument);
                                return;
                            },
                        };

                        var sum: [ROW_CHUNK_MAX]f32 = undefined;
                        var sumsq: [ROW_CHUNK_MAX]f32 = undefined;
                        var mean: [ROW_CHUNK_MAX]f32 = undefined;
                        var inv: [ROW_CHUNK_MAX]f32 = undefined;

                        var row_base: usize = 0;
                        while (row_base < rows) : (row_base += ROW_CHUNK_MAX) {
                            const rows_chunk: usize = @min(ROW_CHUNK_MAX, rows - row_base);
                            @memset(sum[0..rows_chunk], 0.0);
                            @memset(sumsq[0..rows_chunk], 0.0);

                            var nt: usize = 0;
                            while (nt < t.norm_tile_total) : (nt += 1) {
                                if (t.stop.load(.acquire)) return;
                                var remn: usize = nt;
                                var ndi: usize = 0;
                                while (ndi < t.norm_rank) : (ndi += 1) {
                                    const stride_n: usize = t.norm_strides[ndi];
                                    const v: usize = remn / stride_n;
                                    norm_coords[ndi] = v;
                                    remn -= v * stride_n;
                                }

                                ndi = 0;
                                while (ndi < t.norm_rank) : (ndi += 1) {
                                    coords[t.norm_dims[ndi]] = norm_coords[ndi];
                                }

                                const tile_index: usize = tensor_store.encodeTileIndex(t.out_meta, coords[0..t.out_meta.tile_counts.len]) catch |e| {
                                    t.fail(e);
                                    return;
                                };
                                const x_tile = t.store.acquireTileConstLinear(t.x, tile_index) catch |e| {
                                    t.fail(e);
                                    return;
                                };
                                defer t.store.releaseConst(x_tile.token);

                                var cols: usize = 1;
                                ndi = 0;
                                while (ndi < t.norm_rank) : (ndi += 1) {
                                    const dim: usize = t.norm_dims[ndi];
                                    const dim_len: usize = tileDim(t.out_meta.shape, t.out_meta.tile_shape, coords[dim], dim) catch |e| {
                                        t.fail(e);
                                        return;
                                    };
                                    cols = std.math.mul(usize, cols, dim_len) catch {
                                        t.fail(BackendError.InvalidArgument);
                                        return;
                                    };
                                }

                                const row_bytes: usize = std.math.mul(usize, cols, elem_bytes) catch {
                                    t.fail(BackendError.InvalidArgument);
                                    return;
                                };
                                const off_bytes: usize = std.math.mul(usize, row_base, row_bytes) catch {
                                    t.fail(BackendError.InvalidArgument);
                                    return;
                                };
                                const chunk_bytes: usize = std.math.mul(usize, rows_chunk, row_bytes) catch {
                                    t.fail(BackendError.InvalidArgument);
                                    return;
                                };
                                const end_bytes: usize = std.math.add(usize, off_bytes, chunk_bytes) catch {
                                    t.fail(BackendError.InvalidArgument);
                                    return;
                                };
                                if (end_bytes > x_tile.bytes.len) {
                                    t.fail(BackendError.InvalidArgument);
                                    return;
                                }

                                var shape_mem: [2]usize = .{ rows_chunk, cols };
                                var strides_mem: [2]isize = .{ @intCast(row_bytes), @intCast(elem_bytes) };
                                const x_view: types.BufferViewConst = .{ .bytes = x_tile.bytes[off_bytes..end_bytes], .dtype = t.out_meta.dtype, .layout = .{ .rank = 2, .shape = shape_mem[0..2], .strides_bytes = strides_mem[0..2] } };
                                layernorm_kernels.accumulateStats(sum[0..rows_chunk], sumsq[0..rows_chunk], x_view);
                            }

                            var r0: usize = 0;
                            while (r0 < rows_chunk) : (r0 += 1) {
                                const mu: f32 = sum[r0] * t.inv_denom;
                                mean[r0] = mu;
                                const msq: f32 = sumsq[r0] * t.inv_denom;
                                const v: f32 = if (t.mode == .layernorm) @max(@as(f32, 0.0), msq - (mu * mu)) else msq;
                                const d0: f32 = v + t.eps;
                                if (!(d0 > 0.0) or !std.math.isFinite(d0)) {
                                    t.fail(BackendError.InvalidArgument);
                                    return;
                                }
                                inv[r0] = 1.0 / std.math.sqrt(d0);
                            }

                            nt = 0;
                            while (nt < t.norm_tile_total) : (nt += 1) {
                                if (t.stop.load(.acquire)) return;
                                var remn2: usize = nt;
                                var ndi2: usize = 0;
                                while (ndi2 < t.norm_rank) : (ndi2 += 1) {
                                    const stride_n: usize = t.norm_strides[ndi2];
                                    const v2: usize = remn2 / stride_n;
                                    norm_coords[ndi2] = v2;
                                    remn2 -= v2 * stride_n;
                                }

                                ndi2 = 0;
                                while (ndi2 < t.norm_rank) : (ndi2 += 1) {
                                    coords[t.norm_dims[ndi2]] = norm_coords[ndi2];
                                }

                                const tile_index: usize = tensor_store.encodeTileIndex(t.out_meta, coords[0..t.out_meta.tile_counts.len]) catch |e| {
                                    t.fail(e);
                                    return;
                                };
                                const out_tile = t.store.acquireTileMutLinear(t.out, tile_index) catch |e| {
                                    t.fail(e);
                                    return;
                                };
                                defer t.store.releaseMut(out_tile.token);
                                const x_tile = t.store.acquireTileConstLinear(t.x, tile_index) catch |e| {
                                    t.fail(e);
                                    return;
                                };
                                defer t.store.releaseConst(x_tile.token);

                                const g_index: usize = tensor_store.encodeTileIndex(t.g_meta, norm_coords[0..t.norm_rank]) catch |e| {
                                    t.fail(e);
                                    return;
                                };
                                const g_tile = t.store.acquireTileConstLinear(t.gamma, g_index) catch |e| {
                                    t.fail(e);
                                    return;
                                };
                                defer t.store.releaseConst(g_tile.token);

                                const b_index: usize = tensor_store.encodeTileIndex(t.b_meta, norm_coords[0..t.norm_rank]) catch |e| {
                                    t.fail(e);
                                    return;
                                };
                                const b_tile = t.store.acquireTileConstLinear(t.beta, b_index) catch |e| {
                                    t.fail(e);
                                    return;
                                };
                                defer t.store.releaseConst(b_tile.token);

                                var cols2: usize = 1;
                                ndi2 = 0;
                                while (ndi2 < t.norm_rank) : (ndi2 += 1) {
                                    const dim: usize = t.norm_dims[ndi2];
                                    const dim_len: usize = tileDim(t.out_meta.shape, t.out_meta.tile_shape, coords[dim], dim) catch |e| {
                                        t.fail(e);
                                        return;
                                    };
                                    cols2 = std.math.mul(usize, cols2, dim_len) catch {
                                        t.fail(BackendError.InvalidArgument);
                                        return;
                                    };
                                }

                                const row_bytes2: usize = std.math.mul(usize, cols2, elem_bytes) catch {
                                    t.fail(BackendError.InvalidArgument);
                                    return;
                                };
                                const off_bytes2: usize = std.math.mul(usize, row_base, row_bytes2) catch {
                                    t.fail(BackendError.InvalidArgument);
                                    return;
                                };
                                const chunk_bytes2: usize = std.math.mul(usize, rows_chunk, row_bytes2) catch {
                                    t.fail(BackendError.InvalidArgument);
                                    return;
                                };
                                const end_bytes2: usize = std.math.add(usize, off_bytes2, chunk_bytes2) catch {
                                    t.fail(BackendError.InvalidArgument);
                                    return;
                                };
                                if (end_bytes2 > out_tile.bytes.len or end_bytes2 > x_tile.bytes.len) {
                                    t.fail(BackendError.InvalidArgument);
                                    return;
                                }

                                var shape_mem: [2]usize = .{ rows_chunk, cols2 };
                                var strides_mem: [2]isize = .{ @intCast(row_bytes2), @intCast(elem_bytes) };
                                var shape_vec: [1]usize = .{cols2};
                                var strides_vec: [1]isize = .{@intCast(elem_bytes)};

                                const out_view: types.BufferViewMut = .{ .bytes = out_tile.bytes[off_bytes2..end_bytes2], .dtype = t.out_meta.dtype, .layout = .{ .rank = 2, .shape = shape_mem[0..2], .strides_bytes = strides_mem[0..2] } };
                                const x_view: types.BufferViewConst = .{ .bytes = x_tile.bytes[off_bytes2..end_bytes2], .dtype = t.out_meta.dtype, .layout = .{ .rank = 2, .shape = shape_mem[0..2], .strides_bytes = strides_mem[0..2] } };
                                const g_view: types.BufferViewConst = .{ .bytes = g_tile.bytes, .dtype = t.out_meta.dtype, .layout = .{ .rank = 1, .shape = shape_vec[0..1], .strides_bytes = strides_vec[0..1] } };
                                const b_view: types.BufferViewConst = .{ .bytes = b_tile.bytes, .dtype = t.out_meta.dtype, .layout = .{ .rank = 1, .shape = shape_vec[0..1], .strides_bytes = strides_vec[0..1] } };
                                layernorm_kernels.applyNorm(t.mode, mean[0..rows_chunk], inv[0..rows_chunk], out_view, x_view, g_view, b_view);
                            }
                        }
                    }
                }
            };

            var task: Task = .{
                .store = store,
                .out_meta = out_meta,
                .g_meta = g_meta,
                .b_meta = b_meta,
                .mode = mode,
                .out = out,
                .x = x,
                .gamma = gamma,
                .beta = beta,
                .eps = eps,
                .norm_rank = norm_rank,
                .row_dim_count = row_dim_count,
                .row_dims = row_dims,
                .row_counts = row_counts,
                .row_strides = row_strides,
                .norm_dims = norm_dims,
                .norm_counts = norm_counts,
                .norm_strides = norm_strides,
                .norm_tile_total = norm_tile_total,
                .inv_denom = inv_denom,
            };
            p.parallelForAny(@ptrCast(&task), row_tile_total, 1, Task.runRowTile);
            if (task.err_any) |e| return @errorCast(e);
            return;
        }
    }

    // Sequential fallback.
    var coords: [MAX_RANK]usize = @splat(0);
    var norm_coords: [MAX_RANK]usize = @splat(0);

    var row_idx: usize = 0;
    while (row_idx < row_tile_total) : (row_idx += 1) {
        var rem: usize = row_idx;
        var rdi: usize = 0;
        while (rdi < row_dim_count) : (rdi += 1) {
            const stride0: usize = row_strides[rdi];
            const v: usize = rem / stride0;
            coords[row_dims[rdi]] = v;
            rem -= v * stride0;
        }

        var rows: usize = 1;
        rdi = 0;
        while (rdi < row_dim_count) : (rdi += 1) {
            const dim: usize = row_dims[rdi];
            const dim_len: usize = try tileDim(out_meta.shape, out_meta.tile_shape, coords[dim], dim);
            rows = std.math.mul(usize, rows, dim_len) catch return BackendError.InvalidArgument;
        }
        if (rows == 0) return BackendError.InvalidArgument;

        const elem_bytes: usize = switch (out_meta.dtype) {
            .f32 => 4,
            .f16 => 2,
            else => return BackendError.InvalidArgument,
        };

        var sum: [ROW_CHUNK_MAX]f32 = undefined;
        var sumsq: [ROW_CHUNK_MAX]f32 = undefined;
        var mean: [ROW_CHUNK_MAX]f32 = undefined;
        var inv: [ROW_CHUNK_MAX]f32 = undefined;

        var row_base: usize = 0;
        while (row_base < rows) : (row_base += ROW_CHUNK_MAX) {
            const rows_chunk: usize = @min(ROW_CHUNK_MAX, rows - row_base);
            @memset(sum[0..rows_chunk], 0.0);
            @memset(sumsq[0..rows_chunk], 0.0);

            var nt: usize = 0;
            while (nt < norm_tile_total) : (nt += 1) {
                var remn: usize = nt;
                var ndi: usize = 0;
                while (ndi < norm_rank) : (ndi += 1) {
                    const stride_n: usize = norm_strides[ndi];
                    const v: usize = remn / stride_n;
                    norm_coords[ndi] = v;
                    remn -= v * stride_n;
                }

                ndi = 0;
                while (ndi < norm_rank) : (ndi += 1) {
                    coords[norm_dims[ndi]] = norm_coords[ndi];
                }

                const tile_index: usize = try tensor_store.encodeTileIndex(out_meta, coords[0..out_meta.tile_counts.len]);
                const x_tile = try store.acquireTileConstLinear(x, tile_index);
                defer store.releaseConst(x_tile.token);

                var cols: usize = 1;
                ndi = 0;
                while (ndi < norm_rank) : (ndi += 1) {
                    const dim: usize = norm_dims[ndi];
                    const dim_len: usize = try tileDim(out_meta.shape, out_meta.tile_shape, coords[dim], dim);
                    cols = std.math.mul(usize, cols, dim_len) catch return BackendError.InvalidArgument;
                }

                const row_bytes: usize = std.math.mul(usize, cols, elem_bytes) catch return BackendError.InvalidArgument;
                const off_bytes: usize = std.math.mul(usize, row_base, row_bytes) catch return BackendError.InvalidArgument;
                const chunk_bytes: usize = std.math.mul(usize, rows_chunk, row_bytes) catch return BackendError.InvalidArgument;
                const end_bytes: usize = std.math.add(usize, off_bytes, chunk_bytes) catch return BackendError.InvalidArgument;
                if (end_bytes > x_tile.bytes.len) return BackendError.InvalidArgument;

                var shape_mem: [2]usize = .{ rows_chunk, cols };
                var strides_mem: [2]isize = .{ @intCast(row_bytes), @intCast(elem_bytes) };
                const x_view: types.BufferViewConst = .{ .bytes = x_tile.bytes[off_bytes..end_bytes], .dtype = out_meta.dtype, .layout = .{ .rank = 2, .shape = shape_mem[0..2], .strides_bytes = strides_mem[0..2] } };
                layernorm_kernels.accumulateStats(sum[0..rows_chunk], sumsq[0..rows_chunk], x_view);
            }

            var r0: usize = 0;
            while (r0 < rows_chunk) : (r0 += 1) {
                const mu: f32 = sum[r0] * inv_denom;
                mean[r0] = mu;
                const msq: f32 = sumsq[r0] * inv_denom;
                const v: f32 = if (mode == .layernorm) @max(@as(f32, 0.0), msq - (mu * mu)) else msq;
                const d0: f32 = v + eps;
                if (!(d0 > 0.0) or !std.math.isFinite(d0)) return BackendError.InvalidArgument;
                inv[r0] = 1.0 / std.math.sqrt(d0);
            }

            nt = 0;
            while (nt < norm_tile_total) : (nt += 1) {
                var remn2: usize = nt;
                var ndi2: usize = 0;
                while (ndi2 < norm_rank) : (ndi2 += 1) {
                    const stride_n: usize = norm_strides[ndi2];
                    const v2: usize = remn2 / stride_n;
                    norm_coords[ndi2] = v2;
                    remn2 -= v2 * stride_n;
                }

                ndi2 = 0;
                while (ndi2 < norm_rank) : (ndi2 += 1) {
                    coords[norm_dims[ndi2]] = norm_coords[ndi2];
                }

                const tile_index: usize = try tensor_store.encodeTileIndex(out_meta, coords[0..out_meta.tile_counts.len]);
                const out_tile = try store.acquireTileMutLinear(out, tile_index);
                defer store.releaseMut(out_tile.token);
                const x_tile = try store.acquireTileConstLinear(x, tile_index);
                defer store.releaseConst(x_tile.token);

                const g_index: usize = try tensor_store.encodeTileIndex(g_meta, norm_coords[0..norm_rank]);
                const b_index: usize = try tensor_store.encodeTileIndex(b_meta, norm_coords[0..norm_rank]);
                const g_tile = try store.acquireTileConstLinear(gamma, g_index);
                defer store.releaseConst(g_tile.token);
                const b_tile = try store.acquireTileConstLinear(beta, b_index);
                defer store.releaseConst(b_tile.token);

                var cols2: usize = 1;
                ndi2 = 0;
                while (ndi2 < norm_rank) : (ndi2 += 1) {
                    const dim: usize = norm_dims[ndi2];
                    const dim_len: usize = try tileDim(out_meta.shape, out_meta.tile_shape, coords[dim], dim);
                    cols2 = std.math.mul(usize, cols2, dim_len) catch return BackendError.InvalidArgument;
                }

                const row_bytes2: usize = std.math.mul(usize, cols2, elem_bytes) catch return BackendError.InvalidArgument;
                const off_bytes2: usize = std.math.mul(usize, row_base, row_bytes2) catch return BackendError.InvalidArgument;
                const chunk_bytes2: usize = std.math.mul(usize, rows_chunk, row_bytes2) catch return BackendError.InvalidArgument;
                const end_bytes2: usize = std.math.add(usize, off_bytes2, chunk_bytes2) catch return BackendError.InvalidArgument;
                if (end_bytes2 > out_tile.bytes.len or end_bytes2 > x_tile.bytes.len) return BackendError.InvalidArgument;

                var shape_mem: [2]usize = .{ rows_chunk, cols2 };
                var strides_mem: [2]isize = .{ @intCast(row_bytes2), @intCast(elem_bytes) };
                var shape_vec: [1]usize = .{cols2};
                var strides_vec: [1]isize = .{@intCast(elem_bytes)};

                const out_view: types.BufferViewMut = .{ .bytes = out_tile.bytes[off_bytes2..end_bytes2], .dtype = out_meta.dtype, .layout = .{ .rank = 2, .shape = shape_mem[0..2], .strides_bytes = strides_mem[0..2] } };
                const x_view: types.BufferViewConst = .{ .bytes = x_tile.bytes[off_bytes2..end_bytes2], .dtype = out_meta.dtype, .layout = .{ .rank = 2, .shape = shape_mem[0..2], .strides_bytes = strides_mem[0..2] } };
                const g_view: types.BufferViewConst = .{ .bytes = g_tile.bytes, .dtype = out_meta.dtype, .layout = .{ .rank = 1, .shape = shape_vec[0..1], .strides_bytes = strides_vec[0..1] } };
                const b_view: types.BufferViewConst = .{ .bytes = b_tile.bytes, .dtype = out_meta.dtype, .layout = .{ .rank = 1, .shape = shape_vec[0..1], .strides_bytes = strides_vec[0..1] } };
                layernorm_kernels.applyNorm(mode, mean[0..rows_chunk], inv[0..rows_chunk], out_view, x_view, g_view, b_view);
            }
        }
    }
}

fn normTi0(
    store: tensor_store.TensorStore,
    out_meta: tensor_store.TensorMeta,
    mode: Mode,
    out: tensor_store.TensorId,
    x: tensor_store.TensorId,
    gamma: tensor_store.TensorId,
    beta: tensor_store.TensorId,
    eps: f32,
    ti0: usize,
) ExecuteProgramError!void {
    // Stats are per-row within the output tile-row.
    const tm: usize = out_meta.tile_shape[0];
    const tc1: usize = out_meta.tile_counts[1];
    const total_cols: usize = out_meta.shape[1];
    if (total_cols == 0) return BackendError.InvalidArgument;
    const denom: f32 = @as(f32, @floatFromInt(total_cols));
    const inv_denom: f32 = 1.0 / denom;

    const elem_bytes: usize = switch (out_meta.dtype) {
        .f32 => 4,
        .f16 => 2,
        else => return BackendError.InvalidArgument,
    };

    // Last tile-row may be short.
    const full_rows: usize = out_meta.shape[0];
    const start_row: usize = ti0 * out_meta.tile_shape[0];
    const valid_rows: usize = if (start_row >= full_rows) 0 else @min(tm, full_rows - start_row);

    // Chunked row processing keeps stack scratch bounded.
    var sum: [ROW_CHUNK_MAX]f32 = undefined;
    var sumsq: [ROW_CHUNK_MAX]f32 = undefined;
    var mean: [ROW_CHUNK_MAX]f32 = undefined;
    var inv: [ROW_CHUNK_MAX]f32 = undefined;

    var r_base: usize = 0;
    while (r_base < valid_rows) : (r_base += ROW_CHUNK_MAX) {
        const r_len: usize = @min(ROW_CHUNK_MAX, valid_rows - r_base);
        @memset(sum[0..r_len], 0.0);
        @memset(sumsq[0..r_len], 0.0);

        // Pass 1: accumulate sum and sumsq per row chunk.
        var ti1: usize = 0;
        while (ti1 < tc1) : (ti1 += 1) {
            if (ti1 + 1 < tc1) store.prefetch(x, ti0, ti1 + 1);
            const xt = try store.acquireTileConst(x, ti0, ti1);
            defer store.releaseConst(xt.token);
            const xv_full = xt.bufferView();
            if (xv_full.dtype != out_meta.dtype) return BackendError.InvalidArgument;
            if (xv_full.layout.rank != 2) return BackendError.InvalidArgument;

            const n_tile: usize = xv_full.layout.shape[1];
            const row_bytes: usize = std.math.mul(usize, n_tile, elem_bytes) catch return BackendError.InvalidArgument;
            const off_bytes: usize = std.math.mul(usize, r_base, row_bytes) catch return BackendError.InvalidArgument;
            const chunk_bytes: usize = std.math.mul(usize, r_len, row_bytes) catch return BackendError.InvalidArgument;
            const end_bytes: usize = std.math.add(usize, off_bytes, chunk_bytes) catch return BackendError.InvalidArgument;
            if (end_bytes > xv_full.bytes.len) return BackendError.InvalidArgument;

            var shape_mem: [2]usize = .{ r_len, n_tile };
            var strides_mem: [2]isize = .{ @intCast(row_bytes), @intCast(elem_bytes) };
            const xv: types.BufferViewConst = .{ .bytes = xv_full.bytes[off_bytes..end_bytes], .dtype = out_meta.dtype, .layout = .{ .rank = 2, .shape = shape_mem[0..2], .strides_bytes = strides_mem[0..2] } };
            layernorm_kernels.accumulateStats(sum[0..r_len], sumsq[0..r_len], xv);
        }

        // Compute mean and inv for each row within the chunk.
        var r0: usize = 0;
        while (r0 < r_len) : (r0 += 1) {
            const mu: f32 = sum[r0] * inv_denom;
            mean[r0] = mu;
            const msq: f32 = sumsq[r0] * inv_denom;
            const v: f32 = if (mode == .layernorm) @max(@as(f32, 0.0), msq - (mu * mu)) else msq;
            const d: f32 = v + eps;
            // Guard against pathological inputs (shouldn't happen for finite x and eps>0,
            // but keep the runtime robust).
            if (!(d > 0.0) or !std.math.isFinite(d)) return BackendError.InvalidArgument;
            inv[r0] = 1.0 / std.math.sqrt(d);
        }

        // Pass 2: apply normalization, gamma/beta.
        ti1 = 0;
        while (ti1 < tc1) : (ti1 += 1) {
            if (ti1 + 1 < tc1) {
                store.prefetch(x, ti0, ti1 + 1);
                store.prefetch(out, ti0, ti1 + 1);
                store.prefetch(gamma, ti1 + 1, 0);
                store.prefetch(beta, ti1 + 1, 0);
            }

            var out_t = try store.acquireTileMut(out, ti0, ti1);
            defer store.releaseMut(out_t.token);
            const xt2 = try store.acquireTileConst(x, ti0, ti1);
            defer store.releaseConst(xt2.token);
            const gt = try store.acquireTileConst(gamma, ti1, 0);
            defer store.releaseConst(gt.token);
            const bt = try store.acquireTileConst(beta, ti1, 0);
            defer store.releaseConst(bt.token);

            const ov_full = out_t.bufferView();
            const xv_full2 = xt2.bufferView();
            const gv = gt.bufferView();
            const bv = bt.bufferView();
            if (ov_full.dtype != out_meta.dtype or xv_full2.dtype != out_meta.dtype or gv.dtype != out_meta.dtype or bv.dtype != out_meta.dtype) return BackendError.InvalidArgument;
            if (ov_full.layout.rank != 2 or xv_full2.layout.rank != 2) return BackendError.InvalidArgument;
            if (gv.layout.rank != 1 or bv.layout.rank != 1) return BackendError.InvalidArgument;

            const n_tile: usize = ov_full.layout.shape[1];
            const row_bytes: usize = std.math.mul(usize, n_tile, elem_bytes) catch return BackendError.InvalidArgument;
            const off_bytes: usize = std.math.mul(usize, r_base, row_bytes) catch return BackendError.InvalidArgument;
            const chunk_bytes: usize = std.math.mul(usize, r_len, row_bytes) catch return BackendError.InvalidArgument;
            const end_bytes: usize = std.math.add(usize, off_bytes, chunk_bytes) catch return BackendError.InvalidArgument;
            if (end_bytes > ov_full.bytes.len or end_bytes > xv_full2.bytes.len) return BackendError.InvalidArgument;

            var shape_mem: [2]usize = .{ r_len, n_tile };
            var strides_mem: [2]isize = .{ @intCast(row_bytes), @intCast(elem_bytes) };
            const ov: types.BufferViewMut = .{ .bytes = ov_full.bytes[off_bytes..end_bytes], .dtype = out_meta.dtype, .layout = .{ .rank = 2, .shape = shape_mem[0..2], .strides_bytes = strides_mem[0..2] } };
            const xv: types.BufferViewConst = .{ .bytes = xv_full2.bytes[off_bytes..end_bytes], .dtype = out_meta.dtype, .layout = .{ .rank = 2, .shape = shape_mem[0..2], .strides_bytes = strides_mem[0..2] } };
            layernorm_kernels.applyNorm(mode, mean[0..r_len], inv[0..r_len], ov, xv, gv, bv);
        }
    }
}
