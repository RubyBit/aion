// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const elemwise = @import("../kernels/elemwise.zig");
const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const exec_utils = @import("utils.zig");
const unary_exec = @import("unary.zig");
const executable = @import("../../../runtime/executable.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

fn execElemwiseBinaryTile(
    store: tensor_store.TensorStore,
    out_meta: tensor_store.TensorMeta,
    a_meta: tensor_store.TensorMeta,
    b_meta: tensor_store.TensorMeta,
    s: executable.StepElemwiseBinaryTiled,
    tile_index: usize,
) !void {
    var coords_buf: [tensor_store.INLINE_RANK]usize = @splat(0);
    const coords = coords_buf[0..@as(usize, out_meta.rank)];
    try tensor_store.decodeTileCoords(out_meta, tile_index, coords);
    const a_index = try tensor_store.projectTileIndex(a_meta, coords, &.{}, s.broadcast.a_broadcast_axes);
    const b_index = try tensor_store.projectTileIndex(b_meta, coords, &.{}, s.broadcast.b_broadcast_axes);

    const out_tile = try store.acquireTileMutLinear(s.out, tile_index);
    defer store.releaseMut(out_tile.token);
    const a_tile = try store.acquireTileConstLinear(s.a, a_index);
    defer store.releaseConst(a_tile.token);
    const b_tile = try store.acquireTileConstLinear(s.b, b_index);
    defer store.releaseConst(b_tile.token);

    const out_view = out_tile.bufferView();
    const a_view = a_tile.bufferView();
    const b_view = b_tile.bufferView();
    const n = exec_utils.elemCountFromTileView(out_view);

    // A gate is the activation folded into the multiply: apply `act` into `out`, then
    // multiply in place by `b`. Same order of operations as the unfused
    // `UnaryTiled(act)` + `ElemwiseBinaryTiled(mul)` pair, so the result is
    // bit-identical to it — which is what lets the GPU fused kernel be tested against
    // this. Threading, tiling and broadcast validation all come from the surrounding
    // elementwise machinery instead of a second copy of it.
    if (s.op == .gate) {
        if (s.broadcast.kind != .identical) return BackendError.InvalidArgument;
        if (out_view.dtype != .f32) return BackendError.InvalidArgument;
        try unary_exec.dispatchF32(s.act, out_view.bytes, a_view.bytes, n);
        return elemwise.elemwiseBinaryF32(.mul, out_view.bytes, out_view.bytes, b_view.bytes, n);
    }

    if (s.broadcast.kind == .identical) {
        return switch (out_view.dtype) {
            .f32 => elemwise.elemwiseBinaryF32(s.op, out_view.bytes, a_view.bytes, b_view.bytes, n),
            .f16 => elemwise.elemwiseBinaryF16(s.op, out_view.bytes, a_view.bytes, b_view.bytes, n),
            .i32 => elemwise.elemwiseBinaryI32(s.op, out_view.bytes, a_view.bytes, b_view.bytes, n),
            else => BackendError.InvalidArgument,
        };
    }

    if (s.broadcast.kind == .scalar_b or s.broadcast.kind == .contiguous_suffix_b) {
        const cols = exec_utils.elemCountFromTileView(b_view);
        return switch (out_view.dtype) {
            .f32 => switch (s.op) {
                .add => elemwise.contiguousSuffixBinaryF32Packed(.add, out_view.bytes, a_view.bytes, b_view.bytes, n, cols),
                .sub => elemwise.contiguousSuffixBinaryF32Packed(.sub, out_view.bytes, a_view.bytes, b_view.bytes, n, cols),
                .mul => elemwise.contiguousSuffixBinaryF32Packed(.mul, out_view.bytes, a_view.bytes, b_view.bytes, n, cols),
                .div => elemwise.contiguousSuffixBinaryF32Packed(.div, out_view.bytes, a_view.bytes, b_view.bytes, n, cols),
                else => BackendError.InvalidArgument,
            },
            .f16 => switch (s.op) {
                .add => elemwise.contiguousSuffixBinaryF16Packed(.add, out_view.bytes, a_view.bytes, b_view.bytes, n, cols),
                .sub => elemwise.contiguousSuffixBinaryF16Packed(.sub, out_view.bytes, a_view.bytes, b_view.bytes, n, cols),
                .mul => elemwise.contiguousSuffixBinaryF16Packed(.mul, out_view.bytes, a_view.bytes, b_view.bytes, n, cols),
                .div => elemwise.contiguousSuffixBinaryF16Packed(.div, out_view.bytes, a_view.bytes, b_view.bytes, n, cols),
                else => BackendError.InvalidArgument,
            },
            .i32 => elemwise.contiguousSuffixBinaryI32Packed(s.op, out_view.bytes, a_view.bytes, b_view.bytes, n, cols),
            else => BackendError.InvalidArgument,
        };
    }

    return switch (out_view.dtype) {
        .f32 => elemwise.elemwiseBroadcastF32(
            s.op,
            out_view,
            a_view,
            b_view,
            s.broadcast.a_broadcast_axes,
            s.broadcast.b_broadcast_axes,
        ),
        .f16 => elemwise.elemwiseBroadcastF16(
            s.op,
            out_view,
            a_view,
            b_view,
            s.broadcast.a_broadcast_axes,
            s.broadcast.b_broadcast_axes,
        ),
        .i32 => elemwise.elemwiseBroadcastI32(
            s.op,
            out_view,
            a_view,
            b_view,
            s.broadcast.a_broadcast_axes,
            s.broadcast.b_broadcast_axes,
        ),
        else => BackendError.InvalidArgument,
    };
}

pub fn execElemwiseBinaryTiled(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepElemwiseBinaryTiled,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    const out_meta = try store.meta(s.out);
    const a_meta = try store.meta(s.a);
    const b_meta = try store.meta(s.b);
    var tile_total: usize = 1;
    var d: usize = 0;
    while (d < @as(usize, out_meta.rank)) : (d += 1) {
        tile_total *= out_meta.tile_counts[d];
    }
    const tile_bytes: usize = exec_utils.tileByteSize(out_meta);
    const min_total_bytes: usize = 256 * 1024; // aim for at least 256KiB of work

    if (pool) |p| {
        if (exec_utils.shouldParallelTiles(thread_count, tile_total, tile_bytes, min_total_bytes)) {
            const Task = struct {
                store: tensor_store.TensorStore,
                out_meta: tensor_store.TensorMeta,
                a_meta: tensor_store.TensorMeta,
                b_meta: tensor_store.TensorMeta,
                op: types.ElemwiseBinaryOp,
                out: tensor_store.TensorId,
                a: tensor_store.TensorId,
                b: tensor_store.TensorId,
                broadcast: executable.ElementwiseBroadcastPlan,

                stop: std.atomic.Value(bool) = .init(false),
                err_mutex: std.Io.Mutex = .init,
                err_any: ?anyerror = null,

                fn fail(t: *@This(), err: anyerror) void {
                    if (t.stop.swap(true, .acq_rel)) return;
                    std.Io.Threaded.mutexLock(&t.err_mutex);
                    defer std.Io.Threaded.mutexUnlock(&t.err_mutex);
                    if (t.err_any == null) t.err_any = err;
                }

                fn runTiles(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
                    _ = tid;
                    const t: *@This() = @ptrCast(@alignCast(ctx_any));
                    if (start >= end) return;

                    if (t.stop.load(.acquire)) return;

                    var i: usize = start;
                    while (i < end) : (i += 1) {
                        if (t.stop.load(.acquire)) return;
                        const step: executable.StepElemwiseBinaryTiled = .{
                            .op = t.op,
                            .out = t.out,
                            .a = t.a,
                            .b = t.b,
                            .broadcast = t.broadcast,
                        };
                        execElemwiseBinaryTile(t.store, t.out_meta, t.a_meta, t.b_meta, step, i) catch |e| {
                            t.fail(e);
                            return;
                        };
                    }
                }
            };

            var task: Task = .{
                .store = store,
                .out_meta = out_meta,
                .a_meta = a_meta,
                .b_meta = b_meta,
                .op = s.op,
                .out = s.out,
                .a = s.a,
                .b = s.b,
                .broadcast = s.broadcast,
            };
            // Grain based on bytes-per-tile: target ~256KiB per chunk.
            var grain: usize = if (tile_bytes == 0) 32 else @max(@as(usize, 1), min_total_bytes / tile_bytes);
            if (grain > tile_total) grain = tile_total;
            p.parallelForAny(@ptrCast(&task), tile_total, grain, Task.runTiles);
            if (task.err_any) |e| return @errorCast(e);
            return;
        }
    }

    // Sequential fallback.
    var tile_index: usize = 0;
    while (tile_index < tile_total) : (tile_index += 1) {
        try execElemwiseBinaryTile(store, out_meta, a_meta, b_meta, s, tile_index);
    }
}

pub fn execCopyTiled(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepCopyTiled,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    const dst_meta = try store.meta(s.dst);
    var tile_total: usize = 1;
    var d: usize = 0;
    while (d < @as(usize, dst_meta.rank)) : (d += 1) {
        tile_total *= dst_meta.tile_counts[d];
    }
    const tile_bytes: usize = exec_utils.tileByteSize(dst_meta);
    const min_total_bytes: usize = 256 * 1024;

    const bytesForTileView = struct {
        fn calc(dtype: types.DType, view: anytype) usize {
            const elems: usize = if (view.layout.rank == 1) view.layout.shape[0] else (view.layout.shape[0] * view.layout.shape[1]);
            return switch (dtype) {
                .f32 => elems * 4,
                .f16 => elems * 2,
                .i8 => elems,
                .i32 => elems * 4,
                .q4_0, .q8_0 => blk: {
                    const info = dtype.info();
                    const blocks = std.math.divCeil(usize, elems, info.block_elems) catch return 0;
                    break :blk blocks * info.block_bytes;
                },
            };
        }
    };

    if (pool) |p| {
        if (exec_utils.shouldParallelTiles(thread_count, tile_total, tile_bytes, min_total_bytes)) {
            const Task = struct {
                store: tensor_store.TensorStore,
                dst_meta: tensor_store.TensorMeta,
                dst: tensor_store.TensorId,
                src: tensor_store.TensorId,

                stop: std.atomic.Value(bool) = .init(false),
                err_mutex: std.Io.Mutex = .init,
                err_any: ?anyerror = null,

                fn fail(t: *@This(), err: anyerror) void {
                    if (t.stop.swap(true, .acq_rel)) return;
                    std.Io.Threaded.mutexLock(&t.err_mutex);
                    defer std.Io.Threaded.mutexUnlock(&t.err_mutex);
                    if (t.err_any == null) t.err_any = err;
                }

                fn runTiles(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
                    _ = tid;
                    const t: *@This() = @ptrCast(@alignCast(ctx_any));
                    if (start >= end) return;
                    if (t.stop.load(.acquire)) return;

                    var i: usize = start;
                    while (i < end) : (i += 1) {
                        if (t.stop.load(.acquire)) return;
                        if (i + 1 < end) {
                            t.store.prefetchLinear(t.src, i + 1);
                        }

                        var dst_tile = t.store.acquireTileMutLinear(t.dst, i) catch |e| {
                            t.fail(e);
                            return;
                        };
                        defer t.store.releaseMut(dst_tile.token);
                        const src_tile = t.store.acquireTileConstLinear(t.src, i) catch |e| {
                            t.fail(e);
                            return;
                        };
                        defer t.store.releaseConst(src_tile.token);

                        const dst_view = dst_tile.bufferView();
                        const src_view = src_tile.bufferView();
                        if (dst_view.dtype != src_view.dtype) {
                            t.fail(BackendError.InvalidArgument);
                            return;
                        }
                        const need: usize = bytesForTileView.calc(dst_view.dtype, dst_view);
                        if (need == 0 or dst_view.bytes.len < need or src_view.bytes.len < need) {
                            t.fail(BackendError.InvalidArgument);
                            return;
                        }
                        @memcpy(dst_view.bytes[0..need], src_view.bytes[0..need]);
                    }
                }
            };

            var task: Task = .{ .store = store, .dst_meta = dst_meta, .dst = s.dst, .src = s.src };
            var grain: usize = if (tile_bytes == 0) 32 else @max(@as(usize, 1), min_total_bytes / tile_bytes);
            if (grain > tile_total) grain = tile_total;
            p.parallelForAny(@ptrCast(&task), tile_total, grain, Task.runTiles);
            if (task.err_any) |e| return @errorCast(e);
            return;
        }
    }

    // Sequential fallback.
    var tile_index: usize = 0;
    while (tile_index < tile_total) : (tile_index += 1) {
        var dst_tile = try store.acquireTileMutLinear(s.dst, tile_index);
        defer store.releaseMut(dst_tile.token);
        const src_tile = try store.acquireTileConstLinear(s.src, tile_index);
        defer store.releaseConst(src_tile.token);

        const dst_view = dst_tile.bufferView();
        const src_view = src_tile.bufferView();
        if (dst_view.dtype != src_view.dtype) return BackendError.InvalidArgument;
        const need: usize = bytesForTileView.calc(dst_view.dtype, dst_view);
        if (need == 0 or dst_view.bytes.len < need or src_view.bytes.len < need) return BackendError.InvalidArgument;
        @memcpy(dst_view.bytes[0..need], src_view.bytes[0..need]);
    }
}
