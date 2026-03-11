const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const elemwise = @import("../kernels/elemwise.zig");
const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const exec_utils = @import("utils.zig");
const executable = @import("../../../runtime/executable.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

pub fn execElemwiseBinaryTiled(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepElemwiseBinaryTiled,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    const out_meta = try store.meta(s.out);
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
                op: types.ElemwiseBinaryOp,
                out: tensor_store.TensorId,
                a: tensor_store.TensorId,
                b: tensor_store.TensorId,

                stop: std.atomic.Value(bool) = .init(false),
                err_mutex: std.Thread.Mutex = .{},
                err_any: ?anyerror = null,

                fn fail(t: *@This(), err: anyerror) void {
                    if (t.stop.swap(true, .acq_rel)) return;
                    t.err_mutex.lock();
                    defer t.err_mutex.unlock();
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
                            t.store.prefetchLinear(t.a, i + 1);
                            t.store.prefetchLinear(t.b, i + 1);
                        }

                        var out_tile = t.store.acquireTileMutLinear(t.out, i) catch |e| {
                            t.fail(e);
                            return;
                        };
                        defer t.store.releaseMut(out_tile.token);
                        const a_tile = t.store.acquireTileConstLinear(t.a, i) catch |e| {
                            t.fail(e);
                            return;
                        };
                        defer t.store.releaseConst(a_tile.token);
                        const b_tile = t.store.acquireTileConstLinear(t.b, i) catch |e| {
                            t.fail(e);
                            return;
                        };
                        defer t.store.releaseConst(b_tile.token);

                        const out_view = out_tile.bufferView();
                        const a_view = a_tile.bufferView();
                        const b_view = b_tile.bufferView();

                        const n: usize = if (out_view.layout.rank == 1) out_view.layout.shape[0] else (out_view.layout.shape[0] * out_view.layout.shape[1]);
                        switch (out_view.dtype) {
                            .f32 => elemwise.elemwiseBinaryF32(t.op, out_view.bytes, a_view.bytes, b_view.bytes, n) catch |e| {
                                t.fail(e);
                                return;
                            },
                            .f16 => elemwise.elemwiseBinaryF16(t.op, out_view.bytes, a_view.bytes, b_view.bytes, n) catch |e| {
                                t.fail(e);
                                return;
                            },
                            else => {
                                t.fail(BackendError.InvalidArgument);
                                return;
                            },
                        }
                    }
                }
            };

            var task: Task = .{ .store = store, .out_meta = out_meta, .op = s.op, .out = s.out, .a = s.a, .b = s.b };
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
        var out_tile = try store.acquireTileMutLinear(s.out, tile_index);
        defer store.releaseMut(out_tile.token);
        const a_tile = try store.acquireTileConstLinear(s.a, tile_index);
        defer store.releaseConst(a_tile.token);
        const b_tile = try store.acquireTileConstLinear(s.b, tile_index);
        defer store.releaseConst(b_tile.token);

        const out_view = out_tile.bufferView();
        const a_view = a_tile.bufferView();
        const b_view = b_tile.bufferView();
        const n: usize = exec_utils.elemCountFromTileView(out_view);
        switch (out_view.dtype) {
            .f32 => try elemwise.elemwiseBinaryF32(s.op, out_view.bytes, a_view.bytes, b_view.bytes, n),
            .f16 => try elemwise.elemwiseBinaryF16(s.op, out_view.bytes, a_view.bytes, b_view.bytes, n),
            else => return BackendError.InvalidArgument,
        }
    }
}

pub fn execBroadcastLastDimBinaryTiled(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepBroadcastLastDimBinaryTiled,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    const out_meta = try store.meta(s.out);
    if (out_meta.rank < 2) return BackendError.InvalidArgument;
    if (out_meta.rank > 8) return BackendError.InvalidArgument;

    var tile_total: usize = 1;
    var d: usize = 0;
    while (d < @as(usize, out_meta.rank)) : (d += 1) {
        tile_total *= out_meta.tile_counts[d];
    }
    const tile_bytes: usize = exec_utils.tileByteSize(out_meta);
    const min_total_bytes: usize = 256 * 1024;

    if (pool) |p| {
        if (exec_utils.shouldParallelTiles(thread_count, tile_total, tile_bytes, min_total_bytes)) {
            const Task = struct {
                store: tensor_store.TensorStore,
                out_meta: tensor_store.TensorMeta,
                op: types.ElemwiseBinaryOp,
                out: tensor_store.TensorId,
                a: tensor_store.TensorId,
                b: tensor_store.TensorId,

                stop: std.atomic.Value(bool) = .init(false),
                err_mutex: std.Thread.Mutex = .{},
                err_any: ?anyerror = null,

                fn fail(t: *@This(), err: anyerror) void {
                    if (t.stop.swap(true, .acq_rel)) return;
                    t.err_mutex.lock();
                    defer t.err_mutex.unlock();
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
                        var coords_buf: [8]usize = .{ 0, 0, 0, 0, 0, 0, 0, 0 };
                        const coords: []usize = coords_buf[0..@as(usize, t.out_meta.rank)];
                        tensor_store.decodeTileCoords(t.out_meta, i, coords) catch |e| {
                            t.fail(e);
                            return;
                        };
                        const ti_last: usize = coords[@as(usize, t.out_meta.rank) - 1];

                        var out_tile = t.store.acquireTileMutLinear(t.out, i) catch |e| {
                            t.fail(e);
                            return;
                        };
                        defer t.store.releaseMut(out_tile.token);
                        const a_tile = t.store.acquireTileConstLinear(t.a, i) catch |e| {
                            t.fail(e);
                            return;
                        };
                        defer t.store.releaseConst(a_tile.token);
                        const b_tile = t.store.acquireTileConst(t.b, ti_last, 0) catch |e| {
                            t.fail(e);
                            return;
                        };
                        defer t.store.releaseConst(b_tile.token);

                        const out_view = out_tile.bufferView();
                        const a_view = a_tile.bufferView();
                        const b_view = b_tile.bufferView();
                        const col_count: usize = out_view.layout.shape[@as(usize, out_view.layout.rank) - 1];
                        const elem_count: usize = exec_utils.elemCountFromTileView(out_view);
                        if (out_view.layout.rank < 2) {
                            t.fail(BackendError.InvalidArgument);
                            return;
                        }

                        switch (out_view.dtype) {
                            .f32 => switch (t.op) {
                                .add => elemwise.broadcastLastDimBinaryF32Packed(.add, out_view.bytes, a_view.bytes, b_view.bytes, elem_count, col_count) catch |e| {
                                    t.fail(e);
                                    return;
                                },
                                .sub => elemwise.broadcastLastDimBinaryF32Packed(.sub, out_view.bytes, a_view.bytes, b_view.bytes, elem_count, col_count) catch |e| {
                                    t.fail(e);
                                    return;
                                },
                                .mul => elemwise.broadcastLastDimBinaryF32Packed(.mul, out_view.bytes, a_view.bytes, b_view.bytes, elem_count, col_count) catch |e| {
                                    t.fail(e);
                                    return;
                                },
                                .div => elemwise.broadcastLastDimBinaryF32Packed(.div, out_view.bytes, a_view.bytes, b_view.bytes, elem_count, col_count) catch |e| {
                                    t.fail(e);
                                    return;
                                },
                            },
                            .f16 => switch (t.op) {
                                .add => elemwise.broadcastLastDimBinaryF16Packed(.add, out_view.bytes, a_view.bytes, b_view.bytes, elem_count, col_count) catch |e| {
                                    t.fail(e);
                                    return;
                                },
                                .sub => elemwise.broadcastLastDimBinaryF16Packed(.sub, out_view.bytes, a_view.bytes, b_view.bytes, elem_count, col_count) catch |e| {
                                    t.fail(e);
                                    return;
                                },
                                .mul => elemwise.broadcastLastDimBinaryF16Packed(.mul, out_view.bytes, a_view.bytes, b_view.bytes, elem_count, col_count) catch |e| {
                                    t.fail(e);
                                    return;
                                },
                                .div => elemwise.broadcastLastDimBinaryF16Packed(.div, out_view.bytes, a_view.bytes, b_view.bytes, elem_count, col_count) catch |e| {
                                    t.fail(e);
                                    return;
                                },
                            },
                            else => {
                                t.fail(BackendError.InvalidArgument);
                                return;
                            },
                        }
                    }
                }
            };

            var task: Task = .{ .store = store, .out_meta = out_meta, .op = s.op, .out = s.out, .a = s.a, .b = s.b };
            var grain: usize = if (tile_bytes == 0) 16 else @max(@as(usize, 1), min_total_bytes / tile_bytes);
            if (grain > tile_total) grain = tile_total;
            p.parallelForAny(@ptrCast(&task), tile_total, grain, Task.runTiles);
            if (task.err_any) |e| return @errorCast(e);
            return;
        }
    }

    // Sequential fallback.
    var tile_index: usize = 0;
    while (tile_index < tile_total) : (tile_index += 1) {
        var coords_buf: [8]usize = .{ 0, 0, 0, 0, 0, 0, 0, 0 };
        const coords: []usize = coords_buf[0..@as(usize, out_meta.rank)];
        try tensor_store.decodeTileCoords(out_meta, tile_index, coords);
        const ti_last: usize = coords[@as(usize, out_meta.rank) - 1];

        var out_tile = try store.acquireTileMutLinear(s.out, tile_index);
        defer store.releaseMut(out_tile.token);
        const a_tile = try store.acquireTileConstLinear(s.a, tile_index);
        defer store.releaseConst(a_tile.token);
        const b_tile = try store.acquireTileConst(s.b, ti_last, 0);
        defer store.releaseConst(b_tile.token);

        const out_view = out_tile.bufferView();
        const a_view = a_tile.bufferView();
        const b_view = b_tile.bufferView();
        if (out_view.layout.rank < 2) return BackendError.InvalidArgument;
        const col_count: usize = out_view.layout.shape[@as(usize, out_view.layout.rank) - 1];
        const elem_count: usize = exec_utils.elemCountFromTileView(out_view);

        switch (out_view.dtype) {
            .f32 => switch (s.op) {
                .add => try elemwise.broadcastLastDimBinaryF32Packed(.add, out_view.bytes, a_view.bytes, b_view.bytes, elem_count, col_count),
                .sub => try elemwise.broadcastLastDimBinaryF32Packed(.sub, out_view.bytes, a_view.bytes, b_view.bytes, elem_count, col_count),
                .mul => try elemwise.broadcastLastDimBinaryF32Packed(.mul, out_view.bytes, a_view.bytes, b_view.bytes, elem_count, col_count),
                .div => try elemwise.broadcastLastDimBinaryF32Packed(.div, out_view.bytes, a_view.bytes, b_view.bytes, elem_count, col_count),
            },
            .f16 => switch (s.op) {
                .add => try elemwise.broadcastLastDimBinaryF16Packed(.add, out_view.bytes, a_view.bytes, b_view.bytes, elem_count, col_count),
                .sub => try elemwise.broadcastLastDimBinaryF16Packed(.sub, out_view.bytes, a_view.bytes, b_view.bytes, elem_count, col_count),
                .mul => try elemwise.broadcastLastDimBinaryF16Packed(.mul, out_view.bytes, a_view.bytes, b_view.bytes, elem_count, col_count),
                .div => try elemwise.broadcastLastDimBinaryF16Packed(.div, out_view.bytes, a_view.bytes, b_view.bytes, elem_count, col_count),
            },
            else => return BackendError.InvalidArgument,
        }
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
                err_mutex: std.Thread.Mutex = .{},
                err_any: ?anyerror = null,

                fn fail(t: *@This(), err: anyerror) void {
                    if (t.stop.swap(true, .acq_rel)) return;
                    t.err_mutex.lock();
                    defer t.err_mutex.unlock();
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
