// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");
const exec_utils = @import("utils.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;
const DType = types.DType;

/// Per-tile elementwise cast between scalar dtypes.
///
/// v1 supports f16<->f32 and same-dtype (no-op memcpy). Adding more pairs is a
/// matter of extending the dispatch below; the kernel body is always a straight
/// per-tile loop since Cast is shape- and layout-preserving.
pub fn execCastTiled(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepCastTiled,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    const out_meta = try store.meta(s.out);
    const in_meta = try store.meta(s.x);

    if (out_meta.dtype != s.to_dtype) return BackendError.InvalidArgument;
    if (out_meta.rank != in_meta.rank) return BackendError.InvalidArgument;

    var tile_total: usize = 1;
    var d: usize = 0;
    while (d < @as(usize, out_meta.rank)) : (d += 1) {
        tile_total *= out_meta.tile_counts[d];
    }

    const Runner = struct {
        store: tensor_store.TensorStore,
        out: tensor_store.TensorId,
        x: tensor_store.TensorId,
        from: DType,
        to: DType,

        fn runRange(self: *@This(), start: usize, end: usize) ExecuteProgramError!void {
            var i: usize = start;
            while (i < end) : (i += 1) {
                var out_tile = try self.store.acquireTileMutLinear(self.out, i);
                defer self.store.releaseMut(out_tile.token);
                const in_tile = try self.store.acquireTileConstLinear(self.x, i);
                defer self.store.releaseConst(in_tile.token);

                const out_bytes: []u8 = out_tile.bufferView().bytes;
                const in_bytes: []const u8 = in_tile.bufferView().bytes;

                try castBytes(self.from, self.to, in_bytes, out_bytes);
            }
        }
    };

    var runner: Runner = .{ .store = store, .out = s.out, .x = s.x, .from = in_meta.dtype, .to = s.to_dtype };

    const tile_bytes: usize = exec_utils.tileByteSize(out_meta);
    const min_total_bytes: usize = 256 * 1024;

    if (pool) |p| {
        if (exec_utils.shouldParallelTiles(thread_count, tile_total, tile_bytes, min_total_bytes)) {
            const Task = struct {
                runner: *Runner,
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
                    t.runner.runRange(start, end) catch |e| {
                        t.fail(e);
                        return;
                    };
                }
            };

            var task: Task = .{ .runner = &runner };
            var grain: usize = if (tile_bytes == 0) 16 else @max(@as(usize, 1), min_total_bytes / tile_bytes);
            if (grain > tile_total) grain = tile_total;

            p.parallelForAny(@ptrCast(&task), tile_total, grain, Task.runTiles);
            if (task.err_any) |e| return @errorCast(e);
            return;
        }
    }

    try runner.runRange(0, tile_total);
}

fn castBytes(from: DType, to: DType, in_bytes: []const u8, out_bytes: []u8) BackendError!void {
    if (from == to) {
        if (in_bytes.len != out_bytes.len) return BackendError.InvalidArgument;
        @memcpy(out_bytes, in_bytes);
        return;
    }

    if (from == .f32 and to == .f16) {
        if ((in_bytes.len % @sizeOf(f32)) != 0) return BackendError.InvalidArgument;
        const n: usize = in_bytes.len / @sizeOf(f32);
        if (out_bytes.len < n * @sizeOf(f16)) return BackendError.InvalidArgument;
        const src: [*]align(1) const f32 = @ptrCast(in_bytes.ptr);
        const dst: [*]align(1) f16 = @ptrCast(out_bytes.ptr);
        var i: usize = 0;
        while (i < n) : (i += 1) {
            dst[i] = @floatCast(src[i]);
        }
        return;
    }

    if (from == .f16 and to == .f32) {
        if ((in_bytes.len % @sizeOf(f16)) != 0) return BackendError.InvalidArgument;
        const n: usize = in_bytes.len / @sizeOf(f16);
        if (out_bytes.len < n * @sizeOf(f32)) return BackendError.InvalidArgument;
        const src: [*]align(1) const f16 = @ptrCast(in_bytes.ptr);
        const dst: [*]align(1) f32 = @ptrCast(out_bytes.ptr);
        var i: usize = 0;
        while (i < n) : (i += 1) {
            dst[i] = @floatCast(src[i]);
        }
        return;
    }

    return BackendError.Unsupported;
}
