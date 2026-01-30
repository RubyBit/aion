const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");

const relu_k = @import("../kernels/relu.zig");
const gelu_k = @import("../kernels/gelu.zig");
const silu_k = @import("../kernels/silu.zig");
const sigmoid_k = @import("../kernels/sigmoid.zig");
const tanh_k = @import("../kernels/tanh.zig");

const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const exec_utils = @import("utils.zig");
const executable = @import("../../../runtime/executable.zig");

const BackendError = types.BackendError;
const UnaryOp = types.UnaryOp;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

pub fn execUnaryTiled(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepUnaryTiled,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    const out_meta = try store.meta(s.out);
    const tile_total: usize = out_meta.tile_counts[0] * out_meta.tile_counts[1];
    const tile_bytes: usize = exec_utils.tileByteSize(out_meta);
    const min_total_bytes: usize = 256 * 1024;

    if (pool) |p| {
        if (exec_utils.shouldParallelTiles(thread_count, tile_total, tile_bytes, min_total_bytes)) {
            const Task = struct {
                store: tensor_store.TensorStore,
                out_meta: tensor_store.TensorMeta,
                op: UnaryOp,
                out: tensor_store.TensorId,
                a: tensor_store.TensorId,

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

                    const tc1: usize = t.out_meta.tile_counts[1];
                    var i: usize = start;
                    while (i < end) : (i += 1) {
                        if (t.stop.load(.acquire)) return;
                        const ti0: usize = i / tc1;
                        const ti1: usize = i - ti0 * tc1;

                        if (i + 1 < end) {
                            const nti0: usize = (i + 1) / tc1;
                            const nti1: usize = (i + 1) - nti0 * tc1;
                            t.store.prefetch(t.a, nti0, nti1);
                        }

                        var out_tile = t.store.acquireTileMut(t.out, ti0, ti1) catch |e| {
                            t.fail(e);
                            return;
                        };
                        defer t.store.releaseMut(out_tile.token);
                        const a_tile = t.store.acquireTileConst(t.a, ti0, ti1) catch |e| {
                            t.fail(e);
                            return;
                        };
                        defer t.store.releaseConst(a_tile.token);

                        const out_view = out_tile.bufferView();
                        const a_view = a_tile.bufferView();
                        const n: usize = if (out_view.layout.rank == 1) out_view.layout.shape[0] else (out_view.layout.shape[0] * out_view.layout.shape[1]);

                        switch (out_view.dtype) {
                            .f32 => dispatchF32(t.op, out_view.bytes, a_view.bytes, n) catch |e| {
                                t.fail(e);
                                return;
                            },
                            .f16 => dispatchF16(t.op, out_view.bytes, a_view.bytes, n) catch |e| {
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

            var task: Task = .{ .store = store, .out_meta = out_meta, .op = s.op, .out = s.out, .a = s.a };
            var grain: usize = if (tile_bytes == 0) 32 else @max(@as(usize, 1), min_total_bytes / tile_bytes);
            if (grain > tile_total) grain = tile_total;
            p.parallelForAny(@ptrCast(&task), tile_total, grain, Task.runTiles);
            if (task.err_any) |e| return @errorCast(e);
            return;
        }
    }

    // Sequential fallback.
    var ti0: usize = 0;
    while (ti0 < out_meta.tile_counts[0]) : (ti0 += 1) {
        var ti1: usize = 0;
        while (ti1 < out_meta.tile_counts[1]) : (ti1 += 1) {
            var out_tile = try store.acquireTileMut(s.out, ti0, ti1);
            defer store.releaseMut(out_tile.token);
            const a_tile = try store.acquireTileConst(s.a, ti0, ti1);
            defer store.releaseConst(a_tile.token);

            const out_view = out_tile.bufferView();
            const a_view = a_tile.bufferView();
            const n: usize = exec_utils.elemCountFromTileView(out_view);

            switch (out_view.dtype) {
                .f32 => try dispatchF32(s.op, out_view.bytes, a_view.bytes, n),
                .f16 => try dispatchF16(s.op, out_view.bytes, a_view.bytes, n),
                else => return BackendError.InvalidArgument,
            }
        }
    }
}

fn dispatchF32(op: UnaryOp, out_bytes: []u8, a_bytes: []const u8, n: usize) BackendError!void {
    return switch (op) {
        .relu => relu_k.reluF32(out_bytes, a_bytes, n),
        .gelu => gelu_k.geluF32(out_bytes, a_bytes, n),
        .silu => silu_k.siluF32(out_bytes, a_bytes, n),
        .sigmoid => sigmoid_k.sigmoidF32(out_bytes, a_bytes, n),
        .tanh => tanh_k.tanhF32(out_bytes, a_bytes, n),
    };
}

fn dispatchF16(op: UnaryOp, out_bytes: []u8, a_bytes: []const u8, n: usize) BackendError!void {
    return switch (op) {
        .relu => relu_k.reluF16(out_bytes, a_bytes, n),
        .gelu => gelu_k.geluF16(out_bytes, a_bytes, n),
        .silu => silu_k.siluF16(out_bytes, a_bytes, n),
        .sigmoid => sigmoid_k.sigmoidF16(out_bytes, a_bytes, n),
        .tanh => tanh_k.tanhF16(out_bytes, a_bytes, n),
    };
}
