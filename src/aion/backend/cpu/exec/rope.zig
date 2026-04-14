const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");
const exec_utils = @import("utils.zig");
const rope_k = @import("../kernels/rope.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

pub fn execRoPE1DTiled(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepRoPE1DTiled,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    const out_meta: tensor_store.TensorMeta = try store.meta(s.out);
    const x_meta: tensor_store.TensorMeta = try store.meta(s.x);
    const pos_meta: tensor_store.TensorMeta = try store.meta(s.positions);

    if (out_meta.rank != 4 or x_meta.rank != 4 or pos_meta.rank != 2) return BackendError.InvalidArgument;
    if (out_meta.dtype != x_meta.dtype) return BackendError.InvalidArgument;
    if (!(out_meta.dtype == .f16 or out_meta.dtype == .f32)) return BackendError.InvalidArgument;
    if (pos_meta.dtype != .i32) return BackendError.InvalidArgument;

    var d: usize = 0;
    while (d < 4) : (d += 1) {
        if (out_meta.shape[d] != x_meta.shape[d]) return BackendError.InvalidArgument;
        if (out_meta.tile_shape[d] != x_meta.tile_shape[d]) return BackendError.InvalidArgument;
        if (out_meta.tile_counts[d] != x_meta.tile_counts[d]) return BackendError.InvalidArgument;
    }

    if (pos_meta.shape[0] != out_meta.shape[0] or pos_meta.shape[1] != out_meta.shape[1]) return BackendError.InvalidArgument;
    if (pos_meta.tile_shape[0] != out_meta.tile_shape[0] or pos_meta.tile_shape[1] != out_meta.tile_shape[1]) return BackendError.InvalidArgument;
    if (pos_meta.tile_counts[0] != out_meta.tile_counts[0] or pos_meta.tile_counts[1] != out_meta.tile_counts[1]) return BackendError.InvalidArgument;

    if (!(s.base_frequency > 0.0) or !std.math.isFinite(s.base_frequency)) return BackendError.InvalidArgument;
    if (!(s.scale_factor > 0.0) or !std.math.isFinite(s.scale_factor)) return BackendError.InvalidArgument;
    if (!std.math.isFinite(s.rope_proportion) or s.rope_proportion < 0.0 or s.rope_proportion > 1.0) return BackendError.InvalidArgument;

    const head_dim: usize = out_meta.shape[3];
    if (out_meta.tile_counts[3] != 1 or x_meta.tile_counts[3] != 1) return BackendError.InvalidArgument;
    if (out_meta.tile_shape[3] != head_dim or x_meta.tile_shape[3] != head_dim) return BackendError.InvalidArgument;

    const pairs_total: usize = head_dim / 2;
    const rope_pairs_f: f32 = @floor(s.rope_proportion * @as(f32, @floatFromInt(pairs_total)));
    const rope_pairs_i: i64 = @intFromFloat(rope_pairs_f);
    if (rope_pairs_i < 0) return BackendError.InvalidArgument;
    var rope_pairs: usize = @intCast(rope_pairs_i);
    if (rope_pairs > pairs_total) rope_pairs = pairs_total;

    const head_dim_f32: f32 = @floatFromInt(head_dim);
    const freq_step: f32 = @floatCast(std.math.pow(f64, @as(f64, s.base_frequency), @as(f64, -2.0 / head_dim_f32)));

    const rank: usize = 4;
    var tile_total: usize = 1;
    d = 0;
    while (d < rank) : (d += 1) {
        tile_total = std.math.mul(usize, tile_total, out_meta.tile_counts[d]) catch return BackendError.InvalidArgument;
    }

    const Runner = struct {
        store: tensor_store.TensorStore,
        out_meta: tensor_store.TensorMeta,
        out: tensor_store.TensorId,
        x: tensor_store.TensorId,
        positions: tensor_store.TensorId,
        pairs_total: usize,
        rope_pairs: usize,
        freq_step: f32,
        scale_factor: f32,

        fn runRange(self: *@This(), start: usize, end: usize) ExecuteProgramError!void {
            var coords: [8]usize = .{0} ** 8;

            var i: usize = start;
            while (i < end) : (i += 1) {
                var out_tile = try self.store.acquireTileMutLinear(self.out, i);
                defer self.store.releaseMut(out_tile.token);
                const x_tile = try self.store.acquireTileConstLinear(self.x, i);
                defer self.store.releaseConst(x_tile.token);

                try tensor_store.decodeTileCoords(self.out_meta, i, coords[0..4]);

                const pos_tile = try self.store.acquireTileConst(self.positions, coords[0], coords[1]);
                defer self.store.releaseConst(pos_tile.token);

                const out_view = out_tile.bufferView();
                const x_view = x_tile.bufferView();
                const pos_view = pos_tile.bufferView();

                if (out_view.layout.rank != 4 or x_view.layout.rank != 4 or pos_view.layout.rank != 2) return BackendError.InvalidArgument;
                if (out_view.layout.shape[3] != self.out_meta.shape[3]) return BackendError.InvalidArgument;

                switch (out_view.dtype) {
                    .f32 => try rope_k.runTileF32(out_view, x_view, pos_view, self.pairs_total, self.rope_pairs, self.freq_step, self.scale_factor),
                    .f16 => try rope_k.runTileF16(out_view, x_view, pos_view, self.pairs_total, self.rope_pairs, self.freq_step, self.scale_factor),
                    else => return BackendError.InvalidArgument,
                }
            }
        }
    };

    var runner: Runner = .{
        .store = store,
        .out_meta = out_meta,
        .out = s.out,
        .x = s.x,
        .positions = s.positions,
        .pairs_total = pairs_total,
        .rope_pairs = rope_pairs,
        .freq_step = freq_step,
        .scale_factor = s.scale_factor,
    };

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
