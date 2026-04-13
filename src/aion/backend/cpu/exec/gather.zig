const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");
const exec_utils = @import("utils.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

fn copyRowVectorized(dst: []u8, src: []const u8) void {
    std.debug.assert(dst.len == src.len);

    const Vec64 = @Vector(64, u8);
    const Vec32 = @Vector(32, u8);
    const Vec16 = @Vector(16, u8);

    const n: usize = dst.len;
    var i: usize = 0;

    while (i + 64 <= n) : (i += 64) {
        const v: Vec64 = @as(*align(1) const Vec64, @ptrCast(src.ptr + i)).*;
        @as(*align(1) Vec64, @ptrCast(dst.ptr + i)).* = v;
    }
    while (i + 32 <= n) : (i += 32) {
        const v: Vec32 = @as(*align(1) const Vec32, @ptrCast(src.ptr + i)).*;
        @as(*align(1) Vec32, @ptrCast(dst.ptr + i)).* = v;
    }
    while (i + 16 <= n) : (i += 16) {
        const v: Vec16 = @as(*align(1) const Vec16, @ptrCast(src.ptr + i)).*;
        @as(*align(1) Vec16, @ptrCast(dst.ptr + i)).* = v;
    }
    while (i < n) : (i += 1) {
        dst[i] = src[i];
    }
}

pub fn execGatherRowsTiled(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepGatherRowsTiled,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    const out_meta: tensor_store.TensorMeta = try store.meta(s.out);
    const table_meta: tensor_store.TensorMeta = try store.meta(s.table);
    const indices_meta: tensor_store.TensorMeta = try store.meta(s.indices);

    // The compiler is responsible for full validation; keep runtime checks minimal.
    if (out_meta.rank != 3 or table_meta.rank != 2 or indices_meta.rank != 2) return BackendError.InvalidArgument;
    if (indices_meta.dtype != .i32) return BackendError.InvalidArgument;
    if (out_meta.dtype != table_meta.dtype) return BackendError.InvalidArgument;
    if (!(table_meta.dtype == .f16 or table_meta.dtype == .f32)) return BackendError.InvalidArgument;

    const elem_bytes: usize = switch (table_meta.dtype) {
        .f16 => 2,
        .f32 => 4,
        else => return BackendError.InvalidArgument,
    };

    const b_total: usize = indices_meta.shape[0];
    const l_total: usize = indices_meta.shape[1];
    const v_total: usize = table_meta.shape[0];

    // Acquire indices once (expected to be single-tile in v0).
    var idx_tile: tensor_store.TileRefConst = try store.acquireTileConst(s.indices, 0, 0);
    defer store.releaseConst(idx_tile.token);
    const idx_view = idx_tile.bufferView();
    if (idx_view.layout.rank != 2) return BackendError.InvalidArgument;

    if ((idx_view.bytes.len % @sizeOf(i32)) != 0) return BackendError.InvalidArgument;
    const idx_ptr: [*]align(1) const i32 = @ptrCast(idx_view.bytes.ptr);
    const idx_vals: []align(1) const i32 = idx_ptr[0 .. idx_view.bytes.len / @sizeOf(i32)];
    if (idx_vals.len < b_total * l_total) return BackendError.InvalidArgument;

    // Total output tiles (linear iteration).
    const rank: usize = 3;
    var tile_total: usize = 1;
    var d: usize = 0;
    while (d < rank) : (d += 1) {
        tile_total = std.math.mul(usize, tile_total, out_meta.tile_counts[d]) catch return BackendError.InvalidArgument;
    }

    const Runner = struct {
        store: tensor_store.TensorStore,
        out_meta: tensor_store.TensorMeta,
        table_meta: tensor_store.TensorMeta,
        out: tensor_store.TensorId,
        table: tensor_store.TensorId,
        b_total: usize,
        l_total: usize,
        v_total: usize,
        elem_bytes: usize,
        idx_vals: []align(1) const i32,

        fn runRange(self: *@This(), start: usize, end: usize) ExecuteProgramError!void {
            var out_tile_coords: [8]usize = .{0} ** 8;

            var table_cached: bool = false;
            var table_cached_ti0: usize = 0;
            var table_tile: tensor_store.TileRefConst = undefined;
            defer if (table_cached) self.store.releaseConst(table_tile.token);

            var tile_index: usize = start;
            while (tile_index < end) : (tile_index += 1) {
                var out_tile: tensor_store.TileRefMut = try self.store.acquireTileMutLinear(self.out, tile_index);
                defer self.store.releaseMut(out_tile.token);
                const out_view = out_tile.bufferView();
                if (out_view.layout.rank != 3) return BackendError.InvalidArgument;

                try tensor_store.decodeTileCoords(self.out_meta, tile_index, out_tile_coords[0..rank]);

                const base_b: usize = out_tile_coords[0] * self.out_meta.tile_shape[0];
                const base_l: usize = out_tile_coords[1] * self.out_meta.tile_shape[1];

                const tb: usize = out_view.layout.shape[0];
                const tl: usize = out_view.layout.shape[1];
                const td: usize = out_view.layout.shape[2];
                if (td == 0) return BackendError.InvalidArgument;
                if (self.out_meta.shape[2] != td) return BackendError.InvalidArgument;

                const bytes_per_row: usize = std.math.mul(usize, td, self.elem_bytes) catch return BackendError.InvalidArgument;
                const need_bytes: usize = std.math.mul(usize, std.math.mul(usize, tb, tl) catch return BackendError.InvalidArgument, bytes_per_row) catch return BackendError.InvalidArgument;
                if (need_bytes > out_view.bytes.len) return BackendError.InvalidArgument;

                var lb: usize = 0;
                while (lb < tb) : (lb += 1) {
                    const b: usize = base_b + lb;
                    if (b >= self.b_total) return BackendError.InvalidArgument;

                    var ll: usize = 0;
                    while (ll < tl) : (ll += 1) {
                        const l: usize = base_l + ll;
                        if (l >= self.l_total) return BackendError.InvalidArgument;

                        const idx_i32: i32 = self.idx_vals[b * self.l_total + l];
                        if (idx_i32 < 0) return BackendError.InvalidArgument;
                        const row: usize = @intCast(idx_i32);
                        if (row >= self.v_total) return BackendError.InvalidArgument;

                        // Prefetch the next row's tile while copying the current row.
                        // This helps hide cache / storage latency for random-token gathers.
                        var next_row_opt: ?usize = null;
                        if (ll + 1 < tl) {
                            const nb: usize = b;
                            const nl: usize = l + 1;
                            if (nl < self.l_total) {
                                const next_i32: i32 = self.idx_vals[nb * self.l_total + nl];
                                if (next_i32 >= 0) {
                                    const next_row: usize = @intCast(next_i32);
                                    if (next_row < self.v_total) {
                                        next_row_opt = next_row;
                                        self.store.prefetch(self.table, next_row / self.table_meta.tile_shape[0], 0);
                                    }
                                }
                            }
                        } else if (lb + 1 < tb) {
                            const nb: usize = b + 1;
                            const nl: usize = base_l;
                            if (nb < self.b_total and nl < self.l_total) {
                                const next_i32: i32 = self.idx_vals[nb * self.l_total + nl];
                                if (next_i32 >= 0) {
                                    const next_row: usize = @intCast(next_i32);
                                    if (next_row < self.v_total) {
                                        next_row_opt = next_row;
                                        self.store.prefetch(self.table, next_row / self.table_meta.tile_shape[0], 0);
                                    }
                                }
                            }
                        }

                        const ti0: usize = row / self.table_meta.tile_shape[0];
                        const local_r: usize = row - ti0 * self.table_meta.tile_shape[0];

                        if (!table_cached or table_cached_ti0 != ti0) {
                            if (table_cached) self.store.releaseConst(table_tile.token);
                            table_tile = try self.store.acquireTileConst(self.table, ti0, 0);
                            table_cached = true;
                            table_cached_ti0 = ti0;
                        }

                        const table_view = table_tile.bufferView();
                        if (table_view.layout.rank != 2) return BackendError.InvalidArgument;
                        const rows_in_tile: usize = table_view.layout.shape[0];
                        const d_in_tile: usize = table_view.layout.shape[1];
                        if (local_r >= rows_in_tile) return BackendError.InvalidArgument;
                        if (d_in_tile != td) return BackendError.InvalidArgument;

                        if (next_row_opt) |next_row| {
                            const next_ti0: usize = next_row / self.table_meta.tile_shape[0];
                            if (next_ti0 == table_cached_ti0) {
                                const next_local_r: usize = next_row - next_ti0 * self.table_meta.tile_shape[0];
                                if (next_local_r < rows_in_tile) {
                                    const pf_off: usize = (next_local_r * d_in_tile) * self.elem_bytes;
                                    if (pf_off < table_view.bytes.len) {
                                        @prefetch(table_view.bytes[pf_off..].ptr, .{ .rw = .read, .locality = 3, .cache = .data });
                                    }
                                }
                            }
                        }

                        const src_off: usize = (local_r * d_in_tile) * self.elem_bytes;
                        const dst_off: usize = ((lb * tl + ll) * td) * self.elem_bytes;

                        copyRowVectorized(
                            out_view.bytes[dst_off .. dst_off + bytes_per_row],
                            table_view.bytes[src_off .. src_off + bytes_per_row],
                        );
                    }
                }
            }
        }
    };

    var runner: Runner = .{
        .store = store,
        .out_meta = out_meta,
        .table_meta = table_meta,
        .out = s.out,
        .table = s.table,
        .b_total = b_total,
        .l_total = l_total,
        .v_total = v_total,
        .elem_bytes = elem_bytes,
        .idx_vals = idx_vals,
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

    // Sequential fallback.
    try runner.runRange(0, tile_total);
}
