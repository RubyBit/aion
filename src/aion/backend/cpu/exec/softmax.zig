// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");

const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");

const exec_utils = @import("utils.zig");
const simd = @import("../kernels/simd.zig");
const softmax_kernels = @import("../kernels/softmax.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

const MAX_RANK: usize = 8;

fn normalizeAxis(axis: i32, rank: usize) ExecuteProgramError!usize {
    if (rank == 0) return BackendError.InvalidArgument;
    const r_i32: i32 = @intCast(rank);
    var ax: i32 = axis;
    if (ax < 0) ax += r_i32;
    if (ax < 0 or ax >= r_i32) return BackendError.InvalidArgument;
    return @intCast(ax);
}

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

pub fn execSoftmaxTiled(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    scratch_f32: []f32,
    s: executable.StepSoftmaxTiled,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    const out_meta = try store.meta(s.out);
    if (out_meta.dtype != .f32) return BackendError.InvalidArgument;
    const rank: usize = @as(usize, out_meta.rank);
    if (rank == 0 or rank > MAX_RANK) return BackendError.InvalidArgument;
    const axis: usize = try normalizeAxis(s.axis, rank);
    if (out_meta.rank == 1) {
        return softmaxRank1(pool, thread_count, scratch_f32, store, out_meta, s.out, s.a);
    }

    if (axis + 1 == rank) {
        if (out_meta.rank > 2) {
            return softmaxAxisLastND(pool, thread_count, store, out_meta, s.out, s.a, axis);
        }

        // Rank-2 axis-last uses the 2D path below.
    } else {
        return softmaxAxisStridedND(pool, thread_count, store, out_meta, s.out, s.a, axis);
    }

    // Work units: tile rows.
    const ti0_total: usize = out_meta.tile_counts[0];
    const tile_bytes: usize = exec_utils.tileByteSize(out_meta);
    const min_total_bytes: usize = 256 * 1024;

    if (pool) |p| {
        const total_bytes: usize = tile_bytes * (out_meta.tile_counts[0] * out_meta.tile_counts[1]);
        if (thread_count > 1 and ti0_total >= 2 and total_bytes >= min_total_bytes) {
            const Task = struct {
                store: tensor_store.TensorStore,
                out_meta: tensor_store.TensorMeta,
                out: tensor_store.TensorId,
                a: tensor_store.TensorId,

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
                        softmaxTi0(t.store, t.out_meta, t.out, t.a, ti0) catch |e| {
                            t.fail(e);
                            return;
                        };
                    }
                }
            };

            var task: Task = .{ .store = store, .out_meta = out_meta, .out = s.out, .a = s.a };
            // One work item per tile-row.
            p.parallelForAny(@ptrCast(&task), ti0_total, 1, Task.runTi0);
            if (task.err_any) |e| return @errorCast(e);
            return;
        }
    }

    // Sequential fallback.
    var ti0: usize = 0;
    while (ti0 < ti0_total) : (ti0 += 1) {
        try softmaxTi0(store, out_meta, s.out, s.a, ti0);
    }
}

fn computeRowOffsets(
    row_offsets: []usize,
    row_coords: []usize,
    row_count: usize,
    row_dim_count: usize,
    row_dims: []const usize,
    row_strides: []const usize,
    layout_strides_bytes: []const isize,
) ExecuteProgramError!void {
    if (row_offsets.len < row_count or row_coords.len < row_dim_count) return BackendError.InvalidArgument;
    var r: usize = 0;
    while (r < row_count) : (r += 1) {
        var rem_row: usize = r;
        var base_off: usize = 0;
        var rdi: usize = 0;
        while (rdi < row_dim_count) : (rdi += 1) {
            const stride_row: usize = row_strides[rdi];
            const v: usize = rem_row / stride_row;
            row_coords[rdi] = v;
            rem_row -= v * stride_row;

            const dim: usize = row_dims[rdi];
            const stride_bytes_isize: isize = layout_strides_bytes[dim];
            if (stride_bytes_isize < 0) return BackendError.InvalidArgument;
            const stride_bytes: usize = @intCast(stride_bytes_isize);
            base_off = std.math.add(usize, base_off, v * stride_bytes) catch return BackendError.InvalidArgument;
        }
        row_offsets[r] = base_off;
    }
}

fn softmaxAxisStridedND(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    store: tensor_store.TensorStore,
    out_meta: tensor_store.TensorMeta,
    out: tensor_store.TensorId,
    a: tensor_store.TensorId,
    axis: usize,
) ExecuteProgramError!void {
    const rank: usize = @as(usize, out_meta.rank);
    if (rank < 2 or axis >= rank) return BackendError.InvalidArgument;

    var row_dims: [MAX_RANK]usize = undefined;
    var row_counts: [MAX_RANK]usize = undefined;
    var row_strides: [MAX_RANK]usize = undefined;
    var row_dim_count: usize = 0;

    var d: usize = 0;
    while (d < rank) : (d += 1) {
        if (d == axis) continue;
        row_dims[row_dim_count] = d;
        row_counts[row_dim_count] = out_meta.tile_counts[d];
        row_dim_count += 1;
    }

    var stride: usize = 1;
    var i: usize = row_dim_count;
    while (i > 0) : (i -= 1) {
        const idx: usize = i - 1;
        row_strides[idx] = stride;
        stride = std.math.mul(usize, stride, row_counts[idx]) catch return BackendError.InvalidArgument;
    }
    const row_tile_total: usize = stride;

    const axis_tiles: usize = out_meta.tile_counts[axis];
    if (axis_tiles == 0) return BackendError.InvalidArgument;

    const tile_bytes: usize = exec_utils.tileByteSize(out_meta);
    const min_total_bytes: usize = 256 * 1024;
    var tile_total: usize = 1;
    d = 0;
    while (d < rank) : (d += 1) {
        tile_total *= out_meta.tile_counts[d];
    }

    if (pool) |p| {
        if (exec_utils.shouldParallelTiles(thread_count, tile_total, tile_bytes, min_total_bytes) and row_tile_total >= 2) {
            const Task = struct {
                store: tensor_store.TensorStore,
                out_meta: tensor_store.TensorMeta,
                out: tensor_store.TensorId,
                a: tensor_store.TensorId,
                axis: usize,
                row_dim_count: usize,
                row_dims: [MAX_RANK]usize,
                row_counts: [MAX_RANK]usize,
                row_strides: [MAX_RANK]usize,
                axis_tiles: usize,

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

                    var coords: [MAX_RANK]usize = .{0} ** MAX_RANK;
                    var row_sizes: [MAX_RANK]usize = .{0} ** MAX_RANK;
                    var row_local_strides: [MAX_RANK]usize = .{0} ** MAX_RANK;
                    var row_coords: [MAX_RANK]usize = .{0} ** MAX_RANK;
                    var row_offsets_in: [256]usize = .{0} ** 256;
                    var row_offsets_out: [256]usize = .{0} ** 256;

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
                            row_sizes[rdi] = dim_len;
                            rows = std.math.mul(usize, rows, dim_len) catch |e| {
                                t.fail(e);
                                return;
                            };
                        }
                        if (rows == 0 or rows > 256) {
                            t.fail(BackendError.InvalidArgument);
                            return;
                        }

                        exec_utils.computePackedStrides(row_sizes[0..t.row_dim_count], row_local_strides[0..t.row_dim_count]) catch |e| {
                            t.fail(e);
                            return;
                        };

                        var max_buf: [256]f32 = undefined;
                        var sum_buf: [256]f32 = undefined;
                        @memset(max_buf[0..rows], -std.math.inf(f32));
                        @memset(sum_buf[0..rows], 0.0);

                        var ti_axis: usize = 0;
                        while (ti_axis < t.axis_tiles) : (ti_axis += 1) {
                            if (t.stop.load(.acquire)) return;
                            coords[t.axis] = ti_axis;
                            const tile_index: usize = tensor_store.encodeTileIndex(t.out_meta, coords[0..t.out_meta.tile_counts.len]) catch |e| {
                                t.fail(e);
                                return;
                            };
                            const in_tile = t.store.acquireTileConstLinear(t.a, tile_index) catch |e| {
                                t.fail(e);
                                return;
                            };
                            defer t.store.releaseConst(in_tile.token);
                            const in_view = in_tile.bufferView();

                            const axis_len: usize = tileDim(t.out_meta.shape, t.out_meta.tile_shape, ti_axis, t.axis) catch |e| {
                                t.fail(e);
                                return;
                            };
                            const stride_axis_isize: isize = in_view.layout.strides_bytes[t.axis];
                            if (stride_axis_isize < 0) {
                                t.fail(BackendError.InvalidArgument);
                                return;
                            }
                            const stride_axis: usize = @intCast(stride_axis_isize);

                            computeRowOffsets(
                                row_offsets_in[0..rows],
                                row_coords[0..t.row_dim_count],
                                rows,
                                t.row_dim_count,
                                t.row_dims[0..t.row_dim_count],
                                row_local_strides[0..t.row_dim_count],
                                in_view.layout.strides_bytes,
                            ) catch |e| {
                                t.fail(e);
                                return;
                            };

                            softmax_kernels.updateMaxStridedRowsF32(
                                max_buf[0..rows],
                                in_view.bytes,
                                axis_len,
                                stride_axis,
                                row_offsets_in[0..rows],
                            ) catch |e| {
                                t.fail(e);
                                return;
                            };
                        }

                        ti_axis = 0;
                        while (ti_axis < t.axis_tiles) : (ti_axis += 1) {
                            if (t.stop.load(.acquire)) return;
                            coords[t.axis] = ti_axis;
                            const tile_index: usize = tensor_store.encodeTileIndex(t.out_meta, coords[0..t.out_meta.tile_counts.len]) catch |e| {
                                t.fail(e);
                                return;
                            };
                            const out_tile = t.store.acquireTileMutLinear(t.out, tile_index) catch |e| {
                                t.fail(e);
                                return;
                            };
                            defer t.store.releaseMut(out_tile.token);
                            const in_tile = t.store.acquireTileConstLinear(t.a, tile_index) catch |e| {
                                t.fail(e);
                                return;
                            };
                            defer t.store.releaseConst(in_tile.token);

                            const out_view = out_tile.bufferView();
                            const in_view = in_tile.bufferView();

                            const axis_len: usize = tileDim(t.out_meta.shape, t.out_meta.tile_shape, ti_axis, t.axis) catch |e| {
                                t.fail(e);
                                return;
                            };
                            const stride_axis_isize: isize = in_view.layout.strides_bytes[t.axis];
                            if (stride_axis_isize < 0) {
                                t.fail(BackendError.InvalidArgument);
                                return;
                            }
                            const stride_axis: usize = @intCast(stride_axis_isize);
                            const out_stride_axis_isize: isize = out_view.layout.strides_bytes[t.axis];
                            if (out_stride_axis_isize < 0) {
                                t.fail(BackendError.InvalidArgument);
                                return;
                            }
                            const out_stride_axis: usize = @intCast(out_stride_axis_isize);

                            computeRowOffsets(
                                row_offsets_in[0..rows],
                                row_coords[0..t.row_dim_count],
                                rows,
                                t.row_dim_count,
                                t.row_dims[0..t.row_dim_count],
                                row_local_strides[0..t.row_dim_count],
                                in_view.layout.strides_bytes,
                            ) catch |e| {
                                t.fail(e);
                                return;
                            };

                            computeRowOffsets(
                                row_offsets_out[0..rows],
                                row_coords[0..t.row_dim_count],
                                rows,
                                t.row_dim_count,
                                t.row_dims[0..t.row_dim_count],
                                row_local_strides[0..t.row_dim_count],
                                out_view.layout.strides_bytes,
                            ) catch |e| {
                                t.fail(e);
                                return;
                            };

                            softmax_kernels.expSumStoreStridedRowsF32(
                                sum_buf[0..rows],
                                out_view.bytes,
                                in_view.bytes,
                                axis_len,
                                stride_axis,
                                out_stride_axis,
                                row_offsets_in[0..rows],
                                row_offsets_out[0..rows],
                                max_buf[0..rows],
                            ) catch |e| {
                                t.fail(e);
                                return;
                            };
                        }

                        var r_check: usize = 0;
                        while (r_check < rows) : (r_check += 1) {
                            if (!(sum_buf[r_check] > 0.0) or !std.math.isFinite(sum_buf[r_check])) {
                                t.fail(BackendError.InvalidArgument);
                                return;
                            }
                        }

                        ti_axis = 0;
                        while (ti_axis < t.axis_tiles) : (ti_axis += 1) {
                            if (t.stop.load(.acquire)) return;
                            coords[t.axis] = ti_axis;
                            const tile_index: usize = tensor_store.encodeTileIndex(t.out_meta, coords[0..t.out_meta.tile_counts.len]) catch |e| {
                                t.fail(e);
                                return;
                            };
                            const out_tile = t.store.acquireTileMutLinear(t.out, tile_index) catch |e| {
                                t.fail(e);
                                return;
                            };
                            defer t.store.releaseMut(out_tile.token);

                            const out_view = out_tile.bufferView();

                            const axis_len: usize = tileDim(t.out_meta.shape, t.out_meta.tile_shape, ti_axis, t.axis) catch |e| {
                                t.fail(e);
                                return;
                            };
                            const out_stride_axis_isize: isize = out_view.layout.strides_bytes[t.axis];
                            if (out_stride_axis_isize < 0) {
                                t.fail(BackendError.InvalidArgument);
                                return;
                            }
                            const out_stride_axis: usize = @intCast(out_stride_axis_isize);

                            computeRowOffsets(
                                row_offsets_out[0..rows],
                                row_coords[0..t.row_dim_count],
                                rows,
                                t.row_dim_count,
                                t.row_dims[0..t.row_dim_count],
                                row_local_strides[0..t.row_dim_count],
                                out_view.layout.strides_bytes,
                            ) catch |e| {
                                t.fail(e);
                                return;
                            };

                            softmax_kernels.normalizeStridedRowsF32(
                                out_view.bytes,
                                axis_len,
                                out_stride_axis,
                                row_offsets_out[0..rows],
                                sum_buf[0..rows],
                            ) catch |e| {
                                t.fail(e);
                                return;
                            };
                        }
                    }
                }
            };

            var task: Task = .{
                .store = store,
                .out_meta = out_meta,
                .out = out,
                .a = a,
                .axis = axis,
                .row_dim_count = row_dim_count,
                .row_dims = row_dims,
                .row_counts = row_counts,
                .row_strides = row_strides,
                .axis_tiles = axis_tiles,
            };
            p.parallelForAny(@ptrCast(&task), row_tile_total, 1, Task.runRowTile);
            if (task.err_any) |e| return @errorCast(e);
            return;
        }
    }

    // Sequential fallback.
    var coords_seq: [MAX_RANK]usize = .{0} ** MAX_RANK;
    var row_sizes_seq: [MAX_RANK]usize = .{0} ** MAX_RANK;
    var row_strides_seq: [MAX_RANK]usize = .{0} ** MAX_RANK;
    var row_coords_seq: [MAX_RANK]usize = .{0} ** MAX_RANK;
    var row_offsets_in_seq: [256]usize = .{0} ** 256;
    var row_offsets_out_seq: [256]usize = .{0} ** 256;

    var row_idx_seq: usize = 0;
    while (row_idx_seq < row_tile_total) : (row_idx_seq += 1) {
        var rem: usize = row_idx_seq;
        var rdi: usize = 0;
        while (rdi < row_dim_count) : (rdi += 1) {
            const stride0: usize = row_strides[rdi];
            const v: usize = rem / stride0;
            coords_seq[row_dims[rdi]] = v;
            rem -= v * stride0;
        }

        var rows: usize = 1;
        rdi = 0;
        while (rdi < row_dim_count) : (rdi += 1) {
            const dim: usize = row_dims[rdi];
            const dim_len: usize = try tileDim(out_meta.shape, out_meta.tile_shape, coords_seq[dim], dim);
            row_sizes_seq[rdi] = dim_len;
            rows = std.math.mul(usize, rows, dim_len) catch return BackendError.InvalidArgument;
        }
        if (rows == 0 or rows > 256) return BackendError.InvalidArgument;

        try exec_utils.computePackedStrides(row_sizes_seq[0..row_dim_count], row_strides_seq[0..row_dim_count]);

        var max_buf: [256]f32 = undefined;
        var sum_buf: [256]f32 = undefined;
        @memset(max_buf[0..rows], -std.math.inf(f32));
        @memset(sum_buf[0..rows], 0.0);

        var ti_axis_seq: usize = 0;
        while (ti_axis_seq < axis_tiles) : (ti_axis_seq += 1) {
            coords_seq[axis] = ti_axis_seq;
            const tile_index: usize = try tensor_store.encodeTileIndex(out_meta, coords_seq[0..out_meta.tile_counts.len]);
            const in_tile = try store.acquireTileConstLinear(a, tile_index);
            defer store.releaseConst(in_tile.token);
            const in_view = in_tile.bufferView();

            const axis_len: usize = try tileDim(out_meta.shape, out_meta.tile_shape, ti_axis_seq, axis);
            const stride_axis_isize: isize = in_view.layout.strides_bytes[axis];
            if (stride_axis_isize < 0) return BackendError.InvalidArgument;
            const stride_axis: usize = @intCast(stride_axis_isize);

            try computeRowOffsets(
                row_offsets_in_seq[0..rows],
                row_coords_seq[0..row_dim_count],
                rows,
                row_dim_count,
                row_dims[0..row_dim_count],
                row_strides_seq[0..row_dim_count],
                in_view.layout.strides_bytes,
            );

            try softmax_kernels.updateMaxStridedRowsF32(
                max_buf[0..rows],
                in_view.bytes,
                axis_len,
                stride_axis,
                row_offsets_in_seq[0..rows],
            );
        }

        ti_axis_seq = 0;
        while (ti_axis_seq < axis_tiles) : (ti_axis_seq += 1) {
            coords_seq[axis] = ti_axis_seq;
            const tile_index: usize = try tensor_store.encodeTileIndex(out_meta, coords_seq[0..out_meta.tile_counts.len]);
            const out_tile = try store.acquireTileMutLinear(out, tile_index);
            defer store.releaseMut(out_tile.token);
            const in_tile = try store.acquireTileConstLinear(a, tile_index);
            defer store.releaseConst(in_tile.token);

            const out_view = out_tile.bufferView();
            const in_view = in_tile.bufferView();

            const axis_len: usize = try tileDim(out_meta.shape, out_meta.tile_shape, ti_axis_seq, axis);
            const stride_axis_isize: isize = in_view.layout.strides_bytes[axis];
            if (stride_axis_isize < 0) return BackendError.InvalidArgument;
            const stride_axis: usize = @intCast(stride_axis_isize);
            const out_stride_axis_isize: isize = out_view.layout.strides_bytes[axis];
            if (out_stride_axis_isize < 0) return BackendError.InvalidArgument;
            const out_stride_axis: usize = @intCast(out_stride_axis_isize);

            try computeRowOffsets(
                row_offsets_in_seq[0..rows],
                row_coords_seq[0..row_dim_count],
                rows,
                row_dim_count,
                row_dims[0..row_dim_count],
                row_strides_seq[0..row_dim_count],
                in_view.layout.strides_bytes,
            );

            try computeRowOffsets(
                row_offsets_out_seq[0..rows],
                row_coords_seq[0..row_dim_count],
                rows,
                row_dim_count,
                row_dims[0..row_dim_count],
                row_strides_seq[0..row_dim_count],
                out_view.layout.strides_bytes,
            );

            try softmax_kernels.expSumStoreStridedRowsF32(
                sum_buf[0..rows],
                out_view.bytes,
                in_view.bytes,
                axis_len,
                stride_axis,
                out_stride_axis,
                row_offsets_in_seq[0..rows],
                row_offsets_out_seq[0..rows],
                max_buf[0..rows],
            );
        }

        var r_check: usize = 0;
        while (r_check < rows) : (r_check += 1) {
            if (!(sum_buf[r_check] > 0.0) or !std.math.isFinite(sum_buf[r_check])) return BackendError.InvalidArgument;
        }

        ti_axis_seq = 0;
        while (ti_axis_seq < axis_tiles) : (ti_axis_seq += 1) {
            coords_seq[axis] = ti_axis_seq;
            const tile_index: usize = try tensor_store.encodeTileIndex(out_meta, coords_seq[0..out_meta.tile_counts.len]);
            const out_tile = try store.acquireTileMutLinear(out, tile_index);
            defer store.releaseMut(out_tile.token);

            const out_view = out_tile.bufferView();

            const axis_len: usize = try tileDim(out_meta.shape, out_meta.tile_shape, ti_axis_seq, axis);
            const out_stride_axis_isize: isize = out_view.layout.strides_bytes[axis];
            if (out_stride_axis_isize < 0) return BackendError.InvalidArgument;
            const out_stride_axis: usize = @intCast(out_stride_axis_isize);

            try computeRowOffsets(
                row_offsets_out_seq[0..rows],
                row_coords_seq[0..row_dim_count],
                rows,
                row_dim_count,
                row_dims[0..row_dim_count],
                row_strides_seq[0..row_dim_count],
                out_view.layout.strides_bytes,
            );

            try softmax_kernels.normalizeStridedRowsF32(
                out_view.bytes,
                axis_len,
                out_stride_axis,
                row_offsets_out_seq[0..rows],
                sum_buf[0..rows],
            );
        }
    }
}

fn softmaxAxisLastND(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    store: tensor_store.TensorStore,
    out_meta: tensor_store.TensorMeta,
    out: tensor_store.TensorId,
    a: tensor_store.TensorId,
    axis: usize,
) ExecuteProgramError!void {
    const rank: usize = @as(usize, out_meta.rank);
    if (axis + 1 != rank) return BackendError.InvalidArgument;

    var row_dims: [MAX_RANK]usize = undefined;
    var row_counts: [MAX_RANK]usize = undefined;
    var row_strides: [MAX_RANK]usize = undefined;
    var row_dim_count: usize = 0;

    var d: usize = 0;
    while (d < rank) : (d += 1) {
        if (d == axis) continue;
        row_dims[row_dim_count] = d;
        row_counts[row_dim_count] = out_meta.tile_counts[d];
        row_dim_count += 1;
    }

    var stride: usize = 1;
    var i: usize = row_dim_count;
    while (i > 0) : (i -= 1) {
        const idx: usize = i - 1;
        row_strides[idx] = stride;
        stride = std.math.mul(usize, stride, row_counts[idx]) catch return BackendError.InvalidArgument;
    }
    const row_tile_total: usize = stride;

    const tile_bytes: usize = exec_utils.tileByteSize(out_meta);
    const min_total_bytes: usize = 256 * 1024;
    var tile_total: usize = 1;
    d = 0;
    while (d < rank) : (d += 1) {
        tile_total *= out_meta.tile_counts[d];
    }

    const axis_tiles: usize = out_meta.tile_counts[axis];

    if (pool) |p| {
        if (exec_utils.shouldParallelTiles(thread_count, tile_total, tile_bytes, min_total_bytes) and row_tile_total >= 2) {
            const Task = struct {
                store: tensor_store.TensorStore,
                out_meta: tensor_store.TensorMeta,
                out: tensor_store.TensorId,
                a: tensor_store.TensorId,
                axis: usize,
                row_dim_count: usize,
                row_dims: [MAX_RANK]usize,
                row_counts: [MAX_RANK]usize,
                row_strides: [MAX_RANK]usize,
                axis_tiles: usize,

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

                    var coords: [MAX_RANK]usize = .{0} ** MAX_RANK;

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

                        // Rows per tile = product of tile dims for leading dims.
                        var rows: usize = 1;
                        rdi = 0;
                        while (rdi < t.row_dim_count) : (rdi += 1) {
                            const dim: usize = t.row_dims[rdi];
                            const dim_len: usize = tileDim(t.out_meta.shape, t.out_meta.tile_shape, coords[dim], dim) catch |e| {
                                t.fail(e);
                                return;
                            };
                            rows = std.math.mul(usize, rows, dim_len) catch |e| {
                                t.fail(e);
                                return;
                            };
                        }
                        if (rows == 0 or rows > 256) {
                            t.fail(BackendError.InvalidArgument);
                            return;
                        }

                        var max_buf: [256]f32 = undefined;
                        var sum_buf: [256]f32 = undefined;
                        @memset(max_buf[0..rows], -std.math.inf(f32));
                        @memset(sum_buf[0..rows], 0.0);

                        var ti_axis: usize = 0;
                        while (ti_axis < t.axis_tiles) : (ti_axis += 1) {
                            if (t.stop.load(.acquire)) return;
                            coords[t.axis] = ti_axis;
                            const tile_index: usize = tensor_store.encodeTileIndex(t.out_meta, coords[0..t.out_meta.tile_counts.len]) catch |e| {
                                t.fail(e);
                                return;
                            };
                            const in_tile = t.store.acquireTileConstLinear(t.a, tile_index) catch |e| {
                                t.fail(e);
                                return;
                            };
                            defer t.store.releaseConst(in_tile.token);

                            const cols: usize = tileDim(t.out_meta.shape, t.out_meta.tile_shape, ti_axis, t.axis) catch |e| {
                                t.fail(e);
                                return;
                            };
                            const elem_bytes: usize = 4;
                            var shape_mem: [2]usize = .{ rows, cols };
                            var strides_mem: [2]isize = .{ @intCast(cols * elem_bytes), @intCast(elem_bytes) };
                            const in_view: types.BufferViewConst = .{ .bytes = in_tile.bytes, .dtype = .f32, .layout = .{ .rank = 2, .shape = shape_mem[0..2], .strides_bytes = strides_mem[0..2] } };
                            softmax_kernels.updateMaxF32(max_buf[0..rows], in_view, 2);
                        }

                        ti_axis = 0;
                        while (ti_axis < t.axis_tiles) : (ti_axis += 1) {
                            if (t.stop.load(.acquire)) return;
                            coords[t.axis] = ti_axis;
                            const tile_index: usize = tensor_store.encodeTileIndex(t.out_meta, coords[0..t.out_meta.tile_counts.len]) catch |e| {
                                t.fail(e);
                                return;
                            };
                            const out_tile = t.store.acquireTileMutLinear(t.out, tile_index) catch |e| {
                                t.fail(e);
                                return;
                            };
                            defer t.store.releaseMut(out_tile.token);
                            const in_tile = t.store.acquireTileConstLinear(t.a, tile_index) catch |e| {
                                t.fail(e);
                                return;
                            };
                            defer t.store.releaseConst(in_tile.token);

                            const cols: usize = tileDim(t.out_meta.shape, t.out_meta.tile_shape, ti_axis, t.axis) catch |e| {
                                t.fail(e);
                                return;
                            };
                            const elem_bytes: usize = 4;
                            var shape_mem: [2]usize = .{ rows, cols };
                            var strides_mem: [2]isize = .{ @intCast(cols * elem_bytes), @intCast(elem_bytes) };
                            const out_view: types.BufferViewMut = .{ .bytes = out_tile.bytes, .dtype = .f32, .layout = .{ .rank = 2, .shape = shape_mem[0..2], .strides_bytes = strides_mem[0..2] } };
                            const in_view: types.BufferViewConst = .{ .bytes = in_tile.bytes, .dtype = .f32, .layout = .{ .rank = 2, .shape = shape_mem[0..2], .strides_bytes = strides_mem[0..2] } };
                            softmax_kernels.expSumStoreF32(sum_buf[0..rows], out_view, in_view, max_buf[0..rows], 2);
                        }

                        ti_axis = 0;
                        while (ti_axis < t.axis_tiles) : (ti_axis += 1) {
                            if (t.stop.load(.acquire)) return;
                            coords[t.axis] = ti_axis;
                            const tile_index: usize = tensor_store.encodeTileIndex(t.out_meta, coords[0..t.out_meta.tile_counts.len]) catch |e| {
                                t.fail(e);
                                return;
                            };
                            const out_tile = t.store.acquireTileMutLinear(t.out, tile_index) catch |e| {
                                t.fail(e);
                                return;
                            };
                            defer t.store.releaseMut(out_tile.token);

                            const cols: usize = tileDim(t.out_meta.shape, t.out_meta.tile_shape, ti_axis, t.axis) catch |e| {
                                t.fail(e);
                                return;
                            };
                            const elem_bytes: usize = 4;
                            var shape_mem: [2]usize = .{ rows, cols };
                            var strides_mem: [2]isize = .{ @intCast(cols * elem_bytes), @intCast(elem_bytes) };
                            const out_view: types.BufferViewMut = .{ .bytes = out_tile.bytes, .dtype = .f32, .layout = .{ .rank = 2, .shape = shape_mem[0..2], .strides_bytes = strides_mem[0..2] } };
                            softmax_kernels.normalizeF32(out_view, sum_buf[0..rows], 2);
                        }
                    }
                }
            };

            var task: Task = .{
                .store = store,
                .out_meta = out_meta,
                .out = out,
                .a = a,
                .axis = axis,
                .row_dim_count = row_dim_count,
                .row_dims = row_dims,
                .row_counts = row_counts,
                .row_strides = row_strides,
                .axis_tiles = axis_tiles,
            };
            p.parallelForAny(@ptrCast(&task), row_tile_total, 1, Task.runRowTile);
            if (task.err_any) |e| return @errorCast(e);
            return;
        }
    }

    // Sequential fallback.
    var coords_seq: [MAX_RANK]usize = .{0} ** MAX_RANK;
    var row_idx_seq: usize = 0;
    while (row_idx_seq < row_tile_total) : (row_idx_seq += 1) {
        var rem: usize = row_idx_seq;
        var rdi: usize = 0;
        while (rdi < row_dim_count) : (rdi += 1) {
            const stride0: usize = row_strides[rdi];
            const v: usize = rem / stride0;
            coords_seq[row_dims[rdi]] = v;
            rem -= v * stride0;
        }

        var rows: usize = 1;
        rdi = 0;
        while (rdi < row_dim_count) : (rdi += 1) {
            const dim: usize = row_dims[rdi];
            const dim_len: usize = try tileDim(out_meta.shape, out_meta.tile_shape, coords_seq[dim], dim);
            rows = std.math.mul(usize, rows, dim_len) catch return BackendError.InvalidArgument;
        }
        if (rows == 0 or rows > 256) return BackendError.InvalidArgument;

        var max_buf: [256]f32 = undefined;
        var sum_buf: [256]f32 = undefined;
        @memset(max_buf[0..rows], -std.math.inf(f32));
        @memset(sum_buf[0..rows], 0.0);

        var ti_axis_seq: usize = 0;
        while (ti_axis_seq < axis_tiles) : (ti_axis_seq += 1) {
            coords_seq[axis] = ti_axis_seq;
            const tile_index: usize = try tensor_store.encodeTileIndex(out_meta, coords_seq[0..out_meta.tile_counts.len]);
            const in_tile = try store.acquireTileConstLinear(a, tile_index);
            defer store.releaseConst(in_tile.token);

            const cols: usize = try tileDim(out_meta.shape, out_meta.tile_shape, ti_axis_seq, axis);
            const elem_bytes: usize = 4;
            var shape_mem: [2]usize = .{ rows, cols };
            var strides_mem: [2]isize = .{ @intCast(cols * elem_bytes), @intCast(elem_bytes) };
            const in_view: types.BufferViewConst = .{ .bytes = in_tile.bytes, .dtype = .f32, .layout = .{ .rank = 2, .shape = shape_mem[0..2], .strides_bytes = strides_mem[0..2] } };
            softmax_kernels.updateMaxF32(max_buf[0..rows], in_view, 2);
        }

        ti_axis_seq = 0;
        while (ti_axis_seq < axis_tiles) : (ti_axis_seq += 1) {
            coords_seq[axis] = ti_axis_seq;
            const tile_index: usize = try tensor_store.encodeTileIndex(out_meta, coords_seq[0..out_meta.tile_counts.len]);
            const out_tile = try store.acquireTileMutLinear(out, tile_index);
            defer store.releaseMut(out_tile.token);
            const in_tile = try store.acquireTileConstLinear(a, tile_index);
            defer store.releaseConst(in_tile.token);

            const cols: usize = try tileDim(out_meta.shape, out_meta.tile_shape, ti_axis_seq, axis);
            const elem_bytes: usize = 4;
            var shape_mem: [2]usize = .{ rows, cols };
            var strides_mem: [2]isize = .{ @intCast(cols * elem_bytes), @intCast(elem_bytes) };
            const out_view: types.BufferViewMut = .{ .bytes = out_tile.bytes, .dtype = .f32, .layout = .{ .rank = 2, .shape = shape_mem[0..2], .strides_bytes = strides_mem[0..2] } };
            const in_view: types.BufferViewConst = .{ .bytes = in_tile.bytes, .dtype = .f32, .layout = .{ .rank = 2, .shape = shape_mem[0..2], .strides_bytes = strides_mem[0..2] } };
            softmax_kernels.expSumStoreF32(sum_buf[0..rows], out_view, in_view, max_buf[0..rows], 2);
        }

        ti_axis_seq = 0;
        while (ti_axis_seq < axis_tiles) : (ti_axis_seq += 1) {
            coords_seq[axis] = ti_axis_seq;
            const tile_index: usize = try tensor_store.encodeTileIndex(out_meta, coords_seq[0..out_meta.tile_counts.len]);
            const out_tile = try store.acquireTileMutLinear(out, tile_index);
            defer store.releaseMut(out_tile.token);

            const cols: usize = try tileDim(out_meta.shape, out_meta.tile_shape, ti_axis_seq, axis);
            const elem_bytes: usize = 4;
            var shape_mem: [2]usize = .{ rows, cols };
            var strides_mem: [2]isize = .{ @intCast(cols * elem_bytes), @intCast(elem_bytes) };
            const out_view: types.BufferViewMut = .{ .bytes = out_tile.bytes, .dtype = .f32, .layout = .{ .rank = 2, .shape = shape_mem[0..2], .strides_bytes = strides_mem[0..2] } };
            softmax_kernels.normalizeF32(out_view, sum_buf[0..rows], 2);
        }
    }
}

fn softmaxRank1(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    scratch_f32: []f32,
    store: tensor_store.TensorStore,
    out_meta: tensor_store.TensorMeta,
    out: tensor_store.TensorId,
    a: tensor_store.TensorId,
) ExecuteProgramError!void {
    std.debug.assert(out_meta.rank == 1);

    const ti0_total: usize = out_meta.tile_counts[0];
    const tc1: usize = 1;

    const tile_bytes: usize = exec_utils.tileByteSize(out_meta);
    const min_total_bytes: usize = 256 * 1024;
    const total_bytes: usize = tile_bytes * (out_meta.tile_counts[0] * tc1);

    if (pool) |p| {
        if (thread_count > 1 and ti0_total >= 2 and total_bytes >= min_total_bytes and scratch_f32.len >= thread_count * 2) {
            const scratch_max: []f32 = scratch_f32[0..thread_count];
            const scratch_sum: []f32 = scratch_f32[thread_count .. thread_count * 2];

            @memset(scratch_max, -std.math.inf(f32));
            @memset(scratch_sum, 0.0);

            const TaskMax = struct {
                store: tensor_store.TensorStore,
                out_meta: tensor_store.TensorMeta,
                a: tensor_store.TensorId,
                scratch_max: []f32,
                tc1: usize,

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
                    const t: *@This() = @ptrCast(@alignCast(ctx_any));
                    if (tid < t.scratch_max.len) t.scratch_max[tid] = -std.math.inf(f32);
                    if (start >= end) return;
                    if (t.stop.load(.acquire)) return;

                    var max_buf: [1]f32 = .{-std.math.inf(f32)};
                    var ti0: usize = start;
                    while (ti0 < end) : (ti0 += 1) {
                        if (t.stop.load(.acquire)) return;
                        var ti1: usize = 0;
                        while (ti1 < t.tc1) : (ti1 += 1) {
                            if (ti1 + 1 < t.tc1) t.store.prefetch(t.a, ti0, ti1 + 1);
                            const in_tile = t.store.acquireTileConst(t.a, ti0, ti1) catch |e| {
                                t.fail(e);
                                return;
                            };
                            defer t.store.releaseConst(in_tile.token);
                            const in_view = in_tile.bufferView();
                            if (in_view.dtype != .f32) {
                                t.fail(BackendError.InvalidArgument);
                                return;
                            }
                            softmax_kernels.updateMaxF32(max_buf[0..1], in_view, 1);
                        }
                    }
                    if (tid < t.scratch_max.len) t.scratch_max[tid] = max_buf[0];
                }
            };

            var task_max: TaskMax = .{ .store = store, .out_meta = out_meta, .a = a, .scratch_max = scratch_max, .tc1 = tc1 };
            p.parallelForAny(@ptrCast(&task_max), ti0_total, 1, TaskMax.runTi0);
            if (task_max.err_any) |e| return @errorCast(e);

            var global_max: f32 = -std.math.inf(f32);
            for (scratch_max) |m| global_max = @max(global_max, m);

            const TaskExpSum = struct {
                store: tensor_store.TensorStore,
                out_meta: tensor_store.TensorMeta,
                out: tensor_store.TensorId,
                a: tensor_store.TensorId,
                global_max: f32,
                scratch_sum: []f32,
                tc1: usize,

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
                    const t: *@This() = @ptrCast(@alignCast(ctx_any));
                    if (tid < t.scratch_sum.len) t.scratch_sum[tid] = 0.0;
                    if (start >= end) return;
                    if (t.stop.load(.acquire)) return;

                    var sum_buf: [1]f32 = .{0.0};
                    const max_buf: [1]f32 = .{t.global_max};

                    var ti0: usize = start;
                    while (ti0 < end) : (ti0 += 1) {
                        if (t.stop.load(.acquire)) return;
                        var ti1: usize = 0;
                        while (ti1 < t.tc1) : (ti1 += 1) {
                            if (ti1 + 1 < t.tc1) {
                                t.store.prefetch(t.a, ti0, ti1 + 1);
                                t.store.prefetch(t.out, ti0, ti1 + 1);
                            }
                            const out_tile = t.store.acquireTileMut(t.out, ti0, ti1) catch |e| {
                                t.fail(e);
                                return;
                            };
                            defer t.store.releaseMut(out_tile.token);
                            const in_tile = t.store.acquireTileConst(t.a, ti0, ti1) catch |e| {
                                t.fail(e);
                                return;
                            };
                            defer t.store.releaseConst(in_tile.token);

                            const out_view = out_tile.bufferView();
                            const in_view = in_tile.bufferView();
                            if (out_view.dtype != .f32 or in_view.dtype != .f32) {
                                t.fail(BackendError.InvalidArgument);
                                return;
                            }
                            softmax_kernels.expSumStoreF32(sum_buf[0..1], out_view, in_view, max_buf[0..1], 1);
                        }
                    }

                    if (tid < t.scratch_sum.len) t.scratch_sum[tid] = sum_buf[0];
                }
            };

            var task_sum: TaskExpSum = .{ .store = store, .out_meta = out_meta, .out = out, .a = a, .global_max = global_max, .scratch_sum = scratch_sum, .tc1 = tc1 };
            p.parallelForAny(@ptrCast(&task_sum), ti0_total, 1, TaskExpSum.runTi0);
            if (task_sum.err_any) |e| return @errorCast(e);

            var global_sum: f32 = 0.0;
            for (scratch_sum) |v| global_sum += v;
            if (!(global_sum > 0.0) or !std.math.isFinite(global_sum)) return BackendError.InvalidArgument;

            const inv: f32 = 1.0 / global_sum;

            const TaskNorm = struct {
                store: tensor_store.TensorStore,
                out_meta: tensor_store.TensorMeta,
                out: tensor_store.TensorId,
                inv: f32,
                tc1: usize,

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

                    const lanes: usize = comptime simd.lanesF32();
                    const Vec = @Vector(lanes, f32);
                    const inv_v: Vec = @splat(t.inv);

                    var ti0: usize = start;
                    while (ti0 < end) : (ti0 += 1) {
                        if (t.stop.load(.acquire)) return;
                        var ti1: usize = 0;
                        while (ti1 < t.tc1) : (ti1 += 1) {
                            if (ti1 + 1 < t.tc1) t.store.prefetch(t.out, ti0, ti1 + 1);
                            const out_tile = t.store.acquireTileMut(t.out, ti0, ti1) catch |e| {
                                t.fail(e);
                                return;
                            };
                            defer t.store.releaseMut(out_tile.token);
                            const out_view = out_tile.bufferView();
                            if (out_view.dtype != .f32) {
                                t.fail(BackendError.InvalidArgument);
                                return;
                            }
                            var out_slice: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, out_view.bytes);
                            var i: usize = 0;
                            const vec_end: usize = out_slice.len - (out_slice.len % lanes);
                            while (i < vec_end) : (i += lanes) {
                                const v: Vec = @as(*align(1) const Vec, @ptrCast(out_slice.ptr + i)).*;
                                @as(*align(1) Vec, @ptrCast(out_slice.ptr + i)).* = v * inv_v;
                            }
                            while (i < out_slice.len) : (i += 1) out_slice[i] *= t.inv;
                        }
                    }
                }
            };

            var task_norm: TaskNorm = .{ .store = store, .out_meta = out_meta, .out = out, .inv = inv, .tc1 = tc1 };
            p.parallelForAny(@ptrCast(&task_norm), ti0_total, 1, TaskNorm.runTi0);
            if (task_norm.err_any) |e| return @errorCast(e);
            return;
        }
    }

    var global_max: f32 = -std.math.inf(f32);
    var ti0: usize = 0;
    while (ti0 < ti0_total) : (ti0 += 1) {
        var ti1: usize = 0;
        while (ti1 < tc1) : (ti1 += 1) {
            if (ti1 + 1 < tc1) store.prefetch(a, ti0, ti1 + 1);
            const in_tile = try store.acquireTileConst(a, ti0, ti1);
            defer store.releaseConst(in_tile.token);
            const in_view = in_tile.bufferView();
            if (in_view.dtype != .f32) return BackendError.InvalidArgument;
            var max_buf: [1]f32 = .{global_max};
            softmax_kernels.updateMaxF32(max_buf[0..1], in_view, 1);
            global_max = max_buf[0];
        }
    }

    var global_sum: f32 = 0.0;
    ti0 = 0;
    while (ti0 < ti0_total) : (ti0 += 1) {
        var ti1: usize = 0;
        while (ti1 < tc1) : (ti1 += 1) {
            if (ti1 + 1 < tc1) {
                store.prefetch(a, ti0, ti1 + 1);
                store.prefetch(out, ti0, ti1 + 1);
            }
            const out_tile = try store.acquireTileMut(out, ti0, ti1);
            defer store.releaseMut(out_tile.token);
            const in_tile = try store.acquireTileConst(a, ti0, ti1);
            defer store.releaseConst(in_tile.token);

            const out_view = out_tile.bufferView();
            const in_view = in_tile.bufferView();
            if (out_view.dtype != .f32 or in_view.dtype != .f32) return BackendError.InvalidArgument;
            var sum_buf: [1]f32 = .{global_sum};
            const max_buf: [1]f32 = .{global_max};
            softmax_kernels.expSumStoreF32(sum_buf[0..1], out_view, in_view, max_buf[0..1], 1);
            global_sum = sum_buf[0];
        }
    }
    if (!(global_sum > 0.0) or !std.math.isFinite(global_sum)) return BackendError.InvalidArgument;

    const inv: f32 = 1.0 / global_sum;
    ti0 = 0;
    while (ti0 < ti0_total) : (ti0 += 1) {
        var ti1: usize = 0;
        while (ti1 < tc1) : (ti1 += 1) {
            if (ti1 + 1 < tc1) store.prefetch(out, ti0, ti1 + 1);
            const out_tile = try store.acquireTileMut(out, ti0, ti1);
            defer store.releaseMut(out_tile.token);
            const out_view = out_tile.bufferView();
            if (out_view.dtype != .f32) return BackendError.InvalidArgument;

            var out_slice: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, out_view.bytes);
            const lanes: usize = comptime simd.lanesF32();
            const Vec = @Vector(lanes, f32);
            const inv_v: Vec = @splat(inv);
            var i: usize = 0;
            const vec_end: usize = out_slice.len - (out_slice.len % lanes);
            while (i < vec_end) : (i += lanes) {
                const v: Vec = @as(*align(1) const Vec, @ptrCast(out_slice.ptr + i)).*;
                @as(*align(1) Vec, @ptrCast(out_slice.ptr + i)).* = v * inv_v;
            }
            while (i < out_slice.len) : (i += 1) out_slice[i] *= inv;
        }
    }
}

fn softmaxTi0(
    store: tensor_store.TensorStore,
    out_meta: tensor_store.TensorMeta,
    out: tensor_store.TensorId,
    a: tensor_store.TensorId,
    ti0: usize,
) ExecuteProgramError!void {
    // v0 contract: tile_shape[0] <= 256 validated at compile time.
    const tm: usize = out_meta.tile_shape[0];
    if (tm == 0 or tm > 256) return BackendError.InvalidArgument;

    var max_buf: [256]f32 = undefined;
    var sum_buf: [256]f32 = undefined;
    @memset(max_buf[0..tm], -std.math.inf(f32));
    @memset(sum_buf[0..tm], 0.0);

    const tc1: usize = out_meta.tile_counts[1];

    // Pass 1: max per row.
    var ti1: usize = 0;
    while (ti1 < tc1) : (ti1 += 1) {
        if (ti1 + 1 < tc1) store.prefetch(a, ti0, ti1 + 1);
        const in_tile = try store.acquireTileConst(a, ti0, ti1);
        defer store.releaseConst(in_tile.token);
        const in_view = in_tile.bufferView();
        if (in_view.dtype != .f32) return BackendError.InvalidArgument;
        softmax_kernels.updateMaxF32(max_buf[0..tm], in_view, out_meta.rank);
    }

    // Pass 2: exp(x-max), store to out, accumulate sum.
    ti1 = 0;
    while (ti1 < tc1) : (ti1 += 1) {
        if (ti1 + 1 < tc1) {
            store.prefetch(a, ti0, ti1 + 1);
            store.prefetch(out, ti0, ti1 + 1);
        }
        const out_tile = try store.acquireTileMut(out, ti0, ti1);
        defer store.releaseMut(out_tile.token);
        const in_tile = try store.acquireTileConst(a, ti0, ti1);
        defer store.releaseConst(in_tile.token);

        const out_view = out_tile.bufferView();
        const in_view = in_tile.bufferView();
        if (out_view.dtype != .f32 or in_view.dtype != .f32) return BackendError.InvalidArgument;
        softmax_kernels.expSumStoreF32(sum_buf[0..tm], out_view, in_view, max_buf[0..tm], out_meta.rank);
    }

    // Pass 3: normalize.
    ti1 = 0;
    while (ti1 < tc1) : (ti1 += 1) {
        if (ti1 + 1 < tc1) store.prefetch(out, ti0, ti1 + 1);
        const out_tile = try store.acquireTileMut(out, ti0, ti1);
        defer store.releaseMut(out_tile.token);
        const out_view = out_tile.bufferView();
        if (out_view.dtype != .f32) return BackendError.InvalidArgument;
        softmax_kernels.normalizeF32(out_view, sum_buf[0..tm], out_meta.rank);
    }
}
