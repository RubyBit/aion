const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const reduce_k = @import("../kernels/reduce.zig");
const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");

const BackendError = types.BackendError;
const DType = types.DType;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

pub fn tileElemCount(meta: tensor_store.TensorMeta) usize {
    return switch (meta.rank) {
        0 => 1,
        1 => meta.tile_shape[0],
        else => meta.tile_shape[0] * meta.tile_shape[1],
    };
}

pub fn tileByteSize(meta: tensor_store.TensorMeta) usize {
    const elems: usize = tileElemCount(meta);
    return switch (meta.dtype) {
        .f32 => elems * 4,
        .f16 => elems * 2,
        .i8 => elems,
        .q4_0, .q8_0 => blk: {
            const info = meta.dtype.info();
            const blocks = std.math.divCeil(usize, elems, info.block_elems) catch return 0;
            break :blk blocks * info.block_bytes;
        },
    };
}

/// Heuristic: parallelize tiled ops when either (a) we have enough tiles to hand out
/// work, or (b) the total byte volume is large enough to amortize scheduling overhead.
pub fn shouldParallelTiles(thread_count: usize, tile_total: usize, tile_bytes: usize, min_total_bytes: usize) bool {
    if (thread_count <= 1) return false;
    if (tile_total == 0) return false;
    const total_bytes: usize = tile_bytes * tile_total;
    if (total_bytes >= min_total_bytes) return true;
    // Also allow when we have at least two tiles; matmul-like ops are compute heavy per tile.
    return tile_total >= 2;
}

pub fn elemCountFromTileView(view: anytype) usize {
    // view.layout.rank is u8; shape slice length matches rank.
    if (view.layout.rank == 1) return view.layout.shape[0];
    return view.layout.shape[0] * view.layout.shape[1];
}

fn scalarElemBytes(dtype: DType) ExecuteProgramError!usize {
    return switch (dtype) {
        .f32 => 4,
        .f16 => 2,
        else => return BackendError.InvalidArgument,
    };
}

const ScalarTileCacheConst = struct {
    valid: bool = false,
    ti0: usize = 0,
    ti1: usize = 0,
    tile: tensor_store.TileRefConst = undefined,
};

const ScalarTileCacheMut = struct {
    valid: bool = false,
    ti0: usize = 0,
    ti1: usize = 0,
    tile: tensor_store.TileRefMut = undefined,
};

fn ensureConstTile(cache: *ScalarTileCacheConst, store: tensor_store.TensorStore, id: tensor_store.TensorId, ti0: usize, ti1: usize) ExecuteProgramError!void {
    if (cache.valid and cache.ti0 == ti0 and cache.ti1 == ti1) return;
    if (cache.valid) store.releaseConst(cache.tile.token);
    cache.tile = try store.acquireTileConst(id, ti0, ti1);
    cache.ti0 = ti0;
    cache.ti1 = ti1;
    cache.valid = true;
}

fn ensureMutTile(cache: *ScalarTileCacheMut, store: tensor_store.TensorStore, id: tensor_store.TensorId, ti0: usize, ti1: usize) ExecuteProgramError!void {
    if (cache.valid and cache.ti0 == ti0 and cache.ti1 == ti1) return;
    if (cache.valid) store.releaseMut(cache.tile.token);
    cache.tile = try store.acquireTileMut(id, ti0, ti1);
    cache.ti0 = ti0;
    cache.ti1 = ti1;
    cache.valid = true;
}

fn readScalarBytesAt(
    store: tensor_store.TensorStore,
    meta: tensor_store.TensorMeta,
    id: tensor_store.TensorId,
    elem_bytes: usize,
    idx0: usize,
    idx1: usize,
    cache: *ScalarTileCacheConst,
    out: []u8,
) ExecuteProgramError!void {
    if (meta.rank == 1) {
        const ti: usize = idx0 / meta.tile_shape[0];
        const in_i: usize = idx0 - ti * meta.tile_shape[0];
        try ensureConstTile(cache, store, id, ti, 0);
        const off: usize = in_i * elem_bytes;
        @memcpy(out, cache.tile.bytes[off .. off + elem_bytes]);
        return;
    }

    const ti0: usize = idx0 / meta.tile_shape[0];
    const ti1: usize = idx1 / meta.tile_shape[1];
    const in0: usize = idx0 - ti0 * meta.tile_shape[0];
    const in1: usize = idx1 - ti1 * meta.tile_shape[1];

    try ensureConstTile(cache, store, id, ti0, ti1);
    const n_tile: usize = cache.tile.shape_mem[1];
    const off: usize = (in0 * n_tile + in1) * elem_bytes;
    @memcpy(out, cache.tile.bytes[off .. off + elem_bytes]);
}

fn writeScalarBytesAt(
    store: tensor_store.TensorStore,
    meta: tensor_store.TensorMeta,
    id: tensor_store.TensorId,
    elem_bytes: usize,
    idx0: usize,
    idx1: usize,
    cache: *ScalarTileCacheMut,
    bytes: []const u8,
) ExecuteProgramError!void {
    if (meta.rank == 1) {
        const ti: usize = idx0 / meta.tile_shape[0];
        const in_i: usize = idx0 - ti * meta.tile_shape[0];
        try ensureMutTile(cache, store, id, ti, 0);
        const off: usize = in_i * elem_bytes;
        @memcpy(cache.tile.bytes[off .. off + elem_bytes], bytes);
        return;
    }

    const ti0: usize = idx0 / meta.tile_shape[0];
    const ti1: usize = idx1 / meta.tile_shape[1];
    const in0: usize = idx0 - ti0 * meta.tile_shape[0];
    const in1: usize = idx1 - ti1 * meta.tile_shape[1];

    try ensureMutTile(cache, store, id, ti0, ti1);
    const n_tile: usize = cache.tile.shape_mem[1];
    const off: usize = (in0 * n_tile + in1) * elem_bytes;
    @memcpy(cache.tile.bytes[off .. off + elem_bytes], bytes);
}

pub fn retileCopyScalar(store: tensor_store.TensorStore, dst_id: tensor_store.TensorId, src_id: tensor_store.TensorId) ExecuteProgramError!void {
    const dst_meta = try store.meta(dst_id);
    const src_meta = try store.meta(src_id);
    const elem_bytes: usize = try scalarElemBytes(dst_meta.dtype);

    var src_cache: ScalarTileCacheConst = .{};
    defer if (src_cache.valid) store.releaseConst(src_cache.tile.token);
    var dst_cache: ScalarTileCacheMut = .{};
    defer if (dst_cache.valid) store.releaseMut(dst_cache.tile.token);

    var tmp: [4]u8 = .{ 0, 0, 0, 0 };
    const buf: []u8 = tmp[0..elem_bytes];

    if (dst_meta.rank == 1) {
        var i: usize = 0;
        while (i < dst_meta.shape[0]) : (i += 1) {
            try readScalarBytesAt(store, src_meta, src_id, elem_bytes, i, 0, &src_cache, buf);
            try writeScalarBytesAt(store, dst_meta, dst_id, elem_bytes, i, 0, &dst_cache, buf);
        }
        return;
    }

    var r0: usize = 0;
    while (r0 < dst_meta.shape[0]) : (r0 += 1) {
        var c0: usize = 0;
        while (c0 < dst_meta.shape[1]) : (c0 += 1) {
            try readScalarBytesAt(store, src_meta, src_id, elem_bytes, r0, c0, &src_cache, buf);
            try writeScalarBytesAt(store, dst_meta, dst_id, elem_bytes, r0, c0, &dst_cache, buf);
        }
    }
}

pub fn reshapeCopyScalar(store: tensor_store.TensorStore, dst_id: tensor_store.TensorId, src_id: tensor_store.TensorId) ExecuteProgramError!void {
    const dst_meta = try store.meta(dst_id);
    const src_meta = try store.meta(src_id);

    const elem_bytes: usize = try scalarElemBytes(dst_meta.dtype);
    const dst_elems: usize = if (dst_meta.rank == 1) dst_meta.shape[0] else (dst_meta.shape[0] * dst_meta.shape[1]);

    var src_cache: ScalarTileCacheConst = .{};
    defer if (src_cache.valid) store.releaseConst(src_cache.tile.token);
    var dst_cache: ScalarTileCacheMut = .{};
    defer if (dst_cache.valid) store.releaseMut(dst_cache.tile.token);

    var tmp: [4]u8 = .{ 0, 0, 0, 0 };
    const buf: []u8 = tmp[0..elem_bytes];

    var lin: usize = 0;
    while (lin < dst_elems) : (lin += 1) {
        const src_i0: usize = if (src_meta.rank == 1) lin else (lin / src_meta.shape[1]);
        const src_i1: usize = if (src_meta.rank == 1) 0 else (lin - src_i0 * src_meta.shape[1]);

        const dst_i0: usize = if (dst_meta.rank == 1) lin else (lin / dst_meta.shape[1]);
        const dst_i1: usize = if (dst_meta.rank == 1) 0 else (lin - dst_i0 * dst_meta.shape[1]);

        try readScalarBytesAt(store, src_meta, src_id, elem_bytes, src_i0, src_i1, &src_cache, buf);
        try writeScalarBytesAt(store, dst_meta, dst_id, elem_bytes, dst_i0, dst_i1, &dst_cache, buf);
    }
}

pub fn transpose2DCopyScalar(store: tensor_store.TensorStore, dst_id: tensor_store.TensorId, src_id: tensor_store.TensorId) ExecuteProgramError!void {
    const dst_meta = try store.meta(dst_id);
    const src_meta = try store.meta(src_id);

    const elem_bytes: usize = try scalarElemBytes(dst_meta.dtype);
    var src_cache: ScalarTileCacheConst = .{};
    defer if (src_cache.valid) store.releaseConst(src_cache.tile.token);
    var dst_cache: ScalarTileCacheMut = .{};
    defer if (dst_cache.valid) store.releaseMut(dst_cache.tile.token);

    var tmp: [4]u8 = .{ 0, 0, 0, 0 };
    const buf: []u8 = tmp[0..elem_bytes];

    var r0: usize = 0;
    while (r0 < dst_meta.shape[0]) : (r0 += 1) {
        var c0: usize = 0;
        while (c0 < dst_meta.shape[1]) : (c0 += 1) {
            try readScalarBytesAt(store, src_meta, src_id, elem_bytes, c0, r0, &src_cache, buf);
            try writeScalarBytesAt(store, dst_meta, dst_id, elem_bytes, r0, c0, &dst_cache, buf);
        }
    }
}

pub fn slice2DCopyScalar(store: tensor_store.TensorStore, dst_id: tensor_store.TensorId, src_id: tensor_store.TensorId, start0: usize, start1: usize) ExecuteProgramError!void {
    const dst_meta = try store.meta(dst_id);
    const src_meta = try store.meta(src_id);

    const elem_bytes: usize = try scalarElemBytes(dst_meta.dtype);
    var src_cache: ScalarTileCacheConst = .{};
    defer if (src_cache.valid) store.releaseConst(src_cache.tile.token);
    var dst_cache: ScalarTileCacheMut = .{};
    defer if (dst_cache.valid) store.releaseMut(dst_cache.tile.token);

    var tmp: [4]u8 = .{ 0, 0, 0, 0 };
    const buf: []u8 = tmp[0..elem_bytes];

    var r0: usize = 0;
    while (r0 < dst_meta.shape[0]) : (r0 += 1) {
        var c0: usize = 0;
        while (c0 < dst_meta.shape[1]) : (c0 += 1) {
            try readScalarBytesAt(store, src_meta, src_id, elem_bytes, start0 + r0, start1 + c0, &src_cache, buf);
            try writeScalarBytesAt(store, dst_meta, dst_id, elem_bytes, r0, c0, &dst_cache, buf);
        }
    }
}

pub fn reduceAllScalar(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    scratch_f32: []f32,
    op: types.ReduceOp,
    out_id: tensor_store.TensorId,
    a_id: tensor_store.TensorId,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    const out_meta = try store.meta(out_id);
    const a_meta = try store.meta(a_id);

    if (out_meta.dtype != a_meta.dtype) return BackendError.InvalidArgument;

    const total_elems_u64: u64 = switch (a_meta.rank) {
        1 => @as(u64, @intCast(a_meta.shape[0])),
        2 => @as(u64, @intCast(a_meta.shape[0])) * @as(u64, @intCast(a_meta.shape[1])),
        else => return BackendError.InvalidArgument,
    };

    var sum_f64: f64 = 0.0;
    var did_parallel: bool = false;

    if (pool) |p| {
        const tile_total: usize = a_meta.tile_counts[0] * a_meta.tile_counts[1];

        if (thread_count > 1 and scratch_f32.len >= thread_count and tile_total >= 2) {
            @memset(scratch_f32[0..thread_count], 0.0);

            const Task = struct {
                store: tensor_store.TensorStore,
                a_id: tensor_store.TensorId,
                a_meta: tensor_store.TensorMeta,
                dtype: DType,
                partials: []f32,

                fn runTiles(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
                    const t: *@This() = @ptrCast(@alignCast(ctx_any));
                    if (start >= end) return;
                    if (tid >= t.partials.len) return;

                    const tc1: usize = t.a_meta.tile_counts[1];
                    var acc: f32 = 0.0;
                    var i: usize = start;
                    while (i < end) : (i += 1) {
                        const ti0: usize = i / tc1;
                        const ti1: usize = i - ti0 * tc1;
                        const tile = t.store.acquireTileConst(t.a_id, ti0, ti1) catch continue;
                        defer t.store.releaseConst(tile.token);
                        const v = tile.bufferView();
                        const n: usize = if (v.layout.rank == 1) v.layout.shape[0] else (v.layout.shape[0] * v.layout.shape[1]);
                        const part: f32 = switch (t.dtype) {
                            .f32 => reduce_k.sumF32Range(v.bytes, 0, n) catch 0.0,
                            .f16 => reduce_k.sumF16RangeToF32(v.bytes, 0, n) catch 0.0,
                            else => 0.0,
                        };
                        acc += part;
                    }
                    t.partials[tid] += acc;
                }
            };

            var task: Task = .{ .store = store, .a_id = a_id, .a_meta = a_meta, .dtype = a_meta.dtype, .partials = scratch_f32[0..thread_count] };
            const bytes_per_tile: usize = tileByteSize(a_meta);
            var grain: usize = if (bytes_per_tile == 0) 32 else @max(@as(usize, 1), (256 * 1024) / bytes_per_tile);
            if (grain > tile_total) grain = tile_total;
            p.parallelForAny(@ptrCast(&task), tile_total, grain, Task.runTiles);

            var i: usize = 0;
            while (i < thread_count) : (i += 1) {
                sum_f64 += @as(f64, scratch_f32[i]);
            }
            did_parallel = true;
        }
    }

    if (!did_parallel) {
        var ti0: usize = 0;
        while (ti0 < a_meta.tile_counts[0]) : (ti0 += 1) {
            var ti1: usize = 0;
            while (ti1 < a_meta.tile_counts[1]) : (ti1 += 1) {
                const tile = try store.acquireTileConst(a_id, ti0, ti1);
                defer store.releaseConst(tile.token);
                const v = tile.bufferView();
                const n: usize = elemCountFromTileView(v);
                const part: f32 = switch (a_meta.dtype) {
                    .f32 => try reduce_k.sumF32Range(v.bytes, 0, n),
                    .f16 => try reduce_k.sumF16RangeToF32(v.bytes, 0, n),
                    else => return BackendError.InvalidArgument,
                };
                sum_f64 += @as(f64, part);
            }
        }
    }

    var result_f64: f64 = sum_f64;
    if (op == .mean) {
        if (total_elems_u64 == 0) return BackendError.InvalidArgument;
        result_f64 /= @as(f64, @floatFromInt(total_elems_u64));
    }

    var out_tile = try store.acquireTileMut(out_id, 0, 0);
    defer store.releaseMut(out_tile.token);
    const out_view = out_tile.bufferView();

    switch (out_meta.dtype) {
        .f32 => @as(*align(1) f32, @ptrCast(out_view.bytes.ptr)).* = @floatCast(result_f64),
        .f16 => @as(*align(1) f16, @ptrCast(out_view.bytes.ptr)).* = @floatCast(@as(f32, @floatCast(result_f64))),
        else => return BackendError.InvalidArgument,
    }
}
