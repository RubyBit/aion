const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const reduce_k = @import("../kernels/reduce.zig");
const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");

const BackendError = types.BackendError;
const DType = types.DType;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

pub fn tileElemCount(meta: tensor_store.TensorMeta) usize {
    if (meta.rank == 0) return 1;
    var acc: usize = 1;
    var d: usize = 0;
    while (d < @as(usize, meta.rank)) : (d += 1) {
        acc *= meta.tile_shape[d];
    }
    return acc;
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
    if (view.layout.rank == 0) return 1;
    var acc: usize = 1;
    var d: usize = 0;
    while (d < @as(usize, view.layout.rank)) : (d += 1) {
        acc *= view.layout.shape[d];
    }
    return acc;
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
    if (meta.rank > 2) return BackendError.InvalidArgument;
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
    if (meta.rank > 2) return BackendError.InvalidArgument;
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
    if (dst_meta.rank != src_meta.rank) return BackendError.InvalidArgument;
    const elem_bytes: usize = try scalarElemBytes(dst_meta.dtype);

    if (dst_meta.rank > 2) {
        return retileCopyScalarND(store, dst_meta, src_meta, dst_id, src_id, elem_bytes);
    }

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

const MAX_RANK: usize = 8;

pub fn computePackedStrides(shape: []const usize, out: []usize) ExecuteProgramError!void {
    if (out.len != shape.len) return BackendError.InvalidArgument;
    if (shape.len == 0) return BackendError.InvalidArgument;

    var stride: usize = 1;
    var d: usize = shape.len;
    while (d > 0) : (d -= 1) {
        const idx: usize = d - 1;
        out[idx] = stride;
        stride = std.math.mul(usize, stride, shape[idx]) catch return BackendError.InvalidArgument;
    }
}

pub fn decodeLinearToCoords(linear: usize, strides: []const usize, shape: []const usize, out: []usize) ExecuteProgramError!void {
    if (out.len != shape.len or strides.len != shape.len) return BackendError.InvalidArgument;
    var rem: usize = linear;
    var d: usize = 0;
    while (d < shape.len) : (d += 1) {
        const stride: usize = strides[d];
        if (stride == 0) return BackendError.InvalidArgument;
        const v: usize = rem / stride;
        if (v >= shape[d]) return BackendError.InvalidArgument;
        out[d] = v;
        rem -= v * stride;
    }
}

const TileCacheConstND = struct {
    valid: bool = false,
    tensor_id: tensor_store.TensorId = 0,
    tile_index: usize = 0,
    tile: tensor_store.TileRefConst = undefined,
    strides: [MAX_RANK]usize = .{0} ** MAX_RANK,
};

const TileCacheMutND = struct {
    valid: bool = false,
    tensor_id: tensor_store.TensorId = 0,
    tile_index: usize = 0,
    tile: tensor_store.TileRefMut = undefined,
    strides: [MAX_RANK]usize = .{0} ** MAX_RANK,
};

fn ensureConstTileLinear(cache: *TileCacheConstND, store: tensor_store.TensorStore, id: tensor_store.TensorId, tile_index: usize, rank: usize) ExecuteProgramError!void {
    if (cache.valid and cache.tensor_id == id and cache.tile_index == tile_index) return;
    if (cache.valid) store.releaseConst(cache.tile.token);
    cache.tile = try store.acquireTileConstLinear(id, tile_index);
    cache.tensor_id = id;
    cache.tile_index = tile_index;
    cache.valid = true;
    const shape: []const usize = cache.tile.bufferView().layout.shape;
    try computePackedStrides(shape, cache.strides[0..rank]);
}

fn ensureMutTileLinear(cache: *TileCacheMutND, store: tensor_store.TensorStore, id: tensor_store.TensorId, tile_index: usize, rank: usize) ExecuteProgramError!void {
    if (cache.valid and cache.tensor_id == id and cache.tile_index == tile_index) return;
    if (cache.valid) store.releaseMut(cache.tile.token);
    cache.tile = try store.acquireTileMutLinear(id, tile_index);
    cache.tensor_id = id;
    cache.tile_index = tile_index;
    cache.valid = true;
    const shape: []const usize = cache.tile.bufferView().layout.shape;
    try computePackedStrides(shape, cache.strides[0..rank]);
}

fn retileCopyScalarND(
    store: tensor_store.TensorStore,
    dst_meta: tensor_store.TensorMeta,
    src_meta: tensor_store.TensorMeta,
    dst_id: tensor_store.TensorId,
    src_id: tensor_store.TensorId,
    elem_bytes: usize,
) ExecuteProgramError!void {
    const rank: usize = @as(usize, dst_meta.rank);
    if (rank == 0 or rank > MAX_RANK) return BackendError.InvalidArgument;

    var strides: [MAX_RANK]usize = undefined;
    try computePackedStrides(dst_meta.shape, strides[0..rank]);

    var coords: [MAX_RANK]usize = undefined;
    var src_tile_coords: [MAX_RANK]usize = undefined;
    var dst_tile_coords: [MAX_RANK]usize = undefined;
    var src_local_coords: [MAX_RANK]usize = undefined;
    var dst_local_coords: [MAX_RANK]usize = undefined;

    var src_cache: TileCacheConstND = .{};
    defer if (src_cache.valid) store.releaseConst(src_cache.tile.token);
    var dst_cache: TileCacheMutND = .{};
    defer if (dst_cache.valid) store.releaseMut(dst_cache.tile.token);

    var tmp: [4]u8 = .{ 0, 0, 0, 0 };
    const buf: []u8 = tmp[0..elem_bytes];

    const total_elems: usize = elemCountFromShape(dst_meta.shape) catch return BackendError.InvalidArgument;
    var lin: usize = 0;
    while (lin < total_elems) : (lin += 1) {
        try decodeLinearToCoords(lin, strides[0..rank], dst_meta.shape, coords[0..rank]);

        var d: usize = 0;
        while (d < rank) : (d += 1) {
            dst_tile_coords[d] = coords[d] / dst_meta.tile_shape[d];
            dst_local_coords[d] = coords[d] - dst_tile_coords[d] * dst_meta.tile_shape[d];

            src_tile_coords[d] = coords[d] / src_meta.tile_shape[d];
            src_local_coords[d] = coords[d] - src_tile_coords[d] * src_meta.tile_shape[d];
        }

        const dst_tile_index: usize = try tensor_store.encodeTileIndex(dst_meta, dst_tile_coords[0..rank]);
        const src_tile_index: usize = try tensor_store.encodeTileIndex(src_meta, src_tile_coords[0..rank]);

        try ensureConstTileLinear(&src_cache, store, src_id, src_tile_index, rank);
        try ensureMutTileLinear(&dst_cache, store, dst_id, dst_tile_index, rank);

        const src_strides: []const usize = src_cache.strides[0..rank];
        const dst_strides: []const usize = dst_cache.strides[0..rank];

        var src_lin: usize = 0;
        var dst_lin: usize = 0;
        d = 0;
        while (d < rank) : (d += 1) {
            src_lin = std.math.add(usize, src_lin, src_local_coords[d] * src_strides[d]) catch return BackendError.InvalidArgument;
            dst_lin = std.math.add(usize, dst_lin, dst_local_coords[d] * dst_strides[d]) catch return BackendError.InvalidArgument;
        }

        const src_off: usize = src_lin * elem_bytes;
        const dst_off: usize = dst_lin * elem_bytes;

        @memcpy(buf, src_cache.tile.bytes[src_off .. src_off + elem_bytes]);
        @memcpy(dst_cache.tile.bytes[dst_off .. dst_off + elem_bytes], buf);
    }
}

pub fn elemCountFromShape(shape: []const usize) ExecuteProgramError!usize {
    if (shape.len == 0) return BackendError.InvalidArgument;
    var acc: usize = 1;
    for (shape) |d| {
        acc = std.math.mul(usize, acc, d) catch return BackendError.InvalidArgument;
    }
    return acc;
}

pub fn reshapeCopyScalar(store: tensor_store.TensorStore, dst_id: tensor_store.TensorId, src_id: tensor_store.TensorId) ExecuteProgramError!void {
    const dst_meta = try store.meta(dst_id);
    const src_meta = try store.meta(src_id);
    if (dst_meta.dtype != src_meta.dtype) return BackendError.InvalidArgument;

    // Scalar dtypes only (v0).
    const elem_bytes: usize = try scalarElemBytes(dst_meta.dtype);

    const src_total: usize = try elemCountFromShape(src_meta.shape);
    const dst_total: usize = try elemCountFromShape(dst_meta.shape);
    if (src_total != dst_total) return BackendError.InvalidArgument;

    // Stream scalar elements in linear order across tiles.
    var src_tile_total: usize = 1;
    var d0: usize = 0;
    while (d0 < @as(usize, src_meta.rank)) : (d0 += 1) {
        src_tile_total = std.math.mul(usize, src_tile_total, src_meta.tile_counts[d0]) catch return BackendError.InvalidArgument;
    }
    var dst_tile_total: usize = 1;
    var d1: usize = 0;
    while (d1 < @as(usize, dst_meta.rank)) : (d1 += 1) {
        dst_tile_total = std.math.mul(usize, dst_tile_total, dst_meta.tile_counts[d1]) catch return BackendError.InvalidArgument;
    }
    if (src_tile_total == 0 or dst_tile_total == 0) return BackendError.InvalidArgument;

    var src_tile_index: usize = 0;
    var dst_tile_index: usize = 0;

    var src_tile = try store.acquireTileConstLinear(src_id, src_tile_index);
    defer store.releaseConst(src_tile.token);
    var dst_tile = try store.acquireTileMutLinear(dst_id, dst_tile_index);
    defer store.releaseMut(dst_tile.token);

    var src_view = src_tile.bufferView();
    var dst_view = dst_tile.bufferView();
    var src_n: usize = elemCountFromTileView(src_view);
    var dst_n: usize = elemCountFromTileView(dst_view);
    if (src_n * elem_bytes > src_view.bytes.len) return BackendError.InvalidArgument;
    if (dst_n * elem_bytes > dst_view.bytes.len) return BackendError.InvalidArgument;

    var src_pos: usize = 0;
    var dst_pos: usize = 0;
    var remaining: usize = dst_total;

    while (remaining > 0) {
        if (src_pos == src_n) {
            store.releaseConst(src_tile.token);
            src_tile_index += 1;
            if (src_tile_index >= src_tile_total) return BackendError.InvalidArgument;
            src_tile = try store.acquireTileConstLinear(src_id, src_tile_index);
            src_view = src_tile.bufferView();
            src_n = elemCountFromTileView(src_view);
            if (src_n * elem_bytes > src_view.bytes.len) return BackendError.InvalidArgument;
            src_pos = 0;
        }

        if (dst_pos == dst_n) {
            store.releaseMut(dst_tile.token);
            dst_tile_index += 1;
            if (dst_tile_index >= dst_tile_total) return BackendError.InvalidArgument;
            dst_tile = try store.acquireTileMutLinear(dst_id, dst_tile_index);
            dst_view = dst_tile.bufferView();
            dst_n = elemCountFromTileView(dst_view);
            if (dst_n * elem_bytes > dst_view.bytes.len) return BackendError.InvalidArgument;
            dst_pos = 0;
        }

        const src_avail: usize = src_n - src_pos;
        const dst_avail: usize = dst_n - dst_pos;
        var chunk: usize = if (src_avail < dst_avail) src_avail else dst_avail;
        if (chunk > remaining) chunk = remaining;

        const src_off: usize = src_pos * elem_bytes;
        const dst_off: usize = dst_pos * elem_bytes;
        const bytes: usize = chunk * elem_bytes;

        @memcpy(dst_view.bytes[dst_off .. dst_off + bytes], src_view.bytes[src_off .. src_off + bytes]);

        src_pos += chunk;
        dst_pos += chunk;
        remaining -= chunk;
    }
}

pub fn transpose2DCopyScalar(store: tensor_store.TensorStore, dst_id: tensor_store.TensorId, src_id: tensor_store.TensorId) ExecuteProgramError!void {
    const dst_meta = try store.meta(dst_id);
    const src_meta = try store.meta(src_id);
    if (dst_meta.rank > 2 or src_meta.rank > 2) return BackendError.InvalidArgument;

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

pub fn sliceNDCopyScalar(store: tensor_store.TensorStore, dst_id: tensor_store.TensorId, src_id: tensor_store.TensorId, starts: []const usize) ExecuteProgramError!void {
    const dst_meta = try store.meta(dst_id);
    const src_meta = try store.meta(src_id);

    if (dst_meta.dtype != src_meta.dtype) return BackendError.InvalidArgument;
    const rank: usize = @as(usize, dst_meta.rank);
    if (rank == 0 or rank != @as(usize, src_meta.rank) or rank > MAX_RANK) return BackendError.InvalidArgument;
    if (starts.len != rank) return BackendError.InvalidArgument;

    const elem_bytes: usize = try scalarElemBytes(dst_meta.dtype);

    var d: usize = 0;
    while (d < rank) : (d += 1) {
        if (starts[d] + dst_meta.shape[d] > src_meta.shape[d]) return BackendError.InvalidArgument;
        if (dst_meta.shape[d] == 0) return BackendError.InvalidArgument;
    }

    var dst_strides: [MAX_RANK]usize = .{0} ** MAX_RANK;
    try computePackedStrides(dst_meta.shape, dst_strides[0..rank]);

    var dst_coords: [MAX_RANK]usize = .{0} ** MAX_RANK;
    var src_coords: [MAX_RANK]usize = .{0} ** MAX_RANK;
    var dst_tile_coords: [MAX_RANK]usize = .{0} ** MAX_RANK;
    var dst_local_coords: [MAX_RANK]usize = .{0} ** MAX_RANK;
    var src_tile_coords: [MAX_RANK]usize = .{0} ** MAX_RANK;
    var src_local_coords: [MAX_RANK]usize = .{0} ** MAX_RANK;

    var src_cache: TileCacheConstND = .{};
    defer if (src_cache.valid) store.releaseConst(src_cache.tile.token);
    var dst_cache: TileCacheMutND = .{};
    defer if (dst_cache.valid) store.releaseMut(dst_cache.tile.token);

    const total_out: usize = try elemCountFromShape(dst_meta.shape);
    var lin: usize = 0;
    while (lin < total_out) : (lin += 1) {
        try decodeLinearToCoords(lin, dst_strides[0..rank], dst_meta.shape, dst_coords[0..rank]);

        d = 0;
        while (d < rank) : (d += 1) {
            src_coords[d] = starts[d] + dst_coords[d];

            dst_tile_coords[d] = dst_coords[d] / dst_meta.tile_shape[d];
            dst_local_coords[d] = dst_coords[d] - dst_tile_coords[d] * dst_meta.tile_shape[d];

            src_tile_coords[d] = src_coords[d] / src_meta.tile_shape[d];
            src_local_coords[d] = src_coords[d] - src_tile_coords[d] * src_meta.tile_shape[d];
        }

        const dst_tile_index: usize = try tensor_store.encodeTileIndex(dst_meta, dst_tile_coords[0..rank]);
        const src_tile_index: usize = try tensor_store.encodeTileIndex(src_meta, src_tile_coords[0..rank]);

        try ensureMutTileLinear(&dst_cache, store, dst_id, dst_tile_index, rank);
        try ensureConstTileLinear(&src_cache, store, src_id, src_tile_index, rank);

        var dst_local_lin: usize = 0;
        var src_local_lin: usize = 0;
        d = 0;
        while (d < rank) : (d += 1) {
            dst_local_lin = std.math.add(usize, dst_local_lin, dst_local_coords[d] * dst_cache.strides[d]) catch return BackendError.InvalidArgument;
            src_local_lin = std.math.add(usize, src_local_lin, src_local_coords[d] * src_cache.strides[d]) catch return BackendError.InvalidArgument;
        }

        const dst_off: usize = dst_local_lin * elem_bytes;
        const src_off: usize = src_local_lin * elem_bytes;
        @memcpy(dst_cache.tile.bytes[dst_off .. dst_off + elem_bytes], src_cache.tile.bytes[src_off .. src_off + elem_bytes]);
    }
}

pub fn concatScalar(step: executable.StepConcatScalar, store: tensor_store.TensorStore) ExecuteProgramError!void {
    const out_meta = try store.meta(step.out);
    const rank: usize = @as(usize, out_meta.rank);
    if (rank == 0 or rank > MAX_RANK) return BackendError.InvalidArgument;
    if (step.axis >= rank) return BackendError.InvalidArgument;

    const count: usize = @as(usize, step.input_count);
    if (count == 0) return BackendError.InvalidArgument;

    const elem_bytes: usize = try scalarElemBytes(out_meta.dtype);

    var out_coords: [MAX_RANK]usize = .{0} ** MAX_RANK;
    var in_coords: [MAX_RANK]usize = .{0} ** MAX_RANK;
    var out_tile_coords: [MAX_RANK]usize = .{0} ** MAX_RANK;
    var out_local_coords: [MAX_RANK]usize = .{0} ** MAX_RANK;
    var in_tile_coords: [MAX_RANK]usize = .{0} ** MAX_RANK;
    var in_local_coords: [MAX_RANK]usize = .{0} ** MAX_RANK;

    var in_axis_offsets: [16]usize = .{0} ** 16;
    var in_metas: [16]tensor_store.TensorMeta = undefined;
    var non_axis_dims: [MAX_RANK]usize = .{0} ** MAX_RANK;
    var non_axis_shape: [MAX_RANK]usize = .{0} ** MAX_RANK;
    var non_axis_strides: [MAX_RANK]usize = .{0} ** MAX_RANK;
    var non_axis_count: usize = 0;

    var axis_sum: usize = 0;
    var prefix_total: usize = 1;
    var d_init: usize = 0;
    while (d_init < rank) : (d_init += 1) {
        if (d_init == step.axis) continue;
        non_axis_dims[non_axis_count] = d_init;
        non_axis_shape[non_axis_count] = out_meta.shape[d_init];
        non_axis_count += 1;
        prefix_total = std.math.mul(usize, prefix_total, out_meta.shape[d_init]) catch return BackendError.InvalidArgument;
    }
    if (non_axis_count > 1) {
        try computePackedStrides(non_axis_shape[0..non_axis_count], non_axis_strides[0..non_axis_count]);
    }

    var i: usize = 0;
    while (i < count) : (i += 1) {
        const in_id: tensor_store.TensorId = step.inputs[i];
        const m = try store.meta(in_id);
        in_metas[i] = m;

        if (m.dtype != out_meta.dtype) return BackendError.InvalidArgument;
        if (@as(usize, m.rank) != rank) return BackendError.InvalidArgument;

        var d: usize = 0;
        while (d < rank) : (d += 1) {
            if (d == step.axis) continue;
            if (m.shape[d] != out_meta.shape[d]) return BackendError.InvalidArgument;
        }

        in_axis_offsets[i] = axis_sum;
        axis_sum = std.math.add(usize, axis_sum, m.shape[step.axis]) catch return BackendError.InvalidArgument;
    }
    if (axis_sum != out_meta.shape[step.axis]) return BackendError.InvalidArgument;

    var out_cache: TileCacheMutND = .{};
    defer if (out_cache.valid) store.releaseMut(out_cache.tile.token);
    var in_cache: TileCacheConstND = .{};
    defer if (in_cache.valid) store.releaseConst(in_cache.tile.token);

    var p: usize = 0;
    while (p < prefix_total) : (p += 1) {
        var rem: usize = p;
        if (non_axis_count > 0) {
            var nd: usize = 0;
            while (nd < non_axis_count) : (nd += 1) {
                const dim: usize = non_axis_dims[nd];
                const coord: usize = if (non_axis_count == 1) rem else rem / non_axis_strides[nd];
                if (coord >= out_meta.shape[dim]) return BackendError.InvalidArgument;
                out_coords[dim] = coord;
                in_coords[dim] = coord;
                if (non_axis_count != 1) rem -= coord * non_axis_strides[nd];
            }
        }

        var src_idx: usize = 0;
        while (src_idx < count) : (src_idx += 1) {
            const src_meta: tensor_store.TensorMeta = in_metas[src_idx];
            const src_axis_len: usize = src_meta.shape[step.axis];
            const dst_axis_base: usize = in_axis_offsets[src_idx];

            var dst_axis_pos: usize = dst_axis_base;
            var src_axis_pos: usize = 0;
            var left: usize = src_axis_len;
            while (left > 0) {
                out_coords[step.axis] = dst_axis_pos;
                in_coords[step.axis] = src_axis_pos;

                var d: usize = 0;
                while (d < rank) : (d += 1) {
                    out_tile_coords[d] = out_coords[d] / out_meta.tile_shape[d];
                    out_local_coords[d] = out_coords[d] - out_tile_coords[d] * out_meta.tile_shape[d];

                    in_tile_coords[d] = in_coords[d] / src_meta.tile_shape[d];
                    in_local_coords[d] = in_coords[d] - in_tile_coords[d] * src_meta.tile_shape[d];
                }

                const out_tile_index: usize = try tensor_store.encodeTileIndex(out_meta, out_tile_coords[0..rank]);
                const in_tile_index: usize = try tensor_store.encodeTileIndex(src_meta, in_tile_coords[0..rank]);

                try ensureMutTileLinear(&out_cache, store, step.out, out_tile_index, rank);
                try ensureConstTileLinear(&in_cache, store, step.inputs[src_idx], in_tile_index, rank);

                const out_view = out_cache.tile.bufferView();
                const in_view = in_cache.tile.bufferView();

                const out_run_max: usize = out_view.layout.shape[step.axis] - out_local_coords[step.axis];
                const in_run_max: usize = in_view.layout.shape[step.axis] - in_local_coords[step.axis];
                var run_len: usize = left;
                if (run_len > out_run_max) run_len = out_run_max;
                if (run_len > in_run_max) run_len = in_run_max;
                if (run_len == 0) return BackendError.InvalidArgument;

                var out_local_lin: usize = 0;
                var in_local_lin: usize = 0;
                d = 0;
                while (d < rank) : (d += 1) {
                    out_local_lin = std.math.add(usize, out_local_lin, out_local_coords[d] * out_cache.strides[d]) catch return BackendError.InvalidArgument;
                    in_local_lin = std.math.add(usize, in_local_lin, in_local_coords[d] * in_cache.strides[d]) catch return BackendError.InvalidArgument;
                }

                if (out_cache.strides[step.axis] == 1 and in_cache.strides[step.axis] == 1) {
                    const out_off: usize = out_local_lin * elem_bytes;
                    const in_off: usize = in_local_lin * elem_bytes;
                    const bytes: usize = run_len * elem_bytes;
                    @memcpy(out_cache.tile.bytes[out_off .. out_off + bytes], in_cache.tile.bytes[in_off .. in_off + bytes]);
                } else {
                    var r: usize = 0;
                    while (r < run_len) : (r += 1) {
                        const out_elem_lin: usize = out_local_lin + r * out_cache.strides[step.axis];
                        const in_elem_lin: usize = in_local_lin + r * in_cache.strides[step.axis];
                        const out_off: usize = out_elem_lin * elem_bytes;
                        const in_off: usize = in_elem_lin * elem_bytes;
                        @memcpy(out_cache.tile.bytes[out_off .. out_off + elem_bytes], in_cache.tile.bytes[in_off .. in_off + elem_bytes]);
                    }
                }

                dst_axis_pos += run_len;
                src_axis_pos += run_len;
                left -= run_len;
            }
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

    var total_elems_u64: u64 = 1;
    var d_total: usize = 0;
    while (d_total < @as(usize, a_meta.rank)) : (d_total += 1) {
        total_elems_u64 = std.math.mul(u64, total_elems_u64, @as(u64, @intCast(a_meta.shape[d_total]))) catch return BackendError.InvalidArgument;
    }

    var sum_f64: f64 = 0.0;
    var did_parallel: bool = false;

    if (pool) |p| {
        var tile_total: usize = 1;
        var d: usize = 0;
        while (d < @as(usize, a_meta.rank)) : (d += 1) {
            tile_total *= a_meta.tile_counts[d];
        }

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

                    var acc: f32 = 0.0;
                    var i: usize = start;
                    while (i < end) : (i += 1) {
                        const tile = t.store.acquireTileConstLinear(t.a_id, i) catch continue;
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
        var tile_total_seq: usize = 1;
        var d2: usize = 0;
        while (d2 < @as(usize, a_meta.rank)) : (d2 += 1) {
            tile_total_seq *= a_meta.tile_counts[d2];
        }

        var tile_index: usize = 0;
        while (tile_index < tile_total_seq) : (tile_index += 1) {
            const tile = try store.acquireTileConstLinear(a_id, tile_index);
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

    var result_f64: f64 = sum_f64;
    if (op == .mean) {
        if (total_elems_u64 == 0) return BackendError.InvalidArgument;
        result_f64 /= @as(f64, @floatFromInt(total_elems_u64));
    }

    const out_tile = try store.acquireTileMut(out_id, 0, 0);
    defer store.releaseMut(out_tile.token);
    const out_view = out_tile.bufferView();

    switch (out_meta.dtype) {
        .f32 => @as(*align(1) f32, @ptrCast(out_view.bytes.ptr)).* = @floatCast(result_f64),
        .f16 => @as(*align(1) f16, @ptrCast(out_view.bytes.ptr)).* = @floatCast(@as(f32, @floatCast(result_f64))),
        else => return BackendError.InvalidArgument,
    }
}

pub fn reduceAxisScalar(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    scratch_f32: []f32,
    op: types.ReduceOp,
    out_id: tensor_store.TensorId,
    a_id: tensor_store.TensorId,
    axis: usize,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    const out_meta = try store.meta(out_id);
    const a_meta = try store.meta(a_id);

    if (out_meta.dtype != a_meta.dtype) return BackendError.InvalidArgument;
    if (a_meta.rank == 0 or a_meta.rank > MAX_RANK) return BackendError.InvalidArgument;
    const in_rank: usize = @as(usize, a_meta.rank);
    if (axis >= in_rank) return BackendError.InvalidArgument;

    const out_rank_expected: usize = if (in_rank == 1) 1 else in_rank - 1;
    if (@as(usize, out_meta.rank) != out_rank_expected) return BackendError.InvalidArgument;

    const axis_len: usize = a_meta.shape[axis];
    if (axis_len == 0) return BackendError.InvalidArgument;

    const elem_bytes: usize = try scalarElemBytes(a_meta.dtype);

    var out_tile_total: usize = 1;
    var d_tot: usize = 0;
    while (d_tot < out_rank_expected) : (d_tot += 1) {
        out_tile_total = std.math.mul(usize, out_tile_total, out_meta.tile_counts[d_tot]) catch return BackendError.InvalidArgument;
    }

    var out_tile_count_strides: [MAX_RANK]usize = .{0} ** MAX_RANK;
    try computePackedStrides(out_meta.tile_counts[0..out_rank_expected], out_tile_count_strides[0..out_rank_expected]);

    const maybe_parallel: bool = pool != null and thread_count > 1 and scratch_f32.len >= thread_count and out_tile_total >= 2 and shouldParallelTiles(thread_count, out_tile_total, tileByteSize(out_meta), 256 * 1024);

    const Task = struct {
        store: tensor_store.TensorStore,
        op: types.ReduceOp,
        axis: usize,
        axis_len: usize,
        out_id: tensor_store.TensorId,
        a_id: tensor_store.TensorId,
        out_meta: tensor_store.TensorMeta,
        a_meta: tensor_store.TensorMeta,
        in_rank: usize,
        out_rank: usize,
        elem_bytes: usize,
        out_tile_count_strides: [MAX_RANK]usize,

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

            var out_tile_coords: [MAX_RANK]usize = .{0} ** MAX_RANK;
            var out_local_coords: [MAX_RANK]usize = .{0} ** MAX_RANK;
            var out_global_coords: [MAX_RANK]usize = .{0} ** MAX_RANK;
            var in_coords: [MAX_RANK]usize = .{0} ** MAX_RANK;
            var in_tile_coords: [MAX_RANK]usize = .{0} ** MAX_RANK;
            var in_local_coords: [MAX_RANK]usize = .{0} ** MAX_RANK;

            var in_cache: TileCacheConstND = .{};
            defer if (in_cache.valid) t.store.releaseConst(in_cache.tile.token);

            var tile_index: usize = start;
            while (tile_index < end) : (tile_index += 1) {
                if (t.stop.load(.acquire)) return;

                decodeLinearToCoords(tile_index, t.out_tile_count_strides[0..t.out_rank], t.out_meta.tile_counts[0..t.out_rank], out_tile_coords[0..t.out_rank]) catch |e| {
                    t.fail(e);
                    return;
                };

                const out_tile = t.store.acquireTileMutLinear(t.out_id, tile_index) catch |e| {
                    t.fail(e);
                    return;
                };
                defer t.store.releaseMut(out_tile.token);

                const out_view = out_tile.bufferView();
                const tile_shape: []const usize = out_view.layout.shape;

                var out_local_strides: [MAX_RANK]usize = .{0} ** MAX_RANK;
                computePackedStrides(tile_shape, out_local_strides[0..t.out_rank]) catch |e| {
                    t.fail(e);
                    return;
                };

                const tile_elems: usize = elemCountFromTileView(out_view);
                var local_lin: usize = 0;
                while (local_lin < tile_elems) : (local_lin += 1) {
                    decodeLinearToCoords(local_lin, out_local_strides[0..t.out_rank], tile_shape, out_local_coords[0..t.out_rank]) catch |e| {
                        t.fail(e);
                        return;
                    };

                    var od: usize = 0;
                    while (od < t.out_rank) : (od += 1) {
                        out_global_coords[od] = out_tile_coords[od] * t.out_meta.tile_shape[od] + out_local_coords[od];
                    }

                    if (t.in_rank == 1) {
                        in_coords[0] = 0;
                    } else {
                        var src_d: usize = 0;
                        var dst_d: usize = 0;
                        while (src_d < t.in_rank) : (src_d += 1) {
                            if (src_d == t.axis) continue;
                            in_coords[src_d] = out_global_coords[dst_d];
                            dst_d += 1;
                        }
                    }

                    var acc: f64 = 0.0;
                    var a_i: usize = 0;

                    // Fast path: if reducing over innermost axis, values within a tile segment
                    // are contiguous and can use SIMD range-reduction kernels.
                    if (t.axis + 1 == t.in_rank) {
                        while (a_i < t.axis_len) {
                            in_coords[t.axis] = a_i;

                            var d_in: usize = 0;
                            while (d_in < t.in_rank) : (d_in += 1) {
                                in_tile_coords[d_in] = in_coords[d_in] / t.a_meta.tile_shape[d_in];
                                in_local_coords[d_in] = in_coords[d_in] - in_tile_coords[d_in] * t.a_meta.tile_shape[d_in];
                            }

                            const in_tile_index: usize = tensor_store.encodeTileIndex(t.a_meta, in_tile_coords[0..t.in_rank]) catch |e| {
                                t.fail(e);
                                return;
                            };

                            ensureConstTileLinear(&in_cache, t.store, t.a_id, in_tile_index, t.in_rank) catch |e| {
                                t.fail(e);
                                return;
                            };

                            var in_local_lin: usize = 0;
                            var d_lin: usize = 0;
                            while (d_lin < t.in_rank) : (d_lin += 1) {
                                in_local_lin = std.math.add(usize, in_local_lin, in_local_coords[d_lin] * in_cache.strides[d_lin]) catch |e| {
                                    t.fail(e);
                                    return;
                                };
                            }

                            const run_max: usize = t.a_meta.tile_shape[t.axis] - in_local_coords[t.axis];
                            const remain: usize = t.axis_len - a_i;
                            const run_len: usize = if (run_max < remain) run_max else remain;

                            const seg_start: usize = in_local_lin;
                            const seg_end: usize = in_local_lin + run_len;
                            const part: f32 = switch (t.a_meta.dtype) {
                                .f32 => reduce_k.sumF32Range(in_cache.tile.bytes, seg_start, seg_end) catch |e| {
                                    t.fail(e);
                                    return;
                                },
                                .f16 => reduce_k.sumF16RangeToF32(in_cache.tile.bytes, seg_start, seg_end) catch |e| {
                                    t.fail(e);
                                    return;
                                },
                                else => {
                                    t.fail(BackendError.InvalidArgument);
                                    return;
                                },
                            };

                            acc += @as(f64, part);
                            a_i += run_len;
                        }
                    } else {
                        // Generic fallback for arbitrary axis: scalar gather.
                        while (a_i < t.axis_len) : (a_i += 1) {
                            in_coords[t.axis] = a_i;

                            var d_in: usize = 0;
                            while (d_in < t.in_rank) : (d_in += 1) {
                                in_tile_coords[d_in] = in_coords[d_in] / t.a_meta.tile_shape[d_in];
                                in_local_coords[d_in] = in_coords[d_in] - in_tile_coords[d_in] * t.a_meta.tile_shape[d_in];
                            }

                            const in_tile_index: usize = tensor_store.encodeTileIndex(t.a_meta, in_tile_coords[0..t.in_rank]) catch |e| {
                                t.fail(e);
                                return;
                            };

                            ensureConstTileLinear(&in_cache, t.store, t.a_id, in_tile_index, t.in_rank) catch |e| {
                                t.fail(e);
                                return;
                            };

                            var in_local_lin: usize = 0;
                            var d_lin: usize = 0;
                            while (d_lin < t.in_rank) : (d_lin += 1) {
                                in_local_lin = std.math.add(usize, in_local_lin, in_local_coords[d_lin] * in_cache.strides[d_lin]) catch |e| {
                                    t.fail(e);
                                    return;
                                };
                            }

                            const in_off: usize = in_local_lin * t.elem_bytes;
                            switch (t.a_meta.dtype) {
                                .f32 => {
                                    const v: f32 = @as(*align(1) const f32, @ptrCast(in_cache.tile.bytes[in_off .. in_off + 4].ptr)).*;
                                    acc += @as(f64, v);
                                },
                                .f16 => {
                                    const v: f16 = @as(*align(1) const f16, @ptrCast(in_cache.tile.bytes[in_off .. in_off + 2].ptr)).*;
                                    acc += @as(f64, @floatCast(v));
                                },
                                else => {
                                    t.fail(BackendError.InvalidArgument);
                                    return;
                                },
                            }
                        }
                    }

                    if (t.op == .mean) {
                        acc /= @as(f64, @floatFromInt(t.axis_len));
                    }

                    const out_off: usize = local_lin * t.elem_bytes;
                    switch (t.out_meta.dtype) {
                        .f32 => {
                            @as(*align(1) f32, @ptrCast(out_view.bytes[out_off .. out_off + 4].ptr)).* = @floatCast(acc);
                        },
                        .f16 => {
                            @as(*align(1) f16, @ptrCast(out_view.bytes[out_off .. out_off + 2].ptr)).* = @floatCast(@as(f32, @floatCast(acc)));
                        },
                        else => {
                            t.fail(BackendError.InvalidArgument);
                            return;
                        },
                    }
                }
            }
        }
    };

    var task: Task = .{
        .store = store,
        .op = op,
        .axis = axis,
        .axis_len = axis_len,
        .out_id = out_id,
        .a_id = a_id,
        .out_meta = out_meta,
        .a_meta = a_meta,
        .in_rank = in_rank,
        .out_rank = out_rank_expected,
        .elem_bytes = elem_bytes,
        .out_tile_count_strides = out_tile_count_strides,
    };

    if (maybe_parallel) {
        const p: *thread_pool.ThreadPool = pool.?;
        var grain: usize = @max(@as(usize, 1), out_tile_total / (thread_count * 4));
        if (grain > out_tile_total) grain = out_tile_total;
        p.parallelForAny(@ptrCast(&task), out_tile_total, grain, Task.runTiles);
        if (task.err_any) |e| return @errorCast(e);
        return;
    }

    Task.runTiles(@ptrCast(&task), 0, out_tile_total, 0);
    if (task.err_any) |e| return @errorCast(e);
}
