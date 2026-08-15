// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
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
        .i32 => elems * 4,
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
    // For small total byte volumes, avoid parallel launch overhead unless we have
    // enough tiles to keep a meaningful subset of workers busy.
    //
    // Previous logic (`tile_total >= 2`) was too eager on high-thread-count CPUs,
    // especially in decode-time graphs with many tiny ops, and could regress
    // throughput due to synchronization/scheduling overhead.
    const half_threads_ceil: usize = (thread_count + 1) / 2;
    const min_tiles_for_parallel: usize = @max(@as(usize, 2), @min(@as(usize, 8), half_threads_ceil));
    return tile_total >= min_tiles_for_parallel;
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
        .i8 => 1,
        .i32 => 4,
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

fn retileCopyScalar2D(
    store: tensor_store.TensorStore,
    dst_meta: tensor_store.TensorMeta,
    src_meta: tensor_store.TensorMeta,
    dst_id: tensor_store.TensorId,
    src_id: tensor_store.TensorId,
    elem_bytes: usize,
) ExecuteProgramError!void {
    if (dst_meta.rank != 2 or src_meta.rank != 2) return BackendError.InvalidArgument;
    if (dst_meta.shape[0] != src_meta.shape[0] or dst_meta.shape[1] != src_meta.shape[1]) return BackendError.InvalidArgument;

    const src_ts0: usize = src_meta.tile_shape[0];
    const src_ts1: usize = src_meta.tile_shape[1];
    const dst_ts0: usize = dst_meta.tile_shape[0];
    const dst_ts1: usize = dst_meta.tile_shape[1];
    if (src_ts0 == 0 or src_ts1 == 0 or dst_ts0 == 0 or dst_ts1 == 0) return BackendError.InvalidArgument;

    var src_cache: ScalarTileCacheConst = .{};
    defer if (src_cache.valid) store.releaseConst(src_cache.tile.token);
    var dst_cache: ScalarTileCacheMut = .{};
    defer if (dst_cache.valid) store.releaseMut(dst_cache.tile.token);

    const dst_tc0: usize = dst_meta.tile_counts[0];
    const dst_tc1: usize = dst_meta.tile_counts[1];
    if (dst_tc0 == 0 or dst_tc1 == 0) return BackendError.InvalidArgument;

    var dti0: usize = 0;
    while (dti0 < dst_tc0) : (dti0 += 1) {
        var dti1: usize = 0;
        while (dti1 < dst_tc1) : (dti1 += 1) {
            try ensureMutTile(&dst_cache, store, dst_id, dti0, dti1);

            const dst_m: usize = dst_cache.tile.shape_mem[0];
            const dst_n: usize = dst_cache.tile.shape_mem[1];
            if (dst_m == 0 or dst_n == 0) continue;

            const dst_r0: usize = dti0 * dst_ts0;
            const dst_c0: usize = dti1 * dst_ts1;

            var lr: usize = 0;
            while (lr < dst_m) : (lr += 1) {
                const r: usize = dst_r0 + lr;

                var lc: usize = 0;
                while (lc < dst_n) {
                    const c: usize = dst_c0 + lc;

                    const sti0: usize = r / src_ts0;
                    const sti1: usize = c / src_ts1;
                    try ensureConstTile(&src_cache, store, src_id, sti0, sti1);

                    const src_m: usize = src_cache.tile.shape_mem[0];
                    const src_n: usize = src_cache.tile.shape_mem[1];

                    const src_r0: usize = sti0 * src_ts0;
                    const src_c0: usize = sti1 * src_ts1;

                    const sr: usize = r - src_r0;
                    const sc: usize = c - src_c0;
                    if (sr >= src_m or sc >= src_n) return BackendError.InvalidArgument;

                    const src_cols_left: usize = src_n - sc;
                    const dst_cols_left: usize = dst_n - lc;
                    const run: usize = if (src_cols_left < dst_cols_left) src_cols_left else dst_cols_left;
                    if (run == 0) return BackendError.InvalidArgument;

                    const src_off: usize = (sr * src_n + sc) * elem_bytes;
                    const dst_off: usize = (lr * dst_n + lc) * elem_bytes;
                    const bytes: usize = run * elem_bytes;

                    if (src_off + bytes > src_cache.tile.bytes.len) return BackendError.InvalidArgument;
                    if (dst_off + bytes > dst_cache.tile.bytes.len) return BackendError.InvalidArgument;

                    @memcpy(dst_cache.tile.bytes[dst_off .. dst_off + bytes], src_cache.tile.bytes[src_off .. src_off + bytes]);
                    lc += run;
                }
            }
        }
    }
}

pub fn retileCopyScalar(store: tensor_store.TensorStore, dst_id: tensor_store.TensorId, src_id: tensor_store.TensorId) ExecuteProgramError!void {
    const dst_meta = try store.meta(dst_id);
    const src_meta = try store.meta(src_id);
    if (dst_meta.dtype != src_meta.dtype) return BackendError.InvalidArgument;
    if (dst_meta.rank != src_meta.rank) return BackendError.InvalidArgument;
    const rank: usize = @as(usize, dst_meta.rank);
    var d: usize = 0;
    while (d < rank) : (d += 1) {
        if (dst_meta.shape[d] != src_meta.shape[d]) return BackendError.InvalidArgument;
    }
    const elem_bytes: usize = try scalarElemBytes(dst_meta.dtype);

    // Rank-0: single scalar value.
    if (dst_meta.rank == 0) {
        const src_tile = try store.acquireTileConst(src_id, 0, 0);
        defer store.releaseConst(src_tile.token);
        const dst_tile = try store.acquireTileMut(dst_id, 0, 0);
        defer store.releaseMut(dst_tile.token);

        if (src_tile.bytes.len < elem_bytes) return BackendError.InvalidArgument;
        if (dst_tile.bytes.len < elem_bytes) return BackendError.InvalidArgument;
        @memcpy(dst_tile.bytes[0..elem_bytes], src_tile.bytes[0..elem_bytes]);
        return;
    }

    // Rank-1: tiling changes preserve linear order; stream copy across tiles.
    if (dst_meta.rank == 1) {
        return reshapeCopyScalar(store, dst_id, src_id);
    }

    // Rank-2: copy by row segments, mapping across tile grids.
    if (dst_meta.rank == 2) {
        return retileCopyScalar2D(store, dst_meta, src_meta, dst_id, src_id, elem_bytes);
    }

    // Rank>2: fallback to coordinate-based copier (guarded by compile-time policy).
    return retileCopyScalarND(store, dst_meta, src_meta, dst_id, src_id, elem_bytes);
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

pub const TileCacheConstND = struct {
    valid: bool = false,
    tensor_id: tensor_store.TensorId = 0,
    tile_index: usize = 0,
    tile: tensor_store.TileRefConst = undefined,
    strides: [MAX_RANK]usize = @splat(0),

    pub fn deinit(self: *TileCacheConstND, store: tensor_store.TensorStore) void {
        if (self.valid) {
            store.releaseConst(self.tile.token);
            self.valid = false;
        }
    }
};

pub const TileCacheMutND = struct {
    valid: bool = false,
    tensor_id: tensor_store.TensorId = 0,
    tile_index: usize = 0,
    tile: tensor_store.TileRefMut = undefined,
    strides: [MAX_RANK]usize = @splat(0),

    pub fn deinit(self: *TileCacheMutND, store: tensor_store.TensorStore) void {
        if (self.valid) {
            store.releaseMut(self.tile.token);
            self.valid = false;
        }
    }
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

/// Packed in-tile element offset for logical `coords` given the tile's element
/// `strides`. Shared by the scalar read/write helpers below; assumes the caller
/// has already resolved (and cached) the enclosing tile.
fn scalarElemOffset(
    meta: tensor_store.TensorMeta,
    coords: []const usize,
    strides: []const usize,
) ExecuteProgramError!usize {
    const rank: usize = @as(usize, meta.rank);
    var off_elems: usize = 0;
    var d: usize = 0;
    while (d < rank) : (d += 1) {
        const local: usize = coords[d] - (coords[d] / meta.tile_shape[d]) * meta.tile_shape[d];
        off_elems = std.math.add(usize, off_elems, local * strides[d]) catch return BackendError.InvalidArgument;
    }
    return off_elems;
}

/// Read one f32 (widening f16) at logical `coords` of tensor `id`. Valid for any
/// rank up to MAX_RANK and any tiling; the cache amortizes repeated accesses
/// within the same tile. Layout-independent (slow) per-element path — for the
/// common single-tile contiguous case, index the raw buffer directly instead.
pub fn readScalarF32At(
    store: tensor_store.TensorStore,
    meta: tensor_store.TensorMeta,
    id: tensor_store.TensorId,
    coords: []const usize,
    cache: *TileCacheConstND,
) ExecuteProgramError!f32 {
    const rank: usize = @as(usize, meta.rank);
    if (coords.len != rank) return BackendError.InvalidArgument;
    if (rank == 0 or rank > MAX_RANK) return BackendError.InvalidArgument;

    var tile_coords: [MAX_RANK]usize = @splat(0);

    var d: usize = 0;
    while (d < rank) : (d += 1) {
        if (coords[d] >= meta.shape[d]) return BackendError.InvalidArgument;
        tile_coords[d] = coords[d] / meta.tile_shape[d];
    }
    const tile_index: usize = try tensor_store.encodeTileIndex(meta, tile_coords[0..rank]);
    try ensureConstTileLinear(cache, store, id, tile_index, rank);

    const v = cache.tile.bufferView();
    if (v.dtype != meta.dtype) return BackendError.InvalidArgument;
    if (@as(usize, v.layout.rank) != rank) return BackendError.InvalidArgument;

    const off_elems: usize = try scalarElemOffset(meta, coords, cache.strides[0..rank]);

    switch (meta.dtype) {
        .f32 => {
            const off_bytes: usize = off_elems * 4;
            if (off_bytes + 4 > v.bytes.len) return BackendError.InvalidArgument;
            return @as(*align(1) const f32, @ptrCast(v.bytes.ptr + off_bytes)).*;
        },
        .f16 => {
            const off_bytes: usize = off_elems * 2;
            if (off_bytes + 2 > v.bytes.len) return BackendError.InvalidArgument;
            const x: f16 = @as(*align(1) const f16, @ptrCast(v.bytes.ptr + off_bytes)).*;
            return @floatCast(x);
        },
        else => return BackendError.InvalidArgument,
    }
}

/// Write one f32 (narrowing to f16 if needed) at logical `coords` of tensor `id`.
/// See `readScalarF32At` for layout/perf notes.
pub fn writeScalarFromF32At(
    store: tensor_store.TensorStore,
    meta: tensor_store.TensorMeta,
    id: tensor_store.TensorId,
    coords: []const usize,
    v_in: f32,
    cache: *TileCacheMutND,
) ExecuteProgramError!void {
    const rank: usize = @as(usize, meta.rank);
    if (coords.len != rank) return BackendError.InvalidArgument;
    if (rank == 0 or rank > MAX_RANK) return BackendError.InvalidArgument;

    var tile_coords: [MAX_RANK]usize = @splat(0);

    var d: usize = 0;
    while (d < rank) : (d += 1) {
        if (coords[d] >= meta.shape[d]) return BackendError.InvalidArgument;
        tile_coords[d] = coords[d] / meta.tile_shape[d];
    }
    const tile_index: usize = try tensor_store.encodeTileIndex(meta, tile_coords[0..rank]);
    try ensureMutTileLinear(cache, store, id, tile_index, rank);

    const v = cache.tile.bufferView();
    if (v.dtype != meta.dtype) return BackendError.InvalidArgument;
    if (@as(usize, v.layout.rank) != rank) return BackendError.InvalidArgument;

    const off_elems: usize = try scalarElemOffset(meta, coords, cache.strides[0..rank]);

    switch (meta.dtype) {
        .f32 => {
            const off_bytes: usize = off_elems * 4;
            if (off_bytes + 4 > v.bytes.len) return BackendError.InvalidArgument;
            @as(*align(1) f32, @ptrCast(v.bytes.ptr + off_bytes)).* = v_in;
        },
        .f16 => {
            const off_bytes: usize = off_elems * 2;
            if (off_bytes + 2 > v.bytes.len) return BackendError.InvalidArgument;
            @as(*align(1) f16, @ptrCast(v.bytes.ptr + off_bytes)).* = @floatCast(v_in);
        },
        else => return BackendError.InvalidArgument,
    }
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

    // Fast ND retile: stream contiguous runs along the last dimension.
    //
    // For any fixed coordinate of dims [0..rank-2], the last dim forms a contiguous
    // row in packed row-major layout (stride=1). We copy that row in segments that
    // do not cross source or destination tile boundaries.
    if (rank < 2) return BackendError.InvalidArgument;

    const last_dim: usize = rank - 1;
    const row_rank: usize = last_dim; // dims excluding the last
    const row_shape: []const usize = dst_meta.shape[0..row_rank];
    if (row_rank == 0) return BackendError.InvalidArgument;

    var row_strides: [MAX_RANK]usize = undefined;
    try computePackedStrides(row_shape, row_strides[0..row_rank]);

    var row_coords: [MAX_RANK]usize = undefined;
    var src_tile_coords: [MAX_RANK]usize = undefined;
    var dst_tile_coords: [MAX_RANK]usize = undefined;
    var src_local_coords: [MAX_RANK]usize = undefined;
    var dst_local_coords: [MAX_RANK]usize = undefined;

    var src_cache: TileCacheConstND = .{};
    defer if (src_cache.valid) store.releaseConst(src_cache.tile.token);
    var dst_cache: TileCacheMutND = .{};
    defer if (dst_cache.valid) store.releaseMut(dst_cache.tile.token);

    const row_total: usize = elemCountFromShape(row_shape) catch return BackendError.InvalidArgument;
    const col_total: usize = dst_meta.shape[last_dim];

    var row_lin: usize = 0;
    while (row_lin < row_total) : (row_lin += 1) {
        try decodeLinearToCoords(row_lin, row_strides[0..row_rank], row_shape, row_coords[0..row_rank]);

        // Tile/local coords for all dims except last.
        var d: usize = 0;
        while (d < row_rank) : (d += 1) {
            dst_tile_coords[d] = row_coords[d] / dst_meta.tile_shape[d];
            dst_local_coords[d] = row_coords[d] - dst_tile_coords[d] * dst_meta.tile_shape[d];

            src_tile_coords[d] = row_coords[d] / src_meta.tile_shape[d];
            src_local_coords[d] = row_coords[d] - src_tile_coords[d] * src_meta.tile_shape[d];
        }

        var col: usize = 0;
        while (col < col_total) {
            dst_tile_coords[last_dim] = col / dst_meta.tile_shape[last_dim];
            dst_local_coords[last_dim] = col - dst_tile_coords[last_dim] * dst_meta.tile_shape[last_dim];

            src_tile_coords[last_dim] = col / src_meta.tile_shape[last_dim];
            src_local_coords[last_dim] = col - src_tile_coords[last_dim] * src_meta.tile_shape[last_dim];

            const dst_tile_index: usize = try tensor_store.encodeTileIndex(dst_meta, dst_tile_coords[0..rank]);
            const src_tile_index: usize = try tensor_store.encodeTileIndex(src_meta, src_tile_coords[0..rank]);

            try ensureConstTileLinear(&src_cache, store, src_id, src_tile_index, rank);
            try ensureMutTileLinear(&dst_cache, store, dst_id, dst_tile_index, rank);

            const src_tile_cols: usize = src_cache.tile.shape_mem[last_dim];
            const dst_tile_cols: usize = dst_cache.tile.shape_mem[last_dim];
            if (src_tile_cols == 0 or dst_tile_cols == 0) return BackendError.InvalidArgument;
            if (src_local_coords[last_dim] >= src_tile_cols) return BackendError.InvalidArgument;
            if (dst_local_coords[last_dim] >= dst_tile_cols) return BackendError.InvalidArgument;

            const src_left: usize = src_tile_cols - src_local_coords[last_dim];
            const dst_left: usize = dst_tile_cols - dst_local_coords[last_dim];
            var run: usize = if (src_left < dst_left) src_left else dst_left;
            const total_left: usize = col_total - col;
            if (run > total_left) run = total_left;
            if (run == 0) return BackendError.InvalidArgument;

            const src_strides: []const usize = src_cache.strides[0..rank];
            const dst_strides: []const usize = dst_cache.strides[0..rank];

            // Base linear index for this row within the current tile (excluding last dim).
            var src_base: usize = 0;
            var dst_base: usize = 0;
            d = 0;
            while (d < row_rank) : (d += 1) {
                src_base = std.math.add(usize, src_base, src_local_coords[d] * src_strides[d]) catch return BackendError.InvalidArgument;
                dst_base = std.math.add(usize, dst_base, dst_local_coords[d] * dst_strides[d]) catch return BackendError.InvalidArgument;
            }

            const src_lin: usize = std.math.add(usize, src_base, src_local_coords[last_dim] * src_strides[last_dim]) catch return BackendError.InvalidArgument;
            const dst_lin: usize = std.math.add(usize, dst_base, dst_local_coords[last_dim] * dst_strides[last_dim]) catch return BackendError.InvalidArgument;

            const src_off: usize = src_lin * elem_bytes;
            const dst_off: usize = dst_lin * elem_bytes;
            const bytes: usize = run * elem_bytes;
            if (src_off + bytes > src_cache.tile.bytes.len) return BackendError.InvalidArgument;
            if (dst_off + bytes > dst_cache.tile.bytes.len) return BackendError.InvalidArgument;

            @memcpy(dst_cache.tile.bytes[dst_off .. dst_off + bytes], src_cache.tile.bytes[src_off .. src_off + bytes]);
            col += run;
        }
    }
}

test "retileCopyScalarND: rank-3 roundtrips packed values" {
    const manager_mod = @import("../../../storage/manager.zig");
    const testing = std.testing;

    var sm = manager_mod.StorageManager.init(testing.allocator);
    defer sm.deinit();

    const shape = [_]usize{ 2, 3, 17 };
    const src_tile = [_]usize{ 1, 3, 5 };
    const dst_tile = [_]usize{ 2, 1, 7 };

    const src_id = try sm.createTiledTensor(.f32, &shape, &src_tile, .{ .tile_alignment = 64 });
    const dst_id = try sm.createTiledTensor(.f32, &shape, &dst_tile, .{ .tile_alignment = 64 });

    const total: usize = shape[0] * shape[1] * shape[2];
    const bytes_len: usize = total * @sizeOf(f32);
    const buf = try testing.allocator.alloc(u8, bytes_len);
    defer testing.allocator.free(buf);
    const vals_ptr: [*]align(1) f32 = @ptrCast(buf.ptr);
    const vals: []align(1) f32 = vals_ptr[0 .. bytes_len / @sizeOf(f32)];
    for (vals, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 17)) * 0.125;
    }
    try sm.writeFromPackedScalar(src_id, buf);

    try retileCopyScalar(sm.tensorStore(), dst_id, src_id);

    const out = try testing.allocator.alloc(u8, bytes_len);
    defer testing.allocator.free(out);
    try sm.readToPackedScalar(dst_id, out);
    try testing.expectEqualSlices(u8, buf, out);
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

    // Fast path: if both tensors are single-tile, packed payload is contiguous
    // row-major for each side, so reshape is a plain contiguous memcpy.
    //
    // This keeps the general row-major-correct ND implementation below for
    // multi-tile cases (where tile-major byte order can differ from row-major),
    // while recovering the previous low overhead for common tiny tensors.
    var src_tile_total: usize = 1;
    var dst_tile_total: usize = 1;
    var td: usize = 0;
    while (td < @as(usize, src_meta.rank)) : (td += 1) {
        src_tile_total = std.math.mul(usize, src_tile_total, src_meta.tile_counts[td]) catch return BackendError.InvalidArgument;
    }
    td = 0;
    while (td < @as(usize, dst_meta.rank)) : (td += 1) {
        dst_tile_total = std.math.mul(usize, dst_tile_total, dst_meta.tile_counts[td]) catch return BackendError.InvalidArgument;
    }

    if (src_tile_total == 1 and dst_tile_total == 1) {
        const src_tile = try store.acquireTileConstLinear(src_id, 0);
        defer store.releaseConst(src_tile.token);
        const dst_tile = try store.acquireTileMutLinear(dst_id, 0);
        defer store.releaseMut(dst_tile.token);

        const total_bytes: usize = std.math.mul(usize, dst_total, elem_bytes) catch return BackendError.InvalidArgument;
        if (src_tile.bytes.len < total_bytes) return BackendError.InvalidArgument;
        if (dst_tile.bytes.len < total_bytes) return BackendError.InvalidArgument;

        @memcpy(dst_tile.bytes[0..total_bytes], src_tile.bytes[0..total_bytes]);
        return;
    }

    // Reshape must preserve *row-major element order*.
    //
    // IMPORTANT: tiled tensors are stored as packed tiles, so streaming across
    // tile bytes in linear tile-index order does NOT necessarily match the
    // logical packed row-major order of the full tensor (unless tiles span the
    // full trailing dims). For example, a 2x4 matrix with 2x2 tiles packs to:
    //   [a00,a01,a10,a11, a02,a03,a12,a13] (tile-major)
    // but packed row-major order is:
    //   [a00,a01,a02,a03, a10,a11,a12,a13]
    //
    // So we copy in row-major linear order by mapping current coordinates to
    // per-tile offsets, and memcpy contiguous runs along the last dim.

    const src_rank: usize = @as(usize, src_meta.rank);
    const dst_rank: usize = @as(usize, dst_meta.rank);
    if (src_rank == 0 or src_rank > MAX_RANK) return BackendError.InvalidArgument;
    if (dst_rank == 0 or dst_rank > MAX_RANK) return BackendError.InvalidArgument;

    const src_last_dim: usize = src_rank - 1;
    const dst_last_dim: usize = dst_rank - 1;

    var src_coords: [MAX_RANK]usize = @splat(0);
    var dst_coords: [MAX_RANK]usize = @splat(0);

    var src_tile_coords: [MAX_RANK]usize = undefined;
    var dst_tile_coords: [MAX_RANK]usize = undefined;
    var src_local_coords: [MAX_RANK]usize = undefined;
    var dst_local_coords: [MAX_RANK]usize = undefined;

    var src_cache: TileCacheConstND = .{};
    defer if (src_cache.valid) store.releaseConst(src_cache.tile.token);
    var dst_cache: TileCacheMutND = .{};
    defer if (dst_cache.valid) store.releaseMut(dst_cache.tile.token);

    var remaining: usize = dst_total;
    while (remaining > 0) {
        // Compute tile/local coords for src.
        var d0: usize = 0;
        while (d0 < src_rank) : (d0 += 1) {
            src_tile_coords[d0] = src_coords[d0] / src_meta.tile_shape[d0];
            src_local_coords[d0] = src_coords[d0] - (src_tile_coords[d0] * src_meta.tile_shape[d0]);
        }
        const src_tile_index: usize = try tensor_store.encodeTileIndex(src_meta, src_tile_coords[0..src_rank]);
        try ensureConstTileLinear(&src_cache, store, src_id, src_tile_index, src_rank);

        // Compute tile/local coords for dst.
        var d1: usize = 0;
        while (d1 < dst_rank) : (d1 += 1) {
            dst_tile_coords[d1] = dst_coords[d1] / dst_meta.tile_shape[d1];
            dst_local_coords[d1] = dst_coords[d1] - (dst_tile_coords[d1] * dst_meta.tile_shape[d1]);
        }
        const dst_tile_index: usize = try tensor_store.encodeTileIndex(dst_meta, dst_tile_coords[0..dst_rank]);
        try ensureMutTileLinear(&dst_cache, store, dst_id, dst_tile_index, dst_rank);

        // Limit run so we don't cross either tensor's row boundary (carry) or
        // either tile boundary along the last dim.
        const src_row_left: usize = src_meta.shape[src_last_dim] - src_coords[src_last_dim];
        const dst_row_left: usize = dst_meta.shape[dst_last_dim] - dst_coords[dst_last_dim];

        const src_tile_cols: usize = src_cache.tile.shape_mem[src_last_dim];
        const dst_tile_cols: usize = dst_cache.tile.shape_mem[dst_last_dim];
        if (src_tile_cols == 0 or dst_tile_cols == 0) return BackendError.InvalidArgument;
        if (src_local_coords[src_last_dim] >= src_tile_cols) return BackendError.InvalidArgument;
        if (dst_local_coords[dst_last_dim] >= dst_tile_cols) return BackendError.InvalidArgument;

        const src_tile_left: usize = src_tile_cols - src_local_coords[src_last_dim];
        const dst_tile_left: usize = dst_tile_cols - dst_local_coords[dst_last_dim];

        var run: usize = src_row_left;
        if (dst_row_left < run) run = dst_row_left;
        if (src_tile_left < run) run = src_tile_left;
        if (dst_tile_left < run) run = dst_tile_left;
        if (run > remaining) run = remaining;
        if (run == 0) return BackendError.InvalidArgument;

        const src_strides: []const usize = src_cache.strides[0..src_rank];
        const dst_strides: []const usize = dst_cache.strides[0..dst_rank];

        var src_base: usize = 0;
        var dst_base: usize = 0;

        d0 = 0;
        while (d0 < src_last_dim) : (d0 += 1) {
            src_base = std.math.add(usize, src_base, src_local_coords[d0] * src_strides[d0]) catch return BackendError.InvalidArgument;
        }
        d1 = 0;
        while (d1 < dst_last_dim) : (d1 += 1) {
            dst_base = std.math.add(usize, dst_base, dst_local_coords[d1] * dst_strides[d1]) catch return BackendError.InvalidArgument;
        }

        const src_lin: usize = std.math.add(usize, src_base, src_local_coords[src_last_dim] * src_strides[src_last_dim]) catch return BackendError.InvalidArgument;
        const dst_lin: usize = std.math.add(usize, dst_base, dst_local_coords[dst_last_dim] * dst_strides[dst_last_dim]) catch return BackendError.InvalidArgument;

        const src_off: usize = src_lin * elem_bytes;
        const dst_off: usize = dst_lin * elem_bytes;
        const bytes: usize = run * elem_bytes;
        if (src_off + bytes > src_cache.tile.bytes.len) return BackendError.InvalidArgument;
        if (dst_off + bytes > dst_cache.tile.bytes.len) return BackendError.InvalidArgument;

        @memcpy(dst_cache.tile.bytes[dst_off .. dst_off + bytes], src_cache.tile.bytes[src_off .. src_off + bytes]);

        // Advance coords by `run` along each tensor's last dim (with carry by 1
        // when we hit the end of the row).
        src_coords[src_last_dim] += run;
        if (src_coords[src_last_dim] == src_meta.shape[src_last_dim]) {
            src_coords[src_last_dim] = 0;
            var cd: usize = src_last_dim;
            while (cd > 0) {
                cd -= 1;
                src_coords[cd] += 1;
                if (src_coords[cd] < src_meta.shape[cd]) break;
                src_coords[cd] = 0;
            }
        }

        dst_coords[dst_last_dim] += run;
        if (dst_coords[dst_last_dim] == dst_meta.shape[dst_last_dim]) {
            dst_coords[dst_last_dim] = 0;
            var cd2: usize = dst_last_dim;
            while (cd2 > 0) {
                cd2 -= 1;
                dst_coords[cd2] += 1;
                if (dst_coords[cd2] < dst_meta.shape[cd2]) break;
                dst_coords[cd2] = 0;
            }
        }

        remaining -= run;
    }
}

test "reshapeCopyScalar: preserves packed row-major order across tiled tensors" {
    const manager_mod = @import("../../../storage/manager.zig");
    const testing = std.testing;

    var sm = manager_mod.StorageManager.init(testing.allocator);
    defer sm.deinit();

    // A 2x4 tensor with 2x2 tiling is packed tile-major in memory, so a naive
    // linear tile-stream reshape will reorder elements. This test ensures
    // reshapeCopyScalar uses logical packed row-major order.
    const src_shape = [_]usize{ 2, 4 };
    const src_tile = [_]usize{ 2, 2 };
    const dst_shape = [_]usize{ 4, 2 };
    const dst_tile = [_]usize{ 2, 2 };

    const src_id = try sm.createTiledTensor(.f32, &src_shape, &src_tile, .{ .tile_alignment = 64 });
    const dst_id = try sm.createTiledTensor(.f32, &dst_shape, &dst_tile, .{ .tile_alignment = 64 });

    var vals: [8]f32 = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
    try sm.writeFromPackedScalar(src_id, std.mem.sliceAsBytes(vals[0..]));

    try reshapeCopyScalar(sm.tensorStore(), dst_id, src_id);

    var out: [8]f32 = undefined;
    try sm.readToPackedScalar(dst_id, std.mem.sliceAsBytes(out[0..]));
    try testing.expectEqualSlices(f32, vals[0..], out[0..]);
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

    // Find the largest trailing run of dims that are FULLY copied (starts==0,
    // dst.shape==src.shape) AND single-tile in both tensors. Such dims are the
    // innermost packed dims of the tile (tile strides are packed over the tile's
    // shape), so for each outer coordinate the whole run is one contiguous block in
    // both src and dst tiles — copy it with a single memcpy instead of per element.
    var run_dims: usize = 0;
    var run_len: usize = 1;
    {
        var dd: usize = rank;
        while (dd > 0) : (dd -= 1) {
            const d2: usize = dd - 1;
            const full_copy: bool = (starts[d2] == 0 and dst_meta.shape[d2] == src_meta.shape[d2]);
            const single_tile: bool = (dst_meta.tile_shape[d2] >= dst_meta.shape[d2] and src_meta.tile_shape[d2] >= src_meta.shape[d2]);
            if (full_copy and single_tile) {
                run_len *= dst_meta.shape[d2];
                run_dims += 1;
            } else break;
        }
    }
    const outer_dims: usize = rank - run_dims;

    var outer_shape: [MAX_RANK]usize = @splat(0);
    d = 0;
    while (d < outer_dims) : (d += 1) outer_shape[d] = dst_meta.shape[d];
    var outer_strides: [MAX_RANK]usize = @splat(0);
    if (outer_dims > 0) try computePackedStrides(outer_shape[0..outer_dims], outer_strides[0..outer_dims]);

    var dst_coords: [MAX_RANK]usize = @splat(0);
    var src_coords: [MAX_RANK]usize = @splat(0);
    var dst_tile_coords: [MAX_RANK]usize = @splat(0);
    var dst_local_coords: [MAX_RANK]usize = @splat(0);
    var src_tile_coords: [MAX_RANK]usize = @splat(0);
    var src_local_coords: [MAX_RANK]usize = @splat(0);

    var src_cache: TileCacheConstND = .{};
    defer if (src_cache.valid) store.releaseConst(src_cache.tile.token);
    var dst_cache: TileCacheMutND = .{};
    defer if (dst_cache.valid) store.releaseMut(dst_cache.tile.token);

    const run_bytes: usize = run_len * elem_bytes;
    const outer_total: usize = if (outer_dims == 0) 1 else try elemCountFromShape(outer_shape[0..outer_dims]);

    // Fast path for the no-trailing-run case (e.g. a slice along the last dim, or a
    // fully-copied-but-multi-tile last dim). The generic loop below would degrade to a
    // 1-element memcpy with full tile-coordinate math *per element*. Instead, walk the
    // last dim in tile-contiguous chunks (stride-1 within a tile), copying up to a whole
    // in-tile run per memcpy — ~tile_shape[last]x fewer iterations. Correct for any
    // `starts` (aligned or not): the chunk is bounded by the distance to the next tile
    // boundary in both src and dst.
    if (run_dims == 0 and rank >= 1) {
        const last: usize = rank - 1;
        const o_dims: usize = rank - 1; // outer = all dims except the last
        var o_strides: [MAX_RANK]usize = @splat(0);
        if (o_dims > 0) try computePackedStrides(dst_meta.shape[0..o_dims], o_strides[0..o_dims]);
        const o_total: usize = if (o_dims == 0) 1 else try elemCountFromShape(dst_meta.shape[0..o_dims]);
        const n_last: usize = dst_meta.shape[last];

        var ob: usize = 0;
        while (ob < o_total) : (ob += 1) {
            if (o_dims > 0) try decodeLinearToCoords(ob, o_strides[0..o_dims], dst_meta.shape[0..o_dims], dst_coords[0..o_dims]);
            var c: usize = 0;
            while (c < n_last) {
                dst_coords[last] = c;
                d = 0;
                while (d < rank) : (d += 1) {
                    src_coords[d] = starts[d] + dst_coords[d];
                    dst_tile_coords[d] = dst_coords[d] / dst_meta.tile_shape[d];
                    dst_local_coords[d] = dst_coords[d] - dst_tile_coords[d] * dst_meta.tile_shape[d];
                    src_tile_coords[d] = src_coords[d] / src_meta.tile_shape[d];
                    src_local_coords[d] = src_coords[d] - src_tile_coords[d] * src_meta.tile_shape[d];
                }

                // Max contiguous run: to the next tile boundary in both tensors, capped
                // by the elements left in the slice.
                var chunk: usize = n_last - c;
                const dst_room: usize = dst_meta.tile_shape[last] - dst_local_coords[last];
                const src_room: usize = src_meta.tile_shape[last] - src_local_coords[last];
                if (dst_room < chunk) chunk = dst_room;
                if (src_room < chunk) chunk = src_room;

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
                const nbytes: usize = chunk * elem_bytes;
                @memcpy(dst_cache.tile.bytes[dst_off .. dst_off + nbytes], src_cache.tile.bytes[src_off .. src_off + nbytes]);
                c += chunk;
            }
        }
        return;
    }

    var blk: usize = 0;
    while (blk < outer_total) : (blk += 1) {
        if (outer_dims > 0) try decodeLinearToCoords(blk, outer_strides[0..outer_dims], outer_shape[0..outer_dims], dst_coords[0..outer_dims]);

        d = 0;
        while (d < rank) : (d += 1) {
            const dc: usize = if (d < outer_dims) dst_coords[d] else 0; // run dims start at 0
            dst_coords[d] = dc;
            src_coords[d] = starts[d] + dc;
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
        @memcpy(dst_cache.tile.bytes[dst_off .. dst_off + run_bytes], src_cache.tile.bytes[src_off .. src_off + run_bytes]);
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

    var out_coords: [MAX_RANK]usize = @splat(0);
    var in_coords: [MAX_RANK]usize = @splat(0);
    var out_tile_coords: [MAX_RANK]usize = @splat(0);
    var out_local_coords: [MAX_RANK]usize = @splat(0);
    var in_tile_coords: [MAX_RANK]usize = @splat(0);
    var in_local_coords: [MAX_RANK]usize = @splat(0);

    var in_axis_offsets: [16]usize = @splat(0);
    var in_metas: [16]tensor_store.TensorMeta = undefined;
    var non_axis_dims: [MAX_RANK]usize = @splat(0);
    var non_axis_shape: [MAX_RANK]usize = @splat(0);
    var non_axis_strides: [MAX_RANK]usize = @splat(0);
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

    // Fast path: when `axis` and all trailing dims are single-tile in the output and
    // every input, then for each pre-axis coordinate each input contributes ONE
    // contiguous block of (axis_len * inner) elements — copy it with a single memcpy
    // instead of walking the axis (and possibly each element) one at a time. This is
    // the common case (e.g. cache-prepend Concat along a time axis). The general path
    // below handles tiled/awkward layouts.
    fast: {
        var inner_size: usize = 1;
        var dchk: usize = step.axis;
        while (dchk < rank) : (dchk += 1) {
            if (out_meta.tile_shape[dchk] < out_meta.shape[dchk]) break :fast;
            var ii: usize = 0;
            while (ii < count) : (ii += 1) {
                if (in_metas[ii].tile_shape[dchk] < in_metas[ii].shape[dchk]) break :fast;
            }
            if (dchk > step.axis) inner_size *= out_meta.shape[dchk];
        }

        const pre_count: usize = step.axis;
        var pre_shape: [MAX_RANK]usize = @splat(0);
        var pre_strides: [MAX_RANK]usize = @splat(0);
        var dd: usize = 0;
        while (dd < pre_count) : (dd += 1) pre_shape[dd] = out_meta.shape[dd];
        if (pre_count >= 1) try computePackedStrides(pre_shape[0..pre_count], pre_strides[0..pre_count]);
        const pre_total: usize = if (pre_count == 0) 1 else try elemCountFromShape(pre_shape[0..pre_count]);

        var ocache: TileCacheMutND = .{};
        defer if (ocache.valid) store.releaseMut(ocache.tile.token);
        var icache: TileCacheConstND = .{};
        defer if (icache.valid) store.releaseConst(icache.tile.token);

        var pp: usize = 0;
        while (pp < pre_total) : (pp += 1) {
            if (pre_count > 0) try decodeLinearToCoords(pp, pre_strides[0..pre_count], pre_shape[0..pre_count], out_coords[0..pre_count]);
            var si: usize = 0;
            while (si < count) : (si += 1) {
                const sm: tensor_store.TensorMeta = in_metas[si];
                const in_axis_len: usize = sm.shape[step.axis];
                if (in_axis_len == 0) continue;
                // out: (pre.., axis = offset, inner = 0); in: (pre.., axis = 0, inner = 0).
                var d2: usize = 0;
                while (d2 < rank) : (d2 += 1) {
                    const pre_c: usize = if (d2 < pre_count) out_coords[d2] else 0;
                    const out_axis_c: usize = if (d2 == step.axis) in_axis_offsets[si] else pre_c;
                    out_tile_coords[d2] = out_axis_c / out_meta.tile_shape[d2];
                    out_local_coords[d2] = out_axis_c - out_tile_coords[d2] * out_meta.tile_shape[d2];
                    in_tile_coords[d2] = pre_c / sm.tile_shape[d2];
                    in_local_coords[d2] = pre_c - in_tile_coords[d2] * sm.tile_shape[d2];
                }
                const out_ti: usize = try tensor_store.encodeTileIndex(out_meta, out_tile_coords[0..rank]);
                const in_ti: usize = try tensor_store.encodeTileIndex(sm, in_tile_coords[0..rank]);
                try ensureMutTileLinear(&ocache, store, step.out, out_ti, rank);
                try ensureConstTileLinear(&icache, store, step.inputs[si], in_ti, rank);
                var out_lin: usize = 0;
                var in_lin: usize = 0;
                d2 = 0;
                while (d2 < rank) : (d2 += 1) {
                    out_lin = std.math.add(usize, out_lin, out_local_coords[d2] * ocache.strides[d2]) catch return BackendError.InvalidArgument;
                    in_lin = std.math.add(usize, in_lin, in_local_coords[d2] * icache.strides[d2]) catch return BackendError.InvalidArgument;
                }
                const bytes: usize = in_axis_len * inner_size * elem_bytes;
                const out_off: usize = out_lin * elem_bytes;
                const in_off: usize = in_lin * elem_bytes;
                @memcpy(ocache.tile.bytes[out_off .. out_off + bytes], icache.tile.bytes[in_off .. in_off + bytes]);
            }
        }
        return;
    }

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

    var out_tile_count_strides: [MAX_RANK]usize = @splat(0);
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

            var out_tile_coords: [MAX_RANK]usize = @splat(0);
            var out_local_coords: [MAX_RANK]usize = @splat(0);
            var out_global_coords: [MAX_RANK]usize = @splat(0);
            var in_coords: [MAX_RANK]usize = @splat(0);
            var in_tile_coords: [MAX_RANK]usize = @splat(0);
            var in_local_coords: [MAX_RANK]usize = @splat(0);

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

                var out_local_strides: [MAX_RANK]usize = @splat(0);
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
                            const part: f64 = switch (t.a_meta.dtype) {
                                .f32 => @as(f64, reduce_k.sumF32Range(in_cache.tile.bytes, seg_start, seg_end) catch |e| {
                                    t.fail(e);
                                    return;
                                }),
                                .f16 => @as(f64, reduce_k.sumF16RangeToF32(in_cache.tile.bytes, seg_start, seg_end) catch |e| {
                                    t.fail(e);
                                    return;
                                }),
                                .i32 => sum: {
                                    const vals: []align(1) const i32 = @ptrCast(in_cache.tile.bytes);
                                    var integer_sum: i64 = 0;
                                    for (vals[seg_start..seg_end]) |v| integer_sum += v;
                                    break :sum @floatFromInt(integer_sum);
                                },
                                else => {
                                    t.fail(BackendError.InvalidArgument);
                                    return;
                                },
                            };

                            acc += part;
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
                                .i32 => {
                                    const v: i32 = @as(*align(1) const i32, @ptrCast(in_cache.tile.bytes[in_off .. in_off + 4].ptr)).*;
                                    acc += @floatFromInt(v);
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
                        .i32 => {
                            @as(*align(1) i32, @ptrCast(out_view.bytes[out_off .. out_off + 4].ptr)).* = @intFromFloat(acc);
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
