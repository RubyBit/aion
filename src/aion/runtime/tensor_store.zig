const std = @import("std");
const types = @import("../backend/types.zig");

pub const TensorId = u32;

pub const StoreError = error{ InvalidArgument, OutOfMemory };

pub const TensorMeta = struct {
    dtype: types.DType,
    rank: u8,
    shape: []const usize,
    tile_shape: []const usize,
    tile_counts: []const usize,
    tile_strides: []const usize,
};

const INLINE_RANK: usize = 8;

pub const TileRefConst = struct {
    bytes: []const u8,
    dtype: types.DType,
    rank: u8,
    shape_mem: [INLINE_RANK]usize,
    strides_mem: [INLINE_RANK]isize,
    token: usize = 0,

    pub fn bufferView(self: *const TileRefConst) types.BufferViewConst {
        const r: usize = @as(usize, self.rank);
        return .{
            .bytes = self.bytes,
            .dtype = self.dtype,
            .layout = .{
                .rank = self.rank,
                .shape = self.shape_mem[0..r],
                .strides_bytes = self.strides_mem[0..r],
            },
        };
    }
};

pub const TileRefMut = struct {
    bytes: []u8,
    dtype: types.DType,
    rank: u8,
    shape_mem: [INLINE_RANK]usize,
    strides_mem: [INLINE_RANK]isize,
    token: usize = 0,

    pub fn bufferView(self: *TileRefMut) types.BufferViewMut {
        const r: usize = @as(usize, self.rank);
        return .{
            .bytes = self.bytes,
            .dtype = self.dtype,
            .layout = .{
                .rank = self.rank,
                .shape = self.shape_mem[0..r],
                .strides_bytes = self.strides_mem[0..r],
            },
        };
    }
};

/// Backend-facing storage interface.
///
/// v0 implementation is RAM-only and tokens are always 0.
/// Future out-of-core can use tokens for pin/unpin and deterministic I/O errors.
pub const TensorStore = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        meta: *const fn (ctx: *anyopaque, id: TensorId) StoreError!TensorMeta,
        acquireTileConst: *const fn (ctx: *anyopaque, id: TensorId, ti0: usize, ti1: usize) StoreError!TileRefConst,
        acquireTileMut: *const fn (ctx: *anyopaque, id: TensorId, ti0: usize, ti1: usize) StoreError!TileRefMut,
        acquireTileConstLinear: *const fn (ctx: *anyopaque, id: TensorId, tile_index: usize) StoreError!TileRefConst,
        acquireTileMutLinear: *const fn (ctx: *anyopaque, id: TensorId, tile_index: usize) StoreError!TileRefMut,
        releaseConst: *const fn (ctx: *anyopaque, token: usize) void,
        releaseMut: *const fn (ctx: *anyopaque, token: usize) void,

        /// Prefetch hint. Non-blocking.
        prefetch: ?*const fn (ctx: *anyopaque, id: TensorId, ti0: usize, ti1: usize) void = null,
        prefetchLinear: ?*const fn (ctx: *anyopaque, id: TensorId, tile_index: usize) void = null,
    };

    pub fn meta(self: TensorStore, id: TensorId) StoreError!TensorMeta {
        return self.vtable.meta(self.ctx, id);
    }

    pub fn acquireTileConst(self: TensorStore, id: TensorId, ti0: usize, ti1: usize) StoreError!TileRefConst {
        return self.vtable.acquireTileConst(self.ctx, id, ti0, ti1);
    }

    pub fn acquireTileMut(self: TensorStore, id: TensorId, ti0: usize, ti1: usize) StoreError!TileRefMut {
        return self.vtable.acquireTileMut(self.ctx, id, ti0, ti1);
    }

    pub fn acquireTileConstLinear(self: TensorStore, id: TensorId, tile_index: usize) StoreError!TileRefConst {
        return self.vtable.acquireTileConstLinear(self.ctx, id, tile_index);
    }

    pub fn acquireTileMutLinear(self: TensorStore, id: TensorId, tile_index: usize) StoreError!TileRefMut {
        return self.vtable.acquireTileMutLinear(self.ctx, id, tile_index);
    }

    pub fn releaseConst(self: TensorStore, token: usize) void {
        return self.vtable.releaseConst(self.ctx, token);
    }

    pub fn releaseMut(self: TensorStore, token: usize) void {
        return self.vtable.releaseMut(self.ctx, token);
    }

    pub fn prefetch(self: TensorStore, id: TensorId, ti0: usize, ti1: usize) void {
        if (self.vtable.prefetch) |p| p(self.ctx, id, ti0, ti1);
    }

    pub fn prefetchLinear(self: TensorStore, id: TensorId, tile_index: usize) void {
        if (self.vtable.prefetchLinear) |p| p(self.ctx, id, tile_index);
    }
};

pub fn decodeTileCoords(meta: TensorMeta, tile_index: usize, out: []usize) StoreError!void {
    if (out.len != meta.tile_counts.len) return StoreError.InvalidArgument;
    if (meta.tile_counts.len != meta.tile_strides.len) return StoreError.InvalidArgument;
    if (meta.tile_counts.len == 0) return StoreError.InvalidArgument;

    var d: usize = 0;
    while (d < meta.tile_counts.len) : (d += 1) {
        const stride: usize = meta.tile_strides[d];
        if (stride == 0) return StoreError.InvalidArgument;
        const v: usize = tile_index / stride;
        out[d] = v % meta.tile_counts[d];
    }
}

pub fn encodeTileIndex(meta: TensorMeta, coords: []const usize) StoreError!usize {
    if (coords.len != meta.tile_counts.len) return StoreError.InvalidArgument;
    if (meta.tile_counts.len != meta.tile_strides.len) return StoreError.InvalidArgument;
    if (meta.tile_counts.len == 0) return StoreError.InvalidArgument;

    var idx: usize = 0;
    var d: usize = 0;
    while (d < coords.len) : (d += 1) {
        if (coords[d] >= meta.tile_counts[d]) return StoreError.InvalidArgument;
        idx = std.math.add(usize, idx, coords[d] * meta.tile_strides[d]) catch return StoreError.InvalidArgument;
    }
    return idx;
}
