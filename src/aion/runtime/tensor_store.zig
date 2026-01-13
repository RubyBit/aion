const std = @import("std");
const types = @import("../backend/types.zig");

pub const TensorId = u32;

pub const StoreError = error{ InvalidArgument, OutOfMemory };

pub const TensorMeta = struct {
    dtype: types.DType,
    rank: u8,
    shape: [2]usize,
    tile_shape: [2]usize,
    tile_counts: [2]usize,
};

pub const TileRefConst = struct {
    bytes: []const u8,
    dtype: types.DType,
    rank: u8,
    shape_mem: [2]usize,
    strides_mem: [2]isize,
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
    shape_mem: [2]usize,
    strides_mem: [2]isize,
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
        releaseConst: *const fn (ctx: *anyopaque, token: usize) void,
        releaseMut: *const fn (ctx: *anyopaque, token: usize) void,
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

    pub fn releaseConst(self: TensorStore, token: usize) void {
        return self.vtable.releaseConst(self.ctx, token);
    }

    pub fn releaseMut(self: TensorStore, token: usize) void {
        return self.vtable.releaseMut(self.ctx, token);
    }
};
