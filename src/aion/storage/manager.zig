const std = @import("std");

const storage_mod = @import("storage.zig");
const types = @import("../backend/types.zig");
const tensor_store = @import("../runtime/tensor_store.zig");

pub const StorageError = storage_mod.StorageError;
pub const TiledTensor = storage_mod.TiledTensor;
pub const DType = types.DType;

/// Opaque handle to a tensor owned by `StorageManager`.
///
/// Stable across internal resizes because it is an index into a pointer table.
pub const TensorId = u32;

/// RAM-only storage manager (v0).
///
/// This is the seam where tiered storage (RAM->SSD) will plug in later.
/// For now it owns `TiledTensor` allocations and provides tile acquisition.
pub const StorageManager = struct {
    allocator: std.mem.Allocator,
    tensors: std.ArrayList(*TiledTensor) = .{},

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Self) void {
        // Destroy tensors first.
        for (self.tensors.items) |t| {
            t.deinit();
            self.allocator.destroy(t);
        }
        self.tensors.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn createTiledTensor(
        self: *Self,
        dtype: DType,
        shape: []const usize,
        tile_shape: []const usize,
        opts: TiledTensor.InitOptions,
    ) StorageError!TensorId {
        var t: *TiledTensor = self.allocator.create(TiledTensor) catch return StorageError.OutOfMemory;
        errdefer self.allocator.destroy(t);

        t.* = try TiledTensor.init(self.allocator, dtype, shape, tile_shape, opts);
        errdefer t.deinit();

        const idx_usize: usize = self.tensors.items.len;
        self.tensors.append(self.allocator, t) catch {
            t.deinit();
            self.allocator.destroy(t);
            return StorageError.OutOfMemory;
        };

        return @intCast(idx_usize);
    }

    pub fn getConst(self: *const Self, id: TensorId) StorageError!*const TiledTensor {
        const idx: usize = @intCast(id);
        if (idx >= self.tensors.items.len) return StorageError.InvalidArgument;
        return self.tensors.items[idx];
    }

    pub fn getMut(self: *Self, id: TensorId) StorageError!*TiledTensor {
        const idx: usize = @intCast(id);
        if (idx >= self.tensors.items.len) return StorageError.InvalidArgument;
        return self.tensors.items[idx];
    }

    pub fn writeFromPackedScalar(self: *Self, id: TensorId, packed_bytes: []const u8) StorageError!void {
        var t: *TiledTensor = try self.getMut(id);
        return t.writeFromPackedScalar(packed_bytes);
    }

    pub fn readToPackedScalar(self: *const Self, id: TensorId, out: []u8) StorageError!void {
        const t: *const TiledTensor = try self.getConst(id);
        return t.readToPackedScalar(out);
    }

    pub fn writeFromPackedQuant(self: *Self, id: TensorId, packed_bytes: []const u8) StorageError!void {
        var t: *TiledTensor = try self.getMut(id);
        return t.writeFromPackedQuant(packed_bytes);
    }

    pub fn readToPackedQuant(self: *const Self, id: TensorId, out: []u8) StorageError!void {
        const t: *const TiledTensor = try self.getConst(id);
        return t.readToPackedQuant(out);
    }

    pub fn tensorStore(self: *Self) tensor_store.TensorStore {
        const Vt = struct {
            fn meta(ctx: *anyopaque, id: tensor_store.TensorId) tensor_store.StoreError!tensor_store.TensorMeta {
                const sm: *StorageManager = @ptrCast(@alignCast(ctx));
                const t: *const TiledTensor = sm.getConst(@intCast(id)) catch return tensor_store.StoreError.InvalidArgument;
                return .{
                    .dtype = t.dtype,
                    .rank = t.rank,
                    .shape = t.shape,
                    .tile_shape = t.tile_shape,
                    .tile_counts = t.tile_counts,
                };
            }

            fn acquireTileConst(ctx: *anyopaque, id: tensor_store.TensorId, ti0: usize, ti1: usize) tensor_store.StoreError!tensor_store.TileRefConst {
                const sm: *StorageManager = @ptrCast(@alignCast(ctx));
                const t: *const TiledTensor = sm.getConst(@intCast(id)) catch return tensor_store.StoreError.InvalidArgument;
                const tile = t.acquireTileConst(ti0, ti1) catch return tensor_store.StoreError.InvalidArgument;
                return .{
                    .bytes = tile.bytes,
                    .dtype = tile.dtype,
                    .rank = tile.rank,
                    .shape_mem = tile.shape_mem,
                    .strides_mem = tile.strides_mem,
                    .token = 0,
                };
            }

            fn acquireTileMut(ctx: *anyopaque, id: tensor_store.TensorId, ti0: usize, ti1: usize) tensor_store.StoreError!tensor_store.TileRefMut {
                const sm: *StorageManager = @ptrCast(@alignCast(ctx));
                const t: *TiledTensor = sm.getMut(@intCast(id)) catch return tensor_store.StoreError.InvalidArgument;
                var tile = t.acquireTileMut(ti0, ti1) catch return tensor_store.StoreError.InvalidArgument;
                return .{
                    .bytes = tile.bytes,
                    .dtype = tile.dtype,
                    .rank = tile.rank,
                    .shape_mem = tile.shape_mem,
                    .strides_mem = tile.strides_mem,
                    .token = 0,
                };
            }

            fn releaseConst(_: *anyopaque, _: usize) void {}
            fn releaseMut(_: *anyopaque, _: usize) void {}
        };

        return .{
            .ctx = @ptrCast(self),
            .vtable = &.{
                .meta = Vt.meta,
                .acquireTileConst = Vt.acquireTileConst,
                .acquireTileMut = Vt.acquireTileMut,
                .releaseConst = Vt.releaseConst,
                .releaseMut = Vt.releaseMut,
            },
        };
    }
};
