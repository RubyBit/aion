const std = @import("std");

const aion_file = @import("aion_file.zig");
const storage_mod = @import("storage.zig");
const types = @import("../backend/types.zig");
const tensor_store = @import("../runtime/tensor_store.zig");

pub const StorageError = storage_mod.StorageError;
pub const TiledTensor = storage_mod.TiledTensor;
pub const DType = types.DType;
pub const LoadAionError = StorageError || aion_file.FileError;

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
    tensor_readonly: std.ArrayList(bool) = .{},
    imported_aion_mappings: std.ArrayList(ImportedAionMapping) = .{},
    tensor_names: std.StringHashMapUnmanaged(TensorId) = .{},

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
        self.tensor_readonly.deinit(self.allocator);
        self.tensor_names.deinit(self.allocator);
        for (self.imported_aion_mappings.items) |*mapping| {
            self.releaseImportedMapping(mapping);
        }
        self.imported_aion_mappings.deinit(self.allocator);
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

        try t.init(self.allocator, dtype, shape, tile_shape, opts);
        errdefer t.deinit();

        const idx_usize: usize = self.tensors.items.len;
        self.tensors.append(self.allocator, t) catch {
            t.deinit();
            self.allocator.destroy(t);
            return StorageError.OutOfMemory;
        };
        errdefer _ = self.tensors.pop();

        self.tensor_readonly.append(self.allocator, false) catch {
            t.deinit();
            self.allocator.destroy(t);
            _ = self.tensors.pop();
            return StorageError.OutOfMemory;
        };

        return @intCast(idx_usize);
    }

    fn createBorrowedTiledTensor(
        self: *Self,
        dtype: DType,
        shape: []const usize,
        tile_shape: []const usize,
        tile_offsets: []const usize,
        tile_lens: []const usize,
        data: []align(64) u8,
        opts: TiledTensor.InitOptions,
    ) StorageError!TensorId {
        var t: *TiledTensor = self.allocator.create(TiledTensor) catch return StorageError.OutOfMemory;
        errdefer self.allocator.destroy(t);

        try t.initBorrowed(self.allocator, dtype, shape, tile_shape, tile_offsets, tile_lens, data, opts);
        errdefer t.deinit();

        const idx_usize: usize = self.tensors.items.len;
        self.tensors.append(self.allocator, t) catch {
            t.deinit();
            self.allocator.destroy(t);
            return StorageError.OutOfMemory;
        };
        errdefer _ = self.tensors.pop();

        self.tensor_readonly.append(self.allocator, true) catch {
            t.deinit();
            self.allocator.destroy(t);
            _ = self.tensors.pop();
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
        if (idx >= self.tensor_readonly.items.len) return StorageError.InvalidArgument;
        if (self.tensor_readonly.items[idx]) return StorageError.InvalidArgument;
        return self.tensors.items[idx];
    }

    pub fn tensorIdByName(self: *const Self, name: []const u8) ?TensorId {
        return self.tensor_names.get(name);
    }

    pub fn importAionFile(self: *Self, file: std.fs.File) LoadAionError!void {
        const end_pos: u64 = file.getEndPos() catch return aion_file.FileError.IoFailure;
        const file_size: usize = std.math.cast(usize, end_pos) orelse return aion_file.FileError.InvalidFormat;
        var mapped: aion_file.MappedFile = try aion_file.MappedFile.mapReadOnly(file);
        errdefer mapped.deinit();
        try self.importAionMapped(mapped, file_size);
    }

    pub fn importAionBytes(self: *Self, owned_bytes: []align(64) u8) LoadAionError!void {
        const start_tensor_count: usize = self.tensors.items.len;
        const start_mapping_count: usize = self.imported_aion_mappings.items.len;

        var added_names: std.ArrayList([]const u8) = .{};
        defer added_names.deinit(self.allocator);

        errdefer {
            while (added_names.items.len != 0) {
                const key: []const u8 = added_names.pop().?;
                _ = self.tensor_names.remove(key);
            }
            while (self.tensors.items.len > start_tensor_count) {
                const t: *TiledTensor = self.tensors.pop().?;
                _ = self.tensor_readonly.pop();
                t.deinit();
                self.allocator.destroy(t);
            }
            while (self.imported_aion_mappings.items.len > start_mapping_count) {
                var mapping: ImportedAionMapping = self.imported_aion_mappings.pop().?;
                self.releaseImportedMapping(&mapping);
            }
        }

        const view: aion_file.View = try aion_file.parse(owned_bytes);

        try self.imported_aion_mappings.append(self.allocator, .{ .owned = owned_bytes });

        try self.importAionView(view);
    }

    fn importAionMapped(self: *Self, mapped_file: aion_file.MappedFile, file_size: usize) LoadAionError!void {
        const start_tensor_count: usize = self.tensors.items.len;
        const start_mapping_count: usize = self.imported_aion_mappings.items.len;

        var added_names: std.ArrayList([]const u8) = .{};
        defer added_names.deinit(self.allocator);

        var mapped = mapped_file;
        var mapping_moved: bool = false;
        errdefer if (!mapping_moved) mapped.deinit();

        errdefer {
            while (added_names.items.len != 0) {
                const key: []const u8 = added_names.pop().?;
                _ = self.tensor_names.remove(key);
            }
            while (self.tensors.items.len > start_tensor_count) {
                const t: *TiledTensor = self.tensors.pop().?;
                _ = self.tensor_readonly.pop();
                t.deinit();
                self.allocator.destroy(t);
            }
            while (self.imported_aion_mappings.items.len > start_mapping_count) {
                var mapping: ImportedAionMapping = self.imported_aion_mappings.pop().?;
                self.releaseImportedMapping(&mapping);
            }
        }

        const bytes = try mapped.logicalBytes(file_size);
        const view: aion_file.View = try aion_file.parse(bytes);

        try self.imported_aion_mappings.append(self.allocator, .{ .mapped = .{ .file = mapped, .file_size = file_size } });
        mapping_moved = true;
        try self.importAionViewWithNames(view, &added_names);
    }

    fn importAionView(self: *Self, view: aion_file.View) LoadAionError!void {
        var added_names: std.ArrayList([]const u8) = .{};
        defer added_names.deinit(self.allocator);
        try self.importAionViewWithNames(view, &added_names);
    }

    fn importAionViewWithNames(self: *Self, view: aion_file.View, added_names: *std.ArrayList([]const u8)) LoadAionError!void {
        const tensor_count: usize = std.math.cast(usize, view.header.tensor_count) orelse return StorageError.InvalidArgument;
        var ti: usize = 0;
        while (ti < tensor_count) : (ti += 1) {
            const desc: aion_file.TensorDescriptor = try view.tensor(ti);
            try view.validateTensorCrc32(desc);
            if (self.tensor_names.contains(desc.name)) return StorageError.InvalidArgument;

            const rank: usize = desc.rank;
            var shape_mem: [8]usize = .{0} ** 8;
            var tile_shape_mem: [8]usize = .{0} ** 8;
            var d: usize = 0;
            while (d < rank) : (d += 1) {
                shape_mem[d] = std.math.cast(usize, desc.shape_mem[d]) orelse return StorageError.InvalidArgument;
                tile_shape_mem[d] = std.math.cast(usize, desc.tile_shape_mem[d]) orelse return StorageError.InvalidArgument;
            }

            const tile_count: usize = std.math.cast(usize, desc.tile_count) orelse return StorageError.InvalidArgument;
            const tile_offsets: []usize = try self.allocator.alloc(usize, tile_count);
            defer self.allocator.free(tile_offsets);
            const tile_lens: []usize = try self.allocator.alloc(usize, tile_count);
            defer self.allocator.free(tile_lens);

            var tile_index: usize = 0;
            while (tile_index < tile_count) : (tile_index += 1) {
                tile_offsets[tile_index] = std.math.cast(usize, try view.tileOffsetAt(desc, tile_index)) orelse return StorageError.InvalidArgument;
                tile_lens[tile_index] = std.math.cast(usize, try view.tileLenAt(desc, tile_index)) orelse return StorageError.InvalidArgument;
            }

            const data_start: usize = std.math.cast(usize, desc.tensor_data_offset) orelse return StorageError.InvalidArgument;
            const data_len: usize = std.math.cast(usize, desc.tensor_data_size) orelse return StorageError.InvalidArgument;
            const data_unaligned: []u8 = @constCast(view.bytes[data_start .. data_start + data_len]);
            const data: []align(64) u8 = @alignCast(data_unaligned);

            const tid: TensorId = try self.createBorrowedTiledTensor(
                desc.dtype,
                shape_mem[0..rank],
                tile_shape_mem[0..rank],
                tile_offsets,
                tile_lens,
                data,
                .{ .tile_alignment = 64 },
            );

            try self.tensor_names.put(self.allocator, desc.name, tid);
            try added_names.append(self.allocator, desc.name);
        }
    }

    fn releaseImportedMapping(self: *Self, mapping: *ImportedAionMapping) void {
        switch (mapping.*) {
            .owned => |buf| self.allocator.free(buf),
            .mapped => |*m| {
                _ = m.file_size;
                m.file.deinit();
            },
        }
        mapping.* = undefined;
    }

    const ImportedAionMapping = union(enum) {
        owned: []align(64) u8,
        mapped: struct {
            file: aion_file.MappedFile,
            file_size: usize,
        },
    };

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
                    .tile_strides = t.tile_strides,
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

            fn acquireTileConstLinear(ctx: *anyopaque, id: tensor_store.TensorId, tile_index: usize) tensor_store.StoreError!tensor_store.TileRefConst {
                const sm: *StorageManager = @ptrCast(@alignCast(ctx));
                const t: *const TiledTensor = sm.getConst(@intCast(id)) catch return tensor_store.StoreError.InvalidArgument;
                const tile = t.acquireTileConstLinear(tile_index) catch return tensor_store.StoreError.InvalidArgument;
                return .{
                    .bytes = tile.bytes,
                    .dtype = tile.dtype,
                    .rank = tile.rank,
                    .shape_mem = tile.shape_mem,
                    .strides_mem = tile.strides_mem,
                    .token = 0,
                };
            }

            fn acquireTileMutLinear(ctx: *anyopaque, id: tensor_store.TensorId, tile_index: usize) tensor_store.StoreError!tensor_store.TileRefMut {
                const sm: *StorageManager = @ptrCast(@alignCast(ctx));
                const t: *TiledTensor = sm.getMut(@intCast(id)) catch return tensor_store.StoreError.InvalidArgument;
                var tile = t.acquireTileMutLinear(tile_index) catch return tensor_store.StoreError.InvalidArgument;
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

            fn prefetch(ctx: *anyopaque, id: tensor_store.TensorId, ti0: usize, ti1: usize) void {
                const sm: *StorageManager = @ptrCast(@alignCast(ctx));
                const t: *const TiledTensor = sm.getConst(@intCast(id)) catch return;
                const tile = t.acquireTileConst(ti0, ti1) catch return;
                // v0: RAM-only prefetch uses intrinsic.
                @prefetch(tile.bytes.ptr, .{ .rw = .read, .locality = 3, .cache = .data });
            }

            fn prefetchLinear(ctx: *anyopaque, id: tensor_store.TensorId, tile_index: usize) void {
                const sm: *StorageManager = @ptrCast(@alignCast(ctx));
                const t: *const TiledTensor = sm.getConst(@intCast(id)) catch return;
                const tile = t.acquireTileConstLinear(tile_index) catch return;
                @prefetch(tile.bytes.ptr, .{ .rw = .read, .locality = 3, .cache = .data });
            }
        };

        return .{
            .ctx = @ptrCast(self),
            .vtable = &.{
                .meta = Vt.meta,
                .acquireTileConst = Vt.acquireTileConst,
                .acquireTileMut = Vt.acquireTileMut,
                .acquireTileConstLinear = Vt.acquireTileConstLinear,
                .acquireTileMutLinear = Vt.acquireTileMutLinear,
                .releaseConst = Vt.releaseConst,
                .releaseMut = Vt.releaseMut,
                .prefetch = Vt.prefetch,
                .prefetchLinear = Vt.prefetchLinear,
            },
        };
    }
};
