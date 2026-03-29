const std = @import("std");

const aion_file = @import("aion_file.zig");
const storage_mod = @import("storage.zig");
const types = @import("../backend/types.zig");
const tensor_store = @import("../runtime/tensor_store.zig");

pub const StorageError = storage_mod.StorageError;
pub const TiledTensor = storage_mod.TiledTensor;
pub const DType = types.DType;
pub const LoadAionError = StorageError || aion_file.FileError;
pub const ImportResidency = enum {
    mapped,
    copied,
};

pub const ImportAionOptions = struct {
    residency: ImportResidency = .mapped,
};

pub const TensorResidency = enum {
    owned,
    imported_mapped,
    imported_copied,
};

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
    tensors: std.ArrayList(*TiledTensor) = .empty,
    tensor_readonly: std.ArrayList(bool) = .empty,
    tensor_residency: std.ArrayList(TensorResidency) = .empty,
    tensor_import_mapping_indices: std.ArrayList(?u32) = .empty,
    imported_aion_mappings: std.ArrayList(ImportedAionMapping) = .empty,
    tensor_names: std.StringHashMapUnmanaged(TensorId) = .{},
    owned_tensor_names: std.ArrayList([]u8) = .empty,

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
        self.tensor_residency.deinit(self.allocator);
        self.tensor_import_mapping_indices.deinit(self.allocator);
        for (self.owned_tensor_names.items) |name| {
            self.allocator.free(name);
        }
        self.owned_tensor_names.deinit(self.allocator);
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

        return self.registerTensor(t, false, .owned, null) catch {
            t.deinit();
            self.allocator.destroy(t);
            return StorageError.OutOfMemory;
        };
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
        residency: TensorResidency,
        mapping_index: u32,
    ) StorageError!TensorId {
        var t: *TiledTensor = self.allocator.create(TiledTensor) catch return StorageError.OutOfMemory;
        errdefer self.allocator.destroy(t);

        try t.initBorrowed(self.allocator, dtype, shape, tile_shape, tile_offsets, tile_lens, data, opts);
        errdefer t.deinit();

        return self.registerTensor(t, true, residency, mapping_index) catch {
            t.deinit();
            self.allocator.destroy(t);
            return StorageError.OutOfMemory;
        };
    }

    fn registerTensor(self: *Self, t: *TiledTensor, readonly: bool, residency: TensorResidency, mapping_index: ?u32) StorageError!TensorId {
        const idx_usize: usize = self.tensors.items.len;
        self.tensors.append(self.allocator, t) catch return StorageError.OutOfMemory;
        errdefer _ = self.tensors.pop();

        self.tensor_readonly.append(self.allocator, readonly) catch return StorageError.OutOfMemory;
        errdefer _ = self.tensor_readonly.pop();

        self.tensor_residency.append(self.allocator, residency) catch return StorageError.OutOfMemory;
        errdefer _ = self.tensor_residency.pop();

        self.tensor_import_mapping_indices.append(self.allocator, mapping_index) catch return StorageError.OutOfMemory;
        errdefer _ = self.tensor_import_mapping_indices.pop();

        if (mapping_index) |mi| {
            const mapping_idx: usize = @intCast(mi);
            if (mapping_idx >= self.imported_aion_mappings.items.len) return StorageError.InvalidArgument;
            self.imported_aion_mappings.items[mapping_idx].borrowed_tensor_count += 1;
        }

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

    pub fn importAionFile(self: *Self, file: std.Io.File, opts: ImportAionOptions) LoadAionError!void {
        switch (opts.residency) {
            .mapped => {
                var io_backend: std.Io.Threaded = .init_single_threaded;
                const io = io_backend.io();
                const end_pos: u64 = file.length(io) catch return aion_file.FileError.IoFailure;
                const file_size: usize = std.math.cast(usize, end_pos) orelse return aion_file.FileError.InvalidFormat;
                var mapped: aion_file.MappedFile = try aion_file.MappedFile.mapReadOnly(file);
                errdefer mapped.deinit();
                try self.importAionMapped(mapped, file_size);
            },
            .copied => {
                const bytes: []align(64) u8 = try aion_file.readAlloc(self.allocator, file);
                errdefer self.allocator.free(bytes);
                try self.importAionOwnedBytes(bytes);
            },
        }
    }

    fn importAionOwnedBytes(self: *Self, bytes: []align(64) u8) LoadAionError!void {
        const start_tensor_count: usize = self.tensors.items.len;
        const start_mapping_count: usize = self.imported_aion_mappings.items.len;
        const start_owned_name_count: usize = self.owned_tensor_names.items.len;

        var added_names: std.ArrayList([]const u8) = .empty;
        defer added_names.deinit(self.allocator);

        const owned_bytes = bytes;
        var ownership_moved: bool = false;
        errdefer if (!ownership_moved) self.allocator.free(owned_bytes);

        errdefer {
            while (added_names.items.len != 0) {
                const key: []const u8 = added_names.pop().?;
                _ = self.tensor_names.remove(key);
            }
            while (self.owned_tensor_names.items.len > start_owned_name_count) {
                const name: []u8 = self.owned_tensor_names.pop().?;
                self.allocator.free(name);
            }
            while (self.tensors.items.len > start_tensor_count) {
                const t: *TiledTensor = self.tensors.pop().?;
                _ = self.tensor_readonly.pop();
                _ = self.tensor_residency.pop();
                _ = self.tensor_import_mapping_indices.pop();
                t.deinit();
                self.allocator.destroy(t);
            }
            while (self.imported_aion_mappings.items.len > start_mapping_count) {
                var mapping: ImportedAionMapping = self.imported_aion_mappings.pop().?;
                self.releaseImportedMapping(&mapping);
            }
        }

        const view: aion_file.View = try aion_file.parse(owned_bytes);
        try self.imported_aion_mappings.append(self.allocator, .{ .storage = .{ .owned = owned_bytes } });
        ownership_moved = true;

        const mapping_index: u32 = @intCast(self.imported_aion_mappings.items.len - 1);
        try self.importAionViewWithNames(view, &added_names, .imported_copied, mapping_index);
    }

    fn importAionMapped(self: *Self, mapped_file: aion_file.MappedFile, file_size: usize) LoadAionError!void {
        const start_tensor_count: usize = self.tensors.items.len;
        const start_mapping_count: usize = self.imported_aion_mappings.items.len;
        const start_owned_name_count: usize = self.owned_tensor_names.items.len;

        var added_names: std.ArrayList([]const u8) = .empty;
        defer added_names.deinit(self.allocator);

        var mapped = mapped_file;
        var mapping_moved: bool = false;
        errdefer if (!mapping_moved) mapped.deinit();

        errdefer {
            while (added_names.items.len != 0) {
                const key: []const u8 = added_names.pop().?;
                _ = self.tensor_names.remove(key);
            }
            while (self.owned_tensor_names.items.len > start_owned_name_count) {
                const name: []u8 = self.owned_tensor_names.pop().?;
                self.allocator.free(name);
            }
            while (self.tensors.items.len > start_tensor_count) {
                const t: *TiledTensor = self.tensors.pop().?;
                _ = self.tensor_readonly.pop();
                _ = self.tensor_residency.pop();
                _ = self.tensor_import_mapping_indices.pop();
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

        try self.imported_aion_mappings.append(self.allocator, .{ .storage = .{ .mapped = .{ .file = mapped, .file_size = file_size } } });
        mapping_moved = true;

        const mapping_index: u32 = @intCast(self.imported_aion_mappings.items.len - 1);
        try self.importAionViewWithNames(view, &added_names, .imported_mapped, mapping_index);
    }

    fn importAionView(self: *Self, view: aion_file.View) LoadAionError!void {
        var added_names: std.ArrayList([]const u8) = .empty;
        defer added_names.deinit(self.allocator);
        try self.importAionViewWithNames(view, &added_names, .owned, null);
    }

    fn importAionViewWithNames(self: *Self, view: aion_file.View, added_names: *std.ArrayList([]const u8), residency: TensorResidency, mapping_index: ?u32) LoadAionError!void {
        const tensor_count: usize = std.math.cast(usize, view.header.tensor_count) orelse return StorageError.InvalidArgument;
        var ti: usize = 0;
        while (ti < tensor_count) : (ti += 1) {
            const desc: aion_file.TensorDescriptor = try view.tensor(ti);
            try view.validateTensorCrc32(desc);
            if (self.tensor_names.contains(desc.name)) return StorageError.InvalidArgument;

            const owned_name: []u8 = try self.allocator.dupe(u8, desc.name);
            var owned_name_moved: bool = false;
            errdefer if (!owned_name_moved) self.allocator.free(owned_name);

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
                residency,
                if (mapping_index) |mi| mi else return StorageError.InvalidArgument,
            );

            try self.tensor_names.put(self.allocator, owned_name, tid);
            errdefer _ = self.tensor_names.remove(owned_name);
            try self.owned_tensor_names.append(self.allocator, owned_name);
            owned_name_moved = true;
            try added_names.append(self.allocator, owned_name);
        }
    }

    pub fn tensorResidency(self: *const Self, id: TensorId) StorageError!TensorResidency {
        const idx: usize = @intCast(id);
        if (idx >= self.tensor_residency.items.len) return StorageError.InvalidArgument;
        return self.tensor_residency.items[idx];
    }

    pub fn isMappedTensor(self: *const Self, id: TensorId) StorageError!bool {
        return (try self.tensorResidency(id)) == .imported_mapped;
    }

    pub fn promoteToOwnedRam(self: *Self, id: TensorId) StorageError!void {
        const idx: usize = @intCast(id);
        if (idx >= self.tensors.items.len or idx >= self.tensor_readonly.items.len or idx >= self.tensor_residency.items.len or idx >= self.tensor_import_mapping_indices.items.len) {
            return StorageError.InvalidArgument;
        }

        if (self.tensor_residency.items[idx] == .owned) return;

        const mapping_index_opt: ?u32 = self.tensor_import_mapping_indices.items[idx];
        if (mapping_index_opt == null) return StorageError.InvalidArgument;

        var t: *TiledTensor = self.tensors.items[idx];
        _ = try t.promoteToOwned();

        self.tensor_readonly.items[idx] = false;
        self.tensor_residency.items[idx] = .owned;
        self.tensor_import_mapping_indices.items[idx] = null;

        const mapping_index: usize = @intCast(mapping_index_opt.?);
        if (mapping_index >= self.imported_aion_mappings.items.len) return StorageError.InvalidArgument;

        var mapping: *ImportedAionMapping = &self.imported_aion_mappings.items[mapping_index];
        if (mapping.borrowed_tensor_count == 0) return StorageError.InvalidArgument;
        mapping.borrowed_tensor_count -= 1;
        if (mapping.borrowed_tensor_count == 0) {
            self.releaseImportedMapping(mapping);
        }
    }

    fn releaseImportedMapping(self: *Self, mapping: *ImportedAionMapping) void {
        switch (mapping.storage) {
            .released => {},
            .owned => |buf| self.allocator.free(buf),
            .mapped => |*m| {
                _ = m.file_size;
                m.file.deinit();
            },
        }
        mapping.storage = .released;
        mapping.borrowed_tensor_count = 0;
    }

    const ImportedAionMapping = struct {
        storage: Storage,
        borrowed_tensor_count: usize = 0,

        const Storage = union(enum) {
            released: void,
            owned: []align(64) u8,
            mapped: struct {
                file: aion_file.MappedFile,
                file_size: usize,
            },
        };
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
                const tile = t.acquireTileMut(ti0, ti1) catch return tensor_store.StoreError.InvalidArgument;
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
                const tile = t.acquireTileMutLinear(tile_index) catch return tensor_store.StoreError.InvalidArgument;
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
