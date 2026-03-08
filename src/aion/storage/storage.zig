const std = @import("std");

const types = @import("../backend/types.zig");
const utils = @import("../backend/utils.zig");
const small_vec = @import("../small_vec.zig");

const SmallVec = small_vec.SmallVec;
const INLINE_RANK: usize = 8;

pub const DType = types.DType;
pub const Layout = types.Layout;
pub const BufferViewConst = types.BufferViewConst;
pub const BufferViewMut = types.BufferViewMut;

/// Errors for in-memory tiled storage (v0).
///
/// Notes:
/// - This module is intentionally backend-agnostic.
/// - Deterministic out-of-core errors will live in `cache.zig` / `io.zig` later.
pub const StorageError = error{
    InvalidArgument,
    OutOfMemory,
};

fn asStorageError(err: anyerror) StorageError {
    return switch (err) {
        error.OutOfMemory => StorageError.OutOfMemory,
        else => StorageError.InvalidArgument,
    };
}

fn alignForward(value: usize, alignment: usize) usize {
    return std.mem.alignForward(usize, value, alignment);
}

pub const TileViewConst = struct {
    bytes: []const u8,
    dtype: DType,
    rank: u8,
    shape_mem: [INLINE_RANK]usize,
    strides_mem: [INLINE_RANK]isize,

    pub fn init(bytes: []const u8, dtype: DType, rank: u8, dims: []const usize, elem_bytes: usize) StorageError!TileViewConst {
        if (dims.len != @as(usize, rank)) return StorageError.InvalidArgument;
        if (dims.len > INLINE_RANK) return StorageError.InvalidArgument;

        var self: TileViewConst = undefined;
        self.bytes = bytes;
        self.dtype = dtype;
        self.rank = rank;
        @memset(self.shape_mem[0..INLINE_RANK], 0);
        @memset(self.strides_mem[0..INLINE_RANK], 0);

        var i: usize = 0;
        while (i < dims.len) : (i += 1) {
            self.shape_mem[i] = dims[i];
        }

        // For quant dtypes, strides are ignored in v0 packedness checks.
        if (!dtype.info().is_quantized) {
            // Packed row-major scalar: stride[d] = elem_bytes * product(dims[d+1..]).
            var stride: usize = elem_bytes;
            var d: usize = dims.len;
            while (d > 0) : (d -= 1) {
                const idx: usize = d - 1;
                self.strides_mem[idx] = @intCast(stride);
                stride = std.math.mul(usize, stride, dims[idx]) catch return StorageError.InvalidArgument;
            }
        }

        return self;
    }

    pub fn bufferView(self: *const TileViewConst) BufferViewConst {
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

pub const TileViewMut = struct {
    bytes: []u8,
    dtype: DType,
    rank: u8,
    shape_mem: [INLINE_RANK]usize,
    strides_mem: [INLINE_RANK]isize,

    pub fn init(bytes: []u8, dtype: DType, rank: u8, dims: []const usize, elem_bytes: usize) StorageError!TileViewMut {
        if (dims.len != @as(usize, rank)) return StorageError.InvalidArgument;
        if (dims.len > INLINE_RANK) return StorageError.InvalidArgument;

        var self: TileViewMut = undefined;
        self.bytes = bytes;
        self.dtype = dtype;
        self.rank = rank;
        @memset(self.shape_mem[0..INLINE_RANK], 0);
        @memset(self.strides_mem[0..INLINE_RANK], 0);

        var i: usize = 0;
        while (i < dims.len) : (i += 1) {
            self.shape_mem[i] = dims[i];
        }

        if (!dtype.info().is_quantized) {
            var stride: usize = elem_bytes;
            var d: usize = dims.len;
            while (d > 0) : (d -= 1) {
                const idx: usize = d - 1;
                self.strides_mem[idx] = @intCast(stride);
                stride = std.math.mul(usize, stride, dims[idx]) catch return StorageError.InvalidArgument;
            }
        }

        return self;
    }

    pub fn bufferView(self: *TileViewMut) BufferViewMut {
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

/// RAM-only tiled tensor storage, backed by a single contiguous allocation.
///
/// Important: This storage is *physically tiled* (tiles are stored contiguously, in tile-major order).
/// This keeps backend execution packed-first and avoids per-tile allocations, while still enabling
/// out-of-core semantics later (tiles become the cache unit).
///
/// v0 scope:
/// - scalar tensors support arbitrary rank (packed row-major)
/// - tiles are always packed
/// - quant types are supported for rank 1/2 with strict K-axis alignment (axis 0)
pub const TiledTensor = struct {
    allocator: std.mem.Allocator,

    dtype: DType,
    rank: u8,
    shape: []const usize,
    tile_shape: []const usize,
    tile_counts: []const usize,
    tile_strides: []const usize,

    shape_storage: SmallVec(usize, INLINE_RANK),
    tile_shape_storage: SmallVec(usize, INLINE_RANK),
    tile_counts_storage: SmallVec(usize, INLINE_RANK),
    tile_strides_storage: SmallVec(usize, INLINE_RANK),

    // Metadata: one allocation, split into offsets and lens.
    meta: []usize,
    tile_offsets: []usize,
    tile_lens: []usize,

    // Tile backing buffer.
    data: []align(64) u8,

    owns_data: bool = true,

    // Alignment between tiles in the backing buffer.
    tile_alignment: usize = 64,

    const Self = @This();

    pub const InitOptions = struct {
        /// Align tile starts in the single backing allocation.
        tile_alignment: usize = 64,
    };

    pub fn init(
        self: *Self,
        allocator: std.mem.Allocator,
        dtype: DType,
        shape_in: []const usize,
        tile_shape_in: []const usize,
        opts: InitOptions,
    ) StorageError!void {
        if (shape_in.len == 0) return StorageError.InvalidArgument;
        if (tile_shape_in.len != shape_in.len) return StorageError.InvalidArgument;

        const rank: u8 = @intCast(shape_in.len);

        var shape_storage: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initFromSlice(allocator, shape_in) catch return StorageError.OutOfMemory;
        var tile_shape_storage: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initFromSlice(allocator, tile_shape_in) catch {
            shape_storage.deinit();
            return StorageError.OutOfMemory;
        };

        var tile_counts_storage: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(allocator, shape_in.len) catch {
            shape_storage.deinit();
            tile_shape_storage.deinit();
            return StorageError.OutOfMemory;
        };

        var tile_strides_storage: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(allocator, shape_in.len) catch {
            shape_storage.deinit();
            tile_shape_storage.deinit();
            tile_counts_storage.deinit();
            return StorageError.OutOfMemory;
        };
        var moved: bool = false;
        errdefer if (!moved) {
            shape_storage.deinit();
            tile_shape_storage.deinit();
            tile_counts_storage.deinit();
            tile_strides_storage.deinit();
        };

        self.* = .{
            .allocator = allocator,
            .dtype = dtype,
            .rank = rank,
            .shape = &[_]usize{},
            .tile_shape = &[_]usize{},
            .tile_counts = &[_]usize{},
            .tile_strides = &[_]usize{},
            .shape_storage = shape_storage,
            .tile_shape_storage = tile_shape_storage,
            .tile_counts_storage = tile_counts_storage,
            .tile_strides_storage = tile_strides_storage,
            .meta = &[_]usize{},
            .tile_offsets = &[_]usize{},
            .tile_lens = &[_]usize{},
            .data = &[_]u8{},
            .owns_data = true,
            .tile_alignment = opts.tile_alignment,
        };
        moved = true;
        errdefer self.deinit();

        self.shape = self.shape_storage.constSlice();
        self.tile_shape = self.tile_shape_storage.constSlice();

        var tile_counts_mut: []usize = self.tile_counts_storage.slice();
        var tile_strides_mut: []usize = self.tile_strides_storage.slice();

        var d: usize = 0;
        while (d < self.shape.len) : (d += 1) {
            if (self.shape[d] == 0) return StorageError.InvalidArgument;
            if (self.tile_shape[d] == 0) return StorageError.InvalidArgument;
            const count: usize = (self.shape[d] + self.tile_shape[d] - 1) / self.tile_shape[d];
            if (count == 0) return StorageError.InvalidArgument;
            tile_counts_mut[d] = count;
        }

        var tile_total: usize = 1;
        var rev: usize = self.shape.len;
        while (rev > 0) : (rev -= 1) {
            const idx: usize = rev - 1;
            tile_strides_mut[idx] = tile_total;
            tile_total = std.math.mul(usize, tile_total, tile_counts_mut[idx]) catch return StorageError.InvalidArgument;
        }

        self.tile_counts = self.tile_counts_storage.constSlice();
        self.tile_strides = self.tile_strides_storage.constSlice();

        // Quant constraints: for now, enforce that every tile's innermost dimension is compatible.
        // This is required so matmul kernels can be called on tiles with k % 32 == 0.
        const di = dtype.info();
        if (di.is_quantized) {
            // v0 rule: enforce block alignment on the first axis.
            if (self.shape[0] % di.block_elems != 0) return StorageError.InvalidArgument;
            if (self.tile_shape[0] % di.block_elems != 0) return StorageError.InvalidArgument;
            const rem: usize = self.shape[0] % self.tile_shape[0];
            if (rem != 0 and (rem % di.block_elems != 0)) return StorageError.InvalidArgument;
        }

        // Allocate metadata (offsets + lens) in one block.
        const meta: []usize = allocator.alloc(usize, tile_total * 2) catch return StorageError.OutOfMemory;
        const tile_offsets: []usize = meta[0..tile_total];
        const tile_lens: []usize = meta[tile_total..];

        // Precompute offsets and byte lens.
        var off: usize = 0;
        var idx: usize = 0;

        const elem_bytes: usize = if (di.is_quantized) 0 else di.block_bytes;

        var tile_coords: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(allocator, self.shape.len) catch {
            allocator.free(meta);
            return StorageError.OutOfMemory;
        };
        defer tile_coords.deinit();

        var tile_dims: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(allocator, self.shape.len) catch {
            allocator.free(meta);
            return StorageError.OutOfMemory;
        };
        defer tile_dims.deinit();

        while (idx < tile_total) : (idx += 1) {
            try decodeTileCoords(idx, self.tile_counts, self.tile_strides, tile_coords.slice());
            try computeTileDimsND(self.shape, self.tile_shape, tile_coords.constSlice(), tile_dims.slice());

            const tile_elems: usize = try mulAll(tile_dims.constSlice());
            const tile_bytes: usize = utils.requiredBytesForElems(dtype, tile_elems) catch {
                allocator.free(meta);
                return StorageError.InvalidArgument;
            };

            off = alignForward(off, opts.tile_alignment);
            tile_offsets[idx] = off;
            tile_lens[idx] = tile_bytes;
            off = std.math.add(usize, off, tile_bytes) catch {
                allocator.free(meta);
                return StorageError.InvalidArgument;
            };
        }

        // Allocate backing buffer aligned for SIMD-friendly accesses.
        const data: []align(64) u8 = allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(64), off) catch return StorageError.OutOfMemory;
        @memset(data, 0);

        // Keep compiler from complaining about elem_bytes in unused branches.
        _ = elem_bytes;

        self.meta = meta;
        self.tile_offsets = tile_offsets;
        self.tile_lens = tile_lens;
        self.data = data;
        self.owns_data = true;
        return;
    }

    pub fn initBorrowed(
        self: *Self,
        allocator: std.mem.Allocator,
        dtype: DType,
        shape_in: []const usize,
        tile_shape_in: []const usize,
        tile_offsets_in: []const usize,
        tile_lens_in: []const usize,
        data_in: []align(64) u8,
        opts: InitOptions,
    ) StorageError!void {
        if (shape_in.len == 0) return StorageError.InvalidArgument;
        if (tile_shape_in.len != shape_in.len) return StorageError.InvalidArgument;
        if (tile_offsets_in.len != tile_lens_in.len) return StorageError.InvalidArgument;

        const rank: u8 = @intCast(shape_in.len);

        var shape_storage: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initFromSlice(allocator, shape_in) catch return StorageError.OutOfMemory;
        var tile_shape_storage: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initFromSlice(allocator, tile_shape_in) catch {
            shape_storage.deinit();
            return StorageError.OutOfMemory;
        };

        var tile_counts_storage: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(allocator, shape_in.len) catch {
            shape_storage.deinit();
            tile_shape_storage.deinit();
            return StorageError.OutOfMemory;
        };

        var tile_strides_storage: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(allocator, shape_in.len) catch {
            shape_storage.deinit();
            tile_shape_storage.deinit();
            tile_counts_storage.deinit();
            return StorageError.OutOfMemory;
        };
        var moved: bool = false;
        errdefer if (!moved) {
            shape_storage.deinit();
            tile_shape_storage.deinit();
            tile_counts_storage.deinit();
            tile_strides_storage.deinit();
        };

        self.* = .{
            .allocator = allocator,
            .dtype = dtype,
            .rank = rank,
            .shape = &[_]usize{},
            .tile_shape = &[_]usize{},
            .tile_counts = &[_]usize{},
            .tile_strides = &[_]usize{},
            .shape_storage = shape_storage,
            .tile_shape_storage = tile_shape_storage,
            .tile_counts_storage = tile_counts_storage,
            .tile_strides_storage = tile_strides_storage,
            .meta = &[_]usize{},
            .tile_offsets = &[_]usize{},
            .tile_lens = &[_]usize{},
            .data = data_in,
            .owns_data = false,
            .tile_alignment = opts.tile_alignment,
        };
        moved = true;
        errdefer self.deinit();

        self.shape = self.shape_storage.constSlice();
        self.tile_shape = self.tile_shape_storage.constSlice();

        var tile_counts_mut: []usize = self.tile_counts_storage.slice();
        var tile_strides_mut: []usize = self.tile_strides_storage.slice();

        var d: usize = 0;
        while (d < self.shape.len) : (d += 1) {
            if (self.shape[d] == 0) return StorageError.InvalidArgument;
            if (self.tile_shape[d] == 0) return StorageError.InvalidArgument;
            const count: usize = (self.shape[d] + self.tile_shape[d] - 1) / self.tile_shape[d];
            if (count == 0) return StorageError.InvalidArgument;
            tile_counts_mut[d] = count;
        }

        var tile_total: usize = 1;
        var rev: usize = self.shape.len;
        while (rev > 0) : (rev -= 1) {
            const idx: usize = rev - 1;
            tile_strides_mut[idx] = tile_total;
            tile_total = std.math.mul(usize, tile_total, tile_counts_mut[idx]) catch return StorageError.InvalidArgument;
        }

        if (tile_total != tile_offsets_in.len) return StorageError.InvalidArgument;

        self.tile_counts = self.tile_counts_storage.constSlice();
        self.tile_strides = self.tile_strides_storage.constSlice();

        const di = dtype.info();
        if (di.is_quantized) {
            if (self.shape[0] % di.block_elems != 0) return StorageError.InvalidArgument;
            if (self.tile_shape[0] % di.block_elems != 0) return StorageError.InvalidArgument;
            const rem: usize = self.shape[0] % self.tile_shape[0];
            if (rem != 0 and (rem % di.block_elems != 0)) return StorageError.InvalidArgument;
        }

        const meta: []usize = allocator.alloc(usize, tile_total * 2) catch return StorageError.OutOfMemory;
        const tile_offsets: []usize = meta[0..tile_total];
        const tile_lens: []usize = meta[tile_total..];
        errdefer allocator.free(meta);

        @memcpy(tile_offsets, tile_offsets_in);
        @memcpy(tile_lens, tile_lens_in);

        for (tile_offsets, tile_lens) |off, len| {
            if (off % opts.tile_alignment != 0) return StorageError.InvalidArgument;
            if (off > data_in.len or len > data_in.len - off) return StorageError.InvalidArgument;
        }

        self.meta = meta;
        self.tile_offsets = tile_offsets;
        self.tile_lens = tile_lens;
    }

    pub fn deinit(self: *Self) void {
        if (self.meta.len != 0) {
            self.allocator.free(self.meta);
            self.meta = &[_]usize{};
            self.tile_offsets = &[_]usize{};
            self.tile_lens = &[_]usize{};
        }
        if (self.data.len != 0 and self.owns_data) {
            self.allocator.free(self.data);
        }
        self.data = &[_]u8{};
        self.owns_data = true;
        self.shape_storage.deinit();
        self.tile_shape_storage.deinit();
        self.tile_counts_storage.deinit();
        self.tile_strides_storage.deinit();
        self.shape = &[_]usize{};
        self.tile_shape = &[_]usize{};
        self.tile_counts = &[_]usize{};
        self.tile_strides = &[_]usize{};
        self.rank = 0;
    }

    pub fn tileCountTotal(self: Self) usize {
        return self.tile_offsets.len;
    }

    pub fn tileIndex(self: Self, ti0: usize, ti1: usize) StorageError!usize {
        if (self.rank == 1) {
            if (ti1 != 0) return StorageError.InvalidArgument;
            if (ti0 >= self.tile_counts[0]) return StorageError.InvalidArgument;
            return tileIndexFromCoords(self.tile_strides, self.tile_counts, &[_]usize{ti0});
        }
        if (self.rank == 2) {
            if (ti0 >= self.tile_counts[0]) return StorageError.InvalidArgument;
            if (ti1 >= self.tile_counts[1]) return StorageError.InvalidArgument;
            return tileIndexFromCoords(self.tile_strides, self.tile_counts, &[_]usize{ ti0, ti1 });
        }
        return StorageError.InvalidArgument;
    }

    pub fn tileDims(self: Self, ti0: usize, ti1: usize) StorageError![2]usize {
        if (self.rank == 1) {
            if (ti1 != 0) return StorageError.InvalidArgument;
            if (ti0 >= self.tile_counts[0]) return StorageError.InvalidArgument;
            var dims: [2]usize = .{ 0, 1 };
            try computeTileDimsND(self.shape, self.tile_shape, &[_]usize{ti0}, dims[0..1]);
            return dims;
        }
        if (self.rank == 2) {
            if (ti0 >= self.tile_counts[0]) return StorageError.InvalidArgument;
            if (ti1 >= self.tile_counts[1]) return StorageError.InvalidArgument;
            var dims: [2]usize = .{ 0, 0 };
            try computeTileDimsND(self.shape, self.tile_shape, &[_]usize{ ti0, ti1 }, dims[0..2]);
            return dims;
        }
        return StorageError.InvalidArgument;
    }

    pub fn acquireTileConst(self: *const Self, ti0: usize, ti1: usize) StorageError!TileViewConst {
        if (self.rank > 2) return StorageError.InvalidArgument;
        const idx: usize = try self.tileIndex(ti0, ti1);
        const off: usize = self.tile_offsets[idx];
        const len: usize = self.tile_lens[idx];
        if (off + len > self.data.len) return StorageError.InvalidArgument;

        const dims: [2]usize = try self.tileDims(ti0, ti1);
        const di = self.dtype.info();
        const elem_bytes: usize = if (di.is_quantized) 0 else di.block_bytes;
        return TileViewConst.init(self.data[off .. off + len], self.dtype, self.rank, dims[0..@as(usize, self.rank)], elem_bytes);
    }

    pub fn acquireTileConstLinear(self: *const Self, tile_index: usize) StorageError!TileViewConst {
        if (tile_index >= self.tile_offsets.len) return StorageError.InvalidArgument;
        if (@as(usize, self.rank) > INLINE_RANK) return StorageError.InvalidArgument;

        var tile_coords: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, @as(usize, self.rank)) catch return StorageError.OutOfMemory;
        defer tile_coords.deinit();
        var tile_dims: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, @as(usize, self.rank)) catch return StorageError.OutOfMemory;
        defer tile_dims.deinit();

        try decodeTileCoords(tile_index, self.tile_counts, self.tile_strides, tile_coords.slice());
        try computeTileDimsND(self.shape, self.tile_shape, tile_coords.constSlice(), tile_dims.slice());

        const off: usize = self.tile_offsets[tile_index];
        const len: usize = self.tile_lens[tile_index];
        if (off + len > self.data.len) return StorageError.InvalidArgument;

        const di = self.dtype.info();
        const elem_bytes: usize = if (di.is_quantized) 0 else di.block_bytes;
        return TileViewConst.init(self.data[off .. off + len], self.dtype, self.rank, tile_dims.constSlice(), elem_bytes);
    }

    pub fn acquireTileMut(self: *Self, ti0: usize, ti1: usize) StorageError!TileViewMut {
        if (self.rank > 2) return StorageError.InvalidArgument;
        const idx: usize = try self.tileIndex(ti0, ti1);
        const off: usize = self.tile_offsets[idx];
        const len: usize = self.tile_lens[idx];
        if (off + len > self.data.len) return StorageError.InvalidArgument;

        const dims: [2]usize = try self.tileDims(ti0, ti1);
        const di = self.dtype.info();
        const elem_bytes: usize = if (di.is_quantized) 0 else di.block_bytes;
        return TileViewMut.init(self.data[off .. off + len], self.dtype, self.rank, dims[0..@as(usize, self.rank)], elem_bytes);
    }

    pub fn acquireTileMutLinear(self: *Self, tile_index: usize) StorageError!TileViewMut {
        if (tile_index >= self.tile_offsets.len) return StorageError.InvalidArgument;
        if (@as(usize, self.rank) > INLINE_RANK) return StorageError.InvalidArgument;

        var tile_coords: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, @as(usize, self.rank)) catch return StorageError.OutOfMemory;
        defer tile_coords.deinit();
        var tile_dims: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, @as(usize, self.rank)) catch return StorageError.OutOfMemory;
        defer tile_dims.deinit();

        try decodeTileCoords(tile_index, self.tile_counts, self.tile_strides, tile_coords.slice());
        try computeTileDimsND(self.shape, self.tile_shape, tile_coords.constSlice(), tile_dims.slice());

        const off: usize = self.tile_offsets[tile_index];
        const len: usize = self.tile_lens[tile_index];
        if (off + len > self.data.len) return StorageError.InvalidArgument;

        const di = self.dtype.info();
        const elem_bytes: usize = if (di.is_quantized) 0 else di.block_bytes;
        return TileViewMut.init(self.data[off .. off + len], self.dtype, self.rank, tile_dims.constSlice(), elem_bytes);
    }

    /// Writes a packed row-major scalar tensor into this tiled storage.
    ///
    /// For rank-1, `packed` is a contiguous vector.
    /// For rank-2, `packed` is row-major contiguous.
    pub fn writeFromPackedScalar(self: *Self, packed_bytes: []const u8) StorageError!void {
        if (self.dtype.info().is_quantized) return StorageError.InvalidArgument;

        const need_total: usize = self.requiredBytesPackedScalar();
        if (packed_bytes.len < need_total) return StorageError.InvalidArgument;

        const elem_bytes: usize = self.dtype.info().block_bytes;
        const shape: []const usize = self.shape;
        const tile_shape: []const usize = self.tile_shape;
        const tile_counts: []const usize = self.tile_counts;
        const tile_strides: []const usize = self.tile_strides;
        const rank: usize = @as(usize, self.rank);

        var packed_strides: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, rank) catch return StorageError.OutOfMemory;
        defer packed_strides.deinit();
        try computePackedStridesElems(shape, packed_strides.slice());

        var tile_coords: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, rank) catch return StorageError.OutOfMemory;
        defer tile_coords.deinit();
        var tile_dims: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, rank) catch return StorageError.OutOfMemory;
        defer tile_dims.deinit();
        var tile_local_strides: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, rank) catch return StorageError.OutOfMemory;
        defer tile_local_strides.deinit();
        var local_idx: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, rank) catch return StorageError.OutOfMemory;
        defer local_idx.deinit();

        var tile_index: usize = 0;
        while (tile_index < self.tile_offsets.len) : (tile_index += 1) {
            try decodeTileCoords(tile_index, tile_counts, tile_strides, tile_coords.slice());
            try computeTileDimsND(shape, tile_shape, tile_coords.constSlice(), tile_dims.slice());
            try computePackedStridesElems(tile_dims.constSlice(), tile_local_strides.slice());

            const tile_elems: usize = try mulAll(tile_dims.constSlice());
            const tile_bytes: usize = tile_elems * elem_bytes;
            const off: usize = self.tile_offsets[tile_index];
            const out_len: usize = self.tile_lens[tile_index];
            if (tile_bytes != out_len) return StorageError.InvalidArgument;

            var t_lin: usize = 0;
            while (t_lin < tile_elems) : (t_lin += 1) {
                try decodeLinearIndex(t_lin, tile_local_strides.constSlice(), tile_dims.constSlice(), local_idx.slice());

                var packed_lin: usize = 0;
                var d: usize = 0;
                while (d < rank) : (d += 1) {
                    const g: usize = tile_coords.constSlice()[d] * tile_shape[d] + local_idx.constSlice()[d];
                    if (g >= shape[d]) return StorageError.InvalidArgument;
                    packed_lin = std.math.add(usize, packed_lin, g * packed_strides.constSlice()[d]) catch return StorageError.InvalidArgument;
                }

                const src_off: usize = packed_lin * elem_bytes;
                const dst_off: usize = t_lin * elem_bytes;
                @memcpy(self.data[(off + dst_off)..(off + dst_off + elem_bytes)], packed_bytes[src_off .. src_off + elem_bytes]);
            }
        }
    }

    /// Reads tiled scalar storage back into packed row-major.
    pub fn readToPackedScalar(self: *const Self, out: []u8) StorageError!void {
        if (self.dtype.info().is_quantized) return StorageError.InvalidArgument;

        const need_total: usize = self.requiredBytesPackedScalar();
        if (out.len < need_total) return StorageError.InvalidArgument;

        const elem_bytes: usize = self.dtype.info().block_bytes;
        const shape: []const usize = self.shape;
        const tile_shape: []const usize = self.tile_shape;
        const tile_counts: []const usize = self.tile_counts;
        const tile_strides: []const usize = self.tile_strides;
        const rank: usize = @as(usize, self.rank);

        var packed_strides: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, rank) catch return StorageError.OutOfMemory;
        defer packed_strides.deinit();
        try computePackedStridesElems(shape, packed_strides.slice());

        var tile_coords: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, rank) catch return StorageError.OutOfMemory;
        defer tile_coords.deinit();
        var tile_dims: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, rank) catch return StorageError.OutOfMemory;
        defer tile_dims.deinit();
        var tile_local_strides: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, rank) catch return StorageError.OutOfMemory;
        defer tile_local_strides.deinit();
        var local_idx: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, rank) catch return StorageError.OutOfMemory;
        defer local_idx.deinit();

        var tile_index: usize = 0;
        while (tile_index < self.tile_offsets.len) : (tile_index += 1) {
            try decodeTileCoords(tile_index, tile_counts, tile_strides, tile_coords.slice());
            try computeTileDimsND(shape, tile_shape, tile_coords.constSlice(), tile_dims.slice());
            try computePackedStridesElems(tile_dims.constSlice(), tile_local_strides.slice());

            const tile_elems: usize = try mulAll(tile_dims.constSlice());
            const tile_bytes: usize = tile_elems * elem_bytes;
            const off: usize = self.tile_offsets[tile_index];
            const in_len: usize = self.tile_lens[tile_index];
            if (tile_bytes != in_len) return StorageError.InvalidArgument;

            var t_lin: usize = 0;
            while (t_lin < tile_elems) : (t_lin += 1) {
                try decodeLinearIndex(t_lin, tile_local_strides.constSlice(), tile_dims.constSlice(), local_idx.slice());

                var packed_lin: usize = 0;
                var d: usize = 0;
                while (d < rank) : (d += 1) {
                    const g: usize = tile_coords.constSlice()[d] * tile_shape[d] + local_idx.constSlice()[d];
                    if (g >= shape[d]) return StorageError.InvalidArgument;
                    packed_lin = std.math.add(usize, packed_lin, g * packed_strides.constSlice()[d]) catch return StorageError.InvalidArgument;
                }

                const dst_off: usize = packed_lin * elem_bytes;
                const src_off: usize = t_lin * elem_bytes;
                @memcpy(out[dst_off .. dst_off + elem_bytes], self.data[(off + src_off)..(off + src_off + elem_bytes)]);
            }
        }
    }

    /// Writes a packed quant tensor into this tiled storage.
    ///
    /// v0 packed quant convention (arbitrary rank):
    /// - treat axis-0 as blockized: shape_blocks[0] = shape[0] / block_elems
    /// - remaining axes are unchanged
    /// - packed bytes are row-major over shape_blocks with element size = block_bytes
    pub fn writeFromPackedQuant(self: *Self, packed_bytes: []const u8) StorageError!void {
        const di = self.dtype.info();
        if (!di.is_quantized) return StorageError.InvalidArgument;
        if (self.rank == 0) return StorageError.InvalidArgument;

        const total_elems: usize = try mulAll(self.shape);
        const need_total: usize = utils.requiredBytesForElems(self.dtype, total_elems) catch return StorageError.InvalidArgument;
        if (packed_bytes.len < need_total) return StorageError.InvalidArgument;

        const rank: usize = @as(usize, self.rank);

        var packed_block_shape: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, rank) catch return StorageError.OutOfMemory;
        defer packed_block_shape.deinit();
        var packed_block_strides: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, rank) catch return StorageError.OutOfMemory;
        defer packed_block_strides.deinit();
        try computeBlockShape(self.shape, di.block_elems, packed_block_shape.slice());
        try computePackedStridesElems(packed_block_shape.constSlice(), packed_block_strides.slice());

        var tile_coords: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, rank) catch return StorageError.OutOfMemory;
        defer tile_coords.deinit();
        var tile_dims: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, rank) catch return StorageError.OutOfMemory;
        defer tile_dims.deinit();
        var tile_block_shape: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, rank) catch return StorageError.OutOfMemory;
        defer tile_block_shape.deinit();
        var tile_block_strides: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, rank) catch return StorageError.OutOfMemory;
        defer tile_block_strides.deinit();
        var local_block_coords: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, rank) catch return StorageError.OutOfMemory;
        defer local_block_coords.deinit();

        var tile_index: usize = 0;
        while (tile_index < self.tile_offsets.len) : (tile_index += 1) {
            try decodeTileCoords(tile_index, self.tile_counts, self.tile_strides, tile_coords.slice());
            try computeTileDimsND(self.shape, self.tile_shape, tile_coords.constSlice(), tile_dims.slice());
            try computeBlockShape(tile_dims.constSlice(), di.block_elems, tile_block_shape.slice());
            try computePackedStridesElems(tile_block_shape.constSlice(), tile_block_strides.slice());

            const tile_blocks: usize = try mulAll(tile_block_shape.constSlice());
            const tile_bytes: usize = tile_blocks * di.block_bytes;
            const off: usize = self.tile_offsets[tile_index];
            const out_len: usize = self.tile_lens[tile_index];
            if (tile_bytes != out_len) return StorageError.InvalidArgument;

            var b_lin: usize = 0;
            while (b_lin < tile_blocks) : (b_lin += 1) {
                try decodeLinearIndex(b_lin, tile_block_strides.constSlice(), tile_block_shape.constSlice(), local_block_coords.slice());

                var packed_block_lin: usize = 0;
                var d: usize = 0;
                while (d < rank) : (d += 1) {
                    const base: usize = if (d == 0) blk: {
                        const tile_start: usize = tile_coords.constSlice()[0] * self.tile_shape[0];
                        if (tile_start % di.block_elems != 0) return StorageError.InvalidArgument;
                        break :blk (tile_start / di.block_elems) + local_block_coords.constSlice()[0];
                    } else blk: {
                        break :blk tile_coords.constSlice()[d] * self.tile_shape[d] + local_block_coords.constSlice()[d];
                    };
                    if (base >= packed_block_shape.constSlice()[d]) return StorageError.InvalidArgument;
                    packed_block_lin = std.math.add(usize, packed_block_lin, base * packed_block_strides.constSlice()[d]) catch return StorageError.InvalidArgument;
                }

                const src_off: usize = packed_block_lin * di.block_bytes;
                const dst_off: usize = b_lin * di.block_bytes;

                @memcpy(
                    self.data[(off + dst_off)..(off + dst_off + di.block_bytes)],
                    packed_bytes[src_off .. src_off + di.block_bytes],
                );
            }
        }
    }

    /// Reads tiled quant storage back into the packed quant convention.
    pub fn readToPackedQuant(self: *const Self, out: []u8) StorageError!void {
        const di = self.dtype.info();
        if (!di.is_quantized) return StorageError.InvalidArgument;
        if (self.rank == 0) return StorageError.InvalidArgument;

        const total_elems: usize = try mulAll(self.shape);
        const need_total: usize = utils.requiredBytesForElems(self.dtype, total_elems) catch return StorageError.InvalidArgument;
        if (out.len < need_total) return StorageError.InvalidArgument;

        const rank: usize = @as(usize, self.rank);

        var packed_block_shape: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, rank) catch return StorageError.OutOfMemory;
        defer packed_block_shape.deinit();
        var packed_block_strides: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, rank) catch return StorageError.OutOfMemory;
        defer packed_block_strides.deinit();
        try computeBlockShape(self.shape, di.block_elems, packed_block_shape.slice());
        try computePackedStridesElems(packed_block_shape.constSlice(), packed_block_strides.slice());

        var tile_coords: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, rank) catch return StorageError.OutOfMemory;
        defer tile_coords.deinit();
        var tile_dims: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, rank) catch return StorageError.OutOfMemory;
        defer tile_dims.deinit();
        var tile_block_shape: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, rank) catch return StorageError.OutOfMemory;
        defer tile_block_shape.deinit();
        var tile_block_strides: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, rank) catch return StorageError.OutOfMemory;
        defer tile_block_strides.deinit();
        var local_block_coords: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, rank) catch return StorageError.OutOfMemory;
        defer local_block_coords.deinit();

        var tile_index: usize = 0;
        while (tile_index < self.tile_offsets.len) : (tile_index += 1) {
            try decodeTileCoords(tile_index, self.tile_counts, self.tile_strides, tile_coords.slice());
            try computeTileDimsND(self.shape, self.tile_shape, tile_coords.constSlice(), tile_dims.slice());
            try computeBlockShape(tile_dims.constSlice(), di.block_elems, tile_block_shape.slice());
            try computePackedStridesElems(tile_block_shape.constSlice(), tile_block_strides.slice());

            const tile_blocks: usize = try mulAll(tile_block_shape.constSlice());
            const tile_bytes: usize = tile_blocks * di.block_bytes;
            const off: usize = self.tile_offsets[tile_index];
            const in_len: usize = self.tile_lens[tile_index];
            if (tile_bytes != in_len) return StorageError.InvalidArgument;

            var b_lin: usize = 0;
            while (b_lin < tile_blocks) : (b_lin += 1) {
                try decodeLinearIndex(b_lin, tile_block_strides.constSlice(), tile_block_shape.constSlice(), local_block_coords.slice());

                var packed_block_lin: usize = 0;
                var d: usize = 0;
                while (d < rank) : (d += 1) {
                    const base: usize = if (d == 0) blk: {
                        const tile_start: usize = tile_coords.constSlice()[0] * self.tile_shape[0];
                        if (tile_start % di.block_elems != 0) return StorageError.InvalidArgument;
                        break :blk (tile_start / di.block_elems) + local_block_coords.constSlice()[0];
                    } else blk: {
                        break :blk tile_coords.constSlice()[d] * self.tile_shape[d] + local_block_coords.constSlice()[d];
                    };
                    if (base >= packed_block_shape.constSlice()[d]) return StorageError.InvalidArgument;
                    packed_block_lin = std.math.add(usize, packed_block_lin, base * packed_block_strides.constSlice()[d]) catch return StorageError.InvalidArgument;
                }

                const dst_off: usize = packed_block_lin * di.block_bytes;
                const src_off: usize = b_lin * di.block_bytes;

                @memcpy(
                    out[dst_off .. dst_off + di.block_bytes],
                    self.data[(off + src_off)..(off + src_off + di.block_bytes)],
                );
            }
        }
    }

    fn requiredBytesPackedScalar(self: Self) usize {
        const di = self.dtype.info();
        const elem_bytes: usize = di.block_bytes;
        const elems: usize = mulAll(self.shape) catch return 0;
        return elems * elem_bytes;
    }
};

fn mulAll(vals: []const usize) StorageError!usize {
    if (vals.len == 0) return StorageError.InvalidArgument;
    var acc: usize = 1;
    for (vals) |v| {
        acc = std.math.mul(usize, acc, v) catch return StorageError.InvalidArgument;
    }
    return acc;
}

fn computePackedStridesElems(shape: []const usize, out: []usize) StorageError!void {
    if (out.len != shape.len) return StorageError.InvalidArgument;
    if (shape.len == 0) return StorageError.InvalidArgument;

    var stride: usize = 1;
    var d: usize = shape.len;
    while (d > 0) : (d -= 1) {
        const idx: usize = d - 1;
        out[idx] = stride;
        stride = std.math.mul(usize, stride, shape[idx]) catch return StorageError.InvalidArgument;
    }
}

fn computeBlockShape(shape: []const usize, block_elems: usize, out: []usize) StorageError!void {
    if (out.len != shape.len) return StorageError.InvalidArgument;
    if (shape.len == 0) return StorageError.InvalidArgument;
    if (block_elems == 0) return StorageError.InvalidArgument;

    if (shape[0] % block_elems != 0) return StorageError.InvalidArgument;
    out[0] = shape[0] / block_elems;

    var d: usize = 1;
    while (d < shape.len) : (d += 1) {
        out[d] = shape[d];
    }
}

fn computeTileDimsND(shape: []const usize, tile_shape: []const usize, tile_coords: []const usize, out: []usize) StorageError!void {
    if (shape.len != tile_shape.len) return StorageError.InvalidArgument;
    if (tile_coords.len != shape.len) return StorageError.InvalidArgument;
    if (out.len != shape.len) return StorageError.InvalidArgument;

    var d: usize = 0;
    while (d < shape.len) : (d += 1) {
        const start: usize = tile_coords[d] * tile_shape[d];
        if (start >= shape[d]) return StorageError.InvalidArgument;
        out[d] = @min(tile_shape[d], shape[d] - start);
    }
}

fn decodeTileCoords(tile_index: usize, tile_counts: []const usize, tile_strides: []const usize, out: []usize) StorageError!void {
    if (tile_counts.len != tile_strides.len) return StorageError.InvalidArgument;
    if (out.len != tile_counts.len) return StorageError.InvalidArgument;

    var d: usize = 0;
    while (d < tile_counts.len) : (d += 1) {
        const stride: usize = tile_strides[d];
        if (stride == 0) return StorageError.InvalidArgument;
        const v: usize = tile_index / stride;
        out[d] = v % tile_counts[d];
    }
}

fn tileIndexFromCoords(tile_strides: []const usize, tile_counts: []const usize, coords: []const usize) StorageError!usize {
    if (coords.len != tile_counts.len) return StorageError.InvalidArgument;
    if (tile_strides.len != tile_counts.len) return StorageError.InvalidArgument;

    var idx: usize = 0;
    var d: usize = 0;
    while (d < coords.len) : (d += 1) {
        if (coords[d] >= tile_counts[d]) return StorageError.InvalidArgument;
        idx = std.math.add(usize, idx, coords[d] * tile_strides[d]) catch return StorageError.InvalidArgument;
    }
    return idx;
}

fn decodeLinearIndex(linear: usize, strides: []const usize, dims: []const usize, out: []usize) StorageError!void {
    if (strides.len != dims.len) return StorageError.InvalidArgument;
    if (out.len != dims.len) return StorageError.InvalidArgument;

    var rem: usize = linear;
    var d: usize = 0;
    while (d < dims.len) : (d += 1) {
        const stride: usize = strides[d];
        if (stride == 0) return StorageError.InvalidArgument;
        const v: usize = rem / stride;
        if (v >= dims[d]) return StorageError.InvalidArgument;
        out[d] = v;
        rem -= out[d] * stride;
    }
}
