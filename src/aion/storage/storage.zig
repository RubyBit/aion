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
/// Quantized layout:
/// - A quantized tensor has one *block axis* (`quant_axis`, configurable; default 0).
///   Along this axis, every `block_elems` consecutive elements form one on-disk block of
///   `block_bytes` bytes. Along all other axes, elements are indexed element-wise.
/// - The `shape[quant_axis]` and `tile_shape[quant_axis]` must both be multiples of `block_elems`.
/// - Packed-quant byte layout (see `writeFromPackedQuant` / `readToPackedQuant`) is row-major
///   over `block_shape` (the shape with `shape[quant_axis]` replaced by `shape[quant_axis] / block_elems`),
///   with each element being one `block_bytes` block.
/// - Typical uses:
///   - `quant_axis = rank-2` for matmul-B weights `[..., K, N]` → blocks along the reduction axis K.
///   - `quant_axis = last` for embedding tables `[V, D]` → blocks along the feature axis D,
///     so one row of the table is a contiguous run of `D / block_elems` blocks.
pub const TiledTensor = struct {
    allocator: std.mem.Allocator,

    dtype: DType,
    rank: u8,
    /// Block axis for quantized tensors. Ignored for scalar dtypes.
    quant_axis: u8 = 0,
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
        /// Block axis for quantized tensors. Ignored for scalar dtypes.
        /// Must be < rank when the dtype is quantized.
        quant_axis: u8 = 0,
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
            .quant_axis = opts.quant_axis,
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

        const di = dtype.info();
        if (di.is_quantized) {
            try validateQuantAxisAlignment(self.shape, self.tile_shape, di.block_elems, self.quant_axis);
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
            .quant_axis = opts.quant_axis,
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
            try validateQuantAxisAlignment(self.shape, self.tile_shape, di.block_elems, self.quant_axis);
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

    pub fn isOwned(self: *const Self) bool {
        return self.owns_data;
    }

    pub fn promoteToOwned(self: *Self) StorageError!bool {
        if (self.owns_data) return false;

        const owned_copy: []align(64) u8 = self.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(64), self.data.len) catch return StorageError.OutOfMemory;
        errdefer self.allocator.free(owned_copy);

        @memcpy(owned_copy, self.data);
        self.data = owned_copy;
        self.owns_data = true;
        return true;
    }

    /// Grow one axis while preserving existing scalar values.
    ///
    /// v1 scope:
    /// - scalar dtypes only (no quantized remap yet)
    /// - growth only (`new_size >= current`)
    /// - shape rank is preserved
    pub fn growAxisPreserveScalar(self: *Self, axis: usize, new_size: usize) StorageError!void {
        const rank_usize: usize = @as(usize, self.rank);
        if (rank_usize == 0 or axis >= rank_usize) return StorageError.InvalidArgument;
        if (self.dtype.info().is_quantized) return StorageError.InvalidArgument;

        const old_size: usize = self.shape[axis];
        if (new_size < old_size) return StorageError.InvalidArgument;
        if (new_size == old_size) return;

        var new_shape_mem: [INLINE_RANK]usize = .{0} ** INLINE_RANK;
        var d: usize = 0;
        while (d < rank_usize) : (d += 1) {
            new_shape_mem[d] = self.shape[d];
        }
        new_shape_mem[axis] = new_size;
        const new_shape: []const usize = new_shape_mem[0..rank_usize];

        const elem_bytes: usize = self.dtype.info().block_bytes;
        const old_elems: usize = try mulAll(self.shape);
        const new_elems: usize = try mulAll(new_shape);

        const old_bytes: usize = std.math.mul(usize, old_elems, elem_bytes) catch return StorageError.InvalidArgument;
        const new_bytes: usize = std.math.mul(usize, new_elems, elem_bytes) catch return StorageError.InvalidArgument;

        const old_packed: []u8 = self.allocator.alloc(u8, old_bytes) catch return StorageError.OutOfMemory;
        defer self.allocator.free(old_packed);
        try self.readToPackedScalar(old_packed);

        const new_packed: []u8 = self.allocator.alloc(u8, new_bytes) catch return StorageError.OutOfMemory;
        defer self.allocator.free(new_packed);
        @memset(new_packed, 0);

        var old_strides_mem: [INLINE_RANK]usize = .{0} ** INLINE_RANK;
        var new_strides_mem: [INLINE_RANK]usize = .{0} ** INLINE_RANK;
        try computePackedStridesElems(self.shape, old_strides_mem[0..rank_usize]);
        try computePackedStridesElems(new_shape, new_strides_mem[0..rank_usize]);

        var coords_mem: [INLINE_RANK]usize = .{0} ** INLINE_RANK;
        var old_lin: usize = 0;
        while (old_lin < old_elems) : (old_lin += 1) {
            try decodeLinearIndex(old_lin, old_strides_mem[0..rank_usize], self.shape, coords_mem[0..rank_usize]);

            var new_lin: usize = 0;
            d = 0;
            while (d < rank_usize) : (d += 1) {
                new_lin = std.math.add(usize, new_lin, coords_mem[d] * new_strides_mem[d]) catch return StorageError.InvalidArgument;
            }

            const src_off: usize = std.math.mul(usize, old_lin, elem_bytes) catch return StorageError.InvalidArgument;
            const dst_off: usize = std.math.mul(usize, new_lin, elem_bytes) catch return StorageError.InvalidArgument;
            @memcpy(new_packed[dst_off .. dst_off + elem_bytes], old_packed[src_off .. src_off + elem_bytes]);
        }

        var new_tensor: TiledTensor = undefined;
        try new_tensor.init(
            self.allocator,
            self.dtype,
            new_shape,
            self.tile_shape,
            .{ .tile_alignment = self.tile_alignment },
        );
        errdefer new_tensor.deinit();

        try new_tensor.writeFromPackedScalar(new_packed);

        self.deinit();
        self.* = new_tensor;

        // Rebind slices to this instance's storage after moving from a stack-local temp.
        self.shape = self.shape_storage.constSlice();
        self.tile_shape = self.tile_shape_storage.constSlice();
        self.tile_counts = self.tile_counts_storage.constSlice();
        self.tile_strides = self.tile_strides_storage.constSlice();
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

        // Fast path: single tile means packed row-major order is identical to the
        // physical tile bytes.
        if (self.tile_offsets.len == 1) {
            const off0: usize = self.tile_offsets[0];
            if (self.tile_lens[0] != need_total) return StorageError.InvalidArgument;
            if (off0 + need_total > self.data.len) return StorageError.InvalidArgument;
            @memcpy(self.data[off0 .. off0 + need_total], packed_bytes[0..need_total]);
            return;
        }

        // Fast path: rank-1 packed vector to tiled vector.
        if (rank == 1) {
            const ts0: usize = tile_shape[0];
            var ti0: usize = 0;
            while (ti0 < tile_counts[0]) : (ti0 += 1) {
                const idx: usize = ti0 * tile_strides[0];
                const g0: usize = ti0 * ts0;
                const n0: usize = @min(ts0, shape[0] - g0);

                const tile_bytes: usize = n0 * elem_bytes;
                const dst_off: usize = self.tile_offsets[idx];
                if (self.tile_lens[idx] != tile_bytes) return StorageError.InvalidArgument;
                if (dst_off + tile_bytes > self.data.len) return StorageError.InvalidArgument;

                const src_off: usize = g0 * elem_bytes;
                @memcpy(self.data[dst_off .. dst_off + tile_bytes], packed_bytes[src_off .. src_off + tile_bytes]);
            }
            return;
        }

        // Fast path: rank-2 packed row-major matrix to tiled matrix.
        if (rank == 2) {
            const rows: usize = shape[0];
            const cols: usize = shape[1];
            const ts0: usize = tile_shape[0];
            const ts1: usize = tile_shape[1];

            var ti0: usize = 0;
            while (ti0 < tile_counts[0]) : (ti0 += 1) {
                const row0: usize = ti0 * ts0;
                const m_tile: usize = @min(ts0, rows - row0);

                var ti1: usize = 0;
                while (ti1 < tile_counts[1]) : (ti1 += 1) {
                    const idx: usize = ti0 * tile_strides[0] + ti1 * tile_strides[1];
                    const col0: usize = ti1 * ts1;
                    const n_tile: usize = @min(ts1, cols - col0);

                    const tile_bytes: usize = (m_tile * n_tile) * elem_bytes;
                    const dst_tile_off: usize = self.tile_offsets[idx];
                    if (self.tile_lens[idx] != tile_bytes) return StorageError.InvalidArgument;
                    if (dst_tile_off + tile_bytes > self.data.len) return StorageError.InvalidArgument;

                    const row_bytes: usize = n_tile * elem_bytes;
                    var r: usize = 0;
                    while (r < m_tile) : (r += 1) {
                        const src_elem_off: usize = (row0 + r) * cols + col0;
                        const src_off: usize = src_elem_off * elem_bytes;
                        const dst_off: usize = dst_tile_off + r * row_bytes;
                        @memcpy(self.data[dst_off .. dst_off + row_bytes], packed_bytes[src_off .. src_off + row_bytes]);
                    }
                }
            }
            return;
        }

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
    /// Packed-quant convention (arbitrary rank):
    /// - Along `self.quant_axis`, every `block_elems` consecutive elements form one
    ///   `block_bytes`-sized block. Block-space shape is therefore
    ///   `shape` with `shape[quant_axis]` replaced by `shape[quant_axis] / block_elems`.
    /// - `packed_bytes` is row-major over that block-space shape, each element being one block.
    pub fn writeFromPackedQuant(self: *Self, packed_bytes: []const u8) StorageError!void {
        return self.forEachQuantBlock(packed_bytes, null);
    }

    /// Reads tiled quant storage back into the packed quant convention.
    pub fn readToPackedQuant(self: *const Self, out: []u8) StorageError!void {
        return @constCast(self).forEachQuantBlock(null, out);
    }

    /// Iterate every quant block in this tensor and copy it between the tiled backing
    /// buffer and a packed-quant byte buffer in the direction implied by which buffer
    /// is non-null. Exactly one of `packed_src` / `packed_dst` must be non-null.
    fn forEachQuantBlock(
        self: *Self,
        packed_src: ?[]const u8,
        packed_dst: ?[]u8,
    ) StorageError!void {
        std.debug.assert((packed_src == null) != (packed_dst == null));

        const di = self.dtype.info();
        if (!di.is_quantized) return StorageError.InvalidArgument;
        if (self.rank == 0) return StorageError.InvalidArgument;

        const rank: usize = @as(usize, self.rank);
        const quant_axis: usize = @as(usize, self.quant_axis);
        if (quant_axis >= rank) return StorageError.InvalidArgument;

        const total_elems: usize = try mulAll(self.shape);
        const need_total: usize = utils.requiredBytesForElems(self.dtype, total_elems) catch return StorageError.InvalidArgument;
        const packed_len: usize = if (packed_src) |s| s.len else packed_dst.?.len;
        if (packed_len < need_total) return StorageError.InvalidArgument;

        var packed_block_shape: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, rank) catch return StorageError.OutOfMemory;
        defer packed_block_shape.deinit();
        var packed_block_strides: SmallVec(usize, INLINE_RANK) = SmallVec(usize, INLINE_RANK).initWithLen(self.allocator, rank) catch return StorageError.OutOfMemory;
        defer packed_block_strides.deinit();
        try computeBlockShapeAxis(self.shape, di.block_elems, quant_axis, packed_block_shape.slice());
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
            try computeBlockShapeAxis(tile_dims.constSlice(), di.block_elems, quant_axis, tile_block_shape.slice());
            try computePackedStridesElems(tile_block_shape.constSlice(), tile_block_strides.slice());

            const tile_blocks: usize = try mulAll(tile_block_shape.constSlice());
            const tile_bytes: usize = tile_blocks * di.block_bytes;
            const off: usize = self.tile_offsets[tile_index];
            const slot_len: usize = self.tile_lens[tile_index];
            if (tile_bytes != slot_len) return StorageError.InvalidArgument;

            var b_lin: usize = 0;
            while (b_lin < tile_blocks) : (b_lin += 1) {
                try decodeLinearIndex(b_lin, tile_block_strides.constSlice(), tile_block_shape.constSlice(), local_block_coords.slice());

                const packed_block_lin: usize = try globalBlockIndex(
                    self.tile_shape,
                    tile_coords.constSlice(),
                    local_block_coords.constSlice(),
                    packed_block_strides.constSlice(),
                    packed_block_shape.constSlice(),
                    di.block_elems,
                    quant_axis,
                );

                const packed_off: usize = packed_block_lin * di.block_bytes;
                const tile_off: usize = b_lin * di.block_bytes;

                if (packed_src) |src| {
                    @memcpy(
                        self.data[(off + tile_off)..(off + tile_off + di.block_bytes)],
                        src[packed_off .. packed_off + di.block_bytes],
                    );
                } else {
                    const dst = packed_dst.?;
                    @memcpy(
                        dst[packed_off .. packed_off + di.block_bytes],
                        self.data[(off + tile_off)..(off + tile_off + di.block_bytes)],
                    );
                }
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

fn computeBlockShapeAxis(shape: []const usize, block_elems: usize, quant_axis: usize, out: []usize) StorageError!void {
    if (out.len != shape.len) return StorageError.InvalidArgument;
    if (shape.len == 0) return StorageError.InvalidArgument;
    if (block_elems == 0) return StorageError.InvalidArgument;
    if (quant_axis >= shape.len) return StorageError.InvalidArgument;

    var d: usize = 0;
    while (d < shape.len) : (d += 1) {
        if (d == quant_axis) {
            if (shape[d] % block_elems != 0) return StorageError.InvalidArgument;
            out[d] = shape[d] / block_elems;
        } else {
            out[d] = shape[d];
        }
    }
}

fn validateQuantAxisAlignment(shape: []const usize, tile_shape: []const usize, block_elems: usize, quant_axis_u8: u8) StorageError!void {
    const quant_axis: usize = @as(usize, quant_axis_u8);
    if (quant_axis >= shape.len) return StorageError.InvalidArgument;
    if (block_elems == 0) return StorageError.InvalidArgument;
    if (shape[quant_axis] % block_elems != 0) return StorageError.InvalidArgument;
    if (tile_shape[quant_axis] % block_elems != 0) return StorageError.InvalidArgument;
    const rem: usize = shape[quant_axis] % tile_shape[quant_axis];
    if (rem != 0 and (rem % block_elems != 0)) return StorageError.InvalidArgument;
}

/// Maps a local block coord (inside one tile) to the global block index in the
/// packed-quant byte layout. `quant_axis` is the block axis; along that axis the
/// global offset is measured in blocks (not elements), which requires the tile to
/// start on a block boundary.
fn globalBlockIndex(
    tile_shape: []const usize,
    tile_coords: []const usize,
    local_block_coords: []const usize,
    packed_block_strides: []const usize,
    packed_block_shape: []const usize,
    block_elems: usize,
    quant_axis: usize,
) StorageError!usize {
    const rank: usize = tile_shape.len;
    if (tile_coords.len != rank or local_block_coords.len != rank or
        packed_block_strides.len != rank or packed_block_shape.len != rank)
        return StorageError.InvalidArgument;
    if (quant_axis >= rank) return StorageError.InvalidArgument;

    var packed_block_lin: usize = 0;
    var d: usize = 0;
    while (d < rank) : (d += 1) {
        const base: usize = if (d == quant_axis) blk: {
            const tile_start: usize = tile_coords[d] * tile_shape[d];
            if (tile_start % block_elems != 0) return StorageError.InvalidArgument;
            break :blk (tile_start / block_elems) + local_block_coords[d];
        } else blk: {
            break :blk tile_coords[d] * tile_shape[d] + local_block_coords[d];
        };
        if (base >= packed_block_shape[d]) return StorageError.InvalidArgument;
        packed_block_lin = std.math.add(usize, packed_block_lin, base * packed_block_strides[d]) catch return StorageError.InvalidArgument;
    }
    return packed_block_lin;
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
