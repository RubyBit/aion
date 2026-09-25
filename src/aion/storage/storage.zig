// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const types = @import("../backend/types.zig");
const utils = @import("../backend/utils.zig");
const small_vec = @import("../small_vec.zig");
// Device-memory vtable + handle type. This module is std-only (no wgpu), so
// importing it keeps `storage.zig` GPU-API-agnostic and introduces no import
// cycle: the dependency edge is storage -> runtime/device_memory -> std.
const dm = @import("../runtime/device_memory.zig");

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

/// Which device a tensor's bytes currently live on. Move semantics: a tensor is
/// resident on exactly one device at a time. `.gpu` uses `index` to name which GPU
/// in the Context's device registry (multi-GPU); `.cpu` ignores `index`.
pub const DeviceRef = packed struct(u16) {
    kind: enum(u8) { cpu = 0, gpu = 1 } = .cpu,
    index: u8 = 0,

    pub fn isCpu(self: DeviceRef) bool {
        return self.kind == .cpu;
    }
    pub fn eql(a: DeviceRef, b: DeviceRef) bool {
        return a.kind == b.kind and a.index == b.index;
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
    /// Block order inside a tile. In-memory only: a layout pass sets it on the
    /// weight it derives, and nothing writes it to a package.
    block_order: types.QuantBlockOrder = .row_major,
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

    // --- Device residency (move semantics) ---
    // A tensor lives on exactly one device. On `.cpu`, bytes are in `data` and
    // `tile_handles` is empty. On a `.gpu` device (after `StorageManager.moveTensor`),
    // `data` is freed and the bytes live in per-tile device buffers named by
    // `tile_handles` (index-parallel to `tile_offsets`), allocated from `dev`.
    // Disambiguation: `data.len == 0 && device.kind == .cpu` means released/dead;
    // `data.len == 0 && device.kind == .gpu` means live on the device.
    device: DeviceRef = .{},
    /// Owned: freed (and each handle released via `dev`) in `deinit`.
    tile_handles: []dm.DeviceHandle = &[_]dm.DeviceHandle{},
    /// Borrowed (non-owning) device-memory interface for `tile_handles`.
    dev: ?dm.DeviceMemory = null,
    /// Workspace-only physical alias. Logical layout remains on this tensor;
    /// bytes and device handles are resolved through the canonical owner.
    backing_owner: ?u32 = null,
    /// Bytes reserved by this canonical physical backing.
    backing_bytes: usize = 0,
    workspace_owned: bool = false,
    /// Live compiled programs naming this tensor. Maintained by
    /// `StorageManager.retainTensor`/`releaseTensor`; see those for why the count
    /// lives on the tensor rather than in the model that compiled the program.
    program_refs: u32 = 0,
    /// Holders outside compiled programs — API handles, a builder binding it, a model
    /// with it bound as an input — of a tensor the public API created. Null for one a
    /// model or a pass made, which its maker frees.
    holders: ?u32 = null,

    const Self = @This();

    pub const InitOptions = struct {
        /// Align tile starts in the single backing allocation.
        tile_alignment: usize = 64,
        /// Block axis for quantized tensors. Ignored for scalar dtypes.
        /// Must be < rank when the dtype is quantized.
        quant_axis: u8 = 0,
        block_order: types.QuantBlockOrder = .row_major,
        /// Allocate zeroed host bytes. Off, the tensor starts without backing, for one
        /// whose bytes live elsewhere — a workspace slot, or tiles on a device.
        host_data: bool = true,
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
            .block_order = opts.block_order,
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

        // Keep compiler from complaining about elem_bytes in unused branches.
        _ = elem_bytes;

        self.meta = meta;
        self.tile_offsets = tile_offsets;
        self.tile_lens = tile_lens;
        self.owns_data = true;
        if (!opts.host_data) return;
        // Allocate backing buffer aligned for SIMD-friendly accesses.
        const data: []align(64) u8 = allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(64), off) catch return StorageError.OutOfMemory;
        @memset(data, 0);
        self.data = data;
        self.backing_bytes = data.len;
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
            .block_order = opts.block_order,
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
        self.backing_bytes = data_in.len;
    }

    /// Frees the (large) tile-backing buffer while keeping shape/tiling metadata,
    /// so the tensor id stays valid for metadata-only uses (e.g. external-binding
    /// shape/dtype validation) but holds no data. Used to reclaim a weight that an
    /// optimization pass has derived away; executing against it afterward is a bug.
    /// Idempotent.
    /// Free this tensor's HOST bytes, keeping metadata (and any device backing)
    /// so the id stays valid.
    ///
    /// Deliberately does not free device buffers: a tensor id can be shared by
    /// more than one compiled specialization (weight retargeting rewrites ids
    /// across cached programs), so releasing one entry's view must not pull the
    /// backing out from under another. Device memory is reclaimed at `deinit`.
    /// Freeing here made an evicted specialization's workspace reusable — and
    /// crashed a model whose entries share tensors.
    pub fn releaseData(self: *Self) void {
        if (self.backing_owner != null) return;
        if (self.tile_handles.len != 0) {
            if (self.dev) |d| {
                for (self.tile_handles) |h| d.free(h);
            }
            self.allocator.free(self.tile_handles);
            self.tile_handles = &[_]dm.DeviceHandle{};
        }
        self.dev = null;
        self.device = .{};
        if (self.data.len != 0 and self.owns_data) self.allocator.free(self.data);
        self.data = &[_]u8{};
        self.backing_bytes = 0;
    }

    pub fn deinit(self: *Self) void {
        // Device buffers travel with the tensor: release each handle through the
        // borrowed `dev`, then free the handle slice itself.
        if (self.tile_handles.len != 0) {
            if (self.dev) |d| {
                for (self.tile_handles) |h| d.free(h);
            }
            self.allocator.free(self.tile_handles);
            self.tile_handles = &[_]dm.DeviceHandle{};
        }
        self.dev = null;
        self.device = .{};
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
        self.backing_owner = null;
        self.backing_bytes = 0;
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

    /// True when the tensor's bytes live on a non-cpu device (host `data` freed).
    pub fn onDevice(self: *const Self) bool {
        return self.device.kind != .cpu;
    }

    /// Byte length of this tensor in packed row-major (block-major for quant) form —
    /// the device-independent intermediate used when migrating between devices.
    pub fn packedByteLen(self: *const Self) StorageError!usize {
        const elems: usize = try mulAll(self.shape);
        return utils.requiredBytesForElems(self.dtype, elems) catch StorageError.InvalidArgument;
    }

    pub fn promoteToOwned(self: *Self) StorageError!bool {
        // Nothing to promote when the bytes live on a device (no host buffer).
        if (self.onDevice()) return false;
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
    /// - the grown axis is re-tiled to the new capacity, keeping growable KV
    ///   caches contiguous for device backends that bind the cache as one buffer
    pub fn growAxisPreserveScalar(self: *Self, axis: usize, new_size: usize) StorageError!void {
        const rank_usize: usize = @as(usize, self.rank);
        if (rank_usize == 0 or axis >= rank_usize) return StorageError.InvalidArgument;
        if (self.dtype.info().is_quantized) return StorageError.InvalidArgument;
        // Host-bytes only. A device tensor grows through
        // `StorageManager.ensureTensorAxisCapacity`, which stays on the device.
        if (self.onDevice()) return StorageError.InvalidArgument;

        const old_size: usize = self.shape[axis];
        if (new_size < old_size) return StorageError.InvalidArgument;
        if (new_size == old_size) return;

        var new_shape_mem: [INLINE_RANK]usize = @splat(0);
        var d: usize = 0;
        while (d < rank_usize) : (d += 1) {
            new_shape_mem[d] = self.shape[d];
        }
        new_shape_mem[axis] = new_size;
        const new_shape: []const usize = new_shape_mem[0..rank_usize];

        var new_tile_shape_mem: [INLINE_RANK]usize = @splat(0);
        d = 0;
        while (d < rank_usize) : (d += 1) {
            new_tile_shape_mem[d] = self.tile_shape[d];
        }
        new_tile_shape_mem[axis] = @max(new_tile_shape_mem[axis], new_size);
        const new_tile_shape: []const usize = new_tile_shape_mem[0..rank_usize];

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

        var old_strides_mem: [INLINE_RANK]usize = @splat(0);
        var new_strides_mem: [INLINE_RANK]usize = @splat(0);
        try computePackedStridesElems(self.shape, old_strides_mem[0..rank_usize]);
        try computePackedStridesElems(new_shape, new_strides_mem[0..rank_usize]);

        var coords_mem: [INLINE_RANK]usize = @splat(0);
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
            new_tile_shape,
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
        return self.acquireTileConstFrom(self.data, ti0, ti1);
    }

    pub fn acquireTileConstFrom(self: *const Self, data: []const u8, ti0: usize, ti1: usize) StorageError!TileViewConst {
        if (self.rank > 2) return StorageError.InvalidArgument;
        const idx: usize = try self.tileIndex(ti0, ti1);
        const off: usize = self.tile_offsets[idx];
        const len: usize = self.tile_lens[idx];
        if (off + len > data.len) return StorageError.InvalidArgument;

        const dims: [2]usize = try self.tileDims(ti0, ti1);
        const di = self.dtype.info();
        const elem_bytes: usize = if (di.is_quantized) 0 else di.block_bytes;
        return TileViewConst.init(data[off .. off + len], self.dtype, self.rank, dims[0..@as(usize, self.rank)], elem_bytes);
    }

    pub fn acquireTileConstLinear(self: *const Self, tile_index: usize) StorageError!TileViewConst {
        return self.acquireTileConstLinearFrom(self.data, tile_index);
    }

    pub fn acquireTileConstLinearFrom(self: *const Self, data: []const u8, tile_index: usize) StorageError!TileViewConst {
        if (tile_index >= self.tile_offsets.len) return StorageError.InvalidArgument;
        if (@as(usize, self.rank) > INLINE_RANK) return StorageError.InvalidArgument;

        const rank: usize = @as(usize, self.rank);
        var tile_coords_mem: [INLINE_RANK]usize = undefined;
        var tile_dims_mem: [INLINE_RANK]usize = undefined;
        const tile_coords = tile_coords_mem[0..rank];
        const tile_dims = tile_dims_mem[0..rank];

        try decodeTileCoords(tile_index, self.tile_counts, self.tile_strides, tile_coords);
        try computeTileDimsND(self.shape, self.tile_shape, tile_coords, tile_dims);

        const off: usize = self.tile_offsets[tile_index];
        const len: usize = self.tile_lens[tile_index];
        if (off + len > data.len) return StorageError.InvalidArgument;

        const di = self.dtype.info();
        const elem_bytes: usize = if (di.is_quantized) 0 else di.block_bytes;
        return TileViewConst.init(data[off .. off + len], self.dtype, self.rank, tile_dims, elem_bytes);
    }

    /// Per-tile logical layout (dtype/rank/shape/strides) WITHOUT touching `data`.
    /// Used to describe a device-resident tile whose host bytes have been freed
    /// (`device.kind == .gpu`). The returned view's `bytes` slice is empty.
    pub fn tileLayoutLinear(self: *const Self, tile_index: usize) StorageError!TileViewConst {
        if (tile_index >= self.tile_offsets.len) return StorageError.InvalidArgument;
        if (@as(usize, self.rank) > INLINE_RANK) return StorageError.InvalidArgument;

        const rank: usize = @as(usize, self.rank);
        var tile_coords_mem: [INLINE_RANK]usize = undefined;
        var tile_dims_mem: [INLINE_RANK]usize = undefined;
        const tile_coords = tile_coords_mem[0..rank];
        const tile_dims = tile_dims_mem[0..rank];

        try decodeTileCoords(tile_index, self.tile_counts, self.tile_strides, tile_coords);
        try computeTileDimsND(self.shape, self.tile_shape, tile_coords, tile_dims);

        const di = self.dtype.info();
        const elem_bytes: usize = if (di.is_quantized) 0 else di.block_bytes;
        return TileViewConst.init(&[_]u8{}, self.dtype, self.rank, tile_dims, elem_bytes);
    }

    pub fn acquireTileMut(self: *Self, ti0: usize, ti1: usize) StorageError!TileViewMut {
        return self.acquireTileMutFrom(self.data, ti0, ti1);
    }

    pub fn acquireTileMutFrom(self: *const Self, data: []u8, ti0: usize, ti1: usize) StorageError!TileViewMut {
        if (self.rank > 2) return StorageError.InvalidArgument;
        const idx: usize = try self.tileIndex(ti0, ti1);
        const off: usize = self.tile_offsets[idx];
        const len: usize = self.tile_lens[idx];
        if (off + len > data.len) return StorageError.InvalidArgument;

        const dims: [2]usize = try self.tileDims(ti0, ti1);
        const di = self.dtype.info();
        const elem_bytes: usize = if (di.is_quantized) 0 else di.block_bytes;
        return TileViewMut.init(data[off .. off + len], self.dtype, self.rank, dims[0..@as(usize, self.rank)], elem_bytes);
    }

    pub fn acquireTileMutLinear(self: *Self, tile_index: usize) StorageError!TileViewMut {
        return self.acquireTileMutLinearFrom(self.data, tile_index);
    }

    pub fn acquireTileMutLinearFrom(self: *const Self, data: []u8, tile_index: usize) StorageError!TileViewMut {
        if (tile_index >= self.tile_offsets.len) return StorageError.InvalidArgument;
        if (@as(usize, self.rank) > INLINE_RANK) return StorageError.InvalidArgument;

        const rank: usize = @as(usize, self.rank);
        var tile_coords_mem: [INLINE_RANK]usize = undefined;
        var tile_dims_mem: [INLINE_RANK]usize = undefined;
        const tile_coords = tile_coords_mem[0..rank];
        const tile_dims = tile_dims_mem[0..rank];

        try decodeTileCoords(tile_index, self.tile_counts, self.tile_strides, tile_coords);
        try computeTileDimsND(self.shape, self.tile_shape, tile_coords, tile_dims);

        const off: usize = self.tile_offsets[tile_index];
        const len: usize = self.tile_lens[tile_index];
        if (off + len > data.len) return StorageError.InvalidArgument;

        const di = self.dtype.info();
        const elem_bytes: usize = if (di.is_quantized) 0 else di.block_bytes;
        return TileViewMut.init(data[off .. off + len], self.dtype, self.rank, tile_dims, elem_bytes);
    }

    /// Whether each tile, in order, is the next contiguous run of the packed layout:
    /// tiles split only the first dim (so a tile is whole rows), and a quantized
    /// tensor's blocks sit in row-major order within them.
    pub fn tilesArePackedRuns(self: *const Self) bool {
        if (self.dtype.info().is_quantized and self.block_order != .row_major) return false;
        return std.mem.eql(usize, self.tile_shape[1..], self.shape[1..]);
    }

    /// Which way a packed range moves: into this tensor's tiles, or out of them.
    const PackedRange = union(enum) {
        into_tiles: []const u8,
        out_of_tiles: []u8,

        fn bytes(self: PackedRange) []const u8 {
            return switch (self) {
                .into_tiles => |b| b,
                .out_of_tiles => |b| b,
            };
        }
    };

    /// Writes a packed row-major scalar tensor into this tiled storage.
    pub fn writeFromPackedScalar(self: *Self, packed_bytes: []const u8) StorageError!void {
        const need = self.requiredBytesPackedScalar();
        if (packed_bytes.len < need) return StorageError.InvalidArgument;
        return self.copyScalarRange(0, .{ .into_tiles = packed_bytes[0..need] });
    }

    /// Reads tiled scalar storage back into packed row-major.
    pub fn readToPackedScalar(self: *const Self, out: []u8) StorageError!void {
        const need = self.requiredBytesPackedScalar();
        if (out.len < need) return StorageError.InvalidArgument;
        return @constCast(self).copyScalarRange(0, .{ .out_of_tiles = out[0..need] });
    }

    /// Read elements `[first_elem, first_elem + out.len / elem_bytes)` of the packed
    /// row-major layout; the ranged counterpart of `readToPackedScalar`.
    pub fn readScalarRange(self: *const Self, first_elem: usize, out: []u8) StorageError!void {
        return @constCast(self).copyScalarRange(first_elem, .{ .out_of_tiles = out });
    }

    /// Writes a packed quant tensor into this tiled storage.
    ///
    /// Packed-quant convention (arbitrary rank):
    /// - Along `self.quant_axis`, every `block_elems` consecutive elements form one
    ///   `block_bytes`-sized block. Block-space shape is therefore
    ///   `shape` with `shape[quant_axis]` replaced by `shape[quant_axis] / block_elems`.
    /// - `packed_bytes` is row-major over that block-space shape, each element being one block.
    pub fn writeFromPackedQuant(self: *Self, packed_bytes: []const u8) StorageError!void {
        const need = try self.packedQuantBytes();
        if (packed_bytes.len < need) return StorageError.InvalidArgument;
        return self.copyQuantBlocks(0, .{ .into_tiles = packed_bytes[0..need] });
    }

    /// Reads tiled quant storage back into the packed quant convention.
    pub fn readToPackedQuant(self: *const Self, out: []u8) StorageError!void {
        const need = try self.packedQuantBytes();
        if (out.len < need) return StorageError.InvalidArgument;
        return @constCast(self).copyQuantBlocks(0, .{ .out_of_tiles = out[0..need] });
    }

    fn packedQuantBytes(self: *const Self) StorageError!usize {
        return utils.requiredBytesForElems(self.dtype, try mulAll(self.shape)) catch StorageError.InvalidArgument;
    }

    /// Write packed quant blocks `[first_block, first_block + len / block_bytes)` —
    /// row-major over block space, as `writeFromPackedQuant` takes the whole — into
    /// their tiles. Lets a producer fill a quantized tensor a chunk at a time.
    pub fn writeQuantBlocks(self: *Self, first_block: usize, packed_bytes: []const u8) StorageError!void {
        return self.copyQuantBlocks(first_block, .{ .into_tiles = packed_bytes });
    }

    /// Read packed quant blocks `[first_block, first_block + out.len / block_bytes)`
    /// out of their tiles; the ranged counterpart of `readToPackedQuant`.
    pub fn readQuantBlocks(self: *const Self, first_block: usize, out: []u8) StorageError!void {
        return @constCast(self).copyQuantBlocks(first_block, .{ .out_of_tiles = out });
    }

    fn copyQuantBlocks(self: *Self, first_block: usize, packed_range: PackedRange) StorageError!void {
        const packed_bytes = packed_range.bytes();
        if (self.onDevice()) return StorageError.InvalidArgument;
        const di = self.dtype.info();
        if (!di.is_quantized or packed_bytes.len % di.block_bytes != 0) return StorageError.InvalidArgument;
        const rank: usize = self.rank;
        const axis: usize = self.quant_axis;
        if (rank == 0 or rank > INLINE_RANK or axis >= rank) return StorageError.InvalidArgument;

        var block_shape: [INLINE_RANK]usize = undefined;
        try computeBlockShapeAxis(self.shape, di.block_elems, axis, block_shape[0..rank]);
        const total: usize = try mulAll(block_shape[0..rank]);
        const count: usize = packed_bytes.len / di.block_bytes;
        if (first_block + count > total) return StorageError.InvalidArgument;

        for (0..count) |i| {
            // Block-space coords of this block, then its tile and place within it.
            var rest: usize = first_block + i;
            var coord: [INLINE_RANK]usize = undefined;
            var d: usize = rank;
            while (d > 0) {
                d -= 1;
                coord[d] = rest % block_shape[d];
                rest /= block_shape[d];
            }
            var tile_index: usize = 0;
            var local_lin: usize = 0;
            for (0..rank) |k| {
                const elem = if (k == axis) coord[k] * di.block_elems else coord[k];
                const tile_c = elem / self.tile_shape[k];
                tile_index += tile_c * self.tile_strides[k];
                const tile_dim = @min(self.tile_shape[k], self.shape[k] - tile_c * self.tile_shape[k]);
                const local_dim = if (k == axis) tile_dim / di.block_elems else tile_dim;
                const local_c = (elem - tile_c * self.tile_shape[k]) / (if (k == axis) di.block_elems else 1);
                local_lin = local_lin * local_dim + local_c;
            }
            const at = self.tile_offsets[tile_index] + local_lin * di.block_bytes;
            const tile_block = self.data[at..][0..di.block_bytes];
            switch (packed_range) {
                .into_tiles => |src| @memcpy(tile_block, src[i * di.block_bytes ..][0..di.block_bytes]),
                .out_of_tiles => |dst| @memcpy(dst[i * di.block_bytes ..][0..di.block_bytes], tile_block),
            }
        }
    }

    /// Copy elements `[first_elem, ..)` of the packed row-major layout between it and
    /// the tiles, a contiguous run at a time: a run carries on across every trailing
    /// dim its tile spans in full, so a single-tile tensor moves in one copy.
    fn copyScalarRange(self: *Self, first_elem: usize, packed_range: PackedRange) StorageError!void {
        const bytes = packed_range.bytes();
        if (self.onDevice()) return StorageError.InvalidArgument; // host bytes freed; migrate with .to(.cpu) first
        const di = self.dtype.info();
        if (di.is_quantized) return StorageError.InvalidArgument;
        const eb: usize = di.block_bytes;
        const rank: usize = self.rank;
        if (rank == 0 or rank > INLINE_RANK or bytes.len % eb != 0) return StorageError.InvalidArgument;
        const count: usize = bytes.len / eb;
        if (first_elem + count > try mulAll(self.shape)) return StorageError.InvalidArgument;

        var done: usize = 0;
        while (done < count) {
            var rest: usize = first_elem + done;
            var local: [INLINE_RANK]usize = undefined;
            var tile_dims: [INLINE_RANK]usize = undefined;
            var tile_index: usize = 0;
            var d: usize = rank;
            while (d > 0) {
                d -= 1;
                const coord = rest % self.shape[d];
                rest /= self.shape[d];
                const tile_c = coord / self.tile_shape[d];
                tile_index += tile_c * self.tile_strides[d];
                tile_dims[d] = @min(self.tile_shape[d], self.shape[d] - tile_c * self.tile_shape[d]);
                local[d] = coord - tile_c * self.tile_shape[d];
            }
            // Dims after `k` are spanned in full, so the tile's elements from here to
            // the end of its block at `k` follow each other in both layouts.
            var k: usize = rank - 1;
            while (k > 0 and tile_dims[k] == self.shape[k]) k -= 1;
            var local_lin: usize = 0;
            var span: usize = 1;
            var from: usize = 0;
            for (0..rank) |j| {
                local_lin = local_lin * tile_dims[j] + local[j];
                if (j >= k) {
                    span *= tile_dims[j];
                    from = from * tile_dims[j] + local[j];
                }
            }
            const run = @min(span - from, count - done);
            const at = self.tile_offsets[tile_index] + local_lin * eb;
            switch (packed_range) {
                .into_tiles => |src| @memcpy(self.data[at..][0 .. run * eb], src[done * eb ..][0 .. run * eb]),
                .out_of_tiles => |dst| @memcpy(dst[done * eb ..][0 .. run * eb], self.data[at..][0 .. run * eb]),
            }
            done += run;
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
