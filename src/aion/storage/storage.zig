const std = @import("std");

const types = @import("../backend/types.zig");
const utils = @import("../backend/utils.zig");

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
    shape_mem: [2]usize,
    strides_mem: [2]isize,

    pub fn init(bytes: []const u8, dtype: DType, rank: u8, dims: [2]usize, elem_bytes: usize) TileViewConst {
        var self: TileViewConst = undefined;
        self.bytes = bytes;
        self.dtype = dtype;
        self.rank = rank;
        self.shape_mem = dims;

        // For quant dtypes, strides are ignored in v0 packedness checks.
        if (dtype.info().is_quantized) {
            self.strides_mem = .{ 0, 0 };
        } else {
            // Packed row-major scalar.
            // rank=1: stride[0] = elem_bytes
            // rank=2: stride[1] = elem_bytes, stride[0] = dims[1] * elem_bytes
            var s0: isize = 0;
            var s1: isize = 0;
            if (rank == 0) {
                s0 = 0;
                s1 = 0;
            } else if (rank == 1) {
                s0 = @intCast(elem_bytes);
                s1 = 0;
            } else {
                s1 = @intCast(elem_bytes);
                s0 = @intCast(dims[1] * elem_bytes);
            }
            self.strides_mem = .{ s0, s1 };
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
    shape_mem: [2]usize,
    strides_mem: [2]isize,

    pub fn init(bytes: []u8, dtype: DType, rank: u8, dims: [2]usize, elem_bytes: usize) TileViewMut {
        var self: TileViewMut = undefined;
        self.bytes = bytes;
        self.dtype = dtype;
        self.rank = rank;
        self.shape_mem = dims;

        if (dtype.info().is_quantized) {
            self.strides_mem = .{ 0, 0 };
        } else {
            var s0: isize = 0;
            var s1: isize = 0;
            if (rank == 0) {
                s0 = 0;
                s1 = 0;
            } else if (rank == 1) {
                s0 = @intCast(elem_bytes);
                s1 = 0;
            } else {
                s1 = @intCast(elem_bytes);
                s0 = @intCast(dims[1] * elem_bytes);
            }
            self.strides_mem = .{ s0, s1 };
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
/// - rank 1 or 2 only
/// - tiles are always packed
/// - quant types are supported as opaque bytes with strict K-tile constraints for matmul compatibility
pub const TiledTensor = struct {
    allocator: std.mem.Allocator,

    dtype: DType,
    rank: u8,
    shape: [2]usize,
    tile_shape: [2]usize,
    tile_counts: [2]usize,

    // Metadata: one allocation, split into offsets and lens.
    meta: []usize,
    tile_offsets: []usize,
    tile_lens: []usize,

    // Tile backing buffer.
    data: []align(64) u8,

    // Alignment between tiles in the backing buffer.
    tile_alignment: usize = 64,

    const Self = @This();

    pub const InitOptions = struct {
        /// Align tile starts in the single backing allocation.
        tile_alignment: usize = 64,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        dtype: DType,
        shape_in: []const usize,
        tile_shape_in: []const usize,
        opts: InitOptions,
    ) StorageError!Self {
        if (shape_in.len == 0 or shape_in.len > 2) return StorageError.InvalidArgument;
        if (tile_shape_in.len != shape_in.len) return StorageError.InvalidArgument;

        const rank: u8 = @intCast(shape_in.len);
        var shape: [2]usize = .{ 1, 1 };
        var tile_shape: [2]usize = .{ 1, 1 };

        shape[0] = shape_in[0];
        tile_shape[0] = tile_shape_in[0];
        if (rank == 2) {
            shape[1] = shape_in[1];
            tile_shape[1] = tile_shape_in[1];
        }

        if (shape[0] == 0) return StorageError.InvalidArgument;
        if (rank == 2 and shape[1] == 0) return StorageError.InvalidArgument;
        if (tile_shape[0] == 0) return StorageError.InvalidArgument;
        if (rank == 2 and tile_shape[1] == 0) return StorageError.InvalidArgument;

        const tile_counts0: usize = (shape[0] + tile_shape[0] - 1) / tile_shape[0];
        const tile_counts1: usize = if (rank == 2) ((shape[1] + tile_shape[1] - 1) / tile_shape[1]) else 1;
        if (tile_counts0 == 0 or tile_counts1 == 0) return StorageError.InvalidArgument;

        const tile_total: usize = std.math.mul(usize, tile_counts0, tile_counts1) catch return StorageError.InvalidArgument;

        // Quant constraints: for now, enforce that every tile's innermost dimension is compatible.
        // This is required so `Backend.matmul` can be called on tiles with k % 32 == 0.
        const di = dtype.info();
        if (di.is_quantized) {
            // Require innermost dimension (shape[0] for rank1, shape[1] for rank2) to be multiple of block_elems.
            const inner_dim: usize = if (rank == 1) shape[0] else shape[0];
            _ = inner_dim;
            // NOTE: For rank2 quant, the kernel contract is about K (% block_elems). This storage is generic
            // and does not know which axis is K; we enforce that the *first axis* is block-aligned, which
            // matches the matmul contract for B tiles when stored as [K,N].
            if (shape[0] % di.block_elems != 0) return StorageError.InvalidArgument;
            if (tile_shape[0] % di.block_elems != 0) return StorageError.InvalidArgument;
            // Also ensure boundary K-tiles remain block-aligned.
            // (This allows last tile smaller than tile_shape[0], as long as it is a multiple of block_elems.)
            const rem: usize = shape[0] % tile_shape[0];
            if (rem != 0 and (rem % di.block_elems != 0)) return StorageError.InvalidArgument;
        }

        // Allocate metadata (offsets + lens) in one block.
        const meta: []usize = allocator.alloc(usize, tile_total * 2) catch return StorageError.OutOfMemory;
        errdefer allocator.free(meta);
        const tile_offsets: []usize = meta[0..tile_total];
        const tile_lens: []usize = meta[tile_total..];

        // Precompute offsets and byte lens.
        var off: usize = 0;
        var idx: usize = 0;

        const elem_bytes: usize = if (di.is_quantized) 0 else di.block_bytes;

        var ti0: usize = 0;
        while (ti0 < tile_counts0) : (ti0 += 1) {
            var ti1: usize = 0;
            while (ti1 < tile_counts1) : (ti1 += 1) {
                const dims: [2]usize = computeTileDims(rank, shape, tile_shape, ti0, ti1);
                const tile_elems: usize = std.math.mul(usize, dims[0], dims[1]) catch {
                    allocator.free(meta);
                    return StorageError.InvalidArgument;
                };

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
                idx += 1;
            }
        }

        // Allocate backing buffer aligned for SIMD-friendly accesses.
        const data: []align(64) u8 = allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(64), off) catch return StorageError.OutOfMemory;
        errdefer allocator.free(data);
        @memset(data, 0);

        // Keep compiler from complaining about elem_bytes in unused branches.
        _ = elem_bytes;

        return .{
            .allocator = allocator,
            .dtype = dtype,
            .rank = rank,
            .shape = shape,
            .tile_shape = tile_shape,
            .tile_counts = .{ tile_counts0, tile_counts1 },
            .meta = meta,
            .tile_offsets = tile_offsets,
            .tile_lens = tile_lens,
            .data = data,
            .tile_alignment = opts.tile_alignment,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.meta.len != 0) {
            self.allocator.free(self.meta);
            self.meta = &[_]usize{};
            self.tile_offsets = &[_]usize{};
            self.tile_lens = &[_]usize{};
        }
        if (self.data.len != 0) {
            self.allocator.free(self.data);
            self.data = &[_]u8{};
        }
        self.rank = 0;
        self.shape = .{ 0, 0 };
        self.tile_shape = .{ 0, 0 };
        self.tile_counts = .{ 0, 0 };
    }

    pub fn tileCountTotal(self: Self) usize {
        return self.tile_offsets.len;
    }

    pub fn tileIndex(self: Self, ti0: usize, ti1: usize) StorageError!usize {
        if (ti0 >= self.tile_counts[0]) return StorageError.InvalidArgument;
        if (ti1 >= self.tile_counts[1]) return StorageError.InvalidArgument;
        return ti0 * self.tile_counts[1] + ti1;
    }

    pub fn tileDims(self: Self, ti0: usize, ti1: usize) StorageError![2]usize {
        if (ti0 >= self.tile_counts[0]) return StorageError.InvalidArgument;
        if (ti1 >= self.tile_counts[1]) return StorageError.InvalidArgument;
        return computeTileDims(self.rank, self.shape, self.tile_shape, ti0, ti1);
    }

    pub fn acquireTileConst(self: *const Self, ti0: usize, ti1: usize) StorageError!TileViewConst {
        const idx: usize = try self.tileIndex(ti0, ti1);
        const off: usize = self.tile_offsets[idx];
        const len: usize = self.tile_lens[idx];
        if (off + len > self.data.len) return StorageError.InvalidArgument;

        const dims: [2]usize = try self.tileDims(ti0, ti1);
        const di = self.dtype.info();
        const elem_bytes: usize = if (di.is_quantized) 0 else di.block_bytes;
        return TileViewConst.init(self.data[off .. off + len], self.dtype, self.rank, dims, elem_bytes);
    }

    pub fn acquireTileMut(self: *Self, ti0: usize, ti1: usize) StorageError!TileViewMut {
        const idx: usize = try self.tileIndex(ti0, ti1);
        const off: usize = self.tile_offsets[idx];
        const len: usize = self.tile_lens[idx];
        if (off + len > self.data.len) return StorageError.InvalidArgument;

        const dims: [2]usize = try self.tileDims(ti0, ti1);
        const di = self.dtype.info();
        const elem_bytes: usize = if (di.is_quantized) 0 else di.block_bytes;
        return TileViewMut.init(self.data[off .. off + len], self.dtype, self.rank, dims, elem_bytes);
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

        if (self.rank == 1) {
            const n: usize = self.shape[0];
            const t: usize = self.tile_shape[0];
            const tile_count: usize = self.tile_counts[0];

            var ti: usize = 0;
            while (ti < tile_count) : (ti += 1) {
                const start: usize = ti * t;
                const len_elems: usize = @min(t, n - start);
                const idx: usize = try self.tileIndex(ti, 0);
                const off: usize = self.tile_offsets[idx];
                const out_len: usize = self.tile_lens[idx];

                const want_bytes: usize = len_elems * elem_bytes;
                if (want_bytes != out_len) return StorageError.InvalidArgument;

                const src_off: usize = start * elem_bytes;
                @memcpy(self.data[off .. off + want_bytes], packed_bytes[src_off .. src_off + want_bytes]);
            }
            return;
        }

        // rank == 2
        const rows: usize = self.shape[0];
        const cols: usize = self.shape[1];
        const tm: usize = self.tile_shape[0];
        const tn: usize = self.tile_shape[1];

        var ti0: usize = 0;
        while (ti0 < self.tile_counts[0]) : (ti0 += 1) {
            var ti1: usize = 0;
            while (ti1 < self.tile_counts[1]) : (ti1 += 1) {
                const dims: [2]usize = computeTileDims(2, self.shape, self.tile_shape, ti0, ti1);
                const m_tile: usize = dims[0];
                const n_tile: usize = dims[1];

                const idx: usize = try self.tileIndex(ti0, ti1);
                const off: usize = self.tile_offsets[idx];
                const out_len: usize = self.tile_lens[idx];

                const tile_bytes: usize = m_tile * n_tile * elem_bytes;
                if (tile_bytes != out_len) return StorageError.InvalidArgument;

                const row0: usize = ti0 * tm;
                const col0: usize = ti1 * tn;

                var r: usize = 0;
                while (r < m_tile) : (r += 1) {
                    const src_row: usize = row0 + r;
                    if (src_row >= rows) return StorageError.InvalidArgument;

                    const src_off: usize = (src_row * cols + col0) * elem_bytes;
                    const dst_off: usize = (r * n_tile) * elem_bytes;
                    const row_bytes: usize = n_tile * elem_bytes;

                    @memcpy(
                        self.data[(off + dst_off)..(off + dst_off + row_bytes)],
                        packed_bytes[src_off .. src_off + row_bytes],
                    );
                }
            }
        }
    }

    /// Reads tiled scalar storage back into packed row-major.
    pub fn readToPackedScalar(self: *const Self, out: []u8) StorageError!void {
        if (self.dtype.info().is_quantized) return StorageError.InvalidArgument;

        const need_total: usize = self.requiredBytesPackedScalar();
        if (out.len < need_total) return StorageError.InvalidArgument;

        const elem_bytes: usize = self.dtype.info().block_bytes;

        if (self.rank == 1) {
            const n: usize = self.shape[0];
            const t: usize = self.tile_shape[0];
            const tile_count: usize = self.tile_counts[0];

            var ti: usize = 0;
            while (ti < tile_count) : (ti += 1) {
                const start: usize = ti * t;
                const len_elems: usize = @min(t, n - start);
                const idx: usize = try self.tileIndex(ti, 0);
                const off: usize = self.tile_offsets[idx];
                const in_len: usize = self.tile_lens[idx];

                const want_bytes: usize = len_elems * elem_bytes;
                if (want_bytes != in_len) return StorageError.InvalidArgument;

                const dst_off: usize = start * elem_bytes;
                @memcpy(out[dst_off .. dst_off + want_bytes], self.data[off .. off + want_bytes]);
            }
            return;
        }

        // rank == 2
        const rows: usize = self.shape[0];
        const cols: usize = self.shape[1];
        const tm: usize = self.tile_shape[0];
        const tn: usize = self.tile_shape[1];

        var ti0: usize = 0;
        while (ti0 < self.tile_counts[0]) : (ti0 += 1) {
            var ti1: usize = 0;
            while (ti1 < self.tile_counts[1]) : (ti1 += 1) {
                const dims: [2]usize = computeTileDims(2, self.shape, self.tile_shape, ti0, ti1);
                const m_tile: usize = dims[0];
                const n_tile: usize = dims[1];

                const idx: usize = try self.tileIndex(ti0, ti1);
                const off: usize = self.tile_offsets[idx];
                const in_len: usize = self.tile_lens[idx];

                const tile_bytes: usize = m_tile * n_tile * elem_bytes;
                if (tile_bytes != in_len) return StorageError.InvalidArgument;

                const row0: usize = ti0 * tm;
                const col0: usize = ti1 * tn;

                var r: usize = 0;
                while (r < m_tile) : (r += 1) {
                    const dst_row: usize = row0 + r;
                    if (dst_row >= rows) return StorageError.InvalidArgument;

                    const dst_off: usize = (dst_row * cols + col0) * elem_bytes;
                    const src_off: usize = (r * n_tile) * elem_bytes;
                    const row_bytes: usize = n_tile * elem_bytes;

                    @memcpy(
                        out[dst_off .. dst_off + row_bytes],
                        self.data[(off + src_off)..(off + src_off + row_bytes)],
                    );
                }
            }
        }
    }

    /// Writes a packed quant tensor into this tiled storage.
    ///
    /// v0 packed quant convention:
    /// - rank-1: a flat array of blocks.
    /// - rank-2: a row-major array of blocks with shape [k_blocks, n], where k_blocks = k / block_elems.
    pub fn writeFromPackedQuant(self: *Self, packed_bytes: []const u8) StorageError!void {
        const di = self.dtype.info();
        if (!di.is_quantized) return StorageError.InvalidArgument;
        if (self.rank == 0 or self.rank > 2) return StorageError.InvalidArgument;

        const total_elems: usize = if (self.rank == 1) self.shape[0] else (std.math.mul(usize, self.shape[0], self.shape[1]) catch return StorageError.InvalidArgument);
        const need_total: usize = utils.requiredBytesForElems(self.dtype, total_elems) catch return StorageError.InvalidArgument;
        if (packed_bytes.len < need_total) return StorageError.InvalidArgument;

        if (self.rank == 1) {
            // Each tile is a packed array of blocks; copy blocks with no reinterpretation.
            const n: usize = self.shape[0];
            const t: usize = self.tile_shape[0];
            const tile_count: usize = self.tile_counts[0];

            var ti: usize = 0;
            while (ti < tile_count) : (ti += 1) {
                const start: usize = ti * t;
                const len_elems: usize = @min(t, n - start);
                if (len_elems % di.block_elems != 0) return StorageError.InvalidArgument;

                const tile_bytes: usize = utils.requiredBytesForElems(self.dtype, len_elems) catch return StorageError.InvalidArgument;

                const idx: usize = try self.tileIndex(ti, 0);
                const off: usize = self.tile_offsets[idx];
                const out_len: usize = self.tile_lens[idx];
                if (tile_bytes != out_len) return StorageError.InvalidArgument;

                const src_bytes_off: usize = utils.requiredBytesForElems(self.dtype, start) catch return StorageError.InvalidArgument;
                @memcpy(self.data[off .. off + tile_bytes], packed_bytes[src_bytes_off .. src_bytes_off + tile_bytes]);
            }
            return;
        }

        // rank == 2, block-matrix copy with strides in the source.
        const k: usize = self.shape[0];
        const n: usize = self.shape[1];
        if (k % di.block_elems != 0) return StorageError.InvalidArgument;
        const k_blocks_total: usize = k / di.block_elems;

        const tk: usize = self.tile_shape[0];
        const tn: usize = self.tile_shape[1];

        var ti0: usize = 0;
        while (ti0 < self.tile_counts[0]) : (ti0 += 1) {
            var ti1: usize = 0;
            while (ti1 < self.tile_counts[1]) : (ti1 += 1) {
                const dims: [2]usize = computeTileDims(2, self.shape, self.tile_shape, ti0, ti1);
                const k_tile: usize = dims[0];
                const n_tile: usize = dims[1];
                if (k_tile % di.block_elems != 0) return StorageError.InvalidArgument;

                const kb_start: usize = (ti0 * tk) / di.block_elems;
                const kb_count: usize = k_tile / di.block_elems;

                const idx: usize = try self.tileIndex(ti0, ti1);
                const off: usize = self.tile_offsets[idx];
                const out_len: usize = self.tile_lens[idx];

                const tile_bytes: usize = kb_count * n_tile * di.block_bytes;
                if (tile_bytes != out_len) return StorageError.InvalidArgument;

                const col0: usize = ti1 * tn;

                var jj: usize = 0;
                while (jj < n_tile) : (jj += 1) {
                    const j: usize = col0 + jj;
                    if (j >= n) return StorageError.InvalidArgument;

                    var kb: usize = 0;
                    while (kb < kb_count) : (kb += 1) {
                        const src_kb: usize = kb_start + kb;
                        if (src_kb >= k_blocks_total) return StorageError.InvalidArgument;

                        const src_off: usize = (src_kb * n + j) * di.block_bytes;
                        const dst_off: usize = (kb * n_tile + jj) * di.block_bytes;

                        @memcpy(
                            self.data[(off + dst_off)..(off + dst_off + di.block_bytes)],
                            packed_bytes[src_off .. src_off + di.block_bytes],
                        );
                    }
                }
            }
        }
    }

    /// Reads tiled quant storage back into the packed quant convention.
    pub fn readToPackedQuant(self: *const Self, out: []u8) StorageError!void {
        const di = self.dtype.info();
        if (!di.is_quantized) return StorageError.InvalidArgument;
        if (self.rank == 0 or self.rank > 2) return StorageError.InvalidArgument;

        const total_elems: usize = if (self.rank == 1) self.shape[0] else (std.math.mul(usize, self.shape[0], self.shape[1]) catch return StorageError.InvalidArgument);
        const need_total: usize = utils.requiredBytesForElems(self.dtype, total_elems) catch return StorageError.InvalidArgument;
        if (out.len < need_total) return StorageError.InvalidArgument;

        if (self.rank == 1) {
            const n: usize = self.shape[0];
            const t: usize = self.tile_shape[0];
            const tile_count: usize = self.tile_counts[0];

            var ti: usize = 0;
            while (ti < tile_count) : (ti += 1) {
                const start: usize = ti * t;
                const len_elems: usize = @min(t, n - start);
                if (len_elems % di.block_elems != 0) return StorageError.InvalidArgument;

                const tile_bytes: usize = utils.requiredBytesForElems(self.dtype, len_elems) catch return StorageError.InvalidArgument;

                const idx: usize = try self.tileIndex(ti, 0);
                const off: usize = self.tile_offsets[idx];
                const in_len: usize = self.tile_lens[idx];
                if (tile_bytes != in_len) return StorageError.InvalidArgument;

                const dst_bytes_off: usize = utils.requiredBytesForElems(self.dtype, start) catch return StorageError.InvalidArgument;
                @memcpy(out[dst_bytes_off .. dst_bytes_off + tile_bytes], self.data[off .. off + tile_bytes]);
            }
            return;
        }

        const k: usize = self.shape[0];
        const n: usize = self.shape[1];
        if (k % di.block_elems != 0) return StorageError.InvalidArgument;
        const k_blocks_total: usize = k / di.block_elems;

        const tk: usize = self.tile_shape[0];
        const tn: usize = self.tile_shape[1];

        var ti0: usize = 0;
        while (ti0 < self.tile_counts[0]) : (ti0 += 1) {
            var ti1: usize = 0;
            while (ti1 < self.tile_counts[1]) : (ti1 += 1) {
                const dims: [2]usize = computeTileDims(2, self.shape, self.tile_shape, ti0, ti1);
                const k_tile: usize = dims[0];
                const n_tile: usize = dims[1];
                if (k_tile % di.block_elems != 0) return StorageError.InvalidArgument;

                const kb_start: usize = (ti0 * tk) / di.block_elems;
                const kb_count: usize = k_tile / di.block_elems;

                const idx: usize = try self.tileIndex(ti0, ti1);
                const off: usize = self.tile_offsets[idx];
                const in_len: usize = self.tile_lens[idx];

                const tile_bytes: usize = kb_count * n_tile * di.block_bytes;
                if (tile_bytes != in_len) return StorageError.InvalidArgument;

                const col0: usize = ti1 * tn;

                var jj: usize = 0;
                while (jj < n_tile) : (jj += 1) {
                    const j: usize = col0 + jj;
                    if (j >= n) return StorageError.InvalidArgument;

                    var kb: usize = 0;
                    while (kb < kb_count) : (kb += 1) {
                        const dst_kb: usize = kb_start + kb;
                        if (dst_kb >= k_blocks_total) return StorageError.InvalidArgument;

                        const dst_off: usize = (dst_kb * n + j) * di.block_bytes;
                        const src_off: usize = (kb * n_tile + jj) * di.block_bytes;

                        @memcpy(
                            out[dst_off .. dst_off + di.block_bytes],
                            self.data[(off + src_off)..(off + src_off + di.block_bytes)],
                        );
                    }
                }
            }
        }
    }

    fn requiredBytesPackedScalar(self: Self) usize {
        const di = self.dtype.info();
        const elem_bytes: usize = di.block_bytes;
        if (self.rank == 1) return self.shape[0] * elem_bytes;
        return self.shape[0] * self.shape[1] * elem_bytes;
    }
};

fn computeTileDims(rank: u8, shape: [2]usize, tile_shape: [2]usize, ti0: usize, ti1: usize) [2]usize {
    const d0_start: usize = ti0 * tile_shape[0];
    const d0_len: usize = if (d0_start >= shape[0]) 0 else @min(tile_shape[0], shape[0] - d0_start);

    if (rank == 1) {
        return .{ d0_len, 1 };
    }

    const d1_start: usize = ti1 * tile_shape[1];
    const d1_len: usize = if (d1_start >= shape[1]) 0 else @min(tile_shape[1], shape[1] - d1_start);
    return .{ d0_len, d1_len };
}
