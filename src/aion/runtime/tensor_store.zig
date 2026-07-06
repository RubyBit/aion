// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const types = @import("../backend/types.zig");

pub const TensorId = u32;

pub const StoreError = error{ InvalidArgument, OutOfMemory };

pub const KVCachePolicyKind = enum(u8) {
    none = 0,
    growable = 1,
    ring = 2,
};

pub const KVCachePolicyInfo = struct {
    kind: KVCachePolicyKind = .none,
    ring_window_tokens: usize = 0,
};

pub const TensorMeta = struct {
    dtype: types.DType,
    rank: u8,
    shape: []const usize,
    tile_shape: []const usize,
    tile_counts: []const usize,
    tile_strides: []const usize,
};

pub const INLINE_RANK: usize = 8;

/// A tile whose bytes live in device-owned buffers (a tensor migrated via
/// `moveTensor` under move semantics). `handle` is a `DeviceMemory.DeviceHandle`
/// (kept as `u64` here so this generic seam names no GPU API). `shape`/`strides`
/// describe the logical tile exactly as a host `TileRef` would.
pub const DeviceTileRef = struct {
    handle: u64,
    len: usize,
    dtype: types.DType,
    rank: u8,
    shape_mem: [INLINE_RANK]usize,
    strides_mem: [INLINE_RANK]isize,
};

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

    pub fn bufferView(self: *const TileRefMut) types.BufferViewMut {
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
/// Conceptually, acquiring a tile returns a temporary lease on storage and the
/// returned token must be released when the caller is done with the tile.
///
/// v0 keeps this lightweight:
/// - tokens are always `0`
/// - release hooks are no-ops
/// - tile lifetime is guaranteed by the owning `StorageManager`
///
/// Future out-of-core storage can turn tokens into real pin/lease handles for
/// cache residency, staging buffers, and deterministic I/O failure boundaries.
///
/// Device seam: this vtable is THE boundary a device (GPU) backend intercepts.
/// Execution touches storage only through here (no kernel reaches into the
/// `StorageManager` directly), so a device-residency store can transparently
/// stage tiles host<->device behind these methods. Two contracts the residency
/// layer relies on, and that callers MUST uphold:
///   - Lease discipline: every `acquireTile*` is paired with exactly one
///     matching `release*` (no leak, no double-release). Validated by
///     `runtime/checked_store.zig`'s `CheckedTensorStore`.
///   - Read/write intent: `acquireTileConst` signals read-only access;
///     `acquireTileMut` signals a write. A device store uses this to decide
///     when a host->device upload is needed and when a tile must be marked
///     dirty for device->host flush. Do not acquire `Mut` for read-only use.
pub const TensorStore = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        meta: *const fn (ctx: *anyopaque, id: TensorId) StoreError!TensorMeta,
        acquireTileConst: *const fn (ctx: *anyopaque, id: TensorId, ti0: usize, ti1: usize) StoreError!TileRefConst,
        acquireTileMut: *const fn (ctx: *anyopaque, id: TensorId, ti0: usize, ti1: usize) StoreError!TileRefMut,
        acquireTileConstLinear: *const fn (ctx: *anyopaque, id: TensorId, tile_index: usize) StoreError!TileRefConst,
        acquireTileMutLinear: *const fn (ctx: *anyopaque, id: TensorId, tile_index: usize) StoreError!TileRefMut,
        /// Release a previously acquired read-only tile lease.
        releaseConst: *const fn (ctx: *anyopaque, token: usize) void,
        /// Release a previously acquired mutable tile lease.
        releaseMut: *const fn (ctx: *anyopaque, token: usize) void,

        /// Optional runtime hint for cache policy bound to a tensor id.
        ///
        /// If null, callers must assume `.none` semantics.
        kvCachePolicyInfo: ?*const fn (ctx: *anyopaque, id: TensorId) KVCachePolicyInfo = null,

        /// Optional logical->physical time-index mapper for KV cache tensors.
        ///
        /// If null, mapping defaults to identity with strict bounds checks.
        mapKVCacheTime: ?*const fn (ctx: *anyopaque, id: TensorId, logical_t: usize, physical_capacity_tokens: usize) StoreError!usize = null,

        /// Prefetch hint. Non-blocking.
        ///
        /// In v0 this is only a CPU cache hint. Future implementations may use
        /// it to initiate staged tile loads or cache warming.
        prefetch: ?*const fn (ctx: *anyopaque, id: TensorId, ti0: usize, ti1: usize) void = null,
        prefetchLinear: ?*const fn (ctx: *anyopaque, id: TensorId, tile_index: usize) void = null,

        /// Optional no-copy exchange of two tensor backing buffers.
        ///
        /// This is intended for loop-carried SSA-style state where the loop body
        /// writes `next_state` into a scratch tensor, then the runtime makes that
        /// scratch storage become the carried state for the next iteration.
        /// Implementations must reject tensors with incompatible dtype/layout.
        swapTensors: ?*const fn (ctx: *anyopaque, a: TensorId, b: TensorId) StoreError!void = null,

        /// Optional: if the tensor's bytes have been migrated to device-owned
        /// buffers (move semantics), return the device handle + per-tile layout for
        /// `tile_index`. Returns null when the tensor is host-resident (the caller —
        /// a residency layer — should stage it via `acquireTile*` as usual). If null
        /// (unset), all tensors are treated as host-resident.
        deviceTile: ?*const fn (ctx: *anyopaque, id: TensorId, tile_index: usize) StoreError!?DeviceTileRef = null,
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

    pub fn kvCachePolicyInfo(self: TensorStore, id: TensorId) KVCachePolicyInfo {
        if (self.vtable.kvCachePolicyInfo) |cb| return cb(self.ctx, id);
        return .{};
    }

    pub fn mapKVCacheTime(self: TensorStore, id: TensorId, logical_t: usize, physical_capacity_tokens: usize) StoreError!usize {
        if (physical_capacity_tokens == 0) return StoreError.InvalidArgument;
        if (self.vtable.mapKVCacheTime) |cb| return cb(self.ctx, id, logical_t, physical_capacity_tokens);
        if (logical_t >= physical_capacity_tokens) return StoreError.InvalidArgument;
        return logical_t;
    }

    pub fn prefetch(self: TensorStore, id: TensorId, ti0: usize, ti1: usize) void {
        if (self.vtable.prefetch) |p| p(self.ctx, id, ti0, ti1);
    }

    pub fn prefetchLinear(self: TensorStore, id: TensorId, tile_index: usize) void {
        if (self.vtable.prefetchLinear) |p| p(self.ctx, id, tile_index);
    }

    pub fn swapTensors(self: TensorStore, a: TensorId, b: TensorId) StoreError!void {
        if (self.vtable.swapTensors) |swap| return swap(self.ctx, a, b);
        return StoreError.InvalidArgument;
    }

    /// Null when the store has no device-residency notion or the tensor is
    /// host-resident; otherwise the device handle + layout for `tile_index`.
    pub fn deviceTile(self: TensorStore, id: TensorId, tile_index: usize) StoreError!?DeviceTileRef {
        if (self.vtable.deviceTile) |cb| return cb(self.ctx, id, tile_index);
        return null;
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
