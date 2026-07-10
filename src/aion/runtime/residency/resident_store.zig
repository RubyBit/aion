// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! `ResidentTensorStore` — a `TensorStore` decorator that adds device-memory
//! residency on top of any host store, using a pluggable `DeviceMemory`.
//!
//! It is the structural template for a GPU backend's store: host kernels keep
//! using the pass-through `acquireTile*` methods, device kernels use the
//! `acquireTileDevice*Linear` methods, and this decorator keeps the two copies
//! coherent — uploading (H2D) when a tile is read on-device after a host write,
//! and flushing (D2H) when a tile is read on-host after a device write. Dirty
//! tracking elides redundant transfers.
//!
//! Residency state lives HERE, not in `TiledTensor` — the host hot path is
//! untouched. With `MockDeviceMemory` the whole thing is exercised on CPU.

const std = @import("std");
const tensor_store = @import("../tensor_store.zig");
const device_memory = @import("device_memory.zig");

const TensorStore = tensor_store.TensorStore;
const TensorId = tensor_store.TensorId;
const StoreError = tensor_store.StoreError;
const TileRefConst = tensor_store.TileRefConst;
const TileRefMut = tensor_store.TileRefMut;
const TensorMeta = tensor_store.TensorMeta;
const KVCachePolicyInfo = tensor_store.KVCachePolicyInfo;
const DeviceMemory = device_memory.DeviceMemory;
const DeviceHandle = device_memory.DeviceHandle;

/// A tile resolved to device memory. The handle+offset address device memory;
/// `shape`/`strides` describe the logical tile exactly as for a host tile. This
/// lives here (not on `TensorStore`) because device residency is owned by this
/// concrete decorator / a GPU backend, not by the universal host-store
/// interface — host kernels never see device memory.
pub const TileRefDevice = struct {
    handle: DeviceHandle,
    offset: usize,
    len: usize,
    dtype: @import("../../backend/types.zig").DType,
    rank: u8,
    shape_mem: [tensor_store.INLINE_RANK]usize,
    strides_mem: [tensor_store.INLINE_RANK]isize,
    token: usize = 0,
};

/// High bit marks a token issued by THIS decorator (a device lease) vs a token
/// that belongs to the inner host store (which must be released to the inner).
const DEVICE_TOKEN_FLAG: usize = @as(usize, 1) << (@bitSizeOf(usize) - 1);

fn deviceErrToStore(e: device_memory.DeviceError) StoreError {
    return switch (e) {
        error.OutOfDeviceMemory => StoreError.OutOfMemory,
        error.InvalidArgument, error.Unsupported => StoreError.InvalidArgument,
    };
}

const TileState = struct {
    handle: DeviceHandle = 0, // 0 = not yet allocated on device
    len: usize = 0,
    resident: bool = false, // device buffer holds a valid copy
    host_dirty: bool = false, // host written since last H2D (device stale)
    dev_dirty: bool = false, // device written since last D2H (host stale)
    /// The owning tensor's `host_seq` at the last H2D upload. When the current
    /// `host_seq` differs, the host bytes changed out-of-band (via the C ABI or a
    /// model's per-run seeding / io-alias write-back) and we must re-upload.
    uploaded_seq: u64 = 0,
};

const Lease = struct {
    key: u64,
    is_mut: bool,
};

pub const ResidentTensorStore = struct {
    allocator: std.mem.Allocator,
    inner: TensorStore,
    dev: DeviceMemory,

    states: std.AutoHashMap(u64, TileState),
    leases: std.AutoHashMap(usize, Lease),
    next_token: usize = 1,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, inner: TensorStore, dev: DeviceMemory) Self {
        return .{
            .allocator = allocator,
            .inner = inner,
            .dev = dev,
            .states = std.AutoHashMap(u64, TileState).init(allocator),
            .leases = std.AutoHashMap(usize, Lease).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        // Free any device buffers we allocated.
        var it = self.states.iterator();
        while (it.next()) |e| {
            if (e.value_ptr.handle != 0) self.dev.free(e.value_ptr.handle);
        }
        self.states.deinit();
        self.leases.deinit();
        self.* = undefined;
    }

    pub fn tensorStore(self: *Self) TensorStore {
        return .{ .ctx = @ptrCast(self), .vtable = &vtable };
    }

    fn key(id: TensorId, tile_index: usize) u64 {
        return (@as(u64, id) << 32) | @as(u64, @intCast(tile_index));
    }

    fn statePtr(self: *Self, k: u64) StoreError!*TileState {
        const gop = self.states.getOrPut(k) catch return StoreError.OutOfMemory;
        if (!gop.found_existing) gop.value_ptr.* = .{};
        return gop.value_ptr;
    }

    fn issueToken(self: *Self, k: u64, is_mut: bool) StoreError!usize {
        const token = (self.next_token & ~DEVICE_TOKEN_FLAG) | DEVICE_TOKEN_FLAG;
        self.next_token += 1;
        self.leases.put(token, .{ .key = k, .is_mut = is_mut }) catch return StoreError.OutOfMemory;
        return token;
    }

    /// Flush device->host for a tile if the device copy is newer.
    fn flushIfDevDirty(self: *Self, id: TensorId, tile_index: usize) StoreError!void {
        const k = key(id, tile_index);
        const st = self.states.getPtr(k) orelse return;
        if (!st.dev_dirty) return;
        const mut = try self.inner.acquireTileMutLinear(id, tile_index);
        defer self.inner.releaseMut(mut.token);
        self.dev.copyD2H(mut.bytes, st.handle, 0) catch |e| return deviceErrToStore(e);
        st.dev_dirty = false;
    }

    fn deviceAcquire(self: *Self, id: TensorId, tile_index: usize, is_mut: bool) StoreError!TileRefDevice {
        // Move semantics: if the tensor's bytes already live in device-owned buffers
        // (migrated via `moveTensor`/`Tensor.to`), hand that buffer straight through —
        // no staging, no dirty tracking, no host copy. We still issue a device token so
        // lease discipline holds; release is a no-op lookup (no `TileState` exists).
        if (try self.inner.deviceTile(id, tile_index)) |dt| {
            return .{
                .handle = dt.handle,
                .offset = 0,
                .len = dt.len,
                .dtype = dt.dtype,
                .rank = dt.rank,
                .shape_mem = dt.shape_mem,
                .strides_mem = dt.strides_mem,
                .token = try self.issueToken(key(id, tile_index), is_mut),
            };
        }

        // Unified memory: host and device share physical memory. Bind the host
        // allocation directly — no device buffer, no copy, no dirty tracking.
        // Re-resolve the current host slice each call so this stays correct
        // across swapTensors (the host buffer pointer may have moved).
        if (self.dev.model() == .unified) {
            const ht = try self.inner.acquireTileMutLinear(id, tile_index);
            defer self.inner.releaseMut(ht.token);
            const handle = self.dev.importHost(ht.bytes) catch |e| return deviceErrToStore(e);
            return .{
                .handle = handle,
                .offset = 0,
                .len = ht.bytes.len,
                .dtype = ht.dtype,
                .rank = ht.rank,
                .shape_mem = ht.shape_mem,
                .strides_mem = ht.strides_mem,
                .token = try self.issueToken(key(id, tile_index), is_mut),
            };
        }

        // Discrete memory: stage into a device-owned buffer with dirty tracking.
        // Source-of-truth host tile (also gives us the layout metadata).
        const host_seq = (try self.inner.meta(id)).host_seq;
        const host = try self.inner.acquireTileConstLinear(id, tile_index);
        defer self.inner.releaseConst(host.token);

        const k = key(id, tile_index);
        const st = try self.statePtr(k);
        if (st.handle == 0) {
            st.handle = self.dev.alloc(host.bytes.len, 64) catch |e| return deviceErrToStore(e);
            st.len = host.bytes.len;
        }
        // Upload if the device copy is missing, stale via a tracked host lease
        // (host_dirty), or stale via an out-of-band host write (host_seq changed
        // since our last upload — the C ABI / model per-run seeding / io-alias
        // write-back mutate the host store directly, bypassing our vtable). After
        // H2D the two copies match, so neither side is dirty.
        if (!st.resident or st.host_dirty or st.uploaded_seq != host_seq) {
            self.dev.copyH2D(st.handle, 0, host.bytes) catch |e| return deviceErrToStore(e);
            st.resident = true;
            st.host_dirty = false;
            st.dev_dirty = false;
            st.uploaded_seq = host_seq;
        }

        return .{
            .handle = st.handle,
            .offset = 0,
            .len = st.len,
            .dtype = host.dtype,
            .rank = host.rank,
            .shape_mem = host.shape_mem,
            .strides_mem = host.strides_mem,
            .token = try self.issueToken(k, is_mut),
        };
    }

    // ---- device-residency acquire (concrete methods, not on the vtable) ----
    //
    // A GPU backend holds a `*ResidentTensorStore` and calls these directly to
    // get device-resident tiles, while passing `tensorStore()` (the host vtable
    // view) wherever a generic host step needs storage. Device leases are
    // released through the vtable `releaseConst`/`releaseMut` (token-flag
    // dispatch), so host steps and the loop/swap machinery release uniformly.

    pub fn acquireTileDeviceConstLinear(self: *Self, id: TensorId, tile_index: usize) StoreError!TileRefDevice {
        return self.deviceAcquire(id, tile_index, false);
    }

    pub fn acquireTileDeviceMutLinear(self: *Self, id: TensorId, tile_index: usize) StoreError!TileRefDevice {
        return self.deviceAcquire(id, tile_index, true);
    }

    /// True if tile `(id, tile_index)` has a device write not yet flushed to host
    /// (the device copy is newer than host). Lets a caller that wants to read this
    /// tile on the host decide whether a device sync is actually required: a
    /// host-resident (clean) tile can be read directly with no GPU stall.
    ///
    /// Note dev_dirty is set at device-mut-lease *release* time — i.e. as soon as
    /// a dispatch that writes the tile is recorded — so a `true` result can refer
    /// to a dispatch still pending in the current (unsubmitted) frame. Callers
    /// must therefore submit any pending GPU work before reading such a tile.
    pub fn tileDeviceDirty(self: *Self, id: TensorId, tile_index: usize) bool {
        const st = self.states.get(key(id, tile_index)) orelse return false;
        return st.dev_dirty;
    }

    // ---- vtable: host pass-through (with coherency) ----

    fn meta(ctx: *anyopaque, id: TensorId) StoreError!TensorMeta {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return self.inner.meta(id);
    }

    fn acquireTileConstLinear(ctx: *anyopaque, id: TensorId, tile_index: usize) StoreError!TileRefConst {
        const self: *Self = @ptrCast(@alignCast(ctx));
        try self.flushIfDevDirty(id, tile_index);
        return self.inner.acquireTileConstLinear(id, tile_index);
    }

    fn acquireTileMutLinear(ctx: *anyopaque, id: TensorId, tile_index: usize) StoreError!TileRefMut {
        const self: *Self = @ptrCast(@alignCast(ctx));
        try self.flushIfDevDirty(id, tile_index);
        // Host is about to be written: any existing device copy goes stale.
        if (self.states.getPtr(key(id, tile_index))) |st| st.host_dirty = true;
        return self.inner.acquireTileMutLinear(id, tile_index);
    }

    fn coordsToLinear(self: *Self, id: TensorId, ti0: usize, ti1: usize) StoreError!usize {
        const m = try self.inner.meta(id);
        // 2D host fast-path tiles: rank-2 coords -> linear index.
        if (m.tile_counts.len == 2) {
            var coords = [_]usize{ ti0, ti1 };
            return tensor_store.encodeTileIndex(m, coords[0..2]);
        }
        // Fall back: treat ti0 as already-linear when the store isn't rank-2.
        return ti0;
    }

    fn acquireTileConst(ctx: *anyopaque, id: TensorId, ti0: usize, ti1: usize) StoreError!TileRefConst {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const lin = try self.coordsToLinear(id, ti0, ti1);
        try self.flushIfDevDirty(id, lin);
        return self.inner.acquireTileConst(id, ti0, ti1);
    }

    fn acquireTileMut(ctx: *anyopaque, id: TensorId, ti0: usize, ti1: usize) StoreError!TileRefMut {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const lin = try self.coordsToLinear(id, ti0, ti1);
        try self.flushIfDevDirty(id, lin);
        if (self.states.getPtr(key(id, lin))) |st| st.host_dirty = true;
        return self.inner.acquireTileMut(id, ti0, ti1);
    }

    fn releaseConst(ctx: *anyopaque, token: usize) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        if (token & DEVICE_TOKEN_FLAG != 0) {
            _ = self.leases.remove(token); // const device lease: nothing to flush
            return;
        }
        self.inner.releaseConst(token);
    }

    fn releaseMut(ctx: *anyopaque, token: usize) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        if (token & DEVICE_TOKEN_FLAG != 0) {
            if (self.leases.fetchRemove(token)) |kv| {
                if (self.states.getPtr(kv.value.key)) |st| {
                    // Device wrote this tile: device copy is now newer than host.
                    st.dev_dirty = true;
                    st.resident = true;
                    st.host_dirty = false;
                }
            }
            return;
        }
        self.inner.releaseMut(token);
    }

    fn kvCachePolicyInfo(ctx: *anyopaque, id: TensorId) KVCachePolicyInfo {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return self.inner.kvCachePolicyInfo(id);
    }

    fn mapKVCacheTime(ctx: *anyopaque, id: TensorId, logical_t: usize, physical_capacity_tokens: usize) StoreError!usize {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const info = self.inner.kvCachePolicyInfo(id);
        if (info.kind != .growable) return self.inner.mapKVCacheTime(id, logical_t, physical_capacity_tokens);

        const before = try self.inner.meta(id);
        const before_sig = GeometrySnapshot.init(before);
        try self.flushAllDevDirty(id);
        const mapped = try self.inner.mapKVCacheTime(id, logical_t, physical_capacity_tokens);
        const after = try self.inner.meta(id);
        if (!before_sig.eql(after)) try self.dropDeviceStatesForTensor(id);
        return mapped;
    }

    fn prefetch(ctx: *anyopaque, id: TensorId, ti0: usize, ti1: usize) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.inner.prefetch(id, ti0, ti1);
    }

    fn prefetchLinear(ctx: *anyopaque, id: TensorId, tile_index: usize) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.inner.prefetchLinear(id, tile_index);
    }

    fn totalTiles(self: *Self, id: TensorId) StoreError!usize {
        const m = try self.inner.meta(id);
        var total: usize = 1;
        for (m.tile_counts) |c| total *= c;
        return total;
    }

    const GeometrySnapshot = struct {
        dtype: @import("../../backend/types.zig").DType,
        rank: u8,
        shape: [tensor_store.INLINE_RANK]usize = @splat(0),
        tile_shape: [tensor_store.INLINE_RANK]usize = @splat(0),
        tile_counts: [tensor_store.INLINE_RANK]usize = @splat(0),
        tile_strides: [tensor_store.INLINE_RANK]usize = @splat(0),

        fn init(m: TensorMeta) GeometrySnapshot {
            var out: GeometrySnapshot = .{ .dtype = m.dtype, .rank = m.rank };
            const rank: usize = @as(usize, m.rank);
            @memcpy(out.shape[0..rank], m.shape[0..rank]);
            @memcpy(out.tile_shape[0..rank], m.tile_shape[0..rank]);
            @memcpy(out.tile_counts[0..rank], m.tile_counts[0..rank]);
            @memcpy(out.tile_strides[0..rank], m.tile_strides[0..rank]);
            return out;
        }

        fn eql(self: GeometrySnapshot, m: TensorMeta) bool {
            const rank: usize = @as(usize, self.rank);
            if (self.rank != m.rank or self.dtype != m.dtype) return false;
            if (!std.mem.eql(usize, self.shape[0..rank], m.shape)) return false;
            if (!std.mem.eql(usize, self.tile_shape[0..rank], m.tile_shape)) return false;
            if (!std.mem.eql(usize, self.tile_counts[0..rank], m.tile_counts)) return false;
            if (!std.mem.eql(usize, self.tile_strides[0..rank], m.tile_strides)) return false;
            return true;
        }
    };

    fn flushAllDevDirty(self: *Self, id: TensorId) StoreError!void {
        const n = try self.totalTiles(id);
        var i: usize = 0;
        while (i < n) : (i += 1) try self.flushIfDevDirty(id, i);
    }

    fn dropDeviceStatesForTensor(self: *Self, id: TensorId) StoreError!void {
        var keys = std.ArrayList(u64).empty;
        defer keys.deinit(self.allocator);

        var it = self.states.iterator();
        while (it.next()) |e| {
            if ((e.key_ptr.* >> 32) == @as(u64, id)) {
                keys.append(self.allocator, e.key_ptr.*) catch return StoreError.OutOfMemory;
            }
        }

        for (keys.items) |k| {
            if (self.states.fetchRemove(k)) |kv| {
                if (kv.value.handle != 0) self.dev.free(kv.value.handle);
            }
        }
    }

    fn swapTensors(ctx: *anyopaque, a: TensorId, b: TensorId) StoreError!void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        // Swap host backing first (validates layout compatibility).
        try self.inner.swapTensors(a, b);
        // Device state follows the data: tile (a,i) and (b,i) exchange residency.
        const n = try self.totalTiles(a);
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const ka = key(a, i);
            const kb = key(b, i);
            const sa: ?TileState = if (self.states.get(ka)) |v| v else null;
            const sb: ?TileState = if (self.states.get(kb)) |v| v else null;
            if (sb) |v| {
                self.states.put(ka, v) catch return StoreError.OutOfMemory;
            } else {
                _ = self.states.remove(ka);
            }
            if (sa) |v| {
                self.states.put(kb, v) catch return StoreError.OutOfMemory;
            } else {
                _ = self.states.remove(kb);
            }
        }
    }

    const vtable = TensorStore.VTable{
        .meta = meta,
        .acquireTileConst = acquireTileConst,
        .acquireTileMut = acquireTileMut,
        .acquireTileConstLinear = acquireTileConstLinear,
        .acquireTileMutLinear = acquireTileMutLinear,
        .releaseConst = releaseConst,
        .releaseMut = releaseMut,
        .kvCachePolicyInfo = kvCachePolicyInfo,
        .mapKVCacheTime = mapKVCacheTime,
        .prefetch = prefetch,
        .prefetchLinear = prefetchLinear,
        .swapTensors = swapTensors,
    };
};
