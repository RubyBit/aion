// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! `CheckedTensorStore` — a `TensorStore` decorator that validates the
//! acquire/release lease discipline that a device (GPU) residency layer will
//! depend on.
//!
//! Why this exists: today the CPU store's tokens are no-ops, so a missing
//! `release` is invisible. A GPU residency store, by contrast, uses
//! acquire→use→release to decide when a device buffer may be evicted and when a
//! mutated tile must be flushed back. Before we build that store (see the
//! residency decorator in WS4), we want a mechanical guarantee that the
//! existing execution paths are lease-disciplined.
//!
//! This decorator wraps any inner `TensorStore`, delegates every call
//! unchanged, and tracks a signed count of live leases. The invariants:
//!   - the live count must never go negative (no release without acquire), and
//!   - it must be zero once a program finishes (no leaked tile lease).
//!
//! It is a test/diagnostic tool: production code keeps using the plain store,
//! so there is zero runtime cost on the CPU hot path. It also doubles as the
//! structural template for the device-residency decorator.

const std = @import("std");
const tensor_store = @import("../tensor_store.zig");

const TensorStore = tensor_store.TensorStore;
const TensorId = tensor_store.TensorId;
const StoreError = tensor_store.StoreError;
const TileRefConst = tensor_store.TileRefConst;
const TileRefMut = tensor_store.TileRefMut;
const TensorMeta = tensor_store.TensorMeta;
const KVCachePolicyInfo = tensor_store.KVCachePolicyInfo;

pub const LeaseStats = struct {
    /// Currently-live leases (acquires minus releases). Must be >= 0 always and
    /// == 0 at end of a completed program.
    live: i64 = 0,
    /// High-water mark of `live` — useful as a sanity figure in tests.
    max_live: i64 = 0,
    /// Total acquire calls observed.
    total_acquires: u64 = 0,
    /// Set if a release ever drove `live` below zero (release without acquire).
    underflow: bool = false,

    fn onAcquire(self: *LeaseStats) void {
        self.live += 1;
        self.total_acquires += 1;
        if (self.live > self.max_live) self.max_live = self.live;
    }

    fn onRelease(self: *LeaseStats) void {
        self.live -= 1;
        if (self.live < 0) self.underflow = true;
    }

    pub fn balanced(self: *const LeaseStats) bool {
        return self.live == 0 and !self.underflow;
    }
};

pub const CheckedTensorStore = struct {
    inner: TensorStore,
    stats: LeaseStats = .{},

    const Self = @This();

    pub fn init(inner: TensorStore) Self {
        return .{ .inner = inner, .stats = .{} };
    }

    /// Returns a `TensorStore` view bound to this decorator. The decorator must
    /// remain pinned (not moved) for the lifetime of the returned handle.
    pub fn tensorStore(self: *Self) TensorStore {
        return .{ .ctx = @ptrCast(self), .vtable = &vtable };
    }

    fn meta(ctx: *anyopaque, id: TensorId) StoreError!TensorMeta {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return self.inner.meta(id);
    }

    fn acquireTileConst(ctx: *anyopaque, id: TensorId, ti0: usize, ti1: usize) StoreError!TileRefConst {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const r = try self.inner.acquireTileConst(id, ti0, ti1);
        self.stats.onAcquire();
        return r;
    }

    fn acquireTileMut(ctx: *anyopaque, id: TensorId, ti0: usize, ti1: usize) StoreError!TileRefMut {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const r = try self.inner.acquireTileMut(id, ti0, ti1);
        self.stats.onAcquire();
        return r;
    }

    fn acquireTileConstLinear(ctx: *anyopaque, id: TensorId, tile_index: usize) StoreError!TileRefConst {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const r = try self.inner.acquireTileConstLinear(id, tile_index);
        self.stats.onAcquire();
        return r;
    }

    fn acquireTileMutLinear(ctx: *anyopaque, id: TensorId, tile_index: usize) StoreError!TileRefMut {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const r = try self.inner.acquireTileMutLinear(id, tile_index);
        self.stats.onAcquire();
        return r;
    }

    fn releaseConst(ctx: *anyopaque, token: usize) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.stats.onRelease();
        self.inner.releaseConst(token);
    }

    fn releaseMut(ctx: *anyopaque, token: usize) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.stats.onRelease();
        self.inner.releaseMut(token);
    }

    fn kvCachePolicyInfo(ctx: *anyopaque, id: TensorId) KVCachePolicyInfo {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return self.inner.kvCachePolicyInfo(id);
    }

    fn mapKVCacheTime(ctx: *anyopaque, id: TensorId, logical_t: usize, physical_capacity_tokens: usize) StoreError!usize {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return self.inner.mapKVCacheTime(id, logical_t, physical_capacity_tokens);
    }

    fn prefetch(ctx: *anyopaque, id: TensorId, ti0: usize, ti1: usize) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.inner.prefetch(id, ti0, ti1);
    }

    fn prefetchLinear(ctx: *anyopaque, id: TensorId, tile_index: usize) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.inner.prefetchLinear(id, tile_index);
    }

    fn swapTensors(ctx: *anyopaque, a: TensorId, b: TensorId) StoreError!void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return self.inner.swapTensors(a, b);
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
