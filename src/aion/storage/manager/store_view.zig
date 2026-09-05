// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Adapts `StorageManager` to the runtime's `TensorStore` interface.
//! This keeps backends independent of tensor lifetime, residency, and derived weights.

const std = @import("std");

const manager_mod = @import("../manager.zig");
const cache_mod = @import("../cache.zig");
const storage_mod = @import("../storage.zig");
const tensor_store = @import("../../runtime/tensor_store.zig");
const device_store = @import("../../runtime/device_store.zig");
const dm = @import("../../runtime/device_memory.zig");

const StorageManager = manager_mod.StorageManager;
const TiledTensor = storage_mod.TiledTensor;
const SequenceCachePolicy = cache_mod.SequenceCachePolicy;
const StorageError = storage_mod.StorageError;
const TensorId = manager_mod.TensorId;
const DeviceRef = storage_mod.DeviceRef;

/// The interface view of `mgr`. Borrows it: the returned store is valid for as long as
/// the manager is.
pub fn of(mgr: *StorageManager) tensor_store.TensorStore {
    const Vt = struct {
        fn shouldLease(policy: SequenceCachePolicy) bool {
            return switch (policy) {
                .none => false,
                else => true,
            };
        }

        fn toStorePolicyInfo(info: cache_mod.SequenceCachePolicyInfo) tensor_store.SequenceCachePolicyInfo {
            const kind: tensor_store.SequenceCachePolicyKind = switch (info.kind) {
                .none => .none,
                .growable => .growable,
                .rolling => .rolling,
            };
            return .{ .kind = kind, .rolling_history_tokens = info.rolling_history_tokens };
        }

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
            const backing = sm.backingConst(@intCast(id)) catch return tensor_store.StoreError.InvalidArgument;
            const tile = t.acquireTileConstFrom(backing.data, ti0, ti1) catch return tensor_store.StoreError.InvalidArgument;

            var token: usize = 0;
            if (sm.cache) |*cache| {
                if (shouldLease(cache.tensorPolicy(@intCast(id)))) {
                    token = cache.acquireLease(@intCast(id), t.tileIndex(ti0, ti1) catch return tensor_store.StoreError.InvalidArgument, false) catch |e| {
                        return switch (e) {
                            error.OutOfMemoryRam => tensor_store.StoreError.OutOfMemory,
                            else => tensor_store.StoreError.InvalidArgument,
                        };
                    };
                }
            }

            return .{
                .bytes = tile.bytes,
                .dtype = tile.dtype,
                .rank = tile.rank,
                .shape_mem = tile.shape_mem,
                .strides_mem = tile.strides_mem,
                .token = token,
            };
        }

        fn acquireTileMut(ctx: *anyopaque, id: tensor_store.TensorId, ti0: usize, ti1: usize) tensor_store.StoreError!tensor_store.TileRefMut {
            const sm: *StorageManager = @ptrCast(@alignCast(ctx));
            const t: *TiledTensor = sm.getMut(@intCast(id)) catch return tensor_store.StoreError.InvalidArgument;
            const backing = sm.backingMut(@intCast(id)) catch return tensor_store.StoreError.InvalidArgument;
            const tile = t.acquireTileMutFrom(backing.data, ti0, ti1) catch return tensor_store.StoreError.InvalidArgument;

            var token: usize = 0;
            if (sm.cache) |*cache| {
                if (shouldLease(cache.tensorPolicy(@intCast(id)))) {
                    token = cache.acquireLease(@intCast(id), t.tileIndex(ti0, ti1) catch return tensor_store.StoreError.InvalidArgument, true) catch |e| {
                        return switch (e) {
                            error.OutOfMemoryRam => tensor_store.StoreError.OutOfMemory,
                            else => tensor_store.StoreError.InvalidArgument,
                        };
                    };
                }
            }

            return .{
                .bytes = tile.bytes,
                .dtype = tile.dtype,
                .rank = tile.rank,
                .shape_mem = tile.shape_mem,
                .strides_mem = tile.strides_mem,
                .token = token,
            };
        }

        fn acquireTileConstLinear(ctx: *anyopaque, id: tensor_store.TensorId, tile_index: usize) tensor_store.StoreError!tensor_store.TileRefConst {
            const sm: *StorageManager = @ptrCast(@alignCast(ctx));
            const t: *const TiledTensor = sm.getConst(@intCast(id)) catch return tensor_store.StoreError.InvalidArgument;
            const backing = sm.backingConst(@intCast(id)) catch return tensor_store.StoreError.InvalidArgument;
            const tile = t.acquireTileConstLinearFrom(backing.data, tile_index) catch return tensor_store.StoreError.InvalidArgument;

            var token: usize = 0;
            if (sm.cache) |*cache| {
                if (shouldLease(cache.tensorPolicy(@intCast(id)))) {
                    token = cache.acquireLease(@intCast(id), tile_index, false) catch |e| {
                        return switch (e) {
                            error.OutOfMemoryRam => tensor_store.StoreError.OutOfMemory,
                            else => tensor_store.StoreError.InvalidArgument,
                        };
                    };
                }
            }

            return .{
                .bytes = tile.bytes,
                .dtype = tile.dtype,
                .rank = tile.rank,
                .shape_mem = tile.shape_mem,
                .strides_mem = tile.strides_mem,
                .token = token,
            };
        }

        fn acquireTileMutLinear(ctx: *anyopaque, id: tensor_store.TensorId, tile_index: usize) tensor_store.StoreError!tensor_store.TileRefMut {
            const sm: *StorageManager = @ptrCast(@alignCast(ctx));
            const t: *TiledTensor = sm.getMut(@intCast(id)) catch return tensor_store.StoreError.InvalidArgument;
            const backing = sm.backingMut(@intCast(id)) catch return tensor_store.StoreError.InvalidArgument;
            const tile = t.acquireTileMutLinearFrom(backing.data, tile_index) catch return tensor_store.StoreError.InvalidArgument;

            var token: usize = 0;
            if (sm.cache) |*cache| {
                if (shouldLease(cache.tensorPolicy(@intCast(id)))) {
                    token = cache.acquireLease(@intCast(id), tile_index, true) catch |e| {
                        return switch (e) {
                            error.OutOfMemoryRam => tensor_store.StoreError.OutOfMemory,
                            else => tensor_store.StoreError.InvalidArgument,
                        };
                    };
                }
            }

            return .{
                .bytes = tile.bytes,
                .dtype = tile.dtype,
                .rank = tile.rank,
                .shape_mem = tile.shape_mem,
                .strides_mem = tile.strides_mem,
                .token = token,
            };
        }

        fn releaseConst(ctx: *anyopaque, token: usize) void {
            if (token == 0) return;
            const sm: *StorageManager = @ptrCast(@alignCast(ctx));
            if (sm.cache) |*cache| cache.releaseLease(token);
        }

        fn releaseMut(ctx: *anyopaque, token: usize) void {
            if (token == 0) return;
            const sm: *StorageManager = @ptrCast(@alignCast(ctx));
            if (sm.cache) |*cache| cache.releaseLease(token);
        }

        fn sequenceCachePolicyInfo(ctx: *anyopaque, id: tensor_store.TensorId) tensor_store.SequenceCachePolicyInfo {
            const sm: *StorageManager = @ptrCast(@alignCast(ctx));
            const info: cache_mod.SequenceCachePolicyInfo = sm.sequenceCachePolicyInfo(@intCast(id));
            return toStorePolicyInfo(info);
        }

        fn mapSequenceStep(ctx: *anyopaque, id: tensor_store.TensorId, logical_t: usize, physical_capacity_tokens: usize) tensor_store.StoreError!usize {
            const sm: *StorageManager = @ptrCast(@alignCast(ctx));
            const tid: TensorId = @intCast(id);

            // Default identity mapping without policy support.
            if (sm.cache == null) {
                if (logical_t >= physical_capacity_tokens) return tensor_store.StoreError.InvalidArgument;
                return logical_t;
            }

            const policy: SequenceCachePolicy = sm.sequenceCachePolicy(tid);
            switch (policy) {
                .growable => |g| {
                    // Grow along the canonical time axis (axis 1).
                    const t_const: *const TiledTensor = sm.getConst(tid) catch return tensor_store.StoreError.InvalidArgument;
                    if (t_const.rank != 4) return tensor_store.StoreError.InvalidArgument;
                    const current_cap: usize = t_const.shape[1];

                    if (logical_t >= current_cap) {
                        // Past the growth ceiling (the caller's max bound) is an error,
                        // not an unbounded grow.
                        if (g.max_capacity_tokens != 0 and logical_t >= g.max_capacity_tokens) return tensor_store.StoreError.InvalidArgument;
                        var target: usize = current_cap;
                        if (g.initial_capacity_tokens > target) target = g.initial_capacity_tokens;
                        while (target <= logical_t) {
                            target = StorageManager.growTarget(target, g.growth_numerator, g.growth_denominator) catch return tensor_store.StoreError.InvalidArgument;
                        }
                        // Never overshoot the ceiling.
                        if (g.max_capacity_tokens != 0 and target > g.max_capacity_tokens) target = g.max_capacity_tokens;
                        sm.ensureTensorAxisCapacity(tid, 1, target) catch return tensor_store.StoreError.InvalidArgument;
                    }

                    // Update cache policy internal state bookkeeping.
                    if (sm.cache) |*cache| {
                        const latest_cap: usize = (sm.getConst(tid) catch return tensor_store.StoreError.InvalidArgument).shape[1];
                        _ = cache.mapLogicalTime(tid, logical_t, latest_cap) catch |e| {
                            return switch (e) {
                                error.OutOfMemoryRam => tensor_store.StoreError.OutOfMemory,
                                else => tensor_store.StoreError.InvalidArgument,
                            };
                        };
                    }

                    return logical_t;
                },
                else => {
                    if (sm.cache) |*cache| {
                        const t_const: *const TiledTensor = sm.getConst(tid) catch return tensor_store.StoreError.InvalidArgument;
                        var cap: usize = physical_capacity_tokens;
                        if (@as(usize, t_const.rank) > 1) cap = t_const.shape[1];
                        if (cap == 0) return tensor_store.StoreError.InvalidArgument;
                        return cache.mapLogicalTime(tid, logical_t, cap) catch |e| {
                            return switch (e) {
                                error.OutOfMemoryRam => tensor_store.StoreError.OutOfMemory,
                                else => tensor_store.StoreError.InvalidArgument,
                            };
                        };
                    }

                    if (logical_t >= physical_capacity_tokens) return tensor_store.StoreError.InvalidArgument;
                    return logical_t;
                },
            }
        }

        fn prefetch(ctx: *anyopaque, id: tensor_store.TensorId, ti0: usize, ti1: usize) void {
            const sm: *StorageManager = @ptrCast(@alignCast(ctx));
            const t: *const TiledTensor = sm.getConst(@intCast(id)) catch return;
            const backing = sm.backingConst(@intCast(id)) catch return;
            const tile = t.acquireTileConstFrom(backing.data, ti0, ti1) catch return;
            @prefetch(tile.bytes.ptr, .{ .rw = .read, .locality = 3, .cache = .data });
        }

        fn prefetchLinear(ctx: *anyopaque, id: tensor_store.TensorId, tile_index: usize) void {
            const sm: *StorageManager = @ptrCast(@alignCast(ctx));
            const t: *const TiledTensor = sm.getConst(@intCast(id)) catch return;
            const backing = sm.backingConst(@intCast(id)) catch return;
            const tile = t.acquireTileConstLinearFrom(backing.data, tile_index) catch return;
            @prefetch(tile.bytes.ptr, .{ .rw = .read, .locality = 3, .cache = .data });
        }

        fn sameShape(a: []const usize, b: []const usize) bool {
            if (a.len != b.len) return false;
            for (a, 0..) |v, i| if (v != b[i]) return false;
            return true;
        }

        fn deviceTile(ctx: *anyopaque, id: tensor_store.TensorId, tile_index: usize) tensor_store.StoreError!?tensor_store.DeviceTileRef {
            const sm: *StorageManager = @ptrCast(@alignCast(ctx));
            const t: *const TiledTensor = sm.getConst(@intCast(id)) catch return tensor_store.StoreError.InvalidArgument;
            const backing = sm.backingConst(@intCast(id)) catch return tensor_store.StoreError.InvalidArgument;
            if (backing.device.kind == .cpu) return null;
            if (tile_index >= backing.tile_handles.len) return tensor_store.StoreError.InvalidArgument;
            const layout = t.tileLayoutLinear(tile_index) catch return tensor_store.StoreError.InvalidArgument;
            return .{
                .handle = backing.tile_handles[tile_index],
                .len = t.tile_lens[tile_index],
                .dtype = layout.dtype,
                .rank = layout.rank,
                .shape_mem = layout.shape_mem,
                .strides_mem = layout.strides_mem,
            };
        }

        fn swapTensors(ctx: *anyopaque, a_id: tensor_store.TensorId, b_id: tensor_store.TensorId) tensor_store.StoreError!void {
            const sm: *StorageManager = @ptrCast(@alignCast(ctx));
            var a: *TiledTensor = sm.getMut(@intCast(a_id)) catch return tensor_store.StoreError.InvalidArgument;
            var b: *TiledTensor = sm.getMut(@intCast(b_id)) catch return tensor_store.StoreError.InvalidArgument;

            if (a.dtype != b.dtype) return tensor_store.StoreError.InvalidArgument;
            if (a.rank != b.rank) return tensor_store.StoreError.InvalidArgument;
            if (a.quant_axis != b.quant_axis) return tensor_store.StoreError.InvalidArgument;
            if (!sameShape(a.shape, b.shape)) return tensor_store.StoreError.InvalidArgument;
            if (!sameShape(a.tile_shape, b.tile_shape)) return tensor_store.StoreError.InvalidArgument;
            if (!sameShape(a.tile_counts, b.tile_counts)) return tensor_store.StoreError.InvalidArgument;
            if (!sameShape(a.tile_strides, b.tile_strides)) return tensor_store.StoreError.InvalidArgument;
            if (a.tile_offsets.len != b.tile_offsets.len or a.tile_lens.len != b.tile_lens.len) return tensor_store.StoreError.InvalidArgument;
            if (a.tile_alignment != b.tile_alignment) return tensor_store.StoreError.InvalidArgument;
            // Zero-copy carried-variable swap for CPU loop execution. Move
            // the complete backing record between the logical tensor ids.
            std.mem.swap([]align(64) u8, &a.data, &b.data);
            std.mem.swap(bool, &a.owns_data, &b.owns_data);
            std.mem.swap(DeviceRef, &a.device, &b.device);
            std.mem.swap([]dm.DeviceHandle, &a.tile_handles, &b.tile_handles);
            std.mem.swap(?dm.DeviceMemory, &a.dev, &b.dev);
            // The host-write counter travels with the bytes so a residency
            // layer's per-tile `uploaded_seq` (which the resident store swaps
            // alongside) stays consistent and avoids a spurious re-upload.
        }
    };

    return .{
        .ctx = @ptrCast(mgr),
        .vtable = &.{
            .meta = Vt.meta,
            .acquireTileConst = Vt.acquireTileConst,
            .acquireTileMut = Vt.acquireTileMut,
            .acquireTileConstLinear = Vt.acquireTileConstLinear,
            .acquireTileMutLinear = Vt.acquireTileMutLinear,
            .releaseConst = Vt.releaseConst,
            .releaseMut = Vt.releaseMut,
            .sequenceCachePolicyInfo = Vt.sequenceCachePolicyInfo,
            .mapSequenceStep = Vt.mapSequenceStep,
            .prefetch = Vt.prefetch,
            .prefetchLinear = Vt.prefetchLinear,
            .swapTensors = Vt.swapTensors,
            .deviceTile = Vt.deviceTile,
        },
    };
}
