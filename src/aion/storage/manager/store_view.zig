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
const Tensor = storage_mod.Tensor;
const SequenceCachePolicy = cache_mod.SequenceCachePolicy;
const StorageError = storage_mod.StorageError;
const TensorId = manager_mod.TensorId;
const DeviceRef = storage_mod.DeviceRef;

/// The interface view of `mgr`. Borrows it: the returned store is valid for as long as
/// the manager is.
pub fn of(mgr: *StorageManager) tensor_store.TensorStore {
    const Vt = struct {
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
            const t: *const Tensor = sm.getConst(@intCast(id)) catch return tensor_store.StoreError.InvalidArgument;
            return .{
                .dtype = t.dtype,
                .rank = t.rank,
                .shape = t.shape,
                .chunks = t.chunkCount(),
                .block_order = t.block_order,
            };
        }

        /// `t`'s bytes as `backing` holds them: `t` itself, or the workspace slot it aliases.
        fn hostBytes(t: *const Tensor, backing: *const Tensor) tensor_store.StoreError![]u8 {
            const len = t.byteLen() catch return tensor_store.StoreError.InvalidArgument;
            if (backing.data.len < len) return tensor_store.StoreError.InvalidArgument;
            return backing.data[0..len];
        }

        fn layoutOf(t: *const Tensor, shape: *[tensor_store.INLINE_RANK]usize, strides: *[tensor_store.INLINE_RANK]isize) void {
            shape.* = @splat(0);
            strides.* = @splat(0);
            @memcpy(shape[0..t.rank], t.shape);
            tensor_store.packedStrides(t.dtype, t.shape, strides[0..t.rank]);
        }

        fn acquireConst(ctx: *anyopaque, id: tensor_store.TensorId) tensor_store.StoreError!tensor_store.ViewConst {
            const sm: *StorageManager = @ptrCast(@alignCast(ctx));
            const t: *const Tensor = sm.getConst(@intCast(id)) catch return tensor_store.StoreError.InvalidArgument;
            const backing = sm.backingConst(@intCast(id)) catch return tensor_store.StoreError.InvalidArgument;
            var v: tensor_store.ViewConst = .{ .bytes = try hostBytes(t, backing), .dtype = t.dtype, .rank = t.rank, .shape_mem = undefined, .strides_mem = undefined };
            layoutOf(t, &v.shape_mem, &v.strides_mem);
            return v;
        }

        fn acquireMut(ctx: *anyopaque, id: tensor_store.TensorId) tensor_store.StoreError!tensor_store.ViewMut {
            const sm: *StorageManager = @ptrCast(@alignCast(ctx));
            const t: *const Tensor = sm.getConst(@intCast(id)) catch return tensor_store.StoreError.InvalidArgument;
            const backing = sm.backingMut(@intCast(id)) catch return tensor_store.StoreError.InvalidArgument;
            // Shared bytes are read-only (a mapping would fault): take them back or copy.
            backing.ensureWritable() catch |e| return switch (e) {
                error.OutOfMemory => tensor_store.StoreError.OutOfMemory,
                else => tensor_store.StoreError.InvalidArgument,
            };
            var v: tensor_store.ViewMut = .{ .bytes = try hostBytes(t, backing), .dtype = t.dtype, .rank = t.rank, .shape_mem = undefined, .strides_mem = undefined };
            layoutOf(t, &v.shape_mem, &v.strides_mem);
            return v;
        }

        // Host bytes live as long as their tensor, so a lease pins nothing.
        fn releaseConst(_: *anyopaque, _: usize) void {}

        fn releaseMut(_: *anyopaque, _: usize) void {}

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
                    const t_const: *const Tensor = sm.getConst(tid) catch return tensor_store.StoreError.InvalidArgument;
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
                        const t_const: *const Tensor = sm.getConst(tid) catch return tensor_store.StoreError.InvalidArgument;
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

        fn prefetch(ctx: *anyopaque, id: tensor_store.TensorId) void {
            const sm: *StorageManager = @ptrCast(@alignCast(ctx));
            const backing = sm.backingConst(@intCast(id)) catch return;
            if (backing.data.len == 0) return;
            @prefetch(backing.data.ptr, .{ .rw = .read, .locality = 3, .cache = .data });
        }

        fn sameShape(a: []const usize, b: []const usize) bool {
            return std.mem.eql(usize, a, b);
        }

        fn deviceChunk(ctx: *anyopaque, id: tensor_store.TensorId, chunk: usize) tensor_store.StoreError!?tensor_store.DeviceChunkRef {
            const sm: *StorageManager = @ptrCast(@alignCast(ctx));
            const t: *const Tensor = sm.getConst(@intCast(id)) catch return tensor_store.StoreError.InvalidArgument;
            const backing = sm.backingConst(@intCast(id)) catch return tensor_store.StoreError.InvalidArgument;
            if (backing.device.kind == .cpu) return null;
            if (chunk >= t.chunkCount() or chunk >= backing.chunk_handles.len) return tensor_store.StoreError.InvalidArgument;
            const c = t.chunk(chunk);
            return .{ .handle = backing.chunk_handles[chunk], .len = c.len, .rows = c.rows };
        }

        fn swapTensors(ctx: *anyopaque, a_id: tensor_store.TensorId, b_id: tensor_store.TensorId) tensor_store.StoreError!void {
            const sm: *StorageManager = @ptrCast(@alignCast(ctx));
            var a: *Tensor = sm.getMut(@intCast(a_id)) catch return tensor_store.StoreError.InvalidArgument;
            var b: *Tensor = sm.getMut(@intCast(b_id)) catch return tensor_store.StoreError.InvalidArgument;

            if (a.dtype != b.dtype) return tensor_store.StoreError.InvalidArgument;
            if (a.rank != b.rank) return tensor_store.StoreError.InvalidArgument;
            if (a.quant_axis != b.quant_axis) return tensor_store.StoreError.InvalidArgument;
            if (!sameShape(a.shape, b.shape)) return tensor_store.StoreError.InvalidArgument;
            if (a.chunk_rows != b.chunk_rows) return tensor_store.StoreError.InvalidArgument;
            // Zero-copy carried-variable swap for CPU loop execution. Move
            // the complete backing record between the logical tensor ids.
            std.mem.swap([]u8, &a.data, &b.data);
            std.mem.swap(bool, &a.owns_data, &b.owns_data);
            std.mem.swap(?*storage_mod.SharedBytes, &a.shared, &b.shared);
            std.mem.swap(DeviceRef, &a.device, &b.device);
            std.mem.swap([]dm.DeviceHandle, &a.chunk_handles, &b.chunk_handles);
            std.mem.swap(?dm.DeviceMemory, &a.dev, &b.dev);
        }
    };

    return .{
        .ctx = @ptrCast(mgr),
        .vtable = &.{
            .meta = Vt.meta,
            .acquireConst = Vt.acquireConst,
            .acquireMut = Vt.acquireMut,
            .releaseConst = Vt.releaseConst,
            .releaseMut = Vt.releaseMut,
            .sequenceCachePolicyInfo = Vt.sequenceCachePolicyInfo,
            .mapSequenceStep = Vt.mapSequenceStep,
            .prefetch = Vt.prefetch,
            .swapTensors = Vt.swapTensors,
            .deviceChunk = Vt.deviceChunk,
        },
    };
}
