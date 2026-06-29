// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const storage_mod = @import("storage.zig");
const cache_mod = @import("cache.zig");
const types = @import("../backend/types.zig");
const tensor_store = @import("../runtime/tensor_store.zig");

pub const StorageError = storage_mod.StorageError;
pub const TiledTensor = storage_mod.TiledTensor;
pub const DType = types.DType;
pub const Cache = cache_mod.Cache;
pub const CacheConfig = cache_mod.CacheConfig;
pub const CachePolicy = cache_mod.CachePolicy;
pub const KVCachePolicy = cache_mod.KVCachePolicy;
pub const KVCachePolicyInfo = cache_mod.KVCachePolicyInfo;

/// Opaque handle to a tensor owned by `StorageManager`.
///
/// Stable across internal resizes because it is an index into a pointer table.
pub const TensorId = u32;

/// RAM-only storage manager.
///
/// This owns all `TiledTensor` allocations for a context.
pub const StorageManager = struct {
    allocator: std.mem.Allocator,
    tensors: std.ArrayList(*TiledTensor) = .empty,
    cache: ?Cache = null,

    /// Memoizes weight tensors derived from a set of source tensors by an
    /// optimization pass — concatenated, bias-folded, norm-folded, or otherwise
    /// fused weights. Keyed by an opaque pass-defined `kind` (so distinct
    /// derivations don't alias over the same sources) plus the ordered source ids.
    /// Model weights are shape-independent, so a derived weight is identical across
    /// the per-shape recompiles that hit the same store; without this memo each
    /// recompile would re-derive and leak (tensors are only freed on `deinit`).
    derived_weight_cache: std.ArrayList(DerivedWeight) = .empty,

    const DerivedWeight = struct { kind: u32, sources: []TensorId, result: TensorId };

    const Self = @This();

    fn mapCacheError(err: cache_mod.CacheError) StorageError {
        return switch (err) {
            error.OutOfMemoryRam => StorageError.OutOfMemory,
            else => StorageError.InvalidArgument,
        };
    }

    fn growTarget(current: usize, num: usize, den: usize) StorageError!usize {
        if (current == 0 or num == 0 or den == 0) return StorageError.InvalidArgument;
        const prod: usize = std.math.mul(usize, current, num) catch return StorageError.InvalidArgument;
        const rounded: usize = std.math.add(usize, prod, den - 1) catch return StorageError.InvalidArgument;
        const next: usize = rounded / den;
        if (next <= current) return std.math.add(usize, current, 1) catch return StorageError.InvalidArgument;
        return next;
    }

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{ .allocator = allocator };
    }

    pub fn initWithCache(allocator: std.mem.Allocator, cfg: CacheConfig) StorageError!Self {
        var out: Self = .{ .allocator = allocator };
        out.cache = cache_mod.Cache.init(allocator, cfg) catch |e| return mapCacheError(e);
        return out;
    }

    pub fn deinit(self: *Self) void {
        if (self.cache) |*c| c.deinit();
        for (self.derived_weight_cache.items) |entry| self.allocator.free(entry.sources);
        self.derived_weight_cache.deinit(self.allocator);
        for (self.tensors.items) |t| {
            t.deinit();
            self.allocator.destroy(t);
        }
        self.tensors.deinit(self.allocator);
        self.* = undefined;
    }

    /// Look up a previously derived weight for `(kind, sources)`. `kind` namespaces
    /// distinct derivations so they don't alias over the same sources. Null on miss.
    pub fn lookupDerivedWeight(self: *const Self, kind: u32, sources: []const TensorId) ?TensorId {
        for (self.derived_weight_cache.items) |entry| {
            if (entry.kind != kind or entry.sources.len != sources.len) continue;
            if (std.mem.eql(TensorId, entry.sources, sources)) return entry.result;
        }
        return null;
    }

    /// Record a derived weight, taking an owned copy of `sources` as the key.
    pub fn recordDerivedWeight(self: *Self, kind: u32, sources: []const TensorId, result: TensorId) StorageError!void {
        const ids_copy: []TensorId = self.allocator.dupe(TensorId, sources) catch return StorageError.OutOfMemory;
        errdefer self.allocator.free(ids_copy);
        self.derived_weight_cache.append(self.allocator, .{ .kind = kind, .sources = ids_copy, .result = result }) catch return StorageError.OutOfMemory;
    }

    /// A derived-weight entry seen from one of its sources: the fused `result`, the
    /// full ordered `sources`, and the queried source's `index` within them.
    /// Lets a weight-swap resolve "this logical weight is region `index` of `result`".
    pub const DerivedSourceRef = struct { kind: u32, result: TensorId, sources: []const TensorId, index: usize };

    /// Find the derived weight `source_tid` was folded into (if any), so a swap can
    /// write through to the fused tensor. Returns the first match.
    pub fn findDerivedWeightBySource(self: *const Self, source_tid: TensorId) ?DerivedSourceRef {
        for (self.derived_weight_cache.items) |entry| {
            for (entry.sources, 0..) |sid, i| {
                if (sid == source_tid) return .{ .kind = entry.kind, .result = entry.result, .sources = entry.sources, .index = i };
            }
        }
        return null;
    }

    pub fn configureCache(self: *Self, cfg: CacheConfig) StorageError!void {
        if (self.cache) |*existing| existing.deinit();
        self.cache = cache_mod.Cache.init(self.allocator, cfg) catch |e| return mapCacheError(e);
    }

    pub fn hasCache(self: *const Self) bool {
        return self.cache != null;
    }

    pub fn registerKVCachePolicy(self: *Self, id: TensorId, policy: KVCachePolicy) StorageError!void {
        _ = try self.getConst(id);
        var cache_ptr: *Cache = if (self.cache) |*c| c else return StorageError.InvalidArgument;
        cache_ptr.registerTensorPolicy(id, policy) catch |e| return mapCacheError(e);
    }

    pub fn ensureTensorAxisCapacity(self: *Self, id: TensorId, axis: usize, min_size: usize) StorageError!void {
        var t: *TiledTensor = try self.getMut(id);
        if (axis >= @as(usize, t.rank)) return StorageError.InvalidArgument;
        if (t.shape[axis] >= min_size) return;
        try t.growAxisPreserveScalar(axis, min_size);
    }

    pub fn kvCachePolicy(self: *const Self, id: TensorId) KVCachePolicy {
        if (self.cache) |*c| return c.tensorPolicy(id);
        return .{ .none = {} };
    }

    pub fn kvCachePolicyInfo(self: *const Self, id: TensorId) KVCachePolicyInfo {
        if (self.cache) |*c| return c.tensorPolicyInfo(id);
        return .{};
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

        const idx_usize: usize = self.tensors.items.len;
        self.tensors.append(self.allocator, t) catch return StorageError.OutOfMemory;
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
        return self.tensors.items[idx];
    }

    /// Free a tensor's backing buffer (keeping metadata) to reclaim memory once it
    /// is provably unused by the compiled program — e.g. a weight an optimization
    /// pass has fused away. The id stays valid for shape/dtype validation;
    /// executing against the tensor afterward is a bug. Idempotent.
    pub fn releaseTensorData(self: *Self, id: TensorId) StorageError!void {
        const t: *TiledTensor = try self.getMut(id);
        t.releaseData();
    }

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
            fn shouldLease(policy: KVCachePolicy) bool {
                return switch (policy) {
                    .none => false,
                    else => true,
                };
            }

            fn toStorePolicyInfo(info: cache_mod.KVCachePolicyInfo) tensor_store.KVCachePolicyInfo {
                const kind: tensor_store.KVCachePolicyKind = switch (info.kind) {
                    .none => .none,
                    .growable => .growable,
                    .ring => .ring,
                };
                return .{ .kind = kind, .ring_window_tokens = info.ring_window_tokens };
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
                const tile = t.acquireTileConst(ti0, ti1) catch return tensor_store.StoreError.InvalidArgument;

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
                const tile = t.acquireTileMut(ti0, ti1) catch return tensor_store.StoreError.InvalidArgument;

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
                const tile = t.acquireTileConstLinear(tile_index) catch return tensor_store.StoreError.InvalidArgument;

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
                const tile = t.acquireTileMutLinear(tile_index) catch return tensor_store.StoreError.InvalidArgument;

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

            fn kvCachePolicyInfo(ctx: *anyopaque, id: tensor_store.TensorId) tensor_store.KVCachePolicyInfo {
                const sm: *StorageManager = @ptrCast(@alignCast(ctx));
                const info: cache_mod.KVCachePolicyInfo = sm.kvCachePolicyInfo(@intCast(id));
                return toStorePolicyInfo(info);
            }

            fn mapKVCacheTime(ctx: *anyopaque, id: tensor_store.TensorId, logical_t: usize, physical_capacity_tokens: usize) tensor_store.StoreError!usize {
                const sm: *StorageManager = @ptrCast(@alignCast(ctx));
                const tid: TensorId = @intCast(id);

                // Default identity mapping without policy support.
                if (sm.cache == null) {
                    if (logical_t >= physical_capacity_tokens) return tensor_store.StoreError.InvalidArgument;
                    return logical_t;
                }

                const policy: KVCachePolicy = sm.kvCachePolicy(tid);
                switch (policy) {
                    .growable => |g| {
                        // Grow along T axis for rank-4 KV cache tensors.
                        const t_const: *const TiledTensor = sm.getConst(tid) catch return tensor_store.StoreError.InvalidArgument;
                        if (t_const.rank != 4) return tensor_store.StoreError.InvalidArgument;
                        const current_cap: usize = t_const.shape[2];

                        if (logical_t >= current_cap) {
                            var target: usize = current_cap;
                            if (g.initial_capacity_tokens > target) target = g.initial_capacity_tokens;
                            while (target <= logical_t) {
                                target = growTarget(target, g.growth_numerator, g.growth_denominator) catch return tensor_store.StoreError.InvalidArgument;
                            }
                            sm.ensureTensorAxisCapacity(tid, 2, target) catch return tensor_store.StoreError.InvalidArgument;
                        }

                        // Update cache policy internal state bookkeeping.
                        if (sm.cache) |*cache| {
                            const latest_cap: usize = (sm.getConst(tid) catch return tensor_store.StoreError.InvalidArgument).shape[2];
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
                            if (@as(usize, t_const.rank) > 2) cap = t_const.shape[2];
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
                const tile = t.acquireTileConst(ti0, ti1) catch return;
                @prefetch(tile.bytes.ptr, .{ .rw = .read, .locality = 3, .cache = .data });
            }

            fn prefetchLinear(ctx: *anyopaque, id: tensor_store.TensorId, tile_index: usize) void {
                const sm: *StorageManager = @ptrCast(@alignCast(ctx));
                const t: *const TiledTensor = sm.getConst(@intCast(id)) catch return;
                const tile = t.acquireTileConstLinear(tile_index) catch return;
                @prefetch(tile.bytes.ptr, .{ .rw = .read, .locality = 3, .cache = .data });
            }

            fn sameShape(a: []const usize, b: []const usize) bool {
                if (a.len != b.len) return false;
                for (a, 0..) |v, i| if (v != b[i]) return false;
                return true;
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
                if (a.owns_data != b.owns_data) return tensor_store.StoreError.InvalidArgument;

                std.mem.swap([]align(64) u8, &a.data, &b.data);
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
                .kvCachePolicyInfo = Vt.kvCachePolicyInfo,
                .mapKVCacheTime = Vt.mapKVCacheTime,
                .prefetch = Vt.prefetch,
                .prefetchLinear = Vt.prefetchLinear,
                .swapTensors = Vt.swapTensors,
            },
        };
    }
};
