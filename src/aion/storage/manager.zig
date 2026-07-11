// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const storage_mod = @import("storage.zig");
const cache_mod = @import("cache.zig");
const types = @import("../backend/types.zig");
const tensor_store = @import("../runtime/tensor_store.zig");
const dm = @import("../runtime/residency/device_memory.zig");
const plan = @import("../graph/plan.zig");

pub const StorageError = storage_mod.StorageError;
pub const TiledTensor = storage_mod.TiledTensor;
pub const DeviceRef = storage_mod.DeviceRef;
pub const DType = types.DType;
pub const Cache = cache_mod.Cache;
pub const CacheConfig = cache_mod.CacheConfig;
pub const CachePolicy = cache_mod.CachePolicy;
pub const SequenceCachePolicy = cache_mod.SequenceCachePolicy;
pub const SequenceCachePolicyInfo = cache_mod.SequenceCachePolicyInfo;

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

    /// Device registry for tensor migration (`moveTensor` / `Tensor.to`). Set by the
    /// owning `Context` via `setDeviceRegistry`. `device_registry[i]` is gpu[i]; the
    /// slice is borrowed (Context owns the backing). `cpu_policy` retiles on `.to(.cpu)`.
    device_registry: []const DeviceEntry = &[_]DeviceEntry{},
    cpu_policy: plan.TilePolicy = .{},

    const DerivedWeight = struct { kind: u32, sources: []TensorId, result: TensorId };

    /// One registered non-cpu device: how to (de)allocate/transfer its buffers, plus
    /// the tile policy used when a tensor is migrated onto it.
    pub const DeviceEntry = struct { mem: dm.DeviceMemory, policy: plan.TilePolicy };

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

    pub fn registerSequenceCachePolicy(self: *Self, id: TensorId, policy: SequenceCachePolicy) StorageError!void {
        _ = try self.getConst(id);
        var cache_ptr: *Cache = if (self.cache) |*c| c else return StorageError.InvalidArgument;
        cache_ptr.registerTensorPolicy(id, policy) catch |e| return mapCacheError(e);
    }

    pub fn ensureTensorAxisCapacity(self: *Self, id: TensorId, axis: usize, min_size: usize) StorageError!void {
        const t0: *const TiledTensor = try self.getConst(id);
        if (axis >= @as(usize, t0.rank)) return StorageError.InvalidArgument;
        if (t0.shape[axis] >= min_size) return;
        if (t0.device.kind == .cpu) {
            return (try self.getMut(id)).growAxisPreserveScalar(axis, min_size);
        }

        // Device-exclusive: round-trip through the host (D2H gather -> host grow ->
        // H2D re-migrate), reusing `moveTensor` — no bespoke device copy. Growth is
        // geometric, so this amortizes to O(final size) and never touches the
        // fixed/ring fast path. `moveTensor` frees the old device buffer only after
        // the re-upload, and the tensor id + single-tile geometry are preserved.
        const target: DeviceRef = t0.device;
        const dev: dm.DeviceMemory = t0.dev orelse return StorageError.InvalidArgument;
        const tile_align: usize = t0.tile_alignment;
        const rank: usize = @as(usize, t0.rank);
        var shape_buf: [8]usize = undefined;
        @memcpy(shape_buf[0..rank], t0.shape);

        try self.moveTensor(id, .{ .kind = .cpu }, null, shape_buf[0..rank], tile_align);
        try (try self.getMut(id)).growAxisPreserveScalar(axis, min_size);

        const grown: *const TiledTensor = try self.getConst(id);
        var grown_shape: [8]usize = undefined;
        @memcpy(grown_shape[0..rank], grown.shape);
        try self.moveTensor(id, target, dev, grown_shape[0..rank], tile_align);
    }

    pub fn sequenceCachePolicy(self: *const Self, id: TensorId) SequenceCachePolicy {
        if (self.cache) |*c| return c.tensorPolicy(id);
        return .{ .none = {} };
    }

    pub fn sequenceCachePolicyInfo(self: *const Self, id: TensorId) SequenceCachePolicyInfo {
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

    // --- Device registry + migration (move semantics) ---

    /// Install the device registry used by `moveTensor` / `Tensor.to`. `gpu_entries`
    /// is borrowed (the caller — a `Context` — owns the backing for its lifetime).
    pub fn setDeviceRegistry(self: *Self, cpu_policy: plan.TilePolicy, gpu_entries: []const DeviceEntry) void {
        self.cpu_policy = cpu_policy;
        self.device_registry = gpu_entries;
    }

    /// The `DeviceMemory` for `ref`, or null for cpu / an unregistered index.
    pub fn deviceMemoryFor(self: *const Self, ref: DeviceRef) ?dm.DeviceMemory {
        if (ref.kind == .cpu) return null;
        if (ref.index >= self.device_registry.len) return null;
        return self.device_registry[ref.index].mem;
    }

    /// The tile policy for `ref` (cpu policy for cpu / unknown index).
    pub fn policyFor(self: *const Self, ref: DeviceRef) plan.TilePolicy {
        if (ref.kind == .cpu) return self.cpu_policy;
        if (ref.index >= self.device_registry.len) return self.cpu_policy;
        return self.device_registry[ref.index].policy;
    }

    /// Which device a tensor's bytes currently live on.
    pub fn tensorDevice(self: *const Self, id: TensorId) StorageError!DeviceRef {
        const t: *const TiledTensor = try self.getConst(id);
        return t.device;
    }

    /// Gather a tensor's bytes into a packed (device-independent) buffer, regardless
    /// of where it currently lives. For a gpu-resident tensor this rebuilds the bytes
    /// host-side (in the tensor's current tiling) via D2H, then reads them packed.
    fn gatherPacked(self: *Self, t: *const TiledTensor, out: []u8) StorageError!void {
        if (t.device.kind == .cpu) {
            return if (t.dtype.info().is_quantized) t.readToPackedQuant(out) else t.readToPackedScalar(out);
        }
        const d = t.dev orelse return StorageError.InvalidArgument;
        var tmp: TiledTensor = undefined;
        try tmp.init(self.allocator, t.dtype, t.shape, t.tile_shape, .{ .tile_alignment = t.tile_alignment, .quant_axis = t.quant_axis });
        defer tmp.deinit();
        // `tmp` has geometry identical to `t` (same params) → matching tile offsets/lens.
        for (t.tile_handles, 0..) |h, i| {
            const off = tmp.tile_offsets[i];
            const len = tmp.tile_lens[i];
            d.copyD2H(tmp.data[off .. off + len], h, 0) catch return StorageError.InvalidArgument;
        }
        return if (t.dtype.info().is_quantized) tmp.readToPackedQuant(out) else tmp.readToPackedScalar(out);
    }

    fn scatterPacked(t: *TiledTensor, packed_bytes: []const u8) StorageError!void {
        return if (t.dtype.info().is_quantized) t.writeFromPackedQuant(packed_bytes) else t.writeFromPackedScalar(packed_bytes);
    }

    /// Scatter device-independent packed bytes into `t`, honoring residency: a
    /// host tensor writes directly; a device-resident tensor stages into a host
    /// buffer in `t`'s tiling and uploads each tile (H2D). The device-writing
    /// counterpart of `gatherPacked` (same identical-geometry `tmp` trick).
    fn scatterPackedResident(self: *Self, t: *TiledTensor, packed_bytes: []const u8) StorageError!void {
        if (t.device.kind == .cpu) return scatterPacked(t, packed_bytes);
        const d = t.dev orelse return StorageError.InvalidArgument;
        var tmp: TiledTensor = undefined;
        try tmp.init(self.allocator, t.dtype, t.shape, t.tile_shape, .{ .tile_alignment = t.tile_alignment, .quant_axis = t.quant_axis });
        defer tmp.deinit();
        try scatterPacked(&tmp, packed_bytes);
        for (t.tile_handles, 0..) |h, i| {
            const off = tmp.tile_offsets[i];
            const len = tmp.tile_lens[i];
            d.copyH2D(h, 0, tmp.data[off .. off + len]) catch return StorageError.InvalidArgument;
        }
    }

    /// Copy `src_id`'s logical contents into `dst_id`, honoring each tensor's
    /// residency (any host/device combination). Gathers the source into the
    /// device-independent packed layout (D2H when the source is on a device),
    /// then scatters it into the destination (H2D when the destination is). dtype
    /// and logical shape must match. Used to seed a device-resident recurrent
    /// state slot from a caller's host tensor, and to mirror device state back to
    /// a host tensor for reads.
    pub fn copyTensorData(self: *Self, dst_id: TensorId, src_id: TensorId) StorageError!void {
        const src: *const TiledTensor = try self.getConst(src_id);
        const dst: *TiledTensor = try self.getMut(dst_id);
        if (src.dtype != dst.dtype or src.rank != dst.rank) return StorageError.InvalidArgument;
        for (src.shape, dst.shape) |a, b| if (a != b) return StorageError.InvalidArgument;

        const plen: usize = try src.packedByteLen();
        const buf: []u8 = self.allocator.alloc(u8, plen) catch return StorageError.OutOfMemory;
        defer self.allocator.free(buf);
        try self.gatherPacked(src, buf);
        try self.scatterPackedResident(dst, buf);
    }

    /// Zero a tensor's contents in place, honoring residency: a host tensor is
    /// memset (bumping `host_seq` so any staged device copy re-uploads); a
    /// device-resident tensor gets zeros uploaded (H2D) into each tile buffer.
    /// Used by `Model.resetState` so recurrent state clears correctly whether it
    /// lives on the host or exclusively on a device.
    pub fn zeroTensorData(self: *Self, id: TensorId) StorageError!void {
        const t: *TiledTensor = try self.getMut(id);
        if (t.device.kind == .cpu) {
            @memset(t.data, 0);
            t.host_seq +%= 1;
            return;
        }
        const d = t.dev orelse return StorageError.InvalidArgument;
        var max_len: usize = 0;
        for (t.tile_lens) |l| max_len = @max(max_len, l);
        const zeros: []u8 = self.allocator.alloc(u8, max_len) catch return StorageError.OutOfMemory;
        defer self.allocator.free(zeros);
        @memset(zeros, 0);
        for (t.tile_handles, 0..) |h, i| {
            d.copyH2D(h, 0, zeros[0..t.tile_lens[i]]) catch return StorageError.InvalidArgument;
        }
    }

    /// Migrate a tensor to `target` (move semantics: the source-device copy is freed),
    /// re-tiling to `target_tile_shape`. `dev` must be the target device's `DeviceMemory`
    /// when `target.kind != .cpu`. The tensor id and logical shape are preserved.
    pub fn moveTensor(
        self: *Self,
        id: TensorId,
        target: DeviceRef,
        dev: ?dm.DeviceMemory,
        target_tile_shape: []const usize,
        tile_alignment: usize,
    ) StorageError!void {
        const t: *TiledTensor = try self.getMut(id);
        if (t.device.eql(target)) return; // idempotent

        // 1. Gather current bytes into the device-independent packed layout.
        const plen: usize = try t.packedByteLen();
        const packed_buf: []u8 = self.allocator.alloc(u8, plen) catch return StorageError.OutOfMemory;
        defer self.allocator.free(packed_buf);
        try self.gatherPacked(t, packed_buf);

        // 2. Build staging tensor in the TARGET tiling (host-backed, zeroed) and scatter.
        var staging: TiledTensor = undefined;
        try staging.init(self.allocator, t.dtype, t.shape, target_tile_shape, .{
            .tile_alignment = tile_alignment,
            .quant_axis = t.quant_axis,
        });
        errdefer staging.deinit();
        try scatterPacked(&staging, packed_buf);

        // 3. If migrating to a device, upload each tile and hand ownership of the device
        //    buffers to `staging`, freeing its host staging buffer.
        if (target.kind != .cpu) {
            const d = dev orelse return StorageError.InvalidArgument;
            const n: usize = staging.tile_offsets.len;
            const handles: []dm.DeviceHandle = self.allocator.alloc(dm.DeviceHandle, n) catch return StorageError.OutOfMemory;
            var uploaded: usize = 0;
            errdefer {
                var j: usize = 0;
                while (j < uploaded) : (j += 1) d.free(handles[j]);
                self.allocator.free(handles);
            }
            while (uploaded < n) : (uploaded += 1) {
                const off = staging.tile_offsets[uploaded];
                const len = staging.tile_lens[uploaded];
                handles[uploaded] = d.alloc(len, 64) catch return StorageError.OutOfMemory;
                d.copyH2D(handles[uploaded], 0, staging.data[off .. off + len]) catch return StorageError.InvalidArgument;
            }
            staging.releaseData(); // host staging bytes are now on the device
            staging.device = target;
            staging.tile_handles = handles;
            staging.dev = d;
            staging.owns_data = false;
        }

        // 4. Replace in place: free the old backing (host or device), move staging in,
        //    and rebind the metadata slices to the moved-in instance's own storage.
        t.deinit();
        t.* = staging;
        t.shape = t.shape_storage.constSlice();
        t.tile_shape = t.tile_shape_storage.constSlice();
        t.tile_counts = t.tile_counts_storage.constSlice();
        t.tile_strides = t.tile_strides_storage.constSlice();
    }

    pub fn tensorStore(self: *Self) tensor_store.TensorStore {
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
                    .host_seq = t.host_seq,
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
                        // Grow along the sequence axis (axis 2) for rank-4 sequence caches.
                        const t_const: *const TiledTensor = sm.getConst(tid) catch return tensor_store.StoreError.InvalidArgument;
                        if (t_const.rank != 4) return tensor_store.StoreError.InvalidArgument;
                        const current_cap: usize = t_const.shape[2];

                        if (logical_t >= current_cap) {
                            // Past the growth ceiling (the caller's max bound) is an error,
                            // not an unbounded grow.
                            if (g.max_capacity_tokens != 0 and logical_t >= g.max_capacity_tokens) return tensor_store.StoreError.InvalidArgument;
                            var target: usize = current_cap;
                            if (g.initial_capacity_tokens > target) target = g.initial_capacity_tokens;
                            while (target <= logical_t) {
                                target = growTarget(target, g.growth_numerator, g.growth_denominator) catch return tensor_store.StoreError.InvalidArgument;
                            }
                            // Never overshoot the ceiling.
                            if (g.max_capacity_tokens != 0 and target > g.max_capacity_tokens) target = g.max_capacity_tokens;
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

            fn deviceTile(ctx: *anyopaque, id: tensor_store.TensorId, tile_index: usize) tensor_store.StoreError!?tensor_store.DeviceTileRef {
                const sm: *StorageManager = @ptrCast(@alignCast(ctx));
                const t: *const TiledTensor = sm.getConst(@intCast(id)) catch return tensor_store.StoreError.InvalidArgument;
                if (t.device.kind == .cpu) return null; // host-resident: caller stages it
                if (tile_index >= t.tile_handles.len) return tensor_store.StoreError.InvalidArgument;
                const layout = t.tileLayoutLinear(tile_index) catch return tensor_store.StoreError.InvalidArgument;
                return .{
                    .handle = t.tile_handles[tile_index],
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
                if (a.owns_data != b.owns_data) return tensor_store.StoreError.InvalidArgument;
                if (!a.device.eql(b.device)) return tensor_store.StoreError.InvalidArgument;

                // Zero-copy carried-variable swap (Loop control flow). For gpu-resident
                // tensors the bytes live in device buffers, so swap the handle slices;
                // in-flight commands hold raw buffer refs, future acquires see the swap.
                if (a.device.kind == .cpu) {
                    std.mem.swap([]align(64) u8, &a.data, &b.data);
                } else {
                    std.mem.swap([]dm.DeviceHandle, &a.tile_handles, &b.tile_handles);
                }
                // The host-write counter travels with the bytes so a residency
                // layer's per-tile `uploaded_seq` (which the resident store swaps
                // alongside) stays consistent and avoids a spurious re-upload.
                std.mem.swap(u64, &a.host_seq, &b.host_seq);
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
                .sequenceCachePolicyInfo = Vt.sequenceCachePolicyInfo,
                .mapSequenceStep = Vt.mapSequenceStep,
                .prefetch = Vt.prefetch,
                .prefetchLinear = Vt.prefetchLinear,
                .swapTensors = Vt.swapTensors,
                .deviceTile = Vt.deviceTile,
            },
        };
    }
};
