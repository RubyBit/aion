// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const storage_mod = @import("storage.zig");
const cache_mod = @import("cache.zig");
const types = @import("../backend/types.zig");
const tensor_store = @import("../runtime/tensor_store.zig");
const dm = @import("../runtime/device_memory.zig");
const plan = @import("../graph/plan.zig");
const derived_mod = @import("derived.zig");
const store_view = @import("manager/store_view.zig");
const fold = @import("manager/fold.zig");
const grow = @import("manager/grow.zig");

pub const StorageError = storage_mod.StorageError;
pub const TiledTensor = storage_mod.TiledTensor;
pub const DeviceRef = storage_mod.DeviceRef;
pub const DType = types.DType;
pub const Cache = cache_mod.Cache;
pub const CacheConfig = cache_mod.CacheConfig;
pub const CachePolicy = cache_mod.CachePolicy;
pub const SequenceCachePolicy = cache_mod.SequenceCachePolicy;
pub const SequenceCachePolicyInfo = cache_mod.SequenceCachePolicyInfo;
pub const Derived = derived_mod;

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

    /// Weights an optimization pass derived from other weights (see `derived.zig`).
    /// Memoized because model weights are shape-independent: without it every
    /// per-shape recompile would re-derive and leak (tensors free only on `deinit`).
    derived: derived_mod.Table = .{},

    /// Device registry for tensor migration (`moveTensor` / `Tensor.to`). Set by the
    /// owning `Context` via `setDeviceRegistry`. `device_registry[i]` is gpu[i]; the
    /// slice is borrowed (Context owns the backing). `cpu_policy` retiles on `.to(.cpu)`.
    device_registry: []const DeviceEntry = &[_]DeviceEntry{},
    cpu_policy: plan.TilePolicy = .{},

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

    /// The next capacity a geometric growth schedule asks for. Public because both the
    /// growth path and the store view's `mapSequenceStep` need the same schedule.
    pub fn growTarget(current: usize, num: usize, den: usize) StorageError!usize {
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
        self.derived.deinit(self.allocator);
        for (self.tensors.items) |t| {
            t.deinit();
            self.allocator.destroy(t);
        }
        self.tensors.deinit(self.allocator);
        self.* = undefined;
    }

    /// A derived weight already built for these sources, at this tiling, on this device.
    pub fn derivedFind(
        self: *const Self,
        kind: derived_mod.Kind,
        tiles: []const usize,
        device: DeviceRef,
        sources: []const TensorId,
    ) ?TensorId {
        return self.derived.find(kind, tiles, device, sources);
    }

    pub fn derivedRecord(
        self: *Self,
        kind: derived_mod.Kind,
        tiles: []const usize,
        device: DeviceRef,
        result: TensorId,
        sources: []const derived_mod.Source,
    ) StorageError!void {
        return self.derived.record(self.allocator, kind, tiles, device, result, sources);
    }

    /// Where `id`'s bytes live, if a pass folded it into a derived weight.
    pub fn derivedLocate(self: *const Self, id: TensorId) ?derived_mod.Located {
        return self.derived.locate(id);
    }

    // Operations over the derived table live in `manager/fold.zig`; these keep them
    // reachable as `mgr.<op>` because that is how every caller reads.

    pub fn writeDerivedSource(self: *Self, id: TensorId, src: TensorId) StorageError!void {
        return fold.writeDerivedSource(self, id, src);
    }

    pub fn readDerivedSource(self: *Self, id: TensorId, dst: TensorId) StorageError!void {
        return fold.readDerivedSource(self, id, dst);
    }

    pub fn unfoldTensor(self: *Self, id: TensorId) StorageError!void {
        return fold.unfoldTensor(self, id);
    }

    pub fn collectDerived(self: *Self) void {
        return fold.collectDerived(self);
    }

    /// Give `id` its physical backing on `target`, preserving its tile geometry.
    ///
    /// The one way anything says "this tensor belongs on that device": placement,
    /// weight retargeting and the fusion passes all want exactly this, and each
    /// reimplementing the tile-shape/alignment plumbing is how they drift apart.
    /// No-op for a cpu target or a tensor already there. Unified-memory devices keep
    /// the host allocation and alias it inside the device-memory implementation.
    pub fn placeTensor(self: *Self, id: TensorId, target: DeviceRef) StorageError!void {
        if (target.kind == .cpu) return;
        const t = try self.getConst(id);
        if (t.device.eql(target)) return;
        const dev = self.deviceMemoryFor(target) orelse return StorageError.InvalidArgument;
        var tile_shape: [tensor_store.INLINE_RANK]usize = undefined;
        const rank: usize = @intCast(t.rank);
        @memcpy(tile_shape[0..rank], t.tile_shape);
        return self.moveTensor(id, target, dev, tile_shape[0..rank], self.policyFor(target).tile_alignment);
    }

    /// Count one live compiled program as naming `id`.
    ///
    /// Tensor liveness has to be a store-wide fact: a `Context` shares one store between
    /// models, so "does anything still read this weight" is not a question one model can
    /// answer by walking its own specializations. `graph/program/lease.zig` is the only
    /// caller; `collectDerived` is what reads the result. Both halves ignore an id the
    /// store does not hold, so a partly applied batch can only under-count a tensor
    /// nothing named.
    pub fn retainTensor(self: *Self, id: TensorId) void {
        const t = self.getMut(id) catch return;
        t.program_refs += 1;
    }

    pub fn releaseTensor(self: *Self, id: TensorId) void {
        const t = self.getMut(id) catch return;
        if (t.program_refs > 0) t.program_refs -= 1;
    }

    pub fn tensorProgramRefs(self: *const Self, id: TensorId) u32 {
        const t = self.getConst(id) catch return 0;
        return t.program_refs;
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

    /// Grow `id`'s axis `axis` to at least `min_size`, following its cache policy.
    /// Executed in `manager/grow.zig`.
    pub fn ensureTensorAxisCapacity(self: *Self, id: TensorId, axis: usize, min_size: usize) StorageError!void {
        return grow.ensureTensorAxisCapacity(self, id, axis, min_size);
    }

    /// Grow a rank-4 rolling cache and remap its retained logical rows from the
    /// old modulo to the new one. `logical_ends[b]` is the next write position.
    pub fn ensureRollingCacheCapacity(
        self: *Self,
        id: TensorId,
        min_size: usize,
        retained_history: usize,
        logical_ends: []const usize,
    ) StorageError!void {
        return grow.ensureRollingCacheCapacity(self, id, min_size, retained_history, logical_ends);
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

    /// Create only a tensor's logical tiled geometry. Compiler workspace values
    /// do not need physical bytes until liveness has assigned them to a slot.
    pub fn createTiledTensorMetadata(
        self: *Self,
        dtype: DType,
        shape: []const usize,
        tile_shape: []const usize,
        opts: TiledTensor.InitOptions,
    ) StorageError!TensorId {
        const id = try self.createTiledTensor(dtype, shape, tile_shape, opts);
        (try self.getMut(id)).releaseData();
        return id;
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

    fn backingId(self: *const Self, id: TensorId) StorageError!TensorId {
        var cur = id;
        var depth: usize = 0;
        while (true) : (depth += 1) {
            if (depth > 1) return StorageError.InvalidArgument;
            const t = try self.getConst(cur);
            cur = t.backing_owner orelse return cur;
        }
    }

    pub fn physicalBackingId(self: *const Self, id: TensorId) StorageError!TensorId {
        return self.backingId(id);
    }

    pub fn markWorkspaceTensor(self: *Self, id: TensorId) StorageError!void {
        (try self.getMut(id)).workspace_owned = true;
    }

    pub fn tensorIsWorkspace(self: *const Self, id: TensorId) StorageError!bool {
        return (try self.getConst(id)).workspace_owned;
    }

    /// The tensor holding `id`'s bytes: itself, or the workspace slot it aliases.
    /// Public for `manager/store_view.zig`, which resolves every tile through it.
    pub fn backingConst(self: *const Self, id: TensorId) StorageError!*const TiledTensor {
        return self.getConst(try self.backingId(id));
    }

    pub fn backingMut(self: *Self, id: TensorId) StorageError!*TiledTensor {
        return self.getMut(try self.backingId(id));
    }

    /// Make `id` a logical view of `owner`'s physical workspace allocation.
    /// Aliases are deliberately one level deep and never own backing bytes.
    pub fn aliasTensorBacking(self: *Self, id: TensorId, owner: TensorId) StorageError!void {
        if (id == owner) return;
        const canonical = try self.backingId(owner);
        if (canonical != owner) return StorageError.InvalidArgument;
        const t = try self.getMut(id);
        t.releaseData();
        t.backing_owner = owner;
    }

    /// Grow a canonical host backing to the capacity chosen by workspace
    /// planning. Contents are intentionally discarded: reusable workspace
    /// lifetimes must begin with a full write.
    pub fn reserveHostBacking(self: *Self, id: TensorId, bytes: usize) StorageError!void {
        const t = try self.getMut(id);
        if (t.backing_owner != null or t.device.kind != .cpu) return StorageError.InvalidArgument;
        if (t.data.len >= bytes) {
            t.backing_bytes = t.data.len;
            return;
        }
        if (t.data.len != 0 and t.owns_data) self.allocator.free(t.data);
        t.data = self.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(64), bytes) catch return StorageError.OutOfMemory;
        @memset(t.data, 0);
        t.owns_data = true;
        t.backing_bytes = bytes;
    }

    /// Free a tensor's backing buffer (keeping metadata) to reclaim memory once it
    /// is provably unused by the compiled program — e.g. a weight an optimization
    /// pass has fused away. The id stays valid for shape/dtype validation;
    /// executing against the tensor afterward is a bug. Idempotent.
    pub fn releaseTensorData(self: *Self, id: TensorId) StorageError!void {
        const t: *TiledTensor = try self.getMut(id);
        t.releaseData();
    }

    /// Bytes in the tensor's physical backing. This is the resource used for
    /// specialization-cache budgeting; it intentionally measures tiled storage,
    /// not logical packed size.
    pub fn tensorBackingBytes(self: *const Self, id: TensorId) StorageError!usize {
        const t = try self.getConst(id);
        if (t.backing_owner != null) return 0;
        return t.backing_bytes;
    }

    /// Bytes required by this tensor's own logical tiled layout, independent of
    /// whether it currently owns or aliases a physical backing.
    pub fn tensorLogicalBackingBytes(self: *const Self, id: TensorId) StorageError!usize {
        const t = try self.getConst(id);
        var bytes: usize = 0;
        for (t.tile_offsets, t.tile_lens) |offset, len| {
            bytes = @max(bytes, std.math.add(usize, offset, len) catch return StorageError.InvalidArgument);
        }
        return bytes;
    }

    pub fn tensorHasBacking(self: *const Self, id: TensorId) StorageError!bool {
        const t = try self.backingConst(id);
        return if (t.device.kind == .cpu) t.data.len != 0 else t.tile_handles.len != 0;
    }

    pub fn writeFromPackedScalar(self: *Self, id: TensorId, packed_bytes: []const u8) StorageError!void {
        if (!(try self.tensorHasBacking(id))) try self.reserveHostBacking(id, try self.tensorLogicalBackingBytes(id));
        var t: *TiledTensor = try self.getMut(id);
        return t.writeFromPackedScalar(packed_bytes);
    }

    pub fn readToPackedScalar(self: *const Self, id: TensorId, out: []u8) StorageError!void {
        const t: *const TiledTensor = try self.getConst(id);
        return t.readToPackedScalar(out);
    }

    pub fn writeFromPackedQuant(self: *Self, id: TensorId, packed_bytes: []const u8) StorageError!void {
        if (!(try self.tensorHasBacking(id))) try self.reserveHostBacking(id, try self.tensorLogicalBackingBytes(id));
        var t: *TiledTensor = try self.getMut(id);
        return t.writeFromPackedQuant(packed_bytes);
    }

    pub fn readToPackedQuant(self: *const Self, id: TensorId, out: []u8) StorageError!void {
        const t: *const TiledTensor = try self.getConst(id);
        return t.readToPackedQuant(out);
    }

    /// Read `id` into the device-independent packed layout wherever it lives,
    /// staging D2H when it is device-resident. The `readToPacked*`/
    /// `writeFromPacked*` pair above stays host-only on purpose: they back
    /// `Tensor.read`/`write`, whose move semantics require `.to(.cpu)` first.
    /// This pair is for internals that must reach a value at its actual
    /// placement — weight write-through being the motivating case.
    ///
    /// "Wherever it lives" includes inside a derived weight: a weight a pass folded away
    /// keeps its metadata and loses its bytes, and a read of it has to find them anyway.
    /// The write direction is deliberately NOT symmetric — a write has to reach every
    /// copy, which is `writeDerivedSource`, not one buffer.
    pub fn readPackedAtPlacement(self: *Self, id: TensorId, out: []u8) StorageError!void {
        const t: *const TiledTensor = try self.getConst(id);
        if (!try self.tensorHasBacking(id)) {
            if (self.derivedLocate(id)) |at| return fold.readDerivedPacked(self, at, out);
        }
        return self.gatherPacked(t, out);
    }

    /// Write device-independent packed bytes into `id` at its actual placement,
    /// staging H2D when it is device-resident. See `readPackedAtPlacement`.
    pub fn writePackedAtPlacement(self: *Self, id: TensorId, packed_bytes: []const u8) StorageError!void {
        const t: *TiledTensor = try self.getMut(id);
        return self.scatterPackedResident(t, packed_bytes);
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
        const t: *const TiledTensor = try self.backingConst(id);
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

    /// Zero a tensor's contents at its current placement: host bytes are cleared
    /// directly; a device tensor gets zero bytes uploaded into each tile buffer.
    /// Used by `Model.resetState` so recurrent state clears correctly whether it
    /// lives on the host or exclusively on a device.
    pub fn zeroTensorData(self: *Self, id: TensorId) StorageError!void {
        const t: *TiledTensor = try self.getMut(id);
        if (t.device.kind == .cpu) {
            @memset(t.data, 0);
            return;
        }
        const d = t.dev orelse return StorageError.InvalidArgument;
        for (t.tile_handles, 0..) |h, i| try grow.zeroDeviceRange(self, d, h, 0, t.tile_lens[i]);
    }

    /// Materialize one compiler-planned workspace slot directly at its final
    /// placement. Compile-time contents are uploaded only for synthetic values
    /// that actually own initialized host bytes.
    pub fn materializeWorkspaceSlot(
        self: *Self,
        owner_id: TensorId,
        target: DeviceRef,
        dev: ?dm.DeviceMemory,
        tile_capacities: []const usize,
        host_bytes: usize,
        preserve_contents: bool,
    ) StorageError!void {
        const owner = try self.getMut(owner_id);
        if (owner.backing_owner != null) return StorageError.InvalidArgument;
        if (target.kind == .cpu) {
            try self.reserveHostBacking(owner_id, host_bytes);
            return;
        }
        if (owner.device.eql(target) and owner.tile_handles.len == tile_capacities.len) return;
        if (preserve_contents) {
            return self.moveTensor(owner_id, target, dev, owner.tile_shape, owner.tile_alignment);
        }

        owner.releaseData();
        const d = dev orelse return StorageError.InvalidArgument;
        const handles = self.allocator.alloc(dm.DeviceHandle, tile_capacities.len) catch return StorageError.OutOfMemory;
        var allocated: usize = 0;
        errdefer {
            for (handles[0..allocated]) |h| d.free(h);
            self.allocator.free(handles);
        }
        var total: usize = 0;
        while (allocated < tile_capacities.len) : (allocated += 1) {
            const cap = tile_capacities[allocated];
            handles[allocated] = d.alloc(cap, 64) catch return StorageError.OutOfMemory;
            total = std.math.add(usize, total, cap) catch return StorageError.InvalidArgument;
        }
        owner.device = target;
        owner.tile_handles = handles;
        owner.dev = d;
        owner.owns_data = false;
        owner.backing_bytes = total;
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

        // Fast path: host -> device with the tiling unchanged (what placement does
        // for every weight — it re-requests the tensor's own tile shape). The
        // general path below gathers into a packed buffer and scatters into a
        // re-tiled staging tensor, two full copies of the tensor plus two transient
        // allocations, before uploading a byte-identical result. Each tile is
        // already contiguous at `data[off..off+len]`, so upload straight from there.
        // `tile_alignment` only spaces tiles inside the host buffer and every device
        // tile is its own buffer sized `tile_lens[i]`, so it cannot change the result.
        if (target.kind != .cpu and t.device.kind == .cpu and t.backing_owner == null and
            t.owns_data and t.data.len != 0 and std.mem.eql(usize, t.tile_shape, target_tile_shape))
        {
            const d = dev orelse return StorageError.InvalidArgument;
            const n: usize = t.tile_offsets.len;
            const handles: []dm.DeviceHandle = self.allocator.alloc(dm.DeviceHandle, n) catch return StorageError.OutOfMemory;
            var uploaded: usize = 0;
            errdefer {
                var j: usize = 0;
                while (j < uploaded) : (j += 1) d.free(handles[j]);
                self.allocator.free(handles);
            }
            var device_bytes: usize = 0;
            while (uploaded < n) : (uploaded += 1) {
                const off = t.tile_offsets[uploaded];
                const len = t.tile_lens[uploaded];
                const h = d.alloc(len, 64) catch return StorageError.OutOfMemory;
                handles[uploaded] = h;
                d.copyH2D(h, 0, t.data[off .. off + len]) catch {
                    d.free(h);
                    return StorageError.InvalidArgument;
                };
                device_bytes = std.math.add(usize, device_bytes, len) catch return StorageError.InvalidArgument;
            }
            t.releaseData(); // host bytes are now on the device
            t.device = target;
            t.tile_handles = handles;
            t.dev = d;
            t.owns_data = false;
            t.backing_bytes = device_bytes;
            return;
        }

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
            var device_bytes: usize = 0;
            for (staging.tile_lens) |len| device_bytes += len;
            staging.backing_bytes = device_bytes;
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

    /// The store as the runtime's `TensorStore` interface — how an executor reaches
    /// tiles without depending on this layer (see `manager/store_view.zig`).
    pub fn tensorStore(self: *Self) tensor_store.TensorStore {
        return store_view.of(self);
    }
};
