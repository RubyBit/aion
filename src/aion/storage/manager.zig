// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const storage_mod = @import("storage.zig");
const cache_mod = @import("cache.zig");
const types = @import("../backend/types.zig");
const tensor_store = @import("../runtime/tensor_store.zig");
const dm = @import("../runtime/device_memory.zig");
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

    /// Zero `bytes` of `handle` from `offset`, in bounded chunks so zeroing a
    /// large buffer never needs an equally large host staging allocation.
    fn zeroDeviceRange(self: *Self, dev: dm.DeviceMemory, handle: dm.DeviceHandle, offset: usize, bytes: usize) StorageError!void {
        if (bytes == 0) return;
        const chunk_cap: usize = @min(bytes, 4 << 20);
        const zeros: []u8 = self.allocator.alloc(u8, chunk_cap) catch return StorageError.OutOfMemory;
        defer self.allocator.free(zeros);
        @memset(zeros, 0);

        var done: usize = 0;
        while (done < bytes) {
            const n = @min(chunk_cap, bytes - done);
            dev.copyH2D(handle, offset + done, zeros[0..n]) catch return StorageError.InvalidArgument;
            done += n;
        }
    }

    /// Grow a device-resident tensor without leaving the device: allocate the
    /// new backing, zero it, and move the old contents across with `copyD2D`.
    ///
    /// Applies to the single, unpadded tile that role-declared sequence caches
    /// use — there the two backings differ only in the stride of `axis`, so the
    /// move is `outer` contiguous runs. Returns false for any other layout,
    /// leaving the caller its host round-trip.
    fn growAxisOnDevice(self: *Self, id: TensorId, axis: usize, new_size: usize, dev: dm.DeviceMemory) StorageError!bool {
        const t: *TiledTensor = try self.getMut(id);
        if (t.dtype.info().is_quantized) return false;
        if (t.tile_handles.len != 1 or !std.mem.eql(usize, t.tile_shape, t.shape)) return false;

        const rank: usize = @intCast(t.rank);
        var new_shape_mem: [8]usize = undefined;
        @memcpy(new_shape_mem[0..rank], t.shape);
        new_shape_mem[axis] = new_size;
        const new_shape: []const usize = new_shape_mem[0..rank];

        // Bytes spanned by one index of `axis`, and the runs on each side of it.
        var trailing: usize = t.dtype.info().block_bytes;
        for (t.shape[axis + 1 ..]) |d| trailing = std.math.mul(usize, trailing, d) catch return StorageError.InvalidArgument;
        var outer: usize = 1;
        for (t.shape[0..axis]) |d| outer = std.math.mul(usize, outer, d) catch return StorageError.InvalidArgument;
        const old_run = std.math.mul(usize, t.shape[axis], trailing) catch return StorageError.InvalidArgument;
        const new_run = std.math.mul(usize, new_size, trailing) catch return StorageError.InvalidArgument;
        // Device copies move whole 4-byte words; odd runs (f16) take the host path.
        if (old_run % 4 != 0 or new_run % 4 != 0) return false;

        // Built host-backed for its geometry, then immediately released: the
        // bytes live on the device, same handoff `moveTensor` performs.
        var staging: TiledTensor = undefined;
        try staging.init(self.allocator, t.dtype, new_shape, new_shape, .{ .tile_alignment = t.tile_alignment });
        errdefer staging.deinit();
        const total = staging.tile_lens[0];
        staging.releaseData();

        const handle = dev.alloc(total, 64) catch return StorageError.OutOfMemory;
        errdefer dev.free(handle);
        try self.zeroDeviceRange(dev, handle, 0, total);
        var i: usize = 0;
        while (i < outer) : (i += 1) {
            dev.copyD2D(handle, i * new_run, t.tile_handles[0], i * old_run, old_run) catch return StorageError.InvalidArgument;
        }

        const handles: []dm.DeviceHandle = self.allocator.alloc(dm.DeviceHandle, 1) catch return StorageError.OutOfMemory;
        handles[0] = handle;
        staging.device = t.device;
        staging.tile_handles = handles;
        staging.dev = dev;
        staging.owns_data = false;

        // Frees the old device buffer, which the submitted copy above keeps
        // alive until it retires.
        t.deinit();
        t.* = staging;
        t.shape = t.shape_storage.constSlice();
        t.tile_shape = t.tile_shape_storage.constSlice();
        t.tile_counts = t.tile_counts_storage.constSlice();
        t.tile_strides = t.tile_strides_storage.constSlice();
        return true;
    }

    pub fn ensureTensorAxisCapacity(self: *Self, id: TensorId, axis: usize, min_size: usize) StorageError!void {
        const t0: *const TiledTensor = try self.getConst(id);
        if (axis >= @as(usize, t0.rank)) return StorageError.InvalidArgument;
        if (t0.shape[axis] >= min_size) return;
        if (t0.device.kind == .cpu) {
            return (try self.getMut(id)).growAxisPreserveScalar(axis, min_size);
        }

        const target: DeviceRef = t0.device;
        if (t0.dev) |d| {
            if (try self.growAxisOnDevice(id, axis, min_size, d)) return;
        }

        // Layouts device growth does not cover round-trip through the host (D2H
        // gather -> host grow -> H2D re-migrate), reusing `moveTensor`. Growth is
        // geometric, so this amortizes to O(final size) and never touches the
        // fixed/ring fast path. `moveTensor` frees the old device buffer only
        // after the re-upload, and the tensor id is preserved.
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

    fn backingConst(self: *const Self, id: TensorId) StorageError!*const TiledTensor {
        return self.getConst(try self.backingId(id));
    }

    fn backingMut(self: *Self, id: TensorId) StorageError!*TiledTensor {
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

    /// Place every cached derived weight involving `source` on `target`.
    pub fn moveDerivedWeightsForSource(self: *Self, source: TensorId, target: DeviceRef) StorageError!void {
        if (target.kind == .cpu) return;
        const dev = self.deviceMemoryFor(target) orelse return StorageError.InvalidArgument;
        const policy = self.policyFor(target);
        for (self.derived_weight_cache.items) |entry| {
            var contains = false;
            for (entry.sources) |id| {
                if (id == source) {
                    contains = true;
                    break;
                }
            }
            if (!contains) continue;
            const t = try self.getConst(entry.result);
            if (t.device.eql(target)) continue;
            var tile_shape: [8]usize = undefined;
            const rank: usize = @intCast(t.rank);
            @memcpy(tile_shape[0..rank], t.tile_shape);
            try self.moveTensor(entry.result, target, dev, tile_shape[0..rank], policy.tile_alignment);
        }
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
    pub fn readPackedAtPlacement(self: *Self, id: TensorId, out: []u8) StorageError!void {
        const t: *const TiledTensor = try self.getConst(id);
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
        for (t.tile_handles, 0..) |h, i| try self.zeroDeviceRange(d, h, 0, t.tile_lens[i]);
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
                                target = growTarget(target, g.growth_numerator, g.growth_denominator) catch return tensor_store.StoreError.InvalidArgument;
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
