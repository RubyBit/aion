// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const backend_mod = @import("../backend/backend.zig");
const cpu_backend_mod = @import("../backend/cpu/cpu_backend.zig");
const types = @import("../backend/types.zig");
const package_file = @import("../storage/aion_file.zig");
const cache_mod = @import("../storage/cache.zig");
const manager_mod = @import("../storage/manager.zig");
const quantize_mod = @import("../storage/quantize.zig");
const plan_mod = @import("../graph/plan.zig");
const graph_mod = @import("../graph/graph.zig");
const program_mod = @import("../graph/program.zig");
const infer_mod = @import("../graph/infer.zig");

const api_builder = @import("builder.zig");
const api_loaded_model = @import("loaded_model.zig");
const api_weights = @import("weights.zig");
const api_initializers = @import("loaded_model/initializers.zig");
const api_package_export = @import("package_export.zig");
const api_tensor = @import("tensor.zig");
const api_tiling = @import("tiling.zig");
const api_errors = @import("errors.zig");
const device_mod = @import("device.zig");
const build_options = @import("build_options");
/// GPU device bundle module — only reached (and thus only analyzed / linked
/// against wgpu) when the GPU feature is enabled. Empty struct otherwise.
const gpu_device_mod = if (build_options.enable_gpu) @import("gpu_device.zig") else struct {};

pub const DType = types.DType;
pub const TilePolicy = plan_mod.TilePolicy;
pub const Model = api_loaded_model.Model;
pub const LoadedModel = api_loaded_model.LoadedModel;
pub const Weights = api_weights.Weights;
pub const LoadModelOptions = api_loaded_model.LoadModelOptions;
pub const CacheOptions = api_loaded_model.CacheOptions;
pub const CacheGrowth = api_loaded_model.CacheGrowth;
pub const NamedTensorRef = api_builder.NamedTensorRef;
pub const DimSymbol = api_builder.DimSymbol;
pub const ExportMetadata = api_package_export.Metadata;
pub const OutputAlias = api_package_export.OutputAlias;
pub const InputRoleDecl = api_package_export.InputRoleDecl;
pub const InputRoleKind = package_file.InputRoleKind;
pub const ExportModelOptions = api_package_export.ExportModelOptions;
pub const CompileOptions = api_package_export.CompileOptions;
pub const CacheConfig = cache_mod.CacheConfig;
pub const CachePolicy = cache_mod.CachePolicy;
pub const SequenceCachePolicy = cache_mod.SequenceCachePolicy;
pub const GrowablePolicy = cache_mod.GrowablePolicy;
pub const RollingPolicy = cache_mod.RollingPolicy;
pub const DeviceSelector = device_mod.DeviceSelector;
pub const GpuOptions = device_mod.GpuOptions;

/// `[]*GpuDevice` when GPU is enabled, `void` otherwise (standard conditional-
/// compilation field pattern — mirrors `root.zig`'s gated `gpu` decl).
const GpuDeviceSlot = if (build_options.enable_gpu) []*gpu_device_mod.GpuDevice else void;

pub const Context = struct {
    allocator: std.mem.Allocator,

    /// The built-in CPU backend, constructed and owned by the Context.
    cpu: cpu_backend_mod.CpuBackend,

    store: manager_mod.StorageManager,
    policy: plan_mod.TilePolicy,

    /// Registered GPU devices (heap-pinned bundles), index-aligned with the
    /// `.gpu = i` selector. Empty when none requested; `void` on a CPU-only build.
    gpu_devices: GpuDeviceSlot = if (build_options.enable_gpu) &.{} else {},
    /// Heap-owned registry backing handed to `store.setDeviceRegistry` (borrowed
    /// by the store). One entry per GPU device; heap-stable across Context moves.
    device_entries: []manager_mod.StorageManager.DeviceEntry = &.{},

    const Self = @This();

    pub const Options = struct {
        /// Total threads including the calling thread.
        thread_count: usize = 1,

        /// Optional tiling policy override.
        ///
        /// If null, a default policy is used and users don't need to think about tiling.
        /// Power users can provide this to tune performance.
        tile_policy_override: ?plan_mod.TilePolicy = null,

        /// Optional runtime cache manager configuration.
        ///
        /// If null, storage behaves as plain RAM-backed tiled tensors.
        cache_config: ?cache_mod.CacheConfig = null,

        /// GPU devices to create, one per entry (index = the `.gpu = i` selector).
        /// Empty (default) → CPU-only, unchanged behavior. Requires `-Dgpu`; a
        /// missing adapter or a CPU-only build yields `error.BackendUnavailable`.
        gpus: []const device_mod.GpuOptions = &.{},
    };

    /// Construct a context. With no `opts.gpus` this is the CPU-only path,
    /// byte-identical to before. Requested GPU devices are created here (pinned).
    pub fn init(allocator: std.mem.Allocator, opts: Options) api_errors.InitError!Self {
        var cpu = try initCpuBackend(allocator, opts);
        errdefer cpu.deinit();
        var sm = try makeStore(allocator, opts);
        errdefer sm.deinit();

        var self: Self = .{
            .allocator = allocator,
            .cpu = cpu,
            .store = sm,
            .policy = opts.tile_policy_override orelse plan_mod.tilePolicyForTarget(.cpu),
            .gpu_devices = if (build_options.enable_gpu) &.{} else {},
            .device_entries = &.{},
        };
        // On failure past this point, the errdefers above free `cpu`/`sm` (which
        // `self` shares by value); `self` is discarded, never returned.
        if (opts.gpus.len != 0) try self.initGpuDevices(opts.gpus);
        return self;
    }

    /// Create the requested GPU device bundles and register them with the store so
    /// `Tensor.to` / `moveTensor` can migrate tensors. Gated: on a CPU-only build
    /// this returns `BackendUnavailable` (the GPU branch is comptime-pruned).
    fn initGpuDevices(self: *Self, gpus: []const device_mod.GpuOptions) api_errors.InitError!void {
        if (build_options.enable_gpu) {
            const n = gpus.len;
            const bundles = self.allocator.alloc(*gpu_device_mod.GpuDevice, n) catch return api_errors.InitError.OutOfMemory;
            errdefer self.allocator.free(bundles);
            const entries = self.allocator.alloc(manager_mod.StorageManager.DeviceEntry, n) catch return api_errors.InitError.OutOfMemory;
            errdefer self.allocator.free(entries);

            var created: usize = 0;
            errdefer {
                var j: usize = 0;
                while (j < created) : (j += 1) gpu_device_mod.destroy(self.allocator, bundles[j]);
            }
            while (created < n) : (created += 1) {
                const bundle = gpu_device_mod.create(self.allocator, gpus[created]) catch |e| return switch (e) {
                    error.OutOfMemory => api_errors.InitError.OutOfMemory,
                    error.BackendUnavailable => api_errors.InitError.BackendUnavailable,
                };
                bundles[created] = bundle;
                entries[created] = .{ .mem = bundle.deviceMemory(), .policy = bundle.policy };
            }

            self.gpu_devices = bundles;
            self.device_entries = entries;
            self.store.setDeviceRegistry(self.policy, entries);
        } else {
            return api_errors.InitError.BackendUnavailable;
        }
    }

    /// Resolve a device selector to a concrete backend + tile policy (+ device
    /// memory for GPU). Computed on demand — never cached on the Context.
    fn resolveDevice(self: *Self, sel: device_mod.DeviceSelector) error{InvalidArgument}!device_mod.Device {
        switch (sel) {
            .cpu => return .{ .ref = .{ .kind = .cpu }, .backend = self.backend(), .policy = self.policy, .device_memory = null },
            .gpu => |idx| {
                if (build_options.enable_gpu) {
                    if (idx >= self.gpu_devices.len or idx > 255) return error.InvalidArgument;
                    const b = self.gpu_devices[idx];
                    return .{
                        .ref = .{ .kind = .gpu, .index = @intCast(idx) },
                        .backend = b.backend.backend(),
                        .policy = b.policy,
                        .device_memory = b.deviceMemory(),
                    };
                } else return error.InvalidArgument;
            },
        }
    }

    /// `init` alias kept for callers/tests that name the CPU path explicitly.
    pub fn initCpu(allocator: std.mem.Allocator, opts: Options) api_errors.InitError!Self {
        return init(allocator, opts);
    }

    fn makeStore(allocator: std.mem.Allocator, opts: Options) api_errors.InitError!manager_mod.StorageManager {
        var sm: manager_mod.StorageManager = manager_mod.StorageManager.init(allocator);
        errdefer sm.deinit();
        if (opts.cache_config) |cfg| {
            sm.configureCache(cfg) catch |e| {
                return switch (e) {
                    error.OutOfMemory => api_errors.InitError.OutOfMemory,
                    else => api_errors.InitError.InvalidArgument,
                };
            };
        }
        return sm;
    }

    fn initCpuBackend(allocator: std.mem.Allocator, opts: Options) api_errors.InitError!cpu_backend_mod.CpuBackend {
        return try cpu_backend_mod.CpuBackend.initWithOptions(allocator, .{ .thread_count = opts.thread_count });
    }

    pub fn deinit(self: *Self) void {
        // Free the store first: device-resident tensors release their buffers
        // through the (still-alive) GPU device memory. Only then tear down bundles.
        self.store.deinit();
        if (build_options.enable_gpu) {
            for (self.gpu_devices) |b| gpu_device_mod.destroy(self.allocator, b);
            if (self.gpu_devices.len != 0) self.allocator.free(self.gpu_devices);
        }
        if (self.device_entries.len != 0) self.allocator.free(self.device_entries);
        self.cpu.deinit();
        self.* = undefined;
    }

    pub fn storage(self: *Self) *manager_mod.StorageManager {
        return &self.store;
    }

    pub fn configureCache(self: *Self, cfg: cache_mod.CacheConfig) api_errors.ApiError!void {
        try self.store.configureCache(cfg);
    }

    pub fn setTensorSequenceCachePolicy(self: *Self, t: api_tensor.Tensor, policy: cache_mod.SequenceCachePolicy) api_errors.ApiError!void {
        if (t.store != &self.store) return api_errors.ApiError.InvalidArgument;
        try self.store.registerSequenceCachePolicy(t.id, policy);
    }

    pub fn tilePolicy(self: *const Self) plan_mod.TilePolicy {
        return self.policy;
    }

    /// Return a backend handle bound to this context's CPU backend.
    ///
    /// Important: this is computed on-demand to avoid stale internal pointers
    /// if a `Context` value is moved.
    pub fn backend(self: *Self) backend_mod.Backend {
        return self.cpu.backend();
    }

    /// A `Builder` bound to this context. The context must outlive the builder and
    /// must not be moved while it is alive (the builder holds a pointer to it).
    pub fn builder(self: *Self) api_builder.Builder {
        return api_builder.Builder.init(self);
    }

    pub fn exportModel(
        self: *Self,
        file: std.Io.File,
        bld: *api_builder.Builder,
        outputs: []const NamedTensorRef,
        opts: ExportModelOptions,
    ) api_errors.LoadError!void {
        var pkg = try api_package_export.buildPackage(self.allocator, &self.store, bld, outputs, opts);
        defer pkg.deinit();
        return package_file.writeFile(file, &pkg);
    }

    pub fn exportModelPath(
        self: *Self,
        path: []const u8,
        bld: *api_builder.Builder,
        outputs: []const NamedTensorRef,
        opts: ExportModelOptions,
    ) api_errors.LoadError!void {
        var io_backend: std.Io.Threaded = .init_single_threaded;
        const io = io_backend.io();
        const file = std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true }) catch return package_file.PackageError.IoFailure;
        defer file.close(io);
        return self.exportModel(file, bld, outputs, opts);
    }

    pub fn exportModelPathAbsolute(
        self: *Self,
        absolute_path: []const u8,
        bld: *api_builder.Builder,
        outputs: []const NamedTensorRef,
        opts: ExportModelOptions,
    ) api_errors.LoadError!void {
        var io_backend: std.Io.Threaded = .init_single_threaded;
        const io = io_backend.io();
        const file = std.Io.Dir.createFileAbsolute(io, absolute_path, .{ .truncate = true }) catch return package_file.PackageError.IoFailure;
        defer file.close(io);
        return self.exportModel(file, bld, outputs, opts);
    }

    pub fn loadModel(self: *Self, file: std.Io.File, opts: LoadModelOptions) api_errors.LoadError!LoadedModel {
        // Resolve the target device: its backend runs the model and its tile policy
        // shapes both the imported weights and the per-shape JIT compiles.
        const dev = try self.resolveDevice(opts.device);
        // `parseTakeOwned` transfers ownership of `bytes` into the returned Package
        // (Initializer.data slices borrow into it). `pkg.deinit` frees the buffer.
        const bytes = try package_file.readAlloc(self.allocator, file);
        // `parseTakeOwned` consumes `bytes` unconditionally (frees on error,
        // transfers ownership to `pkg` on success), so we don't guard `bytes` with
        // its own errdefer — that would double-free with `pkg.deinit()` / the
        // streaming import's `releaseSourceBytes()`. Nothing between here and the
        // call below can fail.
        const hash = std.hash.Wyhash.hash(0, bytes);
        var pkg = try package_file.parseTakeOwned(self.allocator, bytes);
        errdefer pkg.deinit();
        // Stream weights into the store one initializer at a time, reading each back
        // from the file and releasing the whole-file buffer up front. This keeps peak
        // RSS at ~1x the weight size instead of ~2x (the file image and the populated
        // store never coexist). `importInitializersStreaming` consumes `bytes` via
        // `pkg.releaseSourceBytes()`, so the `bytes` errdefer above is now a no-op
        // (frees an emptied buffer) and `pkg.deinit()` owns any later teardown.
        var params = try api_initializers.importParamsStreaming(self.allocator, &self.store, dev.policy, &pkg, file, bytes);
        errdefer params.deinit(self.allocator);
        return api_loaded_model.LoadedModel.init(self.allocator, dev.backend, &self.store, dev.target(opts.passes), .{ .package = pkg }, params, hash, opts);
    }

    /// Load an AION package as a weights-only container.
    ///
    /// This parses the package and imports initializer tensors into the current
    /// context's storage, but does not instantiate or run the package graph.
    pub fn loadWeights(self: *Self, file: std.Io.File, _: LoadModelOptions) api_errors.LoadError!Weights {
        const bytes = try package_file.readAlloc(self.allocator, file);
        errdefer self.allocator.free(bytes);
        const hash = std.hash.Wyhash.hash(0, bytes);
        var pkg = try package_file.parseTakeOwned(self.allocator, bytes);
        errdefer pkg.deinit();
        var params = try api_initializers.importParams(self.allocator, &self.store, self.policy, &pkg);
        errdefer params.deinit(self.allocator);
        return api_weights.Weights.initLoaded(self.allocator, &self.store, self.policy, pkg, params, hash);
    }

    pub fn loadModelPath(self: *Self, path: []const u8, opts: LoadModelOptions) api_errors.LoadError!LoadedModel {
        var io_backend: std.Io.Threaded = .init_single_threaded;
        const io = io_backend.io();
        const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return package_file.PackageError.IoFailure;
        defer file.close(io);
        return self.loadModel(file, opts);
    }

    pub fn loadWeightsPath(self: *Self, path: []const u8, opts: LoadModelOptions) api_errors.LoadError!Weights {
        var io_backend: std.Io.Threaded = .init_single_threaded;
        const io = io_backend.io();
        const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return package_file.PackageError.IoFailure;
        defer file.close(io);
        return self.loadWeights(file, opts);
    }

    pub fn loadModelPathAbsolute(self: *Self, absolute_path: []const u8, opts: LoadModelOptions) api_errors.LoadError!LoadedModel {
        var io_backend: std.Io.Threaded = .init_single_threaded;
        const io = io_backend.io();
        const file = std.Io.Dir.openFileAbsolute(io, absolute_path, .{}) catch return package_file.PackageError.IoFailure;
        defer file.close(io);
        return self.loadModel(file, opts);
    }

    pub fn loadWeightsPathAbsolute(self: *Self, absolute_path: []const u8, opts: LoadModelOptions) api_errors.LoadError!Weights {
        var io_backend: std.Io.Threaded = .init_single_threaded;
        const io = io_backend.io();
        const file = std.Io.Dir.openFileAbsolute(io, absolute_path, .{}) catch return package_file.PackageError.IoFailure;
        defer file.close(io);
        return self.loadWeights(file, opts);
    }

    /// Create a new owned tensor with a default tile shape.
    pub fn tensor(self: *Self, dtype: DType, shape: []const usize) api_errors.ApiError!api_tensor.Tensor {
        if (shape.len == 0 or shape.len > api_tiling.MAX_RANK) return api_errors.ApiError.InvalidArgument;

        var tile_mem: [api_tiling.MAX_RANK]usize = undefined;
        const tile_slice: []usize = tile_mem[0..shape.len];
        try api_tiling.fillDefaultTileShape(self.policy, dtype, shape, tile_slice);

        const tid: manager_mod.TensorId = try self.store.createTiledTensor(dtype, shape, tile_slice, .{ .tile_alignment = self.policy.tile_alignment });
        const t = try self.store.getConst(tid);
        return .{ .store = &self.store, .id = tid, .dtype = t.dtype, .shape = t.shape };
    }

    /// Create a new tensor with an explicit tile shape.
    pub fn tensorTiled(self: *Self, dtype: DType, shape: []const usize, tile_shape: []const usize) api_errors.ApiError!api_tensor.Tensor {
        const tid: manager_mod.TensorId = try self.store.createTiledTensor(dtype, shape, tile_shape, .{ .tile_alignment = self.policy.tile_alignment });
        const t = try self.store.getConst(tid);
        return .{ .store = &self.store, .id = tid, .dtype = t.dtype, .shape = t.shape };
    }

    /// Convenience: allocate and initialize from packed scalar bytes.
    pub fn fromPackedScalar(self: *Self, dtype: DType, shape: []const usize, packed_bytes: []const u8) api_errors.ApiError!api_tensor.Tensor {
        var t: api_tensor.Tensor = try self.tensor(dtype, shape);
        try t.writePackedScalar(packed_bytes);
        return t;
    }

    /// Convenience: allocate and initialize from packed quant bytes, blocking
    /// along `quant_axis`.
    ///
    /// Tiling comes from the shared chooser, so an authored weight lands on the same
    /// tiling the loader gives the identical weight. It cannot be left to the
    /// compiler: quantized tensors are the one thing `ensureTilingMaybeRetile`
    /// refuses to re-tile, so a wrong tiling here is a compile error at first use.
    pub fn fromPackedQuant(self: *Self, dtype: DType, shape: []const usize, quant_axis: usize, packed_bytes: []const u8) api_errors.ApiError!api_tensor.Tensor {
        if (!dtype.info().is_quantized) return api_errors.ApiError.InvalidArgument;
        if (quant_axis >= shape.len) return api_errors.ApiError.InvalidArgument;
        if (shape.len > api_tiling.MAX_RANK) return api_errors.ApiError.InvalidArgument;

        var tile_mem: [api_tiling.MAX_RANK]usize = undefined;
        const tile_shape = tile_mem[0..shape.len];
        api_tiling.chooseTileShapeForTensor(self.policy, dtype, shape, @intCast(quant_axis), tile_shape) catch
            return api_errors.ApiError.InvalidArgument;

        const tid: manager_mod.TensorId = try self.store.createTiledTensor(
            dtype,
            shape,
            tile_shape,
            .{ .tile_alignment = self.policy.tile_alignment, .quant_axis = @intCast(quant_axis) },
        );
        const ct = try self.store.getConst(tid);
        var t0: api_tensor.Tensor = .{ .store = &self.store, .id = tid, .dtype = ct.dtype, .shape = ct.shape };
        try t0.writePackedQuant(packed_bytes);
        return t0;
    }

    /// Author a block-quantized tensor from row-major f32 `values`, blocking
    /// along `quant_axis` (the matmul-B K axis is rank-2; an embedding table
    /// blocked along its feature dim uses the last axis).
    pub fn fromF32Quantized(self: *Self, dtype: DType, shape: []const usize, quant_axis: usize, values: []const f32) api_errors.ApiError!api_tensor.Tensor {
        if (!dtype.info().is_quantized) return api_errors.ApiError.InvalidArgument;
        const packed_bytes: []u8 = quantize_mod.quantizeF32(self.allocator, dtype, shape, quant_axis, values) catch |e| return switch (e) {
            error.OutOfMemory => api_errors.ApiError.OutOfMemory,
            error.Unsupported => api_errors.ApiError.UnsupportedFeature,
            error.InvalidArgument => api_errors.ApiError.InvalidArgument,
        };
        defer self.allocator.free(packed_bytes);
        return self.fromPackedQuant(dtype, shape, quant_axis, packed_bytes);
    }

    /// Convenience: allocate and initialize an f32 tensor from typed values.
    pub fn fromF32(self: *Self, shape: []const usize, values: []const f32) api_errors.ApiError!api_tensor.Tensor {
        return self.from(shape, values);
    }

    /// Convenience: allocate and initialize an f16 tensor from typed values.
    pub fn fromF16(self: *Self, shape: []const usize, values: []const f16) api_errors.ApiError!api_tensor.Tensor {
        return self.from(shape, values);
    }

    /// Convenience: allocate and initialize a scalar tensor (shape `{1}`).
    pub fn scalar(self: *Self, comptime T: type, value: T) api_errors.ApiError!api_tensor.Tensor {
        const shape: [1]usize = .{1};
        var tmp: [1]T = .{value};
        return self.from(shape[0..1], tmp[0..1]);
    }

    fn elemTypeOfValues(comptime V: type) ?type {
        // []T / []const T
        if (api_tensor.Tensor.sliceElemType(V)) |Elem| return Elem;

        // *[N]T
        const pinfo = switch (@typeInfo(V)) {
            .pointer => |p| p,
            else => return null,
        };
        if (pinfo.size != .one) return null;
        return switch (@typeInfo(pinfo.child)) {
            .array => |a| a.child,
            else => null,
        };
    }

    fn valuesLen(values: anytype) ?usize {
        const V: type = @TypeOf(values);
        return switch (@typeInfo(V)) {
            .pointer => |p| switch (p.size) {
                .slice => values.len,
                .one => switch (@typeInfo(p.child)) {
                    .array => |a| @as(usize, @intCast(a.len)),
                    else => null,
                },
                else => null,
            },
            .array => |a| @as(usize, @intCast(a.len)),
            else => null,
        };
    }

    /// Convenience: allocate a 1D tensor whose length is inferred from `values`.
    pub fn vector(self: *Self, values: anytype) api_errors.ApiError!api_tensor.Tensor {
        const n_opt: ?usize = valuesLen(values);
        if (n_opt == null) return api_errors.ApiError.InvalidArgument;
        const n: usize = n_opt.?;
        const shape: [1]usize = .{n};
        return self.from(shape[0..1], values);
    }

    /// Convenience: allocate a 2D tensor and fill it from a flat value buffer.
    pub fn matrix(self: *Self, rows: usize, cols: usize, values: anytype) api_errors.ApiError!api_tensor.Tensor {
        const shape: [2]usize = .{ rows, cols };
        return self.from(shape[0..2], values);
    }

    /// Convenience: allocate and initialize a tensor from typed values.
    ///
    /// The tensor dtype is inferred from the element type of `values`.
    pub fn from(self: *Self, shape: []const usize, values: anytype) api_errors.ApiError!api_tensor.Tensor {
        const ElemOpt: ?type = elemTypeOfValues(@TypeOf(values));
        if (ElemOpt == null) return api_errors.ApiError.InvalidArgument;
        const Elem: type = ElemOpt.?;
        const dt_opt: ?DType = api_tensor.Tensor.dtypeOf(Elem);
        if (dt_opt == null) return api_errors.ApiError.InvalidArgument;
        const dt: DType = dt_opt.?;

        var t: api_tensor.Tensor = try self.tensor(dt, shape);
        try t.write(values);
        return t;
    }

    /// Convenience: allocate and initialize a tensor from a (possibly nested) Zig array.
    ///
    /// Example:
    /// - `ctx.fromArray([2][3]f32{ ... })` infers shape `{2,3}` and dtype `f32`.
    ///
    /// This uses packed scalar I/O (no intermediate flattening allocations).
    pub fn fromArray(self: *Self, arr: anytype) api_errors.ApiError!api_tensor.Tensor {
        const ArrT: type = @TypeOf(arr);

        const base_arr_t: type = switch (@typeInfo(ArrT)) {
            .array => ArrT,
            .pointer => |p| switch (p.size) {
                .one => p.child,
                else => return api_errors.ApiError.InvalidArgument,
            },
            else => return api_errors.ApiError.InvalidArgument,
        };

        switch (@typeInfo(base_arr_t)) {
            .array => {},
            else => return api_errors.ApiError.InvalidArgument,
        }

        const ShapeInfo = comptime blk: {
            var dims: [api_tiling.MAX_RANK]usize = @splat(0);
            var rank: usize = 0;
            var cur: type = base_arr_t;
            while (true) {
                switch (@typeInfo(cur)) {
                    .array => |a| {
                        if (rank >= api_tiling.MAX_RANK) break;
                        dims[rank] = @as(usize, @intCast(a.len));
                        rank += 1;
                        cur = a.child;
                    },
                    else => break,
                }
            }
            const dt_opt: ?DType = api_tensor.Tensor.dtypeOf(cur);
            const dims_out: [api_tiling.MAX_RANK]usize = dims;
            break :blk .{ .rank = rank, .dims = dims_out, .dt_opt = dt_opt };
        };

        if (ShapeInfo.rank == 0 or ShapeInfo.rank > api_tiling.MAX_RANK) return api_errors.ApiError.InvalidArgument;
        if (ShapeInfo.dt_opt == null) return api_errors.ApiError.InvalidArgument;
        const dt: DType = ShapeInfo.dt_opt.?;

        var shape_mem: [api_tiling.MAX_RANK]usize = undefined;
        var i: usize = 0;
        while (i < ShapeInfo.rank) : (i += 1) {
            shape_mem[i] = ShapeInfo.dims[i];
        }
        const shape: []const usize = shape_mem[0..ShapeInfo.rank];

        const packed_bytes: []const u8 = switch (@typeInfo(ArrT)) {
            .array => std.mem.asBytes(&arr),
            .pointer => std.mem.asBytes(arr),
            else => unreachable,
        };

        return self.fromPackedScalar(dt, shape, packed_bytes);
    }

    /// Compile a builder into a runnable `Model` (in-process graph source) — the
    /// in-process analogue of `exportModel`.
    ///
    /// Outputs are bare `TensorRef`s: each output's *name* is the one already attached
    /// to the value by the builder (`bld.name(ref, "...")`, else the op's auto-name),
    /// so there is no need to wrap them in `NamedTensorRef`. Access outputs by name via
    /// `outputTensor` or by position via `outputTensorAt`/`runOutputTensor`. Name an
    /// output explicitly when you need to reference it (e.g. in `opts.output_aliases`).
    ///
    /// `opts.output_aliases` declares io-alias recurrent-state carry (by name).
    /// Variable (symbolic) input dims declared on the builder via `symbolicDim`
    /// are honored: the compiled model serves any size on those axes (re-lowering
    /// per distinct shape), so one compile covers e.g. prefill then decode.
    /// Compile a builder graph into an in-process model on the CPU (default device).
    pub fn compile(
        self: *Self,
        b: *api_builder.Builder,
        outputs: []const api_builder.TensorRef,
        opts: CompileOptions,
    ) api_errors.ApiError!Model {
        return self.compileOn(.cpu, b, outputs, opts);
    }

    /// Like `compile`, but targets `dev_sel` (e.g. `.{ .gpu = 0 }`). The model's
    /// backend + tile policy follow the device; auto-allocated inputs/outputs and
    /// the per-shape JIT are tiled for it.
    pub fn compileOn(
        self: *Self,
        dev_sel: device_mod.DeviceSelector,
        b: *api_builder.Builder,
        outputs: []const api_builder.TensorRef,
        opts: CompileOptions,
    ) api_errors.ApiError!Model {
        if (outputs.len == 0) return api_errors.ApiError.InvalidArgument;

        const dev = try self.resolveDevice(dev_sel);

        const g: *graph_mod.Graph = b.innerGraph();
        try infer_mod.infer(g);

        // Outputs are named by whatever the builder attached to the value (an explicit
        // `bld.name` or the op's auto-name), falling back to a position.
        var scratch = std.heap.ArenaAllocator.init(self.allocator);
        defer scratch.deinit();
        const sa = scratch.allocator();
        const named = sa.alloc(api_package_export.NamedTensorRef, outputs.len) catch return error.OutOfMemory;
        for (outputs, 0..) |o, i| {
            named[i] = .{
                .name = b.valueName(o) orelse (std.fmt.allocPrint(sa, "output{d}", .{i}) catch return error.OutOfMemory),
                .tensor = o,
            };
        }

        // Snapshot the authored graph in symbolic form. This is the same conversion
        // `exportModel` runs; the difference is that nothing serializes the weights, so
        // the parameters stay the caller's tensors in the store.
        var parts = api_package_export.collectTemplate(self.allocator, b, named, .{}) catch |e| {
            return switch (e) {
                error.OutOfMemory => api_errors.ApiError.OutOfMemory,
                else => api_errors.ApiError.InvalidArgument,
            };
        };
        errdefer parts.deinit(self.allocator);

        // Aliases and roles address inputs and outputs by INDEX, so they are resolved
        // against the template's own lists.
        parts.io_aliases = try self.allocator.alloc(package_file.IoAlias, opts.output_aliases.len);
        for (opts.output_aliases, 0..) |al, i| {
            const in_idx = indexOfValue(parts.input_values, al.input.value) orelse return api_errors.ApiError.InvalidArgument;
            const out_idx = indexOfValue(parts.output_values, al.output.value) orelse return api_errors.ApiError.InvalidArgument;
            parts.io_aliases[i] = .{ .input = @intCast(in_idx), .output = @intCast(out_idx) };
        }

        parts.input_roles = try self.allocator.alloc(package_file.InputRole, opts.input_roles.len);
        for (opts.input_roles, 0..) |decl, i| {
            // A capacity symbol names a free dim of a cache the RUNTIME sizes; an
            // in-process compile has no such thing to point at.
            if (decl.capacity_symbol != null) return api_errors.ApiError.InvalidArgument;
            const in_idx = indexOfValue(parts.input_values, decl.input.value) orelse return api_errors.ApiError.InvalidArgument;
            var flags: u8 = 0;
            if (decl.zero_init) flags |= package_file.InputRoleFlags.zero_init;
            if (decl.allow_growable) flags |= package_file.InputRoleFlags.allow_growable;
            parts.input_roles[i] = .{
                .input = @intCast(in_idx),
                .kind = decl.kind,
                .axis = decl.axis orelse package_file.input_role_no_axis,
                .flags = flags,
                .retained_history_tokens = decl.retained_history_tokens,
            };
        }

        return api_loaded_model.Model.init(
            self.allocator,
            dev.backend,
            &self.store,
            dev.target(opts.passes),
            .{ .parts = parts },
            try paramsFromTemplate(self.allocator, parts),
            0,
            .{},
        );
    }
};

/// A parameter map keyed by graph value, from the template's snapshot of the bindings.
fn paramsFromTemplate(
    allocator: std.mem.Allocator,
    parts: api_package_export.Parts,
) api_errors.ApiError!api_loaded_model.Params {
    var out = try api_loaded_model.Params.init(allocator, parts.params.len);
    errdefer out.deinit(allocator);
    for (parts.params, 0..) |tid, idx| {
        if (tid == api_loaded_model.no_param) continue;
        out.set(@intCast(idx), tid);
    }
    return out;
}

fn indexOfValue(values: []const u32, value: u32) ?usize {
    for (values, 0..) |v, i| if (v == value) return i;
    return null;
}
