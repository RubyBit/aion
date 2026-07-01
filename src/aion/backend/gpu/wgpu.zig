// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Thin, use-driven ergonomics over the translate-c'd wgpu.h (standard webgpu.h
//! + wgpu-native extensions). NOT a general WebGPU abstraction — it collapses
//! the three genuine footguns of the raw C API (callback+pump async,
//! status->error, StringView), owns the instance/adapter/device/queue lifetime,
//! and handles adapter (GPU) selection. Raw handles stay reachable via `wgpu.c`.

const std = @import("std");

/// The translate-c'd wgpu.h, re-exported so callers drop to raw C any time.
pub const c = @import("wgpu");

pub const Error = error{
    NoInstance,
    NoAdapter,
    NoDevice,
    AdapterTimeout,
    DeviceTimeout,
    MapTimeout,
    MapFailed,
    BufferCreate,
    BadAdapterIndex,
};

/// Which GPU to prefer when several are present (e.g. iGPU + discrete).
pub const Power = enum { default, low, high };

/// Force a particular graphics backend, or let wgpu choose.
pub const Backend = enum { any, vulkan, d3d12, metal, gl };

/// GPU selection. `adapter_index` (from `listAdapters`) overrides power/backend.
pub const Options = struct {
    power: Power = .default,
    backend: Backend = .any,
    adapter_index: ?usize = null,
};

fn powerPref(p: Power) c.WGPUPowerPreference {
    return switch (p) {
        .default => c.WGPUPowerPreference_Undefined,
        .low => c.WGPUPowerPreference_LowPower,
        .high => c.WGPUPowerPreference_HighPerformance,
    };
}

fn backendType(b: Backend) c.WGPUBackendType {
    return switch (b) {
        .any => c.WGPUBackendType_Undefined,
        .vulkan => c.WGPUBackendType_Vulkan,
        .d3d12 => c.WGPUBackendType_D3D12,
        .metal => c.WGPUBackendType_Metal,
        .gl => c.WGPUBackendType_OpenGL,
    };
}

/// WGPUStringView over a Zig slice.
pub fn strv(s: []const u8) c.WGPUStringView {
    return .{ .data = s.ptr, .length = s.len };
}

/// Zig slice over a WGPUStringView (empty if null).
pub fn fromStrv(s: c.WGPUStringView) []const u8 {
    if (s.data == null or s.length == 0) return "";
    return s.data[0..s.length];
}

fn pumpUntil(instance: c.WGPUInstance, done: *const bool) bool {
    var i: usize = 0;
    while (!done.*) : (i += 1) {
        if (i > 1_000_000) return false;
        c.wgpuInstanceProcessEvents(instance);
    }
    return true;
}

const AdapterReq = struct {
    done: bool = false,
    status: c.WGPURequestAdapterStatus = 0,
    adapter: c.WGPUAdapter = null,
};

fn onAdapter(status: c.WGPURequestAdapterStatus, adapter: c.WGPUAdapter, message: c.WGPUStringView, ud1: ?*anyopaque, ud2: ?*anyopaque) callconv(.c) void {
    _ = message;
    _ = ud2;
    const r: *AdapterReq = @ptrCast(@alignCast(ud1.?));
    r.status = status;
    r.adapter = adapter;
    r.done = true;
}

const DeviceReq = struct {
    done: bool = false,
    status: c.WGPURequestDeviceStatus = 0,
    device: c.WGPUDevice = null,
};

fn onDevice(status: c.WGPURequestDeviceStatus, device: c.WGPUDevice, message: c.WGPUStringView, ud1: ?*anyopaque, ud2: ?*anyopaque) callconv(.c) void {
    _ = message;
    _ = ud2;
    const r: *DeviceReq = @ptrCast(@alignCast(ud1.?));
    r.status = status;
    r.device = device;
    r.done = true;
}

const MapReq = struct {
    done: bool = false,
    status: c.WGPUMapAsyncStatus = 0,
};

fn onMap(status: c.WGPUMapAsyncStatus, message: c.WGPUStringView, ud1: ?*anyopaque, ud2: ?*anyopaque) callconv(.c) void {
    _ = message;
    _ = ud2;
    const r: *MapReq = @ptrCast(@alignCast(ud1.?));
    r.status = status;
    r.done = true;
}

/// One enumerated GPU's identity (for selection / display).
pub const AdapterDesc = struct {
    name: [128]u8 = undefined,
    name_len: usize = 0,
    backend: c.WGPUBackendType = 0,
    kind: c.WGPUAdapterType = 0,
    vendor_id: u32 = 0,

    pub fn nameSlice(self: *const AdapterDesc) []const u8 {
        return self.name[0..self.name_len];
    }
    pub fn backendName(self: *const AdapterDesc) []const u8 {
        return switch (self.backend) {
            c.WGPUBackendType_Vulkan => "Vulkan",
            c.WGPUBackendType_D3D12 => "D3D12",
            c.WGPUBackendType_Metal => "Metal",
            c.WGPUBackendType_OpenGL => "OpenGL",
            else => "?",
        };
    }
    pub fn kindName(self: *const AdapterDesc) []const u8 {
        return switch (self.kind) {
            c.WGPUAdapterType_DiscreteGPU => "discrete",
            c.WGPUAdapterType_IntegratedGPU => "integrated",
            c.WGPUAdapterType_CPU => "cpu",
            else => "unknown",
        };
    }
};

pub fn describeAdapter(adapter: c.WGPUAdapter) AdapterDesc {
    var d: AdapterDesc = .{};
    var info: c.WGPUAdapterInfo = std.mem.zeroes(c.WGPUAdapterInfo);
    if (c.wgpuAdapterGetInfo(adapter, &info) == c.WGPUStatus_Success) {
        const name = fromStrv(info.device);
        d.name_len = @min(name.len, d.name.len);
        @memcpy(d.name[0..d.name_len], name[0..d.name_len]);
        d.backend = info.backendType;
        d.kind = info.adapterType;
        d.vendor_id = info.vendorID;
        c.wgpuAdapterInfoFreeMembers(info);
    }
    return d;
}

/// Enumerate all adapters into `buf`; returns how many were written. Adapters
/// are owned by the caller (release with `c.wgpuAdapterRelease`).
pub fn listAdapters(instance: c.WGPUInstance, buf: []c.WGPUAdapter) usize {
    const total = c.wgpuInstanceEnumerateAdapters(instance, null, null);
    const n = @min(total, buf.len);
    if (n == 0) return 0;
    _ = c.wgpuInstanceEnumerateAdapters(instance, null, buf.ptr);
    return n;
}

/// The device limits this backend actually cares about (subset of `WGPULimits`).
/// Defaults are the WebGPU spec MINIMUMS — every conformant device guarantees at
/// least these, so they're a safe floor if a query ever fails.
pub const Limits = struct {
    /// `maxComputeWorkgroupStorageSize` — shared-memory bytes per workgroup.
    max_shared_bytes: u32 = 16384,
    /// `maxStorageBufferBindingSize` — largest storage buffer bindable in a shader.
    max_storage_binding_bytes: u64 = 134217728, // 128 MiB
    /// `maxComputeInvocationsPerWorkgroup` — threads per workgroup.
    max_invocations: u32 = 256,
    max_workgroup_size_x: u32 = 256,
    /// Whether the native `Subgroup` feature was granted on this device (lets
    /// kernels use `enable subgroups;` + subgroup builtins). Gated per-config.
    subgroups: bool = false,
    /// Reported subgroup (warp/wave) width bounds from `WGPUAdapterInfo`. When
    /// min == max the size is fixed (32 on NVIDIA, 64 on AMD, 32 on most Intel),
    /// which lets codegen specialize a subgroup kernel to that width.
    subgroup_min: u32 = 0,
    subgroup_max: u32 = 0,

    /// A fixed subgroup width to specialize a kernel to, or null if the device
    /// reports a variable range (in which case subgroup-width-dependent kernels
    /// must not be emitted).
    pub fn fixedSubgroupSize(self: Limits) ?u32 {
        if (!self.subgroups) return null;
        if (self.subgroup_min == 0 or self.subgroup_min != self.subgroup_max) return null;
        return self.subgroup_min;
    }

    fn fromWgpu(l: c.WGPULimits) Limits {
        var out: Limits = .{};
        // Ignore UNDEFINED sentinels (keep the safe spec-minimum default).
        if (l.maxComputeWorkgroupStorageSize != c.WGPU_LIMIT_U32_UNDEFINED) out.max_shared_bytes = l.maxComputeWorkgroupStorageSize;
        if (l.maxStorageBufferBindingSize != c.WGPU_LIMIT_U64_UNDEFINED) out.max_storage_binding_bytes = l.maxStorageBufferBindingSize;
        if (l.maxComputeInvocationsPerWorkgroup != c.WGPU_LIMIT_U32_UNDEFINED) out.max_invocations = l.maxComputeInvocationsPerWorkgroup;
        if (l.maxComputeWorkgroupSizeX != c.WGPU_LIMIT_U32_UNDEFINED) out.max_workgroup_size_x = l.maxComputeWorkgroupSizeX;
        return out;
    }
};

/// Owns the WebGPU instance/adapter/device/queue. One `deinit` tears it down.
pub const Gpu = struct {
    instance: c.WGPUInstance,
    adapter: c.WGPUAdapter,
    device: c.WGPUDevice,
    queue: c.WGPUQueue,
    /// Granted device limits (used to gate kernel configs per GPU).
    limits: Limits = .{},

    pub fn init(opts: Options) Error!Gpu {
        const instance = c.wgpuCreateInstance(null) orelse return Error.NoInstance;
        errdefer c.wgpuInstanceRelease(instance);

        const adapter = try pickAdapter(instance, opts);
        errdefer c.wgpuAdapterRelease(adapter);

        // Learn what the adapter supports, then REQUEST those limits so a capable
        // GPU grants more than the spec-minimum defaults (e.g. >16KB shared memory,
        // which lets larger kernel tiles become usable). Requesting exactly the
        // adapter's reported limits is always valid; if the request fails for any
        // reason we retry with null (spec defaults).
        var adapter_limits: c.WGPULimits = std.mem.zeroes(c.WGPULimits);
        const have_adapter_limits = c.wgpuAdapterGetLimits(adapter, &adapter_limits) == c.WGPUStatus_Success;

        // Request the native Subgroup feature when the adapter advertises it, so
        // kernels may use `enable subgroups;`. It's optional: if the adapter lacks
        // it (or the device request rejects it) we fall back to a feature-less
        // device, and subgroup configs are simply never eligible.
        const subgroup_feat: c.WGPUFeatureName = @intCast(c.WGPUNativeFeature_Subgroup);
        const want_subgroup = c.wgpuAdapterHasFeature(adapter, subgroup_feat) != 0;
        const feats = [_]c.WGPUFeatureName{subgroup_feat};

        const device = blk: {
            if (want_subgroup) {
                if (requestDevice(instance, adapter, if (have_adapter_limits) &adapter_limits else null, &feats)) |d| break :blk d;
            }
            // Retry without the optional feature, then without the requested limits.
            break :blk requestDevice(instance, adapter, if (have_adapter_limits) &adapter_limits else null, &.{}) orelse
                (requestDevice(instance, adapter, null, &.{}) orelse return Error.NoDevice);
        };
        errdefer c.wgpuDeviceRelease(device);

        // Re-query the GRANTED limits (may be the requested ones, or defaults).
        var granted: Limits = .{};
        var dev_limits: c.WGPULimits = std.mem.zeroes(c.WGPULimits);
        if (c.wgpuDeviceGetLimits(device, &dev_limits) == c.WGPUStatus_Success) {
            granted = Limits.fromWgpu(dev_limits);
        }
        granted.subgroups = c.wgpuDeviceHasFeature(device, subgroup_feat) != 0;
        // Subgroup width bounds live in WGPUAdapterInfo (not WGPULimits).
        var info: c.WGPUAdapterInfo = std.mem.zeroes(c.WGPUAdapterInfo);
        if (c.wgpuAdapterGetInfo(adapter, &info) == c.WGPUStatus_Success) {
            granted.subgroup_min = info.subgroupMinSize;
            granted.subgroup_max = info.subgroupMaxSize;
            c.wgpuAdapterInfoFreeMembers(info);
        }

        const queue = c.wgpuDeviceGetQueue(device);
        var self: Gpu = .{ .instance = instance, .adapter = adapter, .device = device, .queue = queue, .limits = granted };

        // Some drivers (incl. this wgpu-native build on NVIDIA) leave
        // subgroupMin/MaxSize at 0 in WGPUAdapterInfo. When subgroups are granted
        // but the reported width is unusable, probe the true width at runtime with a
        // one-workgroup shader that writes `@builtin(subgroup_size)`. This also
        // proves Naga compiles `enable subgroups;` before any real kernel relies on
        // it — on failure we leave the size at 0 (subgroup configs stay ineligible).
        if (granted.subgroups and granted.fixedSubgroupSize() == null) {
            if (self.probeSubgroupSize()) |sz| {
                self.limits.subgroup_min = sz;
                self.limits.subgroup_max = sz;
            } else {
                self.limits.subgroups = false; // couldn't compile/run subgroup WGSL
            }
        }
        return self;
    }

    /// Request a device from `adapter` with optional `required` limits. Returns null
    /// on timeout/failure (caller decides whether to retry with relaxed limits).
    fn requestDevice(instance: c.WGPUInstance, adapter: c.WGPUAdapter, required: ?*const c.WGPULimits, features: []const c.WGPUFeatureName) ?c.WGPUDevice {
        var desc: c.WGPUDeviceDescriptor = std.mem.zeroes(c.WGPUDeviceDescriptor);
        desc.requiredLimits = required;
        if (features.len > 0) {
            desc.requiredFeatureCount = features.len;
            desc.requiredFeatures = features.ptr;
        }
        var dreq: DeviceReq = .{};
        _ = c.wgpuAdapterRequestDevice(adapter, &desc, .{
            .nextInChain = null,
            .mode = c.WGPUCallbackMode_AllowProcessEvents,
            .callback = onDevice,
            .userdata1 = &dreq,
            .userdata2 = null,
        });
        if (!pumpUntil(instance, &dreq.done)) return null;
        if (dreq.status != c.WGPURequestDeviceStatus_Success or dreq.device == null) return null;
        return dreq.device;
    }

    fn pickAdapter(instance: c.WGPUInstance, opts: Options) Error!c.WGPUAdapter {
        // Explicit index: enumerate and select, releasing the rest.
        if (opts.adapter_index) |want| {
            var bufs: [16]c.WGPUAdapter = undefined;
            const n = listAdapters(instance, &bufs);
            if (want >= n) {
                for (bufs[0..n]) |a| c.wgpuAdapterRelease(a);
                return Error.BadAdapterIndex;
            }
            for (bufs[0..n], 0..) |a, i| {
                if (i != want) c.wgpuAdapterRelease(a);
            }
            return bufs[want];
        }

        // Otherwise request by power preference + optional backend.
        var areq: AdapterReq = .{};
        const ro: c.WGPURequestAdapterOptions = .{
            .nextInChain = null,
            .featureLevel = c.WGPUFeatureLevel_Undefined,
            .powerPreference = powerPref(opts.power),
            .forceFallbackAdapter = 0,
            .backendType = backendType(opts.backend),
            .compatibleSurface = null,
        };
        _ = c.wgpuInstanceRequestAdapter(instance, &ro, .{
            .nextInChain = null,
            .mode = c.WGPUCallbackMode_AllowProcessEvents,
            .callback = onAdapter,
            .userdata1 = &areq,
            .userdata2 = null,
        });
        if (!pumpUntil(instance, &areq.done)) return Error.AdapterTimeout;
        if (areq.status != c.WGPURequestAdapterStatus_Success or areq.adapter == null) return Error.NoAdapter;
        return areq.adapter;
    }

    /// Runtime probe of the device's subgroup (warp/wave) width: dispatch one
    /// workgroup of a shader that stores `subgroup_size`, then read it back. Returns
    /// null if the shader fails to compile or run (which also tells the caller
    /// subgroup WGSL isn't usable on this device). All transient GPU objects are
    /// released before returning.
    ///
    /// NOTE: Naga in wgpu-native v29 rejects the `enable subgroups;` directive
    /// ("not yet supported") but DOES accept the subgroup builtins and ops
    /// (subgroup_size/subgroup_invocation_id, subgroupBroadcast/Shuffle/Add/...),
    /// so subgroup kernels must be emitted WITHOUT that enable directive.
    fn probeSubgroupSize(self: *Gpu) ?u32 {
        const src: [:0]const u8 =
            \\@group(0) @binding(0) var<storage, read_write> o: array<u32>;
            \\@compute @workgroup_size(64)
            \\fn probe(@builtin(subgroup_size) s: u32, @builtin(local_invocation_index) i: u32) {
            \\  if (i == 0u) { o[0] = s; }
            \\}
        ;
        var w: c.WGPUShaderSourceWGSL = std.mem.zeroes(c.WGPUShaderSourceWGSL);
        w.chain.sType = c.WGPUSType_ShaderSourceWGSL;
        w.code = strv(src);
        var smd: c.WGPUShaderModuleDescriptor = std.mem.zeroes(c.WGPUShaderModuleDescriptor);
        smd.nextInChain = &w.chain;
        const module = c.wgpuDeviceCreateShaderModule(self.device, &smd) orelse return null;
        defer c.wgpuShaderModuleRelease(module);

        var cpd: c.WGPUComputePipelineDescriptor = std.mem.zeroes(c.WGPUComputePipelineDescriptor);
        cpd.compute.module = module;
        cpd.compute.entryPoint = strv("probe");
        const pipeline = c.wgpuDeviceCreateComputePipeline(self.device, &cpd) orelse return null;
        defer c.wgpuComputePipelineRelease(pipeline);
        const bgl = c.wgpuComputePipelineGetBindGroupLayout(pipeline, 0) orelse return null;
        defer c.wgpuBindGroupLayoutRelease(bgl);

        const buf = createBuffer(self.device, 4, c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc) catch return null;
        defer c.wgpuBufferRelease(buf);

        var entry: c.WGPUBindGroupEntry = std.mem.zeroes(c.WGPUBindGroupEntry);
        entry.binding = 0;
        entry.buffer = buf;
        entry.size = 4;
        var bgd: c.WGPUBindGroupDescriptor = std.mem.zeroes(c.WGPUBindGroupDescriptor);
        bgd.layout = bgl;
        bgd.entryCount = 1;
        bgd.entries = &entry;
        const bg = c.wgpuDeviceCreateBindGroup(self.device, &bgd) orelse return null;
        defer c.wgpuBindGroupRelease(bg);

        const enc = c.wgpuDeviceCreateCommandEncoder(self.device, null) orelse return null;
        const pass = c.wgpuCommandEncoderBeginComputePass(enc, null);
        c.wgpuComputePassEncoderSetPipeline(pass, pipeline);
        c.wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, null);
        c.wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
        c.wgpuComputePassEncoderEnd(pass);
        c.wgpuComputePassEncoderRelease(pass);
        const cmd = c.wgpuCommandEncoderFinish(enc, null);
        c.wgpuCommandEncoderRelease(enc);
        c.wgpuQueueSubmit(self.queue, 1, &cmd);
        c.wgpuCommandBufferRelease(cmd);
        _ = c.wgpuDevicePoll(self.device, 1, null);

        var out_bytes: [4]u8 = undefined;
        self.readBuffer(buf, 0, &out_bytes) catch return null;
        const sz = std.mem.readInt(u32, &out_bytes, .little);
        if (sz == 0 or sz > 128) return null; // sanity
        return sz;
    }

    pub fn deinit(self: *Gpu) void {
        c.wgpuQueueRelease(self.queue);
        c.wgpuDeviceRelease(self.device);
        c.wgpuAdapterRelease(self.adapter);
        c.wgpuInstanceRelease(self.instance);
        self.* = undefined;
    }

    /// Identity of the selected adapter (for logging).
    pub fn describe(self: *Gpu) AdapterDesc {
        return describeAdapter(self.adapter);
    }

    /// Synchronously map `buffer` (blocks via the event loop).
    pub fn mapBlocking(self: *Gpu, buffer: c.WGPUBuffer, mode: c.WGPUMapMode, offset: usize, size: usize) Error!void {
        var mreq: MapReq = .{};
        _ = c.wgpuBufferMapAsync(buffer, mode, offset, size, .{
            .nextInChain = null,
            .mode = c.WGPUCallbackMode_AllowProcessEvents,
            .callback = onMap,
            .userdata1 = &mreq,
            .userdata2 = null,
        });
        if (!pumpUntil(self.instance, &mreq.done)) return Error.MapTimeout;
        if (mreq.status != c.WGPUMapAsyncStatus_Success) return Error.MapFailed;
    }

    /// Read `dst.len` bytes from device `src` (at `src_offset`) into host `dst`,
    /// staging through a temporary mappable buffer and blocking on completion.
    /// Owns the full lifetime — encoder, command buffer, and staging buffer are
    /// all released. (Storage buffers aren't MAP_READ, hence the copy.)
    pub fn readBuffer(self: *Gpu, src: c.WGPUBuffer, src_offset: u64, dst: []u8) Error!void {
        const staging = try createBuffer(self.device, dst.len, c.WGPUBufferUsage_MapRead | c.WGPUBufferUsage_CopyDst);
        defer c.wgpuBufferRelease(staging);

        const enc = c.wgpuDeviceCreateCommandEncoder(self.device, null);
        c.wgpuCommandEncoderCopyBufferToBuffer(enc, src, src_offset, staging, 0, dst.len);
        const cmd = c.wgpuCommandEncoderFinish(enc, null);
        c.wgpuCommandEncoderRelease(enc);
        c.wgpuQueueSubmit(self.queue, 1, &cmd);
        c.wgpuCommandBufferRelease(cmd);

        // Block until all submitted work (the compute that produced `src`, then the
        // copy above) is done. Without this, `mapBlocking`'s bounded event-pump can
        // give up before a long compute finishes (a heavy batched submit can run
        // longer than the pump's iteration budget), surfacing as a spurious map
        // timeout. Polling with wait=true makes readback robust regardless of how
        // much work was queued.
        _ = c.wgpuDevicePoll(self.device, 1, null);

        try self.mapBlocking(staging, c.WGPUMapMode_Read, 0, dst.len);
        const mapped = c.wgpuBufferGetConstMappedRange(staging, 0, dst.len) orelse return Error.MapFailed;
        @memcpy(dst, @as([*]const u8, @ptrCast(mapped))[0..dst.len]);
        c.wgpuBufferUnmap(staging);
    }
};

/// Create a buffer with the given byte size + usage flags.
pub fn createBuffer(device: c.WGPUDevice, size: u64, usage: c.WGPUBufferUsage) Error!c.WGPUBuffer {
    var bd: c.WGPUBufferDescriptor = std.mem.zeroes(c.WGPUBufferDescriptor);
    bd.usage = usage;
    bd.size = size;
    return c.wgpuDeviceCreateBuffer(device, &bd) orelse Error.BufferCreate;
}

/// Create a uniform buffer (host-writable via `wgpuQueueWriteBuffer`). Uniform
/// buffers carry kernel parameters (shapes/strides/scalars) so shaders don't have
/// to infer them from `arrayLength`. WebGPU requires uniform buffer bindings be a
/// multiple of 16 bytes; callers round `size` up accordingly.
pub fn createUniformBuffer(device: c.WGPUDevice, size: u64) Error!c.WGPUBuffer {
    return createBuffer(device, size, c.WGPUBufferUsage_Uniform | c.WGPUBufferUsage_CopyDst);
}
