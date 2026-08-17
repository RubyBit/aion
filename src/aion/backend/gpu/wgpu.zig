// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Thin, use-driven ergonomics over the translate-c'd wgpu.h (standard webgpu.h
//! + wgpu-native extensions). NOT a general WebGPU abstraction — it collapses
//! the three genuine footguns of the raw C API (callback+pump async,
//! status->error, StringView), owns the instance/adapter/device/queue lifetime,
//! and handles adapter (GPU) selection. Raw handles stay reachable via `wgpu.c`.

const std = @import("std");
const builtin = @import("builtin");
const env_util = @import("../../env.zig");

/// The translate-c'd wgpu.h: **types, enums and constants only** (`c.WGPU…`).
/// These are header-level declarations, so referencing them creates no link
/// dependency. wgpu *functions* are never called through `c` — they are resolved
/// at runtime and dispatched through `w` (see the loader below).
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
    /// The adapter lacks `shader-f16`; f16 tensors are stored and addressed natively.
    NoShaderF16,
    /// wgpu-native could not be loaded (no `aion-wgpu` package / system library).
    Unavailable,
};

// ── wgpu-native is loaded at runtime, never linked ──────────────────────────
// wgpu-native ships as a standalone dynamic library (~a few MiB). Linking it
// would make the whole Python extension fail to load whenever it is absent, so
// instead the static `aion` archive references ZERO wgpu symbols: every function
// the backend calls is looked up by name on first use and dispatched through the
// table `w`. A CPU-only deployment (no `aion-wgpu` package, no system wgpu) never
// loads it, and GPU device creation fails cleanly with `error.Unavailable`
// (surfaced to the C ABI as `AION_UNSUPPORTED`).
//
// The library path comes from `AION_WGPU_LIB` when set — the Python `aion`
// package points it at the bundled `aion-wgpu` library; the in-tree Zig GPU
// test/bench run steps point it at the fetched wgpu-native prebuilt — otherwise
// the platform's default search is used (a system install or a copy beside the
// process).

/// Runtime dispatch table over every wgpu-native function the backend calls: one
/// field per function, typed straight from the translate-c'd signature so call
/// sites stay fully type-checked. This declaration is the single source of truth
/// — `ensureLoaded` resolves exactly these fields by name, and calling a wgpu
/// function that is not a field here is a COMPILE error, which keeps the set
/// exhaustive by construction. Call through it as `fns.wgpuXxx(args)`.
/// Handrolled because std.DynLib is @compileError on win
const Procs = struct {
    wgpuAdapterGetInfo: *const @TypeOf(c.wgpuAdapterGetInfo),
    wgpuAdapterGetLimits: *const @TypeOf(c.wgpuAdapterGetLimits),
    wgpuAdapterHasFeature: *const @TypeOf(c.wgpuAdapterHasFeature),
    wgpuAdapterInfoFreeMembers: *const @TypeOf(c.wgpuAdapterInfoFreeMembers),
    wgpuAdapterRelease: *const @TypeOf(c.wgpuAdapterRelease),
    wgpuAdapterRequestDevice: *const @TypeOf(c.wgpuAdapterRequestDevice),
    wgpuBindGroupLayoutRelease: *const @TypeOf(c.wgpuBindGroupLayoutRelease),
    wgpuBindGroupRelease: *const @TypeOf(c.wgpuBindGroupRelease),
    wgpuBufferGetConstMappedRange: *const @TypeOf(c.wgpuBufferGetConstMappedRange),
    wgpuBufferMapAsync: *const @TypeOf(c.wgpuBufferMapAsync),
    wgpuBufferRelease: *const @TypeOf(c.wgpuBufferRelease),
    wgpuBufferUnmap: *const @TypeOf(c.wgpuBufferUnmap),
    wgpuCommandBufferRelease: *const @TypeOf(c.wgpuCommandBufferRelease),
    wgpuCommandEncoderBeginComputePass: *const @TypeOf(c.wgpuCommandEncoderBeginComputePass),
    wgpuCommandEncoderCopyBufferToBuffer: *const @TypeOf(c.wgpuCommandEncoderCopyBufferToBuffer),
    wgpuCommandEncoderFinish: *const @TypeOf(c.wgpuCommandEncoderFinish),
    wgpuCommandEncoderRelease: *const @TypeOf(c.wgpuCommandEncoderRelease),
    wgpuCommandEncoderResolveQuerySet: *const @TypeOf(c.wgpuCommandEncoderResolveQuerySet),
    wgpuComputePassEncoderDispatchWorkgroups: *const @TypeOf(c.wgpuComputePassEncoderDispatchWorkgroups),
    wgpuComputePassEncoderEnd: *const @TypeOf(c.wgpuComputePassEncoderEnd),
    wgpuComputePassEncoderRelease: *const @TypeOf(c.wgpuComputePassEncoderRelease),
    wgpuComputePassEncoderSetBindGroup: *const @TypeOf(c.wgpuComputePassEncoderSetBindGroup),
    wgpuComputePassEncoderSetPipeline: *const @TypeOf(c.wgpuComputePassEncoderSetPipeline),
    wgpuComputePipelineGetBindGroupLayout: *const @TypeOf(c.wgpuComputePipelineGetBindGroupLayout),
    wgpuComputePipelineRelease: *const @TypeOf(c.wgpuComputePipelineRelease),
    wgpuCreateInstance: *const @TypeOf(c.wgpuCreateInstance),
    wgpuDeviceCreateBindGroup: *const @TypeOf(c.wgpuDeviceCreateBindGroup),
    wgpuDeviceCreateBuffer: *const @TypeOf(c.wgpuDeviceCreateBuffer),
    wgpuDeviceCreateCommandEncoder: *const @TypeOf(c.wgpuDeviceCreateCommandEncoder),
    wgpuDeviceCreateComputePipeline: *const @TypeOf(c.wgpuDeviceCreateComputePipeline),
    wgpuDeviceCreateQuerySet: *const @TypeOf(c.wgpuDeviceCreateQuerySet),
    wgpuDeviceCreateShaderModule: *const @TypeOf(c.wgpuDeviceCreateShaderModule),
    wgpuDeviceGetLimits: *const @TypeOf(c.wgpuDeviceGetLimits),
    wgpuDeviceGetQueue: *const @TypeOf(c.wgpuDeviceGetQueue),
    wgpuDevicePoll: *const @TypeOf(c.wgpuDevicePoll),
    wgpuDeviceRelease: *const @TypeOf(c.wgpuDeviceRelease),
    wgpuInstanceEnumerateAdapters: *const @TypeOf(c.wgpuInstanceEnumerateAdapters),
    wgpuInstanceProcessEvents: *const @TypeOf(c.wgpuInstanceProcessEvents),
    wgpuInstanceRelease: *const @TypeOf(c.wgpuInstanceRelease),
    wgpuInstanceRequestAdapter: *const @TypeOf(c.wgpuInstanceRequestAdapter),
    wgpuQuerySetRelease: *const @TypeOf(c.wgpuQuerySetRelease),
    wgpuQueueGetTimestampPeriod: *const @TypeOf(c.wgpuQueueGetTimestampPeriod),
    wgpuQueueRelease: *const @TypeOf(c.wgpuQueueRelease),
    wgpuQueueSubmit: *const @TypeOf(c.wgpuQueueSubmit),
    wgpuQueueWriteBuffer: *const @TypeOf(c.wgpuQueueWriteBuffer),
    wgpuSetLogCallback: *const @TypeOf(c.wgpuSetLogCallback),
    wgpuSetLogLevel: *const @TypeOf(c.wgpuSetLogLevel),
    wgpuShaderModuleRelease: *const @TypeOf(c.wgpuShaderModuleRelease),
};

var procs: Procs = undefined;

/// The wgpu dispatch table. Valid only after a successful `ensureLoaded()`, which
/// every code path guarantees by loading before its first wgpu call.
pub const fns = &procs;

var load_state: enum { unattempted, ok, failed } = .unattempted;

/// Load wgpu-native and populate `fns` (idempotent). Must run before the first
/// wgpu call on any path — instance creation and the log setters. Returns
/// `error.Unavailable` if the library, or any expected symbol, is missing.
pub fn ensureLoaded() Error!void {
    switch (load_state) {
        .ok => return,
        .failed => return Error.Unavailable,
        .unattempted => {},
    }
    load_state = .failed; // until fully wired, so a partial load never looks OK
    const handle = openLibrary() orelse return Error.Unavailable;
    inline for (@typeInfo(Procs).@"struct".field_names) |field_name| {
        const sym = dlSymbol(handle, field_name) orelse {
            dlClose(handle);
            return Error.Unavailable;
        };
        // dlsym/GetProcAddress hand back an align-1 opaque pointer; function
        // pointers require higher alignment on some targets (e.g. 4 on aarch64).
        // The address is a real code symbol, so the alignment cast is sound.
        @field(procs, field_name) = @ptrCast(@alignCast(sym));
    }
    load_state = .ok;
}

const default_names: []const [:0]const u8 = switch (builtin.os.tag) {
    .windows => &.{"wgpu_native.dll"},
    .macos => &.{"libwgpu_native.dylib"},
    else => &.{"libwgpu_native.so"},
};

fn openLibrary() ?LibHandle {
    const override = env_util.getOwned(std.heap.page_allocator, "AION_WGPU_LIB");
    defer if (override) |path| std.heap.page_allocator.free(path);
    if (override) |path| {
        if (path.len != 0) {
            if (dlOpen(path)) |h| return h;
        }
    }
    inline for (default_names) |name| {
        if (dlOpen(name)) |h| return h;
    }
    return null;
}

// Minimal cross-platform dynamic loader. std.DynLib is `@compileError` on Windows
// in this Zig, so the Win32 loader is declared directly; POSIX uses libc
// `dlopen`/`dlsym` (already linked on non-Windows targets). Only the active
// platform's block is analyzed, so the Win32 externs never reach a POSIX build.
const is_windows = builtin.os.tag == .windows;
const LibHandle = if (is_windows) std.os.windows.HMODULE else *anyopaque;

const loader = if (is_windows) struct {
    extern "kernel32" fn LoadLibraryW(name: [*:0]const u16) callconv(.winapi) ?std.os.windows.HMODULE;
    extern "kernel32" fn GetProcAddress(module: std.os.windows.HMODULE, name: [*:0]const u8) callconv(.winapi) ?*anyopaque;
    extern "kernel32" fn FreeLibrary(module: std.os.windows.HMODULE) callconv(.winapi) i32;

    fn open(path: []const u8) ?LibHandle {
        var wbuf: [4096]u16 = undefined;
        if (path.len >= wbuf.len) return null;
        const n = std.unicode.utf8ToUtf16Le(wbuf[0 .. wbuf.len - 1], path) catch return null;
        wbuf[n] = 0;
        return LoadLibraryW(wbuf[0..n :0]);
    }
    fn symbol(handle: LibHandle, name: [:0]const u8) ?*const anyopaque {
        return GetProcAddress(handle, name.ptr);
    }
    fn close(handle: LibHandle) void {
        _ = FreeLibrary(handle);
    }
} else struct {
    fn open(path: []const u8) ?LibHandle {
        var buf: [4096]u8 = undefined;
        if (path.len >= buf.len) return null;
        @memcpy(buf[0..path.len], path);
        buf[path.len] = 0;
        return std.c.dlopen(buf[0..path.len :0].ptr, .{ .NOW = true });
    }
    fn symbol(handle: LibHandle, name: [:0]const u8) ?*const anyopaque {
        return std.c.dlsym(handle, name.ptr);
    }
    fn close(handle: LibHandle) void {
        _ = std.c.dlclose(handle);
    }
};

fn dlOpen(path: []const u8) ?LibHandle {
    return loader.open(path);
}
fn dlSymbol(handle: LibHandle, name: [:0]const u8) ?*const anyopaque {
    return loader.symbol(handle, name);
}
fn dlClose(handle: LibHandle) void {
    loader.close(handle);
}

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
        fns.wgpuInstanceProcessEvents(instance);
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
    if (fns.wgpuAdapterGetInfo(adapter, &info) == c.WGPUStatus_Success) {
        const name = fromStrv(info.device);
        d.name_len = @min(name.len, d.name.len);
        @memcpy(d.name[0..d.name_len], name[0..d.name_len]);
        d.backend = info.backendType;
        d.kind = info.adapterType;
        d.vendor_id = info.vendorID;
        fns.wgpuAdapterInfoFreeMembers(info);
    }
    return d;
}

/// Enumerate all adapters into `buf`; returns how many were written. Adapters
/// are owned by the caller (release with `fns.wgpuAdapterRelease`).
pub fn listAdapters(instance: c.WGPUInstance, buf: []c.WGPUAdapter) usize {
    const total = fns.wgpuInstanceEnumerateAdapters(instance, null, null);
    const n = @min(total, buf.len);
    if (n == 0) return 0;
    _ = fns.wgpuInstanceEnumerateAdapters(instance, null, buf.ptr);
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
    /// True when the device was opened with the standard timestamp-query feature.
    timestamp_query: bool = false,

    pub fn init(opts: Options) Error!Gpu {
        try ensureLoaded();
        const instance = fns.wgpuCreateInstance(null) orelse return Error.NoInstance;
        errdefer fns.wgpuInstanceRelease(instance);

        const adapter = try pickAdapter(instance, opts);
        errdefer fns.wgpuAdapterRelease(adapter);

        // Learn what the adapter supports, then REQUEST those limits so a capable
        // GPU grants more than the spec-minimum defaults (e.g. >16KB shared memory,
        // which lets larger kernel tiles become usable). Requesting exactly the
        // adapter's reported limits is always valid; if the request fails for any
        // reason we retry with null (spec defaults).
        var adapter_limits: c.WGPULimits = std.mem.zeroes(c.WGPULimits);
        const have_adapter_limits = fns.wgpuAdapterGetLimits(adapter, &adapter_limits) == c.WGPUStatus_Success;

        // shader-f16 is required, not probed: f16 tensors are bound as `array<f16>`.
        if (fns.wgpuAdapterHasFeature(adapter, c.WGPUFeatureName_ShaderF16) == 0) return Error.NoShaderF16;

        const want_timestamps = fns.wgpuAdapterHasFeature(adapter, c.WGPUFeatureName_TimestampQuery) != 0;
        var timestamp_query = want_timestamps;
        const lim: ?*const c.WGPULimits = if (have_adapter_limits) &adapter_limits else null;
        const device = requestDevice(instance, adapter, lim, timestamp_query) orelse
            requestDevice(instance, adapter, null, timestamp_query) orelse fallback: {
            // A buggy adapter may advertise TimestampQuery but reject it at
            // device creation. Profiling is optional; device execution is not.
            timestamp_query = false;
            break :fallback requestDevice(instance, adapter, lim, false) orelse
                (requestDevice(instance, adapter, null, false) orelse return Error.NoDevice);
        };
        errdefer fns.wgpuDeviceRelease(device);

        // Re-query the GRANTED limits (may be the requested ones, or defaults).
        var granted: Limits = .{};
        var dev_limits: c.WGPULimits = std.mem.zeroes(c.WGPULimits);
        if (fns.wgpuDeviceGetLimits(device, &dev_limits) == c.WGPUStatus_Success) {
            granted = Limits.fromWgpu(dev_limits);
        }

        const queue = fns.wgpuDeviceGetQueue(device);
        return .{ .instance = instance, .adapter = adapter, .device = device, .queue = queue, .limits = granted, .timestamp_query = timestamp_query };
    }

    /// Request a device from `adapter` with optional `required` limits. Returns null
    /// on timeout/failure (caller decides whether to retry with relaxed limits).
    fn requestDevice(instance: c.WGPUInstance, adapter: c.WGPUAdapter, required: ?*const c.WGPULimits, timestamps: bool) ?c.WGPUDevice {
        var desc: c.WGPUDeviceDescriptor = std.mem.zeroes(c.WGPUDeviceDescriptor);
        desc.requiredLimits = required;
        var required_features = [_]c.WGPUFeatureName{ c.WGPUFeatureName_ShaderF16, c.WGPUFeatureName_TimestampQuery };
        desc.requiredFeatureCount = if (timestamps) required_features.len else 1;
        desc.requiredFeatures = &required_features;
        // Shrink the allocator's suballocation blocks. wgpu's default hint
        // (Performance) commits 128 MiB device + 64 MiB host blocks for the
        // handful of tiny internal allocations made at device open (~200 MiB
        // idle on NVIDIA), and rounds all committed memory up to 256 MiB
        // steps. Manual growth 4->256 MiB keeps the idle footprint ~17 MiB
        // while steady-state blocks still reach full size, and load-time (not
        // per-frame) allocation makes the extra block count free. Guarded:
        // upstream wgpu-native doesn't expose memoryHints yet (PR pending), so
        // this compiles to a no-op against an unpatched header.
        var extras: c.WGPUDeviceExtras = std.mem.zeroes(c.WGPUDeviceExtras);
        if (comptime @hasField(c.WGPUDeviceExtras, "memoryHints")) {
            const MiB = 1024 * 1024;
            extras.chain.sType = c.WGPUSType_DeviceExtras;
            extras.memoryHints = c.WGPUMemoryHints_Manual;
            extras.suballocatedDeviceMemoryBlockSizeStart = 4 * MiB;
            extras.suballocatedDeviceMemoryBlockSizeEnd = 256 * MiB;
            desc.nextInChain = &extras.chain;
        }
        var dreq: DeviceReq = .{};
        _ = fns.wgpuAdapterRequestDevice(adapter, &desc, .{
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
                for (bufs[0..n]) |a| fns.wgpuAdapterRelease(a);
                return Error.BadAdapterIndex;
            }
            for (bufs[0..n], 0..) |a, i| {
                if (i != want) fns.wgpuAdapterRelease(a);
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
        _ = fns.wgpuInstanceRequestAdapter(instance, &ro, .{
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

    pub fn deinit(self: *Gpu) void {
        fns.wgpuQueueRelease(self.queue);
        fns.wgpuDeviceRelease(self.device);
        fns.wgpuAdapterRelease(self.adapter);
        fns.wgpuInstanceRelease(self.instance);
        self.* = undefined;
    }

    /// Identity of the selected adapter (for logging).
    pub fn describe(self: *Gpu) AdapterDesc {
        return describeAdapter(self.adapter);
    }

    /// Synchronously map `buffer` (blocks via the event loop).
    /// Map `buffer` and block until the callback fires.
    ///
    /// The wait is a blocking device poll, not an event-pump spin: a map callback
    /// cannot fire until the queue work the buffer depends on completes, so polling
    /// with wait=true is both the cheapest way to get there (no busy loop) and
    /// robust to an arbitrarily long queue ahead of it. That is why callers need no
    /// separate drain before mapping — this single wait covers both.
    pub fn mapBlocking(self: *Gpu, buffer: c.WGPUBuffer, mode: c.WGPUMapMode, offset: usize, size: usize) Error!void {
        var mreq: MapReq = .{};
        _ = fns.wgpuBufferMapAsync(buffer, mode, offset, size, .{
            .nextInChain = null,
            .mode = c.WGPUCallbackMode_AllowProcessEvents,
            .callback = onMap,
            .userdata1 = &mreq,
            .userdata2 = null,
        });
        var i: usize = 0;
        while (!mreq.done) : (i += 1) {
            if (i > 1024) return Error.MapTimeout;
            _ = fns.wgpuDevicePoll(self.device, 1, null); // wait = true
            fns.wgpuInstanceProcessEvents(self.instance);
        }
        if (mreq.status != c.WGPUMapAsyncStatus_Success) return Error.MapFailed;
    }

    /// Read `dst.len` bytes from device `src` (at `src_offset`) into host `dst`,
    /// staging through a temporary mappable buffer and blocking on completion.
    /// Owns the full lifetime — encoder, command buffer, and staging buffer are
    /// all released. (Storage buffers aren't MAP_READ, hence the copy.)
    pub fn readBuffer(self: *Gpu, src: c.WGPUBuffer, src_offset: u64, dst: []u8) Error!void {
        const staging = try createBuffer(self.device, dst.len, c.WGPUBufferUsage_MapRead | c.WGPUBufferUsage_CopyDst);
        defer fns.wgpuBufferRelease(staging);

        const enc = fns.wgpuDeviceCreateCommandEncoder(self.device, null);
        fns.wgpuCommandEncoderCopyBufferToBuffer(enc, src, src_offset, staging, 0, dst.len);
        const cmd = fns.wgpuCommandEncoderFinish(enc, null);
        fns.wgpuCommandEncoderRelease(enc);
        fns.wgpuQueueSubmit(self.queue, 1, &cmd);
        fns.wgpuCommandBufferRelease(cmd);

        // `mapBlocking` waits on the device, so it also covers the compute that
        // produced `src` and the copy above — no separate drain.
        try self.mapBlocking(staging, c.WGPUMapMode_Read, 0, dst.len);
        const mapped = fns.wgpuBufferGetConstMappedRange(staging, 0, dst.len) orelse return Error.MapFailed;
        @memcpy(dst, @as([*]const u8, @ptrCast(mapped))[0..dst.len]);
        fns.wgpuBufferUnmap(staging);
    }
};

/// Create a buffer with the given byte size + usage flags.
pub fn createBuffer(device: c.WGPUDevice, size: u64, usage: c.WGPUBufferUsage) Error!c.WGPUBuffer {
    var bd: c.WGPUBufferDescriptor = std.mem.zeroes(c.WGPUBufferDescriptor);
    bd.usage = usage;
    bd.size = size;
    return fns.wgpuDeviceCreateBuffer(device, &bd) orelse Error.BufferCreate;
}

/// Create a uniform buffer (host-writable via `wgpuQueueWriteBuffer`). Uniform
/// buffers carry kernel parameters (shapes/strides/scalars) so shaders don't have
/// to infer them from `arrayLength`. WebGPU requires uniform buffer bindings be a
/// multiple of 16 bytes; callers round `size` up accordingly.
pub fn createUniformBuffer(device: c.WGPUDevice, size: u64) Error!c.WGPUBuffer {
    return createBuffer(device, size, c.WGPUBufferUsage_Uniform | c.WGPUBufferUsage_CopyDst);
}
