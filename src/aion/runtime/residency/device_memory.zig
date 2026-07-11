// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! `DeviceMemory` — an API-agnostic device-buffer allocator + host<->device
//! transfer interface. WebGPU/Vulkan/Metal/CUDA each implement this vtable; the
//! residency layer (`resident_store.zig`) is written against it and never names
//! a concrete GPU API.
//!
//! `MockDeviceMemory` is a host-backed implementation (a second allocation +
//! `@memcpy`) that lets the residency staging/dirty/swap logic be exercised in
//! ordinary CPU tests with no GPU present.

const std = @import("std");

/// Opaque device-buffer handle. A real backend maps this to a GPU buffer id /
/// pointer; the mock maps it to a host allocation.
pub const DeviceHandle = u64;

pub const DeviceError = error{ OutOfDeviceMemory, InvalidArgument, Unsupported };

/// How device memory relates to host memory — the single knob that decides
/// whether the residency layer copies or aliases.
///
/// - `.discrete`: separate VRAM. Host<->device needs explicit transfers; the
///   residency decorator allocates device buffers and tracks dirty state.
/// - `.unified`: host and device share physical memory (Apple Silicon, APUs,
///   CUDA managed, host-visible-device-local). A host allocation can be bound
///   directly via `importHost`; no copies, no dirty tracking.
pub const MemoryModel = enum { discrete, unified };

pub const DeviceMemory = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Memory model — decides whether the residency layer copies or aliases.
        model: *const fn (ctx: *anyopaque) MemoryModel,
        /// Allocate a device buffer of `bytes` (with at least `alignment`).
        alloc: *const fn (ctx: *anyopaque, bytes: usize, alignment: usize) DeviceError!DeviceHandle,
        /// Free a previously allocated buffer.
        free: *const fn (ctx: *anyopaque, handle: DeviceHandle) void,
        /// Host -> device copy into `handle` at `dst_offset`.
        copyH2D: *const fn (ctx: *anyopaque, handle: DeviceHandle, dst_offset: usize, src: []const u8) DeviceError!void,
        /// Device -> host copy from `handle` at `src_offset` into `dst`.
        copyD2H: *const fn (ctx: *anyopaque, dst: []u8, handle: DeviceHandle, src_offset: usize) DeviceError!void,
        /// Unified-memory only: register an existing host allocation as a
        /// device-bindable buffer with NO copy (the handle aliases `host`). Null
        /// on discrete devices. Mirrors Vulkan VK_EXT_external_memory_host /
        /// cudaHostRegister / a host-visible mapped buffer.
        importHost: ?*const fn (ctx: *anyopaque, host: []u8) DeviceError!DeviceHandle = null,

        /// Largest single buffer that can be allocated AND bound as storage
        /// (bytes) — the device's reported per-allocation ceiling (WebGPU's
        /// `maxStorageBufferBindingSize`). The residency/model layer uses it as
        /// the fit-gate before making a tensor device-exclusive. This is a
        /// per-buffer limit, NOT total capacity (WebGPU exposes no VRAM total; on
        /// unified hardware device buffers consume RAM). Null → unbounded.
        maxBindingBytes: ?*const fn (ctx: *anyopaque) u64 = null,
    };

    pub fn model(self: DeviceMemory) MemoryModel {
        return self.vtable.model(self.ctx);
    }
    pub fn alloc(self: DeviceMemory, bytes: usize, alignment: usize) DeviceError!DeviceHandle {
        return self.vtable.alloc(self.ctx, bytes, alignment);
    }
    pub fn free(self: DeviceMemory, handle: DeviceHandle) void {
        return self.vtable.free(self.ctx, handle);
    }
    pub fn copyH2D(self: DeviceMemory, handle: DeviceHandle, dst_offset: usize, src: []const u8) DeviceError!void {
        return self.vtable.copyH2D(self.ctx, handle, dst_offset, src);
    }
    pub fn copyD2H(self: DeviceMemory, dst: []u8, handle: DeviceHandle, src_offset: usize) DeviceError!void {
        return self.vtable.copyD2H(self.ctx, dst, handle, src_offset);
    }
    pub fn importHost(self: DeviceMemory, host: []u8) DeviceError!DeviceHandle {
        if (self.vtable.importHost) |f| return f(self.ctx, host);
        return DeviceError.Unsupported;
    }
    pub fn maxBindingBytes(self: DeviceMemory) u64 {
        if (self.vtable.maxBindingBytes) |f| return f(self.ctx);
        return std.math.maxInt(u64);
    }
};

/// Host-backed `DeviceMemory` for tests. Each "device buffer" is either a real
/// heap allocation (discrete: H2D/D2H are `@memcpy`) or an aliased host slice
/// (unified: `importHost`, no copy). Counts transfers so tests can assert that
/// residency elides redundant copies and that the unified path copies nothing.
pub const MockDeviceMemory = struct {
    const Buf = struct { mem: []u8, owned: bool };

    allocator: std.mem.Allocator,
    mem_model: MemoryModel = .discrete,
    buffers: std.ArrayList(Buf) = .empty,
    next_handle: DeviceHandle = 1, // 0 reserved as "none"
    /// Reported per-buffer ceiling (see `VTable.maxBindingBytes`). Unbounded by
    /// default; tests set it small to exercise the device-exclusive fit-gate's
    /// fallback path.
    max_binding_bytes: u64 = std.math.maxInt(u64),

    // Transfer counters for test assertions.
    h2d_count: usize = 0,
    d2h_count: usize = 0,
    bytes_h2d: usize = 0,
    bytes_d2h: usize = 0,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{ .allocator = allocator, .mem_model = .discrete };
    }

    pub fn initWithModel(allocator: std.mem.Allocator, mem_model: MemoryModel) Self {
        return .{ .allocator = allocator, .mem_model = mem_model };
    }

    pub fn deinit(self: *Self) void {
        for (self.buffers.items) |buf| {
            if (buf.owned and buf.mem.len != 0) self.allocator.free(buf.mem);
        }
        self.buffers.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn device(self: *Self) DeviceMemory {
        return .{ .ctx = @ptrCast(self), .vtable = &vtable };
    }

    /// Test helper: the bytes a handle refers to (no transfer counted). For an
    /// imported (unified) handle this aliases the host tensor's memory.
    pub fn bufferSlice(self: *Self, handle: DeviceHandle) ?[]u8 {
        return bufFor(self, handle);
    }

    fn slotOf(handle: DeviceHandle) ?usize {
        if (handle == 0) return null;
        return @intCast(handle - 1);
    }

    fn pushBuf(self: *Self, buf: Buf) DeviceError!DeviceHandle {
        self.buffers.append(self.allocator, buf) catch return DeviceError.OutOfDeviceMemory;
        const handle = self.next_handle;
        self.next_handle += 1;
        return handle;
    }

    fn model(ctx: *anyopaque) MemoryModel {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return self.mem_model;
    }

    fn alloc(ctx: *anyopaque, bytes: usize, alignment: usize) DeviceError!DeviceHandle {
        _ = alignment; // host allocation is already suitably aligned for the mock
        const self: *Self = @ptrCast(@alignCast(ctx));
        const buf = self.allocator.alloc(u8, bytes) catch return DeviceError.OutOfDeviceMemory;
        return self.pushBuf(.{ .mem = buf, .owned = true }) catch |e| {
            self.allocator.free(buf);
            return e;
        };
    }

    fn importHost(ctx: *anyopaque, host: []u8) DeviceError!DeviceHandle {
        const self: *Self = @ptrCast(@alignCast(ctx));
        // Alias the host allocation — no copy, not owned by the device.
        return self.pushBuf(.{ .mem = host, .owned = false });
    }

    fn free(ctx: *anyopaque, handle: DeviceHandle) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const slot = slotOf(handle) orelse return;
        if (slot >= self.buffers.items.len) return;
        const buf = self.buffers.items[slot];
        if (buf.owned and buf.mem.len != 0) self.allocator.free(buf.mem);
        self.buffers.items[slot] = .{ .mem = &[_]u8{}, .owned = false }; // tombstone
    }

    fn bufFor(self: *Self, handle: DeviceHandle) ?[]u8 {
        const slot = slotOf(handle) orelse return null;
        if (slot >= self.buffers.items.len) return null;
        const buf = self.buffers.items[slot];
        if (buf.mem.len == 0) return null;
        return buf.mem;
    }

    fn copyH2D(ctx: *anyopaque, handle: DeviceHandle, dst_offset: usize, src: []const u8) DeviceError!void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const buf = bufFor(self, handle) orelse return DeviceError.InvalidArgument;
        if (dst_offset + src.len > buf.len) return DeviceError.InvalidArgument;
        @memcpy(buf[dst_offset .. dst_offset + src.len], src);
        self.h2d_count += 1;
        self.bytes_h2d += src.len;
    }

    fn copyD2H(ctx: *anyopaque, dst: []u8, handle: DeviceHandle, src_offset: usize) DeviceError!void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const buf = bufFor(self, handle) orelse return DeviceError.InvalidArgument;
        if (src_offset + dst.len > buf.len) return DeviceError.InvalidArgument;
        @memcpy(dst, buf[src_offset .. src_offset + dst.len]);
        self.d2h_count += 1;
        self.bytes_d2h += dst.len;
    }

    fn maxBindingBytes(ctx: *anyopaque) u64 {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return self.max_binding_bytes;
    }

    const vtable = DeviceMemory.VTable{
        .model = model,
        .alloc = alloc,
        .free = free,
        .copyH2D = copyH2D,
        .copyD2H = copyD2H,
        .importHost = importHost,
        .maxBindingBytes = maxBindingBytes,
    };
};
