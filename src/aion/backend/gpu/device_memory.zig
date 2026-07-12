// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! `WgpuDeviceMemory` — the WebGPU implementation of Aion's `DeviceMemory`
//! interface (`runtime/device_memory.zig`, reached here via `@import("aion")`).
//! Drop-in for `MockDeviceMemory`, so the residency layer (`ResidentTensorStore`)
//! and its tests run unchanged against real GPU buffers.
//!
//! WebGPU exposes explicit buffers even on unified hardware, so this reports
//! `.discrete`: H2D is `wgpuQueueWriteBuffer`; D2H stages through a temporary
//! mappable buffer (per-call here — a real backend would pool these). True
//! zero-copy unified memory would come from a native Metal/CUDA-managed backend.

const std = @import("std");
const wgpu = @import("wgpu.zig");

const c = wgpu.c;
const ThreadPool = @import("../../runtime/thread_pool.zig").ThreadPool;

/// D2H readbacks smaller than this stay single-threaded. The pool's wake+join
/// dispatch costs ~100-200us, which exceeds the memcpy savings until the copy is
/// large: measured crossover is ~8 MiB on this hardware (a 4 MiB copy regressed,
/// 16 MiB improved ~40%). Below this the serial memcpy already runs at full
/// single-core bandwidth, so keep it (also covers the decode logits/argmax case).
const PARALLEL_MEMCPY_MIN_BYTES: usize = 8 * 1024 * 1024;

/// Flush the queue after this many H2D bytes. Every `wgpuQueueWriteBuffer`
/// parks its payload in a fresh staging buffer that is only recycled once a
/// submit completes — with no submit during a bulk upload (model weights are
/// ~one writeBuffer per tile), wgpu holds staging for EVERY byte uploaded, so
/// peak memory is ~2x the model and the allocator keeps ~10% of it committed
/// afterwards (measured: +66 MiB resident on a 635 MiB model). An empty submit
/// + wait every 32 MiB bounds staging to this threshold; the poll cost is noise
/// next to the copies themselves.
const H2D_FLUSH_BYTES: usize = 32 * 1024 * 1024;

const dm = @import("../../runtime/residency/device_memory.zig");
const DeviceMemory = dm.DeviceMemory;
const DeviceHandle = dm.DeviceHandle;
const DeviceError = dm.DeviceError;
const MemoryModel = dm.MemoryModel;

pub const WgpuDeviceMemory = struct {
    allocator: std.mem.Allocator,
    gpu: *wgpu.Gpu,
    // handle = index + 1; 0 is "none". Tombstoned (null) on free.
    buffers: std.ArrayList(?c.WGPUBuffer) = .empty,
    // Reusable D2H staging buffer (MapRead|CopyDst). Grown on demand and kept
    // alive across readbacks, so the hot path never re-allocates mappable memory.
    // Grows monotonically; never shrinks.
    staging: ?c.WGPUBuffer = null,
    staging_cap: usize = 0,
    // Optional pool for parallelizing the mapped-staging -> host memcpy on large
    // readbacks (the single serial memcpy is ~half the D2H cost). Set by the owner
    // (GpuBackend); null = single-threaded memcpy.
    pool: ?*ThreadPool = null,
    // H2D bytes written since the last queue flush (see `H2D_FLUSH_BYTES`).
    h2d_since_flush: usize = 0,

    const Self = @This();

    const MemcpyJob = struct { dst: [*]u8, src: [*]const u8 };

    fn memcpyRange(ctx: *anyopaque, start: usize, end: usize, tid: usize) void {
        _ = tid;
        const j: *MemcpyJob = @ptrCast(@alignCast(ctx));
        @memcpy(j.dst[start..end], j.src[start..end]);
    }

    /// Copy `src` -> `dst` (equal lengths), across the pool when large enough.
    fn copyHostBytes(self: *Self, dst: []u8, src: []const u8) void {
        if (self.pool == null or dst.len < PARALLEL_MEMCPY_MIN_BYTES) {
            @memcpy(dst, src);
            return;
        }
        var job: MemcpyJob = .{ .dst = dst.ptr, .src = src.ptr };
        self.pool.?.parallelForAny(&job, dst.len, 0, memcpyRange);
    }

    pub fn init(allocator: std.mem.Allocator, gpu: *wgpu.Gpu) Self {
        return .{ .allocator = allocator, .gpu = gpu };
    }

    pub fn deinit(self: *Self) void {
        for (self.buffers.items) |maybe| {
            if (maybe) |buf| c.wgpuBufferRelease(buf);
        }
        if (self.staging) |s| c.wgpuBufferRelease(s);
        self.buffers.deinit(self.allocator);
        self.* = undefined;
    }

    /// Ensure the pooled staging buffer holds at least `bytes`, growing (with some
    /// slack, rounded to 1 MiB) if needed. Returns the staging buffer.
    fn ensureStaging(self: *Self, bytes: usize) DeviceError!c.WGPUBuffer {
        if (self.staging) |s| {
            if (self.staging_cap >= bytes) return s;
            c.wgpuBufferRelease(s);
            self.staging = null;
            self.staging_cap = 0;
        }
        const MiB = 1024 * 1024;
        const cap = (bytes + MiB - 1) / MiB * MiB;
        const buf = wgpu.createBuffer(self.gpu.device, @intCast(cap), c.WGPUBufferUsage_MapRead | c.WGPUBufferUsage_CopyDst) catch return DeviceError.OutOfDeviceMemory;
        self.staging = buf;
        self.staging_cap = cap;
        return buf;
    }

    pub fn device(self: *Self) DeviceMemory {
        return .{ .ctx = @ptrCast(self), .vtable = &vtable };
    }

    fn bufFor(self: *Self, handle: DeviceHandle) ?c.WGPUBuffer {
        if (handle == 0) return null;
        const slot: usize = @intCast(handle - 1);
        if (slot >= self.buffers.items.len) return null;
        return self.buffers.items[slot];
    }

    /// Resolve an opaque `DeviceHandle` (from a `TileRefDevice`) to its concrete
    /// `WGPUBuffer` so the backend can bind it in a compute pass. The backend
    /// owns this `WgpuDeviceMemory`, so it may reach past the abstract interface.
    pub fn bufferFor(self: *Self, handle: DeviceHandle) ?c.WGPUBuffer {
        return self.bufFor(handle);
    }

    fn model(_: *anyopaque) MemoryModel {
        return .discrete;
    }

    fn alloc(ctx: *anyopaque, bytes: usize, alignment: usize) DeviceError!DeviceHandle {
        _ = alignment; // WebGPU handles buffer alignment internally.
        const self: *Self = @ptrCast(@alignCast(ctx));
        // Resident tiles are uploaded to, copied from, and bound as storage.
        const usage = c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc | c.WGPUBufferUsage_CopyDst;
        const buf = wgpu.createBuffer(self.gpu.device, @intCast(bytes), usage) catch return DeviceError.OutOfDeviceMemory;
        self.buffers.append(self.allocator, buf) catch {
            c.wgpuBufferRelease(buf);
            return DeviceError.OutOfDeviceMemory;
        };
        return @intCast(self.buffers.items.len); // index+1
    }

    fn free(ctx: *anyopaque, handle: DeviceHandle) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        if (handle == 0) return;
        const slot: usize = @intCast(handle - 1);
        if (slot >= self.buffers.items.len) return;
        if (self.buffers.items[slot]) |buf| {
            c.wgpuBufferRelease(buf);
            self.buffers.items[slot] = null;
        }
    }

    fn copyH2D(ctx: *anyopaque, handle: DeviceHandle, dst_offset: usize, src: []const u8) DeviceError!void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const buf = self.bufFor(handle) orelse return DeviceError.InvalidArgument;
        c.wgpuQueueWriteBuffer(self.gpu.queue, buf, @intCast(dst_offset), src.ptr, src.len);
        // Bound wgpu's write-staging: an empty submit flushes the pending
        // writes, the wait lets wgpu recycle their staging buffers. Ordering is
        // unaffected (writeBuffer data was already ordered before any later
        // submit), so this is safe even mid-frame.
        self.h2d_since_flush += src.len;
        if (self.h2d_since_flush >= H2D_FLUSH_BYTES) {
            c.wgpuQueueSubmit(self.gpu.queue, 0, null);
            _ = c.wgpuDevicePoll(self.gpu.device, 1, null);
            self.h2d_since_flush = 0;
        }
    }

    fn copyD2H(ctx: *anyopaque, dst: []u8, handle: DeviceHandle, src_offset: usize) DeviceError!void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        if (dst.len == 0) return;
        const buf = self.bufFor(handle) orelse return DeviceError.InvalidArgument;
        const staging = try self.ensureStaging(dst.len);

        // Copy device->staging on the queue, then block until all prior submits
        // (the compute that produced `buf`) and this copy have completed.
        const enc = c.wgpuDeviceCreateCommandEncoder(self.gpu.device, null);
        c.wgpuCommandEncoderCopyBufferToBuffer(enc, buf, @intCast(src_offset), staging, 0, dst.len);
        const cmd = c.wgpuCommandEncoderFinish(enc, null);
        c.wgpuCommandEncoderRelease(enc);
        c.wgpuQueueSubmit(self.gpu.queue, 1, &cmd);
        c.wgpuCommandBufferRelease(cmd);
        _ = c.wgpuDevicePoll(self.gpu.device, 1, null);

        self.gpu.mapBlocking(staging, c.WGPUMapMode_Read, 0, dst.len) catch return DeviceError.InvalidArgument;
        const mapped = c.wgpuBufferGetConstMappedRange(staging, 0, dst.len) orelse return DeviceError.InvalidArgument;
        self.copyHostBytes(dst, @as([*]const u8, @ptrCast(mapped))[0..dst.len]);
        c.wgpuBufferUnmap(staging);
    }

    fn maxBindingBytes(ctx: *anyopaque) u64 {
        const self: *Self = @ptrCast(@alignCast(ctx));
        // The state slot is a single storage-bound tile, so the binding-size
        // limit is what gates whether it can be device-resident at all.
        return self.gpu.limits.max_storage_binding_bytes;
    }

    const vtable = DeviceMemory.VTable{
        .model = model,
        .alloc = alloc,
        .free = free,
        .copyH2D = copyH2D,
        .copyD2H = copyD2H,
        // importHost stays null: discrete memory has no host aliasing.
        .maxBindingBytes = maxBindingBytes,
    };
};
