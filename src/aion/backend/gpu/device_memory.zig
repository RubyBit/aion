// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! `WgpuDeviceMemory` — the WebGPU implementation of Aion's `DeviceMemory`
//! interface (`runtime/device_memory.zig`, reached here via `@import("aion")`).
//! Drop-in for `MockDeviceMemory`, used by placement and explicit transfers.
//!
//! WebGPU exposes explicit buffers even on unified hardware, so this reports
//! `.discrete`: H2D is `wgpuQueueWriteBuffer`; D2H stages through a temporary
//! mappable buffer (per-call here — a real backend would pool these). True
//! zero-copy unified memory would come from a native Metal/CUDA-managed backend.

const std = @import("std");
const wgpu = @import("wgpu.zig");

const c = wgpu.c;
const fns = wgpu.fns; // runtime wgpu dispatch table (functions)
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

/// WebGPU copy sizes/offsets and storage bindings are whole u32 words, so a tile
/// with an odd byte length (odd-element f16) is backed rounded up — every tile is
/// its own buffer, so the padding belongs to no one else.
const COPY_ALIGN: usize = 4;

/// Staging budget for one batched readback. Bounds peak staging while keeping
/// the number of device waits at ceil(total / this) rather than one per region.
const D2H_STAGE_MAX: usize = 32 << 20;

fn alignUp(n: usize) usize {
    return (n + COPY_ALIGN - 1) / COPY_ALIGN * COPY_ALIGN;
}

/// Walks D2H regions in pieces that fit what is left of a staging budget. A
/// partial piece is a whole number of words, since the budget left always is.
const PieceCursor = struct {
    region: usize = 0,
    at: usize = 0,

    const Piece = struct { region: usize, at: usize, len: usize };

    fn next(self: *PieceCursor, regions: []const dm.D2HRegion, room: usize) ?Piece {
        while (self.region < regions.len and self.at == regions[self.region].dst.len) {
            self.region += 1;
            self.at = 0;
        }
        if (self.region == regions.len or room == 0) return null;
        const piece: Piece = .{ .region = self.region, .at = self.at, .len = @min(regions[self.region].dst.len - self.at, room) };
        self.at += piece.len;
        return piece;
    }
};

const dm = @import("../../runtime/device_memory.zig");
const profile = @import("../../profile.zig");
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
            if (maybe) |buf| fns.wgpuBufferRelease(buf);
        }
        if (self.staging) |s| fns.wgpuBufferRelease(s);
        self.buffers.deinit(self.allocator);
        self.* = undefined;
    }

    /// Ensure the pooled staging buffer holds at least `bytes`, growing (with some
    /// slack, rounded to 1 MiB) if needed. Returns the staging buffer.
    fn ensureStaging(self: *Self, bytes: usize) DeviceError!c.WGPUBuffer {
        if (self.staging) |s| {
            if (self.staging_cap >= bytes) return s;
            fns.wgpuBufferRelease(s);
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
        const buf = wgpu.createBuffer(self.gpu.device, @intCast(alignUp(bytes)), usage) catch return DeviceError.OutOfDeviceMemory;
        self.buffers.append(self.allocator, buf) catch {
            fns.wgpuBufferRelease(buf);
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
            fns.wgpuBufferRelease(buf);
            self.buffers.items[slot] = null;
        }
    }

    fn copyH2D(ctx: *anyopaque, handle: DeviceHandle, dst_offset: usize, src: []const u8) DeviceError!void {
        const t0: u64 = if (profile.capture_transfers) profile.nowNs() else 0;
        defer if (profile.capture_transfers) profile.recordTransfer(.h2d, src.len, profile.nowNs() - t0);
        const self: *Self = @ptrCast(@alignCast(ctx));
        const buf = self.bufFor(handle) orelse return DeviceError.InvalidArgument;
        if (dst_offset % COPY_ALIGN != 0) return DeviceError.InvalidArgument;
        // wgpu stages each write whole, so a tile goes in pieces: one large weight
        // must not hold a staging copy of itself beside its host and device ones.
        const head = src.len / COPY_ALIGN * COPY_ALIGN;
        var at: usize = 0;
        while (at < head) {
            const n = @min(head - at, H2D_FLUSH_BYTES);
            fns.wgpuQueueWriteBuffer(self.gpu.queue, buf, @intCast(dst_offset + at), src.ptr + at, n);
            self.noteStaged(n);
            at += n;
        }
        // A trailing 1..3 bytes go as one zero-padded word (`alloc` left room).
        if (head != src.len) {
            var word: [COPY_ALIGN]u8 = @splat(0);
            @memcpy(word[0 .. src.len - head], src[head..]);
            fns.wgpuQueueWriteBuffer(self.gpu.queue, buf, @intCast(dst_offset + head), &word, word.len);
            self.noteStaged(word.len);
        }
    }

    /// Bound wgpu's write-staging: an empty submit flushes the pending writes, the
    /// wait lets wgpu recycle their staging buffers. Ordering is unaffected
    /// (writeBuffer data was already ordered before any later submit), so this is
    /// safe even mid-frame.
    fn noteStaged(self: *Self, bytes: usize) void {
        self.h2d_since_flush += bytes;
        if (self.h2d_since_flush >= H2D_FLUSH_BYTES) {
            fns.wgpuQueueSubmit(self.gpu.queue, 0, null);
            _ = fns.wgpuDevicePoll(self.gpu.device, 1, null);
            self.h2d_since_flush = 0;
        }
    }

    fn copyD2H(ctx: *anyopaque, dst: []u8, handle: DeviceHandle, src_offset: usize) DeviceError!void {
        const one = [_]dm.D2HRegion{.{ .dst = dst, .handle = handle, .src_offset = src_offset }};
        return copyD2HMany(ctx, &one);
    }

    /// Read any set of device regions in as few waits as the staging budget
    /// allows. Mapping a buffer costs a full device round trip regardless of
    /// size, so the batch — not the region — is what we sync on. Regions need
    /// not be related, ordered, or contiguous; one larger than the budget is read
    /// in budget-sized pieces, so the pooled staging buffer never outgrows it.
    fn copyD2HMany(ctx: *anyopaque, regions: []const dm.D2HRegion) DeviceError!void {
        const t0: u64 = if (profile.capture_transfers) profile.nowNs() else 0;
        defer if (profile.capture_transfers) {
            var moved: usize = 0;
            for (regions) |r| moved += r.dst.len;
            profile.recordTransfer(.d2h, moved, profile.nowNs() - t0);
        };
        const self: *Self = @ptrCast(@alignCast(ctx));
        for (regions) |r| if (r.src_offset % COPY_ALIGN != 0) return DeviceError.InvalidArgument;

        var cur: PieceCursor = .{};
        while (true) {
            // Plan one chunk; the copy and read passes replay it from `start`.
            const start = cur;
            var staged: usize = 0;
            while (cur.next(regions, D2H_STAGE_MAX - staged)) |piece| staged += alignUp(piece.len);
            if (staged == 0) return;
            const staging = try self.ensureStaging(staged);

            const enc = fns.wgpuDeviceCreateCommandEncoder(self.gpu.device, null);
            var walk = start;
            var off: usize = 0;
            while (off < staged) {
                const piece = walk.next(regions, D2H_STAGE_MAX - off).?;
                const r = regions[piece.region];
                const buf = self.bufFor(r.handle) orelse {
                    fns.wgpuCommandEncoderRelease(enc);
                    return DeviceError.InvalidArgument;
                };
                fns.wgpuCommandEncoderCopyBufferToBuffer(enc, buf, @intCast(r.src_offset + piece.at), staging, @intCast(off), alignUp(piece.len));
                off += alignUp(piece.len);
            }
            const cmd = fns.wgpuCommandEncoderFinish(enc, null);
            fns.wgpuCommandEncoderRelease(enc);
            fns.wgpuQueueSubmit(self.gpu.queue, 1, &cmd);
            fns.wgpuCommandBufferRelease(cmd);

            // One wait covers the compute that produced these buffers AND every
            // copy in this chunk, so the batch costs a single round trip.
            self.gpu.mapBlocking(staging, c.WGPUMapMode_Read, 0, staged) catch return DeviceError.InvalidArgument;
            const mapped = fns.wgpuBufferGetConstMappedRange(staging, 0, staged) orelse return DeviceError.InvalidArgument;
            const base: [*]const u8 = @ptrCast(mapped);
            walk = start;
            off = 0;
            while (off < staged) {
                const piece = walk.next(regions, D2H_STAGE_MAX - off).?;
                self.copyHostBytes(regions[piece.region].dst[piece.at..][0..piece.len], base[off .. off + piece.len]);
                off += alignUp(piece.len);
            }
            fns.wgpuBufferUnmap(staging);
        }
    }

    fn copyD2D(ctx: *anyopaque, dst: DeviceHandle, dst_offset: usize, src: DeviceHandle, src_offset: usize, bytes: usize) DeviceError!void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        if (bytes == 0) return;
        const dst_buf = self.bufFor(dst) orelse return DeviceError.InvalidArgument;
        const src_buf = self.bufFor(src) orelse return DeviceError.InvalidArgument;

        // Ordered after any prior submit, so a caller that just wrote `src`
        // through the queue sees those bytes without an explicit wait.
        const enc = fns.wgpuDeviceCreateCommandEncoder(self.gpu.device, null);
        fns.wgpuCommandEncoderCopyBufferToBuffer(enc, src_buf, @intCast(src_offset), dst_buf, @intCast(dst_offset), bytes);
        const cmd = fns.wgpuCommandEncoderFinish(enc, null);
        fns.wgpuCommandEncoderRelease(enc);
        fns.wgpuQueueSubmit(self.gpu.queue, 1, &cmd);
        fns.wgpuCommandBufferRelease(cmd);
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
        .copyD2HMany = copyD2HMany,
        .copyD2D = copyD2D,
        // importHost stays null: discrete memory has no host aliasing.
        .maxBindingBytes = maxBindingBytes,
    };
};
