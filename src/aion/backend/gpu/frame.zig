// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! `Frame` — records a whole `executeProgram` into ONE command encoder and
//! submits it ONCE.
//!
//! The v0 backend created an encoder, a bind group, and a queue submit per
//! dispatch. Instead, a `Frame` opens a single encoder and keeps one compute pass
//! open across adjacent dispatches. Copies and host-visible flushes close it
//! because they are encoder-level operations. Timestamp attribution isolates
//! dispatches because portable WebGPU only exposes pass-boundary timestamps.
//! Per-dispatch transients (uniform buffers carrying kernel params, and the
//! bind groups) are owned by the frame and released after submit — the queue keeps
//! its own references alive until the GPU is done.
//!
//! Host reads mid-program (control-flow predicates) need a SUBMIT/FLUSH SPLIT:
//! the backend submits the pending frame, waits for the GPU, host-reads the
//! predicate, then continues recording into a FRESH frame. `records` counts the
//! commands recorded since init so the backend can skip the round-trip when
//! nothing is pending (see `GpuBackend`'s If/Loop handling).

const std = @import("std");
const wgpu = @import("wgpu.zig");
const pipelines = @import("pipelines.zig");
const TimestampProfiler = @import("timestamp_profile.zig").TimestampProfiler;

const c = wgpu.c;
const fns = wgpu.fns; // runtime wgpu dispatch table (functions)
const Built = pipelines.Built;

pub const FrameError = error{ExecutionFailed};

/// A grow-only ring of same-sized uniform buffers, reused across `executeProgram`
/// runs. Each decode step records ~700 tiny per-dispatch uniforms; without a pool
/// that is ~700 `wgpuDeviceCreateBuffer` + release driver calls per step, a large
/// chunk of the CPU record cost. The pool hands buffers out by a cursor that the
/// backend `reset`s at the start of each run — safe to reuse because the previous
/// run's final readback poll guarantees its GPU work (and thus all its uniform
/// reads) completed. Within a run the cursor only advances (never reuses an
/// in-flight buffer), growing the pool to the run's dispatch count.
pub const UniformPool = struct {
    gpu: *wgpu.Gpu,
    allocator: std.mem.Allocator,
    buffers: std.ArrayList(c.WGPUBuffer) = .empty,
    cursor: usize = 0,
    /// Every uniform in these kernels is <= 96 bytes; one fixed size fits all and
    /// lets any pooled buffer serve any dispatch.
    pub const SLOT_BYTES: u64 = 256;

    pub fn reset(self: *UniformPool) void {
        self.cursor = 0;
    }

    /// Next free pooled buffer (creates one if the cursor passed the high-water
    /// mark). Caller writes its params and must NOT release it.
    pub fn next(self: *UniformPool) FrameError!c.WGPUBuffer {
        if (self.cursor < self.buffers.items.len) {
            const b = self.buffers.items[self.cursor];
            self.cursor += 1;
            return b;
        }
        const b = wgpu.createUniformBuffer(self.gpu.device, SLOT_BYTES) catch return error.ExecutionFailed;
        self.buffers.append(self.allocator, b) catch {
            fns.wgpuBufferRelease(b);
            return error.ExecutionFailed;
        };
        self.cursor += 1;
        return b;
    }

    pub fn deinit(self: *UniformPool) void {
        for (self.buffers.items) |b| fns.wgpuBufferRelease(b);
        self.buffers.deinit(self.allocator);
        self.* = undefined;
    }
};

/// Caches the (uniform buffer + bind group) for each dispatch, reused across
/// `executeProgram` runs. The decode program is IDENTICAL every step — same
/// pipelines, same resident device buffers, same params — so a bind group built
/// once can be replayed, skipping the ~700 `wgpuDeviceCreateBindGroup` (+ uniform
/// create/write) driver calls that otherwise dominate the CPU record cost of a
/// step. Entries are keyed positionally by a per-run cursor (the backend `reset`s
/// it each run); every reuse VALIDATES pipeline + buffer handles + sizes + uniform
/// bytes, so a changed program / migrated buffer / changed param safely rebuilds
/// that entry (self-correcting, never stale). Bounded: steady-state decode reuses
/// the same ~N entries forever.
pub const BindGroupCache = struct {
    gpu: *wgpu.Gpu,
    allocator: std.mem.Allocator,
    entries: std.ArrayList(Entry) = .empty,
    cursor: usize = 0,

    const MAX_B = 10;
    const Entry = struct {
        pipeline: c.WGPUComputePipeline,
        bind_group: c.WGPUBindGroup,
        uniform_buf: c.WGPUBuffer,
        n: usize,
        bufs: [MAX_B]c.WGPUBuffer,
        sizes: [MAX_B]u64,
        ubytes: [UniformPool.SLOT_BYTES]u8,
        ulen: usize,
        ub_size: u64,
    };

    pub fn reset(self: *BindGroupCache) void {
        self.cursor = 0;
    }

    pub fn deinit(self: *BindGroupCache) void {
        for (self.entries.items) |e| {
            fns.wgpuBindGroupRelease(e.bind_group);
            fns.wgpuBufferRelease(e.uniform_buf);
        }
        self.entries.deinit(self.allocator);
        self.* = undefined;
    }

    fn matches(e: *const Entry, built: Built, buffers: []const c.WGPUBuffer, sizes: []const u64, ubytes: []const u8) bool {
        if (e.pipeline != built.pipeline or e.n != buffers.len or e.ulen != ubytes.len) return false;
        for (buffers, 0..) |b, i| {
            if (e.bufs[i] != b or e.sizes[i] != sizes[i]) return false;
        }
        return std.mem.eql(u8, e.ubytes[0..e.ulen], ubytes);
    }

    fn buildEntry(self: *BindGroupCache, built: Built, buffers: []const c.WGPUBuffer, sizes: []const u64, ubytes: []const u8, ub_size: u64) FrameError!Entry {
        const ub = wgpu.createUniformBuffer(self.gpu.device, UniformPool.SLOT_BYTES) catch return error.ExecutionFailed;
        errdefer fns.wgpuBufferRelease(ub);
        fns.wgpuQueueWriteBuffer(self.gpu.queue, ub, 0, ubytes.ptr, ubytes.len);

        var entries: [MAX_B]c.WGPUBindGroupEntry = undefined;
        for (buffers, 0..) |buf, i| {
            entries[i] = std.mem.zeroes(c.WGPUBindGroupEntry);
            entries[i].binding = @intCast(i);
            entries[i].buffer = buf;
            entries[i].size = sizes[i];
        }
        entries[buffers.len] = std.mem.zeroes(c.WGPUBindGroupEntry);
        entries[buffers.len].binding = @intCast(buffers.len);
        entries[buffers.len].buffer = ub;
        entries[buffers.len].size = ub_size;

        var bgd: c.WGPUBindGroupDescriptor = std.mem.zeroes(c.WGPUBindGroupDescriptor);
        bgd.layout = built.bgl;
        bgd.entryCount = buffers.len + 1;
        bgd.entries = &entries;
        const bg = fns.wgpuDeviceCreateBindGroup(self.gpu.device, &bgd) orelse return error.ExecutionFailed;

        var e: Entry = .{
            .pipeline = built.pipeline,
            .bind_group = bg,
            .uniform_buf = ub,
            .n = buffers.len,
            .bufs = undefined,
            .sizes = undefined,
            .ubytes = undefined,
            .ulen = ubytes.len,
            .ub_size = ub_size,
        };
        for (buffers, 0..) |b, i| {
            e.bufs[i] = b;
            e.sizes[i] = sizes[i];
        }
        @memcpy(e.ubytes[0..ubytes.len], ubytes);
        return e;
    }

    /// Return the bind group for this dispatch, building + caching (or rebuilding
    /// on a validation miss) as needed. Advances the cursor.
    pub fn acquire(self: *BindGroupCache, built: Built, buffers: []const c.WGPUBuffer, sizes: []const u64, ubytes: []const u8, ub_size: u64) FrameError!c.WGPUBindGroup {
        const idx = self.cursor;
        if (idx < self.entries.items.len) {
            const e = &self.entries.items[idx];
            if (matches(e, built, buffers, sizes, ubytes)) {
                self.cursor += 1;
                return e.bind_group;
            }
            // Mismatch: rebuild this slot in place.
            const ne = try self.buildEntry(built, buffers, sizes, ubytes, ub_size);
            fns.wgpuBindGroupRelease(e.bind_group);
            fns.wgpuBufferRelease(e.uniform_buf);
            e.* = ne;
            self.cursor += 1;
            return ne.bind_group;
        }
        const ne = try self.buildEntry(built, buffers, sizes, ubytes, ub_size);
        self.entries.append(self.allocator, ne) catch {
            fns.wgpuBindGroupRelease(ne.bind_group);
            fns.wgpuBufferRelease(ne.uniform_buf);
            return error.ExecutionFailed;
        };
        self.cursor += 1;
        return ne.bind_group;
    }
};

pub const Frame = struct {
    gpu: *wgpu.Gpu,
    allocator: std.mem.Allocator,
    encoder: c.WGPUCommandEncoder,
    /// Open across adjacent unprofiled dispatches to avoid a pass boundary per
    /// kernel. Null while recording copies and in timestamp-attribution mode.
    compute_pass: c.WGPUComputePassEncoder = null,
    transient_buffers: std.ArrayList(c.WGPUBuffer),
    transient_groups: std.ArrayList(c.WGPUBindGroup),
    /// Optional backend-owned uniform pool. When set, per-dispatch uniforms are
    /// borrowed from it (not created/released per call). Null for standalone
    /// frames (autotune/bench), which keep the create-and-release path.
    uniform_pool: ?*UniformPool = null,
    /// Optional backend-owned bind-group cache. When set, per-dispatch uniforms
    /// AND bind groups are reused across runs (see `BindGroupCache`). Takes
    /// precedence over `uniform_pool`. Null for standalone frames.
    bind_cache: ?*BindGroupCache = null,
    /// Optional timestamp collector. It only inserts query writes into the
    /// existing passes; resolution follows normal frame submissions.
    timestamps: ?*TimestampProfiler = null,
    /// Leaf executable step currently recording dispatches.
    profile_op: []const u8 = "unknown",
    /// Commands recorded since init — lets the backend skip a submit/sync when
    /// a control-flow host read finds nothing pending.
    records: usize = 0,
    /// Count of `flushInPlace` submits over this frame's lifetime (each a
    /// pipeline break forced by control flow or a record-time host read).
    /// Diagnostic only (the profiler reports it).
    submits: usize = 0,

    const Self = @This();
    // 9 storage buffers + the trailing uniform. WebGPU's default
    // maxStorageBuffersPerShaderStage is 8, which the widest kernels
    // (RelPosMHA, LSTM) hit exactly.
    const MAX_BINDINGS = 10;

    pub fn init(allocator: std.mem.Allocator, gpu: *wgpu.Gpu) FrameError!Self {
        const enc = fns.wgpuDeviceCreateCommandEncoder(gpu.device, null) orelse return error.ExecutionFailed;
        return .{
            .gpu = gpu,
            .allocator = allocator,
            .encoder = enc,
            .transient_buffers = .empty,
            .transient_groups = .empty,
        };
    }

    /// Release transient resources (call after `submit`). Also releases the
    /// encoder if the frame was never submitted (error path).
    pub fn deinit(self: *Self) void {
        if (self.encoder != null) {
            self.endComputePass();
            fns.wgpuCommandEncoderRelease(self.encoder);
        }
        for (self.transient_groups.items) |g| fns.wgpuBindGroupRelease(g);
        for (self.transient_buffers.items) |b| fns.wgpuBufferRelease(b);
        self.transient_groups.deinit(self.allocator);
        self.transient_buffers.deinit(self.allocator);
        self.* = undefined;
    }

    /// Record one compute dispatch: bind `buffers` at successive bindings from 0,
    /// the per-dispatch uniform (`uniform_bytes`) at binding `buffers.len`, then
    /// dispatch `groups` workgroups. The uniform buffer is created and written
    /// here (sized up to the 16-byte uniform requirement).
    pub fn recordCompute(
        self: *Self,
        built: Built,
        buffers: []const c.WGPUBuffer,
        buf_sizes: []const u64,
        uniform_bytes: []const u8,
        groups: [3]u32,
    ) FrameError!void {
        std.debug.assert(buffers.len == buf_sizes.len);
        std.debug.assert(buffers.len + 1 <= MAX_BINDINGS);

        // Uniform buffer (params). Round size up to a multiple of 16.
        const ub_size: u64 = (@as(u64, uniform_bytes.len) + 15) & ~@as(u64, 15);

        // Fast path: reuse a cached (uniform + bind group) for this dispatch.
        if (self.bind_cache) |cache| {
            std.debug.assert(ub_size <= UniformPool.SLOT_BYTES);
            const bg = try cache.acquire(built, buffers, buf_sizes, uniform_bytes, ub_size);
            self.recordDispatch(built, bg, groups, uniform_bytes, buf_sizes);
            self.records += 1;
            return;
        }

        var ub: c.WGPUBuffer = undefined;
        if (self.uniform_pool) |pool| {
            std.debug.assert(ub_size <= UniformPool.SLOT_BYTES);
            ub = try pool.next(); // pool owns the buffer; not appended to transients
        } else {
            ub = wgpu.createUniformBuffer(self.gpu.device, ub_size) catch return error.ExecutionFailed;
            self.transient_buffers.append(self.allocator, ub) catch {
                fns.wgpuBufferRelease(ub);
                return error.ExecutionFailed;
            };
        }
        fns.wgpuQueueWriteBuffer(self.gpu.queue, ub, 0, uniform_bytes.ptr, uniform_bytes.len);

        // Bind group: storage buffers then the uniform at the trailing binding.
        var entries: [MAX_BINDINGS]c.WGPUBindGroupEntry = undefined;
        for (buffers, 0..) |buf, i| {
            entries[i] = std.mem.zeroes(c.WGPUBindGroupEntry);
            entries[i].binding = @intCast(i);
            entries[i].buffer = buf;
            entries[i].offset = 0;
            entries[i].size = buf_sizes[i];
        }
        entries[buffers.len] = std.mem.zeroes(c.WGPUBindGroupEntry);
        entries[buffers.len].binding = @intCast(buffers.len);
        entries[buffers.len].buffer = ub;
        entries[buffers.len].offset = 0;
        entries[buffers.len].size = ub_size;

        var bgd: c.WGPUBindGroupDescriptor = std.mem.zeroes(c.WGPUBindGroupDescriptor);
        bgd.layout = built.bgl;
        bgd.entryCount = buffers.len + 1;
        bgd.entries = &entries;
        const bg = fns.wgpuDeviceCreateBindGroup(self.gpu.device, &bgd) orelse return error.ExecutionFailed;
        self.transient_groups.append(self.allocator, bg) catch {
            fns.wgpuBindGroupRelease(bg);
            return error.ExecutionFailed;
        };

        self.recordDispatch(built, bg, groups, uniform_bytes, buf_sizes);
        self.records += 1;
    }

    fn recordDispatch(self: *Self, built: Built, bg: c.WGPUBindGroup, groups: [3]u32, uniform_bytes: []const u8, buffer_sizes: []const u64) void {
        // Portable timestamp attribution only exposes pass-boundary writes, so
        // attributed dispatches stay isolated. Normal execution batches them.
        const attributed = if (self.timestamps) |prof| prof.mode == .dispatch else false;
        const pass = if (attributed)
            self.beginAttributedPass(built, groups, uniform_bytes, buffer_sizes)
        else
            self.getComputePass();
        fns.wgpuComputePassEncoderSetPipeline(pass, built.pipeline);
        fns.wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, null);
        fns.wgpuComputePassEncoderDispatchWorkgroups(pass, groups[0], groups[1], groups[2]);
        if (attributed) {
            fns.wgpuComputePassEncoderEnd(pass);
            fns.wgpuComputePassEncoderRelease(pass);
        }
    }

    fn getComputePass(self: *Self) c.WGPUComputePassEncoder {
        if (self.compute_pass == null) {
            if (self.timestamps) |prof| {
                if (prof.mode == .pass) {
                    if (prof.reservePass(self.profile_op)) |first| {
                        var writes: c.WGPUPassTimestampWrites = std.mem.zeroes(c.WGPUPassTimestampWrites);
                        writes.querySet = prof.query_set;
                        writes.beginningOfPassWriteIndex = first;
                        writes.endOfPassWriteIndex = first + 1;
                        var desc: c.WGPUComputePassDescriptor = std.mem.zeroes(c.WGPUComputePassDescriptor);
                        desc.timestampWrites = &writes;
                        self.compute_pass = fns.wgpuCommandEncoderBeginComputePass(self.encoder, &desc);
                    }
                }
            }
            if (self.compute_pass == null)
                self.compute_pass = fns.wgpuCommandEncoderBeginComputePass(self.encoder, null);
        }
        return self.compute_pass;
    }

    fn endComputePass(self: *Self) void {
        const pass = self.compute_pass orelse return;
        fns.wgpuComputePassEncoderEnd(pass);
        fns.wgpuComputePassEncoderRelease(pass);
        self.compute_pass = null;
    }

    fn beginAttributedPass(self: *Self, built: Built, groups: [3]u32, uniform_bytes: []const u8, buffer_sizes: []const u64) c.WGPUComputePassEncoder {
        if (self.timestamps) |prof| {
            if (prof.reserveDispatch(built, self.profile_op, groups, uniform_bytes, buffer_sizes)) |first| {
                var writes: c.WGPUPassTimestampWrites = std.mem.zeroes(c.WGPUPassTimestampWrites);
                writes.querySet = prof.query_set;
                writes.beginningOfPassWriteIndex = first;
                writes.endOfPassWriteIndex = first + 1;
                var desc: c.WGPUComputePassDescriptor = std.mem.zeroes(c.WGPUComputePassDescriptor);
                desc.timestampWrites = &writes;
                return fns.wgpuCommandEncoderBeginComputePass(self.encoder, &desc);
            }
        }
        return fns.wgpuCommandEncoderBeginComputePass(self.encoder, null);
    }

    /// Record a device buffer copy. Copies are encoder-level commands, ordered
    /// with the surrounding compute passes. WebGPU requires `bytes` and both
    /// offsets be multiples of 4 (callers check).
    pub fn recordCopy(self: *Self, src: c.WGPUBuffer, src_off: u64, dst: c.WGPUBuffer, dst_off: u64, bytes: u64) void {
        self.endComputePass();
        fns.wgpuCommandEncoderCopyBufferToBuffer(self.encoder, src, src_off, dst, dst_off, bytes);
        self.records += 1;
    }

    /// Submit whatever is pending and continue recording into a FRESH encoder,
    /// reusing the SAME `Frame` object (and its uniform pool / bind-group cache).
    /// No-op when nothing is recorded. Deliberately does NOT poll: queue
    /// submissions execute in order, so a later D2H readback (whose own poll
    /// waits for all prior submitted work) or a later frame sees these results.
    /// This is the single mechanism behind every mid-program flush — control-flow
    /// predicate reads, record-time host index reads, and the chunked-submit
    /// overlap all funnel through here.
    pub fn flushInPlace(self: *Self) FrameError!void {
        if (self.records == 0) return;
        // Timestamp queries survive queue submissions. Resolve them only in the
        // final encoder: partial resolve destinations require 256-byte alignment
        // and resolving once also minimizes profiler command overhead.
        const timestamps = self.timestamps;
        self.timestamps = null;
        self.submit(); // finishes + releases the encoder, sets encoder = null
        self.timestamps = timestamps;
        // The queue holds its own references to the submitted bind groups /
        // uniforms until the GPU is done, so releasing our handles now is safe.
        for (self.transient_groups.items) |g| fns.wgpuBindGroupRelease(g);
        for (self.transient_buffers.items) |b| fns.wgpuBufferRelease(b);
        self.transient_groups.clearRetainingCapacity();
        self.transient_buffers.clearRetainingCapacity();
        self.encoder = fns.wgpuDeviceCreateCommandEncoder(self.gpu.device, null) orelse return error.ExecutionFailed;
        self.compute_pass = null;
        self.records = 0;
        self.submits += 1;
    }

    /// Finish the encoder and submit the whole batch once.
    pub fn submit(self: *Self) void {
        self.endComputePass();
        if (self.timestamps) |prof| prof.resolvePending(self.encoder);
        const cmd = fns.wgpuCommandEncoderFinish(self.encoder, null);
        fns.wgpuCommandEncoderRelease(self.encoder);
        self.encoder = null; // deinit() must not double-release.
        fns.wgpuQueueSubmit(self.gpu.queue, 1, &cmd);
        fns.wgpuCommandBufferRelease(cmd);
    }
};
