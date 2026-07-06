// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! `Frame` — records a whole `executeProgram` into ONE command encoder and
//! submits it ONCE.
//!
//! The v0 backend created an encoder, a bind group, and a queue submit per
//! dispatch. Instead, a `Frame` opens a single encoder, each step records a
//! compute pass into it (WebGPU inserts barriers between passes, so dependent
//! dispatches stay correctly ordered), and `submit()` finishes + submits exactly
//! once. Per-dispatch transients (uniform buffers carrying kernel params, and the
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

const c = wgpu.c;
const Built = pipelines.Built;

pub const FrameError = error{ExecutionFailed};

pub const Frame = struct {
    gpu: *wgpu.Gpu,
    allocator: std.mem.Allocator,
    encoder: c.WGPUCommandEncoder,
    transient_buffers: std.ArrayList(c.WGPUBuffer),
    transient_groups: std.ArrayList(c.WGPUBindGroup),
    /// Commands recorded since init — lets the backend skip a submit/sync when
    /// a control-flow host read finds nothing pending.
    records: usize = 0,

    const Self = @This();
    // 9 storage buffers + the trailing uniform. WebGPU's default
    // maxStorageBuffersPerShaderStage is 8, which the widest kernels
    // (RelPosMHA, LSTM) hit exactly.
    const MAX_BINDINGS = 10;

    pub fn init(allocator: std.mem.Allocator, gpu: *wgpu.Gpu) FrameError!Self {
        const enc = c.wgpuDeviceCreateCommandEncoder(gpu.device, null) orelse return error.ExecutionFailed;
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
        if (self.encoder != null) c.wgpuCommandEncoderRelease(self.encoder);
        for (self.transient_groups.items) |g| c.wgpuBindGroupRelease(g);
        for (self.transient_buffers.items) |b| c.wgpuBufferRelease(b);
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
        const ub = wgpu.createUniformBuffer(self.gpu.device, ub_size) catch return error.ExecutionFailed;
        self.transient_buffers.append(self.allocator, ub) catch {
            c.wgpuBufferRelease(ub);
            return error.ExecutionFailed;
        };
        c.wgpuQueueWriteBuffer(self.gpu.queue, ub, 0, uniform_bytes.ptr, uniform_bytes.len);

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
        const bg = c.wgpuDeviceCreateBindGroup(self.gpu.device, &bgd) orelse return error.ExecutionFailed;
        self.transient_groups.append(self.allocator, bg) catch {
            c.wgpuBindGroupRelease(bg);
            return error.ExecutionFailed;
        };

        const pass = c.wgpuCommandEncoderBeginComputePass(self.encoder, null);
        c.wgpuComputePassEncoderSetPipeline(pass, built.pipeline);
        c.wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, null);
        c.wgpuComputePassEncoderDispatchWorkgroups(pass, groups[0], groups[1], groups[2]);
        c.wgpuComputePassEncoderEnd(pass);
        c.wgpuComputePassEncoderRelease(pass);
        self.records += 1;
    }

    /// Record a device buffer copy. Copies are encoder-level commands, ordered
    /// with the surrounding compute passes. WebGPU requires `bytes` and both
    /// offsets be multiples of 4 (callers check).
    pub fn recordCopy(self: *Self, src: c.WGPUBuffer, src_off: u64, dst: c.WGPUBuffer, dst_off: u64, bytes: u64) void {
        c.wgpuCommandEncoderCopyBufferToBuffer(self.encoder, src, src_off, dst, dst_off, bytes);
        self.records += 1;
    }

    /// Finish the encoder and submit the whole batch once.
    pub fn submit(self: *Self) void {
        const cmd = c.wgpuCommandEncoderFinish(self.encoder, null);
        c.wgpuCommandEncoderRelease(self.encoder);
        self.encoder = null; // deinit() must not double-release.
        c.wgpuQueueSubmit(self.gpu.queue, 1, &cmd);
        c.wgpuCommandBufferRelease(cmd);
    }
};
