// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// Low-overhead GPU profiling using timestamp queries embedded in the normal
// command stream. No per-dispatch submit or poll: results are resolved beside
// each normal frame submission and read once after the run completes.

const std = @import("std");
const wgpu = @import("wgpu.zig");
const pipelines = @import("pipelines.zig");
const profile = @import("../../profile.zig");
const c = wgpu.c;
const fns = wgpu.fns; // runtime wgpu dispatch table (functions)

pub const Mode = enum {
    /// One timestamp pair per compute pass; preserves normal dispatch batching.
    pass,
    /// One timestamp pair per dispatch; precise attribution, but isolates passes.
    dispatch,
};

pub const Sample = struct {
    first_query: u32,
    category: profile.Category,
    name: []const u8,
    operation: []const u8,
    detail: profile.Detail,
};

pub const TimestampProfiler = struct {
    allocator: std.mem.Allocator,
    gpu: *wgpu.Gpu,
    query_set: c.WGPUQuerySet,
    resolve_buffer: c.WGPUBuffer,
    read_buffer: c.WGPUBuffer,
    query_capacity: u32,
    session: *profile.Session,
    track: profile.TrackId,
    mode: Mode,
    query_count: u32 = 0,
    dropped: u64 = 0,
    samples: std.ArrayList(Sample) = .empty,

    pub fn init(allocator: std.mem.Allocator, gpu: *wgpu.Gpu, query_capacity: u32, session: *profile.Session, track: profile.TrackId, mode: Mode) !TimestampProfiler {
        // WebGPU caps a timestamp QuerySet at 4096 entries.
        const cap = @min(@as(u32, 4096), @max(@as(u32, 2), query_capacity & ~@as(u32, 1)));
        var qd: c.WGPUQuerySetDescriptor = std.mem.zeroes(c.WGPUQuerySetDescriptor);
        qd.type = c.WGPUQueryType_Timestamp;
        qd.count = cap;
        const qs = fns.wgpuDeviceCreateQuerySet(gpu.device, &qd) orelse return error.ExecutionFailed;
        errdefer fns.wgpuQuerySetRelease(qs);
        const bytes = @as(u64, cap) * @sizeOf(u64);
        const resolve = wgpu.createBuffer(gpu.device, bytes, c.WGPUBufferUsage_QueryResolve | c.WGPUBufferUsage_CopySrc) catch return error.ExecutionFailed;
        errdefer fns.wgpuBufferRelease(resolve);
        const read = wgpu.createBuffer(gpu.device, bytes, c.WGPUBufferUsage_MapRead | c.WGPUBufferUsage_CopyDst) catch return error.ExecutionFailed;
        errdefer fns.wgpuBufferRelease(read);
        var out: TimestampProfiler = .{ .allocator = allocator, .gpu = gpu, .query_set = qs, .resolve_buffer = resolve, .read_buffer = read, .query_capacity = cap, .session = session, .track = track, .mode = mode };
        try out.samples.ensureTotalCapacity(allocator, cap / 2);
        return out;
    }

    pub fn deinit(self: *TimestampProfiler) void {
        self.samples.deinit(self.allocator);
        fns.wgpuBufferRelease(self.read_buffer);
        fns.wgpuBufferRelease(self.resolve_buffer);
        fns.wgpuQuerySetRelease(self.query_set);
        self.* = undefined;
    }

    /// Reserve a begin/end pair and retain enough identity to compare the same
    /// kernel shape across runs. Returns null after capacity is exhausted.
    pub fn reserveDispatch(self: *TimestampProfiler, built: pipelines.Built, op: []const u8, groups: [3]u32, uniform_bytes: []const u8, buffer_sizes: []const u64) ?u32 {
        if (self.query_count + 2 > self.query_capacity) {
            self.dropped += 1;
            return null;
        }
        const first = self.query_count;
        var bound_bytes: u64 = 0;
        for (buffer_sizes) |size| bound_bytes +|= size;
        self.samples.append(self.allocator, .{
            .first_query = first,
            .category = .kernel,
            .name = built.kernel_name,
            .operation = op,
            .detail = .{ .dispatch = .{
                .entry = built.entry_name,
                .groups = groups,
                .signature = std.hash.Wyhash.hash(0, uniform_bytes),
                .bound_bytes = bound_bytes,
            } },
        }) catch {
            self.dropped += 1;
            return null;
        };
        self.query_count += 2;
        return first;
    }

    /// Reserve one begin/end pair for a normally batched compute pass.
    pub fn reservePass(self: *TimestampProfiler, op: []const u8) ?u32 {
        if (self.query_count + 2 > self.query_capacity) {
            self.dropped += 1;
            return null;
        }
        const first = self.query_count;
        self.samples.append(self.allocator, .{
            .first_query = first,
            .category = .phase,
            .name = "compute pass",
            .operation = op,
            .detail = .none,
        }) catch {
            self.dropped += 1;
            return null;
        };
        self.query_count += 2;
        return first;
    }

    /// Resolve all queries once in the final encoder. Earlier command buffers
    /// may write the same QuerySet; queue ordering makes those values visible.
    pub fn resolvePending(self: *TimestampProfiler, encoder: c.WGPUCommandEncoder) void {
        const count = self.query_count;
        if (count == 0) return;
        const bytes = @as(u64, count) * @sizeOf(u64);
        fns.wgpuCommandEncoderResolveQuerySet(encoder, self.query_set, 0, count, self.resolve_buffer, 0);
        fns.wgpuCommandEncoderCopyBufferToBuffer(encoder, self.resolve_buffer, 0, self.read_buffer, 0, bytes);
    }

    /// Resolve device timestamps into the backend-neutral session. The raw GPU
    /// clock remains a separate track; only durations and ordering within this
    /// clock domain are claimed.
    pub fn readResults(self: *TimestampProfiler) !void {
        if (self.query_count == 0) return;
        _ = fns.wgpuDevicePoll(self.gpu.device, 1, null);
        const bytes: usize = @as(usize, self.query_count) * @sizeOf(u64);
        try self.gpu.mapBlocking(self.read_buffer, c.WGPUMapMode_Read, 0, bytes);
        const raw_ptr = fns.wgpuBufferGetConstMappedRange(self.read_buffer, 0, bytes) orelse return error.MapFailed;
        defer fns.wgpuBufferUnmap(self.read_buffer);
        const ticks: [*]const u64 = @ptrCast(@alignCast(raw_ptr));
        const period: f64 = @floatCast(fns.wgpuQueueGetTimestampPeriod(self.gpu.queue));

        for (self.samples.items) |s| {
            const begin = ticks[s.first_query];
            const end = ticks[s.first_query + 1];
            if (end < begin) continue;
            self.session.record(.{
                .track = self.track,
                .category = s.category,
                .name = s.name,
                .operation = s.operation,
                .start_ns = @intFromFloat(@as(f64, @floatFromInt(begin)) * period),
                .duration_ns = @intFromFloat(@as(f64, @floatFromInt(end - begin)) * period),
                .detail = s.detail,
            });
        }
        self.session.noteDropped(self.dropped);
    }
};
