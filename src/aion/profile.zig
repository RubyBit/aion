// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Small, backend-neutral profiling core.
//!
//! Producers record completed events on independent clock domains. CPU scopes
//! use the host clock directly; asynchronous devices translate their native
//! timestamps to nanoseconds before appending events. Reporting deliberately
//! stays textual and aggregates by default, so profiling remains useful over
//! SSH and in captured benchmark logs.

const std = @import("std");
const env = @import("env.zig");

pub const TrackId = enum(u16) { _ };

pub const Clock = enum {
    host,
    device,
};

pub const Category = enum {
    phase,
    operation,
    kernel,
    copy,
    wait,
    submit,
};

pub const Mode = enum {
    off,
    summary,
    timeline,
};

pub const Config = struct {
    mode: Mode = .off,
    /// Ignore this many program invocations before recording.
    skip: u64 = 0,
    /// Number of consecutive invocations to record after `skip`.
    count: u64 = 1,
    top: usize = 20,
    min_duration_ns: u64 = 0,
    event_capacity: usize = 4096,

    pub fn fromEnv() Config {
        var out: Config = .{};
        if (env.getOwned(std.heap.page_allocator, "AION_PROFILE")) |value| {
            defer std.heap.page_allocator.free(value);
            if (value.len == 0 or std.mem.eql(u8, value, "1") or std.ascii.eqlIgnoreCase(value, "summary")) {
                out.mode = .summary;
            } else if (std.ascii.eqlIgnoreCase(value, "timeline")) {
                out.mode = .timeline;
            }
        }
        if (env.getOwned(std.heap.page_allocator, "AION_PROFILE_SKIP")) |value| {
            defer std.heap.page_allocator.free(value);
            out.skip = std.fmt.parseInt(u64, value, 10) catch out.skip;
        }
        if (env.getOwned(std.heap.page_allocator, "AION_PROFILE_COUNT")) |value| {
            defer std.heap.page_allocator.free(value);
            out.count = std.fmt.parseInt(u64, value, 10) catch out.count;
        }
        if (env.getOwned(std.heap.page_allocator, "AION_PROFILE_TOP")) |value| {
            defer std.heap.page_allocator.free(value);
            out.top = std.fmt.parseInt(usize, value, 10) catch out.top;
        }
        if (env.getOwned(std.heap.page_allocator, "AION_PROFILE_MIN_US")) |value| {
            defer std.heap.page_allocator.free(value);
            const min_us = std.fmt.parseInt(u64, value, 10) catch 0;
            out.min_duration_ns = min_us *| 1000;
        }
        if (env.getOwned(std.heap.page_allocator, "AION_PROFILE_CAPACITY")) |value| {
            defer std.heap.page_allocator.free(value);
            out.event_capacity = std.fmt.parseInt(usize, value, 10) catch out.event_capacity;
        }
        return out;
    }

    pub fn captures(self: Config, invocation: u64) bool {
        if (self.mode == .off or invocation < self.skip) return false;
        return invocation - self.skip < self.count;
    }
};

pub const Dispatch = struct {
    entry: []const u8,
    groups: [3]u32,
    signature: u64,
    bound_bytes: u64,
};

pub const Detail = union(enum) {
    none,
    dispatch: Dispatch,
    bytes: u64,
};

pub const Event = struct {
    track: TrackId,
    category: Category,
    name: []const u8,
    /// Optional higher-level operation which caused a kernel/copy event.
    operation: ?[]const u8 = null,
    start_ns: u64,
    duration_ns: u64,
    detail: Detail = .none,
};

const Track = struct {
    name: []const u8,
    clock: Clock,
};

const Aggregate = struct {
    track: TrackId,
    category: Category,
    name: []const u8,
    operation: ?[]const u8,
    detail: Detail,
    duration_ns: u64 = 0,
    count: u64 = 0,
};

const Gap = struct {
    track: TrackId,
    duration_ns: u64,
    before: []const u8,
    after: []const u8,
};

pub const Session = struct {
    allocator: std.mem.Allocator,
    config: Config,
    name: []const u8,
    tracks: std.ArrayList(Track) = .empty,
    events: std.ArrayList(Event) = .empty,
    dropped: u64 = 0,
    mutex: std.atomic.Mutex = .unlocked,

    pub fn init(allocator: std.mem.Allocator, config: Config, name: []const u8) Session {
        var out: Session = .{ .allocator = allocator, .config = config, .name = name };
        out.events.ensureTotalCapacity(allocator, config.event_capacity) catch {};
        return out;
    }

    pub fn deinit(self: *Session) void {
        self.events.deinit(self.allocator);
        self.tracks.deinit(self.allocator);
        self.* = undefined;
    }

    /// Names passed to the profiler are borrowed and must outlive the session.
    pub fn addTrack(self: *Session, name: []const u8, clock: Clock) ?TrackId {
        self.lock();
        defer self.mutex.unlock();
        const raw = self.tracks.items.len;
        if (raw > std.math.maxInt(u16)) return null;
        self.tracks.append(self.allocator, .{ .name = name, .clock = clock }) catch {
            self.dropped += 1;
            return null;
        };
        return @enumFromInt(@as(u16, @intCast(raw)));
    }

    /// Recording failure never changes program execution; it is counted and
    /// surfaced in the report instead. Recording is safe from CPU worker
    /// threads; reporters must run after all producers have stopped.
    pub fn record(self: *Session, event: Event) void {
        self.lock();
        defer self.mutex.unlock();
        if (@intFromEnum(event.track) >= self.tracks.items.len) {
            self.dropped += 1;
            return;
        }
        self.events.append(self.allocator, event) catch {
            self.dropped += 1;
        };
    }

    pub fn noteDropped(self: *Session, count: u64) void {
        self.lock();
        defer self.mutex.unlock();
        self.dropped +|= count;
    }

    pub fn recordSpan(self: *Session, track: TrackId, category: Category, name: []const u8, start_ns: u64, end_ns: u64) void {
        self.record(.{
            .track = track,
            .category = category,
            .name = name,
            .start_ns = start_ns,
            .duration_ns = end_ns -| start_ns,
        });
    }

    fn lock(self: *Session) void {
        while (!self.mutex.tryLock()) std.atomic.spinLoopHint();
    }

    pub fn report(self: *const Session) void {
        std.debug.print("\n[aion-profile] {s} events={d} dropped={d}\n", .{ self.name, self.events.items.len, self.dropped });
        self.reportTracks();
        self.reportPhases();
        self.reportOperations();
        self.reportKernels();
        self.reportGaps();
        if (self.config.mode == .timeline) self.reportTimeline();
    }

    fn reportPhases(self: *const Session) void {
        var aggs: std.ArrayList(Aggregate) = .empty;
        defer aggs.deinit(self.allocator);
        for (self.events.items) |event| switch (event.category) {
            .phase, .wait, .submit, .copy => addAggregate(self.allocator, &aggs, .{
                .track = event.track,
                .category = event.category,
                .name = event.name,
                .operation = null,
                .detail = event.detail,
                .duration_ns = event.duration_ns,
                .count = 1,
            }) catch return,
            else => {},
        };
        sortAggregates(aggs.items);
        if (aggs.items.len == 0) return;
        std.debug.print("  phases:\n", .{});
        for (aggs.items[0..@min(self.config.top, aggs.items.len)]) |agg|
            std.debug.print("    {s:<12} {d:>9.3} ms  x{d:<5} {s:<7} {s}\n", .{ self.trackName(agg.track), ms(agg.duration_ns), agg.count, @tagName(agg.category), agg.name });
    }

    fn reportTracks(self: *const Session) void {
        for (self.tracks.items, 0..) |track, raw_id| {
            const id: TrackId = @enumFromInt(@as(u16, @intCast(raw_id)));
            var first: u64 = std.math.maxInt(u64);
            var last: u64 = 0;
            var count: usize = 0;
            for (self.events.items) |event| {
                if (event.track != id) continue;
                first = @min(first, event.start_ns);
                last = @max(last, event.start_ns +| event.duration_ns);
                count += 1;
            }
            if (count == 0) continue;
            const span = last -| first;
            const covered = self.coveredDuration(id);
            std.debug.print("  track {s:<12} clock={s:<6} span={d:>9.3} ms  covered={d:>9.3} ms  gaps={d:>9.3} ms  events={d}\n", .{
                track.name,
                @tagName(track.clock),
                ms(span),
                ms(covered),
                ms(span -| covered),
                count,
            });
        }
    }

    fn coveredDuration(self: *const Session, track: TrackId) u64 {
        // Events are normally emitted in time order. Selecting the next earliest
        // event avoids allocating/sorting a copy and also handles nested spans.
        var cursor: u64 = 0;
        var have_cursor = false;
        var covered: u64 = 0;
        var consumed: usize = 0;
        while (true) {
            var next: ?Event = null;
            for (self.events.items) |event| {
                if (event.track != track) continue;
                const end = event.start_ns +| event.duration_ns;
                if (have_cursor and end <= cursor) continue;
                if (next == null or event.start_ns < next.?.start_ns) next = event;
            }
            const event = next orelse break;
            const end = event.start_ns +| event.duration_ns;
            if (!have_cursor) {
                cursor = event.start_ns;
                have_cursor = true;
            }
            if (event.start_ns > cursor) cursor = event.start_ns;
            covered +|= end -| cursor;
            cursor = @max(cursor, end);
            consumed += 1;
            if (consumed > self.events.items.len) break;
        }
        return covered;
    }

    fn reportOperations(self: *const Session) void {
        var aggs: std.ArrayList(Aggregate) = .empty;
        defer aggs.deinit(self.allocator);
        for (self.events.items) |event| {
            const name = event.operation orelse if (event.category == .operation) event.name else continue;
            addAggregate(self.allocator, &aggs, .{
                .track = event.track,
                .category = .operation,
                .name = name,
                .operation = null,
                .detail = .none,
                .duration_ns = event.duration_ns,
                .count = 1,
            }) catch return;
        }
        sortAggregates(aggs.items);
        if (aggs.items.len == 0) return;
        std.debug.print("  operations:\n", .{});
        for (aggs.items[0..@min(self.config.top, aggs.items.len)]) |agg|
            std.debug.print("    {s:<12} {d:>9.3} ms  x{d:<5} {s}\n", .{ self.trackName(agg.track), ms(agg.duration_ns), agg.count, agg.name });
    }

    fn reportKernels(self: *const Session) void {
        var aggs: std.ArrayList(Aggregate) = .empty;
        defer aggs.deinit(self.allocator);
        for (self.events.items) |event| {
            if (event.category != .kernel) continue;
            addAggregate(self.allocator, &aggs, .{
                .track = event.track,
                .category = event.category,
                .name = event.name,
                .operation = event.operation,
                .detail = event.detail,
                .duration_ns = event.duration_ns,
                .count = 1,
            }) catch return;
        }
        sortAggregates(aggs.items);
        if (aggs.items.len == 0) return;
        std.debug.print("  kernels:\n", .{});
        for (aggs.items[0..@min(self.config.top, aggs.items.len)]) |agg| switch (agg.detail) {
            .dispatch => |d| std.debug.print("    {d:>9.3} ms  x{d:<5} {s}/{s} g={d}x{d}x{d} sig={x:0>16} bound={d:.3} MiB\n", .{
                ms(agg.duration_ns), agg.count, agg.name, d.entry, d.groups[0], d.groups[1], d.groups[2], d.signature, mib(d.bound_bytes),
            }),
            else => std.debug.print("    {d:>9.3} ms  x{d:<5} {s}\n", .{ ms(agg.duration_ns), agg.count, agg.name }),
        };
    }

    fn reportGaps(self: *const Session) void {
        var gaps: std.ArrayList(Gap) = .empty;
        defer gaps.deinit(self.allocator);
        for (self.tracks.items, 0..) |track, raw_id| {
            if (track.clock != .device) continue;
            const id: TrackId = @enumFromInt(@as(u16, @intCast(raw_id)));
            for (self.events.items) |event| {
                if (event.track != id) continue;
                const end = event.start_ns +| event.duration_ns;
                var next: ?Event = null;
                for (self.events.items) |candidate| {
                    if (candidate.track != id or candidate.start_ns < end) continue;
                    if (next == null or candidate.start_ns < next.?.start_ns) next = candidate;
                }
                if (next) |after| {
                    const duration = after.start_ns -| end;
                    if (duration > 0) gaps.append(self.allocator, .{
                        .track = id,
                        .duration_ns = duration,
                        .before = event.name,
                        .after = after.name,
                    }) catch return;
                }
            }
        }
        var i: usize = 1;
        while (i < gaps.items.len) : (i += 1) {
            var j = i;
            while (j > 0 and gaps.items[j].duration_ns > gaps.items[j - 1].duration_ns) : (j -= 1)
                std.mem.swap(Gap, &gaps.items[j], &gaps.items[j - 1]);
        }
        if (gaps.items.len == 0) return;
        std.debug.print("  largest device gaps:\n", .{});
        for (gaps.items[0..@min(self.config.top, gaps.items.len)]) |gap|
            std.debug.print("    {s:<12} {d:>9.3} us  {s} -> {s}\n", .{ self.trackName(gap.track), us(gap.duration_ns), gap.before, gap.after });
    }

    fn reportTimeline(self: *const Session) void {
        std.debug.print("  timeline (per-track relative time, min {d:.3} us):\n", .{us(self.config.min_duration_ns)});
        for (self.tracks.items, 0..) |track, raw_id| {
            const id: TrackId = @enumFromInt(@as(u16, @intCast(raw_id)));
            var origin: u64 = std.math.maxInt(u64);
            var order: std.ArrayList(usize) = .empty;
            defer order.deinit(self.allocator);
            for (self.events.items, 0..) |event, event_i| if (event.track == id) {
                origin = @min(origin, event.start_ns);
                order.append(self.allocator, event_i) catch return;
            };
            if (origin == std.math.maxInt(u64)) continue;
            var i: usize = 1;
            while (i < order.items.len) : (i += 1) {
                var j = i;
                while (j > 0 and self.events.items[order.items[j]].start_ns < self.events.items[order.items[j - 1]].start_ns) : (j -= 1)
                    std.mem.swap(usize, &order.items[j], &order.items[j - 1]);
            }
            for (order.items) |event_i| {
                const event = self.events.items[event_i];
                if (event.duration_ns < self.config.min_duration_ns) continue;
                std.debug.print("    {s:<12} {d:>9.3}-{d:>9.3} ms  {s:<9} {s}", .{
                    track.name,
                    ms(event.start_ns -| origin),
                    ms(event.start_ns -| origin +| event.duration_ns),
                    @tagName(event.category),
                    event.name,
                });
                if (event.operation) |op| std.debug.print("  op={s}", .{op});
                switch (event.detail) {
                    .dispatch => |d| std.debug.print("  entry={s} g={d}x{d}x{d}", .{ d.entry, d.groups[0], d.groups[1], d.groups[2] }),
                    .bytes => |bytes| std.debug.print("  bytes={d}", .{bytes}),
                    .none => {},
                }
                std.debug.print("\n", .{});
            }
        }
    }

    fn trackName(self: *const Session, id: TrackId) []const u8 {
        const index: usize = @intFromEnum(id);
        return if (index < self.tracks.items.len) self.tracks.items[index].name else "unknown";
    }
};

pub fn nowNs() u64 {
    const timestamp: std.Io.Timestamp = std.Io.Clock.awake.now(std.Options.debug_io);
    const ns: i96 = timestamp.toNanoseconds();
    if (ns <= 0) return 0;
    return @intCast(@min(ns, @as(i96, std.math.maxInt(u64))));
}

fn addAggregate(allocator: std.mem.Allocator, aggs: *std.ArrayList(Aggregate), incoming: Aggregate) !void {
    for (aggs.items) |*agg| {
        if (sameAggregate(agg.*, incoming)) {
            agg.duration_ns +|= incoming.duration_ns;
            agg.count +|= incoming.count;
            return;
        }
    }
    try aggs.append(allocator, incoming);
}

fn sameAggregate(a: Aggregate, b: Aggregate) bool {
    if (a.track != b.track or a.category != b.category or !std.mem.eql(u8, a.name, b.name)) return false;
    if ((a.operation == null) != (b.operation == null)) return false;
    if (a.operation) |op| if (!std.mem.eql(u8, op, b.operation.?)) return false;
    return switch (a.detail) {
        .none => b.detail == .none,
        .bytes => |bytes| switch (b.detail) { .bytes => |other| bytes == other, else => false },
        .dispatch => |d| switch (b.detail) {
            .dispatch => |other| std.mem.eql(u8, d.entry, other.entry) and d.groups[0] == other.groups[0] and d.groups[1] == other.groups[1] and d.groups[2] == other.groups[2] and d.signature == other.signature,
            else => false,
        },
    };
}

fn sortAggregates(items: []Aggregate) void {
    var i: usize = 1;
    while (i < items.len) : (i += 1) {
        var j = i;
        while (j > 0 and items[j].duration_ns > items[j - 1].duration_ns) : (j -= 1)
            std.mem.swap(Aggregate, &items[j], &items[j - 1]);
    }
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1.0e6;
}

fn us(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1.0e3;
}

fn mib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0);
}

test "profile config applies skip and count capture window" {
    const config: Config = .{ .mode = .summary, .skip = 2, .count = 2 };
    try std.testing.expect(!config.captures(0));
    try std.testing.expect(!config.captures(1));
    try std.testing.expect(config.captures(2));
    try std.testing.expect(config.captures(3));
    try std.testing.expect(!config.captures(4));
    try std.testing.expect(!(Config{}).captures(2));
}

test "profile session aggregates independent CPU and GPU tracks" {
    var session = Session.init(std.testing.allocator, .{ .mode = .summary }, "test");
    defer session.deinit();
    const cpu = session.addTrack("CPU", .host).?;
    const gpu = session.addTrack("GPU", .device).?;
    session.recordSpan(cpu, .operation, "MatMul", 10, 30);
    session.record(.{ .track = gpu, .category = .kernel, .name = "matmul", .operation = "MatMul", .start_ns = 100, .duration_ns = 50 });
    session.record(.{ .track = gpu, .category = .kernel, .name = "matmul", .operation = "MatMul", .start_ns = 160, .duration_ns = 40 });

    try std.testing.expectEqual(@as(usize, 2), session.tracks.items.len);
    try std.testing.expectEqual(@as(usize, 3), session.events.items.len);
    try std.testing.expectEqual(@as(u64, 20), session.coveredDuration(cpu));
    try std.testing.expectEqual(@as(u64, 90), session.coveredDuration(gpu));
}

test "profile session accepts concurrent CPU producers" {
    var session = Session.init(std.testing.allocator, .{ .mode = .summary }, "threads");
    defer session.deinit();
    const a = session.addTrack("CPU 0", .host).?;
    const b = session.addTrack("CPU 1", .host).?;

    const Producer = struct {
        session: *Session,
        track: TrackId,

        fn run(producer: *@This()) void {
            for (0..100) |i| producer.session.record(.{
                .track = producer.track,
                .category = .operation,
                .name = "work",
                .start_ns = i * 10,
                .duration_ns = 5,
            });
        }
    };
    var pa: Producer = .{ .session = &session, .track = a };
    var pb: Producer = .{ .session = &session, .track = b };
    const ta = try std.Thread.spawn(.{}, Producer.run, .{&pa});
    const tb = try std.Thread.spawn(.{}, Producer.run, .{&pb});
    ta.join();
    tb.join();

    try std.testing.expectEqual(@as(usize, 200), session.events.items.len);
    try std.testing.expectEqual(@as(u64, 0), session.dropped);
}
