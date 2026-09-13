// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

pub const Origin = struct { output: u32, operation: []const u8 };

pub const Phase = enum(u32) { none, validation, lowering, execution };

/// Per-thread error detail, like `errno`: one context may be driven from several
/// threads, and a failure belongs to whoever made the call. Fixed storage keeps
/// reporting allocation-free and outlives the temporary graph that produced it.
/// Workers never record; a failed job is re-raised by the submitting thread.
pub const Diagnostic = struct {
    phase: Phase = .none,
    output: u32 = std.math.maxInt(u32),
    code: []const u8 = "",
    operation: []const u8 = "",
    message_buf: [768]u8 = @splat(0),
    message_len: usize = 0,

    /// Runs before every execution, so it resets the fields rather than zeroing
    /// the message buffer: `message()` is bounded by `message_len`.
    pub fn clear(self: *Diagnostic) void {
        self.phase = .none;
        self.output = std.math.maxInt(u32);
        self.code = "";
        self.operation = "";
        self.message_len = 0;
    }

    pub fn message(self: *const Diagnostic) []const u8 {
        return self.message_buf[0..self.message_len];
    }

    pub fn recordStep(self: *Diagnostic, backend: []const u8, step: anytype, err: anyerror) void {
        if (self.phase != .none) return;
        self.phase = .execution;
        self.code = @errorName(err);
        self.operation = if (step.origin) |origin| origin.operation else @tagName(step.op);
        self.output = if (step.origin) |origin| origin.output else std.math.maxInt(u32);
        const text = std.fmt.bufPrint(&self.message_buf, "execution: {s} producing value {d}: {s}; backend={s}, step={s}", .{ self.operation, self.output, self.code, backend, @tagName(step.op) }) catch self.message_buf[0..];
        self.message_len = text.len;
    }

    pub fn recordGraph(self: *Diagnostic, phase: Phase, graph: anytype, node: anytype, err: anyerror) void {
        // A nested region's failure is more specific than its enclosing If/Loop.
        if (self.phase != .none) return;
        self.phase = phase;
        self.output = node.output;
        self.code = @errorName(err);
        self.operation = @tagName(node.op);
        const prefix = std.fmt.bufPrint(&self.message_buf, "{s}: {s} producing value {d}: {s}", .{ @tagName(phase), self.operation, node.output, self.code }) catch self.message_buf[0..];
        self.message_len = prefix.len;
        for (node.inputs, 0..) |id, i| {
            if (id >= graph.values.items.len) continue;
            const value = graph.values.items[id];
            const detail = std.fmt.bufPrint(self.message_buf[self.message_len..], "; input[{d}] value {d} dtype={s} shape={any}", .{ i, id, if (value.dtype) |dt| @tagName(dt) else "unknown", value.shape }) catch break;
            self.message_len += detail.len;
        }
        switch (node.op) {
            .Gather => |g| {
                const detail = std.fmt.bufPrint(self.message_buf[self.message_len..], "; axis={d}, batch_dims={d}", .{ g.axis, g.batch_dims }) catch return;
                self.message_len += detail.len;
            },
            .MaxPool2D => |p| {
                const detail = std.fmt.bufPrint(self.message_buf[self.message_len..], "; NHWC kernel=[{d},{d}] stride=[{d},{d}] dilation=[{d},{d}] pads=[{d},{d},{d},{d}] ceil={}", .{ p.kernel_h, p.kernel_w, p.stride_h, p.stride_w, p.dilation_h, p.dilation_w, p.pad_top, p.pad_bottom, p.pad_left, p.pad_right, p.ceil_mode }) catch return;
                self.message_len += detail.len;
            },
            else => {},
        }
    }
};

/// The calling thread's slot. Recording is unconditional and allocation-free, so
/// no caller has to own, thread through, or opt into a diagnostic.
threadlocal var slot: Diagnostic = .{};

pub fn current() *Diagnostic {
    return &slot;
}

test "diagnostics are per-thread" {
    const Worker = struct {
        fn run(seen: *bool) void {
            seen.* = current().phase == .none;
            current().phase = .execution;
        }
    };
    current().clear();
    current().phase = .validation;
    var seen_clean = false;
    const t = try std.Thread.spawn(.{}, Worker.run, .{&seen_clean});
    t.join();
    // The worker started clean and its write stayed on its own thread.
    try std.testing.expect(seen_clean);
    try std.testing.expectEqual(Phase.validation, current().phase);
    current().clear();
}
