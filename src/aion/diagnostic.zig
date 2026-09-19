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
    message_buf: [768]u8 = undefined,
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
        self.message_len = 0;
        self.appendDetail("execution: {s} producing value {d}: {s}; backend={s}, step={s}", .{ self.operation, self.output, self.code, backend, @tagName(step.op) });
    }

    /// What disagreed when binding a model input. `got_*` stay null when nothing
    /// was bound at all, and `axis` names the dimension when only one disagrees.
    pub const InputMismatch = struct {
        code: []const u8,
        name: []const u8,
        want_dtype: []const u8,
        want_rank: usize,
        got_dtype: ?[]const u8 = null,
        got_shape: ?[]const usize = null,
        axis: ?usize = null,
    };

    /// Binding a model input is where most run-time failures start, and the
    /// caller knows that tensor by NAME, not by the value id the graph phases
    /// report.
    pub fn recordInput(self: *Diagnostic, m: InputMismatch) void {
        if (self.phase != .none) return;
        self.phase = .validation;
        self.code = m.code;
        self.operation = "bind_input";
        self.output = std.math.maxInt(u32);
        self.message_len = 0;
        self.appendDetail("validation: input \"{s}\": {s}; expected dtype={s} rank={d}", .{ m.name, m.code, m.want_dtype, m.want_rank });
        if (m.got_dtype) |dt| self.appendDetail(", got dtype={s}", .{dt});
        if (m.got_shape) |sh| {
            self.appendDetail(" shape={any}", .{sh});
        } else {
            self.appendDetail(", nothing bound", .{});
        }
        if (m.axis) |a| self.appendDetail("; axis {d} disagrees", .{a});
    }

    /// A name that matches no input: expected dtype/rank are meaningless here,
    /// so this says only what was asked for. The caller appends what exists.
    pub fn recordUnknownInput(self: *Diagnostic, name: []const u8) void {
        if (self.phase != .none) return;
        self.phase = .validation;
        self.code = "UnknownInput";
        self.operation = "bind_input";
        self.output = std.math.maxInt(u32);
        self.message_len = 0;
        self.appendDetail("validation: no input named \"{s}\"", .{name});
    }

    /// Append context to whatever was just recorded. Truncation is silent: a
    /// diagnostic must never fail, and the prefix is the part that matters.
    pub fn appendDetail(self: *Diagnostic, comptime fmt: []const u8, args: anytype) void {
        if (self.message_len >= self.message_buf.len) return;
        const extra = std.fmt.bufPrint(self.message_buf[self.message_len..], fmt, args) catch return;
        self.message_len += extra.len;
    }

    pub fn recordGraph(self: *Diagnostic, phase: Phase, graph: anytype, node: anytype, err: anyerror) void {
        // A nested region's failure is more specific than its enclosing If/Loop.
        if (self.phase != .none) return;
        self.phase = phase;
        self.output = node.output;
        self.code = @errorName(err);
        self.operation = @tagName(node.op);
        self.message_len = 0;

        // An author names their values; an integer id means nothing to them.
        const out_name: ?[]const u8 = if (node.output < graph.values.items.len) graph.values.items[node.output].name else null;
        if (out_name) |n| {
            self.appendDetail("{s}: {s} producing \"{s}\": {s}", .{ @tagName(phase), self.operation, n, self.code });
        } else {
            self.appendDetail("{s}: {s} producing value {d}: {s}", .{ @tagName(phase), self.operation, node.output, self.code });
        }

        for (node.inputs, 0..) |id, i| {
            if (id >= graph.values.items.len) continue;
            const value = graph.values.items[id];
            const dt: []const u8 = if (value.dtype) |d| @tagName(d) else "unknown";
            if (value.name) |n| {
                self.appendDetail("; input[{d}] \"{s}\" dtype={s} shape={any}", .{ i, n, dt, value.shape });
            } else {
                self.appendDetail("; input[{d}] value {d} dtype={s} shape={any}", .{ i, id, dt, value.shape });
            }
        }
        self.appendOpAttrs(node.op);
    }

    /// Every op's own attributes, without this module knowing a single op.
    /// `inline else` hands over the active union payload and `@typeInfo` walks
    /// its fields, so an op added tomorrow reports its configuration the day it
    /// lands instead of the day someone remembers to add a case here.
    fn appendOpAttrs(self: *Diagnostic, op: anytype) void {
        switch (op) {
            inline else => |payload| {
                const Payload = @TypeOf(payload);
                if (@typeInfo(Payload) != .@"struct") return;
                inline for (@typeInfo(Payload).@"struct".field_names) |field_name| {
                    self.appendDetail("; {s}={any}", .{ field_name, @field(payload, field_name) });
                }
            },
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
