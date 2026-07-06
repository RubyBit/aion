// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! A tiny line-oriented WGSL writer, shared by the GPU compute kernels.
//!
//! Kernel codegen emits WGSL you can *read*: `w.line("acc += a * b;", .{})` writes
//! that line verbatim (with `{d}`/`{s}` holes filled from the config), and
//! `w.open("loop")` / `w.close()` bracket a block and handle indentation, so the
//! generator source reads like the shader it produces — no manual `\n`, no brace
//! bookkeeping, no AST to mentally compile back into WGSL.
//!
//! Usage: `Wgsl.init(arena)`, emit with `lit`/`line`/`open`/`otherwise`/`close`,
//! then `finish()` for the NUL-terminated source. Generation is runtime (arena);
//! kernels are built once per config when the backend first compiles them.

const std = @import("std");

pub const Wgsl = struct {
    a: std.mem.Allocator,
    buf: std.ArrayList(u8) = .empty,
    depth: usize = 0,

    pub fn init(a: std.mem.Allocator) Wgsl {
        return .{ .a = a };
    }

    fn indent(self: *Wgsl) void {
        var i: usize = 0;
        while (i < self.depth) : (i += 1) self.buf.appendSlice(self.a, "  ") catch oom();
    }

    /// A verbatim line (no formatting) — use for static WGSL, incl. lines with braces.
    pub fn lit(self: *Wgsl, s: []const u8) void {
        self.indent();
        self.buf.appendSlice(self.a, s) catch oom();
        self.buf.append(self.a, '\n') catch oom();
    }

    fn appendFmt(self: *Wgsl, comptime f: []const u8, args: anytype) void {
        self.buf.appendSlice(self.a, self.fmt(f, args)) catch oom();
    }

    /// Format a fragment into the arena — for building a sub-expression string to
    /// splice into a `line`/`open` (e.g. a computed `k0` expression).
    pub fn fmt(self: *Wgsl, comptime f: []const u8, args: anytype) []const u8 {
        return std.fmt.allocPrint(self.a, f, args) catch oom();
    }

    /// A formatted line, e.g. `line("let x = {d}u;", .{n})`. No trailing brace.
    pub fn line(self: *Wgsl, comptime f: []const u8, args: anytype) void {
        self.indent();
        self.appendFmt(f, args);
        self.buf.append(self.a, '\n') catch oom();
    }

    /// Open a block: emit `<formatted> {` and indent. Pair with `close`.
    pub fn open(self: *Wgsl, comptime f: []const u8, args: anytype) void {
        self.indent();
        self.appendFmt(f, args);
        self.buf.appendSlice(self.a, " {\n") catch oom();
        self.depth += 1;
    }

    /// Open a bare `{` block (no header) — for scoping locals.
    pub fn openBlock(self: *Wgsl) void {
        self.lit("{");
        self.depth += 1;
    }

    /// `} else {` — close the current block and open an else arm.
    pub fn otherwise(self: *Wgsl) void {
        self.depth -= 1;
        self.lit("} else {");
        self.depth += 1;
    }

    /// Close a block opened by `open`/`openBlock`.
    pub fn close(self: *Wgsl) void {
        self.depth -= 1;
        self.lit("}");
    }

    /// Close a struct/declaration block that needs a trailing `;`.
    pub fn closeSemi(self: *Wgsl) void {
        self.depth -= 1;
        self.lit("};");
    }

    pub fn blank(self: *Wgsl) void {
        self.buf.append(self.a, '\n') catch oom();
    }

    /// The finished, NUL-terminated WGSL source (arena-owned).
    pub fn finish(self: *Wgsl) [:0]const u8 {
        return self.buf.toOwnedSliceSentinel(self.a, 0) catch oom();
    }

    fn oom() noreturn {
        @panic("wgsl: OOM");
    }
};

test "wgsl writer indents nested blocks" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    var w = Wgsl.init(arena_state.allocator());
    w.open("fn k()", .{});
    w.open("loop", .{});
    w.line("acc += av.{s} * b;", .{"x"});
    w.close();
    w.close();
    const src = w.finish();
    try std.testing.expectEqualStrings(
        \\fn k() {
        \\  loop {
        \\    acc += av.x * b;
        \\  }
        \\}
        \\
    , src);
}
