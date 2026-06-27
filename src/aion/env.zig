// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Cross-platform environment-variable reads.
//
// std's free env functions were removed in this Zig snapshot, and the `Environ`
// "global" block is Windows/WASI-only (POSIX `PosixBlock` has no `.global`), so a
// bare `Environ{ .block = .global }` fails to compile for linux/macos/aarch64.
// These helpers branch by OS so the library builds (and reads env) on every target:
//   * Windows: the global `Environ` block.
//   * POSIX with libc: `getenv`.
//   * POSIX without libc: returns null — env-driven knobs (all optional debug/test
//     flags) simply default to "unset", which is the correct fallback.

const std = @import("std");
const builtin = @import("builtin");

/// Read an env var into an owned, caller-freed buffer; null if unset/unavailable.
pub fn getOwned(allocator: std.mem.Allocator, name: [:0]const u8) ?[]u8 {
    if (builtin.os.tag == .windows) {
        return std.process.Environ.getAlloc(.{ .block = .global }, allocator, name) catch null;
    }
    if (builtin.link_libc) {
        const v = std.c.getenv(name.ptr) orelse return null;
        return allocator.dupe(u8, std.mem.span(v)) catch null;
    }
    return null;
}

/// True when `name` is set to a non-empty, non-"0"/"false" value.
pub fn flagEnabled(name: [:0]const u8) bool {
    const raw = getOwned(std.heap.page_allocator, name) orelse return false;
    defer std.heap.page_allocator.free(raw);
    const t = std.mem.trim(u8, raw, " \t\r\n");
    if (t.len == 0) return false;
    if (std.mem.eql(u8, t, "0")) return false;
    if (std.mem.eql(u8, t, "false")) return false;
    return true;
}
