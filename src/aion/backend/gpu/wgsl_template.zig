// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Filling a WGSL kernel's `// #name` lines at compile time.
//!
//! Matrix-tile kernels write every tile operation out: the shader compiler does
//! not unroll loops, and tiles cannot be indexed. The kernel stays a readable
//! .wgsl file with a marker line where such a run goes, and the executor fills it.

const std = @import("std");

/// `tmpl` with every line `// #name` replaced by `lines`, each indented as the
/// marker was. A marker the kernel lacks is a compile error.
pub fn fill(comptime tmpl: []const u8, comptime name: []const u8, comptime lines: []const []const u8) []const u8 {
    const marker = "// #" ++ name ++ "\n";
    var out: []const u8 = "";
    var rest = tmpl;
    var found = false;
    while (std.mem.indexOf(u8, rest, marker)) |at| {
        const line_start = if (std.mem.lastIndexOfScalar(u8, rest[0..at], '\n')) |nl| nl + 1 else 0;
        const indent = rest[line_start..at];
        out = out ++ rest[0..line_start];
        for (lines) |line| out = out ++ indent ++ line ++ "\n";
        rest = rest[at + marker.len ..];
        found = true;
    }
    if (!found) @compileError("kernel has no `// #" ++ name ++ "` line");
    return out ++ rest;
}

/// One line per index `i` below `count`: `fmt` with `i` and `i * step`.
pub fn each(comptime count: usize, comptime step: usize, comptime fmt: []const u8) []const []const u8 {
    var lines: []const []const u8 = &.{};
    for (0..count) |i| lines = lines ++ .{std.fmt.comptimePrint(fmt, .{ i, i * step })};
    return lines;
}
