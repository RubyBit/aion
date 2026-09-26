// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! A borrowed, strided view of host memory: the one form in which outside data
//! enters or leaves the core. The C ABI builds one from a DLPack `DLTensor`.
//!
//! `elem` is what the bytes ARE, independent of the dtype a tensor stores, so a
//! bf16 checkpoint reads straight into an f32 or f16 tensor, or into a quantizer,
//! and a float64 or int64 array (numpy's defaults) straight into an f32 or i32
//! one, with no such dtypes in the core. Floats convert among themselves (rounding
//! to the nearest even when narrowing); integers convert among themselves, and a
//! value that does not fit the narrower type is an error, never wrapped. Kinds
//! never mix. Elements are in native byte order, as DLPack defines them.
//!
//! A view owns nothing. `HostView` is read-only; only a `HostViewMut` -- a
//! destination the caller hands over for writing -- can be written through.

const std = @import("std");
const types = @import("../backend/types.zig");

pub const DType = types.DType;
pub const max_rank = @import("../runtime/tensor_store.zig").max_rank;

pub const Error = error{InvalidArgument};

/// What a view's bytes are, element by element.
pub const Elem = enum {
    f64,
    f32,
    f16,
    bf16,
    i64,
    i32,
    i8,

    pub fn bytes(self: Elem) usize {
        return switch (self) {
            .f64, .i64 => 8,
            .f32, .i32 => 4,
            .f16, .bf16 => 2,
            .i8 => 1,
        };
    }

    fn isFloat(self: Elem) bool {
        return switch (self) {
            .f64, .f32, .f16, .bf16 => true,
            .i64, .i32, .i8 => false,
        };
    }

    /// The element a scalar tensor dtype stores; null for a quantized one.
    pub fn of(dtype: DType) ?Elem {
        return switch (dtype) {
            .f32 => .f32,
            .f16 => .f16,
            .i32 => .i32,
            .i8 => .i8,
            .q4_0, .q8_0 => null,
        };
    }

    /// Whether values convert between `self` and `other`: the same kind, float or
    /// integer (an integer narrowing can still fail on a value that does not fit).
    pub fn convertsTo(self: Elem, other: Elem) bool {
        return self.isFloat() == other.isFloat();
    }
};

pub const HostView = View(false);
pub const HostViewMut = View(true);

/// A view of `shape` elements of `elem` at `data`; writable only when `mutable`.
pub fn View(comptime mutable: bool) type {
    return struct {
        const Self = @This();
        pub const Bytes = if (mutable) [*]u8 else [*]const u8;

        /// First element.
        data: Bytes,
        elem: Elem,
        shape: []const usize,
        /// Per-axis steps in elements, possibly negative; null is row-major contiguous.
        strides: ?[]const isize = null,

        /// A row-major view of `bytes`, which holds `shape`'s elements of `elem`.
        pub fn contiguous(elem: Elem, shape: []const usize, bytes: if (mutable) []u8 else []const u8) Self {
            return .{ .data = bytes.ptr, .elem = elem, .shape = shape };
        }

        pub fn count(self: Self) usize {
            var n: usize = 1;
            for (self.shape) |d| n *= d;
            return n;
        }

        /// Row-major with no gaps: whole runs copy as one block.
        pub fn isContiguous(self: Self) bool {
            const strides = self.strides orelse return true;
            var want: isize = 1;
            var axis = self.shape.len;
            while (axis > 0) {
                axis -= 1;
                if (self.shape[axis] != 1 and strides[axis] != want) return false;
                want *= @intCast(self.shape[axis]);
            }
            return true;
        }

        /// Validate a view whose shape and strides came from outside the core: rank
        /// in range and every element addressable without overflowing.
        pub fn validate(self: Self) Error!void {
            if (self.shape.len > max_rank) return error.InvalidArgument;
            if (self.strides) |s| if (s.len != self.shape.len) return error.InvalidArgument;
            var n: usize = 1;
            for (self.shape) |d| n = std.math.mul(usize, n, d) catch return error.InvalidArgument;
            _ = std.math.mul(usize, n, self.elem.bytes()) catch return error.InvalidArgument;
        }

        /// Read row-major elements `[first, first + n)` into `out`, packed as `dtype`
        /// stores them (`n = out.len / dtype bytes`).
        pub fn read(self: Self, dtype: DType, first: usize, out: []u8) Error!void {
            const to = Elem.of(dtype) orelse return error.InvalidArgument;
            if (!self.elem.convertsTo(to)) return error.InvalidArgument;
            const n = out.len / to.bytes();
            if (self.elem == to and self.isContiguous()) {
                @memcpy(out[0 .. n * to.bytes()], self.data[first * to.bytes() ..][0 .. n * to.bytes()]);
                return;
            }
            if (self.transposedRows(first, n)) |rows| {
                return readTransposed(self.elem, to, self.data, self.shape[1], self.strides.?[1], rows.first, rows.count, out.ptr);
            }
            var cur = Cursor.init(self.shape, self.strides, first);
            var copied: usize = 0;
            while (copied < n) {
                const run = cur.run(n - copied);
                try convertRun(self.elem, self.data, cur.offset, cur.step(), to, out.ptr + copied * to.bytes(), 1, run);
                copied += run;
                cur.advance(run);
            }
        }

        /// Read row-major elements `[first, first + out.len)` as f32.
        pub fn readF32(self: Self, first: usize, out: []f32) Error!void {
            return self.read(.f32, first, std.mem.sliceAsBytes(out));
        }

        /// Write row-major elements `[first, first + n)` from `src`, packed as `dtype`
        /// stores them, into the view.
        pub fn write(self: Self, dtype: DType, first: usize, src: []const u8) Error!void {
            if (comptime !mutable) @compileError("a read-only view cannot be written; use HostViewMut");
            const from = Elem.of(dtype) orelse return error.InvalidArgument;
            if (!from.convertsTo(self.elem)) return error.InvalidArgument;
            const n = src.len / from.bytes();
            if (self.elem == from and self.isContiguous()) {
                @memcpy(self.data[first * from.bytes() ..][0 .. n * from.bytes()], src[0 .. n * from.bytes()]);
                return;
            }
            var cur = Cursor.init(self.shape, self.strides, first);
            var copied: usize = 0;
            while (copied < n) {
                const run = cur.run(n - copied);
                try convertRun(from, src.ptr + copied * from.bytes(), 0, 1, self.elem, at(self.data, cur.offset, self.elem), cur.step(), run);
                copied += run;
                cur.advance(run);
            }
        }

        /// When the view is a rank-2 transpose (unit stride down a column, not along
        /// a row) and `[first, first + n)` is whole rows, those rows; a transposed
        /// read then goes a tile at a time instead of one column-strided element at
        /// a time.
        fn transposedRows(self: Self, first: usize, n: usize) ?struct { first: usize, count: usize } {
            const strides = self.strides orelse return null;
            if (self.shape.len != 2 or strides[0] != 1 or strides[1] == 1) return null;
            const cols = self.shape[1];
            if (cols == 0 or first % cols != 0 or n % cols != 0) return null;
            return .{ .first = first / cols, .count = n / cols };
        }
    };
}

/// Walks row-major positions of a view in runs along its last axis.
const Cursor = struct {
    shape: [max_rank]usize = undefined,
    strides: [max_rank]isize = undefined,
    index: [max_rank]usize = @splat(0),
    rank: usize,
    /// Element offset of the current position from the view's first element.
    offset: isize = 0,

    fn init(shape: []const usize, strides: ?[]const isize, first: usize) Cursor {
        var c: Cursor = .{ .rank = @max(shape.len, 1) };
        if (shape.len == 0) {
            c.shape[0] = 1;
            c.strides[0] = 1;
        } else {
            @memcpy(c.shape[0..c.rank], shape);
            if (strides) |s| {
                @memcpy(c.strides[0..c.rank], s);
            } else {
                var run_step: isize = 1;
                var axis = c.rank;
                while (axis > 0) {
                    axis -= 1;
                    c.strides[axis] = run_step;
                    run_step *= @intCast(c.shape[axis]);
                }
            }
        }
        var rest = first;
        var axis = c.rank;
        while (axis > 0) {
            axis -= 1;
            if (c.shape[axis] == 0) return c;
            c.index[axis] = rest % c.shape[axis];
            rest /= c.shape[axis];
            c.offset += @as(isize, @intCast(c.index[axis])) * c.strides[axis];
        }
        return c;
    }

    fn step(self: *const Cursor) isize {
        return self.strides[self.rank - 1];
    }

    /// Elements left in the current last-axis run, capped at `limit`.
    fn run(self: *const Cursor, limit: usize) usize {
        const last = self.rank - 1;
        return @min(self.shape[last] - self.index[last], limit);
    }

    fn advance(self: *Cursor, n: usize) void {
        var axis = self.rank - 1;
        self.index[axis] += n;
        self.offset += @as(isize, @intCast(n)) * self.strides[axis];
        while (axis > 0 and self.index[axis] == self.shape[axis]) {
            self.offset -= @as(isize, @intCast(self.shape[axis])) * self.strides[axis];
            self.index[axis] = 0;
            axis -= 1;
            self.index[axis] += 1;
            self.offset += self.strides[axis];
        }
    }
};

/// `base` moved by `elems` elements of `elem` (possibly backwards); keeps constness.
fn at(base: anytype, elems: isize, elem: Elem) @TypeOf(base) {
    const off = elems * @as(isize, @intCast(elem.bytes()));
    return if (off >= 0) base + @as(usize, @intCast(off)) else base - @as(usize, @intCast(-off));
}

/// Convert `n` elements from `src` (element offset `src_off`, stepping `src_step`)
/// to `dst` (stepping `dst_step`). Kinds were checked by the caller.
fn convertRun(from: Elem, src: [*]const u8, src_off: isize, src_step: isize, to: Elem, dst: [*]u8, dst_step: isize, n: usize) Error!void {
    switch (from) {
        inline else => |f| switch (to) {
            inline else => |t| if (comptime f.convertsTo(t)) {
                var s: isize = src_off;
                var d: isize = 0;
                for (0..n) |_| {
                    try convertOne(f, t, at(src, s, f), at(dst, d, t));
                    s += src_step;
                    d += dst_step;
                }
            } else unreachable, // kinds never mix (`convertsTo`)
        },
    }
}

/// One element: floats through f32 (f64 when either side is f64), integers
/// through i64, failing when the value does not fit `to`.
inline fn convertOne(comptime from: Elem, comptime to: Elem, sp: [*]const u8, dp: [*]u8) Error!void {
    if (comptime from == to) {
        @memcpy(dp[0..from.bytes()], sp[0..from.bytes()]);
    } else if (comptime from.isFloat()) {
        const Wide = if (from == .f64 or to == .f64) f64 else f32;
        storeFloat(to, dp, loadFloat(from, Wide, sp));
    } else {
        const v: i64 = switch (from) {
            .i64 => @as(*align(1) const i64, @ptrCast(sp)).*,
            .i32 => @as(*align(1) const i32, @ptrCast(sp)).*,
            .i8 => @as(*align(1) const i8, @ptrCast(sp)).*,
            else => comptime unreachable,
        };
        switch (to) {
            .i64 => @as(*align(1) i64, @ptrCast(dp)).* = v,
            .i32 => @as(*align(1) i32, @ptrCast(dp)).* = std.math.cast(i32, v) orelse return error.InvalidArgument,
            .i8 => @as(*align(1) i8, @ptrCast(dp)).* = std.math.cast(i8, v) orelse return error.InvalidArgument,
            else => comptime unreachable,
        }
    }
}

/// Rows `[row0, row0 + rows)` of a rank-2 transposed view (`cols` wide, `col_step`
/// elements between columns) into row-major `out`, in square tiles: each tile reads
/// the source down its unit-stride axis and stays in cache while its transpose is
/// written.
fn readTransposed(from: Elem, to: Elem, data: [*]const u8, cols: usize, col_step: isize, row0: usize, rows: usize, out: [*]u8) Error!void {
    switch (from) {
        inline else => |f| switch (to) {
            inline else => |t| if (comptime f.convertsTo(t)) {
                const tile = 64;
                var c0: usize = 0;
                while (c0 < cols) : (c0 += tile) {
                    const c1 = @min(c0 + tile, cols);
                    var r0: usize = 0;
                    while (r0 < rows) : (r0 += tile) {
                        const r1 = @min(r0 + tile, rows);
                        for (c0..c1) |c| {
                            // Column `c` of the logical view is contiguous in the source.
                            const base: isize = @as(isize, @intCast(row0)) + @as(isize, @intCast(c)) * col_step;
                            for (r0..r1) |r| {
                                try convertOne(f, t, at(data, base + @as(isize, @intCast(r)), f), out + (r * cols + c) * t.bytes());
                            }
                        }
                    }
                }
            } else unreachable,
        },
    }
}

fn loadFloat(comptime elem: Elem, comptime T: type, p: [*]const u8) T {
    return switch (elem) {
        .f64 => @floatCast(@as(*align(1) const f64, @ptrCast(p)).*),
        .f32 => @as(*align(1) const f32, @ptrCast(p)).*,
        .f16 => @as(*align(1) const f16, @ptrCast(p)).*,
        .bf16 => bf16ToF32(@as(*align(1) const u16, @ptrCast(p)).*),
        .i64, .i32, .i8 => comptime unreachable,
    };
}

fn storeFloat(comptime elem: Elem, p: [*]u8, v: anytype) void {
    switch (elem) {
        .f64 => @as(*align(1) f64, @ptrCast(p)).* = v,
        .f32 => @as(*align(1) f32, @ptrCast(p)).* = @floatCast(v),
        .f16 => @as(*align(1) f16, @ptrCast(p)).* = @floatCast(v),
        // Through f32 first: bf16 keeps f32's exponent, so only its mantissa rounds.
        .bf16 => @as(*align(1) u16, @ptrCast(p)).* = bf16FromF32(@floatCast(v)),
        .i64, .i32, .i8 => comptime unreachable,
    }
}

/// Round to the nearest bf16, ties to even; NaN stays a (quiet) NaN.
fn bf16FromF32(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @intCast((bits >> 16) | 0x40);
    const rounded = bits + 0x7FFF + ((bits >> 16) & 1);
    return @intCast(rounded >> 16);
}

fn bf16ToF32(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

test "host view: a transposed bf16 view reads as the f32 transpose" {
    // [2, 3] row-major bf16 source, viewed as its [3, 2] transpose.
    const vals = [_]f32{ 1, 2, 3, -4, 0.5, 6 };
    var src: [6]u16 = undefined;
    for (&src, vals) |*s, v| s.* = bf16FromF32(v);
    const view: HostView = .{ .data = @ptrCast(&src), .elem = .bf16, .shape = &.{ 3, 2 }, .strides = &.{ 1, 3 } };
    var out: [6]f32 = undefined;
    try view.readF32(0, &out);
    try std.testing.expectEqualSlices(f32, &.{ 1, -4, 2, 0.5, 3, 6 }, &out);
    // A range starting mid-row picks up where the cursor lands.
    var tail: [3]f32 = undefined;
    try view.readF32(3, &tail);
    try std.testing.expectEqualSlices(f32, &.{ 0.5, 3, 6 }, &tail);
}

test "host view: a tiled transposed read matches the element-wise walk" {
    // 70 x 130 (not multiples of the tile), bf16 stored as the [130, 70] transpose.
    const rows = 70;
    const cols = 130;
    var src: [rows * cols]u16 = undefined;
    for (&src, 0..) |*h, i| h.* = bf16FromF32(@as(f32, @floatFromInt(i % 251)) - 125);
    const view: HostView = .{ .data = @ptrCast(&src), .elem = .bf16, .shape = &.{ rows, cols }, .strides = &.{ 1, rows } };
    var tiled: [rows * cols]f32 = undefined;
    try view.readF32(0, &tiled);
    for (0..rows) |r| for (0..cols) |c| {
        try std.testing.expectEqual(bf16ToF32(src[c * rows + r]), tiled[r * cols + c]);
    };
    // A range of whole rows in the middle, and one that is not whole rows (the walk).
    var mid: [3 * cols]f32 = undefined;
    try view.readF32(5 * cols, &mid);
    try std.testing.expectEqualSlices(f32, tiled[5 * cols ..][0 .. 3 * cols], &mid);
    var ragged: [cols + 7]f32 = undefined;
    try view.readF32(2 * cols + 3, &ragged);
    try std.testing.expectEqualSlices(f32, tiled[2 * cols + 3 ..][0 .. cols + 7], &ragged);
}

test "host view: negative strides, f16 targets, and writes through a mutable view" {
    const vals = [_]f32{ 1, 2, 3, 4 };
    // A reversed f32 view: element i reads vals[3 - i].
    const view: HostView = .{ .data = @ptrCast(&vals[3]), .elem = .f32, .shape = &.{4}, .strides = &.{-1} };
    var halves: [4]f16 = undefined;
    try view.read(.f16, 0, std.mem.sliceAsBytes(&halves));
    try std.testing.expectEqualSlices(f16, &.{ 4, 3, 2, 1 }, &halves);

    var dst: [4]u16 = @splat(0);
    const out_view: HostViewMut = .{ .data = @ptrCast(&dst), .elem = .bf16, .shape = &.{ 2, 2 }, .strides = &.{ 1, 2 } };
    try out_view.write(.f32, 0, std.mem.sliceAsBytes(&vals));
    // Written column-major: dst holds 1, 3, 2, 4.
    for (dst, [_]f32{ 1, 3, 2, 4 }) |d, want| try std.testing.expectEqual(want, bf16ToF32(d));
}

test "host view: integers narrow only when they fit; kinds never mix" {
    var ints = [_]i32{ 7, -8 };
    const view = HostView.contiguous(.i32, &.{2}, std.mem.sliceAsBytes(&ints));
    var out: [2]i32 = undefined;
    try view.read(.i32, 0, std.mem.sliceAsBytes(&out));
    try std.testing.expectEqualSlices(i32, &ints, &out);
    var floats: [2]f32 = undefined;
    try std.testing.expectError(error.InvalidArgument, view.readF32(0, &floats));

    // int64 (numpy's default) narrows to i32 and i8 while every value fits.
    var wide = [_]i64{ 3, -120, 1 << 20 };
    const wide_view = HostView.contiguous(.i64, &.{3}, std.mem.sliceAsBytes(&wide));
    var narrow: [3]i32 = undefined;
    try wide_view.read(.i32, 0, std.mem.sliceAsBytes(&narrow));
    try std.testing.expectEqualSlices(i32, &.{ 3, -120, 1 << 20 }, &narrow);
    var bytes: [2]i8 = undefined;
    try wide_view.read(.i8, 0, std.mem.sliceAsBytes(&bytes));
    try std.testing.expectEqualSlices(i8, &.{ 3, -120 }, &bytes);
    var all_bytes: [3]i8 = undefined;
    try std.testing.expectError(error.InvalidArgument, wide_view.read(.i8, 0, std.mem.sliceAsBytes(&all_bytes)));
    wide[0] = 1 << 40;
    try std.testing.expectError(error.InvalidArgument, wide_view.read(.i32, 0, std.mem.sliceAsBytes(&narrow)));
}

test "host view: float64 rounds once to each narrower float" {
    // 1 + 2^-11 + 2^-40: nearer 1 + 2^-10 than 1 in f16, but an f32 stop on the way
    // would drop the 2^-40 and leave an exact tie that rounds down to 1.
    var src = [_]f64{ 1.0 + 0x1p-11 + 0x1p-40, -2.5 };
    const view = HostView.contiguous(.f64, &.{2}, std.mem.sliceAsBytes(&src));
    var halves: [2]f16 = undefined;
    try view.read(.f16, 0, std.mem.sliceAsBytes(&halves));
    try std.testing.expectEqualSlices(f16, &.{ 1.0 + 0x1p-10, -2.5 }, &halves);
    var singles: [2]f32 = undefined;
    try view.readF32(0, &singles);
    try std.testing.expectEqualSlices(f32, &.{ @floatCast(src[0]), -2.5 }, &singles);
}

test "host view: bf16 rounding is to nearest even" {
    try std.testing.expectEqual(@as(u16, 0x3F80), bf16FromF32(1.0));
    // 1 + 2^-8 sits exactly between two bf16 values: ties go to the even one (1.0).
    try std.testing.expectEqual(@as(u16, 0x3F80), bf16FromF32(1.00390625));
    // Just above the tie rounds up.
    try std.testing.expectEqual(@as(u16, 0x3F81), bf16FromF32(1.0040));
}
