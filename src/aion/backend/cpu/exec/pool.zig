// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const ts = @import("../../../runtime/tensor_store.zig");
const exe = @import("../../../runtime/executable.zig");
const thread_pool = @import("../../../runtime/thread_pool.zig");
const Error = @import("../../backend.zig").ExecuteProgramError;

const RANK: usize = 4;

pub fn exec(
    allocator: std.mem.Allocator,
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: exe.StepMaxPool2D,
    store: ts.TensorStore,
) Error!void {
    const meta = try store.meta(s.x);
    const out = try store.meta(s.out);
    const shape = s.opts.output(meta.shape) catch return error.InvalidArgument;
    if (!std.mem.eql(usize, &shape, out.shape) or out.dtype != meta.dtype) return error.InvalidArgument;
    switch (meta.dtype) {
        .f32 => try run(f32, allocator, pool, thread_count, s, store, meta, out),
        .f16 => try run(f16, allocator, pool, thread_count, s, store, meta, out),
        else => return error.Unsupported,
    }
}

fn count(meta: ts.TensorMeta) usize {
    var n: usize = 1;
    for (meta.tile_counts) |d| n *= d;
    return n;
}

/// Which tile holds `coord`, and where inside it. Pooling never mixes channels,
/// so this is called once per contiguous channel run rather than per element.
inline fn locate(meta: ts.TensorMeta, coord: [RANK]usize, local: *[RANK]usize) usize {
    var index: usize = 0;
    for (0..RANK) |axis| {
        index = index * meta.tile_counts[axis] + coord[axis] / meta.tile_shape[axis];
        local[axis] = coord[axis] % meta.tile_shape[axis];
    }
    return index;
}

inline fn byteOffset(strides: [ts.INLINE_RANK]isize, local: [RANK]usize) usize {
    var off: usize = 0;
    for (0..RANK) |axis| off += local[axis] * @as(usize, @intCast(strides[axis]));
    return off;
}

/// `dst = max(dst, src)` over `n` channels. A NaN sample must survive, and
/// `src > dst` is false for NaN, so it is tested for on its own.
inline fn maxInto(comptime T: type, dst: [*]u8, src: [*]const u8, n: usize) void {
    const lanes = std.simd.suggestVectorLength(T) orelse 1;
    const V = @Vector(lanes, T);
    var i: usize = 0;
    while (i + lanes <= n) : (i += lanes) {
        const d: *align(1) V = @ptrCast(dst + i * @sizeOf(T));
        const b: V = @as(*align(1) const V, @ptrCast(src + i * @sizeOf(T))).*;
        const a: V = d.*;
        d.* = @select(T, (b != b) | (b > a), b, a);
    }
    while (i < n) : (i += 1) {
        const d: *align(1) T = @ptrCast(dst + i * @sizeOf(T));
        const b: T = @as(*align(1) const T, @ptrCast(src + i * @sizeOf(T))).*;
        if (b != b or b > d.*) d.* = b;
    }
}

inline fn fillNegInf(comptime T: type, dst: [*]u8, n: usize) void {
    var i: usize = 0;
    while (i < n) : (i += 1) {
        @as(*align(1) T, @ptrCast(dst + i * @sizeOf(T))).* = -std.math.inf(T);
    }
}

/// Rows are independent and every tile is leased before the parallel region, so
/// a worker only does pointer arithmetic.
fn Rows(comptime T: type) type {
    return struct {
        s: exe.StepMaxPool2D,
        meta: ts.TensorMeta,
        out: ts.TensorMeta,
        src_tiles: []const ts.TileRefConst,
        dst_tiles: []const ts.TileRefMut,
        contiguous: bool,

        const Self = @This();

        fn call(ctx: *anyopaque, start: usize, end: usize, _: usize) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            for (start..end) |row| self.runRow(row / self.out.shape[1], row % self.out.shape[1]);
        }

        fn runRow(self: *Self, b: usize, h: usize) void {
            const channels = self.out.shape[3];
            for (0..self.out.shape[2]) |w| {
                var c0: usize = 0;
                while (c0 < channels) {
                    var dl: [RANK]usize = undefined;
                    const dst = self.dst_tiles[locate(self.out, .{ b, h, w, c0 }, &dl)];
                    const span = @min(channels - c0, dst.shape_mem[3] - dl[3]);
                    const dst_ptr = dst.bytes.ptr + byteOffset(dst.strides_mem, dl);
                    if (self.contiguous) fillNegInf(T, dst_ptr, span) else self.fillStrided(dst, dst_ptr, span);
                    self.accumulate(b, h, w, c0, span, dst, dst_ptr);
                    c0 += span;
                }
            }
        }

        /// Fold every in-bounds window tap into one output channel run.
        fn accumulate(self: *Self, b: usize, h: usize, w: usize, c0: usize, span: usize, dst: ts.TileRefMut, dst_ptr: [*]u8) void {
            const p = self.s.opts;
            for (0..p.kernel_h) |kh| {
                const ih = @as(isize, @intCast(h)) * @as(isize, @intCast(p.stride_h)) +
                    @as(isize, @intCast(kh * p.dilation_h)) - @as(isize, @intCast(p.pad_top));
                if (ih < 0 or ih >= self.meta.shape[1]) continue;
                for (0..p.kernel_w) |kw| {
                    const iw = @as(isize, @intCast(w)) * @as(isize, @intCast(p.stride_w)) +
                        @as(isize, @intCast(kw * p.dilation_w)) - @as(isize, @intCast(p.pad_left));
                    if (iw < 0 or iw >= self.meta.shape[2]) continue;

                    // The source may tile channels differently, so walk it in
                    // whatever contiguous pieces it offers.
                    var done: usize = 0;
                    while (done < span) {
                        var sl: [RANK]usize = undefined;
                        const coord: [RANK]usize = .{ b, @intCast(ih), @intCast(iw), c0 + done };
                        const src = self.src_tiles[locate(self.meta, coord, &sl)];
                        const part = @min(span - done, src.shape_mem[3] - sl[3]);
                        const src_ptr = src.bytes.ptr + byteOffset(src.strides_mem, sl);
                        const at = dst_ptr + done * @as(usize, @intCast(dst.strides_mem[3]));
                        if (self.contiguous) {
                            maxInto(T, at, src_ptr, part);
                        } else {
                            self.maxIntoStrided(dst, at, src, src_ptr, part);
                        }
                        done += part;
                    }
                }
            }
        }

        fn fillStrided(self: *Self, dst: ts.TileRefMut, at: [*]u8, n: usize) void {
            _ = self;
            const step: usize = @intCast(dst.strides_mem[3]);
            for (0..n) |i| @as(*align(1) T, @ptrCast(at + i * step)).* = -std.math.inf(T);
        }

        fn maxIntoStrided(self: *Self, dst: ts.TileRefMut, at: [*]u8, src: ts.TileRefConst, src_ptr: [*]const u8, n: usize) void {
            _ = self;
            const ds: usize = @intCast(dst.strides_mem[3]);
            const ss: usize = @intCast(src.strides_mem[3]);
            for (0..n) |i| {
                const d: *align(1) T = @ptrCast(at + i * ds);
                const v: T = @as(*align(1) const T, @ptrCast(src_ptr + i * ss)).*;
                if (v != v or v > d.*) d.* = v;
            }
        }
    };
}

fn run(
    comptime T: type,
    allocator: std.mem.Allocator,
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: exe.StepMaxPool2D,
    store: ts.TensorStore,
    meta: ts.TensorMeta,
    out: ts.TensorMeta,
) Error!void {
    const src_tiles = try allocator.alloc(ts.TileRefConst, count(meta));
    defer allocator.free(src_tiles);
    var src_held: usize = 0;
    defer for (src_tiles[0..src_held]) |t| store.releaseConst(t.token);
    for (src_tiles, 0..) |*t, i| {
        t.* = try store.acquireTileConstLinear(s.x, i);
        src_held += 1;
    }

    const dst_tiles = try allocator.alloc(ts.TileRefMut, count(out));
    defer allocator.free(dst_tiles);
    var dst_held: usize = 0;
    defer for (dst_tiles[0..dst_held]) |t| store.releaseMut(t.token);
    for (dst_tiles, 0..) |*t, i| {
        t.* = try store.acquireTileMutLinear(s.out, i);
        dst_held += 1;
    }

    // Channels are the innermost axis, so a dense tile makes each run a plain
    // vector op; anything else falls back to the strided walk.
    const elem: isize = @sizeOf(T);
    const contiguous = (count(meta) == 0 or src_tiles[0].strides_mem[3] == elem) and
        (count(out) == 0 or dst_tiles[0].strides_mem[3] == elem);

    var rows: Rows(T) = .{
        .s = s,
        .meta = meta,
        .out = out,
        .src_tiles = src_tiles,
        .dst_tiles = dst_tiles,
        .contiguous = contiguous,
    };
    const total: usize = out.shape[0] * out.shape[1];
    if (pool) |p| {
        if (thread_count > 1 and total > 1) {
            p.parallelForAny(@ptrCast(&rows), total, 1, Rows(T).call);
            return;
        }
    }
    Rows(T).call(@ptrCast(&rows), 0, total, 0);
}
