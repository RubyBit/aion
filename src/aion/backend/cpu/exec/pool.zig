// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const ts = @import("../../../runtime/tensor_store.zig");
const exe = @import("../../../runtime/executable.zig");
const thread_pool = @import("../../../runtime/thread_pool.zig");
const Error = @import("../../backend.zig").ExecuteProgramError;

pub fn exec(
    allocator: std.mem.Allocator,
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: exe.StepMaxPool2D,
    store: ts.TensorStore,
) Error!void {
    _ = allocator;
    const meta = try store.meta(s.x);
    const out = try store.meta(s.out);
    const shape = s.opts.output(meta.shape) catch return error.InvalidArgument;
    if (!std.mem.eql(usize, &shape, out.shape) or out.dtype != meta.dtype) return error.InvalidArgument;
    switch (meta.dtype) {
        .f32 => try run(f32, pool, thread_count, s, store, meta, out),
        .f16 => try run(f16, pool, thread_count, s, store, meta, out),
        else => return error.Unsupported,
    }
}

/// `dst = max(dst, src)` over `n` channels. A NaN sample must survive, and
/// `src > dst` is false for NaN, so it is tested for on its own.
inline fn maxInto(comptime T: type, dst: []align(1) T, src: []align(1) const T) void {
    const lanes = std.simd.suggestVectorLength(T) orelse 1;
    const V = @Vector(lanes, T);
    var i: usize = 0;
    while (i + lanes <= dst.len) : (i += lanes) {
        const d: *align(1) V = @ptrCast(dst.ptr + i);
        const b: V = @as(*align(1) const V, @ptrCast(src.ptr + i)).*;
        const a: V = d.*;
        d.* = @select(T, (b != b) | (b > a), b, a);
    }
    while (i < dst.len) : (i += 1) {
        const b = src[i];
        if (b != b or b > dst[i]) dst[i] = b;
    }
}

/// NHWC max pooling: an output row `(b, h)` is independent of every other, and each
/// pixel folds its in-bounds taps as contiguous runs of channels.
fn Rows(comptime T: type) type {
    return struct {
        s: exe.StepMaxPool2D,
        in_shape: [4]usize,
        out_shape: [4]usize,
        x: []align(1) const T,
        y: []align(1) T,

        const Self = @This();

        fn call(ctx: *anyopaque, start: usize, end: usize, _: usize) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            for (start..end) |row| self.runRow(row / self.out_shape[1], row % self.out_shape[1]);
        }

        fn runRow(self: *Self, b: usize, h: usize) void {
            const p = self.s.opts;
            const c = self.out_shape[3];
            const h_in = self.in_shape[1];
            const w_in = self.in_shape[2];
            for (0..self.out_shape[2]) |w| {
                const dst = self.y[((b * self.out_shape[1] + h) * self.out_shape[2] + w) * c ..][0..c];
                @memset(dst, -std.math.inf(T));
                for (0..p.kernel_h) |kh| {
                    const ih = @as(isize, @intCast(h * p.stride_h + kh * p.dilation_h)) - @as(isize, @intCast(p.pad_top));
                    if (ih < 0 or ih >= h_in) continue;
                    for (0..p.kernel_w) |kw| {
                        const iw = @as(isize, @intCast(w * p.stride_w + kw * p.dilation_w)) - @as(isize, @intCast(p.pad_left));
                        if (iw < 0 or iw >= w_in) continue;
                        const at = ((b * h_in + @as(usize, @intCast(ih))) * w_in + @as(usize, @intCast(iw))) * c;
                        maxInto(T, dst, self.x[at..][0..c]);
                    }
                }
            }
        }
    };
}

fn run(
    comptime T: type,
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: exe.StepMaxPool2D,
    store: ts.TensorStore,
    meta: ts.TensorMeta,
    out: ts.TensorMeta,
) Error!void {
    const x = try store.acquireConst(s.x);
    defer store.releaseConst(x.token);
    const y = try store.acquireMut(s.out);
    defer store.releaseMut(y.token);

    var rows: Rows(T) = .{
        .s = s,
        .in_shape = meta.shape[0..4].*,
        .out_shape = out.shape[0..4].*,
        .x = std.mem.bytesAsSlice(T, x.bytes),
        .y = std.mem.bytesAsSlice(T, y.bytes),
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
