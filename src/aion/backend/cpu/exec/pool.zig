// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const ts = @import("../../../runtime/tensor_store.zig");
const exe = @import("../../../runtime/executable.zig");
const Error = @import("../../backend.zig").ExecuteProgramError;

pub fn exec(allocator: std.mem.Allocator, s: exe.StepMaxPool2D, store: ts.TensorStore) Error!void {
    const meta = try store.meta(s.x);
    const out = try store.meta(s.out);
    const shape = s.opts.output(meta.shape) catch return error.InvalidArgument;
    if (!std.mem.eql(usize, &shape, out.shape) or out.dtype != meta.dtype) return error.InvalidArgument;
    switch (meta.dtype) {
        .f32 => try run(f32, allocator, s, store, meta, out),
        .f16 => try run(f16, allocator, s, store, meta, out),
        else => return error.Unsupported,
    }
}

fn count(meta: ts.TensorMeta) usize {
    var n: usize = 1;
    for (meta.tile_counts) |d| n *= d;
    return n;
}

fn run(comptime T: type, allocator: std.mem.Allocator, s: exe.StepMaxPool2D, store: ts.TensorStore, meta: ts.TensorMeta, out: ts.TensorMeta) Error!void {
    if (count(meta) == 1 and count(out) == 1) {
        const src = try store.acquireTileConstLinear(s.x, 0);
        defer store.releaseConst(src.token);
        const dst = try store.acquireTileMutLinear(s.out, 0);
        defer store.releaseMut(dst.token);
        runSingle(T, s, meta, out, src, dst);
        return;
    }
    const tiles = try allocator.alloc(ts.TileRefConst, count(meta));
    defer allocator.free(tiles);
    var acquired: usize = 0;
    defer for (tiles[0..acquired]) |t| store.releaseConst(t.token);
    for (tiles, 0..) |*t, i| {
        t.* = try store.acquireTileConstLinear(s.x, i);
        acquired += 1;
    }
    const p = s.opts;
    for (0..count(out)) |ti| {
        const dst = try store.acquireTileMutLinear(s.out, ti);
        defer store.releaseMut(dst.token);
        var tc: [4]usize = undefined;
        try ts.decodeTileCoords(out, ti, &tc);
        var dims: [4]usize = undefined;
        @memcpy(&dims, dst.shape_mem[0..4]);
        var elements: usize = 1;
        for (dims) |d| elements *= d;
        for (0..elements) |linear| {
            var coord: [4]usize = undefined;
            var rem = linear;
            var d: usize = 4;
            while (d > 0) {
                d -= 1;
                coord[d] = rem % dims[d] + tc[d] * out.tile_shape[d];
                rem /= dims[d];
            }
            var best: T = -std.math.inf(T);
            for (0..p.kernel_h) |kh| {
                const ih = @as(i128, @intCast(coord[1])) * p.stride_h + kh * p.dilation_h - p.pad_top;
                if (ih < 0 or ih >= meta.shape[1]) continue;
                for (0..p.kernel_w) |kw| {
                    const iw = @as(i128, @intCast(coord[2])) * p.stride_w + kw * p.dilation_w - p.pad_left;
                    if (iw < 0 or iw >= meta.shape[2]) continue;
                    const src_coord = [4]usize{ coord[0], @intCast(ih), @intCast(iw), coord[3] };
                    var index: usize = 0;
                    var local: [4]usize = undefined;
                    for (0..4) |axis| {
                        index = index * meta.tile_counts[axis] + src_coord[axis] / meta.tile_shape[axis];
                        local[axis] = src_coord[axis] % meta.tile_shape[axis];
                    }
                    const t = tiles[index];
                    var offset: usize = 0;
                    for (0..4) |axis| offset += local[axis] * @as(usize, @intCast(t.strides_mem[axis]));
                    const v = @as(*align(1) const T, @ptrCast(t.bytes.ptr + offset)).*;
                    // Do not allow a later finite sample to overwrite a NaN.
                    if (std.math.isNan(v) or v > best) best = v;
                }
            }
            var offset: usize = 0;
            for (0..4) |axis| offset += (coord[axis] - tc[axis] * out.tile_shape[axis]) * @as(usize, @intCast(dst.strides_mem[axis]));
            @as(*align(1) T, @ptrCast(dst.bytes.ptr + offset)).* = best;
        }
    }
}

// A single tile needs no coordinate division or tile lookup in the inner loop.
// Keep channel traversal contiguous while respecting the store's byte strides.
fn runSingle(comptime T: type, s: exe.StepMaxPool2D, meta: ts.TensorMeta, out: ts.TensorMeta, src: ts.TileRefConst, dst: ts.TileRefMut) void {
    const p = s.opts;
    for (0..out.shape[0]) |b| for (0..out.shape[1]) |h| for (0..out.shape[2]) |w| {
        const dst_base = b * @as(usize, @intCast(dst.strides_mem[0])) + h * @as(usize, @intCast(dst.strides_mem[1])) + w * @as(usize, @intCast(dst.strides_mem[2]));
        for (0..out.shape[3]) |c| {
            @as(*align(1) T, @ptrCast(dst.bytes.ptr + dst_base + c * @as(usize, @intCast(dst.strides_mem[3])))).* = -std.math.inf(T);
        }
        for (0..p.kernel_h) |kh| {
            const ih = @as(i128, @intCast(h)) * p.stride_h + kh * p.dilation_h - p.pad_top;
            if (ih < 0 or ih >= meta.shape[1]) continue;
            for (0..p.kernel_w) |kw| {
                const iw = @as(i128, @intCast(w)) * p.stride_w + kw * p.dilation_w - p.pad_left;
                if (iw < 0 or iw >= meta.shape[2]) continue;
                const src_base = b * @as(usize, @intCast(src.strides_mem[0])) + @as(usize, @intCast(ih)) * @as(usize, @intCast(src.strides_mem[1])) + @as(usize, @intCast(iw)) * @as(usize, @intCast(src.strides_mem[2]));
                for (0..out.shape[3]) |c| {
                    const v = @as(*align(1) const T, @ptrCast(src.bytes.ptr + src_base + c * @as(usize, @intCast(src.strides_mem[3])))).*;
                    const best = @as(*align(1) T, @ptrCast(dst.bytes.ptr + dst_base + c * @as(usize, @intCast(dst.strides_mem[3]))));
                    if (std.math.isNan(v) or v > best.*) best.* = v;
                }
            }
        }
    };
}
