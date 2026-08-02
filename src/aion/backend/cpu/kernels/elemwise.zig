// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const types = @import("../../types.zig");
const simd = @import("simd.zig");

const BackendError = types.BackendError;
const ElemwiseBinaryOp = types.ElemwiseBinaryOp;

/// Generic broadcast over the last dimension (packed row-major).
///
/// Computes:
///   out[row, col] = op(a[row, col], b[col])
/// where `a` and `out` have shape [row_count, col_count] and `b` has shape [col_count].
///
/// `op` is comptime to fully specialize hot contiguous-suffix paths such as bias-add.
pub fn contiguousSuffixBinaryF32(
    comptime op: ElemwiseBinaryOp,
    out_bytes: []u8,
    a_bytes: []const u8,
    b_bytes: []const u8,
    row_count: usize,
    col_count: usize,
) BackendError!void {
    const out: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, out_bytes);
    const a: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, a_bytes);
    const b: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, b_bytes);

    const elem_count: usize = row_count * col_count;
    if (out.len < elem_count or a.len < elem_count or b.len < col_count) {
        return BackendError.InvalidArgument;
    }

    const lanes: usize = comptime simd.lanesF32();
    const Vec = @Vector(lanes, f32);

    var r: usize = 0;
    while (r < row_count) : (r += 1) {
        const base: usize = r * col_count;
        var c: usize = 0;
        const vec_end: usize = col_count - (col_count % lanes);
        while (c < vec_end) : (c += lanes) {
            const av: Vec = @as(*align(1) const Vec, @ptrCast(a.ptr + base + c)).*;
            const bv: Vec = @as(*align(1) const Vec, @ptrCast(b.ptr + c)).*;
            const rv: Vec = switch (op) {
                .add => av + bv,
                .sub => av - bv,
                .mul => av * bv,
                .div => av / bv,
                else => unreachable,
            };
            @as(*align(1) Vec, @ptrCast(out.ptr + base + c)).* = rv;
        }
        while (c < col_count) : (c += 1) {
            const av: f32 = a[base + c];
            const bv: f32 = b[c];
            out[base + c] = switch (op) {
                .add => av + bv,
                .sub => av - bv,
                .mul => av * bv,
                .div => av / bv,
                else => unreachable,
            };
        }
    }
}

/// Repeat a packed contiguous suffix over an N-D tile.
pub fn contiguousSuffixBinaryF32Packed(
    comptime op: ElemwiseBinaryOp,
    out_bytes: []u8,
    a_bytes: []const u8,
    b_bytes: []const u8,
    elem_count: usize,
    col_count: usize,
) BackendError!void {
    if (col_count == 0) return BackendError.InvalidArgument;
    if ((elem_count % col_count) != 0) return BackendError.InvalidArgument;
    const row_count: usize = elem_count / col_count;
    return contiguousSuffixBinaryF32(op, out_bytes, a_bytes, b_bytes, row_count, col_count);
}

/// f16 variant: arithmetic is done in f32 and narrowed back to f16.
pub fn contiguousSuffixBinaryF16(
    comptime op: ElemwiseBinaryOp,
    out_bytes: []u8,
    a_bytes: []const u8,
    b_bytes: []const u8,
    row_count: usize,
    col_count: usize,
) BackendError!void {
    const out: []align(1) f16 = simd.bytesAsSliceMutUnaligned(f16, out_bytes);
    const a: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, a_bytes);
    const b: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, b_bytes);

    const elem_count: usize = row_count * col_count;
    if (out.len < elem_count or a.len < elem_count or b.len < col_count) {
        return BackendError.InvalidArgument;
    }

    const lanes: usize = comptime simd.lanesF32();
    const VecH = @Vector(lanes, f16);
    const VecF = @Vector(lanes, f32);

    var r: usize = 0;
    while (r < row_count) : (r += 1) {
        const base: usize = r * col_count;
        var c: usize = 0;
        const vec_end: usize = col_count - (col_count % lanes);
        while (c < vec_end) : (c += lanes) {
            const ah: VecH = @as(*align(1) const VecH, @ptrCast(a.ptr + base + c)).*;
            const bh: VecH = @as(*align(1) const VecH, @ptrCast(b.ptr + c)).*;
            const af: VecF = @floatCast(ah);
            const bf: VecF = @floatCast(bh);
            const rf: VecF = switch (op) {
                .add => af + bf,
                .sub => af - bf,
                .mul => af * bf,
                .div => af / bf,
                else => unreachable,
            };
            const rh: VecH = @floatCast(rf);
            @as(*align(1) VecH, @ptrCast(out.ptr + base + c)).* = rh;
        }
        while (c < col_count) : (c += 1) {
            const av: f32 = @as(f32, a[base + c]);
            const bv: f32 = @as(f32, b[c]);
            const rv: f32 = switch (op) {
                .add => av + bv,
                .sub => av - bv,
                .mul => av * bv,
                .div => av / bv,
                else => unreachable,
            };
            out[base + c] = @floatCast(rv);
        }
    }
}

/// Repeat a packed contiguous suffix over an N-D tile.
pub fn contiguousSuffixBinaryF16Packed(
    comptime op: ElemwiseBinaryOp,
    out_bytes: []u8,
    a_bytes: []const u8,
    b_bytes: []const u8,
    elem_count: usize,
    col_count: usize,
) BackendError!void {
    if (col_count == 0) return BackendError.InvalidArgument;
    if ((elem_count % col_count) != 0) return BackendError.InvalidArgument;
    const row_count: usize = elem_count / col_count;
    return contiguousSuffixBinaryF16(op, out_bytes, a_bytes, b_bytes, row_count, col_count);
}

pub fn contiguousSuffixBinaryI32Packed(
    op: ElemwiseBinaryOp,
    out_bytes: []u8,
    a_bytes: []const u8,
    b_bytes: []const u8,
    elem_count: usize,
    col_count: usize,
) BackendError!void {
    if (col_count == 0 or elem_count % col_count != 0) return BackendError.InvalidArgument;
    const out: []align(1) i32 = simd.bytesAsSliceMutUnaligned(i32, out_bytes);
    const a: []align(1) const i32 = simd.bytesAsSliceConstUnaligned(i32, a_bytes);
    const b: []align(1) const i32 = simd.bytesAsSliceConstUnaligned(i32, b_bytes);
    if (out.len < elem_count or a.len < elem_count or b.len < col_count) return BackendError.InvalidArgument;
    return switch (op) {
        inline else => |comptime_op| contiguousSuffixBinaryI32Comptime(comptime_op, out, a, b, elem_count, col_count),
    };
}

fn contiguousSuffixBinaryI32Comptime(
    comptime op: ElemwiseBinaryOp,
    out: []align(1) i32,
    a: []align(1) const i32,
    b: []align(1) const i32,
    elem_count: usize,
    col_count: usize,
) BackendError!void {
    const rows = elem_count / col_count;
    const lanes: usize = comptime simd.lanesF32();
    const Vec = @Vector(lanes, i32);
    const zeros: Vec = @splat(0);
    const ones: Vec = @splat(1);
    for (0..rows) |row| {
        const base = row * col_count;
        var col: usize = 0;
        if (op != .div) {
            const vec_end = col_count - (col_count % lanes);
            while (col < vec_end) : (col += lanes) {
                const av: Vec = @as(*align(1) const Vec, @ptrCast(a.ptr + base + col)).*;
                const bv: Vec = @as(*align(1) const Vec, @ptrCast(b.ptr + col)).*;
                const rv: Vec = switch (op) {
                    .add => av + bv,
                    .sub => av - bv,
                    .mul => av * bv,
                    .eq => @select(i32, av == bv, ones, zeros),
                    .ne => @select(i32, av != bv, ones, zeros),
                    .lt => @select(i32, av < bv, ones, zeros),
                    .gt => @select(i32, av > bv, ones, zeros),
                    .le => @select(i32, av <= bv, ones, zeros),
                    .ge => @select(i32, av >= bv, ones, zeros),
                    .div => unreachable,
                };
                @as(*align(1) Vec, @ptrCast(out.ptr + base + col)).* = rv;
            }
        }
        while (col < col_count) : (col += 1) {
            const av = a[base + col];
            const bv = b[col];
            out[base + col] = scalarBinaryI32(op, av, bv);
        }
    }
}

inline fn scalarBinaryI32(comptime op: ElemwiseBinaryOp, a: i32, b: i32) i32 {
    return switch (op) {
        .add => a + b,
        .sub => a - b,
        .mul => a * b,
        .div => if (b == 0) 0 else @divTrunc(a, b),
        .eq => @intFromBool(a == b),
        .ne => @intFromBool(a != b),
        .lt => @intFromBool(a < b),
        .gt => @intFromBool(a > b),
        .le => @intFromBool(a <= b),
        .ge => @intFromBool(a >= b),
    };
}

fn broadcastElementIndex(
    linear: usize,
    output_shape: []const usize,
    input: types.Layout,
    broadcast_axes: u8,
    elem_bytes: usize,
) BackendError!usize {
    const out_rank = output_shape.len;
    const in_rank: usize = input.rank;
    if (out_rank == 0 or out_rank > 8 or in_rank == 0 or in_rank > out_rank) return BackendError.InvalidArgument;
    const off = out_rank - in_rank;
    var remaining = linear;
    var byte_offset: usize = 0;
    var rev = out_rank;
    while (rev > 0) : (rev -= 1) {
        const axis = rev - 1;
        const dim = output_shape[axis];
        if (dim == 0) return BackendError.InvalidArgument;
        const coord = remaining % dim;
        remaining /= dim;
        if (axis < off or (broadcast_axes & (@as(u8, 1) << @intCast(axis))) != 0) continue;
        const in_axis = axis - off;
        const stride = input.strides_bytes[in_axis];
        if (stride < 0) return BackendError.InvalidArgument;
        byte_offset = std.math.add(usize, byte_offset, std.math.mul(usize, coord, @intCast(stride)) catch return BackendError.InvalidArgument) catch return BackendError.InvalidArgument;
    }
    if (byte_offset % elem_bytes != 0) return BackendError.InvalidArgument;
    return byte_offset / elem_bytes;
}

pub fn elemwiseBroadcastF32(
    op: ElemwiseBinaryOp,
    out_view: types.BufferViewMut,
    a_view: types.BufferViewConst,
    b_view: types.BufferViewConst,
    a_broadcast_axes: u8,
    b_broadcast_axes: u8,
) BackendError!void {
    const out: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, out_view.bytes);
    const a: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, a_view.bytes);
    const b: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, b_view.bytes);
    for (out, 0..) |*dst, linear| {
        const ai = try broadcastElementIndex(linear, out_view.layout.shape, a_view.layout, a_broadcast_axes, @sizeOf(f32));
        const bi = try broadcastElementIndex(linear, out_view.layout.shape, b_view.layout, b_broadcast_axes, @sizeOf(f32));
        if (ai >= a.len or bi >= b.len) return BackendError.InvalidArgument;
        dst.* = switch (op) {
            .add => a[ai] + b[bi],
            .sub => a[ai] - b[bi],
            .mul => a[ai] * b[bi],
            .div => a[ai] / b[bi],
            else => return BackendError.InvalidArgument,
        };
    }
}

pub fn elemwiseBroadcastF16(
    op: ElemwiseBinaryOp,
    out_view: types.BufferViewMut,
    a_view: types.BufferViewConst,
    b_view: types.BufferViewConst,
    a_broadcast_axes: u8,
    b_broadcast_axes: u8,
) BackendError!void {
    const out: []align(1) f16 = simd.bytesAsSliceMutUnaligned(f16, out_view.bytes);
    const a: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, a_view.bytes);
    const b: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, b_view.bytes);
    for (out, 0..) |*dst, linear| {
        const ai = try broadcastElementIndex(linear, out_view.layout.shape, a_view.layout, a_broadcast_axes, @sizeOf(f16));
        const bi = try broadcastElementIndex(linear, out_view.layout.shape, b_view.layout, b_broadcast_axes, @sizeOf(f16));
        if (ai >= a.len or bi >= b.len) return BackendError.InvalidArgument;
        const av: f32 = @floatCast(a[ai]);
        const bv: f32 = @floatCast(b[bi]);
        dst.* = @floatCast(switch (op) {
            .add => av + bv,
            .sub => av - bv,
            .mul => av * bv,
            .div => av / bv,
            else => return BackendError.InvalidArgument,
        });
    }
}

pub fn elemwiseBroadcastI32(
    op: ElemwiseBinaryOp,
    out_view: types.BufferViewMut,
    a_view: types.BufferViewConst,
    b_view: types.BufferViewConst,
    a_broadcast_axes: u8,
    b_broadcast_axes: u8,
) BackendError!void {
    const out: []align(1) i32 = simd.bytesAsSliceMutUnaligned(i32, out_view.bytes);
    const a: []align(1) const i32 = simd.bytesAsSliceConstUnaligned(i32, a_view.bytes);
    const b: []align(1) const i32 = simd.bytesAsSliceConstUnaligned(i32, b_view.bytes);
    for (out, 0..) |*dst, linear| {
        const ai = try broadcastElementIndex(linear, out_view.layout.shape, a_view.layout, a_broadcast_axes, @sizeOf(i32));
        const bi = try broadcastElementIndex(linear, out_view.layout.shape, b_view.layout, b_broadcast_axes, @sizeOf(i32));
        if (ai >= a.len or bi >= b.len) return BackendError.InvalidArgument;
        const av = a[ai];
        const bv = b[bi];
        dst.* = switch (op) {
            .add => av + bv,
            .sub => av - bv,
            .mul => av * bv,
            .div => if (bv == 0) 0 else @divTrunc(av, bv),
            .eq => @intFromBool(av == bv),
            .ne => @intFromBool(av != bv),
            .lt => @intFromBool(av < bv),
            .gt => @intFromBool(av > bv),
            .le => @intFromBool(av <= bv),
            .ge => @intFromBool(av >= bv),
        };
    }
}

pub fn elemwiseBinaryF32(
    op: ElemwiseBinaryOp,
    out_bytes: []u8,
    a_bytes: []const u8,
    b_bytes: []const u8,
    elem_count: usize,
) BackendError!void {
    const out: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, out_bytes);
    const a: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, a_bytes);
    const b: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, b_bytes);

    if (out.len < elem_count or a.len < elem_count or b.len < elem_count) {
        return BackendError.InvalidArgument;
    }

    const lanes: usize = comptime simd.lanesF32();
    const Vec = @Vector(lanes, f32);

    var i: usize = 0;
    const vec_end: usize = elem_count - (elem_count % lanes);
    while (i < vec_end) : (i += lanes) {
        const av: Vec = @as(*align(1) const Vec, @ptrCast(a.ptr + i)).*;
        const bv: Vec = @as(*align(1) const Vec, @ptrCast(b.ptr + i)).*;

        const rv: Vec = switch (op) {
            .add => av + bv,
            .sub => av - bv,
            .mul => av * bv,
            .div => av / bv,
            else => unreachable, // comparisons produce i32 and route to elemwiseBinaryI32
        };

        @as(*align(1) Vec, @ptrCast(out.ptr + i)).* = rv;
    }

    switch (op) {
        .add => {
            while (i < elem_count) : (i += 1) out[i] = a[i] + b[i];
        },
        .sub => {
            while (i < elem_count) : (i += 1) out[i] = a[i] - b[i];
        },
        .mul => {
            while (i < elem_count) : (i += 1) out[i] = a[i] * b[i];
        },
        .div => {
            while (i < elem_count) : (i += 1) out[i] = a[i] / b[i];
        },
        else => unreachable,
    }
}

/// Integer elementwise binary: arithmetic (add/sub/mul/div) and comparisons
/// (eq/ne/lt/gt/le/ge -> 1/0). Inputs and output are i32 (comparisons also output i32).
pub fn elemwiseBinaryI32(
    op: ElemwiseBinaryOp,
    out_bytes: []u8,
    a_bytes: []const u8,
    b_bytes: []const u8,
    elem_count: usize,
) BackendError!void {
    const out: []align(1) i32 = simd.bytesAsSliceMutUnaligned(i32, out_bytes);
    const a: []align(1) const i32 = simd.bytesAsSliceConstUnaligned(i32, a_bytes);
    const b: []align(1) const i32 = simd.bytesAsSliceConstUnaligned(i32, b_bytes);
    if (out.len < elem_count or a.len < elem_count or b.len < elem_count) {
        return BackendError.InvalidArgument;
    }
    return switch (op) {
        inline else => |comptime_op| elemwiseBinaryI32Comptime(comptime_op, out, a, b, elem_count),
    };
}

fn elemwiseBinaryI32Comptime(
    comptime op: ElemwiseBinaryOp,
    out: []align(1) i32,
    a: []align(1) const i32,
    b: []align(1) const i32,
    elem_count: usize,
) BackendError!void {
    const lanes: usize = comptime simd.lanesF32();
    const Vec = @Vector(lanes, i32);
    const zeros: Vec = @splat(0);
    const ones: Vec = @splat(1);
    var i: usize = 0;
    if (op != .div) {
        const vec_end = elem_count - (elem_count % lanes);
        while (i < vec_end) : (i += lanes) {
            const av: Vec = @as(*align(1) const Vec, @ptrCast(a.ptr + i)).*;
            const bv: Vec = @as(*align(1) const Vec, @ptrCast(b.ptr + i)).*;
            const rv: Vec = switch (op) {
                .add => av + bv,
                .sub => av - bv,
                .mul => av * bv,
                .eq => @select(i32, av == bv, ones, zeros),
                .ne => @select(i32, av != bv, ones, zeros),
                .lt => @select(i32, av < bv, ones, zeros),
                .gt => @select(i32, av > bv, ones, zeros),
                .le => @select(i32, av <= bv, ones, zeros),
                .ge => @select(i32, av >= bv, ones, zeros),
                .div => unreachable,
            };
            @as(*align(1) Vec, @ptrCast(out.ptr + i)).* = rv;
        }
    }
    while (i < elem_count) : (i += 1) out[i] = scalarBinaryI32(op, a[i], b[i]);
}

pub fn elemwiseBinaryF16(
    op: ElemwiseBinaryOp,
    out_bytes: []u8,
    a_bytes: []const u8,
    b_bytes: []const u8,
    elem_count: usize,
) BackendError!void {
    const out: []align(1) f16 = simd.bytesAsSliceMutUnaligned(f16, out_bytes);
    const a: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, a_bytes);
    const b: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, b_bytes);

    if (out.len < elem_count or a.len < elem_count or b.len < elem_count) {
        return BackendError.InvalidArgument;
    }

    // SIMD path: load f16 vectors, widen to f32 for arithmetic, narrow back.
    // This keeps behavior consistent with the scalar reference implementation.
    const lanes: usize = comptime simd.lanesF32();
    const VecH = @Vector(lanes, f16);
    const VecF = @Vector(lanes, f32);

    var i: usize = 0;
    const vec_end: usize = elem_count - (elem_count % lanes);
    while (i < vec_end) : (i += lanes) {
        const ah: VecH = @as(*align(1) const VecH, @ptrCast(a.ptr + i)).*;
        const bh: VecH = @as(*align(1) const VecH, @ptrCast(b.ptr + i)).*;
        const af: VecF = @floatCast(ah);
        const bf: VecF = @floatCast(bh);

        const rf: VecF = switch (op) {
            .add => af + bf,
            .sub => af - bf,
            .mul => af * bf,
            .div => af / bf,
            else => unreachable,
        };
        const rh: VecH = @floatCast(rf);
        @as(*align(1) VecH, @ptrCast(out.ptr + i)).* = rh;
    }

    switch (op) {
        .add => {
            while (i < elem_count) : (i += 1) {
                out[i] = @floatCast(@as(f32, a[i]) + @as(f32, b[i]));
            }
        },
        .sub => {
            while (i < elem_count) : (i += 1) {
                out[i] = @floatCast(@as(f32, a[i]) - @as(f32, b[i]));
            }
        },
        .mul => {
            while (i < elem_count) : (i += 1) {
                out[i] = @floatCast(@as(f32, a[i]) * @as(f32, b[i]));
            }
        },
        .div => {
            while (i < elem_count) : (i += 1) {
                out[i] = @floatCast(@as(f32, a[i]) / @as(f32, b[i]));
            }
        },
        else => unreachable,
    }
}
