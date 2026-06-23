// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
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
/// `op` is comptime to allow full specialization for hot paths like bias-add.
pub fn broadcastLastDimBinaryF32(
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

/// Broadcast over last dim for packed N-D (row_count derived from elem_count / col_count).
pub fn broadcastLastDimBinaryF32Packed(
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
    return broadcastLastDimBinaryF32(op, out_bytes, a_bytes, b_bytes, row_count, col_count);
}

/// f16 variant: arithmetic is done in f32 and narrowed back to f16.
pub fn broadcastLastDimBinaryF16(
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

/// Broadcast over last dim for packed N-D (row_count derived from elem_count / col_count).
pub fn broadcastLastDimBinaryF16Packed(
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
    return broadcastLastDimBinaryF16(op, out_bytes, a_bytes, b_bytes, row_count, col_count);
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
    var i: usize = 0;
    while (i < elem_count) : (i += 1) {
        const av = a[i];
        const bv = b[i];
        out[i] = switch (op) {
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
