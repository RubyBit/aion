// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const types = @import("../../types.zig");
const simd = @import("simd.zig");

const BackendError = types.BackendError;

pub fn reluF32(
    out_bytes: []u8,
    a_bytes: []const u8,
    elem_count: usize,
) BackendError!void {
    const out: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, out_bytes);
    const a: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, a_bytes);

    if (out.len < elem_count or a.len < elem_count) {
        return BackendError.InvalidArgument;
    }

    const lanes: usize = comptime simd.lanesF32();
    const Vec = @Vector(lanes, f32);
    const zero: Vec = @splat(@as(f32, 0.0));

    var i: usize = 0;
    const vec_end: usize = elem_count - (elem_count % lanes);
    while (i < vec_end) : (i += lanes) {
        const av: Vec = @as(*align(1) const Vec, @ptrCast(a.ptr + i)).*;
        const rv: Vec = @max(av, zero);
        @as(*align(1) Vec, @ptrCast(out.ptr + i)).* = rv;
    }

    while (i < elem_count) : (i += 1) {
        const x: f32 = a[i];
        out[i] = if (x > 0.0) x else 0.0;
    }
}

pub fn reluF16(
    out_bytes: []u8,
    a_bytes: []const u8,
    elem_count: usize,
) BackendError!void {
    const out: []align(1) f16 = simd.bytesAsSliceMutUnaligned(f16, out_bytes);
    const a: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, a_bytes);

    if (out.len < elem_count or a.len < elem_count) {
        return BackendError.InvalidArgument;
    }

    const lanes: usize = comptime simd.lanesF32();
    const VecH = @Vector(lanes, f16);
    const VecF = @Vector(lanes, f32);
    const zero: VecF = @splat(@as(f32, 0.0));

    var i: usize = 0;
    const vec_end: usize = elem_count - (elem_count % lanes);
    while (i < vec_end) : (i += lanes) {
        const ah: VecH = @as(*align(1) const VecH, @ptrCast(a.ptr + i)).*;
        const af: VecF = @floatCast(ah);
        const rf: VecF = @max(af, zero);
        const rh: VecH = @floatCast(rf);
        @as(*align(1) VecH, @ptrCast(out.ptr + i)).* = rh;
    }

    while (i < elem_count) : (i += 1) {
        const x: f32 = @as(f32, a[i]);
        out[i] = @floatCast(if (x > 0.0) x else 0.0);
    }
}
