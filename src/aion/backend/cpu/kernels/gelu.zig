// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const types = @import("../../types.zig");
const simd = @import("simd.zig");
const fast = @import("fast_math.zig");

const BackendError = types.BackendError;

fn geluApproxF32Scalar(x: f32) f32 {
    const c: f32 = 0.044715;
    const k: f32 = 0.7978845608028654;
    const x3: f32 = x * x * x;
    const t: f32 = k * (x + c * x3);
    return 0.5 * x * (1.0 + fast.tanhApproxF32(t));
}

pub fn geluF32(
    out_bytes: []u8,
    a_bytes: []const u8,
    elem_count: usize,
) BackendError!void {
    const out: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, out_bytes);
    const a: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, a_bytes);
    if (out.len < elem_count or a.len < elem_count) return BackendError.InvalidArgument;

    const lanes: usize = comptime simd.lanesF32();
    const Vec = @Vector(lanes, f32);

    var i: usize = 0;
    const vec_end: usize = elem_count - (elem_count % lanes);
    while (i < vec_end) : (i += lanes) {
        const xv: Vec = @as(*align(1) const Vec, @ptrCast(a.ptr + i)).*;
        // t = k*(x + c*x^3)
        const c: Vec = @splat(@as(f32, 0.044715));
        const k: Vec = @splat(@as(f32, 0.7978845608028654));
        const t: Vec = k * (xv + c * (xv * xv * xv));
        const tv: Vec = fast.tanhApproxVecF32(lanes, t);
        const half: Vec = @splat(@as(f32, 0.5));
        const one: Vec = @splat(@as(f32, 1.0));
        @as(*align(1) Vec, @ptrCast(out.ptr + i)).* = half * xv * (one + tv);
    }
    while (i < elem_count) : (i += 1) {
        out[i] = geluApproxF32Scalar(a[i]);
    }
}

pub fn geluF16(
    out_bytes: []u8,
    a_bytes: []const u8,
    elem_count: usize,
) BackendError!void {
    const out: []align(1) f16 = simd.bytesAsSliceMutUnaligned(f16, out_bytes);
    const a: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, a_bytes);
    if (out.len < elem_count or a.len < elem_count) return BackendError.InvalidArgument;

    var i: usize = 0;
    while (i < elem_count) : (i += 1) {
        const x: f32 = @floatCast(a[i]);
        out[i] = @floatCast(geluApproxF32Scalar(x));
    }
}
