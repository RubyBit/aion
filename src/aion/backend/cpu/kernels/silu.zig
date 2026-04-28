// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const types = @import("../../types.zig");
const simd = @import("simd.zig");
const fast = @import("fast_math.zig");

const BackendError = types.BackendError;

pub fn siluF32(
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
        const sv: Vec = fast.sigmoidApproxVecF32(lanes, xv);
        @as(*align(1) Vec, @ptrCast(out.ptr + i)).* = xv * sv;
    }
    while (i < elem_count) : (i += 1) {
        const x: f32 = a[i];
        out[i] = x * fast.sigmoidApproxF32(x);
    }
}

pub fn siluF16(
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
        out[i] = @floatCast(x * fast.sigmoidApproxF32(x));
    }
}
