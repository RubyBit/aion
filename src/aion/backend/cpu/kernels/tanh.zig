// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const types = @import("../../types.zig");
const simd = @import("simd.zig");
const fast = @import("fast_math.zig");

const BackendError = types.BackendError;

pub fn tanhF32(
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
        const av: Vec = @as(*align(1) const Vec, @ptrCast(a.ptr + i)).*;
        const rv: Vec = fast.tanhApproxVecF32(lanes, av);
        @as(*align(1) Vec, @ptrCast(out.ptr + i)).* = rv;
    }
    while (i < elem_count) : (i += 1) {
        out[i] = fast.tanhApproxF32(a[i]);
    }
}

pub fn tanhF16(
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
        out[i] = @floatCast(fast.tanhApproxF32(x));
    }
}
