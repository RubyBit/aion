// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const types = @import("../../types.zig");
const simd = @import("simd.zig");

const BackendError = types.BackendError;

pub fn sumF32Range(bytes: []const u8, start: usize, end: usize) BackendError!f32 {
    if (start >= end) return 0.0;

    const a: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, bytes);
    if (end > a.len) return BackendError.InvalidArgument;

    const lanes: usize = comptime simd.lanesF32();
    const Vec = @Vector(lanes, f32);

    var acc_v: Vec = @splat(@as(f32, 0.0));

    var i: usize = start;
    const count: usize = end - start;
    const vec_end: usize = start + (count - (count % lanes));

    while (i < vec_end) : (i += lanes) {
        const v: Vec = @as(*align(1) const Vec, @ptrCast(a.ptr + i)).*;
        acc_v += v;
    }

    var acc: f32 = @reduce(.Add, acc_v);
    while (i < end) : (i += 1) {
        acc += a[i];
    }

    return acc;
}

pub fn sumF16RangeToF32(bytes: []const u8, start: usize, end: usize) BackendError!f32 {
    if (start >= end) return 0.0;

    const a: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, bytes);
    if (end > a.len) return BackendError.InvalidArgument;

    const lanes: usize = comptime simd.lanesF32();
    const VecH = @Vector(lanes, f16);
    const VecF = @Vector(lanes, f32);

    var acc_v: VecF = @splat(@as(f32, 0.0));

    var i: usize = start;
    const count: usize = end - start;
    const vec_end: usize = start + (count - (count % lanes));

    while (i < vec_end) : (i += lanes) {
        const vh: VecH = @as(*align(1) const VecH, @ptrCast(a.ptr + i)).*;
        const vf: VecF = @floatCast(vh);
        acc_v += vf;
    }

    var acc: f32 = @reduce(.Add, acc_v);
    while (i < end) : (i += 1) {
        acc += @as(f32, a[i]);
    }

    return acc;
}
