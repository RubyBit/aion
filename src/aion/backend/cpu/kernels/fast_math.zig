const std = @import("std");

// Fast (approximate) math helpers for inference kernels.
//
// These routines intentionally trade accuracy for speed. They should be:
// - monotonic (where applicable)
// - finite over reasonable input ranges
// - vectorizable via @Vector operations

pub inline fn clampF32(x: f32, lo: f32, hi: f32) f32 {
    return std.math.clamp(x, lo, hi);
}

pub inline fn clampVecF32(comptime lanes: usize, x: @Vector(lanes, f32), lo: f32, hi: f32) @Vector(lanes, f32) {
    const vlo: @Vector(lanes, f32) = @splat(lo);
    const vhi: @Vector(lanes, f32) = @splat(hi);
    return @min(@max(x, vlo), vhi);
}

// Schraudolph-style exp approximation.
// Good enough for activation functions, especially in ReleaseFast.
// Reference idea: https://nic.schraudolph.org/pubs/Schraudolph99.pdf
pub inline fn expApproxF32(x_in: f32) f32 {
    // Clamp to avoid INF/0 and keep behavior stable.
    const x: f32 = clampF32(x_in, -80.0, 80.0);

    // 12102203 ~= (1<<23)/ln(2)
    // Use signed intermediate so negative x doesn't trap on cast.
    const ix: i32 = @intFromFloat(x * 12102203.0);
    const iy: i32 = ix + 1065353216;
    const i: u32 = @bitCast(iy);
    return @bitCast(i);
}

pub inline fn expApproxVecF32(comptime lanes: usize, x_in: @Vector(lanes, f32)) @Vector(lanes, f32) {
    const x: @Vector(lanes, f32) = clampVecF32(lanes, x_in, -80.0, 80.0);
    const mul: @Vector(lanes, f32) = @splat(@as(f32, 12102203.0));
    const add: @Vector(lanes, i32) = @splat(@as(i32, 1065353216));

    const ix: @Vector(lanes, i32) = @intFromFloat(x * mul);
    const iy: @Vector(lanes, i32) = ix + add;
    const i: @Vector(lanes, u32) = @bitCast(iy);
    return @bitCast(i);
}

pub inline fn sigmoidApproxF32(x: f32) f32 {
    // Stable-ish form to avoid overflow.
    var y: f32 = undefined;
    if (x >= 0.0) {
        const e: f32 = expApproxF32(-x);
        y = 1.0 / (1.0 + e);
    } else {
        const e: f32 = expApproxF32(x);
        y = e / (1.0 + e);
    }
    // Keep within bounds for safety (approx can overshoot slightly).
    return clampF32(y, 0.0, 1.0);
}

pub inline fn sigmoidApproxVecF32(comptime lanes: usize, x_in: @Vector(lanes, f32)) @Vector(lanes, f32) {
    const zero: @Vector(lanes, f32) = @splat(@as(f32, 0.0));
    const one: @Vector(lanes, f32) = @splat(@as(f32, 1.0));

    const mask: @Vector(lanes, bool) = x_in >= zero;

    // x >= 0: 1/(1+exp(-x))
    const e_pos: @Vector(lanes, f32) = expApproxVecF32(lanes, -x_in);
    const pos: @Vector(lanes, f32) = one / (one + e_pos);

    // x < 0: exp(x)/(1+exp(x))
    const e_neg: @Vector(lanes, f32) = expApproxVecF32(lanes, x_in);
    const neg: @Vector(lanes, f32) = e_neg / (one + e_neg);

    const y: @Vector(lanes, f32) = @select(f32, mask, pos, neg);
    return clampVecF32(lanes, y, 0.0, 1.0);
}

pub inline fn tanhApproxF32(x: f32) f32 {
    // tanh(x) = 2*sigmoid(2x) - 1
    const y: f32 = 2.0 * sigmoidApproxF32(2.0 * x) - 1.0;
    return clampF32(y, -1.0, 1.0);
}

pub inline fn tanhApproxVecF32(comptime lanes: usize, x: @Vector(lanes, f32)) @Vector(lanes, f32) {
    const two: @Vector(lanes, f32) = @splat(@as(f32, 2.0));
    const one: @Vector(lanes, f32) = @splat(@as(f32, 1.0));
    const y: @Vector(lanes, f32) = two * sigmoidApproxVecF32(lanes, two * x) - one;
    return clampVecF32(lanes, y, -1.0, 1.0);
}
