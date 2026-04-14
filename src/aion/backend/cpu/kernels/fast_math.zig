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

fn floorI32(x: f32) i32 {
    // Zig truncates toward zero; adjust to floor for negative fractional values.
    const i: i32 = @intFromFloat(x);
    const xf: f32 = @floatFromInt(i);
    return if (xf > x) (i - 1) else i;
}

fn floorVecI32(comptime lanes: usize, x: @Vector(lanes, f32)) @Vector(lanes, i32) {
    const i: @Vector(lanes, i32) = @intFromFloat(x);
    const xf: @Vector(lanes, f32) = @floatFromInt(i);
    const one_i: @Vector(lanes, i32) = @splat(@as(i32, 1));
    const zero_i: @Vector(lanes, i32) = @splat(@as(i32, 0));
    const fix: @Vector(lanes, i32) = @select(i32, xf > x, one_i, zero_i);
    return i - fix;
}

fn exp2iF32(n: i32) f32 {
    // Construct 2^n as an IEEE-754 float.
    // Valid for n in roughly [-126, 127]. Callers clamp x to keep this safe.
    const biased: i32 = n + 127;
    const bits: u32 = @as(u32, @intCast(biased)) << 23;
    return @bitCast(bits);
}

fn exp2iVecF32(comptime lanes: usize, n: @Vector(lanes, i32)) @Vector(lanes, f32) {
    const bias_i: @Vector(lanes, i32) = @splat(@as(i32, 127));
    const shift_u: @Vector(lanes, u32) = @splat(@as(u32, 23));
    const biased: @Vector(lanes, i32) = n + bias_i;
    const bits: @Vector(lanes, u32) = @as(@Vector(lanes, u32), @bitCast(biased)) << shift_u;
    return @bitCast(bits);
}

fn expPolyF32(t: f32) f32 {
    // 4th-order Taylor for exp(t) on t in ~[-0.35, 0.35].
    // Error term is O(t^5/120) ~ 3e-5 at t=0.35, which is typically sufficient
    // for ~1e-4 sigmoid/tanh absolute accuracy on [-6, 6].
    const c4: f32 = 1.0 / 24.0;
    const c3: f32 = 1.0 / 6.0;
    const c2: f32 = 0.5;
    // 1 + t*(1 + t*(1/2 + t*(1/6 + t*(1/24))))
    return 1.0 + t * (1.0 + t * (c2 + t * (c3 + t * c4)));
}

fn expPolyVecF32(comptime lanes: usize, t: @Vector(lanes, f32)) @Vector(lanes, f32) {
    const c4: @Vector(lanes, f32) = @splat(@as(f32, 1.0 / 24.0));
    const c3: @Vector(lanes, f32) = @splat(@as(f32, 1.0 / 6.0));
    const c2: @Vector(lanes, f32) = @splat(@as(f32, 0.5));
    const one: @Vector(lanes, f32) = @splat(@as(f32, 1.0));
    return one + t * (one + t * (c2 + t * (c3 + t * c4)));
}

// Fast exp approximation.
//
// This uses exp(x) = 2^(x/log(2)) with range reduction to r in [-0.5,0.5]
// and a small polynomial for exp(r*ln2). It is designed to be:
// - SIMD-friendly
// - monotone
// - significantly more accurate than the classic Schraudolph bit-hack
pub inline fn expApproxF32(x_in: f32) f32 {
    const x: f32 = clampF32(x_in, -80.0, 80.0);
    const inv_ln2: f32 = 1.4426950408889634;
    const ln2: f32 = 0.6931471805599453;

    // y = x / ln2
    const y: f32 = x * inv_ln2;
    // n = floor(y + 0.5)  => r = y - n in [-0.5, 0.5]
    const n: i32 = floorI32(y + 0.5);
    const r: f32 = y - @as(f32, @floatFromInt(n));
    const t: f32 = r * ln2;

    return exp2iF32(n) * expPolyF32(t);
}

pub inline fn expApproxVecF32(comptime lanes: usize, x_in: @Vector(lanes, f32)) @Vector(lanes, f32) {
    const x: @Vector(lanes, f32) = clampVecF32(lanes, x_in, -80.0, 80.0);
    const inv_ln2: @Vector(lanes, f32) = @splat(@as(f32, 1.4426950408889634));
    const ln2: @Vector(lanes, f32) = @splat(@as(f32, 0.6931471805599453));

    const y: @Vector(lanes, f32) = x * inv_ln2;
    const half: @Vector(lanes, f32) = @splat(@as(f32, 0.5));
    const n: @Vector(lanes, i32) = floorVecI32(lanes, y + half);
    const r: @Vector(lanes, f32) = y - @as(@Vector(lanes, f32), @floatFromInt(n));
    const t: @Vector(lanes, f32) = r * ln2;

    return exp2iVecF32(lanes, n) * expPolyVecF32(lanes, t);
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

pub const SinCosF32 = struct {
    sin: f32,
    cos: f32,
};

/// Centralized sin/cos helper for kernels.
///
/// NOTE: Kept as a dedicated helper so kernel call-sites can remain stable if
/// we swap in an approximation later.
pub inline fn sinCosFastF32(x: f32) SinCosF32 {
    return .{
        .sin = @floatCast(std.math.sin(@as(f64, x))),
        .cos = @floatCast(std.math.cos(@as(f64, x))),
    };
}
