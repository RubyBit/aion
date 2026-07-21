// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Core f32 -> block-quantized packing.
//!
//! This is the single source of truth for producing packed-quant bytes from f32
//! values. It mirrors the packed-quant convention consumed by
//! `storage.TiledTensor.writeFromPackedQuant` (see its doc comment): along
//! `quant_axis`, every `block_elems` consecutive elements form one `block_bytes`
//! block, and `packed_bytes` is row-major over the resulting block-space shape.
//!
//! Only q8_0 is implemented today (2B f16 scale + 32×i8). The public entry point
//! is dtype-parameterized so q4_0 can be added in one place without touching
//! callers.

const std = @import("std");
const types = @import("../backend/types.zig");

pub const QuantizeError = error{
    /// The dtype is not a block-quantized dtype (or not yet supported here).
    Unsupported,
    /// `shape[quant_axis]` is not a multiple of the dtype's `block_elems`, the
    /// axis is out of range, or `values.len` != product(shape).
    InvalidArgument,
    OutOfMemory,
};

/// Number of packed bytes needed to store `shape` at `dtype` (whole blocks).
pub fn packedLen(dtype: types.DType, shape: []const usize) QuantizeError!usize {
    const di = dtype.info();
    if (!di.is_quantized) return QuantizeError.Unsupported;
    var total: usize = 1;
    for (shape) |d| total = std.math.mul(usize, total, d) catch return QuantizeError.InvalidArgument;
    if (total % di.block_elems != 0) return QuantizeError.InvalidArgument;
    const blocks = total / di.block_elems;
    return std.math.mul(usize, blocks, di.block_bytes) catch return QuantizeError.InvalidArgument;
}

/// Quantize row-major f32 `values` of `shape` into packed-quant bytes, blocking
/// along `quant_axis`. Caller owns the returned buffer.
pub fn quantizeF32(
    allocator: std.mem.Allocator,
    dtype: types.DType,
    shape: []const usize,
    quant_axis: usize,
    values: []const f32,
) QuantizeError![]u8 {
    const di = dtype.info();
    if (!di.is_quantized) return QuantizeError.Unsupported;
    if (dtype != .q8_0) return QuantizeError.Unsupported; // q4_0 slots in here later.
    if (quant_axis >= shape.len) return QuantizeError.InvalidArgument;

    const block_elems = di.block_elems; // 32 for q8_0
    if (shape[quant_axis] % block_elems != 0) return QuantizeError.InvalidArgument;

    var total: usize = 1;
    for (shape) |d| total = std.math.mul(usize, total, d) catch return QuantizeError.InvalidArgument;
    if (values.len != total) return QuantizeError.InvalidArgument;

    // Row-major element strides over `shape`.
    const rank = shape.len;
    var strides = try allocator.alloc(usize, rank);
    defer allocator.free(strides);
    {
        var s: usize = 1;
        var d: usize = rank;
        while (d > 0) {
            d -= 1;
            strides[d] = s;
            s *= shape[d];
        }
    }
    const axis_stride = strides[quant_axis];

    // Block-space shape = `shape` with the quant axis divided by block_elems.
    var block_shape = try allocator.alloc(usize, rank);
    defer allocator.free(block_shape);
    var num_blocks: usize = 1;
    for (shape, 0..) |d, i| {
        block_shape[i] = if (i == quant_axis) d / block_elems else d;
        num_blocks *= block_shape[i];
    }

    const out = try allocator.alloc(u8, num_blocks * di.block_bytes);
    errdefer allocator.free(out);

    // Walk block-space in row-major order (matches the packed convention).
    var coord = try allocator.alloc(usize, rank);
    defer allocator.free(coord);
    @memset(coord, 0);

    var blk: usize = 0;
    while (blk < num_blocks) : (blk += 1) {
        // Base element index for this block: block coord maps to element coord,
        // with the quant-axis coordinate scaled back up by block_elems.
        var base: usize = 0;
        for (coord, 0..) |c, i| {
            const elem_c = if (i == quant_axis) c * block_elems else c;
            base += elem_c * strides[i];
        }
        packQ8Block(values, base, axis_stride, out[blk * 34 ..][0..34]);

        // Increment the row-major block coordinate.
        var d: usize = rank;
        while (d > 0) {
            d -= 1;
            coord[d] += 1;
            if (coord[d] < block_shape[d]) break;
            coord[d] = 0;
        }
    }

    return out;
}

/// Pack one q8_0 block: 32 f32 values at `values[base + t*stride]` for t in 0..32
/// into `dst` (2B little-endian f16 scale + 32 i8).
fn packQ8Block(values: []const f32, base: usize, stride: usize, dst: *[34]u8) void {
    var absmax: f32 = 0;
    var t: usize = 0;
    while (t < 32) : (t += 1) absmax = @max(absmax, @abs(values[base + t * stride]));

    const scale: f32 = if (absmax == 0) 1 else absmax / 127.0;
    const sf16: f16 = @floatCast(scale);
    std.mem.writeInt(u16, dst[0..2], @bitCast(sf16), .little);
    // Quantize against the *stored* (f16-rounded) scale, not the full-f32 scale:
    // reconstruction is `code * f16_scale`, so choosing `code = round(v / f16_scale)`
    // minimizes |v - code*f16_scale| (optimal for the scale actually used at dequant).
    const eff: f32 = @floatCast(sf16);
    const inv: f32 = if (absmax == 0 or eff == 0) 0 else 1.0 / eff;

    t = 0;
    while (t < 32) : (t += 1) {
        var q: i32 = @intFromFloat(@round(values[base + t * stride] * inv));
        q = @max(@as(i32, -128), @min(@as(i32, 127), q));
        dst[2 + t] = @bitCast(@as(i8, @intCast(q)));
    }
}

test "quantize: q8_0 [K,N] matches the packQ8WeightKN reference (axis 0)" {
    const allocator = std.testing.allocator;
    const K: usize = 64;
    const N: usize = 5;
    var vals: [K * N]f32 = undefined;
    for (&vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) * 0.1;

    const got = try quantizeF32(allocator, .q8_0, &[_]usize{ K, N }, 0, &vals);
    defer allocator.free(got);

    // Reference: blocks along K, per column (see test_api.packQ8WeightKN).
    const kb = K / 32;
    var expect: [kb * N * 34]u8 = undefined;
    for (0..kb) |b| {
        for (0..N) |j| {
            var absmax: f32 = 0;
            for (0..32) |t| absmax = @max(absmax, @abs(vals[(b * 32 + t) * N + j]));
            const scale: f32 = if (absmax == 0) 1 else absmax / 127.0;
            const sf16: f16 = @floatCast(scale);
            const eff: f32 = @floatCast(sf16);
            const inv: f32 = if (absmax == 0 or eff == 0) 0 else 1.0 / eff;
            const off = (b * N + j) * 34;
            std.mem.writeInt(u16, expect[off .. off + 2][0..2], @bitCast(sf16), .little);
            for (0..32) |t| {
                var q: i32 = @intFromFloat(@round(vals[(b * 32 + t) * N + j] * inv));
                q = @max(@as(i32, -128), @min(@as(i32, 127), q));
                expect[off + 2 + t] = @bitCast(@as(i8, @intCast(q)));
            }
        }
    }
    try std.testing.expectEqualSlices(u8, expect[0..], got);
}
