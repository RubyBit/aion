// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Core f32 -> block-quantized packing.
//!
//! This is the single source of truth for producing packed-quant bytes from f32
//! values. It mirrors the packed-quant convention consumed by
//! `storage.Tensor.writeFromPackedQuant` (see its doc comment): along
//! `quant_axis`, every `block_elems` consecutive elements form one `block_bytes`
//! block, and `packed_bytes` is row-major over the resulting block-space shape.
//!
//! One entry point, `quantizeBlocks`, packs any range of blocks, so a large tensor
//! is quantized a chunk at a time (see `Context.quantize`). Only q8_0 is implemented
//! today (2B f16 scale + 32×i8); it is dtype-parameterized so q4_0 slots in one place.

const std = @import("std");
const types = @import("../backend/types.zig");
const max_rank = @import("../runtime/tensor_store.zig").max_rank;

pub const QuantizeError = error{
    /// The dtype is not a block-quantized dtype (or not yet supported here).
    Unsupported,
    /// `shape[quant_axis]` is not a multiple of the dtype's `block_elems`, the
    /// axis is out of range, or `values` does not cover the blocks asked for.
    InvalidArgument,
    OutOfMemory,
};

/// Quantize packed blocks `[first_block, first_block + out.len / block_bytes)` of a
/// tensor of `shape`, from `values` holding its row-major elements from element
/// `first_elem` on. `values` must hold every element those blocks cover; a caller
/// quantizing a large tensor a chunk at a time passes one chunk's worth.
pub fn quantizeBlocks(
    dtype: types.DType,
    shape: []const usize,
    quant_axis: usize,
    values: []const f32,
    first_elem: usize,
    first_block: usize,
    out: []u8,
) QuantizeError!void {
    const di = dtype.info();
    if (!di.is_quantized) return QuantizeError.Unsupported;
    if (dtype != .q8_0) return QuantizeError.Unsupported; // q4_0 slots in here later.
    const rank = shape.len;
    if (rank == 0 or rank > max_rank or quant_axis >= rank) return QuantizeError.InvalidArgument;
    const block_elems = di.block_elems; // 32 for q8_0
    if (shape[quant_axis] % block_elems != 0) return QuantizeError.InvalidArgument;
    if (out.len % di.block_bytes != 0) return QuantizeError.InvalidArgument;

    // Row-major element strides, and the block-space shape: `shape` with the quant
    // axis divided by `block_elems`.
    var strides: [max_rank]usize = undefined;
    var block_shape: [max_rank]usize = undefined;
    var s: usize = 1;
    var d: usize = rank;
    while (d > 0) {
        d -= 1;
        strides[d] = s;
        s = std.math.mul(usize, s, shape[d]) catch return QuantizeError.InvalidArgument;
        block_shape[d] = if (d == quant_axis) shape[d] / block_elems else shape[d];
    }
    const axis_stride = strides[quant_axis];

    for (0..out.len / di.block_bytes) |i| {
        // Block-space coords of this block; its first element, relative to `values`.
        var rest: usize = first_block + i;
        var base: usize = 0;
        d = rank;
        while (d > 0) {
            d -= 1;
            const c = rest % block_shape[d];
            rest /= block_shape[d];
            base += (if (d == quant_axis) c * block_elems else c) * strides[d];
        }
        if (rest != 0 or base < first_elem) return QuantizeError.InvalidArgument;
        const local = base - first_elem;
        if (local + (block_elems - 1) * axis_stride >= values.len) return QuantizeError.InvalidArgument;
        packQ8Block(values, local, axis_stride, out[i * di.block_bytes ..][0..34]);
    }
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

    const got = try allocator.alloc(u8, K * N / 32 * 34);
    defer allocator.free(got);
    try quantizeBlocks(.q8_0, &[_]usize{ K, N }, 0, &vals, 0, 0, got);

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
