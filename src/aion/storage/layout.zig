// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Where a tensor's bytes go.
//!
//! A tensor is one row-major buffer. The only exception is a device, which can
//! bind at most so many bytes to one shader: a tensor past that is split along
//! dim 0 into chunks of whole rows, each its own device buffer. Everything here is
//! a pure function of dtype, shape, block axis and that limit, so a tensor's
//! chunking never needs to be chosen, stored in a file, or changed.

const std = @import("std");
const types = @import("../backend/types.zig");
const utils = @import("../backend/utils.zig");

/// Rows of dim 0 a chunk holds a multiple of: a whole quantization block when
/// dim 0 is the block axis, and a whole row group of the grouped block orders
/// (`types.QuantBlockOrder`, up to 32 rows) otherwise.
pub const CHUNK_GRANULE: usize = 32;

/// Rows of dim 0 in each chunk of a `dtype` tensor of `shape` (blocked along
/// `quant_axis`) on a device that binds at most `max_binding_bytes`; 0 means no
/// limit. The whole extent when the tensor fits, which is almost always.
pub fn chunkRows(dtype: types.DType, shape: []const usize, quant_axis: u8, max_binding_bytes: u64) usize {
    if (shape.len == 0) return 0;
    const rows = shape[0];
    if (max_binding_bytes == 0) return rows;
    var inner: usize = 1;
    for (shape[1..]) |d| inner = std.math.mul(usize, inner, d) catch return rows;
    const info = dtype.info();
    // Bytes of one granule of dim 0.
    const granule_bytes: usize = if (info.is_quantized and quant_axis == 0)
        std.math.mul(usize, inner, info.block_bytes) catch return CHUNK_GRANULE
    else
        std.math.mul(usize, utils.requiredBytesForElems(dtype, inner) catch return rows, CHUNK_GRANULE) catch return CHUNK_GRANULE;
    // 3/4 of the limit leaves headroom for allocation rounding and other bindings.
    const budget: u64 = max_binding_bytes / 4 * 3;
    if (granule_bytes == 0) return rows;
    const granules: usize = @intCast(@min(budget / granule_bytes, rows));
    return @min(rows, @max(1, granules) * CHUNK_GRANULE);
}

test "a tensor that fits is one chunk; one that does not splits on dim 0 by whole granules" {
    try std.testing.expectEqual(@as(usize, 1000), chunkRows(.f32, &.{ 1000, 64 }, 0, 0));
    try std.testing.expectEqual(@as(usize, 1000), chunkRows(.f32, &.{ 1000, 64 }, 0, 1 << 30));
    // 64 f32 = 256 B a row, 8 KiB a granule; a 64 KiB limit budgets 48 KiB = 6 granules.
    try std.testing.expectEqual(@as(usize, 192), chunkRows(.f32, &.{ 1000, 64 }, 0, 64 << 10));
    // q8 blocked along dim 0: a granule is one block row of 34-byte blocks.
    const q8_rows = chunkRows(.q8_0, &.{ 4096, 1024 }, 0, 1 << 20);
    try std.testing.expect(q8_rows % 32 == 0 and q8_rows < 4096);
    try std.testing.expect(q8_rows / 32 * 1024 * 34 <= (1 << 20) / 4 * 3);
    // Never below one granule, even past the limit.
    try std.testing.expectEqual(@as(usize, 32), chunkRows(.f32, &.{ 1000, 1 << 20 }, 0, 1 << 20));
}
