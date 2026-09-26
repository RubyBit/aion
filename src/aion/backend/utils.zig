// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const types = @import("types.zig");

const BackendError = types.BackendError;
const DType = types.DType;
const Layout = types.Layout;
const MatMulParams = types.MatMulParams;

/// Returns the number of bytes required to store `elem_count` logical elements of `dtype`.
/// For block-quantized types, `elem_count` must be a multiple of block_elems.
pub fn requiredBytesForElems(dtype: DType, elem_count: usize) BackendError!usize {
    const di = dtype.info();
    if (di.is_quantized) {
        // elem_count must be divisible by block size
        if (elem_count % di.block_elems != 0) return BackendError.InvalidArgument;
        const num_blocks = elem_count / di.block_elems;
        return std.math.mul(usize, num_blocks, di.block_bytes) catch return BackendError.InvalidArgument;
    } else {
        return std.math.mul(usize, elem_count, di.block_bytes) catch return BackendError.InvalidArgument;
    }
}

/// Returns the number of logical elements described by `shape`.
/// Errors on overflow.
pub fn elemCount(shape: []const usize) BackendError!usize {
    var count: usize = 1;
    for (shape) |d| {
        count = std.math.mul(usize, count, d) catch return BackendError.InvalidArgument;
    }
    return count;
}

pub fn requireSameShape(a: Layout, b: Layout) BackendError!void {
    if (a.rank != b.rank) return BackendError.InvalidArgument;
    if (a.shape.len != b.shape.len) return BackendError.InvalidArgument;

    var i: usize = 0;
    while (i < a.shape.len) : (i += 1) {
        if (a.shape[i] != b.shape[i]) return BackendError.InvalidArgument;
    }
}

pub fn requireMatMulShapes(params: MatMulParams, c: Layout, a: Layout, b: Layout) BackendError!void {
    // v0: matmul is only defined for rank-2 packed tensors.
    if (a.rank != 2 or b.rank != 2 or c.rank != 2) return BackendError.InvalidArgument;
    if (a.shape.len != 2 or b.shape.len != 2 or c.shape.len != 2) return BackendError.InvalidArgument;

    if (a.shape[0] != params.m or a.shape[1] != params.k) return BackendError.InvalidArgument;
    if (b.shape[0] != params.k or b.shape[1] != params.n) return BackendError.InvalidArgument;
    if (c.shape[0] != params.m or c.shape[1] != params.n) return BackendError.InvalidArgument;
}
