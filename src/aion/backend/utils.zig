const std = @import("std");
const types = @import("types.zig");

const BackendError = types.BackendError;
const DType = types.DType;
const DTypeInfo = types.DTypeInfo;
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

pub fn requiredByteLen(dtype: DType, layout: Layout) BackendError!usize {
    const count: usize = try elemCount(layout.shape);
    return requiredBytesForElems(dtype, count);
}

/// Checks if layout is packed row-major for a scalar dtype with given element size.
pub fn isPackedRowMajorScalar(layout: Layout, elem_bytes: usize) bool {
    const rank_usize: usize = @as(usize, layout.rank);
    if (rank_usize == 0) return true;
    if (layout.shape.len != rank_usize) return false;
    if (layout.strides_bytes.len != rank_usize) return false;

    var expected: usize = elem_bytes;

    var i: usize = rank_usize;
    while (i != 0) {
        i -= 1;

        const got_isize: isize = layout.strides_bytes[i];
        if (got_isize < 0) return false;

        const got: usize = @intCast(got_isize);
        if (got != expected) return false;
        const dim: usize = layout.shape[i];
        expected = std.math.mul(usize, expected, dim) catch return false;
    }

    return true;
}

/// Checks if layout is packed for a block-quantized dtype.
/// For quant types, the innermost dimension must be a multiple of block_elems,
/// and we treat the buffer as a flat array of blocks.
pub fn isPackedQuant(layout: Layout, di: DTypeInfo) bool {
    const rank_usize: usize = @as(usize, layout.rank);
    if (rank_usize == 0) return true;
    if (layout.shape.len != rank_usize) return false;
    if (layout.strides_bytes.len != rank_usize) return false;

    // Simplified v0: quant tensors are treated as a flat-packed array of blocks.
    // We intentionally do NOT validate any stride pattern for quant dtypes here.
    // The only hard requirement is that the *total element count* is divisible by
    // block_elems, which is enforced by requiredByteLen() in requirePacked().
    _ = di;
    return true;
}

/// Require view to be packed. Works for both scalar and quant types.
pub fn requirePacked(view: anytype) BackendError!void {
    try view.layout.validate();

    const di = view.dtype.info();
    if (di.is_quantized) {
        if (!isPackedQuant(view.layout, di)) return BackendError.InvalidArgument;
    } else {
        if (!isPackedRowMajorScalar(view.layout, di.block_bytes)) return BackendError.InvalidArgument;
    }

    const need: usize = try requiredByteLen(view.dtype, view.layout);
    if (view.bytes.len < need) return BackendError.InvalidArgument;
}

/// Require view to be packed AND scalar (non-quantized). For elemwise ops.
pub fn requirePackedScalar(view: anytype) BackendError!void {
    if (view.dtype.info().is_quantized) return BackendError.Unsupported;
    try requirePacked(view);
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
