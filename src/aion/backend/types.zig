// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

pub const BackendKind = enum(u9) {
    cpu,
    cuda,
    metal,
    vulkan, // I guess with compute shaders
};

pub const BackendCaps = packed struct(u64) {
    simd: bool = false,
    threads: bool = false,
    fp16: bool = false,
    int8: bool = false,
    quant_q4: bool = false,
    quant_q8: bool = false,

    // can implement strided (enables whether to pack/unpack nodes)
    strided_copy: bool = false,

    _pad: u57 = 0, // future additions
};

pub const BackendError = error{ Unsupported, InvalidArgument, ExecutionFailed };

pub const DType = enum(u8) {
    f32,
    f16,
    i8,
    q4_0,
    q8_0,
    /// Signed 32-bit integer (e.g. token/position indices).
    ///
    /// NOTE: Must be appended to preserve stable on-disk / ABI enum codes.
    i32,

    pub fn info(self: DType) DTypeInfo {
        return switch (self) {
            .f32 => .{ .block_elems = 1, .block_bytes = 4, .is_quantized = false },
            .f16 => .{ .block_elems = 1, .block_bytes = 2, .is_quantized = false },
            .i8 => .{ .block_elems = 1, .block_bytes = 1, .is_quantized = false },
            // ggml-compatible block layouts
            .q4_0 => .{ .block_elems = 32, .block_bytes = 18, .is_quantized = true }, // 2B scale + 16B nibbles
            .q8_0 => .{ .block_elems = 32, .block_bytes = 34, .is_quantized = true }, // 2B scale + 32B int8
            .i32 => .{ .block_elems = 1, .block_bytes = 4, .is_quantized = false },
        };
    }

    pub fn isScalar(self: DType) bool {
        return !self.info().is_quantized;
    }
};

/// Describes the memory layout of a dtype (scalar or block-quantized).
pub const DTypeInfo = struct {
    /// Number of logical elements per block (1 for scalar types).
    block_elems: usize,
    /// Bytes per block.
    block_bytes: usize,
    /// True for quantized (block-based) types.
    is_quantized: bool,
};

pub const Layout = struct {
    rank: u8,
    shape: []const usize, // length = rank
    strides_bytes: []const isize, // length = rank (bytes should be non-negative for v0)

    pub fn validate(self: Layout) BackendError!void {
        if (self.shape.len != self.rank) return BackendError.InvalidArgument;
        if (self.strides_bytes.len != self.rank) return BackendError.InvalidArgument;

        var i: usize = 0;
        while (i < @as(usize, self.rank)) : (i += 1) {
            // v0: reject negative strides
            if (self.strides_bytes[i] < 0) return BackendError.InvalidArgument;
        }
    }
};

pub const BufferViewConst = struct { bytes: []const u8, dtype: DType, layout: Layout };

pub const BufferViewMut = struct {
    bytes: []u8,
    dtype: DType,
    layout: Layout,
};

pub const ElemwiseBinaryOp = enum(u8) {
    add,
    sub,
    mul,
    div,
    // Comparisons (appended for on-disk stability). Produce i32 {0,1}; inputs and
    // output are i32. Used to build If/Loop conditions for in-graph decode.
    eq,
    ne,
    lt,
    gt,
    le,
    ge,

    pub fn isComparison(self: ElemwiseBinaryOp) bool {
        return switch (self) {
            .eq, .ne, .lt, .gt, .le, .ge => true,
            else => false,
        };
    }
};

pub const UnaryOp = enum(u8) {
    relu,
    gelu,
    silu,
    sigmoid,
    tanh,
    sqrt,
    /// Natural logarithm (in-graph log-mel front-end). Appended to keep enum ids stable.
    log,
};

pub const ReduceOp = enum(u8) {
    sum,
    mean,
};

pub const PadMode = enum(u8) {
    zero,
    reflect,
};

pub const MatMulParams = struct {
    /// C[M,N] = α(A[M,K] @ B[K,N]) + βC
    m: usize,
    n: usize,
    k: usize,

    /// Leading dimension (row stride) of C. If 0, defaults to n.
    ldc: usize = 0,

    alpha: f32 = 1.0,
    beta: f32 = 0.0,
};
