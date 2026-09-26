// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

pub const BackendKind = enum(u9) {
    cpu,
    cuda,
    metal,
    vulkan, // I guess with compute shaders
    /// Portable GPU backend over WebGPU (wgpu-native): runs on D3D12/Metal/Vulkan.
    webgpu,
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
            // Block layouts: an f16 scale, then the block's quantized values.
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
    /// Gated activation: `gate(a, b) = act(a) * b`, where `act` is the `UnaryOp`
    /// carried alongside the op. GEGLU (`gelu`), SwiGLU (`silu`), GLU (`sigmoid`) and
    /// ReGLU (`relu`) are all this one op — gating is a parameter, not ten op tags.
    /// Same shapes on both sides, f32 only, no broadcast. Appended for on-disk stability.
    gate,

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

/// How an `[n, k]` q8_0 tensor's blocks are laid out.
///
/// `row_major` is the plain contract: one row is an unbroken run of
/// `cols / 32` 34-byte blocks. `lanes*` groups `W` rows so an integer dot's `W`
/// output lanes are `W` rows at once: for each block index, the group holds its
/// `W` f16 scales, then the 32 quantized values in eight 4-byte chunks, each
/// chunk with its `W` rows side by side. One `W`-lane dot then covers `W` rows
/// and the scaling after it is paid once per block for all of them, not once
/// per row. `W` is whatever the target's kernel reads best — a CPU's dot lane
/// count, a GPU's 32 threads per 128-byte load. The blocks' bytes are unchanged,
/// only placed.
pub const QuantBlockOrder = enum(u8) {
    row_major,
    lanes4,
    lanes8,
    lanes16,
    lanes32,

    pub const BLOCK_BYTES: usize = 34;
    const CHUNK_BYTES: usize = 4;
    const CHUNKS: usize = 32 / CHUNK_BYTES;

    /// Rows per group: one for `row_major`.
    pub fn groupRows(self: QuantBlockOrder) usize {
        return switch (self) {
            .row_major => 1,
            .lanes4 => 4,
            .lanes8 => 8,
            .lanes16 => 16,
            .lanes32 => 32,
        };
    }

    /// The order grouping `rows` rows, if there is one.
    pub fn withGroup(rows: usize) ?QuantBlockOrder {
        return switch (rows) {
            1 => .row_major,
            4 => .lanes4,
            8 => .lanes8,
            16 => .lanes16,
            32 => .lanes32,
            else => null,
        };
    }

    /// Byte offset, inside the tensor, of the group segment holding row `r`'s
    /// block `kb`, for rows of `blocks` blocks. A segment is `groupRows()` blocks.
    fn segment(self: QuantBlockOrder, blocks: usize, r: usize, kb: usize) usize {
        const g = self.groupRows();
        return ((r / g) * blocks + kb) * g * BLOCK_BYTES;
    }

    /// Offset of row `r`'s block `kb` scale.
    pub fn scaleAt(self: QuantBlockOrder, blocks: usize, r: usize, kb: usize) usize {
        return self.segment(blocks, r, kb) + (r % self.groupRows()) * 2;
    }

    /// Offset of row `r`'s block `kb` values `[4 * chunk, 4 * chunk + 4)`.
    pub fn chunkAt(self: QuantBlockOrder, blocks: usize, r: usize, kb: usize, chunk: usize) usize {
        const g = self.groupRows();
        return self.segment(blocks, r, kb) + 2 * g + (chunk * g + r % g) * CHUNK_BYTES;
    }

    /// Write a whole 34-byte block to its place in `bytes`.
    pub fn storeBlock(self: QuantBlockOrder, bytes: []u8, blocks: usize, r: usize, kb: usize, block: []const u8) void {
        @memcpy(bytes[self.scaleAt(blocks, r, kb)..][0..2], block[0..2]);
        for (0..CHUNKS) |c| @memcpy(bytes[self.chunkAt(blocks, r, kb, c)..][0..CHUNK_BYTES], block[2 + c * CHUNK_BYTES ..][0..CHUNK_BYTES]);
    }

    /// Read a whole 34-byte block back out of `bytes`.
    pub fn loadBlock(self: QuantBlockOrder, bytes: []const u8, blocks: usize, r: usize, kb: usize, block: []u8) void {
        @memcpy(block[0..2], bytes[self.scaleAt(blocks, r, kb)..][0..2]);
        for (0..CHUNKS) |c| @memcpy(block[2 + c * CHUNK_BYTES ..][0..CHUNK_BYTES], bytes[self.chunkAt(blocks, r, kb, c)..][0..CHUNK_BYTES]);
    }
};

pub const MatMulParams = struct {
    /// C[M,N] = α(A[M,K] @ B[K,N]) + βC
    m: usize,
    n: usize,
    k: usize,

    /// Leading dimension (row stride) of C. If 0, defaults to n.
    ldc: usize = 0,
    /// Row stride of A, in elements. If 0, defaults to k.
    lda: usize = 0,
    /// Row stride of B -- elements for scalar B, blocks for a K-blocked quantized
    /// B. If 0, defaults to n.
    ldb: usize = 0,

    alpha: f32 = 1.0,
    beta: f32 = 0.0,
};
