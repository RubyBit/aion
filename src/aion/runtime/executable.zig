const std = @import("std");
const types = @import("../backend/types.zig");

pub const TensorId = u32;
const MAX_RANK: usize = 8;
const MAX_CONCAT_INPUTS: usize = 16;

pub const StepMatMulTiled = struct { c: TensorId, a: TensorId, b: TensorId, alpha: f32, beta: f32 };
pub const StepElemwiseBinaryTiled = struct { op: types.ElemwiseBinaryOp, out: TensorId, a: TensorId, b: TensorId };
pub const StepBroadcastLastDimBinaryTiled = struct { op: types.ElemwiseBinaryOp, out: TensorId, a: TensorId, b: TensorId };
pub const StepUnaryTiled = struct { op: types.UnaryOp, out: TensorId, a: TensorId };
pub const StepSoftmaxTiled = struct { out: TensorId, a: TensorId, axis: i32 };
pub const StepConv1DTiled = struct {
    out: TensorId,
    x: TensorId,
    w: TensorId,
    bias: ?TensorId,
    stride: usize,
    dilation: usize,
    pad_left: usize,
    pad_right: usize,
    pad_mode: types.PadMode,
    groups: usize,
};
pub const StepConv2DTiled = struct {
    out: TensorId,
    x: TensorId,
    w: TensorId,
    bias: ?TensorId,
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    pad_top: usize,
    pad_bottom: usize,
    pad_left: usize,
    pad_right: usize,
    pad_mode: types.PadMode,
    groups: usize,
};
pub const StepLayerNormTiled = struct { out: TensorId, x: TensorId, gamma: TensorId, beta: TensorId, eps: f32 };
pub const StepRMSNormTiled = struct { out: TensorId, x: TensorId, gamma: TensorId, beta: TensorId, eps: f32 };
pub const StepAttentionTiled = struct { out: TensorId, q: TensorId, k: TensorId, v: TensorId, scale: f32, causal: bool };
pub const StepMultiHeadAttentionTiled = struct { out: TensorId, q: TensorId, k: TensorId, v: TensorId, scale: f32, causal: bool, heads: usize };
pub const StepMultiHeadAttentionCachedTiled = struct {
    out: TensorId,
    q: TensorId,
    k_cache: TensorId,
    v_cache: TensorId,
    positions: TensorId,
    end_index: TensorId,
    scale: f32,
    causal: bool,
    sliding_window: usize,
    attn_logits_soft_cap: f32,
};
pub const StepReduceAll = struct { op: types.ReduceOp, out: TensorId, a: TensorId };
pub const StepReduceAxis = struct { op: types.ReduceOp, out: TensorId, a: TensorId, axis: usize };
pub const StepConcatScalar = struct { out: TensorId, axis: usize, input_count: u8, inputs: [MAX_CONCAT_INPUTS]TensorId };
pub const StepCopyTiled = struct { dst: TensorId, src: TensorId };

/// Gather rows from a 2D table using i32 indices.
///
/// Shapes:
/// - table:   [V, D] (f16/f32)
/// - indices: [B, L] (i32)
/// - out:     [B, L, D]
pub const StepGatherRowsTiled = struct { out: TensorId, table: TensorId, indices: TensorId };

/// Rotary positional embedding over 1D positions.
///
/// Shapes:
/// - x:         [B, L, N, H] (f16/f32)
/// - positions: [B, L] (i32)
/// - out:       [B, L, N, H]
pub const StepRoPE1DTiled = struct {
    out: TensorId,
    x: TensorId,
    positions: TensorId,
    base_frequency: f32,
    scale_factor: f32,
    rope_proportion: f32,
};

/// In-place KV cache append.
///
/// Shapes:
/// - cache:     [B, H_kv, T, D] (f16/f32), written in-place
/// - new_kv:    [B, H_kv, new_len, D] (f16/f32), read-only
/// - end_index: [B] (i32), per-batch append start offsets
pub const StepKVCacheAppendTiled = struct {
    cache: TensorId,
    new_kv: TensorId,
    end_index: TensorId,
};

/// Elementwise scalar-dtype cast (f16 <-> f32 in v1).
pub const StepCastTiled = struct { out: TensorId, x: TensorId, to_dtype: types.DType };

/// Matmul with B conceptually transposed: C[m,n] = sum_k A[m,k] * B[n,k].
///
/// Shapes:
/// - a: f32 with trailing axis `K` (any leading rank; C's leading axes match A's).
/// - b: q8_0 `[N, K]` with `quant_axis == 1` (per-row blocks).
/// - c: f32 `[..., N]`.
pub const StepMatMulNTTiled = struct {
    c: TensorId,
    a: TensorId,
    b: TensorId,
    alpha: f32,
    beta: f32,
};

pub const StepLSTMCellFused = struct {
    out_state: TensorId,
    x: TensorId,
    h_prev: TensorId,
    c_prev: TensorId,
    w_ih: TensorId,
    w_hh: TensorId,
    b_ih: ?TensorId,
    b_hh: ?TensorId,
};

pub const StepComplexAbsMean = struct {
    out: TensorId,
    stft: TensorId,
    /// Number of output channels to produce (must be <= (stft_last_dim/2)).
    out_channels: usize,
};

/// Pack/unpack/re-tiling materialization (scalar only, same shape).
pub const StepReTileCopyScalar = struct { dst: TensorId, src: TensorId };

/// View materializations (scalar only in v0).
pub const StepReshapeScalar = struct { dst: TensorId, src: TensorId };
pub const StepTranspose2DScalar = struct { dst: TensorId, src: TensorId };
pub const StepSliceNDScalar = struct { dst: TensorId, src: TensorId, rank: u8, starts: [MAX_RANK]usize };

pub const Step = union(enum) {
    MatMulTiled: StepMatMulTiled,
    ElemwiseBinaryTiled: StepElemwiseBinaryTiled,
    BroadcastLastDimBinaryTiled: StepBroadcastLastDimBinaryTiled,
    UnaryTiled: StepUnaryTiled,
    SoftmaxTiled: StepSoftmaxTiled,
    Conv1DTiled: StepConv1DTiled,
    Conv2DTiled: StepConv2DTiled,
    LayerNormTiled: StepLayerNormTiled,
    RMSNormTiled: StepRMSNormTiled,
    AttentionTiled: StepAttentionTiled,
    MultiHeadAttentionTiled: StepMultiHeadAttentionTiled,
    MultiHeadAttentionCachedTiled: StepMultiHeadAttentionCachedTiled,
    ReduceAll: StepReduceAll,
    ReduceAxis: StepReduceAxis,
    ConcatScalar: StepConcatScalar,
    CopyTiled: StepCopyTiled,

    GatherRowsTiled: StepGatherRowsTiled,

    RoPE1DTiled: StepRoPE1DTiled,

    KVCacheAppendTiled: StepKVCacheAppendTiled,

    LSTMCellFused: StepLSTMCellFused,

    ComplexAbsMean: StepComplexAbsMean,

    ReTileCopyScalar: StepReTileCopyScalar,

    ReshapeScalar: StepReshapeScalar,
    Transpose2DScalar: StepTranspose2DScalar,
    SliceNDScalar: StepSliceNDScalar,

    CastTiled: StepCastTiled,
    MatMulNTTiled: StepMatMulNTTiled,
};

/// Validated executable schedule.
///
/// Contract:
/// - Produced by `graph/program.compileGraph` after full graph + storage validation.
/// - Execution must assume correctness and avoid per-step argument checks.
/// - Runtime can still fail due to backend errors (e.g. Unsupported) or storage errors.
pub const ExecutableProgram = struct {
    allocator: std.mem.Allocator,
    steps: []Step,
    outputs: []TensorId,

    const Self = @This();

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.steps);
        self.allocator.free(self.outputs);
        self.* = undefined;
    }
};
