// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const types = @import("../backend/types.zig");

pub const TensorId = u32;
pub const BlockId = u32;
const MAX_RANK: usize = 8;
const MAX_CONCAT_INPUTS: usize = 16;
pub const MAX_CONTROL_OUTPUTS: usize = 16;
pub const MAX_LOOP_CARRIED: usize = 16;

pub const StepMatMulTiled = struct { c: TensorId, a: TensorId, b: TensorId, alpha: f32, beta: f32 };
pub const StepElemwiseBinaryTiled = struct { op: types.ElemwiseBinaryOp, out: TensorId, a: TensorId, b: TensorId };
pub const StepGeluMulTiled = struct { out: TensorId, a: TensorId, b: TensorId };
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
/// Relative-positional multi-head self-attention (Transformer-XL / Conformer).
///
/// Shapes:
/// - q, k, v, out: [B, H, T*, D] (q rows T_q; k/v rows T_kv)
/// - pos_emb:      [H, P, D] (P = 2*T_kv - 1)
/// - pos_bias_u/_v:[H, D]
/// - mask (opt):   [T_q, T_kv] additive
pub const StepRelPosMHATiled = struct {
    out: TensorId,
    q: TensorId,
    k: TensorId,
    v: TensorId,
    pos_emb: TensorId,
    pos_bias_u: TensorId,
    pos_bias_v: TensorId,
    mask: ?TensorId,
    scale: f32,
    heads: usize,
};
pub const StepReduceAll = struct { op: types.ReduceOp, out: TensorId, a: TensorId };
pub const StepReduceAxis = struct { op: types.ReduceOp, out: TensorId, a: TensorId, axis: usize };
/// ArgMax over the last axis (v1): out (i32) = index of max of a along axis.
pub const StepArgMax = struct { out: TensorId, a: TensorId, axis: usize };
/// In-place row scatter: buf[idx] = src. Output aliases buf (set in lowering).
pub const StepScatterRow = struct { buf: TensorId, idx: TensorId, src: TensorId };
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
pub const StepSequenceAppendTiled = struct {
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

/// Branch control flow over a scalar i32 predicate (`0=false`, non-zero=true`).
///
/// Both branch blocks are expected to materialize `output_count` tensors with
/// identical dtype/shape/layout. After the selected block runs, selected branch
/// result buffers are swapped into `outputs` when the store supports no-copy
/// swapping. This preserves SSA-style branch-local outputs without copying data.
pub const StepIf = struct {
    cond: TensorId,
    then_block: BlockId,
    else_block: BlockId,
    output_count: u8,
    outputs: [MAX_CONTROL_OUTPUTS]TensorId,
    then_outputs: [MAX_CONTROL_OUTPUTS]TensorId,
    else_outputs: [MAX_CONTROL_OUTPUTS]TensorId,
};

/// Loop control flow with fixed storage and shape-invariant carried tensors.
///
/// The body writes next-state tensors into `body_carried_outputs`. After each
/// body iteration, those tensors are swapped with `carried`, so the next
/// iteration sees the new state without copying the tensor contents. Swapping
/// the buffers also leaves the body output tensors available as scratch storage
/// for the next iteration.
pub const StepLoop = struct {
    trip_count: ?TensorId,
    static_max_trip_count: usize,
    cond: ?TensorId,
    check_before: bool,
    body_block: BlockId,
    carried_count: u8,
    carried: [MAX_LOOP_CARRIED]TensorId,
    body_carried_outputs: [MAX_LOOP_CARRIED]TensorId,
};

pub const StepRFFT = struct {
    out: TensorId,
    x: TensorId,
    /// Real FFT length (power of two); equals the input's last dimension.
    n_fft: usize,
};

pub const StepSTFT = struct {
    out: TensorId,
    signal: TensorId,
    window: TensorId,
    n_fft: usize,
    hop_length: usize,
    center: bool,
    num_frames: usize,
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
    GeluMulTiled: StepGeluMulTiled,
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
    RelPosMHATiled: StepRelPosMHATiled,
    ArgMax: StepArgMax,
    ScatterRow: StepScatterRow,
    ReduceAll: StepReduceAll,
    ReduceAxis: StepReduceAxis,
    ConcatScalar: StepConcatScalar,
    CopyTiled: StepCopyTiled,

    GatherRowsTiled: StepGatherRowsTiled,

    RoPE1DTiled: StepRoPE1DTiled,

    SequenceAppendTiled: StepSequenceAppendTiled,

    LSTMCellFused: StepLSTMCellFused,

    If: StepIf,
    Loop: StepLoop,

    RFFT: StepRFFT,
    STFT: StepSTFT,

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
pub const Block = struct {
    steps: []Step,
};

pub const ExecutableProgram = struct {
    allocator: std.mem.Allocator,
    steps: []Step,
    blocks: []Block = &[_]Block{},
    outputs: []TensorId,
    /// Per-output flag, parallel to `outputs`: true when the output is recurrent
    /// state aliased *in place* to a program input (KV caches, LSTM h/c — the
    /// output's tensor id equals the input slot it feeds back into). A device
    /// backend keeps such an output resident on-device across `execute` calls
    /// instead of flushing it to the host: the model's io-alias write-back is a
    /// no-op for it (source == destination), so the host never reads it between
    /// runs and a device->host flush would be pure, context-length-scaling waste.
    /// Empty (the default) means "flush every output" — programs compiled without
    /// alias annotation, and the CPU backend which has no residency, are unchanged.
    output_device_resident: []const bool = &[_]bool{},

    const Self = @This();

    /// Whether a device backend should keep output `i` resident on-device after
    /// `execute` instead of flushing it to the host. See `output_device_resident`.
    pub fn outputStaysResident(self: *const Self, i: usize) bool {
        return i < self.output_device_resident.len and self.output_device_resident[i];
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.steps);
        for (self.blocks) |block| {
            if (block.steps.len != 0) self.allocator.free(block.steps);
        }
        if (self.blocks.len != 0) self.allocator.free(self.blocks);
        self.allocator.free(self.outputs);
        if (self.output_device_resident.len != 0) self.allocator.free(self.output_device_resident);
        self.* = undefined;
    }
};
