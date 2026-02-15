const std = @import("std");
const types = @import("../backend/types.zig");

pub const TensorId = u32;

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
    groups: usize,
};
pub const StepLayerNormTiled = struct { out: TensorId, x: TensorId, gamma: TensorId, beta: TensorId, eps: f32 };
pub const StepRMSNormTiled = struct { out: TensorId, x: TensorId, gamma: TensorId, beta: TensorId, eps: f32 };
pub const StepAttentionTiled = struct { out: TensorId, q: TensorId, k: TensorId, v: TensorId, scale: f32, causal: bool };
pub const StepMultiHeadAttentionTiled = struct { out: TensorId, q: TensorId, k: TensorId, v: TensorId, scale: f32, causal: bool, heads: usize };
pub const StepReduceAll = struct { op: types.ReduceOp, out: TensorId, a: TensorId };
pub const StepCopyTiled = struct { dst: TensorId, src: TensorId };

/// Pack/unpack/re-tiling materialization (scalar only, same shape).
pub const StepReTileCopyScalar = struct { dst: TensorId, src: TensorId };

/// View materializations (scalar only in v0).
pub const StepReshapeScalar = struct { dst: TensorId, src: TensorId };
pub const StepTranspose2DScalar = struct { dst: TensorId, src: TensorId };
pub const StepSlice2DScalar = struct { dst: TensorId, src: TensorId, start0: usize, start1: usize };

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
    ReduceAll: StepReduceAll,
    CopyTiled: StepCopyTiled,

    ReTileCopyScalar: StepReTileCopyScalar,

    ReshapeScalar: StepReshapeScalar,
    Transpose2DScalar: StepTranspose2DScalar,
    Slice2DScalar: StepSlice2DScalar,
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
