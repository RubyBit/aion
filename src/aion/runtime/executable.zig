const std = @import("std");
const types = @import("../backend/types.zig");

pub const TensorId = u32;

pub const Step = union(enum) {
    MatMulTiled: struct { c: TensorId, a: TensorId, b: TensorId, alpha: f32, beta: f32 },
    ElemwiseBinaryTiled: struct { op: types.ElemwiseBinaryOp, out: TensorId, a: TensorId, b: TensorId },
    BroadcastLastDimBinaryTiled: struct { op: types.ElemwiseBinaryOp, out: TensorId, a: TensorId, b: TensorId },
    ReluTiled: struct { out: TensorId, a: TensorId },
    ReduceAll: struct { op: types.ReduceOp, out: TensorId, a: TensorId },
    CopyTiled: struct { dst: TensorId, src: TensorId },

    /// Pack/unpack/re-tiling materialization (scalar only, same shape).
    ReTileCopyScalar: struct { dst: TensorId, src: TensorId },

    /// View materializations (scalar only in v0).
    ReshapeScalar: struct { dst: TensorId, src: TensorId },
    Transpose2DScalar: struct { dst: TensorId, src: TensorId },
    Slice2DScalar: struct { dst: TensorId, src: TensorId, start0: usize, start1: usize },
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
