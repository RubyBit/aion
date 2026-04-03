const program_mod = @import("../../graph/program.zig");
const types_mod = @import("types.zig");

/// In-place patching of `TensorId`s inside a compiled `ExecutableProgram`.
///
/// Why this exists:
/// - `graph/program.compileGraph` bakes concrete `TensorId`s into every step.
///   (External graph values are bound to a specific tensor id at compile-time.)
/// - `LoadedModel` caches compiled programs keyed mostly by *shapes* and
///   *symbol values*.
/// - If the user later binds a *different* tensor id with the same layout
///   (dtype/shape/tiling), we can avoid recompilation by rewriting those ids
///   in-place.
///
/// This is an optimization / flexibility feature. Correctness does NOT depend
/// on it if callers are willing to rebuild cache entries or copy into the
/// originally-compiled tensors.
///
/// Maintenance note:
/// This file must be kept in sync with `runtime/executable.zig` step layouts.
/// If a new `Step` variant gains `TensorId` fields, it must be handled here or
/// retargeting will silently miss it.
pub fn retargetProgramTensorIds(program: *program_mod.Program, old_tid: types_mod.TensorId, new_tid: types_mod.TensorId) void {
    if (old_tid == new_tid) return;
    for (program.steps) |*step| retargetStepTensorIds(step, old_tid, new_tid);
    for (program.outputs) |*tid| retargetTensorId(tid, old_tid, new_tid);
}

fn retargetStepTensorIds(step: *program_mod.Step, old_tid: types_mod.TensorId, new_tid: types_mod.TensorId) void {
    switch (step.*) {
        .MatMulTiled => |*s| {
            retargetTensorId(&s.c, old_tid, new_tid);
            retargetTensorId(&s.a, old_tid, new_tid);
            retargetTensorId(&s.b, old_tid, new_tid);
        },
        .ElemwiseBinaryTiled => |*s| {
            retargetTensorId(&s.out, old_tid, new_tid);
            retargetTensorId(&s.a, old_tid, new_tid);
            retargetTensorId(&s.b, old_tid, new_tid);
        },
        .BroadcastLastDimBinaryTiled => |*s| {
            retargetTensorId(&s.out, old_tid, new_tid);
            retargetTensorId(&s.a, old_tid, new_tid);
            retargetTensorId(&s.b, old_tid, new_tid);
        },
        .UnaryTiled => |*s| {
            retargetTensorId(&s.out, old_tid, new_tid);
            retargetTensorId(&s.a, old_tid, new_tid);
        },
        .SoftmaxTiled => |*s| {
            retargetTensorId(&s.out, old_tid, new_tid);
            retargetTensorId(&s.a, old_tid, new_tid);
        },
        .Conv1DTiled => |*s| {
            retargetTensorId(&s.out, old_tid, new_tid);
            retargetTensorId(&s.x, old_tid, new_tid);
            retargetTensorId(&s.w, old_tid, new_tid);
            if (s.bias) |bias| {
                var new_bias = bias;
                retargetTensorId(&new_bias, old_tid, new_tid);
                s.bias = new_bias;
            }
        },
        .Conv2DTiled => |*s| {
            retargetTensorId(&s.out, old_tid, new_tid);
            retargetTensorId(&s.x, old_tid, new_tid);
            retargetTensorId(&s.w, old_tid, new_tid);
            if (s.bias) |bias| {
                var new_bias = bias;
                retargetTensorId(&new_bias, old_tid, new_tid);
                s.bias = new_bias;
            }
        },
        .LayerNormTiled => |*s| {
            retargetTensorId(&s.out, old_tid, new_tid);
            retargetTensorId(&s.x, old_tid, new_tid);
            retargetTensorId(&s.gamma, old_tid, new_tid);
            retargetTensorId(&s.beta, old_tid, new_tid);
        },
        .RMSNormTiled => |*s| {
            retargetTensorId(&s.out, old_tid, new_tid);
            retargetTensorId(&s.x, old_tid, new_tid);
            retargetTensorId(&s.gamma, old_tid, new_tid);
            retargetTensorId(&s.beta, old_tid, new_tid);
        },
        .AttentionTiled => |*s| {
            retargetTensorId(&s.out, old_tid, new_tid);
            retargetTensorId(&s.q, old_tid, new_tid);
            retargetTensorId(&s.k, old_tid, new_tid);
            retargetTensorId(&s.v, old_tid, new_tid);
        },
        .MultiHeadAttentionTiled => |*s| {
            retargetTensorId(&s.out, old_tid, new_tid);
            retargetTensorId(&s.q, old_tid, new_tid);
            retargetTensorId(&s.k, old_tid, new_tid);
            retargetTensorId(&s.v, old_tid, new_tid);
        },
        .ReduceAll => |*s| {
            retargetTensorId(&s.out, old_tid, new_tid);
            retargetTensorId(&s.a, old_tid, new_tid);
        },
        .ReduceAxis => |*s| {
            retargetTensorId(&s.out, old_tid, new_tid);
            retargetTensorId(&s.a, old_tid, new_tid);
        },
        .ConcatScalar => |*s| {
            retargetTensorId(&s.out, old_tid, new_tid);
            for (s.inputs[0..s.input_count]) |*tid| retargetTensorId(tid, old_tid, new_tid);
        },
        .CopyTiled => |*s| {
            retargetTensorId(&s.dst, old_tid, new_tid);
            retargetTensorId(&s.src, old_tid, new_tid);
        },
        .LSTMCellFused => |*s| {
            retargetTensorId(&s.out_state, old_tid, new_tid);
            retargetTensorId(&s.x, old_tid, new_tid);
            retargetTensorId(&s.h_prev, old_tid, new_tid);
            retargetTensorId(&s.c_prev, old_tid, new_tid);
            retargetTensorId(&s.w_ih, old_tid, new_tid);
            retargetTensorId(&s.w_hh, old_tid, new_tid);
            if (s.b_ih) |b_ih| {
                var new_b_ih = b_ih;
                retargetTensorId(&new_b_ih, old_tid, new_tid);
                s.b_ih = new_b_ih;
            }
            if (s.b_hh) |b_hh| {
                var new_b_hh = b_hh;
                retargetTensorId(&new_b_hh, old_tid, new_tid);
                s.b_hh = new_b_hh;
            }
        },
        .ComplexAbsMean => |*s| {
            retargetTensorId(&s.out, old_tid, new_tid);
            retargetTensorId(&s.stft, old_tid, new_tid);
        },
        .ReTileCopyScalar => |*s| {
            retargetTensorId(&s.dst, old_tid, new_tid);
            retargetTensorId(&s.src, old_tid, new_tid);
        },
        .ReshapeScalar => |*s| {
            retargetTensorId(&s.dst, old_tid, new_tid);
            retargetTensorId(&s.src, old_tid, new_tid);
        },
        .Transpose2DScalar => |*s| {
            retargetTensorId(&s.dst, old_tid, new_tid);
            retargetTensorId(&s.src, old_tid, new_tid);
        },
        .SliceNDScalar => |*s| {
            retargetTensorId(&s.dst, old_tid, new_tid);
            retargetTensorId(&s.src, old_tid, new_tid);
        },
    }
}

fn retargetTensorId(slot: *types_mod.TensorId, old_tid: types_mod.TensorId, new_tid: types_mod.TensorId) void {
    if (slot.* == old_tid) slot.* = new_tid;
}
