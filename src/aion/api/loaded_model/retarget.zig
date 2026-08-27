// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const program_mod = @import("../../graph/program.zig");
const executable = @import("../../runtime/executable.zig");
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
/// `executable.tensorUses` is the singular operand walk used here and by
/// placement/liveness, so new step variants cannot silently diverge.
pub fn retargetProgramTensorIds(program: *program_mod.Program, old_tid: types_mod.TensorId, new_tid: types_mod.TensorId) void {
    if (old_tid == new_tid) return;
    for (program.steps) |*step| retargetStepTensorIds(&step.op, old_tid, new_tid);
    for (program.blocks) |*block| {
        for (block.steps) |*step| retargetStepTensorIds(&step.op, old_tid, new_tid);
    }
    for (program.outputs) |*tid| retargetTensorId(tid, old_tid, new_tid);
    for (program.growth_requests) |*request| {
        retargetTensorId(&request.cache, old_tid, new_tid);
        retargetTensorId(&request.new_kv, old_tid, new_tid);
        retargetTensorId(&request.end_index, old_tid, new_tid);
    }
    for (program.tensor_placements) |*placement| {
        retargetTensorId(&placement.id, old_tid, new_tid);
    }
    std.mem.sort(executable.TensorPlacement, program.tensor_placements, {}, struct {
        fn lessThan(_: void, a: executable.TensorPlacement, b: executable.TensorPlacement) bool {
            return a.id < b.id;
        }
    }.lessThan);
}

fn retargetStepTensorIds(step: *program_mod.Step, old_tid: types_mod.TensorId, new_tid: types_mod.TensorId) void {
    const uses = executable.tensorUses(step);
    for (uses.slice()) |use| retargetTensorId(use.id, old_tid, new_tid);
}

fn retargetTensorId(slot: *types_mod.TensorId, old_tid: types_mod.TensorId, new_tid: types_mod.TensorId) void {
    if (slot.* == old_tid) slot.* = new_tid;
}
