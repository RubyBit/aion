// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
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
/// This implementation walks step payload fields recursively and patches any
/// `TensorId` / `?TensorId` fields it finds. That keeps it resilient to new
/// step variants and new tensor-id fields without per-variant boilerplate.
pub fn retargetProgramTensorIds(program: *program_mod.Program, old_tid: types_mod.TensorId, new_tid: types_mod.TensorId) void {
    if (old_tid == new_tid) return;
    for (program.steps) |*step| retargetStepTensorIds(step, old_tid, new_tid);
    for (program.blocks) |*block| {
        for (block.steps) |*step| retargetStepTensorIds(step, old_tid, new_tid);
    }
    for (program.outputs) |*tid| retargetTensorId(tid, old_tid, new_tid);
}

fn retargetStepTensorIds(step: *program_mod.Step, old_tid: types_mod.TensorId, new_tid: types_mod.TensorId) void {
    switch (step.*) {
        inline else => |*payload| {
            retargetTensorIdsInValue(@TypeOf(payload.*), payload, old_tid, new_tid);
        },
    }
}

fn retargetTensorIdsInValue(comptime T: type, value: *T, old_tid: types_mod.TensorId, new_tid: types_mod.TensorId) void {
    switch (@typeInfo(T)) {
        .int => {
            if (T == types_mod.TensorId) {
                retargetTensorId(@ptrCast(value), old_tid, new_tid);
            }
        },
        .optional => |opt| {
            if (opt.child == types_mod.TensorId) {
                if (value.*) |tid| {
                    if (tid == old_tid) value.* = new_tid;
                }
                return;
            }

            if (value.*) |*child| {
                retargetTensorIdsInValue(opt.child, child, old_tid, new_tid);
            }
        },
        .array => |arr| {
            var i: usize = 0;
            while (i < arr.len) : (i += 1) {
                retargetTensorIdsInValue(arr.child, &value.*[i], old_tid, new_tid);
            }
        },
        .@"struct" => |s| {
            inline for (s.fields) |field| {
                retargetTensorIdsInValue(field.type, &@field(value.*, field.name), old_tid, new_tid);
            }
        },
        .@"union" => {
            switch (value.*) {
                inline else => |*payload| {
                    retargetTensorIdsInValue(@TypeOf(payload.*), payload, old_tid, new_tid);
                },
            }
        },
        else => {},
    }
}

fn retargetTensorId(slot: *types_mod.TensorId, old_tid: types_mod.TensorId, new_tid: types_mod.TensorId) void {
    if (slot.* == old_tid) slot.* = new_tid;
}

/// Read-only counterpart: does any step/output of `program` reference `tid`?
/// Used to confirm a fused-away weight is safe to reclaim (nothing still reads it).
pub fn programReferencesTensorId(program: *const program_mod.Program, tid: types_mod.TensorId) bool {
    for (program.steps) |*step| if (stepReferencesTensorId(step, tid)) return true;
    for (program.blocks) |*block| {
        for (block.steps) |*step| if (stepReferencesTensorId(step, tid)) return true;
    }
    for (program.outputs) |out| if (out == tid) return true;
    return false;
}

fn stepReferencesTensorId(step: *const program_mod.Step, tid: types_mod.TensorId) bool {
    switch (step.*) {
        inline else => |*payload| return valueReferencesTensorId(@TypeOf(payload.*), payload, tid),
    }
}

fn valueReferencesTensorId(comptime T: type, value: *const T, tid: types_mod.TensorId) bool {
    switch (@typeInfo(T)) {
        .int => return (T == types_mod.TensorId and value.* == tid),
        .optional => |opt| {
            if (value.*) |*child| return valueReferencesTensorId(opt.child, child, tid);
            return false;
        },
        .array => |arr| {
            var i: usize = 0;
            while (i < arr.len) : (i += 1) {
                if (valueReferencesTensorId(arr.child, &value.*[i], tid)) return true;
            }
            return false;
        },
        .@"struct" => |s| {
            inline for (s.fields) |field| {
                if (valueReferencesTensorId(field.type, &@field(value.*, field.name), tid)) return true;
            }
            return false;
        },
        .@"union" => switch (value.*) {
            inline else => |*payload| return valueReferencesTensorId(@TypeOf(payload.*), payload, tid),
        },
        else => return false,
    }
}
