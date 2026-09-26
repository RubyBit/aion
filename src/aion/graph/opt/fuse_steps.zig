// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Fuse lowered add-norm and activation-multiply patterns using resolved layouts to
//! remove dispatches from serial residual paths.

const std = @import("std");

const executable = @import("../../runtime/executable.zig");
const manager_mod = @import("../../storage/manager.zig");
const editor_mod = @import("editor.zig");

const Editor = editor_mod.Editor;
const PlacedStep = executable.PlacedStep;
const StorageManager = manager_mod.StorageManager;
const TensorId = manager_mod.TensorId;
const Tensor = manager_mod.Tensor;

/// Fold a single-use `RMSNorm` into its identical-shape add by configuring the
/// norm's residual and moving it to the add's position.
pub fn addNorm(ed: *Editor) void {
    for (ed.steps, 0..) |*step, i| {
        const norm = switch (step.op) {
            .RMSNorm => |s| s,
            else => continue,
        };
        // The fused step is itself a `RMSNorm`, so without this it could re-match
        // and overwrite the residual it already carries, silently dropping one add.
        if (norm.residual != null) continue;
        const j = ed.soleReader(norm.out, i) orelse continue;
        const add = switch (ed.steps[j].op) {
            .ElemwiseBinary => |s| s,
            else => continue,
        };
        if (add.op != .add or add.broadcast.kind != .identical) continue;
        const residual: TensorId = if (add.a == norm.out) add.b else add.a;
        // A norm added to itself has nothing to fold, and the fused kernel would need
        // the residual and the normalised value to be two bindings.
        if (residual == norm.out) continue;
        if (!movable(ed, ed.steps[i], ed.steps[j], &.{ norm.x, norm.gamma, norm.beta }, add.out, i)) continue;
        if (!fusableAddNorm(ed.mgr, norm, add, residual)) continue;

        // The norm lands where the ADD was, not the reverse: then every consumer
        // downstream still reads the value at the step it read it at before, and only
        // the norm's work moves forward, which `movable` has established is safe.
        ed.steps[j].op = .{ .RMSNorm = .{
            .out = add.out,
            .x = norm.x,
            .gamma = norm.gamma,
            .beta = norm.beta,
            .eps = norm.eps,
            .residual = residual,
        } };
        ed.kill(i);
    }
}

/// Fold a single-use activation into an identical-shape multiply as a gate step.
/// Unsupported dtypes or broadcasting retain the original pair.
pub fn gate(ed: *Editor) void {
    for (ed.steps, 0..) |*step, i| {
        const un = switch (step.op) {
            .Unary => |s| s,
            else => continue,
        };
        const j = ed.soleReader(un.out, i) orelse continue;
        const mul = switch (ed.steps[j].op) {
            .ElemwiseBinary => |s| s,
            else => continue,
        };
        if (mul.op != .mul or mul.broadcast.kind != .identical) continue;
        const other: TensorId = if (mul.a == un.out) mul.b else mul.a;
        // `mul(u, u)` squares the activation; there is no second operand to gate with.
        if (other == un.out) continue;
        if (!movable(ed, ed.steps[i], ed.steps[j], &.{un.a}, mul.out, i)) continue;
        if (!fusableGate(ed.mgr, un, mul, other)) continue;

        // `gate` applies the activation to operand `a`, so the gated side lands there
        // whichever side of the original multiply it was on.
        ed.steps[j].op = .{ .ElemwiseBinary = .{
            .op = .gate,
            .act = un.op,
            .out = mul.out,
            .a = un.a,
            .b = other,
            .broadcast = mul.broadcast,
        } };
        ed.kill(i);
    }
}

/// The hazards of moving a producer forward onto its consumer, shared by both rules.
/// Every condition here is one that would otherwise be silent.
fn movable(
    ed: *const Editor,
    producer: PlacedStep,
    consumer: PlacedStep,
    reads: []const TensorId,
    out: TensorId,
    producer_index: usize,
) bool {
    // One dispatch, one target. `host_operands` is indexed by operand position and the
    // fused step's operand list differs from either input, so a step reading any operand
    // on the host cannot inherit the mask — refuse instead of remapping.
    if (!producer.placement.eql(consumer.placement)) return false;
    if (producer.host_operands != 0 or consumer.host_operands != 0) return false;

    for (reads) |id| {
        // The fused kernel writes `out` while reading these; a shared id would be a
        // read-write aliasing of one binding, which wgpu rejects outright.
        if (id == out) return false;
        // And the read moves to the consumer's position, so a write in between would be
        // observed where it previously was not — which only has a meaning for a tensor
        // confined to the list being edited.
        if (!ed.confined(id)) return false;
        const u = ed.use(id) orelse return false;
        if (u.writes > 0 and u.last_write > producer_index) return false;
    }
    return true;
}

/// The residual clauses of `compiler.validateStep`'s `.RMSNorm` arm, which the
/// reconfigured step never re-enters: it is written into an already-validated program.
/// Only the new operand needs checking, which is the point of making this a field.
fn fusableAddNorm(
    mgr: *StorageManager,
    norm: executable.StepRMSNorm,
    add: executable.StepElemwiseBinary,
    residual: TensorId,
) bool {
    const out: *const Tensor = mgr.getConst(add.out) catch return false;
    const r: *const Tensor = mgr.getConst(residual) catch return false;
    const gamma: *const Tensor = mgr.getConst(norm.gamma) catch return false;

    if (out.dtype != .f32 or r.dtype != .f32) return false;
    if (r.rank != out.rank or !std.mem.eql(usize, out.shape, r.shape)) return false;

    // The fused kernel is one workgroup per row, so a narrower `normalized_shape`
    // must stay unfused.
    if (gamma.rank != 1) return false;
    return gamma.shape[0] == out.shape[@as(usize, out.rank) - 1];
}

/// Mirrors the `.gate` clauses of `compiler.validateStep`'s `.ElemwiseBinary` arm:
/// f32, identical shape on all three, no broadcast.
fn fusableGate(
    mgr: *StorageManager,
    un: executable.StepUnary,
    mul: executable.StepElemwiseBinary,
    other: TensorId,
) bool {
    const out: *const Tensor = mgr.getConst(mul.out) catch return false;
    const a: *const Tensor = mgr.getConst(un.a) catch return false;
    const b: *const Tensor = mgr.getConst(other) catch return false;

    if (out.dtype != .f32 or a.dtype != .f32 or b.dtype != .f32) return false;
    if (a.rank != out.rank or b.rank != out.rank) return false;
    if (!std.mem.eql(usize, out.shape, a.shape)) return false;
    return std.mem.eql(usize, out.shape, b.shape);
}
