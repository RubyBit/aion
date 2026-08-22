// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Step-level peephole fusion: merge adjacent-in-dataflow steps into one dispatch.
//!
//! This is the second inhabitant of the stage `alias_views` opened — a pass over the
//! LOWERED schedule, after placement and before workspace planning. Which IR a fusion
//! belongs to is the whole design question, and the rule is:
//!
//!   * A fusion that changes what the model MEANS belongs in the GRAPH, is authored
//!     directly, and is serialized. It should also be expressible by widening an
//!     existing op — a gated activation is `ElemwiseBinary{ .op = .gate, .act }`, not an
//!     op tag of its own.
//!   * A fusion that leaves the meaning alone and changes only HOW it is scheduled
//!     belongs HERE, and must never reach the graph or the file — otherwise a `.aion`
//!     stops being target-neutral and one file could no longer compile to a different
//!     schedule per backend.
//!
//! Both fusions here are the second kind, and both recover a schedule rather than a
//! concept. `x = x + norm(f(x))` is nobody's named primitive — the converter writes an
//! add. A gate IS a named primitive, so the authoring API emits `gate` directly and
//! `fuseGate` exists only as a safety net for graphs that spell it out as a unary and a
//! multiply; when it fires, no meaning is added, because the two forms are already
//! defined to be the same value.
//!
//! Working at step level also makes a match cheaper to prove correct than the graph
//! equivalents were: tiling is already resolved, so the fused step invariants are checked
//! against real tensors instead of predicted from shapes.
//!
//! Why fuse at all, given a dispatch costs only ~1.5 us: these sit on the serial residual
//! chain, where a step costs 4-9 us of critical path however little it computes. Removing
//! 106 of them measured 0.13 ms/token on Gemma-4 E2B. That is the whole of the win — the
//! arithmetic does not go away, so only pairs where launch dominates are worth a fused
//! kernel.

const std = @import("std");
const executable = @import("../../runtime/executable.zig");
const manager_mod = @import("../../storage/manager.zig");
const plan_mod = @import("../plan.zig");
const step_uses = @import("step_uses.zig");
const placement = @import("placement.zig");

const StorageManager = manager_mod.StorageManager;
const TiledTensor = manager_mod.TiledTensor;
const TensorId = manager_mod.TensorId;
const Program = executable.ExecutableProgram;
const PlacedStep = executable.PlacedStep;

pub const Error = error{ InvalidArgument, OutOfMemory };

/// What a peephole sees and may change: it rewrites consumer steps in place and marks
/// producers in `drop`, and returns how many it marked. Compaction, placement repair and
/// use-collection are the caller's job, so a pass is only its match rule.
const Peephole = fn (
    mgr: *StorageManager,
    prog: *Program,
    uses: *const step_uses.Map,
    is_owned: *const step_uses.OwnedSet,
    drop: []bool,
) usize;

/// Fuse what the target profits from. Returns the number of steps removed, so a caller or
/// a test can assert a pass fired rather than infer it from a timing.
///
/// CPU is deliberately excluded. The unfused forms are the numerical oracle the fused GPU
/// kernels are tested against (`test_gpu_backend.zig` asserts they agree to 1e-4 with the
/// CPU running the unfused version), and the CPU is not launch-bound anyway: its decode
/// profile is 92 % MatMul and 0.19 % elementwise, so there is nothing here to win.
pub fn run(
    allocator: std.mem.Allocator,
    mgr: *StorageManager,
    prog: *Program,
    owned: []const TensorId,
    policy: plan_mod.TilePolicy,
) Error!usize {
    if (policy.target_kind == .cpu) return 0;
    if (prog.steps.len == 0 or owned.len == 0) return 0;

    // Whether each fusion fires is asserted in `graph/test_compile.zig` (one fused step
    // on a device target, an unfused pair on CPU, for every activation), so neither needs
    // a runtime switch to be observable.
    var total: usize = 0;
    total += try apply(allocator, mgr, prog, owned, fuseAddNorm);
    total += try apply(allocator, mgr, prog, owned, fuseGate);

    // Every fused pair orphans the intermediate the producer used to write. Workspace
    // planning will release it, so its stale placement entry has to go first.
    if (total != 0) try placement.pruneOrphaned(allocator, prog);
    return total;
}

/// Collect facts, run one peephole, compact. Each pass gets a freshly collected use map,
/// so no pass has to reason about which facts a previous rewrite moved.
fn apply(
    allocator: std.mem.Allocator,
    mgr: *StorageManager,
    prog: *Program,
    owned: []const TensorId,
    comptime pass: Peephole,
) Error!usize {
    var uses = try step_uses.collect(allocator, prog);
    defer uses.deinit();
    var is_owned = try step_uses.ownedSet(allocator, owned);
    defer is_owned.deinit();

    const drop = allocator.alloc(bool, prog.steps.len) catch return error.OutOfMemory;
    defer allocator.free(drop);
    @memset(drop, false);

    const dropped = pass(mgr, prog, &uses, &is_owned, drop);
    if (dropped == 0) return 0;

    const kept = allocator.alloc(PlacedStep, prog.steps.len - dropped) catch return error.OutOfMemory;
    var w: usize = 0;
    for (prog.steps, 0..) |step, i| {
        if (drop[i]) continue;
        kept[w] = step;
        w += 1;
    }
    allocator.free(prog.steps);
    prog.steps = kept;
    return dropped;
}

/// `RMSNormTiled` whose only reader is an identical-shape `add` ⇒ the norm gains
/// `.residual` and moves to where the add was; the add is dropped.
///
/// The residual is a CONFIGURATION of the norm, not a second norm op, so this sets one
/// field on a step the compiler already validated. That is the whole reason `fusable`
/// below is four lines: every other invariant was checked when the norm was appended.
///
/// This is the sandwich-norm residual `x = x + norm(f(x))`, 105 per Gemma-4 token. f32
/// addition commutes exactly, so folding the add into the apply loop of the norm is
/// bit-identical whichever operand was the residual — the CPU executor keeps the
/// decomposed form precisely so a test can assert that.
fn fuseAddNorm(
    mgr: *StorageManager,
    prog: *Program,
    uses: *const step_uses.Map,
    is_owned: *const step_uses.OwnedSet,
    drop: []bool,
) usize {
    var dropped: usize = 0;
    for (prog.steps, 0..) |*step, i| {
        const norm = switch (step.op) {
            .RMSNormTiled => |s| s,
            else => continue,
        };
        // The fused step is itself a `RMSNormTiled`, so without this it could re-match on
        // a later pass over the same list and overwrite the residual it already carries,
        // silently dropping one add. A tag of its own could not re-match; a configuration
        // can, and this is the price of that.
        if (norm.residual != null) continue;
        const j = soleReader(prog, uses, is_owned, norm.out, i) orelse continue;
        const add = switch (prog.steps[j].op) {
            .ElemwiseBinaryTiled => |s| s,
            else => continue,
        };
        if (add.op != .add or add.broadcast.kind != .identical) continue;
        const residual: TensorId = if (add.a == norm.out) add.b else add.a;
        // A norm added to itself has nothing to fold, and the fused kernel would need the
        // residual and the normalised value to be two bindings.
        if (residual == norm.out) continue;
        if (!movable(uses, prog.steps[i], prog.steps[j], &.{ norm.x, norm.gamma, norm.beta }, add.out, i)) continue;
        if (!fusableAddNorm(mgr, norm, add, residual)) continue;

        // The norm lands at the position of the ADD, not the other way round: then every
        // consumer downstream still reads the value at the step it read it at before, and
        // only the work of the norm moves forward, which `movable` has established is
        // safe. Configuring the norm in place instead would move the residual READ
        // backwards, which is a different proof and a longer live range for `out`.
        prog.steps[j].op = .{ .RMSNormTiled = .{
            .out = add.out,
            .x = norm.x,
            .gamma = norm.gamma,
            .beta = norm.beta,
            .eps = norm.eps,
            .residual = residual,
        } };
        drop[i] = true;
        dropped += 1;
    }
    return dropped;
}

/// `UnaryTiled(act)` whose only reader is an identical-shape `mul` ⇒ the mul becomes
/// `ElemwiseBinaryTiled{ .op = .gate, .act }` and the unary is dropped.
///
/// A gated FFN block authored through `nn.GatedMLP` already emits one `gate` node, so
/// this fires only for graphs that spell the gate out — hand-built ones, and any converter
/// that writes `mul(unary(g), u)` directly. It generalizes over the activation because
/// `gate` does: GEGLU, SwiGLU, GLU and ReGLU are one op with one parameter.
fn fuseGate(
    mgr: *StorageManager,
    prog: *Program,
    uses: *const step_uses.Map,
    is_owned: *const step_uses.OwnedSet,
    drop: []bool,
) usize {
    var dropped: usize = 0;
    for (prog.steps, 0..) |*step, i| {
        const un = switch (step.op) {
            .UnaryTiled => |s| s,
            else => continue,
        };
        const j = soleReader(prog, uses, is_owned, un.out, i) orelse continue;
        const mul = switch (prog.steps[j].op) {
            .ElemwiseBinaryTiled => |s| s,
            else => continue,
        };
        if (mul.op != .mul or mul.broadcast.kind != .identical) continue;
        const other: TensorId = if (mul.a == un.out) mul.b else mul.a;
        // `mul(u, u)` squares the activation; there is no second operand to gate with.
        if (other == un.out) continue;
        if (!movable(uses, prog.steps[i], prog.steps[j], &.{un.a}, mul.out, i)) continue;
        if (!fusableGate(mgr, un, mul, other)) continue;

        // `gate` applies the activation to operand `a`, so the gated side has to land
        // there whichever side of the original multiply it was on.
        prog.steps[j].op = .{ .ElemwiseBinaryTiled = .{
            .op = .gate,
            .act = un.op,
            .out = mul.out,
            .a = un.a,
            .b = other,
            .broadcast = mul.broadcast,
        } };
        drop[i] = true;
        dropped += 1;
    }
    return dropped;
}

/// The one step after `producer` that reads `v`, when `v` is private workspace written
/// only by `producer` and read exactly once. Anything else — a second reader, a graph
/// output, a value a control-flow body touches — means the producer has to stay.
fn soleReader(
    prog: *const Program,
    uses: *const step_uses.Map,
    is_owned: *const step_uses.OwnedSet,
    v: TensorId,
    producer: usize,
) ?usize {
    if (!is_owned.contains(v)) return null;
    for (prog.outputs) |id| if (id == v) return null;
    const u = uses.get(v) orelse return null;
    if (u.in_block) return null;
    if (u.writes != 1 or u.last_write != producer) return null;
    if (u.reads != 1 or u.first_touch != producer) return null;

    var j = producer + 1;
    while (j < prog.steps.len) : (j += 1) {
        const walk = executable.tensorUses(&prog.steps[j].op);
        for (walk.slice()) |use| if (use.id.* == v) return j;
    }
    return null;
}

/// The hazards of moving a producer forward onto its consumer, shared by both fusions.
/// Every condition here is one that would otherwise be silent.
fn movable(
    uses: *const step_uses.Map,
    producer: PlacedStep,
    consumer: PlacedStep,
    reads: []const TensorId,
    out: TensorId,
    producer_index: usize,
) bool {
    // One dispatch, one target. `host_operands` is indexed by operand position and the
    // operand list of the fused step differs from either input, so a step that reads any
    // operand on the host cannot simply inherit the mask — refuse instead of remapping.
    if (!producer.placement.eql(consumer.placement)) return false;
    if (producer.host_operands != 0 or consumer.host_operands != 0) return false;

    for (reads) |id| {
        // The fused kernel writes `out` while reading these. A shared id would be a
        // read-write aliasing of one binding, which wgpu rejects outright.
        if (id == out) return false;
        // And the read moves from the position of the producer to that of the consumer,
        // so a write to it in between would be observed where it previously was not.
        const u = uses.get(id) orelse return false;
        if (u.in_block) return false;
        if (u.writes > 0 and u.last_write > producer_index) return false;
    }
    return true;
}

/// The residual clauses of `compiler.validateStep`'s `.RMSNormTiled` arm, which the
/// reconfigured step never re-enters: it is written into an already-validated program
/// rather than appended. Everything the norm alone had to satisfy still holds untouched —
/// only the new operand needs checking, which is the point of making this a field.
fn fusableAddNorm(
    mgr: *StorageManager,
    norm: executable.StepRMSNormTiled,
    add: executable.StepElemwiseBinaryTiled,
    residual: TensorId,
) bool {
    const out: *const TiledTensor = mgr.getConst(add.out) catch return false;
    const r: *const TiledTensor = mgr.getConst(residual) catch return false;
    const gamma: *const TiledTensor = mgr.getConst(norm.gamma) catch return false;

    if (out.dtype != .f32 or r.dtype != .f32) return false;
    if (r.rank != out.rank or !std.mem.eql(usize, out.shape, r.shape)) return false;
    if (!sameTiling(out, r)) return false;

    // The fused kernel is one workgroup per row and has no cross-tile path, so a
    // narrower `normalized_shape` or a split row must stay unfused.
    if (gamma.rank != 1) return false;
    const k: usize = out.shape[@as(usize, out.rank) - 1];
    if (gamma.tile_shape[0] != k or out.tile_shape[@as(usize, out.rank) - 1] != k) return false;
    return true;
}

/// Mirrors the `.gate` clauses of `compiler.validateStep`'s `.ElemwiseBinaryTiled` arm:
/// f32, identical shape and tiling on all three, no broadcast.
fn fusableGate(
    mgr: *StorageManager,
    un: executable.StepUnaryTiled,
    mul: executable.StepElemwiseBinaryTiled,
    other: TensorId,
) bool {
    const out: *const TiledTensor = mgr.getConst(mul.out) catch return false;
    const a: *const TiledTensor = mgr.getConst(un.a) catch return false;
    const b: *const TiledTensor = mgr.getConst(other) catch return false;

    if (out.dtype != .f32 or a.dtype != .f32 or b.dtype != .f32) return false;
    if (a.rank != out.rank or b.rank != out.rank) return false;
    if (!std.mem.eql(usize, out.shape, a.shape)) return false;
    if (!std.mem.eql(usize, out.shape, b.shape)) return false;
    if (!sameTiling(out, a) or !sameTiling(out, b)) return false;
    return true;
}

fn sameTiling(a: *const TiledTensor, b: *const TiledTensor) bool {
    return std.mem.eql(usize, a.tile_shape, b.tile_shape) and
        std.mem.eql(usize, a.tile_counts, b.tile_counts);
}
