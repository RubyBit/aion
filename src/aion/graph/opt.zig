// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Optional rewrite passes, at both IR levels.
//!
//! `graphPasses` rewrites the Graph before lowering; `stepPasses` rewrites the
//! lowered schedule after placement. Everything here is optional by construction:
//! dropping a pass from the `Policy` must only cost speed, never change results.
//! Mandatory lowering stages live in `program/`.
//!
//! Which level a rewrite belongs to: a rewrite that changes what the model MEANS
//! belongs in the graph and is serialized; one that changes only how it is
//! scheduled belongs at step level and must never reach a `.aion`, or one file
//! could no longer compile to a different schedule per backend.

const std = @import("std");

const graph_mod = @import("graph.zig");
const plan = @import("plan.zig");
const manager_mod = @import("../storage/manager.zig");
const executable = @import("../runtime/executable.zig");
const infer_mod = @import("infer.zig");
const placement = @import("program/placement.zig");
const target_mod = @import("target.zig");

const alias_views = @import("opt/alias_views.zig");
const editor_mod = @import("opt/editor.zig");
const fuse_steps = @import("opt/fuse_steps.zig");
const horizontal_matmul = @import("opt/horizontal_matmul.zig");
const pointwise_conv = @import("opt/pointwise_conv.zig");
const rewriter_mod = @import("opt/rewriter.zig");
const step_uses = @import("opt/step_uses.zig");

const Graph = graph_mod.Graph;
const Program = executable.ExecutableProgram;
const StorageManager = manager_mod.StorageManager;
const TensorId = manager_mod.TensorId;
const Editor = editor_mod.Editor;

pub const Error = graph_mod.GraphError || manager_mod.StorageError || infer_mod.InferError;
pub const AliasMap = alias_views.AliasMap;
/// Exposed so a diagnostic (the bench's no-op view count) asks the pass instead of
/// keeping its own copy of the rule.
pub const layoutsIdentical = alias_views.layoutsIdentical;

pub const Pass = enum {
    /// 1x1 Conv1D -> MatMul, so it takes the autotuned GEMM path.
    pointwise_conv,
    /// Parallel projections off one input -> one wide MatMul + slices.
    horizontal_matmul,
    /// `x + norm(y)` -> one norm step carrying the residual.
    add_norm,
    /// `act(a) * b` -> one gated elementwise step.
    gate,
    /// Drop view steps that copy between byte-identical layouts.
    alias_views,
};

pub const Policy = std.EnumSet(Pass);

/// The passes a target profits from.
///
/// The three fusions are off on CPU. `add_norm`/`gate` because their unfused forms
/// are the numerical oracle the fused GPU kernels are tested against, and the CPU
/// is not launch-bound anyway. `horizontal_matmul` because it was measured: on GPU
/// decode it is worth ~2%/token, but on CPU it only adds slice copies (~1-2% loss
/// at decode, 1.1% at seq 64) since there is no dispatch cost to save.
pub fn defaults(tiles: plan.TilePolicy) Policy {
    var p: Policy = .full;
    if (tiles.target_kind == .cpu) {
        p.remove(.horizontal_matmul);
        p.remove(.add_norm);
        p.remove(.gate);
    }
    return p;
}

pub const Ctx = struct {
    gpa: std.mem.Allocator,
    mgr: *StorageManager,
    /// Where this compile is going. A pass that derives a weight keys it on the device,
    /// because a derived weight belongs to `(sources, tiling, device)` and not to any one
    /// compiled program (see `opt/horizontal_matmul.concatColumns`).
    target: target_mod.Target,
};

/// Graph -> Graph, before the value->tensor map is sized so appended values compile
/// normally. Shapes are concrete here (`infer` has run) and `rewriter.rewrite`
/// re-infers, so a pass never has to predict what it just built.
///
/// `rewrite` runs each rule over every node list, control-flow bodies included, so a
/// model whose hot path is an in-graph `Loop` is optimized like any other.
pub fn graphPasses(ctx: Ctx, g: *Graph) Error!void {
    if (ctx.target.passes.contains(.pointwise_conv)) {
        try rewriter_mod.rewrite(ctx.gpa, g, pointwise_conv.Rule{});
    }
    if (ctx.target.passes.contains(.horizontal_matmul)) {
        try rewriter_mod.rewrite(ctx.gpa, g, horizontal_matmul.Rule{
            .mgr = ctx.mgr,
            .target = ctx.target,
        });
    }
}

/// Program -> Program, after placement (targets and host-operand masks are known)
/// and before workspace planning (so killed intermediates never get a slot). The
/// returned map is destination->source for views this dropped; `workspace.plan`
/// hands each destination the source's backing. Caller owns it.
///
/// Every step list is rewritten, control-flow bodies included, so a model whose hot path
/// is an in-graph `Loop` is scheduled like any other. `step_uses` is what keeps that
/// sound: a rule only reasons by index about tensors confined to the list it edits.
pub fn stepPasses(ctx: Ctx, prog: *Program, owned: []const TensorId) Error!AliasMap {
    var fused: usize = 0;
    if (ctx.target.passes.contains(.add_norm)) fused += try fuseSteps(ctx, prog, owned, fuse_steps.addNorm);
    if (ctx.target.passes.contains(.gate)) fused += try fuseSteps(ctx, prog, owned, fuse_steps.gate);
    // Every fused pair orphans the intermediate its producer wrote; workspace planning
    // releases it, so the stale placement entry has to go first.
    if (fused != 0) try placement.pruneOrphaned(ctx.gpa, prog);

    if (!ctx.target.passes.contains(.alias_views)) return .init(ctx.gpa);
    var map: AliasMap = .init(ctx.gpa);
    errdefer map.deinit();
    var list: usize = 0;
    while (list < step_uses.listCount(prog)) : (list += 1) {
        var ed = try Editor.init(ctx.gpa, ctx.mgr, prog, owned, list);
        defer ed.deinit();
        try alias_views.elide(&ed, &map);
        _ = try ed.commit();
    }
    return map;
}

/// One peephole per step list, each over a freshly collected use map, so no rule has to
/// reason about which facts a previous rewrite moved — including a rewrite this same
/// pass just made to another list.
fn fuseSteps(ctx: Ctx, prog: *Program, owned: []const TensorId, comptime rule: fn (*Editor) void) Error!usize {
    var fused: usize = 0;
    var list: usize = 0;
    while (list < step_uses.listCount(prog)) : (list += 1) {
        var ed = try Editor.init(ctx.gpa, ctx.mgr, prog, owned, list);
        defer ed.deinit();
        rule(&ed);
        fused += try ed.commit();
    }
    return fused;
}
