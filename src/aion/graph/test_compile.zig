// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

//! Compile-time graph validation: inference errors, shape rejection, and
//! tiling decisions. Nothing here executes a backend — numeric conformance
//! lives in `backend/cpu/test_cpu_backend.zig`, and lowered-structure golden
//! snapshots live in `test_program_golden.zig`.

const std = @import("std");

const manager_mod = @import("../storage/manager.zig");
const graph_mod = @import("graph.zig");
const infer_mod = @import("infer.zig");
const plan_mod = @import("plan.zig");

test "gpu general and matmul tile caps are independent" {
    var policy = plan_mod.tilePolicyForTarget(.{ .kind = .webgpu });
    policy.base_square_2d = 32;
    policy.matmul_mn_tile_cap = 512;
    policy.matmul_k_tile_cap = 128;

    try std.testing.expectEqual([2]usize{ 32, 32 }, plan_mod.chooseTileShape2DSquare(policy, 1024, 1024));
    const mm = plan_mod.chooseMatMulTiles(policy, 1024, 1024, 1024, .f32);
    try std.testing.expectEqual(@as(usize, 512), mm.tm);
    try std.testing.expectEqual(@as(usize, 512), mm.tn);
    try std.testing.expectEqual(@as(usize, 128), mm.tk);
    try std.testing.expectEqual(@as(usize, 512), plan_mod.matMulMHint(policy));
}
const program = @import("program.zig");

test "plan: chooseTileShape2DSquare handles skinny matrices" {
    const policy: plan_mod.TilePolicy = .{};

    const t0: [2]usize = plan_mod.chooseTileShape2DSquare(policy, 1, 128);
    try std.testing.expectEqual(@as(usize, 1), t0[0]);
    try std.testing.expectEqual(@min(@as(usize, 128), policy.base_1d), t0[1]);
    try std.testing.expect(t0[1] > 1);

    const t1: [2]usize = plan_mod.chooseTileShape2DSquare(policy, 256, 1);
    try std.testing.expectEqual(@min(@as(usize, 256), policy.base_1d), t1[0]);
    try std.testing.expectEqual(@as(usize, 1), t1[1]);
    try std.testing.expect(t1[0] > 1);
}

test "graph: matmul rejects mismatched non-quant dtypes" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const m: usize = 2;
    const k: usize = 3;
    const n: usize = 4;

    const a_bytes_len: usize = m * k * 2;
    const b_bytes_len: usize = k * n * 4;

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const a_tid = try sm.createTiledTensor(.f16, &[_]usize{ m, k }, &[_]usize{ 2, 2 }, .{ .tile_alignment = 64 });
    const b_tid = try sm.createTiledTensor(.f32, &[_]usize{ k, n }, &[_]usize{ 2, 2 }, .{ .tile_alignment = 64 });

    // Contents are irrelevant; compilation should fail before execution.
    var a_zero: [a_bytes_len]u8 = @splat(0);
    var b_zero: [b_bytes_len]u8 = @splat(0);
    try sm.writeFromPackedScalar(a_tid, a_zero[0..]);
    try sm.writeFromPackedScalar(b_tid, b_zero[0..]);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const a_in = try g.addInput(.f16, &[_]usize{ m, k });
    const b_in = try g.addInput(.f32, &[_]usize{ k, n });
    try g.bindExternal(a_in, @intCast(a_tid));
    try g.bindExternal(b_in, @intCast(b_tid));

    const c = try g.addMatMul(a_in, b_in, 1.0, 0.0);
    try g.setOutputs(&[_]graph_mod.ValueId{c});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 4, .tile_alignment = 64 };
    try std.testing.expectError(infer_mod.InferError.DTypeMismatch, program.compileGraph(allocator, &g, &sm, policy));
}

test "graph: cached grouped-query attention enforces H_q % H_kv == 0" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var sm: manager_mod.StorageManager = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const q_tid: manager_mod.TensorId = try sm.createTiledTensor(.f32, &[_]usize{ 1, 1, 3, 2 }, &[_]usize{ 1, 1, 1, 2 }, .{ .tile_alignment = 64 });
    const k_tid: manager_mod.TensorId = try sm.createTiledTensor(.f32, &[_]usize{ 1, 2, 4, 2 }, &[_]usize{ 1, 1, 2, 2 }, .{ .tile_alignment = 64 });
    const v_tid: manager_mod.TensorId = try sm.createTiledTensor(.f32, &[_]usize{ 1, 2, 4, 2 }, &[_]usize{ 1, 1, 2, 2 }, .{ .tile_alignment = 64 });
    const pos_tid: manager_mod.TensorId = try sm.createTiledTensor(.i32, &[_]usize{ 1, 1 }, &[_]usize{ 1, 1 }, .{ .tile_alignment = 64 });
    const end_tid: manager_mod.TensorId = try sm.createTiledTensor(.i32, &[_]usize{1}, &[_]usize{1}, .{ .tile_alignment = 64 });

    var q_init: [1 * 1 * 3 * 2]f32 = @splat(0.0);
    var k_init: [1 * 2 * 4 * 2]f32 = @splat(0.0);
    var v_init: [1 * 2 * 4 * 2]f32 = @splat(0.0);
    var pos_init: [1]i32 = .{0};
    var end_init: [1]i32 = .{1};

    try sm.writeFromPackedScalar(q_tid, std.mem.sliceAsBytes(q_init[0..]));
    try sm.writeFromPackedScalar(k_tid, std.mem.sliceAsBytes(k_init[0..]));
    try sm.writeFromPackedScalar(v_tid, std.mem.sliceAsBytes(v_init[0..]));
    try sm.writeFromPackedScalar(pos_tid, std.mem.sliceAsBytes(pos_init[0..]));
    try sm.writeFromPackedScalar(end_tid, std.mem.sliceAsBytes(end_init[0..]));

    var g: graph_mod.Graph = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const q_in: graph_mod.ValueId = try g.addInput(.f32, &[_]usize{ 1, 1, 3, 2 });
    const k_in: graph_mod.ValueId = try g.addInput(.f32, &[_]usize{ 1, 2, 4, 2 });
    const v_in: graph_mod.ValueId = try g.addInput(.f32, &[_]usize{ 1, 2, 4, 2 });
    const pos_in: graph_mod.ValueId = try g.addInput(.i32, &[_]usize{ 1, 1 });
    const end_in: graph_mod.ValueId = try g.addInput(.i32, &[_]usize{1});

    try g.bindExternal(q_in, @intCast(q_tid));
    try g.bindExternal(k_in, @intCast(k_tid));
    try g.bindExternal(v_in, @intCast(v_tid));
    try g.bindExternal(pos_in, @intCast(pos_tid));
    try g.bindExternal(end_in, @intCast(end_tid));

    const out: graph_mod.ValueId = try g.addAttention(q_in, k_in, v_in, pos_in, end_in, 1.0, true, 0, 0.0);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 2, .tile_alignment = 64 };
    try std.testing.expectError(infer_mod.InferError.ShapeMismatch, program.compileGraph(allocator, &g, &sm, policy));
}

test "graph: attention controls are independent and ambiguous causal defaults are rejected" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    {
        var g = graph_mod.Graph.init(allocator);
        defer g.deinit();
        const q = try g.addInput(.f32, &[_]usize{ 2, 1, 4, 8 });
        const k = try g.addInput(.f32, &[_]usize{ 2, 7, 2, 8 });
        const v = try g.addInput(.f32, &[_]usize{ 2, 7, 2, 6 });
        const positions = try g.addInput(.i32, &[_]usize{ 2, 1 });
        const out = try g.addAttention(q, k, v, positions, null, 0.25, true, 0, 0.0);
        try infer_mod.infer(&g);
        try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 1, 4, 6 }, g.values.items[out].shape);
    }

    {
        var g = graph_mod.Graph.init(allocator);
        defer g.deinit();
        const q = try g.addInput(.f32, &[_]usize{ 2, 3, 4, 8 });
        const k = try g.addInput(.f32, &[_]usize{ 2, 7, 2, 8 });
        const v = try g.addInput(.f32, &[_]usize{ 2, 7, 2, 6 });
        const lengths = try g.addInput(.i32, &[_]usize{2});
        _ = try g.addAttention(q, k, v, null, lengths, 0.25, false, 0, 0.0);
        try infer_mod.infer(&g);
    }

    {
        var g = graph_mod.Graph.init(allocator);
        defer g.deinit();
        const q = try g.addInput(.f32, &[_]usize{ 1, 1, 4, 8 });
        const k = try g.addInput(.f32, &[_]usize{ 1, 7, 2, 8 });
        const v = try g.addInput(.f32, &[_]usize{ 1, 7, 2, 6 });
        _ = try g.addAttention(q, k, v, null, null, 0.25, true, 0, 0.0);
        try std.testing.expectError(infer_mod.InferError.ShapeMismatch, infer_mod.infer(&g));
    }

    {
        var g = graph_mod.Graph.init(allocator);
        defer g.deinit();
        const q = try g.addInput(.f32, &[_]usize{ 1, 7, 4, 8 });
        const k = try g.addInput(.f32, &[_]usize{ 1, 7, 2, 8 });
        const v = try g.addInput(.f32, &[_]usize{ 1, 7, 2, 6 });
        _ = try g.addAttention(q, k, v, null, null, 0.25, false, 4, 0.0);
        try std.testing.expectError(infer_mod.InferError.InvalidGraph, infer_mod.infer(&g));
    }
}

test "graph: rope1d retiles a head-dim-split input instead of rejecting it" {
    // A *computed* `[B, L, H, D]` q/k — the shape every attention layer feeds RoPE —
    // is square-tiled across its last two axes by the default policy once it passes
    // `small_tensor_threshold`, which splits D. RoPE needs whole head-dim vectors
    // (it rotates i against i + D/2), and used to reject that tiling outright,
    // failing any prefill long enough to cross the threshold. It must retile now.
    const allocator: std.mem.Allocator = std.testing.allocator;

    const l: usize = 2;
    const k_in_dim: usize = 2;
    const heads: usize = 2;
    const head_dim: usize = 4;

    var sm: manager_mod.StorageManager = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const x_tid: manager_mod.TensorId = try sm.createTiledTensor(.f32, &[_]usize{ 1, l, k_in_dim }, &[_]usize{ 1, 1, k_in_dim }, .{ .tile_alignment = 64 });
    const w_tid: manager_mod.TensorId = try sm.createTiledTensor(.f32, &[_]usize{ k_in_dim, heads * head_dim }, &[_]usize{ 1, 2 }, .{ .tile_alignment = 64 });
    const pos_tid: manager_mod.TensorId = try sm.createTiledTensor(.i32, &[_]usize{ 1, l }, &[_]usize{ 1, l }, .{ .tile_alignment = 64 });

    var x_init: [1 * l * k_in_dim]f32 = @splat(0.0);
    var w_init: [k_in_dim * heads * head_dim]f32 = @splat(0.0);
    var pos_init: [l]i32 = .{ 0, 1 };
    try sm.writeFromPackedScalar(x_tid, std.mem.sliceAsBytes(x_init[0..]));
    try sm.writeFromPackedScalar(w_tid, std.mem.sliceAsBytes(w_init[0..]));
    try sm.writeFromPackedScalar(pos_tid, std.mem.sliceAsBytes(pos_init[0..]));

    var g: graph_mod.Graph = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in: graph_mod.ValueId = try g.addInput(.f32, &[_]usize{ 1, l, k_in_dim });
    const w_in: graph_mod.ValueId = try g.addInput(.f32, &[_]usize{ k_in_dim, heads * head_dim });
    const pos_in: graph_mod.ValueId = try g.addInput(.i32, &[_]usize{ 1, l });
    try g.bindExternal(x_in, @intCast(x_tid));
    try g.bindExternal(w_in, @intCast(w_tid));
    try g.bindExternal(pos_in, @intCast(pos_tid));

    // Projected then split into heads: a computed value, so its tiling is the
    // policy's, not an input signature's.
    const proj: graph_mod.ValueId = try g.addMatMul(x_in, w_in, 1.0, 0.0);
    const q: graph_mod.ValueId = try g.addViewReshape(proj, &[_]usize{ 1, l, heads, head_dim });
    const out: graph_mod.ValueId = try g.addRoPE1D(q, pos_in, 10000.0, 1.0, 1.0);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    // `small_tensor_threshold = 1` forces the square-tiling path that splits D;
    // `base_square_2d = 2` makes the split observable at these tiny sizes.
    const policy: plan_mod.TilePolicy = .{
        .base_square_2d = 2,
        .base_1d = 2,
        .small_tensor_threshold = 1,
        .tile_alignment = 64,
    };
    var prog: program.Program = try program.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    // The lowered step must see whole head-dim vectors on both x and out.
    var saw_rope: bool = false;
    for (prog.steps) |step| {
        switch (step.op) {
            .RoPE1DTiled => |s| {
                saw_rope = true;
                for ([_]manager_mod.TensorId{ s.x, s.out }) |tid| {
                    const t = try sm.getConst(tid);
                    try std.testing.expectEqual(@as(usize, head_dim), t.tile_shape[3]);
                    try std.testing.expectEqual(@as(usize, 1), t.tile_counts[3]);
                }
            },
            else => {},
        }
    }
    try std.testing.expect(saw_rope);
}

const executable = @import("../runtime/executable.zig");
const types = @import("../backend/types.zig");

// The invariant the placement pass exists for: when the device writes a value
// the runtime must read on the host, the compiler emits the transfer and
// rewrites the consumer, rather than leaving the executor to notice at record
// time. A control-flow predicate is the only thing that asks for this — kernels
// either consume an operand on the device or the op is unsupported there.
test "placement: a device-written control predicate gets a transfer" {
    const allocator = std.testing.allocator;

    for ([_]types.BackendKind{ .webgpu, .cpu }) |target| {
        var sm = manager_mod.StorageManager.init(allocator);
        defer sm.deinit();

        var g = graph_mod.Graph.init(allocator);
        defer g.deinit();

        const a = try g.addInput(.i32, &.{1});
        try g.bindExternal(a, try sm.createTiledTensor(.i32, &.{1}, &.{1}, .{}));
        const b = try g.addInput(.i32, &.{1});
        try g.bindExternal(b, try sm.createTiledTensor(.i32, &.{1}, &.{1}, .{}));
        const x = try g.addInput(.f32, &.{4});
        try g.bindExternal(x, try sm.createTiledTensor(.f32, &.{4}, &.{4}, .{}));

        // The predicate is computed by a step, so the device owns it.
        const cond = try g.addElemwiseBinary(.add, a, b);
        try g.beginRegion();
        const then_v = try g.addUnary(.relu, x);
        const then_r = try g.endRegion(&.{then_v});
        try g.beginRegion();
        const else_v = try g.addUnary(.relu, x);
        const else_r = try g.endRegion(&.{else_v});
        const out = try g.addIf(cond, then_r, else_r);
        try g.setOutputs(&.{out});

        var prog = try program.compileGraph(allocator, &g, &sm, .{ .target_kind = target });
        defer prog.deinit();

        var transfers: usize = 0;
        for (prog.steps) |step| {
            if (step.op == .Transfer) transfers += 1;
        }
        // On CPU the value is already host-placed, so the transfer elides and the
        // schedule comes out unchanged — the pass is a no-op there by construction.
        try std.testing.expectEqual(@as(usize, if (target == .webgpu) 1 else 0), transfers);

        for (prog.steps) |step| {
            switch (step.op) {
                .If => |s| {
                    // The predicate is always a declared host read; only its
                    // placement differs between targets.
                    try std.testing.expect(step.host_operands != 0);
                    try std.testing.expectEqual(
                        executable.Placement{},
                        prog.placementOf(s.cond).?,
                    );
                },
                else => {},
            }
        }

        try prog.validatePlacements();
    }
}


// `x + rmsnorm(y)` is a SCHEDULE, not a meaning. There is no graph op for it: nobody
// authors a "residual norm", so `program/fuse_steps.zig` recovers it from the lowered
// step list for a device target, and leaves the pair alone on CPU where the unfused form
// is the numerical oracle the fused kernel is checked against
// (`backend/gpu/test_gpu_backend.zig`). Asserting it here costs no device.
test "fuse_steps: residual + rmsnorm is one step on device and a pair on cpu" {
    const allocator = std.testing.allocator;

    for ([_]types.BackendKind{ .webgpu, .cpu }) |target| {
        var sm = manager_mod.StorageManager.init(allocator);
        defer sm.deinit();

        var g = graph_mod.Graph.init(allocator);
        defer g.deinit();

        const M = 2;
        const N = 8;
        const res = try g.addInput(.f32, &.{ M, N });
        try g.bindExternal(res, try sm.createTiledTensor(.f32, &.{ M, N }, &.{ M, N }, .{}));
        const x = try g.addInput(.f32, &.{ M, N });
        try g.bindExternal(x, try sm.createTiledTensor(.f32, &.{ M, N }, &.{ M, N }, .{}));
        const gamma = try g.addInput(.f32, &.{N});
        try g.bindExternal(gamma, try sm.createTiledTensor(.f32, &.{N}, &.{N}, .{}));
        const beta = try g.addInput(.f32, &.{N});
        try g.bindExternal(beta, try sm.createTiledTensor(.f32, &.{N}, &.{N}, .{}));

        const normed = try g.addRMSNorm(x, gamma, beta, 1e-6, &.{N});
        const out = try g.addElemwiseBinary(.add, res, normed);
        try g.setOutputs(&.{out});

        var prog = try program.compileGraph(allocator, &g, &sm, .{ .target_kind = target });
        defer prog.deinit();

        var norms: usize = 0;
        var adds: usize = 0;
        var fused: usize = 0;
        for (prog.steps) |step| switch (step.op) {
            // The residual is a field, not a tag, so what distinguishes a fused norm from
            // a plain one is the operand being there — assert on that, not on a name.
            .RMSNormTiled => |s| if (s.residual != null) {
                fused += 1;
            } else {
                norms += 1;
            },
            .ElemwiseBinaryTiled => adds += 1,
            else => {},
        };

        const device = target != .cpu;
        try std.testing.expectEqual(@as(usize, if (device) 0 else 1), norms);
        try std.testing.expectEqual(@as(usize, if (device) 0 else 1), adds);
        try std.testing.expectEqual(@as(usize, if (device) 1 else 0), fused);

        // The pass runs after placement, so the entry for the intermediate it killed has
        // to be gone too: `materializePlacements` demands backing for everything listed,
        // while `workspace.plan` releases anything with no remaining use.
        try prog.validatePlacements();
        for (prog.tensor_placements) |entry| {
            try std.testing.expect(try sm.tensorHasBacking(entry.id));
        }
    }
}


// The other half of the tiering rule: a gate IS a named primitive, so it is graph
// vocabulary — but as a PARAMETER of the generic elementwise op, not a tag of its own.
// `nn.GatedMLP` emits it directly; this asserts the step-level peephole still recovers it
// from a graph that spells it out, for every activation, and still leaves CPU alone.
test "fuse_steps: unary + mul becomes one gate step on device, stays a pair on cpu" {
    const allocator = std.testing.allocator;

    for ([_]types.BackendKind{ .webgpu, .cpu }) |target| {
        for ([_]types.UnaryOp{ .gelu, .silu, .relu }) |act| {
            var sm = manager_mod.StorageManager.init(allocator);
            defer sm.deinit();

            var g = graph_mod.Graph.init(allocator);
            defer g.deinit();

            const N = 8;
            const x = try g.addInput(.f32, &.{ 2, N });
            try g.bindExternal(x, try sm.createTiledTensor(.f32, &.{ 2, N }, &.{ 2, N }, .{}));
            const y = try g.addInput(.f32, &.{ 2, N });
            try g.bindExternal(y, try sm.createTiledTensor(.f32, &.{ 2, N }, &.{ 2, N }, .{}));

            const out = try g.addElemwiseBinary(.mul, try g.addUnary(act, x), y);
            try g.setOutputs(&.{out});

            var prog = try program.compileGraph(allocator, &g, &sm, .{ .target_kind = target });
            defer prog.deinit();

            var unaries: usize = 0;
            var muls: usize = 0;
            var gates: usize = 0;
            for (prog.steps) |step| switch (step.op) {
                .UnaryTiled => unaries += 1,
                .ElemwiseBinaryTiled => |s| if (s.op == .gate) {
                    gates += 1;
                    // The activation has to survive the rewrite, or every gate would be
                    // a GEGLU regardless of what the graph asked for.
                    try std.testing.expectEqual(act, s.act);
                } else {
                    muls += 1;
                },
                else => {},
            };

            const device = target != .cpu;
            try std.testing.expectEqual(@as(usize, if (device) 0 else 1), unaries);
            try std.testing.expectEqual(@as(usize, if (device) 0 else 1), muls);
            try std.testing.expectEqual(@as(usize, if (device) 1 else 0), gates);
            try prog.validatePlacements();
        }
    }
}
