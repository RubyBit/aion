// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

//! Compile-time graph validation: inference errors and shape rejection. Nothing here executes a backend — numeric conformance
//! lives in `backend/cpu/test_cpu_backend.zig`, and lowered-structure golden
//! snapshots live in `test_program_golden.zig`.

const std = @import("std");

const manager_mod = @import("../storage/manager.zig");
const graph_mod = @import("graph.zig");
const infer_mod = @import("infer.zig");

const program = @import("program.zig");

test "graph: matmul rejects mismatched non-quant dtypes" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const m: usize = 2;
    const k: usize = 3;
    const n: usize = 4;

    const a_bytes_len: usize = m * k * 2;
    const b_bytes_len: usize = k * n * 4;

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const a_tid = try sm.createTensor(.f16, &[_]usize{ m, k }, .{});
    const b_tid = try sm.createTensor(.f32, &[_]usize{ k, n }, .{});

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

    try std.testing.expectError(infer_mod.InferError.DTypeMismatch, program.compileGraph(allocator, &g, &sm, .cpu()));
}

test "graph: cached grouped-query attention enforces H_q % H_kv == 0" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var sm: manager_mod.StorageManager = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const q_tid: manager_mod.TensorId = try sm.createTensor(.f32, &[_]usize{ 1, 1, 3, 2 }, .{});
    const k_tid: manager_mod.TensorId = try sm.createTensor(.f32, &[_]usize{ 1, 2, 4, 2 }, .{});
    const v_tid: manager_mod.TensorId = try sm.createTensor(.f32, &[_]usize{ 1, 2, 4, 2 }, .{});
    const pos_tid: manager_mod.TensorId = try sm.createTensor(.i32, &[_]usize{ 1, 1 }, .{});
    const end_tid: manager_mod.TensorId = try sm.createTensor(.i32, &[_]usize{1}, .{});

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

    const out: graph_mod.ValueId = try g.addAttention(q_in, k_in, v_in, pos_in, end_in, 1.0, .causal, 0.0);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    try std.testing.expectError(infer_mod.InferError.ShapeMismatch, program.compileGraph(allocator, &g, &sm, .cpu()));
}

test "graph: attention controls are independent and a bounded window needs aligned runs" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    {
        var g = graph_mod.Graph.init(allocator);
        defer g.deinit();
        const q = try g.addInput(.f32, &[_]usize{ 2, 1, 4, 8 });
        const k = try g.addInput(.f32, &[_]usize{ 2, 7, 2, 8 });
        const v = try g.addInput(.f32, &[_]usize{ 2, 7, 2, 6 });
        const positions = try g.addInput(.i32, &[_]usize{ 2, 1 });
        const out = try g.addAttention(q, k, v, positions, null, 0.25, .causal, 0.0);
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
        _ = try g.addAttention(q, k, v, null, lengths, 0.25, .full, 0.0);
        try infer_mod.infer(&g);
    }

    {
        var g = graph_mod.Graph.init(allocator);
        defer g.deinit();
        const q = try g.addInput(.f32, &[_]usize{ 1, 1, 4, 8 });
        const k = try g.addInput(.f32, &[_]usize{ 1, 7, 2, 8 });
        const v = try g.addInput(.f32, &[_]usize{ 1, 7, 2, 6 });
        _ = try g.addAttention(q, k, v, null, null, 0.25, .causal, 0.0);
        try std.testing.expectError(infer_mod.InferError.ShapeMismatch, infer_mod.infer(&g));
    }

    {
        var g = graph_mod.Graph.init(allocator);
        defer g.deinit();
        const q = try g.addInput(.f32, &[_]usize{ 1, 7, 4, 8 });
        const k = try g.addInput(.f32, &[_]usize{ 1, 7, 2, 8 });
        const v = try g.addInput(.f32, &[_]usize{ 1, 7, 2, 6 });
        _ = try g.addAttention(q, k, v, null, null, 0.25, .sliding(3, graph_mod.AttentionWindow.unbounded), 0.0);
        try infer_mod.infer(&g);
    }
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
        try g.bindExternal(a, try sm.createTensor(.i32, &.{1}, .{}));
        const b = try g.addInput(.i32, &.{1});
        try g.bindExternal(b, try sm.createTensor(.i32, &.{1}, .{}));
        const x = try g.addInput(.f32, &.{4});
        try g.bindExternal(x, try sm.createTensor(.f32, &.{4}, .{}));

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

        var prog = try program.compileGraph(allocator, &g, &sm, .init(.{ .kind = if (target == .cpu) .cpu else .gpu }, .row_major));
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
