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
    var a_zero: [a_bytes_len]u8 = [_]u8{0} ** a_bytes_len;
    var b_zero: [b_bytes_len]u8 = [_]u8{0} ** b_bytes_len;
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

    var q_init: [1 * 1 * 3 * 2]f32 = .{0.0} ** (1 * 1 * 3 * 2);
    var k_init: [1 * 2 * 4 * 2]f32 = .{0.0} ** (1 * 2 * 4 * 2);
    var v_init: [1 * 2 * 4 * 2]f32 = .{0.0} ** (1 * 2 * 4 * 2);
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

    const out: graph_mod.ValueId = try g.addMultiHeadAttentionCached(q_in, k_in, v_in, pos_in, end_in, 1.0, true, 0, 0.0);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 2, .tile_alignment = 64 };
    try std.testing.expectError(infer_mod.InferError.ShapeMismatch, program.compileGraph(allocator, &g, &sm, policy));
}
