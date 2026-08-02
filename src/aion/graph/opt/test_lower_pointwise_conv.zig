// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Tests for pointwise (1x1) Conv1D -> MatMul lowering: the rewrite fires only on
//! qualifying convs and is numerically identical to the direct conv path.

const std = @import("std");

const types = @import("../../backend/types.zig");
const manager_mod = @import("../../storage/manager.zig");
const graph_mod = @import("../graph.zig");
const plan_mod = @import("../plan.zig");
const program = @import("../program.zig");
const cpu_backend_mod = @import("../../backend/cpu/cpu_backend.zig");
const backend_mod = @import("../../backend/backend.zig");

const TensorId = manager_mod.TensorId;
const ValueId = graph_mod.ValueId;

fn asF32(buf: []u8) []align(1) f32 {
    const ptr: [*]align(1) f32 = @ptrCast(buf.ptr);
    return ptr[0 .. buf.len / @sizeOf(f32)];
}

fn f32Tensor(sm: *manager_mod.StorageManager, shape: []const usize, seed: usize) !TensorId {
    var n: usize = 1;
    for (shape) |d| n *= d;
    const vals = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(vals);
    for (vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i + seed) % 11)) - 5)) * 0.1;
    const tid = try sm.createTiledTensor(.f32, shape, shape, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(tid, std.mem.sliceAsBytes(vals));
    return tid;
}

fn countOp(g: *const graph_mod.Graph, tag: std.meta.Tag(graph_mod.Op)) usize {
    var c: usize = 0;
    for (g.nodes.items) |node| {
        if (std.meta.activeTag(node.op) == tag) c += 1;
    }
    return c;
}

const Dims = struct { l: usize = 3, cin: usize = 8, cout: usize = 4 };

fn buildConvGraph(g: *graph_mod.Graph, x_tid: TensorId, w_tid: TensorId, bias_tid: ?TensorId, d: Dims) !void {
    const x = try g.addInput(.f32, &[_]usize{ 1, d.l, d.cin });
    try g.bindExternal(x, @intCast(x_tid));
    const w = try g.addInput(.f32, &[_]usize{ 1, d.cin, d.cout }); // [k=1, c_in, c_out]
    try g.bindExternal(w, @intCast(w_tid));
    var bias: ?ValueId = null;
    if (bias_tid) |bt| {
        const bv = try g.addInput(.f32, &[_]usize{d.cout});
        try g.bindExternal(bv, @intCast(bt));
        bias = bv;
    }
    const out = try g.addConv1D(x, w, bias, 1, 1, 0, 0, 1); // stride/dil 1, no pad, groups 1
    try g.setOutputs(&[_]ValueId{out});
}

fn runParity(with_bias: bool) !void {
    const allocator = std.testing.allocator;
    const d = Dims{};
    const policy: plan_mod.TilePolicy = .{ .tile_alignment = 64 };

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const x_tid = try f32Tensor(&sm, &[_]usize{ 1, d.l, d.cin }, 1);
    const w_tid = try f32Tensor(&sm, &[_]usize{ 1, d.cin, d.cout }, 7);
    const bias_tid: ?TensorId = if (with_bias) try f32Tensor(&sm, &[_]usize{d.cout}, 3) else null;

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: backend_mod.Backend = cpu.backend();

    const out_elems = d.l * d.cout;

    // Reference: pass disabled — the graph stays a Conv1D.
    const ref = try allocator.alloc(u8, out_elems * 4);
    defer allocator.free(ref);
    {
        var g = graph_mod.Graph.init(allocator);
        defer g.deinit();
        try buildConvGraph(&g, x_tid, w_tid, bias_tid, d);
        var prog = try program.compileGraphOpt(allocator, &g, &sm, policy, .{ .lower_pointwise_conv = false });
        defer prog.deinit();
        try backend.executeProgram(&prog, sm.tensorStore());
        try std.testing.expectEqual(@as(usize, 1), countOp(&g, .Conv1D));
        try std.testing.expectEqual(@as(usize, 0), countOp(&g, .MatMul));
        try sm.readToPackedScalar(prog.outputs[0], ref);
    }

    // Lowered: Conv1D becomes MatMul (+ broadcast-add when biased).
    {
        var g = graph_mod.Graph.init(allocator);
        defer g.deinit();
        try buildConvGraph(&g, x_tid, w_tid, bias_tid, d);
        var prog = try program.compileGraphOpt(allocator, &g, &sm, policy, .{});
        defer prog.deinit();
        try backend.executeProgram(&prog, sm.tensorStore());

        try std.testing.expectEqual(@as(usize, 0), countOp(&g, .Conv1D));
        try std.testing.expectEqual(@as(usize, 1), countOp(&g, .MatMul));
        if (with_bias) {
            try std.testing.expectEqual(@as(usize, 1), countOp(&g, .ElemwiseBinary));
        }

        const got = try allocator.alloc(u8, out_elems * 4);
        defer allocator.free(got);
        try sm.readToPackedScalar(prog.outputs[0], got);

        const gv = asF32(got);
        const rv = asF32(ref);
        var max_abs: f32 = 0;
        for (gv, rv) |x, y| max_abs = @max(max_abs, @abs(x - y));
        try std.testing.expect(max_abs <= 1e-5);
    }
}

test "lower_pointwise_conv: 1x1 conv1d lowers to matmul and matches conv output" {
    try runParity(false);
}

test "lower_pointwise_conv: biased 1x1 conv1d lowers to matmul + broadcast add" {
    try runParity(true);
}

test "lower_pointwise_conv: non-pointwise conv1d is left as Conv1D" {
    const allocator = std.testing.allocator;
    const policy: plan_mod.TilePolicy = .{ .tile_alignment = 64 };

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    // k=3 conv (not pointwise): must NOT be rewritten.
    const x_tid = try f32Tensor(&sm, &[_]usize{ 1, 6, 8 }, 1);
    const w_tid = try f32Tensor(&sm, &[_]usize{ 3, 8, 4 }, 7); // [k=3, c_in, c_out]

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();
    const x = try g.addInput(.f32, &[_]usize{ 1, 6, 8 });
    try g.bindExternal(x, @intCast(x_tid));
    const w = try g.addInput(.f32, &[_]usize{ 3, 8, 4 });
    try g.bindExternal(w, @intCast(w_tid));
    const out = try g.addConv1D(x, w, null, 1, 1, 1, 1, 1);
    try g.setOutputs(&[_]ValueId{out});

    var prog = try program.compileGraphOpt(allocator, &g, &sm, policy, .{});
    defer prog.deinit();
    try std.testing.expectEqual(@as(usize, 1), countOp(&g, .Conv1D));
    try std.testing.expectEqual(@as(usize, 0), countOp(&g, .MatMul));
}
