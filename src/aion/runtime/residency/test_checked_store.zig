// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Validates that CPU execution is lease-disciplined by running representative
//! programs through `CheckedTensorStore` and asserting the live-lease count is
//! balanced (zero leaks, no underflow). This is the contract the device
//! residency layer relies on.

const std = @import("std");

const graph_mod = @import("../../graph/graph.zig");
const program = @import("../../graph/program.zig");
const manager_mod = @import("../../storage/manager.zig");
const cpu_backend_mod = @import("../../backend/cpu/cpu_backend.zig");
const checked = @import("checked_store.zig");
const api_tiling = @import("../../api/tiling.zig");
const types = @import("../../backend/types.zig");

const ValueId = graph_mod.ValueId;
const TensorId = manager_mod.TensorId;

fn bindInput(g: *graph_mod.Graph, mgr: *manager_mod.StorageManager, dtype: types.DType, shape: []const usize) !ValueId {
    var tile_mem: [api_tiling.MAX_RANK]usize = undefined;
    const tile = tile_mem[0..shape.len];
    try api_tiling.fillDefaultTileShape(.{}, dtype, shape, tile);
    const tid = try mgr.createTiledTensor(dtype, shape, tile, .{});
    const v = try g.addInput(dtype, shape);
    try g.bindExternal(v, tid);
    return v;
}

fn writeScalarI32(mgr: *manager_mod.StorageManager, id: TensorId, value: i32) !void {
    var buf: [@sizeOf(i32)]u8 = undefined;
    std.mem.writeInt(i32, buf[0..@sizeOf(i32)], value, .little);
    try mgr.writeFromPackedScalar(id, buf[0..]);
}

/// Compile `g` and run it through a CheckedTensorStore; assert lease balance.
fn runChecked(g: *graph_mod.Graph, mgr: *manager_mod.StorageManager) !void {
    const a = std.testing.allocator;

    var prog = try program.compileGraph(a, g, mgr, .{});
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(a);
    defer cpu.deinit();

    var cstore = checked.CheckedTensorStore.init(mgr.tensorStore());
    try cpu.backend().executeProgram(&prog, cstore.tensorStore());

    try std.testing.expect(cstore.stats.total_acquires > 0);
    try std.testing.expect(!cstore.stats.underflow);
    try std.testing.expectEqual(@as(i64, 0), cstore.stats.live);
    try std.testing.expect(cstore.stats.balanced());
}

test "checked store: decode chain is lease-balanced" {
    const a = std.testing.allocator;
    var mgr = manager_mod.StorageManager.init(a);
    defer mgr.deinit();
    var g = graph_mod.Graph.init(a);
    defer g.deinit();

    const table = try bindInput(&g, &mgr, .f32, &[_]usize{ 256, 128 });
    const idx = try bindInput(&g, &mgr, .i32, &[_]usize{ 1, 1 }); // zeroed -> index 0, valid
    const emb = try g.addGatherRows(table, idx);
    const x = try g.addViewReshape(emb, &[_]usize{ 1, 128 });
    const gamma = try bindInput(&g, &mgr, .f32, &[_]usize{128});
    const beta = try bindInput(&g, &mgr, .f32, &[_]usize{128});
    const normed = try g.addRMSNorm(x, gamma, beta, 1e-5, &[_]usize{128});
    const W = try bindInput(&g, &mgr, .f32, &[_]usize{ 128, 256 });
    const logits = try g.addMatMul(normed, W, 1.0, 0.0);
    const out = try g.addSoftmax(logits, -1);
    try g.setOutputs(&[_]ValueId{out});

    try runChecked(&g, &mgr);
}

test "checked store: If control-flow is lease-balanced" {
    const a = std.testing.allocator;
    var mgr = manager_mod.StorageManager.init(a);
    defer mgr.deinit();

    const cond_tid = try mgr.createTiledTensor(.i32, &[_]usize{1}, &[_]usize{1}, .{});
    try writeScalarI32(&mgr, cond_tid, 1);

    var g = graph_mod.Graph.init(a);
    defer g.deinit();
    const cond = try g.addInput(.i32, &[_]usize{1});
    try g.bindExternal(cond, cond_tid);
    const then_v = try bindInput(&g, &mgr, .f32, &[_]usize{1});
    const else_v = try bindInput(&g, &mgr, .f32, &[_]usize{1});

    try g.beginRegion();
    const then_region = try g.endRegion(&[_]ValueId{then_v});
    try g.beginRegion();
    const else_region = try g.endRegion(&[_]ValueId{else_v});
    const out = try g.addIf(cond, then_region, else_region);
    try g.setOutputs(&[_]ValueId{out});

    try runChecked(&g, &mgr);
}

test "checked store: Loop control-flow (swap path) is lease-balanced" {
    const a = std.testing.allocator;
    var mgr = manager_mod.StorageManager.init(a);
    defer mgr.deinit();

    const carried_tid = try mgr.createTiledTensor(.f32, &[_]usize{1}, &[_]usize{1}, .{});
    const inc_tid = try mgr.createTiledTensor(.f32, &[_]usize{1}, &[_]usize{1}, .{});

    var g = graph_mod.Graph.init(a);
    defer g.deinit();
    const carried = try g.addInput(.f32, &[_]usize{1});
    try g.bindExternal(carried, carried_tid);
    const inc = try g.addInput(.f32, &[_]usize{1});
    try g.bindExternal(inc, inc_tid);

    try g.beginRegion();
    const next = try g.addElemwiseBinary(.add, carried, inc);
    const body_region = try g.endRegion(&[_]ValueId{next});
    const out = try g.addLoop(carried, body_region, 4);
    try g.setOutputs(&[_]ValueId{out});

    try runChecked(&g, &mgr);
}
