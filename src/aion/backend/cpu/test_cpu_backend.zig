// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

//! CPU backend conformance: compile graphs into executable programs and run
//! them through `CpuBackend`, checking numeric results against explicit math
//! references (incl. control-flow regions). Compile-time validation lives in
//! `graph/test_compile.zig`; lowered-structure golden snapshots live in
//! `graph/test_program_golden.zig`; leaf kernels in `test_cpu_kernels.zig`.

const std = @import("std");

const backend_mod = @import("../backend.zig");
const cpu_backend_mod = @import("cpu_backend.zig");
const types = @import("../types.zig");
const manager_mod = @import("../../storage/manager.zig");
const graph_mod = @import("../../graph/graph.zig");
const infer_mod = @import("../../graph/infer.zig");
const plan_mod = @import("../../graph/plan.zig");
const program = @import("../../graph/program.zig");
const Backend = backend_mod.Backend;

fn asF32Slice(buf: []u8) []align(1) f32 {
    std.debug.assert((buf.len % @sizeOf(f32)) == 0);
    const ptr: [*]align(1) f32 = @ptrCast(buf.ptr);
    return ptr[0 .. buf.len / @sizeOf(f32)];
}

fn asF16Slice(buf: []u8) []align(1) f16 {
    std.debug.assert((buf.len % @sizeOf(f16)) == 0);
    const ptr: [*]align(1) f16 = @ptrCast(buf.ptr);
    return ptr[0 .. buf.len / @sizeOf(f16)];
}

fn writeScalarF32(mgr: *manager_mod.StorageManager, id: manager_mod.TensorId, value: f32) !void {
    var buf: [@sizeOf(f32)]u8 = undefined;
    asF32Slice(buf[0..])[0] = value;
    try mgr.writeFromPackedScalar(id, buf[0..]);
}

fn readScalarF32(mgr: *manager_mod.StorageManager, id: manager_mod.TensorId) !f32 {
    var buf: [@sizeOf(f32)]u8 = undefined;
    try mgr.readToPackedScalar(id, buf[0..]);
    return asF32Slice(buf[0..])[0];
}

fn writeScalarI32(mgr: *manager_mod.StorageManager, id: manager_mod.TensorId, value: i32) !void {
    var buf: [@sizeOf(i32)]u8 = undefined;
    std.mem.writeInt(i32, buf[0..@sizeOf(i32)], value, .little);
    try mgr.writeFromPackedScalar(id, buf[0..]);
}

fn readScalarI32(mgr: *manager_mod.StorageManager, id: manager_mod.TensorId) !i32 {
    var buf: [@sizeOf(i32)]u8 = undefined;
    try mgr.readToPackedScalar(id, buf[0..]);
    return std.mem.readInt(i32, buf[0..@sizeOf(i32)], .little);
}

test "cpu backend: if selects region output" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var mgr = manager_mod.StorageManager.init(allocator);
    defer mgr.deinit();

    const cond_tid: manager_mod.TensorId = try mgr.createTiledTensor(.i32, &[_]usize{1}, &[_]usize{1}, .{});
    const then_tid: manager_mod.TensorId = try mgr.createTiledTensor(.f32, &[_]usize{1}, &[_]usize{1}, .{});
    const else_tid: manager_mod.TensorId = try mgr.createTiledTensor(.f32, &[_]usize{1}, &[_]usize{1}, .{});
    try writeScalarI32(&mgr, cond_tid, 1);
    try writeScalarF32(&mgr, then_tid, 42.0);
    try writeScalarF32(&mgr, else_tid, 7.0);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();
    const cond = try g.addInput(.i32, &[_]usize{1});
    try g.bindExternal(cond, cond_tid);
    const then_v = try g.addInput(.f32, &[_]usize{1});
    try g.bindExternal(then_v, then_tid);
    const else_v = try g.addInput(.f32, &[_]usize{1});
    try g.bindExternal(else_v, else_tid);

    try g.beginRegion();
    const then_region = try g.endRegion(&[_]graph_mod.ValueId{then_v});
    try g.beginRegion();
    const else_region = try g.endRegion(&[_]graph_mod.ValueId{else_v});
    const out = try g.addIf(cond, then_region, else_region);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    var prog = try program.compileGraph(allocator, &g, &mgr, .cpu(.{}));
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, mgr.tensorStore());
    try std.testing.expectEqual(@as(f32, 42.0), try readScalarF32(&mgr, prog.outputs[0]));

    try writeScalarI32(&mgr, cond_tid, 0);
    try cpu.backend().executeProgram(&prog, mgr.tensorStore());
    try std.testing.expectEqual(@as(f32, 7.0), try readScalarF32(&mgr, prog.outputs[0]));
}

test "cpu backend: loop carries state with tensor-id alias swap" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var mgr = manager_mod.StorageManager.init(allocator);
    defer mgr.deinit();

    const carried_tid: manager_mod.TensorId = try mgr.createTiledTensor(.f32, &[_]usize{1}, &[_]usize{1}, .{});
    const inc_tid: manager_mod.TensorId = try mgr.createTiledTensor(.f32, &[_]usize{1}, &[_]usize{1}, .{});
    try writeScalarF32(&mgr, carried_tid, 1.0);
    try writeScalarF32(&mgr, inc_tid, 2.0);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();
    const carried = try g.addInput(.f32, &[_]usize{1});
    try g.bindExternal(carried, carried_tid);
    const inc = try g.addInput(.f32, &[_]usize{1});
    try g.bindExternal(inc, inc_tid);

    try g.beginRegion();
    const next = try g.addElemwiseBinary(.add, carried, inc);
    const body_region = try g.endRegion(&[_]graph_mod.ValueId{next});
    const out = try g.addLoop(carried, body_region, 4);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    var prog = try program.compileGraph(allocator, &g, &mgr, .cpu(.{}));
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, mgr.tensorStore());
    try std.testing.expectEqual(@as(f32, 9.0), try readScalarF32(&mgr, prog.outputs[0]));
}

test "cpu backend: multi-carry loop with early-exit condition" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var mgr = manager_mod.StorageManager.init(allocator);
    defer mgr.deinit();

    // Carries: i (i32), acc (f32), active (i32 predicate). Constants one/ten/limit.
    const i_tid: manager_mod.TensorId = try mgr.createTiledTensor(.i32, &[_]usize{1}, &[_]usize{1}, .{});
    const acc_tid: manager_mod.TensorId = try mgr.createTiledTensor(.f32, &[_]usize{1}, &[_]usize{1}, .{});
    const active_tid: manager_mod.TensorId = try mgr.createTiledTensor(.i32, &[_]usize{1}, &[_]usize{1}, .{});
    const one_tid: manager_mod.TensorId = try mgr.createTiledTensor(.i32, &[_]usize{1}, &[_]usize{1}, .{});
    const ten_tid: manager_mod.TensorId = try mgr.createTiledTensor(.f32, &[_]usize{1}, &[_]usize{1}, .{});
    const limit_tid: manager_mod.TensorId = try mgr.createTiledTensor(.i32, &[_]usize{1}, &[_]usize{1}, .{});
    try writeScalarI32(&mgr, i_tid, 0);
    try writeScalarF32(&mgr, acc_tid, 0.0);
    try writeScalarI32(&mgr, active_tid, 1);
    try writeScalarI32(&mgr, one_tid, 1);
    try writeScalarF32(&mgr, ten_tid, 10.0);
    try writeScalarI32(&mgr, limit_tid, 3);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();
    const i_in = try g.addInput(.i32, &[_]usize{1});
    try g.bindExternal(i_in, i_tid);
    const acc_in = try g.addInput(.f32, &[_]usize{1});
    try g.bindExternal(acc_in, acc_tid);
    const active_in = try g.addInput(.i32, &[_]usize{1});
    try g.bindExternal(active_in, active_tid);
    const one = try g.addInput(.i32, &[_]usize{1});
    try g.bindExternal(one, one_tid);
    const ten = try g.addInput(.f32, &[_]usize{1});
    try g.bindExternal(ten, ten_tid);
    const limit = try g.addInput(.i32, &[_]usize{1});
    try g.bindExternal(limit, limit_tid);

    try g.beginRegion();
    const i_next = try g.addElemwiseBinary(.add, i_in, one);
    const acc_next = try g.addElemwiseBinary(.add, acc_in, ten);
    const active_next = try g.addElemwiseBinary(.lt, i_next, limit); // i32 {0,1}
    const body_region = try g.endRegion(&[_]graph_mod.ValueId{ i_next, acc_next, active_next });

    // cond_carry = 2 (active); static_max 100 but should stop at 3 iterations.
    const outs = try g.addLoopMulti(
        &[_]graph_mod.ValueId{ i_in, acc_in, active_in },
        body_region,
        100,
        2,
        true,
    );
    try g.setOutputs(&[_]graph_mod.ValueId{ outs[1], outs[0] }); // acc, i

    var prog = try program.compileGraph(allocator, &g, &mgr, .cpu(.{}));
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, mgr.tensorStore());
    try std.testing.expectEqual(@as(f32, 30.0), try readScalarF32(&mgr, prog.outputs[0]));
    try std.testing.expectEqual(@as(i32, 3), try readScalarI32(&mgr, prog.outputs[1]));
}

test "cpu backend: rmsnorm supports row tiles >256" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const batch: usize = 1;
    const seq: usize = 14;
    const groups: usize = 35;
    const hidden: usize = 300;

    const rows: usize = batch * seq * groups;
    const elems: usize = rows * hidden;

    const x_bytes_len: usize = elems * 4;
    const g_bytes_len: usize = hidden * 4;
    const b_bytes_len: usize = hidden * 4;

    const x_buf: []u8 = try allocator.alloc(u8, x_bytes_len);
    defer allocator.free(x_buf);
    const g_buf: []u8 = try allocator.alloc(u8, g_bytes_len);
    defer allocator.free(g_buf);
    const b_buf: []u8 = try allocator.alloc(u8, b_bytes_len);
    defer allocator.free(b_buf);

    const x_vals: []align(1) f32 = asF32Slice(x_buf);
    const g_vals: []align(1) f32 = asF32Slice(g_buf);
    const b_vals: []align(1) f32 = asF32Slice(b_buf);

    for (0..elems) |i| {
        const s: i32 = @as(i32, @intCast(i % 97)) - 48;
        x_vals[i] = @as(f32, @floatFromInt(s)) * 0.01;
    }
    for (0..hidden) |i| {
        g_vals[i] = 1.0 + (@as(f32, @floatFromInt(i % 13))) * 0.01;
        b_vals[i] = (@as(f32, @floatFromInt(i % 7))) * 0.001;
    }

    const eps: f32 = 1e-5;

    // Reference RMSNorm on packed input: y = (x * inv_rms) * gamma + beta.
    const y_ref_buf: []u8 = try allocator.alloc(u8, x_bytes_len);
    defer allocator.free(y_ref_buf);
    const y_ref: []align(1) f32 = asF32Slice(y_ref_buf);

    var r: usize = 0;
    while (r < rows) : (r += 1) {
        var sumsq: f32 = 0.0;
        var c: usize = 0;
        while (c < hidden) : (c += 1) {
            const v: f32 = x_vals[r * hidden + c];
            sumsq += v * v;
        }
        const msq: f32 = sumsq / @as(f32, @floatFromInt(hidden));
        const inv: f32 = 1.0 / std.math.sqrt(msq + eps);
        c = 0;
        while (c < hidden) : (c += 1) {
            const x0: f32 = x_vals[r * hidden + c];
            y_ref[r * hidden + c] = (x0 * inv * g_vals[c]) + b_vals[c];
        }
    }

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: Backend = cpu.backend();

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    // Force a single tile over the leading dims so rows_per_tile = 1*14*35 = 490 (>256).
    const x_tid = try sm.createTiledTensor(
        .f32,
        &[_]usize{ batch, seq, groups, hidden },
        &[_]usize{ batch, seq, groups, 128 },
        .{ .tile_alignment = 64 },
    );
    const gamma_tid = try sm.createTiledTensor(.f32, &[_]usize{hidden}, &[_]usize{128}, .{ .tile_alignment = 64 });
    const beta_tid = try sm.createTiledTensor(.f32, &[_]usize{hidden}, &[_]usize{128}, .{ .tile_alignment = 64 });

    try sm.writeFromPackedScalar(x_tid, x_buf);
    try sm.writeFromPackedScalar(gamma_tid, g_buf);
    try sm.writeFromPackedScalar(beta_tid, b_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f32, &[_]usize{ batch, seq, groups, hidden });
    const gamma_in = try g.addInput(.f32, &[_]usize{hidden});
    const beta_in = try g.addInput(.f32, &[_]usize{hidden});

    try g.bindExternal(x_in, @intCast(x_tid));
    try g.bindExternal(gamma_in, @intCast(gamma_tid));
    try g.bindExternal(beta_in, @intCast(beta_tid));

    const y = try g.addRMSNorm(x_in, gamma_in, beta_in, eps, &[_]usize{hidden});
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    const policy: plan_mod.TilePolicy = .{ .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    try backend.executeProgram(&prog, sm.tensorStore());

    const y_buf: []u8 = try allocator.alloc(u8, x_bytes_len);
    defer allocator.free(y_buf);
    try sm.readToPackedScalar(prog.outputs[0], y_buf);
    const y_vals: []align(1) f32 = asF32Slice(y_buf);

    var i: usize = 0;
    while (i < elems) : (i += 1) {
        try std.testing.expectApproxEqAbs(y_ref[i], y_vals[i], 5e-4);
    }
}

const PackedLayout2 = struct {
    shape_mem: [2]usize,
    strides_mem: [2]isize,

    pub fn init(shape0: usize, shape1: usize, elem_bytes: usize) PackedLayout2 {
        return .{
            .shape_mem = .{ shape0, shape1 },
            .strides_mem = .{ @intCast(shape1 * elem_bytes), @intCast(elem_bytes) },
        };
    }

    pub fn layout(self: *const PackedLayout2) types.Layout {
        return .{
            .rank = 2,
            .shape = self.shape_mem[0..2],
            .strides_bytes = self.strides_mem[0..2],
        };
    }
};

test "cpu backend: compile+run covers matmul/broadcast/elemwise/relu/copy/reduce" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const m: usize = 2;
    const k: usize = 3;
    const n: usize = 4;

    // Packed inputs.
    const a_bytes_len: usize = m * k * 4;
    const b_bytes_len: usize = k * n * 4;
    const bias_bytes_len: usize = n * 4;

    const a_buf: []u8 = try allocator.alloc(u8, a_bytes_len);
    defer allocator.free(a_buf);
    const b_buf: []u8 = try allocator.alloc(u8, b_bytes_len);
    defer allocator.free(b_buf);
    const bias_buf: []u8 = try allocator.alloc(u8, bias_bytes_len);
    defer allocator.free(bias_buf);

    const a_vals: []align(1) f32 = asF32Slice(a_buf);
    const b_vals: []align(1) f32 = asF32Slice(b_buf);
    const bias_vals: []align(1) f32 = asF32Slice(bias_buf);

    for (0..m * k) |i| a_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 2)) * 0.5;
    for (0..k * n) |i| b_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 7))) - 3)) * 0.25;
    for (0..n) |i| bias_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 1)) * 0.1;

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: Backend = cpu.backend();

    // Reference (all packed).
    const c_bytes_len: usize = m * n * 4;
    const d_bytes_len: usize = c_bytes_len;
    const e_bytes_len: usize = c_bytes_len;
    const f_bytes_len: usize = c_bytes_len;
    const g_bytes_len: usize = c_bytes_len;
    const out_bytes_len: usize = 4;

    const c_ref_buf: []u8 = try allocator.alloc(u8, c_bytes_len);
    defer allocator.free(c_ref_buf);
    const d_ref_buf: []u8 = try allocator.alloc(u8, d_bytes_len);
    defer allocator.free(d_ref_buf);
    const e_ref_buf: []u8 = try allocator.alloc(u8, e_bytes_len);
    defer allocator.free(e_ref_buf);
    const f_ref_buf: []u8 = try allocator.alloc(u8, f_bytes_len);
    defer allocator.free(f_ref_buf);
    const g_ref_buf: []u8 = try allocator.alloc(u8, g_bytes_len);
    defer allocator.free(g_ref_buf);
    const out_ref_buf: []u8 = try allocator.alloc(u8, out_bytes_len);
    defer allocator.free(out_ref_buf);

    // Naive reference on packed inputs (computed in plain Zig).
    const c_ref: []align(1) f32 = asF32Slice(c_ref_buf);
    const d_ref: []align(1) f32 = asF32Slice(d_ref_buf);
    const e_ref: []align(1) f32 = asF32Slice(e_ref_buf);
    const f_ref: []align(1) f32 = asF32Slice(f_ref_buf);
    const g_ref: []align(1) f32 = asF32Slice(g_ref_buf);
    const out_ref: []align(1) f32 = asF32Slice(out_ref_buf);

    // c = a @ b
    for (0..m) |i| {
        for (0..n) |j| {
            var acc: f32 = 0.0;
            for (0..k) |kk| {
                acc += a_vals[i * k + kk] * b_vals[kk * n + j];
            }
            c_ref[i * n + j] = acc;
        }
    }
    // d = c + bias (broadcast over last dim)
    for (0..m) |i| {
        for (0..n) |j| {
            d_ref[i * n + j] = c_ref[i * n + j] + bias_vals[j];
        }
    }
    // e = relu(d)
    for (0..m * n) |idx| {
        const v: f32 = d_ref[idx];
        e_ref[idx] = if (v > 0) v else 0;
    }
    // f = copy(e)
    @memcpy(f_ref[0 .. m * n], e_ref[0 .. m * n]);
    // g = f * f
    for (0..m * n) |idx| {
        const v: f32 = f_ref[idx];
        g_ref[idx] = v * v;
    }
    // out = mean(g)
    var sum: f32 = 0.0;
    for (0..m * n) |idx| sum += g_ref[idx];
    out_ref[0] = sum / @as(f32, @floatFromInt(m * n));

    // Build tiled inputs and graph.
    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    // Intentionally use non-matmul-friendly tiling to exercise ReTileCopyScalar.
    const a_tid = try sm.createTiledTensor(.f32, &[_]usize{ m, k }, &[_]usize{ 1, 2 }, .{ .tile_alignment = 64 });
    const b_tid = try sm.createTiledTensor(.f32, &[_]usize{ k, n }, &[_]usize{ 2, 3 }, .{ .tile_alignment = 64 });
    const bias_tid = try sm.createTiledTensor(.f32, &[_]usize{n}, &[_]usize{3}, .{ .tile_alignment = 64 });

    try sm.writeFromPackedScalar(a_tid, a_buf);
    try sm.writeFromPackedScalar(b_tid, b_buf);
    try sm.writeFromPackedScalar(bias_tid, bias_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const a_in = try g.addInput(.f32, &[_]usize{ m, k });
    const b_in = try g.addInput(.f32, &[_]usize{ k, n });
    const bias_in = try g.addInput(.f32, &[_]usize{n});
    try g.bindExternal(a_in, @intCast(a_tid));
    try g.bindExternal(b_in, @intCast(b_tid));
    try g.bindExternal(bias_in, @intCast(bias_tid));

    const c = try g.addMatMul(a_in, b_in, 1.0, 0.0);
    const d = try g.addElemwiseBinary(.add, c, bias_in);
    const e = try g.addRelu(d);
    const f = try g.addCopy(e);
    const gg = try g.addElemwiseBinary(.mul, f, f);
    const out = try g.addReduce(.mean, gg);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 2, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    try backend.executeProgram(&prog, sm.tensorStore());

    // Read output.
    var out_buf: [4]u8 = .{ 0, 0, 0, 0 };
    try sm.readToPackedScalar(prog.outputs[0], out_buf[0..4]);
    const out_val: f32 = @as(*align(1) const f32, @ptrCast(out_buf[0..4].ptr)).*;
    try std.testing.expect(std.math.isFinite(out_val));
    // Keep this reasonably tight, but allow minor FP differences due to kernel accumulation order.
    try std.testing.expectApproxEqAbs(out_ref[0], out_val, 1e-5);
}

test "cpu backend: batched matmul rank-3 matches reference (f32)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const batch: usize = 2;
    const m: usize = 2;
    const k: usize = 3;
    const n: usize = 4;

    const a_bytes_len: usize = batch * m * k * 4;
    const b_bytes_len: usize = batch * k * n * 4;
    const c_bytes_len: usize = batch * m * n * 4;

    const a_buf: []u8 = try allocator.alloc(u8, a_bytes_len);
    defer allocator.free(a_buf);
    const b_buf: []u8 = try allocator.alloc(u8, b_bytes_len);
    defer allocator.free(b_buf);

    const a_vals: []align(1) f32 = asF32Slice(a_buf);
    const b_vals: []align(1) f32 = asF32Slice(b_buf);

    for (0..batch * m * k) |i| a_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 3)) * 0.2;
    for (0..batch * k * n) |i| b_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 11))) - 5)) * 0.15;

    // Reference.
    const c_ref_buf: []u8 = try allocator.alloc(u8, c_bytes_len);
    defer allocator.free(c_ref_buf);
    const c_ref: []align(1) f32 = asF32Slice(c_ref_buf);
    @memset(c_ref, 0.0);

    var b0: usize = 0;
    while (b0 < batch) : (b0 += 1) {
        var i: usize = 0;
        while (i < m) : (i += 1) {
            var j: usize = 0;
            while (j < n) : (j += 1) {
                var acc: f32 = 0.0;
                var kk: usize = 0;
                while (kk < k) : (kk += 1) {
                    const a_idx: usize = ((b0 * m + i) * k) + kk;
                    const b_idx: usize = ((b0 * k + kk) * n) + j;
                    acc += a_vals[a_idx] * b_vals[b_idx];
                }
                const c_idx: usize = ((b0 * m + i) * n) + j;
                c_ref[c_idx] = acc;
            }
        }
    }

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: Backend = cpu.backend();

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 2, .tile_alignment = 64 };
    const tiles = plan_mod.chooseMatMulTiles(policy, m, n, k, .f32);

    const a_tid = try sm.createTiledTensor(.f32, &[_]usize{ batch, m, k }, &[_]usize{ 1, tiles.tm, tiles.tk }, .{ .tile_alignment = 64 });
    const b_tid = try sm.createTiledTensor(.f32, &[_]usize{ batch, k, n }, &[_]usize{ 1, tiles.tk, tiles.tn }, .{ .tile_alignment = 64 });

    try sm.writeFromPackedScalar(a_tid, a_buf);
    try sm.writeFromPackedScalar(b_tid, b_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const a_in = try g.addInput(.f32, &[_]usize{ batch, m, k });
    const b_in = try g.addInput(.f32, &[_]usize{ batch, k, n });
    try g.bindExternal(a_in, @intCast(a_tid));
    try g.bindExternal(b_in, @intCast(b_tid));

    const c = try g.addMatMul(a_in, b_in, 1.0, 0.0);
    try g.setOutputs(&[_]graph_mod.ValueId{c});

    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    try backend.executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, c_bytes_len);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf[0..c_bytes_len]);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    var max_abs: f32 = 0.0;
    for (out_vals, c_ref) |g0, r0| max_abs = @max(max_abs, @abs(g0 - r0));
    try std.testing.expect(max_abs <= 1e-5);
}

test "cpu backend: batched matmul broadcast B rank-3 matches reference (f32)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const batch: usize = 3;
    const m: usize = 2;
    const k: usize = 3;
    const n: usize = 4;

    const a_bytes_len: usize = batch * m * k * 4;
    const b_bytes_len: usize = 1 * k * n * 4;
    const c_bytes_len: usize = batch * m * n * 4;

    const a_buf: []u8 = try allocator.alloc(u8, a_bytes_len);
    defer allocator.free(a_buf);
    const b_buf: []u8 = try allocator.alloc(u8, b_bytes_len);
    defer allocator.free(b_buf);

    const a_vals: []align(1) f32 = asF32Slice(a_buf);
    const b_vals: []align(1) f32 = asF32Slice(b_buf);

    for (0..batch * m * k) |i| a_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 4)) * 0.1;
    for (0..k * n) |i| b_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 9))) - 4)) * 0.2;

    const c_ref_buf: []u8 = try allocator.alloc(u8, c_bytes_len);
    defer allocator.free(c_ref_buf);
    const c_ref: []align(1) f32 = asF32Slice(c_ref_buf);
    @memset(c_ref, 0.0);

    var b0: usize = 0;
    while (b0 < batch) : (b0 += 1) {
        var i: usize = 0;
        while (i < m) : (i += 1) {
            var j: usize = 0;
            while (j < n) : (j += 1) {
                var acc: f32 = 0.0;
                var kk: usize = 0;
                while (kk < k) : (kk += 1) {
                    const a_idx: usize = ((b0 * m + i) * k) + kk;
                    const b_idx: usize = (kk * n) + j;
                    acc += a_vals[a_idx] * b_vals[b_idx];
                }
                const c_idx: usize = ((b0 * m + i) * n) + j;
                c_ref[c_idx] = acc;
            }
        }
    }

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: Backend = cpu.backend();

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 2, .tile_alignment = 64 };
    const tiles = plan_mod.chooseMatMulTiles(policy, m, n, k, .f32);

    const a_tid = try sm.createTiledTensor(.f32, &[_]usize{ batch, m, k }, &[_]usize{ 1, tiles.tm, tiles.tk }, .{ .tile_alignment = 64 });
    const b_tid = try sm.createTiledTensor(.f32, &[_]usize{ 1, k, n }, &[_]usize{ 1, tiles.tk, tiles.tn }, .{ .tile_alignment = 64 });

    try sm.writeFromPackedScalar(a_tid, a_buf);
    try sm.writeFromPackedScalar(b_tid, b_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const a_in = try g.addInput(.f32, &[_]usize{ batch, m, k });
    const b_in = try g.addInput(.f32, &[_]usize{ 1, k, n });
    try g.bindExternal(a_in, @intCast(a_tid));
    try g.bindExternal(b_in, @intCast(b_tid));

    const c = try g.addMatMul(a_in, b_in, 1.0, 0.0);
    try g.setOutputs(&[_]graph_mod.ValueId{c});

    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    try backend.executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, c_bytes_len);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf[0..c_bytes_len]);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    var max_abs: f32 = 0.0;
    for (out_vals, c_ref) |g0, r0| max_abs = @max(max_abs, @abs(g0 - r0));
    try std.testing.expect(max_abs <= 1e-5);
}

test "cpu backend: batched matmul broadcast A rank-3 matches reference (f32)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const batch: usize = 3;
    const m: usize = 2;
    const k: usize = 3;
    const n: usize = 4;

    const a_bytes_len: usize = 1 * m * k * 4;
    const b_bytes_len: usize = batch * k * n * 4;
    const c_bytes_len: usize = batch * m * n * 4;

    const a_buf: []u8 = try allocator.alloc(u8, a_bytes_len);
    defer allocator.free(a_buf);
    const b_buf: []u8 = try allocator.alloc(u8, b_bytes_len);
    defer allocator.free(b_buf);

    const a_vals: []align(1) f32 = asF32Slice(a_buf);
    const b_vals: []align(1) f32 = asF32Slice(b_buf);

    for (0..m * k) |i| a_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 4)) * 0.1;
    for (0..batch * k * n) |i| b_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 9))) - 4)) * 0.2;

    const c_ref_buf: []u8 = try allocator.alloc(u8, c_bytes_len);
    defer allocator.free(c_ref_buf);
    const c_ref: []align(1) f32 = asF32Slice(c_ref_buf);
    @memset(c_ref, 0.0);

    var b0: usize = 0;
    while (b0 < batch) : (b0 += 1) {
        var i: usize = 0;
        while (i < m) : (i += 1) {
            var j: usize = 0;
            while (j < n) : (j += 1) {
                var acc: f32 = 0.0;
                var kk: usize = 0;
                while (kk < k) : (kk += 1) {
                    const a_idx: usize = (i * k) + kk;
                    const b_idx: usize = ((b0 * k + kk) * n) + j;
                    acc += a_vals[a_idx] * b_vals[b_idx];
                }
                const c_idx: usize = ((b0 * m + i) * n) + j;
                c_ref[c_idx] = acc;
            }
        }
    }

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: Backend = cpu.backend();

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 2, .tile_alignment = 64 };
    const tiles = plan_mod.chooseMatMulTiles(policy, m, n, k, .f32);

    const a_tid = try sm.createTiledTensor(.f32, &[_]usize{ 1, m, k }, &[_]usize{ 1, tiles.tm, tiles.tk }, .{ .tile_alignment = 64 });
    const b_tid = try sm.createTiledTensor(.f32, &[_]usize{ batch, k, n }, &[_]usize{ 1, tiles.tk, tiles.tn }, .{ .tile_alignment = 64 });

    try sm.writeFromPackedScalar(a_tid, a_buf);
    try sm.writeFromPackedScalar(b_tid, b_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const a_in = try g.addInput(.f32, &[_]usize{ 1, m, k });
    const b_in = try g.addInput(.f32, &[_]usize{ batch, k, n });
    try g.bindExternal(a_in, @intCast(a_tid));
    try g.bindExternal(b_in, @intCast(b_tid));

    const c = try g.addMatMul(a_in, b_in, 1.0, 0.0);
    try g.setOutputs(&[_]graph_mod.ValueId{c});

    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    try backend.executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, c_bytes_len);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf[0..c_bytes_len]);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    var max_abs: f32 = 0.0;
    for (out_vals, c_ref) |g0, r0| max_abs = @max(max_abs, @abs(g0 - r0));
    try std.testing.expect(max_abs <= 1e-5);
}

test "cpu backend: batch retile guard accepts small, rejects large" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const batch: usize = 4;
    const m: usize = 2;
    const k: usize = 3;
    const n: usize = 2;

    const a_bytes_len: usize = batch * m * k * 4;
    const b_bytes_len: usize = batch * k * n * 4;
    const c_bytes_len: usize = batch * m * n * 4;

    const a_buf: []u8 = try allocator.alloc(u8, a_bytes_len);
    defer allocator.free(a_buf);
    const b_buf: []u8 = try allocator.alloc(u8, b_bytes_len);
    defer allocator.free(b_buf);

    const a_vals: []align(1) f32 = asF32Slice(a_buf);
    const b_vals: []align(1) f32 = asF32Slice(b_buf);
    for (0..batch * m * k) |i| a_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 3)) * 0.2;
    for (0..batch * k * n) |i| b_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 7))) - 3)) * 0.25;

    // Reference.
    const c_ref_buf: []u8 = try allocator.alloc(u8, c_bytes_len);
    defer allocator.free(c_ref_buf);
    const c_ref: []align(1) f32 = asF32Slice(c_ref_buf);
    @memset(c_ref, 0.0);

    var b0: usize = 0;
    while (b0 < batch) : (b0 += 1) {
        var i: usize = 0;
        while (i < m) : (i += 1) {
            var j: usize = 0;
            while (j < n) : (j += 1) {
                var acc: f32 = 0.0;
                var kk: usize = 0;
                while (kk < k) : (kk += 1) {
                    const a_idx: usize = ((b0 * m + i) * k) + kk;
                    const b_idx: usize = ((b0 * k + kk) * n) + j;
                    acc += a_vals[a_idx] * b_vals[b_idx];
                }
                const c_idx: usize = ((b0 * m + i) * n) + j;
                c_ref[c_idx] = acc;
            }
        }
    }

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const policy_ok: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 2, .tile_alignment = 64, .batch_retile_max_tiles = 2 };
    const tiles = plan_mod.chooseMatMulTiles(policy_ok, m, n, k, .f32);

    // Inputs are tiled with batch tile size 2, forcing retile to batch tile size 1.
    const a_tid = try sm.createTiledTensor(.f32, &[_]usize{ batch, m, k }, &[_]usize{ 2, tiles.tm, tiles.tk }, .{ .tile_alignment = 64 });
    const b_tid = try sm.createTiledTensor(.f32, &[_]usize{ batch, k, n }, &[_]usize{ 2, tiles.tk, tiles.tn }, .{ .tile_alignment = 64 });

    try sm.writeFromPackedScalar(a_tid, a_buf);
    try sm.writeFromPackedScalar(b_tid, b_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const a_in = try g.addInput(.f32, &[_]usize{ batch, m, k });
    const b_in = try g.addInput(.f32, &[_]usize{ batch, k, n });
    try g.bindExternal(a_in, @intCast(a_tid));
    try g.bindExternal(b_in, @intCast(b_tid));

    const c = try g.addMatMul(a_in, b_in, 1.0, 0.0);
    try g.setOutputs(&[_]graph_mod.ValueId{c});

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: Backend = cpu.backend();

    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy_ok));
    defer prog.deinit();

    try backend.executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, c_bytes_len);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf[0..c_bytes_len]);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    var max_abs: f32 = 0.0;
    for (out_vals, c_ref) |g0, r0| max_abs = @max(max_abs, @abs(g0 - r0));
    try std.testing.expect(max_abs <= 1e-5);

    // Reject when guard is tighter than required batch tiles.
    const policy_bad: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 2, .tile_alignment = 64, .batch_retile_max_tiles = 1 };
    try std.testing.expectError(program.CompileError.InvalidArgument, program.compileGraph(allocator, &g, &sm, .cpu(policy_bad)));
}

test "cpu backend: unary ops match reference (f32)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    // Moderate-sized 1D input to exercise vector loops + tails.
    const n: usize = 257;
    const x_bytes_len: usize = n * 4;
    const x_buf: []u8 = try allocator.alloc(u8, x_bytes_len);
    defer allocator.free(x_buf);
    const x_vals: []align(1) f32 = asF32Slice(x_buf);

    // Deterministic mix of values within a safe domain for exp/tanh comparisons.
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const t: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 128));
        x_vals[i] = (t / 32.0); // ~[-4,4]
    }

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: Backend = cpu.backend();

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const x_tid = try sm.createTiledTensor(.f32, &[_]usize{n}, &[_]usize{16}, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(x_tid, x_buf);

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 16, .tile_alignment = 64 };

    inline for (.{
        .{ .op = types.UnaryOp.relu, .name = "relu" },
        .{ .op = types.UnaryOp.sigmoid, .name = "sigmoid" },
        .{ .op = types.UnaryOp.tanh, .name = "tanh" },
        .{ .op = types.UnaryOp.silu, .name = "silu" },
        .{ .op = types.UnaryOp.gelu, .name = "gelu" },
        .{ .op = types.UnaryOp.sqrt, .name = "sqrt" },
    }) |case| {
        var g = graph_mod.Graph.init(allocator);
        defer g.deinit();

        const x_in = try g.addInput(.f32, &[_]usize{n});
        try g.bindExternal(x_in, @intCast(x_tid));
        const y = try g.addUnary(case.op, x_in);
        try g.setOutputs(&[_]graph_mod.ValueId{y});

        var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
        defer prog.deinit();
        try backend.executeProgram(&prog, sm.tensorStore());

        // Read output (packed).
        var out_buf: []u8 = try allocator.alloc(u8, x_bytes_len);
        defer allocator.free(out_buf);
        try sm.readToPackedScalar(prog.outputs[0], out_buf[0..x_bytes_len]);
        const out_vals: []align(1) f32 = asF32Slice(out_buf);

        // Reference and comparisons.
        i = 0;
        while (i < n) : (i += 1) {
            const x: f32 = x_vals[i];
            const got: f32 = out_vals[i];
            if (!(case.op == .sqrt and x < 0.0)) {
                try std.testing.expect(std.math.isFinite(got));
            }

            const xf: f64 = @floatCast(x);
            const ref64: f64 = switch (case.op) {
                .relu => if (x > 0.0) @as(f64, xf) else 0.0,
                .sigmoid => 1.0 / (1.0 + std.math.exp(-xf)),
                .tanh => std.math.tanh(xf),
                .silu => xf * (1.0 / (1.0 + std.math.exp(-xf))),
                // Compare to the common tanh-based GELU approximation.
                .gelu => blk: {
                    const k0: f64 = std.math.sqrt(2.0 / std.math.pi);
                    const inner: f64 = k0 * (xf + 0.044715 * xf * xf * xf);
                    break :blk 0.5 * xf * (1.0 + std.math.tanh(inner));
                },
                .sqrt => std.math.sqrt(xf),
                .log => @log(xf), // not in the iterated cases above; arm kept for exhaustiveness
            };
            const ref: f32 = @as(f32, @floatCast(ref64));

            if (case.op == .sqrt and x < 0.0) {
                try std.testing.expect(std.math.isNan(got));
                continue;
            }

            const abs_err: f32 = @abs(got - ref);
            const tol: f32 = switch (case.op) {
                .relu => 0.0,
                .sigmoid => 3e-2,
                .tanh => 3e-2,
                .silu => 5e-2,
                .gelu => 6e-2,
                .sqrt => 1e-6,
                .log => 1e-6, // not in the iterated cases above; arm kept for exhaustiveness
            };

            // If this fails, include op name for quick debugging.
            if (!(abs_err <= tol)) {
                std.debug.print("unary {s} mismatch at i={} x={} got={} ref={} abs_err={} tol={}\n", .{ case.name, i, x, got, ref, abs_err, tol });
                return error.TestExpectedEqual;
            }
        }
    }
}

fn softmaxRefRowF32(out: []f32, x: []align(1) const f32) void {
    std.debug.assert(out.len == x.len);
    var maxv: f64 = -std.math.inf(f64);
    for (x) |v| maxv = @max(maxv, @as(f64, @floatCast(v)));
    var sum: f64 = 0.0;
    for (x, 0..) |v, i| {
        const e: f64 = std.math.exp(@as(f64, @floatCast(v)) - maxv);
        out[i] = @floatCast(e);
        sum += e;
    }
    const inv: f64 = 1.0 / sum;
    for (out) |*v| v.* = @floatCast(@as(f64, @floatCast(v.*)) * inv);
}

fn softmaxRefColF32(out: []f32, x: []align(1) const f32, m: usize, n: usize) void {
    std.debug.assert(out.len == x.len);
    var c: usize = 0;
    while (c < n) : (c += 1) {
        var maxv: f64 = -std.math.inf(f64);
        var r: usize = 0;
        while (r < m) : (r += 1) {
            const v: f32 = x[r * n + c];
            maxv = @max(maxv, @as(f64, @floatCast(v)));
        }
        var sum: f64 = 0.0;
        r = 0;
        while (r < m) : (r += 1) {
            const v: f32 = x[r * n + c];
            const e: f64 = std.math.exp(@as(f64, @floatCast(v)) - maxv);
            out[r * n + c] = @floatCast(e);
            sum += e;
        }
        const inv: f64 = 1.0 / sum;
        r = 0;
        while (r < m) : (r += 1) {
            const idx: usize = r * n + c;
            out[idx] = @floatCast(@as(f64, @floatCast(out[idx])) * inv);
        }
    }
}

fn layernormRefRowF32(out: []f32, x: []align(1) const f32, gamma: []align(1) const f32, beta: []align(1) const f32, eps: f32) void {
    std.debug.assert(out.len == x.len);
    std.debug.assert(gamma.len == x.len);
    std.debug.assert(beta.len == x.len);
    var sum: f64 = 0.0;
    var sumsq: f64 = 0.0;
    for (x) |v| {
        const vf: f64 = @floatCast(v);
        sum += vf;
        sumsq += vf * vf;
    }
    const n: f64 = @floatFromInt(x.len);
    const mean: f64 = sum / n;
    const msq: f64 = sumsq / n;
    const var0: f64 = @max(0.0, msq - mean * mean);
    const inv: f64 = 1.0 / std.math.sqrt(var0 + @as(f64, @floatCast(eps)));
    for (x, 0..) |v, i| {
        const xf: f64 = @floatCast(v);
        const y: f64 = ((xf - mean) * inv) * @as(f64, @floatCast(gamma[i])) + @as(f64, @floatCast(beta[i]));
        out[i] = @floatCast(y);
    }
}

fn rmsnormRefRowF32(out: []f32, x: []align(1) const f32, gamma: []align(1) const f32, beta: []align(1) const f32, eps: f32) void {
    std.debug.assert(out.len == x.len);
    std.debug.assert(gamma.len == x.len);
    std.debug.assert(beta.len == x.len);
    var sumsq: f64 = 0.0;
    for (x) |v| {
        const vf: f64 = @floatCast(v);
        sumsq += vf * vf;
    }
    const n: f64 = @floatFromInt(x.len);
    const msq: f64 = sumsq / n;
    const inv: f64 = 1.0 / std.math.sqrt(msq + @as(f64, @floatCast(eps)));
    for (x, 0..) |v, i| {
        const xf: f64 = @floatCast(v);
        const y: f64 = (xf * inv) * @as(f64, @floatCast(gamma[i])) + @as(f64, @floatCast(beta[i]));
        out[i] = @floatCast(y);
    }
}

fn attentionRefF32(
    allocator: std.mem.Allocator,
    out: []f32,
    q: []align(1) const f32,
    k: []align(1) const f32,
    v: []align(1) const f32,
    m: usize,
    n: usize,
    dk: usize,
    dv: usize,
    scale: f32,
    causal: bool,
) void {
    std.debug.assert(out.len == m * dv);
    std.debug.assert(q.len == m * dk);
    std.debug.assert(k.len == n * dk);
    std.debug.assert(v.len == n * dv);

    var scores: []f64 = allocator.alloc(f64, n) catch unreachable;
    defer allocator.free(scores);
    var probs: []f64 = allocator.alloc(f64, n) catch unreachable;
    defer allocator.free(probs);

    for (0..m) |r| {
        // scores[c] = scale * dot(q[r], k[c]) with optional causal masking.
        var maxv: f64 = -std.math.inf(f64);
        for (0..n) |c| {
            var s: f64 = 0.0;
            if (causal and c > r) {
                s = -std.math.inf(f64);
            } else {
                const q_off: usize = r * dk;
                const k_off: usize = c * dk;
                for (0..dk) |kk| {
                    s += @as(f64, @floatCast(q[q_off + kk])) * @as(f64, @floatCast(k[k_off + kk]));
                }
                s *= @as(f64, @floatCast(scale));
            }
            scores[c] = s;
            maxv = @max(maxv, s);
        }

        var sum: f64 = 0.0;
        for (0..n) |c| {
            const s: f64 = scores[c];
            const e: f64 = if (std.math.isFinite(s)) std.math.exp(s - maxv) else 0.0;
            probs[c] = e;
            sum += e;
        }
        const inv: f64 = 1.0 / sum;
        for (0..n) |c| probs[c] *= inv;

        for (0..dv) |j| {
            var acc: f64 = 0.0;
            for (0..n) |c| {
                acc += probs[c] * @as(f64, @floatCast(v[c * dv + j]));
            }
            out[r * dv + j] = @floatCast(acc);
        }
    }
}

fn attentionRefBatchedF32(
    allocator: std.mem.Allocator,
    out: []f32,
    q: []align(1) const f32,
    k: []align(1) const f32,
    v: []align(1) const f32,
    batch: usize,
    m: usize,
    n: usize,
    dk: usize,
    dv: usize,
    scale: f32,
    causal: bool,
) void {
    std.debug.assert(out.len == batch * m * dv);
    std.debug.assert(q.len == batch * m * dk);
    std.debug.assert(k.len == batch * n * dk);
    std.debug.assert(v.len == batch * n * dv);

    var b: usize = 0;
    while (b < batch) : (b += 1) {
        const q_off: usize = b * m * dk;
        const k_off: usize = b * n * dk;
        const v_off: usize = b * n * dv;
        const out_off: usize = b * m * dv;
        attentionRefF32(
            allocator,
            out[out_off .. out_off + (m * dv)],
            q[q_off .. q_off + (m * dk)],
            k[k_off .. k_off + (n * dk)],
            v[v_off .. v_off + (n * dv)],
            m,
            n,
            dk,
            dv,
            scale,
            causal,
        );
    }
}

fn attentionRefBatchedMultiHeadF32(
    allocator: std.mem.Allocator,
    out: []f32,
    q: []align(1) const f32,
    k: []align(1) const f32,
    v: []align(1) const f32,
    batch: usize,
    heads: usize,
    m_head: usize,
    n_head: usize,
    dk: usize,
    dv: usize,
    scale: f32,
    causal: bool,
) void {
    std.debug.assert(out.len == batch * heads * m_head * dv);
    std.debug.assert(q.len == batch * heads * m_head * dk);
    std.debug.assert(k.len == batch * heads * n_head * dk);
    std.debug.assert(v.len == batch * heads * n_head * dv);

    var b: usize = 0;
    while (b < batch) : (b += 1) {
        var h: usize = 0;
        while (h < heads) : (h += 1) {
            const q_off: usize = ((b * heads + h) * m_head * dk);
            const k_off: usize = ((b * heads + h) * n_head * dk);
            const v_off: usize = ((b * heads + h) * n_head * dv);
            const out_off: usize = ((b * heads + h) * m_head * dv);
            attentionRefF32(
                allocator,
                out[out_off .. out_off + (m_head * dv)],
                q[q_off .. q_off + (m_head * dk)],
                k[k_off .. k_off + (n_head * dk)],
                v[v_off .. v_off + (n_head * dv)],
                m_head,
                n_head,
                dk,
                dv,
                scale,
                causal,
            );
        }
    }
}

test "cpu backend: softmax rank-1 matches reference (f32)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const n: usize = 257;
    const x_bytes_len: usize = n * 4;
    const x_buf: []u8 = try allocator.alloc(u8, x_bytes_len);
    defer allocator.free(x_buf);
    const x_vals: []align(1) f32 = asF32Slice(x_buf);

    // Deterministic values in a safe range.
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const t: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 128));
        x_vals[i] = (t / 32.0); // ~[-4,4]
    }

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: Backend = cpu.backend();

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    // Intentionally awkward tiling to exercise tile loops.
    const x_tid = try sm.createTiledTensor(.f32, &[_]usize{n}, &[_]usize{17}, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(x_tid, x_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f32, &[_]usize{n});
    try g.bindExternal(x_in, @intCast(x_tid));
    const y = try g.addSoftmax(x_in, -1);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 32, .base_1d = 128, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();
    try backend.executeProgram(&prog, sm.tensorStore());

    // Read output and compare.
    const out_tid = prog.outputs[0];
    const out_buf: []u8 = try allocator.alloc(u8, x_bytes_len);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(out_tid, out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    const ref: []f32 = try allocator.alloc(f32, n);
    defer allocator.free(ref);
    softmaxRefRowF32(ref, x_vals);

    var sum: f32 = 0.0;
    for (out_vals) |v| {
        try std.testing.expect(std.math.isFinite(v));
        try std.testing.expect(v >= 0.0 and v <= 1.0);
        sum += v;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 2e-3);

    // Compare loosely since the kernel uses fast exp.
    var max_abs: f32 = 0.0;
    for (out_vals, ref) |g0, r0| max_abs = @max(max_abs, @abs(g0 - r0));
    try std.testing.expect(max_abs <= 3e-2);
}

test "cpu backend: softmax rank-2 matches reference (f32)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const m: usize = 5;
    const n: usize = 9;
    const x_bytes_len: usize = m * n * 4;

    const x_buf: []u8 = try allocator.alloc(u8, x_bytes_len);
    defer allocator.free(x_buf);
    const x_vals: []align(1) f32 = asF32Slice(x_buf);

    // Deterministic values in a safe range.
    for (0..m) |r| {
        for (0..n) |c| {
            const idx: usize = r * n + c;
            const t: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(idx)) - @as(i32, @intCast((m * n) / 2))));
            // Add a small row-dependent offset so row maxima differ.
            x_vals[idx] = (t / 32.0) + (@as(f32, @floatFromInt(@as(i32, @intCast(r)) - 2)) * 0.05);
        }
    }

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: Backend = cpu.backend();

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    // Intentionally awkward tiling to exercise tile loops + possible retile.
    const x_tid = try sm.createTiledTensor(.f32, &[_]usize{ m, n }, &[_]usize{ 2, 5 }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(x_tid, x_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f32, &[_]usize{ m, n });
    try g.bindExternal(x_in, @intCast(x_tid));
    const y = try g.addSoftmax(x_in, -1);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    // Force softmax tiling to span multiple tiles in both dims.
    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 4, .base_1d = 3, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();
    try backend.executeProgram(&prog, sm.tensorStore());

    // Read output.
    const out_tid = prog.outputs[0];
    const out_buf: []u8 = try allocator.alloc(u8, x_bytes_len);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(out_tid, out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    // Reference and comparisons per row.
    const ref_row: []f32 = try allocator.alloc(f32, n);
    defer allocator.free(ref_row);

    for (0..m) |r| {
        const x_row: []align(1) const f32 = x_vals[(r * n)..(r * n + n)];
        softmaxRefRowF32(ref_row, x_row);

        var sum: f32 = 0.0;
        var max_abs: f32 = 0.0;
        for (0..n) |c| {
            const idx: usize = r * n + c;
            const got: f32 = out_vals[idx];
            const ref: f32 = ref_row[c];
            try std.testing.expect(std.math.isFinite(got));
            try std.testing.expect(got >= 0.0 and got <= 1.0);
            sum += got;
            max_abs = @max(max_abs, @abs(got - ref));
        }

        // Fast exp approx: keep sum fairly tight, value-wise compare loosely.
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 3e-3);
        try std.testing.expect(max_abs <= 3e-2);
    }
}

test "cpu backend: softmax rank-2 axis-0 matches reference (f32)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const m: usize = 6;
    const n: usize = 8;
    const x_bytes_len: usize = m * n * 4;

    const x_buf: []u8 = try allocator.alloc(u8, x_bytes_len);
    defer allocator.free(x_buf);
    const x_vals: []align(1) f32 = asF32Slice(x_buf);

    for (0..m) |r| {
        for (0..n) |c| {
            const idx: usize = r * n + c;
            const t: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(idx)) - @as(i32, @intCast((m * n) / 2))));
            x_vals[idx] = (t / 28.0) + (@as(f32, @floatFromInt(@as(i32, @intCast(c)) - 3)) * 0.04);
        }
    }

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: Backend = cpu.backend();

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const x_tid = try sm.createTiledTensor(.f32, &[_]usize{ m, n }, &[_]usize{ 2, 5 }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(x_tid, x_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f32, &[_]usize{ m, n });
    try g.bindExternal(x_in, @intCast(x_tid));
    const y = try g.addSoftmax(x_in, 0);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 4, .base_1d = 3, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();
    try backend.executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, x_bytes_len);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    const ref: []f32 = try allocator.alloc(f32, m * n);
    defer allocator.free(ref);
    softmaxRefColF32(ref, x_vals, m, n);

    var max_abs: f32 = 0.0;
    var c: usize = 0;
    while (c < n) : (c += 1) {
        var sum: f32 = 0.0;
        var r: usize = 0;
        while (r < m) : (r += 1) {
            const idx: usize = r * n + c;
            const got: f32 = out_vals[idx];
            const r0: f32 = ref[idx];
            try std.testing.expect(std.math.isFinite(got));
            sum += got;
            max_abs = @max(max_abs, @abs(got - r0));
        }
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 4e-3);
    }
    try std.testing.expect(max_abs <= 4e-2);
}

test "cpu backend: softmax rank-3 axis-last matches reference (f32)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const b: usize = 2;
    const m: usize = 3;
    const n: usize = 7;
    const x_bytes_len: usize = b * m * n * 4;

    const x_buf: []u8 = try allocator.alloc(u8, x_bytes_len);
    defer allocator.free(x_buf);
    const x_vals: []align(1) f32 = asF32Slice(x_buf);

    for (0..b) |bb| {
        for (0..m) |r| {
            for (0..n) |c| {
                const idx: usize = (bb * m + r) * n + c;
                const t: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(idx)) - 12));
                x_vals[idx] = (t / 32.0) + (@as(f32, @floatFromInt(@as(i32, @intCast(bb)) - 1)) * 0.03);
            }
        }
    }

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: Backend = cpu.backend();

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const x_tid = try sm.createTiledTensor(.f32, &[_]usize{ b, m, n }, &[_]usize{ 1, 2, 4 }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(x_tid, x_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f32, &[_]usize{ b, m, n });
    try g.bindExternal(x_in, @intCast(x_tid));
    const y = try g.addSoftmax(x_in, -1);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 4, .base_1d = 3, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();
    try backend.executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, x_bytes_len);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    const ref_row: []f32 = try allocator.alloc(f32, n);
    defer allocator.free(ref_row);

    for (0..b) |bb| {
        for (0..m) |r| {
            const base: usize = (bb * m + r) * n;
            const x_row: []align(1) const f32 = x_vals[base .. base + n];
            softmaxRefRowF32(ref_row, x_row);

            var sum: f32 = 0.0;
            var max_abs: f32 = 0.0;
            for (0..n) |c| {
                const got: f32 = out_vals[base + c];
                const ref: f32 = ref_row[c];
                try std.testing.expect(std.math.isFinite(got));
                sum += got;
                max_abs = @max(max_abs, @abs(got - ref));
            }
            try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 4e-3);
            try std.testing.expect(max_abs <= 4e-2);
        }
    }
}

test "cpu backend: layernorm and rmsnorm rank-2 match reference (f32)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const m: usize = 6;
    const n: usize = 17;
    const eps: f32 = 1e-5;

    const x_bytes_len: usize = m * n * 4;
    const v_bytes_len: usize = n * 4;

    const x_buf: []u8 = try allocator.alloc(u8, x_bytes_len);
    defer allocator.free(x_buf);
    const g_buf: []u8 = try allocator.alloc(u8, v_bytes_len);
    defer allocator.free(g_buf);
    const b_buf: []u8 = try allocator.alloc(u8, v_bytes_len);
    defer allocator.free(b_buf);

    const x_vals: []align(1) f32 = asF32Slice(x_buf);
    const g_vals: []align(1) f32 = asF32Slice(g_buf);
    const b_vals: []align(1) f32 = asF32Slice(b_buf);

    // Deterministic values; keep magnitudes modest.
    for (0..m * n) |i| {
        const t: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((i % 71))) - 35));
        const r: usize = i / n;
        x_vals[i] = (t / 16.0) + (@as(f32, @floatFromInt(@as(i32, @intCast(r)) - 3)) * 0.05);
    }
    for (0..n) |i| {
        const t: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 8));
        g_vals[i] = 1.0 + (t * 0.01);
        b_vals[i] = (t * 0.02);
    }

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: Backend = cpu.backend();

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    // Intentionally awkward tiling to exercise retile + edge tiles.
    const x_tid = try sm.createTiledTensor(.f32, &[_]usize{ m, n }, &[_]usize{ 2, 5 }, .{ .tile_alignment = 64 });
    const g_tid = try sm.createTiledTensor(.f32, &[_]usize{n}, &[_]usize{6}, .{ .tile_alignment = 64 });
    const b_tid = try sm.createTiledTensor(.f32, &[_]usize{n}, &[_]usize{7}, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(x_tid, x_buf);
    try sm.writeFromPackedScalar(g_tid, g_buf);
    try sm.writeFromPackedScalar(b_tid, b_buf);

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 8, .base_1d = 8, .tile_alignment = 64 };

    // Run both ops.
    inline for (.{
        .{ .is_rms = false, .name = "layernorm" },
        .{ .is_rms = true, .name = "rmsnorm" },
    }) |case| {
        var g = graph_mod.Graph.init(allocator);
        defer g.deinit();

        const x_in = try g.addInput(.f32, &[_]usize{ m, n });
        const gamma_in = try g.addInput(.f32, &[_]usize{n});
        const beta_in = try g.addInput(.f32, &[_]usize{n});
        try g.bindExternal(x_in, @intCast(x_tid));
        try g.bindExternal(gamma_in, @intCast(g_tid));
        try g.bindExternal(beta_in, @intCast(b_tid));

        const norm_shape: [1]usize = .{n};
        const y = if (case.is_rms) try g.addRMSNorm(x_in, gamma_in, beta_in, eps, norm_shape[0..]) else try g.addLayerNorm(x_in, gamma_in, beta_in, eps, norm_shape[0..]);
        try g.setOutputs(&[_]graph_mod.ValueId{y});

        var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
        defer prog.deinit();
        try backend.executeProgram(&prog, sm.tensorStore());

        const out_buf: []u8 = try allocator.alloc(u8, x_bytes_len);
        defer allocator.free(out_buf);
        try sm.readToPackedScalar(prog.outputs[0], out_buf);
        const out_vals: []align(1) f32 = asF32Slice(out_buf);

        const ref_row: []f32 = try allocator.alloc(f32, n);
        defer allocator.free(ref_row);

        for (0..m) |r| {
            const x_row: []align(1) const f32 = x_vals[(r * n)..(r * n + n)];
            if (case.is_rms) {
                rmsnormRefRowF32(ref_row, x_row, g_vals, b_vals, eps);
            } else {
                layernormRefRowF32(ref_row, x_row, g_vals, b_vals, eps);
            }
            var max_abs: f32 = 0.0;
            for (0..n) |c| {
                const got: f32 = out_vals[r * n + c];
                const ref: f32 = ref_row[c];
                try std.testing.expect(std.math.isFinite(got));
                max_abs = @max(max_abs, @abs(got - ref));
            }
            if (!(max_abs <= 2e-4)) {
                std.debug.print("{s} mismatch row={} max_abs={}\n", .{ case.name, r, max_abs });
                return error.TestExpectedEqual;
            }
        }
    }
}

test "cpu backend: layernorm rank-3 normalized-shape matches reference (f32)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const b: usize = 2;
    const m: usize = 4;
    const n: usize = 13;
    const eps: f32 = 1e-5;

    const x_bytes_len: usize = b * m * n * 4;
    const v_bytes_len: usize = n * 4;

    const x_buf: []u8 = try allocator.alloc(u8, x_bytes_len);
    defer allocator.free(x_buf);
    const g_buf: []u8 = try allocator.alloc(u8, v_bytes_len);
    defer allocator.free(g_buf);
    const b_buf: []u8 = try allocator.alloc(u8, v_bytes_len);
    defer allocator.free(b_buf);

    const x_vals: []align(1) f32 = asF32Slice(x_buf);
    const g_vals: []align(1) f32 = asF32Slice(g_buf);
    const b_vals: []align(1) f32 = asF32Slice(b_buf);

    for (0..b * m * n) |i| {
        const t: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((i % 71))) - 35));
        const r: usize = (i / n) % m;
        x_vals[i] = (t / 16.0) + (@as(f32, @floatFromInt(@as(i32, @intCast(r)) - 2)) * 0.07);
    }
    for (0..n) |i| {
        const t: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 6));
        g_vals[i] = 1.0 + (t * 0.01);
        b_vals[i] = (t * 0.02);
    }

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: Backend = cpu.backend();

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const x_tid = try sm.createTiledTensor(.f32, &[_]usize{ b, m, n }, &[_]usize{ 1, 2, 5 }, .{ .tile_alignment = 64 });
    const g_tid = try sm.createTiledTensor(.f32, &[_]usize{n}, &[_]usize{6}, .{ .tile_alignment = 64 });
    const b_tid = try sm.createTiledTensor(.f32, &[_]usize{n}, &[_]usize{7}, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(x_tid, x_buf);
    try sm.writeFromPackedScalar(g_tid, g_buf);
    try sm.writeFromPackedScalar(b_tid, b_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f32, &[_]usize{ b, m, n });
    const gamma_in = try g.addInput(.f32, &[_]usize{n});
    const beta_in = try g.addInput(.f32, &[_]usize{n});
    try g.bindExternal(x_in, @intCast(x_tid));
    try g.bindExternal(gamma_in, @intCast(g_tid));
    try g.bindExternal(beta_in, @intCast(b_tid));

    const norm_shape: [1]usize = .{n};
    const y = try g.addLayerNorm(x_in, gamma_in, beta_in, eps, norm_shape[0..]);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 8, .base_1d = 8, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();
    try backend.executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, x_bytes_len);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    const ref_row: []f32 = try allocator.alloc(f32, n);
    defer allocator.free(ref_row);

    for (0..b) |bb| {
        for (0..m) |r| {
            const base: usize = (bb * m + r) * n;
            const x_row: []align(1) const f32 = x_vals[base .. base + n];
            layernormRefRowF32(ref_row, x_row, g_vals, b_vals, eps);

            var max_abs: f32 = 0.0;
            for (0..n) |c| {
                const got: f32 = out_vals[base + c];
                const ref: f32 = ref_row[c];
                try std.testing.expect(std.math.isFinite(got));
                max_abs = @max(max_abs, @abs(got - ref));
            }
            if (!(max_abs <= 3e-4)) {
                std.debug.print("layernorm rank3 mismatch row={} max_abs={}\n", .{ r, max_abs });
                return error.TestExpectedEqual;
            }
        }
    }
}

// The three tests that used to sit here drove `Attention` (rank-2/3) and
// `MultiHeadAttention` (rank-4 head-major), the pre-cache ops. They are gone; the
// unified `Attention` op is covered by the cached tests below plus this one, which
// pins the no-index path: with the index operands omitted, the result must equal
// the cached path fed the indices those defaults stand for.
test "cpu backend: attention over a plain sequence equals the cached path with implied indices" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const bsz: usize = 1;
    const l_q: usize = 4;
    const h_q: usize = 4;
    const h_kv: usize = 2;
    const t: usize = 4; // no cache: T == the sequence length
    const d_k: usize = 3;
    const d_v: usize = 2;

    const scale: f32 = 0.5;

    const q_buf: []u8 = try allocator.alloc(u8, bsz * l_q * h_q * d_k * @sizeOf(f32));
    defer allocator.free(q_buf);
    const k_buf: []u8 = try allocator.alloc(u8, bsz * h_kv * t * d_k * @sizeOf(f32));
    defer allocator.free(k_buf);
    const v_buf: []u8 = try allocator.alloc(u8, bsz * h_kv * t * d_v * @sizeOf(f32));
    defer allocator.free(v_buf);

    const q_vals: []align(1) f32 = asF32Slice(q_buf);
    const k_vals: []align(1) f32 = asF32Slice(k_buf);
    const v_vals: []align(1) f32 = asF32Slice(v_buf);
    for (q_vals, 0..) |*x, i| x.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8))) * 0.07;
    for (k_vals, 0..) |*x, i| x.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 19)) - 9))) * 0.05;
    for (v_vals, 0..) |*x, i| x.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 23)) - 11))) * 0.06;

    // The defaults the op supplies when the operands are absent: position == row,
    // every key live.
    const implied_pos: []i32 = try allocator.alloc(i32, bsz * l_q);
    defer allocator.free(implied_pos);
    for (implied_pos, 0..) |*x, i| x.* = @intCast(i % l_q);
    const implied_end: []i32 = try allocator.alloc(i32, bsz);
    defer allocator.free(implied_end);
    for (implied_end) |*x| x.* = @intCast(t);

    inline for (.{
        graph_mod.AttentionWindow.causal,
        graph_mod.AttentionWindow.full,
        graph_mod.AttentionWindow.sliding(1, 0),
        graph_mod.AttentionWindow.sliding(0, 0),
        graph_mod.AttentionWindow.sliding(1, 1),
        graph_mod.AttentionWindow.sliding(2, graph_mod.AttentionWindow.unbounded),
        graph_mod.AttentionWindow.chunked(2, 0),
        graph_mod.AttentionWindow.chunked(2, 2),
    }) |window| {
        const ref_vals: []f32 = try allocator.alloc(f32, bsz * l_q * h_q * d_v);
        defer allocator.free(ref_vals);
        cachedGqaRefF32(
            ref_vals,
            q_vals,
            k_vals,
            v_vals,
            implied_pos,
            implied_end,
            bsz,
            l_q,
            h_q,
            h_kv,
            t,
            d_k,
            d_v,
            scale,
            window,
            0.0,
        );

        var sm: manager_mod.StorageManager = manager_mod.StorageManager.init(allocator);
        defer sm.deinit();

        const q_tid = try sm.createTiledTensor(.f32, &[_]usize{ bsz, l_q, h_q, d_k }, &[_]usize{ 1, 2, 2, d_k }, .{ .tile_alignment = 64 });
        const k_tid = try sm.createTiledTensor(.f32, &[_]usize{ bsz, t, h_kv, d_k }, &[_]usize{ 1, 2, 1, d_k }, .{ .tile_alignment = 64 });
        const v_tid = try sm.createTiledTensor(.f32, &[_]usize{ bsz, t, h_kv, d_v }, &[_]usize{ 1, 2, 1, d_v }, .{ .tile_alignment = 64 });
        try sm.writeFromPackedScalar(q_tid, q_buf);
        try sm.writeFromPackedScalar(k_tid, k_buf);
        try sm.writeFromPackedScalar(v_tid, v_buf);

        var g: graph_mod.Graph = graph_mod.Graph.init(allocator);
        defer g.deinit();

        const q_in = try g.addInput(.f32, &[_]usize{ bsz, l_q, h_q, d_k });
        const k_in = try g.addInput(.f32, &[_]usize{ bsz, t, h_kv, d_k });
        const v_in = try g.addInput(.f32, &[_]usize{ bsz, t, h_kv, d_v });
        try g.bindExternal(q_in, @intCast(q_tid));
        try g.bindExternal(k_in, @intCast(k_tid));
        try g.bindExternal(v_in, @intCast(v_tid));

        // No positions, no end_index.
        const out = try g.addAttention(q_in, k_in, v_in, null, null, scale, window, 0.0);
        try g.setOutputs(&[_]graph_mod.ValueId{out});

        const policy: plan_mod.TilePolicy = .{ .base_square_2d = 4, .base_1d = 4, .tile_alignment = 64 };
        var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
        defer prog.deinit();

        var cpu: cpu_backend_mod.CpuBackend = cpu_backend_mod.CpuBackend.init(allocator);
        defer cpu.deinit();
        try cpu.backend().executeProgram(&prog, sm.tensorStore());

        const out_buf: []u8 = try allocator.alloc(u8, bsz * l_q * h_q * d_v * @sizeOf(f32));
        defer allocator.free(out_buf);
        try sm.readToPackedScalar(prog.outputs[0], out_buf);
        const out_vals: []align(1) f32 = asF32Slice(out_buf);

        for (out_vals, ref_vals) |got, want| {
            try std.testing.expect(std.math.isFinite(got));
            try std.testing.expectApproxEqAbs(want, got, 1e-5);
        }
    }
}

fn relPosMHARefF32(
    allocator: std.mem.Allocator,
    ref: []f32,
    q: []align(1) const f32,
    k: []align(1) const f32,
    v: []align(1) const f32,
    pe: []align(1) const f32,
    u: []align(1) const f32,
    vb: []align(1) const f32,
    batch: usize,
    heads: usize,
    t: usize, // self-attention: T_q == T_kv == t
    d: usize,
    scale: f32,
) void {
    const p_len: usize = 2 * t - 1;
    const sc: []f32 = allocator.alloc(f32, t) catch unreachable;
    defer allocator.free(sc);
    var b: usize = 0;
    while (b < batch) : (b += 1) {
        var h: usize = 0;
        while (h < heads) : (h += 1) {
            var i: usize = 0;
            while (i < t) : (i += 1) {
                // Layout [B, T, H, D].
                const q_base: usize = (((b * t + i) * heads + h) * d);
                var maxv: f32 = -std.math.inf(f32);
                var j: usize = 0;
                while (j < t) : (j += 1) {
                    const k_base: usize = (((b * t + j) * heads + h) * d);
                    const p_idx: usize = (t - 1) - i + j;
                    const pe_base: usize = ((h * p_len + p_idx) * d);
                    var ac: f32 = 0.0;
                    var bd: f32 = 0.0;
                    var dd: usize = 0;
                    while (dd < d) : (dd += 1) {
                        const qval = q[q_base + dd];
                        ac += (qval + u[h * d + dd]) * k[k_base + dd];
                        bd += (qval + vb[h * d + dd]) * pe[pe_base + dd];
                    }
                    sc[j] = (ac + bd) * scale;
                    maxv = @max(maxv, sc[j]);
                }
                var sum: f32 = 0.0;
                j = 0;
                while (j < t) : (j += 1) {
                    sc[j] = std.math.exp(sc[j] - maxv);
                    sum += sc[j];
                }
                const out_base: usize = (((b * t + i) * heads + h) * d);
                var dd: usize = 0;
                while (dd < d) : (dd += 1) {
                    var acc: f32 = 0.0;
                    j = 0;
                    while (j < t) : (j += 1) {
                        const v_base: usize = (((b * t + j) * heads + h) * d);
                        acc += sc[j] * v[v_base + dd];
                    }
                    ref[out_base + dd] = acc / sum;
                }
            }
        }
    }
}

test "cpu backend: rel-pos multi-head attention matches reference (f32)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const batch: usize = 2;
    const heads: usize = 2;
    const t: usize = 4; // self-attention: T_q == T_kv
    const d: usize = 8;
    const p_len: usize = 2 * t - 1;

    const qkv_len: usize = batch * heads * t * d;
    const q_buf: []u8 = try allocator.alloc(u8, qkv_len * 4);
    defer allocator.free(q_buf);
    const k_buf: []u8 = try allocator.alloc(u8, qkv_len * 4);
    defer allocator.free(k_buf);
    const v_buf: []u8 = try allocator.alloc(u8, qkv_len * 4);
    defer allocator.free(v_buf);
    const pe_buf: []u8 = try allocator.alloc(u8, heads * p_len * d * 4);
    defer allocator.free(pe_buf);
    const u_buf: []u8 = try allocator.alloc(u8, heads * d * 4);
    defer allocator.free(u_buf);
    const vb_buf: []u8 = try allocator.alloc(u8, heads * d * 4);
    defer allocator.free(vb_buf);

    const q_vals: []align(1) f32 = asF32Slice(q_buf);
    const k_vals: []align(1) f32 = asF32Slice(k_buf);
    const v_vals: []align(1) f32 = asF32Slice(v_buf);
    const pe_vals: []align(1) f32 = asF32Slice(pe_buf);
    const u_vals: []align(1) f32 = asF32Slice(u_buf);
    const vb_vals: []align(1) f32 = asF32Slice(vb_buf);

    for (0..qkv_len) |i| q_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 31)) - 15))) * 0.03;
    for (0..qkv_len) |i| k_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 29)) - 14))) * 0.03;
    for (0..qkv_len) |i| v_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 37)) - 18))) * 0.02;
    for (0..heads * p_len * d) |i| pe_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 23)) - 11))) * 0.025;
    for (0..heads * d) |i| u_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3))) * 0.05;
    for (0..heads * d) |i| vb_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 5)) - 2))) * 0.04;

    const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(d)));

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: Backend = cpu.backend();

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 8, .base_1d = 5, .tile_alignment = 64 };

    const q_tid = try sm.createTiledTensor(.f32, &[_]usize{ batch, t, heads, d }, &[_]usize{ 1, t, 1, d }, .{ .tile_alignment = 64 });
    const k_tid = try sm.createTiledTensor(.f32, &[_]usize{ batch, t, heads, d }, &[_]usize{ 1, t, 1, d }, .{ .tile_alignment = 64 });
    const v_tid = try sm.createTiledTensor(.f32, &[_]usize{ batch, t, heads, d }, &[_]usize{ 1, t, 1, d }, .{ .tile_alignment = 64 });
    const pe_tid = try sm.createTiledTensor(.f32, &[_]usize{ heads, p_len, d }, &[_]usize{ 1, p_len, d }, .{ .tile_alignment = 64 });
    const u_tid = try sm.createTiledTensor(.f32, &[_]usize{ heads, d }, &[_]usize{ heads, d }, .{ .tile_alignment = 64 });
    const vb_tid = try sm.createTiledTensor(.f32, &[_]usize{ heads, d }, &[_]usize{ heads, d }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(q_tid, q_buf);
    try sm.writeFromPackedScalar(k_tid, k_buf);
    try sm.writeFromPackedScalar(v_tid, v_buf);
    try sm.writeFromPackedScalar(pe_tid, pe_buf);
    try sm.writeFromPackedScalar(u_tid, u_buf);
    try sm.writeFromPackedScalar(vb_tid, vb_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const q_in = try g.addInput(.f32, &[_]usize{ batch, t, heads, d });
    const k_in = try g.addInput(.f32, &[_]usize{ batch, t, heads, d });
    const v_in = try g.addInput(.f32, &[_]usize{ batch, t, heads, d });
    const pe_in = try g.addInput(.f32, &[_]usize{ heads, p_len, d });
    const u_in = try g.addInput(.f32, &[_]usize{ heads, d });
    const vb_in = try g.addInput(.f32, &[_]usize{ heads, d });
    try g.bindExternal(q_in, @intCast(q_tid));
    try g.bindExternal(k_in, @intCast(k_tid));
    try g.bindExternal(v_in, @intCast(v_tid));
    try g.bindExternal(pe_in, @intCast(pe_tid));
    try g.bindExternal(u_in, @intCast(u_tid));
    try g.bindExternal(vb_in, @intCast(vb_tid));

    const y = try g.addRelPosMHA(q_in, k_in, v_in, pe_in, u_in, vb_in, null, scale, .full, t - 1, 0);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();
    try backend.executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, qkv_len * 4);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    const ref: []f32 = try allocator.alloc(f32, qkv_len);
    defer allocator.free(ref);
    relPosMHARefF32(allocator, ref, q_vals, k_vals, v_vals, pe_vals, u_vals, vb_vals, batch, heads, t, d, scale);

    var max_abs: f32 = 0.0;
    for (out_vals, ref) |got, r0| {
        try std.testing.expect(std.math.isFinite(got));
        max_abs = @max(max_abs, @abs(got - r0));
    }
    if (!(max_abs <= 6e-2)) {
        std.debug.print("rel-pos mha mismatch max_abs={}\n", .{max_abs});
        return error.TestExpectedEqual;
    }
}

// The structural chunked-limited window must be indistinguishable from the additive
// mask that used to encode the same pattern — that equivalence is the whole premise
// of the attribute, since it lets the executors skip out-of-window keys instead of
// scoring them and adding -1e9. Runs BOTH forms through the same graph shape and
// compares outputs elementwise.
test "cpu backend: chunked-limited window equals the equivalent additive mask" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const batch: usize = 2;
    const heads: usize = 2;
    const t: usize = 12; // T_q == T_kv
    const d: usize = 8;
    const p_len: usize = 2 * t - 1;
    const chunk: usize = 4;
    const left: usize = 5; // deliberately NOT a multiple of chunk
    const qkv_len: usize = batch * t * heads * d;

    const q_buf: []u8 = try allocator.alloc(u8, qkv_len * 4);
    defer allocator.free(q_buf);
    const k_buf: []u8 = try allocator.alloc(u8, qkv_len * 4);
    defer allocator.free(k_buf);
    const v_buf: []u8 = try allocator.alloc(u8, qkv_len * 4);
    defer allocator.free(v_buf);
    const pe_buf: []u8 = try allocator.alloc(u8, heads * p_len * d * 4);
    defer allocator.free(pe_buf);
    const u_buf: []u8 = try allocator.alloc(u8, heads * d * 4);
    defer allocator.free(u_buf);
    const vb_buf: []u8 = try allocator.alloc(u8, heads * d * 4);
    defer allocator.free(vb_buf);
    const mask_buf: []u8 = try allocator.alloc(u8, t * t * 4);
    defer allocator.free(mask_buf);

    for (asF32Slice(q_buf), 0..) |*x, i| x.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 31)) - 15)) * 0.03;
    for (asF32Slice(k_buf), 0..) |*x, i| x.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 29)) - 14)) * 0.03;
    for (asF32Slice(v_buf), 0..) |*x, i| x.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 37)) - 18)) * 0.02;
    for (asF32Slice(pe_buf), 0..) |*x, i| x.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 23)) - 11)) * 0.025;
    for (asF32Slice(u_buf), 0..) |*x, i| x.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) * 0.05;
    for (asF32Slice(vb_buf), 0..) |*x, i| x.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 5)) - 2)) * 0.04;

    // The mask the Nemotron converter used to bake in for `chunked_limited`.
    const mask_vals: []align(1) f32 = asF32Slice(mask_buf);
    for (0..t) |i| {
        for (0..t) |j| {
            const cs: usize = (i / chunk) * chunk;
            const lo: usize = cs - @min(cs, left);
            const hi: usize = @min(cs + chunk, t);
            mask_vals[i * t + j] = if (j >= lo and j < hi) 0.0 else -1.0e9;
        }
    }

    const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(d)));
    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 8, .base_1d = 5, .tile_alignment = 64 };

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: Backend = cpu.backend();

    // `use_mask`: the old encoding. Otherwise the structural window.
    const run = struct {
        fn go(
            alloc: std.mem.Allocator,
            be: Backend,
            pol: plan_mod.TilePolicy,
            sc: f32,
            use_mask: bool,
            bufs: [7][]u8,
            out: []u8,
        ) !void {
            var sm = manager_mod.StorageManager.init(alloc);
            defer sm.deinit();

            const q_tid = try sm.createTiledTensor(.f32, &[_]usize{ batch, t, heads, d }, &[_]usize{ 1, t, 1, d }, .{ .tile_alignment = 64 });
            const k_tid = try sm.createTiledTensor(.f32, &[_]usize{ batch, t, heads, d }, &[_]usize{ 1, t, 1, d }, .{ .tile_alignment = 64 });
            const v_tid = try sm.createTiledTensor(.f32, &[_]usize{ batch, t, heads, d }, &[_]usize{ 1, t, 1, d }, .{ .tile_alignment = 64 });
            const pe_tid = try sm.createTiledTensor(.f32, &[_]usize{ heads, p_len, d }, &[_]usize{ 1, p_len, d }, .{ .tile_alignment = 64 });
            const u_tid = try sm.createTiledTensor(.f32, &[_]usize{ heads, d }, &[_]usize{ heads, d }, .{ .tile_alignment = 64 });
            const vb_tid = try sm.createTiledTensor(.f32, &[_]usize{ heads, d }, &[_]usize{ heads, d }, .{ .tile_alignment = 64 });
            const m_tid = try sm.createTiledTensor(.f32, &[_]usize{ t, t }, &[_]usize{ t, t }, .{ .tile_alignment = 64 });
            try sm.writeFromPackedScalar(q_tid, bufs[0]);
            try sm.writeFromPackedScalar(k_tid, bufs[1]);
            try sm.writeFromPackedScalar(v_tid, bufs[2]);
            try sm.writeFromPackedScalar(pe_tid, bufs[3]);
            try sm.writeFromPackedScalar(u_tid, bufs[4]);
            try sm.writeFromPackedScalar(vb_tid, bufs[5]);
            try sm.writeFromPackedScalar(m_tid, bufs[6]);

            var g = graph_mod.Graph.init(alloc);
            defer g.deinit();
            const q_in = try g.addInput(.f32, &[_]usize{ batch, t, heads, d });
            const k_in = try g.addInput(.f32, &[_]usize{ batch, t, heads, d });
            const v_in = try g.addInput(.f32, &[_]usize{ batch, t, heads, d });
            const pe_in = try g.addInput(.f32, &[_]usize{ heads, p_len, d });
            const u_in = try g.addInput(.f32, &[_]usize{ heads, d });
            const vb_in = try g.addInput(.f32, &[_]usize{ heads, d });
            const m_in = try g.addInput(.f32, &[_]usize{ t, t });
            try g.bindExternal(q_in, @intCast(q_tid));
            try g.bindExternal(k_in, @intCast(k_tid));
            try g.bindExternal(v_in, @intCast(v_tid));
            try g.bindExternal(pe_in, @intCast(pe_tid));
            try g.bindExternal(u_in, @intCast(u_tid));
            try g.bindExternal(vb_in, @intCast(vb_tid));
            try g.bindExternal(m_in, @intCast(m_tid));

            const y = if (use_mask)
                try g.addRelPosMHA(q_in, k_in, v_in, pe_in, u_in, vb_in, m_in, sc, .full, t - 1, 0)
            else
                try g.addRelPosMHA(q_in, k_in, v_in, pe_in, u_in, vb_in, null, sc, .chunked(chunk, left), t - 1, 0);
            try g.setOutputs(&[_]graph_mod.ValueId{y});

            var prog = try program.compileGraph(alloc, &g, &sm, .cpu(pol));
            defer prog.deinit();
            try be.executeProgram(&prog, sm.tensorStore());
            try sm.readToPackedScalar(prog.outputs[0], out);
        }
    }.go;

    const bufs: [7][]u8 = .{ q_buf, k_buf, v_buf, pe_buf, u_buf, vb_buf, mask_buf };
    const masked: []u8 = try allocator.alloc(u8, qkv_len * 4);
    defer allocator.free(masked);
    const windowed: []u8 = try allocator.alloc(u8, qkv_len * 4);
    defer allocator.free(windowed);

    try run(allocator, backend, policy, scale, true, bufs, masked);
    try run(allocator, backend, policy, scale, false, bufs, windowed);

    var max_abs: f32 = 0.0;
    for (asF32Slice(masked), asF32Slice(windowed)) |a, b| {
        try std.testing.expect(std.math.isFinite(a) and std.math.isFinite(b));
        max_abs = @max(max_abs, @abs(a - b));
    }
    // Not bit-identical: the masked path softmaxes over all T keys (the -1e9 entries
    // contribute exp(-80)-ish), the windowed path over the window only.
    if (!(max_abs <= 1e-5)) {
        std.debug.print("windowed vs masked mismatch max_abs={}\n", .{max_abs});
        return error.TestExpectedEqual;
    }
}

test "cpu backend: argmax over last axis returns i32 indices" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const rows: usize = 3;
    const n: usize = 8;
    const vals = [_]f32{
        0.1, 0.2, 0.9, 0.0, -1.0, 0.3, 0.4, 0.5, // argmax 2
        5.0, 1.0, 2.0, 3.0, 4.0, 9.0, 8.0, 7.0, // argmax 5
        -1.0, -2.0, -0.5, -3.0, -4.0, -5.0, -6.0, -7.0, // argmax 2
    };
    const expected = [_]i32{ 2, 5, 2 };

    const in_buf: []u8 = try allocator.alloc(u8, rows * n * 4);
    defer allocator.free(in_buf);
    const in_vals: []align(1) f32 = asF32Slice(in_buf);
    for (vals, 0..) |v, i| in_vals[i] = v;

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: Backend = cpu.backend();

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 8, .base_1d = 5, .tile_alignment = 64 };
    const in_tid = try sm.createTiledTensor(.f32, &[_]usize{ rows, n }, &[_]usize{ rows, n }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(in_tid, in_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();
    const x = try g.addInput(.f32, &[_]usize{ rows, n });
    try g.bindExternal(x, @intCast(in_tid));
    const y = try g.addArgMax(x, 1);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();
    try backend.executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, rows * 4);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    for (0..rows) |i| {
        const got: i32 = std.mem.readInt(i32, out_buf[i * 4 ..][0..4], .little);
        try std.testing.expectEqual(expected[i], got);
    }
}

test "cpu backend: packed-state loop (slice/cast/i32-add/scatter/concat) — in-graph decode pattern" {
    const allocator: std.mem.Allocator = std.testing.allocator;
    const M: usize = 3; // buffer length
    const L: usize = 2 + M; // [count, last, buf...]
    var mgr = manager_mod.StorageManager.init(allocator);
    defer mgr.deinit();

    const state_tid = try mgr.createTiledTensor(.f32, &[_]usize{L}, &[_]usize{L}, .{});
    const one_tid = try mgr.createTiledTensor(.i32, &[_]usize{1}, &[_]usize{1}, .{});
    var s0: [L]f32 = @splat(0.0);
    try mgr.writeFromPackedScalar(state_tid, std.mem.sliceAsBytes(s0[0..]));
    var onev: [1]i32 = .{1};
    try mgr.writeFromPackedScalar(one_tid, std.mem.sliceAsBytes(onev[0..]));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();
    const state = try g.addInput(.f32, &[_]usize{L});
    try g.bindExternal(state, state_tid);
    const one = try g.addInput(.i32, &[_]usize{1});
    try g.bindExternal(one, one_tid);

    // Body: buf[count] = count; count += 1. Repack into a new [L] state.
    try g.beginRegion();
    const count_f = try g.addViewSliceND(state, &[_]usize{0}, &[_]usize{1}); // [1] f32
    const last_f = try g.addViewSliceND(state, &[_]usize{1}, &[_]usize{1}); // [1] f32 (unchanged)
    const buf = try g.addViewSliceND(state, &[_]usize{2}, &[_]usize{M}); // [M] f32
    const count_i = try g.addCast(count_f, .i32); // [1] i32
    const buf2 = try g.addScatterRow(buf, count_i, count_f); // buf[count] = count
    const next_i = try g.addElemwiseBinary(.add, count_i, one); // count+1 (i32)
    const next_f = try g.addCast(next_i, .f32); // [1] f32
    const new_state = try g.addConcat(&[_]graph_mod.ValueId{ next_f, last_f, buf2 }, 0); // [L]
    const body_region = try g.endRegion(&[_]graph_mod.ValueId{new_state});
    const out = try g.addLoop(state, body_region, M);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    var prog = try program.compileGraph(allocator, &g, &mgr, .cpu(.{}));
    defer prog.deinit();
    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, mgr.tensorStore());

    var obuf: [L * 4]u8 = undefined;
    try mgr.readToPackedScalar(prog.outputs[0], obuf[0..]);
    const count_out: f32 = @bitCast(std.mem.readInt(u32, obuf[0..4], .little));
    try std.testing.expect(@abs(count_out - @as(f32, @floatFromInt(M))) <= 1e-5);
    for (0..M) |i| {
        const v: f32 = @bitCast(std.mem.readInt(u32, obuf[(2 + i) * 4 ..][0..4], .little));
        try std.testing.expect(@abs(v - @as(f32, @floatFromInt(i))) <= 1e-5);
    }
}

test "cpu backend: scatter row writes value at dynamic index" {
    const allocator: std.mem.Allocator = std.testing.allocator;
    const N: usize = 4;
    var mgr = manager_mod.StorageManager.init(allocator);
    defer mgr.deinit();

    const buf_tid = try mgr.createTiledTensor(.i32, &[_]usize{N}, &[_]usize{N}, .{});
    const idx_tid = try mgr.createTiledTensor(.i32, &[_]usize{1}, &[_]usize{1}, .{});
    const src_tid = try mgr.createTiledTensor(.i32, &[_]usize{1}, &[_]usize{1}, .{});
    var bufv: [N]i32 = .{ 10, 20, 30, 40 };
    var idxv: [1]i32 = .{2};
    var srcv: [1]i32 = .{99};
    try mgr.writeFromPackedScalar(buf_tid, std.mem.sliceAsBytes(bufv[0..]));
    try mgr.writeFromPackedScalar(idx_tid, std.mem.sliceAsBytes(idxv[0..]));
    try mgr.writeFromPackedScalar(src_tid, std.mem.sliceAsBytes(srcv[0..]));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();
    const buf = try g.addInput(.i32, &[_]usize{N});
    try g.bindExternal(buf, buf_tid);
    const idx = try g.addInput(.i32, &[_]usize{1});
    try g.bindExternal(idx, idx_tid);
    const src = try g.addInput(.i32, &[_]usize{1});
    try g.bindExternal(src, src_tid);
    const out = try g.addScatterRow(buf, idx, src);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    var prog = try program.compileGraph(allocator, &g, &mgr, .cpu(.{}));
    defer prog.deinit();
    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, mgr.tensorStore());

    const expect = [_]i32{ 10, 20, 99, 40 };
    var obuf: [N * 4]u8 = undefined;
    try mgr.readToPackedScalar(prog.outputs[0], obuf[0..]);
    for (0..N) |i| try std.testing.expectEqual(expect[i], std.mem.readInt(i32, obuf[i * 4 ..][0..4], .little));
}

test "cpu backend: i32 elementwise add and eq comparison" {
    const allocator: std.mem.Allocator = std.testing.allocator;
    const N: usize = 4;
    var mgr = manager_mod.StorageManager.init(allocator);
    defer mgr.deinit();

    const a_tid = try mgr.createTiledTensor(.i32, &[_]usize{N}, &[_]usize{N}, .{});
    const b_tid = try mgr.createTiledTensor(.i32, &[_]usize{N}, &[_]usize{N}, .{});
    var abuf: [N]i32 = .{ 4, 3, 5, 2 };
    var bbuf: [N]i32 = .{ 1, 3, 5, 9 };
    try mgr.writeFromPackedScalar(a_tid, std.mem.sliceAsBytes(abuf[0..]));
    try mgr.writeFromPackedScalar(b_tid, std.mem.sliceAsBytes(bbuf[0..]));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();
    const a = try g.addInput(.i32, &[_]usize{N});
    try g.bindExternal(a, a_tid);
    const b = try g.addInput(.i32, &[_]usize{N});
    try g.bindExternal(b, b_tid);
    const sum = try g.addElemwiseBinary(.add, a, b);
    const eq = try g.addElemwiseBinary(.eq, a, b);
    try g.setOutputs(&[_]graph_mod.ValueId{ sum, eq });

    var prog = try program.compileGraph(allocator, &g, &mgr, .cpu(.{}));
    defer prog.deinit();
    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, mgr.tensorStore());

    const expect_sum = [_]i32{ 5, 6, 10, 11 };
    const expect_eq = [_]i32{ 0, 1, 1, 0 };
    var buf: [N * 4]u8 = undefined;
    try mgr.readToPackedScalar(prog.outputs[0], buf[0..]);
    for (0..N) |i| try std.testing.expectEqual(expect_sum[i], std.mem.readInt(i32, buf[i * 4 ..][0..4], .little));
    try mgr.readToPackedScalar(prog.outputs[1], buf[0..]);
    for (0..N) |i| try std.testing.expectEqual(expect_eq[i], std.mem.readInt(i32, buf[i * 4 ..][0..4], .little));
}

test "cpu backend: loop body lowers matmul + relu (region full-op lowering)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const N: usize = 4;
    var mgr = manager_mod.StorageManager.init(allocator);
    defer mgr.deinit();

    const carried_tid = try mgr.createTiledTensor(.f32, &[_]usize{ 1, N }, &[_]usize{ 1, N }, .{});
    const w_tid = try mgr.createTiledTensor(.f32, &[_]usize{ N, N }, &[_]usize{ N, N }, .{});

    var cbuf: [N]f32 = .{ 1.0, 1.0, 1.0, 1.0 };
    try mgr.writeFromPackedScalar(carried_tid, std.mem.sliceAsBytes(cbuf[0..]));
    var wbuf: [N * N]f32 = @splat(0.0);
    {
        var i: usize = 0;
        while (i < N) : (i += 1) wbuf[i * N + i] = 0.5; // W = 0.5 * I
    }
    try mgr.writeFromPackedScalar(w_tid, std.mem.sliceAsBytes(wbuf[0..]));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();
    const carried = try g.addInput(.f32, &[_]usize{ 1, N });
    try g.bindExternal(carried, carried_tid);
    const w = try g.addInput(.f32, &[_]usize{ N, N });
    try g.bindExternal(w, w_tid);

    // Loop body: carried = relu(carried @ W). With W = 0.5*I and positive inputs,
    // each iteration halves the carried vector (relu is a no-op here but exercises
    // a non-elementwise + unary op inside a region — previously rejected).
    try g.beginRegion();
    const mm = try g.addMatMul(carried, w, 1.0, 0.0);
    const next = try g.addUnary(.relu, mm);
    const body_region = try g.endRegion(&[_]graph_mod.ValueId{next});
    const out = try g.addLoop(carried, body_region, 3);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    var prog = try program.compileGraph(allocator, &g, &mgr, .cpu(.{}));
    defer prog.deinit();
    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, mgr.tensorStore());

    var obytes: [N * 4]u8 = undefined;
    try mgr.readToPackedScalar(prog.outputs[0], obytes[0..]);
    var i: usize = 0;
    while (i < N) : (i += 1) {
        const v: f32 = @bitCast(std.mem.readInt(u32, obytes[i * 4 ..][0..4], .little));
        try std.testing.expect(@abs(v - 0.125) <= 1e-5); // 1.0 * 0.5^3
    }
}

test "cpu backend: matmul f16 allows promoted f32 output" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const m: usize = 2;
    const k: usize = 4;
    const n: usize = 3;

    const a_bytes_len: usize = m * k * 2;
    const b_bytes_len: usize = k * n * 2;

    const a_buf: []u8 = try allocator.alloc(u8, a_bytes_len);
    defer allocator.free(a_buf);
    const b_buf: []u8 = try allocator.alloc(u8, b_bytes_len);
    defer allocator.free(b_buf);

    const a_vals: []align(1) f16 = asF16Slice(a_buf);
    const b_vals: []align(1) f16 = asF16Slice(b_buf);

    for (0..m * k) |i| a_vals[i] = @as(f16, @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast(i)) - 3)) * 0.25));
    for (0..k * n) |i| b_vals[i] = @as(f16, @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i % 5))) - 2)) * 0.5));

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: Backend = cpu.backend();

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    // Intentionally choose non-matmul-friendly tiling to ensure retile logic is exercised.
    const a_tid = try sm.createTiledTensor(.f16, &[_]usize{ m, k }, &[_]usize{ 1, 2 }, .{ .tile_alignment = 64 });
    const b_tid = try sm.createTiledTensor(.f16, &[_]usize{ k, n }, &[_]usize{ 2, 1 }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(a_tid, a_buf);
    try sm.writeFromPackedScalar(b_tid, b_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const a_in = try g.addInput(.f16, &[_]usize{ m, k });
    const b_in = try g.addInput(.f16, &[_]usize{ k, n });
    try g.bindExternal(a_in, @intCast(a_tid));
    try g.bindExternal(b_in, @intCast(b_tid));

    const c = try g.addMatMul(a_in, b_in, 1.0, 0.0);
    // Request f16->f32 promotion by constraining the matmul output dtype.
    g.values.items[@intCast(c)].dtype = .f32;

    const out = try g.addReduce(.sum, c);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    // Naive reference: matmul in f32 then sum.
    var expected_sum: f32 = 0.0;
    for (0..m) |i| {
        for (0..n) |j| {
            var acc: f32 = 0.0;
            for (0..k) |kk| {
                const av: f32 = @floatCast(a_vals[i * k + kk]);
                const bv: f32 = @floatCast(b_vals[kk * n + j]);
                acc += av * bv;
            }
            expected_sum += acc;
        }
    }

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    try backend.executeProgram(&prog, sm.tensorStore());

    var out_buf: [4]u8 = .{ 0, 0, 0, 0 };
    try sm.readToPackedScalar(prog.outputs[0], out_buf[0..4]);
    const out_val: f32 = @as(*align(1) const f32, @ptrCast(out_buf[0..4].ptr)).*;
    try std.testing.expect(std.math.isFinite(out_val));
    // f16 inputs may accumulate slightly differently depending on kernel, but should be very close.
    try std.testing.expectApproxEqAbs(expected_sum, out_val, 5e-3);
}

test "cpu backend: view ops lower to materialization (transpose/slice/reshape)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const rows: usize = 3;
    const cols: usize = 4;

    // Packed input.
    const x_bytes_len: usize = rows * cols * 4;
    const x_buf: []u8 = try allocator.alloc(u8, x_bytes_len);
    defer allocator.free(x_buf);

    const x_vals: []align(1) f32 = @ptrCast(x_buf);
    for (0..rows * cols) |i| x_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 6)) * 0.25;

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: Backend = cpu.backend();

    // Expected: transpose -> slice -> reshape -> relu -> sum.
    // Slice is on the transposed tensor: take rows [1..3) and cols [0..3).
    var expected_sum: f32 = 0.0;
    // This corresponds to x[row=cc, col=1+rr] for rr in 0..2, cc in 0..3? see derivation.
    // Here slice shape is [2,3] (len0=2,len1=3).
    {
        // Elements in reshape order: s[0,0..2], s[1,0..2]
        const coords = [_][2]usize{
            .{ 0, 1 }, .{ 1, 1 }, .{ 2, 1 },
            .{ 0, 2 }, .{ 1, 2 }, .{ 2, 2 },
        };
        for (coords) |rc| {
            const r: usize = rc[0];
            const c: usize = rc[1];
            const v: f32 = x_vals[r * cols + c];
            expected_sum += if (v > 0) v else 0;
        }
    }

    // Storage + graph.
    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const x_tid = try sm.createTiledTensor(.f32, &[_]usize{ rows, cols }, &[_]usize{ 2, 2 }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(x_tid, x_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f32, &[_]usize{ rows, cols });
    try g.bindExternal(x_in, @intCast(x_tid));

    const t = try g.addViewTranspose2D(x_in);
    const s = try g.addViewSlice2D(t, 1, 2, 0, 3);
    const r = try g.addViewReshape(s, &[_]usize{6});
    const u = try g.addRelu(r);
    const out = try g.addReduce(.sum, u);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();
    try backend.executeProgram(&prog, sm.tensorStore());

    var out_buf: [4]u8 = .{ 0, 0, 0, 0 };
    try sm.readToPackedScalar(prog.outputs[0], out_buf[0..4]);
    const out_val: f32 = @as(*align(1) const f32, @ptrCast(out_buf[0..4].ptr)).*;
    try std.testing.expect(@abs(out_val - expected_sum) <= 1e-5);
}

test "cpu backend: reshape supports rank-3 materialization" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const rows: usize = 2;
    const cols: usize = 3;

    const x_bytes_len: usize = rows * cols * 4;
    const x_buf: []u8 = try allocator.alloc(u8, x_bytes_len);
    defer allocator.free(x_buf);
    const x_vals: []align(1) f32 = asF32Slice(x_buf);

    for (0..rows * cols) |i| {
        x_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 2));
    }

    var expected_sum: f32 = 0.0;
    for (x_vals) |v| {
        const r: f32 = if (v > 0.0) v else 0.0;
        expected_sum += r;
    }

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const x_tid = try sm.createTiledTensor(.f32, &[_]usize{ rows, cols }, &[_]usize{ 2, 2 }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(x_tid, x_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f32, &[_]usize{ rows, cols });
    try g.bindExternal(x_in, @intCast(x_tid));

    const r3 = try g.addViewReshape(x_in, &[_]usize{ 1, rows, cols });
    const r1 = try g.addViewReshape(r3, &[_]usize{rows * cols});
    const u = try g.addRelu(r1);
    const out = try g.addReduce(.sum, u);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    var out_buf: [4]u8 = .{ 0, 0, 0, 0 };
    try sm.readToPackedScalar(prog.outputs[0], out_buf[0..4]);
    const out_val: f32 = @as(*align(1) const f32, @ptrCast(out_buf[0..4].ptr)).*;
    try std.testing.expectApproxEqAbs(expected_sum, out_val, 1e-6);
}

test "cpu backend: view slice nd materialization rank-3" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const d0: usize = 2;
    const d1: usize = 3;
    const d2: usize = 4;
    const x_len: usize = d0 * d1 * d2;

    const x_buf: []u8 = try allocator.alloc(u8, x_len * 4);
    defer allocator.free(x_buf);
    const x_vals: []align(1) f32 = asF32Slice(x_buf);
    for (0..x_len) |i| x_vals[i] = @floatFromInt(i + 1);

    // Slice: starts [1,1,0], lens [1,2,3] => values:
    // (1,1,0..2) and (1,2,0..2)
    var expected_sum: f32 = 0.0;
    var r_idx: usize = 1;
    while (r_idx < 3) : (r_idx += 1) {
        var c_idx: usize = 0;
        while (c_idx < 3) : (c_idx += 1) {
            const idx: usize = (1 * d1 + r_idx) * d2 + c_idx;
            expected_sum += x_vals[idx];
        }
    }

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const x_tid = try sm.createTiledTensor(.f32, &[_]usize{ d0, d1, d2 }, &[_]usize{ 1, 2, 2 }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(x_tid, x_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f32, &[_]usize{ d0, d1, d2 });
    try g.bindExternal(x_in, @intCast(x_tid));

    const s = try g.addViewSliceND(x_in, &[_]usize{ 1, 1, 0 }, &[_]usize{ 1, 2, 3 });
    const r = try g.addViewReshape(s, &[_]usize{6});
    const out = try g.addReduce(.sum, r);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    var out_buf: [4]u8 = .{ 0, 0, 0, 0 };
    try sm.readToPackedScalar(prog.outputs[0], out_buf[0..4]);
    const out_val: f32 = @as(*align(1) const f32, @ptrCast(out_buf[0..4].ptr)).*;
    try std.testing.expectApproxEqAbs(expected_sum, out_val, 1e-5);
}

test "cpu backend: concat axis-1 materialization" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const rows: usize = 2;
    const c0: usize = 2;
    const c1: usize = 1;

    const a_buf: []u8 = try allocator.alloc(u8, rows * c0 * 4);
    defer allocator.free(a_buf);
    const b_buf: []u8 = try allocator.alloc(u8, rows * c1 * 4);
    defer allocator.free(b_buf);
    const a_vals: []align(1) f32 = asF32Slice(a_buf);
    const b_vals: []align(1) f32 = asF32Slice(b_buf);

    a_vals[0] = 1.0;
    a_vals[1] = 2.0;
    a_vals[2] = 3.0;
    a_vals[3] = 4.0;
    b_vals[0] = 10.0;
    b_vals[1] = 20.0;

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const a_tid = try sm.createTiledTensor(.f32, &[_]usize{ rows, c0 }, &[_]usize{ 2, 2 }, .{ .tile_alignment = 64 });
    const b_tid = try sm.createTiledTensor(.f32, &[_]usize{ rows, c1 }, &[_]usize{ 2, 1 }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(a_tid, a_buf);
    try sm.writeFromPackedScalar(b_tid, b_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();
    const a_in = try g.addInput(.f32, &[_]usize{ rows, c0 });
    const b_in = try g.addInput(.f32, &[_]usize{ rows, c1 });
    try g.bindExternal(a_in, @intCast(a_tid));
    try g.bindExternal(b_in, @intCast(b_tid));

    const c = try g.addConcat(&[_]graph_mod.ValueId{ a_in, b_in }, 1);
    try g.setOutputs(&[_]graph_mod.ValueId{c});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, rows * (c0 + c1) * 4);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);
    const expected: [6]f32 = .{ 1.0, 2.0, 10.0, 3.0, 4.0, 20.0 };
    for (expected, 0..) |e, idx| {
        try std.testing.expectApproxEqAbs(e, out_vals[idx], 1e-6);
    }
}

test "cpu backend: gather rows matches reference (f32)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const v: usize = 8;
    const d: usize = 4;
    const b: usize = 2;
    const l: usize = 3;

    const table_buf: []u8 = try allocator.alloc(u8, v * d * 4);
    defer allocator.free(table_buf);
    const table_vals: []align(1) f32 = asF32Slice(table_buf);
    for (0..v) |r| {
        for (0..d) |c| {
            table_vals[r * d + c] = @floatFromInt(r * 10 + c);
        }
    }

    const idx_buf: []u8 = try allocator.alloc(u8, b * l * @sizeOf(i32));
    defer allocator.free(idx_buf);
    const idx_ptr: [*]align(1) i32 = @ptrCast(idx_buf.ptr);
    const idx_vals: []align(1) i32 = idx_ptr[0 .. idx_buf.len / @sizeOf(i32)];
    idx_vals[0] = 0;
    idx_vals[1] = 3;
    idx_vals[2] = 7;
    idx_vals[3] = 1;
    idx_vals[4] = 2;
    idx_vals[5] = 5;

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    // Indices are deliberately mis-tiled to exercise i32 scalar retiling.
    const table_tid = try sm.createTiledTensor(.f32, &[_]usize{ v, d }, &[_]usize{ 4, d }, .{ .tile_alignment = 64 });
    const idx_tid = try sm.createTiledTensor(.i32, &[_]usize{ b, l }, &[_]usize{ 1, 2 }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(table_tid, table_buf);
    try sm.writeFromPackedScalar(idx_tid, idx_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const table_in = try g.addInput(.f32, &[_]usize{ v, d });
    const idx_in = try g.addInput(.i32, &[_]usize{ b, l });
    try g.bindExternal(table_in, @intCast(table_tid));
    try g.bindExternal(idx_in, @intCast(idx_tid));

    const out = try g.addGather(table_in, idx_in, 0, 0);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, b * l * d * 4);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    for (0..b) |bi| {
        for (0..l) |li| {
            const row: usize = @intCast(idx_vals[bi * l + li]);
            for (0..d) |di| {
                const want: f32 = table_vals[row * d + di];
                const got: f32 = out_vals[(bi * l + li) * d + di];
                try std.testing.expectApproxEqAbs(want, got, 0.0);
            }
        }
    }
}

test "cpu backend: shape index ops and batched gather derive pooling indices" {
    const allocator = std.testing.allocator;
    const b: usize = 2;
    const s: usize = 4;
    const d: usize = 3;
    const l: usize = 2;

    var data_vals: [b * s * d]f32 = undefined;
    for (0..b) |bi| {
        for (0..s) |si| {
            for (0..d) |di| {
                data_vals[(bi * s + si) * d + di] = @floatFromInt(bi * 100 + si * 10 + di);
            }
        }
    }
    const indices_vals = [_]i32{ 3, 1, 0, 2 };
    const tokens_vals: [b * s]i32 = @splat(0);

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();
    const data_tid = try sm.createTiledTensor(.f32, &.{ b, s, d }, &.{ 1, 2, d }, .{ .tile_alignment = 64 });
    const indices_tid = try sm.createTiledTensor(.i32, &.{ b, l }, &.{ 1, 1 }, .{ .tile_alignment = 64 });
    const tokens_tid = try sm.createTiledTensor(.i32, &.{ b, s }, &.{ 1, 2 }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(data_tid, std.mem.sliceAsBytes(&data_vals));
    try sm.writeFromPackedScalar(indices_tid, std.mem.sliceAsBytes(&indices_vals));
    try sm.writeFromPackedScalar(tokens_tid, std.mem.sliceAsBytes(&tokens_vals));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();
    const data = try g.addInput(.f32, &.{ b, s, d });
    const indices = try g.addInput(.i32, &.{ b, l });
    const tokens = try g.addInput(.i32, &.{ b, s });
    try g.bindExternal(data, @intCast(data_tid));
    try g.bindExternal(indices, @intCast(indices_tid));
    try g.bindExternal(tokens, @intCast(tokens_tid));

    const gathered = try g.addGather(data, indices, 1, 1);
    const positions = try g.addIota(tokens, 1);
    const seq_len = try g.addDim(tokens, 1);
    try g.setOutputs(&.{ gathered, positions, seq_len });

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 2, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();
    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    var gathered_vals: [b * l * d]f32 = undefined;
    var positions_vals: [b * s]i32 = undefined;
    var seq_len_vals: [1]i32 = undefined;
    try sm.readToPackedScalar(prog.outputs[0], std.mem.sliceAsBytes(&gathered_vals));
    try sm.readToPackedScalar(prog.outputs[1], std.mem.sliceAsBytes(&positions_vals));
    try sm.readToPackedScalar(prog.outputs[2], std.mem.sliceAsBytes(&seq_len_vals));

    for (0..b) |bi| {
        for (0..l) |li| {
            const source: usize = @intCast(indices_vals[bi * l + li]);
            for (0..d) |di| {
                try std.testing.expectEqual(
                    data_vals[(bi * s + source) * d + di],
                    gathered_vals[(bi * l + li) * d + di],
                );
            }
        }
        for (0..s) |si| try std.testing.expectEqual(@as(i32, @intCast(si)), positions_vals[bi * s + si]);
    }
    try std.testing.expectEqual(@as(i32, @intCast(s)), seq_len_vals[0]);
}

test "cpu backend: gather rows matches reference (f16)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const v: usize = 5;
    const d: usize = 3;
    const b: usize = 1;
    const l: usize = 4;

    const table_buf: []u8 = try allocator.alloc(u8, v * d * @sizeOf(f16));
    defer allocator.free(table_buf);
    const table_vals: []align(1) f16 = asF16Slice(table_buf);
    for (0..v) |r| {
        for (0..d) |c| {
            // Small integers are exactly representable in f16.
            table_vals[r * d + c] = @floatFromInt(r * 10 + c);
        }
    }

    const idx_buf: []u8 = try allocator.alloc(u8, b * l * @sizeOf(i32));
    defer allocator.free(idx_buf);
    const idx_ptr: [*]align(1) i32 = @ptrCast(idx_buf.ptr);
    const idx_vals: []align(1) i32 = idx_ptr[0 .. idx_buf.len / @sizeOf(i32)];
    idx_vals[0] = 4;
    idx_vals[1] = 0;
    idx_vals[2] = 2;
    idx_vals[3] = 1;

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const table_tid = try sm.createTiledTensor(.f16, &[_]usize{ v, d }, &[_]usize{ 4, d }, .{ .tile_alignment = 64 });
    const idx_tid = try sm.createTiledTensor(.i32, &[_]usize{ b, l }, &[_]usize{ b, l }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(table_tid, table_buf);
    try sm.writeFromPackedScalar(idx_tid, idx_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const table_in = try g.addInput(.f16, &[_]usize{ v, d });
    const idx_in = try g.addInput(.i32, &[_]usize{ b, l });
    try g.bindExternal(table_in, @intCast(table_tid));
    try g.bindExternal(idx_in, @intCast(idx_tid));

    const out = try g.addGather(table_in, idx_in, 0, 0);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, b * l * d * @sizeOf(f16));
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f16 = asF16Slice(out_buf);

    for (0..b) |bi| {
        for (0..l) |li| {
            const row: usize = @intCast(idx_vals[bi * l + li]);
            for (0..d) |di| {
                const want: f16 = table_vals[row * d + di];
                const got: f16 = out_vals[(bi * l + li) * d + di];
                try std.testing.expectEqual(want, got);
            }
        }
    }
}

test "cpu backend: gather rows matches reference (q8_0 table, f32 output)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    // One q8_0 block per row keeps the packing/dequant trivial to audit.
    const v: usize = 5;
    const d: usize = 32;
    const b: usize = 2;
    const l: usize = 3;

    // Reference f32 table. Choose a pattern that exercises both signs and a row-varying scale.
    const ref_vals: []f32 = try allocator.alloc(f32, v * d);
    defer allocator.free(ref_vals);
    for (0..v) |r| {
        const row_scale: f32 = @as(f32, @floatFromInt(r + 1)) * 0.125;
        for (0..d) |c| {
            const c_signed: i32 = @as(i32, @intCast(c)) - 16;
            ref_vals[r * d + c] = @as(f32, @floatFromInt(c_signed)) * row_scale;
        }
    }

    // Pack reference table into q8_0 per-row blocks.
    const block_bytes: usize = 34;
    const packed_table: []u8 = try allocator.alloc(u8, v * 1 * block_bytes);
    defer allocator.free(packed_table);

    const dequant_ref: []f32 = try allocator.alloc(f32, v * d);
    defer allocator.free(dequant_ref);

    for (0..v) |r| {
        // Find absmax over this row.
        var absmax: f32 = 0.0;
        for (0..d) |c| {
            const a: f32 = @abs(ref_vals[r * d + c]);
            if (a > absmax) absmax = a;
        }
        const scale_f32: f32 = if (absmax == 0.0) 1.0 else absmax / 127.0;
        const inv_scale: f32 = if (absmax == 0.0) 0.0 else 1.0 / scale_f32;
        // Round to nearest int8.
        const scale_f16: f16 = @floatCast(scale_f32);
        const scale_f32_quantized: f32 = @as(f32, scale_f16);
        const block_off: usize = r * block_bytes;
        const scale_bits: u16 = @bitCast(scale_f16);
        std.mem.writeInt(u16, packed_table[block_off .. block_off + 2][0..2], scale_bits, .little);
        for (0..d) |c| {
            const qv_f: f32 = ref_vals[r * d + c] * inv_scale;
            var qv: i32 = @intFromFloat(@round(qv_f));
            if (qv > 127) qv = 127;
            if (qv < -128) qv = -128;
            const qv_i8: i8 = @intCast(qv);
            packed_table[block_off + 2 + c] = @bitCast(qv_i8);
            dequant_ref[r * d + c] = @as(f32, @floatFromInt(qv_i8)) * scale_f32_quantized;
        }
    }

    const idx_buf: []u8 = try allocator.alloc(u8, b * l * @sizeOf(i32));
    defer allocator.free(idx_buf);
    const idx_ptr: [*]align(1) i32 = @ptrCast(idx_buf.ptr);
    const idx_vals: []align(1) i32 = idx_ptr[0 .. idx_buf.len / @sizeOf(i32)];
    idx_vals[0] = 0;
    idx_vals[1] = 3;
    idx_vals[2] = 4;
    idx_vals[3] = 1;
    idx_vals[4] = 2;
    idx_vals[5] = 0;

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    // tile_shape[1] must equal shape[1] for q8_0 embedding tables (one row = one contiguous
    // block run). tile_shape[0] is a small batch of rows; `2` forces more than one tile.
    const table_tid = try sm.createTiledTensor(
        .q8_0,
        &[_]usize{ v, d },
        &[_]usize{ 2, d },
        .{ .tile_alignment = 64, .quant_axis = 1 },
    );
    const idx_tid = try sm.createTiledTensor(.i32, &[_]usize{ b, l }, &[_]usize{ 1, 2 }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedQuant(table_tid, packed_table);
    try sm.writeFromPackedScalar(idx_tid, idx_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const table_in = try g.addInput(.q8_0, &[_]usize{ v, d });
    const idx_in = try g.addInput(.i32, &[_]usize{ b, l });
    try g.bindExternal(table_in, @intCast(table_tid));
    try g.bindExternal(idx_in, @intCast(idx_tid));

    const out = try g.addGather(table_in, idx_in, 0, 0);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, b * l * d * @sizeOf(f32));
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    for (0..b) |bi| {
        for (0..l) |li| {
            const row: usize = @intCast(idx_vals[bi * l + li]);
            for (0..d) |di| {
                const want: f32 = dequant_ref[row * d + di];
                const got: f32 = out_vals[(bi * l + li) * d + di];
                try std.testing.expectApproxEqAbs(want, got, 0.0);
            }
        }
    }
}

test "cpu backend: cast f32 -> f16 roundtrip matches reference" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const m: usize = 3;
    const n: usize = 8;

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const in_tid = try sm.createTiledTensor(.f32, &[_]usize{ m, n }, &[_]usize{ m, n }, .{ .tile_alignment = 64 });
    const in_buf: []u8 = try allocator.alloc(u8, m * n * @sizeOf(f32));
    defer allocator.free(in_buf);
    const in_vals: []align(1) f32 = asF32Slice(in_buf);
    for (0..m * n) |i| in_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 12)) * 0.125;
    try sm.writeFromPackedScalar(in_tid, in_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();
    const x = try g.addInput(.f32, &[_]usize{ m, n });
    try g.bindExternal(x, @intCast(in_tid));
    const y = try g.addCast(x, .f16);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 4, .base_1d = 8, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, m * n * @sizeOf(f16));
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f16 = asF16Slice(out_buf);

    for (0..m * n) |i| {
        const want: f16 = @floatCast(in_vals[i]);
        try std.testing.expectEqual(want, out_vals[i]);
    }
}

test "cpu backend: cast f16 -> f32 matches reference" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const n: usize = 6;

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const in_tid = try sm.createTiledTensor(.f16, &[_]usize{ 1, n }, &[_]usize{ 1, n }, .{ .tile_alignment = 64 });
    const in_buf: []u8 = try allocator.alloc(u8, n * @sizeOf(f16));
    defer allocator.free(in_buf);
    const in_vals: []align(1) f16 = asF16Slice(in_buf);
    // Small exact-representable values.
    for (0..n) |i| in_vals[i] = @floatFromInt(@as(i32, @intCast(i)) - 3);
    try sm.writeFromPackedScalar(in_tid, in_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();
    const x = try g.addInput(.f16, &[_]usize{ 1, n });
    try g.bindExternal(x, @intCast(in_tid));
    const y = try g.addCast(x, .f32);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 4, .base_1d = 8, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, n * @sizeOf(f32));
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    for (0..n) |i| {
        try std.testing.expectEqual(@as(f32, in_vals[i]), out_vals[i]);
    }
}

test "cpu backend: matmul NT (A f32 @ B^T q8_0 quant_axis=1) matches reference" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    // C[m, n] = sum_k A[m, k] * B[n, k] — B stored as [N, K] q8_0 with per-row blocks.
    const m: usize = 3;
    const k: usize = 64; // two q8_0 blocks per row
    const n: usize = 5;

    // Reference float values for B with per-row scales.
    const b_ref: []f32 = try allocator.alloc(f32, n * k);
    defer allocator.free(b_ref);
    for (0..n) |row| {
        const row_scale: f32 = @as(f32, @floatFromInt(row + 1)) * 0.0625;
        for (0..k) |c| {
            const c_signed: i32 = @as(i32, @intCast(c)) - 32;
            b_ref[row * k + c] = @as(f32, @floatFromInt(c_signed)) * row_scale;
        }
    }

    // Pack B into q8_0 per-row blocks (mirrors the layout Aion's tiled storage expects).
    const block_bytes: usize = 34;
    const blocks_per_row: usize = k / 32;
    const packed_b: []u8 = try allocator.alloc(u8, n * blocks_per_row * block_bytes);
    defer allocator.free(packed_b);
    const dequant_b: []f32 = try allocator.alloc(f32, n * k);
    defer allocator.free(dequant_b);

    for (0..n) |row| {
        var absmax: f32 = 0.0;
        for (0..k) |c| {
            const a: f32 = @abs(b_ref[row * k + c]);
            if (a > absmax) absmax = a;
        }
        const scale_f32: f32 = if (absmax == 0.0) 1.0 else absmax / 127.0;
        const inv_scale: f32 = if (absmax == 0.0) 0.0 else 1.0 / scale_f32;
        const scale_f16: f16 = @floatCast(scale_f32);
        const scale_rounded: f32 = @as(f32, scale_f16);
        var b: usize = 0;
        while (b < blocks_per_row) : (b += 1) {
            const block_off: usize = (row * blocks_per_row + b) * block_bytes;
            const scale_bits: u16 = @bitCast(scale_f16);
            std.mem.writeInt(u16, packed_b[block_off .. block_off + 2][0..2], scale_bits, .little);
            for (0..32) |c_off| {
                const c: usize = b * 32 + c_off;
                const qv_f: f32 = b_ref[row * k + c] * inv_scale;
                var qv: i32 = @intFromFloat(@round(qv_f));
                if (qv > 127) qv = 127;
                if (qv < -128) qv = -128;
                const qv_i8: i8 = @intCast(qv);
                packed_b[block_off + 2 + c_off] = @bitCast(qv_i8);
                dequant_b[row * k + c] = @as(f32, @floatFromInt(qv_i8)) * scale_rounded;
            }
        }
    }

    // A values.
    const a_buf: []f32 = try allocator.alloc(f32, m * k);
    defer allocator.free(a_buf);
    for (0..m * k) |i| a_buf[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 50)) * 0.01;

    // Reference: C = A @ dequant_b.T
    const c_ref: []f32 = try allocator.alloc(f32, m * n);
    defer allocator.free(c_ref);
    for (0..m) |mi| {
        for (0..n) |ni| {
            var acc: f32 = 0.0;
            for (0..k) |kk| {
                acc += a_buf[mi * k + kk] * dequant_b[ni * k + kk];
            }
            c_ref[mi * n + ni] = acc;
        }
    }

    // Build graph.
    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const a_tid = try sm.createTiledTensor(.f32, &[_]usize{ m, k }, &[_]usize{ m, k }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(a_tid, std.mem.sliceAsBytes(a_buf));

    const b_tid = try sm.createTiledTensor(
        .q8_0,
        &[_]usize{ n, k },
        &[_]usize{ n, k },
        .{ .tile_alignment = 64, .quant_axis = 1 },
    );
    try sm.writeFromPackedQuant(b_tid, packed_b);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();
    const a_in = try g.addInput(.f32, &[_]usize{ m, k });
    const b_in = try g.addInput(.q8_0, &[_]usize{ n, k });
    try g.bindExternal(a_in, @intCast(a_tid));
    try g.bindExternal(b_in, @intCast(b_tid));
    const c_out = try g.addMatMulNT(a_in, b_in, 1.0, 0.0);
    try g.setOutputs(&[_]graph_mod.ValueId{c_out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 4, .base_1d = 8, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, m * n * @sizeOf(f32));
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    for (0..m * n) |i| {
        // SIMD accumulation order differs from the scalar reference, so tolerance is
        // dominated by f32 rounding rather than q8_0 error.
        try std.testing.expectApproxEqAbs(c_ref[i], out_vals[i], 1e-4);
    }
}

test "cpu backend: gather rows out-of-bounds returns InvalidArgument" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const v: usize = 4;
    const d: usize = 2;
    const b: usize = 1;
    const l: usize = 3;

    const table_buf: []u8 = try allocator.alloc(u8, v * d * 4);
    defer allocator.free(table_buf);
    const table_vals: []align(1) f32 = asF32Slice(table_buf);
    for (0..v * d) |i| table_vals[i] = @floatFromInt(i);

    const idx_buf: []u8 = try allocator.alloc(u8, b * l * @sizeOf(i32));
    defer allocator.free(idx_buf);
    const idx_ptr: [*]align(1) i32 = @ptrCast(idx_buf.ptr);
    const idx_vals: []align(1) i32 = idx_ptr[0 .. idx_buf.len / @sizeOf(i32)];
    idx_vals[0] = 0;
    idx_vals[1] = -1; // invalid
    idx_vals[2] = 2;

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const table_tid = try sm.createTiledTensor(.f32, &[_]usize{ v, d }, &[_]usize{ 4, d }, .{ .tile_alignment = 64 });
    const idx_tid = try sm.createTiledTensor(.i32, &[_]usize{ b, l }, &[_]usize{ b, l }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(table_tid, table_buf);
    try sm.writeFromPackedScalar(idx_tid, idx_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const table_in = try g.addInput(.f32, &[_]usize{ v, d });
    const idx_in = try g.addInput(.i32, &[_]usize{ b, l });
    try g.bindExternal(table_in, @intCast(table_tid));
    try g.bindExternal(idx_in, @intCast(idx_tid));

    const out = try g.addGather(table_in, idx_in, 0, 0);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try std.testing.expectError(types.BackendError.InvalidArgument, cpu.backend().executeProgram(&prog, sm.tensorStore()));
}

test "cpu backend: rope1d matches chunked-halves reference (f32)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const b: usize = 2;
    const l: usize = 3;
    const n: usize = 2;
    const h: usize = 8;
    const total: usize = b * l * n * h;

    const base_frequency: f32 = 10000.0;
    const scale_factor: f32 = 1.0;
    const rope_proportion: f32 = 0.5;

    const x_buf: []u8 = try allocator.alloc(u8, total * @sizeOf(f32));
    defer allocator.free(x_buf);
    const x_vals: []align(1) f32 = asF32Slice(x_buf);
    for (0..total) |i| {
        x_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i % 23)) - 11)) * 0.125;
    }

    const pos_buf: []u8 = try allocator.alloc(u8, b * l * @sizeOf(i32));
    defer allocator.free(pos_buf);
    const pos_ptr: [*]align(1) i32 = @ptrCast(pos_buf.ptr);
    const pos_vals: []align(1) i32 = pos_ptr[0 .. pos_buf.len / @sizeOf(i32)];
    pos_vals[0] = 0;
    pos_vals[1] = 1;
    pos_vals[2] = 2;
    pos_vals[3] = 5;
    pos_vals[4] = 3;
    pos_vals[5] = 7;

    const ref_buf: []u8 = try allocator.alloc(u8, total * @sizeOf(f32));
    defer allocator.free(ref_buf);
    const ref_vals: []align(1) f32 = asF32Slice(ref_buf);

    const pairs_total: usize = h / 2;
    const rope_pairs_f: f32 = @floor(rope_proportion * @as(f32, @floatFromInt(pairs_total)));
    const rope_pairs_i: i64 = @intFromFloat(rope_pairs_f);
    const rope_pairs: usize = if (rope_pairs_i <= 0) 0 else @intCast(rope_pairs_i);
    const freq_step: f32 = @floatCast(std.math.pow(f64, @as(f64, base_frequency), @as(f64, -2.0 / @as(f32, @floatFromInt(h)))));

    for (0..b) |bi| {
        for (0..l) |li| {
            const pos: f32 = @floatFromInt(pos_vals[bi * l + li]);
            for (0..n) |ni| {
                const row_off: usize = (((bi * l + li) * n) + ni) * h;
                const x_row: []align(1) const f32 = x_vals[row_off .. row_off + h];
                const out_row: []align(1) f32 = ref_vals[row_off .. row_off + h];

                @memcpy(out_row, x_row);

                var ri: usize = 0;
                var freq: f32 = scale_factor;
                while (ri < rope_pairs) : (ri += 1) {
                    const xl: f32 = x_row[ri];
                    const xr: f32 = x_row[pairs_total + ri];
                    const angle: f32 = pos * freq;
                    const c: f32 = @floatCast(std.math.cos(@as(f64, angle)));
                    const s: f32 = @floatCast(std.math.sin(@as(f64, angle)));
                    out_row[ri] = xl * c - xr * s;
                    out_row[pairs_total + ri] = xl * s + xr * c;
                    freq *= freq_step;
                }
            }
        }
    }

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const x_tid = try sm.createTiledTensor(.f32, &[_]usize{ b, l, n, h }, &[_]usize{ 1, 2, 1, h }, .{ .tile_alignment = 64 });
    const pos_tid = try sm.createTiledTensor(.i32, &[_]usize{ b, l }, &[_]usize{ 1, 1 }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(x_tid, x_buf);
    try sm.writeFromPackedScalar(pos_tid, pos_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f32, &[_]usize{ b, l, n, h });
    const pos_in = try g.addInput(.i32, &[_]usize{ b, l });
    try g.bindExternal(x_in, @intCast(x_tid));
    try g.bindExternal(pos_in, @intCast(pos_tid));

    const out = try g.addRoPE1D(x_in, pos_in, base_frequency, scale_factor, rope_proportion);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, total * @sizeOf(f32));
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    var i: usize = 0;
    while (i < total) : (i += 1) {
        try std.testing.expectApproxEqAbs(ref_vals[i], out_vals[i], 2e-5);
    }
}

test "cpu backend: rope1d matches chunked-halves reference (f16)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const b: usize = 1;
    const l: usize = 4;
    const n: usize = 3;
    const h: usize = 6;
    const total: usize = b * l * n * h;

    const base_frequency: f32 = 10000.0;
    const scale_factor: f32 = 1.0;
    const rope_proportion: f32 = 1.0;

    const x_buf: []u8 = try allocator.alloc(u8, total * @sizeOf(f16));
    defer allocator.free(x_buf);
    const x_vals: []align(1) f16 = asF16Slice(x_buf);
    for (0..total) |i| {
        const v: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(i % 19)) - 9)) * 0.2;
        x_vals[i] = @floatCast(v);
    }

    const pos_buf: []u8 = try allocator.alloc(u8, b * l * @sizeOf(i32));
    defer allocator.free(pos_buf);
    const pos_ptr: [*]align(1) i32 = @ptrCast(pos_buf.ptr);
    const pos_vals: []align(1) i32 = pos_ptr[0 .. pos_buf.len / @sizeOf(i32)];
    pos_vals[0] = 0;
    pos_vals[1] = 2;
    pos_vals[2] = 4;
    pos_vals[3] = 7;

    const ref_f32_buf: []u8 = try allocator.alloc(u8, total * @sizeOf(f32));
    defer allocator.free(ref_f32_buf);
    const ref_f32: []align(1) f32 = asF32Slice(ref_f32_buf);

    const pairs_total: usize = h / 2;
    const rope_pairs_f: f32 = @floor(rope_proportion * @as(f32, @floatFromInt(pairs_total)));
    const rope_pairs_i: i64 = @intFromFloat(rope_pairs_f);
    const rope_pairs: usize = if (rope_pairs_i <= 0) 0 else @intCast(rope_pairs_i);
    const freq_step: f32 = @floatCast(std.math.pow(f64, @as(f64, base_frequency), @as(f64, -2.0 / @as(f32, @floatFromInt(h)))));

    for (0..b) |bi| {
        for (0..l) |li| {
            const pos: f32 = @floatFromInt(pos_vals[bi * l + li]);
            for (0..n) |ni| {
                const row_off: usize = (((bi * l + li) * n) + ni) * h;
                var j: usize = 0;
                while (j < h) : (j += 1) {
                    ref_f32[row_off + j] = @floatCast(x_vals[row_off + j]);
                }

                var ri: usize = 0;
                var freq: f32 = scale_factor;
                while (ri < rope_pairs) : (ri += 1) {
                    const xl: f32 = @floatCast(x_vals[row_off + ri]);
                    const xr: f32 = @floatCast(x_vals[row_off + pairs_total + ri]);
                    const angle: f32 = pos * freq;
                    const c: f32 = @floatCast(std.math.cos(@as(f64, angle)));
                    const s: f32 = @floatCast(std.math.sin(@as(f64, angle)));
                    ref_f32[row_off + ri] = xl * c - xr * s;
                    ref_f32[row_off + pairs_total + ri] = xl * s + xr * c;
                    freq *= freq_step;
                }
            }
        }
    }

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const x_tid = try sm.createTiledTensor(.f16, &[_]usize{ b, l, n, h }, &[_]usize{ 1, 2, 1, h }, .{ .tile_alignment = 64 });
    const pos_tid = try sm.createTiledTensor(.i32, &[_]usize{ b, l }, &[_]usize{ 1, 1 }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(x_tid, x_buf);
    try sm.writeFromPackedScalar(pos_tid, pos_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f16, &[_]usize{ b, l, n, h });
    const pos_in = try g.addInput(.i32, &[_]usize{ b, l });
    try g.bindExternal(x_in, @intCast(x_tid));
    try g.bindExternal(pos_in, @intCast(pos_tid));

    const out = try g.addRoPE1D(x_in, pos_in, base_frequency, scale_factor, rope_proportion);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, total * @sizeOf(f16));
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f16 = asF16Slice(out_buf);

    var i: usize = 0;
    while (i < total) : (i += 1) {
        const want: f32 = ref_f32[i];
        const got: f32 = @floatCast(out_vals[i]);
        try std.testing.expectApproxEqAbs(want, got, 3e-2);
    }
}

test "cpu backend: conv1d depthwise (NLC) matches reference" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const bsz: usize = 2;
    const l_in: usize = 7;
    const c_in: usize = 4;
    const groups: usize = 4; // depthwise
    const c_out: usize = 4;
    const k: usize = 3;
    const stride: usize = 2;
    const dilation: usize = 1;
    const pad_left: usize = 1;
    const pad_right: usize = 1;

    const l_out: usize = ((l_in + pad_left + pad_right - ((k - 1) * dilation + 1)) / stride) + 1;

    const x_len: usize = bsz * l_in * c_in;
    const w_len: usize = k * (c_in / groups) * c_out;
    const b_len: usize = c_out;
    const y_len: usize = bsz * l_out * c_out;

    const x_buf: []u8 = try allocator.alloc(u8, x_len * 4);
    defer allocator.free(x_buf);
    const w_buf: []u8 = try allocator.alloc(u8, w_len * 4);
    defer allocator.free(w_buf);
    const bias_buf: []u8 = try allocator.alloc(u8, b_len * 4);
    defer allocator.free(bias_buf);

    const x_vals: []align(1) f32 = asF32Slice(x_buf);
    const w_vals: []align(1) f32 = asF32Slice(w_buf);
    const bias_vals: []align(1) f32 = asF32Slice(bias_buf);

    for (0..x_len) |i| x_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 13))) - 6)) * 0.1;
    for (0..w_len) |i| w_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 7))) - 3)) * 0.05;
    for (0..b_len) |i| bias_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 2)) * 0.02;

    const ref_buf: []u8 = try allocator.alloc(u8, y_len * 4);
    defer allocator.free(ref_buf);
    const ref_vals: []align(1) f32 = asF32Slice(ref_buf);

    const c_in_g: usize = c_in / groups;
    const c_out_g: usize = c_out / groups;
    var b: usize = 0;
    while (b < bsz) : (b += 1) {
        var lo: usize = 0;
        while (lo < l_out) : (lo += 1) {
            var oc: usize = 0;
            while (oc < c_out) : (oc += 1) {
                const g: usize = oc / c_out_g;
                var acc: f32 = bias_vals[oc];
                var kw: usize = 0;
                while (kw < k) : (kw += 1) {
                    const in_nom: usize = lo * stride + kw * dilation;
                    if (in_nom < pad_left) continue;
                    const li: usize = in_nom - pad_left;
                    if (li >= l_in) continue;

                    var icg: usize = 0;
                    while (icg < c_in_g) : (icg += 1) {
                        const ic: usize = g * c_in_g + icg;
                        const x_idx: usize = ((b * l_in + li) * c_in) + ic;
                        const w_idx: usize = ((kw * c_in_g + icg) * c_out) + oc;
                        acc += x_vals[x_idx] * w_vals[w_idx];
                    }
                }
                ref_vals[((b * l_out + lo) * c_out) + oc] = acc;
            }
        }
    }

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const x_tid = try sm.createTiledTensor(.f32, &[_]usize{ bsz, l_in, c_in }, &[_]usize{ 1, 3, 2 }, .{ .tile_alignment = 64 });
    const w_tid = try sm.createTiledTensor(.f32, &[_]usize{ k, c_in_g, c_out }, &[_]usize{ 2, 1, 2 }, .{ .tile_alignment = 64 });
    const bias_tid = try sm.createTiledTensor(.f32, &[_]usize{c_out}, &[_]usize{2}, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(x_tid, x_buf);
    try sm.writeFromPackedScalar(w_tid, w_buf);
    try sm.writeFromPackedScalar(bias_tid, bias_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f32, &[_]usize{ bsz, l_in, c_in });
    const w_in = try g.addInput(.f32, &[_]usize{ k, c_in_g, c_out });
    const bias_in = try g.addInput(.f32, &[_]usize{c_out});
    try g.bindExternal(x_in, @intCast(x_tid));
    try g.bindExternal(w_in, @intCast(w_tid));
    try g.bindExternal(bias_in, @intCast(bias_tid));

    const y = try g.addConv1D(x_in, w_in, bias_in, stride, dilation, pad_left, pad_right, groups);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 4, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, y_len * 4);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    var max_abs: f32 = 0.0;
    for (out_vals, ref_vals) |got, want| max_abs = @max(max_abs, @abs(got - want));
    try std.testing.expect(max_abs <= 1e-5);
}

test "cpu backend: conv1d depthwise reflect padding matches reference" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const bsz: usize = 1;
    const l_in: usize = 6;
    const c_in: usize = 4;
    const groups: usize = c_in;
    const c_out: usize = c_in;
    const k: usize = 3;
    const stride: usize = 1;
    const dilation: usize = 1;
    const pad_left: usize = 2;
    const pad_right: usize = 1;

    const l_out: usize = ((l_in + pad_left + pad_right - ((k - 1) * dilation + 1)) / stride) + 1;

    const x_len: usize = bsz * l_in * c_in;
    const w_len: usize = k * 1 * c_out;
    const b_len: usize = c_out;
    const y_len: usize = bsz * l_out * c_out;

    const x_buf: []u8 = try allocator.alloc(u8, x_len * 4);
    defer allocator.free(x_buf);
    const w_buf: []u8 = try allocator.alloc(u8, w_len * 4);
    defer allocator.free(w_buf);
    const bias_buf: []u8 = try allocator.alloc(u8, b_len * 4);
    defer allocator.free(bias_buf);

    const x_vals: []align(1) f32 = asF32Slice(x_buf);
    const w_vals: []align(1) f32 = asF32Slice(w_buf);
    const bias_vals: []align(1) f32 = asF32Slice(bias_buf);

    for (0..x_len) |i| x_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 15))) - 7)) * 0.09;
    for (0..w_len) |i| w_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 9))) - 4)) * 0.04;
    for (0..b_len) |i| bias_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 1)) * 0.03;

    const ref_buf: []u8 = try allocator.alloc(u8, y_len * 4);
    defer allocator.free(ref_buf);
    const ref_vals: []align(1) f32 = asF32Slice(ref_buf);

    const reflectIndex = struct {
        fn f(idx_nom: isize, len: usize) usize {
            var x: isize = idx_nom;
            const l: isize = @intCast(len);
            while (x < 0 or x >= l) {
                if (x < 0) x = -x else x = (2 * l - 2) - x;
            }
            return @intCast(x);
        }
    }.f;

    var b: usize = 0;
    while (b < bsz) : (b += 1) {
        var lo: usize = 0;
        while (lo < l_out) : (lo += 1) {
            var oc: usize = 0;
            while (oc < c_out) : (oc += 1) {
                var acc: f32 = bias_vals[oc];
                var kw: usize = 0;
                while (kw < k) : (kw += 1) {
                    const in_nom: usize = lo * stride + kw * dilation;
                    const li: usize = reflectIndex(@as(isize, @intCast(in_nom)) - @as(isize, @intCast(pad_left)), l_in);
                    const x_idx: usize = ((b * l_in + li) * c_in) + oc;
                    const w_idx: usize = ((kw * 1) * c_out) + oc;
                    acc += x_vals[x_idx] * w_vals[w_idx];
                }
                ref_vals[((b * l_out + lo) * c_out) + oc] = acc;
            }
        }
    }

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const x_tid = try sm.createTiledTensor(.f32, &[_]usize{ bsz, l_in, c_in }, &[_]usize{ 1, 4, 2 }, .{ .tile_alignment = 64 });
    const w_tid = try sm.createTiledTensor(.f32, &[_]usize{ k, 1, c_out }, &[_]usize{ 2, 1, 2 }, .{ .tile_alignment = 64 });
    const bias_tid = try sm.createTiledTensor(.f32, &[_]usize{c_out}, &[_]usize{2}, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(x_tid, x_buf);
    try sm.writeFromPackedScalar(w_tid, w_buf);
    try sm.writeFromPackedScalar(bias_tid, bias_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f32, &[_]usize{ bsz, l_in, c_in });
    const w_in = try g.addInput(.f32, &[_]usize{ k, 1, c_out });
    const bias_in = try g.addInput(.f32, &[_]usize{c_out});
    try g.bindExternal(x_in, @intCast(x_tid));
    try g.bindExternal(w_in, @intCast(w_tid));
    try g.bindExternal(bias_in, @intCast(bias_tid));

    const y = try g.addConv1DWithPadMode(x_in, w_in, bias_in, stride, dilation, pad_left, pad_right, .reflect, groups);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 2, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, y_len * 4);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    var max_abs: f32 = 0.0;
    for (out_vals, ref_vals) |got, want| max_abs = @max(max_abs, @abs(got - want));
    try std.testing.expect(max_abs <= 1e-5);
}

test "cpu backend: conv1d reflect padding matches reference" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const bsz: usize = 1;
    const l_in: usize = 5;
    const c_in: usize = 2;
    const c_out: usize = 2;
    const groups: usize = 1;
    const k: usize = 3;
    const stride: usize = 1;
    const dilation: usize = 1;
    const pad_left: usize = 2;
    const pad_right: usize = 1;

    const l_out: usize = ((l_in + pad_left + pad_right - ((k - 1) * dilation + 1)) / stride) + 1;

    const x_len: usize = bsz * l_in * c_in;
    const w_len: usize = k * c_in * c_out;
    const y_len: usize = bsz * l_out * c_out;

    const x_buf: []u8 = try allocator.alloc(u8, x_len * 4);
    defer allocator.free(x_buf);
    const w_buf: []u8 = try allocator.alloc(u8, w_len * 4);
    defer allocator.free(w_buf);

    const x_vals: []align(1) f32 = asF32Slice(x_buf);
    const w_vals: []align(1) f32 = asF32Slice(w_buf);
    for (0..x_len) |i| x_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 11))) - 5)) * 0.1;
    for (0..w_len) |i| w_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 7))) - 3)) * 0.05;

    const ref_buf: []u8 = try allocator.alloc(u8, y_len * 4);
    defer allocator.free(ref_buf);
    const ref_vals: []align(1) f32 = asF32Slice(ref_buf);

    const reflectIndex = struct {
        fn f(idx_nom: isize, len: usize) usize {
            var x: isize = idx_nom;
            const l: isize = @intCast(len);
            while (x < 0 or x >= l) {
                if (x < 0) x = -x else x = (2 * l - 2) - x;
            }
            return @intCast(x);
        }
    }.f;

    var b: usize = 0;
    while (b < bsz) : (b += 1) {
        var lo: usize = 0;
        while (lo < l_out) : (lo += 1) {
            var oc: usize = 0;
            while (oc < c_out) : (oc += 1) {
                var acc: f32 = 0.0;
                var kw: usize = 0;
                while (kw < k) : (kw += 1) {
                    const in_nom: usize = lo * stride + kw * dilation;
                    const li_nom: isize = @as(isize, @intCast(in_nom)) - @as(isize, @intCast(pad_left));
                    const li: usize = reflectIndex(li_nom, l_in);
                    var ic: usize = 0;
                    while (ic < c_in) : (ic += 1) {
                        const x_idx: usize = ((b * l_in + li) * c_in) + ic;
                        const w_idx: usize = ((kw * c_in + ic) * c_out) + oc;
                        acc += x_vals[x_idx] * w_vals[w_idx];
                    }
                }
                ref_vals[((b * l_out + lo) * c_out) + oc] = acc;
            }
        }
    }

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();
    const x_tid = try sm.createTiledTensor(.f32, &[_]usize{ bsz, l_in, c_in }, &[_]usize{ 1, 3, 2 }, .{ .tile_alignment = 64 });
    const w_tid = try sm.createTiledTensor(.f32, &[_]usize{ k, c_in, c_out }, &[_]usize{ 2, c_in, 2 }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(x_tid, x_buf);
    try sm.writeFromPackedScalar(w_tid, w_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();
    const x_in = try g.addInput(.f32, &[_]usize{ bsz, l_in, c_in });
    const w_in = try g.addInput(.f32, &[_]usize{ k, c_in, c_out });
    try g.bindExternal(x_in, @intCast(x_tid));
    try g.bindExternal(w_in, @intCast(w_tid));
    const y = try g.addConv1DWithPadMode(x_in, w_in, null, stride, dilation, pad_left, pad_right, .reflect, groups);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 4, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, y_len * 4);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    var max_abs: f32 = 0.0;
    for (out_vals, ref_vals) |got, want| max_abs = @max(max_abs, @abs(got - want));
    try std.testing.expect(max_abs <= 1e-5);
}

test "cpu backend: conv2d reflect padding matches reference" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const bsz: usize = 1;
    const h_in: usize = 4;
    const w_in: usize = 5;
    const c_in: usize = 2;
    const c_out: usize = 3;
    const groups: usize = 1;
    const k_h: usize = 3;
    const k_w: usize = 3;
    const stride_h: usize = 1;
    const stride_w: usize = 1;
    const dilation_h: usize = 1;
    const dilation_w: usize = 1;
    const pad_top: usize = 1;
    const pad_bottom: usize = 2;
    const pad_left: usize = 2;
    const pad_right: usize = 1;

    const h_out: usize = ((h_in + pad_top + pad_bottom - ((k_h - 1) * dilation_h + 1)) / stride_h) + 1;
    const w_out: usize = ((w_in + pad_left + pad_right - ((k_w - 1) * dilation_w + 1)) / stride_w) + 1;

    const x_len: usize = bsz * h_in * w_in * c_in;
    const w_len: usize = k_h * k_w * c_in * c_out;
    const y_len: usize = bsz * h_out * w_out * c_out;

    const x_buf: []u8 = try allocator.alloc(u8, x_len * 4);
    defer allocator.free(x_buf);
    const w_buf: []u8 = try allocator.alloc(u8, w_len * 4);
    defer allocator.free(w_buf);

    const x_vals: []align(1) f32 = asF32Slice(x_buf);
    const w_vals: []align(1) f32 = asF32Slice(w_buf);
    for (0..x_len) |i| x_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 13))) - 6)) * 0.07;
    for (0..w_len) |i| w_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 17))) - 8)) * 0.03;

    const ref_buf: []u8 = try allocator.alloc(u8, y_len * 4);
    defer allocator.free(ref_buf);
    const ref_vals: []align(1) f32 = asF32Slice(ref_buf);

    const reflectIndex = struct {
        fn f(idx_nom: isize, len: usize) usize {
            var x: isize = idx_nom;
            const l: isize = @intCast(len);
            while (x < 0 or x >= l) {
                if (x < 0) x = -x else x = (2 * l - 2) - x;
            }
            return @intCast(x);
        }
    }.f;

    var b: usize = 0;
    while (b < bsz) : (b += 1) {
        var oh: usize = 0;
        while (oh < h_out) : (oh += 1) {
            var ow: usize = 0;
            while (ow < w_out) : (ow += 1) {
                var oc: usize = 0;
                while (oc < c_out) : (oc += 1) {
                    var acc: f32 = 0.0;
                    var kh: usize = 0;
                    while (kh < k_h) : (kh += 1) {
                        const h_nom: usize = oh * stride_h + kh * dilation_h;
                        const ih: usize = reflectIndex(@as(isize, @intCast(h_nom)) - @as(isize, @intCast(pad_top)), h_in);
                        var kw: usize = 0;
                        while (kw < k_w) : (kw += 1) {
                            const w_nom: usize = ow * stride_w + kw * dilation_w;
                            const iw: usize = reflectIndex(@as(isize, @intCast(w_nom)) - @as(isize, @intCast(pad_left)), w_in);
                            var ic: usize = 0;
                            while (ic < c_in) : (ic += 1) {
                                const x_idx: usize = (((b * h_in + ih) * w_in + iw) * c_in) + ic;
                                const w_idx: usize = (((kh * k_w + kw) * c_in + ic) * c_out) + oc;
                                acc += x_vals[x_idx] * w_vals[w_idx];
                            }
                        }
                    }
                    ref_vals[(((b * h_out + oh) * w_out + ow) * c_out) + oc] = acc;
                }
            }
        }
    }

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();
    const x_tid = try sm.createTiledTensor(.f32, &[_]usize{ bsz, h_in, w_in, c_in }, &[_]usize{ 1, 2, 2, c_in }, .{ .tile_alignment = 64 });
    const w_tid = try sm.createTiledTensor(.f32, &[_]usize{ k_h, k_w, c_in, c_out }, &[_]usize{ k_h, k_w, c_in, 3 }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(x_tid, x_buf);
    try sm.writeFromPackedScalar(w_tid, w_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();
    const x_input = try g.addInput(.f32, &[_]usize{ bsz, h_in, w_in, c_in });
    const w_input = try g.addInput(.f32, &[_]usize{ k_h, k_w, c_in, c_out });
    try g.bindExternal(x_input, @intCast(x_tid));
    try g.bindExternal(w_input, @intCast(w_tid));
    const y = try g.addConv2DWithPadMode(x_input, w_input, null, stride_h, stride_w, dilation_h, dilation_w, pad_top, pad_bottom, pad_left, pad_right, .reflect, groups);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 4, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, y_len * 4);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    var max_abs: f32 = 0.0;
    for (out_vals, ref_vals) |got, want| max_abs = @max(max_abs, @abs(got - want));
    try std.testing.expect(max_abs <= 1e-5);
}

test "cpu backend: conv1d pointwise (NLC) matches reference" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const bsz: usize = 2;
    const l_in: usize = 9;
    const c_in: usize = 6;
    const c_out: usize = 12;
    const groups: usize = 1;
    const k: usize = 1;
    const stride: usize = 1;
    const dilation: usize = 1;
    const pad_left: usize = 0;
    const pad_right: usize = 0;

    const l_out: usize = l_in;

    const x_len: usize = bsz * l_in * c_in;
    const w_len: usize = k * c_in * c_out;
    const b_len: usize = c_out;
    const y_len: usize = bsz * l_out * c_out;

    const x_buf: []u8 = try allocator.alloc(u8, x_len * 4);
    defer allocator.free(x_buf);
    const w_buf: []u8 = try allocator.alloc(u8, w_len * 4);
    defer allocator.free(w_buf);
    const bias_buf: []u8 = try allocator.alloc(u8, b_len * 4);
    defer allocator.free(bias_buf);

    const x_vals: []align(1) f32 = asF32Slice(x_buf);
    const w_vals: []align(1) f32 = asF32Slice(w_buf);
    const bias_vals: []align(1) f32 = asF32Slice(bias_buf);

    for (0..x_len) |i| x_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 19))) - 9)) * 0.07;
    for (0..w_len) |i| w_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 13))) - 6)) * 0.05;
    for (0..b_len) |i| bias_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 5)) * 0.02;

    const ref_buf: []u8 = try allocator.alloc(u8, y_len * 4);
    defer allocator.free(ref_buf);
    const ref_vals: []align(1) f32 = asF32Slice(ref_buf);

    var b: usize = 0;
    while (b < bsz) : (b += 1) {
        var lo: usize = 0;
        while (lo < l_out) : (lo += 1) {
            var oc: usize = 0;
            while (oc < c_out) : (oc += 1) {
                var acc: f32 = bias_vals[oc];
                var ic: usize = 0;
                while (ic < c_in) : (ic += 1) {
                    const x_idx: usize = ((b * l_in + lo) * c_in) + ic;
                    const w_idx: usize = ic * c_out + oc;
                    acc += x_vals[x_idx] * w_vals[w_idx];
                }
                ref_vals[((b * l_out + lo) * c_out) + oc] = acc;
            }
        }
    }

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    // Length-tiling is allowed for pointwise conv1d; channels must be full tiles.
    const x_tid = try sm.createTiledTensor(.f32, &[_]usize{ bsz, l_in, c_in }, &[_]usize{ 1, 4, c_in }, .{ .tile_alignment = 64 });
    const w_tid = try sm.createTiledTensor(.f32, &[_]usize{ k, c_in, c_out }, &[_]usize{ 1, c_in, c_out }, .{ .tile_alignment = 64 });
    const bias_tid = try sm.createTiledTensor(.f32, &[_]usize{c_out}, &[_]usize{c_out}, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(x_tid, x_buf);
    try sm.writeFromPackedScalar(w_tid, w_buf);
    try sm.writeFromPackedScalar(bias_tid, bias_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f32, &[_]usize{ bsz, l_in, c_in });
    const w_in = try g.addInput(.f32, &[_]usize{ k, c_in, c_out });
    const bias_in = try g.addInput(.f32, &[_]usize{c_out});
    try g.bindExternal(x_in, @intCast(x_tid));
    try g.bindExternal(w_in, @intCast(w_tid));
    try g.bindExternal(bias_in, @intCast(bias_tid));

    const y = try g.addConv1D(x_in, w_in, bias_in, stride, dilation, pad_left, pad_right, groups);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 4, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, y_len * 4);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    var max_abs: f32 = 0.0;
    for (out_vals, ref_vals) |got, want| max_abs = @max(max_abs, @abs(got - want));
    try std.testing.expect(max_abs <= 1e-5);
}

test "cpu backend: reduce axis sum/mean matches reference" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const rows: usize = 2;
    const cols: usize = 3;
    const x_len: usize = rows * cols;

    const x_buf: []u8 = try allocator.alloc(u8, x_len * 4);
    defer allocator.free(x_buf);
    const x_vals: []align(1) f32 = asF32Slice(x_buf);
    x_vals[0] = 1.0;
    x_vals[1] = 2.0;
    x_vals[2] = 3.0;
    x_vals[3] = 4.0;
    x_vals[4] = 5.0;
    x_vals[5] = 6.0;

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const x_tid = try sm.createTiledTensor(.f32, &[_]usize{ rows, cols }, &[_]usize{ 2, 2 }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(x_tid, x_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f32, &[_]usize{ rows, cols });
    try g.bindExternal(x_in, @intCast(x_tid));

    const sum_axis0 = try g.addReduceAxis(.sum, x_in, 0);
    const mean_axis_neg1 = try g.addReduceAxis(.mean, x_in, -1);
    try g.setOutputs(&[_]graph_mod.ValueId{ sum_axis0, mean_axis_neg1 });

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 2, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    var out0_buf: [3 * 4]u8 = undefined;
    try sm.readToPackedScalar(prog.outputs[0], out0_buf[0..]);
    const out0: []align(1) f32 = asF32Slice(out0_buf[0..]);
    try std.testing.expectEqual(@as(usize, 3), out0.len);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), out0[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), out0[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), out0[2], 1e-6);

    var out1_buf: [2 * 4]u8 = undefined;
    try sm.readToPackedScalar(prog.outputs[1], out1_buf[0..]);
    const out1: []align(1) f32 = asF32Slice(out1_buf[0..]);
    try std.testing.expectEqual(@as(usize, 2), out1.len);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out1[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), out1[1], 1e-6);
}

test "cpu backend: conv1d general (NLC) supports large c_out" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const bsz: usize = 1;
    const l_in: usize = 5;
    const c_in: usize = 3;
    const c_out: usize = 513;
    const groups: usize = 1;
    const k: usize = 3;
    const stride: usize = 1;
    const dilation: usize = 1;
    const pad_left: usize = 1;
    const pad_right: usize = 1;

    const l_out: usize = ((l_in + pad_left + pad_right - ((k - 1) * dilation + 1)) / stride) + 1;

    const x_len: usize = bsz * l_in * c_in;
    const w_len: usize = k * c_in * c_out;
    const b_len: usize = c_out;
    const y_len: usize = bsz * l_out * c_out;

    const x_buf: []u8 = try allocator.alloc(u8, x_len * 4);
    defer allocator.free(x_buf);
    const w_buf: []u8 = try allocator.alloc(u8, w_len * 4);
    defer allocator.free(w_buf);
    const bias_buf: []u8 = try allocator.alloc(u8, b_len * 4);
    defer allocator.free(bias_buf);

    const x_vals: []align(1) f32 = asF32Slice(x_buf);
    const w_vals: []align(1) f32 = asF32Slice(w_buf);
    const bias_vals: []align(1) f32 = asF32Slice(bias_buf);

    for (0..x_len) |i| x_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 17))) - 8)) * 0.06;
    for (0..w_len) |i| w_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 23))) - 11)) * 0.03;
    for (0..b_len) |i| bias_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 9))) - 4)) * 0.01;

    const ref_buf: []u8 = try allocator.alloc(u8, y_len * 4);
    defer allocator.free(ref_buf);
    const ref_vals: []align(1) f32 = asF32Slice(ref_buf);

    var b: usize = 0;
    while (b < bsz) : (b += 1) {
        var lo: usize = 0;
        while (lo < l_out) : (lo += 1) {
            var oc: usize = 0;
            while (oc < c_out) : (oc += 1) {
                var acc: f32 = bias_vals[oc];
                var kw: usize = 0;
                while (kw < k) : (kw += 1) {
                    const in_nom: usize = lo * stride + kw * dilation;
                    if (in_nom < pad_left) continue;
                    const li: usize = in_nom - pad_left;
                    if (li >= l_in) continue;

                    var ic: usize = 0;
                    while (ic < c_in) : (ic += 1) {
                        const x_idx: usize = ((b * l_in + li) * c_in) + ic;
                        const w_idx: usize = ((kw * c_in + ic) * c_out) + oc;
                        acc += x_vals[x_idx] * w_vals[w_idx];
                    }
                }
                ref_vals[((b * l_out + lo) * c_out) + oc] = acc;
            }
        }
    }

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const x_tid = try sm.createTiledTensor(.f32, &[_]usize{ bsz, l_in, c_in }, &[_]usize{ 1, 3, c_in }, .{ .tile_alignment = 64 });
    const w_tid = try sm.createTiledTensor(.f32, &[_]usize{ k, c_in, c_out }, &[_]usize{ k, c_in, 128 }, .{ .tile_alignment = 64 });
    const bias_tid = try sm.createTiledTensor(.f32, &[_]usize{c_out}, &[_]usize{128}, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(x_tid, x_buf);
    try sm.writeFromPackedScalar(w_tid, w_buf);
    try sm.writeFromPackedScalar(bias_tid, bias_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f32, &[_]usize{ bsz, l_in, c_in });
    const w_in = try g.addInput(.f32, &[_]usize{ k, c_in, c_out });
    const bias_in = try g.addInput(.f32, &[_]usize{c_out});
    try g.bindExternal(x_in, @intCast(x_tid));
    try g.bindExternal(w_in, @intCast(w_tid));
    try g.bindExternal(bias_in, @intCast(bias_tid));

    const y = try g.addConv1D(x_in, w_in, bias_in, stride, dilation, pad_left, pad_right, groups);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 4, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, y_len * 4);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    var max_abs: f32 = 0.0;
    for (out_vals, ref_vals) |got, want| max_abs = @max(max_abs, @abs(got - want));
    try std.testing.expect(max_abs <= 1e-5);
}

test "cpu backend: conv2d pointwise (NHWC) matches reference" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const bsz: usize = 1;
    const h_in: usize = 3;
    const w_in: usize = 4;
    const c_in: usize = 5;
    const c_out: usize = 7;

    const k_h: usize = 1;
    const k_w: usize = 1;
    const groups: usize = 1;
    const stride_h: usize = 1;
    const stride_w: usize = 1;
    const dilation_h: usize = 1;
    const dilation_w: usize = 1;
    const pad_top: usize = 0;
    const pad_bottom: usize = 0;
    const pad_left: usize = 0;
    const pad_right: usize = 0;

    const h_out: usize = h_in;
    const w_out: usize = w_in;

    const x_len: usize = bsz * h_in * w_in * c_in;
    const w_len: usize = k_h * k_w * c_in * c_out;
    const b_len: usize = c_out;
    const y_len: usize = bsz * h_out * w_out * c_out;

    const x_buf: []u8 = try allocator.alloc(u8, x_len * 4);
    defer allocator.free(x_buf);
    const w_buf: []u8 = try allocator.alloc(u8, w_len * 4);
    defer allocator.free(w_buf);
    const bias_buf: []u8 = try allocator.alloc(u8, b_len * 4);
    defer allocator.free(bias_buf);

    const x_vals: []align(1) f32 = asF32Slice(x_buf);
    const w_vals: []align(1) f32 = asF32Slice(w_buf);
    const bias_vals: []align(1) f32 = asF32Slice(bias_buf);

    for (0..x_len) |i| x_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 17))) - 8)) * 0.08;
    for (0..w_len) |i| w_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 11))) - 5)) * 0.04;
    for (0..b_len) |i| bias_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 3)) * 0.03;

    const ref_buf: []u8 = try allocator.alloc(u8, y_len * 4);
    defer allocator.free(ref_buf);
    const ref_vals: []align(1) f32 = asF32Slice(ref_buf);

    var b: usize = 0;
    while (b < bsz) : (b += 1) {
        var h: usize = 0;
        while (h < h_out) : (h += 1) {
            var w0: usize = 0;
            while (w0 < w_out) : (w0 += 1) {
                var oc: usize = 0;
                while (oc < c_out) : (oc += 1) {
                    var acc: f32 = bias_vals[oc];
                    var ic: usize = 0;
                    while (ic < c_in) : (ic += 1) {
                        const x_idx: usize = (((b * h_in + h) * w_in + w0) * c_in) + ic;
                        const w_idx: usize = (((0 * k_w + 0) * c_in + ic) * c_out) + oc;
                        acc += x_vals[x_idx] * w_vals[w_idx];
                    }
                    ref_vals[(((b * h_out + h) * w_out + w0) * c_out) + oc] = acc;
                }
            }
        }
    }

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const x_tid = try sm.createTiledTensor(.f32, &[_]usize{ bsz, h_in, w_in, c_in }, &[_]usize{ 1, 2, 3, 4 }, .{ .tile_alignment = 64 });
    const w_tid = try sm.createTiledTensor(.f32, &[_]usize{ k_h, k_w, c_in, c_out }, &[_]usize{ 1, 1, 3, 4 }, .{ .tile_alignment = 64 });
    const bias_tid = try sm.createTiledTensor(.f32, &[_]usize{c_out}, &[_]usize{4}, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(x_tid, x_buf);
    try sm.writeFromPackedScalar(w_tid, w_buf);
    try sm.writeFromPackedScalar(bias_tid, bias_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f32, &[_]usize{ bsz, h_in, w_in, c_in });
    const w_val = try g.addInput(.f32, &[_]usize{ k_h, k_w, c_in, c_out });
    const bias_in = try g.addInput(.f32, &[_]usize{c_out});
    try g.bindExternal(x_in, @intCast(x_tid));
    try g.bindExternal(w_val, @intCast(w_tid));
    try g.bindExternal(bias_in, @intCast(bias_tid));

    const y = try g.addConv2D(x_in, w_val, bias_in, stride_h, stride_w, dilation_h, dilation_w, pad_top, pad_bottom, pad_left, pad_right, groups);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 4, .base_1d = 8, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, y_len * 4);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    var max_abs: f32 = 0.0;
    for (out_vals, ref_vals) |got, want| max_abs = @max(max_abs, @abs(got - want));
    try std.testing.expect(max_abs <= 1e-5);
}

test "cpu backend: conv2d depthwise (NHWC) matches reference" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const bsz: usize = 2;
    const h_in: usize = 5;
    const w_in: usize = 4;
    const c_in: usize = 3;
    const c_out: usize = 3;
    const groups: usize = c_in;

    const k_h: usize = 3;
    const k_w: usize = 3;
    const stride_h: usize = 1;
    const stride_w: usize = 1;
    const dilation_h: usize = 1;
    const dilation_w: usize = 1;
    const pad_top: usize = 1;
    const pad_bottom: usize = 1;
    const pad_left: usize = 1;
    const pad_right: usize = 1;

    const h_out: usize = ((h_in + pad_top + pad_bottom - ((k_h - 1) * dilation_h + 1)) / stride_h) + 1;
    const w_out: usize = ((w_in + pad_left + pad_right - ((k_w - 1) * dilation_w + 1)) / stride_w) + 1;

    const x_len: usize = bsz * h_in * w_in * c_in;
    const w_len: usize = k_h * k_w * 1 * c_out;
    const b_len: usize = c_out;
    const y_len: usize = bsz * h_out * w_out * c_out;

    const x_buf: []u8 = try allocator.alloc(u8, x_len * 4);
    defer allocator.free(x_buf);
    const w_buf: []u8 = try allocator.alloc(u8, w_len * 4);
    defer allocator.free(w_buf);
    const bias_buf: []u8 = try allocator.alloc(u8, b_len * 4);
    defer allocator.free(bias_buf);

    const x_vals: []align(1) f32 = asF32Slice(x_buf);
    const w_vals: []align(1) f32 = asF32Slice(w_buf);
    const bias_vals: []align(1) f32 = asF32Slice(bias_buf);

    for (0..x_len) |i| x_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 19))) - 9)) * 0.07;
    for (0..w_len) |i| w_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 11))) - 5)) * 0.05;
    for (0..b_len) |i| bias_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 2)) * 0.02;

    const ref_buf: []u8 = try allocator.alloc(u8, y_len * 4);
    defer allocator.free(ref_buf);
    const ref_vals: []align(1) f32 = asF32Slice(ref_buf);

    var b: usize = 0;
    while (b < bsz) : (b += 1) {
        var h: usize = 0;
        while (h < h_out) : (h += 1) {
            var w0: usize = 0;
            while (w0 < w_out) : (w0 += 1) {
                var oc: usize = 0;
                while (oc < c_out) : (oc += 1) {
                    var acc: f32 = bias_vals[oc];
                    var kh: usize = 0;
                    while (kh < k_h) : (kh += 1) {
                        var kw: usize = 0;
                        while (kw < k_w) : (kw += 1) {
                            const pos_h0: usize = h * stride_h + kh * dilation_h;
                            const pos_w0: usize = w0 * stride_w + kw * dilation_w;
                            if (pos_h0 < pad_top or pos_w0 < pad_left) continue;
                            const ih: usize = pos_h0 - pad_top;
                            const iw: usize = pos_w0 - pad_left;
                            if (ih >= h_in or iw >= w_in) continue;

                            const x_idx: usize = (((b * h_in + ih) * w_in + iw) * c_in) + oc;
                            const w_idx: usize = (((kh * k_w + kw) * 1) * c_out) + oc;
                            acc += x_vals[x_idx] * w_vals[w_idx];
                        }
                    }
                    ref_vals[(((b * h_out + h) * w_out + w0) * c_out) + oc] = acc;
                }
            }
        }
    }

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const x_tid = try sm.createTiledTensor(.f32, &[_]usize{ bsz, h_in, w_in, c_in }, &[_]usize{ 1, h_in, w_in, c_in }, .{ .tile_alignment = 64 });
    const w_tid = try sm.createTiledTensor(.f32, &[_]usize{ k_h, k_w, 1, c_out }, &[_]usize{ k_h, k_w, 1, c_out }, .{ .tile_alignment = 64 });
    const bias_tid = try sm.createTiledTensor(.f32, &[_]usize{c_out}, &[_]usize{c_out}, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(x_tid, x_buf);
    try sm.writeFromPackedScalar(w_tid, w_buf);
    try sm.writeFromPackedScalar(bias_tid, bias_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f32, &[_]usize{ bsz, h_in, w_in, c_in });
    const w_val = try g.addInput(.f32, &[_]usize{ k_h, k_w, 1, c_out });
    const bias_in = try g.addInput(.f32, &[_]usize{c_out});
    try g.bindExternal(x_in, @intCast(x_tid));
    try g.bindExternal(w_val, @intCast(w_tid));
    try g.bindExternal(bias_in, @intCast(bias_tid));

    const y = try g.addConv2D(x_in, w_val, bias_in, stride_h, stride_w, dilation_h, dilation_w, pad_top, pad_bottom, pad_left, pad_right, groups);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 4, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, y_len * 4);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    var max_abs: f32 = 0.0;
    for (out_vals, ref_vals) |got, want| max_abs = @max(max_abs, @abs(got - want));
    try std.testing.expect(max_abs <= 1e-5);
}

test "cpu backend: conv2d depthwise reflect padding matches reference" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const bsz: usize = 1;
    const h_in: usize = 4;
    const w_in: usize = 5;
    const c_in: usize = 4;
    const c_out: usize = 4;
    const groups: usize = c_in;

    const k_h: usize = 3;
    const k_w: usize = 3;
    const stride_h: usize = 1;
    const stride_w: usize = 1;
    const dilation_h: usize = 1;
    const dilation_w: usize = 1;
    const pad_top: usize = 1;
    const pad_bottom: usize = 2;
    const pad_left: usize = 2;
    const pad_right: usize = 1;

    const h_out: usize = ((h_in + pad_top + pad_bottom - ((k_h - 1) * dilation_h + 1)) / stride_h) + 1;
    const w_out: usize = ((w_in + pad_left + pad_right - ((k_w - 1) * dilation_w + 1)) / stride_w) + 1;

    const x_len: usize = bsz * h_in * w_in * c_in;
    const w_len: usize = k_h * k_w * 1 * c_out;
    const b_len: usize = c_out;
    const y_len: usize = bsz * h_out * w_out * c_out;

    const x_buf: []u8 = try allocator.alloc(u8, x_len * 4);
    defer allocator.free(x_buf);
    const w_buf: []u8 = try allocator.alloc(u8, w_len * 4);
    defer allocator.free(w_buf);
    const bias_buf: []u8 = try allocator.alloc(u8, b_len * 4);
    defer allocator.free(bias_buf);

    const x_vals: []align(1) f32 = asF32Slice(x_buf);
    const w_vals: []align(1) f32 = asF32Slice(w_buf);
    const bias_vals: []align(1) f32 = asF32Slice(bias_buf);

    for (0..x_len) |i| x_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 21))) - 10)) * 0.06;
    for (0..w_len) |i| w_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 13))) - 6)) * 0.05;
    for (0..b_len) |i| bias_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 2)) * 0.02;

    const ref_buf: []u8 = try allocator.alloc(u8, y_len * 4);
    defer allocator.free(ref_buf);
    const ref_vals: []align(1) f32 = asF32Slice(ref_buf);

    const reflectIndex = struct {
        fn f(idx_nom: isize, len: usize) usize {
            var x: isize = idx_nom;
            const l: isize = @intCast(len);
            while (x < 0 or x >= l) {
                if (x < 0) x = -x else x = (2 * l - 2) - x;
            }
            return @intCast(x);
        }
    }.f;

    var b: usize = 0;
    while (b < bsz) : (b += 1) {
        var oh: usize = 0;
        while (oh < h_out) : (oh += 1) {
            var ow: usize = 0;
            while (ow < w_out) : (ow += 1) {
                var oc: usize = 0;
                while (oc < c_out) : (oc += 1) {
                    var acc: f32 = bias_vals[oc];
                    var kh: usize = 0;
                    while (kh < k_h) : (kh += 1) {
                        const h_nom: usize = oh * stride_h + kh * dilation_h;
                        const ih: usize = reflectIndex(@as(isize, @intCast(h_nom)) - @as(isize, @intCast(pad_top)), h_in);
                        var kw: usize = 0;
                        while (kw < k_w) : (kw += 1) {
                            const w_nom: usize = ow * stride_w + kw * dilation_w;
                            const iw: usize = reflectIndex(@as(isize, @intCast(w_nom)) - @as(isize, @intCast(pad_left)), w_in);
                            const x_idx: usize = (((b * h_in + ih) * w_in + iw) * c_in) + oc;
                            const w_idx: usize = (((kh * k_w + kw) * 1) * c_out) + oc;
                            acc += x_vals[x_idx] * w_vals[w_idx];
                        }
                    }
                    ref_vals[(((b * h_out + oh) * w_out + ow) * c_out) + oc] = acc;
                }
            }
        }
    }

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const x_tid = try sm.createTiledTensor(.f32, &[_]usize{ bsz, h_in, w_in, c_in }, &[_]usize{ 1, 2, 2, 2 }, .{ .tile_alignment = 64 });
    const w_tid = try sm.createTiledTensor(.f32, &[_]usize{ k_h, k_w, 1, c_out }, &[_]usize{ 2, 2, 1, 2 }, .{ .tile_alignment = 64 });
    const bias_tid = try sm.createTiledTensor(.f32, &[_]usize{c_out}, &[_]usize{2}, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(x_tid, x_buf);
    try sm.writeFromPackedScalar(w_tid, w_buf);
    try sm.writeFromPackedScalar(bias_tid, bias_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f32, &[_]usize{ bsz, h_in, w_in, c_in });
    const w_val = try g.addInput(.f32, &[_]usize{ k_h, k_w, 1, c_out });
    const bias_in = try g.addInput(.f32, &[_]usize{c_out});
    try g.bindExternal(x_in, @intCast(x_tid));
    try g.bindExternal(w_val, @intCast(w_tid));
    try g.bindExternal(bias_in, @intCast(bias_tid));

    const y = try g.addConv2DWithPadMode(x_in, w_val, bias_in, stride_h, stride_w, dilation_h, dilation_w, pad_top, pad_bottom, pad_left, pad_right, .reflect, groups);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 2, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, y_len * 4);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    var max_abs: f32 = 0.0;
    for (out_vals, ref_vals) |got, want| max_abs = @max(max_abs, @abs(got - want));
    try std.testing.expect(max_abs <= 1e-5);
}

test "cpu backend: conv2d general (NHWC) supports large c_out" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const bsz: usize = 1;
    const h_in: usize = 4;
    const w_in: usize = 4;
    const c_in: usize = 2;
    const c_out: usize = 513;
    const groups: usize = 1;

    const k_h: usize = 3;
    const k_w: usize = 3;
    const stride_h: usize = 1;
    const stride_w: usize = 1;
    const dilation_h: usize = 1;
    const dilation_w: usize = 1;
    const pad_top: usize = 1;
    const pad_bottom: usize = 1;
    const pad_left: usize = 1;
    const pad_right: usize = 1;

    const h_out: usize = ((h_in + pad_top + pad_bottom - ((k_h - 1) * dilation_h + 1)) / stride_h) + 1;
    const w_out: usize = ((w_in + pad_left + pad_right - ((k_w - 1) * dilation_w + 1)) / stride_w) + 1;

    const x_len: usize = bsz * h_in * w_in * c_in;
    const w_len: usize = k_h * k_w * c_in * c_out;
    const b_len: usize = c_out;
    const y_len: usize = bsz * h_out * w_out * c_out;

    const x_buf: []u8 = try allocator.alloc(u8, x_len * 4);
    defer allocator.free(x_buf);
    const w_buf: []u8 = try allocator.alloc(u8, w_len * 4);
    defer allocator.free(w_buf);
    const bias_buf: []u8 = try allocator.alloc(u8, b_len * 4);
    defer allocator.free(bias_buf);

    const x_vals: []align(1) f32 = asF32Slice(x_buf);
    const w_vals: []align(1) f32 = asF32Slice(w_buf);
    const bias_vals: []align(1) f32 = asF32Slice(bias_buf);

    for (0..x_len) |i| x_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 17))) - 8)) * 0.06;
    for (0..w_len) |i| w_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 23))) - 11)) * 0.03;
    for (0..b_len) |i| bias_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 9))) - 4)) * 0.01;

    const ref_buf: []u8 = try allocator.alloc(u8, y_len * 4);
    defer allocator.free(ref_buf);
    const ref_vals: []align(1) f32 = asF32Slice(ref_buf);

    var b: usize = 0;
    while (b < bsz) : (b += 1) {
        var h: usize = 0;
        while (h < h_out) : (h += 1) {
            var w0: usize = 0;
            while (w0 < w_out) : (w0 += 1) {
                var oc: usize = 0;
                while (oc < c_out) : (oc += 1) {
                    var acc: f32 = bias_vals[oc];
                    var kh: usize = 0;
                    while (kh < k_h) : (kh += 1) {
                        var kw: usize = 0;
                        while (kw < k_w) : (kw += 1) {
                            const pos_h0: usize = h * stride_h + kh * dilation_h;
                            const pos_w0: usize = w0 * stride_w + kw * dilation_w;
                            if (pos_h0 < pad_top or pos_w0 < pad_left) continue;
                            const ih: usize = pos_h0 - pad_top;
                            const iw: usize = pos_w0 - pad_left;
                            if (ih >= h_in or iw >= w_in) continue;

                            var ic: usize = 0;
                            while (ic < c_in) : (ic += 1) {
                                const x_idx: usize = (((b * h_in + ih) * w_in + iw) * c_in) + ic;
                                const w_idx: usize = (((kh * k_w + kw) * c_in + ic) * c_out) + oc;
                                acc += x_vals[x_idx] * w_vals[w_idx];
                            }
                        }
                    }
                    ref_vals[(((b * h_out + h) * w_out + w0) * c_out) + oc] = acc;
                }
            }
        }
    }

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const x_tid = try sm.createTiledTensor(.f32, &[_]usize{ bsz, h_in, w_in, c_in }, &[_]usize{ 1, 2, 2, c_in }, .{ .tile_alignment = 64 });
    const w_tid = try sm.createTiledTensor(.f32, &[_]usize{ k_h, k_w, c_in, c_out }, &[_]usize{ k_h, k_w, c_in, 128 }, .{ .tile_alignment = 64 });
    const bias_tid = try sm.createTiledTensor(.f32, &[_]usize{c_out}, &[_]usize{128}, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(x_tid, x_buf);
    try sm.writeFromPackedScalar(w_tid, w_buf);
    try sm.writeFromPackedScalar(bias_tid, bias_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f32, &[_]usize{ bsz, h_in, w_in, c_in });
    const w_val = try g.addInput(.f32, &[_]usize{ k_h, k_w, c_in, c_out });
    const bias_in = try g.addInput(.f32, &[_]usize{c_out});
    try g.bindExternal(x_in, @intCast(x_tid));
    try g.bindExternal(w_val, @intCast(w_tid));
    try g.bindExternal(bias_in, @intCast(bias_tid));

    const y = try g.addConv2D(x_in, w_val, bias_in, stride_h, stride_w, dilation_h, dilation_w, pad_top, pad_bottom, pad_left, pad_right, groups);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 4, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, y_len * 4);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    var max_abs: f32 = 0.0;
    for (out_vals, ref_vals) |got, want| max_abs = @max(max_abs, @abs(got - want));
    try std.testing.expect(max_abs <= 1e-5);
}

test "cpu backend: kv cache append mutates cache in-place (f32)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const bsz: usize = 2;
    const heads: usize = 2;
    const t_cap: usize = 6;
    const d_head: usize = 3;
    const append_len: usize = 2;

    const cache_elems: usize = bsz * heads * t_cap * d_head;
    const new_elems: usize = bsz * heads * append_len * d_head;

    const cache_buf: []u8 = try allocator.alloc(u8, cache_elems * @sizeOf(f32));
    defer allocator.free(cache_buf);
    const new_buf: []u8 = try allocator.alloc(u8, new_elems * @sizeOf(f32));
    defer allocator.free(new_buf);
    const end_buf: []u8 = try allocator.alloc(u8, bsz * @sizeOf(i32));
    defer allocator.free(end_buf);

    const cache_vals: []align(1) f32 = asF32Slice(cache_buf);
    const new_vals: []align(1) f32 = asF32Slice(new_buf);
    const end_ptr: [*]align(1) i32 = @ptrCast(end_buf.ptr);
    const end_vals: []align(1) i32 = end_ptr[0 .. end_buf.len / @sizeOf(i32)];

    var b: usize = 0;
    while (b < bsz) : (b += 1) {
        var h: usize = 0;
        while (h < heads) : (h += 1) {
            var t: usize = 0;
            while (t < t_cap) : (t += 1) {
                var d: usize = 0;
                while (d < d_head) : (d += 1) {
                    const idx: usize = (((b * heads + h) * t_cap + t) * d_head) + d;
                    cache_vals[idx] = @floatFromInt((@as(i32, @intCast(b)) * 1000) + (@as(i32, @intCast(h)) * 100) + (@as(i32, @intCast(t)) * 10) + @as(i32, @intCast(d)));
                }
            }
        }
    }

    b = 0;
    while (b < bsz) : (b += 1) {
        var h: usize = 0;
        while (h < heads) : (h += 1) {
            var t: usize = 0;
            while (t < append_len) : (t += 1) {
                var d: usize = 0;
                while (d < d_head) : (d += 1) {
                    const idx: usize = (((b * heads + h) * append_len + t) * d_head) + d;
                    new_vals[idx] = @floatFromInt(5000 + (@as(i32, @intCast(b)) * 1000) + (@as(i32, @intCast(h)) * 100) + (@as(i32, @intCast(t)) * 10) + @as(i32, @intCast(d)));
                }
            }
        }
    }

    end_vals[0] = 1;
    end_vals[1] = 3;

    const expected: []f32 = try allocator.alloc(f32, cache_elems);
    defer allocator.free(expected);
    @memcpy(expected, cache_vals);

    b = 0;
    while (b < bsz) : (b += 1) {
        const t_start: usize = @intCast(end_vals[b]);
        var h: usize = 0;
        while (h < heads) : (h += 1) {
            var t: usize = 0;
            while (t < append_len) : (t += 1) {
                const dst_t: usize = t_start + t;
                var d: usize = 0;
                while (d < d_head) : (d += 1) {
                    const src_idx: usize = (((b * append_len + t) * heads + h) * d_head) + d;
                    const dst_idx: usize = (((b * t_cap + dst_t) * heads + h) * d_head) + d;
                    expected[dst_idx] = new_vals[src_idx];
                }
            }
        }
    }

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    // v1 constraints: single tile over batch+heads, full head-dim tile.
    const cache_tid = try sm.createTiledTensor(.f32, &[_]usize{ bsz, t_cap, heads, d_head }, &[_]usize{ bsz, 3, heads, d_head }, .{ .tile_alignment = 64 });
    const new_tid = try sm.createTiledTensor(.f32, &[_]usize{ bsz, append_len, heads, d_head }, &[_]usize{ bsz, 1, heads, d_head }, .{ .tile_alignment = 64 });
    const end_tid = try sm.createTiledTensor(.i32, &[_]usize{bsz}, &[_]usize{bsz}, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(cache_tid, cache_buf);
    try sm.writeFromPackedScalar(new_tid, new_buf);
    try sm.writeFromPackedScalar(end_tid, end_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const cache_in = try g.addInput(.f32, &[_]usize{ bsz, t_cap, heads, d_head });
    const new_in = try g.addInput(.f32, &[_]usize{ bsz, append_len, heads, d_head });
    const end_in = try g.addInput(.i32, &[_]usize{bsz});
    try g.bindExternal(cache_in, @intCast(cache_tid));
    try g.bindExternal(new_in, @intCast(new_tid));
    try g.bindExternal(end_in, @intCast(end_tid));

    const out = try g.addSequenceAppend(cache_in, new_in, end_in);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 4, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    // In-place alias contract: output tensor id is cache tensor id.
    try std.testing.expectEqual(cache_tid, prog.outputs[0]);

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, cache_elems * @sizeOf(f32));
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    var i: usize = 0;
    while (i < cache_elems) : (i += 1) {
        try std.testing.expectApproxEqAbs(expected[i], out_vals[i], 0.0);
    }
}

test "cpu backend: kv cache append rejects out-of-bounds end index" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const bsz: usize = 1;
    const heads: usize = 1;
    const t_cap: usize = 4;
    const d_head: usize = 2;
    const append_len: usize = 2;

    const cache_elems: usize = bsz * heads * t_cap * d_head;
    const new_elems: usize = bsz * heads * append_len * d_head;

    const cache_buf: []u8 = try allocator.alloc(u8, cache_elems * @sizeOf(f32));
    defer allocator.free(cache_buf);
    const new_buf: []u8 = try allocator.alloc(u8, new_elems * @sizeOf(f32));
    defer allocator.free(new_buf);
    const end_buf: []u8 = try allocator.alloc(u8, bsz * @sizeOf(i32));
    defer allocator.free(end_buf);

    const cache_vals: []align(1) f32 = asF32Slice(cache_buf);
    const new_vals: []align(1) f32 = asF32Slice(new_buf);
    for (cache_vals, 0..) |*v, i| v.* = @floatFromInt(i);
    for (new_vals, 0..) |*v, i| v.* = @floatFromInt(100 + i);

    const end_ptr: [*]align(1) i32 = @ptrCast(end_buf.ptr);
    end_ptr[0] = 3; // 3 + append_len(2) > t_cap(4)

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const cache_tid = try sm.createTiledTensor(.f32, &[_]usize{ bsz, t_cap, heads, d_head }, &[_]usize{ bsz, 2, heads, d_head }, .{ .tile_alignment = 64 });
    const new_tid = try sm.createTiledTensor(.f32, &[_]usize{ bsz, append_len, heads, d_head }, &[_]usize{ bsz, 1, heads, d_head }, .{ .tile_alignment = 64 });
    const end_tid = try sm.createTiledTensor(.i32, &[_]usize{bsz}, &[_]usize{bsz}, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(cache_tid, cache_buf);
    try sm.writeFromPackedScalar(new_tid, new_buf);
    try sm.writeFromPackedScalar(end_tid, end_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const cache_in = try g.addInput(.f32, &[_]usize{ bsz, t_cap, heads, d_head });
    const new_in = try g.addInput(.f32, &[_]usize{ bsz, append_len, heads, d_head });
    const end_in = try g.addInput(.i32, &[_]usize{bsz});
    try g.bindExternal(cache_in, @intCast(cache_tid));
    try g.bindExternal(new_in, @intCast(new_tid));
    try g.bindExternal(end_in, @intCast(end_tid));

    const out = try g.addSequenceAppend(cache_in, new_in, end_in);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 2, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try std.testing.expectError(types.BackendError.InvalidArgument, cpu.backend().executeProgram(&prog, sm.tensorStore()));
}

test "cpu backend: kv cache append ring policy wraps time index" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const bsz: usize = 1;
    const heads: usize = 1;
    const t_cap: usize = 4;
    const d_head: usize = 1;
    const append_len: usize = 2;

    const cache_buf: []u8 = try allocator.alloc(u8, bsz * heads * t_cap * d_head * @sizeOf(f32));
    defer allocator.free(cache_buf);
    const new_buf: []u8 = try allocator.alloc(u8, bsz * heads * append_len * d_head * @sizeOf(f32));
    defer allocator.free(new_buf);
    const end_buf: []u8 = try allocator.alloc(u8, bsz * @sizeOf(i32));
    defer allocator.free(end_buf);

    const cache_vals: []align(1) f32 = asF32Slice(cache_buf);
    const new_vals: []align(1) f32 = asF32Slice(new_buf);
    cache_vals[0] = 0.0;
    cache_vals[1] = 1.0;
    cache_vals[2] = 2.0;
    cache_vals[3] = 3.0;
    new_vals[0] = 90.0;
    new_vals[1] = 91.0;

    const end_ptr: [*]align(1) i32 = @ptrCast(end_buf.ptr);
    end_ptr[0] = 3;

    var sm: manager_mod.StorageManager = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();
    try sm.configureCache(.{ .ram_budget_bytes = 1 << 20 });

    const cache_tid: manager_mod.TensorId = try sm.createTiledTensor(.f32, &[_]usize{ bsz, t_cap, heads, d_head }, &[_]usize{ bsz, 2, heads, d_head }, .{ .tile_alignment = 64 });
    const new_tid: manager_mod.TensorId = try sm.createTiledTensor(.f32, &[_]usize{ bsz, append_len, heads, d_head }, &[_]usize{ bsz, 1, heads, d_head }, .{ .tile_alignment = 64 });
    const end_tid: manager_mod.TensorId = try sm.createTiledTensor(.i32, &[_]usize{bsz}, &[_]usize{bsz}, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(cache_tid, cache_buf);
    try sm.writeFromPackedScalar(new_tid, new_buf);
    try sm.writeFromPackedScalar(end_tid, end_buf);
    try sm.registerSequenceCachePolicy(cache_tid, .{ .ring = .{ .window_tokens = t_cap } });

    var g: graph_mod.Graph = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const cache_in: graph_mod.ValueId = try g.addInput(.f32, &[_]usize{ bsz, t_cap, heads, d_head });
    const new_in: graph_mod.ValueId = try g.addInput(.f32, &[_]usize{ bsz, append_len, heads, d_head });
    const end_in: graph_mod.ValueId = try g.addInput(.i32, &[_]usize{bsz});
    try g.bindExternal(cache_in, @intCast(cache_tid));
    try g.bindExternal(new_in, @intCast(new_tid));
    try g.bindExternal(end_in, @intCast(end_tid));

    const out: graph_mod.ValueId = try g.addSequenceAppend(cache_in, new_in, end_in);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 2, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    var cpu: cpu_backend_mod.CpuBackend = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, bsz * heads * t_cap * d_head * @sizeOf(f32));
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    try std.testing.expectEqual(@as(f32, 91.0), out_vals[0]);
    try std.testing.expectEqual(@as(f32, 1.0), out_vals[1]);
    try std.testing.expectEqual(@as(f32, 2.0), out_vals[2]);
    try std.testing.expectEqual(@as(f32, 90.0), out_vals[3]);
}

test "cpu backend: kv cache append growable policy expands physical capacity" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const bsz: usize = 1;
    const heads: usize = 1;
    const t_cap: usize = 4;
    const d_head: usize = 1;
    const append_len: usize = 2;

    const cache_buf: []u8 = try allocator.alloc(u8, bsz * heads * t_cap * d_head * @sizeOf(f32));
    defer allocator.free(cache_buf);
    const new_buf: []u8 = try allocator.alloc(u8, bsz * heads * append_len * d_head * @sizeOf(f32));
    defer allocator.free(new_buf);
    const end_buf: []u8 = try allocator.alloc(u8, bsz * @sizeOf(i32));
    defer allocator.free(end_buf);

    const cache_vals: []align(1) f32 = asF32Slice(cache_buf);
    const new_vals: []align(1) f32 = asF32Slice(new_buf);
    cache_vals[0] = 0.0;
    cache_vals[1] = 1.0;
    cache_vals[2] = 2.0;
    cache_vals[3] = 3.0;
    new_vals[0] = 90.0;
    new_vals[1] = 91.0;

    const end_ptr: [*]align(1) i32 = @ptrCast(end_buf.ptr);
    end_ptr[0] = 5;

    var sm: manager_mod.StorageManager = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();
    try sm.configureCache(.{ .ram_budget_bytes = 1 << 20 });

    const cache_tid: manager_mod.TensorId = try sm.createTiledTensor(.f32, &[_]usize{ bsz, t_cap, heads, d_head }, &[_]usize{ bsz, 2, heads, d_head }, .{ .tile_alignment = 64 });
    const new_tid: manager_mod.TensorId = try sm.createTiledTensor(.f32, &[_]usize{ bsz, append_len, heads, d_head }, &[_]usize{ bsz, 1, heads, d_head }, .{ .tile_alignment = 64 });
    const end_tid: manager_mod.TensorId = try sm.createTiledTensor(.i32, &[_]usize{bsz}, &[_]usize{bsz}, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(cache_tid, cache_buf);
    try sm.writeFromPackedScalar(new_tid, new_buf);
    try sm.writeFromPackedScalar(end_tid, end_buf);
    try sm.registerSequenceCachePolicy(cache_tid, .{ .growable = .{ .initial_capacity_tokens = 2, .growth_numerator = 2, .growth_denominator = 1 } });

    var g: graph_mod.Graph = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const cache_in: graph_mod.ValueId = try g.addInput(.f32, &[_]usize{ bsz, t_cap, heads, d_head });
    const new_in: graph_mod.ValueId = try g.addInput(.f32, &[_]usize{ bsz, append_len, heads, d_head });
    const end_in: graph_mod.ValueId = try g.addInput(.i32, &[_]usize{bsz});
    try g.bindExternal(cache_in, @intCast(cache_tid));
    try g.bindExternal(new_in, @intCast(new_tid));
    try g.bindExternal(end_in, @intCast(end_tid));

    const out: graph_mod.ValueId = try g.addSequenceAppend(cache_in, new_in, end_in);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 2, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    var cpu: cpu_backend_mod.CpuBackend = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    const grown_meta: *const manager_mod.TiledTensor = try sm.getConst(cache_tid);
    try std.testing.expectEqual(@as(usize, 8), grown_meta.shape[1]);

    const out_buf: []u8 = try allocator.alloc(u8, bsz * heads * 8 * d_head * @sizeOf(f32));
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    try std.testing.expectEqual(@as(f32, 0.0), out_vals[0]);
    try std.testing.expectEqual(@as(f32, 1.0), out_vals[1]);
    try std.testing.expectEqual(@as(f32, 2.0), out_vals[2]);
    try std.testing.expectEqual(@as(f32, 3.0), out_vals[3]);
    try std.testing.expectEqual(@as(f32, 0.0), out_vals[4]);
    try std.testing.expectEqual(@as(f32, 90.0), out_vals[5]);
    try std.testing.expectEqual(@as(f32, 91.0), out_vals[6]);
    try std.testing.expectEqual(@as(f32, 0.0), out_vals[7]);
}

/// Is key `k_pos` visible to a query at `q_pos`? Written per pair, independent of
/// the interval arithmetic the kernels use, so the two can disagree.
fn windowAllows(w: graph_mod.AttentionWindow, q_pos: usize, k_pos: usize) bool {
    const anchor: usize = if (w.chunk > 0) (q_pos / w.chunk) * w.chunk else q_pos;
    const span: usize = if (w.chunk > 0) w.chunk else 1;
    const first: usize = anchor - @min(anchor, @as(usize, w.left));
    const last: usize = anchor +| span +| @as(usize, w.right);
    return k_pos >= first and k_pos < last;
}

fn cachedGqaRefF32(
    out: []f32,
    q: []align(1) const f32,
    k_cache: []align(1) const f32,
    v_cache: []align(1) const f32,
    positions: []align(1) const i32,
    end_index: []align(1) const i32,
    bsz: usize,
    l_q: usize,
    h_q: usize,
    h_kv: usize,
    t_cap: usize,
    d_k: usize,
    d_v: usize,
    scale: f32,
    window: graph_mod.AttentionWindow,
    attn_logits_soft_cap: f32,
) void {
    std.debug.assert(out.len == bsz * l_q * h_q * d_v);
    std.debug.assert(q.len == bsz * l_q * h_q * d_k);
    std.debug.assert(k_cache.len == bsz * h_kv * t_cap * d_k);
    std.debug.assert(v_cache.len == bsz * h_kv * t_cap * d_v);
    std.debug.assert(positions.len == bsz * l_q);
    std.debug.assert(end_index.len == bsz);
    std.debug.assert((h_q % h_kv) == 0);

    const groups_per_kv: usize = h_q / h_kv;

    var b: usize = 0;
    while (b < bsz) : (b += 1) {
        const valid_end: usize = @intCast(@max(end_index[b], 0));

        var l: usize = 0;
        while (l < l_q) : (l += 1) {
            const q_pos: usize = @intCast(@max(positions[b * l_q + l], 0));

            const lower: usize = 0;
            const upper: usize = valid_end;

            var hq_idx: usize = 0;
            while (hq_idx < h_q) : (hq_idx += 1) {
                const out_base: usize = (((b * l_q + l) * h_q + hq_idx) * d_v);
                @memset(out[out_base .. out_base + d_v], 0.0);

                if (lower >= upper) continue;

                const hkv_idx: usize = hq_idx / groups_per_kv;

                const q_base: usize = (((b * l_q + l) * h_q + hq_idx) * d_k);
                const q_row: []align(1) const f32 = q[q_base .. q_base + d_k];

                var m_state: f32 = -std.math.inf(f32);
                var l_state: f32 = 0.0;

                var t: usize = lower;
                while (t < upper) : (t += 1) {
                    if (!windowAllows(window, q_pos, t)) continue;
                    const k_base: usize = (((b * t_cap + t) * h_kv + hkv_idx) * d_k);
                    const v_base: usize = (((b * t_cap + t) * h_kv + hkv_idx) * d_v);

                    var dot: f32 = 0.0;
                    var dk_i: usize = 0;
                    while (dk_i < d_k) : (dk_i += 1) {
                        dot += q_row[dk_i] * k_cache[k_base + dk_i];
                    }

                    var logit: f32 = dot * scale;
                    if (attn_logits_soft_cap > 0.0) {
                        const x: f64 = @as(f64, @floatCast(logit / attn_logits_soft_cap));
                        logit = @floatCast(@as(f64, @floatCast(attn_logits_soft_cap)) * std.math.tanh(x));
                    }

                    const m_new: f32 = @max(m_state, logit);
                    const rescale: f32 = if (m_state == -std.math.inf(f32))
                        0.0
                    else
                        @floatCast(std.math.exp(@as(f64, @floatCast(m_state - m_new))));
                    const p_new: f32 = @floatCast(std.math.exp(@as(f64, @floatCast(logit - m_new))));

                    var dv_i: usize = 0;
                    while (dv_i < d_v) : (dv_i += 1) {
                        out[out_base + dv_i] = out[out_base + dv_i] * rescale + v_cache[v_base + dv_i] * p_new;
                    }

                    l_state = l_state * rescale + p_new;
                    m_state = m_new;
                }

                if (l_state > 0.0) {
                    const inv: f32 = 1.0 / l_state;
                    var dv_i: usize = 0;
                    while (dv_i < d_v) : (dv_i += 1) {
                        out[out_base + dv_i] *= inv;
                    }
                }
            }
        }
    }
}

test "cpu backend: cached grouped-query attention matches reference (f32)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const bsz: usize = 1;
    const l_q: usize = 2;
    const h_q: usize = 4;
    const h_kv: usize = 2;
    const t_cap: usize = 6;
    const d_k: usize = 3;
    const d_v: usize = 2;

    const scale: f32 = 0.5;
    const window: graph_mod.AttentionWindow = .sliding(2, 0);
    const softcap: f32 = 1.75;

    const q_buf: []u8 = try allocator.alloc(u8, bsz * l_q * h_q * d_k * @sizeOf(f32));
    defer allocator.free(q_buf);
    const k_buf: []u8 = try allocator.alloc(u8, bsz * h_kv * t_cap * d_k * @sizeOf(f32));
    defer allocator.free(k_buf);
    const v_buf: []u8 = try allocator.alloc(u8, bsz * h_kv * t_cap * d_v * @sizeOf(f32));
    defer allocator.free(v_buf);
    const pos_buf: []u8 = try allocator.alloc(u8, bsz * l_q * @sizeOf(i32));
    defer allocator.free(pos_buf);
    const end_buf: []u8 = try allocator.alloc(u8, bsz * @sizeOf(i32));
    defer allocator.free(end_buf);

    const q_vals: []align(1) f32 = asF32Slice(q_buf);
    const k_vals: []align(1) f32 = asF32Slice(k_buf);
    const v_vals: []align(1) f32 = asF32Slice(v_buf);

    for (q_vals, 0..) |*v0, i| v0.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8))) * 0.07;
    for (k_vals, 0..) |*v0, i| v0.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 19)) - 9))) * 0.05;
    for (v_vals, 0..) |*v0, i| v0.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 23)) - 11))) * 0.06;

    const pos_ptr: [*]align(1) i32 = @ptrCast(pos_buf.ptr);
    const pos_vals: []align(1) i32 = pos_ptr[0 .. pos_buf.len / @sizeOf(i32)];
    pos_vals[0] = 4;
    pos_vals[1] = 5;

    const end_ptr: [*]align(1) i32 = @ptrCast(end_buf.ptr);
    const end_vals: []align(1) i32 = end_ptr[0 .. end_buf.len / @sizeOf(i32)];
    end_vals[0] = 6;

    const ref_vals: []f32 = try allocator.alloc(f32, bsz * l_q * h_q * d_v);
    defer allocator.free(ref_vals);
    cachedGqaRefF32(
        ref_vals,
        q_vals,
        k_vals,
        v_vals,
        pos_vals,
        end_vals,
        bsz,
        l_q,
        h_q,
        h_kv,
        t_cap,
        d_k,
        d_v,
        scale,
        window,
        softcap,
    );

    var sm: manager_mod.StorageManager = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const q_tid: manager_mod.TensorId = try sm.createTiledTensor(.f32, &[_]usize{ bsz, l_q, h_q, d_k }, &[_]usize{ 1, 1, 2, d_k }, .{ .tile_alignment = 64 });
    const k_tid: manager_mod.TensorId = try sm.createTiledTensor(.f32, &[_]usize{ bsz, t_cap, h_kv, d_k }, &[_]usize{ 1, 2, 1, d_k }, .{ .tile_alignment = 64 });
    const v_tid: manager_mod.TensorId = try sm.createTiledTensor(.f32, &[_]usize{ bsz, t_cap, h_kv, d_v }, &[_]usize{ 1, 2, 1, d_v }, .{ .tile_alignment = 64 });
    const pos_tid: manager_mod.TensorId = try sm.createTiledTensor(.i32, &[_]usize{ bsz, l_q }, &[_]usize{ 1, 1 }, .{ .tile_alignment = 64 });
    const end_tid: manager_mod.TensorId = try sm.createTiledTensor(.i32, &[_]usize{bsz}, &[_]usize{bsz}, .{ .tile_alignment = 64 });

    try sm.writeFromPackedScalar(q_tid, q_buf);
    try sm.writeFromPackedScalar(k_tid, k_buf);
    try sm.writeFromPackedScalar(v_tid, v_buf);
    try sm.writeFromPackedScalar(pos_tid, pos_buf);
    try sm.writeFromPackedScalar(end_tid, end_buf);

    var g: graph_mod.Graph = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const q_in: graph_mod.ValueId = try g.addInput(.f32, &[_]usize{ bsz, l_q, h_q, d_k });
    const k_in: graph_mod.ValueId = try g.addInput(.f32, &[_]usize{ bsz, t_cap, h_kv, d_k });
    const v_in: graph_mod.ValueId = try g.addInput(.f32, &[_]usize{ bsz, t_cap, h_kv, d_v });
    const pos_in: graph_mod.ValueId = try g.addInput(.i32, &[_]usize{ bsz, l_q });
    const end_in: graph_mod.ValueId = try g.addInput(.i32, &[_]usize{bsz});

    try g.bindExternal(q_in, @intCast(q_tid));
    try g.bindExternal(k_in, @intCast(k_tid));
    try g.bindExternal(v_in, @intCast(v_tid));
    try g.bindExternal(pos_in, @intCast(pos_tid));
    try g.bindExternal(end_in, @intCast(end_tid));

    const out: graph_mod.ValueId = try g.addAttention(
        q_in,
        k_in,
        v_in,
        pos_in,
        end_in,
        scale,
        window,
        softcap,
    );
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 4, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    var cpu: cpu_backend_mod.CpuBackend = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, bsz * l_q * h_q * d_v * @sizeOf(f32));
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    var max_abs: f32 = 0.0;
    for (out_vals, ref_vals) |got, want| {
        try std.testing.expect(std.math.isFinite(got));
        max_abs = @max(max_abs, @abs(got - want));
    }
    try std.testing.expect(max_abs <= 8e-2);
}

test "cpu backend: cached grouped-query attention supports q=f32, kv=f16 with f32 output" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const bsz: usize = 1;
    const l_q: usize = 2;
    const h_q: usize = 4;
    const h_kv: usize = 2;
    const t_cap: usize = 6;
    const d_k: usize = 4;
    const d_v: usize = 3;

    const scale: f32 = 0.5;
    const window: graph_mod.AttentionWindow = .sliding(3, 0);
    const softcap: f32 = 0.0;

    const q_buf: []u8 = try allocator.alloc(u8, bsz * l_q * h_q * d_k * @sizeOf(f32));
    defer allocator.free(q_buf);
    const k_buf: []u8 = try allocator.alloc(u8, bsz * h_kv * t_cap * d_k * @sizeOf(f16));
    defer allocator.free(k_buf);
    const v_buf: []u8 = try allocator.alloc(u8, bsz * h_kv * t_cap * d_v * @sizeOf(f16));
    defer allocator.free(v_buf);
    const pos_buf: []u8 = try allocator.alloc(u8, bsz * l_q * @sizeOf(i32));
    defer allocator.free(pos_buf);
    const end_buf: []u8 = try allocator.alloc(u8, bsz * @sizeOf(i32));
    defer allocator.free(end_buf);

    const q_vals: []align(1) f32 = asF32Slice(q_buf);
    const k_vals_f16: []align(1) f16 = asF16Slice(k_buf);
    const v_vals_f16: []align(1) f16 = asF16Slice(v_buf);

    const k_ref_f32: []f32 = try allocator.alloc(f32, bsz * h_kv * t_cap * d_k);
    defer allocator.free(k_ref_f32);
    const v_ref_f32: []f32 = try allocator.alloc(f32, bsz * h_kv * t_cap * d_v);
    defer allocator.free(v_ref_f32);

    for (q_vals, 0..) |*v0, i| v0.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 29)) - 14))) * 0.04;
    for (k_ref_f32, 0..) |*v0, i| {
        const val: f32 = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 31)) - 15))) * 0.03;
        v0.* = val;
        k_vals_f16[i] = @floatCast(val);
    }
    for (v_ref_f32, 0..) |*v0, i| {
        const val: f32 = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 37)) - 18))) * 0.025;
        v0.* = val;
        v_vals_f16[i] = @floatCast(val);
    }

    const pos_ptr: [*]align(1) i32 = @ptrCast(pos_buf.ptr);
    const pos_vals: []align(1) i32 = pos_ptr[0 .. pos_buf.len / @sizeOf(i32)];
    pos_vals[0] = 4;
    pos_vals[1] = 5;

    const end_ptr: [*]align(1) i32 = @ptrCast(end_buf.ptr);
    const end_vals: []align(1) i32 = end_ptr[0 .. end_buf.len / @sizeOf(i32)];
    end_vals[0] = 6;

    const ref_vals: []f32 = try allocator.alloc(f32, bsz * l_q * h_q * d_v);
    defer allocator.free(ref_vals);
    cachedGqaRefF32(
        ref_vals,
        q_vals,
        k_ref_f32,
        v_ref_f32,
        pos_vals,
        end_vals,
        bsz,
        l_q,
        h_q,
        h_kv,
        t_cap,
        d_k,
        d_v,
        scale,
        window,
        softcap,
    );

    var sm: manager_mod.StorageManager = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const q_tid: manager_mod.TensorId = try sm.createTiledTensor(.f32, &[_]usize{ bsz, l_q, h_q, d_k }, &[_]usize{ 1, 1, 2, d_k }, .{ .tile_alignment = 64 });
    const k_tid: manager_mod.TensorId = try sm.createTiledTensor(.f16, &[_]usize{ bsz, t_cap, h_kv, d_k }, &[_]usize{ 1, 2, 1, d_k }, .{ .tile_alignment = 64 });
    const v_tid: manager_mod.TensorId = try sm.createTiledTensor(.f16, &[_]usize{ bsz, t_cap, h_kv, d_v }, &[_]usize{ 1, 2, 1, d_v }, .{ .tile_alignment = 64 });
    const pos_tid: manager_mod.TensorId = try sm.createTiledTensor(.i32, &[_]usize{ bsz, l_q }, &[_]usize{ 1, 1 }, .{ .tile_alignment = 64 });
    const end_tid: manager_mod.TensorId = try sm.createTiledTensor(.i32, &[_]usize{bsz}, &[_]usize{bsz}, .{ .tile_alignment = 64 });

    try sm.writeFromPackedScalar(q_tid, q_buf);
    try sm.writeFromPackedScalar(k_tid, k_buf);
    try sm.writeFromPackedScalar(v_tid, v_buf);
    try sm.writeFromPackedScalar(pos_tid, pos_buf);
    try sm.writeFromPackedScalar(end_tid, end_buf);

    var g: graph_mod.Graph = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const q_in: graph_mod.ValueId = try g.addInput(.f32, &[_]usize{ bsz, l_q, h_q, d_k });
    const k_in: graph_mod.ValueId = try g.addInput(.f16, &[_]usize{ bsz, t_cap, h_kv, d_k });
    const v_in: graph_mod.ValueId = try g.addInput(.f16, &[_]usize{ bsz, t_cap, h_kv, d_v });
    const pos_in: graph_mod.ValueId = try g.addInput(.i32, &[_]usize{ bsz, l_q });
    const end_in: graph_mod.ValueId = try g.addInput(.i32, &[_]usize{bsz});

    try g.bindExternal(q_in, @intCast(q_tid));
    try g.bindExternal(k_in, @intCast(k_tid));
    try g.bindExternal(v_in, @intCast(v_tid));
    try g.bindExternal(pos_in, @intCast(pos_tid));
    try g.bindExternal(end_in, @intCast(end_tid));

    const out: graph_mod.ValueId = try g.addAttention(
        q_in,
        k_in,
        v_in,
        pos_in,
        end_in,
        scale,
        window,
        softcap,
    );
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 4, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();

    const out_meta: *const manager_mod.TiledTensor = try sm.getConst(prog.outputs[0]);
    try std.testing.expectEqual(types.DType.f32, out_meta.dtype);

    var cpu: cpu_backend_mod.CpuBackend = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, bsz * l_q * h_q * d_v * @sizeOf(f32));
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    var max_abs: f32 = 0.0;
    for (out_vals, ref_vals) |got, want| {
        try std.testing.expect(std.math.isFinite(got));
        max_abs = @max(max_abs, @abs(got - want));
    }
    try std.testing.expect(max_abs <= 1.5e-1);
}

// --- f16 coverage for ops whose `infer` rule always allowed f16 -------------
//
// Softmax, ArgMax and LSTMCell were blessed by graph/infer.zig for f16 but
// rejected by their executors, so an f16 graph compiled fine and then failed at
// run time. These pin the wired paths.

/// f64 softmax reference over f16 inputs, widened exactly as the kernels do.
fn softmaxRefRowF16(out: []f32, x: []align(1) const f16) void {
    std.debug.assert(out.len == x.len);
    var maxv: f64 = -std.math.inf(f64);
    for (x) |v| maxv = @max(maxv, @as(f64, @floatCast(v)));
    var sum: f64 = 0.0;
    for (x, 0..) |v, i| {
        const e: f64 = std.math.exp(@as(f64, @floatCast(v)) - maxv);
        out[i] = @floatCast(e);
        sum += e;
    }
    const inv: f64 = 1.0 / sum;
    for (out) |*v| v.* = @floatCast(@as(f64, @floatCast(v.*)) * inv);
}

test "cpu backend: softmax rank-1 (f16) normalizes without an f16 intermediate" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const n: usize = 257;
    const x_buf: []u8 = try allocator.alloc(u8, n * 2);
    defer allocator.free(x_buf);
    const x_vals: []align(1) f16 = asF16Slice(x_buf);

    // A deliberately wide spread: the low tail sits far enough below the max
    // that exp(x - max) lands in (and under) f16 subnormals. Parking that
    // intermediate in f16 is exactly what the wired path refuses to do.
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const t: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 128));
        x_vals[i] = @floatCast(t / 8.0); // ~[-16, 16]
    }

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: Backend = cpu.backend();

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    // Awkward tiling so the multi-tile max/sum/normalize passes all run.
    const x_tid = try sm.createTiledTensor(.f16, &[_]usize{n}, &[_]usize{17}, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(x_tid, x_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f16, &[_]usize{n});
    try g.bindExternal(x_in, @intCast(x_tid));
    const y = try g.addSoftmax(x_in, -1);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 32, .base_1d = 128, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();
    try backend.executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, n * 2);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f16 = asF16Slice(out_buf);

    const ref: []f32 = try allocator.alloc(f32, n);
    defer allocator.free(ref);
    softmaxRefRowF16(ref, x_vals);

    var sum: f32 = 0.0;
    for (out_vals) |v| {
        const f: f32 = @floatCast(v);
        try std.testing.expect(std.math.isFinite(f));
        try std.testing.expect(f >= 0.0 and f <= 1.0);
        sum += f;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 5e-3);

    // f16 storage dominates the error budget; the fast exp adds far less.
    var max_abs: f32 = 0.0;
    for (out_vals, ref) |g0, r0| max_abs = @max(max_abs, @abs(@as(f32, @floatCast(g0)) - r0));
    try std.testing.expect(max_abs <= 3e-3);

    // The largest input must still carry the largest probability.
    var best: usize = 0;
    for (out_vals, 0..) |v, k| {
        if (@as(f32, @floatCast(v)) > @as(f32, @floatCast(out_vals[best]))) best = k;
    }
    try std.testing.expectEqual(@as(usize, n - 1), best);
}

test "cpu backend: softmax rank-2 (f16) matches reference per row" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const m: usize = 5;
    const n: usize = 9;
    const x_buf: []u8 = try allocator.alloc(u8, m * n * 2);
    defer allocator.free(x_buf);
    const x_vals: []align(1) f16 = asF16Slice(x_buf);

    var r: usize = 0;
    while (r < m) : (r += 1) {
        var c: usize = 0;
        while (c < n) : (c += 1) {
            const t: f32 = @floatFromInt(@as(i32, @intCast(r * n + c)));
            x_vals[r * n + c] = @floatCast((t / 7.0) - 3.0);
        }
    }

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: Backend = cpu.backend();

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const x_tid = try sm.createTiledTensor(.f16, &[_]usize{ m, n }, &[_]usize{ 2, 4 }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(x_tid, x_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();
    const x_in = try g.addInput(.f16, &[_]usize{ m, n });
    try g.bindExternal(x_in, @intCast(x_tid));
    const y = try g.addSoftmax(x_in, -1);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 4, .base_1d = 8, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();
    try backend.executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, m * n * 2);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f16 = asF16Slice(out_buf);

    const ref_row: []f32 = try allocator.alloc(f32, n);
    defer allocator.free(ref_row);

    r = 0;
    while (r < m) : (r += 1) {
        softmaxRefRowF16(ref_row, x_vals[r * n .. r * n + n]);
        var row_sum: f32 = 0.0;
        var c: usize = 0;
        while (c < n) : (c += 1) {
            const got: f32 = @floatCast(out_vals[r * n + c]);
            row_sum += got;
            try std.testing.expect(@abs(got - ref_row[c]) <= 3e-3);
        }
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), row_sum, 5e-3);
    }
}

test "cpu backend: argmax (f16) picks the same index as f32" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const rows: usize = 3;
    const n: usize = 11;
    const x_buf: []u8 = try allocator.alloc(u8, rows * n * 2);
    defer allocator.free(x_buf);
    const x_vals: []align(1) f16 = asF16Slice(x_buf);

    // One distinct winner per row, at a different column each time.
    const winners = [_]usize{ 0, 7, 10 };
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        var c: usize = 0;
        while (c < n) : (c += 1) x_vals[r * n + c] = @floatCast(@as(f32, @floatFromInt(c)) * 0.25 - 3.0);
        x_vals[r * n + winners[r]] = 9.5;
    }

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: Backend = cpu.backend();

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const x_tid = try sm.createTiledTensor(.f16, &[_]usize{ rows, n }, &[_]usize{ rows, n }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(x_tid, x_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();
    const x_in = try g.addInput(.f16, &[_]usize{ rows, n });
    try g.bindExternal(x_in, @intCast(x_tid));
    const y = try g.addArgMax(x_in, -1);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 32, .base_1d = 128, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();
    try backend.executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, rows * 4);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) const i32 = @alignCast(std.mem.bytesAsSlice(i32, out_buf));

    r = 0;
    while (r < rows) : (r += 1) {
        try std.testing.expectEqual(@as(i32, @intCast(winners[r])), out_vals[r]);
    }
}

/// Build + run one LSTMCell whose operands all carry `dt`, returning the output
/// state widened to f32. Values are the same regardless of `dt`: they are chosen
/// to be exactly representable in f16, so the only difference between a f32 and
/// a f16 run is the storage the cell reads and writes through.
fn runLSTMCellAsF32(
    allocator: std.mem.Allocator,
    dt: types.DType,
    batch: usize,
    input_size: usize,
    hidden: usize,
    out: []f32,
) !void {
    const gate_dim: usize = hidden * 4;
    const elem: usize = if (dt == .f16) 2 else 4;

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    // Deterministic, f16-exact values: k/16 over a small range has an exact
    // binary representation in both dtypes, so neither run rounds its inputs.
    const Fill = struct {
        fn go(dtype: types.DType, buf: []u8, seed: usize) void {
            const count: usize = buf.len / (if (dtype == .f16) @as(usize, 2) else 4);
            var k: usize = 0;
            while (k < count) : (k += 1) {
                const raw: f32 = @as(f32, @floatFromInt((k * 7 + seed) % 33)) / 16.0 - 1.0;
                if (dtype == .f16) {
                    @as(*align(1) f16, @ptrCast(buf.ptr + k * 2)).* = @floatCast(raw);
                } else {
                    @as(*align(1) f32, @ptrCast(buf.ptr + k * 4)).* = raw;
                }
            }
        }
    };

    const specs = [_]struct { shape: []const usize, seed: usize }{
        .{ .shape = &[_]usize{ batch, input_size }, .seed = 1 },
        .{ .shape = &[_]usize{ batch, hidden }, .seed = 5 },
        .{ .shape = &[_]usize{ batch, hidden }, .seed = 9 },
        .{ .shape = &[_]usize{ input_size, gate_dim }, .seed = 13 },
        .{ .shape = &[_]usize{ hidden, gate_dim }, .seed = 17 },
        .{ .shape = &[_]usize{gate_dim}, .seed = 21 },
        .{ .shape = &[_]usize{gate_dim}, .seed = 25 },
    };

    var tids: [specs.len]manager_mod.TensorId = undefined;
    inline for (specs, 0..) |spec, si| {
        var n: usize = 1;
        for (spec.shape) |d| n *= d;
        const buf: []u8 = try allocator.alloc(u8, n * elem);
        defer allocator.free(buf);
        Fill.go(dt, buf, spec.seed);
        tids[si] = try sm.createTiledTensor(dt, spec.shape, spec.shape, .{ .tile_alignment = 64 });
        try sm.writeFromPackedScalar(tids[si], buf);
    }

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    var ids: [specs.len]graph_mod.ValueId = undefined;
    inline for (specs, 0..) |spec, si| {
        ids[si] = try g.addInput(dt, spec.shape);
        try g.bindExternal(ids[si], @intCast(tids[si]));
    }

    const y = try g.addLSTMCell(ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6]);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 32, .base_1d = 128, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, .cpu(policy));
    defer prog.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    const out_elems: usize = batch * hidden * 2;
    std.debug.assert(out.len == out_elems);
    const out_buf: []u8 = try allocator.alloc(u8, out_elems * elem);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);

    var k: usize = 0;
    while (k < out_elems) : (k += 1) {
        out[k] = if (dt == .f16)
            @floatCast(@as(*align(1) const f16, @ptrCast(out_buf.ptr + k * 2)).*)
        else
            @as(*align(1) const f32, @ptrCast(out_buf.ptr + k * 4)).*;
    }
}

test "cpu backend: lstm cell (f16) tracks the f32 cell" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const batch: usize = 3;
    const input_size: usize = 6;
    const hidden: usize = 8;
    const out_elems: usize = batch * hidden * 2;

    const got_f32: []f32 = try allocator.alloc(f32, out_elems);
    defer allocator.free(got_f32);
    const got_f16: []f32 = try allocator.alloc(f32, out_elems);
    defer allocator.free(got_f16);

    try runLSTMCellAsF32(allocator, .f32, batch, input_size, hidden, got_f32);
    try runLSTMCellAsF32(allocator, .f16, batch, input_size, hidden, got_f16);

    // The gate math is f32 in both runs (only the I/O helpers know the dtype), so
    // the f16 result is the f32 result rounded once on store -- nothing more.
    var max_abs: f32 = 0.0;
    for (got_f32, got_f16) |a, b| {
        try std.testing.expect(std.math.isFinite(b));
        max_abs = @max(max_abs, @abs(a - b));
    }
    try std.testing.expect(max_abs <= 1e-3);

    // Guard against a degenerate all-zero comparison.
    var any_nonzero = false;
    for (got_f32) |v| {
        if (@abs(v) > 1e-4) any_nonzero = true;
    }
    try std.testing.expect(any_nonzero);
}
