const std = @import("std");

const backend_mod = @import("../backend/backend.zig");
const cpu_backend_mod = @import("../backend/cpu/cpu_backend.zig");
const types = @import("../backend/types.zig");
const manager_mod = @import("../storage/manager.zig");
const graph_mod = @import("graph.zig");
const infer_mod = @import("infer.zig");
const plan_mod = @import("plan.zig");
const program = @import("program.zig");

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

test "graph: rmsnorm supports row tiles >256" {
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
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: compile+run covers matmul/broadcast/elemwise/relu/copy/reduce" {
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
    const d = try g.addBroadcastLastDimBinary(.add, c, bias_in);
    const e = try g.addRelu(d);
    const f = try g.addCopy(e);
    const gg = try g.addElemwiseBinary(.mul, f, f);
    const out = try g.addReduce(.mean, gg);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 2, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: batched matmul rank-3 matches reference (f32)" {
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

    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: batched matmul broadcast B rank-3 matches reference (f32)" {
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

    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: batched matmul broadcast A rank-3 matches reference (f32)" {
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

    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: batch retile guard accepts small, rejects large" {
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

    var prog = try program.compileGraph(allocator, &g, &sm, policy_ok);
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
    try std.testing.expectError(program.CompileError.InvalidArgument, program.compileGraph(allocator, &g, &sm, policy_bad));
}

test "graph: unary ops match reference (f32)" {
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

        var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: softmax rank-1 matches reference (f32)" {
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
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: softmax rank-2 matches reference (f32)" {
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
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: softmax rank-2 axis-0 matches reference (f32)" {
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
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: softmax rank-3 axis-last matches reference (f32)" {
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
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: layernorm and rmsnorm rank-2 match reference (f32)" {
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

        var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: layernorm rank-3 normalized-shape matches reference (f32)" {
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
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: attention rank-2 matches reference (f32)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const m: usize = 7;
    const n: usize = 11;
    const dk: usize = 9;
    const dv: usize = 13;

    const q_bytes_len: usize = m * dk * 4;
    const k_bytes_len: usize = n * dk * 4;
    const v_bytes_len: usize = n * dv * 4;
    const out_bytes_len: usize = m * dv * 4;

    const q_buf: []u8 = try allocator.alloc(u8, q_bytes_len);
    defer allocator.free(q_buf);
    const k_buf: []u8 = try allocator.alloc(u8, k_bytes_len);
    defer allocator.free(k_buf);
    const v_buf: []u8 = try allocator.alloc(u8, v_bytes_len);
    defer allocator.free(v_buf);

    const q_vals: []align(1) f32 = asF32Slice(q_buf);
    const k_vals: []align(1) f32 = asF32Slice(k_buf);
    const v_vals: []align(1) f32 = asF32Slice(v_buf);

    // Keep magnitudes modest so expApprox and reference stay close.
    for (0..m * dk) |i| q_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast((i % 31))) - 15))) * 0.03;
    for (0..n * dk) |i| k_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast((i % 29))) - 14))) * 0.03;
    for (0..n * dv) |i| v_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast((i % 37))) - 18))) * 0.02;

    const scale: f32 = 0.25;

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: Backend = cpu.backend();

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    // Intentionally awkward tiling to exercise retile.
    const q_tid = try sm.createTiledTensor(.f32, &[_]usize{ m, dk }, &[_]usize{ 3, 4 }, .{ .tile_alignment = 64 });
    const k_tid = try sm.createTiledTensor(.f32, &[_]usize{ n, dk }, &[_]usize{ 4, 5 }, .{ .tile_alignment = 64 });
    const v_tid = try sm.createTiledTensor(.f32, &[_]usize{ n, dv }, &[_]usize{ 5, 3 }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(q_tid, q_buf);
    try sm.writeFromPackedScalar(k_tid, k_buf);
    try sm.writeFromPackedScalar(v_tid, v_buf);

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 8, .base_1d = 5, .tile_alignment = 64 };

    inline for (.{
        .{ .causal = false, .name = "noncausal" },
        .{ .causal = true, .name = "causal" },
    }) |case| {
        var g = graph_mod.Graph.init(allocator);
        defer g.deinit();

        const q_in = try g.addInput(.f32, &[_]usize{ m, dk });
        const k_in = try g.addInput(.f32, &[_]usize{ n, dk });
        const v_in = try g.addInput(.f32, &[_]usize{ n, dv });
        try g.bindExternal(q_in, @intCast(q_tid));
        try g.bindExternal(k_in, @intCast(k_tid));
        try g.bindExternal(v_in, @intCast(v_tid));

        const y = try g.addAttention(q_in, k_in, v_in, scale, case.causal);
        try g.setOutputs(&[_]graph_mod.ValueId{y});

        var prog = try program.compileGraph(allocator, &g, &sm, policy);
        defer prog.deinit();
        try backend.executeProgram(&prog, sm.tensorStore());

        const out_buf: []u8 = try allocator.alloc(u8, out_bytes_len);
        defer allocator.free(out_buf);
        try sm.readToPackedScalar(prog.outputs[0], out_buf);
        const out_vals: []align(1) f32 = asF32Slice(out_buf);

        const ref: []f32 = try allocator.alloc(f32, m * dv);
        defer allocator.free(ref);
        attentionRefF32(allocator, ref, q_vals, k_vals, v_vals, m, n, dk, dv, scale, case.causal);

        var max_abs: f32 = 0.0;
        for (out_vals, ref) |got, r0| {
            try std.testing.expect(std.math.isFinite(got));
            max_abs = @max(max_abs, @abs(got - r0));
        }
        if (!(max_abs <= 6e-2)) {
            std.debug.print("attention {s} mismatch max_abs={}\n", .{ case.name, max_abs });
            return error.TestExpectedEqual;
        }
    }
}

test "graph: attention rank-3 batched matches reference (f32)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const batch: usize = 2;
    const m: usize = 4;
    const n: usize = 5;
    const dk: usize = 6;
    const dv: usize = 7;

    const q_bytes_len: usize = batch * m * dk * 4;
    const k_bytes_len: usize = batch * n * dk * 4;
    const v_bytes_len: usize = batch * n * dv * 4;
    const out_bytes_len: usize = batch * m * dv * 4;

    const q_buf: []u8 = try allocator.alloc(u8, q_bytes_len);
    defer allocator.free(q_buf);
    const k_buf: []u8 = try allocator.alloc(u8, k_bytes_len);
    defer allocator.free(k_buf);
    const v_buf: []u8 = try allocator.alloc(u8, v_bytes_len);
    defer allocator.free(v_buf);

    const q_vals: []align(1) f32 = asF32Slice(q_buf);
    const k_vals: []align(1) f32 = asF32Slice(k_buf);
    const v_vals: []align(1) f32 = asF32Slice(v_buf);

    for (0..batch * m * dk) |i| q_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast((i % 31))) - 15))) * 0.03;
    for (0..batch * n * dk) |i| k_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast((i % 29))) - 14))) * 0.03;
    for (0..batch * n * dv) |i| v_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast((i % 37))) - 18))) * 0.02;

    const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(dk)));

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: Backend = cpu.backend();

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 8, .base_1d = 5, .tile_alignment = 64 };
    const st = plan_mod.chooseAttentionTiles(policy, m, n, dk, dv);

    const q_tid = try sm.createTiledTensor(.f32, &[_]usize{ batch, m, dk }, &[_]usize{ 1, st.tm, st.tk }, .{ .tile_alignment = 64 });
    const k_tid = try sm.createTiledTensor(.f32, &[_]usize{ batch, n, dk }, &[_]usize{ 1, st.tn, st.tk }, .{ .tile_alignment = 64 });
    const v_tid = try sm.createTiledTensor(.f32, &[_]usize{ batch, n, dv }, &[_]usize{ 1, st.tn, st.tv }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(q_tid, q_buf);
    try sm.writeFromPackedScalar(k_tid, k_buf);
    try sm.writeFromPackedScalar(v_tid, v_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const q_in = try g.addInput(.f32, &[_]usize{ batch, m, dk });
    const k_in = try g.addInput(.f32, &[_]usize{ batch, n, dk });
    const v_in = try g.addInput(.f32, &[_]usize{ batch, n, dv });
    try g.bindExternal(q_in, @intCast(q_tid));
    try g.bindExternal(k_in, @intCast(k_tid));
    try g.bindExternal(v_in, @intCast(v_tid));

    const y = try g.addAttention(q_in, k_in, v_in, scale, false);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    var prog = try program.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();
    try backend.executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, out_bytes_len);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    const ref: []f32 = try allocator.alloc(f32, batch * m * dv);
    defer allocator.free(ref);
    attentionRefBatchedF32(allocator, ref, q_vals, k_vals, v_vals, batch, m, n, dk, dv, scale, false);

    var max_abs: f32 = 0.0;
    for (out_vals, ref) |got, r0| {
        try std.testing.expect(std.math.isFinite(got));
        max_abs = @max(max_abs, @abs(got - r0));
    }
    if (!(max_abs <= 6e-2)) {
        std.debug.print("batched attention mismatch max_abs={}\n", .{max_abs});
        return error.TestExpectedEqual;
    }
}

test "graph: multi-head attention rank-4 batched matches reference (f32)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const batch: usize = 2;
    const heads: usize = 2;
    const m_head: usize = 3;
    const n_head: usize = 4;
    const dk: usize = 5;
    const dv: usize = 6;

    const q_bytes_len: usize = batch * heads * m_head * dk * 4;
    const k_bytes_len: usize = batch * heads * n_head * dk * 4;
    const v_bytes_len: usize = batch * heads * n_head * dv * 4;
    const out_bytes_len: usize = batch * heads * m_head * dv * 4;

    const q_buf: []u8 = try allocator.alloc(u8, q_bytes_len);
    defer allocator.free(q_buf);
    const k_buf: []u8 = try allocator.alloc(u8, k_bytes_len);
    defer allocator.free(k_buf);
    const v_buf: []u8 = try allocator.alloc(u8, v_bytes_len);
    defer allocator.free(v_buf);

    const q_vals: []align(1) f32 = asF32Slice(q_buf);
    const k_vals: []align(1) f32 = asF32Slice(k_buf);
    const v_vals: []align(1) f32 = asF32Slice(v_buf);

    for (0..batch * heads * m_head * dk) |i| q_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast((i % 31))) - 15))) * 0.03;
    for (0..batch * heads * n_head * dk) |i| k_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast((i % 29))) - 14))) * 0.03;
    for (0..batch * heads * n_head * dv) |i| v_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast((i % 37))) - 18))) * 0.02;

    const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(dk)));

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: Backend = cpu.backend();

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 8, .base_1d = 5, .tile_alignment = 64 };
    const st = plan_mod.chooseAttentionTiles(policy, m_head, n_head, dk, dv);

    const q_tid = try sm.createTiledTensor(.f32, &[_]usize{ batch, heads, m_head, dk }, &[_]usize{ 1, 1, st.tm, st.tk }, .{ .tile_alignment = 64 });
    const k_tid = try sm.createTiledTensor(.f32, &[_]usize{ batch, heads, n_head, dk }, &[_]usize{ 1, 1, st.tn, st.tk }, .{ .tile_alignment = 64 });
    const v_tid = try sm.createTiledTensor(.f32, &[_]usize{ batch, heads, n_head, dv }, &[_]usize{ 1, 1, st.tn, st.tv }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(q_tid, q_buf);
    try sm.writeFromPackedScalar(k_tid, k_buf);
    try sm.writeFromPackedScalar(v_tid, v_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const q_in = try g.addInput(.f32, &[_]usize{ batch, heads, m_head, dk });
    const k_in = try g.addInput(.f32, &[_]usize{ batch, heads, n_head, dk });
    const v_in = try g.addInput(.f32, &[_]usize{ batch, heads, n_head, dv });
    try g.bindExternal(q_in, @intCast(q_tid));
    try g.bindExternal(k_in, @intCast(k_tid));
    try g.bindExternal(v_in, @intCast(v_tid));

    const y = try g.addMultiHeadAttention(q_in, k_in, v_in, scale, false, heads);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    var prog = try program.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();
    try backend.executeProgram(&prog, sm.tensorStore());

    const out_buf: []u8 = try allocator.alloc(u8, out_bytes_len);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    const ref: []f32 = try allocator.alloc(f32, batch * heads * m_head * dv);
    defer allocator.free(ref);
    attentionRefBatchedMultiHeadF32(allocator, ref, q_vals, k_vals, v_vals, batch, heads, m_head, n_head, dk, dv, scale, false);

    var max_abs: f32 = 0.0;
    for (out_vals, ref) |got, r0| {
        try std.testing.expect(std.math.isFinite(got));
        max_abs = @max(max_abs, @abs(got - r0));
    }
    if (!(max_abs <= 6e-2)) {
        std.debug.print("batched multi-head attention mismatch max_abs={}\n", .{max_abs});
        return error.TestExpectedEqual;
    }
}

test "graph: matmul f16 allows promoted f32 output" {
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
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    try backend.executeProgram(&prog, sm.tensorStore());

    var out_buf: [4]u8 = .{ 0, 0, 0, 0 };
    try sm.readToPackedScalar(prog.outputs[0], out_buf[0..4]);
    const out_val: f32 = @as(*align(1) const f32, @ptrCast(out_buf[0..4].ptr)).*;
    try std.testing.expect(std.math.isFinite(out_val));
    // f16 inputs may accumulate slightly differently depending on kernel, but should be very close.
    try std.testing.expectApproxEqAbs(expected_sum, out_val, 5e-3);
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

test "graph: view ops lower to materialization (transpose/slice/reshape)" {
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
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();
    try backend.executeProgram(&prog, sm.tensorStore());

    var out_buf: [4]u8 = .{ 0, 0, 0, 0 };
    try sm.readToPackedScalar(prog.outputs[0], out_buf[0..4]);
    const out_val: f32 = @as(*align(1) const f32, @ptrCast(out_buf[0..4].ptr)).*;
    try std.testing.expect(@abs(out_val - expected_sum) <= 1e-5);
}

test "graph: reshape supports rank-3 materialization" {
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
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    var out_buf: [4]u8 = .{ 0, 0, 0, 0 };
    try sm.readToPackedScalar(prog.outputs[0], out_buf[0..4]);
    const out_val: f32 = @as(*align(1) const f32, @ptrCast(out_buf[0..4].ptr)).*;
    try std.testing.expectApproxEqAbs(expected_sum, out_val, 1e-6);
}

test "graph: view slice nd materialization rank-3" {
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
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    var out_buf: [4]u8 = .{ 0, 0, 0, 0 };
    try sm.readToPackedScalar(prog.outputs[0], out_buf[0..4]);
    const out_val: f32 = @as(*align(1) const f32, @ptrCast(out_buf[0..4].ptr)).*;
    try std.testing.expectApproxEqAbs(expected_sum, out_val, 1e-5);
}

test "graph: concat axis-1 materialization" {
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
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: gather rows matches reference (f32)" {
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

    const out = try g.addGatherRows(table_in, idx_in);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: gather rows matches reference (f16)" {
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

    const out = try g.addGatherRows(table_in, idx_in);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: gather rows matches reference (q8_0 table, f32 output)" {
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

    const out = try g.addGatherRows(table_in, idx_in);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: cast f32 -> f16 roundtrip matches reference" {
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
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: cast f16 -> f32 matches reference" {
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
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: matmul NT (A f32 @ B^T q8_0 quant_axis=1) matches reference" {
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
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: gather rows out-of-bounds returns InvalidArgument" {
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

    const out = try g.addGatherRows(table_in, idx_in);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try std.testing.expectError(types.BackendError.InvalidArgument, cpu.backend().executeProgram(&prog, sm.tensorStore()));
}

test "graph: rope1d matches chunked-halves reference (f32)" {
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
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: rope1d matches chunked-halves reference (f16)" {
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
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: conv1d depthwise (NLC) matches reference" {
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
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: conv1d depthwise reflect padding matches reference" {
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
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: conv1d reflect padding matches reference" {
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
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: conv2d reflect padding matches reference" {
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
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: conv1d pointwise (NLC) matches reference" {
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
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: reduce axis sum/mean matches reference" {
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
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: conv1d general (NLC) supports large c_out" {
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
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: conv2d pointwise (NHWC) matches reference" {
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
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: conv2d depthwise (NHWC) matches reference" {
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
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: conv2d depthwise reflect padding matches reference" {
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
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: conv2d general (NHWC) supports large c_out" {
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
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: kv cache append mutates cache in-place (f32)" {
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
                    const src_idx: usize = (((b * heads + h) * append_len + t) * d_head) + d;
                    const dst_idx: usize = (((b * heads + h) * t_cap + dst_t) * d_head) + d;
                    expected[dst_idx] = new_vals[src_idx];
                }
            }
        }
    }

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    // v1 constraints: single tile over batch+heads, full head-dim tile.
    const cache_tid = try sm.createTiledTensor(.f32, &[_]usize{ bsz, heads, t_cap, d_head }, &[_]usize{ bsz, heads, 3, d_head }, .{ .tile_alignment = 64 });
    const new_tid = try sm.createTiledTensor(.f32, &[_]usize{ bsz, heads, append_len, d_head }, &[_]usize{ bsz, heads, 1, d_head }, .{ .tile_alignment = 64 });
    const end_tid = try sm.createTiledTensor(.i32, &[_]usize{bsz}, &[_]usize{bsz}, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(cache_tid, cache_buf);
    try sm.writeFromPackedScalar(new_tid, new_buf);
    try sm.writeFromPackedScalar(end_tid, end_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const cache_in = try g.addInput(.f32, &[_]usize{ bsz, heads, t_cap, d_head });
    const new_in = try g.addInput(.f32, &[_]usize{ bsz, heads, append_len, d_head });
    const end_in = try g.addInput(.i32, &[_]usize{bsz});
    try g.bindExternal(cache_in, @intCast(cache_tid));
    try g.bindExternal(new_in, @intCast(new_tid));
    try g.bindExternal(end_in, @intCast(end_tid));

    const out = try g.addKVCacheAppend(cache_in, new_in, end_in);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 4, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: kv cache append rejects out-of-bounds end index" {
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

    const cache_tid = try sm.createTiledTensor(.f32, &[_]usize{ bsz, heads, t_cap, d_head }, &[_]usize{ bsz, heads, 2, d_head }, .{ .tile_alignment = 64 });
    const new_tid = try sm.createTiledTensor(.f32, &[_]usize{ bsz, heads, append_len, d_head }, &[_]usize{ bsz, heads, 1, d_head }, .{ .tile_alignment = 64 });
    const end_tid = try sm.createTiledTensor(.i32, &[_]usize{bsz}, &[_]usize{bsz}, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(cache_tid, cache_buf);
    try sm.writeFromPackedScalar(new_tid, new_buf);
    try sm.writeFromPackedScalar(end_tid, end_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const cache_in = try g.addInput(.f32, &[_]usize{ bsz, heads, t_cap, d_head });
    const new_in = try g.addInput(.f32, &[_]usize{ bsz, heads, append_len, d_head });
    const end_in = try g.addInput(.i32, &[_]usize{bsz});
    try g.bindExternal(cache_in, @intCast(cache_tid));
    try g.bindExternal(new_in, @intCast(new_tid));
    try g.bindExternal(end_in, @intCast(end_tid));

    const out = try g.addKVCacheAppend(cache_in, new_in, end_in);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 2, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try std.testing.expectError(types.BackendError.InvalidArgument, cpu.backend().executeProgram(&prog, sm.tensorStore()));
}

test "graph: kv cache append ring policy wraps time index" {
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

    const cache_tid: manager_mod.TensorId = try sm.createTiledTensor(.f32, &[_]usize{ bsz, heads, t_cap, d_head }, &[_]usize{ bsz, heads, 2, d_head }, .{ .tile_alignment = 64 });
    const new_tid: manager_mod.TensorId = try sm.createTiledTensor(.f32, &[_]usize{ bsz, heads, append_len, d_head }, &[_]usize{ bsz, heads, 1, d_head }, .{ .tile_alignment = 64 });
    const end_tid: manager_mod.TensorId = try sm.createTiledTensor(.i32, &[_]usize{bsz}, &[_]usize{bsz}, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(cache_tid, cache_buf);
    try sm.writeFromPackedScalar(new_tid, new_buf);
    try sm.writeFromPackedScalar(end_tid, end_buf);
    try sm.registerKVCachePolicy(cache_tid, .{ .ring = .{ .window_tokens = t_cap } });

    var g: graph_mod.Graph = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const cache_in: graph_mod.ValueId = try g.addInput(.f32, &[_]usize{ bsz, heads, t_cap, d_head });
    const new_in: graph_mod.ValueId = try g.addInput(.f32, &[_]usize{ bsz, heads, append_len, d_head });
    const end_in: graph_mod.ValueId = try g.addInput(.i32, &[_]usize{bsz});
    try g.bindExternal(cache_in, @intCast(cache_tid));
    try g.bindExternal(new_in, @intCast(new_tid));
    try g.bindExternal(end_in, @intCast(end_tid));

    const out: graph_mod.ValueId = try g.addKVCacheAppend(cache_in, new_in, end_in);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 2, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: kv cache append growable policy expands physical capacity" {
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

    const cache_tid: manager_mod.TensorId = try sm.createTiledTensor(.f32, &[_]usize{ bsz, heads, t_cap, d_head }, &[_]usize{ bsz, heads, 2, d_head }, .{ .tile_alignment = 64 });
    const new_tid: manager_mod.TensorId = try sm.createTiledTensor(.f32, &[_]usize{ bsz, heads, append_len, d_head }, &[_]usize{ bsz, heads, 1, d_head }, .{ .tile_alignment = 64 });
    const end_tid: manager_mod.TensorId = try sm.createTiledTensor(.i32, &[_]usize{bsz}, &[_]usize{bsz}, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(cache_tid, cache_buf);
    try sm.writeFromPackedScalar(new_tid, new_buf);
    try sm.writeFromPackedScalar(end_tid, end_buf);
    try sm.registerKVCachePolicy(cache_tid, .{ .growable = .{ .initial_capacity_tokens = 2, .growth_numerator = 2, .growth_denominator = 1 } });

    var g: graph_mod.Graph = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const cache_in: graph_mod.ValueId = try g.addInput(.f32, &[_]usize{ bsz, heads, t_cap, d_head });
    const new_in: graph_mod.ValueId = try g.addInput(.f32, &[_]usize{ bsz, heads, append_len, d_head });
    const end_in: graph_mod.ValueId = try g.addInput(.i32, &[_]usize{bsz});
    try g.bindExternal(cache_in, @intCast(cache_tid));
    try g.bindExternal(new_in, @intCast(new_tid));
    try g.bindExternal(end_in, @intCast(end_tid));

    const out: graph_mod.ValueId = try g.addKVCacheAppend(cache_in, new_in, end_in);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 2, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    var cpu: cpu_backend_mod.CpuBackend = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    try cpu.backend().executeProgram(&prog, sm.tensorStore());

    const grown_meta: *const manager_mod.TiledTensor = try sm.getConst(cache_tid);
    try std.testing.expectEqual(@as(usize, 8), grown_meta.shape[2]);

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
    causal: bool,
    sliding_window: usize,
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

            var upper: usize = valid_end;
            if (causal) {
                const q_next: usize = q_pos + 1;
                if (q_next < upper) upper = q_next;
            }

            var lower: usize = 0;
            if (sliding_window > 0) {
                const q_next: usize = q_pos + 1;
                lower = if (q_next > sliding_window) q_next - sliding_window else 0;
            }

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
                    const k_base: usize = (((b * h_kv + hkv_idx) * t_cap + t) * d_k);
                    const v_base: usize = (((b * h_kv + hkv_idx) * t_cap + t) * d_v);

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

test "graph: cached grouped-query attention matches reference (f32)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const bsz: usize = 1;
    const l_q: usize = 2;
    const h_q: usize = 4;
    const h_kv: usize = 2;
    const t_cap: usize = 6;
    const d_k: usize = 3;
    const d_v: usize = 2;

    const scale: f32 = 0.5;
    const causal: bool = true;
    const sliding_window: usize = 3;
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
        causal,
        sliding_window,
        softcap,
    );

    var sm: manager_mod.StorageManager = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const q_tid: manager_mod.TensorId = try sm.createTiledTensor(.f32, &[_]usize{ bsz, l_q, h_q, d_k }, &[_]usize{ 1, 1, 2, d_k }, .{ .tile_alignment = 64 });
    const k_tid: manager_mod.TensorId = try sm.createTiledTensor(.f32, &[_]usize{ bsz, h_kv, t_cap, d_k }, &[_]usize{ 1, 1, 2, d_k }, .{ .tile_alignment = 64 });
    const v_tid: manager_mod.TensorId = try sm.createTiledTensor(.f32, &[_]usize{ bsz, h_kv, t_cap, d_v }, &[_]usize{ 1, 1, 2, d_v }, .{ .tile_alignment = 64 });
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
    const k_in: graph_mod.ValueId = try g.addInput(.f32, &[_]usize{ bsz, h_kv, t_cap, d_k });
    const v_in: graph_mod.ValueId = try g.addInput(.f32, &[_]usize{ bsz, h_kv, t_cap, d_v });
    const pos_in: graph_mod.ValueId = try g.addInput(.i32, &[_]usize{ bsz, l_q });
    const end_in: graph_mod.ValueId = try g.addInput(.i32, &[_]usize{bsz});

    try g.bindExternal(q_in, @intCast(q_tid));
    try g.bindExternal(k_in, @intCast(k_tid));
    try g.bindExternal(v_in, @intCast(v_tid));
    try g.bindExternal(pos_in, @intCast(pos_tid));
    try g.bindExternal(end_in, @intCast(end_tid));

    const out: graph_mod.ValueId = try g.addMultiHeadAttentionCached(
        q_in,
        k_in,
        v_in,
        pos_in,
        end_in,
        scale,
        causal,
        sliding_window,
        softcap,
    );
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 4, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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

test "graph: cached grouped-query attention supports q=f32, kv=f16 with f32 output" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const bsz: usize = 1;
    const l_q: usize = 2;
    const h_q: usize = 4;
    const h_kv: usize = 2;
    const t_cap: usize = 6;
    const d_k: usize = 4;
    const d_v: usize = 3;

    const scale: f32 = 0.5;
    const causal: bool = true;
    const sliding_window: usize = 4;
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
        causal,
        sliding_window,
        softcap,
    );

    var sm: manager_mod.StorageManager = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const q_tid: manager_mod.TensorId = try sm.createTiledTensor(.f32, &[_]usize{ bsz, l_q, h_q, d_k }, &[_]usize{ 1, 1, 2, d_k }, .{ .tile_alignment = 64 });
    const k_tid: manager_mod.TensorId = try sm.createTiledTensor(.f16, &[_]usize{ bsz, h_kv, t_cap, d_k }, &[_]usize{ 1, 1, 2, d_k }, .{ .tile_alignment = 64 });
    const v_tid: manager_mod.TensorId = try sm.createTiledTensor(.f16, &[_]usize{ bsz, h_kv, t_cap, d_v }, &[_]usize{ 1, 1, 2, d_v }, .{ .tile_alignment = 64 });
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
    const k_in: graph_mod.ValueId = try g.addInput(.f16, &[_]usize{ bsz, h_kv, t_cap, d_k });
    const v_in: graph_mod.ValueId = try g.addInput(.f16, &[_]usize{ bsz, h_kv, t_cap, d_v });
    const pos_in: graph_mod.ValueId = try g.addInput(.i32, &[_]usize{ bsz, l_q });
    const end_in: graph_mod.ValueId = try g.addInput(.i32, &[_]usize{bsz});

    try g.bindExternal(q_in, @intCast(q_tid));
    try g.bindExternal(k_in, @intCast(k_tid));
    try g.bindExternal(v_in, @intCast(v_tid));
    try g.bindExternal(pos_in, @intCast(pos_tid));
    try g.bindExternal(end_in, @intCast(end_tid));

    const out: graph_mod.ValueId = try g.addMultiHeadAttentionCached(
        q_in,
        k_in,
        v_in,
        pos_in,
        end_in,
        scale,
        causal,
        sliding_window,
        softcap,
    );
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 4, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
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
