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
            try std.testing.expect(std.math.isFinite(got));

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
            };
            const ref: f32 = @as(f32, @floatCast(ref64));

            const abs_err: f32 = @abs(got - ref);
            const tol: f32 = switch (case.op) {
                .relu => 0.0,
                .sigmoid => 3e-2,
                .tanh => 3e-2,
                .silu => 5e-2,
                .gelu => 6e-2,
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
    const y = try g.addSoftmax(x_in);
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
    const y = try g.addSoftmax(x_in);
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

        const y = if (case.is_rms) try g.addRMSNorm(x_in, gamma_in, beta_in, eps) else try g.addLayerNorm(x_in, gamma_in, beta_in, eps);
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
