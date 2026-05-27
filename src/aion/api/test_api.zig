// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const api = @import("api.zig");
const manager_mod = @import("../storage/manager.zig");
const package_file = @import("../storage/aion_file.zig");
const types = @import("../backend/types.zig");
const fast = @import("../backend/cpu/kernels/fast_math.zig");

const nn = api.nn;

fn createTestFile(dir: std.Io.Dir, sub_path: []const u8, flags: std.Io.File.CreateFlags) !std.Io.File {
    return try dir.createFile(std.testing.io, sub_path, flags);
}

test "api: build+compile+run (matmul/broadcast/relu/copy/reduce)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const m: usize = 2;
    const k: usize = 3;
    const n: usize = 4;

    const a_vals: []f32 = try allocator.alloc(f32, m * k);
    defer allocator.free(a_vals);
    const b_vals: []f32 = try allocator.alloc(f32, k * n);
    defer allocator.free(b_vals);
    const bias_vals: []f32 = try allocator.alloc(f32, n);
    defer allocator.free(bias_vals);

    for (0..m * k) |i| a_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 2)) * 0.5;
    for (0..k * n) |i| b_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 7))) - 3)) * 0.25;
    for (0..n) |i| bias_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 1)) * 0.1;

    // Reference.
    var out_ref: f32 = 0.0;

    var c_ref: [m * n]f32 = undefined;
    var d_ref: [m * n]f32 = undefined;
    var e_ref: [m * n]f32 = undefined;
    var f_ref: [m * n]f32 = undefined;
    var g_ref: [m * n]f32 = undefined;

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

    // d = c + bias
    for (0..m) |i| {
        for (0..n) |j| {
            d_ref[i * n + j] = c_ref[i * n + j] + bias_vals[j];
        }
    }

    // e = relu(d)
    for (0..m * n) |idx| {
        const x: f32 = d_ref[idx];
        e_ref[idx] = if (x > 0.0) x else 0.0;
    }

    // f = copy(e)
    @memcpy(f_ref[0 .. m * n], e_ref[0 .. m * n]);

    // g = f * f
    for (0..m * n) |idx| {
        g_ref[idx] = f_ref[idx] * f_ref[idx];
    }

    // out = mean(g)
    var sum: f32 = 0.0;
    for (0..m * n) |idx| sum += g_ref[idx];
    out_ref = sum / @as(f32, @floatFromInt(m * n));

    // API path.
    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    const a_t = try ctx.fromF32(&[_]usize{ m, k }, a_vals);
    const b_t = try ctx.fromF32(&[_]usize{ k, n }, b_vals);
    const bias_t = try ctx.fromF32(&[_]usize{n}, bias_vals);

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const A = try bld.param(a_t);
    const B = try bld.param(b_t);
    const Bias = try bld.param(bias_t);

    const C = try bld.matmul(A, B, 1.0, 0.0);
    const D = try bld.broadcastAddLastDim(C, Bias);
    const E = try bld.relu(D);
    const F = try bld.copy(E);
    const G = try bld.mul(F, F);
    const Out = try bld.reduce(.mean, G);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Out});
    defer model.deinit();

    const out_t: api.Tensor = try model.runOutputTensor(0);
    const out_val: f32 = try out_t.readScalar(f32);

    try std.testing.expect(std.math.isFinite(out_val));
    try std.testing.expectApproxEqAbs(out_ref, out_val, 1e-5);
}

test "api: context constructors + tensor typed IO" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    // scalar()
    const s: api.Tensor = try ctx.scalar(f32, 3.25);
    try std.testing.expectEqual(@as(usize, 1), try s.elemCount());
    const s_val: f32 = try s.readScalar(f32);
    try std.testing.expectApproxEqAbs(@as(f32, 3.25), s_val, 0.0);

    // vector(): infer shape from values
    const v: api.Tensor = try ctx.vector([_]f32{ 1.0, 2.0, 3.0 });
    try std.testing.expectEqualSlices(usize, &[_]usize{3}, v.getShape());
    var v_out: [3]f32 = undefined;
    try v.read(&v_out);
    try std.testing.expectEqual(@as(f32, 1.0), v_out[0]);
    try std.testing.expectEqual(@as(f32, 2.0), v_out[1]);
    try std.testing.expectEqual(@as(f32, 3.0), v_out[2]);

    // matrix(): explicit 2D shape
    const m: api.Tensor = try ctx.matrix(2, 2, &[_]f32{ 1.0, 2.0, 3.0, 4.0 });
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 2 }, m.getShape());

    // fromArray(): infer shape from nested array type
    const arr: [2][3]f32 = .{ .{ 1.0, 2.0, 3.0 }, .{ 4.0, 5.0, 6.0 } };
    const a: api.Tensor = try ctx.fromArray(arr);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 3 }, a.getShape());

    var flat: [6]f32 = undefined;
    try a.read(&flat);
    try std.testing.expectEqual(@as(f32, 1.0), flat[0]);
    try std.testing.expectEqual(@as(f32, 6.0), flat[5]);

    const flat_alloc: []f32 = try a.readAlloc(allocator, f32);
    defer allocator.free(flat_alloc);
    try std.testing.expectEqual(@as(usize, 6), flat_alloc.len);
    try std.testing.expectEqual(@as(f32, 4.0), flat_alloc[3]);

    // Error paths: scalar read from non-scalar
    const nons: api.Tensor = try ctx.tensor(.f32, &[_]usize{2});
    try std.testing.expectError(manager_mod.StorageError.InvalidArgument, nons.readScalar(f32));

    // Error paths: type mismatch write
    const tf16: api.Tensor = try ctx.tensor(.f16, &[_]usize{1});
    const bad: [1]f32 = .{1.0};
    try std.testing.expectError(manager_mod.StorageError.InvalidArgument, tf16.write(bad[0..]));
}

test "api: i32 tensor typed IO + compile+run copy" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    // Create an i32 tensor from a Zig array (dtype inference).
    const in_t: api.Tensor = try ctx.vector([_]i32{ 10, -3, 7, 42 });
    try std.testing.expectEqual(types.DType.i32, in_t.getDType());
    try std.testing.expectEqualSlices(usize, &[_]usize{4}, in_t.getShape());

    // Roundtrip read/write for i32.
    var tmp: [4]i32 = undefined;
    try in_t.read(&tmp);
    try std.testing.expectEqual(@as(i32, 10), tmp[0]);
    try std.testing.expectEqual(@as(i32, -3), tmp[1]);
    try std.testing.expectEqual(@as(i32, 7), tmp[2]);
    try std.testing.expectEqual(@as(i32, 42), tmp[3]);

    // Compile a tiny model: out = copy(in)
    var bld = api.Builder.init(allocator);
    defer bld.deinit();
    const X = try bld.param(in_t);
    const Y = try bld.copy(X);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Y});
    defer model.deinit();

    const out_t: api.Tensor = try model.runOutputTensor(0);
    try std.testing.expectEqual(types.DType.i32, out_t.getDType());
    try std.testing.expectEqualSlices(usize, &[_]usize{4}, out_t.getShape());

    var out_vals: [4]i32 = undefined;
    try out_t.read(&out_vals);
    try std.testing.expectEqualSlices(i32, &tmp, &out_vals);
}

test "api: reduceAxis mean over last dim" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    const x_t: api.Tensor = try ctx.fromArray([2][3]f32{
        .{ 1.0, 2.0, 3.0 },
        .{ 4.0, 5.0, 6.0 },
    });

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const X: api.TensorRef = try bld.param(x_t);
    const Y: api.TensorRef = try bld.reduceAxis(.mean, X, -1);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Y});
    defer model.deinit();

    const out_t: api.Tensor = try model.runOutputTensor(0);
    try std.testing.expectEqualSlices(usize, &[_]usize{2}, out_t.getShape());

    var out_vals: [2]f32 = undefined;
    try out_t.read(&out_vals);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out_vals[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), out_vals[1], 1e-6);
}

test "api: squeeze and unsqueeze roundtrip" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    const x_t: api.Tensor = try ctx.fromArray([1][2][1][3]f32{
        .{
            .{
                .{ 1.0, 2.0, 3.0 },
            },
            .{
                .{ 4.0, 5.0, 6.0 },
            },
        },
    });

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const X: api.TensorRef = try bld.param(x_t);

    const S: api.TensorRef = try bld.squeeze(X, null);
    const U0: api.TensorRef = try bld.unsqueeze(S, 0);
    const U1: api.TensorRef = try bld.unsqueeze(U0, 2);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{ U1, S });
    defer model.deinit();

    const out_u: api.Tensor = try model.runOutputTensor(0);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 2, 1, 3 }, out_u.getShape());

    var out_u_vals: [1 * 2 * 1 * 3]f32 = undefined;
    try out_u.read(&out_u_vals);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, out_u_vals[0..]);

    const out_s: api.Tensor = model.outputTensor(1);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 3 }, out_s.getShape());

    // Invalid squeeze axis (dimension != 1) is validated during infer/compile.
    var bld_bad = api.Builder.init(allocator);
    defer bld_bad.deinit();
    const X_bad: api.TensorRef = try bld_bad.param(x_t);
    const S_bad: api.TensorRef = try bld_bad.squeeze(X_bad, null);
    const Bad: api.TensorRef = try bld_bad.squeeze(S_bad, 1);
    try std.testing.expectError(error.ShapeMismatch, ctx.compile(&bld_bad, &[_]api.TensorRef{Bad}));
}

test "api.nn: Conv1D bind+forward" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    // NLC input: [batch=1, length=4, channels=1]
    const x_t: api.Tensor = try ctx.fromArray([1][4][1]f32{
        .{
            .{1.0},
            .{2.0},
            .{3.0},
            .{4.0},
        },
    });

    // Weight: [k=1, c_in=1, c_out=1] => identity for stride=1, pad=0.
    const w_t: api.Tensor = try ctx.fromArray([1][1][1]f32{
        .{.{1.0}},
    });

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const X: api.TensorRef = try bld.param(x_t);

    const conv: nn.Conv1D = try nn.Conv1D.bind(&bld, w_t, null, .{
        .stride = 1,
        .dilation = 1,
        .pad_left = 0,
        .pad_right = 0,
        .pad_mode = .zero,
        .groups = 1,
    });

    const Y: api.TensorRef = try conv.forward(&bld, X);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Y});
    defer model.deinit();

    const out_t: api.Tensor = try model.runOutputTensor(0);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 4, 1 }, out_t.getShape());

    var out_vals: [4]f32 = undefined;
    try out_t.read(out_vals[0..]);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out_vals[0], 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out_vals[1], 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), out_vals[2], 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), out_vals[3], 0.0);
}

test "api: generic nd slice" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    const x_t: api.Tensor = try ctx.fromArray([2][3][4]f32{
        .{
            .{ 1.0, 2.0, 3.0, 4.0 },
            .{ 5.0, 6.0, 7.0, 8.0 },
            .{ 9.0, 10.0, 11.0, 12.0 },
        },
        .{
            .{ 13.0, 14.0, 15.0, 16.0 },
            .{ 17.0, 18.0, 19.0, 20.0 },
            .{ 21.0, 22.0, 23.0, 24.0 },
        },
    });

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const X: api.TensorRef = try bld.param(x_t);
    const S: api.TensorRef = try bld.slice(X, &[_]usize{ 1, 1, 0 }, &[_]usize{ 1, 2, 3 });

    var model = try ctx.compile(&bld, &[_]api.TensorRef{S});
    defer model.deinit();

    const out_t: api.Tensor = try model.runOutputTensor(0);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 2, 3 }, out_t.getShape());

    var out_vals: [1 * 2 * 3]f32 = undefined;
    try out_t.read(&out_vals);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 17.0, 18.0, 19.0, 21.0, 22.0, 23.0 }, out_vals[0..]);
}

test "api: concat and stack" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    const a_t: api.Tensor = try ctx.fromArray([2][2]f32{ .{ 1.0, 2.0 }, .{ 3.0, 4.0 } });
    const b_t: api.Tensor = try ctx.fromArray([2][1]f32{ .{10.0}, .{20.0} });

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const A: api.TensorRef = try bld.param(a_t);
    const B: api.TensorRef = try bld.param(b_t);

    const C: api.TensorRef = try bld.concat(&[_]api.TensorRef{ A, B }, 1);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{C});
    defer model.deinit();

    const out_c: api.Tensor = try model.runOutputTensor(0);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 3 }, out_c.getShape());
    var c_vals: [6]f32 = undefined;
    try out_c.read(&c_vals);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1.0, 2.0, 10.0, 3.0, 4.0, 20.0 }, c_vals[0..]);

    var bld2 = api.Builder.init(allocator);
    defer bld2.deinit();
    const A2: api.TensorRef = try bld2.param(a_t);
    const B2: api.TensorRef = try bld2.param(a_t);
    const S: api.TensorRef = try bld2.stack(&[_]api.TensorRef{ A2, B2 }, 0);

    var model2 = try ctx.compile(&bld2, &[_]api.TensorRef{S});
    defer model2.deinit();

    const out_s: api.Tensor = try model2.runOutputTensor(0);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 2, 2 }, out_s.getShape());
    var s_vals: [8]f32 = undefined;
    try out_s.read(&s_vals);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0 }, s_vals[0..]);
}

test "api: model outputCount/outputTensor + builder.name" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    const m_dim: usize = 2;
    const n_dim: usize = 3;

    const a_t: api.Tensor = try ctx.fromArray([2][2]f32{ .{ 1.0, 2.0 }, .{ 3.0, 4.0 } });
    const b_t: api.Tensor = try ctx.fromArray([2][3]f32{ .{ 0.5, -1.0, 2.0 }, .{ 1.5, 0.0, -0.5 } });

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const A0: api.TensorRef = try bld.param(a_t);
    const A: api.TensorRef = try bld.name(A0, "A");
    const B: api.TensorRef = try bld.param(b_t);

    const C: api.TensorRef = try bld.matmul(A, B, 1.0, 0.0);
    const Y: api.TensorRef = try bld.relu(C);
    const Mean: api.TensorRef = try bld.reduce(.mean, C);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{ Mean, Y });
    defer model.deinit();

    try std.testing.expectEqual(@as(usize, 2), model.outputCount());

    const out0_t: api.Tensor = try model.runOutputTensor(0);
    const out0: f32 = try out0_t.readScalar(f32);
    try std.testing.expect(std.math.isFinite(out0));

    const out1_t: api.Tensor = model.outputTensor(1);
    try std.testing.expectEqual(types.DType.f32, out1_t.getDType());
    try std.testing.expectEqualSlices(usize, &[_]usize{ m_dim, n_dim }, out1_t.getShape());

    var y_vals: [m_dim * n_dim]f32 = undefined;
    try out1_t.read(&y_vals);
    for (y_vals) |v0| try std.testing.expect(std.math.isFinite(v0));
}

test "api: fromPackedQuant rejects non-quant dtype" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    const dummy: [1]u8 = .{0};
    try std.testing.expectError(error.InvalidArgument, ctx.fromPackedQuant(.f32, &[_]usize{ 1, 1 }, dummy[0..]));
}

test "api.nn: mini resnet (conv/residual/linear/softmax)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    // Dimensions (NHWC conv): [N,H,W,C]
    const N: usize = 1;
    const H: usize = 4;
    const W: usize = 4;
    const C: usize = 3;
    const K: usize = 3;
    const Cout: usize = C; // keep channels constant for residual add
    const classes: usize = 5;

    // Allocate and fill deterministic inputs/weights.
    const x_vals: []f32 = try allocator.alloc(f32, N * H * W * C);
    defer allocator.free(x_vals);
    const w1_vals: []f32 = try allocator.alloc(f32, K * K * C * Cout);
    defer allocator.free(w1_vals);
    const b1_vals: []f32 = try allocator.alloc(f32, Cout);
    defer allocator.free(b1_vals);
    const w2_vals: []f32 = try allocator.alloc(f32, K * K * C * Cout);
    defer allocator.free(w2_vals);
    const b2_vals: []f32 = try allocator.alloc(f32, Cout);
    defer allocator.free(b2_vals);
    const fc_w_vals: []f32 = try allocator.alloc(f32, 1 * classes);
    defer allocator.free(fc_w_vals);
    const fc_b_vals: []f32 = try allocator.alloc(f32, classes);
    defer allocator.free(fc_b_vals);

    for (0..x_vals.len) |i| {
        const m11: usize = i % 11;
        x_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast(m11)))) - 5.0) * 0.05;
    }
    for (0..w1_vals.len) |i| {
        const m13: usize = i % 13;
        w1_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast(m13)))) - 6.0) * 0.01;
    }
    for (0..w2_vals.len) |i| {
        const m17: usize = i % 17;
        w2_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast(m17)))) - 8.0) * 0.01;
    }
    for (0..b1_vals.len) |i| b1_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast(i)) - 1))) * 0.02;
    for (0..b2_vals.len) |i| b2_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast(i)) - 1))) * 0.015;
    for (0..fc_w_vals.len) |i| {
        const m19: usize = i % 19;
        fc_w_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast(m19)))) - 9.0) * 0.005;
    }
    for (0..fc_b_vals.len) |i| fc_b_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast(i)) - 2))) * 0.01;

    // Reference implementation (naive conv2d NHWC, residual block, reduce->linear, softmax).
    var y1_ref: [N * H * W * Cout]f32 = undefined;
    var a1_ref: [N * H * W * Cout]f32 = undefined;
    var y2_ref: [N * H * W * Cout]f32 = undefined;
    var y3_ref: [N * H * W * Cout]f32 = undefined;
    var logits_ref: [classes]f32 = undefined;
    var probs_ref: [classes]f32 = undefined;

    const pad: isize = 1;
    // conv1
    for (0..N) |n0| {
        for (0..H) |oh| {
            for (0..W) |ow| {
                for (0..Cout) |co| {
                    var acc: f32 = b1_vals[co];
                    for (0..K) |kh| {
                        const ih_i: isize = @as(isize, @intCast(oh)) + @as(isize, @intCast(kh)) - pad;
                        if (ih_i < 0 or ih_i >= @as(isize, @intCast(H))) continue;
                        for (0..K) |kw| {
                            const iw_i: isize = @as(isize, @intCast(ow)) + @as(isize, @intCast(kw)) - pad;
                            if (iw_i < 0 or iw_i >= @as(isize, @intCast(W))) continue;
                            const ih: usize = @intCast(ih_i);
                            const iw: usize = @intCast(iw_i);
                            for (0..C) |ci| {
                                const x_idx: usize = (((n0 * H + ih) * W + iw) * C) + ci;
                                const w_idx: usize = (((kh * K + kw) * C + ci) * Cout) + co;
                                acc += x_vals[x_idx] * w1_vals[w_idx];
                            }
                        }
                    }
                    const out_idx: usize = (((n0 * H + oh) * W + ow) * Cout) + co;
                    y1_ref[out_idx] = acc;
                }
            }
        }
    }

    // relu
    for (0..y1_ref.len) |i| {
        const v: f32 = y1_ref[i];
        a1_ref[i] = if (v > 0.0) v else 0.0;
    }

    // conv2
    for (0..N) |n0| {
        for (0..H) |oh| {
            for (0..W) |ow| {
                for (0..Cout) |co| {
                    var acc: f32 = b2_vals[co];
                    for (0..K) |kh| {
                        const ih_i: isize = @as(isize, @intCast(oh)) + @as(isize, @intCast(kh)) - pad;
                        if (ih_i < 0 or ih_i >= @as(isize, @intCast(H))) continue;
                        for (0..K) |kw| {
                            const iw_i: isize = @as(isize, @intCast(ow)) + @as(isize, @intCast(kw)) - pad;
                            if (iw_i < 0 or iw_i >= @as(isize, @intCast(W))) continue;
                            const ih: usize = @intCast(ih_i);
                            const iw: usize = @intCast(iw_i);
                            for (0..C) |ci| {
                                const x_idx: usize = (((n0 * H + ih) * W + iw) * C) + ci;
                                const w_idx: usize = (((kh * K + kw) * C + ci) * Cout) + co;
                                acc += a1_ref[x_idx] * w2_vals[w_idx];
                            }
                        }
                    }
                    const out_idx: usize = (((n0 * H + oh) * W + ow) * Cout) + co;
                    y2_ref[out_idx] = acc;
                }
            }
        }
    }

    // residual add + relu
    for (0..y2_ref.len) |i| {
        const sum: f32 = y2_ref[i] + x_vals[i];
        y3_ref[i] = if (sum > 0.0) sum else 0.0;
    }

    // reduce mean over all elements (global average)
    var mean_ref: f32 = 0.0;
    for (0..(H * W * Cout)) |i| mean_ref += y3_ref[i];
    mean_ref /= @as(f32, @floatFromInt(H * W * Cout));

    // linear: logits = mean_ref @ W + b, where W is [1, classes]
    for (0..classes) |j| {
        var acc: f32 = fc_b_vals[j];
        // k==1
        acc += mean_ref * fc_w_vals[j];
        logits_ref[j] = acc;
    }

    // softmax (stable)
    var maxv: f32 = logits_ref[0];
    for (1..classes) |i| {
        if (logits_ref[i] > maxv) maxv = logits_ref[i];
    }
    var denom: f32 = 0.0;
    for (0..classes) |i| {
        const e: f32 = std.math.exp(logits_ref[i] - maxv);
        probs_ref[i] = e;
        denom += e;
    }
    for (0..classes) |i| probs_ref[i] /= denom;

    // API path.
    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    const x_t: api.Tensor = try ctx.from(&[_]usize{ N, H, W, C }, x_vals);
    const w1_t: api.Tensor = try ctx.from(&[_]usize{ K, K, C, Cout }, w1_vals);
    const b1_t: api.Tensor = try ctx.from(&[_]usize{Cout}, b1_vals);
    const w2_t: api.Tensor = try ctx.from(&[_]usize{ K, K, C, Cout }, w2_vals);
    const b2_t: api.Tensor = try ctx.from(&[_]usize{Cout}, b2_vals);
    const fc_w_t: api.Tensor = try ctx.from(&[_]usize{ 1, classes }, fc_w_vals);
    const fc_b_t: api.Tensor = try ctx.from(&[_]usize{classes}, fc_b_vals);

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const X: api.TensorRef = try bld.param(x_t);

    const pad_opts: nn.Conv2D.Options = .{ .pad_top = 1, .pad_bottom = 1, .pad_left = 1, .pad_right = 1 };
    const conv1: nn.Conv2D = try nn.Conv2D.bind(&bld, w1_t, b1_t, pad_opts);
    const conv2: nn.Conv2D = try nn.Conv2D.bind(&bld, w2_t, b2_t, pad_opts);
    const block: nn.ResBlock2D = .{ .conv1 = conv1, .conv2 = conv2 };
    const fc: nn.Linear = try nn.Linear.bind(&bld, fc_w_t, fc_b_t);

    const Y: api.TensorRef = try block.forward(&bld, X);
    const Mean: api.TensorRef = try bld.reduce(.mean, Y);
    const Mean2: api.TensorRef = try bld.reshape(Mean, &[_]usize{ 1, 1 });
    const Logits: api.TensorRef = try fc.forward(&bld, Mean2);
    const Probs: api.TensorRef = try nn.softmaxLastDim(&bld, Logits);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Probs});
    defer model.deinit();

    const out_t: api.Tensor = try model.runOutputTensor(0);
    var probs: [classes]f32 = undefined;
    try out_t.read(&probs);

    var sum: f32 = 0.0;
    for (probs, 0..) |p, i| {
        try std.testing.expect(std.math.isFinite(p));
        try std.testing.expect(p >= 0.0);
        sum += p;
        try std.testing.expectApproxEqAbs(probs_ref[i], p, 2e-3);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 2e-3);
}

test "api.nn: lstm cell (single step)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const batch: usize = 2;
    const input_size: usize = 3;
    const hidden: usize = 4;
    const gate_dim: usize = 4 * hidden;

    // Deterministic values.
    var x_vals: [batch * input_size]f32 = undefined;
    var h0_vals: [batch * hidden]f32 = undefined;
    var c0_vals: [batch * hidden]f32 = undefined;
    var w_ih_vals: [input_size * gate_dim]f32 = undefined;
    var w_hh_vals: [hidden * gate_dim]f32 = undefined;
    var b_ih_vals: [gate_dim]f32 = undefined;
    var b_hh_vals: [gate_dim]f32 = undefined;

    for (0..x_vals.len) |i| x_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3))) * 0.1;
    for (0..h0_vals.len) |i| h0_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 5)) - 2))) * 0.07;
    for (0..c0_vals.len) |i| c0_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 9)) - 4))) * 0.05;
    for (0..w_ih_vals.len) |i| w_ih_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5))) * 0.02;
    for (0..w_hh_vals.len) |i| w_hh_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6))) * 0.015;
    for (0..b_ih_vals.len) |i| b_ih_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3))) * 0.01;
    for (0..b_hh_vals.len) |i| b_hh_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 5)) - 2))) * 0.008;

    // Reference implementation.
    var gates_ref: [batch * gate_dim]f32 = undefined;
    var i_ref: [batch * hidden]f32 = undefined;
    var f_ref: [batch * hidden]f32 = undefined;
    var g_ref: [batch * hidden]f32 = undefined;
    var o_ref: [batch * hidden]f32 = undefined;
    var c_ref: [batch * hidden]f32 = undefined;
    var h_ref: [batch * hidden]f32 = undefined;

    const sigmoid = struct {
        fn f(x: f32) f32 {
            return fast.sigmoidApproxF32(x);
        }
    }.f;

    // gates = x@w_ih + h0@w_hh + b_ih + b_hh
    for (0..batch) |b| {
        for (0..gate_dim) |j| {
            var acc: f32 = b_ih_vals[j] + b_hh_vals[j];
            for (0..input_size) |k| {
                acc += x_vals[b * input_size + k] * w_ih_vals[k * gate_dim + j];
            }
            for (0..hidden) |k| {
                acc += h0_vals[b * hidden + k] * w_hh_vals[k * gate_dim + j];
            }
            gates_ref[b * gate_dim + j] = acc;
        }
    }

    for (0..batch) |b| {
        for (0..hidden) |j| {
            const base: usize = b * gate_dim;
            i_ref[b * hidden + j] = sigmoid(gates_ref[base + 0 * hidden + j]);
            f_ref[b * hidden + j] = sigmoid(gates_ref[base + 1 * hidden + j]);
            g_ref[b * hidden + j] = fast.tanhApproxF32(gates_ref[base + 2 * hidden + j]);
            o_ref[b * hidden + j] = sigmoid(gates_ref[base + 3 * hidden + j]);
        }
    }

    for (0..batch) |b| {
        for (0..hidden) |j| {
            const idx: usize = b * hidden + j;
            const c_t: f32 = f_ref[idx] * c0_vals[idx] + i_ref[idx] * g_ref[idx];
            c_ref[idx] = c_t;
            h_ref[idx] = o_ref[idx] * fast.tanhApproxF32(c_t);
        }
    }

    // API path.
    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    const x_t: api.Tensor = try ctx.fromF32(&[_]usize{ batch, input_size }, x_vals[0..]);
    const h0_t: api.Tensor = try ctx.fromF32(&[_]usize{ batch, hidden }, h0_vals[0..]);
    const c0_t: api.Tensor = try ctx.fromF32(&[_]usize{ batch, hidden }, c0_vals[0..]);
    const w_ih_t: api.Tensor = try ctx.fromF32(&[_]usize{ input_size, gate_dim }, w_ih_vals[0..]);
    const w_hh_t: api.Tensor = try ctx.fromF32(&[_]usize{ hidden, gate_dim }, w_hh_vals[0..]);
    const b_ih_t: api.Tensor = try ctx.fromF32(&[_]usize{gate_dim}, b_ih_vals[0..]);
    const b_hh_t: api.Tensor = try ctx.fromF32(&[_]usize{gate_dim}, b_hh_vals[0..]);

    // Error path: wrong bias shape.
    const bad_bias: api.Tensor = try ctx.tensor(.f32, &[_]usize{gate_dim - 1});
    var tmp: [gate_dim - 1]f32 = .{0.0} ** (gate_dim - 1);
    try bad_bias.write(tmp[0..]);

    var bld_bad = api.Builder.init(allocator);
    defer bld_bad.deinit();
    try std.testing.expectError(error.InvalidArgument, nn.LSTMCell.bind(&bld_bad, w_ih_t, w_hh_t, b_ih_t, bad_bias));

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const X: api.TensorRef = try bld.param(x_t);
    const H0: api.TensorRef = try bld.param(h0_t);
    const C0: api.TensorRef = try bld.param(c0_t);

    const cell: nn.LSTMCell = try nn.LSTMCell.bind(&bld, w_ih_t, w_hh_t, b_ih_t, b_hh_t);
    const st: nn.LSTMState = try cell.forward(&bld, X, H0, C0);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{ st.h, st.c });
    defer model.deinit();

    const out_h_t: api.Tensor = try model.runOutputTensor(0);
    const out_c_t: api.Tensor = try model.runOutputTensor(1);
    try std.testing.expectEqualSlices(usize, &[_]usize{ batch, hidden }, out_h_t.getShape());
    try std.testing.expectEqualSlices(usize, &[_]usize{ batch, hidden }, out_c_t.getShape());

    var out_h: [batch * hidden]f32 = undefined;
    var out_c: [batch * hidden]f32 = undefined;
    try out_h_t.read(&out_h);
    try out_c_t.read(&out_c);

    for (0..out_h.len) |i| {
        try std.testing.expect(std.math.isFinite(out_h[i]));
        try std.testing.expect(std.math.isFinite(out_c[i]));
        try std.testing.expectApproxEqAbs(h_ref[i], out_h[i], 1e-5);
        try std.testing.expectApproxEqAbs(c_ref[i], out_c[i], 1e-5);
    }
}

test "api: complexAbsMean (split-complex abs mean over time)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const batch: usize = 2;
    const time: usize = 3;
    const cutoff: usize = 3;
    const chans2: usize = cutoff * 2;
    const out_channels: usize = 2;

    var stft_vals: [batch * time * chans2]f32 = undefined;
    for (0..stft_vals.len) |i| {
        stft_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5))) * 0.25;
    }

    // Reference: out[b,c] = mean_t sqrt(re^2 + im^2).
    var ref: [batch * out_channels]f32 = .{0.0} ** (batch * out_channels);
    for (0..batch) |b| {
        for (0..out_channels) |c| {
            var acc: f32 = 0.0;
            for (0..time) |t| {
                const base: usize = (b * time + t) * chans2;
                const re: f32 = stft_vals[base + c];
                const im: f32 = stft_vals[base + (c + cutoff)];
                acc += @sqrt(re * re + im * im);
            }
            ref[b * out_channels + c] = acc / @as(f32, @floatFromInt(time));
        }
    }

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    const stft_t: api.Tensor = try ctx.fromF32(&[_]usize{ batch, time, chans2 }, stft_vals[0..]);

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const STFT: api.TensorRef = try bld.param(stft_t);
    const Out: api.TensorRef = try bld.complexAbsMean(STFT, out_channels);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Out});
    defer model.deinit();

    const out_t: api.Tensor = try model.runOutputTensor(0);
    try std.testing.expectEqualSlices(usize, &[_]usize{ batch, out_channels }, out_t.getShape());

    var out_vals: [batch * out_channels]f32 = undefined;
    try out_t.read(&out_vals);

    for (0..out_vals.len) |i| {
        try std.testing.expect(std.math.isFinite(out_vals[i]));
        try std.testing.expectApproxEqAbs(ref[i], out_vals[i], 1e-6);
    }
}

test "api: exportModel + loadModel roundtrip executes without rebuilding architecture" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var export_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer export_ctx.deinit();

    const w_t: api.Tensor = try export_ctx.fromArray([2][3]f32{
        .{ 1.0, -2.0, 0.5 },
        .{ 3.0, 4.0, -1.5 },
    });

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try createTestFile(tmp.dir, "weights.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const X: api.TensorRef = try bld.name(try bld.input(.f32, &[_]usize{ 1, 2 }), "x");
    const W: api.TensorRef = try bld.param(w_t);
    const Y: api.TensorRef = try bld.matmul(X, W, 1.0, 0.0);
    try export_ctx.exportModel(file, &bld, &[_]api.NamedTensorRef{
        .{ .name = "y", .tensor = Y },
    }, .{
        .metadata = &[_]api.ExportMetadata{
            .{ .key = "arch", .value = "roundtrip-test" },
        },
    });

    var load_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer load_ctx.deinit();

    var model = try load_ctx.loadModel(file, .{});
    defer model.deinit();

    try std.testing.expectEqual(@as(usize, 1), model.inputNames().len);
    try std.testing.expectEqual(@as(usize, 1), model.outputNames().len);
    try std.testing.expectEqualStrings("x", model.inputNames()[0].name);
    try std.testing.expectEqualStrings("y", model.outputNames()[0].name);
    try std.testing.expectEqual(types.DType.f32, model.inputNames()[0].dtype);

    const x_t: api.Tensor = try load_ctx.fromArray([1][2]f32{.{ 2.0, -1.0 }});
    try model.bindInput("x", x_t);
    try model.run();

    const y_t: api.Tensor = try model.outputTensor("y");
    var y_vals: [3]f32 = undefined;
    try y_t.read(&y_vals);

    try std.testing.expectApproxEqAbs(@as(f32, -1.0), y_vals[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -8.0), y_vals[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), y_vals[2], 1e-6);
}

test "api: loadModel can swap initializers by debug name (overwrite + retarget)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var export_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer export_ctx.deinit();

    // Weight W: [2,3]
    const w_t: api.Tensor = try export_ctx.fromArray([2][3]f32{
        .{ 1.0, 2.0, 3.0 },
        .{ 4.0, 5.0, 6.0 },
    });

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try createTestFile(tmp.dir, "swap_init.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const X: api.TensorRef = try bld.name(try bld.input(.f32, &[_]usize{ 1, 2 }), "x");
    const W0: api.TensorRef = try bld.param(w_t);
    const W: api.TensorRef = try bld.name(W0, "W");
    const Y: api.TensorRef = try bld.matmul(X, W, 1.0, 0.0);

    try export_ctx.exportModel(file, &bld, &[_]api.NamedTensorRef{.{ .name = "y", .tensor = Y }}, .{});

    var load_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer load_ctx.deinit();

    var model = try load_ctx.loadModel(file, .{});
    defer model.deinit();

    // Sanity: W must be present in the persisted debug-name table.
    try std.testing.expect(model.findValueByDebugName("W") != null);
    const init_t: api.Tensor = try model.initializerTensorByDebugName("W");
    try std.testing.expectEqual(types.DType.f32, init_t.getDType());
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 3 }, init_t.getShape());

    const x_t: api.Tensor = try load_ctx.fromArray([1][2]f32{.{ 2.0, -1.0 }});
    try model.bindInput("x", x_t);
    try model.run();

    var y_vals: [3]f32 = undefined;
    {
        const y_t: api.Tensor = try model.outputTensor("y");
        try y_t.read(&y_vals);
    }

    // y = [2,-1] @ W => [ -2, -1, 0 ]
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), y_vals[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), y_vals[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), y_vals[2], 1e-6);

    // Overwrite initializer bytes in-place.
    const zero_w: api.Tensor = try load_ctx.fromArray([2][3]f32{
        .{ 0.0, 0.0, 0.0 },
        .{ 0.0, 0.0, 0.0 },
    });
    try model.overwriteInitializerByDebugName("W", zero_w);

    try model.run();
    {
        const y_t: api.Tensor = try model.outputTensor("y");
        try y_t.read(&y_vals);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), y_vals[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), y_vals[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), y_vals[2], 1e-6);

    // Retarget initializer tensor-id to a new tensor (and patch cached programs).
    var ones_w: api.Tensor = try load_ctx.tensor(.f32, &[_]usize{ 2, 3 });
    var ones_vals: [6]f32 = .{1.0} ** 6;
    try ones_w.write(ones_vals[0..]);
    try model.retargetInitializerByDebugName("W", ones_w);

    try model.run();
    {
        const y_t: api.Tensor = try model.outputTensor("y");
        try y_t.read(&y_vals);
    }
    // [2,-1] @ ones => [1,1,1]
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), y_vals[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), y_vals[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), y_vals[2], 1e-6);
}

test "api: loadModel can switch a linear head (overwrite head.w + head.b)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    // Export model with head0 params.
    var export_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer export_ctx.deinit();

    const head0_w: api.Tensor = try export_ctx.fromArray([2][3]f32{
        .{ 1.0, -2.0, 0.5 },
        .{ 3.0, 4.0, -1.5 },
    });
    const head0_b: api.Tensor = try export_ctx.fromArray([3]f32{ 0.25, -0.5, 1.0 });

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try createTestFile(tmp.dir, "switch_head.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const X: api.TensorRef = try bld.name(try bld.input(.f32, &[_]usize{ 1, 2 }), "x");

    var head: nn.Linear = try nn.Linear.bind(&bld, head0_w, head0_b);
    head.w = try bld.name(head.w, "head.w");
    if (head.b) |b0| head.b = try bld.name(b0, "head.b");

    const Y: api.TensorRef = try head.forward(&bld, X);
    try export_ctx.exportModel(file, &bld, &[_]api.NamedTensorRef{.{ .name = "y", .tensor = Y }}, .{});

    // Load and run with head0.
    var load_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer load_ctx.deinit();

    var model = try load_ctx.loadModel(file, .{});
    defer model.deinit();

    try std.testing.expect(model.findValueByDebugName("head.w") != null);
    try std.testing.expect(model.findValueByDebugName("head.b") != null);

    const x_t: api.Tensor = try load_ctx.fromArray([1][2]f32{.{ 2.0, -1.0 }});
    try model.bindInput("x", x_t);
    try model.run();
    try std.testing.expectEqual(@as(usize, 1), model.cacheEntryCount());

    var y_vals: [3]f32 = undefined;
    {
        const y_t: api.Tensor = try model.outputTensor("y");
        try y_t.read(&y_vals);
    }

    // [2, -1] @ head0_w + head0_b => [-0.75, -8.5, 3.5]
    try std.testing.expectApproxEqAbs(@as(f32, -0.75), y_vals[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -8.5), y_vals[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), y_vals[2], 1e-6);

    // Switch to head1 by overwriting both initializers.
    const head1_w: api.Tensor = try load_ctx.fromArray([2][3]f32{
        .{ 0.5, 1.0, -1.0 },
        .{ 2.0, -0.5, 0.25 },
    });
    const head1_b: api.Tensor = try load_ctx.fromArray([3]f32{ -1.0, 0.0, 0.5 });

    try model.overwriteInitializerByDebugName("head.w", head1_w);
    try model.overwriteInitializerByDebugName("head.b", head1_b);

    try model.run();
    try std.testing.expectEqual(@as(usize, 1), model.cacheEntryCount());

    {
        const y_t: api.Tensor = try model.outputTensor("y");
        try y_t.read(&y_vals);
    }

    // [2, -1] @ head1_w + head1_b => [-2, 2.5, -1.75]
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), y_vals[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), y_vals[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -1.75), y_vals[2], 1e-6);
}

test "api: loadWeights lets you load a backbone and attach different heads" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    // Export a small "backbone" package containing only backbone weights.
    // We'll then load only its weights and attach different heads in Zig.
    var export_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer export_ctx.deinit();

    const bb_w: api.Tensor = try export_ctx.fromArray([2][4]f32{
        .{ 1.0, 0.5, -1.0, 2.0 },
        .{ -2.0, 1.0, 0.25, -0.5 },
    });
    const bb_b: api.Tensor = try export_ctx.fromArray([4]f32{ 0.1, -0.2, 0.3, 0.0 });

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try createTestFile(tmp.dir, "backbone_only.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);

    {
        var bld = api.Builder.init(allocator);
        defer bld.deinit();

        const X: api.TensorRef = try bld.name(try bld.input(.f32, &[_]usize{ 1, 2 }), "x");
        var bb: nn.Linear = try nn.Linear.bind(&bld, bb_w, bb_b);
        bb.w = try bld.name(bb.w, "backbone.w");
        if (bb.b) |b0| bb.b = try bld.name(b0, "backbone.b");

        const Hidden: api.TensorRef = try bb.forward(&bld, X);
        try export_ctx.exportModel(file, &bld, &[_]api.NamedTensorRef{.{ .name = "hidden", .tensor = Hidden }}, .{});
    }

    // Load *only* weights from that package.
    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    var weights = try ctx.loadWeights(file, .{});
    defer weights.deinit();

    const bb_w_loaded: api.Tensor = try weights.initializerTensorByDebugName("backbone.w");
    const bb_b_loaded: api.Tensor = try weights.initializerTensorByDebugName("backbone.b");

    // Shared input.
    const x_t: api.Tensor = try ctx.fromArray([1][2]f32{.{ 2.0, -1.0 }});

    // Reference hidden = x@W + b.
    const hidden_ref: [4]f32 = .{
        2.0 * 1.0 + (-1.0) * (-2.0) + 0.1,
        2.0 * 0.5 + (-1.0) * (1.0) + (-0.2),
        2.0 * (-1.0) + (-1.0) * (0.25) + 0.3,
        2.0 * (2.0) + (-1.0) * (-0.5) + 0.0,
    };

    // 1) Attach a classification head: hidden@Wc + bc (4 -> 3)
    const cls_w: api.Tensor = try ctx.fromArray([4][3]f32{
        .{ 1.0, 0.0, -1.0 },
        .{ 0.5, 2.0, 1.0 },
        .{ -2.0, 1.0, 0.25 },
        .{ 0.0, -1.5, 0.5 },
    });
    const cls_b: api.Tensor = try ctx.fromArray([3]f32{ 0.0, 0.1, -0.2 });

    var bld_cls = api.Builder.init(allocator);
    defer bld_cls.deinit();
    const Xc: api.TensorRef = try bld_cls.name(try bld_cls.param(x_t), "x");
    var bb_cls: nn.Linear = try nn.Linear.bind(&bld_cls, bb_w_loaded, bb_b_loaded);
    const HiddenC: api.TensorRef = try bb_cls.forward(&bld_cls, Xc);
    var head_cls: nn.Linear = try nn.Linear.bind(&bld_cls, cls_w, cls_b);
    head_cls.w = try bld_cls.name(head_cls.w, "head.classifier.w");
    if (head_cls.b) |b0| head_cls.b = try bld_cls.name(b0, "head.classifier.b");
    const LogitsC: api.TensorRef = try head_cls.forward(&bld_cls, HiddenC);

    var model_cls = try ctx.compile(&bld_cls, &[_]api.TensorRef{LogitsC});
    defer model_cls.deinit();
    const out_cls_t: api.Tensor = try model_cls.runOutputTensor(0);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 3 }, out_cls_t.getShape());
    var out_cls: [3]f32 = undefined;
    try out_cls_t.read(&out_cls);

    const cls_ref: [3]f32 = .{
        hidden_ref[0] * 1.0 + hidden_ref[1] * 0.5 + hidden_ref[2] * (-2.0) + hidden_ref[3] * 0.0 + 0.0,
        hidden_ref[0] * 0.0 + hidden_ref[1] * 2.0 + hidden_ref[2] * 1.0 + hidden_ref[3] * (-1.5) + 0.1,
        hidden_ref[0] * (-1.0) + hidden_ref[1] * 1.0 + hidden_ref[2] * 0.25 + hidden_ref[3] * 0.5 + (-0.2),
    };
    for (0..3) |i| try std.testing.expectApproxEqAbs(cls_ref[i], out_cls[i], 1e-6);

    // 2) Attach a QA head: two separate linear projections (4 -> 2)
    const qa_w_s: api.Tensor = try ctx.fromArray([4][2]f32{
        .{ 1.0, 0.0 },
        .{ 0.0, 1.0 },
        .{ 0.5, -0.5 },
        .{ -1.0, 2.0 },
    });
    const qa_w_e: api.Tensor = try ctx.fromArray([4][2]f32{
        .{ -1.0, 0.25 },
        .{ 2.0, -0.5 },
        .{ 0.0, 1.0 },
        .{ 1.5, -1.0 },
    });

    var bld_qa = api.Builder.init(allocator);
    defer bld_qa.deinit();
    const Xq: api.TensorRef = try bld_qa.name(try bld_qa.param(x_t), "x");
    var bb_qa: nn.Linear = try nn.Linear.bind(&bld_qa, bb_w_loaded, bb_b_loaded);
    const HiddenQ: api.TensorRef = try bb_qa.forward(&bld_qa, Xq);

    const Ws: api.TensorRef = try bld_qa.name(try bld_qa.param(qa_w_s), "head.qa.start.w");
    const We: api.TensorRef = try bld_qa.name(try bld_qa.param(qa_w_e), "head.qa.end.w");
    const Start: api.TensorRef = try bld_qa.matmul(HiddenQ, Ws, 1.0, 0.0);
    const End: api.TensorRef = try bld_qa.matmul(HiddenQ, We, 1.0, 0.0);

    var model_qa = try ctx.compile(&bld_qa, &[_]api.TensorRef{ Start, End });
    defer model_qa.deinit();
    const out_s_t: api.Tensor = try model_qa.runOutputTensor(0);
    const out_e_t: api.Tensor = model_qa.outputTensor(1);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 2 }, out_s_t.getShape());
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 2 }, out_e_t.getShape());

    var out_s: [2]f32 = undefined;
    var out_e: [2]f32 = undefined;
    try out_s_t.read(&out_s);
    try out_e_t.read(&out_e);

    const qa_s_ref: [2]f32 = .{
        hidden_ref[0] * 1.0 + hidden_ref[1] * 0.0 + hidden_ref[2] * 0.5 + hidden_ref[3] * (-1.0),
        hidden_ref[0] * 0.0 + hidden_ref[1] * 1.0 + hidden_ref[2] * (-0.5) + hidden_ref[3] * 2.0,
    };
    const qa_e_ref: [2]f32 = .{
        hidden_ref[0] * (-1.0) + hidden_ref[1] * 2.0 + hidden_ref[2] * 0.0 + hidden_ref[3] * 1.5,
        hidden_ref[0] * 0.25 + hidden_ref[1] * (-0.5) + hidden_ref[2] * 1.0 + hidden_ref[3] * (-1.0),
    };

    for (0..2) |i| try std.testing.expectApproxEqAbs(qa_s_ref[i], out_s[i], 1e-6);
    for (0..2) |i| try std.testing.expectApproxEqAbs(qa_e_ref[i], out_e[i], 1e-6);
}

test "api: loadModel output aliases keep recurrent state internal" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var export_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer export_ctx.deinit();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try createTestFile(tmp.dir, "aliased_state.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const X = try bld.name(try bld.input(.f32, &[_]usize{ 1, 2 }), "x");
    const State = try bld.name(try bld.input(.f32, &[_]usize{ 1, 2 }), "state");
    const Sum = try bld.add(X, State);
    const NextState = try bld.copy(Sum);

    try export_ctx.exportModel(file, &bld, &[_]api.NamedTensorRef{
        .{ .name = "y", .tensor = Sum },
        .{ .name = "next_state", .tensor = NextState },
    }, .{
        .output_aliases = &[_]api.OutputAlias{
            .{ .input_name = "state", .output_name = "next_state" },
        },
    });

    var load_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer load_ctx.deinit();

    var model = try load_ctx.loadModel(file, .{});
    defer model.deinit();

    try std.testing.expectEqual(@as(usize, 1), model.outputAliases().len);
    try std.testing.expectEqualStrings("state", model.outputAliases()[0].input_name);
    try std.testing.expectEqualStrings("next_state", model.outputAliases()[0].output_name);

    const x_t = try load_ctx.fromArray([1][2]f32{.{ 1.0, 2.0 }});
    const state_t = try load_ctx.fromArray([1][2]f32{.{ 0.0, 0.0 }});
    try model.bindInput("x", x_t);
    try model.bindInput("state", state_t);
    try model.run();

    try std.testing.expectEqual(@as(usize, 1), model.cacheEntryCount());

    const next_state_t = try model.outputTensor("next_state");

    var state_vals: [2]f32 = undefined;
    try next_state_t.read(&state_vals);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), state_vals[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), state_vals[1], 1e-6);

    const y_t = try model.outputTensor("y");
    var y_vals: [2]f32 = undefined;
    try y_t.read(&y_vals);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), y_vals[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), y_vals[1], 1e-6);

    try model.run();
    try std.testing.expectEqual(@as(usize, 1), model.cacheEntryCount());
    const next_state_t_second = try model.outputTensor("next_state");
    try std.testing.expectEqual(next_state_t.id, next_state_t_second.id);
    try next_state_t_second.read(&state_vals);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), state_vals[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), state_vals[1], 1e-6);

    const replacement_state = try load_ctx.fromArray([1][2]f32{.{ 5.0, 6.0 }});
    try model.bindInput("state", replacement_state);
    try model.run();
    try std.testing.expectEqual(@as(usize, 1), model.cacheEntryCount());

    const replaced_next_state = try model.outputTensor("next_state");
    try std.testing.expectEqual(next_state_t.id, replaced_next_state.id);
    try replaced_next_state.read(&state_vals);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), state_vals[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), state_vals[1], 1e-6);
}

test "api: exportModelPathAbsolute + loadModelPathAbsolute support symbolic shapes and cache reuse" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var export_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer export_ctx.deinit();

    const w_t: api.Tensor = try export_ctx.fromArray([3][2]f32{
        .{ 1.0, 2.0 },
        .{ 3.0, 4.0 },
        .{ 5.0, 6.0 },
    });

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const absolute_path: [:0]u8 = try tmp.parent_dir.realPathFileAlloc(std.testing.io, &tmp.sub_path, allocator);
    defer allocator.free(absolute_path);
    const absolute_file_path: []const u8 = try std.fs.path.join(allocator, &.{ absolute_path, "dynamic_model.aion" });
    defer allocator.free(absolute_file_path);

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const X0 = try bld.input(.f32, &[_]usize{ 1, 3 });
    const X = try bld.name(X0, "x");
    const W: api.TensorRef = try bld.param(w_t);
    const Y: api.TensorRef = try bld.matmul(X, W, 1.0, 0.0);

    try export_ctx.exportModelPathAbsolute(absolute_file_path, &bld, &[_]api.NamedTensorRef{
        .{ .name = "y", .tensor = Y },
    }, .{
        .input_symbols = &[_]api.DimensionSymbol{
            .{ .tensor = X, .axis = 0, .name = "batch" },
        },
    });

    var load_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer load_ctx.deinit();

    var model = try load_ctx.loadModelPathAbsolute(absolute_file_path, .{});
    defer model.deinit();

    const x_first = try load_ctx.fromArray([2][3]f32{
        .{ 1.0, 0.0, 1.0 },
        .{ 0.5, -1.0, 2.0 },
    });
    try model.bindInput("x", x_first);
    try model.run();
    try std.testing.expectEqual(@as(usize, 1), model.cacheEntryCount());

    const y_first = try model.outputTensor("y");
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 2 }, y_first.getShape());
    var y_first_vals: [4]f32 = undefined;
    try y_first.read(&y_first_vals);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), y_first_vals[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), y_first_vals[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 7.5), y_first_vals[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), y_first_vals[3], 1e-6);

    const x_same_shape = try load_ctx.fromArray([2][3]f32{
        .{ 2.0, 1.0, 0.0 },
        .{ -1.0, 3.0, 0.5 },
    });
    try model.bindInput("x", x_same_shape);
    try model.run();
    try std.testing.expectEqual(@as(usize, 1), model.cacheEntryCount());

    const x_new_shape = try load_ctx.fromArray([4][3]f32{
        .{ 1.0, 0.0, 0.0 },
        .{ 0.0, 1.0, 0.0 },
        .{ 0.0, 0.0, 1.0 },
        .{ 1.0, 1.0, 1.0 },
    });
    try model.bindInput("x", x_new_shape);
    try model.run();
    try std.testing.expectEqual(@as(usize, 2), model.cacheEntryCount());

    const y_new = try model.outputTensor("y");
    try std.testing.expectEqualSlices(usize, &[_]usize{ 4, 2 }, y_new.getShape());
}

test "api.module: vtable dynamic module can build graph" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    const x_t: api.Tensor = try ctx.fromArray([1][3]f32{.{ 1.0, 2.0, 3.0 }});
    const b_t: api.Tensor = try ctx.fromArray([1][3]f32{.{ 0.5, -1.0, 2.0 }});

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const X: api.TensorRef = try bld.param(x_t);

    const DynBiasAdd = struct {
        bias: api.TensorRef,

        fn name(ctx0: *const anyopaque) []const u8 {
            _ = ctx0;
            return "bias_add_dyn";
        }

        fn forward(ctx0: *anyopaque, bld0: *api.Builder, input: api.TensorRef) anyerror!api.TensorRef {
            const self: *@This() = @ptrCast(@alignCast(ctx0));
            return bld0.add(input, self.bias);
        }

        fn deinit(ctx0: *anyopaque, alloc: std.mem.Allocator) void {
            _ = alloc;
            const self: *@This() = @ptrCast(@alignCast(ctx0));
            self.* = undefined;
        }

        pub const vtable: api.ModuleDyn.VTable = .{
            .name = name,
            .forward = forward,
            .deinit = deinit,
        };
    };

    var dyn_mod_state: DynBiasAdd = .{ .bias = try bld.param(b_t) };
    var dyn_mod: api.ModuleDyn = api.ModuleDyn.init(&dyn_mod_state, &DynBiasAdd.vtable);
    defer dyn_mod.deinit(allocator);

    try std.testing.expectEqualStrings("bias_add_dyn", dyn_mod.name());

    const Y: api.TensorRef = try dyn_mod.forward(&bld, X);

    // ModuleDyn.forward should auto-scope when no scope is active.
    try std.testing.expect(bld.valueName(Y) != null);
    try std.testing.expectEqualStrings("bias_add_dyn#0/add#0", bld.valueName(Y).?);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Y});
    defer model.deinit();

    const y_t: api.Tensor = try model.runOutputTensor(0);
    var y_vals: [3]f32 = undefined;
    try y_t.read(&y_vals);

    try std.testing.expectApproxEqAbs(@as(f32, 1.5), y_vals[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), y_vals[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), y_vals[2], 1e-6);
}

test "api.module: isModuleType detects bind+forward protocol" {
    const Good = struct {
        pub fn bind() void {}
        pub fn forward() void {}
    };

    const MissingBind = struct {
        pub fn forward() void {}
    };

    try std.testing.expect(api.isModuleType(nn.Linear));
    try std.testing.expect(api.isModuleType(Good));
    try std.testing.expect(!api.isModuleType(MissingBind));
}

test "api.module: moduleDynFrom converts nn.Linear" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    const x_t: api.Tensor = try ctx.fromArray([1][2]f32{.{ 2.0, -1.0 }});
    const w_t: api.Tensor = try ctx.fromArray([2][3]f32{
        .{ 1.0, -2.0, 0.5 },
        .{ 3.0, 4.0, -1.5 },
    });
    const b_t: api.Tensor = try ctx.fromArray([3]f32{ 0.25, -0.5, 1.0 });

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const X: api.TensorRef = try bld.param(x_t);

    var linear: nn.Linear = try nn.Linear.bind(&bld, w_t, b_t);
    var dyn_mod: api.ModuleDyn = api.moduleDynFrom(nn.Linear, &linear);

    try std.testing.expect(api.isForwardModuleType(nn.Linear));
    try std.testing.expectEqualStrings("Linear", dyn_mod.name());

    const Y: api.TensorRef = try dyn_mod.forward(&bld, X);

    // ModuleDyn.forward should auto-scope, and nn.Linear.forward should *not*
    // start a nested scope when one is already active.
    try std.testing.expect(bld.valueName(Y) != null);
    try std.testing.expectEqualStrings("Linear#0/broadcast_add_last_dim#1", bld.valueName(Y).?);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Y});
    defer model.deinit();

    const out_t: api.Tensor = try model.runOutputTensor(0);
    var out: [3]f32 = undefined;
    try out_t.read(&out);

    // [2, -1] @ W + b => [-0.75, -8.5, 3.5]
    try std.testing.expectApproxEqAbs(@as(f32, -0.75), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -8.5), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), out[2], 1e-6);
}

test "api: module scopes auto-generate persisted debug names (nn.Linear)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var export_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer export_ctx.deinit();

    const w_t: api.Tensor = try export_ctx.fromArray([2][3]f32{
        .{ 1.0, -2.0, 0.5 },
        .{ 3.0, 4.0, -1.5 },
    });

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try createTestFile(tmp.dir, "infer_modules.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const X: api.TensorRef = try bld.name(try bld.input(.f32, &[_]usize{ 1, 2 }), "x");
    const linear: nn.Linear = try nn.Linear.bind(&bld, w_t, null);
    const Y: api.TensorRef = try linear.forward(&bld, X);

    try std.testing.expect(bld.valueName(Y) != null);
    try std.testing.expectEqualStrings("Linear#0/matmul#0", bld.valueName(Y).?);

    try export_ctx.exportModel(file, &bld, &[_]api.NamedTensorRef{
        .{ .name = "y", .tensor = Y },
    }, .{});

    var pkg = try package_file.readPackageFile(allocator, file);
    defer pkg.deinit();

    var found: bool = false;
    for (pkg.debug_names) |entry| {
        if (entry.value == Y.value) {
            found = true;
            try std.testing.expectEqualStrings("Linear#0/matmul#0", entry.name);
        }
    }
    try std.testing.expect(found);
}

test "api.module: custom module can use introspection scope helpers" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var export_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer export_ctx.deinit();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try createTestFile(tmp.dir, "custom_helper_scope.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);

    const Custom = struct {
        gain: api.TensorRef,

        fn forwardInner(self: @This(), bld: *api.Builder, x: api.TensorRef) api.Builder.Error!api.TensorRef {
            const gx: api.TensorRef = try bld.broadcastAddLastDim(x, self.gain);
            return bld.relu(gx);
        }

        pub fn bind(bld: *api.Builder, gain_t: api.Tensor) api.Builder.Error!@This() {
            return .{ .gain = try bld.param(gain_t) };
        }

        pub fn forward(self: @This(), bld: *api.Builder, x: api.TensorRef) api.Builder.Error!api.TensorRef {
            return api.withModuleScope(@This(), bld, null, forwardInner, .{ self, bld, x });
        }
    };

    // Build custom graph and export without explicit module declarations.
    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const x_t: api.Tensor = try export_ctx.fromArray([1][2]f32{.{ -2.0, 3.0 }});
    const gain_t: api.Tensor = try export_ctx.fromArray([2]f32{ 1.0, 1.0 });
    const X: api.TensorRef = try bld.param(x_t);

    const cmod: Custom = try Custom.bind(&bld, gain_t);
    const Y: api.TensorRef = try cmod.forward(&bld, X);

    try std.testing.expectEqualStrings("Custom", api.moduleTypeName(Custom));
    try std.testing.expect(bld.valueName(Y) != null);
    try std.testing.expectEqualStrings("Custom#0/relu#1", bld.valueName(Y).?);

    try export_ctx.exportModel(file, &bld, &[_]api.NamedTensorRef{.{ .name = "y", .tensor = Y }}, .{});

    var pkg = try package_file.readPackageFile(allocator, file);
    defer pkg.deinit();

    var found: bool = false;
    for (pkg.debug_names) |entry| {
        if (entry.value == Y.value) {
            found = true;
            try std.testing.expectEqualStrings("Custom#0/relu#1", entry.name);
        }
    }
    try std.testing.expect(found);
}

test "api.module: withModuleScope closes scope on error" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const Dummy = struct {};

    const Impl = struct {
        fn fail(bld0: *api.Builder) api.Builder.Error!void {
            _ = bld0;
            return api.Builder.Error.InvalidArgument;
        }
    };

    try std.testing.expect(!bld.hasActiveScope());
    try std.testing.expectError(api.Builder.Error.InvalidArgument, api.withModuleScope(Dummy, &bld, null, Impl.fail, .{&bld}));
    try std.testing.expect(!bld.hasActiveScope());
}

test "api: explicit builder scope prefixes debug names" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var export_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer export_ctx.deinit();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try createTestFile(tmp.dir, "infer_custom_module.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const X: api.TensorRef = try bld.name(try bld.input(.f32, &[_]usize{ 1, 3 }), "x");
    const scope = try bld.beginScope("my.custom.module");
    defer bld.endScope(scope);
    const Y1: api.TensorRef = try bld.relu(X);
    const Y2: api.TensorRef = try bld.mul(Y1, Y1);

    try std.testing.expectEqualStrings("my.custom.module/relu#0", bld.valueName(Y1).?);
    try std.testing.expectEqualStrings("my.custom.module/mul#1", bld.valueName(Y2).?);

    try export_ctx.exportModel(file, &bld, &[_]api.NamedTensorRef{
        .{ .name = "y", .tensor = Y2 },
    }, .{});

    var pkg = try package_file.readPackageFile(allocator, file);
    defer pkg.deinit();

    var found: bool = false;
    for (pkg.debug_names) |entry| {
        if (entry.value == Y2.value) {
            found = true;
            try std.testing.expectEqualStrings("my.custom.module/mul#1", entry.name);
        }
    }
    try std.testing.expect(found);
}

test "api: explicit custom module scope overrides nn auto module scope" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var export_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer export_ctx.deinit();

    const w_t: api.Tensor = try export_ctx.fromArray([2][3]f32{
        .{ 1.0, -2.0, 0.5 },
        .{ 3.0, 4.0, -1.5 },
    });

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try createTestFile(tmp.dir, "custom_overrides_auto_module.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const X: api.TensorRef = try bld.name(try bld.input(.f32, &[_]usize{ 1, 2 }), "x");
    const linear: nn.Linear = try nn.Linear.bind(&bld, w_t, null);

    const scope = try bld.beginScope("user.block");
    defer bld.endScope(scope);
    const Y: api.TensorRef = try linear.forward(&bld, X);

    try std.testing.expect(bld.valueName(Y) != null);
    try std.testing.expectEqualStrings("user.block/matmul#0", bld.valueName(Y).?);

    try export_ctx.exportModel(file, &bld, &[_]api.NamedTensorRef{
        .{ .name = "y", .tensor = Y },
    }, .{});

    var pkg = try package_file.readPackageFile(allocator, file);
    defer pkg.deinit();

    var found: bool = false;
    for (pkg.debug_names) |entry| {
        if (entry.value == Y.value) {
            found = true;
            try std.testing.expectEqualStrings("user.block/matmul#0", entry.name);
        }
    }
    try std.testing.expect(found);
}

test "api: builder.param auto-generates persisted debug names" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var export_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer export_ctx.deinit();

    const w_t: api.Tensor = try export_ctx.fromArray([2][3]f32{
        .{ 1.0, 2.0, 3.0 },
        .{ 4.0, 5.0, 6.0 },
    });

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try createTestFile(tmp.dir, "param_names.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const X: api.TensorRef = try bld.name(try bld.input(.f32, &[_]usize{ 1, 2 }), "x");
    const W: api.TensorRef = try bld.param(w_t);

    const expected = try std.fmt.allocPrint(allocator, "param@{d}", .{W.value});
    defer allocator.free(expected);
    try std.testing.expect(bld.valueName(W) != null);
    try std.testing.expectEqualStrings(expected, bld.valueName(W).?);

    const Y: api.TensorRef = try bld.matmul(X, W, 1.0, 0.0);
    try export_ctx.exportModel(file, &bld, &[_]api.NamedTensorRef{.{ .name = "y", .tensor = Y }}, .{});

    var pkg = try package_file.readPackageFile(allocator, file);
    defer pkg.deinit();

    var found: bool = false;
    for (pkg.debug_names) |entry| {
        if (entry.value == W.value) {
            found = true;
            try std.testing.expectEqualStrings(expected, entry.name);
        }
    }
    try std.testing.expect(found);
}

test "api: kvCacheAppend mutates cache in-place" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    const cache_t: api.Tensor = try ctx.fromArray([1][1][4][2]f32{
        .{
            .{
                .{ 0.0, 1.0 },
                .{ 2.0, 3.0 },
                .{ 4.0, 5.0 },
                .{ 6.0, 7.0 },
            },
        },
    });
    const new_t: api.Tensor = try ctx.fromArray([1][1][2][2]f32{
        .{
            .{
                .{ 50.0, 51.0 },
                .{ 60.0, 61.0 },
            },
        },
    });
    const end_t: api.Tensor = try ctx.fromArray([1]i32{1});

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const Cache: api.TensorRef = try bld.param(cache_t);
    const New: api.TensorRef = try bld.param(new_t);
    const End: api.TensorRef = try bld.param(end_t);
    const Out: api.TensorRef = try bld.kvCacheAppend(Cache, New, End);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Out});
    defer model.deinit();

    const out_t: api.Tensor = try model.runOutputTensor(0);
    try std.testing.expectEqual(cache_t.id, out_t.id);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 1, 4, 2 }, out_t.getShape());

    var out_vals: [8]f32 = undefined;
    try out_t.read(&out_vals);
    try std.testing.expectEqualSlices(f32, &[_]f32{
        0.0,
        1.0,
        50.0,
        51.0,
        60.0,
        61.0,
        6.0,
        7.0,
    }, out_vals[0..]);
}

test "api: kvCacheAppend ring policy wraps" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{
        .thread_count = 1,
        .cache_config = .{ .ram_budget_bytes = 1 << 20 },
    });
    defer ctx.deinit();

    const cache_t: api.Tensor = try ctx.fromArray([1][1][4][1]f32{
        .{
            .{
                .{0.0},
                .{1.0},
                .{2.0},
                .{3.0},
            },
        },
    });
    const new_t: api.Tensor = try ctx.fromArray([1][1][2][1]f32{
        .{
            .{
                .{90.0},
                .{91.0},
            },
        },
    });
    const end_t: api.Tensor = try ctx.fromArray([1]i32{3});
    try ctx.setTensorKVCachePolicy(cache_t, .{ .ring = .{ .window_tokens = 4 } });

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const Cache: api.TensorRef = try bld.param(cache_t);
    const New: api.TensorRef = try bld.param(new_t);
    const End: api.TensorRef = try bld.param(end_t);
    const Out: api.TensorRef = try bld.kvCacheAppend(Cache, New, End);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Out});
    defer model.deinit();

    const out_t: api.Tensor = try model.runOutputTensor(0);
    try std.testing.expectEqual(cache_t.id, out_t.id);

    var out_vals: [4]f32 = undefined;
    try out_t.read(&out_vals);
    try std.testing.expectEqualSlices(f32, &[_]f32{
        91.0,
        1.0,
        2.0,
        90.0,
    }, out_vals[0..]);
}

test "api: kvCacheAppend growable policy expands physical capacity" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{
        .thread_count = 1,
        .cache_config = .{ .ram_budget_bytes = 1 << 20 },
    });
    defer ctx.deinit();

    const cache_t: api.Tensor = try ctx.fromArray([1][1][4][1]f32{
        .{
            .{
                .{0.0},
                .{1.0},
                .{2.0},
                .{3.0},
            },
        },
    });
    const new_t: api.Tensor = try ctx.fromArray([1][1][2][1]f32{
        .{
            .{
                .{90.0},
                .{91.0},
            },
        },
    });
    const end_t: api.Tensor = try ctx.fromArray([1]i32{5});

    try ctx.setTensorKVCachePolicy(cache_t, .{ .growable = .{ .initial_capacity_tokens = 2, .growth_numerator = 2, .growth_denominator = 1 } });

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const Cache: api.TensorRef = try bld.param(cache_t);
    const New: api.TensorRef = try bld.param(new_t);
    const End: api.TensorRef = try bld.param(end_t);
    const Out: api.TensorRef = try bld.kvCacheAppend(Cache, New, End);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Out});
    defer model.deinit();

    const out_t: api.Tensor = try model.runOutputTensor(0);
    try std.testing.expectEqual(cache_t.id, out_t.id);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 1, 8, 1 }, out_t.getShape());

    var out_vals: [8]f32 = undefined;
    try out_t.read(&out_vals);
    try std.testing.expectEqualSlices(f32, &[_]f32{
        0.0,
        1.0,
        2.0,
        3.0,
        0.0,
        90.0,
        91.0,
        0.0,
    }, out_vals[0..]);
}

test "api: multiHeadAttentionCached matches deterministic windowed-causal averages" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    // Shapes:
    // q:       [B=1, Lq=2, Hq=4, Dk=2]
    // k_cache: [B=1, Hkv=2, T=6, Dk=2]
    // v_cache: [B=1, Hkv=2, T=6, Dv=1]
    // positions: [1,2], end_index: [1]
    var q_vals: [1 * 2 * 4 * 2]f32 = .{0.0} ** (1 * 2 * 4 * 2);
    var k_vals: [1 * 2 * 6 * 2]f32 = .{0.0} ** (1 * 2 * 6 * 2);
    var v_vals: [1 * 2 * 6 * 1]f32 = .{0.0} ** (1 * 2 * 6 * 1);

    // kv-head 0 tokens => [1,2,3,4,5,6]
    // kv-head 1 tokens => [10,20,30,40,50,60]
    for (0..6) |t| {
        v_vals[(0 * 2 * 6 * 1) + (0 * 6 * 1) + (t * 1)] = @as(f32, @floatFromInt(@as(i32, @intCast(t + 1))));
        v_vals[(0 * 2 * 6 * 1) + (1 * 6 * 1) + (t * 1)] = @as(f32, @floatFromInt(@as(i32, @intCast((t + 1) * 10))));
    }

    const pos_t: api.Tensor = try ctx.fromArray([1][2]i32{.{ 4, 5 }});
    const end_t: api.Tensor = try ctx.fromArray([1]i32{6});
    const q_t: api.Tensor = try ctx.fromF32(&[_]usize{ 1, 2, 4, 2 }, q_vals[0..]);
    const k_t: api.Tensor = try ctx.fromF32(&[_]usize{ 1, 2, 6, 2 }, k_vals[0..]);
    const v_t: api.Tensor = try ctx.fromF32(&[_]usize{ 1, 2, 6, 1 }, v_vals[0..]);

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const Q: api.TensorRef = try bld.param(q_t);
    const K: api.TensorRef = try bld.param(k_t);
    const V: api.TensorRef = try bld.param(v_t);
    const Pos: api.TensorRef = try bld.param(pos_t);
    const End: api.TensorRef = try bld.param(end_t);

    const Out: api.TensorRef = try bld.multiHeadAttentionCached(
        Q,
        K,
        V,
        Pos,
        End,
        1.0,
        true,
        3,
        0.0,
    );

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Out});
    defer model.deinit();

    const out_t: api.Tensor = try model.runOutputTensor(0);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 2, 4, 1 }, out_t.getShape());

    var out_vals: [1 * 2 * 4 * 1]f32 = undefined;
    try out_t.read(&out_vals);

    // With q/k all zeros, softmax is uniform over valid window.
    // l=0 (q_pos=4): t in [2,3,4]
    // l=1 (q_pos=5): t in [3,4,5]
    // hq 0,1 -> kv 0 ; hq 2,3 -> kv 1
    const expected: [8]f32 = .{
        4.0,
        4.0,
        40.0,
        40.0,
        5.0,
        5.0,
        50.0,
        50.0,
    };
    for (out_vals, expected) |got, want| {
        try std.testing.expect(std.math.isFinite(got));
        try std.testing.expectApproxEqAbs(want, got, 1e-6);
    }
}

test "api: multiHeadAttentionCached validates H_q % H_kv == 0" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    var q_vals: [1 * 1 * 3 * 2]f32 = .{0.0} ** (1 * 1 * 3 * 2);
    var k_vals: [1 * 2 * 4 * 2]f32 = .{0.0} ** (1 * 2 * 4 * 2);
    var v_vals: [1 * 2 * 4 * 1]f32 = .{0.0} ** (1 * 2 * 4 * 1);

    const q_t: api.Tensor = try ctx.fromF32(&[_]usize{ 1, 1, 3, 2 }, q_vals[0..]);
    const k_t: api.Tensor = try ctx.fromF32(&[_]usize{ 1, 2, 4, 2 }, k_vals[0..]);
    const v_t: api.Tensor = try ctx.fromF32(&[_]usize{ 1, 2, 4, 1 }, v_vals[0..]);
    const pos_t: api.Tensor = try ctx.fromArray([1][1]i32{.{0}});
    const end_t: api.Tensor = try ctx.fromArray([1]i32{1});

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const Q: api.TensorRef = try bld.param(q_t);
    const K: api.TensorRef = try bld.param(k_t);
    const V: api.TensorRef = try bld.param(v_t);
    const Pos: api.TensorRef = try bld.param(pos_t);
    const End: api.TensorRef = try bld.param(end_t);

    const Out: api.TensorRef = try bld.multiHeadAttentionCached(Q, K, V, Pos, End, 1.0, true, 0, 0.0);
    try std.testing.expectError(error.ShapeMismatch, ctx.compile(&bld, &[_]api.TensorRef{Out}));
}

// Opt-in load test: loads a pre-built `.aion` package from `AION_GEMMA_VERIFY_PATH`
// and asserts every materialized weight tensor is either `q8_0` or `f16`, with total
// bytes below a caller-supplied budget (`AION_GEMMA_MAX_GB`, default 6).
//
// Skipped unless the env var is set; used for manual validation of a real Gemma 4 E2B
// conversion without dragging multi-GB files into CI.
test "api: loaded package keeps weights quantized (opt-in via AION_GEMMA_VERIFY_PATH)" {
    // Use the OS page allocator rather than `std.testing.allocator` here: loading a
    // multi-GB package via the debug allocator pulls in bookkeeping that, combined
    // with the transient peak during initializer import, trips the process's address
    // space reservations on Windows. Page_allocator serves requests directly from the
    // OS which is the right fit for this one-shot big-file load.
    const allocator: std.mem.Allocator = std.heap.page_allocator;
    const env: std.process.Environ = .{ .block = .global };

    const path: []u8 = std.process.Environ.getAlloc(env, allocator, "AION_GEMMA_VERIFY_PATH") catch
        return error.SkipZigTest;
    defer allocator.free(path);
    if (path.len == 0) return error.SkipZigTest;

    const max_gb_raw: ?[]u8 = std.process.Environ.getAlloc(env, allocator, "AION_GEMMA_MAX_GB") catch null;
    defer if (max_gb_raw) |p| allocator.free(p);
    const max_gb: f64 = if (max_gb_raw) |raw|
        std.fmt.parseFloat(f64, raw) catch 6.0
    else
        6.0;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    // Narrow the failing step.
    {
        var io_backend: std.Io.Threaded = .init_single_threaded;
        const io = io_backend.io();
        const pf = try std.Io.Dir.openFileAbsolute(io, path, .{});
        defer pf.close(io);
        const raw = try package_file.readAlloc(allocator, pf);
        std.debug.print("[gemma-verify] readAlloc ok ({d} bytes)\n", .{raw.len});
        var parsed = try package_file.parseTakeOwned(allocator, raw);
        std.debug.print("[gemma-verify] parseTakeOwned ok (initializers={d} nodes={d})\n", .{ parsed.initializers.len, parsed.nodes.len });
        parsed.deinit();
        std.debug.print("[gemma-verify] parsed.deinit ok\n", .{});
    }

    var model = try ctx.loadModelPathAbsolute(path, .{});
    defer model.deinit();

    var total_bytes: usize = 0;
    var weights_bytes: usize = 0;
    var weights_q8_count: usize = 0;
    var weights_f16_count: usize = 0;
    var weights_f32_count: usize = 0;
    var weights_other_count: usize = 0;

    // Count only initializer-backed tensors for the "weights remain quantized" claim.
    // The store also contains many produced/intermediate tensors allocated during
    // compilation, and those are not expected to be quantized.
    for (model.package.values, 0..) |value, idx| {
        if (value.source != .initializer) continue;
        const wt = try model.initializerTensorByValue(@intCast(idx));
        const meta = try ctx.store.getConst(wt.tensorId());
        weights_bytes += meta.data.len;
        switch (meta.dtype) {
            .q8_0 => weights_q8_count += 1,
            .f16 => weights_f16_count += 1,
            .f32 => weights_f32_count += 1,
            else => weights_other_count += 1,
        }
    }

    for (ctx.store.tensors.items) |t| {
        total_bytes += t.data.len;
    }

    std.debug.print(
        "[gemma-verify] weights: q8_0={d} f16={d} f32={d} other={d} ({d:.3} GB) | total={d:.3} GB (budget {d:.3} GB)\n",
        .{
            weights_q8_count,
            weights_f16_count,
            weights_f32_count,
            weights_other_count,
            @as(f64, @floatFromInt(weights_bytes)) / 1e9,
            @as(f64, @floatFromInt(total_bytes)) / 1e9,
            max_gb,
        },
    );

    try std.testing.expect(weights_other_count == 0);
    try std.testing.expect(@as(f64, @floatFromInt(total_bytes)) / 1e9 <= max_gb);
}

test "api: exportModel + loadModel roundtrip for If control-flow model" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var export_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer export_ctx.deinit();

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const cond_ref: api.TensorRef = try bld.name(try bld.input(.i32, &[_]usize{1}), "cond");
    const then_ref: api.TensorRef = try bld.name(try bld.input(.f32, &[_]usize{1}), "then_v");
    const else_ref: api.TensorRef = try bld.name(try bld.input(.f32, &[_]usize{1}), "else_v");

    const g = bld.innerGraph();
    try g.beginRegion();
    const then_region = try g.endRegion(&[_]u32{then_ref.value});
    try g.beginRegion();
    const else_region = try g.endRegion(&[_]u32{else_ref.value});
    const out_ref: api.TensorRef = .{ .value = try g.addIf(cond_ref.value, then_region, else_region) };

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const file = try createTestFile(tmp.dir, "if_model.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);
    try export_ctx.exportModel(file, &bld, &[_]api.NamedTensorRef{.{ .name = "out", .tensor = out_ref }}, .{});

    var load_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer load_ctx.deinit();

    var model = try load_ctx.loadModel(file, .{});
    defer model.deinit();

    const cond_t: api.Tensor = try load_ctx.vector([_]i32{1});
    const then_t: api.Tensor = try load_ctx.vector([_]f32{42.0});
    const else_t: api.Tensor = try load_ctx.vector([_]f32{7.0});
    try model.bindInput("cond", cond_t);
    try model.bindInput("then_v", then_t);
    try model.bindInput("else_v", else_t);
    try model.run();

    var vals: [1]f32 = undefined;
    {
        const out_t: api.Tensor = try model.outputTensor("out");
        try out_t.read(&vals);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 42.0), vals[0], 1e-6);

    const cond_zero: api.Tensor = try load_ctx.vector([_]i32{0});
    try model.bindInput("cond", cond_zero);
    try model.run();
    {
        const out_t: api.Tensor = try model.outputTensor("out");
        try out_t.read(&vals);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), vals[0], 1e-6);
}

test "api: exportModel + loadModel roundtrip for Loop control-flow model" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var export_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer export_ctx.deinit();

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const carried_ref: api.TensorRef = try bld.name(try bld.input(.f32, &[_]usize{1}), "carried");
    const inc_ref: api.TensorRef = try bld.name(try bld.input(.f32, &[_]usize{1}), "inc");

    const g = bld.innerGraph();
    try g.beginRegion();
    const next = try g.addElemwiseBinary(.add, carried_ref.value, inc_ref.value);
    const body_region = try g.endRegion(&[_]u32{next});
    const out_ref: api.TensorRef = .{ .value = try g.addLoop(carried_ref.value, body_region, 4) };

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const file = try createTestFile(tmp.dir, "loop_model.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);
    try export_ctx.exportModel(file, &bld, &[_]api.NamedTensorRef{.{ .name = "out", .tensor = out_ref }}, .{});

    var load_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer load_ctx.deinit();

    var model = try load_ctx.loadModel(file, .{});
    defer model.deinit();

    const carried_t: api.Tensor = try load_ctx.vector([_]f32{1.0});
    const inc_t: api.Tensor = try load_ctx.vector([_]f32{2.0});
    try model.bindInput("carried", carried_t);
    try model.bindInput("inc", inc_t);
    try model.run();

    var vals: [1]f32 = undefined;
    {
        const out_t: api.Tensor = try model.outputTensor("out");
        try out_t.read(&vals);
    }
    // 1.0 + 2.0 * 4 trips = 9.0 (matches graph-level loop test).
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), vals[0], 1e-6);
}
