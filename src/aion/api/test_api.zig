const std = @import("std");

const api = @import("api.zig");
const aion_file = @import("../storage/aion_file.zig");
const manager_mod = @import("../storage/manager.zig");
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

test "api: importAionFile + tensorByName + readonly param execution" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var src_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer src_ctx.deinit();

    const w_t: api.Tensor = try src_ctx.fromArray([2][3]f32{
        .{ 1.0, -2.0, 0.5 },
        .{ 3.0, 4.0, -1.5 },
    });

    const w_src = try src_ctx.storage().getConst(w_t.tensorId());

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try createTestFile(tmp.dir, "weights.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);

    const metadata = [_]aion_file.MetadataSource{aion_file.MetadataSource.string("arch", "import-test")};
    const tensors = [_]aion_file.TensorSource{.{ .name = "w", .tensor = w_src }};
    try aion_file.writeFile(file, metadata[0..], tensors[0..], .{});

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    try ctx.importAionFile(file, .{});

    const w_loaded: api.Tensor = try ctx.tensorByName("w");
    try std.testing.expectEqual(types.DType.f32, w_loaded.getDType());
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 3 }, w_loaded.getShape());

    const w_vals: []f32 = try w_loaded.readAlloc(allocator, f32);
    defer allocator.free(w_vals);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1.0, -2.0, 0.5, 3.0, 4.0, -1.5 }, w_vals);

    try std.testing.expectError(manager_mod.StorageError.InvalidArgument, w_loaded.writeF32(&[_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }));

    const x_t: api.Tensor = try ctx.fromArray([1][2]f32{.{ 2.0, -1.0 }});

    var bld = api.Builder.init(allocator);
    defer bld.deinit();

    const X: api.TensorRef = try bld.param(x_t);
    const W: api.TensorRef = try bld.param(w_loaded);
    const Y: api.TensorRef = try bld.matmul(X, W, 1.0, 0.0);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Y});
    defer model.deinit();

    const y_t: api.Tensor = try model.runOutputTensor(0);
    var y_vals: [3]f32 = undefined;
    try y_t.read(&y_vals);

    try std.testing.expectApproxEqAbs(@as(f32, -1.0), y_vals[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -8.0), y_vals[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), y_vals[2], 1e-6);
}

test "api: importAionPath and importAionPathAbsolute convenience apis" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var src_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer src_ctx.deinit();

    const w_t: api.Tensor = try src_ctx.fromArray([2][2]f32{
        .{ 1.0, 2.0 },
        .{ 3.0, 4.0 },
    });
    const w_src = try src_ctx.storage().getConst(w_t.tensorId());

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try createTestFile(tmp.dir, "path_import.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);

    const tensors = [_]aion_file.TensorSource{.{ .name = "w_path", .tensor = w_src }};
    try aion_file.writeFile(file, &.{}, tensors[0..], .{});

    const relative_path: []const u8 = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..], "path_import.aion" });
    defer allocator.free(relative_path);

    const absolute_path: [:0]u8 = try tmp.parent_dir.realPathFileAlloc(std.testing.io, &tmp.sub_path, allocator);
    defer allocator.free(absolute_path);
    const absolute_file_path: []const u8 = try std.fs.path.join(allocator, &.{ absolute_path, "path_import.aion" });
    defer allocator.free(absolute_file_path);

    var rel_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer rel_ctx.deinit();
    try rel_ctx.importAionPath(relative_path, .{});
    const rel_w = try rel_ctx.tensorByName("w_path");
    const rel_vals = try rel_w.readAlloc(allocator, f32);
    defer allocator.free(rel_vals);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1.0, 2.0, 3.0, 4.0 }, rel_vals);

    var abs_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer abs_ctx.deinit();
    try abs_ctx.importAionPathAbsolute(absolute_file_path, .{});
    const abs_w = try abs_ctx.tensorByName("w_path");
    const abs_vals = try abs_w.readAlloc(allocator, f32);
    defer allocator.free(abs_vals);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1.0, 2.0, 3.0, 4.0 }, abs_vals);
}

test "api: import options expose mapped state and promotion to owned ram" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var src_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer src_ctx.deinit();

    const w_t: api.Tensor = try src_ctx.fromArray([2][2]f32{
        .{ 1.0, 2.0 },
        .{ 3.0, 4.0 },
    });
    const w_src = try src_ctx.storage().getConst(w_t.tensorId());

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try createTestFile(tmp.dir, "import_options.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);

    const tensors = [_]aion_file.TensorSource{.{ .name = "w_opts", .tensor = w_src }};
    try aion_file.writeFile(file, &.{}, tensors[0..], .{});

    var mapped_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer mapped_ctx.deinit();
    try mapped_ctx.importAionFile(file, .{ .residency = .mapped });
    try std.testing.expect(try mapped_ctx.isMappedTensor("w_opts"));

    const mapped_tensor = try mapped_ctx.tensorByName("w_opts");
    try std.testing.expectError(manager_mod.StorageError.InvalidArgument, mapped_tensor.writeF32(&[_]f32{ 9.0, 8.0, 7.0, 6.0 }));

    try mapped_ctx.promoteTensorToOwnedRam("w_opts");
    try std.testing.expect(!(try mapped_ctx.isMappedTensor("w_opts")));
    try mapped_tensor.writeF32(&[_]f32{ 9.0, 8.0, 7.0, 6.0 });

    const mapped_vals = try mapped_tensor.readAlloc(allocator, f32);
    defer allocator.free(mapped_vals);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 9.0, 8.0, 7.0, 6.0 }, mapped_vals);

    var copied_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer copied_ctx.deinit();
    try copied_ctx.importAionFile(file, .{ .residency = .copied });
    try std.testing.expect(!(try copied_ctx.isMappedTensor("w_opts")));

    const copied_tensor = try copied_ctx.tensorByName("w_opts");
    try std.testing.expectError(manager_mod.StorageError.InvalidArgument, copied_tensor.writeF32(&[_]f32{ 5.0, 6.0, 7.0, 8.0 }));

    try copied_ctx.promoteTensorToOwnedRam("w_opts");
    try copied_tensor.writeF32(&[_]f32{ 5.0, 6.0, 7.0, 8.0 });

    const copied_vals = try copied_tensor.readAlloc(allocator, f32);
    defer allocator.free(copied_vals);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 5.0, 6.0, 7.0, 8.0 }, copied_vals);
}
