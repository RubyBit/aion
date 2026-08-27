// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const api = @import("api.zig");
const manager_mod = @import("../storage/manager.zig");
const package_file = @import("../storage/aion_file.zig");
const quantize = @import("../storage/quantize.zig");
const types = @import("../backend/types.zig");
const fast = @import("../backend/cpu/kernels/fast_math.zig");

const nn = api.nn;

fn createTestFile(dir: std.Io.Dir, sub_path: []const u8, flags: std.Io.Dir.CreateFileOptions) !std.Io.File {
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

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const A = try bld.param(a_t);
    const B = try bld.param(b_t);
    const Bias = try bld.param(bias_t);

    const C = try bld.matmul(A, B, 1.0, 0.0);
    const D = try bld.add(C, Bias);
    const E = try bld.relu(D);
    const F = try bld.copy(E);
    const G = try bld.mul(F, F);
    const Out = try bld.reduce(.mean, G);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Out}, .{});
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
    var bld = api.Builder.init(&ctx);
    defer bld.deinit();
    const X = try bld.param(in_t);
    const Y = try bld.copy(X);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Y}, .{});
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

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const X: api.TensorRef = try bld.param(x_t);
    const Y: api.TensorRef = try bld.reduceAxis(.mean, X, -1);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Y}, .{});
    defer model.deinit();

    const out_t: api.Tensor = try model.runOutputTensor(0);
    try std.testing.expectEqualSlices(usize, &[_]usize{2}, out_t.getShape());

    var out_vals: [2]f32 = undefined;
    try out_t.read(&out_vals);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out_vals[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), out_vals[1], 1e-6);
}

test "api: elemwise broadcasting accepts a size-1 scalar operand" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    const x_t: api.Tensor = try ctx.fromArray([2][3]f32{
        .{ 1.0, 2.0, 3.0 },
        .{ 4.0, 5.0, 6.0 },
    });
    // A single-element vector, not a [3] one: it broadcasts over every column.
    const k_t: api.Tensor = try ctx.fromF32(&[_]usize{1}, &[_]f32{0.5});

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const X: api.TensorRef = try bld.param(x_t);
    const K: api.TensorRef = try bld.param(k_t);
    const Scaled: api.TensorRef = try bld.mul(X, K);
    const Shifted: api.TensorRef = try bld.add(Scaled, K);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Shifted}, .{});
    defer model.deinit();

    const out_t: api.Tensor = try model.runOutputTensor(0);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 3 }, out_t.getShape());

    var out_vals: [6]f32 = undefined;
    try out_t.read(&out_vals);
    const want: [6]f32 = .{ 1.0, 1.5, 2.0, 2.5, 3.0, 3.5 }; // x*0.5 + 0.5
    for (want, 0..) |w, i| try std.testing.expectApproxEqAbs(w, out_vals[i], 1e-6);
}

test "api: scalar broadcast spans multiple column tiles" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    // Last dim 200 > base_square_2d (64), so out has 4 column tiles while the
    // scalar has exactly one — the scalar must pair with every tile.
    const rows: usize = 3;
    const cols: usize = 200;
    const x_vals: []f32 = try allocator.alloc(f32, rows * cols);
    defer allocator.free(x_vals);
    for (x_vals, 0..) |*v, i| v.* = @floatFromInt(i % 17);

    const x_t: api.Tensor = try ctx.fromF32(&[_]usize{ rows, cols }, x_vals);
    const k_t: api.Tensor = try ctx.fromF32(&[_]usize{1}, &[_]f32{0.25});

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const X: api.TensorRef = try bld.param(x_t);
    const K: api.TensorRef = try bld.param(k_t);
    const Y: api.TensorRef = try bld.mul(X, K);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Y}, .{});
    defer model.deinit();

    const out_t: api.Tensor = try model.runOutputTensor(0);
    const out_vals: []f32 = try allocator.alloc(f32, rows * cols);
    defer allocator.free(out_vals);
    try out_t.read(out_vals);

    for (x_vals, 0..) |xv, i| {
        try std.testing.expectApproxEqAbs(xv * 0.25, out_vals[i], 1e-6);
    }
}

test "api: elemwise broadcasting rejects a mismatched vector length" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    const x_t: api.Tensor = try ctx.fromArray([2][3]f32{
        .{ 1.0, 2.0, 3.0 },
        .{ 4.0, 5.0, 6.0 },
    });
    // 2 is neither the last dim (3) nor 1, so it stays an error.
    const bad_t: api.Tensor = try ctx.fromF32(&[_]usize{2}, &[_]f32{ 1.0, 2.0 });

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const X: api.TensorRef = try bld.param(x_t);
    const B: api.TensorRef = try bld.param(bad_t);
    try std.testing.expectError(error.ShapeMismatch, bld.mul(X, B));
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

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const X: api.TensorRef = try bld.param(x_t);

    const S: api.TensorRef = try bld.squeeze(X, null);
    const U0: api.TensorRef = try bld.unsqueeze(S, 0);
    const U1: api.TensorRef = try bld.unsqueeze(U0, 2);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{ U1, S }, .{});
    defer model.deinit();

    const out_u: api.Tensor = try model.runOutputTensor(0);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 2, 1, 3 }, out_u.getShape());

    var out_u_vals: [1 * 2 * 1 * 3]f32 = undefined;
    try out_u.read(&out_u_vals);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, out_u_vals[0..]);

    const out_s: api.Tensor = try model.outputTensorAt(1);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 3 }, out_s.getShape());

    // Invalid squeeze axis (dimension != 1): with eager per-op inference this now
    // fails at the offending op call, not at compile.
    var bld_bad = api.Builder.init(&ctx);
    defer bld_bad.deinit();
    const X_bad: api.TensorRef = try bld_bad.param(x_t);
    const S_bad: api.TensorRef = try bld_bad.squeeze(X_bad, null);
    try std.testing.expectError(error.ShapeMismatch, bld_bad.squeeze(S_bad, 1));
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

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const X: api.TensorRef = try bld.param(x_t);

    const conv: nn.Conv1D = try nn.Conv1D.bind(&bld, .{ .weight = w_t }, .{
        .stride = 1,
        .dilation = 1,
        .pad_left = 0,
        .pad_right = 0,
        .pad_mode = .zero,
        .groups = 1,
    });

    const Y: api.TensorRef = try conv.forward(&bld, X);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Y}, .{});
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

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const X: api.TensorRef = try bld.param(x_t);
    const S: api.TensorRef = try bld.slice(X, &[_]usize{ 1, 1, 0 }, &[_]usize{ 1, 2, 3 });

    var model = try ctx.compile(&bld, &[_]api.TensorRef{S}, .{});
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

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const A: api.TensorRef = try bld.param(a_t);
    const B: api.TensorRef = try bld.param(b_t);

    const C: api.TensorRef = try bld.concat(&[_]api.TensorRef{ A, B }, 1);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{C}, .{});
    defer model.deinit();

    const out_c: api.Tensor = try model.runOutputTensor(0);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 3 }, out_c.getShape());
    var c_vals: [6]f32 = undefined;
    try out_c.read(&c_vals);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1.0, 2.0, 10.0, 3.0, 4.0, 20.0 }, c_vals[0..]);

    var bld2 = api.Builder.init(&ctx);
    defer bld2.deinit();
    const A2: api.TensorRef = try bld2.param(a_t);
    const B2: api.TensorRef = try bld2.param(a_t);
    const S: api.TensorRef = try bld2.stack(&[_]api.TensorRef{ A2, B2 }, 0);

    var model2 = try ctx.compile(&bld2, &[_]api.TensorRef{S}, .{});
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

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const A0: api.TensorRef = try bld.param(a_t);
    const A: api.TensorRef = try bld.name(A0, "A");
    const B: api.TensorRef = try bld.param(b_t);

    const C: api.TensorRef = try bld.matmul(A, B, 1.0, 0.0);
    const Y: api.TensorRef = try bld.relu(C);
    const Mean: api.TensorRef = try bld.reduce(.mean, C);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{ Mean, Y }, .{});
    defer model.deinit();

    try std.testing.expectEqual(@as(usize, 2), model.outputCount());

    const out0_t: api.Tensor = try model.runOutputTensor(0);
    const out0: f32 = try out0_t.readScalar(f32);
    try std.testing.expect(std.math.isFinite(out0));

    const out1_t: api.Tensor = try model.outputTensorAt(1);
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
    try std.testing.expectError(error.InvalidArgument, ctx.fromPackedQuant(.f32, &[_]usize{ 1, 1 }, 0, dummy[0..]));
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

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const X: api.TensorRef = try bld.param(x_t);

    const pad_opts: nn.Conv2D.Options = .{ .pad_top = 1, .pad_bottom = 1, .pad_left = 1, .pad_right = 1 };
    const block: nn.ResBlock2D = try nn.ResBlock2D.bind(&bld, .{
        .conv1 = .{ .weight = w1_t, .bias = b1_t },
        .conv2 = .{ .weight = w2_t, .bias = b2_t },
    }, .{ .conv = pad_opts });
    const fc: nn.Linear = try nn.Linear.bind(&bld, .{ .weight = fc_w_t, .bias = fc_b_t }, .{});

    const Y: api.TensorRef = try block.forward(&bld, X);
    const Mean: api.TensorRef = try bld.reduce(.mean, Y);
    const Mean2: api.TensorRef = try bld.reshape(Mean, &[_]usize{ 1, 1 });
    const Logits: api.TensorRef = try fc.forward(&bld, Mean2);
    const Probs: api.TensorRef = try nn.softmaxLastDim(&bld, Logits);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Probs}, .{});
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
    var tmp: [gate_dim - 1]f32 = @splat(0.0);
    try bad_bias.write(tmp[0..]);

    var bld_bad = api.Builder.init(&ctx);
    defer bld_bad.deinit();
    try std.testing.expectError(error.InvalidArgument, nn.LSTMCell.bind(&bld_bad, .{
        .weight_ih = w_ih_t,
        .weight_hh = w_hh_t,
        .bias_ih = b_ih_t,
        .bias_hh = bad_bias,
    }, .{}));

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const X: api.TensorRef = try bld.param(x_t);
    const H0: api.TensorRef = try bld.param(h0_t);
    const C0: api.TensorRef = try bld.param(c0_t);

    const cell: nn.LSTMCell = try nn.LSTMCell.bind(&bld, .{
        .weight_ih = w_ih_t,
        .weight_hh = w_hh_t,
        .bias_ih = b_ih_t,
        .bias_hh = b_hh_t,
    }, .{});
    const st: nn.LSTMState = try cell.forward(&bld, X, H0, C0);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{ st.h, st.c }, .{});
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

fn refOneSidedDft(x: []const f32, out_re: []f32, out_im: []f32) void {
    const n: usize = x.len;
    const bins: usize = n / 2 + 1;
    const n_f: f64 = @floatFromInt(n);
    for (0..bins) |k| {
        var sr: f64 = 0;
        var si: f64 = 0;
        for (0..n) |m| {
            const ang: f64 = -2.0 * std.math.pi * @as(f64, @floatFromInt(k)) *
                @as(f64, @floatFromInt(m)) / n_f;
            sr += @as(f64, x[m]) * @cos(ang);
            si += @as(f64, x[m]) * @sin(ang);
        }
        out_re[k] = @floatCast(sr);
        out_im[k] = @floatCast(si);
    }
}

test "api: rfft matches one-sided DFT" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const frames: usize = 2;
    const n_fft: usize = 8;
    const bins: usize = n_fft / 2 + 1;

    var x_vals: [frames * n_fft]f32 = undefined;
    for (0..x_vals.len) |i| {
        x_vals[i] = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3))) * 0.3;
    }

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    const x_t: api.Tensor = try ctx.fromF32(&[_]usize{ frames, n_fft }, x_vals[0..]);

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const X: api.TensorRef = try bld.param(x_t);
    const Out: api.TensorRef = try bld.rfft(X);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Out}, .{});
    defer model.deinit();

    const out_t: api.Tensor = try model.runOutputTensor(0);
    try std.testing.expectEqualSlices(usize, &[_]usize{ frames, 2 * bins }, out_t.getShape());

    var out_vals: [frames * 2 * bins]f32 = undefined;
    try out_t.read(&out_vals);

    var ref_re: [bins]f32 = undefined;
    var ref_im: [bins]f32 = undefined;
    for (0..frames) |f| {
        refOneSidedDft(x_vals[f * n_fft ..][0..n_fft], ref_re[0..], ref_im[0..]);
        for (0..bins) |k| {
            try std.testing.expectApproxEqAbs(ref_re[k], out_vals[f * 2 * bins + k], 1e-3);
            try std.testing.expectApproxEqAbs(ref_im[k], out_vals[f * 2 * bins + k + bins], 1e-3);
        }
    }
}

fn refReflectIndex(i: i64, n: usize) usize {
    if (n <= 1) return 0;
    const period: i64 = 2 * (@as(i64, @intCast(n)) - 1);
    var m: i64 = @mod(i, period);
    const ni: i64 = @intCast(n);
    if (m >= ni) m = period - m;
    return @intCast(m);
}

test "api: stft with center reflect-padding matches reference" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const n_fft: usize = 8;
    const hop: usize = 4;
    const samples: usize = 20;
    const bins: usize = n_fft / 2 + 1;
    const num_frames: usize = 1 + samples / hop; // center=true

    var sig_vals: [samples]f32 = undefined;
    for (0..samples) |i| sig_vals[i] = @sin(@as(f32, @floatFromInt(i)) * 0.37) + 0.1 * @as(f32, @floatFromInt(i % 3));
    var win_vals: [n_fft]f32 = undefined;
    for (0..n_fft) |i| win_vals[i] = 0.5 - 0.5 * @cos(2.0 * std.math.pi * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n_fft)));

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    const sig_t: api.Tensor = try ctx.fromF32(&[_]usize{ 1, samples }, sig_vals[0..]);
    const win_t: api.Tensor = try ctx.fromF32(&[_]usize{n_fft}, win_vals[0..]);

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const Sig: api.TensorRef = try bld.param(sig_t);
    const Win: api.TensorRef = try bld.param(win_t);
    const Out: api.TensorRef = try bld.stft(Sig, Win, n_fft, hop, true);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Out}, .{});
    defer model.deinit();

    const out_t: api.Tensor = try model.runOutputTensor(0);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, num_frames, 2 * bins }, out_t.getShape());

    var out_vals: [num_frames * 2 * bins]f32 = undefined;
    try out_t.read(&out_vals);

    const pad: i64 = @intCast(n_fft / 2);
    var frame: [n_fft]f32 = undefined;
    var ref_re: [bins]f32 = undefined;
    var ref_im: [bins]f32 = undefined;
    for (0..num_frames) |t| {
        const origin: i64 = @as(i64, @intCast(t * hop)) - pad;
        for (0..n_fft) |j| {
            const idx: i64 = origin + @as(i64, @intCast(j));
            frame[j] = sig_vals[refReflectIndex(idx, samples)] * win_vals[j];
        }
        refOneSidedDft(frame[0..], ref_re[0..], ref_im[0..]);
        for (0..bins) |k| {
            try std.testing.expectApproxEqAbs(ref_re[k], out_vals[t * 2 * bins + k], 1e-3);
            try std.testing.expectApproxEqAbs(ref_im[k], out_vals[t * 2 * bins + k + bins], 1e-3);
        }
    }
}

test "api: stft frames + windows + rfft (no center)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const n_fft: usize = 8;
    const hop: usize = 4;
    const samples: usize = 16;
    const bins: usize = n_fft / 2 + 1;
    const num_frames: usize = 1 + (samples - n_fft) / hop; // = 3

    var sig_vals: [samples]f32 = undefined;
    for (0..samples) |i| sig_vals[i] = @sin(@as(f32, @floatFromInt(i)) * 0.5);
    var win_vals: [n_fft]f32 = undefined;
    for (0..n_fft) |i| win_vals[i] = 0.5 - 0.5 * @cos(2.0 * std.math.pi * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n_fft)));

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    const sig_t: api.Tensor = try ctx.fromF32(&[_]usize{ 1, samples }, sig_vals[0..]);
    const win_t: api.Tensor = try ctx.fromF32(&[_]usize{n_fft}, win_vals[0..]);

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const Sig: api.TensorRef = try bld.param(sig_t);
    const Win: api.TensorRef = try bld.param(win_t);
    const Out: api.TensorRef = try bld.stft(Sig, Win, n_fft, hop, false);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Out}, .{});
    defer model.deinit();

    const out_t: api.Tensor = try model.runOutputTensor(0);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, num_frames, 2 * bins }, out_t.getShape());

    var out_vals: [num_frames * 2 * bins]f32 = undefined;
    try out_t.read(&out_vals);

    var frame: [n_fft]f32 = undefined;
    var ref_re: [bins]f32 = undefined;
    var ref_im: [bins]f32 = undefined;
    for (0..num_frames) |t| {
        for (0..n_fft) |j| frame[j] = sig_vals[t * hop + j] * win_vals[j];
        refOneSidedDft(frame[0..], ref_re[0..], ref_im[0..]);
        for (0..bins) |k| {
            try std.testing.expectApproxEqAbs(ref_re[k], out_vals[t * 2 * bins + k], 1e-3);
            try std.testing.expectApproxEqAbs(ref_im[k], out_vals[t * 2 * bins + k + bins], 1e-3);
        }
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

    var bld = api.Builder.init(&export_ctx);
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

test "api: unbound recurrent state auto-initializes, carries across runs, and resets" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var export_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer export_ctx.deinit();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try createTestFile(tmp.dir, "recurrent.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);

    // state_out = state_in + x ; state_out is aliased back to state_in so the
    // accumulator persists across runs.
    var bld = api.Builder.init(&export_ctx);
    defer bld.deinit();

    const X: api.TensorRef = try bld.name(try bld.input(.f32, &[_]usize{ 1, 2 }), "x");
    const StateIn: api.TensorRef = try bld.name(try bld.input(.f32, &[_]usize{ 1, 2 }), "state_in");
    const StateOut: api.TensorRef = try bld.add(StateIn, X);

    try export_ctx.exportModel(file, &bld, &[_]api.NamedTensorRef{
        .{ .name = "state_out", .tensor = StateOut },
    }, .{
        .output_aliases = &[_]api.OutputAlias{
            .{ .input = StateIn, .output = StateOut },
        },
    });

    var load_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer load_ctx.deinit();

    var model = try load_ctx.loadModel(file, .{});
    defer model.deinit();

    // Bind only the real input; never bind "state_in".
    const x_t: api.Tensor = try load_ctx.fromArray([1][2]f32{.{ 1.0, 2.0 }});
    try model.bindInput("x", x_t);

    const expect = struct {
        fn run(m: *api.LoadedModel, e0: f32, e1: f32) !void {
            try m.run();
            const out: api.Tensor = try m.outputTensor("state_out");
            var vals: [2]f32 = undefined;
            try out.read(&vals);
            try std.testing.expectApproxEqAbs(e0, vals[0], 1e-6);
            try std.testing.expectApproxEqAbs(e1, vals[1], 1e-6);
        }
    };

    // Run 1: state auto-zeroed -> 0 + x.
    try expect.run(&model, 1.0, 2.0);
    // Run 2/3: state carried via the io-alias -> accumulates.
    try expect.run(&model, 2.0, 4.0);
    try expect.run(&model, 3.0, 6.0);

    // Reset clears the carried state; accumulation starts over.
    try model.resetState();
    try expect.run(&model, 1.0, 2.0);
    try expect.run(&model, 2.0, 4.0);
}

test "api: recurrent state carries across cache entries with different input shapes" {
    // Mirrors the prefill(seq=S) -> decode(seq=1) shape change of an LLM: a model
    // run at two different sequence lengths compiles two distinct cache entries.
    // The io-aliased accumulator must carry across them (shared model-level state).
    const allocator: std.mem.Allocator = std.testing.allocator;

    var export_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer export_ctx.deinit();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try createTestFile(tmp.dir, "recurrent_multishape.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);

    // state_out = sum_over_seq(x) + state_in ; state_out aliased back to state_in.
    // x has a symbolic seq dim, so different seq lengths -> different cache entries.
    var bld = api.Builder.init(&export_ctx);
    defer bld.deinit();

    const X: api.TensorRef = try bld.name(try bld.input(.f32, &[_]usize{ 1, 1, 2 }), "x");
    const StateIn: api.TensorRef = try bld.name(try bld.input(.f32, &[_]usize{ 1, 2 }), "state_in");
    const Summed: api.TensorRef = try bld.reduceAxis(.sum, X, 1); // [1, 2]
    const StateOut: api.TensorRef = try bld.add(Summed, StateIn);

    try bld.symbolicDim(X, 1, "S");
    try export_ctx.exportModel(file, &bld, &[_]api.NamedTensorRef{
        .{ .name = "state_out", .tensor = StateOut },
    }, .{
        .output_aliases = &[_]api.OutputAlias{
            .{ .input = StateIn, .output = StateOut },
        },
    });

    var load_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer load_ctx.deinit();

    var model = try load_ctx.loadModel(file, .{});
    defer model.deinit();

    const readState = struct {
        fn go(m: *api.LoadedModel) ![2]f32 {
            try m.run();
            const out: api.Tensor = try m.outputTensor("state_out");
            var vals: [2]f32 = undefined;
            try out.read(&vals);
            return vals;
        }
    };

    // Run 1: "prefill" seq=2, x = [[[1,1],[1,1]]] -> sum=[2,2], state 0 -> [2,2].
    const x2: api.Tensor = try load_ctx.fromArray([1][2][2]f32{.{ .{ 1.0, 1.0 }, .{ 1.0, 1.0 } }});
    try model.bindInput("x", x2);
    {
        const v = try readState.go(&model);
        try std.testing.expectApproxEqAbs(@as(f32, 2.0), v[0], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 2.0), v[1], 1e-6);
    }

    // Run 2: "decode" seq=1 (a different cache entry), x=[[[1,1]]] -> sum=[1,1].
    // If state carries across the shape change, this accumulates onto [2,2] -> [3,3].
    const x1: api.Tensor = try load_ctx.fromArray([1][1][2]f32{.{.{ 1.0, 1.0 }}});
    try model.bindInput("x", x1);
    {
        const v = try readState.go(&model);
        try std.testing.expectApproxEqAbs(@as(f32, 3.0), v[0], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 3.0), v[1], 1e-6);
    }
    // Run 3: same seq=1 entry -> [4,4].
    {
        const v = try readState.go(&model);
        try std.testing.expectApproxEqAbs(@as(f32, 4.0), v[0], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 4.0), v[1], 1e-6);
    }

    // Reset clears the shared state; accumulation restarts.
    try model.resetState();
    {
        const v = try readState.go(&model);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), v[0], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), v[1], 1e-6);
    }
}

test "api: growable state input grows on demand up to its bound" {
    // Grow-on-demand recurrent state: the cache is allocated small and the runtime
    // grows its slot as appends cross capacity, up to a caller-set ceiling. No
    // pre-allocation of the maximum.
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    // cache [1,1,T,1] starts at T=2; sequenceAppend writes `new` at cache[end].
    const Cache = try bld.name(try bld.input(.f32, &[_]usize{ 1, 2, 1, 1 }), "cache");
    const New = try bld.name(try bld.input(.f32, &[_]usize{ 1, 1, 1, 1 }), "new");
    const End = try bld.name(try bld.input(.i32, &[_]usize{1}), "end");
    const Out = try bld.name(try bld.sequenceAppend(Cache, New, End), "cache_out");

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Out}, .{
        .output_aliases = &[_]api.OutputAlias{.{ .input = Cache, .output = Out }},
    });
    defer model.deinit();

    // Initial capacity 2, double on demand, hard ceiling 8.
    try model.setStateInputPolicy("cache", .{ .growable = .{
        .initial_capacity_tokens = 2,
        .growth_numerator = 2,
        .growth_denominator = 1,
        .max_capacity_tokens = 8,
    } });

    const cache0 = try ctx.fromF32(&[_]usize{ 1, 2, 1, 1 }, &@as([2]f32, @splat(0)));
    try model.bindInput("cache", cache0);

    // Append 1..5 at positions 0..4; capacity grows 2 -> 4 (at pos 2) -> 8 (at pos 4).
    var pos: i32 = 0;
    while (pos < 5) : (pos += 1) {
        const new_t = try ctx.fromArray([1][1][1][1]f32{.{.{.{@as(f32, @floatFromInt(pos + 1))}}}});
        const end_t = try ctx.fromArray([1]i32{pos});
        try model.bindInput("new", new_t);
        try model.bindInput("end", end_t);
        try model.run();
    }

    // The slot grew to capacity 8; the first five entries hold the appended values.
    const out = try model.outputTensorAt(0);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 8, 1, 1 }, out.getShape());
    var vals: [8]f32 = undefined;
    try out.read(&vals);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 3, 4, 5, 0, 0, 0 }, &vals);

    // A step past the ceiling fails rather than growing without bound.
    const new_t = try ctx.fromArray([1][1][1][1]f32{.{.{.{99.0}}}});
    const end_over = try ctx.fromArray([1]i32{8});
    try model.bindInput("new", new_t);
    try model.bindInput("end", end_over);
    try std.testing.expectError(error.InvalidArgument, model.run());
}

test "api: input roles auto-drive cache indices and positions (compile path)" {
    // Decode-style graph with role-declared control inputs: the caller binds ONLY
    // `tokens` per step; the runtime feeds cache_write_index / cache_visible_end /
    // positions from its tracked position and advances it per run.
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const table_t: api.Tensor = try ctx.fromArray([4][1]f32{ .{10}, .{20}, .{30}, .{40} });
    const Table = try bld.param(table_t);

    const Tokens = try bld.name(try bld.input(.i32, &[_]usize{ 1, 1 }), "tokens");
    const WriteIdx = try bld.name(try bld.input(.i32, &[_]usize{1}), "cache_write_index");
    const VisibleEnd = try bld.name(try bld.input(.i32, &[_]usize{1}), "cache_visible_end");
    const Positions = try bld.name(try bld.input(.i32, &[_]usize{ 1, 1 }), "positions");
    const Cache = try bld.name(try bld.input(.f32, &[_]usize{ 1, 4, 1, 1 }), "cache");

    const Emb = try bld.gather(Table, Tokens, 0, 0); // [1,1,1]
    const New4 = try bld.reshape(Emb, &[_]usize{ 1, 1, 1, 1 });
    const CacheOut = try bld.name(try bld.sequenceAppend(Cache, New4, WriteIdx), "next_cache");
    // `cache_visible_end` is consumed by no node in this toy graph; it is still a
    // public input the runtime must feed.
    const PosProbe = try bld.name(try bld.gather(Table, Positions, 0, 0), "pos_probe"); // table[position]

    var model = try ctx.compile(&bld, &[_]api.TensorRef{ CacheOut, PosProbe }, .{
        .output_aliases = &[_]api.OutputAlias{.{ .input = Cache, .output = CacheOut }},
        .input_roles = &[_]api.InputRoleDecl{
            .{ .input = Tokens, .kind = .tokens, .axis = 1 },
            .{ .input = WriteIdx, .kind = .cache_write_index },
            .{ .input = VisibleEnd, .kind = .cache_visible_end },
            .{ .input = Positions, .kind = .positions, .axis = 1 },
            .{ .input = Cache, .kind = .sequence_cache, .axis = 1 },
        },
    });
    defer model.deinit();

    const step = struct {
        fn run(m: *api.Model, c: *api.Context, token: i32) ![2]f32 {
            const tok_t = try c.fromArray([1][1]i32{.{token}});
            try m.bindInput("tokens", tok_t);
            try m.run();
            var out: [2]f32 = undefined;
            var cache_vals: [4]f32 = undefined;
            const cache_out = try m.outputTensor("next_cache");
            try cache_out.read(&cache_vals);
            const probe = try m.outputTensor("pos_probe");
            var probe_val: [1]f32 = undefined;
            try probe.read(&probe_val);
            out[0] = cache_vals[3]; // untouched tail stays zero until position 3
            out[1] = probe_val[0];
            return out;
        }
    };

    try std.testing.expectEqual(@as(u64, 0), model.currentPosition());

    // Step 1: position 0 -> cache[0]=table[1]=20, pos_probe=table[0]=10.
    var r = try step.run(&model, &ctx, 1);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), r[1], 1e-6);
    try std.testing.expectEqual(@as(u64, 1), model.currentPosition());

    // Step 2: position 1 -> cache[1]=table[3]=40, pos_probe=table[1]=20.
    r = try step.run(&model, &ctx, 3);
    try std.testing.expectApproxEqAbs(@as(f32, 20.0), r[1], 1e-6);
    try std.testing.expectEqual(@as(u64, 2), model.currentPosition());

    {
        var cache_vals: [4]f32 = undefined;
        const cache_out = try model.outputTensor("next_cache");
        try cache_out.read(&cache_vals);
        try std.testing.expectEqualSlices(f32, &[_]f32{ 20, 40, 0, 0 }, &cache_vals);
    }

    // resetState rewinds the position and zeroes the carried cache.
    try model.resetState();
    try std.testing.expectEqual(@as(u64, 0), model.currentPosition());
    r = try step.run(&model, &ctx, 2);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), r[1], 1e-6); // position back to 0
    {
        var cache_vals: [4]f32 = undefined;
        const cache_out = try model.outputTensor("next_cache");
        try cache_out.read(&cache_vals);
        try std.testing.expectEqualSlices(f32, &[_]f32{ 30, 0, 0, 0 }, &cache_vals);
    }

    // A manual bind on a control input overrides its auto value (escape hatch);
    // the other control inputs stay auto-fed.
    const manual_idx = try ctx.fromArray([1]i32{3});
    try model.bindInput("cache_write_index", manual_idx);
    _ = try step.run(&model, &ctx, 0); // appends at index 3 despite position 1
    {
        var cache_vals: [4]f32 = undefined;
        const cache_out = try model.outputTensor("next_cache");
        try cache_out.read(&cache_vals);
        try std.testing.expectEqualSlices(f32, &[_]f32{ 30, 0, 0, 10 }, &cache_vals);
    }
    // Position still advanced (documented); setPosition re-syncs.
    try std.testing.expectEqual(@as(u64, 2), model.currentPosition());
    model.setPosition(0);
    try std.testing.expectEqual(@as(u64, 0), model.currentPosition());
}

test "api: role-declared symbolic cache auto-sizes from LoadModelOptions.cache" {
    // Export a package whose cache capacity axis is a free symbol `G` with a
    // sequence_cache role, then load WITHOUT binding the cache: the runtime sizes
    // it from `cache.capacity_tokens` (fixed) or grows it on demand (growable).
    const allocator: std.mem.Allocator = std.testing.allocator;

    var export_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer export_ctx.deinit();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const file = try createTestFile(tmp.dir, "roles_cache.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);

    {
        const table_t: api.Tensor = try export_ctx.fromArray([4][1]f32{ .{10}, .{20}, .{30}, .{40} });

        var bld = api.Builder.init(&export_ctx);
        defer bld.deinit();

        const Table = try bld.param(table_t);
        const Tokens = try bld.name(try bld.input(.i32, &[_]usize{ 1, 1 }), "tokens");
        const WriteIdx = try bld.name(try bld.input(.i32, &[_]usize{1}), "cache_write_index");
        const Cache = try bld.name(try bld.input(.f32, &[_]usize{ 1, 4, 1, 1 }), "cache");

        const Emb = try bld.gather(Table, Tokens, 0, 0); // [1,1,1]
        const New4 = try bld.reshape(Emb, &[_]usize{ 1, 1, 1, 1 });
        const CacheOut = try bld.name(try bld.sequenceAppend(Cache, New4, WriteIdx), "next_cache");

        try bld.symbolicDim(Cache, 1, "G");
        try export_ctx.exportModel(file, &bld, &[_]api.NamedTensorRef{
            .{ .name = "next_cache", .tensor = CacheOut },
        }, .{
            .output_aliases = &[_]api.OutputAlias{.{ .input = Cache, .output = CacheOut }},
            .input_roles = &[_]api.InputRoleDecl{
                .{ .input = Tokens, .kind = .tokens, .axis = 1 },
                .{ .input = WriteIdx, .kind = .cache_write_index },
                .{ .input = Cache, .kind = .sequence_cache, .axis = 1, .capacity_symbol = "G", .allow_growable = true },
            },
        });
    }

    const drive = struct {
        fn steps(m: *api.Model, c: *api.Context, tokens: []const i32) !void {
            for (tokens) |tok| {
                const tok_t = try c.fromArray([1][1]i32{.{tok}});
                try m.bindInput("tokens", tok_t);
                try m.run();
            }
        }
    };

    // (a) Fixed pre-allocation at capacity_tokens.
    {
        var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
        defer ctx.deinit();
        var model = try ctx.loadModel(file, .{ .cache = .{ .capacity_tokens = 6 } });
        defer model.deinit();

        try drive.steps(&model, &ctx, &[_]i32{ 1, 2, 3 });
        try std.testing.expectEqual(@as(u64, 3), model.currentPosition());

        const out = try model.outputTensor("next_cache");
        try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 6, 1, 1 }, out.getShape());
        var vals: [6]f32 = undefined;
        try out.read(&vals);
        try std.testing.expectEqualSlices(f32, &[_]f32{ 20, 30, 40, 0, 0, 0 }, &vals);
    }

    // (b) Growable: starts at 2 tokens, grows on demand, ceiling 6.
    {
        var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
        defer ctx.deinit();
        var model = try ctx.loadModel(file, .{ .cache = .{
            .capacity_tokens = 6,
            .growable = .{ .initial_capacity_tokens = 2, .growth_numerator = 2, .growth_denominator = 1 },
        } });
        defer model.deinit();

        try drive.steps(&model, &ctx, &[_]i32{ 1, 2, 3, 1 });

        // Appends at positions 0..3 grow the slot 2 -> 4 (doubling), under the ceiling 6.
        const out = try model.outputTensor("next_cache");
        try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 4, 1, 1 }, out.getShape());
        var vals: [4]f32 = undefined;
        try out.read(&vals);
        try std.testing.expectEqualSlices(f32, &[_]f32{ 20, 30, 40, 20 }, &vals);
    }

    // (c) No cache options: the free symbol stays undetermined, matching the old
    // behavior — run() refuses, and binding the cache explicitly still works.
    {
        var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
        defer ctx.deinit();
        var model = try ctx.loadModel(file, .{});
        defer model.deinit();

        const tok_t = try ctx.fromArray([1][1]i32{.{1}});
        try model.bindInput("tokens", tok_t);
        try std.testing.expectError(error.InvalidArgument, model.run());

        const cache_t = try ctx.fromArray([1][3][1][1]f32{.{ .{.{0}}, .{.{0}}, .{.{0}} }});
        try model.bindInput("cache", cache_t);
        try model.run();
        const out = try model.outputTensor("next_cache");
        var vals: [3]f32 = undefined;
        try out.read(&vals);
        try std.testing.expectEqualSlices(f32, &[_]f32{ 20, 0, 0 }, &vals);
    }
}

test "api: compile io-alias gives auto-init + carry + reset (no export/load)" {
    // The unified Model from ctx.compile must support the same recurrent-state
    // ergonomics as a loaded model: unbound aliased state auto-zeros, carries across
    // runs, and resetState clears it — all in-process, no .aion round trip.
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const X: api.TensorRef = try bld.name(try bld.input(.f32, &[_]usize{ 1, 2 }), "x");
    const StateIn: api.TensorRef = try bld.name(try bld.input(.f32, &[_]usize{ 1, 2 }), "state_in");
    // Name the output so the io-alias can reference it; pass it as a bare TensorRef.
    const StateOut: api.TensorRef = try bld.name(try bld.add(StateIn, X), "state_out");

    var model = try ctx.compile(&bld, &[_]api.TensorRef{StateOut}, .{
        .output_aliases = &[_]api.OutputAlias{
            .{ .input = StateIn, .output = StateOut },
        },
    });
    defer model.deinit();

    const x_t: api.Tensor = try ctx.fromArray([1][2]f32{.{ 1.0, 2.0 }});
    try model.bindInput("x", x_t); // never bind state_in

    const expect = struct {
        fn run(m: *api.Model, e0: f32, e1: f32) !void {
            try m.run();
            const out: api.Tensor = try m.outputTensor("state_out");
            var vals: [2]f32 = undefined;
            try out.read(&vals);
            try std.testing.expectApproxEqAbs(e0, vals[0], 1e-6);
            try std.testing.expectApproxEqAbs(e1, vals[1], 1e-6);
        }
    };

    try expect.run(&model, 1.0, 2.0); // auto-zeroed state
    try expect.run(&model, 2.0, 4.0); // carried
    try expect.run(&model, 3.0, 6.0);
    try model.resetState();
    try expect.run(&model, 1.0, 2.0); // cleared
}

test "api: loadModel with auto_init_inputs=false errors on an unbound input" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var export_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer export_ctx.deinit();

    const w_t: api.Tensor = try export_ctx.fromArray([2][3]f32{
        .{ 1.0, -2.0, 0.5 },
        .{ 3.0, 4.0, -1.5 },
    });

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try createTestFile(tmp.dir, "strict.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);

    var bld = api.Builder.init(&export_ctx);
    defer bld.deinit();

    const X: api.TensorRef = try bld.name(try bld.input(.f32, &[_]usize{ 1, 2 }), "x");
    const W: api.TensorRef = try bld.param(w_t);
    const Y: api.TensorRef = try bld.matmul(X, W, 1.0, 0.0);
    try export_ctx.exportModel(file, &bld, &[_]api.NamedTensorRef{.{ .name = "y", .tensor = Y }}, .{});

    var load_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer load_ctx.deinit();

    var model = try load_ctx.loadModel(file, .{ .auto_init_inputs = false });
    defer model.deinit();

    // "x" is never bound; strict mode must refuse to run.
    try std.testing.expectError(error.InvalidArgument, model.run());
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

    var bld = api.Builder.init(&export_ctx);
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
    var ones_vals: [6]f32 = @splat(1.0);
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

/// Pack a `[K, N]` f32 matmul-B weight to q8_0 (blocks along K, per column).
fn packQ8WeightKN(allocator: std.mem.Allocator, vals: []const f32, k: usize, n: usize) ![]u8 {
    const kb = k / 32;
    const buf = try allocator.alloc(u8, kb * n * 34);
    for (0..kb) |b| {
        for (0..n) |j| {
            var absmax: f32 = 0;
            for (0..32) |t| absmax = @max(absmax, @abs(vals[(b * 32 + t) * n + j]));
            const scale: f32 = if (absmax == 0) 1 else absmax / 127.0;
            const inv: f32 = if (absmax == 0) 0 else 1.0 / scale;
            const sf16: f16 = @floatCast(scale);
            const off = (b * n + j) * 34;
            std.mem.writeInt(u16, buf[off .. off + 2][0..2], @bitCast(sf16), .little);
            for (0..32) |t| {
                var q: i32 = @intFromFloat(@round(vals[(b * 32 + t) * n + j] * inv));
                q = @max(@as(i32, -128), @min(@as(i32, 127), q));
                buf[off + 2 + t] = @bitCast(@as(i8, @intCast(q)));
            }
        }
    }
    return buf;
}

test "api: weight-swap writes through a fused projection (in-place handle refused)" {
    const allocator: std.mem.Allocator = std.testing.allocator;
    const K: usize = 64;
    const Nq: usize = 32;
    const Nk: usize = 32;
    const M: usize = 2;

    var export_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer export_ctx.deinit();

    // Two q8_0 projections off a shared input — the compiler fuses them at load.
    const qv = try allocator.alloc(f32, K * Nq);
    defer allocator.free(qv);
    for (qv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) * 0.1;
    const kv = try allocator.alloc(f32, K * Nk);
    defer allocator.free(kv);
    for (kv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) * 0.1;
    const q_packed = try packQ8WeightKN(allocator, qv, K, Nq);
    defer allocator.free(q_packed);
    const k_packed = try packQ8WeightKN(allocator, kv, K, Nk);
    defer allocator.free(k_packed);

    const wq = try export_ctx.fromPackedQuant(.q8_0, &[_]usize{ K, Nq }, 0, q_packed);
    const wk = try export_ctx.fromPackedQuant(.q8_0, &[_]usize{ K, Nk }, 0, k_packed);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const file = try createTestFile(tmp.dir, "fused_swap.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);

    var bld = api.Builder.init(&export_ctx);
    defer bld.deinit();
    const X = try bld.name(try bld.input(.f32, &[_]usize{ M, K }), "x");
    const Q = try bld.name(try bld.param(wq), "q");
    const Kp = try bld.name(try bld.param(wk), "k");
    const Oq = try bld.matmul(X, Q, 1.0, 0.0);
    const Ok = try bld.matmul(X, Kp, 1.0, 0.0);
    try export_ctx.exportModel(file, &bld, &[_]api.NamedTensorRef{
        .{ .name = "oq", .tensor = Oq },
        .{ .name = "ok", .tensor = Ok },
    }, .{});

    var load_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer load_ctx.deinit();
    var model = try load_ctx.loadModel(file, .{ .passes = .initOne(.horizontal_matmul) });
    defer model.deinit();

    const xv = try allocator.alloc(f32, M * K);
    defer allocator.free(xv);
    for (xv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) * 0.05;
    const x_t = try load_ctx.fromF32(&[_]usize{ M, K }, xv);
    try model.bindInput("x", x_t);
    try model.run();

    var oq_before: [M * Nq]f32 = undefined;
    var ok_before: [M * Nk]f32 = undefined;
    {
        const t = try model.outputTensor("oq");
        try t.read(&oq_before);
    }
    {
        const t = try model.outputTensor("ok");
        try t.read(&ok_before);
    }
    // Sanity: q's output is meaningfully non-zero so the swap below is observable.
    var any_nonzero = false;
    for (oq_before) |v| {
        if (@abs(v) > 1e-3) any_nonzero = true;
    }
    try std.testing.expect(any_nonzero);

    // "q" was fused into the combined weight: no standalone in-place handle...
    try std.testing.expectError(error.InvalidArgument, model.initializerTensorByDebugName("q"));

    // ...but its current value is still readable — materialized out of the fused
    // weight (the read counterpart of write-through). It must be non-zero here.
    const zero_packed = try allocator.alloc(u8, (K / 32) * Nq * 34);
    defer allocator.free(zero_packed);
    @memset(zero_packed, 0);
    const q_back = try allocator.alloc(u8, (K / 32) * Nq * 34);
    defer allocator.free(q_back);

    var scratch_q = try load_ctx.fromPackedQuant(.q8_0, &[_]usize{ K, Nq }, 0, zero_packed);
    try model.readInitializerByDebugName("q", scratch_q);
    try scratch_q.readPackedQuant(q_back);
    var q_back_nonzero = false;
    for (q_back) |b| {
        if (b != 0) q_back_nonzero = true;
    }
    try std.testing.expect(q_back_nonzero);

    // Swap "q" for an all-zero weight via write-through, then re-run.
    const zero_w = try load_ctx.fromPackedQuant(.q8_0, &[_]usize{ K, Nq }, 0, zero_packed);
    try model.overwriteInitializerByDebugName("q", zero_w);
    try model.run();

    var oq_after: [M * Nq]f32 = undefined;
    var ok_after: [M * Nk]f32 = undefined;
    {
        const t = try model.outputTensor("oq");
        try t.read(&oq_after);
    }
    {
        const t = try model.outputTensor("ok");
        try t.read(&ok_after);
    }
    // q's output collapses to zero (write-through landed in q's columns)...
    for (oq_after) |v| try std.testing.expectApproxEqAbs(@as(f32, 0.0), v, 1e-6);
    // ...and k's output is untouched (isolation across the fused weight).
    for (ok_after, ok_before) |a, b| try std.testing.expectApproxEqAbs(b, a, 1e-6);

    // Read-back now reflects the swap: q materializes as all-zero.
    try model.readInitializerByDebugName("q", scratch_q);
    try scratch_q.readPackedQuant(q_back);
    for (q_back) |b| try std.testing.expectEqual(@as(u8, 0), b);
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

    var bld = api.Builder.init(&export_ctx);
    defer bld.deinit();

    const X: api.TensorRef = try bld.name(try bld.input(.f32, &[_]usize{ 1, 2 }), "x");

    var head: nn.Linear = try nn.Linear.bind(&bld, .{ .weight = head0_w, .bias = head0_b }, .{});
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
        var bld = api.Builder.init(&export_ctx);
        defer bld.deinit();

        const X: api.TensorRef = try bld.name(try bld.input(.f32, &[_]usize{ 1, 2 }), "x");
        var bb: nn.Linear = try nn.Linear.bind(&bld, .{ .weight = bb_w, .bias = bb_b }, .{});
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

    var bld_cls = api.Builder.init(&ctx);
    defer bld_cls.deinit();
    const Xc: api.TensorRef = try bld_cls.name(try bld_cls.param(x_t), "x");
    var bb_cls: nn.Linear = try nn.Linear.bind(&bld_cls, .{ .weight = bb_w_loaded, .bias = bb_b_loaded }, .{});
    const HiddenC: api.TensorRef = try bb_cls.forward(&bld_cls, Xc);
    var head_cls: nn.Linear = try nn.Linear.bind(&bld_cls, .{ .weight = cls_w, .bias = cls_b }, .{});
    head_cls.w = try bld_cls.name(head_cls.w, "head.classifier.w");
    if (head_cls.b) |b0| head_cls.b = try bld_cls.name(b0, "head.classifier.b");
    const LogitsC: api.TensorRef = try head_cls.forward(&bld_cls, HiddenC);

    var model_cls = try ctx.compile(&bld_cls, &[_]api.TensorRef{LogitsC}, .{});
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

    var bld_qa = api.Builder.init(&ctx);
    defer bld_qa.deinit();
    const Xq: api.TensorRef = try bld_qa.name(try bld_qa.param(x_t), "x");
    var bb_qa: nn.Linear = try nn.Linear.bind(&bld_qa, .{ .weight = bb_w_loaded, .bias = bb_b_loaded }, .{});
    const HiddenQ: api.TensorRef = try bb_qa.forward(&bld_qa, Xq);

    const Ws: api.TensorRef = try bld_qa.name(try bld_qa.param(qa_w_s), "head.qa.start.w");
    const We: api.TensorRef = try bld_qa.name(try bld_qa.param(qa_w_e), "head.qa.end.w");
    const Start: api.TensorRef = try bld_qa.matmul(HiddenQ, Ws, 1.0, 0.0);
    const End: api.TensorRef = try bld_qa.matmul(HiddenQ, We, 1.0, 0.0);

    var model_qa = try ctx.compile(&bld_qa, &[_]api.TensorRef{ Start, End }, .{});
    defer model_qa.deinit();
    const out_s_t: api.Tensor = try model_qa.runOutputTensor(0);
    const out_e_t: api.Tensor = try model_qa.outputTensorAt(1);
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

    var bld = api.Builder.init(&export_ctx);
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
            .{ .input = State, .output = NextState },
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

    var bld = api.Builder.init(&export_ctx);
    defer bld.deinit();

    const X0 = try bld.input(.f32, &[_]usize{ 1, 3 });
    const X = try bld.name(X0, "x");
    const W: api.TensorRef = try bld.param(w_t);
    const Y: api.TensorRef = try bld.matmul(X, W, 1.0, 0.0);

    try bld.symbolicDim(X, 0, "batch");
    try export_ctx.exportModelPathAbsolute(absolute_file_path, &bld, &[_]api.NamedTensorRef{
        .{ .name = "y", .tensor = Y },
    }, .{});

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

    var bld = api.Builder.init(&ctx);
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

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Y}, .{});
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

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const X: api.TensorRef = try bld.param(x_t);

    var linear: nn.Linear = try nn.Linear.bind(&bld, .{ .weight = w_t, .bias = b_t }, .{});
    var dyn_mod: api.ModuleDyn = api.moduleDynFrom(nn.Linear, &linear);

    try std.testing.expect(api.isForwardModuleType(nn.Linear));
    try std.testing.expectEqualStrings("Linear", dyn_mod.name());

    const Y: api.TensorRef = try dyn_mod.forward(&bld, X);

    // `moduleDynFrom` marks the vtable `self_scoping`, so ModuleDyn does not add
    // a scope of its own — the wrapped `nn.Linear.forward` already opens
    // `Linear#0`. Without that flag this would double-nest as
    // `Linear#0/Linear#0/...`.
    try std.testing.expect(bld.valueName(Y) != null);
    try std.testing.expectEqualStrings("Linear#0/add#1", bld.valueName(Y).?);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Y}, .{});
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

    var bld = api.Builder.init(&export_ctx);
    defer bld.deinit();

    const X: api.TensorRef = try bld.name(try bld.input(.f32, &[_]usize{ 1, 2 }), "x");
    const linear: nn.Linear = try nn.Linear.bind(&bld, .{ .weight = w_t }, .{});
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
            const gx: api.TensorRef = try bld.add(x, self.gain);
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
    var bld = api.Builder.init(&export_ctx);
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

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    var bld = api.Builder.init(&ctx);
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

    var bld = api.Builder.init(&export_ctx);
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

test "api: an explicit builder scope nests above the nn auto module scope" {
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

    var bld = api.Builder.init(&export_ctx);
    defer bld.deinit();

    const X: api.TensorRef = try bld.name(try bld.input(.f32, &[_]usize{ 1, 2 }), "x");
    const linear: nn.Linear = try nn.Linear.bind(&bld, .{ .weight = w_t }, .{});

    const scope = try bld.beginScope("user.block");
    defer bld.endScope(scope);
    const Y: api.TensorRef = try linear.forward(&bld, X);

    // Scopes nest rather than the outer one suppressing the layer's own: the
    // layer stays identifiable inside the block it was used in.
    const want: []const u8 = "user.block/Linear#0/matmul#0";
    try std.testing.expect(bld.valueName(Y) != null);
    try std.testing.expectEqualStrings(want, bld.valueName(Y).?);

    try export_ctx.exportModel(file, &bld, &[_]api.NamedTensorRef{
        .{ .name = "y", .tensor = Y },
    }, .{});

    var pkg = try package_file.readPackageFile(allocator, file);
    defer pkg.deinit();

    var found: bool = false;
    for (pkg.debug_names) |entry| {
        if (entry.value == Y.value) {
            found = true;
            try std.testing.expectEqualStrings(want, entry.name);
        }
    }
    try std.testing.expect(found);
}

test "api: matmul aligns operand ranks so a 2-D weight serves a 3-D activation" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    // [1, 2, 2] activation against a plain [2, 2] weight. The weight broadcasts
    // into the activation's batch dim rather than being reshaped to [1, 2, 2] —
    // the reshape every converter used to write by hand.
    const x_t: api.Tensor = try ctx.fromF32(&[_]usize{ 1, 2, 2 }, &[_]f32{ 1.0, 2.0, 3.0, 4.0 });
    const w_t: api.Tensor = try ctx.fromF32(&[_]usize{ 2, 2 }, &[_]f32{ 1.0, 0.0, 0.0, 1.0 });

    const X: api.TensorRef = try bld.param(x_t);
    const W: api.TensorRef = try bld.param(w_t);
    const Y: api.TensorRef = try bld.matmul(X, W, 1.0, 0.0);

    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 2, 2 }, bld.knownShape(Y).?);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Y}, .{});
    defer model.deinit();

    const out_t: api.Tensor = try model.runOutputTensor(0);
    var out: [4]f32 = undefined;
    try out_t.read(&out);
    // Identity weight, so the activation passes through unchanged.
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1.0, 2.0, 3.0, 4.0 }, &out);
}

test "api: a quantized 2-D weight serves a 3-D activation without reshaping" {
    // The shape a real transformer has, and the case rank-padding could not serve:
    // a quantized weight's packing does not survive a reshape, so the weight has to
    // reach the kernel exactly as bound. Broadcasting is what makes that possible.
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const k: usize = 64;
    const n: usize = 32;

    var w_vals: [k * n]f32 = undefined;
    for (&w_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) * 0.05;
    var x_vals: [2 * k]f32 = undefined;
    for (&x_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) * 0.1;

    const X: api.TensorRef = try bld.param(try ctx.fromF32(&[_]usize{ 1, 2, k }, &x_vals));
    const W: api.TensorRef = try bld.param(try ctx.fromF32Quantized(.q8_0, &[_]usize{ k, n }, 0, &w_vals));
    const Y: api.TensorRef = try bld.matmul(X, W, 1.0, 0.0);

    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 2, n }, bld.knownShape(Y).?);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Y}, .{});
    defer model.deinit();

    const out_t: api.Tensor = try model.runOutputTensor(0);
    var out: [2 * n]f32 = undefined;
    try out_t.read(&out);

    // Against a q8_0 reference computed from the same values, so a wrong tile walk
    // over the broadcast weight cannot agree with itself.
    for (0..2) |row| {
        for (0..n) |col| {
            var acc: f32 = 0.0;
            for (0..k) |d| acc += x_vals[row * k + d] * w_vals[d * n + col];
            try std.testing.expectApproxEqAbs(acc, out[row * n + col], 0.05);
        }
    }
}

test "api: matmul still rejects a genuine inner-dim mismatch" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    // Rank alignment must not paper over a real shape error: K=2 vs K=3.
    const X: api.TensorRef = try bld.param(try ctx.fromF32(&[_]usize{ 1, 2, 2 }, &[_]f32{ 1, 2, 3, 4 }));
    const W: api.TensorRef = try bld.param(try ctx.fromF32(&[_]usize{ 3, 2 }, &[_]f32{ 1, 2, 3, 4, 5, 6 }));
    try std.testing.expectError(error.ShapeMismatch, bld.matmul(X, W, 1.0, 0.0));
}

test "api: builder constants are cached and named independently of scope" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const z1: api.TensorRef = try bld.zeros(4);
    const o1: api.TensorRef = try bld.ones(4);
    const k1: api.TensorRef = try bld.constant(0.5);

    // Requested again from inside a scope: still the same value, and the name is
    // NOT scope-qualified — a shared constant belongs to no single layer.
    {
        const scope = try bld.beginScope("block");
        defer bld.endScope(scope);
        try std.testing.expectEqual(z1.value, (try bld.zeros(4)).value);
        try std.testing.expectEqual(o1.value, (try bld.ones(4)).value);
        try std.testing.expectEqual(k1.value, (try bld.constant(0.5)).value);
    }

    try std.testing.expectEqualStrings("const.zeros.4", bld.valueName(z1).?);
    try std.testing.expectEqualStrings("const.ones.4", bld.valueName(o1).?);

    // Distinct widths and distinct values get distinct constants.
    try std.testing.expect((try bld.zeros(8)).value != z1.value);
    try std.testing.expect((try bld.constant(0.25)).value != k1.value);
    try std.testing.expectError(error.InvalidArgument, bld.zeros(0));
}

test "api: scopes nest and number siblings per parent" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const X: api.TensorRef = try bld.input(.f32, &[_]usize{ 1, 4 });
    try std.testing.expectEqualStrings("", bld.scopePath());

    var names: [4][]const u8 = undefined;
    var block_idx: usize = 0;
    while (block_idx < 2) : (block_idx += 1) {
        const outer = try bld.beginAutoScope("Block");
        defer bld.endScope(outer);

        var inner_idx: usize = 0;
        while (inner_idx < 2) : (inner_idx += 1) {
            const inner = try bld.beginAutoScope("Lin");
            defer bld.endScope(inner);
            const y: api.TensorRef = try bld.relu(X);
            names[block_idx * 2 + inner_idx] = bld.valueName(y).?;
        }
    }

    // Sibling counters restart under each parent, and op counters restart in each
    // scope — so the two blocks produce the same inner names under their own path.
    try std.testing.expectEqualStrings("Block#0/Lin#0/relu#0", names[0]);
    try std.testing.expectEqualStrings("Block#0/Lin#1/relu#0", names[1]);
    try std.testing.expectEqualStrings("Block#1/Lin#0/relu#0", names[2]);
    try std.testing.expectEqualStrings("Block#1/Lin#1/relu#0", names[3]);

    // Every scope closed, so the path is empty again and root numbering resumes.
    try std.testing.expect(!bld.hasActiveScope());
    try std.testing.expectEqualStrings("", bld.scopePath());
    const z: api.TensorRef = try bld.relu(X);
    try std.testing.expectEqualStrings("relu#0", bld.valueName(z).?);
}

test "api: paramNamed gives params a semantic, scope-qualified debug name" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var export_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer export_ctx.deinit();

    const w_t: api.Tensor = try export_ctx.fromArray([2][3]f32{
        .{ 1.0, -2.0, 0.5 },
        .{ 3.0, 4.0, -1.5 },
    });
    const b_t: api.Tensor = try export_ctx.fromArray([3]f32{ 0.25, -0.5, 1.0 });

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const file = try createTestFile(tmp.dir, "param_named.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);

    var bld = api.Builder.init(&export_ctx);
    defer bld.deinit();

    const X: api.TensorRef = try bld.name(try bld.input(.f32, &[_]usize{ 1, 2 }), "x");

    var W: api.TensorRef = undefined;
    var B: api.TensorRef = undefined;
    var MM: api.TensorRef = undefined;
    var Y: api.TensorRef = undefined;
    {
        const outer = try bld.beginScope("layers.3");
        defer bld.endScope(outer);
        const inner = try bld.beginScope("q_proj");
        defer bld.endScope(inner);

        W = try bld.paramNamed(w_t, "weight");
        B = try bld.paramNamed(b_t, "bias");
        MM = try bld.matmul(X, W, 1.0, 0.0);
        Y = try bld.add(MM, B);
    }

    try std.testing.expectEqualStrings("layers.3/q_proj/weight", bld.valueName(W).?);
    try std.testing.expectEqualStrings("layers.3/q_proj/bias", bld.valueName(B).?);
    // Params must not consume op indices: the matmul is still #0 even though two
    // params were bound before it.
    try std.testing.expectEqualStrings("layers.3/q_proj/matmul#0", bld.valueName(MM).?);
    try std.testing.expectEqualStrings("layers.3/q_proj/add#1", bld.valueName(Y).?);

    // An empty name is rejected rather than silently producing a trailing slash.
    try std.testing.expectError(error.InvalidArgument, bld.paramNamed(b_t, ""));

    // The semantic names persist into the package, which is what makes
    // swap-by-debug-name usable.
    try export_ctx.exportModel(file, &bld, &[_]api.NamedTensorRef{.{ .name = "y", .tensor = Y }}, .{});
    var pkg = try package_file.readPackageFile(allocator, file);
    defer pkg.deinit();

    var saw_w: bool = false;
    var saw_b: bool = false;
    for (pkg.debug_names) |entry| {
        if (std.mem.eql(u8, entry.name, "layers.3/q_proj/weight")) saw_w = true;
        if (std.mem.eql(u8, entry.name, "layers.3/q_proj/bias")) saw_b = true;
    }
    try std.testing.expect(saw_w);
    try std.testing.expect(saw_b);
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

    var bld = api.Builder.init(&export_ctx);
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

test "api: sequenceAppend mutates cache in-place" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    const cache_t: api.Tensor = try ctx.fromF32(&[_]usize{ 1, 4, 1, 2 }, &[_]f32{ 0, 1, 2, 3, 4, 5, 6, 7 });
    const new_t: api.Tensor = try ctx.fromF32(&[_]usize{ 1, 2, 1, 2 }, &[_]f32{ 50, 51, 60, 61 });
    const end_t: api.Tensor = try ctx.fromArray([1]i32{1});

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const Cache: api.TensorRef = try bld.param(cache_t);
    const New: api.TensorRef = try bld.param(new_t);
    const End: api.TensorRef = try bld.param(end_t);
    const Out: api.TensorRef = try bld.sequenceAppend(Cache, New, End);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Out}, .{});
    defer model.deinit();

    const out_t: api.Tensor = try model.runOutputTensor(0);
    try std.testing.expectEqual(cache_t.id, out_t.id);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 4, 1, 2 }, out_t.getShape());

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

test "api: sequenceAppend ring policy wraps" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{
        .thread_count = 1,
        .cache_config = .{ .ram_budget_bytes = 1 << 20 },
    });
    defer ctx.deinit();

    const cache_t: api.Tensor = try ctx.fromF32(&[_]usize{ 1, 4, 1, 1 }, &[_]f32{ 0, 1, 2, 3 });
    const new_t: api.Tensor = try ctx.fromF32(&[_]usize{ 1, 2, 1, 1 }, &[_]f32{ 90, 91 });
    const end_t: api.Tensor = try ctx.fromArray([1]i32{3});
    try ctx.setTensorSequenceCachePolicy(cache_t, .{ .ring = .{ .window_tokens = 4 } });

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const Cache: api.TensorRef = try bld.param(cache_t);
    const New: api.TensorRef = try bld.param(new_t);
    const End: api.TensorRef = try bld.param(end_t);
    const Out: api.TensorRef = try bld.sequenceAppend(Cache, New, End);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Out}, .{});
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

test "api: sequenceAppend growable policy expands physical capacity" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{
        .thread_count = 1,
        .cache_config = .{ .ram_budget_bytes = 1 << 20 },
    });
    defer ctx.deinit();

    const cache_t: api.Tensor = try ctx.fromF32(&[_]usize{ 1, 4, 1, 1 }, &[_]f32{ 0, 1, 2, 3 });
    const new_t: api.Tensor = try ctx.fromF32(&[_]usize{ 1, 2, 1, 1 }, &[_]f32{ 90, 91 });
    const end_t: api.Tensor = try ctx.fromArray([1]i32{5});

    try ctx.setTensorSequenceCachePolicy(cache_t, .{ .growable = .{ .initial_capacity_tokens = 2, .growth_numerator = 2, .growth_denominator = 1 } });

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const Cache: api.TensorRef = try bld.param(cache_t);
    const New: api.TensorRef = try bld.param(new_t);
    const End: api.TensorRef = try bld.param(end_t);
    const Out: api.TensorRef = try bld.sequenceAppend(Cache, New, End);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Out}, .{});
    defer model.deinit();

    const out_t: api.Tensor = try model.runOutputTensor(0);
    try std.testing.expectEqual(cache_t.id, out_t.id);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 8, 1, 1 }, out_t.getShape());

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

test "api: attention matches deterministic windowed-causal averages" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    // Shapes:
    // q:       [B=1, Lq=2, Hq=4, Dk=2]
    // k_cache: [B=1, T=6, Hkv=2, Dk=2]
    // v_cache: [B=1, T=6, Hkv=2, Dv=1]
    // positions: [1,2], end_index: [1]
    var q_vals: [1 * 2 * 4 * 2]f32 = @splat(0.0);
    var k_vals: [1 * 2 * 6 * 2]f32 = @splat(0.0);
    var v_vals: [1 * 2 * 6 * 1]f32 = @splat(0.0);

    // kv-head 0 tokens => [1,2,3,4,5,6]
    // kv-head 1 tokens => [10,20,30,40,50,60]
    for (0..6) |t| {
        v_vals[t * 2] = @as(f32, @floatFromInt(@as(i32, @intCast(t + 1))));
        v_vals[t * 2 + 1] = @as(f32, @floatFromInt(@as(i32, @intCast((t + 1) * 10))));
    }

    const pos_t: api.Tensor = try ctx.fromArray([1][2]i32{.{ 4, 5 }});
    const end_t: api.Tensor = try ctx.fromArray([1]i32{6});
    const q_t: api.Tensor = try ctx.fromF32(&[_]usize{ 1, 2, 4, 2 }, q_vals[0..]);
    const k_t: api.Tensor = try ctx.fromF32(&[_]usize{ 1, 6, 2, 2 }, k_vals[0..]);
    const v_t: api.Tensor = try ctx.fromF32(&[_]usize{ 1, 6, 2, 1 }, v_vals[0..]);

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const Q: api.TensorRef = try bld.param(q_t);
    const K: api.TensorRef = try bld.param(k_t);
    const V: api.TensorRef = try bld.param(v_t);
    const Pos: api.TensorRef = try bld.param(pos_t);
    const End: api.TensorRef = try bld.param(end_t);

    const Out: api.TensorRef = try bld.attention(
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

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Out}, .{});
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

test "api: attention validates H_q % H_kv == 0" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    var q_vals: [1 * 1 * 3 * 2]f32 = @splat(0.0);
    var k_vals: [1 * 2 * 4 * 2]f32 = @splat(0.0);
    var v_vals: [1 * 2 * 4 * 1]f32 = @splat(0.0);

    const q_t: api.Tensor = try ctx.fromF32(&[_]usize{ 1, 1, 3, 2 }, q_vals[0..]);
    const k_t: api.Tensor = try ctx.fromF32(&[_]usize{ 1, 4, 2, 2 }, k_vals[0..]);
    const v_t: api.Tensor = try ctx.fromF32(&[_]usize{ 1, 4, 2, 1 }, v_vals[0..]);
    const pos_t: api.Tensor = try ctx.fromArray([1][1]i32{.{0}});
    const end_t: api.Tensor = try ctx.fromArray([1]i32{1});

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const Q: api.TensorRef = try bld.param(q_t);
    const K: api.TensorRef = try bld.param(k_t);
    const V: api.TensorRef = try bld.param(v_t);
    const Pos: api.TensorRef = try bld.param(pos_t);
    const End: api.TensorRef = try bld.param(end_t);

    // H_q (3) % H_kv (2) != 0: eager per-op inference rejects this at the op call.
    try std.testing.expectError(error.ShapeMismatch, bld.attention(Q, K, V, Pos, End, 1.0, true, 0, 0.0));
}

/// Dequantize a `[K, N]` q8_0 weight packed by `packQ8WeightKN` back to f32.
fn dequantQ8KN(allocator: std.mem.Allocator, packed_bytes: []const u8, k: usize, n: usize) ![]f32 {
    const kb = k / 32;
    const out = try allocator.alloc(f32, k * n);
    for (0..kb) |b| {
        for (0..n) |j| {
            const off = (b * n + j) * 34;
            const scale_bits = std.mem.readInt(u16, packed_bytes[off .. off + 2][0..2], .little);
            const scale: f32 = @floatCast(@as(f16, @bitCast(scale_bits)));
            for (0..32) |t| {
                const q: i8 = @bitCast(packed_bytes[off + 2 + t]);
                out[(b * 32 + t) * n + j] = @as(f32, @floatFromInt(q)) * scale;
            }
        }
    }
    return out;
}

// Synthetic replacement for the old opt-in real-checkpoint (Gemma) verification:
// a q8_0 weight exported + loaded must stay q8_0 at its packed byte size (not be
// silently dequantized/inflated), and the loaded model must execute to the same
// result as a reference dequant matmul.
test "api: export/load keeps q8_0 weights packed (dtype + byte size) and runs" {
    const allocator: std.mem.Allocator = std.testing.allocator;
    const K: usize = 64;
    const N: usize = 32;
    const M: usize = 2;

    var export_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer export_ctx.deinit();

    const wv = try allocator.alloc(f32, K * N);
    defer allocator.free(wv);
    for (wv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 23)) - 11)) * 0.07;
    const w_packed = try packQ8WeightKN(allocator, wv, K, N);
    defer allocator.free(w_packed);

    var bias: [M * N]f32 = undefined;
    for (&bias, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 5)) - 2)) * 0.5;

    const wq = try export_ctx.fromPackedQuant(.q8_0, &[_]usize{ K, N }, 0, w_packed);
    const bf = try export_ctx.fromF32(&[_]usize{ M, N }, &bias);

    var bld = api.Builder.init(&export_ctx);
    defer bld.deinit();
    const X = try bld.name(try bld.input(.f32, &[_]usize{ M, K }), "x");
    const W = try bld.name(try bld.param(wq), "w");
    const B = try bld.name(try bld.param(bf), "bias");
    const Y = try bld.add(try bld.matmul(X, W, 1.0, 0.0), B);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const file = try createTestFile(tmp.dir, "q8_keep.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);
    try export_ctx.exportModel(file, &bld, &[_]api.NamedTensorRef{.{ .name = "y", .tensor = Y }}, .{});

    var load_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer load_ctx.deinit();
    var model = try load_ctx.loadModel(file, .{});
    defer model.deinit();

    // Initializer-backed weights must keep their dtype AND their exact stored
    // byte size: q8_0 stays packed (34 B per 32-element block), f32 stays scalar.
    var q8_count: usize = 0;
    var f32_count: usize = 0;
    for (model.template.values, 0..) |value, idx| {
        if (value.source != .initializer) continue;
        const wt = try model.initializerTensorByValue(@intCast(idx));
        const meta = try load_ctx.store.getConst(wt.tensorId());
        switch (meta.dtype) {
            .q8_0 => {
                q8_count += 1;
                try std.testing.expectEqual((K / 32) * N * 34, meta.data.len);
            },
            .f32 => {
                f32_count += 1;
                try std.testing.expectEqual(M * N * @sizeOf(f32), meta.data.len);
            },
            else => return error.TestUnexpectedResult,
        }
    }
    try std.testing.expectEqual(@as(usize, 1), q8_count);
    try std.testing.expectEqual(@as(usize, 1), f32_count);

    // Execute and compare against a reference matmul over the dequantized weight.
    var xv: [M * K]f32 = undefined;
    for (&xv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) * 0.05;
    const x_t = try load_ctx.fromF32(&[_]usize{ M, K }, &xv);
    try model.bindInput("x", x_t);
    try model.run();

    var got: [M * N]f32 = undefined;
    {
        const t = try model.outputTensor("y");
        try t.read(&got);
    }
    const w_deq = try dequantQ8KN(allocator, w_packed, K, N);
    defer allocator.free(w_deq);
    for (0..M) |m| {
        for (0..N) |j| {
            var acc: f32 = bias[m * N + j];
            for (0..K) |kk| acc += xv[m * K + kk] * w_deq[kk * N + j];
            try std.testing.expectApproxEqAbs(acc, got[m * N + j], 1e-3);
        }
    }
}

test "api: exportModel + loadModel roundtrip for If control-flow model" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var export_ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer export_ctx.deinit();

    var bld = api.Builder.init(&export_ctx);
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

    var bld = api.Builder.init(&export_ctx);
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

test "api: fromF32Quantized authors a q8_0 matmul matching the dequant reference" {
    const allocator: std.mem.Allocator = std.testing.allocator;
    const K: usize = 64;
    const N: usize = 32;
    const M: usize = 2;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    // Float weight authored the idiomatic way: pass f32, the core quantizes.
    const wv = try allocator.alloc(f32, K * N);
    defer allocator.free(wv);
    for (wv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 23)) - 11)) * 0.07;

    const wq = try ctx.fromF32Quantized(.q8_0, &[_]usize{ K, N }, 0, wv);
    try std.testing.expectEqual(types.DType.q8_0, wq.getDType());

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();
    const X = try bld.name(try bld.input(.f32, &[_]usize{ M, K }), "x");
    const W = try bld.param(wq);
    const Y = try bld.matmul(X, W, 1.0, 0.0);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Y}, .{});
    defer model.deinit();

    var xv: [M * K]f32 = undefined;
    for (&xv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) * 0.05;
    const x_t = try ctx.fromF32(&[_]usize{ M, K }, &xv);
    try model.bindInput("x", x_t);
    try model.run();

    var got: [M * N]f32 = undefined;
    {
        const t = try model.runOutputTensor(0);
        try t.read(&got);
    }

    // Reference: dequantize the SAME packed bytes the core produced and matmul.
    const w_packed = try quantize.quantizeF32(allocator, .q8_0, &[_]usize{ K, N }, 0, wv);
    defer allocator.free(w_packed);
    const w_deq = try dequantQ8KN(allocator, w_packed, K, N);
    defer allocator.free(w_deq);
    for (0..M) |m| {
        for (0..N) |j| {
            var acc: f32 = 0;
            for (0..K) |kk| acc += xv[m * K + kk] * w_deq[kk * N + j];
            try std.testing.expectApproxEqAbs(acc, got[m * N + j], 1e-3);
        }
    }
}

test "api: builder op-parity wrappers compile and run (cast / gate / elemwiseBinary)" {
    const allocator: std.mem.Allocator = std.testing.allocator;
    const N: usize = 4;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    // a (f16 input, cast to f32), b, c: y = (a_f32 - b) * c, then gate(gelu, y, c).
    const A = try bld.name(try bld.input(.f16, &[_]usize{ 1, N }), "a");
    const B = try bld.name(try bld.input(.f32, &[_]usize{ 1, N }), "b");
    const C = try bld.name(try bld.input(.f32, &[_]usize{ 1, N }), "c");
    const a_f32 = try bld.cast(A, .f32);
    const diff = try bld.elemwiseBinary(.sub, a_f32, B);
    const scaled = try bld.mul(diff, C);
    const Y = try bld.elemwiseBinary(.mul, try bld.unary(.gelu, scaled), C);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Y}, .{});
    defer model.deinit();

    var av: [N]f16 = undefined;
    for (&av, 0..) |*v, i| v.* = @floatFromInt(i + 2);
    var bv: [N]f32 = .{ 1.0, 1.0, 1.0, 1.0 };
    var cv: [N]f32 = .{ 0.5, 0.5, 0.5, 0.5 };
    const a_t = try ctx.fromF16(&[_]usize{ 1, N }, &av);
    const b_t = try ctx.fromF32(&[_]usize{ 1, N }, &bv);
    const c_t = try ctx.fromF32(&[_]usize{ 1, N }, &cv);
    try model.bindInput("a", a_t);
    try model.bindInput("b", b_t);
    try model.bindInput("c", c_t);
    try model.run();

    var got: [N]f32 = undefined;
    {
        const t = try model.runOutputTensor(0);
        try t.read(&got);
    }
    for (0..N) |i| {
        const scaled_v: f32 = (@as(f32, @floatCast(av[i])) - bv[i]) * cv[i];
        const gelu: f32 = 0.5 * scaled_v * (1.0 + std.math.tanh(0.7978845608 * (scaled_v + 0.044715 * scaled_v * scaled_v * scaled_v)));
        try std.testing.expectApproxEqAbs(gelu * cv[i], got[i], 2e-2);
    }
}

test "api: builder control-flow wrappers compile and run (If + Loop, in-process)" {
    const allocator: std.mem.Allocator = std.testing.allocator;
    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    // If: select then/else by an i32 cond, all via Builder region wrappers.
    {
        var bld = api.Builder.init(&ctx);
        defer bld.deinit();
        const cond = try bld.name(try bld.input(.i32, &[_]usize{1}), "cond");
        const then_v = try bld.name(try bld.input(.f32, &[_]usize{1}), "then_v");
        const else_v = try bld.name(try bld.input(.f32, &[_]usize{1}), "else_v");
        try bld.beginRegion();
        const tr = try bld.endRegion(&[_]api.TensorRef{then_v});
        try bld.beginRegion();
        const er = try bld.endRegion(&[_]api.TensorRef{else_v});
        const out = try bld.name(try bld.ifThenElse(cond, tr, er), "out");

        var model = try ctx.compile(&bld, &[_]api.TensorRef{out}, .{});
        defer model.deinit();
        try model.bindInput("cond", try ctx.vector([_]i32{1}));
        try model.bindInput("then_v", try ctx.vector([_]f32{42.0}));
        try model.bindInput("else_v", try ctx.vector([_]f32{7.0}));
        try model.run();
        var v: [1]f32 = undefined;
        try (try model.outputTensor("out")).read(&v);
        try std.testing.expectApproxEqAbs(@as(f32, 42.0), v[0], 1e-6);
    }

    // Loop: carried += inc, 4 trips → 1 + 2*4 = 9.
    {
        var bld = api.Builder.init(&ctx);
        defer bld.deinit();
        const carried = try bld.name(try bld.input(.f32, &[_]usize{1}), "carried");
        const inc = try bld.name(try bld.input(.f32, &[_]usize{1}), "inc");
        try bld.beginRegion();
        const next = try bld.add(carried, inc);
        const body = try bld.endRegion(&[_]api.TensorRef{next});
        const out = try bld.name(try bld.loop(carried, body, 4), "out");

        var model = try ctx.compile(&bld, &[_]api.TensorRef{out}, .{});
        defer model.deinit();
        try model.bindInput("carried", try ctx.vector([_]f32{1.0}));
        try model.bindInput("inc", try ctx.vector([_]f32{2.0}));
        try model.run();
        var v: [1]f32 = undefined;
        try (try model.outputTensor("out")).read(&v);
        try std.testing.expectApproxEqAbs(@as(f32, 9.0), v[0], 1e-6);
    }
}

test "api: symbolic input dim lets one compiled model serve multiple shapes" {
    const allocator: std.mem.Allocator = std.testing.allocator;
    const K: usize = 4;
    const N: usize = 3;
    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    // `x` has a symbolic row count ("seq"). WHY THIS MATTERS: the model is compiled
    // ONCE, yet runs for any number of rows — e.g. a transformer prefill (many
    // tokens) followed by decode (one token at a time) reuse the same compiled
    // model (and its state), instead of recompiling per sequence length. The
    // declared row count (1) is just a placeholder for the free axis.
    const X = try bld.name(try bld.input(.f32, &[_]usize{ 1, K }), "x");
    var wv: [K * N]f32 = undefined;
    for (&wv, 0..) |*v, i| v.* = @floatFromInt(i + 1);
    const W = try bld.param(try ctx.fromF32(&[_]usize{ K, N }, &wv));
    const Y = try bld.name(try bld.matmul(X, W, 1.0, 0.0), "y");

    try bld.symbolicDim(X, 0, "seq");
    var model = try ctx.compileOn(.cpu, &bld, &[_]api.TensorRef{Y}, .{});
    defer model.deinit();

    const runFor = struct {
        fn go(c: *api.Context, m: *api.Model, comptime M: usize, w: []const f32) !void {
            var xv: [M * K]f32 = undefined;
            for (&xv, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast(i % 7)) - 3);
            try m.bindInput("x", try c.fromF32(&[_]usize{ M, K }, &xv));
            try m.run();
            var got: [M * N]f32 = undefined;
            try (try m.outputTensor("y")).read(&got);
            for (0..M) |mm| {
                for (0..N) |n| {
                    var acc: f32 = 0;
                    for (0..K) |k| acc += xv[mm * K + k] * w[k * N + n];
                    try std.testing.expectApproxEqAbs(acc, got[mm * N + n], 1e-4);
                }
            }
        }
    }.go;

    // Same compiled model, three different row counts — no recompile from scratch.
    try runFor(&ctx, &model, 2, &wv);
    try runFor(&ctx, &model, 5, &wv);
    try runFor(&ctx, &model, 1, &wv);
    const warm = model.planCacheStats();
    try std.testing.expectEqual(@as(u64, 3), warm.builds);
    try std.testing.expectEqual(@as(usize, 3), warm.entries);
    try runFor(&ctx, &model, 2, &wv);
    const reused = model.planCacheStats();
    try std.testing.expectEqual(warm.builds, reused.builds);
}

test "api: symbolic compile still optimizes (parallel projections fuse) across shapes" {
    const allocator: std.mem.Allocator = std.testing.allocator;
    const K: usize = 8;
    const Nq: usize = 4;
    const Nk: usize = 4;
    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    // Two projections off a SHARED input with a symbolic row count. The
    // horizontal-matmul fusion pass runs during each per-shape compile (it is NOT
    // disabled for symbolic models); outputs must stay correct at every shape.
    const X = try bld.name(try bld.input(.f32, &[_]usize{ 1, K }), "x");
    var wqv: [K * Nq]f32 = undefined;
    for (&wqv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 5)) - 2)) * 0.1;
    var wkv: [K * Nk]f32 = undefined;
    for (&wkv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) * 0.1;
    const Wq = try bld.param(try ctx.fromF32(&[_]usize{ K, Nq }, &wqv));
    const Wk = try bld.param(try ctx.fromF32(&[_]usize{ K, Nk }, &wkv));
    const Q = try bld.matmul(X, Wq, 1.0, 0.0);
    const Kp = try bld.matmul(X, Wk, 1.0, 0.0);
    const Y = try bld.name(try bld.concat(&[_]api.TensorRef{ Q, Kp }, 1), "y");

    try bld.symbolicDim(X, 0, "seq");
    var model = try ctx.compileOn(.cpu, &bld, &[_]api.TensorRef{Y}, .{});
    defer model.deinit();

    const check = struct {
        fn go(c: *api.Context, m: *api.Model, comptime M: usize, wq: []const f32, wk: []const f32) !void {
            var xv: [M * K]f32 = undefined;
            for (&xv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 9)) - 4)) * 0.1;
            try m.bindInput("x", try c.fromF32(&[_]usize{ M, K }, &xv));
            try m.run();
            var got: [M * (Nq + Nk)]f32 = undefined;
            try (try m.outputTensor("y")).read(&got);
            for (0..M) |mm| {
                for (0..Nq) |n| {
                    var acc: f32 = 0;
                    for (0..K) |k| acc += xv[mm * K + k] * wq[k * Nq + n];
                    try std.testing.expectApproxEqAbs(acc, got[mm * (Nq + Nk) + n], 1e-4);
                }
                for (0..Nk) |n| {
                    var acc: f32 = 0;
                    for (0..K) |k| acc += xv[mm * K + k] * wk[k * Nk + n];
                    try std.testing.expectApproxEqAbs(acc, got[mm * (Nq + Nk) + Nq + n], 1e-4);
                }
            }
        }
    }.go;

    try check(&ctx, &model, 2, &wqv, &wkv);
    try check(&ctx, &model, 3, &wqv, &wkv);
}

test "api: eager inference exposes shapes during authoring (placeholder propagates)" {
    const allocator: std.mem.Allocator = std.testing.allocator;
    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    // Declare a dynamic row axis with placeholder 8. Eager inference resolves the
    // whole graph at the placeholder, so shapes are queryable while authoring and
    // the placeholder propagates deterministically to derived values.
    const X = try bld.input(.f32, &[_]usize{ 8, 4 });
    try bld.symbolicDim(X, 0, "seq");
    var wv: [4 * 3]f32 = undefined;
    for (&wv, 0..) |*v, i| v.* = @floatFromInt(i);
    const W = try bld.param(try ctx.fromF32(&[_]usize{ 4, 3 }, &wv));
    const Y = try bld.matmul(X, W, 1.0, 0.0);

    // Input shape = placeholder.
    try std.testing.expectEqualSlices(usize, &[_]usize{ 8, 4 }, bld.knownShape(X).?);
    // Derived shape is known immediately (no compile needed) and carries the
    // placeholder on the dynamic axis.
    try std.testing.expectEqualSlices(usize, &[_]usize{ 8, 3 }, bld.knownShape(Y).?);
    try std.testing.expectEqual(types.DType.f32, bld.dtypeOf(Y).?);
}

test "api: eager inference reports a shape error at the offending op" {
    const allocator: std.mem.Allocator = std.testing.allocator;
    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const A = try bld.input(.f32, &[_]usize{ 2, 4 });
    var wv: [3 * 3]f32 = undefined; // K=3 mismatches A's K=4
    for (&wv, 0..) |*v, i| v.* = @floatFromInt(i);
    const W = try bld.param(try ctx.fromF32(&[_]usize{ 3, 3 }, &wv));
    // The error is raised right here, not deferred to compile.
    try std.testing.expectError(error.ShapeMismatch, bld.matmul(A, W, 1.0, 0.0));
}

test "api: compile lowers only what the requested outputs need" {
    // Authoring something and not asking for it must not put it in the program.
    // The canary is a view of a quantized weight — an op the compiler rejects
    // outright — so if the dead branch were lowered this would fail rather than
    // silently produce a bigger program.
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const k: usize = 64;
    const n: usize = 32;
    var w_vals: [k * n]f32 = undefined;
    for (&w_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) * 0.05;
    var x_vals: [k]f32 = undefined;
    for (&x_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) * 0.1;

    const W: api.TensorRef = try bld.param(try ctx.fromF32Quantized(.q8_0, &[_]usize{ k, n }, 0, &w_vals));
    const X: api.TensorRef = try bld.param(try ctx.fromF32(&[_]usize{ 1, k }, &x_vals));
    const Y: api.TensorRef = try bld.matmul(X, W, 1.0, 0.0);

    // Dead branch: never reaches an output.
    _ = try bld.unsqueeze(W, 0);

    var model = try ctx.compile(&bld, &[_]api.TensorRef{Y}, .{});
    defer model.deinit();

    const out_t: api.Tensor = try model.runOutputTensor(0);
    var out: [n]f32 = undefined;
    try out_t.read(&out);

    for (0..n) |col| {
        var acc: f32 = 0.0;
        for (0..k) |d| acc += x_vals[d] * w_vals[d * n + col];
        try std.testing.expectApproxEqAbs(acc, out[col], 0.05);
    }
}

test "api: authoring survives compilation" {
    // `compile` hands the model a copy of the graph, so the builder stays usable:
    // compile again, compile a different output, keep building. This is what lets a
    // value be evaluated mid-authoring just to look at it.
    const allocator: std.mem.Allocator = std.testing.allocator;

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const X: api.TensorRef = try bld.param(try ctx.fromF32(&[_]usize{ 1, 2 }, &[_]f32{ 3.0, 4.0 }));
    const W: api.TensorRef = try bld.param(try ctx.fromF32(&[_]usize{ 2, 2 }, &[_]f32{ 1.0, 0.0, 0.0, 1.0 }));
    const Y: api.TensorRef = try bld.matmul(X, W, 1.0, 0.0);

    {
        var m1 = try ctx.compile(&bld, &[_]api.TensorRef{Y}, .{});
        defer m1.deinit();
        var got: [2]f32 = undefined;
        try (try m1.runOutputTensor(0)).read(&got);
        try std.testing.expectApproxEqAbs(@as(f32, 3.0), got[0], 1e-6);
    }

    // The graph is still here after compiling...
    try std.testing.expect(bld.innerGraph().nodes.items.len > 0);

    // ...so more ops build on what came before, and compile again.
    const Z: api.TensorRef = try bld.relu(try bld.add(Y, Y));
    {
        var m2 = try ctx.compile(&bld, &[_]api.TensorRef{Z}, .{});
        defer m2.deinit();
        var got: [2]f32 = undefined;
        try (try m2.runOutputTensor(0)).read(&got);
        try std.testing.expectApproxEqAbs(@as(f32, 6.0), got[0], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 8.0), got[1], 1e-6);
    }
}

// ---------------------------------------------------------------------------
// Two models, one store: a weight one of them fused away
// ---------------------------------------------------------------------------

/// Two q8_0 projections off one input, summed. `Context.compile` bakes the params as
/// initializers bound to the caller's tensors, so every model compiled from this builder
/// shares those tensor ids — and the store's derived table with them.
const SharedProj = struct {
    const K: usize = 64;
    const N: usize = 32;
    const M: usize = 2;

    ctx: api.Context,
    bld: api.Builder,
    out: api.TensorRef,
    q: api.TensorRef,
    k: api.TensorRef,
    /// A second copy of q's original bytes: the swap below overwrites q's own tensor.
    q_backup: api.Tensor,
    x: api.Tensor,

    fn init(allocator: std.mem.Allocator) !*SharedProj {
        const self = try allocator.create(SharedProj);
        errdefer allocator.destroy(self);
        self.ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });

        const qv = try allocator.alloc(f32, K * N);
        defer allocator.free(qv);
        for (qv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) * 0.1;
        const kv = try allocator.alloc(f32, K * N);
        defer allocator.free(kv);
        for (kv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) * 0.1;
        const q_packed = try packQ8WeightKN(allocator, qv, K, N);
        defer allocator.free(q_packed);
        const k_packed = try packQ8WeightKN(allocator, kv, K, N);
        defer allocator.free(k_packed);

        const wq = try self.ctx.fromPackedQuant(.q8_0, &[_]usize{ K, N }, 0, q_packed);
        self.q_backup = try self.ctx.fromPackedQuant(.q8_0, &[_]usize{ K, N }, 0, q_packed);
        const wk = try self.ctx.fromPackedQuant(.q8_0, &[_]usize{ K, N }, 0, k_packed);

        self.bld = api.Builder.init(&self.ctx);
        const X = try self.bld.name(try self.bld.input(.f32, &[_]usize{ M, K }), "x");
        const Q = try self.bld.name(try self.bld.param(wq), "q");
        const Kp = try self.bld.name(try self.bld.param(wk), "k");
        self.out = try self.bld.add(
            try self.bld.matmul(X, Q, 1.0, 0.0),
            try self.bld.matmul(X, Kp, 1.0, 0.0),
        );
        self.q = Q;
        self.k = Kp;

        const xv = try allocator.alloc(f32, M * K);
        defer allocator.free(xv);
        for (xv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) * 0.05;
        self.x = try self.ctx.fromF32(&[_]usize{ M, K }, xv);
        return self;
    }

    fn deinit(self: *SharedProj, allocator: std.mem.Allocator) void {
        self.bld.deinit();
        self.ctx.deinit();
        allocator.destroy(self);
    }

    /// Compile with an explicit pass set, run once, and return the summed output.
    fn runWith(self: *SharedProj, passes: api.OptPolicy, out: *[M * N]f32) !void {
        var model = try self.ctx.compile(&self.bld, &[_]api.TensorRef{self.out}, .{ .passes = passes });
        defer model.deinit();
        try model.bindInput("x", self.x);
        try model.run();
        const t = try model.outputTensorAt(0);
        try t.read(out);
    }
};

// Folding makes the fused weight the canonical store and frees the sources, which is only
// sound while every program reads the fused weight. A model on the same store that did
// NOT fuse names the sources directly, so their bytes are materialized back out — the
// alternative is a program reading released memory.
test "api: a weight one model fused away is still readable by a model that did not" {
    const allocator: std.mem.Allocator = std.testing.allocator;
    const P = SharedProj;
    const self = try P.init(allocator);
    defer self.deinit(allocator);

    var reference: [P.M * P.N]f32 = undefined;
    try self.runWith(.empty, &reference);

    var fused: [P.M * P.N]f32 = undefined;
    try self.runWith(.initOne(.horizontal_matmul), &fused);
    try std.testing.expectEqualSlices(f32, &reference, &fused);

    // The fusing model is gone, so nothing reads the fused weight and nothing reads the
    // sources either — this is exactly when reclaim frees them. A fresh unfused compile
    // has to get them back.
    var after: [P.M * P.N]f32 = undefined;
    try self.runWith(.empty, &after);
    try std.testing.expectEqualSlices(f32, &reference, &after);
}

// The reverse order is the one a per-model scan gets wrong: the unfused model is still
// alive when the fusing one reclaims, and its program still names the sources.
test "api: reclaim leaves a weight another live model still reads" {
    const allocator: std.mem.Allocator = std.testing.allocator;
    const P = SharedProj;
    const self = try P.init(allocator);
    defer self.deinit(allocator);

    var plain = try self.ctx.compile(&self.bld, &[_]api.TensorRef{self.out}, .{ .passes = .empty });
    defer plain.deinit();
    try plain.bindInput("x", self.x);
    try plain.run();
    var reference: [P.M * P.N]f32 = undefined;
    {
        const t = try plain.outputTensorAt(0);
        try t.read(&reference);
    }

    // Compiling and running a fusing model folds the same weights and triggers reclaim.
    var fused: [P.M * P.N]f32 = undefined;
    try self.runWith(.initOne(.horizontal_matmul), &fused);
    try std.testing.expectEqualSlices(f32, &reference, &fused);

    // `plain` never recompiled: it reads the same weights it was compiled against.
    try plain.run();
    var again: [P.M * P.N]f32 = undefined;
    {
        const t = try plain.outputTensorAt(0);
        try t.read(&again);
    }
    try std.testing.expectEqualSlices(f32, &reference, &again);
}

// A parameter is addressed by its graph value on both paths now, so the weight-swap API
// is no longer package-only: an in-process compiled model gets it too, and the swap
// reaches programs that were already compiled.
test "api: a compiled model's weights can be swapped in place" {
    const allocator: std.mem.Allocator = std.testing.allocator;
    const P = SharedProj;
    const self = try P.init(allocator);
    defer self.deinit(allocator);

    var model = try self.ctx.compile(&self.bld, &[_]api.TensorRef{self.out}, .{});
    defer model.deinit();
    try model.bindInput("x", self.x);

    var reference: [P.M * P.N]f32 = undefined;
    try model.run();
    {
        const t = try model.outputTensorAt(0);
        try t.read(&reference);
    }

    // Point "q" at k's weight. Same layout, so this is a byte copy into q's tensor.
    const k_tensor = try model.initializerTensorByValue(self.k.value);
    try model.overwriteInitializerByValue(self.q.value, k_tensor);

    var swapped: [P.M * P.N]f32 = undefined;
    try model.run();
    {
        const t = try model.outputTensorAt(0);
        try t.read(&swapped);
    }
    // Observable: q contributed something, and it is gone.
    try std.testing.expect(!std.mem.eql(f32, &reference, &swapped));

    // Swapping back restores it, with no recompile in between.
    try model.overwriteInitializerByValue(self.q.value, self.q_backup);
    var restored: [P.M * P.N]f32 = undefined;
    try model.run();
    {
        const t = try model.outputTensorAt(0);
        try t.read(&restored);
    }
    try std.testing.expectEqualSlices(f32, &reference, &restored);
}
