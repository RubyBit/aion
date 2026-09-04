// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Tests for the `api.nn` layer catalog.
//!
//! Each layer is checked against a hand-computed reference rather than against
//! another Aion path, so a wrong lowering cannot agree with itself.
const std = @import("std");

const api = @import("api.zig");
const types = @import("../backend/types.zig");

const nn = api.nn;
const TensorRef = api.TensorRef;

/// Context + Builder pair. Heap-allocated because the Builder borrows a pointer to
/// the Context, so neither may move after construction.
const Fixture = struct {
    ctx: api.Context,
    bld: api.Builder,

    fn init(allocator: std.mem.Allocator) !*Fixture {
        const self: *Fixture = try allocator.create(Fixture);
        self.ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
        self.bld = api.Builder.init(&self.ctx);
        return self;
    }

    fn deinit(self: *Fixture, allocator: std.mem.Allocator) void {
        self.bld.deinit();
        self.ctx.deinit();
        allocator.destroy(self);
    }
};

/// Compile `out`, run once, and read the result into `dst`.
fn runOnce(fx: *Fixture, out: TensorRef, dst: []f32) !void {
    var model = try fx.ctx.compile(&fx.bld, &[_]TensorRef{out}, .{});
    defer model.deinit();
    const t: api.Tensor = try model.runOutputTensor(0);
    try t.read(dst);
}

// ---------------------------------------------------------------- norms -----

test "api.nn: RMSNorm with gamma only synthesizes a zero beta" {
    const allocator = std.testing.allocator;
    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    const d: usize = 4;
    const x_vals = [_]f32{ 1.0, -2.0, 3.0, -4.0 };
    const g_vals = [_]f32{ 2.0, 0.5, 1.0, -1.0 };

    const X = try fx.bld.param(try fx.ctx.fromF32(&[_]usize{ 1, d }, &x_vals));
    // No `bias` entry, so the source simply does not have one.
    const norm = try nn.RMSNorm.bind(&fx.bld, .{
        .weight = try fx.ctx.fromF32(&[_]usize{d}, &g_vals),
    }, .{});
    const Y = try norm.forward(&fx.bld, X);

    var got: [d]f32 = undefined;
    try runOnce(fx, Y, &got);

    // rms = sqrt(mean(x^2) + eps); y = x/rms * gamma + 0
    var msq: f32 = 0.0;
    for (x_vals) |v| msq += v * v;
    msq /= @floatFromInt(d);
    const inv: f32 = 1.0 / @sqrt(msq + 1e-6);
    for (x_vals, 0..) |v, i| {
        try std.testing.expectApproxEqAbs(v * inv * g_vals[i], got[i], 1e-5);
    }
}

test "api.nn: parameterless RMSNorm normalizes without scaling" {
    const allocator = std.testing.allocator;
    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    const d: usize = 4;
    const x_vals = [_]f32{ 3.0, 4.0, 0.0, 0.0 };

    const X = try fx.bld.param(try fx.ctx.fromF32(&[_]usize{ 1, d }, &x_vals));
    // An empty source: neither gamma nor beta, so identity scale and shift are
    // synthesized and cached by the Builder.
    const norm = try nn.RMSNorm.bind(&fx.bld, .{}, .{ .normalized_shape = &[_]usize{d} });
    const Y = try norm.forward(&fx.bld, X);

    var got: [d]f32 = undefined;
    try runOnce(fx, Y, &got);

    const msq: f32 = (9.0 + 16.0) / 4.0;
    const inv: f32 = 1.0 / @sqrt(msq + 1e-6);
    for (x_vals, 0..) |v, i| try std.testing.expectApproxEqAbs(v * inv, got[i], 1e-5);
}

test "api.nn: RMSNorm without an explicit shape or gamma is rejected" {
    const allocator = std.testing.allocator;
    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    // Nothing determines the normalized width, so this cannot be guessed.
    try std.testing.expectError(
        error.InvalidArgument,
        nn.RMSNorm.bind(&fx.bld, .{}, .{}),
    );
}

test "api.nn: LayerNorm applies gamma and beta" {
    const allocator = std.testing.allocator;
    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    const d: usize = 4;
    const x_vals = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const g_vals = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const b_vals = [_]f32{ 0.5, 0.5, 0.5, 0.5 };

    const X = try fx.bld.param(try fx.ctx.fromF32(&[_]usize{ 1, d }, &x_vals));
    const norm = try nn.LayerNorm.bind(&fx.bld, .{
        .weight = try fx.ctx.fromF32(&[_]usize{d}, &g_vals),
        .bias = try fx.ctx.fromF32(&[_]usize{d}, &b_vals),
    }, .{});
    const Y = try norm.forward(&fx.bld, X);

    var got: [d]f32 = undefined;
    try runOnce(fx, Y, &got);

    const mean: f32 = 2.5;
    var variance: f32 = 0.0;
    for (x_vals) |v| variance += (v - mean) * (v - mean);
    variance /= @floatFromInt(d);
    const inv: f32 = 1.0 / @sqrt(variance + 1e-5);
    for (x_vals, 0..) |v, i| {
        try std.testing.expectApproxEqAbs((v - mean) * inv + 0.5, got[i], 1e-4);
    }
}

test "api.nn: the Builder shares one identity constant across same-width norms" {
    const allocator = std.testing.allocator;
    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    const d: usize = 8;
    const g = try fx.ctx.fromF32(&[_]usize{d}, &@as([d]f32, @splat(1.0)));

    const n1 = try nn.RMSNorm.bind(&fx.bld, .{ .weight = g }, .{});
    const n2 = try nn.RMSNorm.bind(&fx.bld, .{ .weight = g }, .{});
    // A different width, so a different constant. (Gamma omitted; the width-8
    // gamma here would be a genuine shape error.)
    const n3 = try nn.RMSNorm.bind(&fx.bld, .{}, .{ .normalized_shape = &[_]usize{4} });

    // Same width -> the very same zero-beta value, so a 35-layer model pays for one.
    try std.testing.expectEqual(n1.beta.value, n2.beta.value);
    // Different width -> its own constant.
    try std.testing.expect(n1.beta.value != n3.beta.value);

    try std.testing.expectEqualStrings("const.zeros.8", fx.bld.valueName(n1.beta).?);
    try std.testing.expectEqualStrings("const.zeros.4", fx.bld.valueName(n3.beta).?);
}

// ------------------------------------------------------- linear / embed -----

test "api.nn: Linear alpha folds the output scale into the matmul" {
    const allocator = std.testing.allocator;
    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    const x_vals = [_]f32{ 1.0, 2.0 };
    const w_vals = [_]f32{ 1.0, 0.0, 0.0, 1.0 }; // identity [2,2]

    const X = try fx.bld.param(try fx.ctx.fromF32(&[_]usize{ 1, 2 }, &x_vals));
    const lin = try nn.Linear.bind(&fx.bld, .{
        .weight = try fx.ctx.fromF32(&[_]usize{ 2, 2 }, &w_vals),
    }, .{ .alpha = 3.0 });
    const Y = try lin.forward(&fx.bld, X);

    var got: [2]f32 = undefined;
    try runOnce(fx, Y, &got);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), got[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), got[1], 1e-6);
}

test "api.nn: Embedding lookup and a tied output head share one table" {
    const allocator = std.testing.allocator;
    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    const vocab: usize = 3;
    const dim: usize = 2;
    const table = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }; // rows: [1,2] [3,4] [5,6]

    const emb = try nn.Embedding.bind(&fx.bld, .{
        .weight = try fx.ctx.fromF32(&[_]usize{ vocab, dim }, &table),
    }, .{});
    const ids = try fx.bld.param(try fx.ctx.from(&[_]usize{ 1, 2 }, &[_]i32{ 2, 0 }));
    const Looked = try emb.forward(&fx.bld, ids);

    var got: [4]f32 = undefined;
    try runOnce(fx, Looked, &got);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), got[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), got[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), got[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), got[3], 1e-6);

    // A tied head reuses the very same parameter (no second copy of the table).
    const head = try nn.Linear.bindShared(&fx.bld, emb.weightRef(), null, .{ .nt = true, .name = "lm_head" });
    try std.testing.expectEqual(emb.weightRef().value, head.w.value);
}

// --------------------------------------------------------------- blocks -----

test "api.nn: GatedMLP gelu path computes gelu(gate) * up" {
    const allocator = std.testing.allocator;
    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    // in=2, ffn=2. gate = identity, up = identity, down = identity.
    const eye = try fx.ctx.fromF32(&[_]usize{ 2, 2 }, &[_]f32{ 1.0, 0.0, 0.0, 1.0 });
    const x_vals = [_]f32{ 1.0, -1.0 };

    const X = try fx.bld.param(try fx.ctx.fromF32(&[_]usize{ 1, 2 }, &x_vals));
    const mlp = try nn.GatedMLP.bind(&fx.bld, .{
        .gate_proj = .{ .weight = eye },
        .up_proj = .{ .weight = eye },
        .down_proj = .{ .weight = eye },
    }, .{ .act = .gelu });
    const Y = try mlp.forward(&fx.bld, X);

    var got: [2]f32 = undefined;
    try runOnce(fx, Y, &got);

    // gelu(x) * x, elementwise, with identity projections.
    for (x_vals, 0..) |v, i| {
        const cdf: f32 = 0.5 * (1.0 + std.math.tanh(@as(f32, 0.7978845608) * (v + 0.044715 * v * v * v)));
        try std.testing.expectApproxEqAbs(v * cdf * v, got[i], 2e-3);
    }
}

test "api.nn: GatedMLP silu path multiplies the two projections" {
    const allocator = std.testing.allocator;
    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    // in=1, ffn=2. Concatenating gate and up into one wide weight is a fusion the
    // compiler performs (`opt/horizontal_matmul`), so this layer only ever
    // describes the two projections and has one code path.
    const gate = [_]f32{ 1.0, 2.0 }; // [1, 2]
    const up = [_]f32{ 10.0, 20.0 }; // [1, 2]
    const down = [_]f32{ 1.0, 1.0 }; // [2, 1] sums the two ffn lanes
    const x_vals = [_]f32{1.0};

    const X = try fx.bld.param(try fx.ctx.fromF32(&[_]usize{ 1, 1 }, &x_vals));
    const mlp = try nn.GatedMLP.bind(&fx.bld, .{
        .gate_proj = .{ .weight = try fx.ctx.fromF32(&[_]usize{ 1, 2 }, &gate) },
        .up_proj = .{ .weight = try fx.ctx.fromF32(&[_]usize{ 1, 2 }, &up) },
        .down_proj = .{ .weight = try fx.ctx.fromF32(&[_]usize{ 2, 1 }, &down) },
    }, .{ .act = .silu });
    const Y = try mlp.forward(&fx.bld, X);

    var got: [1]f32 = undefined;
    try runOnce(fx, Y, &got);

    // silu(gate) * up, summed: silu(1)*10 + silu(2)*20
    const s1: f32 = 1.0 / (1.0 + @exp(@as(f32, -1.0)));
    const s2: f32 = 2.0 / (1.0 + @exp(@as(f32, -2.0)));
    try std.testing.expectApproxEqAbs(s1 * 10.0 + s2 * 20.0, got[0], 1e-3);
}

test "api.nn: a q8_0 GatedMLP runs against a rank-3 activation" {
    // The shape a real transformer has: a `[batch, seq, dim]` residual stream and
    // quantized `[1, K, N]` projections. Weights are stored rank-aligned with the
    // activation, which is also what keeps them eligible for
    // `opt/horizontal_matmul` (it only fuses *externally bound* weights, and
    // rank-padding a weight would replace it with an Unsqueeze result).
    const allocator = std.testing.allocator;
    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    const in: usize = 64;
    const ffn: usize = 32;

    var gate_vals: [in * ffn]f32 = undefined;
    var up_vals: [in * ffn]f32 = undefined;
    for (&gate_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) * 0.05;
    for (&up_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) * 0.05;
    var down_vals: [ffn * in]f32 = undefined;
    for (&down_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) * 0.05;

    const X = try fx.bld.name(try fx.bld.input(.f32, &[_]usize{ 1, 4, in }), "x");
    const mlp = try nn.GatedMLP.bind(&fx.bld, .{
        .gate_proj = .{ .weight = try fx.ctx.fromF32Quantized(.q8_0, &[_]usize{ 1, in, ffn }, 1, &gate_vals) },
        .up_proj = .{ .weight = try fx.ctx.fromF32Quantized(.q8_0, &[_]usize{ 1, in, ffn }, 1, &up_vals) },
        .down_proj = .{ .weight = try fx.ctx.fromF32Quantized(.q8_0, &[_]usize{ 1, ffn, in }, 1, &down_vals) },
    }, .{ .act = .silu });
    const Y = try mlp.forward(&fx.bld, X);

    var model = try fx.ctx.compile(&fx.bld, &[_]TensorRef{Y}, .{});
    defer model.deinit();

    var x_vals: [4 * in]f32 = undefined;
    for (&x_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) * 0.1;
    try model.bindInput("x", try fx.ctx.fromF32(&[_]usize{ 1, 4, in }, &x_vals));
    try model.run();

    var got: [4 * in]f32 = undefined;
    try (try model.outputTensorAt(0)).read(&got);
    for (got) |v| try std.testing.expect(std.math.isFinite(v));
}

test "api.nn: fusing a split GatedMLP reclaims the weights it replaced" {
    // The reason `GatedMLP` describes gate and up as two projections rather than one
    // pre-concatenated weight: the compiler fuses them, and the model layer then
    // frees the originals — so split storage costs exactly what pre-fused storage
    // would. Without the reclamation, fusion would *add* a full copy of every fused
    // weight, which on a multi-GB model is hundreds of MB.
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const file = try tmp.dir.createFile(std.testing.io, "fused_mlp.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);

    const in: usize = 64;
    const ffn: usize = 32;

    var gate_vals: [in * ffn]f32 = undefined;
    var up_vals: [in * ffn]f32 = undefined;
    for (&gate_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) * 0.05;
    for (&up_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) * 0.05;
    var down_vals: [ffn * in]f32 = undefined;
    for (&down_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) * 0.05;

    var x_vals: [4 * in]f32 = undefined;
    for (&x_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) * 0.1;

    {
        const fx = try Fixture.init(allocator);
        defer fx.deinit(allocator);

        const X = try fx.bld.name(try fx.bld.input(.f32, &[_]usize{ 1, 4, in }), "x");
        const mlp = try nn.GatedMLP.bind(&fx.bld, .{
            .gate_proj = .{ .weight = try fx.ctx.fromF32Quantized(.q8_0, &[_]usize{ in, ffn }, 0, &gate_vals) },
            .up_proj = .{ .weight = try fx.ctx.fromF32Quantized(.q8_0, &[_]usize{ in, ffn }, 0, &up_vals) },
            .down_proj = .{ .weight = try fx.ctx.fromF32Quantized(.q8_0, &[_]usize{ ffn, in }, 0, &down_vals) },
        }, .{ .name = "mlp" });
        const Y = try fx.bld.name(try mlp.forward(&fx.bld, X), "y");
        try fx.ctx.exportModel(file, &fx.bld, &[_]api.NamedTensorRef{.{ .name = "y", .tensor = Y }}, .{});
    }

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    var model = try ctx.loadModel(file, .{});
    defer model.deinit();

    try model.bindInput("x", try ctx.fromF32(&[_]usize{ 1, 4, in }, &x_vals));
    try model.run();

    var before: [4 * in]f32 = undefined;
    try (try model.outputTensor("y")).read(&before);

    // The fused-away weight is still addressable by name: reading it gathers the
    // columns back out of the fused tensor, since its own buffer is gone.
    const probe: api.Tensor = try ctx.tensor(.q8_0, &[_]usize{ in, ffn });
    try model.readInitializerByDebugName("mlp/gate_proj/weight", probe);

    // And swapping it writes through to the fused tensor's sub-region. Zeroing the
    // gate must change the output, or the write went somewhere unread.
    const zeros: api.Tensor = try ctx.fromF32Quantized(.q8_0, &[_]usize{ in, ffn }, 0, &@as([in * ffn]f32, @splat(0.0)));
    try model.overwriteInitializerByDebugName("mlp/gate_proj/weight", zeros);
    try model.run();

    var after: [4 * in]f32 = undefined;
    try (try model.outputTensor("y")).read(&after);

    // silu(0) * up == 0, so the whole block collapses to zero.
    for (after) |v| try std.testing.expectApproxEqAbs(@as(f32, 0.0), v, 1e-6);
    var changed: bool = false;
    for (before) |v| {
        if (@abs(v) > 1e-6) changed = true;
    }
    try std.testing.expect(changed);
}

test "api.nn: a symbolic-shape model loads with a 2-D quantized weight" {
    // The shape a real decoder has: a free `seq` axis, so the loader instantiates a
    // concrete graph per shape from the package's shape terms rather than reusing an
    // authored one. A quantized weight is stored at its natural `[K, N]` rank and has
    // to survive that round trip.
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const file = try tmp.dir.createFile(std.testing.io, "sym_q8.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);

    const k: usize = 64;
    const n: usize = 32;
    const vocab: usize = 128;

    var w_vals: [k * n]f32 = undefined;
    for (&w_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) * 0.05;

    {
        const fx = try Fixture.init(allocator);
        defer fx.deinit(allocator);

        // Authoring placeholder of 1 on the free axis; the loader supplies the real one.
        const Tokens = try fx.bld.name(try fx.bld.input(.i32, &[_]usize{ 1, 1 }), "tokens");
        try fx.bld.symbolicDim(Tokens, 1, "seq");

        // The activation is produced by a lookup, so its shape reaches the package as
        // symbolic terms rather than as a public input's bound shape.
        var tab_vals: [vocab * k]f32 = undefined;
        for (&tab_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) * 0.1;
        const emb = try nn.Embedding.bind(&fx.bld, .{
            .weight = try fx.ctx.fromF32Quantized(.q8_0, &[_]usize{ vocab, k }, 1, &tab_vals),
        }, .{ .name = "embed" });
        const X = try emb.forward(&fx.bld, Tokens);

        // Exactly one projection: nothing for horizontal fusion to group, so the
        // quantized weight has to be tiled for the matmul as it was stored.
        const fc = try nn.Linear.bind(&fx.bld, .{
            .weight = try fx.ctx.fromF32Quantized(.q8_0, &[_]usize{ k, n }, 0, &w_vals),
        }, .{ .name = "fc" });
        const Y = try fx.bld.name(try fc.forward(&fx.bld, X), "y");

        try fx.ctx.exportModel(file, &fx.bld, &[_]api.NamedTensorRef{.{ .name = "y", .tensor = Y }}, .{
            .input_roles = &[_]api.InputRoleDecl{.{ .input = Tokens, .kind = .tokens, .axis = 1 }},
        });
    }

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    var model = try ctx.loadModel(file, .{});
    defer model.deinit();

    const seq: usize = 3;
    try model.bindInput("tokens", try ctx.from(&[_]usize{ 1, seq }, &[_]i32{ 1, 5, 9 }));
    try model.run();

    var got: [seq * n]f32 = undefined;
    try (try model.outputTensor("y")).read(&got);
    for (got) |v| try std.testing.expect(std.math.isFinite(v));
}

test "api.nn: a free axis survives a layer's reshape" {
    // A layer that reshapes must not freeze an axis the caller declared free. This is
    // the case that broke the Gemma port: `project` split the head dim using the
    // *concrete* dims it read back, so `seq` was pinned to its authoring placeholder
    // of 1 and the model could only ever run one token at a time.
    //
    // The declaration happens once, on the input. Nothing tells the layer about it.
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const file = try tmp.dir.createFile(std.testing.io, "free_axis.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);

    const heads: usize = 2;
    const head_dim: usize = 4;
    const model_dim: usize = heads * head_dim;
    const vocab: usize = 16;

    var tab_vals: [vocab * model_dim]f32 = undefined;
    for (&tab_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) * 0.1;
    var proj_vals: [model_dim * model_dim]f32 = undefined;
    for (&proj_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 5)) - 2)) * 0.2;

    {
        const fx = try Fixture.init(allocator);
        defer fx.deinit(allocator);

        const Tokens = try fx.bld.name(try fx.bld.input(.i32, &[_]usize{ 1, 1 }), "tokens");
        try fx.bld.symbolicDim(Tokens, 1, "seq");

        const emb = try nn.Embedding.bind(&fx.bld, .{
            .weight = try fx.ctx.fromF32(&[_]usize{ vocab, model_dim }, &tab_vals),
        }, .{ .name = "embed" });
        const X = try emb.forward(&fx.bld, Tokens);

        // The symbol reached a value two ops downstream of the input purely by
        // propagation — a gather and then whatever the layer does.
        try std.testing.expectEqualStrings("seq", (try fx.bld.dimAt(X, 1)).symbol);

        // A reader layer: no K/V projections, so `project` returns Q alone.
        const attn = try nn.Attention.bind(&fx.bld, .{
            .q_proj = .{ .weight = try fx.ctx.fromF32(&[_]usize{ model_dim, model_dim }, &proj_vals) },
            .o_proj = .{ .weight = try fx.ctx.fromF32(&[_]usize{ model_dim, model_dim }, &proj_vals) },
        }, .{ .heads = heads, .kv_heads = 1, .head_dim = head_dim, .scale = 1.0 }, .{ .name = "self_attn" });

        const projected = try attn.project(&fx.bld, X);
        try std.testing.expect(projected.k == null);
        try std.testing.expect(projected.v == null);

        // The point of the whole exercise: axis 1 of the reshaped Q is still free.
        try std.testing.expectEqualStrings("seq", (try fx.bld.dimAt(projected.q, 1)).symbol);

        const Y = try fx.bld.name(projected.q, "q");
        try fx.ctx.exportModel(file, &fx.bld, &[_]api.NamedTensorRef{.{ .name = "q", .tensor = Y }}, .{
            .input_roles = &[_]api.InputRoleDecl{.{ .input = Tokens, .kind = .tokens, .axis = 1 }},
        });
    }

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();
    var model = try ctx.loadModel(file, .{});
    defer model.deinit();

    // One compile, two sequence lengths. Before the symbol was propagated the second
    // of these failed with `ShapeMismatch` inside the reshape.
    for ([_]usize{ 1, 5 }) |seq| {
        var ids: [5]i32 = undefined;
        for (0..seq) |i| ids[i] = @intCast(i + 1);
        try model.bindInput("tokens", try ctx.from(&[_]usize{ 1, seq }, ids[0..seq]));
        try model.run();

        const out = try model.outputTensor("q");
        var got: [5 * heads * head_dim]f32 = undefined;
        const want: usize = seq * heads * head_dim;
        try std.testing.expectEqual(want, out.elemCount());
        try (out).read(got[0..want]);
        for (got[0..want]) |v| try std.testing.expect(std.math.isFinite(v));
    }
}

// The same case, compiled in process instead of round-tripped through a file.
//
// `ctx.compile` used to carry only symbolic INPUT axes: a reshape's target sizes went in
// as the authoring placeholders, so this model could only ever run at seq 1 — exactly the
// bug the loaded path above exists to catch, still live on the other path. Both paths
// snapshot the same template now, so the reshape's free axis is symbolic either way.
test "api.nn: a free axis survives a reshape in a compiled model" {
    const allocator = std.testing.allocator;
    const heads: usize = 2;
    const head_dim: usize = 4;
    const model_dim: usize = heads * head_dim;
    const vocab: usize = 16;

    var tab_vals: [vocab * model_dim]f32 = undefined;
    for (&tab_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) * 0.1;
    var proj_vals: [model_dim * model_dim]f32 = undefined;
    for (&proj_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 5)) - 2)) * 0.2;

    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    const Tokens = try fx.bld.name(try fx.bld.input(.i32, &[_]usize{ 1, 1 }), "tokens");
    try fx.bld.symbolicDim(Tokens, 1, "seq");

    const emb = try nn.Embedding.bind(&fx.bld, .{
        .weight = try fx.ctx.fromF32(&[_]usize{ vocab, model_dim }, &tab_vals),
    }, .{ .name = "embed" });
    const X = try emb.forward(&fx.bld, Tokens);

    const attn = try nn.Attention.bind(&fx.bld, .{
        .q_proj = .{ .weight = try fx.ctx.fromF32(&[_]usize{ model_dim, model_dim }, &proj_vals) },
        .o_proj = .{ .weight = try fx.ctx.fromF32(&[_]usize{ model_dim, model_dim }, &proj_vals) },
    }, .{ .heads = heads, .kv_heads = 1, .head_dim = head_dim, .scale = 1.0 }, .{ .name = "self_attn" });

    const projected = try attn.project(&fx.bld, X);
    const Y = try fx.bld.name(projected.q, "q");

    var model = try fx.ctx.compile(&fx.bld, &[_]TensorRef{Y}, .{});
    defer model.deinit();

    // One compile, two sequence lengths.
    for ([_]usize{ 1, 5 }) |seq| {
        var ids: [5]i32 = undefined;
        for (0..seq) |i| ids[i] = @intCast(i + 1);
        try model.bindInput("tokens", try fx.ctx.from(&[_]usize{ 1, seq }, ids[0..seq]));
        try model.run();

        const out = try model.outputTensor("q");
        var got: [5 * heads * head_dim]f32 = undefined;
        const want: usize = seq * heads * head_dim;
        try std.testing.expectEqual(want, out.elemCount());
        try out.read(got[0..want]);
        for (got[0..want]) |v| try std.testing.expect(std.math.isFinite(v));
    }
}

// A parameter is addressable by the name its author gave it on both paths: a package
// persists debug names and a Builder has them in hand. This was package-only before —
// `debugNames` returned an empty slice for a compiled model.
test "api.nn: a compiled model's weights are addressable by debug name" {
    const allocator = std.testing.allocator;
    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    const w = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const X = try fx.bld.name(try fx.bld.input(.f32, &[_]usize{ 1, 2 }), "x");
    const lin = try nn.Linear.bind(&fx.bld, .{
        .weight = try fx.ctx.fromF32(&[_]usize{ 2, 2 }, &w),
    }, .{ .name = "proj" });
    const Y = try lin.forward(&fx.bld, X);

    var model = try fx.ctx.compile(&fx.bld, &[_]TensorRef{Y}, .{});
    defer model.deinit();
    try std.testing.expect(model.debugNames().len != 0);

    // The name the layer gave it: scopes join with "/".
    try std.testing.expectEqualStrings("proj/weight", fx.bld.valueName(lin.w).?);
    const weight = try model.initializerTensorByDebugName("proj/weight");
    var got: [4]f32 = undefined;
    try weight.read(&got);
    try std.testing.expectEqualSlices(f32, &w, &got);
}

test "api.nn: GLU gates one half of a projection with the other" {
    const allocator = std.testing.allocator;
    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    // in=1, out=4 -> halves of width 2.
    const w = [_]f32{ 1.0, 2.0, 0.0, 100.0 };
    const X = try fx.bld.param(try fx.ctx.fromF32(&[_]usize{ 1, 1 }, &[_]f32{1.0}));
    const glu = try nn.GLU.bind(&fx.bld, .{
        .proj = .{ .weight = try fx.ctx.fromF32(&[_]usize{ 1, 4 }, &w) },
    }, .{});
    const Y = try glu.forward(&fx.bld, X);

    var got: [2]f32 = undefined;
    try runOnce(fx, Y, &got);

    // a = [1, 2], b = [0, 100] -> a * sigmoid(b)
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), got[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), got[1], 1e-5);
}

test "api.nn: GatedMLP rejects gate and up projections of different widths" {
    const allocator = std.testing.allocator;
    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    // The halves are multiplied elementwise, so unequal ffn widths cannot work.
    try std.testing.expectError(error.InvalidArgument, nn.GatedMLP.bind(&fx.bld, .{
        .gate_proj = .{ .weight = try fx.ctx.fromF32(&[_]usize{ 1, 3 }, &[_]f32{ 1.0, 2.0, 3.0 }) },
        .up_proj = .{ .weight = try fx.ctx.fromF32(&[_]usize{ 1, 2 }, &[_]f32{ 1.0, 2.0 }) },
        .down_proj = .{ .weight = try fx.ctx.fromF32(&[_]usize{ 1, 1 }, &[_]f32{1.0}) },
    }, .{}));
}

// ----------------------------------------------------------- containers -----

test "api.nn: Sequential chains layers and Residual scales the branch" {
    const allocator = std.testing.allocator;
    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    const eye = try fx.ctx.fromF32(&[_]usize{ 2, 2 }, &[_]f32{ 1.0, 0.0, 0.0, 1.0 });
    const x_vals = [_]f32{ 1.0, -2.0 };

    const Stack = nn.Sequential(struct {
        fc1: nn.Linear,
        act: nn.Activation,
        fc2: nn.Linear,
    });

    const X = try fx.bld.param(try fx.ctx.fromF32(&[_]usize{ 1, 2 }, &x_vals));
    const stack = Stack.init(.{
        .fc1 = try nn.Linear.bind(&fx.bld, .{ .weight = eye }, .{ .name = "fc1" }),
        .act = .{ .kind = .relu },
        .fc2 = try nn.Linear.bind(&fx.bld, .{ .weight = eye }, .{ .name = "fc2" }),
    });

    // x + 0.5 * relu(x), with identity projections.
    const block = try nn.Residual(Stack).bind(&fx.bld, stack, 0.5);
    const Y = try block.forward(&fx.bld, X);

    var got: [2]f32 = undefined;
    try runOnce(fx, Y, &got);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), got[0], 1e-6); // 1 + 0.5*1
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), got[1], 1e-6); // -2 + 0.5*0
}

// ---------------------------------------------------- constant helpers ------

test "api.nn: scale and softcap use one-element constants" {
    const allocator = std.testing.allocator;
    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    const x_vals = [_]f32{ 0.0, 30.0, -30.0, 1.0 };
    const X = try fx.bld.param(try fx.ctx.fromF32(&[_]usize{ 1, 4 }, &x_vals));

    const Scaled = try nn.scale(&fx.bld, X, 2.0);
    const Capped = try nn.softcap(&fx.bld, Scaled, 10.0);

    // A no-op factor must not add a node at all.
    try std.testing.expectEqual(X.value, (try nn.scale(&fx.bld, X, 1.0)).value);
    try std.testing.expectEqual(X.value, (try nn.shift(&fx.bld, X, 0.0)).value);
    // A disabled cap is likewise a pass-through.
    try std.testing.expectEqual(X.value, (try nn.softcap(&fx.bld, X, 0.0)).value);

    var got: [4]f32 = undefined;
    try runOnce(fx, Capped, &got);
    for (x_vals, 0..) |v, i| {
        const want: f32 = 10.0 * std.math.tanh(v * 2.0 / 10.0);
        try std.testing.expectApproxEqAbs(want, got[i], 1e-4);
    }
}

test "api.nn: identical scalars are bound once" {
    const allocator = std.testing.allocator;
    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    const a = try fx.bld.constant(0.5);
    const b = try fx.bld.constant(0.5);
    const c = try fx.bld.constant(0.25);

    try std.testing.expectEqual(a.value, b.value);
    try std.testing.expect(a.value != c.value);
    // -0.0 and 0.0 are distinct bit patterns and must not collide.
    try std.testing.expect((try fx.bld.constant(0.0)).value != (try fx.bld.constant(-0.0)).value);
}

// ------------------------------------------------------------ depthwise -----

test "api.nn: nn.depthwise derives groups and rejects a dense weight" {
    const allocator = std.testing.allocator;
    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    const channels: usize = 3;
    // [k=1, c_in/groups=1, c_out=3] scales each channel independently.
    const w = [_]f32{ 2.0, 3.0, 4.0 };
    const dw = try nn.Conv1D.bind(&fx.bld, .{
        .weight = try fx.ctx.fromF32(&[_]usize{ 1, 1, channels }, &w),
    }, nn.depthwise(.{}));
    // Derived from the weight, so "groups == channels" cannot be set wrong.
    try std.testing.expectEqual(channels, dw.opts.groups);

    const x_vals = [_]f32{ 1.0, 1.0, 1.0 };
    const X = try fx.bld.param(try fx.ctx.fromF32(&[_]usize{ 1, 1, channels }, &x_vals));
    const Y = try dw.forward(&fx.bld, X);

    var got: [3]f32 = undefined;
    try runOnce(fx, Y, &got);
    for (w, 0..) |wv, i| try std.testing.expectApproxEqAbs(wv, got[i], 1e-6);

    // c_in/groups != 1 is not depthwise.
    try std.testing.expectError(error.InvalidArgument, nn.Conv1D.bind(&fx.bld, .{
        .weight = try fx.ctx.fromF32(&[_]usize{ 1, 2, 2 }, &[_]f32{ 1, 2, 3, 4 }),
    }, nn.depthwise(.{})));
}

// --------------------------------------------------------------- naming -----

test "api.nn: parameter names are hierarchical and semantic" {
    const allocator = std.testing.allocator;
    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    const eye = try fx.ctx.fromF32(&[_]usize{ 2, 2 }, &[_]f32{ 1.0, 0.0, 0.0, 1.0 });
    const mlp = try nn.GatedMLP.bind(&fx.bld, .{
        .gate_proj = .{ .weight = eye },
        .up_proj = .{ .weight = eye },
        .down_proj = .{ .weight = eye },
    }, .{ .name = "mlp" });

    try std.testing.expectEqualStrings("mlp/gate_proj/weight", fx.bld.valueName(mlp.gate.w).?);
    try std.testing.expectEqualStrings("mlp/up_proj/weight", fx.bld.valueName(mlp.up.w).?);
    try std.testing.expectEqualStrings("mlp/down_proj/weight", fx.bld.valueName(mlp.down.w).?);

    // Unnamed layers auto-number per parent instead of colliding.
    const a = try nn.Linear.bind(&fx.bld, .{ .weight = eye }, .{});
    const b = try nn.Linear.bind(&fx.bld, .{ .weight = eye }, .{});
    try std.testing.expectEqualStrings("Linear#0/weight", fx.bld.valueName(a.w).?);
    try std.testing.expectEqualStrings("Linear#1/weight", fx.bld.valueName(b.w).?);

    // A layer's ops land under the same path as its parameters.
    const X = try fx.bld.input(.f32, &[_]usize{ 1, 2 });
    const Y = try a.forward(&fx.bld, X);
    try std.testing.expectEqualStrings("Linear#0/matmul#0", fx.bld.valueName(Y).?);
}

test "api.nn: a literal, a shared Named, and a Params bind identically" {
    // The three things a layer accepts as `params`. They differ only in who owns
    // the storage, so they must produce the same parameter — otherwise the
    // convenient spelling would be a subtly different one.
    const allocator = std.testing.allocator;
    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    const eye = try fx.ctx.fromF32(&[_]usize{ 2, 2 }, &[_]f32{ 1.0, 0.0, 0.0, 1.0 });
    const shared = nn.Source(nn.Linear.Weights).from(.{ .weight = eye });

    // Literal: storage lives on `bind`'s frame, nothing to keep alive here.
    const a = try nn.Linear.bind(&fx.bld, .{ .weight = eye }, .{ .name = "a" });
    // A source built once and reused across layers.
    const b = try nn.Linear.bind(&fx.bld, shared, .{ .name = "b" });
    // An explicit `Params`, as a composite hands to its children.
    const c = try nn.Linear.bind(&fx.bld, shared.at("c"), .{});

    try std.testing.expectEqualStrings("a/weight", fx.bld.valueName(a.w).?);
    try std.testing.expectEqualStrings("b/weight", fx.bld.valueName(b.w).?);
    try std.testing.expectEqualStrings("c/weight", fx.bld.valueName(c.w).?);
}

// ------------------------------------------------------------ attention -----

test "api.nn: KVCache + Attention decode two steps against real history" {
    // A one-head, one-dim attention step. With identity projections and scale 1,
    // attention over a growing cache reduces to a softmax-weighted average of the
    // values seen so far, which is checkable by hand.
    const allocator = std.testing.allocator;
    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    const capacity: usize = 4;

    // A tokens-role input is what makes the runtime auto-feed positions,
    // cache_write_index and cache_visible_end; without one they stay zero and
    // nothing in the cache is ever visible.
    const Tokens = try fx.bld.name(try fx.bld.input(.i32, &[_]usize{ 1, 1 }), "tokens");
    const emb = try nn.Embedding.bind(&fx.bld, .{
        .weight = try fx.ctx.fromF32(&[_]usize{ 4, 1 }, &[_]f32{ 1.0, 2.0, 3.0, 4.0 }),
    }, .{ .name = "embed" });
    const X = try emb.forward(&fx.bld, Tokens); // [1, 1, 1]

    const Positions = try fx.bld.name(try fx.bld.input(.i32, &[_]usize{ 1, 1 }), "positions");
    const WriteIdx = try fx.bld.name(try fx.bld.input(.i32, &[_]usize{1}), "cache_write_index");
    const VisibleEnd = try fx.bld.name(try fx.bld.input(.i32, &[_]usize{1}), "cache_visible_end");

    var k_cache = try nn.KVCache.bind(&fx.bld, 1, 1, 1, .{
        .name = "k_cache",
        .dtype = .f32,
        .capacity = capacity,
    });
    var v_cache = try nn.KVCache.bind(&fx.bld, 1, 1, 1, .{
        .name = "v_cache",
        .dtype = .f32,
        .capacity = capacity,
    });

    const eye = try fx.ctx.fromF32(&[_]usize{ 1, 1 }, &[_]f32{1.0}); // [1,1] identity
    const attn = try nn.Attention.bind(&fx.bld, .{
        .q_proj = .{ .weight = eye },
        .k_proj = .{ .weight = eye },
        .v_proj = .{ .weight = eye },
        .o_proj = .{ .weight = eye },
    }, .{
        .heads = 1,
        .kv_heads = 1,
        .head_dim = 1,
        .scale = 1.0,
    }, .{ .name = "attn" });

    const proj = try attn.project(&fx.bld, X);
    // RoPE on a 1-wide head is a no-op numerically, but it exercises the plumbing.
    const rope = nn.RoPE.init(.{ .base_frequency = 10000.0 });
    const q = try rope.forward(&fx.bld, proj.q, Positions);

    // This layer owns its cache, so `project` produced K/V for it to append.
    const k_now = try fx.bld.name(try k_cache.append(&fx.bld, proj.k.?, WriteIdx), "next_k");
    const v_now = try fx.bld.name(try v_cache.append(&fx.bld, proj.v.?, WriteIdx), "next_v");
    const Y = try fx.bld.name(try attn.attend(&fx.bld, q, k_now, v_now, Positions, VisibleEnd), "y");

    // The aliased cache outputs must be compiled outputs — an alias is resolved by
    // output index, so `KVCache.outputRef()` has to appear here.
    var model = try fx.ctx.compile(&fx.bld, &[_]TensorRef{ Y, k_cache.outputRef(), v_cache.outputRef() }, .{
        .output_aliases = &[_]api.OutputAlias{ k_cache.alias(), v_cache.alias() },
        .input_roles = &[_]api.InputRoleDecl{
            .{ .input = Tokens, .kind = .tokens, .axis = 1 },
            .{ .input = Positions, .kind = .positions, .axis = 1 },
            .{ .input = WriteIdx, .kind = .cache_write_index },
            .{ .input = VisibleEnd, .kind = .cache_visible_end },
            k_cache.roleDecl(),
            v_cache.roleDecl(),
        },
    });
    defer model.deinit();

    // Token t embeds to t+1, so tokens 1/2/3 feed values 2/3/4.
    const step = struct {
        fn run(m: *api.Model, c: *api.Context, token: i32) !f32 {
            const tok = try c.from(&[_]usize{ 1, 1 }, &[_]i32{token});
            try m.bindInput("tokens", tok);
            try m.run();
            var out: [1]f32 = undefined;
            try (try m.outputTensor("y")).read(&out);
            return out[0];
        }
    };

    // Step 1: only position 0 is visible, so the average is x0 itself.
    const y0 = try step.run(&model, &fx.ctx, 1);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), y0, 1e-5);

    // Step 2: q=3 attends over k=[2,3], v=[2,3] with logits [q*2, q*3] = [6, 9].
    const y1 = try step.run(&model, &fx.ctx, 2);
    // Softmax over [6, 9] weighting values [2, 3]. That step 2 sees k/v from step 1
    // at all is the proof the cache carried across runs.
    const w0: f32 = @exp(@as(f32, 6.0) - 9.0);
    const expected: f32 = (w0 * 2.0 + 1.0 * 3.0) / (w0 + 1.0);
    try std.testing.expectApproxEqAbs(expected, y1, 1e-4);

    // A third step must keep accumulating rather than overwrite slot 0.
    const y2 = try step.run(&model, &fx.ctx, 3);
    const e0: f32 = @exp(@as(f32, 4.0) * 2.0 - 16.0);
    const e1: f32 = @exp(@as(f32, 4.0) * 3.0 - 16.0);
    const want2: f32 = (e0 * 2.0 + e1 * 3.0 + 1.0 * 4.0) / (e0 + e1 + 1.0);
    try std.testing.expectApproxEqAbs(want2, y2, 1e-4);
}

test "api.nn: a reader layer attends over a cache another layer wrote" {
    // Gemma 4 shares one KV cache across a run of layers: the first projects K/V and
    // writes, the rest only project Q and attend over it. A reader layer therefore
    // has no K/V projections at all — omitting them from `Weights` is what says so.
    const allocator = std.testing.allocator;
    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    const eye = try fx.ctx.fromF32(&[_]usize{ 1, 1 }, &[_]f32{1.0});

    const writer = try nn.Attention.bind(&fx.bld, .{
        .q_proj = .{ .weight = eye },
        .k_proj = .{ .weight = eye },
        .v_proj = .{ .weight = eye },
        .o_proj = .{ .weight = eye },
    }, .{ .heads = 1, .kv_heads = 1, .head_dim = 1, .scale = 1.0 }, .{ .name = "w" });

    const reader = try nn.Attention.bind(&fx.bld, .{
        .q_proj = .{ .weight = eye },
        .o_proj = .{ .weight = eye },
    }, .{ .heads = 1, .kv_heads = 1, .head_dim = 1, .scale = 1.0 }, .{ .name = "r" });

    try std.testing.expect(writer.k_proj != null);
    try std.testing.expect(reader.k_proj == null);

    const X = try fx.bld.input(.f32, &[_]usize{ 1, 1, 1 });

    // The writer produces K/V to append; the reader produces Q only.
    const wp = try writer.project(&fx.bld, X);
    try std.testing.expect(wp.k != null and wp.v != null);

    const rp = try reader.project(&fx.bld, X);
    try std.testing.expect(rp.k == null and rp.v == null);

    // Both attend over the *writer's* cache — `attend` takes the caches as
    // arguments, which is why sharing needed no new plumbing.
    var k_cache = try nn.KVCache.bind(&fx.bld, 1, 1, 1, .{ .dtype = .f32, .capacity = 4 });
    var v_cache = try nn.KVCache.bind(&fx.bld, 1, 1, 1, .{ .dtype = .f32, .capacity = 4 });
    const idx = try fx.bld.input(.i32, &[_]usize{1});
    const k_now = try k_cache.append(&fx.bld, wp.k.?, idx);
    const v_now = try v_cache.append(&fx.bld, wp.v.?, idx);

    const positions = try fx.bld.input(.i32, &[_]usize{ 1, 1 });
    const visible = try fx.bld.input(.i32, &[_]usize{1});
    _ = try writer.attend(&fx.bld, wp.q, k_now, v_now, positions, visible);
    _ = try reader.attend(&fx.bld, rp.q, k_now, v_now, positions, visible);
}

test "api.nn: Attention rejects K without V" {
    const allocator = std.testing.allocator;
    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    const eye = try fx.ctx.fromF32(&[_]usize{ 1, 1 }, &[_]f32{1.0});
    // A layer that projects K but not V cannot fill a cache; catching it here beats
    // a confusing shape failure later.
    try std.testing.expectError(error.InvalidArgument, nn.Attention.bind(&fx.bld, .{
        .q_proj = .{ .weight = eye },
        .k_proj = .{ .weight = eye },
        .o_proj = .{ .weight = eye },
    }, .{ .heads = 1, .kv_heads = 1, .head_dim = 1, .scale = 1.0 }, .{}));
}

test "api.nn: Attention rejects head counts that are not a GQA multiple" {
    const allocator = std.testing.allocator;
    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    const eye = try fx.ctx.fromF32(&[_]usize{ 1, 1 }, &[_]f32{1.0});
    // 3 query heads cannot be split across 2 KV heads.
    try std.testing.expectError(error.InvalidArgument, nn.Attention.bind(&fx.bld, .{
        .q_proj = .{ .weight = eye },
        .k_proj = .{ .weight = eye },
        .v_proj = .{ .weight = eye },
        .o_proj = .{ .weight = eye },
    }, .{
        .heads = 3,
        .kv_heads = 2,
        .head_dim = 1,
        .scale = 1.0,
    }, .{}));
}

test "api.nn: KVCache append casts to the cache dtype" {
    const allocator = std.testing.allocator;
    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    var cache = try nn.KVCache.bind(&fx.bld, 1, 1, 4, .{ .dtype = .f16, .capacity = 8 });

    // `append` takes what `project` returns — `[batch, seq, kv_heads, head_dim]` —
    // and owns both halves of the difference from the store: it moves heads next to
    // batch and casts f32 down to the f16 cache, rather than silently mismatching.
    const idx = try fx.bld.input(.i32, &[_]usize{1});
    const kv = try fx.bld.input(.f32, &[_]usize{ 1, 1, 1, 4 });
    const appended = try cache.append(&fx.bld, kv, idx);
    try std.testing.expectEqual(types.DType.f16, fx.bld.dtypeOf(appended).?);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 8, 1, 4 }, fx.bld.knownShape(appended).?);

    // Cache layout passed in by mistake: `kv_heads` is not where it should be.
    var swapped = try nn.KVCache.bind(&fx.bld, 1, 1, 4, .{ .dtype = .f16, .capacity = 8 });
    const wrong = try fx.bld.input(.f32, &[_]usize{ 1, 4, 1, 1 });
    try std.testing.expectError(error.ShapeMismatch, swapped.append(&fx.bld, wrong, idx));

    // Multi-head GQA uses the same projection-native layout.
    var multi = try nn.KVCache.bind(&fx.bld, 1, 2, 4, .{ .dtype = .f16, .capacity = 8 });
    const kv2 = try fx.bld.input(.f32, &[_]usize{ 1, 3, 2, 4 });
    const appended2 = try multi.append(&fx.bld, kv2, idx);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 8, 2, 4 }, fx.bld.knownShape(appended2).?);

    // A capacity of zero has no valid interpretation.
    try std.testing.expectError(error.InvalidArgument, nn.KVCache.bind(&fx.bld, 1, 1, 1, .{ .capacity = 0 }));
}

// ------------------------------------------------- params / round-trip ------

test "api.nn: forEachParam walks a module tree and reports its debug names" {
    const allocator = std.testing.allocator;
    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    const eye = try fx.ctx.fromF32(&[_]usize{ 2, 2 }, &[_]f32{ 1, 0, 0, 1 });
    const bias = try fx.ctx.fromF32(&[_]usize{2}, &[_]f32{ 0.0, 0.0 });

    const Block = struct {
        attn: nn.Linear,
        norm: nn.RMSNorm,
        ffn: nn.GatedMLP,

        pub fn forward(self: @This(), bld: *api.Builder, x: TensorRef) api.Builder.Error!TensorRef {
            return self.ffn.forward(bld, try self.norm.forward(bld, try self.attn.forward(bld, x)));
        }
    };

    const block: Block = .{
        .attn = try nn.Linear.bind(&fx.bld, .{ .weight = eye, .bias = bias }, .{ .name = "attn" }),
        .norm = try nn.RMSNorm.bind(&fx.bld, .{ .weight = bias }, .{ .name = "norm" }),
        .ffn = try nn.GatedMLP.bind(&fx.bld, .{
            .gate_proj = .{ .weight = eye },
            .up_proj = .{ .weight = eye },
            .down_proj = .{ .weight = eye },
        }, .{ .name = "ffn" }),
    };

    var names: [16][]const u8 = undefined;
    const n = try nn.collectParamNames(Block, &block, &fx.bld, &names);

    // Real weights only. The norm's zero beta is a Builder-synthesized constant,
    // not model state, so it is not reported — and being shared, it would otherwise
    // show up once per layer pointing at it.
    const want = [_][]const u8{
        "attn/weight",
        "attn/bias",
        "norm/weight",
        "ffn/gate_proj/weight",
        "ffn/up_proj/weight",
        "ffn/down_proj/weight",
    };
    try std.testing.expectEqual(want.len, n);
    for (want, 0..) |w, i| try std.testing.expectEqualStrings(w, names[i]);
}

test "api.nn: export then load rebuilds the same model from its parameter names" {
    // The whole naming story end to end: layers write hierarchical names, the
    // package keeps them, and a `Package` source reads them back by the same path —
    // so a model can be rebuilt in a fresh context without a lookup table, and with
    // the very same `bind` calls.
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const file = try tmp.dir.createFile(std.testing.io, "nn_roundtrip.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);

    const x_vals = [_]f32{ 0.5, -1.5, 2.0, 0.25 };

    // --- author + export ---
    var want: [4]f32 = undefined;
    {
        const fx = try Fixture.init(allocator);
        defer fx.deinit(allocator);

        const w1 = try fx.ctx.fromF32(&[_]usize{ 4, 4 }, &[_]f32{
            1.0, 0.5,  -1.0, 0.25,
            0.0, 2.0,  0.5,  -1.0,
            0.5, -0.5, 1.0,  0.0,
            2.0, 1.0,  0.0,  0.5,
        });
        const b1 = try fx.ctx.fromF32(&[_]usize{4}, &[_]f32{ 0.1, -0.2, 0.3, 0.0 });
        const g = try fx.ctx.fromF32(&[_]usize{4}, &[_]f32{ 1.5, 1.0, 0.5, 2.0 });

        const X = try fx.bld.name(try fx.bld.input(.f32, &[_]usize{ 1, 4 }), "x");
        const fc = try nn.Linear.bind(&fx.bld, .{ .weight = w1, .bias = b1 }, .{ .name = "fc" });
        const norm = try nn.RMSNorm.bind(&fx.bld, .{ .weight = g }, .{ .name = "norm" });
        const Y = try fx.bld.name(try norm.forward(&fx.bld, try fc.forward(&fx.bld, X)), "y");

        try fx.ctx.exportModel(file, &fx.bld, &[_]api.NamedTensorRef{.{ .name = "y", .tensor = Y }}, .{});

        // Reference output from the authored graph.
        var model = try fx.ctx.compile(&fx.bld, &[_]TensorRef{Y}, .{});
        defer model.deinit();
        try model.bindInput("x", try fx.ctx.fromF32(&[_]usize{ 1, 4 }, &x_vals));
        try model.run();
        try (try model.outputTensor("y")).read(&want);
    }

    // --- load the weights back and rebuild the same architecture ---
    {
        const fx = try Fixture.init(allocator);
        defer fx.deinit(allocator);

        var weights = try fx.ctx.loadWeights(file, .{});
        defer weights.deinit();

        const pkg = nn.Package.init(&weights);

        const X = try fx.bld.name(try fx.bld.input(.f32, &[_]usize{ 1, 4 }), "x");
        // Identical calls to the authoring pass; only the source differs.
        const fc = try nn.Linear.bind(&fx.bld, pkg.at("fc"), .{});
        const norm = try nn.RMSNorm.bind(&fx.bld, pkg.at("norm"), .{});
        const Y = try fx.bld.name(try norm.forward(&fx.bld, try fc.forward(&fx.bld, X)), "y");

        // Nothing in the package is unaccounted for by the rebuilt architecture.
        try pkg.expectAllLoaded(&fx.bld);

        var model = try fx.ctx.compile(&fx.bld, &[_]TensorRef{Y}, .{});
        defer model.deinit();
        try model.bindInput("x", try fx.ctx.fromF32(&[_]usize{ 1, 4 }, &x_vals));
        try model.run();

        var got: [4]f32 = undefined;
        try (try model.outputTensor("y")).read(&got);
        for (want, 0..) |w, i| try std.testing.expectApproxEqAbs(w, got[i], 1e-6);
    }
}

test "api.nn: a Package reports weights no layer claimed" {
    // The failure this exists to catch: a renamed or forgotten layer leaves package
    // weights unread, and the model still builds and runs — quietly wrong.
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const file = try tmp.dir.createFile(std.testing.io, "nn_unclaimed.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);

    {
        const fx = try Fixture.init(allocator);
        defer fx.deinit(allocator);

        const w = try fx.ctx.fromF32(&[_]usize{ 2, 2 }, &[_]f32{ 1, 0, 0, 1 });
        const X = try fx.bld.name(try fx.bld.input(.f32, &[_]usize{ 1, 2 }), "x");
        const a = try nn.Linear.bind(&fx.bld, .{ .weight = w }, .{ .name = "fc1" });
        const b = try nn.Linear.bind(&fx.bld, .{ .weight = w }, .{ .name = "fc2" });
        const Y = try b.forward(&fx.bld, try a.forward(&fx.bld, X));
        try fx.ctx.exportModel(file, &fx.bld, &[_]api.NamedTensorRef{.{ .name = "y", .tensor = Y }}, .{});
    }

    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    var weights = try fx.ctx.loadWeights(file, .{});
    defer weights.deinit();

    const pkg = nn.Package.init(&weights);

    // Rebuild only the first layer, as if the second were renamed.
    _ = try nn.Linear.bind(&fx.bld, pkg.at("fc1"), .{});
    try std.testing.expectEqualStrings("fc2/weight", pkg.firstUnaccounted(&fx.bld).?);
    try std.testing.expectError(error.InvalidArgument, pkg.expectAllLoaded(&fx.bld));

    // Claiming it clears the report.
    _ = try nn.Linear.bind(&fx.bld, pkg.at("fc2"), .{});
    try std.testing.expect(pkg.firstUnaccounted(&fx.bld) == null);
    try pkg.expectAllLoaded(&fx.bld);
}

test "api.nn: a missing required parameter fails loudly" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const file = try tmp.dir.createFile(std.testing.io, "nn_missing.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);

    {
        const fx = try Fixture.init(allocator);
        defer fx.deinit(allocator);
        const w = try fx.ctx.fromF32(&[_]usize{ 2, 2 }, &[_]f32{ 1, 0, 0, 1 });
        const X = try fx.bld.name(try fx.bld.input(.f32, &[_]usize{ 1, 2 }), "x");
        const a = try nn.Linear.bind(&fx.bld, .{ .weight = w }, .{ .name = "fc1" });
        const Y = try a.forward(&fx.bld, X);
        try fx.ctx.exportModel(file, &fx.bld, &[_]api.NamedTensorRef{.{ .name = "y", .tensor = Y }}, .{});
    }

    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    var weights = try fx.ctx.loadWeights(file, .{});
    defer weights.deinit();

    const pkg = nn.Package.init(&weights);

    // A required weight under the wrong name is an error, not a silent zero.
    try std.testing.expectError(error.InvalidArgument, nn.Linear.bind(&fx.bld, pkg.at("typo"), .{}));
}

// -------------------------------------------------------- initializers ------

test "api.nn: causal and chunked masks have the documented shape" {
    const allocator = std.testing.allocator;
    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    const seq: usize = 4;

    {
        const m = try nn.causalMask(&fx.ctx, allocator, seq);
        var vals: [seq * seq]f32 = undefined;
        try m.read(&vals);
        for (0..seq) |i| for (0..seq) |j| {
            const v = vals[i * seq + j];
            if (j <= i) {
                try std.testing.expectEqual(@as(f32, 0.0), v);
            } else {
                try std.testing.expect(v < -1e8);
            }
        };
    }

    {
        // left=0, right=1 -> chunks of 2, each frame sees its whole chunk only.
        const m = try nn.chunkedLimitedMask(&fx.ctx, allocator, seq, 0, 1);
        var vals: [seq * seq]f32 = undefined;
        try m.read(&vals);
        for (0..seq) |i| for (0..seq) |j| {
            const chunk_start: usize = (i / 2) * 2;
            const allowed: bool = (j >= chunk_start and j < chunk_start + 2);
            const v = vals[i * seq + j];
            if (allowed) {
                try std.testing.expectEqual(@as(f32, 0.0), v);
            } else {
                try std.testing.expect(v < -1e8);
            }
        };
    }
}

test "api.nn: sinusoidal relative-position table is centered and normalized" {
    const allocator = std.testing.allocator;
    const fx = try Fixture.init(allocator);
    defer fx.deinit(allocator);

    const length: usize = 3;
    const d_model: usize = 4;
    const t = try nn.sinusoidalRelPos(&fx.ctx, allocator, length, d_model);

    const rows: usize = 2 * length - 1;
    try std.testing.expectEqualSlices(usize, &[_]usize{ rows, d_model }, t.getShape());

    var vals: [(2 * length - 1) * d_model]f32 = undefined;
    try t.read(&vals);

    // The middle row is relative offset 0: sin(0)=0, cos(0)=1.
    const mid: usize = length - 1;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), vals[mid * d_model + 0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), vals[mid * d_model + 1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), vals[mid * d_model + 2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), vals[mid * d_model + 3], 1e-6);

    // Row 0 is the most positive offset, the last row its negation (sin is odd).
    try std.testing.expectApproxEqAbs(vals[0], -vals[(rows - 1) * d_model], 1e-6);

    // An odd feature width has no sin/cos pairing.
    try std.testing.expectError(error.InvalidArgument, nn.sinusoidalRelPos(&fx.ctx, allocator, 2, 3));
}
