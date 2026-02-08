const std = @import("std");
const types = @import("../types.zig");

const elemwise = @import("kernels/elemwise.zig");
const relu_k = @import("kernels/relu.zig");
const gelu_k = @import("kernels/gelu.zig");
const silu_k = @import("kernels/silu.zig");
const sigmoid_k = @import("kernels/sigmoid.zig");
const tanh_k = @import("kernels/tanh.zig");
const softmax_k = @import("kernels/softmax.zig");
const reduce_k = @import("kernels/reduce.zig");
const matmul_registry = @import("registry/matmul_registry.zig");
const quant_matmul_registry = @import("registry/quant_matmul_registry.zig");
const matvec_registry = @import("registry/matvec_registry.zig");
const attention_registry = @import("registry/attention_registry.zig");

const BackendError = types.BackendError;
const MatMulParams = types.MatMulParams;

fn expectSliceApproxEqAbs(expected: []const f32, got: []const f32, tol: f32) !void {
    try std.testing.expectEqual(expected.len, got.len);
    for (expected, got) |e, g| {
        try std.testing.expectApproxEqAbs(e, g, tol);
    }
}

fn expectAllFinite(vals: []const f32) !void {
    for (vals) |v| try std.testing.expect(std.math.isFinite(v));
}

fn expectMonotonicNonDecreasing(vals: []const f32, eps: f32) !void {
    if (vals.len < 2) return;
    var i: usize = 0;
    while (i + 1 < vals.len) : (i += 1) {
        try std.testing.expect(vals[i] <= vals[i + 1] + eps);
    }
}

fn maxAbsDiffF32(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var m: f32 = 0.0;
    for (a, b) |x, y| m = @max(m, @abs(x - y));
    return m;
}

fn matmulKernelsById(id: matmul_registry.VariantId) matmul_registry.F32Kernels {
    inline for (matmul_registry.candidates) |c| {
        if (c.id == id) return c.kernels;
    }
    @panic("missing matmul kernel variant");
}

fn quantMatmulKernelsById(id: quant_matmul_registry.VariantId) quant_matmul_registry.QuantKernels {
    inline for (quant_matmul_registry.candidates) |c| {
        if (c.id == id) return c.kernels;
    }
    @panic("missing quant matmul kernel variant");
}

fn matvecKernelsById(id: matvec_registry.VariantId) matvec_registry.Kernels {
    inline for (matvec_registry.candidates) |c| {
        if (c.id == id) return c.kernels;
    }
    @panic("missing matvec kernel variant");
}

fn attentionKernelsById(id: attention_registry.VariantId) attention_registry.Kernels {
    inline for (attention_registry.candidates) |c| {
        if (c.id == id) return c.kernels;
    }
    @panic("missing attention kernel variant");
}

fn sigmoidRefF32(x: f32) f32 {
    const xf: f64 = @floatCast(x);
    // Stable enough for our small test domain.
    const y: f64 = 1.0 / (1.0 + std.math.exp(-xf));
    return @floatCast(y);
}

fn tanhRefF32(x: f32) f32 {
    const xf: f64 = @floatCast(x);
    return @floatCast(std.math.tanh(xf));
}

fn siluRefF32(x: f32) f32 {
    return x * sigmoidRefF32(x);
}

fn geluTanhApproxRefF32(x: f32) f32 {
    // Standard tanh-based GELU approximation.
    const xf: f64 = @floatCast(x);
    const k0: f64 = std.math.sqrt(2.0 / std.math.pi);
    const inner: f64 = k0 * (xf + 0.044715 * xf * xf * xf);
    const y: f64 = 0.5 * xf * (1.0 + std.math.tanh(inner));
    return @floatCast(y);
}

test "cpu kernels: elemwise add f32" {
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    var out = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    try elemwise.elemwiseBinaryF32(.add, std.mem.sliceAsBytes(out[0..]), std.mem.sliceAsBytes(a[0..]), std.mem.sliceAsBytes(b[0..]), out.len);
    try expectSliceApproxEqAbs(&[_]f32{ 6.0, 8.0, 10.0, 12.0 }, out[0..], 1e-6);
}

test "cpu kernels: unary relu f32" {
    var a = [_]f32{ -2.0, -0.5, 0.0, 0.5, 3.0 };
    var out = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0 };
    try relu_k.reluF32(std.mem.sliceAsBytes(out[0..]), std.mem.sliceAsBytes(a[0..]), a.len);
    try expectSliceApproxEqAbs(&[_]f32{ 0.0, 0.0, 0.0, 0.5, 3.0 }, out[0..], 1e-6);
}

test "cpu kernels: unary sigmoid f32 properties and accuracy" {
    // Symmetric, ascending domain with a center point at 0.
    var x = [_]f32{ -6.0, -3.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 3.0, 6.0 };
    var out: [x.len]f32 = undefined;

    try sigmoid_k.sigmoidF32(std.mem.sliceAsBytes(out[0..]), std.mem.sliceAsBytes(x[0..]), x.len);
    try expectAllFinite(out[0..]);
    for (out) |v| try std.testing.expect(v >= 0.0 and v <= 1.0);

    // Known point.
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[5], 2e-2);

    // Monotone increasing.
    try expectMonotonicNonDecreasing(out[0..], 1e-3);

    // Symmetry: s(-x) ~= 1 - s(x).
    var i: usize = 0;
    while (i < x.len / 2) : (i += 1) {
        const j: usize = x.len - 1 - i;
        try std.testing.expectApproxEqAbs(@as(f32, 1.0) - out[j], out[i], 3e-2);
    }

    // Accuracy vs exp-based sigmoid.
    var ref: [x.len]f32 = undefined;
    for (x, 0..) |v, idx| ref[idx] = sigmoidRefF32(v);
    const max_abs: f32 = maxAbsDiffF32(out[0..], ref[0..]);
    try std.testing.expect(max_abs <= 3e-2);
}

test "cpu kernels: unary tanh f32 properties and accuracy" {
    var x = [_]f32{ -6.0, -3.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 3.0, 6.0 };
    var out: [x.len]f32 = undefined;

    try tanh_k.tanhF32(std.mem.sliceAsBytes(out[0..]), std.mem.sliceAsBytes(x[0..]), x.len);
    try expectAllFinite(out[0..]);
    for (out) |v| try std.testing.expect(v >= -1.0 and v <= 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[5], 2e-2);

    try expectMonotonicNonDecreasing(out[0..], 1e-3);

    // Odd symmetry: tanh(-x) ~= -tanh(x).
    var i: usize = 0;
    while (i < x.len / 2) : (i += 1) {
        const j: usize = x.len - 1 - i;
        try std.testing.expectApproxEqAbs(-out[j], out[i], 3e-2);
    }

    var ref: [x.len]f32 = undefined;
    for (x, 0..) |v, idx| ref[idx] = tanhRefF32(v);
    const max_abs: f32 = maxAbsDiffF32(out[0..], ref[0..]);
    try std.testing.expect(max_abs <= 3e-2);
}

test "cpu kernels: unary silu f32 properties and accuracy" {
    var x = [_]f32{ -6.0, -3.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 3.0, 6.0 };
    var out: [x.len]f32 = undefined;

    try silu_k.siluF32(std.mem.sliceAsBytes(out[0..]), std.mem.sliceAsBytes(x[0..]), x.len);
    try expectAllFinite(out[0..]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[5], 2e-2);

    // Note: SiLU/Swish is not strictly monotone on the negative side.
    // Avoid monotonicity assertions here; rely on identities + reference checks.

    // Identity from definition silu(x) = x*sigmoid(x): silu(x) - silu(-x) = x.
    var i: usize = 0;
    while (i < x.len / 2) : (i += 1) {
        const j: usize = x.len - 1 - i;
        // x[j] is the positive counterpart.
        try std.testing.expectApproxEqAbs(x[j], out[j] - out[i], 5e-2);
    }

    // Accuracy vs exp-based reference.
    var ref: [x.len]f32 = undefined;
    for (x, 0..) |v, idx| ref[idx] = siluRefF32(v);
    const max_abs: f32 = maxAbsDiffF32(out[0..], ref[0..]);
    try std.testing.expect(max_abs <= 5e-2);
}

test "cpu kernels: unary gelu f32 properties and accuracy" {
    var x = [_]f32{ -6.0, -3.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 3.0, 6.0 };
    var out: [x.len]f32 = undefined;

    try gelu_k.geluF32(std.mem.sliceAsBytes(out[0..]), std.mem.sliceAsBytes(x[0..]), x.len);
    try expectAllFinite(out[0..]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[5], 2e-2);

    // Gelu should approach 0 for large negative and ~x for large positive.
    try std.testing.expect(out[0] <= 0.0);
    try std.testing.expectApproxEqAbs(x[x.len - 1], out[out.len - 1], 2e-1);

    // The widely-used tanh GELU approximation is "approximately" odd in the sense:
    // gelu(x) - gelu(-x) ~= x (exact for the true CDF-based GELU).
    var i: usize = 0;
    while (i < x.len / 2) : (i += 1) {
        const j: usize = x.len - 1 - i;
        try std.testing.expectApproxEqAbs(x[j], out[j] - out[i], 8e-2);
    }

    // Accuracy vs tanh-based GELU formula (but with accurate tanh).
    var ref: [x.len]f32 = undefined;
    for (x, 0..) |v, idx| ref[idx] = geluTanhApproxRefF32(v);
    const max_abs: f32 = maxAbsDiffF32(out[0..], ref[0..]);
    try std.testing.expect(max_abs <= 6e-2);
}

fn softmaxRefRowF32(out: []f32, x: []const f32) void {
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

test "cpu kernels: softmax f32 matches reference (rank-1)" {
    // This tests the core softmax math used by the tiled exec.
    var x = [_]f32{ -4.0, -1.0, 0.0, 0.5, 1.0, 2.0, 4.0 };
    var out = [_]f32{0} ** x.len;
    var ref = [_]f32{0} ** x.len;
    softmaxRefRowF32(ref[0..], x[0..]);

    // Use the exec helpers by building fake BufferViews around packed memory.
    const out_bytes = std.mem.sliceAsBytes(out[0..]);
    const in_bytes = std.mem.sliceAsBytes(x[0..]);

    const layout = types.Layout{ .rank = 1, .shape = &[_]usize{x.len}, .strides_bytes = &[_]isize{@intCast(@sizeOf(f32))} };
    const in_view = types.BufferViewConst{ .bytes = in_bytes, .dtype = .f32, .layout = layout };
    const out_view = types.BufferViewMut{ .bytes = out_bytes, .dtype = .f32, .layout = layout };

    var max_buf: [1]f32 = .{-std.math.inf(f32)};
    var sum_buf: [1]f32 = .{0.0};
    softmax_k.updateMaxF32(max_buf[0..], in_view, 1);
    softmax_k.expSumStoreF32(sum_buf[0..], out_view, in_view, max_buf[0..], 1);
    softmax_k.normalizeF32(out_view, sum_buf[0..], 1);

    try expectAllFinite(out[0..]);
    var s: f32 = 0.0;
    for (out) |v| {
        try std.testing.expect(v >= 0.0 and v <= 1.0);
        s += v;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), s, 2e-3);
    try expectSliceApproxEqAbs(ref[0..], out[0..], 3e-2);
}

test "cpu kernels: reduce sum/mean f32" {
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var sum_out = [_]f32{0.0};
    var mean_out = [_]f32{0.0};

    const sum: f32 = try reduce_k.sumF32Range(std.mem.sliceAsBytes(a[0..]), 0, a.len);
    const mean: f32 = sum / @as(f32, @floatFromInt(a.len));
    sum_out[0] = sum;
    mean_out[0] = mean;

    try std.testing.expectApproxEqAbs(@as(f32, 10.0), sum, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), mean, 1e-6);
}

fn naiveMatmulF32(params: MatMulParams, c: []f32, a: []const f32, b: []const f32) void {
    const m = params.m;
    const n = params.n;
    const k = params.k;

    var i: usize = 0;
    while (i < m) : (i += 1) {
        var j: usize = 0;
        while (j < n) : (j += 1) {
            var acc: f32 = 0.0;
            var kk: usize = 0;
            while (kk < k) : (kk += 1) {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            const idx = i * n + j;
            c[idx] = params.alpha * acc + params.beta * c[idx];
        }
    }
}

test "cpu kernels: tuned f32 matmul via registry" {
    const m: usize = 8;
    const n: usize = 16;
    const k: usize = 32;
    const params: MatMulParams = .{ .m = m, .n = n, .k = k, .alpha = 1.0, .beta = 0.0 };

    var prng = std.Random.DefaultPrng.init(0xabcdef);
    const rnd = prng.random();

    var a: [m * k]f32 = undefined;
    var b: [k * n]f32 = undefined;
    for (a[0..]) |*x| x.* = (rnd.float(f32) - 0.5) * 2.0;
    for (b[0..]) |*x| x.* = (rnd.float(f32) - 0.5) * 2.0;

    var c_ref: [m * n]f32 = [_]f32{0.0} ** (m * n);
    var c: [m * n]f32 = [_]f32{0.0} ** (m * n);

    naiveMatmulF32(params, c_ref[0..], a[0..], b[0..]);

    const kernels: matmul_registry.F32Kernels = matmulKernelsById(.medium);
    try std.testing.expect(k > 0 and n > 0);

    const pb_f32_len: usize = kernels.tuning.kc * kernels.tuning.nc;
    const pb_bytes_len: usize = pb_f32_len * @sizeOf(f32);

    var scratch: []align(32) u8 = try std.testing.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), kernels.scratch_bytes);
    defer std.testing.allocator.free(scratch);

    // Pack B and run per-row (m==1) just like program execution does.
    // This keeps the test independent of any higher-level executor.
    try kernels.pack_b_tile(scratch, k, n, std.mem.sliceAsBytes(b[0..]));

    const packed_b_view: []align(32) const f32 = @alignCast(std.mem.bytesAsSlice(f32, scratch[0..pb_bytes_len]));

    var row: usize = 0;
    while (row < m) : (row += 1) {
        const a_row: []const u8 = std.mem.sliceAsBytes(a[row * k .. (row + 1) * k]);
        const c_row: []u8 = std.mem.sliceAsBytes(c[row * n .. (row + 1) * n]);
        const pp: MatMulParams = .{ .m = 1, .n = n, .k = k, .alpha = 1.0, .beta = 0.0 };
        try kernels.matmul_packed_b(scratch, packed_b_view, pp, c_row, a_row);
    }

    try expectSliceApproxEqAbs(c_ref[0..], c[0..], 1e-3);
}

test "cpu kernels: attention calcScores handles partial tiles" {
    const kernels: attention_registry.Kernels = attentionKernelsById(.baseline);

    const M: usize = 7;
    const N: usize = 19;
    const K: usize = 13;

    var prng = std.Random.DefaultPrng.init(0x4242);
    const rnd = prng.random();

    var q: [M * K]f32 = undefined;
    var k: [N * K]f32 = undefined;
    for (q[0..]) |*x| x.* = (rnd.float(f32) - 0.5) * 2.0;
    for (k[0..]) |*x| x.* = (rnd.float(f32) - 0.5) * 2.0;

    var scores: [M * N]f32 = [_]f32{0.0} ** (M * N);
    var qt: [M * K]f32 = [_]f32{0.0} ** (M * K);
    var kt: [N * K]f32 = [_]f32{0.0} ** (N * K);

    kernels.pack_q_block(M, K, q[0..], K, qt[0..]);
    kernels.pack_k_block(N, K, k[0..], K, kt[0..]);
    kernels.calc_scores_f32(M, N, K, q[0..], K, k[0..], K, kt[0..], qt[0..], scores[0..], N);

    var ref: [M * N]f32 = undefined;
    var r: usize = 0;
    while (r < M) : (r += 1) {
        var c: usize = 0;
        while (c < N) : (c += 1) {
            var acc: f32 = 0.0;
            var kk: usize = 0;
            while (kk < K) : (kk += 1) {
                acc += q[r * K + kk] * k[c * K + kk];
            }
            ref[r * N + c] = acc;
        }
    }

    try expectSliceApproxEqAbs(ref[0..], scores[0..], 1e-3);
}

test "cpu kernels: matvec f32" {
    const k: usize = 32;
    const n: usize = 64;
    const params: MatMulParams = .{ .m = 1, .n = n, .k = k, .alpha = 1.0, .beta = 0.0 };

    var prng = std.Random.DefaultPrng.init(0x12345678);
    const rnd = prng.random();

    var a: [k]f32 = undefined;
    var b: [k * n]f32 = undefined;
    for (a[0..]) |*x| x.* = (rnd.float(f32) - 0.5) * 2.0;
    for (b[0..]) |*x| x.* = (rnd.float(f32) - 0.5) * 2.0;

    var c_ref: [n]f32 = [_]f32{0.0} ** n;
    var c: [n]f32 = [_]f32{0.0} ** n;

    // naive
    var j: usize = 0;
    while (j < n) : (j += 1) {
        var acc: f32 = 0.0;
        var kk: usize = 0;
        while (kk < k) : (kk += 1) {
            acc += a[kk] * b[kk * n + j];
        }
        c_ref[j] = acc;
    }

    try matvecKernelsById(.tuned).matvec_f32(params, std.mem.sliceAsBytes(c[0..]), std.mem.sliceAsBytes(a[0..]), std.mem.sliceAsBytes(b[0..]));
    try expectSliceApproxEqAbs(c_ref[0..], c[0..], 1e-3);
}

const Q8_BYTES: usize = types.DType.q8_0.info().block_bytes;
const Q4_BYTES: usize = types.DType.q4_0.info().block_bytes;

fn quantizeQ8_0FromF32Block32(vals: *const [32]f32, out: *[Q8_BYTES]u8) void {
    var max_abs: f32 = 0.0;
    for (vals.*) |v| {
        const a = @abs(v);
        if (a > max_abs) max_abs = a;
    }
    const scale: f32 = if (max_abs == 0.0) 1.0 else (max_abs / 127.0);
    const scale_f16: f16 = @floatCast(scale);
    const scale_bits: [2]u8 = @bitCast(scale_f16);
    out[0] = scale_bits[0];
    out[1] = scale_bits[1];
    var i: usize = 0;
    while (i < 32) : (i += 1) {
        const qf: f32 = vals[i] / scale;
        const qi32: i32 = @intFromFloat(std.math.round(qf));
        const qi8: i8 = @intCast(std.math.clamp(qi32, -128, 127));
        out[2 + i] = @bitCast(qi8);
    }
}

fn quantizeQ4_0FromF32Block32(vals: *const [32]f32, out: *[Q4_BYTES]u8) void {
    var max_abs: f32 = 0.0;
    for (vals.*) |v| {
        const a: f32 = @abs(v);
        if (a > max_abs) max_abs = a;
    }

    // q4_0 represents values in [-8, 7] with a per-block scale.
    const scale: f32 = if (max_abs == 0.0) 1.0 else (max_abs / 7.0);
    const scale_f16: f16 = @floatCast(scale);
    const scale_bits: [2]u8 = @bitCast(scale_f16);
    out[0] = scale_bits[0];
    out[1] = scale_bits[1];

    // 16 bytes of packed nibbles.
    var i: usize = 0;
    while (i < 16) : (i += 1) out[2 + i] = 0;

    var t: usize = 0;
    while (t < 32) : (t += 1) {
        const qf: f32 = vals[t] / scale;
        const qi32: i32 = @intFromFloat(std.math.round(qf));
        const qi: i32 = std.math.clamp(qi32, -8, 7);
        const nib: u8 = @intCast(qi + 8); // store bias +8

        const byte_idx: usize = t >> 1;
        const is_lo: bool = ((t & 1) == 0);
        if (is_lo) {
            out[2 + byte_idx] = (out[2 + byte_idx] & 0xF0) | (nib & 0x0F);
        } else {
            out[2 + byte_idx] = (out[2 + byte_idx] & 0x0F) | (nib << 4);
        }
    }
}

test "cpu kernels: tuned q8_0 matmul via registry" {
    const m: usize = 8;
    const n: usize = 32;
    const k: usize = 64;
    const params: MatMulParams = .{ .m = m, .n = n, .k = k, .alpha = 1.0, .beta = 0.0 };

    var prng = std.Random.DefaultPrng.init(0x8888);
    const rnd = prng.random();

    var a: [m * k]f32 = undefined;
    var b_f32: [k * n]f32 = undefined;
    for (a[0..]) |*x| x.* = (rnd.float(f32) - 0.5) * 2.0;
    for (b_f32[0..]) |*x| x.* = (rnd.float(f32) - 0.5) * 2.0;

    // Quantize B into block layout [k_blocks, n].
    const k_blocks: usize = k / 32;
    var bq: [k_blocks * n * Q8_BYTES]u8 = undefined;
    var kb: usize = 0;
    while (kb < k_blocks) : (kb += 1) {
        var j: usize = 0;
        while (j < n) : (j += 1) {
            var block: [32]f32 = undefined;
            var t: usize = 0;
            while (t < 32) : (t += 1) {
                block[t] = b_f32[(kb * 32 + t) * n + j];
            }
            const off: usize = (kb * n + j) * Q8_BYTES;
            const dst: *[Q8_BYTES]u8 = @ptrCast(bq[off..][0..Q8_BYTES]);
            quantizeQ8_0FromF32Block32(&block, dst);
        }
    }

    var c_ref: [m * n]f32 = [_]f32{0.0} ** (m * n);
    var c: [m * n]f32 = [_]f32{0.0} ** (m * n);
    naiveMatmulF32(params, c_ref[0..], a[0..], b_f32[0..]);

    const kernels: quant_matmul_registry.QuantKernels = quantMatmulKernelsById(.medium);
    var scratch: []align(32) u8 = try std.testing.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), kernels.scratch_bytes);
    defer std.testing.allocator.free(scratch);

    try kernels.pack_b_tile_q8_0(scratch, k, n, bq[0..]);
    const packed_b_view: quant_matmul_registry.PackedBView = @alignCast(scratch[0..kernels.packed_b_bytes]);
    try kernels.matmul_packed_b(scratch, packed_b_view, params, std.mem.sliceAsBytes(c[0..]), std.mem.sliceAsBytes(a[0..]));

    try expectSliceApproxEqAbs(c_ref[0..], c[0..], 2e-1);
}

test "cpu kernels: tuned q4_0 matmul via registry" {
    const m: usize = 8;
    const n: usize = 32;
    const k: usize = 64;
    const params: MatMulParams = .{ .m = m, .n = n, .k = k, .alpha = 1.0, .beta = 0.0 };

    var prng = std.Random.DefaultPrng.init(0x9999);
    const rnd = prng.random();

    var a: [m * k]f32 = undefined;
    var b_f32: [k * n]f32 = undefined;
    for (a[0..]) |*x| x.* = (rnd.float(f32) - 0.5) * 2.0;
    for (b_f32[0..]) |*x| x.* = (rnd.float(f32) - 0.5) * 2.0;

    const k_blocks: usize = k / 32;
    var bq: [k_blocks * n * Q4_BYTES]u8 = undefined;
    var kb: usize = 0;
    while (kb < k_blocks) : (kb += 1) {
        var j: usize = 0;
        while (j < n) : (j += 1) {
            var block: [32]f32 = undefined;
            var t: usize = 0;
            while (t < 32) : (t += 1) {
                block[t] = b_f32[(kb * 32 + t) * n + j];
            }
            const off: usize = (kb * n + j) * Q4_BYTES;
            const dst: *[Q4_BYTES]u8 = @ptrCast(bq[off..][0..Q4_BYTES]);
            quantizeQ4_0FromF32Block32(&block, dst);
        }
    }

    var c_ref: [m * n]f32 = [_]f32{0.0} ** (m * n);
    var c: [m * n]f32 = [_]f32{0.0} ** (m * n);
    naiveMatmulF32(params, c_ref[0..], a[0..], b_f32[0..]);

    const kernels: quant_matmul_registry.QuantKernels = quantMatmulKernelsById(.medium);
    var scratch: []align(32) u8 = try std.testing.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), kernels.scratch_bytes);
    defer std.testing.allocator.free(scratch);

    try kernels.pack_b_tile_q4_0(scratch, k, n, bq[0..]);
    const packed_b_view: quant_matmul_registry.PackedBView = @alignCast(scratch[0..kernels.packed_b_bytes]);
    try kernels.matmul_packed_b(scratch, packed_b_view, params, std.mem.sliceAsBytes(c[0..]), std.mem.sliceAsBytes(a[0..]));

    // q4 is much noisier.
    try expectSliceApproxEqAbs(c_ref[0..], c[0..], 8e-1);
}

test "cpu kernels: matvec q8_0" {
    const k: usize = 32;
    const n: usize = 16;
    const params: MatMulParams = .{ .m = 1, .n = n, .k = k, .alpha = 1.0, .beta = 0.0 };

    var prng = std.Random.DefaultPrng.init(0x1111);
    const rnd = prng.random();

    var a: [k]f32 = undefined;
    for (a[0..]) |*x| x.* = (rnd.float(f32) - 0.5) * 2.0;

    // Build B in block layout: n blocks, each block is 32 elems.
    var bq: [n * Q8_BYTES]u8 = undefined;
    var b_f32: [k * n]f32 = undefined;

    for (0..n) |j| {
        var block: [32]f32 = undefined;
        for (0..32) |t| {
            const v: f32 = (rnd.float(f32) - 0.5) * 2.0;
            block[t] = v;
            b_f32[t * n + j] = v;
        }
        const dst: *[Q8_BYTES]u8 = @ptrCast(bq[j * Q8_BYTES ..][0..Q8_BYTES]);
        quantizeQ8_0FromF32Block32(&block, dst);
    }

    // naive reference
    var c_ref: [n]f32 = [_]f32{0.0} ** n;
    var j: usize = 0;
    while (j < n) : (j += 1) {
        var acc: f32 = 0.0;
        var kk: usize = 0;
        while (kk < k) : (kk += 1) {
            acc += a[kk] * b_f32[kk * n + j];
        }
        c_ref[j] = acc;
    }

    var c: [n]f32 = [_]f32{0.0} ** n;
    const kernels: quant_matmul_registry.QuantKernels = quantMatmulKernelsById(.medium);
    var scratch: []align(32) u8 = try std.testing.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), kernels.scratch_bytes);
    defer std.testing.allocator.free(scratch);

    try kernels.pack_b_tile_q8_0(scratch, k, n, bq[0..]);
    const packed_b_view: quant_matmul_registry.PackedBView = @alignCast(scratch[0..kernels.packed_b_bytes]);
    try kernels.matmul_packed_b(scratch, packed_b_view, params, std.mem.sliceAsBytes(c[0..]), std.mem.sliceAsBytes(a[0..]));
    try expectSliceApproxEqAbs(c_ref[0..], c[0..], 2e-1);
}

comptime {
    _ = BackendError;
    _ = quant_matmul_registry;
}
