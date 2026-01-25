const std = @import("std");
const types = @import("../types.zig");

const elemwise = @import("kernels/elemwise.zig");
const reduce_k = @import("kernels/reduce.zig");
const matmul_registry = @import("kernels/matmul_registry.zig");
const quant_matmul_registry = @import("kernels/quant_matmul_registry.zig");
const matvec_registry = @import("kernels/matvec_registry.zig");
const quant = @import("kernels/quant.zig");

const BackendError = types.BackendError;
const MatMulParams = types.MatMulParams;

fn expectSliceApproxEqAbs(expected: []const f32, got: []const f32, tol: f32) !void {
    try std.testing.expectEqual(expected.len, got.len);
    for (expected, got) |e, g| {
        try std.testing.expectApproxEqAbs(e, g, tol);
    }
}

test "cpu kernels: elemwise add f32" {
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    var out = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    try elemwise.elemwiseBinaryF32(.add, std.mem.sliceAsBytes(out[0..]), std.mem.sliceAsBytes(a[0..]), std.mem.sliceAsBytes(b[0..]), out.len);
    try expectSliceApproxEqAbs(&[_]f32{ 6.0, 8.0, 10.0, 12.0 }, out[0..], 1e-6);
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

    const kernels = matmul_registry.candidates[1].kernels;
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

    try matvec_registry.candidates[0].kernels.matvec_f32(params, std.mem.sliceAsBytes(c[0..]), std.mem.sliceAsBytes(a[0..]), std.mem.sliceAsBytes(b[0..]));
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

    const kernels = quant_matmul_registry.candidates[1].kernels;
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

    const kernels = quant_matmul_registry.candidates[1].kernels;
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
    const kernels = quant_matmul_registry.candidates[1].kernels;
    var scratch: []align(32) u8 = try std.testing.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), kernels.scratch_bytes);
    defer std.testing.allocator.free(scratch);

    try kernels.pack_b_tile_q8_0(scratch, k, n, bq[0..]);
    const packed_b_view: quant_matmul_registry.PackedBView = @alignCast(scratch[0..kernels.packed_b_bytes]);
    try kernels.matmul_packed_b(scratch, packed_b_view, params, std.mem.sliceAsBytes(c[0..]), std.mem.sliceAsBytes(a[0..]));
    try expectSliceApproxEqAbs(c_ref[0..], c[0..], 2e-1);
}

comptime {
    _ = BackendError;
    _ = quant;
    _ = quant_matmul_registry;
}
