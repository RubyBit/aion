const std = @import("std");
const types = @import("../../types.zig");
const simd = @import("simd.zig");
const quant = @import("quant.zig");

const BackendError = types.BackendError;
const MatMulParams = types.MatMulParams;

/// C[M,N] (f16) = alpha * A[M,K] (f16) @ B[K,N] (f16) + beta * C
pub fn matmulF16(params: MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) BackendError!void {
    const m: usize = params.m;
    const n: usize = params.n;
    const k: usize = params.k;
    const alpha: f32 = params.alpha;
    const beta: f32 = params.beta;

    const c: []align(1) f16 = simd.bytesAsSliceMutUnaligned(f16, c_bytes);
    const a: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, a_bytes);
    const b: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, b_bytes);

    if (c.len < m * n or a.len < m * k or b.len < k * n) {
        return BackendError.InvalidArgument;
    }

    for (0..m) |i| {
        for (0..n) |j| {
            var acc: f32 = 0.0;
            for (0..k) |kk| {
                acc += @as(f32, a[i * k + kk]) * @as(f32, b[kk * n + j]);
            }
            const c_idx: usize = i * n + j;
            const old_c: f32 = @as(f32, c[c_idx]);
            c[c_idx] = @floatCast(alpha * acc + beta * old_c);
        }
    }
}

/// C[M,N] (f32) = alpha * A[M,K] (f16) @ B[K,N] (f16) + beta * C
pub fn matmulF16ToF32(params: MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) BackendError!void {
    const m: usize = params.m;
    const n: usize = params.n;
    const k: usize = params.k;
    const alpha: f32 = params.alpha;
    const beta: f32 = params.beta;

    const c: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, c_bytes);
    const a: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, a_bytes);
    const b: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, b_bytes);

    if (c.len < m * n or a.len < m * k or b.len < k * n) {
        return BackendError.InvalidArgument;
    }

    for (0..m) |i| {
        for (0..n) |j| {
            var acc: f32 = 0.0;
            for (0..k) |kk| {
                acc += @as(f32, a[i * k + kk]) * @as(f32, b[kk * n + j]);
            }
            const c_idx: usize = i * n + j;
            c[c_idx] = alpha * acc + beta * c[c_idx];
        }
    }
}

pub fn matmulQ4_0(params: MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) BackendError!void {
    return quant.matmulQ4_0(params, c_bytes, a_bytes, b_bytes);
}

pub fn matmulQ8_0(params: MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) BackendError!void {
    return quant.matmulQ8_0(params, c_bytes, a_bytes, b_bytes);
}
