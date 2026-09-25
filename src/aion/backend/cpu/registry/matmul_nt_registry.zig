// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const builtin = @import("builtin");
const types = @import("../../types.zig");
const matmul_nt = @import("../kernels/matmul_nt.zig");
const matmul_nt_q = @import("../kernels/matmul_nt_q.zig");
const matmul_q_i8 = @import("../kernels/matmul_q_i8.zig");
const cpuid = @import("../tuning/cpuid.zig");
const cpu_target = @import("cpu_target.zig");

/// Tuning metadata for native-transposed GEMM kernels (`A[M,K] @ B[N,K]^T`).
///
/// Unlike packed GEMM, this path does not own KC/MC/NC cache-blocking knobs because
/// B is already row-contiguous in the access order consumed by the kernel. The values
/// here describe the SIMD shape selected for the CPU target and leave room for future
/// NT-specific blocking/prefetch tuning without coupling this registry to packed GEMM.
pub const Tuning = matmul_nt.Tuning;

/// One N-tile of `C = alpha * A @ B^T + beta * C`, with A `[m, k]` f32 and B `[n, k]`
/// f32 (row-major over K, i.e. already transposed w.r.t. a standard matmul). No pack
/// step: B's rows are already in the access order the kernel wants.
///
/// The slices are the tile, not the whole matrix — the caller (`exec/matmul_nt.zig`)
/// splits N into tiles and hands each worker `b_bytes`/`c_bytes` for its own tile, so
/// `params.n` is that tile's column count and there is no tile offset in the ABI.
/// `params.ldc` is C's row stride (defaulting to `params.n`); every current caller
/// passes a contiguous `[m, n]` tile.
pub const MatMulNtF32Fn = *const fn (
    params: types.MatMulParams,
    c_bytes: []u8,
    a_bytes: []const u8,
    b_bytes: []const u8,
) types.BackendError!void;

/// Same contract as `MatMulNtF32Fn`, with B a `[n, k]` q8_0 weight and A already
/// quantized (`matmul_nt_q.prepareActivation`), shared read-only across N tiles.
pub const MatMulNtQ8_0Fn = matmul_nt.MatMulNtQ8_0Fn;

pub const Kernels = struct {
    tuning: Tuning,

    matmul_f32: MatMulNtF32Fn,
    /// One kernel per block order, since the order is a property of the weight.
    matmul_q8_0: std.EnumArray(types.QuantBlockOrder, MatMulNtQ8_0Fn),
};

pub const VariantId = cpu_target.SimdWidth;

pub const Candidate = struct {
    id: VariantId,
    kernels: Kernels,
};

fn kernelsFor(comptime t: Tuning) Kernels {
    const F32 = matmul_nt.Kernel(t);
    const Q8_0 = matmul_nt_q.Kernel(t);
    return .{
        .tuning = t,
        .matmul_f32 = F32.matmulNtF32,
        .matmul_q8_0 = q8Kernels(Q8_0),
    };
}

fn q8Kernels(comptime Q8_0: type) std.EnumArray(types.QuantBlockOrder, MatMulNtQ8_0Fn) {
    var out: std.EnumArray(types.QuantBlockOrder, MatMulNtQ8_0Fn) = undefined;
    inline for (comptime std.enums.values(types.QuantBlockOrder)) |order| out.set(order, Q8_0.matmulNtQ8_0(order));
    return out;
}

fn candidateFor(comptime id: VariantId, comptime dot_enc: matmul_q_i8.DotEnc) Candidate {
    const lanes = id.f32Lanes();
    return .{ .id = id, .kernels = kernelsFor(.{ .lanes = lanes, .nr = 2 * lanes, .dot_enc = dot_enc }) };
}

/// The kernels at `target`'s width, with its byte dot for q8.
pub fn selectForTarget(comptime target: cpu_target.Target) Candidate {
    return candidateFor(target.simd_width, comptime cpu_target.emittable(target.int8_dot));
}
