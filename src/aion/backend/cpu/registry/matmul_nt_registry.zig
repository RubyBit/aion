// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const types = @import("../../types.zig");
const matmul_nt = @import("../kernels/matmul_nt.zig");
const matmul_nt_q = @import("../kernels/matmul_nt_q.zig");
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

/// Same contract as `MatMulNtF32Fn`, with `b_bytes` holding B `[n, k]` as q8_0 in its
/// on-disk layout: one contiguous run of `k / 32` blocks per row, no pre-pack.
pub const MatMulNtQ8_0Fn = *const fn (
    params: types.MatMulParams,
    c_bytes: []u8,
    a_bytes: []const u8,
    b_bytes: []const u8,
) types.BackendError!void;

pub const Kernels = struct {
    tuning: Tuning,

    matmul_f32: MatMulNtF32Fn,
    matmul_q8_0: MatMulNtQ8_0Fn,
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
        .matmul_q8_0 = Q8_0.matmulNtQ8_0,
    };
}

pub const candidates = [_]Candidate{
    .{ .id = .simd128, .kernels = kernelsFor(.{ .lanes = 4, .nr = 8 }) },
    .{ .id = .simd256, .kernels = kernelsFor(.{ .lanes = 8, .nr = 16 }) },
    .{ .id = .simd512, .kernels = kernelsFor(.{ .lanes = 16, .nr = 32 }) },
};

fn candidateForId(id: VariantId) Candidate {
    for (candidates) |c| {
        if (c.id == id) return c;
    }
    return candidates[0];
}

pub fn selectForTarget(target: cpu_target.Target) Candidate {
    return candidateForId(target.simd_width);
}

pub fn selectHeuristic(info: cpuid.CpuInfo) Candidate {
    return selectForTarget(cpu_target.fromCpuInfo(info));
}
