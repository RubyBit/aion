// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const types = @import("../../types.zig");
const matmul_nt = @import("../kernels/matmul_nt.zig");
const quant_matmul_nt = @import("../kernels/quant_matmul_nt.zig");
const cpuid = @import("../tuning/cpuid.zig");
const cpu_target = @import("cpu_target.zig");

/// Tuning metadata for native-transposed GEMM kernels (`A[M,K] @ B[N,K]^T`).
///
/// Unlike packed GEMM, this path does not own KC/MC/NC cache-blocking knobs because
/// B is already row-contiguous in the access order consumed by the kernel. The values
/// here describe the SIMD shape selected for the CPU target and leave room for future
/// NT-specific blocking/prefetch tuning without coupling this registry to packed GEMM.
pub const Tuning = matmul_nt.Tuning;

/// Compute C[:, n_start..n_start+n_count] = alpha * A @ B[n_start..n_start+n_count, :]^T + beta*C[...]
/// where A is M×K f32 and B is N×K f32 (row-major, contiguous along K). No packing step.
pub const MatMulNtF32Fn = *const fn (
    a_ptr: [*]align(1) const f32,
    b_ptr: [*]align(1) const f32,
    c_ptr: [*]align(1) f32,
    m_total: usize,
    k: usize,
    n_total: usize,
    n_start: usize,
    n_count: usize,
    alpha: f32,
    beta: f32,
) types.BackendError!void;

/// Compute C[:, n_start..n_start+n_count] = alpha * A @ B[n_start..n_start+n_count, :]^T + beta*C[...]
/// where A is M×K f32, B is N×K q8_0 (row-major, contiguous along K), and C is M×n_count f32.
/// No pre-pack: B's rows are already in the natural access order for A @ B^T.
pub const MatMulNtQ8_0Fn = *const fn (
    a_ptr: [*]align(1) const f32,
    b_ptr: [*]const u8,
    c_ptr: [*]align(1) f32,
    m_total: usize,
    k: usize,
    n_total: usize,
    n_start: usize,
    n_count: usize,
    alpha: f32,
    beta: f32,
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
    const Q8_0 = quant_matmul_nt.Kernel(t);
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
