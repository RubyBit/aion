// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const types = @import("../../types.zig");
const matvec_tuned = @import("../kernels/matvec.zig");
const quant_matmul = @import("../kernels/quant_matmul.zig");
const cpuid = @import("../tuning/cpuid.zig");
const cpu_target = @import("cpu_target.zig");

pub const Tuning = struct {
    /// Micro-tile along N.
    nr: usize,

    /// SIMD lane width used by the tuned kernel.
    lanes: usize,

    /// Outer blocking along N.
    nc: usize,

    /// How far ahead to prefetch B along K (in rows).
    prefetch_k_dist: usize = 4,
};

pub const MatvecFn = *const fn (params: types.MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) types.BackendError!void;
pub const QuantMatvecFn = *const fn (params: types.MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) types.BackendError!void;
pub const MatvecRangeFn = *const fn (
    params: types.MatMulParams,
    col_start: usize,
    col_count: usize,
    c_bytes: []u8,
    a_bytes: []const u8,
    b_bytes: []const u8,
) types.BackendError!void;

pub const Kernels = struct {
    tuning: Tuning,

    matvec_f32: MatvecFn,
    matvec_f32_range: MatvecRangeFn,
    matvec_f16: MatvecFn,
    matvec_f16_range: MatvecRangeFn,

    /// Direct q8_0 matvec for K-major/block-major B layout used to avoid pack-B on large M=1 tiles.
    matvec_q8_0_kmajor: QuantMatvecFn,
};

pub const VariantId = cpu_target.SimdWidth;

pub const Candidate = struct {
    id: VariantId,
    kernels: Kernels,
};

fn kernelsFor(comptime t: Tuning) Kernels {
    const K = matvec_tuned.Kernel(t);
    const Q8_0 = quant_matmul.MatvecKernel(.{ .lanes = t.lanes });
    return .{
        .tuning = t,
        .matvec_f32 = K.matvecF32,
        .matvec_f32_range = K.matvecF32Range,
        .matvec_f16 = K.matvecF16,
        .matvec_f16_range = K.matvecF16Range,
        .matvec_q8_0_kmajor = Q8_0.matvecQ8_0KMajor,
    };
}

pub const candidates = [_]Candidate{
    .{ .id = .simd128, .kernels = kernelsFor(.{ .nr = 8, .lanes = 4, .nc = 128, .prefetch_k_dist = 4 }) },
    .{ .id = .simd256, .kernels = kernelsFor(.{ .nr = 16, .lanes = 8, .nc = 256, .prefetch_k_dist = 4 }) },
    .{ .id = .simd512, .kernels = kernelsFor(.{ .nr = 32, .lanes = 16, .nc = 256, .prefetch_k_dist = 4 }) },
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
