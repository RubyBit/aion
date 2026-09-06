// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const types = @import("../../types.zig");
const matvec_tuned = @import("../kernels/matvec.zig");
const matvec_q = @import("../kernels/matvec_q.zig");
const cpuid = @import("../tuning/cpuid.zig");
const cpu_target = @import("cpu_target.zig");
const builtin = @import("builtin");

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
pub const QuantMatvecAccumulateFn = *const fn (params: types.MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8, acc_bytes: []align(32) u8, prepared_a: []align(32) u8, prepare_a: bool, first_k_tile: bool, last_k_tile: bool) types.BackendError!void;
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
    matvec_q8_0_kmajor_accumulate: QuantMatvecAccumulateFn,
};

pub const VariantId = cpu_target.SimdWidth;

pub const Candidate = struct {
    id: VariantId,
    kernels: Kernels,
};

fn kernelsFor(comptime t: Tuning, comptime dot_enc: matvec_q.DotEnc) Kernels {
    const K = matvec_tuned.Kernel(t);
    const Q8_0 = matvec_q.MatvecKernel(.{ .lanes = t.lanes, .dot_enc = dot_enc });
    return .{
        .tuning = t,
        .matvec_f32 = K.matvecF32,
        .matvec_f32_range = K.matvecF32Range,
        .matvec_f16 = K.matvecF16,
        .matvec_f16_range = K.matvecF16Range,
        .matvec_q8_0_kmajor = Q8_0.matvecQ8_0KMajor,
        .matvec_q8_0_kmajor_accumulate = Q8_0.matvecQ8_0KMajorAccumulate,
    };
}

pub const candidates = [_]Candidate{
    .{ .id = .simd128, .kernels = kernelsFor(.{ .nr = 8, .lanes = 4, .nc = 128, .prefetch_k_dist = 4 }, .f32) },
    .{ .id = .simd256, .kernels = kernelsFor(.{ .nr = 16, .lanes = 8, .nc = 256, .prefetch_k_dist = 4 }, .f32) },
    .{ .id = .simd512, .kernels = kernelsFor(.{ .nr = 32, .lanes = 16, .nc = 256, .prefetch_k_dist = 4 }, .f32) },
};

fn candidateForId(id: VariantId) Candidate {
    for (candidates) |c| {
        if (c.id == id) return c;
    }
    return candidates[0];
}

pub fn selectForTarget(target: cpu_target.Target) Candidate {
    // The self-hosted x86 assembler cannot emit VPDPBUSD or its {vex}
    // prefix. Eliminate these variants at comptime: runtime dispatch alone
    // still instantiates their assembly, even on CPUs without VNNI.
    if (comptime builtin.cpu.arch.isX86() and builtin.zig_backend != .stage2_x86_64) {
        return switch (target.quant_dot) {
            .vex => candidateForIdDot(target.simd_width, .vex),
            .evex => candidateForIdDot(target.simd_width, .evex),
            else => candidateForId(target.simd_width),
        };
    }
    if (comptime builtin.cpu.arch.isAARCH64()) {
        return if (target.quant_dot == .sdot) candidateForIdDot(target.simd_width, .sdot) else candidateForId(target.simd_width);
    }
    return candidateForId(target.simd_width);
}

fn candidateForIdDot(id: VariantId, comptime enc: matvec_q.DotEnc) Candidate {
    return switch (id) {
        .simd128 => .{ .id = id, .kernels = kernelsFor(.{ .nr = 8, .lanes = 4, .nc = 128, .prefetch_k_dist = 4 }, enc) },
        .simd256 => .{ .id = id, .kernels = kernelsFor(.{ .nr = 16, .lanes = 8, .nc = 256, .prefetch_k_dist = 4 }, enc) },
        .simd512 => .{ .id = id, .kernels = kernelsFor(.{ .nr = 32, .lanes = 16, .nc = 256, .prefetch_k_dist = 4 }, enc) },
    };
}

pub fn selectHeuristic(info: cpuid.CpuInfo) Candidate {
    return selectForTarget(cpu_target.fromCpuInfo(info));
}
