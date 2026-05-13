// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const types = @import("../../types.zig");
const matvec_tuned = @import("../kernels/matvec.zig");
const cpuid = @import("../tuning/cpuid.zig");

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
};

pub const VariantId = enum {
    /// Baseline SIMD kernel in `matvecmul.zig`.
    baseline,

    avx2,
    avx512,
};

pub const Candidate = struct {
    id: VariantId,
    kernels: Kernels,
};

fn kernelsFor(comptime t: Tuning) Kernels {
    const K = matvec_tuned.Kernel(t);
    return .{
        .tuning = t,
        .matvec_f32 = K.matvecF32,
        .matvec_f32_range = K.matvecF32Range,
        .matvec_f16 = K.matvecF16,
        .matvec_f16_range = K.matvecF16Range,
    };
}

pub const candidates = [_]Candidate{
    .{ .id = .baseline, .kernels = kernelsFor(.{ .nr = 8, .lanes = 4, .nc = 128, .prefetch_k_dist = 4 }) },
    .{ .id = .avx2, .kernels = kernelsFor(.{ .nr = 16, .lanes = 8, .nc = 256, .prefetch_k_dist = 4 }) },
    .{ .id = .avx512, .kernels = kernelsFor(.{ .nr = 32, .lanes = 16, .nc = 256, .prefetch_k_dist = 4 }) },
};

pub fn selectHeuristic(info: cpuid.CpuInfo) Candidate {
    const lanes: usize = cpuid.preferredF32Lanes(info);
    return switch (lanes) {
        16 => candidates[2],
        8 => candidates[1],
        else => candidates[0],
    };
}
