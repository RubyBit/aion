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

    /// Vector width the f32/f16 matvec accumulates with, when that should differ
    /// from the ISA width. The loop is memory-bound, so what governs it is how
    /// many independent loads it keeps in flight, not the native vector size:
    /// on NEON, accumulating two registers' worth per step instead of one
    /// measured 46 -> 65 GB/s on VGG-19's classifier. 0 means "same as `lanes`".
    ///
    /// The kernel keeps `nr == 2 * lanes`, so its row tile widens to match.
    acc_lanes: usize = 0,

    pub fn accLanes(t: Tuning) usize {
        return if (t.acc_lanes != 0) t.acc_lanes else t.lanes;
    }
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

fn kernelsFor(comptime t: Tuning, comptime dot_enc: cpu_target.DotEnc) Kernels {
    // The quantized kernel below stays at the ISA width; only the f32/f16 loop
    // trades register pressure for loads in flight.
    const K = matvec_tuned.Kernel(.{ .nr = 2 * t.accLanes(), .lanes = t.accLanes(), .nc = t.nc, .prefetch_k_dist = t.prefetch_k_dist });
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

/// The kernels at `target`'s width, with its byte dot for q8.
pub fn selectForTarget(comptime target: cpu_target.Target) Candidate {
    return candidateAt(target.simd_width, comptime cpu_target.emittable(target.int8_dot));
}

pub fn candidateAt(comptime id: VariantId, comptime enc: cpu_target.DotEnc) Candidate {
    return switch (id) {
        .simd128 => .{ .id = id, .kernels = kernelsFor(.{ .nr = 8, .lanes = 4, .nc = matvec_tuned.NC_MAX, .prefetch_k_dist = 4, .acc_lanes = 8 }, enc) },
        .simd256 => .{ .id = id, .kernels = kernelsFor(.{ .nr = 16, .lanes = 8, .nc = matvec_tuned.NC_MAX, .prefetch_k_dist = 4 }, enc) },
        .simd512 => .{ .id = id, .kernels = kernelsFor(.{ .nr = 32, .lanes = 16, .nc = matvec_tuned.NC_MAX, .prefetch_k_dist = 4 }, enc) },
    };
}
