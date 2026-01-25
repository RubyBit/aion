const types = @import("../../types.zig");
const matvec_tuned = @import("matvec.zig");
const cpuid = @import("../tuning/cpuid.zig");

pub const Tuning = struct {
    /// Micro-tile along N.
    nr: usize,

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
};

pub const VariantId = enum {
    /// Baseline SIMD kernel in `matvecmul.zig`.
    baseline,

    /// Comptime-generated kernel (currently same algorithm, parameterized).
    tuned,
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
    };
}

pub const candidates = [_]Candidate{
    .{ .id = .tuned, .kernels = kernelsFor(.{ .nr = 16, .nc = 256, .prefetch_k_dist = 4 }) },
};

pub fn selectHeuristic(info: cpuid.CpuInfo) Candidate {
    _ = info;
    // Default to tuned; fall back to baseline by switching this if needed.
    return candidates[0];
}
