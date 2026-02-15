const std = @import("std");
const types = @import("../../types.zig");
const matmul_tuned = @import("../kernels/matmul_tuned.zig");
const cpuid = @import("../tuning/cpuid.zig");

pub const Tuning = struct {
    // Micro-kernel tiles
    mr: usize,
    nr: usize,

    // Cache blocking params.
    kc: usize,
    mc: usize,
    nc: usize,
};

pub const F32Kernels = struct {
    tuning: Tuning,

    scratch_bytes: usize,
    scratch_alignment: usize,

    pack_b_tile: *const fn (scratch_bytes: []u8, k: usize, n: usize, b_bytes: []const u8) types.BackendError!void,
    pack_a_tile: *const fn (k: usize, m: usize, a_bytes: []const u8, packed_a_out: []align(32) f32) types.BackendError!void,
    matmul_packed_b: *const fn (scratch_bytes: []u8, packed_b_view: []align(32) const f32, params: types.MatMulParams, c_bytes: []u8, a_bytes: []const u8) types.BackendError!void,
    matmul_packed_ab: *const fn (packed_a: []align(32) const f32, packed_b_view: []align(32) const f32, params: types.MatMulParams, c_bytes: []u8) types.BackendError!void,
};

pub const VariantId = enum { small, medium, large };

/// Special-purpose variant for conv-like workloads where K is large (so we want KC=512)
/// but N is small (e.g. cout<=128). This keeps the number of K-blocks low while
/// drastically shrinking packed-B (KC*NC) and scratch footprint.
///
/// Note: this is NOT part of `candidates` used by `selectHeuristic`, because that
/// heuristic assumes larger variants also have larger packed-B footprints.
pub const f32_kc512_nc128: F32Kernels = kernelsFor(.{ .mr = 6, .nr = 16, .kc = 512, .mc = 144, .nc = 128 });

pub const Candidate = struct {
    id: VariantId,
    kernels: F32Kernels,
};

fn kernelsFor(comptime t: Tuning) F32Kernels {
    const K = matmul_tuned.Kernel(.{ .kc = t.kc, .mc = t.mc, .nc = t.nc, .mr = t.mr, .nr = t.nr });
    return .{
        .tuning = t,
        .scratch_bytes = K.scratchBytes(),
        .scratch_alignment = K.ScratchAlignment,
        .pack_b_tile = K.packBTileF32,
        .pack_a_tile = K.packATileF32,
        .matmul_packed_b = K.matmulF32PackedB,
        .matmul_packed_ab = K.matmulF32PackedAB,
    };
}

pub const candidates = [_]Candidate{
    .{ .id = .small, .kernels = kernelsFor(.{ .mr = 6, .nr = 16, .kc = 128, .mc = 144, .nc = 128 }) },
    .{ .id = .medium, .kernels = kernelsFor(.{ .mr = 6, .nr = 16, .kc = 256, .mc = 144, .nc = 256 }) },
    // Tuned for conv-like workloads with large K: fewer K-blocks (KC=512) while keeping
    // packed-B footprint reasonable (KC*NC*4 = 1MiB).
    .{ .id = .large, .kernels = kernelsFor(.{ .mr = 6, .nr = 16, .kc = 512, .mc = 288, .nc = 512 }) },
};

pub fn selectHeuristic(info: cpuid.CpuInfo) Candidate {
    // Heuristic goal: pick the largest variant that still keeps the packed-B tile
    // in L2 (or as much of it as possible).
    //
    // Important: L2 is typically per-core, and on many CPUs it's "only" 1-2MiB.
    // If we require the whole KC*NC panel to fit in a tiny fraction of L2, we end
    // up selecting variants that don't even cover the program tiler (e.g. tk=256).
    //
    // So we:
    // - default to `medium` when we can't detect caches
    // - otherwise, pick the largest candidate whose B-panel footprint fits in ~75% of L2
    // - but never return `small` when L2 is reasonably sized (>= 1MiB), because
    //   the current tiler commonly uses tk=256.

    const l2_bytes: usize = info.caches.l2_bytes;
    if (info.features.avx2 and l2_bytes >= (256 * 1024)) return candidates[2];
    if (l2_bytes == 0) return candidates[1];

    const budget: usize = (l2_bytes * 3) / 4;

    var best: Candidate = candidates[0];
    inline for (candidates) |c| {
        const footprint: usize = c.kernels.tuning.kc * c.kernels.tuning.nc * @sizeOf(f32);
        if (footprint <= budget) best = c;
    }

    if (l2_bytes >= (1 * 1024 * 1024) and best.id == .small) return candidates[1];
    return best;
}
