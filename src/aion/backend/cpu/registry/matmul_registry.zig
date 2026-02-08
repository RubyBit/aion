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
    matmul_packed_b: *const fn (scratch_bytes: []u8, packed_b_view: []align(32) const f32, params: types.MatMulParams, c_bytes: []u8, a_bytes: []const u8) types.BackendError!void,
};

pub const VariantId = enum { small, medium, large };

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
        .matmul_packed_b = K.matmulF32PackedB,
    };
}

pub const candidates = [_]Candidate{
    .{ .id = .small, .kernels = kernelsFor(.{ .mr = 6, .nr = 16, .kc = 128, .mc = 144, .nc = 128 }) },
    .{ .id = .medium, .kernels = kernelsFor(.{ .mr = 6, .nr = 16, .kc = 256, .mc = 144, .nc = 256 }) },
    .{ .id = .large, .kernels = kernelsFor(.{ .mr = 6, .nr = 16, .kc = 256, .mc = 240, .nc = 512 }) },
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
