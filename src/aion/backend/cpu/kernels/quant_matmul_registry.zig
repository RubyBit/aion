const std = @import("std");
const types = @import("../../types.zig");
const quant_matmul_tuned = @import("quant_matmul_tuned.zig");
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

pub const PackedBView = []align(32) const u8;

pub const PackBFn = *const fn (scratch_bytes: []u8, k: usize, n: usize, b_bytes: []const u8) types.BackendError!void;
pub const MatMulPackedBFn = *const fn (scratch_bytes: []u8, packed_b_view: PackedBView, params: types.MatMulParams, c_bytes: []u8, a_bytes: []const u8) types.BackendError!void;

pub const QuantKernels = struct {
    tuning: Tuning,

    scratch_bytes: usize,
    scratch_alignment: usize,

    packed_b_bytes: usize,

    pack_b_tile_q4_0: PackBFn,
    pack_b_tile_q8_0: PackBFn,

    matmul_packed_b: MatMulPackedBFn,
};

pub const VariantId = enum { small, medium, large };

pub const Candidate = struct {
    id: VariantId,
    kernels: QuantKernels,
};

fn kernelsFor(comptime t: Tuning) QuantKernels {
    const K = quant_matmul_tuned.Kernel(.{ .kc = t.kc, .mc = t.mc, .nc = t.nc, .mr = t.mr, .nr = t.nr });
    return .{
        .tuning = t,
        .scratch_bytes = K.scratchBytes(),
        .scratch_alignment = K.ScratchAlignment,
        .packed_b_bytes = K.packedBBytes(),
        .pack_b_tile_q4_0 = K.packBTileQ4_0,
        .pack_b_tile_q8_0 = K.packBTileQ8_0,
        .matmul_packed_b = K.matmulQx0PackedB,
    };
}

pub const candidates = [_]Candidate{
    .{ .id = .small, .kernels = kernelsFor(.{ .mr = 6, .nr = 16, .kc = 128, .mc = 144, .nc = 128 }) },
    .{ .id = .medium, .kernels = kernelsFor(.{ .mr = 6, .nr = 16, .kc = 256, .mc = 144, .nc = 256 }) },
    .{ .id = .large, .kernels = kernelsFor(.{ .mr = 6, .nr = 16, .kc = 256, .mc = 240, .nc = 512 }) },
};

pub fn maxScratchBytes() usize {
    var best: usize = 0;
    inline for (candidates) |c| {
        best = @max(best, c.kernels.scratch_bytes);
    }
    return best;
}

pub fn selectForTile(k: usize, n: usize) ?QuantKernels {
    // Choose the smallest candidate that can cover the requested tile.
    // (Smaller NC/KC reduces packed-B footprint and scratch bandwidth.)
    var best: ?QuantKernels = null;
    inline for (candidates) |c| {
        if (k <= c.kernels.tuning.kc and n <= c.kernels.tuning.nc) {
            if (best == null) {
                best = c.kernels;
            } else {
                const cur = best.?;
                const cur_area: usize = cur.tuning.kc * cur.tuning.nc;
                const cand_area: usize = c.kernels.tuning.kc * c.kernels.tuning.nc;
                if (cand_area < cur_area) best = c.kernels;
            }
        }
    }
    return best;
}

pub fn selectHeuristic(info: cpuid.CpuInfo) Candidate {
    // Similar heuristic to f32: pick largest candidate whose packed-B footprint
    // fits in ~75% of L2.
    const l2_bytes = info.caches.l2_bytes;
    if (l2_bytes == 0) return candidates[1];

    const budget: usize = (l2_bytes * 3) / 4;

    var best: Candidate = candidates[0];
    inline for (candidates) |c| {
        const footprint: usize = c.kernels.packed_b_bytes;
        if (footprint <= budget) best = c;
    }

    if (l2_bytes >= (1 * 1024 * 1024) and best.id == .small) return candidates[1];
    return best;
}
