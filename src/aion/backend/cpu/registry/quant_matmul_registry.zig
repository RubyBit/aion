// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const types = @import("../../types.zig");
const quant_matmul = @import("../kernels/quant_matmul.zig");
const cpuid = @import("../tuning/cpuid.zig");
const cpu_target = @import("cpu_target.zig");
const gemm_shapes = @import("gemm_shapes.zig");

pub const Tuning = struct {
    // Micro-kernel tiles
    mr: usize,
    nr: usize,

    // SIMD lane width used by the tuned kernel.
    lanes: usize,

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

pub const VariantId = gemm_shapes.PackedVariantId;

pub const Candidate = struct {
    id: VariantId,
    kernels: QuantKernels,
};

fn kernelsFor(comptime t: Tuning) QuantKernels {
    const K = quant_matmul.Kernel(.{ .kc = t.kc, .mc = t.mc, .nc = t.nc, .mr = t.mr, .nr = t.nr, .lanes = t.lanes });
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
    .{ .id = .small, .kernels = kernelsFor(.{ .mr = 6, .nr = 8, .lanes = 4, .kc = 128, .mc = 144, .nc = 128 }) },
    .{ .id = .medium, .kernels = kernelsFor(.{ .mr = 6, .nr = 8, .lanes = 4, .kc = 256, .mc = 144, .nc = 256 }) },
    .{ .id = .large, .kernels = kernelsFor(.{ .mr = 6, .nr = 8, .lanes = 4, .kc = 512, .mc = 240, .nc = 512 }) },

    .{ .id = .small, .kernels = kernelsFor(.{ .mr = 6, .nr = 16, .lanes = 8, .kc = 128, .mc = 144, .nc = 128 }) },
    .{ .id = .medium, .kernels = kernelsFor(.{ .mr = 6, .nr = 16, .lanes = 8, .kc = 256, .mc = 144, .nc = 256 }) },
    .{ .id = .large, .kernels = kernelsFor(.{ .mr = 6, .nr = 16, .lanes = 8, .kc = 512, .mc = 240, .nc = 512 }) },

    .{ .id = .small, .kernels = kernelsFor(.{ .mr = 6, .nr = 32, .lanes = 16, .kc = 128, .mc = 144, .nc = 128 }) },
    .{ .id = .medium, .kernels = kernelsFor(.{ .mr = 6, .nr = 32, .lanes = 16, .kc = 256, .mc = 144, .nc = 256 }) },
    .{ .id = .large, .kernels = kernelsFor(.{ .mr = 6, .nr = 32, .lanes = 16, .kc = 512, .mc = 240, .nc = 512 }) },
};

pub fn maxScratchBytes() usize {
    var best: usize = 0;
    inline for (candidates) |c| {
        best = @max(best, c.kernels.scratch_bytes);
    }
    return best;
}

pub fn selectForTile(default_kernels: QuantKernels, k: usize, n: usize) ?QuantKernels {
    // Choose the smallest candidate that can cover the requested tile.
    // (Smaller NC/KC reduces packed-B footprint and scratch bandwidth.)
    return gemm_shapes.selectSmallestCoveringKernels(candidates, default_kernels, k, n);
}

pub fn selectForTarget(target: cpu_target.Target) Candidate {
    const lanes: usize = target.preferred_f32_lanes;

    // Similar heuristic to f32: pick largest candidate whose packed-B footprint
    // fits in ~75% of L2.
    const l2_bytes: usize = target.caches.l2_bytes;
    if (l2_bytes == 0) {
        for (candidates) |c| {
            if (c.id == .medium and c.kernels.tuning.lanes == lanes) return c;
        }
        @panic("missing quant matmul medium candidate for lane group");
    }

    const budget: usize = gemm_shapes.l2Budget75(l2_bytes);

    var best: Candidate = blk: {
        for (candidates) |c| {
            if (c.id == .small and c.kernels.tuning.lanes == lanes) break :blk c;
        }
        @panic("missing quant matmul small candidate for lane group");
    };
    for (candidates) |c| {
        if (c.kernels.tuning.lanes != lanes) continue;
        const footprint: usize = c.kernels.packed_b_bytes;
        if (footprint <= budget) best = c;
    }

    if (l2_bytes >= (1 * 1024 * 1024) and best.id == .small) {
        for (candidates) |c| {
            if (c.id == .medium and c.kernels.tuning.lanes == lanes) return c;
        }
        @panic("missing quant matmul medium candidate for lane group");
    }
    return best;
}

pub fn selectHeuristic(info: cpuid.CpuInfo) Candidate {
    return selectForTarget(cpu_target.fromCpuInfo(info));
}
