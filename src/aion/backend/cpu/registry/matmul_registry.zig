// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const types = @import("../../types.zig");
const matmul_tuned = @import("../kernels/matmul_tuned.zig");
const cpuid = @import("../tuning/cpuid.zig");

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

pub const F32Kernels = struct {
    tuning: Tuning,

    scratch_bytes: usize,
    scratch_alignment: usize,

    pack_b_tile: *const fn (scratch_bytes: []u8, k: usize, n: usize, b_bytes: []const u8) types.BackendError!void,
    pack_a_tile: *const fn (k: usize, m: usize, a_bytes: []const u8, packed_a_out: []align(32) f32) types.BackendError!void,
    pack_b_tile_f16_to_packed_f32: *const fn (packed_b: []align(32) f32, k: usize, n: usize, b_bytes: []const u8) types.BackendError!void,
    pack_a_tile_f16_to_packed_f32: *const fn (packed_a_out: []align(32) f32, m: usize, k: usize, a_bytes: []const u8) types.BackendError!void,
    matmul_packed_b: *const fn (scratch_bytes: []u8, packed_b_view: []align(32) const f32, params: types.MatMulParams, c_bytes: []u8, a_bytes: []const u8) types.BackendError!void,
    matmul_packed_ab: *const fn (packed_a: []align(32) const f32, packed_b_view: []align(32) const f32, params: types.MatMulParams, c_bytes: []u8) types.BackendError!void,
    matmul_nt_f32: MatMulNtF32Fn,
};

pub const VariantId = enum { small, medium, large };

pub const Candidate = struct {
    id: VariantId,
    kernels: F32Kernels,
};

fn kernelsFor(comptime t: Tuning) F32Kernels {
    const K = matmul_tuned.Kernel(.{ .kc = t.kc, .mc = t.mc, .nc = t.nc, .mr = t.mr, .nr = t.nr, .lanes = t.lanes });
    return .{
        .tuning = t,
        .scratch_bytes = K.scratchBytes(),
        .scratch_alignment = K.ScratchAlignment,
        .pack_b_tile = K.packBTileF32,
        .pack_a_tile = K.packATileF32,
        .pack_b_tile_f16_to_packed_f32 = K.packBTileF16ToPackedF32,
        .pack_a_tile_f16_to_packed_f32 = K.packATileF16ToPackedF32,
        .matmul_packed_b = K.matmulF32PackedB,
        .matmul_packed_ab = K.matmulF32PackedAB,
        // NT kernel doesn't use (KC, MC, NC) tuning — B is already in the right layout.
        .matmul_nt_f32 = matmul_tuned.matmulNtF32,
    };
}

pub const candidates = [_]Candidate{
    .{ .id = .small, .kernels = kernelsFor(.{ .mr = 6, .nr = 8, .lanes = 4, .kc = 128, .mc = 144, .nc = 128 }) },
    .{ .id = .medium, .kernels = kernelsFor(.{ .mr = 6, .nr = 8, .lanes = 4, .kc = 256, .mc = 144, .nc = 256 }) },
    .{ .id = .large, .kernels = kernelsFor(.{ .mr = 6, .nr = 8, .lanes = 4, .kc = 512, .mc = 288, .nc = 512 }) },

    .{ .id = .small, .kernels = kernelsFor(.{ .mr = 6, .nr = 16, .lanes = 8, .kc = 128, .mc = 144, .nc = 128 }) },
    .{ .id = .medium, .kernels = kernelsFor(.{ .mr = 6, .nr = 16, .lanes = 8, .kc = 256, .mc = 144, .nc = 256 }) },
    .{ .id = .large, .kernels = kernelsFor(.{ .mr = 6, .nr = 16, .lanes = 8, .kc = 512, .mc = 288, .nc = 512 }) },

    .{ .id = .small, .kernels = kernelsFor(.{ .mr = 6, .nr = 32, .lanes = 16, .kc = 128, .mc = 144, .nc = 128 }) },
    .{ .id = .medium, .kernels = kernelsFor(.{ .mr = 6, .nr = 32, .lanes = 16, .kc = 256, .mc = 144, .nc = 256 }) },
    .{ .id = .large, .kernels = kernelsFor(.{ .mr = 6, .nr = 32, .lanes = 16, .kc = 512, .mc = 288, .nc = 512 }) },
};

fn candidateFor(id: VariantId, lanes: usize) Candidate {
    for (candidates) |c| {
        if (c.id == id and c.kernels.tuning.lanes == lanes) return c;
    }
    @panic("missing matmul kernel candidate for lane group");
}

pub fn maxScratchBytes() usize {
    var best: usize = 0;
    inline for (candidates) |c| {
        best = @max(best, c.kernels.scratch_bytes);

        // f16 execution path may route through packed-f32 kernels and, for f16 output,
        // needs an additional f32 accumulation tile buffer.
        const pb_bytes: usize = c.kernels.tuning.kc * c.kernels.tuning.nc * @sizeOf(f32);
        const pa_bytes: usize = c.kernels.tuning.mc * c.kernels.tuning.kc * @sizeOf(f32);
        const c_tmp_bytes: usize = c.kernels.tuning.mc * c.kernels.tuning.nc * @sizeOf(f32);
        const f16_via_f32_need: usize = pb_bytes + pa_bytes + c_tmp_bytes;
        best = @max(best, f16_via_f32_need);
    }
    return best;
}

pub fn selectForTile(default_kernels: F32Kernels, k: usize, n: usize) ?F32Kernels {
    // Choose the smallest kernel variant that can cover the requested tile.
    // Smaller KC/NC reduces packed-B footprint and scratch bandwidth.
    var best: ?F32Kernels = null;
    const lanes: usize = default_kernels.tuning.lanes;

    for (candidates) |c| {
        if (c.kernels.tuning.lanes != lanes) continue;
        if (k <= c.kernels.tuning.kc and n <= c.kernels.tuning.nc) {
            if (best == null) {
                best = c.kernels;
            } else {
                const cur: F32Kernels = best.?;
                const cur_area: usize = cur.tuning.kc * cur.tuning.nc;
                const cand_area: usize = c.kernels.tuning.kc * c.kernels.tuning.nc;
                if (cand_area < cur_area) best = c.kernels;
            }
        }
    }

    return best;
}

pub fn selectForConvOcTile(default_kernels: F32Kernels, oc_count: usize) F32Kernels {
    // Conv implicit-GEMM packs weights by OC tiles (N dimension). For small N,
    // reducing NC can shrink packed-B panels and scratch footprint significantly.
    if (oc_count == 0) return default_kernels;

    return selectForTile(default_kernels, default_kernels.tuning.kc, oc_count) orelse default_kernels;
}

pub fn selectHeuristic(info: cpuid.CpuInfo) Candidate {
    const lanes: usize = cpuid.preferredF32Lanes(info);

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
    if (l2_bytes == 0) return candidateFor(.medium, lanes);

    const budget: usize = (l2_bytes * 3) / 4;

    var best: Candidate = candidateFor(.small, lanes);
    for (candidates) |c| {
        if (c.kernels.tuning.lanes != lanes) continue;
        const footprint: usize = c.kernels.tuning.kc * c.kernels.tuning.nc * @sizeOf(f32);
        if (footprint <= budget) best = c;
    }

    if (l2_bytes >= (1 * 1024 * 1024) and best.id == .small) return candidateFor(.medium, lanes);
    return best;
}
