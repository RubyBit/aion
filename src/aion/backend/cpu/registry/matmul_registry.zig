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

/// Special-purpose variant for conv-like workloads where K is large (so we want KC=512)
/// but N is small (e.g. cout<=128). This keeps the number of K-blocks low while
/// drastically shrinking packed-B (KC*NC) and scratch footprint.
///
/// Note: this is NOT part of `candidates` used by `selectHeuristic`, because that
/// heuristic assumes larger variants also have larger packed-B footprints.
pub const f32_kc512_nc128: F32Kernels = kernelsFor(.{ .mr = 6, .nr = 16, .kc = 512, .mc = 144, .nc = 128 });

/// Special-purpose variant for conv-like workloads where N is small (e.g. cout<=128)
/// and the default matmul uses KC=256.
pub const f32_kc256_nc128: F32Kernels = kernelsFor(.{ .mr = 6, .nr = 16, .kc = 256, .mc = 144, .nc = 128 });

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
        .pack_b_tile_f16_to_packed_f32 = K.packBTileF16ToPackedF32,
        .pack_a_tile_f16_to_packed_f32 = K.packATileF16ToPackedF32,
        .matmul_packed_b = K.matmulF32PackedB,
        .matmul_packed_ab = K.matmulF32PackedAB,
        // NT kernel doesn't use (KC, MC, NC) tuning — B is already in the right layout.
        .matmul_nt_f32 = matmul_tuned.matmulNtF32,
    };
}

pub const candidates = [_]Candidate{
    .{ .id = .small, .kernels = kernelsFor(.{ .mr = 6, .nr = 16, .kc = 128, .mc = 144, .nc = 128 }) },
    .{ .id = .medium, .kernels = kernelsFor(.{ .mr = 6, .nr = 16, .kc = 256, .mc = 144, .nc = 256 }) },
    // Tuned for conv-like workloads with large K: fewer K-blocks (KC=512) while keeping
    // packed-B footprint reasonable (KC*NC*4 = 1MiB).
    .{ .id = .large, .kernels = kernelsFor(.{ .mr = 6, .nr = 16, .kc = 512, .mc = 288, .nc = 512 }) },
};

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
    // Include non-default/special-purpose variants.
    best = @max(best, f32_kc512_nc128.scratch_bytes);
    best = @max(best, f32_kc256_nc128.scratch_bytes);

    const pb_512_128: usize = f32_kc512_nc128.tuning.kc * f32_kc512_nc128.tuning.nc * @sizeOf(f32);
    const pa_512_128: usize = f32_kc512_nc128.tuning.mc * f32_kc512_nc128.tuning.kc * @sizeOf(f32);
    const ctmp_512_128: usize = f32_kc512_nc128.tuning.mc * f32_kc512_nc128.tuning.nc * @sizeOf(f32);
    best = @max(best, pb_512_128 + pa_512_128 + ctmp_512_128);

    const pb_256_128: usize = f32_kc256_nc128.tuning.kc * f32_kc256_nc128.tuning.nc * @sizeOf(f32);
    const pa_256_128: usize = f32_kc256_nc128.tuning.mc * f32_kc256_nc128.tuning.kc * @sizeOf(f32);
    const ctmp_256_128: usize = f32_kc256_nc128.tuning.mc * f32_kc256_nc128.tuning.nc * @sizeOf(f32);
    best = @max(best, pb_256_128 + pa_256_128 + ctmp_256_128);

    return best;
}

pub fn selectForTile(k: usize, n: usize) ?F32Kernels {
    // Choose the smallest kernel variant that can cover the requested tile.
    // Smaller KC/NC reduces packed-B footprint and scratch bandwidth.
    var best: ?F32Kernels = null;

    inline for (candidates) |c| {
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

    // Consider special-purpose narrow-N variants too.
    if (k <= f32_kc512_nc128.tuning.kc and n <= f32_kc512_nc128.tuning.nc) {
        if (best == null) {
            best = f32_kc512_nc128;
        } else {
            const cur: F32Kernels = best.?;
            const cur_area: usize = cur.tuning.kc * cur.tuning.nc;
            const cand_area: usize = f32_kc512_nc128.tuning.kc * f32_kc512_nc128.tuning.nc;
            if (cand_area < cur_area) best = f32_kc512_nc128;
        }
    }
    if (k <= f32_kc256_nc128.tuning.kc and n <= f32_kc256_nc128.tuning.nc) {
        if (best == null) {
            best = f32_kc256_nc128;
        } else {
            const cur: F32Kernels = best.?;
            const cur_area: usize = cur.tuning.kc * cur.tuning.nc;
            const cand_area: usize = f32_kc256_nc128.tuning.kc * f32_kc256_nc128.tuning.nc;
            if (cand_area < cur_area) best = f32_kc256_nc128;
        }
    }

    return best;
}

pub fn selectForConvOcTile(default_kernels: F32Kernels, oc_count: usize) F32Kernels {
    // Conv implicit-GEMM packs weights by OC tiles (N dimension). For small N,
    // reducing NC can shrink packed-B panels and scratch footprint significantly.
    if (oc_count == 0) return default_kernels;

    if (default_kernels.tuning.kc == 512 and default_kernels.tuning.nc > 128 and oc_count <= 128) {
        return f32_kc512_nc128;
    }
    if (default_kernels.tuning.kc == 256 and default_kernels.tuning.nc > 128 and oc_count <= 128) {
        return f32_kc256_nc128;
    }
    return default_kernels;
}

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
