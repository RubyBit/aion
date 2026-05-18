// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

/// Shared policy helpers for packed GEMM registries.
///
/// The f32 and quantized packed GEMM registries intentionally keep concrete kernel
/// structs and function pointers, but they share candidate identifiers and selection
/// rules. Keeping that policy here avoids each dtype registry growing its own copy of
/// the same lane/tile lookup code.
pub const PackedVariantId = enum { small, medium, large };

pub fn l2Budget75(l2_bytes: usize) usize {
    return (l2_bytes * 3) / 4;
}

pub fn candidateFor(comptime candidates: anytype, id: PackedVariantId, lanes: usize) @TypeOf(candidates[0]) {
    for (candidates) |c| {
        if (c.id == id and c.kernels.tuning.lanes == lanes) return c;
    }
    @panic("missing packed GEMM candidate for lane group");
}

pub fn selectSmallestCoveringKernels(comptime candidates: anytype, default_kernels: anytype, k: usize, n: usize) ?@TypeOf(default_kernels) {
    var best: ?@TypeOf(default_kernels) = null;
    const lanes: usize = default_kernels.tuning.lanes;

    for (candidates) |c| {
        if (c.kernels.tuning.lanes != lanes) continue;
        if (k <= c.kernels.tuning.kc and n <= c.kernels.tuning.nc) {
            if (best == null) {
                best = c.kernels;
            } else {
                const cur: @TypeOf(default_kernels) = best.?;
                const cur_area: usize = cur.tuning.kc * cur.tuning.nc;
                const cand_area: usize = c.kernels.tuning.kc * c.kernels.tuning.nc;
                if (cand_area < cur_area) best = c.kernels;
            }
        }
    }

    return best;
}
