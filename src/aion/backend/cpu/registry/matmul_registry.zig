// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const types = @import("../../types.zig");
const builtin = @import("builtin");
const matmul = @import("../kernels/matmul.zig");
const matmul_sme = @import("../kernels/matmul_sme.zig");
const cpuid = @import("../tuning/cpuid.zig");
const cpu_target = @import("cpu_target.zig");
const matmul_shapes = @import("matmul_shapes.zig");

/// How a kernel wants A's packed panels laid out.
///
/// A broadcast-FMA micro-kernel walks one row's k values, so it wants them
/// adjacent. An outer-product kernel consumes one k across all `mr` rows at once,
/// so it wants THOSE adjacent — and a caller gathering into that layout writes
/// whole cache lines instead of feeding a separate corner turn.
pub const PackedALayout = enum {
    /// `[mr][kc]`: a row's k values are contiguous.
    row_major,
    /// `[kc][mr]`: a k's rows are contiguous.
    k_major,
};

pub const Tuning = struct {
    // Micro-kernel tiles
    mr: usize,
    nr: usize,

    /// The layout `pack_a_tile` writes and `matmul_packed_ab` reads.
    a_layout: PackedALayout = .row_major,

    // SIMD lane width used by the tuned kernel.
    lanes: usize,

    // Cache blocking params.
    kc: usize,
    mc: usize,
    nc: usize,
};

pub const F32Kernels = struct {
    tuning: Tuning,

    scratch_bytes: usize,
    scratch_alignment: usize,

    pack_b_tile: *const fn (scratch_bytes: []u8, k: usize, n: usize, ldb: usize, b_bytes: []const u8) types.BackendError!void,
    pack_a_tile: *const fn (k: usize, m: usize, a_bytes: []const u8, packed_a_out: []align(32) f32) types.BackendError!void,
    pack_b_tile_f16_to_packed_f32: *const fn (packed_b: []align(32) f32, k: usize, n: usize, ldb: usize, b_bytes: []const u8) types.BackendError!void,
    pack_a_tile_f16_to_packed_f32: *const fn (packed_a_out: []align(32) f32, m: usize, k: usize, a_bytes: []const u8) types.BackendError!void,
    matmul_packed_b: *const fn (scratch_bytes: []u8, packed_b_view: []align(32) const f32, params: types.MatMulParams, c_bytes: []u8, a_bytes: []const u8) types.BackendError!void,
    matmul_packed_ab: *const fn (packed_a: []align(32) const f32, packed_b_view: []align(32) const f32, params: types.MatMulParams, c_bytes: []u8) types.BackendError!void,

    /// A GEMM whose A operand is described by pointers rather than packed into a
    /// panel: one pointer per (row panel, reduction index), panel-major, each
    /// addressing `mr` consecutive floats.
    ///
    /// This is for callers whose A does not exist contiguously in memory and
    /// would be expensive to materialise — a conv's im2col operand is a
    /// `k_h*k_w`-times redundant view of its activations, so gathering it costs
    /// far more than the addressing does.
    ///
    /// Null on kernels that cannot take A this way, which is every kernel whose
    /// micro-kernel wants A row-major.
    matmul_indirect: ?*const fn (a_tables: []const [*]const f32, packed_b_view: []align(32) const f32, params: types.MatMulParams, c_bytes: []u8) types.BackendError!void = null,
};

/// An SME-backed kernel set at this cache-blocking size. `lanes` stays the NEON
/// width: it describes the elementwise helpers (bias add), not the GEMM.
pub fn smeKernels(comptime kc: usize, comptime mc: usize, comptime nc: usize, comptime lanes: usize) F32Kernels {
    const K = matmul_sme.Kernel(.{ .kc = kc, .mc = mc, .nc = nc });
    return .{
        .tuning = .{ .mr = matmul_sme.MR, .nr = matmul_sme.NR, .a_layout = .k_major, .lanes = lanes, .kc = kc, .mc = mc, .nc = nc },
        .scratch_bytes = K.scratchBytes(),
        .scratch_alignment = K.ScratchAlignment,
        .pack_b_tile = K.packBTileF32,
        .pack_a_tile = K.packATileF32,
        .pack_b_tile_f16_to_packed_f32 = K.packBTileF16ToPackedF32,
        .pack_a_tile_f16_to_packed_f32 = K.packATileF16ToPackedF32,
        .matmul_packed_b = K.matmulF32PackedB,
        .matmul_packed_ab = K.matmulF32PackedAB,
        .matmul_indirect = K.matmulF32Indirect,
    };
}

/// Only instantiated when the module is compiled for a target with FEAT_SME.
///
/// Deeper K blocks than the SIMD kernel's: an outer product reuses a loaded
/// vector across a whole 32x32 tile, so the cost that dominates is re-walking C
/// and re-packing A once per K block, and a deeper block means fewer of both.
/// Measured on VGG-19 at one thread: 164 ms at kc=128, 116 at 256, 102 at 512,
/// 95 at 1024, 97 at 2048; re-swept once the bias pass stopped dominating, 768
/// edged out 1024.
pub const sme_candidates = if (matmul_sme.compiledFor()) [_]Candidate{
    .{ .id = .small, .kernels = smeKernels(256, 160, 128, 4) },
    .{ .id = .medium, .kernels = smeKernels(512, 288, 256, 4) },
    .{ .id = .large, .kernels = smeKernels(768, 288, 512, 4) },
} else [_]Candidate{};

pub const VariantId = matmul_shapes.PackedVariantId;

pub const Candidate = struct {
    id: VariantId,
    kernels: F32Kernels,
};

fn kernelsFor(comptime t: Tuning) F32Kernels {
    const K = matmul.Kernel(.{ .kc = t.kc, .mc = t.mc, .nc = t.nc, .mr = t.mr, .nr = t.nr, .lanes = t.lanes });
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
    return matmul_shapes.candidateFor(candidates, id, lanes);
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
    return matmul_shapes.selectSmallestCoveringKernels(candidates, default_kernels, k, n);
}

pub fn selectForConvOcTile(default_kernels: F32Kernels, oc_count: usize) F32Kernels {
    // Conv implicit-GEMM packs weights by OC tiles (N dimension). For small N,
    // reducing NC can shrink packed-B panels and scratch footprint significantly.
    if (oc_count == 0) return default_kernels;

    return selectForTile(default_kernels, default_kernels.tuning.kc, oc_count) orelse default_kernels;
}

pub fn selectForTarget(target: cpu_target.Target) Candidate {
    // One `fmopa` is a whole outer product, so when the CPU has SME it replaces
    // the candidate list outright rather than competing inside it.
    if (sme_candidates.len != 0 and matmul_sme.usable()) {
        return pickForL2(sme_candidates, target.preferred_f32_lanes, target.caches.l2_bytes);
    }
    return pickForL2(candidates, target.preferred_f32_lanes, target.caches.l2_bytes);
}

/// Largest variant whose packed-B tile still fits L2, at this lane width.
///
/// L2 is typically per-core and often only 1-2MiB. Requiring the whole KC*NC
/// panel to fit in a small fraction of it selects variants that don't even cover
/// the program tiler (e.g. tk=256), so this:
/// - defaults to `medium` when caches are undetectable,
/// - otherwise takes the largest candidate whose B-panel fits ~75% of L2,
/// - but never returns `small` on a reasonably sized L2 (>= 1MiB).
fn pickForL2(comptime list: anytype, lanes: usize, l2_bytes: usize) Candidate {
    if (l2_bytes == 0) return matmul_shapes.candidateFor(list, .medium, lanes);

    const budget: usize = matmul_shapes.l2Budget75(l2_bytes);
    var best: Candidate = matmul_shapes.candidateFor(list, .small, lanes);
    for (list) |c| {
        if (c.kernels.tuning.lanes != lanes) continue;
        const footprint: usize = c.kernels.tuning.kc * c.kernels.tuning.nc * @sizeOf(f32);
        if (footprint <= budget) best = c;
    }

    if (l2_bytes >= (1 * 1024 * 1024) and best.id == .small) return matmul_shapes.candidateFor(list, .medium, lanes);
    return best;
}

pub fn selectHeuristic(info: cpuid.CpuInfo) Candidate {
    return selectForTarget(cpu_target.fromCpuInfo(info));
}
