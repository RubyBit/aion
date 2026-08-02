// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const attn_kernels = @import("../kernels/attention.zig");
const cpuid = @import("../tuning/cpuid.zig");
const cpu_target = @import("cpu_target.zig");

/// Tuning for one attention-family kernel variant: the micro-kernel tiles the panel
/// kernels are specialized on, plus the blocking the executors walk with.
///
/// The blocking half is HEURISTICS, measured once — not autotuned, and not derived from
/// the detected cache sizes the way the packed-GEMM `kc`/`nc` are (`matmul_registry`).
/// Measured on a 14-core hybrid laptop CPU (AVX2, 8 bench threads), 2026-07-30. Every
/// candidate takes the defaults below; they are fields rather than module constants so a
/// variant that measures differently (an AVX-512 target with `nr = 32`, say) overrides
/// just the one it needs. To re-tune, the shapes that matter are
/// `--suite kernels --op attn_seq / attn_cached / attn_window / relpos_mha /
/// relpos_chunked` on both backends. The GPU counterparts are deliberately inline in
/// `backend/gpu/exec/attention.zig`, next to the WGSL literals some of them mirror.
pub const Tuning = struct {
    // Micro-kernel tiles. Projected into `attn_kernels.Kernel` by `kernelsFor`.
    mr: usize,
    nr: usize,
    mr_acc: usize,
    vec_lanes: usize,

    // Blocking for the attention-family executors (`exec/attention.zig`,
    // `exec/relpos_mha.zig`).

    /// Query rows per block in the blocked attention path. Bounds the stack row
    /// descriptors, and with `key_block` sets the score-tile footprint
    /// (`row_block * key_block` floats) that should stay resident in L2 alongside the
    /// packed K panel. Also the cap on how many rows of one GQA group are gathered
    /// together, so it wants to be a multiple of `mr`.
    row_block: usize = 32,

    /// Keys per block. Larger amortizes the per-block panel packing over more work;
    /// smaller keeps `[row_block, key_block]` scores plus the K/V panels in L2.
    key_block: usize = 256,

    /// Don't split a row group's key range finer than this when manufacturing work for
    /// idle threads (decode's single query row is the case that needs the split at all).
    /// Below this, per-segment overhead and the log-sum-exp merge dominate.
    min_segment_keys: usize = 256,

    /// Query rows per panel in RelPosMHA. Separate from `row_block` because it also
    /// bounds the `pos_emb` band a panel spans (`window + relpos_panel - 1` table rows,
    /// since consecutive rows' bands are offset by one), so raising it costs `pos_emb`
    /// GEMM width, not just scratch.
    relpos_panel: usize = 64,
};

pub const Kernels = struct {
    tuning: Tuning,

    pack_q_block: *const fn (rows: usize, cols: usize, src: []align(1) const f32, src_stride: usize, dst: []f32) void,
    pack_k_block: *const fn (rows: usize, cols: usize, src: []align(1) const f32, src_stride: usize, dst: []f32) void,

    calc_scores_f32: *const fn (
        m_q: usize,
        n_k: usize,
        k_dim: usize,
        qs: []align(1) const f32,
        q_stride: usize,
        ks: []align(1) const f32,
        k_stride: usize,
        kt: []f32,
        qt: []f32,
        scores: []f32,
        score_stride: usize,
    ) void,

    /// Score path for fewer than `tuning.mr` q rows (decode / GQA groups), where
    /// `calc_scores_f32` would fall back to scalar. Takes a gathered q panel.
    calc_scores_narrow_f32: *const fn (
        m_q: usize,
        n_k: usize,
        k_dim: usize,
        qs: []align(1) const f32,
        q_stride: usize,
        ks: []align(1) const f32,
        k_stride: usize,
        scores: []f32,
        score_stride: usize,
    ) void,

    accumulate_values_f32: *const fn (
        m_tile: usize,
        n_v: usize,
        dv: usize,
        scores: []f32,
        score_stride: usize,
        vs: []align(1) const f32,
        v_stride: usize,
        acc: []f32,
        acc_stride: usize,
    ) void,

    exp_softmax: *const fn (x: f32) f32,

    // Softmax / score post-processing. Dispatched like everything else so they run at
    // the SAME vector width as the GEMMs above; calling the kernels module directly
    // would bake in the build target's width instead of the selected variant's.
    row_max_f32: *const fn (scores: []const f32, m_tile: usize, tn: usize, row_max: []f32) void,
    exp_normalize_scores_f32: *const fn (
        scores: []f32,
        m_tile: usize,
        tn: usize,
        m_new: []const f32,
        l_state: []f32,
    ) void,

    /// Ranged variants: a per-row `[lo, hi)` key window instead of an additive mask.
    row_max_range_f32: *const fn (
        scores: []const f32,
        rows: usize,
        score_stride: usize,
        lo: []const u32,
        hi: []const u32,
        row_max: []f32,
    ) void,
    exp_normalize_range_f32: *const fn (
        scores: []f32,
        rows: usize,
        score_stride: usize,
        tn: usize,
        lo: []const u32,
        hi: []const u32,
        m: []const f32,
        l_state: []f32,
    ) void,
    soft_cap_range_f32: *const fn (
        scores: []f32,
        rows: usize,
        score_stride: usize,
        lo: []const u32,
        hi: []const u32,
        cap: f32,
    ) void,
    rescale_rows_f32: *const fn (
        acc: []f32,
        rows: usize,
        dv: usize,
        acc_stride: usize,
        factors: []const f32,
    ) void,
};

pub const VariantId = cpu_target.SimdWidth;

pub const Candidate = struct {
    id: VariantId,
    kernels: Kernels,
};

fn kernelsFor(comptime t: Tuning) Kernels {
    comptime {
        if (t.row_block == 0 or t.key_block == 0 or t.min_segment_keys == 0 or t.relpos_panel == 0) {
            @compileError("attention blocking must be non-zero");
        }
        // A key block is walked by the panel kernels in `nr`-wide steps, and the row
        // accumulator by `mr_acc`-row steps; a partial trailing step degrades to scalar.
        if ((t.key_block % t.nr) != 0) {
            @compileError("attention tuning.key_block must be a multiple of tuning.nr");
        }
        if ((t.row_block % t.mr_acc) != 0) {
            @compileError("attention tuning.row_block must be a multiple of tuning.mr_acc");
        }
    }

    const K = attn_kernels.Kernel(.{
        .mr = t.mr,
        .nr = t.nr,
        .mr_acc = t.mr_acc,
        .vec_lanes = t.vec_lanes,
    });
    return .{
        .tuning = t,
        .pack_q_block = K.packQBlock,
        .pack_k_block = K.packKBlock,
        .calc_scores_f32 = K.calcScoresF32,
        .calc_scores_narrow_f32 = K.calcScoresNarrowF32,
        .accumulate_values_f32 = K.accumulateValuesF32,
        .exp_softmax = K.expSoftmax,
        .row_max_f32 = K.ranged.rowMaxF32,
        .exp_normalize_scores_f32 = K.ranged.expNormalizeScoresF32,
        .row_max_range_f32 = K.ranged.rowMaxRangeF32,
        .exp_normalize_range_f32 = K.ranged.expNormalizeRangeF32,
        .soft_cap_range_f32 = K.ranged.softCapRangeF32,
        .rescale_rows_f32 = K.ranged.rescaleRowsF32,
    };
}

pub const candidates = [_]Candidate{
    .{ .id = .simd128, .kernels = kernelsFor(.{ .mr = 6, .nr = 16, .mr_acc = 8, .vec_lanes = 4 }) },
    .{ .id = .simd256, .kernels = kernelsFor(.{ .mr = 8, .nr = 16, .mr_acc = 8, .vec_lanes = 8 }) },
    .{ .id = .simd512, .kernels = kernelsFor(.{ .mr = 8, .nr = 32, .mr_acc = 8, .vec_lanes = 16 }) },
};

/// Largest `row_block` across the candidates. The executors' per-row stack descriptors
/// must be comptime-sized; the selected variant's `tuning.row_block` bounds how many of
/// them are actually used.
pub const max_row_block: usize = blk: {
    var m: usize = 0;
    for (candidates) |c| m = @max(m, c.kernels.tuning.row_block);
    break :blk m;
};

fn candidateForId(id: VariantId) Candidate {
    for (candidates) |c| {
        if (c.id == id) return c;
    }
    return candidates[0];
}

pub fn selectForTarget(target: cpu_target.Target) Candidate {
    return candidateForId(target.simd_width);
}

pub fn selectHeuristic(info: cpuid.CpuInfo) Candidate {
    return selectForTarget(cpu_target.fromCpuInfo(info));
}

comptime {
    _ = std;
}
