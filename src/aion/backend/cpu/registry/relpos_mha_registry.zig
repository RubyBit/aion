// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Registry for the relative-positional multi-head attention (`RelPosMHA`) op.
//
// RelPosMHA reuses the same SIMD GEMM/softmax kernels as the plain attention op
// (`kernels/attention.zig`), selected per detected SIMD lane width. The op-level
// novelty (the `(q+v)·pos_emb` term + rel-shift combine) lives in the executor;
// the hot inner loops are the shared, tuned attention kernels. Keeping a dedicated
// registry namespace mirrors `attention_registry` so the node is wired through the
// same kernel-selection machinery as every other tiled op.
const std = @import("std");
const attn_kernels = @import("../kernels/attention.zig");
const cpuid = @import("../tuning/cpuid.zig");
const cpu_target = @import("cpu_target.zig");

pub const Tuning = attn_kernels.Tuning;

pub const Kernels = struct {
    tuning: Tuning,

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
};

pub const VariantId = cpu_target.SimdWidth;

pub const Candidate = struct {
    id: VariantId,
    kernels: Kernels,
};

fn kernelsFor(comptime t: Tuning) Kernels {
    const K = attn_kernels.Kernel(t);
    return .{
        .tuning = t,
        .pack_k_block = K.packKBlock,
        .calc_scores_f32 = K.calcScoresF32,
        .accumulate_values_f32 = K.accumulateValuesF32,
        .exp_softmax = K.expSoftmax,
    };
}

pub const candidates = [_]Candidate{
    .{ .id = .simd128, .kernels = kernelsFor(.{ .mr = 6, .nr = 16, .mr_acc = 8, .vec_lanes = 4 }) },
    .{ .id = .simd256, .kernels = kernelsFor(.{ .mr = 8, .nr = 16, .mr_acc = 8, .vec_lanes = 8 }) },
    .{ .id = .simd512, .kernels = kernelsFor(.{ .mr = 8, .nr = 32, .mr_acc = 8, .vec_lanes = 16 }) },
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
