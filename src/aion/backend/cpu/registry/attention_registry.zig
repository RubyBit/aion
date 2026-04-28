// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const attn_kernels = @import("../kernels/attention.zig");
const cpuid = @import("../tuning/cpuid.zig");
const simd = @import("../kernels/simd.zig");

pub const Tuning = attn_kernels.Tuning;

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

pub const VariantId = enum { baseline, avx2, avx512 };

pub const Candidate = struct {
    id: VariantId,
    kernels: Kernels,
};

fn kernelsFor(comptime t: Tuning) Kernels {
    const K = attn_kernels.Kernel(t);
    return .{
        .tuning = t,
        .pack_q_block = K.packQBlock,
        .pack_k_block = K.packKBlock,
        .calc_scores_f32 = K.calcScoresF32,
        .accumulate_values_f32 = K.accumulateValuesF32,
        .exp_softmax = K.expSoftmax,
    };
}

pub const candidates = [_]Candidate{
    .{ .id = .baseline, .kernels = kernelsFor(.{ .mr = 6, .nr = 16, .mr_acc = 8, .vec_lanes = simd.lanesF32() }) },
    .{ .id = .avx2, .kernels = kernelsFor(.{ .mr = 8, .nr = 16, .mr_acc = 8, .vec_lanes = simd.lanesF32() }) },
    .{ .id = .avx512, .kernels = kernelsFor(.{ .mr = 8, .nr = 32, .mr_acc = 8, .vec_lanes = simd.lanesF32() }) },
};

pub fn selectHeuristic(info: cpuid.CpuInfo) Candidate {
    if (info.arch != .x86_64) return candidates[0];
    if (info.features.avx512_vnni) return candidates[2];
    if (info.features.avx2) return candidates[1];
    return candidates[0];
}

comptime {
    _ = std;
}
