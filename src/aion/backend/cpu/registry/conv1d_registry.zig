// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const cpuid = @import("../tuning/cpuid.zig");
const conv1d_k = @import("../kernels/conv1d.zig");

/// Depthwise Conv1D direct-kernel tuning parameters.
///
/// We reuse the kernel module's tuning type so registry candidates can directly
/// refer to `conv1d_k.Kernel(tuning)` instantiations.
pub const Tuning = conv1d_k.DepthwiseConv1DTuning;

pub const Kernels = struct {
    tuning: Tuning,

    run_items: *const fn (ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void,
    run_item_range: *const fn (t: *const conv1d_k.DepthwiseConv1DTask, start: usize, end: usize) void,
};

pub const VariantId = enum {
    baseline,
    k3_unroll_avx2,
    k3_unroll_noavx2,
};

pub const Candidate = struct {
    id: VariantId,
    kernels: Kernels,
};

fn kernelsFor(comptime t: Tuning) Kernels {
    const K = conv1d_k.Kernel(t);
    return .{
        .tuning = t,
        .run_items = K.runItems,
        .run_item_range = K.runItemRange,
    };
}

pub const candidates = [_]Candidate{
    .{ .id = .baseline, .kernels = kernelsFor(.{}) },
    // x86_64 AVX2-ish choice: keep the native lanes (currently 8 on x86_64).
    .{ .id = .k3_unroll_avx2, .kernels = kernelsFor(.{ .unroll_k3 = true, .lanes = 8 }) },
    // x86_64 without AVX2: prefer 4 lanes to avoid forcing wide vectors.
    .{ .id = .k3_unroll_noavx2, .kernels = kernelsFor(.{ .unroll_k3 = true, .lanes = 4 }) },
};

pub fn selectHeuristic(cpu: cpuid.CpuInfo) Candidate {
    // Keep this conservative: only use the unrolled 3-tap specialization on x86_64
    // for now. Pick lanes based on AVX2 availability.
    if (cpu.arch == .x86_64) {
        if (cpu.features.avx2) return candidates[1];
        return candidates[2];
    }
    return candidates[0];
}

comptime {
    _ = std;
}
