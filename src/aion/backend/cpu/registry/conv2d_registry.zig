// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const cpuid = @import("../tuning/cpuid.zig");
const conv2d_k = @import("../kernels/conv2d.zig");

/// Depthwise Conv2D direct-kernel tuning parameters.
///
/// We reuse the kernel module's tuning type so registry candidates can directly
/// refer to `conv2d_k.Kernel(tuning)` instantiations.
pub const Tuning = conv2d_k.DepthwiseConv2DTuning;

pub const Kernels = struct {
    tuning: Tuning,

    run_items: *const fn (ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void,
    run_item_range: *const fn (t: *const conv2d_k.DepthwiseConv2DTask, start: usize, end: usize) void,
};

pub const VariantId = enum {
    baseline,
    avx2,
    avx512,
};

pub const Candidate = struct {
    id: VariantId,
    kernels: Kernels,
};

fn kernelsFor(comptime t: Tuning) Kernels {
    const K = conv2d_k.Kernel(t);
    return .{
        .tuning = t,
        .run_items = K.runItems,
        .run_item_range = K.runItemRange,
    };
}

pub const candidates = [_]Candidate{
    .{ .id = .baseline, .kernels = kernelsFor(.{ .unroll_3x3 = false, .lanes = 4 }) },
    .{ .id = .avx2, .kernels = kernelsFor(.{ .unroll_3x3 = true, .lanes = 8 }) },
    .{ .id = .avx512, .kernels = kernelsFor(.{ .unroll_3x3 = true, .lanes = 16 }) },
};

pub fn selectHeuristic(cpu: cpuid.CpuInfo) Candidate {
    const lanes: usize = cpuid.preferredF32Lanes(cpu);
    return switch (lanes) {
        16 => candidates[2],
        8 => candidates[1],
        else => candidates[0],
    };
}

comptime {
    _ = std;
}
