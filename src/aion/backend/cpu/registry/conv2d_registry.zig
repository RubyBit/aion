// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const cpuid = @import("../tuning/cpuid.zig");
const conv2d_k = @import("../kernels/conv2d.zig");
const cpu_target = @import("cpu_target.zig");

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

pub const VariantId = cpu_target.SimdWidth;

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
    .{ .id = .simd128, .kernels = kernelsFor(.{ .unroll_3x3 = false, .lanes = 4 }) },
    .{ .id = .simd256, .kernels = kernelsFor(.{ .unroll_3x3 = true, .lanes = 8 }) },
    .{ .id = .simd512, .kernels = kernelsFor(.{ .unroll_3x3 = true, .lanes = 16 }) },
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

pub fn selectHeuristic(cpu: cpuid.CpuInfo) Candidate {
    return selectForTarget(cpu_target.fromCpuInfo(cpu));
}

comptime {
    _ = std;
}
