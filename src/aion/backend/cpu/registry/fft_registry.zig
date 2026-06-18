// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Lane-width selection for the real-FFT kernel, mirroring matmul_registry.
//
// The FFT core (`kernels/fft.zig`) vectorizes across frames, so the only tuning
// knob is the SIMD lane width. We expose one candidate per width and pick the
// one matching the detected CPU's preferred f32 lane count.

const cpuid = @import("../tuning/cpuid.zig");
const cpu_target = @import("cpu_target.zig");
const fft = @import("../kernels/fft.zig");

pub const FftKernels = fft.FftKernels;

pub const Candidate = struct {
    lanes: usize,
    kernels: FftKernels,
};

pub const candidates = [_]Candidate{
    .{ .lanes = 4, .kernels = fft.kernels(4) },
    .{ .lanes = 8, .kernels = fft.kernels(8) },
    .{ .lanes = 16, .kernels = fft.kernels(16) },
};

fn candidateForLanes(lanes: usize) Candidate {
    for (candidates) |c| {
        if (c.lanes == lanes) return c;
    }
    // Fall back to the 8-wide variant; the compiler splits/scalarizes as needed.
    return candidates[1];
}

pub fn selectForTarget(target: cpu_target.Target) Candidate {
    return candidateForLanes(target.preferred_f32_lanes);
}

pub fn selectHeuristic(info: cpuid.CpuInfo) Candidate {
    return selectForTarget(cpu_target.fromCpuInfo(info));
}
