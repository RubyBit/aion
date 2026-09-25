// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const builtin = @import("builtin");
const cpuid = @import("../tuning/cpuid.zig");
const types = @import("../../types.zig");
const matmul_q_i8 = @import("../kernels/matmul_q_i8.zig");

/// Architecture-neutral SIMD vector width used by CPU registries.
///
/// These names describe the vector width consumed by a kernel family, not the ISA
/// feature that happened to make that width available on a particular CPU. For
/// example, x86 AVX2 and a future non-x86 256-bit vector target both map to
/// `.simd256`.
pub const SimdWidth = enum {
    simd128,
    simd256,
    simd512,

    pub fn f32Lanes(width: SimdWidth) usize {
        return switch (width) {
            .simd128 => 4,
            .simd256 => 8,
            .simd512 => 16,
        };
    }
};

pub const Target = struct {
    simd_width: SimdWidth,
    preferred_f32_lanes: usize,
    /// The byte dot this CPU has; `portable` where it has none.
    int8_dot: DotEnc,
    /// FEAT_I8MM's `smmla`, which beats the byte dot for a quantized GEMM.
    int8_mm: bool = false,
    caches: cpuid.Caches,
};

pub const DotEnc = matmul_q_i8.DotEnc;

/// Whether this build can emit `enc`'s instructions: each is arch-specific, and
/// the self-hosted x86 assembler cannot encode VPDPBUSD at all. A registry keeps
/// every other encoding out of the build, since instantiating one emits its asm.
pub fn canEmit(comptime enc: DotEnc) bool {
    return switch (enc) {
        .portable => true,
        .sdot => builtin.cpu.arch.isAARCH64(),
        .vex, .evex, .avx2 => builtin.cpu.arch.isX86() and builtin.zig_backend != .stage2_x86_64,
    };
}

/// `enc` if this build can emit it, else the portable dot with the same results.
pub fn emittable(enc: DotEnc) DotEnc {
    return switch (enc) {
        inline else => |e| if (comptime canEmit(e)) e else .portable,
    };
}


pub fn simdWidthFromF32Lanes(lanes: usize) SimdWidth {
    return switch (lanes) {
        16 => .simd512,
        8 => .simd256,
        else => .simd128,
    };
}

/// What this object's own code can use: the width and byte dot it was compiled
/// for. Quantized kernels are chosen from it at comptime, so an object carries only
/// the ones its ISA runs.
pub const compiled: Target = blk: {
    const features = builtin.cpu.features;
    const x86 = builtin.cpu.arch.isX86();
    const arm = builtin.cpu.arch.isAARCH64();
    const lanes: usize = if (x86 and std.Target.x86.featureSetHas(features, .avx512f))
        16
    else if (x86 and std.Target.x86.featureSetHas(features, .avx2))
        8
    else
        4;
    const dot: DotEnc = if (x86 and std.Target.x86.featureSetHas(features, .avx512vnni))
        .evex
    else if (x86 and std.Target.x86.featureSetHas(features, .avxvnni))
        .vex
    else if (x86 and std.Target.x86.featureSetHas(features, .avx2))
        .avx2
    else if (arm and std.Target.aarch64.featureSetHas(features, .dotprod))
        .sdot
    else
        .portable;
    break :blk .{
        .simd_width = simdWidthFromF32Lanes(lanes),
        .preferred_f32_lanes = lanes,
        .int8_dot = emittable(dot),
        .int8_mm = arm and std.Target.aarch64.featureSetHas(features, .i8mm),
        .caches = .{},
    };
};

pub fn fromCpuInfo(info: cpuid.CpuInfo) Target {
    const lanes: usize = cpuid.preferredF32Lanes(info);
    return .{
        .simd_width = simdWidthFromF32Lanes(lanes),
        .preferred_f32_lanes = lanes,
        .int8_dot = emittable(switch (info.arch) {
            .x86_64 => if (info.features.avx512vnni) .evex else if (info.features.avxvnni) .vex else if (info.features.avx2) .avx2 else .portable,
            .aarch64 => if (info.features.dotprod or info.features.i8mm) .sdot else .portable,
            else => .portable,
        }),
        .caches = info.caches,
    };
}
