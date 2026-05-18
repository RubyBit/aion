// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const cpuid = @import("../tuning/cpuid.zig");

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
    caches: cpuid.Caches,
};

pub fn simdWidthFromF32Lanes(lanes: usize) SimdWidth {
    return switch (lanes) {
        16 => .simd512,
        8 => .simd256,
        else => .simd128,
    };
}

pub fn fromCpuInfo(info: cpuid.CpuInfo) Target {
    const lanes: usize = cpuid.preferredF32Lanes(info);
    return .{
        .simd_width = simdWidthFromF32Lanes(lanes),
        .preferred_f32_lanes = lanes,
        .caches = info.caches,
    };
}
