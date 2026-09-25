// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const builtin = @import("builtin");
const x86 = @import("x86.zig");
const arm = @import("arm.zig");
const tier_kinds = @import("../multiversion/tier_kinds.zig");

pub const Arch = enum {
    x86_64,
    aarch64,
    unknown,
};

/// Field names follow `tier_kinds.Feature` where a tier can require them, so
/// `has` checks any tier's list without a translation table.
pub const CpuFeatures = struct {
    // x86. Each is reported only when the OS also saves the registers it uses.
    avx2: bool = false,
    avxvnni: bool = false,
    avx512f: bool = false,
    avx512bw: bool = false,
    avx512cd: bool = false,
    avx512dq: bool = false,
    avx512vl: bool = false,
    avx512vnni: bool = false,
    amx_int8: bool = false,
    // ARM
    neon: bool = false,
    dotprod: bool = false,
    i8mm: bool = false,
    sme: bool = false,
    sme2: bool = false,

    pub fn has(self: CpuFeatures, f: tier_kinds.Feature) bool {
        return switch (f) {
            inline else => |t| @field(self, @tagName(t)),
        };
    }
};

pub const Caches = struct {
    l1d_bytes: usize = 0,
    l2_bytes: usize = 0,
    l3_bytes: usize = 0,
};

pub const CpuInfo = struct {
    arch: Arch = .unknown,
    features: CpuFeatures = .{},
    caches: Caches = .{},
    /// Logical processors visible to the OS (best-effort).
    logical_processors: usize = 0,
    /// Physical core count (best-effort; 0 when unknown).
    physical_cores: usize = 0,
};

pub fn detect() CpuInfo {
    if (builtin.cpu.arch.isX86()) {
        return x86.detect();
    } else if (builtin.cpu.arch.isAARCH64()) {
        return arm.detect();
    }
    return .{};
}

/// Best-effort preferred SIMD lane width for f32 kernels on the detected CPU.
///
/// Registries use this as their lane-group source of truth instead of relying on
/// compile-time architecture defaults.
pub fn preferredF32Lanes(info: CpuInfo) usize {
    return switch (info.arch) {
        .x86_64 => if (info.features.avx512f)
            16
        else if (info.features.avx2)
            8
        else
            4,
        .aarch64 => 4,
        else => 4,
    };
}

test "a tier is chosen only when the CPU has every feature it was compiled for" {
    const arm_t = tier_kinds.arm_tiers;
    const x86_t = tier_kinds.x86_tiers;
    const name = struct {
        fn of(list: []const tier_kinds.Tier, f: CpuFeatures) []const u8 {
            return list[tier_kinds.best(list, f)].name;
        }
    }.of;
    // SME without SME2 cannot run the SME kernel's multi-vector loads.
    try std.testing.expectEqualStrings("arm_i8mm", name(&arm_t, .{ .dotprod = true, .i8mm = true, .sme = true }));
    try std.testing.expectEqualStrings("arm_sme", name(&arm_t, .{ .dotprod = true, .i8mm = true, .sme = true, .sme2 = true }));
    try std.testing.expectEqualStrings("arm_baseline", name(&arm_t, .{}));
    // AVX-512F alone (Knights Landing) lacks the BW/DQ/VL the v4 object uses.
    try std.testing.expectEqualStrings("v3", name(&x86_t, .{ .avx2 = true, .avx512f = true, .avx512cd = true }));
    try std.testing.expectEqualStrings("v4", name(&x86_t, .{ .avx2 = true, .avx512f = true, .avx512bw = true, .avx512cd = true, .avx512dq = true, .avx512vl = true, .avx512vnni = true }));
}
