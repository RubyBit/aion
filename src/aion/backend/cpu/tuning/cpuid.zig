const std = @import("std");
const builtin = @import("builtin");
const x86 = @import("x86.zig");
const arm = @import("arm.zig");

pub const Arch = enum {
    x86_64,
    aarch64,
    unknown,
};

pub const CpuFeatures = struct {
    // x86 Features
    avx2: bool = false,
    avx_vnni: bool = false,
    avx512_vnni: bool = false,
    amx_int8: bool = false,
    // ARM Features
    neon: bool = false,
    dotprod: bool = false,
    i8mm: bool = false,
    sme: bool = false,
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
