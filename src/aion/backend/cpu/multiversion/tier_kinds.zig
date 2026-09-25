// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! The kernel tiers: what each needs from the CPU and what it implements.
//!
//! Imported by `build.zig`, which compiles one object per tier, by `tier_export.zig`,
//! which builds each tier's dispatch table, and by `dispatch.zig`, which picks one
//! at runtime — so the three cannot drift, and none has to recognise a tier by its
//! name. Keep this file free of imports: `build.zig` compiles it on its own.
//!
//! The numbers are part of the build-option ABI between those two files. Append
//! rather than renumber.

/// The byte-dot instruction a tier's quantized kernels use.
pub const QuantGemm = enum(u8) {
    /// No byte-dot instruction: int8 kernels use the portable dot, same results.
    portable = 0,
    /// x86 AVX-VNNI (`vpdpbusd`, VEX encoding).
    avx_vnni = 1,
    /// x86 AVX-512-VNNI (EVEX encoding).
    avx512_vnni = 2,
    /// aarch64 FEAT_DotProd (`sdot`, grouped-by-4 dot).
    dotprod = 3,
    /// aarch64 FEAT_I8MM (`smmla`, int8 2x2 matrix multiply).
    i8mm = 4,
    /// x86 AVX2 byte multiplies (`vpmaddubsw` + `vpmaddwd`), below VNNI.
    avx2 = 5,
};

/// The f32 GEMM a tier carries.
pub const F32Gemm = enum(u8) {
    /// The portable register-blocked kernel, at the tier's SIMD lane width.
    simd = 0,
    /// aarch64 FEAT_SME (`fmopa`, one outer product per instruction).
    sme = 1,
};

/// A CPU feature a tier is compiled for, under the compiler's own name so
/// `build.zig` passes it straight through and dispatch checks the same set.
pub const Feature = enum {
    avx512f,
    avx512bw,
    avx512cd,
    avx512dq,
    avx512vl,
    avxvnni,
    avx512vnni,
    dotprod,
    i8mm,
    sme,
    sme2,
};

/// One kernel object: what it needs beyond the library's own floor, and what it
/// carries. A tier is compiled for exactly `requires` and selected only when the
/// CPU reports every one of them, so the two can no longer disagree.
pub const Tier = struct {
    name: []const u8,
    requires: []const Feature,
    lanes: u32,
    quant: QuantGemm,
    f32_gemm: F32Gemm = .simd,
    /// x86 only: the CPU model to schedule for, when not the floor's. It must add
    /// no instruction set beyond `requires`, only tuning.
    x86_model: []const u8 = "x86_64_v3",
};

/// Best first: dispatch takes the first tier the CPU can run. The floor is
/// x86-64-v3, which the library itself needs, so the last tier requires nothing.
pub const x86_tiers = [_]Tier{
    .{ .name = "v4", .requires = &.{ .avx512f, .avx512bw, .avx512cd, .avx512dq, .avx512vl, .avx512vnni }, .lanes = 16, .quant = .avx512_vnni, .x86_model = "x86_64_v4" },
    .{ .name = "v3_vnni", .requires = &.{.avxvnni}, .lanes = 8, .quant = .avx_vnni },
    .{ .name = "v3", .requires = &.{}, .lanes = 8, .quant = .avx2 },
};

/// Best first, over the ARMv8.2-A floor. The SME f32 GEMM loads with SME2's
/// multi-vector `ld1w`, so the tier needs both; it keeps `smmla` for int8.
pub const arm_tiers = [_]Tier{
    .{ .name = "arm_sme", .requires = &.{ .dotprod, .i8mm, .sme, .sme2 }, .lanes = 4, .quant = .i8mm, .f32_gemm = .sme },
    .{ .name = "arm_i8mm", .requires = &.{ .dotprod, .i8mm }, .lanes = 4, .quant = .i8mm },
    .{ .name = "arm_dotprod", .requires = &.{.dotprod}, .lanes = 4, .quant = .dotprod },
    .{ .name = "arm_baseline", .requires = &.{}, .lanes = 4, .quant = .portable },
};

/// Index of the best tier in `list` that `cpu` can run, where `cpu.has(f)` says
/// whether a feature is present. The last tier requires nothing.
pub fn best(list: []const Tier, cpu: anytype) usize {
    next: for (list, 0..) |tier, i| {
        for (tier.requires) |f| if (!cpu.has(f)) continue :next;
        return i;
    }
    return list.len - 1;
}
