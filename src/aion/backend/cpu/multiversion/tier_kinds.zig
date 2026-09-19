// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! What a kernel tier implements.
//!
//! Imported by BOTH `build.zig`, which declares the tiers, and `tier_export.zig`,
//! which builds each tier's dispatch table — so the two cannot drift, and neither
//! has to recognise a tier by its name. Keep this file free of imports: `build.zig`
//! compiles it on its own.
//!
//! The numbers are part of the build-option ABI between those two files. Append
//! rather than renumber.

/// The int8 GEMM a tier carries.
pub const QuantGemm = enum(u8) {
    /// No int8 kernel; the quant paths accumulate in f32 at the tier's lane width.
    f32_accumulate = 0,
    /// x86 AVX-VNNI (`vpdpbusd`, VEX encoding).
    avx_vnni = 1,
    /// x86 AVX-512-VNNI (EVEX encoding).
    avx512_vnni = 2,
    /// aarch64 FEAT_DotProd (`sdot`, grouped-by-4 dot).
    dotprod = 3,
    /// aarch64 FEAT_I8MM (`smmla`, int8 2x2 matrix multiply).
    i8mm = 4,
};

/// The f32 GEMM a tier carries.
pub const F32Gemm = enum(u8) {
    /// The portable register-blocked kernel, at the tier's SIMD lane width.
    simd = 0,
    /// aarch64 FEAT_SME (`fmopa`, one outer product per instruction).
    sme = 1,
};
