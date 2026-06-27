// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Per-tier kernel object root.
//
// `build.zig` compiles this single source file once per CPU feature tier (with a
// distinct `-target` cpu model and `tier_options` injected per tier), producing
// one object per tier. Each object materializes a `DispatchTable` specialized to
// its tier's SIMD lane width and exports a uniquely-named accessor for it.
//
// Because `tier_options.lanes` is comptime-known, the table is built entirely at
// comptime: `matmul_shapes.candidateFor` / each registry's `selectForTarget` are
// evaluated at comptime and reference *only* the kernel instantiations for this
// tier's lane width — so the object stays lean (no other-width kernels emitted),
// and the lane-width kernels are lowered against this object's full ISA (e.g. the
// v4 object emits real AVX-512).

const opts = @import("tier_options");
const dt = @import("table.zig");
const cpu_target = @import("../registry/cpu_target.zig");
const matmul_shapes = @import("../registry/matmul_shapes.zig");
const matmul_registry = @import("../registry/matmul_registry.zig");
const matmul_q_registry = @import("../registry/matmul_q_registry.zig");
const matmul_nt_registry = @import("../registry/matmul_nt_registry.zig");
const matvec_registry = @import("../registry/matvec_registry.zig");
const attention_registry = @import("../registry/attention_registry.zig");
const relpos_mha_registry = @import("../registry/relpos_mha_registry.zig");
const conv1d_registry = @import("../registry/conv1d_registry.zig");
const conv2d_registry = @import("../registry/conv2d_registry.zig");
const fft_registry = @import("../registry/fft_registry.zig");
const matmul_q_i8 = @import("../kernels/matmul_q_i8.zig");

const lanes: usize = opts.lanes;

/// int8 GEMM encoding for this tier:
///   0 = none (f32-accumulate quant kernels), 1 = AVX-VNNI (VEX), 2 = AVX-512-VNNI
///   (EVEX), 3 = aarch64 FEAT_DotProd (sdot), 4 = aarch64 FEAT_I8MM (smmla).
const quant_enc_raw: u8 = opts.quant_enc;

/// Build a `QuantKernels` backed by the int8 VNNI/sdot GEMM (`Kernel`) for this enc.
fn vnniQuant(comptime kc: usize, comptime nc: usize, comptime enc: matmul_q_i8.DotEnc) matmul_q_registry.QuantKernels {
    const K = matmul_q_i8.Kernel(.{ .kc = kc, .nc = nc, .enc = enc });
    return .{
        .tuning = .{ .mr = matmul_q_i8.MR, .nr = matmul_q_i8.NR, .lanes = matmul_q_i8.NR, .kc = kc, .mc = 0, .nc = nc },
        .scratch_bytes = K.scratchBytes(),
        .scratch_alignment = K.ScratchAlignment,
        .packed_b_bytes = K.packedBBytes(),
        .pack_b_tile_q4_0 = K.packBTileQ4_0,
        .pack_b_tile_q8_0 = K.packBTileQ8_0,
        .matmul_packed_b = K.matmulPackedB,
    };
}

/// Build a `QuantKernels` backed by the FEAT_I8MM `smmla` GEMM (`KernelMM`).
fn mmQuant(comptime kc: usize, comptime nc: usize, comptime enc: matmul_q_i8.MmEnc) matmul_q_registry.QuantKernels {
    const K = matmul_q_i8.KernelMM(.{ .kc = kc, .nc = nc, .enc = enc });
    return .{
        .tuning = .{ .mr = matmul_q_i8.MM_MR, .nr = matmul_q_i8.MM_NR, .lanes = matmul_q_i8.MM_NR, .kc = kc, .mc = 0, .nc = nc },
        .scratch_bytes = K.scratchBytes(),
        .scratch_alignment = K.ScratchAlignment,
        .packed_b_bytes = K.packedBBytes(),
        .pack_b_tile_q4_0 = K.packBTileQ4_0,
        .pack_b_tile_q8_0 = K.packBTileQ8_0,
        .matmul_packed_b = K.matmulPackedB,
    };
}

/// The tier's three packed-GEMM quant variants (small/medium/large). VNNI/sdot tiers
/// use the int8 dot kernel; the i8mm tier uses the `smmla` matrix kernel; others keep
/// the f32-accumulate kernels at their lane width.
const quant_table: [3]matmul_q_registry.QuantKernels = if (quant_enc_raw == 0) .{
    matmul_shapes.candidateFor(matmul_q_registry.candidates, .small, lanes).kernels,
    matmul_shapes.candidateFor(matmul_q_registry.candidates, .medium, lanes).kernels,
    matmul_shapes.candidateFor(matmul_q_registry.candidates, .large, lanes).kernels,
} else if (quant_enc_raw == 4) .{
    mmQuant(128, 128, .smmla),
    mmQuant(256, 256, .smmla),
    mmQuant(512, 512, .smmla),
} else blk: {
    const enc: matmul_q_i8.DotEnc = switch (quant_enc_raw) {
        2 => .evex, // x86 AVX-512-VNNI
        3 => .sdot, // aarch64 FEAT_DotProd
        else => .vex, // x86 AVX-VNNI
    };
    break :blk .{
        vnniQuant(128, 128, enc),
        vnniQuant(256, 256, enc),
        vnniQuant(512, 512, enc),
    };
};

/// Synthetic target used to drive the comptime registry selectors. The lane width
/// is fixed by the tier; caches are irrelevant here because the packed-GEMM tile
/// (L2-budget) decision is deferred to the main module at runtime.
const tier_target = cpu_target.Target{
    .simd_width = cpu_target.simdWidthFromF32Lanes(lanes),
    .preferred_f32_lanes = lanes,
    .caches = .{},
};

const table: dt.DispatchTable = .{
    .abi_version = dt.ABI_VERSION,
    .lanes = @intCast(lanes),

    .matmul = .{
        matmul_shapes.candidateFor(matmul_registry.candidates, .small, lanes).kernels,
        matmul_shapes.candidateFor(matmul_registry.candidates, .medium, lanes).kernels,
        matmul_shapes.candidateFor(matmul_registry.candidates, .large, lanes).kernels,
    },
    .matmul_q = quant_table,

    .matmul_nt = matmul_nt_registry.selectForTarget(tier_target).kernels,
    .matvec = matvec_registry.selectForTarget(tier_target).kernels,
    .attention = attention_registry.selectForTarget(tier_target).kernels,
    .relpos_mha = relpos_mha_registry.selectForTarget(tier_target).kernels,
    .conv1d = conv1d_registry.selectForTarget(tier_target).kernels,
    .conv2d = conv2d_registry.selectForTarget(tier_target).kernels,
    .fft = fft_registry.selectForTarget(tier_target).kernels,
};

fn accessor() callconv(.c) *const dt.DispatchTable {
    return &table;
}

comptime {
    @export(&accessor, .{ .name = "aion_cpu_kernels_" ++ opts.tier_name, .linkage = .strong });
}
