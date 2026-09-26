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
const types = @import("../../types.zig");
const dt = @import("table.zig");
const cpu_target = @import("../registry/cpu_target.zig");
const matmul_shapes = @import("../registry/matmul_shapes.zig");
const matmul_registry = @import("../registry/matmul_registry.zig");
const matmul_q_registry = @import("../registry/matmul_q_registry.zig");
const matmul_nt_registry = @import("../registry/matmul_nt_registry.zig");
const matvec_registry = @import("../registry/matvec_registry.zig");
const attention_registry = @import("../registry/attention_registry.zig");
const conv2d_registry = @import("../registry/conv2d_registry.zig");
const fft_registry = @import("../registry/fft_registry.zig");
const tier_kinds = @import("tier_kinds.zig");

const lanes: usize = opts.lanes;

/// Which kernels this tier carries, as `build.zig` declared them.
const quant_gemm: tier_kinds.QuantGemm = @enumFromInt(opts.quant_enc);
const f32_gemm: tier_kinds.F32Gemm = @enumFromInt(opts.f32_gemm);

/// The tier's three packed f32 GEMM variants (small/medium/large).
const matmul_table: [3]matmul_registry.F32Kernels = if (f32_gemm == .sme) .{
    matmul_registry.smeKernels(128, 160, 128, lanes),
    matmul_registry.smeKernels(256, 160, 256, lanes),
    matmul_registry.smeKernels(512, 288, 512, lanes),
} else .{
    matmul_shapes.candidateFor(matmul_registry.candidates, .small, lanes).kernels,
    matmul_shapes.candidateFor(matmul_registry.candidates, .medium, lanes).kernels,
    matmul_shapes.candidateFor(matmul_registry.candidates, .large, lanes).kernels,
};

/// Synthetic target used to drive the comptime registry selectors. The lane width
/// is fixed by the tier; caches are irrelevant here because the packed-GEMM tile
/// (L2-budget) decision is deferred to the main module at runtime.
fn tierTarget() cpu_target.Target {
    return .{
        .simd_width = cpu_target.simdWidthFromF32Lanes(lanes),
        .preferred_f32_lanes = lanes,
        .int8_dot = switch (quant_gemm) {
            .avx_vnni => .vex,
            .avx512_vnni => .evex,
            .avx2 => .avx2,
            .dotprod, .i8mm => .sdot,
            .portable => .portable,
        },
        .int8_mm = quant_gemm == .i8mm,
        .caches = .{},
    };
}

const tier_target = tierTarget();

const table: dt.DispatchTable = .{
    .abi_version = dt.ABI_VERSION,
    .lanes = @intCast(lanes),

    .matmul = matmul_table,
    .quantized = dt.quantizedFor(tier_target),

    .attention = attention_registry.selectForTarget(tier_target).kernels,
    .relpos_mha = attention_registry.selectForTarget(tier_target).kernels,
    .conv2d = conv2d_registry.selectForTarget(tier_target).kernels,
    .fft = fft_registry.selectForTarget(tier_target).kernels,
};

fn accessor() callconv(.c) *const dt.DispatchTable {
    return &table;
}

comptime {
    @export(&accessor, .{ .name = "aion_cpu_kernels_" ++ opts.tier_name, .linkage = .strong });
}
