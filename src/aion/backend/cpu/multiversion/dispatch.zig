// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Runtime selection among the per-tier kernel objects (see `dispatch_table.zig`).
//
// This module is only referenced when the library is built with
// `-Dmultiversion=true` (see `cpu_backend.zig`'s comptime gate). In that build,
// `build.zig` links one object per tier, each defining one of the
// `aion_cpu_kernels_*` accessors declared below. In every other build the
// referencing branch is comptime-dead, so these `extern` symbols are never
// emitted and no tier objects are required.

const builtin = @import("builtin");
const dt = @import("table.zig");
const matmul_shapes = @import("../registry/matmul_shapes.zig");
const cpuid = @import("../tuning/cpuid.zig");
const matmul_registry = @import("../registry/matmul_registry.zig");
const matmul_q_registry = @import("../registry/matmul_q_registry.zig");

const L2_MIN_FOR_MEDIUM: usize = 1 * 1024 * 1024;

// x86 tier accessors (defined by the x86 tier objects). v3 (AVX2 + FMA) is the
// floor: the multiversion main module itself requires AVX2, so a CPU without it
// faults before dispatch ever runs and there is no sub-v3 tier to fall back to.
extern fn aion_cpu_kernels_v3() callconv(.c) *const dt.DispatchTable;
extern fn aion_cpu_kernels_v3_vnni() callconv(.c) *const dt.DispatchTable;
extern fn aion_cpu_kernels_v4() callconv(.c) *const dt.DispatchTable;
// aarch64 tier accessors (defined by the aarch64 tier objects).
extern fn aion_cpu_kernels_arm_baseline() callconv(.c) *const dt.DispatchTable;
extern fn aion_cpu_kernels_arm_dotprod() callconv(.c) *const dt.DispatchTable;
extern fn aion_cpu_kernels_arm_i8mm() callconv(.c) *const dt.DispatchTable;

/// Pick the kernel tier whose ISA the detected CPU supports. Each arch's accessors
/// are only referenced inside its own comptime branch, so the other arch's `extern`
/// symbols are never emitted and need no tier objects.
pub fn selectTable(info: cpuid.CpuInfo) *const dt.DispatchTable {
    if (comptime builtin.cpu.arch.isX86()) {
        // Feature precedence mirrors cpu_target.preferredF32Lanes.
        if (info.features.avx512f or info.features.avx512_vnni) return aion_cpu_kernels_v4();
        if (info.features.avx2 and info.features.avx_vnni) return aion_cpu_kernels_v3_vnni();
        return aion_cpu_kernels_v3();
    } else if (comptime builtin.cpu.arch.isAARCH64()) {
        // NEON f32 width is fixed; the int8 quant path is what varies. FEAT_I8MM
        // (`smmla`) beats FEAT_DotProd (`sdot`) beats the f32-accumulate baseline.
        if (info.features.i8mm) return aion_cpu_kernels_arm_i8mm();
        if (info.features.dotprod) return aion_cpu_kernels_arm_dotprod();
        return aion_cpu_kernels_arm_baseline();
    } else {
        @compileError("kernel_dispatch: unsupported arch");
    }
}

/// Choose the packed-f32 cache-blocking variant for this L2 size.
///
/// Faithful port of `matmul_registry.selectForTarget`'s budget heuristic, but
/// operating on the tier table's three variants (already the right lane width) so
/// it never references the main module's floor-compiled kernels.
pub fn pickMatmul(table: *const dt.DispatchTable, l2_bytes: usize) matmul_registry.F32Kernels {
    if (l2_bytes == 0) return table.matmul[dt.TILE_MEDIUM];

    const budget: usize = matmul_shapes.l2Budget75(l2_bytes);
    var best: matmul_registry.F32Kernels = table.matmul[dt.TILE_SMALL];
    for (table.matmul) |k| {
        const footprint: usize = k.tuning.kc * k.tuning.nc * @sizeOf(f32);
        if (footprint <= budget) best = k;
    }

    const small = table.matmul[dt.TILE_SMALL];
    const small_fp: usize = small.tuning.kc * small.tuning.nc * @sizeOf(f32);
    const best_fp: usize = best.tuning.kc * best.tuning.nc * @sizeOf(f32);
    if (l2_bytes >= L2_MIN_FOR_MEDIUM and best_fp == small_fp) return table.matmul[dt.TILE_MEDIUM];
    return best;
}

/// Quant analogue of `pickMatmul`; uses packed-B footprint as the budget metric,
/// matching `matmul_q_registry.selectForTarget`.
pub fn pickQuant(table: *const dt.DispatchTable, l2_bytes: usize) matmul_q_registry.QuantKernels {
    if (l2_bytes == 0) return table.matmul_q[dt.TILE_MEDIUM];

    const budget: usize = matmul_shapes.l2Budget75(l2_bytes);
    var best: matmul_q_registry.QuantKernels = table.matmul_q[dt.TILE_SMALL];
    for (table.matmul_q) |k| {
        if (k.packed_b_bytes <= budget) best = k;
    }

    const small_fp: usize = table.matmul_q[dt.TILE_SMALL].packed_b_bytes;
    if (l2_bytes >= L2_MIN_FOR_MEDIUM and best.packed_b_bytes == small_fp) return table.matmul_q[dt.TILE_MEDIUM];
    return best;
}
