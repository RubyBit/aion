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

const L2_MIN_FOR_MEDIUM: usize = 1 * 1024 * 1024;

const tier_kinds = @import("tier_kinds.zig");

const tiers = if (builtin.cpu.arch.isX86())
    &tier_kinds.x86_tiers
else if (builtin.cpu.arch.isAARCH64())
    &tier_kinds.arm_tiers
else
    @compileError("kernel_dispatch: unsupported arch");

/// The best tier the CPU can run (`tier_kinds.best`). Only this arch's accessors
/// are referenced, so the other arch's objects are never needed.
pub fn selectTable(info: cpuid.CpuInfo) *const dt.DispatchTable {
    const chosen = tier_kinds.best(tiers, info.features);
    inline for (tiers, 0..) |tier, i| {
        if (i == chosen) {
            const accessor = @extern(*const fn () callconv(.c) *const dt.DispatchTable, .{ .name = "aion_cpu_kernels_" ++ tier.name });
            return accessor();
        }
    }
    unreachable;
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
