// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! The matmul autotune menu: tuning policy, separate from the codegen mechanism
//! in `codegen.zig`. The executor benchmarks the eligible configs on-device and
//! caches the fastest per problem shape.

const codegen = @import("codegen.zig");
const MatmulConfig = codegen.MatmulConfig;

/// Config menu. Always keep a scalar (non-vec4) config so every shape has an
/// eligible fallback when vec4 stride alignment fails. `_db` = double-buffered.
/// `_nb` (`bounds_check = false`) variants are admitted only when the executor
/// proves all storage tiles are full and block-aligned.
pub const configs = [_]MatmulConfig{
    // Baseline square blocks.
    .{ .bm = 128, .bn = 128, .bk = 8, .tm = 8, .tn = 8, .vec4_load = false },
    .{ .bm = 128, .bn = 128, .bk = 8, .tm = 8, .tn = 8, .vec4_load = true },
    .{ .bm = 128, .bn = 128, .bk = 16, .tm = 8, .tn = 8, .vec4_load = true },
    .{ .bm = 128, .bn = 128, .bk = 16, .tm = 8, .tn = 8, .vec4_load = true, .double_buffer = true },

    // The 16x4 register tile measured well on NVIDIA through Naga/SPIR-V.
    .{ .bm = 128, .bn = 128, .bk = 16, .tm = 16, .tn = 4, .vec4_load = true, .double_buffer = true },
    .{ .bm = 128, .bn = 128, .bk = 16, .tm = 16, .tn = 4, .vec4_load = true, .double_buffer = true, .bounds_check = false },

    // Smaller blocks: useful fallback / occupancy alternatives.
    .{ .bm = 64, .bn = 64, .bk = 8, .tm = 4, .tn = 4, .vec4_load = true },
    .{ .bm = 64, .bn = 64, .bk = 16, .tm = 4, .tn = 4, .vec4_load = true },
    .{ .bm = 64, .bn = 64, .bk = 32, .tm = 4, .tn = 4, .vec4_load = true },
    .{ .bm = 64, .bn = 64, .bk = 32, .tm = 4, .tn = 4, .vec4_load = true, .double_buffer = true },

    // Rectangular blocks: these were strong on the 4080 laptop WebGPU path.
    .{ .bm = 128, .bn = 64, .bk = 16, .tm = 8, .tn = 4, .vec4_load = true },
    .{ .bm = 128, .bn = 64, .bk = 16, .tm = 8, .tn = 4, .vec4_load = true, .double_buffer = true },
    .{ .bm = 64, .bn = 128, .bk = 16, .tm = 4, .tn = 8, .vec4_load = true },
    .{ .bm = 128, .bn = 256, .bk = 8, .tm = 8, .tn = 16, .vec4_load = true, .double_buffer = true },
    .{ .bm = 256, .bn = 64, .bk = 8, .tm = 8, .tn = 8, .vec4_load = true, .double_buffer = true },
    .{ .bm = 64, .bn = 256, .bk = 8, .tm = 8, .tn = 8, .vec4_load = true, .double_buffer = true },
    .{ .bm = 256, .bn = 128, .bk = 16, .tm = 16, .tn = 4, .vec4_load = true, .double_buffer = true },

    // NOTE: 2D warp-tiled configs (`wm`/`wn`/`wn_iter`, see codegen.emitWarp) are
    // implemented but measured slower than thread-tiled kernels through Naga.
    //
    // NOTE: subgroup outer-product configs (`subgroup = true`, see
    // codegen.emitSubgroup) are implemented and numerically correct, but measured
    // ~40-46% SLOWER than the thread-tiled kernels on the NVIDIA RTX 4080 (Vulkan)
    // FP32 path: best subgroup ~7.6 TFLOP/s vs ~12.4 thread-tiled at 2048^3. The
    // shuffle/broadcast instructions add issue-slot pressure without cutting the
    // FFMA count, and this kernel is FFMA-bound rather than shared-bandwidth-bound,
    // so the trade loses. They are kept out of the active menu here (autotune would
    // otherwise re-benchmark losers on every new shape). Re-add a candidate such as
    //   .{ .bm = 64, .bn = 256, .bk = 16, .tm = 16, .tn = 4, .vec4_load = true, .subgroup = true, .bounds_check = false }
    // when targeting a GPU whose shared-memory/occupancy trade-offs differ (some
    // mobile/Metal parts). Eligibility is gated on a fixed 32-wide subgroup.
};

/// Comptime array pairing each menu config with generated WGSL + entry name.
pub const generated: [configs.len]codegen.Generated = blk: {
    var arr: [configs.len]codegen.Generated = undefined;
    for (configs, 0..) |cfg, i| arr[i] = codegen.gen(cfg);
    break :blk arr;
};
