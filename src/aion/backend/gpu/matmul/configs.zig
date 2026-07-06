// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! The matmul autotune menu: tuning policy, separate from the codegen mechanism
//! in `codegen.zig`. The executor benchmarks the eligible configs on-device and
//! caches the fastest per problem shape.

const std = @import("std");
const codegen = @import("codegen.zig");
const MatmulConfig = codegen.MatmulConfig;

/// Config menu — thread-tiled register-blocked GEMM only (the one kernel family
/// measured competitive on this hardware). Always keep a scalar (non-vec4) config
/// so every shape has an eligible fallback when vec4 stride alignment fails.
/// `_db` = double-buffered. `_nb` (`bounds_check = false`) variants are admitted
/// only when the executor proves all storage tiles are full and block-aligned.
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
};

/// Render every menu config's WGSL into `arena` (called once at backend init). The
/// config set is comptime, so `codegen.gen` still specializes/validates per config;
/// only the WGSL string is built at runtime. Returns an arena-owned slice.
pub fn generate(arena: std.mem.Allocator) []const codegen.Generated {
    var arr = arena.alloc(codegen.Generated, configs.len) catch @panic("codegen: OOM");
    inline for (configs, 0..) |cfg, i| arr[i] = codegen.gen(arena, cfg);
    return arr;
}
