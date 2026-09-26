// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Registry naming convention (CPU kernels):
//   * One registry file per operation. `matmul_nt_registry` and `matvec_registry`
//     bundle f32 + quant in a single `Kernels` struct because those kernels share a
//     call shape.
//   * Packed GEMM is the exception: f32 (`matmul_registry`) and quant
//     (`matmul_q_registry`, here) have genuinely different pack/scratch/packed-B
//     ABIs, so they stay in separate files with distinct `*Kernels` structs.
//   * ISA is never in a filename — it's a comptime parameter (e.g. `matmul_q_i8`'s
//     `DotEnc`), selected at runtime via the multiversion tier dispatch.
const std = @import("std");
const types = @import("../../types.zig");
const matmul_q_i8 = @import("../kernels/matmul_q_i8.zig");
const cpuid = @import("../tuning/cpuid.zig");
const cpu_target = @import("cpu_target.zig");
const matmul_shapes = @import("matmul_shapes.zig");

pub const Tuning = struct {
    // Micro-kernel tiles
    mr: usize,
    nr: usize,

    // SIMD lane width used by the tuned kernel.
    lanes: usize,

    // Cache blocking params.
    kc: usize,
    mc: usize,
    nc: usize,
};

pub const PackedBView = []align(32) const u8;

pub const PackBFn = *const fn (scratch_bytes: []u8, k: usize, n: usize, ldb: usize, b_bytes: []const u8) types.BackendError!void;
pub const MatMulPackedBFn = *const fn (scratch_bytes: []u8, packed_b_view: PackedBView, params: types.MatMulParams, c_bytes: []u8, a_bytes: []const u8) types.BackendError!void;

pub const QuantKernels = struct {
    tuning: Tuning,

    scratch_bytes: usize,
    scratch_alignment: usize,

    packed_b_bytes: usize,

    pack_b_tile_q4_0: PackBFn,
    pack_b_tile_q8_0: PackBFn,

    matmul_packed_b: MatMulPackedBFn,
};

/// One GEMM family at its three packed sizes: small, medium, large. A backend
/// keeps to one family, so a tile too big for its default still multiplies the
/// same way.
pub const Set = [3]QuantKernels;

pub const SMALL = 0;
pub const MEDIUM = 1;
pub const LARGE = 2;

/// Quantizes A and multiplies with `enc`'s byte dot.
pub fn int8Set(comptime enc: matmul_q_i8.DotEnc) Set {
    return .{ int8Kernels(128, 128, enc), int8Kernels(256, 256, enc), int8Kernels(512, 512, enc) };
}

/// Quantizes A and multiplies with FEAT_I8MM's `smmla` 2x2 matrix product.
pub fn mmSet(comptime enc: matmul_q_i8.MmEnc) Set {
    return .{ mmKernels(128, 128, enc), mmKernels(256, 256, enc), mmKernels(512, 512, enc) };
}

fn int8Kernels(comptime kc: usize, comptime nc: usize, comptime enc: matmul_q_i8.DotEnc) QuantKernels {
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

fn mmKernels(comptime kc: usize, comptime nc: usize, comptime enc: matmul_q_i8.MmEnc) QuantKernels {
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

/// The kernels a backend multiplies quantized weights with: its default blocking,
/// and the family it stays within when a tile is bigger than that.
pub const Choice = struct {
    default: QuantKernels,
    set: Set,

    pub fn of(set: Set, l2_bytes: usize) Choice {
        return .{ .default = pickForL2(set, l2_bytes), .set = set };
    }

    pub fn forTile(self: Choice, k: usize, n: usize) ?QuantKernels {
        if (k <= self.default.tuning.kc and n <= self.default.tuning.nc) return self.default;
        return selectForTile(self.set, k, n);
    }

    pub fn scratchBytes(self: Choice) usize {
        return setScratchBytes(self.set);
    }
};

/// The family `target` multiplies with.
pub fn setForTarget(comptime target: cpu_target.Target) Set {
    return int8Set(comptime cpu_target.emittable(target.int8_dot));
}

/// Scratch every member of `set` fits in.
pub fn setScratchBytes(set: Set) usize {
    var best: usize = 0;
    for (set) |k| best = @max(best, k.scratch_bytes);
    return best;
}

/// The smallest member of `set` whose blocking covers a `k x n` tile.
pub fn selectForTile(set: Set, k: usize, n: usize) ?QuantKernels {
    for (set) |c| {
        if (k <= c.tuning.kc and n <= c.tuning.nc) return c;
    }
    return null;
}

/// The member whose packed B fits ~75% of L2: the largest that does, medium when
/// L2 is unknown, and never small on an L2 of a megabyte or more.
pub fn pickForL2(set: Set, l2_bytes: usize) QuantKernels {
    if (l2_bytes == 0) return set[MEDIUM];
    const budget: usize = matmul_shapes.l2Budget75(l2_bytes);
    var best: usize = SMALL;
    for (set, 0..) |k, i| {
        if (k.packed_b_bytes <= budget) best = i;
    }
    if (l2_bytes >= (1 * 1024 * 1024) and best == SMALL) return set[MEDIUM];
    return set[best];
}
