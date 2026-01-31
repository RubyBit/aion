const std = @import("std");

const types = @import("../backend/types.zig");

pub const DType = types.DType;

pub const TilePolicy = struct {
    /// Default square tile side for rank-2 tensors.
    ///
    /// Note: transpose materialization is simplest when tiles are square.
    base_square_2d: usize = 64,

    /// Default tile length for rank-1 tensors.
    base_1d: usize = 256,

    /// Required alignment for quantized K blocks (ggml q4/q8 use 32).
    quant_k_block: usize = 32,

    /// Tile alignment in bytes for `TiledTensor` backing.
    tile_alignment: usize = 64,
};

pub fn chooseTileShape1D(policy: TilePolicy, n: usize) [1]usize {
    const t: usize = @min(policy.base_1d, if (n == 0) 1 else n);
    return .{@max(@as(usize, 1), t)};
}

pub fn chooseTileShape2DSquare(policy: TilePolicy, m: usize, n: usize) [2]usize {
    const side: usize = @max(@as(usize, 1), @min(policy.base_square_2d, @min(m, n)));
    return .{ side, side };
}

pub fn chooseSoftmaxTiles(policy: TilePolicy, m: usize, n: usize) struct { tm: usize, tn: usize } {
    // Softmax is row-wise and needs per-row scratch. Prefer modest row tiles.
    // Keep tm small to allow stack scratch in exec and increase parallelism.
    const tm_cap: usize = 256;
    const tm: usize = @max(@as(usize, 1), @min(tm_cap, @min(m, policy.base_1d)));

    // Favor wide tiles along the reduction axis to reduce tile overhead.
    // Cap tn to base_square_2d to avoid giant tiles.
    const tn: usize = @max(@as(usize, 1), @min(n, policy.base_square_2d));
    return .{ .tm = tm, .tn = tn };
}

pub fn chooseNormTiles(policy: TilePolicy, m: usize, n: usize) struct { tm: usize, tn: usize } {
    // LayerNorm/RMSNorm are row-wise and need per-row scratch. Keep tm small.
    const tm_cap: usize = 256;
    const tm: usize = @max(@as(usize, 1), @min(tm_cap, @min(m, policy.base_1d)));
    // Favor decent width along last dim; cap to base_square_2d like softmax.
    const tn: usize = @max(@as(usize, 1), @min(n, policy.base_square_2d));
    return .{ .tm = tm, .tn = tn };
}

pub fn chooseAttentionTiles(policy: TilePolicy, m: usize, n: usize, dk: usize, dv: usize) struct { tm: usize, tn: usize, tk: usize, tv: usize } {
    // Attention is row-wise over queries, and needs per-query scratch.
    // IMPORTANT: we parallelize attention primarily across query tiles.
    // Keep tm modest so we have enough tile-level parallelism on many-core CPUs.
    const tm_cap: usize = 32;
    const tm: usize = @max(@as(usize, 1), @min(tm_cap, @min(m, policy.base_1d)));

    // Block keys modestly so per-row score scratch stays small.
    const tn_cap: usize = 128;
    const tn: usize = @max(@as(usize, 1), @min(n, @min(tn_cap, policy.base_square_2d)));

    // Block value/output columns to bound the accumulator scratch.
    const tv_cap: usize = 64;
    var tv_target: usize = @min(dv, @min(tv_cap, policy.base_square_2d));
    if (tv_target >= 16) {
        tv_target = roundDownToMultiple(tv_target, 16);
        if (tv_target == 0) tv_target = @min(@as(usize, 16), dv);
    }
    const tv: usize = @max(@as(usize, 1), @min(tv_target, dv));

    // Block head dim reasonably; keep it SIMD-friendly.
    var tk_target: usize = @min(@as(usize, 128), dk);
    if (tk_target >= 16) {
        tk_target = roundDownToMultiple(tk_target, 16);
        if (tk_target == 0) tk_target = @min(@as(usize, 16), dk);
    }
    const tk: usize = @max(@as(usize, 1), @min(tk_target, dk));

    return .{ .tm = tm, .tn = tn, .tk = tk, .tv = tv };
}

pub fn roundDownToMultiple(x: usize, m: usize) usize {
    if (m == 0) return x;
    return x - (x % m);
}

pub fn roundUpToMultiple(x: usize, m: usize) usize {
    if (m == 0) return x;
    const r: usize = x % m;
    if (r == 0) return x;
    return x + (m - r);
}

pub fn chooseMatMulTk(policy: TilePolicy, k: usize, b_dtype: DType) usize {
    // Heuristic: try to keep K tiles reasonably sized, but ensure quant alignment.
    const base: usize = @min(@as(usize, 256), @max(policy.quant_k_block, k));
    if (!b_dtype.info().is_quantized) return @min(base, k);

    const be: usize = b_dtype.info().block_elems;
    const want: usize = roundDownToMultiple(@min(base, k), be);
    return @max(be, want);
}

pub fn chooseMatMulTiles(policy: TilePolicy, m: usize, n: usize, k: usize, b_dtype: DType) struct { tm: usize, tn: usize, tk: usize } {
    // Macro tiles: choose C tiles.
    //
    // Important trade-off:
    // - Larger tiles improve per-tile kernel efficiency and amortize packing.
    // - But too-large tiles reduce the number of output tiles, which limits parallelism
    //   for tiled program execution (we intentionally avoid intra-tile parallel writes).
    //
    // Heuristic (v1): for matrices >= 512, cap the tile side to 128 to ensure enough
    // tile-level parallelism on many-core CPUs.
    // Special-case very skinny matrices (e.g. matvec with m==1): a square tile would
    // degenerate to tn==1, which is disastrous for kernel efficiency and tile overhead.
    //
    // Heuristic: when one dimension is tiny, tile fully along that dim and use a wide
    // tile along the other dim (rounded to SIMD-friendly multiples of 16).
    if (m <= 4 and n >= 64) {
        const tm: usize = @max(@as(usize, 1), m);

        // Keep N tiles reasonably large to amortize per-tile overhead, but not so large
        // that we destroy parallelism. 256 matches CPU kernel NC.
        var tn_target: usize = @min(@as(usize, 256), n);
        if (tn_target >= 16) {
            tn_target = roundDownToMultiple(tn_target, 16);
            if (tn_target == 0) tn_target = 16;
        }
        const tn: usize = @max(@as(usize, 1), @min(tn_target, n));

        const tk: usize = chooseMatMulTk(policy, k, b_dtype);
        return .{ .tm = tm, .tn = tn, .tk = tk };
    }
    if (n <= 4 and m >= 64) {
        const tn: usize = @max(@as(usize, 1), n);
        var tm_target: usize = @min(@as(usize, 256), m);
        if (tm_target >= 16) {
            tm_target = roundDownToMultiple(tm_target, 16);
            if (tm_target == 0) tm_target = 16;
        }
        const tm: usize = @max(@as(usize, 1), @min(tm_target, m));
        const tk: usize = chooseMatMulTk(policy, k, b_dtype);
        return .{ .tm = tm, .tn = tn, .tk = tk };
    }

    // Default: square-ish tiles.
    const min_mn: usize = @min(m, n);
    var side: usize = @max(@as(usize, 1), @min(policy.base_square_2d, min_mn));
    if (side > 128 and min_mn >= 512) side = 128;
    const tm: usize = @min(side, m);
    const tn: usize = @min(side, n);
    const tk: usize = chooseMatMulTk(policy, k, b_dtype);
    return .{ .tm = tm, .tn = tn, .tk = tk };
}
