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
    // Macro tiles: square-ish C tiles.
    const side: usize = @max(@as(usize, 1), @min(policy.base_square_2d, @min(m, n)));
    const tm: usize = @min(side, m);
    const tn: usize = @min(side, n);
    const tk: usize = chooseMatMulTk(policy, k, b_dtype);
    return .{ .tm = tm, .tn = tn, .tk = tk };
}
