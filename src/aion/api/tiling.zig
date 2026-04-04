const std = @import("std");

const types = @import("../backend/types.zig");
const plan_mod = @import("../graph/plan.zig");

pub const MAX_RANK: usize = 8;

pub const TilingError = error{InvalidArgument};

/// Fill a conservative default tile shape for an arbitrary-rank tensor.
///
/// Policy:
/// - rank == 1: chooseTileShape1D
/// - rank >= 2: tile leading/batch dims as 1, last two dims as a square-ish 2D tile
///
/// Notes:
/// - This is intentionally conservative; the compiler may still insert retile steps.
/// - For quantized tensors, callers should prefer op-specific tiling helpers.
pub fn fillDefaultTileShape(policy: plan_mod.TilePolicy, dtype: types.DType, shape: []const usize, out: []usize) TilingError!void {
    if (shape.len == 0 or shape.len > MAX_RANK) return TilingError.InvalidArgument;
    if (out.len != shape.len) return TilingError.InvalidArgument;

    // Small tensors: store as a single full tile.
    //
    // This mirrors the compiler's tiling policy in `graph/program.zig` and is
    // especially important for initializer tensors (weights/biases): over-tiling
    // small constants dramatically increases per-tile acquire/copy overhead.
    //
    // For quantized tensors, callers should prefer op-specific tiling helpers
    // (e.g. chooseQuantMatMulBTiles) to preserve block alignment.
    if (!dtype.info().is_quantized) {
        var total: usize = 1;
        for (shape) |dim| {
            total = std.math.mul(usize, total, dim) catch {
                total = policy.small_tensor_threshold + 1;
                break;
            };
        }
        if (total <= policy.small_tensor_threshold) {
            @memcpy(out, shape);
            return;
        }
    }

    if (shape.len == 1) {
        const t1: [1]usize = plan_mod.chooseTileShape1D(policy, shape[0]);
        out[0] = t1[0];
        return;
    }

    var d: usize = 0;
    while (d + 2 < shape.len) : (d += 1) {
        out[d] = 1;
    }

    const m: usize = shape[shape.len - 2];
    const n: usize = shape[shape.len - 1];
    const t2: [2]usize = plan_mod.chooseTileShape2DSquare(policy, m, n);
    out[shape.len - 2] = t2[0];
    out[shape.len - 1] = t2[1];
}

/// Suggested tiling for a quantized matmul-B weight matrix of shape [k, n].
///
/// This is used to avoid users having to reason about quant block alignment.
/// The returned tile shape is [tk, tn].
pub fn chooseQuantMatMulBTiles(policy: plan_mod.TilePolicy, k: usize, n: usize, b_dtype: types.DType) [2]usize {
    // Pick a "typical" m to drive the heuristic; B is reused across many m.
    const m_hint: usize = @max(@as(usize, 1), policy.base_square_2d);
    const tiles = plan_mod.chooseMatMulTiles(policy, m_hint, n, k, b_dtype);
    return .{ tiles.tk, tiles.tn };
}
