// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
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
        return capTileToBinding(policy, dtype, out);
    }

    var d: usize = 0;
    while (d + 2 < shape.len) : (d += 1) {
        // CPU splits the batch dims to hand threads independent work. A GPU wants
        // the opposite: one tile per tensor, so a kernel sees the whole thing in
        // one binding and indexed ops (gather, KV append) address it flat.
        // `capTileToBinding` below is what keeps that bindable.
        out[d] = if (policy.target_kind == .cpu) 1 else shape[d];
    }

    const m: usize = shape[shape.len - 2];
    const n: usize = shape[shape.len - 1];
    const t2: [2]usize = plan_mod.chooseTileShape2DSquare(policy, m, n);
    out[shape.len - 2] = t2[0];
    out[shape.len - 1] = t2[1];
    capTileToBinding(policy, dtype, out);
}

/// Shrink `out` until one tile fits the device's storage-binding limit.
///
/// The GPU tile bases are deliberately huge — a tile per tensor is what kernels
/// want — but a shader can only bind so much at once, and that limit is far
/// below what can be *allocated*. Without this an oversized tensor gets a single
/// unbindable tile and its op fails at record time; with it the tensor splits
/// and the per-tile dispatch paths take over.
///
/// Shrinking works from the OUTERMOST divisible extent inward, so inner axes stay
/// whole for as long as possible — the layout every indexed kernel needs, since a
/// gathered row, a scattered row, or a key must live entirely in one tile.
/// No-op when the policy declares no limit, which is every CPU policy.
fn capTileToBinding(policy: plan_mod.TilePolicy, dtype: types.DType, out: []usize) void {
    if (policy.max_binding_bytes == 0) return;
    const info = dtype.info();
    // Quantized layouts have block-alignment rules of their own; their callers
    // pick binding-aware tiles directly (see `chooseQuantEmbeddingTableTiles`).
    if (info.is_quantized) return;
    // Same 3/4 margin the quantized table path uses: headroom for allocation
    // rounding and any other per-binding limit.
    const budget: usize = policy.max_binding_bytes / 4 * 3;

    while (true) {
        var bytes: usize = info.block_bytes;
        for (out) |extent| bytes = std.math.mul(usize, bytes, extent) catch return;
        if (bytes <= budget) return;

        var i: usize = 0;
        while (i < out.len and out[i] <= 1) i += 1;
        // Every axis is already a single element: one row alone exceeds the
        // budget, so no tiling can make this bindable. Leave it and let the op
        // report `Unsupported` rather than loop forever.
        if (i == out.len) return;
        out[i] = (out[i] + 1) / 2;
    }
}

/// Choose the tile shape a tensor of `shape`/`dtype`/`quant_axis` should use for a
/// given policy — the same selection `createTensorForShapeWithQuantAxis` applies at
/// weight import, factored out so device migration (`Tensor.to`) re-tiles identically.
/// Fills `out` (must be `shape.len` long).
pub fn chooseTileShapeForTensor(
    policy: plan_mod.TilePolicy,
    dtype: types.DType,
    shape: []const usize,
    quant_axis: u8,
    out: []usize,
) TilingError!void {
    if (shape.len == 0 or shape.len > MAX_RANK) return TilingError.InvalidArgument;
    if (out.len != shape.len) return TilingError.InvalidArgument;

    const is_quant = dtype.info().is_quantized;
    const rank: usize = shape.len;

    if (is_quant and rank >= 2 and @as(usize, quant_axis) == (rank - 2)) {
        var d: usize = 0;
        while (d + 2 < rank) : (d += 1) out[d] = 1;
        const tiles = chooseQuantMatMulBTiles(policy, shape[rank - 2], shape[rank - 1], dtype);
        out[rank - 2] = tiles[0];
        out[rank - 1] = tiles[1];
    } else if (is_quant and rank == 2 and quant_axis == 1) {
        const tiles = chooseQuantEmbeddingTableTiles(policy, dtype, shape[0], shape[1]);
        out[0] = tiles[0];
        out[1] = tiles[1];
    } else {
        try fillDefaultTileShape(policy, dtype, shape, out);
    }
}

/// Suggested tiling for a quantized matmul-B weight matrix of shape [k, n].
///
/// This is used to avoid users having to reason about quant block alignment.
/// The returned tile shape is [tk, tn].
pub fn chooseQuantMatMulBTiles(policy: plan_mod.TilePolicy, k: usize, n: usize, b_dtype: types.DType) [2]usize {
    // Pick a "typical" m to drive the heuristic; B is reused across many m.
    const m_hint = plan_mod.matMulMHint(policy);
    const tiles = plan_mod.chooseMatMulTiles(policy, m_hint, n, k, b_dtype);
    return .{ tiles.tk, tiles.tn };
}

/// Suggested tiling for a rank-2 quantized embedding table of shape [V, D] with
/// `quant_axis == 1` (per-row quantization).
///
/// Returns `[tv, td]` such that `td == D` and `tv` is a conservative row-count per tile.
/// Keeping `td == D` means each row of the table is exactly one contiguous run of
/// `D / block_elems` blocks inside a tile — the layout a row-gather kernel wants.
pub fn chooseQuantEmbeddingTableTiles(policy: plan_mod.TilePolicy, dtype: types.DType, v: usize, d: usize) [2]usize {
    var tv: usize = @max(@as(usize, 1), @min(v, policy.base_1d));
    // Keep whole rows (dim1 = d) but cap the row count so a single tile fits the
    // device binding limit. A multi-GB quantized vocab table is split along v; the
    // gather op then resolves the right tile per row.
    if (policy.max_binding_bytes > 0) {
        const info = dtype.info();
        const per_row: usize = (d / info.block_elems) * info.block_bytes;
        if (per_row > 0) {
            // 3/4 margin leaves headroom for allocation rounding / other limits.
            const budget: usize = policy.max_binding_bytes / 4 * 3;
            const cap_rows: usize = @max(@as(usize, 1), budget / per_row);
            tv = @min(tv, cap_rows);
        }
    }
    return .{ tv, d };
}

/// Same policy with and without a declared binding limit, so a test can compare
/// the capped choice against the one the tile choosers would have made alone.
fn cappedAndUncapped(shape: []const usize, capped: []usize, uncapped: []usize, limit: usize) !void {
    const base: plan_mod.TilePolicy = .{
        .target_kind = .webgpu,
        .base_square_2d = 1 << 20,
        .base_1d = 1 << 24,
        .small_tensor_threshold = 0,
    };
    var with_limit = base;
    with_limit.max_binding_bytes = limit;
    try fillDefaultTileShape(with_limit, .f32, shape, capped);
    try fillDefaultTileShape(base, .f32, shape, uncapped);
}

test "tiling: an oversized tensor splits to fit the storage-binding limit" {
    const limit: usize = 4 << 30;
    const budget: usize = limit / 4 * 3;
    // Square tiling alone would pick a tile far past what a shader can bind.
    const shape = [_]usize{ 1 << 16, 1 << 16 };
    var capped: [2]usize = undefined;
    var uncapped: [2]usize = undefined;
    try cappedAndUncapped(&shape, &capped, &uncapped, limit);

    try std.testing.expect(uncapped[0] * uncapped[1] * 4 > budget); // the case is real
    try std.testing.expect(capped[0] * capped[1] * 4 <= budget);
    // Shrinking runs outermost-first, so the inner axis is untouched here.
    try std.testing.expectEqual(uncapped[1], capped[1]);
    try std.testing.expect(capped[0] < uncapped[0]);
}

test "tiling: a tile that already fits is left as the choosers picked it" {
    const shape = [_]usize{ 1024, 1024 };
    var capped: [2]usize = undefined;
    var uncapped: [2]usize = undefined;
    try cappedAndUncapped(&shape, &capped, &uncapped, 4 << 30);
    try std.testing.expectEqualSlices(usize, &uncapped, &capped);
}

test "tiling: no binding limit declared leaves tiles untouched" {
    const shape = [_]usize{ 1 << 16, 1 << 16 };
    var capped: [2]usize = undefined;
    var uncapped: [2]usize = undefined;
    // limit 0 == "unknown/uncapped", the CPU policy default.
    try cappedAndUncapped(&shape, &capped, &uncapped, 0);
    try std.testing.expectEqualSlices(usize, &uncapped, &capped);
}
