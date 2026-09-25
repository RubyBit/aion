// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const manager_mod = @import("../../storage/manager.zig");
const package_file = @import("../../storage/aion_file.zig");
const types_mod = @import("types.zig");
const params_mod = @import("params.zig");
const api_tiling = @import("../tiling.zig");
const api_errors = @import("../errors.zig");

/// Import every initializer of `package` into `store`, returning the parameters keyed by
/// graph value. The `initializer_index` that names a slot in the file's weight section
/// does not survive this call.
/// Import every initializer into `store`, on `device`: a weight bound for a GPU is
/// created there and written from `package`'s payload directly, never as a host copy.
pub fn importParams(
    allocator: std.mem.Allocator,
    store: *types_mod.StorageManager,
    policy: types_mod.TilePolicy,
    package: *const types_mod.Package,
    device: manager_mod.DeviceRef,
) api_errors.LoadError!params_mod.Params {
    var out = try params_mod.Params.init(allocator, package.values.len);
    errdefer out.deinit(allocator);
    for (package.values, 0..) |value, value_idx| {
        if (value.source != .initializer) continue;
        const init_idx: u32 = value.initializer_index orelse return error.InvalidArgument;
        if (init_idx >= package.initializers.len) return error.InvalidArgument;
        const init = package.initializers[init_idx];
        const tid = try createInitializerTensor(allocator, store, policy, package, value, init, device);
        try store.writePackedAtPlacement(tid, init.data.bytes);
        out.set(@intCast(value_idx), tid);
    }
    return out;
}

/// The store tensor for one initializer value, sized and tiled from its record.
fn createInitializerTensor(
    allocator: std.mem.Allocator,
    store: *types_mod.StorageManager,
    policy: types_mod.TilePolicy,
    package: *const types_mod.Package,
    value: package_file.ValueRecord,
    init: package_file.Initializer,
    device: manager_mod.DeviceRef,
) api_errors.LoadError!types_mod.TensorId {
    const shape = try resolveConstShape(allocator, package, value);
    defer allocator.free(shape);
    const quant_axis: u8 = switch (init.encoding) {
        .plain => 0,
        .quantized => |q| try quantAxisToU8(q.quant_axis, shape.len),
    };
    var tile_mem: [api_tiling.MAX_RANK]usize = undefined;
    const tile_shape = tile_mem[0..shape.len];
    try tileShapeFor(policy, value.dtype, shape, quant_axis, tile_shape);
    const opts: manager_mod.TiledTensor.InitOptions = .{ .tile_alignment = policy.tile_alignment, .quant_axis = quant_axis };
    if (device.kind == .cpu) return store.createTiledTensor(value.dtype, shape, tile_shape, opts);
    return store.createDeviceTensor(value.dtype, shape, tile_shape, opts, device);
}

pub fn createTensorForShape(
    store: *types_mod.StorageManager,
    policy: types_mod.TilePolicy,
    dtype: types_mod.DType,
    shape: []const usize,
) (error{ InvalidArgument, OutOfMemory } || manager_mod.StorageError)!types_mod.TensorId {
    return createTensorForShapeWithQuantAxis(store, policy, dtype, shape, 0);
}

/// Create a tensor sized to `shape` but tiled as a single physical tile (tile_shape == shape).
///
/// Intended for graph inputs whose on-disk semantics require persisting full-shape data
/// across runs (notably KV caches aliased to outputs): those tensors are populated by
/// `copyFrom` on each bind, so any tile split inside them is either wasted work or a
/// kernel invariant violation. Shape-aligned single-tile allocation avoids both.
pub fn createTensorSingleTile(
    store: *types_mod.StorageManager,
    policy: types_mod.TilePolicy,
    dtype: types_mod.DType,
    shape: []const usize,
) (error{ InvalidArgument, OutOfMemory } || manager_mod.StorageError)!types_mod.TensorId {
    if (shape.len == 0 or shape.len > api_tiling.MAX_RANK) return error.InvalidArgument;
    if (dtype.info().is_quantized) {
        // Quantized single-tile allocation is out of scope for this helper: the block-axis
        // alignment rules overlap with per-op tiling constraints. Callers that need quant
        // layouts go through `createTensorForShapeWithQuantAxis`.
        return error.InvalidArgument;
    }
    return store.createTiledTensor(dtype, shape, shape, .{ .tile_alignment = policy.tile_alignment });
}

/// Like `createTensorForShape`, but with an explicit `quant_axis` for quantized dtypes.
///
/// Tiling strategy depends on the block axis:
/// - `quant_axis == rank-2` (matmul-B weights `[..., K, N]`): use the matmul-B tiling helper.
/// - `quant_axis == 1` (embedding table `[V, D]`): use the full-row tiling helper so
///   gather kernels can read one row as a contiguous block run.
/// - other combinations: fall back to the default tile helper (must keep `tile_shape[quant_axis]`
///   a multiple of the dtype's `block_elems`).
pub fn createTensorForShapeWithQuantAxis(
    store: *types_mod.StorageManager,
    policy: types_mod.TilePolicy,
    dtype: types_mod.DType,
    shape: []const usize,
    quant_axis: u8,
) (error{ InvalidArgument, OutOfMemory } || manager_mod.StorageError)!types_mod.TensorId {
    var tile_mem: [api_tiling.MAX_RANK]usize = undefined;
    const tile_shape = tile_mem[0..shape.len];
    try tileShapeFor(policy, dtype, shape, quant_axis, tile_shape);
    return store.createTiledTensor(dtype, shape, tile_shape, .{
        .tile_alignment = policy.tile_alignment,
        .quant_axis = quant_axis,
    });
}

/// The tiling a tensor of `shape` gets under `policy`, blocking along `quant_axis`.
fn tileShapeFor(
    policy: types_mod.TilePolicy,
    dtype: types_mod.DType,
    shape: []const usize,
    quant_axis: u8,
    tile_shape: []usize,
) (error{ InvalidArgument, OutOfMemory } || manager_mod.StorageError)!void {
    const is_quant = dtype.info().is_quantized;
    if (is_quant and @as(usize, quant_axis) >= shape.len) return error.InvalidArgument;
    if (shape.len > api_tiling.MAX_RANK) return error.InvalidArgument;
    const rank: usize = shape.len;

    // Quantized matmul-B tensors use the K axis as the block axis.
    // This is the canonical layout for dense weights and must preserve block alignment.
    if (is_quant and rank >= 2 and @as(usize, quant_axis) == (rank - 2)) {
        var d: usize = 0;
        while (d + 2 < rank) : (d += 1) {
            // Batch dims are always tiled as 1 (matches compiler invariants for MatMul).
            tile_shape[d] = 1;
        }

        const k: usize = shape[rank - 2];
        const n: usize = shape[rank - 1];
        const tiles = api_tiling.chooseQuantMatMulBTiles(policy, k, n, dtype);
        tile_shape[rank - 2] = tiles[0];
        tile_shape[rank - 1] = tiles[1];
    } else if (is_quant and shape.len == 2 and quant_axis == 1) {
        const tiles = api_tiling.chooseQuantRowTiles(policy, dtype, shape[0], shape[1]);
        tile_shape[0] = tiles[0];
        tile_shape[1] = tiles[1];
    } else {
        try api_tiling.fillDefaultTileShape(policy, dtype, shape, tile_shape);
    }
}

fn resolveConstShape(
    allocator: std.mem.Allocator,
    pkg: *const types_mod.Package,
    value: package_file.ValueRecord,
) api_errors.LoadError![]usize {
    const zero_symbols = try allocator.alloc(?u64, pkg.dim_symbols.len);
    defer allocator.free(zero_symbols);
    @memset(zero_symbols, null);
    return package_file.resolveShapeTerms(allocator, pkg, value.shape_terms, zero_symbols);
}

fn quantAxisToU8(raw: i32, rank: usize) error{InvalidArgument}!u8 {
    if (raw < 0) return error.InvalidArgument;
    const as_usize: usize = @intCast(raw);
    if (as_usize >= rank) return error.InvalidArgument;
    if (as_usize > std.math.maxInt(u8)) return error.InvalidArgument;
    return @intCast(as_usize);
}

