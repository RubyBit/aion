const std = @import("std");

const manager_mod = @import("../../storage/manager.zig");
const package_file = @import("../../storage/aion_file.zig");
const types_mod = @import("types.zig");
const api_tiling = @import("../tiling.zig");
const api_errors = @import("../errors.zig");

pub fn importInitializersForLoadedModel(
    allocator: std.mem.Allocator,
    store: *types_mod.StorageManager,
    policy: types_mod.TilePolicy,
    package: *const types_mod.Package,
) api_errors.LoadError![]types_mod.TensorId {
    return initInitializerTensors(allocator, store, policy, package);
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
    const is_quant = dtype.info().is_quantized;
    if (is_quant and @as(usize, quant_axis) >= shape.len) return error.InvalidArgument;

    var tile_mem: [api_tiling.MAX_RANK]usize = undefined;
    const tile_shape = tile_mem[0..shape.len];

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
        const tiles = api_tiling.chooseQuantEmbeddingTableTiles(policy, shape[0], shape[1]);
        tile_shape[0] = tiles[0];
        tile_shape[1] = tiles[1];
    } else {
        try api_tiling.fillDefaultTileShape(policy, dtype, shape, tile_shape);
    }

    return store.createTiledTensor(dtype, shape, tile_shape, .{
        .tile_alignment = policy.tile_alignment,
        .quant_axis = quant_axis,
    });
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

fn initInitializerTensors(
    allocator: std.mem.Allocator,
    store: *types_mod.StorageManager,
    policy: types_mod.TilePolicy,
    package: *const types_mod.Package,
) api_errors.LoadError![]types_mod.TensorId {
    const tids = try allocator.alloc(types_mod.TensorId, package.initializers.len);
    errdefer allocator.free(tids);
    for (tids, 0..) |*slot, init_idx| {
        const value_idx = findInitializerValueIndex(package, @intCast(init_idx)) orelse return error.InvalidArgument;
        const value = package.values[value_idx];
        const shape = try resolveConstShape(allocator, package, value);
        defer allocator.free(shape);

        const init = package.initializers[init_idx];
        const quant_axis: u8 = switch (init.encoding) {
            .plain => 0,
            .quantized => |q| try quantAxisToU8(q.quant_axis, shape.len),
        };
        slot.* = try createTensorForShapeWithQuantAxis(store, policy, value.dtype, shape, quant_axis);

        const meta = try store.getConst(slot.*);
        const tensor = types_mod.Tensor{ .store = store, .id = slot.*, .dtype = meta.dtype, .shape = meta.shape };
        switch (init.encoding) {
            .plain => try tensor.writePackedScalar(init.data),
            .quantized => try tensor.writePackedQuant(init.data),
        }
    }
    return tids;
}

fn quantAxisToU8(raw: i32, rank: usize) error{InvalidArgument}!u8 {
    if (raw < 0) return error.InvalidArgument;
    const as_usize: usize = @intCast(raw);
    if (as_usize >= rank) return error.InvalidArgument;
    if (as_usize > std.math.maxInt(u8)) return error.InvalidArgument;
    return @intCast(as_usize);
}

fn findInitializerValueIndex(package: *const types_mod.Package, initializer_index: u32) ?usize {
    for (package.values, 0..) |value, idx| {
        if (value.initializer_index == initializer_index) return idx;
    }
    return null;
}
