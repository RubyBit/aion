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
    var tile_mem: [api_tiling.MAX_RANK]usize = undefined;
    const tile_shape = tile_mem[0..shape.len];
    if (dtype.info().is_quantized and shape.len == 2) {
        const tiles = api_tiling.chooseQuantMatMulBTiles(policy, shape[0], shape[1], dtype);
        tile_shape[0] = tiles[0];
        tile_shape[1] = tiles[1];
    } else {
        try api_tiling.fillDefaultTileShape(policy, dtype, shape, tile_shape);
    }
    return store.createTiledTensor(dtype, shape, tile_shape, .{ .tile_alignment = policy.tile_alignment });
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
        slot.* = try createTensorForShape(store, policy, value.dtype, shape);

        const meta = try store.getConst(slot.*);
        const tensor = types_mod.Tensor{ .store = store, .id = slot.*, .dtype = meta.dtype, .shape = meta.shape };
        const init = package.initializers[init_idx];
        switch (init.encoding) {
            .plain => try tensor.writePackedScalar(init.data),
            .quantized => try tensor.writePackedQuant(init.data),
        }
    }
    return tids;
}

fn findInitializerValueIndex(package: *const types_mod.Package, initializer_index: u32) ?usize {
    for (package.values, 0..) |value, idx| {
        if (value.initializer_index == initializer_index) return idx;
    }
    return null;
}
