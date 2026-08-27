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
pub fn importParams(
    allocator: std.mem.Allocator,
    store: *types_mod.StorageManager,
    policy: types_mod.TilePolicy,
    package: *const types_mod.Package,
) api_errors.LoadError!params_mod.Params {
    var out = try params_mod.Params.init(allocator, package.values.len);
    errdefer out.deinit(allocator);
    for (package.values, 0..) |value, value_idx| {
        if (value.source != .initializer) continue;
        const init_idx: u32 = value.initializer_index orelse return error.InvalidArgument;
        if (init_idx >= package.initializers.len) return error.InvalidArgument;
        const init = package.initializers[init_idx];
        const tid = try createInitializerTensor(allocator, store, policy, package, value, init);
        const meta = try store.getConst(tid);
        const tensor = types_mod.Tensor{ .store = store, .id = tid, .dtype = meta.dtype, .shape = meta.shape };
        switch (init.encoding) {
            .plain => try tensor.writePackedScalar(init.data),
            .quantized => try tensor.writePackedQuant(init.data),
        }
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
) api_errors.LoadError!types_mod.TensorId {
    const shape = try resolveConstShape(allocator, package, value);
    defer allocator.free(shape);
    const quant_axis: u8 = switch (init.encoding) {
        .plain => 0,
        .quantized => |q| try quantAxisToU8(q.quant_axis, shape.len),
    };
    return createTensorForShapeWithQuantAxis(store, policy, value.dtype, shape, quant_axis);
}

/// Like `importInitializersForLoadedModel`, but streams each initializer's bytes
/// straight from `file` into its store tensor instead of copying from the in-memory
/// file buffer — then frees that buffer up front.
///
/// Rationale: the default path keeps the whole `.aion` file resident (initializer
/// `data` slices borrow into it) *while* it copies every weight into the store,
/// briefly doubling peak RSS (~2x the file). Here we capture each initializer's
/// byte offset within the file, release the file buffer via `releaseSourceBytes`,
/// then read each weight back from disk (OS-cached) into a single reused scratch
/// buffer as we fill the store. The file blob and the populated store never coexist,
/// so peak RSS is ~1x the weight size + one initializer's worth of scratch.
///
/// `source_bytes` must be the exact buffer the package's initializer slices borrow
/// into (i.e. the full file image starting at file offset 0). `package.source_bytes`
/// is consumed (freed) by this call.
pub fn importParamsStreaming(
    allocator: std.mem.Allocator,
    store: *types_mod.StorageManager,
    policy: types_mod.TilePolicy,
    package: *types_mod.Package,
    file: std.Io.File,
    source_bytes: []const u8,
) api_errors.LoadError!params_mod.Params {
    const n: usize = package.initializers.len;

    // 1. Capture each initializer's (file offset, len) while the borrowed data
    //    slices are still valid, and find the largest payload (scratch size).
    const Span = struct { off: u64, len: usize };
    const spans = try allocator.alloc(Span, n);
    defer allocator.free(spans);
    const base: usize = @intFromPtr(source_bytes.ptr);
    var max_len: usize = 0;
    for (package.initializers, 0..) |init, i| {
        const ptr: usize = @intFromPtr(init.data.ptr);
        if (ptr < base or (ptr + init.data.len) > base + source_bytes.len) {
            // Initializer doesn't borrow into the file buffer (shouldn't happen for a
            // freshly parsed package); fall back to the in-memory copy path.
            return importParams(allocator, store, policy, package);
        }
        spans[i] = .{ .off = @intCast(ptr - base), .len = init.data.len };
        max_len = @max(max_len, init.data.len);
    }

    // 2. Release the file buffer now — before the store fills — so the two never
    //    coexist. (Empties the borrowed data/params slices; encoding scalars stay.)
    package.releaseSourceBytes();

    // 3. One reusable scratch buffer, sized to the largest single initializer.
    const scratch = try allocator.alloc(u8, @max(max_len, 1));
    defer allocator.free(scratch);

    var io_backend: std.Io.Threaded = .init_single_threaded;
    const io = io_backend.io();

    var out = try params_mod.Params.init(allocator, package.values.len);
    errdefer out.deinit(allocator);
    for (package.values, 0..) |value, value_idx| {
        if (value.source != .initializer) continue;
        const init_idx: u32 = value.initializer_index orelse return error.InvalidArgument;
        if (init_idx >= n) return error.InvalidArgument;
        const init = package.initializers[init_idx];
        const tid = try createInitializerTensor(allocator, store, policy, package, value, init);

        // Read this initializer's packed bytes back from disk (OS page cache) into scratch.
        const span = spans[init_idx];
        const buf: []u8 = scratch[0..span.len];
        const got = file.readPositionalAll(io, buf, span.off) catch return error.IoFailure;
        if (got != span.len) return error.IoFailure;

        const meta = try store.getConst(tid);
        const tensor = types_mod.Tensor{ .store = store, .id = tid, .dtype = meta.dtype, .shape = meta.shape };
        switch (init.encoding) {
            .plain => try tensor.writePackedScalar(buf),
            .quantized => try tensor.writePackedQuant(buf),
        }
        out.set(@intCast(value_idx), tid);
    }
    return out;
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
        const tiles = api_tiling.chooseQuantEmbeddingTableTiles(policy, dtype, shape[0], shape[1]);
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

fn quantAxisToU8(raw: i32, rank: usize) error{InvalidArgument}!u8 {
    if (raw < 0) return error.InvalidArgument;
    const as_usize: usize = @intCast(raw);
    if (as_usize >= rank) return error.InvalidArgument;
    if (as_usize > std.math.maxInt(u8)) return error.InvalidArgument;
    return @intCast(as_usize);
}

