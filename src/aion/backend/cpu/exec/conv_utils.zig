const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const matmul_registry = @import("../registry/matmul_registry.zig");
const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const exec_utils = @import("utils.zig");

pub const BackendError = types.BackendError;
pub const MatMulParams = types.MatMulParams;
pub const ExecuteProgramError = backend_mod.ExecuteProgramError;

pub const MAX_RANK: usize = 8;

pub const elemCountFromShape = exec_utils.elemCountFromShape;
pub const computePackedStrides = exec_utils.computePackedStrides;
pub const decodeLinearToCoords = exec_utils.decodeLinearToCoords;

pub const ConvExecCtx = struct {
    allocator: std.mem.Allocator,
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    matmul_f32: matmul_registry.F32Kernels,
    // Per-thread scratch for matmul packing. May be empty in single-thread mode.
    matmul_scratch: [][]align(32) u8,
};

pub const PackedWeightKey = struct {
    w_id: tensor_store.TensorId,
    oc_start: usize,
    k_dim: usize,
    c_out: usize,
    groups: usize,
    kc: usize,
    nc: usize,
};

pub const PackedWeightEntry = struct {
    key: PackedWeightKey,
    // Concatenated packed-B panels, each of length kc*nc f32.
    blocks: []align(32) f32,
    block_count: usize,
    block_elems: usize,
};

var g_packed_w_mutex: std.Thread.Mutex = .{};
var g_packed_w_init: bool = false;
var g_packed_w_cache: std.AutoHashMap(PackedWeightKey, PackedWeightEntry) = undefined;

fn ensurePackedWeightCacheInit() void {
    if (g_packed_w_init) return;
    g_packed_w_cache = std.AutoHashMap(PackedWeightKey, PackedWeightEntry).init(std.heap.page_allocator);
    g_packed_w_init = true;
}

pub fn getOrCreatePackedWeights(
    matmul_f32: matmul_registry.F32Kernels,
    key: PackedWeightKey,
    w_matrix: ?[]align(1) const f32,
) ExecuteProgramError!PackedWeightEntry {
    g_packed_w_mutex.lock();
    defer g_packed_w_mutex.unlock();

    ensurePackedWeightCacheInit();
    if (g_packed_w_cache.get(key)) |entry| return entry;

    const wv: []align(1) const f32 = w_matrix orelse return BackendError.InvalidArgument;
    if (wv.len < key.k_dim * key.c_out) return BackendError.InvalidArgument;

    const k_blocks: usize = (key.k_dim + key.kc - 1) / key.kc;
    const block_elems: usize = key.kc * key.nc;
    const total_elems: usize = k_blocks * block_elems;

    const alloc: std.mem.Allocator = std.heap.page_allocator;
    const blocks: []align(32) f32 = try alloc.alignedAlloc(f32, std.mem.Alignment.fromByteUnits(32), total_elems);
    errdefer alloc.free(blocks);

    const scratch_tmp: []align(32) u8 = try alloc.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), matmul_f32.scratch_bytes);
    defer alloc.free(scratch_tmp);

    var bi: usize = 0;
    while (bi < k_blocks) : (bi += 1) {
        const kk0: usize = bi * key.kc;
        const k_sub: usize = @min(key.kc, key.k_dim - kk0);
        const b_off: usize = kk0 * key.c_out;
        const b_need: usize = k_sub * key.c_out;
        const b_bytes: []const u8 = std.mem.sliceAsBytes(wv[b_off .. b_off + b_need]);

        try matmul_f32.pack_b_tile(scratch_tmp, k_sub, key.c_out, b_bytes);

        const dst_start: usize = bi * block_elems;
        const dst: []align(32) f32 = @alignCast(blocks[dst_start .. dst_start + block_elems]);
        const src: []align(32) const f32 = @alignCast(std.mem.bytesAsSlice(f32, scratch_tmp[0 .. block_elems * @sizeOf(f32)]));
        @memcpy(dst, src);
    }

    const entry: PackedWeightEntry = .{
        .key = key,
        .blocks = blocks,
        .block_count = k_blocks,
        .block_elems = block_elems,
    };
    try g_packed_w_cache.put(key, entry);
    return entry;
}

pub fn scratchForTid(ctx: *const ConvExecCtx, tid: usize) ExecuteProgramError![]align(32) u8 {
    if (ctx.matmul_scratch.len != 0 and tid < ctx.matmul_scratch.len) return ctx.matmul_scratch[tid];
    if (ctx.matmul_scratch.len != 0) return ctx.matmul_scratch[0];
    return ctx.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), ctx.matmul_f32.scratch_bytes) catch return BackendError.ExecutionFailed;
}

pub fn fillWeightBlock(
    dst: []f32,
    w_full: []const f32,
    k_dim: usize,
    c_out: usize,
    oc_start: usize,
    oc_count: usize,
) ExecuteProgramError!void {
    if (oc_count == 0) return BackendError.InvalidArgument;
    if (oc_start + oc_count > c_out) return BackendError.InvalidArgument;
    if (dst.len < k_dim * oc_count) return BackendError.InvalidArgument;
    if (w_full.len < k_dim * c_out) return BackendError.InvalidArgument;

    var k: usize = 0;
    while (k < k_dim) : (k += 1) {
        const src_off: usize = k * c_out + oc_start;
        const dst_off: usize = k * oc_count;
        @memcpy(dst[dst_off .. dst_off + oc_count], w_full[src_off .. src_off + oc_count]);
    }
}

pub fn readTensorPackedF32(store: tensor_store.TensorStore, meta: tensor_store.TensorMeta, id: tensor_store.TensorId, out: []f32) ExecuteProgramError!void {
    if (meta.dtype != .f32) return BackendError.InvalidArgument;
    const rank: usize = @as(usize, meta.rank);
    if (rank == 0 or rank > MAX_RANK) return BackendError.InvalidArgument;

    const total: usize = try elemCountFromShape(meta.shape);
    if (out.len != total) return BackendError.InvalidArgument;

    var packed_strides: [MAX_RANK]usize = undefined;
    var local_strides: [MAX_RANK]usize = undefined;
    var tile_coords: [MAX_RANK]usize = undefined;
    var local_coords: [MAX_RANK]usize = undefined;
    var global_coords: [MAX_RANK]usize = undefined;

    try computePackedStrides(meta.shape, packed_strides[0..rank]);

    var tile_total: usize = 1;
    var d: usize = 0;
    while (d < rank) : (d += 1) {
        tile_total *= meta.tile_counts[d];
    }

    // Hot fast-path: a single full tile that matches the logical tensor shape.
    // In this case, data is already packed-contiguous and can be copied in one shot.
    if (tile_total == 1) {
        const t0 = try store.acquireTileConstLinear(id, 0);
        defer store.releaseConst(t0.token);
        const tv0 = t0.bufferView();

        var full_match: bool = (tv0.layout.shape.len == rank);
        d = 0;
        while (full_match and d < rank) : (d += 1) {
            full_match = (tv0.layout.shape[d] == meta.shape[d]);
        }

        if (full_match) {
            const need_bytes: usize = total * @sizeOf(f32);
            if (tv0.bytes.len < need_bytes) return BackendError.InvalidArgument;
            const src_vals: []align(1) const f32 = std.mem.bytesAsSlice(f32, tv0.bytes[0..need_bytes]);
            @memcpy(out, src_vals[0..out.len]);
            return;
        }
    }

    // Fast-path: batch-tiled rank-3 (NLC) or rank-4 (NHWC) where each tile is a full slice.
    if (rank == 3 and meta.tile_counts[1] == 1 and meta.tile_counts[2] == 1 and meta.tile_shape[0] == 1 and meta.tile_shape[1] == meta.shape[1] and meta.tile_shape[2] == meta.shape[2]) {
        const elems_per_tile: usize = meta.shape[1] * meta.shape[2];
        var ti: usize = 0;
        while (ti < meta.tile_counts[0]) : (ti += 1) {
            const t = try store.acquireTileConstLinear(id, ti);
            defer store.releaseConst(t.token);
            const tv = t.bufferView();
            std.debug.assert(tv.layout.shape.len == 3);
            std.debug.assert(tv.layout.shape[0] == 1 and tv.layout.shape[1] == meta.shape[1] and tv.layout.shape[2] == meta.shape[2]);
            const need_bytes: usize = elems_per_tile * @sizeOf(f32);
            std.debug.assert(tv.bytes.len >= need_bytes);
            const vals: []align(1) const f32 = std.mem.bytesAsSlice(f32, tv.bytes[0..need_bytes]);
            const dst_off: usize = ti * elems_per_tile;
            @memcpy(out[dst_off .. dst_off + elems_per_tile], vals[0..elems_per_tile]);
        }
        return;
    }

    // Fast-path: rank-3 tiles with padded channel dimension (tile_shape[2] >= shape[2]).
    if (rank == 3 and meta.tile_shape[0] == 1 and meta.tile_counts[2] == 1 and meta.tile_shape[2] >= meta.shape[2]) {
        const c: usize = meta.shape[2];
        var ti: usize = 0;
        while (ti < tile_total) : (ti += 1) {
            const t = try store.acquireTileConstLinear(id, ti);
            defer store.releaseConst(t.token);
            const tv = t.bufferView();
            const vals: []align(1) const f32 = std.mem.bytesAsSlice(f32, tv.bytes);

            try tensor_store.decodeTileCoords(meta, ti, tile_coords[0..rank]);
            const base0: usize = tile_coords[0] * meta.tile_shape[0];
            const base1: usize = tile_coords[1] * meta.tile_shape[1];
            const t0: usize = tv.layout.shape[0];
            const t1: usize = tv.layout.shape[1];
            const t2: usize = tv.layout.shape[2];

            if (base0 + t0 <= meta.shape[0] and base1 + t1 <= meta.shape[1]) {
                var d0: usize = 0;
                while (d0 < t0) : (d0 += 1) {
                    const g0: usize = base0 + d0;
                    var d1: usize = 0;
                    while (d1 < t1) : (d1 += 1) {
                        const g1: usize = base1 + d1;
                        const li_base: usize = (d0 * t1 + d1) * t2;
                        const idx_base: usize = ((g0 * meta.shape[1] + g1) * meta.shape[2]);
                        @memcpy(out[idx_base .. idx_base + c], vals[li_base .. li_base + c]);
                    }
                }
                continue;
            }

            var d0: usize = 0;
            while (d0 < t0) : (d0 += 1) {
                const g0: usize = base0 + d0;
                if (g0 >= meta.shape[0]) continue;
                var d1: usize = 0;
                while (d1 < t1) : (d1 += 1) {
                    const g1: usize = base1 + d1;
                    if (g1 >= meta.shape[1]) continue;
                    const li_base: usize = (d0 * t1 + d1) * t2;
                    const idx_base: usize = (g0 * meta.shape[1] + g1) * meta.shape[2];
                    @memcpy(out[idx_base .. idx_base + c], vals[li_base .. li_base + c]);
                }
            }
        }
        return;
    }

    if (rank == 4 and meta.tile_counts[1] == 1 and meta.tile_counts[2] == 1 and meta.tile_counts[3] == 1 and meta.tile_shape[0] == 1 and meta.tile_shape[1] == meta.shape[1] and meta.tile_shape[2] == meta.shape[2] and meta.tile_shape[3] == meta.shape[3]) {
        const elems_per_tile: usize = meta.shape[1] * meta.shape[2] * meta.shape[3];
        var ti: usize = 0;
        while (ti < meta.tile_counts[0]) : (ti += 1) {
            const t = try store.acquireTileConstLinear(id, ti);
            defer store.releaseConst(t.token);
            const tv = t.bufferView();
            std.debug.assert(tv.layout.shape.len == 4);
            std.debug.assert(tv.layout.shape[0] == 1 and tv.layout.shape[1] == meta.shape[1] and tv.layout.shape[2] == meta.shape[2] and tv.layout.shape[3] == meta.shape[3]);
            const need_bytes: usize = elems_per_tile * @sizeOf(f32);
            std.debug.assert(tv.bytes.len >= need_bytes);
            const vals: []align(1) const f32 = std.mem.bytesAsSlice(f32, tv.bytes[0..need_bytes]);
            const dst_off: usize = ti * elems_per_tile;
            @memcpy(out[dst_off .. dst_off + elems_per_tile], vals[0..elems_per_tile]);
        }
        return;
    }

    var ti: usize = 0;
    while (ti < tile_total) : (ti += 1) {
        const t = try store.acquireTileConstLinear(id, ti);
        defer store.releaseConst(t.token);
        const tv = t.bufferView();
        const vals: []align(1) const f32 = std.mem.bytesAsSlice(f32, tv.bytes);

        try tensor_store.decodeTileCoords(meta, ti, tile_coords[0..rank]);
        if (rank == 4) {
            const base0: usize = tile_coords[0] * meta.tile_shape[0];
            const base1: usize = tile_coords[1] * meta.tile_shape[1];
            const base2: usize = tile_coords[2] * meta.tile_shape[2];
            const base3: usize = tile_coords[3] * meta.tile_shape[3];

            const t0: usize = tv.layout.shape[0];
            const t1: usize = tv.layout.shape[1];
            const t2: usize = tv.layout.shape[2];
            const t3: usize = tv.layout.shape[3];

            if (base0 + t0 <= meta.shape[0] and base1 + t1 <= meta.shape[1] and base2 + t2 <= meta.shape[2] and base3 + t3 <= meta.shape[3]) {
                var d0: usize = 0;
                while (d0 < t0) : (d0 += 1) {
                    const g0: usize = base0 + d0;
                    var d1: usize = 0;
                    while (d1 < t1) : (d1 += 1) {
                        const g1: usize = base1 + d1;
                        var d2: usize = 0;
                        while (d2 < t2) : (d2 += 1) {
                            const g2: usize = base2 + d2;
                            const li_base: usize = ((d0 * t1 + d1) * t2 + d2) * t3;
                            const idx_base: usize = (((g0 * meta.shape[1] + g1) * meta.shape[2] + g2) * meta.shape[3]) + base3;
                            @memcpy(out[idx_base .. idx_base + t3], vals[li_base .. li_base + t3]);
                        }
                    }
                }
                continue;
            }

            var d0: usize = 0;
            while (d0 < t0) : (d0 += 1) {
                const g0: usize = base0 + d0;
                if (g0 >= meta.shape[0]) continue;
                var d1: usize = 0;
                while (d1 < t1) : (d1 += 1) {
                    const g1: usize = base1 + d1;
                    if (g1 >= meta.shape[1]) continue;
                    var d2: usize = 0;
                    while (d2 < t2) : (d2 += 1) {
                        const g2: usize = base2 + d2;
                        if (g2 >= meta.shape[2]) continue;
                        const li_base: usize = ((d0 * t1 + d1) * t2 + d2) * t3;
                        const idx_base: usize = ((g0 * meta.shape[1] + g1) * meta.shape[2] + g2) * meta.shape[3];
                        var d3: usize = 0;
                        while (d3 < t3) : (d3 += 1) {
                            const g3: usize = base3 + d3;
                            if (g3 >= meta.shape[3]) continue;
                            out[idx_base + g3] = vals[li_base + d3];
                        }
                    }
                }
            }
        } else if (rank == 3) {
            const base0: usize = tile_coords[0] * meta.tile_shape[0];
            const base1: usize = tile_coords[1] * meta.tile_shape[1];
            const base2: usize = tile_coords[2] * meta.tile_shape[2];

            const t0: usize = tv.layout.shape[0];
            const t1: usize = tv.layout.shape[1];
            const t2: usize = tv.layout.shape[2];

            if (base0 + t0 <= meta.shape[0] and base1 + t1 <= meta.shape[1] and base2 + t2 <= meta.shape[2]) {
                var d0: usize = 0;
                while (d0 < t0) : (d0 += 1) {
                    const g0: usize = base0 + d0;
                    var d1: usize = 0;
                    while (d1 < t1) : (d1 += 1) {
                        const g1: usize = base1 + d1;
                        const li_base: usize = (d0 * t1 + d1) * t2;
                        const idx_base: usize = ((g0 * meta.shape[1] + g1) * meta.shape[2]) + base2;
                        @memcpy(out[idx_base .. idx_base + t2], vals[li_base .. li_base + t2]);
                    }
                }
                continue;
            }

            var d0: usize = 0;
            while (d0 < t0) : (d0 += 1) {
                const g0: usize = base0 + d0;
                if (g0 >= meta.shape[0]) continue;
                var d1: usize = 0;
                while (d1 < t1) : (d1 += 1) {
                    const g1: usize = base1 + d1;
                    if (g1 >= meta.shape[1]) continue;
                    const li_base: usize = (d0 * t1 + d1) * t2;
                    const idx_base: usize = (g0 * meta.shape[1] + g1) * meta.shape[2];
                    var d2: usize = 0;
                    while (d2 < t2) : (d2 += 1) {
                        const g2: usize = base2 + d2;
                        if (g2 >= meta.shape[2]) continue;
                        out[idx_base + g2] = vals[li_base + d2];
                    }
                }
            }
        } else {
            try computePackedStrides(tv.layout.shape, local_strides[0..rank]);
            const local_total: usize = try elemCountFromShape(tv.layout.shape);

            var li: usize = 0;
            while (li < local_total) : (li += 1) {
                try decodeLinearToCoords(li, local_strides[0..rank], tv.layout.shape, local_coords[0..rank]);
                var idx: usize = 0;
                var in_bounds: bool = true;
                d = 0;
                while (d < rank) : (d += 1) {
                    global_coords[d] = tile_coords[d] * meta.tile_shape[d] + local_coords[d];
                    if (global_coords[d] >= meta.shape[d]) {
                        in_bounds = false;
                        break;
                    }
                    idx = std.math.add(usize, idx, global_coords[d] * packed_strides[d]) catch return BackendError.InvalidArgument;
                }
                if (!in_bounds) continue;
                out[idx] = vals[li];
            }
        }
    }
}

pub fn writeTensorPackedF32(store: tensor_store.TensorStore, meta: tensor_store.TensorMeta, id: tensor_store.TensorId, src: []const f32) ExecuteProgramError!void {
    if (meta.dtype != .f32) return BackendError.InvalidArgument;
    const rank: usize = @as(usize, meta.rank);
    if (rank == 0 or rank > MAX_RANK) return BackendError.InvalidArgument;

    const total: usize = try elemCountFromShape(meta.shape);
    if (src.len != total) return BackendError.InvalidArgument;

    var packed_strides: [MAX_RANK]usize = undefined;
    var local_strides: [MAX_RANK]usize = undefined;
    var tile_coords: [MAX_RANK]usize = undefined;
    var local_coords: [MAX_RANK]usize = undefined;
    var global_coords: [MAX_RANK]usize = undefined;

    try computePackedStrides(meta.shape, packed_strides[0..rank]);

    var tile_total: usize = 1;
    var d: usize = 0;
    while (d < rank) : (d += 1) {
        tile_total *= meta.tile_counts[d];
    }

    // Hot fast-path: a single full tile that matches the logical tensor shape.
    if (tile_total == 1) {
        var t0 = try store.acquireTileMutLinear(id, 0);
        defer store.releaseMut(t0.token);
        const tv0 = t0.bufferView();

        var full_match: bool = (tv0.layout.shape.len == rank);
        d = 0;
        while (full_match and d < rank) : (d += 1) {
            full_match = (tv0.layout.shape[d] == meta.shape[d]);
        }

        if (full_match) {
            const need_bytes: usize = total * @sizeOf(f32);
            if (tv0.bytes.len < need_bytes) return BackendError.InvalidArgument;
            const dst_vals: []align(1) f32 = std.mem.bytesAsSlice(f32, tv0.bytes[0..need_bytes]);
            @memcpy(dst_vals[0..src.len], src);
            return;
        }
    }

    // Fast-path: batch-tiled rank-3 (NLC) or rank-4 (NHWC) where each tile is a full slice.
    if (rank == 3 and meta.tile_counts[1] == 1 and meta.tile_counts[2] == 1 and meta.tile_shape[0] == 1 and meta.tile_shape[1] == meta.shape[1] and meta.tile_shape[2] == meta.shape[2]) {
        const elems_per_tile: usize = meta.shape[1] * meta.shape[2];
        var ti: usize = 0;
        while (ti < meta.tile_counts[0]) : (ti += 1) {
            var t = try store.acquireTileMutLinear(id, ti);
            defer store.releaseMut(t.token);
            const tv = t.bufferView();
            std.debug.assert(tv.layout.shape.len == 3);
            std.debug.assert(tv.layout.shape[0] == 1 and tv.layout.shape[1] == meta.shape[1] and tv.layout.shape[2] == meta.shape[2]);
            const need_bytes: usize = elems_per_tile * @sizeOf(f32);
            std.debug.assert(tv.bytes.len >= need_bytes);
            const vals: []align(1) f32 = std.mem.bytesAsSlice(f32, tv.bytes[0..need_bytes]);
            const src_off: usize = ti * elems_per_tile;
            @memcpy(vals[0..elems_per_tile], src[src_off .. src_off + elems_per_tile]);
        }
        return;
    }

    // Fast-path: rank-3 tiles with padded channel dimension (tile_shape[2] >= shape[2]).
    if (rank == 3 and meta.tile_shape[0] == 1 and meta.tile_counts[2] == 1 and meta.tile_shape[2] >= meta.shape[2]) {
        const c: usize = meta.shape[2];
        var ti: usize = 0;
        while (ti < tile_total) : (ti += 1) {
            var t = try store.acquireTileMutLinear(id, ti);
            defer store.releaseMut(t.token);
            const tv = t.bufferView();
            const vals: []align(1) f32 = std.mem.bytesAsSlice(f32, tv.bytes);

            try tensor_store.decodeTileCoords(meta, ti, tile_coords[0..rank]);
            const base0: usize = tile_coords[0] * meta.tile_shape[0];
            const base1: usize = tile_coords[1] * meta.tile_shape[1];
            const t0: usize = tv.layout.shape[0];
            const t1: usize = tv.layout.shape[1];
            const t2: usize = tv.layout.shape[2];

            if (base0 + t0 <= meta.shape[0] and base1 + t1 <= meta.shape[1]) {
                var d0: usize = 0;
                while (d0 < t0) : (d0 += 1) {
                    const g0: usize = base0 + d0;
                    var d1: usize = 0;
                    while (d1 < t1) : (d1 += 1) {
                        const g1: usize = base1 + d1;
                        const li_base: usize = (d0 * t1 + d1) * t2;
                        const idx_base: usize = ((g0 * meta.shape[1] + g1) * meta.shape[2]);
                        @memcpy(vals[li_base .. li_base + c], src[idx_base .. idx_base + c]);
                    }
                }
                continue;
            }

            var d0: usize = 0;
            while (d0 < t0) : (d0 += 1) {
                const g0: usize = base0 + d0;
                if (g0 >= meta.shape[0]) continue;
                var d1: usize = 0;
                while (d1 < t1) : (d1 += 1) {
                    const g1: usize = base1 + d1;
                    if (g1 >= meta.shape[1]) continue;
                    const li_base: usize = (d0 * t1 + d1) * t2;
                    const idx_base: usize = (g0 * meta.shape[1] + g1) * meta.shape[2];
                    @memcpy(vals[li_base .. li_base + c], src[idx_base .. idx_base + c]);
                }
            }
        }
        return;
    }

    if (rank == 4 and meta.tile_counts[1] == 1 and meta.tile_counts[2] == 1 and meta.tile_counts[3] == 1 and meta.tile_shape[0] == 1 and meta.tile_shape[1] == meta.shape[1] and meta.tile_shape[2] == meta.shape[2] and meta.tile_shape[3] == meta.shape[3]) {
        const elems_per_tile: usize = meta.shape[1] * meta.shape[2] * meta.shape[3];
        var ti: usize = 0;
        while (ti < meta.tile_counts[0]) : (ti += 1) {
            var t = try store.acquireTileMutLinear(id, ti);
            defer store.releaseMut(t.token);
            const tv = t.bufferView();
            std.debug.assert(tv.layout.shape.len == 4);
            std.debug.assert(tv.layout.shape[0] == 1 and tv.layout.shape[1] == meta.shape[1] and tv.layout.shape[2] == meta.shape[2] and tv.layout.shape[3] == meta.shape[3]);
            const need_bytes: usize = elems_per_tile * @sizeOf(f32);
            std.debug.assert(tv.bytes.len >= need_bytes);
            const vals: []align(1) f32 = std.mem.bytesAsSlice(f32, tv.bytes[0..need_bytes]);
            const src_off: usize = ti * elems_per_tile;
            @memcpy(vals[0..elems_per_tile], src[src_off .. src_off + elems_per_tile]);
        }
        return;
    }

    var ti: usize = 0;
    while (ti < tile_total) : (ti += 1) {
        var t = try store.acquireTileMutLinear(id, ti);
        defer store.releaseMut(t.token);
        const tv = t.bufferView();
        const vals: []align(1) f32 = std.mem.bytesAsSlice(f32, tv.bytes);

        try tensor_store.decodeTileCoords(meta, ti, tile_coords[0..rank]);
        if (rank == 4) {
            const base0: usize = tile_coords[0] * meta.tile_shape[0];
            const base1: usize = tile_coords[1] * meta.tile_shape[1];
            const base2: usize = tile_coords[2] * meta.tile_shape[2];
            const base3: usize = tile_coords[3] * meta.tile_shape[3];

            const t0: usize = tv.layout.shape[0];
            const t1: usize = tv.layout.shape[1];
            const t2: usize = tv.layout.shape[2];
            const t3: usize = tv.layout.shape[3];

            if (base0 + t0 <= meta.shape[0] and base1 + t1 <= meta.shape[1] and base2 + t2 <= meta.shape[2] and base3 + t3 <= meta.shape[3]) {
                var d0: usize = 0;
                while (d0 < t0) : (d0 += 1) {
                    const g0: usize = base0 + d0;
                    var d1: usize = 0;
                    while (d1 < t1) : (d1 += 1) {
                        const g1: usize = base1 + d1;
                        var d2: usize = 0;
                        while (d2 < t2) : (d2 += 1) {
                            const g2: usize = base2 + d2;
                            const li_base: usize = ((d0 * t1 + d1) * t2 + d2) * t3;
                            const idx_base: usize = (((g0 * meta.shape[1] + g1) * meta.shape[2] + g2) * meta.shape[3]) + base3;
                            @memcpy(vals[li_base .. li_base + t3], src[idx_base .. idx_base + t3]);
                        }
                    }
                }
                continue;
            }

            var d0: usize = 0;
            while (d0 < t0) : (d0 += 1) {
                const g0: usize = base0 + d0;
                if (g0 >= meta.shape[0]) continue;
                var d1: usize = 0;
                while (d1 < t1) : (d1 += 1) {
                    const g1: usize = base1 + d1;
                    if (g1 >= meta.shape[1]) continue;
                    var d2: usize = 0;
                    while (d2 < t2) : (d2 += 1) {
                        const g2: usize = base2 + d2;
                        if (g2 >= meta.shape[2]) continue;
                        const li_base: usize = ((d0 * t1 + d1) * t2 + d2) * t3;
                        const idx_base: usize = ((g0 * meta.shape[1] + g1) * meta.shape[2] + g2) * meta.shape[3];
                        var d3: usize = 0;
                        while (d3 < t3) : (d3 += 1) {
                            const g3: usize = base3 + d3;
                            if (g3 >= meta.shape[3]) continue;
                            vals[li_base + d3] = src[idx_base + g3];
                        }
                    }
                }
            }
        } else if (rank == 3) {
            const base0: usize = tile_coords[0] * meta.tile_shape[0];
            const base1: usize = tile_coords[1] * meta.tile_shape[1];
            const base2: usize = tile_coords[2] * meta.tile_shape[2];

            const t0: usize = tv.layout.shape[0];
            const t1: usize = tv.layout.shape[1];
            const t2: usize = tv.layout.shape[2];

            if (base0 + t0 <= meta.shape[0] and base1 + t1 <= meta.shape[1] and base2 + t2 <= meta.shape[2]) {
                var d0: usize = 0;
                while (d0 < t0) : (d0 += 1) {
                    const g0: usize = base0 + d0;
                    var d1: usize = 0;
                    while (d1 < t1) : (d1 += 1) {
                        const g1: usize = base1 + d1;
                        const li_base: usize = (d0 * t1 + d1) * t2;
                        const idx_base: usize = ((g0 * meta.shape[1] + g1) * meta.shape[2]) + base2;
                        @memcpy(vals[li_base .. li_base + t2], src[idx_base .. idx_base + t2]);
                    }
                }
                continue;
            }

            var d0: usize = 0;
            while (d0 < t0) : (d0 += 1) {
                const g0: usize = base0 + d0;
                if (g0 >= meta.shape[0]) continue;
                var d1: usize = 0;
                while (d1 < t1) : (d1 += 1) {
                    const g1: usize = base1 + d1;
                    if (g1 >= meta.shape[1]) continue;
                    const li_base: usize = (d0 * t1 + d1) * t2;
                    const idx_base: usize = (g0 * meta.shape[1] + g1) * meta.shape[2];
                    var d2: usize = 0;
                    while (d2 < t2) : (d2 += 1) {
                        const g2: usize = base2 + d2;
                        if (g2 >= meta.shape[2]) continue;
                        vals[li_base + d2] = src[idx_base + g2];
                    }
                }
            }
        } else {
            try computePackedStrides(tv.layout.shape, local_strides[0..rank]);
            const local_total: usize = try elemCountFromShape(tv.layout.shape);

            var li: usize = 0;
            while (li < local_total) : (li += 1) {
                try decodeLinearToCoords(li, local_strides[0..rank], tv.layout.shape, local_coords[0..rank]);
                var idx: usize = 0;
                var in_bounds: bool = true;
                d = 0;
                while (d < rank) : (d += 1) {
                    global_coords[d] = tile_coords[d] * meta.tile_shape[d] + local_coords[d];
                    if (global_coords[d] >= meta.shape[d]) {
                        in_bounds = false;
                        break;
                    }
                    idx = std.math.add(usize, idx, global_coords[d] * packed_strides[d]) catch return BackendError.InvalidArgument;
                }
                if (!in_bounds) continue;
                vals[li] = src[idx];
            }
        }
    }
}
