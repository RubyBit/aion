// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");

const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");

const attention_registry = @import("../registry/attention_registry.zig");
const attn_kernels = @import("../kernels/attention.zig");
const fast_math = @import("../kernels/fast_math.zig");
const simd = @import("../kernels/simd.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;
const DType = types.DType;

const simd_lanes: usize = attn_kernels.simd_lanes;
const Vec = @Vector(simd_lanes, f32);

/// Logical -> physical time mapping for a KV cache. Derived from the tensor's
/// sequence-cache policy: `none`/`growable` are the identity (the executor pre-touches
/// growth before dispatch, so metadata is stable), `ring` is a modulo.
///
/// There used to be a third mode that called `store.mapSequenceStep` per key; it was
/// unreachable — `execAttentionTiled` only ever derives identity or ring — and it sat
/// in the innermost key loop of both score paths.
const TimeMapMode = enum {
    identity,
    ring,
};

const ConstTileCache = struct {
    valid: bool = false,
    tile_index: usize = 0,
    tile: tensor_store.TileRefConst = undefined,
};

const MutTileCache = struct {
    valid: bool = false,
    tile_index: usize = 0,
    tile: tensor_store.TileRefMut = undefined,
};

fn releaseConstCache(store: tensor_store.TensorStore, cache: *ConstTileCache) void {
    if (cache.valid) {
        store.releaseConst(cache.tile.token);
        cache.valid = false;
    }
}

fn releaseMutCache(store: tensor_store.TensorStore, cache: *MutTileCache) void {
    if (cache.valid) {
        store.releaseMut(cache.tile.token);
        cache.valid = false;
    }
}

fn acquireConstTileCached(
    store: tensor_store.TensorStore,
    id: executable.TensorId,
    tile_index: usize,
    cache: *ConstTileCache,
) ExecuteProgramError!tensor_store.TileRefConst {
    if (!cache.valid or cache.tile_index != tile_index) {
        if (cache.valid) store.releaseConst(cache.tile.token);
        cache.tile = try store.acquireTileConstLinear(id, tile_index);
        cache.tile_index = tile_index;
        cache.valid = true;
    }
    return cache.tile;
}

fn acquireMutTileCached(
    store: tensor_store.TensorStore,
    id: executable.TensorId,
    tile_index: usize,
    cache: *MutTileCache,
) ExecuteProgramError!tensor_store.TileRefMut {
    if (!cache.valid or cache.tile_index != tile_index) {
        if (cache.valid) store.releaseMut(cache.tile.token);
        cache.tile = try store.acquireTileMutLinear(id, tile_index);
        cache.tile_index = tile_index;
        cache.valid = true;
    }
    return cache.tile;
}

fn asPositiveStride(v: isize) ExecuteProgramError!usize {
    return std.math.cast(usize, v) orelse BackendError.InvalidArgument;
}

fn readI32Rank1(
    store: tensor_store.TensorStore,
    id: executable.TensorId,
    meta: tensor_store.TensorMeta,
    idx0: usize,
    cache: *ConstTileCache,
) ExecuteProgramError!i32 {
    if (meta.rank != 1 or meta.dtype != .i32) return BackendError.InvalidArgument;
    if (idx0 >= meta.shape[0]) return BackendError.InvalidArgument;

    const coords: [1]usize = .{idx0 / meta.tile_shape[0]};
    const tile_index: usize = try tensor_store.encodeTileIndex(meta, coords[0..]);
    const tile: tensor_store.TileRefConst = try acquireConstTileCached(store, id, tile_index, cache);

    const view = tile.bufferView();
    if (view.layout.rank != 1) return BackendError.InvalidArgument;
    const s0: usize = try asPositiveStride(view.layout.strides_bytes[0]);

    const l0: usize = idx0 - coords[0] * meta.tile_shape[0];
    const off: usize = std.math.mul(usize, l0, s0) catch return BackendError.InvalidArgument;
    const end: usize = std.math.add(usize, off, @sizeOf(i32)) catch return BackendError.InvalidArgument;
    if (end > view.bytes.len) return BackendError.InvalidArgument;

    const ptr: *align(1) const i32 = @ptrCast(view.bytes.ptr + off);
    return ptr.*;
}

fn readI32Rank2(
    store: tensor_store.TensorStore,
    id: executable.TensorId,
    meta: tensor_store.TensorMeta,
    idx0: usize,
    idx1: usize,
    cache: *ConstTileCache,
) ExecuteProgramError!i32 {
    if (meta.rank != 2 or meta.dtype != .i32) return BackendError.InvalidArgument;
    if (idx0 >= meta.shape[0] or idx1 >= meta.shape[1]) return BackendError.InvalidArgument;

    const coords: [2]usize = .{ idx0 / meta.tile_shape[0], idx1 / meta.tile_shape[1] };
    const tile_index: usize = try tensor_store.encodeTileIndex(meta, coords[0..]);
    const tile: tensor_store.TileRefConst = try acquireConstTileCached(store, id, tile_index, cache);

    const view = tile.bufferView();
    if (view.layout.rank != 2) return BackendError.InvalidArgument;
    const s0: usize = try asPositiveStride(view.layout.strides_bytes[0]);
    const s1: usize = try asPositiveStride(view.layout.strides_bytes[1]);

    const l0: usize = idx0 - coords[0] * meta.tile_shape[0];
    const l1: usize = idx1 - coords[1] * meta.tile_shape[1];

    const off0: usize = std.math.mul(usize, l0, s0) catch return BackendError.InvalidArgument;
    const off1: usize = std.math.mul(usize, l1, s1) catch return BackendError.InvalidArgument;
    const off: usize = std.math.add(usize, off0, off1) catch return BackendError.InvalidArgument;
    const end: usize = std.math.add(usize, off, @sizeOf(i32)) catch return BackendError.InvalidArgument;
    if (end > view.bytes.len) return BackendError.InvalidArgument;

    const ptr: *align(1) const i32 = @ptrCast(view.bytes.ptr + off);
    return ptr.*;
}

fn rowSliceRank4ConstF32(
    store: tensor_store.TensorStore,
    id: executable.TensorId,
    meta: tensor_store.TensorMeta,
    idx0: usize,
    idx1: usize,
    idx2: usize,
    row_len: usize,
    cache: *ConstTileCache,
) ExecuteProgramError![]align(1) const f32 {
    if (meta.rank != 4 or meta.dtype != .f32) return BackendError.InvalidArgument;
    if (meta.tile_counts[3] != 1) return BackendError.InvalidArgument;
    if (row_len != meta.shape[3]) return BackendError.InvalidArgument;
    if (idx0 >= meta.shape[0] or idx1 >= meta.shape[1] or idx2 >= meta.shape[2]) return BackendError.InvalidArgument;

    const coords: [4]usize = .{
        idx0 / meta.tile_shape[0],
        idx1 / meta.tile_shape[1],
        idx2 / meta.tile_shape[2],
        0,
    };
    const tile_index: usize = try tensor_store.encodeTileIndex(meta, coords[0..]);
    const tile: tensor_store.TileRefConst = try acquireConstTileCached(store, id, tile_index, cache);

    const view = tile.bufferView();
    if (view.layout.rank != 4 or view.dtype != .f32) return BackendError.InvalidArgument;

    const s0: usize = try asPositiveStride(view.layout.strides_bytes[0]);
    const s1: usize = try asPositiveStride(view.layout.strides_bytes[1]);
    const s2: usize = try asPositiveStride(view.layout.strides_bytes[2]);
    const s3: usize = try asPositiveStride(view.layout.strides_bytes[3]);
    if (s3 != @sizeOf(f32)) return BackendError.InvalidArgument;

    const l0: usize = idx0 - coords[0] * meta.tile_shape[0];
    const l1: usize = idx1 - coords[1] * meta.tile_shape[1];
    const l2: usize = idx2 - coords[2] * meta.tile_shape[2];

    const off0: usize = std.math.mul(usize, l0, s0) catch return BackendError.InvalidArgument;
    const off1: usize = std.math.mul(usize, l1, s1) catch return BackendError.InvalidArgument;
    const off2: usize = std.math.mul(usize, l2, s2) catch return BackendError.InvalidArgument;
    const off01: usize = std.math.add(usize, off0, off1) catch return BackendError.InvalidArgument;
    const off: usize = std.math.add(usize, off01, off2) catch return BackendError.InvalidArgument;

    const row_bytes: usize = std.math.mul(usize, row_len, @sizeOf(f32)) catch return BackendError.InvalidArgument;
    const row_end: usize = std.math.add(usize, off, row_bytes) catch return BackendError.InvalidArgument;
    if (row_end > view.bytes.len) return BackendError.InvalidArgument;

    return simd.bytesAsSliceConstUnaligned(f32, view.bytes[off..row_end]);
}

fn rowSliceRank4ConstF16(
    store: tensor_store.TensorStore,
    id: executable.TensorId,
    meta: tensor_store.TensorMeta,
    idx0: usize,
    idx1: usize,
    idx2: usize,
    row_len: usize,
    cache: *ConstTileCache,
) ExecuteProgramError![]align(1) const f16 {
    if (meta.rank != 4 or meta.dtype != .f16) return BackendError.InvalidArgument;
    if (meta.tile_counts[3] != 1) return BackendError.InvalidArgument;
    if (row_len != meta.shape[3]) return BackendError.InvalidArgument;
    if (idx0 >= meta.shape[0] or idx1 >= meta.shape[1] or idx2 >= meta.shape[2]) return BackendError.InvalidArgument;

    const coords: [4]usize = .{
        idx0 / meta.tile_shape[0],
        idx1 / meta.tile_shape[1],
        idx2 / meta.tile_shape[2],
        0,
    };
    const tile_index: usize = try tensor_store.encodeTileIndex(meta, coords[0..]);
    const tile: tensor_store.TileRefConst = try acquireConstTileCached(store, id, tile_index, cache);

    const view = tile.bufferView();
    if (view.layout.rank != 4 or view.dtype != .f16) return BackendError.InvalidArgument;

    const s0: usize = try asPositiveStride(view.layout.strides_bytes[0]);
    const s1: usize = try asPositiveStride(view.layout.strides_bytes[1]);
    const s2: usize = try asPositiveStride(view.layout.strides_bytes[2]);
    const s3: usize = try asPositiveStride(view.layout.strides_bytes[3]);
    if (s3 != @sizeOf(f16)) return BackendError.InvalidArgument;

    const l0: usize = idx0 - coords[0] * meta.tile_shape[0];
    const l1: usize = idx1 - coords[1] * meta.tile_shape[1];
    const l2: usize = idx2 - coords[2] * meta.tile_shape[2];

    const off0: usize = std.math.mul(usize, l0, s0) catch return BackendError.InvalidArgument;
    const off1: usize = std.math.mul(usize, l1, s1) catch return BackendError.InvalidArgument;
    const off2: usize = std.math.mul(usize, l2, s2) catch return BackendError.InvalidArgument;
    const off01: usize = std.math.add(usize, off0, off1) catch return BackendError.InvalidArgument;
    const off: usize = std.math.add(usize, off01, off2) catch return BackendError.InvalidArgument;

    const row_bytes: usize = std.math.mul(usize, row_len, @sizeOf(f16)) catch return BackendError.InvalidArgument;
    const row_end: usize = std.math.add(usize, off, row_bytes) catch return BackendError.InvalidArgument;
    if (row_end > view.bytes.len) return BackendError.InvalidArgument;

    return simd.bytesAsSliceConstUnaligned(f16, view.bytes[off..row_end]);
}

fn rowSliceRank4MutF32(
    store: tensor_store.TensorStore,
    id: executable.TensorId,
    meta: tensor_store.TensorMeta,
    idx0: usize,
    idx1: usize,
    idx2: usize,
    row_len: usize,
    cache: *MutTileCache,
) ExecuteProgramError![]align(1) f32 {
    if (meta.rank != 4 or meta.dtype != .f32) return BackendError.InvalidArgument;
    if (meta.tile_counts[3] != 1) return BackendError.InvalidArgument;
    if (row_len != meta.shape[3]) return BackendError.InvalidArgument;
    if (idx0 >= meta.shape[0] or idx1 >= meta.shape[1] or idx2 >= meta.shape[2]) return BackendError.InvalidArgument;

    const coords: [4]usize = .{
        idx0 / meta.tile_shape[0],
        idx1 / meta.tile_shape[1],
        idx2 / meta.tile_shape[2],
        0,
    };
    const tile_index: usize = try tensor_store.encodeTileIndex(meta, coords[0..]);
    const tile: tensor_store.TileRefMut = try acquireMutTileCached(store, id, tile_index, cache);

    const view = tile.bufferView();
    if (view.layout.rank != 4 or view.dtype != .f32) return BackendError.InvalidArgument;

    const s0: usize = try asPositiveStride(view.layout.strides_bytes[0]);
    const s1: usize = try asPositiveStride(view.layout.strides_bytes[1]);
    const s2: usize = try asPositiveStride(view.layout.strides_bytes[2]);
    const s3: usize = try asPositiveStride(view.layout.strides_bytes[3]);
    if (s3 != @sizeOf(f32)) return BackendError.InvalidArgument;

    const l0: usize = idx0 - coords[0] * meta.tile_shape[0];
    const l1: usize = idx1 - coords[1] * meta.tile_shape[1];
    const l2: usize = idx2 - coords[2] * meta.tile_shape[2];

    const off0: usize = std.math.mul(usize, l0, s0) catch return BackendError.InvalidArgument;
    const off1: usize = std.math.mul(usize, l1, s1) catch return BackendError.InvalidArgument;
    const off2: usize = std.math.mul(usize, l2, s2) catch return BackendError.InvalidArgument;
    const off01: usize = std.math.add(usize, off0, off1) catch return BackendError.InvalidArgument;
    const off: usize = std.math.add(usize, off01, off2) catch return BackendError.InvalidArgument;

    const row_bytes: usize = std.math.mul(usize, row_len, @sizeOf(f32)) catch return BackendError.InvalidArgument;
    const row_end: usize = std.math.add(usize, off, row_bytes) catch return BackendError.InvalidArgument;
    if (row_end > view.bytes.len) return BackendError.InvalidArgument;

    return simd.bytesAsSliceMutUnaligned(f32, view.bytes[off..row_end]);
}

inline fn vecLoad(ptr: [*]align(1) const f32) Vec {
    return @as(*align(1) const [simd_lanes]f32, @ptrCast(ptr)).*;
}

inline fn vecStore(ptr: [*]align(1) f32, v: Vec) void {
    @as(*align(1) [simd_lanes]f32, @ptrCast(ptr)).* = v;
}

inline fn dotRowF32(a: []align(1) const f32, b: []align(1) const f32, n: usize) f32 {
    var acc: f32 = 0.0;
    var i: usize = 0;

    if (n >= simd_lanes) {
        var vacc: Vec = @splat(0.0);
        const vec_end: usize = n - (n % simd_lanes);
        while (i < vec_end) : (i += simd_lanes) {
            vacc += vecLoad(a[i..].ptr) * vecLoad(b[i..].ptr);
        }
        acc += @reduce(.Add, vacc);
    }

    while (i < n) : (i += 1) {
        acc += a[i] * b[i];
    }
    return acc;
}

inline fn dotRowF16F16(a: []align(1) const f16, b: []align(1) const f16, n: usize) f32 {
    const VF = Vec;
    const VH = @Vector(simd_lanes, f16);

    var acc: f32 = 0.0;
    var i: usize = 0;
    if (n >= simd_lanes) {
        var vacc: VF = @splat(0.0);
        const vec_end: usize = n - (n % simd_lanes);
        while (i < vec_end) : (i += simd_lanes) {
            const ah: VH = @as(*align(1) const VH, @ptrCast(a[i..].ptr)).*;
            const bh: VH = @as(*align(1) const VH, @ptrCast(b[i..].ptr)).*;
            const af: VF = @floatCast(ah);
            const bf: VF = @floatCast(bh);
            vacc = @mulAdd(VF, af, bf, vacc);
        }
        acc += @reduce(.Add, vacc);
    }
    while (i < n) : (i += 1) {
        acc += @as(f32, @floatCast(a[i])) * @as(f32, @floatCast(b[i]));
    }
    return acc;
}

inline fn dotRowF32F16(a: []align(1) const f32, b: []align(1) const f16, n: usize) f32 {
    const VF = Vec;
    const VH = @Vector(simd_lanes, f16);

    var acc: f32 = 0.0;
    var i: usize = 0;
    if (n >= simd_lanes) {
        var vacc: VF = @splat(0.0);
        const vec_end: usize = n - (n % simd_lanes);
        while (i < vec_end) : (i += simd_lanes) {
            const av: VF = @as(*align(1) const VF, @ptrCast(a[i..].ptr)).*;
            const bh: VH = @as(*align(1) const VH, @ptrCast(b[i..].ptr)).*;
            const bf: VF = @floatCast(bh);
            vacc = @mulAdd(VF, av, bf, vacc);
        }
        acc += @reduce(.Add, vacc);
    }
    while (i < n) : (i += 1) {
        acc += a[i] * @as(f32, @floatCast(b[i]));
    }
    return acc;
}

inline fn dotRowF16F32(a: []align(1) const f16, b: []align(1) const f32, n: usize) f32 {
    const VF = Vec;
    const VH = @Vector(simd_lanes, f16);

    var acc: f32 = 0.0;
    var i: usize = 0;
    if (n >= simd_lanes) {
        var vacc: VF = @splat(0.0);
        const vec_end: usize = n - (n % simd_lanes);
        while (i < vec_end) : (i += simd_lanes) {
            const ah: VH = @as(*align(1) const VH, @ptrCast(a[i..].ptr)).*;
            const bv: VF = @as(*align(1) const VF, @ptrCast(b[i..].ptr)).*;
            const af: VF = @floatCast(ah);
            vacc = @mulAdd(VF, af, bv, vacc);
        }
        acc += @reduce(.Add, vacc);
    }
    while (i < n) : (i += 1) {
        acc += @as(f32, @floatCast(a[i])) * b[i];
    }
    return acc;
}

inline fn dotRowsDynamic(
    q_dtype: DType,
    q_row_f32: ?[]align(1) const f32,
    q_row_f16: ?[]align(1) const f16,
    k_dtype: DType,
    k_row_f32: ?[]align(1) const f32,
    k_row_f16: ?[]align(1) const f16,
    n: usize,
) f32 {
    if (q_dtype == .f32 and k_dtype == .f32) return dotRowF32(q_row_f32.?, k_row_f32.?, n);
    if (q_dtype == .f32 and k_dtype == .f16) return dotRowF32F16(q_row_f32.?, k_row_f16.?, n);
    if (q_dtype == .f16 and k_dtype == .f32) return dotRowF16F32(q_row_f16.?, k_row_f32.?, n);
    return dotRowF16F16(q_row_f16.?, k_row_f16.?, n);
}

inline fn fusedRescaleAccumulateRowF32(
    out_row: []align(1) f32,
    v_row: []align(1) const f32,
    d_v: usize,
    rescale: f32,
    p_new: f32,
) void {
    var i: usize = 0;
    const v_rescale: Vec = @splat(rescale);
    const v_pnew: Vec = @splat(p_new);
    const vec_end: usize = d_v - (d_v % simd_lanes);

    while (i < vec_end) : (i += simd_lanes) {
        const vo: Vec = vecLoad(out_row[i..].ptr);
        const vv: Vec = vecLoad(v_row[i..].ptr);
        vecStore(out_row[i..].ptr, vo * v_rescale + vv * v_pnew);
    }
    while (i < d_v) : (i += 1) {
        out_row[i] = out_row[i] * rescale + v_row[i] * p_new;
    }
}

inline fn accumulateRowScaledF32(
    out_row: []align(1) f32,
    v_row: []align(1) const f32,
    d_v: usize,
    alpha: f32,
) void {
    var i: usize = 0;
    const v_alpha: Vec = @splat(alpha);
    const vec_end: usize = d_v - (d_v % simd_lanes);

    while (i < vec_end) : (i += simd_lanes) {
        const vo: Vec = vecLoad(out_row[i..].ptr);
        const vv: Vec = vecLoad(v_row[i..].ptr);
        vecStore(out_row[i..].ptr, vo + vv * v_alpha);
    }
    while (i < d_v) : (i += 1) {
        out_row[i] += v_row[i] * alpha;
    }
}

inline fn fusedRescaleAccumulateRowF16(
    out_row: []align(1) f32,
    v_row: []align(1) const f16,
    d_v: usize,
    rescale: f32,
    p_new: f32,
) void {
    const VF = Vec;
    const VH = @Vector(simd_lanes, f16);
    const v_rescale: VF = @splat(rescale);
    const v_pnew: VF = @splat(p_new);

    var i: usize = 0;
    const vec_end: usize = d_v - (d_v % simd_lanes);
    while (i < vec_end) : (i += simd_lanes) {
        const vo: VF = vecLoad(out_row[i..].ptr);
        const vh: VH = @as(*align(1) const VH, @ptrCast(v_row[i..].ptr)).*;
        const vv: VF = @floatCast(vh);
        vecStore(out_row[i..].ptr, vo * v_rescale + vv * v_pnew);
    }
    while (i < d_v) : (i += 1) {
        out_row[i] = out_row[i] * rescale + @as(f32, @floatCast(v_row[i])) * p_new;
    }
}

inline fn accumulateRowScaledF16(
    out_row: []align(1) f32,
    v_row: []align(1) const f16,
    d_v: usize,
    alpha: f32,
) void {
    const VF = Vec;
    const VH = @Vector(simd_lanes, f16);
    const v_alpha: VF = @splat(alpha);

    var i: usize = 0;
    const vec_end: usize = d_v - (d_v % simd_lanes);
    while (i < vec_end) : (i += simd_lanes) {
        const vo: VF = vecLoad(out_row[i..].ptr);
        const vh: VH = @as(*align(1) const VH, @ptrCast(v_row[i..].ptr)).*;
        const vv: VF = @floatCast(vh);
        vecStore(out_row[i..].ptr, @mulAdd(VF, vv, v_alpha, vo));
    }
    while (i < d_v) : (i += 1) {
        out_row[i] += @as(f32, @floatCast(v_row[i])) * alpha;
    }
}

inline fn accumulateScaledDynamic(
    v_dtype: DType,
    out_row: []align(1) f32,
    v_row_f32: ?[]align(1) const f32,
    v_row_f16: ?[]align(1) const f16,
    d_v: usize,
    alpha: f32,
) void {
    if (v_dtype == .f32) {
        accumulateRowScaledF32(out_row, v_row_f32.?, d_v, alpha);
    } else {
        accumulateRowScaledF16(out_row, v_row_f16.?, d_v, alpha);
    }
}

inline fn fusedRescaleAccumulateDynamic(
    v_dtype: DType,
    out_row: []align(1) f32,
    v_row_f32: ?[]align(1) const f32,
    v_row_f16: ?[]align(1) const f16,
    d_v: usize,
    rescale: f32,
    p_new: f32,
) void {
    if (v_dtype == .f32) {
        fusedRescaleAccumulateRowF32(out_row, v_row_f32.?, d_v, rescale, p_new);
    } else {
        fusedRescaleAccumulateRowF16(out_row, v_row_f16.?, d_v, rescale, p_new);
    }
}

inline fn scaleRowF32(row: []align(1) f32, n: usize, scale: f32) void {
    var i: usize = 0;
    const v_scale: Vec = @splat(scale);
    const vec_end: usize = n - (n % simd_lanes);

    while (i < vec_end) : (i += simd_lanes) {
        vecStore(row[i..].ptr, vecLoad(row[i..].ptr) * v_scale);
    }
    while (i < n) : (i += 1) row[i] *= scale;
}

// ===========================================================================
// Blocked (flash-style) f32 path
// ===========================================================================
//
// `ExecCtx` below walks ONE (query row, key) pair at a time and re-derives a tile
// pointer from tensor metadata for every key — divisions, stride casts, overflow
// and bounds checks around a single d_k-long dot product — and every query row
// re-streams the whole K/V range from memory. Both costs dominate the arithmetic.
//
// This path instead:
//   * groups query rows that share a K/V panel. K/V depend only on (batch, kv
//     head), so the group spans multiple L rows (prefill) AND all query heads of
//     one GQA group (decode) — the shared K row is loaded once for the whole group;
//   * walks keys in blocks with the panel pointer hoisted out of the key loop;
//   * runs each block as GEMM -> online softmax -> GEMM through the same
//     SIMD-tuned panel kernels RelPosMHA already uses, with a narrow-row variant
//     for when a group has fewer rows than the microkernel's `mr`;
//   * skips key blocks entirely outside a block's window range;
//   * can split ONE group's key range across workers (flash-decoding). That is the
//     only available parallelism when decode has a single query row, where work
//     partitioned by output tile leaves every thread but one idle.
//
// Restricted to f32 q/k/v/out and identity time mapping; f16 caches and
// ring mapping fall through to the generic path, which stays authoritative.

/// Comptime bound for the per-row stack descriptors below. The block size actually
/// walked is the selected variant's `tuning.row_block` (carried as `BlockedCtx.row_block`),
/// which this bounds; see `attention_registry.Tuning` for the blocking trade-offs.
const MAX_ROWS: usize = attention_registry.max_row_block;

fn ceilDiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

fn alignUp(x: usize, a: usize) usize {
    return ceilDiv(x, a) * a;
}

/// Rows `t .. t + rows` of one (batch, head) slice inside a SINGLE tile, at a
/// constant element stride — the shape the panel kernels want. Derived once per
/// tile instead of once per key.
const Panel = struct {
    data: []align(1) const f32,
    row_stride: usize,
    rows: usize,
};

fn panelF32(
    store: tensor_store.TensorStore,
    id: executable.TensorId,
    meta: tensor_store.TensorMeta,
    b: usize,
    t: usize,
    h: usize,
    d: usize,
    cache: *ConstTileCache,
) ExecuteProgramError!Panel {
    if (meta.rank != 4 or meta.dtype != .f32) return BackendError.InvalidArgument;
    if (t >= meta.shape[1] or b >= meta.shape[0] or h >= meta.shape[2]) return BackendError.InvalidArgument;

    const coords: [4]usize = .{
        b / meta.tile_shape[0],
        t / meta.tile_shape[1],
        h / meta.tile_shape[2],
        0,
    };
    const tile_index: usize = try tensor_store.encodeTileIndex(meta, coords[0..]);
    const tile: tensor_store.TileRefConst = try acquireConstTileCached(store, id, tile_index, cache);

    const view = tile.bufferView();
    if (view.layout.rank != 4 or view.dtype != .f32) return BackendError.InvalidArgument;
    const s0: usize = try asPositiveStride(view.layout.strides_bytes[0]);
    const s1: usize = try asPositiveStride(view.layout.strides_bytes[1]);
    const s2: usize = try asPositiveStride(view.layout.strides_bytes[2]);
    const s3: usize = try asPositiveStride(view.layout.strides_bytes[3]);
    if (s3 != @sizeOf(f32) or (s1 % @sizeOf(f32)) != 0) return BackendError.InvalidArgument;

    const b_l: usize = b - coords[0] * meta.tile_shape[0];
    const t_l: usize = t - coords[1] * meta.tile_shape[1];
    const h_l: usize = h - coords[2] * meta.tile_shape[2];
    const off: usize = b_l * s0 + t_l * s1 + h_l * s2;

    const tile_rows: usize = tile.shape_mem[1];
    if (t_l >= tile_rows) return BackendError.InvalidArgument;
    const rows: usize = tile_rows - t_l;

    // Bytes actually spanned by [row 0 .. row rows-1] of this panel.
    const span: usize = (rows - 1) * s1 + d * @sizeOf(f32);
    const end: usize = std.math.add(usize, off, span) catch return BackendError.InvalidArgument;
    if (end > view.bytes.len) return BackendError.InvalidArgument;

    return .{
        .data = simd.bytesAsSliceConstUnaligned(f32, view.bytes[off..end]),
        .row_stride = s1 / @sizeOf(f32),
        .rows = rows,
    };
}

/// One parallel work item: the query rows of a (batch, kv head) group covering an
/// L chunk and a head chunk, over key segment `seg` of that group's range.
const Unit = struct {
    b: usize,
    hkv: usize,
    l0: usize,
    l_n: usize,
    hq0: usize,
    hq_n: usize,
    seg: usize,
};

const Worker = struct {
    qp: []f32, // [row_block, d_k] gathered query rows, pre-scaled
    scores: []f32, // [row_block, key_block]
    kt: []f32, // packed K panel
    qt: []f32, // packed Q panel
    acc_local: []f32, // [row_block, d_v] when segs == 1

    q_cache: ConstTileCache = .{},
    k_cache: ConstTileCache = .{},
    v_cache: ConstTileCache = .{},
    pos_cache: ConstTileCache = .{},
    end_cache: ConstTileCache = .{},
    out_cache: MutTileCache = .{},

    fn release(self: *Worker, store: tensor_store.TensorStore) void {
        releaseConstCache(store, &self.q_cache);
        releaseConstCache(store, &self.k_cache);
        releaseConstCache(store, &self.v_cache);
        releaseConstCache(store, &self.pos_cache);
        releaseConstCache(store, &self.end_cache);
        releaseMutCache(store, &self.out_cache);
    }
};

const BlockedCtx = struct {
    store: tensor_store.TensorStore,
    kernels: attention_registry.Kernels,
    s: executable.StepAttentionTiled,

    q_meta: tensor_store.TensorMeta,
    k_meta: tensor_store.TensorMeta,
    v_meta: tensor_store.TensorMeta,
    positions_meta: ?tensor_store.TensorMeta,
    lengths_meta: ?tensor_store.TensorMeta,
    out_meta: tensor_store.TensorMeta,

    batch: usize,
    l_q: usize,
    h_q: usize,
    h_kv: usize,
    d_k: usize,
    d_v: usize,
    groups_per_kv: usize,
    k_cap: usize,

    row_block: usize, // `kernels.tuning.row_block`: query rows per unit, bounded by MAX_ROWS
    key_block: usize, // `kernels.tuning.key_block`: keys per block
    lb: usize, // L rows per unit
    hb: usize, // group-local heads per unit
    n_l_chunks: usize,
    n_h_chunks: usize,
    segs: usize,

    scratch: []f32,
    per_worker: usize,
    /// [unit, seg] partials: `row_block * d_v` accumulators then `row_block` m and
    /// `row_block` l.
    /// Empty when `segs == 1` (units finalize in place).
    partials: []f32,
    part_stride: usize,

    stop: std.atomic.Value(bool) = .init(false),
    err_mutex: std.Io.Mutex = .init,
    err_any: ?anyerror = null,

    fn fail(self: *@This(), err: anyerror) void {
        if (self.stop.swap(true, .acq_rel)) return;
        std.Io.Threaded.mutexLock(&self.err_mutex);
        defer std.Io.Threaded.mutexUnlock(&self.err_mutex);
        if (self.err_any == null) self.err_any = err;
    }

    fn unitsPerSlice(self: *const @This()) usize {
        return self.n_l_chunks * self.n_h_chunks;
    }

    fn totalUnits(self: *const @This()) usize {
        return self.batch * self.h_kv * self.unitsPerSlice();
    }

    fn unitAt(self: *const @This(), idx: usize) Unit {
        const ups: usize = self.unitsPerSlice();
        const u_in_slice: usize = idx % ups;
        const slice: usize = idx / ups;
        const hkv: usize = slice % self.h_kv;
        const b: usize = slice / self.h_kv;

        const h_chunk: usize = u_in_slice % self.n_h_chunks;
        const l_chunk: usize = u_in_slice / self.n_h_chunks;

        const l0: usize = l_chunk * self.lb;
        const g0: usize = h_chunk * self.hb;
        return .{
            .b = b,
            .hkv = hkv,
            .l0 = l0,
            .l_n = @min(self.lb, self.l_q - l0),
            .hq0 = hkv * self.groups_per_kv + g0,
            .hq_n = @min(self.hb, self.groups_per_kv - g0),
            .seg = 0,
        };
    }

    /// Accumulate `unit`'s key segment into `acc`/`m_state`/`l_state`. When
    /// `segs == 1` this is the whole range and the unit finalizes its own out rows.
    fn computeUnit(
        self: *@This(),
        u: Unit,
        w: *Worker,
        acc: []f32,
        m_state: []f32,
        l_state: []f32,
    ) ExecuteProgramError!void {
        const d_k: usize = self.d_k;
        const d_v: usize = self.d_v;
        const rows: usize = u.l_n * u.hq_n;
        if (rows == 0) return;
        if (rows > self.row_block) return BackendError.InvalidArgument;

        var row_l: [MAX_ROWS]usize = undefined;
        var row_h: [MAX_ROWS]usize = undefined;
        var row_lower: [MAX_ROWS]usize = undefined;
        var row_upper: [MAX_ROWS]usize = undefined;
        var lo_blk: [MAX_ROWS]u32 = undefined;
        var hi_blk: [MAX_ROWS]u32 = undefined;
        var m_new: [MAX_ROWS]f32 = undefined;
        var resc: [MAX_ROWS]f32 = undefined;

        // --- live key range for this batch ---
        var valid_end: usize = self.k_cap;
        if (self.lengths_meta) |meta| {
            const e: i32 = try readI32Rank1(self.store, self.s.kv_lengths.?, meta, u.b, &w.end_cache);
            if (e < 0) return BackendError.InvalidArgument;
            valid_end = @intCast(e);
            if (valid_end > self.k_cap) return BackendError.InvalidArgument;
        }

        // --- per-row descriptors: position, then the [lower, upper) key range ---
        var uni_lo: usize = std.math.maxInt(usize);
        var uni_hi: usize = 0;
        var r: usize = 0;
        var li: usize = 0;
        while (li < u.l_n) : (li += 1) {
            const l: usize = u.l0 + li;
            const q_pos: usize = if (self.positions_meta) |meta| blk: {
                const p: i32 = try readI32Rank2(self.store, self.s.query_positions.?, meta, u.b, l, &w.pos_cache);
                if (p < 0) return BackendError.InvalidArgument;
                break :blk @intCast(p);
            } else l;

            const win = self.s.window.keys(q_pos, valid_end);
            const lower: usize = win.lo;
            const upper: usize = win.hi;
            if (upper > lower) {
                uni_lo = @min(uni_lo, lower);
                uni_hi = @max(uni_hi, upper);
            }

            var hi: usize = 0;
            while (hi < u.hq_n) : (hi += 1) {
                row_l[r] = l;
                row_h[r] = u.hq0 + hi;
                row_lower[r] = lower;
                row_upper[r] = upper;
                r += 1;
            }
        }

        @memset(acc[0 .. rows * d_v], 0.0);
        @memset(m_state[0..rows], -std.math.inf(f32));
        @memset(l_state[0..rows], 0.0);

        if (uni_hi <= uni_lo) {
            // Every row's window is empty: zeros, matching the generic path.
            if (self.segs == 1) try self.finalize(u, w, rows, &row_l, &row_h, acc, l_state, &row_lower, &row_upper);
            return;
        }

        // --- this worker's slice of the group's key range ---
        const span: usize = uni_hi - uni_lo;
        const seg_len: usize = ceilDiv(span, self.segs);
        const t_start: usize = uni_lo + u.seg * seg_len;
        const t_end: usize = @min(t_start + seg_len, uni_hi);

        // --- gather q rows once per unit, folding in `scale` ---
        var i: usize = 0;
        while (i < rows) : (i += 1) {
            const q_row = try rowSliceRank4ConstF32(self.store, self.s.q, self.q_meta, u.b, row_l[i], row_h[i], d_k, &w.q_cache);
            const dst = w.qp[i * d_k ..][0..d_k];
            var j: usize = 0;
            while (j < d_k) : (j += 1) dst[j] = q_row[j] * self.s.scale;
        }

        const narrow: bool = rows < self.kernels.tuning.mr;

        var t: usize = t_start;
        while (t < t_end) {
            if (self.stop.load(.acquire)) return;

            const kp: Panel = try panelF32(self.store, self.s.k, self.k_meta, u.b, t, u.hkv, d_k, &w.k_cache);
            const vp: Panel = try panelF32(self.store, self.s.v, self.v_meta, u.b, t, u.hkv, d_v, &w.v_cache);
            const n: usize = @min(@min(kp.rows, vp.rows), @min(self.key_block, t_end - t));
            if (n == 0) return BackendError.InvalidArgument;

            // Per-row valid columns inside [t, t + n) — one contiguous interval,
            // so a masked block needs no mask matrix, and a block outside every
            // row's range is skipped outright (the causal-triangle win).
            var any: bool = false;
            i = 0;
            while (i < rows) : (i += 1) {
                const lo: usize = if (row_lower[i] > t) @min(row_lower[i] - t, n) else 0;
                const hi: usize = if (row_upper[i] > t) @min(row_upper[i] - t, n) else 0;
                lo_blk[i] = @intCast(lo);
                hi_blk[i] = @intCast(if (hi > lo) hi else lo);
                if (hi > lo) any = true;
            }

            if (any) {
                const scores = w.scores[0 .. rows * n];
                @memset(scores, 0.0);

                if (narrow) {
                    self.kernels.calc_scores_narrow_f32(rows, n, d_k, w.qp, d_k, kp.data, kp.row_stride, scores, n);
                } else {
                    self.kernels.pack_k_block(n, d_k, kp.data, kp.row_stride, w.kt);
                    self.kernels.calc_scores_f32(rows, n, d_k, w.qp, d_k, kp.data, kp.row_stride, w.kt, w.qt, scores, n);
                }

                if (self.s.attn_logits_soft_cap > 0.0) {
                    self.kernels.soft_cap_range_f32(scores, rows, n, lo_blk[0..rows], hi_blk[0..rows], self.s.attn_logits_soft_cap);
                }

                @memcpy(m_new[0..rows], m_state[0..rows]);
                self.kernels.row_max_range_f32(scores, rows, n, lo_blk[0..rows], hi_blk[0..rows], m_new[0..rows]);

                i = 0;
                while (i < rows) : (i += 1) {
                    resc[i] = if (m_state[i] == -std.math.inf(f32)) 1.0 else self.kernels.exp_softmax(m_state[i] - m_new[i]);
                    l_state[i] *= resc[i];
                }
                self.kernels.rescale_rows_f32(acc, rows, d_v, d_v, resc[0..rows]);

                self.kernels.exp_normalize_range_f32(scores, rows, n, n, lo_blk[0..rows], hi_blk[0..rows], m_new[0..rows], l_state[0..rows]);
                self.kernels.accumulate_values_f32(rows, n, d_v, scores, n, vp.data, vp.row_stride, acc, d_v);

                @memcpy(m_state[0..rows], m_new[0..rows]);
            }

            t += n;
        }

        if (self.segs == 1) try self.finalize(u, w, rows, &row_l, &row_h, acc, l_state, &row_lower, &row_upper);
    }

    /// `out = acc / l`, or zeros for a row whose window was empty.
    fn finalize(
        self: *@This(),
        u: Unit,
        w: *Worker,
        rows: usize,
        row_l: *const [MAX_ROWS]usize,
        row_h: *const [MAX_ROWS]usize,
        acc: []const f32,
        l_state: []const f32,
        row_lower: *const [MAX_ROWS]usize,
        row_upper: *const [MAX_ROWS]usize,
    ) ExecuteProgramError!void {
        const d_v: usize = self.d_v;
        var i: usize = 0;
        while (i < rows) : (i += 1) {
            const out_row = try rowSliceRank4MutF32(self.store, self.s.out, self.out_meta, u.b, row_l[i], row_h[i], d_v, &w.out_cache);
            const empty: bool = row_upper[i] <= row_lower[i];
            const denom: f32 = l_state[i];
            if (empty) {
                @memset(out_row[0..d_v], 0.0);
                continue;
            }
            if (!(denom > 0.0) or !std.math.isFinite(denom)) return BackendError.InvalidArgument;
            const inv: f32 = 1.0 / denom;
            const src = acc[i * d_v ..][0..d_v];
            var j: usize = 0;
            while (j < d_v) : (j += 1) out_row[j] = src[j] * inv;
        }
    }

    fn runRange(self: *@This(), start: usize, end: usize, tid: usize) void {
        if (start >= end) return;

        const slots: usize = @max(@as(usize, 1), self.scratch.len / self.per_worker);
        const base: usize = (tid % slots) * self.per_worker;
        var off: usize = base;
        const d_k: usize = self.d_k;
        const d_v: usize = self.d_v;
        const take = struct {
            fn f(buf: []f32, o: *usize, n: usize) []f32 {
                const out = buf[o.* .. o.* + n];
                o.* += n;
                return out;
            }
        }.f;

        var w: Worker = .{
            .qp = take(self.scratch, &off, self.row_block * d_k),
            .scores = take(self.scratch, &off, self.row_block * self.key_block),
            .kt = take(self.scratch, &off, alignUp(self.key_block, self.kernels.tuning.nr) * d_k),
            .qt = take(self.scratch, &off, alignUp(self.row_block, self.kernels.tuning.mr) * d_k),
            .acc_local = take(self.scratch, &off, self.row_block * d_v),
        };
        defer w.release(self.store);

        var m_local: [MAX_ROWS]f32 = undefined;
        var l_local: [MAX_ROWS]f32 = undefined;

        var idx: usize = start;
        while (idx < end) : (idx += 1) {
            if (self.stop.load(.acquire)) return;

            var u: Unit = self.unitAt(idx / self.segs);
            u.seg = idx % self.segs;

            if (self.segs == 1) {
                self.computeUnit(u, &w, w.acc_local, m_local[0..], l_local[0..]) catch |e| {
                    self.fail(e);
                    return;
                };
            } else {
                const slot = self.partials[idx * self.part_stride ..][0..self.part_stride];
                const acc = slot[0 .. self.row_block * d_v];
                const ml = slot[self.row_block * d_v ..];
                self.computeUnit(u, &w, acc, ml[0..self.row_block], ml[self.row_block..][0..self.row_block]) catch |e| {
                    self.fail(e);
                    return;
                };
            }
        }
    }

    fn runWork(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
        const self: *@This() = @ptrCast(@alignCast(ctx_any));
        self.runRange(start, end, tid);
    }

    /// Log-sum-exp combine of a unit's per-segment partials, then write out rows.
    /// Serial: only reached when the key range was split, i.e. when there were far
    /// fewer units than threads (decode), so this is a handful of rows.
    fn mergeUnits(self: *@This()) ExecuteProgramError!void {
        const d_v: usize = self.d_v;
        var w: Worker = .{ .qp = &.{}, .scores = &.{}, .kt = &.{}, .qt = &.{}, .acc_local = &.{} };
        defer w.release(self.store);

        var m_out: [MAX_ROWS]f32 = undefined;
        var l_out: [MAX_ROWS]f32 = undefined;
        var row_l: [MAX_ROWS]usize = undefined;
        var row_h: [MAX_ROWS]usize = undefined;
        var row_lower: [MAX_ROWS]usize = undefined;
        var row_upper: [MAX_ROWS]usize = undefined;

        const units: usize = self.totalUnits();
        var ui: usize = 0;
        while (ui < units) : (ui += 1) {
            const u: Unit = self.unitAt(ui);
            const rows: usize = u.l_n * u.hq_n;
            if (rows == 0) continue;

            // Row identity + emptiness must match `computeUnit`'s derivation.
            var valid_end: usize = self.k_cap;
            if (self.lengths_meta) |meta| {
                const e: i32 = try readI32Rank1(self.store, self.s.kv_lengths.?, meta, u.b, &w.end_cache);
                if (e < 0) return BackendError.InvalidArgument;
                valid_end = @intCast(e);
            }
            var r: usize = 0;
            var li: usize = 0;
            while (li < u.l_n) : (li += 1) {
                const l: usize = u.l0 + li;
                const q_pos: usize = if (self.positions_meta) |meta| blk: {
                    const p: i32 = try readI32Rank2(self.store, self.s.query_positions.?, meta, u.b, l, &w.pos_cache);
                    if (p < 0) return BackendError.InvalidArgument;
                    break :blk @intCast(p);
                } else l;
                const win = self.s.window.keys(q_pos, valid_end);
                const lower: usize = win.lo;
                const upper: usize = win.hi;
                var hi: usize = 0;
                while (hi < u.hq_n) : (hi += 1) {
                    row_l[r] = l;
                    row_h[r] = u.hq0 + hi;
                    row_lower[r] = lower;
                    row_upper[r] = upper;
                    r += 1;
                }
            }

            // m = max_s m_s ; l = sum_s l_s * e^(m_s - m) ; acc = sum_s acc_s * e^(m_s - m)
            @memset(m_out[0..rows], -std.math.inf(f32));
            var seg: usize = 0;
            while (seg < self.segs) : (seg += 1) {
                const ml = self.partials[(ui * self.segs + seg) * self.part_stride + self.row_block * d_v ..];
                var i: usize = 0;
                while (i < rows) : (i += 1) m_out[i] = @max(m_out[i], ml[i]);
            }
            @memset(l_out[0..rows], 0.0);

            const dst_slot = self.partials[(ui * self.segs) * self.part_stride ..][0 .. self.row_block * d_v];
            seg = 0;
            while (seg < self.segs) : (seg += 1) {
                const slot = self.partials[(ui * self.segs + seg) * self.part_stride ..][0..self.part_stride];
                const acc_s = slot[0 .. self.row_block * d_v];
                const ml = slot[self.row_block * d_v ..];
                var i: usize = 0;
                while (i < rows) : (i += 1) {
                    if (m_out[i] == -std.math.inf(f32)) continue;
                    const f: f32 = if (ml[i] == -std.math.inf(f32)) 0.0 else self.kernels.exp_softmax(ml[i] - m_out[i]);
                    l_out[i] += ml[self.row_block + i] * f;
                    const src = acc_s[i * d_v ..][0..d_v];
                    const dst = dst_slot[i * d_v ..][0..d_v];
                    if (seg == 0) {
                        var j: usize = 0;
                        while (j < d_v) : (j += 1) dst[j] = src[j] * f;
                    } else {
                        var j: usize = 0;
                        while (j < d_v) : (j += 1) dst[j] += src[j] * f;
                    }
                }
            }

            try self.finalize(u, &w, rows, &row_l, &row_h, dst_slot, l_out[0..], &row_lower, &row_upper);
        }
    }
};

/// Blocked path preconditions. f16 caches and ring time mapping keep the
/// generic executor (this path's panels assume physically contiguous key rows).
fn blockedEligible(
    q_dtype: DType,
    k_dtype: DType,
    v_dtype: DType,
    map_mode: TimeMapMode,
    d_k: usize,
    d_v: usize,
) bool {
    if (q_dtype != .f32 or k_dtype != .f32 or v_dtype != .f32) return false;
    if (map_mode != .identity) return false;
    if (d_k == 0 or d_v == 0) return false;
    return true;
}

const ExecCtx = struct {
    const max_fast_keys: usize = 2048;

    store: tensor_store.TensorStore,
    kernels: attention_registry.Kernels,
    s: executable.StepAttentionTiled,

    q_meta: tensor_store.TensorMeta,
    k_meta: tensor_store.TensorMeta,
    v_meta: tensor_store.TensorMeta,
    q_dtype: DType,
    k_dtype: DType,
    v_dtype: DType,
    // Null together when k/v are a plain sequence rather than a cache.
    positions_meta: ?tensor_store.TensorMeta,
    lengths_meta: ?tensor_store.TensorMeta,
    out_meta: tensor_store.TensorMeta,

    batch: usize,
    l_q: usize,
    h_q: usize,
    d_k: usize,
    d_v: usize,
    groups_per_kv: usize,

    map_mode: TimeMapMode,
    ring_window: usize,
    k_cap: usize,
    v_cap: usize,

    hq_tiles: usize,
    tiles_per_b: usize,

    stop: std.atomic.Value(bool) = .init(false),
    err_mutex: std.Io.Mutex = .init,
    err_any: ?anyerror = null,

    fn fail(self: *@This(), err: anyerror) void {
        if (self.stop.swap(true, .acq_rel)) return;
        std.Io.Threaded.mutexLock(&self.err_mutex);
        defer std.Io.Threaded.mutexUnlock(&self.err_mutex);
        if (self.err_any == null) self.err_any = err;
    }

    fn runRange(self: *@This(), start: usize, end: usize) void {
        if (start >= end) return;

        var q_cache: ConstTileCache = .{};
        defer releaseConstCache(self.store, &q_cache);
        var k_cache_tiles: ConstTileCache = .{};
        defer releaseConstCache(self.store, &k_cache_tiles);
        var v_cache_tiles: ConstTileCache = .{};
        defer releaseConstCache(self.store, &v_cache_tiles);
        var pos_cache: ConstTileCache = .{};
        defer releaseConstCache(self.store, &pos_cache);
        var end_cache: ConstTileCache = .{};
        defer releaseConstCache(self.store, &end_cache);
        var out_cache: MutTileCache = .{};
        defer releaseMutCache(self.store, &out_cache);

        var work_idx: usize = start;
        while (work_idx < end) : (work_idx += 1) {
            if (self.stop.load(.acquire)) return;

            const b_tile: usize = work_idx / self.tiles_per_b;
            const rem: usize = work_idx - b_tile * self.tiles_per_b;
            const l_tile: usize = rem / self.hq_tiles;
            const hq_tile: usize = rem - l_tile * self.hq_tiles;

            const b_start: usize = b_tile * self.out_meta.tile_shape[0];
            const b_end: usize = @min(self.batch, b_start + self.out_meta.tile_shape[0]);
            const l_start: usize = l_tile * self.out_meta.tile_shape[1];
            const l_end: usize = @min(self.l_q, l_start + self.out_meta.tile_shape[1]);
            const hq_start: usize = hq_tile * self.out_meta.tile_shape[2];
            const hq_end: usize = @min(self.h_q, hq_start + self.out_meta.tile_shape[2]);

            var b: usize = b_start;
            while (b < b_end) : (b += 1) {
                const valid_end: usize = if (self.lengths_meta) |meta| blk: {
                    const end_i32: i32 = readI32Rank1(self.store, self.s.kv_lengths.?, meta, b, &end_cache) catch |e| {
                        self.fail(e);
                        return;
                    };
                    if (end_i32 < 0) {
                        self.fail(BackendError.InvalidArgument);
                        return;
                    }
                    break :blk @intCast(end_i32);
                } else self.k_cap;

                if (self.map_mode == .identity and valid_end > self.k_cap) {
                    self.fail(BackendError.InvalidArgument);
                    return;
                }

                var available_start: usize = 0;
                if (self.map_mode == .ring and valid_end > self.ring_window) {
                    available_start = valid_end - self.ring_window;
                }

                var l: usize = l_start;
                while (l < l_end) : (l += 1) {
                    // Without explicit positions a query's position is its row.
                    // This is the one per-row store read the sequence path skips.
                    const q_pos: usize = if (self.positions_meta) |meta| blk: {
                        const q_pos_i32: i32 = readI32Rank2(self.store, self.s.query_positions.?, meta, b, l, &pos_cache) catch |e| {
                            self.fail(e);
                            return;
                        };
                        if (q_pos_i32 < 0) {
                            self.fail(BackendError.InvalidArgument);
                            return;
                        }
                        break :blk @intCast(q_pos_i32);
                    } else l;

                    const win = self.s.window.keys(q_pos, valid_end);
                    const upper: usize = win.hi;
                    const lower: usize = @max(win.lo, available_start);

                    var hq: usize = hq_start;
                    while (hq < hq_end) : (hq += 1) {
                        var out_row: []align(1) f32 = rowSliceRank4MutF32(self.store, self.s.out, self.out_meta, b, l, hq, self.d_v, &out_cache) catch |e| {
                            self.fail(e);
                            return;
                        };
                        @memset(out_row[0..self.d_v], 0.0);

                        if (lower >= upper) continue;

                        const q_row_f32: ?[]align(1) const f32 = if (self.q_dtype == .f32)
                            rowSliceRank4ConstF32(self.store, self.s.q, self.q_meta, b, l, hq, self.d_k, &q_cache) catch |e| {
                                self.fail(e);
                                return;
                            }
                        else
                            null;
                        const q_row_f16: ?[]align(1) const f16 = if (self.q_dtype == .f16)
                            rowSliceRank4ConstF16(self.store, self.s.q, self.q_meta, b, l, hq, self.d_k, &q_cache) catch |e| {
                                self.fail(e);
                                return;
                            }
                        else
                            null;

                        if (q_row_f32 == null and q_row_f16 == null) {
                            self.fail(BackendError.InvalidArgument);
                            return;
                        }

                        const hkv: usize = hq / self.groups_per_kv;

                        const n_keys: usize = upper - lower;
                        if (n_keys <= max_fast_keys and n_keys <= std.math.maxInt(u32)) {
                            var logits_buf: [max_fast_keys]f32 = undefined;
                            var v_idx_buf: [max_fast_keys]u32 = undefined;
                            var m_state: f32 = -std.math.inf(f32);

                            var key_i: usize = 0;
                            while (key_i < n_keys) : (key_i += 1) {
                                const logical_t: usize = lower + key_i;

                                var k_t: usize = 0;
                                var v_t: usize = 0;
                                switch (self.map_mode) {
                                    .identity => {
                                        if (logical_t >= self.k_cap or logical_t >= self.v_cap) {
                                            self.fail(BackendError.InvalidArgument);
                                            return;
                                        }
                                        k_t = logical_t;
                                        v_t = logical_t;
                                    },
                                    .ring => {
                                        k_t = logical_t % self.ring_window;
                                        v_t = logical_t % self.ring_window;
                                    },
                                }

                                const k_row_f32: ?[]align(1) const f32 = if (self.k_dtype == .f32)
                                    rowSliceRank4ConstF32(self.store, self.s.k, self.k_meta, b, k_t, hkv, self.d_k, &k_cache_tiles) catch |e| {
                                        self.fail(e);
                                        return;
                                    }
                                else
                                    null;
                                const k_row_f16: ?[]align(1) const f16 = if (self.k_dtype == .f16)
                                    rowSliceRank4ConstF16(self.store, self.s.k, self.k_meta, b, k_t, hkv, self.d_k, &k_cache_tiles) catch |e| {
                                        self.fail(e);
                                        return;
                                    }
                                else
                                    null;

                                if (k_row_f32 == null and k_row_f16 == null) {
                                    self.fail(BackendError.InvalidArgument);
                                    return;
                                }

                                const dot: f32 = dotRowsDynamic(self.q_dtype, q_row_f32, q_row_f16, self.k_dtype, k_row_f32, k_row_f16, self.d_k);
                                var logit: f32 = dot * self.s.scale;
                                if (self.s.attn_logits_soft_cap > 0.0) {
                                    const cap: f32 = self.s.attn_logits_soft_cap;
                                    logit = cap * fast_math.tanhApproxF32(logit / cap);
                                }
                                logits_buf[key_i] = logit;
                                v_idx_buf[key_i] = @intCast(v_t);
                                m_state = @max(m_state, logit);
                            }

                            var l_state: f32 = 0.0;
                            key_i = 0;
                            while (key_i < n_keys) : (key_i += 1) {
                                const p: f32 = self.kernels.exp_softmax(logits_buf[key_i] - m_state);
                                logits_buf[key_i] = p;
                                l_state += p;
                            }

                            if (!(l_state > 0.0) or !std.math.isFinite(l_state)) {
                                self.fail(BackendError.InvalidArgument);
                                return;
                            }
                            const inv: f32 = 1.0 / l_state;

                            key_i = 0;
                            while (key_i < n_keys) : (key_i += 1) {
                                const alpha: f32 = logits_buf[key_i] * inv;
                                const v_t: usize = @intCast(v_idx_buf[key_i]);
                                const v_row_f32: ?[]align(1) const f32 = if (self.v_dtype == .f32)
                                    rowSliceRank4ConstF32(self.store, self.s.v, self.v_meta, b, v_t, hkv, self.d_v, &v_cache_tiles) catch |e| {
                                        self.fail(e);
                                        return;
                                    }
                                else
                                    null;
                                const v_row_f16: ?[]align(1) const f16 = if (self.v_dtype == .f16)
                                    rowSliceRank4ConstF16(self.store, self.s.v, self.v_meta, b, v_t, hkv, self.d_v, &v_cache_tiles) catch |e| {
                                        self.fail(e);
                                        return;
                                    }
                                else
                                    null;

                                if (v_row_f32 == null and v_row_f16 == null) {
                                    self.fail(BackendError.InvalidArgument);
                                    return;
                                }

                                accumulateScaledDynamic(self.v_dtype, out_row, v_row_f32, v_row_f16, self.d_v, alpha);
                            }
                        } else {
                            var m_state: f32 = -std.math.inf(f32);
                            var l_state: f32 = 0.0;

                            var logical_t: usize = lower;
                            while (logical_t < upper) : (logical_t += 1) {
                                var k_t: usize = 0;
                                var v_t: usize = 0;
                                switch (self.map_mode) {
                                    .identity => {
                                        if (logical_t >= self.k_cap or logical_t >= self.v_cap) {
                                            self.fail(BackendError.InvalidArgument);
                                            return;
                                        }
                                        k_t = logical_t;
                                        v_t = logical_t;
                                    },
                                    .ring => {
                                        k_t = logical_t % self.ring_window;
                                        v_t = logical_t % self.ring_window;
                                    },
                                }

                                const k_row_f32: ?[]align(1) const f32 = if (self.k_dtype == .f32)
                                    rowSliceRank4ConstF32(self.store, self.s.k, self.k_meta, b, k_t, hkv, self.d_k, &k_cache_tiles) catch |e| {
                                        self.fail(e);
                                        return;
                                    }
                                else
                                    null;
                                const k_row_f16: ?[]align(1) const f16 = if (self.k_dtype == .f16)
                                    rowSliceRank4ConstF16(self.store, self.s.k, self.k_meta, b, k_t, hkv, self.d_k, &k_cache_tiles) catch |e| {
                                        self.fail(e);
                                        return;
                                    }
                                else
                                    null;

                                if (k_row_f32 == null and k_row_f16 == null) {
                                    self.fail(BackendError.InvalidArgument);
                                    return;
                                }

                                const dot: f32 = dotRowsDynamic(self.q_dtype, q_row_f32, q_row_f16, self.k_dtype, k_row_f32, k_row_f16, self.d_k);
                                var logit: f32 = dot * self.s.scale;
                                if (self.s.attn_logits_soft_cap > 0.0) {
                                    const cap: f32 = self.s.attn_logits_soft_cap;
                                    logit = cap * fast_math.tanhApproxF32(logit / cap);
                                }

                                const m_new: f32 = @max(m_state, logit);
                                const rescale: f32 = if (m_state == -std.math.inf(f32)) 0.0 else self.kernels.exp_softmax(m_state - m_new);
                                const p_new: f32 = self.kernels.exp_softmax(logit - m_new);

                                const v_row_f32: ?[]align(1) const f32 = if (self.v_dtype == .f32)
                                    rowSliceRank4ConstF32(self.store, self.s.v, self.v_meta, b, v_t, hkv, self.d_v, &v_cache_tiles) catch |e| {
                                        self.fail(e);
                                        return;
                                    }
                                else
                                    null;
                                const v_row_f16: ?[]align(1) const f16 = if (self.v_dtype == .f16)
                                    rowSliceRank4ConstF16(self.store, self.s.v, self.v_meta, b, v_t, hkv, self.d_v, &v_cache_tiles) catch |e| {
                                        self.fail(e);
                                        return;
                                    }
                                else
                                    null;

                                if (v_row_f32 == null and v_row_f16 == null) {
                                    self.fail(BackendError.InvalidArgument);
                                    return;
                                }

                                fusedRescaleAccumulateDynamic(self.v_dtype, out_row, v_row_f32, v_row_f16, self.d_v, rescale, p_new);

                                l_state = l_state * rescale + p_new;
                                m_state = m_new;
                            }

                            if (!(l_state > 0.0) or !std.math.isFinite(l_state)) {
                                self.fail(BackendError.InvalidArgument);
                                return;
                            }
                            const inv: f32 = 1.0 / l_state;
                            scaleRowF32(out_row, self.d_v, inv);
                        }
                    }
                }
            }
        }
    }

    fn runWork(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
        _ = tid;
        const self: *@This() = @ptrCast(@alignCast(ctx_any));
        self.runRange(start, end);
    }
};

/// Shapes and metadata shared by both executors, resolved once by the caller.
const Shapes = struct {
    q_meta: tensor_store.TensorMeta,
    k_meta: tensor_store.TensorMeta,
    v_meta: tensor_store.TensorMeta,
    positions_meta: ?tensor_store.TensorMeta,
    lengths_meta: ?tensor_store.TensorMeta,
    out_meta: tensor_store.TensorMeta,
    batch: usize,
    l_q: usize,
    h_q: usize,
    h_kv: usize,
    d_k: usize,
    d_v: usize,
    groups_per_kv: usize,
    k_cap: usize,
};

/// Returns `false` (having done nothing) when the scratch allocation fails, so the
/// caller can fall back to the allocation-free generic executor.
fn execBlocked(
    allocator: std.mem.Allocator,
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    kernels: attention_registry.Kernels,
    s: executable.StepAttentionTiled,
    store: tensor_store.TensorStore,
    sh: Shapes,
) ExecuteProgramError!bool {
    const row_block: usize = kernels.tuning.row_block;
    const key_block: usize = kernels.tuning.key_block;

    // Query rows per unit: fill up to `row_block` with whole GQA groups when they fit,
    // so one unit's rows always share a K/V panel.
    const hb: usize = @min(sh.groups_per_kv, row_block);
    const lb: usize = @max(@as(usize, 1), row_block / hb);
    const n_l_chunks: usize = ceilDiv(sh.l_q, lb);
    const n_h_chunks: usize = ceilDiv(sh.groups_per_kv, hb);
    const base_units: usize = sh.batch * sh.h_kv * n_l_chunks * n_h_chunks;
    if (base_units == 0) return true;

    // Split a unit's key range only when there are too few units to fill the pool
    // — decode's single query row is the case that needs it.
    var segs: usize = 1;
    if (thread_count > 1 and base_units < thread_count) {
        const span_hint: usize = s.window.maxKeys(sh.k_cap);
        const by_keys: usize = @max(@as(usize, 1), span_hint / kernels.tuning.min_segment_keys);
        segs = @min(ceilDiv(thread_count, base_units), by_keys);
        if (segs == 0) segs = 1;
    }

    const workers: usize = if (pool != null and thread_count > 1) thread_count else 1;
    const per_worker: usize = row_block * sh.d_k + row_block * key_block +
        alignUp(key_block, kernels.tuning.nr) * sh.d_k +
        alignUp(row_block, kernels.tuning.mr) * sh.d_k +
        row_block * sh.d_v;

    const scratch: []f32 = allocator.alloc(f32, per_worker * workers) catch return false;
    defer allocator.free(scratch);

    const part_stride: usize = row_block * sh.d_v + 2 * row_block;
    const partials: []f32 = if (segs > 1)
        allocator.alloc(f32, part_stride * base_units * segs) catch return false
    else
        &.{};
    defer if (segs > 1) allocator.free(partials);

    var ctx: BlockedCtx = .{
        .store = store,
        .kernels = kernels,
        .s = s,
        .q_meta = sh.q_meta,
        .k_meta = sh.k_meta,
        .v_meta = sh.v_meta,
        .positions_meta = sh.positions_meta,
        .lengths_meta = sh.lengths_meta,
        .out_meta = sh.out_meta,
        .batch = sh.batch,
        .l_q = sh.l_q,
        .h_q = sh.h_q,
        .h_kv = sh.h_kv,
        .d_k = sh.d_k,
        .d_v = sh.d_v,
        .groups_per_kv = sh.groups_per_kv,
        .k_cap = sh.k_cap,
        .row_block = row_block,
        .key_block = key_block,
        .lb = lb,
        .hb = hb,
        .n_l_chunks = n_l_chunks,
        .n_h_chunks = n_h_chunks,
        .segs = segs,
        .scratch = scratch,
        .per_worker = per_worker,
        .partials = partials,
        .part_stride = part_stride,
    };

    const total: usize = base_units * segs;
    if (pool) |p| {
        if (thread_count > 1 and total >= 2) {
            p.parallelForAny(@ptrCast(&ctx), total, 1, BlockedCtx.runWork);
            if (ctx.err_any) |e| return @errorCast(e);
            if (segs > 1) try ctx.mergeUnits();
            return true;
        }
    }

    ctx.runRange(0, total, 0);
    if (ctx.err_any) |e| return @errorCast(e);
    if (segs > 1) try ctx.mergeUnits();
    return true;
}

pub fn execAttentionTiled(
    allocator: std.mem.Allocator,
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    kernels: attention_registry.Kernels,
    s: executable.StepAttentionTiled,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    if (!(s.scale > 0.0) or !std.math.isFinite(s.scale)) return BackendError.InvalidArgument;
    if (!std.math.isFinite(s.attn_logits_soft_cap) or s.attn_logits_soft_cap < 0.0) return BackendError.InvalidArgument;
    const q_meta: tensor_store.TensorMeta = try store.meta(s.q);
    var k_meta: tensor_store.TensorMeta = try store.meta(s.k);
    var v_meta: tensor_store.TensorMeta = try store.meta(s.v);
    const positions_meta: ?tensor_store.TensorMeta = if (s.query_positions) |t| try store.meta(t) else null;
    const lengths_meta: ?tensor_store.TensorMeta = if (s.kv_lengths) |t| try store.meta(t) else null;
    const out_meta: tensor_store.TensorMeta = try store.meta(s.out);

    if (q_meta.rank != 4 or k_meta.rank != 4 or v_meta.rank != 4 or out_meta.rank != 4) return BackendError.InvalidArgument;

    if (!(q_meta.dtype == .f16 or q_meta.dtype == .f32)) return BackendError.InvalidArgument;
    if (!(k_meta.dtype == .f16 or k_meta.dtype == .f32)) return BackendError.InvalidArgument;
    if (!(v_meta.dtype == .f16 or v_meta.dtype == .f32)) return BackendError.InvalidArgument;
    if (out_meta.dtype != .f32) return BackendError.InvalidArgument;
    if (positions_meta) |pm| {
        if (pm.rank != 2 or pm.dtype != .i32) return BackendError.InvalidArgument;
    }
    if (lengths_meta) |lm| if (lm.rank != 1 or lm.dtype != .i32) return BackendError.InvalidArgument;

    const q_dtype: DType = q_meta.dtype;
    const k_dtype: DType = k_meta.dtype;
    const v_dtype: DType = v_meta.dtype;

    const batch: usize = q_meta.shape[0];
    const l_q: usize = q_meta.shape[1];
    const h_q: usize = q_meta.shape[2];
    const d_k: usize = q_meta.shape[3];

    if (k_meta.shape[0] != batch or v_meta.shape[0] != batch) return BackendError.InvalidArgument;
    if (positions_meta) |pm| {
        if (pm.shape[0] != batch or pm.shape[1] != l_q) return BackendError.InvalidArgument;
    }
    if (lengths_meta) |lm| if (lm.shape[0] != batch) return BackendError.InvalidArgument;

    if (out_meta.shape[0] != batch or out_meta.shape[1] != l_q or out_meta.shape[2] != h_q) {
        return BackendError.InvalidArgument;
    }

    const h_kv: usize = k_meta.shape[2];
    if (h_kv == 0 or h_q == 0) return BackendError.InvalidArgument;
    if (v_meta.shape[2] != h_kv) return BackendError.InvalidArgument;
    if ((h_q % h_kv) != 0) return BackendError.InvalidArgument;

    if (k_meta.shape[3] != d_k) return BackendError.InvalidArgument;
    if (k_meta.shape[1] != v_meta.shape[1]) return BackendError.InvalidArgument;

    const d_v: usize = v_meta.shape[3];
    if (out_meta.shape[3] != d_v) return BackendError.InvalidArgument;

    if (q_meta.tile_counts[3] != 1 or k_meta.tile_counts[3] != 1 or v_meta.tile_counts[3] != 1 or out_meta.tile_counts[3] != 1) {
        return BackendError.InvalidArgument;
    }

    const groups_per_kv: usize = h_q / h_kv;

    // Growable mapping can resize physical storage. Pre-touch per-batch tail once
    // so worker threads run against stable metadata. Only a cache read can grow
    // (a plain sequence is exactly as long as it is), so this is index-gated.
    if (lengths_meta) |em| {
        var end_cache_pretouch: ConstTileCache = .{};
        defer releaseConstCache(store, &end_cache_pretouch);

        var b0: usize = 0;
        while (b0 < batch) : (b0 += 1) {
            const end_i32: i32 = try readI32Rank1(store, s.kv_lengths.?, em, b0, &end_cache_pretouch);
            if (end_i32 < 0) return BackendError.InvalidArgument;
            const valid_end: usize = @intCast(end_i32);
            if (valid_end == 0) continue;

            _ = store.mapSequenceStep(s.k, valid_end - 1, k_meta.shape[1]) catch return BackendError.InvalidArgument;
            _ = store.mapSequenceStep(s.v, valid_end - 1, v_meta.shape[1]) catch return BackendError.InvalidArgument;
        }
    }

    k_meta = try store.meta(s.k);
    v_meta = try store.meta(s.v);

    if (k_meta.rank != 4 or v_meta.rank != 4) return BackendError.InvalidArgument;
    if (k_meta.dtype != k_dtype or v_meta.dtype != v_dtype) return BackendError.InvalidArgument;
    if (k_meta.shape[0] != batch or v_meta.shape[0] != batch) return BackendError.InvalidArgument;
    if (k_meta.shape[2] != h_kv or v_meta.shape[2] != h_kv) return BackendError.InvalidArgument;
    if (k_meta.shape[3] != d_k or v_meta.shape[3] != d_v) return BackendError.InvalidArgument;
    if (k_meta.shape[1] != v_meta.shape[1]) return BackendError.InvalidArgument;
    if (k_meta.tile_counts[3] != 1 or v_meta.tile_counts[3] != 1) return BackendError.InvalidArgument;

    const k_cap: usize = k_meta.shape[1];
    const v_cap: usize = v_meta.shape[1];
    if (k_cap == 0 or v_cap == 0) return BackendError.InvalidArgument;

    const policy_info: tensor_store.SequenceCachePolicyInfo = store.sequenceCachePolicyInfo(s.k);
    const map_mode: TimeMapMode = switch (policy_info.kind) {
        .none => .identity,
        .growable => .identity,
        .ring => .ring,
    };
    const ring_window: usize = if (map_mode == .ring) blk: {
        const configured: usize = policy_info.ring_window_tokens;
        break :blk if (configured == 0) k_cap else @min(configured, k_cap);
    } else 0;
    if (map_mode == .ring and ring_window == 0) return BackendError.InvalidArgument;

    const b_tiles: usize = out_meta.tile_counts[0];
    const l_tiles: usize = out_meta.tile_counts[1];
    const hq_tiles: usize = out_meta.tile_counts[2];
    const tiles_per_b: usize = std.math.mul(usize, l_tiles, hq_tiles) catch return BackendError.InvalidArgument;
    const total_work: usize = std.math.mul(usize, b_tiles, tiles_per_b) catch return BackendError.InvalidArgument;
    if (total_work == 0) return;

    if (blockedEligible(q_dtype, k_dtype, v_dtype, map_mode, d_k, d_v)) {
        // `false` means only that the scratch didn't fit — fall through to the
        // generic path rather than failing the step.
        if (try execBlocked(allocator, pool, thread_count, kernels, s, store, .{
            .q_meta = q_meta,
            .k_meta = k_meta,
            .v_meta = v_meta,
            .positions_meta = positions_meta,
            .lengths_meta = lengths_meta,
            .out_meta = out_meta,
            .batch = batch,
            .l_q = l_q,
            .h_q = h_q,
            .h_kv = h_kv,
            .d_k = d_k,
            .d_v = d_v,
            .groups_per_kv = groups_per_kv,
            .k_cap = k_cap,
        })) return;
    }

    var ctx: ExecCtx = .{
        .store = store,
        .kernels = kernels,
        .s = s,
        .q_meta = q_meta,
        .k_meta = k_meta,
        .v_meta = v_meta,
        .q_dtype = q_dtype,
        .k_dtype = k_dtype,
        .v_dtype = v_dtype,
        .positions_meta = positions_meta,
        .lengths_meta = lengths_meta,
        .out_meta = out_meta,
        .batch = batch,
        .l_q = l_q,
        .h_q = h_q,
        .d_k = d_k,
        .d_v = d_v,
        .groups_per_kv = groups_per_kv,
        .map_mode = map_mode,
        .ring_window = ring_window,
        .k_cap = k_cap,
        .v_cap = v_cap,
        .hq_tiles = hq_tiles,
        .tiles_per_b = tiles_per_b,
    };

    if (pool) |p| {
        if (thread_count > 1 and total_work >= 2) {
            p.parallelForAny(@ptrCast(&ctx), total_work, 1, ExecCtx.runWork);
            if (ctx.err_any) |e| return @errorCast(e);
            return;
        }
    }

    ctx.runRange(0, total_work);
    if (ctx.err_any) |e| return @errorCast(e);
}
