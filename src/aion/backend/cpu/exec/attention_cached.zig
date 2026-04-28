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

const TimeMapMode = enum {
    identity,
    ring,
    callback,
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

const ExecCtx = struct {
    const max_fast_keys: usize = 2048;

    store: tensor_store.TensorStore,
    kernels: attention_registry.Kernels,
    s: executable.StepMultiHeadAttentionCachedTiled,

    q_meta: tensor_store.TensorMeta,
    k_meta: tensor_store.TensorMeta,
    v_meta: tensor_store.TensorMeta,
    q_dtype: DType,
    k_dtype: DType,
    v_dtype: DType,
    positions_meta: tensor_store.TensorMeta,
    end_idx_meta: tensor_store.TensorMeta,
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
                const end_i32: i32 = readI32Rank1(self.store, self.s.end_index, self.end_idx_meta, b, &end_cache) catch |e| {
                    self.fail(e);
                    return;
                };
                if (end_i32 < 0) {
                    self.fail(BackendError.InvalidArgument);
                    return;
                }
                const valid_end: usize = @intCast(end_i32);

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
                    const q_pos_i32: i32 = readI32Rank2(self.store, self.s.positions, self.positions_meta, b, l, &pos_cache) catch |e| {
                        self.fail(e);
                        return;
                    };
                    if (q_pos_i32 < 0) {
                        self.fail(BackendError.InvalidArgument);
                        return;
                    }
                    const q_pos: usize = @intCast(q_pos_i32);

                    var upper: usize = valid_end;
                    if (self.s.causal) {
                        const q_pos_next: usize = std.math.add(usize, q_pos, 1) catch {
                            self.fail(BackendError.InvalidArgument);
                            return;
                        };
                        if (q_pos_next < upper) upper = q_pos_next;
                    }

                    var lower: usize = available_start;
                    if (self.s.sliding_window > 0) {
                        const q_pos_next: usize = std.math.add(usize, q_pos, 1) catch {
                            self.fail(BackendError.InvalidArgument);
                            return;
                        };
                        const sw_lower: usize = if (q_pos_next > self.s.sliding_window) q_pos_next - self.s.sliding_window else 0;
                        if (sw_lower > lower) lower = sw_lower;
                    }

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
                                    .callback => {
                                        k_t = self.store.mapKVCacheTime(self.s.k_cache, logical_t, self.k_cap) catch |e| {
                                            self.fail(e);
                                            return;
                                        };
                                        v_t = self.store.mapKVCacheTime(self.s.v_cache, logical_t, self.v_cap) catch |e| {
                                            self.fail(e);
                                            return;
                                        };
                                    },
                                }

                                const k_row_f32: ?[]align(1) const f32 = if (self.k_dtype == .f32)
                                    rowSliceRank4ConstF32(self.store, self.s.k_cache, self.k_meta, b, hkv, k_t, self.d_k, &k_cache_tiles) catch |e| {
                                        self.fail(e);
                                        return;
                                    }
                                else
                                    null;
                                const k_row_f16: ?[]align(1) const f16 = if (self.k_dtype == .f16)
                                    rowSliceRank4ConstF16(self.store, self.s.k_cache, self.k_meta, b, hkv, k_t, self.d_k, &k_cache_tiles) catch |e| {
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
                                    rowSliceRank4ConstF32(self.store, self.s.v_cache, self.v_meta, b, hkv, v_t, self.d_v, &v_cache_tiles) catch |e| {
                                        self.fail(e);
                                        return;
                                    }
                                else
                                    null;
                                const v_row_f16: ?[]align(1) const f16 = if (self.v_dtype == .f16)
                                    rowSliceRank4ConstF16(self.store, self.s.v_cache, self.v_meta, b, hkv, v_t, self.d_v, &v_cache_tiles) catch |e| {
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
                                    .callback => {
                                        k_t = self.store.mapKVCacheTime(self.s.k_cache, logical_t, self.k_cap) catch |e| {
                                            self.fail(e);
                                            return;
                                        };
                                        v_t = self.store.mapKVCacheTime(self.s.v_cache, logical_t, self.v_cap) catch |e| {
                                            self.fail(e);
                                            return;
                                        };
                                    },
                                }

                                const k_row_f32: ?[]align(1) const f32 = if (self.k_dtype == .f32)
                                    rowSliceRank4ConstF32(self.store, self.s.k_cache, self.k_meta, b, hkv, k_t, self.d_k, &k_cache_tiles) catch |e| {
                                        self.fail(e);
                                        return;
                                    }
                                else
                                    null;
                                const k_row_f16: ?[]align(1) const f16 = if (self.k_dtype == .f16)
                                    rowSliceRank4ConstF16(self.store, self.s.k_cache, self.k_meta, b, hkv, k_t, self.d_k, &k_cache_tiles) catch |e| {
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
                                    rowSliceRank4ConstF32(self.store, self.s.v_cache, self.v_meta, b, hkv, v_t, self.d_v, &v_cache_tiles) catch |e| {
                                        self.fail(e);
                                        return;
                                    }
                                else
                                    null;
                                const v_row_f16: ?[]align(1) const f16 = if (self.v_dtype == .f16)
                                    rowSliceRank4ConstF16(self.store, self.s.v_cache, self.v_meta, b, hkv, v_t, self.d_v, &v_cache_tiles) catch |e| {
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

pub fn execMultiHeadAttentionCachedTiled(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    kernels: attention_registry.Kernels,
    s: executable.StepMultiHeadAttentionCachedTiled,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    if (!(s.scale > 0.0) or !std.math.isFinite(s.scale)) return BackendError.InvalidArgument;
    if (!std.math.isFinite(s.attn_logits_soft_cap) or s.attn_logits_soft_cap < 0.0) return BackendError.InvalidArgument;

    const q_meta: tensor_store.TensorMeta = try store.meta(s.q);
    var k_meta: tensor_store.TensorMeta = try store.meta(s.k_cache);
    var v_meta: tensor_store.TensorMeta = try store.meta(s.v_cache);
    const positions_meta: tensor_store.TensorMeta = try store.meta(s.positions);
    const end_idx_meta: tensor_store.TensorMeta = try store.meta(s.end_index);
    const out_meta: tensor_store.TensorMeta = try store.meta(s.out);

    if (q_meta.rank != 4 or k_meta.rank != 4 or v_meta.rank != 4 or out_meta.rank != 4) return BackendError.InvalidArgument;
    if (positions_meta.rank != 2 or end_idx_meta.rank != 1) return BackendError.InvalidArgument;

    if (!(q_meta.dtype == .f16 or q_meta.dtype == .f32)) return BackendError.InvalidArgument;
    if (!(k_meta.dtype == .f16 or k_meta.dtype == .f32)) return BackendError.InvalidArgument;
    if (!(v_meta.dtype == .f16 or v_meta.dtype == .f32)) return BackendError.InvalidArgument;
    if (out_meta.dtype != .f32) return BackendError.InvalidArgument;
    if (positions_meta.dtype != .i32 or end_idx_meta.dtype != .i32) return BackendError.InvalidArgument;

    const q_dtype: DType = q_meta.dtype;
    const k_dtype: DType = k_meta.dtype;
    const v_dtype: DType = v_meta.dtype;

    const batch: usize = q_meta.shape[0];
    const l_q: usize = q_meta.shape[1];
    const h_q: usize = q_meta.shape[2];
    const d_k: usize = q_meta.shape[3];

    if (k_meta.shape[0] != batch or v_meta.shape[0] != batch) return BackendError.InvalidArgument;
    if (positions_meta.shape[0] != batch or positions_meta.shape[1] != l_q) return BackendError.InvalidArgument;
    if (end_idx_meta.shape[0] != batch) return BackendError.InvalidArgument;

    if (out_meta.shape[0] != batch or out_meta.shape[1] != l_q or out_meta.shape[2] != h_q) {
        return BackendError.InvalidArgument;
    }

    const h_kv: usize = k_meta.shape[1];
    if (h_kv == 0 or h_q == 0) return BackendError.InvalidArgument;
    if (v_meta.shape[1] != h_kv) return BackendError.InvalidArgument;
    if ((h_q % h_kv) != 0) return BackendError.InvalidArgument;

    if (k_meta.shape[3] != d_k) return BackendError.InvalidArgument;
    if (k_meta.shape[2] != v_meta.shape[2]) return BackendError.InvalidArgument;

    const d_v: usize = v_meta.shape[3];
    if (out_meta.shape[3] != d_v) return BackendError.InvalidArgument;

    if (q_meta.tile_counts[3] != 1 or k_meta.tile_counts[3] != 1 or v_meta.tile_counts[3] != 1 or out_meta.tile_counts[3] != 1) {
        return BackendError.InvalidArgument;
    }

    const groups_per_kv: usize = h_q / h_kv;

    // Growable mapping can resize physical storage. Pre-touch per-batch tail once
    // so worker threads run against stable metadata.
    {
        var end_cache_pretouch: ConstTileCache = .{};
        defer releaseConstCache(store, &end_cache_pretouch);

        var b0: usize = 0;
        while (b0 < batch) : (b0 += 1) {
            const end_i32: i32 = try readI32Rank1(store, s.end_index, end_idx_meta, b0, &end_cache_pretouch);
            if (end_i32 < 0) return BackendError.InvalidArgument;
            const valid_end: usize = @intCast(end_i32);
            if (valid_end == 0) continue;

            _ = store.mapKVCacheTime(s.k_cache, valid_end - 1, k_meta.shape[2]) catch return BackendError.InvalidArgument;
            _ = store.mapKVCacheTime(s.v_cache, valid_end - 1, v_meta.shape[2]) catch return BackendError.InvalidArgument;
        }
    }

    k_meta = try store.meta(s.k_cache);
    v_meta = try store.meta(s.v_cache);

    if (k_meta.rank != 4 or v_meta.rank != 4) return BackendError.InvalidArgument;
    if (k_meta.dtype != k_dtype or v_meta.dtype != v_dtype) return BackendError.InvalidArgument;
    if (k_meta.shape[0] != batch or v_meta.shape[0] != batch) return BackendError.InvalidArgument;
    if (k_meta.shape[1] != h_kv or v_meta.shape[1] != h_kv) return BackendError.InvalidArgument;
    if (k_meta.shape[3] != d_k or v_meta.shape[3] != d_v) return BackendError.InvalidArgument;
    if (k_meta.shape[2] != v_meta.shape[2]) return BackendError.InvalidArgument;
    if (k_meta.tile_counts[3] != 1 or v_meta.tile_counts[3] != 1) return BackendError.InvalidArgument;

    const k_cap: usize = k_meta.shape[2];
    const v_cap: usize = v_meta.shape[2];
    if (k_cap == 0 or v_cap == 0) return BackendError.InvalidArgument;

    const policy_info: tensor_store.KVCachePolicyInfo = store.kvCachePolicyInfo(s.k_cache);
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
        .end_idx_meta = end_idx_meta,
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
