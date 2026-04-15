const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

pub fn execKVCacheAppendTiled(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepKVCacheAppendTiled,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    _ = pool;
    _ = thread_count;

    var cache_meta: tensor_store.TensorMeta = try store.meta(s.cache);
    const new_kv_meta: tensor_store.TensorMeta = try store.meta(s.new_kv);
    const end_idx_meta: tensor_store.TensorMeta = try store.meta(s.end_index);

    if (cache_meta.rank != 4 or new_kv_meta.rank != 4 or end_idx_meta.rank != 1) {
        return BackendError.InvalidArgument;
    }

    if (cache_meta.dtype != new_kv_meta.dtype) return BackendError.InvalidArgument;
    if (cache_meta.dtype != .f16 and cache_meta.dtype != .f32) return BackendError.InvalidArgument;
    if (end_idx_meta.dtype != .i32) return BackendError.InvalidArgument;

    if (cache_meta.shape[0] != new_kv_meta.shape[0]) return BackendError.InvalidArgument;
    if (cache_meta.shape[1] != new_kv_meta.shape[1]) return BackendError.InvalidArgument;
    if (cache_meta.shape[3] != new_kv_meta.shape[3]) return BackendError.InvalidArgument;
    if (end_idx_meta.shape[0] != cache_meta.shape[0]) return BackendError.InvalidArgument;

    const elem_bytes: usize = switch (cache_meta.dtype) {
        .f16 => 2,
        .f32 => 4,
        else => return BackendError.InvalidArgument,
    };

    const batch: usize = cache_meta.shape[0];
    const heads: usize = cache_meta.shape[1];
    const cache_t: usize = cache_meta.shape[2];
    const head_dim: usize = cache_meta.shape[3];
    const new_len: usize = new_kv_meta.shape[2];

    const row_bytes: usize = std.math.mul(usize, head_dim, elem_bytes) catch return BackendError.InvalidArgument;

    var idx_tile: tensor_store.TileRefConst = try store.acquireTileConstLinear(s.end_index, 0);
    defer store.releaseConst(idx_tile.token);
    const idx_view = idx_tile.bufferView();
    if (idx_view.layout.rank != 1) return BackendError.InvalidArgument;
    if (idx_view.layout.shape[0] < batch) return BackendError.InvalidArgument;
    if (idx_view.bytes.len < batch * @sizeOf(i32)) return BackendError.InvalidArgument;

    const end_idx_ptr: [*]align(1) const i32 = @ptrCast(idx_view.bytes.ptr);
    const end_indices: []align(1) const i32 = end_idx_ptr[0..batch];

    var b_check: usize = 0;
    while (b_check < batch) : (b_check += 1) {
        const idx_i32: i32 = end_indices[b_check];
        if (idx_i32 < 0) return BackendError.InvalidArgument;
        const idx_u: usize = @intCast(idx_i32);

        var t_check: usize = 0;
        while (t_check < new_len) : (t_check += 1) {
            const dst_logical_t: usize = std.math.add(usize, idx_u, t_check) catch return BackendError.InvalidArgument;
            _ = store.mapKVCacheTime(s.cache, dst_logical_t, cache_t) catch return BackendError.InvalidArgument;
        }
    }

    // Mapping may trigger growable physical expansion; refresh metadata before writes.
    cache_meta = try store.meta(s.cache);
    if (cache_meta.rank != 4) return BackendError.InvalidArgument;
    if (cache_meta.shape[0] != batch or cache_meta.shape[1] != heads or cache_meta.shape[3] != head_dim) {
        return BackendError.InvalidArgument;
    }
    const cache_t_after_growth: usize = cache_meta.shape[2];

    if (new_len == 0) return;

    const CacheConst = struct {
        valid: bool = false,
        tile_index: usize = 0,
        tile: tensor_store.TileRefConst = undefined,
    };

    const CacheMut = struct {
        valid: bool = false,
        tile_index: usize = 0,
        tile: tensor_store.TileRefMut = undefined,
    };

    var src_cache: CacheConst = .{};
    defer if (src_cache.valid) store.releaseConst(src_cache.tile.token);
    var dst_cache: CacheMut = .{};
    defer if (dst_cache.valid) store.releaseMut(dst_cache.tile.token);

    var src_tile_coords: [4]usize = .{ 0, 0, 0, 0 };
    var dst_tile_coords: [4]usize = .{ 0, 0, 0, 0 };

    var b: usize = 0;
    while (b < batch) : (b += 1) {
        const t_start: usize = @intCast(end_indices[b]);

        var h: usize = 0;
        while (h < heads) : (h += 1) {
            var t: usize = 0;
            while (t < new_len) : (t += 1) {
                const dst_logical_t: usize = std.math.add(usize, t_start, t) catch return BackendError.InvalidArgument;
                const dst_t: usize = store.mapKVCacheTime(s.cache, dst_logical_t, cache_t_after_growth) catch return BackendError.InvalidArgument;

                src_tile_coords[0] = b / new_kv_meta.tile_shape[0];
                src_tile_coords[1] = h / new_kv_meta.tile_shape[1];
                src_tile_coords[2] = t / new_kv_meta.tile_shape[2];
                src_tile_coords[3] = 0;

                dst_tile_coords[0] = b / cache_meta.tile_shape[0];
                dst_tile_coords[1] = h / cache_meta.tile_shape[1];
                dst_tile_coords[2] = dst_t / cache_meta.tile_shape[2];
                dst_tile_coords[3] = 0;

                const src_tile_index: usize = try tensor_store.encodeTileIndex(new_kv_meta, src_tile_coords[0..4]);
                const dst_tile_index: usize = try tensor_store.encodeTileIndex(cache_meta, dst_tile_coords[0..4]);

                if (!src_cache.valid or src_cache.tile_index != src_tile_index) {
                    if (src_cache.valid) store.releaseConst(src_cache.tile.token);
                    src_cache.tile = try store.acquireTileConstLinear(s.new_kv, src_tile_index);
                    src_cache.tile_index = src_tile_index;
                    src_cache.valid = true;
                }

                if (!dst_cache.valid or dst_cache.tile_index != dst_tile_index) {
                    if (dst_cache.valid) store.releaseMut(dst_cache.tile.token);
                    dst_cache.tile = try store.acquireTileMutLinear(s.cache, dst_tile_index);
                    dst_cache.tile_index = dst_tile_index;
                    dst_cache.valid = true;
                }

                const src_view = src_cache.tile.bufferView();
                const dst_view = dst_cache.tile.bufferView();
                if (src_view.layout.rank != 4 or dst_view.layout.rank != 4) return BackendError.InvalidArgument;

                const src_l_b: usize = b - src_tile_coords[0] * new_kv_meta.tile_shape[0];
                const src_l_h: usize = h - src_tile_coords[1] * new_kv_meta.tile_shape[1];
                const src_l_t: usize = t - src_tile_coords[2] * new_kv_meta.tile_shape[2];

                const dst_l_b: usize = b - dst_tile_coords[0] * cache_meta.tile_shape[0];
                const dst_l_h: usize = h - dst_tile_coords[1] * cache_meta.tile_shape[1];
                const dst_l_t: usize = dst_t - dst_tile_coords[2] * cache_meta.tile_shape[2];

                const src_s0: usize = std.math.cast(usize, src_view.layout.strides_bytes[0]) orelse return BackendError.InvalidArgument;
                const src_s1: usize = std.math.cast(usize, src_view.layout.strides_bytes[1]) orelse return BackendError.InvalidArgument;
                const src_s2: usize = std.math.cast(usize, src_view.layout.strides_bytes[2]) orelse return BackendError.InvalidArgument;
                const src_s3: usize = std.math.cast(usize, src_view.layout.strides_bytes[3]) orelse return BackendError.InvalidArgument;

                const dst_s0: usize = std.math.cast(usize, dst_view.layout.strides_bytes[0]) orelse return BackendError.InvalidArgument;
                const dst_s1: usize = std.math.cast(usize, dst_view.layout.strides_bytes[1]) orelse return BackendError.InvalidArgument;
                const dst_s2: usize = std.math.cast(usize, dst_view.layout.strides_bytes[2]) orelse return BackendError.InvalidArgument;
                const dst_s3: usize = std.math.cast(usize, dst_view.layout.strides_bytes[3]) orelse return BackendError.InvalidArgument;

                if (src_s3 != elem_bytes or dst_s3 != elem_bytes) return BackendError.InvalidArgument;

                const src_off_0: usize = std.math.mul(usize, src_l_b, src_s0) catch return BackendError.InvalidArgument;
                const src_off_1: usize = std.math.mul(usize, src_l_h, src_s1) catch return BackendError.InvalidArgument;
                const src_off_2: usize = std.math.mul(usize, src_l_t, src_s2) catch return BackendError.InvalidArgument;
                const src_off: usize = std.math.add(usize, std.math.add(usize, src_off_0, src_off_1) catch return BackendError.InvalidArgument, src_off_2) catch return BackendError.InvalidArgument;

                const dst_off_0: usize = std.math.mul(usize, dst_l_b, dst_s0) catch return BackendError.InvalidArgument;
                const dst_off_1: usize = std.math.mul(usize, dst_l_h, dst_s1) catch return BackendError.InvalidArgument;
                const dst_off_2: usize = std.math.mul(usize, dst_l_t, dst_s2) catch return BackendError.InvalidArgument;
                const dst_off: usize = std.math.add(usize, std.math.add(usize, dst_off_0, dst_off_1) catch return BackendError.InvalidArgument, dst_off_2) catch return BackendError.InvalidArgument;

                const src_end: usize = std.math.add(usize, src_off, row_bytes) catch return BackendError.InvalidArgument;
                const dst_end: usize = std.math.add(usize, dst_off, row_bytes) catch return BackendError.InvalidArgument;
                if (src_end > src_view.bytes.len or dst_end > dst_view.bytes.len) return BackendError.InvalidArgument;

                @memcpy(dst_view.bytes[dst_off..dst_end], src_view.bytes[src_off..src_end]);
            }
        }
    }
}
