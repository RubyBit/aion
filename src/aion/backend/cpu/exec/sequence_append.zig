// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

pub fn execSequenceAppend(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepSequenceAppend,
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
    if (cache_meta.shape[2] != new_kv_meta.shape[2]) return BackendError.InvalidArgument;
    if (cache_meta.shape[3] != new_kv_meta.shape[3]) return BackendError.InvalidArgument;
    if (end_idx_meta.shape[0] != cache_meta.shape[0]) return BackendError.InvalidArgument;

    const elem_bytes: usize = switch (cache_meta.dtype) {
        .f16 => 2,
        .f32 => 4,
        else => return BackendError.InvalidArgument,
    };

    const batch: usize = cache_meta.shape[0];
    const cache_t: usize = cache_meta.shape[1];
    const heads: usize = cache_meta.shape[2];
    const head_dim: usize = cache_meta.shape[3];
    const new_len: usize = new_kv_meta.shape[1];

    const row_bytes: usize = std.math.mul(usize, head_dim, elem_bytes) catch return BackendError.InvalidArgument;

    var idx_v: tensor_store.ViewConst = try store.acquireConst(s.end_index);
    defer store.releaseConst(idx_v.token);
    const idx_view = idx_v.bufferView();
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
            _ = store.mapSequenceStep(s.cache, dst_logical_t, cache_t) catch return BackendError.InvalidArgument;
        }
    }

    // Mapping may trigger growable physical expansion; refresh metadata before writes.
    cache_meta = try store.meta(s.cache);
    if (cache_meta.rank != 4) return BackendError.InvalidArgument;
    if (cache_meta.shape[0] != batch or cache_meta.shape[2] != heads or cache_meta.shape[3] != head_dim) {
        return BackendError.InvalidArgument;
    }
    const cache_t_after_growth: usize = cache_meta.shape[1];

    if (new_len == 0) return;

    // `[B, T, H, D]`: row (b, t, h) starts `((b * T + t) * H + h) * D` elements in.
    const src_t = try store.acquireConst(s.new_kv);
    defer store.releaseConst(src_t.token);
    const dst_t = try store.acquireMut(s.cache);
    defer store.releaseMut(dst_t.token);
    if (src_t.bytes.len < batch * new_len * heads * row_bytes) return BackendError.InvalidArgument;
    if (dst_t.bytes.len < batch * cache_t_after_growth * heads * row_bytes) return BackendError.InvalidArgument;

    for (0..batch) |b| {
        const t_start: usize = @intCast(end_indices[b]);
        for (0..new_len) |t| {
            const dst_logical_t: usize = std.math.add(usize, t_start, t) catch return BackendError.InvalidArgument;
            const dst_row: usize = store.mapSequenceStep(s.cache, dst_logical_t, cache_t_after_growth) catch return BackendError.InvalidArgument;
            // All heads of one time step are adjacent in both tensors: one copy.
            const src_off = (b * new_len + t) * heads * row_bytes;
            const dst_off = (b * cache_t_after_growth + dst_row) * heads * row_bytes;
            @memcpy(dst_t.bytes[dst_off..][0 .. heads * row_bytes], src_t.bytes[src_off..][0 .. heads * row_bytes]);
        }
    }
}
