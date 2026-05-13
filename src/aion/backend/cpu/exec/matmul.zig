// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");

const executable = @import("../../../runtime/executable.zig");
const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");

const simd = @import("../kernels/simd.zig");
const quant_matmul_kernels = @import("../kernels/quant_matmul.zig");
const matmul_registry = @import("../registry/matmul_registry.zig");
const quant_matmul_registry = @import("../registry/quant_matmul_registry.zig");
const matvec_registry = @import("../registry/matvec_registry.zig");

const BackendError = types.BackendError;
const DType = types.DType;
const MatMulParams = types.MatMulParams;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

pub const MatMulExecCtx = struct {
    allocator: std.mem.Allocator,
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,

    matmul_f32: matmul_registry.F32Kernels,
    matmul_qx0: quant_matmul_registry.QuantKernels,
    matvec: matvec_registry.Kernels,

    // Per-thread scratch.
    matmul_scratch: [][]align(32) u8,
};

fn runF16TilePackedB(
    mk: matmul_registry.F32Kernels,
    scratch: []align(32) u8,
    packed_b: []align(32) const f32,
    params: MatMulParams,
    c_dtype: DType,
    c_bytes: []u8,
    a_bytes: []const u8,
) BackendError!void {
    if (params.k > mk.tuning.kc or params.n > mk.tuning.nc) return BackendError.InvalidArgument;
    if (params.m > mk.tuning.mc) return BackendError.InvalidArgument;

    const c_elem_bytes: usize = switch (c_dtype) {
        .f32 => @sizeOf(f32),
        .f16 => @sizeOf(f16),
        else => return BackendError.InvalidArgument,
    };

    const a_row_bytes: usize = params.k * @sizeOf(f16);
    const c_row_bytes: usize = params.n * c_elem_bytes;
    if (a_bytes.len < params.m * a_row_bytes) return BackendError.InvalidArgument;
    if (c_bytes.len < params.m * c_row_bytes) return BackendError.InvalidArgument;

    const pb_f32_len: usize = mk.tuning.kc * mk.tuning.nc;
    const pb_bytes_len: usize = pb_f32_len * @sizeOf(f32);
    const pa_cap_f32_len: usize = mk.tuning.mc * mk.tuning.kc;
    const pa_cap_bytes_len: usize = pa_cap_f32_len * @sizeOf(f32);
    if (scratch.len < pb_bytes_len + pa_cap_bytes_len) return BackendError.InvalidArgument;
    if (packed_b.len < pb_f32_len) return BackendError.InvalidArgument;

    const pa_panel_count: usize = (params.m + mk.tuning.mr - 1) / mk.tuning.mr;
    const pa_need_f32_len: usize = pa_panel_count * mk.tuning.mr * mk.tuning.kc;
    if (pa_need_f32_len > pa_cap_f32_len) return BackendError.InvalidArgument;

    const pa_bytes_len: usize = pa_need_f32_len * @sizeOf(f32);
    const pa_off: usize = pb_bytes_len;
    const packed_a: []align(32) f32 = @alignCast(std.mem.bytesAsSlice(f32, scratch[pa_off .. pa_off + pa_bytes_len]));
    try mk.pack_a_tile_f16_to_packed_f32(packed_a, params.m, params.k, a_bytes);

    switch (c_dtype) {
        .f32 => {
            try mk.matmul_packed_ab(packed_a, packed_b, params, c_bytes);
        },
        .f16 => {
            const lanes: usize = 8;
            const VF16 = @Vector(lanes, f16);
            const VF32 = @Vector(lanes, f32);

            const c_out: []align(1) f16 = simd.bytesAsSliceMutUnaligned(f16, c_bytes);
            if (c_out.len < params.m * params.n) return BackendError.InvalidArgument;

            const tail_bytes: usize = scratch.len - pb_bytes_len;
            const a_row_bytes_f16: usize = params.k * @sizeOf(f16);
            const c_row_bytes_f16: usize = params.n * @sizeOf(f16);

            var row0: usize = 0;
            while (row0 < params.m) {
                const rows_left: usize = params.m - row0;
                var rows_chunk: usize = rows_left;
                while (rows_chunk > 0) {
                    const panels: usize = (rows_chunk + mk.tuning.mr - 1) / mk.tuning.mr;
                    const pa_need_f32: usize = panels * mk.tuning.mr * mk.tuning.kc;
                    const pa_need_bytes: usize = pa_need_f32 * @sizeOf(f32);
                    const c_tmp_bytes: usize = rows_chunk * params.n * @sizeOf(f32);
                    if (pa_need_bytes + c_tmp_bytes <= tail_bytes) break;
                    if (rows_chunk > mk.tuning.mr) {
                        rows_chunk -= mk.tuning.mr;
                    } else {
                        rows_chunk -= 1;
                    }
                }
                if (rows_chunk == 0) return BackendError.InvalidArgument;

                const panels: usize = (rows_chunk + mk.tuning.mr - 1) / mk.tuning.mr;
                const pa_need_f32: usize = panels * mk.tuning.mr * mk.tuning.kc;
                const pa_need_bytes: usize = pa_need_f32 * @sizeOf(f32);
                const c_tmp_off: usize = pb_bytes_len + pa_need_bytes;
                const c_tmp_bytes: usize = rows_chunk * params.n * @sizeOf(f32);

                const packed_a_chunk: []align(32) f32 = @alignCast(std.mem.bytesAsSlice(f32, scratch[pb_bytes_len .. pb_bytes_len + pa_need_bytes]));
                const a_off: usize = row0 * a_row_bytes_f16;
                const a_len: usize = rows_chunk * a_row_bytes_f16;
                const a_chunk_bytes: []const u8 = a_bytes[a_off .. a_off + a_len];
                try mk.pack_a_tile_f16_to_packed_f32(packed_a_chunk, rows_chunk, params.k, a_chunk_bytes);

                const c_tmp: []align(32) f32 = @alignCast(std.mem.bytesAsSlice(f32, scratch[c_tmp_off .. c_tmp_off + c_tmp_bytes]));
                const c_off: usize = row0 * c_row_bytes_f16;
                const c_len: usize = rows_chunk * c_row_bytes_f16;
                const c_chunk_bytes: []u8 = c_bytes[c_off .. c_off + c_len];
                const c_chunk: []align(1) f16 = simd.bytesAsSliceMutUnaligned(f16, c_chunk_bytes);

                const c_elems_chunk: usize = rows_chunk * params.n;
                if (params.beta == 0.0) {
                    @memset(c_tmp, 0.0);
                } else {
                    var i: usize = 0;
                    while (i + lanes <= c_elems_chunk) : (i += lanes) {
                        const src_ptr: [*]align(1) const f16 = @ptrCast(c_chunk.ptr + i);
                        const hv: VF16 = @as(*align(1) const VF16, @ptrCast(src_ptr)).*;
                        const fv: VF32 = @floatCast(hv);
                        const dst_ptr: [*]align(1) f32 = @ptrCast(c_tmp.ptr + i);
                        @as(*align(1) VF32, @ptrCast(dst_ptr)).* = fv;
                    }
                    while (i < c_elems_chunk) : (i += 1) {
                        c_tmp[i] = @as(f32, @floatCast(c_chunk[i]));
                    }
                }

                const p_chunk: MatMulParams = .{
                    .m = rows_chunk,
                    .n = params.n,
                    .k = params.k,
                    .alpha = params.alpha,
                    .beta = params.beta,
                };
                try mk.matmul_packed_ab(packed_a_chunk, packed_b, p_chunk, std.mem.sliceAsBytes(c_tmp));

                var i: usize = 0;
                while (i + lanes <= c_elems_chunk) : (i += lanes) {
                    const src_ptr: [*]align(1) const f32 = @ptrCast(c_tmp.ptr + i);
                    const fv: VF32 = @as(*align(1) const VF32, @ptrCast(src_ptr)).*;
                    const hv: VF16 = @floatCast(fv);
                    const dst_ptr: [*]align(1) f16 = @ptrCast(c_chunk.ptr + i);
                    @as(*align(1) VF16, @ptrCast(dst_ptr)).* = hv;
                }
                while (i < c_elems_chunk) : (i += 1) {
                    c_chunk[i] = @floatCast(c_tmp[i]);
                }

                row0 += rows_chunk;
            }
        },
        else => return BackendError.InvalidArgument,
    }
}

fn matmulF16ViaPackedF32(
    mk: matmul_registry.F32Kernels,
    scratch: []align(32) u8,
    params: MatMulParams,
    c_dtype: DType,
    c_bytes: []u8,
    a_bytes: []const u8,
    b_bytes: []const u8,
) BackendError!void {
    if (params.k > mk.tuning.kc or params.n > mk.tuning.nc) return BackendError.InvalidArgument;

    const pb_f32_len: usize = mk.tuning.kc * mk.tuning.nc;
    const pb_bytes_len: usize = pb_f32_len * @sizeOf(f32);
    if (scratch.len < pb_bytes_len) return BackendError.InvalidArgument;

    const packed_b_mut: []align(32) f32 = @alignCast(std.mem.bytesAsSlice(f32, scratch[0..pb_bytes_len]));
    try mk.pack_b_tile_f16_to_packed_f32(packed_b_mut, params.k, params.n, b_bytes);

    return runF16TilePackedB(mk, scratch, packed_b_mut, params, c_dtype, c_bytes, a_bytes);
}

fn shouldUseQ8DirectMatvec(params: MatMulParams, thread_count: usize) bool {
    if (params.m != 1) return false;
    if (thread_count <= 1) return false;
    const work: usize = std.math.mul(usize, params.k, params.n) catch return false;
    // Direct K-major q8 matvec avoids pack-B cost, but packed kernels can be
    // faster on tiny workloads (e.g. Silero-sized layers). Use direct mode only
    // once the tile is large enough to amortize less cache-friendly B traversal.
    return work >= (64 * 1024);
}

pub fn execMatMulTiled(ctx: *MatMulExecCtx, s: executable.StepMatMulTiled, store: tensor_store.TensorStore) ExecuteProgramError!void {
    const c_meta = try store.meta(s.c);
    const a_meta = try store.meta(s.a);
    const b_meta = try store.meta(s.b);

    if (c_meta.rank > 2) {
        return execMatMulTiledBatched(ctx, s, store, c_meta, a_meta, b_meta);
    }

    const a_dtype: DType = a_meta.dtype;
    const b_dtype: DType = b_meta.dtype;
    const c_dtype: DType = c_meta.dtype;

    const tile_total: usize = c_meta.tile_counts[0] * c_meta.tile_counts[1];
    if (ctx.pool) |p| {
        const total_work: usize = tile_total;

        // MatMul tiles are compute-heavy; parallelize if we have enough work.
        if (ctx.thread_count > 1 and total_work >= 2) {
            const Task = struct {
                store: tensor_store.TensorStore,
                c_meta: tensor_store.TensorMeta,
                a_meta: tensor_store.TensorMeta,

                a_dtype: DType,
                b_dtype: DType,
                c_dtype: DType,

                s: executable.StepMatMulTiled,

                scratch: [][]align(32) u8,
                matmul_f32: matmul_registry.F32Kernels,
                matmul_qx0: quant_matmul_registry.QuantKernels,
                matvec: matvec_registry.Kernels,
                thread_count: usize,

                stop: std.atomic.Value(bool) = .init(false),
                err_mutex: std.Io.Mutex = .init,
                err_any: ?anyerror = null,

                fn fail(t: *@This(), err: anyerror) void {
                    if (t.stop.swap(true, .acq_rel)) return;
                    std.Io.Threaded.mutexLock(&t.err_mutex);
                    defer std.Io.Threaded.mutexUnlock(&t.err_mutex);
                    if (t.err_any == null) t.err_any = err;
                }

                fn runTiles(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
                    const t: *@This() = @ptrCast(@alignCast(ctx_any));
                    if (start >= end) return;
                    if (tid >= t.scratch.len) return;
                    if (t.stop.load(.acquire)) return;

                    const tc0: usize = t.c_meta.tile_counts[0];
                    const is_matvec: bool = (t.c_meta.shape[0] == 1);
                    const k_tiles: usize = t.a_meta.tile_counts[1];

                    var i: usize = start;
                    while (i < end) {
                        if (t.stop.load(.acquire)) return;

                        const tile_idx0: usize = i;
                        const ti_n: usize = tile_idx0 / tc0;
                        const group_end: usize = @min(end, (ti_n + 1) * tc0);

                        const ti_m0: usize = tile_idx0 - ti_n * tc0;
                        const ti_m_end: usize = group_end - ti_n * tc0;

                        var ti_k: usize = 0;
                        while (ti_k < k_tiles) : (ti_k += 1) {
                            if (t.stop.load(.acquire)) return;
                            const beta_tile: f32 = if (ti_k == 0) t.s.beta else 1.0;

                            const b_tile = t.store.acquireTileConst(t.s.b, ti_k, ti_n) catch |e| {
                                t.fail(e);
                                return;
                            };
                            defer t.store.releaseConst(b_tile.token);
                            const b_view = b_tile.bufferView();

                            // Fast path for matvec-shaped problems in tiled execution:
                            // when there is exactly one M-tile, packing B for reuse is pointless.
                            if (is_matvec and (t.b_dtype != .f16 or t.c_dtype == .f16)) {
                                const k_tile: usize = b_view.layout.shape[0];
                                const n_tile: usize = b_view.layout.shape[1];

                                const ti_m: usize = 0;
                                var c_tile = t.store.acquireTileMut(t.s.c, ti_m, ti_n) catch |e| {
                                    t.fail(e);
                                    return;
                                };
                                defer t.store.releaseMut(c_tile.token);
                                const c_view0 = c_tile.bufferView();

                                const a_tile = t.store.acquireTileConst(t.s.a, ti_m, ti_k) catch |e| {
                                    t.fail(e);
                                    return;
                                };
                                defer t.store.releaseConst(a_tile.token);
                                const a_view = a_tile.bufferView();

                                const beta_eff: f32 = if (ti_k == 0) t.s.beta else 1.0;
                                const params: MatMulParams = .{ .m = 1, .n = n_tile, .k = k_tile, .alpha = t.s.alpha, .beta = beta_eff };

                                switch (t.b_dtype) {
                                    .f32 => t.matvec.matvec_f32(params, c_view0.bytes, a_view.bytes, b_view.bytes) catch |e| {
                                        t.fail(e);
                                        return;
                                    },
                                    .f16 => t.matvec.matvec_f16(params, c_view0.bytes, a_view.bytes, b_view.bytes) catch |e| {
                                        t.fail(e);
                                        return;
                                    },
                                    .q8_0 => {
                                        if (shouldUseQ8DirectMatvec(params, t.thread_count)) {
                                            quant_matmul_kernels.matvecQ8_0KMajor(params, c_view0.bytes, a_view.bytes, b_view.bytes) catch |e| {
                                                t.fail(e);
                                                return;
                                            };
                                        } else {
                                            const qk: quant_matmul_registry.QuantKernels = if (k_tile <= t.matmul_qx0.tuning.kc and n_tile <= t.matmul_qx0.tuning.nc)
                                                t.matmul_qx0
                                            else
                                                (quant_matmul_registry.selectForTile(t.matmul_qx0, k_tile, n_tile) orelse {
                                                    t.fail(BackendError.InvalidArgument);
                                                    return;
                                                });

                                            qk.pack_b_tile_q8_0(t.scratch[tid], k_tile, n_tile, b_view.bytes) catch |e| {
                                                t.fail(e);
                                                return;
                                            };

                                            const packed_b_view: quant_matmul_registry.PackedBView = @alignCast(t.scratch[tid][0..qk.packed_b_bytes]);
                                            qk.matmul_packed_b(t.scratch[tid], packed_b_view, params, c_view0.bytes, a_view.bytes) catch |e| {
                                                t.fail(e);
                                                return;
                                            };
                                        }
                                    },
                                    .q4_0 => {
                                        const qk: quant_matmul_registry.QuantKernels = if (k_tile <= t.matmul_qx0.tuning.kc and n_tile <= t.matmul_qx0.tuning.nc)
                                            t.matmul_qx0
                                        else
                                            (quant_matmul_registry.selectForTile(t.matmul_qx0, k_tile, n_tile) orelse {
                                                t.fail(BackendError.InvalidArgument);
                                                return;
                                            });

                                        if (t.b_dtype == .q4_0) {
                                            qk.pack_b_tile_q4_0(t.scratch[tid], k_tile, n_tile, b_view.bytes) catch |e| {
                                                t.fail(e);
                                                return;
                                            };
                                        } else {
                                            qk.pack_b_tile_q8_0(t.scratch[tid], k_tile, n_tile, b_view.bytes) catch |e| {
                                                t.fail(e);
                                                return;
                                            };
                                        }

                                        const packed_b_view: quant_matmul_registry.PackedBView = @alignCast(t.scratch[tid][0..qk.packed_b_bytes]);
                                        qk.matmul_packed_b(t.scratch[tid], packed_b_view, params, c_view0.bytes, a_view.bytes) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                    },
                                    else => {
                                        t.fail(BackendError.Unsupported);
                                        return;
                                    },
                                }

                                continue;
                            }

                            switch (t.b_dtype) {
                                .q4_0, .q8_0 => {
                                    const k_tile: usize = b_view.layout.shape[0];
                                    const n_tile: usize = b_view.layout.shape[1];

                                    const qk: quant_matmul_registry.QuantKernels = if (k_tile <= t.matmul_qx0.tuning.kc and n_tile <= t.matmul_qx0.tuning.nc)
                                        t.matmul_qx0
                                    else
                                        (quant_matmul_registry.selectForTile(t.matmul_qx0, k_tile, n_tile) orelse {
                                            t.fail(BackendError.InvalidArgument);
                                            return;
                                        });

                                    if (t.b_dtype == .q4_0) {
                                        qk.pack_b_tile_q4_0(t.scratch[tid], k_tile, n_tile, b_view.bytes) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                    } else {
                                        qk.pack_b_tile_q8_0(t.scratch[tid], k_tile, n_tile, b_view.bytes) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                    }

                                    const pb_bytes_len: usize = qk.packed_b_bytes;
                                    const packed_b_view: quant_matmul_registry.PackedBView = @alignCast(t.scratch[tid][0..pb_bytes_len]);

                                    var ti_m: usize = ti_m0;
                                    while (ti_m < ti_m_end) : (ti_m += 1) {
                                        if (t.stop.load(.acquire)) return;

                                        var c_tile = t.store.acquireTileMut(t.s.c, ti_m, ti_n) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                        defer t.store.releaseMut(c_tile.token);
                                        const c_view0 = c_tile.bufferView();
                                        const m_tile: usize = c_view0.layout.shape[0];

                                        const a_tile = t.store.acquireTileConst(t.s.a, ti_m, ti_k) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                        defer t.store.releaseConst(a_tile.token);
                                        const a_view = a_tile.bufferView();

                                        const params: MatMulParams = .{ .m = m_tile, .n = n_tile, .k = k_tile, .alpha = t.s.alpha, .beta = beta_tile };
                                        qk.matmul_packed_b(t.scratch[tid], packed_b_view, params, c_view0.bytes, a_view.bytes) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                    }
                                },
                                .f32 => {
                                    const k_tile: usize = b_view.layout.shape[0];
                                    const n_tile: usize = b_view.layout.shape[1];

                                    if (k_tile > t.matmul_f32.tuning.kc or n_tile > t.matmul_f32.tuning.nc) {
                                        var ti_m: usize = ti_m0;
                                        while (ti_m < ti_m_end) : (ti_m += 1) {
                                            if (t.stop.load(.acquire)) return;

                                            var c_tile = t.store.acquireTileMut(t.s.c, ti_m, ti_n) catch |e| {
                                                t.fail(e);
                                                return;
                                            };
                                            defer t.store.releaseMut(c_tile.token);

                                            const c_view0 = c_tile.bufferView();
                                            const m_total: usize = c_view0.layout.shape[0];

                                            const a_tile = t.store.acquireTileConst(t.s.a, ti_m, ti_k) catch |e| {
                                                t.fail(e);
                                                return;
                                            };
                                            defer t.store.releaseConst(a_tile.token);
                                            const a_view = a_tile.bufferView();

                                            const kk_total: usize = k_tile;
                                            const jj_total: usize = n_tile;

                                            const pb_f32_len: usize = t.matmul_f32.tuning.kc * t.matmul_f32.tuning.nc;
                                            const pb_bytes_len: usize = pb_f32_len * @sizeOf(f32);
                                            const packed_b_view: []align(32) const f32 = @alignCast(std.mem.bytesAsSlice(f32, t.scratch[tid][0..pb_bytes_len]));

                                            var jj: usize = 0;
                                            while (jj < jj_total) : (jj += t.matmul_f32.tuning.nc) {
                                                const n_sub: usize = @min(t.matmul_f32.tuning.nc, jj_total - jj);

                                                var kk: usize = 0;
                                                while (kk < kk_total) : (kk += t.matmul_f32.tuning.kc) {
                                                    const k_sub: usize = @min(t.matmul_f32.tuning.kc, kk_total - kk);
                                                    const beta_eff: f32 = if (kk == 0) beta_tile else 1.0;

                                                    const b_off: usize = (kk * jj_total + jj) * @sizeOf(f32);
                                                    const b_need: usize = k_sub * n_sub * @sizeOf(f32);
                                                    if (b_off + b_need > b_view.bytes.len) {
                                                        t.fail(BackendError.InvalidArgument);
                                                        return;
                                                    }
                                                    const b_sub_bytes: []const u8 = b_view.bytes[b_off .. b_off + b_need];
                                                    t.matmul_f32.pack_b_tile(t.scratch[tid], k_sub, n_sub, b_sub_bytes) catch |e| {
                                                        t.fail(e);
                                                        return;
                                                    };

                                                    const a_row_bytes: usize = kk_total * @sizeOf(f32);
                                                    const a_k_off: usize = kk * @sizeOf(f32);

                                                    const c_row_bytes: usize = jj_total * @sizeOf(f32);
                                                    const c_n_off: usize = jj * @sizeOf(f32);

                                                    var row: usize = 0;
                                                    while (row < m_total) : (row += 1) {
                                                        const a_row: []const u8 = a_view.bytes[row * a_row_bytes .. (row + 1) * a_row_bytes];
                                                        const c_row: []u8 = c_view0.bytes[row * c_row_bytes .. (row + 1) * c_row_bytes];

                                                        const a_sub: []const u8 = a_row[a_k_off .. a_k_off + k_sub * @sizeOf(f32)];
                                                        const c_sub: []u8 = c_row[c_n_off .. c_n_off + n_sub * @sizeOf(f32)];

                                                        const pp: MatMulParams = .{ .m = 1, .n = n_sub, .k = k_sub, .alpha = t.s.alpha, .beta = beta_eff };
                                                        t.matmul_f32.matmul_packed_b(t.scratch[tid], packed_b_view, pp, c_sub, a_sub) catch |e| {
                                                            t.fail(e);
                                                            return;
                                                        };
                                                    }
                                                }
                                            }

                                            continue;
                                        }

                                        continue;
                                    }

                                    t.matmul_f32.pack_b_tile(t.scratch[tid], k_tile, n_tile, b_view.bytes) catch |e| {
                                        t.fail(e);
                                        return;
                                    };

                                    var ti_m: usize = ti_m0;
                                    while (ti_m < ti_m_end) : (ti_m += 1) {
                                        if (t.stop.load(.acquire)) return;

                                        var c_tile = t.store.acquireTileMut(t.s.c, ti_m, ti_n) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                        defer t.store.releaseMut(c_tile.token);
                                        const c_view0 = c_tile.bufferView();
                                        const m_tile: usize = c_view0.layout.shape[0];

                                        const a_tile = t.store.acquireTileConst(t.s.a, ti_m, ti_k) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                        defer t.store.releaseConst(a_tile.token);
                                        const a_view = a_tile.bufferView();

                                        const params: MatMulParams = .{ .m = m_tile, .n = n_tile, .k = k_tile, .alpha = t.s.alpha, .beta = beta_tile };

                                        const pb_f32_len: usize = t.matmul_f32.tuning.kc * t.matmul_f32.tuning.nc;
                                        const pb_bytes_len: usize = pb_f32_len * @sizeOf(f32);
                                        const packed_b_view: []align(32) const f32 = @alignCast(std.mem.bytesAsSlice(f32, t.scratch[tid][0..pb_bytes_len]));
                                        t.matmul_f32.matmul_packed_b(t.scratch[tid], packed_b_view, params, c_view0.bytes, a_view.bytes) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                    }
                                },
                                .f16 => {
                                    const k_tile: usize = b_view.layout.shape[0];
                                    const n_tile: usize = b_view.layout.shape[1];

                                    const mk: matmul_registry.F32Kernels = matmul_registry.selectForTile(t.matmul_f32, k_tile, n_tile) orelse t.matmul_f32;
                                    if (k_tile > mk.tuning.kc or n_tile > mk.tuning.nc) {
                                        t.fail(BackendError.InvalidArgument);
                                        return;
                                    }

                                    const pb_f32_len: usize = mk.tuning.kc * mk.tuning.nc;
                                    const pb_bytes_len: usize = pb_f32_len * @sizeOf(f32);
                                    if (t.scratch[tid].len < pb_bytes_len) {
                                        t.fail(BackendError.InvalidArgument);
                                        return;
                                    }

                                    const packed_b_mut: []align(32) f32 = @alignCast(std.mem.bytesAsSlice(f32, t.scratch[tid][0..pb_bytes_len]));
                                    mk.pack_b_tile_f16_to_packed_f32(packed_b_mut, k_tile, n_tile, b_view.bytes) catch |e| {
                                        t.fail(e);
                                        return;
                                    };

                                    var ti_m: usize = ti_m0;
                                    while (ti_m < ti_m_end) : (ti_m += 1) {
                                        if (t.stop.load(.acquire)) return;

                                        var c_tile = t.store.acquireTileMut(t.s.c, ti_m, ti_n) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                        defer t.store.releaseMut(c_tile.token);
                                        const c_view0 = c_tile.bufferView();
                                        const m_tile: usize = c_view0.layout.shape[0];
                                        const n_tile_m: usize = c_view0.layout.shape[1];

                                        const a_tile = t.store.acquireTileConst(t.s.a, ti_m, ti_k) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                        defer t.store.releaseConst(a_tile.token);
                                        const a_view = a_tile.bufferView();

                                        const k_tile_m: usize = a_view.layout.shape[1];
                                        const params: MatMulParams = .{ .m = m_tile, .n = n_tile_m, .k = k_tile_m, .alpha = t.s.alpha, .beta = beta_tile };

                                        runF16TilePackedB(mk, t.scratch[tid], packed_b_mut, params, t.c_dtype, c_view0.bytes, a_view.bytes) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                    }
                                },
                                else => {
                                    t.fail(BackendError.Unsupported);
                                    return;
                                },
                            }
                        }

                        i = group_end;
                    }
                }
            };

            var task: Task = .{
                .store = store,
                .c_meta = c_meta,
                .a_meta = a_meta,
                .a_dtype = a_dtype,
                .b_dtype = b_dtype,
                .c_dtype = c_dtype,
                .s = s,
                .scratch = ctx.matmul_scratch,
                .matmul_f32 = ctx.matmul_f32,
                .matmul_qx0 = ctx.matmul_qx0,
                .matvec = ctx.matvec,
                .thread_count = ctx.thread_count,
            };

            const threads_total: usize = ctx.thread_count;
            const tc0: usize = c_meta.tile_counts[0];
            const max_grain_for_parallelism: usize = @max(@as(usize, 1), total_work / threads_total);

            var grain: usize = @min(@min(tc0, max_grain_for_parallelism), @as(usize, 32));
            while (grain > 1 and (tc0 % grain) != 0) : (grain -= 1) {}

            p.parallelForAny(@ptrCast(&task), total_work, grain, Task.runTiles);
            if (task.err_any) |e| return @errorCast(e);
            return;
        }
    }

    // Sequential fallback.
    var ti_m: usize = 0;
    while (ti_m < c_meta.tile_counts[0]) : (ti_m += 1) {
        var ti_n: usize = 0;
        while (ti_n < c_meta.tile_counts[1]) : (ti_n += 1) {
            var c_tile = try store.acquireTileMut(s.c, ti_m, ti_n);
            defer store.releaseMut(c_tile.token);

            const c_view0 = c_tile.bufferView();
            const m_tile: usize = c_view0.layout.shape[0];
            const n_tile: usize = c_view0.layout.shape[1];

            var ti_k: usize = 0;
            while (ti_k < a_meta.tile_counts[1]) : (ti_k += 1) {
                const a_tile = try store.acquireTileConst(s.a, ti_m, ti_k);
                defer store.releaseConst(a_tile.token);
                const b_tile = try store.acquireTileConst(s.b, ti_k, ti_n);
                defer store.releaseConst(b_tile.token);

                const a_view = a_tile.bufferView();
                const b_view = b_tile.bufferView();
                const c_view = c_tile.bufferView();

                const k_tile: usize = a_view.layout.shape[1];
                const beta_tile: f32 = if (ti_k == 0) s.beta else 1.0;
                const params: MatMulParams = .{ .m = m_tile, .n = n_tile, .k = k_tile, .alpha = s.alpha, .beta = beta_tile };

                switch (b_dtype) {
                    .f32 => {
                        if (m_tile == 1) {
                            try ctx.matvec.matvec_f32(params, c_view.bytes, a_view.bytes, b_view.bytes);
                        } else {
                            const mk: matmul_registry.F32Kernels = if (k_tile <= ctx.matmul_f32.tuning.kc and n_tile <= ctx.matmul_f32.tuning.nc)
                                ctx.matmul_f32
                            else
                                (matmul_registry.selectForTile(ctx.matmul_f32, k_tile, n_tile) orelse return BackendError.InvalidArgument);

                            var scratch_buf: []align(32) u8 = if (ctx.matmul_scratch.len != 0) ctx.matmul_scratch[0] else blk: {
                                const scratch_need: usize = @max(matmul_registry.maxScratchBytes(), quant_matmul_registry.maxScratchBytes());
                                const tmp = ctx.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), scratch_need) catch return BackendError.ExecutionFailed;
                                break :blk tmp;
                            };
                            defer if (ctx.matmul_scratch.len == 0) ctx.allocator.free(scratch_buf);

                            try mk.pack_b_tile(scratch_buf, k_tile, n_tile, b_view.bytes);
                            const pb_f32_len: usize = mk.tuning.kc * mk.tuning.nc;
                            const pb_bytes_len: usize = pb_f32_len * @sizeOf(f32);
                            const packed_b_view: []align(32) const f32 = @alignCast(std.mem.bytesAsSlice(f32, scratch_buf[0..pb_bytes_len]));
                            try mk.matmul_packed_b(scratch_buf, packed_b_view, params, c_view.bytes, a_view.bytes);
                        }
                    },
                    .f16 => {
                        const mk: matmul_registry.F32Kernels = matmul_registry.selectForTile(ctx.matmul_f32, k_tile, n_tile) orelse ctx.matmul_f32;
                        const scratch_buf: []align(32) u8 = if (ctx.matmul_scratch.len != 0) ctx.matmul_scratch[0] else blk: {
                            const scratch_need: usize = @max(matmul_registry.maxScratchBytes(), quant_matmul_registry.maxScratchBytes());
                            const tmp = ctx.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), scratch_need) catch return BackendError.ExecutionFailed;
                            break :blk tmp;
                        };
                        defer if (ctx.matmul_scratch.len == 0) ctx.allocator.free(scratch_buf);

                        if (m_tile == 1 and c_dtype == .f16) {
                            try ctx.matvec.matvec_f16(params, c_view.bytes, a_view.bytes, b_view.bytes);
                        } else {
                            try matmulF16ViaPackedF32(mk, scratch_buf, params, c_dtype, c_view.bytes, a_view.bytes, b_view.bytes);
                        }
                    },
                    .q4_0 => {
                        const qk: quant_matmul_registry.QuantKernels = if (k_tile <= ctx.matmul_qx0.tuning.kc and n_tile <= ctx.matmul_qx0.tuning.nc)
                            ctx.matmul_qx0
                        else
                            (quant_matmul_registry.selectForTile(ctx.matmul_qx0, k_tile, n_tile) orelse return BackendError.InvalidArgument);

                        var scratch_buf: []align(32) u8 = if (ctx.matmul_scratch.len != 0) ctx.matmul_scratch[0] else blk: {
                            const scratch_need: usize = @max(matmul_registry.maxScratchBytes(), quant_matmul_registry.maxScratchBytes());
                            const tmp = ctx.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), scratch_need) catch return BackendError.ExecutionFailed;
                            break :blk tmp;
                        };
                        defer if (ctx.matmul_scratch.len == 0) ctx.allocator.free(scratch_buf);

                        try qk.pack_b_tile_q4_0(scratch_buf, k_tile, n_tile, b_view.bytes);
                        const packed_b_view: quant_matmul_registry.PackedBView = @alignCast(scratch_buf[0..qk.packed_b_bytes]);
                        try qk.matmul_packed_b(scratch_buf, packed_b_view, params, c_view.bytes, a_view.bytes);
                    },
                    .q8_0 => {
                        if (m_tile == 1 and shouldUseQ8DirectMatvec(params, ctx.thread_count)) {
                            try quant_matmul_kernels.matvecQ8_0KMajor(params, c_view.bytes, a_view.bytes, b_view.bytes);
                        } else {
                            const qk: quant_matmul_registry.QuantKernels = if (k_tile <= ctx.matmul_qx0.tuning.kc and n_tile <= ctx.matmul_qx0.tuning.nc)
                                ctx.matmul_qx0
                            else
                                (quant_matmul_registry.selectForTile(ctx.matmul_qx0, k_tile, n_tile) orelse return BackendError.InvalidArgument);

                            var scratch_buf: []align(32) u8 = if (ctx.matmul_scratch.len != 0) ctx.matmul_scratch[0] else blk: {
                                const scratch_need: usize = @max(matmul_registry.maxScratchBytes(), quant_matmul_registry.maxScratchBytes());
                                const tmp = ctx.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), scratch_need) catch return BackendError.ExecutionFailed;
                                break :blk tmp;
                            };
                            defer if (ctx.matmul_scratch.len == 0) ctx.allocator.free(scratch_buf);

                            try qk.pack_b_tile_q8_0(scratch_buf, k_tile, n_tile, b_view.bytes);
                            const packed_b_view: quant_matmul_registry.PackedBView = @alignCast(scratch_buf[0..qk.packed_b_bytes]);
                            try qk.matmul_packed_b(scratch_buf, packed_b_view, params, c_view.bytes, a_view.bytes);
                        }
                    },
                    else => return BackendError.Unsupported,
                }
            }
        }
    }
}

fn execMatMulTiledBatched(
    ctx: *MatMulExecCtx,
    s: executable.StepMatMulTiled,
    store: tensor_store.TensorStore,
    c_meta: tensor_store.TensorMeta,
    a_meta: tensor_store.TensorMeta,
    b_meta: tensor_store.TensorMeta,
) ExecuteProgramError!void {
    const b_dtype: DType = b_meta.dtype;
    const c_dtype: DType = c_meta.dtype;

    if (b_dtype == .f16) {
        return execMatMulTiledBatchedF16Grouped(ctx, s, store, c_meta, a_meta, b_meta, c_dtype);
    }

    const rank: usize = @as(usize, c_meta.rank);
    if (rank < 3 or rank > 8) return BackendError.InvalidArgument;

    var tile_total: usize = 1;
    var d: usize = 0;
    while (d < rank) : (d += 1) {
        tile_total *= c_meta.tile_counts[d];
    }

    const k_tiles: usize = a_meta.tile_counts[rank - 1];

    if (ctx.pool) |p| {
        if (ctx.thread_count > 1 and tile_total >= 2) {
            const Task = struct {
                store: tensor_store.TensorStore,
                c_meta: tensor_store.TensorMeta,
                a_meta: tensor_store.TensorMeta,
                b_meta: tensor_store.TensorMeta,

                b_dtype: DType,
                c_dtype: DType,

                s: executable.StepMatMulTiled,
                scratch: [][]align(32) u8,
                matmul_f32: matmul_registry.F32Kernels,
                matmul_qx0: quant_matmul_registry.QuantKernels,
                matvec: matvec_registry.Kernels,
                thread_count: usize,

                stop: std.atomic.Value(bool) = .init(false),
                err_mutex: std.Io.Mutex = .init,
                err_any: ?anyerror = null,

                fn fail(t: *@This(), err: anyerror) void {
                    if (t.stop.swap(true, .acq_rel)) return;
                    std.Io.Threaded.mutexLock(&t.err_mutex);
                    defer std.Io.Threaded.mutexUnlock(&t.err_mutex);
                    if (t.err_any == null) t.err_any = err;
                }

                fn runTiles(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
                    const t: *@This() = @ptrCast(@alignCast(ctx_any));
                    if (start >= end) return;
                    if (tid >= t.scratch.len) return;
                    if (t.stop.load(.acquire)) return;

                    const rank_local: usize = @as(usize, t.c_meta.rank);
                    const k_tiles_local: usize = t.a_meta.tile_counts[rank_local - 1];

                    var coords_buf: [8]usize = undefined;
                    var a_coords_buf: [8]usize = undefined;
                    var b_coords_buf: [8]usize = undefined;

                    var tile_index: usize = start;
                    while (tile_index < end) : (tile_index += 1) {
                        if (t.stop.load(.acquire)) return;

                        const coords: []usize = coords_buf[0..rank_local];
                        tensor_store.decodeTileCoords(t.c_meta, tile_index, coords) catch |e| {
                            t.fail(e);
                            return;
                        };

                        const ti_m: usize = coords[rank_local - 2];
                        const ti_n: usize = coords[rank_local - 1];

                        var c_tile = t.store.acquireTileMutLinear(t.s.c, tile_index) catch |e| {
                            t.fail(e);
                            return;
                        };
                        defer t.store.releaseMut(c_tile.token);
                        const c_view0 = c_tile.bufferView();
                        const m_tile: usize = c_view0.layout.shape[rank_local - 2];
                        const n_tile: usize = c_view0.layout.shape[rank_local - 1];

                        var ti_k: usize = 0;
                        while (ti_k < k_tiles_local) : (ti_k += 1) {
                            const beta_tile: f32 = if (ti_k == 0) t.s.beta else 1.0;

                            const a_coords: []usize = a_coords_buf[0..rank_local];
                            const b_coords: []usize = b_coords_buf[0..rank_local];

                            var bd: usize = 0;
                            while (bd + 2 < rank_local) : (bd += 1) {
                                a_coords[bd] = if (t.a_meta.shape[bd] == 1) 0 else coords[bd];
                                b_coords[bd] = if (t.b_meta.shape[bd] == 1) 0 else coords[bd];
                            }
                            a_coords[rank_local - 2] = ti_m;
                            a_coords[rank_local - 1] = ti_k;
                            b_coords[rank_local - 2] = ti_k;
                            b_coords[rank_local - 1] = ti_n;

                            const a_tile_index: usize = tensor_store.encodeTileIndex(t.a_meta, a_coords) catch |e| {
                                t.fail(e);
                                return;
                            };
                            const b_tile_index: usize = tensor_store.encodeTileIndex(t.b_meta, b_coords) catch |e| {
                                t.fail(e);
                                return;
                            };

                            const a_tile = t.store.acquireTileConstLinear(t.s.a, a_tile_index) catch |e| {
                                t.fail(e);
                                return;
                            };
                            defer t.store.releaseConst(a_tile.token);
                            const b_tile = t.store.acquireTileConstLinear(t.s.b, b_tile_index) catch |e| {
                                t.fail(e);
                                return;
                            };
                            defer t.store.releaseConst(b_tile.token);

                            const a_view = a_tile.bufferView();
                            const b_view = b_tile.bufferView();
                            const k_tile: usize = a_view.layout.shape[rank_local - 1];
                            const params: MatMulParams = .{ .m = m_tile, .n = n_tile, .k = k_tile, .alpha = t.s.alpha, .beta = beta_tile };

                            switch (t.b_dtype) {
                                .f32 => {
                                    if (m_tile == 1) {
                                        t.matvec.matvec_f32(params, c_view0.bytes, a_view.bytes, b_view.bytes) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                    } else {
                                        const mk: matmul_registry.F32Kernels = if (k_tile <= t.matmul_f32.tuning.kc and n_tile <= t.matmul_f32.tuning.nc)
                                            t.matmul_f32
                                        else
                                            (matmul_registry.selectForTile(t.matmul_f32, k_tile, n_tile) orelse {
                                                t.fail(BackendError.InvalidArgument);
                                                return;
                                            });

                                        mk.pack_b_tile(t.scratch[tid], k_tile, n_tile, b_view.bytes) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                        const pb_f32_len: usize = mk.tuning.kc * mk.tuning.nc;
                                        const pb_bytes_len: usize = pb_f32_len * @sizeOf(f32);
                                        const packed_b_view: []align(32) const f32 = @alignCast(std.mem.bytesAsSlice(f32, t.scratch[tid][0..pb_bytes_len]));
                                        mk.matmul_packed_b(t.scratch[tid], packed_b_view, params, c_view0.bytes, a_view.bytes) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                    }
                                },
                                .f16 => {
                                    const mk: matmul_registry.F32Kernels = matmul_registry.selectForTile(t.matmul_f32, k_tile, n_tile) orelse t.matmul_f32;
                                    if (m_tile == 1 and t.c_dtype == .f16) {
                                        t.matvec.matvec_f16(params, c_view0.bytes, a_view.bytes, b_view.bytes) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                    } else {
                                        matmulF16ViaPackedF32(mk, t.scratch[tid], params, t.c_dtype, c_view0.bytes, a_view.bytes, b_view.bytes) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                    }
                                },
                                .q4_0 => {
                                    const qk: quant_matmul_registry.QuantKernels = if (k_tile <= t.matmul_qx0.tuning.kc and n_tile <= t.matmul_qx0.tuning.nc)
                                        t.matmul_qx0
                                    else
                                        (quant_matmul_registry.selectForTile(t.matmul_qx0, k_tile, n_tile) orelse {
                                            t.fail(BackendError.InvalidArgument);
                                            return;
                                        });

                                    qk.pack_b_tile_q4_0(t.scratch[tid], k_tile, n_tile, b_view.bytes) catch |e| {
                                        t.fail(e);
                                        return;
                                    };

                                    const packed_b_view: quant_matmul_registry.PackedBView = @alignCast(t.scratch[tid][0..qk.packed_b_bytes]);
                                    qk.matmul_packed_b(t.scratch[tid], packed_b_view, params, c_view0.bytes, a_view.bytes) catch |e| {
                                        t.fail(e);
                                        return;
                                    };
                                },
                                .q8_0 => {
                                    if (m_tile == 1 and shouldUseQ8DirectMatvec(params, t.thread_count)) {
                                        quant_matmul_kernels.matvecQ8_0KMajor(params, c_view0.bytes, a_view.bytes, b_view.bytes) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                    } else {
                                        const qk: quant_matmul_registry.QuantKernels = if (k_tile <= t.matmul_qx0.tuning.kc and n_tile <= t.matmul_qx0.tuning.nc)
                                            t.matmul_qx0
                                        else
                                            (quant_matmul_registry.selectForTile(t.matmul_qx0, k_tile, n_tile) orelse {
                                                t.fail(BackendError.InvalidArgument);
                                                return;
                                            });

                                        qk.pack_b_tile_q8_0(t.scratch[tid], k_tile, n_tile, b_view.bytes) catch |e| {
                                            t.fail(e);
                                            return;
                                        };

                                        const packed_b_view: quant_matmul_registry.PackedBView = @alignCast(t.scratch[tid][0..qk.packed_b_bytes]);
                                        qk.matmul_packed_b(t.scratch[tid], packed_b_view, params, c_view0.bytes, a_view.bytes) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                    }
                                },
                                else => {
                                    t.fail(BackendError.Unsupported);
                                    return;
                                },
                            }
                        }
                    }
                }
            };

            var task: Task = .{
                .store = store,
                .c_meta = c_meta,
                .a_meta = a_meta,
                .b_meta = b_meta,
                .b_dtype = b_dtype,
                .c_dtype = c_dtype,
                .s = s,
                .scratch = ctx.matmul_scratch,
                .matmul_f32 = ctx.matmul_f32,
                .matmul_qx0 = ctx.matmul_qx0,
                .matvec = ctx.matvec,
                .thread_count = ctx.thread_count,
            };

            const threads_total: usize = ctx.thread_count;
            const max_grain_for_parallelism: usize = @max(@as(usize, 1), tile_total / threads_total);
            var grain: usize = @min(@as(usize, 32), max_grain_for_parallelism);
            if (grain == 0) grain = 1;

            p.parallelForAny(@ptrCast(&task), tile_total, grain, Task.runTiles);
            if (task.err_any) |e| return @errorCast(e);
            return;
        }
    }

    var coords_buf: [8]usize = undefined;
    var a_coords_buf: [8]usize = undefined;
    var b_coords_buf: [8]usize = undefined;

    var tile_index: usize = 0;
    while (tile_index < tile_total) : (tile_index += 1) {
        const coords: []usize = coords_buf[0..rank];
        try tensor_store.decodeTileCoords(c_meta, tile_index, coords);

        const ti_m: usize = coords[rank - 2];
        const ti_n: usize = coords[rank - 1];

        var c_tile = try store.acquireTileMutLinear(s.c, tile_index);
        defer store.releaseMut(c_tile.token);
        const c_view0 = c_tile.bufferView();
        const m_tile: usize = c_view0.layout.shape[rank - 2];
        const n_tile: usize = c_view0.layout.shape[rank - 1];

        var ti_k: usize = 0;
        while (ti_k < k_tiles) : (ti_k += 1) {
            const beta_tile: f32 = if (ti_k == 0) s.beta else 1.0;

            const a_coords: []usize = a_coords_buf[0..rank];
            const b_coords: []usize = b_coords_buf[0..rank];

            var bd: usize = 0;
            while (bd + 2 < rank) : (bd += 1) {
                a_coords[bd] = if (a_meta.shape[bd] == 1) 0 else coords[bd];
                b_coords[bd] = if (b_meta.shape[bd] == 1) 0 else coords[bd];
            }
            a_coords[rank - 2] = ti_m;
            a_coords[rank - 1] = ti_k;
            b_coords[rank - 2] = ti_k;
            b_coords[rank - 1] = ti_n;

            const a_tile_index: usize = try tensor_store.encodeTileIndex(a_meta, a_coords);
            const b_tile_index: usize = try tensor_store.encodeTileIndex(b_meta, b_coords);

            const a_tile = try store.acquireTileConstLinear(s.a, a_tile_index);
            defer store.releaseConst(a_tile.token);
            const b_tile = try store.acquireTileConstLinear(s.b, b_tile_index);
            defer store.releaseConst(b_tile.token);

            const a_view = a_tile.bufferView();
            const b_view = b_tile.bufferView();
            const k_tile: usize = a_view.layout.shape[rank - 1];
            const params: MatMulParams = .{ .m = m_tile, .n = n_tile, .k = k_tile, .alpha = s.alpha, .beta = beta_tile };

            switch (b_dtype) {
                .f32 => {
                    if (m_tile == 1) {
                        try ctx.matvec.matvec_f32(params, c_view0.bytes, a_view.bytes, b_view.bytes);
                    } else {
                        const mk: matmul_registry.F32Kernels = if (k_tile <= ctx.matmul_f32.tuning.kc and n_tile <= ctx.matmul_f32.tuning.nc)
                            ctx.matmul_f32
                        else
                            (matmul_registry.selectForTile(ctx.matmul_f32, k_tile, n_tile) orelse return BackendError.InvalidArgument);

                        var scratch_buf: []align(32) u8 = if (ctx.matmul_scratch.len != 0) ctx.matmul_scratch[0] else blk: {
                            const scratch_need: usize = @max(matmul_registry.maxScratchBytes(), quant_matmul_registry.maxScratchBytes());
                            const tmp = ctx.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), scratch_need) catch return BackendError.ExecutionFailed;
                            break :blk tmp;
                        };
                        defer if (ctx.matmul_scratch.len == 0) ctx.allocator.free(scratch_buf);

                        try mk.pack_b_tile(scratch_buf, k_tile, n_tile, b_view.bytes);
                        const pb_f32_len: usize = mk.tuning.kc * mk.tuning.nc;
                        const pb_bytes_len: usize = pb_f32_len * @sizeOf(f32);
                        const packed_b_view: []align(32) const f32 = @alignCast(std.mem.bytesAsSlice(f32, scratch_buf[0..pb_bytes_len]));
                        try mk.matmul_packed_b(scratch_buf, packed_b_view, params, c_view0.bytes, a_view.bytes);
                    }
                },
                .f16 => {
                    const mk: matmul_registry.F32Kernels = matmul_registry.selectForTile(ctx.matmul_f32, k_tile, n_tile) orelse ctx.matmul_f32;
                    const scratch_buf: []align(32) u8 = if (ctx.matmul_scratch.len != 0) ctx.matmul_scratch[0] else blk: {
                        const scratch_need: usize = @max(matmul_registry.maxScratchBytes(), quant_matmul_registry.maxScratchBytes());
                        const tmp = ctx.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), scratch_need) catch return BackendError.ExecutionFailed;
                        break :blk tmp;
                    };
                    defer if (ctx.matmul_scratch.len == 0) ctx.allocator.free(scratch_buf);

                    if (m_tile == 1 and c_dtype == .f16) {
                        try ctx.matvec.matvec_f16(params, c_view0.bytes, a_view.bytes, b_view.bytes);
                    } else {
                        try matmulF16ViaPackedF32(mk, scratch_buf, params, c_dtype, c_view0.bytes, a_view.bytes, b_view.bytes);
                    }
                },
                .q4_0 => {
                    const qk: quant_matmul_registry.QuantKernels = if (k_tile <= ctx.matmul_qx0.tuning.kc and n_tile <= ctx.matmul_qx0.tuning.nc)
                        ctx.matmul_qx0
                    else
                        (quant_matmul_registry.selectForTile(ctx.matmul_qx0, k_tile, n_tile) orelse return BackendError.InvalidArgument);

                    var scratch_buf: []align(32) u8 = if (ctx.matmul_scratch.len != 0) ctx.matmul_scratch[0] else blk: {
                        const scratch_need: usize = @max(matmul_registry.maxScratchBytes(), quant_matmul_registry.maxScratchBytes());
                        const tmp = ctx.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), scratch_need) catch return BackendError.ExecutionFailed;
                        break :blk tmp;
                    };
                    defer if (ctx.matmul_scratch.len == 0) ctx.allocator.free(scratch_buf);

                    try qk.pack_b_tile_q4_0(scratch_buf, k_tile, n_tile, b_view.bytes);
                    const packed_b_view: quant_matmul_registry.PackedBView = @alignCast(scratch_buf[0..qk.packed_b_bytes]);
                    try qk.matmul_packed_b(scratch_buf, packed_b_view, params, c_view0.bytes, a_view.bytes);
                },
                .q8_0 => {
                    if (m_tile == 1 and shouldUseQ8DirectMatvec(params, ctx.thread_count)) {
                        try quant_matmul_kernels.matvecQ8_0KMajor(params, c_view0.bytes, a_view.bytes, b_view.bytes);
                    } else {
                        const qk: quant_matmul_registry.QuantKernels = if (k_tile <= ctx.matmul_qx0.tuning.kc and n_tile <= ctx.matmul_qx0.tuning.nc)
                            ctx.matmul_qx0
                        else
                            (quant_matmul_registry.selectForTile(ctx.matmul_qx0, k_tile, n_tile) orelse return BackendError.InvalidArgument);

                        var scratch_buf: []align(32) u8 = if (ctx.matmul_scratch.len != 0) ctx.matmul_scratch[0] else blk: {
                            const scratch_need: usize = @max(matmul_registry.maxScratchBytes(), quant_matmul_registry.maxScratchBytes());
                            const tmp = ctx.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), scratch_need) catch return BackendError.ExecutionFailed;
                            break :blk tmp;
                        };
                        defer if (ctx.matmul_scratch.len == 0) ctx.allocator.free(scratch_buf);

                        try qk.pack_b_tile_q8_0(scratch_buf, k_tile, n_tile, b_view.bytes);
                        const packed_b_view: quant_matmul_registry.PackedBView = @alignCast(scratch_buf[0..qk.packed_b_bytes]);
                        try qk.matmul_packed_b(scratch_buf, packed_b_view, params, c_view0.bytes, a_view.bytes);
                    }
                },
                else => return BackendError.Unsupported,
            }
        }
    }
}

fn execMatMulTiledBatchedF16Grouped(
    ctx: *MatMulExecCtx,
    s: executable.StepMatMulTiled,
    store: tensor_store.TensorStore,
    c_meta: tensor_store.TensorMeta,
    a_meta: tensor_store.TensorMeta,
    b_meta: tensor_store.TensorMeta,
    c_dtype: DType,
) ExecuteProgramError!void {
    const rank: usize = @as(usize, c_meta.rank);
    if (rank < 3 or rank > 8) return BackendError.InvalidArgument;

    const prefix_rank: usize = rank - 2;
    const m_dim: usize = rank - 2;
    const n_dim: usize = rank - 1;

    const m_tiles: usize = c_meta.tile_counts[m_dim];
    const n_tiles: usize = c_meta.tile_counts[n_dim];
    const k_tiles: usize = a_meta.tile_counts[n_dim];

    var prefix_total: usize = 1;
    var d: usize = 0;
    while (d < prefix_rank) : (d += 1) {
        prefix_total *= c_meta.tile_counts[d];
    }
    const group_total: usize = prefix_total * n_tiles;

    if (ctx.pool) |p| {
        if (ctx.thread_count > 1 and group_total >= 2) {
            const Task = struct {
                store: tensor_store.TensorStore,
                c_meta: tensor_store.TensorMeta,
                a_meta: tensor_store.TensorMeta,
                b_meta: tensor_store.TensorMeta,
                c_dtype: DType,
                s: executable.StepMatMulTiled,
                scratch: [][]align(32) u8,
                matmul_f32: matmul_registry.F32Kernels,

                stop: std.atomic.Value(bool) = .init(false),
                err_mutex: std.Io.Mutex = .init,
                err_any: ?anyerror = null,

                fn fail(t: *@This(), err: anyerror) void {
                    if (t.stop.swap(true, .acq_rel)) return;
                    std.Io.Threaded.mutexLock(&t.err_mutex);
                    defer std.Io.Threaded.mutexUnlock(&t.err_mutex);
                    if (t.err_any == null) t.err_any = err;
                }

                fn decodePrefix(t: @This(), prefix_idx: usize, out: []usize) void {
                    var x: usize = prefix_idx;
                    var rr: usize = 0;
                    while (rr < out.len) : (rr += 1) {
                        const d_rev: usize = out.len - 1 - rr;
                        const tc: usize = t.c_meta.tile_counts[d_rev];
                        out[d_rev] = x % tc;
                        x /= tc;
                    }
                }

                fn runGroups(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
                    const t: *@This() = @ptrCast(@alignCast(ctx_any));
                    if (start >= end) return;
                    if (tid >= t.scratch.len) return;
                    if (t.stop.load(.acquire)) return;

                    const rank_local: usize = @as(usize, t.c_meta.rank);
                    const prefix_rank_local: usize = rank_local - 2;
                    const m_dim_local: usize = rank_local - 2;
                    const n_dim_local: usize = rank_local - 1;
                    const m_tiles_local: usize = t.c_meta.tile_counts[m_dim_local];
                    const n_tiles_local: usize = t.c_meta.tile_counts[n_dim_local];
                    const k_tiles_local: usize = t.a_meta.tile_counts[n_dim_local];

                    var prefix_coords: [8]usize = .{0} ** 8;
                    var c_coords: [8]usize = .{0} ** 8;
                    var a_coords: [8]usize = .{0} ** 8;
                    var b_coords: [8]usize = .{0} ** 8;

                    var g: usize = start;
                    while (g < end) : (g += 1) {
                        if (t.stop.load(.acquire)) return;

                        const prefix_idx: usize = g / n_tiles_local;
                        const ti_n: usize = g % n_tiles_local;
                        t.decodePrefix(prefix_idx, prefix_coords[0..prefix_rank_local]);

                        var ti_k: usize = 0;
                        while (ti_k < k_tiles_local) : (ti_k += 1) {
                            if (t.stop.load(.acquire)) return;

                            var pd: usize = 0;
                            while (pd < prefix_rank_local) : (pd += 1) {
                                b_coords[pd] = if (t.b_meta.shape[pd] == 1) 0 else prefix_coords[pd];
                            }
                            b_coords[m_dim_local] = ti_k;
                            b_coords[n_dim_local] = ti_n;

                            const b_idx: usize = tensor_store.encodeTileIndex(t.b_meta, b_coords[0..rank_local]) catch |e| {
                                t.fail(e);
                                return;
                            };
                            const b_tile = t.store.acquireTileConstLinear(t.s.b, b_idx) catch |e| {
                                t.fail(e);
                                return;
                            };
                            defer t.store.releaseConst(b_tile.token);
                            const b_view = b_tile.bufferView();

                            const k_tile: usize = b_view.layout.shape[m_dim_local];
                            const n_tile_b: usize = b_view.layout.shape[n_dim_local];
                            const mk: matmul_registry.F32Kernels = matmul_registry.selectForTile(t.matmul_f32, k_tile, n_tile_b) orelse t.matmul_f32;
                            if (k_tile > mk.tuning.kc or n_tile_b > mk.tuning.nc) {
                                t.fail(BackendError.InvalidArgument);
                                return;
                            }

                            const pb_f32_len: usize = mk.tuning.kc * mk.tuning.nc;
                            const pb_bytes_len: usize = pb_f32_len * @sizeOf(f32);
                            if (t.scratch[tid].len < pb_bytes_len) {
                                t.fail(BackendError.InvalidArgument);
                                return;
                            }

                            const packed_b_mut: []align(32) f32 = @alignCast(std.mem.bytesAsSlice(f32, t.scratch[tid][0..pb_bytes_len]));
                            mk.pack_b_tile_f16_to_packed_f32(packed_b_mut, k_tile, n_tile_b, b_view.bytes) catch |e| {
                                t.fail(e);
                                return;
                            };

                            var ti_m: usize = 0;
                            while (ti_m < m_tiles_local) : (ti_m += 1) {
                                if (t.stop.load(.acquire)) return;

                                var qd: usize = 0;
                                while (qd < prefix_rank_local) : (qd += 1) {
                                    c_coords[qd] = prefix_coords[qd];
                                    a_coords[qd] = if (t.a_meta.shape[qd] == 1) 0 else prefix_coords[qd];
                                }
                                c_coords[m_dim_local] = ti_m;
                                c_coords[n_dim_local] = ti_n;
                                a_coords[m_dim_local] = ti_m;
                                a_coords[n_dim_local] = ti_k;

                                const c_idx: usize = tensor_store.encodeTileIndex(t.c_meta, c_coords[0..rank_local]) catch |e| {
                                    t.fail(e);
                                    return;
                                };
                                const a_idx: usize = tensor_store.encodeTileIndex(t.a_meta, a_coords[0..rank_local]) catch |e| {
                                    t.fail(e);
                                    return;
                                };

                                var c_tile = t.store.acquireTileMutLinear(t.s.c, c_idx) catch |e| {
                                    t.fail(e);
                                    return;
                                };
                                defer t.store.releaseMut(c_tile.token);
                                const c_view = c_tile.bufferView();

                                const a_tile = t.store.acquireTileConstLinear(t.s.a, a_idx) catch |e| {
                                    t.fail(e);
                                    return;
                                };
                                defer t.store.releaseConst(a_tile.token);
                                const a_view = a_tile.bufferView();

                                const m_tile: usize = c_view.layout.shape[m_dim_local];
                                const n_tile: usize = c_view.layout.shape[n_dim_local];
                                const k_tile_m: usize = a_view.layout.shape[n_dim_local];
                                const beta_tile: f32 = if (ti_k == 0) t.s.beta else 1.0;
                                const params: MatMulParams = .{ .m = m_tile, .n = n_tile, .k = k_tile_m, .alpha = t.s.alpha, .beta = beta_tile };

                                runF16TilePackedB(mk, t.scratch[tid], packed_b_mut, params, t.c_dtype, c_view.bytes, a_view.bytes) catch |e| {
                                    t.fail(e);
                                    return;
                                };
                            }
                        }
                    }
                }
            };

            var task: Task = .{
                .store = store,
                .c_meta = c_meta,
                .a_meta = a_meta,
                .b_meta = b_meta,
                .c_dtype = c_dtype,
                .s = s,
                .scratch = ctx.matmul_scratch,
                .matmul_f32 = ctx.matmul_f32,
            };

            const threads_total: usize = ctx.thread_count;
            const max_grain_for_parallelism: usize = @max(@as(usize, 1), group_total / threads_total);
            var grain: usize = @min(@as(usize, 16), max_grain_for_parallelism);
            if (grain == 0) grain = 1;

            p.parallelForAny(@ptrCast(&task), group_total, grain, Task.runGroups);
            if (task.err_any) |e| return @errorCast(e);
            return;
        }
    }

    var prefix_coords: [8]usize = .{0} ** 8;
    var c_coords: [8]usize = .{0} ** 8;
    var a_coords: [8]usize = .{0} ** 8;
    var b_coords: [8]usize = .{0} ** 8;

    var prefix_idx: usize = 0;
    while (prefix_idx < prefix_total) : (prefix_idx += 1) {
        var x: usize = prefix_idx;
        var rr: usize = 0;
        while (rr < prefix_rank) : (rr += 1) {
            const d_rev: usize = prefix_rank - 1 - rr;
            const tc: usize = c_meta.tile_counts[d_rev];
            prefix_coords[d_rev] = x % tc;
            x /= tc;
        }

        var ti_n: usize = 0;
        while (ti_n < n_tiles) : (ti_n += 1) {
            var ti_k: usize = 0;
            while (ti_k < k_tiles) : (ti_k += 1) {
                var pd: usize = 0;
                while (pd < prefix_rank) : (pd += 1) {
                    b_coords[pd] = if (b_meta.shape[pd] == 1) 0 else prefix_coords[pd];
                }
                b_coords[m_dim] = ti_k;
                b_coords[n_dim] = ti_n;

                const b_idx: usize = try tensor_store.encodeTileIndex(b_meta, b_coords[0..rank]);
                const b_tile = try store.acquireTileConstLinear(s.b, b_idx);
                defer store.releaseConst(b_tile.token);
                const b_view = b_tile.bufferView();

                const k_tile: usize = b_view.layout.shape[m_dim];
                const n_tile_b: usize = b_view.layout.shape[n_dim];
                const mk: matmul_registry.F32Kernels = matmul_registry.selectForTile(ctx.matmul_f32, k_tile, n_tile_b) orelse ctx.matmul_f32;

                const scratch_buf: []align(32) u8 = if (ctx.matmul_scratch.len != 0) ctx.matmul_scratch[0] else blk: {
                    const scratch_need: usize = @max(matmul_registry.maxScratchBytes(), quant_matmul_registry.maxScratchBytes());
                    const tmp = ctx.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), scratch_need) catch return BackendError.ExecutionFailed;
                    break :blk tmp;
                };
                defer if (ctx.matmul_scratch.len == 0) ctx.allocator.free(scratch_buf);

                const pb_f32_len: usize = mk.tuning.kc * mk.tuning.nc;
                const pb_bytes_len: usize = pb_f32_len * @sizeOf(f32);
                if (scratch_buf.len < pb_bytes_len) return BackendError.InvalidArgument;
                const packed_b_mut: []align(32) f32 = @alignCast(std.mem.bytesAsSlice(f32, scratch_buf[0..pb_bytes_len]));
                try mk.pack_b_tile_f16_to_packed_f32(packed_b_mut, k_tile, n_tile_b, b_view.bytes);

                var ti_m: usize = 0;
                while (ti_m < m_tiles) : (ti_m += 1) {
                    var qd: usize = 0;
                    while (qd < prefix_rank) : (qd += 1) {
                        c_coords[qd] = prefix_coords[qd];
                        a_coords[qd] = if (a_meta.shape[qd] == 1) 0 else prefix_coords[qd];
                    }
                    c_coords[m_dim] = ti_m;
                    c_coords[n_dim] = ti_n;
                    a_coords[m_dim] = ti_m;
                    a_coords[n_dim] = ti_k;

                    const c_idx: usize = try tensor_store.encodeTileIndex(c_meta, c_coords[0..rank]);
                    const a_idx: usize = try tensor_store.encodeTileIndex(a_meta, a_coords[0..rank]);

                    var c_tile = try store.acquireTileMutLinear(s.c, c_idx);
                    defer store.releaseMut(c_tile.token);
                    const c_view = c_tile.bufferView();

                    const a_tile = try store.acquireTileConstLinear(s.a, a_idx);
                    defer store.releaseConst(a_tile.token);
                    const a_view = a_tile.bufferView();

                    const m_tile: usize = c_view.layout.shape[m_dim];
                    const n_tile: usize = c_view.layout.shape[n_dim];
                    const k_tile_m: usize = a_view.layout.shape[n_dim];
                    const beta_tile: f32 = if (ti_k == 0) s.beta else 1.0;
                    const params: MatMulParams = .{ .m = m_tile, .n = n_tile, .k = k_tile_m, .alpha = s.alpha, .beta = beta_tile };

                    try runF16TilePackedB(mk, scratch_buf, packed_b_mut, params, c_dtype, c_view.bytes, a_view.bytes);
                }
            }
        }
    }
}
