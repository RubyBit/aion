const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");

const executable = @import("../../../runtime/executable.zig");
const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");

const matmul_k = @import("../kernels/matmul.zig");
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

                stop: std.atomic.Value(bool) = .init(false),
                err_mutex: std.Thread.Mutex = .{},
                err_any: ?anyerror = null,

                fn fail(t: *@This(), err: anyerror) void {
                    if (t.stop.swap(true, .acq_rel)) return;
                    t.err_mutex.lock();
                    defer t.err_mutex.unlock();
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
                            if (is_matvec and t.b_dtype != .f16) {
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
                                    .q4_0, .q8_0 => {
                                        const qk: quant_matmul_registry.QuantKernels = if (k_tile <= t.matmul_qx0.tuning.kc and n_tile <= t.matmul_qx0.tuning.nc)
                                            t.matmul_qx0
                                        else
                                            (quant_matmul_registry.selectForTile(k_tile, n_tile) orelse {
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
                                        (quant_matmul_registry.selectForTile(k_tile, n_tile) orelse {
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
                                        const n_tile: usize = c_view0.layout.shape[1];

                                        const a_tile = t.store.acquireTileConst(t.s.a, ti_m, ti_k) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                        defer t.store.releaseConst(a_tile.token);
                                        const a_view = a_tile.bufferView();

                                        const k_tile: usize = a_view.layout.shape[1];
                                        const params: MatMulParams = .{ .m = m_tile, .n = n_tile, .k = k_tile, .alpha = t.s.alpha, .beta = beta_tile };

                                        if (t.c_dtype == .f32) {
                                            matmul_k.matmulF16ToF32(params, c_view0.bytes, a_view.bytes, b_view.bytes) catch |e| {
                                                t.fail(e);
                                                return;
                                            };
                                        } else {
                                            matmul_k.matmulF16(params, c_view0.bytes, a_view.bytes, b_view.bytes) catch |e| {
                                                t.fail(e);
                                                return;
                                            };
                                        }
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
                            if (k_tile > ctx.matmul_f32.tuning.kc or n_tile > ctx.matmul_f32.tuning.nc) return BackendError.InvalidArgument;

                            var scratch_buf: []align(32) u8 = if (ctx.matmul_scratch.len != 0) ctx.matmul_scratch[0] else blk: {
                                const tmp = ctx.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), ctx.matmul_f32.scratch_bytes) catch return BackendError.ExecutionFailed;
                                break :blk tmp;
                            };
                            defer if (ctx.matmul_scratch.len == 0) ctx.allocator.free(scratch_buf);

                            try ctx.matmul_f32.pack_b_tile(scratch_buf, k_tile, n_tile, b_view.bytes);
                            const pb_f32_len: usize = ctx.matmul_f32.tuning.kc * ctx.matmul_f32.tuning.nc;
                            const pb_bytes_len: usize = pb_f32_len * @sizeOf(f32);
                            const packed_b_view: []align(32) const f32 = @alignCast(std.mem.bytesAsSlice(f32, scratch_buf[0..pb_bytes_len]));
                            try ctx.matmul_f32.matmul_packed_b(scratch_buf, packed_b_view, params, c_view.bytes, a_view.bytes);
                        }
                    },
                    .f16 => {
                        if (c_dtype == .f32) {
                            try matmul_k.matmulF16ToF32(params, c_view.bytes, a_view.bytes, b_view.bytes);
                        } else {
                            try matmul_k.matmulF16(params, c_view.bytes, a_view.bytes, b_view.bytes);
                        }
                    },
                    .q4_0 => {
                        const qk: quant_matmul_registry.QuantKernels = if (k_tile <= ctx.matmul_qx0.tuning.kc and n_tile <= ctx.matmul_qx0.tuning.nc)
                            ctx.matmul_qx0
                        else
                            (quant_matmul_registry.selectForTile(k_tile, n_tile) orelse return BackendError.InvalidArgument);

                        var scratch_buf: []align(32) u8 = if (ctx.matmul_scratch.len != 0) ctx.matmul_scratch[0] else blk: {
                            const scratch_need: usize = @max(ctx.matmul_f32.scratch_bytes, quant_matmul_registry.maxScratchBytes());
                            const tmp = ctx.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), scratch_need) catch return BackendError.ExecutionFailed;
                            break :blk tmp;
                        };
                        defer if (ctx.matmul_scratch.len == 0) ctx.allocator.free(scratch_buf);

                        try qk.pack_b_tile_q4_0(scratch_buf, k_tile, n_tile, b_view.bytes);
                        const packed_b_view: quant_matmul_registry.PackedBView = @alignCast(scratch_buf[0..qk.packed_b_bytes]);
                        try qk.matmul_packed_b(scratch_buf, packed_b_view, params, c_view.bytes, a_view.bytes);
                    },
                    .q8_0 => {
                        const qk: quant_matmul_registry.QuantKernels = if (k_tile <= ctx.matmul_qx0.tuning.kc and n_tile <= ctx.matmul_qx0.tuning.nc)
                            ctx.matmul_qx0
                        else
                            (quant_matmul_registry.selectForTile(k_tile, n_tile) orelse return BackendError.InvalidArgument);

                        var scratch_buf: []align(32) u8 = if (ctx.matmul_scratch.len != 0) ctx.matmul_scratch[0] else blk: {
                            const scratch_need: usize = @max(ctx.matmul_f32.scratch_bytes, quant_matmul_registry.maxScratchBytes());
                            const tmp = ctx.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), scratch_need) catch return BackendError.ExecutionFailed;
                            break :blk tmp;
                        };
                        defer if (ctx.matmul_scratch.len == 0) ctx.allocator.free(scratch_buf);

                        try qk.pack_b_tile_q8_0(scratch_buf, k_tile, n_tile, b_view.bytes);
                        const packed_b_view: quant_matmul_registry.PackedBView = @alignCast(scratch_buf[0..qk.packed_b_bytes]);
                        try qk.matmul_packed_b(scratch_buf, packed_b_view, params, c_view.bytes, a_view.bytes);
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

                stop: std.atomic.Value(bool) = .init(false),
                err_mutex: std.Thread.Mutex = .{},
                err_any: ?anyerror = null,

                fn fail(t: *@This(), err: anyerror) void {
                    if (t.stop.swap(true, .acq_rel)) return;
                    t.err_mutex.lock();
                    defer t.err_mutex.unlock();
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
                                        if (k_tile > t.matmul_f32.tuning.kc or n_tile > t.matmul_f32.tuning.nc) {
                                            t.fail(BackendError.InvalidArgument);
                                            return;
                                        }
                                        t.matmul_f32.pack_b_tile(t.scratch[tid], k_tile, n_tile, b_view.bytes) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
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
                                    if (t.c_dtype == .f32) {
                                        matmul_k.matmulF16ToF32(params, c_view0.bytes, a_view.bytes, b_view.bytes) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                    } else {
                                        matmul_k.matmulF16(params, c_view0.bytes, a_view.bytes, b_view.bytes) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                    }
                                },
                                .q4_0, .q8_0 => {
                                    const qk: quant_matmul_registry.QuantKernels = if (k_tile <= t.matmul_qx0.tuning.kc and n_tile <= t.matmul_qx0.tuning.nc)
                                        t.matmul_qx0
                                    else
                                        (quant_matmul_registry.selectForTile(k_tile, n_tile) orelse {
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
                        if (k_tile > ctx.matmul_f32.tuning.kc or n_tile > ctx.matmul_f32.tuning.nc) return BackendError.InvalidArgument;

                        var scratch_buf: []align(32) u8 = if (ctx.matmul_scratch.len != 0) ctx.matmul_scratch[0] else blk: {
                            const tmp = ctx.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), ctx.matmul_f32.scratch_bytes) catch return BackendError.ExecutionFailed;
                            break :blk tmp;
                        };
                        defer if (ctx.matmul_scratch.len == 0) ctx.allocator.free(scratch_buf);

                        try ctx.matmul_f32.pack_b_tile(scratch_buf, k_tile, n_tile, b_view.bytes);
                        const pb_f32_len: usize = ctx.matmul_f32.tuning.kc * ctx.matmul_f32.tuning.nc;
                        const pb_bytes_len: usize = pb_f32_len * @sizeOf(f32);
                        const packed_b_view: []align(32) const f32 = @alignCast(std.mem.bytesAsSlice(f32, scratch_buf[0..pb_bytes_len]));
                        try ctx.matmul_f32.matmul_packed_b(scratch_buf, packed_b_view, params, c_view0.bytes, a_view.bytes);
                    }
                },
                .f16 => {
                    if (c_dtype == .f32) {
                        try matmul_k.matmulF16ToF32(params, c_view0.bytes, a_view.bytes, b_view.bytes);
                    } else {
                        try matmul_k.matmulF16(params, c_view0.bytes, a_view.bytes, b_view.bytes);
                    }
                },
                .q4_0 => {
                    const qk: quant_matmul_registry.QuantKernels = if (k_tile <= ctx.matmul_qx0.tuning.kc and n_tile <= ctx.matmul_qx0.tuning.nc)
                        ctx.matmul_qx0
                    else
                        (quant_matmul_registry.selectForTile(k_tile, n_tile) orelse return BackendError.InvalidArgument);

                    var scratch_buf: []align(32) u8 = if (ctx.matmul_scratch.len != 0) ctx.matmul_scratch[0] else blk: {
                        const scratch_need: usize = @max(ctx.matmul_f32.scratch_bytes, quant_matmul_registry.maxScratchBytes());
                        const tmp = ctx.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), scratch_need) catch return BackendError.ExecutionFailed;
                        break :blk tmp;
                    };
                    defer if (ctx.matmul_scratch.len == 0) ctx.allocator.free(scratch_buf);

                    try qk.pack_b_tile_q4_0(scratch_buf, k_tile, n_tile, b_view.bytes);
                    const packed_b_view: quant_matmul_registry.PackedBView = @alignCast(scratch_buf[0..qk.packed_b_bytes]);
                    try qk.matmul_packed_b(scratch_buf, packed_b_view, params, c_view0.bytes, a_view.bytes);
                },
                .q8_0 => {
                    const qk: quant_matmul_registry.QuantKernels = if (k_tile <= ctx.matmul_qx0.tuning.kc and n_tile <= ctx.matmul_qx0.tuning.nc)
                        ctx.matmul_qx0
                    else
                        (quant_matmul_registry.selectForTile(k_tile, n_tile) orelse return BackendError.InvalidArgument);

                    var scratch_buf: []align(32) u8 = if (ctx.matmul_scratch.len != 0) ctx.matmul_scratch[0] else blk: {
                        const scratch_need: usize = @max(ctx.matmul_f32.scratch_bytes, quant_matmul_registry.maxScratchBytes());
                        const tmp = ctx.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), scratch_need) catch return BackendError.ExecutionFailed;
                        break :blk tmp;
                    };
                    defer if (ctx.matmul_scratch.len == 0) ctx.allocator.free(scratch_buf);

                    try qk.pack_b_tile_q8_0(scratch_buf, k_tile, n_tile, b_view.bytes);
                    const packed_b_view: quant_matmul_registry.PackedBView = @alignCast(scratch_buf[0..qk.packed_b_bytes]);
                    try qk.matmul_packed_b(scratch_buf, packed_b_view, params, c_view0.bytes, a_view.bytes);
                },
                else => return BackendError.Unsupported,
            }
        }
    }
}
