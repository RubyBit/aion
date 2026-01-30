const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");

const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");

const exec_utils = @import("utils.zig");
const simd = @import("../kernels/simd.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

const Mode = enum { layernorm, rmsnorm };

pub fn execLayerNormTiled(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepLayerNormTiled,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    return execNormTiled(pool, thread_count, .layernorm, s.out, s.x, s.gamma, s.beta, s.eps, store);
}

pub fn execRMSNormTiled(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepRMSNormTiled,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    return execNormTiled(pool, thread_count, .rmsnorm, s.out, s.x, s.gamma, s.beta, s.eps, store);
}

fn execNormTiled(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    mode: Mode,
    out: tensor_store.TensorId,
    x: tensor_store.TensorId,
    gamma: tensor_store.TensorId,
    beta: tensor_store.TensorId,
    eps: f32,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    if (!(eps > 0.0) or !std.math.isFinite(eps)) return BackendError.InvalidArgument;
    const out_meta = try store.meta(out);
    if (out_meta.rank != 2) return BackendError.InvalidArgument;
    if (!(out_meta.dtype == .f32 or out_meta.dtype == .f16)) return BackendError.InvalidArgument;

    // v0 contract: keep tm small so we can allocate per-row stats on stack.
    const tm: usize = out_meta.tile_shape[0];
    if (tm == 0 or tm > 256) return BackendError.InvalidArgument;

    const ti0_total: usize = out_meta.tile_counts[0];
    const tile_bytes: usize = exec_utils.tileByteSize(out_meta);
    const min_total_bytes: usize = 256 * 1024;
    const tile_total: usize = out_meta.tile_counts[0] * out_meta.tile_counts[1];

    if (pool) |p| {
        if (exec_utils.shouldParallelTiles(thread_count, tile_total, tile_bytes, min_total_bytes) and ti0_total >= 2) {
            const Task = struct {
                store: tensor_store.TensorStore,
                out_meta: tensor_store.TensorMeta,
                mode: Mode,
                out: tensor_store.TensorId,
                x: tensor_store.TensorId,
                gamma: tensor_store.TensorId,
                beta: tensor_store.TensorId,
                eps: f32,

                stop: std.atomic.Value(bool) = .init(false),
                err_mutex: std.Thread.Mutex = .{},
                err_any: ?anyerror = null,

                fn fail(t: *@This(), err: anyerror) void {
                    if (t.stop.swap(true, .acq_rel)) return;
                    t.err_mutex.lock();
                    defer t.err_mutex.unlock();
                    if (t.err_any == null) t.err_any = err;
                }

                fn runTi0(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
                    _ = tid;
                    const t: *@This() = @ptrCast(@alignCast(ctx_any));
                    if (start >= end) return;
                    if (t.stop.load(.acquire)) return;
                    var ti0: usize = start;
                    while (ti0 < end) : (ti0 += 1) {
                        if (t.stop.load(.acquire)) return;
                        normTi0(t.store, t.out_meta, t.mode, t.out, t.x, t.gamma, t.beta, t.eps, ti0) catch |e| {
                            t.fail(e);
                            return;
                        };
                    }
                }
            };

            var task: Task = .{ .store = store, .out_meta = out_meta, .mode = mode, .out = out, .x = x, .gamma = gamma, .beta = beta, .eps = eps };
            p.parallelForAny(@ptrCast(&task), ti0_total, 1, Task.runTi0);
            if (task.err_any) |e| return @errorCast(e);
            return;
        }
    }

    // Sequential fallback.
    var ti0: usize = 0;
    while (ti0 < ti0_total) : (ti0 += 1) {
        try normTi0(store, out_meta, mode, out, x, gamma, beta, eps, ti0);
    }
}

fn normTi0(
    store: tensor_store.TensorStore,
    out_meta: tensor_store.TensorMeta,
    mode: Mode,
    out: tensor_store.TensorId,
    x: tensor_store.TensorId,
    gamma: tensor_store.TensorId,
    beta: tensor_store.TensorId,
    eps: f32,
    ti0: usize,
) ExecuteProgramError!void {
    // Stats are per-row within the output tile-row.
    const tm: usize = out_meta.tile_shape[0];
    const tc1: usize = out_meta.tile_counts[1];

    var sum: [256]f32 = undefined;
    var sumsq: [256]f32 = undefined;
    @memset(sum[0..tm], 0.0);
    @memset(sumsq[0..tm], 0.0);

    const total_cols: usize = out_meta.shape[1];
    if (total_cols == 0) return BackendError.InvalidArgument;
    const denom: f32 = @as(f32, @floatFromInt(total_cols));
    const inv_denom: f32 = 1.0 / denom;

    // Pass 1: accumulate sum and sumsq per row.
    var ti1: usize = 0;
    while (ti1 < tc1) : (ti1 += 1) {
        if (ti1 + 1 < tc1) store.prefetch(x, ti0, ti1 + 1);
        const xt = try store.acquireTileConst(x, ti0, ti1);
        defer store.releaseConst(xt.token);
        const xv = xt.bufferView();
        if (xv.dtype != out_meta.dtype) return BackendError.InvalidArgument;
        if (xv.layout.rank != 2) return BackendError.InvalidArgument;

        const m_tile: usize = xv.layout.shape[0];
        const n_tile: usize = xv.layout.shape[1];
        const lanes: usize = comptime simd.lanesF32();
        const VecF = @Vector(lanes, f32);

        var r: usize = 0;
        while (r < m_tile) : (r += 1) {
            var acc_sum_v: VecF = @splat(@as(f32, 0.0));
            var acc_sq_v: VecF = @splat(@as(f32, 0.0));

            var c: usize = 0;
            const vec_end: usize = n_tile - (n_tile % lanes);

            if (xv.dtype == .f32) {
                const xs: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, xv.bytes);
                const off: usize = r * n_tile;
                while (c < vec_end) : (c += lanes) {
                    const v: VecF = @as(*align(1) const VecF, @ptrCast(xs.ptr + off + c)).*;
                    acc_sum_v += v;
                    acc_sq_v += v * v;
                }
                var acc_sum: f32 = @reduce(.Add, acc_sum_v);
                var acc_sq: f32 = @reduce(.Add, acc_sq_v);
                while (c < n_tile) : (c += 1) {
                    const v: f32 = xs[off + c];
                    acc_sum += v;
                    acc_sq += v * v;
                }
                sum[r] += acc_sum;
                sumsq[r] += acc_sq;
            } else {
                const xs: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, xv.bytes);
                const VecH = @Vector(lanes, f16);
                const off: usize = r * n_tile;
                while (c < vec_end) : (c += lanes) {
                    const vh: VecH = @as(*align(1) const VecH, @ptrCast(xs.ptr + off + c)).*;
                    const v: VecF = @floatCast(vh);
                    acc_sum_v += v;
                    acc_sq_v += v * v;
                }
                var acc_sum: f32 = @reduce(.Add, acc_sum_v);
                var acc_sq: f32 = @reduce(.Add, acc_sq_v);
                while (c < n_tile) : (c += 1) {
                    const v: f32 = @floatCast(xs[off + c]);
                    acc_sum += v;
                    acc_sq += v * v;
                }
                sum[r] += acc_sum;
                sumsq[r] += acc_sq;
            }
        }
    }

    // Zero out unused rows in edge tiles (last tile-row may be short).
    {
        const full_rows: usize = out_meta.shape[0];
        const start_row: usize = ti0 * out_meta.tile_shape[0];
        const valid_rows: usize = if (start_row >= full_rows) 0 else @min(tm, full_rows - start_row);
        if (valid_rows < tm) {
            @memset(sum[valid_rows..tm], 0.0);
            @memset(sumsq[valid_rows..tm], 0.0);
        }
    }

    // Compute mean and inv for each row.
    var mean: [256]f32 = undefined;
    var inv: [256]f32 = undefined;
    var r0: usize = 0;
    while (r0 < tm) : (r0 += 1) {
        const mu: f32 = sum[r0] * inv_denom;
        mean[r0] = mu;
        const msq: f32 = sumsq[r0] * inv_denom;
        const v: f32 = if (mode == .layernorm) @max(@as(f32, 0.0), msq - (mu * mu)) else msq;
        const d: f32 = v + eps;
        // Guard against pathological inputs (shouldn't happen for finite x and eps>0,
        // but keep the runtime robust).
        if (!(d > 0.0) or !std.math.isFinite(d)) return BackendError.InvalidArgument;
        inv[r0] = 1.0 / std.math.sqrt(d);
    }

    // Pass 2: apply normalization, gamma/beta.
    ti1 = 0;
    while (ti1 < tc1) : (ti1 += 1) {
        if (ti1 + 1 < tc1) {
            store.prefetch(x, ti0, ti1 + 1);
            store.prefetch(out, ti0, ti1 + 1);
            store.prefetch(gamma, ti1 + 1, 0);
            store.prefetch(beta, ti1 + 1, 0);
        }

        var out_t = try store.acquireTileMut(out, ti0, ti1);
        defer store.releaseMut(out_t.token);
        const xt = try store.acquireTileConst(x, ti0, ti1);
        defer store.releaseConst(xt.token);
        const gt = try store.acquireTileConst(gamma, ti1, 0);
        defer store.releaseConst(gt.token);
        const bt = try store.acquireTileConst(beta, ti1, 0);
        defer store.releaseConst(bt.token);

        const ov = out_t.bufferView();
        const xv = xt.bufferView();
        const gv = gt.bufferView();
        const bv = bt.bufferView();
        if (ov.dtype != out_meta.dtype or xv.dtype != out_meta.dtype or gv.dtype != out_meta.dtype or bv.dtype != out_meta.dtype) return BackendError.InvalidArgument;
        if (ov.layout.rank != 2 or xv.layout.rank != 2) return BackendError.InvalidArgument;
        if (gv.layout.rank != 1 or bv.layout.rank != 1) return BackendError.InvalidArgument;

        const m_tile: usize = ov.layout.shape[0];
        const n_tile: usize = ov.layout.shape[1];
        const lanes: usize = comptime simd.lanesF32();
        const VecF = @Vector(lanes, f32);

        if (ov.dtype == .f32) {
            var out_s: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, ov.bytes);
            const x_s: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, xv.bytes);
            const g_s: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, gv.bytes);
            const b_s: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, bv.bytes);

            var rr: usize = 0;
            while (rr < m_tile) : (rr += 1) {
                const mu: f32 = mean[rr];
                const inv0: f32 = inv[rr];
                const mu_v: VecF = @splat(mu);
                const inv_v: VecF = @splat(inv0);

                const off: usize = rr * n_tile;
                var c: usize = 0;
                const vec_end: usize = n_tile - (n_tile % lanes);
                while (c < vec_end) : (c += lanes) {
                    const xv0: VecF = @as(*align(1) const VecF, @ptrCast(x_s.ptr + off + c)).*;
                    const gv0: VecF = @as(*align(1) const VecF, @ptrCast(g_s.ptr + c)).*;
                    const bv0: VecF = @as(*align(1) const VecF, @ptrCast(b_s.ptr + c)).*;
                    const norm: VecF = if (mode == .layernorm) (xv0 - mu_v) * inv_v else xv0 * inv_v;
                    @as(*align(1) VecF, @ptrCast(out_s.ptr + off + c)).* = (norm * gv0) + bv0;
                }
                while (c < n_tile) : (c += 1) {
                    const x0: f32 = x_s[off + c];
                    const norm: f32 = if (mode == .layernorm) (x0 - mu) * inv0 else x0 * inv0;
                    out_s[off + c] = (norm * g_s[c]) + b_s[c];
                }
            }
        } else {
            var out_s: []align(1) f16 = simd.bytesAsSliceMutUnaligned(f16, ov.bytes);
            const x_s: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, xv.bytes);
            const g_s: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, gv.bytes);
            const b_s: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, bv.bytes);

            const VecH = @Vector(lanes, f16);

            var rr: usize = 0;
            while (rr < m_tile) : (rr += 1) {
                const mu: f32 = mean[rr];
                const inv0: f32 = inv[rr];
                const mu_v: VecF = @splat(mu);
                const inv_v: VecF = @splat(inv0);

                const off: usize = rr * n_tile;
                var c: usize = 0;
                const vec_end: usize = n_tile - (n_tile % lanes);
                while (c < vec_end) : (c += lanes) {
                    const xh: VecH = @as(*align(1) const VecH, @ptrCast(x_s.ptr + off + c)).*;
                    const gh: VecH = @as(*align(1) const VecH, @ptrCast(g_s.ptr + c)).*;
                    const bh: VecH = @as(*align(1) const VecH, @ptrCast(b_s.ptr + c)).*;

                    const xv0: VecF = @floatCast(xh);
                    const gv0: VecF = @floatCast(gh);
                    const bv0: VecF = @floatCast(bh);

                    const norm: VecF = if (mode == .layernorm) (xv0 - mu_v) * inv_v else xv0 * inv_v;
                    const y: VecF = (norm * gv0) + bv0;
                    @as(*align(1) VecH, @ptrCast(out_s.ptr + off + c)).* = @floatCast(y);
                }
                while (c < n_tile) : (c += 1) {
                    const x0: f32 = @floatCast(x_s[off + c]);
                    const norm: f32 = if (mode == .layernorm) (x0 - mu) * inv0 else x0 * inv0;
                    const y: f32 = (norm * @as(f32, @floatCast(g_s[c]))) + @as(f32, @floatCast(b_s[c]));
                    out_s[off + c] = @floatCast(y);
                }
            }
        }
    }
}
