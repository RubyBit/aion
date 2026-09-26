// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");

const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");

const exec_utils = @import("utils.zig");
const layernorm_kernels = @import("../kernels/layernorm.zig");
const elemwise = @import("../kernels/elemwise.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

const Mode = layernorm_kernels.Mode;

const ROW_CHUNK_MAX: usize = 256;

fn productUsize(vals: []const usize) ExecuteProgramError!usize {
    if (vals.len == 0) return BackendError.InvalidArgument;
    var acc: usize = 1;
    for (vals) |v| {
        acc = std.math.mul(usize, acc, v) catch return BackendError.InvalidArgument;
    }
    return acc;
}

pub fn execLayerNorm(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepLayerNorm,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    return normWhole(pool, thread_count, .layernorm, s.out, s.x, s.gamma, s.beta, null, s.eps, store);
}

/// RMSNorm, with the optional residual added to each row right after it is
/// normalized. f32 addition commutes exactly, so the operand order the fusion pass
/// picked cannot change the result.
pub fn execRMSNorm(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepRMSNorm,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    return normWhole(pool, thread_count, .rmsnorm, s.out, s.x, s.gamma, s.beta, s.residual, s.eps, store);
}

/// The normalized dims are trailing, so each row is one contiguous run and rows
/// split across threads directly. A residual is added to each row
/// right after it is normalized, while the row is still in cache.
fn normWhole(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    mode: Mode,
    out: tensor_store.TensorId,
    x: tensor_store.TensorId,
    gamma: tensor_store.TensorId,
    beta: tensor_store.TensorId,
    residual: ?tensor_store.TensorId,
    eps: f32,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    if (!(eps > 0.0) or !std.math.isFinite(eps)) return BackendError.InvalidArgument;
    const out_meta = try store.meta(out);
    const g_meta = try store.meta(gamma);
    const dtype = out_meta.dtype;
    if (dtype != .f32 and dtype != .f16) return BackendError.InvalidArgument;
    inline for (.{ x, gamma, beta }) |id| if ((try store.meta(id)).dtype != dtype) return BackendError.InvalidArgument;
    if (residual) |r| if (dtype != .f32 or (try store.meta(r)).dtype != .f32) return BackendError.InvalidArgument;
    const norm_rank: usize = g_meta.rank;
    if (norm_rank == 0 or norm_rank > out_meta.rank) return BackendError.InvalidArgument;
    for (0..norm_rank) |d| if (g_meta.shape[d] != out_meta.shape[out_meta.rank - norm_rank + d]) return BackendError.InvalidArgument;
    const cols: usize = try productUsize(g_meta.shape[0..norm_rank]);
    const rows: usize = try productUsize(out_meta.shape[0..out_meta.rank]) / cols;

    const out_view = try store.acquireMut(out);
    defer store.releaseMut(out_view.token);
    const x_view = try store.acquireConst(x);
    defer store.releaseConst(x_view.token);
    const g_view = try store.acquireConst(gamma);
    defer store.releaseConst(g_view.token);
    const b_view = try store.acquireConst(beta);
    defer store.releaseConst(b_view.token);
    const res_view = if (residual) |r| try store.acquireConst(r) else null;
    defer if (res_view) |t| store.releaseConst(t.token);

    const Ctx = struct {
        mode: Mode,
        eps: f32,
        dtype: types.DType,
        cols: usize,
        out: []u8,
        x: []const u8,
        gamma: []const u8,
        beta: []const u8,
        residual: ?[]const u8,

        fn run(c: @This(), lo: usize, hi: usize, _: usize) BackendError!void {
            const eb = c.dtype.info().block_bytes;
            const row_bytes = c.cols * eb;
            const inv_denom: f32 = 1.0 / @as(f32, @floatFromInt(c.cols));
            const cols_mem = [1]usize{c.cols};
            const col_stride = [1]isize{@intCast(eb)};
            const vec: types.Layout = .{ .rank = 1, .shape = cols_mem[0..1], .strides_bytes = col_stride[0..1] };
            const gv: types.BufferViewConst = .{ .bytes = c.gamma, .dtype = c.dtype, .layout = vec };
            const bv: types.BufferViewConst = .{ .bytes = c.beta, .dtype = c.dtype, .layout = vec };

            var sum: [ROW_CHUNK_MAX]f32 = undefined;
            var sumsq: [ROW_CHUNK_MAX]f32 = undefined;
            var mean: [ROW_CHUNK_MAX]f32 = undefined;
            var inv: [ROW_CHUNK_MAX]f32 = undefined;
            var r0 = lo;
            while (r0 < hi) : (r0 += ROW_CHUNK_MAX) {
                const n = @min(ROW_CHUNK_MAX, hi - r0);
                const span = c.out[r0 * row_bytes ..][0 .. n * row_bytes];
                const shape = [2]usize{ n, c.cols };
                const strides = [2]isize{ @intCast(row_bytes), @intCast(eb) };
                const layout: types.Layout = .{ .rank = 2, .shape = shape[0..2], .strides_bytes = strides[0..2] };
                const xv: types.BufferViewConst = .{ .bytes = c.x[r0 * row_bytes ..][0 .. n * row_bytes], .dtype = c.dtype, .layout = layout };
                const ov: types.BufferViewMut = .{ .bytes = span, .dtype = c.dtype, .layout = layout };

                @memset(sum[0..n], 0.0);
                @memset(sumsq[0..n], 0.0);
                layernorm_kernels.accumulateStats(sum[0..n], sumsq[0..n], xv);
                for (0..n) |r| {
                    const mu = sum[r] * inv_denom;
                    const msq = sumsq[r] * inv_denom;
                    const v = if (c.mode == .layernorm) @max(@as(f32, 0.0), msq - mu * mu) else msq;
                    const d = v + c.eps;
                    if (!(d > 0.0) or !std.math.isFinite(d)) return BackendError.InvalidArgument;
                    mean[r] = mu;
                    inv[r] = 1.0 / std.math.sqrt(d);
                }
                layernorm_kernels.applyNorm(c.mode, mean[0..n], inv[0..n], ov, xv, gv, bv);
                if (c.residual) |res| try elemwise.elemwiseBinaryF32(.add, span, span, res[r0 * row_bytes ..][0 .. n * row_bytes], n * c.cols);
            }
        }
    };
    const ctx: Ctx = .{
        .mode = mode,
        .eps = eps,
        .dtype = dtype,
        .cols = cols,
        .out = out_view.bufferView().bytes,
        .x = x_view.bufferView().bytes,
        .gamma = g_view.bufferView().bytes,
        .beta = b_view.bufferView().bytes,
        .residual = if (res_view) |t| t.bufferView().bytes else null,
    };
    // Two passes over x and one over out: a row is three of its own size in traffic.
    return exec_utils.parallelRange(BackendError, pool, thread_count, rows, 3 * cols * dtype.info().block_bytes, ctx, Ctx.run);
}
