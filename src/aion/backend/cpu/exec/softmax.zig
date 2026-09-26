// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");

const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");

const exec_utils = @import("utils.zig");
const simd = @import("../kernels/simd.zig");
const softmax_kernels = @import("../kernels/softmax.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

const MAX_RANK: usize = 8;

/// Both strided max kernels share one signature, so the dtype pick is a pointer
/// selection rather than a branch duplicated around each call.
const UpdateMaxStridedFn = *const fn ([]f32, []const u8, usize, usize, []const usize) BackendError!void;

fn normalizeAxis(axis: i32, rank: usize) ExecuteProgramError!usize {
    if (rank == 0) return BackendError.InvalidArgument;
    const r_i32: i32 = @intCast(rank);
    var ax: i32 = axis;
    if (ax < 0) ax += r_i32;
    if (ax < 0 or ax >= r_i32) return BackendError.InvalidArgument;
    return @intCast(ax);
}

fn productUsize(vals: []const usize) ExecuteProgramError!usize {
    if (vals.len == 0) return BackendError.InvalidArgument;
    var acc: usize = 1;
    for (vals) |v| {
        acc = std.math.mul(usize, acc, v) catch return BackendError.InvalidArgument;
    }
    return acc;
}

pub fn execSoftmax(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepSoftmax,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    const out_meta = try store.meta(s.out);
    if (out_meta.dtype != .f32 and out_meta.dtype != .f16) return BackendError.InvalidArgument;
    if ((try store.meta(s.a)).dtype != out_meta.dtype) return BackendError.InvalidArgument;
    const rank: usize = @as(usize, out_meta.rank);
    if (rank == 0 or rank > MAX_RANK) return BackendError.InvalidArgument;
    return softmaxWhole(pool, thread_count, store, out_meta, s.out, s.a, try normalizeAxis(s.axis, rank));
}

/// Rows one pass of `softmaxWhole` holds maxima and sums for.
const WHOLE_ROW_BLOCK: usize = 256;

/// The input is `[outer, axis, inner]` in one flat run, and each of its
/// `outer * inner` rows splits across threads directly. With `inner == 1`
/// a row is contiguous; otherwise its elements are `inner` apart.
fn softmaxWhole(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    store: tensor_store.TensorStore,
    meta: tensor_store.TensorMeta,
    out: tensor_store.TensorId,
    a: tensor_store.TensorId,
    axis: usize,
) ExecuteProgramError!void {
    const rank: usize = meta.rank;
    const axis_len = meta.shape[axis];
    if (axis_len == 0) return BackendError.InvalidArgument;
    var inner: usize = 1;
    for (meta.shape[axis + 1 .. rank]) |d| inner *= d;
    var outer: usize = 1;
    for (meta.shape[0..axis]) |d| outer *= d;

    const out_view = try store.acquireMut(out);
    defer store.releaseMut(out_view.token);
    const a_view = try store.acquireConst(a);
    defer store.releaseConst(a_view.token);

    const Ctx = struct {
        f16: bool,
        axis_len: usize,
        inner: usize,
        out: []u8,
        a: []const u8,

        fn run(c: @This(), lo: usize, hi: usize, _: usize) BackendError!void {
            const eb: usize = if (c.f16) 2 else 4;
            var max: [WHOLE_ROW_BLOCK]f32 = undefined;
            var sum: [WHOLE_ROW_BLOCK]f32 = undefined;
            var offsets: [WHOLE_ROW_BLOCK]usize = undefined;
            var r0 = lo;
            while (r0 < hi) : (r0 += WHOLE_ROW_BLOCK) {
                const n = @min(WHOLE_ROW_BLOCK, hi - r0);
                @memset(max[0..n], -std.math.inf(f32));
                @memset(sum[0..n], 0.0);
                if (c.inner == 1) {
                    const row_bytes = c.axis_len * eb;
                    const shape = [2]usize{ n, c.axis_len };
                    const strides = [2]isize{ @intCast(row_bytes), @intCast(eb) };
                    const layout: types.Layout = .{ .rank = 2, .shape = shape[0..2], .strides_bytes = strides[0..2] };
                    const in: types.BufferViewConst = .{ .bytes = c.a[r0 * row_bytes ..][0 .. n * row_bytes], .dtype = if (c.f16) .f16 else .f32, .layout = layout };
                    const o: types.BufferViewMut = .{ .bytes = c.out[r0 * row_bytes ..][0 .. n * row_bytes], .dtype = in.dtype, .layout = layout };
                    if (c.f16) {
                        softmax_kernels.updateMaxF16(max[0..n], in, 2);
                        softmax_kernels.sumExpF16(sum[0..n], in, max[0..n], 2);
                        softmax_kernels.expNormalizeStoreF16(o, in, max[0..n], sum[0..n], 2);
                    } else {
                        softmax_kernels.updateMaxF32(max[0..n], in, 2);
                        softmax_kernels.expSumStoreF32(sum[0..n], o, in, max[0..n], 2);
                        softmax_kernels.normalizeF32(o, sum[0..n], 2);
                    }
                    continue;
                }
                // Row `q` is outer index `q / inner`, inner index `q % inner`.
                for (0..n) |r| {
                    const q = r0 + r;
                    offsets[r] = ((q / c.inner) * c.axis_len * c.inner + q % c.inner) * eb;
                }
                const stride = c.inner * eb;
                const offs = offsets[0..n];
                if (c.f16) {
                    try softmax_kernels.updateMaxStridedRowsF16(max[0..n], c.a, c.axis_len, stride, offs);
                    try softmax_kernels.sumExpStridedRowsF16(sum[0..n], c.a, c.axis_len, stride, offs, max[0..n]);
                    try softmax_kernels.expNormalizeStoreStridedRowsF16(c.out, c.a, c.axis_len, stride, stride, offs, offs, max[0..n], sum[0..n]);
                } else {
                    try softmax_kernels.updateMaxStridedRowsF32(max[0..n], c.a, c.axis_len, stride, offs);
                    try softmax_kernels.expSumStoreStridedRowsF32(sum[0..n], c.out, c.a, c.axis_len, stride, stride, offs, offs, max[0..n]);
                    try softmax_kernels.normalizeStridedRowsF32(c.out, c.axis_len, stride, offs, sum[0..n]);
                }
            }
        }
    };
    const half = meta.dtype == .f16;
    const ctx: Ctx = .{ .f16 = half, .axis_len = axis_len, .inner = inner, .out = out_view.bufferView().bytes, .a = a_view.bufferView().bytes };
    // Three passes over the row: two reads and a read-write.
    const row_bytes = axis_len * @as(usize, if (half) 2 else 4);
    return exec_utils.parallelRange(BackendError, pool, thread_count, outer * inner, 3 * row_bytes, ctx, Ctx.run);
}
