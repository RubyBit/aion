// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");
const exec_utils = @import("utils.zig");
const rope_k = @import("../kernels/rope.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

pub fn execRoPE1D(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepRoPE1D,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    const out_meta: tensor_store.TensorMeta = try store.meta(s.out);
    const x_meta: tensor_store.TensorMeta = try store.meta(s.x);
    const pos_meta: tensor_store.TensorMeta = try store.meta(s.positions);

    if (out_meta.rank != 4 or x_meta.rank != 4 or pos_meta.rank != 2) return BackendError.InvalidArgument;
    if (out_meta.dtype != x_meta.dtype) return BackendError.InvalidArgument;
    if (!(out_meta.dtype == .f16 or out_meta.dtype == .f32)) return BackendError.InvalidArgument;
    if (pos_meta.dtype != .i32) return BackendError.InvalidArgument;

    var d: usize = 0;
    while (d < 4) : (d += 1) {
        if (out_meta.shape[d] != x_meta.shape[d]) return BackendError.InvalidArgument;
    }

    if (pos_meta.shape[0] != out_meta.shape[0] or pos_meta.shape[1] != out_meta.shape[1]) return BackendError.InvalidArgument;

    if (!(s.base_frequency > 0.0) or !std.math.isFinite(s.base_frequency)) return BackendError.InvalidArgument;
    if (!(s.scale_factor > 0.0) or !std.math.isFinite(s.scale_factor)) return BackendError.InvalidArgument;
    if (!std.math.isFinite(s.rope_proportion) or s.rope_proportion < 0.0 or s.rope_proportion > 1.0) return BackendError.InvalidArgument;

    const head_dim: usize = out_meta.shape[3];

    const pairs_total: usize = head_dim / 2;
    const rope_pairs_f: f32 = @floor(s.rope_proportion * @as(f32, @floatFromInt(pairs_total)));
    const rope_pairs_i: i64 = @intFromFloat(rope_pairs_f);
    if (rope_pairs_i < 0) return BackendError.InvalidArgument;
    var rope_pairs: usize = @intCast(rope_pairs_i);
    if (rope_pairs > pairs_total) rope_pairs = pairs_total;

    const head_dim_f32: f32 = @floatFromInt(head_dim);
    const freq_step: f32 = @floatCast(std.math.pow(f64, @as(f64, s.base_frequency), @as(f64, -2.0 / head_dim_f32)));

    return ropeWhole(pool, thread_count, s, store, pairs_total, rope_pairs, freq_step);
}

/// `x [B, T, H, D]` and `positions [B, T]` split together along the outermost of
/// B and T longer than one.
fn ropeWhole(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepRoPE1D,
    store: tensor_store.TensorStore,
    pairs_total: usize,
    rope_pairs: usize,
    freq_step: f32,
) ExecuteProgramError!void {
    const out_view = try store.acquireMut(s.out);
    defer store.releaseMut(out_view.token);
    const x_view = try store.acquireConst(s.x);
    defer store.releaseConst(x_view.token);
    const pos_view = try store.acquireConst(s.positions);
    defer store.releaseConst(pos_view.token);
    const Ctx = struct {
        out: types.BufferViewMut,
        x: types.BufferViewConst,
        pos: types.BufferViewConst,
        ax: usize,
        pairs_total: usize,
        rope_pairs: usize,
        freq_step: f32,
        scale_factor: f32,

        fn run(c: @This(), lo: usize, hi: usize, _: usize) BackendError!void {
            var so: [8]usize = undefined;
            var sx: [8]usize = undefined;
            var sp: [8]usize = undefined;
            const out = exec_utils.sliceAxis(c.out, c.ax, lo, hi, &so);
            const x = exec_utils.sliceAxis(c.x, c.ax, lo, hi, &sx);
            const pos = exec_utils.sliceAxis(c.pos, c.ax, lo, hi, &sp);
            return switch (out.dtype) {
                .f32 => rope_k.runF32(out, x, pos, c.pairs_total, c.rope_pairs, c.freq_step, c.scale_factor),
                .f16 => rope_k.runF16(out, x, pos, c.pairs_total, c.rope_pairs, c.freq_step, c.scale_factor),
                else => BackendError.InvalidArgument,
            };
        }
    };
    const out = out_view.bufferView();
    const shape = out.layout.shape;
    const ax: usize = if (shape[0] > 1) 0 else 1;
    const ctx: Ctx = .{
        .out = out,
        .x = x_view.bufferView(),
        .pos = pos_view.bufferView(),
        .ax = ax,
        .pairs_total = pairs_total,
        .rope_pairs = rope_pairs,
        .freq_step = freq_step,
        .scale_factor = s.scale_factor,
    };
    const row_bytes = @as(usize, @intCast(out.layout.strides_bytes[ax]));
    return exec_utils.parallelRange(BackendError, pool, thread_count, shape[ax], row_bytes, ctx, Ctx.run);
}
