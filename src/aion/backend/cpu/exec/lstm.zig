// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const fast_math = @import("../kernels/fast_math.zig");
const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

const MAX_HIDDEN: usize = 1024;
const MAX_INPUT: usize = 2048;

// Every operand is read into (and written from) f32 scratch, so the gate math
// below runs in f32 whatever the tensors are stored as. That is what makes the
// f16 cell bit-identical to the f32 one for the same widened inputs, and what
// keeps it in step with the GPU kernel — only these four helpers know the dtype.

/// Dispatch a dtype-generic helper over the scalar float types the cell accepts.
/// `else` is unreachable for a validated step; it stays a hard error rather than
/// a silent f32 read of f16 bytes.
fn floatDispatch(comptime name: []const u8, dtype: types.DType, args: anytype) ExecuteProgramError!void {
    return switch (dtype) {
        .f32 => @call(.auto, @field(@This(), name), .{f32} ++ args),
        .f16 => @call(.auto, @field(@This(), name), .{f16} ++ args),
        else => BackendError.InvalidArgument,
    };
}

fn loadRowAsF32(store: tensor_store.TensorStore, meta: tensor_store.TensorMeta, id: tensor_store.TensorId, row: usize, out: []f32) ExecuteProgramError!void {
    return floatDispatch("loadRowT", meta.dtype, .{ store, meta, id, row, out });
}

fn loadRowT(comptime T: type, store: tensor_store.TensorStore, meta: tensor_store.TensorMeta, id: tensor_store.TensorId, row: usize, out: []f32) ExecuteProgramError!void {
    if (meta.rank != 2) return BackendError.InvalidArgument;
    if (row >= meta.shape[0]) return BackendError.InvalidArgument;
    if (out.len != meta.shape[1]) return BackendError.InvalidArgument;
    const t = try store.acquireConst(id);
    defer store.releaseConst(t.token);
    const src = std.mem.bytesAsSlice(T, t.bytes)[row * out.len ..][0..out.len];
    for (out, src) |*d, v| d.* = @floatCast(v);
}

fn loadVecAsF32(store: tensor_store.TensorStore, meta: tensor_store.TensorMeta, id: tensor_store.TensorId, out: []f32) ExecuteProgramError!void {
    return floatDispatch("loadVecT", meta.dtype, .{ store, meta, id, out });
}

fn loadVecT(comptime T: type, store: tensor_store.TensorStore, meta: tensor_store.TensorMeta, id: tensor_store.TensorId, out: []f32) ExecuteProgramError!void {
    if (meta.rank != 1) return BackendError.InvalidArgument;
    if (out.len != meta.shape[0]) return BackendError.InvalidArgument;
    const t = try store.acquireConst(id);
    defer store.releaseConst(t.token);
    const src = std.mem.bytesAsSlice(T, t.bytes)[0..out.len];
    for (out, src) |*d, v| d.* = @floatCast(v);
}

fn storeRowState(
    store: tensor_store.TensorStore,
    meta: tensor_store.TensorMeta,
    id: tensor_store.TensorId,
    row: usize,
    h: []const f32,
    c: []const f32,
) ExecuteProgramError!void {
    return floatDispatch("storeRowStateT", meta.dtype, .{ store, meta, id, row, h, c });
}

/// Row `row` of the `[B, 2*H]` state is `h` followed by `c`.
fn storeRowStateT(
    comptime T: type,
    store: tensor_store.TensorStore,
    meta: tensor_store.TensorMeta,
    id: tensor_store.TensorId,
    row: usize,
    h: []const f32,
    c: []const f32,
) ExecuteProgramError!void {
    if (meta.rank != 2) return BackendError.InvalidArgument;
    if (row >= meta.shape[0]) return BackendError.InvalidArgument;
    if (meta.shape[1] != h.len + c.len) return BackendError.InvalidArgument;
    const t = try store.acquireMut(id);
    defer store.releaseMut(t.token);
    const dst = std.mem.bytesAsSlice(T, t.bytes)[row * meta.shape[1] ..][0..meta.shape[1]];
    for (dst[0..h.len], h) |*d, v| d.* = @floatCast(v);
    for (dst[h.len..], c) |*d, v| d.* = @floatCast(v);
}

/// `out += x @ W` for `W [rows, cols]`.
fn accumMatVec(
    store: tensor_store.TensorStore,
    w_meta: tensor_store.TensorMeta,
    w_id: tensor_store.TensorId,
    x: []const f32,
    out: []f32,
) ExecuteProgramError!void {
    return floatDispatch("accumMatVecT", w_meta.dtype, .{ store, w_meta, w_id, x, out });
}

fn accumMatVecT(
    comptime T: type,
    store: tensor_store.TensorStore,
    w_meta: tensor_store.TensorMeta,
    w_id: tensor_store.TensorId,
    x: []const f32,
    out: []f32,
) ExecuteProgramError!void {
    if (w_meta.rank != 2) return BackendError.InvalidArgument;
    if (x.len != w_meta.shape[0]) return BackendError.InvalidArgument;
    if (out.len != w_meta.shape[1]) return BackendError.InvalidArgument;
    const t = try store.acquireConst(w_id);
    defer store.releaseConst(t.token);
    const w = std.mem.bytesAsSlice(T, t.bytes);
    for (x, 0..) |xv, r| {
        if (xv == 0.0) continue;
        const w_row = w[r * out.len ..][0..out.len];
        for (out, w_row) |*o, wv| o.* += xv * @as(f32, @floatCast(wv));
    }
}

pub fn execLSTMCellFused(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepLSTMCellFused,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    _ = pool;
    _ = thread_count;

    const out_meta = try store.meta(s.out_state);
    const x_meta = try store.meta(s.x);
    const h_meta = try store.meta(s.h_prev);
    const c_meta = try store.meta(s.c_prev);
    const wih_meta = try store.meta(s.w_ih);
    const whh_meta = try store.meta(s.w_hh);

    // f32 and f16 storage; the cell itself always computes in f32 (see the
    // helpers above). `infer` already requires every operand to share this dtype.
    if (out_meta.dtype != .f32 and out_meta.dtype != .f16) return BackendError.Unsupported;
    if (x_meta.dtype != out_meta.dtype or h_meta.dtype != out_meta.dtype or c_meta.dtype != out_meta.dtype) return BackendError.InvalidArgument;
    if (wih_meta.dtype != out_meta.dtype or whh_meta.dtype != out_meta.dtype) return BackendError.InvalidArgument;

    if (out_meta.rank != 2 or x_meta.rank != 2 or h_meta.rank != 2 or c_meta.rank != 2) return BackendError.InvalidArgument;
    if (wih_meta.rank != 2 or whh_meta.rank != 2) return BackendError.InvalidArgument;

    const batch: usize = x_meta.shape[0];
    const input_size: usize = x_meta.shape[1];
    const hidden: usize = h_meta.shape[1];

    if (batch == 0 or input_size == 0 or hidden == 0) return BackendError.InvalidArgument;
    if (input_size > MAX_INPUT or hidden > MAX_HIDDEN) return BackendError.InvalidArgument;

    const gate_dim: usize = hidden * 4;
    if (wih_meta.shape[0] != input_size or wih_meta.shape[1] != gate_dim) return BackendError.InvalidArgument;
    if (whh_meta.shape[0] != hidden or whh_meta.shape[1] != gate_dim) return BackendError.InvalidArgument;

    if (h_meta.shape[0] != batch or c_meta.shape[0] != batch) return BackendError.InvalidArgument;
    if (c_meta.shape[1] != hidden) return BackendError.InvalidArgument;

    if (out_meta.shape[0] != batch or out_meta.shape[1] != hidden * 2) return BackendError.InvalidArgument;

    var b_ih_buf: [MAX_HIDDEN * 4]f32 = undefined;
    var b_hh_buf: [MAX_HIDDEN * 4]f32 = undefined;
    const has_bias: bool = (s.b_ih != null);
    if (has_bias != (s.b_hh != null)) return BackendError.InvalidArgument;

    if (has_bias) {
        const bih_id: tensor_store.TensorId = s.b_ih.?;
        const bhh_id: tensor_store.TensorId = s.b_hh.?;
        const bih_meta = try store.meta(bih_id);
        const bhh_meta = try store.meta(bhh_id);
        if (bih_meta.dtype != out_meta.dtype or bhh_meta.dtype != out_meta.dtype) return BackendError.InvalidArgument;
        if (bih_meta.rank != 1 or bhh_meta.rank != 1) return BackendError.InvalidArgument;
        if (bih_meta.shape[0] != gate_dim or bhh_meta.shape[0] != gate_dim) return BackendError.InvalidArgument;

        try loadVecAsF32(store, bih_meta, bih_id, b_ih_buf[0..gate_dim]);
        try loadVecAsF32(store, bhh_meta, bhh_id, b_hh_buf[0..gate_dim]);
    }

    var x_row_buf: [MAX_INPUT]f32 = undefined;
    var h_row_buf: [MAX_HIDDEN]f32 = undefined;
    var c_row_buf: [MAX_HIDDEN]f32 = undefined;
    var gates_buf: [MAX_HIDDEN * 4]f32 = undefined;
    var h_out_buf: [MAX_HIDDEN]f32 = undefined;
    var c_out_buf: [MAX_HIDDEN]f32 = undefined;

    var b: usize = 0;
    while (b < batch) : (b += 1) {
        try loadRowAsF32(store, x_meta, s.x, b, x_row_buf[0..input_size]);
        try loadRowAsF32(store, h_meta, s.h_prev, b, h_row_buf[0..hidden]);
        try loadRowAsF32(store, c_meta, s.c_prev, b, c_row_buf[0..hidden]);

        // gates = bias (optional)
        var j: usize = 0;
        while (j < gate_dim) : (j += 1) {
            gates_buf[j] = if (has_bias) (b_ih_buf[j] + b_hh_buf[j]) else 0.0;
        }

        // gates += x @ w_ih
        try accumMatVec(store, wih_meta, s.w_ih, x_row_buf[0..input_size], gates_buf[0..gate_dim]);
        // gates += h_prev @ w_hh
        try accumMatVec(store, whh_meta, s.w_hh, h_row_buf[0..hidden], gates_buf[0..gate_dim]);

        // Compute new state.
        const h_off0: usize = 0;
        const h_off1: usize = hidden;
        const h_off2: usize = hidden * 2;
        const h_off3: usize = hidden * 3;

        j = 0;
        while (j < hidden) : (j += 1) {
            const i_lin: f32 = gates_buf[h_off0 + j];
            const f_lin: f32 = gates_buf[h_off1 + j];
            const g_lin: f32 = gates_buf[h_off2 + j];
            const o_lin: f32 = gates_buf[h_off3 + j];

            const i_gate: f32 = fast_math.sigmoidApproxF32(i_lin);
            const f_gate: f32 = fast_math.sigmoidApproxF32(f_lin);
            const g_gate: f32 = fast_math.tanhApproxF32(g_lin);
            const o_gate: f32 = fast_math.sigmoidApproxF32(o_lin);

            const c_t: f32 = f_gate * c_row_buf[j] + i_gate * g_gate;
            const h_t: f32 = o_gate * fast_math.tanhApproxF32(c_t);

            c_out_buf[j] = c_t;
            h_out_buf[j] = h_t;
        }

        try storeRowState(store, out_meta, s.out_state, b, h_out_buf[0..hidden], c_out_buf[0..hidden]);
    }
}
