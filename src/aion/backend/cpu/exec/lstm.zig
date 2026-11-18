// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! The fused LSTM cell: `gates = x @ W_ih + h @ W_hh + b_ih + b_hh` for the whole
//! batch as two GEMMs through the f32 matmul (threaded, packed, SME where the CPU
//! has it), then the gate math per row across the pool.
//!
//! The cell computes in f32 whatever its tensors are stored as: f16 x and h are
//! widened before the GEMMs (f16 weights the matmul reads itself), and the gate
//! math reads and writes through f32. That keeps the f16 cell in step with the f32
//! one for the same widened inputs, and with the GPU kernels, which use the same
//! sigmoid/tanh approximations (gpu/kernels/lstm_act.wgsl).
const std = @import("std");

const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const fast_math = @import("../kernels/fast_math.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");
const exec_matmul = @import("matmul.zig");
const exec_utils = @import("utils.zig");

const BackendError = types.BackendError;
const DType = types.DType;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

/// Rough cost of one hidden unit's gate math (four sigmoid/tanh and the update)
/// in `parallelRange`'s multiply-add units.
const GATE_UNIT_COST: usize = 64;

pub fn execLSTMCellFused(mm: *exec_matmul.MatMulExecCtx, s: executable.StepLSTMCellFused, store: tensor_store.TensorStore) ExecuteProgramError!void {
    const out_meta = try store.meta(s.out_state);
    const x_meta = try store.meta(s.x);
    const h_meta = try store.meta(s.h_prev);
    const c_meta = try store.meta(s.c_prev);
    const wih_meta = try store.meta(s.w_ih);
    const whh_meta = try store.meta(s.w_hh);

    // `infer` already requires every operand to share this dtype.
    const dtype = out_meta.dtype;
    if (dtype != .f32 and dtype != .f16) return BackendError.Unsupported;
    for ([_]DType{ x_meta.dtype, h_meta.dtype, c_meta.dtype, wih_meta.dtype, whh_meta.dtype }) |d| {
        if (d != dtype) return BackendError.InvalidArgument;
    }
    for ([_]tensor_store.TensorMeta{ out_meta, x_meta, h_meta, c_meta, wih_meta, whh_meta }) |m| {
        if (m.rank != 2) return BackendError.InvalidArgument;
    }

    const batch: usize = x_meta.shape[0];
    const input_size: usize = x_meta.shape[1];
    const hidden: usize = h_meta.shape[1];
    const gate_dim: usize = hidden * 4;
    if (batch == 0 or input_size == 0 or hidden == 0) return BackendError.InvalidArgument;
    if (wih_meta.shape[0] != input_size or wih_meta.shape[1] != gate_dim) return BackendError.InvalidArgument;
    if (whh_meta.shape[0] != hidden or whh_meta.shape[1] != gate_dim) return BackendError.InvalidArgument;
    if (h_meta.shape[0] != batch or c_meta.shape[0] != batch or c_meta.shape[1] != hidden) return BackendError.InvalidArgument;
    if (out_meta.shape[0] != batch or out_meta.shape[1] != hidden * 2) return BackendError.InvalidArgument;
    if ((s.b_ih == null) != (s.b_hh == null)) return BackendError.InvalidArgument;

    const alloc = mm.allocator;
    const gates = alloc.alloc(f32, batch * gate_dim) catch return BackendError.ExecutionFailed;
    defer alloc.free(gates);
    const bias = alloc.alloc(f32, gate_dim) catch return BackendError.ExecutionFailed;
    defer alloc.free(bias);
    @memset(bias, 0.0);
    if (s.b_ih) |bih| {
        try accumVec(store, bih, gate_dim, bias);
        try accumVec(store, s.b_hh.?, gate_dim, bias);
    }

    try gemmInto(mm, store, dtype, s.x, s.w_ih, .{ .beta = 0.0 }, batch, gate_dim, input_size, gates);
    try gemmInto(mm, store, dtype, s.h_prev, s.w_hh, .{ .beta = 1.0 }, batch, gate_dim, hidden, gates);

    const c_prev = try store.acquireConst(s.c_prev);
    defer store.releaseConst(c_prev.token);
    const out = try store.acquireMut(s.out_state);
    defer store.releaseMut(out.token);
    const rows: Rows = .{ .gates = gates, .bias = bias, .hidden = hidden, .c_prev = c_prev.bytes, .out = out.bytes, .f16 = dtype == .f16 };
    return exec_utils.parallelRange(BackendError, mm.pool, mm.thread_count, batch, hidden * GATE_UNIT_COST, rows, Rows.run);
}

/// `gates = a @ w + scale.beta * gates` for a float `[m, k]` A and `[k, n]` W.
fn gemmInto(mm: *exec_matmul.MatMulExecCtx, store: tensor_store.TensorStore, dtype: DType, a_id: tensor_store.TensorId, w_id: tensor_store.TensorId, scale: exec_matmul.Scale, m: usize, n: usize, k: usize, gates: []f32) ExecuteProgramError!void {
    const a = try store.acquireConst(a_id);
    defer store.releaseConst(a.token);
    const w = try store.acquireConst(w_id);
    defer store.releaseConst(w.token);
    const a32 = try exec_matmul.widen(mm.allocator, dtype, @constCast(a.bytes));
    defer if (a32.ptr != a.bytes.ptr) exec_matmul.unwiden(mm.allocator, a32);
    try exec_matmul.gemm2D(mm, scale, m, n, k, std.mem.sliceAsBytes(gates), a32, w.bytes, dtype);
}

/// `out += v` for a float vector tensor of `n` elements.
fn accumVec(store: tensor_store.TensorStore, id: tensor_store.TensorId, n: usize, out: []f32) ExecuteProgramError!void {
    const meta = try store.meta(id);
    if (meta.rank != 1 or meta.shape[0] != n) return BackendError.InvalidArgument;
    const t = try store.acquireConst(id);
    defer store.releaseConst(t.token);
    switch (meta.dtype) {
        .f32 => for (out, std.mem.bytesAsSlice(f32, t.bytes)[0..n]) |*o, v| {
            o.* += v;
        },
        .f16 => for (out, std.mem.bytesAsSlice(f16, t.bytes)[0..n]) |*o, v| {
            o.* += v;
        },
        else => return BackendError.InvalidArgument,
    }
}

/// The gate math for a range of batch rows: row `b` of the `[B, 2H]` state is
/// `h` followed by `c`.
const Rows = struct {
    gates: []const f32,
    bias: []const f32,
    hidden: usize,
    c_prev: []const u8,
    out: []u8,
    f16: bool,

    fn run(r: Rows, lo: usize, hi: usize, _: usize) BackendError!void {
        if (r.f16) r.rowsT(f16, lo, hi) else r.rowsT(f32, lo, hi);
    }

    fn rowsT(r: Rows, comptime T: type, lo: usize, hi: usize) void {
        const hd = r.hidden;
        const c_prev = std.mem.bytesAsSlice(T, r.c_prev);
        const out = std.mem.bytesAsSlice(T, r.out);
        for (lo..hi) |b| {
            const g = r.gates[b * 4 * hd ..][0 .. 4 * hd];
            for (0..hd) |j| {
                const i_gate = fast_math.sigmoidApproxF32(g[j] + r.bias[j]);
                const f_gate = fast_math.sigmoidApproxF32(g[hd + j] + r.bias[hd + j]);
                const g_gate = fast_math.tanhApproxF32(g[2 * hd + j] + r.bias[2 * hd + j]);
                const o_gate = fast_math.sigmoidApproxF32(g[3 * hd + j] + r.bias[3 * hd + j]);
                const c_t = f_gate * @as(f32, @floatCast(c_prev[b * hd + j])) + i_gate * g_gate;
                out[b * 2 * hd + j] = @floatCast(o_gate * fast_math.tanhApproxF32(c_t));
                out[b * 2 * hd + hd + j] = @floatCast(c_t);
            }
        }
    }
};
