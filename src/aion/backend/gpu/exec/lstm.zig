// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! LSTM cell execution for the GPU backend. An f32 cell runs its gates as two
//! matmuls into pooled scratch (a matvec at one row, `Matmul.execBound`), then the
//! update (kernels/lstm_gates.wgsl). An f16 cell, or one whose gates outgrow a
//! binding, runs fused (kernels/lstm.wgsl): one thread per (batch, hidden) element
//! computes all four gates and writes the [h_t | c_t] state row, in f32 whatever
//! the storage. Every operand must be one buffer and share the cell's dtype.
//! Both paths use the CPU cell's sigmoid/tanh approximations (kernels/lstm_act.wgsl).

const std = @import("std");
const wgpu = @import("../wgpu.zig");
const pipelines = @import("../pipelines.zig");
const context = @import("../context.zig");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const matmul_exec = @import("matmul.zig");
const executable = @import("../../../runtime/executable.zig");
const device_store = @import("../../../runtime/device_store.zig");

const c = wgpu.c;
const Ctx = context.Ctx;
const Frame = @import("../frame.zig").Frame;
const ExecuteProgramError = backend_mod.ExecuteProgramError;
const KernelDesc = pipelines.KernelDesc;

/// Scalar size of the cell's storage dtype. Only `.f32` and `.f16` reach here.
fn elemBytes(dtype: types.DType) usize {
    return if (dtype == .f16) 2 else 4;
}

const act_wgsl = @embedFile("../kernels/lstm_act.wgsl");
const lstm_kernel: KernelDesc = .{ .name = "lstm", .wgsl = @embedFile("../kernels/lstm.wgsl") ++ act_wgsl };
const gates_kernel: KernelDesc = .{ .name = "lstm_gates", .wgsl = @embedFile("../kernels/lstm_gates.wgsl") ++ act_wgsl };

const WG_1D: u32 = 64;

/// Field order matches `struct Params` in lstm.wgsl.
const LstmParams = extern struct {
    batch: u32,
    input_size: u32,
    hidden: u32,
    has_bias: u32,
    total: u32,
    _p0: u32 = 0,
    _p1: u32 = 0,
    _p2: u32 = 0,
};

/// Acquire tensor `id`'s one buffer, of `dtype` and at least `min_elems` elements
/// (Unsupported otherwise). Caller releases via `hs.releaseConst(t.token)`.
fn acquirePackedConst(ctx: Ctx, id: executable.TensorId, min_elems: usize, dtype: types.DType) ExecuteProgramError!device_store.Chunk {
    const hs = ctx.store;
    const meta = hs.meta(id) catch return error.ExecutionFailed;
    // Every operand shares the cell's dtype — `infer` requires it, so a mismatch
    // here is a compiler bug rather than input.
    if (meta.dtype != dtype or meta.chunks != 1) return error.Unsupported;
    var n: usize = 1;
    for (meta.shape) |d| n *= d;
    if (n < min_elems) return error.Unsupported;
    const t = ctx.store.acquireConst(id) catch return error.ExecutionFailed;
    errdefer hs.releaseConst(t.token);
    if (!context.storageBindingFits(ctx, t.len)) return error.Unsupported;
    return t;
}

pub fn execLSTMCell(ctx: Ctx, frame: *Frame, s: executable.StepLSTMCellFused, mm: *matmul_exec.Matmul) ExecuteProgramError!void {
    const hs = ctx.store;
    const out_meta = hs.meta(s.out_state) catch return error.ExecutionFailed;
    const x_meta = hs.meta(s.x) catch return error.ExecutionFailed;
    const h_meta = hs.meta(s.h_prev) catch return error.ExecutionFailed;

    // f32 and f16 both run the cell in f32 and differ only in storage (lstm.wgsl).
    const dt = out_meta.dtype;
    if ((dt != .f32 and dt != .f16) or out_meta.rank != 2 or x_meta.rank != 2 or h_meta.rank != 2) return error.Unsupported;
    if (out_meta.chunks != 1) return error.Unsupported;

    const batch = x_meta.shape[0];
    const input_size = x_meta.shape[1];
    const hidden = h_meta.shape[1];
    if (batch == 0 or input_size == 0 or hidden == 0) return error.Unsupported;
    const gate_dim = hidden * 4;

    const has_bias = s.b_ih != null;
    if (has_bias != (s.b_hh != null)) return error.Unsupported;

    const dx = try acquirePackedConst(ctx, s.x, batch * input_size, dt);
    defer hs.releaseConst(dx.token);
    const dh = try acquirePackedConst(ctx, s.h_prev, batch * hidden, dt);
    defer hs.releaseConst(dh.token);
    const dc = try acquirePackedConst(ctx, s.c_prev, batch * hidden, dt);
    defer hs.releaseConst(dc.token);
    const dwih = try acquirePackedConst(ctx, s.w_ih, input_size * gate_dim, dt);
    defer hs.releaseConst(dwih.token);
    const dwhh = try acquirePackedConst(ctx, s.w_hh, hidden * gate_dim, dt);
    defer hs.releaseConst(dwhh.token);

    var dbih: ?device_store.Chunk = null;
    var dbhh: ?device_store.Chunk = null;
    defer {
        if (dbih) |t| hs.releaseConst(t.token);
        if (dbhh) |t| hs.releaseConst(t.token);
    }
    if (has_bias) {
        dbih = try acquirePackedConst(ctx, s.b_ih.?, gate_dim, dt);
        dbhh = try acquirePackedConst(ctx, s.b_hh.?, gate_dim, dt);
    }

    const dout = ctx.store.acquireMut(s.out_state) catch return error.ExecutionFailed;
    defer hs.releaseMut(dout.token);
    if (out_meta.shape[0] * out_meta.shape[1] < batch * hidden * 2) return error.Unsupported;

    const total = std.math.cast(u32, batch * hidden) orelse return error.Unsupported;
    // Dummy bias bindings when absent (never read: has_bias == 0).
    const bih_buf = if (dbih) |t| ctx.devmem.bufferFor(t.handle).? else ctx.devmem.bufferFor(dwih.handle).?;
    const bih_len = if (dbih) |t| t.len else dwih.len;
    const bhh_buf = if (dbhh) |t| ctx.devmem.bufferFor(t.handle).? else ctx.devmem.bufferFor(dwih.handle).?;
    const bhh_len = if (dbhh) |t| t.len else dwih.len;
    const params: LstmParams = .{
        .batch = @intCast(batch),
        .input_size = @intCast(input_size),
        .hidden = @intCast(hidden),
        .has_bias = @intFromBool(has_bias),
        .total = total,
    };
    const groups = @max(1, @min(context.ceilDiv(total, WG_1D), context.MAX_GROUPS_1D));

    const gate_bytes = @as(u64, batch) * gate_dim * @sizeOf(f32);
    // The fused kernel's threads each walk four weight columns alone, so the GPU
    // runs a few hundred of them: matmuls spread the weights over all of it.
    if (dt == .f32 and context.storageBindingFits(ctx, gate_bytes)) {
        // gates = x @ w_ih, gates += h_prev @ w_hh, then the update.
        const gates = try ctx.scratch.ensure(ctx.gpu, gate_bytes);
        const n: u32 = @intCast(gate_dim);
        const m: u32 = @intCast(batch);
        try mm.execBound(ctx, frame, .{ .a = ctx.devmem.bufferFor(dx.handle).?, .a_len = dx.len, .b = ctx.devmem.bufferFor(dwih.handle).?, .b_len = dwih.len, .c = gates, .c_len = gate_bytes, .m = m, .n = n, .k = params.input_size, .c_row = n, .c_off = 0 });
        try mm.execBound(ctx, frame, .{ .a = ctx.devmem.bufferFor(dh.handle).?, .a_len = dh.len, .b = ctx.devmem.bufferFor(dwhh.handle).?, .b_len = dwhh.len, .c = gates, .c_len = gate_bytes, .m = m, .n = n, .k = params.hidden, .c_row = n, .c_off = 0, .beta = 1.0 });
        const bufs = [_]c.WGPUBuffer{ gates, ctx.devmem.bufferFor(dc.handle).?, bih_buf, bhh_buf, ctx.devmem.bufferFor(dout.handle).? };
        const sizes = [_]u64{ gate_bytes, dc.len, bih_len, bhh_len, dout.len };
        return frame.recordCompute(try ctx.pipes.get(gates_kernel, "lstm_gates"), &bufs, &sizes, std.mem.asBytes(&params), .{ groups, 1, 1 });
    }

    const built = try ctx.pipes.get(lstm_kernel, if (dt == .f16) "lstm_cell_f16" else "lstm_cell");
    const bufs = [_]c.WGPUBuffer{
        ctx.devmem.bufferFor(dx.handle).?,
        ctx.devmem.bufferFor(dh.handle).?,
        ctx.devmem.bufferFor(dc.handle).?,
        ctx.devmem.bufferFor(dwih.handle).?,
        ctx.devmem.bufferFor(dwhh.handle).?,
        bih_buf,
        bhh_buf,
        ctx.devmem.bufferFor(dout.handle).?,
    };
    const sizes = [_]u64{ dx.len, dh.len, dc.len, dwih.len, dwhh.len, bih_len, bhh_len, dout.len };
    try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups, 1, 1 });
}
