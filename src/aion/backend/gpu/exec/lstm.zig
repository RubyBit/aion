// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Fused LSTM cell execution for the GPU backend (kernels/lstm.wgsl): one
//! thread per (batch, hidden) element computes all four gates and writes the
//! [h_t | c_t] state row. v1: f32, every operand a single packed tile.
//!
//! Numerics note: the CPU exec uses sigmoid/tanh fast approximations; the GPU
//! uses exact builtins — CPU-vs-GPU comparisons need a ~1e-3 tolerance.

const std = @import("std");
const wgpu = @import("../wgpu.zig");
const pipelines = @import("../pipelines.zig");
const context = @import("../context.zig");
const backend_mod = @import("../../backend.zig");
const executable = @import("../../../runtime/executable.zig");
const device_store = @import("../../../runtime/device_store.zig");

const c = wgpu.c;
const Ctx = context.Ctx;
const Frame = @import("../frame.zig").Frame;
const ExecuteProgramError = backend_mod.ExecuteProgramError;
const KernelDesc = pipelines.KernelDesc;

const lstm_kernel: KernelDesc = .{ .name = "lstm", .wgsl = @embedFile("../kernels/lstm.wgsl") };

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

/// Acquire tensor `id`'s single packed f32 tile (Unsupported otherwise).
/// Caller releases via `hs.releaseConst(tile.token)`.
fn acquirePackedConst(ctx: Ctx, id: executable.TensorId, min_elems: usize) ExecuteProgramError!device_store.TileRef {
    const hs = ctx.store;
    const meta = hs.meta(id) catch return error.ExecutionFailed;
    if (meta.dtype != .f32 or context.totalTiles(meta) != 1) return error.Unsupported;
    const t = ctx.store.acquireTileDeviceConstLinear(id, 0) catch return error.ExecutionFailed;
    errdefer hs.releaseConst(t.token);
    const rank: usize = @as(usize, t.rank);
    const n = context.packedElems(t.rank, t.shape_mem[0..rank], t.strides_mem[0..rank]) orelse return error.Unsupported;
    if (n < min_elems) return error.Unsupported;
    if (!context.storageBindingFits(ctx, t.len)) return error.Unsupported;
    return t;
}

pub fn execLSTMCell(ctx: Ctx, frame: *Frame, s: executable.StepLSTMCellFused) ExecuteProgramError!void {
    const hs = ctx.store;
    const out_meta = hs.meta(s.out_state) catch return error.ExecutionFailed;
    const x_meta = hs.meta(s.x) catch return error.ExecutionFailed;
    const h_meta = hs.meta(s.h_prev) catch return error.ExecutionFailed;

    if (out_meta.dtype != .f32 or out_meta.rank != 2 or x_meta.rank != 2 or h_meta.rank != 2) return error.Unsupported;
    if (context.totalTiles(out_meta) != 1) return error.Unsupported;

    const batch = x_meta.shape[0];
    const input_size = x_meta.shape[1];
    const hidden = h_meta.shape[1];
    if (batch == 0 or input_size == 0 or hidden == 0) return error.Unsupported;
    const gate_dim = hidden * 4;

    const has_bias = s.b_ih != null;
    if (has_bias != (s.b_hh != null)) return error.Unsupported;

    const dx = try acquirePackedConst(ctx, s.x, batch * input_size);
    defer hs.releaseConst(dx.token);
    const dh = try acquirePackedConst(ctx, s.h_prev, batch * hidden);
    defer hs.releaseConst(dh.token);
    const dc = try acquirePackedConst(ctx, s.c_prev, batch * hidden);
    defer hs.releaseConst(dc.token);
    const dwih = try acquirePackedConst(ctx, s.w_ih, input_size * gate_dim);
    defer hs.releaseConst(dwih.token);
    const dwhh = try acquirePackedConst(ctx, s.w_hh, hidden * gate_dim);
    defer hs.releaseConst(dwhh.token);

    var dbih: ?device_store.TileRef = null;
    var dbhh: ?device_store.TileRef = null;
    defer {
        if (dbih) |t| hs.releaseConst(t.token);
        if (dbhh) |t| hs.releaseConst(t.token);
    }
    if (has_bias) {
        dbih = try acquirePackedConst(ctx, s.b_ih.?, gate_dim);
        dbhh = try acquirePackedConst(ctx, s.b_hh.?, gate_dim);
    }

    const dout = ctx.store.acquireTileDeviceMutLinear(s.out_state, 0) catch return error.ExecutionFailed;
    defer hs.releaseMut(dout.token);
    const out_n = context.packedElems(dout.rank, dout.shape_mem[0..2], dout.strides_mem[0..2]) orelse return error.Unsupported;
    if (out_n < batch * hidden * 2) return error.Unsupported;

    const total = std.math.cast(u32, batch * hidden) orelse return error.Unsupported;
    const params: LstmParams = .{
        .batch = @intCast(batch),
        .input_size = @intCast(input_size),
        .hidden = @intCast(hidden),
        .has_bias = @intFromBool(has_bias),
        .total = total,
    };
    const built = try ctx.pipes.get(lstm_kernel, "lstm_cell");

    // Dummy bias bindings when absent (never read: has_bias == 0).
    const bih_buf = if (dbih) |t| ctx.devmem.bufferFor(t.handle).? else ctx.devmem.bufferFor(dwih.handle).?;
    const bih_len = if (dbih) |t| t.len else dwih.len;
    const bhh_buf = if (dbhh) |t| ctx.devmem.bufferFor(t.handle).? else ctx.devmem.bufferFor(dwih.handle).?;
    const bhh_len = if (dbhh) |t| t.len else dwih.len;

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
    const groups = @max(1, @min(context.ceilDiv(total, WG_1D), context.MAX_GROUPS_1D));
    try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups, 1, 1 });
}
