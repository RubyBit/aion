// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Conv1D / Conv2D execution for the GPU backend (kernels/conv.wgsl): a direct
//! one-thread-per-output-element kernel, f32, channel-last. Conv1D lowers onto
//! the same kernel as Conv2D with the width axis collapsed to 1.
//!
//! v1 contract (Unsupported otherwise):
//!   - rank 3 (conv1d [B, L, C]) / rank 4 (conv2d [B, H, W, C]), f32;
//!   - every operand one device buffer;
//!   - pad_mode zero or reflect (reflect needs input extent >= 2).
//! One dispatch covers every batch.
//!
//! A single-group conv instead lowers onto the generated implicit-GEMM kernels
//! (`matmul/codegen.zig`, `kind = .conv`): same register blocking as the GEMM,
//! with A gathered from the activation (zero or reflect padding), or on a device
//! with matrix units the same tiles as the GEMM, read from the activation in place
//! (`recordConvTiles`, zero padding). The direct kernel stays as the general case:
//! grouped, depthwise, and anything the GEMM grid cannot cover.

const std = @import("std");
const wgpu = @import("../wgpu.zig");
const pipelines = @import("../pipelines.zig");
const context = @import("../context.zig");
const autotune = @import("../autotune.zig");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const tensor_store_mod = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");
const device_store = @import("../../../runtime/device_store.zig");

const c = wgpu.c;
const Ctx = context.Ctx;
const Frame = @import("../frame.zig").Frame;
const ExecuteProgramError = backend_mod.ExecuteProgramError;
const KernelDesc = pipelines.KernelDesc;
const codegen = @import("../matmul/codegen.zig");
const Generated = codegen.Generated;
const matmul_exec = @import("matmul.zig");

/// Uniform for the implicit-GEMM conv: the GEMM's own `Params` with the conv
/// geometry appended, matching the struct `codegen.header` emits.
const ConvGemmParams = extern struct {
    m: u32,
    n: u32,
    k: u32,
    dims3: u32 = 0,
    a_row: u32 = 0,
    b_row: u32,
    c_row: u32,
    strides3: u32 = 0,
    alpha: f32 = 1.0,
    beta: f32 = 0.0,
    ab2: f32 = 0.0,
    ab3: f32 = 0.0,
    ow_out: u32,
    h_in: u32,
    w_in: u32,
    c_in: u32,
    kh: u32,
    kw: u32,
    x_batch: u32,
    pad_top: u32,
    pad_left: u32,
    stride_h: u32,
    stride_w: u32,
    dil_h: u32,
    dil_w: u32,
    has_bias: u32,
    ohw: u32,
    reflect: u32,
};

/// The implicit-GEMM operands of one conv: x, w, bias (w when absent), out.
const ConvGemmOperands = struct { bufs: [4]c.WGPUBuffer, sizes: [4]u64 };

/// Record `gen` over the whole conv, or null when its grid exceeds the device's.
fn recordConvGemm(ctx: Ctx, frame: *Frame, gen: Generated, gp: ConvGemmParams, ops: ConvGemmOperands) ExecuteProgramError!?void {
    if (gen.cfg.matrix) return recordConvTiles(ctx, frame, gen, gp, ops);
    const built = try ctx.pipes.get(gen.desc, gen.entry);
    const gx = context.ceilDiv(gp.n, gen.cfg.bn);
    const gy = context.ceilDiv(gp.m, gen.cfg.bm);
    if (gx > context.MAX_GROUPS_PER_DIM or gy > context.MAX_GROUPS_PER_DIM) return null;
    try frame.recordCompute(built, &ops.bufs, &ops.sizes, std.mem.asBytes(&gp), .{ gx, gy, 1 });
}

const pad_kernel: KernelDesc = .{ .name = "pad", .wgsl = @embedFile("../kernels/pad.wgsl") };
/// Field order matches `struct Params` in pad.wgsl.
const PadParams = extern struct { src_shape: [4]u32, dst_shape: [4]u32, at: [4]u32 };

/// Whether a conv can run on matrix tiles at all: its channels divide into
/// tiles, an output row holds one, and padding is zero (the padded copy below).
fn tilesFit(gp: ConvGemmParams) bool {
    return gp.c_in % codegen.TILE == 0 and gp.ow_out >= codegen.TILE and gp.reflect == 0;
}

/// The matrix-tile conv (see `codegen.convTileFn`). Its tiles are read from the
/// activation in place, so no window may leave it: where padding would, the
/// activation is first copied, zero-padded, into the pooled scratch, and the
/// kernel is pointed at that copy and told there is no padding.
fn recordConvTiles(ctx: Ctx, frame: *Frame, gen: Generated, conv: ConvGemmParams, operands: ConvGemmOperands) ExecuteProgramError!?void {
    var gp = conv;
    var ops = operands;
    const rows = gp.m / gp.ow_out; // batch * output height
    const span_h = (gp.ohw / gp.ow_out - 1) * gp.stride_h + (gp.kh - 1) * gp.dil_h + 1;
    const span_w = (gp.ow_out - 1) * gp.stride_w + (gp.kw - 1) * gp.dil_w + 1;
    if (gp.pad_top != 0 or gp.pad_left != 0 or span_h > gp.h_in or span_w > gp.w_in) {
        const batch = gp.m / gp.ohw;
        const vec4s = @as(u64, batch) * span_h * span_w * (gp.c_in / 4);
        const bytes = vec4s * 16;
        const count = std.math.cast(u32, vec4s) orelse return null;
        if (!context.storageBindingFits(ctx, bytes)) return null;
        const padded = try ctx.scratch.ensure(ctx.gpu, bytes);
        const pp: PadParams = .{
            .src_shape = .{ batch, gp.h_in, gp.w_in, gp.c_in / 4 },
            .dst_shape = .{ batch, span_h, span_w, gp.c_in / 4 },
            .at = .{ gp.pad_top, gp.pad_left, count, 0 },
        };
        const built = try ctx.pipes.get(pad_kernel, "pad_f32");
        try frame.recordCompute(built, &.{ ops.bufs[0], padded }, &.{ ops.sizes[0], bytes }, std.mem.asBytes(&pp), .{ groups1D(count), 1, 1 });
        ops.bufs[0] = padded;
        ops.sizes[0] = bytes;
        gp.h_in = span_h;
        gp.w_in = span_w;
        gp.x_batch = span_h * span_w * gp.c_in;
        gp.pad_top = 0;
        gp.pad_left = 0;
    }
    const built = try ctx.pipes.get(gen.desc, gen.entry);
    const gx = context.ceilDiv(gp.n, gen.cfg.bn);
    const gy = context.ceilDiv(rows * context.ceilDiv(gp.ow_out, codegen.TILE), gen.cfg.bm / codegen.TILE);
    if (gx > context.MAX_GROUPS_PER_DIM or gy > context.MAX_GROUPS_PER_DIM) return null;
    try frame.recordCompute(built, &ops.bufs, &ops.sizes, std.mem.asBytes(&gp), .{ gx, gy, 1 });
}

/// The fastest implicit-GEMM block for this conv, timed on the device once per
/// shape: how wide a block its channels fill and how many blocks keep the GPU
/// busy are both properties of the device, which WebGPU does not describe.
fn chooseConvConfig(mm: *const matmul_exec.Matmul, ctx: Ctx, frame: *Frame, gp: ConvGemmParams, ops: ConvGemmOperands) ExecuteProgramError!?usize {
    const Cand = struct {
        ctx: Ctx,
        gp: ConvGemmParams,
        ops: ConvGemmOperands,
        generated: []const Generated,

        pub fn eligible(t: @This(), idx: usize) bool {
            const cfg = t.generated[idx].cfg;
            const in_grid = context.ceilDiv(t.gp.n, cfg.bn) <= context.MAX_GROUPS_PER_DIM and
                context.ceilDiv(t.gp.m, cfg.bm) <= context.MAX_GROUPS_PER_DIM;
            return in_grid and codegen.fits(cfg, t.gp.m, t.gp.n, t.gp.k) and (!cfg.matrix or tilesFit(t.gp)) and
                matmul_exec.eligibleConfig(cfg, t.ctx.gpu.limits, 16, @intCast(t.gp.n * @sizeOf(f32)));
        }

        pub fn record(t: @This(), idx: usize, f: *Frame) ExecuteProgramError!void {
            _ = (try recordConvGemm(t.ctx, f, t.generated[idx], t.gp, t.ops)) orelse return error.Unsupported;
        }
    };
    const cand: Cand = .{ .ctx = ctx, .gp = gp, .ops = ops, .generated = mm.generated_conv };
    return ctx.tune.pick(ctx, frame, .of(.conv, .{ autotune.sizeClass(gp.m), gp.n, gp.k }), mm.generated_conv.len, cand);
}

const conv_kernel: KernelDesc = .{ .name = "conv", .wgsl = @embedFile("../kernels/conv.wgsl") };

const WG_1D: u32 = 64;

// Depthwise conv1d fast-path tiling (mirrors conv.wgsl `conv_dw_f32`).
const DW_NCG: u32 = 64; // channel-groups (of 4 channels) per workgroup
const DW_LT: u32 = 16; // output length positions per workgroup
const DW_SPAN_MAX: u32 = 40;
const DW_SHARED_BYTES: u64 = DW_SPAN_MAX * DW_NCG * 16; // Xs: vec4<f32> halo

/// Whether the depthwise conv1d fast kernel applies to this whole conv.
fn depthwiseOk(geo: Geometry, c_in_g: usize, c_out: usize, rank: usize) bool {
    if (rank != 3) return false; // conv1d only
    if (geo.pad_mode != .zero) return false; // fast path is zero-pad only
    if (c_in_g != 1 or geo.groups != geo.c_in or c_out != geo.c_in) return false;
    if (c_out % 4 != 0) return false; // channel-groups are always full
    const span = (DW_LT - 1) * geo.stride_h + (geo.kh - 1) * geo.dil_h + 1;
    return span <= DW_SPAN_MAX;
}

/// Field order matches `struct Params` in conv.wgsl.
const ConvParams = extern struct {
    x_batch: u32,
    h_in: u32,
    w_in: u32,
    c_in: u32,
    kh: u32,
    kw: u32,
    c_in_g: u32,
    c_out_g: u32,
    c_out: u32,
    stride_h: u32,
    stride_w: u32,
    dil_h: u32,
    dil_w: u32,
    pad_top: u32,
    pad_left: u32,
    batch: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    oh_cnt: u32,
    ow_cnt: u32,
    c_cnt: u32,
    total: u32,
    reflect: u32,
    has_bias: u32,
};

fn groups1D(n: u32) u32 {
    return @max(1, @min(context.ceilDiv(n, WG_1D), context.MAX_GROUPS_1D));
}

/// Geometry normalized to the 2D kernel (conv1d: width axis == 1).
const Geometry = struct {
    h_in: usize,
    w_in: usize,
    c_in: usize,
    kh: usize,
    kw: usize,
    groups: usize,
    stride_h: usize,
    stride_w: usize,
    dil_h: usize,
    dil_w: usize,
    pad_top: usize,
    pad_left: usize,
    pad_mode: types.PadMode,
};

fn prodU(xs: []const usize) usize {
    var p: usize = 1;
    for (xs) |x| p *= x;
    return p;
}

pub fn execConv1D(ctx: Ctx, frame: *Frame, s: executable.StepConv1D, mm: *matmul_exec.Matmul) ExecuteProgramError!void {
    const hs = ctx.store;
    const x_meta = hs.meta(s.x) catch return error.ExecutionFailed;
    if (x_meta.rank != 3) return error.Unsupported;
    const geo: Geometry = .{
        .h_in = x_meta.shape[1],
        .w_in = 1,
        .c_in = x_meta.shape[2],
        .kh = 0, // filled from w below
        .kw = 1,
        .groups = s.groups,
        .stride_h = s.stride,
        .stride_w = 1,
        .dil_h = s.dilation,
        .dil_w = 1,
        .pad_top = s.pad_left,
        .pad_left = 0,
        .pad_mode = s.pad_mode,
    };
    return execConv(ctx, frame, s.out, s.x, s.w, s.bias, geo, 3, mm);
}

pub fn execConv2D(ctx: Ctx, frame: *Frame, s: executable.StepConv2D, mm: *matmul_exec.Matmul) ExecuteProgramError!void {
    const hs = ctx.store;
    const x_meta = hs.meta(s.x) catch return error.ExecutionFailed;
    if (x_meta.rank != 4) return error.Unsupported;
    const geo: Geometry = .{
        .h_in = x_meta.shape[1],
        .w_in = x_meta.shape[2],
        .c_in = x_meta.shape[3],
        .kh = 0,
        .kw = 0,
        .groups = s.groups,
        .stride_h = s.stride_h,
        .stride_w = s.stride_w,
        .dil_h = s.dilation_h,
        .dil_w = s.dilation_w,
        .pad_top = s.pad_top,
        .pad_left = s.pad_left,
        .pad_mode = s.pad_mode,
    };
    return execConv(ctx, frame, s.out, s.x, s.w, s.bias, geo, 4, mm);
}

fn execConv(
    ctx: Ctx,
    frame: *Frame,
    out_id: executable.TensorId,
    x_id: executable.TensorId,
    w_id: executable.TensorId,
    bias_id: ?executable.TensorId,
    geo_in: Geometry,
    rank: usize,
    /// The implicit-GEMM kernels and their tuning.
    mm: *matmul_exec.Matmul,
) ExecuteProgramError!void {
    const hs = ctx.store;
    const out_meta = hs.meta(out_id) catch return error.ExecutionFailed;
    const x_meta = hs.meta(x_id) catch return error.ExecutionFailed;
    const w_meta = hs.meta(w_id) catch return error.ExecutionFailed;

    if (@as(usize, out_meta.rank) != rank or @as(usize, x_meta.rank) != rank) return error.Unsupported;
    if (@as(usize, w_meta.rank) != rank) return error.Unsupported; // [k(,kw), c_in_g, c_out]
    if (out_meta.dtype != .f32 or x_meta.dtype != .f32 or w_meta.dtype != .f32) return error.Unsupported;

    var geo = geo_in;
    geo.kh = w_meta.shape[0];
    if (rank == 4) geo.kw = w_meta.shape[1];
    const c_in_g = w_meta.shape[rank - 2];
    const c_out = out_meta.shape[rank - 1];
    if (w_meta.shape[rank - 1] != c_out) return error.Unsupported;

    if (geo.groups == 0 or c_out % geo.groups != 0) return error.Unsupported;
    if (c_in_g * geo.groups != geo.c_in) return error.Unsupported;
    switch (geo.pad_mode) {
        .zero => {},
        .reflect => {
            // reflect_idx diverges on extent 1 (compile validates this too).
            if (geo.h_in < 2 or (rank == 4 and geo.w_in < 2)) return error.Unsupported;
        },
    }

    // Every operand is one device buffer (a tensor past the binding limit would be
    // chunked, which this kernel does not address).
    inline for (.{ out_meta, x_meta, w_meta }) |m| if (m.chunks != 1) return error.Unsupported;
    var dbias: ?device_store.Chunk = null;
    defer if (dbias) |bt| hs.releaseConst(bt.token);
    if (bias_id) |bid| {
        const b_meta = hs.meta(bid) catch return error.ExecutionFailed;
        if (b_meta.rank != 1 or b_meta.dtype != .f32) return error.Unsupported;
        if (b_meta.shape[0] < c_out or b_meta.chunks != 1) return error.Unsupported;
        dbias = ctx.store.acquireConst(bid) catch return error.ExecutionFailed;
    }
    const dw = ctx.store.acquireConst(w_id) catch return error.ExecutionFailed;
    defer hs.releaseConst(dw.token);
    const dx = ctx.store.acquireConst(x_id) catch return error.ExecutionFailed;
    defer hs.releaseConst(dx.token);
    const dout = ctx.store.acquireMut(out_id) catch return error.ExecutionFailed;
    defer hs.releaseMut(dout.token);
    inline for (.{ dw.len, dx.len, dout.len }) |len| if (!context.storageBindingFits(ctx, len)) return error.Unsupported;

    const batch = out_meta.shape[0];
    const oh_cnt = out_meta.shape[1];
    const ow_cnt = if (rank == 4) out_meta.shape[2] else 1;
    const x_batch = geo.h_in * geo.w_in * geo.c_in;
    const out_total = batch * oh_cnt * ow_cnt * c_out;
    if (out_total == 0) return;

    const bias_buf = if (dbias) |bt| ctx.devmem.bufferFor(bt.handle).? else ctx.devmem.bufferFor(dw.handle).?;
    const bias_len = if (dbias) |bt| bt.len else dw.len;
    const bufs = [_]c.WGPUBuffer{ ctx.devmem.bufferFor(dx.handle).?, ctx.devmem.bufferFor(dw.handle).?, bias_buf, ctx.devmem.bufferFor(dout.handle).? };
    const sizes = [_]u64{ dx.len, dw.len, bias_len, dout.len };

    // Implicit GEMM needs a single group: a group splits A by output column. M
    // runs over every batch's pixels, so C is the output buffer exactly.
    if (geo.groups == 1) gemm: {
        const gp: ConvGemmParams = .{
            .m = std.math.cast(u32, batch * oh_cnt * ow_cnt) orelse break :gemm,
            .n = std.math.cast(u32, c_out) orelse break :gemm,
            .k = std.math.cast(u32, geo.kh * geo.kw * geo.c_in) orelse break :gemm,
            .b_row = std.math.cast(u32, c_out) orelse break :gemm,
            .c_row = std.math.cast(u32, c_out) orelse break :gemm,
            .ow_out = @intCast(ow_cnt),
            .h_in = @intCast(geo.h_in),
            .w_in = @intCast(geo.w_in),
            .c_in = @intCast(geo.c_in),
            .kh = @intCast(geo.kh),
            .kw = @intCast(geo.kw),
            .x_batch = std.math.cast(u32, x_batch) orelse break :gemm,
            .pad_top = @intCast(geo.pad_top),
            .pad_left = @intCast(geo.pad_left),
            .stride_h = @intCast(geo.stride_h),
            .stride_w = @intCast(geo.stride_w),
            .dil_h = @intCast(geo.dil_h),
            .dil_w = @intCast(geo.dil_w),
            .has_bias = @intFromBool(dbias != null),
            .ohw = @intCast(oh_cnt * ow_cnt),
            .reflect = @intFromBool(geo.pad_mode == .reflect),
        };
        const ops: ConvGemmOperands = .{ .bufs = bufs, .sizes = sizes };
        const gi = (try chooseConvConfig(mm, ctx, frame, gp, ops)) orelse break :gemm;
        if (try recordConvGemm(ctx, frame, mm.generated_conv[gi], gp, ops)) |_| return;
    }

    const use_dw = depthwiseOk(geo, c_in_g, c_out, rank) and
        ctx.gpu.limits.max_shared_bytes >= DW_SHARED_BYTES;
    // vec4 channel contraction: any pad mode, needs the group channel count % 4.
    const use_vec4c = !use_dw and c_in_g % 4 == 0;
    const entry = if (use_dw) "conv_dw_f32" else if (use_vec4c) "conv_f32_vec4c" else "conv_f32";
    const params: ConvParams = .{
        .x_batch = std.math.cast(u32, x_batch) orelse return error.Unsupported,
        .h_in = @intCast(geo.h_in),
        .w_in = @intCast(geo.w_in),
        .c_in = @intCast(geo.c_in),
        .kh = @intCast(geo.kh),
        .kw = @intCast(geo.kw),
        .c_in_g = @intCast(c_in_g),
        .c_out_g = @intCast(c_out / geo.groups),
        .c_out = @intCast(c_out),
        .stride_h = @intCast(geo.stride_h),
        .stride_w = @intCast(geo.stride_w),
        .dil_h = @intCast(geo.dil_h),
        .dil_w = @intCast(geo.dil_w),
        .pad_top = @intCast(geo.pad_top),
        .pad_left = @intCast(geo.pad_left),
        .batch = @intCast(batch),
        .oh_cnt = @intCast(oh_cnt),
        .ow_cnt = @intCast(ow_cnt),
        .c_cnt = @intCast(c_out),
        .total = std.math.cast(u32, out_total) orelse return error.Unsupported,
        .reflect = @intFromBool(geo.pad_mode == .reflect),
        .has_bias = @intFromBool(dbias != null),
    };

    // Depthwise: a 3D grid (channel-groups x length-blocks x batch); fall back to
    // the grid-strided generic kernel if any grid dim exceeds the per-dim cap.
    if (use_dw) {
        const cg = context.ceilDiv(@intCast(c_out), DW_NCG * 4);
        const lg = context.ceilDiv(@intCast(oh_cnt), DW_LT);
        if (cg <= context.MAX_GROUPS_PER_DIM and lg <= context.MAX_GROUPS_PER_DIM and batch <= context.MAX_GROUPS_PER_DIM) {
            const built = try ctx.pipes.get(conv_kernel, entry);
            return frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ cg, lg, @intCast(batch) });
        }
    }
    const built = try ctx.pipes.get(conv_kernel, if (use_dw) "conv_f32" else entry);
    try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(params.total), 1, 1 });
}
