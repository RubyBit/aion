// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Conv1D / Conv2D execution for the GPU backend (kernels/conv.wgsl): a direct
//! one-thread-per-output-element kernel, f32, channel-last. Conv1D lowers onto
//! the same kernel as Conv2D with the width axis collapsed to 1.
//!
//! v1 contract (Unsupported otherwise):
//!   - rank 3 (conv1d [B, L, C]) / rank 4 (conv2d [B, H, W, C]), f32;
//!   - out tiles carry one batch each (tile_shape[0] == 1);
//!   - x is one tile per batch group (spatial/channel dims untiled);
//!   - w and bias are single packed tiles;
//!   - pad_mode zero or reflect (reflect needs input extent >= 2).
//!
//! Performance note: this is the correctness base case. The CPU backend's
//! implicit-GEMM path has no GPU counterpart yet — when conv shows up hot on a
//! GPU profile, stage patches into shared memory or lower onto the existing
//! GEMM pipelines.

const std = @import("std");
const wgpu = @import("../wgpu.zig");
const pipelines = @import("../pipelines.zig");
const context = @import("../context.zig");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const tensor_store_mod = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");
const resident_mod = @import("../../../runtime/residency/resident_store.zig");

const c = wgpu.c;
const Ctx = context.Ctx;
const Frame = @import("../frame.zig").Frame;
const ExecuteProgramError = backend_mod.ExecuteProgramError;
const KernelDesc = pipelines.KernelDesc;

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
    x_base: u32,
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
    base_h: u32,
    base_w: u32,
    base_c: u32,
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

/// Whether x has any spatial/channel dim (dims 1..) split into multiple tiles.
fn xSpatiallyTiled(x_meta: TensorMetaT, rank: usize) bool {
    var d: usize = 1;
    while (d < rank) : (d += 1) if (x_meta.tile_counts[d] != 1) return true;
    return false;
}

const TensorMetaT = tensor_store_mod.TensorMeta;

/// A conv input materialized into one contiguous scratch tile.
const MaterializedX = struct { buf: c.WGPUBuffer, len: u64 };

/// Materialize a spatially-tiled x into one contiguous `[batch,H,W,C]` scratch
/// buffer (the conv kernel needs the whole spatial extent addressable from one
/// binding). Handles x tiled along a SINGLE dim with the others whole — e.g. a
/// mel/subsampling input tiled per time-frame. Each tile is a contiguous run in
/// packed order, so this is `n_tiles * outer` buffer copies. Returns the scratch
/// buffer + its byte length.
fn materializeX(ctx: Ctx, frame: *Frame, x_id: executable.TensorId, x_meta: TensorMetaT, rank: usize) ExecuteProgramError!MaterializedX {
    const hs = ctx.rstore.tensorStore();
    const elem: usize = 4;

    // Find the single split dim.
    var sp: ?usize = null;
    var d: usize = 0;
    while (d < rank) : (d += 1) {
        if (x_meta.tile_counts[d] != 1) {
            if (sp != null) return error.Unsupported; // more than one split dim
            sp = d;
        }
    }
    const split = sp orelse return error.Unsupported;

    const total_elems = prodU(x_meta.shape[0..rank]);
    const total_bytes: u64 = @as(u64, total_elems) * elem;
    if (!context.storageBindingFits(ctx, total_bytes)) return error.Unsupported;
    const scratch = try ctx.scratch.ensure(ctx.gpu, total_bytes);

    const outer = prodU(x_meta.shape[0..split]);
    const inner = prodU(x_meta.shape[split + 1 .. rank]);
    const dim = x_meta.shape[split];
    const ts = x_meta.tile_shape[split];
    const n_tiles = x_meta.tile_counts[split];
    if (ts == 0) return error.Unsupported;

    var t: usize = 0;
    while (t < n_tiles) : (t += 1) {
        const lo = t * ts;
        const this_ts = @min(ts, dim - lo);
        const xt = ctx.rstore.acquireTileDeviceConstLinear(x_id, t) catch return error.ExecutionFailed;
        defer hs.releaseConst(xt.token);
        if (!context.storageBindingFits(ctx, xt.len)) return error.Unsupported;
        const xt_buf = ctx.devmem.bufferFor(xt.handle).?;
        const run_bytes: u64 = @as(u64, this_ts) * inner * elem;
        var o: usize = 0;
        while (o < outer) : (o += 1) {
            const src_off: u64 = @as(u64, o * this_ts) * inner * elem;
            const dst_off: u64 = @as(u64, o * dim + lo) * inner * elem;
            if (src_off + run_bytes > xt.len) return error.ExecutionFailed;
            frame.recordCopy(xt_buf, src_off, scratch, dst_off, run_bytes);
        }
    }
    return .{ .buf = scratch, .len = total_bytes };
}

pub fn execConv1D(ctx: Ctx, frame: *Frame, s: executable.StepConv1DTiled) ExecuteProgramError!void {
    const hs = ctx.rstore.tensorStore();
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
    return execConv(ctx, frame, s.out, s.x, s.w, s.bias, geo, 3);
}

pub fn execConv2D(ctx: Ctx, frame: *Frame, s: executable.StepConv2DTiled) ExecuteProgramError!void {
    const hs = ctx.rstore.tensorStore();
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
    return execConv(ctx, frame, s.out, s.x, s.w, s.bias, geo, 4);
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
) ExecuteProgramError!void {
    const hs = ctx.rstore.tensorStore();
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

    // x: the kernel needs the whole spatial/channel extent from one binding. If
    // the compiler split a spatial/channel dim (e.g. a mel input tiled per
    // time-frame), materialize x into one contiguous scratch tile first.
    const x_materialized: ?MaterializedX = if (xSpatiallyTiled(x_meta, rank))
        try materializeX(ctx, frame, x_id, x_meta, rank)
    else
        null;
    // w / bias: single packed tiles.
    if (context.totalTiles(w_meta) != 1) return error.Unsupported;
    var bias_tile: ?resident_mod.TileRefDevice = null;
    defer if (bias_tile) |bt| hs.releaseConst(bt.token);
    if (bias_id) |bid| {
        const b_meta = hs.meta(bid) catch return error.ExecutionFailed;
        if (b_meta.rank != 1 or b_meta.dtype != .f32) return error.Unsupported;
        if (b_meta.shape[0] < c_out or context.totalTiles(b_meta) != 1) return error.Unsupported;
        const bt = ctx.rstore.acquireTileDeviceConstLinear(bid, 0) catch return error.ExecutionFailed;
        bias_tile = bt;
        if (context.packedElems(bt.rank, bt.shape_mem[0..1], bt.strides_mem[0..1]) == null) return error.Unsupported;
    }

    const dw = ctx.rstore.acquireTileDeviceConstLinear(w_id, 0) catch return error.ExecutionFailed;
    defer hs.releaseConst(dw.token);
    if (!context.storageBindingFits(ctx, dw.len)) return error.Unsupported;
    if (context.packedElems(dw.rank, dw.shape_mem[0..rank], dw.strides_mem[0..rank]) == null) return error.Unsupported;

    const use_dw = depthwiseOk(geo, c_in_g, c_out, rank) and
        ctx.gpu.limits.max_shared_bytes >= DW_SHARED_BYTES;
    // vec4 channel contraction: any pad mode, needs the group channel count % 4.
    const use_vec4c = !use_dw and c_in_g % 4 == 0;
    const entry = if (use_dw) "conv_dw_f32" else if (use_vec4c) "conv_f32_vec4c" else "conv_f32";
    const built = try ctx.pipes.get(conv_kernel, entry);
    const built_generic = if (use_dw) try ctx.pipes.get(conv_kernel, "conv_f32") else built;
    const x_batch_elems = geo.h_in * geo.w_in * geo.c_in;

    // Cached x device tile (out tiles for the same batch group reuse it).
    var x_cached: ?usize = null;
    var x_tile: resident_mod.TileRefDevice = undefined;
    defer if (x_cached != null) hs.releaseConst(x_tile.token);

    const total = context.totalTiles(out_meta);
    var ti: usize = 0;
    while (ti < total) : (ti += 1) {
        var coords: [tensor_store_mod.INLINE_RANK]usize = @splat(0);
        tensor_store_mod.decodeTileCoords(out_meta, ti, coords[0..rank]) catch return error.ExecutionFailed;
        if (out_meta.tile_shape[0] != 1) return error.Unsupported; // one batch per out tile
        const b = coords[0];

        var x_buf: c.WGPUBuffer = undefined;
        var x_len: u64 = undefined;
        var x_base: usize = undefined;
        if (x_materialized) |xm| {
            // Whole [batch,H,W,C] contiguous in scratch — index by batch directly.
            x_buf = xm.buf;
            x_len = xm.len;
            x_base = b * x_batch_elems;
        } else {
            const x_lin = b / x_meta.tile_shape[0];
            if (x_cached == null or x_cached.? != x_lin) {
                if (x_cached != null) hs.releaseConst(x_tile.token);
                x_cached = null;
                x_tile = ctx.rstore.acquireTileDeviceConstLinear(x_id, x_lin) catch return error.ExecutionFailed;
                x_cached = x_lin;
                if (!context.storageBindingFits(ctx, x_tile.len)) return error.Unsupported;
                if (context.packedElems(x_tile.rank, x_tile.shape_mem[0..rank], x_tile.strides_mem[0..rank]) == null) return error.Unsupported;
                var xd: usize = 1;
                while (xd < rank) : (xd += 1) {
                    if (x_tile.shape_mem[xd] != x_meta.shape[xd]) return error.Unsupported;
                }
            }
            x_buf = ctx.devmem.bufferFor(x_tile.handle).?;
            x_len = x_tile.len;
            x_base = (b % x_meta.tile_shape[0]) * x_batch_elems;
        }

        const dout = ctx.rstore.acquireTileDeviceMutLinear(out_id, ti) catch return error.ExecutionFailed;
        defer hs.releaseMut(dout.token);
        if (!context.storageBindingFits(ctx, dout.len)) return error.Unsupported;
        if (context.packedElems(dout.rank, dout.shape_mem[0..rank], dout.strides_mem[0..rank]) == null) return error.Unsupported;

        const oh_cnt = dout.shape_mem[1];
        const ow_cnt = if (rank == 4) dout.shape_mem[2] else 1;
        const c_cnt = dout.shape_mem[rank - 1];
        const out_total = oh_cnt * ow_cnt * c_cnt;
        if (out_total == 0) continue;

        const params: ConvParams = .{
            .x_base = std.math.cast(u32, x_base) orelse return error.Unsupported,
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
            .base_h = @intCast(coords[1] * out_meta.tile_shape[1]),
            .base_w = if (rank == 4) @intCast(coords[2] * out_meta.tile_shape[2]) else 0,
            .base_c = @intCast(coords[rank - 1] * out_meta.tile_shape[rank - 1]),
            .oh_cnt = @intCast(oh_cnt),
            .ow_cnt = @intCast(ow_cnt),
            .c_cnt = @intCast(c_cnt),
            .total = std.math.cast(u32, out_total) orelse return error.Unsupported,
            .reflect = @intFromBool(geo.pad_mode == .reflect),
            .has_bias = @intFromBool(bias_tile != null),
        };
        const bias_buf = if (bias_tile) |bt| ctx.devmem.bufferFor(bt.handle).? else ctx.devmem.bufferFor(dw.handle).?;
        const bias_len = if (bias_tile) |bt| bt.len else dw.len;
        const bufs = [_]c.WGPUBuffer{
            x_buf,
            ctx.devmem.bufferFor(dw.handle).?,
            bias_buf,
            ctx.devmem.bufferFor(dout.handle).?,
        };
        const sizes = [_]u64{ x_len, dw.len, bias_len, dout.len };

        // Depthwise: 2D grid (channel-groups x length-blocks); fall back to the
        // grid-strided generic kernel if either grid dim exceeds the per-dim cap.
        var tile_built = built;
        var groups: [3]u32 = .{ groups1D(params.total), 1, 1 };
        if (use_dw) {
            const cg = context.ceilDiv(@intCast(c_cnt), DW_NCG * 4);
            const lg = context.ceilDiv(@intCast(oh_cnt), DW_LT);
            if (cg <= context.MAX_GROUPS_PER_DIM and lg <= context.MAX_GROUPS_PER_DIM) {
                groups = .{ cg, lg, 1 };
            } else {
                tile_built = built_generic;
            }
        }
        try frame.recordCompute(tile_built, &bufs, &sizes, std.mem.asBytes(&params), groups);
    }
}
