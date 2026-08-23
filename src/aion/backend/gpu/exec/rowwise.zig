// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Row-wise reduction ops for the GPU backend: softmax, LayerNorm/RMSNorm, and
//! sum/mean reductions. Each maps a tile to one dispatch of one 256-thread
//! workgroup PER ROW, with shared-memory tree reductions inside the workgroup
//! (kernels/softmax.wgsl, norm.wgsl, reduce.wgsl).
//!
//! The reduced axis is the last dimension. Single-column-tile rows take the
//! original one-dispatch path; wider rows use native staged reductions across
//! column tiles while preserving the leading-axis tile grid. No path packs the
//! complete tensor or stages it through the host.

const std = @import("std");
const wgpu = @import("../wgpu.zig");
const pipelines = @import("../pipelines.zig");
const context = @import("../context.zig");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const executable = @import("../../../runtime/executable.zig");
const device_store = @import("../../../runtime/device_store.zig");

const c = wgpu.c;
const Ctx = context.Ctx;
const Frame = @import("../frame.zig").Frame;
const ExecuteProgramError = backend_mod.ExecuteProgramError;
const KernelDesc = pipelines.KernelDesc;
const TileRefDevice = device_store.TileRef;

const softmax_kernel: KernelDesc = .{ .name = "softmax", .wgsl = @embedFile("../kernels/softmax.wgsl") };
const norm_kernel: KernelDesc = .{ .name = "norm", .wgsl = @embedFile("../kernels/norm.wgsl") };
const add_norm_kernel: KernelDesc = .{ .name = "add_norm", .wgsl = @embedFile("../kernels/add_norm.wgsl") };
const reduce_kernel: KernelDesc = .{ .name = "reduce", .wgsl = @embedFile("../kernels/reduce.wgsl") };
const reduce_i32_kernel: KernelDesc = .{ .name = "reduce_i32", .wgsl = @embedFile("../kernels/reduce_i32.wgsl") };
const argmax_kernel: KernelDesc = .{ .name = "argmax", .wgsl = @embedFile("../kernels/argmax.wgsl") };
const argmax_cross_kernel: KernelDesc = .{ .name = "argmax_cross", .wgsl = @embedFile("../kernels/argmax_cross.wgsl") };
const rowwise_partial_kernel: KernelDesc = .{ .name = "rowwise_partial", .wgsl = @embedFile("../kernels/rowwise_partial.wgsl") };
const rowwise_finish_kernel: KernelDesc = .{ .name = "rowwise_finish", .wgsl = @embedFile("../kernels/rowwise_finish.wgsl") };
const softmax_cross_apply_kernel: KernelDesc = .{ .name = "softmax_cross_apply", .wgsl = @embedFile("../kernels/softmax_cross_apply.wgsl") };
const norm_cross_apply_kernel: KernelDesc = .{ .name = "norm_cross_apply", .wgsl = @embedFile("../kernels/norm_cross_apply.wgsl") };

/// Uniform params shared by softmax (`rows/cols/x_row/o_row`) and, prefix-wise,
/// reduce (`rows/cols/x_row`). Field order matches the WGSL structs.
const RowParams = extern struct { rows: u32, cols: u32, x_row: u32, o_row: u32 = 0 };
const NormParams = extern struct { rows: u32, cols: u32, x_row: u32, o_row: u32, eps: f32, _p0: u32 = 0, _p1: u32 = 0, _p2: u32 = 0 };
/// Field order matches `Params` in add_norm.wgsl. Pads are scalar `u32` on purpose: a
/// `vec3<u32>` pad forces 16-byte alignment in WGSL and the struct sizes then disagree,
/// which surfaces only as a bare wgpu uncaptured error.
const AddNormParams = extern struct { rows: u32, cols: u32, x_row: u32, o_row: u32, a_row: u32, eps: f32, _p0: u32 = 0, _p1: u32 = 0 };
const CrossParams = extern struct {
    rows: u32,
    cols: u32,
    x_row: u32,
    o_row: u32 = 0,
    parts: u32,
    part: u32 = 0,
    col_base: u32 = 0,
    full_cols: u32,
    stat_base: u32,
    mode: u32 = 0,
    eps: f32 = 0,
    _pad: u32 = 0,
};
const ArgmaxCrossParams = extern struct {
    rows: u32,
    cols: u32,
    x_row: u32,
    parts: u32,
    part: u32 = 0,
    col_base: u32 = 0,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
};

pub const NormMode = enum { rmsnorm, layernorm };

fn requireF32(dtype: types.DType) ExecuteProgramError!void {
    if (dtype != .f32) return error.Unsupported;
}

fn deviceRowView(t: TileRefDevice) ExecuteProgramError!context.RowView {
    return context.rowView(t.rank, t.shape_mem[0..@as(usize, t.rank)], t.strides_mem[0..@as(usize, t.rank)]) orelse error.Unsupported;
}

/// Row view in the tile's own scalar size, so an f16 tile reports element counts
/// rather than being rejected for a 2-byte innermost stride.
fn deviceRowViewDt(t: TileRefDevice, dtype: types.DType) ExecuteProgramError!context.RowView {
    return context.rowViewSized(t.rank, t.shape_mem[0..@as(usize, t.rank)], t.strides_mem[0..@as(usize, t.rank)], try scalarBytes(dtype)) orelse error.Unsupported;
}

/// The scalar float dtypes the row-wise kernels accept. Each f16 kernel is an
/// entry point in the SAME module as its f32 twin, keeps its row statistics in
/// f32, and rounds only on store — matching the CPU kernels exactly.
fn scalarBytes(dtype: types.DType) ExecuteProgramError!usize {
    return switch (dtype) {
        .f32 => 4,
        .f16 => 2,
        else => error.Unsupported,
    };
}

fn requireScalarFloat(dtype: types.DType) ExecuteProgramError!void {
    _ = try scalarBytes(dtype);
}

fn devicePackedElems(t: TileRefDevice) ExecuteProgramError!u32 {
    return context.packedElems(t.rank, t.shape_mem[0..@as(usize, t.rank)], t.strides_mem[0..@as(usize, t.rank)]) orelse error.Unsupported;
}

/// `devicePackedElems` in the tile's own scalar size (see `deviceRowViewDt`).
fn devicePackedElemsDt(t: TileRefDevice, dtype: types.DType) ExecuteProgramError!u32 {
    const n = context.packedElemsSized(t.rank, t.shape_mem[0..@as(usize, t.rank)], t.strides_mem[0..@as(usize, t.rank)], try scalarBytes(dtype)) orelse return error.Unsupported;
    return std.math.cast(u32, n) orelse error.Unsupported;
}

/// Softmax over the last axis. Negative axes were normalized by the graph API;
/// the step still carries i32, so normalize again here.
pub fn execSoftmax(ctx: Ctx, frame: *Frame, s: executable.StepSoftmaxTiled) ExecuteProgramError!void {
    const hs = ctx.store;
    const meta = hs.meta(s.out) catch return error.ExecutionFailed;
    try requireScalarFloat(meta.dtype);
    if ((hs.meta(s.a) catch return error.ExecutionFailed).dtype != meta.dtype) return error.Unsupported;

    const rank: usize = @as(usize, meta.rank);
    var axis: i32 = s.axis;
    if (axis < 0) axis += @intCast(rank);
    if (axis < 0 or axis != @as(i32, @intCast(rank - 1))) return error.Unsupported; // v1: last axis only
    if (meta.tile_counts[rank - 1] != 1) return execSoftmaxCrossTile(ctx, frame, s, meta, rank);

    const built = try ctx.pipes.get(softmax_kernel, if (meta.dtype == .f16) "softmax_row_f16" else "softmax_row");

    const total = context.totalTiles(meta);
    var ti: usize = 0;
    while (ti < total) : (ti += 1) {
        const dx = ctx.store.acquireTileDeviceConstLinear(s.a, ti) catch return error.ExecutionFailed;
        const dout = ctx.store.acquireTileDeviceMutLinear(s.out, ti) catch return error.ExecutionFailed;
        defer {
            hs.releaseConst(dx.token);
            hs.releaseMut(dout.token);
        }
        if (!context.storageBindingFits(ctx, dx.len) or !context.storageBindingFits(ctx, dout.len)) return error.Unsupported;

        const xv = try deviceRowViewDt(dx, meta.dtype);
        const ov = try deviceRowViewDt(dout, meta.dtype);
        if (xv.rows != ov.rows or xv.cols != ov.cols) return error.Unsupported;
        if (xv.rows > context.MAX_GROUPS_PER_DIM) return error.Unsupported;

        const bufs = [_]c.WGPUBuffer{
            ctx.devmem.bufferFor(dx.handle).?,
            ctx.devmem.bufferFor(dout.handle).?,
        };
        const sizes = [_]u64{ dx.len, dout.len };
        const params: RowParams = .{ .rows = xv.rows, .cols = xv.cols, .x_row = xv.row_stride, .o_row = ov.row_stride };
        try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ xv.rows, 1, 1 });
    }
}

/// LayerNorm/RMSNorm over the trailing dim (norm_rank == 1): gamma/beta are
/// single-tile rank-1 vectors of the full row width.
pub fn execNorm(ctx: Ctx, frame: *Frame, mode: NormMode, s: anytype) ExecuteProgramError!void {
    const hs = ctx.store;
    const meta = hs.meta(s.out) catch return error.ExecutionFailed;
    const g_meta = hs.meta(s.gamma) catch return error.ExecutionFailed;
    const b_meta = hs.meta(s.beta) catch return error.ExecutionFailed;
    try requireScalarFloat(meta.dtype);
    if (g_meta.dtype != meta.dtype or b_meta.dtype != meta.dtype) return error.Unsupported;

    const rank: usize = @as(usize, meta.rank);
    if (rank < 2) return error.Unsupported;
    // v1: normalize over exactly the last dim, whole row in one tile, and the
    // gamma/beta vectors each in one tile.
    if (g_meta.rank != 1 or b_meta.rank != 1) return error.Unsupported;
    if (meta.tile_counts[rank - 1] != 1) return execNormCrossTile(ctx, frame, mode, s, meta, g_meta, b_meta, rank);
    if (g_meta.tile_counts[0] != 1 or b_meta.tile_counts[0] != 1) return error.Unsupported;

    const entry: [:0]const u8 = if (meta.dtype == .f16) switch (mode) {
        .rmsnorm => "rmsnorm_row_f16",
        .layernorm => "layernorm_row_f16",
    } else switch (mode) {
        .rmsnorm => "rmsnorm_row",
        .layernorm => "layernorm_row",
    };
    const built = try ctx.pipes.get(norm_kernel, entry);

    const total = context.totalTiles(meta);
    var ti: usize = 0;
    while (ti < total) : (ti += 1) {
        const dx = ctx.store.acquireTileDeviceConstLinear(s.x, ti) catch return error.ExecutionFailed;
        const dg = ctx.store.acquireTileDeviceConstLinear(s.gamma, 0) catch return error.ExecutionFailed;
        const db = ctx.store.acquireTileDeviceConstLinear(s.beta, 0) catch return error.ExecutionFailed;
        const dout = ctx.store.acquireTileDeviceMutLinear(s.out, ti) catch return error.ExecutionFailed;
        defer {
            hs.releaseConst(dx.token);
            hs.releaseConst(dg.token);
            hs.releaseConst(db.token);
            hs.releaseMut(dout.token);
        }
        if (!context.storageBindingFits(ctx, dx.len) or !context.storageBindingFits(ctx, dout.len)) return error.Unsupported;

        const xv = try deviceRowViewDt(dx, meta.dtype);
        const ov = try deviceRowViewDt(dout, meta.dtype);
        if (xv.rows != ov.rows or xv.cols != ov.cols) return error.Unsupported;
        if (xv.rows > context.MAX_GROUPS_PER_DIM) return error.Unsupported;
        if (dg.shape_mem[0] < xv.cols or db.shape_mem[0] < xv.cols) return error.Unsupported;

        const bufs = [_]c.WGPUBuffer{
            ctx.devmem.bufferFor(dx.handle).?,
            ctx.devmem.bufferFor(dg.handle).?,
            ctx.devmem.bufferFor(db.handle).?,
            ctx.devmem.bufferFor(dout.handle).?,
        };
        const sizes = [_]u64{ dx.len, dg.len, db.len, dout.len };
        const params: NormParams = .{ .rows = xv.rows, .cols = xv.cols, .x_row = xv.row_stride, .o_row = ov.row_stride, .eps = s.eps };
        try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ xv.rows, 1, 1 });
    }
}

/// `StepRMSNormTiled` WITH a residual: o = residual + rmsnorm(x)·gamma + beta, one
/// workgroup per row. Same contract as `execNorm` plus a residual of identical shape AND
/// tiling.
///
/// There is deliberately no cross-tile fallback: `validateStep` only accepts a residual
/// when the last dim is whole in one tile, and `fuse_steps` only sets one then. A split row
/// would need the two-stage reduce, and at that point the pair is not worth fusing.
pub fn execAddNorm(ctx: Ctx, frame: *Frame, s: executable.StepRMSNormTiled) ExecuteProgramError!void {
    const hs = ctx.store;
    const meta = hs.meta(s.out) catch return error.ExecutionFailed;
    const g_meta = hs.meta(s.gamma) catch return error.ExecutionFailed;
    const b_meta = hs.meta(s.beta) catch return error.ExecutionFailed;
    try requireF32(meta.dtype);
    try requireF32(g_meta.dtype);
    try requireF32(b_meta.dtype);

    const rank: usize = @as(usize, meta.rank);
    if (rank < 2) return error.Unsupported;
    if (g_meta.rank != 1 or b_meta.rank != 1) return error.Unsupported;
    if (meta.tile_counts[rank - 1] != 1) return error.Unsupported;
    if (g_meta.tile_counts[0] != 1 or b_meta.tile_counts[0] != 1) return error.Unsupported;

    const built = try ctx.pipes.get(add_norm_kernel, "add_rmsnorm_row");
    const residual = s.residual orelse return error.Unsupported;

    const total = context.totalTiles(meta);
    var ti: usize = 0;
    while (ti < total) : (ti += 1) {
        const dx = ctx.store.acquireTileDeviceConstLinear(s.x, ti) catch return error.ExecutionFailed;
        const dg = ctx.store.acquireTileDeviceConstLinear(s.gamma, 0) catch return error.ExecutionFailed;
        const db = ctx.store.acquireTileDeviceConstLinear(s.beta, 0) catch return error.ExecutionFailed;
        const da = ctx.store.acquireTileDeviceConstLinear(residual, ti) catch return error.ExecutionFailed;
        const dout = ctx.store.acquireTileDeviceMutLinear(s.out, ti) catch return error.ExecutionFailed;
        defer {
            hs.releaseConst(dx.token);
            hs.releaseConst(dg.token);
            hs.releaseConst(db.token);
            hs.releaseConst(da.token);
            hs.releaseMut(dout.token);
        }
        if (!context.storageBindingFits(ctx, dx.len) or !context.storageBindingFits(ctx, da.len) or
            !context.storageBindingFits(ctx, dout.len)) return error.Unsupported;

        const xv = try deviceRowView(dx);
        const av = try deviceRowView(da);
        const ov = try deviceRowView(dout);
        if (xv.rows != ov.rows or xv.cols != ov.cols) return error.Unsupported;
        if (av.rows != ov.rows or av.cols != ov.cols) return error.Unsupported;
        if (xv.rows > context.MAX_GROUPS_PER_DIM) return error.Unsupported;
        if (dg.shape_mem[0] < xv.cols or db.shape_mem[0] < xv.cols) return error.Unsupported;

        const bufs = [_]c.WGPUBuffer{
            ctx.devmem.bufferFor(dx.handle).?,
            ctx.devmem.bufferFor(dg.handle).?,
            ctx.devmem.bufferFor(db.handle).?,
            ctx.devmem.bufferFor(da.handle).?,
            ctx.devmem.bufferFor(dout.handle).?,
        };
        const sizes = [_]u64{ dx.len, dg.len, db.len, da.len, dout.len };
        const params: AddNormParams = .{
            .rows = xv.rows,
            .cols = xv.cols,
            .x_row = xv.row_stride,
            .o_row = ov.row_stride,
            .a_row = av.row_stride,
            .eps = s.eps,
        };
        try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ xv.rows, 1, 1 });
    }
}

fn crossRowLayout(input_meta: anytype, output_meta: anytype, rank: usize) ExecuteProgramError!struct { parts: usize, groups: usize } {
    if (@as(usize, output_meta.rank) != rank) return error.Unsupported;
    const parts = input_meta.tile_counts[rank - 1];
    if (parts <= 1 or output_meta.tile_counts[rank - 1] != parts) return error.Unsupported;
    var d: usize = 0;
    while (d < rank) : (d += 1) {
        if (input_meta.tile_counts[d] != output_meta.tile_counts[d] or input_meta.tile_shape[d] != output_meta.tile_shape[d]) return error.Unsupported;
    }
    const total = context.totalTiles(input_meta);
    if (total != context.totalTiles(output_meta) or total % parts != 0) return error.Unsupported;
    return .{ .parts = parts, .groups = total / parts };
}

fn crossParams(rows: u32, cols: u32, x_row: u32, parts: usize, part: usize, col_base: usize, full_cols: usize, stat_base: usize) ExecuteProgramError!CrossParams {
    return .{
        .rows = rows,
        .cols = cols,
        .x_row = x_row,
        .parts = std.math.cast(u32, parts) orelse return error.Unsupported,
        .part = std.math.cast(u32, part) orelse return error.Unsupported,
        .col_base = std.math.cast(u32, col_base) orelse return error.Unsupported,
        .full_cols = std.math.cast(u32, full_cols) orelse return error.Unsupported,
        .stat_base = std.math.cast(u32, stat_base) orelse return error.Unsupported,
    };
}

fn execSoftmaxCrossTile(ctx: Ctx, frame: *Frame, s: executable.StepSoftmaxTiled, meta: anytype, rank: usize) ExecuteProgramError!void {
    const hs = ctx.store;
    const out_meta = hs.meta(s.out) catch return error.ExecutionFailed;
    const layout = try crossRowLayout(meta, out_meta, rank);
    const f16_rows = meta.dtype == .f16;
    const partial_max = try ctx.pipes.get(rowwise_partial_kernel, if (f16_rows) "softmax_max_partial_f16" else "softmax_max_partial");
    const finish_max = try ctx.pipes.get(rowwise_finish_kernel, "softmax_max_finish");
    const partial_exp = try ctx.pipes.get(rowwise_partial_kernel, if (f16_rows) "softmax_exp_partial_f16" else "softmax_exp_partial");
    const finish_sum = try ctx.pipes.get(rowwise_finish_kernel, "softmax_sum_finish");
    const apply = try ctx.pipes.get(softmax_cross_apply_kernel, if (f16_rows) "softmax_apply_f16" else "softmax_apply");

    var group: usize = 0;
    while (group < layout.groups) : (group += 1) {
        const first = ctx.store.acquireTileDeviceConstLinear(s.a, group * layout.parts) catch return error.ExecutionFailed;
        const first_view = try deviceRowViewDt(first, meta.dtype);
        hs.releaseConst(first.token);
        const rows = first_view.rows;
        if (rows == 0 or rows > context.MAX_GROUPS_PER_DIM) return error.Unsupported;
        const partial_count = @as(usize, rows) * layout.parts;
        const stat_base = partial_count;
        const scratch_bytes: u64 = @as(u64, @intCast(partial_count + @as(usize, rows) * 2)) * @sizeOf(f32);
        if (!context.storageBindingFits(ctx, scratch_bytes)) return error.Unsupported;
        const scratch = try ctx.scratch.ensure(ctx.gpu, scratch_bytes);

        // Tile maxima.
        var part: usize = 0;
        while (part < layout.parts) : (part += 1) {
            const dx = ctx.store.acquireTileDeviceConstLinear(s.a, group * layout.parts + part) catch return error.ExecutionFailed;
            defer hs.releaseConst(dx.token);
            const xv = try deviceRowViewDt(dx, meta.dtype);
            if (xv.rows != rows) return error.Unsupported;
            const params = try crossParams(rows, xv.cols, xv.row_stride, layout.parts, part, part * meta.tile_shape[rank - 1], meta.shape[rank - 1], stat_base);
            const bufs = [_]c.WGPUBuffer{ ctx.devmem.bufferFor(dx.handle).?, scratch };
            const sizes = [_]u64{ dx.len, scratch_bytes };
            try frame.recordCompute(partial_max, &bufs, &sizes, std.mem.asBytes(&params), .{ rows, 1, 1 });
        }
        var finish_params = try crossParams(rows, 0, 0, layout.parts, 0, 0, meta.shape[rank - 1], stat_base);
        try frame.recordCompute(finish_max, &.{scratch}, &.{scratch_bytes}, std.mem.asBytes(&finish_params), .{ rows, 1, 1 });

        // Exponential sums relative to the completed global maximum.
        part = 0;
        while (part < layout.parts) : (part += 1) {
            const dx = ctx.store.acquireTileDeviceConstLinear(s.a, group * layout.parts + part) catch return error.ExecutionFailed;
            defer hs.releaseConst(dx.token);
            const xv = try deviceRowViewDt(dx, meta.dtype);
            const params = try crossParams(rows, xv.cols, xv.row_stride, layout.parts, part, part * meta.tile_shape[rank - 1], meta.shape[rank - 1], stat_base);
            const bufs = [_]c.WGPUBuffer{ ctx.devmem.bufferFor(dx.handle).?, scratch };
            const sizes = [_]u64{ dx.len, scratch_bytes };
            try frame.recordCompute(partial_exp, &bufs, &sizes, std.mem.asBytes(&params), .{ rows, 1, 1 });
        }
        try frame.recordCompute(finish_sum, &.{scratch}, &.{scratch_bytes}, std.mem.asBytes(&finish_params), .{ rows, 1, 1 });

        // Normalize each output tile with the shared row statistics.
        part = 0;
        while (part < layout.parts) : (part += 1) {
            const tile_index = group * layout.parts + part;
            const dx = ctx.store.acquireTileDeviceConstLinear(s.a, tile_index) catch return error.ExecutionFailed;
            const dout = ctx.store.acquireTileDeviceMutLinear(s.out, tile_index) catch return error.ExecutionFailed;
            defer {
                hs.releaseConst(dx.token);
                hs.releaseMut(dout.token);
            }
            const xv = try deviceRowViewDt(dx, meta.dtype);
            const ov = try deviceRowViewDt(dout, meta.dtype);
            if (xv.rows != rows or ov.rows != rows or xv.cols != ov.cols) return error.Unsupported;
            var params = try crossParams(rows, xv.cols, xv.row_stride, layout.parts, part, part * meta.tile_shape[rank - 1], meta.shape[rank - 1], stat_base);
            params.o_row = ov.row_stride;
            const bufs = [_]c.WGPUBuffer{ ctx.devmem.bufferFor(dx.handle).?, scratch, ctx.devmem.bufferFor(dout.handle).? };
            const sizes = [_]u64{ dx.len, scratch_bytes, dout.len };
            try frame.recordCompute(apply, &bufs, &sizes, std.mem.asBytes(&params), .{ rows, 1, 1 });
        }
    }
}

fn execNormCrossTile(ctx: Ctx, frame: *Frame, mode: NormMode, s: anytype, meta: anytype, g_meta: anytype, b_meta: anytype, rank: usize) ExecuteProgramError!void {
    const hs = ctx.store;
    const out_meta = hs.meta(s.out) catch return error.ExecutionFailed;
    const layout = try crossRowLayout(meta, out_meta, rank);
    const gamma_single = g_meta.tile_counts[0] == 1;
    const beta_single = b_meta.tile_counts[0] == 1;
    if (gamma_single != beta_single) return error.Unsupported;
    if (!gamma_single and g_meta.tile_counts[0] != layout.parts) return error.Unsupported;
    if (!beta_single and b_meta.tile_counts[0] != layout.parts) return error.Unsupported;
    if (g_meta.shape[0] < meta.shape[rank - 1] or b_meta.shape[0] < meta.shape[rank - 1]) return error.Unsupported;

    const f16_rows = meta.dtype == .f16;
    const partial = try ctx.pipes.get(rowwise_partial_kernel, if (f16_rows) "norm_moments_partial_f16" else "norm_moments_partial");
    const finish = try ctx.pipes.get(rowwise_finish_kernel, "norm_finish");
    const apply = try ctx.pipes.get(norm_cross_apply_kernel, if (f16_rows) "norm_apply_f16" else "norm_apply");
    var group: usize = 0;
    while (group < layout.groups) : (group += 1) {
        const first = ctx.store.acquireTileDeviceConstLinear(s.x, group * layout.parts) catch return error.ExecutionFailed;
        const first_view = try deviceRowViewDt(first, meta.dtype);
        hs.releaseConst(first.token);
        const rows = first_view.rows;
        if (rows == 0 or rows > context.MAX_GROUPS_PER_DIM) return error.Unsupported;
        const partial_count = @as(usize, rows) * layout.parts * 2;
        const stat_base = partial_count;
        const scratch_bytes: u64 = @as(u64, @intCast(partial_count + @as(usize, rows) * 2)) * @sizeOf(f32);
        if (!context.storageBindingFits(ctx, scratch_bytes)) return error.Unsupported;
        const scratch = try ctx.scratch.ensure(ctx.gpu, scratch_bytes);

        var part: usize = 0;
        while (part < layout.parts) : (part += 1) {
            const dx = ctx.store.acquireTileDeviceConstLinear(s.x, group * layout.parts + part) catch return error.ExecutionFailed;
            defer hs.releaseConst(dx.token);
            const xv = try deviceRowViewDt(dx, meta.dtype);
            if (xv.rows != rows) return error.Unsupported;
            const params = try crossParams(rows, xv.cols, xv.row_stride, layout.parts, part, part * meta.tile_shape[rank - 1], meta.shape[rank - 1], stat_base);
            const bufs = [_]c.WGPUBuffer{ ctx.devmem.bufferFor(dx.handle).?, scratch };
            const sizes = [_]u64{ dx.len, scratch_bytes };
            try frame.recordCompute(partial, &bufs, &sizes, std.mem.asBytes(&params), .{ rows, 1, 1 });
        }

        var finish_params = try crossParams(rows, 0, 0, layout.parts, 0, 0, meta.shape[rank - 1], stat_base);
        finish_params.mode = if (mode == .layernorm) 1 else 0;
        finish_params.eps = s.eps;
        try frame.recordCompute(finish, &.{scratch}, &.{scratch_bytes}, std.mem.asBytes(&finish_params), .{ rows, 1, 1 });

        part = 0;
        while (part < layout.parts) : (part += 1) {
            const tile_index = group * layout.parts + part;
            const dx = ctx.store.acquireTileDeviceConstLinear(s.x, tile_index) catch return error.ExecutionFailed;
            const dg = ctx.store.acquireTileDeviceConstLinear(s.gamma, if (gamma_single) 0 else part) catch return error.ExecutionFailed;
            const db = ctx.store.acquireTileDeviceConstLinear(s.beta, if (beta_single) 0 else part) catch return error.ExecutionFailed;
            const dout = ctx.store.acquireTileDeviceMutLinear(s.out, tile_index) catch return error.ExecutionFailed;
            defer {
                hs.releaseConst(dx.token);
                hs.releaseConst(dg.token);
                hs.releaseConst(db.token);
                hs.releaseMut(dout.token);
            }
            const xv = try deviceRowViewDt(dx, meta.dtype);
            const ov = try deviceRowViewDt(dout, meta.dtype);
            if (xv.rows != rows or ov.rows != rows or xv.cols != ov.cols) return error.Unsupported;
            var params = try crossParams(rows, xv.cols, xv.row_stride, layout.parts, part, if (gamma_single) part * meta.tile_shape[rank - 1] else 0, meta.shape[rank - 1], stat_base);
            params.o_row = ov.row_stride;
            params.mode = finish_params.mode;
            params.eps = s.eps;
            const bufs = [_]c.WGPUBuffer{
                ctx.devmem.bufferFor(dx.handle).?,
                ctx.devmem.bufferFor(dg.handle).?,
                ctx.devmem.bufferFor(db.handle).?,
                scratch,
                ctx.devmem.bufferFor(dout.handle).?,
            };
            const sizes = [_]u64{ dx.len, dg.len, db.len, scratch_bytes, dout.len };
            try frame.recordCompute(apply, &bufs, &sizes, std.mem.asBytes(&params), .{ rows, 1, 1 });
        }
    }
}

/// Entry point for a single-dispatch row reduce: data in, result out, same dtype.
fn reduceEntry(op: types.ReduceOp, dtype: types.DType) [:0]const u8 {
    const f16_data = dtype == .f16;
    return switch (op) {
        .sum => if (f16_data) "reduce_sum_row_h2h" else "reduce_sum_row",
        .mean => if (f16_data) "reduce_mean_row_h2h" else "reduce_mean_row",
    };
}

/// Whole-tensor reduce to a [1] scalar. Small tensors run the single-workgroup
/// row kernel; anything past a few workgroups' worth goes TWO-STAGE — many
/// workgroups write per-workgroup partial sums into the shared scratch pool,
/// then one workgroup folds the partials (a single WG striding megabytes runs
/// at ~1% of DRAM bandwidth). v1: single-tile input and output.
pub fn execReduceAll(ctx: Ctx, frame: *Frame, s: executable.StepReduceAll) ExecuteProgramError!void {
    const hs = ctx.store;
    const meta = hs.meta(s.a) catch return error.ExecutionFailed;
    try requireScalarFloat(meta.dtype);
    const out_meta = hs.meta(s.out) catch return error.ExecutionFailed;
    if (out_meta.dtype != meta.dtype) return error.Unsupported;
    // Staged reductions keep their partials in f32 whatever the data dtype, so
    // stage 1 is data->f32 (`_h2f`) and the fold is f32->data (`_f2h`).
    const f16_data = meta.dtype == .f16;
    const tile_count = context.totalTiles(meta);
    if (context.totalTiles(out_meta) != 1) return error.Unsupported;
    if (tile_count != 1) return execReduceAllCrossTile(ctx, frame, s, tile_count);

    const dx = ctx.store.acquireTileDeviceConstLinear(s.a, 0) catch return error.ExecutionFailed;
    const dout = ctx.store.acquireTileDeviceMutLinear(s.out, 0) catch return error.ExecutionFailed;
    defer {
        hs.releaseConst(dx.token);
        hs.releaseMut(dout.token);
    }
    if (!context.storageBindingFits(ctx, dx.len)) return error.Unsupported;

    const n = try devicePackedElemsDt(dx, meta.dtype);
    const x_buf = ctx.devmem.bufferFor(dx.handle).?;
    const o_buf = ctx.devmem.bufferFor(dout.handle).?;
    const sizes_xo = [_]u64{ dx.len, dout.len };

    // Each stage-1 thread should sum a few dozen elements; below ~2 workgroups
    // of work the single-stage kernel wins on dispatch overhead.
    const WG: u32 = 256;
    const PER_THREAD: u32 = 32;
    const groups: u32 = @min(context.ceilDiv(n, WG * PER_THREAD), 1024);
    if (groups <= 2) {
        const built = try ctx.pipes.get(reduce_kernel, reduceEntry(s.op, meta.dtype));
        const bufs = [_]c.WGPUBuffer{ x_buf, o_buf };
        const params: RowParams = .{ .rows = 1, .cols = n, .x_row = 0 };
        try frame.recordCompute(built, &bufs, &sizes_xo, std.mem.asBytes(&params), .{ 1, 1, 1 });
        return;
    }

    const scratch = try ctx.scratch.ensure(ctx.gpu, @as(u64, groups) * @sizeOf(f32));

    // Stage 1: partial sums, one f32 per workgroup.
    {
        const built = try ctx.pipes.get(reduce_kernel, if (f16_data) "reduce_all_partial_h2f" else "reduce_all_partial");
        const bufs = [_]c.WGPUBuffer{ x_buf, scratch };
        const sizes = [_]u64{ dx.len, @as(u64, groups) * @sizeOf(f32) };
        const params: RowParams = .{ .rows = 1, .cols = n, .x_row = 0 };
        try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups, 1, 1 });
    }
    // Stage 2: fold the partials; mean divides by the ORIGINAL n (in x_row).
    {
        const entry: [:0]const u8 = if (f16_data) switch (s.op) {
            .sum => "reduce_sum_row_f2h",
            .mean => "reduce_mean_finish_f2h",
        } else switch (s.op) {
            .sum => "reduce_sum_row",
            .mean => "reduce_mean_finish",
        };
        const built = try ctx.pipes.get(reduce_kernel, entry);
        const bufs = [_]c.WGPUBuffer{ scratch, o_buf };
        const sizes = [_]u64{ @as(u64, groups) * @sizeOf(f32), dout.len };
        const params: RowParams = .{ .rows = 1, .cols = groups, .x_row = n };
        try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ 1, 1, 1 });
    }
}

fn execReduceAllCrossTile(ctx: Ctx, frame: *Frame, s: executable.StepReduceAll, tile_count: usize) ExecuteProgramError!void {
    const hs = ctx.store;
    const WG: u32 = 256;
    const PER_THREAD: u32 = 32;
    // Partials are f32 for either data dtype (see reduce.wgsl), so stage 1 is
    // data->f32 and the fold is f32->data.
    const data_dtype = (hs.meta(s.a) catch return error.ExecutionFailed).dtype;
    const f16_data = data_dtype == .f16;

    var partial_count: usize = 0;
    var logical_count: usize = 0;
    var tile_index: usize = 0;
    while (tile_index < tile_count) : (tile_index += 1) {
        const dx = ctx.store.acquireTileDeviceConstLinear(s.a, tile_index) catch return error.ExecutionFailed;
        const n = try devicePackedElemsDt(dx, data_dtype);
        hs.releaseConst(dx.token);
        partial_count = std.math.add(usize, partial_count, @min(context.ceilDiv(n, WG * PER_THREAD), 1024)) catch return error.Unsupported;
        logical_count = std.math.add(usize, logical_count, n) catch return error.Unsupported;
    }
    if (partial_count == 0 or logical_count == 0) return error.Unsupported;

    const scratch_bytes: u64 = @as(u64, @intCast(partial_count)) * @sizeOf(f32);
    if (!context.storageBindingFits(ctx, scratch_bytes)) return error.Unsupported;
    const scratch = try ctx.scratch.ensure(ctx.gpu, scratch_bytes);
    const partial_kernel = try ctx.pipes.get(reduce_kernel, if (f16_data) "reduce_all_partial_h2f" else "reduce_all_partial");

    var partial_base: usize = 0;
    tile_index = 0;
    while (tile_index < tile_count) : (tile_index += 1) {
        const dx = ctx.store.acquireTileDeviceConstLinear(s.a, tile_index) catch return error.ExecutionFailed;
        defer hs.releaseConst(dx.token);
        const n = try devicePackedElemsDt(dx, data_dtype);
        const groups = @min(context.ceilDiv(n, WG * PER_THREAD), 1024);
        const params: RowParams = .{
            .rows = 1,
            .cols = n,
            .x_row = 0,
            .o_row = std.math.cast(u32, partial_base) orelse return error.Unsupported,
        };
        const bufs = [_]c.WGPUBuffer{ ctx.devmem.bufferFor(dx.handle).?, scratch };
        const sizes = [_]u64{ dx.len, scratch_bytes };
        try frame.recordCompute(partial_kernel, &bufs, &sizes, std.mem.asBytes(&params), .{ groups, 1, 1 });
        partial_base += groups;
    }

    const dout = ctx.store.acquireTileDeviceMutLinear(s.out, 0) catch return error.ExecutionFailed;
    defer hs.releaseMut(dout.token);
    const finish_entry: [:0]const u8 = if (f16_data) switch (s.op) {
        .sum => "reduce_sum_row_f2h",
        .mean => "reduce_mean_finish_f2h",
    } else switch (s.op) {
        .sum => "reduce_sum_row",
        .mean => "reduce_mean_finish",
    };
    const finish = try ctx.pipes.get(reduce_kernel, finish_entry);
    const params: RowParams = .{
        .rows = 1,
        .cols = std.math.cast(u32, partial_count) orelse return error.Unsupported,
        .x_row = std.math.cast(u32, logical_count) orelse return error.Unsupported,
    };
    const bufs = [_]c.WGPUBuffer{ scratch, ctx.devmem.bufferFor(dout.handle).? };
    const sizes = [_]u64{ scratch_bytes, dout.len };
    try frame.recordCompute(finish, &bufs, &sizes, std.mem.asBytes(&params), .{ 1, 1, 1 });
}

/// ArgMax over the last axis: o[row] (i32) = index of the row max, lowest index
/// on ties. Column-tiled rows emit global-index partials and fold them once.
pub fn execArgMax(ctx: Ctx, frame: *Frame, s: executable.StepArgMax) ExecuteProgramError!void {
    const hs = ctx.store;
    const meta = hs.meta(s.a) catch return error.ExecutionFailed;
    // f32 or f16 in, i32 index out. Widened comparison preserves f16 ordering and
    // the lowest-index tie-break, so both dtypes pick the same column.
    try requireScalarFloat(meta.dtype);
    const out_meta = hs.meta(s.out) catch return error.ExecutionFailed;
    if (out_meta.dtype != .i32) return error.Unsupported;

    const rank: usize = @as(usize, meta.rank);
    if (rank == 0 or s.axis != rank - 1) return error.Unsupported; // v1: last axis only
    const input_tiles = context.totalTiles(meta);
    const output_tiles = context.totalTiles(out_meta);
    if (input_tiles != 1) return execArgMaxCrossTile(ctx, frame, s, meta, out_meta, rank, input_tiles, output_tiles);
    if (output_tiles != 1) return error.Unsupported;

    const built = try ctx.pipes.get(argmax_kernel, if (meta.dtype == .f16) "argmax_row_f16" else "argmax_row");

    const dx = ctx.store.acquireTileDeviceConstLinear(s.a, 0) catch return error.ExecutionFailed;
    const dout = ctx.store.acquireTileDeviceMutLinear(s.out, 0) catch return error.ExecutionFailed;
    defer {
        hs.releaseConst(dx.token);
        hs.releaseMut(dout.token);
    }
    if (!context.storageBindingFits(ctx, dx.len)) return error.Unsupported;

    const xv = try deviceRowViewDt(dx, meta.dtype);
    const out_rank: usize = @as(usize, dout.rank);
    const out_n = context.packedElemsSized(dout.rank, dout.shape_mem[0..out_rank], dout.strides_mem[0..out_rank], @sizeOf(i32)) orelse return error.Unsupported;
    if (out_n != xv.rows) return error.Unsupported;
    if (xv.rows > context.MAX_GROUPS_PER_DIM) return error.Unsupported;

    const x_buf = ctx.devmem.bufferFor(dx.handle).?;
    const o_buf = ctx.devmem.bufferFor(dout.handle).?;

    // Decode shape: FEW rows over a whole vocab. One workgroup per row leaves
    // the GPU idle, so split wide rows into column segments (stage 1 writes
    // (value, column) pairs into scratch, stage 2 folds per row).
    const SEG_COLS: u32 = 4096;
    const segs: u32 = context.ceilDiv(xv.cols, SEG_COLS);
    if (segs > 1 and segs <= context.MAX_GROUPS_PER_DIM) {
        const entries: u64 = @as(u64, xv.rows) * segs;
        const scratch = try ctx.scratch.ensure(ctx.gpu, entries * 2 * @sizeOf(f32));
        {
            const b1 = try ctx.pipes.get(argmax_kernel, if (meta.dtype == .f16) "argmax_partial_f16" else "argmax_partial");
            const bufs = [_]c.WGPUBuffer{ x_buf, scratch };
            const sizes = [_]u64{ dx.len, entries * 2 * @sizeOf(f32) };
            const params: RowParams = .{ .rows = xv.rows, .cols = xv.cols, .x_row = xv.row_stride, .o_row = SEG_COLS };
            try frame.recordCompute(b1, &bufs, &sizes, std.mem.asBytes(&params), .{ segs, xv.rows, 1 });
        }
        {
            const b2 = try ctx.pipes.get(argmax_kernel, "argmax_finish");
            const bufs = [_]c.WGPUBuffer{ scratch, o_buf };
            const sizes = [_]u64{ entries * 2 * @sizeOf(f32), dout.len };
            const params: RowParams = .{ .rows = xv.rows, .cols = segs, .x_row = 0 };
            try frame.recordCompute(b2, &bufs, &sizes, std.mem.asBytes(&params), .{ xv.rows, 1, 1 });
        }
        return;
    }

    const bufs = [_]c.WGPUBuffer{ x_buf, o_buf };
    const sizes = [_]u64{ dx.len, dout.len };
    const params: RowParams = .{ .rows = xv.rows, .cols = xv.cols, .x_row = xv.row_stride };
    try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ xv.rows, 1, 1 });
}

fn execArgMaxCrossTile(ctx: Ctx, frame: *Frame, s: executable.StepArgMax, meta: anytype, out_meta: anytype, rank: usize, input_tiles: usize, output_tiles: usize) ExecuteProgramError!void {
    const hs = ctx.store;
    // Reducing a vector keeps the API's scalar representation, shape [1].
    // Higher ranks remove the reduced last dimension.
    const output_rank = if (rank == 1) 1 else rank - 1;
    if (@as(usize, out_meta.rank) != output_rank) return error.Unsupported;
    const parts = meta.tile_counts[rank - 1];
    if (parts <= 1 or input_tiles % parts != 0) return error.Unsupported;
    const groups = input_tiles / parts;
    if (output_tiles != groups) return error.Unsupported;
    var d: usize = 0;
    while (d + 1 < rank) : (d += 1) {
        if (meta.tile_counts[d] != out_meta.tile_counts[d]) return error.Unsupported;
    }

    const partial = try ctx.pipes.get(argmax_cross_kernel, "argmax_tile_partial");
    const finish = try ctx.pipes.get(argmax_cross_kernel, "argmax_tile_finish");
    var group: usize = 0;
    while (group < groups) : (group += 1) {
        const dout = ctx.store.acquireTileDeviceMutLinear(s.out, group) catch return error.ExecutionFailed;
        defer hs.releaseMut(dout.token);
        const rows = try devicePackedElems(dout);
        if (rows == 0 or rows > context.MAX_GROUPS_PER_DIM) return error.Unsupported;
        const scratch_bytes: u64 = @as(u64, rows) * @as(u64, @intCast(parts)) * 2 * @sizeOf(u32);
        if (!context.storageBindingFits(ctx, scratch_bytes)) return error.Unsupported;
        const scratch = try ctx.scratch.ensure(ctx.gpu, scratch_bytes);

        var part: usize = 0;
        while (part < parts) : (part += 1) {
            const dx = ctx.store.acquireTileDeviceConstLinear(s.a, group * parts + part) catch return error.ExecutionFailed;
            defer hs.releaseConst(dx.token);
            const xv = try deviceRowViewDt(dx, meta.dtype);
            if (xv.rows != rows) return error.Unsupported;
            const params: ArgmaxCrossParams = .{
                .rows = rows,
                .cols = xv.cols,
                .x_row = xv.row_stride,
                .parts = std.math.cast(u32, parts) orelse return error.Unsupported,
                .part = std.math.cast(u32, part) orelse return error.Unsupported,
                .col_base = std.math.cast(u32, part * meta.tile_shape[rank - 1]) orelse return error.Unsupported,
            };
            const bufs = [_]c.WGPUBuffer{ ctx.devmem.bufferFor(dx.handle).?, scratch };
            const sizes = [_]u64{ dx.len, scratch_bytes };
            try frame.recordCompute(partial, &bufs, &sizes, std.mem.asBytes(&params), .{ rows, 1, 1 });
        }
        const params: ArgmaxCrossParams = .{
            .rows = rows,
            .cols = 0,
            .x_row = 0,
            .parts = std.math.cast(u32, parts) orelse return error.Unsupported,
        };
        const bufs = [_]c.WGPUBuffer{ scratch, ctx.devmem.bufferFor(dout.handle).? };
        const sizes = [_]u64{ scratch_bytes, dout.len };
        try frame.recordCompute(finish, &bufs, &sizes, std.mem.asBytes(&params), .{ rows, 1, 1 });
    }
}

/// Reduce over the last axis: o[row] = sum/mean of that row. v1: single-tile
/// input and output (the output row order must match the input's flat row index).
pub fn execReduceAxis(ctx: Ctx, frame: *Frame, s: executable.StepReduceAxis) ExecuteProgramError!void {
    const hs = ctx.store;
    const meta = hs.meta(s.a) catch return error.ExecutionFailed;
    const out_meta = hs.meta(s.out) catch return error.ExecutionFailed;
    if (meta.dtype != out_meta.dtype) return error.Unsupported;
    switch (meta.dtype) {
        .f32, .f16 => {},
        .i32 => if (s.op != .sum) return error.Unsupported,
        else => return error.Unsupported,
    }

    const rank: usize = @as(usize, meta.rank);
    if (rank >= 2 and s.axis != rank - 1) return error.Unsupported; // v1: last axis only
    const input_tiles = context.totalTiles(meta);
    const output_tiles = context.totalTiles(out_meta);
    if (input_tiles != 1) {
        // i32 has no staged path (its partials would need an i32 scratch); f32 and
        // f16 both stage through f32 partials.
        if (meta.dtype == .i32) return error.Unsupported;
        return execReduceAxisCrossTile(ctx, frame, s, meta, out_meta, rank, input_tiles, output_tiles);
    }
    if (output_tiles != 1) return error.Unsupported;

    const built = if (meta.dtype == .i32)
        try ctx.pipes.get(reduce_i32_kernel, "reduce_sum_row")
    else
        try ctx.pipes.get(reduce_kernel, reduceEntry(s.op, meta.dtype));

    const dx = ctx.store.acquireTileDeviceConstLinear(s.a, 0) catch return error.ExecutionFailed;
    const dout = ctx.store.acquireTileDeviceMutLinear(s.out, 0) catch return error.ExecutionFailed;
    defer {
        hs.releaseConst(dx.token);
        hs.releaseMut(dout.token);
    }
    if (!context.storageBindingFits(ctx, dx.len)) return error.Unsupported;

    const xv = if (meta.dtype == .i32) try deviceRowView(dx) else try deviceRowViewDt(dx, meta.dtype);
    // i32 sum uses the same 4-byte row layout as f32. Keep it out of
    // `scalarBytes`, which intentionally accepts only floating-point dtypes for
    // the f16/f32 row-wise kernels above.
    const out_elem_bytes: usize = if (out_meta.dtype == .i32) @sizeOf(i32) else try scalarBytes(out_meta.dtype);
    const out_n = context.packedElemsSized(dout.rank, dout.shape_mem[0..@as(usize, dout.rank)], dout.strides_mem[0..@as(usize, dout.rank)], out_elem_bytes) orelse return error.Unsupported;
    if (out_n != xv.rows) return error.Unsupported;
    if (xv.rows > context.MAX_GROUPS_PER_DIM) return error.Unsupported;

    const bufs = [_]c.WGPUBuffer{
        ctx.devmem.bufferFor(dx.handle).?,
        ctx.devmem.bufferFor(dout.handle).?,
    };
    const sizes = [_]u64{ dx.len, dout.len };
    const params: RowParams = .{ .rows = xv.rows, .cols = xv.cols, .x_row = xv.row_stride };
    try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ xv.rows, 1, 1 });
}

/// Reduce a last axis split across column tiles. Stage 1 writes one sum per
/// (column tile, local row) into part-major scratch; stage 2 folds the parts into
/// the corresponding output tile. Leading tile coordinates must match between
/// input and output, which is the layout the compiler's default tiler produces
/// after removing the reduced last dimension.
fn execReduceAxisCrossTile(
    ctx: Ctx,
    frame: *Frame,
    s: executable.StepReduceAxis,
    meta: anytype,
    out_meta: anytype,
    rank: usize,
    input_tiles: usize,
    output_tiles: usize,
) ExecuteProgramError!void {
    const hs = ctx.store;
    const parts = meta.tile_counts[rank - 1];
    if (parts <= 1 or input_tiles % parts != 0) return error.Unsupported;
    const row_groups = input_tiles / parts;
    const output_rank = if (rank == 1) 1 else rank - 1;
    if (output_tiles != row_groups or @as(usize, out_meta.rank) != output_rank) return error.Unsupported;
    var d: usize = 0;
    while (d + 1 < rank) : (d += 1) {
        if (out_meta.tile_counts[d] != meta.tile_counts[d]) return error.Unsupported;
    }

    // Stage 1 writes f32 partials whatever the data dtype; the fold writes back in
    // the tensor dtype.
    const f16_data = meta.dtype == .f16;
    const partial = try ctx.pipes.get(reduce_kernel, if (f16_data) "reduce_sum_row_h2f" else "reduce_sum_row");
    const finish = try ctx.pipes.get(reduce_kernel, if (f16_data) switch (s.op) {
        .sum => "reduce_parts_sum_f2h",
        .mean => "reduce_parts_mean_f2h",
    } else switch (s.op) {
        .sum => "reduce_parts_sum",
        .mean => "reduce_parts_mean",
    });

    var group: usize = 0;
    while (group < row_groups) : (group += 1) {
        const dout = ctx.store.acquireTileDeviceMutLinear(s.out, group) catch return error.ExecutionFailed;
        defer hs.releaseMut(dout.token);
        const rows = try devicePackedElemsDt(dout, meta.dtype);
        if (rows == 0 or rows > context.MAX_GROUPS_PER_DIM) return error.Unsupported;

        const scratch_bytes: u64 = @as(u64, rows) * @as(u64, @intCast(parts)) * @sizeOf(f32);
        if (!context.storageBindingFits(ctx, scratch_bytes)) return error.Unsupported;
        const scratch = try ctx.scratch.ensure(ctx.gpu, scratch_bytes);

        var part: usize = 0;
        while (part < parts) : (part += 1) {
            const input_index = group * parts + part;
            const dx = ctx.store.acquireTileDeviceConstLinear(s.a, input_index) catch return error.ExecutionFailed;
            defer hs.releaseConst(dx.token);
            if (!context.storageBindingFits(ctx, dx.len)) return error.Unsupported;
            const xv = try deviceRowViewDt(dx, meta.dtype);
            if (xv.rows != rows) return error.Unsupported;

            const params: RowParams = .{
                .rows = rows,
                .cols = xv.cols,
                .x_row = xv.row_stride,
                .o_row = std.math.cast(u32, part * @as(usize, rows)) orelse return error.Unsupported,
            };
            const bufs = [_]c.WGPUBuffer{ ctx.devmem.bufferFor(dx.handle).?, scratch };
            const sizes = [_]u64{ dx.len, scratch_bytes };
            try frame.recordCompute(partial, &bufs, &sizes, std.mem.asBytes(&params), .{ rows, 1, 1 });
        }

        const params: RowParams = .{
            .rows = rows,
            .cols = std.math.cast(u32, parts) orelse return error.Unsupported,
            .x_row = rows,
            .o_row = std.math.cast(u32, meta.shape[rank - 1]) orelse return error.Unsupported,
        };
        const bufs = [_]c.WGPUBuffer{ scratch, ctx.devmem.bufferFor(dout.handle).? };
        const sizes = [_]u64{ scratch_bytes, dout.len };
        try frame.recordCompute(finish, &bufs, &sizes, std.mem.asBytes(&params), .{ rows, 1, 1 });
    }
}
