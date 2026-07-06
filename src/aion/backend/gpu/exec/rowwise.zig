// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Row-wise reduction ops for the GPU backend: softmax, LayerNorm/RMSNorm, and
//! sum/mean reductions. Each maps a tile to one dispatch of one 256-thread
//! workgroup PER ROW, with shared-memory tree reductions inside the workgroup
//! (kernels/softmax.wgsl, norm.wgsl, reduce.wgsl).
//!
//! v1 scope (f32): the reduced axis is the LAST dim and must live entirely in
//! one tile (`tile_counts[axis] == 1`). Under the GPU tile policy that holds for
//! rows up to `base_square_2d` (4096) columns — the common LLM shapes; wider
//! rows fall back with `error.Unsupported` (a cross-tile two-pass reduction is a
//! follow-up). Leading dims may tile freely: each tile is its own dispatch.

const std = @import("std");
const wgpu = @import("../wgpu.zig");
const pipelines = @import("../pipelines.zig");
const context = @import("../context.zig");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const executable = @import("../../../runtime/executable.zig");
const resident_mod = @import("../../../runtime/residency/resident_store.zig");

const c = wgpu.c;
const Ctx = context.Ctx;
const Frame = @import("../frame.zig").Frame;
const ExecuteProgramError = backend_mod.ExecuteProgramError;
const KernelDesc = pipelines.KernelDesc;
const TileRefDevice = resident_mod.TileRefDevice;

const softmax_kernel: KernelDesc = .{ .name = "softmax", .wgsl = @embedFile("../kernels/softmax.wgsl") };
const norm_kernel: KernelDesc = .{ .name = "norm", .wgsl = @embedFile("../kernels/norm.wgsl") };
const reduce_kernel: KernelDesc = .{ .name = "reduce", .wgsl = @embedFile("../kernels/reduce.wgsl") };
const argmax_kernel: KernelDesc = .{ .name = "argmax", .wgsl = @embedFile("../kernels/argmax.wgsl") };

/// Uniform params shared by softmax (`rows/cols/x_row/o_row`) and, prefix-wise,
/// reduce (`rows/cols/x_row`). Field order matches the WGSL structs.
const RowParams = extern struct { rows: u32, cols: u32, x_row: u32, o_row: u32 = 0 };
const NormParams = extern struct { rows: u32, cols: u32, x_row: u32, o_row: u32, eps: f32, _p0: u32 = 0, _p1: u32 = 0, _p2: u32 = 0 };

pub const NormMode = enum { rmsnorm, layernorm };

fn requireF32(dtype: types.DType) ExecuteProgramError!void {
    if (dtype != .f32) return error.Unsupported;
}

fn deviceRowView(t: TileRefDevice) ExecuteProgramError!context.RowView {
    return context.rowView(t.rank, t.shape_mem[0..@as(usize, t.rank)], t.strides_mem[0..@as(usize, t.rank)]) orelse error.Unsupported;
}

fn devicePackedElems(t: TileRefDevice) ExecuteProgramError!u32 {
    return context.packedElems(t.rank, t.shape_mem[0..@as(usize, t.rank)], t.strides_mem[0..@as(usize, t.rank)]) orelse error.Unsupported;
}

/// Softmax over the last axis. Negative axes were normalized by the graph API;
/// the step still carries i32, so normalize again here.
pub fn execSoftmax(ctx: Ctx, frame: *Frame, s: executable.StepSoftmaxTiled) ExecuteProgramError!void {
    const hs = ctx.rstore.tensorStore();
    const meta = hs.meta(s.out) catch return error.ExecutionFailed;
    try requireF32(meta.dtype);

    const rank: usize = @as(usize, meta.rank);
    var axis: i32 = s.axis;
    if (axis < 0) axis += @intCast(rank);
    if (axis < 0 or axis != @as(i32, @intCast(rank - 1))) return error.Unsupported; // v1: last axis only
    if (meta.tile_counts[rank - 1] != 1) return error.Unsupported; // reduced axis must be intra-tile

    const built = try ctx.pipes.get(softmax_kernel, "softmax_row");

    const total = context.totalTiles(meta);
    var ti: usize = 0;
    while (ti < total) : (ti += 1) {
        const dx = ctx.rstore.acquireTileDeviceConstLinear(s.a, ti) catch return error.ExecutionFailed;
        const dout = ctx.rstore.acquireTileDeviceMutLinear(s.out, ti) catch return error.ExecutionFailed;
        defer {
            hs.releaseConst(dx.token);
            hs.releaseMut(dout.token);
        }
        if (!context.storageBindingFits(ctx, dx.len) or !context.storageBindingFits(ctx, dout.len)) return error.Unsupported;

        const xv = try deviceRowView(dx);
        const ov = try deviceRowView(dout);
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
    const hs = ctx.rstore.tensorStore();
    const meta = hs.meta(s.out) catch return error.ExecutionFailed;
    const g_meta = hs.meta(s.gamma) catch return error.ExecutionFailed;
    const b_meta = hs.meta(s.beta) catch return error.ExecutionFailed;
    try requireF32(meta.dtype);
    try requireF32(g_meta.dtype);
    try requireF32(b_meta.dtype);

    const rank: usize = @as(usize, meta.rank);
    if (rank < 2) return error.Unsupported;
    // v1: normalize over exactly the last dim, whole row in one tile, and the
    // gamma/beta vectors each in one tile.
    if (g_meta.rank != 1 or b_meta.rank != 1) return error.Unsupported;
    if (meta.tile_counts[rank - 1] != 1) return error.Unsupported;
    if (g_meta.tile_counts[0] != 1 or b_meta.tile_counts[0] != 1) return error.Unsupported;

    const entry: [:0]const u8 = switch (mode) {
        .rmsnorm => "rmsnorm_row",
        .layernorm => "layernorm_row",
    };
    const built = try ctx.pipes.get(norm_kernel, entry);

    const total = context.totalTiles(meta);
    var ti: usize = 0;
    while (ti < total) : (ti += 1) {
        const dx = ctx.rstore.acquireTileDeviceConstLinear(s.x, ti) catch return error.ExecutionFailed;
        const dg = ctx.rstore.acquireTileDeviceConstLinear(s.gamma, 0) catch return error.ExecutionFailed;
        const db = ctx.rstore.acquireTileDeviceConstLinear(s.beta, 0) catch return error.ExecutionFailed;
        const dout = ctx.rstore.acquireTileDeviceMutLinear(s.out, ti) catch return error.ExecutionFailed;
        defer {
            hs.releaseConst(dx.token);
            hs.releaseConst(dg.token);
            hs.releaseConst(db.token);
            hs.releaseMut(dout.token);
        }
        if (!context.storageBindingFits(ctx, dx.len) or !context.storageBindingFits(ctx, dout.len)) return error.Unsupported;

        const xv = try deviceRowView(dx);
        const ov = try deviceRowView(dout);
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

fn reduceEntry(op: types.ReduceOp) [:0]const u8 {
    return switch (op) {
        .sum => "reduce_sum_row",
        .mean => "reduce_mean_row",
    };
}

/// Whole-tensor reduce to a [1] scalar. Small tensors run the single-workgroup
/// row kernel; anything past a few workgroups' worth goes TWO-STAGE — many
/// workgroups write per-workgroup partial sums into the shared scratch pool,
/// then one workgroup folds the partials (a single WG striding megabytes runs
/// at ~1% of DRAM bandwidth). v1: single-tile input and output.
pub fn execReduceAll(ctx: Ctx, frame: *Frame, s: executable.StepReduceAll) ExecuteProgramError!void {
    const hs = ctx.rstore.tensorStore();
    const meta = hs.meta(s.a) catch return error.ExecutionFailed;
    try requireF32(meta.dtype);
    const out_meta = hs.meta(s.out) catch return error.ExecutionFailed;
    try requireF32(out_meta.dtype);
    if (context.totalTiles(meta) != 1 or context.totalTiles(out_meta) != 1) return error.Unsupported;

    const dx = ctx.rstore.acquireTileDeviceConstLinear(s.a, 0) catch return error.ExecutionFailed;
    const dout = ctx.rstore.acquireTileDeviceMutLinear(s.out, 0) catch return error.ExecutionFailed;
    defer {
        hs.releaseConst(dx.token);
        hs.releaseMut(dout.token);
    }
    if (!context.storageBindingFits(ctx, dx.len)) return error.Unsupported;

    const n = try devicePackedElems(dx);
    const x_buf = ctx.devmem.bufferFor(dx.handle).?;
    const o_buf = ctx.devmem.bufferFor(dout.handle).?;
    const sizes_xo = [_]u64{ dx.len, dout.len };

    // Each stage-1 thread should sum a few dozen elements; below ~2 workgroups
    // of work the single-stage kernel wins on dispatch overhead.
    const WG: u32 = 256;
    const PER_THREAD: u32 = 32;
    const groups: u32 = @min(context.ceilDiv(n, WG * PER_THREAD), 1024);
    if (groups <= 2) {
        const built = try ctx.pipes.get(reduce_kernel, reduceEntry(s.op));
        const bufs = [_]c.WGPUBuffer{ x_buf, o_buf };
        const params: RowParams = .{ .rows = 1, .cols = n, .x_row = 0 };
        try frame.recordCompute(built, &bufs, &sizes_xo, std.mem.asBytes(&params), .{ 1, 1, 1 });
        return;
    }

    const scratch = try ctx.scratch.ensure(ctx.gpu, @as(u64, groups) * @sizeOf(f32));

    // Stage 1: partial sums, one f32 per workgroup.
    {
        const built = try ctx.pipes.get(reduce_kernel, "reduce_all_partial");
        const bufs = [_]c.WGPUBuffer{ x_buf, scratch };
        const sizes = [_]u64{ dx.len, @as(u64, groups) * @sizeOf(f32) };
        const params: RowParams = .{ .rows = 1, .cols = n, .x_row = 0 };
        try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups, 1, 1 });
    }
    // Stage 2: fold the partials; mean divides by the ORIGINAL n (in x_row).
    {
        const entry: [:0]const u8 = switch (s.op) {
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

/// ArgMax over the last axis: o[row] (i32) = index of the row max, lowest index
/// on ties (kernels/argmax.wgsl). v1 contract mirrors the CPU exec: single-tile
/// input and output, f32 in / i32 out.
pub fn execArgMax(ctx: Ctx, frame: *Frame, s: executable.StepArgMax) ExecuteProgramError!void {
    const hs = ctx.rstore.tensorStore();
    const meta = hs.meta(s.a) catch return error.ExecutionFailed;
    try requireF32(meta.dtype);
    const out_meta = hs.meta(s.out) catch return error.ExecutionFailed;
    if (out_meta.dtype != .i32) return error.Unsupported;

    const rank: usize = @as(usize, meta.rank);
    if (rank == 0 or s.axis != rank - 1) return error.Unsupported; // v1: last axis only
    if (context.totalTiles(meta) != 1 or context.totalTiles(out_meta) != 1) return error.Unsupported;

    const built = try ctx.pipes.get(argmax_kernel, "argmax_row");

    const dx = ctx.rstore.acquireTileDeviceConstLinear(s.a, 0) catch return error.ExecutionFailed;
    const dout = ctx.rstore.acquireTileDeviceMutLinear(s.out, 0) catch return error.ExecutionFailed;
    defer {
        hs.releaseConst(dx.token);
        hs.releaseMut(dout.token);
    }
    if (!context.storageBindingFits(ctx, dx.len)) return error.Unsupported;

    const xv = try deviceRowView(dx);
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
            const b1 = try ctx.pipes.get(argmax_kernel, "argmax_partial");
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

/// Reduce over the last axis: o[row] = sum/mean of that row. v1: single-tile
/// input and output (the output row order must match the input's flat row index).
pub fn execReduceAxis(ctx: Ctx, frame: *Frame, s: executable.StepReduceAxis) ExecuteProgramError!void {
    const hs = ctx.rstore.tensorStore();
    const meta = hs.meta(s.a) catch return error.ExecutionFailed;
    try requireF32(meta.dtype);
    const out_meta = hs.meta(s.out) catch return error.ExecutionFailed;
    try requireF32(out_meta.dtype);

    const rank: usize = @as(usize, meta.rank);
    if (rank >= 2 and s.axis != rank - 1) return error.Unsupported; // v1: last axis only
    if (context.totalTiles(meta) != 1 or context.totalTiles(out_meta) != 1) return error.Unsupported;

    const built = try ctx.pipes.get(reduce_kernel, reduceEntry(s.op));

    const dx = ctx.rstore.acquireTileDeviceConstLinear(s.a, 0) catch return error.ExecutionFailed;
    const dout = ctx.rstore.acquireTileDeviceMutLinear(s.out, 0) catch return error.ExecutionFailed;
    defer {
        hs.releaseConst(dx.token);
        hs.releaseMut(dout.token);
    }
    if (!context.storageBindingFits(ctx, dx.len)) return error.Unsupported;

    const xv = try deviceRowView(dx);
    const out_n = try devicePackedElems(dout);
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
