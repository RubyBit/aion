// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Row-wise reduction ops for the GPU backend: softmax, LayerNorm/RMSNorm, and
//! sum/mean reductions. Each is one dispatch of one 256-thread workgroup PER ROW,
//! with shared-memory tree reductions inside the workgroup (kernels/softmax.wgsl,
//! norm.wgsl, reduce.wgsl). The reduced axis is the last dimension; rows past the
//! per-dimension group cap spill into grid y.

const std = @import("std");
const wgpu = @import("../wgpu.zig");
const pipelines = @import("../pipelines.zig");
const context = @import("../context.zig");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const executable = @import("../../../runtime/executable.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");

const c = wgpu.c;
const Ctx = context.Ctx;
const Frame = @import("../frame.zig").Frame;
const ExecuteProgramError = backend_mod.ExecuteProgramError;
const KernelDesc = pipelines.KernelDesc;
const TensorMeta = tensor_store.TensorMeta;

const softmax_kernel: KernelDesc = .{ .name = "softmax", .wgsl = @embedFile("../kernels/softmax.wgsl") };
const norm_kernel: KernelDesc = .{ .name = "norm", .wgsl = @embedFile("../kernels/norm.wgsl") };
const add_norm_kernel: KernelDesc = .{ .name = "add_norm", .wgsl = @embedFile("../kernels/add_norm.wgsl") };
const reduce_kernel: KernelDesc = .{ .name = "reduce", .wgsl = @embedFile("../kernels/reduce.wgsl") };
const reduce_i32_kernel: KernelDesc = .{ .name = "reduce_i32", .wgsl = @embedFile("../kernels/reduce_i32.wgsl") };
const argmax_kernel: KernelDesc = .{ .name = "argmax", .wgsl = @embedFile("../kernels/argmax.wgsl") };
const topk_kernel: KernelDesc = .{ .name = "topk", .wgsl = @embedFile("../kernels/topk.wgsl") };
const topk_split_kernel: KernelDesc = .{ .name = "topk_split", .wgsl = @embedFile("../kernels/topk_split.wgsl") };

/// Uniform params shared by softmax (`rows/cols/x_row/o_row`) and, prefix-wise,
/// reduce (`rows/cols/x_row`). Field order matches the WGSL structs.
const RowParams = extern struct { rows: u32, cols: u32, x_row: u32, o_row: u32 = 0 };
/// Field order matches `Params` in topk.wgsl.
const TopKParams = extern struct { rows: u32, cols: u32, k: u32, largest: u32, x_row: u32, o_row: u32, groups_x: u32 = 0, seg: u32 = 0 };
const NormParams = extern struct { rows: u32, cols: u32, x_row: u32, o_row: u32, eps: f32, _p0: u32 = 0, _p1: u32 = 0, _p2: u32 = 0 };
/// Field order matches `Params` in add_norm.wgsl. Pads are scalar `u32` on purpose: a
/// `vec3<u32>` pad forces 16-byte alignment in WGSL and the struct sizes then disagree,
/// which surfaces only as a bare wgpu uncaptured error.
const AddNormParams = extern struct { rows: u32, cols: u32, x_row: u32, o_row: u32, a_row: u32, eps: f32, _p0: u32 = 0, _p1: u32 = 0 };

pub const NormMode = enum { rmsnorm, layernorm };

fn rowGrid(rows: u32) ExecuteProgramError![3]u32 {
    if (rows == 0) return error.Unsupported;
    const x = @min(rows, context.MAX_GROUPS_PER_DIM);
    const y = std.math.divCeil(u32, rows, x) catch return error.Unsupported;
    if (y > context.MAX_GROUPS_PER_DIM) return error.Unsupported;
    return .{ x, y, 1 };
}

/// A tensor seen as `[rows, cols]` over its last axis (a vector is one row).
const Rows = struct { rows: u32, cols: u32 };

fn rowsOf(meta: TensorMeta) ExecuteProgramError!Rows {
    const rank: usize = meta.rank;
    if (rank == 0) return error.Unsupported;
    var rows: usize = 1;
    for (meta.shape[0 .. rank - 1]) |d| rows = std.math.mul(usize, rows, d) catch return error.Unsupported;
    return .{
        .rows = std.math.cast(u32, rows) orelse return error.Unsupported,
        .cols = std.math.cast(u32, meta.shape[rank - 1]) orelse return error.Unsupported,
    };
}

/// The row kernels bind each operand whole; a tensor past the binding limit is
/// chunked, which they do not address.
fn requireWhole(metas: anytype) ExecuteProgramError!void {
    inline for (metas) |m| if (m.chunks != 1) return error.Unsupported;
}

fn requireF32(dtype: types.DType) ExecuteProgramError!void {
    if (dtype != .f32) return error.Unsupported;
}

/// The scalar float dtypes the row-wise kernels accept. Each f16 kernel is an
/// entry point in the SAME module as its f32 twin, keeps its row statistics in
/// f32, and rounds only on store — matching the CPU kernels exactly.
fn requireScalarFloat(dtype: types.DType) ExecuteProgramError!void {
    switch (dtype) {
        .f32, .f16 => {},
        else => return error.Unsupported,
    }
}

/// Softmax over the last axis. Negative axes were normalized by the graph API;
/// the step still carries i32, so normalize again here.
pub fn execSoftmax(ctx: Ctx, frame: *Frame, s: executable.StepSoftmax) ExecuteProgramError!void {
    const hs = ctx.store;
    const meta = hs.meta(s.out) catch return error.ExecutionFailed;
    const a_meta = hs.meta(s.a) catch return error.ExecutionFailed;
    try requireScalarFloat(meta.dtype);
    if (a_meta.dtype != meta.dtype) return error.Unsupported;
    try requireWhole(.{ meta, a_meta });

    const rank: usize = @as(usize, meta.rank);
    var axis: i32 = s.axis;
    if (axis < 0) axis += @intCast(rank);
    if (axis < 0 or axis != @as(i32, @intCast(rank - 1))) return error.Unsupported; // last axis only
    const rv = try rowsOf(meta);
    if (rv.rows == 0) return;

    const dx = hs.acquireConst(s.a) catch return error.ExecutionFailed;
    defer hs.releaseConst(dx.token);
    const dout = hs.acquireMut(s.out) catch return error.ExecutionFailed;
    defer hs.releaseMut(dout.token);
    if (!context.storageBindingFits(ctx, dx.len) or !context.storageBindingFits(ctx, dout.len)) return error.Unsupported;

    const built = try ctx.pipes.get(softmax_kernel, if (meta.dtype == .f16) "softmax_row_f16" else "softmax_row");
    const bufs = [_]c.WGPUBuffer{ ctx.devmem.bufferFor(dx.handle).?, ctx.devmem.bufferFor(dout.handle).? };
    const sizes = [_]u64{ dx.len, dout.len };
    const params: RowParams = .{ .rows = rv.rows, .cols = rv.cols, .x_row = rv.cols, .o_row = rv.cols };
    try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), try rowGrid(rv.rows));
}

/// LayerNorm/RMSNorm over the trailing dim (norm_rank == 1): gamma/beta are
/// rank-1 vectors of at least the row width.
pub fn execNorm(ctx: Ctx, frame: *Frame, mode: NormMode, s: anytype) ExecuteProgramError!void {
    const hs = ctx.store;
    const meta = hs.meta(s.out) catch return error.ExecutionFailed;
    const x_meta = hs.meta(s.x) catch return error.ExecutionFailed;
    const g_meta = hs.meta(s.gamma) catch return error.ExecutionFailed;
    const b_meta = hs.meta(s.beta) catch return error.ExecutionFailed;
    try requireScalarFloat(meta.dtype);
    if (x_meta.dtype != meta.dtype or g_meta.dtype != meta.dtype or b_meta.dtype != meta.dtype) return error.Unsupported;
    try requireWhole(.{ meta, x_meta, g_meta, b_meta });

    const rank: usize = @as(usize, meta.rank);
    if (rank < 2 or g_meta.rank != 1 or b_meta.rank != 1) return error.Unsupported;
    const rv = try rowsOf(meta);
    if (g_meta.shape[0] < rv.cols or b_meta.shape[0] < rv.cols) return error.Unsupported;
    if (rv.rows == 0) return;

    const entry: [:0]const u8 = if (meta.dtype == .f16) switch (mode) {
        .rmsnorm => "rmsnorm_row_f16",
        .layernorm => "layernorm_row_f16",
    } else switch (mode) {
        .rmsnorm => "rmsnorm_row",
        .layernorm => "layernorm_row",
    };
    const built = try ctx.pipes.get(norm_kernel, entry);

    const dx = hs.acquireConst(s.x) catch return error.ExecutionFailed;
    defer hs.releaseConst(dx.token);
    const dg = hs.acquireConst(s.gamma) catch return error.ExecutionFailed;
    defer hs.releaseConst(dg.token);
    const db = hs.acquireConst(s.beta) catch return error.ExecutionFailed;
    defer hs.releaseConst(db.token);
    const dout = hs.acquireMut(s.out) catch return error.ExecutionFailed;
    defer hs.releaseMut(dout.token);
    if (!context.storageBindingFits(ctx, dx.len) or !context.storageBindingFits(ctx, dout.len)) return error.Unsupported;

    const bufs = [_]c.WGPUBuffer{
        ctx.devmem.bufferFor(dx.handle).?,
        ctx.devmem.bufferFor(dg.handle).?,
        ctx.devmem.bufferFor(db.handle).?,
        ctx.devmem.bufferFor(dout.handle).?,
    };
    const sizes = [_]u64{ dx.len, dg.len, db.len, dout.len };
    const grid = try rowGrid(rv.rows);
    const params: NormParams = .{
        .rows = rv.rows,
        .cols = rv.cols,
        .x_row = rv.cols,
        .o_row = rv.cols,
        .eps = s.eps,
        ._p0 = grid[0],
    };
    try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), grid);
}

/// `StepRMSNorm` WITH a residual: o = residual + rmsnorm(x)·gamma + beta, one
/// workgroup per row. Same contract as `execNorm` plus a residual of identical shape.
pub fn execAddNorm(ctx: Ctx, frame: *Frame, s: executable.StepRMSNorm) ExecuteProgramError!void {
    const hs = ctx.store;
    const residual = s.residual orelse return error.Unsupported;
    const meta = hs.meta(s.out) catch return error.ExecutionFailed;
    const x_meta = hs.meta(s.x) catch return error.ExecutionFailed;
    const r_meta = hs.meta(residual) catch return error.ExecutionFailed;
    const g_meta = hs.meta(s.gamma) catch return error.ExecutionFailed;
    const b_meta = hs.meta(s.beta) catch return error.ExecutionFailed;
    inline for (.{ meta, x_meta, r_meta, g_meta, b_meta }) |m| try requireF32(m.dtype);
    try requireWhole(.{ meta, x_meta, r_meta, g_meta, b_meta });

    const rank: usize = @as(usize, meta.rank);
    if (rank < 2 or g_meta.rank != 1 or b_meta.rank != 1) return error.Unsupported;
    if (!std.mem.eql(usize, r_meta.shape, meta.shape) or !std.mem.eql(usize, x_meta.shape, meta.shape)) return error.Unsupported;
    const rv = try rowsOf(meta);
    if (g_meta.shape[0] < rv.cols or b_meta.shape[0] < rv.cols) return error.Unsupported;
    if (rv.rows == 0) return;

    const built = try ctx.pipes.get(add_norm_kernel, "add_rmsnorm_row");
    const dx = hs.acquireConst(s.x) catch return error.ExecutionFailed;
    defer hs.releaseConst(dx.token);
    const dg = hs.acquireConst(s.gamma) catch return error.ExecutionFailed;
    defer hs.releaseConst(dg.token);
    const db = hs.acquireConst(s.beta) catch return error.ExecutionFailed;
    defer hs.releaseConst(db.token);
    const da = hs.acquireConst(residual) catch return error.ExecutionFailed;
    defer hs.releaseConst(da.token);
    const dout = hs.acquireMut(s.out) catch return error.ExecutionFailed;
    defer hs.releaseMut(dout.token);
    inline for (.{ dx.len, da.len, dout.len }) |len| if (!context.storageBindingFits(ctx, len)) return error.Unsupported;

    const bufs = [_]c.WGPUBuffer{
        ctx.devmem.bufferFor(dx.handle).?,
        ctx.devmem.bufferFor(dg.handle).?,
        ctx.devmem.bufferFor(db.handle).?,
        ctx.devmem.bufferFor(da.handle).?,
        ctx.devmem.bufferFor(dout.handle).?,
    };
    const sizes = [_]u64{ dx.len, dg.len, db.len, da.len, dout.len };
    const params: AddNormParams = .{
        .rows = rv.rows,
        .cols = rv.cols,
        .x_row = rv.cols,
        .o_row = rv.cols,
        .a_row = rv.cols,
        .eps = s.eps,
    };
    try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), try rowGrid(rv.rows));
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
/// at ~1% of DRAM bandwidth).
pub fn execReduceAll(ctx: Ctx, frame: *Frame, s: executable.StepReduceAll) ExecuteProgramError!void {
    const hs = ctx.store;
    const meta = hs.meta(s.a) catch return error.ExecutionFailed;
    try requireScalarFloat(meta.dtype);
    const out_meta = hs.meta(s.out) catch return error.ExecutionFailed;
    if (out_meta.dtype != meta.dtype) return error.Unsupported;
    // Staged reductions keep their partials in f32 whatever the data dtype, so
    // stage 1 is data->f32 (`_h2f`) and the fold is f32->data (`_f2h`).
    const f16_data = meta.dtype == .f16;
    try requireWhole(.{ meta, out_meta });
    var elems: usize = 1;
    for (meta.shape) |d| elems *= d;
    const n = std.math.cast(u32, elems) orelse return error.Unsupported;

    const dx = ctx.store.acquireConst(s.a) catch return error.ExecutionFailed;
    const dout = ctx.store.acquireMut(s.out) catch return error.ExecutionFailed;
    defer {
        hs.releaseConst(dx.token);
        hs.releaseMut(dout.token);
    }
    if (!context.storageBindingFits(ctx, dx.len)) return error.Unsupported;

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

/// ArgMax over the last axis: o[row] (i32) = index of the row max, lowest index
/// on ties.
pub fn execArgMax(ctx: Ctx, frame: *Frame, s: executable.StepArgMax) ExecuteProgramError!void {
    const hs = ctx.store;
    const meta = hs.meta(s.a) catch return error.ExecutionFailed;
    // f32 or f16 in, i32 index out. Widened comparison preserves f16 ordering and
    // the lowest-index tie-break, so both dtypes pick the same column.
    try requireScalarFloat(meta.dtype);
    const out_meta = hs.meta(s.out) catch return error.ExecutionFailed;
    if (out_meta.dtype != .i32) return error.Unsupported;

    const rank: usize = @as(usize, meta.rank);
    if (rank == 0 or s.axis != rank - 1) return error.Unsupported; // last axis only
    try requireWhole(.{ meta, out_meta });
    const xv = try rowsOf(meta);
    var out_n: usize = 1;
    for (out_meta.shape) |d| out_n *= d;
    if (out_n != xv.rows) return error.Unsupported;
    if (xv.rows > context.MAX_GROUPS_PER_DIM) return error.Unsupported;

    const built = try ctx.pipes.get(argmax_kernel, if (meta.dtype == .f16) "argmax_row_f16" else "argmax_row");

    const dx = ctx.store.acquireConst(s.a) catch return error.ExecutionFailed;
    const dout = ctx.store.acquireMut(s.out) catch return error.ExecutionFailed;
    defer {
        hs.releaseConst(dx.token);
        hs.releaseMut(dout.token);
    }
    if (!context.storageBindingFits(ctx, dx.len)) return error.Unsupported;

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
            const params: RowParams = .{ .rows = xv.rows, .cols = xv.cols, .x_row = xv.cols, .o_row = SEG_COLS };
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
    const params: RowParams = .{ .rows = xv.rows, .cols = xv.cols, .x_row = xv.cols };
    try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ xv.rows, 1, 1 });
}

/// Top-k over the last axis: `k` values and their i32 indices per row, sorted
/// best-first with ties to the lowest index — identical to the CPU kernel's order.
pub fn execTopK(ctx: Ctx, frame: *Frame, s: executable.StepTopK) ExecuteProgramError!void {
    const hs = ctx.store;
    const meta = hs.meta(s.a) catch return error.ExecutionFailed;
    try requireScalarFloat(meta.dtype);
    const val_meta = hs.meta(s.values) catch return error.ExecutionFailed;
    const idx_meta = hs.meta(s.indices) catch return error.ExecutionFailed;
    if (val_meta.dtype != meta.dtype or idx_meta.dtype != .i32) return error.Unsupported;

    const rank: usize = @as(usize, meta.rank);
    if (rank == 0 or s.axis != rank - 1) return error.Unsupported; // last axis only
    if (s.k == 0 or s.k > meta.shape[rank - 1]) return error.Unsupported;
    try requireWhole(.{ meta, val_meta, idx_meta });
    const xv = try rowsOf(meta);
    const vv = try rowsOf(val_meta);
    const iv = try rowsOf(idx_meta);
    if (vv.rows != xv.rows or iv.rows != xv.rows) return error.Unsupported;
    if (vv.cols != s.k or iv.cols != s.k) return error.Unsupported;

    const built = try ctx.pipes.get(topk_kernel, if (meta.dtype == .f16) "topk_row_f16" else "topk_row");

    const dx = ctx.store.acquireConst(s.a) catch return error.ExecutionFailed;
    const dval = ctx.store.acquireMut(s.values) catch return error.ExecutionFailed;
    const didx = ctx.store.acquireMut(s.indices) catch return error.ExecutionFailed;
    defer {
        hs.releaseConst(dx.token);
        hs.releaseMut(dval.token);
        hs.releaseMut(didx.token);
    }
    if (!context.storageBindingFits(ctx, dx.len)) return error.Unsupported;

    const x_buf = ctx.devmem.bufferFor(dx.handle).?;
    const val_buf = ctx.devmem.bufferFor(dval.handle).?;
    const idx_buf = ctx.devmem.bufferFor(didx.handle).?;
    const k: u32 = @intCast(s.k);

    // Decode shape: ONE row of a whole vocab. `k` rounds on a single workgroup use
    // one SM and leave the rest of the GPU idle, so take each segment's top-k in
    // parallel and fold the candidates. Segments must hold at least `k` columns for
    // a segment's top-k to be meaningful.
    const SEG_COLS: u32 = 4096;
    const segs: u32 = context.ceilDiv(xv.cols, SEG_COLS);
    if (segs > 1 and segs <= context.MAX_GROUPS_PER_DIM and SEG_COLS >= k) {
        const entries: u64 = @as(u64, xv.rows) * segs * k;
        const scratch_bytes: u64 = entries * 2 * @sizeOf(f32);
        const scratch = try ctx.scratch.ensure(ctx.gpu, scratch_bytes);
        {
            const b1 = try ctx.pipes.get(topk_split_kernel, if (meta.dtype == .f16) "topk_partial_f16" else "topk_partial");
            const bufs = [_]c.WGPUBuffer{ x_buf, scratch };
            const sizes = [_]u64{ dx.len, scratch_bytes };
            const params: TopKParams = .{
                .rows = xv.rows,
                .cols = xv.cols,
                .k = k,
                .largest = @intFromBool(s.largest),
                .x_row = xv.cols,
                .o_row = 0,
                .seg = SEG_COLS,
            };
            try frame.recordCompute(b1, &bufs, &sizes, std.mem.asBytes(&params), .{ segs, xv.rows, 1 });
        }
        {
            const b2 = try ctx.pipes.get(topk_kernel, if (meta.dtype == .f16) "topk_finish_f16" else "topk_finish");
            const grid = try rowGrid(xv.rows);
            const bufs = [_]c.WGPUBuffer{ scratch, val_buf, idx_buf };
            const sizes = [_]u64{ scratch_bytes, dval.len, didx.len };
            const params: TopKParams = .{
                .rows = xv.rows,
                // The fold's "columns" are the candidates stage 1 produced per row.
                .cols = segs * k,
                .k = k,
                .largest = @intFromBool(s.largest),
                .x_row = 0,
                .o_row = vv.cols,
                .groups_x = grid[0],
            };
            try frame.recordCompute(b2, &bufs, &sizes, std.mem.asBytes(&params), grid);
        }
        return;
    }

    const grid = try rowGrid(xv.rows);
    const params: TopKParams = .{
        .rows = xv.rows,
        .cols = xv.cols,
        .k = k,
        .largest = @intFromBool(s.largest),
        .x_row = xv.cols,
        .o_row = vv.cols,
        .groups_x = grid[0],
    };
    const bufs = [_]c.WGPUBuffer{ x_buf, val_buf, idx_buf };
    const sizes = [_]u64{ dx.len, dval.len, didx.len };
    try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), grid);
}

/// Reduce over the last axis: o[row] = sum/mean of that row.
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
    if (rank >= 2 and s.axis != rank - 1) return error.Unsupported; // last axis only
    try requireWhole(.{ meta, out_meta });
    const xv = try rowsOf(meta);
    var out_n: usize = 1;
    for (out_meta.shape) |d| out_n *= d;
    if (out_n != xv.rows) return error.Unsupported;
    if (xv.rows == 0) return;

    const built = if (meta.dtype == .i32)
        try ctx.pipes.get(reduce_i32_kernel, "reduce_sum_row")
    else
        try ctx.pipes.get(reduce_kernel, reduceEntry(s.op, meta.dtype));

    const dx = ctx.store.acquireConst(s.a) catch return error.ExecutionFailed;
    const dout = ctx.store.acquireMut(s.out) catch return error.ExecutionFailed;
    defer {
        hs.releaseConst(dx.token);
        hs.releaseMut(dout.token);
    }
    if (!context.storageBindingFits(ctx, dx.len)) return error.Unsupported;

    const bufs = [_]c.WGPUBuffer{
        ctx.devmem.bufferFor(dx.handle).?,
        ctx.devmem.bufferFor(dout.handle).?,
    };
    const sizes = [_]u64{ dx.len, dout.len };
    const params: RowParams = .{ .rows = xv.rows, .cols = xv.cols, .x_row = xv.cols };
    try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), try rowGrid(xv.rows));
}
