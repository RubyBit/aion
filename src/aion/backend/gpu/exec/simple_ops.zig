// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Elementwise-binary, unary, broadcast-last-dim, and copy execution for the GPU
//! backend. These are the "simple" ops: f32, multi-tile, no autotuning — one
//! dispatch (or buffer copy) per tile over an embedded hand-written WGSL kernel.
//! Kept out of `backend.zig` (plumbing) and out of `matmul/` (the heavy, tuned
//! codegen family); row-wise reduction ops live beside this file in `rowwise.zig`.

const std = @import("std");
const wgpu = @import("../wgpu.zig");
const pipelines = @import("../pipelines.zig");
const context = @import("../context.zig");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const executable = @import("../../../runtime/executable.zig");
const tensor_store_mod = @import("../../../runtime/tensor_store.zig");

const c = wgpu.c;
const Ctx = context.Ctx;
const Frame = @import("../frame.zig").Frame;
const ExecuteProgramError = backend_mod.ExecuteProgramError;
const KernelDesc = pipelines.KernelDesc;

const elementwise_kernel: KernelDesc = .{ .name = "elementwise", .wgsl = @embedFile("../kernels/elementwise.wgsl") };
const elementwise_i32_kernel: KernelDesc = .{ .name = "elementwise_i32", .wgsl = @embedFile("../kernels/elementwise_i32.wgsl") };
const elementwise_broadcast_kernel: KernelDesc = .{ .name = "elementwise_broadcast", .wgsl = @embedFile("../kernels/elementwise_broadcast.wgsl") };
const elementwise_broadcast_i32_kernel: KernelDesc = .{ .name = "elementwise_broadcast_i32", .wgsl = @embedFile("../kernels/elementwise_broadcast_i32.wgsl") };
const unary_kernel: KernelDesc = .{ .name = "unary", .wgsl = @embedFile("../kernels/unary.wgsl") };
const elementwise_suffix_kernel: KernelDesc = .{ .name = "elementwise_suffix", .wgsl = @embedFile("../kernels/elementwise_suffix.wgsl") };
const elementwise_suffix_i32_kernel: KernelDesc = .{ .name = "elementwise_suffix_i32", .wgsl = @embedFile("../kernels/elementwise_suffix_i32.wgsl") };
const dequant_kernel: KernelDesc = .{ .name = "dequant", .wgsl = @embedFile("../kernels/dequant.wgsl") };

const WORKGROUP_1D: u32 = 64; // must match @workgroup_size in the 1-D elementwise and unary shaders

/// Uniform params for the 1D elementwise/unary kernels (16-byte aligned).
const ScalarParams = extern struct { n: u32, _pad0: u32 = 0, _pad1: u32 = 0, _pad2: u32 = 0 };
/// Uniform params for arbitrary right-aligned broadcasting, up to MAX_RANK=8.
const ElementwiseBroadcastParams = extern struct {
    n: u32,
    rank: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    shape0: [4]u32,
    shape1: [4]u32,
    a_stride0: [4]u32,
    a_stride1: [4]u32,
    b_stride0: [4]u32,
    b_stride1: [4]u32,
};
/// Uniform params for the packed contiguous-suffix fast path.
const SuffixParams = extern struct { n: u32, cols: u32, _pad0: u32 = 0, _pad1: u32 = 0 };
/// Uniform params matching dequant.wgsl's `Params` (cast uses only `count`).
const CastParams = extern struct { n: u32 = 0, k: u32 = 0, src_wpr: u32 = 0, dst_row: u32 = 0, count: u32, _p0: u32 = 0, _p1: u32 = 0, _p2: u32 = 0 };

/// Workgroup count for a grid-strided 1D kernel over `n` elements: enough
/// groups to cover `n` directly when small, capped under the WebGPU per-dim
/// dispatch limit (the kernels loop with stride = total threads).
fn groups1D(n: u32) u32 {
    return @max(1, @min(context.ceilDiv(n, WORKGROUP_1D), context.MAX_GROUPS_1D));
}

fn requireF32(meta: types.DType) ExecuteProgramError!void {
    if (meta != .f32) return error.Unsupported;
}

pub fn execElemwiseBinary(ctx: Ctx, frame: *Frame, s: executable.StepElemwiseBinaryTiled) ExecuteProgramError!void {
    const hs = ctx.store;
    const meta = hs.meta(s.out) catch return error.ExecutionFailed;
    const a_meta = hs.meta(s.a) catch return error.ExecutionFailed;
    const b_meta = hs.meta(s.b) catch return error.ExecutionFailed;
    const broadcast = s.broadcast.kind != .identical;
    const fast_suffix = (s.broadcast.kind == .scalar_b or s.broadcast.kind == .contiguous_suffix_b) and !s.op.isComparison();

    // Comparisons are i32-in/i32-out (infer enforces it); arithmetic is
    // dtype-preserving f32 or i32. Both element sizes are 4 bytes, so the
    // dispatch math below is dtype-agnostic — only the kernel module differs.
    const kernel: KernelDesc = switch (meta.dtype) {
        .f32 => if (s.op.isComparison())
            return error.Unsupported
        else if (fast_suffix)
            elementwise_suffix_kernel
        else if (broadcast)
            elementwise_broadcast_kernel
        else
            elementwise_kernel,
        .i32 => if (fast_suffix)
            elementwise_suffix_i32_kernel
        else if (broadcast)
            elementwise_broadcast_i32_kernel
        else
            elementwise_i32_kernel,
        else => return error.Unsupported,
    };
    const entry: [:0]const u8 = if (fast_suffix) switch (s.op) {
        .add => "suffix_add",
        .sub => "suffix_sub",
        .mul => "suffix_mul",
        .div => "suffix_div",
        else => unreachable,
    } else switch (s.op) {
        .add => "add",
        .sub => "sub",
        .mul => "mul",
        .div => "divide",
        .eq => "eq",
        .ne => "ne",
        .lt => "lt",
        .gt => "gt",
        .le => "le",
        .ge => "ge",
    };
    const built = try ctx.pipes.get(kernel, entry);

    const total = context.totalTiles(meta);
    var ti: usize = 0;
    while (ti < total) : (ti += 1) {
        var coords: [tensor_store_mod.INLINE_RANK]usize = @splat(0);
        tensor_store_mod.decodeTileCoords(meta, ti, coords[0..@as(usize, meta.rank)]) catch return error.ExecutionFailed;
        const a_ti = tensor_store_mod.projectTileIndex(a_meta, coords[0..@as(usize, meta.rank)], &.{}, s.broadcast.a_broadcast_axes) catch return error.ExecutionFailed;
        const b_ti = tensor_store_mod.projectTileIndex(b_meta, coords[0..@as(usize, meta.rank)], &.{}, s.broadcast.b_broadcast_axes) catch return error.ExecutionFailed;

        const da = ctx.store.acquireTileDeviceConstLinear(s.a, a_ti) catch return error.ExecutionFailed;
        const db = ctx.store.acquireTileDeviceConstLinear(s.b, b_ti) catch return error.ExecutionFailed;
        const dout = ctx.store.acquireTileDeviceMutLinear(s.out, ti) catch return error.ExecutionFailed;
        defer {
            hs.releaseConst(da.token);
            hs.releaseConst(db.token);
            hs.releaseMut(dout.token);
        }
        if (!context.storageBindingFits(ctx, da.len) or !context.storageBindingFits(ctx, db.len) or !context.storageBindingFits(ctx, dout.len)) {
            return error.Unsupported;
        }

        const bufs = [_]c.WGPUBuffer{
            ctx.devmem.bufferFor(da.handle).?,
            ctx.devmem.bufferFor(db.handle).?,
            ctx.devmem.bufferFor(dout.handle).?,
        };
        const sizes = [_]u64{ da.len, db.len, dout.len };
        const n = context.packedElems(dout.rank, dout.shape_mem[0..@as(usize, dout.rank)], dout.strides_mem[0..@as(usize, dout.rank)]) orelse {
            return error.Unsupported;
        };
        if (fast_suffix) {
            const cols = context.packedElems(db.rank, db.shape_mem[0..@as(usize, db.rank)], db.strides_mem[0..@as(usize, db.rank)]) orelse return error.Unsupported;
            const a_n = context.packedElems(da.rank, da.shape_mem[0..@as(usize, da.rank)], da.strides_mem[0..@as(usize, da.rank)]) orelse return error.Unsupported;
            if (a_n != n or cols == 0 or n % cols != 0) return error.Unsupported;
            const params: SuffixParams = .{ .n = n, .cols = cols };
            try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(n), 1, 1 });
        } else if (broadcast) {
            var params: ElementwiseBroadcastParams = .{
                .n = n,
                .rank = meta.rank,
                .shape0 = @splat(1),
                .shape1 = @splat(1),
                .a_stride0 = @splat(0),
                .a_stride1 = @splat(0),
                .b_stride0 = @splat(0),
                .b_stride1 = @splat(0),
            };
            const out_rank: usize = meta.rank;
            const a_rank: usize = a_meta.rank;
            const b_rank: usize = b_meta.rank;
            for (0..out_rank) |axis| {
                const lane = axis % 4;
                const high = axis >= 4;
                const dim = std.math.cast(u32, dout.shape_mem[axis]) orelse return error.Unsupported;
                if (high) params.shape1[lane] = dim else params.shape0[lane] = dim;

                if (axis >= out_rank - a_rank) {
                    const aa = axis - (out_rank - a_rank);
                    if ((s.broadcast.a_broadcast_axes & (@as(u8, 1) << @intCast(axis))) == 0) {
                        if (da.strides_mem[aa] < 0 or @rem(da.strides_mem[aa], @sizeOf(u32)) != 0) return error.Unsupported;
                        const stride = std.math.cast(u32, @divExact(da.strides_mem[aa], @sizeOf(u32))) orelse return error.Unsupported;
                        if (high) params.a_stride1[lane] = stride else params.a_stride0[lane] = stride;
                    }
                }
                if (axis >= out_rank - b_rank) {
                    const ba = axis - (out_rank - b_rank);
                    if ((s.broadcast.b_broadcast_axes & (@as(u8, 1) << @intCast(axis))) == 0) {
                        if (db.strides_mem[ba] < 0 or @rem(db.strides_mem[ba], @sizeOf(u32)) != 0) return error.Unsupported;
                        const stride = std.math.cast(u32, @divExact(db.strides_mem[ba], @sizeOf(u32))) orelse return error.Unsupported;
                        if (high) params.b_stride1[lane] = stride else params.b_stride0[lane] = stride;
                    }
                }
            }
            try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(n), 1, 1 });
        } else {
            const params: ScalarParams = .{ .n = n };
            try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(n), 1, 1 });
        }
    }
}

pub fn execGeluMul(ctx: Ctx, frame: *Frame, s: executable.StepGeluMulTiled) ExecuteProgramError!void {
    const hs = ctx.store;
    const meta = hs.meta(s.out) catch return error.ExecutionFailed;
    try requireF32(meta.dtype);
    const built = try ctx.pipes.get(elementwise_kernel, "mul_gelu_a");
    const total = context.totalTiles(meta);
    var ti: usize = 0;
    while (ti < total) : (ti += 1) {
        const da = ctx.store.acquireTileDeviceConstLinear(s.a, ti) catch return error.ExecutionFailed;
        const db = ctx.store.acquireTileDeviceConstLinear(s.b, ti) catch return error.ExecutionFailed;
        const dout = ctx.store.acquireTileDeviceMutLinear(s.out, ti) catch return error.ExecutionFailed;
        defer hs.releaseConst(da.token);
        defer hs.releaseConst(db.token);
        defer hs.releaseMut(dout.token);
        const bufs = [_]c.WGPUBuffer{ ctx.devmem.bufferFor(da.handle).?, ctx.devmem.bufferFor(db.handle).?, ctx.devmem.bufferFor(dout.handle).? };
        const sizes = [_]u64{ da.len, db.len, dout.len };
        const n: u32 = @intCast(dout.len / @sizeOf(f32));
        const params: ScalarParams = .{ .n = n };
        try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(n), 1, 1 });
    }
}

/// Scalar dtype casts (kernels/dequant.wgsl). f16 <-> f32 works on word pairs
/// (the f16 side is bound as u32/f32 words since WGSL needs shader-f16 for real
/// f16 bindings — tiles must pack an even element count); f32 <-> i32 is one
/// element per work item with the i32 side smuggled through bitcasts.
pub fn execCast(ctx: Ctx, frame: *Frame, s: executable.StepCastTiled) ExecuteProgramError!void {
    const hs = ctx.store;
    const out_meta = hs.meta(s.out) catch return error.ExecutionFailed;
    const x_meta = hs.meta(s.x) catch return error.ExecutionFailed;

    const has_f16 = (x_meta.dtype == .f16 or out_meta.dtype == .f16);
    const entry: [:0]const u8 = if (x_meta.dtype == .f16 and out_meta.dtype == .f32)
        "f16_to_f32"
    else if (x_meta.dtype == .f32 and out_meta.dtype == .f16)
        "f32_to_f16"
    else if (x_meta.dtype == .f32 and out_meta.dtype == .i32)
        "f32_to_i32"
    else if (x_meta.dtype == .i32 and out_meta.dtype == .f32)
        "i32_to_f32"
    else
        return error.Unsupported;
    const built = try ctx.pipes.get(dequant_kernel, entry);

    const total = context.totalTiles(out_meta);
    var ti: usize = 0;
    while (ti < total) : (ti += 1) {
        const dx = ctx.store.acquireTileDeviceConstLinear(s.x, ti) catch return error.ExecutionFailed;
        const dout = ctx.store.acquireTileDeviceMutLinear(s.out, ti) catch return error.ExecutionFailed;
        defer {
            hs.releaseConst(dx.token);
            hs.releaseMut(dout.token);
        }
        if (!context.storageBindingFits(ctx, dx.len) or !context.storageBindingFits(ctx, dout.len)) return error.Unsupported;

        var elems: u64 = 1;
        for (dout.shape_mem[0..@as(usize, dout.rank)]) |dim| elems *= dim;
        const src_need: u64 = elems * (if (x_meta.dtype == .f16) @as(u64, 2) else 4);
        const dst_need: u64 = elems * (if (out_meta.dtype == .f16) @as(u64, 2) else 4);
        if (dx.len < src_need or dout.len < dst_need) return error.Unsupported;

        // 2 f16 per u32 word; an odd tail's second lane is the tile's padding.
        const work: u64 = if (has_f16) (elems + 1) / 2 else elems;
        const count = std.math.cast(u32, work) orelse return error.Unsupported;
        const bufs = [_]c.WGPUBuffer{
            ctx.devmem.bufferFor(dx.handle).?,
            ctx.devmem.bufferFor(dout.handle).?,
        };
        const sizes = [_]u64{ dx.len, dout.len };
        const params: CastParams = .{ .count = count };
        try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(count), 1, 1 });
    }
}

/// Tile-for-tile device copy (same tiling + dtype, compile-validated), recorded
/// as buffer-to-buffer copies in the frame — no kernel, and dtype-agnostic (works
/// for quantized tiles too). WebGPU requires copy sizes be 4-byte multiples.
pub fn execCopy(ctx: Ctx, frame: *Frame, s: executable.StepCopyTiled) ExecuteProgramError!void {
    const hs = ctx.store;
    const meta = hs.meta(s.dst) catch return error.ExecutionFailed;

    const total = context.totalTiles(meta);
    var ti: usize = 0;
    while (ti < total) : (ti += 1) {
        const src = ctx.store.acquireTileDeviceConstLinear(s.src, ti) catch return error.ExecutionFailed;
        const dst = ctx.store.acquireTileDeviceMutLinear(s.dst, ti) catch return error.ExecutionFailed;
        defer {
            hs.releaseConst(src.token);
            hs.releaseMut(dst.token);
        }
        if (src.len != dst.len or src.len % 4 != 0) return error.Unsupported;
        frame.recordCopy(
            ctx.devmem.bufferFor(src.handle).?,
            0,
            ctx.devmem.bufferFor(dst.handle).?,
            0,
            src.len,
        );
    }
}

pub fn execUnary(ctx: Ctx, frame: *Frame, s: executable.StepUnaryTiled) ExecuteProgramError!void {
    const hs = ctx.store;
    const meta = hs.meta(s.out) catch return error.ExecutionFailed;
    try requireF32(meta.dtype);

    const entry: [:0]const u8 = switch (s.op) {
        .relu => "relu",
        .gelu => "gelu",
        .silu => "silu",
        .sigmoid => "sigmoid",
        .tanh => "tanh_",
        .sqrt => "sqrt_",
        .log => "log_",
    };
    const built = try ctx.pipes.get(unary_kernel, entry);

    const total = context.totalTiles(meta);
    var ti: usize = 0;
    while (ti < total) : (ti += 1) {
        const dx = ctx.store.acquireTileDeviceConstLinear(s.a, ti) catch return error.ExecutionFailed;
        const dout = ctx.store.acquireTileDeviceMutLinear(s.out, ti) catch return error.ExecutionFailed;
        if (!context.storageBindingFits(ctx, dx.len) or !context.storageBindingFits(ctx, dout.len)) {
            hs.releaseConst(dx.token);
            hs.releaseMut(dout.token);
            return error.Unsupported;
        }

        const bufs = [_]c.WGPUBuffer{
            ctx.devmem.bufferFor(dx.handle).?,
            ctx.devmem.bufferFor(dout.handle).?,
        };
        const sizes = [_]u64{ dx.len, dout.len };
        const n: u32 = @intCast(dout.len / @sizeOf(f32));
        const params: ScalarParams = .{ .n = n };
        try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(n), 1, 1 });

        hs.releaseConst(dx.token);
        hs.releaseMut(dout.token);
    }
}
