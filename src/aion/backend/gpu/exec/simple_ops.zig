// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Elementwise-binary, unary, broadcast-last-dim, and copy execution for the GPU
//! backend. These are the "simple" ops: no autotuning — one dispatch (or buffer
//! copy) over an embedded hand-written WGSL kernel.
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

/// Must match `@workgroup_size` in the 1-D elementwise and unary shaders.
///
/// Do NOT widen this to chase decode latency. It looks tempting — a `[1,1,1536]`
/// residual add is 24 workgroups of 64 threads, one element each, and the token runs
/// 105 of those plus 36 suffix multiplies and 70 gelu·mul — but measured 2026-08-19,
/// 256 threads is a severe regression at real sizes: large elementwise fell from
/// 350/374/380 GB/s (add/silu/suffix at 8192x8192) to 123/119/120, about 3x. Decode
/// showed no end-to-end gain to trade against that, because mid-size ops like the
/// 12288-wide geglu regress too. The small-shape launch overhead has to be removed by
/// FUSING these ops away, not by making each dispatch wider.
const WORKGROUP_1D: u32 = 64;

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

pub fn execElemwiseBinary(ctx: Ctx, frame: *Frame, s: executable.StepElemwiseBinary) ExecuteProgramError!void {
    const hs = ctx.store;
    const meta = hs.meta(s.out) catch return error.ExecutionFailed;
    const a_meta = hs.meta(s.a) catch return error.ExecutionFailed;
    const b_meta = hs.meta(s.b) catch return error.ExecutionFailed;
    const broadcast = s.broadcast.kind != .identical;
    const fast_suffix = (s.broadcast.kind == .scalar_b or s.broadcast.kind == .contiguous_suffix_b) and !s.op.isComparison() and s.op != .gate;

    // Comparisons are i32-in/i32-out (infer enforces it); arithmetic is
    // dtype-preserving f32, f16 or i32.
    // A gate carries an activation and exists only as a same-shape f32 kernel; the
    // compiler refuses anything else, so a mismatch here is a compiler bug, not input.
    if (s.op == .gate and (broadcast or meta.dtype != .f32)) return error.Unsupported;

    switch (meta.dtype) {
        .f32, .i32, .f16 => {},
        else => return error.Unsupported,
    }

    // f16 shares each module with its f32 twin, differing only in entry point.
    const is_f16 = meta.dtype == .f16;
    if (is_f16 and (a_meta.dtype != .f16 or b_meta.dtype != .f16)) return error.Unsupported;

    const kernel: KernelDesc = switch (meta.dtype) {
        .f32, .f16 => if (s.op.isComparison())
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
    const entry: [:0]const u8 = if (fast_suffix and is_f16) switch (s.op) {
        .add => "suffix_add_f16",
        .sub => "suffix_sub_f16",
        .mul => "suffix_mul_f16",
        .div => "suffix_div_f16",
        else => unreachable,
    } else if (fast_suffix) switch (s.op) {
        .add => "suffix_add",
        .sub => "suffix_sub",
        .mul => "suffix_mul",
        .div => "suffix_div",
        else => unreachable,
    } else if (is_f16) switch (s.op) {
        .add => "add_f16",
        .sub => "sub_f16",
        .mul => "mul_f16",
        .div => "divide_f16",
        else => return error.Unsupported,
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
        // One entry point per activation, named for it, so adding an activation to
        // `UnaryOp` is a one-line kernel and nothing else.
        .gate => switch (s.act) {
            .relu => "gate_relu",
            .gelu => "gate_gelu",
            .silu => "gate_silu",
            .sigmoid => "gate_sigmoid",
            .tanh => "gate_tanh",
            .sqrt => "gate_sqrt",
            .log => "gate_log",
        },
    };
    const built = try ctx.pipes.get(kernel, entry);

    // Every operand is one device buffer; a tensor past the binding limit is
    // chunked, which the pointwise kernels do not address.
    inline for (.{ meta, a_meta, b_meta }) |m| if (m.chunks != 1) return error.Unsupported;
    const n_usize = elemCount(meta.shape);
    const n = std.math.cast(u32, n_usize) orelse return error.Unsupported;
    if (n == 0) return;

    const da = hs.acquireConst(s.a) catch return error.ExecutionFailed;
    defer hs.releaseConst(da.token);
    const db = hs.acquireConst(s.b) catch return error.ExecutionFailed;
    defer hs.releaseConst(db.token);
    const dout = hs.acquireMut(s.out) catch return error.ExecutionFailed;
    defer hs.releaseMut(dout.token);
    inline for (.{ da.len, db.len, dout.len }) |len| if (!context.storageBindingFits(ctx, len)) return error.Unsupported;

    const bufs = [_]c.WGPUBuffer{
        ctx.devmem.bufferFor(da.handle).?,
        ctx.devmem.bufferFor(db.handle).?,
        ctx.devmem.bufferFor(dout.handle).?,
    };
    const sizes = [_]u64{ da.len, db.len, dout.len };
    if (fast_suffix) {
        const cols = std.math.cast(u32, elemCount(b_meta.shape)) orelse return error.Unsupported;
        if (elemCount(a_meta.shape) != n_usize or cols == 0 or n % cols != 0) return error.Unsupported;
        const params: SuffixParams = .{ .n = n, .cols = cols };
        return frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(n), 1, 1 });
    }
    if (!broadcast) {
        const params: ScalarParams = .{ .n = n };
        return frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(n), 1, 1 });
    }

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
    const a_strides = rowMajorStrides(a_meta.shape);
    const b_strides = rowMajorStrides(b_meta.shape);
    for (0..out_rank) |axis| {
        const lane = axis % 4;
        const high = axis >= 4;
        const dim = std.math.cast(u32, meta.shape[axis]) orelse return error.Unsupported;
        if (high) params.shape1[lane] = dim else params.shape0[lane] = dim;

        // Operands right-align against the output; a broadcast axis keeps stride 0.
        if (axis >= out_rank - a_meta.rank and (s.broadcast.a_broadcast_axes & (@as(u8, 1) << @intCast(axis))) == 0) {
            const stride = std.math.cast(u32, a_strides[axis - (out_rank - a_meta.rank)]) orelse return error.Unsupported;
            if (high) params.a_stride1[lane] = stride else params.a_stride0[lane] = stride;
        }
        if (axis >= out_rank - b_meta.rank and (s.broadcast.b_broadcast_axes & (@as(u8, 1) << @intCast(axis))) == 0) {
            const stride = std.math.cast(u32, b_strides[axis - (out_rank - b_meta.rank)]) orelse return error.Unsupported;
            if (high) params.b_stride1[lane] = stride else params.b_stride0[lane] = stride;
        }
    }
    try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(n), 1, 1 });
}

fn elemCount(shape: []const usize) usize {
    var n: usize = 1;
    for (shape) |d| n *= d;
    return n;
}

/// Element strides of a packed row-major tensor of `shape`.
fn rowMajorStrides(shape: []const usize) [tensor_store_mod.INLINE_RANK]usize {
    var strides: [tensor_store_mod.INLINE_RANK]usize = @splat(0);
    var acc: usize = 1;
    var d = shape.len;
    while (d > 0) {
        d -= 1;
        strides[d] = acc;
        acc *= shape[d];
    }
    return strides;
}

/// Scalar dtype casts (kernels/dequant.wgsl). Every pair is ONE ELEMENT per work
/// item: `shader-f16` is a required device feature, so the f16 side binds as a
/// native `array<f16>` instead of being packed two-per-u32-word. That packing
/// used to round the dispatch up to `(elems + 1) / 2` and let an odd tensor's last
/// work item write a whole word — half of it past the logical extent. Addressing elements removes both
/// the rounding and the constraint. f32 <-> i32 smuggles the i32 side through
/// bitcasts on the f32 binding.
pub fn execCast(ctx: Ctx, frame: *Frame, s: executable.StepCast) ExecuteProgramError!void {
    const hs = ctx.store;
    const out_meta = hs.meta(s.out) catch return error.ExecutionFailed;
    const x_meta = hs.meta(s.x) catch return error.ExecutionFailed;

    const entry: [:0]const u8 = if (x_meta.dtype == .f16 and out_meta.dtype == .f32)
        "f16_to_f32_elem"
    else if (x_meta.dtype == .f32 and out_meta.dtype == .f16)
        "f32_to_f16_elem"
    else if (x_meta.dtype == .f32 and out_meta.dtype == .i32)
        "f32_to_i32"
    else if (x_meta.dtype == .i32 and out_meta.dtype == .f32)
        "i32_to_f32"
    else
        return error.Unsupported;
    const built = try ctx.pipes.get(dequant_kernel, entry);

    if (out_meta.chunks != 1 or x_meta.chunks != 1) return error.Unsupported;
    const elems: u64 = elemCount(out_meta.shape);
    const count = std.math.cast(u32, elems) orelse return error.Unsupported;
    if (count == 0) return;

    const dx = hs.acquireConst(s.x) catch return error.ExecutionFailed;
    defer hs.releaseConst(dx.token);
    const dout = hs.acquireMut(s.out) catch return error.ExecutionFailed;
    defer hs.releaseMut(dout.token);
    if (!context.storageBindingFits(ctx, dx.len) or !context.storageBindingFits(ctx, dout.len)) return error.Unsupported;
    const src_need: u64 = elems * (if (x_meta.dtype == .f16) @as(u64, 2) else 4);
    const dst_need: u64 = elems * (if (out_meta.dtype == .f16) @as(u64, 2) else 4);
    if (dx.len < src_need or dout.len < dst_need) return error.Unsupported;

    const bufs = [_]c.WGPUBuffer{
        ctx.devmem.bufferFor(dx.handle).?,
        ctx.devmem.bufferFor(dout.handle).?,
    };
    const sizes = [_]u64{ dx.len, dout.len };
    const params: CastParams = .{ .count = count };
    try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(count), 1, 1 });
}

/// Chunk-for-chunk device copy (same shape + dtype, compile-validated), recorded
/// as buffer-to-buffer copies in the frame — no kernel, and dtype-agnostic (works
/// for quantized tensors too).
///
/// WebGPU requires copy sizes be 4-byte multiples, and a chunk's LOGICAL length is
/// not always one: an f16 tensor with an odd element count is 2 mod 4. The copy is
/// therefore rounded up to the same 4-byte granularity the allocator already
/// applied (`alignUp` in device_memory.zig). That is safe because each chunk owns
/// its device buffer (these copies pass offset 0), and src and dst have equal
/// `len` and so equal allocations — the extra bytes are padding nothing reads.
pub fn execCopy(ctx: Ctx, frame: *Frame, s: executable.StepCopy) ExecuteProgramError!void {
    const hs = ctx.store;
    const meta = hs.meta(s.dst) catch return error.ExecutionFailed;

    const chunks = meta.chunks;
    var ti: usize = 0;
    while (ti < chunks) : (ti += 1) {
        const src = ctx.store.acquireChunkConst(s.src, ti) catch return error.ExecutionFailed;
        const dst = ctx.store.acquireChunkMut(s.dst, ti) catch return error.ExecutionFailed;
        defer {
            hs.releaseConst(src.token);
            hs.releaseMut(dst.token);
        }
        if (src.len != dst.len) return error.Unsupported;
        const copy_len = (src.len + 3) / 4 * 4;
        frame.recordCopy(
            ctx.devmem.bufferFor(src.handle).?,
            0,
            ctx.devmem.bufferFor(dst.handle).?,
            0,
            copy_len,
        );
    }
}

pub fn execUnary(ctx: Ctx, frame: *Frame, s: executable.StepUnary) ExecuteProgramError!void {
    const hs = ctx.store;
    const meta = hs.meta(s.out) catch return error.ExecutionFailed;
    // f16 runs the same math widened to f32 and rounds on store (unary.wgsl), so
    // it only picks a different entry point in the same module.
    const elem_bytes: u64 = switch (meta.dtype) {
        .f32 => 4,
        .f16 => 2,
        else => return error.Unsupported,
    };
    const a_meta = hs.meta(s.a) catch return error.ExecutionFailed;
    if (a_meta.dtype != meta.dtype) return error.Unsupported;

    const entry: [:0]const u8 = if (meta.dtype == .f16) switch (s.op) {
        .relu => "relu_f16",
        .gelu => "gelu_f16",
        .silu => "silu_f16",
        .sigmoid => "sigmoid_f16",
        .tanh => "tanh__f16",
        .sqrt => "sqrt__f16",
        .log => "log__f16",
    } else switch (s.op) {
        .relu => "relu",
        .gelu => "gelu",
        .silu => "silu",
        .sigmoid => "sigmoid",
        .tanh => "tanh_",
        .sqrt => "sqrt_",
        .log => "log_",
    };
    const built = try ctx.pipes.get(unary_kernel, entry);

    if (meta.chunks != 1 or a_meta.chunks != 1) return error.Unsupported;
    const n = std.math.cast(u32, elemCount(meta.shape)) orelse return error.Unsupported;
    if (n == 0) return;

    const dx = hs.acquireConst(s.a) catch return error.ExecutionFailed;
    defer hs.releaseConst(dx.token);
    const dout = hs.acquireMut(s.out) catch return error.ExecutionFailed;
    defer hs.releaseMut(dout.token);
    if (!context.storageBindingFits(ctx, dx.len) or !context.storageBindingFits(ctx, dout.len)) return error.Unsupported;
    if (dout.len < @as(u64, n) * elem_bytes) return error.Unsupported;

    const bufs = [_]c.WGPUBuffer{
        ctx.devmem.bufferFor(dx.handle).?,
        ctx.devmem.bufferFor(dout.handle).?,
    };
    const sizes = [_]u64{ dx.len, dout.len };
    const params: ScalarParams = .{ .n = n };
    try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(n), 1, 1 });
}
