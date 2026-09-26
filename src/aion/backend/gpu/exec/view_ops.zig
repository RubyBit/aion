// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! View materializations + concat for the GPU backend. Every tensor is one packed
//! row-major buffer, so most of these are contiguous copies:
//!   - ReshapeScalar: packed row-major order is invariant under reshape — ONE
//!     whole-buffer copy.
//!   - ConcatScalar: contiguous [axis-block * inner] runs land at strided
//!     offsets in the output — one `strided_copy_u32` dispatch per input
//!     (encoder copies when there are only a few runs).
//!   - Transpose2DScalar / SliceNDScalar: the `gather_nd_u32` kernel
//!     (kernels/view.wgsl) pulls each dst element from a strided src offset;
//!     transpose is just the rank-2 parameterization with swapped strides.
//! Copy-like paths address 4-byte words. Narrow scalar layouts use a word view
//! of the innermost axis when logical extents and view offsets preserve word
//! alignment; the bits themselves are never converted.

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

const view_kernel: KernelDesc = .{ .name = "view", .wgsl = @embedFile("../kernels/view.wgsl") };
/// Element-addressed twins, for f16 layouts with no word view (see view_f16.wgsl).
const view_f16_kernel: KernelDesc = .{ .name = "view_f16", .wgsl = @embedFile("../kernels/view_f16.wgsl") };

/// Must match @workgroup_size in the view shaders. Stays at 64: unlike the tiny
/// pointwise ops (see WORKGROUP_1D in exec/simple_ops.zig) a view copy moves real
/// bytes -- a fused-projection slice is ~48 KiB -- so it wants MORE workgroups, not
/// fewer. Measured: widening these to 256 cost 0.7 ms/token.
const WG_1D: u32 = 64;
const MAX_RANK: usize = 8;

/// Field order matches `struct Params` in view.wgsl (arrays are vec4-packed).
const GatherParams = extern struct {
    total: u32,
    rank: u32,
    base: u32,
    src_base: u32 = 0,
    dshape: [8]u32,
    sstride: [8]u32,
};

fn groups1D(n: u32) u32 {
    return @max(1, @min(context.ceilDiv(n, WG_1D), context.MAX_GROUPS_1D));
}

/// Small contiguous copies stay in the open compute pass. Ending a pass for a
/// few KiB activation copy costs more than moving the data; large transfers keep
/// using the hardware copy path.
fn recordContiguousCopy(ctx: Ctx, frame: *Frame, src: c.WGPUBuffer, src_off: u64, dst: c.WGPUBuffer, dst_off: u64, bytes: u64) ExecuteProgramError!void {
    const COMPUTE_COPY_MAX_BYTES: u64 = 1024 * 1024;
    if (bytes > COMPUTE_COPY_MAX_BYTES) {
        frame.recordCopy(src, src_off, dst, dst_off, bytes);
        return;
    }
    if (bytes % 4 != 0 or src_off % 4 != 0 or dst_off % 4 != 0) return error.Unsupported;
    const words = std.math.cast(u32, bytes / 4) orelse return error.Unsupported;
    var params: GatherParams = .{
        .total = words,
        .rank = 1,
        .base = std.math.cast(u32, dst_off / 4) orelse return error.Unsupported,
        .src_base = std.math.cast(u32, src_off / 4) orelse return error.Unsupported,
        .dshape = @splat(1),
        .sstride = @splat(0),
    };
    params.dshape[0] = words;
    params.sstride[0] = words;
    const built = try ctx.pipes.get(view_kernel, "strided_copy_u32");
    const bufs = [_]c.WGPUBuffer{ src, dst };
    const sizes = [_]u64{ src_off + bytes, dst_off + bytes };
    try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(words), 1, 1 });
}

fn scalarBytes(dt: types.DType) ?usize {
    return switch (dt) {
        .f32, .i32 => 4,
        .f16 => 2,
        .i8 => 1,
        else => null,
    };
}

fn elementsPerWord(dt: types.DType) ?usize {
    const elem = scalarBytes(dt) orelse return null;
    if (elem == 0 or 4 % elem != 0) return null;
    return 4 / elem;
}

const Packed = struct {
    buf: device_store.Chunk,
    elems: usize,
};

/// Acquire tensor `id`'s one buffer (by its dtype's scalar size). A tensor past the
/// binding limit is chunked, which these copies do not address.
fn acquirePacked(ctx: Ctx, id: executable.TensorId, comptime mut: bool) ExecuteProgramError!Packed {
    const hs = ctx.store;
    const meta = hs.meta(id) catch return error.ExecutionFailed;
    if (scalarBytes(meta.dtype) == null or meta.chunks != 1) return error.Unsupported;
    var n: usize = 1;
    for (meta.shape) |d| n *= d;
    const t = if (mut)
        ctx.store.acquireMut(id) catch return error.ExecutionFailed
    else
        ctx.store.acquireConst(id) catch return error.ExecutionFailed;
    errdefer if (mut) hs.releaseMut(t.token) else hs.releaseConst(t.token);
    if (!context.storageBindingFits(ctx, t.len)) return error.Unsupported;
    return .{ .buf = t, .elems = n };
}

// ---- Reshape --------------------------------------------------------------------

/// Packed src -> packed dst with identical scalar content: reshape preserves packed
/// row-major order, so it is one raw buffer copy (any 4-byte-aligned size).
pub fn execPackedCopy(ctx: Ctx, frame: *Frame, dst: executable.TensorId, src: executable.TensorId) ExecuteProgramError!void {
    const hs = ctx.store;
    const dst_meta = hs.meta(dst) catch return error.ExecutionFailed;
    const src_meta = hs.meta(src) catch return error.ExecutionFailed;
    if (dst_meta.dtype != src_meta.dtype) return error.Unsupported;

    const dsrc = try acquirePacked(ctx, src, false);
    defer hs.releaseConst(dsrc.buf.token);
    const ddst = try acquirePacked(ctx, dst, true);
    defer hs.releaseMut(ddst.buf.token);

    if (dsrc.elems != ddst.elems) return error.Unsupported;
    const bytes = dsrc.elems * scalarBytes(src_meta.dtype).?;
    if (bytes % 4 != 0) return error.Unsupported; // copy granularity
    if (bytes > dsrc.buf.len or bytes > ddst.buf.len) return error.ExecutionFailed;
    try recordContiguousCopy(ctx, frame, ctx.devmem.bufferFor(dsrc.buf.handle).?, 0, ctx.devmem.bufferFor(ddst.buf.handle).?, 0, bytes);
}

// ---- Concat --------------------------------------------------------------------

pub fn execConcat(ctx: Ctx, frame: *Frame, s: executable.StepConcatScalar) ExecuteProgramError!void {
    const hs = ctx.store;
    const out_meta = hs.meta(s.out) catch return error.ExecutionFailed;
    const elem = scalarBytes(out_meta.dtype) orelse return error.Unsupported;
    if (elem != 4 and elem != 2) return error.Unsupported;
    const rank: usize = @as(usize, out_meta.rank);
    if (s.axis >= rank) return error.Unsupported;

    var outer: usize = 1;
    var d: usize = 0;
    while (d < s.axis) : (d += 1) outer *= out_meta.shape[d];
    var inner: usize = 1;
    d = s.axis + 1;
    while (d < rank) : (d += 1) inner *= out_meta.shape[d];
    const out_axis = out_meta.shape[s.axis];

    // Large-`outer` concats take the strided-copy kernel below, so the only
    // encoder copies recorded are the small-outer cases — no copy-count cap.
    const n_inputs: usize = @intCast(s.input_count);

    // Validate the complete operation before recording work. f16 inputs can
    // meet at opposite halves of one destination word, so discovering an
    // invalid later input after earlier dispatches were recorded is especially
    // undesirable.
    var expected_axis: usize = 0;
    var validate_i: usize = 0;
    while (validate_i < n_inputs) : (validate_i += 1) {
        const in_meta = hs.meta(s.inputs[validate_i]) catch return error.ExecutionFailed;
        if (in_meta.dtype != out_meta.dtype or @as(usize, in_meta.rank) != rank) return error.Unsupported;
        var validate_d: usize = 0;
        while (validate_d < rank) : (validate_d += 1) {
            if (validate_d != s.axis and in_meta.shape[validate_d] != out_meta.shape[validate_d]) return error.Unsupported;
        }
        expected_axis = std.math.add(usize, expected_axis, in_meta.shape[s.axis]) catch return error.Unsupported;
    }
    if (expected_axis != out_axis) return error.ExecutionFailed;

    const ddst = try acquirePacked(ctx, s.out, true);
    defer hs.releaseMut(ddst.buf.token);
    const dst_buf = ctx.devmem.bufferFor(ddst.buf.handle).?;

    // Past a handful of runs per input, encoder-copy overhead (~1 µs each)
    // dwarfs the data movement — switch to one strided-copy dispatch per input.
    const KERNEL_MIN_RUNS: usize = 8;

    var prefix: usize = 0;
    var i: usize = 0;
    while (i < n_inputs) : (i += 1) {
        const in_id = s.inputs[i];
        const in_meta = hs.meta(in_id) catch return error.ExecutionFailed;
        if (in_meta.dtype != out_meta.dtype or @as(usize, in_meta.rank) != rank) return error.Unsupported;
        const ax_i = in_meta.shape[s.axis];

        const dsrc = try acquirePacked(ctx, in_id, false);
        defer hs.releaseConst(dsrc.buf.token);
        if (dsrc.elems < outer * ax_i * inner) return error.ExecutionFailed;
        const src_buf = ctx.devmem.bufferFor(dsrc.buf.handle).?;

        const run_elems = ax_i * inner;
        const dst_base = prefix * inner;
        if (elem == 2) {
            // Element-addressed: adjacent inputs may meet inside one u32 word.
            const total = std.math.cast(u32, outer * run_elems) orelse return error.Unsupported;
            if ((outer - 1) * out_axis * inner + dst_base + run_elems > ddst.elems) return error.ExecutionFailed;
            var params: GatherParams = .{
                .total = total,
                .rank = 1,
                .base = std.math.cast(u32, dst_base) orelse return error.Unsupported,
                .dshape = @splat(1),
                .sstride = @splat(0),
            };
            params.dshape[0] = std.math.cast(u32, run_elems) orelse return error.Unsupported;
            params.sstride[0] = std.math.cast(u32, out_axis * inner) orelse return error.Unsupported;
            const built = try ctx.pipes.get(view_f16_kernel, "strided_copy_f16");
            const bufs = [_]c.WGPUBuffer{ src_buf, dst_buf };
            const sizes = [_]u64{ dsrc.buf.len, ddst.buf.len };
            try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(total), 1, 1 });
        } else if (outer > KERNEL_MIN_RUNS) {
            const total = std.math.cast(u32, outer * run_elems) orelse return error.Unsupported;
            if ((outer - 1) * out_axis * inner + dst_base + run_elems > ddst.elems) return error.ExecutionFailed;
            var params: GatherParams = .{
                .total = total,
                .rank = 1,
                .base = std.math.cast(u32, dst_base) orelse return error.Unsupported,
                .dshape = @splat(1),
                .sstride = @splat(0),
            };
            params.dshape[0] = std.math.cast(u32, run_elems) orelse return error.Unsupported;
            params.sstride[0] = std.math.cast(u32, out_axis * inner) orelse return error.Unsupported;
            const built = try ctx.pipes.get(view_kernel, "strided_copy_u32");
            const bufs = [_]c.WGPUBuffer{ src_buf, dst_buf };
            const sizes = [_]u64{ dsrc.buf.len, ddst.buf.len };
            try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(total), 1, 1 });
        } else {
            const block_bytes = run_elems * elem;
            var o: usize = 0;
            while (o < outer) : (o += 1) {
                const src_off = o * block_bytes;
                const dst_off = (o * out_axis + prefix) * inner * elem;
                if (dst_off + block_bytes > ddst.buf.len) return error.ExecutionFailed;
                try recordContiguousCopy(ctx, frame, src_buf, src_off, dst_buf, dst_off, block_bytes);
            }
        }
        prefix += ax_i;
    }
    if (prefix != out_axis) return error.ExecutionFailed;
}

// ---- Transpose2D / SliceND -------------------------------------------------------

fn recordGather(ctx: Ctx, frame: *Frame, src: Packed, dst: Packed, params: GatherParams, kernel: KernelDesc, entry: [:0]const u8) ExecuteProgramError!void {
    const built = try ctx.pipes.get(kernel, entry);
    const bufs = [_]c.WGPUBuffer{
        ctx.devmem.bufferFor(src.buf.handle).?,
        ctx.devmem.bufferFor(dst.buf.handle).?,
    };
    const sizes = [_]u64{ src.buf.len, dst.buf.len };
    try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(params.total), 1, 1 });
}

pub fn execTranspose2D(ctx: Ctx, frame: *Frame, s: executable.StepTranspose2DScalar) ExecuteProgramError!void {
    const hs = ctx.store;
    const src_meta = hs.meta(s.src) catch return error.ExecutionFailed;
    const dst_meta = hs.meta(s.dst) catch return error.ExecutionFailed;
    if (src_meta.dtype != dst_meta.dtype) return error.Unsupported;
    const elem = scalarBytes(src_meta.dtype) orelse return error.Unsupported;
    if (elem != 4 and elem != 2) return error.Unsupported;
    if (src_meta.rank != 2 or dst_meta.rank != 2) return error.Unsupported;

    const m = src_meta.shape[0];
    const n = src_meta.shape[1];
    if (dst_meta.shape[0] != n or dst_meta.shape[1] != m) return error.Unsupported;

    const dsrc = try acquirePacked(ctx, s.src, false);
    defer hs.releaseConst(dsrc.buf.token);
    const ddst = try acquirePacked(ctx, s.dst, true);
    defer hs.releaseMut(ddst.buf.token);
    if (dsrc.elems < m * n or ddst.elems < m * n) return error.ExecutionFailed;

    // A word view cannot express this: transpose permutes at ELEMENT granularity,
    // so the two halves of a source word land in different destination words. Both
    // dtypes take the strided gather, f16 addressing elements.
    var params: GatherParams = .{
        .total = std.math.cast(u32, m * n) orelse return error.Unsupported,
        .rank = 2,
        .base = 0,
        .dshape = @splat(1),
        .sstride = @splat(0),
    };
    // dst[j, i] = src[i, j]: dst dim0 = j (stride 1 in src), dim1 = i (stride n).
    params.dshape[0] = @intCast(n);
    params.dshape[1] = @intCast(m);
    params.sstride[0] = 1;
    params.sstride[1] = @intCast(n);
    if (elem == 2)
        try recordGather(ctx, frame, dsrc, ddst, params, view_f16_kernel, "gather_nd_f16")
    else
        try recordGather(ctx, frame, dsrc, ddst, params, view_kernel, "gather_nd_u32");
}

pub fn execSliceND(ctx: Ctx, frame: *Frame, s: executable.StepSliceNDScalar) ExecuteProgramError!void {
    const hs = ctx.store;
    const src_meta = hs.meta(s.src) catch return error.ExecutionFailed;
    const dst_meta = hs.meta(s.dst) catch return error.ExecutionFailed;
    if (src_meta.dtype != dst_meta.dtype) return error.Unsupported;
    // The gather kernel addresses 4-byte words. A narrower scalar is handled by
    // viewing the innermost axis in words when both extents AND the slice start
    // divide evenly there; f16 that doesn't takes the element-addressed gather.
    const per_word = elementsPerWord(src_meta.dtype) orelse return error.Unsupported;

    const rank: usize = @as(usize, s.rank);
    if (rank == 0 or rank > MAX_RANK) return error.Unsupported;
    if (@as(usize, src_meta.rank) != rank or @as(usize, dst_meta.rank) != rank) return error.Unsupported;

    // An f16 whose innermost axis is odd in any of src extent / dst extent / start
    // has no word view; it takes the element-addressed gather instead.
    const last = rank - 1;
    const half_word_f16 = per_word == 2 and
        (src_meta.shape[last] % 2 != 0 or dst_meta.shape[last] % 2 != 0 or s.starts[last] % 2 != 0);
    const lanes: usize = if (half_word_f16) 1 else per_word;

    const dsrc = try acquirePacked(ctx, s.src, false);
    defer hs.releaseConst(dsrc.buf.token);
    const ddst = try acquirePacked(ctx, s.dst, true);
    defer hs.releaseMut(ddst.buf.token);

    // Word view of the innermost axis; identity when the scalar is 4 bytes.
    var src_w: [MAX_RANK]usize = undefined;
    var dst_w: [MAX_RANK]usize = undefined;
    var start_w: [MAX_RANK]usize = undefined;
    var total_elems: usize = 1;
    for (0..rank) |i| {
        if (s.starts[i] + dst_meta.shape[i] > src_meta.shape[i]) return error.ExecutionFailed;
        src_w[i] = src_meta.shape[i];
        dst_w[i] = dst_meta.shape[i];
        start_w[i] = s.starts[i];
        total_elems = std.math.mul(usize, total_elems, dst_meta.shape[i]) catch return error.Unsupported;
    }
    // The guard for narrower scalars (i8) whose innermost axis is not whole words.
    if (lanes != 1) {
        if (src_w[last] % lanes != 0 or dst_w[last] % lanes != 0 or start_w[last] % lanes != 0) return error.Unsupported;
        src_w[last] /= lanes;
        dst_w[last] /= lanes;
        start_w[last] /= lanes;
    }
    if (ddst.elems < total_elems) return error.ExecutionFailed;

    // Packed row-major src strides (words) and the flat word offset of `starts`.
    var strides: [MAX_RANK]usize = undefined;
    var stride: usize = 1;
    var d: usize = rank;
    while (d > 0) : (d -= 1) {
        strides[d - 1] = stride;
        stride = std.math.mul(usize, stride, src_w[d - 1]) catch return error.Unsupported;
    }
    var base: usize = 0;
    var total: usize = 1;
    d = 0;
    while (d < rank) : (d += 1) {
        base += start_w[d] * strides[d];
        total = std.math.mul(usize, total, dst_w[d]) catch return error.Unsupported;
    }

    var params: GatherParams = .{
        .total = std.math.cast(u32, total) orelse return error.Unsupported,
        .rank = @intCast(rank),
        .base = std.math.cast(u32, base) orelse return error.Unsupported,
        .dshape = @splat(1),
        .sstride = @splat(0),
    };
    d = 0;
    while (d < rank) : (d += 1) {
        params.dshape[d] = std.math.cast(u32, dst_w[d]) orelse return error.Unsupported;
        params.sstride[d] = std.math.cast(u32, strides[d]) orelse return error.Unsupported;
    }
    if (half_word_f16)
        try recordGather(ctx, frame, dsrc, ddst, params, view_f16_kernel, "gather_nd_f16")
    else
        try recordGather(ctx, frame, dsrc, ddst, params, view_kernel, "gather_nd_u32");
}
