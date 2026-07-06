// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! View materializations + concat for the GPU backend. Under the GPU tile
//! policy these tensors are single packed tiles, which collapses most of the
//! CPU's scalar re-tiling machinery into buffer copies:
//!   - ReshapeScalar / ReTileCopyScalar: packed row-major order is invariant
//!     under reshape, and a single-tile -> single-tile retile is the identity
//!     layout — both are ONE whole-buffer `CopyBufferToBuffer`.
//!   - ConcatScalar: contiguous [axis-block * inner] runs land at strided
//!     offsets in the output — one `strided_copy_u32` dispatch per input
//!     (encoder copies when there are only a few runs).
//!   - Transpose2DScalar / SliceNDScalar: the `gather_nd_u32` kernel
//!     (kernels/view.wgsl) pulls each dst element from a strided src offset;
//!     transpose is just the rank-2 parameterization with swapped strides.
//! All paths move 4-byte scalars (f32/i32) except reshape/retile, which are
//! raw byte copies (any dtype with 4-byte-aligned tiles).

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

const view_kernel: KernelDesc = .{ .name = "view", .wgsl = @embedFile("../kernels/view.wgsl") };

const WG_1D: u32 = 64;
const MAX_RANK: usize = 8;

/// Field order matches `struct Params` in view.wgsl (arrays are vec4-packed).
const GatherParams = extern struct {
    total: u32,
    rank: u32,
    base: u32,
    _pad: u32 = 0,
    dshape: [8]u32,
    sstride: [8]u32,
};

fn groups1D(n: u32) u32 {
    return @max(1, @min(context.ceilDiv(n, WG_1D), context.MAX_GROUPS_1D));
}

fn scalarBytes(dt: types.DType) ?usize {
    return switch (dt) {
        .f32, .i32 => 4,
        .f16 => 2,
        .i8 => 1,
        else => null,
    };
}

const PackedTile = struct {
    tile: resident_mod.TileRefDevice,
    elems: usize,
};

/// Acquire tensor `id`'s single packed tile (by its dtype's scalar size).
fn acquirePacked(ctx: Ctx, id: executable.TensorId, comptime mut: bool) ExecuteProgramError!PackedTile {
    const hs = ctx.rstore.tensorStore();
    const meta = hs.meta(id) catch return error.ExecutionFailed;
    const elem = scalarBytes(meta.dtype) orelse return error.Unsupported;
    if (context.totalTiles(meta) != 1) return error.Unsupported;
    const t = if (mut)
        ctx.rstore.acquireTileDeviceMutLinear(id, 0) catch return error.ExecutionFailed
    else
        ctx.rstore.acquireTileDeviceConstLinear(id, 0) catch return error.ExecutionFailed;
    errdefer if (mut) hs.releaseMut(t.token) else hs.releaseConst(t.token);
    const rank: usize = @as(usize, t.rank);
    const n = context.packedElemsSized(t.rank, t.shape_mem[0..rank], t.strides_mem[0..rank], elem) orelse return error.Unsupported;
    if (!context.storageBindingFits(ctx, t.len)) return error.Unsupported;
    return .{ .tile = t, .elems = n };
}

// ---- Reshape / ReTile ----------------------------------------------------------

/// Packed single-tile src -> packed single-tile dst with identical scalar
/// content: reshape and (single-tile) retile are both a raw buffer copy.
pub fn execPackedCopy(ctx: Ctx, frame: *Frame, dst: executable.TensorId, src: executable.TensorId) ExecuteProgramError!void {
    const hs = ctx.rstore.tensorStore();
    const dst_meta = hs.meta(dst) catch return error.ExecutionFailed;
    const src_meta = hs.meta(src) catch return error.ExecutionFailed;
    if (dst_meta.dtype != src_meta.dtype) return error.Unsupported;

    const dsrc = try acquirePacked(ctx, src, false);
    defer hs.releaseConst(dsrc.tile.token);
    const ddst = try acquirePacked(ctx, dst, true);
    defer hs.releaseMut(ddst.tile.token);

    if (dsrc.elems != ddst.elems) return error.Unsupported;
    const elem = scalarBytes(src_meta.dtype).?;
    const bytes = dsrc.elems * elem;
    if (bytes % 4 != 0) return error.Unsupported; // copy granularity
    if (bytes > dsrc.tile.len or bytes > ddst.tile.len) return error.ExecutionFailed;

    frame.recordCopy(
        ctx.devmem.bufferFor(dsrc.tile.handle).?,
        0,
        ctx.devmem.bufferFor(ddst.tile.handle).?,
        0,
        bytes,
    );
}

// ---- Concat --------------------------------------------------------------------

pub fn execConcat(ctx: Ctx, frame: *Frame, s: executable.StepConcatScalar) ExecuteProgramError!void {
    const hs = ctx.rstore.tensorStore();
    const out_meta = hs.meta(s.out) catch return error.ExecutionFailed;
    const elem = scalarBytes(out_meta.dtype) orelse return error.Unsupported;
    if (elem != 4) return error.Unsupported; // aligned copies at any offset
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

    const ddst = try acquirePacked(ctx, s.out, true);
    defer hs.releaseMut(ddst.tile.token);
    const dst_buf = ctx.devmem.bufferFor(ddst.tile.handle).?;

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
        defer hs.releaseConst(dsrc.tile.token);
        if (dsrc.elems < outer * ax_i * inner) return error.ExecutionFailed;
        const src_buf = ctx.devmem.bufferFor(dsrc.tile.handle).?;

        const run_elems = ax_i * inner;
        const dst_base = prefix * inner;
        if (outer > KERNEL_MIN_RUNS) {
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
            const sizes = [_]u64{ dsrc.tile.len, ddst.tile.len };
            try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(total), 1, 1 });
        } else {
            const block_bytes = run_elems * elem;
            var o: usize = 0;
            while (o < outer) : (o += 1) {
                const src_off = o * block_bytes;
                const dst_off = (o * out_axis + prefix) * inner * elem;
                if (dst_off + block_bytes > ddst.tile.len) return error.ExecutionFailed;
                frame.recordCopy(src_buf, src_off, dst_buf, dst_off, block_bytes);
            }
        }
        prefix += ax_i;
    }
    if (prefix != out_axis) return error.ExecutionFailed;
}

// ---- Transpose2D / SliceND -------------------------------------------------------

fn recordGather(ctx: Ctx, frame: *Frame, src: PackedTile, dst: PackedTile, params: GatherParams) ExecuteProgramError!void {
    const built = try ctx.pipes.get(view_kernel, "gather_nd_u32");
    const bufs = [_]c.WGPUBuffer{
        ctx.devmem.bufferFor(src.tile.handle).?,
        ctx.devmem.bufferFor(dst.tile.handle).?,
    };
    const sizes = [_]u64{ src.tile.len, dst.tile.len };
    try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(params.total), 1, 1 });
}

pub fn execTranspose2D(ctx: Ctx, frame: *Frame, s: executable.StepTranspose2DScalar) ExecuteProgramError!void {
    const hs = ctx.rstore.tensorStore();
    const src_meta = hs.meta(s.src) catch return error.ExecutionFailed;
    const dst_meta = hs.meta(s.dst) catch return error.ExecutionFailed;
    if (src_meta.dtype != dst_meta.dtype) return error.Unsupported;
    if ((scalarBytes(src_meta.dtype) orelse return error.Unsupported) != 4) return error.Unsupported;
    if (src_meta.rank != 2 or dst_meta.rank != 2) return error.Unsupported;

    const m = src_meta.shape[0];
    const n = src_meta.shape[1];
    if (dst_meta.shape[0] != n or dst_meta.shape[1] != m) return error.Unsupported;

    const dsrc = try acquirePacked(ctx, s.src, false);
    defer hs.releaseConst(dsrc.tile.token);
    const ddst = try acquirePacked(ctx, s.dst, true);
    defer hs.releaseMut(ddst.tile.token);
    if (dsrc.elems < m * n or ddst.elems < m * n) return error.ExecutionFailed;

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
    try recordGather(ctx, frame, dsrc, ddst, params);
}

pub fn execSliceND(ctx: Ctx, frame: *Frame, s: executable.StepSliceNDScalar) ExecuteProgramError!void {
    const hs = ctx.rstore.tensorStore();
    const src_meta = hs.meta(s.src) catch return error.ExecutionFailed;
    const dst_meta = hs.meta(s.dst) catch return error.ExecutionFailed;
    if (src_meta.dtype != dst_meta.dtype) return error.Unsupported;
    if ((scalarBytes(src_meta.dtype) orelse return error.Unsupported) != 4) return error.Unsupported;

    const rank: usize = @as(usize, s.rank);
    if (rank == 0 or rank > MAX_RANK) return error.Unsupported;
    if (@as(usize, src_meta.rank) != rank or @as(usize, dst_meta.rank) != rank) return error.Unsupported;

    const dsrc = try acquirePacked(ctx, s.src, false);
    defer hs.releaseConst(dsrc.tile.token);
    const ddst = try acquirePacked(ctx, s.dst, true);
    defer hs.releaseMut(ddst.tile.token);

    // Packed row-major src strides (elements) and the flat offset of `starts`.
    var strides: [MAX_RANK]usize = undefined;
    var stride: usize = 1;
    var d: usize = rank;
    while (d > 0) : (d -= 1) {
        strides[d - 1] = stride;
        stride = std.math.mul(usize, stride, src_meta.shape[d - 1]) catch return error.Unsupported;
    }
    var base: usize = 0;
    var total: usize = 1;
    d = 0;
    while (d < rank) : (d += 1) {
        if (s.starts[d] + dst_meta.shape[d] > src_meta.shape[d]) return error.ExecutionFailed;
        base += s.starts[d] * strides[d];
        total = std.math.mul(usize, total, dst_meta.shape[d]) catch return error.Unsupported;
    }
    if (ddst.elems < total) return error.ExecutionFailed;

    var params: GatherParams = .{
        .total = std.math.cast(u32, total) orelse return error.Unsupported,
        .rank = @intCast(rank),
        .base = std.math.cast(u32, base) orelse return error.Unsupported,
        .dshape = @splat(1),
        .sstride = @splat(0),
    };
    d = 0;
    while (d < rank) : (d += 1) {
        params.dshape[d] = std.math.cast(u32, dst_meta.shape[d]) orelse return error.Unsupported;
        params.sstride[d] = std.math.cast(u32, strides[d]) orelse return error.Unsupported;
    }
    try recordGather(ctx, frame, dsrc, ddst, params);
}
