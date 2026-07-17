// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! View materializations + concat for the GPU backend. Under the GPU tile
//! policy these tensors are single packed tiles, which collapses most of the
//! CPU's scalar re-tiling machinery into contiguous copies:
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

const tensor_store_mod = @import("../../../runtime/tensor_store.zig");
const TensorMetaT = tensor_store_mod.TensorMeta;

/// Cap on strided runs recorded for a multi-tile packed copy.
const MAX_COPY_RUNS: usize = 8192;

fn prod(xs: []const usize) usize {
    var p: usize = 1;
    for (xs) |x| p *= x;
    return p;
}

/// A tensor whose tiling splits exactly ONE dim (all others single full tiles):
/// returns that split dim `p`. Its tiles are, per outer index, contiguous runs of
/// `tile_shape[p] * inner` elements — reconstructible into packed order by strided
/// copies. Returns null when zero or many dims are split.
fn singleSplitDim(meta: TensorMetaT) ?usize {
    const rank = meta.tile_counts.len;
    var split: ?usize = null;
    var d: usize = 0;
    while (d < rank) : (d += 1) {
        if (meta.tile_counts[d] != 1) {
            if (split != null) return null;
            split = d;
        }
    }
    return split;
}

/// Packed src -> packed dst with identical scalar content: reshape and retile
/// preserve packed row-major order. Single-tile both sides is one raw buffer copy
/// (any 4-byte-aligned dtype). Otherwise exactly one side must be a single packed
/// tile and the other split along a single dim; we reconstruct packed order with
/// `outer * n_tiles` strided buffer copies (4-byte scalars only).
pub fn execPackedCopy(ctx: Ctx, frame: *Frame, dst: executable.TensorId, src: executable.TensorId) ExecuteProgramError!void {
    const hs = ctx.rstore.tensorStore();
    const dst_meta = hs.meta(dst) catch return error.ExecutionFailed;
    const src_meta = hs.meta(src) catch return error.ExecutionFailed;
    if (dst_meta.dtype != src_meta.dtype) return error.Unsupported;

    const src_tiles = context.totalTiles(src_meta);
    const dst_tiles = context.totalTiles(dst_meta);

    // Fast path: single tile both sides — one whole-buffer copy (any dtype).
    if (src_tiles == 1 and dst_tiles == 1) {
        const dsrc = try acquirePacked(ctx, src, false);
        defer hs.releaseConst(dsrc.tile.token);
        const ddst = try acquirePacked(ctx, dst, true);
        defer hs.releaseMut(ddst.tile.token);

        if (dsrc.elems != ddst.elems) return error.Unsupported;
        const elem = scalarBytes(src_meta.dtype).?;
        const bytes = dsrc.elems * elem;
        if (bytes % 4 != 0) return error.Unsupported; // copy granularity
        if (bytes > dsrc.tile.len or bytes > ddst.tile.len) return error.ExecutionFailed;

        try recordContiguousCopy(
            ctx,
            frame,
            ctx.devmem.bufferFor(dsrc.tile.handle).?,
            0,
            ctx.devmem.bufferFor(ddst.tile.handle).?,
            0,
            bytes,
        );
        return;
    }

    // Multi-tile. Reshape/retile preserve packed row-major order, so we move data
    // through a packed representation. A single-tile side IS packed; a multi-tile
    // side is (un)packed per-tile with a scatter/gather kernel that maps each tile
    // element to its flat packed offset (handles ANY tiling, incl. several split
    // dims). Exactly one multi side round-trips against the other's buffer; two
    // multi sides round-trip through a packed scratch.
    const elem = scalarBytes(src_meta.dtype) orelse return error.Unsupported;
    if (elem != 4) return error.Unsupported;
    const total_elems = prod(src_meta.shape);
    if (total_elems != prod(dst_meta.shape)) return error.Unsupported;

    const src_single = src_tiles == 1;
    const dst_single = dst_tiles == 1;

    // Resolve the packed buffer: the single side's tile, or a scratch for both-multi.
    var packed_buf: c.WGPUBuffer = undefined;
    var packed_len: u64 = 0;
    var packed_token: ?usize = null;
    var packed_mut = false;
    defer if (packed_token) |tok| {
        if (packed_mut) hs.releaseMut(tok) else hs.releaseConst(tok);
    };
    if (src_single) {
        const pt = ctx.rstore.acquireTileDeviceConstLinear(src, 0) catch return error.ExecutionFailed;
        packed_token = pt.token;
        packed_buf = ctx.devmem.bufferFor(pt.handle) orelse return error.ExecutionFailed;
        packed_len = pt.len;
    } else if (dst_single) {
        const pt = ctx.rstore.acquireTileDeviceMutLinear(dst, 0) catch return error.ExecutionFailed;
        packed_token = pt.token;
        packed_mut = true;
        packed_buf = ctx.devmem.bufferFor(pt.handle) orelse return error.ExecutionFailed;
        packed_len = pt.len;
    } else {
        const bytes: u64 = @as(u64, total_elems) * elem;
        if (!context.storageBindingFits(ctx, bytes)) return error.Unsupported;
        packed_buf = try ctx.scratch.ensure(ctx.gpu, bytes);
        packed_len = bytes;
    }

    // src(tiled) -> packed, then packed -> dst(tiled). A single side is a no-op here.
    if (!src_single) try gatherScatterTiles(ctx, frame, src_meta, src, packed_buf, packed_len, false);
    if (!dst_single) try gatherScatterTiles(ctx, frame, dst_meta, dst, packed_buf, packed_len, true);
}

/// Move a multi-tile tensor to/from a contiguous packed buffer, one dispatch per
/// tile. `gather` true fills each dst tile FROM packed (packed -> tiled); false
/// scatters each src tile INTO packed (tiled -> packed). Each tile element maps to
/// its flat packed offset via row-major strides of the tensor's logical shape, so
/// arbitrary tilings (any number of split dims) are handled.
fn gatherScatterTiles(
    ctx: Ctx,
    frame: *Frame,
    meta: TensorMetaT,
    id: executable.TensorId,
    packed_buf: c.WGPUBuffer,
    packed_len: u64,
    gather: bool,
) ExecuteProgramError!void {
    const hs = ctx.rstore.tensorStore();
    const rank: usize = meta.tile_counts.len;
    if (rank == 0 or rank > MAX_RANK) return error.Unsupported;

    // Flat row-major element strides of the logical shape.
    var pstride: [MAX_RANK]usize = undefined;
    var acc: usize = 1;
    var d: usize = rank;
    while (d > 0) : (d -= 1) {
        pstride[d - 1] = acc;
        acc = std.math.mul(usize, acc, meta.shape[d - 1]) catch return error.Unsupported;
    }

    const entry = if (gather) "gather_nd_u32" else "scatter_nd_u32";
    const built = try ctx.pipes.get(view_kernel, entry);

    const n_tiles = context.totalTiles(meta);
    var coords_buf: [MAX_RANK]usize = undefined;
    var ti: usize = 0;
    while (ti < n_tiles) : (ti += 1) {
        const coords = coords_buf[0..rank];
        tensor_store_mod.decodeTileCoords(meta, ti, coords) catch return error.ExecutionFailed;

        var base: usize = 0;
        var total: usize = 1;
        var extents: [MAX_RANK]usize = undefined;
        var params: GatherParams = .{ .total = 0, .rank = @intCast(rank), .base = 0, .dshape = @splat(1), .sstride = @splat(0) };
        d = 0;
        while (d < rank) : (d += 1) {
            const origin = coords[d] * meta.tile_shape[d];
            const extent = @min(meta.tile_shape[d], meta.shape[d] - origin);
            extents[d] = extent;
            base += origin * pstride[d];
            total = std.math.mul(usize, total, extent) catch return error.Unsupported;
            params.dshape[d] = std.math.cast(u32, extent) orelse return error.Unsupported;
            params.sstride[d] = std.math.cast(u32, pstride[d]) orelse return error.Unsupported;
        }
        params.total = std.math.cast(u32, total) orelse return error.Unsupported;
        params.base = std.math.cast(u32, base) orelse return error.Unsupported;

        // Contiguity test: this tile is a single packed slab (packed[base .. base+total])
        // iff — scanning to the outermost partial dim d0 — every inner dim (d > d0) is
        // full-extent and every outer dim (d < d0) has extent 1. Then the strided
        // gather/scatter kernel is just a buffer copy: the hardware copy engine at
        // ~DRAM bandwidth instead of a per-element compute pass with div/mod. This is
        // the common case for reshapes that only re-group whole trailing dims (e.g.
        // [1,T,1024] <-> [1,T,8,128] one-row-tiled along T → T contiguous 1024-slabs).
        var contiguous = true;
        var saw_partial = false;
        d = 0;
        while (d < rank) : (d += 1) {
            const full = extents[d] == meta.shape[d];
            if (saw_partial) {
                if (!full) {
                    contiguous = false;
                    break;
                }
            } else if (!full) {
                // d is the outermost partial dim; all dims above it must be extent 1.
                var e: usize = 0;
                while (e < d) : (e += 1) {
                    if (extents[e] != 1) {
                        contiguous = false;
                        break;
                    }
                }
                saw_partial = true;
            }
        }

        const dt = if (gather)
            ctx.rstore.acquireTileDeviceMutLinear(id, ti) catch return error.ExecutionFailed
        else
            ctx.rstore.acquireTileDeviceConstLinear(id, ti) catch return error.ExecutionFailed;
        defer if (gather) hs.releaseMut(dt.token) else hs.releaseConst(dt.token);
        if (!context.storageBindingFits(ctx, dt.len)) return error.Unsupported;
        const tile_buf = ctx.devmem.bufferFor(dt.handle).?;

        if (contiguous) {
            // elem == 4 (caller-checked), so all offsets/lengths are 4-aligned.
            const bytes: u64 = @as(u64, total) * 4;
            const off: u64 = @as(u64, base) * 4;
            if (bytes > dt.len or off + bytes > packed_len) return error.ExecutionFailed;
            if (gather)
                try recordContiguousCopy(ctx, frame, packed_buf, off, tile_buf, 0, bytes) // packed -> tile
            else
                try recordContiguousCopy(ctx, frame, tile_buf, 0, packed_buf, off, bytes); // tile -> packed
            continue;
        }

        // gather_nd: o[idx]=x[strided] → x=packed, o=tile. scatter_nd: o[strided]=x[idx] → x=tile, o=packed.
        const bufs = if (gather)
            [_]c.WGPUBuffer{ packed_buf, tile_buf }
        else
            [_]c.WGPUBuffer{ tile_buf, packed_buf };
        const sizes = if (gather)
            [_]u64{ packed_len, dt.len }
        else
            [_]u64{ dt.len, packed_len };
        try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(params.total), 1, 1 });
    }
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
                try recordContiguousCopy(ctx, frame, src_buf, src_off, dst_buf, dst_off, block_bytes);
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

/// Slice a src tiled along a SINGLE dim `sp` (all other dims taken whole) into a
/// single packed dst — copy each (outer index, overlapping src tile) segment. The
/// src tile layout is `[outer dims (full), sp chunk, inner dims (full)]` row-major,
/// so for a fixed outer index the sliced `[sp range × inner]` block is contiguous.
/// Handles slicing the last dim (QKV / gate-up matmul-output splits) AND an outer
/// dim (e.g. dropping leading rows of a long tiled audio vector).
fn execSliceNDMultiTile(ctx: Ctx, frame: *Frame, s: executable.StepSliceNDScalar, src_meta: TensorMetaT, dst_meta: TensorMetaT, rank: usize) ExecuteProgramError!void {
    const hs = ctx.rstore.tensorStore();
    const elem: usize = 4;

    const sp = singleSplitDim(src_meta) orelse return error.Unsupported;
    if (context.totalTiles(dst_meta) != 1) return error.Unsupported;

    // Every dim except the split dim must be taken whole (start 0, full extent).
    var d: usize = 0;
    while (d < rank) : (d += 1) {
        if (d == sp) continue;
        if (s.starts[d] != 0 or dst_meta.shape[d] != src_meta.shape[d]) return error.Unsupported;
    }

    const outer = prod(src_meta.shape[0..sp]);
    const inner = prod(src_meta.shape[sp + 1 .. rank]);
    const start = s.starts[sp];
    const extent = dst_meta.shape[sp];
    const last = src_meta.shape[sp];
    if (start + extent > last) return error.ExecutionFailed;
    const tw = src_meta.tile_shape[sp];
    const n_src_tiles = src_meta.tile_counts[sp];
    if (tw == 0) return error.Unsupported;
    // Guard against pathological run counts (each (tile, outer) is one copy).
    if (outer * n_src_tiles > MAX_COPY_RUNS) return error.Unsupported;

    const ddst = try acquirePacked(ctx, s.dst, true);
    defer hs.releaseMut(ddst.tile.token);
    if (ddst.elems < outer * extent * inner) return error.ExecutionFailed;
    const dst_buf = ctx.devmem.bufferFor(ddst.tile.handle).?;

    var t: usize = 0;
    while (t < n_src_tiles) : (t += 1) {
        const tile_lo = t * tw;
        const this_tw = @min(tw, last - tile_lo);
        const ov_lo = @max(start, tile_lo);
        const ov_hi = @min(start + extent, tile_lo + this_tw);
        if (ov_lo >= ov_hi) continue;

        const dt = ctx.rstore.acquireTileDeviceConstLinear(s.src, t) catch return error.ExecutionFailed;
        defer hs.releaseConst(dt.token);
        if (!context.storageBindingFits(ctx, dt.len)) return error.Unsupported;
        const src_buf = ctx.devmem.bufferFor(dt.handle).?;

        const run_bytes: u64 = @as(u64, ov_hi - ov_lo) * inner * elem;
        var o: usize = 0;
        while (o < outer) : (o += 1) {
            const src_off: u64 = @as(u64, o * this_tw + (ov_lo - tile_lo)) * inner * elem;
            const dst_off: u64 = @as(u64, o * extent + (ov_lo - start)) * inner * elem;
            if (src_off + run_bytes > dt.len or dst_off + run_bytes > ddst.tile.len) return error.ExecutionFailed;
            try recordContiguousCopy(ctx, frame, src_buf, src_off, dst_buf, dst_off, run_bytes);
        }
    }
}

/// Slice from a single packed src into a MULTI-tile dst: run one `gather_nd`
/// dispatch per dst tile, each pulling that tile's elements from the (contiguous
/// row-major) src at the sliced offset. Handles e.g. slicing a single-tile STFT
/// output whose consumer wants a row-tiled layout.
fn execSliceNDSingleSrcMultiDst(ctx: Ctx, frame: *Frame, s: executable.StepSliceNDScalar, src_meta: TensorMetaT, dst_meta: TensorMetaT, rank: usize) ExecuteProgramError!void {
    const hs = ctx.rstore.tensorStore();
    const dsrc = try acquirePacked(ctx, s.src, false);
    defer hs.releaseConst(dsrc.tile.token);
    const src_buf = ctx.devmem.bufferFor(dsrc.tile.handle).?;
    try sliceGatherFromBuffer(ctx, frame, s, src_meta, dst_meta, rank, src_buf, dsrc.tile.len);
}

/// Slice from a CONTIGUOUS row-major src buffer (of `src_meta.shape`) into `dst`
/// (any tile count): one `gather_nd` dispatch per dst tile, each pulling its
/// elements from the src at the sliced flat offset. The src buffer is either a
/// single-tile src's packed tile OR a scratch buffer we packed a multi-tile src
/// into (see `execSliceNDViaScratch`) — both are contiguous over `src_meta.shape`.
fn sliceGatherFromBuffer(
    ctx: Ctx,
    frame: *Frame,
    s: executable.StepSliceNDScalar,
    src_meta: TensorMetaT,
    dst_meta: TensorMetaT,
    rank: usize,
    src_buf: c.WGPUBuffer,
    src_len: u64,
) ExecuteProgramError!void {
    const hs = ctx.rstore.tensorStore();

    var strides: [MAX_RANK]usize = undefined;
    var stride: usize = 1;
    var d: usize = rank;
    while (d > 0) : (d -= 1) {
        strides[d - 1] = stride;
        stride = std.math.mul(usize, stride, src_meta.shape[d - 1]) catch return error.Unsupported;
    }
    d = 0;
    while (d < rank) : (d += 1) {
        if (s.starts[d] + dst_meta.shape[d] > src_meta.shape[d]) return error.ExecutionFailed;
    }

    const n_tiles = context.totalTiles(dst_meta);
    var coords_buf: [MAX_RANK]usize = undefined;
    var ti: usize = 0;
    while (ti < n_tiles) : (ti += 1) {
        const coords = coords_buf[0..rank];
        tensor_store_mod.decodeTileCoords(dst_meta, ti, coords) catch return error.ExecutionFailed;

        // This dst tile's element origin + actual (edge-clamped) shape, and the
        // src flat offset of its first element (origin + slice starts).
        var base: usize = 0;
        var total: usize = 1;
        var params: GatherParams = .{ .total = 0, .rank = @intCast(rank), .base = 0, .dshape = @splat(1), .sstride = @splat(0) };
        d = 0;
        while (d < rank) : (d += 1) {
            const origin = coords[d] * dst_meta.tile_shape[d];
            const extent = @min(dst_meta.tile_shape[d], dst_meta.shape[d] - origin);
            base += (s.starts[d] + origin) * strides[d];
            total = std.math.mul(usize, total, extent) catch return error.Unsupported;
            params.dshape[d] = std.math.cast(u32, extent) orelse return error.Unsupported;
            params.sstride[d] = std.math.cast(u32, strides[d]) orelse return error.Unsupported;
        }
        params.total = std.math.cast(u32, total) orelse return error.Unsupported;
        params.base = std.math.cast(u32, base) orelse return error.Unsupported;

        const dt = ctx.rstore.acquireTileDeviceMutLinear(s.dst, ti) catch return error.ExecutionFailed;
        defer hs.releaseMut(dt.token);
        if (!context.storageBindingFits(ctx, dt.len)) return error.Unsupported;
        const built = try ctx.pipes.get(view_kernel, "gather_nd_u32");
        const bufs = [_]c.WGPUBuffer{ src_buf, ctx.devmem.bufferFor(dt.handle).? };
        const sizes = [_]u64{ src_len, dt.len };
        try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(params.total), 1, 1 });
    }
}

/// True iff `execSliceNDMultiTile`'s fast path structurally applies: the src is
/// split along exactly one dim, the dst is a single tile, and every non-split dim
/// is taken whole (so the slice runs along the split dim only). Checked upfront so
/// the general scratch fallback is chosen WITHOUT the fast path recording partial
/// work first.
fn sliceMultiTileFastEligible(src_meta: TensorMetaT, dst_meta: TensorMetaT, s: executable.StepSliceNDScalar, rank: usize) bool {
    const sp = singleSplitDim(src_meta) orelse return false;
    if (context.totalTiles(dst_meta) != 1) return false;
    var d: usize = 0;
    while (d < rank) : (d += 1) {
        if (d == sp) continue;
        if (s.starts[d] != 0 or dst_meta.shape[d] != src_meta.shape[d]) return false;
    }
    const tw = src_meta.tile_shape[sp];
    if (tw == 0) return false;
    const outer = prod(src_meta.shape[0..sp]);
    if (outer * src_meta.tile_counts[sp] > MAX_COPY_RUNS) return false;
    return true;
}

/// General multi-tile-src slice: materialize the src into a contiguous packed
/// scratch (via the same tile pack used by reshape/retile — a buffer copy for
/// contiguous tiles, strided otherwise), then slice from the scratch. Handles any
/// slice on any tiling (e.g. a streaming attention cache tiled by HEAD but sliced
/// along TIME), which the single-split-dim fast path cannot.
fn execSliceNDViaScratch(ctx: Ctx, frame: *Frame, s: executable.StepSliceNDScalar, src_meta: TensorMetaT, dst_meta: TensorMetaT, rank: usize) ExecuteProgramError!void {
    const total_src = prod(src_meta.shape);
    const bytes: u64 = @as(u64, total_src) * 4;
    if (!context.storageBindingFits(ctx, bytes)) return error.Unsupported;
    const scratch = try ctx.scratch.ensure(ctx.gpu, bytes);
    // src (multi-tile) -> contiguous packed scratch, then gather the slice out.
    try gatherScatterTiles(ctx, frame, src_meta, s.src, scratch, bytes, false);
    try sliceGatherFromBuffer(ctx, frame, s, src_meta, dst_meta, rank, scratch, bytes);
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

    // Multi-tile src. Fast path (single split dim, slice along it, whole elsewhere,
    // single-tile dst): per-(outer, tile) contiguous copies — the QKV / gate-up
    // splits of an N-tiled matmul output, or dropping leading rows of a tiled
    // vector. Otherwise (e.g. a cache tiled by HEAD but sliced along TIME) fall back
    // to packing the src into a contiguous scratch and slicing from that.
    if (context.totalTiles(src_meta) != 1) {
        if (sliceMultiTileFastEligible(src_meta, dst_meta, s, rank))
            return execSliceNDMultiTile(ctx, frame, s, src_meta, dst_meta, rank);
        return execSliceNDViaScratch(ctx, frame, s, src_meta, dst_meta, rank);
    }

    // Single-tile src, multi-tile dst: gather each dst tile out of the src.
    if (context.totalTiles(dst_meta) != 1) return execSliceNDSingleSrcMultiDst(ctx, frame, s, src_meta, dst_meta, rank);

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
