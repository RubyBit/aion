// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! The decode-loop data-movement ops: GatherRows (embedding lookup), RoPE1D,
//! and SequenceAppend.
//!
//! Every index resolves ON DEVICE. There is no host-index fallback: a layout
//! without a device form returns `error.Unsupported` so the gap in GPU coverage
//! is visible, rather than hidden behind a CPU path costing a synchronization
//! per step. Operands split across bindings are handled by dispatching per tile
//! tuple, each dispatch carrying the tile's index range (see `kernels/gather.wgsl`).
//!
//! RoPE reads positions on-device (no host round-trip): one element-wise
//! dispatch per x tile, `kernels/rope.wgsl`.

const std = @import("std");
const wgpu = @import("../wgpu.zig");
const pipelines = @import("../pipelines.zig");
const context = @import("../context.zig");
const backend_mod = @import("../../backend.zig");
const tensor_store_mod = @import("../../../runtime/tensor_store.zig");
const types = @import("../../types.zig");
const device_store = @import("../../../runtime/device_store.zig");
const executable = @import("../../../runtime/executable.zig");

const c = wgpu.c;
const Ctx = context.Ctx;
const Frame = @import("../frame.zig").Frame;
const ExecuteProgramError = backend_mod.ExecuteProgramError;
const KernelDesc = pipelines.KernelDesc;
const TensorMeta = tensor_store_mod.TensorMeta;

const rope_kernel: KernelDesc = .{ .name = "rope", .wgsl = @embedFile("../kernels/rope.wgsl") };
const dequant_kernel: KernelDesc = .{ .name = "dequant", .wgsl = @embedFile("../kernels/dequant.wgsl") };
const gather_kernel: KernelDesc = .{ .name = "gather", .wgsl = @embedFile("../kernels/gather.wgsl") };

const Q8_BLOCK_ELEMS: usize = 32;
const Q8_BLOCK_BYTES: usize = 34;
/// Must match @workgroup_size in the rope / gather / sequence-append / scatter-row
/// shaders. Stays at 64. Widening to 256 was tried with the pointwise ops (which it
/// helped a lot) and measured a REGRESSION here: 12.62 -> 13.24 ms/token. These move
/// real bytes per dispatch, so they want workgroups, not wider ones.
const WG_1D: u32 = 64;

/// Matches dequant.wgsl `Params`; the q8_row entry reuses the fields as
/// {src word offset, dst element offset, pair count}.
const Q8RowParams = extern struct { n: u32 = 0, k: u32 = 0, src_wpr: u32, dst_row: u32, count: u32, _p0: u32 = 0, _p1: u32 = 0, _p2: u32 = 0 };
/// Matches rope.wgsl `Params`.
const RopeParams = extern struct { count: u32, th: u32, tn: u32, pairs_total: u32, rope_pairs: u32, freq_step: f32, scale_factor: f32, _pad: u32 = 0 };
/// Matches gather.wgsl `Params`, shared by every entry point there.
/// `wpr` = u32 words per q8_0 table row (unused by the copying gathers).
/// `total` = work items: output words (copying gathers), block pairs (q8
/// gather), or words per row (scatter). See each entry point for the mapping.
const GatherParams = extern struct {
    rows: u32,
    d: u32,
    v: u32,
    wpr: u32 = 0,
    total: u32,
    _p0: u32 = 0,
    _p1: u32 = 0,
    _p2: u32 = 0,
    _p3: u32 = 0,
    _pad: [3]u32 = .{ 0, 0, 0 },
};

fn groups1D(n: u32) u32 {
    return @max(1, @min(context.ceilDiv(n, WG_1D), context.MAX_GROUPS_1D));
}

// ---- GatherRows --------------------------------------------------------------

/// Whether the gather lowers to device dispatches that read the index on-device
/// (one per table tile). The table may be split across bindings — each tile
/// covers a row range and the kernel skips indices outside it — but every tile
/// must hold whole rows, since a row is the unit a work item addresses.

pub fn gatherRowsOnDevice(out_meta: TensorMeta, table_meta: TensorMeta, idx_meta: TensorMeta) bool {
    if (out_meta.rank != 3 or table_meta.rank != 2 or idx_meta.rank != 2) return false;
    if (idx_meta.dtype != .i32 or context.totalTiles(idx_meta) != 1) return false;
    const d_total = table_meta.shape[1];
    // Whole rows per tile on both sides: a row is the unit a work item addresses,
    // and an output tile's rows must be contiguous in the index buffer.
    if (table_meta.tile_shape[1] != d_total) return false;
    if (out_meta.tile_shape[2] != d_total) return false;
    // An output tile's rows must be contiguous in the index buffer. That holds
    // when the tile spans whole L (any batch count), and also when it holds a
    // single batch — then it is a contiguous L range. Only a tile that is
    // partial in BOTH axes would interleave.
    if (out_meta.tile_shape[1] != out_meta.shape[1] and out_meta.tile_shape[0] != 1) return false;

    // The q8_0 gather dequantizes, so it interprets bits and emits f32. Every
    // other dtype is a pure word copy and needs only whole 4-byte words per row.
    if (table_meta.dtype == .q8_0) return out_meta.dtype == .f32 and d_total % 64 == 0;
    if (out_meta.dtype != table_meta.dtype) return false;
    return rowWords(out_meta.dtype, d_total) != null;
}

/// 4-byte words spanned by `elems` of `dtype`, or null when a row is not a whole
/// number of words (an odd f16 row) or the dtype is quantized.
fn rowWords(dtype: types.DType, elems: usize) ?usize {
    const info = dtype.info();
    if (info.is_quantized) return null;
    const bytes = elems * info.block_bytes;
    return if (bytes % 4 == 0) bytes / 4 else null;
}

/// Whether the row scatter lowers to the device dispatch that reads its index
/// on-device (4-byte scalars, single tiles). See `gatherRowsOnDevice`.
pub fn scatterRowOnDevice(buf_meta: TensorMeta, idx_meta: TensorMeta, src_meta: TensorMeta) bool {
    if (buf_meta.rank == 0 or idx_meta.dtype != .i32) return false;
    if (src_meta.dtype != buf_meta.dtype) return false;
    // A pure word copy: any dtype whose row is a whole number of 4-byte words.
    var row_elems: usize = 1;
    for (buf_meta.shape[1..]) |d| row_elems *= d;
    if (rowWords(buf_meta.dtype, row_elems) == null) return false;
    // `buf` may be split across bindings, but only along the row axis, so a row
    // is wholly inside one tile. `src` is that row and `idx` a scalar.
    for (buf_meta.tile_counts[1..]) |n| {
        if (n != 1) return false;
    }
    return context.totalTiles(idx_meta) == 1 and context.totalTiles(src_meta) == 1;
}

/// out[b, l, :] = table[indices[b, l], :], resolved entirely on the device.
pub fn execGatherRows(ctx: Ctx, frame: *Frame, s: executable.StepGatherRowsTiled) ExecuteProgramError!void {
    const hs = ctx.store;
    const out_meta = hs.meta(s.out) catch return error.ExecutionFailed;
    const table_meta = hs.meta(s.table) catch return error.ExecutionFailed;
    const idx_meta = hs.meta(s.indices) catch return error.ExecutionFailed;

    if (out_meta.rank != 3 or table_meta.rank != 2 or idx_meta.rank != 2) return error.Unsupported;
    if (idx_meta.dtype != .i32) return error.Unsupported;
    if (context.totalTiles(idx_meta) != 1) return error.Unsupported; // v0 contract, same as CPU

    const table_is_quant = table_meta.dtype == .q8_0;
    const d_total = table_meta.shape[1];
    const out_elem_bytes = out_meta.dtype.info().block_bytes;

    const b_total = idx_meta.shape[0];
    const l_total = idx_meta.shape[1];
    const v_total = table_meta.shape[0];

    // Device path: a dispatch per (output tile, table tile) pair. Each table tile
    // covers a row range and skips indices outside it; each output tile is
    // written tile-locally while `idx` stays globally indexed. Operands that fit
    // one binding collapse to the single-dispatch case. Every index resolves ON
    // DEVICE — a host index read forces a sync when a GPU op produced it, which
    // is exactly the argmax -> gather edge of an autoregressive decode step.
    if (!gatherRowsOnDevice(out_meta, table_meta, idx_meta)) return error.Unsupported;
    {
        const di = ctx.store.acquireTileDeviceConstLinear(s.indices, 0) catch return error.ExecutionFailed;
        defer hs.releaseConst(di.token);
        const i_n = context.packedElemsSized(di.rank, di.shape_mem[0..2], di.strides_mem[0..2], @sizeOf(i32)) orelse return error.Unsupported;
        if (i_n < b_total * l_total) return error.Unsupported;

        const wpr = (d_total / 64) * 17; // u32 words per q8_0 table row
        // The copying kernel addresses words so it serves any non-quantized
        // dtype; the q8 kernel addresses elements because it dequantizes.
        const row_unit = if (table_is_quant) d_total else rowWords(out_meta.dtype, d_total) orelse return error.Unsupported;
        const built = try ctx.pipes.get(gather_kernel, if (table_is_quant) "gather_q8_rows_f32" else "gather_rows_words");
        const idx_buf = ctx.devmem.bufferFor(di.handle).?;
        const table_tiles = context.totalTiles(table_meta);
        const out_tiles = context.totalTiles(out_meta);

        var out_row: usize = 0;
        var o_i: usize = 0;
        while (o_i < out_tiles) : (o_i += 1) {
            const dout = ctx.store.acquireTileDeviceMutLinear(s.out, o_i) catch return error.ExecutionFailed;
            defer hs.releaseMut(dout.token);
            if (!context.storageBindingFits(ctx, dout.len)) return error.Unsupported;
            const o_n = context.packedElemsSized(dout.rank, dout.shape_mem[0..3], dout.strides_mem[0..3], out_elem_bytes) orelse return error.Unsupported;

            // Rows of one output tile must be contiguous in the index buffer, so
            // only the batch axis may be split.
            const tile_rows = dout.shape_mem[0] * dout.shape_mem[1];
            if (dout.shape_mem[2] != d_total) return error.Unsupported;
            if (o_n < tile_rows * d_total) return error.Unsupported;

            const work = if (table_is_quant) tile_rows * (d_total / 64) else tile_rows * row_unit;
            const total = std.math.cast(u32, work) orelse return error.Unsupported;
            if (total == 0) continue;
            const out_buf = ctx.devmem.bufferFor(dout.handle).?;

            var row_begin: usize = 0;
            var t_i: usize = 0;
            while (t_i < table_tiles) : (t_i += 1) {
                const dt = ctx.store.acquireTileDeviceConstLinear(s.table, t_i) catch return error.ExecutionFailed;
                defer hs.releaseConst(dt.token);
                if (!context.storageBindingFits(ctx, dt.len)) return error.Unsupported;
                const table_rows = dt.shape_mem[0];
                const row_end = row_begin + table_rows;

                if (table_is_quant) {
                    if (dt.len < table_rows * wpr * @sizeOf(u32)) return error.Unsupported;
                } else {
                    const t_n = context.packedElemsSized(dt.rank, dt.shape_mem[0..2], dt.strides_mem[0..2], out_elem_bytes) orelse return error.Unsupported;
                    if (t_n < table_rows * d_total) return error.Unsupported;
                }

                const params: GatherParams = .{
                    .rows = @intCast(tile_rows),
                    .d = @intCast(row_unit),
                    .v = @intCast(v_total),
                    .wpr = @intCast(wpr),
                    .total = total,
                    ._p0 = @intCast(row_begin),
                    ._p1 = @intCast(row_end),
                    ._p2 = @intCast(out_row),
                };
                const bufs = [_]c.WGPUBuffer{ ctx.devmem.bufferFor(dt.handle).?, idx_buf, out_buf };
                const sizes = [_]u64{ dt.len, di.len, dout.len };
                try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(total), 1, 1 });
                row_begin = row_end;
            }
            if (row_begin < v_total) return error.Unsupported; // tiles did not cover the table
            out_row += tile_rows;
        }
        if (out_row != b_total * l_total) return error.Unsupported; // tiles did not cover the output
        return;
    }

    // No host-index fallback: a layout without a device kernel is a real gap in
    // GPU coverage and must surface as one, not hide behind a CPU path that
    // silently costs a synchronization per step.
    return error.Unsupported;
}

/// Batched canonical gather:
///   out[b, l, :] = data[b, indices[b, l], :]
///
/// The compiler tiles batch as one and keeps the feature row contiguous. As
/// Whether the batched gather lowers to device dispatches that read the index
/// on-device: f32 operands in the axis-1/one-batch-dim form. `data` and the
/// output may be split across bindings along batch; `idx` may not, since it is
/// indexed globally.
pub fn gatherOnDevice(out_meta: TensorMeta, data_meta: TensorMeta, idx_meta: TensorMeta, axis: usize, batch_dims: usize) bool {
    if (axis != 1 or batch_dims != 1) return false;
    if (out_meta.rank != 3 or data_meta.rank != 3 or idx_meta.rank != 2) return false;
    if (idx_meta.dtype != .i32 or out_meta.dtype != data_meta.dtype) return false;
    if (rowWords(out_meta.dtype, out_meta.shape[2]) == null) return false;
    // `idx` is globally indexed so it must be one binding; `data` and the output
    // may be split, but only along batch (checked per tile at record time).
    return context.totalTiles(idx_meta) == 1;
}

pub fn execGather(ctx: Ctx, frame: *Frame, s: executable.StepGatherTiled) ExecuteProgramError!void {
    if (s.axis != 1 or s.batch_dims != 1) return error.Unsupported;

    const hs = ctx.store;
    const out_meta = hs.meta(s.out) catch return error.ExecutionFailed;
    const data_meta = hs.meta(s.data) catch return error.ExecutionFailed;
    const idx_meta = hs.meta(s.indices) catch return error.ExecutionFailed;
    if (out_meta.rank != 3 or data_meta.rank != 3 or idx_meta.rank != 2) return error.Unsupported;
    if (idx_meta.dtype != .i32 or context.totalTiles(idx_meta) != 1) return error.Unsupported;
    if (out_meta.dtype != data_meta.dtype) return error.Unsupported;

    const elem_bytes: usize = switch (out_meta.dtype) {
        .f32 => 4,
        .f16 => 2,
        else => return error.Unsupported,
    };
    const batch = data_meta.shape[0];
    const sequence = data_meta.shape[1];
    const width = data_meta.shape[2];
    const gathered = idx_meta.shape[1];
    if (idx_meta.shape[0] != batch or out_meta.shape[0] != batch or
        out_meta.shape[1] != gathered or out_meta.shape[2] != width)
    {
        return error.Unsupported;
    }
    const row_bytes = width * elem_bytes;
    if (row_bytes % 4 != 0 or data_meta.tile_counts[2] != 1 or out_meta.tile_counts[2] != 1) return error.Unsupported;

    // Device path: one dispatch per output tile, one work item per output
    // element, index read on-device. `data` and `idx` stay globally indexed
    // while `o` is written tile-locally.
    if (!gatherOnDevice(out_meta, data_meta, idx_meta, s.axis, s.batch_dims)) return error.Unsupported;
    {
        const di = ctx.store.acquireTileDeviceConstLinear(s.indices, 0) catch return error.ExecutionFailed;
        defer hs.releaseConst(di.token);
        const i_n = context.packedElemsSized(di.rank, di.shape_mem[0..2], di.strides_mem[0..2], @sizeOf(i32)) orelse return error.Unsupported;
        if (i_n < batch * gathered) return error.Unsupported;

        const row_unit = rowWords(out_meta.dtype, width) orelse return error.Unsupported;
        const built = try ctx.pipes.get(gather_kernel, "gather_batched_words");
        const idx_buf = ctx.devmem.bufferFor(di.handle).?;
        const data_tiles = context.totalTiles(data_meta);
        const out_tiles = context.totalTiles(out_meta);

        // Tile origins, walked in the store's linear tile order (row-major over
        // tile coordinates), so G cycles inside B.
        var base_b: usize = 0;
        var base_g: usize = 0;
        var o_i: usize = 0;
        while (o_i < out_tiles) : (o_i += 1) {
            const dout = ctx.store.acquireTileDeviceMutLinear(s.out, o_i) catch return error.ExecutionFailed;
            defer hs.releaseMut(dout.token);
            if (!context.storageBindingFits(ctx, dout.len)) return error.Unsupported;
            const o_n = context.packedElemsSized(dout.rank, dout.shape_mem[0..3], dout.strides_mem[0..3], elem_bytes) orelse return error.Unsupported;

            // The output may split along batch AND along the gathered axis; a
            // tile must still hold whole rows so `idx` indexing stays an offset.
            const tile_batch = dout.shape_mem[0];
            const tile_g = dout.shape_mem[1];
            if (dout.shape_mem[2] != width) return error.Unsupported;
            if (o_n < tile_batch * tile_g * width) return error.Unsupported;

            const total = std.math.cast(u32, tile_batch * tile_g * row_unit) orelse return error.Unsupported;
            if (total == 0) continue;
            const out_buf = ctx.devmem.bufferFor(dout.handle).?;

            var data_b: usize = 0;
            var d_i: usize = 0;
            while (d_i < data_tiles) : (d_i += 1) {
                const dd = ctx.store.acquireTileDeviceConstLinear(s.data, d_i) catch return error.ExecutionFailed;
                defer hs.releaseConst(dd.token);
                if (!context.storageBindingFits(ctx, dd.len)) return error.Unsupported;
                const d_n = context.packedElemsSized(dd.rank, dd.shape_mem[0..3], dd.strides_mem[0..3], elem_bytes) orelse return error.Unsupported;
                const tile_data_batch = dd.shape_mem[0];
                if (dd.shape_mem[1] != sequence or dd.shape_mem[2] != width) return error.Unsupported;
                if (d_n < tile_data_batch * sequence * width) return error.Unsupported;

                const params: GatherParams = .{
                    .rows = @intCast(tile_batch),
                    .d = @intCast(row_unit),
                    .v = @intCast(sequence),
                    .wpr = @intCast(gathered),
                    .total = total,
                    ._p0 = @intCast(data_b),
                    ._p1 = @intCast(data_b + tile_data_batch),
                    ._p2 = @intCast(base_b),
                    ._p3 = @intCast(base_g),
                };
                const bufs = [_]c.WGPUBuffer{ ctx.devmem.bufferFor(dd.handle).?, idx_buf, out_buf };
                const sizes = [_]u64{ dd.len, di.len, dout.len };
                try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(total), 1, 1 });
                data_b += tile_data_batch;
            }
            if (data_b != batch) return error.Unsupported; // tiles did not cover the data
            base_g += tile_g;
            if (base_g == gathered) {
                base_g = 0;
                base_b += tile_batch;
            }
        }
        if (base_b != batch or base_g != 0) return error.Unsupported; // tiles did not cover the output
        return;
    }

    // No host-index fallback: a layout without a device kernel is a real gap in
    // GPU coverage and must surface as one, not hide behind a CPU path that
    // silently costs a synchronization per step.
    return error.Unsupported;
}

// ---- RoPE1D ------------------------------------------------------------------

/// out = rope(x, positions) over packed [B, L, N, H] tiles (f32, full head dim
/// per tile — same contract as the CPU exec). One dispatch per x tile; positions
/// are read on-device.
pub fn execRoPE(ctx: Ctx, frame: *Frame, s: executable.StepRoPE1DTiled) ExecuteProgramError!void {
    const hs = ctx.store;
    const out_meta = hs.meta(s.out) catch return error.ExecutionFailed;
    const x_meta = hs.meta(s.x) catch return error.ExecutionFailed;
    const pos_meta = hs.meta(s.positions) catch return error.ExecutionFailed;

    if (out_meta.rank != 4 or x_meta.rank != 4 or pos_meta.rank != 2) return error.Unsupported;
    if (out_meta.dtype != .f32 or x_meta.dtype != .f32) return error.Unsupported; // f16 later
    if (pos_meta.dtype != .i32) return error.Unsupported;
    if (out_meta.tile_counts[3] != 1) return error.Unsupported;

    const head_dim = out_meta.shape[3];
    const pairs_total = head_dim / 2;
    const rope_pairs_f = @floor(s.rope_proportion * @as(f32, @floatFromInt(pairs_total)));
    const rope_pairs: usize = @min(pairs_total, @as(usize, @intFromFloat(@max(rope_pairs_f, 0))));
    const freq_step: f32 = @floatCast(std.math.pow(f64, @as(f64, s.base_frequency), -2.0 / @as(f64, @floatFromInt(head_dim))));

    const built = try ctx.pipes.get(rope_kernel, "rope_f32");

    const total = context.totalTiles(out_meta);
    var ti: usize = 0;
    while (ti < total) : (ti += 1) {
        var coords: [tensor_store_mod.INLINE_RANK]usize = @splat(0);
        tensor_store_mod.decodeTileCoords(out_meta, ti, coords[0..4]) catch return error.ExecutionFailed;

        const dx = ctx.store.acquireTileDeviceConstLinear(s.x, ti) catch return error.ExecutionFailed;
        const dout = ctx.store.acquireTileDeviceMutLinear(s.out, ti) catch return error.ExecutionFailed;
        const pos_lin = tensor_store_mod.encodeTileIndex(pos_meta, coords[0..2]) catch return error.ExecutionFailed;
        const dpos = ctx.store.acquireTileDeviceConstLinear(s.positions, pos_lin) catch return error.ExecutionFailed;
        defer {
            hs.releaseConst(dx.token);
            hs.releaseMut(dout.token);
            hs.releaseConst(dpos.token);
        }
        if (!context.storageBindingFits(ctx, dx.len) or !context.storageBindingFits(ctx, dout.len)) return error.Unsupported;

        const count = context.packedElems(dout.rank, dout.shape_mem[0..4], dout.strides_mem[0..4]) orelse return error.Unsupported;
        const x_count = context.packedElems(dx.rank, dx.shape_mem[0..4], dx.strides_mem[0..4]) orelse return error.Unsupported;
        if (context.packedElems(dpos.rank, dpos.shape_mem[0..2], dpos.strides_mem[0..2]) == null) return error.Unsupported;
        if (x_count != count) return error.Unsupported;
        if (dpos.shape_mem[0] != dout.shape_mem[0] or dpos.shape_mem[1] != dout.shape_mem[1]) return error.Unsupported;

        const params: RopeParams = .{
            .count = count,
            .th = @intCast(dout.shape_mem[3]),
            .tn = @intCast(dout.shape_mem[2]),
            .pairs_total = @intCast(pairs_total),
            .rope_pairs = @intCast(rope_pairs),
            .freq_step = freq_step,
            .scale_factor = s.scale_factor,
        };
        const bufs = [_]c.WGPUBuffer{
            ctx.devmem.bufferFor(dx.handle).?,
            ctx.devmem.bufferFor(dpos.handle).?,
            ctx.devmem.bufferFor(dout.handle).?,
        };
        const sizes = [_]u64{ dx.len, dpos.len, dout.len };
        try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(count), 1, 1 });
    }
}

// ---- SequenceAppend -----------------------------------------------------------

pub fn sequenceAppendOnDevice(cache: TensorMeta, new_kv: TensorMeta, end_index: TensorMeta) bool {
    if (cache.rank != 4 or new_kv.rank != 4 or end_index.rank != 1) return false;
    if (cache.dtype != new_kv.dtype or (cache.dtype != .f32 and cache.dtype != .f16)) return false;
    if (end_index.dtype != .i32) return false;
    // The cache may be split across bindings along TIME only: a key row must live
    // wholly in one tile. `new_kv` and the index stay single-binding.
    for ([_]usize{ 0, 2, 3 }) |axis| {
        if (cache.tile_counts[axis] != 1) return false;
    }
    if (context.totalTiles(new_kv) != 1 or context.totalTiles(end_index) != 1) return false;
    const elem_bytes: usize = if (cache.dtype == .f32) 4 else 2;
    return (cache.shape[3] * elem_bytes) % 4 == 0;
}

/// cache[b, end[b] + t, h, :] = new_kv[b, t, h, :], in place. Mirrors the CPU
/// row loop but records one device dispatch. Ring mapping happens in-kernel.
///
/// The host-index fallback below is also the growth path: it maps every
/// destination time before writing, so a cache whose capacity only the append
/// itself can discover grows here. Placement decides which path runs — it puts
/// the position on the host exactly when this one is needed.
pub fn execSequenceAppend(ctx: Ctx, frame: *Frame, s: executable.StepSequenceAppendTiled) ExecuteProgramError!void {
    const hs = ctx.store;
    const cache_meta = hs.meta(s.cache) catch return error.ExecutionFailed;
    const new_meta = hs.meta(s.new_kv) catch return error.ExecutionFailed;
    const end_meta = hs.meta(s.end_index) catch return error.ExecutionFailed;

    if (cache_meta.rank != 4 or new_meta.rank != 4 or end_meta.rank != 1) return error.Unsupported;
    if (cache_meta.dtype != new_meta.dtype) return error.Unsupported;
    const elem_bytes: usize = switch (cache_meta.dtype) {
        .f32 => 4,
        .f16 => 2,
        else => return error.Unsupported,
    };

    const batch = cache_meta.shape[0];
    const heads = cache_meta.shape[2];
    const head_dim = cache_meta.shape[3];
    const new_len = new_meta.shape[1];
    const row_bytes = head_dim * elem_bytes;
    if (row_bytes % 4 != 0) return error.Unsupported;
    if (end_meta.dtype != .i32 or end_meta.shape[0] < batch) return error.Unsupported;

    const policy = hs.sequenceCachePolicyInfo(s.cache);
    if (!ctx.control.isHostPlaced(s.end_index) and sequenceAppendOnDevice(cache_meta, new_meta, end_meta)) {
        const row_words = (head_dim * elem_bytes) / 4;
        const total_words = batch * new_len * heads * row_words;
        if (total_words == 0) return;
        const dnew = ctx.store.acquireTileDeviceConstLinear(s.new_kv, 0) catch return error.ExecutionFailed;
        defer hs.releaseConst(dnew.token);
        const dend = ctx.store.acquireTileDeviceConstLinear(s.end_index, 0) catch return error.ExecutionFailed;
        defer hs.releaseConst(dend.token);
        if (!context.storageBindingFits(ctx, dnew.len) or !context.storageBindingFits(ctx, dend.len)) return error.Unsupported;

        const built = try ctx.pipes.get(gather_kernel, "sequence_append_u32");
        const ring_window: usize = if (policy.kind == .ring) @min(policy.ring_window_tokens, cache_meta.shape[1]) else 0;

        // One dispatch per cache tile: an appended row lands in exactly one, and
        // the rest no-op on it.
        var t_begin: usize = 0;
        var c_i: usize = 0;
        const cache_tiles = context.totalTiles(cache_meta);
        while (c_i < cache_tiles) : (c_i += 1) {
            const dcache = ctx.store.acquireTileDeviceMutLinear(s.cache, c_i) catch return error.ExecutionFailed;
            defer hs.releaseMut(dcache.token);
            if (!context.storageBindingFits(ctx, dcache.len)) return error.Unsupported;
            const tile_times = dcache.shape_mem[1];

            const params: GatherParams = .{
                .rows = @intCast(batch),
                .d = @intCast(tile_times),
                .v = @intCast(new_len),
                .wpr = @intCast(heads),
                .total = @intCast(row_words),
                ._p0 = @intCast(ring_window),
                ._p1 = @intCast(total_words),
                ._p2 = @intCast(t_begin),
            };
            const bufs = [_]c.WGPUBuffer{
                ctx.devmem.bufferFor(dnew.handle).?,
                ctx.devmem.bufferFor(dend.handle).?,
                ctx.devmem.bufferFor(dcache.handle).?,
            };
            const sizes = [_]u64{ dnew.len, dend.len, dcache.len };
            try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(@intCast(total_words)), 1, 1 });
            t_begin += tile_times;
        }
        if (t_begin != cache_meta.shape[1]) return error.Unsupported; // tiles did not cover the cache
        return;
    }

    // No host-index fallback: a layout without a device kernel is a real gap in
    // GPU coverage and must surface as one, not hide behind a CPU path that
    // silently costs a synchronization per step.
    return error.Unsupported;
}

// ---- ScatterRow ----------------------------------------------------------------

/// In-place row write buf[idx] = src (decode token emission). v1 mirrors the CPU
/// exec: buf/idx/src are each a single tile. Two paths: a device-side scatter
/// that reads the destination index ON DEVICE (4-byte scalars) so a GPU-computed
/// emit index forces no host round-trip, and a record-time host-index fallback
/// (f16/i8 or non-packed layouts) that submits pending work before its host read.
pub fn execScatterRow(ctx: Ctx, frame: *Frame, s: executable.StepScatterRow) ExecuteProgramError!void {
    const hs = ctx.store;
    const buf_meta = hs.meta(s.buf) catch return error.ExecutionFailed;
    const idx_meta = hs.meta(s.idx) catch return error.ExecutionFailed;
    const src_meta = hs.meta(s.src) catch return error.ExecutionFailed;

    if (buf_meta.rank == 0 or idx_meta.dtype != .i32) return error.Unsupported;
    if (src_meta.dtype != buf_meta.dtype) return error.Unsupported;
    const elem_bytes: usize = switch (buf_meta.dtype) {
        .f32, .i32 => 4,
        .f16 => 2,
        .i8 => 1,
        else => return error.Unsupported,
    };
    const m = buf_meta.shape[0];
    var row_size: usize = 1;
    var d: usize = 1;
    while (d < @as(usize, buf_meta.rank)) : (d += 1) row_size *= buf_meta.shape[d];
    const row_bytes = row_size * elem_bytes;
    if (row_bytes % 4 != 0) return error.Unsupported; // device copy granularity

    const dsrc = ctx.store.acquireTileDeviceConstLinear(s.src, 0) catch return error.ExecutionFailed;
    defer hs.releaseConst(dsrc.token);

    const src_rank: usize = @as(usize, dsrc.rank);
    const src_elems = context.packedElemsSized(dsrc.rank, dsrc.shape_mem[0..src_rank], dsrc.strides_mem[0..src_rank], elem_bytes) orelse return error.Unsupported;
    if (src_elems < row_size) return error.ExecutionFailed;

    // Device path: a dispatch per buf tile, each covering a row range and
    // skipping an index outside it. A buf that fits one binding is the
    // single-dispatch case; the index is read on-device either way.
    if (!scatterRowOnDevice(buf_meta, idx_meta, src_meta)) return error.Unsupported;
    {
        const di = ctx.store.acquireTileDeviceConstLinear(s.idx, 0) catch return error.ExecutionFailed;
        defer hs.releaseConst(di.token);
        if (!context.storageBindingFits(ctx, dsrc.len)) return error.Unsupported;
        const row_words = std.math.cast(u32, row_bytes / 4) orelse return error.Unsupported;
        const built = try ctx.pipes.get(gather_kernel, "scatter_row_u32");
        const src_buf = ctx.devmem.bufferFor(dsrc.handle).?;

        var row_begin: usize = 0;
        var b_i: usize = 0;
        const buf_tiles = context.totalTiles(buf_meta);
        while (b_i < buf_tiles) : (b_i += 1) {
            const dtile = ctx.store.acquireTileDeviceMutLinear(s.buf, b_i) catch return error.ExecutionFailed;
            defer hs.releaseMut(dtile.token);
            if (!context.storageBindingFits(ctx, dtile.len)) return error.Unsupported;
            const tile_rows = dtile.shape_mem[0];
            if (tile_rows * row_bytes > dtile.len) return error.Unsupported;

            const params: GatherParams = .{
                .rows = 1,
                .d = row_words,
                .v = @intCast(m),
                .total = row_words,
                ._p0 = @intCast(row_begin),
                ._p1 = @intCast(row_begin + tile_rows),
            };
            const bufs = [_]c.WGPUBuffer{ src_buf, ctx.devmem.bufferFor(di.handle).?, ctx.devmem.bufferFor(dtile.handle).? };
            const sizes = [_]u64{ dsrc.len, di.len, dtile.len };
            try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(row_words), 1, 1 });
            row_begin += tile_rows;
        }
        if (row_begin != m) return error.Unsupported; // tiles did not cover the buffer
        return;
    }

    // No host-index fallback: a layout without a device kernel is a real gap in
    // GPU coverage and must surface as one, not hide behind a CPU path that
    // silently costs a synchronization per step.
    return error.Unsupported;
}

/// Byte offset of row (b, h, t, 0) inside its tile, from the tile's memory
/// strides. Null on negative strides (unsupported).
fn tileRowOffset(tile: device_store.TileRef, meta: tensor_store_mod.TensorMeta, b: usize, h: usize, t: usize, coords: [4]usize) ?usize {
    const lb = b - coords[0] * meta.tile_shape[0];
    const lh = h - coords[1] * meta.tile_shape[1];
    const lt = t - coords[2] * meta.tile_shape[2];
    var off: usize = 0;
    const locals = [3]usize{ lb, lh, lt };
    for (locals, 0..) |l, d| {
        const stride = tile.strides_mem[d];
        if (stride < 0) return null;
        off += l * @as(usize, @intCast(stride));
    }
    return off;
}
