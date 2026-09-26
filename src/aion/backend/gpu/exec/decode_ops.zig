// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! The decode-loop data-movement ops: GatherRows (embedding lookup), RoPE1D,
//! and SequenceAppend.
//!
//! Every index resolves ON DEVICE. There is no host-index fallback: a layout
//! without a device form returns `error.Unsupported` so the gap in GPU coverage
//! is visible, rather than hidden behind a CPU path costing a synchronization
//! per step. A weight past the binding limit (an embedding table, a row buffer) is
//! split along dim 0 into chunks of whole rows; those ops dispatch per chunk, each
//! dispatch carrying the chunk's row range (see `kernels/gather.wgsl`).
//!
//! RoPE reads positions on-device (no host round-trip): one element-wise
//! dispatch, `kernels/rope.wgsl`.

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
/// `wpr` = u32 words per q8_0 table row, or rows per group for a grouped one
/// (unused by the copying gathers).
/// `total` = work items: output words (copying gathers), block pairs (q8
/// gather), or words per row (scatter). See each entry point for the mapping.
const GatherParams = extern struct {
    rows: u32,
    d: u32,
    v: u32,
    wpr: u32 = 0,
    total: u32,
    /// gather/scatter: the bound chunk's row range. sequence append: ring
    /// window and total words.
    p0: u32 = 0,
    p1: u32 = 0,
    _p2: u32 = 0,
    _p3: u32 = 0,
    _pad: [3]u32 = .{ 0, 0, 0 },
};

fn groups1D(n: u32) u32 {
    return @max(1, @min(context.ceilDiv(n, WG_1D), context.MAX_GROUPS_1D));
}

// ---- GatherRows --------------------------------------------------------------

/// Whether the gather lowers to device dispatches that read the index on-device
/// (one per table chunk). The table may be chunked — each chunk covers a row range
/// and the kernel skips indices outside it; the index and output are one buffer.
pub fn gatherRowsOnDevice(out_meta: TensorMeta, table_meta: TensorMeta, idx_meta: TensorMeta) bool {
    if (out_meta.rank != 3 or table_meta.rank != 2 or idx_meta.rank != 2) return false;
    if (idx_meta.dtype != .i32 or idx_meta.chunks != 1 or out_meta.chunks != 1) return false;
    const d_total = table_meta.shape[1];
    if (out_meta.shape[2] != d_total) return false;

    // The q8_0 gather dequantizes, so it interprets bits and emits f32. Every
    // other dtype is a pure word copy and needs only whole 4-byte words per row.
    // Row-major reads 64-element block pairs; a grouped order reads single blocks.
    if (table_meta.dtype == .q8_0) {
        const unit: usize = if (table_meta.block_order == .row_major) 64 else 32;
        return out_meta.dtype == .f32 and d_total % unit == 0;
    }
    if (out_meta.dtype != table_meta.dtype) return false;
    return rowAddressable(out_meta.dtype, d_total);
}

/// 4-byte words spanned by `elems` of `dtype`, or null when a row is not a whole
/// number of words (an odd f16 row) or the dtype is quantized.
fn rowWords(dtype: types.DType, elems: usize) ?usize {
    const info = dtype.info();
    if (info.is_quantized) return null;
    const bytes = elems * info.block_bytes;
    return if (bytes % 4 == 0) bytes / 4 else null;
}

/// Whether a row of `elems` can be addressed by SOME kernel in gather.wgsl.
///
/// The word kernels need whole 4-byte words per row. f16 additionally has
/// element-addressed twins (`shader-f16` is a required device feature), so an
/// f16 row is always addressable however odd its width — which is what removes
/// the old "odd f16 row" cliff from gather / scatter / sequence-append.
fn rowAddressable(dtype: types.DType, elems: usize) bool {
    if (dtype == .f16) return true;
    return rowWords(dtype, elems) != null;
}

/// True when this row must take the f16 element-addressed twin. An EVEN f16 row
/// keeps the word path: it moves 4 bytes per work item instead of 2, so it runs
/// half the invocations. Only the odd case has no word offset to use.
fn needsF16Elems(dtype: types.DType, elems: usize) bool {
    return dtype == .f16 and rowWords(dtype, elems) == null;
}

/// Whether the row scatter lowers to the device dispatch that reads its index
/// on-device. See `gatherRowsOnDevice`.
pub fn scatterRowOnDevice(buf_meta: TensorMeta, idx_meta: TensorMeta, src_meta: TensorMeta) bool {
    if (buf_meta.rank == 0 or idx_meta.dtype != .i32) return false;
    if (src_meta.dtype != buf_meta.dtype) return false;
    // A pure word copy: any dtype whose row is a whole number of 4-byte words.
    var row_elems: usize = 1;
    for (buf_meta.shape[1..]) |d| row_elems *= d;
    if (!rowAddressable(buf_meta.dtype, row_elems)) return false;
    // `buf` may be chunked (along rows, so a row is wholly inside one chunk).
    // `src` is that row and `idx` a scalar.
    return idx_meta.chunks == 1 and src_meta.chunks == 1;
}

/// out[b, l, :] = table[indices[b, l], :], resolved entirely on the device.
pub fn execGatherRows(ctx: Ctx, frame: *Frame, s: executable.StepGatherRows) ExecuteProgramError!void {
    const hs = ctx.store;
    const out_meta = hs.meta(s.out) catch return error.ExecutionFailed;
    const table_meta = hs.meta(s.table) catch return error.ExecutionFailed;
    const idx_meta = hs.meta(s.indices) catch return error.ExecutionFailed;

    if (out_meta.rank != 3 or table_meta.rank != 2 or idx_meta.rank != 2) return error.Unsupported;
    if (idx_meta.dtype != .i32) return error.Unsupported;

    const table_is_quant = table_meta.dtype == .q8_0;
    const grouped = table_is_quant and table_meta.block_order != .row_major;
    const d_total = table_meta.shape[1];
    const out_elem_bytes = out_meta.dtype.info().block_bytes;

    const b_total = idx_meta.shape[0];
    const l_total = idx_meta.shape[1];
    const v_total = table_meta.shape[0];

    // Device path: a dispatch per table chunk, each covering a row range and
    // skipping indices outside it. A table that fits one binding is the
    // single-dispatch case. Every index resolves ON DEVICE — a host index read forces a sync when a GPU op produced it, which
    // is exactly the argmax -> gather edge of an autoregressive decode step.
    if (!gatherRowsOnDevice(out_meta, table_meta, idx_meta)) return error.Unsupported;
    {
        const di = ctx.store.acquireConst(s.indices) catch return error.ExecutionFailed;
        defer hs.releaseConst(di.token);
        if (di.len < b_total * l_total * @sizeOf(i32)) return error.Unsupported;

        // u32 words per row-major q8_0 row, or the rows per group of a grouped one.
        const wpr = if (grouped) table_meta.block_order.groupRows() else (d_total / 64) * 17;
        // The copying kernel addresses words so it serves any non-quantized
        // dtype; the q8 kernel addresses elements because it dequantizes.
        // The q8 kernel addresses elements because it dequantizes; the copying
        // kernels address words, except an ODD f16 row which has no word offset
        // and takes the element-addressed twin.
        const f16_elems = needsF16Elems(out_meta.dtype, d_total);
        const row_unit = if (table_is_quant or f16_elems)
            d_total
        else
            rowWords(out_meta.dtype, d_total) orelse return error.Unsupported;
        const built = try ctx.pipes.get(gather_kernel, if (grouped)
            "gather_q8g_rows_f32"
        else if (table_is_quant)
            "gather_q8_rows_f32"
        else if (f16_elems)
            "gather_rows_f16"
        else
            "gather_rows_words");
        const idx_buf = ctx.devmem.bufferFor(di.handle).?;
        const table_chunks = table_meta.chunks;

        const dout = ctx.store.acquireMut(s.out) catch return error.ExecutionFailed;
        defer hs.releaseMut(dout.token);
        if (!context.storageBindingFits(ctx, dout.len)) return error.Unsupported;
        const rows = b_total * l_total;
        if (dout.len < rows * d_total * out_elem_bytes) return error.Unsupported;

        const work = if (grouped)
            rows * (d_total / 32)
        else if (table_is_quant)
            rows * (d_total / 64)
        else
            rows * row_unit;
        const total = std.math.cast(u32, work) orelse return error.Unsupported;
        if (total == 0) return;
        const out_buf = ctx.devmem.bufferFor(dout.handle).?;

        var row_begin: usize = 0;
        var t_i: usize = 0;
        while (t_i < table_chunks) : (t_i += 1) {
            const dt = ctx.store.acquireChunkConst(s.table, t_i) catch return error.ExecutionFailed;
            defer hs.releaseConst(dt.token);
            if (!context.storageBindingFits(ctx, dt.len)) return error.Unsupported;
            const table_rows = dt.rows;
            const row_end = row_begin + table_rows;

            const need: usize = if (grouped)
                table_rows * (d_total / 32) * Q8_BLOCK_BYTES
            else if (table_is_quant)
                table_rows * wpr * @sizeOf(u32)
            else
                table_rows * d_total * out_elem_bytes;
            if (dt.len < need) return error.Unsupported;

            const params: GatherParams = .{
                .rows = @intCast(rows),
                .d = @intCast(row_unit),
                .v = @intCast(v_total),
                .wpr = @intCast(wpr),
                .total = total,
                .p0 = @intCast(row_begin),
                .p1 = @intCast(row_end),
            };
            const bufs = [_]c.WGPUBuffer{ ctx.devmem.bufferFor(dt.handle).?, idx_buf, out_buf };
            const sizes = [_]u64{ dt.len, di.len, dout.len };
            try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(total), 1, 1 });
            row_begin = row_end;
        }
        if (row_begin != v_total) return error.Unsupported; // chunks did not cover the table
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
/// Whether the batched gather lowers to the device dispatch that reads the index
/// on-device: the axis-1/one-batch-dim form over whole buffers.
pub fn gatherOnDevice(out_meta: TensorMeta, data_meta: TensorMeta, idx_meta: TensorMeta, axis: usize, batch_dims: usize) bool {
    if (axis != 1 or batch_dims != 1) return false;
    if (out_meta.rank != 3 or data_meta.rank != 3 or idx_meta.rank != 2) return false;
    if (idx_meta.dtype != .i32 or out_meta.dtype != data_meta.dtype) return false;
    if (!rowAddressable(out_meta.dtype, out_meta.shape[2])) return false;
    return idx_meta.chunks == 1 and data_meta.chunks == 1 and out_meta.chunks == 1;
}

pub fn execGather(ctx: Ctx, frame: *Frame, s: executable.StepGather) ExecuteProgramError!void {
    if (s.axis != 1 or s.batch_dims != 1) return error.Unsupported;

    const hs = ctx.store;
    const out_meta = hs.meta(s.out) catch return error.ExecutionFailed;
    const data_meta = hs.meta(s.data) catch return error.ExecutionFailed;
    const idx_meta = hs.meta(s.indices) catch return error.ExecutionFailed;
    if (out_meta.rank != 3 or data_meta.rank != 3 or idx_meta.rank != 2) return error.Unsupported;
    if (idx_meta.dtype != .i32 or idx_meta.chunks != 1) return error.Unsupported;
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
    // An odd f16 row (a per-token scalar gather is `width == 1`) has no word
    // offset, so it takes the element-addressed twin instead of being rejected.
    const f16_elems = needsF16Elems(out_meta.dtype, width);
    if (!f16_elems and row_bytes % 4 != 0) return error.Unsupported;

    // Device path: one dispatch, one work item per output element, index read
    // on-device.
    if (!gatherOnDevice(out_meta, data_meta, idx_meta, s.axis, s.batch_dims)) return error.Unsupported;
    {
        const di = ctx.store.acquireConst(s.indices) catch return error.ExecutionFailed;
        defer hs.releaseConst(di.token);
        if (di.len < batch * gathered * @sizeOf(i32)) return error.Unsupported;

        const row_unit = if (f16_elems) width else rowWords(out_meta.dtype, width) orelse return error.Unsupported;
        const built = try ctx.pipes.get(gather_kernel, if (f16_elems) "gather_batched_f16" else "gather_batched_words");
        const idx_buf = ctx.devmem.bufferFor(di.handle).?;
        const dout = ctx.store.acquireMut(s.out) catch return error.ExecutionFailed;
        defer hs.releaseMut(dout.token);
        const dd = ctx.store.acquireConst(s.data) catch return error.ExecutionFailed;
        defer hs.releaseConst(dd.token);
        if (!context.storageBindingFits(ctx, dout.len) or !context.storageBindingFits(ctx, dd.len)) return error.Unsupported;
        if (dout.len < batch * gathered * row_bytes or dd.len < batch * sequence * row_bytes) return error.Unsupported;

        const total = std.math.cast(u32, batch * gathered * row_unit) orelse return error.Unsupported;
        if (total == 0) return;
        const params: GatherParams = .{
            .rows = @intCast(batch),
            .d = @intCast(row_unit),
            .v = @intCast(sequence),
            .wpr = @intCast(gathered),
            .total = total,
        };
        const bufs = [_]c.WGPUBuffer{ ctx.devmem.bufferFor(dd.handle).?, idx_buf, ctx.devmem.bufferFor(dout.handle).? };
        const sizes = [_]u64{ dd.len, di.len, dout.len };
        try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(total), 1, 1 });
        return;
    }

    // No host-index fallback: a layout without a device kernel is a real gap in
    // GPU coverage and must surface as one, not hide behind a CPU path that
    // silently costs a synchronization per step.
    return error.Unsupported;
}

// ---- RoPE1D ------------------------------------------------------------------

/// out = rope(x, positions) over packed [B, L, N, H] (same contract as the CPU
/// exec). One dispatch; positions are read on-device.
pub fn execRoPE(ctx: Ctx, frame: *Frame, s: executable.StepRoPE1D) ExecuteProgramError!void {
    const hs = ctx.store;
    const out_meta = hs.meta(s.out) catch return error.ExecutionFailed;
    const x_meta = hs.meta(s.x) catch return error.ExecutionFailed;
    const pos_meta = hs.meta(s.positions) catch return error.ExecutionFailed;

    if (out_meta.rank != 4 or x_meta.rank != 4 or pos_meta.rank != 2) return error.Unsupported;
    // f32 and f16, the latter natively addressed (`shader-f16` is a required
    // device feature). The angle, the sincos and the rotation are f32 in both,
    // matching the CPU kernel; only the load widens and the store rounds.
    const elem_bytes: usize = switch (out_meta.dtype) {
        .f32 => 4,
        .f16 => 2,
        else => return error.Unsupported,
    };
    if (x_meta.dtype != out_meta.dtype) return error.Unsupported;
    if (pos_meta.dtype != .i32) return error.Unsupported;
    inline for (.{ out_meta, x_meta, pos_meta }) |m| if (m.chunks != 1) return error.Unsupported;
    if (!std.mem.eql(usize, x_meta.shape, out_meta.shape)) return error.Unsupported;
    if (pos_meta.shape[0] != out_meta.shape[0] or pos_meta.shape[1] != out_meta.shape[1]) return error.Unsupported;

    const head_dim = out_meta.shape[3];
    const pairs_total = head_dim / 2;
    const rope_pairs_f = @floor(s.rope_proportion * @as(f32, @floatFromInt(pairs_total)));
    const rope_pairs: usize = @min(pairs_total, @as(usize, @intFromFloat(@max(rope_pairs_f, 0))));
    const freq_step: f32 = @floatCast(std.math.pow(f64, @as(f64, s.base_frequency), -2.0 / @as(f64, @floatFromInt(head_dim))));

    const built = try ctx.pipes.get(rope_kernel, if (out_meta.dtype == .f16) "rope_f16" else "rope_f32");

    const count_usize = out_meta.shape[0] * out_meta.shape[1] * out_meta.shape[2] * head_dim;
    const count = std.math.cast(u32, count_usize) orelse return error.Unsupported;
    if (count == 0) return;

    const dx = ctx.store.acquireConst(s.x) catch return error.ExecutionFailed;
    defer hs.releaseConst(dx.token);
    const dout = ctx.store.acquireMut(s.out) catch return error.ExecutionFailed;
    defer hs.releaseMut(dout.token);
    const dpos = ctx.store.acquireConst(s.positions) catch return error.ExecutionFailed;
    defer hs.releaseConst(dpos.token);
    if (!context.storageBindingFits(ctx, dx.len) or !context.storageBindingFits(ctx, dout.len)) return error.Unsupported;
    if (dout.len < count_usize * elem_bytes) return error.Unsupported;

    const params: RopeParams = .{
        .count = count,
        .th = @intCast(head_dim),
        .tn = @intCast(out_meta.shape[2]),
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

// ---- SequenceAppend -----------------------------------------------------------

pub fn sequenceAppendOnDevice(cache: TensorMeta, new_kv: TensorMeta, end_index: TensorMeta) bool {
    if (cache.rank != 4 or new_kv.rank != 4 or end_index.rank != 1) return false;
    if (cache.dtype != new_kv.dtype or (cache.dtype != .f32 and cache.dtype != .f16)) return false;
    if (end_index.dtype != .i32) return false;
    if (cache.chunks != 1 or new_kv.chunks != 1 or end_index.chunks != 1) return false;
    return rowAddressable(cache.dtype, cache.shape[3]);
}

/// cache[b, end[b] + t, h, :] = new_kv[b, t, h, :], in place. Mirrors the CPU
/// row loop but records one device dispatch. Ring mapping happens in-kernel.
///
/// The host-index fallback below is also the growth path: it maps every
/// destination time before writing, so a cache whose capacity only the append
/// itself can discover grows here. Placement decides which path runs — it puts
/// the position on the host exactly when this one is needed.
pub fn execSequenceAppend(ctx: Ctx, frame: *Frame, s: executable.StepSequenceAppend) ExecuteProgramError!void {
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
    // An odd f16 head dim has no word offset and takes the element-addressed
    // twin. This gate sits ahead of BOTH the device and the record-time path, so
    // before the twin existed an odd head dim had no way through at all.
    const f16_elems = needsF16Elems(cache_meta.dtype, head_dim);
    if (!f16_elems and row_bytes % 4 != 0) return error.Unsupported;
    if (end_meta.dtype != .i32 or end_meta.shape[0] < batch) return error.Unsupported;

    const policy = hs.sequenceCachePolicyInfo(s.cache);
    if (!ctx.control.isHostPlaced(s.end_index) and sequenceAppendOnDevice(cache_meta, new_meta, end_meta)) {
        const row_units = if (f16_elems) head_dim else (head_dim * elem_bytes) / 4;
        const total_words = batch * new_len * heads * row_units;
        if (total_words == 0) return;
        const dnew = ctx.store.acquireConst(s.new_kv) catch return error.ExecutionFailed;
        defer hs.releaseConst(dnew.token);
        const dend = ctx.store.acquireConst(s.end_index) catch return error.ExecutionFailed;
        defer hs.releaseConst(dend.token);
        if (!context.storageBindingFits(ctx, dnew.len) or !context.storageBindingFits(ctx, dend.len)) return error.Unsupported;

        const built = try ctx.pipes.get(gather_kernel, if (f16_elems) "sequence_append_f16" else "sequence_append_u32");
        const ring_modulus: usize = if (policy.kind == .rolling) cache_meta.shape[1] else 0;

        const dcache = ctx.store.acquireMut(s.cache) catch return error.ExecutionFailed;
        defer hs.releaseMut(dcache.token);
        if (!context.storageBindingFits(ctx, dcache.len)) return error.Unsupported;

        const params: GatherParams = .{
            .rows = @intCast(batch),
            .d = @intCast(cache_meta.shape[1]),
            .v = @intCast(new_len),
            .wpr = @intCast(heads),
            .total = @intCast(row_units),
            .p0 = @intCast(ring_modulus),
            .p1 = @intCast(total_words),
        };
        const bufs = [_]c.WGPUBuffer{
            ctx.devmem.bufferFor(dnew.handle).?,
            ctx.devmem.bufferFor(dend.handle).?,
            ctx.devmem.bufferFor(dcache.handle).?,
        };
        const sizes = [_]u64{ dnew.len, dend.len, dcache.len };
        try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(@intCast(total_words)), 1, 1 });
        return;
    }

    // No host-index fallback: a layout without a device kernel is a real gap in
    // GPU coverage and must surface as one, not hide behind a CPU path that
    // silently costs a synchronization per step.
    return error.Unsupported;
}

// ---- ScatterRow ----------------------------------------------------------------

/// In-place row write buf[idx] = src (decode token emission), reading the
/// destination index ON DEVICE so a GPU-computed emit index forces no host round
/// trip. `buf` may be chunked along rows: one dispatch per chunk.
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
    // An odd f16 row takes the element-addressed twin; every other dtype still
    // needs whole 4-byte words (device copy granularity).
    const f16_elems = needsF16Elems(buf_meta.dtype, row_size);
    if (!f16_elems and row_bytes % 4 != 0) return error.Unsupported;

    const dsrc = ctx.store.acquireConst(s.src) catch return error.ExecutionFailed;
    defer hs.releaseConst(dsrc.token);

    if (dsrc.len < row_bytes) return error.ExecutionFailed;

    // Device path: a dispatch per buf chunk, each covering a row range and
    // skipping an index outside it. A buf that fits one binding is the
    // single-dispatch case; the index is read on-device either way.
    if (!scatterRowOnDevice(buf_meta, idx_meta, src_meta)) return error.Unsupported;
    {
        const di = ctx.store.acquireConst(s.idx) catch return error.ExecutionFailed;
        defer hs.releaseConst(di.token);
        if (!context.storageBindingFits(ctx, dsrc.len)) return error.Unsupported;
        const row_units = std.math.cast(u32, if (f16_elems) row_size else row_bytes / 4) orelse return error.Unsupported;
        const built = try ctx.pipes.get(gather_kernel, if (f16_elems) "scatter_row_f16" else "scatter_row_u32");
        const src_buf = ctx.devmem.bufferFor(dsrc.handle).?;

        var row_begin: usize = 0;
        var b_i: usize = 0;
        const buf_chunks = buf_meta.chunks;
        while (b_i < buf_chunks) : (b_i += 1) {
            const dchunk = ctx.store.acquireChunkMut(s.buf, b_i) catch return error.ExecutionFailed;
            defer hs.releaseMut(dchunk.token);
            if (!context.storageBindingFits(ctx, dchunk.len)) return error.Unsupported;
            const chunk_rows = dchunk.rows;
            if (chunk_rows * row_bytes > dchunk.len) return error.Unsupported;

            const params: GatherParams = .{
                .rows = 1,
                .d = row_units,
                .v = @intCast(m),
                .total = row_units,
                .p0 = @intCast(row_begin),
                .p1 = @intCast(row_begin + chunk_rows),
            };
            const bufs = [_]c.WGPUBuffer{ src_buf, ctx.devmem.bufferFor(di.handle).?, ctx.devmem.bufferFor(dchunk.handle).? };
            const sizes = [_]u64{ dsrc.len, di.len, dchunk.len };
            try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(row_units), 1, 1 });
            row_begin += chunk_rows;
        }
        if (row_begin != m) return error.Unsupported; // chunks did not cover the buffer
        return;
    }

    // No host-index fallback: a layout without a device kernel is a real gap in
    // GPU coverage and must surface as one, not hide behind a CPU path that
    // silently costs a synchronization per step.
    return error.Unsupported;
}

/// General gather over whole buffers — the axis/batch_dims/rank combinations
/// the specialized row and batched-row paths decline.
///
/// Out-of-range indices clamp here; the CPU raises. A shader cannot raise without
/// a device fault channel, so the two agree on every valid index (including ONNX
/// negative indexing) and differ only for a program that is already invalid.
pub fn execGatherND(ctx: Ctx, frame: *Frame, s: executable.StepGatherND) ExecuteProgramError!void {
    const hs = ctx.store;
    const out_meta = hs.meta(s.out) catch return error.ExecutionFailed;
    const data_meta = hs.meta(s.data) catch return error.ExecutionFailed;
    const idx_meta = hs.meta(s.indices) catch return error.ExecutionFailed;
    if (idx_meta.dtype != .i32 or out_meta.dtype != data_meta.dtype) return error.Unsupported;
    if (out_meta.chunks != 1 or data_meta.chunks != 1 or idx_meta.chunks != 1) return error.Unsupported;

    const axis: usize = s.axis;
    const bd: usize = s.batch_dims;
    const dr: usize = data_meta.rank;
    const ir: usize = idx_meta.rank;
    if (axis >= dr or bd > axis or bd > ir) return error.Unsupported;

    var batch: usize = 1;
    for (data_meta.shape[0..bd]) |d| batch *= d;
    var lead: usize = 1;
    for (data_meta.shape[0..axis]) |d| lead *= d;
    var inner: usize = 1;
    for (data_meta.shape[axis + 1 .. dr]) |d| inner *= d;
    var picked: usize = 1;
    for (idx_meta.shape[bd..ir]) |d| picked *= d;
    const axis_len: usize = data_meta.shape[axis];
    if (batch == 0 or lead == 0 or axis_len == 0) return error.Unsupported;

    const f16_mode = out_meta.dtype == .f16;
    if (!f16_mode and out_meta.dtype != .f32) return error.Unsupported;

    const total: usize = lead * picked * inner;
    var p: extern struct { lead: u32, picked: u32, inner: u32, axis_len: u32, mid: u32, total: u32, pad0: u32 = 0, pad1: u32 = 0 } = .{
        .lead = std.math.cast(u32, lead) orelse return error.Unsupported,
        .picked = std.math.cast(u32, picked) orelse return error.Unsupported,
        .inner = std.math.cast(u32, inner) orelse return error.Unsupported,
        .axis_len = std.math.cast(u32, axis_len) orelse return error.Unsupported,
        .mid = std.math.cast(u32, lead / batch) orelse return error.Unsupported,
        .total = std.math.cast(u32, total) orelse return error.Unsupported,
    };

    const ds = ctx.store.acquireConst(s.data) catch return error.ExecutionFailed;
    defer hs.releaseConst(ds.token);
    const di = ctx.store.acquireConst(s.indices) catch return error.ExecutionFailed;
    defer hs.releaseConst(di.token);
    const dout = ctx.store.acquireMut(s.out) catch return error.ExecutionFailed;
    defer hs.releaseMut(dout.token);

    const built = try ctx.pipes.get(gather_kernel, if (f16_mode) "gather_nd_f16" else "gather_nd_words");
    const bufs = [_]c.WGPUBuffer{
        ctx.devmem.bufferFor(ds.handle).?,
        ctx.devmem.bufferFor(di.handle).?,
        ctx.devmem.bufferFor(dout.handle).?,
    };
    const sizes = [_]u64{ ds.len, di.len, dout.len };
    try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&p), .{
        @max(1, @min(context.ceilDiv(p.total, 64), context.MAX_GROUPS_1D)), 1, 1,
    });
}
