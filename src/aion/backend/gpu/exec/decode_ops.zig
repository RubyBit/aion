// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! The decode-loop data-movement ops: GatherRows (embedding lookup), RoPE1D,
//! and SequenceAppend.
//!
//! Gather and KV-append are RECORD-TIME resolved: their indices (token ids /
//! end offsets) are read from the HOST store while encoding the frame, and each
//! row becomes either a `CopyBufferToBuffer` command (scalar rows) or a tiny
//! dequant dispatch (q8_0 rows). That sidesteps the bind-slot problem of
//! runtime-indexed multi-tile tables — the right tile is picked per row on the
//! CPU. The host read is coherent even if a future GPU op produces the indices
//! (the resident store flushes device-dirty tiles on host access) — it just
//! costs a sync, which is the price until the full decode loop lives on-device.
//!
//! RoPE reads positions on-device (no host round-trip): one element-wise
//! dispatch per x tile, `kernels/rope.wgsl`.

const std = @import("std");
const wgpu = @import("../wgpu.zig");
const pipelines = @import("../pipelines.zig");
const context = @import("../context.zig");
const backend_mod = @import("../../backend.zig");
const tensor_store_mod = @import("../../../runtime/tensor_store.zig");
const resident_mod = @import("../../../runtime/residency/resident_store.zig");
const executable = @import("../../../runtime/executable.zig");

const c = wgpu.c;
const Ctx = context.Ctx;
const Frame = @import("../frame.zig").Frame;
const ExecuteProgramError = backend_mod.ExecuteProgramError;
const KernelDesc = pipelines.KernelDesc;

const rope_kernel: KernelDesc = .{ .name = "rope", .wgsl = @embedFile("../kernels/rope.wgsl") };
const dequant_kernel: KernelDesc = .{ .name = "dequant", .wgsl = @embedFile("../kernels/dequant.wgsl") };
const gather_kernel: KernelDesc = .{ .name = "gather", .wgsl = @embedFile("../kernels/gather.wgsl") };

const Q8_BLOCK_ELEMS: usize = 32;
const Q8_BLOCK_BYTES: usize = 34;
const WG_1D: u32 = 64;

/// Matches dequant.wgsl `Params`; the q8_row entry reuses the fields as
/// {src word offset, dst element offset, pair count}.
const Q8RowParams = extern struct { n: u32 = 0, k: u32 = 0, src_wpr: u32, dst_row: u32, count: u32, _p0: u32 = 0, _p1: u32 = 0, _p2: u32 = 0 };
/// Matches rope.wgsl `Params`.
const RopeParams = extern struct { count: u32, th: u32, tn: u32, pairs_total: u32, rope_pairs: u32, freq_step: f32, scale_factor: f32, _pad: u32 = 0 };
/// Matches gather.wgsl `Params` (shared by all three gather.wgsl entry points).
/// `wpr` = u32 words per q8_0 table row (unused by the f32 gather / scatter).
/// `total` = work items: output elements (f32 gather), block pairs (q8 gather),
/// or words per row (scatter). See each entry point for the field mapping.
const GatherParams = extern struct { rows: u32, d: u32, v: u32, wpr: u32 = 0, total: u32 };

fn groups1D(n: u32) u32 {
    return @max(1, @min(context.ceilDiv(n, WG_1D), context.MAX_GROUPS_1D));
}

/// Read a host tensor's single tile as i32s (indices/end offsets). Coherent with
/// device state via the resident store's flush-on-host-read.
const HostI32 = struct {
    token: usize,
    vals: []align(1) const i32,
    hs: tensor_store_mod.TensorStore,

    fn acquire(hs: tensor_store_mod.TensorStore, id: tensor_store_mod.TensorId) ExecuteProgramError!HostI32 {
        const tile = hs.acquireTileConstLinear(id, 0) catch return error.ExecutionFailed;
        const ptr: [*]align(1) const i32 = @ptrCast(tile.bytes.ptr);
        return .{ .token = tile.token, .vals = ptr[0 .. tile.bytes.len / @sizeOf(i32)], .hs = hs };
    }
    fn release(self: HostI32) void {
        self.hs.releaseConst(self.token);
    }
};

// ---- GatherRows --------------------------------------------------------------

/// out[b, l, :] = table[indices[b, l], :]. Table rows resolve to (tile, offset)
/// at record time; scalar rows are buffer copies, q8_0 rows one dequant dispatch
/// each (decode is B=L=1 → a single command per gather).
pub fn execGatherRows(ctx: Ctx, frame: *Frame, s: executable.StepGatherRowsTiled) ExecuteProgramError!void {
    const hs = ctx.rstore.tensorStore();
    const out_meta = hs.meta(s.out) catch return error.ExecutionFailed;
    const table_meta = hs.meta(s.table) catch return error.ExecutionFailed;
    const idx_meta = hs.meta(s.indices) catch return error.ExecutionFailed;

    if (out_meta.rank != 3 or table_meta.rank != 2 or idx_meta.rank != 2) return error.Unsupported;
    if (idx_meta.dtype != .i32) return error.Unsupported;
    if (context.totalTiles(idx_meta) != 1) return error.Unsupported; // v0 contract, same as CPU

    const table_is_quant = table_meta.dtype == .q8_0;
    const elem_bytes: usize = switch (out_meta.dtype) {
        .f32 => 4,
        .f16 => 2,
        else => return error.Unsupported,
    };
    const d_total = table_meta.shape[1];
    if (table_is_quant) {
        if (out_meta.dtype != .f32) return error.Unsupported; // q8 dequant emits f32 (v1)
        if (d_total % 64 != 0) return error.Unsupported; // word-aligned block pairs
        if (table_meta.tile_shape[1] != d_total) return error.Unsupported;
    } else {
        if (out_meta.dtype != table_meta.dtype) return error.Unsupported;
        if ((d_total * elem_bytes) % 4 != 0) return error.Unsupported; // copy granularity
    }
    const table_row_bytes: usize = if (table_is_quant)
        (d_total / Q8_BLOCK_ELEMS) * Q8_BLOCK_BYTES
    else
        d_total * elem_bytes;

    const b_total = idx_meta.shape[0];
    const l_total = idx_meta.shape[1];
    const v_total = table_meta.shape[0];

    // Fast path: f32 single-tile table → ONE device-side gather dispatch with
    // the indices read on-device. No per-row encoder copies and, crucially, no
    // host read of the indices (which forces a sync when a GPU op produced
    // them — the argmax→gather edge of an on-device decode step).
    if (!table_is_quant and out_meta.dtype == .f32 and
        context.totalTiles(table_meta) == 1 and context.totalTiles(out_meta) == 1)
    {
        const dt = ctx.rstore.acquireTileDeviceConstLinear(s.table, 0) catch return error.ExecutionFailed;
        defer hs.releaseConst(dt.token);
        const di = ctx.rstore.acquireTileDeviceConstLinear(s.indices, 0) catch return error.ExecutionFailed;
        defer hs.releaseConst(di.token);
        const dout = ctx.rstore.acquireTileDeviceMutLinear(s.out, 0) catch return error.ExecutionFailed;
        defer hs.releaseMut(dout.token);
        if (context.storageBindingFits(ctx, dt.len) and context.storageBindingFits(ctx, dout.len)) fast: {
            const t_n = context.packedElems(dt.rank, dt.shape_mem[0..2], dt.strides_mem[0..2]) orelse break :fast;
            const o_n = context.packedElems(dout.rank, dout.shape_mem[0..3], dout.strides_mem[0..3]) orelse break :fast;
            const i_n = context.packedElemsSized(di.rank, di.shape_mem[0..2], di.strides_mem[0..2], @sizeOf(i32)) orelse break :fast;
            const rows = b_total * l_total;
            if (t_n < v_total * d_total or o_n < rows * d_total or i_n < rows) break :fast;

            const total = std.math.cast(u32, rows * d_total) orelse break :fast;
            const params: GatherParams = .{
                .rows = @intCast(rows),
                .d = @intCast(d_total),
                .v = @intCast(v_total),
                .total = total,
            };
            const built = try ctx.pipes.get(gather_kernel, "gather_rows_f32");
            const bufs = [_]c.WGPUBuffer{
                ctx.devmem.bufferFor(dt.handle).?,
                ctx.devmem.bufferFor(di.handle).?,
                ctx.devmem.bufferFor(dout.handle).?,
            };
            const sizes = [_]u64{ dt.len, di.len, dout.len };
            try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(total), 1, 1 });
            return;
        }
    }

    // Fast path: q8_0 single-tile table → ONE device-side dequant-gather dispatch
    // with the index read on-device. Same win as the f32 path for quantized
    // embedding tables (the autoregressive-decode common case): no per-row encoder
    // copies and no host index read (which forces a control-flow sync when a GPU
    // op produced the index). The checks above already guarantee d_total % 64 == 0
    // and table cols whole for a quant table.
    if (table_is_quant and out_meta.dtype == .f32 and
        context.totalTiles(table_meta) == 1 and context.totalTiles(out_meta) == 1)
    {
        const dt = ctx.rstore.acquireTileDeviceConstLinear(s.table, 0) catch return error.ExecutionFailed;
        defer hs.releaseConst(dt.token);
        const di = ctx.rstore.acquireTileDeviceConstLinear(s.indices, 0) catch return error.ExecutionFailed;
        defer hs.releaseConst(di.token);
        const dout = ctx.rstore.acquireTileDeviceMutLinear(s.out, 0) catch return error.ExecutionFailed;
        defer hs.releaseMut(dout.token);
        if (context.storageBindingFits(ctx, dt.len) and context.storageBindingFits(ctx, dout.len)) fast: {
            const o_n = context.packedElems(dout.rank, dout.shape_mem[0..3], dout.strides_mem[0..3]) orelse break :fast;
            const i_n = context.packedElemsSized(di.rank, di.shape_mem[0..2], di.strides_mem[0..2], @sizeOf(i32)) orelse break :fast;
            const rows = b_total * l_total;
            const wpr = (d_total / 64) * 17; // u32 words per q8_0 table row
            if (o_n < rows * d_total or i_n < rows) break :fast;
            if (dt.len < v_total * wpr * @sizeOf(u32)) break :fast;

            const total_pairs = std.math.cast(u32, rows * (d_total / 64)) orelse break :fast;
            const params: GatherParams = .{
                .rows = @intCast(rows),
                .d = @intCast(d_total),
                .v = @intCast(v_total),
                .wpr = @intCast(wpr),
                .total = total_pairs,
            };
            const built = try ctx.pipes.get(gather_kernel, "gather_q8_rows_f32");
            const bufs = [_]c.WGPUBuffer{
                ctx.devmem.bufferFor(dt.handle).?,
                ctx.devmem.bufferFor(di.handle).?,
                ctx.devmem.bufferFor(dout.handle).?,
            };
            const sizes = [_]u64{ dt.len, di.len, dout.len };
            try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(total_pairs), 1, 1 });
            return;
        }
    }

    // Host fallback (multi-tile / f16 / non-64-aligned q8 tables): the index is
    // read on the host at record time, so submit any producer still pending in
    // this frame first (no-op for host-resident indices). See
    // `Ctx.submitPendingIfDeviceDirty`.
    try ctx.submitPendingIfDeviceDirty(frame, s.indices);
    const idx = try HostI32.acquire(hs, s.indices);
    defer idx.release();
    if (idx.vals.len < b_total * l_total) return error.ExecutionFailed;

    const q8_built = if (table_is_quant) try ctx.pipes.get(dequant_kernel, "q8_row_to_f32") else undefined;

    // Cached table device tile (gathers cluster heavily within one tile).
    var cached_ti0: ?usize = null;
    var table_tile: resident_mod.TileRefDevice = undefined;
    defer if (cached_ti0 != null) hs.releaseConst(table_tile.token);

    const total = context.totalTiles(out_meta);
    var ti: usize = 0;
    while (ti < total) : (ti += 1) {
        var coords: [tensor_store_mod.INLINE_RANK]usize = @splat(0);
        tensor_store_mod.decodeTileCoords(out_meta, ti, coords[0..3]) catch return error.ExecutionFailed;
        const base_b = coords[0] * out_meta.tile_shape[0];
        const base_l = coords[1] * out_meta.tile_shape[1];

        const dout = ctx.rstore.acquireTileDeviceMutLinear(s.out, ti) catch return error.ExecutionFailed;
        defer hs.releaseMut(dout.token);
        if (!context.storageBindingFits(ctx, dout.len)) return error.Unsupported;
        const tb = dout.shape_mem[0];
        const tl = dout.shape_mem[1];
        const td = dout.shape_mem[2];
        if (td != d_total) return error.Unsupported;
        // dst offsets assume a packed [tb, tl, td] tile.
        if (context.packedElems(dout.rank, dout.shape_mem[0..3], dout.strides_mem[0..3]) == null) return error.Unsupported;
        const out_buf = ctx.devmem.bufferFor(dout.handle).?;

        var lb: usize = 0;
        while (lb < tb) : (lb += 1) {
            var ll: usize = 0;
            while (ll < tl) : (ll += 1) {
                const b = base_b + lb;
                const l = base_l + ll;
                if (b >= b_total or l >= l_total) return error.ExecutionFailed;
                const idx_i32 = idx.vals[b * l_total + l];
                if (idx_i32 < 0) return error.ExecutionFailed;
                const row: usize = @intCast(idx_i32);
                if (row >= v_total) return error.ExecutionFailed;

                const ti0 = row / table_meta.tile_shape[0];
                const local_r = row - ti0 * table_meta.tile_shape[0];
                if (cached_ti0 == null or cached_ti0.? != ti0) {
                    if (cached_ti0 != null) hs.releaseConst(table_tile.token);
                    cached_ti0 = null;
                    const lin = tensor_store_mod.encodeTileIndex(table_meta, &[_]usize{ ti0, 0 }) catch return error.ExecutionFailed;
                    table_tile = ctx.rstore.acquireTileDeviceConstLinear(s.table, lin) catch return error.ExecutionFailed;
                    cached_ti0 = ti0;
                    if (!context.storageBindingFits(ctx, table_tile.len)) return error.Unsupported;
                }
                const table_buf = ctx.devmem.bufferFor(table_tile.handle).?;

                const src_off = local_r * table_row_bytes;
                const dst_elem = (lb * tl + ll) * td;
                if (src_off + table_row_bytes > table_tile.len) return error.ExecutionFailed;

                if (table_is_quant) {
                    if (src_off % 4 != 0) return error.Unsupported;
                    const pairs: u32 = @intCast(td / 64);
                    const params: Q8RowParams = .{
                        .src_wpr = @intCast(src_off / 4),
                        .dst_row = @intCast(dst_elem),
                        .count = pairs,
                    };
                    const bufs = [_]c.WGPUBuffer{ table_buf, out_buf };
                    const sizes = [_]u64{ table_tile.len, dout.len };
                    try frame.recordCompute(q8_built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(pairs), 1, 1 });
                } else {
                    const dst_off = dst_elem * elem_bytes;
                    if (src_off % 4 != 0 or dst_off % 4 != 0) return error.Unsupported;
                    frame.recordCopy(table_buf, src_off, out_buf, dst_off, table_row_bytes);
                }
            }
        }
    }
}

// ---- RoPE1D ------------------------------------------------------------------

/// out = rope(x, positions) over packed [B, L, N, H] tiles (f32, full head dim
/// per tile — same contract as the CPU exec). One dispatch per x tile; positions
/// are read on-device.
pub fn execRoPE(ctx: Ctx, frame: *Frame, s: executable.StepRoPE1DTiled) ExecuteProgramError!void {
    const hs = ctx.rstore.tensorStore();
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

        const dx = ctx.rstore.acquireTileDeviceConstLinear(s.x, ti) catch return error.ExecutionFailed;
        const dout = ctx.rstore.acquireTileDeviceMutLinear(s.out, ti) catch return error.ExecutionFailed;
        const pos_lin = tensor_store_mod.encodeTileIndex(pos_meta, coords[0..2]) catch return error.ExecutionFailed;
        const dpos = ctx.rstore.acquireTileDeviceConstLinear(s.positions, pos_lin) catch return error.ExecutionFailed;
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

/// cache[b, h, end[b] + t, :] = new_kv[b, h, t, :], in place. Mirrors the CPU
/// row loop (incl. `mapSequenceStep` ring/growth mapping) but records one device
/// buffer copy per row. end_index is read on the host at record time.
pub fn execSequenceAppend(ctx: Ctx, frame: *Frame, s: executable.StepSequenceAppendTiled) ExecuteProgramError!void {
    const hs = ctx.rstore.tensorStore();
    var cache_meta = hs.meta(s.cache) catch return error.ExecutionFailed;
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
    const heads = cache_meta.shape[1];
    const head_dim = cache_meta.shape[3];
    const new_len = new_meta.shape[2];
    const row_bytes = head_dim * elem_bytes;
    if (row_bytes % 4 != 0) return error.Unsupported;
    if (end_meta.dtype != .i32 or end_meta.shape[0] < batch) return error.Unsupported;

    // end_index is read on the host at record time; submit any pending producer
    // first (no-op for host-resident offsets). See `Ctx.submitPendingIfDeviceDirty`.
    try ctx.submitPendingIfDeviceDirty(frame, s.end_index);
    const end_idx = try HostI32.acquire(hs, s.end_index);
    defer end_idx.release();
    if (end_idx.vals.len < batch) return error.ExecutionFailed;

    // Map all destination times first (may grow the physical cache), then
    // refresh the metadata — same protocol as the CPU exec.
    var b: usize = 0;
    while (b < batch) : (b += 1) {
        if (end_idx.vals[b] < 0) return error.ExecutionFailed;
        const t_start: usize = @intCast(end_idx.vals[b]);
        var t: usize = 0;
        while (t < new_len) : (t += 1) {
            _ = hs.mapSequenceStep(s.cache, t_start + t, cache_meta.shape[2]) catch return error.ExecutionFailed;
        }
    }
    cache_meta = hs.meta(s.cache) catch return error.ExecutionFailed;
    const cache_t = cache_meta.shape[2];
    if (new_len == 0) return;

    // Cached device tiles: decode touches exactly one src and one dst tile.
    var src_cached: ?usize = null;
    var src_tile: resident_mod.TileRefDevice = undefined;
    defer if (src_cached != null) hs.releaseConst(src_tile.token);
    var dst_cached: ?usize = null;
    var dst_tile: resident_mod.TileRefDevice = undefined;
    defer if (dst_cached != null) hs.releaseMut(dst_tile.token);

    b = 0;
    while (b < batch) : (b += 1) {
        const t_start: usize = @intCast(end_idx.vals[b]);
        var h: usize = 0;
        while (h < heads) : (h += 1) {
            var t: usize = 0;
            while (t < new_len) : (t += 1) {
                const dst_t = hs.mapSequenceStep(s.cache, t_start + t, cache_t) catch return error.ExecutionFailed;

                const src_coords = [4]usize{ b / new_meta.tile_shape[0], h / new_meta.tile_shape[1], t / new_meta.tile_shape[2], 0 };
                const dst_coords = [4]usize{ b / cache_meta.tile_shape[0], h / cache_meta.tile_shape[1], dst_t / cache_meta.tile_shape[2], 0 };
                const src_lin = tensor_store_mod.encodeTileIndex(new_meta, &src_coords) catch return error.ExecutionFailed;
                const dst_lin = tensor_store_mod.encodeTileIndex(cache_meta, &dst_coords) catch return error.ExecutionFailed;

                if (src_cached == null or src_cached.? != src_lin) {
                    if (src_cached != null) hs.releaseConst(src_tile.token);
                    src_cached = null;
                    src_tile = ctx.rstore.acquireTileDeviceConstLinear(s.new_kv, src_lin) catch return error.ExecutionFailed;
                    src_cached = src_lin;
                }
                if (dst_cached == null or dst_cached.? != dst_lin) {
                    if (dst_cached != null) hs.releaseMut(dst_tile.token);
                    dst_cached = null;
                    dst_tile = ctx.rstore.acquireTileDeviceMutLinear(s.cache, dst_lin) catch return error.ExecutionFailed;
                    dst_cached = dst_lin;
                }

                const src_off = tileRowOffset(src_tile, new_meta, b, h, t, src_coords) orelse return error.Unsupported;
                const dst_off = tileRowOffset(dst_tile, cache_meta, b, h, dst_t, dst_coords) orelse return error.Unsupported;
                if (src_off % 4 != 0 or dst_off % 4 != 0) return error.Unsupported;
                if (src_off + row_bytes > src_tile.len or dst_off + row_bytes > dst_tile.len) return error.ExecutionFailed;

                frame.recordCopy(
                    ctx.devmem.bufferFor(src_tile.handle).?,
                    src_off,
                    ctx.devmem.bufferFor(dst_tile.handle).?,
                    dst_off,
                    row_bytes,
                );
            }
        }
    }
}

// ---- ScatterRow ----------------------------------------------------------------

/// In-place row write buf[idx] = src (decode token emission). v1 mirrors the CPU
/// exec: buf/idx/src are each a single tile. Two paths: a device-side scatter
/// that reads the destination index ON DEVICE (4-byte scalars) so a GPU-computed
/// emit index forces no host round-trip, and a record-time host-index fallback
/// (f16/i8 or non-packed layouts) that submits pending work before its host read.
pub fn execScatterRow(ctx: Ctx, frame: *Frame, s: executable.StepScatterRow) ExecuteProgramError!void {
    const hs = ctx.rstore.tensorStore();
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
    if (context.totalTiles(buf_meta) != 1 or context.totalTiles(idx_meta) != 1 or context.totalTiles(src_meta) != 1) return error.Unsupported;

    const m = buf_meta.shape[0];
    var row_size: usize = 1;
    var d: usize = 1;
    while (d < @as(usize, buf_meta.rank)) : (d += 1) row_size *= buf_meta.shape[d];
    const row_bytes = row_size * elem_bytes;
    if (row_bytes % 4 != 0) return error.Unsupported; // device copy granularity

    const dsrc = ctx.rstore.acquireTileDeviceConstLinear(s.src, 0) catch return error.ExecutionFailed;
    const dbuf = ctx.rstore.acquireTileDeviceMutLinear(s.buf, 0) catch return error.ExecutionFailed;
    defer {
        hs.releaseConst(dsrc.token);
        hs.releaseMut(dbuf.token);
    }

    const src_rank: usize = @as(usize, dsrc.rank);
    const buf_rank: usize = @as(usize, dbuf.rank);
    const src_elems = context.packedElemsSized(dsrc.rank, dsrc.shape_mem[0..src_rank], dsrc.strides_mem[0..src_rank], elem_bytes) orelse return error.Unsupported;
    if (context.packedElemsSized(dbuf.rank, dbuf.shape_mem[0..buf_rank], dbuf.strides_mem[0..buf_rank], elem_bytes) == null) return error.Unsupported;
    if (src_elems < row_size) return error.ExecutionFailed;

    // Fast path: 4-byte scalar → ONE device dispatch with the index read on-device.
    if (elem_bytes == 4) {
        const di = ctx.rstore.acquireTileDeviceConstLinear(s.idx, 0) catch return error.ExecutionFailed;
        defer hs.releaseConst(di.token);
        if (context.storageBindingFits(ctx, dsrc.len) and context.storageBindingFits(ctx, dbuf.len)) fast: {
            if (m * row_bytes > dbuf.len) break :fast;
            const row_words = std.math.cast(u32, row_bytes / 4) orelse break :fast;
            const params: GatherParams = .{
                .rows = 1,
                .d = row_words,
                .v = @intCast(m),
                .total = row_words,
            };
            const built = try ctx.pipes.get(gather_kernel, "scatter_row_u32");
            const bufs = [_]c.WGPUBuffer{
                ctx.devmem.bufferFor(dsrc.handle).?,
                ctx.devmem.bufferFor(di.handle).?,
                ctx.devmem.bufferFor(dbuf.handle).?,
            };
            const sizes = [_]u64{ dsrc.len, di.len, dbuf.len };
            try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(row_words), 1, 1 });
            return;
        }
    }

    // Host fallback (f16/i8 or a layout the device path rejected): read the index
    // on the host at record time — submit any pending producer first (no-op for a
    // host-resident index). See `Ctx.submitPendingIfDeviceDirty`.
    try ctx.submitPendingIfDeviceDirty(frame, s.idx);
    const idx = try HostI32.acquire(hs, s.idx);
    defer idx.release();
    if (idx.vals.len < 1 or idx.vals[0] < 0) return error.ExecutionFailed;
    const row: usize = @intCast(idx.vals[0]);
    if (row >= m) return error.ExecutionFailed;
    if ((row + 1) * row_bytes > dbuf.len) return error.ExecutionFailed;

    frame.recordCopy(
        ctx.devmem.bufferFor(dsrc.handle).?,
        0,
        ctx.devmem.bufferFor(dbuf.handle).?,
        row * row_bytes,
        row_bytes,
    );
}

/// Byte offset of row (b, h, t, 0) inside its tile, from the tile's memory
/// strides. Null on negative strides (unsupported).
fn tileRowOffset(tile: resident_mod.TileRefDevice, meta: tensor_store_mod.TensorMeta, b: usize, h: usize, t: usize, coords: [4]usize) ?usize {
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
