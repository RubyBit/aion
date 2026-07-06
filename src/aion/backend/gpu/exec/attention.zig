// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Attention execution for the GPU backend.
//!
//! Three steps route here:
//!   - `AttentionTiled` / `MultiHeadAttentionTiled` -> `execAttention`
//!     (kernels/attention.wgsl): the compiler tiles lead (batch/head) dims as
//!     size-1 and — under the GPU tile policy — the whole [m, dk]/[n, dk]/
//!     [n, dv] slice into ONE tile, so each slice is one dispatch with one
//!     workgroup per query row.
//!   - `MultiHeadAttentionCachedTiled` -> `execAttentionCached`
//!     (kernels/attention_cached.wgsl): cached GQA over single-buffer KV caches
//!     (f32 or f16 — bound as u32 words, no shader-f16 extension needed), with
//!     positions/end_index read ON DEVICE and identity/ring time mapping done
//!     in-kernel. Only cache GROWTH needs the host: end_index is read at record
//!     time to pre-touch `mapKVCacheTime` (same protocol as the CPU executor
//!     and the GPU KV-append), then metadata is re-fetched.
//!
//! v1 scope: f32 q/out (+f32 kernels for plain attention); dk <= 512 (the
//! kernels stage the q row in shared memory) and dv <= 1024 (per-thread
//! accumulator registers). Wider heads fall back with `error.Unsupported`.

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

const attn_kernel: KernelDesc = .{ .name = "attention", .wgsl = @embedFile("../kernels/attention.wgsl") };
const cached_kernel: KernelDesc = .{ .name = "attention_cached", .wgsl = @embedFile("../kernels/attention_cached.wgsl") };
const merge_kernel: KernelDesc = .{ .name = "attention_merge", .wgsl = @embedFile("../kernels/attention_merge.wgsl") };
const relpos_kernel: KernelDesc = .{ .name = "relpos_mha", .wgsl = @embedFile("../kernels/relpos_mha.wgsl") };

/// Kernel limits — must match MAX_DK / ACC * WG in the WGSL.
const MAX_DK: u32 = 512;
const MAX_DV: u32 = 1024;

/// Field order matches `struct Params` in attention.wgsl.
const AttnParams = extern struct {
    m: u32,
    n: u32,
    dk: u32,
    dv: u32,
    q_row: u32,
    k_row: u32,
    v_row: u32,
    o_row: u32,
    scale: f32,
    causal: u32,
    _p0: u32 = 0,
    _p1: u32 = 0,
};

/// Field order matches `struct Params` in attention_cached.wgsl.
const CachedParams = extern struct {
    base_b: u32,
    base_h: u32,
    tl: u32,
    th: u32,
    dk: u32,
    dv: u32,
    t_cap: u32,
    h_kv: u32,
    gqa: u32,
    causal: u32,
    sliding: u32,
    ring: u32,
    ring_window: u32,
    kv_f16: u32,
    scale: f32,
    soft_cap: f32,
    segs: u32 = 1,
    _p0: u32 = 0,
    _p1: u32 = 0,
    _p2: u32 = 0,
};

/// Field order matches `struct Params` in attention_merge.wgsl.
const MergeParams = extern struct { rows: u32, segs: u32, dv: u32, stride: u32 };

const packedElemsSized = context.packedElemsSized;

// ---- Attention / MultiHeadAttention -------------------------------------------

/// Fused attention over lead-dim slices. `s` is either `StepAttentionTiled` or
/// `StepMultiHeadAttentionTiled` (identical field layout for our purposes: the
/// head count is just another size-1-tiled lead dim).
pub fn execAttention(ctx: Ctx, frame: *Frame, s: anytype) ExecuteProgramError!void {
    const hs = ctx.rstore.tensorStore();
    const out_meta = hs.meta(s.out) catch return error.ExecutionFailed;
    const q_meta = hs.meta(s.q) catch return error.ExecutionFailed;
    const k_meta = hs.meta(s.k) catch return error.ExecutionFailed;
    const v_meta = hs.meta(s.v) catch return error.ExecutionFailed;

    if (out_meta.dtype != .f32 or q_meta.dtype != .f32 or k_meta.dtype != .f32 or v_meta.dtype != .f32) return error.Unsupported;
    const rank: usize = @as(usize, out_meta.rank);
    if (rank < 2) return error.Unsupported;

    // v1: each slice's [m,dk]/[n,dk]/[n,dv]/[m,dv] must live in ONE tile (the
    // GPU tile policy produces exactly this; CPU-policy programs fall back).
    inline for (.{ out_meta, q_meta, k_meta, v_meta }) |meta| {
        if (meta.tile_counts[rank - 2] != 1 or meta.tile_counts[rank - 1] != 1) return error.Unsupported;
    }

    const built = try ctx.pipes.get(attn_kernel, "attn_row_f32");

    // Lead dims are tiled size-1 with equal counts across operands (compile
    // contract), and trailing counts are all 1 — so one linear tile index
    // enumerates the same slice in all four tensors.
    const total = context.totalTiles(out_meta);
    var ti: usize = 0;
    while (ti < total) : (ti += 1) {
        const dq = ctx.rstore.acquireTileDeviceConstLinear(s.q, ti) catch return error.ExecutionFailed;
        const dk_t = ctx.rstore.acquireTileDeviceConstLinear(s.k, ti) catch return error.ExecutionFailed;
        const dv_t = ctx.rstore.acquireTileDeviceConstLinear(s.v, ti) catch return error.ExecutionFailed;
        const dout = ctx.rstore.acquireTileDeviceMutLinear(s.out, ti) catch return error.ExecutionFailed;
        defer {
            hs.releaseConst(dq.token);
            hs.releaseConst(dk_t.token);
            hs.releaseConst(dv_t.token);
            hs.releaseMut(dout.token);
        }
        if (!context.storageBindingFits(ctx, dq.len) or !context.storageBindingFits(ctx, dk_t.len) or
            !context.storageBindingFits(ctx, dv_t.len) or !context.storageBindingFits(ctx, dout.len)) return error.Unsupported;

        const qv = context.rowView(dq.rank, dq.shape_mem[0..@as(usize, dq.rank)], dq.strides_mem[0..@as(usize, dq.rank)]) orelse return error.Unsupported;
        const kv = context.rowView(dk_t.rank, dk_t.shape_mem[0..@as(usize, dk_t.rank)], dk_t.strides_mem[0..@as(usize, dk_t.rank)]) orelse return error.Unsupported;
        const vv = context.rowView(dv_t.rank, dv_t.shape_mem[0..@as(usize, dv_t.rank)], dv_t.strides_mem[0..@as(usize, dv_t.rank)]) orelse return error.Unsupported;
        const ov = context.rowView(dout.rank, dout.shape_mem[0..@as(usize, dout.rank)], dout.strides_mem[0..@as(usize, dout.rank)]) orelse return error.Unsupported;

        if (qv.cols != kv.cols) return error.Unsupported; // dk
        if (kv.rows != vv.rows) return error.Unsupported; // n
        if (ov.rows != qv.rows or ov.cols != vv.cols) return error.Unsupported;
        if (qv.cols > MAX_DK or ov.cols > MAX_DV) return error.Unsupported;
        if (ov.rows > context.MAX_GROUPS_PER_DIM) return error.Unsupported;
        if (kv.rows == 0) return error.Unsupported;

        const params: AttnParams = .{
            .m = ov.rows,
            .n = kv.rows,
            .dk = qv.cols,
            .dv = ov.cols,
            .q_row = qv.row_stride,
            .k_row = kv.row_stride,
            .v_row = vv.row_stride,
            .o_row = ov.row_stride,
            .scale = s.scale,
            .causal = @intFromBool(s.causal),
        };
        const bufs = [_]c.WGPUBuffer{
            ctx.devmem.bufferFor(dq.handle).?,
            ctx.devmem.bufferFor(dk_t.handle).?,
            ctx.devmem.bufferFor(dv_t.handle).?,
            ctx.devmem.bufferFor(dout.handle).?,
        };
        const sizes = [_]u64{ dq.len, dk_t.len, dv_t.len, dout.len };
        try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ ov.rows, 1, 1 });
    }
}

// ---- RelPosMHA ----------------------------------------------------------------

/// Field order matches `struct Params` in relpos_mha.wgsl.
const RelPosParams = extern struct {
    t_q: u32,
    t_kv: u32,
    d: u32,
    pe_base: u32,
    u_base: u32,
    v_base: u32,
    has_mask: u32,
    _pad: u32 = 0,
    scale: f32,
};

/// Conformer relative-positional MHA: one dispatch per (batch, head) slice.
/// The compile contract tiles q/k/v/out `[B, T, H, D]` as `[1, T, 1, D]`, so
/// each slice is one contiguous `[T, D]` panel.
pub fn execRelPosMHA(ctx: Ctx, frame: *Frame, s: executable.StepRelPosMHATiled) ExecuteProgramError!void {
    const hs = ctx.rstore.tensorStore();
    const out_meta = hs.meta(s.out) catch return error.ExecutionFailed;
    const q_meta = hs.meta(s.q) catch return error.ExecutionFailed;
    const k_meta = hs.meta(s.k) catch return error.ExecutionFailed;
    const pe_meta = hs.meta(s.pos_emb) catch return error.ExecutionFailed;
    const bu_meta = hs.meta(s.pos_bias_u) catch return error.ExecutionFailed;
    const bv_meta = hs.meta(s.pos_bias_v) catch return error.ExecutionFailed;

    if (out_meta.rank != 4 or q_meta.rank != 4 or pe_meta.rank != 3) return error.Unsupported;
    if (out_meta.dtype != .f32 or q_meta.dtype != .f32 or pe_meta.dtype != .f32) return error.Unsupported;

    const batch = q_meta.shape[0];
    const t_q = q_meta.shape[1];
    const heads = q_meta.shape[2];
    const d = q_meta.shape[3];
    const t_kv = k_meta.shape[1];
    const p_len = pe_meta.shape[1];
    if (p_len != 2 * t_kv - 1 or t_q > t_kv or t_q == 0) return error.Unsupported;
    if (d > MAX_DK or d > MAX_DV) return error.Unsupported;
    if (t_q > context.MAX_GROUPS_PER_DIM) return error.Unsupported;
    // Per-head [P, D] / [D] panels must be intra-tile.
    if (pe_meta.tile_counts[1] != 1 or pe_meta.tile_counts[2] != 1) return error.Unsupported;
    if (bu_meta.tile_counts[1] != 1 or bv_meta.tile_counts[1] != 1) return error.Unsupported;

    const built = try ctx.pipes.get(relpos_kernel, "relpos_mha_row");

    // Mask: single packed tile, bound once (dummy = q when absent).
    var mask_tile: ?resident_mod.TileRefDevice = null;
    defer if (mask_tile) |mt| hs.releaseConst(mt.token);
    if (s.mask) |mid| {
        const m_meta = hs.meta(mid) catch return error.ExecutionFailed;
        if (m_meta.rank != 2 or m_meta.dtype != .f32) return error.Unsupported;
        if (context.totalTiles(m_meta) != 1) return error.Unsupported;
        const mt = ctx.rstore.acquireTileDeviceConstLinear(mid, 0) catch return error.ExecutionFailed;
        mask_tile = mt;
        const mn = context.packedElems(mt.rank, mt.shape_mem[0..2], mt.strides_mem[0..2]) orelse return error.Unsupported;
        if (mn < t_q * t_kv) return error.Unsupported;
    }

    var b: usize = 0;
    while (b < batch) : (b += 1) {
        var h: usize = 0;
        while (h < heads) : (h += 1) {
            const qkv_coords = [4]usize{ b, 0, h, 0 };
            const q_lin = tensor_store_mod.encodeTileIndex(q_meta, &qkv_coords) catch return error.ExecutionFailed;
            const dq = ctx.rstore.acquireTileDeviceConstLinear(s.q, q_lin) catch return error.ExecutionFailed;
            const dk_t = ctx.rstore.acquireTileDeviceConstLinear(s.k, q_lin) catch return error.ExecutionFailed;
            const dv_t = ctx.rstore.acquireTileDeviceConstLinear(s.v, q_lin) catch return error.ExecutionFailed;
            const dout = ctx.rstore.acquireTileDeviceMutLinear(s.out, q_lin) catch return error.ExecutionFailed;

            const pe_ts0 = pe_meta.tile_shape[0];
            const dpe = ctx.rstore.acquireTileDeviceConstLinear(s.pos_emb, h / pe_ts0) catch return error.ExecutionFailed;
            const bu_ts0 = bu_meta.tile_shape[0];
            const dbu = ctx.rstore.acquireTileDeviceConstLinear(s.pos_bias_u, h / bu_ts0) catch return error.ExecutionFailed;
            const bv_ts0 = bv_meta.tile_shape[0];
            const dbv = ctx.rstore.acquireTileDeviceConstLinear(s.pos_bias_v, h / bv_ts0) catch return error.ExecutionFailed;
            defer {
                hs.releaseConst(dq.token);
                hs.releaseConst(dk_t.token);
                hs.releaseConst(dv_t.token);
                hs.releaseMut(dout.token);
                hs.releaseConst(dpe.token);
                hs.releaseConst(dbu.token);
                hs.releaseConst(dbv.token);
            }

            // Slice panels and the per-head tables must be packed.
            if ((context.packedElems(dq.rank, dq.shape_mem[0..4], dq.strides_mem[0..4]) orelse return error.Unsupported) < t_q * d) return error.Unsupported;
            if ((context.packedElems(dk_t.rank, dk_t.shape_mem[0..4], dk_t.strides_mem[0..4]) orelse return error.Unsupported) < t_kv * d) return error.Unsupported;
            if ((context.packedElems(dv_t.rank, dv_t.shape_mem[0..4], dv_t.strides_mem[0..4]) orelse return error.Unsupported) < t_kv * d) return error.Unsupported;
            if (context.packedElems(dout.rank, dout.shape_mem[0..4], dout.strides_mem[0..4]) == null) return error.Unsupported;
            if (context.packedElems(dpe.rank, dpe.shape_mem[0..3], dpe.strides_mem[0..3]) == null) return error.Unsupported;
            if (context.packedElems(dbu.rank, dbu.shape_mem[0..2], dbu.strides_mem[0..2]) == null) return error.Unsupported;
            if (context.packedElems(dbv.rank, dbv.shape_mem[0..2], dbv.strides_mem[0..2]) == null) return error.Unsupported;
            if (!context.storageBindingFits(ctx, dpe.len)) return error.Unsupported;

            const params: RelPosParams = .{
                .t_q = @intCast(t_q),
                .t_kv = @intCast(t_kv),
                .d = @intCast(d),
                .pe_base = @intCast((h % pe_ts0) * p_len * d),
                .u_base = @intCast((h % bu_ts0) * d),
                .v_base = @intCast((h % bv_ts0) * d),
                .has_mask = @intFromBool(mask_tile != null),
                .scale = s.scale,
            };
            const mask_buf = if (mask_tile) |mt| ctx.devmem.bufferFor(mt.handle).? else ctx.devmem.bufferFor(dq.handle).?;
            const mask_len = if (mask_tile) |mt| mt.len else dq.len;
            const bufs = [_]c.WGPUBuffer{
                ctx.devmem.bufferFor(dq.handle).?,
                ctx.devmem.bufferFor(dk_t.handle).?,
                ctx.devmem.bufferFor(dv_t.handle).?,
                ctx.devmem.bufferFor(dpe.handle).?,
                ctx.devmem.bufferFor(dbu.handle).?,
                ctx.devmem.bufferFor(dbv.handle).?,
                mask_buf,
                ctx.devmem.bufferFor(dout.handle).?,
            };
            const sizes = [_]u64{ dq.len, dk_t.len, dv_t.len, dpe.len, dbu.len, dbv.len, mask_len, dout.len };
            try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ @intCast(t_q), 1, 1 });
        }
    }
}

// ---- MultiHeadAttentionCached --------------------------------------------------

/// Cached GQA over external KV caches. One dispatch per out/q tile with a
/// workgroup per (b, l, hq); everything except cache growth stays on-device.
pub fn execAttentionCached(ctx: Ctx, frame: *Frame, s: executable.StepMultiHeadAttentionCachedTiled) ExecuteProgramError!void {
    const hs = ctx.rstore.tensorStore();
    const out_meta = hs.meta(s.out) catch return error.ExecutionFailed;
    const q_meta = hs.meta(s.q) catch return error.ExecutionFailed;
    var k_meta = hs.meta(s.k_cache) catch return error.ExecutionFailed;
    var v_meta = hs.meta(s.v_cache) catch return error.ExecutionFailed;
    const pos_meta = hs.meta(s.positions) catch return error.ExecutionFailed;
    const end_meta = hs.meta(s.end_index) catch return error.ExecutionFailed;

    if (out_meta.rank != 4 or q_meta.rank != 4 or k_meta.rank != 4 or v_meta.rank != 4) return error.Unsupported;
    if (pos_meta.rank != 2 or end_meta.rank != 1) return error.Unsupported;
    if (out_meta.dtype != .f32 or q_meta.dtype != .f32) return error.Unsupported; // f16 q later
    if (k_meta.dtype != v_meta.dtype) return error.Unsupported;
    const kv_f16 = switch (k_meta.dtype) {
        .f32 => false,
        .f16 => true,
        else => return error.Unsupported,
    };
    if (pos_meta.dtype != .i32 or end_meta.dtype != .i32) return error.Unsupported;

    if (out_meta.tile_counts[3] != 1 or q_meta.tile_counts[3] != 1) return error.Unsupported;
    // v1: each cache is ONE device buffer (decode caches are created that way);
    // end_index is a single tiny tile.
    if (context.totalTiles(k_meta) != 1 or context.totalTiles(v_meta) != 1) return error.Unsupported;
    if (context.totalTiles(end_meta) != 1) return error.Unsupported;

    const batch = q_meta.shape[0];
    const h_q = q_meta.shape[2];
    const d_k = q_meta.shape[3];
    const h_kv = k_meta.shape[1];
    const d_v = v_meta.shape[3];
    if (h_q == 0 or h_kv == 0 or h_q % h_kv != 0) return error.Unsupported;
    if (k_meta.shape[0] != batch or v_meta.shape[0] != batch or v_meta.shape[1] != h_kv) return error.Unsupported;
    if (k_meta.shape[3] != d_k) return error.Unsupported;
    if (out_meta.shape[3] != d_v) return error.Unsupported;
    if (d_k > MAX_DK or d_v > MAX_DV) return error.Unsupported;
    if (kv_f16 and (d_k % 2 != 0 or d_v % 2 != 0)) return error.Unsupported; // word-aligned rows
    if (end_meta.shape[0] < batch) return error.Unsupported;

    // Pre-touch cache growth on the host (growable policies may reallocate),
    // then re-fetch metadata — same protocol as the CPU executor.
    {
        const tile = hs.acquireTileConstLinear(s.end_index, 0) catch return error.ExecutionFailed;
        defer hs.releaseConst(tile.token);
        const ptr: [*]align(1) const i32 = @ptrCast(tile.bytes.ptr);
        const vals = ptr[0 .. tile.bytes.len / @sizeOf(i32)];
        if (vals.len < batch) return error.ExecutionFailed;
        var b: usize = 0;
        while (b < batch) : (b += 1) {
            if (vals[b] < 0) return error.ExecutionFailed;
            const valid_end: usize = @intCast(vals[b]);
            if (valid_end == 0) continue;
            _ = hs.mapKVCacheTime(s.k_cache, valid_end - 1, k_meta.shape[2]) catch return error.ExecutionFailed;
            _ = hs.mapKVCacheTime(s.v_cache, valid_end - 1, v_meta.shape[2]) catch return error.ExecutionFailed;
        }
    }
    k_meta = hs.meta(s.k_cache) catch return error.ExecutionFailed;
    v_meta = hs.meta(s.v_cache) catch return error.ExecutionFailed;
    const t_cap = k_meta.shape[2];
    if (v_meta.shape[2] != t_cap or t_cap == 0) return error.Unsupported;

    // Time mapping: identity for none/growable, modulo for ring — resolved
    // in-kernel (no per-token host round-trips).
    const policy_info = hs.kvCachePolicyInfo(s.k_cache);
    const is_ring = policy_info.kind == .ring;
    const ring_window: usize = if (is_ring) blk: {
        const configured: usize = policy_info.ring_window_tokens;
        break :blk if (configured == 0) t_cap else @min(configured, t_cap);
    } else 0;
    if (is_ring and ring_window == 0) return error.Unsupported;

    const built = try ctx.pipes.get(cached_kernel, "mha_cached");

    const kv_elem: usize = if (kv_f16) 2 else 4;
    const dk_c = ctx.rstore.acquireTileDeviceConstLinear(s.k_cache, 0) catch return error.ExecutionFailed;
    const dv_c = ctx.rstore.acquireTileDeviceConstLinear(s.v_cache, 0) catch return error.ExecutionFailed;
    const dend = ctx.rstore.acquireTileDeviceConstLinear(s.end_index, 0) catch return error.ExecutionFailed;
    defer {
        hs.releaseConst(dk_c.token);
        hs.releaseConst(dv_c.token);
        hs.releaseConst(dend.token);
    }
    if (!context.storageBindingFits(ctx, dk_c.len) or !context.storageBindingFits(ctx, dv_c.len)) return error.Unsupported;
    if (packedElemsSized(dk_c.rank, dk_c.shape_mem[0..4], dk_c.strides_mem[0..4], kv_elem) == null) return error.Unsupported;
    if (packedElemsSized(dv_c.rank, dv_c.shape_mem[0..4], dv_c.strides_mem[0..4], kv_elem) == null) return error.Unsupported;
    if (dk_c.shape_mem[2] != t_cap or dv_c.shape_mem[2] != t_cap) return error.Unsupported;

    const total = context.totalTiles(out_meta);
    var ti: usize = 0;
    while (ti < total) : (ti += 1) {
        var coords: [tensor_store_mod.INLINE_RANK]usize = @splat(0);
        tensor_store_mod.decodeTileCoords(out_meta, ti, coords[0..4]) catch return error.ExecutionFailed;

        const dout = ctx.rstore.acquireTileDeviceMutLinear(s.out, ti) catch return error.ExecutionFailed;
        const q_lin = tensor_store_mod.encodeTileIndex(q_meta, coords[0..4]) catch return error.ExecutionFailed;
        const dq = ctx.rstore.acquireTileDeviceConstLinear(s.q, q_lin) catch return error.ExecutionFailed;
        const pos_lin = tensor_store_mod.encodeTileIndex(pos_meta, coords[0..2]) catch return error.ExecutionFailed;
        const dpos = ctx.rstore.acquireTileDeviceConstLinear(s.positions, pos_lin) catch return error.ExecutionFailed;
        defer {
            hs.releaseMut(dout.token);
            hs.releaseConst(dq.token);
            hs.releaseConst(dpos.token);
        }
        if (!context.storageBindingFits(ctx, dq.len) or !context.storageBindingFits(ctx, dout.len)) return error.Unsupported;

        // Packed tiles: the kernel computes flat offsets from tile-local dims.
        if (context.packedElems(dout.rank, dout.shape_mem[0..4], dout.strides_mem[0..4]) == null) return error.Unsupported;
        if (context.packedElems(dq.rank, dq.shape_mem[0..4], dq.strides_mem[0..4]) == null) return error.Unsupported;
        if (packedElemsSized(dpos.rank, dpos.shape_mem[0..2], dpos.strides_mem[0..2], 4) == null) return error.Unsupported;

        const tb = dout.shape_mem[0];
        const tl = dout.shape_mem[1];
        const th = dout.shape_mem[2];
        if (dq.shape_mem[0] != tb or dq.shape_mem[1] != tl or dq.shape_mem[2] != th) return error.Unsupported;
        if (dq.shape_mem[3] != d_k or dout.shape_mem[3] != d_v) return error.Unsupported;
        if (dpos.shape_mem[0] != tb or dpos.shape_mem[1] != tl) return error.Unsupported;
        if (tb > context.MAX_GROUPS_PER_DIM or tl > context.MAX_GROUPS_PER_DIM or th > context.MAX_GROUPS_PER_DIM) return error.Unsupported;

        // Split-K (flash-decoding) when the plain grid can't occupy the GPU:
        // few output rows over a long cache. Each segment covers ~512 keys;
        // capped so partials stay small and the merge loop stays cheap.
        const rows_total: usize = tb * tl * th;
        var segs: usize = 1;
        if (rows_total <= 64 and t_cap >= 1024) {
            segs = @min(@min(t_cap / 512, 32), @as(usize, context.MAX_GROUPS_PER_DIM) / @max(tb, 1));
            if (segs < 2) segs = 1;
        }

        const params: CachedParams = .{
            .base_b = @intCast(coords[0] * out_meta.tile_shape[0]),
            .base_h = @intCast(coords[2] * out_meta.tile_shape[2]),
            .tl = @intCast(tl),
            .th = @intCast(th),
            .dk = @intCast(d_k),
            .dv = @intCast(d_v),
            .t_cap = std.math.cast(u32, t_cap) orelse return error.Unsupported,
            .h_kv = @intCast(h_kv),
            .gqa = @intCast(h_q / h_kv),
            .causal = @intFromBool(s.causal),
            .sliding = std.math.cast(u32, s.sliding_window) orelse return error.Unsupported,
            .ring = @intFromBool(is_ring),
            .ring_window = std.math.cast(u32, ring_window) orelse return error.Unsupported,
            .kv_f16 = @intFromBool(kv_f16),
            .scale = s.scale,
            .soft_cap = s.attn_logits_soft_cap,
            .segs = @intCast(segs),
        };

        if (segs > 1) {
            const stride: usize = d_v + 2;
            const part_bytes: u64 = @as(u64, rows_total) * segs * stride * @sizeOf(f32);
            const scratch = ctx.scratch.ensure(ctx.gpu, part_bytes) catch return error.ExecutionFailed;
            {
                const b1 = try ctx.pipes.get(cached_kernel, "mha_cached_split");
                const bufs = [_]c.WGPUBuffer{
                    ctx.devmem.bufferFor(dq.handle).?,
                    ctx.devmem.bufferFor(dk_c.handle).?,
                    ctx.devmem.bufferFor(dv_c.handle).?,
                    ctx.devmem.bufferFor(dpos.handle).?,
                    ctx.devmem.bufferFor(dend.handle).?,
                    scratch,
                };
                const sizes = [_]u64{ dq.len, dk_c.len, dv_c.len, dpos.len, dend.len, part_bytes };
                try frame.recordCompute(b1, &bufs, &sizes, std.mem.asBytes(&params), .{
                    @intCast(th),
                    @intCast(tl),
                    @intCast(tb * segs),
                });
            }
            {
                const b2 = try ctx.pipes.get(merge_kernel, "mha_cached_merge");
                const mp: MergeParams = .{
                    .rows = @intCast(rows_total),
                    .segs = @intCast(segs),
                    .dv = @intCast(d_v),
                    .stride = @intCast(stride),
                };
                const bufs = [_]c.WGPUBuffer{ scratch, ctx.devmem.bufferFor(dout.handle).? };
                const sizes = [_]u64{ part_bytes, dout.len };
                try frame.recordCompute(b2, &bufs, &sizes, std.mem.asBytes(&mp), .{
                    @intCast(rows_total),
                    1,
                    1,
                });
            }
            continue;
        }

        const bufs = [_]c.WGPUBuffer{
            ctx.devmem.bufferFor(dq.handle).?,
            ctx.devmem.bufferFor(dk_c.handle).?,
            ctx.devmem.bufferFor(dv_c.handle).?,
            ctx.devmem.bufferFor(dpos.handle).?,
            ctx.devmem.bufferFor(dend.handle).?,
            ctx.devmem.bufferFor(dout.handle).?,
        };
        const sizes = [_]u64{ dq.len, dk_c.len, dv_c.len, dpos.len, dend.len, dout.len };
        try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{
            @intCast(th),
            @intCast(tl),
            @intCast(tb),
        });
    }
}
