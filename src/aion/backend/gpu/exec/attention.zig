// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Attention execution for the GPU backend.
//!
//! Two steps route here:
//!   - `AttentionTiled` -> `execAttention` (kernels/attention.wgsl):
//!     GQA over single-buffer k/v (f32 or f16 — bound as u32 words, no
//!     shader-f16 extension needed). Optional query positions and K/V lengths
//!     are read ON DEVICE, with identity/ring time mapping done in-kernel;
//!     only cache GROWTH needs the host, where K/V lengths are read at record time
//!     to pre-touch `mapSequenceStep` (same protocol as the CPU executor and the
//!     GPU KV-append) before metadata is re-fetched. Defaults are position == row
//!     and all T keys live, with no host round-trip.
//!   - `RelPosMHATiled` -> `execRelPosMHA` (kernels/relpos_mha.wgsl).
//!
//! v1 scope: f32 q/out; dk <= 512 (the kernels stage the q row in shared memory)
//! and dv <= 1024 (per-thread accumulator registers). Wider heads fall back with
//! `error.Unsupported`.

const std = @import("std");
const wgpu = @import("../wgpu.zig");
const pipelines = @import("../pipelines.zig");
const context = @import("../context.zig");
const backend_mod = @import("../../backend.zig");
const tensor_store_mod = @import("../../../runtime/tensor_store.zig");
const device_store = @import("../../../runtime/device_store.zig");
const executable = @import("../../../runtime/executable.zig");

const c = wgpu.c;
const Ctx = context.Ctx;
const Frame = @import("../frame.zig").Frame;
const ExecuteProgramError = backend_mod.ExecuteProgramError;
const KernelDesc = pipelines.KernelDesc;

const attn_kernel: KernelDesc = .{ .name = "attention", .wgsl = @embedFile("../kernels/attention.wgsl") };
const merge_kernel: KernelDesc = .{ .name = "attention_merge", .wgsl = @embedFile("../kernels/attention_merge.wgsl") };
const relpos_kernel: KernelDesc = .{ .name = "relpos_mha", .wgsl = @embedFile("../kernels/relpos_mha.wgsl") };

/// Kernel limits — must match MAX_DK / ACC * WG in the WGSL.
const MAX_DK: u32 = 512;
const MAX_DV: u32 = 1024;

// ---------------------------------------------------------------------------
// Row-block and split-K tuning.
//
// HEURISTICS, measured on an RTX 4080 Laptop (Vulkan) 2026-07-30 — deliberately NOT in
// `autotune.zig`, which exists for the matmul candidate MENU where the right tiling is
// genuinely device- and shape-dependent and worth timing on device. These are static
// parameters feeding a shape-adaptive rule (`chooseRowBlock`). Re-tune with
// `gpu-bench --suite kernels --op attn_seq|attn_cached|attn_window|relpos_mha|
// relpos_chunked`, and mind the clock-ramp trap that `bench_gpu.warmUp` exists for.
// ---------------------------------------------------------------------------

/// Query rows per attention workgroup; must match `RMAX` in attention.wgsl. 8 measured
/// SLOWER on all three attention shapes (register pressure from the per-row
/// `dots`/`sv`/`m_new`/`resc`/`acc` arrays) even though the freed workgroup memory
/// allows it — a measured ceiling, not an arbitrary one.
const MAX_ROWS: usize = 4;
/// Query rows per RelPosMHA workgroup; must match `RMAX` in relpos_mha.wgsl.
const MAX_RELPOS_ROWS: usize = 4;

/// Floats in each kernel's workgroup q-staging array — `q_s` in attention.wgsl and
/// `qu_s`/`qv_s` in relpos_mha.wgsl. A row block must satisfy `rows * dk <= this`, and
/// the value is what keeps both kernels inside the 16 KiB workgroup-storage floor.
/// WGSL array sizes must be literals, so this is a hand-kept mirror: change it here and
/// in the kernel together (the shaders name this constant in a comment).
const Q_STAGE_FLOATS: usize = 1024;
/// Block-grid size below which split-K pays off; ~2x the SM count of a mid/high-end
/// part. Raising it to 256 made the long-cache decode fall back to a narrower row block
/// and measured 1.6x slower, so it is a two-sided choice, not a floor to raise freely.
const MIN_BLOCKS: usize = 128;
/// Preferred keys per split-K segment (a full 256-key chunk plus headroom); shrunk
/// toward `MIN_SEG_KEYS` only as far as needed to fill the device.
const SEG_KEYS: usize = 512;
/// Floor on segment length — below this the score phase idles most of its threads.
///
/// 64 was tuned when attention was measured on prefill-ish shapes, where the block
/// grid is large and split-K never engages. At DECODE it is the binding constraint:
/// one query row over 8 heads with gqa=8 leaves only heads x segments as parallelism,
/// and 64 caps a 512-key span at 8 segments, so 4 head-blocks x 8 = 32 workgroups on
/// a 58-SM part.
///
/// Swept on device 2026-08-19 (`gpu-bench --suite attn --ctx 512`, per-op cost taken
/// as the slope of a --repeat sweep so the ~0.65 ms/execute harness floor drops out):
///
///   min_seg_keys   local us/op   global us/op   ms/token
///   64 (was)          57.1          132.2         2.52
///   32 (now)          44.3           91.4         1.88   <-- best
///   16                52.1           95.5         2.13
///    8                75.5           95.6         2.78
///
/// So it is a genuine optimum, not a floor to keep lowering: below 32 the segments
/// get too short and the score phase idles again. Confirmed on the full step:
/// 15.61 -> 14.99 ms/token, matching the isolated prediction of -0.64 ms.
/// Re-sweep by editing this constant; `gpu-bench --suite attn` reports the per-op cost.
const MIN_SEG_KEYS: usize = 32;
/// Cap on segments, keeping the partials buffer and the merge loop small.
const MAX_SEGS_DEFAULT: usize = 64;

// The three split-K parameters above were swept on device via `gpu-bench --suite attn`
// (the sweep table is in the MIN_SEG_KEYS doc comment) and are constants again now that
// it has concluded. They are decode-shaped, not prefill-shaped: with one query row, 8
// heads and gqa=8 the only parallelism left is heads x segments, and the old
// MIN_SEG_KEYS = 64 capped a 512-key span at 8 segments -> 4 head-blocks x 8 = 32
// workgroups on 58 SMs, measured at 63 us for a 0.5 MiB read (8.4 GB/s, ~2 % of peak).
//
// Also swept and rejected: forcing the head-rows per workgroup. `rh` is how many query
// heads share one read of the single KV head, so raising it cuts traffic 8x — and
// measured SLOWER (rh=1 1.84 ms vs rh=4 2.79 ms), because at decode it costs workgroups.
// `chooseRowBlock` derives it, and nothing overrides that.

fn ceilDiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

const RowBlock = struct { rh: usize, rl: usize };

/// Pick the (heads x rows) block each workgroup handles.
///
/// A bigger block reads the K/V range once for more query rows — traffic falls as
/// 1/rows — but it also divides the block grid, and a small grid can't keep the
/// device busy. So: take the LARGEST block whose grid still fills the device once
/// split-K has been applied. `rh` must divide `gqa` (never straddle a kv head) and
/// the tile's head count (the head base is a multiple of it).
///
/// When no candidate fills the device — a short sliding window over few heads, where
/// there simply isn't enough work — hedge at two rows rather than one: half the
/// traffic reduction for double the blocks measured best on that shape.
fn chooseRowBlock(gqa: usize, th: usize, tl: usize, tb: usize, span_hint: usize, d_k: usize) RowBlock {
    const max_segs: usize = @min(@max(@as(usize, 1), span_hint / MIN_SEG_KEYS), MAX_SEGS_DEFAULT);
    const rows_cap: usize = @min(MAX_ROWS, @max(@as(usize, 1), Q_STAGE_FLOATS / @max(d_k, 1)));

    var hedge: ?RowBlock = null;
    var target: usize = rows_cap;
    while (true) {
        var rh: usize = @min(@min(gqa, th), target);
        while (rh > 1 and ((gqa % rh) != 0 or (th % rh) != 0)) rh -= 1;
        const rl: usize = @max(@as(usize, 1), @min(tl, target / rh));
        const cand: RowBlock = .{ .rh = rh, .rl = rl };

        const blocks: usize = tb * ceilDiv(th, rh) * ceilDiv(tl, rl);
        if (blocks * max_segs >= MIN_BLOCKS) return cand;
        if (rh * rl == 2) hedge = cand;
        if (target <= 1) break;
        target /= 2;
    }
    return hedge orelse .{ .rh = 1, .rl = 1 };
}

/// Field order matches `struct Params` in attention.wgsl.
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
    base_l: u32,
    has_pos: u32,
    has_lengths: u32,
    rl: u32,
    rh: u32,
    /// The bound k/v tile's physical time range, and which partial slots this
    /// dispatch owns. A single-tile cache is `{0, t_cap, 0, segs}`.
    kv_t0: u32 = 0,
    kv_tile_t: u32 = 0,
    seg_base: u32 = 0,
    segs_local: u32 = 1,
};

/// Field order matches `struct Params` in attention_merge.wgsl.
const MergeParams = extern struct { rows: u32, segs: u32, dv: u32, stride: u32 };

/// Field order matches `struct Params` in relpos_mha.wgsl.
const RelPosParams = extern struct {
    t_q: u32,
    t_kv: u32,
    d: u32,
    pe_base: u32,
    u_base: u32,
    v_base: u32,
    has_mask: u32,
    chunk_size: u32,
    chunk_left: u32,
    rl: u32,
    scale: f32,
};

const packedElemsSized = context.packedElemsSized;

/// Conformer relative-positional MHA: one dispatch per (batch, head) slice.
/// The compile contract tiles q/k/v/out `[B, T, H, D]` as `[1, T, 1, D]`, so
/// each slice is one contiguous `[T, D]` panel.
pub fn execRelPosMHA(ctx: Ctx, frame: *Frame, s: executable.StepRelPosMHATiled) ExecuteProgramError!void {
    const hs = ctx.store;
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
    // RelPosMHA has ONE head dim for q/k/v, so the attention kernel's separate
    // dk/dv ceilings don't apply: the binding limits are the q staging arrays
    // (`rl * d <= RELPOS_Q_STAGE`, checked once `rl` is chosen) and the per-thread
    // accumulator array (`d <= ACC * WG`).
    if (d > MAX_DV) return error.Unsupported;
    if (t_q > context.MAX_GROUPS_PER_DIM) return error.Unsupported;
    // Per-head [P, D] / [D] panels must be intra-tile.
    if (pe_meta.tile_counts[1] != 1 or pe_meta.tile_counts[2] != 1) return error.Unsupported;
    if (bu_meta.tile_counts[1] != 1 or bv_meta.tile_counts[1] != 1) return error.Unsupported;

    const built = try ctx.pipes.get(relpos_kernel, "relpos_mha_row");

    // Mask: single packed tile, bound once (dummy = q when absent).
    var mask_tile: ?device_store.TileRef = null;
    defer if (mask_tile) |mt| hs.releaseConst(mt.token);
    if (s.mask) |mid| {
        const m_meta = hs.meta(mid) catch return error.ExecutionFailed;
        if (m_meta.rank != 2 or m_meta.dtype != .f32) return error.Unsupported;
        if (context.totalTiles(m_meta) != 1) return error.Unsupported;
        const mt = ctx.store.acquireTileDeviceConstLinear(mid, 0) catch return error.ExecutionFailed;
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
            const dq = ctx.store.acquireTileDeviceConstLinear(s.q, q_lin) catch return error.ExecutionFailed;
            const dk_t = ctx.store.acquireTileDeviceConstLinear(s.k, q_lin) catch return error.ExecutionFailed;
            const dv_t = ctx.store.acquireTileDeviceConstLinear(s.v, q_lin) catch return error.ExecutionFailed;
            const dout = ctx.store.acquireTileDeviceMutLinear(s.out, q_lin) catch return error.ExecutionFailed;

            const pe_ts0 = pe_meta.tile_shape[0];
            const dpe = ctx.store.acquireTileDeviceConstLinear(s.pos_emb, h / pe_ts0) catch return error.ExecutionFailed;
            const bu_ts0 = bu_meta.tile_shape[0];
            const dbu = ctx.store.acquireTileDeviceConstLinear(s.pos_bias_u, h / bu_ts0) catch return error.ExecutionFailed;
            const bv_ts0 = bv_meta.tile_shape[0];
            const dbv = ctx.store.acquireTileDeviceConstLinear(s.pos_bias_v, h / bv_ts0) catch return error.ExecutionFailed;
            defer {
                hs.releaseConst(dq.token);
                hs.releaseConst(dk_t.token);
                hs.releaseConst(dv_t.token);
                hs.releaseMut(dout.token);
                hs.releaseConst(dpe.token);
                hs.releaseConst(dbu.token);
                hs.releaseConst(dbv.token);
            }

            // `rl` query rows per workgroup: K/V rows are shared by the whole block
            // (pos_emb is not — its band shifts per row). Bounded by the kernel's
            // RMAX and by the q staging arrays.
            const rl: usize = @min(@min(MAX_RELPOS_ROWS, t_q), @max(@as(usize, 1), Q_STAGE_FLOATS / @max(d, 1)));
            if (rl * d > Q_STAGE_FLOATS) return error.Unsupported;

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
                .chunk_size = std.math.cast(u32, s.chunk_size) orelse return error.Unsupported,
                .chunk_left = std.math.cast(u32, s.chunk_left) orelse return error.Unsupported,
                .rl = @intCast(rl),
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
            try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ @intCast(ceilDiv(t_q, rl)), 1, 1 });
        }
    }
}

// ---- Attention --------------------------------------------------

/// Grouped-query attention over k/v — a KV cache when the step carries the index
/// operands, a plain sequence otherwise. One dispatch per out/q tile with a
/// workgroup per (b, l, hq); everything except cache growth stays on-device.
pub fn execAttention(ctx: Ctx, frame: *Frame, s: executable.StepAttentionTiled) ExecuteProgramError!void {
    const hs = ctx.store;
    const has_pos: bool = s.query_positions != null;
    const has_lengths: bool = s.kv_lengths != null;
    const out_meta = hs.meta(s.out) catch return error.ExecutionFailed;
    const q_meta = hs.meta(s.q) catch return error.ExecutionFailed;
    const k_meta = hs.meta(s.k) catch return error.ExecutionFailed;
    const v_meta = hs.meta(s.v) catch return error.ExecutionFailed;
    const pos_meta = if (s.query_positions) |t| hs.meta(t) catch return error.ExecutionFailed else null;
    const lengths_meta = if (s.kv_lengths) |t| hs.meta(t) catch return error.ExecutionFailed else null;

    if (out_meta.rank != 4 or q_meta.rank != 4 or k_meta.rank != 4 or v_meta.rank != 4) return error.Unsupported;
    if (has_pos and pos_meta.?.rank != 2) return error.Unsupported;
    if (has_lengths and lengths_meta.?.rank != 1) return error.Unsupported;
    // The output is always f32 (the CPU exec requires it too); q may be f16
    // independently of the cache dtype, matching the CPU's mixed q/k/v support.
    if (out_meta.dtype != .f32) return error.Unsupported;
    const q_f16 = switch (q_meta.dtype) {
        .f32 => false,
        .f16 => true,
        else => return error.Unsupported,
    };
    if (k_meta.dtype != v_meta.dtype) return error.Unsupported;
    const kv_f16 = switch (k_meta.dtype) {
        .f32 => false,
        .f16 => true,
        else => return error.Unsupported,
    };
    if (has_pos and pos_meta.?.dtype != .i32) return error.Unsupported;
    if (has_lengths and lengths_meta.?.dtype != .i32) return error.Unsupported;

    if (out_meta.tile_counts[3] != 1 or q_meta.tile_counts[3] != 1) return error.Unsupported;
    // A cache may be split across bindings along TIME only: every other axis
    // must be whole, since a key row must live entirely in one tile. K and V
    // must be split identically so one segment covers the same range of both.
    // K/V lengths are a single tiny tile.
    for ([_]usize{ 0, 2, 3 }) |axis| {
        if (k_meta.tile_counts[axis] != 1 or v_meta.tile_counts[axis] != 1) return error.Unsupported;
    }
    const kv_tiles = context.totalTiles(k_meta);
    if (context.totalTiles(v_meta) != kv_tiles) return error.Unsupported;
    if (k_meta.tile_shape[1] != v_meta.tile_shape[1]) return error.Unsupported;
    if (has_lengths and context.totalTiles(lengths_meta.?) != 1) return error.Unsupported;

    const batch = q_meta.shape[0];
    const h_q = q_meta.shape[2];
    const d_k = q_meta.shape[3];
    const h_kv = k_meta.shape[2];
    const d_v = v_meta.shape[3];
    if (h_q == 0 or h_kv == 0 or h_q % h_kv != 0) return error.Unsupported;
    if (k_meta.shape[0] != batch or v_meta.shape[0] != batch or v_meta.shape[2] != h_kv) return error.Unsupported;
    if (k_meta.shape[3] != d_k) return error.Unsupported;
    if (out_meta.shape[3] != d_v) return error.Unsupported;
    if (d_k > MAX_DK or d_v > MAX_DV) return error.Unsupported;
    if (kv_f16 and (d_k % 2 != 0 or d_v % 2 != 0)) return error.Unsupported; // word-aligned rows
    if (has_lengths and lengths_meta.?.shape[0] < batch) return error.Unsupported;

    // Attention is a read of K/V state, not an allocator. Sequence append (or
    // the model boundary for growable state) establishes capacity before this
    // frame is recorded.
    const t_cap = k_meta.shape[1];
    if (v_meta.shape[1] != t_cap or t_cap == 0) return error.Unsupported;

    // Time mapping: identity for none/growable, modulo for ring — resolved
    // in-kernel (no per-token host round-trips).
    const policy_info = hs.sequenceCachePolicyInfo(s.k);
    const is_ring = policy_info.kind == .ring;
    const ring_window: usize = if (is_ring) blk: {
        const configured: usize = policy_info.ring_window_tokens;
        break :blk if (configured == 0) t_cap else @min(configured, t_cap);
    } else 0;
    if (is_ring and ring_window == 0) return error.Unsupported;

    const built = try ctx.pipes.get(attn_kernel, if (q_f16) "attn_row_qf16" else "attn_row");

    const kv_elem: usize = if (kv_f16) 2 else 4;
    const kv_tile_t = k_meta.tile_shape[1];
    // WGSL has no optional binding, so slots 3/4 must be filled even when the
    // kernel never reads them: `dq` stands in, gated off by `has_idx`. Same
    // dummy-operand idiom `execRelPosMHA` uses for its optional mask.
    const dend_opt: ?device_store.TileRef = if (s.kv_lengths) |t|
        ctx.store.acquireTileDeviceConstLinear(t, 0) catch return error.ExecutionFailed
    else
        null;
    defer {
        if (dend_opt) |d| hs.releaseConst(d.token);
    }

    const total = context.totalTiles(out_meta);
    var ti: usize = 0;
    while (ti < total) : (ti += 1) {
        var coords: [tensor_store_mod.INLINE_RANK]usize = @splat(0);
        tensor_store_mod.decodeTileCoords(out_meta, ti, coords[0..4]) catch return error.ExecutionFailed;

        const dout = ctx.store.acquireTileDeviceMutLinear(s.out, ti) catch return error.ExecutionFailed;
        const q_lin = tensor_store_mod.encodeTileIndex(q_meta, coords[0..4]) catch return error.ExecutionFailed;
        const dq = ctx.store.acquireTileDeviceConstLinear(s.q, q_lin) catch return error.ExecutionFailed;
        const dpos_opt: ?device_store.TileRef = if (s.query_positions) |pid| blk: {
            const pos_lin = tensor_store_mod.encodeTileIndex(pos_meta.?, coords[0..2]) catch return error.ExecutionFailed;
            break :blk ctx.store.acquireTileDeviceConstLinear(pid, pos_lin) catch return error.ExecutionFailed;
        } else null;
        defer {
            hs.releaseMut(dout.token);
            hs.releaseConst(dq.token);
            if (dpos_opt) |d| hs.releaseConst(d.token);
        }
        if (!context.storageBindingFits(ctx, dq.len) or !context.storageBindingFits(ctx, dout.len)) return error.Unsupported;

        // Packed tiles: the kernel computes flat offsets from tile-local dims.
        if (context.packedElems(dout.rank, dout.shape_mem[0..4], dout.strides_mem[0..4]) == null) return error.Unsupported;
        // Packedness in q's OWN scalar size: an f16 query has a 2-byte innermost stride.
        if (context.packedElemsSized(dq.rank, dq.shape_mem[0..4], dq.strides_mem[0..4], if (q_f16) 2 else 4) == null) return error.Unsupported;
        if (dpos_opt) |dpos| {
            if (packedElemsSized(dpos.rank, dpos.shape_mem[0..2], dpos.strides_mem[0..2], 4) == null) return error.Unsupported;
        }

        const tb = dout.shape_mem[0];
        const tl = dout.shape_mem[1];
        const th = dout.shape_mem[2];
        if (dq.shape_mem[0] != tb or dq.shape_mem[1] != tl or dq.shape_mem[2] != th) return error.Unsupported;
        if (dq.shape_mem[3] != d_k or dout.shape_mem[3] != d_v) return error.Unsupported;
        if (dpos_opt) |dpos| {
            if (dpos.shape_mem[0] != tb or dpos.shape_mem[1] != tl) return error.Unsupported;
        }

        // Slots 3/4 fall back to `dq` when there are no index operands; the kernel
        // gates every read of them on `has_idx`, so the contents are never touched.
        const idx_bufs = [_]c.WGPUBuffer{
            ctx.devmem.bufferFor(if (dpos_opt) |d| d.handle else dq.handle).?,
            ctx.devmem.bufferFor(if (dend_opt) |d| d.handle else dq.handle).?,
        };
        const idx_sizes = [_]u64{
            if (dpos_opt) |d| d.len else dq.len,
            if (dend_opt) |d| d.len else dq.len,
        };
        if (tb > context.MAX_GROUPS_PER_DIM or tl > context.MAX_GROUPS_PER_DIM or th > context.MAX_GROUPS_PER_DIM) return error.Unsupported;

        const rows_total: usize = tb * tl * th;

        // `rh` query heads x `rl` query rows per workgroup, all sharing one K/V panel.
        // The split is sized off the range a row ACTUALLY scans — with a sliding
        // window that is the window, not the cache capacity.
        const gqa: usize = h_q / h_kv;
        const span_hint: usize = if (s.sliding_window > 0) @min(t_cap, s.sliding_window) else t_cap;
        const rb: RowBlock = chooseRowBlock(gqa, th, tl, tb, span_hint, d_k);
        const rh: usize = rb.rh;
        const rl: usize = rb.rl;
        // Both are kernel invariants: `rows` indexes fixed-size arrays there, and q
        // staging is bounded by `Q_STAGE_FLOATS`.
        if (rh * rl > MAX_ROWS or rh * rl * d_k > Q_STAGE_FLOATS) return error.Unsupported;

        const grid_h: usize = ceilDiv(th, rh);
        const grid_l: usize = ceilDiv(tl, rl);
        const blocks: usize = tb * grid_h * grid_l;
        var segs: usize = 1;
        if (blocks < MIN_BLOCKS and span_hint >= 2 * MIN_SEG_KEYS) {
            // Prefer segments that fill a whole 256-key chunk; shrink only as far as
            // needed to fill the device, since a short segment leaves threads idle in
            // the score phase.
            var seg_keys: usize = SEG_KEYS;
            while (seg_keys > MIN_SEG_KEYS and blocks * ceilDiv(span_hint, seg_keys) < MIN_BLOCKS) {
                seg_keys /= 2;
            }
            segs = @min(ceilDiv(span_hint, seg_keys), MAX_SEGS_DEFAULT);
            segs = @min(segs, @as(usize, context.MAX_GROUPS_PER_DIM) / @max(tb, 1));
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
            .segs = @intCast(kv_tiles * segs),
            .kv_tile_t = std.math.cast(u32, kv_tile_t) orelse return error.Unsupported,
            .segs_local = @intCast(segs),
            // Without explicit query positions, a query's position is its GLOBAL row, so the
            // kernel needs this tile's offset along L.
            .base_l = @intCast(coords[1] * out_meta.tile_shape[1]),
            .has_pos = @intFromBool(has_pos),
            .has_lengths = @intFromBool(has_lengths),
            .rl = @intCast(rl),
            .rh = @intCast(rh),
        };

        // Split path whenever there is more than one partial slot in total: an
        // adaptive key split, a cache split across bindings, or both. Each k/v
        // tile is its own dispatch (a tile is a distinct buffer), writing the
        // slots `[tile * segs, (tile + 1) * segs)`; `attn_merge` then combines
        // every slot exactly as it combines ordinary split-K segments.
        if (kv_tiles * segs > 1) {
            const stride: usize = d_v + 2;
            const part_bytes: u64 = @as(u64, rows_total) * kv_tiles * segs * stride * @sizeOf(f32);
            const scratch = ctx.scratch.ensure(ctx.gpu, part_bytes) catch return error.ExecutionFailed;
            const b1 = try ctx.pipes.get(attn_kernel, if (q_f16) "attn_split_qf16" else "attn_split");

            var kv_i: usize = 0;
            while (kv_i < kv_tiles) : (kv_i += 1) {
                const dk_c = ctx.store.acquireTileDeviceConstLinear(s.k, kv_i) catch return error.ExecutionFailed;
                defer hs.releaseConst(dk_c.token);
                const dv_c = ctx.store.acquireTileDeviceConstLinear(s.v, kv_i) catch return error.ExecutionFailed;
                defer hs.releaseConst(dv_c.token);
                if (!context.storageBindingFits(ctx, dk_c.len) or !context.storageBindingFits(ctx, dv_c.len)) return error.Unsupported;
                if (packedElemsSized(dk_c.rank, dk_c.shape_mem[0..4], dk_c.strides_mem[0..4], kv_elem) == null) return error.Unsupported;
                if (packedElemsSized(dv_c.rank, dv_c.shape_mem[0..4], dv_c.strides_mem[0..4], kv_elem) == null) return error.Unsupported;
                if (dk_c.shape_mem[1] != dv_c.shape_mem[1]) return error.Unsupported;

                var tile_params = params;
                tile_params.kv_t0 = std.math.cast(u32, kv_i * kv_tile_t) orelse return error.Unsupported;
                tile_params.kv_tile_t = std.math.cast(u32, dk_c.shape_mem[1]) orelse return error.Unsupported;
                tile_params.seg_base = std.math.cast(u32, kv_i * segs) orelse return error.Unsupported;

                const bufs = [_]c.WGPUBuffer{
                    ctx.devmem.bufferFor(dq.handle).?,
                    ctx.devmem.bufferFor(dk_c.handle).?,
                    ctx.devmem.bufferFor(dv_c.handle).?,
                    idx_bufs[0],
                    idx_bufs[1],
                    scratch,
                };
                const sizes = [_]u64{ dq.len, dk_c.len, dv_c.len, idx_sizes[0], idx_sizes[1], part_bytes };
                try frame.recordCompute(b1, &bufs, &sizes, std.mem.asBytes(&tile_params), .{
                    @intCast(grid_h),
                    @intCast(grid_l),
                    @intCast(tb * segs),
                });
            }
            {
                const b2 = try ctx.pipes.get(merge_kernel, "attn_merge");
                const mp: MergeParams = .{
                    .rows = @intCast(rows_total),
                    .segs = @intCast(kv_tiles * segs),
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

        // One partial slot means one k/v tile and no key split: bind it directly
        // and let the kernel normalize in place.
        const dk_c = ctx.store.acquireTileDeviceConstLinear(s.k, 0) catch return error.ExecutionFailed;
        defer hs.releaseConst(dk_c.token);
        const dv_c = ctx.store.acquireTileDeviceConstLinear(s.v, 0) catch return error.ExecutionFailed;
        defer hs.releaseConst(dv_c.token);
        if (!context.storageBindingFits(ctx, dk_c.len) or !context.storageBindingFits(ctx, dv_c.len)) return error.Unsupported;
        if (packedElemsSized(dk_c.rank, dk_c.shape_mem[0..4], dk_c.strides_mem[0..4], kv_elem) == null) return error.Unsupported;
        if (packedElemsSized(dv_c.rank, dv_c.shape_mem[0..4], dv_c.strides_mem[0..4], kv_elem) == null) return error.Unsupported;
        if (dk_c.shape_mem[1] != t_cap or dv_c.shape_mem[1] != t_cap) return error.Unsupported;

        const bufs = [_]c.WGPUBuffer{
            ctx.devmem.bufferFor(dq.handle).?,
            ctx.devmem.bufferFor(dk_c.handle).?,
            ctx.devmem.bufferFor(dv_c.handle).?,
            idx_bufs[0],
            idx_bufs[1],
            ctx.devmem.bufferFor(dout.handle).?,
        };
        const sizes = [_]u64{ dq.len, dk_c.len, dv_c.len, idx_sizes[0], idx_sizes[1], dout.len };
        try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{
            @intCast(grid_h),
            @intCast(grid_l),
            @intCast(tb),
        });
    }
}
