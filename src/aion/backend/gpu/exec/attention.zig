// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Attention execution for the GPU backend.
//!
//! Two steps route here:
//!   - `Attention` -> `execAttention` (kernels/attention.wgsl):
//!     GQA over single-buffer k/v (f32 or f16 — bound as u32 words, no
//!     shader-f16 extension needed). Optional query positions and K/V lengths
//!     are read ON DEVICE, with identity/rolling time mapping done in-kernel;
//!     only cache GROWTH needs the host, where K/V lengths are read at record time
//!     to pre-touch `mapSequenceStep` (same protocol as the CPU executor and the
//!     GPU KV-append) before metadata is re-fetched. Defaults are position == row
//!     and all T keys live, with no host round-trip. An f32 prefill takes
//!     kernels/attention_block.wgsl: the same math, with a block of query rows per
//!     workgroup sharing each K/V row.
//!   - `RelPosMHA` -> `execRelPosMHA` (kernels/relpos_mha.wgsl).
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

/// Both attention kernels share the window helper; appended, since WGSL wants its
/// `enable` directives first and does not need forward declarations.
const window_wgsl = @embedFile("../kernels/window.wgsl");
const attn_kernel: KernelDesc = .{ .name = "attention", .wgsl = @embedFile("../kernels/attention.wgsl") ++ window_wgsl };
const block_kernel: KernelDesc = .{ .name = "attention_block", .wgsl = @embedFile("../kernels/attention_block.wgsl") ++ window_wgsl };
const merge_kernel: KernelDesc = .{ .name = "attention_merge", .wgsl = @embedFile("../kernels/attention_merge.wgsl") };
const relpos_kernel: KernelDesc = .{ .name = "relpos_mha", .wgsl = @embedFile("../kernels/relpos_mha.wgsl") ++ window_wgsl };

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

/// Rows and value dims per block-kernel workgroup; must match `R` and `DVS` in
/// attention_block.wgsl.
const BLOCK_ROWS: usize = 32;
const BLOCK_DV_SLICE: usize = 256;
/// Query positions from which a prefill takes the block kernel: below it the block is
/// mostly empty rows, and the decode kernel's key split fills the device instead.
const BLOCK_MIN_L: usize = 8;

/// Heads per block-kernel row block: the most of one kv head's group that divide
/// the head count, leaving at least four positions per block.
fn blockHeads(gqa: usize, th: usize) usize {
    var rh: usize = @min(@min(gqa, th), BLOCK_ROWS / 4);
    while (rh > 1 and ((gqa % rh) != 0 or (th % rh) != 0)) rh -= 1;
    return rh;
}

fn ceilDiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

const RowBlock = struct { rh: usize, rl: usize };

/// Keys one workgroup scores per pass; must match `WG` in attention.wgsl.
const KEY_CHUNK: usize = 256;

/// Keys the longest sequence has live, when the lengths are on the host to read at
/// record time; otherwise the capacity, which bounds them.
fn liveKeys(ctx: Ctx, lengths: ?tensor_store_mod.TensorId, t_cap: usize) ExecuteProgramError!usize {
    const id = lengths orelse return t_cap;
    if (!ctx.control.isHostPlaced(id)) return t_cap;
    const lease = try ctx.control.readI32(id);
    defer lease.release();
    var longest: usize = 0;
    for (lease.vals) |v| longest = @max(longest, @as(usize, @intCast(@max(v, 0))));
    return @min(longest, t_cap);
}

fn largestBlock(gqa: usize, th: usize, tl: usize, rows_cap: usize) RowBlock {
    var rh: usize = @min(@min(gqa, th), rows_cap);
    while (rh > 1 and ((gqa % rh) != 0 or (th % rh) != 0)) rh -= 1;
    return .{ .rh = rh, .rl = @max(@as(usize, 1), @min(tl, rows_cap / rh)) };
}

/// Pick the (heads x rows) block each workgroup handles.
///
/// A bigger block reads the K/V range once for more query rows — traffic falls as
/// 1/rows — but it also divides the block grid, and a small grid can't keep the
/// device busy. So: take the LARGEST block whose grid still fills the device once
/// split-K has been applied. `rh` must divide `gqa` (never straddle a kv head) and
/// the head count (the head base is a multiple of it).
///
/// When no candidate fills the device — a short sliding window over few heads, where
/// there simply isn't enough work — hedge at two rows rather than one: half the
/// traffic reduction for double the blocks measured best on that shape.
fn chooseRowBlock(gqa: usize, th: usize, tl: usize, tb: usize, span_hint: usize, d_k: usize) RowBlock {
    const max_segs: usize = @min(@max(@as(usize, 1), span_hint / MIN_SEG_KEYS), MAX_SEGS_DEFAULT);
    const rows_cap: usize = @min(MAX_ROWS, @max(@as(usize, 1), Q_STAGE_FLOATS / @max(d_k, 1)));
    // A range one workgroup scores in a single pass has no scan to parallelize:
    // more blocks would only repeat the q staging and barrier chains per block.
    // Take the largest block, sharing each K/V read across the most rows.
    if (span_hint <= KEY_CHUNK) return largestBlock(gqa, th, tl, rows_cap);

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

/// Field order matches `struct Params` in attention.wgsl (and attention_block.wgsl).
const CachedParams = extern struct {
    tl: u32,
    th: u32,
    dk: u32,
    dv: u32,
    t_cap: u32,
    h_kv: u32,
    gqa: u32,
    win_left: u32,
    win_right: u32,
    win_chunk: u32,
    ring: u32,
    ring_modulus: u32,
    kv_f16: u32,
    scale: f32,
    soft_cap: f32,
    segs: u32 = 1,
    has_pos: u32,
    has_lengths: u32,
    rl: u32,
    rh: u32,
};

/// Field order matches `struct Params` in attention_merge.wgsl.
const MergeParams = extern struct { rows: u32, segs: u32, dv: u32, stride: u32 };

/// Field order matches `struct Params` in relpos_mha.wgsl.
const RelPosParams = extern struct {
    t_q: u32,
    t_kv: u32,
    d: u32,
    p_len: u32,
    heads: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    has_mask: u32,
    win_left: u32,
    win_right: u32,
    win_chunk: u32,
    relative_zero_index: u32,
    rl: u32,
    scale: f32,
    attn_logits_soft_cap: f32,
};

/// Conformer relative-positional MHA over `[B, T, H, D]` q/k/v/out: one dispatch,
/// a workgroup per (row block, head, batch).
pub fn execRelPosMHA(ctx: Ctx, frame: *Frame, s: executable.StepRelPosMHA) ExecuteProgramError!void {
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
    if (p_len == 0 or s.relative_zero_index >= p_len or t_q > t_kv or t_q == 0) return error.Unsupported;
    // RelPosMHA has ONE head dim for q/k/v, so the attention kernel's separate
    // dk/dv ceilings don't apply: the binding limits are the q staging arrays
    // (`rl * d <= RELPOS_Q_STAGE`, checked once `rl` is chosen) and the per-thread
    // accumulator array (`d <= ACC * WG`).
    if (d > MAX_DV) return error.Unsupported;
    if (heads > context.MAX_GROUPS_PER_DIM or batch > context.MAX_GROUPS_PER_DIM) return error.Unsupported;
    inline for (.{ out_meta, q_meta, k_meta, pe_meta, bu_meta, bv_meta }) |m| if (m.chunks != 1) return error.Unsupported;
    if ((hs.meta(s.v) catch return error.ExecutionFailed).chunks != 1) return error.Unsupported;

    const built = try ctx.pipes.get(relpos_kernel, "relpos_mha_row");

    // Mask: bound once (dummy = q when absent).
    var dmask: ?device_store.Chunk = null;
    defer if (dmask) |mt| hs.releaseConst(mt.token);
    if (s.mask) |mid| {
        const m_meta = hs.meta(mid) catch return error.ExecutionFailed;
        if (m_meta.rank != 2 or m_meta.dtype != .f32) return error.Unsupported;
        if (m_meta.chunks != 1) return error.Unsupported;
        dmask = ctx.store.acquireConst(mid) catch return error.ExecutionFailed;
    }

    const dq = ctx.store.acquireConst(s.q) catch return error.ExecutionFailed;
    defer hs.releaseConst(dq.token);
    const dk_t = ctx.store.acquireConst(s.k) catch return error.ExecutionFailed;
    defer hs.releaseConst(dk_t.token);
    const dv_t = ctx.store.acquireConst(s.v) catch return error.ExecutionFailed;
    defer hs.releaseConst(dv_t.token);
    const dout = ctx.store.acquireMut(s.out) catch return error.ExecutionFailed;
    defer hs.releaseMut(dout.token);
    const dpe = ctx.store.acquireConst(s.pos_emb) catch return error.ExecutionFailed;
    defer hs.releaseConst(dpe.token);
    const dbu = ctx.store.acquireConst(s.pos_bias_u) catch return error.ExecutionFailed;
    defer hs.releaseConst(dbu.token);
    const dbv = ctx.store.acquireConst(s.pos_bias_v) catch return error.ExecutionFailed;
    defer hs.releaseConst(dbv.token);
    inline for (.{ dq.len, dk_t.len, dv_t.len, dout.len, dpe.len }) |len| if (!context.storageBindingFits(ctx, len)) return error.Unsupported;

    // `rl` query rows per workgroup: K/V rows are shared by the whole block
    // (pos_emb is not — its band shifts per row). Bounded by the kernel's RMAX
    // and by the q staging arrays.
    const rl: usize = @min(@min(MAX_RELPOS_ROWS, t_q), @max(@as(usize, 1), Q_STAGE_FLOATS / @max(d, 1)));
    if (rl * d > Q_STAGE_FLOATS) return error.Unsupported;
    const row_blocks = ceilDiv(t_q, rl);
    if (row_blocks > context.MAX_GROUPS_PER_DIM) return error.Unsupported;

    const params: RelPosParams = .{
        .t_q = @intCast(t_q),
        .t_kv = @intCast(t_kv),
        .d = @intCast(d),
        .p_len = @intCast(p_len),
        .heads = @intCast(heads),
        .has_mask = @intFromBool(dmask != null),
        .win_left = s.window.left,
        .win_right = s.window.right,
        .win_chunk = s.window.chunk,
        .relative_zero_index = std.math.cast(u32, s.relative_zero_index) orelse return error.Unsupported,
        .rl = @intCast(rl),
        .scale = s.scale,
        .attn_logits_soft_cap = s.attn_logits_soft_cap,
    };
    const mask_buf = if (dmask) |mt| ctx.devmem.bufferFor(mt.handle).? else ctx.devmem.bufferFor(dq.handle).?;
    const mask_len = if (dmask) |mt| mt.len else dq.len;
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
    try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ @intCast(row_blocks), @intCast(heads), @intCast(batch) });
}

// ---- Attention --------------------------------------------------

/// Grouped-query attention over k/v — a KV cache when the step carries the index
/// operands, a plain sequence otherwise. A workgroup per (b, row block, key
/// segment); everything except cache growth stays on-device.
pub fn execAttention(ctx: Ctx, frame: *Frame, s: executable.StepAttention) ExecuteProgramError!void {
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
    // Every operand is one device buffer; a tensor past the binding limit is
    // chunked, which the kernels do not address.
    inline for (.{ out_meta, q_meta, k_meta, v_meta }) |m| if (m.chunks != 1) return error.Unsupported;
    if (has_pos and pos_meta.?.chunks != 1) return error.Unsupported;
    if (has_lengths and lengths_meta.?.chunks != 1) return error.Unsupported;

    const batch = q_meta.shape[0];
    const l_q = q_meta.shape[1];
    const h_q = q_meta.shape[2];
    const d_k = q_meta.shape[3];
    const h_kv = k_meta.shape[2];
    const d_v = v_meta.shape[3];
    if (h_q == 0 or h_kv == 0 or h_q % h_kv != 0) return error.Unsupported;
    if (k_meta.shape[0] != batch or v_meta.shape[0] != batch or v_meta.shape[2] != h_kv) return error.Unsupported;
    if (k_meta.shape[3] != d_k) return error.Unsupported;
    if (out_meta.shape[0] != batch or out_meta.shape[1] != l_q or out_meta.shape[2] != h_q or out_meta.shape[3] != d_v) return error.Unsupported;
    if (has_pos and (pos_meta.?.shape[0] != batch or pos_meta.?.shape[1] != l_q)) return error.Unsupported;
    if (d_k > MAX_DK or d_v > MAX_DV) return error.Unsupported;
    if (kv_f16 and (d_k % 2 != 0 or d_v % 2 != 0)) return error.Unsupported; // word-aligned rows
    if (has_lengths and lengths_meta.?.shape[0] < batch) return error.Unsupported;
    if (batch > context.MAX_GROUPS_PER_DIM or l_q > context.MAX_GROUPS_PER_DIM or h_q > context.MAX_GROUPS_PER_DIM) return error.Unsupported;
    if (batch == 0 or l_q == 0) return;

    // Attention is a read of K/V state, not an allocator. Sequence append (or
    // the model boundary for growable state) establishes capacity before this
    // frame is recorded.
    const t_cap = k_meta.shape[1];
    if (v_meta.shape[1] != t_cap or t_cap == 0) return error.Unsupported;

    // Time mapping: identity for none/growable, modulo for rolling — resolved
    // in-kernel (no per-token host round-trips). Ring is the physical layout; the
    // modulus is the whole allocation, not the retained-history bound the model
    // declared.
    const is_ring = hs.sequenceCachePolicyInfo(s.k).kind == .rolling;
    const ring_modulus: usize = if (is_ring) t_cap else 0;

    const dq = hs.acquireConst(s.q) catch return error.ExecutionFailed;
    defer hs.releaseConst(dq.token);
    const dk = hs.acquireConst(s.k) catch return error.ExecutionFailed;
    defer hs.releaseConst(dk.token);
    const dv = hs.acquireConst(s.v) catch return error.ExecutionFailed;
    defer hs.releaseConst(dv.token);
    const dout = hs.acquireMut(s.out) catch return error.ExecutionFailed;
    defer hs.releaseMut(dout.token);
    // WGSL has no optional binding, so slots 3/4 must be filled even when the
    // kernel never reads them: `dq` stands in, gated off by `has_pos` /
    // `has_lengths`. Same dummy-operand idiom `execRelPosMHA` uses for its mask.
    const dpos_opt: ?device_store.Chunk = if (s.query_positions) |t| hs.acquireConst(t) catch return error.ExecutionFailed else null;
    defer if (dpos_opt) |d| hs.releaseConst(d.token);
    const dend_opt: ?device_store.Chunk = if (s.kv_lengths) |t| hs.acquireConst(t) catch return error.ExecutionFailed else null;
    defer if (dend_opt) |d| hs.releaseConst(d.token);
    inline for (.{ dq.len, dk.len, dv.len, dout.len }) |len| if (!context.storageBindingFits(ctx, len)) return error.Unsupported;

    const q_buf = ctx.devmem.bufferFor(dq.handle).?;
    const k_buf = ctx.devmem.bufferFor(dk.handle).?;
    const v_buf = ctx.devmem.bufferFor(dv.handle).?;
    const out_buf = ctx.devmem.bufferFor(dout.handle).?;
    const pos_buf = if (dpos_opt) |d| ctx.devmem.bufferFor(d.handle).? else q_buf;
    const pos_len = if (dpos_opt) |d| d.len else dq.len;
    const end_buf = if (dend_opt) |d| ctx.devmem.bufferFor(d.handle).? else q_buf;
    const end_len = if (dend_opt) |d| d.len else dq.len;

    const gqa: usize = h_q / h_kv;
    var params: CachedParams = .{
        .tl = @intCast(l_q),
        .th = @intCast(h_q),
        .dk = @intCast(d_k),
        .dv = @intCast(d_v),
        .t_cap = std.math.cast(u32, t_cap) orelse return error.Unsupported,
        .h_kv = @intCast(h_kv),
        .gqa = @intCast(gqa),
        .win_left = s.window.left,
        .win_right = s.window.right,
        .win_chunk = s.window.chunk,
        .ring = @intFromBool(is_ring),
        .ring_modulus = std.math.cast(u32, ring_modulus) orelse return error.Unsupported,
        .kv_f16 = @intFromBool(kv_f16),
        .scale = s.scale,
        .soft_cap = s.attn_logits_soft_cap,
        .has_pos = @intFromBool(has_pos),
        .has_lengths = @intFromBool(has_lengths),
        .rl = 0,
        .rh = 0,
    };

    // Prefill: a block of query rows shares every K/V row it reads.
    if (!q_f16 and l_q >= BLOCK_MIN_L) {
        const rh = blockHeads(gqa, h_q);
        params.rh = @intCast(rh);
        params.rl = @intCast(BLOCK_ROWS / rh);
        const built = try ctx.pipes.get(block_kernel, "attn_block");
        const bufs = [_]c.WGPUBuffer{ q_buf, k_buf, v_buf, pos_buf, end_buf, out_buf };
        const sizes = [_]u64{ dq.len, dk.len, dv.len, pos_len, end_len, dout.len };
        return frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{
            @intCast(ceilDiv(h_q, rh) * ceilDiv(d_v, BLOCK_DV_SLICE)),
            @intCast(ceilDiv(l_q, params.rl)),
            @intCast(batch),
        });
    }

    // `rh` query heads x `rl` query rows per workgroup, all sharing one K/V panel.
    // The split is sized off the range a row ACTUALLY scans — with a sliding
    // window that is the window, not the cache capacity; with host-visible
    // lengths, the keys written so far, not the slots allocated for them.
    const span_hint: usize = s.window.maxKeys(try liveKeys(ctx, s.kv_lengths, t_cap));
    const rb: RowBlock = chooseRowBlock(gqa, h_q, l_q, batch, span_hint, d_k);
    // Both are kernel invariants: `rows` indexes fixed-size arrays there, and q
    // staging is bounded by `Q_STAGE_FLOATS`.
    if (rb.rh * rb.rl > MAX_ROWS or rb.rh * rb.rl * d_k > Q_STAGE_FLOATS) return error.Unsupported;
    params.rh = @intCast(rb.rh);
    params.rl = @intCast(rb.rl);

    const grid_h: usize = ceilDiv(h_q, rb.rh);
    const grid_l: usize = ceilDiv(l_q, rb.rl);
    const blocks: usize = batch * grid_h * grid_l;
    var segs: usize = 1;
    if (blocks < MIN_BLOCKS and span_hint > KEY_CHUNK) {
        // Prefer segments that fill a whole 256-key chunk; shrink only as far as
        // needed to fill the device, since a short segment leaves threads idle in
        // the score phase.
        var seg_keys: usize = SEG_KEYS;
        while (seg_keys > MIN_SEG_KEYS and blocks * ceilDiv(span_hint, seg_keys) < MIN_BLOCKS) {
            seg_keys /= 2;
        }
        segs = @min(ceilDiv(span_hint, seg_keys), MAX_SEGS_DEFAULT);
        segs = @min(segs, @as(usize, context.MAX_GROUPS_PER_DIM) / batch);
        if (segs < 2) segs = 1;
    }
    params.segs = @intCast(segs);

    if (segs == 1) {
        // No key split: the kernel normalizes in place.
        const built = try ctx.pipes.get(attn_kernel, if (q_f16) "attn_row_qf16" else "attn_row");
        const bufs = [_]c.WGPUBuffer{ q_buf, k_buf, v_buf, pos_buf, end_buf, out_buf };
        const sizes = [_]u64{ dq.len, dk.len, dv.len, pos_len, end_len, dout.len };
        return frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ @intCast(grid_h), @intCast(grid_l), @intCast(batch) });
    }

    // Split-K: each segment writes an unnormalized partial, `attn_merge` combines.
    const rows_total: usize = batch * l_q * h_q;
    const stride: usize = d_v + 2;
    const part_bytes: u64 = @as(u64, rows_total) * segs * stride * @sizeOf(f32);
    const scratch = ctx.scratch.ensure(ctx.gpu, part_bytes) catch return error.ExecutionFailed;
    {
        const built = try ctx.pipes.get(attn_kernel, if (q_f16) "attn_split_qf16" else "attn_split");
        const bufs = [_]c.WGPUBuffer{ q_buf, k_buf, v_buf, pos_buf, end_buf, scratch };
        const sizes = [_]u64{ dq.len, dk.len, dv.len, pos_len, end_len, part_bytes };
        try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ @intCast(grid_h), @intCast(grid_l), @intCast(batch * segs) });
    }
    const built = try ctx.pipes.get(merge_kernel, "attn_merge");
    const mp: MergeParams = .{ .rows = @intCast(rows_total), .segs = @intCast(segs), .dv = @intCast(d_v), .stride = @intCast(stride) };
    const bufs = [_]c.WGPUBuffer{ scratch, out_buf };
    const sizes = [_]u64{ part_bytes, dout.len };
    try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&mp), .{ @intCast(rows_total), 1, 1 });
}
