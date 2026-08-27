// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Shared benchmark core for the CPU (`backend/cpu/bench_cpu.zig`) and GPU
//! (`src/aion/backend/gpu/bench_gpu.zig`) benches. Owns the two things that must
//! stay in lockstep between them: the **op list** (`KOp`/`kInfo`/`buildK`, so the
//! two benches test the same kernels at the same shapes) and the **report format**
//! (`Sample`/`report`, so every row looks the same regardless of op or backend).
//!
//! Timing is here too (`nowNs`/`timeBackend`) since it's backend-agnostic; the GPU
//! bench adds its own kernel-only timer on top. Nothing here is GPU-specific, so
//! both artifacts can import it via the `aion` module.

const std = @import("std");
const aion = @import("aion");

pub const Backend = aion.backend.Backend;
pub const StorageManager = aion.storage_manager.StorageManager;
pub const Graph = aion.graph.Graph;
pub const TensorId = aion.storage_manager.TensorId;
pub const plan = aion.plan;

const out = std.debug;

// ---------------------------------------------------------------------------
// Unified report row
// ---------------------------------------------------------------------------

/// One benchmark result. `ns` is the primary per-run time (kernel time on GPU,
/// compute time on CPU) and is ALWAYS reported as ms/iter — the ground-truth
/// metric. GB/s and GFLOP/s are derived and only shown when the op provides the
/// basis (`bytes`/`flops`); ms/iter is the headline, throughput is context.
/// The optional tail carries backend extras: GPU end-to-end (`e2e_ns`), a
/// cross-backend reference (`ref_ns` + correctness), or a CPU checksum (`chk`).
pub const Sample = struct {
    label: []const u8,
    iters: usize,
    ns: u64,
    bytes: f64 = 0,
    flops: f64 = 0,
    e2e_ns: ?u64 = null,
    ref_ns: ?u64 = null,
    ref_label: []const u8 = "cpu",
    max_abs: ?f32 = null,
    i32_diffs: ?usize = null,
    chk: ?f64 = null,
};

/// Print one row in the shared columnar format. Columns are fixed-width so rows
/// line up whether or not a given op reports GB/s, GFLOP/s, or a comparison.
pub fn report(s: Sample) void {
    const f: f64 = @floatFromInt(@max(s.iters, 1));
    const ns_f: f64 = @floatFromInt(@max(s.ns, 1));
    const ms: f64 = @as(f64, @floatFromInt(s.ns)) / f / 1.0e6;

    out.print("  {s:<16} {d:8.3} ms", .{ s.label, ms });

    if (s.bytes > 0) {
        out.print(" {d:8.1} GB/s", .{s.bytes * f / ns_f});
    } else {
        out.print("             ", .{});
    }
    if (s.flops > 0) {
        out.print(" {d:8.1} GF/s", .{s.flops * f / ns_f});
    } else {
        out.print("             ", .{});
    }

    if (s.e2e_ns) |e| out.print(" | e2e {d:8.3} ms", .{@as(f64, @floatFromInt(e)) / f / 1.0e6});

    if (s.ref_ns) |r| {
        const ref_ms: f64 = @as(f64, @floatFromInt(r)) / f / 1.0e6;
        // Speed relative to the fullest primary time we have (e2e if present).
        const prim: f64 = @floatFromInt(@max(s.e2e_ns orelse s.ns, 1));
        const speedup: f64 = @as(f64, @floatFromInt(r)) / prim;
        out.print(" | {s} {d:8.3} ms {d:6.2}x", .{ s.ref_label, ref_ms, speedup });
        if (s.i32_diffs) |diffs| {
            out.print("  diffs={d}", .{diffs});
        } else if (s.max_abs) |m| {
            out.print("  max|d|={e:.1}", .{m});
        }
    } else if (s.ref_label.len != 0 and s.max_abs == null and s.i32_diffs == null and s.e2e_ns != null) {
        out.print(" | (gpu-only)", .{});
    }

    if (s.chk) |c| out.print("  chk={d:.4}", .{c});
    out.print("\n", .{});
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

pub fn nowNs() u64 {
    const ts: std.Io.Timestamp = std.Io.Clock.awake.now(std.Options.debug_io);
    const ns: i96 = ts.toNanoseconds();
    if (ns <= 0) return 0;
    return @intCast(@min(ns, @as(i96, std.math.maxInt(u64))));
}

/// Time `iters` executions of `prog` on `be` (one warmup excluded). Total ns for
/// the timed iterations. End-to-end: includes the per-call output D->H readback
/// for the GPU backend.
pub fn timeBackend(be: Backend, prog: *const aion.program.Program, store: aion.tensor_store.TensorStore, iters: usize) !u64 {
    // One persistent session across warmup + timed loop, so GPU residency stays
    // warm (weights upload once) instead of rebuilding per call.
    var session = try be.createSession(store);
    defer session.deinit();
    try session.execute(prog); // warmup (uploads weights / builds residency)
    const start = nowNs();
    var i: usize = 0;
    while (i < iters) : (i += 1) try session.execute(prog);
    return nowNs() - start;
}

// ---------------------------------------------------------------------------
// Deterministic tensor fills
// ---------------------------------------------------------------------------

/// Fill an f32 tensor with a deterministic pseudo-random pattern in ~[-2, 2).
pub fn fillTensor(alloc: std.mem.Allocator, mgr: *StorageManager, id: TensorId, count: usize, seed: usize) !void {
    const data = try alloc.alloc(f32, count);
    defer alloc.free(data);
    for (data, 0..) |*v, i| {
        const k: usize = (i * 2654435761 + seed * 97) % 1000;
        v.* = (@as(f32, @floatFromInt(k)) - 500.0) * 0.004;
    }
    try mgr.writeFromPackedScalar(id, std.mem.sliceAsBytes(data));
}

fn inputF32(alloc: std.mem.Allocator, g: *Graph, mgr: *StorageManager, shape: []const usize, tile: []const usize, seed: usize) !aion.graph.ValueId {
    var n: usize = 1;
    for (shape) |d| n *= d;
    const id = try mgr.createTiledTensor(.f32, shape, tile, .{});
    try fillTensor(alloc, mgr, id, n, seed);
    const v = try g.addInput(.f32, shape);
    try g.bindExternal(v, id);
    return v;
}

fn inputI32(g: *Graph, mgr: *StorageManager, shape: []const usize, tile: []const usize, vals: []const i32) !aion.graph.ValueId {
    const id = try mgr.createTiledTensor(.i32, shape, tile, .{});
    try mgr.writeFromPackedScalar(id, std.mem.sliceAsBytes(vals));
    const v = try g.addInput(.i32, shape);
    try g.bindExternal(v, id);
    return v;
}

/// The device a tile policy was derived for. The kernels bench sweeps CPU and GPU with
/// the same builders, so the target has to follow the policy rather than be assumed.
fn deviceFor(policy: plan.TilePolicy) aion.storage.DeviceRef {
    return .{ .kind = if (policy.target_kind == .cpu) .cpu else .gpu };
}

fn inputI32Pattern(alloc: std.mem.Allocator, g: *Graph, mgr: *StorageManager, shape: []const usize, tile: []const usize, seed: usize) !aion.graph.ValueId {
    var n: usize = 1;
    for (shape) |d| n *= d;
    const vals = try alloc.alloc(i32, n);
    defer alloc.free(vals);
    for (vals, 0..) |*v, i| v.* = @intCast((i * 2654435761 + seed * 97) % 1000);
    return inputI32(g, mgr, shape, tile, vals);
}

// ---------------------------------------------------------------------------
// Kernel suite: one op per row, fixed model-representative shapes so the CPU and
// GPU benches sweep an identical, drift-proof set. Shapes look like Gemma-decode
// / Nemotron-ASR work.
// ---------------------------------------------------------------------------

pub const KOp = enum {
    add_i32,
    lt_i32,
    cast_f16,
    cast_i32,
    copy,
    reduce_row,
    reduce_all,
    concat,
    transpose,
    slice,
    gather,
    rope,
    kv_append,
    argmax,
    scatter_row,
    attention,
    mha_cached,
    mha_window,
    relpos,
    relpos_chunked,
    conv1d,
    conv1d_dw,
    conv2d,
    lstm,
    rfft,
    stft,
};

pub const KInfo = struct {
    label: []const u8,
    /// Packed output element count (4 bytes/elem for both f32 and i32).
    out_elems: usize,
    /// Logical bytes moved per iteration (0 => don't report GB/s).
    bytes: f64 = 0,
    /// FLOPs per iteration (0 => don't report GFLOP/s).
    flops: f64 = 0,
    /// Output is i32 (compare bit-exact instead of float abs diff).
    out_i32: bool = false,
};

/// Fixed shapes, referenced by both `kInfo` and `buildK` so they can't drift.
pub const KA = struct {
    pub const EW = 2048 * 2048; // elementwise-family element count
    pub const RED_M = 2048;
    pub const RED_N = 2048;
    pub const CAT_M = 2048;
    pub const CAT_N = 1024; // per input; out [CAT_M, 2*CAT_N]
    pub const TR = 2048; // transpose [TR, TR]
    pub const SL_SRC = 4096;
    pub const SL = 2048; // slice [SL, SL] out of [SL_SRC, SL_SRC]
    pub const GA_V = 32768;
    pub const GA_D = 512;
    pub const GA_ROWS = 256; // gather rows
    pub const RO_T = 512;
    pub const RO_H = 8;
    pub const RO_D = 64; // rope [1, T, H, D]
    pub const KV_H = 8;
    pub const KV_T = 4096;
    pub const KV_D = 64; // kv cache [1, T, H, D]
    pub const AM_ROWS = 4;
    pub const AM_COLS = 262144; // argmax (decode logits row)
    pub const SC_ROWS = 4096;
    pub const SC_D = 512; // scatter buf
    pub const AT_B = 1;
    pub const AT_H = 8;
    pub const AT_T = 1024;
    pub const AT_D = 64; // attention
    pub const MC_HQ = 8;
    pub const MC_HKV = 2;
    pub const MC_T = 4096;
    pub const MC_D = 64; // cached GQA decode
    pub const MC_WIN = 512; // sliding window (Gemma-4 local layers)
    pub const MC_SOFT_CAP = 30.0; // attn logit soft cap (tanh), same layers
    pub const RP_T = 512;
    pub const RP_H = 8;
    pub const RP_D = 64; // rel-pos MHA
    // Chunked-limited config the Nemotron checkpoint was trained on
    // (att_context_size = [70, 13] => chunk = 14, left = 70, window = 84 of 512).
    pub const RP_CHUNK = 14;
    pub const RP_LEFT = 70;
    pub const RP_WIN = RP_LEFT + RP_CHUNK;
    pub const C1_L = 2048;
    pub const C1_C = 256;
    pub const C1_K = 9; // conv1d
    pub const CDW_C = 1024; // depthwise conv1d channels (Nemotron Conformer)
    pub const C2_HW = 128;
    pub const C2_CI = 32;
    pub const C2_CO = 64; // conv2d 3x3 stride 2
    pub const LS_B = 64;
    pub const LS_I = 1024;
    pub const LS_H = 1024; // lstm
    pub const FF_ROWS = 512;
    pub const FF_N = 1024; // rfft
    pub const ST_B = 4;
    pub const ST_S = 128000;
    pub const ST_NFFT = 512;
    pub const ST_HOP = 160; // stft
    pub const ST_FRAMES = 1 + ST_S / ST_HOP; // center framing
    pub const ST_BINS = ST_NFFT / 2 + 1;
};

pub fn kInfo(op: KOp) KInfo {
    const f = struct {
        fn ew(label: []const u8, passes: f64, i32out: bool) KInfo {
            return .{ .label = label, .out_elems = KA.EW, .bytes = passes * @as(f64, KA.EW) * 4.0, .out_i32 = i32out };
        }
    };
    return switch (op) {
        .add_i32 => f.ew("add_i32", 3, true),
        .lt_i32 => f.ew("lt_i32", 3, true),
        .cast_f16 => f.ew("cast_f16_rt", 3, false), // 4+2 down, 2+4 up = 12B/elem
        .cast_i32 => f.ew("cast_i32_rt", 4, false),
        .copy => f.ew("relu+copy", 4, false),
        .reduce_row => .{ .label = "reduce_row", .out_elems = KA.RED_M, .bytes = @as(f64, KA.RED_M * KA.RED_N + KA.RED_M) * 4.0 },
        .reduce_all => .{ .label = "reduce_all", .out_elems = 1, .bytes = @as(f64, KA.RED_M * KA.RED_N) * 4.0 },
        .concat => .{ .label = "concat", .out_elems = KA.CAT_M * KA.CAT_N * 2, .bytes = @as(f64, KA.CAT_M * KA.CAT_N * 2 * 2) * 4.0 },
        .transpose => .{ .label = "transpose2d", .out_elems = KA.TR * KA.TR, .bytes = @as(f64, KA.TR * KA.TR * 2) * 4.0 },
        .slice => .{ .label = "slice_nd", .out_elems = KA.SL * KA.SL, .bytes = @as(f64, KA.SL * KA.SL * 2) * 4.0 },
        .gather => .{ .label = "gather_rows", .out_elems = KA.GA_ROWS * KA.GA_D, .bytes = @as(f64, KA.GA_ROWS * KA.GA_D * 2) * 4.0 },
        .rope => .{ .label = "rope", .out_elems = KA.RO_T * KA.RO_H * KA.RO_D, .bytes = @as(f64, KA.RO_T * KA.RO_H * KA.RO_D * 2) * 4.0 },
        .kv_append => .{ .label = "kv_append", .out_elems = KA.KV_H * KA.KV_T * KA.KV_D, .bytes = @as(f64, KA.KV_H * KA.KV_D * 2) * 4.0 },
        .argmax => .{ .label = "argmax", .out_elems = KA.AM_ROWS, .bytes = @as(f64, KA.AM_ROWS * KA.AM_COLS) * 4.0, .out_i32 = true },
        .scatter_row => .{ .label = "scatter_row", .out_elems = KA.SC_ROWS * KA.SC_D, .bytes = @as(f64, KA.SC_D * 2) * 4.0 },
        .attention => .{
            .label = "attn_seq",
            .out_elems = KA.AT_B * KA.AT_H * KA.AT_T * KA.AT_D,
            // ~half the score matrix is masked out; QK^T + PV, 2 FLOP each.
            .flops = 2.0 * @as(f64, KA.AT_B * KA.AT_H) * @as(f64, KA.AT_T) * @as(f64, KA.AT_T) * @as(f64, KA.AT_D),
        },
        .mha_cached => .{
            .label = "attn_cached",
            .out_elems = KA.MC_HQ * KA.MC_D,
            .bytes = @as(f64, 2 * KA.MC_HKV * KA.MC_T * KA.MC_D) * 4.0, // k+v cache read
        },
        // Same cache, but only the last MC_WIN keys are in range — bytes count the
        // rows actually touched, so GB/s stays comparable with `attn_cached` and a
        // window that fails to narrow the read shows up as a collapsed rate.
        .mha_window => .{
            .label = "attn_window",
            .out_elems = KA.MC_HQ * KA.MC_D,
            .bytes = @as(f64, 2 * KA.MC_HKV * KA.MC_WIN * KA.MC_D) * 4.0,
        },
        .relpos => .{
            .label = "relpos_mha",
            .out_elems = KA.RP_T * KA.RP_H * KA.RP_D,
            .flops = 6.0 * @as(f64, KA.RP_H) * @as(f64, KA.RP_T) * @as(f64, KA.RP_T) * @as(f64, KA.RP_D),
        },
        // Same shape with the chunked-limited window engaged. FLOPs count only the
        // keys inside each row's window, so this row's GF/s is comparable with
        // `relpos_mha` above and its *time* shows what the window actually saves.
        .relpos_chunked => .{
            .label = "relpos_chunked",
            .out_elems = KA.RP_T * KA.RP_H * KA.RP_D,
            .flops = 6.0 * @as(f64, KA.RP_H) * @as(f64, KA.RP_T) * @as(f64, KA.RP_WIN) * @as(f64, KA.RP_D),
        },
        .conv1d => .{
            .label = "conv1d",
            .out_elems = KA.C1_L * KA.C1_C,
            .flops = 2.0 * @as(f64, KA.C1_L) * @as(f64, KA.C1_C) * @as(f64, KA.C1_K) * @as(f64, KA.C1_C),
        },
        .conv1d_dw => .{
            .label = "conv1d_dw",
            .out_elems = KA.C1_L * KA.CDW_C,
            // depthwise: k MACs per output element (no channel contraction).
            .flops = 2.0 * @as(f64, KA.C1_L) * @as(f64, KA.CDW_C) * @as(f64, KA.C1_K),
        },
        .conv2d => .{
            .label = "conv2d",
            .out_elems = (KA.C2_HW / 2) * (KA.C2_HW / 2) * KA.C2_CO,
            .flops = 2.0 * @as(f64, (KA.C2_HW / 2) * (KA.C2_HW / 2)) * @as(f64, KA.C2_CO) * 9.0 * @as(f64, KA.C2_CI),
        },
        .lstm => .{
            .label = "lstm_cell",
            .out_elems = KA.LS_B * 2 * KA.LS_H,
            .flops = 2.0 * @as(f64, KA.LS_B) * 4.0 * @as(f64, KA.LS_H) * @as(f64, KA.LS_I + KA.LS_H),
        },
        // FFT/STFT are O(n log n) on both backends now; the old naive-DFT FLOP
        // count would wildly overstate throughput, so report memory traffic
        // (input read once + one-sided spectrum written) as GB/s instead.
        .rfft => .{
            .label = "rfft",
            .out_elems = KA.FF_ROWS * (KA.FF_N + 2),
            .bytes = @as(f64, KA.FF_ROWS * KA.FF_N + KA.FF_ROWS * 2 * (KA.FF_N / 2 + 1)) * 4.0,
        },
        .stft => .{
            .label = "stft",
            .out_elems = KA.ST_B * KA.ST_FRAMES * (KA.ST_NFFT + 2),
            .bytes = @as(f64, KA.ST_B * KA.ST_S + KA.ST_B * KA.ST_FRAMES * 2 * KA.ST_BINS) * 4.0,
        },
    };
}

/// Build the graph+program for one kernel op with the given tile `policy` (each
/// backend passes its own). Returns the compiled program and its output tensor.
pub fn buildK(alloc: std.mem.Allocator, mgr: *StorageManager, op: KOp, policy: plan.TilePolicy) !struct { prog: aion.program.Program, out: TensorId } {
    var g = Graph.init(alloc);
    defer g.deinit();

    const out_v: aion.graph.ValueId = switch (op) {
        .add_i32 => blk: {
            const a = try inputI32Pattern(alloc, &g, mgr, &.{ 2048, 2048 }, &.{ 2048, 2048 }, 1);
            const b = try inputI32Pattern(alloc, &g, mgr, &.{ 2048, 2048 }, &.{ 2048, 2048 }, 2);
            break :blk try g.addElemwiseBinary(.add, a, b);
        },
        .lt_i32 => blk: {
            const a = try inputI32Pattern(alloc, &g, mgr, &.{ 2048, 2048 }, &.{ 2048, 2048 }, 3);
            const b = try inputI32Pattern(alloc, &g, mgr, &.{ 2048, 2048 }, &.{ 2048, 2048 }, 4);
            break :blk try g.addElemwiseBinary(.lt, a, b);
        },
        .cast_f16 => blk: {
            const x = try inputF32(alloc, &g, mgr, &.{ 2048, 2048 }, &.{ 2048, 2048 }, 5);
            break :blk try g.addCast(try g.addCast(x, .f16), .f32);
        },
        .cast_i32 => blk: {
            const x = try inputF32(alloc, &g, mgr, &.{ 2048, 2048 }, &.{ 2048, 2048 }, 6);
            break :blk try g.addCast(try g.addCast(x, .i32), .f32);
        },
        .copy => blk: {
            const x = try inputF32(alloc, &g, mgr, &.{ 2048, 2048 }, &.{ 2048, 2048 }, 7);
            break :blk try g.addCopy(try g.addRelu(x));
        },
        .reduce_row => blk: {
            const x = try inputF32(alloc, &g, mgr, &.{ KA.RED_M, KA.RED_N }, &.{ KA.RED_M, KA.RED_N }, 8);
            break :blk try g.addReduceAxis(.mean, x, -1);
        },
        .reduce_all => blk: {
            const x = try inputF32(alloc, &g, mgr, &.{ KA.RED_M, KA.RED_N }, &.{ KA.RED_M, KA.RED_N }, 9);
            break :blk try g.addReduce(.sum, x);
        },
        .concat => blk: {
            const a = try inputF32(alloc, &g, mgr, &.{ KA.CAT_M, KA.CAT_N }, &.{ KA.CAT_M, KA.CAT_N }, 10);
            const b = try inputF32(alloc, &g, mgr, &.{ KA.CAT_M, KA.CAT_N }, &.{ KA.CAT_M, KA.CAT_N }, 11);
            break :blk try g.addConcat(&.{ a, b }, 1);
        },
        .transpose => blk: {
            const x = try inputF32(alloc, &g, mgr, &.{ KA.TR, KA.TR }, &.{ KA.TR, KA.TR }, 12);
            break :blk try g.addViewTranspose2D(x);
        },
        .slice => blk: {
            const x = try inputF32(alloc, &g, mgr, &.{ KA.SL_SRC, KA.SL_SRC }, &.{ KA.SL_SRC, KA.SL_SRC }, 13);
            break :blk try g.addViewSliceND(x, &.{ 1024, 1024 }, &.{ KA.SL, KA.SL });
        },
        .gather => blk: {
            const table = try inputF32(alloc, &g, mgr, &.{ KA.GA_V, KA.GA_D }, &.{ KA.GA_V, KA.GA_D }, 14);
            const idx_vals = try alloc.alloc(i32, KA.GA_ROWS);
            defer alloc.free(idx_vals);
            for (idx_vals, 0..) |*v, i| v.* = @intCast((i * 2654435761 + 5) % KA.GA_V);
            const idx = try inputI32(&g, mgr, &.{ 1, KA.GA_ROWS }, &.{ 1, KA.GA_ROWS }, idx_vals);
            break :blk try g.addGather(table, idx, 0, 0);
        },
        .rope => blk: {
            const x = try inputF32(alloc, &g, mgr, &.{ 1, KA.RO_T, KA.RO_H, KA.RO_D }, &.{ 1, KA.RO_T, KA.RO_H, KA.RO_D }, 15);
            const pos_vals = try alloc.alloc(i32, KA.RO_T);
            defer alloc.free(pos_vals);
            for (pos_vals, 0..) |*v, i| v.* = @intCast(i);
            const pos = try inputI32(&g, mgr, &.{ 1, KA.RO_T }, &.{ 1, KA.RO_T }, pos_vals);
            break :blk try g.addRoPE1D(x, pos, 10000.0, 1.0, 1.0);
        },
        .kv_append => blk: {
            // Cache is [B, T, H, D] — time is dim 1, so one token is [B, 1, H, D].
            const cache = try inputF32(alloc, &g, mgr, &.{ 1, KA.KV_T, KA.KV_H, KA.KV_D }, &.{ 1, KA.KV_T, KA.KV_H, KA.KV_D }, 16);
            const new_kv = try inputF32(alloc, &g, mgr, &.{ 1, 1, KA.KV_H, KA.KV_D }, &.{ 1, 1, KA.KV_H, KA.KV_D }, 17);
            const end = try inputI32(&g, mgr, &.{1}, &.{1}, &.{KA.KV_T / 2});
            break :blk try g.addSequenceAppend(cache, new_kv, end);
        },
        .argmax => blk: {
            const x = try inputF32(alloc, &g, mgr, &.{ KA.AM_ROWS, KA.AM_COLS }, &.{ KA.AM_ROWS, KA.AM_COLS }, 18);
            break :blk try g.addArgMax(x, -1);
        },
        .scatter_row => blk: {
            const buf = try inputF32(alloc, &g, mgr, &.{ KA.SC_ROWS, KA.SC_D }, &.{ KA.SC_ROWS, KA.SC_D }, 19);
            const idx = try inputI32(&g, mgr, &.{1}, &.{1}, &.{1234});
            const src = try inputF32(alloc, &g, mgr, &.{KA.SC_D}, &.{KA.SC_D}, 20);
            break :blk try g.addScatterRow(buf, idx, src);
        },
        .attention => blk: {
            // Plain sequence, causal, no query-position/KV-length controls.
            // Unified layout: q [B, L_q, H_q, D], k/v [B, T, H_kv, D] — time is
            // dim 1. q is tiled per head so both backends get parallel work (CPU
            // threads over out tiles, GPU takes one dispatch per tile); k/v stay
            // single-tile because the GPU exec binds each cache as ONE buffer.
            const q = try inputF32(alloc, &g, mgr, &.{ KA.AT_B, KA.AT_T, KA.AT_H, KA.AT_D }, &.{ 1, KA.AT_T, 1, KA.AT_D }, 21);
            const k = try inputF32(alloc, &g, mgr, &.{ KA.AT_B, KA.AT_T, KA.AT_H, KA.AT_D }, &.{ KA.AT_B, KA.AT_T, KA.AT_H, KA.AT_D }, 22);
            const v = try inputF32(alloc, &g, mgr, &.{ KA.AT_B, KA.AT_T, KA.AT_H, KA.AT_D }, &.{ KA.AT_B, KA.AT_T, KA.AT_H, KA.AT_D }, 23);
            break :blk try g.addAttention(q, k, v, null, null, 0.125, true, 0, 0.0);
        },
        .mha_cached => blk: {
            // Decode: one query row (L_q = 1) against a long GQA cache. The 8
            // output rows over a 4096-key cache are what put the GPU on the
            // split-K (flash-decoding) path.
            const q = try inputF32(alloc, &g, mgr, &.{ 1, 1, KA.MC_HQ, KA.MC_D }, &.{ 1, 1, KA.MC_HQ, KA.MC_D }, 24);
            const kc = try inputF32(alloc, &g, mgr, &.{ 1, KA.MC_T, KA.MC_HKV, KA.MC_D }, &.{ 1, KA.MC_T, KA.MC_HKV, KA.MC_D }, 25);
            const vc = try inputF32(alloc, &g, mgr, &.{ 1, KA.MC_T, KA.MC_HKV, KA.MC_D }, &.{ 1, KA.MC_T, KA.MC_HKV, KA.MC_D }, 26);
            const pos = try inputI32(&g, mgr, &.{ 1, 1 }, &.{ 1, 1 }, &.{KA.MC_T - 1});
            const end = try inputI32(&g, mgr, &.{1}, &.{1}, &.{KA.MC_T});
            break :blk try g.addAttention(q, kc, vc, pos, end, 0.125, true, 0, 0.0);
        },
        .mha_window => blk: {
            // Gemma-4 local layer at decode: same cache as `mha_cached`, but with
            // a sliding window + logit soft cap (tanh per score) engaged.
            const q = try inputF32(alloc, &g, mgr, &.{ 1, 1, KA.MC_HQ, KA.MC_D }, &.{ 1, 1, KA.MC_HQ, KA.MC_D }, 24);
            const kc = try inputF32(alloc, &g, mgr, &.{ 1, KA.MC_T, KA.MC_HKV, KA.MC_D }, &.{ 1, KA.MC_T, KA.MC_HKV, KA.MC_D }, 25);
            const vc = try inputF32(alloc, &g, mgr, &.{ 1, KA.MC_T, KA.MC_HKV, KA.MC_D }, &.{ 1, KA.MC_T, KA.MC_HKV, KA.MC_D }, 26);
            const pos = try inputI32(&g, mgr, &.{ 1, 1 }, &.{ 1, 1 }, &.{KA.MC_T - 1});
            const end = try inputI32(&g, mgr, &.{1}, &.{1}, &.{KA.MC_T});
            break :blk try g.addAttention(q, kc, vc, pos, end, 0.125, true, KA.MC_WIN, KA.MC_SOFT_CAP);
        },
        .relpos => blk: {
            const t = KA.RP_T;
            const q = try inputF32(alloc, &g, mgr, &.{ 1, t, KA.RP_H, KA.RP_D }, &.{ 1, t, 1, KA.RP_D }, 27);
            const k = try inputF32(alloc, &g, mgr, &.{ 1, t, KA.RP_H, KA.RP_D }, &.{ 1, t, 1, KA.RP_D }, 28);
            const v = try inputF32(alloc, &g, mgr, &.{ 1, t, KA.RP_H, KA.RP_D }, &.{ 1, t, 1, KA.RP_D }, 29);
            const pe = try inputF32(alloc, &g, mgr, &.{ KA.RP_H, 2 * t - 1, KA.RP_D }, &.{ 1, 2 * t - 1, KA.RP_D }, 30);
            const u = try inputF32(alloc, &g, mgr, &.{ KA.RP_H, KA.RP_D }, &.{ KA.RP_H, KA.RP_D }, 31);
            const vb = try inputF32(alloc, &g, mgr, &.{ KA.RP_H, KA.RP_D }, &.{ KA.RP_H, KA.RP_D }, 32);
            break :blk try g.addRelPosMHA(q, k, v, pe, u, vb, null, 0.125, 0, 0);
        },
        .relpos_chunked => blk: {
            const t = KA.RP_T;
            const q = try inputF32(alloc, &g, mgr, &.{ 1, t, KA.RP_H, KA.RP_D }, &.{ 1, t, 1, KA.RP_D }, 27);
            const k = try inputF32(alloc, &g, mgr, &.{ 1, t, KA.RP_H, KA.RP_D }, &.{ 1, t, 1, KA.RP_D }, 28);
            const v = try inputF32(alloc, &g, mgr, &.{ 1, t, KA.RP_H, KA.RP_D }, &.{ 1, t, 1, KA.RP_D }, 29);
            const pe = try inputF32(alloc, &g, mgr, &.{ KA.RP_H, 2 * t - 1, KA.RP_D }, &.{ 1, 2 * t - 1, KA.RP_D }, 30);
            const u = try inputF32(alloc, &g, mgr, &.{ KA.RP_H, KA.RP_D }, &.{ KA.RP_H, KA.RP_D }, 31);
            const vb = try inputF32(alloc, &g, mgr, &.{ KA.RP_H, KA.RP_D }, &.{ KA.RP_H, KA.RP_D }, 32);
            break :blk try g.addRelPosMHA(q, k, v, pe, u, vb, null, 0.125, KA.RP_CHUNK, KA.RP_LEFT);
        },
        .conv1d => blk: {
            const x = try inputF32(alloc, &g, mgr, &.{ 1, KA.C1_L, KA.C1_C }, &.{ 1, KA.C1_L, KA.C1_C }, 33);
            const w = try inputF32(alloc, &g, mgr, &.{ KA.C1_K, KA.C1_C, KA.C1_C }, &.{ KA.C1_K, KA.C1_C, KA.C1_C }, 34);
            const b = try inputF32(alloc, &g, mgr, &.{KA.C1_C}, &.{KA.C1_C}, 35);
            break :blk try g.addConv1D(x, w, b, 1, 1, KA.C1_K / 2, KA.C1_K / 2, 1);
        },
        .conv1d_dw => blk: {
            // Depthwise causal (Nemotron Conformer): groups == c_in == c_out, k=9.
            const x = try inputF32(alloc, &g, mgr, &.{ 1, KA.C1_L, KA.CDW_C }, &.{ 1, KA.C1_L, KA.CDW_C }, 36);
            const w = try inputF32(alloc, &g, mgr, &.{ KA.C1_K, 1, KA.CDW_C }, &.{ KA.C1_K, 1, KA.CDW_C }, 37);
            const b = try inputF32(alloc, &g, mgr, &.{KA.CDW_C}, &.{KA.CDW_C}, 38);
            break :blk try g.addConv1D(x, w, b, 1, 1, KA.C1_K - 1, 0, KA.CDW_C);
        },
        .conv2d => blk: {
            const x = try inputF32(alloc, &g, mgr, &.{ 1, KA.C2_HW, KA.C2_HW, KA.C2_CI }, &.{ 1, KA.C2_HW, KA.C2_HW, KA.C2_CI }, 36);
            const w = try inputF32(alloc, &g, mgr, &.{ 3, 3, KA.C2_CI, KA.C2_CO }, &.{ 3, 3, KA.C2_CI, KA.C2_CO }, 37);
            const b = try inputF32(alloc, &g, mgr, &.{KA.C2_CO}, &.{KA.C2_CO}, 38);
            break :blk try g.addConv2D(x, w, b, 2, 2, 1, 1, 1, 0, 1, 0, 1);
        },
        .lstm => blk: {
            const x = try inputF32(alloc, &g, mgr, &.{ KA.LS_B, KA.LS_I }, &.{ KA.LS_B, KA.LS_I }, 39);
            const h = try inputF32(alloc, &g, mgr, &.{ KA.LS_B, KA.LS_H }, &.{ KA.LS_B, KA.LS_H }, 40);
            const cc = try inputF32(alloc, &g, mgr, &.{ KA.LS_B, KA.LS_H }, &.{ KA.LS_B, KA.LS_H }, 41);
            const w_ih = try inputF32(alloc, &g, mgr, &.{ KA.LS_I, 4 * KA.LS_H }, &.{ KA.LS_I, 4 * KA.LS_H }, 42);
            const w_hh = try inputF32(alloc, &g, mgr, &.{ KA.LS_H, 4 * KA.LS_H }, &.{ KA.LS_H, 4 * KA.LS_H }, 43);
            const b_ih = try inputF32(alloc, &g, mgr, &.{4 * KA.LS_H}, &.{4 * KA.LS_H}, 44);
            const b_hh = try inputF32(alloc, &g, mgr, &.{4 * KA.LS_H}, &.{4 * KA.LS_H}, 45);
            break :blk try g.addLSTMCell(x, h, cc, w_ih, w_hh, b_ih, b_hh);
        },
        .rfft => blk: {
            const x = try inputF32(alloc, &g, mgr, &.{ KA.FF_ROWS, KA.FF_N }, &.{ KA.FF_ROWS, KA.FF_N }, 46);
            break :blk try g.addRFFT(x);
        },
        .stft => blk: {
            const sig = try inputF32(alloc, &g, mgr, &.{ KA.ST_B, KA.ST_S }, &.{ KA.ST_B, KA.ST_S }, 47);
            const win = try inputF32(alloc, &g, mgr, &.{KA.ST_NFFT}, &.{KA.ST_NFFT}, 48);
            break :blk try g.addSTFT(sig, win, KA.ST_NFFT, KA.ST_HOP, true);
        },
    };
    try g.setOutputs(&[_]aion.graph.ValueId{out_v});
    const prog = try aion.program.compileGraph(alloc, &g, mgr, .init(deviceFor(policy), policy));
    return .{ .prog = prog, .out = prog.outputs[0] };
}
