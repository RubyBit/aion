// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Model-faithful decode graphs for the benches.
//!
//! `kernels.zig` beside it benchmarks single ops; this benchmarks a whole **model
//! step**, so a change can be judged against the thing that actually ships. The
//! Gemma-4 E2B builder mirrors `scripts/convert_gemma4_e2b_to_aion.py`'s
//! `_emit_forward` op for op and compiles through the real `compileGraph`, so
//! `fuse_horizontal_matmul` and `fuse_gpu_decode` run exactly as they do on the
//! loaded model.
//!
//! Two properties are load-bearing, and both exist because of measurement bugs
//! this module is meant to prevent:
//!
//!  1. **Every weight gets its own storage at its real size.** Decode is a
//!     bandwidth problem; the working set has to be the real ~2.4 GB or the GPU
//!     serves weights out of L2. On a 432 GB/s part with 48 MB of L2, a bench that
//!     re-reads one 20 MB weight reports figures ABOVE the DRAM peak -- which is
//!     arithmetically impossible and has previously been mistaken for a win.
//!  2. **Weights are filled per tile, in place.** `writeFromPackedQuant` wants a
//!     full-size packed staging buffer; the per-layer embedding table alone is
//!     2.5 GB. Writing `TiledTensor.data` directly keeps host peak negligible while
//!     still producing well-formed q8_0 blocks (see `fillQuantPattern`).
//!
//! Fidelity is not assumed: `printStepHistogram` dumps the step counts so they can
//! be diffed against the real model's, and a scaled-down `--layers` build still
//! covers every layer *kind* (source/non-source x local/global, and both elastic
//! FFN widths).

const std = @import("std");
const aion = @import("aion");

pub const Graph = aion.graph.Graph;
pub const StorageManager = aion.storage_manager.StorageManager;
pub const TensorId = aion.storage_manager.TensorId;
pub const ValueId = aion.graph.ValueId;
pub const DeviceRef = aion.storage_manager.DeviceRef;
pub const plan = aion.plan;
pub const tiling = aion.tiling;

// ---------------------------------------------------------------------------
// Gemma-4 E2B (text) architecture
// ---------------------------------------------------------------------------

/// Mirrors the `Model constants` block of `convert_gemma4_e2b_to_aion.py`. Kept as
/// comptime constants so a drift between converter and bench is a compile-time edit
/// in one place, not a silently wrong benchmark.
pub const G4 = struct {
    pub const num_layers: usize = 35;
    pub const embed_dim: usize = 1536;
    pub const vocab_size: usize = 262144;
    pub const num_heads: usize = 8;
    pub const num_kv_heads: usize = 1;
    pub const local_head_dim: usize = 256;
    pub const global_head_dim: usize = 512;
    pub const local_sliding_window: usize = 512;
    pub const final_logit_softcap: f32 = 30.0;
    pub const pli_dim: usize = 256;
    pub const pli_total: usize = num_layers * pli_dim; // 8960
    pub const rms_eps: f32 = 1e-6;
    pub const rope_local_base: f32 = 10_000.0;
    pub const rope_global_base: f32 = 1_000_000.0;
    pub const rope_local_proportion: f32 = 1.0;
    pub const rope_global_proportion: f32 = 0.25;

    /// MatFormer: the FFN width is elastic per layer. Hard-coding one width is a
    /// known past footgun -- layers 0-14 are 6144, layers 15-34 are 12288.
    pub const ffn_small: usize = 6144;
    pub const ffn_large: usize = 12288;

    /// Only the first 15 layers own a KV cache; the rest read a source layer's.
    pub const num_source_layers: usize = 15;
    pub const local_source_layer: usize = 13;
    pub const global_source_layer: usize = 14;

    pub fn isGlobal(layer: usize) bool {
        return (layer % 5) == 4;
    }
    pub fn headDim(layer: usize) usize {
        return if (isGlobal(layer)) global_head_dim else local_head_dim;
    }
    pub fn ffnDim(layer: usize) usize {
        return if (layer < num_source_layers) ffn_small else ffn_large;
    }
    pub fn isSource(layer: usize) bool {
        return layer < num_source_layers;
    }
    pub fn kvSourceOf(layer: usize) usize {
        if (isSource(layer)) return layer;
        return if (isGlobal(layer)) global_source_layer else local_source_layer;
    }
    /// Sliding window for the attention op: 0 means unbounded (global layers).
    pub fn slidingWindow(layer: usize) usize {
        return if (isGlobal(layer)) 0 else local_sliding_window;
    }
    pub fn ropeBase(layer: usize) f32 {
        return if (isGlobal(layer)) rope_global_base else rope_local_base;
    }
    pub fn ropeProportion(layer: usize) f32 {
        return if (isGlobal(layer)) rope_global_proportion else rope_local_proportion;
    }

    /// The four structural kinds a layer can have. A scaled-down bench must cover
    /// all of them or it is not measuring the same model.
    pub fn layerKind(layer: usize) u2 {
        const g: u2 = if (isGlobal(layer)) 1 else 0;
        const s: u2 = if (isSource(layer)) 2 else 0;
        return g | s;
    }
};

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

pub const DecodeOptions = struct {
    /// How many transformer blocks to emit. Below `num_layers` a stratified subset
    /// is chosen so every layer kind (and both FFN widths) still appears.
    layers: usize = G4.num_layers,
    /// Visible KV length, and the capacity of the global caches. Local caches are
    /// always the fixed 512-entry sliding window, as in the model.
    ctx: usize = 512,
    /// Emit the tied logits head + softcap + argmax. It is 17.7% of the decode byte
    /// stream, so leaving it out changes the roofline -- off is for isolating the
    /// transformer body, not for headline numbers.
    head: bool = true,
    /// Rows in the per-layer embedding table. The real 262144 costs 2.5 GB of VRAM
    /// but contributes one gathered row per token, so shrinking it is the one cheap
    /// way to fit a big run -- at the cost of that gather's tile count.
    pli_vocab: usize = G4.vocab_size,
    /// Query length. 1 is decode; >1 walks the same graph as prefill.
    seq: usize = 1,
    /// Replace a component with a shape-preserving no-op to measure what removing it
    /// would actually be worth.
    ///
    /// This is the only honest way to size a component of a decode step. A per-op
    /// microbenchmark measures the op in isolation, but the step is a long dependent
    /// chain where work can partly overlap, so an op's *busy time* is not its
    /// *marginal cost*. Measured proof: disabling attention split-K made an isolated
    /// dependent chain 7x slower (1.9 -> 13.9 ms/token) while the real model lost only
    /// ~1.3 ms. Ablation asks the question directly -- build the same graph without it
    /// and diff.
    ablate: Ablate = .{},
    /// Optimizer passes. Exposed so the bench can size what a pass costs as well
    /// as what it saves: horizontal MatMul fusion buys one wide GEMV but pays 150
    /// SliceND dispatches per token, and at decode a dispatch is ~8-20 us of
    /// critical path whatever it does.
    opt: aion.program.OptPolicy = .{},
};

pub const Ablate = struct {
    /// Attention -> relu(q). Same shape (D_v == D_k here), so the graph is otherwise
    /// untouched; the KV caches simply stop being read.
    attn: bool = false,
    /// Every RMSNorm -> relu: keeps the dispatch, drops only the norm's arithmetic.
    /// Measures how much the norm COMPUTE costs.
    norms_compute: bool = false,
    /// Every RMSNorm removed outright. Measures dispatch + compute, i.e. the ceiling
    /// for fusing norms into their consumer.
    ///
    /// The two are separate because they answer different questions and the answers
    /// differ by an order of magnitude: on the serial residual path a dispatch costs
    /// ~24 us whatever it computes, so replacing an op with a cheaper op saves almost
    /// nothing while REMOVING it saves the whole 24 us.
    norms: bool = false,
    /// Every broadcast scalar multiply -> identity (the `* skip_scale`, the
    /// `* sqrt(dim)` embedding scales, and the softcap's two scalings).
    scales: bool = false,
    /// Drop both RoPE applications (q and k).
    rope: bool = false,
    /// Drop the f16 cast and the cache append on source layers. Attention then reads
    /// a cache nothing wrote this step, which is fine for timing.
    kv_write: bool = false,
    /// Drop the whole per-layer-input branch: gate matmul, slice, gelu*mul, projection,
    /// norm and residual. NOTE this also removes 2 weights per layer, so
    /// `stream_bytes` falls -- it sizes the PLI machinery, not just its dispatches.
    pli: bool = false,

    pub fn any(self: Ablate) bool {
        return self.attn or self.norms or self.norms_compute or self.scales or self.rope or self.kv_write or self.pli;
    }
};

/// What the built program is expected to move, so the bench can report GB/s against
/// the bytes the model actually streams rather than against a guess.
pub const DecodeStats = struct {
    /// q8_0 bytes of weight read once per token (the roofline denominator).
    stream_bytes: u64 = 0,
    /// Bytes resident on the device (weights + tables + caches), i.e. footprint.
    resident_bytes: u64 = 0,
    layers_emitted: usize = 0,
    weight_tensors: usize = 0,
};

pub const Built = struct {
    prog: aion.program.Program,
    out: TensorId,
    stats: DecodeStats,
};

// ---------------------------------------------------------------------------
// Synthetic weights
// ---------------------------------------------------------------------------

/// Bytes a q8_0 tensor of `elems` logical elements occupies when packed.
fn q8Bytes(elems: usize) u64 {
    const info = aion.types.DType.q8_0.info();
    return @as(u64, elems / info.block_elems) * info.block_bytes;
}

/// Fill a tensor's backing bytes with a valid, cheap q8_0 pattern, tile by tile.
///
/// A q8_0 block is an opaque 34-byte unit (f16 scale + 32 i8) and each tile is
/// itself a packed-quant buffer, so restarting the pattern at every tile offset
/// keeps every block well-formed. Writing `data` directly is deliberate: the packed
/// path would need a staging buffer the size of the tensor, and one of these tensors
/// is 2.5 GB. Contents are arbitrary but finite -- the scale is a normal f16, so no
/// kernel ever sees NaN or Inf.
fn fillQuantPattern(mgr: *StorageManager, id: TensorId, seed: usize) !void {
    const t = try mgr.getMut(id);
    var block: [34]u8 = undefined;
    std.mem.writeInt(u16, block[0..2], @bitCast(@as(f16, 0.004)), .little);
    for (block[2..], 0..) |*b, i| {
        const q: i32 = @intCast((i * 7 + seed * 13) % 127);
        b.* = @bitCast(@as(i8, @intCast(q - 63)));
    }
    for (t.tile_offsets, t.tile_lens) |off, len| {
        var w: usize = 0;
        while (w < len) : (w += block.len) {
            const n = @min(block.len, len - w);
            @memcpy(t.data[off + w ..][0..n], block[0..n]);
        }
    }
}

/// Fill an f32 tensor in place with a bounded pattern (norm gammas, scalars).
fn fillF32Pattern(mgr: *StorageManager, id: TensorId, seed: usize) !void {
    const t = try mgr.getMut(id);
    for (t.tile_offsets, t.tile_lens) |off, len| {
        const words: []align(1) f32 = @alignCast(std.mem.bytesAsSlice(f32, t.data[off..][0..len]));
        for (words, 0..) |*v, i| {
            const k: usize = (i * 2654435761 + seed * 97) % 1000;
            v.* = 0.5 + (@as(f32, @floatFromInt(k)) - 500.0) * 0.0004;
        }
    }
}

/// Builder state shared by every weight/input helper: tracks footprint and pushes
/// each tensor to the device as soon as it is filled, so host peak stays at one
/// tensor rather than the whole ~5 GB model.
const Ctx = struct {
    alloc: std.mem.Allocator,
    mgr: *StorageManager,
    g: *Graph,
    policy: plan.TilePolicy,
    target: DeviceRef,
    dev: ?aion.device_memory.DeviceMemory,
    stats: *DecodeStats,
    ablate: Ablate = .{},
    seed: usize = 0,

    fn nextSeed(self: *Ctx) usize {
        self.seed += 1;
        return self.seed;
    }

    /// Move a just-filled tensor to the benchmark's target device. Placement later
    /// skips anything already there, so this only relocates the peak, never the
    /// result.
    fn place(self: *Ctx, id: TensorId) !void {
        const dev = self.dev orelse return;
        if (self.target.kind == .cpu) return;
        const tile_shape = blk: {
            const t = try self.mgr.getConst(id);
            break :blk try self.alloc.dupe(usize, t.tile_shape);
        };
        defer self.alloc.free(tile_shape);
        try self.mgr.moveTensor(id, self.target, dev, tile_shape, self.policy.tile_alignment);
    }

    /// A q8_0 matmul-B weight `[k, n]`, blocked along K, tiled the way the model
    /// loader tiles one (`chooseQuantMatMulBTiles`), given its OWN storage.
    fn weight(self: *Ctx, k: usize, n: usize) !ValueId {
        const tile = tiling.chooseQuantMatMulBTiles(self.policy, k, n, .q8_0);
        const id = try self.mgr.createTiledTensor(.q8_0, &.{ k, n }, &tile, .{
            .tile_alignment = self.policy.tile_alignment,
            .quant_axis = 0,
        });
        try fillQuantPattern(self.mgr, id, self.nextSeed());
        const v = try self.g.addInput(.q8_0, &.{ k, n });
        try self.g.bindExternal(v, id);
        self.stats.stream_bytes += q8Bytes(k * n);
        self.stats.resident_bytes += q8Bytes(k * n);
        self.stats.weight_tensors += 1;
        try self.place(id);
        return v;
    }

    /// A q8_0 embedding table `[v_rows, d]`, per-row quantized. `streamed` says
    /// whether its bytes belong in the roofline: the token table does (it is the
    /// tied logits head), the per-layer table does not (one gathered row per token).
    fn table(self: *Ctx, v_rows: usize, d: usize, streamed: bool) !ValueId {
        const tile = tiling.chooseQuantEmbeddingTableTiles(self.policy, .q8_0, v_rows, d);
        const id = try self.mgr.createTiledTensor(.q8_0, &.{ v_rows, d }, &tile, .{
            .tile_alignment = self.policy.tile_alignment,
            .quant_axis = 1,
        });
        try fillQuantPattern(self.mgr, id, self.nextSeed());
        const val = try self.g.addInput(.q8_0, &.{ v_rows, d });
        try self.g.bindExternal(val, id);
        if (streamed) self.stats.stream_bytes += q8Bytes(v_rows * d);
        self.stats.resident_bytes += q8Bytes(v_rows * d);
        self.stats.weight_tensors += 1;
        try self.place(id);
        return val;
    }

    /// An f32 vector input (norm gamma / beta).
    fn vec(self: *Ctx, n: usize) !ValueId {
        const t1 = plan.chooseTileShape1D(self.policy, n);
        const id = try self.mgr.createTiledTensor(.f32, &.{n}, &t1, .{ .tile_alignment = self.policy.tile_alignment });
        try fillF32Pattern(self.mgr, id, self.nextSeed());
        const v = try self.g.addInput(.f32, &.{n});
        try self.g.bindExternal(v, id);
        self.stats.resident_bytes += @as(u64, n) * 4;
        try self.place(id);
        return v;
    }

    /// A one-element f32 parameter used as a broadcast scalar. Aion has no
    /// scalar-typed operand; the converter relies on this same size-1 trick.
    fn scalar(self: *Ctx, value: f32) !ValueId {
        const id = try self.mgr.createTiledTensor(.f32, &.{1}, &.{1}, .{ .tile_alignment = self.policy.tile_alignment });
        {
            const t = try self.mgr.getMut(id);
            std.mem.writeInt(u32, t.data[t.tile_offsets[0]..][0..4], @bitCast(value), .little);
        }
        const v = try self.g.addInput(.f32, &.{1});
        try self.g.bindExternal(v, id);
        try self.place(id);
        return v;
    }

    fn scaled(self: *Ctx, x: ValueId, factor: f32) !ValueId {
        if (self.ablate.scales) return x;
        return self.g.addElemwiseBinary(.mul, x, try self.scalar(factor));
    }

    /// RMSNorm over the last axis of width `n`.
    fn rms(self: *Ctx, x: ValueId, n: usize) !ValueId {
        if (self.ablate.norms) return x;
        if (self.ablate.norms_compute) return self.g.addUnary(.relu, x);
        const gamma = try self.vec(n);
        const beta = try self.vec(n);
        return self.g.addRMSNorm(x, gamma, beta, G4.rms_eps, &.{n});
    }

    fn i32In(self: *Ctx, shape: []const usize, vals: []const i32) !ValueId {
        const id = try self.mgr.createTiledTensor(.i32, shape, shape, .{ .tile_alignment = self.policy.tile_alignment });
        try self.mgr.writeFromPackedScalar(id, std.mem.sliceAsBytes(vals));
        const v = try self.g.addInput(.i32, shape);
        try self.g.bindExternal(v, id);
        try self.place(id);
        return v;
    }

    /// A zeroed f16 KV cache `[1, t_len, kv_heads, head_dim]` (time is dim 1).
    fn cache(self: *Ctx, t_len: usize, head_dim: usize) !ValueId {
        const shape = [_]usize{ 1, t_len, G4.num_kv_heads, head_dim };
        const id = try self.mgr.createTiledTensor(.f16, &shape, &shape, .{ .tile_alignment = self.policy.tile_alignment });
        try self.mgr.zeroTensorData(id);
        const v = try self.g.addInput(.f16, &shape);
        try self.g.bindExternal(v, id);
        self.stats.resident_bytes += @as(u64, t_len * G4.num_kv_heads * head_dim) * 2;
        try self.place(id);
        return v;
    }
};

// ---------------------------------------------------------------------------
// Layer selection for scaled-down runs
// ---------------------------------------------------------------------------

/// Choose `want` real layer indices out of `G4.num_layers`, stratified by layer kind
/// so a reduced run still exercises source/non-source x local/global and both
/// elastic FFN widths. Returns them ascending; `want >= num_layers` returns all.
pub fn selectLayers(buf: []usize, want: usize) []usize {
    const total = G4.num_layers;
    const n = @min(want, total);
    if (n == total) {
        for (buf[0..total], 0..) |*slot, i| slot.* = i;
        return buf[0..total];
    }
    var taken: [total]bool = @splat(false);
    var chosen: usize = 0;
    // Round-robin over the four kinds, taking the lowest unused layer of each, so
    // the first four picks are guaranteed to be one of every kind.
    outer: while (chosen < n) {
        var progressed = false;
        var kind: u8 = 0;
        while (kind < 4) : (kind += 1) {
            if (chosen == n) break :outer;
            var layer: usize = 0;
            while (layer < total) : (layer += 1) {
                if (taken[layer] or @as(u8, G4.layerKind(layer)) != kind) continue;
                taken[layer] = true;
                buf[chosen] = layer;
                chosen += 1;
                progressed = true;
                break;
            }
        }
        if (!progressed) break;
    }
    std.mem.sort(usize, buf[0..chosen], {}, std.sort.asc(usize));
    return buf[0..chosen];
}

// ---------------------------------------------------------------------------
// The decode graph
// ---------------------------------------------------------------------------

/// Build one Gemma-4 E2B decode step. Op-for-op the same sequence as the
/// converter's `_emit_forward`, compiled with `policy` through the real optimizer
/// pipeline. `dev` (with a non-cpu `target`) streams each weight to the device as it
/// is built; pass null to keep everything on the host for the CPU bench.
pub fn gemma4E2BDecode(
    alloc: std.mem.Allocator,
    mgr: *StorageManager,
    policy: plan.TilePolicy,
    opts: DecodeOptions,
    target: DeviceRef,
    dev: ?aion.device_memory.DeviceMemory,
) !Built {
    var g = Graph.init(alloc);
    defer g.deinit();

    var stats: DecodeStats = .{};
    var ctx: Ctx = .{
        .alloc = alloc,
        .mgr = mgr,
        .g = &g,
        .policy = policy,
        .target = target,
        .dev = dev,
        .stats = &stats,
        .ablate = opts.ablate,
    };

    const bsz: usize = 1;
    const seq: usize = opts.seq;
    const embed = G4.embed_dim;

    var layer_buf: [G4.num_layers]usize = undefined;
    const layers = selectLayers(&layer_buf, opts.layers);
    stats.layers_emitted = layers.len;

    // ---- runtime inputs ----
    const tok_vals = try alloc.alloc(i32, bsz * seq);
    defer alloc.free(tok_vals);
    for (tok_vals, 0..) |*v, i| v.* = @intCast((i * 7919 + 13) % G4.vocab_size);
    const tokens = try ctx.i32In(&.{ bsz, seq }, tok_vals);

    const pos_vals = try alloc.alloc(i32, bsz * seq);
    defer alloc.free(pos_vals);
    // Decode at the far end of the window: the realistic, worst-case cache read.
    for (pos_vals, 0..) |*v, i| v.* = @intCast(opts.ctx - seq + i);
    const positions = try ctx.i32In(&.{ bsz, seq }, pos_vals);

    const write_index = try ctx.i32In(&.{bsz}, &.{@intCast(opts.ctx - seq)});
    const visible_end = try ctx.i32In(&.{bsz}, &.{@intCast(opts.ctx)});

    // ---- embedding + per-layer-input encoder ----
    // The token table is tied to the output head: bound ONCE and reused below, so
    // its 428 MB is counted once, exactly as the model streams it.
    const embed_table = try ctx.table(G4.vocab_size, embed, opts.head);
    const emb = try g.addGather(embed_table, tokens, 0, 0);
    const emb_scaled = try ctx.scaled(emb, @sqrt(@as(f32, @floatFromInt(embed))));

    const pli_model_proj = try ctx.weight(embed, G4.pli_total);
    const pli_proj = try ctx.scaled(
        try g.addMatMul(emb_scaled, pli_model_proj, 1.0, 0.0),
        1.0 / @sqrt(@as(f32, @floatFromInt(embed))),
    );
    const pli_proj_4d = try g.addViewReshape(pli_proj, &.{ bsz, seq, G4.num_layers, G4.pli_dim });
    const pli_proj_norm = try ctx.rms(pli_proj_4d, G4.pli_dim);

    const pli_table = try ctx.table(opts.pli_vocab, G4.pli_total, false);
    const pli_emb = try g.addGather(pli_table, tokens, 0, 0);
    const pli_emb_s = try ctx.scaled(pli_emb, @sqrt(@as(f32, @floatFromInt(G4.pli_dim))));
    const pli_emb_4d = try g.addViewReshape(pli_emb_s, &.{ bsz, seq, G4.num_layers, G4.pli_dim });

    const pli = try ctx.scaled(
        try g.addElemwiseBinary(.add, pli_proj_norm, pli_emb_4d),
        1.0 / @sqrt(2.0),
    );

    var x = emb_scaled;

    // KV caches, one pair per source layer that some emitted layer actually reads.
    var k_cache: [G4.num_layers]?ValueId = @splat(null);
    var v_cache: [G4.num_layers]?ValueId = @splat(null);
    for (layers) |layer| {
        const src = G4.kvSourceOf(layer);
        if (k_cache[src] != null) continue;
        const hd = G4.headDim(src);
        const t_len = if (G4.isGlobal(src)) opts.ctx else G4.local_sliding_window;
        k_cache[src] = try ctx.cache(t_len, hd);
        v_cache[src] = try ctx.cache(t_len, hd);
    }

    // ---- transformer blocks ----
    for (layers) |layer| {
        const hd = G4.headDim(layer);
        const q_width = G4.num_heads * hd;
        const kv_width = G4.num_kv_heads * hd;
        const ffn = G4.ffnDim(layer);
        const src = G4.kvSourceOf(layer);

        const x_norm = try ctx.rms(x, embed);

        // Q/K/V stay separate here, as the checkpoint ships them; fusing them is
        // `fuse_horizontal_matmul`'s job and part of what this bench measures.
        var q = try g.addMatMul(x_norm, try ctx.weight(embed, q_width), 1.0, 0.0);
        q = try g.addViewReshape(q, &.{ bsz, seq, G4.num_heads, hd });
        q = try ctx.rms(q, hd);
        if (!opts.ablate.rope) q = try g.addRoPE1D(q, positions, G4.ropeBase(layer), 1.0, G4.ropeProportion(layer));

        if (G4.isSource(layer)) {
            var k = try g.addMatMul(x_norm, try ctx.weight(embed, kv_width), 1.0, 0.0);
            k = try g.addViewReshape(k, &.{ bsz, seq, G4.num_kv_heads, hd });
            k = try ctx.rms(k, hd);
            if (!opts.ablate.rope) k = try g.addRoPE1D(k, positions, G4.ropeBase(layer), 1.0, G4.ropeProportion(layer));

            var v = try g.addMatMul(x_norm, try ctx.weight(embed, kv_width), 1.0, 0.0);
            v = try g.addViewReshape(v, &.{ bsz, seq, G4.num_kv_heads, hd });
            v = try ctx.rms(v, hd); // parameterless V-norm in the model; same op shape

            if (!opts.ablate.kv_write) {
                k_cache[src] = try g.addSequenceAppend(k_cache[src].?, try g.addCast(k, .f16), write_index);
                v_cache[src] = try g.addSequenceAppend(v_cache[src].?, try g.addCast(v, .f16), write_index);
            }
        }

        var o = if (opts.ablate.attn)
            try g.addUnary(.relu, q)
        else
            try g.addAttention(
                q,
                k_cache[src].?,
                v_cache[src].?,
                positions,
                visible_end,
                1.0,
                true,
                G4.slidingWindow(layer),
                0.0,
            );
        o = try g.addViewReshape(o, &.{ bsz, seq, q_width });
        o = try g.addMatMul(o, try ctx.weight(q_width, embed), 1.0, 0.0);
        x = try g.addElemwiseBinary(.add, x, try ctx.rms(o, embed));

        // Gated MLP. gate/up split so `fuse_horizontal_matmul` sees the same pattern it
        // sees on the model, and one `gate` node for the GEGLU because that is what
        // `nn.GatedMLP` emits — the converters author the gate rather than leaving a
        // unary and a multiply for `program/fuse_steps.zig` to recover.
        const ff_in = try ctx.rms(x, embed);
        const gate = try g.addMatMul(ff_in, try ctx.weight(embed, ffn), 1.0, 0.0);
        const up = try g.addMatMul(ff_in, try ctx.weight(embed, ffn), 1.0, 0.0);
        const act = try g.addGate(.gelu, gate, up);
        const ff = try g.addMatMul(act, try ctx.weight(ffn, embed), 1.0, 0.0);
        x = try g.addElemwiseBinary(.add, x, try ctx.rms(ff, embed));

        // Per-layer input: gate against this layer's slice of the PLI tensor.
        const pli_gate = try g.addUnary(.gelu, try g.addMatMul(x, try ctx.weight(embed, G4.pli_dim), 1.0, 0.0));
        const pli_slice = try g.addViewReshape(
            try g.addViewSliceND(pli, &.{ 0, 0, layer, 0 }, &.{ bsz, seq, 1, G4.pli_dim }),
            &.{ bsz, seq, G4.pli_dim },
        );
        const pli_out = try g.addMatMul(
            try g.addElemwiseBinary(.mul, pli_gate, pli_slice),
            try ctx.weight(G4.pli_dim, embed),
            1.0,
            0.0,
        );
        x = try g.addElemwiseBinary(.add, x, try ctx.rms(pli_out, embed));
        x = try ctx.scaled(x, 1.0); // the per-layer `layer_scalar` skip scale
    }

    // ---- tail ----
    x = try ctx.rms(x, embed);
    var out_v = x;
    if (opts.head) {
        // Tied head: contract against the embedding table's rows, reusing the very
        // parameter the gather bound rather than a second copy of the table.
        const logits = try g.addMatMulNT(x, embed_table, 1.0, 0.0);
        const capped = try ctx.scaled(
            try g.addUnary(.tanh, try ctx.scaled(logits, 1.0 / G4.final_logit_softcap)),
            G4.final_logit_softcap,
        );
        out_v = try g.addArgMax(capped, -1);
    }
    try g.setOutputs(&.{out_v});

    const prog = try aion.program.compileGraphOpt(alloc, &g, mgr, policy, opts.opt);
    return .{ .prog = prog, .out = prog.outputs[0], .stats = stats };
}

// ---------------------------------------------------------------------------
// Attention in isolation, at the model's exact decode shapes
// ---------------------------------------------------------------------------

/// Gemma-4 has two attention shapes at decode, and they behave very differently:
/// local layers read a 512-entry window with head_dim 256, global layers read the
/// whole cache with head_dim 512.
pub const AttnKind = enum { local, global };

/// One `Attention` op alone, at the model's real decode shapes and dtypes (f32 q
/// against an f16 cache, 8 query heads over 1 KV head).
///
/// This exists because per-op GPU attribution inside the full step needs
/// `AION_PROFILE_GPU=dispatch`, which puts every dispatch in its own compute pass
/// and so inflates exactly the short kernels it is trying to measure. Timing the op
/// on its own, over many iterations, measures the kernel instead of the profiler.
/// `repeat` emits that many independent copies of the op in ONE program. This is not
/// cosmetic: a program with a single dispatch measures the harness, not the kernel --
/// there is a fixed cost of roughly 0.7 ms per `session.execute`, so a lone
/// attention dispatch times at ~750 us whether it does work or not. Repeating
/// amortizes that, and sweeping `repeat` recovers the per-op cost as the slope.
pub fn gemma4Attention(
    alloc: std.mem.Allocator,
    mgr: *StorageManager,
    policy: plan.TilePolicy,
    kind: AttnKind,
    ctx_len: usize,
    repeat: usize,
    target: DeviceRef,
    dev: ?aion.device_memory.DeviceMemory,
) !Built {
    var g = Graph.init(alloc);
    defer g.deinit();

    var stats: DecodeStats = .{};
    var ctx: Ctx = .{
        .alloc = alloc,
        .mgr = mgr,
        .g = &g,
        .policy = policy,
        .target = target,
        .dev = dev,
        .stats = &stats,
    };

    const head_dim: usize = if (kind == .global) G4.global_head_dim else G4.local_head_dim;
    const t_len: usize = if (kind == .global) ctx_len else G4.local_sliding_window;
    const window: usize = if (kind == .global) 0 else G4.local_sliding_window;

    const q_shape = [_]usize{ 1, 1, G4.num_heads, head_dim };
    const q_id = try mgr.createTiledTensor(.f32, &q_shape, &q_shape, .{ .tile_alignment = policy.tile_alignment });
    try fillF32Pattern(mgr, q_id, 1);
    const q = try g.addInput(.f32, &q_shape);
    try g.bindExternal(q, q_id);
    try ctx.place(q_id);

    const positions = try ctx.i32In(&.{ 1, 1 }, &.{@intCast(t_len - 1)});
    const kv_len = try ctx.i32In(&.{1}, &.{@intCast(t_len)});

    // The copies form a DEPENDENT chain: each one's q is the previous one's output
    // (attention returns [B, L, H_q, D_v], the same shape q came in as, so this
    // chains with no reshape).
    //
    // This is load-bearing. Independent copies let the GPU overlap them, which
    // measures throughput -- and then a config that overlaps well but has poor
    // latency wins the benchmark and loses on the model. Measured: with split-K
    // disabled, independent copies read 6x FASTER (0.31 vs 1.88 ms/token) while the
    // real model got 8% SLOWER. The model's 35 attentions sit on a dependent
    // critical path separated by other work, so per-op latency is what it pays.
    const n_rep = @max(@as(usize, 1), repeat);
    var cur = q;
    for (0..n_rep) |_| {
        // A fresh cache pair per copy: sharing one would let the L2 serve every
        // repeat after the first and turn this into a cache benchmark.
        const k = try ctx.cache(t_len, head_dim);
        const v = try ctx.cache(t_len, head_dim);
        cur = try g.addAttention(cur, k, v, positions, kv_len, 1.0, true, window, 0.0);
    }
    try g.setOutputs(&.{cur});

    // Bytes the op must read: the visible span of both caches, f16, per copy.
    const visible: usize = if (kind == .global) t_len else @min(t_len, G4.local_sliding_window);
    stats.stream_bytes = @as(u64, 2 * visible * G4.num_kv_heads * head_dim) * 2 * n_rep;

    const prog = try aion.program.compileGraph(alloc, &g, mgr, policy);
    return .{ .prog = prog, .out = prog.outputs[0], .stats = stats };
}

/// How many of each attention kind a full decode step performs: global layers are
/// every 5th (l % 5 == 4), so 7 of 35, and the other 28 are local.
pub fn attnKindCount(kind: AttnKind) usize {
    var n: usize = 0;
    for (0..G4.num_layers) |l| {
        const is_global = G4.isGlobal(l);
        if ((kind == .global) == is_global) n += 1;
    }
    return n;
}

// ---------------------------------------------------------------------------
// Step histogram (the fidelity gate)
// ---------------------------------------------------------------------------

/// Count lowered steps by tag. Diffing this against the real model's histogram is
/// what makes the synthetic graph trustworthy: if they disagree, the bench is wrong.
pub fn stepHistogram(prog: *const aion.program.Program, counts: *std.StringHashMap(usize)) !void {
    for (prog.steps) |step| {
        const gop = try counts.getOrPut(@tagName(step.op));
        if (!gop.found_existing) gop.value_ptr.* = 0;
        gop.value_ptr.* += 1;
    }
}

/// The step histogram of the REAL `models/gemma/gemma4_e2b_q8.aion` compiled for
/// WebGPU at seq=1, captured 2026-08-19 from
/// `AION_PROFILE=summary AION_PROFILE_GPU=pass` on
/// `bindings/python/examples/gemma4_e2b_generate_one.py --device gpu` (the CPU
/// per-step spans are exact step counts).
///
/// The synthetic graph reproduced every one of these on the first attempt, which is
/// what licenses using it as a stand-in for the model. Keep it that way: if this
/// check fails, the bench has stopped measuring the model and any number it reports
/// is about something else. A deliberate change to the converter or the optimizer
/// updates the fixture -- after re-capturing it, not by loosening the check.
pub const gemma4_e2b_expected_steps = [_]struct { name: []const u8, count: usize }{
    .{ .name = "MatMulTiled", .count = 211 },
    .{ .name = "SliceNDScalar", .count = 150 },
    // All 242 norms, of which 106 carry a residual: the "sandwich norm" pairs
    // (`x = x + norm(f(x))`) that `program/fuse_steps.zig` folds into the norm step.
    //
    // NOTE this histogram is keyed on step TAG, and the residual is a FIELD, so these
    // counts can no longer tell a fused norm from a plain one — if the fusion stopped
    // firing, only `ElemwiseBinaryTiled` below would move. Checking the property instead
    // of the name belongs with the bench rework (this file and the CPU bench into one
    // bench/ directory), not here.
    .{ .name = "RMSNormTiled", .count = 242 },
    .{ .name = "ElemwiseBinaryTiled", .count = 111 },
    // No ReshapeScalar or ReTileCopyScalar at all any more. They were 137 and 353 at
    // the start of 2026-08-19; three changes removed them, and each removal was a
    // pointless copy rather than real work:
    //   * the M-hint tile fix in `program/compiler.zig` (a 4096-row tile stamped on a
    //     1-row decode activation made every neighbour disagree),
    //   * `program/alias_views.zig` (copies between byte-identical layouts),
    //   * the quantized tile cap in `plan.zig` (a 4096 N/K cap split wide weights and
    //     left the split pieces tiled unlike their neighbours).
    // Re-captured from the model after each; text output identical throughout.
    // The 70 GEGLU gates are `ElemwiseBinaryTiled{ .op = .gate }` now, not a tag of
    // their own, so they land in the ElemwiseBinaryTiled row above (41 + 70 = 111).
    .{ .name = "RoPE1DTiled", .count = 50 },
    .{ .name = "AttentionTiled", .count = 35 },
    .{ .name = "CastTiled", .count = 30 },
    .{ .name = "SequenceAppendTiled", .count = 30 },
    .{ .name = "GatherRowsTiled", .count = 2 },
    .{ .name = "ArgMax", .count = 1 },
    .{ .name = "MatMulNTTiled", .count = 1 },
    .{ .name = "UnaryTiled", .count = 1 },
};

/// Compare a built program against `gemma4_e2b_expected_steps`. Returns true on an
/// exact match; otherwise prints every disagreement and returns false. Only valid
/// for a full-scale build (all 35 layers, head on, full vocab) -- any reduced knob
/// legitimately changes the counts, so the caller checks that first.
pub fn verifyGemma4Steps(alloc: std.mem.Allocator, prog: *const aion.program.Program) !bool {
    var counts: std.StringHashMap(usize) = .init(alloc);
    defer counts.deinit();
    try stepHistogram(prog, &counts);

    var ok = true;
    var expected_total: usize = 0;
    for (gemma4_e2b_expected_steps) |e| {
        expected_total += e.count;
        const got = counts.get(e.name) orelse 0;
        if (got != e.count) {
            std.debug.print("    MISMATCH {s:<24} expected {d}, got {d}\n", .{ e.name, e.count, got });
            ok = false;
        }
    }
    // Also catch steps the model does NOT have: a new tag is just as much a drift.
    var it = counts.iterator();
    while (it.next()) |entry| {
        var known = false;
        for (gemma4_e2b_expected_steps) |e| {
            if (std.mem.eql(u8, e.name, entry.key_ptr.*)) {
                known = true;
                break;
            }
        }
        if (!known) {
            std.debug.print("    UNEXPECTED {s:<24} x{d} (not in the model)\n", .{ entry.key_ptr.*, entry.value_ptr.* });
            ok = false;
        }
    }
    if (prog.steps.len != expected_total) {
        std.debug.print("    MISMATCH total steps expected {d}, got {d}\n", .{ expected_total, prog.steps.len });
        ok = false;
    }
    return ok;
}

/// Print the lowered steps in execution order. Counts say what is expensive; the
/// ORDER says why it exists -- a retile only makes sense next to the ops whose tiling
/// disagreed. Run with `--layers 1` to get one readable layer.
pub fn printStepSequence(mgr: *StorageManager, prog: *const aion.program.Program, limit: usize) void {
    std.debug.print("  step sequence (first {d} of {d}):\n", .{ @min(limit, prog.steps.len), prog.steps.len });
    for (prog.steps[0..@min(limit, prog.steps.len)], 0..) |step, i| {
        std.debug.print("    {d:>4}  {s:<22}", .{ i, @tagName(step.op) });
        // For the view family, the shapes and TILINGS are the whole story: a retile
        // exists only because two neighbours disagreed about tiling, and the printout
        // is how you see which pair.
        switch (step.op) {
            .ReTileCopyScalar => |s| printPair(mgr, s.src, s.dst),
            .ReshapeScalar => |s| printPair(mgr, s.src, s.dst),
            .SliceNDScalar => |s| printPair(mgr, s.src, s.dst),
            else => {},
        }
        std.debug.print("\n", .{});
    }
}

fn printPair(mgr: *StorageManager, src: TensorId, dst: TensorId) void {
    printOne(mgr, "src", src);
    std.debug.print(" -> ", .{});
    printOne(mgr, "dst", dst);
}

fn printOne(mgr: *StorageManager, label: []const u8, id: TensorId) void {
    const t = mgr.getConst(id) catch {
        std.debug.print("{s}=?", .{label});
        return;
    };
    std.debug.print("{s} shape[", .{label});
    for (t.shape, 0..) |d, i| std.debug.print("{s}{d}", .{ if (i == 0) "" else ",", d });
    std.debug.print("] tile[", .{});
    for (t.tile_shape, 0..) |d, i| std.debug.print("{s}{d}", .{ if (i == 0) "" else ",", d });
    std.debug.print("]", .{});
}

/// Count view steps whose source and destination have **byte-identical physical
/// layout** — same dtype, same tile byte-lengths, same offsets. Such a step copies a
/// buffer onto an identically-shaped buffer, so it does no work at all; it exists only
/// because the two values are different graph values and therefore different tensors.
///
/// This sizes the aliasing opportunity before any is built. Note the fix is NOT to
/// reuse the source tensor id: consumers read logical shape from tensor metadata, and
/// `[1,1,8,256]` and `[1,1,2048]` disagree there. It is to give the two ids one
/// backing, which the workspace planner's slot mechanism already expresses.
pub fn countNoopViews(mgr: *StorageManager, prog: *const aion.program.Program) struct { noop: usize, total: usize } {
    var noop: usize = 0;
    var total: usize = 0;
    for (prog.steps) |step| {
        const pair: ?[2]TensorId = switch (step.op) {
            .ReshapeScalar => |s| .{ s.src, s.dst },
            .ReTileCopyScalar => |s| .{ s.src, s.dst },
            else => null,
        };
        const p = pair orelse continue;
        total += 1;
        if (layoutsIdentical(mgr, p[0], p[1])) noop += 1;
    }
    return .{ .noop = noop, .total = total };
}

/// True when two tensors occupy the same bytes in the same arrangement, so a copy
/// between them is a no-op. Per-tile lengths must match one-for-one: on the device
/// each tile is its own buffer, so equal tiling means a tile-for-tile duplicate.
pub fn layoutsIdentical(mgr: *StorageManager, a_id: TensorId, b_id: TensorId) bool {
    const a = mgr.getConst(a_id) catch return false;
    const b = mgr.getConst(b_id) catch return false;
    if (a.dtype != b.dtype) return false;
    if (a.tile_lens.len != b.tile_lens.len) return false;
    for (a.tile_lens, b.tile_lens) |x, y| if (x != y) return false;
    for (a.tile_offsets, b.tile_offsets) |x, y| if (x != y) return false;
    return true;
}

/// Print the histogram, densest first, plus the total. Formatted to be diffed by eye
/// against the operation table `AION_PROFILE=summary` prints for the real model.
pub fn printStepHistogram(alloc: std.mem.Allocator, prog: *const aion.program.Program) !void {
    var counts: std.StringHashMap(usize) = .init(alloc);
    defer counts.deinit();
    try stepHistogram(prog, &counts);

    const Row = struct { name: []const u8, n: usize };
    var rows: std.ArrayList(Row) = .empty;
    defer rows.deinit(alloc);
    var it = counts.iterator();
    while (it.next()) |e| try rows.append(alloc, .{ .name = e.key_ptr.*, .n = e.value_ptr.* });
    std.mem.sort(Row, rows.items, {}, struct {
        fn lt(_: void, a: Row, b: Row) bool {
            if (a.n != b.n) return a.n > b.n;
            return std.mem.lessThan(u8, a.name, b.name);
        }
    }.lt);

    std.debug.print("  steps: {d} total\n", .{prog.steps.len});
    for (rows.items) |r| std.debug.print("    {s:<28} x{d}\n", .{ r.name, r.n });
}
