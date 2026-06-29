// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Horizontal MatMul fusion.
//!
//! Parallel projections off a shared input — attention Q/K/V, MoE gate/up,
//! multi-head readouts — are emitted as separate `MatMul`s by converters. Each
//! pays its own dispatch / tiling / weight-pack overhead. This pass fuses any
//! group of `MatMul`s that share the same `A` operand and whose `B` operands are
//! bound, identically-shaped (in K) quantized weights into:
//!
//!   1. one wide `MatMul` over the column-concatenated weight `[..,K,ΣNᵢ]`, plus
//!   2. one cheap `ViewSliceND` per original output recovering `[..,Nᵢ]`.
//!
//! The transform is numerically identical (same weights, same math) and general:
//! it keys off graph structure, not model identity.
//!
//! Runs AFTER shape inference inside `compileGraph` (so original matmul output
//! shapes are known and slice lengths are concrete) but BEFORE the value→tensor
//! map is sized, so the values it appends are compiled normally.

const std = @import("std");

const graph_mod = @import("../graph.zig");
const plan = @import("../plan.zig");
const manager_mod = @import("../../storage/manager.zig");
const types = @import("../../backend/types.zig");

const Graph = graph_mod.Graph;
const Node = graph_mod.Node;
const ValueId = graph_mod.ValueId;
const StorageManager = manager_mod.StorageManager;
const TensorId = manager_mod.TensorId;
const TilePolicy = plan.TilePolicy;
const DType = types.DType;

pub const Error = graph_mod.GraphError || manager_mod.StorageError;

const MAX_RANK: usize = 8;

/// This pass's discriminator for `StorageManager.derived_weight_cache`. Must be
/// unique across passes — see the registry in `optimize.zig`.
const DERIVED_KIND: u32 = @intFromEnum(@import("../optimize.zig").DerivedWeightKind.horizontal_matmul_concat);

/// A fusable matmul: `out = A @ B` where B is a bound quantized weight.
const Cand = struct {
    node_idx: usize,
    a_id: ValueId,
    out_id: ValueId,
    b_tid: TensorId,
    alpha: f32,
    beta: f32,
    dtype: DType,
    rank: usize,
    k: usize,
    n: usize,
    used: bool = false,
};

pub fn run(allocator: std.mem.Allocator, graph: *Graph, mgr: *StorageManager, policy: TilePolicy) Error!void {
    // 1. Collect candidates: top-level `A @ B` matmuls with a bound quantized B.
    var cands: std.ArrayList(Cand) = .empty;
    defer cands.deinit(allocator);

    for (graph.nodes.items, 0..) |node, idx| {
        const mm = switch (node.op) {
            .MatMul => |m| m,
            else => continue,
        };
        if (node.inputs.len != 2) continue;
        const b_v = graph.values.items[@intCast(node.inputs[1])];
        const b_ext = b_v.external orelse continue;
        const b_dt = b_v.dtype orelse continue;
        if (!b_dt.info().is_quantized) continue;
        const rank = b_v.shape.len;
        if (rank < 2) continue;

        try cands.append(allocator, .{
            .node_idx = idx,
            .a_id = node.inputs[0],
            .out_id = node.output,
            .b_tid = @intCast(b_ext),
            .alpha = mm.alpha,
            .beta = mm.beta,
            .dtype = b_dt,
            .rank = rank,
            .k = b_v.shape[rank - 2],
            .n = b_v.shape[rank - 1],
        });
    }
    if (cands.items.len < 2) return;

    // Per-node bookkeeping for the rebuild.
    const node_count = graph.nodes.items.len;
    const drop: []bool = try allocator.alloc(bool, node_count);
    defer allocator.free(drop);
    @memset(drop, false);

    // For each fused set, the spliced-in replacement nodes, keyed by the set's
    // first (anchor) node index.
    var anchor_idx: std.ArrayList(usize) = .empty;
    defer anchor_idx.deinit(allocator);
    var anchor_nodes: std.ArrayList([]Node) = .empty;
    defer {
        for (anchor_nodes.items) |ns| allocator.free(ns);
        anchor_nodes.deinit(allocator);
    }

    // old output ValueId -> replacement slice ValueId
    var remap = std.AutoHashMap(ValueId, ValueId).init(allocator);
    defer remap.deinit();

    var any_fused = false;

    // 2. Group candidates that share (A, dtype, K, rank, leading dims, alpha, beta).
    var i: usize = 0;
    while (i < cands.items.len) : (i += 1) {
        if (cands.items[i].used) continue;
        const head = cands.items[i];
        cands.items[i].used = true;

        var set: std.ArrayList(Cand) = .empty;
        defer set.deinit(allocator);
        try set.append(allocator, head);

        var j: usize = i + 1;
        while (j < cands.items.len) : (j += 1) {
            if (cands.items[j].used) continue;
            if (fusable(graph, head, cands.items[j])) {
                cands.items[j].used = true;
                try set.append(allocator, cands.items[j]);
            }
        }
        if (set.items.len < 2) continue;

        try fuseSet(allocator, graph, mgr, policy, set.items, drop, &anchor_idx, &anchor_nodes, &remap);
        any_fused = true;
    }

    if (!any_fused) return;

    // 3. Rebuild the node list: drop fused matmuls, splice replacements at the
    //    anchor, and rewrite every surviving consumer + graph output via `remap`.
    var new_nodes: std.ArrayList(Node) = .empty;
    errdefer new_nodes.deinit(allocator);

    for (graph.nodes.items, 0..) |node, idx| {
        if (drop[idx]) {
            // If this is a set anchor, emit the replacement nodes in its place.
            for (anchor_idx.items, 0..) |a, s| {
                if (a == idx) {
                    for (anchor_nodes.items[s]) |rn| {
                        remapInputs(rn, &remap);
                        try new_nodes.append(allocator, rn);
                    }
                }
            }
            continue;
        }
        remapInputs(node, &remap);
        try new_nodes.append(allocator, node);
    }

    graph.nodes.deinit(allocator);
    graph.nodes = new_nodes;

    for (graph.outputs.items) |*oid| {
        if (remap.get(oid.*)) |new_id| oid.* = new_id;
    }
}

/// Two candidates fuse if they share input A and produce compatible-to-concat
/// weights (same dtype, K, rank, leading dims) under identical matmul scaling.
fn fusable(graph: *const Graph, a: Cand, b: Cand) bool {
    if (a.a_id != b.a_id) return false;
    if (a.dtype != b.dtype) return false;
    if (a.k != b.k or a.rank != b.rank) return false;
    if (a.alpha != b.alpha or a.beta != b.beta) return false;
    // Leading (batch) dims of the two weights must match block-for-block.
    const sa = graph.values.items[@intCast(a.out_id)].shape; // [.., M, Na]
    const sb = graph.values.items[@intCast(b.out_id)].shape; // [.., M, Nb]
    if (sa.len != sb.len) return false;
    for (sa[0 .. sa.len - 1], sb[0 .. sb.len - 1]) |da, db| {
        if (da != db) return false;
    }
    return true;
}

fn fuseSet(
    allocator: std.mem.Allocator,
    graph: *Graph,
    mgr: *StorageManager,
    policy: TilePolicy,
    set: []const Cand,
    drop: []bool,
    anchor_idx: *std.ArrayList(usize),
    anchor_nodes: *std.ArrayList([]Node),
    remap: *std.AutoHashMap(ValueId, ValueId),
) Error!void {
    const head = set[0];
    const rank = head.rank;

    // Concatenate the weights along N (cached on the store across recompiles).
    const sources: []TensorId = try allocator.alloc(TensorId, set.len);
    defer allocator.free(sources);
    var sum_n: usize = 0;
    for (set, 0..) |c, idx| {
        sources[idx] = c.b_tid;
        sum_n += c.n;
    }
    const w_cat: TensorId = try concatQuantMatmulB(allocator, mgr, policy, sources);

    // Cat-weight value: leaf bound to the fused tensor; shape [.., K, ΣN].
    const w_t = try mgr.getConst(w_cat);
    const w_vid = try newValue(graph, head.dtype, w_t.shape, @intCast(w_cat));

    // Cat matmul output: same as any member output but with last dim = ΣN.
    const head_out = graph.values.items[@intCast(head.out_id)];
    var cat_shape: [MAX_RANK]usize = undefined;
    @memcpy(cat_shape[0..rank], head_out.shape);
    cat_shape[rank - 1] = sum_n;
    const cat_vid = try newValue(graph, head_out.dtype.?, cat_shape[0..rank], null);

    // Replacement nodes: one wide matmul + one slice per original output.
    const nodes = try allocator.alloc(Node, 1 + set.len);
    errdefer allocator.free(nodes);

    nodes[0] = .{
        .op = .{ .MatMul = .{ .alpha = head.alpha, .beta = head.beta } },
        .inputs = try arenaIds(graph, &[_]ValueId{ head.a_id, w_vid }),
        .output = cat_vid,
    };

    var offset: usize = 0;
    for (set, 0..) |c, idx| {
        const out_v = graph.values.items[@intCast(c.out_id)];
        const slice_vid = try newValue(graph, out_v.dtype.?, out_v.shape, null);

        var starts: [MAX_RANK]usize = .{0} ** MAX_RANK;
        starts[rank - 1] = offset;
        nodes[idx + 1] = .{
            .op = .{ .ViewSliceND = .{
                .starts = try arenaUsize(graph, starts[0..rank]),
                .lens = try arenaUsize(graph, out_v.shape),
            } },
            .inputs = try arenaIds(graph, &[_]ValueId{cat_vid}),
            .output = slice_vid,
        };

        try remap.put(c.out_id, slice_vid);
        drop[c.node_idx] = true;
        offset += c.n;
    }

    try anchor_idx.append(allocator, head.node_idx);
    try anchor_nodes.append(allocator, nodes);
}

/// Concatenate quantized matmul-B weights `[.., K, Nᵢ]` along N into `[.., K, ΣN]`.
///
/// The packed-quant layout is block-space row-major (`[.., K/blk, N]`, each block
/// a self-contained `block_bytes` unit), so concatenation along N is a per-block-row
/// append of each source's `Nᵢ` blocks — no requantization, scales preserved, and
/// dtype-agnostic across q8_0 / q4_0. Result is cached on the store by source-id set.
pub fn concatQuantMatmulB(
    allocator: std.mem.Allocator,
    mgr: *StorageManager,
    policy: TilePolicy,
    sources: []const TensorId,
) Error!TensorId {
    if (sources.len == 0) return Error.InvalidArgument;

    const first = try mgr.getConst(sources[0]);
    const dtype = first.dtype;
    const di = dtype.info();
    if (!di.is_quantized) return Error.InvalidArgument;
    const rank: usize = first.rank;
    if (rank < 2) return Error.InvalidArgument;

    if (mgr.lookupDerivedWeight(DERIVED_KIND, sources)) |tid| return tid;

    const k: usize = first.shape[rank - 2];
    const blk: usize = di.block_elems;
    const bb: usize = di.block_bytes;
    if (k % blk != 0) return Error.InvalidArgument;

    var lead: usize = 1;
    for (first.shape[0 .. rank - 2]) |d| lead *= d;
    const rows: usize = lead * (k / blk); // number of block-rows

    var sum_n: usize = 0;
    for (sources) |sid| {
        const t = try mgr.getConst(sid);
        if (t.dtype != dtype or @as(usize, t.rank) != rank) return Error.InvalidArgument;
        if (t.shape[rank - 2] != k) return Error.InvalidArgument;
        if (@as(usize, t.quant_axis) != rank - 2) return Error.InvalidArgument;
        for (first.shape[0 .. rank - 2], t.shape[0 .. rank - 2]) |fd, td| {
            if (fd != td) return Error.InvalidArgument;
        }
        sum_n += t.shape[rank - 1];
    }

    const out_bytes: usize = rows * sum_n * bb;
    const out_buf = allocator.alloc(u8, out_bytes) catch return Error.OutOfMemory;
    defer allocator.free(out_buf);

    var col_off: usize = 0; // blocks already placed in each row
    for (sources) |sid| {
        const t = try mgr.getConst(sid);
        const n_i: usize = t.shape[rank - 1];
        const src_bytes: usize = rows * n_i * bb;
        const src_buf = allocator.alloc(u8, src_bytes) catch return Error.OutOfMemory;
        defer allocator.free(src_buf);
        try mgr.readToPackedQuant(sid, src_buf);

        var r: usize = 0;
        while (r < rows) : (r += 1) {
            const src_off = r * n_i * bb;
            const dst_off = (r * sum_n + col_off) * bb;
            @memcpy(out_buf[dst_off .. dst_off + n_i * bb], src_buf[src_off .. src_off + n_i * bb]);
        }
        col_off += n_i;
    }

    // Tile the fused weight exactly as the MatMul lowering tiles a quant B
    // (fixed M-hint), so no (impossible) quant retile is demanded downstream.
    const m_hint: usize = @max(@as(usize, 1), policy.base_square_2d);
    const tiles = plan.chooseMatMulTiles(policy, m_hint, sum_n, k, dtype);

    var shape_buf: [MAX_RANK]usize = undefined;
    var tile_buf: [MAX_RANK]usize = undefined;
    var d: usize = 0;
    while (d + 2 < rank) : (d += 1) {
        shape_buf[d] = first.shape[d];
        tile_buf[d] = 1;
    }
    shape_buf[rank - 2] = k;
    shape_buf[rank - 1] = sum_n;
    tile_buf[rank - 2] = tiles.tk;
    tile_buf[rank - 1] = tiles.tn;

    const out_tid = try mgr.createTiledTensor(dtype, shape_buf[0..rank], tile_buf[0..rank], .{
        .tile_alignment = policy.tile_alignment,
        .quant_axis = @intCast(rank - 2),
    });
    try mgr.writeFromPackedQuant(out_tid, out_buf);
    try mgr.recordDerivedWeight(DERIVED_KIND, sources, out_tid);
    return out_tid;
}

/// Geometry of one logical source's slice within a fused (column-concatenated)
/// weight, in packed block-space: each block-row of the fused weight holds
/// `sum_n` blocks; this source occupies `[col_off, col_off+n_i)` of them.
const ColumnRegion = struct { rows: usize, sum_n: usize, col_off: usize, n_i: usize, bb: usize };

/// Resolve `ref` to its column region and validate that `external_tid` (the swap
/// source / read destination) matches the logical weight's layout block-for-block.
fn columnRegion(mgr: *StorageManager, ref: StorageManager.DerivedSourceRef, external_tid: TensorId) Error!ColumnRegion {
    if (ref.kind != DERIVED_KIND) return Error.InvalidArgument;

    const fused = try mgr.getConst(ref.result);
    const dtype = fused.dtype;
    const di = dtype.info();
    const rank: usize = fused.rank;
    if (rank < 2 or !di.is_quantized) return Error.InvalidArgument;

    const blk: usize = di.block_elems;
    const k: usize = fused.shape[rank - 2];
    var lead: usize = 1;
    for (fused.shape[0 .. rank - 2]) |d| lead *= d;

    // Column offset of this source within the fused weight = Σ N of preceding sources.
    var col_off: usize = 0;
    for (ref.sources[0..ref.index]) |sid| {
        col_off += (try mgr.getConst(sid)).shape[rank - 1];
    }
    const n_i: usize = (try mgr.getConst(ref.sources[ref.index])).shape[rank - 1];

    // The external tensor must match the logical source's layout.
    const ext = try mgr.getConst(external_tid);
    if (ext.dtype != dtype or @as(usize, ext.rank) != rank) return Error.InvalidArgument;
    if (ext.shape[rank - 2] != k or ext.shape[rank - 1] != n_i) return Error.InvalidArgument;
    for (fused.shape[0 .. rank - 2], ext.shape[0 .. rank - 2]) |fd, ed| {
        if (fd != ed) return Error.InvalidArgument;
    }

    return .{ .rows = lead * (k / blk), .sum_n = fused.shape[rank - 1], .col_off = col_off, .n_i = n_i, .bb = di.block_bytes };
}

/// Write `src`'s bytes into the column range of the fused weight that `ref`
/// designates — the inverse of `concatQuantMatmulB` for one source. This is the
/// write-through path for swapping a weight that was fused: the fused weight is
/// the canonical store (the original source buffer may be freed), so a swap lands
/// directly in its sub-region and takes effect on the next run with no recompile.
pub fn overwriteFusedColumns(
    allocator: std.mem.Allocator,
    mgr: *StorageManager,
    ref: StorageManager.DerivedSourceRef,
    src_tid: TensorId,
) Error!void {
    const reg = try columnRegion(mgr, ref, src_tid);

    const fused_buf = allocator.alloc(u8, reg.rows * reg.sum_n * reg.bb) catch return Error.OutOfMemory;
    defer allocator.free(fused_buf);
    const src_buf = allocator.alloc(u8, reg.rows * reg.n_i * reg.bb) catch return Error.OutOfMemory;
    defer allocator.free(src_buf);

    try mgr.readToPackedQuant(ref.result, fused_buf);
    try mgr.readToPackedQuant(src_tid, src_buf);

    var r: usize = 0;
    while (r < reg.rows) : (r += 1) {
        const dst_off = (r * reg.sum_n + reg.col_off) * reg.bb;
        const src_off = r * reg.n_i * reg.bb;
        @memcpy(fused_buf[dst_off .. dst_off + reg.n_i * reg.bb], src_buf[src_off .. src_off + reg.n_i * reg.bb]);
    }

    try mgr.writeFromPackedQuant(ref.result, fused_buf);
}

/// Read a fused-away weight's current bytes back out of the fused weight into
/// `dst` — the read counterpart of `overwriteFusedColumns` (gather, not scatter).
/// Mirrors `_packed_params`' unpack path: the standalone weight no longer exists,
/// so the canonical value is materialized from the fused tensor on demand.
pub fn readFusedColumns(
    allocator: std.mem.Allocator,
    mgr: *StorageManager,
    ref: StorageManager.DerivedSourceRef,
    dst_tid: TensorId,
) Error!void {
    const reg = try columnRegion(mgr, ref, dst_tid);

    const fused_buf = allocator.alloc(u8, reg.rows * reg.sum_n * reg.bb) catch return Error.OutOfMemory;
    defer allocator.free(fused_buf);
    const dst_buf = allocator.alloc(u8, reg.rows * reg.n_i * reg.bb) catch return Error.OutOfMemory;
    defer allocator.free(dst_buf);

    try mgr.readToPackedQuant(ref.result, fused_buf);

    var r: usize = 0;
    while (r < reg.rows) : (r += 1) {
        const src_off = (r * reg.sum_n + reg.col_off) * reg.bb;
        const dst_off = r * reg.n_i * reg.bb;
        @memcpy(dst_buf[dst_off .. dst_off + reg.n_i * reg.bb], fused_buf[src_off .. src_off + reg.n_i * reg.bb]);
    }

    try mgr.writeFromPackedQuant(dst_tid, dst_buf);
}

fn newValue(graph: *Graph, dtype: DType, shape: []const usize, external: ?graph_mod.ExternalId) Error!ValueId {
    const id = try graph.addValue();
    const sh = try graph.dupeShape(shape);
    graph.values.items[@intCast(id)] = .{ .dtype = dtype, .shape = sh, .external = external };
    return id;
}

fn arenaIds(graph: *Graph, ids: []const ValueId) Error![]ValueId {
    const out = graph.arenaAlloc().alloc(ValueId, ids.len) catch return Error.OutOfMemory;
    @memcpy(out, ids);
    return out;
}

fn arenaUsize(graph: *Graph, vals: []const usize) Error![]usize {
    const out = graph.arenaAlloc().alloc(usize, vals.len) catch return Error.OutOfMemory;
    @memcpy(out, vals);
    return out;
}

fn remapInputs(node: Node, remap: *std.AutoHashMap(ValueId, ValueId)) void {
    const ins: []ValueId = @constCast(node.inputs);
    for (ins) |*in| {
        if (remap.get(in.*)) |new_id| in.* = new_id;
    }
}
