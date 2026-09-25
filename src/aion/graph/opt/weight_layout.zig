// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Re-lay a q8 matmul weight into the order a decode matvec reads it in, and
//! contract against its rows instead (`MatMul` -> `MatMulNT`). Only q8: that is
//! the one dtype the CPU's NT lowering has a kernel for.
//!
//! Two changes, both pure byte permutations of the same blocks:
//!
//! 1. *Transposed.* A q8 `[K, N]` matmul-B holds its blocks block-major,
//!    `[K/32][N]`: the blocks one output column needs are `N * 34` bytes apart.
//!    A matvec reading them either strides — re-reading each tile once per
//!    column group — or gives up register-resident sums. Transposed to `[N, K]`
//!    it is one unbroken run per column and gets both. Measured on 257 MB of
//!    weights, one core: 30 GB/s block-major against 60 GB/s as `[N, K]`.
//! 2. *Grouped* (`types.QuantBlockOrder`). Rows still sit a whole row apart,
//!    so a kernel computing several walks several streams, and an integer dot
//!    gives one row per lane only if the rows' bytes share a vector. Grouping as
//!    many rows as the target's dot has lanes gives both: one stream, and the
//!    per-block scaling paid once for the group. The target names the grouping
//!    its kernel reads (`TilePolicy.quant_block_order`).
//!
//! The weight itself is unchanged: both forms block along the same reduction
//! axis in the same groups of 32 with the same scales, so nothing is requantized
//! and `test_opt.zig` checks each block arrives whole.
//!
//! Nor are the results: which kernel runs changes, but every quantized kernel
//! multiplies the same block-quantized activation, and `test_opt.zig` pins that
//! at several M with the pass and without.

const std = @import("std");
const builtin = @import("builtin");

const graph_mod = @import("../graph.zig");
const plan = @import("../plan.zig");
const manager_mod = @import("../../storage/manager.zig");
const derived = @import("../../storage/derived.zig");
const types = @import("../../backend/types.zig");
const rewriter_mod = @import("rewriter.zig");
const target_mod = @import("../target.zig");


const Graph = graph_mod.Graph;
const Node = graph_mod.Node;
const ValueId = graph_mod.ValueId;
const Rewriter = rewriter_mod.Rewriter;
const StorageManager = manager_mod.StorageManager;
const TiledTensor = @import("../../storage/storage.zig").TiledTensor;
const TensorId = manager_mod.TensorId;
const Target = target_mod.Target;

pub const Error = graph_mod.GraphError || manager_mod.StorageError;

/// A `MatMul` whose B is a stored rank-2 quantized weight, resolved from storage.
const Cand = struct {
    b_tid: TensorId,
    dtype: types.DType,
    k: usize,
    n: usize,
    alpha: f32,
    beta: f32,
    /// B is `[n, k]` already; the re-lay only has to interleave it.
    already_nt: bool = false,
};

pub const Rule = struct {
    mgr: *StorageManager,
    target: Target,

    pub fn run(self: Rule, rw: *Rewriter) Error!void {
        // A row lookup of a weight a matmul re-lays reads the re-laid copy too, so
        // the source is left unread and freed instead of kept beside it.
        var laid_rows: std.AutoHashMapUnmanaged(TensorId, void) = .empty;
        defer laid_rows.deinit(rw.gpa);
        for (rw.input()) |node| {
            const c = candidate(rw.g, self.mgr, self.target, node) orelse continue;
            if (c.already_nt) laid_rows.put(rw.gpa, c.b_tid, {}) catch return Error.OutOfMemory;
        }

        var changed = false;
        for (rw.input()) |node| {
            if (lookupTable(rw.g, node)) |table| if (laid_rows.contains(table)) {
                const t = try self.mgr.getConst(table);
                const laid = try relayout(rw.gpa, self.mgr, self.target, table);
                const weight = try rw.bound(t.dtype, t.shape, @intCast(laid));
                try rw.add(.{ .op = node.op, .inputs = try rw.ids(&.{ weight, node.inputs[1] }), .output = node.output });
                changed = true;
                continue;
            };
            const c = candidate(rw.g, self.mgr, self.target, node) orelse {
                try rw.add(node);
                continue;
            };
            const laid = relayout(rw.gpa, self.mgr, self.target, c.b_tid) catch {
                try rw.add(node);
                continue;
            };
            const weight = try rw.bound(c.dtype, &.{ c.n, c.k }, @intCast(laid));
            try rw.add(.{
                .op = .{ .MatMulNT = .{ .alpha = c.alpha, .beta = c.beta } },
                .inputs = try rw.ids(&.{ node.inputs[0], weight }),
                .output = node.output,
            });
            changed = true;
        }
        if (!changed) rw.keepAll();
    }
};

/// The stored table an embedding-shaped row lookup reads, if it reads one.
fn lookupTable(g: *const Graph, node: Node) ?TensorId {
    const gather = switch (node.op) {
        .Gather => |x| x,
        else => return null,
    };
    if (node.inputs.len != 2 or gather.batch_dims != 0) return null;
    const table = g.values.items[@intCast(node.inputs[0])];
    if (table.shape.len != 2 or (gather.axis != 0 and gather.axis != -2)) return null;
    return @intCast(table.external orelse return null);
}

fn candidate(g: *const Graph, mgr: *const StorageManager, target: Target, node: Node) ?Cand {
    // A `MatMulNT` is already contracting against rows; it still wants its
    // blocks interleaved, which is the other half of what this pass does.
    const scale: struct { alpha: f32, beta: f32 } = switch (node.op) {
        .MatMul => |m| .{ .alpha = m.alpha, .beta = m.beta },
        .MatMulNT => |m| .{ .alpha = m.alpha, .beta = m.beta },
        else => return null,
    };
    const already_nt = std.meta.activeTag(node.op) == .MatMulNT;
    if (node.inputs.len != 2) return null;
    const b = g.values.items[@intCast(node.inputs[1])];
    const ext = b.external orelse return null;
    const b_tid: TensorId = @intCast(ext);

    const t = mgr.getConst(b_tid) catch return null;
    // The NT lowering has a q8 kernel only; any other quantized B would lower to
    // an op the backend cannot run.
    if (t.dtype != .q8_0) return null;
    const info = t.dtype.info();
    // The NT lowering takes a rank-2 B only, and reads its rows, so a batched
    // weight has no `[n, k]` form to move to.
    if (t.rank != 2) return null;
    if (t.block_order != .row_major) return null;
    if (already_nt) {
        // `[n, k]` already, so only the interleave is left to do; a weight that
        // cannot take it would be copied into the layout it already has.
        if (t.quant_axis != 1) return null;
        const k_nt: usize = t.shape[1];
        if (k_nt % info.block_elems != 0) return null;
        if (layoutFor(target, t.dtype, t.shape[0], k_nt).order != target.tiles.quant_block_order or target.tiles.quant_block_order == .row_major) return null;
        return .{ .b_tid = b_tid, .dtype = t.dtype, .k = k_nt, .n = t.shape[0], .alpha = scale.alpha, .beta = scale.beta, .already_nt = true };
    }
    if (t.quant_axis != 0) return null;
    const k: usize = t.shape[0];
    if (k % info.block_elems != 0) return null;
    // A target reads its own grouping fastest; a weight that cannot take it stays
    // as it is rather than moving to a layout the target never asked for.
    if (layoutFor(target, t.dtype, t.shape[1], k).order != target.tiles.quant_block_order) return null;

    return .{
        .b_tid = b_tid,
        .dtype = t.dtype,
        .k = k,
        .n = t.shape[1],
        .alpha = scale.alpha,
        .beta = scale.beta,
    };
}

/// The tile an `[n, k]` weight gets on `target`, and the block order
/// inside it: the one the target's kernel reads, wherever every tile splits into
/// whole groups of it.
fn layoutFor(target: Target, dtype: types.DType, n: usize, k: usize) struct { tile: [2]usize, order: types.QuantBlockOrder } {
    const tile = plan.chooseQuantRowTiles(target.tiles, dtype, n, k);
    const want = target.tiles.quant_block_order;
    const g = want.groupRows();
    return .{ .tile = tile, .order = if (tile[0] % g == 0 and n % g == 0) want else .row_major };
}

/// Bytes of rows re-laid per step: the most of a weight on the host at once.
/// Small under test, so test-sized weights span several chunks.
const relayout_chunk_bytes: usize = if (builtin.is_test) 512 else 16 << 20;

/// A q8 `[K, N]` matmul-B as `[N, K]` with `quant_axis == 1`, or an `[N, K]` one
/// interleaved, without requantization. Memoized by source, tiling, and device: a
/// quantized weight cannot be retiled downstream, and a tensor lives on one device.
pub fn relayout(
    gpa: std.mem.Allocator,
    mgr: *StorageManager,
    target: Target,
    source: TensorId,
) Error!TensorId {
    const src = try mgr.getConst(source);
    const info = src.dtype.info();
    if (src.dtype != .q8_0 or src.rank != 2) return Error.InvalidArgument;
    if (src.quant_axis != 0 and src.quant_axis != 1) return Error.InvalidArgument;

    // `quant_axis == 0` is `[k, n]` block-major; `== 1` is already `[n, k]` and
    // only needs the interleave.
    const from_nt = src.quant_axis == 1;
    const k: usize = if (from_nt) src.shape[1] else src.shape[0];
    const n: usize = if (from_nt) src.shape[0] else src.shape[1];
    if (k % info.block_elems != 0) return Error.InvalidArgument;

    const layout = layoutFor(target, src.dtype, n, k);
    if (from_nt and layout.order == .row_major) return Error.InvalidArgument;
    const tile = layout.tile;
    // The order inside the tile is as much the layout as the tile shape is.
    const key = [_]usize{ tile[0], tile[1], @intFromEnum(layout.order) };
    if (mgr.derivedFind(.relayout, &key, target.device, &.{source})) |tid| return tid;

    const blocks: usize = k / info.block_elems;
    const bb: usize = info.block_bytes;
    const row_bytes = blocks * bb;
    const opts: TiledTensor.InitOptions = .{
        .tile_alignment = target.tiles.tile_alignment,
        .quant_axis = 1,
        .block_order = layout.order,
    };
    // Made where it will be read, so a device target never stages the whole copy.
    const out = if (mgr.deviceMemoryFor(target.device) != null)
        try mgr.createDeviceTensor(src.dtype, &.{ n, k }, &tile, opts, target.device)
    else
        try mgr.createTiledTensor(src.dtype, &.{ n, k }, &tile, opts);
    const mapping: derived.Relayout = .{ .transposed = !from_nt, .order = layout.order, .tile_rows = tile[0] };

    // Re-laid a chunk of rows at a time, whole groups each, so a chunk is one byte
    // range of both its source rows and its tile: only chunks are ever on the host.
    // A `[k, n]` source is read whole instead, since each of its rows is strided.
    const g = layout.order.groupRows();
    const chunk_rows = if (from_nt) @max(g, relayout_chunk_bytes / row_bytes / g * g) else n;
    const in_buf = gpa.alloc(u8, @min(chunk_rows, n) * row_bytes) catch return Error.OutOfMemory;
    defer gpa.free(in_buf);
    if (!from_nt) try mgr.readPackedAtPlacement(source, in_buf);
    const out_buf = gpa.alloc(u8, @min(chunk_rows, tile[0]) * row_bytes) catch return Error.OutOfMemory;
    defer gpa.free(out_buf);

    const tile_counts = (try mgr.getConst(out)).tile_counts[0];
    for (0..tile_counts) |ti| {
        const row0 = ti * tile[0];
        const rows = @min(tile[0], n - row0);
        var c0: usize = 0;
        while (c0 < rows) : (c0 += chunk_rows) {
            const first = row0 + c0;
            const count = @min(chunk_rows, rows - c0);
            const base = if (from_nt) first * blocks else 0;
            if (from_nt) try mgr.readQuantBlocksAtPlacement(source, base, in_buf[0 .. count * row_bytes]);
            // Through the order: the chunk starts on a group, so its offsets are the tile's.
            const dst = out_buf[0 .. count * row_bytes];
            for (0..count) |r| {
                for (0..blocks) |kb| {
                    const from = (mapping.sourceAt(blocks, n, first + r, kb) - base) * bb;
                    layout.order.storeBlock(dst, blocks, r, kb, in_buf[from..][0..bb]);
                }
            }
            try mgr.writeTileAtPlacement(out, ti, c0 * row_bytes, dst);
        }
    }

    try mgr.derivedRecord(.relayout, &key, target.device, out, &.{.{
        .tid = source,
        .view = .{
            .rows = blocks,
            .row_stride = n,
            .offset = 0,
            .len = n,
            .block_bytes = bb,
            .mapping = .{ .relayout = mapping },
        },
    }});
    return out;
}
