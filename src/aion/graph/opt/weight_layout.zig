// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Re-lay a q8 matmul weight into the order a decode matvec reads it in, and
//! contract against its rows instead (`MatMul` -> `MatMulNT`). Only q8: that is
//! the one dtype the CPU's NT lowering has a kernel for.
//!
//! Two changes, both pure byte permutations of the same blocks:
//!
//! 1. *Transposed.* A q8 `[K, N]` matmul-B holds its blocks block-major,
//!    `[K/32][N]`: the blocks one output column needs are `N * 34` bytes apart.
//!    A matvec reading them either strides — re-reading each cache line once per
//!    column group — or gives up register-resident sums. Transposed to `[N, K]`
//!    it is one unbroken run per column and gets both. Measured on 257 MB of
//!    weights, one core: 30 GB/s block-major against 60 GB/s as `[N, K]`.
//! 2. *Grouped* (`types.QuantBlockOrder`). Rows still sit a whole row apart,
//!    so a kernel computing several walks several streams, and an integer dot
//!    gives one row per lane only if the rows' bytes share a vector. Grouping as
//!    many rows as the target's dot has lanes gives both: one stream, and the
//!    per-block scaling paid once for the group. The target names the grouping
//!    its kernel reads (`Target.quant_block_order`).
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
const manager_mod = @import("../../storage/manager.zig");
const derived = @import("../../storage/derived.zig");
const types = @import("../../backend/types.zig");
const rewriter_mod = @import("rewriter.zig");
const target_mod = @import("../target.zig");
const thread_pool = @import("../../runtime/thread_pool.zig");


const Graph = graph_mod.Graph;
const Node = graph_mod.Node;
const ValueId = graph_mod.ValueId;
const Rewriter = rewriter_mod.Rewriter;
const StorageManager = manager_mod.StorageManager;
const Tensor = @import("../../storage/storage.zig").Tensor;
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
        var workers: Workers = .{ .threads = self.mgr.bulk_threads };
        defer workers.deinit();

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
                const laid = try relayoutWith(rw.gpa, self.mgr, self.target, table, &workers);
                const weight = try rw.bound(t.dtype, t.shape, @intCast(laid));
                try rw.add(.{ .op = node.op, .inputs = try rw.ids(&.{ weight, node.inputs[1] }), .output = node.output });
                changed = true;
                continue;
            };
            const c = candidate(rw.g, self.mgr, self.target, node) orelse {
                try rw.add(node);
                continue;
            };
            const laid = relayoutWith(rw.gpa, self.mgr, self.target, c.b_tid, &workers) catch {
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
        if (orderFor(target, t.shape[0]) != target.quant_block_order or target.quant_block_order == .row_major) return null;
        return .{ .b_tid = b_tid, .dtype = t.dtype, .k = k_nt, .n = t.shape[0], .alpha = scale.alpha, .beta = scale.beta, .already_nt = true };
    }
    if (t.quant_axis != 0) return null;
    const k: usize = t.shape[0];
    if (k % info.block_elems != 0) return null;
    // A target reads its own grouping fastest; a weight that cannot take it stays
    // as it is rather than moving to a layout the target never asked for.
    if (orderFor(target, t.shape[1]) != target.quant_block_order) return null;

    return .{
        .b_tid = b_tid,
        .dtype = t.dtype,
        .k = k,
        .n = t.shape[1],
        .alpha = scale.alpha,
        .beta = scale.beta,
    };
}

/// The block order an `[n, k]` weight takes on `target`: the one its kernel reads,
/// wherever the rows split into whole groups of it. A device chunk holds a multiple
/// of 32 rows (`storage/layout.zig`), a whole number of any group, so the order never
/// depends on how the weight is chunked.
fn orderFor(target: Target, n: usize) types.QuantBlockOrder {
    const want = target.quant_block_order;
    return if (n % want.groupRows() == 0) want else .row_major;
}

/// Bytes of rows re-laid per step when the rows have to be staged on the host: the
/// most of a weight held twice at once. Small under test, so test-sized weights span
/// several steps.
const relayout_step_bytes: usize = if (builtin.is_test) 512 else 16 << 20;

/// Rows a worker claims at a time: enough bytes that claiming costs nothing next to
/// the copy.
const relayout_grain_bytes: usize = 256 << 10;

/// The pool one pass's relayouts share, made on the first one that can use it. A
/// weight is re-laid once per (source, order, device), so most compiles never make it.
const Workers = struct {
    threads: usize,
    pool: ?thread_pool.ThreadPool = null,

    fn get(self: *Workers, gpa: std.mem.Allocator) ?*thread_pool.ThreadPool {
        if (self.threads < 2) return null;
        if (self.pool == null) self.pool = thread_pool.ThreadPool.init(gpa, .{ .thread_count = self.threads }) catch return null;
        return &self.pool.?;
    }

    fn deinit(self: *Workers) void {
        if (self.pool) |*p| p.deinit();
    }
};

/// Rows `[first, first + count)` of a relayout: each result row gathers its blocks
/// from the source through `view` and stores them through its order. Rows touch
/// disjoint bytes of `dst` (a group's rows interleave, but never on the same byte),
/// so any split of them across threads is exact.
const RelayJob = struct {
    dst: []u8,
    src: []const u8,
    view: derived.View,
    /// The result row `dst` starts at, and the source block `src` starts at.
    first: usize,
    base: usize,

    fn rows(self: *const RelayJob, lo: usize, hi: usize) void {
        const bb = self.view.block_bytes;
        for (lo..hi) |r| {
            for (0..self.view.blocks) |kb| {
                const from = (self.view.sourceAt(self.first + r, kb) - self.base) * bb;
                self.view.order.storeBlock(self.dst, self.view.blocks, r, kb, self.src[from..][0..bb]);
            }
        }
    }

    fn run(raw: *anyopaque, lo: usize, hi: usize, _: usize) void {
        rows(@ptrCast(@alignCast(raw)), lo, hi);
    }
};

/// A q8 `[K, N]` matmul-B as `[N, K]` with `quant_axis == 1`, or an `[N, K]` one
/// interleaved, without requantization. Memoized by source, order, and device: a
/// quantized weight's order is fixed once laid out, and a tensor lives on one device.
pub fn relayout(
    gpa: std.mem.Allocator,
    mgr: *StorageManager,
    target: Target,
    source: TensorId,
) Error!TensorId {
    var workers: Workers = .{ .threads = mgr.bulk_threads };
    defer workers.deinit();
    return relayoutWith(gpa, mgr, target, source, &workers);
}

fn relayoutWith(
    gpa: std.mem.Allocator,
    mgr: *StorageManager,
    target: Target,
    source: TensorId,
    workers: *Workers,
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

    const order = orderFor(target, n);
    if (from_nt and order == .row_major) return Error.InvalidArgument;
    if (mgr.derivedFind(order, target.device, source)) |tid| return tid;

    const blocks: usize = k / info.block_elems;
    const bb: usize = info.block_bytes;
    const row_bytes = blocks * bb;
    // Every block of the result is written below, so its bytes start unfilled.
    const opts: Tensor.InitOptions = .{ .quant_axis = 1, .block_order = order, .zero_fill = false };
    // Made where it will be read, so a device target never stages the whole copy.
    const out = if (mgr.deviceMemoryFor(target.device) != null)
        try mgr.createDeviceTensor(src.dtype, &.{ n, k }, opts, target.device)
    else
        try mgr.createTensor(src.dtype, &.{ n, k }, opts);
    const view: derived.View = .{ .blocks = blocks, .cols = n, .block_bytes = bb, .transposed = !from_nt, .order = order };
    const pool = workers.get(gpa);

    // A host source is read where it lies (the mapped file, for a loaded model). Any
    // other — on a device, or folded into another derived weight — is staged: a step
    // of rows at a time from `[n, k]`, or whole from `[k, n]`, whose rows are strided.
    const in_place: ?[]const u8 = if (src.device.kind == .cpu and try mgr.tensorHasBacking(source)) (try mgr.backingConst(source)).data[0 .. n * row_bytes] else null;
    const g = order.groupRows();
    const step_rows = if (in_place != null or !from_nt) n else @max(g, relayout_step_bytes / row_bytes / g * g);
    const in_buf: []u8 = if (in_place != null) &.{} else gpa.alloc(u8, @min(step_rows, n) * row_bytes) catch return Error.OutOfMemory;
    defer gpa.free(in_buf);
    if (in_place == null and !from_nt) try mgr.readPackedAtPlacement(source, in_buf);
    // A host result is written in place; a device one is staged a step at a time.
    const out_t = try mgr.getMut(out);
    const host = out_t.device.kind == .cpu;
    const dev_step = if (host) 0 else @max(g, relayout_step_bytes / row_bytes / g * g);
    const out_buf: []u8 = if (host) &.{} else gpa.alloc(u8, @min(dev_step, n) * row_bytes) catch return Error.OutOfMemory;
    defer if (!host) gpa.free(out_buf);

    var first: usize = 0;
    while (first < n) {
        const count = @min(if (host) step_rows else @min(step_rows, dev_step), n - first);
        const base = if (from_nt and in_place == null) first * blocks else 0;
        if (from_nt and in_place == null) try mgr.readQuantBlocksAtPlacement(source, base, in_buf[0 .. count * row_bytes]);
        // Through the order: the step starts on a group, so its offsets are the result's.
        const dst = if (host) out_t.data[first * row_bytes ..][0 .. count * row_bytes] else out_buf[0 .. count * row_bytes];
        const job: RelayJob = .{ .dst = dst, .src = in_place orelse in_buf, .view = view, .first = first, .base = base };
        if (pool) |p| {
            p.parallelForDynamic(@constCast(@ptrCast(&job)), count, @max(1, relayout_grain_bytes / row_bytes), RelayJob.run);
        } else {
            RelayJob.rows(&job, 0, count);
        }
        if (!host) try mgr.writeAtPlacement(out, first * row_bytes, dst);
        first += count;
    }

    try mgr.derivedRecord(target.device, out, source, view);
    return out;
}
