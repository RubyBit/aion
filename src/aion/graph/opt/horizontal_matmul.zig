// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Fuse parallel MatMuls sharing A using a column-concatenated quantized weight and
//! sliced outputs; target defaults decide whether reduced dispatch cost justifies it.

const std = @import("std");

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
const TensorId = manager_mod.TensorId;
const Target = target_mod.Target;
const DType = types.DType;

pub const Error = graph_mod.GraphError || manager_mod.StorageError;

const MAX_RANK: usize = 8;

/// A fusable `out = A @ B` with B's quantization and batch layout resolved from storage.
/// `compileGraphOpt` validates stored and declared layouts agree.
const Cand = struct {
    node: usize,
    a: ValueId,
    out: ValueId,
    b_tid: TensorId,
    alpha: f32,
    beta: f32,
    dtype: DType,
    rank: usize,
    k: usize,
    n: usize,
    /// B's batch extents, `shape[0 .. rank-2]`. Borrowed from the stored tensor, whose
    /// pointer is stable for the store's lifetime, so it outlives the pass.
    lead: []const usize,
    taken: bool = false,
};

/// One fused group, resolved: the wide weight and the values its members now read.
const Group = struct {
    anchor: usize,
    members: []Cand,
    weight: ValueId,
    wide: ValueId,
    slices: []ValueId,
};

/// The pass, as the shared list driver wants it: one `run` per node list.
pub const Rule = struct {
    mgr: *StorageManager,
    target: Target,

    pub fn run(self: Rule, rw: *Rewriter) Error!void {
        return fuseList(rw, self.mgr, self.target);
    }
};

fn fuseList(rw: *Rewriter, mgr: *StorageManager, target: Target) Error!void {
    const gpa = rw.gpa;
    const g = rw.g;
    const nodes = rw.input();

    var cands: std.ArrayList(Cand) = .empty;
    defer cands.deinit(gpa);
    for (nodes, 0..) |node, idx| {
        if (candidate(g, mgr, node, idx)) |c| try cands.append(gpa, c);
    }
    if (cands.items.len < 2) return rw.keepAll();

    var groups: std.ArrayList(Group) = .empty;
    defer {
        for (groups.items) |grp| {
            gpa.free(grp.members);
            gpa.free(grp.slices);
        }
        groups.deinit(gpa);
    }

    var set: std.ArrayList(Cand) = .empty;
    defer set.deinit(gpa);

    for (0..cands.items.len) |i| {
        if (cands.items[i].taken) continue;
        cands.items[i].taken = true;
        const head = cands.items[i];

        set.clearRetainingCapacity();
        try set.append(gpa, head);
        for (cands.items[i + 1 ..]) |*other| {
            if (other.taken or !fusable(head, other.*)) continue;
            other.taken = true;
            try set.append(gpa, other.*);
        }
        if (set.items.len < 2) continue;
        try groups.append(gpa, try fuse(rw, mgr, target, set.items));
    }
    if (groups.items.len == 0) return rw.keepAll();

    var dropped = std.AutoHashMap(usize, void).init(gpa);
    defer dropped.deinit();
    for (groups.items) |grp| {
        for (grp.members) |m| dropped.put(m.node, {}) catch return Error.OutOfMemory;
    }

    for (nodes, 0..) |node, idx| {
        if (!dropped.contains(idx)) {
            try rw.add(node);
            continue;
        }
        // The group's replacement lands at its anchor, so every surviving consumer still
        // reads its value at the step it read it at before.
        for (groups.items) |grp| {
            if (grp.anchor == idx) try emit(rw, grp);
        }
    }
}

/// Return a candidate whose B is quantized, K-blocked, and safe for per-block-row concat.
/// Pairwise compatibility is checked by `fusable`.
fn candidate(g: *const Graph, mgr: *const StorageManager, node: Node, idx: usize) ?Cand {
    const mm = switch (node.op) {
        .MatMul => |m| m,
        else => return null,
    };
    if (node.inputs.len != 2) return null;
    const b = g.values.items[@intCast(node.inputs[1])];
    const ext = b.external orelse return null;
    const b_tid: TensorId = @intCast(ext);

    const t = mgr.getConst(b_tid) catch return null;
    const info = t.dtype.info();
    if (!info.is_quantized) return null;
    const rank: usize = t.rank;
    if (rank < 2) return null;
    if (@as(usize, t.quant_axis) != rank - 2) return null;
    const k: usize = t.shape[rank - 2];
    if (k % info.block_elems != 0) return null;

    return .{
        .node = idx,
        .a = node.inputs[0],
        .out = node.output,
        .b_tid = b_tid,
        .alpha = mm.alpha,
        .beta = mm.beta,
        .dtype = t.dtype,
        .rank = rank,
        .k = k,
        .n = t.shape[rank - 1],
        .lead = t.shape[0 .. rank - 2],
    };
}

/// Two candidates fuse when they share A and have identical stored batch extents.
/// Broadcast-compatible but unequal batches cannot share one concatenated layout.
fn fusable(a: Cand, b: Cand) bool {
    if (a.a != b.a) return false;
    if (a.alpha != b.alpha or a.beta != b.beta) return false;
    return a.dtype == b.dtype and a.rank == b.rank and a.k == b.k and
        std.mem.eql(usize, a.lead, b.lead);
}

/// Build the wide weight and the values the members will read.
fn fuse(rw: *Rewriter, mgr: *StorageManager, target: Target, set: []const Cand) Error!Group {
    const gpa = rw.gpa;
    const head = set[0];
    // The weight's rank and the output's need not agree: a `[K, N]` weight broadcasts
    // into a `[batch, seq, K]` activation, so the concat is shaped by the former and the
    // replacement matmul and slices by the latter.
    const out = rw.g.values.items[@intCast(head.out)];
    const out_rank = out.shape.len;

    const sources = gpa.alloc(TensorId, set.len) catch return Error.OutOfMemory;
    defer gpa.free(sources);
    var sum_n: usize = 0;
    for (set, 0..) |c, i| {
        sources[i] = c.b_tid;
        sum_n += c.n;
    }

    const cat = try concatColumns(gpa, mgr, target, sources);
    const cat_t = try mgr.getConst(cat);
    const weight = try rw.bound(head.dtype, cat_t.shape, @intCast(cat));

    var wide_shape: [MAX_RANK]usize = undefined;
    @memcpy(wide_shape[0..out_rank], out.shape);
    wide_shape[out_rank - 1] = sum_n;
    const wide = try rw.value(out.dtype.?, wide_shape[0..out_rank]);

    const members = gpa.dupe(Cand, set) catch return Error.OutOfMemory;
    errdefer gpa.free(members);
    const slices = gpa.alloc(ValueId, set.len) catch return Error.OutOfMemory;
    errdefer gpa.free(slices);

    for (set, 0..) |c, i| {
        const member_out = rw.g.values.items[@intCast(c.out)];
        slices[i] = try rw.value(member_out.dtype.?, member_out.shape);
        try rw.redirect(c.out, slices[i]);
    }
    return .{ .anchor = head.node, .members = members, .weight = weight, .wide = wide, .slices = slices };
}

fn emit(rw: *Rewriter, grp: Group) Error!void {
    const head = grp.members[0];
    try rw.add(.{
        .op = .{ .MatMul = .{ .alpha = head.alpha, .beta = head.beta } },
        .inputs = try rw.ids(&.{ head.a, grp.weight }),
        .output = grp.wide,
    });

    const wide_rank = rw.g.values.items[@intCast(grp.wide)].shape.len;
    var offset: usize = 0;
    for (grp.members, grp.slices) |m, slice| {
        const shape = rw.g.values.items[@intCast(slice)].shape;
        var starts: [MAX_RANK]usize = @splat(0);
        starts[wide_rank - 1] = offset;
        try rw.add(.{
            .op = .{ .ViewSliceND = .{
                .starts = try rw.dims(starts[0..wide_rank]),
                .lens = try rw.dims(shape),
            } },
            .inputs = try rw.ids(&.{grp.wide}),
            .output = slice,
        });
        offset += m.n;
    }
}

/// Concatenate quantized `[.., K, Ni]` weights along N without requantization.
/// Results are memoized by sources, tiling, and device because quantized weights cannot
/// be retiled and each tensor has one residency; public callers are fully validated.
pub fn concatColumns(
    gpa: std.mem.Allocator,
    mgr: *StorageManager,
    target: Target,
    sources: []const TensorId,
) Error!TensorId {
    if (sources.len == 0) return Error.InvalidArgument;

    const first = try mgr.getConst(sources[0]);
    const dtype = first.dtype;
    const info = dtype.info();
    if (!info.is_quantized) return Error.InvalidArgument;
    const rank: usize = first.rank;
    if (rank < 2) return Error.InvalidArgument;

    const k: usize = first.shape[rank - 2];
    if (k % info.block_elems != 0) return Error.InvalidArgument;

    var sum_n: usize = 0;
    for (sources) |sid| {
        const t = try mgr.getConst(sid);
        if (t.dtype != dtype or @as(usize, t.rank) != rank) return Error.InvalidArgument;
        if (t.shape[rank - 2] != k) return Error.InvalidArgument;
        if (@as(usize, t.quant_axis) != rank - 2) return Error.InvalidArgument;
        if (!std.mem.eql(usize, first.shape[0 .. rank - 2], t.shape[0 .. rank - 2])) return Error.InvalidArgument;
        sum_n += t.shape[rank - 1];
    }

    // Tile the fused weight exactly as the MatMul lowering tiles a quant B (fixed
    // M-hint), so no (impossible) quant retile is demanded downstream.
    const chosen = plan.chooseMatMulTiles(target.tiles, plan.matMulMHint(target.tiles), sum_n, k, dtype);
    var shape: [MAX_RANK]usize = undefined;
    var tile: [MAX_RANK]usize = undefined;
    @memcpy(shape[0 .. rank - 2], first.shape[0 .. rank - 2]);
    @memset(tile[0 .. rank - 2], 1);
    shape[rank - 2] = k;
    shape[rank - 1] = sum_n;
    tile[rank - 2] = chosen.tk;
    tile[rank - 1] = chosen.tn;

    if (mgr.derivedFind(.column_concat, tile[0..rank], target.device, sources)) |tid| return tid;

    var lead: usize = 1;
    for (first.shape[0 .. rank - 2]) |d| lead *= d;
    const rows: usize = lead * (k / info.block_elems);
    const bb = info.block_bytes;

    const out_buf = gpa.alloc(u8, rows * sum_n * bb) catch return Error.OutOfMemory;
    defer gpa.free(out_buf);

    const views = gpa.alloc(derived.Source, sources.len) catch return Error.OutOfMemory;
    defer gpa.free(views);

    var col: usize = 0;
    for (sources, 0..) |sid, i| {
        const n_i: usize = (try mgr.getConst(sid)).shape[rank - 1];
        const view: derived.View = .{
            .rows = rows,
            .row_stride = sum_n,
            .offset = col,
            .len = n_i,
            .block_bytes = bb,
        };
        const src_buf = gpa.alloc(u8, view.sourceBytes()) catch return Error.OutOfMemory;
        defer gpa.free(src_buf);
        try mgr.readPackedAtPlacement(sid, src_buf);

        var r: usize = 0;
        while (r < rows) : (r += 1) {
            const dst_off = (r * sum_n + col) * bb;
            const src_off = r * n_i * bb;
            @memcpy(out_buf[dst_off..][0 .. n_i * bb], src_buf[src_off..][0 .. n_i * bb]);
        }
        views[i] = .{ .tid = sid, .view = view };
        col += n_i;
    }

    const out = try mgr.createTiledTensor(dtype, shape[0..rank], tile[0..rank], .{
        .tile_alignment = target.tiles.tile_alignment,
        .quant_axis = @intCast(rank - 2),
    });
    try mgr.writePackedAtPlacement(out, out_buf);
    try mgr.derivedRecord(.column_concat, tile[0..rank], target.device, out, views);
    return out;
}
