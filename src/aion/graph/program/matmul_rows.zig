// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Which matmuls contract against B's rows: those over a transposed B.
//!
//! A model says `a @ bᵀ` as `MatMul(a, ViewTranspose2D(b))`, and that is all the
//! graph (and a package) ever holds. Lowering is where it becomes the NT step,
//! reading `b` as it is stored. That is lowering's to decide, not an optional
//! pass's: a quantized `b` has no transposed form, since its blocks run along K,
//! so only a kernel reading its rows can use it.
//!
//! A transpose that only folding matmuls read is never lowered. It gets no tensor,
//! and none of its readers wants one. A transpose something else reads lowers as it
//! always did.

const std = @import("std");

const graph_mod = @import("../graph.zig");

const Graph = graph_mod.Graph;
const Node = graph_mod.Node;
const ValueId = graph_mod.ValueId;

pub const Error = error{OutOfMemory};

/// Every transpose's source, graph-wide: a weight is transposed once at the top
/// level and read inside a `Loop` body.
pub const Sources = std.AutoHashMapUnmanaged(ValueId, ValueId);

pub fn transposeSources(gpa: std.mem.Allocator, g: *const Graph) Error!Sources {
    var sources: Sources = .empty;
    errdefer sources.deinit(gpa);
    try noteTransposes(gpa, &sources, g.nodes.items);
    for (g.regions.items) |region| try noteTransposes(gpa, &sources, region.nodes);
    return sources;
}

fn noteTransposes(gpa: std.mem.Allocator, sources: *Sources, nodes: []const Node) Error!void {
    for (nodes) |node| {
        if (node.op != .ViewTranspose2D) continue;
        try sources.put(gpa, node.output, node.inputs[0]);
    }
}

pub const Rows = struct {
    sources: Sources,
    /// The transposes something other than a folding matmul reads.
    still_read: std.AutoHashMapUnmanaged(ValueId, void) = .empty,

    pub fn init(gpa: std.mem.Allocator, g: *const Graph) Error!Rows {
        var self: Rows = .{ .sources = try transposeSources(gpa, g) };
        errdefer self.deinit(gpa);
        try self.noteReads(gpa, g, g.nodes.items, g.outputs.items);
        for (g.regions.items) |region| try self.noteReads(gpa, g, region.nodes, region.outputs);
        return self;
    }

    pub fn deinit(self: *Rows, gpa: std.mem.Allocator) void {
        self.sources.deinit(gpa);
        self.still_read.deinit(gpa);
    }

    /// The `[N, K]` value `node` contracts against the rows of, if it is a matmul
    /// that lowers to the NT step.
    pub fn of(self: *const Rows, g: *const Graph, node: Node) ?ValueId {
        return rows(g, &self.sources, node);
    }

    /// Whether `node` is a transpose that folds away with its readers.
    pub fn vanishes(self: *const Rows, node: Node) bool {
        return node.op == .ViewTranspose2D and !self.still_read.contains(node.output);
    }

    fn noteReads(self: *Rows, gpa: std.mem.Allocator, g: *const Graph, nodes: []const Node, outputs: []const ValueId) Error!void {
        for (nodes) |node| {
            const folds = rows(g, &self.sources, node) != null;
            for (node.inputs, 0..) |in, slot| {
                if (folds and slot == 1) continue;
                if (self.sources.contains(in)) try self.still_read.put(gpa, in, {});
            }
        }
        for (outputs) |out| {
            if (self.sources.contains(out)) try self.still_read.put(gpa, out, {});
        }
    }
};

/// The `[N, K]` operand a matmul over `bᵀ` contracts against, when the NT step
/// takes it: f32 A, and an f32 B or a q8_0 one of whole blocks. An f32 B gains too:
/// its transpose would otherwise be copied every run.
pub fn rows(g: *const Graph, sources: *const Sources, node: Node) ?ValueId {
    if (node.op != .MatMul) return null;
    const b = sources.get(node.inputs[1]) orelse return null;
    const a_v = g.values.items[@intCast(node.inputs[0])];
    const b_v = g.values.items[@intCast(b)];
    if (a_v.dtype != .f32 or b_v.shape.len != 2) return null;
    return switch (b_v.dtype orelse return null) {
        .q8_0 => if (b_v.shape[1] % 32 == 0) b else null,
        .f32 => b,
        else => null,
    };
}
