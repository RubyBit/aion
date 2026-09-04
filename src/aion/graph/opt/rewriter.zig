// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Graph editing for list-local rewrite passes; the driver applies value remaps
//! graph-wide and re-infers metadata after rebuilding each schedule/body.

const std = @import("std");

const graph_mod = @import("../graph.zig");
const infer_mod = @import("../infer.zig");
const types = @import("../../backend/types.zig");

const Graph = graph_mod.Graph;
const Node = graph_mod.Node;
const ValueId = graph_mod.ValueId;
const ExternalId = graph_mod.ExternalId;

pub const Error = graph_mod.GraphError;

/// Run a `fn run(*Rewriter) Error!void` rule over every node list, then remap and
/// re-infer once.
pub fn rewrite(
    gpa: std.mem.Allocator,
    g: *Graph,
    rule: anytype,
) (Error || infer_mod.InferError)!void {
    var rw = Rewriter.init(gpa, g);
    defer rw.deinit();

    var at: usize = 0;
    while (at < rw.lists()) : (at += 1) {
        rw.at = at;
        try rule.run(&rw);
        try rw.install();
    }
    try rw.finish();
}

pub const Rewriter = struct {
    gpa: std.mem.Allocator,
    g: *Graph,
    remap: std.AutoHashMapUnmanaged(ValueId, ValueId) = .empty,

    /// The list being rewritten: 0 is the top-level schedule, `i + 1` is region `i`'s
    /// body. Owned by `rewrite`.
    at: usize = 0,
    out: std.ArrayList(Node) = .empty,
    emitted: bool = false,
    kept: bool = false,

    pub fn init(gpa: std.mem.Allocator, g: *Graph) Rewriter {
        return .{ .gpa = gpa, .g = g };
    }

    pub fn deinit(self: *Rewriter) void {
        self.out.deinit(self.gpa);
        self.remap.deinit(self.gpa);
        self.* = undefined;
    }

    /// Node lists in the graph: the top-level schedule plus one body per region.
    pub fn lists(self: *const Rewriter) usize {
        return 1 + self.g.regions.items.len;
    }

    /// The nodes of the list being rewritten. A rule reads this, never `g.nodes`.
    pub fn input(self: *const Rewriter) []const Node {
        if (self.at == 0) return self.g.nodes.items;
        return self.g.regions.items[self.at - 1].nodes;
    }

    /// This list is unchanged. A rule has to say so, because an empty emit would
    /// otherwise be indistinguishable from "delete every node".
    pub fn keepAll(self: *Rewriter) void {
        self.kept = true;
    }

    pub fn add(self: *Rewriter, node: Node) Error!void {
        self.emitted = true;
        self.out.append(self.gpa, node) catch return Error.OutOfMemory;
    }

    /// A new intermediate. Metadata is set so the pass can build on it immediately;
    /// the driver re-infers, so it does not have to be right about anything else.
    pub fn value(self: *Rewriter, dtype: types.DType, shape: []const usize) Error!ValueId {
        const id = try self.g.addValue();
        self.g.values.items[@intCast(id)] = .{ .dtype = dtype, .shape = try self.g.dupeShape(shape) };
        return id;
    }

    /// A new leaf bound to an existing tensor.
    pub fn bound(self: *Rewriter, dtype: types.DType, shape: []const usize, ext: ExternalId) Error!ValueId {
        const id = try self.value(dtype, shape);
        self.g.values.items[@intCast(id)].external = ext;
        return id;
    }

    pub fn ids(self: *Rewriter, from: []const ValueId) Error![]ValueId {
        const out = self.g.arenaAlloc().alloc(ValueId, from.len) catch return Error.OutOfMemory;
        @memcpy(out, from);
        return out;
    }

    pub fn dims(self: *Rewriter, from: []const usize) Error![]usize {
        const out = self.g.arenaAlloc().alloc(usize, from.len) catch return Error.OutOfMemory;
        @memcpy(out, from);
        return out;
    }

    /// Every read of `old` becomes a read of `new`.
    pub fn redirect(self: *Rewriter, old: ValueId, new: ValueId) Error!void {
        self.remap.put(self.gpa, old, new) catch return Error.OutOfMemory;
    }

    /// Install the rebuilt list, if the rule rebuilt it.
    fn install(self: *Rewriter) Error!void {
        defer {
            self.emitted = false;
            self.kept = false;
        }
        if (!self.emitted) {
            std.debug.assert(self.kept); // the rule said nothing about this list
            return;
        }
        const nodes = self.out.toOwnedSlice(self.gpa) catch return Error.OutOfMemory;
        if (self.at == 0) {
            self.g.nodes.deinit(self.gpa);
            self.g.nodes = .fromOwnedSlice(nodes);
            return;
        }
        const region = &self.g.regions.items[self.at - 1];
        self.gpa.free(region.nodes);
        region.nodes = nodes;
    }

    /// Apply the accumulated value remap everywhere a value can be READ, then re-infer.
    fn finish(self: *Rewriter) (Error || infer_mod.InferError)!void {
        if (self.remap.count() != 0) {
            for (self.g.nodes.items) |node| self.apply(node.inputs);
            for (self.g.regions.items) |region| {
                for (region.nodes) |node| self.apply(node.inputs);
                self.apply(region.outputs);
            }
            self.apply(self.g.outputs.items);
        }
        try infer_mod.infer(self.g);
    }

    fn apply(self: *const Rewriter, values: []ValueId) void {
        for (values) |*v| {
            if (self.remap.get(v.*)) |new| v.* = new;
        }
    }
};
