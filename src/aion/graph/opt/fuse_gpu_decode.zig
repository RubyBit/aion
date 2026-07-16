// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! GPU-only compile-time GELU/multiply fusion.
const std = @import("std");
const graph_mod = @import("../graph.zig");
const plan = @import("../plan.zig");
const Graph = graph_mod.Graph;
const Node = graph_mod.Node;
const ValueId = graph_mod.ValueId;
pub const Error = graph_mod.GraphError;

pub fn run(allocator: std.mem.Allocator, graph: *Graph, policy: plan.TilePolicy) Error!void {
    if (policy.target_kind == .cpu) return;
    const reads = allocator.alloc(u32, graph.values.items.len) catch return Error.OutOfMemory;
    defer allocator.free(reads);
    @memset(reads, 0);
    const output = allocator.alloc(bool, graph.values.items.len) catch return Error.OutOfMemory;
    defer allocator.free(output);
    @memset(output, false);
    for (graph.nodes.items) |node| {
        for (node.inputs) |id| reads[@intCast(id)] += 1;
    }
    for (graph.outputs.items) |id| output[@intCast(id)] = true;
    const drop = allocator.alloc(bool, graph.nodes.items.len) catch return Error.OutOfMemory;
    defer allocator.free(drop);
    @memset(drop, false);
    var changed = false;
    for (graph.nodes.items, 0..) |node, i| {
        const u = switch (node.op) { .Unary => |x| x, else => continue };
        if (u.op != .gelu or reads[@intCast(node.output)] != 1 or output[@intCast(node.output)]) continue;
        var consumer: ?usize = null;
        var j = i + 1;
        while (j < graph.nodes.items.len) : (j += 1) for (graph.nodes.items[j].inputs) |id| {
            if (id == node.output) { consumer = j; break; }
        };
        const ci = consumer orelse continue;
        const b = switch (graph.nodes.items[ci].op) { .ElemwiseBinary => |x| x, else => continue };
        if (b.op != .mul) continue;
        const other = if (graph.nodes.items[ci].inputs[0] == node.output) graph.nodes.items[ci].inputs[1] else graph.nodes.items[ci].inputs[0];
        const inputs = graph.arenaAlloc().alloc(ValueId, 2) catch return Error.OutOfMemory;
        inputs[0] = node.inputs[0]; inputs[1] = other;
        graph.nodes.items[ci].op = .GeluMul;
        graph.nodes.items[ci].inputs = inputs;
        drop[i] = true; changed = true;
    }
    if (!changed) return;
    var kept: std.ArrayList(Node) = .empty;
    errdefer kept.deinit(allocator);
    for (graph.nodes.items, 0..) |node, i| if (!drop[i]) kept.append(allocator, node) catch return Error.OutOfMemory;
    graph.nodes.deinit(allocator);
    graph.nodes = kept;
}
