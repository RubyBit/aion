// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Backward reachability for requested graph outputs.
//!
//! Region bodies can capture outer graph values directly, so their reads are
//! part of reachability even though they are absent from the enclosing If/Loop
//! node's ordinary input list.

const std = @import("std");
const graph_mod = @import("../graph.zig");

pub const Error = error{OutOfMemory};

/// Return one bit per top-level node. A node is live when any requested output
/// depends on it, including dependencies captured by live control-flow regions.
pub fn findLiveNodes(allocator: std.mem.Allocator, graph: *const graph_mod.Graph) Error![]bool {
    const live_values = allocator.alloc(bool, graph.values.items.len) catch return error.OutOfMemory;
    defer allocator.free(live_values);
    @memset(live_values, false);

    const live = allocator.alloc(bool, graph.nodes.items.len) catch return error.OutOfMemory;
    errdefer allocator.free(live);
    @memset(live, false);

    for (graph.outputs.items) |output| markValue(output, live_values);

    var i = graph.nodes.items.len;
    while (i > 0) {
        i -= 1;
        const node = graph.nodes.items[i];

        var produces_live = live_values[@intCast(node.output)];
        for (node.extra_outputs) |output| {
            if (live_values[@intCast(output)]) produces_live = true;
        }
        if (!produces_live) continue;

        live[i] = true;
        for (node.inputs) |input| markValue(input, live_values);
        markRegionReads(graph, node, live_values);
    }
    return live;
}

fn markValue(value: graph_mod.ValueId, live_values: []bool) void {
    const idx: usize = @intCast(value);
    if (idx < live_values.len) live_values[idx] = true;
}

fn markRegionReads(graph: *const graph_mod.Graph, node: graph_mod.Node, live_values: []bool) void {
    const region_ids: [2]?u32 = switch (node.op) {
        .If => |iff| .{ iff.then_region, iff.else_region },
        .Loop => |loop| .{ loop.body_region, null },
        else => return,
    };

    for (region_ids) |maybe_id| {
        const id = maybe_id orelse continue;
        if (id >= graph.regions.items.len) continue;
        const region = graph.regions.items[id];
        for (region.nodes) |region_node| {
            for (region_node.inputs) |input| markValue(input, live_values);
        }
        for (region.outputs) |output| markValue(output, live_values);
    }
}
