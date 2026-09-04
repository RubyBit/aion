// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Lower eligible unquantized 1x1 Conv1D operations to MatMul using byte-identical
//! weights; requires groups/stride/dilation 1 and no padding.

const graph_mod = @import("../graph.zig");
const rewriter_mod = @import("rewriter.zig");

const Graph = graph_mod.Graph;
const Node = graph_mod.Node;
const Rewriter = rewriter_mod.Rewriter;

pub const Error = rewriter_mod.Error;

/// The pass, as the shared list driver wants it: one `run` per node list.
pub const Rule = struct {
    pub fn run(_: Rule, rw: *Rewriter) Error!void {
        return lowerList(rw);
    }
};

fn lowerList(rw: *Rewriter) Error!void {
    const g = rw.g;
    const nodes = rw.input();
    var any = false;
    for (nodes) |node| {
        if (qualifies(g, node)) {
            any = true;
            break;
        }
    }
    if (!any) return rw.keepAll();

    const mm: graph_mod.Op = .{ .MatMul = .{ .alpha = 1.0, .beta = 0.0 } };
    for (nodes) |node| {
        if (!qualifies(g, node)) {
            try rw.add(node);
            continue;
        }
        const x = node.inputs[0];
        const w = node.inputs[1];

        if (node.inputs.len == 2) {
            // No bias: same inputs, same output value.
            try rw.add(.{
                .op = mm,
                .inputs = try rw.ids(&.{ x, w }),
                .output = node.output,
                .extra_outputs = node.extra_outputs,
            });
            continue;
        }
        // With bias [c_out]: matmul -> tmp, then an ordinary broadcast add.
        const out_v = g.values.items[@intCast(node.output)];
        const tmp = try rw.value(out_v.dtype.?, out_v.shape);
        try rw.add(.{ .op = mm, .inputs = try rw.ids(&.{ x, w }), .output = tmp });
        try rw.add(.{
            .op = .{ .ElemwiseBinary = .{ .op = .add } },
            .inputs = try rw.ids(&.{ tmp, node.inputs[2] }),
            .output = node.output,
            .extra_outputs = node.extra_outputs,
        });
    }
}

/// A Conv1D is a pure matmul iff it is 1x1, groups==1, unit stride/dilation, unpadded,
/// with rank-3 x/w and a non-quantized weight.
fn qualifies(g: *const Graph, node: Node) bool {
    const cv = switch (node.op) {
        .Conv1D => |c| c,
        else => return false,
    };
    if (cv.groups != 1 or cv.stride != 1 or cv.dilation != 1) return false;
    if (cv.pad_left != 0 or cv.pad_right != 0) return false;
    if (node.inputs.len < 2 or node.inputs.len > 3) return false;

    const xv = g.values.items[@intCast(node.inputs[0])];
    const wv = g.values.items[@intCast(node.inputs[1])];
    if (xv.shape.len != 3 or wv.shape.len != 3) return false;
    if (wv.shape[0] != 1) return false; // pointwise

    const wdt = wv.dtype orelse return false;
    return !wdt.info().is_quantized;
}
