// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Pointwise (1x1) Conv1D -> MatMul lowering.
//!
//! A 1x1 conv with groups==1, stride==1, dilation==1 and no padding is exactly a
//! matmul: `out[b,l,co] = sum_ci x[b,l,ci] * w[0,ci,co]`. The conv weight (aion
//! layout `[k=1, c_in, c_out]`) is byte-identical to a MatMul B `[1, K, N]`, and
//! `x [B,L,K] @ B [1,K,N] -> [B,L,N]` via matmul's leading-dim broadcast — so the
//! rewrite reuses the same weight/x tensors and the same output value (no shape
//! re-inference needed). This routes pointwise convs onto the autotuned GEMM/GEMV
//! path instead of the direct conv kernel.
//!
//! This pass changes graph SHAPE only; it does NOT quantize. Weight dtype is
//! preserved (an f32 conv weight stays f32) — quantizing a pointwise projection
//! is a convert-time decision (store it as a `MatMul` q8 B up front). Accordingly
//! the pass skips already-quantized conv weights (which never reach a Conv node
//! today) so it can't emit a MatMul B with an incompatible quant layout.
//!
//! Runs AFTER shape inference inside `compileGraph` (output shapes are known) but
//! BEFORE the value->tensor map is sized, so any value it appends is compiled
//! normally.

const std = @import("std");

const graph_mod = @import("../graph.zig");
const types = @import("../../backend/types.zig");

const Graph = graph_mod.Graph;
const Node = graph_mod.Node;
const ValueId = graph_mod.ValueId;

pub const Error = graph_mod.GraphError;

pub fn run(allocator: std.mem.Allocator, graph: *Graph) Error!void {
    var any = false;
    for (graph.nodes.items) |node| {
        if (qualifies(graph, node)) {
            any = true;
            break;
        }
    }
    if (!any) return;

    var new_nodes: std.ArrayList(Node) = .empty;
    errdefer new_nodes.deinit(allocator);

    for (graph.nodes.items) |node| {
        if (!qualifies(graph, node)) {
            try new_nodes.append(allocator, node);
            continue;
        }

        const x = node.inputs[0];
        const w = node.inputs[1];
        const mm: graph_mod.Op = .{ .MatMul = .{ .alpha = 1.0, .beta = 0.0 } };

        if (node.inputs.len == 2) {
            // No bias: pure op swap — same inputs, same output value.
            try new_nodes.append(allocator, .{
                .op = mm,
                .inputs = try arenaIds(graph, &[_]ValueId{ x, w }),
                .output = node.output,
                .extra_outputs = node.extra_outputs,
            });
        } else {
            // With bias [c_out]: matmul -> tmp, then broadcast-add bias -> output.
            const out_v = graph.values.items[@intCast(node.output)];
            const tmp = try newValue(graph, out_v.dtype.?, out_v.shape);
            try new_nodes.append(allocator, .{
                .op = mm,
                .inputs = try arenaIds(graph, &[_]ValueId{ x, w }),
                .output = tmp,
            });
            try new_nodes.append(allocator, .{
                .op = .{ .BroadcastLastDimBinary = .{ .op = .add } },
                .inputs = try arenaIds(graph, &[_]ValueId{ tmp, node.inputs[2] }),
                .output = node.output,
                .extra_outputs = node.extra_outputs,
            });
        }
    }

    graph.nodes.deinit(allocator);
    graph.nodes = new_nodes;
}

/// A Conv1D is a pure matmul iff it is 1x1, groups==1, unit stride/dilation, and
/// unpadded, with rank-3 x/w and a non-quantized weight.
fn qualifies(graph: *const Graph, node: Node) bool {
    const cv = switch (node.op) {
        .Conv1D => |c| c,
        else => return false,
    };
    if (cv.groups != 1 or cv.stride != 1 or cv.dilation != 1) return false;
    if (cv.pad_left != 0 or cv.pad_right != 0) return false;
    if (node.inputs.len < 2 or node.inputs.len > 3) return false;

    const xv = graph.values.items[@intCast(node.inputs[0])];
    const wv = graph.values.items[@intCast(node.inputs[1])];
    if (xv.shape.len != 3 or wv.shape.len != 3) return false;
    if (wv.shape[0] != 1) return false; // kernel size 1 (pointwise)

    const wdt = wv.dtype orelse return false;
    if (wdt.info().is_quantized) return false; // quantized pointwise is emitted as MatMul upstream
    return true;
}

fn newValue(graph: *Graph, dtype: types.DType, shape: []const usize) Error!ValueId {
    const id = try graph.addValue();
    const sh = try graph.dupeShape(shape);
    graph.values.items[@intCast(id)] = .{ .dtype = dtype, .shape = sh };
    return id;
}

fn arenaIds(graph: *Graph, ids: []const ValueId) Error![]ValueId {
    const out = graph.arenaAlloc().alloc(ValueId, ids.len) catch return Error.OutOfMemory;
    @memcpy(out, ids);
    return out;
}
