const std = @import("std");

const graph_mod = @import("graph.zig");
const types = @import("../backend/types.zig");
const backend_utils = @import("../backend/utils.zig");

pub const Graph = graph_mod.Graph;
pub const Node = graph_mod.Node;
pub const Op = graph_mod.Op;
pub const ValueId = graph_mod.ValueId;

pub const DType = types.DType;
pub const UnaryOp = types.UnaryOp;

pub const InferError = error{
    InvalidGraph,
    Unsupported,
    RankMismatch,
    ShapeMismatch,
    DTypeMismatch,
};

fn require(cond: bool) InferError!void {
    if (!cond) return InferError.InvalidGraph;
}

fn getValue(graph: *const Graph, id: ValueId) InferError!graph_mod.Value {
    const idx: usize = @intCast(id);
    if (idx >= graph.values.items.len) return InferError.InvalidGraph;
    return graph.values.items[idx];
}

fn setInferred(graph: *Graph, id: ValueId, dtype: DType, shape: []const usize) InferError!void {
    const idx: usize = @intCast(id);
    if (idx >= graph.values.items.len) return InferError.InvalidGraph;

    var v: graph_mod.Value = graph.values.items[idx];

    if (v.dtype) |dt| {
        if (dt != dtype) return InferError.DTypeMismatch;
    } else {
        v.dtype = dtype;
    }

    if (v.shape.len != 0) {
        if (v.shape.len != shape.len) return InferError.ShapeMismatch;
        for (v.shape, 0..) |d, i| {
            if (d != shape[i]) return InferError.ShapeMismatch;
        }
    } else {
        // shapes are arena-owned already (for view reshape), or derived from other arena-owned shapes.
        // We allocate a copy here so every value has arena-owned storage.
        const sh: []usize = graph.arenaAlloc().alloc(usize, shape.len) catch return InferError.InvalidGraph;
        @memcpy(sh, shape);
        v.shape = sh;
    }

    graph.values.items[idx] = v;
}

fn elemCount(shape: []const usize) InferError!usize {
    return backend_utils.elemCount(shape) catch return InferError.InvalidGraph;
}

pub fn infer(graph: *Graph) InferError!void {
    // Ensure declared inputs are valid.
    for (graph.values.items) |v| {
        if (v.shape.len != 0) {
            if (v.shape.len == 0 or v.shape.len > 2) return InferError.Unsupported;
            for (v.shape) |d| if (d == 0) return InferError.InvalidGraph;
        }
    }

    // Nodes are assumed to be in topological order (builder appends in that order).
    for (graph.nodes.items) |node| {
        try inferNode(graph, node);
    }

    // Outputs must be inferred.
    for (graph.outputs.items) |oid| {
        const v = try getValue(graph, oid);
        try require(v.dtype != null);
        try require(v.shape.len != 0);
    }
}

fn inferNode(graph: *Graph, node: Node) InferError!void {
    switch (node.op) {
        .MatMul => |mm| {
            try require(node.inputs.len == 2);
            const a = try getValue(graph, node.inputs[0]);
            const b = try getValue(graph, node.inputs[1]);
            try require(a.dtype != null and b.dtype != null);
            try require(a.shape.len == 2 and b.shape.len == 2);

            const a_dt: DType = a.dtype.?;
            const b_dt: DType = b.dtype.?;

            // Shape.
            const m: usize = a.shape[0];
            const k: usize = a.shape[1];
            const k_b: usize = b.shape[0];
            const n: usize = b.shape[1];
            if (k != k_b) return InferError.ShapeMismatch;

            // DType routing (v0 strict):
            // - If B is quant, A must be f32 and output is f32.
            // - If B is f32, A must be f32 and output is f32.
            // - If B is f16, A must be f16 and output is f16 by default.
            //   If the output dtype is pre-constrained to f32, allow f16->f32 promotion.
            const out_v = try getValue(graph, node.output);
            if (b_dt.info().is_quantized) {
                if (a_dt != .f32) return InferError.DTypeMismatch;
                try setInferred(graph, node.output, .f32, &[_]usize{ m, n });
            } else if (b_dt == .f32) {
                if (a_dt != .f32) return InferError.DTypeMismatch;
                try setInferred(graph, node.output, .f32, &[_]usize{ m, n });
            } else if (b_dt == .f16) {
                if (a_dt != .f16) return InferError.DTypeMismatch;
                const out_dt: DType = out_v.dtype orelse .f16;
                if (out_dt != .f16 and out_dt != .f32) return InferError.DTypeMismatch;
                try setInferred(graph, node.output, out_dt, &[_]usize{ m, n });
            } else {
                return InferError.Unsupported;
            }

            _ = mm; // alpha/beta handled in lowering.
        },

        .ElemwiseBinary => |_| {
            try require(node.inputs.len == 2);
            const a = try getValue(graph, node.inputs[0]);
            const b = try getValue(graph, node.inputs[1]);
            try require(a.dtype != null and b.dtype != null);
            try require(a.shape.len != 0 and b.shape.len != 0);

            if (a.dtype.? != b.dtype.?) return InferError.DTypeMismatch;
            if (a.dtype.?.info().is_quantized) return InferError.Unsupported;

            if (a.shape.len != b.shape.len) return InferError.ShapeMismatch;
            for (a.shape, 0..) |d, i| if (d != b.shape[i]) return InferError.ShapeMismatch;

            try setInferred(graph, node.output, a.dtype.?, a.shape);
        },

        .BroadcastLastDimBinary => |_| {
            try require(node.inputs.len == 2);
            const a = try getValue(graph, node.inputs[0]);
            const b = try getValue(graph, node.inputs[1]);
            try require(a.dtype != null and b.dtype != null);
            try require(a.shape.len != 0 and b.shape.len != 0);

            if (a.dtype.? != b.dtype.?) return InferError.DTypeMismatch;
            if (a.dtype.?.info().is_quantized) return InferError.Unsupported;

            if (b.shape.len != 1) return InferError.RankMismatch;
            const last_dim: usize = a.shape[a.shape.len - 1];
            if (b.shape[0] != last_dim) return InferError.ShapeMismatch;

            try setInferred(graph, node.output, a.dtype.?, a.shape);
        },

        .Unary => |u| {
            try require(node.inputs.len == 1);
            const a = try getValue(graph, node.inputs[0]);
            try require(a.dtype != null and a.shape.len != 0);
            if (a.dtype.?.info().is_quantized) return InferError.Unsupported;
            try setInferred(graph, node.output, a.dtype.?, a.shape);

            _ = u.op;
        },

        .Softmax => {
            try require(node.inputs.len == 1);
            const a = try getValue(graph, node.inputs[0]);
            try require(a.dtype != null and a.shape.len != 0);
            if (a.dtype.?.info().is_quantized) return InferError.Unsupported;
            // v0: softmax over last dimension; preserve shape/dtype.
            try setInferred(graph, node.output, a.dtype.?, a.shape);
        },

        .LayerNorm => |_| {
            try require(node.inputs.len == 3);
            const x = try getValue(graph, node.inputs[0]);
            const gamma = try getValue(graph, node.inputs[1]);
            const beta = try getValue(graph, node.inputs[2]);

            try require(x.dtype != null and gamma.dtype != null and beta.dtype != null);
            try require(x.shape.len == 2);
            try require(gamma.shape.len == 1 and beta.shape.len == 1);

            if (x.dtype.? != gamma.dtype.? or x.dtype.? != beta.dtype.?) return InferError.DTypeMismatch;
            if (x.dtype.?.info().is_quantized) return InferError.Unsupported;

            const n: usize = x.shape[1];
            if (gamma.shape[0] != n or beta.shape[0] != n) return InferError.ShapeMismatch;

            try setInferred(graph, node.output, x.dtype.?, x.shape);
        },

        .RMSNorm => |_| {
            try require(node.inputs.len == 3);
            const x = try getValue(graph, node.inputs[0]);
            const gamma = try getValue(graph, node.inputs[1]);
            const beta = try getValue(graph, node.inputs[2]);

            try require(x.dtype != null and gamma.dtype != null and beta.dtype != null);
            try require(x.shape.len == 2);
            try require(gamma.shape.len == 1 and beta.shape.len == 1);

            if (x.dtype.? != gamma.dtype.? or x.dtype.? != beta.dtype.?) return InferError.DTypeMismatch;
            if (x.dtype.?.info().is_quantized) return InferError.Unsupported;

            const n: usize = x.shape[1];
            if (gamma.shape[0] != n or beta.shape[0] != n) return InferError.ShapeMismatch;

            try setInferred(graph, node.output, x.dtype.?, x.shape);
        },

        .Reduce => |_| {
            try require(node.inputs.len == 1);
            const a = try getValue(graph, node.inputs[0]);
            try require(a.dtype != null and a.shape.len != 0);
            if (a.dtype.?.info().is_quantized) return InferError.Unsupported;

            // Reduce over all elements. Output is a 1-element rank-1 tensor for now.
            _ = try elemCount(a.shape);
            try setInferred(graph, node.output, a.dtype.?, &[_]usize{1});
        },

        .Copy => {
            try require(node.inputs.len == 1);
            const a = try getValue(graph, node.inputs[0]);
            try require(a.dtype != null and a.shape.len != 0);
            try setInferred(graph, node.output, a.dtype.?, a.shape);
        },

        .ViewReshape => |vr| {
            try require(node.inputs.len == 1);
            const a = try getValue(graph, node.inputs[0]);
            try require(a.dtype != null and a.shape.len != 0);

            if (vr.new_shape.len == 0 or vr.new_shape.len > 2) return InferError.Unsupported;

            const a_elems: usize = try elemCount(a.shape);
            const b_elems: usize = try elemCount(vr.new_shape);
            if (a_elems != b_elems) return InferError.ShapeMismatch;

            try setInferred(graph, node.output, a.dtype.?, vr.new_shape);
        },

        .ViewTranspose2D => {
            try require(node.inputs.len == 1);
            const a = try getValue(graph, node.inputs[0]);
            try require(a.dtype != null and a.shape.len == 2);
            try setInferred(graph, node.output, a.dtype.?, &[_]usize{ a.shape[1], a.shape[0] });
        },

        .ViewSlice2D => |sl| {
            try require(node.inputs.len == 1);
            const a = try getValue(graph, node.inputs[0]);
            try require(a.dtype != null and a.shape.len == 2);

            if (sl.start0 + sl.len0 > a.shape[0]) return InferError.ShapeMismatch;
            if (sl.start1 + sl.len1 > a.shape[1]) return InferError.ShapeMismatch;

            try setInferred(graph, node.output, a.dtype.?, &[_]usize{ sl.len0, sl.len1 });
        },
    }
}
