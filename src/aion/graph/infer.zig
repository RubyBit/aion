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

fn normalizeAxis(axis: i32, rank: usize) InferError!usize {
    if (rank == 0) return InferError.InvalidGraph;
    const r_i32: i32 = @intCast(rank);
    var ax: i32 = axis;
    if (ax < 0) ax += r_i32;
    if (ax < 0 or ax >= r_i32) return InferError.InvalidGraph;
    return @intCast(ax);
}

pub fn infer(graph: *Graph) InferError!void {
    // Ensure declared inputs are valid.
    for (graph.values.items) |v| {
        if (v.shape.len != 0) {
            if (v.shape.len == 0) return InferError.Unsupported;
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
            try require(a.shape.len >= 2 and b.shape.len >= 2);
            if (a.shape.len != b.shape.len) return InferError.RankMismatch;

            const a_dt: DType = a.dtype.?;
            const b_dt: DType = b.dtype.?;

            const rank: usize = a.shape.len;

            // Shape (batched with broadcast):
            // - batch dims can be equal or one side can be 1
            // - output batch dims follow the max of (a,b)
            var out_shape: []usize = graph.arenaAlloc().alloc(usize, rank) catch return InferError.InvalidGraph;
            var d: usize = 0;
            while (d + 2 < rank) : (d += 1) {
                const ad: usize = a.shape[d];
                const bd: usize = b.shape[d];
                if (ad != bd and ad != 1 and bd != 1) return InferError.ShapeMismatch;
                out_shape[d] = if (ad >= bd) ad else bd;
            }

            const m: usize = a.shape[rank - 2];
            const k: usize = a.shape[rank - 1];
            const k_b: usize = b.shape[rank - 2];
            const n: usize = b.shape[rank - 1];
            if (k != k_b) return InferError.ShapeMismatch;

            // DType routing (v0 strict):
            // - If B is quant, A must be f32 and output is f32.
            // - If B is f32, A must be f32 and output is f32.
            // - If B is f16, A must be f16 and output is f16 by default.
            //   If the output dtype is pre-constrained to f32, allow f16->f32 promotion.
            const out_v = try getValue(graph, node.output);
            if (b_dt.info().is_quantized) {
                if (a_dt != .f32) return InferError.DTypeMismatch;
                out_shape[rank - 2] = m;
                out_shape[rank - 1] = n;
                try setInferred(graph, node.output, .f32, out_shape);
            } else if (b_dt == .f32) {
                if (a_dt != .f32) return InferError.DTypeMismatch;
                out_shape[rank - 2] = m;
                out_shape[rank - 1] = n;
                try setInferred(graph, node.output, .f32, out_shape);
            } else if (b_dt == .f16) {
                if (a_dt != .f16) return InferError.DTypeMismatch;
                const out_dt: DType = out_v.dtype orelse .f16;
                if (out_dt != .f16 and out_dt != .f32) return InferError.DTypeMismatch;
                out_shape[rank - 2] = m;
                out_shape[rank - 1] = n;
                try setInferred(graph, node.output, out_dt, out_shape);
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

        .Softmax => |sm| {
            try require(node.inputs.len == 1);
            const a = try getValue(graph, node.inputs[0]);
            try require(a.dtype != null and a.shape.len != 0);
            if (a.dtype.?.info().is_quantized) return InferError.Unsupported;
            _ = try normalizeAxis(sm.axis, a.shape.len);
            try setInferred(graph, node.output, a.dtype.?, a.shape);
        },

        .LayerNorm => |ln| {
            try require(node.inputs.len == 3);
            const x = try getValue(graph, node.inputs[0]);
            const gamma = try getValue(graph, node.inputs[1]);
            const beta = try getValue(graph, node.inputs[2]);

            try require(x.dtype != null and gamma.dtype != null and beta.dtype != null);
            try require(x.shape.len != 0);
            try require(gamma.shape.len != 0 and beta.shape.len != 0);

            if (x.dtype.? != gamma.dtype.? or x.dtype.? != beta.dtype.?) return InferError.DTypeMismatch;
            if (x.dtype.?.info().is_quantized) return InferError.Unsupported;

            if (ln.normalized_shape.len == 0) return InferError.InvalidGraph;
            if (gamma.shape.len != ln.normalized_shape.len or beta.shape.len != ln.normalized_shape.len) return InferError.ShapeMismatch;
            if (x.shape.len < ln.normalized_shape.len) return InferError.RankMismatch;

            // normalized_shape must match trailing dims of x and gamma/beta shapes.
            const norm_len: usize = ln.normalized_shape.len;
            var d: usize = 0;
            while (d < norm_len) : (d += 1) {
                const x_dim: usize = x.shape[x.shape.len - norm_len + d];
                if (x_dim != ln.normalized_shape[d]) return InferError.ShapeMismatch;
                if (gamma.shape[d] != ln.normalized_shape[d] or beta.shape[d] != ln.normalized_shape[d]) return InferError.ShapeMismatch;
            }
            try setInferred(graph, node.output, x.dtype.?, x.shape);
        },

        .RMSNorm => |rn| {
            try require(node.inputs.len == 3);
            const x = try getValue(graph, node.inputs[0]);
            const gamma = try getValue(graph, node.inputs[1]);
            const beta = try getValue(graph, node.inputs[2]);

            try require(x.dtype != null and gamma.dtype != null and beta.dtype != null);
            try require(x.shape.len != 0);
            try require(gamma.shape.len != 0 and beta.shape.len != 0);

            if (x.dtype.? != gamma.dtype.? or x.dtype.? != beta.dtype.?) return InferError.DTypeMismatch;
            if (x.dtype.?.info().is_quantized) return InferError.Unsupported;

            if (rn.normalized_shape.len == 0) return InferError.InvalidGraph;
            if (gamma.shape.len != rn.normalized_shape.len or beta.shape.len != rn.normalized_shape.len) return InferError.ShapeMismatch;
            if (x.shape.len < rn.normalized_shape.len) return InferError.RankMismatch;

            const norm_len: usize = rn.normalized_shape.len;
            var d: usize = 0;
            while (d < norm_len) : (d += 1) {
                const x_dim: usize = x.shape[x.shape.len - norm_len + d];
                if (x_dim != rn.normalized_shape[d]) return InferError.ShapeMismatch;
                if (gamma.shape[d] != rn.normalized_shape[d] or beta.shape[d] != rn.normalized_shape[d]) return InferError.ShapeMismatch;
            }
            try setInferred(graph, node.output, x.dtype.?, x.shape);
        },

        .Attention => |attn| {
            try require(node.inputs.len == 3);
            const q = try getValue(graph, node.inputs[0]);
            const k = try getValue(graph, node.inputs[1]);
            const v = try getValue(graph, node.inputs[2]);

            try require(q.dtype != null and k.dtype != null and v.dtype != null);
            try require(q.shape.len >= 2 and k.shape.len >= 2 and v.shape.len >= 2);
            if (q.shape.len != k.shape.len or q.shape.len != v.shape.len) return InferError.RankMismatch;

            if (q.dtype.? != k.dtype.? or q.dtype.? != v.dtype.?) return InferError.DTypeMismatch;
            if (q.dtype.?.info().is_quantized) return InferError.Unsupported;

            const rank: usize = q.shape.len;
            const lead_dims: usize = rank - 2;
            var d: usize = 0;
            while (d < lead_dims) : (d += 1) {
                if (q.shape[d] != k.shape[d] or q.shape[d] != v.shape[d]) return InferError.ShapeMismatch;
            }

            // q:[..., m, dk], k:[..., n, dk], v:[..., n, dv] -> out:[..., m, dv]
            if (q.shape[rank - 1] != k.shape[rank - 1]) return InferError.ShapeMismatch;
            if (k.shape[rank - 2] != v.shape[rank - 2]) return InferError.ShapeMismatch;

            if (!(attn.scale > 0.0) or !std.math.isFinite(attn.scale)) return InferError.InvalidGraph;
            _ = attn.causal;

            var out_shape: []usize = graph.arenaAlloc().alloc(usize, rank) catch return InferError.InvalidGraph;
            d = 0;
            while (d < lead_dims) : (d += 1) {
                out_shape[d] = q.shape[d];
            }
            out_shape[rank - 2] = q.shape[rank - 2];
            out_shape[rank - 1] = v.shape[rank - 1];
            try setInferred(graph, node.output, q.dtype.?, out_shape);
        },

        .MultiHeadAttention => |attn| {
            try require(node.inputs.len == 3);
            const q = try getValue(graph, node.inputs[0]);
            const k = try getValue(graph, node.inputs[1]);
            const v = try getValue(graph, node.inputs[2]);

            try require(q.dtype != null and k.dtype != null and v.dtype != null);
            try require(q.shape.len >= 3 and k.shape.len >= 3 and v.shape.len >= 3);
            if (q.shape.len != k.shape.len or q.shape.len != v.shape.len) return InferError.RankMismatch;

            if (q.dtype.? != k.dtype.? or q.dtype.? != v.dtype.?) return InferError.DTypeMismatch;
            if (q.dtype.?.info().is_quantized) return InferError.Unsupported;

            if (attn.heads == 0) return InferError.InvalidGraph;

            // Separate head dim: q:[..., h, m, dk], k:[..., h, n, dk], v:[..., h, n, dv]
            const rank: usize = q.shape.len;
            const lead_dims: usize = rank - 3;
            var d: usize = 0;
            while (d < lead_dims) : (d += 1) {
                if (q.shape[d] != k.shape[d] or q.shape[d] != v.shape[d]) return InferError.ShapeMismatch;
            }
            const head_dim: usize = rank - 3;
            if (q.shape[head_dim] != attn.heads) return InferError.ShapeMismatch;
            if (k.shape[head_dim] != attn.heads or v.shape[head_dim] != attn.heads) return InferError.ShapeMismatch;

            if (q.shape[rank - 1] != k.shape[rank - 1]) return InferError.ShapeMismatch;
            if (k.shape[rank - 2] != v.shape[rank - 2]) return InferError.ShapeMismatch;

            if (!(attn.scale > 0.0) or !std.math.isFinite(attn.scale)) return InferError.InvalidGraph;
            _ = attn.causal;

            var out_shape: []usize = graph.arenaAlloc().alloc(usize, rank) catch return InferError.InvalidGraph;
            d = 0;
            while (d < lead_dims) : (d += 1) {
                out_shape[d] = q.shape[d];
            }
            out_shape[head_dim] = attn.heads;
            out_shape[rank - 2] = q.shape[rank - 2];
            out_shape[rank - 1] = v.shape[rank - 1];
            try setInferred(graph, node.output, q.dtype.?, out_shape);
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
