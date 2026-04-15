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

fn convOutDim(in_len: usize, kernel: usize, stride: usize, dilation: usize, pad_before: usize, pad_after: usize) InferError!usize {
    if (kernel == 0 or stride == 0 or dilation == 0) return InferError.InvalidGraph;
    const eff_kernel_sub1: usize = std.math.mul(usize, dilation, kernel - 1) catch return InferError.InvalidGraph;
    const eff_kernel: usize = std.math.add(usize, eff_kernel_sub1, 1) catch return InferError.InvalidGraph;
    const padded: usize = std.math.add(usize, std.math.add(usize, in_len, pad_before) catch return InferError.InvalidGraph, pad_after) catch return InferError.InvalidGraph;
    if (padded < eff_kernel) return InferError.ShapeMismatch;
    const numer: usize = padded - eff_kernel;
    return (numer / stride) + 1;
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
    if (!graph_mod.opInputCountValid(node.op, node.inputs.len)) return InferError.InvalidGraph;

    switch (node.op) {
        .MatMul => |mm| {
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

        .ElemwiseBinary => {
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

        .BroadcastLastDimBinary => {
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
            const a = try getValue(graph, node.inputs[0]);
            try require(a.dtype != null and a.shape.len != 0);
            if (a.dtype.?.info().is_quantized) return InferError.Unsupported;
            try setInferred(graph, node.output, a.dtype.?, a.shape);

            _ = u.op;
        },

        .Softmax => |sm| {
            const a = try getValue(graph, node.inputs[0]);
            try require(a.dtype != null and a.shape.len != 0);
            if (a.dtype.?.info().is_quantized) return InferError.Unsupported;
            _ = try normalizeAxis(sm.axis, a.shape.len);
            try setInferred(graph, node.output, a.dtype.?, a.shape);
        },

        .Conv1D => |cv| {
            const x = try getValue(graph, node.inputs[0]);
            const w = try getValue(graph, node.inputs[1]);
            try require(x.dtype != null and w.dtype != null);
            try require(x.shape.len >= 2 and w.shape.len == 3);

            if (x.dtype.? != .f32 or w.dtype.? != .f32) return InferError.Unsupported;
            if (x.dtype.? != w.dtype.?) return InferError.DTypeMismatch;

            const rank: usize = x.shape.len;
            const l_in: usize = x.shape[rank - 2];
            const c_in: usize = x.shape[rank - 1];

            const k: usize = w.shape[0];
            const c_in_g: usize = w.shape[1];
            const c_out: usize = w.shape[2];

            if (cv.groups == 0) return InferError.InvalidGraph;
            if (c_in % cv.groups != 0) return InferError.ShapeMismatch;
            if (c_out % cv.groups != 0) return InferError.ShapeMismatch;
            if (c_in_g * cv.groups != c_in) return InferError.ShapeMismatch;

            const l_out: usize = try convOutDim(l_in, k, cv.stride, cv.dilation, cv.pad_left, cv.pad_right);

            if (cv.pad_mode == .reflect) {
                if (l_in <= 1) return InferError.InvalidGraph;
                if (cv.pad_left >= l_in or cv.pad_right >= l_in) return InferError.InvalidGraph;
            }

            if (node.inputs.len == 3) {
                const b = try getValue(graph, node.inputs[2]);
                try require(b.dtype != null);
                if (b.dtype.? != .f32) return InferError.Unsupported;
                if (b.dtype.? != x.dtype.?) return InferError.DTypeMismatch;
                if (b.shape.len != 1 or b.shape[0] != c_out) return InferError.ShapeMismatch;
            }

            var out_shape: []usize = graph.arenaAlloc().alloc(usize, rank) catch return InferError.InvalidGraph;
            if (rank > 2) @memcpy(out_shape[0 .. rank - 2], x.shape[0 .. rank - 2]);
            out_shape[rank - 2] = l_out;
            out_shape[rank - 1] = c_out;
            try setInferred(graph, node.output, .f32, out_shape);
        },

        .Conv2D => |cv| {
            const x = try getValue(graph, node.inputs[0]);
            const w = try getValue(graph, node.inputs[1]);
            try require(x.dtype != null and w.dtype != null);
            try require(x.shape.len >= 3 and w.shape.len == 4);

            if (x.dtype.? != .f32 or w.dtype.? != .f32) return InferError.Unsupported;
            if (x.dtype.? != w.dtype.?) return InferError.DTypeMismatch;

            const rank: usize = x.shape.len;
            const h_in: usize = x.shape[rank - 3];
            const w_in: usize = x.shape[rank - 2];
            const c_in: usize = x.shape[rank - 1];

            const k_h: usize = w.shape[0];
            const k_w: usize = w.shape[1];
            const c_in_g: usize = w.shape[2];
            const c_out: usize = w.shape[3];

            if (cv.groups == 0) return InferError.InvalidGraph;
            if (c_in % cv.groups != 0) return InferError.ShapeMismatch;
            if (c_out % cv.groups != 0) return InferError.ShapeMismatch;
            if (c_in_g * cv.groups != c_in) return InferError.ShapeMismatch;

            const h_out: usize = try convOutDim(h_in, k_h, cv.stride_h, cv.dilation_h, cv.pad_top, cv.pad_bottom);
            const w_out: usize = try convOutDim(w_in, k_w, cv.stride_w, cv.dilation_w, cv.pad_left, cv.pad_right);

            if (cv.pad_mode == .reflect) {
                if (h_in <= 1 or w_in <= 1) return InferError.InvalidGraph;
                if (cv.pad_top >= h_in or cv.pad_bottom >= h_in) return InferError.InvalidGraph;
                if (cv.pad_left >= w_in or cv.pad_right >= w_in) return InferError.InvalidGraph;
            }

            if (node.inputs.len == 3) {
                const b = try getValue(graph, node.inputs[2]);
                try require(b.dtype != null);
                if (b.dtype.? != .f32) return InferError.Unsupported;
                if (b.dtype.? != x.dtype.?) return InferError.DTypeMismatch;
                if (b.shape.len != 1 or b.shape[0] != c_out) return InferError.ShapeMismatch;
            }

            var out_shape: []usize = graph.arenaAlloc().alloc(usize, rank) catch return InferError.InvalidGraph;
            if (rank > 3) @memcpy(out_shape[0 .. rank - 3], x.shape[0 .. rank - 3]);
            out_shape[rank - 3] = h_out;
            out_shape[rank - 2] = w_out;
            out_shape[rank - 1] = c_out;
            try setInferred(graph, node.output, .f32, out_shape);
        },

        .LayerNorm => |ln| {
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

        .MultiHeadAttentionCached => |attn| {
            const q = try getValue(graph, node.inputs[0]);
            const k_cache = try getValue(graph, node.inputs[1]);
            const v_cache = try getValue(graph, node.inputs[2]);
            const positions = try getValue(graph, node.inputs[3]);
            const end_index = try getValue(graph, node.inputs[4]);

            try require(q.dtype != null and k_cache.dtype != null and v_cache.dtype != null);
            try require(positions.dtype != null and end_index.dtype != null);

            // q:[B,L_q,H_q,D_k], k_cache:[B,H_kv,T,D_k], v_cache:[B,H_kv,T,D_v]
            // positions:[B,L_q], end_index:[B]
            try require(q.shape.len == 4 and k_cache.shape.len == 4 and v_cache.shape.len == 4);
            try require(positions.shape.len == 2 and end_index.shape.len == 1);

            if (q.dtype.?.info().is_quantized or k_cache.dtype.?.info().is_quantized or v_cache.dtype.?.info().is_quantized) {
                return InferError.Unsupported;
            }

            // v2 cached attention accepts mixed scalar dtypes for q/k/v,
            // while the kernel accumulates in f32 and emits f32.
            if (q.dtype.? != .f16 and q.dtype.? != .f32) return InferError.Unsupported;
            if (k_cache.dtype.? != .f16 and k_cache.dtype.? != .f32) return InferError.Unsupported;
            if (v_cache.dtype.? != .f16 and v_cache.dtype.? != .f32) return InferError.Unsupported;

            if (positions.dtype.? != .i32 or end_index.dtype.? != .i32) return InferError.DTypeMismatch;

            const bsz: usize = q.shape[0];
            const l_q: usize = q.shape[1];
            const h_q: usize = q.shape[2];
            const d_k: usize = q.shape[3];

            const k_b: usize = k_cache.shape[0];
            const h_kv: usize = k_cache.shape[1];
            const t_cap_k: usize = k_cache.shape[2];
            const d_k_cache: usize = k_cache.shape[3];

            const v_b: usize = v_cache.shape[0];
            const v_h_kv: usize = v_cache.shape[1];
            const t_cap_v: usize = v_cache.shape[2];
            const d_v: usize = v_cache.shape[3];

            if (k_b != bsz or v_b != bsz) return InferError.ShapeMismatch;
            if (h_kv == 0 or h_q == 0) return InferError.ShapeMismatch;
            if (v_h_kv != h_kv) return InferError.ShapeMismatch;
            if (h_q % h_kv != 0) return InferError.ShapeMismatch;
            if (d_k_cache != d_k) return InferError.ShapeMismatch;
            if (t_cap_k != t_cap_v) return InferError.ShapeMismatch;

            if (positions.shape[0] != bsz or positions.shape[1] != l_q) return InferError.ShapeMismatch;
            if (end_index.shape[0] != bsz) return InferError.ShapeMismatch;

            if (!(attn.scale > 0.0) or !std.math.isFinite(attn.scale)) return InferError.InvalidGraph;
            if (!std.math.isFinite(attn.attn_logits_soft_cap) or attn.attn_logits_soft_cap < 0.0) return InferError.InvalidGraph;
            _ = attn.causal;
            _ = attn.sliding_window;

            const out_shape: []usize = graph.arenaAlloc().alloc(usize, 4) catch return InferError.InvalidGraph;
            out_shape[0] = bsz;
            out_shape[1] = l_q;
            out_shape[2] = h_q;
            out_shape[3] = d_v;
            try setInferred(graph, node.output, .f32, out_shape);
        },

        .Reduce => |rr| {
            const a = try getValue(graph, node.inputs[0]);
            try require(a.dtype != null and a.shape.len != 0);
            if (a.dtype.?.info().is_quantized) return InferError.Unsupported;

            if (rr.axis) |axis_raw| {
                const rank: usize = a.shape.len;
                const axis: usize = try normalizeAxis(axis_raw, rank);

                if (rank == 1) {
                    // v0 keeps rank>=1 tensors; reducing the only axis yields [1].
                    try setInferred(graph, node.output, a.dtype.?, &[_]usize{1});
                } else {
                    var out_shape: []usize = graph.arenaAlloc().alloc(usize, rank - 1) catch return InferError.InvalidGraph;
                    var src_d: usize = 0;
                    var dst_d: usize = 0;
                    while (src_d < rank) : (src_d += 1) {
                        if (src_d == axis) continue;
                        out_shape[dst_d] = a.shape[src_d];
                        dst_d += 1;
                    }
                    try setInferred(graph, node.output, a.dtype.?, out_shape);
                }
            } else {
                // Reduce over all elements. Output is a 1-element rank-1 tensor for now.
                _ = try elemCount(a.shape);
                try setInferred(graph, node.output, a.dtype.?, &[_]usize{1});
            }
        },

        .Concat => |cc| {
            const first = try getValue(graph, node.inputs[0]);
            try require(first.dtype != null and first.shape.len != 0);
            if (first.dtype.?.info().is_quantized) return InferError.Unsupported;

            const rank: usize = first.shape.len;
            const axis: usize = try normalizeAxis(cc.axis, rank);

            var out_shape: []usize = graph.arenaAlloc().alloc(usize, rank) catch return InferError.InvalidGraph;
            @memcpy(out_shape, first.shape);

            var axis_sum: usize = first.shape[axis];

            var i: usize = 1;
            while (i < node.inputs.len) : (i += 1) {
                const v = try getValue(graph, node.inputs[i]);
                try require(v.dtype != null and v.shape.len != 0);
                if (v.dtype.? != first.dtype.?) return InferError.DTypeMismatch;
                if (v.dtype.?.info().is_quantized) return InferError.Unsupported;
                if (v.shape.len != rank) return InferError.RankMismatch;

                var d: usize = 0;
                while (d < rank) : (d += 1) {
                    if (d == axis) continue;
                    if (v.shape[d] != first.shape[d]) return InferError.ShapeMismatch;
                }

                axis_sum = std.math.add(usize, axis_sum, v.shape[axis]) catch return InferError.InvalidGraph;
            }

            out_shape[axis] = axis_sum;
            try setInferred(graph, node.output, first.dtype.?, out_shape);
        },

        .ComplexAbsMean => |cm| {
            const stft = try getValue(graph, node.inputs[0]);
            try require(stft.dtype != null and stft.shape.len != 0);

            // v0: scalar-only and non-quantized.
            if (stft.dtype.?.info().is_quantized) return InferError.Unsupported;
            if (stft.dtype.? != .f32 and stft.dtype.? != .f16) return InferError.Unsupported;

            if (stft.shape.len != 3) return InferError.RankMismatch;
            const batch: usize = stft.shape[0];
            const time: usize = stft.shape[1];
            const chans2: usize = stft.shape[2];
            if (batch == 0 or time == 0 or chans2 == 0) return InferError.InvalidGraph;
            if (chans2 % 2 != 0) return InferError.ShapeMismatch;

            const cutoff: usize = chans2 / 2;
            if (cm.out_channels == 0 or cm.out_channels > cutoff) return InferError.ShapeMismatch;

            const out_shape: []usize = graph.arenaAlloc().alloc(usize, 2) catch return InferError.InvalidGraph;
            out_shape[0] = batch;
            out_shape[1] = cm.out_channels;
            try setInferred(graph, node.output, stft.dtype.?, out_shape);
        },

        .LSTMCell => |lc| {
            const x = try getValue(graph, node.inputs[0]);
            const h_prev = try getValue(graph, node.inputs[1]);
            const c_prev = try getValue(graph, node.inputs[2]);
            const w_ih = try getValue(graph, node.inputs[3]);
            const w_hh = try getValue(graph, node.inputs[4]);

            try require(x.dtype != null and x.shape.len != 0);
            try require(h_prev.dtype != null and h_prev.shape.len != 0);
            try require(c_prev.dtype != null and c_prev.shape.len != 0);
            try require(w_ih.dtype != null and w_ih.shape.len != 0);
            try require(w_hh.dtype != null and w_hh.shape.len != 0);

            // v0: fused LSTM is scalar-only (f32/f16) and non-quantized.
            if (x.dtype.?.info().is_quantized) return InferError.Unsupported;
            if (x.dtype.? != .f32 and x.dtype.? != .f16) return InferError.Unsupported;
            if (h_prev.dtype.? != x.dtype.? or c_prev.dtype.? != x.dtype.? or w_ih.dtype.? != x.dtype.? or w_hh.dtype.? != x.dtype.?) return InferError.DTypeMismatch;

            if (x.shape.len != 2 or h_prev.shape.len != 2 or c_prev.shape.len != 2) return InferError.RankMismatch;
            if (w_ih.shape.len != 2 or w_hh.shape.len != 2) return InferError.RankMismatch;

            const batch: usize = x.shape[0];
            const input_size: usize = x.shape[1];
            const hidden: usize = h_prev.shape[1];
            if (batch == 0 or input_size == 0 or hidden == 0) return InferError.InvalidGraph;
            if (h_prev.shape[0] != batch or c_prev.shape[0] != batch) return InferError.ShapeMismatch;
            if (c_prev.shape[1] != hidden) return InferError.ShapeMismatch;

            const gate_dim: usize = std.math.mul(usize, hidden, 4) catch return InferError.InvalidGraph;
            if (w_ih.shape[0] != input_size or w_ih.shape[1] != gate_dim) return InferError.ShapeMismatch;
            if (w_hh.shape[0] != hidden or w_hh.shape[1] != gate_dim) return InferError.ShapeMismatch;

            if (lc.has_bias) {
                const b_ih = try getValue(graph, node.inputs[5]);
                const b_hh = try getValue(graph, node.inputs[6]);
                try require(b_ih.dtype != null and b_ih.shape.len != 0);
                try require(b_hh.dtype != null and b_hh.shape.len != 0);
                if (b_ih.dtype.? != x.dtype.? or b_hh.dtype.? != x.dtype.?) return InferError.DTypeMismatch;
                if (b_ih.shape.len != 1 or b_hh.shape.len != 1) return InferError.RankMismatch;
                if (b_ih.shape[0] != gate_dim or b_hh.shape[0] != gate_dim) return InferError.ShapeMismatch;
            }

            const out_hidden2: usize = std.math.mul(usize, hidden, 2) catch return InferError.InvalidGraph;
            const out_shape: []usize = graph.arenaAlloc().alloc(usize, 2) catch return InferError.InvalidGraph;
            out_shape[0] = batch;
            out_shape[1] = out_hidden2;
            try setInferred(graph, node.output, x.dtype.?, out_shape);
        },

        .Copy => {
            const a = try getValue(graph, node.inputs[0]);
            try require(a.dtype != null and a.shape.len != 0);
            try setInferred(graph, node.output, a.dtype.?, a.shape);
        },

        .ViewReshape => |vr| {
            const a = try getValue(graph, node.inputs[0]);
            try require(a.dtype != null and a.shape.len != 0);

            if (vr.new_shape.len == 0) return InferError.Unsupported;

            const a_elems: usize = try elemCount(a.shape);
            const b_elems: usize = try elemCount(vr.new_shape);
            if (a_elems != b_elems) return InferError.ShapeMismatch;

            try setInferred(graph, node.output, a.dtype.?, vr.new_shape);
        },

        .ViewSqueeze => |vs| {
            const a = try getValue(graph, node.inputs[0]);
            try require(a.dtype != null and a.shape.len != 0);

            const rank: usize = a.shape.len;
            if (vs.axis) |axis_raw| {
                const axis: usize = try normalizeAxis(axis_raw, rank);
                if (a.shape[axis] != 1) return InferError.ShapeMismatch;

                if (rank == 1) {
                    try setInferred(graph, node.output, a.dtype.?, &[_]usize{1});
                } else {
                    var out_shape: []usize = graph.arenaAlloc().alloc(usize, rank - 1) catch return InferError.InvalidGraph;
                    var src_d: usize = 0;
                    var dst_d: usize = 0;
                    while (src_d < rank) : (src_d += 1) {
                        if (src_d == axis) continue;
                        out_shape[dst_d] = a.shape[src_d];
                        dst_d += 1;
                    }
                    try setInferred(graph, node.output, a.dtype.?, out_shape);
                }
            } else {
                var out_shape_buf: [8]usize = undefined;
                var out_rank: usize = 0;
                for (a.shape) |d| {
                    if (d != 1) {
                        out_shape_buf[out_rank] = d;
                        out_rank += 1;
                    }
                }

                if (out_rank == 0) {
                    try setInferred(graph, node.output, a.dtype.?, &[_]usize{1});
                } else {
                    try setInferred(graph, node.output, a.dtype.?, out_shape_buf[0..out_rank]);
                }
            }
        },

        .ViewUnsqueeze => |vu| {
            const a = try getValue(graph, node.inputs[0]);
            try require(a.dtype != null and a.shape.len != 0);

            const out_rank: usize = a.shape.len + 1;
            if (out_rank > 8) return InferError.Unsupported;

            const out_rank_i32: i32 = @intCast(out_rank);
            var ax: i32 = vu.axis;
            if (ax < 0) ax += out_rank_i32;
            if (ax < 0 or ax >= out_rank_i32) return InferError.InvalidGraph;
            const axis: usize = @intCast(ax);

            var out_shape: []usize = graph.arenaAlloc().alloc(usize, out_rank) catch return InferError.InvalidGraph;
            var src_d: usize = 0;
            var dst_d: usize = 0;
            while (dst_d < out_rank) : (dst_d += 1) {
                if (dst_d == axis) {
                    out_shape[dst_d] = 1;
                } else {
                    out_shape[dst_d] = a.shape[src_d];
                    src_d += 1;
                }
            }

            try setInferred(graph, node.output, a.dtype.?, out_shape);
        },

        .ViewTranspose2D => {
            const a = try getValue(graph, node.inputs[0]);
            try require(a.dtype != null and a.shape.len == 2);
            try setInferred(graph, node.output, a.dtype.?, &[_]usize{ a.shape[1], a.shape[0] });
        },

        .ViewSliceND => |sl| {
            const a = try getValue(graph, node.inputs[0]);
            try require(a.dtype != null and a.shape.len != 0);

            if (sl.starts.len != a.shape.len or sl.lens.len != a.shape.len) return InferError.ShapeMismatch;
            var d: usize = 0;
            while (d < a.shape.len) : (d += 1) {
                if (sl.starts[d] + sl.lens[d] > a.shape[d]) return InferError.ShapeMismatch;
                if (sl.lens[d] == 0) return InferError.ShapeMismatch;
            }

            try setInferred(graph, node.output, a.dtype.?, sl.lens);
        },

        .GatherRows => {
            const table = try getValue(graph, node.inputs[0]);
            const indices = try getValue(graph, node.inputs[1]);

            try require(table.dtype != null and indices.dtype != null);
            try require(table.shape.len == 2);
            try require(indices.shape.len == 2);

            // Table: f16/f32 only (quant embeddings can be added later).
            if (table.dtype.? != .f16 and table.dtype.? != .f32) return InferError.Unsupported;

            // Indices must be i32.
            if (indices.dtype.? != .i32) return InferError.DTypeMismatch;

            var out_shape: []usize = graph.arenaAlloc().alloc(usize, 3) catch return InferError.InvalidGraph;
            out_shape[0] = indices.shape[0];
            out_shape[1] = indices.shape[1];
            out_shape[2] = table.shape[1];
            try setInferred(graph, node.output, table.dtype.?, out_shape);
        },

        .RoPE1D => |rp| {
            const x = try getValue(graph, node.inputs[0]);
            const positions = try getValue(graph, node.inputs[1]);

            try require(x.dtype != null and positions.dtype != null);
            try require(x.shape.len == 4);
            try require(positions.shape.len == 2);

            if (x.dtype.? != .f16 and x.dtype.? != .f32) return InferError.Unsupported;
            if (positions.dtype.? != .i32) return InferError.DTypeMismatch;

            if (positions.shape[0] != x.shape[0] or positions.shape[1] != x.shape[1]) return InferError.ShapeMismatch;

            if (!(rp.base_frequency > 0.0) or !std.math.isFinite(rp.base_frequency)) return InferError.InvalidGraph;
            if (!(rp.scale_factor > 0.0) or !std.math.isFinite(rp.scale_factor)) return InferError.InvalidGraph;
            if (!std.math.isFinite(rp.rope_proportion)) return InferError.InvalidGraph;
            if (rp.rope_proportion < 0.0 or rp.rope_proportion > 1.0) return InferError.InvalidGraph;

            try setInferred(graph, node.output, x.dtype.?, x.shape);
        },

        .KVCacheAppend => {
            const cache = try getValue(graph, node.inputs[0]);
            const new_kv = try getValue(graph, node.inputs[1]);
            const end_index = try getValue(graph, node.inputs[2]);

            try require(cache.dtype != null and new_kv.dtype != null and end_index.dtype != null);
            try require(cache.shape.len == 4 and new_kv.shape.len == 4 and end_index.shape.len == 1);

            if (cache.dtype.? != new_kv.dtype.?) return InferError.DTypeMismatch;
            if (cache.dtype.? != .f32 and cache.dtype.? != .f16) return InferError.Unsupported;
            if (end_index.dtype.? != .i32) return InferError.DTypeMismatch;

            if (cache.shape[0] != new_kv.shape[0]) return InferError.ShapeMismatch;
            if (cache.shape[1] != new_kv.shape[1]) return InferError.ShapeMismatch;
            if (cache.shape[3] != new_kv.shape[3]) return InferError.ShapeMismatch;
            if (end_index.shape[0] != cache.shape[0]) return InferError.ShapeMismatch;

            // In-place append semantics: output aliases cache shape/dtype.
            try setInferred(graph, node.output, cache.dtype.?, cache.shape);
        },
    }
}
