// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
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

fn isPowerOfTwoUsize(n: usize) bool {
    return n != 0 and (n & (n - 1)) == 0;
}

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

fn inferRegion(graph: *Graph, region_id: graph_mod.RegionId) InferError!graph_mod.Region {
    const idx: usize = @intCast(region_id);
    if (idx >= graph.regions.items.len) return InferError.InvalidGraph;
    const region: graph_mod.Region = graph.regions.items[idx];
    for (region.nodes) |node| {
        try inferNode(graph, node);
    }
    for (region.outputs) |oid| {
        const v = try getValue(graph, oid);
        try require(v.dtype != null);
        try require(v.shape.len != 0);
    }
    return region;
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

        .ElemwiseBinary => |eb| {
            const a = try getValue(graph, node.inputs[0]);
            const b = try getValue(graph, node.inputs[1]);
            try require(a.dtype != null and b.dtype != null);
            try require(a.shape.len != 0 and b.shape.len != 0);

            if (a.dtype.? != b.dtype.?) return InferError.DTypeMismatch;
            if (a.dtype.?.info().is_quantized) return InferError.Unsupported;

            if (a.shape.len != b.shape.len) return InferError.ShapeMismatch;
            for (a.shape, 0..) |d, i| if (d != b.shape[i]) return InferError.ShapeMismatch;

            if (eb.op.isComparison()) {
                // v1: comparisons operate on i32 and produce i32 {0,1}.
                if (a.dtype.? != .i32) return InferError.Unsupported;
                try setInferred(graph, node.output, .i32, a.shape);
            } else {
                try setInferred(graph, node.output, a.dtype.?, a.shape);
            }
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

        .RFFT => {
            const x = try getValue(graph, node.inputs[0]);
            try require(x.dtype != null and x.shape.len != 0);

            // v0: f32 only.
            if (x.dtype.? != .f32) return InferError.Unsupported;

            const n_fft: usize = x.shape[x.shape.len - 1];
            if (!isPowerOfTwoUsize(n_fft) or n_fft < 4) return InferError.ShapeMismatch;

            const out_shape: []usize = graph.arenaAlloc().alloc(usize, x.shape.len) catch return InferError.InvalidGraph;
            for (x.shape[0 .. x.shape.len - 1], 0..) |d, i| out_shape[i] = d;
            // One-sided bins = n_fft/2 + 1; packed real+imag = n_fft + 2.
            out_shape[x.shape.len - 1] = n_fft + 2;
            try setInferred(graph, node.output, .f32, out_shape);
        },

        .STFT => |st| {
            const signal = try getValue(graph, node.inputs[0]);
            const window = try getValue(graph, node.inputs[1]);
            try require(signal.dtype != null and window.dtype != null);

            // v0: f32 only.
            if (signal.dtype.? != .f32 or window.dtype.? != .f32) return InferError.Unsupported;

            if (signal.shape.len != 2) return InferError.RankMismatch;
            if (window.shape.len != 1) return InferError.RankMismatch;

            const n_fft: usize = st.n_fft;
            const hop: usize = st.hop_length;
            if (!isPowerOfTwoUsize(n_fft) or n_fft < 4) return InferError.ShapeMismatch;
            if (hop == 0) return InferError.InvalidGraph;
            if (window.shape[0] != n_fft) return InferError.ShapeMismatch;

            const batch: usize = signal.shape[0];
            const samples: usize = signal.shape[1];
            if (batch == 0 or samples == 0) return InferError.InvalidGraph;

            var num_frames: usize = 0;
            if (st.center) {
                num_frames = 1 + samples / hop;
            } else {
                if (samples < n_fft) return InferError.ShapeMismatch;
                num_frames = 1 + (samples - n_fft) / hop;
            }

            const out_shape: []usize = graph.arenaAlloc().alloc(usize, 3) catch return InferError.InvalidGraph;
            out_shape[0] = batch;
            out_shape[1] = num_frames;
            out_shape[2] = n_fft + 2;
            try setInferred(graph, node.output, .f32, out_shape);
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

            // Indices must be i32.
            if (indices.dtype.? != .i32) return InferError.DTypeMismatch;

            // Two legal dtype regimes (output dtype inferred per regime):
            //   - scalar table (f16/f32): output matches table.
            //   - q8_0 table (per-row quant_axis=1, blocks along D): output defaults to f32
            //     (the residual stream dtype in the Gemma pipeline). If a caller needs f16
            //     output they should insert an explicit cast; keeping inference deterministic
            //     avoids ambiguity about which scalar dtype to materialize.
            const out_dtype: types.DType = switch (table.dtype.?) {
                .f16, .f32 => table.dtype.?,
                .q8_0 => blk: {
                    if ((table.shape[1] % 32) != 0) return InferError.Unsupported;
                    break :blk .f32;
                },
                else => return InferError.Unsupported,
            };

            var out_shape: []usize = graph.arenaAlloc().alloc(usize, 3) catch return InferError.InvalidGraph;
            out_shape[0] = indices.shape[0];
            out_shape[1] = indices.shape[1];
            out_shape[2] = table.shape[1];
            try setInferred(graph, node.output, out_dtype, out_shape);
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

        .Cast => |ct| {
            const x = try getValue(graph, node.inputs[0]);
            try require(x.dtype != null);
            if (x.dtype.?.info().is_quantized) return InferError.Unsupported;
            if (ct.to_dtype.info().is_quantized) return InferError.Unsupported;
            // Supported: f16<->f32 and f32<->i32 (the latter for decode indices/counters).
            const ok: bool = (x.dtype.? == .f16 and ct.to_dtype == .f32) or
                (x.dtype.? == .f32 and ct.to_dtype == .f16) or
                (x.dtype.? == .f32 and ct.to_dtype == .i32) or
                (x.dtype.? == .i32 and ct.to_dtype == .f32) or
                (x.dtype.? == ct.to_dtype);
            if (!ok) return InferError.Unsupported;
            try setInferred(graph, node.output, ct.to_dtype, x.shape);
        },

        .If => |iff| {
            const cond = try getValue(graph, node.inputs[0]);
            try require(cond.dtype != null and cond.shape.len != 0);
            if (cond.dtype.? != .i32 or cond.shape.len != 1 or cond.shape[0] != 1) return InferError.DTypeMismatch;

            const then_region: graph_mod.Region = try inferRegion(graph, iff.then_region);
            const else_region: graph_mod.Region = try inferRegion(graph, iff.else_region);
            if (then_region.outputs.len != 1 or else_region.outputs.len != 1) return InferError.InvalidGraph;
            const then_v = try getValue(graph, then_region.outputs[0]);
            const else_v = try getValue(graph, else_region.outputs[0]);
            try require(then_v.dtype != null and else_v.dtype != null);
            if (then_v.dtype.? != else_v.dtype.?) return InferError.DTypeMismatch;
            if (then_v.shape.len != else_v.shape.len) return InferError.ShapeMismatch;
            for (then_v.shape, 0..) |dim, i| if (dim != else_v.shape[i]) return InferError.ShapeMismatch;
            try setInferred(graph, node.output, then_v.dtype.?, then_v.shape);
        },

        .Loop => |lp| {
            if (lp.static_max_trip_count == 0) return InferError.InvalidGraph;
            const n = node.inputs.len;
            if (n == 0 or node.extra_outputs.len + 1 != n) return InferError.InvalidGraph;
            const body_region: graph_mod.Region = try inferRegion(graph, lp.body_region);
            if (body_region.outputs.len != n) return InferError.InvalidGraph;
            var i: usize = 0;
            while (i < n) : (i += 1) {
                const init_v = try getValue(graph, node.inputs[i]);
                try require(init_v.dtype != null and init_v.shape.len != 0);
                const body_v = try getValue(graph, body_region.outputs[i]);
                try require(body_v.dtype != null and body_v.shape.len != 0);
                if (body_v.dtype.? != init_v.dtype.?) return InferError.DTypeMismatch;
                if (body_v.shape.len != init_v.shape.len) return InferError.ShapeMismatch;
                for (body_v.shape, 0..) |dim, j| if (dim != init_v.shape[j]) return InferError.ShapeMismatch;
                const out_id = if (i == 0) node.output else node.extra_outputs[i - 1];
                try setInferred(graph, out_id, init_v.dtype.?, init_v.shape);
            }
            if (lp.cond_carry) |ci| {
                if (ci >= n) return InferError.InvalidGraph;
                const cv = try getValue(graph, node.inputs[ci]);
                if (cv.dtype.? != .i32 or cv.shape.len != 1 or cv.shape[0] != 1) return InferError.DTypeMismatch;
            }
        },

        .MatMulNT => |mm| {
            const a = try getValue(graph, node.inputs[0]);
            const b = try getValue(graph, node.inputs[1]);
            try require(a.dtype != null and b.dtype != null);
            try require(a.shape.len >= 2 and b.shape.len == 2);

            if (!std.math.isFinite(mm.alpha) or !std.math.isFinite(mm.beta)) return InferError.InvalidGraph;

            // A: f32 (residual stream dtype). B: q8_0 with per-row blocks
            // (quant_axis=1) or plain f32 [N, K].
            if (a.dtype.? != .f32) return InferError.Unsupported;
            if (b.dtype.? != .q8_0 and b.dtype.? != .f32) return InferError.Unsupported;

            const k_a: usize = a.shape[a.shape.len - 1];
            const k_b: usize = b.shape[1];
            const n: usize = b.shape[0];
            if (k_a != k_b) return InferError.ShapeMismatch;
            if (b.dtype.? == .q8_0 and (k_a % 32) != 0) return InferError.Unsupported;

            var out_shape: []usize = graph.arenaAlloc().alloc(usize, a.shape.len) catch return InferError.InvalidGraph;
            var d: usize = 0;
            while (d + 1 < a.shape.len) : (d += 1) {
                out_shape[d] = a.shape[d];
            }
            out_shape[a.shape.len - 1] = n;
            try setInferred(graph, node.output, .f32, out_shape);
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

        .RelPosMHA => |attn| {
            const q = try getValue(graph, node.inputs[0]);
            const k = try getValue(graph, node.inputs[1]);
            const v = try getValue(graph, node.inputs[2]);
            const pos_emb = try getValue(graph, node.inputs[3]);
            const bu = try getValue(graph, node.inputs[4]);
            const bv = try getValue(graph, node.inputs[5]);

            try require(q.dtype != null and k.dtype != null and v.dtype != null);
            try require(pos_emb.dtype != null and bu.dtype != null and bv.dtype != null);

            // q,k,v:[B,H,T*,D]; pos_emb:[H,P,D]; pos_bias_u/_v:[H,D]
            try require(q.shape.len == 4 and k.shape.len == 4 and v.shape.len == 4);
            try require(pos_emb.shape.len == 3 and bu.shape.len == 2 and bv.shape.len == 2);

            if (q.dtype.? != .f32 or k.dtype.? != .f32 or v.dtype.? != .f32) return InferError.DTypeMismatch;
            if (pos_emb.dtype.? != .f32 or bu.dtype.? != .f32 or bv.dtype.? != .f32) return InferError.DTypeMismatch;

            if (attn.heads == 0) return InferError.InvalidGraph;
            if (!(attn.scale > 0.0) or !std.math.isFinite(attn.scale)) return InferError.InvalidGraph;

            // Layout [B, T*, H, D]; pos_emb [H, P, D]; biases [H, D].
            const B: usize = q.shape[0];
            const T_q: usize = q.shape[1];
            const H: usize = q.shape[2];
            const D: usize = q.shape[3];
            const T_kv: usize = k.shape[1];
            const P: usize = pos_emb.shape[1];

            if (H != attn.heads) return InferError.ShapeMismatch;
            if (k.shape[0] != B or k.shape[2] != H or k.shape[3] != D) return InferError.ShapeMismatch;
            if (v.shape[0] != B or v.shape[1] != T_kv or v.shape[2] != H or v.shape[3] != D) return InferError.ShapeMismatch;
            if (pos_emb.shape[0] != H or pos_emb.shape[2] != D) return InferError.ShapeMismatch;
            if (bu.shape[0] != H or bu.shape[1] != D) return InferError.ShapeMismatch;
            if (bv.shape[0] != H or bv.shape[1] != D) return InferError.ShapeMismatch;
            if (T_kv == 0 or P != 2 * T_kv - 1) return InferError.ShapeMismatch;
            if (T_q > T_kv) return InferError.ShapeMismatch;

            if (attn.has_mask) {
                try require(node.inputs.len == 7);
                const mask = try getValue(graph, node.inputs[6]);
                try require(mask.dtype != null and mask.shape.len == 2);
                if (mask.dtype.? != .f32) return InferError.DTypeMismatch;
                if (mask.shape[0] != T_q or mask.shape[1] != T_kv) return InferError.ShapeMismatch;
            }

            var out_shape: []usize = graph.arenaAlloc().alloc(usize, 4) catch return InferError.InvalidGraph;
            out_shape[0] = B;
            out_shape[1] = T_q;
            out_shape[2] = H;
            out_shape[3] = D;
            try setInferred(graph, node.output, q.dtype.?, out_shape);
        },

        .ArgMax => |am| {
            const x = try getValue(graph, node.inputs[0]);
            try require(x.dtype != null and x.shape.len >= 1);
            if (x.dtype.?.info().is_quantized) return InferError.Unsupported;
            const rank: usize = x.shape.len;
            var axis: usize = undefined;
            if (am.axis < 0) {
                const a = @as(i64, am.axis) + @as(i64, @intCast(rank));
                if (a < 0) return InferError.InvalidGraph;
                axis = @intCast(a);
            } else {
                axis = @intCast(am.axis);
            }
            if (axis >= rank) return InferError.InvalidGraph;
            if (x.shape[axis] == 0) return InferError.InvalidGraph;

            if (rank == 1) {
                var out_shape: []usize = graph.arenaAlloc().alloc(usize, 1) catch return InferError.InvalidGraph;
                out_shape[0] = 1;
                try setInferred(graph, node.output, .i32, out_shape);
            } else {
                var out_shape: []usize = graph.arenaAlloc().alloc(usize, rank - 1) catch return InferError.InvalidGraph;
                var dst: usize = 0;
                var src: usize = 0;
                while (src < rank) : (src += 1) {
                    if (src == axis) continue;
                    out_shape[dst] = x.shape[src];
                    dst += 1;
                }
                try setInferred(graph, node.output, .i32, out_shape);
            }
        },

        .ScatterRow => {
            const buf = try getValue(graph, node.inputs[0]);
            const idx = try getValue(graph, node.inputs[1]);
            const src = try getValue(graph, node.inputs[2]);
            try require(buf.dtype != null and idx.dtype != null and src.dtype != null);
            try require(buf.shape.len >= 1);
            if (buf.dtype.?.info().is_quantized) return InferError.Unsupported;
            if (idx.dtype.? != .i32) return InferError.DTypeMismatch;
            if (idx.shape.len != 1 or idx.shape[0] != 1) return InferError.ShapeMismatch;
            if (src.dtype.? != buf.dtype.?) return InferError.DTypeMismatch;

            var row_size: usize = 1;
            var d: usize = 1;
            while (d < buf.shape.len) : (d += 1) row_size *= buf.shape[d];
            var src_elems: usize = 1;
            for (src.shape) |sd| src_elems *= sd;
            if (src_elems != row_size) return InferError.ShapeMismatch;

            // In-place: output aliases buf shape/dtype.
            try setInferred(graph, node.output, buf.dtype.?, buf.shape);
        },
    }
}
