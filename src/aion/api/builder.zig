const std = @import("std");

const graph_mod = @import("../graph/graph.zig");
const types = @import("../backend/types.zig");

const api_tensor = @import("tensor.zig");

pub const ValueId = graph_mod.ValueId;

/// Lightweight handle to a value produced/consumed by the builder.
///
/// This is intentionally opaque to keep the underlying graph hidden.
pub const TensorRef = struct {
    value: ValueId,
};

/// Graph-hidden model builder.
///
/// Under the hood, this builds an `aion.graph.Graph`.
pub const Builder = struct {
    allocator: std.mem.Allocator,
    graph: graph_mod.Graph,

    // Optional per-value names for diagnostics/debugging.
    value_names: std.ArrayListUnmanaged(?[]const u8) = .{},

    const Self = @This();

    pub const Error = graph_mod.GraphError;

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{ .allocator = allocator, .graph = graph_mod.Graph.init(allocator), .value_names = .{} };
    }

    pub fn deinit(self: *Self) void {
        self.value_names.deinit(self.allocator);
        self.graph.deinit();
        self.* = undefined;
    }

    pub fn innerGraph(self: *Self) *graph_mod.Graph {
        return &self.graph;
    }

    fn ensureNameSlot(self: *Self, vid: ValueId) Error!void {
        const idx: usize = @intCast(vid);
        if (idx < self.value_names.items.len) return;

        const old_len: usize = self.value_names.items.len;
        const new_len: usize = idx + 1;
        try self.value_names.ensureTotalCapacity(self.allocator, new_len);
        self.value_names.items.len = new_len;
        var i: usize = old_len;
        while (i < new_len) : (i += 1) {
            self.value_names.items[i] = null;
        }
    }

    pub fn name(self: *Self, t: TensorRef, value_name: []const u8) Error!TensorRef {
        try self.ensureNameSlot(t.value);
        const duped: []u8 = self.graph.arenaAlloc().alloc(u8, value_name.len) catch return Error.OutOfMemory;
        @memcpy(duped, value_name);
        self.value_names.items[@intCast(t.value)] = duped;
        return t;
    }

    /// Bind an existing owned tensor as a graph input.
    pub fn param(self: *Self, t: api_tensor.Tensor) Error!TensorRef {
        const v: ValueId = try self.graph.addInput(t.dtype, t.shape);
        try self.graph.bindExternal(v, @intCast(t.id));
        return .{ .value = v };
    }

    pub fn input(self: *Self, dtype: types.DType, shape: []const usize) Error!TensorRef {
        const v: ValueId = try self.graph.addInput(dtype, shape);
        return .{ .value = v };
    }

    pub fn matmul(self: *Self, a: TensorRef, b: TensorRef, alpha: f32, beta: f32) Error!TensorRef {
        const out: ValueId = try self.graph.addMatMul(a.value, b.value, alpha, beta);
        return .{ .value = out };
    }

    pub fn add(self: *Self, a: TensorRef, b: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addElemwiseBinary(.add, a.value, b.value);
        return .{ .value = out };
    }

    pub fn mul(self: *Self, a: TensorRef, b: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addElemwiseBinary(.mul, a.value, b.value);
        return .{ .value = out };
    }

    pub fn broadcastAddLastDim(self: *Self, a: TensorRef, b: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addBroadcastLastDimBinary(.add, a.value, b.value);
        return .{ .value = out };
    }

    pub fn relu(self: *Self, a: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addRelu(a.value);
        return .{ .value = out };
    }

    pub fn sqrt(self: *Self, a: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addUnary(.sqrt, a.value);
        return .{ .value = out };
    }

    pub fn unary(self: *Self, op: types.UnaryOp, a: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addUnary(op, a.value);
        return .{ .value = out };
    }

    pub fn softmax(self: *Self, a: TensorRef, axis: i32) Error!TensorRef {
        const out: ValueId = try self.graph.addSoftmax(a.value, axis);
        return .{ .value = out };
    }

    pub fn layernorm(self: *Self, x: TensorRef, gamma: TensorRef, beta: TensorRef, eps: f32, normalized_shape: []const usize) Error!TensorRef {
        const out: ValueId = try self.graph.addLayerNorm(x.value, gamma.value, beta.value, eps, normalized_shape);
        return .{ .value = out };
    }

    pub fn rmsnorm(self: *Self, x: TensorRef, gamma: TensorRef, beta: TensorRef, eps: f32, normalized_shape: []const usize) Error!TensorRef {
        const out: ValueId = try self.graph.addRMSNorm(x.value, gamma.value, beta.value, eps, normalized_shape);
        return .{ .value = out };
    }

    pub fn attention(self: *Self, q: TensorRef, k: TensorRef, v: TensorRef, scale: f32, causal: bool) Error!TensorRef {
        const out: ValueId = try self.graph.addAttention(q.value, k.value, v.value, scale, causal);
        return .{ .value = out };
    }

    pub fn mha(self: *Self, q: TensorRef, k: TensorRef, v: TensorRef, scale: f32, causal: bool, heads: usize) Error!TensorRef {
        const out: ValueId = try self.graph.addMultiHeadAttention(q.value, k.value, v.value, scale, causal, heads);
        return .{ .value = out };
    }

    pub fn conv1d(
        self: *Self,
        x: TensorRef,
        w: TensorRef,
        bias: ?TensorRef,
        stride: usize,
        dilation: usize,
        pad_left: usize,
        pad_right: usize,
        groups: usize,
    ) Error!TensorRef {
        return self.conv1dPadMode(x, w, bias, stride, dilation, pad_left, pad_right, .zero, groups);
    }

    pub fn conv1dPadMode(
        self: *Self,
        x: TensorRef,
        w: TensorRef,
        bias: ?TensorRef,
        stride: usize,
        dilation: usize,
        pad_left: usize,
        pad_right: usize,
        pad_mode: types.PadMode,
        groups: usize,
    ) Error!TensorRef {
        const out: ValueId = try self.graph.addConv1DWithPadMode(
            x.value,
            w.value,
            if (bias) |b0| b0.value else null,
            stride,
            dilation,
            pad_left,
            pad_right,
            pad_mode,
            groups,
        );
        return .{ .value = out };
    }

    pub fn conv2d(
        self: *Self,
        x: TensorRef,
        w: TensorRef,
        bias: ?TensorRef,
        stride_h: usize,
        stride_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        pad_top: usize,
        pad_bottom: usize,
        pad_left: usize,
        pad_right: usize,
        groups: usize,
    ) Error!TensorRef {
        return self.conv2dPadMode(x, w, bias, stride_h, stride_w, dilation_h, dilation_w, pad_top, pad_bottom, pad_left, pad_right, .zero, groups);
    }

    pub fn conv2dPadMode(
        self: *Self,
        x: TensorRef,
        w: TensorRef,
        bias: ?TensorRef,
        stride_h: usize,
        stride_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        pad_top: usize,
        pad_bottom: usize,
        pad_left: usize,
        pad_right: usize,
        pad_mode: types.PadMode,
        groups: usize,
    ) Error!TensorRef {
        const out: ValueId = try self.graph.addConv2DWithPadMode(
            x.value,
            w.value,
            if (bias) |b0| b0.value else null,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            pad_mode,
            groups,
        );
        return .{ .value = out };
    }

    pub fn copy(self: *Self, a: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addCopy(a.value);
        return .{ .value = out };
    }

    pub fn reduce(self: *Self, op: types.ReduceOp, a: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addReduce(op, a.value);
        return .{ .value = out };
    }

    pub fn reduceAxis(self: *Self, op: types.ReduceOp, a: TensorRef, axis: i32) Error!TensorRef {
        const out: ValueId = try self.graph.addReduceAxis(op, a.value, axis);
        return .{ .value = out };
    }

    pub fn concat(self: *Self, tensors: []const TensorRef, axis: i32) Error!TensorRef {
        if (tensors.len == 0) return Error.InvalidArgument;
        var ids: [16]ValueId = .{0} ** 16;
        if (tensors.len > ids.len) return Error.InvalidArgument;
        var i: usize = 0;
        while (i < tensors.len) : (i += 1) {
            ids[i] = tensors[i].value;
        }
        const out: ValueId = try self.graph.addConcat(ids[0..tensors.len], axis);
        return .{ .value = out };
    }

    pub fn stack(self: *Self, tensors: []const TensorRef, axis: i32) Error!TensorRef {
        if (tensors.len == 0) return Error.InvalidArgument;
        var unsq: [16]TensorRef = undefined;
        if (tensors.len > unsq.len) return Error.InvalidArgument;

        var i: usize = 0;
        while (i < tensors.len) : (i += 1) {
            unsq[i] = try self.unsqueeze(tensors[i], axis);
        }
        return self.concat(unsq[0..tensors.len], axis);
    }

    pub fn reshape(self: *Self, a: TensorRef, new_shape: []const usize) Error!TensorRef {
        const out: ValueId = try self.graph.addViewReshape(a.value, new_shape);
        return .{ .value = out };
    }

    pub fn squeeze(self: *Self, a: TensorRef, axis: ?i32) Error!TensorRef {
        const out: ValueId = try self.graph.addViewSqueeze(a.value, axis);
        return .{ .value = out };
    }

    pub fn unsqueeze(self: *Self, a: TensorRef, axis: i32) Error!TensorRef {
        const out: ValueId = try self.graph.addViewUnsqueeze(a.value, axis);
        return .{ .value = out };
    }

    pub fn transpose2d(self: *Self, a: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addViewTranspose2D(a.value);
        return .{ .value = out };
    }

    pub fn slice(self: *Self, a: TensorRef, starts: []const usize, lens: []const usize) Error!TensorRef {
        const out: ValueId = try self.graph.addViewSliceND(a.value, starts, lens);
        return .{ .value = out };
    }

    pub fn slice2d(self: *Self, a: TensorRef, start0: usize, len0: usize, start1: usize, len1: usize) Error!TensorRef {
        return self.slice(a, &[_]usize{ start0, start1 }, &[_]usize{ len0, len1 });
    }
};
