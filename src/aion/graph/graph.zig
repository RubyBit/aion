const std = @import("std");

const types = @import("../backend/types.zig");

pub const DType = types.DType;
pub const ElemwiseBinaryOp = types.ElemwiseBinaryOp;
pub const UnaryOp = types.UnaryOp;
pub const ReduceOp = types.ReduceOp;
pub const PadMode = types.PadMode;
const MAX_CONCAT_INPUTS: usize = 16;

pub const ValueId = u32;
pub const NodeId = u32;

/// External binding id (typically a `storage.manager.TensorId`).
///
/// This is kept as a plain integer to avoid importing storage modules here.
pub const ExternalId = u32;

pub const Value = struct {
    /// If null, inference will fill this in.
    dtype: ?DType = null,
    /// Shape owned by the graph arena. Empty => unknown.
    shape: []const usize = &[_]usize{},

    /// Producing node if any.
    producer: ?NodeId = null,

    /// Optional binding to an externally managed tensor.
    external: ?ExternalId = null,
};

pub const Op = union(enum) {
    /// out = alpha * (a @ b) + beta * out
    MatMul: struct { alpha: f32 = 1.0, beta: f32 = 0.0 },

    ElemwiseBinary: struct { op: ElemwiseBinaryOp },
    BroadcastLastDimBinary: struct { op: ElemwiseBinaryOp },
    Unary: struct { op: UnaryOp },
    Softmax: struct { axis: i32 },

    /// 1D convolution (NLC-style channel-last).
    ///
    /// Shapes:
    /// - x: [..., l_in, c_in]
    /// - w: [k, c_in/groups, c_out]
    /// - bias (optional): [c_out]
    /// - out: [..., l_out, c_out]
    Conv1D: struct {
        stride: usize = 1,
        dilation: usize = 1,
        pad_left: usize = 0,
        pad_right: usize = 0,
        pad_mode: PadMode = .zero,
        groups: usize = 1,
    },

    /// 2D convolution (NHWC-style channel-last).
    ///
    /// Shapes:
    /// - x: [..., h_in, w_in, c_in]
    /// - w: [k_h, k_w, c_in/groups, c_out]
    /// - bias (optional): [c_out]
    /// - out: [..., h_out, w_out, c_out]
    Conv2D: struct {
        stride_h: usize = 1,
        stride_w: usize = 1,
        dilation_h: usize = 1,
        dilation_w: usize = 1,
        pad_top: usize = 0,
        pad_bottom: usize = 0,
        pad_left: usize = 0,
        pad_right: usize = 0,
        pad_mode: PadMode = .zero,
        groups: usize = 1,
    },

    /// Normalize over the last dimensions (rank>=1; normalized_shape is required).
    /// out = ((x - mean) / sqrt(var + eps)) * gamma + beta
    LayerNorm: struct { eps: f32, normalized_shape: []const usize },

    /// Normalize over the last dimensions (rank>=1; normalized_shape is required).
    /// out = (x / sqrt(mean(x^2) + eps)) * gamma + beta
    RMSNorm: struct { eps: f32, normalized_shape: []const usize },

    /// Fused attention (batched over leading dims).
    ///
    /// Shapes:
    /// - q: [..., m, dk]
    /// - k: [..., n, dk]
    /// - v: [..., n, dv]
    /// - out: [..., m, dv]
    ///
    /// Computes softmax(scale * q @ k^T) @ v per leading-dim slice.
    /// If causal, masks keys where key_index > query_index.
    Attention: struct { scale: f32, causal: bool },

    /// Fused multi-head attention (separate head dim).
    ///
    /// Separate head dim (batched):
    /// - q: [..., h, m, dk]
    /// - k: [..., h, n, dk]
    /// - v: [..., h, n, dv]
    /// - out: [..., h, m, dv]
    ///
    /// Computes softmax(scale * q @ k^T) @ v per head/slice.
    /// If causal, masks keys where key_index > query_index (within head).
    MultiHeadAttention: struct { scale: f32, causal: bool, heads: usize },
    Reduce: struct { op: ReduceOp, axis: ?i32 = null },
    Concat: struct { axis: i32 },

    /// Fused single-timestep LSTM cell.
    ///
    /// Inputs (N=rank-2):
    /// - x:      [batch, input_size]
    /// - h_prev: [batch, hidden]
    /// - c_prev: [batch, hidden]
    /// - w_ih:   [input_size, 4*hidden]
    /// - w_hh:   [hidden, 4*hidden]
    /// - b_ih:   [4*hidden] (optional)
    /// - b_hh:   [4*hidden] (optional)
    ///
    /// Output:
    /// - state: [batch, 2*hidden] where state[:,0:h]=h_t and state[:,h:2h]=c_t
    LSTMCell: struct { has_bias: bool },

    /// Fused complex-abs (magnitude) + mean reduction over time.
    ///
    /// This is a common pattern in audio/signal front-ends when complex values are
    /// represented as split real/imag halves.
    ///
    /// Input:
    /// - x: [batch, time, 2*cutoff] interpreted as real/imag halves.
    ///
    /// Output:
    /// - out: [batch, out_channels] where out[b,c] = mean_t sqrt(re^2 + im^2)
    ///   using re=x[b,t,c], im=x[b,t,c+cutoff].
    ///
    /// Contract:
    /// - x.shape[2] must be even
    /// - out_channels must be in 1..=cutoff
    ComplexAbsMean: struct { out_channels: usize },

    Copy: void,

    /// View ops (lowered into materialization steps in v0).
    ViewReshape: struct { new_shape: []const usize },
    ViewSqueeze: struct { axis: ?i32 = null },
    ViewUnsqueeze: struct { axis: i32 },
    ViewTranspose2D: void,
    ViewSliceND: struct { starts: []const usize, lens: []const usize },
};

pub const Node = struct {
    op: Op,
    inputs: []const ValueId,
    output: ValueId,
};

pub const GraphError = error{ InvalidArgument, OutOfMemory };

pub const Graph = struct {
    arena: std.heap.ArenaAllocator,
    allocator: std.mem.Allocator,

    values: std.ArrayList(Value) = .empty,
    nodes: std.ArrayList(Node) = .empty,
    outputs: std.ArrayList(ValueId) = .empty,

    const Self = @This();

    const MAX_RANK: usize = 8;

    pub fn init(allocator: std.mem.Allocator) Self {
        const arena = std.heap.ArenaAllocator.init(allocator);
        return .{ .arena = arena, .allocator = allocator };
    }

    pub fn deinit(self: *Self) void {
        self.values.deinit(self.allocator);
        self.nodes.deinit(self.allocator);
        self.outputs.deinit(self.allocator);
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn arenaAlloc(self: *Self) std.mem.Allocator {
        return self.arena.allocator();
    }

    pub fn dupeShape(self: *Self, shape: []const usize) GraphError![]const usize {
        if (shape.len == 0 or shape.len > MAX_RANK) return GraphError.InvalidArgument;
        const out: []usize = self.arenaAlloc().alloc(usize, shape.len) catch return GraphError.OutOfMemory;
        @memcpy(out, shape);
        return out;
    }

    pub fn addValue(self: *Self) GraphError!ValueId {
        const id: ValueId = @intCast(self.values.items.len);
        self.values.append(self.allocator, .{}) catch return GraphError.OutOfMemory;
        return id;
    }

    pub fn addInput(self: *Self, dtype: DType, shape: []const usize) GraphError!ValueId {
        const id: ValueId = try self.addValue();
        const sh: []const usize = try self.dupeShape(shape);
        self.values.items[@intCast(id)] = .{ .dtype = dtype, .shape = sh };
        return id;
    }

    pub fn bindExternal(self: *Self, value: ValueId, external: ExternalId) GraphError!void {
        const idx: usize = @intCast(value);
        if (idx >= self.values.items.len) return GraphError.InvalidArgument;
        self.values.items[idx].external = external;
    }

    fn addNodeInternal(self: *Self, op: Op, inputs: []const ValueId) GraphError!ValueId {
        const out_id: ValueId = try self.addValue();

        const inputs_copy: []ValueId = self.arenaAlloc().alloc(ValueId, inputs.len) catch return GraphError.OutOfMemory;
        @memcpy(inputs_copy, inputs);

        const node_id: NodeId = @intCast(self.nodes.items.len);
        self.nodes.append(self.allocator, .{ .op = op, .inputs = inputs_copy, .output = out_id }) catch return GraphError.OutOfMemory;

        self.values.items[@intCast(out_id)].producer = node_id;
        return out_id;
    }

    pub fn addMatMul(self: *Self, a: ValueId, b: ValueId, alpha: f32, beta: f32) GraphError!ValueId {
        return self.addNodeInternal(.{ .MatMul = .{ .alpha = alpha, .beta = beta } }, &[_]ValueId{ a, b });
    }

    pub fn addElemwiseBinary(self: *Self, op: ElemwiseBinaryOp, a: ValueId, b: ValueId) GraphError!ValueId {
        return self.addNodeInternal(.{ .ElemwiseBinary = .{ .op = op } }, &[_]ValueId{ a, b });
    }

    pub fn addBroadcastLastDimBinary(self: *Self, op: ElemwiseBinaryOp, a: ValueId, b: ValueId) GraphError!ValueId {
        return self.addNodeInternal(.{ .BroadcastLastDimBinary = .{ .op = op } }, &[_]ValueId{ a, b });
    }

    pub fn addUnary(self: *Self, op: UnaryOp, a: ValueId) GraphError!ValueId {
        return self.addNodeInternal(.{ .Unary = .{ .op = op } }, &[_]ValueId{a});
    }

    /// Softmax over the specified axis (negative axes are allowed).
    /// - axis == -1 => last dimension
    pub fn addSoftmax(self: *Self, a: ValueId, axis: i32) GraphError!ValueId {
        return self.addNodeInternal(.{ .Softmax = .{ .axis = axis } }, &[_]ValueId{a});
    }

    pub fn addConv1D(
        self: *Self,
        x: ValueId,
        w: ValueId,
        bias: ?ValueId,
        stride: usize,
        dilation: usize,
        pad_left: usize,
        pad_right: usize,
        groups: usize,
    ) GraphError!ValueId {
        return self.addConv1DWithPadMode(x, w, bias, stride, dilation, pad_left, pad_right, .zero, groups);
    }

    pub fn addConv1DWithPadMode(
        self: *Self,
        x: ValueId,
        w: ValueId,
        bias: ?ValueId,
        stride: usize,
        dilation: usize,
        pad_left: usize,
        pad_right: usize,
        pad_mode: PadMode,
        groups: usize,
    ) GraphError!ValueId {
        const op: Op = .{ .Conv1D = .{
            .stride = stride,
            .dilation = dilation,
            .pad_left = pad_left,
            .pad_right = pad_right,
            .pad_mode = pad_mode,
            .groups = groups,
        } };
        if (bias) |b| {
            return self.addNodeInternal(op, &[_]ValueId{ x, w, b });
        }
        return self.addNodeInternal(op, &[_]ValueId{ x, w });
    }

    pub fn addConv2D(
        self: *Self,
        x: ValueId,
        w: ValueId,
        bias: ?ValueId,
        stride_h: usize,
        stride_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        pad_top: usize,
        pad_bottom: usize,
        pad_left: usize,
        pad_right: usize,
        groups: usize,
    ) GraphError!ValueId {
        return self.addConv2DWithPadMode(x, w, bias, stride_h, stride_w, dilation_h, dilation_w, pad_top, pad_bottom, pad_left, pad_right, .zero, groups);
    }

    pub fn addConv2DWithPadMode(
        self: *Self,
        x: ValueId,
        w: ValueId,
        bias: ?ValueId,
        stride_h: usize,
        stride_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        pad_top: usize,
        pad_bottom: usize,
        pad_left: usize,
        pad_right: usize,
        pad_mode: PadMode,
        groups: usize,
    ) GraphError!ValueId {
        const op: Op = .{ .Conv2D = .{
            .stride_h = stride_h,
            .stride_w = stride_w,
            .dilation_h = dilation_h,
            .dilation_w = dilation_w,
            .pad_top = pad_top,
            .pad_bottom = pad_bottom,
            .pad_left = pad_left,
            .pad_right = pad_right,
            .pad_mode = pad_mode,
            .groups = groups,
        } };
        if (bias) |b| {
            return self.addNodeInternal(op, &[_]ValueId{ x, w, b });
        }
        return self.addNodeInternal(op, &[_]ValueId{ x, w });
    }

    pub fn addLayerNorm(self: *Self, x: ValueId, gamma: ValueId, beta: ValueId, eps: f32, normalized_shape: []const usize) GraphError!ValueId {
        const ns: []const usize = try self.dupeShape(normalized_shape);
        return self.addNodeInternal(.{ .LayerNorm = .{ .eps = eps, .normalized_shape = ns } }, &[_]ValueId{ x, gamma, beta });
    }

    pub fn addRMSNorm(self: *Self, x: ValueId, gamma: ValueId, beta: ValueId, eps: f32, normalized_shape: []const usize) GraphError!ValueId {
        const ns: []const usize = try self.dupeShape(normalized_shape);
        return self.addNodeInternal(.{ .RMSNorm = .{ .eps = eps, .normalized_shape = ns } }, &[_]ValueId{ x, gamma, beta });
    }

    pub fn addAttention(self: *Self, q: ValueId, k: ValueId, v: ValueId, scale: f32, causal: bool) GraphError!ValueId {
        return self.addNodeInternal(.{ .Attention = .{ .scale = scale, .causal = causal } }, &[_]ValueId{ q, k, v });
    }

    pub fn addMultiHeadAttention(self: *Self, q: ValueId, k: ValueId, v: ValueId, scale: f32, causal: bool, heads: usize) GraphError!ValueId {
        return self.addNodeInternal(.{ .MultiHeadAttention = .{ .scale = scale, .causal = causal, .heads = heads } }, &[_]ValueId{ q, k, v });
    }

    pub fn addRelu(self: *Self, a: ValueId) GraphError!ValueId {
        return self.addUnary(.relu, a);
    }

    pub fn addReduce(self: *Self, op: ReduceOp, a: ValueId) GraphError!ValueId {
        return self.addNodeInternal(.{ .Reduce = .{ .op = op, .axis = null } }, &[_]ValueId{a});
    }

    pub fn addReduceAxis(self: *Self, op: ReduceOp, a: ValueId, axis: i32) GraphError!ValueId {
        return self.addNodeInternal(.{ .Reduce = .{ .op = op, .axis = axis } }, &[_]ValueId{a});
    }

    pub fn addConcat(self: *Self, inputs: []const ValueId, axis: i32) GraphError!ValueId {
        if (inputs.len == 0 or inputs.len > MAX_CONCAT_INPUTS) return GraphError.InvalidArgument;
        return self.addNodeInternal(.{ .Concat = .{ .axis = axis } }, inputs);
    }

    pub fn addCopy(self: *Self, a: ValueId) GraphError!ValueId {
        return self.addNodeInternal(.Copy, &[_]ValueId{a});
    }

    pub fn addLSTMCell(
        self: *Self,
        x: ValueId,
        h_prev: ValueId,
        c_prev: ValueId,
        w_ih: ValueId,
        w_hh: ValueId,
        b_ih: ?ValueId,
        b_hh: ?ValueId,
    ) GraphError!ValueId {
        // Bias policy: either both provided or both omitted.
        if ((b_ih != null) != (b_hh != null)) return GraphError.InvalidArgument;

        const op: Op = .{ .LSTMCell = .{ .has_bias = (b_ih != null) } };
        if (b_ih) |b0| {
            return self.addNodeInternal(op, &[_]ValueId{ x, h_prev, c_prev, w_ih, w_hh, b0, b_hh.? });
        }
        return self.addNodeInternal(op, &[_]ValueId{ x, h_prev, c_prev, w_ih, w_hh });
    }

    pub fn addComplexAbsMean(self: *Self, x: ValueId, out_channels: usize) GraphError!ValueId {
        if (out_channels == 0) return GraphError.InvalidArgument;
        return self.addNodeInternal(.{ .ComplexAbsMean = .{ .out_channels = out_channels } }, &[_]ValueId{x});
    }

    pub fn addViewReshape(self: *Self, a: ValueId, new_shape: []const usize) GraphError!ValueId {
        const sh: []const usize = try self.dupeShape(new_shape);
        return self.addNodeInternal(.{ .ViewReshape = .{ .new_shape = sh } }, &[_]ValueId{a});
    }

    pub fn addViewSqueeze(self: *Self, a: ValueId, axis: ?i32) GraphError!ValueId {
        return self.addNodeInternal(.{ .ViewSqueeze = .{ .axis = axis } }, &[_]ValueId{a});
    }

    pub fn addViewUnsqueeze(self: *Self, a: ValueId, axis: i32) GraphError!ValueId {
        return self.addNodeInternal(.{ .ViewUnsqueeze = .{ .axis = axis } }, &[_]ValueId{a});
    }

    pub fn addViewTranspose2D(self: *Self, a: ValueId) GraphError!ValueId {
        return self.addNodeInternal(.ViewTranspose2D, &[_]ValueId{a});
    }

    pub fn addViewSliceND(self: *Self, a: ValueId, starts: []const usize, lens: []const usize) GraphError!ValueId {
        if (starts.len == 0 or starts.len != lens.len or starts.len > MAX_RANK) return GraphError.InvalidArgument;
        const starts_copy: []const usize = try self.dupeShape(starts);
        const lens_copy: []const usize = try self.dupeShape(lens);
        return self.addNodeInternal(.{ .ViewSliceND = .{ .starts = starts_copy, .lens = lens_copy } }, &[_]ValueId{a});
    }

    pub fn addViewSlice2D(self: *Self, a: ValueId, start0: usize, len0: usize, start1: usize, len1: usize) GraphError!ValueId {
        return self.addViewSliceND(a, &[_]usize{ start0, start1 }, &[_]usize{ len0, len1 });
    }

    pub fn setOutputs(self: *Self, outs: []const ValueId) GraphError!void {
        self.outputs.clearRetainingCapacity();
        self.outputs.appendSlice(self.allocator, outs) catch return GraphError.OutOfMemory;
    }
};
