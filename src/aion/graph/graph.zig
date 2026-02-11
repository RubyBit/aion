const std = @import("std");

const types = @import("../backend/types.zig");

pub const DType = types.DType;
pub const ElemwiseBinaryOp = types.ElemwiseBinaryOp;
pub const UnaryOp = types.UnaryOp;
pub const ReduceOp = types.ReduceOp;

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
    Reduce: struct { op: ReduceOp },
    Copy: void,

    /// View ops (lowered into materialization steps in v0).
    ViewReshape: struct { new_shape: []const usize },
    ViewTranspose2D: void,
    ViewSlice2D: struct { start0: usize, len0: usize, start1: usize, len1: usize },
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

    values: std.ArrayList(Value) = .{},
    nodes: std.ArrayList(Node) = .{},
    outputs: std.ArrayList(ValueId) = .{},

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
        return self.addNodeInternal(.{ .Reduce = .{ .op = op } }, &[_]ValueId{a});
    }

    pub fn addCopy(self: *Self, a: ValueId) GraphError!ValueId {
        return self.addNodeInternal(.Copy, &[_]ValueId{a});
    }

    pub fn addViewReshape(self: *Self, a: ValueId, new_shape: []const usize) GraphError!ValueId {
        const sh: []const usize = try self.dupeShape(new_shape);
        return self.addNodeInternal(.{ .ViewReshape = .{ .new_shape = sh } }, &[_]ValueId{a});
    }

    pub fn addViewTranspose2D(self: *Self, a: ValueId) GraphError!ValueId {
        return self.addNodeInternal(.ViewTranspose2D, &[_]ValueId{a});
    }

    pub fn addViewSlice2D(self: *Self, a: ValueId, start0: usize, len0: usize, start1: usize, len1: usize) GraphError!ValueId {
        return self.addNodeInternal(.{ .ViewSlice2D = .{ .start0 = start0, .len0 = len0, .start1 = start1, .len1 = len1 } }, &[_]ValueId{a});
    }

    pub fn setOutputs(self: *Self, outs: []const ValueId) GraphError!void {
        self.outputs.clearRetainingCapacity();
        self.outputs.appendSlice(self.allocator, outs) catch return GraphError.OutOfMemory;
    }
};
