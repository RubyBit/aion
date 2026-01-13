const std = @import("std");

const types = @import("../backend/types.zig");

pub const DType = types.DType;
pub const ElemwiseBinaryOp = types.ElemwiseBinaryOp;
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
    Relu: void,
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
        if (shape.len == 0 or shape.len > 2) return GraphError.InvalidArgument;
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

    pub fn addRelu(self: *Self, a: ValueId) GraphError!ValueId {
        return self.addNodeInternal(.Relu, &[_]ValueId{a});
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
