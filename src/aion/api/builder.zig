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

pub const NamedTensorRef = struct {
    name: []const u8,
    tensor: TensorRef,
};

/// Graph-hidden model builder.
///
/// Under the hood, this builds an `aion.graph.Graph`.
pub const Builder = struct {
    allocator: std.mem.Allocator,
    graph: graph_mod.Graph,

    // Optional per-value names for diagnostics/debugging.
    value_names: std.ArrayList(?[]const u8) = .empty,
    active_scope_name: ?[]const u8 = null,
    active_scope_counter: usize = 0,
    auto_scope_counts: std.StringHashMapUnmanaged(usize) = .{},

    const Self = @This();

    pub const Error = graph_mod.GraphError;

    pub const Scope = struct {
        name: []const u8,
    };

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .graph = graph_mod.Graph.init(allocator),
            .value_names = .empty,
            .active_scope_name = null,
            .active_scope_counter = 0,
            .auto_scope_counts = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        self.auto_scope_counts.deinit(self.allocator);
        self.value_names.deinit(self.allocator);
        self.graph.deinit();
        self.* = undefined;
    }

    pub fn hasActiveScope(self: *const Self) bool {
        return self.active_scope_name != null;
    }

    pub fn beginScope(self: *Self, scope_name: []const u8) Error!Scope {
        if (self.active_scope_name != null) return Error.InvalidArgument;

        const name_copy: []u8 = self.graph.arenaAlloc().alloc(u8, scope_name.len) catch return Error.OutOfMemory;
        @memcpy(name_copy, scope_name);
        self.active_scope_name = name_copy;
        self.active_scope_counter = 0;
        return .{ .name = name_copy };
    }

    pub fn beginAutoScope(self: *Self, base_name: []const u8) Error!Scope {
        if (self.active_scope_name != null) return Error.InvalidArgument;

        const gop = try self.auto_scope_counts.getOrPut(self.allocator, base_name);
        if (!gop.found_existing) {
            const key_copy: []u8 = self.graph.arenaAlloc().alloc(u8, base_name.len) catch return Error.OutOfMemory;
            @memcpy(key_copy, base_name);
            gop.key_ptr.* = key_copy;
            gop.value_ptr.* = 0;
        }
        const idx: usize = gop.value_ptr.*;
        gop.value_ptr.* = idx + 1;

        const unique_name = std.fmt.allocPrint(self.graph.arenaAlloc(), "{s}#{d}", .{ base_name, idx }) catch return Error.OutOfMemory;
        self.active_scope_name = unique_name;
        self.active_scope_counter = 0;
        return .{ .name = unique_name };
    }

    pub fn endScope(self: *Self, scope: Scope) void {
        _ = scope;
        self.active_scope_name = null;
        self.active_scope_counter = 0;
    }

    pub fn innerGraph(self: *Self) *graph_mod.Graph {
        return &self.graph;
    }

    /// Return the known shape for a value, if available.
    ///
    /// Note: many intermediate values only become known after inference at
    /// compile time. Parameters and explicit inputs will have shapes.
    pub fn knownShape(self: *Self, t: TensorRef) ?[]const usize {
        const idx: usize = @intCast(t.value);
        if (idx >= self.graph.values.items.len) return null;
        const v = self.graph.values.items[idx];
        if (v.shape.len == 0) return null;
        return v.shape;
    }

    /// Like `knownShape`, but returns `error.InvalidArgument` if shape is unknown.
    pub fn requireKnownShape(self: *Self, t: TensorRef) Error![]const usize {
        return self.knownShape(t) orelse Error.InvalidArgument;
    }

    pub fn valueName(self: *const Self, t: TensorRef) ?[]const u8 {
        const idx: usize = @intCast(t.value);
        if (idx >= self.value_names.items.len) return null;
        return self.value_names.items[idx];
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

        // Give parameters (external-bound inputs) a stable debug name so loaded
        // models can find and swap them later.
        //
        // Important: do NOT use `autoNameIfUnnamed` here, since that would bump
        // `active_scope_counter` and perturb op numbering (e.g. `matmul#0`).
        try self.ensureNameSlot(v);
        const idx: usize = @intCast(v);
        if (self.value_names.items[idx] == null) {
            const generated_name = if (self.active_scope_name) |scope_name|
                std.fmt.allocPrint(self.graph.arenaAlloc(), "{s}/param@{d}", .{ scope_name, v }) catch return Error.OutOfMemory
            else
                std.fmt.allocPrint(self.graph.arenaAlloc(), "param@{d}", .{v}) catch return Error.OutOfMemory;
            self.value_names.items[idx] = generated_name;
        }

        return .{ .value = v };
    }

    pub fn input(self: *Self, dtype: types.DType, shape: []const usize) Error!TensorRef {
        const v: ValueId = try self.graph.addInput(dtype, shape);
        return .{ .value = v };
    }

    pub fn matmul(self: *Self, a: TensorRef, b: TensorRef, alpha: f32, beta: f32) Error!TensorRef {
        const out: ValueId = try self.graph.addMatMul(a.value, b.value, alpha, beta);
        try self.autoNameIfUnnamed(out, "matmul");
        return .{ .value = out };
    }

    pub fn add(self: *Self, a: TensorRef, b: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addElemwiseBinary(.add, a.value, b.value);
        try self.autoNameIfUnnamed(out, "add");
        return .{ .value = out };
    }

    pub fn mul(self: *Self, a: TensorRef, b: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addElemwiseBinary(.mul, a.value, b.value);
        try self.autoNameIfUnnamed(out, "mul");
        return .{ .value = out };
    }

    pub fn broadcastAddLastDim(self: *Self, a: TensorRef, b: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addBroadcastLastDimBinary(.add, a.value, b.value);
        try self.autoNameIfUnnamed(out, "broadcast_add_last_dim");
        return .{ .value = out };
    }

    pub fn relu(self: *Self, a: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addRelu(a.value);
        try self.autoNameIfUnnamed(out, "relu");
        return .{ .value = out };
    }

    pub fn sigmoid(self: *Self, a: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addUnary(.sigmoid, a.value);
        try self.autoNameIfUnnamed(out, "sigmoid");
        return .{ .value = out };
    }

    pub fn tanh(self: *Self, a: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addUnary(.tanh, a.value);
        try self.autoNameIfUnnamed(out, "tanh");
        return .{ .value = out };
    }

    pub fn sqrt(self: *Self, a: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addUnary(.sqrt, a.value);
        try self.autoNameIfUnnamed(out, "sqrt");
        return .{ .value = out };
    }

    pub fn unary(self: *Self, op: types.UnaryOp, a: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addUnary(op, a.value);
        try self.autoNameIfUnnamed(out, @tagName(op));
        return .{ .value = out };
    }

    pub fn softmax(self: *Self, a: TensorRef, axis: i32) Error!TensorRef {
        const out: ValueId = try self.graph.addSoftmax(a.value, axis);
        try self.autoNameIfUnnamed(out, "softmax");
        return .{ .value = out };
    }

    pub fn layernorm(self: *Self, x: TensorRef, gamma: TensorRef, beta: TensorRef, eps: f32, normalized_shape: []const usize) Error!TensorRef {
        const out: ValueId = try self.graph.addLayerNorm(x.value, gamma.value, beta.value, eps, normalized_shape);
        try self.autoNameIfUnnamed(out, "layernorm");
        return .{ .value = out };
    }

    pub fn rmsnorm(self: *Self, x: TensorRef, gamma: TensorRef, beta: TensorRef, eps: f32, normalized_shape: []const usize) Error!TensorRef {
        const out: ValueId = try self.graph.addRMSNorm(x.value, gamma.value, beta.value, eps, normalized_shape);
        try self.autoNameIfUnnamed(out, "rmsnorm");
        return .{ .value = out };
    }

    pub fn attention(self: *Self, q: TensorRef, k: TensorRef, v: TensorRef, scale: f32, causal: bool) Error!TensorRef {
        const out: ValueId = try self.graph.addAttention(q.value, k.value, v.value, scale, causal);
        try self.autoNameIfUnnamed(out, "attention");
        return .{ .value = out };
    }

    pub fn mha(self: *Self, q: TensorRef, k: TensorRef, v: TensorRef, scale: f32, causal: bool, heads: usize) Error!TensorRef {
        const out: ValueId = try self.graph.addMultiHeadAttention(q.value, k.value, v.value, scale, causal, heads);
        try self.autoNameIfUnnamed(out, "mha");
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
        try self.autoNameIfUnnamed(out, "conv1d");
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
        try self.autoNameIfUnnamed(out, "conv2d");
        return .{ .value = out };
    }

    pub fn copy(self: *Self, a: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addCopy(a.value);
        try self.autoNameIfUnnamed(out, "copy");
        return .{ .value = out };
    }

    /// Gather rows from a 2D table using i32 indices.
    ///
    /// Shapes:
    /// - table:   [V, D]
    /// - indices: [B, L]
    /// - out:     [B, L, D]
    pub fn gatherRows(self: *Self, table: TensorRef, indices: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addGatherRows(table.value, indices.value);
        try self.autoNameIfUnnamed(out, "gather_rows");
        return .{ .value = out };
    }

    /// Rotary positional embedding over 1D positions.
    ///
    /// Inputs:
    /// - x:         [B, L, N, H]
    /// - positions: [B, L] (i32)
    ///
    /// Output:
    /// - out:       [B, L, N, H]
    pub fn rope1d(
        self: *Self,
        x: TensorRef,
        positions: TensorRef,
        base_frequency: f32,
        scale_factor: f32,
        rope_proportion: f32,
    ) Error!TensorRef {
        const out: ValueId = try self.graph.addRoPE1D(
            x.value,
            positions.value,
            base_frequency,
            scale_factor,
            rope_proportion,
        );
        try self.autoNameIfUnnamed(out, "rope1d");
        return .{ .value = out };
    }

    /// In-place KV cache append.
    ///
    /// Inputs:
    /// - cache:     [B, H_kv, T, D]
    /// - new_kv:    [B, H_kv, new_len, D]
    /// - end_index: [B] (i32)
    ///
    /// Output:
    /// - out: same shape/dtype as cache (mutated in-place semantics)
    pub fn kvCacheAppend(
        self: *Self,
        cache: TensorRef,
        new_kv: TensorRef,
        end_index: TensorRef,
    ) Error!TensorRef {
        const out: ValueId = try self.graph.addKVCacheAppend(cache.value, new_kv.value, end_index.value);
        try self.autoNameIfUnnamed(out, "kv_cache_append");
        return .{ .value = out };
    }

    pub fn reduce(self: *Self, op: types.ReduceOp, a: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addReduce(op, a.value);
        try self.autoNameIfUnnamed(out, "reduce");
        return .{ .value = out };
    }

    pub fn reduceAxis(self: *Self, op: types.ReduceOp, a: TensorRef, axis: i32) Error!TensorRef {
        const out: ValueId = try self.graph.addReduceAxis(op, a.value, axis);
        try self.autoNameIfUnnamed(out, "reduce_axis");
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
        try self.autoNameIfUnnamed(out, "concat");
        return .{ .value = out };
    }

    /// Fused single-timestep LSTM cell.
    ///
    /// Output is a packed state tensor of shape `[batch, 2*hidden]` where:
    /// - state[:, 0:hidden]   == h_t
    /// - state[:, hidden:2*h] == c_t
    pub fn lstmCell(
        self: *Self,
        x: TensorRef,
        h_prev: TensorRef,
        c_prev: TensorRef,
        w_ih: TensorRef,
        w_hh: TensorRef,
        b_ih: ?TensorRef,
        b_hh: ?TensorRef,
    ) Error!TensorRef {
        const out: ValueId = try self.graph.addLSTMCell(
            x.value,
            h_prev.value,
            c_prev.value,
            w_ih.value,
            w_hh.value,
            if (b_ih) |t| t.value else null,
            if (b_hh) |t| t.value else null,
        );
        try self.autoNameIfUnnamed(out, "lstm_cell");
        return .{ .value = out };
    }

    /// Fused complex-abs (magnitude) + mean reduction over time.
    ///
    /// Expects split-complex layout: `x[batch, time, 2*cutoff]` where the last
    /// dimension is `[real(0..cutoff), imag(0..cutoff)]`.
    ///
    /// Output: `[batch, out_channels]` where
    /// $$out[b,c] = \mathrm{mean}_t\, \sqrt{re^2 + im^2}$$
    /// with `re=x[b,t,c]`, `im=x[b,t,c+cutoff]`.
    pub fn complexAbsMean(self: *Self, x: TensorRef, out_channels: usize) Error!TensorRef {
        const out: ValueId = try self.graph.addComplexAbsMean(x.value, out_channels);
        try self.autoNameIfUnnamed(out, "complex_abs_mean");
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
        try self.autoNameIfUnnamed(out, "reshape");
        return .{ .value = out };
    }

    pub fn squeeze(self: *Self, a: TensorRef, axis: ?i32) Error!TensorRef {
        const out: ValueId = try self.graph.addViewSqueeze(a.value, axis);
        try self.autoNameIfUnnamed(out, "squeeze");
        return .{ .value = out };
    }

    pub fn unsqueeze(self: *Self, a: TensorRef, axis: i32) Error!TensorRef {
        const out: ValueId = try self.graph.addViewUnsqueeze(a.value, axis);
        try self.autoNameIfUnnamed(out, "unsqueeze");
        return .{ .value = out };
    }

    pub fn transpose2d(self: *Self, a: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addViewTranspose2D(a.value);
        try self.autoNameIfUnnamed(out, "transpose2d");
        return .{ .value = out };
    }

    pub fn slice(self: *Self, a: TensorRef, starts: []const usize, lens: []const usize) Error!TensorRef {
        const out: ValueId = try self.graph.addViewSliceND(a.value, starts, lens);
        try self.autoNameIfUnnamed(out, "slice");
        return .{ .value = out };
    }

    pub fn slice2d(self: *Self, a: TensorRef, start0: usize, len0: usize, start1: usize, len1: usize) Error!TensorRef {
        return self.slice(a, &[_]usize{ start0, start1 }, &[_]usize{ len0, len1 });
    }

    fn autoNameIfUnnamed(self: *Self, vid: ValueId, tag: []const u8) Error!void {
        try self.ensureNameSlot(vid);

        const idx: usize = @intCast(vid);
        if (self.value_names.items[idx] != null) return;

        const n: usize = self.active_scope_counter;
        self.active_scope_counter = n + 1;

        const generated_name = if (self.active_scope_name) |scope_name|
            std.fmt.allocPrint(self.graph.arenaAlloc(), "{s}/{s}#{d}", .{ scope_name, tag, n }) catch return Error.OutOfMemory
        else
            std.fmt.allocPrint(self.graph.arenaAlloc(), "{s}#{d}", .{ tag, n }) catch return Error.OutOfMemory;

        self.value_names.items[idx] = generated_name;
    }
};
