// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const graph_mod = @import("../graph/graph.zig");
const infer_mod = @import("../graph/infer.zig");
const types = @import("../backend/types.zig");

const api_tensor = @import("tensor.zig");

pub const ValueId = graph_mod.ValueId;
pub const RegionId = graph_mod.RegionId;

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

/// Declares that an input tensor's axis is a free (variable) dimension named
/// `name`. Reusing a name across axes/inputs constrains them to the same runtime
/// size. Declared on the builder via `symbolicDim`; read by `compile`/`export`.
pub const DimSymbol = struct {
    tensor: TensorRef,
    axis: usize,
    name: []const u8,
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

    // Symbolic (variable) input dimensions declared via `symbolicDim`.
    dim_symbols: std.ArrayList(DimSymbol) = .empty,

    // Symbolic dims used in view-op attributes (reshape new_shape / slice lens).
    // `tensor` is the view op's *output* value; `axis` indexes the attr dim; the
    // concrete attr value at that axis is the authoring placeholder. Emitted as a
    // dim-symbol expr term at export (the symbol must also be an input dim symbol).
    view_dim_symbols: std.ArrayList(DimSymbol) = .empty,

    const Self = @This();

    /// Includes `InferError` because ops infer their output shape eagerly as they
    /// are added, so a shape/dtype error surfaces at the offending op call.
    pub const Error = graph_mod.GraphError || infer_mod.InferError;

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
            .dim_symbols = .empty,
            .view_dim_symbols = .empty,
        };
    }

    pub fn deinit(self: *Self) void {
        self.auto_scope_counts.deinit(self.allocator);
        self.value_names.deinit(self.allocator);
        self.dim_symbols.deinit(self.allocator);
        self.view_dim_symbols.deinit(self.allocator);
        self.graph.deinit();
        self.* = undefined;
    }

    /// Declare input `t`'s axis `axis` as a free (variable) dimension named
    /// `sym_name`. One `compile`/`export` then serves any size on that axis;
    /// reuse a name to tie axes to the same runtime size.
    pub fn symbolicDim(self: *Self, t: TensorRef, axis: usize, sym_name: []const u8) Error!void {
        const name_copy: []u8 = self.graph.arenaAlloc().alloc(u8, sym_name.len) catch return Error.OutOfMemory;
        @memcpy(name_copy, sym_name);
        self.dim_symbols.append(self.allocator, .{ .tensor = t, .axis = axis, .name = name_copy }) catch return Error.OutOfMemory;
    }

    /// The symbolic input dimensions declared so far (read by `compile`/`export`).
    pub fn dimSymbols(self: *const Self) []const DimSymbol {
        return self.dim_symbols.items;
    }

    /// Symbolic dims used in view-op attributes (read by `export`).
    pub fn viewDimSymbols(self: *const Self) []const DimSymbol {
        return self.view_dim_symbols.items;
    }

    fn recordViewSymbols(self: *Self, out: TensorRef, symbols: []const ?[]const u8) Error!void {
        for (symbols, 0..) |maybe_name, axis| {
            const sym_name = maybe_name orelse continue;
            const name_copy: []u8 = self.graph.arenaAlloc().alloc(u8, sym_name.len) catch return Error.OutOfMemory;
            @memcpy(name_copy, sym_name);
            self.view_dim_symbols.append(self.allocator, .{ .tensor = out, .axis = axis, .name = name_copy }) catch return Error.OutOfMemory;
        }
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

    /// Transfer ownership of the underlying graph to the caller, replacing it with a
    /// fresh empty graph so the Builder remains safe to `deinit`. Used by
    /// `ctx.compile` to hand the graph to the compiled `Model` without a round trip.
    /// The Builder should not be used to add more ops after this.
    pub fn takeGraph(self: *Self) graph_mod.Graph {
        const g = self.graph;
        self.graph = graph_mod.Graph.init(self.allocator);
        return g;
    }

    /// Return the known shape for a value, if available.
    ///
    /// With eager per-op inference, every value produced so far (inputs, params,
    /// and op outputs) has a shape. For a value on a declared-symbolic axis the
    /// reported size is the authoring *placeholder* (the size the axis was
    /// declared with), which propagates deterministically through derived values.
    pub fn knownShape(self: *const Self, t: TensorRef) ?[]const usize {
        const idx: usize = @intCast(t.value);
        if (idx >= self.graph.values.items.len) return null;
        const v = self.graph.values.items[idx];
        if (v.shape.len == 0) return null;
        return v.shape;
    }

    /// The dtype of a value, if known (null for an out-of-range id).
    pub fn dtypeOf(self: *const Self, t: TensorRef) ?types.DType {
        const idx: usize = @intCast(t.value);
        if (idx >= self.graph.values.items.len) return null;
        return self.graph.values.items[idx].dtype;
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

    pub fn multiHeadAttentionCached(
        self: *Self,
        q: TensorRef,
        k_cache: TensorRef,
        v_cache: TensorRef,
        positions: TensorRef,
        end_index: TensorRef,
        scale: f32,
        causal: bool,
        sliding_window: usize,
        attn_logits_soft_cap: f32,
    ) Error!TensorRef {
        const out: ValueId = try self.graph.addMultiHeadAttentionCached(
            q.value,
            k_cache.value,
            v_cache.value,
            positions.value,
            end_index.value,
            scale,
            causal,
            sliding_window,
            attn_logits_soft_cap,
        );
        try self.autoNameIfUnnamed(out, "mha_cached");
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
    pub fn sequenceAppend(
        self: *Self,
        cache: TensorRef,
        new_kv: TensorRef,
        end_index: TensorRef,
    ) Error!TensorRef {
        const out: ValueId = try self.graph.addSequenceAppend(cache.value, new_kv.value, end_index.value);
        try self.autoNameIfUnnamed(out, "sequence_append");
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

    /// Real FFT over the last (power-of-two) dimension.
    ///
    /// Input `x[.., n_fft]` (f32) → output `[.., n_fft+2]` packed complex:
    /// one-sided bins `n_fft/2+1` with real parts in `[0..bins)` and imaginary
    /// parts in `[bins..2*bins)`.
    pub fn rfft(self: *Self, x: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addRFFT(x.value);
        try self.autoNameIfUnnamed(out, "rfft");
        return .{ .value = out };
    }

    /// Short-time Fourier transform.
    ///
    /// Inputs: `signal[batch, samples]` (f32), `window[n_fft]` (f32, padded to
    /// `n_fft` by the caller). Output: `[batch, num_frames, n_fft+2]` packed
    /// complex (same layout as `rfft`). See `Op.STFT` for framing semantics.
    pub fn stft(
        self: *Self,
        signal: TensorRef,
        window: TensorRef,
        n_fft: usize,
        hop_length: usize,
        center: bool,
    ) Error!TensorRef {
        const out: ValueId = try self.graph.addSTFT(signal.value, window.value, n_fft, hop_length, center);
        try self.autoNameIfUnnamed(out, "stft");
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

    /// Reshape where some target dims are symbolic. `new_shape[i]` is the concrete
    /// authoring placeholder; `symbols[i]`, if non-null, names a dim symbol that
    /// axis takes at runtime (emitted as a symbolic term at export).
    pub fn reshapeSym(self: *Self, a: TensorRef, new_shape: []const usize, symbols: []const ?[]const u8) Error!TensorRef {
        const ref = try self.reshape(a, new_shape);
        try self.recordViewSymbols(ref, symbols);
        return ref;
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

    /// Slice where some lengths are symbolic. `lens[i]` is the concrete authoring
    /// placeholder; `symbols[i]`, if non-null, names a dim symbol that length
    /// takes at runtime (emitted as a symbolic term at export).
    pub fn sliceSym(self: *Self, a: TensorRef, starts: []const usize, lens: []const usize, symbols: []const ?[]const u8) Error!TensorRef {
        const ref = try self.slice(a, starts, lens);
        try self.recordViewSymbols(ref, symbols);
        return ref;
    }

    pub fn slice2d(self: *Self, a: TensorRef, start0: usize, len0: usize, start1: usize, len1: usize) Error!TensorRef {
        return self.slice(a, &[_]usize{ start0, start1 }, &[_]usize{ len0, len1 });
    }

    /// Generic elementwise binary op (`add`/`sub`/`mul`/`div` and the comparison
    /// ops `eq`/`ne`/`lt`/`gt`/`le`/`ge`). `add`/`mul` also have named shortcuts.
    pub fn elemwiseBinary(self: *Self, op: types.ElemwiseBinaryOp, a: TensorRef, b: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addElemwiseBinary(op, a.value, b.value);
        try self.autoNameIfUnnamed(out, @tagName(op));
        return .{ .value = out };
    }

    /// Generic last-dim-broadcast binary op (`b` broadcasts along the last dim of
    /// `a`). `broadcastAddLastDim` is the named `add` shortcut.
    pub fn broadcastLastDim(self: *Self, op: types.ElemwiseBinaryOp, a: TensorRef, b: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addBroadcastLastDimBinary(op, a.value, b.value);
        try self.autoNameIfUnnamed(out, "broadcast_last_dim");
        return .{ .value = out };
    }

    /// Matmul with a transposed / per-row-quantized B: `C[m,n] = Σ A[m,k]·B[n,k]`
    /// (B is `[N, K]`). Used for tied-embedding logits.
    pub fn matmulNT(self: *Self, a: TensorRef, b: TensorRef, alpha: f32, beta: f32) Error!TensorRef {
        const out: ValueId = try self.graph.addMatMulNT(a.value, b.value, alpha, beta);
        try self.autoNameIfUnnamed(out, "matmul_nt");
        return .{ .value = out };
    }

    /// Scalar dtype cast (e.g. f32 <-> f16).
    pub fn cast(self: *Self, a: TensorRef, to_dtype: types.DType) Error!TensorRef {
        const out: ValueId = try self.graph.addCast(a.value, to_dtype);
        try self.autoNameIfUnnamed(out, "cast");
        return .{ .value = out };
    }

    /// Index of the max value along `axis` (i32 output).
    pub fn argmax(self: *Self, a: TensorRef, axis: i32) Error!TensorRef {
        const out: ValueId = try self.graph.addArgMax(a.value, axis);
        try self.autoNameIfUnnamed(out, "argmax");
        return .{ .value = out };
    }

    /// In-place row scatter: `buf[idx] = src`. Output aliases `buf`.
    pub fn scatterRow(self: *Self, buf: TensorRef, idx: TensorRef, src: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addScatterRow(buf.value, idx.value, src.value);
        try self.autoNameIfUnnamed(out, "scatter_row");
        return .{ .value = out };
    }

    /// Fused gate: `gelu(a) * b`.
    pub fn geluMul(self: *Self, a: TensorRef, b: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addGeluMul(a.value, b.value);
        try self.autoNameIfUnnamed(out, "gelu_mul");
        return .{ .value = out };
    }

    /// Relative-position multi-head self-attention (Transformer-XL / Conformer).
    /// `mask` is optional.
    pub fn relPosMHA(
        self: *Self,
        q: TensorRef,
        k: TensorRef,
        v: TensorRef,
        pos_emb: TensorRef,
        pos_bias_u: TensorRef,
        pos_bias_v: TensorRef,
        mask: ?TensorRef,
        scale: f32,
        heads: usize,
    ) Error!TensorRef {
        const out: ValueId = try self.graph.addRelPosMHA(
            q.value,
            k.value,
            v.value,
            pos_emb.value,
            pos_bias_u.value,
            pos_bias_v.value,
            if (mask) |m| m.value else null,
            scale,
            heads,
        );
        try self.autoNameIfUnnamed(out, "relpos_mha");
        return .{ .value = out };
    }

    // --- Control flow (regions) --------------------------------------------

    /// Begin a sub-region. Ops added until `endRegion` become the region body
    /// (used for `If` branches and `Loop` bodies). Regions do not nest.
    pub fn beginRegion(self: *Self) Error!void {
        return self.graph.beginRegion();
    }

    /// Close the active region, declaring its outputs. Returns a region id for
    /// use with `ifThenElse` / `loop`.
    pub fn endRegion(self: *Self, outputs: []const TensorRef) Error!RegionId {
        var ids: [16]ValueId = undefined;
        if (outputs.len == 0 or outputs.len > ids.len) return Error.InvalidArgument;
        for (outputs, 0..) |o, i| ids[i] = o.value;
        return self.graph.endRegion(ids[0..outputs.len]);
    }

    /// Single-output conditional: evaluate `then_region` or `else_region` based
    /// on the i32 `[1]` `cond`. Both regions must have exactly one output.
    pub fn ifThenElse(self: *Self, cond: TensorRef, then_region: RegionId, else_region: RegionId) Error!TensorRef {
        const out: ValueId = try self.graph.addIf(cond.value, then_region, else_region);
        try self.autoNameIfUnnamed(out, "if");
        return .{ .value = out };
    }

    /// Single-carry loop: runs `body_region` up to `static_max_trip_count` times,
    /// threading the carry. `body_region` must have exactly one output.
    pub fn loop(self: *Self, carried_init: TensorRef, body_region: RegionId, static_max_trip_count: usize) Error!TensorRef {
        const out: ValueId = try self.graph.addLoop(carried_init.value, body_region, static_max_trip_count);
        try self.autoNameIfUnnamed(out, "loop");
        return .{ .value = out };
    }

    /// Multi-carry loop. `carried_inits[i]` pairs with body-region output `i`;
    /// the final carried values are written to `out_refs` (same length). If
    /// `cond_carry` is set, that carry index is the i32 `[1]` continue predicate.
    pub fn loopMulti(
        self: *Self,
        carried_inits: []const TensorRef,
        body_region: RegionId,
        static_max_trip_count: usize,
        cond_carry: ?usize,
        check_before: bool,
        out_refs: []TensorRef,
    ) Error!void {
        var ids: [16]ValueId = undefined;
        const n = carried_inits.len;
        if (n == 0 or n > ids.len or out_refs.len != n) return Error.InvalidArgument;
        for (carried_inits, 0..) |c, i| ids[i] = c.value;
        const outs = try self.graph.addLoopMulti(ids[0..n], body_region, static_max_trip_count, cond_carry, check_before);
        for (outs, 0..) |o, i| {
            try self.autoNameIfUnnamed(o, "loop");
            out_refs[i] = .{ .value = o };
        }
    }

    fn autoNameIfUnnamed(self: *Self, vid: ValueId, tag: []const u8) Error!void {
        // Eager per-op inference: infer the node just added (it produced `vid`).
        // Every op funnels through here exactly once after appending its node, so
        // this keeps the whole graph inferred as it is built — `knownShape` is
        // always current and shape/dtype errors surface at the offending op.
        // Runs before the naming early-return so a pre-named output still infers.
        if (self.graph.lastNode()) |node| {
            try infer_mod.inferNode(&self.graph, node);
        }

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
