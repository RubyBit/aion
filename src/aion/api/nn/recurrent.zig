// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const layer_mod = @import("layer.zig");
const state_mod = @import("state.zig");

pub const Builder = layer_mod.Builder;
pub const TensorRef = layer_mod.TensorRef;
pub const Tensor = layer_mod.Tensor;
pub const LayerName = layer_mod.LayerName;
pub const Params = state_mod.Params;
pub const BindError = state_mod.BindError;

pub const LSTMState = struct {
    h: TensorRef,
    c: TensorRef,
};

/// Single-timestep LSTM cell.
///
/// Weight layout (Aion-native):
/// - w_ih: [input_size, 4*hidden]
/// - w_hh: [hidden, 4*hidden]
/// - b_ih: [4*hidden] (optional)
/// - b_hh: [4*hidden] (optional)
///
/// Forward expects rank-2 inputs:
/// - x:      [batch, input_size]
/// - h_prev: [batch, hidden]
/// - c_prev: [batch, hidden]
///
/// The fused op returns a packed `[batch, 2*hidden]` state; `forward` splits it so
/// callers get `(h, c)` instead of re-deriving the offsets.
pub const LSTMCell = struct {
    w_ih: TensorRef,
    w_hh: TensorRef,
    b_ih: ?TensorRef = null,
    b_hh: ?TensorRef = null,

    input_size: usize,
    hidden_size: usize,
    id: LayerName = .{},

    const Self = @This();

    /// Both weights are required; the two biases are all-or-nothing (checked in
    /// `bind`, since "either both or neither" is not something a type can say).
    pub const Weights = struct {
        weight_ih: Tensor,
        weight_hh: Tensor,
        bias_ih: ?Tensor = null,
        bias_hh: ?Tensor = null,
    };

    pub const Options = struct {
        /// Scope segment for this layer, prefixing its parameter names.
        name: ?[]const u8 = null,
    };

    pub fn bind(bld: *Builder, params: anytype, opts: Options) BindError!Self {
        var src = state_mod.binding(Weights, params);
        const p: Params = src.params();

        const id, const scope = try LayerName.open(Self, bld, p.name orelse opts.name);
        defer bld.endScope(scope);

        const w_ih = try p.get(bld, Weights, .weight_ih);
        const w_hh = try p.get(bld, Weights, .weight_hh);
        const b_ih = try p.getOpt(bld, Weights, .bias_ih);
        const b_hh = try p.getOpt(bld, Weights, .bias_hh);

        // Bias policy: either both provided or both omitted.
        if ((b_ih != null) != (b_hh != null)) return error.InvalidArgument;

        const ih = bld.knownShape(w_ih) orelse return error.InvalidArgument;
        const hh = bld.knownShape(w_hh) orelse return error.InvalidArgument;
        if (ih.len != 2 or hh.len != 2) return error.InvalidArgument;

        const input_size: usize = ih[0];
        const gate_dim: usize = ih[1];
        if (input_size == 0 or gate_dim == 0 or gate_dim % 4 != 0) return error.InvalidArgument;

        const hidden_size: usize = gate_dim / 4;
        if (hidden_size == 0) return error.InvalidArgument;
        if (hh[0] != hidden_size or hh[1] != gate_dim) return error.InvalidArgument;

        if (b_ih) |bt| {
            const bs = bld.knownShape(bt) orelse return error.InvalidArgument;
            if (bs.len != 1 or bs[0] != gate_dim) return error.InvalidArgument;
        }
        if (b_hh) |bt| {
            const bs = bld.knownShape(bt) orelse return error.InvalidArgument;
            if (bs.len != 1 or bs[0] != gate_dim) return error.InvalidArgument;
        }

        return .{
            .w_ih = w_ih,
            .w_hh = w_hh,
            .b_ih = b_ih,
            .b_hh = b_hh,
            .input_size = input_size,
            .hidden_size = hidden_size,
            .id = id,
        };
    }

    pub fn forward(self: Self, bld: *Builder, x: TensorRef, h_prev: TensorRef, c_prev: TensorRef) Builder.Error!LSTMState {
        const scope = try self.id.enter(bld);
        defer bld.endScope(scope);

        const x_shape_opt: ?[]const usize = bld.knownShape(x);
        const h_shape_opt: ?[]const usize = bld.knownShape(h_prev);
        const c_shape_opt: ?[]const usize = bld.knownShape(c_prev);

        if (x_shape_opt) |x_shape| {
            if (x_shape.len != 2) return Builder.Error.InvalidArgument;
            if (x_shape[1] != self.input_size) return Builder.Error.InvalidArgument;
        }
        if (h_shape_opt) |h_shape| {
            if (h_shape.len != 2) return Builder.Error.InvalidArgument;
            if (h_shape[1] != self.hidden_size) return Builder.Error.InvalidArgument;
        }
        if (c_shape_opt) |c_shape| {
            if (c_shape.len != 2) return Builder.Error.InvalidArgument;
            if (c_shape[1] != self.hidden_size) return Builder.Error.InvalidArgument;
        }

        var batch_opt: ?usize = null;
        if (h_shape_opt) |h_shape| {
            batch_opt = h_shape[0];
        }
        if (c_shape_opt) |c_shape| {
            if (batch_opt) |b0| {
                if (c_shape[0] != b0) return Builder.Error.InvalidArgument;
            } else {
                batch_opt = c_shape[0];
            }
        }
        if (x_shape_opt) |x_shape| {
            if (batch_opt) |b0| {
                if (x_shape[0] != b0) return Builder.Error.InvalidArgument;
            } else {
                batch_opt = x_shape[0];
            }
        }

        const batch: usize = batch_opt orelse return Builder.Error.InvalidArgument;
        if (batch == 0) return Builder.Error.InvalidArgument;

        // Fused single-step LSTM cell.
        // Output is a packed [batch, 2h] state: [h_t | c_t].
        const packed_state: TensorRef = try bld.lstmCell(x, h_prev, c_prev, self.w_ih, self.w_hh, self.b_ih, self.b_hh);

        const h: usize = self.hidden_size;
        const h_t: TensorRef = try bld.slice2d(packed_state, 0, batch, 0, h);
        const c_t: TensorRef = try bld.slice2d(packed_state, 0, batch, h, h);
        return .{ .h = h_t, .c = c_t };
    }
};
