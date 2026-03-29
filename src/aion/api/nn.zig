const std = @import("std");

const api_builder = @import("builder.zig");
const api_tensor = @import("tensor.zig");
const types = @import("../backend/types.zig");

pub const Builder = api_builder.Builder;
pub const TensorRef = api_builder.TensorRef;
pub const Tensor = api_tensor.Tensor;

/// Simple neural-network helper layers built on top of `api.Builder`.
///
/// Design goals:
/// - Keep the core API graph-hidden and tensor-first.
/// - Provide ergonomic wrappers that bind parameters once (`bind`) and then
///   apply them multiple times (`forward`).
/// - Avoid runtime allocations in `forward`.
pub const Linear = struct {
    w: TensorRef,
    b: ?TensorRef = null,

    const Self = @This();

    pub fn bind(bld: *Builder, w: Tensor, bias: ?Tensor) Builder.Error!Self {
        const w_ref: TensorRef = try bld.param(w);
        const b_ref: ?TensorRef = if (bias) |bt| try bld.param(bt) else null;
        return .{ .w = w_ref, .b = b_ref };
    }

    /// y = x @ w + b (bias broadcast over last dim if present)
    pub fn forward(self: Self, bld: *Builder, x: TensorRef) Builder.Error!TensorRef {
        const y: TensorRef = try bld.matmul(x, self.w, 1.0, 0.0);
        if (self.b) |b0| {
            return bld.broadcastAddLastDim(y, b0);
        }
        return y;
    }
};

pub const Conv2D = struct {
    w: TensorRef,
    b: ?TensorRef = null,
    opts: Options = .{},

    const Self = @This();

    pub const Options = struct {
        stride_h: usize = 1,
        stride_w: usize = 1,
        dilation_h: usize = 1,
        dilation_w: usize = 1,
        pad_top: usize = 0,
        pad_bottom: usize = 0,
        pad_left: usize = 0,
        pad_right: usize = 0,
        pad_mode: types.PadMode = .zero,
        groups: usize = 1,
    };

    pub fn bind(bld: *Builder, w: Tensor, bias: ?Tensor, opts: Options) Builder.Error!Self {
        const w_ref: TensorRef = try bld.param(w);
        const b_ref: ?TensorRef = if (bias) |bt| try bld.param(bt) else null;
        return .{ .w = w_ref, .b = b_ref, .opts = opts };
    }

    /// NHWC conv2d (channel-last) with weight layout [k_h, k_w, c_in/groups, c_out].
    pub fn forward(self: Self, bld: *Builder, x: TensorRef) Builder.Error!TensorRef {
        return bld.conv2dPadMode(
            x,
            self.w,
            self.b,
            self.opts.stride_h,
            self.opts.stride_w,
            self.opts.dilation_h,
            self.opts.dilation_w,
            self.opts.pad_top,
            self.opts.pad_bottom,
            self.opts.pad_left,
            self.opts.pad_right,
            self.opts.pad_mode,
            self.opts.groups,
        );
    }
};

/// A tiny residual block: conv -> relu -> conv -> add(skip) -> relu.
pub const ResBlock2D = struct {
    conv1: Conv2D,
    conv2: Conv2D,

    const Self = @This();

    pub fn forward(self: Self, bld: *Builder, x: TensorRef) Builder.Error!TensorRef {
        const y1: TensorRef = try self.conv1.forward(bld, x);
        const a1: TensorRef = try bld.relu(y1);
        const y2: TensorRef = try self.conv2.forward(bld, a1);
        const sum: TensorRef = try bld.add(y2, x);
        return bld.relu(sum);
    }
};

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
pub const LSTMCell = struct {
    w_ih: TensorRef,
    w_hh: TensorRef,
    b_ih: ?TensorRef = null,
    b_hh: ?TensorRef = null,

    input_size: usize,
    hidden_size: usize,

    const Self = @This();

    pub fn bind(bld: *Builder, w_ih: Tensor, w_hh: Tensor, b_ih: ?Tensor, b_hh: ?Tensor) Builder.Error!Self {
        // Bias policy: either both provided or both omitted.
        if ((b_ih != null) != (b_hh != null)) return Builder.Error.InvalidArgument;

        if (w_ih.shape.len != 2) return Builder.Error.InvalidArgument;
        if (w_hh.shape.len != 2) return Builder.Error.InvalidArgument;

        const input_size: usize = w_ih.shape[0];
        const gate_dim: usize = w_ih.shape[1];
        if (input_size == 0 or gate_dim == 0) return Builder.Error.InvalidArgument;
        if (gate_dim % 4 != 0) return Builder.Error.InvalidArgument;

        const hidden_size: usize = gate_dim / 4;
        if (hidden_size == 0) return Builder.Error.InvalidArgument;

        const want_w_hh0: usize = hidden_size;
        const want_w_hh1: usize = std.math.mul(usize, hidden_size, 4) catch return Builder.Error.InvalidArgument;
        if (w_hh.shape[0] != want_w_hh0 or w_hh.shape[1] != want_w_hh1) return Builder.Error.InvalidArgument;

        if (b_ih) |bt| {
            if (bt.shape.len != 1) return Builder.Error.InvalidArgument;
            if (bt.shape[0] != gate_dim) return Builder.Error.InvalidArgument;
        }
        if (b_hh) |bt| {
            if (bt.shape.len != 1) return Builder.Error.InvalidArgument;
            if (bt.shape[0] != gate_dim) return Builder.Error.InvalidArgument;
        }

        const w_ih_ref: TensorRef = try bld.param(w_ih);
        const w_hh_ref: TensorRef = try bld.param(w_hh);
        const b_ih_ref: ?TensorRef = if (b_ih) |bt| try bld.param(bt) else null;
        const b_hh_ref: ?TensorRef = if (b_hh) |bt| try bld.param(bt) else null;

        return .{
            .w_ih = w_ih_ref,
            .w_hh = w_hh_ref,
            .b_ih = b_ih_ref,
            .b_hh = b_hh_ref,
            .input_size = input_size,
            .hidden_size = hidden_size,
        };
    }

    pub fn forward(self: Self, bld: *Builder, x: TensorRef, h_prev: TensorRef, c_prev: TensorRef) Builder.Error!LSTMState {
        const x_shape: []const usize = try bld.requireKnownShape(x);
        const h_shape: []const usize = try bld.requireKnownShape(h_prev);
        const c_shape: []const usize = try bld.requireKnownShape(c_prev);

        if (x_shape.len != 2 or h_shape.len != 2 or c_shape.len != 2) return Builder.Error.InvalidArgument;

        const batch: usize = x_shape[0];
        if (batch == 0) return Builder.Error.InvalidArgument;
        if (x_shape[1] != self.input_size) return Builder.Error.InvalidArgument;
        if (h_shape[0] != batch or c_shape[0] != batch) return Builder.Error.InvalidArgument;
        if (h_shape[1] != self.hidden_size or c_shape[1] != self.hidden_size) return Builder.Error.InvalidArgument;

        // gates = x @ w_ih + h_prev @ w_hh + b_ih + b_hh
        const gx: TensorRef = try bld.matmul(x, self.w_ih, 1.0, 0.0);
        const gh: TensorRef = try bld.matmul(h_prev, self.w_hh, 1.0, 0.0);
        var gates: TensorRef = try bld.add(gx, gh);
        if (self.b_ih) |b0| {
            gates = try bld.broadcastAddLastDim(gates, b0);
        }
        if (self.b_hh) |b1| {
            gates = try bld.broadcastAddLastDim(gates, b1);
        }

        // Split [batch, 4h] -> 4 x [batch, h]
        const h: usize = self.hidden_size;
        const i_lin: TensorRef = try bld.slice2d(gates, 0, batch, 0 * h, h);
        const f_lin: TensorRef = try bld.slice2d(gates, 0, batch, 1 * h, h);
        const g_lin: TensorRef = try bld.slice2d(gates, 0, batch, 2 * h, h);
        const o_lin: TensorRef = try bld.slice2d(gates, 0, batch, 3 * h, h);

        const i_gate: TensorRef = try bld.sigmoid(i_lin);
        const f_gate: TensorRef = try bld.sigmoid(f_lin);
        const g_gate: TensorRef = try bld.tanh(g_lin);
        const o_gate: TensorRef = try bld.sigmoid(o_lin);

        const fc: TensorRef = try bld.mul(f_gate, c_prev);
        const ig: TensorRef = try bld.mul(i_gate, g_gate);
        const c_t: TensorRef = try bld.add(fc, ig);

        const tanh_c: TensorRef = try bld.tanh(c_t);
        const h_t: TensorRef = try bld.mul(o_gate, tanh_c);

        return .{ .h = h_t, .c = c_t };
    }
};

pub fn softmaxLastDim(bld: *Builder, x: TensorRef) Builder.Error!TensorRef {
    return bld.softmax(x, -1);
}
