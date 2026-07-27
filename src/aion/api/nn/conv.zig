// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const types = @import("../../backend/types.zig");
const layer_mod = @import("layer.zig");
const state_mod = @import("state.zig");

pub const Builder = layer_mod.Builder;
pub const TensorRef = layer_mod.TensorRef;
pub const Tensor = layer_mod.Tensor;
pub const LayerName = layer_mod.LayerName;
pub const Params = state_mod.Params;
pub const BindError = state_mod.BindError;
pub const PadMode = types.PadMode;

/// NLC (channel-last) conv1d with weight layout `[k, c_in/groups, c_out]`.
pub const Conv1D = struct {
    w: TensorRef,
    b: ?TensorRef = null,
    opts: Options = .{},
    id: LayerName = .{},

    const Self = @This();

    pub const Weights = struct {
        weight: Tensor,
        bias: ?Tensor = null,
    };

    pub const Options = struct {
        /// Scope segment, used when `Params` does not name the layer.
        name: ?[]const u8 = null,
        stride: usize = 1,
        dilation: usize = 1,
        pad_left: usize = 0,
        pad_right: usize = 0,
        pad_mode: PadMode = .zero,
        /// 0 means "one filter per channel", derived from the weight — see
        /// `DepthwiseConv1D`, which is this with `groups` fixed that way.
        groups: usize = 1,
    };

    pub fn bind(bld: *Builder, params: anytype, opts: Options) BindError!Self {
        var src = state_mod.binding(Weights, params);
        const p: Params = src.params();

        const id, const scope = try LayerName.open(Self, bld, p.name orelse opts.name);
        defer bld.endScope(scope);

        const w = try p.get(bld, Weights, .weight);
        var o = opts;
        if (o.groups == 0) o.groups = try depthwiseGroups(bld, w);

        return .{ .w = w, .b = try p.getOpt(bld, Weights, .bias), .opts = o, .id = id };
    }

    pub fn forward(self: Self, bld: *Builder, x: TensorRef) Builder.Error!TensorRef {
        const scope = try self.id.enter(bld);
        defer bld.endScope(scope);

        return bld.conv1dPadMode(
            x,
            self.w,
            self.b,
            self.opts.stride,
            self.opts.dilation,
            self.opts.pad_left,
            self.opts.pad_right,
            self.opts.pad_mode,
            self.opts.groups,
        );
    }

    /// Left-only padding of `(k - 1) * dilation`, so no output frame depends on a
    /// future input frame. This is the padding a streaming/causal model wants, and
    /// getting it off by one is a classic source of train/serve skew.
    pub fn causalPad(kernel: usize, dilation: usize) Builder.Error!Options {
        if (kernel == 0 or dilation == 0) return Builder.Error.InvalidArgument;
        const pad: usize = std.math.mul(usize, kernel - 1, dilation) catch return Builder.Error.InvalidArgument;
        return .{ .dilation = dilation, .pad_left = pad, .pad_right = 0 };
    }
};

/// Depthwise conv1d: one filter per channel, weight `[k, 1, channels]`.
///
/// Not a separate type — `groups` is simply derived from the weight, so the
/// "groups must equal the channel count" invariant cannot be set wrong.
pub fn depthwise(opts: Conv1D.Options) Conv1D.Options {
    var o = opts;
    o.groups = 0; // resolved from the weight at bind time
    return o;
}

fn depthwiseGroups(bld: *Builder, w: TensorRef) Builder.Error!usize {
    const shape = bld.knownShape(w) orelse return Builder.Error.InvalidArgument;
    // [k, c_in/groups, c_out]: depthwise means c_in/groups == 1.
    if (shape.len != 3 or shape[1] != 1 or shape[2] == 0) return Builder.Error.InvalidArgument;
    return shape[2];
}

/// NHWC conv2d with weight layout `[k_h, k_w, c_in/groups, c_out]`.
pub const Conv2D = struct {
    w: TensorRef,
    b: ?TensorRef = null,
    opts: Options = .{},
    id: LayerName = .{},

    const Self = @This();

    pub const Weights = struct {
        weight: Tensor,
        bias: ?Tensor = null,
    };

    pub const Options = struct {
        /// Scope segment, used when `Params` does not name the layer.
        name: ?[]const u8 = null,
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
    };

    pub fn bind(bld: *Builder, params: anytype, opts: Options) BindError!Self {
        var src = state_mod.binding(Weights, params);
        const p: Params = src.params();

        const id, const scope = try LayerName.open(Self, bld, p.name orelse opts.name);
        defer bld.endScope(scope);

        return .{
            .w = try p.get(bld, Weights, .weight),
            .b = try p.getOpt(bld, Weights, .bias),
            .opts = opts,
            .id = id,
        };
    }

    pub fn forward(self: Self, bld: *Builder, x: TensorRef) Builder.Error!TensorRef {
        const scope = try self.id.enter(bld);
        defer bld.endScope(scope);

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

/// conv -> relu -> conv -> add(skip) -> relu.
pub const ResBlock2D = struct {
    conv1: Conv2D,
    conv2: Conv2D,
    id: LayerName = .{},

    const Self = @This();

    pub const Weights = struct {
        conv1: Conv2D.Weights,
        conv2: Conv2D.Weights,
    };

    pub const Options = struct {
        name: ?[]const u8 = null,
        conv: Conv2D.Options = .{},
    };

    /// Both convolutions come from `p`'s children, so this block loads from a
    /// package exactly the way it binds from tensors.
    pub fn bind(bld: *Builder, params: anytype, opts: Options) BindError!Self {
        var src = state_mod.binding(Weights, params);
        const p: Params = src.params();

        const id, const scope = try LayerName.open(Self, bld, p.name orelse opts.name);
        defer bld.endScope(scope);

        return .{
            .conv1 = try Conv2D.bind(bld, p.child(Weights, .conv1), opts.conv),
            .conv2 = try Conv2D.bind(bld, p.child(Weights, .conv2), opts.conv),
            .id = id,
        };
    }

    pub fn forward(self: Self, bld: *Builder, x: TensorRef) Builder.Error!TensorRef {
        const scope = try self.id.enter(bld);
        defer bld.endScope(scope);

        const y1: TensorRef = try self.conv1.forward(bld, x);
        const a1: TensorRef = try bld.relu(y1);
        const y2: TensorRef = try self.conv2.forward(bld, a1);
        const sum: TensorRef = try bld.add(y2, x);
        return bld.relu(sum);
    }
};
