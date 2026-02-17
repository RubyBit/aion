const std = @import("std");

const api_builder = @import("builder.zig");
const api_tensor = @import("tensor.zig");

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
        groups: usize = 1,
    };

    pub fn bind(bld: *Builder, w: Tensor, bias: ?Tensor, opts: Options) Builder.Error!Self {
        const w_ref: TensorRef = try bld.param(w);
        const b_ref: ?TensorRef = if (bias) |bt| try bld.param(bt) else null;
        return .{ .w = w_ref, .b = b_ref, .opts = opts };
    }

    /// NHWC conv2d (channel-last) with weight layout [k_h, k_w, c_in/groups, c_out].
    pub fn forward(self: Self, bld: *Builder, x: TensorRef) Builder.Error!TensorRef {
        return bld.conv2d(
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

pub fn softmaxLastDim(bld: *Builder, x: TensorRef) Builder.Error!TensorRef {
    return bld.softmax(x, -1);
}
