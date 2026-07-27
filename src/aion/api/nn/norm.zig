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

const MAX_RANK: usize = 8;

/// Trailing dims a norm reduces over, stored inline so the layer owns no memory.
const NormShape = struct {
    dims: [MAX_RANK]usize = @splat(0),
    len: usize = 0,

    fn fromSlice(shape: []const usize) Builder.Error!NormShape {
        if (shape.len == 0 or shape.len > MAX_RANK) return Builder.Error.InvalidArgument;
        var out: NormShape = .{ .len = shape.len };
        for (shape, 0..) |d, i| {
            if (d == 0) return Builder.Error.InvalidArgument;
            out.dims[i] = d;
        }
        return out;
    }

    fn slice(self: *const NormShape) []const usize {
        return self.dims[0..self.len];
    }

    fn elemCount(self: *const NormShape) Builder.Error!usize {
        var acc: usize = 1;
        for (self.slice()) |d| {
            acc = std.math.mul(usize, acc, d) catch return Builder.Error.InvalidArgument;
        }
        return acc;
    }
};

/// Both are optional: an absent one becomes the shared identity vector the Builder
/// synthesizes, so scale-only norms and fully parameterless ones are the same type.
pub const NormWeights = struct {
    weight: ?Tensor = null,
    bias: ?Tensor = null,
};

pub const Options = struct {
    /// Scope segment, used when `Params` does not name the layer.
    name: ?[]const u8 = null,
    /// Numerical floor added under the square root. The default differs per norm
    /// (see `LayerNorm`/`RMSNorm`), matching the Builder op defaults.
    eps: ?f32 = null,
    /// Trailing dims to normalize over. Defaults to the resolved `weight`'s shape;
    /// required when there is no `weight`, since nothing else determines the width.
    normalized_shape: ?[]const usize = null,
};

/// `y = (x - mean) / sqrt(var + eps) * gamma + beta` over the trailing dims.
///
/// `weight`/`bias` are both optional: an absent one becomes the shared identity
/// vector the Builder synthesizes. The underlying op takes both operands
/// unconditionally, so this is where "LayerNorm without a bias" stops being the
/// caller's problem.
pub const LayerNorm = struct {
    gamma: TensorRef,
    beta: TensorRef,
    eps: f32,
    shape: NormShape,
    id: LayerName = .{},

    const Self = @This();
    const default_eps: f32 = 1e-5;

    pub const Weights = NormWeights;

    pub fn bind(bld: *Builder, params: anytype, opts: Options) BindError!Self {
        var src = state_mod.binding(Weights, params);
        return bindNorm(Self, bld, src.params(), opts, default_eps);
    }

    pub fn forward(self: Self, bld: *Builder, x: TensorRef) Builder.Error!TensorRef {
        const scope = try self.id.enter(bld);
        defer bld.endScope(scope);
        return bld.layernorm(x, self.gamma, self.beta, self.eps, self.shape.slice());
    }
};

/// `y = x / sqrt(mean(x^2) + eps) * gamma + beta` over the trailing dims.
///
/// Same optional-parameter handling as `LayerNorm`. Scale-only RMSNorm (no bias)
/// is the common transformer case, and a parameterless RMSNorm (neither, e.g. a
/// V-norm) needs only `opts.normalized_shape`.
pub const RMSNorm = struct {
    gamma: TensorRef,
    beta: TensorRef,
    eps: f32,
    shape: NormShape,
    id: LayerName = .{},

    const Self = @This();
    const default_eps: f32 = 1e-6;

    pub const Weights = NormWeights;

    pub fn bind(bld: *Builder, params: anytype, opts: Options) BindError!Self {
        var src = state_mod.binding(Weights, params);
        return bindNorm(Self, bld, src.params(), opts, default_eps);
    }

    pub fn forward(self: Self, bld: *Builder, x: TensorRef) Builder.Error!TensorRef {
        const scope = try self.id.enter(bld);
        defer bld.endScope(scope);
        return bld.rmsnorm(x, self.gamma, self.beta, self.eps, self.shape.slice());
    }
};

/// Shared bind for both norms: resolve whichever parameters the source has, then
/// fill the gaps with the Builder's shared identity constants.
fn bindNorm(
    comptime Self: type,
    bld: *Builder,
    p: Params,
    opts: Options,
    default_eps: f32,
) BindError!Self {
    const eps: f32 = opts.eps orelse default_eps;
    if (!(eps > 0.0) or !std.math.isFinite(eps)) return error.InvalidArgument;

    const id, const scope = try LayerName.open(Self, bld, p.name orelse opts.name);
    defer bld.endScope(scope);

    const gamma: ?TensorRef = try p.getOpt(bld, NormWeights, .weight);
    const beta: ?TensorRef = try p.getOpt(bld, NormWeights, .bias);

    const shape: NormShape = blk: {
        if (opts.normalized_shape) |ns| break :blk try NormShape.fromSlice(ns);
        if (gamma) |g| break :blk try NormShape.fromSlice(bld.knownShape(g) orelse return error.InvalidArgument);
        // No weight and no explicit shape: nothing determines the width.
        return error.InvalidArgument;
    };
    const width: usize = try shape.elemCount();

    if (gamma) |g| try requireShape(bld, g, shape);
    if (beta) |b| try requireShape(bld, b, shape);

    return .{
        // Synthesized identities are bound outside this layer's scope (see
        // `Builder.zeros`/`ones`) so they stay shared across the model.
        .gamma = gamma orelse try bld.ones(width),
        .beta = beta orelse try bld.zeros(width),
        .eps = eps,
        .shape = shape,
        .id = id,
    };
}

/// A norm parameter must match the normalized dims exactly. Rank-1 `[width]` is
/// also accepted for a multi-dim `normalized_shape`, since that is how a flattened
/// gamma is usually stored.
fn requireShape(bld: *Builder, ref: TensorRef, shape: NormShape) Builder.Error!void {
    const got: []const usize = bld.knownShape(ref) orelse return Builder.Error.InvalidArgument;
    if (got.len == shape.len) {
        for (got, 0..) |d, i| {
            if (d != shape.dims[i]) return Builder.Error.InvalidArgument;
        }
        return;
    }
    if (got.len == 1 and got[0] == try shape.elemCount()) return;
    return Builder.Error.InvalidArgument;
}
