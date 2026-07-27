// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const types = @import("../../backend/types.zig");
const layer_mod = @import("layer.zig");

pub const Builder = layer_mod.Builder;
pub const TensorRef = layer_mod.TensorRef;

pub const Kind = enum {
    relu,
    gelu,
    silu,
    sigmoid,
    tanh,
    sqrt,
    log,

    pub fn toUnaryOp(self: Kind) types.UnaryOp {
        return switch (self) {
            .relu => .relu,
            .gelu => .gelu,
            .silu => .silu,
            .sigmoid => .sigmoid,
            .tanh => .tanh,
            .sqrt => .sqrt,
            .log => .log,
        };
    }
};

pub fn apply(bld: *Builder, kind: Kind, x: TensorRef) Builder.Error!TensorRef {
    return bld.unary(kind.toUnaryOp(), x);
}

/// A stateless activation as a module, so it composes inside `Sequential`.
///
/// `Activation{ .kind = .gelu }` needs no `bind` — there is nothing to bind — but
/// it still exposes one so it satisfies the `bind`+`forward` module protocol.
pub const Activation = struct {
    kind: Kind,

    const Self = @This();

    pub fn bind(bld: *Builder, kind: Kind) Builder.Error!Self {
        _ = bld;
        return .{ .kind = kind };
    }

    /// Deliberately does not open a module scope: an activation adds no parameters
    /// and one op, so a `Gelu#0/gelu#0` level would only deepen every path.
    pub fn forward(self: Self, bld: *Builder, x: TensorRef) Builder.Error!TensorRef {
        return apply(bld, self.kind, x);
    }
};

pub fn relu(bld: *Builder, x: TensorRef) Builder.Error!TensorRef {
    return bld.unary(.relu, x);
}

pub fn gelu(bld: *Builder, x: TensorRef) Builder.Error!TensorRef {
    return bld.unary(.gelu, x);
}

pub fn silu(bld: *Builder, x: TensorRef) Builder.Error!TensorRef {
    return bld.unary(.silu, x);
}

pub fn sigmoid(bld: *Builder, x: TensorRef) Builder.Error!TensorRef {
    return bld.unary(.sigmoid, x);
}

pub fn tanh(bld: *Builder, x: TensorRef) Builder.Error!TensorRef {
    return bld.unary(.tanh, x);
}

pub fn log(bld: *Builder, x: TensorRef) Builder.Error!TensorRef {
    return bld.unary(.log, x);
}

pub fn softmaxLastDim(bld: *Builder, x: TensorRef) Builder.Error!TensorRef {
    return bld.softmax(x, -1);
}
