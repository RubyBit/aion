// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const layer_mod = @import("layer.zig");

pub const Builder = layer_mod.Builder;
pub const TensorRef = layer_mod.TensorRef;
pub const LayerName = layer_mod.LayerName;

/// Chain single-input/single-output layers held in a struct, calling `forward` on
/// each field in declaration order.
///
/// `Mods` is a struct type whose fields are layers, which keeps every layer's
/// concrete type — no vtables, no allocation, no runtime dispatch:
///
/// ```zig
/// const Mlp = nn.Sequential(struct {
///     fc1: nn.Linear,
///     act: nn.Activation,
///     fc2: nn.Linear,
/// });
/// const mlp: Mlp = .init(.{
///     .fc1 = try nn.Linear.bind(&env, w1, b1, .{ .name = "fc1" }),
///     .act = .{ .kind = .relu },
///     .fc2 = try nn.Linear.bind(&env, w2, b2, .{ .name = "fc2" }),
/// });
/// const y = try mlp.forward(&bld, x);
/// ```
///
/// It adds no scope of its own: each layer already carries the identity it was
/// bound under, so wrapping would desync a layer's parameter names (fixed at
/// `bind`) from its op names. Pass `.name` when binding to choose the path.
///
/// Use `ModuleDyn` instead when the layer list is only known at runtime.
pub fn Sequential(comptime Mods: type) type {
    const info = @typeInfo(Mods);
    if (info != .@"struct") @compileError("nn.Sequential expects a struct of layers");
    if (info.@"struct".fields.len == 0) @compileError("nn.Sequential needs at least one layer");

    return struct {
        mods: Mods,

        const Self = @This();

        pub fn init(mods: Mods) Self {
            return .{ .mods = mods };
        }

        /// Present so the type satisfies the `bind`+`forward` module protocol.
        pub fn bind(mods: Mods) Self {
            return init(mods);
        }

        pub fn forward(self: Self, bld: *Builder, x: TensorRef) Builder.Error!TensorRef {
            var cur: TensorRef = x;
            inline for (info.@"struct".fields) |field| {
                cur = try @field(self.mods, field.name).forward(bld, cur);
            }
            return cur;
        }
    };
}

/// `x + scale * body(x)`.
///
/// `scale` covers the macaron-Conformer convention of adding half of each
/// feed-forward branch. It is bound as a shared one-element constant, so it costs
/// one float for the whole model rather than a vector per use.
pub fn Residual(comptime Body: type) type {
    return struct {
        body: Body,
        scale: TensorRef = .{ .value = 0 },
        has_scale: bool = false,

        const Self = @This();

        pub fn bind(bld: *Builder, body: Body, scale_factor: f32) Builder.Error!Self {
            if (!std.math.isFinite(scale_factor)) return Builder.Error.InvalidArgument;
            if (scale_factor == 1.0) return .{ .body = body };
            return .{ .body = body, .scale = try bld.constant(scale_factor), .has_scale = true };
        }

        /// Adds no scope of its own — `body` already carries its identity.
        pub fn forward(self: Self, bld: *Builder, x: TensorRef) Builder.Error!TensorRef {
            const y: TensorRef = try self.body.forward(bld, x);
            const scaled: TensorRef = if (self.has_scale)
                try bld.broadcastLastDim(.mul, y, self.scale)
            else
                y;
            return bld.add(x, scaled);
        }
    };
}
