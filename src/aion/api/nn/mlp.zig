// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const layer_mod = @import("layer.zig");
const state_mod = @import("state.zig");
const linear_mod = @import("linear.zig");
const activation = @import("activation.zig");

pub const Builder = layer_mod.Builder;
pub const TensorRef = layer_mod.TensorRef;
pub const Tensor = layer_mod.Tensor;
pub const LayerName = layer_mod.LayerName;
pub const Params = state_mod.Params;
pub const BindError = state_mod.BindError;
pub const Linear = linear_mod.Linear;
pub const Kind = activation.Kind;

/// `w2(act(w1(x)))` — the plain two-matmul feed-forward block.
pub const FeedForward = struct {
    fc1: Linear,
    fc2: Linear,
    act: Kind = .silu,
    id: LayerName = .{},

    const Self = @This();

    pub const Weights = struct {
        fc1: Linear.Weights,
        fc2: Linear.Weights,
    };

    pub const Options = struct {
        /// Scope segment for this block, prefixing both projections.
        name: ?[]const u8 = null,
        act: Kind = .silu,
    };

    pub fn bind(bld: *Builder, params: anytype, opts: Options) BindError!Self {
        var src = state_mod.binding(Weights, params);
        const p: Params = src.params();

        const id, const scope = try LayerName.open(Self, bld, p.name orelse opts.name);
        defer bld.endScope(scope);

        return .{
            .fc1 = try Linear.bind(bld, p.child(Weights, .fc1), .{}),
            .fc2 = try Linear.bind(bld, p.child(Weights, .fc2), .{}),
            .act = opts.act,
            .id = id,
        };
    }

    pub fn forward(self: Self, bld: *Builder, x: TensorRef) Builder.Error!TensorRef {
        const scope = try self.id.enter(bld);
        defer bld.endScope(scope);

        const h: TensorRef = try self.fc1.forward(bld, x);
        const a: TensorRef = try activation.apply(bld, self.act, h);
        return self.fc2.forward(bld, a);
    }
};

/// `down(act(gate(x)) * up(x))` — the gated feed-forward used by SwiGLU (`.silu`)
/// and GeGLU (`.gelu`) transformers.
///
/// `gate` and `up` are two separate projections, which is how every checkpoint
/// ships them. Pre-concatenating the pair into one `[in, 2*ffn]` weight is a
/// *fusion*, not a layout, and it is the compiler's: `opt/fuse_horizontal_matmul`
/// rewrites matmuls sharing an operand into one wide matmul plus a slice each,
/// numerically identically. So there is nothing to choose here and nothing to
/// probe for — one shape, one code path.
///
/// The gelu case routes through the fused `geluMul` op rather than a separate
/// unary and multiply.
pub const GatedMLP = struct {
    gate: Linear,
    up: Linear,
    down: Linear,
    act: Kind,
    id: LayerName = .{},

    const Self = @This();

    pub const Weights = struct {
        gate_proj: Linear.Weights,
        up_proj: Linear.Weights,
        down_proj: Linear.Weights,
    };

    pub const Options = struct {
        /// Scope segment for this block, prefixing its projections.
        name: ?[]const u8 = null,
        act: Kind = .silu,
    };

    pub fn bind(bld: *Builder, params: anytype, opts: Options) BindError!Self {
        var src = state_mod.binding(Weights, params);
        const p: Params = src.params();

        const id, const scope = try LayerName.open(Self, bld, p.name orelse opts.name);
        defer bld.endScope(scope);

        const gate = try Linear.bind(bld, p.child(Weights, .gate_proj), .{});
        const up = try Linear.bind(bld, p.child(Weights, .up_proj), .{});
        const down = try Linear.bind(bld, p.child(Weights, .down_proj), .{});

        // The two halves are multiplied elementwise, so a width mismatch is a wiring
        // error that would otherwise surface as a confusing shape failure downstream.
        const ffn = lastDim(bld, gate.w) orelse return error.InvalidArgument;
        if (ffn == 0 or (lastDim(bld, up.w) orelse 0) != ffn) return error.InvalidArgument;

        return .{ .gate = gate, .up = up, .down = down, .act = opts.act, .id = id };
    }

    pub fn forward(self: Self, bld: *Builder, x: TensorRef) Builder.Error!TensorRef {
        const scope = try self.id.enter(bld);
        defer bld.endScope(scope);

        const gate_v: TensorRef = try self.gate.forward(bld, x);
        const up_v: TensorRef = try self.up.forward(bld, x);

        // `geluMul` fuses gelu(a)*b; the converters hand-rolled the pair and missed
        // this op entirely.
        const gated: TensorRef = if (self.act == .gelu)
            try bld.geluMul(gate_v, up_v)
        else
            try bld.mul(try activation.apply(bld, self.act, gate_v), up_v);

        return self.down.forward(bld, gated);
    }
};

/// `a * sigmoid(b)` over a projection split in half along the last dim — the GLU
/// in a Conformer convolution module.
pub const GLU = struct {
    proj: Linear,
    half: usize,
    id: LayerName = .{},

    const Self = @This();

    pub const Weights = struct {
        proj: Linear.Weights,
    };

    pub const Options = struct {
        name: ?[]const u8 = null,
    };

    pub fn bind(bld: *Builder, params: anytype, opts: Options) BindError!Self {
        var src = state_mod.binding(Weights, params);
        const p: Params = src.params();

        const id, const scope = try LayerName.open(Self, bld, p.name orelse opts.name);
        defer bld.endScope(scope);

        const proj = try Linear.bind(bld, p.child(Weights, .proj), .{});
        const total: usize = lastDim(bld, proj.w) orelse return error.InvalidArgument;
        if (total == 0 or total % 2 != 0) return error.InvalidArgument;

        return .{ .proj = proj, .half = total / 2, .id = id };
    }

    pub fn forward(self: Self, bld: *Builder, x: TensorRef) Builder.Error!TensorRef {
        const scope = try self.id.enter(bld);
        defer bld.endScope(scope);

        const both: TensorRef = try self.proj.forward(bld, x);
        const a: TensorRef = try bld.sliceLastDim(both, 0, self.half);
        const b: TensorRef = try bld.sliceLastDim(both, self.half, self.half);
        return bld.mul(a, try bld.sigmoid(b));
    }
};

fn lastDim(bld: *Builder, ref: TensorRef) ?usize {
    const shape = bld.knownShape(ref) orelse return null;
    if (shape.len == 0) return null;
    return shape[shape.len - 1];
}

