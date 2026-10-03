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

/// `y = x @ wᵀ + b`, with `w` as `[out, in]`: the layout torch and GGUF store, so a
/// checkpoint's weight binds as it is. The compiler turns the transposed read into
/// a contraction against `w`'s rows; nothing is copied.
pub const Linear = struct {
    w: TensorRef,
    b: ?TensorRef = null,
    opts: Options = .{},
    id: LayerName = .{},

    const Self = @This();

    /// `weight` has no default, so a `Linear` that was handed no weight does not
    /// compile rather than failing at bind time.
    pub const Weights = struct {
        weight: Tensor,
        bias: ?Tensor = null,
    };

    pub const Options = struct {
        /// Scope segment for this layer, used when `Params` does not name it.
        /// Leave null to auto-number from the type (`Linear#0`, `Linear#1`, ...).
        name: ?[]const u8 = null,
        /// Output scale, folded into the matmul (`alpha * x @ w`). Free — the op
        /// already carries it, so prefer this over a separate multiply.
        alpha: f32 = 1.0,
        /// Accumulator pre-scale on the matmul.
        beta: f32 = 0.0,
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

    /// Reuse an already-bound weight instead of resolving one, for weight tying: an
    /// `Embedding`'s `[vocab, dim]` table is already a `[out, in]` output head.
    pub fn bindShared(bld: *Builder, w: TensorRef, bias: ?TensorRef, opts: Options) BindError!Self {
        const id, const scope = try LayerName.open(Self, bld, opts.name);
        defer bld.endScope(scope);
        return .{ .w = w, .b = bias, .opts = opts, .id = id };
    }

    pub fn forward(self: Self, bld: *Builder, x: TensorRef) Builder.Error!TensorRef {
        const scope = try self.id.enter(bld);
        defer bld.endScope(scope);

        // `matmul` aligns operand ranks itself, so the `[in, out]` view works
        // against a `[batch, seq, in]` activation with no reshape here.
        const y: TensorRef = try bld.matmul(x, try bld.transpose2d(self.w), self.opts.alpha, self.opts.beta);

        if (self.b) |b0| return bld.add(y, b0);
        return y;
    }
};

/// Row lookup into a `[vocab, dim]` table: `out[b, t, :] = table[ids[b, t], :]`.
///
/// The table may be quantized (a q8_0 table blocked along `dim` is the usual
/// choice); `weightRef` hands it back so a tied output head can reuse it via
/// `Linear.bindShared(bld, emb.weightRef(), null, .{})`.
pub const Embedding = struct {
    table: TensorRef,
    id: LayerName = .{},

    const Self = @This();

    pub const Weights = struct {
        weight: Tensor,
    };

    pub const Options = struct {
        name: ?[]const u8 = null,
    };

    pub fn bind(bld: *Builder, params: anytype, opts: Options) BindError!Self {
        var src = state_mod.binding(Weights, params);
        const p: Params = src.params();

        const id, const scope = try LayerName.open(Self, bld, p.name orelse opts.name);
        defer bld.endScope(scope);

        const table = try p.get(bld, Weights, .weight);
        const shape = bld.knownShape(table) orelse return error.InvalidArgument;
        if (shape.len != 2) return error.InvalidArgument;

        return .{ .table = table, .id = id };
    }

    pub fn weightRef(self: Self) TensorRef {
        return self.table;
    }

    /// `ids` is i32 `[.., seq]`; the output gains a trailing `dim` axis.
    pub fn forward(self: Self, bld: *Builder, ids: TensorRef) Builder.Error!TensorRef {
        const scope = try self.id.enter(bld);
        defer bld.endScope(scope);
        return bld.gather(self.table, ids, 0, 0);
    }
};
