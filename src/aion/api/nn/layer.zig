// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const api_builder = @import("../builder.zig");
const api_tensor = @import("../tensor.zig");
const module_api = @import("../module.zig");

pub const Builder = api_builder.Builder;
pub const TensorRef = api_builder.TensorRef;
pub const Tensor = api_tensor.Tensor;
pub const Context = api_builder.Context;

/// A layer's identity: the scope segment it was bound under.
///
/// Layers establish this in `bind` and re-enter it in `forward`, so a layer's
/// parameters and its ops share one path (`Linear#0/weight`, `Linear#0/matmul#0`).
/// Without it, `bind` — which runs outside any scope — would name every `Linear`'s
/// weight just `weight` and they would collide instead of identifying anything.
///
/// Note the tradeoff of leaving `name` unset: auto-numbered segments are assigned in
/// bind order, so inserting a layer ahead of others shifts their numbers and
/// re-keys their parameters. Pass an explicit name for anything you intend to look
/// up or swap in a loaded model.
pub const LayerName = struct {
    /// Arena-owned, so it lives as long as the graph.
    segment: []const u8 = "",

    /// Open the layer's scope during `bind`. `explicit` names the layer; otherwise
    /// it is auto-numbered from the type name (`Linear#0`, `Linear#1`, ... per
    /// parent scope).
    pub fn open(comptime T: type, bld: *Builder, explicit: ?[]const u8) Builder.Error!struct { LayerName, Builder.Scope } {
        const scope: Builder.Scope = if (explicit) |n|
            try bld.beginScope(n)
        else
            try bld.beginAutoScope(module_api.moduleTypeName(T));
        return .{ .{ .segment = scope.name }, scope };
    }

    /// Re-enter the scope this layer was bound under.
    ///
    /// An unset identity (a layer built by struct literal rather than `bind`) opens
    /// nothing and returns a scope whose `endScope` is a no-op, so such a layer's
    /// ops are named in its parent's scope instead of under an invented segment.
    pub fn enter(self: LayerName, bld: *Builder) Builder.Error!Builder.Scope {
        if (self.segment.len == 0) {
            return .{ .name = "", .path_len = bld.scopePath().len, .depth = bld.scopeDepth() };
        }
        return bld.beginScope(self.segment);
    }
};
