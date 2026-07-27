// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Where a layer's parameters come from, and walking a bound module's parameters.
//!
//! Every layer declares its parameters as a type — `pub const Weights` — whose
//! shape *is* the layer's shape: a `Tensor` field is a parameter, a struct field is
//! a sub-layer, and the field name is the path segment. `Linear.Weights` is
//! `{ weight, bias }`; `GatedMLP.Weights` is `{ gate_proj, up_proj, gate_up_proj,
//! down_proj }` of `Linear.Weights`.
//!
//! That one declaration does three jobs at once:
//!   - the call site is a plain typed literal, checked by the compiler —
//!     `.{ .down_proj = .{ .weight = d } }`, no path strings and no `@""`;
//!   - a required weight has no default, so omitting it does not compile;
//!   - `bind` names its children *through* the type (`p.child(Weights, .down_proj)`),
//!     so the field name and the graph path cannot drift apart.
//!
//! A package is the other source: `Package.at("mlp")` hands a layer a `Params`
//! resolving the same paths out of a loaded `.aion`. Both reach every layer through
//! the same single `bind`.
const std = @import("std");

const api_errors = @import("../errors.zig");
const api_weights = @import("../weights.zig");
const module_api = @import("../module.zig");
const layer_mod = @import("layer.zig");

pub const Builder = layer_mod.Builder;
pub const TensorRef = layer_mod.TensorRef;
pub const Tensor = layer_mod.Tensor;
pub const Weights = api_weights.Weights;

/// Everything binding a layer can fail with: graph/shape errors, plus the package
/// errors from resolving a name.
pub const BindError = Builder.Error || api_errors.ApiError;

// --------------------------------------------------------------- Params -----

/// A layer's parameter source: it answers "give me the tensor called `weight`".
///
/// Type-erased on purpose — it is what lets one `bind` serve tensors in hand and a
/// loaded package alike, and what lets a caller supply a source of their own by
/// writing a `resolveFn`.
///
/// It also carries `name`, the path segment this layer sits at. That is what makes
/// the lookup name and the scope name *the same value* rather than two strings a
/// caller has to keep in sync: `pkg.at("q_proj")` both scopes the layer and
/// resolves `.../q_proj/weight`.
pub const Params = struct {
    /// Path segment for the layer being built, or null to let it name itself
    /// (from `opts.name`, else auto-numbered from the type).
    name: ?[]const u8 = null,

    ctx: *anyopaque,
    resolveFn: *const fn (ctx: *anyopaque, path: []const u8) BindError!?Tensor,

    const Self = @This();

    /// The source for the sub-layer at `field` of `W`.
    ///
    /// Taking the field rather than a string is the point: the segment `bind` asks
    /// for is the same one the caller filled in, checked by the compiler.
    pub fn child(self: Self, comptime W: type, comptime field: std.meta.FieldEnum(W)) Self {
        comptime assertSubLayer(W, field);
        return .{ .name = @tagName(field), .ctx = self.ctx, .resolveFn = self.resolveFn };
    }

    /// Bind a required parameter. Naming happens here, so every source produces
    /// identical `debug_names`.
    pub fn get(self: Self, bld: *Builder, comptime W: type, comptime field: std.meta.FieldEnum(W)) BindError!TensorRef {
        comptime assertParameter(W, field);
        const name = @tagName(field);
        const t = try self.resolve(bld, name) orelse return error.InvalidArgument;
        return bld.paramNamed(t, name);
    }

    /// Bind an optional parameter (a bias a model may not have), or null.
    pub fn getOpt(self: Self, bld: *Builder, comptime W: type, comptime field: std.meta.FieldEnum(W)) BindError!?TensorRef {
        comptime assertParameter(W, field);
        const name = @tagName(field);
        const t = try self.resolve(bld, name) orelse return null;
        return try bld.paramNamed(t, name);
    }

    /// Whether the source has `<outer>/<leaf>`, without binding anything.
    ///
    /// For a layer whose *architecture* varies with what the source holds — e.g. an
    /// attention layer that only reads a KV cache another layer writes, so it has no
    /// K/V projections. Both segments are named by the caller and checked against
    /// their declared `Weights`, so nothing here depends on field order (an earlier
    /// version derived the leaf from "the child's first required field", which made
    /// declaration order silently load-bearing).
    pub fn hasNested(
        self: Self,
        bld: *Builder,
        comptime W: type,
        comptime outer: std.meta.FieldEnum(W),
        comptime Inner: type,
        comptime leaf: std.meta.FieldEnum(Inner),
    ) BindError!bool {
        comptime assertSubLayer(W, outer);
        comptime assertParameter(Inner, leaf);
        const path = comptime @tagName(outer) ++ "/" ++ @tagName(leaf);
        return (try self.resolve(bld, path)) != null;
    }

    /// Ask the source for `<this layer's path>/<leaf>`.
    fn resolve(self: Self, bld: *Builder, leaf: []const u8) BindError!?Tensor {
        // The builder's scope is already open at this point (the layer opened it),
        // so its path is exactly the prefix the source should be queried with.
        const scope: []const u8 = bld.scopePath();
        if (scope.len == 0) return self.resolveFn(self.ctx, leaf);

        const arena = bld.innerGraph().arenaAlloc();
        const path = std.fmt.allocPrint(arena, "{s}/{s}", .{ scope, leaf }) catch return error.OutOfMemory;
        return self.resolveFn(self.ctx, path);
    }
};

// ------------------------------------------------- weights-tree reflection --

fn unwrapOptional(comptime T: type) type {
    const info = @typeInfo(T);
    return if (info == .optional) info.optional.child else T;
}

fn assertParameter(comptime W: type, comptime field: std.meta.FieldEnum(W)) void {
    const T = unwrapOptional(@FieldType(W, @tagName(field)));
    if (T != Tensor) @compileError(
        @typeName(W) ++ "." ++ @tagName(field) ++ " is a sub-layer, not a parameter" ++
            " — use `child` to descend into it",
    );
}

fn assertSubLayer(comptime W: type, comptime field: std.meta.FieldEnum(W)) void {
    const T = unwrapOptional(@FieldType(W, @tagName(field)));
    if (T == Tensor) @compileError(
        @typeName(W) ++ "." ++ @tagName(field) ++ " is a parameter, not a sub-layer" ++
            " — use `get`/`getOpt` to bind it",
    );
}

/// How many `Tensor` leaves a weights tree can hold.
fn leafCount(comptime W: type) usize {
    var n: usize = 0;
    for (@typeInfo(W).@"struct".fields) |f| {
        const T = unwrapOptional(f.type);
        n += if (T == Tensor) 1 else leafCount(T);
    }
    return n;
}

// --------------------------------------------------------------- Source -----

pub const Entry = struct { name: []const u8, tensor: Tensor };

/// A weights tree flattened to the paths a layer will ask for.
///
/// Built once and reused when several layers share one set of weights — a whole
/// model's, say. A single layer's parameters go straight into its `bind` call
/// instead, which builds one of these on `bind`'s own frame; that is why the caller
/// never has to name a source just to hand over a weight.
pub fn Source(comptime W: type) type {
    return struct {
        entries: [leafCount(W)]Entry = undefined,
        len: usize = 0,

        const Self = @This();
        /// Marks an already-built source, so `binding` routes it as one rather than
        /// mistaking its fields for weights.
        pub const is_param_source = true;

        pub fn from(w: W) Self {
            var out: Self = .{};
            appendLeaves(W, w, "", &out);
            return out;
        }

        fn push(self: *Self, comptime name: []const u8, t: Tensor) void {
            self.entries[self.len] = .{ .name = name, .tensor = t };
            self.len += 1;
        }

        /// A `Params` that lets the layer name itself. The usual call: you already
        /// hold the tensors, so there is no external name to honour.
        pub fn auto(self: *const Self) Params {
            return .{ .name = null, .ctx = @constCast(@ptrCast(self)), .resolveFn = resolve };
        }

        /// A `Params` that puts the layer at `name`.
        pub fn at(self: *const Self, name: []const u8) Params {
            return .{ .name = name, .ctx = @constCast(@ptrCast(self)), .resolveFn = resolve };
        }

        /// An entry answers a query whose path *ends* with it, on a segment
        /// boundary — `down_proj/weight` answers `Block#0/mlp/down_proj/weight`.
        /// Matching the tail is what makes a source independent of where the layer
        /// it feeds ends up in the graph.
        ///
        /// The longest match wins, so a nested entry always beats a shallower one
        /// rather than the answer depending on field order.
        fn resolve(ctx: *anyopaque, path: []const u8) BindError!?Tensor {
            const self: *const Self = @ptrCast(@alignCast(ctx));

            var best: ?Entry = null;
            for (self.entries[0..self.len]) |e| {
                if (!isPathSuffix(path, e.name)) continue;
                if (best) |b| {
                    if (e.name.len <= b.name.len) continue;
                }
                best = e;
            }
            return if (best) |b| b.tensor else null;
        }
    };
}

fn appendLeaves(comptime W: type, w: W, comptime prefix: []const u8, out: anytype) void {
    inline for (@typeInfo(W).@"struct".fields) |f| {
        const path = comptime if (prefix.len == 0) f.name else prefix ++ "/" ++ f.name;
        const value = @field(w, f.name);

        if (@typeInfo(f.type) == .optional) {
            // A null field means the model simply does not have that parameter or
            // sub-layer, so it contributes no entries and every lookup misses it.
            if (value) |v| {
                if (@TypeOf(v) == Tensor) out.push(path, v) else appendLeaves(@TypeOf(v), v, path, out);
            }
        } else if (f.type == Tensor) {
            out.push(path, value);
        } else {
            appendLeaves(f.type, value, path, out);
        }
    }
}

fn isPathSuffix(path: []const u8, suffix: []const u8) bool {
    if (!std.mem.endsWith(u8, path, suffix)) return false;
    // Whole path, or the character before it closes a segment — so `weight` matches
    // `fc/weight` but not `qk_weight`.
    return path.len == suffix.len or path[path.len - suffix.len - 1] == '/';
}

// -------------------------------------------------------------- binding -----

/// Whatever a layer was handed, normalized to a `Params`.
///
/// Layers take `anytype` so that all three spellings reach one `bind`:
///
///     try nn.Linear.bind(&bld, .{ .weight = w, .bias = b }, .{});   // literal
///     try nn.Linear.bind(&bld, shared.at("fc"), .{});               // shared Source
///     try nn.Linear.bind(&bld, pkg.at("fc"), .{});                  // package
///
/// A literal is coerced to the layer's declared `Weights` — that assignment is
/// where field names get checked and a missing required weight becomes a compile
/// error — and flattened into storage held by this value, which the layer keeps on
/// its own frame for the duration of `bind`. That is the whole reason the caller
/// does not have to.
pub fn Binding(comptime W: type, comptime T: type) type {
    if (T == Params) return struct {
        inner: Params,
        pub fn params(self: *const @This()) Params {
            return self.inner;
        }
    };

    if (@typeInfo(T) != .@"struct") @compileError(
        "layer parameters must be a `" ++ @typeName(W) ++ "` literal or a `Params`, got " ++ @typeName(T),
    );

    if (@hasDecl(T, "is_param_source")) return struct {
        inner: T,
        pub fn params(self: *const @This()) Params {
            return self.inner.auto();
        }
    };

    return struct {
        inner: Source(W),
        pub fn params(self: *const @This()) Params {
            return self.inner.auto();
        }
    };
}

/// Normalize a layer's `params` argument. Assign the result to a `var` local in
/// `bind` and call `.params()` on it: the local is what keeps a literal's tensors
/// alive for the rest of the call.
pub fn binding(comptime W: type, v: anytype) Binding(W, @TypeOf(v)) {
    const T = @TypeOf(v);
    if (comptime T == Params) return .{ .inner = v };
    if (comptime @hasDecl(T, "is_param_source")) return .{ .inner = v };
    return .{ .inner = Source(W).from(coerce(W, v)) };
}

/// Map an anonymous literal onto a layer's declared `Weights`.
///
/// This is where the checking happens, and it is done by hand rather than by
/// assignment because a literal reaching an `anytype` parameter has already
/// hardened into its own type, which Zig will not then coerce. Doing it explicitly
/// also lets the errors name the layer and the offending parameter.
fn coerce(comptime W: type, v: anytype) W {
    const V = @TypeOf(v);
    if (V == W) return v;

    if (@typeInfo(V) != .@"struct") @compileError(
        "expected a `" ++ @typeName(W) ++ "` literal, got " ++ @typeName(V),
    );

    inline for (@typeInfo(V).@"struct".fields) |f| {
        if (!@hasField(W, f.name)) @compileError(
            @typeName(W) ++ " has no parameter `" ++ f.name ++ "` (fields: " ++ fieldList(W) ++ ")",
        );
    }

    var out: W = undefined;
    inline for (@typeInfo(W).@"struct".fields) |wf| {
        if (@hasField(V, wf.name)) {
            @field(out, wf.name) = coerceField(wf.type, @field(v, wf.name));
        } else if (wf.default_value_ptr) |dp| {
            @field(out, wf.name) = @as(*const wf.type, @ptrCast(@alignCast(dp))).*;
        } else {
            // No default means the layer cannot run without it.
            @compileError(@typeName(W) ++ " requires `" ++ wf.name ++ "`");
        }
    }
    return out;
}

fn coerceField(comptime F: type, v: anytype) F {
    const V = @TypeOf(v);
    if (V == F) return v;
    if (V == @TypeOf(null)) return null;

    const info = @typeInfo(F);
    const Child = if (info == .optional) info.optional.child else F;
    if (V == Child or Child == Tensor) return v;
    return coerce(Child, v);
}

fn fieldList(comptime W: type) []const u8 {
    var out: []const u8 = "";
    for (@typeInfo(W).@"struct".fields, 0..) |f, i| {
        out = out ++ (if (i == 0) "" else ", ") ++ f.name;
    }
    return out;
}

// -------------------------------------------------------------- Package -----

/// Parameters read out of a loaded `.aion` package by name.
///
/// `at(name)` hands a layer both its scope segment and its lookup prefix, so the
/// two cannot drift.
pub const Package = struct {
    weights: *Weights,

    const Self = @This();

    pub fn init(weights: *Weights) Self {
        return .{ .weights = weights };
    }

    /// A `Params` for the layer stored under `name`.
    pub fn at(self: *const Self, name: []const u8) Params {
        return .{
            .name = name,
            .ctx = @constCast(@ptrCast(self)),
            .resolveFn = resolve,
        };
    }

    /// A `Params` that lets the layer name itself; the lookup path follows whatever
    /// scope it opens, so a round-tripped auto-numbered graph reloads.
    pub fn auto(self: *const Self) Params {
        return .{ .name = null, .ctx = @constCast(@ptrCast(self)), .resolveFn = resolve };
    }

    fn resolve(ctx: *anyopaque, path: []const u8) BindError!?Tensor {
        const self: *const Self = @ptrCast(@alignCast(ctx));
        return self.weights.initializerTensorByDebugName(path) catch null;
    }

    fn isInitializer(self: *const Self, value_index: u32) bool {
        if (value_index >= self.weights.package.values.len) return false;
        return self.weights.package.values[value_index].source == .initializer;
    }

    /// The first weight in the package that the rebuilt graph has no slot for, or
    /// null when the package and the architecture agree.
    ///
    /// A package weight is accounted for when `bld` binds a parameter of the same
    /// name. That single rule covers every case: a weight a layer read is bound
    /// under that name; a constant the Builder synthesized is regenerated by the
    /// rebuild; a tied weight is bound once and used twice.
    pub fn firstUnaccounted(self: *const Self, bld: *const Builder) ?[]const u8 {
        for (self.weights.debugNames()) |entry| {
            if (!self.isInitializer(entry.value)) continue;
            if (bld.hasParamNamed(entry.name)) continue;
            return entry.name;
        }
        return null;
    }

    /// Assert the package holds no weight the rebuilt graph lacks a slot for.
    ///
    /// Worth calling after rebuilding: a package weight nobody claimed almost always
    /// means a layer was renamed or skipped, which otherwise produces a model that
    /// builds, runs, and is quietly wrong.
    pub fn expectAllLoaded(self: *const Self, bld: *const Builder) BindError!void {
        if (self.firstUnaccounted(bld) != null) return error.InvalidArgument;
    }
};

// --------------------------------------------------------- introspection ----

/// Emit `(debug_name, ref)` for every parameter reachable from `module`.
///
/// Walks the struct at comptime: any `TensorRef` field is a parameter, any field
/// whose type exposes `forward` is a submodule to recurse into, and optional/array
/// fields of either are followed. `visit` is called as `visit(context, name, ref)`
/// and may return an error.
///
/// Constants the Builder synthesized are skipped — they are not model state, and a
/// shared one would otherwise be reported once per layer that points at it.
///
/// Names come from the builder, so they are the same strings that land in the
/// package as `debug_names` — which makes this the way to see a model's
/// `state_dict` before exporting it.
pub fn forEachParam(
    comptime T: type,
    module: *const T,
    bld: *const Builder,
    context: anytype,
    comptime visit: anytype,
) !void {
    const info = @typeInfo(T);
    if (info != .@"struct") return;

    inline for (info.@"struct".fields) |field| {
        try visitValue(field.type, &@field(module, field.name), bld, context, visit);
    }
}

fn visitValue(
    comptime F: type,
    value: *const F,
    bld: *const Builder,
    context: anytype,
    comptime visit: anytype,
) !void {
    if (F == TensorRef) {
        // A layer field may point at something that is not a bound parameter at all
        // (a shared activation, or a default-initialized ref), and a synthesized
        // constant is a parameter but not model state.
        if (bld.paramKind(value.*) != .user) return;
        if (bld.valueName(value.*)) |n| try visit(context, n, value.*);
        return;
    }

    switch (@typeInfo(F)) {
        .optional => |o| {
            if (value.*) |*inner| try visitValue(o.child, inner, bld, context, visit);
        },
        .array => |a| {
            for (value) |*item| try visitValue(a.child, item, bld, context, visit);
        },
        .@"struct" => {
            // Only descend into things that look like layers; skipping plain config
            // structs keeps the walk from wandering into option bags.
            if (comptime module_api.isForwardModuleType(F)) {
                try forEachParam(F, value, bld, context, visit);
            }
        },
        else => {},
    }
}

/// Collect parameter names into `out`, returning how many were written.
///
/// Returns `error.InvalidArgument` if `out` is too small, so a caller sizing a
/// buffer notices rather than silently truncating.
pub fn collectParamNames(
    comptime T: type,
    module: *const T,
    bld: *const Builder,
    out: [][]const u8,
) !usize {
    const Ctx = struct { out: [][]const u8, n: usize = 0 };
    var ctx: Ctx = .{ .out = out };

    const Visit = struct {
        fn call(c: *Ctx, name: []const u8, ref: TensorRef) !void {
            _ = ref;
            if (c.n >= c.out.len) return error.InvalidArgument;
            c.out[c.n] = name;
            c.n += 1;
        }
    };

    try forEachParam(T, module, bld, &ctx, Visit.call);
    return ctx.n;
}
