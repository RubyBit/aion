// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const graph_mod = @import("../graph/graph.zig");
const infer_mod = @import("../graph/infer.zig");
const types = @import("../backend/types.zig");

const api_tensor = @import("tensor.zig");
const api_context = @import("context.zig");

pub const Context = api_context.Context;
pub const ValueId = graph_mod.ValueId;
pub const RegionId = graph_mod.RegionId;

/// Lightweight handle to a value produced/consumed by the builder.
///
/// This is intentionally opaque to keep the underlying graph hidden.
pub const TensorRef = struct {
    value: ValueId,
};

pub const NamedTensorRef = struct {
    name: []const u8,
    tensor: TensorRef,
};

/// Declares that an input tensor's axis is a free (variable) dimension named
/// `name`. Reusing a name across axes/inputs constrains them to the same runtime
/// size. Declared on the builder via `symbolicDim`; read by `compile`/`export`.
pub const DimSymbol = struct {
    tensor: TensorRef,
    axis: usize,
    name: []const u8,
};

/// Graph-hidden model builder.
///
/// Under the hood, this builds an `aion.graph.Graph`.
pub const Builder = struct {
    allocator: std.mem.Allocator,
    /// The context whose storage backs every `param`, and which `compile`/`export`
    /// must be called on. Held rather than passed at the end because the Builder
    /// already depends on it transitively — `param` takes a `Tensor` this context
    /// allocated — and because several ops need a constant operand the Builder must
    /// be able to materialize itself (see `constant`/`zeros`/`ones`).
    ctx: *Context,
    graph: graph_mod.Graph,

    // Optional per-value names for diagnostics/debugging.
    value_names: std.ArrayList(?[]const u8) = .empty,

    // --- scope stack -------------------------------------------------------
    // Scopes nest, so a module tree produces `state_dict`-style paths
    // (`Block#0/attn#0/q_proj#0/weight`). `scope_path` holds the joined path of
    // every open scope with no trailing separator; a `Scope` records the length
    // to truncate back to, so push/pop are O(1) with no re-joining.
    scope_path: std.ArrayList(u8) = .empty,
    // One op counter per open scope, so `#N` numbering restarts inside a scope.
    scope_counters: std.ArrayList(usize) = .empty,
    // Op counter used while no scope is open.
    root_counter: usize = 0,
    // `beginAutoScope` disambiguation counts, keyed by "<parent path>\x00<base>"
    // so sibling `Linear`s number independently under each parent.
    auto_scope_counts: std.StringHashMapUnmanaged(usize) = .{},

    // Symbolic (variable) input dimensions declared via `symbolicDim`.
    dim_symbols: std.ArrayList(DimSymbol) = .empty,

    // Every bound parameter, and where it came from. This is what distinguishes a
    // model weight from a constant the Builder had to invent, structurally rather
    // than by inspecting names.
    params: std.AutoHashMapUnmanaged(ValueId, ParamKind) = .{},

    // Cached synthesized constants, so a model that needs the same identity vector
    // or scalar in 35 places binds it once. Keyed by width / by f32 bit pattern.
    zero_vecs: std.AutoHashMapUnmanaged(usize, ValueId) = .{},
    one_vecs: std.AutoHashMapUnmanaged(usize, ValueId) = .{},
    scalars: std.AutoHashMapUnmanaged(u32, ValueId) = .{},

    // Symbolic dims used in view-op attributes (reshape new_shape / slice lens).
    // `tensor` is the view op's *output* value; `axis` indexes the attr dim; the
    // concrete attr value at that axis is the authoring placeholder. Emitted as a
    // dim-symbol expr term at export (the symbol must also be an input dim symbol).
    view_dim_symbols: std.ArrayList(DimSymbol) = .empty,

    const Self = @This();

    /// Includes `InferError` because ops infer their output shape eagerly as they
    /// are added, so a shape/dtype error surfaces at the offending op call.
    pub const Error = graph_mod.GraphError || infer_mod.InferError;

    /// Where a bound parameter came from.
    pub const ParamKind = enum {
        /// A weight the caller supplied: part of the model's state.
        user,
        /// A constant the Builder synthesized because an op requires an operand
        /// where the maths wants a number: an identity `gamma`/`beta`, or the
        /// one-element vector behind a scalar multiply. Rebuilding the same graph
        /// regenerates it, and one is typically shared by many layers, so it is not
        /// part of the model's state.
        synthesized,
    };

    /// Handle for one open scope. Pass it back to `endScope` to close it; the
    /// recorded `path_len`/`depth` make closing idempotent-ish and self-correcting
    /// if scopes are closed out of order.
    pub const Scope = struct {
        name: []const u8,
        path_len: usize = 0,
        depth: usize = 0,
    };

    /// Borrows `ctx`, which must outlive the Builder and must be the same context
    /// passed to `compile`/`export`.
    pub fn init(ctx: *Context) Self {
        const allocator: std.mem.Allocator = ctx.allocator;
        return .{
            .allocator = allocator,
            .ctx = ctx,
            .graph = graph_mod.Graph.init(allocator),
            .value_names = .empty,
            .scope_path = .empty,
            .scope_counters = .empty,
            .root_counter = 0,
            .auto_scope_counts = .{},
            .dim_symbols = .empty,
            .view_dim_symbols = .empty,
        };
    }

    pub fn deinit(self: *Self) void {
        self.params.deinit(self.allocator);
        self.zero_vecs.deinit(self.allocator);
        self.one_vecs.deinit(self.allocator);
        self.scalars.deinit(self.allocator);
        self.auto_scope_counts.deinit(self.allocator);
        self.scope_path.deinit(self.allocator);
        self.scope_counters.deinit(self.allocator);
        self.value_names.deinit(self.allocator);
        self.dim_symbols.deinit(self.allocator);
        self.view_dim_symbols.deinit(self.allocator);
        self.graph.deinit();
        self.* = undefined;
    }

    /// Declare input `t`'s axis `axis` as a free (variable) dimension named
    /// `sym_name`. One `compile`/`export` then serves any size on that axis;
    /// reuse a name to tie axes to the same runtime size.
    pub fn symbolicDim(self: *Self, t: TensorRef, axis: usize, sym_name: []const u8) Error!void {
        const name_copy: []u8 = self.graph.arenaAlloc().alloc(u8, sym_name.len) catch return Error.OutOfMemory;
        @memcpy(name_copy, sym_name);
        self.dim_symbols.append(self.allocator, .{ .tensor = t, .axis = axis, .name = name_copy }) catch return Error.OutOfMemory;
        // Also record it on the value itself, which is what carries the symbol down
        // the graph: inference propagates a value's symbols to everything derived
        // from it, so a layer can read a free axis off its input (`dimAt`) instead
        // of having to be told about it in its own configuration.
        try self.setValueDimSymbol(t, axis, name_copy);
    }

    /// Attach `name` to `t`'s axis. The name must already live in the graph arena.
    fn setValueDimSymbol(self: *Self, t: TensorRef, axis: usize, sym_name: []const u8) Error!void {
        const idx: usize = @intCast(t.value);
        if (idx >= self.graph.values.items.len) return Error.InvalidArgument;
        var v = self.graph.values.items[idx];
        if (axis >= v.shape.len) return Error.InvalidArgument;

        if (v.dim_symbols.len != v.shape.len) {
            const syms = self.graph.arenaAlloc().alloc(?[]const u8, v.shape.len) catch return Error.OutOfMemory;
            @memset(syms, null);
            for (v.dim_symbols, 0..) |existing, i| syms[i] = existing;
            v.dim_symbols = syms;
        }
        // Safe: allocated above, or grown from a shorter slice this Builder owns.
        const writable: []?[]const u8 = @constCast(v.dim_symbols);
        writable[axis] = sym_name;
        self.graph.values.items[idx] = v;
    }

    /// One axis of a shape: a fixed size, or a free axis named by a dim symbol.
    ///
    /// Read a value's axes with `dimAt` and pass them to `reshapeDims`, so a layer
    /// that has to rebuild a shape (splitting a projection into heads, say) keeps
    /// whatever freedom its input had without needing to know which axes those are.
    pub const Dim = union(enum) {
        size: usize,
        symbol: []const u8,
    };

    /// `t`'s axis as a `Dim`: symbolic if that axis was declared free (directly or
    /// by inheriting from a value it was derived from), otherwise its fixed size.
    pub fn dimAt(self: *const Self, t: TensorRef, axis: usize) Error!Dim {
        const idx: usize = @intCast(t.value);
        if (idx >= self.graph.values.items.len) return Error.InvalidArgument;
        const v = self.graph.values.items[idx];
        if (axis >= v.shape.len) return Error.InvalidArgument;
        if (axis < v.dim_symbols.len) {
            if (v.dim_symbols[axis]) |sym| return .{ .symbol = sym };
        }
        return .{ .size = v.shape[axis] };
    }

    /// The authoring placeholder size for a declared dim symbol, or null if no
    /// input declared it. Symbolic axes still need a concrete size to author with;
    /// this is where a `Dim.symbol` gets one, so a caller never has to supply both.
    pub fn symbolSize(self: *const Self, sym_name: []const u8) ?usize {
        for (self.dim_symbols.items) |spec| {
            if (!std.mem.eql(u8, spec.name, sym_name)) continue;
            const idx: usize = @intCast(spec.tensor.value);
            if (idx >= self.graph.values.items.len) continue;
            const v = self.graph.values.items[idx];
            if (spec.axis < v.shape.len) return v.shape[spec.axis];
        }
        return null;
    }

    /// The symbolic input dimensions declared so far (read by `compile`/`export`).
    pub fn dimSymbols(self: *const Self) []const DimSymbol {
        return self.dim_symbols.items;
    }

    /// Symbolic dims used in view-op attributes (read by `export`).
    pub fn viewDimSymbols(self: *const Self) []const DimSymbol {
        return self.view_dim_symbols.items;
    }

    fn recordViewSymbols(self: *Self, out: TensorRef, symbols: []const ?[]const u8) Error!void {
        for (symbols, 0..) |maybe_name, axis| {
            const sym_name = maybe_name orelse continue;
            const name_copy: []u8 = self.graph.arenaAlloc().alloc(u8, sym_name.len) catch return Error.OutOfMemory;
            @memcpy(name_copy, sym_name);
            self.view_dim_symbols.append(self.allocator, .{ .tensor = out, .axis = axis, .name = name_copy }) catch return Error.OutOfMemory;
            // A view's sizes come from its attribute, so inference cannot re-derive
            // which of them are free and leaves the value's symbols alone (`.keep`).
            // Recording them here is what keeps the chain unbroken across a reshape.
            try self.setValueDimSymbol(out, axis, name_copy);
        }
    }

    pub fn hasActiveScope(self: *const Self) bool {
        return self.scope_counters.items.len != 0;
    }

    /// The joined path of all open scopes (`""` when none are open).
    pub fn scopePath(self: *const Self) []const u8 {
        return self.scope_path.items;
    }

    /// How many scopes are currently open.
    pub fn scopeDepth(self: *const Self) usize {
        return self.scope_counters.items.len;
    }

    /// Open a scope named `scope_name`, nested inside any already-open scope.
    pub fn beginScope(self: *Self, scope_name: []const u8) Error!Scope {
        const name_copy: []u8 = self.graph.arenaAlloc().alloc(u8, scope_name.len) catch return Error.OutOfMemory;
        @memcpy(name_copy, scope_name);
        return self.pushScope(name_copy);
    }

    /// Open a scope named `{base_name}#{n}`, where `n` counts prior sibling
    /// scopes with the same base under the *same parent* — so two `Linear`s in
    /// one block are `Linear#0`/`Linear#1`, and a `Linear` in the next block
    /// restarts at `Linear#0`.
    pub fn beginAutoScope(self: *Self, base_name: []const u8) Error!Scope {
        const parent: []const u8 = self.scope_path.items;

        // Key on parent path + base so counters are per-parent. NUL separates the
        // two halves so ("a", "b/c") and ("a/b", "c") cannot collide.
        const key = std.fmt.allocPrint(self.graph.arenaAlloc(), "{s}\x00{s}", .{ parent, base_name }) catch return Error.OutOfMemory;
        const gop = try self.auto_scope_counts.getOrPut(self.allocator, key);
        if (!gop.found_existing) gop.value_ptr.* = 0;
        const idx: usize = gop.value_ptr.*;
        gop.value_ptr.* = idx + 1;

        const unique_name = std.fmt.allocPrint(self.graph.arenaAlloc(), "{s}#{d}", .{ base_name, idx }) catch return Error.OutOfMemory;
        return self.pushScope(unique_name);
    }

    /// Append `segment` (already arena-owned) to the scope path and start a fresh
    /// op counter for it.
    fn pushScope(self: *Self, segment: []const u8) Error!Scope {
        const prev_len: usize = self.scope_path.items.len;
        const depth: usize = self.scope_counters.items.len;

        if (prev_len != 0) self.scope_path.append(self.allocator, '/') catch return Error.OutOfMemory;
        self.scope_path.appendSlice(self.allocator, segment) catch {
            self.scope_path.shrinkRetainingCapacity(prev_len);
            return Error.OutOfMemory;
        };
        self.scope_counters.append(self.allocator, 0) catch {
            self.scope_path.shrinkRetainingCapacity(prev_len);
            return Error.OutOfMemory;
        };

        return .{ .name = segment, .path_len = prev_len, .depth = depth };
    }

    /// Close `scope`, restoring the path and counter depth it was opened at.
    ///
    /// Truncating to the recorded depth (rather than popping exactly one level)
    /// means an inner scope left open by a `return` cannot leak into sibling
    /// names — the enclosing `defer endScope(...)` cleans up both.
    pub fn endScope(self: *Self, scope: Scope) void {
        if (scope.depth >= self.scope_counters.items.len) return;
        self.scope_counters.shrinkRetainingCapacity(scope.depth);
        if (scope.path_len <= self.scope_path.items.len) {
            self.scope_path.shrinkRetainingCapacity(scope.path_len);
        }
    }

    /// Close every scope opened at or below `depth`.
    ///
    /// The same truncate-to-depth semantics as `endScope`, addressed by depth
    /// alone so a caller that cannot hold a `Scope` — notably the C ABI — still
    /// gets the safety of closing back to a known level.
    pub fn endScopeAtDepth(self: *Self, depth: usize) void {
        if (depth >= self.scope_counters.items.len) return;
        // Path length is recoverable from the surviving segments: truncate the
        // joined path to the end of the last one that stays open.
        var keep: usize = 0;
        var i: usize = 0;
        var cursor: usize = 0;
        while (i < depth) : (i += 1) {
            const next = std.mem.indexOfScalarPos(u8, self.scope_path.items, cursor, '/') orelse self.scope_path.items.len;
            keep = next;
            cursor = next + 1;
        }
        self.scope_counters.shrinkRetainingCapacity(depth);
        self.scope_path.shrinkRetainingCapacity(if (depth == 0) 0 else keep);
    }

    pub fn innerGraph(self: *Self) *graph_mod.Graph {
        return &self.graph;
    }

    /// Transfer ownership of the underlying graph to the caller, replacing it with a
    /// fresh empty graph so the Builder remains safe to `deinit`. Used by
    /// `ctx.compile` to hand the graph to the compiled `Model` without a round trip.
    /// The Builder should not be used to add more ops after this.
    /// A standalone copy of the authored graph, for a consumer that takes ownership.
    ///
    /// `compile` hands the model a copy rather than the builder's own graph so that
    /// authoring survives compilation: you can compile a second time, compile a
    /// different set of outputs, or evaluate one value to look at it, and keep
    /// building afterwards. Moving the graph out instead left the builder silently
    /// empty — every op after a `compile` landed in a fresh graph that shared
    /// nothing with what came before.
    ///
    /// Cheap next to compiling: this copies node/value metadata into a new arena and
    /// no tensor data (external bindings are ids, copied by value).
    pub fn cloneGraph(self: *const Self) Error!graph_mod.Graph {
        return self.graph.clone(self.allocator);
    }

    /// Return the known shape for a value, if available.
    ///
    /// With eager per-op inference, every value produced so far (inputs, params,
    /// and op outputs) has a shape. For a value on a declared-symbolic axis the
    /// reported size is the authoring *placeholder* (the size the axis was
    /// declared with), which propagates deterministically through derived values.
    pub fn knownShape(self: *const Self, t: TensorRef) ?[]const usize {
        const idx: usize = @intCast(t.value);
        if (idx >= self.graph.values.items.len) return null;
        const v = self.graph.values.items[idx];
        if (v.shape.len == 0) return null;
        return v.shape;
    }

    /// The dtype of a value, if known (null for an out-of-range id).
    pub fn dtypeOf(self: *const Self, t: TensorRef) ?types.DType {
        const idx: usize = @intCast(t.value);
        if (idx >= self.graph.values.items.len) return null;
        return self.graph.values.items[idx].dtype;
    }

    /// Like `knownShape`, but returns `error.InvalidArgument` if shape is unknown.
    pub fn requireKnownShape(self: *Self, t: TensorRef) Error![]const usize {
        return self.knownShape(t) orelse Error.InvalidArgument;
    }

    pub fn valueName(self: *const Self, t: TensorRef) ?[]const u8 {
        const idx: usize = @intCast(t.value);
        if (idx >= self.value_names.items.len) return null;
        return self.value_names.items[idx];
    }

    fn ensureNameSlot(self: *Self, vid: ValueId) Error!void {
        const idx: usize = @intCast(vid);
        if (idx < self.value_names.items.len) return;

        const old_len: usize = self.value_names.items.len;
        const new_len: usize = idx + 1;
        try self.value_names.ensureTotalCapacity(self.allocator, new_len);
        self.value_names.items.len = new_len;
        var i: usize = old_len;
        while (i < new_len) : (i += 1) {
            self.value_names.items[i] = null;
        }
    }

    pub fn name(self: *Self, t: TensorRef, value_name: []const u8) Error!TensorRef {
        try self.ensureNameSlot(t.value);
        const duped: []u8 = self.graph.arenaAlloc().alloc(u8, value_name.len) catch return Error.OutOfMemory;
        @memcpy(duped, value_name);
        self.value_names.items[@intCast(t.value)] = duped;
        return t;
    }

    /// Bind an existing owned tensor as a graph input.
    ///
    /// The generated debug name is positional (`param@{valueId}`), so it shifts if
    /// construction order changes. Prefer `paramNamed` for anything you intend to
    /// look up or swap later.
    pub fn param(self: *Self, t: api_tensor.Tensor) Error!TensorRef {
        return self.paramInner(t, null, .user);
    }

    /// Bind an owned tensor as a parameter under a *semantic* name, scoped by the
    /// enclosing module scopes: `paramNamed(w, "weight")` inside
    /// `Block#0`/`attn#0` becomes `Block#0/attn#0/weight`.
    ///
    /// These names persist into the package as `debug_names`, which is the key the
    /// weight-swap API (`LoadedModel.overwriteInitializerByDebugName`) looks up —
    /// so a stable, hierarchical name is what makes swapping usable.
    pub fn paramNamed(self: *Self, t: api_tensor.Tensor, param_name: []const u8) Error!TensorRef {
        if (param_name.len == 0) return Error.InvalidArgument;
        return self.paramInner(t, param_name, .user);
    }

    fn paramInner(self: *Self, t: api_tensor.Tensor, param_name: ?[]const u8, kind: ParamKind) Error!TensorRef {
        const v: ValueId = try self.graph.addInput(t.dtype, t.shape);
        try self.graph.bindExternal(v, @intCast(t.id));
        self.params.put(self.allocator, v, kind) catch return Error.OutOfMemory;

        // Give parameters (external-bound inputs) a stable debug name so loaded
        // models can find and swap them later.
        //
        // Important: do NOT use `autoNameIfUnnamed` here, since that would consume
        // an op index and perturb op numbering (e.g. `matmul#0`).
        try self.ensureNameSlot(v);
        const idx: usize = @intCast(v);
        if (self.value_names.items[idx] == null) {
            const arena = self.graph.arenaAlloc();
            const scope: []const u8 = self.scope_path.items;
            const generated_name = if (param_name) |pn|
                (if (scope.len != 0)
                    std.fmt.allocPrint(arena, "{s}/{s}", .{ scope, pn }) catch return Error.OutOfMemory
                else
                    std.fmt.allocPrint(arena, "{s}", .{pn}) catch return Error.OutOfMemory)
            else if (scope.len != 0)
                std.fmt.allocPrint(arena, "{s}/param@{d}", .{ scope, v }) catch return Error.OutOfMemory
            else
                std.fmt.allocPrint(arena, "param@{d}", .{v}) catch return Error.OutOfMemory;
            self.value_names.items[idx] = generated_name;
        }

        return .{ .value = v };
    }

    /// Where `t` came from, or null if it is not a bound parameter.
    pub fn paramKind(self: *const Self, t: TensorRef) ?ParamKind {
        return self.params.get(t.value);
    }

    /// Whether this graph binds a parameter under `name`, of either kind.
    ///
    /// Lets a caller check a package's weights against the architecture it just
    /// rebuilt without knowing which of them the Builder invented: a synthesized
    /// constant is regenerated by the rebuild, so it answers true here just like a
    /// real weight does.
    pub fn hasParamNamed(self: *const Self, param_name: []const u8) bool {
        var it = self.params.keyIterator();
        while (it.next()) |v| {
            const idx: usize = @intCast(v.*);
            if (idx >= self.value_names.items.len) continue;
            const existing = self.value_names.items[idx] orelse continue;
            if (std.mem.eql(u8, existing, param_name)) return true;
        }
        return false;
    }

    /// A cached one-element f32 constant, for ops that need a scalar as an operand.
    ///
    /// Aion has no scalar-typed operand: `x * k` goes through the broadcast
    /// elementwise op with a size-1 vector. Binding it here means the constant is
    /// one element regardless of what it multiplies, and identical values across the
    /// whole model share a single parameter.
    pub fn constant(self: *Self, value: f32) Error!TensorRef {
        const key: u32 = @bitCast(value);
        if (self.scalars.get(key)) |v| return .{ .value = v };

        const t: api_tensor.Tensor = self.ctx.fromF32(&[_]usize{1}, &[_]f32{value}) catch return Error.OutOfMemory;
        // Named by bit pattern, not by `{d}`: distinct floats must not share a name,
        // and 0.0 / -0.0 stay distinguishable.
        const ref: TensorRef = try self.bindConstant(t, "const.scalar.{x}", key);
        self.scalars.put(self.allocator, key, ref.value) catch return Error.OutOfMemory;
        return ref;
    }

    /// A cached `[dim]` f32 vector of zeros — the identity `beta` of a norm.
    pub fn zeros(self: *Self, dim: usize) Error!TensorRef {
        return self.filledVec(&self.zero_vecs, dim, 0.0, "const.zeros.{d}");
    }

    /// A cached `[dim]` f32 vector of ones — the identity `gamma` of a norm.
    pub fn ones(self: *Self, dim: usize) Error!TensorRef {
        return self.filledVec(&self.one_vecs, dim, 1.0, "const.ones.{d}");
    }

    fn filledVec(
        self: *Self,
        cache: *std.AutoHashMapUnmanaged(usize, ValueId),
        dim: usize,
        value: f32,
        comptime name_fmt: []const u8,
    ) Error!TensorRef {
        if (dim == 0) return Error.InvalidArgument;
        if (cache.get(dim)) |v| return .{ .value = v };

        const vals: []f32 = self.allocator.alloc(f32, dim) catch return Error.OutOfMemory;
        defer self.allocator.free(vals);
        @memset(vals, value);

        const t: api_tensor.Tensor = self.ctx.fromF32(&[_]usize{dim}, vals) catch return Error.OutOfMemory;
        const ref: TensorRef = try self.bindConstant(t, name_fmt, dim);
        cache.put(self.allocator, dim, ref.value) catch return Error.OutOfMemory;
        return ref;
    }

    /// Bind a synthesized constant under a stable, scope-*independent* name.
    ///
    /// Deliberately not scope-qualified: a shared constant is reachable from many
    /// scopes, so naming it after whichever one happened to request it first would
    /// be misleading.
    fn bindConstant(self: *Self, t: api_tensor.Tensor, comptime name_fmt: []const u8, key: anytype) Error!TensorRef {
        const arena = self.graph.arenaAlloc();
        const const_name = std.fmt.allocPrint(arena, name_fmt, .{key}) catch return Error.OutOfMemory;
        const ref: TensorRef = try self.paramInner(t, null, .synthesized);
        return self.name(ref, const_name);
    }

    pub fn input(self: *Self, dtype: types.DType, shape: []const usize) Error!TensorRef {
        const v: ValueId = try self.graph.addInput(dtype, shape);
        return .{ .value = v };
    }

    /// `out = alpha * (a @ b) + beta * out`, batched over all leading dims.
    ///
    /// `b` broadcasts into `a`'s batch dims, including by having fewer of them: a
    /// plain `[in, out]` weight multiplies a `[batch, seq, in]` activation directly.
    /// Nothing is reshaped to make that work — the weight reaches the kernel exactly
    /// as it was bound, which is what keeps a quantized weight usable (its packing
    /// cannot survive a reshape) and keeps it eligible for horizontal matmul fusion.
    ///
    /// `a` may not be the shorter operand: its batch dims are the output's, so a
    /// rank-2 activation against a `[1, in, out]` weight is left-padded instead.
    pub fn matmul(self: *Self, a: TensorRef, b: TensorRef, alpha: f32, beta: f32) Error!TensorRef {
        const out: ValueId = try self.graph.addMatMul((try self.padAToB(a, b)).value, b.value, alpha, beta);
        try self.autoNameIfUnnamed(out, "matmul");
        return .{ .value = out };
    }

    /// Left-pad `a`'s shape with 1s when `b` has more batch dims than it does.
    ///
    /// Only `a` is ever padded. `a` is an activation — a node result whose reshape is
    /// an ordinary copy — whereas `b` is typically a bound weight, for which the same
    /// operation is both lossy and unnecessary.
    fn padAToB(self: *Self, a: TensorRef, b: TensorRef) Error!TensorRef {
        const a_shape = self.knownShape(a) orelse return a;
        const b_shape = self.knownShape(b) orelse return a;

        // Only a rank-2-or-higher operand can be a matmul operand at all; padding a
        // rank-1 vector would silently pick an interpretation, so leave it to fail.
        if (a_shape.len < 2 or b_shape.len < 2) return a;
        if (a_shape.len >= b_shape.len) return a;

        var cur: TensorRef = a;
        var rank: usize = a_shape.len;
        while (rank < b_shape.len) : (rank += 1) {
            cur = try self.unsqueeze(cur, 0);
        }
        return cur;
    }

    pub fn add(self: *Self, a: TensorRef, b: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addElemwiseBinary(.add, a.value, b.value);
        try self.autoNameIfUnnamed(out, "add");
        return .{ .value = out };
    }

    pub fn mul(self: *Self, a: TensorRef, b: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addElemwiseBinary(.mul, a.value, b.value);
        try self.autoNameIfUnnamed(out, "mul");
        return .{ .value = out };
    }

    pub fn relu(self: *Self, a: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addRelu(a.value);
        try self.autoNameIfUnnamed(out, "relu");
        return .{ .value = out };
    }

    pub fn sigmoid(self: *Self, a: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addUnary(.sigmoid, a.value);
        try self.autoNameIfUnnamed(out, "sigmoid");
        return .{ .value = out };
    }

    pub fn tanh(self: *Self, a: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addUnary(.tanh, a.value);
        try self.autoNameIfUnnamed(out, "tanh");
        return .{ .value = out };
    }

    pub fn sqrt(self: *Self, a: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addUnary(.sqrt, a.value);
        try self.autoNameIfUnnamed(out, "sqrt");
        return .{ .value = out };
    }

    pub fn unary(self: *Self, op: types.UnaryOp, a: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addUnary(op, a.value);
        try self.autoNameIfUnnamed(out, @tagName(op));
        return .{ .value = out };
    }

    pub fn softmax(self: *Self, a: TensorRef, axis: i32) Error!TensorRef {
        const out: ValueId = try self.graph.addSoftmax(a.value, axis);
        try self.autoNameIfUnnamed(out, "softmax");
        return .{ .value = out };
    }

    pub fn layernorm(self: *Self, x: TensorRef, gamma: TensorRef, beta: TensorRef, eps: f32, normalized_shape: []const usize) Error!TensorRef {
        const out: ValueId = try self.graph.addLayerNorm(x.value, gamma.value, beta.value, eps, normalized_shape);
        try self.autoNameIfUnnamed(out, "layernorm");
        return .{ .value = out };
    }

    pub fn rmsnorm(self: *Self, x: TensorRef, gamma: TensorRef, beta: TensorRef, eps: f32, normalized_shape: []const usize) Error!TensorRef {
        const out: ValueId = try self.graph.addRMSNorm(x.value, gamma.value, beta.value, eps, normalized_shape);
        try self.autoNameIfUnnamed(out, "rmsnorm");
        return .{ .value = out };
    }

    /// Grouped-query attention over time-major k/v `[B, T, H_kv, D]`.
    /// Optional query positions and K/V lengths are independent.
    pub fn attention(
        self: *Self,
        q: TensorRef,
        k: TensorRef,
        v: TensorRef,
        query_positions: ?TensorRef,
        kv_lengths: ?TensorRef,
        scale: f32,
        causal: bool,
        sliding_window: usize,
        attn_logits_soft_cap: f32,
    ) Error!TensorRef {
        const out: ValueId = try self.graph.addAttention(
            q.value,
            k.value,
            v.value,
            if (query_positions) |x| x.value else null,
            if (kv_lengths) |x| x.value else null,
            scale,
            causal,
            sliding_window,
            attn_logits_soft_cap,
        );
        try self.autoNameIfUnnamed(out, "attention");
        return .{ .value = out };
    }

    pub fn conv1d(
        self: *Self,
        x: TensorRef,
        w: TensorRef,
        bias: ?TensorRef,
        stride: usize,
        dilation: usize,
        pad_left: usize,
        pad_right: usize,
        groups: usize,
    ) Error!TensorRef {
        return self.conv1dPadMode(x, w, bias, stride, dilation, pad_left, pad_right, .zero, groups);
    }

    pub fn conv1dPadMode(
        self: *Self,
        x: TensorRef,
        w: TensorRef,
        bias: ?TensorRef,
        stride: usize,
        dilation: usize,
        pad_left: usize,
        pad_right: usize,
        pad_mode: types.PadMode,
        groups: usize,
    ) Error!TensorRef {
        const out: ValueId = try self.graph.addConv1DWithPadMode(
            x.value,
            w.value,
            if (bias) |b0| b0.value else null,
            stride,
            dilation,
            pad_left,
            pad_right,
            pad_mode,
            groups,
        );
        try self.autoNameIfUnnamed(out, "conv1d");
        return .{ .value = out };
    }

    pub fn conv2d(
        self: *Self,
        x: TensorRef,
        w: TensorRef,
        bias: ?TensorRef,
        stride_h: usize,
        stride_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        pad_top: usize,
        pad_bottom: usize,
        pad_left: usize,
        pad_right: usize,
        groups: usize,
    ) Error!TensorRef {
        return self.conv2dPadMode(x, w, bias, stride_h, stride_w, dilation_h, dilation_w, pad_top, pad_bottom, pad_left, pad_right, .zero, groups);
    }

    pub fn conv2dPadMode(
        self: *Self,
        x: TensorRef,
        w: TensorRef,
        bias: ?TensorRef,
        stride_h: usize,
        stride_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        pad_top: usize,
        pad_bottom: usize,
        pad_left: usize,
        pad_right: usize,
        pad_mode: types.PadMode,
        groups: usize,
    ) Error!TensorRef {
        const out: ValueId = try self.graph.addConv2DWithPadMode(
            x.value,
            w.value,
            if (bias) |b0| b0.value else null,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            pad_mode,
            groups,
        );
        try self.autoNameIfUnnamed(out, "conv2d");
        return .{ .value = out };
    }

    pub fn copy(self: *Self, a: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addCopy(a.value);
        try self.autoNameIfUnnamed(out, "copy");
        return .{ .value = out };
    }

    /// Gather rows from a 2D table using i32 indices.
    ///
    /// Shapes:
    /// - table:   [V, D]
    /// - indices: [B, L]
    /// - out:     [B, L, D]
    /// Gather slices along `axis`, sharing the first `batch_dims` axes between
    /// `data` and `indices`. This subsumes embedding lookup
    /// (`axis=0,batch_dims=0`) and per-batch row selection
    /// (`axis=1,batch_dims=1`).
    pub fn gather(self: *Self, data: TensorRef, indices: TensorRef, axis: i32, batch_dims: usize) Error!TensorRef {
        const out: ValueId = try self.graph.addGather(data.value, indices.value, axis, batch_dims);
        try self.autoNameIfUnnamed(out, "gather");
        return .{ .value = out };
    }

    pub fn dimSize(self: *Self, value: TensorRef, axis: i32) Error!TensorRef {
        const out: ValueId = try self.graph.addDim(value.value, axis);
        try self.autoNameIfUnnamed(out, "dim");
        return .{ .value = out };
    }

    /// Coordinate tensor with `shape_like`'s shape, increasing from zero along
    /// `axis` and repeating over every other axis.
    pub fn iota(self: *Self, shape_like: TensorRef, axis: i32) Error!TensorRef {
        const out: ValueId = try self.graph.addIota(shape_like.value, axis);
        try self.autoNameIfUnnamed(out, "iota");
        return .{ .value = out };
    }

    /// Rotary positional embedding over 1D positions.
    ///
    /// Inputs:
    /// - x:         [B, L, N, H]
    /// - positions: [B, L] (i32)
    ///
    /// Output:
    /// - out:       [B, L, N, H]
    pub fn rope1d(
        self: *Self,
        x: TensorRef,
        positions: TensorRef,
        base_frequency: f32,
        scale_factor: f32,
        rope_proportion: f32,
    ) Error!TensorRef {
        const out: ValueId = try self.graph.addRoPE1D(
            x.value,
            positions.value,
            base_frequency,
            scale_factor,
            rope_proportion,
        );
        try self.autoNameIfUnnamed(out, "rope1d");
        return .{ .value = out };
    }

    /// In-place KV cache append.
    ///
    /// Inputs:
    /// - cache:     [B, T, H_kv, D]
    /// - new_kv:    [B, new_len, H_kv, D]
    /// - end_index: [B] (i32)
    ///
    /// Output:
    /// - out: same shape/dtype as cache (mutated in-place semantics)
    pub fn sequenceAppend(
        self: *Self,
        cache: TensorRef,
        new_kv: TensorRef,
        end_index: TensorRef,
    ) Error!TensorRef {
        const out: ValueId = try self.graph.addSequenceAppend(cache.value, new_kv.value, end_index.value);
        try self.autoNameIfUnnamed(out, "sequence_append");
        return .{ .value = out };
    }

    pub fn reduce(self: *Self, op: types.ReduceOp, a: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addReduce(op, a.value);
        try self.autoNameIfUnnamed(out, "reduce");
        return .{ .value = out };
    }

    pub fn reduceAxis(self: *Self, op: types.ReduceOp, a: TensorRef, axis: i32) Error!TensorRef {
        const out: ValueId = try self.graph.addReduceAxis(op, a.value, axis);
        try self.autoNameIfUnnamed(out, "reduce_axis");
        return .{ .value = out };
    }

    pub fn concat(self: *Self, tensors: []const TensorRef, axis: i32) Error!TensorRef {
        if (tensors.len == 0) return Error.InvalidArgument;
        var ids: [16]ValueId = @splat(0);
        if (tensors.len > ids.len) return Error.InvalidArgument;
        var i: usize = 0;
        while (i < tensors.len) : (i += 1) {
            ids[i] = tensors[i].value;
        }
        const out: ValueId = try self.graph.addConcat(ids[0..tensors.len], axis);
        try self.autoNameIfUnnamed(out, "concat");
        return .{ .value = out };
    }

    /// Fused single-timestep LSTM cell.
    ///
    /// Output is a packed state tensor of shape `[batch, 2*hidden]` where:
    /// - state[:, 0:hidden]   == h_t
    /// - state[:, hidden:2*h] == c_t
    pub fn lstmCell(
        self: *Self,
        x: TensorRef,
        h_prev: TensorRef,
        c_prev: TensorRef,
        w_ih: TensorRef,
        w_hh: TensorRef,
        b_ih: ?TensorRef,
        b_hh: ?TensorRef,
    ) Error!TensorRef {
        const out: ValueId = try self.graph.addLSTMCell(
            x.value,
            h_prev.value,
            c_prev.value,
            w_ih.value,
            w_hh.value,
            if (b_ih) |t| t.value else null,
            if (b_hh) |t| t.value else null,
        );
        try self.autoNameIfUnnamed(out, "lstm_cell");
        return .{ .value = out };
    }

    /// Real FFT over the last (power-of-two) dimension.
    ///
    /// Input `x[.., n_fft]` (f32) → output `[.., n_fft+2]` packed complex:
    /// one-sided bins `n_fft/2+1` with real parts in `[0..bins)` and imaginary
    /// parts in `[bins..2*bins)`.
    pub fn rfft(self: *Self, x: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addRFFT(x.value);
        try self.autoNameIfUnnamed(out, "rfft");
        return .{ .value = out };
    }

    /// Short-time Fourier transform.
    ///
    /// Inputs: `signal[batch, samples]` (f32), `window[n_fft]` (f32, padded to
    /// `n_fft` by the caller). Output: `[batch, num_frames, n_fft+2]` packed
    /// complex (same layout as `rfft`). See `Op.STFT` for framing semantics.
    pub fn stft(
        self: *Self,
        signal: TensorRef,
        window: TensorRef,
        n_fft: usize,
        hop_length: usize,
        center: bool,
    ) Error!TensorRef {
        const out: ValueId = try self.graph.addSTFT(signal.value, window.value, n_fft, hop_length, center);
        try self.autoNameIfUnnamed(out, "stft");
        return .{ .value = out };
    }

    pub fn stack(self: *Self, tensors: []const TensorRef, axis: i32) Error!TensorRef {
        if (tensors.len == 0) return Error.InvalidArgument;
        var unsq: [16]TensorRef = undefined;
        if (tensors.len > unsq.len) return Error.InvalidArgument;

        var i: usize = 0;
        while (i < tensors.len) : (i += 1) {
            unsq[i] = try self.unsqueeze(tensors[i], axis);
        }
        return self.concat(unsq[0..tensors.len], axis);
    }

    pub fn reshape(self: *Self, a: TensorRef, new_shape: []const usize) Error!TensorRef {
        const out: ValueId = try self.graph.addViewReshape(a.value, new_shape);
        try self.autoNameIfUnnamed(out, "reshape");
        return .{ .value = out };
    }

    /// Reshape where some target dims are symbolic. `new_shape[i]` is the concrete
    /// authoring placeholder; `symbols[i]`, if non-null, names a dim symbol that
    /// axis takes at runtime (emitted as a symbolic term at export).
    pub fn reshapeSym(self: *Self, a: TensorRef, new_shape: []const usize, symbols: []const ?[]const u8) Error!TensorRef {
        const ref = try self.reshape(a, new_shape);
        try self.recordViewSymbols(ref, symbols);
        return ref;
    }

    /// Reshape to a mix of fixed and free axes.
    ///
    /// Unlike `reshapeSym` this needs no placeholder sizes: a `.symbol` dim resolves
    /// its authoring size from the input that declared it. That is what lets a layer
    /// write `&.{ try bld.dimAt(x, 0), try bld.dimAt(x, 1), .{ .size = heads }, ... }`
    /// and stay correct whether or not the caller made those axes free.
    pub fn reshapeDims(self: *Self, a: TensorRef, dims: []const Dim) Error!TensorRef {
        var shape: [graph_mod.Graph.MAX_RANK]usize = undefined;
        var symbols: [graph_mod.Graph.MAX_RANK]?[]const u8 = undefined;
        try self.resolveDims(dims, &shape, &symbols);
        return self.reshapeSym(a, shape[0..dims.len], symbols[0..dims.len]);
    }

    /// Slice to a mix of fixed and free lengths. See `reshapeDims`.
    pub fn sliceDims(self: *Self, a: TensorRef, starts: []const usize, lens: []const Dim) Error!TensorRef {
        var shape: [graph_mod.Graph.MAX_RANK]usize = undefined;
        var symbols: [graph_mod.Graph.MAX_RANK]?[]const u8 = undefined;
        try self.resolveDims(lens, &shape, &symbols);
        return self.sliceSym(a, starts, shape[0..lens.len], symbols[0..lens.len]);
    }

    /// Split `dims` into the concrete sizes to author with and the symbol names to
    /// record. A symbol no input declared is rejected here rather than at export,
    /// where the offending call is no longer in view.
    fn resolveDims(self: *const Self, dims: []const Dim, shape: []usize, symbols: []?[]const u8) Error!void {
        if (dims.len == 0 or dims.len > graph_mod.Graph.MAX_RANK) return Error.InvalidArgument;
        for (dims, 0..) |dim, i| {
            switch (dim) {
                .size => |n| {
                    shape[i] = n;
                    symbols[i] = null;
                },
                .symbol => |sym| {
                    shape[i] = self.symbolSize(sym) orelse return Error.InvalidArgument;
                    symbols[i] = sym;
                },
            }
        }
    }

    pub fn squeeze(self: *Self, a: TensorRef, axis: ?i32) Error!TensorRef {
        const out: ValueId = try self.graph.addViewSqueeze(a.value, axis);
        try self.autoNameIfUnnamed(out, "squeeze");
        return .{ .value = out };
    }

    pub fn unsqueeze(self: *Self, a: TensorRef, axis: i32) Error!TensorRef {
        const out: ValueId = try self.graph.addViewUnsqueeze(a.value, axis);
        try self.autoNameIfUnnamed(out, "unsqueeze");
        return .{ .value = out };
    }

    pub fn transpose2d(self: *Self, a: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addViewTranspose2D(a.value);
        try self.autoNameIfUnnamed(out, "transpose2d");
        return .{ .value = out };
    }

    pub fn slice(self: *Self, a: TensorRef, starts: []const usize, lens: []const usize) Error!TensorRef {
        const out: ValueId = try self.graph.addViewSliceND(a.value, starts, lens);
        try self.autoNameIfUnnamed(out, "slice");
        return .{ .value = out };
    }

    /// Slice where some lengths are symbolic. `lens[i]` is the concrete authoring
    /// placeholder; `symbols[i]`, if non-null, names a dim symbol that length
    /// takes at runtime (emitted as a symbolic term at export).
    pub fn sliceSym(self: *Self, a: TensorRef, starts: []const usize, lens: []const usize, symbols: []const ?[]const u8) Error!TensorRef {
        const ref = try self.slice(a, starts, lens);
        try self.recordViewSymbols(ref, symbols);
        return ref;
    }

    /// Take `len` elements starting at `start` along the last axis, keeping every
    /// leading axis whole.
    ///
    /// This is how a fused projection is split back into its parts (QKV, gate/up),
    /// which otherwise means building full `starts`/`lens` arrays at each call site.
    pub fn sliceLastDim(self: *Self, a: TensorRef, start: usize, len: usize) Error!TensorRef {
        const shape: []const usize = self.knownShape(a) orelse return Error.InvalidArgument;
        if (shape.len == 0 or shape.len > graph_mod.Graph.MAX_RANK) return Error.InvalidArgument;

        var starts: [graph_mod.Graph.MAX_RANK]usize = @splat(0);
        var lens: [graph_mod.Graph.MAX_RANK]usize = @splat(0);
        for (shape, 0..) |d, i| lens[i] = d;

        const last: usize = shape.len - 1;
        const end: usize = std.math.add(usize, start, len) catch return Error.InvalidArgument;
        if (end > shape[last]) return Error.InvalidArgument;
        starts[last] = start;
        lens[last] = len;

        return self.slice(a, starts[0..shape.len], lens[0..shape.len]);
    }

    pub fn slice2d(self: *Self, a: TensorRef, start0: usize, len0: usize, start1: usize, len1: usize) Error!TensorRef {
        return self.slice(a, &[_]usize{ start0, start1 }, &[_]usize{ len0, len1 });
    }

    /// Generic elementwise binary op with standard right-aligned broadcasting
    /// (`add`/`sub`/`mul`/`div` and `eq`/`ne`/`lt`/`gt`/`le`/`ge`).
    pub fn elemwiseBinary(self: *Self, op: types.ElemwiseBinaryOp, a: TensorRef, b: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addElemwiseBinary(op, a.value, b.value);
        try self.autoNameIfUnnamed(out, @tagName(op));
        return .{ .value = out };
    }

    /// Matmul with a transposed / per-row-quantized B: `C[m,n] = Σ A[m,k]·B[n,k]`
    /// (B is `[N, K]`). Used for tied-embedding logits.
    pub fn matmulNT(self: *Self, a: TensorRef, b: TensorRef, alpha: f32, beta: f32) Error!TensorRef {
        const out: ValueId = try self.graph.addMatMulNT(a.value, b.value, alpha, beta);
        try self.autoNameIfUnnamed(out, "matmul_nt");
        return .{ .value = out };
    }

    /// Scalar dtype cast (e.g. f32 <-> f16).
    pub fn cast(self: *Self, a: TensorRef, to_dtype: types.DType) Error!TensorRef {
        const out: ValueId = try self.graph.addCast(a.value, to_dtype);
        try self.autoNameIfUnnamed(out, "cast");
        return .{ .value = out };
    }

    /// Index of the max value along `axis` (i32 output).
    pub fn argmax(self: *Self, a: TensorRef, axis: i32) Error!TensorRef {
        const out: ValueId = try self.graph.addArgMax(a.value, axis);
        try self.autoNameIfUnnamed(out, "argmax");
        return .{ .value = out };
    }

    /// In-place row scatter: `buf[idx] = src`. Output aliases `buf`.
    pub fn scatterRow(self: *Self, buf: TensorRef, idx: TensorRef, src: TensorRef) Error!TensorRef {
        const out: ValueId = try self.graph.addScatterRow(buf.value, idx.value, src.value);
        try self.autoNameIfUnnamed(out, "scatter_row");
        return .{ .value = out };
    }

    /// Chunked-limited attention window: a query attends to its own chunk of
    /// `size` keys plus `left` keys before that chunk's start. `size = 0` (the
    /// default) means every key. Prefer this over an additive mask — it is two
    /// integers instead of a [T_q, T_kv] tensor, and it lets the kernels skip the
    /// keys outside the window instead of scoring and then discarding them.
    pub const ChunkWindow = struct { size: usize = 0, left: usize = 0 };

    /// Relative-position multi-head self-attention (Transformer-XL / Conformer).
    /// `mask` is optional and composes with `window` (use it only for what an
    /// interval cannot express, e.g. streaming padding). The head count comes from
    /// `q`'s shape — it is never passed separately.
    pub fn relPosMHA(
        self: *Self,
        q: TensorRef,
        k: TensorRef,
        v: TensorRef,
        pos_emb: TensorRef,
        pos_bias_u: TensorRef,
        pos_bias_v: TensorRef,
        mask: ?TensorRef,
        scale: f32,
        window: ChunkWindow,
    ) Error!TensorRef {
        const out: ValueId = try self.graph.addRelPosMHA(
            q.value,
            k.value,
            v.value,
            pos_emb.value,
            pos_bias_u.value,
            pos_bias_v.value,
            if (mask) |m| m.value else null,
            scale,
            window.size,
            window.left,
        );
        try self.autoNameIfUnnamed(out, "relpos_mha");
        return .{ .value = out };
    }

    // --- Control flow (regions) --------------------------------------------

    /// Begin a sub-region. Ops added until `endRegion` become the region body
    /// (used for `If` branches and `Loop` bodies). Regions do not nest.
    pub fn beginRegion(self: *Self) Error!void {
        return self.graph.beginRegion();
    }

    /// Close the active region, declaring its outputs. Returns a region id for
    /// use with `ifThenElse` / `loop`.
    pub fn endRegion(self: *Self, outputs: []const TensorRef) Error!RegionId {
        var ids: [16]ValueId = undefined;
        if (outputs.len == 0 or outputs.len > ids.len) return Error.InvalidArgument;
        for (outputs, 0..) |o, i| ids[i] = o.value;
        return self.graph.endRegion(ids[0..outputs.len]);
    }

    /// Single-output conditional: evaluate `then_region` or `else_region` based
    /// on the i32 `[1]` `cond`. Both regions must have exactly one output.
    pub fn ifThenElse(self: *Self, cond: TensorRef, then_region: RegionId, else_region: RegionId) Error!TensorRef {
        const out: ValueId = try self.graph.addIf(cond.value, then_region, else_region);
        try self.autoNameIfUnnamed(out, "if");
        return .{ .value = out };
    }

    /// Single-carry loop: runs `body_region` up to `static_max_trip_count` times,
    /// threading the carry. `body_region` must have exactly one output.
    pub fn loop(self: *Self, carried_init: TensorRef, body_region: RegionId, static_max_trip_count: usize) Error!TensorRef {
        const out: ValueId = try self.graph.addLoop(carried_init.value, body_region, static_max_trip_count);
        try self.autoNameIfUnnamed(out, "loop");
        return .{ .value = out };
    }

    /// Multi-carry loop. `carried_inits[i]` pairs with body-region output `i`;
    /// the final carried values are written to `out_refs` (same length). If
    /// `cond_carry` is set, that carry index is the i32 `[1]` continue predicate.
    pub fn loopMulti(
        self: *Self,
        carried_inits: []const TensorRef,
        body_region: RegionId,
        static_max_trip_count: usize,
        cond_carry: ?usize,
        check_before: bool,
        out_refs: []TensorRef,
    ) Error!void {
        var ids: [16]ValueId = undefined;
        const n = carried_inits.len;
        if (n == 0 or n > ids.len or out_refs.len != n) return Error.InvalidArgument;
        for (carried_inits, 0..) |c, i| ids[i] = c.value;
        const outs = try self.graph.addLoopMulti(ids[0..n], body_region, static_max_trip_count, cond_carry, check_before);
        for (outs, 0..) |o, i| {
            try self.autoNameIfUnnamed(o, "loop");
            out_refs[i] = .{ .value = o };
        }
    }

    /// Take the next op index for the innermost open scope (or the root).
    fn nextOpIndex(self: *Self) usize {
        if (self.scope_counters.items.len != 0) {
            const slot: *usize = &self.scope_counters.items[self.scope_counters.items.len - 1];
            const n: usize = slot.*;
            slot.* = n + 1;
            return n;
        }
        const n: usize = self.root_counter;
        self.root_counter = n + 1;
        return n;
    }

    fn autoNameIfUnnamed(self: *Self, vid: ValueId, tag: []const u8) Error!void {
        // Eager per-op inference: infer the node just added (it produced `vid`).
        // Every op funnels through here exactly once after appending its node, so
        // this keeps the whole graph inferred as it is built — `knownShape` is
        // always current and shape/dtype errors surface at the offending op.
        // Runs before the naming early-return so a pre-named output still infers.
        if (self.graph.lastNode()) |node| {
            try infer_mod.inferNode(&self.graph, node);
        }

        try self.ensureNameSlot(vid);

        const idx: usize = @intCast(vid);
        if (self.value_names.items[idx] != null) return;

        const n: usize = self.nextOpIndex();

        const generated_name = if (self.scope_path.items.len != 0)
            std.fmt.allocPrint(self.graph.arenaAlloc(), "{s}/{s}#{d}", .{ self.scope_path.items, tag, n }) catch return Error.OutOfMemory
        else
            std.fmt.allocPrint(self.graph.arenaAlloc(), "{s}#{d}", .{ tag, n }) catch return Error.OutOfMemory;

        self.value_names.items[idx] = generated_name;
    }
};
