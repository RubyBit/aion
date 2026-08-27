// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! A model's graph before its free dimensions are known.
//!
//! Values carry their shapes as `ShapeTerm`s and a few node attributes are symbolic too,
//! so one template serves every input size a model is run at. `specialize` resolves the
//! symbols to numbers and builds a concrete `Graph`; inference then runs on something
//! fully numeric, which is why nothing in `infer` needs shape algebra.
//!
//! This is the one representation of "where a model's graph comes from". A parsed `.aion`
//! already is a template, and an authored Builder graph is snapshotted into one, so the
//! runtime specializes the same way for both instead of forking per source. That fork is
//! what this type exists to delete: the two paths differ only in who OWNS the records.
//!
//! The records are BORROWED. Whatever produced them keeps them alive for the template's
//! lifetime — a retained `Package` for a loaded model, an arena for a compiled one.
//! Deliberately not copied: a deep copy of a few thousand node records would buy uniform
//! ownership and nothing else, and the reading side cannot tell the difference.

const std = @import("std");

const graph_mod = @import("graph.zig");
const file = @import("../storage/aion_file.zig");
const manager_mod = @import("../storage/manager.zig");

const Graph = graph_mod.Graph;
const ValueId = graph_mod.ValueId;
const RegionId = graph_mod.RegionId;
const TensorId = manager_mod.TensorId;

pub const Error = graph_mod.GraphError || file.PackageError;

/// No parameter bound to this value. Matches `api/loaded_model/params.zig`.
pub const no_param: TensorId = std.math.maxInt(TensorId);

pub const Template = struct {
    values: []const file.ValueRecord,
    nodes: []const file.NodeRecord,
    regions: []const file.RegionRecord = &.{},
    /// Public input value ids in signature order.
    inputs: []const u32,
    /// Output value ids in signature order.
    outputs: []const u32,
    /// The table `ShapeTerm.expr` refers to.
    dim_exprs: []const file.DimExpr = &.{},
    dim_symbol_count: usize = 0,

    /// The shape terms of public input `index` — its rank, and which axes are free.
    pub fn inputShapeTerms(self: Template, index: usize) []const file.ShapeTerm {
        return self.values[self.inputs[index]].shape_terms;
    }

    /// Build a concrete `Graph` for these symbol values and public-input shapes.
    ///
    /// `params` gives the tensor bound to each parameter value, indexed by value id
    /// (`no_param` elsewhere) — the one thing a template cannot carry itself, because a
    /// weight can be retargeted after load. `input_shapes` is the public inputs'
    /// concrete shapes concatenated in signature order; each input's rank comes from its
    /// own shape terms. `in_ids` and `out_ids` are filled with the graph value id of each
    /// public input and output.
    ///
    /// Public inputs are left UNBOUND: which slot they read is the runtime's decision
    /// (a shared recurrent-state slot, a device slot, or the caller's own tensor).
    pub fn specialize(
        self: Template,
        gpa: std.mem.Allocator,
        params: []const TensorId,
        symbols: []const u64,
        input_shapes: []const usize,
        in_ids: []ValueId,
        out_ids: []ValueId,
    ) Error!Graph {
        std.debug.assert(in_ids.len == self.inputs.len);
        std.debug.assert(out_ids.len == self.outputs.len);

        var g = Graph.init(gpa);
        errdefer g.deinit();
        const arena = g.arenaAlloc();

        // Every value up front, in template order, so a node's ids ARE the template's:
        // nothing downstream has to be remapped, and a region body can name a value a
        // later top-level node produces without the build order caring.
        var cursor: usize = 0;
        const optional = arena.alloc(?u64, symbols.len) catch return Error.OutOfMemory;
        for (symbols, 0..) |v, i| optional[i] = if (v == 0) null else v;

        var next_input: usize = 0;
        for (self.values, 0..) |rec, idx| {
            const id = try g.addValue();
            std.debug.assert(id == @as(ValueId, @intCast(idx)));
            switch (rec.source) {
                .public_input => {
                    const rank = rec.shape_terms.len;
                    if (cursor + rank > input_shapes.len) return Error.InvalidArgument;
                    g.values.items[idx] = .{ .dtype = rec.dtype, .shape = try g.dupeShape(input_shapes[cursor..][0..rank]) };
                    cursor += rank;
                    if (next_input >= in_ids.len or self.inputs[next_input] != idx) return Error.InvalidArgument;
                    in_ids[next_input] = id;
                    next_input += 1;
                },
                .initializer => {
                    const shape = try file.resolveShapeTermsExprs(arena, self.dim_exprs, rec.shape_terms, optional);
                    g.values.items[idx] = .{ .dtype = rec.dtype, .shape = shape };
                    if (idx >= params.len or params[idx] == no_param) return Error.InvalidArgument;
                    try g.bindExternal(id, @intCast(params[idx]));
                },
                // Produced: dtype and shape come from inference, which runs at compile.
                .produced => {},
            }
        }
        if (next_input != in_ids.len) return Error.InvalidArgument;

        // Regions are built in dependency order, so their graph ids differ from the
        // template's; `region_map` is the only remapping `specialize` does.
        const built = arena.alloc(bool, self.regions.len) catch return Error.OutOfMemory;
        @memset(built, false);
        const region_map = arena.alloc(RegionId, self.regions.len) catch return Error.OutOfMemory;

        var ctx: Instantiate = .{
            .t = self,
            .g = &g,
            .arena = arena,
            .optional = optional,
            .built = built,
            .region_map = region_map,
        };
        for (self.nodes) |node| {
            try ctx.ensureRegions(node);
            try ctx.emit(node);
        }

        for (self.outputs, 0..) |vid, i| {
            if (vid >= self.values.len) return Error.InvalidArgument;
            out_ids[i] = @intCast(vid);
        }
        return g;
    }
};

/// The walk that materializes a template's nodes.
///
/// No case per op: the records hold `graph.Op` already, so all this does is resolve the
/// one attribute that can be symbolic, point control flow at the regions it built, and
/// append. The 32-arm conversion this replaced existed only because a package used to
/// carry its own op type.
const Instantiate = struct {
    t: Template,
    g: *Graph,
    arena: std.mem.Allocator,
    optional: []const ?u64,
    built: []bool,
    region_map: []RegionId,

    /// Build every region `rec` names, and every region THOSE name, first. A region body
    /// is open between `beginRegion` and `endRegion`, and only one can be open at a time,
    /// so a nested body has to be complete before its parent's is started.
    fn ensureRegions(self: *Instantiate, rec: file.NodeRecord) Error!void {
        switch (rec.op) {
            .If => |iff| {
                try self.region(iff.then_region);
                try self.region(iff.else_region);
            },
            .Loop => |lp| try self.region(lp.body_region),
            else => {},
        }
    }

    fn emit(self: *Instantiate, rec: file.NodeRecord) Error!void {
        var op = rec.op;
        try self.resolveAttr(&op);
        switch (op) {
            .If => |*iff| {
                iff.then_region = try self.mapped(iff.then_region);
                iff.else_region = try self.mapped(iff.else_region);
            },
            .Loop => |*lp| lp.body_region = try self.mapped(lp.body_region),
            else => {},
        }
        try self.g.appendNode(op, rec.inputs, rec.output, rec.extra_outputs);
    }

    /// Replace an op attribute's free axes with this specialization's sizes. The template
    /// is shared across specializations, so the resolved sizes go in a fresh slice.
    fn resolveAttr(self: *Instantiate, op: *file.NodeOp) Error!void {
        const attr = graph_mod.symbolicAttr(op) orelse return;
        if (attr.free_dims.len == 0) return;
        if (attr.free_dims.len != attr.sizes.len) return Error.InvalidArgument;

        const sizes = self.arena.alloc(usize, attr.sizes.len) catch return Error.OutOfMemory;
        for (sizes, attr.sizes, attr.free_dims) |*out, size, symbol| {
            out.* = if (symbol) |expr|
                try file.evaluateDimExpr(self.arena, self.t.dim_exprs, expr, self.optional)
            else
                size;
        }
        setAttrSizes(op, sizes);
    }

    fn mapped(self: *const Instantiate, template_region: u32) Error!u32 {
        if (template_region >= self.region_map.len) return Error.InvalidArgument;
        return self.region_map[template_region];
    }

    fn region(self: *Instantiate, id: u32) Error!void {
        const idx: usize = @intCast(id);
        if (idx >= self.t.regions.len) return Error.InvalidArgument;
        if (self.built[idx]) return;
        const rec = self.t.regions[idx];

        for (rec.nodes) |n| try self.ensureRegions(n);

        try self.g.beginRegion();
        for (rec.nodes) |n| try self.emit(n);
        self.region_map[idx] = try self.g.endRegion(rec.outputs);
        self.built[idx] = true;
    }
};

fn setAttrSizes(op: *file.NodeOp, sizes: []const usize) void {
    switch (op.*) {
        .LayerNorm => |*a| a.normalized_shape = sizes,
        .RMSNorm => |*a| a.normalized_shape = sizes,
        .ViewReshape => |*a| a.new_shape = sizes,
        .ViewSliceND => |*a| a.lens = sizes,
        else => {},
    }
}
