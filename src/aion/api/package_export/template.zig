// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! A Builder graph in symbolic form: the shared half of exporting a package and compiling
//! one in process.
//!
//! `buildPackage` adds the parameter BYTES and the file-only metadata on top of this;
//! `Context.compile` keeps just this. That split is the point — copying every weight out
//! of the store to serialize it, and back in to run it, is exactly the round trip an
//! in-process compile exists to avoid.

const std = @import("std");

const graph_mod = @import("../../graph/graph.zig");
const template_mod = @import("../../graph/template.zig");
const package_file = @import("../../storage/aion_file.zig");

const types_mod = @import("types.zig");
const collect_mod = @import("collect.zig");

const Builder = types_mod.Builder;
const NamedTensorRef = types_mod.NamedTensorRef;
const TensorId = types_mod.TensorId;
const Template = template_mod.Template;
const max_rank = types_mod.max_rank;

/// A Builder graph's symbolic records, each slice individually owned so `buildPackage`
/// can hand them straight to a `Package`.
pub const Parts = struct {
    values: []package_file.ValueRecord,
    /// The store tensor for each parameter value; `template.no_param` elsewhere.
    ///
    /// The records deliberately do NOT carry this. A package names a slot in its own
    /// weight section instead, and that — where the bytes live — is the only thing the
    /// two forms of a template disagree about.
    params: []TensorId,
    nodes: []package_file.NodeRecord,
    regions: []package_file.RegionRecord,
    inputs: []package_file.NamedValue,
    outputs: []package_file.NamedValue,
    /// Value ids of `inputs` / `outputs`, which is what `Template` reads.
    input_values: []u32,
    output_values: []u32,
    debug_names: []package_file.DebugName,
    dim_symbols: []package_file.DimSymbol,
    dim_exprs: []package_file.DimExpr,
    /// Set by the caller after collecting: they come from compile options, not the graph.
    /// A package keeps the same two in its own sections.
    io_aliases: []package_file.IoAlias = &.{},
    input_roles: []package_file.InputRole = &.{},

    pub fn deinit(self: *Parts, allocator: std.mem.Allocator) void {
        collect_mod.freeValues(allocator, self.values);
        allocator.free(self.params);
        collect_mod.freeNodes(allocator, self.nodes);
        collect_mod.freeRegions(allocator, self.regions);
        collect_mod.freeNamedValues(allocator, self.inputs);
        collect_mod.freeNamedValues(allocator, self.outputs);
        allocator.free(self.input_values);
        allocator.free(self.output_values);
        collect_mod.freeDebugNames(allocator, self.debug_names);
        for (self.dim_symbols) |sym| allocator.free(sym.name);
        allocator.free(self.dim_symbols);
        allocator.free(self.dim_exprs);
        if (self.io_aliases.len != 0) allocator.free(self.io_aliases);
        if (self.input_roles.len != 0) allocator.free(self.input_roles);
        self.* = undefined;
    }

    pub fn view(self: Parts) Template {
        return .{
            .values = self.values,
            .nodes = self.nodes,
            .regions = self.regions,
            .inputs = self.input_values,
            .outputs = self.output_values,
            .dim_exprs = self.dim_exprs,
            .dim_symbol_count = self.dim_symbols.len,
        };
    }
};

/// Snapshot `builder`'s graph. Shapes become terms — a declared symbolic input axis
/// becomes an `.expr`, everything else a `.constant` — so the result serves every input
/// size the model is run at.
pub const Options = struct {
    /// Fail rather than inventing a name for an unnamed public input. A file's input
    /// names are its API, so `exportModel` requires them; an in-process compile is free
    /// to number them.
    require_input_names: bool = false,
};

pub fn collect(
    allocator: std.mem.Allocator,
    builder: *Builder,
    outputs: []const NamedTensorRef,
    opts: Options,
) !Parts {
    if (outputs.len == 0) return error.InvalidArgument;
    const graph = builder.innerGraph();

    var symbols = try Symbols.collect(allocator, builder, graph);
    errdefer symbols.deinit(allocator);

    const values = try allocator.alloc(package_file.ValueRecord, graph.values.items.len);
    errdefer collect_mod.freeValues(allocator, values);
    const params = try allocator.alloc(TensorId, graph.values.items.len);
    errdefer allocator.free(params);
    @memset(params, template_mod.no_param);

    for (graph.values.items, 0..) |value, idx| {
        const dtype = value.dtype orelse return error.InvalidArgument;
        const rank: u8 = @intCast(value.shape.len);
        if (value.producer != null) {
            values[idx] = .{
                .dtype = dtype,
                .rank = rank,
                .source = .produced,
                .shape_terms = try allocator.alloc(package_file.ShapeTerm, 0),
            };
            continue;
        }
        if (value.external) |ext| {
            params[idx] = @intCast(ext);
            values[idx] = .{
                .dtype = dtype,
                .rank = rank,
                .source = .initializer,
                .shape_terms = try package_file.makeConstantShapeTerms(allocator, value.shape),
            };
            continue;
        }
        values[idx] = .{
            .dtype = dtype,
            .rank = rank,
            .source = .public_input,
            .shape_terms = try symbols.inputTerms(allocator, idx, value.shape),
        };
    }

    const inputs = try collect_mod.collectInputs(allocator, builder, graph, opts.require_input_names);
    errdefer collect_mod.freeNamedValues(allocator, inputs);
    const outs = try collect_mod.collectOutputs(allocator, outputs);
    errdefer collect_mod.freeNamedValues(allocator, outs);
    const input_values = try valueIds(allocator, inputs);
    errdefer allocator.free(input_values);
    const output_values = try valueIds(allocator, outs);
    errdefer allocator.free(output_values);
    const debug_names = try collect_mod.collectDebugNames(allocator, builder);
    errdefer collect_mod.freeDebugNames(allocator, debug_names);
    const nodes = try collect_mod.collectNodes(allocator, graph);
    errdefer collect_mod.freeNodes(allocator, nodes);
    const regions = try collect_mod.collectRegions(allocator, graph);
    errdefer collect_mod.freeRegions(allocator, regions);

    const owned = symbols.toOwned(allocator);
    return .{
        .values = values,
        .params = params,
        .nodes = nodes,
        .regions = regions,
        .inputs = inputs,
        .outputs = outs,
        .input_values = input_values,
        .output_values = output_values,
        .debug_names = debug_names,
        .dim_symbols = owned.dim_symbols,
        .dim_exprs = owned.dim_exprs,
    };
}

fn valueIds(allocator: std.mem.Allocator, named: []const package_file.NamedValue) ![]u32 {
    const out = try allocator.alloc(u32, named.len);
    for (named, 0..) |entry, i| out[i] = entry.value;
    return out;
}

/// The Builder's declared dim symbols, resolved into the tables a template refers to.
///
/// Two lookup tables come out of this: which input axes are free (`input_expr`) and which
/// view-op attribute axes are (`view_expr`). Both are (value, axis) -> expr index.
pub const Symbols = struct {
    dim_symbols: []package_file.DimSymbol,
    dim_exprs: []package_file.DimExpr,
    input_expr: []?u32,

    /// The Builder's declared symbols, copied, plus the (value, axis) -> expr table the
    /// input value records need.
    ///
    /// A view op's free axes are NOT here: the op carries its own symbol indices, which
    /// are indices into this same list because the Builder interns in declaration order.
    /// That is the whole reason a side table stopped being necessary.
    pub fn collect(allocator: std.mem.Allocator, builder: *Builder, graph: *graph_mod.Graph) !Symbols {
        const names = builder.symbolNames();
        const dim_symbols = try allocator.alloc(package_file.DimSymbol, names.len);
        errdefer allocator.free(dim_symbols);
        var named: usize = 0;
        errdefer for (dim_symbols[0..named]) |sym| allocator.free(sym.name);
        for (names, 0..) |sym_name, idx| {
            dim_symbols[idx] = .{ .name = try allocator.dupe(u8, sym_name) };
            named += 1;
        }

        const dim_exprs = try allocator.alloc(package_file.DimExpr, names.len);
        errdefer allocator.free(dim_exprs);
        for (dim_exprs, 0..) |*expr, idx| expr.* = .{ .symbol = @intCast(idx) };

        const input_expr = try allocator.alloc(?u32, graph.values.items.len * max_rank);
        errdefer allocator.free(input_expr);
        @memset(input_expr, null);

        for (builder.dimSymbols()) |spec| {
            const value_idx: usize = @intCast(spec.tensor.value);
            if (value_idx >= graph.values.items.len) return error.InvalidArgument;
            const value = graph.values.items[value_idx];
            if (value.producer != null or value.external != null) return error.InvalidArgument;
            if (spec.axis >= value.shape.len) return error.InvalidArgument;
            input_expr[value_idx * max_rank + spec.axis] =
                builder.symbolIndex(spec.name) orelse return error.InvalidArgument;
        }

        return .{ .dim_symbols = dim_symbols, .dim_exprs = dim_exprs, .input_expr = input_expr };
    }

    /// This value's shape as terms, with declared free axes as expressions.
    pub fn inputTerms(
        self: *const Symbols,
        allocator: std.mem.Allocator,
        value_idx: usize,
        shape: []const usize,
    ) ![]package_file.ShapeTerm {
        const terms = try allocator.alloc(package_file.ShapeTerm, shape.len);
        for (shape, 0..) |dim, axis| {
            terms[axis] = if (self.input_expr[value_idx * max_rank + axis]) |expr|
                .{ .expr = expr }
            else
                .{ .constant = @intCast(dim) };
        }
        return terms;
    }

    pub const Tables = struct {
        dim_symbols: []package_file.DimSymbol,
        dim_exprs: []package_file.DimExpr,
    };

    /// Hand over the symbol tables and drop the scratch lookup.
    pub fn toOwned(self: *Symbols, allocator: std.mem.Allocator) Tables {
        allocator.free(self.input_expr);
        const out: Tables = .{ .dim_symbols = self.dim_symbols, .dim_exprs = self.dim_exprs };
        self.* = undefined;
        return out;
    }

    pub fn deinit(self: *Symbols, allocator: std.mem.Allocator) void {
        for (self.dim_symbols) |sym| allocator.free(sym.name);
        allocator.free(self.dim_symbols);
        allocator.free(self.dim_exprs);
        allocator.free(self.input_expr);
        self.* = undefined;
    }
};
