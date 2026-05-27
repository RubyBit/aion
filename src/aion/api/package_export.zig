// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const infer_mod = @import("../graph/infer.zig");
const package_file = @import("../storage/aion_file.zig");

const types_mod = @import("package_export/types.zig");
const collect = @import("package_export/collect.zig");
const initializers = @import("package_export/initializers.zig");

pub const Builder = types_mod.Builder;
pub const NamedTensorRef = types_mod.NamedTensorRef;
pub const TensorRef = types_mod.TensorRef;
pub const StorageManager = types_mod.StorageManager;
pub const TensorId = types_mod.TensorId;
pub const Package = types_mod.Package;

pub const Metadata = types_mod.Metadata;
pub const DimensionSymbol = types_mod.DimensionSymbol;
pub const OutputAlias = types_mod.OutputAlias;
pub const ExportModelOptions = types_mod.ExportModelOptions;

pub fn buildPackage(
    allocator: std.mem.Allocator,
    store: *StorageManager,
    builder: *Builder,
    outputs: []const NamedTensorRef,
    opts: ExportModelOptions,
) !Package {
    if (outputs.len == 0) return error.InvalidArgument;

    const graph = builder.innerGraph();
    try infer_mod.infer(graph);

    var symbol_map: std.StringHashMapUnmanaged(u32) = .{};
    defer symbol_map.deinit(allocator);

    var dim_symbols: std.ArrayList(package_file.DimSymbol) = .empty;
    defer dim_symbols.deinit(allocator);
    var dim_exprs: std.ArrayList(package_file.DimExpr) = .empty;
    defer dim_exprs.deinit(allocator);

    const input_symbol_expr = try allocator.alloc(?u32, graph.values.items.len * types_mod.max_rank);
    defer allocator.free(input_symbol_expr);
    @memset(input_symbol_expr, null);

    for (opts.input_symbols) |spec| {
        const value_idx: usize = @intCast(spec.tensor.value);
        if (value_idx >= graph.values.items.len) return error.InvalidArgument;
        const value = graph.values.items[value_idx];
        if (value.producer != null or value.external != null) return error.InvalidArgument;
        if (spec.axis >= value.shape.len) return error.InvalidArgument;

        const symbol_index: u32 = if (symbol_map.get(spec.name)) |idx|
            idx
        else blk: {
            const name_copy = try allocator.dupe(u8, spec.name);
            errdefer allocator.free(name_copy);
            const idx: u32 = @intCast(dim_symbols.items.len);
            try dim_symbols.append(allocator, .{ .name = name_copy });
            try dim_exprs.append(allocator, .{ .symbol = idx });
            try symbol_map.put(allocator, name_copy, idx);
            break :blk idx;
        };

        const expr_index: u32 = symbol_index;
        input_symbol_expr[value_idx * types_mod.max_rank + spec.axis] = expr_index;
    }

    var initializers_list: std.ArrayList(package_file.Initializer) = .empty;
    defer initializers_list.deinit(allocator);
    var initializer_map = std.AutoHashMap(TensorId, u32).init(allocator);
    defer initializer_map.deinit();

    const values = try allocator.alloc(package_file.ValueRecord, graph.values.items.len);
    errdefer collect.freeValues(allocator, values);

    for (graph.values.items, 0..) |value, idx| {
        const dtype = value.dtype orelse return error.InvalidArgument;
        const rank: u8 = @intCast(value.shape.len);
        if (value.producer == null) {
            if (value.external) |ext| {
                const tid: TensorId = @intCast(ext);
                const initializer_index: u32 = if (initializer_map.get(tid)) |existing|
                    existing
                else blk: {
                    const init = try initializers.exportInitializer(allocator, store, tid);
                    const new_index: u32 = @intCast(initializers_list.items.len);
                    try initializers_list.append(allocator, init);
                    try initializer_map.put(tid, new_index);
                    break :blk new_index;
                };
                values[idx] = .{
                    .dtype = dtype,
                    .rank = rank,
                    .source = .initializer,
                    .shape_terms = try package_file.makeConstantShapeTerms(allocator, value.shape),
                    .initializer_index = initializer_index,
                };
            } else {
                const shape_terms = try allocator.alloc(package_file.ShapeTerm, value.shape.len);
                for (value.shape, 0..) |dim, axis| {
                    if (input_symbol_expr[idx * types_mod.max_rank + axis]) |expr_idx| {
                        shape_terms[axis] = .{ .expr = expr_idx };
                    } else {
                        shape_terms[axis] = .{ .constant = @intCast(dim) };
                    }
                }
                values[idx] = .{
                    .dtype = dtype,
                    .rank = rank,
                    .source = .public_input,
                    .shape_terms = shape_terms,
                };
            }
        } else {
            values[idx] = .{
                .dtype = dtype,
                .rank = rank,
                .source = .produced,
                .shape_terms = try allocator.alloc(package_file.ShapeTerm, 0),
            };
        }
    }

    const inputs = try collect.collectInputs(allocator, builder, graph);
    errdefer collect.freeNamedValues(allocator, inputs);
    const output_values = try collect.collectOutputs(allocator, outputs);
    errdefer collect.freeNamedValues(allocator, output_values);
    const debug_names = try collect.collectDebugNames(allocator, builder);
    errdefer collect.freeDebugNames(allocator, debug_names);
    const metadata = try collect.collectMetadata(allocator, opts.metadata);
    errdefer collect.freeMetadata(allocator, metadata);
    const io_aliases = try collect.collectIoAliases(allocator, inputs, output_values, opts.output_aliases);
    errdefer allocator.free(io_aliases);
    const nodes = try collect.collectNodes(allocator, graph);
    errdefer collect.freeNodes(allocator, nodes);
    const regions = try collect.collectRegions(allocator, graph);
    errdefer collect.freeRegions(allocator, regions);

    return .{
        .allocator = allocator,
        .initializers = try initializers_list.toOwnedSlice(allocator),
        .values = values,
        .nodes = nodes,
        .regions = regions,
        .inputs = inputs,
        .outputs = output_values,
        .dim_symbols = try dim_symbols.toOwnedSlice(allocator),
        .dim_exprs = try dim_exprs.toOwnedSlice(allocator),
        .metadata = metadata,
        .debug_names = debug_names,
        .io_aliases = io_aliases,
    };
}
