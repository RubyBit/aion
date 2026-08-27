// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const infer_mod = @import("../graph/infer.zig");
const package_file = @import("../storage/aion_file.zig");

const types_mod = @import("package_export/types.zig");
const collect = @import("package_export/collect.zig");
const template_mod = @import("package_export/template.zig");
const graph_template = @import("../graph/template.zig");
const initializers = @import("package_export/initializers.zig");

pub const Builder = types_mod.Builder;
pub const NamedTensorRef = types_mod.NamedTensorRef;
pub const TensorRef = types_mod.TensorRef;
pub const StorageManager = types_mod.StorageManager;
pub const TensorId = types_mod.TensorId;
pub const Package = types_mod.Package;

pub const Metadata = types_mod.Metadata;
pub const OutputAlias = types_mod.OutputAlias;
pub const InputRoleDecl = types_mod.InputRoleDecl;
pub const ExportModelOptions = types_mod.ExportModelOptions;
pub const Parts = template_mod.Parts;
pub const collectTemplate = template_mod.collect;
pub const CompileOptions = types_mod.CompileOptions;

pub fn buildPackage(
    allocator: std.mem.Allocator,
    store: *StorageManager,
    builder: *Builder,
    outputs: []const NamedTensorRef,
    opts: ExportModelOptions,
) !Package {
    const graph = builder.innerGraph();
    try infer_mod.infer(graph);

    // The graph in symbolic form. Everything below is what a FILE needs on top: the
    // parameter bytes, and the metadata that describes the file rather than the graph.
    var parts = try template_mod.collect(allocator, builder, outputs, .{ .require_input_names = true });
    errdefer parts.deinit(allocator);

    var initializers_list: std.ArrayList(package_file.Initializer) = .empty;
    errdefer initializers_list.deinit(allocator);
    var slot_of = std.AutoHashMap(TensorId, u32).init(allocator);
    defer slot_of.deinit();

    // Serialize each parameter once and point its record at the slot. A weight shared by
    // two values is one slot, which is what `initializer_index` is for.
    for (parts.params, 0..) |tid, idx| {
        if (tid == graph_template.no_param) continue;
        const slot: u32 = if (slot_of.get(tid)) |existing| existing else blk: {
            const init = try initializers.exportInitializer(allocator, store, tid);
            const new_slot: u32 = @intCast(initializers_list.items.len);
            try initializers_list.append(allocator, init);
            try slot_of.put(tid, new_slot);
            break :blk new_slot;
        };
        parts.values[idx].initializer_index = slot;
    }

    const metadata = try collect.collectMetadata(allocator, opts.metadata);
    errdefer collect.freeMetadata(allocator, metadata);
    const io_aliases = try collect.collectIoAliases(allocator, parts.inputs, parts.outputs, opts.output_aliases);
    errdefer allocator.free(io_aliases);
    var symbol_map = try symbolMap(allocator, parts.dim_symbols);
    defer symbol_map.deinit(allocator);
    const input_roles = try collect.collectInputRoles(allocator, parts.inputs, opts.input_roles, &symbol_map);
    errdefer if (input_roles.len != 0) allocator.free(input_roles);

    const inits = try initializers_list.toOwnedSlice(allocator);
    errdefer allocator.free(inits);

    // Three of `parts` have no place in a file: `params` is where the bytes were read
    // FROM (they are in `initializers` now), and the value-id lists are a template's
    // index into its own records. Everything else the Package adopts.
    allocator.free(parts.params);
    allocator.free(parts.input_values);
    allocator.free(parts.output_values);

    return .{
        .allocator = allocator,
        .initializers = inits,
        .values = parts.values,
        .nodes = parts.nodes,
        .regions = parts.regions,
        .inputs = parts.inputs,
        .outputs = parts.outputs,
        .dim_symbols = parts.dim_symbols,
        .dim_exprs = parts.dim_exprs,
        .metadata = metadata,
        .debug_names = parts.debug_names,
        .io_aliases = io_aliases,
        .input_roles = input_roles,
    };
}

/// `collectInputRoles` resolves a capacity symbol by NAME, so rebuild the name->index map
/// the template already interned.
fn symbolMap(
    allocator: std.mem.Allocator,
    dim_symbols: []const package_file.DimSymbol,
) !std.StringHashMapUnmanaged(u32) {
    var out: std.StringHashMapUnmanaged(u32) = .{};
    errdefer out.deinit(allocator);
    for (dim_symbols, 0..) |sym, idx| try out.put(allocator, sym.name, @intCast(idx));
    return out;
}
