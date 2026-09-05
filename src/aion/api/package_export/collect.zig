// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const graph_mod = @import("../../graph/graph.zig");
const package_file = @import("../../storage/aion_file.zig");

const types_mod = @import("types.zig");

/// The graph's public inputs — leaves with no producer and no parameter binding — in
/// value order, which is signature order.
pub fn collectInputs(
    allocator: std.mem.Allocator,
    builder: *types_mod.Builder,
    graph: *graph_mod.Graph,
    require_names: bool,
) ![]package_file.NamedValue {
    var out: std.ArrayList(package_file.NamedValue) = .empty;
    errdefer {
        for (out.items) |entry| allocator.free(entry.name);
        out.deinit(allocator);
    }
    for (graph.values.items, 0..) |value, idx| {
        if (value.producer != null or value.external != null) continue;
        const name: []u8 = if (builder.valueName(.{ .value = @intCast(idx) })) |n|
            try allocator.dupe(u8, n)
        else if (require_names)
            return error.InvalidArgument
        else
            try std.fmt.allocPrint(allocator, "input{d}", .{out.items.len});
        try out.append(allocator, .{ .name = name, .value = @intCast(idx) });
    }
    return out.toOwnedSlice(allocator);
}

pub fn collectOutputs(allocator: std.mem.Allocator, outputs: []const types_mod.NamedTensorRef) ![]package_file.NamedValue {
    var out: std.ArrayList(package_file.NamedValue) = .empty;
    errdefer out.deinit(allocator);
    for (outputs) |entry| {
        try out.append(allocator, .{ .name = try allocator.dupe(u8, entry.name), .value = @intCast(entry.tensor.value) });
    }
    return out.toOwnedSlice(allocator);
}

pub fn collectDebugNames(allocator: std.mem.Allocator, builder: *const types_mod.Builder) ![]package_file.DebugName {
    var out: std.ArrayList(package_file.DebugName) = .empty;
    errdefer out.deinit(allocator);
    for (builder.value_names.items, 0..) |name_opt, idx| {
        if (name_opt) |name| {
            try out.append(allocator, .{ .value = @intCast(idx), .name = try allocator.dupe(u8, name) });
        }
    }
    return out.toOwnedSlice(allocator);
}

pub fn collectMetadata(allocator: std.mem.Allocator, metadata: []const types_mod.Metadata) ![]package_file.MetadataEntry {
    const out = try allocator.alloc(package_file.MetadataEntry, metadata.len);
    errdefer allocator.free(out);
    for (metadata, 0..) |entry, idx| {
        out[idx] = .{ .key = try allocator.dupe(u8, entry.key), .value = try allocator.dupe(u8, entry.value) };
    }
    return out;
}

pub fn collectIoAliases(
    allocator: std.mem.Allocator,
    inputs: []const package_file.NamedValue,
    outputs: []const package_file.NamedValue,
    aliases: []const types_mod.OutputAlias,
) ![]package_file.IoAlias {
    const out = try allocator.alloc(package_file.IoAlias, aliases.len);
    errdefer allocator.free(out);
    for (aliases, 0..) |alias, idx| {
        out[idx] = .{
            .input = try findNamedValueByValueId(inputs, alias.input.value),
            .output = try findNamedValueByValueId(outputs, alias.output.value),
        };
    }
    return out;
}

pub fn collectInputRoles(
    allocator: std.mem.Allocator,
    inputs: []const package_file.NamedValue,
    decls: []const types_mod.InputRoleDecl,
    symbol_map: *const std.StringHashMapUnmanaged(u32),
) ![]package_file.InputRole {
    if (decls.len == 0) return &[_]package_file.InputRole{};
    const out = try allocator.alloc(package_file.InputRole, decls.len);
    errdefer allocator.free(out);
    for (decls, 0..) |decl, idx| {
        var flags: u8 = 0;
        if (decl.zero_init) flags |= package_file.InputRoleFlags.zero_init;
        if (decl.allow_growable) flags |= package_file.InputRoleFlags.allow_growable;
        out[idx] = .{
            .input = try findNamedValueByValueId(inputs, decl.input.value),
            .kind = decl.kind,
            .axis = decl.axis orelse package_file.input_role_no_axis,
            .flags = flags,
            .capacity_symbol = if (decl.capacity_symbol) |name|
                symbol_map.get(name) orelse return error.InvalidArgument
            else
                package_file.invalid_index,
            .retained_history_tokens = decl.retained_history_tokens,
        };
    }
    return out;
}

const max_rank: usize = 8;

fn setAttrSymbols(op: *package_file.NodeOp, symbols: []const ?u32) void {
    switch (op.*) {
        .LayerNorm => |*a| a.free_dims = symbols,
        .RMSNorm => |*a| a.free_dims = symbols,
        .ViewReshape => |*a| a.free_dims = symbols,
        .ViewSliceND => |*a| a.free_dims = symbols,
        else => {},
    }
}

/// Copy a graph's nodes into owned records. The op type is the graph's own, so this is a
/// deep copy of the slice attributes plus the view symbols the Builder recorded — not a
/// conversion.
fn collectNodeSlice(allocator: std.mem.Allocator, nodes: []const graph_mod.Node) ![]package_file.NodeRecord {
    const out = try allocator.alloc(package_file.NodeRecord, nodes.len);
    errdefer allocator.free(out);
    var initialized: usize = 0;
    errdefer for (out[0..initialized]) |rec| {
        allocator.free(rec.inputs);
        if (rec.extra_outputs.len != 0) allocator.free(@constCast(rec.extra_outputs));
        package_file.deinitNodeOp(allocator, rec.op);
    };
    for (nodes, 0..) |node, idx| {
        const op = try cloneOpOwned(allocator, node.op);
        out[idx] = .{
            .inputs = try allocator.dupe(graph_mod.ValueId, node.inputs),
            .output = node.output,
            .op = op,
            .extra_outputs = try allocator.dupe(graph_mod.ValueId, node.extra_outputs),
        };
        initialized += 1;
    }
    return out;
}

/// The op with its slice attributes owned by `allocator` (the graph owns them in an arena
/// that does not outlive the export).
fn cloneOpOwned(allocator: std.mem.Allocator, op: graph_mod.Op) !package_file.NodeOp {
    var out = op;
    if (graph_mod.symbolicAttr(&out)) |attr| {
        const sizes = try allocator.dupe(usize, attr.sizes);
        const symbols: []const ?u32 = if (attr.free_dims.len == 0) &.{} else try allocator.dupe(?u32, attr.free_dims);
        setAttrSizes(&out, sizes);
        setAttrSymbols(&out, symbols);
    }
    switch (out) {
        .ViewSliceND => |*sl| sl.starts = try allocator.dupe(usize, sl.starts),
        else => {},
    }
    return out;
}

fn setAttrSizes(op: *package_file.NodeOp, sizes: []const usize) void {
    switch (op.*) {
        .LayerNorm => |*a| a.normalized_shape = sizes,
        .RMSNorm => |*a| a.normalized_shape = sizes,
        .ViewReshape => |*a| a.new_shape = sizes,
        .ViewSliceND => |*a| a.lens = sizes,
        else => {},
    }
}

pub fn collectNodes(allocator: std.mem.Allocator, graph: *graph_mod.Graph) ![]package_file.NodeRecord {
    return collectNodeSlice(allocator, graph.nodes.items);
}

pub fn collectRegions(allocator: std.mem.Allocator, graph: *graph_mod.Graph) ![]package_file.RegionRecord {
    const out = try allocator.alloc(package_file.RegionRecord, graph.regions.items.len);
    errdefer allocator.free(out);
    var initialized: usize = 0;
    errdefer {
        for (out[0..initialized]) |region| {
            freeNodes(allocator, region.nodes);
            allocator.free(region.outputs);
        }
    }
    for (graph.regions.items, 0..) |region, idx| {
        out[idx] = .{
            .nodes = try collectNodeSlice(allocator, region.nodes),
            .outputs = try allocator.dupe(u32, region.outputs),
        };
        initialized += 1;
    }
    return out;
}

pub fn findNamedValueIndex(named_values: []const package_file.NamedValue, name: []const u8) !u32 {
    for (named_values, 0..) |entry, idx| {
        if (std.mem.eql(u8, entry.name, name)) return @intCast(idx);
    }
    return error.InvalidArgument;
}

fn findNamedValueByValueId(named_values: []const package_file.NamedValue, value: u32) !u32 {
    for (named_values, 0..) |entry, idx| {
        if (entry.value == value) return @intCast(idx);
    }
    return error.InvalidArgument;
}

pub fn freeValues(allocator: std.mem.Allocator, values: []package_file.ValueRecord) void {
    for (values) |value| allocator.free(value.shape_terms);
    allocator.free(values);
}

pub fn freeNamedValues(allocator: std.mem.Allocator, values: []package_file.NamedValue) void {
    for (values) |entry| allocator.free(entry.name);
    allocator.free(values);
}

pub fn freeDebugNames(allocator: std.mem.Allocator, values: []package_file.DebugName) void {
    for (values) |entry| allocator.free(entry.name);
    allocator.free(values);
}

pub fn freeMetadata(allocator: std.mem.Allocator, values: []package_file.MetadataEntry) void {
    for (values) |entry| {
        allocator.free(entry.key);
        allocator.free(entry.value);
    }
    allocator.free(values);
}

pub fn freeRegions(allocator: std.mem.Allocator, regions: []package_file.RegionRecord) void {
    for (regions) |region| {
        freeNodes(allocator, region.nodes);
        allocator.free(region.outputs);
    }
    allocator.free(regions);
}

pub fn freeNodes(allocator: std.mem.Allocator, nodes: []package_file.NodeRecord) void {
    for (nodes) |node| {
        allocator.free(node.inputs);
        // `collectNodeSlice` dupes these, so a multi-output op (TopK, Loop) owns
        // them exactly like its inputs.
        if (node.extra_outputs.len != 0) allocator.free(@constCast(node.extra_outputs));
        package_file.deinitNodeOp(allocator, node.op);
    }
    allocator.free(nodes);
}
