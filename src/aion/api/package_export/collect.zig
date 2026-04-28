// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const graph_mod = @import("../../graph/graph.zig");
const package_file = @import("../../storage/aion_file.zig");

const types_mod = @import("types.zig");
const convert = @import("convert.zig");

pub fn collectInputs(
    allocator: std.mem.Allocator,
    builder: *types_mod.Builder,
    graph: *graph_mod.Graph,
) ![]package_file.NamedValue {
    var out: std.ArrayList(package_file.NamedValue) = .empty;
    errdefer out.deinit(allocator);
    for (graph.values.items, 0..) |value, idx| {
        if (value.producer != null or value.external != null) continue;
        const name = builder.valueName(.{ .value = @intCast(idx) }) orelse return error.InvalidArgument;
        try out.append(allocator, .{ .name = try allocator.dupe(u8, name), .value = @intCast(idx) });
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
            .input = try findNamedValueIndex(inputs, alias.input_name),
            .output = try findNamedValueIndex(outputs, alias.output_name),
        };
    }
    return out;
}

pub fn collectNodes(allocator: std.mem.Allocator, graph: *graph_mod.Graph) ![]package_file.NodeRecord {
    const out = try allocator.alloc(package_file.NodeRecord, graph.nodes.items.len);
    errdefer allocator.free(out);
    for (graph.nodes.items, 0..) |node, idx| {
        out[idx] = .{
            .inputs = try allocator.dupe(u32, node.inputs),
            .output = node.output,
            .op = try convert.convertOp(allocator, node.op),
        };
    }
    return out;
}

pub fn findNamedValueIndex(named_values: []const package_file.NamedValue, name: []const u8) !u32 {
    for (named_values, 0..) |entry, idx| {
        if (std.mem.eql(u8, entry.name, name)) return @intCast(idx);
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

pub fn freeNodes(allocator: std.mem.Allocator, nodes: []package_file.NodeRecord) void {
    for (nodes) |node| {
        allocator.free(node.inputs);
        package_file.deinitNodeOp(allocator, node.op);
    }
    allocator.free(nodes);
}
