// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const manager_mod = @import("../../storage/manager.zig");
const storage_mod = @import("../../storage/storage.zig");
const package_file = @import("../../storage/aion_file.zig");
const types_mod = @import("types.zig");
const params_mod = @import("params.zig");
const api_errors = @import("../errors.zig");

/// Import every initializer of `package` into `store`, on `device`, returning the
/// parameters keyed by graph value. The `initializer_index` that names a slot in the
/// file's weight section does not survive this call.
///
/// A host weight whose payload is a view of `mapping` IS that view: no allocation and
/// no copy (the payloads are aligned for it, see `payload_alignment`). A weight bound
/// for a GPU is created there and uploaded from the payload directly, never staged.
pub fn importParams(
    allocator: std.mem.Allocator,
    store: *types_mod.StorageManager,
    package: *const types_mod.Package,
    device: manager_mod.DeviceRef,
    mapping: ?*storage_mod.Mapping,
) api_errors.LoadError!params_mod.Params {
    var out = try params_mod.Params.init(allocator, package.values.len);
    errdefer out.deinit(allocator);
    for (package.values, 0..) |value, value_idx| {
        if (value.source != .initializer) continue;
        const init_idx: u32 = value.initializer_index orelse return error.InvalidArgument;
        if (init_idx >= package.initializers.len) return error.InvalidArgument;
        const init = package.initializers[init_idx];
        const tid = try createInitializerTensor(allocator, store, package, value, init, device, mapping);
        out.set(@intCast(value_idx), tid);
    }
    return out;
}

/// The store tensor for one initializer value, sized from its record.
fn createInitializerTensor(
    allocator: std.mem.Allocator,
    store: *types_mod.StorageManager,
    package: *const types_mod.Package,
    value: package_file.ValueRecord,
    init: package_file.Initializer,
    device: manager_mod.DeviceRef,
    mapping: ?*storage_mod.Mapping,
) api_errors.LoadError!types_mod.TensorId {
    const shape = try resolveConstShape(allocator, package, value);
    defer allocator.free(shape);
    const quant_axis: u8 = switch (init.encoding) {
        .plain => 0,
        .quantized => |q| try quantAxisToU8(q.quant_axis, shape.len),
    };
    const bytes = init.data.bytes;
    if (device.kind == .cpu) {
        if (mapping) |m| if (std.mem.isAligned(@intFromPtr(bytes.ptr), 64)) {
            return store.createMappedTensor(value.dtype, shape, bytes, m, .{ .quant_axis = quant_axis });
        };
        // Not a view of a live mapping: its own copy, written whole.
        const tid = try store.createTensor(value.dtype, shape, .{ .quant_axis = quant_axis, .zero_fill = false });
        try store.writePackedAtPlacement(tid, bytes);
        return tid;
    }
    const tid = try store.createDeviceTensor(value.dtype, shape, .{ .quant_axis = quant_axis }, device);
    try store.writePackedAtPlacement(tid, bytes);
    return tid;
}

fn resolveConstShape(
    allocator: std.mem.Allocator,
    pkg: *const types_mod.Package,
    value: package_file.ValueRecord,
) api_errors.LoadError![]usize {
    const zero_symbols = try allocator.alloc(?u64, pkg.dim_symbols.len);
    defer allocator.free(zero_symbols);
    @memset(zero_symbols, null);
    return package_file.resolveShapeTerms(allocator, pkg, value.shape_terms, zero_symbols);
}

fn quantAxisToU8(raw: i32, rank: usize) error{InvalidArgument}!u8 {
    if (raw < 0) return error.InvalidArgument;
    const as_usize: usize = @intCast(raw);
    if (as_usize >= rank) return error.InvalidArgument;
    if (as_usize > std.math.maxInt(u8)) return error.InvalidArgument;
    return @intCast(as_usize);
}

