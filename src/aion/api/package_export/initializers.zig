// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const backend_utils = @import("../../backend/utils.zig");
const package_file = @import("../../storage/aion_file.zig");
const types_mod = @import("types.zig");

pub fn exportInitializer(
    allocator: std.mem.Allocator,
    store: *types_mod.StorageManager,
    tid: types_mod.TensorId,
) !package_file.Initializer {
    const tensor = try store.getConst(tid);
    const elem_count = backend_utils.elemCount(tensor.shape) catch return error.InvalidArgument;
    const byte_len = backend_utils.requiredBytesForElems(tensor.dtype, elem_count) catch return error.InvalidArgument;
    // The writer reads the bytes from the store as it reaches them.
    const data: package_file.TensorData = .{ .source = .{ .len = byte_len, .ctx = store, .id = tid, .read = readTensor } };

    if (tensor.dtype.info().is_quantized) {
        const scheme = try allocator.dupe(u8, if (tensor.dtype == .q4_0) "q4_0" else "q8_0");
        errdefer allocator.free(scheme);
        return .{
            .encoding = .{ .quantized = .{
                .scheme = scheme,
                .logical_dtype = .f32,
                .block_elems = @intCast(tensor.dtype.info().block_elems),
                .block_bytes = @intCast(tensor.dtype.info().block_bytes),
                .quant_axis = @intCast(tensor.quant_axis),
                .params = &.{},
            } },
            .data = data,
        };
    }
    return .{ .encoding = .{ .plain = tensor.dtype }, .data = data };
}

fn readTensor(ctx: *anyopaque, id: u32, offset: usize, out: []u8) package_file.PackageError!void {
    const store: *types_mod.StorageManager = @ptrCast(@alignCast(ctx));
    const tensor = store.getConst(id) catch return package_file.PackageError.InvalidArgument;
    const unit = tensor.dtype.info().block_bytes; // a quant block, or one element
    if (offset % unit != 0) return package_file.PackageError.InvalidArgument;
    const read = if (tensor.dtype.info().is_quantized) store.readQuantBlocks(id, offset / unit, out) else store.readScalarRange(id, offset / unit, out);
    read catch return package_file.PackageError.InvalidArgument;
}
