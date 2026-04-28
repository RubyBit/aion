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
    const data = try allocator.alloc(u8, byte_len);
    errdefer allocator.free(data);

    if (tensor.dtype.info().is_quantized) {
        try store.readToPackedQuant(tid, data);
        const scheme = try allocator.dupe(u8, if (tensor.dtype == .q4_0) "q4_0" else "q8_0");
        errdefer allocator.free(scheme);
        const params = try allocator.alloc(u8, 0);
        return .{
            .encoding = .{ .quantized = .{
                .scheme = scheme,
                .logical_dtype = .f32,
                .block_elems = @intCast(tensor.dtype.info().block_elems),
                .block_bytes = @intCast(tensor.dtype.info().block_bytes),
                .quant_axis = 0,
                .params = params,
            } },
            .data = data,
        };
    }

    try store.readToPackedScalar(tid, data);
    return .{ .encoding = .{ .plain = tensor.dtype }, .data = data };
}
