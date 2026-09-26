// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// ScatterRow: in-place row write `buf[idx] = src`. The output aliases buf (set up
// in lowering). The "row" is buf[1:]
// flattened (scalar for rank-1 buf). Used to emit decode tokens into an output
// buffer at a dynamic index.
const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");

const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

fn scalarBytes(dt: types.DType) ExecuteProgramError!usize {
    return switch (dt) {
        .f32, .i32 => 4,
        .f16 => 2,
        .i8 => 1,
        else => BackendError.InvalidArgument,
    };
}

pub fn execScatterRow(
    s: executable.StepScatterRow,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    const buf_meta = try store.meta(s.buf);
    const idx_meta = try store.meta(s.idx);

    if (buf_meta.rank == 0) return BackendError.InvalidArgument;
    if (buf_meta.dtype.info().is_quantized) return BackendError.InvalidArgument;
    if (idx_meta.dtype != .i32) return BackendError.InvalidArgument;

    const m: usize = buf_meta.shape[0];
    var row_size: usize = 1;
    for (buf_meta.shape[1..buf_meta.rank]) |dim| row_size *= dim;
    const elem_bytes: usize = try scalarBytes(buf_meta.dtype);
    const row_bytes: usize = row_size * elem_bytes;

    const idx_view = try store.acquireConst(s.idx);
    defer store.releaseConst(idx_view.token);
    const idx_bytes = idx_view.bufferView().bytes;
    if (idx_bytes.len < 4) return BackendError.InvalidArgument;
    const idx_val: i32 = std.mem.readInt(i32, idx_bytes[0..4], .little);
    if (idx_val < 0 or @as(usize, @intCast(idx_val)) >= m) return BackendError.InvalidArgument;
    const row: usize = @intCast(idx_val);

    const src_view = try store.acquireConst(s.src);
    defer store.releaseConst(src_view.token);
    const src_bytes = src_view.bufferView().bytes;
    if (src_bytes.len < row_bytes) return BackendError.InvalidArgument;

    const buf_view = try store.acquireMut(s.buf);
    defer store.releaseMut(buf_view.token);
    const buf_bytes = buf_view.bufferView().bytes;
    if (buf_bytes.len < (row + 1) * row_bytes) return BackendError.InvalidArgument;

    @memcpy(buf_bytes[row * row_bytes .. row * row_bytes + row_bytes], src_bytes[0..row_bytes]);
}
