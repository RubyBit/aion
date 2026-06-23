// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// ScatterRow: in-place row write `buf[idx] = src`. The output aliases buf (set up
// in lowering). v1: buf/idx/src are each a single tile. The "row" is buf[1:]
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

    const buf_rank: usize = @as(usize, buf_meta.rank);
    var d: usize = 0;
    while (d < buf_rank) : (d += 1) {
        if (buf_meta.tile_counts[d] != 1) return BackendError.InvalidArgument;
    }

    const m: usize = buf_meta.shape[0];
    var row_size: usize = 1;
    d = 1;
    while (d < buf_rank) : (d += 1) row_size *= buf_meta.shape[d];
    const elem_bytes: usize = try scalarBytes(buf_meta.dtype);
    const row_bytes: usize = row_size * elem_bytes;

    const idx_tile = try store.acquireTileConstLinear(s.idx, 0);
    defer store.releaseConst(idx_tile.token);
    const idx_bytes = idx_tile.bufferView().bytes;
    if (idx_bytes.len < 4) return BackendError.InvalidArgument;
    const idx_val: i32 = std.mem.readInt(i32, idx_bytes[0..4], .little);
    if (idx_val < 0 or @as(usize, @intCast(idx_val)) >= m) return BackendError.InvalidArgument;
    const row: usize = @intCast(idx_val);

    const src_tile = try store.acquireTileConstLinear(s.src, 0);
    defer store.releaseConst(src_tile.token);
    const src_bytes = src_tile.bufferView().bytes;
    if (src_bytes.len < row_bytes) return BackendError.InvalidArgument;

    const buf_tile = try store.acquireTileMutLinear(s.buf, 0);
    defer store.releaseMut(buf_tile.token);
    const buf_bytes = buf_tile.bufferView().bytes;
    if (buf_bytes.len < (row + 1) * row_bytes) return BackendError.InvalidArgument;

    @memcpy(buf_bytes[row * row_bytes .. row * row_bytes + row_bytes], src_bytes[0..row_bytes]);
}
