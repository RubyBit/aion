// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// ArgMax over the last axis. v1: the input is a single tile (whole tensor) and the
// output is a single tile of i32 indices. Sufficient for RNNT decode (argmax over
// the small joint-logits vocab axis). Output[o] = index in [0, N) of the max of
// input[o, :] for each of the `outer = prod(shape[:-1])` rows.
const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");

const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");
const simd = @import("../kernels/simd.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

pub fn execArgMax(
    s: executable.StepArgMax,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    const in_meta = try store.meta(s.a);
    const out_meta = try store.meta(s.out);

    if (in_meta.rank == 0) return BackendError.InvalidArgument;
    if (in_meta.dtype != .f32) return BackendError.InvalidArgument;
    if (out_meta.dtype != .i32) return BackendError.InvalidArgument;

    const in_rank: usize = @as(usize, in_meta.rank);
    // v1: reduce the last axis only, single tile on input and output.
    if (s.axis != in_rank - 1) return BackendError.InvalidArgument;
    var d: usize = 0;
    while (d < in_rank) : (d += 1) {
        if (in_meta.tile_counts[d] != 1) return BackendError.InvalidArgument;
    }
    d = 0;
    while (d < @as(usize, out_meta.rank)) : (d += 1) {
        if (out_meta.tile_counts[d] != 1) return BackendError.InvalidArgument;
    }

    const n: usize = in_meta.shape[in_rank - 1];
    if (n == 0) return BackendError.InvalidArgument;
    var outer: usize = 1;
    if (in_rank > 1) {
        d = 0;
        while (d < in_rank - 1) : (d += 1) outer = std.math.mul(usize, outer, in_meta.shape[d]) catch return BackendError.InvalidArgument;
    }

    const in_tile = try store.acquireTileConstLinear(s.a, 0);
    defer store.releaseConst(in_tile.token);
    const out_tile = try store.acquireTileMutLinear(s.out, 0);
    defer store.releaseMut(out_tile.token);

    const in_buf: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, in_tile.bufferView().bytes);
    const out_bytes = out_tile.bufferView().bytes;
    if (in_buf.len < outer * n) return BackendError.InvalidArgument;
    if (out_bytes.len < outer * 4) return BackendError.InvalidArgument;
    const out_buf: []align(1) i32 = simd.bytesAsSliceMutUnaligned(i32, out_bytes);

    var o: usize = 0;
    while (o < outer) : (o += 1) {
        const row = in_buf[o * n .. o * n + n];
        var best_i: usize = 0;
        var best_v: f32 = row[0];
        var j: usize = 1;
        while (j < n) : (j += 1) {
            if (row[j] > best_v) {
                best_v = row[j];
                best_i = j;
            }
        }
        out_buf[o] = @intCast(best_i);
    }
}
