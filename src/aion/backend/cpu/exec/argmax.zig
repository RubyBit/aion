// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// ArgMax over the last axis, into a tensor of i32 indices. Sufficient for RNNT decode (argmax over
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
    if (in_meta.dtype != .f32 and in_meta.dtype != .f16) return BackendError.InvalidArgument;
    if (out_meta.dtype != .i32) return BackendError.InvalidArgument;

    const in_rank: usize = @as(usize, in_meta.rank);
    // v1: reduce the last axis only.
    if (s.axis != in_rank - 1) return BackendError.InvalidArgument;

    const n: usize = in_meta.shape[in_rank - 1];
    if (n == 0) return BackendError.InvalidArgument;
    var outer: usize = 1;
    for (in_meta.shape[0 .. in_rank - 1]) |dim| outer = std.math.mul(usize, outer, dim) catch return BackendError.InvalidArgument;

    const in_view = try store.acquireConst(s.a);
    defer store.releaseConst(in_view.token);
    const out_view = try store.acquireMut(s.out);
    defer store.releaseMut(out_view.token);

    const in_bytes = in_view.bufferView().bytes;
    const out_bytes = out_view.bufferView().bytes;
    if (out_bytes.len < outer * 4) return BackendError.InvalidArgument;
    const out_buf: []align(1) i32 = simd.bytesAsSliceMutUnaligned(i32, out_bytes);

    // f16 compares on the widened value, which is order-preserving, so the index
    // picked is the one the f32 path would pick for the same logical row.
    switch (in_meta.dtype) {
        .f32 => try argmaxRows(f32, in_bytes, out_buf, outer, n),
        .f16 => try argmaxRows(f16, in_bytes, out_buf, outer, n),
        else => return BackendError.InvalidArgument,
    }
}

fn argmaxRows(
    comptime T: type,
    in_bytes: []const u8,
    out_buf: []align(1) i32,
    outer: usize,
    n: usize,
) ExecuteProgramError!void {
    const in_buf: []align(1) const T = simd.bytesAsSliceConstUnaligned(T, in_bytes);
    if (in_buf.len < outer * n) return BackendError.InvalidArgument;

    var o: usize = 0;
    while (o < outer) : (o += 1) {
        const row = in_buf[o * n .. o * n + n];
        var best_i: usize = 0;
        var best_v: f32 = @floatCast(row[0]);
        var j: usize = 1;
        while (j < n) : (j += 1) {
            const v: f32 = @floatCast(row[j]);
            if (v > best_v) {
                best_v = v;
                best_i = j;
            }
        }
        out_buf[o] = @intCast(best_i);
    }
}
