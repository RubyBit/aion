// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");
const exec_utils = @import("utils.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;
const DType = types.DType;

/// Elementwise cast between scalar dtypes: f16<->f32, f32<->i32, and same-dtype
/// (a copy). Its elements split across the pool.
pub fn execCast(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepCast,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    const out_meta = try store.meta(s.out);
    const in_meta = try store.meta(s.x);
    if (out_meta.dtype != s.to_dtype) return BackendError.InvalidArgument;
    if (out_meta.rank != in_meta.rank) return BackendError.InvalidArgument;
    return castWhole(pool, thread_count, s, store, in_meta.dtype);
}

/// A whole tensor: its elements split across the pool.
fn castWhole(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepCast,
    store: tensor_store.TensorStore,
    from: DType,
) ExecuteProgramError!void {
    const out_v = try store.acquireMut(s.out);
    defer store.releaseMut(out_v.token);
    const in_view = try store.acquireConst(s.x);
    defer store.releaseConst(in_view.token);
    const Ctx = struct {
        from: DType,
        to: DType,
        out: []u8,
        in: []const u8,

        fn run(c: @This(), start: usize, end: usize, _: usize) BackendError!void {
            const ib = c.from.info().block_bytes;
            const ob = c.to.info().block_bytes;
            return castBytes(c.from, c.to, c.in[start * ib .. end * ib], c.out[start * ob .. end * ob]);
        }
    };
    const out_view = out_v.bufferView();
    const ctx: Ctx = .{ .from = from, .to = s.to_dtype, .out = out_view.bytes, .in = in_view.bufferView().bytes };
    const n = exec_utils.elemCountFromView(out_view);
    const unit = @max(from.info().block_bytes, s.to_dtype.info().block_bytes);
    return exec_utils.parallelRange(BackendError, pool, thread_count, n, unit, ctx, Ctx.run);
}

fn castBytes(from: DType, to: DType, in_bytes: []const u8, out_bytes: []u8) BackendError!void {
    if (from == to) {
        if (in_bytes.len != out_bytes.len) return BackendError.InvalidArgument;
        @memcpy(out_bytes, in_bytes);
        return;
    }

    if (from == .f32 and to == .f16) {
        if ((in_bytes.len % @sizeOf(f32)) != 0) return BackendError.InvalidArgument;
        const n: usize = in_bytes.len / @sizeOf(f32);
        if (out_bytes.len < n * @sizeOf(f16)) return BackendError.InvalidArgument;
        const src: [*]align(1) const f32 = @ptrCast(in_bytes.ptr);
        const dst: [*]align(1) f16 = @ptrCast(out_bytes.ptr);
        var i: usize = 0;
        while (i < n) : (i += 1) {
            dst[i] = @floatCast(src[i]);
        }
        return;
    }

    if (from == .f16 and to == .f32) {
        if ((in_bytes.len % @sizeOf(f16)) != 0) return BackendError.InvalidArgument;
        const n: usize = in_bytes.len / @sizeOf(f16);
        if (out_bytes.len < n * @sizeOf(f32)) return BackendError.InvalidArgument;
        const src: [*]align(1) const f16 = @ptrCast(in_bytes.ptr);
        const dst: [*]align(1) f32 = @ptrCast(out_bytes.ptr);
        var i: usize = 0;
        while (i < n) : (i += 1) {
            dst[i] = @floatCast(src[i]);
        }
        return;
    }

    if (from == .f32 and to == .i32) {
        if ((in_bytes.len % @sizeOf(f32)) != 0) return BackendError.InvalidArgument;
        const n: usize = in_bytes.len / @sizeOf(f32);
        if (out_bytes.len < n * @sizeOf(i32)) return BackendError.InvalidArgument;
        const src: [*]align(1) const f32 = @ptrCast(in_bytes.ptr);
        const dst: [*]align(1) i32 = @ptrCast(out_bytes.ptr);
        var i: usize = 0;
        while (i < n) : (i += 1) {
            // Round-to-nearest; values are exact small integers in the decode loop.
            dst[i] = @intFromFloat(@round(src[i]));
        }
        return;
    }

    if (from == .i32 and to == .f32) {
        if ((in_bytes.len % @sizeOf(i32)) != 0) return BackendError.InvalidArgument;
        const n: usize = in_bytes.len / @sizeOf(i32);
        if (out_bytes.len < n * @sizeOf(f32)) return BackendError.InvalidArgument;
        const src: [*]align(1) const i32 = @ptrCast(in_bytes.ptr);
        const dst: [*]align(1) f32 = @ptrCast(out_bytes.ptr);
        var i: usize = 0;
        while (i < n) : (i += 1) {
            dst[i] = @floatFromInt(src[i]);
        }
        return;
    }

    return BackendError.Unsupported;
}
