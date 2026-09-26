// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const elemwise = @import("../kernels/elemwise.zig");
const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const exec_utils = @import("utils.zig");
const unary_exec = @import("unary.zig");
const executable = @import("../../../runtime/executable.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

/// The binary op on views of the output and both operands: the row ranges
/// `execElemwiseBinary` hands each thread.
fn binaryViews(
    s: executable.StepElemwiseBinary,
    out_view: types.BufferViewMut,
    a_view: types.BufferViewConst,
    b_view: types.BufferViewConst,
) !void {
    const n = exec_utils.elemCountFromView(out_view);

    // A gate is the activation folded into the multiply: apply `act` into `out`, then
    // multiply in place by `b`. Same order of operations as the unfused
    // `Unary(act)` + `ElemwiseBinary(mul)` pair, so the result is
    // bit-identical to it — which is what lets the GPU fused kernel be tested against
    // this. Threading and broadcast validation all come from the surrounding
    // elementwise machinery instead of a second copy of it.
    if (s.op == .gate) {
        if (s.broadcast.kind != .identical) return BackendError.InvalidArgument;
        if (out_view.dtype != .f32) return BackendError.InvalidArgument;
        try unary_exec.dispatchF32(s.act, out_view.bytes, a_view.bytes, n);
        return elemwise.elemwiseBinaryF32(.mul, out_view.bytes, out_view.bytes, b_view.bytes, n);
    }

    if (s.broadcast.kind == .identical) {
        return switch (out_view.dtype) {
            .f32 => elemwise.elemwiseBinaryF32(s.op, out_view.bytes, a_view.bytes, b_view.bytes, n),
            .f16 => elemwise.elemwiseBinaryF16(s.op, out_view.bytes, a_view.bytes, b_view.bytes, n),
            .i32 => elemwise.elemwiseBinaryI32(s.op, out_view.bytes, a_view.bytes, b_view.bytes, n),
            else => BackendError.InvalidArgument,
        };
    }

    if (s.broadcast.kind == .scalar_b or s.broadcast.kind == .contiguous_suffix_b) {
        const cols = exec_utils.elemCountFromView(b_view);
        return switch (out_view.dtype) {
            .f32 => switch (s.op) {
                .add => elemwise.contiguousSuffixBinaryF32Packed(.add, out_view.bytes, a_view.bytes, b_view.bytes, n, cols),
                .sub => elemwise.contiguousSuffixBinaryF32Packed(.sub, out_view.bytes, a_view.bytes, b_view.bytes, n, cols),
                .mul => elemwise.contiguousSuffixBinaryF32Packed(.mul, out_view.bytes, a_view.bytes, b_view.bytes, n, cols),
                .div => elemwise.contiguousSuffixBinaryF32Packed(.div, out_view.bytes, a_view.bytes, b_view.bytes, n, cols),
                else => BackendError.InvalidArgument,
            },
            .f16 => switch (s.op) {
                .add => elemwise.contiguousSuffixBinaryF16Packed(.add, out_view.bytes, a_view.bytes, b_view.bytes, n, cols),
                .sub => elemwise.contiguousSuffixBinaryF16Packed(.sub, out_view.bytes, a_view.bytes, b_view.bytes, n, cols),
                .mul => elemwise.contiguousSuffixBinaryF16Packed(.mul, out_view.bytes, a_view.bytes, b_view.bytes, n, cols),
                .div => elemwise.contiguousSuffixBinaryF16Packed(.div, out_view.bytes, a_view.bytes, b_view.bytes, n, cols),
                else => BackendError.InvalidArgument,
            },
            .i32 => elemwise.contiguousSuffixBinaryI32Packed(s.op, out_view.bytes, a_view.bytes, b_view.bytes, n, cols),
            else => BackendError.InvalidArgument,
        };
    }

    return switch (out_view.dtype) {
        .f32 => elemwise.elemwiseBroadcastF32(
            s.op,
            out_view,
            a_view,
            b_view,
            s.broadcast.a_broadcast_axes,
            s.broadcast.b_broadcast_axes,
        ),
        .f16 => elemwise.elemwiseBroadcastF16(
            s.op,
            out_view,
            a_view,
            b_view,
            s.broadcast.a_broadcast_axes,
            s.broadcast.b_broadcast_axes,
        ),
        .i32 => elemwise.elemwiseBroadcastI32(
            s.op,
            out_view,
            a_view,
            b_view,
            s.broadcast.a_broadcast_axes,
            s.broadcast.b_broadcast_axes,
        ),
        else => BackendError.InvalidArgument,
    };
}

/// The output split into contiguous row ranges across the pool.
///
/// Same-shape and suffix-broadcast operands split by element or by whole `b` rows.
/// A general broadcast splits along the outermost axis longer than one -- the only
/// split that keeps each output range contiguous -- and each operand follows along
/// that axis unless it broadcasts over it.
pub fn execElemwiseBinary(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepElemwiseBinary,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    const out_lease = try store.acquireMut(s.out);
    defer store.releaseMut(out_lease.token);
    const a_lease = try store.acquireConst(s.a);
    defer store.releaseConst(a_lease.token);
    const b_lease = try store.acquireConst(s.b);
    defer store.releaseConst(b_lease.token);
    const out = out_lease.bufferView();
    const a = a_lease.bufferView();
    const b = b_lease.bufferView();
    const n = exec_utils.elemCountFromView(out);
    const eb = out.dtype.info().block_bytes;

    const flat = s.op == .gate or s.broadcast.kind == .identical;
    const suffix = s.broadcast.kind == .scalar_b or s.broadcast.kind == .contiguous_suffix_b;
    if (flat or suffix) {
        // Rows of `b`'s length (one element when nothing broadcasts), as rank-1 views.
        const row: usize = if (flat) 1 else exec_utils.elemCountFromView(b);
        if (row == 0 or n % row != 0) return BackendError.InvalidArgument;
        const Ctx = struct {
            s: executable.StepElemwiseBinary,
            out: types.BufferViewMut,
            a: types.BufferViewConst,
            b: types.BufferViewConst,
            row: usize,
            flat: bool,

            fn run(c: @This(), lo: usize, hi: usize, _: usize) anyerror!void {
                const eb2 = c.out.dtype.info().block_bytes;
                const first = lo * c.row * eb2;
                const len = (hi - lo) * c.row * eb2;
                var shape: [1]usize = .{(hi - lo) * c.row};
                const strides: [1]isize = .{@intCast(eb2)};
                const layout: types.Layout = .{ .rank = 1, .shape = &shape, .strides_bytes = &strides };
                const out_v: types.BufferViewMut = .{ .bytes = c.out.bytes[first..][0..len], .dtype = c.out.dtype, .layout = layout };
                const a_v: types.BufferViewConst = .{ .bytes = c.a.bytes[first..][0..len], .dtype = c.a.dtype, .layout = layout };
                const b_v: types.BufferViewConst = if (c.flat) .{ .bytes = c.b.bytes[first..][0..len], .dtype = c.b.dtype, .layout = layout } else c.b;
                return binaryViews(c.s, out_v, a_v, b_v);
            }
        };
        const ctx: Ctx = .{ .s = s, .out = out, .a = a, .b = b, .row = row, .flat = flat };
        return exec_utils.parallelRange(anyerror, pool, thread_count, n / row, row * eb, ctx, Ctx.run) catch |e| @errorCast(e);
    }

    // General broadcast: split the outermost axis longer than one.
    const rank: usize = out.layout.rank;
    var ax: usize = 0;
    while (ax < rank and out.layout.shape[ax] <= 1) ax += 1;
    if (ax == rank) return binaryViews(s, out, a, b);
    const Ctx = struct {
        s: executable.StepElemwiseBinary,
        out: types.BufferViewMut,
        a: types.BufferViewConst,
        b: types.BufferViewConst,
        ax: usize,

        /// `v` restricted to `[lo, hi)` along output axis `ax`, unless it broadcasts there.
        fn follow(v: types.BufferViewConst, out_rank: usize, axis: usize, mask: u8, lo: usize, hi: usize, shape: *[8]usize) types.BufferViewConst {
            const r: usize = v.layout.rank;
            const off = out_rank - r;
            if (axis < off or (mask & (@as(u8, 1) << @intCast(axis))) != 0 or v.layout.shape[axis - off] == 1) return v;
            const in_ax = axis - off;
            @memcpy(shape[0..r], v.layout.shape);
            shape[in_ax] = hi - lo;
            const start: usize = lo * @as(usize, @intCast(v.layout.strides_bytes[in_ax]));
            var sub = v;
            sub.bytes = v.bytes[start..];
            sub.layout.shape = shape[0..r];
            return sub;
        }

        fn run(c: @This(), lo: usize, hi: usize, _: usize) anyerror!void {
            const r: usize = c.out.layout.rank;
            var out_shape: [8]usize = undefined;
            @memcpy(out_shape[0..r], c.out.layout.shape);
            out_shape[c.ax] = hi - lo;
            const stride: usize = @intCast(c.out.layout.strides_bytes[c.ax]);
            var out_v = c.out;
            out_v.bytes = c.out.bytes[lo * stride .. hi * stride];
            out_v.layout.shape = out_shape[0..r];
            var a_shape: [8]usize = undefined;
            var b_shape: [8]usize = undefined;
            const a_v = follow(c.a, r, c.ax, c.s.broadcast.a_broadcast_axes, lo, hi, &a_shape);
            const b_v = follow(c.b, r, c.ax, c.s.broadcast.b_broadcast_axes, lo, hi, &b_shape);
            return binaryViews(c.s, out_v, a_v, b_v);
        }
    };
    const rows = out.layout.shape[ax];
    const ctx: Ctx = .{ .s = s, .out = out, .a = a, .b = b, .ax = ax };
    return exec_utils.parallelRange(anyerror, pool, thread_count, rows, (n / rows) * eb, ctx, Ctx.run) catch |e| @errorCast(e);
}

/// A same-shape copy: one memcpy, split across the pool.
pub fn execCopy(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepCopy,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    const dst = try store.acquireMut(s.dst);
    defer store.releaseMut(dst.token);
    const src = try store.acquireConst(s.src);
    defer store.releaseConst(src.token);
    if (dst.dtype != src.dtype or dst.bytes.len != src.bytes.len) return BackendError.InvalidArgument;
    const Ctx = struct {
        dst: []u8,
        src: []const u8,
        fn run(c: @This(), lo: usize, hi: usize, _: usize) BackendError!void {
            @memcpy(c.dst[lo..hi], c.src[lo..hi]);
        }
    };
    return exec_utils.parallelRange(BackendError, pool, thread_count, dst.bytes.len, 1, Ctx{ .dst = dst.bytes, .src = src.bytes }, Ctx.run);
}
