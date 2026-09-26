// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");

const relu_k = @import("../kernels/relu.zig");
const gelu_k = @import("../kernels/gelu.zig");
const silu_k = @import("../kernels/silu.zig");
const sigmoid_k = @import("../kernels/sigmoid.zig");
const tanh_k = @import("../kernels/tanh.zig");
const sqrt_k = @import("../kernels/sqrt.zig");
const log_k = @import("../kernels/log.zig");

const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const exec_utils = @import("utils.zig");
const executable = @import("../../../runtime/executable.zig");

const BackendError = types.BackendError;
const UnaryOp = types.UnaryOp;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

/// `out = op(a)`, its elements split across the pool.
pub fn execUnary(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepUnary,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    const out_v = try store.acquireMut(s.out);
    defer store.releaseMut(out_v.token);
    const a_view = try store.acquireConst(s.a);
    defer store.releaseConst(a_view.token);
    const out_view = out_v.bufferView();
    const Ctx = struct {
        op: UnaryOp,
        dtype: types.DType,
        out: []u8,
        a: []const u8,

        fn run(c: @This(), start: usize, end: usize, _: usize) BackendError!void {
            const eb = c.dtype.info().block_bytes;
            const out = c.out[start * eb .. end * eb];
            const a = c.a[start * eb .. end * eb];
            return switch (c.dtype) {
                .f32 => dispatchF32(c.op, out, a, end - start),
                .f16 => dispatchF16(c.op, out, a, end - start),
                else => BackendError.InvalidArgument,
            };
        }
    };
    const ctx: Ctx = .{ .op = s.op, .dtype = out_view.dtype, .out = out_view.bytes, .a = a_view.bufferView().bytes };
    const n = exec_utils.elemCountFromView(out_view);
    return exec_utils.parallelRange(BackendError, pool, thread_count, n, out_view.dtype.info().block_bytes, ctx, Ctx.run);
}

pub fn dispatchF32(op: UnaryOp, out_bytes: []u8, a_bytes: []const u8, n: usize) BackendError!void {
    return switch (op) {
        .relu => relu_k.reluF32(out_bytes, a_bytes, n),
        .gelu => gelu_k.geluF32(out_bytes, a_bytes, n),
        .silu => silu_k.siluF32(out_bytes, a_bytes, n),
        .sigmoid => sigmoid_k.sigmoidF32(out_bytes, a_bytes, n),
        .tanh => tanh_k.tanhF32(out_bytes, a_bytes, n),
        .sqrt => sqrt_k.sqrtF32(out_bytes, a_bytes, n),
        .log => log_k.logF32(out_bytes, a_bytes, n),
    };
}

pub fn dispatchF16(op: UnaryOp, out_bytes: []u8, a_bytes: []const u8, n: usize) BackendError!void {
    return switch (op) {
        .relu => relu_k.reluF16(out_bytes, a_bytes, n),
        .gelu => gelu_k.geluF16(out_bytes, a_bytes, n),
        .silu => silu_k.siluF16(out_bytes, a_bytes, n),
        .sigmoid => sigmoid_k.sigmoidF16(out_bytes, a_bytes, n),
        .tanh => tanh_k.tanhF16(out_bytes, a_bytes, n),
        .sqrt => sqrt_k.sqrtF16(out_bytes, a_bytes, n),
        .log => log_k.logF16(out_bytes, a_bytes, n),
    };
}
