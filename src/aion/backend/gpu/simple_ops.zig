// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Elementwise-binary and unary op execution for the GPU backend. These are the
//! "simple" ops: f32, multi-tile, no autotuning — one dispatch per tile over an
//! embedded hand-written WGSL kernel. Kept out of `backend.zig` (plumbing) and out
//! of `matmul/` (the heavy, tuned op family).

const std = @import("std");
const wgpu = @import("wgpu.zig");
const pipelines = @import("pipelines.zig");
const context = @import("context.zig");
const backend_mod = @import("../backend.zig");
const types = @import("../types.zig");
const executable = @import("../../runtime/executable.zig");

const c = wgpu.c;
const Ctx = context.Ctx;
const Frame = @import("frame.zig").Frame;
const ExecuteProgramError = backend_mod.ExecuteProgramError;
const KernelDesc = pipelines.KernelDesc;

const elementwise_kernel: KernelDesc = .{ .name = "elementwise", .wgsl = @embedFile("kernels/elementwise.wgsl") };
const unary_kernel: KernelDesc = .{ .name = "unary", .wgsl = @embedFile("kernels/unary.wgsl") };

const WORKGROUP_1D: u32 = 64; // must match @workgroup_size in elementwise/unary.wgsl

/// Uniform params for the 1D elementwise/unary kernels (16-byte aligned).
const ScalarParams = extern struct { n: u32, _pad0: u32 = 0, _pad1: u32 = 0, _pad2: u32 = 0 };

fn requireF32(meta: types.DType) ExecuteProgramError!void {
    if (meta != .f32) return error.Unsupported;
}

pub fn execElemwiseBinary(ctx: Ctx, frame: *Frame, s: executable.StepElemwiseBinaryTiled) ExecuteProgramError!void {
    const hs = ctx.rstore.tensorStore();
    const meta = hs.meta(s.out) catch return error.ExecutionFailed;
    try requireF32(meta.dtype);

    const entry: [:0]const u8 = switch (s.op) {
        .add => "add",
        .sub => "sub",
        .mul => "mul",
        .div => "divide",
        else => return error.Unsupported, // comparisons produce i32 -- later
    };
    const built = try ctx.pipes.get(elementwise_kernel, entry);

    const total = totalTiles(meta);
    var ti: usize = 0;
    while (ti < total) : (ti += 1) {
        const da = ctx.rstore.acquireTileDeviceConstLinear(s.a, ti) catch return error.ExecutionFailed;
        const db = ctx.rstore.acquireTileDeviceConstLinear(s.b, ti) catch return error.ExecutionFailed;
        const dout = ctx.rstore.acquireTileDeviceMutLinear(s.out, ti) catch return error.ExecutionFailed;
        if (!context.storageBindingFits(ctx, da.len) or !context.storageBindingFits(ctx, db.len) or !context.storageBindingFits(ctx, dout.len)) {
            hs.releaseConst(da.token);
            hs.releaseConst(db.token);
            hs.releaseMut(dout.token);
            return error.Unsupported;
        }

        const bufs = [_]c.WGPUBuffer{
            ctx.devmem.bufferFor(da.handle).?,
            ctx.devmem.bufferFor(db.handle).?,
            ctx.devmem.bufferFor(dout.handle).?,
        };
        const sizes = [_]u64{ da.len, db.len, dout.len };
        const n: u32 = @intCast(dout.len / @sizeOf(f32));
        const params: ScalarParams = .{ .n = n };
        try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ context.ceilDiv(n, WORKGROUP_1D), 1, 1 });

        hs.releaseConst(da.token);
        hs.releaseConst(db.token);
        hs.releaseMut(dout.token);
    }
}

pub fn execUnary(ctx: Ctx, frame: *Frame, s: executable.StepUnaryTiled) ExecuteProgramError!void {
    const hs = ctx.rstore.tensorStore();
    const meta = hs.meta(s.out) catch return error.ExecutionFailed;
    try requireF32(meta.dtype);

    const entry: [:0]const u8 = switch (s.op) {
        .relu => "relu",
        .gelu => "gelu",
        .silu => "silu",
        .sigmoid => "sigmoid",
        .tanh => "tanh_",
        .sqrt => "sqrt_",
        .log => "log_",
    };
    const built = try ctx.pipes.get(unary_kernel, entry);

    const total = totalTiles(meta);
    var ti: usize = 0;
    while (ti < total) : (ti += 1) {
        const dx = ctx.rstore.acquireTileDeviceConstLinear(s.a, ti) catch return error.ExecutionFailed;
        const dout = ctx.rstore.acquireTileDeviceMutLinear(s.out, ti) catch return error.ExecutionFailed;
        if (!context.storageBindingFits(ctx, dx.len) or !context.storageBindingFits(ctx, dout.len)) {
            hs.releaseConst(dx.token);
            hs.releaseMut(dout.token);
            return error.Unsupported;
        }

        const bufs = [_]c.WGPUBuffer{
            ctx.devmem.bufferFor(dx.handle).?,
            ctx.devmem.bufferFor(dout.handle).?,
        };
        const sizes = [_]u64{ dx.len, dout.len };
        const n: u32 = @intCast(dout.len / @sizeOf(f32));
        const params: ScalarParams = .{ .n = n };
        try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ context.ceilDiv(n, WORKGROUP_1D), 1, 1 });

        hs.releaseConst(dx.token);
        hs.releaseMut(dout.token);
    }
}

fn totalTiles(meta: anytype) usize {
    var total: usize = 1;
    for (meta.tile_counts) |cnt| total *= cnt;
    return total;
}
