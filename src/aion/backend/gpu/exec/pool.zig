// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const wgpu = @import("../wgpu.zig");
const context = @import("../context.zig");
const Frame = @import("../frame.zig").Frame;
const exe = @import("../../../runtime/executable.zig");
const Error = @import("../../backend.zig").ExecuteProgramError;
const kernel: @import("../pipelines.zig").KernelDesc = .{ .name = "max_pool2d", .wgsl = @embedFile("../kernels/pool.wgsl") };
/// Field order matches `struct Params` in pool.wgsl.
const Params = extern struct {
    src_shape: [4]u32,
    dst_shape: [4]u32,
    window: [4]u32,
    geometry: [4]u32,
    control: [4]u32,
};
fn cast(n: usize) Error!u32 {
    return std.math.cast(u32, n) orelse error.Unsupported;
}

pub fn exec(ctx: context.Ctx, frame: *Frame, s: exe.StepMaxPool2D) Error!void {
    const input = try ctx.store.meta(s.x);
    const output = try ctx.store.meta(s.out);
    const shape = s.opts.output(input.shape) catch return error.InvalidArgument;
    if (!std.mem.eql(usize, &shape, output.shape) or input.dtype != output.dtype) return error.ExecutionFailed;
    if (input.chunks != 1 or output.chunks != 1) return error.Unsupported;
    const f32_data = switch (input.dtype) {
        .f32 => true,
        .f16 => false,
        else => return error.Unsupported,
    };
    var n: usize = 1;
    for (output.shape) |d| n *= d;
    if (n == 0) return;

    const src = try ctx.store.acquireConst(s.x);
    defer ctx.store.releaseConst(src.token);
    const dst = try ctx.store.acquireMut(s.out);
    defer ctx.store.releaseMut(dst.token);
    if (!context.storageBindingFits(ctx, src.len) or !context.storageBindingFits(ctx, dst.len)) return error.Unsupported;

    var p: Params = undefined;
    for (0..4) |d| {
        p.src_shape[d] = try cast(input.shape[d]);
        p.dst_shape[d] = try cast(output.shape[d]);
    }
    const o = s.opts;
    p.window = .{ try cast(o.kernel_h), try cast(o.kernel_w), try cast(o.stride_h), try cast(o.stride_w) };
    p.geometry = .{ try cast(o.dilation_h), try cast(o.dilation_w), try cast(o.pad_top), try cast(o.pad_left) };
    p.control = .{ try cast(n), 0, 0, 0 };
    const built = try ctx.pipes.get(kernel, if (f32_data) "pool_f32" else "pool_f16");
    const bufs = [_]wgpu.c.WGPUBuffer{ ctx.devmem.bufferFor(src.handle).?, ctx.devmem.bufferFor(dst.handle).? };
    const sizes = [_]u64{ src.len, dst.len };
    try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&p), .{ @max(1, @min(context.ceilDiv(p.control[0], 64), context.MAX_GROUPS_1D)), 1, 1 });
}
