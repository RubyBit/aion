// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const wgpu = @import("../wgpu.zig");
const context = @import("../context.zig");
const Frame = @import("../frame.zig").Frame;
const ts = @import("../../../runtime/tensor_store.zig");
const exe = @import("../../../runtime/executable.zig");
const Error = @import("../../backend.zig").ExecuteProgramError;
const kernel: @import("../pipelines.zig").KernelDesc = .{ .name = "max_pool2d", .wgsl = @embedFile("../kernels/pool.wgsl") };
const Params = extern struct {
    src_shape: [4]u32,
    src_origin: [4]u32,
    src_stride: [4]u32,
    dst_shape: [4]u32,
    dst_origin: [4]u32,
    dst_stride: [4]u32,
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
    const bytes: usize = switch (input.dtype) {
        .f32 => 4,
        .f16 => 2,
        else => return error.Unsupported,
    };
    const built = try ctx.pipes.get(kernel, if (bytes == 4) "pool_f32" else "pool_f16");
    for (0..context.totalTiles(output)) |oi| {
        const dst = try ctx.store.acquireTileDeviceMutLinear(s.out, oi);
        defer ctx.store.releaseMut(dst.token);
        if (!context.storageBindingFits(ctx, dst.len)) return error.Unsupported;
        var oc: [4]usize = undefined;
        try ts.decodeTileCoords(output, oi, &oc);
        for (0..context.totalTiles(input)) |ii| {
            const src = try ctx.store.acquireTileDeviceConstLinear(s.x, ii);
            defer ctx.store.releaseConst(src.token);
            if (!context.storageBindingFits(ctx, src.len)) return error.Unsupported;
            var ic: [4]usize = undefined;
            try ts.decodeTileCoords(input, ii, &ic);
            var p: Params = undefined;
            var n: usize = 1;
            for (0..4) |d| {
                p.src_shape[d] = try cast(src.shape_mem[d]);
                p.src_origin[d] = try cast(ic[d] * input.tile_shape[d]);
                p.src_stride[d] = try cast(@as(usize, @intCast(src.strides_mem[d])) / bytes);
                p.dst_shape[d] = try cast(dst.shape_mem[d]);
                p.dst_origin[d] = try cast(oc[d] * output.tile_shape[d]);
                p.dst_stride[d] = try cast(@as(usize, @intCast(dst.strides_mem[d])) / bytes);
                n *= dst.shape_mem[d];
            }
            const o = s.opts;
            p.window = .{ try cast(o.kernel_h), try cast(o.kernel_w), try cast(o.stride_h), try cast(o.stride_w) };
            p.geometry = .{ try cast(o.dilation_h), try cast(o.dilation_w), try cast(o.pad_top), try cast(o.pad_left) };
            p.control = .{ try cast(n), @intFromBool(ii == 0), 0, 0 };
            const bufs = [_]wgpu.c.WGPUBuffer{ ctx.devmem.bufferFor(src.handle).?, ctx.devmem.bufferFor(dst.handle).? };
            const sizes = [_]u64{ src.len, dst.len };
            try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&p), .{ @max(1, @min(context.ceilDiv(try cast(n), 64), context.MAX_GROUPS_1D)), 1, 1 });
        }
    }
}
