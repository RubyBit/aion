// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! A q8_0 GEMM on the device's 8x8 f32 matrix tiles that dequantizes B as it
//! goes (kernels/matmul_q8_mx.wgsl), for `lanes32x16` weights: what a prefill
//! takes wherever the device has the tiles, instead of dequantizing the whole
//! weight into scratch and multiplying that.
//!
//! The `coopLoad`/`coopLoadT`/`coopStore` choices below follow naga 29's inverted
//! Metal layout and must be swapped when moving to wgpu >= 30
//! (gfx-rs/wgpu#9451): see the note at the top of matmul_q8_mx.wgsl.

const std = @import("std");
const wgpu = @import("../wgpu.zig");
const pipelines = @import("../pipelines.zig");
const context = @import("../context.zig");
const template = @import("../wgsl_template.zig");
const backend_mod = @import("../../backend.zig");

const c = wgpu.c;
const Ctx = context.Ctx;
const Frame = @import("../frame.zig").Frame;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

/// Rows of A and of B per workgroup (`BM`, `BN` in the shader).
const BLOCK: u32 = 64;

/// Field order matches `struct Params` in matmul_q8_mx.wgsl.
pub const Params = extern struct { m: u32, n: u32, k: u32, c_row: u32, c_off: u32, alpha: f32, beta: f32, _p0: u32 = 0 };

/// Whether the kernel takes an `[m, k] @ [n, k]^T` product: whole blocks each way
/// (the last one slides back), and K in whole q8 blocks.
pub fn fits(m: u32, n: u32, k: u32) bool {
    return m >= BLOCK and n >= BLOCK and n % 32 == 0 and k % 32 == 0;
}

/// Record the product. `bufs`/`sizes` are A, B, C.
pub fn record(ctx: Ctx, frame: *Frame, bufs: [3]c.WGPUBuffer, sizes: [3]u64, params: Params) ExecuteProgramError!void {
    const built = try ctx.pipes.get(kernel, "gemm_q8");
    try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{
        context.ceilDiv(params.n, BLOCK),
        context.ceilDiv(params.m, BLOCK),
        1,
    });
}

const kernel: pipelines.KernelDesc = .{ .name = "matmul_q8_mx", .wgsl = source() };

/// A SIMD group's quarter of the block is 4 x 4 tiles; a block of K is 4 tiles deep.
fn source() []const u8 {
    @setEvalBranchQuota(1_000_000);
    var s: []const u8 = @embedFile("../kernels/matmul_q8_mx.wgsl");
    var acc: []const []const u8 = &.{};
    var store: []const []const u8 = &.{};
    var tiles: []const []const u8 = &.{};
    for (0..4) |i| for (0..4) |j| {
        acc = acc ++ .{std.fmt.comptimePrint("var c{d}{d} = coop_mat8x8<f32, C>();", .{ i, j })};
        store = store ++ .{std.fmt.comptimePrint("coopStore(c{d}{d}, &sh[(sm + {d}u) * BN + sn + {d}u], BN);", .{ i, j, i * 8, j * 8 })};
    };
    for (0..4) |kk| {
        // A tiles from rows of A; B tiles from rows of B, which are their columns.
        tiles = tiles ++ .{"{"};
        tiles = tiles ++ template.each(4, 8, std.fmt.comptimePrint("let a{{0d}} = coopLoad<coop_mat8x8<f32, A>>(&sh[(sm + {{1d}}u) * BK + {d}u], BK);", .{kk * 8}));
        tiles = tiles ++ template.each(4, 8, std.fmt.comptimePrint("let b{{0d}} = coopLoadT<coop_mat8x8<f32, B>>(&sh[B_AT + (sn + {{1d}}u) * BK + {d}u], BK);", .{kk * 8}));
        for (0..4) |i| for (0..4) |j| {
            tiles = tiles ++ .{std.fmt.comptimePrint("c{d}{d} = coopMultiplyAdd(a{d}, b{d}, c{d}{d});", .{ i, j, i, j, i, j })};
        };
        tiles = tiles ++ .{"}"};
    }
    s = template.fill(s, "acc_decl", acc);
    s = template.fill(s, "store_acc", store);
    s = template.fill(s, "tiles", tiles);
    return s;
}
