// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! A q8_0 GEMM on the device's 8x8 f32 matrix tiles that dequantizes B as it
//! goes (kernels/matmul_q8_mx.wgsl), for `lanes32x16` weights: what a prefill or a
//! streaming chunk takes wherever the device has the tiles, instead of
//! dequantizing the whole weight into scratch and multiplying that.
//!
//! The `coopLoad`/`coopLoadT`/`coopStore` choices below follow naga 29's inverted
//! Metal layout and must be swapped when moving to wgpu >= 30
//! (gfx-rs/wgpu#9451): see the note at the top of matmul_q8_mx.wgsl.

const std = @import("std");
const wgpu = @import("../wgpu.zig");
const pipelines = @import("../pipelines.zig");
const context = @import("../context.zig");
const template = @import("../wgsl_template.zig");
const autotune = @import("../autotune.zig");
const backend_mod = @import("../../backend.zig");

const c = wgpu.c;
const Ctx = context.Ctx;
const Frame = @import("../frame.zig").Frame;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

const TILE: u32 = 8;
const BK: u32 = 32;

/// One block shape: `bm x bn` of C per workgroup, `sgm x sgn` of it per SIMD group.
pub const Config = struct {
    bm: u32,
    bn: u32,
    sgm: u32,
    sgn: u32,

    fn threads(cfg: Config) u32 {
        return (cfg.bm / cfg.sgm) * (cfg.bn / cfg.sgn) * 32;
    }
};

/// Rows of A from which a product takes the full-height block.
const TALL: u32 = 64;

/// The menu: the 64 x 64 block for products of `TALL` rows or more (a prefill), and
/// short blocks, timed against each other, for fewer (a streaming chunk, a small
/// batch), each of which reads every weight once for all its rows.
pub const configs = [_]Config{
    .{ .bm = TALL, .bn = 64, .sgm = 32, .sgn = 32 },
    .{ .bm = 8, .bn = 64, .sgm = 8, .sgn = 32 },
    .{ .bm = 16, .bn = 64, .sgm = 16, .sgn = 32 },
    .{ .bm = 16, .bn = 128, .sgm = 16, .sgn = 32 },
    .{ .bm = 32, .bn = 64, .sgm = 32, .sgn = 32 },
};

/// Field order matches `struct Params` in matmul_q8_mx.wgsl.
pub const Params = extern struct { m: u32, n: u32, k: u32, c_row: u32, c_off: u32, alpha: f32, beta: f32, _p0: u32 = 0 };

/// Whether `cfg` takes an `m x n` product: B in whole blocks (the last one slides
/// back), and the full-height block for `TALL` rows or more, a short one below.
fn configFits(cfg: Config, m: u32, n: u32) bool {
    return n >= cfg.bn and (m >= TALL) == (cfg.bm == TALL);
}

/// Whether the kernel takes an `[m, k] @ [n, k]^T` product: B in whole
/// `lanes32x16` groups and blocks, and K in whole q8 blocks.
pub fn fits(m: u32, n: u32, k: u32) bool {
    if (m == 0 or n % 32 != 0 or k % BK != 0) return false;
    for (configs) |cfg| if (configFits(cfg, m, n)) return true;
    return false;
}

/// Record the product on the config fastest for its shape, timed on the device
/// the first time the shape is seen. `bufs`/`sizes` are A, B, C.
pub fn record(ctx: Ctx, frame: *Frame, bufs: [3]c.WGPUBuffer, sizes: [3]u64, params: Params) ExecuteProgramError!void {
    const Cand = struct {
        ctx: Ctx,
        bufs: [3]c.WGPUBuffer,
        sizes: [3]u64,
        params: Params,

        pub fn eligible(t: @This(), idx: usize) bool {
            return configFits(configs[idx], t.params.m, t.params.n);
        }
        pub fn record(t: @This(), idx: usize, f: *Frame) ExecuteProgramError!void {
            return recordConfig(t.ctx, f, idx, t.bufs, t.sizes, t.params);
        }
    };
    const cand: Cand = .{ .ctx = ctx, .bufs = bufs, .sizes = sizes, .params = params };
    // A trial reruns the product on the real C, which beta != 0 would accumulate
    // into: that takes the first config that fits, untimed.
    const idx = if (params.beta != 0.0)
        for (0..configs.len) |i| {
            if (cand.eligible(i)) break i;
        } else null
    else
        try ctx.tune.pick(ctx, frame, .of(.gemm_q8_mx, .{ autotune.sizeClass(params.m), params.n, params.k }), configs.len, cand);
    return recordConfig(ctx, frame, idx orelse return error.Unsupported, bufs, sizes, params);
}

fn recordConfig(ctx: Ctx, frame: *Frame, idx: usize, bufs: [3]c.WGPUBuffer, sizes: [3]u64, params: Params) ExecuteProgramError!void {
    const built = try ctx.pipes.get(kernels[idx], "gemm_q8");
    try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{
        context.ceilDiv(params.n, configs[idx].bn),
        context.ceilDiv(params.m, configs[idx].bm),
        1,
    });
}

const kernels = blk: {
    var out: [configs.len]pipelines.KernelDesc = undefined;
    for (configs, 0..) |cfg, i| out[i] = .{
        .name = std.fmt.comptimePrint("matmul_q8_mx_{d}x{d}_{d}x{d}", .{ cfg.bm, cfg.bn, cfg.sgm, cfg.sgn }),
        .wgsl = source(cfg),
    };
    break :blk out;
};

/// A SIMD group's share is `sgm/8 x sgn/8` tiles; a block of K is 4 tiles deep.
fn source(comptime cfg: Config) []const u8 {
    @setEvalBranchQuota(10_000_000);
    if (cfg.bm % cfg.sgm != 0 or cfg.bn % cfg.sgn != 0) @compileError("a block is whole SIMD-group shares");
    if (cfg.sgm % TILE != 0 or cfg.sgn % TILE != 0) @compileError("a share is whole tiles");
    if (cfg.bn % 32 != 0) @compileError("B is staged in groups of 32 rows");
    const ti = cfg.sgm / TILE;
    const tj = cfg.sgn / TILE;
    var s: []const u8 = @embedFile("../kernels/matmul_q8_mx.wgsl");
    s = template.fill(s, "consts", &.{
        std.fmt.comptimePrint("const BM: u32 = {d}u;", .{cfg.bm}),
        std.fmt.comptimePrint("const BN: u32 = {d}u;", .{cfg.bn}),
        std.fmt.comptimePrint("const SGM: u32 = {d}u;", .{cfg.sgm}),
        std.fmt.comptimePrint("const SGN: u32 = {d}u;", .{cfg.sgn}),
        std.fmt.comptimePrint("const THREADS: u32 = {d}u;", .{cfg.threads()}),
        std.fmt.comptimePrint("const SH: u32 = {d}u;", .{@max((cfg.bm + cfg.bn) * BK, cfg.bm * cfg.bn)}),
    });
    var acc: []const []const u8 = &.{};
    var store: []const []const u8 = &.{};
    var tiles: []const []const u8 = &.{};
    for (0..ti) |i| for (0..tj) |j| {
        acc = acc ++ .{std.fmt.comptimePrint("var c{d}{d} = coop_mat8x8<f32, C>();", .{ i, j })};
        store = store ++ .{std.fmt.comptimePrint("coopStore(c{d}{d}, &sh[(sm + {d}u) * BN + sn + {d}u], BN);", .{ i, j, i * TILE, j * TILE })};
    };
    for (0..BK / TILE) |kk| {
        // A tiles from rows of A; B tiles from rows of B, which are their columns.
        tiles = tiles ++ .{"{"};
        tiles = tiles ++ template.each(ti, TILE, std.fmt.comptimePrint("let a{{0d}} = coopLoad<coop_mat8x8<f32, A>>(&sh[(sm + {{1d}}u) * BK + {d}u], BK);", .{kk * TILE}));
        tiles = tiles ++ template.each(tj, TILE, std.fmt.comptimePrint("let b{{0d}} = coopLoadT<coop_mat8x8<f32, B>>(&sh[B_AT + (sn + {{1d}}u) * BK + {d}u], BK);", .{kk * TILE}));
        for (0..ti) |i| for (0..tj) |j| {
            tiles = tiles ++ .{std.fmt.comptimePrint("c{d}{d} = coopMultiplyAdd(a{d}, b{d}, c{d}{d});", .{ i, j, i, j, i, j })};
        };
        tiles = tiles ++ .{"}"};
    }
    s = template.fill(s, "acc_decl", acc);
    s = template.fill(s, "store_acc", store);
    s = template.fill(s, "tiles", tiles);
    return s;
}
