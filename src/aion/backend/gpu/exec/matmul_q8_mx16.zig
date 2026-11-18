// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! A q8_0 GEMM on the device's 16x16x16 f16 matrix tiles (f32 accumulation)
//! that dequantizes B as it goes (kernels/matmul_q8_mx16.wgsl), for
//! `lanes32x16` weights: the `matrix_f16` sibling of `matmul_q8_mx.zig`, what a
//! prefill takes on NVIDIA's tensor cores instead of dequantizing the whole
//! weight to f32 and multiplying that on the shader cores.

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

const TILE: u32 = 16;

/// One block shape: `bm x bn` of C per workgroup, `sgm x sgn` of it per SIMD
/// group, staged rows padded by `pad` halves.
pub const Config = struct {
    bm: u32,
    bn: u32,
    sgm: u32,
    sgn: u32,
    pad: u32 = 8,

    fn threads(cfg: Config) u32 {
        return (cfg.bm / cfg.sgm) * (cfg.bn / cfg.sgn) * 32;
    }
};

/// The autotune menu. The first config is the default where timing ties, and the
/// one the narrowest products fit. Measured on an RTX 4080 Laptop over Gemma 4's
/// prefill shapes: 64x64 wins nearly every shape, the other two occasionally;
/// 128x64 and the 64x64-per-group 128x128 never did, and were dropped to keep
/// first-run tuning short (~25 ms per candidate per shape).
pub const configs = [_]Config{
    .{ .bm = 64, .bn = 64, .sgm = 32, .sgn = 32 },
    .{ .bm = 128, .bn = 128, .sgm = 64, .sgn = 32 },
    .{ .bm = 64, .bn = 128, .sgm = 32, .sgn = 64 },
};

/// Field order matches `struct Params` in matmul_q8_mx16.wgsl.
pub const Params = extern struct { m: u32, n: u32, k: u32, c_row: u32, c_off: u32, _p0: u32 = 0, _p1: u32 = 0, _p2: u32 = 0 };

/// Whether `cfg` takes an `m x n` product: whole blocks each way (the last one
/// slides back).
fn configFits(cfg: Config, m: u32, n: u32) bool {
    return m >= cfg.bm and n >= cfg.bn;
}

/// Whether some config takes `C = A @ B^T` here: K in whole q8 blocks, B in whole
/// `lanes32x16` groups, C rows a tile store can address (16-byte aligned), and
/// nothing to scale or accumulate, so the slid-back blocks' shared elements can
/// be written twice and a tuning trial can rerun the product on the real C.
pub fn fits(m: u32, n: u32, k: u32, c_row: u32, c_off: u32, alpha: f32, beta: f32) bool {
    if (n % 32 != 0 or k % 32 != 0 or c_row % 4 != 0 or c_off % 4 != 0) return false;
    if (alpha != 1.0 or beta != 0.0) return false;
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
    const key: autotune.Key = .of(.gemm_q8, .{ autotune.sizeClass(params.m), params.n, params.k });
    const idx = (try ctx.tune.pick(ctx, frame, key, configs.len, cand)) orelse return error.Unsupported;
    return recordConfig(ctx, frame, idx, bufs, sizes, params);
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
        .name = std.fmt.comptimePrint("matmul_q8_mx16_{d}x{d}_{d}x{d}", .{ cfg.bm, cfg.bn, cfg.sgm, cfg.sgn }),
        .wgsl = source(cfg),
    };
    break :blk out;
};

fn source(comptime cfg: Config) []const u8 {
    @setEvalBranchQuota(10_000_000);
    if (cfg.bm % cfg.sgm != 0 or cfg.bn % cfg.sgn != 0) @compileError("a block is whole SIMD-group shares");
    if (cfg.sgm % TILE != 0 or cfg.sgn % TILE != 0) @compileError("a share is whole tiles");
    if (cfg.bn % 32 != 0) @compileError("B is staged in groups of 32 rows");
    if (((32 + cfg.pad) * 2) % 16 != 0) @compileError("a staged row starts 16-byte aligned");
    const ti = cfg.sgm / TILE;
    const tj = cfg.sgn / TILE;
    var s: []const u8 = @embedFile("../kernels/matmul_q8_mx16.wgsl");
    s = template.fill(s, "consts", &.{
        std.fmt.comptimePrint("const BM: u32 = {d}u;", .{cfg.bm}),
        std.fmt.comptimePrint("const BN: u32 = {d}u;", .{cfg.bn}),
        std.fmt.comptimePrint("const SGM: u32 = {d}u;", .{cfg.sgm}),
        std.fmt.comptimePrint("const SGN: u32 = {d}u;", .{cfg.sgn}),
        std.fmt.comptimePrint("const THREADS: u32 = {d}u;", .{cfg.threads()}),
        std.fmt.comptimePrint("const SK: u32 = {d}u;", .{32 + cfg.pad}),
    });
    var acc: []const []const u8 = &.{};
    var store: []const []const u8 = &.{};
    var tiles: []const []const u8 = &.{};
    for (0..ti) |i| for (0..tj) |j| {
        acc = acc ++ .{std.fmt.comptimePrint("var c{d}{d} = coop_mat16x16<f32, C>();", .{ i, j })};
        store = store ++ .{std.fmt.comptimePrint(
            "let at{d}{d} = (m0 + sm + {d}u) * p.c_row + p.c_off + n0 + sn + {d}u;",
            .{ i, j, i * TILE, j * TILE },
        )};
        store = store ++ .{std.fmt.comptimePrint("coopStoreT(c{d}{d}, &c[at{d}{d}], c_row);", .{ i, j, i, j })};
    };
    for (0..2) |kk| {
        // A tiles from rows of A; B tiles from rows of B, which are their columns.
        tiles = tiles ++ .{"{"};
        for (0..ti) |i| tiles = tiles ++ .{std.fmt.comptimePrint(
            "let a{d} = coopLoadT<coop_mat16x16<f16, A>>(&sa[(sm + {d}u) * SK + {d}u], SK);",
            .{ i, i * TILE, kk * TILE },
        )};
        for (0..tj) |j| tiles = tiles ++ .{std.fmt.comptimePrint(
            "let b{d} = coopLoad<coop_mat16x16<f16, B>>(&sb[(sn + {d}u) * SK + {d}u], SK);",
            .{ j, j * TILE, kk * TILE },
        )};
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
