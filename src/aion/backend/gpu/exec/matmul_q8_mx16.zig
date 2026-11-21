// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! A q8_0 GEMM on the device's 16x16x16 f16 matrix tiles (f32 accumulation)
//! that dequantizes B as it goes (kernels/matmul_q8_mx16.wgsl), for
//! `lanes32x16` weights: the `matrix_f16` sibling of `matmul_q8_mx.zig`, what a
//! prefill or a streaming chunk of two rows or more takes on NVIDIA's tensor
//! cores instead of dequantizing the whole weight to f32 and multiplying that on
//! the shader cores, or reading it once per row of A in a GEMV. A grid of few
//! workgroups splits K (`splitFor`) and reduces the partials after
//! (kernels/split_k_reduce.wgsl).

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
const BK: u32 = 32;

/// One block shape: `bm x bn` of C per workgroup, `sgm x sgn` of it per SIMD
/// group, `kb` q8 blocks of K staged per barrier, staged rows padded by `pad`
/// halves. A block shorter than `TALL` writes C through shared memory (`staged`).
pub const Config = struct {
    bm: u32,
    bn: u32,
    sgm: u32,
    sgn: u32,
    kb: u32 = 1,
    pad: u32 = 8,

    fn threads(cfg: Config) u32 {
        return (cfg.bm / cfg.sgm) * (cfg.bn / cfg.sgn) * 32;
    }
    pub fn staged(cfg: Config) bool {
        return cfg.bm < TALL;
    }
    fn sk(cfg: Config) u32 {
        return cfg.kb * BK + cfg.pad;
    }
    fn sharedBytes(cfg: Config) u32 {
        const c_bytes: u32 = if (cfg.staged()) cfg.bm * cfg.bn * 4 else 0;
        return (cfg.bm + cfg.bn) * cfg.sk() * 2 + c_bytes;
    }
};

/// Rows of A from which a product takes a full-height block.
const TALL: u32 = 64;

/// The autotune menu: tall blocks for products of `TALL` rows or more (a
/// prefill), short ones for fewer (a streaming chunk, a small batch), each of
/// which reads every weight once for all its rows. The first of each is the
/// default where timing ties and the one taken untimed (beta != 0). Measured on
/// an RTX 4080 Laptop over Nemotron's (N, K in 1024..4096) and Gemma 4's prefill
/// shapes, M 8..512, K split as `splitFor` does:
///   - tall: 64x64 wins most shapes, 64x64 two blocks deep and the other two
///     the rest; 128x64 and the 64x64-per-group 128x128 never won, nor did
///     64x64 four blocks deep (the slowest at M = 512).
///   - short: four SIMD groups of 16x16 shares win most shapes, staged four
///     blocks deep, and 16x64 two deep the rest; 32x64 blocks, 16x128, and two
///     SIMD groups of 16x32 four deep rarely did, and by no more than noise.
/// Dropped configs keep first-run tuning short (~25 ms per candidate per shape).
pub const configs = [_]Config{
    .{ .bm = 64, .bn = 64, .sgm = 32, .sgn = 32 },
    .{ .bm = 128, .bn = 128, .sgm = 64, .sgn = 32 },
    .{ .bm = 64, .bn = 128, .sgm = 32, .sgn = 64 },
    .{ .bm = 64, .bn = 64, .sgm = 32, .sgn = 32, .kb = 2 },
    .{ .bm = 16, .bn = 64, .sgm = 16, .sgn = 16, .kb = 4 },
    .{ .bm = 32, .bn = 32, .sgm = 16, .sgn = 16, .kb = 4 },
    .{ .bm = 16, .bn = 32, .sgm = 16, .sgn = 16, .kb = 4 },
    .{ .bm = 16, .bn = 64, .sgm = 16, .sgn = 32, .kb = 2 },
};

/// The product: `C[m, c_off + n] = alpha * A[m, k] @ B[n, k]^T + beta * C`, C
/// rows `c_row` apart.
pub const Params = struct { m: u32, n: u32, k: u32, c_row: u32, c_off: u32, alpha: f32 = 1.0, beta: f32 = 0.0 };

/// Field order matches `struct Params` in matmul_q8_mx16.wgsl.
const GemmParams = extern struct {
    m: u32,
    n: u32,
    k: u32,
    c_row: u32,
    c_off: u32,
    alpha: f32,
    beta: f32,
    kb_split: u32,
    c_split: u32,
    _p0: u32 = 0,
    _p1: u32 = 0,
    _p2: u32 = 0,
};

/// Field order matches `struct Params` in split_k_reduce.wgsl.
const ReduceParams = extern struct { m: u32, n: u32, splits: u32, c_row: u32, c_off: u32, alpha: f32, beta: f32, _p0: u32 = 0 };

const reduce_kernel: pipelines.KernelDesc = .{ .name = "split_k_reduce", .wgsl = @embedFile("../kernels/split_k_reduce.wgsl") };
const REDUCE_WG: u32 = 256;

/// Workgroups a grid should reach before K is split. A narrow N (1024 rows of B
/// in 32-64-row blocks) is a grid of a few dozen, each walking all of K a
/// global-load latency per stage, the rest of the GPU idle; splitting K fills it
/// instead, at the price of a partial per split and a reduce. Measured on an RTX
/// 4080 Laptop (N = 1024, K = 4096): M = 64 0.159 -> 0.036 ms, M = 128 0.160 ->
/// 0.055, M = 2 0.062 -> 0.023 (as fast as the GEMV); Gemma 4's down projection
/// (N = 1536, K = 6144) at M = 64 0.240 -> 0.062. 128 and 512 groups measured
/// about the same; no shape measured slower split.
const SPLIT_TARGET_GROUPS: u32 = 256;
/// q8 blocks of K a split walks at least.
const SPLIT_MIN_BLOCKS: u32 = 8;
const MAX_SPLITS: u32 = 16;

/// How `cfg` splits K for `p`: q8 blocks per split, and the number of splits.
fn splitFor(cfg: Config, p: Params) struct { kb: u32, count: u32 } {
    const groups = context.ceilDiv(p.n, cfg.bn) * context.ceilDiv(p.m, cfg.bm);
    const blocks = p.k / BK;
    var s: u32 = 1;
    while (s < MAX_SPLITS and groups * s < SPLIT_TARGET_GROUPS and blocks / (s * 2) >= SPLIT_MIN_BLOCKS) s *= 2;
    // Whole stages per split, then however many splits that takes.
    const kb = context.ceilDiv(context.ceilDiv(blocks, s), cfg.kb) * cfg.kb;
    return .{ .kb = kb, .count = context.ceilDiv(blocks, kb) };
}

/// Whether `cfg` takes the product on this device: B in whole blocks (the last
/// one slides back) and the block's shared memory granted. A tall block also
/// needs whole blocks of A and a C its tiles can store to directly: rows 16-byte
/// aligned, nothing to scale or accumulate (so slid-back blocks may write shared
/// elements twice, and a tuning trial can rerun the product on the real C). A
/// short block takes what the tall ones cannot: fewer than `TALL` rows, or such
/// a C.
fn configFits(cfg: Config, limits: wgpu.Limits, p: Params) bool {
    if (p.n < cfg.bn or cfg.sharedBytes() > limits.max_shared_bytes or cfg.threads() > limits.max_invocations) return false;
    const direct = p.c_row % 4 == 0 and p.c_off % 4 == 0 and p.alpha == 1.0 and p.beta == 0.0;
    if (!cfg.staged()) return direct and p.m >= cfg.bm;
    return p.m < TALL or !direct;
}

/// Whether some config takes `C = alpha * A @ B^T + beta * C` here: K in whole q8
/// blocks and B in whole `lanes32x16` groups.
pub fn fits(limits: wgpu.Limits, p: Params) bool {
    if (p.m == 0 or p.n % 32 != 0 or p.k % BK != 0) return false;
    for (configs) |cfg| if (configFits(cfg, limits, p)) return true;
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
            return configFits(configs[idx], t.ctx.gpu.limits, t.params);
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
        try ctx.tune.pick(ctx, frame, .of(.gemm_q8, .{ autotune.sizeClass(params.m), params.n, params.k }), configs.len, cand);
    return recordConfig(ctx, frame, idx orelse return error.Unsupported, bufs, sizes, params);
}

/// Record config `idx`: straight into C, or with K split, each split's [M, N]
/// partial into scratch and a reduce of them into C.
fn recordConfig(ctx: Ctx, frame: *Frame, idx: usize, bufs: [3]c.WGPUBuffer, sizes: [3]u64, p: Params) ExecuteProgramError!void {
    const cfg = configs[idx];
    const built = try ctx.pipes.get(kernels[idx], "gemm_q8");
    var split = splitFor(cfg, p);
    const part_bytes = @as(u64, split.count) * p.m * p.n * @sizeOf(f32);
    if (split.count > 1 and !context.storageBindingFits(ctx, part_bytes)) split = .{ .kb = p.k / BK, .count = 1 };
    const groups: [3]u32 = .{ context.ceilDiv(p.n, cfg.bn), context.ceilDiv(p.m, cfg.bm), split.count };

    if (split.count == 1) {
        const gp: GemmParams = .{ .m = p.m, .n = p.n, .k = p.k, .c_row = p.c_row, .c_off = p.c_off, .alpha = p.alpha, .beta = p.beta, .kb_split = split.kb, .c_split = 0 };
        return frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&gp), groups);
    }
    const part = try ctx.scratch.ensure(ctx.gpu, part_bytes);
    const gp: GemmParams = .{ .m = p.m, .n = p.n, .k = p.k, .c_row = p.n, .c_off = 0, .alpha = 1.0, .beta = 0.0, .kb_split = split.kb, .c_split = p.m * p.n };
    try frame.recordCompute(built, &.{ bufs[0], bufs[1], part }, &.{ sizes[0], sizes[1], part_bytes }, std.mem.asBytes(&gp), groups);

    const rp: ReduceParams = .{ .m = p.m, .n = p.n, .splits = split.count, .c_row = p.c_row, .c_off = p.c_off, .alpha = p.alpha, .beta = p.beta };
    const reduce = try ctx.pipes.get(reduce_kernel, "split_k_reduce");
    const reduce_groups = @max(1, @min(context.ceilDiv(p.m * p.n / 4, REDUCE_WG), context.MAX_GROUPS_1D));
    try frame.recordCompute(reduce, &.{ part, bufs[2] }, &.{ part_bytes, sizes[2] }, std.mem.asBytes(&rp), .{ reduce_groups, 1, 1 });
}

/// The pipeline name of `cfg`'s kernel, which tests look for.
pub fn kernelName(comptime cfg: Config) []const u8 {
    return std.fmt.comptimePrint("matmul_q8_mx16_{d}x{d}_{d}x{d}_k{d}", .{ cfg.bm, cfg.bn, cfg.sgm, cfg.sgn, cfg.kb });
}

const kernels = blk: {
    var out: [configs.len]pipelines.KernelDesc = undefined;
    for (configs, 0..) |cfg, i| out[i] = .{
        .name = kernelName(cfg),
        .wgsl = source(cfg),
    };
    break :blk out;
};

/// A staged config's way out: each element of `sc` no earlier block owns, scaled
/// and accumulated into C.
const staged_out = [_][]const u8{
    "workgroupBarrier();",
    "for (var e = t; e < BM * BN; e += THREADS) {",
    "    let mm = m0 + e / BN;",
    "    let nn = n0 + e % BN;",
    "    if (mm >= own_m && mm < p.m && nn >= own_n) {",
    "        let i = mm * c_row + c_off + nn;",
    "        let v = p.alpha * sc[e];",
    "        if (p.beta == 0.0) {",
    "            c[i] = v;",
    "        } else {",
    "            c[i] = v + p.beta * c[i];",
    "        }",
    "    }",
    "}",
};

fn source(comptime cfg: Config) []const u8 {
    @setEvalBranchQuota(10_000_000);
    if (cfg.bm % cfg.sgm != 0 or cfg.bn % cfg.sgn != 0) @compileError("a block is whole SIMD-group shares");
    if (cfg.sgm % TILE != 0 or cfg.sgn % TILE != 0) @compileError("a share is whole tiles");
    if (cfg.bn % 32 != 0) @compileError("B is staged in groups of 32 rows");
    if ((cfg.sk() * 2) % 16 != 0) @compileError("a staged row starts 16-byte aligned");
    const ti = cfg.sgm / TILE;
    const tj = cfg.sgn / TILE;
    var s: []const u8 = @embedFile("../kernels/matmul_q8_mx16.wgsl");
    s = template.fill(s, "consts", &.{
        std.fmt.comptimePrint("const BM: u32 = {d}u;", .{cfg.bm}),
        std.fmt.comptimePrint("const BN: u32 = {d}u;", .{cfg.bn}),
        std.fmt.comptimePrint("const SGM: u32 = {d}u;", .{cfg.sgm}),
        std.fmt.comptimePrint("const SGN: u32 = {d}u;", .{cfg.sgn}),
        std.fmt.comptimePrint("const THREADS: u32 = {d}u;", .{cfg.threads()}),
        std.fmt.comptimePrint("const KB: u32 = {d}u;", .{cfg.kb}),
        std.fmt.comptimePrint("const SK: u32 = {d}u;", .{cfg.sk()}),
        if (cfg.staged()) "var<workgroup> sc: array<f32, BM * BN>;" else "",
    });
    var acc: []const []const u8 = &.{};
    var store: []const []const u8 = &.{};
    var tiles: []const []const u8 = &.{};
    for (0..ti) |i| for (0..tj) |j| {
        acc = acc ++ .{std.fmt.comptimePrint("var c{d}{d} = coop_mat16x16<f32, C>();", .{ i, j })};
        if (cfg.staged()) {
            // The stride is the constant itself, never a `let` bound to it
            // (see the kernel on gfx-rs/wgpu#10901).
            store = store ++ .{std.fmt.comptimePrint(
                "coopStoreT(c{d}{d}, &sc[(sm + {d}u) * BN + sn + {d}u], BN);",
                .{ i, j, i * TILE, j * TILE },
            )};
        } else {
            store = store ++ .{std.fmt.comptimePrint(
                "let at{d}{d} = (m0 + sm + {d}u) * c_row + c_off + n0 + sn + {d}u;",
                .{ i, j, i * TILE, j * TILE },
            )};
            store = store ++ .{std.fmt.comptimePrint("coopStoreT(c{d}{d}, &c[at{d}{d}], c_row);", .{ i, j, i, j })};
        }
    };
    if (cfg.staged()) store = store ++ staged_out;
    for (0..2 * cfg.kb) |kk| {
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
