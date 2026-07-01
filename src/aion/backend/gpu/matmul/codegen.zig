// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Comptime WGSL generator for register-blocked f32 GEMM kernels — the
//! CubeCL-style "parameterized kernel" approach, in Zig comptime.
//!
//! `emit(cfg)` builds the WGSL source for a `MatmulConfig` (block dims, micro-tile,
//! K-slab, vec4 loads, and optional double-buffering). The body is FULLY UNROLLED
//! over the micro-tile so the per-thread accumulators are NAMED `vec4`s with static
//! swizzles — dynamically-indexed function-scope arrays get lowered to (slow)
//! addressable memory in WGSL/Naga, which silently kills register blocking, so
//! codegen lets us unroll any tile size while keeping everything in registers.
//!
//! Each config compiles to its own shader module with a UNIQUE entry-point name
//! (the pipeline cache keys pipelines by entry name), so the backend holds a menu
//! of configs and autotunes per shape.
//!
//! Computes C[i,j] = alpha * sum_k A[i,k]*B[k,j] + beta * C[i,j] over one tile.

const std = @import("std");
const pipelines = @import("../pipelines.zig");

/// Largest shared-memory footprint (bytes) we'll GENERATE a config for. The actual
/// per-device ceiling is enforced at runtime via `wgpu.Gpu.limits` (a config over
/// the device's `maxComputeWorkgroupStorageSize` is excluded by the executor); this
/// comptime bound just keeps codegen sane. 48 KB covers NVIDIA/AMD; devices that
/// only grant the 16 KB spec minimum simply won't run the bigger configs.
pub const MAX_SHARED_BYTES: u32 = 49152;
pub const MAX_THREADS: u32 = 1024;

/// The subgroup width the subgroup kernel is specialized to. bn is fixed at
/// SUBGROUP_SIZE*4 so each of the 32 lanes owns exactly one vec4 output column
/// group. The executor only marks subgroup configs eligible on devices whose
/// probed subgroup width equals this.
pub const SUBGROUP_SIZE: u32 = 32;

/// Shared-memory bytes a config uses: As[bm*bk] + Bs[bk*bn], f32.
pub fn sharedBytes(cfg: MatmulConfig) u32 {
    return (cfg.bm * cfg.bk + cfg.bk * cfg.bn) * 4;
}

pub const MatmulConfig = struct {
    bm: u32, // C block rows (one workgroup computes bm×bn of C)
    bn: u32, // C block cols
    bk: u32, // K-slab depth staged through shared memory
    tm: u32, // micro-tile rows per thread
    tn: u32, // micro-tile cols per thread
    vec4_load: bool, // bind A/B as array<vec4<f32>> (true 128-bit loads)
    double_buffer: bool = false, // prefetch next K-slab into registers during compute
    bounds_check: bool = true, // guard edge tiles/partial K slabs

    // Subgroup outer-product kernel (see emitSubgroup). One subgroup (warp/wave)
    // computes a `tm`-row × bn-column strip: each of the 32 lanes owns one vec4
    // column group (so bn must be 128 = 32*4) and accumulates all `tm` rows,
    // sharing each A element across the subgroup via subgroupBroadcast. Requires a
    // device with the Subgroup feature and a FIXED subgroup width of 32; the
    // executor gates eligibility on that. tn is unused (implicitly 4).
    subgroup: bool = false,

    // 2D warp-tiling (CUTLASS / Boehm-kernel-10). When wm != 0, the block is split
    // into warp tiles of wm×wn, each warp iterating wn_iter sub-tiles in N (and a
    // derived number in M). tm×tn is then the per-thread, per-iteration micro-tile.
    wm: u32 = 0,
    wn: u32 = 0,
    wn_iter: u32 = 0,

    pub fn warpTiled(cfg: MatmulConfig) bool {
        return cfg.wm != 0;
    }

    pub fn threads(cfg: MatmulConfig) u32 {
        if (cfg.subgroup) return (cfg.bm / cfg.tm) * SUBGROUP_SIZE;
        if (cfg.warpTiled()) return (cfg.bm / cfg.wm) * (cfg.bn / cfg.wn) * 32;
        return (cfg.bm / cfg.tm) * (cfg.bn / cfg.tn);
    }

    // Warp-tiling derived quantities.
    pub fn wmIter(cfg: MatmulConfig) u32 {
        return (cfg.wm * cfg.wn) / (32 * cfg.tm * cfg.tn * cfg.wn_iter);
    }
    pub fn wsubM(cfg: MatmulConfig) u32 {
        return cfg.wm / cfg.wmIter();
    }
    pub fn wsubN(cfg: MatmulConfig) u32 {
        return cfg.wn / cfg.wn_iter;
    }
};

fn validateWarp(comptime cfg: MatmulConfig) void {
    if (!cfg.vec4_load) @compileError("warp-tiling requires vec4_load");
    if (cfg.tn != 4) @compileError("warp-tiling requires tn == 4");
    if (cfg.tm % 4 != 0) @compileError("warp tm must be a multiple of 4");
    if (cfg.bk % 4 != 0 or cfg.bn % 4 != 0 or cfg.bm % 4 != 0) @compileError("bk/bn/bm must be multiples of 4");
    if (cfg.wm == 0 or cfg.wn == 0 or cfg.wn_iter == 0) @compileError("warp config needs wm/wn/wn_iter");
    if (cfg.bm % cfg.wm != 0 or cfg.bn % cfg.wn != 0) @compileError("block not divisible by warp tile");
    if ((cfg.wm * cfg.wn) % (32 * cfg.tm * cfg.tn * cfg.wn_iter) != 0) @compileError("warp tile not divisible by warp work");
    const wmiter = cfg.wmIter();
    if (wmiter == 0 or cfg.wm % wmiter != 0) @compileError("bad wmIter");
    if (cfg.wsubM() % cfg.tm != 0 or cfg.wsubN() % cfg.tn != 0) @compileError("warp subtile not divisible by thread tile");
    if (cfg.wsubM() % 4 != 0 or cfg.wsubN() % 4 != 0) @compileError("warp subtile must be multiple of 4");
    if ((cfg.wsubM() / cfg.tm) * (cfg.wsubN() / cfg.tn) != 32) @compileError("threads-per-warp must equal 32");
    const nt = cfg.threads();
    if (nt == 0 or nt > MAX_THREADS) @compileError("warp thread count must be in 1..MAX_THREADS");
    if ((cfg.bm * cfg.bk) % nt != 0 or (cfg.bk * cfg.bn) % nt != 0) @compileError("load not divisible by threads");
    if (sharedBytes(cfg) > MAX_SHARED_BYTES) @compileError("shared memory exceeds MAX_SHARED_BYTES");
}

fn validateSubgroup(comptime cfg: MatmulConfig) void {
    if (!cfg.vec4_load) @compileError("subgroup kernel requires vec4_load");
    if (cfg.double_buffer) @compileError("subgroup kernel does not support double_buffer");
    if (cfg.bn % (SUBGROUP_SIZE * 4) != 0) @compileError("subgroup kernel requires bn to be a multiple of 128 (32 lanes * vec4)");
    if (cfg.tm == 0 or cfg.tm > SUBGROUP_SIZE) @compileError("subgroup tm must be in 1..32");
    if (cfg.bm % cfg.tm != 0) @compileError("bm % tm != 0");
    if (cfg.bm % 4 != 0) @compileError("bm must be a multiple of 4 (vec4 A packing)");
    if (cfg.bk % 4 != 0) @compileError("bk must be a multiple of 4 (vec4 loads)");
    const nt = cfg.threads();
    if (nt == 0 or nt > MAX_THREADS) @compileError("subgroup thread count must be in 1..MAX_THREADS");
    if (sharedBytes(cfg) > MAX_SHARED_BYTES) @compileError("shared memory exceeds MAX_SHARED_BYTES");
}

fn validate(comptime cfg: MatmulConfig) void {
    if (cfg.subgroup) return validateSubgroup(cfg);
    if (cfg.warpTiled()) return validateWarp(cfg);
    const nt = cfg.threads();
    if (cfg.bm % cfg.tm != 0) @compileError("bm % tm != 0");
    if (cfg.bn % cfg.tn != 0) @compileError("bn % tn != 0");
    if (cfg.tn % 4 != 0) @compileError("tn must be a multiple of 4 (vec4 accumulators)");
    if (cfg.tm % 4 != 0) @compileError("tm must be a multiple of 4 (vec4 A reads)");
    if (cfg.bm % 4 != 0) @compileError("bm must be a multiple of 4 (vec4 A packing)");
    if (cfg.bk % 4 != 0) @compileError("bk must be a multiple of 4 (vec4 loads)");
    if (cfg.bn % 4 != 0) @compileError("bn must be a multiple of 4 (vec4 loads)");
    if (nt == 0 or nt > MAX_THREADS) @compileError("threads (bm/tm * bn/tn) must be in 1..MAX_THREADS");
    if ((cfg.bm * cfg.bk) % nt != 0) @compileError("bm*bk not divisible by thread count");
    if ((cfg.bk * cfg.bn) % nt != 0) @compileError("bk*bn not divisible by thread count");
    if (sharedBytes(cfg) > MAX_SHARED_BYTES) @compileError("shared memory exceeds MAX_SHARED_BYTES");
}

pub fn entryName(comptime cfg: MatmulConfig) [:0]const u8 {
    if (cfg.subgroup) {
        return std.fmt.comptimePrint("mmsg_{d}x{d}x{d}_r{d}{s}{s}", .{
            cfg.bm, cfg.bn, cfg.bk, cfg.tm, if (cfg.double_buffer) "_db" else "", if (cfg.bounds_check) "" else "_nb",
        });
    }
    if (cfg.warpTiled()) {
        return std.fmt.comptimePrint("mmw_{d}x{d}x{d}_w{d}x{d}i{d}_{d}x{d}{s}", .{
            cfg.bm, cfg.bn, cfg.bk, cfg.wm, cfg.wn, cfg.wn_iter, cfg.tm, cfg.tn, if (cfg.bounds_check) "" else "_nb",
        });
    }
    return std.fmt.comptimePrint("mm_{d}x{d}x{d}_{d}x{d}_{s}{s}{s}", .{
        cfg.bm,                          cfg.bn,                               cfg.bk,                              cfg.tm, cfg.tn,
        if (cfg.vec4_load) "v" else "s", if (cfg.double_buffer) "_db" else "", if (cfg.bounds_check) "" else "_nb",
    });
}

const swiz = [4][]const u8{ "x", "y", "z", "w" };

// ---- shared sub-blocks (reused by both the single- and double-buffered paths) ----

/// The accumulate-the-current-shared-slab inner loop (reads As/Bs, FMAs into the
/// named acc_i_v vec4 registers).
fn computeBlock(comptime cfg: MatmulConfig) []const u8 {
    const bk = cfg.bk;
    const bm = cfg.bm;
    const bn = cfg.bn;
    const tm = cfg.tm;
    const vpr = cfg.tn / 4;
    var s: []const u8 = "";
    s = s ++ std.fmt.comptimePrint("    {{ var dd: u32 = 0u; loop {{ if (dd >= {d}u) {{ break; }}\n", .{bk});
    // Bs is array<vec4<f32>> ([k][col/4]); one real vec4 load per output-col group.
    for (0..vpr) |v| {
        s = s ++ std.fmt.comptimePrint("      let bvec{d} = Bs[dd * {d}u + cn4 + {d}u];\n", .{ v, bn / 4, v });
    }
    // A is stored transposed AND vec4-packed (As[k][row/4] = 4 rows): a thread's tm
    // rows for column dd are tm/4 contiguous vec4 reads (bank-conflict-free).
    for (0..tm / 4) |gg| {
        s = s ++ std.fmt.comptimePrint("      let av{d} = As[dd * {d}u + rm / 4u + {d}u];\n", .{ gg, bm / 4, gg });
    }
    for (0..tm) |i| {
        for (0..vpr) |v| {
            s = s ++ std.fmt.comptimePrint("      acc_{d}_{d} += vec4<f32>(av{d}.{s}) * bvec{d};\n", .{ i, v, i / 4, swiz[i % 4], v });
        }
    }
    s = s ++ "      dd += 1u; } }\n";
    return s;
}

/// One thread's vec4 load of A (the j-th vec4 it owns) from global into a `vec4`
/// expression — used by both the inline (single-buffer) and prefetch paths.
/// Emits the fast path when the contiguous 4 are in-bounds, else a per-lane build.
fn loadAExpr(comptime cfg: MatmulConfig) struct { fast: []const u8, lane: []const u8 } {
    if (cfg.vec4_load) {
        return .{ .fast = "a[(gr * a_row + gk) / 4u]", .lane = "a[oo / 4u][oo % 4u]" };
    } else {
        return .{ .fast = "vec4<f32>(a[bb + 0u], a[bb + 1u], a[bb + 2u], a[bb + 3u])", .lane = "a[oo]" };
    }
}
fn loadBExpr(comptime cfg: MatmulConfig) struct { fast: []const u8, lane: []const u8 } {
    if (cfg.vec4_load) {
        return .{ .fast = "b[(gk * b_row + gc) / 4u]", .lane = "b[oo / 4u][oo % 4u]" };
    } else {
        return .{ .fast = "vec4<f32>(b[bb + 0u], b[bb + 1u], b[bb + 2u], b[bb + 3u])", .lane = "b[oo]" };
    }
}

// ---- double-buffer prefetch/store blocks ----

fn declRegs(comptime cfg: MatmulConfig) []const u8 {
    const nt = cfg.threads();
    const na = (cfg.bm * cfg.bk / 4 + nt - 1) / nt;
    const nb = (cfg.bk * cfg.bn / 4 + nt - 1) / nt;
    var s: []const u8 = "";
    for (0..na) |j| s = s ++ std.fmt.comptimePrint("  var apre{d} = vec4<f32>(0.0);\n", .{j});
    for (0..nb) |j| s = s ++ std.fmt.comptimePrint("  var bpre{d} = vec4<f32>(0.0);\n", .{j});
    return s;
}

/// Prefetch this thread's A vec4s for the slab at `k_expr` into apre0.. registers.
fn prefetchA(comptime cfg: MatmulConfig, comptime k_expr: []const u8) []const u8 {
    const nt = cfg.threads();
    const a_vecs = cfg.bm * cfg.bk / 4;
    const na = (a_vecs + nt - 1) / nt;
    const kpv = cfg.bk / 4;
    const e = loadAExpr(cfg);
    var s: []const u8 = "";
    for (0..na) |j| {
        s = s ++ std.fmt.comptimePrint("    {{ let vj = lidx + {d}u;\n", .{j * nt});
        s = s ++ std.fmt.comptimePrint("      if (vj < {d}u) {{\n", .{a_vecs});
        s = s ++ std.fmt.comptimePrint("        let row = vj / {d}u; let k4 = (vj % {d}u) * 4u; let gr = block_row + row; let gk = ({s}) + k4;\n", .{ kpv, kpv, k_expr });
        if (!cfg.bounds_check) {
            s = s ++ std.fmt.comptimePrint("        let bb = gr * a_row + gk; apre{d} = {s};\n      }} else {{ apre{d} = vec4<f32>(0.0); }} }}\n", .{ j, e.fast, j });
            continue;
        }
        s = s ++ "        if (gr < M && gk + 3u < K) { let bb = gr * a_row + gk;\n";
        s = s ++ std.fmt.comptimePrint("          apre{d} = {s};\n", .{ j, e.fast });
        s = s ++ "        } else { var tmp = vec4<f32>(0.0);\n";
        for (0..4) |lane| {
            s = s ++ std.fmt.comptimePrint("          if (gr < M && gk + {d}u < K) {{ let oo = gr * a_row + gk + {d}u; tmp.{s} = {s}; }}\n", .{ lane, lane, swiz[lane], e.lane });
        }
        s = s ++ std.fmt.comptimePrint("          apre{d} = tmp;\n        }}\n      }} else {{ apre{d} = vec4<f32>(0.0); }} }}\n", .{ j, j });
    }
    return s;
}
fn storeA(comptime cfg: MatmulConfig) []const u8 {
    const nt = cfg.threads();
    const a_vecs = cfg.bm * cfg.bk / 4;
    const na = (a_vecs + nt - 1) / nt;
    const kpv = cfg.bk / 4;
    var s: []const u8 = "";
    for (0..na) |j| {
        s = s ++ std.fmt.comptimePrint("    {{ let vj = lidx + {d}u; if (vj < {d}u) {{\n", .{ j * nt, a_vecs });
        s = s ++ std.fmt.comptimePrint("      let row = vj / {d}u; let k4 = (vj % {d}u) * 4u; let rg = row / 4u; let cmp = row % 4u;\n", .{ kpv, kpv });
        // transposed vec4-packed store: As[(k4+lane)*(bm/4) + row/4][row%4]
        const bm4 = cfg.bm / 4;
        s = s ++ std.fmt.comptimePrint("      As[(k4 + 0u) * {d}u + rg][cmp] = apre{d}.x; As[(k4 + 1u) * {d}u + rg][cmp] = apre{d}.y; As[(k4 + 2u) * {d}u + rg][cmp] = apre{d}.z; As[(k4 + 3u) * {d}u + rg][cmp] = apre{d}.w;\n", .{ bm4, j, bm4, j, bm4, j, bm4, j });
        s = s ++ "    } }\n";
    }
    return s;
}
fn prefetchB(comptime cfg: MatmulConfig, comptime k_expr: []const u8) []const u8 {
    const nt = cfg.threads();
    const b_vecs = cfg.bk * cfg.bn / 4;
    const nb = (b_vecs + nt - 1) / nt;
    const cpv = cfg.bn / 4;
    const e = loadBExpr(cfg);
    var s: []const u8 = "";
    for (0..nb) |j| {
        s = s ++ std.fmt.comptimePrint("    {{ let vj = lidx + {d}u;\n", .{j * nt});
        s = s ++ std.fmt.comptimePrint("      if (vj < {d}u) {{\n", .{b_vecs});
        s = s ++ std.fmt.comptimePrint("        let krow = vj / {d}u; let c4 = (vj % {d}u) * 4u; let gk = ({s}) + krow; let gc = block_col + c4;\n", .{ cpv, cpv, k_expr });
        if (!cfg.bounds_check) {
            s = s ++ std.fmt.comptimePrint("        let bb = gk * b_row + gc; bpre{d} = {s};\n      }} else {{ bpre{d} = vec4<f32>(0.0); }} }}\n", .{ j, e.fast, j });
            continue;
        }
        s = s ++ "        if (gk < K && gc + 3u < N) { let bb = gk * b_row + gc;\n";
        s = s ++ std.fmt.comptimePrint("          bpre{d} = {s};\n", .{ j, e.fast });
        s = s ++ "        } else { var tmp = vec4<f32>(0.0);\n";
        for (0..4) |lane| {
            s = s ++ std.fmt.comptimePrint("          if (gk < K && gc + {d}u < N) {{ let oo = gk * b_row + gc + {d}u; tmp.{s} = {s}; }}\n", .{ lane, lane, swiz[lane], e.lane });
        }
        s = s ++ std.fmt.comptimePrint("          bpre{d} = tmp;\n        }}\n      }} else {{ bpre{d} = vec4<f32>(0.0); }} }}\n", .{ j, j });
    }
    return s;
}
fn storeB(comptime cfg: MatmulConfig) []const u8 {
    const nt = cfg.threads();
    const b_vecs = cfg.bk * cfg.bn / 4;
    const nb = (b_vecs + nt - 1) / nt;
    var s: []const u8 = "";
    for (0..nb) |j| {
        // Bs is vec4-typed and the vec position IS vj → one vec4 store.
        s = s ++ std.fmt.comptimePrint("    {{ let vj = lidx + {d}u; if (vj < {d}u) {{ Bs[vj] = bpre{d}; }} }}\n", .{ j * nt, b_vecs, j });
    }
    return s;
}

/// Cooperative-load body for A (uses `vi`, `k0`): loads a thread's vec4 of A from
/// global and stores it transposed + vec4-packed into shared As. Shared by the
/// thread-tiled and warp-tiled single-buffer paths.
fn aLoadBody(comptime cfg: MatmulConfig) []const u8 {
    const kpv = cfg.bk / 4;
    const bm4 = cfg.bm / 4;
    var body: []const u8 = "";
    body = body ++ std.fmt.comptimePrint("      let row = vi / {d}u; let k4 = (vi % {d}u) * 4u;\n", .{ kpv, kpv });
    body = body ++ "      let gr = block_row + row; let gk = k0 + k4;\n";
    const tstore = std.fmt.comptimePrint("        let rg = row / 4u; let cmp = row % 4u;\n        As[(k4 + 0u) * {d}u + rg][cmp] = av.x; As[(k4 + 1u) * {d}u + rg][cmp] = av.y; As[(k4 + 2u) * {d}u + rg][cmp] = av.z; As[(k4 + 3u) * {d}u + rg][cmp] = av.w;\n", .{ bm4, bm4, bm4, bm4 });
    const tlane = std.fmt.comptimePrint("      }} else {{\n        let rg = row / 4u; let cmp = row % 4u;\n        for (var j: u32 = 0u; j < 4u; j += 1u) {{ var v: f32 = 0.0; if (gr < M && gk + j < K) {{ {s} }} As[(k4 + j) * {d}u + rg][cmp] = v; }}\n      }}\n", .{ if (cfg.vec4_load) "let o = gr * a_row + gk + j; v = a[o / 4u][o % 4u];" else "v = a[gr * a_row + gk + j];", bm4 });
    if (cfg.bounds_check) body = body ++ "      if (gr < M && gk + 3u < K) {\n";
    if (cfg.vec4_load) {
        body = body ++ "        let av = a[(gr * a_row + gk) / 4u];\n" ++ tstore;
    } else {
        body = body ++ "        let bb = gr * a_row + gk; let av = vec4<f32>(a[bb + 0u], a[bb + 1u], a[bb + 2u], a[bb + 3u]);\n" ++ tstore;
    }
    if (cfg.bounds_check) body = body ++ tlane;
    return body;
}

/// Cooperative-load body for B (uses `vi`, `k0`): loads a thread's vec4 of B and
/// stores it to vec4-typed shared Bs (position == vi).
fn bLoadBody(comptime cfg: MatmulConfig) []const u8 {
    const cpv = cfg.bn / 4;
    var body: []const u8 = "";
    body = body ++ std.fmt.comptimePrint("      let krow = vi / {d}u; let c4 = (vi % {d}u) * 4u;\n", .{ cpv, cpv });
    body = body ++ "      let gk = k0 + krow; let gc = block_col + c4;\n";
    if (cfg.bounds_check) body = body ++ "      var bv4 = vec4<f32>(0.0);\n      if (gk < K && gc + 3u < N) {\n";
    if (cfg.vec4_load) {
        body = body ++ if (cfg.bounds_check) "        bv4 = b[(gk * b_row + gc) / 4u];\n" else "      let bv4 = b[(gk * b_row + gc) / 4u];\n";
    } else {
        body = body ++ if (cfg.bounds_check) "        let bb = gk * b_row + gc; bv4 = vec4<f32>(b[bb + 0u], b[bb + 1u], b[bb + 2u], b[bb + 3u]);\n" else "      let bb = gk * b_row + gc; let bv4 = vec4<f32>(b[bb + 0u], b[bb + 1u], b[bb + 2u], b[bb + 3u]);\n";
    }
    if (!cfg.bounds_check) return body ++ "      Bs[vi] = bv4;\n";
    body = body ++ "      } else {\n";
    for (0..4) |lane| {
        const ld = if (cfg.vec4_load)
            std.fmt.comptimePrint("let o = gk * b_row + gc + {d}u; bv4.{s} = b[o / 4u][o % 4u];", .{ lane, swiz[lane] })
        else
            std.fmt.comptimePrint("bv4.{s} = b[gk * b_row + gc + {d}u];", .{ swiz[lane], lane });
        body = body ++ std.fmt.comptimePrint("        if (gk < K && gc + {d}u < N) {{ {s} }}\n", .{ lane, ld });
    }
    body = body ++ "      }\n      Bs[vi] = bv4;\n";
    return body;
}

/// Single-buffer inline cooperative load (straight-line when each thread loads
/// exactly one vec4 — a single-iteration `loop {}` is not elided by Naga and
/// measurably halves throughput).
fn emitLoadWrap(comptime count: u32, comptime nt: u32, comptime body: []const u8) []const u8 {
    if (count == nt) {
        return "    { let vi = lidx;\n" ++ body ++ "    }\n";
    } else if (count < nt) {
        return std.fmt.comptimePrint("    if (lidx < {d}u) {{ let vi = lidx;\n", .{count}) ++ body ++ "    }\n";
    } else {
        return std.fmt.comptimePrint("    {{ var vi = lidx; loop {{ if (vi >= {d}u) {{ break; }}\n", .{count}) ++
            body ++ std.fmt.comptimePrint("      vi += {d}u; }} }}\n", .{nt});
    }
}

/// WGSL for the writeOne helper (one per module; modules are per-config).
fn writeOneFn(comptime cfg: MatmulConfig) []const u8 {
    var s: []const u8 = "fn writeOne(r: u32, c: u32, value: f32) {\n";
    if (cfg.bounds_check) s = s ++ "  if (r >= p.dims.x || c >= p.dims.y) { return; }\n";
    s = s ++ "  let idx = r * p.strides.z + c;\n";
    s = s ++ "  if (p.ab.y == 0.0) { cmat[idx] = p.ab.x * value; }\n" ++
        "  else { cmat[idx] = p.ab.x * value + p.ab.y * cmat[idx]; }\n";
    return s ++ "}\n";
}

/// Common WGSL header (struct + bindings + shared decls) for a config.
fn emitHeader(comptime cfg: MatmulConfig, comptime entry: []const u8, comptime nt: u32) []const u8 {
    var s: []const u8 = "// Generated by matmul_gen.zig — do not edit.\n";
    s = s ++ "struct Params { dims: vec4<u32>, strides: vec4<u32>, ab: vec4<f32> };\n";
    s = s ++ "@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;\n";
    s = s ++ "@group(0) @binding(1) var<storage, read> b: array<vec4<f32>>;\n";
    s = s ++ "@group(0) @binding(2) var<storage, read_write> cmat: array<f32>;\n";
    s = s ++ "@group(0) @binding(3) var<uniform> p: Params;\n";
    s = s ++ std.fmt.comptimePrint("var<workgroup> As: array<vec4<f32>, {d}>;\n", .{cfg.bm * cfg.bk / 4});
    s = s ++ std.fmt.comptimePrint("var<workgroup> Bs: array<vec4<f32>, {d}>;\n", .{cfg.bk * cfg.bn / 4});
    s = s ++ std.fmt.comptimePrint("@compute @workgroup_size({d})\n", .{nt});
    s = s ++ std.fmt.comptimePrint("fn {s}(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {{\n", .{entry});
    s = s ++ "  let M = p.dims.x; let N = p.dims.y; let K = p.dims.z;\n";
    s = s ++ "  let a_row = p.strides.x; let b_row = p.strides.y;\n";
    s = s ++ std.fmt.comptimePrint("  let block_row = wid.y * {d}u;\n", .{cfg.bm});
    s = s ++ std.fmt.comptimePrint("  let block_col = wid.x * {d}u;\n", .{cfg.bn});
    return s;
}

/// 2D warp-tiled kernel (CUTLASS / Boehm-kernel-10). Requires vec4 + tn==4.
fn emitWarp(comptime cfg: MatmulConfig) [:0]const u8 {
    @setEvalBranchQuota(12_000_000);
    comptime {
        validateWarp(cfg);
        const bm = cfg.bm;
        const bn = cfg.bn;
        const bk = cfg.bk;
        const tm = cfg.tm;
        const tn = cfg.tn;
        const wm = cfg.wm;
        const wn = cfg.wn;
        const wniter = cfg.wn_iter;
        const nt = cfg.threads();
        const wmiter = cfg.wmIter();
        const wsubM = cfg.wsubM();
        const wsubN = cfg.wsubN();
        const warpsN = bn / wn;
        const tpwN = wsubN / tn; // threads-per-warp along N
        const a_vecs = bm * bk / 4;
        const b_vecs = bk * bn / 4;
        const tmg = tm / 4; // vec4 groups of A rows per warp-M-iteration

        var s: []const u8 = emitHeader(cfg, entryName(cfg), nt);

        // warp + lane indices
        s = s ++ "  let warpIdx = lidx / 32u; let laneIdx = lidx % 32u;\n";
        s = s ++ std.fmt.comptimePrint("  let warpRow = warpIdx / {d}u; let warpCol = warpIdx % {d}u;\n", .{ warpsN, warpsN });
        s = s ++ std.fmt.comptimePrint("  let trw = laneIdx / {d}u; let tcw = laneIdx % {d}u;\n", .{ tpwN, tpwN });

        // per-iteration row/col bases within the block, and their /4 (vec4) forms.
        for (0..wmiter) |wi| {
            s = s ++ std.fmt.comptimePrint("  let rowbase{d} = warpRow * {d}u + {d}u + trw * {d}u;\n", .{ wi, wm, wi * wsubM, tm });
            s = s ++ std.fmt.comptimePrint("  let rb4_{d} = rowbase{d} / 4u;\n", .{ wi, wi });
        }
        for (0..wniter) |wj| {
            s = s ++ std.fmt.comptimePrint("  let colbase{d} = warpCol * {d}u + {d}u + tcw * {d}u;\n", .{ wj, wn, wj * wsubN, tn });
            s = s ++ std.fmt.comptimePrint("  let cb4_{d} = colbase{d} / 4u;\n", .{ wj, wj });
        }

        // accumulators acc_{wi}_{ti}_{wj} (vec4, tn==4)
        for (0..wmiter) |wi| {
            for (0..tm) |ti| {
                for (0..wniter) |wj| {
                    s = s ++ std.fmt.comptimePrint("  var acc_{d}_{d}_{d} = vec4<f32>(0.0);\n", .{ wi, ti, wj });
                }
            }
        }

        // K loop (single-buffer cooperative load, then warp compute).
        s = s ++ "  var k0: u32 = 0u;\n  loop {\n    if (k0 >= K) { break; }\n";
        s = s ++ emitLoadWrap(a_vecs, nt, aLoadBody(cfg));
        s = s ++ emitLoadWrap(b_vecs, nt, bLoadBody(cfg));
        s = s ++ "    workgroupBarrier();\n";
        s = s ++ std.fmt.comptimePrint("    {{ var dd: u32 = 0u; loop {{ if (dd >= {d}u) {{ break; }}\n", .{bk});
        for (0..wmiter) |wi| {
            for (0..tmg) |g| {
                s = s ++ std.fmt.comptimePrint("      let avA{d}_{d} = As[dd * {d}u + rb4_{d} + {d}u];\n", .{ wi, g, bm / 4, wi, g });
            }
        }
        for (0..wniter) |wj| {
            s = s ++ std.fmt.comptimePrint("      let bN{d} = Bs[dd * {d}u + cb4_{d}];\n", .{ wj, bn / 4, wj });
        }
        for (0..wmiter) |wi| {
            for (0..tm) |ti| {
                for (0..wniter) |wj| {
                    s = s ++ std.fmt.comptimePrint("      acc_{d}_{d}_{d} += vec4<f32>(avA{d}_{d}.{s}) * bN{d};\n", .{ wi, ti, wj, wi, ti / 4, swiz[ti % 4], wj });
                }
            }
        }
        s = s ++ "      dd += 1u; } }\n";
        s = s ++ std.fmt.comptimePrint("    workgroupBarrier(); k0 += {d}u;\n  }}\n", .{bk});

        // write back
        for (0..wmiter) |wi| {
            for (0..tm) |ti| {
                for (0..wniter) |wj| {
                    s = s ++ std.fmt.comptimePrint("  {{ let r = block_row + rowbase{d} + {d}u; let c0 = block_col + colbase{d};\n", .{ wi, ti, wj });
                    s = s ++ std.fmt.comptimePrint("    writeOne(r, c0 + 0u, acc_{d}_{d}_{d}.x); writeOne(r, c0 + 1u, acc_{d}_{d}_{d}.y); writeOne(r, c0 + 2u, acc_{d}_{d}_{d}.z); writeOne(r, c0 + 3u, acc_{d}_{d}_{d}.w); }}\n", .{ wi, ti, wj, wi, ti, wj, wi, ti, wj, wi, ti, wj });
                }
            }
        }
        s = s ++ "}\n" ++ writeOneFn(cfg);
        return s ++ "";
    }
}

/// Subgroup outer-product kernel. One subgroup (32 lanes) computes a `tm`-row ×
/// bn-column (bn == 128) strip of the block: lane L owns vec4 output column group
/// L (columns 4L..4L+3) and accumulates all `tm` rows. A and B slabs are staged in
/// shared (reusing the cooperative-load bodies); in the inner K-loop each lane
/// reads only its own B vec4 and one A element, then A[row=i] is shared across the
/// whole subgroup via subgroupBroadcast(myA, i) — cutting shared-memory A traffic
/// from tm reads/lane to ~1. Assumes subgroup_invocation_id == lidx % 32 and that
/// consecutive 32 lanes form one subgroup (true for 1D workgroups on this driver;
/// correctness is guarded by the CPU-vs-GPU test).
fn emitSubgroup(comptime cfg: MatmulConfig) [:0]const u8 {
    @setEvalBranchQuota(12_000_000);
    comptime {
        validateSubgroup(cfg);
        const bm = cfg.bm;
        const bn = cfg.bn;
        const bk = cfg.bk;
        const tm = cfg.tm;
        const nt = cfg.threads();
        const bm4 = bm / 4;
        const bn4 = bn / 4;
        const vpl = bn / (SUBGROUP_SIZE * 4); // vec4 output-column groups per lane
        const a_vecs = bm * bk / 4;
        const b_vecs = bk * bn / 4;
        const entry = entryName(cfg);

        var s: []const u8 = "// Generated by matmul_gen.zig (subgroup) — do not edit.\n";
        s = s ++ "struct Params { dims: vec4<u32>, strides: vec4<u32>, ab: vec4<f32> };\n";
        s = s ++ "@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;\n";
        s = s ++ "@group(0) @binding(1) var<storage, read> b: array<vec4<f32>>;\n";
        s = s ++ "@group(0) @binding(2) var<storage, read_write> cmat: array<f32>;\n";
        s = s ++ "@group(0) @binding(3) var<uniform> p: Params;\n";
        s = s ++ std.fmt.comptimePrint("var<workgroup> As: array<vec4<f32>, {d}>;\n", .{a_vecs});
        s = s ++ std.fmt.comptimePrint("var<workgroup> Bs: array<vec4<f32>, {d}>;\n", .{b_vecs});
        s = s ++ std.fmt.comptimePrint("@compute @workgroup_size({d})\n", .{nt});
        s = s ++ std.fmt.comptimePrint("fn {s}(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(subgroup_invocation_id) lane: u32) {{\n", .{entry});
        s = s ++ "  let M = p.dims.x; let N = p.dims.y; let K = p.dims.z;\n";
        s = s ++ "  let a_row = p.strides.x; let b_row = p.strides.y;\n";
        s = s ++ std.fmt.comptimePrint("  let block_row = wid.y * {d}u;\n", .{bm});
        s = s ++ std.fmt.comptimePrint("  let block_col = wid.x * {d}u;\n", .{bn});
        // Subgroup index within the workgroup and this subgroup's base row.
        s = s ++ std.fmt.comptimePrint("  let warp = lidx / {d}u;\n", .{SUBGROUP_SIZE});
        s = s ++ std.fmt.comptimePrint("  let sg_row = warp * {d}u;\n", .{tm});

        // Accumulators: one vec4 per (owned row, owned column group).
        for (0..tm) |i| {
            for (0..vpl) |v| s = s ++ std.fmt.comptimePrint("  var acc{d}_{d} = vec4<f32>(0.0);\n", .{ i, v });
        }

        // K loop: cooperative-load a slab into shared, barrier, subgroup compute.
        s = s ++ "  var k0: u32 = 0u;\n  loop {\n    if (k0 >= K) { break; }\n";
        s = s ++ emitLoadWrap(a_vecs, nt, aLoadBody(cfg));
        s = s ++ emitLoadWrap(b_vecs, nt, bLoadBody(cfg));
        s = s ++ "    workgroupBarrier();\n";
        s = s ++ std.fmt.comptimePrint("    {{ var dd: u32 = 0u; loop {{ if (dd >= {d}u) {{ break; }}\n", .{bk});
        // This lane's B vec4s (its owned contiguous column groups) for depth dd.
        for (0..vpl) |v| {
            s = s ++ std.fmt.comptimePrint("      let myB{d} = Bs[dd * {d}u + lane * {d}u + {d}u];\n", .{ v, bn4, vpl, v });
        }
        // This lane's single A element: row (sg_row + lane) at depth dd, if it owns one.
        s = s ++ std.fmt.comptimePrint("      var myA = 0.0; if (lane < {d}u) {{ let r = sg_row + lane; myA = As[dd * {d}u + r / 4u][r % 4u]; }}\n", .{ tm, bm4 });
        // Broadcast A[row=i] across the subgroup; FMA into each owned column group.
        for (0..tm) |i| {
            s = s ++ std.fmt.comptimePrint("      let a{d} = vec4<f32>(subgroupBroadcast(myA, {d}u));\n", .{ i, i });
            for (0..vpl) |v| {
                s = s ++ std.fmt.comptimePrint("      acc{d}_{d} += a{d} * myB{d};\n", .{ i, v, i, v });
            }
        }
        s = s ++ "      dd += 1u; } }\n";
        s = s ++ std.fmt.comptimePrint("    workgroupBarrier(); k0 += {d}u;\n  }}\n", .{bk});

        // Write back: row = block_row + sg_row + i; this lane owns contiguous columns
        // starting at block_col + lane*(vpl*4), one vec4 per owned column group.
        s = s ++ std.fmt.comptimePrint("  let col0 = block_col + lane * {d}u;\n", .{vpl * 4});
        for (0..tm) |i| {
            s = s ++ std.fmt.comptimePrint("  {{ let r = block_row + sg_row + {d}u;\n", .{i});
            for (0..vpl) |v| {
                s = s ++ std.fmt.comptimePrint("    writeOne(r, col0 + {d}u, acc{d}_{d}.x); writeOne(r, col0 + {d}u, acc{d}_{d}.y); writeOne(r, col0 + {d}u, acc{d}_{d}.z); writeOne(r, col0 + {d}u, acc{d}_{d}.w);\n", .{ v * 4 + 0, i, v, v * 4 + 1, i, v, v * 4 + 2, i, v, v * 4 + 3, i, v });
            }
            s = s ++ "  }\n";
        }
        s = s ++ "}\n" ++ writeOneFn(cfg);
        return s ++ "";
    }
}

pub fn emit(comptime cfg: MatmulConfig) [:0]const u8 {
    @setEvalBranchQuota(12_000_000);
    if (cfg.subgroup) return emitSubgroup(cfg);
    if (cfg.warpTiled()) return emitWarp(cfg);
    comptime {
        validate(cfg);
        const bm = cfg.bm;
        const bn = cfg.bn;
        const bk = cfg.bk;
        const tm = cfg.tm;
        const tn = cfg.tn;
        const nt = cfg.threads();
        const cols = bn / tn;
        const vpr = tn / 4;
        const a_vecs = bm * bk / 4;
        const b_vecs = bk * bn / 4;
        const entry = entryName(cfg);

        var s: []const u8 = "";
        s = s ++ "// Generated by matmul_gen.zig — do not edit.\n";
        s = s ++ "struct Params { dims: vec4<u32>, strides: vec4<u32>, ab: vec4<f32> };\n";
        if (cfg.vec4_load) {
            s = s ++ "@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;\n";
            s = s ++ "@group(0) @binding(1) var<storage, read> b: array<vec4<f32>>;\n";
        } else {
            s = s ++ "@group(0) @binding(0) var<storage, read> a: array<f32>;\n";
            s = s ++ "@group(0) @binding(1) var<storage, read> b: array<f32>;\n";
        }
        s = s ++ "@group(0) @binding(2) var<storage, read_write> cmat: array<f32>;\n";
        s = s ++ "@group(0) @binding(3) var<uniform> p: Params;\n";
        s = s ++ std.fmt.comptimePrint("var<workgroup> As: array<vec4<f32>, {d}>;\n", .{bm * bk / 4});
        s = s ++ std.fmt.comptimePrint("var<workgroup> Bs: array<vec4<f32>, {d}>;\n", .{bk * bn / 4});
        s = s ++ std.fmt.comptimePrint("@compute @workgroup_size({d})\n", .{nt});
        s = s ++ std.fmt.comptimePrint("fn {s}(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {{\n", .{entry});

        s = s ++ "  let M = p.dims.x; let N = p.dims.y; let K = p.dims.z;\n";
        s = s ++ "  let a_row = p.strides.x; let b_row = p.strides.y;\n";
        s = s ++ std.fmt.comptimePrint("  let block_row = wid.y * {d}u;\n", .{bm});
        s = s ++ std.fmt.comptimePrint("  let block_col = wid.x * {d}u;\n", .{bn});
        s = s ++ std.fmt.comptimePrint("  let thread_col = lidx % {d}u;\n", .{cols});
        s = s ++ std.fmt.comptimePrint("  let thread_row = lidx / {d}u;\n", .{cols});
        s = s ++ std.fmt.comptimePrint("  let rm = thread_row * {d}u;\n", .{tm});
        s = s ++ std.fmt.comptimePrint("  let cn = thread_col * {d}u;\n", .{tn});
        s = s ++ "  let cn4 = cn / 4u;\n"; // cn in vec4 units (Bs is vec4-typed)

        for (0..tm) |i| {
            for (0..vpr) |v| {
                s = s ++ std.fmt.comptimePrint("  var acc_{d}_{d} = vec4<f32>(0.0);\n", .{ i, v });
            }
        }

        if (cfg.double_buffer) {
            // Register-prefetch software pipeline: prologue loads slab 0, then each
            // loop iteration prefetches the next slab into registers while computing
            // the slab already staged in shared, hiding global-load latency.
            s = s ++ declRegs(cfg);
            s = s ++ prefetchA(cfg, "0u");
            s = s ++ prefetchB(cfg, "0u");
            s = s ++ storeA(cfg);
            s = s ++ storeB(cfg);
            s = s ++ "  workgroupBarrier();\n";
            s = s ++ std.fmt.comptimePrint("  var slab: u32 = 1u;\n  loop {{ if (slab * {d}u >= K) {{ break; }}\n", .{bk});
            s = s ++ prefetchA(cfg, std.fmt.comptimePrint("slab * {d}u", .{bk}));
            s = s ++ prefetchB(cfg, std.fmt.comptimePrint("slab * {d}u", .{bk}));
            s = s ++ computeBlock(cfg);
            s = s ++ "    workgroupBarrier();\n";
            s = s ++ storeA(cfg);
            s = s ++ storeB(cfg);
            s = s ++ "    workgroupBarrier();\n    slab += 1u;\n  }\n";
            s = s ++ computeBlock(cfg); // epilogue: compute last slab in shared
        } else {
            // Single-buffer: stage slab → barrier → compute → barrier, per K-slab.
            s = s ++ "  var k0: u32 = 0u;\n  loop {\n    if (k0 >= K) { break; }\n";
            s = s ++ emitLoadWrap(a_vecs, nt, aLoadBody(cfg));
            s = s ++ emitLoadWrap(b_vecs, nt, bLoadBody(cfg));
            s = s ++ "    workgroupBarrier();\n";
            s = s ++ computeBlock(cfg);
            s = s ++ std.fmt.comptimePrint("    workgroupBarrier(); k0 += {d}u;\n  }}\n", .{bk});
        }

        // --- write back ---
        s = s ++ "  let row0 = block_row + rm; let col0 = block_col + cn;\n";
        for (0..tm) |i| {
            for (0..vpr) |v| {
                for (0..4) |j| {
                    s = s ++ std.fmt.comptimePrint("  writeOne(row0 + {d}u, col0 + {d}u, acc_{d}_{d}.{s});\n", .{ i, v * 4 + j, i, v, swiz[j] });
                }
            }
        }
        s = s ++ "}\n";

        s = s ++ writeOneFn(cfg);

        return s ++ "";
    }
}

pub const Generated = struct {
    desc: pipelines.KernelDesc,
    entry: [:0]const u8,
    cfg: MatmulConfig,
};

pub fn gen(comptime cfg: MatmulConfig) Generated {
    return .{ .desc = .{ .name = entryName(cfg), .wgsl = emit(cfg) }, .entry = entryName(cfg), .cfg = cfg };
}

// The config MENU and the comptime `generated[]` array it builds (via `gen`) live
// in configs.zig — tuning policy, kept separate from this codegen mechanism.
