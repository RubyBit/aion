// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! WGSL generator for register-blocked f32 GEMM kernels — the CubeCL-style
//! "parameterized kernel" approach, emitted line-by-line through the `Wgsl` writer
//! (../wgsl.zig). Each generator function reads like the shader it produces: a
//! `w.line("acc_{d}_{d} += ...;", ...)` writes that WGSL with the config plugged in.
//!
//! The micro-tile is FULLY UNROLLED so each accumulator is a NAMED `vec4` with a
//! static swizzle — dynamically-indexed function-scope arrays get lowered to (slow)
//! addressable memory in WGSL/Naga, which silently kills register blocking. Unrolling
//! is just Zig `for` loops emitting lines; only the genuinely dynamic loops (the
//! K-slab loop and the >1-vec4-per-thread cooperative load) stay as WGSL `loop`s.
//!
//! Structure (the reusable seams for future kernels — attention, quantized matmul):
//!   header    — Params/bindings/shared decls
//!   preamble  — thread/tile index setup + accumulator declarations
//!   load      — cooperative global->shared staging (`coopLoadA`/`coopLoadB`)
//!   compute   — the register FMA inner loop (`computeBlock`)
//!   epilogue  — scaled store back to C (`writeOne`)
//!
//! Generation is runtime (arena) but the config is a comptime parameter to `emit`,
//! so `validate` still fires `@compileError` on a bad config.
//!
//! Computes C[i,j] = alpha * sum_k A[i,k]*B[k,j] + beta * C[i,j] over one tile.

const std = @import("std");
const pipelines = @import("../pipelines.zig");
const Wgsl = @import("../wgsl.zig").Wgsl;

/// Largest shared-memory footprint (bytes) we'll GENERATE a config for. The actual
/// per-device ceiling is enforced at runtime via `wgpu.Gpu.limits`; this bound just
/// keeps codegen sane. 48 KB covers NVIDIA/AMD; 16 KB-only devices skip bigger tiles.
pub const MAX_SHARED_BYTES: u32 = 49152;
pub const MAX_THREADS: u32 = 1024;

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

    pub fn threads(cfg: MatmulConfig) u32 {
        return (cfg.bm / cfg.tm) * (cfg.bn / cfg.tn);
    }
    /// vec4 output-column groups per thread (tn is a multiple of 4).
    fn vpr(cfg: MatmulConfig) u32 {
        return cfg.tn / 4;
    }
    /// A vec4s that must be staged into shared per K-slab, and B likewise.
    fn aVecs(cfg: MatmulConfig) u32 {
        return cfg.bm * cfg.bk / 4;
    }
    fn bVecs(cfg: MatmulConfig) u32 {
        return cfg.bk * cfg.bn / 4;
    }
    /// Prefetch registers a thread needs for A (double-buffer), rounded up.
    fn aRegs(cfg: MatmulConfig) u32 {
        return (cfg.aVecs() + cfg.threads() - 1) / cfg.threads();
    }
    fn bRegs(cfg: MatmulConfig) u32 {
        return (cfg.bVecs() + cfg.threads() - 1) / cfg.threads();
    }
};

fn validate(comptime cfg: MatmulConfig) void {
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
    return std.fmt.comptimePrint("mm_{d}x{d}x{d}_{d}x{d}_{s}{s}{s}", .{
        cfg.bm,                          cfg.bn,                               cfg.bk,                              cfg.tm, cfg.tn,
        if (cfg.vec4_load) "v" else "s", if (cfg.double_buffer) "_db" else "", if (cfg.bounds_check) "" else "_nb",
    });
}

const swiz = [4][]const u8{ "x", "y", "z", "w" };

// ---- global-load expressions (the one place layout/dtype detail lives) ------

/// WGSL for A's contiguous vec4 at (gr, gk). Fast path: 4 in-bounds elements.
fn aVec4(cfg: MatmulConfig) []const u8 {
    return if (cfg.vec4_load)
        "a[(gr * a_row + gk) / 4u]"
    else
        "vec4<f32>(a[gr * a_row + gk], a[gr * a_row + gk + 1u], a[gr * a_row + gk + 2u], a[gr * a_row + gk + 3u])";
}
fn bVec4(cfg: MatmulConfig) []const u8 {
    return if (cfg.vec4_load)
        "b[(gk * b_row + gc) / 4u]"
    else
        "vec4<f32>(b[gk * b_row + gc], b[gk * b_row + gc + 1u], b[gk * b_row + gc + 2u], b[gk * b_row + gc + 3u])";
}
/// WGSL for a single A scalar at column offset `off` (per-lane edge fallback).
fn aScalar(w: *Wgsl, cfg: MatmulConfig, off: u32) []const u8 {
    return if (cfg.vec4_load)
        w.fmt("a[(gr * a_row + gk + {d}u) / 4u][(gr * a_row + gk + {d}u) % 4u]", .{ off, off })
    else
        w.fmt("a[gr * a_row + gk + {d}u]", .{off});
}
fn bScalar(w: *Wgsl, cfg: MatmulConfig, off: u32) []const u8 {
    return if (cfg.vec4_load)
        w.fmt("b[(gk * b_row + gc + {d}u) / 4u][(gk * b_row + gc + {d}u) % 4u]", .{ off, off })
    else
        w.fmt("b[gk * b_row + gc + {d}u]", .{off});
}

// ---- module scaffolding ----------------------------------------------------

fn header(w: *Wgsl, cfg: MatmulConfig) void {
    w.lit("struct Params { dims: vec4<u32>, strides: vec4<u32>, ab: vec4<f32> };");
    const ab_ty = if (cfg.vec4_load) "array<vec4<f32>>" else "array<f32>";
    w.line("@group(0) @binding(0) var<storage, read> a: {s};", .{ab_ty});
    w.line("@group(0) @binding(1) var<storage, read> b: {s};", .{ab_ty});
    w.lit("@group(0) @binding(2) var<storage, read_write> cmat: array<f32>;");
    w.lit("@group(0) @binding(3) var<uniform> p: Params;");
    w.line("var<workgroup> As: array<vec4<f32>, {d}>;", .{cfg.aVecs()});
    w.line("var<workgroup> Bs: array<vec4<f32>, {d}>;", .{cfg.bVecs()});
    w.blank();
}

/// Thread/tile index setup + zeroed accumulator registers. As is transposed +
/// vec4-packed, so `rm/4u` addresses this thread's 4-row group; Bs is vec4-typed,
/// so `cn4` is this thread's column group.
fn preamble(w: *Wgsl, cfg: MatmulConfig) void {
    const cols = cfg.bn / cfg.tn;
    w.lit("let M = p.dims.x; let N = p.dims.y; let K = p.dims.z;");
    w.lit("let a_row = p.strides.x; let b_row = p.strides.y;");
    w.line("let block_row = wid.y * {d}u;", .{cfg.bm});
    w.line("let block_col = wid.x * {d}u;", .{cfg.bn});
    w.line("let thread_col = lidx % {d}u;", .{cols});
    w.line("let thread_row = lidx / {d}u;", .{cols});
    w.line("let rm = thread_row * {d}u;", .{cfg.tm});
    w.line("let cn = thread_col * {d}u;", .{cfg.tn});
    w.lit("let cn4 = cn / 4u;");
    for (0..cfg.tm) |i| {
        for (0..cfg.vpr()) |v| w.line("var acc_{d}_{d} = vec4<f32>(0.0);", .{ i, v });
    }
}

// ---- compute (inner K-slab accumulate over shared) -------------------------

fn computeBlock(w: *Wgsl, cfg: MatmulConfig) void {
    w.openBlock();
    w.lit("var dd = 0u;");
    w.open("loop", .{});
    w.open("if (dd >= {d}u)", .{cfg.bk});
    w.lit("break;");
    w.close();
    // One vec4 load per owned output-col group, and tm/4 contiguous A-row vec4s.
    for (0..cfg.vpr()) |v| w.line("let bvec{d} = Bs[dd * {d}u + cn4 + {d}u];", .{ v, cfg.bn / 4, v });
    for (0..cfg.tm / 4) |g| w.line("let av{d} = As[dd * {d}u + rm / 4u + {d}u];", .{ g, cfg.bm / 4, g });
    for (0..cfg.tm) |i| {
        for (0..cfg.vpr()) |v| w.line("acc_{d}_{d} += vec4<f32>(av{d}.{s}) * bvec{d};", .{ i, v, i / 4, swiz[i % 4], v });
    }
    w.lit("dd += 1u;");
    w.close(); // loop
    w.close(); // block
}

// ---- cooperative global->shared load (single-buffer, inline) ---------------

/// Stage this thread's vec4 of A (`vi`, slab base `k0`) into shared As, transposed
/// + vec4-packed: element (row, k4+lane) lands at As[(k4+lane)*(bm/4) + row/4][row%4].
fn coopLoadA(w: *Wgsl, cfg: MatmulConfig) void {
    const kpv = cfg.bk / 4;
    w.line("let row = vi / {d}u; let k4 = (vi % {d}u) * 4u;", .{ kpv, kpv });
    w.lit("let gr = block_row + row; let gk = k0 + k4;");
    w.lit("let rg = row / 4u; let cmp = row % 4u;");
    if (cfg.bounds_check) {
        w.open("if (gr < M && gk + 3u < K)", .{});
        storeARows(w, cfg);
        w.otherwise();
        storeARowsEdge(w, cfg);
        w.close();
    } else {
        storeARows(w, cfg);
    }
}
/// Fast path: whole vec4 in-bounds, one global read then 4 transposed stores.
fn storeARows(w: *Wgsl, cfg: MatmulConfig) void {
    w.line("let av = {s};", .{aVec4(cfg)});
    for (0..4) |l| w.line("As[(k4 + {d}u) * {d}u + rg][cmp] = av.{s};", .{ l, cfg.bm / 4, swiz[l] });
}
/// Edge path: per-lane bounds-checked scalar (zero-fill out of bounds).
fn storeARowsEdge(w: *Wgsl, cfg: MatmulConfig) void {
    for (0..4) |l| {
        w.open("if (gr < M && gk + {d}u < K)", .{l});
        w.line("As[(k4 + {d}u) * {d}u + rg][cmp] = {s};", .{ l, cfg.bm / 4, aScalar(w, cfg, @intCast(l)) });
        w.otherwise();
        w.line("As[(k4 + {d}u) * {d}u + rg][cmp] = 0.0;", .{ l, cfg.bm / 4 });
        w.close();
    }
}

/// Stage this thread's vec4 of B into shared Bs (vec4-typed, position == vi).
fn coopLoadB(w: *Wgsl, cfg: MatmulConfig) void {
    const cpv = cfg.bn / 4;
    w.line("let krow = vi / {d}u; let c4 = (vi % {d}u) * 4u;", .{ cpv, cpv });
    w.lit("let gk = k0 + krow; let gc = block_col + c4;");
    if (cfg.bounds_check) {
        w.open("if (gk < K && gc + 3u < N)", .{});
        w.line("Bs[vi] = {s};", .{bVec4(cfg)});
        w.otherwise();
        w.lit("var bv4 = vec4<f32>(0.0);");
        for (0..4) |l| {
            w.open("if (gk < K && gc + {d}u < N)", .{l});
            w.line("bv4.{s} = {s};", .{ swiz[l], bScalar(w, cfg, @intCast(l)) });
            w.close();
        }
        w.lit("Bs[vi] = bv4;");
        w.close();
    } else {
        w.line("Bs[vi] = {s};", .{bVec4(cfg)});
    }
}

/// Emit `body` so all `count` vec4s get staged by `nt` threads. Straight-line when
/// count == nt (a 1-iteration loop is not elided by Naga and halves throughput); a
/// single guarded pass when count < nt; else a strided loop.
fn loadWrap(w: *Wgsl, cfg: MatmulConfig, count: u32, comptime body: fn (*Wgsl, MatmulConfig) void) void {
    const nt = cfg.threads();
    if (count == nt) {
        w.openBlock();
        w.lit("let vi = lidx;");
        body(w, cfg);
        w.close();
    } else if (count < nt) {
        w.open("if (lidx < {d}u)", .{count});
        w.lit("let vi = lidx;");
        body(w, cfg);
        w.close();
    } else {
        w.openBlock();
        w.lit("var vi = lidx;");
        w.open("loop", .{});
        w.open("if (vi >= {d}u)", .{count});
        w.lit("break;");
        w.close();
        body(w, cfg);
        w.line("vi += {d}u;", .{nt});
        w.close();
        w.close();
    }
}

// ---- K loops ---------------------------------------------------------------

/// Single-buffer: per K-slab, stage A+B into shared, barrier, accumulate, barrier.
fn kLoopSingle(w: *Wgsl, cfg: MatmulConfig) void {
    w.lit("var k0 = 0u;");
    w.open("loop", .{});
    w.open("if (k0 >= K)", .{});
    w.lit("break;");
    w.close();
    loadWrap(w, cfg, cfg.aVecs(), coopLoadA);
    loadWrap(w, cfg, cfg.bVecs(), coopLoadB);
    w.lit("workgroupBarrier();");
    computeBlock(w, cfg);
    w.line("workgroupBarrier(); k0 += {d}u;", .{cfg.bk});
    w.close();
}

/// Double-buffer: a software pipeline that prefetches the next slab into registers
/// while computing the one already staged in shared, hiding global-load latency.
fn kLoopDouble(w: *Wgsl, cfg: MatmulConfig) void {
    for (0..cfg.aRegs()) |j| w.line("var apre{d} = vec4<f32>(0.0);", .{j});
    for (0..cfg.bRegs()) |j| w.line("var bpre{d} = vec4<f32>(0.0);", .{j});
    // Prologue: prefetch slab 0 into registers, store to shared.
    prefetch(w, cfg, "0u");
    storeShared(w, cfg);
    w.lit("workgroupBarrier();");
    w.lit("var slab = 1u;");
    w.open("loop", .{});
    w.open("if (slab * {d}u >= K)", .{cfg.bk});
    w.lit("break;");
    w.close();
    prefetch(w, cfg, w.fmt("slab * {d}u", .{cfg.bk}));
    computeBlock(w, cfg); // compute the staged slab while the next is in registers
    w.lit("workgroupBarrier();");
    storeShared(w, cfg);
    w.lit("workgroupBarrier();");
    w.lit("slab += 1u;");
    w.close();
    computeBlock(w, cfg); // epilogue: compute the last staged slab
}

/// Prefetch A and B vec4s for the slab based at `k_expr` into apre*/bpre* registers.
fn prefetch(w: *Wgsl, cfg: MatmulConfig, k_expr: []const u8) void {
    const nt = cfg.threads();
    const kpv = cfg.bk / 4;
    const cpv = cfg.bn / 4;
    for (0..cfg.aRegs()) |j| {
        w.openBlock();
        w.line("let vj = lidx + {d}u;", .{j * nt});
        w.open("if (vj < {d}u)", .{cfg.aVecs()});
        w.line("let row = vj / {d}u; let k4 = (vj % {d}u) * 4u;", .{ kpv, kpv });
        w.line("let gr = block_row + row; let gk = {s} + k4;", .{k_expr});
        prefetchOne(w, cfg, w.fmt("apre{d}", .{j}), aVec4(cfg), .a);
        w.otherwise();
        w.line("apre{d} = vec4<f32>(0.0);", .{j});
        w.close();
        w.close();
    }
    for (0..cfg.bRegs()) |j| {
        w.openBlock();
        w.line("let vj = lidx + {d}u;", .{j * nt});
        w.open("if (vj < {d}u)", .{cfg.bVecs()});
        w.line("let krow = vj / {d}u; let c4 = (vj % {d}u) * 4u;", .{ cpv, cpv });
        w.line("let gk = {s} + krow; let gc = block_col + c4;", .{k_expr});
        prefetchOne(w, cfg, w.fmt("bpre{d}", .{j}), bVec4(cfg), .b);
        w.otherwise();
        w.line("bpre{d} = vec4<f32>(0.0);", .{j});
        w.close();
        w.close();
    }
}

/// Load one prefetch register: fast contiguous vec4, or (bounds-checked) a per-lane
/// zero-filled build. `which` picks A vs B bounds/scalars.
fn prefetchOne(w: *Wgsl, cfg: MatmulConfig, reg: []const u8, vec4_expr: []const u8, comptime which: enum { a, b }) void {
    if (!cfg.bounds_check) {
        w.line("{s} = {s};", .{ reg, vec4_expr });
        return;
    }
    const guard = if (which == .a) "gr < M && gk + 3u < K" else "gk < K && gc + 3u < N";
    w.open("if ({s})", .{guard});
    w.line("{s} = {s};", .{ reg, vec4_expr });
    w.otherwise();
    w.lit("var tmp = vec4<f32>(0.0);");
    for (0..4) |l| {
        const lane_guard = if (which == .a) w.fmt("gr < M && gk + {d}u < K", .{l}) else w.fmt("gk < K && gc + {d}u < N", .{l});
        const scalar = if (which == .a) aScalar(w, cfg, @intCast(l)) else bScalar(w, cfg, @intCast(l));
        w.open("if ({s})", .{lane_guard});
        w.line("tmp.{s} = {s};", .{ swiz[l], scalar });
        w.close();
    }
    w.line("{s} = tmp;", .{reg});
    w.close();
}

/// Store prefetched registers into shared (A transposed + vec4-packed, B vec4-typed).
fn storeShared(w: *Wgsl, cfg: MatmulConfig) void {
    const nt = cfg.threads();
    const kpv = cfg.bk / 4;
    for (0..cfg.aRegs()) |j| {
        w.openBlock();
        w.line("let vj = lidx + {d}u;", .{j * nt});
        w.open("if (vj < {d}u)", .{cfg.aVecs()});
        w.line("let row = vj / {d}u; let k4 = (vj % {d}u) * 4u;", .{ kpv, kpv });
        w.lit("let rg = row / 4u; let cmp = row % 4u;");
        for (0..4) |l| w.line("As[(k4 + {d}u) * {d}u + rg][cmp] = apre{d}.{s};", .{ l, cfg.bm / 4, j, swiz[l] });
        w.close();
        w.close();
    }
    for (0..cfg.bRegs()) |j| {
        w.openBlock();
        w.line("let vj = lidx + {d}u;", .{j * nt});
        w.open("if (vj < {d}u)", .{cfg.bVecs()});
        w.line("Bs[vj] = bpre{d};", .{j});
        w.close();
        w.close();
    }
}

// ---- epilogue --------------------------------------------------------------

fn epilogue(w: *Wgsl, cfg: MatmulConfig) void {
    w.lit("let row0 = block_row + rm; let col0 = block_col + cn;");
    for (0..cfg.tm) |i| {
        for (0..cfg.vpr()) |v| {
            for (0..4) |j| w.line("writeOne(row0 + {d}u, col0 + {d}u, acc_{d}_{d}.{s});", .{ i, v * 4 + j, i, v, swiz[j] });
        }
    }
}

/// The scaled store helper: `C = alpha*value` (beta==0) or `alpha*value + beta*C`.
fn writeOneFn(w: *Wgsl, cfg: MatmulConfig) void {
    w.open("fn writeOne(r: u32, c: u32, value: f32)", .{});
    if (cfg.bounds_check) {
        w.open("if (r >= p.dims.x || c >= p.dims.y)", .{});
        w.lit("return;");
        w.close();
    }
    w.lit("let idx = r * p.strides.z + c;");
    w.open("if (p.ab.y == 0.0)", .{});
    w.lit("cmat[idx] = p.ab.x * value;");
    w.otherwise();
    w.lit("cmat[idx] = p.ab.x * value + p.ab.y * cmat[idx];");
    w.close();
    w.close();
}

// ---- entry points ----------------------------------------------------------

/// Render the WGSL for `cfg` into `arena`.
pub fn emit(arena: std.mem.Allocator, comptime cfg: MatmulConfig) [:0]const u8 {
    comptime validate(cfg);
    var w = Wgsl.init(arena);
    header(&w, cfg);
    w.line("@compute @workgroup_size({d})", .{cfg.threads()});
    w.open("fn {s}(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32)", .{entryName(cfg)});
    preamble(&w, cfg);
    if (cfg.double_buffer) kLoopDouble(&w, cfg) else kLoopSingle(&w, cfg);
    epilogue(&w, cfg);
    w.close();
    w.blank();
    writeOneFn(&w, cfg);
    return w.finish();
}

pub const Generated = struct {
    desc: pipelines.KernelDesc,
    entry: [:0]const u8,
    cfg: MatmulConfig,
};

/// Build the generated-kernel record for `cfg`, rendering its WGSL into `arena`.
pub fn gen(arena: std.mem.Allocator, comptime cfg: MatmulConfig) Generated {
    return .{ .desc = .{ .name = entryName(cfg), .wgsl = emit(arena, cfg) }, .entry = entryName(cfg), .cfg = cfg };
}

// The config MENU lives in configs.zig — tuning policy, kept separate from this
// codegen mechanism. The backend renders each config's WGSL once at init.
