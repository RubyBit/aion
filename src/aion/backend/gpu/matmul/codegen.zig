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
//! With `b_rows`, B is stored as its rows instead, `[N, K]` (a `MatMulNT`'s), and
//! the staging transposes it: the tile in shared memory, and everything after it,
//! is the same.
//!
//! A `matrix` config is the same product on the device's matrix units (wgpu's
//! cooperative matrices): whole 8x8 tiles are loaded straight from A and B and
//! multiplied as a unit, so it has no staging and no register tile (`matrixBody`).
//!
//! `matrixBody`'s tile layout depends on the wgpu version: it relies on naga 29's
//! Metal writer inverting the load/store layout (`coopLoad`/`coopStore` row-major,
//! `coopLoadT` column-major). That is an upstream bug, fixed in wgpu 30
//! (gfx-rs/wgpu#9451), after which the spec meaning holds on every backend
//! (`coopLoad` column-major, `coopLoadT` row-major). Moving wgpu-native to >= 30
//! must swap every `coopLoad` <-> `coopLoadT` and `coopStore` <-> `coopStoreT` it
//! emits, checked on the M5: a wrong layout transposes the square tiles silently.

const std = @import("std");
const pipelines = @import("../pipelines.zig");
const Wgsl = @import("../wgsl.zig").Wgsl;

/// Largest shared-memory footprint (bytes) we'll GENERATE a config for. The actual
/// per-device ceiling is enforced at runtime via `wgpu.Gpu.limits`; this bound just
/// keeps codegen sane. 48 KB covers NVIDIA/AMD; 16 KB-only devices skip bigger tiles.
pub const MAX_SHARED_BYTES: u32 = 49152;
pub const MAX_THREADS: u32 = 1024;

/// Rows, columns and inner depth of a matrix tile: the one size wgpu multiplies.
pub const TILE: u32 = 8;
/// Invocations in a matrix kernel's workgroup: one SIMD group, which is what
/// loads, multiplies and stores a tile together.
const MATRIX_GROUP: u32 = 32;
const tile_a = "coop_mat8x8<f32, A>";
const tile_b = "coop_mat8x8<f32, B>";
const tile_c = "coop_mat8x8<f32, C>";

/// Shared-memory bytes a config uses: As[bm*bk] + Bs[bk*bn], f32. A matrix config
/// holds the finished block instead.
pub fn sharedBytes(cfg: MatmulConfig) u32 {
    if (cfg.matrix) return cfg.bm * cfg.bn * 4;
    return (cfg.bm * cfg.bk + cfg.bk * cfg.bn) * 4;
}

/// Whether `cfg` computes an `[m, k] @ [k, n]` product. A config without edge
/// guards needs whole blocks; a matrix config needs whole tiles of K and one
/// block each way, since its edge blocks slide back inside the matrices.
pub fn fits(cfg: MatmulConfig, m: usize, n: usize, k: usize) bool {
    if (cfg.matrix) return m >= cfg.bm and n >= cfg.bn and k % TILE == 0;
    if (cfg.bounds_check) return true;
    return m % cfg.bm == 0 and n % cfg.bn == 0 and k % cfg.bk == 0;
}

/// Where the kernel's A operand comes from.
///
/// `conv` is this same register-blocked GEMM over an A it never materializes:
/// A[m, k] is the im2col of the activation, gathered in the cooperative load.
/// A conv weight is already `[kh*kw*c_in, c_out]` row-major, so B is untouched.
pub const Kind = enum { gemm, conv };

pub const MatmulConfig = struct {
    kind: Kind = .gemm,
    bm: u32, // C block rows (one workgroup computes bm×bn of C)
    bn: u32, // C block cols
    bk: u32, // K-slab depth staged through shared memory
    tm: u32, // micro-tile rows per thread
    tn: u32, // micro-tile cols per thread
    vec4_load: bool, // bind A/B as array<vec4<f32>> (true 128-bit loads)
    double_buffer: bool = false, // prefetch next K-slab into registers during compute
    bounds_check: bool = true, // guard edge tiles/partial K slabs
    b_rows: bool = false, // B is [N, K]: staged 4 rows x 4 k at a time, transposed in registers
    /// Accumulate 8x8 tiles on the device's matrix units instead of in per-thread
    /// registers. The tile is then the only unit (`tm`, `tn` and `bk` are 8) and
    /// A and B are read in place, so the load options do not apply.
    matrix: bool = false,

    /// A matrix config computing `bm x bn` of C per workgroup; `of` says what of.
    pub fn matrixTiles(bm: u32, bn: u32, of: struct { kind: Kind = .gemm, b_rows: bool = false }) MatmulConfig {
        return .{ .kind = of.kind, .bm = bm, .bn = bn, .bk = TILE, .tm = TILE, .tn = TILE, .vec4_load = false, .b_rows = of.b_rows, .matrix = true };
    }

    pub fn threads(cfg: MatmulConfig) u32 {
        if (cfg.matrix) return MATRIX_GROUP;
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
        return (cfg.bUnits() + cfg.threads() - 1) / cfg.threads();
    }
    /// What one thread stages of B per step: a vec4 of a row of `[K, N]`, or for
    /// `b_rows` a 4x4 block (four rows of `[N, K]`, four k each), so shared memory
    /// still takes whole vec4s.
    fn bUnits(cfg: MatmulConfig) u32 {
        return if (cfg.b_rows) cfg.bVecs() / 4 else cfg.bVecs();
    }
};

fn validate(comptime cfg: MatmulConfig) void {
    if (cfg.b_rows and cfg.kind == .conv) @compileError("a conv's B is its weight, [K, N]");
    if (cfg.matrix) {
        if (cfg.bm % TILE != 0 or cfg.bn % TILE != 0) @compileError("a matrix block is whole tiles");
        if (cfg.tm != TILE or cfg.tn != TILE or cfg.bk != TILE) @compileError("a matrix config's tile is 8x8x8");
        if (cfg.vec4_load or cfg.double_buffer or !cfg.bounds_check) @compileError("a matrix config stages nothing and needs no guards");
        if ((cfg.bm * cfg.bn) % MATRIX_GROUP != 0) @compileError("the block is written out in equal shares");
        if (sharedBytes(cfg) > MAX_SHARED_BYTES) @compileError("shared memory exceeds MAX_SHARED_BYTES");
        return;
    }
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
    return std.fmt.comptimePrint("{s}_{d}x{d}x{d}_{d}x{d}_{s}{s}{s}{s}{s}", .{
        if (cfg.kind == .conv) "cv" else "mm",
        cfg.bm,
        cfg.bn,
        cfg.bk,
        cfg.tm,
        cfg.tn,
        if (cfg.vec4_load) "v" else "s",
        if (cfg.double_buffer) "_db" else "",
        if (cfg.bounds_check) "" else "_nb",
        if (cfg.b_rows) "_bt" else "",
        if (cfg.matrix) "_mx" else "",
    });
}

const swiz = [4][]const u8{ "x", "y", "z", "w" };

// ---- global-load expressions (the one place layout/dtype detail lives) ------

/// WGSL for A's contiguous vec4 at (gr, gk). Fast path: 4 in-bounds elements.
fn aVec4(cfg: MatmulConfig) []const u8 {
    if (cfg.kind == .conv) return "im2col_vec4(gr, gk)";
    return if (cfg.vec4_load)
        "a[(a_base + gr * a_row + gk) / 4u]"
    else
        "vec4<f32>(a[a_base + gr * a_row + gk], a[a_base + gr * a_row + gk + 1u], a[a_base + gr * a_row + gk + 2u], a[a_base + gr * a_row + gk + 3u])";
}
fn bVec4(cfg: MatmulConfig) []const u8 {
    return if (cfg.vec4_load)
        "b[(b_base + gk * b_row + gc) / 4u]"
    else
        "vec4<f32>(b[b_base + gk * b_row + gc], b[b_base + gk * b_row + gc + 1u], b[b_base + gk * b_row + gc + 2u], b[b_base + gk * b_row + gc + 3u])";
}
/// WGSL for the vec4 of `[N, K]` row `gc + l` at k `gk .. gk + 3` (`b_rows`).
fn bRowVec4(w: *Wgsl, cfg: MatmulConfig, l: u32) []const u8 {
    return if (cfg.vec4_load)
        w.fmt("b[(b_base + (gc + {d}u) * b_row + gk) / 4u]", .{l})
    else
        w.fmt("vec4<f32>(b[b_base + (gc + {d}u) * b_row + gk], b[b_base + (gc + {d}u) * b_row + gk + 1u], b[b_base + (gc + {d}u) * b_row + gk + 2u], b[b_base + (gc + {d}u) * b_row + gk + 3u])", .{ l, l, l, l });
}
fn bRowScalar(w: *Wgsl, cfg: MatmulConfig, l: u32, off: u32) []const u8 {
    return if (cfg.vec4_load)
        w.fmt("b[(b_base + (gc + {d}u) * b_row + gk + {d}u) / 4u][(b_base + (gc + {d}u) * b_row + gk + {d}u) % 4u]", .{ l, off, l, off })
    else
        w.fmt("b[b_base + (gc + {d}u) * b_row + gk + {d}u]", .{ l, off });
}
/// WGSL for a single A scalar at column offset `off` (per-lane edge fallback).
fn aScalar(w: *Wgsl, cfg: MatmulConfig, off: u32) []const u8 {
    if (cfg.kind == .conv) return w.fmt("im2col_at(gr, gk + {d}u)", .{off});
    return if (cfg.vec4_load)
        w.fmt("a[(a_base + gr * a_row + gk + {d}u) / 4u][(a_base + gr * a_row + gk + {d}u) % 4u]", .{ off, off })
    else
        w.fmt("a[a_base + gr * a_row + gk + {d}u]", .{off});
}
fn bScalar(w: *Wgsl, cfg: MatmulConfig, off: u32) []const u8 {
    return if (cfg.vec4_load)
        w.fmt("b[(b_base + gk * b_row + gc + {d}u) / 4u][(b_base + gk * b_row + gc + {d}u) % 4u]", .{ off, off })
    else
        w.fmt("b[b_base + gk * b_row + gc + {d}u]", .{off});
}

// ---- module scaffolding ----------------------------------------------------

fn header(w: *Wgsl, cfg: MatmulConfig) void {
    if (cfg.matrix) w.lit("enable wgpu_cooperative_matrix;");
    const ab_ty = if (cfg.vec4_load) "array<vec4<f32>>" else "array<f32>";
    if (cfg.kind == .conv) {
        // Conv geometry rides in the same uniform, and bias takes a storage slot,
        // so the bindings match the direct kernel's (x, w, bias, out, params).
        w.lit("struct Params { dims: vec4<u32>, strides: vec4<u32>, ab: vec4<f32>,");
        w.lit("                ow_out: u32, h_in: u32, w_in: u32, c_in: u32,");
        w.lit("                kh: u32, kw: u32, x_batch: u32, pad_top: u32,");
        w.lit("                pad_left: u32, stride_h: u32, stride_w: u32, dil_h: u32,");
        w.lit("                dil_w: u32, has_bias: u32, ohw: u32, reflect: u32 };");
        // A is gathered one activation at a time, so it is always scalar-addressed;
        // B is the weight matrix and still takes the 128-bit path when aligned.
        w.lit("@group(0) @binding(0) var<storage, read> a: array<f32>;");
        w.line("@group(0) @binding(1) var<storage, read> b: {s};", .{ab_ty});
        w.lit("@group(0) @binding(2) var<storage, read> bias: array<f32>;");
        w.lit("@group(0) @binding(3) var<storage, read_write> cmat: array<f32>;");
        w.lit("@group(0) @binding(4) var<uniform> p: Params;");
    } else {
        w.lit("struct Params { dims: vec4<u32>, strides: vec4<u32>, ab: vec4<f32> };");
        w.line("@group(0) @binding(0) var<storage, read> a: {s};", .{ab_ty});
        w.line("@group(0) @binding(1) var<storage, read> b: {s};", .{ab_ty});
        w.lit("@group(0) @binding(2) var<storage, read_write> cmat: array<f32>;");
        w.lit("@group(0) @binding(3) var<uniform> p: Params;");
    }
    // Different invocations stage adjacent rows. Scalar storage gives each
    // writer a separate memory location; vector-component stores can lower to
    // a read/modify/write of the whole vector (notably on Metal).
    // Batch bases (workgroup z): A, B and C of batch `z` start at `z * dims.w`,
    // `z * strides.w` and `c_off + z * M * c_row` (`c_off` rides in `ab.z` as
    // bits). A zero stride broadcasts that operand.
    w.lit("var<private> a_base: u32; var<private> b_base: u32; var<private> c_base: u32;");
    if (cfg.matrix) {
        w.line("var<workgroup> Cs: array<f32, {d}>;", .{cfg.bm * cfg.bn});
        if (cfg.kind == .conv) convTileFn(w);
    } else {
        w.line("var<workgroup> As: array<f32, {d}>;", .{cfg.bm * cfg.bk});
        w.line("var<workgroup> Bs: array<vec4<f32>, {d}>;", .{cfg.bVecs()});
        if (cfg.kind == .conv) im2colFns(w);
    }
    w.blank();
}

/// The two im2col readers — the whole difference between this and the GEMM.
/// A[m, k] is the activation at `(m -> batch, oh, ow)` offset by `(k -> kh, kw,
/// c_in)`; where the window leaves the image it is zero, or with `reflect` the
/// mirrored tap (conv.wgsl's `reflect_idx`). M runs over every batch, so one
/// dispatch covers the whole output.
fn im2colFns(w: *Wgsl) void {
    w.blank();
    w.open("fn reflect_idx(idx: i32, len: i32) -> i32", .{});
    w.lit("var v = idx;");
    w.open("loop", .{});
    w.lit("if (v >= 0 && v < len) { break; }");
    w.lit("if (v < 0) { v = -v; } else { v = (2 * len - 2) - v; }");
    w.close();
    w.lit("return v;");
    w.close();
    w.blank();
    // Where A[m, k]'s tap starts in `a` (k's channel not added), or -1 for a zero.
    w.open("fn im2col_tap(m: u32, k: u32) -> i32", .{});
    w.lit("let kwc = p.kw * p.c_in;");
    w.lit("let mb = m / p.ohw;");
    w.lit("let mr = m % p.ohw;");
    w.lit("var ih = i32((mr / p.ow_out) * p.stride_h) + i32((k / kwc) * p.dil_h) - i32(p.pad_top);");
    w.lit("var iw = i32((mr % p.ow_out) * p.stride_w) + i32(((k % kwc) / p.c_in) * p.dil_w) - i32(p.pad_left);");
    w.open("if (ih < 0 || ih >= i32(p.h_in) || iw < 0 || iw >= i32(p.w_in))", .{});
    w.lit("if (p.reflect == 0u) { return -1; }");
    w.lit("ih = reflect_idx(ih, i32(p.h_in));");
    w.lit("iw = reflect_idx(iw, i32(p.w_in));");
    w.close();
    w.lit("return i32(mb * p.x_batch + (u32(ih) * p.w_in + u32(iw)) * p.c_in);");
    w.close();
    w.blank();
    w.open("fn im2col_at(m: u32, k: u32) -> f32", .{});
    w.lit("let t = im2col_tap(m, k);");
    w.lit("if (t < 0) { return 0.0; }");
    w.lit("return a[u32(t) + k % p.c_in];");
    w.close();
    w.blank();
    // Four consecutive k stay within one (kh, kw) tap unless the channel run
    // wraps, and channels are contiguous, so the common case is one 128-bit read.
    w.open("fn im2col_vec4(m: u32, k: u32) -> vec4<f32>", .{});
    w.lit("let ci = k % p.c_in;");
    w.open("if (ci + 3u < p.c_in)", .{});
    w.lit("let t = im2col_tap(m, k);");
    w.lit("if (t < 0) { return vec4<f32>(0.0); }");
    w.lit("let base = u32(t) + ci;");
    w.lit("return vec4<f32>(a[base], a[base + 1u], a[base + 2u], a[base + 3u]);");
    w.close();
    w.lit("return vec4<f32>(im2col_at(m, k), im2col_at(m, k + 1u), im2col_at(m, k + 2u), im2col_at(m, k + 3u));");
    w.close();
}

/// The product's dims and row strides, and this batch's bases.
fn operands(w: *Wgsl, cfg: MatmulConfig) void {
    w.lit("let M = p.dims.x; let N = p.dims.y; let K = p.dims.z;");
    if (cfg.kind == .conv) {
        w.lit("let b_row = p.strides.y;");
    } else {
        w.lit("let a_row = p.strides.x; let b_row = p.strides.y;");
        w.lit("a_base = wid.z * p.dims.w; b_base = wid.z * p.strides.w; c_base = bitcast<u32>(p.ab.z) + wid.z * M * p.strides.z;");
    }
}

/// Thread/tile index setup + zeroed accumulator registers. As is transposed
/// scalar storage; Bs is vec4-typed, so `cn4` addresses the column group.
fn preamble(w: *Wgsl, cfg: MatmulConfig) void {
    const cols = cfg.bn / cfg.tn;
    operands(w, cfg);
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
    for (0..cfg.tm / 4) |g| {
        w.line("let ai{d} = dd * {d}u + rm + {d}u;", .{ g, cfg.bm, g * 4 });
        w.line("let av{d} = vec4<f32>(As[ai{d}], As[ai{d} + 1u], As[ai{d} + 2u], As[ai{d} + 3u]);", .{ g, g, g, g, g });
    }
    for (0..cfg.tm) |i| {
        for (0..cfg.vpr()) |v| w.line("acc_{d}_{d} += vec4<f32>(av{d}.{s}) * bvec{d};", .{ i, v, i / 4, swiz[i % 4], v });
    }
    w.lit("dd += 1u;");
    w.close(); // loop
    w.close(); // block
}

// ---- cooperative global->shared load (single-buffer, inline) ---------------

/// Stage this thread's vec4 of A (`vi`, slab base `k0`) into shared As, transposed
/// scalar storage: element (row, k4+lane) lands at As[(k4+lane)*bm + row].
fn coopLoadA(w: *Wgsl, cfg: MatmulConfig) void {
    const kpv = cfg.bk / 4;
    w.line("let row = vi / {d}u; let k4 = (vi % {d}u) * 4u;", .{ kpv, kpv });
    w.lit("let gr = block_row + row; let gk = k0 + k4;");
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
    for (0..4) |l| w.line("As[(k4 + {d}u) * {d}u + row] = av.{s};", .{ l, cfg.bm, swiz[l] });
}
/// Edge path: per-lane bounds-checked scalar (zero-fill out of bounds).
fn storeARowsEdge(w: *Wgsl, cfg: MatmulConfig) void {
    for (0..4) |l| {
        w.open("if (gr < M && gk + {d}u < K)", .{l});
        w.line("As[(k4 + {d}u) * {d}u + row] = {s};", .{ l, cfg.bm, aScalar(w, cfg, @intCast(l)) });
        w.otherwise();
        w.line("As[(k4 + {d}u) * {d}u + row] = 0.0;", .{ l, cfg.bm });
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

/// `coopLoadB` for `b_rows`: unit `vi` is four rows of B (`gc ..`) by four k
/// (`gk ..`), read as one vec4 per row and stored as one vec4 per k, so no two
/// threads write parts of one vector. Adjacent threads take adjacent k, so a
/// row's run is read by neighbours.
fn coopLoadBRows(w: *Wgsl, cfg: MatmulConfig) void {
    rowsUnitIndex(w, cfg, "vi", "k0");
    for (0..4) |l| {
        w.line("var r{d} = vec4<f32>(0.0);", .{l});
        loadRow(w, cfg, w.fmt("r{d}", .{l}), @intCast(l));
    }
    for (0..4) |j| w.line("Bs[(kq * 4u + {d}u) * {d}u + c4 / 4u] = vec4<f32>(r0.{s}, r1.{s}, r2.{s}, r3.{s});", .{ j, cfg.bn / 4, swiz[j], swiz[j], swiz[j], swiz[j] });
}

/// The block a `b_rows` unit `idx` covers, for the slab at `k_expr`: `kq`/`c4` in
/// the tile, `gk`/`gc` in B.
fn rowsUnitIndex(w: *Wgsl, cfg: MatmulConfig, idx: []const u8, k_expr: []const u8) void {
    const kpv = cfg.bk / 4;
    w.line("let kq = {s} % {d}u; let c4 = ({s} / {d}u) * 4u;", .{ idx, kpv, idx, kpv });
    w.line("let gk = {s} + kq * 4u; let gc = block_col + c4;", .{k_expr});
}

/// Read `[N, K]` row `gc + l` at `gk .. gk + 3` into `reg`, zero past B's edges.
fn loadRow(w: *Wgsl, cfg: MatmulConfig, reg: []const u8, l: u32) void {
    if (!cfg.bounds_check) {
        w.line("{s} = {s};", .{ reg, bRowVec4(w, cfg, l) });
        return;
    }
    w.open("if (gc + {d}u < N && gk + 3u < K)", .{l});
    w.line("{s} = {s};", .{ reg, bRowVec4(w, cfg, l) });
    w.otherwise();
    for (0..4) |off| {
        w.open("if (gc + {d}u < N && gk + {d}u < K)", .{ l, off });
        w.line("{s}.{s} = {s};", .{ reg, swiz[off], bRowScalar(w, cfg, l, @intCast(off)) });
        w.close();
    }
    w.close();
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
    if (cfg.b_rows) loadWrap(w, cfg, cfg.bUnits(), coopLoadBRows) else loadWrap(w, cfg, cfg.bVecs(), coopLoadB);
    w.lit("workgroupBarrier();");
    computeBlock(w, cfg);
    w.line("workgroupBarrier(); k0 += {d}u;", .{cfg.bk});
    w.close();
}

/// Double-buffer: a software pipeline that prefetches the next slab into registers
/// while computing the one already staged in shared, hiding global-load latency.
fn kLoopDouble(w: *Wgsl, cfg: MatmulConfig) void {
    for (0..cfg.aRegs()) |j| w.line("var apre{d} = vec4<f32>(0.0);", .{j});
    for (0..cfg.bRegs()) |j| {
        if (!cfg.b_rows) {
            w.line("var bpre{d} = vec4<f32>(0.0);", .{j});
            continue;
        }
        for (0..4) |l| w.line("var bpre{d}_{d} = vec4<f32>(0.0);", .{ j, l });
    }
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
    if (cfg.b_rows) return prefetchRows(w, cfg, k_expr);
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

/// `prefetch`'s B half for `b_rows`: each unit's four row vec4s into `bpre{j}_{l}`.
fn prefetchRows(w: *Wgsl, cfg: MatmulConfig, k_expr: []const u8) void {
    for (0..cfg.bRegs()) |j| {
        w.openBlock();
        w.line("let vj = lidx + {d}u;", .{j * cfg.threads()});
        for (0..4) |l| w.line("bpre{d}_{d} = vec4<f32>(0.0);", .{ j, l });
        w.open("if (vj < {d}u)", .{cfg.bUnits()});
        rowsUnitIndex(w, cfg, "vj", k_expr);
        for (0..4) |l| loadRow(w, cfg, w.fmt("bpre{d}_{d}", .{ j, l }), @intCast(l));
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

/// Store prefetched registers into shared (A transposed scalar, B vec4-typed).
fn storeShared(w: *Wgsl, cfg: MatmulConfig) void {
    const nt = cfg.threads();
    const kpv = cfg.bk / 4;
    for (0..cfg.aRegs()) |j| {
        w.openBlock();
        w.line("let vj = lidx + {d}u;", .{j * nt});
        w.open("if (vj < {d}u)", .{cfg.aVecs()});
        w.line("let row = vj / {d}u; let k4 = (vj % {d}u) * 4u;", .{ kpv, kpv });
        for (0..4) |l| w.line("As[(k4 + {d}u) * {d}u + row] = apre{d}.{s};", .{ l, cfg.bm, j, swiz[l] });
        w.close();
        w.close();
    }
    for (0..cfg.bRegs()) |j| {
        w.openBlock();
        w.line("let vj = lidx + {d}u;", .{j * nt});
        w.open("if (vj < {d}u)", .{cfg.bUnits()});
        if (cfg.b_rows) {
            w.line("let kq = vj % {d}u; let c4 = (vj / {d}u) * 4u;", .{ kpv, kpv });
            for (0..4) |q| w.line("Bs[(kq * 4u + {d}u) * {d}u + c4 / 4u] = vec4<f32>(bpre{d}_0.{s}, bpre{d}_1.{s}, bpre{d}_2.{s}, bpre{d}_3.{s});", .{ q, cfg.bn / 4, j, swiz[q], j, swiz[q], j, swiz[q], j, swiz[q] });
        } else {
            w.line("Bs[vj] = bpre{d};", .{j});
        }
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
    if (cfg.bounds_check and !cfg.matrix) {
        w.open("if (r >= p.dims.x || c >= p.dims.y)", .{});
        w.lit("return;");
        w.close();
    }
    w.lit("var v = value;");
    if (cfg.kind == .conv) {
        w.open("if (p.has_bias != 0u)", .{});
        w.lit("v = v + bias[c];");
        w.close();
    }
    w.lit("let idx = c_base + r * p.strides.z + c;");
    w.open("if (p.ab.y == 0.0)", .{});
    w.lit("cmat[idx] = p.ab.x * v;");
    w.otherwise();
    w.lit("cmat[idx] = p.ab.x * v + p.ab.y * cmat[idx];");
    w.close();
    w.close();
}

// ---- matrix tiles ----------------------------------------------------------

/// One SIMD group computes a `bm x bn` block of C as 8x8 tiles: per 8 of K it
/// loads `bm/8` tiles of A and `bn/8` of B straight from their buffers and
/// accumulates every pair. The finished tiles land in shared memory, and each
/// thread writes its share through `writeOne`, where scale, bias and `beta` live.
///
/// Nothing guards a read, so the last block each way slides back until every
/// tile it reads exists, and then writes only what no earlier block owns.
fn matrixBody(w: *Wgsl, cfg: MatmulConfig) void {
    const ti = cfg.bm / TILE;
    const tj = cfg.bn / TILE;
    operands(w, cfg);
    w.line("let own_col = wid.x * {d}u; let block_col = min(own_col, N - {d}u);", .{ cfg.bn, cfg.bn });
    if (cfg.kind == .conv) {
        for (0..ti) |i| w.line("let t{d} = conv_tile(wid.y * {d}u + {d}u);", .{ i, ti, i });
    } else {
        w.line("let own_row = wid.y * {d}u; let block_row = min(own_row, M - {d}u);", .{ cfg.bm, cfg.bm });
    }
    for (0..ti) |i| {
        for (0..tj) |j| w.line("var acc_{d}_{d} = {s}();", .{ i, j, tile_c });
    }
    w.lit("var k0 = 0u;");
    w.open("loop", .{});
    w.open("if (k0 >= K)", .{});
    w.lit("break;");
    w.close();
    if (cfg.kind == .conv) {
        // 8 of K is 8 channels of one kernel tap, which sit side by side in the
        // activation; the tile's 8 pixels are `stride_w` apart along a row.
        w.lit("let tap = k0 / p.c_in;");
        w.lit("let at = ((tap / p.kw) * p.dil_h * p.w_in + (tap % p.kw) * p.dil_w) * p.c_in + k0 % p.c_in;");
        for (0..ti) |i| w.line("let a{d} = coopLoad<{s}>(&a[t{d}.z + at], p.stride_w * p.c_in);", .{ i, tile_a, i });
    } else {
        for (0..ti) |i| w.line("let a{d} = coopLoad<{s}>(&a[a_base + (block_row + {d}u) * a_row + k0], a_row);", .{ i, tile_a, i * TILE });
    }
    for (0..tj) |j| {
        // B's rows are its tile's columns: the same load, transposed.
        if (cfg.b_rows) {
            w.line("let b{d} = coopLoadT<{s}>(&b[b_base + (block_col + {d}u) * b_row + k0], b_row);", .{ j, tile_b, j * TILE });
        } else {
            w.line("let b{d} = coopLoad<{s}>(&b[b_base + k0 * b_row + block_col + {d}u], b_row);", .{ j, tile_b, j * TILE });
        }
    }
    for (0..ti) |i| {
        for (0..tj) |j| w.line("acc_{d}_{d} = coopMultiplyAdd(a{d}, b{d}, acc_{d}_{d});", .{ i, j, i, j, i, j });
    }
    w.line("k0 += {d}u;", .{TILE});
    w.close();

    for (0..ti) |i| {
        for (0..tj) |j| w.line("coopStore(acc_{d}_{d}, &Cs[{d}], {d}u);", .{ i, j, i * TILE * cfg.bn + j * TILE, cfg.bn });
    }
    w.lit("workgroupBarrier();");
    w.lit("var e = lidx;");
    w.open("loop", .{});
    w.open("if (e >= {d}u)", .{cfg.bm * cfg.bn});
    w.lit("break;");
    w.close();
    w.line("let row = e / {d}u; let c = block_col + e % {d}u;", .{ cfg.bn, cfg.bn });
    if (cfg.kind == .conv) {
        w.line("let t = conv_tile(wid.y * {d}u + row / {d}u); let px = row % {d}u;", .{ ti, TILE, TILE });
        w.open("if (t.w == 1u && px >= t.y && c >= own_col)", .{});
        w.lit("writeOne(t.x + px, c, Cs[e]);");
    } else {
        w.lit("let r = block_row + row;");
        w.open("if (r >= own_row && c >= own_col)", .{});
        w.lit("writeOne(r, c, Cs[e]);");
    }
    w.close();
    w.line("e += {d}u;", .{MATRIX_GROUP});
    w.close();
}

/// A conv's tiles of A run along output rows: 8 pixels of one row, under 8
/// channels of one kernel tap, are evenly spaced in the activation, so a tile is
/// read in place. That needs every window inside the activation — the executor
/// pads it first where one is not — and channels that divide into tiles.
///
/// `conv_tile(t)` is tile `t`: its first output pixel, how many of its pixels an
/// earlier tile owns (a row's last tile slides back to stay on the row), where
/// its first window starts in the activation, and whether `t` exists at all.
fn convTileFn(w: *Wgsl) void {
    w.blank();
    w.open("fn conv_tile(t: u32) -> vec4<u32>", .{});
    w.line("let per_row = (p.ow_out + {d}u) / {d}u;", .{ TILE - 1, TILE });
    w.lit("let rows = p.dims.x / p.ow_out;");
    w.lit("let tt = min(t, rows * per_row - 1u);");
    w.lit("let row = tt / per_row;");
    w.line("let own = (tt % per_row) * {d}u;", .{TILE});
    w.line("let ow = min(own, p.ow_out - {d}u);", .{TILE});
    w.lit("let oh_count = p.ohw / p.ow_out;");
    w.lit("let at = (row / oh_count) * p.x_batch + ((row % oh_count) * p.stride_h * p.w_in + ow * p.stride_w) * p.c_in;");
    w.lit("return vec4<u32>(row * p.ow_out + ow, own - ow, at, u32(t < rows * per_row));");
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
    if (cfg.matrix) {
        matrixBody(&w, cfg);
    } else {
        preamble(&w, cfg);
        if (cfg.double_buffer) kLoopDouble(&w, cfg) else kLoopSingle(&w, cfg);
        epilogue(&w, cfg);
    }
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
