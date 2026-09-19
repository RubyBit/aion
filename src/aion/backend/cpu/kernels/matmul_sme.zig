// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! f32 GEMM on the ARM Scalable Matrix Extension.
//!
//! One `fmopa` accumulates a whole SVL×SVL f32 outer product into a ZA tile, so a
//! 32×32 output block is four instructions per k — an order of magnitude more
//! arithmetic per instruction than NEON's 4-lane FMA. Measured on an M5 P-core:
//! 910 GFLOP/s against 86 GFLOP/s for back-to-back NEON FMAs.
//!
//! `fmopa` wants both operands k-major, so both panels are: A is
//! `[panel][KC][MR]` and B is `[panel][KC][NR]`. A caller that cannot afford to
//! pack A can instead pass one pointer per reduction index (`matmulF32Indirect`),
//! which is how conv2d feeds this straight from the activations.
//!
//! Written for a 512-bit streaming vector (16 f32), which is what Apple implements.
//! `usable()` reports false on any other SVL and every entry point then falls back
//! to the portable path, so this kernel is safe to select purely on FEAT_SME.

const std = @import("std");
const builtin = @import("builtin");
const types = @import("../../types.zig");
const simd = @import("simd.zig");


const BackendError = types.BackendError;
const MatMulParams = types.MatMulParams;

/// f32 lanes in a streaming vector this kernel is written for.
pub const SVL: usize = 16;
pub const MR: usize = 2 * SVL;
pub const NR: usize = 2 * SVL;

const have_sme = builtin.cpu.arch.isAARCH64() and
    std.Target.aarch64.featureSetHas(builtin.cpu.features, .sme);

var svl_cache: std.atomic.Value(usize) = std.atomic.Value(usize).init(0);

/// Streaming vector length in bytes, read once inside streaming mode.
fn streamingSvlBytes() usize {
    if (!have_sme) return 0;
    const cached = svl_cache.load(.monotonic);
    if (cached != 0) return cached - 1;
    var svl: usize = 0;
    asm volatile (
        \\ smstart
        \\ rdsvl %[out], #1
        \\ smstop
        : [out] "=r" (svl),
        :
        : .{ .memory = true, .v0 = true, .v1 = true, .v2 = true, .v3 = true, .v4 = true, .v5 = true, .v6 = true, .v7 = true, .v8 = true, .v9 = true, .v10 = true, .v11 = true, .v12 = true, .v13 = true, .v14 = true, .v15 = true, .v16 = true, .v17 = true, .v18 = true, .v19 = true, .v20 = true, .v21 = true, .v22 = true, .v23 = true, .v24 = true, .v25 = true, .v26 = true, .v27 = true, .v28 = true, .v29 = true, .v30 = true, .v31 = true });
    svl_cache.store(svl + 1, .monotonic);
    return svl;
}

/// Whether this module was compiled for a target with FEAT_SME. Comptime, so a
/// build without it never instantiates the kernel or emits its inline asm.
pub fn compiledFor() bool {
    return have_sme;
}

/// Whether the SME path applies: built for it, and the streaming vector is the
/// width the kernel is written against.
pub fn usable() bool {
    return have_sme and streamingSvlBytes() == SVL * @sizeOf(f32);
}

/// What the drain does with a finished 32x32 block.
pub const Epilogue = enum { block, store, accumulate };

/// The k loop: four ZA tiles hold the quadrants of a 32x32 accumulator, so one
/// pass over k is four `fmopa` on two A and two B vectors — four loads feeding
/// four outer products, which is the most a 32x32 block can reuse (ZA holds
/// exactly four f32 tiles, so the block cannot grow).
///
/// Deliberately not unrolled. Unrolling by four amortises the pointer bumps and
/// the branch, and on a standalone loop whose panels stay L1-resident it is worth
/// 910 -> 1970 GFLOP/s. In the engine it measured no difference at all (97.3 ms
/// on VGG-19 either way): the panels stream, so the loop is fed, not issued.
/// Where the A operand for one reduction index comes from: `MR` consecutive
/// floats either way, found by walking a packed panel or by following a pointer
/// the caller supplies per index.
pub const ASource = enum { panel, table };

/// The four outer products that fold one reduction step into the 32x32 tile.
/// `a_lo`/`a_hi` hold that step's 32 A values, `b_lo`/`b_hi` its 32 B values.
fn mopaStep(comptime a_lo: []const u8, comptime a_hi: []const u8, comptime b_lo: []const u8, comptime b_hi: []const u8) []const u8 {
    return " fmopa za0.s, p0/m, p0/m, " ++ a_lo ++ ", " ++ b_lo ++
        "\n fmopa za1.s, p0/m, p0/m, " ++ a_lo ++ ", " ++ b_hi ++
        "\n fmopa za2.s, p0/m, p0/m, " ++ a_hi ++ ", " ++ b_lo ++
        "\n fmopa za3.s, p0/m, p0/m, " ++ a_hi ++ ", " ++ b_hi ++ "\n";
}

/// A for two reduction steps, into z4-z7. x10 is the panel cursor or the table
/// cursor depending on the source.
fn loadTwoA(comptime src: ASource) []const u8 {
    return switch (src) {
        // Consecutive steps are adjacent in a packed panel, so a single
        // four-register load covers both.
        .panel => " ld1w { z4.s - z7.s }, pn9/z, [x10]\n add x10, x10, #256\n",
        // Each index is its own run, so the best available is one two-register
        // load per step.
        .table => " ldr x14, [x10], #8\n ldr x15, [x10], #8\n" ++
            " ld1w { z4.s, z5.s }, pn9/z, [x14]\n ld1w { z6.s, z7.s }, pn9/z, [x15]\n",
    };
}

/// A for a single reduction step, into z4-z5.
fn loadOneA(comptime src: ASource) []const u8 {
    return switch (src) {
        .panel => " ld1w { z4.s, z5.s }, pn9/z, [x10]\n",
        .table => " ldr x14, [x10]\n ld1w { z4.s, z5.s }, pn9/z, [x14]\n",
    };
}

/// `za0..za3 = sum_k A[k] (x) B[k]`, two reduction steps per iteration.
///
/// `fmopa` is the only instruction that must run four times per step, so the
/// loop is issue-bound on it once the loads around it are folded into SME2's
/// multi-vector forms. Two steps per iteration is enough to get there: measured
/// 1748 -> 1839 GFLOP/s, within a few percent of four-deep software pipelining,
/// which is the shape ORT's KleidiAI kernel uses and is not worth the extra
/// prologue and drain here.
fn kLoop(comptime src: ASource) []const u8 {
    return " smstart\n ptrue p0.s\n ptrue pn9.b\n zero { za }\n" ++
        " mov x9, %[k]\n mov x10, %[at]\n mov x11, %[b]\n" ++
        " cmp x9, #2\n b.lt 2f\n1:\n" ++
        loadTwoA(src) ++
        " ld1w { z20.s - z23.s }, pn9/z, [x11]\n add x11, x11, #256\n" ++
        mopaStep("z4.s", "z5.s", "z20.s", "z21.s") ++
        mopaStep("z6.s", "z7.s", "z22.s", "z23.s") ++
        " sub x9, x9, #2\n cmp x9, #2\n b.ge 1b\n2:\n" ++
        " cbz x9, 3f\n" ++
        loadOneA(src) ++
        " ld1w { z20.s, z21.s }, pn9/z, [x11]\n" ++
        mopaStep("z4.s", "z5.s", "z20.s", "z21.s") ++
        "3:\n mov x12, %[c]\n mov x13, %[ldc]\n";
}

/// Walk up to 16 rows of one ZA tile pair out to memory. `tiles` names the pair,
/// `label`/`endl` keep the unrolled halves' branch targets distinct, and `limit`
/// is the register holding this half's row count — a short panel stores only the
/// rows it owns rather than staging all 32 and copying them back.
fn drain(comptime lo: []const u8, comptime hi: []const u8, comptime label: []const u8, comptime endl: []const u8, comptime limit: []const u8, comptime ep: Epilogue) []const u8 {
    const read = " mova z4.s, p0/m, " ++ lo ++ "h.s[w14, 0]\n mova z5.s, p0/m, " ++ hi ++ "h.s[w14, 0]\n";
    const combine = switch (ep) {
        .accumulate =>
        \\ ld1w { z6.s }, p0/z, [x12]
        \\ ld1w { z7.s }, p0/z, [x12, #1, mul vl]
        \\ fadd z4.s, z4.s, z6.s
        \\ fadd z5.s, z5.s, z7.s
        \\
        ,
        else => "",
    };
    return " cbz " ++ limit ++ ", " ++ endl ++ "f\n mov x14, #0\n" ++ label ++ ":\n" ++ read ++ combine ++
        \\ st1w { z4.s }, p0, [x12]
        \\ st1w { z5.s }, p0, [x12, #1, mul vl]
        \\ add x12, x12, x13
        \\ add x14, x14, #1
        \\
    ++ " cmp x14, " ++ limit ++ "\n b.ne " ++ label ++ "b\n" ++ endl ++ ":\n";
}

/// `C = sum_k at[k][32] * b[k][32]`, placed per `ep`.
///
/// `ldc` is the destination row stride in BYTES; for `.block` it is 128 (a dense
/// 32-wide row). Writing straight to C is what keeps a 32x32 block off the
/// round trip through a staging buffer.
fn kernel32x32(comptime src: ASource, comptime ep: Epilogue, at: anytype, b: [*]const f32, k: usize, m: usize, c: [*]align(1) f32, ldc: usize) void {
    // Rows split 16/16 across the two ZA tile pairs; a panel shorter than 32
    // simply stops early, and one shorter than 16 skips the second pair.
    const m_lo: usize = @min(m, 16);
    const m_hi: usize = m -| 16;
    asm volatile (kLoop(src) ++ drain("za0", "za1", "2", "4", "%[mlo]", ep) ++ drain("za2", "za3", "3", "5", "%[mhi]", ep) ++ " smstop"
        :
        : [at] "r" (at),
          [mlo] "r" (m_lo),
          [mhi] "r" (m_hi),
          [b] "r" (b),
          [k] "r" (k),
          [c] "r" (c),
          [ldc] "r" (ldc),
          // `smstart` invalidates the whole vector register file, and the loop
          // writes z0-z7 and p0 on top of that. Without naming v0-v31 (which
          // alias z0-z31) the compiler happily keeps a float live across this
          // block — `beta` did, and every accumulating call silently overwrote.
        : .{ .memory = true, .x9 = true, .x10 = true, .x11 = true, .x12 = true, .x13 = true, .x14 = true, .x15 = true, .v0 = true, .v1 = true, .v2 = true, .v3 = true, .v4 = true, .v5 = true, .v6 = true, .v7 = true, .v8 = true, .v9 = true, .v10 = true, .v11 = true, .v12 = true, .v13 = true, .v14 = true, .v15 = true, .v16 = true, .v17 = true, .v18 = true, .v19 = true, .v20 = true, .v21 = true, .v22 = true, .v23 = true, .v24 = true, .v25 = true, .v26 = true, .v27 = true, .v28 = true, .v29 = true, .v30 = true, .v31 = true });
}

pub const Tuning = struct { kc: usize, mc: usize, nc: usize };

pub fn Kernel(comptime t: Tuning) type {
    return struct {
        pub const KC = t.kc;
        pub const MC = t.mc;
        pub const NC = t.nc;
        pub const ScratchAlignment: usize = 32;

        /// Panels are whole, so a tuning whose MC/NC are not multiples of the
        /// 32-wide block still gets enough room for the rounded-up panel count.
        const A_PANELS = (MC + MR - 1) / MR;
        const B_PANELS = (NC + NR - 1) / NR;
        const PB_ELEMS = B_PANELS * KC * NR;
        const PA_ELEMS = A_PANELS * MR * KC;

        /// The dense 32x32 accumulator for blocks the kernel cannot land straight
        /// in C. Thread-local rather than stack: 4 KiB per worker.
        threadlocal var acc_buf: [MR * NR]f32 align(32) = undefined;

        /// Packed B first, then packed A — the layout `conv_utils` relies on when
        /// it lifts a packed weight straight out of scratch.
        pub fn scratchBytes() usize {
            return (PB_ELEMS + PA_ELEMS) * @sizeOf(f32);
        }

        fn scratchF32(scratch_bytes: []u8, elems: usize) BackendError![]align(ScratchAlignment) f32 {
            if ((@intFromPtr(scratch_bytes.ptr) & (ScratchAlignment - 1)) != 0) return BackendError.InvalidArgument;
            if (scratch_bytes.len < elems * @sizeOf(f32)) return BackendError.InvalidArgument;
            return @alignCast(std.mem.bytesAsSlice(f32, scratch_bytes[0 .. elems * @sizeOf(f32)]));
        }

        fn splitScratch(scratch_bytes: []u8) BackendError!struct { pb: []align(32) f32, pa: []align(32) f32 } {
            const full = try scratchF32(scratch_bytes, PB_ELEMS + PA_ELEMS);
            return .{ .pb = @alignCast(full[0..PB_ELEMS]), .pa = @alignCast(full[PB_ELEMS..]) };
        }

        pub fn packBTileF32(scratch_bytes: []u8, k: usize, n: usize, b_bytes: []const u8) BackendError!void {
            if (k > KC or n > NC) return BackendError.InvalidArgument;
            const b: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, b_bytes);
            if (b.len < k * n) return BackendError.InvalidArgument;
            const s = try splitScratch(scratch_bytes);
            packB(s.pb, k, n, b, n);
        }

        pub fn packBTileF16ToPackedF32(packed_b: []align(32) f32, k: usize, n: usize, b_bytes: []const u8) BackendError!void {
            if (k > KC or n > NC) return BackendError.InvalidArgument;
            const b: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, b_bytes);
            if (b.len < k * n) return BackendError.InvalidArgument;
            if (packed_b.len < ((n + NR - 1) / NR) * KC * NR) return BackendError.InvalidArgument;
            for (0..(n + NR - 1) / NR) |panel| {
                const nj = panel * NR;
                const nr = @min(NR, n - nj);
                const dst = packed_b[panel * (KC * NR) ..];
                for (0..k) |kk| {
                    for (0..nr) |j| dst[kk * NR + j] = @floatCast(b[kk * n + nj + j]);
                    @memset(dst[kk * NR + nr .. kk * NR + NR], 0.0);
                }
            }
        }

        fn packB(pb: []align(32) f32, k: usize, n: usize, b: []align(1) const f32, ldb: usize) void {
            for (0..(n + NR - 1) / NR) |panel| {
                const nj = panel * NR;
                const nr = @min(NR, n - nj);
                const dst = pb[panel * (KC * NR) ..];
                for (0..k) |kk| {
                    @memcpy(dst[kk * NR ..][0..nr], b[kk * ldb + nj ..][0..nr]);
                    @memset(dst[kk * NR + nr ..][0 .. NR - nr], 0.0);
                }
            }
        }

        pub fn packATileF32(k: usize, m: usize, a_bytes: []const u8, packed_a_out: []align(32) f32) BackendError!void {
            if (k > KC) return BackendError.InvalidArgument;
            const a: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, a_bytes);
            if (a.len < m * k) return BackendError.InvalidArgument;
            if (packed_a_out.len < ((m + MR - 1) / MR) * MR * KC) return BackendError.InvalidArgument;
            for (0..(m + MR - 1) / MR) |panel| {
                const mi = panel * MR;
                const mr = @min(MR, m - mi);
                const dst = packed_a_out[panel * (MR * KC) ..];
                for (0..k) |kk| {
                    for (0..mr) |r| dst[kk * MR + r] = a[(mi + r) * k + kk];
                }
            }
        }

        pub fn packATileF16ToPackedF32(packed_a_out: []align(32) f32, m: usize, k: usize, a_bytes: []const u8) BackendError!void {
            if (k > KC) return BackendError.InvalidArgument;
            const a: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, a_bytes);
            if (a.len < m * k) return BackendError.InvalidArgument;
            if (packed_a_out.len < ((m + MR - 1) / MR) * MR * KC) return BackendError.InvalidArgument;
            for (0..(m + MR - 1) / MR) |panel| {
                const mi = panel * MR;
                const mr = @min(MR, m - mi);
                const dst = packed_a_out[panel * (MR * KC) ..];
                for (0..k) |kk| {
                    for (0..mr) |r| dst[kk * MR + r] = @floatCast(a[(mi + r) * k + kk]);
                }
            }
        }

        pub fn matmulF32PackedAB(packed_a: []align(32) const f32, packed_b_view: []align(32) const f32, params: MatMulParams, c_bytes: []u8) BackendError!void {
            const m = params.m;
            const n = params.n;
            const k = params.k;
            if (m == 0 or n == 0) return BackendError.InvalidArgument;
            if (k > KC or n > NC) return BackendError.InvalidArgument;
            if (packed_b_view.len < ((n + NR - 1) / NR) * (KC * NR)) return BackendError.InvalidArgument;
            if (packed_a.len < ((m + MR - 1) / MR) * MR * KC) return BackendError.InvalidArgument;

            const c_stride: usize = if (params.ldc != 0) params.ldc else n;
            const c: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, c_bytes);
            if (c.len < (m - 1) * c_stride + n) return BackendError.InvalidArgument;

            const acc = &acc_buf;

            var mi: usize = 0;
            while (mi < m) : (mi += MR) {
                const mr = @min(MR, m - mi);
                const at = packed_a[(mi / MR) * (MR * KC) ..];

                var nj: usize = 0;
                while (nj < n) : (nj += NR) {
                    const nr = @min(NR, n - nj);
                    const b_panel = packed_b_view[(nj / NR) * (KC * NR) ..];

                    // A whole block at alpha=1 is what conv and matmul ask for
                    // almost every time, and it needs no scaling — let the kernel
                    // land it in C and skip the staging buffer entirely.
                    if (nr == NR and params.alpha == 1.0 and (params.beta == 0.0 or params.beta == 1.0)) {
                        const dst: [*]align(1) f32 = c.ptr + mi * c_stride + nj;
                        const ldc_bytes = c_stride * @sizeOf(f32);
                        if (params.beta == 0.0) {
                            kernel32x32(.panel, .store, at[0..].ptr, b_panel.ptr, k, mr, dst, ldc_bytes);
                        } else {
                            kernel32x32(.panel, .accumulate, at[0..].ptr, b_panel.ptr, k, mr, dst, ldc_bytes);
                        }
                        continue;
                    }

                    kernel32x32(.panel, .block, at[0..].ptr, b_panel.ptr, k, mr, acc[0..].ptr, NR * @sizeOf(f32));
                    store(c, c_stride, mi, nj, mr, nr, acc, params.alpha, params.beta);
                }
            }
        }

        /// `C[0..m][0..n] = alpha * sum_k A[k] (x) B[k] + beta * C`, with A given
        /// as one pointer per (row panel, reduction index).
        ///
        /// The table is panel-major: row panel `p` owns `a_tables[p*k ..][0..k]`,
        /// and every pointer addresses `MR` consecutive floats. That is what lets
        /// a caller whose A does not exist in memory — a conv's im2col operand is
        /// a 9x-redundant view of its activations — feed this kernel without
        /// materialising it.
        ///
        /// A short final panel is fine: its runs stay in bounds past `m` and the
        /// surplus rows are dropped on the way out.
        pub fn matmulF32Indirect(a_tables: []const [*]const f32, packed_b_view: []align(32) const f32, params: MatMulParams, c_bytes: []u8) BackendError!void {
            const m = params.m;
            const n = params.n;
            const k = params.k;
            if (m == 0 or n == 0) return BackendError.InvalidArgument;
            if (k > KC or n > NC) return BackendError.InvalidArgument;
            if (a_tables.len < ((m + MR - 1) / MR) * k) return BackendError.InvalidArgument;
            if (packed_b_view.len < ((n + NR - 1) / NR) * (KC * NR)) return BackendError.InvalidArgument;

            const c_stride: usize = if (params.ldc != 0) params.ldc else n;
            const c: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, c_bytes);
            if (c.len < (m - 1) * c_stride + n) return BackendError.InvalidArgument;

            const acc = &acc_buf;

            var mi: usize = 0;
            while (mi < m) : (mi += MR) {
                const mr = @min(MR, m - mi);
                const table = a_tables[(mi / MR) * k ..][0..k];

                var nj: usize = 0;
                while (nj < n) : (nj += NR) {
                    const nr = @min(NR, n - nj);
                    const b_panel = packed_b_view[(nj / NR) * (KC * NR) ..];
                    if (nr == NR and params.alpha == 1.0 and (params.beta == 0.0 or params.beta == 1.0)) {
                        const dst: [*]align(1) f32 = c.ptr + mi * c_stride + nj;
                        const ldc_bytes = c_stride * @sizeOf(f32);
                        if (params.beta == 0.0) {
                            kernel32x32(.table, .store, table.ptr, b_panel.ptr, k, mr, dst, ldc_bytes);
                        } else {
                            kernel32x32(.table, .accumulate, table.ptr, b_panel.ptr, k, mr, dst, ldc_bytes);
                        }
                        continue;
                    }

                    kernel32x32(.table, .block, table.ptr, b_panel.ptr, k, mr, acc[0..].ptr, NR * @sizeOf(f32));
                    store(c, c_stride, mi, nj, mr, nr, acc, params.alpha, params.beta);
                }
            }
        }

        /// `C = alpha*acc + beta*C` over the valid part of the 32×32 block. The
        /// kernel always fills all 32×32; rows and columns past the edge are
        /// dropped here rather than predicated in the inner loop.
        fn store(c: []align(1) f32, ldc: usize, mi: usize, nj: usize, mr: usize, nr: usize, acc: *const [MR * NR]f32, alpha: f32, beta: f32) void {
            for (0..mr) |r| {
                const dst = c[(mi + r) * ldc + nj ..][0..nr];
                const src = acc[r * NR ..][0..nr];
                if (beta == 0.0) {
                    for (dst, src) |*d, s| d.* = alpha * s;
                } else {
                    for (dst, src) |*d, s| d.* = alpha * s + beta * d.*;
                }
            }
        }

        pub fn matmulF32PackedB(scratch_bytes: []u8, packed_b_view: []align(32) const f32, params: MatMulParams, c_bytes: []u8, a_bytes: []const u8) BackendError!void {
            const s = try splitScratch(scratch_bytes);
            try packATileF32(params.k, params.m, a_bytes, s.pa);
            return matmulF32PackedAB(@alignCast(s.pa), packed_b_view, params, c_bytes);
        }
    };
}

test "sme f32 gemm matches a reference across shapes and edges" {
    if (!usable()) return error.SkipZigTest;
    try gemmCases(Kernel(.{ .kc = 128, .mc = 128, .nc = 128 }));
    try gemmCases(Kernel(.{ .kc = 512, .mc = 288, .nc = 512 }));
}

fn gemmCases(comptime K: type) !void {
    const alloc = std.testing.allocator;

    const cases = [_][3]usize{
        .{ 32, 32, 128 }, // exactly one block
        .{ 64, 64, 64 }, // several blocks
        .{ 33, 31, 17 }, // both edges, short k
        .{ 1, 1, 1 }, // degenerate
        .{ 96, 32, 128 }, // m-only multi-panel
        .{ 32, 96, 5 }, // n-only multi-panel
        .{ 17, 65, 96 },
        .{ 81, 128, 96 }, // n = 4 panels, the conv c_out=128 shape
        .{ 81, 128, 128 },
        .{ 144, 128, 128 }, // m = the conv m_cap, not a multiple of 32
        .{ 160, 128, 64 },
    };
    for (cases) |c| {
        const m = c[0];
        const n = c[1];
        const k = c[2];

        const a = try alloc.alloc(f32, m * k);
        defer alloc.free(a);
        const b = try alloc.alloc(f32, k * n);
        defer alloc.free(b);
        var rng = std.Random.DefaultPrng.init(m * 131 + n * 17 + k);
        const r = rng.random();
        for (a) |*v| v.* = r.float(f32) - 0.5;
        for (b) |*v| v.* = r.float(f32) - 0.5;

        const pa = try alloc.alignedAlloc(f32, .@"32", ((m + MR - 1) / MR) * MR * K.KC);
        defer alloc.free(pa);
        @memset(pa, 0);
        try K.packATileF32(k, m, std.mem.sliceAsBytes(a), pa);

        const scratch = try alloc.alignedAlloc(u8, .@"32", K.scratchBytes());
        defer alloc.free(scratch);
        try K.packBTileF32(scratch, k, n, std.mem.sliceAsBytes(b));
        const pb: []align(32) const f32 = @alignCast(std.mem.bytesAsSlice(f32, scratch[0 .. K.KC * K.NC * @sizeOf(f32)]));

        // beta != 0 must accumulate onto what is already in C.
        const c_buf = try alloc.alloc(f32, m * n);
        defer alloc.free(c_buf);
        for (c_buf, 0..) |*v, i| v.* = @floatFromInt(i % 7);
        const alpha: f32 = 1.5;
        const beta: f32 = 0.25;

        const want = try alloc.alloc(f32, m * n);
        defer alloc.free(want);
        for (0..m) |i| for (0..n) |j| {
            var acc: f32 = 0;
            for (0..k) |kk| acc += a[i * k + kk] * b[kk * n + j];
            want[i * n + j] = alpha * acc + beta * c_buf[i * n + j];
        };

        try K.matmulF32PackedAB(pa, pb, .{ .m = m, .n = n, .k = k, .alpha = alpha, .beta = beta, .ldc = n }, std.mem.sliceAsBytes(c_buf));
        for (c_buf, want) |got, w| try std.testing.expect(@abs(got - w) <= 1e-3 * @max(@as(f32, 1.0), @abs(w)));
    }
}

test "sme indirect gemm matches a reference across row panels and scattered rows" {
    if (!usable()) return error.SkipZigTest;
    try indirectCases(Kernel(.{ .kc = 256, .mc = 288, .nc = 256 }));
}

fn indirectCases(comptime K: type) !void {
    const alloc = std.testing.allocator;

    const cases = [_][3]usize{
        .{ 32, 32, 64 }, // exactly one block
        .{ 32, 128, 96 }, // several n panels, all full
        .{ 17, 31, 33 }, // both edges
        .{ 1, 1, 1 }, // degenerate
        .{ 32, 64, 1 }, // k = 1
        .{ 7, 128, 128 }, // short panel, deep k
        .{ 64, 32, 64 }, // two whole row panels
        .{ 96, 128, 96 }, // several row panels and several n panels
        .{ 81, 65, 33 }, // row panels with a short tail, both edges
        .{ 200, 32, 17 }, // many row panels
    };
    for (cases) |c| {
        const m = c[0];
        const n = c[1];
        const k = c[2];

        // One MR-wide run per (row panel, reduction index), spaced so that
        // walking them as a packed panel would read the wrong rows.
        const panels = (m + MR - 1) / MR;
        const stride = MR + 3;
        const runs = try alloc.alloc(f32, panels * k * stride);
        defer alloc.free(runs);
        const b = try alloc.alloc(f32, k * n);
        defer alloc.free(b);
        var rng = std.Random.DefaultPrng.init(m * 131 + n * 17 + k);
        const r = rng.random();
        for (runs) |*v| v.* = r.float(f32) - 0.5;
        for (b) |*v| v.* = r.float(f32) - 0.5;

        const table = try alloc.alloc([*]const f32, panels * k);
        defer alloc.free(table);
        for (0..panels * k) |i| table[i] = runs.ptr + i * stride;

        const scratch = try alloc.alignedAlloc(u8, .@"32", K.scratchBytes());
        defer alloc.free(scratch);
        try K.packBTileF32(scratch, k, n, std.mem.sliceAsBytes(b));
        const pb: []align(32) const f32 = @alignCast(std.mem.bytesAsSlice(f32, scratch[0 .. K.KC * K.NC * @sizeOf(f32)]));

        const c_buf = try alloc.alloc(f32, m * n);
        defer alloc.free(c_buf);
        const want = try alloc.alloc(f32, m * n);
        defer alloc.free(want);

        // (1, 0) takes the straight-to-C path, (1.5, 0.25) the staged one.
        for ([_][2]f32{ .{ 1.0, 0.0 }, .{ 1.0, 1.0 }, .{ 1.5, 0.25 } }) |ab| {
            for (c_buf, 0..) |*v, i| v.* = @floatFromInt(i % 5);
            for (0..m) |i| for (0..n) |j| {
                var acc: f32 = 0;
                for (0..k) |kk| acc += table[(i / MR) * k + kk][i % MR] * b[kk * n + j];
                want[i * n + j] = ab[0] * acc + ab[1] * c_buf[i * n + j];
            };
            try K.matmulF32Indirect(
                table,
                pb,
                .{ .m = m, .n = n, .k = k, .alpha = ab[0], .beta = ab[1], .ldc = n },
                std.mem.sliceAsBytes(c_buf),
            );
            for (c_buf, want) |got, w| try std.testing.expect(@abs(got - w) <= 1e-3 * @max(@as(f32, 1.0), @abs(w)));
        }
    }
}
