// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// VNNI int8 GEMM for q8_0 / q4_0 weights (compute-bound prefill / large-M).
//
// The mainstream quant kernel (`matmul_q.zig`) dequantizes B to f32 and uses
// f32 FMA, which can never emit a dot-product instruction. This kernel does a
// genuine int8-accumulate GEMM so it can use VPDPBUSD (x86 AVX-VNNI / AVX-512-VNNI),
// the unsigned×signed byte dot-product that accumulates 4 products into each i32 lane.
//
// Math (per 32-element K block):
//   * Activations A[i,blk] are quantized to int8 with a per-(row,block) scale
//     `a_scale = max|a|/127`. VPDPBUSD needs an UNSIGNED first operand, so we store
//     `a_u8 = a_i8 + 128` and correct with the precomputed per-(col,block) `Σb_i8`:
//         Σ a_i8·b_i8 = vpdpbusd(a_u8,b_i8) − 128·Σb_i8
//   * B is packed VNNI-friendly: per panel of NR=8 columns, each block stores the 32
//     weights of all 8 columns reordered into 8 groups of 4 K, so one ymm feeds one
//     VPDPBUSD advancing 4 K for all 8 columns. MR rows reuse each B load.
//   * C[i,j] += a_scale[i,blk]·b_scale[j,blk]·(vpdpbusd − 128·sum_b[j,blk]).
//
// `DotEnc` selects emission: `.vex` (AVX-VNNI, runs on Alder/Raptor Lake — no AVX-512),
// `.evex` (AVX-512-VNNI), or `.portable` (scalar, identical semantics — for tests and
// any non-VNNI use). The broadcast-A layout is intended to extend to ARM `sdot` via a
// new `DotEnc` variant later; everything above `dotI8` is ISA-neutral.

const std = @import("std");
const types = @import("../../types.zig");

const BackendError = types.BackendError;
const MatMulParams = types.MatMulParams;

pub const Q8_0_BLOCK_ELEMS: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 34; // 2-byte f16 scale + 32 i8
pub const Q4_0_BLOCK_ELEMS: usize = 32;
pub const Q4_0_BLOCK_BYTES: usize = 18; // 2-byte f16 scale + 16 nibble bytes

/// Columns per VPDPBUSD accumulator (one ymm of i32).
pub const NR: usize = 8;
/// Rows blocked together so each B load is reused MR times.
pub const MR: usize = 4;

pub const PB_SCALES_BYTES: usize = NR * @sizeOf(f32); // 32
pub const PB_SUMB_BYTES: usize = NR * @sizeOf(i32); // 32
pub const PB_Q_BYTES: usize = Q8_0_BLOCK_ELEMS * NR; // 256
pub const PB_BLOCK_BYTES: usize = PB_SCALES_BYTES + PB_SUMB_BYTES + PB_Q_BYTES; // 320

/// Byte-dot instruction selection:
///   * `.vex`  — x86 AVX-VNNI (`{vex} vpdpbusd`), runs on Alder/Raptor Lake.
///   * `.evex` — x86 AVX-512-VNNI (`vpdpbusd`).
///   * `.sdot` — aarch64 FEAT_DotProd (`sdot`), signed×signed.
///   * `.portable` — scalar, signed semantics (matches `.sdot`); for tests / any CPU.
pub const DotEnc = enum { vex, evex, sdot, portable };

/// True for encodings whose dot is unsigned×signed (x86 VPDPBUSD needs an unsigned
/// first operand). Those require the `+128` bias on activations and a `-128·Σb`
/// correction; signed dots (SDOT / portable) need neither.
pub fn encNeedsBias(comptime enc: DotEnc) bool {
    return enc == .vex or enc == .evex;
}

/// Computes, per output lane j: `acc[j] += Σ_{kk=0..3} a[j*4+kk] · b[j*4+kk]`.
///
/// Activations `a` are SIGNED i8. For the unsigned×signed x86 instruction we bias
/// to u8 internally (`a_i8 + 128`, i.e. `^0x80`); the caller then subtracts
/// `128·Σb` once per block (see `encNeedsBias`). SDOT/portable are signed and need
/// no bias/correction.
pub inline fn dotI8(
    comptime enc: DotEnc,
    acc: @Vector(8, i32),
    a: @Vector(32, i8),
    b: @Vector(32, i8),
) @Vector(8, i32) {
    switch (enc) {
        .vex => {
            const u: @Vector(32, u8) = @as(@Vector(32, u8), @bitCast(a)) ^ @as(@Vector(32, u8), @splat(0x80));
            return asm (
                \\{vex} vpdpbusd %[s], %[u], %[acc]
                : [acc] "=x" (-> @Vector(8, i32)),
                : [u] "x" (u),
                  [s] "x" (b),
                  [acc0] "0" (acc),
            );
        },
        .evex => {
            const u: @Vector(32, u8) = @as(@Vector(32, u8), @bitCast(a)) ^ @as(@Vector(32, u8), @splat(0x80));
            return asm (
                \\vpdpbusd %[s], %[u], %[acc]
                : [acc] "=x" (-> @Vector(8, i32)),
                : [u] "x" (u),
                  [s] "x" (b),
                  [acc0] "0" (acc),
            );
        },
        .sdot => return dotSdot(acc, a, b),
        .portable => {
            var r = acc;
            inline for (0..8) |j| {
                var s: i32 = 0;
                inline for (0..4) |kk| s += @as(i32, a[j * 4 + kk]) * @as(i32, b[j * 4 + kk]);
                r[j] += s;
            }
            return r;
        },
    }
}

/// aarch64 `sdot Vd.4s, Vn.16b, Vm.16b`: signed int8 grouped-by-4 dot into 4 i32.
inline fn sdot4(acc: @Vector(4, i32), a: @Vector(16, i8), b: @Vector(16, i8)) @Vector(4, i32) {
    return asm (
        \\sdot %[acc].4s, %[a].16b, %[b].16b
        : [acc] "=w" (-> @Vector(4, i32)),
        : [a] "w" (a),
          [b] "w" (b),
          [acc0] "0" (acc),
    );
}

/// Two SDOTs cover the 8-lane / 32-byte tile the broadcast-A layout uses.
inline fn dotSdot(acc: @Vector(8, i32), a: @Vector(32, i8), b: @Vector(32, i8)) @Vector(8, i32) {
    const aa: [32]i8 = a;
    const bb: [32]i8 = b;
    const acc_arr: [8]i32 = acc;
    const lo = sdot4(acc_arr[0..4].*, aa[0..16].*, bb[0..16].*);
    const hi = sdot4(acc_arr[4..8].*, aa[16..32].*, bb[16..32].*);
    var out: [8]i32 = undefined;
    out[0..4].* = lo;
    out[4..8].* = hi;
    return out;
}

inline fn scaleF16BitsToF32(bits: u16) f32 {
    return @as(f32, @as(f16, @bitCast(bits)));
}

/// Quantize one K block (32 f32) to signed i8 in [-127,127]; returns the block scale.
/// (The x86 unsigned×signed dot re-biases to u8 internally; SDOT uses these directly.)
pub inline fn quantizeABlock(a_blk: [*]align(1) const f32, out_i8: [*]i8) f32 {
    var amax: f32 = 0.0;
    var t: usize = 0;
    while (t < Q8_0_BLOCK_ELEMS) : (t += 1) amax = @max(amax, @abs(a_blk[t]));
    if (amax == 0.0) {
        @memset(out_i8[0..Q8_0_BLOCK_ELEMS], 0);
        return 0.0;
    }
    const scale: f32 = amax / 127.0;
    const inv: f32 = 127.0 / amax;
    t = 0;
    while (t < Q8_0_BLOCK_ELEMS) : (t += 1) {
        const qi: i32 = @intFromFloat(std.math.clamp(@round(a_blk[t] * inv), -127.0, 127.0));
        out_i8[t] = @intCast(qi);
    }
    return scale;
}

pub const QuantSource = enum { q8_0, q4_0 };

pub fn Kernel(comptime opts: struct { kc: usize, nc: usize, enc: DotEnc }) type {
    return struct {
        pub const KC: usize = opts.kc;
        pub const NC: usize = opts.nc;
        pub const ENC: DotEnc = opts.enc;
        pub const ScratchAlignment: usize = 32;

        comptime {
            if ((KC % Q8_0_BLOCK_ELEMS) != 0) @compileError("VNNI KC must be a multiple of 32");
            if ((NC % NR) != 0) @compileError("VNNI NC must be a multiple of NR(8)");
        }

        const KBLOCKS_MAX: usize = KC / Q8_0_BLOCK_ELEMS;
        const PANELS_MAX: usize = NC / NR;

        pub const PB_TOTAL_BYTES: usize = PANELS_MAX * KBLOCKS_MAX * PB_BLOCK_BYTES;

        pub fn packedBBytes() usize {
            return PB_TOTAL_BYTES;
        }

        fn pbPadded() usize {
            return std.mem.alignForward(usize, PB_TOTAL_BYTES, ScratchAlignment);
        }

        // Activation-quant scratch: MR rows × KC bytes (i8) + MR × KBLOCKS scales.
        const AQ_ACT_BYTES: usize = MR * KC;
        const AQ_SCALE_BYTES: usize = MR * KBLOCKS_MAX * @sizeOf(f32);

        pub fn scratchBytes() usize {
            return pbPadded() + std.mem.alignForward(usize, AQ_ACT_BYTES + AQ_SCALE_BYTES, ScratchAlignment);
        }

        fn writeColBlock(q_base: [*]u8, jj: usize, vals: *const [Q8_0_BLOCK_ELEMS]i8) i32 {
            var sum: i32 = 0;
            var t: usize = 0;
            while (t < Q8_0_BLOCK_ELEMS) : (t += 1) {
                const v = vals[t];
                sum += v;
                const g: usize = t >> 2;
                const kk: usize = t & 3;
                q_base[g * (NR * 4) + jj * 4 + kk] = @bitCast(v);
            }
            return sum;
        }

        fn packGeneric(
            comptime src: QuantSource,
            scratch_bytes: []u8,
            k: usize,
            n: usize,
            b_bytes: []const u8,
        ) BackendError!void {
            if (k > KC or n > NC) return BackendError.InvalidArgument;
            const blk_elems = if (src == .q8_0) Q8_0_BLOCK_ELEMS else Q4_0_BLOCK_ELEMS;
            const blk_bytes = if (src == .q8_0) Q8_0_BLOCK_BYTES else Q4_0_BLOCK_BYTES;
            if ((k % blk_elems) != 0) return BackendError.InvalidArgument;

            const k_blocks: usize = k / blk_elems;
            const panels: usize = (n + NR - 1) / NR;
            if (b_bytes.len < k_blocks * n * blk_bytes) return BackendError.InvalidArgument;
            if (scratch_bytes.len < PB_TOTAL_BYTES) return BackendError.InvalidArgument;

            var p: usize = 0;
            while (p < panels) : (p += 1) {
                const j0 = p * NR;
                var kb: usize = 0;
                while (kb < k_blocks) : (kb += 1) {
                    const base = (p * k_blocks + kb) * PB_BLOCK_BYTES;
                    const scales = scratch_bytes.ptr + base;
                    const sumb = scratch_bytes.ptr + base + PB_SCALES_BYTES;
                    const q = scratch_bytes.ptr + base + PB_SCALES_BYTES + PB_SUMB_BYTES;

                    var jj: usize = 0;
                    while (jj < NR) : (jj += 1) {
                        const col = j0 + jj;
                        if (col >= n) {
                            @as(*align(1) f32, @ptrCast(scales + jj * 4)).* = 0.0;
                            @as(*align(1) i32, @ptrCast(sumb + jj * 4)).* = 0;
                            var t: usize = 0;
                            while (t < Q8_0_BLOCK_ELEMS) : (t += 1) q[(t >> 2) * (NR * 4) + jj * 4 + (t & 3)] = 0;
                            continue;
                        }
                        const blk_off = (kb * n + col) * blk_bytes;
                        const sb: u16 = @as(*align(1) const u16, @ptrCast(b_bytes.ptr + blk_off)).*;
                        @as(*align(1) f32, @ptrCast(scales + jj * 4)).* = scaleF16BitsToF32(sb);

                        var vals: [Q8_0_BLOCK_ELEMS]i8 = undefined;
                        if (src == .q8_0) {
                            const src_q: [*]const i8 = @ptrCast(b_bytes.ptr + blk_off + 2);
                            var t: usize = 0;
                            while (t < Q8_0_BLOCK_ELEMS) : (t += 1) vals[t] = src_q[t];
                        } else {
                            const nib: [*]const u8 = @ptrCast(b_bytes.ptr + blk_off + 2);
                            var t: usize = 0;
                            while (t < Q4_0_BLOCK_ELEMS) : (t += 1) {
                                const byte = nib[t >> 1];
                                const nib_u: u8 = if ((t & 1) == 0) (byte & 0x0F) else (byte >> 4);
                                vals[t] = @as(i8, @intCast(nib_u)) - 8;
                            }
                        }
                        @as(*align(1) i32, @ptrCast(sumb + jj * 4)).* = writeColBlock(q, jj, &vals);
                    }
                }
            }
        }

        pub fn packBTileQ8_0(scratch_bytes: []u8, k: usize, n: usize, b_bytes: []const u8) BackendError!void {
            return packGeneric(.q8_0, scratch_bytes, k, n, b_bytes);
        }

        pub fn packBTileQ4_0(scratch_bytes: []u8, k: usize, n: usize, b_bytes: []const u8) BackendError!void {
            return packGeneric(.q4_0, scratch_bytes, k, n, b_bytes);
        }

        pub fn matmulPackedB(
            scratch_bytes: []u8,
            packed_b_view: []align(32) const u8,
            params: MatMulParams,
            c_bytes: []u8,
            a_bytes: []const u8,
        ) BackendError!void {
            @setEvalBranchQuota(20000);
            const m = params.m;
            const n = params.n;
            const k = params.k;
            const ldc = if (params.ldc == 0) n else params.ldc;
            const alpha = params.alpha;
            const beta = params.beta;

            if (k > KC or n > NC) return BackendError.InvalidArgument;
            if ((k % Q8_0_BLOCK_ELEMS) != 0) return BackendError.InvalidArgument;
            const k_blocks = k / Q8_0_BLOCK_ELEMS;
            const panels = (n + NR - 1) / NR;
            if (packed_b_view.len < PB_TOTAL_BYTES) return BackendError.InvalidArgument;

            const c: []align(1) f32 = std.mem.bytesAsSlice(f32, c_bytes);
            const a: []align(1) const f32 = std.mem.bytesAsSlice(f32, a_bytes);
            if (a.len < m * k) return BackendError.InvalidArgument;

            // Carve activation-quant scratch out of the tail of scratch_bytes.
            const aq_off = pbPadded();
            if (scratch_bytes.len < aq_off + AQ_ACT_BYTES + AQ_SCALE_BYTES) return BackendError.InvalidArgument;
            const aq_i8: [*]i8 = @ptrCast(scratch_bytes.ptr + aq_off);
            const aq_scale: [*]f32 = @ptrCast(@alignCast(scratch_bytes.ptr + aq_off + AQ_ACT_BYTES));

            var i_base: usize = 0;
            while (i_base < m) : (i_base += MR) {
                const mr = @min(MR, m - i_base);

                // Quantize this row block once; reused across all column panels.
                var r: usize = 0;
                while (r < mr) : (r += 1) {
                    var kb: usize = 0;
                    while (kb < k_blocks) : (kb += 1) {
                        aq_scale[r * k_blocks + kb] = quantizeABlock(
                            a.ptr + (i_base + r) * k + kb * Q8_0_BLOCK_ELEMS,
                            aq_i8 + r * KC + kb * Q8_0_BLOCK_ELEMS,
                        );
                    }
                }

                var p: usize = 0;
                while (p < panels) : (p += 1) {
                    const rem = @min(NR, n - p * NR);
                    var cacc: [MR]@Vector(8, f32) = undefined;
                    inline for (0..MR) |rr| cacc[rr] = @splat(0.0);

                    var kb: usize = 0;
                    while (kb < k_blocks) : (kb += 1) {
                        const base = (p * k_blocks + kb) * PB_BLOCK_BYTES;
                        const scales_ptr = packed_b_view.ptr + base;
                        const sumb_ptr = packed_b_view.ptr + base + PB_SCALES_BYTES;
                        const q_ptr = packed_b_view.ptr + base + PB_SCALES_BYTES + PB_SUMB_BYTES;

                        var acc: [MR]@Vector(8, i32) = undefined;
                        inline for (0..MR) |rr| acc[rr] = @splat(0);

                        inline for (0..8) |g| {
                            const b_grp: @Vector(32, i8) = @as(*align(1) const @Vector(32, i8), @ptrCast(q_ptr + g * (NR * 4))).*;
                            inline for (0..MR) |rr| {
                                if (rr < mr) {
                                    const a_dword: u32 = @as(*align(1) const u32, @ptrCast(aq_i8 + rr * KC + kb * Q8_0_BLOCK_ELEMS + g * 4)).*;
                                    const a_bc: @Vector(32, i8) = @bitCast(@as(@Vector(8, u32), @splat(a_dword)));
                                    acc[rr] = dotI8(ENC, acc[rr], a_bc, b_grp);
                                }
                            }
                        }

                        const sumb: @Vector(8, i32) = @as(*align(1) const @Vector(8, i32), @ptrCast(sumb_ptr)).*;
                        const bscale: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(scales_ptr)).*;
                        const c128: @Vector(8, i32) = @splat(128);
                        inline for (0..MR) |rr| {
                            if (rr < mr) {
                                // Unsigned×signed dots (vex/evex) added a +128 bias per
                                // activation byte → subtract 128·Σb. SDOT/portable are signed.
                                const corrected: @Vector(8, i32) = if (comptime encNeedsBias(ENC)) acc[rr] - c128 * sumb else acc[rr];
                                const ascale: @Vector(8, f32) = @splat(aq_scale[rr * k_blocks + kb]);
                                cacc[rr] += @as(@Vector(8, f32), @floatFromInt(corrected)) * bscale * ascale;
                            }
                        }
                    }

                    const alpha_v: @Vector(8, f32) = @splat(alpha);
                    inline for (0..MR) |rr| {
                        if (rr < mr) {
                            const out_v: @Vector(8, f32) = cacc[rr] * alpha_v;
                            const c_off = (i_base + rr) * ldc + p * NR;
                            if (rem == NR) {
                                if (beta == 0.0) {
                                    @as(*align(1) @Vector(8, f32), @ptrCast(c.ptr + c_off)).* = out_v;
                                } else {
                                    const c_old: @Vector(8, f32) = @as(*align(1) const @Vector(8, f32), @ptrCast(c.ptr + c_off)).*;
                                    @as(*align(1) @Vector(8, f32), @ptrCast(c.ptr + c_off)).* = out_v + @as(@Vector(8, f32), @splat(beta)) * c_old;
                                }
                            } else {
                                const arr: [8]f32 = out_v;
                                var jj: usize = 0;
                                while (jj < rem) : (jj += 1) {
                                    if (beta == 0.0) c[c_off + jj] = arr[jj] else c[c_off + jj] = arr[jj] + beta * c[c_off + jj];
                                }
                            }
                        }
                    }
                }
            }
        }
    };
}

// ===========================================================================
// FEAT_I8MM matrix-multiply GEMM (`smmla`) for q8_0 / q4_0 weights.
//
// `smmla Vd.4s, Vn.16b, Vm.16b` computes a signed int8 2×2 outer-product tile
// over 8 K per instruction: Vn holds 2 A-rows × 8 K, Vm holds 2 B-cols × 8 K,
// and Vd accumulates the 2×2 = 4 i32 dot products (row-major:
// [r0·c0, r0·c1, r1·c0, r1·c1]). Versus FEAT_DotProd `sdot` (which broadcasts a
// single A-row across the panel and does grouped-by-4 dots), `smmla` does twice
// the MACs per instruction on compute-bound prefill / large-M GEMM.
//
// smmla is signed×signed, so — like `.sdot` and unlike the x86 unsigned×signed
// VPDPBUSD path above — it needs no `+128` activation bias and no `Σb`
// correction. The packing is therefore B = per-block scales + raw i8 weights
// reordered into col-pair / 8-K tiles; no `sumb` table.
//
// `.portable` reproduces the exact 2×2 semantics in scalar code so the kernel is
// testable on any architecture (the only execution-tested path on non-aarch64).
// ===========================================================================

/// smmla output-tile dimensions: rows and cols blocked together per microtile.
pub const MM_MR: usize = 4; // 2 row-pairs
pub const MM_NR: usize = NR; // 8 cols = 4 col-pairs (shares the panel width)
/// K elements consumed by one smmla.
pub const MM_KSUB: usize = 8;

pub const MM_PB_SCALES_BYTES: usize = MM_NR * @sizeOf(f32); // 32
pub const MM_PB_Q_BYTES: usize = Q8_0_BLOCK_ELEMS * MM_NR; // 256 (col-pair-major)
pub const MM_PB_BLOCK_BYTES: usize = MM_PB_SCALES_BYTES + MM_PB_Q_BYTES; // 288

/// smmla encoding: `.smmla` emits the aarch64 instruction; `.portable` is the
/// scalar reference with identical 2×2 semantics (tests / non-aarch64).
pub const MmEnc = enum { smmla, portable };

/// `smmla Vd.4s, Vn.16b, Vm.16b`: signed int8 2×2 matrix-multiply-accumulate.
/// Vn = 2 A-rows × 8 K, Vm = 2 B-cols × 8 K; acc lanes = [r0·c0, r0·c1, r1·c0, r1·c1].
inline fn smmla4(acc: @Vector(4, i32), a: @Vector(16, i8), b: @Vector(16, i8)) @Vector(4, i32) {
    return asm (
        \\smmla %[acc].4s, %[a].16b, %[b].16b
        : [acc] "=w" (-> @Vector(4, i32)),
        : [a] "w" (a),
          [b] "w" (b),
          [acc0] "0" (acc),
    );
}

/// Computes the 2×2 int8 outer-product tile: `acc[i*2+j] += Σ_{k=0..7} a[i*8+k]·b[j*8+k]`.
pub inline fn mmlaI8(
    comptime enc: MmEnc,
    acc: @Vector(4, i32),
    a: @Vector(16, i8),
    b: @Vector(16, i8),
) @Vector(4, i32) {
    switch (enc) {
        .smmla => return smmla4(acc, a, b),
        .portable => {
            var r = acc;
            inline for (0..2) |i| {
                inline for (0..2) |j| {
                    var s: i32 = 0;
                    inline for (0..MM_KSUB) |k| s += @as(i32, a[i * MM_KSUB + k]) * @as(i32, b[j * MM_KSUB + k]);
                    r[i * 2 + j] += s;
                }
            }
            return r;
        },
    }
}

/// FEAT_I8MM packed-GEMM kernel. Same registry ABI as `Kernel` (pack + matmul on a
/// preallocated scratch), but B is packed col-pair-major for `smmla` and there is
/// no `sumb` correction table (signed dot).
pub fn KernelMM(comptime opts: struct { kc: usize, nc: usize, enc: MmEnc }) type {
    return struct {
        pub const KC: usize = opts.kc;
        pub const NC: usize = opts.nc;
        pub const ENC: MmEnc = opts.enc;
        pub const ScratchAlignment: usize = 32;

        comptime {
            if ((KC % Q8_0_BLOCK_ELEMS) != 0) @compileError("i8mm KC must be a multiple of 32");
            if ((NC % MM_NR) != 0) @compileError("i8mm NC must be a multiple of MM_NR(8)");
        }

        const KBLOCKS_MAX: usize = KC / Q8_0_BLOCK_ELEMS;
        const PANELS_MAX: usize = NC / MM_NR;
        const KSUBS: usize = Q8_0_BLOCK_ELEMS / MM_KSUB; // 4 smmla sub-blocks per 32-K block

        pub const PB_TOTAL_BYTES: usize = PANELS_MAX * KBLOCKS_MAX * MM_PB_BLOCK_BYTES;

        pub fn packedBBytes() usize {
            return PB_TOTAL_BYTES;
        }

        fn pbPadded() usize {
            return std.mem.alignForward(usize, PB_TOTAL_BYTES, ScratchAlignment);
        }

        const AQ_ACT_BYTES: usize = MM_MR * KC;
        const AQ_SCALE_BYTES: usize = MM_MR * KBLOCKS_MAX * @sizeOf(f32);

        pub fn scratchBytes() usize {
            return pbPadded() + std.mem.alignForward(usize, AQ_ACT_BYTES + AQ_SCALE_BYTES, ScratchAlignment);
        }

        /// Scatter one column's 32 decoded i8 weights into the col-pair / 8-K layout.
        /// Column `jj` lives at pair `jj/2`, sub-row `jj&1`; each smmla sub-block of
        /// 8 K is stored contiguously (first col then second col of the pair).
        fn writeColBlockMM(q_base: [*]u8, jj: usize, vals: *const [Q8_0_BLOCK_ELEMS]i8) void {
            const cp = jj >> 1;
            const which = jj & 1;
            var sg: usize = 0;
            while (sg < KSUBS) : (sg += 1) {
                const dst = q_base + (cp * KSUBS + sg) * 16 + which * MM_KSUB;
                var t: usize = 0;
                while (t < MM_KSUB) : (t += 1) dst[t] = @bitCast(vals[sg * MM_KSUB + t]);
            }
        }

        fn packGenericMM(
            comptime src: QuantSource,
            scratch_bytes: []u8,
            k: usize,
            n: usize,
            b_bytes: []const u8,
        ) BackendError!void {
            if (k > KC or n > NC) return BackendError.InvalidArgument;
            const blk_elems = if (src == .q8_0) Q8_0_BLOCK_ELEMS else Q4_0_BLOCK_ELEMS;
            const blk_bytes = if (src == .q8_0) Q8_0_BLOCK_BYTES else Q4_0_BLOCK_BYTES;
            if ((k % blk_elems) != 0) return BackendError.InvalidArgument;

            const k_blocks: usize = k / blk_elems;
            const panels: usize = (n + MM_NR - 1) / MM_NR;
            if (b_bytes.len < k_blocks * n * blk_bytes) return BackendError.InvalidArgument;
            if (scratch_bytes.len < PB_TOTAL_BYTES) return BackendError.InvalidArgument;

            var p: usize = 0;
            while (p < panels) : (p += 1) {
                const j0 = p * MM_NR;
                var kb: usize = 0;
                while (kb < k_blocks) : (kb += 1) {
                    const base = (p * k_blocks + kb) * MM_PB_BLOCK_BYTES;
                    const scales = scratch_bytes.ptr + base;
                    const q = scratch_bytes.ptr + base + MM_PB_SCALES_BYTES;

                    var jj: usize = 0;
                    while (jj < MM_NR) : (jj += 1) {
                        const col = j0 + jj;
                        if (col >= n) {
                            @as(*align(1) f32, @ptrCast(scales + jj * 4)).* = 0.0;
                            var vals: [Q8_0_BLOCK_ELEMS]i8 = @splat(0);
                            writeColBlockMM(q, jj, &vals);
                            continue;
                        }
                        const blk_off = (kb * n + col) * blk_bytes;
                        const sb: u16 = @as(*align(1) const u16, @ptrCast(b_bytes.ptr + blk_off)).*;
                        @as(*align(1) f32, @ptrCast(scales + jj * 4)).* = scaleF16BitsToF32(sb);

                        var vals: [Q8_0_BLOCK_ELEMS]i8 = undefined;
                        if (src == .q8_0) {
                            const src_q: [*]const i8 = @ptrCast(b_bytes.ptr + blk_off + 2);
                            var t: usize = 0;
                            while (t < Q8_0_BLOCK_ELEMS) : (t += 1) vals[t] = src_q[t];
                        } else {
                            const nib: [*]const u8 = @ptrCast(b_bytes.ptr + blk_off + 2);
                            var t: usize = 0;
                            while (t < Q4_0_BLOCK_ELEMS) : (t += 1) {
                                const byte = nib[t >> 1];
                                const nib_u: u8 = if ((t & 1) == 0) (byte & 0x0F) else (byte >> 4);
                                vals[t] = @as(i8, @intCast(nib_u)) - 8;
                            }
                        }
                        writeColBlockMM(q, jj, &vals);
                    }
                }
            }
        }

        pub fn packBTileQ8_0(scratch_bytes: []u8, k: usize, n: usize, b_bytes: []const u8) BackendError!void {
            return packGenericMM(.q8_0, scratch_bytes, k, n, b_bytes);
        }

        pub fn packBTileQ4_0(scratch_bytes: []u8, k: usize, n: usize, b_bytes: []const u8) BackendError!void {
            return packGenericMM(.q4_0, scratch_bytes, k, n, b_bytes);
        }

        pub fn matmulPackedB(
            scratch_bytes: []u8,
            packed_b_view: []align(32) const u8,
            params: MatMulParams,
            c_bytes: []u8,
            a_bytes: []const u8,
        ) BackendError!void {
            @setEvalBranchQuota(20000);
            const m = params.m;
            const n = params.n;
            const k = params.k;
            const ldc = if (params.ldc == 0) n else params.ldc;
            const alpha = params.alpha;
            const beta = params.beta;

            if (k > KC or n > NC) return BackendError.InvalidArgument;
            if ((k % Q8_0_BLOCK_ELEMS) != 0) return BackendError.InvalidArgument;
            const k_blocks = k / Q8_0_BLOCK_ELEMS;
            const panels = (n + MM_NR - 1) / MM_NR;
            if (packed_b_view.len < PB_TOTAL_BYTES) return BackendError.InvalidArgument;

            const c: []align(1) f32 = std.mem.bytesAsSlice(f32, c_bytes);
            const a: []align(1) const f32 = std.mem.bytesAsSlice(f32, a_bytes);
            if (a.len < m * k) return BackendError.InvalidArgument;

            const aq_off = pbPadded();
            if (scratch_bytes.len < aq_off + AQ_ACT_BYTES + AQ_SCALE_BYTES) return BackendError.InvalidArgument;
            const aq_i8: [*]i8 = @ptrCast(scratch_bytes.ptr + aq_off);
            const aq_scale: [*]f32 = @ptrCast(@alignCast(scratch_bytes.ptr + aq_off + AQ_ACT_BYTES));

            var i_base: usize = 0;
            while (i_base < m) : (i_base += MM_MR) {
                const mr = @min(MM_MR, m - i_base);

                // Quantize this row block once; reused across all column panels.
                var r: usize = 0;
                while (r < mr) : (r += 1) {
                    var kb: usize = 0;
                    while (kb < k_blocks) : (kb += 1) {
                        aq_scale[r * k_blocks + kb] = quantizeABlock(
                            a.ptr + (i_base + r) * k + kb * Q8_0_BLOCK_ELEMS,
                            aq_i8 + r * KC + kb * Q8_0_BLOCK_ELEMS,
                        );
                    }
                }

                var p: usize = 0;
                while (p < panels) : (p += 1) {
                    const rem = @min(MM_NR, n - p * MM_NR);
                    // f32 output tile, MM_MR rows × MM_NR cols.
                    var cf: [MM_MR][MM_NR]f32 = undefined;
                    inline for (0..MM_MR) |rr| cf[rr] = @splat(0.0);

                    var kb: usize = 0;
                    while (kb < k_blocks) : (kb += 1) {
                        const base = (p * k_blocks + kb) * MM_PB_BLOCK_BYTES;
                        const scales_ptr = packed_b_view.ptr + base;
                        const q_ptr = packed_b_view.ptr + base + MM_PB_SCALES_BYTES;

                        // acc[rp][cp] : 2×2 i32 tile for row-pair rp, col-pair cp.
                        var acc: [MM_MR / 2][MM_NR / 2]@Vector(4, i32) = undefined;
                        inline for (0..MM_MR / 2) |rp| inline for (0..MM_NR / 2) |cp| {
                            acc[rp][cp] = @splat(0);
                        };

                        inline for (0..KSUBS) |sg| {
                            var bvec: [MM_NR / 2]@Vector(16, i8) = undefined;
                            inline for (0..MM_NR / 2) |cp| {
                                bvec[cp] = @as(*align(1) const @Vector(16, i8), @ptrCast(q_ptr + (cp * KSUBS + sg) * 16)).*;
                            }
                            inline for (0..MM_MR / 2) |rp| {
                                const r0 = rp * 2;
                                const r1 = rp * 2 + 1;
                                var a16: [16]i8 = @splat(0);
                                if (r0 < mr) {
                                    const sp = aq_i8 + r0 * KC + kb * Q8_0_BLOCK_ELEMS + sg * MM_KSUB;
                                    inline for (0..MM_KSUB) |t| a16[t] = sp[t];
                                }
                                if (r1 < mr) {
                                    const sp = aq_i8 + r1 * KC + kb * Q8_0_BLOCK_ELEMS + sg * MM_KSUB;
                                    inline for (0..MM_KSUB) |t| a16[MM_KSUB + t] = sp[t];
                                }
                                const avec: @Vector(16, i8) = a16;
                                inline for (0..MM_NR / 2) |cp| {
                                    acc[rp][cp] = mmlaI8(ENC, acc[rp][cp], avec, bvec[cp]);
                                }
                            }
                        }

                        // Scale the 2×2 tiles by per-row × per-col block scales.
                        var bscale: [MM_NR]f32 = undefined;
                        inline for (0..MM_NR) |jj| bscale[jj] = @as(*align(1) const f32, @ptrCast(scales_ptr + jj * 4)).*;

                        inline for (0..MM_MR / 2) |rp| {
                            const r0 = rp * 2;
                            const r1 = rp * 2 + 1;
                            const as0: f32 = if (r0 < mr) aq_scale[r0 * k_blocks + kb] else 0.0;
                            const as1: f32 = if (r1 < mr) aq_scale[r1 * k_blocks + kb] else 0.0;
                            inline for (0..MM_NR / 2) |cp| {
                                const c0 = cp * 2;
                                const c1 = cp * 2 + 1;
                                const t: [4]i32 = acc[rp][cp];
                                cf[r0][c0] += @as(f32, @floatFromInt(t[0])) * as0 * bscale[c0];
                                cf[r0][c1] += @as(f32, @floatFromInt(t[1])) * as0 * bscale[c1];
                                cf[r1][c0] += @as(f32, @floatFromInt(t[2])) * as1 * bscale[c0];
                                cf[r1][c1] += @as(f32, @floatFromInt(t[3])) * as1 * bscale[c1];
                            }
                        }
                    }

                    var rr: usize = 0;
                    while (rr < mr) : (rr += 1) {
                        const c_off = (i_base + rr) * ldc + p * MM_NR;
                        var jj: usize = 0;
                        while (jj < rem) : (jj += 1) {
                            const v = cf[rr][jj] * alpha;
                            if (beta == 0.0) c[c_off + jj] = v else c[c_off + jj] = v + beta * c[c_off + jj];
                        }
                    }
                }
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
const testing = std.testing;
const builtin = @import("builtin");

/// Whether the `.vex` encoding can be compiled *and* executed in this build:
/// x86_64 with the AVX-VNNI CPUID bit (the VEX-encoded `vpdpbusd` #UDs without
/// it, even on CPUs that have AVX-512-VNNI), and a backend whose assembler
/// knows the LLVM-only `{vex}` prefix (the self-hosted x86 backend does not).
/// Comptime-known so that gated test bodies are fully eliminated — a runtime
/// skip would still codegen the asm and break non-LLVM builds.
const vex_testable: bool = builtin.cpu.arch == .x86_64 and
    std.Target.x86.featureSetHas(builtin.cpu.features, .avxvnni) and
    builtin.zig_backend != .stage2_x86_64;

fn f32ToF16Bits(x: f32) u16 {
    return @bitCast(@as(f16, @floatCast(x)));
}

fn buildQ8_0(allocator: std.mem.Allocator, w: []const f32, k: usize, n: usize) ![]u8 {
    const k_blocks = k / Q8_0_BLOCK_ELEMS;
    const out = try allocator.alloc(u8, k_blocks * n * Q8_0_BLOCK_BYTES);
    var kb: usize = 0;
    while (kb < k_blocks) : (kb += 1) {
        var j: usize = 0;
        while (j < n) : (j += 1) {
            var amax: f32 = 0;
            var t: usize = 0;
            while (t < Q8_0_BLOCK_ELEMS) : (t += 1) amax = @max(amax, @abs(w[(kb * Q8_0_BLOCK_ELEMS + t) * n + j]));
            const scale: f32 = if (amax == 0) 0 else amax / 127.0;
            const off = (kb * n + j) * Q8_0_BLOCK_BYTES;
            @as(*align(1) u16, @ptrCast(out.ptr + off)).* = f32ToF16Bits(scale);
            const q: [*]i8 = @ptrCast(out.ptr + off + 2);
            t = 0;
            while (t < Q8_0_BLOCK_ELEMS) : (t += 1) {
                const wv = w[(kb * Q8_0_BLOCK_ELEMS + t) * n + j];
                q[t] = if (scale == 0) 0 else @intCast(@as(i32, @intFromFloat(std.math.clamp(@round(wv / scale), -127.0, 127.0))));
            }
        }
    }
    return out;
}

fn dequantQ8_0(allocator: std.mem.Allocator, b: []const u8, k: usize, n: usize) ![]f32 {
    const k_blocks = k / Q8_0_BLOCK_ELEMS;
    const out = try allocator.alloc(f32, k * n);
    var kb: usize = 0;
    while (kb < k_blocks) : (kb += 1) {
        var j: usize = 0;
        while (j < n) : (j += 1) {
            const off = (kb * n + j) * Q8_0_BLOCK_BYTES;
            const sc = scaleF16BitsToF32(@as(*align(1) const u16, @ptrCast(b.ptr + off)).*);
            const q: [*]const i8 = @ptrCast(b.ptr + off + 2);
            var t: usize = 0;
            while (t < Q8_0_BLOCK_ELEMS) : (t += 1) out[(kb * Q8_0_BLOCK_ELEMS + t) * n + j] = @as(f32, @floatFromInt(q[t])) * sc;
        }
    }
    return out;
}

test "vnni dotI8: vex (unsigned×signed, +128 bias) matches portable signed after de-bias" {
    if (comptime !vex_testable) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(0xABCDEF);
    const rnd = prng.random();
    var iter: usize = 0;
    while (iter < 256) : (iter += 1) {
        var a: [32]i8 = undefined;
        var b: [32]i8 = undefined;
        for (&a) |*x| x.* = @bitCast(rnd.int(u8));
        for (&b) |*x| x.* = @bitCast(rnd.int(u8));
        const va: @Vector(32, i8) = a;
        const vb: @Vector(32, i8) = b;
        const zero: @Vector(8, i32) = @splat(0);
        // vex result carries the +128·Σb bias per lane; subtract it to compare to signed.
        const r_vex = dotI8(.vex, zero, va, vb);
        const r_port = dotI8(.portable, zero, va, vb);
        var sumb: @Vector(8, i32) = @splat(0);
        inline for (0..8) |j| {
            var s: i32 = 0;
            inline for (0..4) |kk| s += @as(i32, b[j * 4 + kk]);
            sumb[j] = s;
        }
        const debiased = r_vex - @as(@Vector(8, i32), @splat(128)) * sumb;
        try testing.expect(@reduce(.And, debiased == r_port));
    }
}

fn refMatmul(c: []f32, a: []const f32, w: []const f32, m: usize, n: usize, k: usize) void {
    var i: usize = 0;
    while (i < m) : (i += 1) {
        var j: usize = 0;
        while (j < n) : (j += 1) {
            var s: f32 = 0;
            var kk: usize = 0;
            while (kk < k) : (kk += 1) s += a[i * k + kk] * w[kk * n + j];
            c[i * n + j] = s;
        }
    }
}

fn runKernelTest(comptime enc: DotEnc, m: usize, n: usize, k: usize) !void {
    // `.portable` runs everywhere (it already backs the i8mm reference on
    // aarch64); only the x86 asm encodings need capability gating.
    if (comptime enc == .vex and !vex_testable) return error.SkipZigTest;
    const allocator = testing.allocator;
    const K = Kernel(.{ .kc = 256, .nc = 256, .enc = enc });

    var prng = std.Random.DefaultPrng.init(0x1234 +% m +% n *% 7 +% k *% 13);
    const rnd = prng.random();
    const a = try allocator.alloc(f32, m * k);
    defer allocator.free(a);
    const w = try allocator.alloc(f32, k * n);
    defer allocator.free(w);
    for (a) |*x| x.* = rnd.float(f32) * 2 - 1;
    for (w) |*x| x.* = rnd.float(f32) * 2 - 1;

    const b_q8 = try buildQ8_0(allocator, w, k, n);
    defer allocator.free(b_q8);

    const scratch = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), K.scratchBytes());
    defer allocator.free(scratch);
    try K.packBTileQ8_0(scratch, k, n, b_q8);
    const packed_view: []align(32) const u8 = @alignCast(scratch[0..K.packedBBytes()]);

    const c = try allocator.alloc(f32, m * n);
    defer allocator.free(c);
    @memset(c, 0);
    try K.matmulPackedB(scratch, packed_view, .{ .m = m, .n = n, .k = k, .alpha = 1, .beta = 0 }, std.mem.sliceAsBytes(c), std.mem.sliceAsBytes(a));

    const w_deq = try dequantQ8_0(allocator, b_q8, k, n);
    defer allocator.free(w_deq);
    const c_ref = try allocator.alloc(f32, m * n);
    defer allocator.free(c_ref);
    refMatmul(c_ref, a, w_deq, m, n, k);

    var idx: usize = 0;
    while (idx < m * n) : (idx += 1) {
        const diff = @abs(c[idx] - c_ref[idx]);
        const tol = 0.03 * (@abs(c_ref[idx]) + 1.0);
        try testing.expect(diff <= tol);
    }
}

test "vnni kernel q8_0 (portable) full + tail shapes" {
    try runKernelTest(.portable, 5, 16, 64); // n=2 panels
    try runKernelTest(.portable, 4, 13, 96); // tail cols (13 not mult of 8), 3 k-blocks
    try runKernelTest(.portable, 1, 8, 32); // single row, single panel/block
    try runKernelTest(.portable, 9, 24, 128); // mr tail (9 = 2*MR+1)
}

test "vnni kernel q8_0 (vex) full + tail shapes" {
    try runKernelTest(.vex, 5, 16, 64);
    try runKernelTest(.vex, 4, 13, 96);
    try runKernelTest(.vex, 9, 24, 128);
}

test "i8mm mmlaI8 (portable) matches hand-rolled 2x2 outer product" {
    var prng = std.Random.DefaultPrng.init(0x5115);
    const rnd = prng.random();
    var iter: usize = 0;
    while (iter < 256) : (iter += 1) {
        var a: [16]i8 = undefined;
        var b: [16]i8 = undefined;
        for (&a) |*x| x.* = @bitCast(rnd.int(u8));
        for (&b) |*x| x.* = @bitCast(rnd.int(u8));
        const va: @Vector(16, i8) = a;
        const vb: @Vector(16, i8) = b;
        const got = mmlaI8(.portable, @splat(0), va, vb);
        var want: [4]i32 = .{ 0, 0, 0, 0 };
        inline for (0..2) |i| inline for (0..2) |j| {
            var s: i32 = 0;
            inline for (0..8) |k| s += @as(i32, a[i * 8 + k]) * @as(i32, b[j * 8 + k]);
            want[i * 2 + j] = s;
        };
        const got_arr: [4]i32 = got;
        try testing.expectEqual(want, got_arr);
    }
}

/// The `smmla` kernel performs the same exact-integer signed int8 block-GEMM as the
/// trusted `sdot`-equivalent `Kernel(.portable)`. Comparing the two directly (rather
/// than against a full-precision-A reference) isolates the smmla tiling/packing logic
/// from A-quantization noise, so the tolerance reflects only float multiply-order.
fn runMMKernelTest(comptime enc: MmEnc, m: usize, n: usize, k: usize) !void {
    const allocator = testing.allocator;
    const MM = KernelMM(.{ .kc = 256, .nc = 256, .enc = enc });
    const REF = Kernel(.{ .kc = 256, .nc = 256, .enc = .portable });

    var prng = std.Random.DefaultPrng.init(0x9A1C +% m +% n *% 7 +% k *% 13);
    const rnd = prng.random();
    const a = try allocator.alloc(f32, m * k);
    defer allocator.free(a);
    const w = try allocator.alloc(f32, k * n);
    defer allocator.free(w);
    for (a) |*x| x.* = rnd.float(f32) * 2 - 1;
    for (w) |*x| x.* = rnd.float(f32) * 2 - 1;

    const b_q8 = try buildQ8_0(allocator, w, k, n);
    defer allocator.free(b_q8);

    const c_mm = try allocator.alloc(f32, m * n);
    defer allocator.free(c_mm);
    const c_ref = try allocator.alloc(f32, m * n);
    defer allocator.free(c_ref);
    @memset(c_mm, 0);
    @memset(c_ref, 0);
    const params: MatMulParams = .{ .m = m, .n = n, .k = k, .alpha = 1, .beta = 0 };

    {
        const scratch = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), MM.scratchBytes());
        defer allocator.free(scratch);
        try MM.packBTileQ8_0(scratch, k, n, b_q8);
        const pv: []align(32) const u8 = @alignCast(scratch[0..MM.packedBBytes()]);
        try MM.matmulPackedB(scratch, pv, params, std.mem.sliceAsBytes(c_mm), std.mem.sliceAsBytes(a));
    }
    {
        const scratch = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), REF.scratchBytes());
        defer allocator.free(scratch);
        try REF.packBTileQ8_0(scratch, k, n, b_q8);
        const pv: []align(32) const u8 = @alignCast(scratch[0..REF.packedBBytes()]);
        try REF.matmulPackedB(scratch, pv, params, std.mem.sliceAsBytes(c_ref), std.mem.sliceAsBytes(a));
    }

    var idx: usize = 0;
    while (idx < m * n) : (idx += 1) {
        const diff = @abs(c_mm[idx] - c_ref[idx]);
        const tol = 1e-3 * (@abs(c_ref[idx]) + 1.0);
        try testing.expect(diff <= tol);
    }
}

test "i8mm kernel q8_0 (portable) matches sdot kernel: full + tail shapes" {
    try runMMKernelTest(.portable, 5, 16, 64); // n=2 panels
    try runMMKernelTest(.portable, 4, 13, 96); // tail cols (13 not mult of 8), 3 k-blocks
    try runMMKernelTest(.portable, 1, 8, 32); // single row → odd row-pair tail
    try runMMKernelTest(.portable, 9, 24, 128); // mr tail (9 = 2*MM_MR+1)
    try runMMKernelTest(.portable, 3, 8, 64); // odd row tail within a pair
    try runMMKernelTest(.portable, 7, 17, 96); // odd col tail within a pair
    try runMMKernelTest(.portable, 16, 32, 256); // larger, multi-block + multi-panel
}
