// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Quantized matrix-vector (small-M GEMV) kernels — the q8_0 counterpart to the
// f32 `matvec.zig`. Used for autoregressive decode (M = a few rows), where B is
// streamed once from memory rather than packed. Selected via `matvec_registry`.

const std = @import("std");
const types = @import("../../types.zig");
const simd = @import("simd.zig");

const BackendError = types.BackendError;
const MatMulParams = types.MatMulParams;

/// q8_0 block: 32 elements stored as 2-byte f16 scale + 32 int8s.
pub const Q8_0_BLOCK_ELEMS: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 34;

pub const MatvecTuning = struct {
    /// SIMD lane width selected by `matvec_registry`.
    lanes: usize,
};

/// Compute `C[0, :] = alpha * A[0, :] @ B + beta * C[0, :]` for q8_0 B without
/// pre-packing, where:
/// - A is f32 with shape [1, K] (contiguous over K)
/// - B is q8_0 in block-major [K/32, N] layout (same as packBTileQ8_0 input)
/// - C is f32 with shape [1, N]
fn matvecQ8_0KMajorImpl(comptime LANES: comptime_int, params: MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) BackendError!void {
    if (params.m == 0) return;
    if ((params.k % Q8_0_BLOCK_ELEMS) != 0) return BackendError.InvalidArgument;

    const m: usize = params.m;
    const n: usize = params.n;
    const k: usize = params.k;
    const alpha: f32 = params.alpha;
    const beta: f32 = params.beta;

    const c: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, c_bytes);
    const a: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, a_bytes);
    if (c.len < m * n) return BackendError.InvalidArgument;
    if (a.len < m * k) return BackendError.InvalidArgument;

    const blocks_per_col: usize = k / Q8_0_BLOCK_ELEMS;
    const needed_b: usize = blocks_per_col * n * Q8_0_BLOCK_BYTES;
    if (b_bytes.len < needed_b) return BackendError.InvalidArgument;

    comptime {
        if ((Q8_0_BLOCK_ELEMS % LANES) != 0) @compileError("q8 matvec helper requires Q8_0_BLOCK_ELEMS to be divisible by SIMD lanes");
    }

    // Small-M direct q8 GEMV/GEMM. Two properties matter for streaming (M = a few
    // frames): (1) each B block is dequantized ONCE and reused across all M rows
    // (no pack-B repack, no per-row B re-read); (2) B is traversed CONTIGUOUSLY.
    // B is block-major [k_blocks, n], so for a fixed k-block the columns are
    // adjacent — we tile the output columns (NT) and sweep k-blocks within a tile,
    // keeping the NT*M accumulators hot while reading B sequentially. (Reading one
    // column down all its k-blocks instead would stride by n and is bandwidth-bound,
    // which capped scaling at ~4 cores.)
    const MAX_DIRECT_M: usize = 16;
    const NT: usize = 16; // output-column tile
    if (m > MAX_DIRECT_M) return BackendError.InvalidArgument;

    const chunks_per_block: comptime_int = Q8_0_BLOCK_ELEMS / LANES;
    const VF = @Vector(LANES, f32);
    const VI8 = @Vector(LANES, i8);
    const b_ptr: [*]const u8 = b_bytes.ptr;

    var jt: usize = 0;
    while (jt < n) : (jt += NT) {
        const nt: usize = @min(NT, n - jt);
        var acc: [MAX_DIRECT_M][NT]f32 = undefined;
        for (0..m) |row| {
            for (0..nt) |jj| acc[row][jj] = 0.0;
        }
        var kb: usize = 0;
        while (kb < blocks_per_col) : (kb += 1) {
            const a_kb: usize = kb * Q8_0_BLOCK_ELEMS;
            const kb_base: usize = kb * n + jt;
            var jj: usize = 0;
            while (jj < nt) : (jj += 1) {
                const block_ptr: [*]const u8 = b_ptr + (kb_base + jj) * Q8_0_BLOCK_BYTES; // contiguous in jj
                const scale_bits: u16 = @as(*align(1) const u16, @ptrCast(block_ptr)).*;
                const v_scale: VF = @splat(@as(f32, @as(f16, @bitCast(scale_bits))));
                var bf: [Q8_0_BLOCK_ELEMS]f32 = undefined; // dequant once, reused by all rows
                inline for (0..chunks_per_block) |chunk| {
                    const lane_off: usize = chunk * LANES;
                    const qv: VI8 = @as(*align(1) const VI8, @ptrCast(block_ptr + 2 + lane_off)).*;
                    @as(*align(1) VF, @ptrCast(&bf[lane_off])).* = @as(VF, @floatFromInt(qv)) * v_scale;
                }
                var row: usize = 0;
                while (row < m) : (row += 1) {
                    const a_seg: [*]align(1) const f32 = a.ptr + row * k + a_kb;
                    var sv: VF = @splat(0.0);
                    inline for (0..chunks_per_block) |chunk| {
                        const lane_off: usize = chunk * LANES;
                        const av: VF = @as(*align(1) const VF, @ptrCast(a_seg + lane_off)).*;
                        const bv: VF = @as(*align(1) const VF, @ptrCast(&bf[lane_off])).*;
                        sv = @mulAdd(VF, av, bv, sv);
                    }
                    acc[row][jj] += @reduce(.Add, sv);
                }
            }
        }
        for (0..m) |row| {
            var jj: usize = 0;
            while (jj < nt) : (jj += 1) {
                const idx: usize = row * n + jt + jj;
                if (beta == 0.0) {
                    c[idx] = alpha * acc[row][jj];
                } else {
                    c[idx] = alpha * acc[row][jj] + beta * c[idx];
                }
            }
        }
    }
}

pub fn MatvecKernel(comptime t: MatvecTuning) type {
    return struct {
        pub fn matvecQ8_0KMajor(params: MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) BackendError!void {
            return matvecQ8_0KMajorImpl(t.lanes, params, c_bytes, a_bytes, b_bytes);
        }
    };
}
