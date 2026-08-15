// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Quantized matrix-vector (small-M GEMV) kernels — the q8_0 counterpart to the
// f32 `matvec.zig`. Used for autoregressive decode (M = a few rows), where B is
// streamed once from memory rather than packed. Selected via `matvec_registry`.

const std = @import("std");
const types = @import("../../types.zig");
const simd = @import("simd.zig");
const matmul_q_i8 = @import("matmul_q_i8.zig");

const BackendError = types.BackendError;
const MatMulParams = types.MatMulParams;

/// q8_0 block: 32 elements stored as 2-byte f16 scale + 32 int8s.
pub const Q8_0_BLOCK_ELEMS: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 34;

pub const MatvecTuning = struct {
    /// SIMD lane width selected by `matvec_registry`.
    lanes: usize,
    dot_enc: DotEnc = .f32,
};

pub const DotEnc = enum { f32, vex, evex, sdot };

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

/// Accumulate one K tile of an M=1 q8 matvec into SIMD scratch.  Tiled matmul
/// execution calls this for every K tile owned by the same output-N worker, so
/// horizontal reduction and the output read/write happen once for the complete
/// dot product instead of once per storage tile.
fn matvecQ8_0KMajorAccumulateImpl(
    comptime LANES: comptime_int,
    params: MatMulParams,
    c_bytes: []u8,
    a_bytes: []const u8,
    b_bytes: []const u8,
    acc_bytes: []align(32) u8,
    first_k_tile: bool,
    last_k_tile: bool,
) BackendError!void {
    if (params.m != 1 or (params.k % Q8_0_BLOCK_ELEMS) != 0) return BackendError.InvalidArgument;

    const n = params.n;
    const k = params.k;
    const blocks = k / Q8_0_BLOCK_ELEMS;
    const VF = @Vector(LANES, f32);
    const VI8 = @Vector(LANES, i8);
    const chunks_per_block: comptime_int = Q8_0_BLOCK_ELEMS / LANES;
    const scratch_need = n * @sizeOf(VF);
    if (acc_bytes.len < scratch_need) return BackendError.InvalidArgument;
    if (a_bytes.len < k * @sizeOf(f32) or b_bytes.len < blocks * n * Q8_0_BLOCK_BYTES) return BackendError.InvalidArgument;
    if (last_k_tile and c_bytes.len < n * @sizeOf(f32)) return BackendError.InvalidArgument;

    const a = simd.bytesAsSliceConstUnaligned(f32, a_bytes);
    const c = simd.bytesAsSliceMutUnaligned(f32, c_bytes);
    const acc_store: []align(32) VF = @alignCast(std.mem.bytesAsSlice(VF, acc_bytes[0..scratch_need]));
    const NT: usize = 16;

    var jt: usize = 0;
    while (jt < n) : (jt += NT) {
        const nt = @min(NT, n - jt);
        var vacc: [NT]VF = undefined;
        var jj: usize = 0;
        while (jj < nt) : (jj += 1) {
            vacc[jj] = if (first_k_tile) @splat(0.0) else acc_store[jt + jj];
        }

        var kb: usize = 0;
        while (kb < blocks) : (kb += 1) {
            const a_base = kb * Q8_0_BLOCK_ELEMS;
            const b_base = kb * n + jt;
            jj = 0;
            while (jj < nt) : (jj += 1) {
                const block_ptr = b_bytes.ptr + (b_base + jj) * Q8_0_BLOCK_BYTES;
                const scale_bits: u16 = @as(*align(1) const u16, @ptrCast(block_ptr)).*;
                const scale: VF = @splat(@as(f32, @as(f16, @bitCast(scale_bits))));
                inline for (0..chunks_per_block) |chunk| {
                    const off = chunk * LANES;
                    const qv: VI8 = @as(*align(1) const VI8, @ptrCast(block_ptr + 2 + off)).*;
                    const av: VF = @as(*align(1) const VF, @ptrCast(a.ptr + a_base + off)).*;
                    vacc[jj] = @mulAdd(VF, av, @as(VF, @floatFromInt(qv)) * scale, vacc[jj]);
                }
            }
        }

        jj = 0;
        while (jj < nt) : (jj += 1) {
            if (last_k_tile) {
                const dot = @reduce(.Add, vacc[jj]);
                const out_idx = jt + jj;
                c[out_idx] = if (params.beta == 0.0)
                    params.alpha * dot
                else
                    params.alpha * dot + params.beta * c[out_idx];
            } else {
                acc_store[jt + jj] = vacc[jj];
            }
        }
    }
}

fn matvecQ8_0KMajorDotAccumulateImpl(
    comptime enc: matmul_q_i8.DotEnc,
    params: MatMulParams,
    c_bytes: []u8,
    a_bytes: []const u8,
    b_bytes: []const u8,
    acc_bytes: []align(32) u8,
    prepared_a: []align(32) u8,
    prepare_a: bool,
    first_k_tile: bool,
    last_k_tile: bool,
) BackendError!void {
    if (params.m != 1 or (params.k % Q8_0_BLOCK_ELEMS) != 0) return BackendError.InvalidArgument;
    const n = params.n;
    const k = params.k;
    const blocks = k / Q8_0_BLOCK_ELEMS;
    const VF = @Vector(8, f32);
    const VI = @Vector(8, i32);
    const scratch_need = n * @sizeOf(VF);
    const prepared_block_bytes = Q8_0_BLOCK_ELEMS + @sizeOf(f32);
    const prepared_need = blocks * prepared_block_bytes;
    if (acc_bytes.len < scratch_need) return BackendError.InvalidArgument;
    if (prepared_a.len < prepared_need) return BackendError.InvalidArgument;
    if (a_bytes.len < k * @sizeOf(f32) or b_bytes.len < blocks * n * Q8_0_BLOCK_BYTES) return BackendError.InvalidArgument;
    if (last_k_tile and c_bytes.len < n * @sizeOf(f32)) return BackendError.InvalidArgument;

    const a = simd.bytesAsSliceConstUnaligned(f32, a_bytes);
    const c = simd.bytesAsSliceMutUnaligned(f32, c_bytes);
    const acc: []align(32) VF = @alignCast(std.mem.bytesAsSlice(VF, acc_bytes[0..scratch_need]));
    if (first_k_tile) @memset(acc, @as(VF, @splat(0.0)));

    var kb: usize = 0;
    while (kb < blocks) : (kb += 1) {
        const prepared_block = prepared_a[kb * prepared_block_bytes ..][0..prepared_block_bytes];
        const aq_ptr: [*]i8 = @ptrCast(prepared_block.ptr);
        const scale_ptr: *align(1) f32 = @ptrCast(prepared_block.ptr + Q8_0_BLOCK_ELEMS);
        if (prepare_a) scale_ptr.* = matmul_q_i8.quantizeABlock(a.ptr + kb * Q8_0_BLOCK_ELEMS, aq_ptr);
        const a_scale = scale_ptr.*;
        const aqv: @Vector(32, i8) = @as(*align(1) const @Vector(32, i8), @ptrCast(aq_ptr)).*;
        var correction: VI = @splat(0);
        if (comptime matmul_q_i8.encNeedsBias(enc)) {
            inline for (0..8) |g| {
                inline for (0..4) |kk| correction[g] += @as(i32, aq_ptr[g * 4 + kk]);
            }
            correction *= @as(VI, @splat(128));
        }

        var j: usize = 0;
        while (j < n) : (j += 1) {
            const block_ptr = b_bytes.ptr + (kb * n + j) * Q8_0_BLOCK_BYTES;
            const scale_bits: u16 = @as(*align(1) const u16, @ptrCast(block_ptr)).*;
            const scale: VF = @splat(a_scale * @as(f32, @as(f16, @bitCast(scale_bits))));
            const qv: @Vector(32, i8) = @as(*align(1) const @Vector(32, i8), @ptrCast(block_ptr + 2)).*;
            const dots = matmul_q_i8.dotI8(enc, @as(VI, @splat(0)), qv, aqv) - correction;
            acc[j] += @as(VF, @floatFromInt(dots)) * scale;
        }
    }

    if (last_k_tile) {
        var j: usize = 0;
        while (j < n) : (j += 1) {
            const dot = @reduce(.Add, acc[j]);
            c[j] = if (params.beta == 0.0) params.alpha * dot else params.alpha * dot + params.beta * c[j];
        }
    }
}

pub fn MatvecKernel(comptime t: MatvecTuning) type {
    return struct {
        pub fn matvecQ8_0KMajor(params: MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) BackendError!void {
            return matvecQ8_0KMajorImpl(t.lanes, params, c_bytes, a_bytes, b_bytes);
        }

        pub fn matvecQ8_0KMajorAccumulate(params: MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8, acc_bytes: []align(32) u8, prepared_a: []align(32) u8, prepare_a: bool, first_k_tile: bool, last_k_tile: bool) BackendError!void {
            return switch (t.dot_enc) {
                .f32 => matvecQ8_0KMajorAccumulateImpl(t.lanes, params, c_bytes, a_bytes, b_bytes, acc_bytes, first_k_tile, last_k_tile),
                .vex => matvecQ8_0KMajorDotAccumulateImpl(.vex, params, c_bytes, a_bytes, b_bytes, acc_bytes, prepared_a, prepare_a, first_k_tile, last_k_tile),
                .evex => matvecQ8_0KMajorDotAccumulateImpl(.evex, params, c_bytes, a_bytes, b_bytes, acc_bytes, prepared_a, prepare_a, first_k_tile, last_k_tile),
                .sdot => matvecQ8_0KMajorDotAccumulateImpl(.sdot, params, c_bytes, a_bytes, b_bytes, acc_bytes, prepared_a, prepare_a, first_k_tile, last_k_tile),
            };
        }
    };
}
