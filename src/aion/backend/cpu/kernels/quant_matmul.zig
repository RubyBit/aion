// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const types = @import("../../types.zig");
const simd = @import("simd.zig");
const quant_matmul_registry = @import("../registry/quant_matmul_registry.zig");

const BackendError = types.BackendError;
const MatMulParams = types.MatMulParams;
const Tuning = quant_matmul_registry.Tuning;

pub const MatvecTuning = struct {
    /// SIMD lane width selected by `matvec_registry`.
    lanes: usize,
};

/// q4_0 block: 32 elements stored as 2-byte f16 scale + 16 bytes (32 nibbles)
pub const Q4_0_BLOCK_ELEMS: usize = 32;
pub const Q4_0_BLOCK_BYTES: usize = 18;

/// q8_0 block: 32 elements stored as 2-byte f16 scale + 32 bytes (int8s)
pub const Q8_0_BLOCK_ELEMS: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 34;

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

pub fn Kernel(comptime t: Tuning) type {
    return struct {
        pub const KC: usize = t.kc;
        pub const MC: usize = t.mc;
        pub const NC: usize = t.nc;

        pub const MR: usize = t.mr;
        pub const NR: usize = t.nr;

        pub const LANES: usize = t.lanes;

        pub const ScratchAlignment: usize = 32;

        comptime {
            if (NR != 2 * LANES) @compileError("quant matmul tuned kernel requires NR == 2 * LANES");
            if ((KC % Q8_0_BLOCK_ELEMS) != 0) @compileError("KC must be a multiple of 32");
            if ((NC % NR) != 0) @compileError("NC must be a multiple of NR");
        }

        // Packed-B layout matches `kernels/quant.zig`, but is parameterized by KC/NC.
        pub const PB_KBLOCK_BYTES: usize = NR * @sizeOf(f32) + Q8_0_BLOCK_ELEMS * NR * @sizeOf(i8);
        pub const PB_PANEL_BYTES: usize = (KC / Q8_0_BLOCK_ELEMS) * PB_KBLOCK_BYTES;
        pub const PB_TOTAL_BYTES: usize = (NC / NR) * PB_PANEL_BYTES;

        fn pbBytesPadded() usize {
            return std.mem.alignForward(usize, PB_TOTAL_BYTES, ScratchAlignment);
        }

        pub fn packedBBytes() usize {
            return PB_TOTAL_BYTES;
        }

        pub fn scratchBytes() usize {
            const pb: usize = pbBytesPadded();
            const pa: usize = (MC * KC) * @sizeOf(f32);
            return pb + pa;
        }

        fn scratchAsBytesAligned(scratch_bytes: []u8, need: usize) BackendError![]align(ScratchAlignment) u8 {
            if ((@intFromPtr(scratch_bytes.ptr) & (ScratchAlignment - 1)) != 0) return BackendError.InvalidArgument;
            if (scratch_bytes.len < need) return BackendError.InvalidArgument;
            return @alignCast(scratch_bytes[0..need]);
        }

        fn splitScratch(scratch_bytes: []u8) BackendError!struct { pb: []align(32) u8, pa: []align(32) f32 } {
            const pb_len: usize = pbBytesPadded();
            const pa_bytes: usize = (MC * KC) * @sizeOf(f32);
            const total: usize = pb_len + pa_bytes;

            const full: []align(32) u8 = try scratchAsBytesAligned(scratch_bytes, total);

            // Note: even when pb_len is a multiple of 32, Zig may not be able to
            // prove the alignment of the subslice. We enforce it explicitly.
            const pb: []align(32) u8 = @alignCast(full[0..pb_len]);
            const pa_u8: []align(32) u8 = @alignCast(full[pb_len .. pb_len + pa_bytes]);
            const pa: []align(32) f32 = @alignCast(std.mem.bytesAsSlice(f32, pa_u8));
            return .{ .pb = pb, .pa = pa };
        }

        inline fn scaleF16BitsToF32(scale_bits: u16) f32 {
            return @as(f32, @as(f16, @bitCast(scale_bits)));
        }

        inline fn loadScaleF32FromBlockBytes(block_ptr: [*]align(1) const u8) f32 {
            const scale_bits: u16 = @as(*align(1) const u16, @ptrCast(block_ptr)).*;
            return scaleF16BitsToF32(scale_bits);
        }

        fn zeroTailCols(nr: usize, scales: []f32, q: [*]i8) void {
            if (nr >= NR) return;

            @memset(scales[nr..NR], 0.0);
            var t_el: usize = 0;
            while (t_el < Q8_0_BLOCK_ELEMS) : (t_el += 1) {
                const row_off: usize = t_el * NR;
                @memset(q[row_off + nr .. row_off + NR], 0);
            }
        }

        // Packs A from [M, K] row-major into a contiguous [MR, KC] panel.
        // Layout in packed_out is row-major with fixed stride KC.
        fn packPanelA(
            kc: usize,
            mr: usize,
            a: []align(1) const f32,
            k_stride: usize,
            m_start: usize,
            k_start: usize,
            packed_out: *[MR * KC]f32,
        ) void {
            var r: usize = 0;
            while (r < mr) : (r += 1) {
                const a_row_off: usize = (m_start + r) * k_stride + k_start;
                const p_row_off: usize = r * KC;
                @memcpy(packed_out[p_row_off .. p_row_off + kc], a[a_row_off .. a_row_off + kc]);
            }
        }

        fn packPanelB_Q8_0(
            kc: usize,
            nr: usize,
            b_bytes: []const u8,
            n_total: usize,
            kb_start: usize,
            j_start: usize,
            packed_out: *[PB_PANEL_BYTES]u8,
        ) void {
            const k_blocks: usize = kc / Q8_0_BLOCK_ELEMS;

            var kb: usize = 0;
            while (kb < k_blocks) : (kb += 1) {
                const dst_base: [*]u8 = packed_out.ptr + kb * PB_KBLOCK_BYTES;
                const scales_ptr: [*]align(32) f32 = @ptrCast(@alignCast(dst_base));
                const q_ptr: [*]i8 = @ptrCast(dst_base + NR * @sizeOf(f32));

                var j: usize = 0;
                while (j < nr) : (j += 1) {
                    const block_off: usize = ((kb_start + kb) * n_total + (j_start + j)) * Q8_0_BLOCK_BYTES;
                    const block_ptr: [*]align(1) const u8 = b_bytes.ptr + block_off;
                    scales_ptr[j] = loadScaleF32FromBlockBytes(block_ptr);

                    const src_q: [*]align(1) const i8 = @ptrCast(block_ptr + 2);
                    var t_el: usize = 0;
                    while (t_el < Q8_0_BLOCK_ELEMS) : (t_el += 1) {
                        q_ptr[t_el * NR + j] = src_q[t_el];
                    }
                }

                zeroTailCols(nr, scales_ptr[0..NR], q_ptr);
            }
        }

        fn packPanelB_Q4_0(
            kc: usize,
            nr: usize,
            b_bytes: []const u8,
            n_total: usize,
            kb_start: usize,
            j_start: usize,
            packed_out: *[PB_PANEL_BYTES]u8,
        ) void {
            const k_blocks: usize = kc / Q4_0_BLOCK_ELEMS;

            var kb: usize = 0;
            while (kb < k_blocks) : (kb += 1) {
                const dst_base: [*]u8 = packed_out.ptr + kb * PB_KBLOCK_BYTES;
                const scales_ptr: [*]align(32) f32 = @ptrCast(@alignCast(dst_base));
                const q_ptr: [*]i8 = @ptrCast(dst_base + NR * @sizeOf(f32));

                var j: usize = 0;
                while (j < nr) : (j += 1) {
                    const block_off: usize = ((kb_start + kb) * n_total + (j_start + j)) * Q4_0_BLOCK_BYTES;
                    const block_ptr: [*]align(1) const u8 = b_bytes.ptr + block_off;
                    scales_ptr[j] = loadScaleF32FromBlockBytes(block_ptr);

                    const nib_ptr: [*]align(1) const u8 = @ptrCast(block_ptr + 2);
                    var t_el: usize = 0;
                    while (t_el < Q4_0_BLOCK_ELEMS) : (t_el += 1) {
                        const byte: u8 = nib_ptr[t_el >> 1];
                        const nib_u: u8 = if ((t_el & 1) == 0) (byte & 0x0F) else (byte >> 4);
                        q_ptr[t_el * NR + j] = @as(i8, @intCast(nib_u)) - @as(i8, 8);
                    }
                }

                zeroTailCols(nr, scales_ptr[0..NR], q_ptr);
            }
        }

        pub fn packBTileQ8_0(scratch_bytes: []u8, k: usize, n: usize, b_bytes: []const u8) BackendError!void {
            if (k > KC or n > NC) return BackendError.InvalidArgument;
            if ((k % Q8_0_BLOCK_ELEMS) != 0) return BackendError.InvalidArgument;

            const k_blocks: usize = k / Q8_0_BLOCK_ELEMS;
            if (b_bytes.len < k_blocks * n * Q8_0_BLOCK_BYTES) return BackendError.InvalidArgument;

            const s = try splitScratch(scratch_bytes);
            const packed_b: []align(32) u8 = s.pb;

            var jr: usize = 0;
            while (jr < n) : (jr += NR) {
                const nr: usize = @min(NR, n - jr);
                const panel_idx: usize = jr / NR;
                const offset: usize = panel_idx * PB_PANEL_BYTES;
                const dest: *[PB_PANEL_BYTES]u8 = @ptrCast(packed_b[offset..][0..PB_PANEL_BYTES]);
                packPanelB_Q8_0(k, nr, b_bytes, n, 0, jr, dest);
            }
        }

        pub fn packBTileQ4_0(scratch_bytes: []u8, k: usize, n: usize, b_bytes: []const u8) BackendError!void {
            if (k > KC or n > NC) return BackendError.InvalidArgument;
            if ((k % Q4_0_BLOCK_ELEMS) != 0) return BackendError.InvalidArgument;

            const k_blocks: usize = k / Q4_0_BLOCK_ELEMS;
            if (b_bytes.len < k_blocks * n * Q4_0_BLOCK_BYTES) return BackendError.InvalidArgument;

            const s = try splitScratch(scratch_bytes);
            const packed_b: []align(32) u8 = s.pb;

            var jr: usize = 0;
            while (jr < n) : (jr += NR) {
                const nr: usize = @min(NR, n - jr);
                const panel_idx: usize = jr / NR;
                const offset: usize = panel_idx * PB_PANEL_BYTES;
                const dest: *[PB_PANEL_BYTES]u8 = @ptrCast(packed_b[offset..][0..PB_PANEL_BYTES]);
                packPanelB_Q4_0(k, nr, b_bytes, n, 0, jr, dest);
            }
        }

        fn microKernel_Qx0_Fast(
            kc: usize,
            alpha: f32,
            beta: f32,
            packed_a: *const [MR * KC]f32,
            packed_b: *const [PB_PANEL_BYTES]u8,
            c: []align(1) f32,
            c_stride: usize,
            idx_m: usize,
            idx_n: usize,
        ) void {
            @setRuntimeSafety(false);
            const lanes: comptime_int = LANES;
            const num_vecs: comptime_int = NR / lanes;
            const VecF = @Vector(lanes, f32);
            const VecI8 = @Vector(lanes, i8);

            var acc: [MR][num_vecs]VecF = undefined;
            inline for (0..MR) |r| {
                inline for (0..num_vecs) |v| {
                    acc[r][v] = @splat(0.0);
                }
            }

            const a_ptr_base: [*]const f32 = @ptrCast(&packed_a[0]);
            const b_ptr_base: [*]const u8 = packed_b.ptr;
            const kb_count: usize = kc / Q8_0_BLOCK_ELEMS;

            var kb: usize = 0;
            while (kb < kb_count) : (kb += 1) {
                const kb_base: [*]const u8 = b_ptr_base + kb * PB_KBLOCK_BYTES;
                const scales_ptr: [*]align(32) const f32 = @ptrCast(@alignCast(kb_base));
                const q_ptr: [*]align(1) const i8 = @ptrCast(kb_base + NR * @sizeOf(f32));

                var scales: [num_vecs]VecF = undefined;
                inline for (0..num_vecs) |v| {
                    scales[v] = @as(*align(1) const VecF, @ptrCast(scales_ptr + v * lanes)).*;
                }

                var t_el: usize = 0;
                while (t_el < Q8_0_BLOCK_ELEMS) : (t_el += 1) {
                    const q_row: [*]align(1) const i8 = q_ptr + t_el * NR;

                    var qs: [num_vecs]VecF = undefined;
                    inline for (0..num_vecs) |v| {
                        const qx_i8: VecI8 = @as(*align(1) const VecI8, @ptrCast(q_row + v * lanes)).*;
                        const qx_f: VecF = @floatFromInt(qx_i8);
                        qs[v] = qx_f * scales[v];
                    }

                    inline for (0..MR) |r| {
                        const a_val: f32 = a_ptr_base[r * KC + kb * Q8_0_BLOCK_ELEMS + t_el];
                        const a_v: VecF = @splat(a_val);
                        inline for (0..num_vecs) |v| {
                            acc[r][v] = @mulAdd(VecF, a_v, qs[v], acc[r][v]);
                        }
                    }
                }
            }

            const alpha_v: VecF = @splat(alpha);
            const beta_v: VecF = @splat(beta);

            inline for (0..MR) |r| {
                const row_off: usize = (idx_m + r) * c_stride + idx_n;
                const c_row_ptr: [*]align(1) f32 = c.ptr + row_off;

                inline for (0..num_vecs) |v| {
                    const c_ptr_loc: [*]align(1) f32 = c_row_ptr + v * lanes;
                    const c_old: VecF = @as(*align(1) const VecF, @ptrCast(c_ptr_loc)).*;
                    const res: VecF = @mulAdd(VecF, alpha_v, acc[r][v], beta_v * c_old);
                    @as(*align(1) VecF, @ptrCast(c_ptr_loc)).* = res;
                }
            }
        }

        fn microKernel_Qx0_MR1_Fast(
            kc: usize,
            alpha: f32,
            beta: f32,
            packed_a: *const [MR * KC]f32,
            packed_b: *const [PB_PANEL_BYTES]u8,
            c: []align(1) f32,
            c_stride: usize,
            idx_m: usize,
            idx_n: usize,
        ) void {
            @setRuntimeSafety(false);
            const lanes: comptime_int = LANES;
            const num_vecs: comptime_int = NR / lanes;
            const VecF = @Vector(lanes, f32);
            const VecI8 = @Vector(lanes, i8);

            var acc: [num_vecs]VecF = undefined;
            inline for (0..num_vecs) |v| {
                acc[v] = @splat(0.0);
            }

            const a_ptr_base: [*]const f32 = @ptrCast(&packed_a[0]);
            const b_ptr_base: [*]const u8 = packed_b.ptr;
            const kb_count: usize = kc / Q8_0_BLOCK_ELEMS;

            var kb: usize = 0;
            while (kb < kb_count) : (kb += 1) {
                const kb_base: [*]const u8 = b_ptr_base + kb * PB_KBLOCK_BYTES;
                const scales_ptr: [*]align(32) const f32 = @ptrCast(@alignCast(kb_base));
                const q_ptr: [*]align(1) const i8 = @ptrCast(kb_base + NR * @sizeOf(f32));

                var scales: [num_vecs]VecF = undefined;
                inline for (0..num_vecs) |v| {
                    scales[v] = @as(*align(1) const VecF, @ptrCast(scales_ptr + v * lanes)).*;
                }

                var t_el: usize = 0;
                while (t_el < Q8_0_BLOCK_ELEMS) : (t_el += 1) {
                    const q_row: [*]align(1) const i8 = q_ptr + t_el * NR;

                    var qs: [num_vecs]VecF = undefined;
                    inline for (0..num_vecs) |v| {
                        const qx_i8: VecI8 = @as(*align(1) const VecI8, @ptrCast(q_row + v * lanes)).*;
                        const qx_f: VecF = @floatFromInt(qx_i8);
                        qs[v] = qx_f * scales[v];
                    }

                    const a_val: f32 = a_ptr_base[kb * Q8_0_BLOCK_ELEMS + t_el];
                    const a_v: VecF = @splat(a_val);
                    inline for (0..num_vecs) |v| {
                        acc[v] = @mulAdd(VecF, a_v, qs[v], acc[v]);
                    }
                }
            }

            const alpha_v: VecF = @splat(alpha);
            const row_off: usize = idx_m * c_stride + idx_n;
            const c_row_ptr: [*]align(1) f32 = c.ptr + row_off;

            if (beta == 0.0) {
                inline for (0..num_vecs) |v| {
                    const c_ptr_loc: [*]align(1) f32 = c_row_ptr + v * lanes;
                    @as(*align(1) VecF, @ptrCast(c_ptr_loc)).* = alpha_v * acc[v];
                }
            } else {
                const beta_v: VecF = @splat(beta);
                inline for (0..num_vecs) |v| {
                    const c_ptr_loc: [*]align(1) f32 = c_row_ptr + v * lanes;
                    const c_old: VecF = @as(*align(1) const VecF, @ptrCast(c_ptr_loc)).*;
                    const res: VecF = @mulAdd(VecF, alpha_v, acc[v], beta_v * c_old);
                    @as(*align(1) VecF, @ptrCast(c_ptr_loc)).* = res;
                }
            }
        }

        fn microKernel_Qx0_MR1(
            kc: usize,
            alpha: f32,
            beta: f32,
            packed_a: *const [MR * KC]f32,
            packed_b: *const [PB_PANEL_BYTES]u8,
            c: []align(1) f32,
            c_stride: usize,
            idx_m: usize,
            idx_n: usize,
            nr: usize,
        ) void {
            @setRuntimeSafety(false);
            const lanes: comptime_int = LANES;
            const num_vecs: comptime_int = NR / lanes;
            const VecF = @Vector(lanes, f32);
            const VecI8 = @Vector(lanes, i8);

            var acc: [num_vecs]VecF = undefined;
            inline for (0..num_vecs) |v| {
                acc[v] = @splat(0.0);
            }

            const a_ptr_base: [*]const f32 = @ptrCast(&packed_a[0]);
            const b_ptr_base: [*]const u8 = packed_b.ptr;
            const kb_count: usize = kc / Q8_0_BLOCK_ELEMS;

            var kb: usize = 0;
            while (kb < kb_count) : (kb += 1) {
                const kb_base: [*]const u8 = b_ptr_base + kb * PB_KBLOCK_BYTES;
                const scales_ptr: [*]align(32) const f32 = @ptrCast(@alignCast(kb_base));
                const q_ptr: [*]align(1) const i8 = @ptrCast(kb_base + NR * @sizeOf(f32));

                var scales: [num_vecs]VecF = undefined;
                inline for (0..num_vecs) |v| {
                    scales[v] = @as(*align(1) const VecF, @ptrCast(scales_ptr + v * lanes)).*;
                }

                var t_el: usize = 0;
                while (t_el < Q8_0_BLOCK_ELEMS) : (t_el += 1) {
                    const q_row: [*]align(1) const i8 = q_ptr + t_el * NR;

                    var qs: [num_vecs]VecF = undefined;
                    inline for (0..num_vecs) |v| {
                        const qx_i8: VecI8 = @as(*align(1) const VecI8, @ptrCast(q_row + v * lanes)).*;
                        const qx_f: VecF = @floatFromInt(qx_i8);
                        qs[v] = qx_f * scales[v];
                    }

                    const a_val: f32 = a_ptr_base[kb * Q8_0_BLOCK_ELEMS + t_el];
                    const a_v: VecF = @splat(a_val);
                    inline for (0..num_vecs) |v| {
                        acc[v] = @mulAdd(VecF, a_v, qs[v], acc[v]);
                    }
                }
            }

            const alpha_v: VecF = @splat(alpha);
            const row_off: usize = idx_m * c_stride + idx_n;

            inline for (0..num_vecs) |v| {
                const start_n: usize = v * lanes;
                if (nr > start_n) {
                    const rem: usize = nr - start_n;
                    const c_ptr_loc: [*]align(1) f32 = c.ptr + row_off + start_n;
                    if (rem >= lanes) {
                        if (beta == 0.0) {
                            @as(*align(1) VecF, @ptrCast(c_ptr_loc)).* = alpha_v * acc[v];
                        } else {
                            const beta_v: VecF = @splat(beta);
                            const c_old: VecF = @as(*align(1) const VecF, @ptrCast(c_ptr_loc)).*;
                            const res: VecF = @mulAdd(VecF, alpha_v, acc[v], beta_v * c_old);
                            @as(*align(1) VecF, @ptrCast(c_ptr_loc)).* = res;
                        }
                    } else {
                        const res_arr: [lanes]f32 = acc[v];
                        var vi: usize = 0;
                        while (vi < rem) : (vi += 1) {
                            const dst_i: usize = row_off + start_n + vi;
                            if (beta == 0.0) {
                                c[dst_i] = alpha * res_arr[vi];
                            } else {
                                c[dst_i] = alpha * res_arr[vi] + beta * c[dst_i];
                            }
                        }
                    }
                }
            }
        }

        fn microKernel_Qx0(
            kc: usize,
            alpha: f32,
            beta: f32,
            packed_a: *const [MR * KC]f32,
            packed_b: *const [PB_PANEL_BYTES]u8,
            c: []align(1) f32,
            c_stride: usize,
            idx_m: usize,
            idx_n: usize,
            mr: usize,
            nr: usize,
        ) void {
            @setRuntimeSafety(false);
            const lanes: comptime_int = LANES;
            const num_vecs: comptime_int = NR / lanes;
            const VecF = @Vector(lanes, f32);
            const VecI8 = @Vector(lanes, i8);

            var acc: [MR][num_vecs]VecF = undefined;
            inline for (0..MR) |r| {
                inline for (0..num_vecs) |v| {
                    acc[r][v] = @splat(0.0);
                }
            }

            const a_ptr_base: [*]const f32 = @ptrCast(&packed_a[0]);
            const b_ptr_base: [*]const u8 = packed_b.ptr;
            const kb_count: usize = kc / Q8_0_BLOCK_ELEMS;

            var kb: usize = 0;
            while (kb < kb_count) : (kb += 1) {
                const kb_base: [*]const u8 = b_ptr_base + kb * PB_KBLOCK_BYTES;
                const scales_ptr: [*]align(32) const f32 = @ptrCast(@alignCast(kb_base));
                const q_ptr: [*]align(1) const i8 = @ptrCast(kb_base + NR * @sizeOf(f32));

                var scales: [num_vecs]VecF = undefined;
                inline for (0..num_vecs) |v| {
                    scales[v] = @as(*align(1) const VecF, @ptrCast(scales_ptr + v * lanes)).*;
                }

                var t_el: usize = 0;
                while (t_el < Q8_0_BLOCK_ELEMS) : (t_el += 1) {
                    const q_row: [*]align(1) const i8 = q_ptr + t_el * NR;

                    var qs: [num_vecs]VecF = undefined;
                    inline for (0..num_vecs) |v| {
                        const qx_i8: VecI8 = @as(*align(1) const VecI8, @ptrCast(q_row + v * lanes)).*;
                        const qx_f: VecF = @floatFromInt(qx_i8);
                        qs[v] = qx_f * scales[v];
                    }

                    inline for (0..MR) |r| {
                        if (r < mr) {
                            const a_val: f32 = a_ptr_base[r * KC + kb * Q8_0_BLOCK_ELEMS + t_el];
                            const a_v: VecF = @splat(a_val);
                            inline for (0..num_vecs) |v| {
                                acc[r][v] = @mulAdd(VecF, a_v, qs[v], acc[r][v]);
                            }
                        }
                    }
                }
            }

            const alpha_v: VecF = @splat(alpha);
            const beta_v: VecF = @splat(beta);

            inline for (0..MR) |r| {
                if (r < mr) {
                    const row_off: usize = (idx_m + r) * c_stride + idx_n;

                    inline for (0..num_vecs) |v| {
                        const start_n = v * lanes;
                        if (nr > start_n) {
                            const rem = nr - start_n;
                            const c_ptr_loc: [*]align(1) f32 = c.ptr + row_off + start_n;
                            if (rem >= lanes) {
                                const c_old: VecF = @as(*align(1) const VecF, @ptrCast(c_ptr_loc)).*;
                                const res: VecF = @mulAdd(VecF, alpha_v, acc[r][v], beta_v * c_old);
                                @as(*align(1) VecF, @ptrCast(c_ptr_loc)).* = res;
                            } else {
                                const res_arr: [lanes]f32 = acc[r][v];
                                var vi: usize = 0;
                                while (vi < rem) : (vi += 1) {
                                    c[row_off + start_n + vi] = alpha * res_arr[vi] + beta * c[row_off + start_n + vi];
                                }
                            }
                        }
                    }
                }
            }
        }

        pub fn matmulQx0PackedB(
            scratch_bytes: []u8,
            packed_b_view: []align(32) const u8,
            params: MatMulParams,
            c_bytes: []u8,
            a_bytes: []const u8,
        ) BackendError!void {
            const m: usize = params.m;
            const n: usize = params.n;
            const k: usize = params.k;
            const alpha: f32 = params.alpha;
            const beta: f32 = params.beta;

            if (k > KC or n > NC) return BackendError.InvalidArgument;
            if ((k % Q8_0_BLOCK_ELEMS) != 0) return BackendError.InvalidArgument;

            if (packed_b_view.len < PB_TOTAL_BYTES) return BackendError.InvalidArgument;

            const c: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, c_bytes);
            const a: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, a_bytes);
            if (c.len < m * n or a.len < m * k) return BackendError.InvalidArgument;

            const s = try splitScratch(scratch_bytes);
            const packed_a: []align(32) f32 = s.pa;

            var ic: usize = 0;
            while (ic < m) : (ic += MC) {
                const mc: usize = @min(MC, m - ic);

                var ir: usize = 0;
                while (ir < mc) : (ir += MR) {
                    const mr: usize = @min(MR, mc - ir);
                    const panel_idx: usize = ir / MR;
                    const a_off: usize = panel_idx * (MR * KC);
                    const dest_a: *[MR * KC]f32 = @ptrCast(packed_a[a_off..][0 .. MR * KC]);
                    packPanelA(k, mr, a, k, ic + ir, 0, dest_a);
                }

                var jr_ex: usize = 0;
                while (jr_ex < n) : (jr_ex += NR) {
                    const nr: usize = @min(NR, n - jr_ex);
                    const b_off: usize = (jr_ex / NR) * PB_PANEL_BYTES;
                    const b_panel: *const [PB_PANEL_BYTES]u8 = @ptrCast(packed_b_view[b_off..][0..PB_PANEL_BYTES]);

                    var ir_ex: usize = 0;
                    while (ir_ex < mc) : (ir_ex += MR) {
                        const mr: usize = @min(MR, mc - ir_ex);
                        const a_off: usize = (ir_ex / MR) * (MR * KC);
                        const a_panel: *const [MR * KC]f32 = @ptrCast(&packed_a[a_off]);

                        if (mr == 1) {
                            if (nr == NR) {
                                microKernel_Qx0_MR1_Fast(k, alpha, beta, a_panel, b_panel, c, n, ic + ir_ex, jr_ex);
                            } else {
                                microKernel_Qx0_MR1(k, alpha, beta, a_panel, b_panel, c, n, ic + ir_ex, jr_ex, nr);
                            }
                        } else if (mr == MR and nr == NR) {
                            microKernel_Qx0_Fast(k, alpha, beta, a_panel, b_panel, c, n, ic + ir_ex, jr_ex);
                        } else {
                            microKernel_Qx0(k, alpha, beta, a_panel, b_panel, c, n, ic + ir_ex, jr_ex, mr, nr);
                        }
                    }
                }
            }
        }
    };
}
