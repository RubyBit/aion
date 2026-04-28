// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const types = @import("../../types.zig");
const simd = @import("simd.zig");
const quant_matmul_registry = @import("../registry/quant_matmul_registry.zig");

const BackendError = types.BackendError;
const MatMulParams = types.MatMulParams;
const Tuning = quant_matmul_registry.Tuning;

/// q4_0 block: 32 elements stored as 2-byte f16 scale + 16 bytes (32 nibbles)
pub const Q4_0_BLOCK_ELEMS: usize = 32;
pub const Q4_0_BLOCK_BYTES: usize = 18;

/// q8_0 block: 32 elements stored as 2-byte f16 scale + 32 bytes (int8s)
pub const Q8_0_BLOCK_ELEMS: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Compute `C[:, n_start..n_start+n_count] = alpha * A @ B[n_start..n_start+n_count, :]^T + beta*C[...]`
/// where A is M×K f32 (row-major, contiguous over K) and B is N×K q8_0 with one contiguous
/// run of `K/32` blocks per row.
///
/// Unlike the tile-packed `matmulQx0PackedB` kernel, this one reads the on-disk Q8_0 layout
/// directly — there is no pre-pack step because B's rows are already contiguous along K, the
/// natural access order for A @ B^T. This is the LM-head-style path used by `StepMatMulNTTiled`.
///
/// SIMD inner:
/// - Each Q8_0 block is 32 elements; loaded as 4x `@Vector(8, i8)` plus a 2-byte f16 scale.
/// - `@floatFromInt(VecI8)` lowers to `vpmovsxbd + vcvtdq2ps` on AVX2, avoiding scalar lane
///   unpacking.
/// - Four independent FMA accumulators hide latency inside each block.
/// - The common decode case (`m_total == 1`) prefetches one B row ahead.
pub fn matmulNtQ8_0(
    a_ptr: [*]align(1) const f32,
    b_ptr: [*]const u8,
    c_ptr: [*]align(1) f32,
    m_total: usize,
    k: usize,
    n_total: usize,
    n_start: usize,
    n_count: usize,
    alpha: f32,
    beta: f32,
) BackendError!void {
    const blocks_per_row: usize = k / Q8_0_BLOCK_ELEMS;
    const row_bytes: usize = blocks_per_row * Q8_0_BLOCK_BYTES;

    if ((k % Q8_0_BLOCK_ELEMS) != 0) return BackendError.InvalidArgument;
    if (n_start > n_total) return BackendError.InvalidArgument;
    if (n_count > (n_total - n_start)) return BackendError.InvalidArgument;

    const LANES: comptime_int = 8;
    const VF = @Vector(LANES, f32);
    const VI8 = @Vector(LANES, i8);

    if (m_total == 1) {
        const a_row: [*]align(1) const f32 = a_ptr;

        var j: usize = 0;
        while (j < n_count) : (j += 1) {
            const b_row_base: [*]const u8 = b_ptr + j * row_bytes;

            if (j + 1 < n_count) {
                @prefetch(b_ptr + (j + 1) * row_bytes, .{ .rw = .read, .locality = 3, .cache = .data });
            }

            var vacc0: VF = @splat(0.0);
            var vacc1: VF = @splat(0.0);
            var vacc2: VF = @splat(0.0);
            var vacc3: VF = @splat(0.0);

            var b: usize = 0;
            while (b < blocks_per_row) : (b += 1) {
                const block_off: usize = b * Q8_0_BLOCK_BYTES;
                const scale_bits: u16 = @as(*align(1) const u16, @ptrCast(b_row_base + block_off)).*;
                const scale: f32 = @as(f32, @as(f16, @bitCast(scale_bits)));
                const v_scale: VF = @splat(scale);

                const q_ptr: [*]align(1) const i8 = @ptrCast(b_row_base + block_off + 2);
                const a_base: [*]align(1) const f32 = a_row + b * Q8_0_BLOCK_ELEMS;

                const q0: VI8 = @as(*align(1) const VI8, @ptrCast(q_ptr + 0)).*;
                const q1: VI8 = @as(*align(1) const VI8, @ptrCast(q_ptr + 8)).*;
                const q2: VI8 = @as(*align(1) const VI8, @ptrCast(q_ptr + 16)).*;
                const q3: VI8 = @as(*align(1) const VI8, @ptrCast(q_ptr + 24)).*;

                const qf0: VF = @floatFromInt(q0);
                const qf1: VF = @floatFromInt(q1);
                const qf2: VF = @floatFromInt(q2);
                const qf3: VF = @floatFromInt(q3);

                const a0: VF = @as(*align(1) const VF, @ptrCast(a_base + 0)).*;
                const a1: VF = @as(*align(1) const VF, @ptrCast(a_base + 8)).*;
                const a2: VF = @as(*align(1) const VF, @ptrCast(a_base + 16)).*;
                const a3: VF = @as(*align(1) const VF, @ptrCast(a_base + 24)).*;

                vacc0 = @mulAdd(VF, a0, qf0 * v_scale, vacc0);
                vacc1 = @mulAdd(VF, a1, qf1 * v_scale, vacc1);
                vacc2 = @mulAdd(VF, a2, qf2 * v_scale, vacc2);
                vacc3 = @mulAdd(VF, a3, qf3 * v_scale, vacc3);
            }

            const acc: f32 = @reduce(.Add, vacc0) + @reduce(.Add, vacc1) + @reduce(.Add, vacc2) + @reduce(.Add, vacc3);

            if (beta == 0.0) {
                c_ptr[j] = alpha * acc;
            } else {
                c_ptr[j] = alpha * acc + beta * c_ptr[j];
            }
        }
        return;
    }

    var j: usize = 0;
    while (j < n_count) : (j += 1) {
        const b_row_base: [*]const u8 = b_ptr + j * row_bytes;

        var m: usize = 0;
        while (m < m_total) : (m += 1) {
            const a_row: [*]align(1) const f32 = a_ptr + m * k;

            var vacc0: VF = @splat(0.0);
            var vacc1: VF = @splat(0.0);
            var vacc2: VF = @splat(0.0);
            var vacc3: VF = @splat(0.0);

            var b: usize = 0;
            while (b < blocks_per_row) : (b += 1) {
                const block_off: usize = b * Q8_0_BLOCK_BYTES;
                const scale_bits: u16 = @as(*align(1) const u16, @ptrCast(b_row_base + block_off)).*;
                const scale: f32 = @as(f32, @as(f16, @bitCast(scale_bits)));
                const v_scale: VF = @splat(scale);

                const q_ptr: [*]align(1) const i8 = @ptrCast(b_row_base + block_off + 2);
                const a_base: [*]align(1) const f32 = a_row + b * Q8_0_BLOCK_ELEMS;

                const q0: VI8 = @as(*align(1) const VI8, @ptrCast(q_ptr + 0)).*;
                const q1: VI8 = @as(*align(1) const VI8, @ptrCast(q_ptr + 8)).*;
                const q2: VI8 = @as(*align(1) const VI8, @ptrCast(q_ptr + 16)).*;
                const q3: VI8 = @as(*align(1) const VI8, @ptrCast(q_ptr + 24)).*;

                const qf0: VF = @floatFromInt(q0);
                const qf1: VF = @floatFromInt(q1);
                const qf2: VF = @floatFromInt(q2);
                const qf3: VF = @floatFromInt(q3);

                const a0: VF = @as(*align(1) const VF, @ptrCast(a_base + 0)).*;
                const a1: VF = @as(*align(1) const VF, @ptrCast(a_base + 8)).*;
                const a2: VF = @as(*align(1) const VF, @ptrCast(a_base + 16)).*;
                const a3: VF = @as(*align(1) const VF, @ptrCast(a_base + 24)).*;

                vacc0 = @mulAdd(VF, a0, qf0 * v_scale, vacc0);
                vacc1 = @mulAdd(VF, a1, qf1 * v_scale, vacc1);
                vacc2 = @mulAdd(VF, a2, qf2 * v_scale, vacc2);
                vacc3 = @mulAdd(VF, a3, qf3 * v_scale, vacc3);
            }

            const acc: f32 = @reduce(.Add, vacc0) + @reduce(.Add, vacc1) + @reduce(.Add, vacc2) + @reduce(.Add, vacc3);

            const c_idx: usize = m * n_count + j;
            if (beta == 0.0) {
                c_ptr[c_idx] = alpha * acc;
            } else {
                c_ptr[c_idx] = alpha * acc + beta * c_ptr[c_idx];
            }
        }
    }
}

/// Compute `C[0, :] = alpha * A[0, :] @ B + beta * C[0, :]` for q8_0 B without
/// pre-packing, where:
/// - A is f32 with shape [1, K] (contiguous over K)
/// - B is q8_0 in block-major [K/32, N] layout (same as packBTileQ8_0 input)
/// - C is f32 with shape [1, N]
pub fn matvecQ8_0KMajor(params: MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) BackendError!void {
    if (params.m != 1) return BackendError.InvalidArgument;
    if ((params.k % Q8_0_BLOCK_ELEMS) != 0) return BackendError.InvalidArgument;

    const n: usize = params.n;
    const k: usize = params.k;
    const alpha: f32 = params.alpha;
    const beta: f32 = params.beta;

    const c: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, c_bytes);
    const a: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, a_bytes);
    if (c.len < n) return BackendError.InvalidArgument;
    if (a.len < k) return BackendError.InvalidArgument;

    const blocks_per_col: usize = k / Q8_0_BLOCK_ELEMS;
    const needed_b: usize = blocks_per_col * n * Q8_0_BLOCK_BYTES;
    if (b_bytes.len < needed_b) return BackendError.InvalidArgument;

    const LANES: comptime_int = 8;
    const VF = @Vector(LANES, f32);
    const VI8 = @Vector(LANES, i8);

    const a_row: [*]align(1) const f32 = a.ptr;
    const b_ptr: [*]const u8 = b_bytes.ptr;

    var j: usize = 0;
    while (j < n) : (j += 1) {
        var vacc0: VF = @splat(0.0);
        var vacc1: VF = @splat(0.0);
        var vacc2: VF = @splat(0.0);
        var vacc3: VF = @splat(0.0);

        var kb: usize = 0;
        while (kb < blocks_per_col) : (kb += 1) {
            const block_off: usize = (kb * n + j) * Q8_0_BLOCK_BYTES;
            const block_ptr: [*]align(1) const u8 = @ptrCast(b_ptr + block_off);

            const scale_bits: u16 = @as(*align(1) const u16, @ptrCast(block_ptr)).*;
            const scale: f32 = @as(f32, @as(f16, @bitCast(scale_bits)));
            const v_scale: VF = @splat(scale);

            const q_ptr: [*]align(1) const i8 = @ptrCast(block_ptr + 2);
            const a_base: [*]align(1) const f32 = a_row + kb * Q8_0_BLOCK_ELEMS;

            const q0: VI8 = @as(*align(1) const VI8, @ptrCast(q_ptr + 0)).*;
            const q1: VI8 = @as(*align(1) const VI8, @ptrCast(q_ptr + 8)).*;
            const q2: VI8 = @as(*align(1) const VI8, @ptrCast(q_ptr + 16)).*;
            const q3: VI8 = @as(*align(1) const VI8, @ptrCast(q_ptr + 24)).*;

            const qf0: VF = @floatFromInt(q0);
            const qf1: VF = @floatFromInt(q1);
            const qf2: VF = @floatFromInt(q2);
            const qf3: VF = @floatFromInt(q3);

            const a0: VF = @as(*align(1) const VF, @ptrCast(a_base + 0)).*;
            const a1: VF = @as(*align(1) const VF, @ptrCast(a_base + 8)).*;
            const a2: VF = @as(*align(1) const VF, @ptrCast(a_base + 16)).*;
            const a3: VF = @as(*align(1) const VF, @ptrCast(a_base + 24)).*;

            vacc0 = @mulAdd(VF, a0, qf0 * v_scale, vacc0);
            vacc1 = @mulAdd(VF, a1, qf1 * v_scale, vacc1);
            vacc2 = @mulAdd(VF, a2, qf2 * v_scale, vacc2);
            vacc3 = @mulAdd(VF, a3, qf3 * v_scale, vacc3);
        }

        const acc: f32 = @reduce(.Add, vacc0) + @reduce(.Add, vacc1) + @reduce(.Add, vacc2) + @reduce(.Add, vacc3);
        if (beta == 0.0) {
            c[j] = alpha * acc;
        } else {
            c[j] = alpha * acc + beta * c[j];
        }
    }
}

pub fn Kernel(comptime t: Tuning) type {
    return struct {
        pub const KC: usize = t.kc;
        pub const MC: usize = t.mc;
        pub const NC: usize = t.nc;

        pub const MR: usize = t.mr;
        pub const NR: usize = t.nr;

        pub const LANES: usize = simd.lanesF32();

        pub const ScratchAlignment: usize = 32;

        comptime {
            if (NR % LANES != 0) @compileError("NR must be a multiple of SIMD lanes");
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
                    scales[v] = @as(*align(32) const VecF, @ptrCast(scales_ptr + v * lanes)).*;
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
                    scales[v] = @as(*align(32) const VecF, @ptrCast(scales_ptr + v * lanes)).*;
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
                    scales[v] = @as(*align(32) const VecF, @ptrCast(scales_ptr + v * lanes)).*;
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
                    scales[v] = @as(*align(32) const VecF, @ptrCast(scales_ptr + v * lanes)).*;
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
