const types = @import("../../types.zig");
const simd = @import("simd.zig");

const BackendError = types.BackendError;
const MatMulParams = types.MatMulParams;

/// q4_0 block: 32 elements stored as 2-byte f16 scale + 16 bytes (32 nibbles)
pub const Q4_0_BLOCK_ELEMS: usize = 32;
pub const Q4_0_BLOCK_BYTES: usize = 18;

/// q8_0 block: 32 elements stored as 2-byte f16 scale + 32 bytes (int8s)
pub const Q8_0_BLOCK_ELEMS: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 34;

// -----------------------------------------------------------------------------
// Quantized GEMM tuning (CPU)
// -----------------------------------------------------------------------------
// We keep the same register tile as the f32 kernel (6x16), but pack B into a
// transposed int8 panel format so that the inner loop loads are contiguous.
//
// This keeps the math identical to the reference path:
//   dot = sum_t (A[t] * (scale * q[t]))
// i.e. A stays f32; only B is dequantized on the fly.
pub const MR: usize = 6;
pub const NR: usize = 16;

// L1/L2 blocking params (must be multiples of Q*_0_BLOCK_ELEMS == 32)
pub const KC: usize = 256;
pub const MC: usize = 144; // 24 * MR. (144*256*4 = 144KB)
pub const NC: usize = 256; // 16 * NR.

comptime {
    if (KC % Q8_0_BLOCK_ELEMS != 0) @compileError("KC must be a multiple of 32");
    if (NC % NR != 0) @compileError("NC must be a multiple of NR");
}

// Packed-B panel format (for one NR-wide column micro-panel):
// For each k-block (32 rows of B):
//   - scales: [NR]f32
//   - q:      [32][NR]i8  (transposed; contiguous NR for each t)
// Layout is identical for q8_0 and q4_0 (q4 is decoded during packing).
pub const PB_KBLOCK_BYTES: usize = NR * @sizeOf(f32) + Q8_0_BLOCK_ELEMS * NR * @sizeOf(i8);
pub const PB_PANEL_BYTES: usize = (KC / Q8_0_BLOCK_ELEMS) * PB_KBLOCK_BYTES;
pub const PB_TOTAL_BYTES: usize = (NC / NR) * PB_PANEL_BYTES;

inline fn scaleF16BitsToF32(scale_bits: u16) f32 {
    return @as(f32, @as(f16, @bitCast(scale_bits)));
}

inline fn loadScaleF32FromBlockBytes(block_ptr: [*]align(1) const u8) f32 {
    const scale_bits: u16 = @as(*align(1) const u16, @ptrCast(block_ptr)).*;
    return scaleF16BitsToF32(scale_bits);
}

// Packs A from [M, K] row-major into a contiguous [MR, KC] panel.
// Layout in packed_out is row-major with fixed stride KC.
fn packPanelA(kc: usize, mr: usize, a: []align(1) const f32, k_stride: usize, m_start: usize, k_start: usize, packed_out: *[MR * KC]f32) void {
    var r: usize = 0;
    while (r < mr) : (r += 1) {
        const a_row_off: usize = (m_start + r) * k_stride + k_start;
        const p_row_off: usize = r * KC;
        @memcpy(packed_out[p_row_off .. p_row_off + kc], a[a_row_off .. a_row_off + kc]);
    }
}

fn zeroTailCols(nr: usize, scales: []f32, q: [*]i8) void {
    if (nr >= NR) return;

    @memset(scales[nr..NR], 0.0);
    var t: usize = 0;
    while (t < Q8_0_BLOCK_ELEMS) : (t += 1) {
        const row_off: usize = t * NR;
        @memset(q[row_off + nr .. row_off + NR], 0);
    }
}

pub fn packPanelB_Q8_0(kc: usize, nr: usize, b_bytes: []const u8, n_total: usize, kb_start: usize, j_start: usize, packed_out: *[PB_PANEL_BYTES]u8) void {
    // kc is in f32 elements; quant B is in blocks of 32.
    const k_blocks: usize = kc / Q8_0_BLOCK_ELEMS;

    var kb: usize = 0;
    while (kb < k_blocks) : (kb += 1) {
        const dst_base: [*]u8 = packed_out.ptr + kb * PB_KBLOCK_BYTES;
        const scales_ptr: [*]align(32) f32 = @ptrCast(@alignCast(dst_base));
        const q_ptr: [*]i8 = @ptrCast(dst_base + NR * @sizeOf(f32));

        // Fill columns.
        var j: usize = 0;
        while (j < nr) : (j += 1) {
            const block_off: usize = ((kb_start + kb) * n_total + (j_start + j)) * Q8_0_BLOCK_BYTES;
            const block_ptr: [*]align(1) const u8 = b_bytes.ptr + block_off;
            scales_ptr[j] = loadScaleF32FromBlockBytes(block_ptr);

            const src_q: [*]align(1) const i8 = @ptrCast(block_ptr + 2);
            var t: usize = 0;
            while (t < Q8_0_BLOCK_ELEMS) : (t += 1) {
                q_ptr[t * NR + j] = src_q[t];
            }
        }

        // Zero tail columns (if any) so we can run a full NR-wide kernel.
        zeroTailCols(nr, scales_ptr[0..NR], q_ptr);
    }
}

pub fn packPanelB_Q4_0(kc: usize, nr: usize, b_bytes: []const u8, n_total: usize, kb_start: usize, j_start: usize, packed_out: *[PB_PANEL_BYTES]u8) void {
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
            var t: usize = 0;
            while (t < Q4_0_BLOCK_ELEMS) : (t += 1) {
                const byte: u8 = nib_ptr[t >> 1];
                const nib_u: u8 = if ((t & 1) == 0) (byte & 0x0F) else (byte >> 4);
                // q4_0 uses bias +8 stored in nibble.
                q_ptr[t * NR + j] = @as(i8, @intCast(nib_u)) - @as(i8, 8);
            }
        }

        zeroTailCols(nr, scales_ptr[0..NR], q_ptr);
    }
}

fn microKernel6x16_Qx0_Fast(
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
    const lanes: usize = 8;
    const VecF = @Vector(lanes, f32);
    const VecI8 = @Vector(lanes, i8);

    var acc: [MR][2]VecF = undefined;
    inline for (0..MR) |r| {
        acc[r][0] = @splat(0.0);
        acc[r][1] = @splat(0.0);
    }

    const a_ptr_base: [*]const f32 = @ptrCast(&packed_a[0]);
    const b_ptr_base: [*]const u8 = packed_b.ptr;
    const kb_count: usize = kc / Q8_0_BLOCK_ELEMS;

    var kb: usize = 0;
    while (kb < kb_count) : (kb += 1) {
        const kb_base: [*]const u8 = b_ptr_base + kb * PB_KBLOCK_BYTES;
        const scales_ptr: [*]align(32) const f32 = @ptrCast(@alignCast(kb_base));
        const scale0: VecF = @as(*align(32) const VecF, @ptrCast(scales_ptr)).*;
        const scale1: VecF = @as(*align(32) const VecF, @ptrCast(scales_ptr + lanes)).*;
        const q_ptr: [*]align(1) const i8 = @ptrCast(kb_base + NR * @sizeOf(f32));

        var t: usize = 0;
        while (t < Q8_0_BLOCK_ELEMS) : (t += 1) {
            const q_row: [*]align(1) const i8 = q_ptr + t * NR;
            const q0_i8: VecI8 = @as(*align(1) const VecI8, @ptrCast(q_row)).*;
            const q1_i8: VecI8 = @as(*align(1) const VecI8, @ptrCast(q_row + lanes)).*;
            const q0_f: VecF = @floatFromInt(q0_i8);
            const q1_f: VecF = @floatFromInt(q1_i8);
            const q0: VecF = q0_f * scale0;
            const q1: VecF = q1_f * scale1;

            inline for (0..MR) |r| {
                const a_val: f32 = a_ptr_base[r * KC + kb * Q8_0_BLOCK_ELEMS + t];
                const a_v: VecF = @splat(a_val);
                acc[r][0] = @mulAdd(VecF, a_v, q0, acc[r][0]);
                acc[r][1] = @mulAdd(VecF, a_v, q1, acc[r][1]);
            }
        }
    }

    const alpha_v: VecF = @splat(alpha);
    const beta_v: VecF = @splat(beta);

    inline for (0..MR) |r| {
        const row_off: usize = (idx_m + r) * c_stride + idx_n;
        const c_row_ptr: [*]align(1) f32 = c.ptr + row_off;

        // cols 0..7
        {
            const c_old: VecF = @as(*align(1) const VecF, @ptrCast(c_row_ptr)).*;
            const res: VecF = @mulAdd(VecF, alpha_v, acc[r][0], beta_v * c_old);
            @as(*align(1) VecF, @ptrCast(c_row_ptr)).* = res;
        }
        // cols 8..15
        {
            const c_ptr_loc: [*]align(1) f32 = c_row_ptr + lanes;
            const c_old: VecF = @as(*align(1) const VecF, @ptrCast(c_ptr_loc)).*;
            const res: VecF = @mulAdd(VecF, alpha_v, acc[r][1], beta_v * c_old);
            @as(*align(1) VecF, @ptrCast(c_ptr_loc)).* = res;
        }
    }
}

fn microKernel6x16_Qx0(
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
    const lanes: usize = 8;
    const VecF = @Vector(lanes, f32);
    const VecI8 = @Vector(lanes, i8);

    var acc: [MR][2]VecF = undefined;
    inline for (0..MR) |r| {
        acc[r][0] = @splat(0.0);
        acc[r][1] = @splat(0.0);
    }

    const a_ptr_base: [*]const f32 = @ptrCast(&packed_a[0]);
    const b_ptr_base: [*]const u8 = packed_b.ptr;
    const kb_count: usize = kc / Q8_0_BLOCK_ELEMS;

    var kb: usize = 0;
    while (kb < kb_count) : (kb += 1) {
        const kb_base: [*]const u8 = b_ptr_base + kb * PB_KBLOCK_BYTES;
        const scales_ptr: [*]align(32) const f32 = @ptrCast(@alignCast(kb_base));
        const scale0: VecF = @as(*align(32) const VecF, @ptrCast(scales_ptr)).*;
        const scale1: VecF = @as(*align(32) const VecF, @ptrCast(scales_ptr + lanes)).*;
        const q_ptr: [*]align(1) const i8 = @ptrCast(kb_base + NR * @sizeOf(f32));

        var t: usize = 0;
        while (t < Q8_0_BLOCK_ELEMS) : (t += 1) {
            const q_row: [*]align(1) const i8 = q_ptr + t * NR;
            const q0_i8: VecI8 = @as(*align(1) const VecI8, @ptrCast(q_row)).*;
            const q1_i8: VecI8 = @as(*align(1) const VecI8, @ptrCast(q_row + lanes)).*;
            const q0_f: VecF = @floatFromInt(q0_i8);
            const q1_f: VecF = @floatFromInt(q1_i8);
            const q0: VecF = q0_f * scale0;
            const q1: VecF = q1_f * scale1;

            inline for (0..MR) |r| {
                if (r < mr) {
                    const a_val: f32 = a_ptr_base[r * KC + kb * Q8_0_BLOCK_ELEMS + t];
                    const a_v: VecF = @splat(a_val);
                    acc[r][0] = @mulAdd(VecF, a_v, q0, acc[r][0]);
                    acc[r][1] = @mulAdd(VecF, a_v, q1, acc[r][1]);
                }
            }
        }
    }

    const alpha_v: VecF = @splat(alpha);
    const beta_v: VecF = @splat(beta);

    inline for (0..MR) |r| {
        if (r < mr) {
            const row_off: usize = (idx_m + r) * c_stride + idx_n;

            // cols 0..7
            if (nr > 0) {
                const c_ptr_loc: [*]align(1) f32 = c.ptr + row_off;
                if (nr >= lanes) {
                    const c_old: VecF = @as(*align(1) const VecF, @ptrCast(c_ptr_loc)).*;
                    const res: VecF = @mulAdd(VecF, alpha_v, acc[r][0], beta_v * c_old);
                    @as(*align(1) VecF, @ptrCast(c_ptr_loc)).* = res;
                } else {
                    const res_arr: [lanes]f32 = acc[r][0];
                    var vi: usize = 0;
                    while (vi < nr) : (vi += 1) {
                        c[row_off + vi] = alpha * res_arr[vi] + beta * c[row_off + vi];
                    }
                }
            }

            // cols 8..15
            if (nr > lanes) {
                const c_ptr_loc: [*]align(1) f32 = c.ptr + row_off + lanes;
                const rem: usize = nr - lanes;
                if (rem >= lanes) {
                    const c_old: VecF = @as(*align(1) const VecF, @ptrCast(c_ptr_loc)).*;
                    const res: VecF = @mulAdd(VecF, alpha_v, acc[r][1], beta_v * c_old);
                    @as(*align(1) VecF, @ptrCast(c_ptr_loc)).* = res;
                } else {
                    const res_arr: [lanes]f32 = acc[r][1];
                    var vi: usize = 0;
                    while (vi < rem) : (vi += 1) {
                        c[row_off + lanes + vi] = alpha * res_arr[vi] + beta * c[row_off + lanes + vi];
                    }
                }
            }
        }
    }
}

/// C[M,N] (f32) = A[M,K] (f32) @ B[K,N] (q4_0)
///
/// B is stored as a row-major array of blocks with shape [k_blocks, n].
pub fn matmulQ4_0(params: MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) BackendError!void {
    const m: usize = params.m;
    const n: usize = params.n;
    const k: usize = params.k;
    const alpha: f32 = params.alpha;
    const beta: f32 = params.beta;

    if (k % Q4_0_BLOCK_ELEMS != 0) return BackendError.InvalidArgument;
    const k_blocks: usize = k / Q4_0_BLOCK_ELEMS;

    const c: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, c_bytes);
    const a: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, a_bytes);

    if (c.len < m * n or a.len < m * k) return BackendError.InvalidArgument;
    if (b_bytes.len < k_blocks * n * Q4_0_BLOCK_BYTES) return BackendError.InvalidArgument;

    // Cache-blocked + packed panels.
    // L2-resident buffers (~220KB total)
    var packed_b: [PB_TOTAL_BYTES]u8 align(32) = undefined;
    var packed_a: [MC * KC]f32 align(32) = undefined;

    var jc: usize = 0;
    while (jc < n) : (jc += NC) {
        const nc: usize = @min(NC, n - jc);

        var kk: usize = 0;
        while (kk < k) : (kk += KC) {
            const kc: usize = @min(KC, k - kk);
            const beta_eff: f32 = if (kk == 0) beta else 1.0;
            // kc must stay a multiple of 32 due to quant blocks.
            if (kc % Q4_0_BLOCK_ELEMS != 0) return BackendError.InvalidArgument;
            const kb_start: usize = kk / Q4_0_BLOCK_ELEMS;

            // 1. Pack B (decode q4_0 -> int8 transposed)
            var jr: usize = 0;
            while (jr < nc) : (jr += NR) {
                const nr: usize = @min(NR, nc - jr);
                const panel_idx: usize = jr / NR;
                const offset: usize = panel_idx * PB_PANEL_BYTES;
                const dest: *[PB_PANEL_BYTES]u8 = @ptrCast(packed_b[offset..][0..PB_PANEL_BYTES]);
                packPanelB_Q4_0(kc, nr, b_bytes, n, kb_start, jc + jr, dest);
            }

            // 2. Loop Ic
            var ic: usize = 0;
            while (ic < m) : (ic += MC) {
                const mc: usize = @min(MC, m - ic);

                // Pack A once per (ic, kk)
                var ir: usize = 0;
                while (ir < mc) : (ir += MR) {
                    const mr: usize = @min(MR, mc - ir);
                    const panel_idx: usize = ir / MR;
                    const a_off: usize = panel_idx * (MR * KC);
                    const dest_a: *[MR * KC]f32 = @ptrCast(packed_a[a_off..][0 .. MR * KC]);
                    packPanelA(kc, mr, a, k, ic + ir, kk, dest_a);
                }

                // 3. Macro-kernel
                var jr_ex: usize = 0;
                while (jr_ex < nc) : (jr_ex += NR) {
                    const nr: usize = @min(NR, nc - jr_ex);
                    const b_off: usize = (jr_ex / NR) * PB_PANEL_BYTES;
                    const b_panel: *const [PB_PANEL_BYTES]u8 = @ptrCast(packed_b[b_off..][0..PB_PANEL_BYTES]);

                    var ir_ex: usize = 0;
                    while (ir_ex < mc) : (ir_ex += MR) {
                        const mr: usize = @min(MR, mc - ir_ex);
                        const a_off: usize = (ir_ex / MR) * (MR * KC);
                        const a_panel: *const [MR * KC]f32 = @ptrCast(&packed_a[a_off]);

                        if (mr == MR and nr == NR) {
                            microKernel6x16_Qx0_Fast(kc, alpha, beta_eff, a_panel, b_panel, c, n, ic + ir_ex, jc + jr_ex);
                        } else {
                            microKernel6x16_Qx0(kc, alpha, beta_eff, a_panel, b_panel, c, n, ic + ir_ex, jc + jr_ex, mr, nr);
                        }
                    }
                }
            }
        }
    }
}

/// C[M,N] (f32) = A[M,K] (f32) @ B[K,N] (q8_0)
///
/// B is stored as a row-major array of blocks with shape [k_blocks, n].
pub fn matmulQ8_0(params: MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) BackendError!void {
    const m: usize = params.m;
    const n: usize = params.n;
    const k: usize = params.k;
    const alpha: f32 = params.alpha;
    const beta: f32 = params.beta;

    if (k % Q8_0_BLOCK_ELEMS != 0) return BackendError.InvalidArgument;
    const k_blocks: usize = k / Q8_0_BLOCK_ELEMS;

    const c: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, c_bytes);
    const a: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, a_bytes);

    if (c.len < m * n or a.len < m * k) return BackendError.InvalidArgument;
    if (b_bytes.len < k_blocks * n * Q8_0_BLOCK_BYTES) return BackendError.InvalidArgument;

    // Cache-blocked + packed panels.
    var packed_b: [PB_TOTAL_BYTES]u8 align(32) = undefined;
    var packed_a: [MC * KC]f32 align(32) = undefined;

    var jc: usize = 0;
    while (jc < n) : (jc += NC) {
        const nc: usize = @min(NC, n - jc);

        var kk: usize = 0;
        while (kk < k) : (kk += KC) {
            const kc: usize = @min(KC, k - kk);
            const beta_eff: f32 = if (kk == 0) beta else 1.0;
            if (kc % Q8_0_BLOCK_ELEMS != 0) return BackendError.InvalidArgument;
            const kb_start: usize = kk / Q8_0_BLOCK_ELEMS;

            // 1. Pack B
            var jr: usize = 0;
            while (jr < nc) : (jr += NR) {
                const nr: usize = @min(NR, nc - jr);
                const panel_idx: usize = jr / NR;
                const offset: usize = panel_idx * PB_PANEL_BYTES;
                const dest: *[PB_PANEL_BYTES]u8 = @ptrCast(packed_b[offset..][0..PB_PANEL_BYTES]);
                packPanelB_Q8_0(kc, nr, b_bytes, n, kb_start, jc + jr, dest);
            }

            // 2. Loop Ic
            var ic: usize = 0;
            while (ic < m) : (ic += MC) {
                const mc: usize = @min(MC, m - ic);

                // Pack A
                var ir: usize = 0;
                while (ir < mc) : (ir += MR) {
                    const mr: usize = @min(MR, mc - ir);
                    const panel_idx: usize = ir / MR;
                    const a_off: usize = panel_idx * (MR * KC);
                    const dest_a: *[MR * KC]f32 = @ptrCast(packed_a[a_off..][0 .. MR * KC]);
                    packPanelA(kc, mr, a, k, ic + ir, kk, dest_a);
                }

                // 3. Macro-kernel
                var jr_ex: usize = 0;
                while (jr_ex < nc) : (jr_ex += NR) {
                    const nr: usize = @min(NR, nc - jr_ex);
                    const b_off: usize = (jr_ex / NR) * PB_PANEL_BYTES;
                    const b_panel: *const [PB_PANEL_BYTES]u8 = @ptrCast(packed_b[b_off..][0..PB_PANEL_BYTES]);

                    var ir_ex: usize = 0;
                    while (ir_ex < mc) : (ir_ex += MR) {
                        const mr: usize = @min(MR, mc - ir_ex);
                        const a_off: usize = (ir_ex / MR) * (MR * KC);
                        const a_panel: *const [MR * KC]f32 = @ptrCast(&packed_a[a_off]);

                        if (mr == MR and nr == NR) {
                            microKernel6x16_Qx0_Fast(kc, alpha, beta_eff, a_panel, b_panel, c, n, ic + ir_ex, jc + jr_ex);
                        } else {
                            microKernel6x16_Qx0(kc, alpha, beta_eff, a_panel, b_panel, c, n, ic + ir_ex, jc + jr_ex, mr, nr);
                        }
                    }
                }
            }
        }
    }
}
