const std = @import("std");
const types = @import("../../types.zig");
const simd = @import("simd.zig");
const quant = @import("quant.zig");

const BackendError = types.BackendError;
const MatMulParams = types.MatMulParams;

// Tuning constants for f32
// Micro-kernel tiles
const MR = 6;
const NR = 16;
// L1/L2 blocking params
const KC = 256;
const MC = 144; // 24 * MR. Fits A-panel in L2 (144*256*4 = 144KB)
const NC = 256; // 16 * NR. Fits B-panel in L2 (256*256*4 = 256KB)

/// C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C[M,N]
///
/// Optimized with cache blocking (GotoBLAS-like) and register tiling (6x16).
pub fn matmulF32(params: MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) BackendError!void {
    const m: usize = params.m;
    const n: usize = params.n;
    const k: usize = params.k;
    const alpha: f32 = params.alpha;
    const beta: f32 = params.beta;

    const c: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, c_bytes);
    const a: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, a_bytes);
    const b: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, b_bytes);

    if (c.len < m * n or a.len < m * k or b.len < k * n) {
        return BackendError.InvalidArgument;
    }

    // L2-resident buffers (~400KB total)
    // Layout: Sequence of micro-panels.
    var packed_b: [KC * NC]f32 align(32) = undefined;
    var packed_a: [MC * KC]f32 align(32) = undefined;

    // Loop Jc (Macro-Column of C)
    var jc: usize = 0;
    while (jc < n) : (jc += NC) {
        const nc = @min(NC, n - jc);

        // Loop K (Chunks of KC)
        var kk: usize = 0;
        while (kk < k) : (kk += KC) {
            const kc = @min(KC, k - kk);
            const beta_eff = if (kk == 0) beta else 1.0;

            // 1. Pack Panel B (Macro)
            // Fill packed_b with `nc` / `NR` micro-panels.
            var jr: usize = 0;
            while (jr < nc) : (jr += NR) {
                const nr = @min(NR, nc - jr);
                // Pointer math to find slice for this panel
                const panel_idx = jr / NR; // Assumes jr is multiple of NR (it is)
                const offset = panel_idx * (KC * NR);
                // Cast to strictly-sized array pointer for the pack function
                const dest: *[KC * NR]f32 = @ptrCast(packed_b[offset..][0 .. KC * NR]);
                packPanelB(kc, nr, b, n, kk, jc + jr, dest);
            }

            // Loop Ic (Macro-Row of C)
            var ic: usize = 0;
            while (ic < m) : (ic += MC) {
                const mc = @min(MC, m - ic);

                // 2. Pack Panel A (Macro)
                // Reuse this packed A for all columns in current Jc block
                var ir: usize = 0;
                while (ir < mc) : (ir += MR) {
                    const mr = @min(MR, mc - ir);
                    const panel_idx = ir / MR;
                    const offset = panel_idx * (MR * KC);
                    const dest: *[MR * KC]f32 = @ptrCast(packed_a[offset..][0 .. MR * KC]);
                    packPanelA(kc, mr, a, k, ic + ir, kk, dest);
                }

                // 3. Macro-Kernel
                // Iterate over the packed buffers
                var jr_ex: usize = 0;
                while (jr_ex < nc) : (jr_ex += NR) {
                    const nr = @min(NR, nc - jr_ex);
                    const b_offset = (jr_ex / NR) * (KC * NR);
                    const b_panel: *const [KC * NR]f32 = @ptrCast(&packed_b[b_offset]);

                    var ir_ex: usize = 0;
                    while (ir_ex < mc) : (ir_ex += MR) {
                        const mr = @min(MR, mc - ir_ex);
                        const a_offset = (ir_ex / MR) * (MR * KC);
                        const a_panel: *const [MR * KC]f32 = @ptrCast(&packed_a[a_offset]);

                        // Determine global indices for C update
                        if (mr == MR and nr == NR) {
                            microKernel6x16_Fast(kc, alpha, beta_eff, a_panel, b_panel, c, n, ic + ir_ex, jc + jr_ex);
                        } else {
                            microKernel6x16(kc, alpha, beta_eff, a_panel, b_panel, c, n, ic + ir_ex, jc + jr_ex, mr, nr);
                        }
                    }
                }
            }
        }
    }
}

// Fast path micro-kernel: assumes full 6x16 block.
// No bounds checks loop-invariant branches.
fn microKernel6x16_Fast(
    kc: usize,
    alpha: f32,
    beta: f32,
    packed_a: *const [MR * KC]f32,
    packed_b: *const [KC * NR]f32,
    c: []align(1) f32,
    c_stride: usize,
    idx_m: usize,
    idx_n: usize,
) void {
    @setRuntimeSafety(false);
    const lanes = 8;
    const Vec = @Vector(lanes, f32);

    var acc: [MR][2]Vec = undefined;
    inline for (0..MR) |r| {
        acc[r][0] = @splat(0.0);
        acc[r][1] = @splat(0.0);
    }

    const a_ptr_base: [*]const f32 = @ptrCast(&packed_a[0]);
    const b_ptr_base: [*]const u8 = @ptrCast(&packed_b[0]);

    // Main K loop unrolled by 8
    var k: usize = 0;
    while (k + 8 <= kc) : (k += 8) {
        // Explicit prefetch for next iteration's data
        // L1 prefetch (T0).
        // 8 k-steps * 16 floats * 4 bytes = 512 bytes ahead.
        // We just prefetch somewhat ahead.
        const prefetch_dist = 512;
        @prefetch(b_ptr_base + k * NR * 4 + prefetch_dist, .{ .rw = .read, .locality = 3, .cache = .data });

        // Unrolling helps pipeline FMA and Loads
        inline for (0..8) |uk| {
            const b_off: usize = (k + uk) * NR;
            const b0_bytes: usize = b_off * @sizeOf(f32);
            const b1_bytes: usize = (b_off + lanes) * @sizeOf(f32);
            const b_vec0: Vec = @as(*align(32) const Vec, @ptrCast(@alignCast(b_ptr_base + b0_bytes))).*;
            const b_vec1: Vec = @as(*align(32) const Vec, @ptrCast(@alignCast(b_ptr_base + b1_bytes))).*;

            inline for (0..MR) |r| {
                const a_val: f32 = a_ptr_base[r * KC + (k + uk)];
                const a_vec: Vec = @splat(a_val);
                acc[r][0] = @mulAdd(Vec, a_vec, b_vec0, acc[r][0]);
                acc[r][1] = @mulAdd(Vec, a_vec, b_vec1, acc[r][1]);
            }
        }
    }

    // Handle remaining K
    while (k < kc) : (k += 1) {
        const b_off: usize = k * NR;
        const b0_bytes: usize = b_off * @sizeOf(f32);
        const b1_bytes: usize = (b_off + lanes) * @sizeOf(f32);
        const b_vec0: Vec = @as(*align(32) const Vec, @ptrCast(@alignCast(b_ptr_base + b0_bytes))).*;
        const b_vec1: Vec = @as(*align(32) const Vec, @ptrCast(@alignCast(b_ptr_base + b1_bytes))).*;

        inline for (0..MR) |r| {
            const a_val: f32 = a_ptr_base[r * KC + k];
            const a_vec: Vec = @splat(a_val); // broadcast
            acc[r][0] = @mulAdd(Vec, a_vec, b_vec0, acc[r][0]);
            acc[r][1] = @mulAdd(Vec, a_vec, b_vec1, acc[r][1]);
        }
    }

    // Write back
    const alpha_v: Vec = @splat(alpha);
    const beta_v: Vec = @splat(beta);

    inline for (0..MR) |r| {
        const row_off = (idx_m + r) * c_stride + idx_n;
        const c_row_ptr = c.ptr + row_off;

        // Vec 0
        {
            const c_ptr_loc = c_row_ptr;
            const c_old: Vec = @as(*align(1) const Vec, @ptrCast(c_ptr_loc)).*;
            const res = @mulAdd(Vec, alpha_v, acc[r][0], beta_v * c_old);
            @as(*align(1) Vec, @ptrCast(c_ptr_loc)).* = res;
        }
        // Vec 1
        {
            const c_ptr_loc = c_row_ptr + lanes;
            const c_old: Vec = @as(*align(1) const Vec, @ptrCast(c_ptr_loc)).*;
            const res = @mulAdd(Vec, alpha_v, acc[r][1], beta_v * c_old);
            @as(*align(1) Vec, @ptrCast(c_ptr_loc)).* = res;
        }
    }
}

// Micro-kernel: computes C_sub += alpha * A_sub * PackedB
// Uses 6x16 register blocking.
fn microKernel6x16(
    kc: usize,
    alpha: f32,
    beta: f32,
    packed_a: *const [MR * KC]f32,
    packed_b: *const [KC * NR]f32,
    c: []align(1) f32,
    c_stride: usize,
    idx_m: usize,
    idx_n: usize,
    mr: usize,
    nr: usize,
) void {
    @setRuntimeSafety(false);
    const lanes = 8; // Fixed for 6x16 kernel logic
    const Vec = @Vector(lanes, f32);

    // Accumulators for 6 rows, 2 vectors (16 elements) per row.
    var acc: [MR][2]Vec = undefined;

    // Initialize accumulators to 0
    inline for (0..MR) |r| {
        inline for (0..2) |v| {
            acc[r][v] = @splat(0.0);
        }
    }

    const a_ptr_base: [*]const f32 = @ptrCast(&packed_a[0]);
    const b_ptr_base: [*]const u8 = @ptrCast(&packed_b[0]);

    // Hot loop over k
    var k: usize = 0;
    while (k < kc) : (k += 1) {
        const b_off: usize = k * NR;
        const b0_bytes: usize = b_off * @sizeOf(f32);
        const b1_bytes: usize = (b_off + lanes) * @sizeOf(f32);
        const b_vec0: Vec = @as(*align(32) const Vec, @ptrCast(@alignCast(b_ptr_base + b0_bytes))).*;
        const b_vec1: Vec = @as(*align(32) const Vec, @ptrCast(@alignCast(b_ptr_base + b1_bytes))).*;

        inline for (0..MR) |r| {
            if (r < mr) {
                const a_val: f32 = a_ptr_base[r * KC + k];
                const a_vec: Vec = @splat(a_val);

                acc[r][0] = @mulAdd(Vec, a_vec, b_vec0, acc[r][0]);
                acc[r][1] = @mulAdd(Vec, a_vec, b_vec1, acc[r][1]);
            }
        }
    }

    // Write back C
    const alpha_v: Vec = @splat(alpha);
    const beta_v: Vec = @splat(beta);

    inline for (0..MR) |r| {
        if (r < mr) {
            const row_off = (idx_m + r) * c_stride + idx_n;

            // Vector 0 (cols 0..7)
            if (0 < nr) {
                const c_ptr_loc = c.ptr + row_off;

                if (nr >= lanes) {
                    const c_old: Vec = @as(*align(1) const Vec, @ptrCast(c_ptr_loc)).*;
                    const res = @mulAdd(Vec, alpha_v, acc[r][0], beta_v * c_old);
                    @as(*align(1) Vec, @ptrCast(c_ptr_loc)).* = res;
                } else {
                    const res_arr: [lanes]f32 = acc[r][0];
                    for (0..nr) |vi| {
                        c[row_off + vi] = alpha * res_arr[vi] + beta * c[row_off + vi];
                    }
                }
            }

            // Vector 1 (cols 8..15)
            if (lanes < nr) {
                const c_ptr_loc = c.ptr + row_off + lanes;
                const rem = nr - lanes;

                if (rem >= lanes) {
                    const c_old: Vec = @as(*align(1) const Vec, @ptrCast(c_ptr_loc)).*;
                    const res = @mulAdd(Vec, alpha_v, acc[r][1], beta_v * c_old);
                    @as(*align(1) Vec, @ptrCast(c_ptr_loc)).* = res;
                } else {
                    const res_arr: [lanes]f32 = acc[r][1];
                    for (0..rem) |vi| {
                        c[row_off + lanes + vi] = alpha * res_arr[vi] + beta * c[row_off + lanes + vi];
                    }
                }
            }
        }
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
    // Only copy the rows that exist.
    var r: usize = 0;
    while (r < mr) : (r += 1) {
        const a_row_off: usize = (m_start + r) * k_stride + k_start;
        const p_row_off: usize = r * KC;
        @memcpy(packed_out[p_row_off .. p_row_off + kc], a[a_row_off .. a_row_off + kc]);
    }
    // We do not need to zero-pad unused rows or the KC tail because kernels only read:
    // - rows < mr
    // - k < kc
}

// Packs B from [K, N] layout to [KC, NR] contiguous buffer.
fn packPanelB(kc: usize, nr: usize, b: []align(1) const f32, n_total: usize, k_start: usize, j_start: usize, packed_out: *[KC * NR]f32) void {
    var k: usize = 0;
    while (k < kc) : (k += 1) {
        // var j: usize = 0; // Unused
        const b_row_start = (k_start + k) * n_total + j_start;
        const p_row_start = k * NR;

        const copy_n = nr;
        @memcpy(packed_out[p_row_start .. p_row_start + copy_n], b[b_row_start .. b_row_start + copy_n]);

        if (copy_n < NR) {
            @memset(packed_out[p_row_start + copy_n .. p_row_start + NR], 0.0);
        }
    }
}

/// C[M,N] (f16) = alpha * A[M,K] (f16) @ B[K,N] (f16) + beta * C
pub fn matmulF16(params: MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) BackendError!void {
    const m: usize = params.m;
    const n: usize = params.n;
    const k: usize = params.k;
    const alpha: f32 = params.alpha;
    const beta: f32 = params.beta;

    const c: []align(1) f16 = simd.bytesAsSliceMutUnaligned(f16, c_bytes);
    const a: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, a_bytes);
    const b: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, b_bytes);

    if (c.len < m * n or a.len < m * k or b.len < k * n) {
        return BackendError.InvalidArgument;
    }

    for (0..m) |i| {
        for (0..n) |j| {
            var acc: f32 = 0.0;
            for (0..k) |kk| {
                acc += @as(f32, a[i * k + kk]) * @as(f32, b[kk * n + j]);
            }
            const c_idx: usize = i * n + j;
            const old_c: f32 = @as(f32, c[c_idx]);
            c[c_idx] = @floatCast(alpha * acc + beta * old_c);
        }
    }
}

/// C[M,N] (f32) = alpha * A[M,K] (f16) @ B[K,N] (f16) + beta * C
pub fn matmulF16ToF32(params: MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) BackendError!void {
    const m: usize = params.m;
    const n: usize = params.n;
    const k: usize = params.k;
    const alpha: f32 = params.alpha;
    const beta: f32 = params.beta;

    const c: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, c_bytes);
    const a: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, a_bytes);
    const b: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, b_bytes);

    if (c.len < m * n or a.len < m * k or b.len < k * n) {
        return BackendError.InvalidArgument;
    }

    for (0..m) |i| {
        for (0..n) |j| {
            var acc: f32 = 0.0;
            for (0..k) |kk| {
                acc += @as(f32, a[i * k + kk]) * @as(f32, b[kk * n + j]);
            }
            const c_idx: usize = i * n + j;
            c[c_idx] = alpha * acc + beta * c[c_idx];
        }
    }
}

pub fn matmulQ4_0(params: MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) BackendError!void {
    return quant.matmulQ4_0(params, c_bytes, a_bytes, b_bytes);
}

pub fn matmulQ8_0(params: MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) BackendError!void {
    return quant.matmulQ8_0(params, c_bytes, a_bytes, b_bytes);
}
