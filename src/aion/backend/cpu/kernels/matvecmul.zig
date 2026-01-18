const types = @import("../../types.zig");
const simd = @import("simd.zig");
const quant = @import("quant.zig");

const BackendError = types.BackendError;
const MatMulParams = types.MatMulParams;

// Targeted at transformer-style GEMV: M == 1.
//
// We intentionally share the same N micro-tile as the GEMM kernels:
// - NR = 16 (two 8-wide vectors on x86_64 AVX2-ish)
//
// For f32 matvec we do NOT pack; we block over N to keep B rows contiguous.
// For quant matvec we reuse quant's packed-B panel format, but skip A packing.
const NR: usize = 16;
const NC: usize = 256; // match GEMM's NC; good cache/blocking default.
const LANES: usize = 8;

comptime {
    if (NR != 2 * LANES) @compileError("NR must equal 2*LANES");
}

pub fn matvecF32(params: MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) BackendError!void {
    return matvecF32Range(params, 0, params.n, c_bytes, a_bytes, b_bytes);
}

/// Compute a column subrange for M==1:
///   C[0, col_start..col_start+col_count] = alpha * A[0,:] @ B[:,col_range] + beta * C[0,col_range]
///
/// Preconditions:
/// - params.m == 1
/// - A=f32, B=f32, C=f32
/// - B is row-major contiguous KxN (stride N)
pub fn matvecF32Range(
    params: MatMulParams,
    col_start: usize,
    col_count: usize,
    c_bytes: []u8,
    a_bytes: []const u8,
    b_bytes: []const u8,
) BackendError!void {
    if (params.m != 1) return BackendError.InvalidArgument;
    const n_total: usize = params.n;
    const k: usize = params.k;
    if (col_start > n_total or col_count > n_total - col_start) return BackendError.InvalidArgument;

    const c_all: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, c_bytes);
    const a_all: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, a_bytes);
    const b_all: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, b_bytes);

    if (c_all.len < n_total) return BackendError.InvalidArgument;
    if (a_all.len < k) return BackendError.InvalidArgument;
    if (b_all.len < k * n_total) return BackendError.InvalidArgument;

    const alpha: f32 = params.alpha;
    const beta: f32 = params.beta;

    const VecF = @Vector(LANES, f32);
    const alpha_v: VecF = @splat(alpha);
    const beta_v: VecF = @splat(beta);

    // Block over N (columns) for cache locality.
    var jc: usize = 0;
    while (jc < col_count) : (jc += NC) {
        const nc: usize = @min(NC, col_count - jc);
        const base_col: usize = col_start + jc;

        var jr: usize = 0;
        while (jr < nc) : (jr += NR) {
            const nr: usize = @min(NR, nc - jr);
            const j0: usize = base_col + jr;

            var acc0: VecF = @splat(0.0);
            var acc1: VecF = @splat(0.0);

            // Main dot loop.
            // Bounds are checked above; keep runtime safety off in the hot loop.
            @setRuntimeSafety(false);
            var kk: usize = 0;
            while (kk < k) : (kk += 1) {
                const a_val: f32 = a_all[kk];
                const a_v: VecF = @splat(a_val);

                const b_row_off: usize = kk * n_total + j0;
                const b_ptr: [*]align(1) const f32 = b_all.ptr + b_row_off;

                // Prefetch ahead in B along K.
                if (kk + 4 < k) {
                    const pf_off: usize = (kk + 4) * n_total + j0;
                    @prefetch(@as([*]const u8, @ptrCast(b_all.ptr + pf_off)), .{ .rw = .read, .locality = 3, .cache = .data });
                }

                // First 8 columns.
                if (nr >= LANES) {
                    const b0: VecF = @as(*align(1) const VecF, @ptrCast(b_ptr)).*;
                    acc0 = @mulAdd(VecF, a_v, b0, acc0);
                } else {
                    var tmp: [LANES]f32 = @splat(0.0);
                    var i: usize = 0;
                    while (i < nr) : (i += 1) {
                        tmp[i] = b_ptr[i];
                    }
                    const b0: VecF = tmp;
                    acc0 = @mulAdd(VecF, a_v, b0, acc0);
                }

                // Second 8 columns.
                if (nr > LANES) {
                    const rem: usize = nr - LANES;
                    if (rem >= LANES) {
                        const b1: VecF = @as(*align(1) const VecF, @ptrCast(b_ptr + LANES)).*;
                        acc1 = @mulAdd(VecF, a_v, b1, acc1);
                    } else {
                        var tmp1: [LANES]f32 = @splat(0.0);
                        var ii: usize = 0;
                        while (ii < rem) : (ii += 1) {
                            tmp1[ii] = b_ptr[LANES + ii];
                        }
                        const b1: VecF = tmp1;
                        acc1 = @mulAdd(VecF, a_v, b1, acc1);
                    }
                }
            }
            @setRuntimeSafety(true);

            // Write back.
            const c_ptr: [*]align(1) f32 = c_all.ptr + j0;

            if (nr >= LANES) {
                const old0: VecF = @as(*align(1) const VecF, @ptrCast(c_ptr)).*;
                const res0: VecF = if (beta == 0.0) (alpha_v * acc0) else @mulAdd(VecF, alpha_v, acc0, beta_v * old0);
                @as(*align(1) VecF, @ptrCast(c_ptr)).* = res0;
            } else {
                const res_arr0: [LANES]f32 = acc0;
                var i: usize = 0;
                while (i < nr) : (i += 1) {
                    const old: f32 = c_all[j0 + i];
                    c_all[j0 + i] = alpha * res_arr0[i] + beta * old;
                }
            }

            if (nr > LANES) {
                const rem: usize = nr - LANES;
                if (rem >= LANES) {
                    const old1: VecF = @as(*align(1) const VecF, @ptrCast(c_ptr + LANES)).*;
                    const res1: VecF = if (beta == 0.0) (alpha_v * acc1) else @mulAdd(VecF, alpha_v, acc1, beta_v * old1);
                    @as(*align(1) VecF, @ptrCast(c_ptr + LANES)).* = res1;
                } else {
                    const res_arr1: [LANES]f32 = acc1;
                    var ii: usize = 0;
                    while (ii < rem) : (ii += 1) {
                        const idx: usize = j0 + LANES + ii;
                        const old: f32 = c_all[idx];
                        c_all[idx] = alpha * res_arr1[ii] + beta * old;
                    }
                }
            }
        }
    }
}

pub fn matvecQ8_0(params: MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) BackendError!void {
    return matvecQ8_0Range(params, 0, params.n, c_bytes, a_bytes, b_bytes);
}

pub fn matvecQ4_0(params: MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) BackendError!void {
    return matvecQ4_0Range(params, 0, params.n, c_bytes, a_bytes, b_bytes);
}

pub fn matvecQ8_0Range(
    params: MatMulParams,
    col_start: usize,
    col_count: usize,
    c_bytes: []u8,
    a_bytes: []const u8,
    b_bytes: []const u8,
) BackendError!void {
    return matvecQx0Range(.q8_0, params, col_start, col_count, c_bytes, a_bytes, b_bytes);
}

pub fn matvecQ4_0Range(
    params: MatMulParams,
    col_start: usize,
    col_count: usize,
    c_bytes: []u8,
    a_bytes: []const u8,
    b_bytes: []const u8,
) BackendError!void {
    return matvecQx0Range(.q4_0, params, col_start, col_count, c_bytes, a_bytes, b_bytes);
}

const QuantKind = enum { q4_0, q8_0 };

fn matvecQx0Range(
    kind: QuantKind,
    params: MatMulParams,
    col_start: usize,
    col_count: usize,
    c_bytes: []u8,
    a_bytes: []const u8,
    b_bytes: []const u8,
) BackendError!void {
    if (params.m != 1) return BackendError.InvalidArgument;

    const n_total: usize = params.n;
    const k: usize = params.k;

    if (col_start > n_total or col_count > n_total - col_start) return BackendError.InvalidArgument;

    const be: usize = 32;
    if ((k % be) != 0) return BackendError.InvalidArgument;

    const k_blocks: usize = k / be;
    const block_bytes: usize = switch (kind) {
        .q8_0 => quant.Q8_0_BLOCK_BYTES,
        .q4_0 => quant.Q4_0_BLOCK_BYTES,
    };
    if (b_bytes.len < k_blocks * n_total * block_bytes) return BackendError.InvalidArgument;

    const c_all: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, c_bytes);
    const a_all: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, a_bytes);
    if (c_all.len < n_total) return BackendError.InvalidArgument;
    if (a_all.len < k) return BackendError.InvalidArgument;

    const alpha: f32 = params.alpha;
    const beta: f32 = params.beta;

    // Follow the quant GEMM kernel's KC=256 chunking.
    const KC: usize = 256;
    if (KC % be != 0) return BackendError.InvalidArgument;

    const VecF = @Vector(LANES, f32);
    const VecI8 = @Vector(LANES, i8);

    var jc: usize = 0;
    while (jc < col_count) : (jc += NC) {
        const nc: usize = @min(NC, col_count - jc);
        const base_col: usize = col_start + jc;

        var kk: usize = 0;
        while (kk < k) : (kk += KC) {
            const kc: usize = @min(KC, k - kk);
            if ((kc % be) != 0) return BackendError.InvalidArgument;

            const kb_start: usize = kk / be;
            const beta_eff: f32 = if (kk == 0) beta else 1.0;

            var jr: usize = 0;
            while (jr < nc) : (jr += NR) {
                const nr: usize = @min(NR, nc - jr);
                const j0: usize = base_col + jr;

                var packed_panel: [quant.PB_PANEL_BYTES]u8 align(32) = undefined;
                switch (kind) {
                    .q8_0 => quant.packPanelB_Q8_0(kc, nr, b_bytes, n_total, kb_start, j0, &packed_panel),
                    .q4_0 => quant.packPanelB_Q4_0(kc, nr, b_bytes, n_total, kb_start, j0, &packed_panel),
                }

                microKernel1x16_Qx0(kc, alpha, beta_eff, a_all[kk..].ptr, &packed_panel, c_all.ptr, n_total, j0, nr, VecF, VecI8);
            }
        }
    }
}

fn microKernel1x16_Qx0(
    kc: usize,
    alpha: f32,
    beta: f32,
    a_ptr_base: [*]align(1) const f32,
    packed_b: *const [quant.PB_PANEL_BYTES]u8,
    c_ptr_base: [*]align(1) f32,
    c_stride: usize,
    idx_n: usize,
    nr: usize,
    comptime VecF: type,
    comptime VecI8: type,
) void {
    _ = c_stride; // row stride is not used for M==1.

    @setRuntimeSafety(false);

    var acc0: VecF = @splat(0.0);
    var acc1: VecF = @splat(0.0);

    const b_ptr_base: [*]const u8 = packed_b.ptr;
    const kb_count: usize = kc / quant.Q8_0_BLOCK_ELEMS;

    var kb: usize = 0;
    while (kb < kb_count) : (kb += 1) {
        const kb_base: [*]const u8 = b_ptr_base + kb * quant.PB_KBLOCK_BYTES;
        const scales_ptr: [*]align(32) const f32 = @ptrCast(@alignCast(kb_base));
        const scale0: VecF = @as(*align(32) const VecF, @ptrCast(scales_ptr)).*;
        const scale1: VecF = @as(*align(32) const VecF, @ptrCast(scales_ptr + LANES)).*;
        const q_ptr: [*]align(1) const i8 = @ptrCast(kb_base + NR * @sizeOf(f32));

        var t: usize = 0;
        while (t < quant.Q8_0_BLOCK_ELEMS) : (t += 1) {
            const q_row: [*]align(1) const i8 = q_ptr + t * NR;
            const q0_i8: VecI8 = @as(*align(1) const VecI8, @ptrCast(q_row)).*;
            const q1_i8: VecI8 = @as(*align(1) const VecI8, @ptrCast(q_row + LANES)).*;

            const q0_f: VecF = @floatFromInt(q0_i8);
            const q1_f: VecF = @floatFromInt(q1_i8);

            const q0: VecF = q0_f * scale0;
            const q1: VecF = q1_f * scale1;

            const a_val: f32 = a_ptr_base[kb * quant.Q8_0_BLOCK_ELEMS + t];
            const a_v: VecF = @splat(a_val);

            acc0 = @mulAdd(VecF, a_v, q0, acc0);
            acc1 = @mulAdd(VecF, a_v, q1, acc1);
        }
    }

    const alpha_v: VecF = @splat(alpha);
    const beta_v: VecF = @splat(beta);

    const c_row_ptr: [*]align(1) f32 = @ptrCast(c_ptr_base + idx_n);

    // cols 0..7
    if (nr > 0) {
        if (nr >= LANES) {
            const c_old: VecF = @as(*align(1) const VecF, @ptrCast(c_row_ptr)).*;
            const res: VecF = @mulAdd(VecF, alpha_v, acc0, beta_v * c_old);
            @as(*align(1) VecF, @ptrCast(c_row_ptr)).* = res;
        } else {
            const res_arr: [LANES]f32 = acc0;
            var i: usize = 0;
            while (i < nr) : (i += 1) {
                const idx: usize = idx_n + i;
                c_ptr_base[idx] = alpha * res_arr[i] + beta * c_ptr_base[idx];
            }
        }
    }

    // cols 8..15
    if (nr > LANES) {
        const rem: usize = nr - LANES;
        const c_ptr_loc: [*]align(1) f32 = c_row_ptr + LANES;
        if (rem >= LANES) {
            const c_old: VecF = @as(*align(1) const VecF, @ptrCast(c_ptr_loc)).*;
            const res: VecF = @mulAdd(VecF, alpha_v, acc1, beta_v * c_old);
            @as(*align(1) VecF, @ptrCast(c_ptr_loc)).* = res;
        } else {
            const res_arr: [LANES]f32 = acc1;
            var i: usize = 0;
            while (i < rem) : (i += 1) {
                const idx: usize = idx_n + LANES + i;
                c_ptr_base[idx] = alpha * res_arr[i] + beta * c_ptr_base[idx];
            }
        }
    }
}
