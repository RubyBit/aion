const std = @import("std");
const types = @import("../../types.zig");
const simd = @import("simd.zig");
const quant = @import("quant.zig");

const BackendError = types.BackendError;
const MatMulParams = types.MatMulParams;

// Tuning constants for f32
// Micro-kernel tiles
pub const MR = 6;
pub const NR = 16;
// L1/L2 blocking params
pub const KC = 256;
pub const MC = 144; // 24 * MR. Fits A-panel in L2 (144*256*4 = 144KB)
pub const NC = 256; // 16 * NR. Fits B-panel in L2 (256*256*4 = 256KB)

/// Reusable scratch buffers for `matmulF32`.
///
/// Motivation:
/// - The optimized `matmulF32` packs panels into ~400KiB temporary buffers.
/// - Allocating those on the stack for every call is very expensive on Windows
///   (stack probing / page-touch), and tiled execution calls matmul many times.
///
/// Contract:
/// - One scratch instance must not be used concurrently by multiple threads.
pub const MatMulScratchF32 = struct {
    packed_b: []align(32) f32,
    packed_a: []align(32) f32,

    pub fn init(allocator: std.mem.Allocator) !MatMulScratchF32 {
        // Over-align for safe vector loads.
        const pb: []align(32) f32 = try allocator.alignedAlloc(f32, std.mem.Alignment.fromByteUnits(32), KC * NC);
        errdefer allocator.free(pb);
        const pa: []align(32) f32 = try allocator.alignedAlloc(f32, std.mem.Alignment.fromByteUnits(32), MC * KC);
        errdefer allocator.free(pa);
        return .{ .packed_b = pb, .packed_a = pa };
    }

    pub fn deinit(self: *MatMulScratchF32, allocator: std.mem.Allocator) void {
        allocator.free(self.packed_b);
        allocator.free(self.packed_a);
        self.* = undefined;
    }
};

/// C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C[M,N]
///
/// Optimized with cache blocking (GotoBLAS-like) and register tiling (6x16).
pub fn matmulF32(params: MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) BackendError!void {
    // Backwards-compatible wrapper: uses stack scratch.
    // Prefer `matmulF32WithScratch` from higher-level parallel tiled execution.
    var packed_b: [KC * NC]f32 align(32) = undefined;
    var packed_a: [MC * KC]f32 align(32) = undefined;
    var scratch: MatMulScratchF32 = .{ .packed_b = packed_b[0..], .packed_a = packed_a[0..] };
    return matmulF32WithScratch(&scratch, params, c_bytes, a_bytes, b_bytes);
}

/// `matmulF32` implementation that uses caller-provided reusable scratch buffers.
///
/// This is the preferred entry point for tiled execution, where matmul is called
/// many times per iteration.
pub fn matmulF32WithScratch(
    scratch: *MatMulScratchF32,
    params: MatMulParams,
    c_bytes: []u8,
    a_bytes: []const u8,
    b_bytes: []const u8,
) BackendError!void {
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
    if (scratch.packed_b.len < KC * NC or scratch.packed_a.len < MC * KC) {
        return BackendError.InvalidArgument;
    }

    const packed_b: []align(32) f32 = scratch.packed_b;
    const packed_a: []align(32) f32 = scratch.packed_a;

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
                const panel_idx: usize = jr / NR; // jr is multiple of NR.
                const offset: usize = panel_idx * (KC * NR);
                const dest: *[KC * NR]f32 = @ptrCast(packed_b[offset..][0 .. KC * NR]);
                packPanelB(kc, nr, b, n, kk, jc + jr, dest);
            }

            // Loop Ic (Macro-Row of C)
            var ic: usize = 0;
            while (ic < m) : (ic += MC) {
                const mc = @min(MC, m - ic);

                // 2. Pack Panel A (Macro)
                var ir: usize = 0;
                while (ir < mc) : (ir += MR) {
                    const mr = @min(MR, mc - ir);
                    const panel_idx: usize = ir / MR;
                    const offset: usize = panel_idx * (MR * KC);
                    const dest: *[MR * KC]f32 = @ptrCast(packed_a[offset..][0 .. MR * KC]);
                    packPanelA(kc, mr, a, k, ic + ir, kk, dest);
                }

                // 3. Macro-Kernel
                var jr_ex: usize = 0;
                while (jr_ex < nc) : (jr_ex += NR) {
                    const nr = @min(NR, nc - jr_ex);
                    const b_offset: usize = (jr_ex / NR) * (KC * NR);
                    const b_panel: *const [KC * NR]f32 = @ptrCast(@alignCast(packed_b.ptr + b_offset));

                    var ir_ex: usize = 0;
                    while (ir_ex < mc) : (ir_ex += MR) {
                        const mr = @min(MR, mc - ir_ex);
                        const a_offset: usize = (ir_ex / MR) * (MR * KC);
                        const a_panel: *const [MR * KC]f32 = @ptrCast(@alignCast(packed_a.ptr + a_offset));

                        if (mr == MR and nr == NR) {
                            if (alpha == 1.0 and beta_eff == 0.0) {
                                microKernel6x16_Fast_A1B0(kc, a_panel, b_panel, c, n, ic + ir_ex, jc + jr_ex);
                            } else if (alpha == 1.0 and beta_eff == 1.0) {
                                microKernel6x16_Fast_A1B1(kc, a_panel, b_panel, c, n, ic + ir_ex, jc + jr_ex);
                            } else {
                                microKernel6x16_Fast(kc, alpha, beta_eff, a_panel, b_panel, c, n, ic + ir_ex, jc + jr_ex);
                            }
                        } else {
                            microKernel6x16(kc, alpha, beta_eff, a_panel, b_panel, c, n, ic + ir_ex, jc + jr_ex, mr, nr);
                        }
                    }
                }
            }
        }
    }
}

/// Pack a single contiguous f32 tile B[K,N] (row-major, stride N) into `scratch.packed_b`.
///
/// Preconditions:
/// - `k <= KC`
/// - `n <= NC`
/// - `b_bytes` contains at least `k*n*sizeof(f32)` bytes.
///
/// After this, callers may run multiple `matmulF32WithScratchPackedB(...)` calls
/// (with the same K and N) without re-packing B.
pub fn packBTileF32WithScratch(
    scratch: *MatMulScratchF32,
    k: usize,
    n: usize,
    b_bytes: []const u8,
) BackendError!void {
    if (k > KC or n > NC) return BackendError.InvalidArgument;
    const b: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, b_bytes);
    if (b.len < k * n) return BackendError.InvalidArgument;
    if (scratch.packed_b.len < KC * NC) return BackendError.InvalidArgument;

    const packed_b: []align(32) f32 = scratch.packed_b;

    var jr: usize = 0;
    while (jr < n) : (jr += NR) {
        const nr = @min(NR, n - jr);
        const panel_idx: usize = jr / NR;
        const offset: usize = panel_idx * (KC * NR);
        const dest: *[KC * NR]f32 = @ptrCast(packed_b[offset..][0 .. KC * NR]);
        packPanelB(k, nr, b, n, 0, jr, dest);
    }
}

/// Compute `C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C[M,N]` using a pre-packed B.
///
/// Preconditions:
/// - `scratch.packed_b` must already contain packed B for the given `K` and `N`.
/// - `params.k <= KC` and `params.n <= NC`.
///
/// This is intended for tiled program execution where the same B tile is reused
/// across multiple A/C tiles (e.g. fixed `ti_k, ti_n` while iterating `ti_m`).
pub fn matmulF32WithScratchPackedB(
    scratch: *MatMulScratchF32,
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

    const c: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, c_bytes);
    const a: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, a_bytes);
    if (c.len < m * n or a.len < m * k) return BackendError.InvalidArgument;
    if (scratch.packed_b.len < KC * NC or scratch.packed_a.len < MC * KC) return BackendError.InvalidArgument;

    const packed_b: []align(32) f32 = scratch.packed_b;
    const packed_a: []align(32) f32 = scratch.packed_a;

    // Only the Ic/Jr/Ir loops. B is assumed packed for (k,n).
    var ic: usize = 0;
    while (ic < m) : (ic += MC) {
        const mc = @min(MC, m - ic);

        // Pack Panel A for this macro row.
        var ir: usize = 0;
        while (ir < mc) : (ir += MR) {
            const mr = @min(MR, mc - ir);
            const panel_idx: usize = ir / MR;
            const offset: usize = panel_idx * (MR * KC);
            const dest: *[MR * KC]f32 = @ptrCast(packed_a[offset..][0 .. MR * KC]);
            packPanelA(k, mr, a, k, ic + ir, 0, dest);
        }

        // Macro-kernel.
        var jr_ex: usize = 0;
        while (jr_ex < n) : (jr_ex += NR) {
            const nr = @min(NR, n - jr_ex);
            const b_offset: usize = (jr_ex / NR) * (KC * NR);
            const b_panel: *const [KC * NR]f32 = @ptrCast(@alignCast(packed_b.ptr + b_offset));

            var ir_ex: usize = 0;
            while (ir_ex < mc) : (ir_ex += MR) {
                const mr = @min(MR, mc - ir_ex);
                const a_offset: usize = (ir_ex / MR) * (MR * KC);
                const a_panel: *const [MR * KC]f32 = @ptrCast(@alignCast(packed_a.ptr + a_offset));

                if (mr == MR and nr == NR) {
                    if (alpha == 1.0 and beta == 0.0) {
                        microKernel6x16_Fast_A1B0(k, a_panel, b_panel, c, n, ic + ir_ex, jr_ex);
                    } else if (alpha == 1.0 and beta == 1.0) {
                        microKernel6x16_Fast_A1B1(k, a_panel, b_panel, c, n, ic + ir_ex, jr_ex);
                    } else {
                        microKernel6x16_Fast(k, alpha, beta, a_panel, b_panel, c, n, ic + ir_ex, jr_ex);
                    }
                } else {
                    microKernel6x16(k, alpha, beta, a_panel, b_panel, c, n, ic + ir_ex, jr_ex, mr, nr);
                }
            }
        }
    }
}

/// Same as `matmulF32WithScratchPackedB`, but takes the packed-B buffer explicitly.
///
/// Intended for tiled execution that wants to pack each B tile once (shared across tasks)
/// and then run many A/C tiles against it.
pub fn matmulF32WithScratchPackedBView(
    scratch: *MatMulScratchF32,
    packed_b_view: []align(32) const f32,
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
    if (packed_b_view.len < KC * NC) return BackendError.InvalidArgument;

    const c: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, c_bytes);
    const a: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, a_bytes);
    if (c.len < m * n or a.len < m * k) return BackendError.InvalidArgument;
    if (scratch.packed_a.len < MC * KC) return BackendError.InvalidArgument;

    const packed_b: []align(32) const f32 = packed_b_view;
    const packed_a: []align(32) f32 = scratch.packed_a;

    // Only the Ic/Jr/Ir loops. B is assumed packed for (k,n).
    var ic: usize = 0;
    while (ic < m) : (ic += MC) {
        const mc = @min(MC, m - ic);

        // Pack Panel A for this macro row.
        var ir: usize = 0;
        while (ir < mc) : (ir += MR) {
            const mr = @min(MR, mc - ir);
            const panel_idx: usize = ir / MR;
            const offset: usize = panel_idx * (MR * KC);
            const dest: *[MR * KC]f32 = @ptrCast(packed_a[offset..][0 .. MR * KC]);
            packPanelA(k, mr, a, k, ic + ir, 0, dest);
        }

        // Macro-kernel.
        var jr_ex: usize = 0;
        while (jr_ex < n) : (jr_ex += NR) {
            const nr = @min(NR, n - jr_ex);
            const b_offset: usize = (jr_ex / NR) * (KC * NR);
            const b_panel: *const [KC * NR]f32 = @ptrCast(@alignCast(packed_b.ptr + b_offset));

            var ir_ex: usize = 0;
            while (ir_ex < mc) : (ir_ex += MR) {
                const mr = @min(MR, mc - ir_ex);
                const a_offset: usize = (ir_ex / MR) * (MR * KC);
                const a_panel: *const [MR * KC]f32 = @ptrCast(@alignCast(packed_a.ptr + a_offset));

                if (mr == MR and nr == NR) {
                    if (alpha == 1.0 and beta == 0.0) {
                        microKernel6x16_Fast_A1B0(k, a_panel, b_panel, c, n, ic + ir_ex, jr_ex);
                    } else if (alpha == 1.0 and beta == 1.0) {
                        microKernel6x16_Fast_A1B1(k, a_panel, b_panel, c, n, ic + ir_ex, jr_ex);
                    } else {
                        microKernel6x16_Fast(k, alpha, beta, a_panel, b_panel, c, n, ic + ir_ex, jr_ex);
                    }
                } else {
                    microKernel6x16(k, alpha, beta, a_panel, b_panel, c, n, ic + ir_ex, jr_ex, mr, nr);
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

    const alpha_is_1: bool = (alpha == 1.0);
    const beta_is_0: bool = (beta == 0.0);
    const beta_is_1: bool = (beta == 1.0);

    inline for (0..MR) |r| {
        const row_off = (idx_m + r) * c_stride + idx_n;
        const c_row_ptr = c.ptr + row_off;

        // Vec 0
        {
            const c_ptr_loc = c_row_ptr;
            const scaled: Vec = if (alpha_is_1) acc[r][0] else (alpha_v * acc[r][0]);
            if (beta_is_0) {
                @as(*align(1) Vec, @ptrCast(c_ptr_loc)).* = scaled;
            } else {
                const c_old: Vec = @as(*align(1) const Vec, @ptrCast(c_ptr_loc)).*;
                const res: Vec = if (beta_is_1) (scaled + c_old) else @mulAdd(Vec, beta_v, c_old, scaled);
                @as(*align(1) Vec, @ptrCast(c_ptr_loc)).* = res;
            }
        }
        // Vec 1
        {
            const c_ptr_loc = c_row_ptr + lanes;
            const scaled: Vec = if (alpha_is_1) acc[r][1] else (alpha_v * acc[r][1]);
            if (beta_is_0) {
                @as(*align(1) Vec, @ptrCast(c_ptr_loc)).* = scaled;
            } else {
                const c_old: Vec = @as(*align(1) const Vec, @ptrCast(c_ptr_loc)).*;
                const res: Vec = if (beta_is_1) (scaled + c_old) else @mulAdd(Vec, beta_v, c_old, scaled);
                @as(*align(1) Vec, @ptrCast(c_ptr_loc)).* = res;
            }
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
    const alpha_is_1: bool = (alpha == 1.0);
    const beta_is_0: bool = (beta == 0.0);
    const beta_is_1: bool = (beta == 1.0);

    inline for (0..MR) |r| {
        if (r < mr) {
            const row_off = (idx_m + r) * c_stride + idx_n;

            // Vector 0 (cols 0..7)
            if (0 < nr) {
                const c_ptr_loc = c.ptr + row_off;

                if (nr >= lanes) {
                    const scaled: Vec = if (alpha_is_1) acc[r][0] else (alpha_v * acc[r][0]);
                    if (beta_is_0) {
                        @as(*align(1) Vec, @ptrCast(c_ptr_loc)).* = scaled;
                    } else {
                        const c_old: Vec = @as(*align(1) const Vec, @ptrCast(c_ptr_loc)).*;
                        const res: Vec = if (beta_is_1) (scaled + c_old) else @mulAdd(Vec, beta_v, c_old, scaled);
                        @as(*align(1) Vec, @ptrCast(c_ptr_loc)).* = res;
                    }
                } else {
                    const res_arr: [lanes]f32 = acc[r][0];
                    for (0..nr) |vi| {
                        if (beta_is_0) {
                            c[row_off + vi] = alpha * res_arr[vi];
                        } else {
                            c[row_off + vi] = alpha * res_arr[vi] + beta * c[row_off + vi];
                        }
                    }
                }
            }

            // Vector 1 (cols 8..15)
            if (lanes < nr) {
                const c_ptr_loc = c.ptr + row_off + lanes;
                const rem = nr - lanes;

                if (rem >= lanes) {
                    const scaled: Vec = if (alpha_is_1) acc[r][1] else (alpha_v * acc[r][1]);
                    if (beta_is_0) {
                        @as(*align(1) Vec, @ptrCast(c_ptr_loc)).* = scaled;
                    } else {
                        const c_old: Vec = @as(*align(1) const Vec, @ptrCast(c_ptr_loc)).*;
                        const res: Vec = if (beta_is_1) (scaled + c_old) else @mulAdd(Vec, beta_v, c_old, scaled);
                        @as(*align(1) Vec, @ptrCast(c_ptr_loc)).* = res;
                    }
                } else {
                    const res_arr: [lanes]f32 = acc[r][1];
                    for (0..rem) |vi| {
                        if (beta_is_0) {
                            c[row_off + lanes + vi] = alpha * res_arr[vi];
                        } else {
                            c[row_off + lanes + vi] = alpha * res_arr[vi] + beta * c[row_off + lanes + vi];
                        }
                    }
                }
            }
        }
    }
}

// Specialized fast path: alpha=1, beta=0, full 6x16 block.
// This is the common case for the first K-tile in GEMM.
fn microKernel6x16_Fast_A1B0(
    kc: usize,
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

    const c_ptr_aligned32: bool = (@intFromPtr(c.ptr) & 31) == 0;
    const c_stride_aligned32: bool = (((c_stride * @sizeOf(f32)) & 31) == 0);
    const c_col_aligned32: bool = ((((idx_n) * @sizeOf(f32)) & 31) == 0);
    const c_rows_aligned32: bool = c_ptr_aligned32 and c_stride_aligned32 and c_col_aligned32;

    var acc: [MR][2]Vec = undefined;
    inline for (0..MR) |r| {
        acc[r][0] = @splat(0.0);
        acc[r][1] = @splat(0.0);
    }

    const a_ptr_base: [*]const f32 = @ptrCast(&packed_a[0]);
    const b_ptr_base: [*]const u8 = @ptrCast(&packed_b[0]);

    var k: usize = 0;
    while (k + 8 <= kc) : (k += 8) {
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
    while (k < kc) : (k += 1) {
        const b_off: usize = k * NR;
        const b0_bytes: usize = b_off * @sizeOf(f32);
        const b1_bytes: usize = (b_off + lanes) * @sizeOf(f32);
        const b_vec0: Vec = @as(*align(32) const Vec, @ptrCast(@alignCast(b_ptr_base + b0_bytes))).*;
        const b_vec1: Vec = @as(*align(32) const Vec, @ptrCast(@alignCast(b_ptr_base + b1_bytes))).*;

        inline for (0..MR) |r| {
            const a_val: f32 = a_ptr_base[r * KC + k];
            const a_vec: Vec = @splat(a_val);
            acc[r][0] = @mulAdd(Vec, a_vec, b_vec0, acc[r][0]);
            acc[r][1] = @mulAdd(Vec, a_vec, b_vec1, acc[r][1]);
        }
    }

    if (c_rows_aligned32) {
        inline for (0..MR) |r| {
            const row_off = (idx_m + r) * c_stride + idx_n;
            const c_row_ptr = c.ptr + row_off;
            @as(*align(32) Vec, @ptrCast(@alignCast(c_row_ptr))).* = acc[r][0];
            @as(*align(32) Vec, @ptrCast(@alignCast(c_row_ptr + lanes))).* = acc[r][1];
        }
    } else {
        inline for (0..MR) |r| {
            const row_off = (idx_m + r) * c_stride + idx_n;
            const c_row_ptr = c.ptr + row_off;
            @as(*align(1) Vec, @ptrCast(c_row_ptr)).* = acc[r][0];
            @as(*align(1) Vec, @ptrCast(c_row_ptr + lanes)).* = acc[r][1];
        }
    }
}

// Specialized fast path: alpha=1, beta=1, full 6x16 block.
// This is the common case for subsequent K-tiles (accumulating into C).
fn microKernel6x16_Fast_A1B1(
    kc: usize,
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

    const c_ptr_aligned32: bool = (@intFromPtr(c.ptr) & 31) == 0;
    const c_stride_aligned32: bool = (((c_stride * @sizeOf(f32)) & 31) == 0);
    const c_col_aligned32: bool = ((((idx_n) * @sizeOf(f32)) & 31) == 0);
    const c_rows_aligned32: bool = c_ptr_aligned32 and c_stride_aligned32 and c_col_aligned32;

    var acc: [MR][2]Vec = undefined;
    inline for (0..MR) |r| {
        acc[r][0] = @splat(0.0);
        acc[r][1] = @splat(0.0);
    }

    const a_ptr_base: [*]const f32 = @ptrCast(&packed_a[0]);
    const b_ptr_base: [*]const u8 = @ptrCast(&packed_b[0]);

    var k: usize = 0;
    while (k + 8 <= kc) : (k += 8) {
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
    while (k < kc) : (k += 1) {
        const b_off: usize = k * NR;
        const b0_bytes: usize = b_off * @sizeOf(f32);
        const b1_bytes: usize = (b_off + lanes) * @sizeOf(f32);
        const b_vec0: Vec = @as(*align(32) const Vec, @ptrCast(@alignCast(b_ptr_base + b0_bytes))).*;
        const b_vec1: Vec = @as(*align(32) const Vec, @ptrCast(@alignCast(b_ptr_base + b1_bytes))).*;

        inline for (0..MR) |r| {
            const a_val: f32 = a_ptr_base[r * KC + k];
            const a_vec: Vec = @splat(a_val);
            acc[r][0] = @mulAdd(Vec, a_vec, b_vec0, acc[r][0]);
            acc[r][1] = @mulAdd(Vec, a_vec, b_vec1, acc[r][1]);
        }
    }

    if (c_rows_aligned32) {
        inline for (0..MR) |r| {
            const row_off = (idx_m + r) * c_stride + idx_n;
            const c_row_ptr = c.ptr + row_off;
            const c_old0: Vec = @as(*align(32) const Vec, @ptrCast(@alignCast(c_row_ptr))).*;
            const c_old1: Vec = @as(*align(32) const Vec, @ptrCast(@alignCast(c_row_ptr + lanes))).*;
            @as(*align(32) Vec, @ptrCast(@alignCast(c_row_ptr))).* = acc[r][0] + c_old0;
            @as(*align(32) Vec, @ptrCast(@alignCast(c_row_ptr + lanes))).* = acc[r][1] + c_old1;
        }
    } else {
        inline for (0..MR) |r| {
            const row_off = (idx_m + r) * c_stride + idx_n;
            const c_row_ptr = c.ptr + row_off;
            const c_old0: Vec = @as(*align(1) const Vec, @ptrCast(c_row_ptr)).*;
            const c_old1: Vec = @as(*align(1) const Vec, @ptrCast(c_row_ptr + lanes)).*;
            @as(*align(1) Vec, @ptrCast(c_row_ptr)).* = acc[r][0] + c_old0;
            @as(*align(1) Vec, @ptrCast(c_row_ptr + lanes)).* = acc[r][1] + c_old1;
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
