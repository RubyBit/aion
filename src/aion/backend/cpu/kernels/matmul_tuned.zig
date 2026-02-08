const std = @import("std");
const types = @import("../../types.zig");
const simd = @import("simd.zig");
const matmul_registry = @import("../registry/matmul_registry.zig");

const BackendError = types.BackendError;
const MatMulParams = types.MatMulParams;
const Tuning = matmul_registry.Tuning;

pub fn Kernel(comptime t: Tuning) type {
    return struct {
        pub const LANES: usize = simd.lanesF32();
        pub const KC = t.kc;
        pub const MC = t.mc;
        pub const NC = t.nc;

        pub const MR = t.mr;
        pub const NR = t.nr;
        pub const ScratchAlignment: usize = 32;

        pub fn scratchBytes() usize {
            return (KC * NC + MC * KC) * @sizeOf(f32);
        }

        fn scratchAsF32Slice(scratch_bytes: []u8, elems: usize) BackendError![]align(ScratchAlignment) f32 {
            if ((@intFromPtr(scratch_bytes.ptr) & (ScratchAlignment - 1)) != 0) return BackendError.InvalidArgument;
            const need = elems * @sizeOf(f32);
            if (scratch_bytes.len < need) return BackendError.InvalidArgument;
            return @alignCast(std.mem.bytesAsSlice(f32, scratch_bytes[0..need]));
        }

        fn splitScratch(scratch_bytes: []u8) BackendError!struct { pb: []align(32) f32, pa: []align(32) f32 } {
            const pb_elems = KC * NC;
            const pa_elems = MC * KC;
            const full = try scratchAsF32Slice(scratch_bytes, pb_elems + pa_elems);
            return .{ .pb = @alignCast(full[0..pb_elems]), .pa = @alignCast(full[pb_elems .. pb_elems + pa_elems]) };
        }

        pub fn packBTileF32(scratch_bytes: []u8, k: usize, n: usize, b_bytes: []const u8) BackendError!void {
            if (k > KC or n > NC) return BackendError.InvalidArgument;
            const b: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, b_bytes);
            if (b.len < k * n) return BackendError.InvalidArgument;

            const s = try splitScratch(scratch_bytes);
            const packed_b: []align(32) f32 = s.pb;

            var jr: usize = 0;
            while (jr < n) : (jr += NR) {
                const nr = @min(NR, n - jr);
                const panel_idx: usize = jr / NR;
                const offset: usize = panel_idx * (KC * NR);
                const dest: *[KC * NR]f32 = @ptrCast(packed_b[offset..][0 .. KC * NR]);
                packPanelB(k, nr, b, n, 0, jr, dest);
            }
        }

        pub fn matmulF32PackedB(scratch_bytes: []u8, packed_b_view: []align(32) const f32, params: MatMulParams, c_bytes: []u8, a_bytes: []const u8) BackendError!void {
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

            const s = try splitScratch(scratch_bytes);
            const packed_a: []align(32) f32 = s.pa;
            const packed_b: []align(32) const f32 = packed_b_view;

            var ic: usize = 0;
            while (ic < m) : (ic += MC) {
                const mc = @min(MC, m - ic);

                var ir: usize = 0;
                while (ir < mc) : (ir += MR) {
                    const mr = @min(MR, mc - ir);
                    const panel_idx: usize = ir / MR;
                    const offset: usize = panel_idx * (MR * KC);
                    const dest: *[MR * KC]f32 = @ptrCast(packed_a[offset..][0 .. MR * KC]);
                    packPanelA(k, mr, a, k, ic + ir, 0, dest);
                }

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
            const lanes = LANES; // 8
            const Vec = @Vector(lanes, f32);

            var acc: [MR][2]Vec = undefined;
            inline for (0..MR) |r| {
                acc[r][0] = @splat(0.0);
                acc[r][1] = @splat(0.0);
            }

            const a_ptr_base: [*]const f32 = @ptrCast(&packed_a[0]);
            const b_ptr_base: [*]const u8 = @ptrCast(&packed_b[0]);

            var kk: usize = 0;
            while (kk + 8 <= kc) : (kk += 8) {
                const prefetch_dist = 512;
                @prefetch(b_ptr_base + kk * NR * 4 + prefetch_dist, .{ .rw = .read, .locality = 3, .cache = .data });

                inline for (0..8) |uk| {
                    const b_off: usize = (kk + uk) * NR;
                    const b0_bytes: usize = b_off * @sizeOf(f32);
                    const b1_bytes: usize = (b_off + lanes) * @sizeOf(f32);
                    const b_vec0: Vec = @as(*align(32) const Vec, @ptrCast(@alignCast(b_ptr_base + b0_bytes))).*;
                    const b_vec1: Vec = @as(*align(32) const Vec, @ptrCast(@alignCast(b_ptr_base + b1_bytes))).*;

                    inline for (0..MR) |r| {
                        const a_val: f32 = a_ptr_base[r * KC + (kk + uk)];
                        const a_vec: Vec = @splat(a_val);
                        acc[r][0] = @mulAdd(Vec, a_vec, b_vec0, acc[r][0]);
                        acc[r][1] = @mulAdd(Vec, a_vec, b_vec1, acc[r][1]);
                    }
                }
            }

            while (kk < kc) : (kk += 1) {
                const b_off: usize = kk * NR;
                const b0_bytes: usize = b_off * @sizeOf(f32);
                const b1_bytes: usize = (b_off + lanes) * @sizeOf(f32);
                const b_vec0: Vec = @as(*align(32) const Vec, @ptrCast(@alignCast(b_ptr_base + b0_bytes))).*;
                const b_vec1: Vec = @as(*align(32) const Vec, @ptrCast(@alignCast(b_ptr_base + b1_bytes))).*;

                inline for (0..MR) |r| {
                    const a_val: f32 = a_ptr_base[r * KC + kk];
                    const a_vec: Vec = @splat(a_val);
                    acc[r][0] = @mulAdd(Vec, a_vec, b_vec0, acc[r][0]);
                    acc[r][1] = @mulAdd(Vec, a_vec, b_vec1, acc[r][1]);
                }
            }

            const alpha_v: Vec = @splat(alpha);
            const beta_v: Vec = @splat(beta);

            const alpha_is_1: bool = (alpha == 1.0);
            const beta_is_0: bool = (beta == 0.0);
            const beta_is_1: bool = (beta == 1.0);

            inline for (0..MR) |r| {
                const row_off = (idx_m + r) * c_stride + idx_n;
                const c_row_ptr = c.ptr + row_off;

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
            const lanes = LANES; // 8
            const Vec = @Vector(lanes, f32);

            var acc: [MR][2]Vec = undefined;
            inline for (0..MR) |r| {
                inline for (0..2) |v| {
                    acc[r][v] = @splat(0.0);
                }
            }

            const a_ptr_base: [*]const f32 = @ptrCast(&packed_a[0]);
            const b_ptr_base: [*]const u8 = @ptrCast(&packed_b[0]);

            var kk: usize = 0;
            while (kk < kc) : (kk += 1) {
                const b_off: usize = kk * NR;
                const b0_bytes: usize = b_off * @sizeOf(f32);
                const b1_bytes: usize = (b_off + lanes) * @sizeOf(f32);
                const b_vec0: Vec = @as(*align(32) const Vec, @ptrCast(@alignCast(b_ptr_base + b0_bytes))).*;
                const b_vec1: Vec = @as(*align(32) const Vec, @ptrCast(@alignCast(b_ptr_base + b1_bytes))).*;

                inline for (0..MR) |r| {
                    if (r < mr) {
                        const a_val: f32 = a_ptr_base[r * KC + kk];
                        const a_vec: Vec = @splat(a_val);
                        acc[r][0] = @mulAdd(Vec, a_vec, b_vec0, acc[r][0]);
                        acc[r][1] = @mulAdd(Vec, a_vec, b_vec1, acc[r][1]);
                    }
                }
            }

            const alpha_v: Vec = @splat(alpha);
            const beta_v: Vec = @splat(beta);
            const alpha_is_1: bool = (alpha == 1.0);
            const beta_is_0: bool = (beta == 0.0);
            const beta_is_1: bool = (beta == 1.0);

            inline for (0..MR) |r| {
                if (r < mr) {
                    const row_off = (idx_m + r) * c_stride + idx_n;

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
            const c_col_aligned32: bool = (((idx_n * @sizeOf(f32)) & 31) == 0);
            const c_rows_aligned32: bool = c_ptr_aligned32 and c_stride_aligned32 and c_col_aligned32;

            var acc: [MR][2]Vec = undefined;
            inline for (0..MR) |r| {
                acc[r][0] = @splat(0.0);
                acc[r][1] = @splat(0.0);
            }

            const a_ptr_base: [*]const f32 = @ptrCast(&packed_a[0]);
            const b_ptr_base: [*]const u8 = @ptrCast(&packed_b[0]);

            var kk: usize = 0;
            while (kk + 8 <= kc) : (kk += 8) {
                inline for (0..8) |uk| {
                    const b_off: usize = (kk + uk) * NR;
                    const b0_bytes: usize = b_off * @sizeOf(f32);
                    const b1_bytes: usize = (b_off + lanes) * @sizeOf(f32);
                    const b_vec0: Vec = @as(*align(32) const Vec, @ptrCast(@alignCast(b_ptr_base + b0_bytes))).*;
                    const b_vec1: Vec = @as(*align(32) const Vec, @ptrCast(@alignCast(b_ptr_base + b1_bytes))).*;

                    inline for (0..MR) |r| {
                        const a_val: f32 = a_ptr_base[r * KC + (kk + uk)];
                        const a_vec: Vec = @splat(a_val);
                        acc[r][0] = @mulAdd(Vec, a_vec, b_vec0, acc[r][0]);
                        acc[r][1] = @mulAdd(Vec, a_vec, b_vec1, acc[r][1]);
                    }
                }
            }
            while (kk < kc) : (kk += 1) {
                const b_off: usize = kk * NR;
                const b0_bytes: usize = b_off * @sizeOf(f32);
                const b1_bytes: usize = (b_off + lanes) * @sizeOf(f32);
                const b_vec0: Vec = @as(*align(32) const Vec, @ptrCast(@alignCast(b_ptr_base + b0_bytes))).*;
                const b_vec1: Vec = @as(*align(32) const Vec, @ptrCast(@alignCast(b_ptr_base + b1_bytes))).*;

                inline for (0..MR) |r| {
                    const a_val: f32 = a_ptr_base[r * KC + kk];
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
            const c_col_aligned32: bool = (((idx_n * @sizeOf(f32)) & 31) == 0);
            const c_rows_aligned32: bool = c_ptr_aligned32 and c_stride_aligned32 and c_col_aligned32;

            var acc: [MR][2]Vec = undefined;
            inline for (0..MR) |r| {
                acc[r][0] = @splat(0.0);
                acc[r][1] = @splat(0.0);
            }

            const a_ptr_base: [*]const f32 = @ptrCast(&packed_a[0]);
            const b_ptr_base: [*]const u8 = @ptrCast(&packed_b[0]);

            var kk: usize = 0;
            while (kk + 8 <= kc) : (kk += 8) {
                inline for (0..8) |uk| {
                    const b_off: usize = (kk + uk) * NR;
                    const b0_bytes: usize = b_off * @sizeOf(f32);
                    const b1_bytes: usize = (b_off + lanes) * @sizeOf(f32);
                    const b_vec0: Vec = @as(*align(32) const Vec, @ptrCast(@alignCast(b_ptr_base + b0_bytes))).*;
                    const b_vec1: Vec = @as(*align(32) const Vec, @ptrCast(@alignCast(b_ptr_base + b1_bytes))).*;

                    inline for (0..MR) |r| {
                        const a_val: f32 = a_ptr_base[r * KC + (kk + uk)];
                        const a_vec: Vec = @splat(a_val);
                        acc[r][0] = @mulAdd(Vec, a_vec, b_vec0, acc[r][0]);
                        acc[r][1] = @mulAdd(Vec, a_vec, b_vec1, acc[r][1]);
                    }
                }
            }
            while (kk < kc) : (kk += 1) {
                const b_off: usize = kk * NR;
                const b0_bytes: usize = b_off * @sizeOf(f32);
                const b1_bytes: usize = (b_off + lanes) * @sizeOf(f32);
                const b_vec0: Vec = @as(*align(32) const Vec, @ptrCast(@alignCast(b_ptr_base + b0_bytes))).*;
                const b_vec1: Vec = @as(*align(32) const Vec, @ptrCast(@alignCast(b_ptr_base + b1_bytes))).*;

                inline for (0..MR) |r| {
                    const a_val: f32 = a_ptr_base[r * KC + kk];
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

        fn packPanelB(
            kc: usize,
            nr: usize,
            b: []align(1) const f32,
            n_total: usize,
            k_start: usize,
            j_start: usize,
            packed_out: *[KC * NR]f32,
        ) void {
            var kk: usize = 0;
            while (kk < kc) : (kk += 1) {
                const b_row_start = (k_start + kk) * n_total + j_start;
                const p_row_start = kk * NR;

                @memcpy(packed_out[p_row_start .. p_row_start + nr], b[b_row_start .. b_row_start + nr]);
                if (nr < NR) {
                    @memset(packed_out[p_row_start + nr .. p_row_start + NR], 0.0);
                }
            }
        }
    };
}
