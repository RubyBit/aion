const types = @import("../../types.zig");
const simd = @import("simd.zig");
const matvec_registry = @import("matvec_registry.zig");

const BackendError = types.BackendError;
const MatMulParams = types.MatMulParams;
const Tuning = matvec_registry.Tuning;

/// A parameterized matvec kernel generator.
///
/// This targets transformer-style GEMV: M == 1.
/// We keep it fully comptime-specialized so the hot loop remains vectorized.
pub fn Kernel(comptime t: Tuning) type {
    return struct {
        pub const LANES: usize = simd.lanesF32();
        pub const NR: usize = t.nr;
        pub const NC: usize = t.nc;
        pub const PREFETCH_K_DIST: usize = t.prefetch_k_dist;

        comptime {
            if (LANES == 0) @compileError("LANES must be > 0");
            if (NR != 2 * LANES) @compileError("NR must equal 2*LANES");
            if (NC == 0) @compileError("NC must be > 0");
        }

        pub fn matvecF32(params: MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) BackendError!void {
            return matvecF32Range(params, 0, params.n, c_bytes, a_bytes, b_bytes);
        }

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

                    @setRuntimeSafety(false);
                    var kk: usize = 0;
                    while (kk < k) : (kk += 1) {
                        const a_val: f32 = a_all[kk];
                        const a_v: VecF = @splat(a_val);

                        const b_row_off: usize = kk * n_total + j0;
                        const b_ptr: [*]align(1) const f32 = b_all.ptr + b_row_off;

                        if (PREFETCH_K_DIST != 0 and kk + PREFETCH_K_DIST < k) {
                            const pf_off: usize = (kk + PREFETCH_K_DIST) * n_total + j0;
                            @prefetch(@as([*]const u8, @ptrCast(b_all.ptr + pf_off)), .{ .rw = .read, .locality = 3, .cache = .data });
                        }

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
    };
}
