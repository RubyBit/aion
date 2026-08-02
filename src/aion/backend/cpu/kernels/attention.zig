// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const fast_math = @import("fast_math.zig");
const simd = @import("simd.zig");

pub const Tuning = struct {
    mr: usize,
    nr: usize,
    mr_acc: usize,
    vec_lanes: usize,
};

pub fn Kernel(comptime tuning: Tuning) type {
    comptime {
        if (tuning.mr == 0 or tuning.nr == 0 or tuning.mr_acc == 0 or tuning.vec_lanes == 0) {
            @compileError("attention kernel tuning must be non-zero");
        }
        if ((tuning.nr % tuning.vec_lanes) != 0) {
            @compileError("attention kernel tuning.nr must be a multiple of tuning.vec_lanes");
        }
    }

    const lanes: usize = tuning.vec_lanes;

    return struct {
        const Self = @This();

        pub const simd_lanes: usize = lanes;
        pub const Vec = @Vector(lanes, f32);

        /// Softmax/score post-processing at THIS variant's width (see `RangedOps`).
        pub const ranged = RangedOps(lanes);

        pub inline fn vecLoad(ptr: [*]align(1) const f32) Self.Vec {
            return @as(*align(1) const [lanes]f32, @ptrCast(ptr)).*;
        }

        pub inline fn vecStore(ptr: [*]align(1) f32, v: Self.Vec) void {
            @as(*align(1) [lanes]f32, @ptrCast(ptr)).* = v;
        }

        pub fn expSoftmax(x: f32) f32 {
            if (!std.math.isFinite(x)) {
                if (x < 0.0) return 0.0;
                return std.math.inf(f32);
            }
            return fast_math.expApproxF32(fast_math.clampF32(x, -20.0, 0.0));
        }

        pub fn packQBlock(
            rows: usize,
            cols: usize,
            src: []align(1) const f32,
            src_stride: usize,
            dst: []f32,
        ) void {
            const MR: usize = tuning.mr;
            var c0: usize = 0;
            while (c0 < rows) : (c0 += MR) {
                if (c0 + MR > rows) break;
                const dst_ptr: [*]f32 = dst[(c0 / MR) * (cols * MR) ..].ptr;
                var k: usize = 0;
                while (k < cols) : (k += 1) {
                    const ptr: [*]f32 = dst_ptr + k * MR;
                    var i: usize = 0;
                    while (i < MR) : (i += 1) {
                        ptr[i] = src[(c0 + i) * src_stride + k];
                    }
                }
            }
        }

        pub fn packKBlock(
            rows: usize,
            cols: usize,
            src: []align(1) const f32,
            src_stride: usize,
            dst: []f32,
        ) void {
            const NR: usize = tuning.nr;
            var c0: usize = 0;
            while (c0 < rows) : (c0 += NR) {
                if (c0 + NR > rows) break;
                const dst_ptr: [*]f32 = dst[(c0 / NR) * (cols * NR) ..].ptr;
                var k: usize = 0;
                while (k < cols) : (k += 1) {
                    const ptr: [*]f32 = dst_ptr + k * NR;
                    var i: usize = 0;
                    while (i < NR) : (i += 1) {
                        ptr[i] = src[(c0 + i) * src_stride + k];
                    }
                }
            }
        }

        pub fn calcScoresF32(
            m_q: usize,
            n_k: usize,
            k_dim: usize,
            qs: []align(1) const f32,
            q_stride: usize,
            ks: []align(1) const f32,
            k_stride: usize,
            kt: []f32,
            qt: []f32,
            scores: []f32,
            score_stride: usize,
        ) void {
            const MR: usize = tuning.mr;
            const NR: usize = tuning.nr;
            const VEC_BLOCKS: usize = tuning.nr / lanes;

            Self.packQBlock(m_q, k_dim, qs, q_stride, qt);

            var r0: usize = 0;
            while (r0 < m_q) : (r0 += MR) {
                const r_end: usize = @min(r0 + MR, m_q);
                var c0: usize = 0;
                while (c0 < n_k) : (c0 += NR) {
                    if (c0 + NR > n_k or r0 + MR > m_q) {
                        var r: usize = r0;
                        while (r < r_end) : (r += 1) {
                            const q_p: []align(1) const f32 = qs[r * q_stride ..];
                            var c: usize = c0;
                            const c_end: usize = @min(c0 + NR, n_k);
                            while (c < c_end) : (c += 1) {
                                const k_p: []align(1) const f32 = ks[c * k_stride ..];
                                var acc: f32 = 0.0;
                                var k: usize = 0;
                                while (k < k_dim) : (k += 1) {
                                    acc += q_p[k] * k_p[k];
                                }
                                scores[r * score_stride + c] += acc;
                            }
                        }
                        continue;
                    }

                    var vScores: [MR][VEC_BLOCKS]Self.Vec = undefined;
                    var i: usize = 0;
                    while (i < MR) : (i += 1) {
                        var b: usize = 0;
                        while (b < VEC_BLOCKS) : (b += 1) {
                            vScores[i][b] = @splat(0.0);
                        }
                    }

                    const q_ptr_base: [*]f32 = qt[(r0 / MR) * (k_dim * MR) ..].ptr;
                    const k_ptr_base: [*]f32 = kt[(c0 / NR) * (k_dim * NR) ..].ptr;

                    var k: usize = 0;
                    while (k < k_dim) : (k += 1) {
                        const vk_ptr: [*]f32 = k_ptr_base + k * NR;
                        var vK: [VEC_BLOCKS]Self.Vec = undefined;
                        var b: usize = 0;
                        while (b < VEC_BLOCKS) : (b += 1) {
                            vK[b] = Self.vecLoad(vk_ptr + b * lanes);
                        }

                        const q_k_ptr: [*]f32 = q_ptr_base + k * MR;

                        inline for (0..MR) |ii| {
                            const vQ: Self.Vec = @as(Self.Vec, @splat(q_k_ptr[ii]));
                            b = 0;
                            while (b < VEC_BLOCKS) : (b += 1) {
                                vScores[ii][b] += vQ * vK[b];
                            }
                        }
                    }

                    var r: usize = r0;
                    while (r < r_end) : (r += 1) {
                        const ri: usize = r - r0;
                        var b: usize = 0;
                        while (b < VEC_BLOCKS) : (b += 1) {
                            const ptr: [*]f32 = scores[r * score_stride + c0 + b * lanes ..].ptr;
                            Self.vecStore(ptr, Self.vecLoad(ptr) + vScores[ri][b]);
                        }
                    }
                }
            }
        }

        /// Scores for a SMALL number of q rows — fewer than `mr`, where
        /// `calcScoresF32`'s packed path degenerates to its scalar fallback
        /// (`r0 + MR > m_q` takes the slow branch for every column block).
        ///
        /// Streams each key row ONCE and fans it into all `m_q` accumulators, so K
        /// traffic drops by a factor of `m_q` — that is the GQA-decode win, where
        /// every query head in a group reads the same K/V rows. `qs` must be a
        /// gathered/contiguous panel (row stride `q_stride`).
        pub fn calcScoresNarrowF32(
            m_q: usize,
            n_k: usize,
            k_dim: usize,
            qs: []align(1) const f32,
            q_stride: usize,
            ks: []align(1) const f32,
            k_stride: usize,
            scores: []f32,
            score_stride: usize,
        ) void {
            var r0: usize = 0;
            while (r0 < m_q) : (r0 += 8) {
                const n: usize = @min(@as(usize, 8), m_q - r0);
                switch (n) {
                    1 => narrowN(1, r0, n_k, k_dim, qs, q_stride, ks, k_stride, scores, score_stride),
                    2 => narrowN(2, r0, n_k, k_dim, qs, q_stride, ks, k_stride, scores, score_stride),
                    3 => narrowN(3, r0, n_k, k_dim, qs, q_stride, ks, k_stride, scores, score_stride),
                    4 => narrowN(4, r0, n_k, k_dim, qs, q_stride, ks, k_stride, scores, score_stride),
                    5 => narrowN(5, r0, n_k, k_dim, qs, q_stride, ks, k_stride, scores, score_stride),
                    6 => narrowN(6, r0, n_k, k_dim, qs, q_stride, ks, k_stride, scores, score_stride),
                    7 => narrowN(7, r0, n_k, k_dim, qs, q_stride, ks, k_stride, scores, score_stride),
                    else => narrowN(8, r0, n_k, k_dim, qs, q_stride, ks, k_stride, scores, score_stride),
                }
            }
        }

        /// `N` q rows at a time with the row count comptime-known, so the FMA fan-out
        /// over accumulators unrolls into registers.
        fn narrowN(
            comptime N: usize,
            r0: usize,
            n_k: usize,
            k_dim: usize,
            qs: []align(1) const f32,
            q_stride: usize,
            ks: []align(1) const f32,
            k_stride: usize,
            scores: []f32,
            score_stride: usize,
        ) void {
            const vec_end: usize = k_dim - (k_dim % lanes);

            var q_ptrs: [N][*]align(1) const f32 = undefined;
            inline for (0..N) |r| q_ptrs[r] = qs[(r0 + r) * q_stride ..].ptr;

            var c: usize = 0;
            while (c < n_k) : (c += 1) {
                const k_ptr: [*]align(1) const f32 = ks[c * k_stride ..].ptr;

                var vacc: [N]Self.Vec = @splat(@as(Self.Vec, @splat(0.0)));
                var i: usize = 0;
                while (i < vec_end) : (i += lanes) {
                    const vk: Self.Vec = Self.vecLoad(k_ptr + i);
                    inline for (0..N) |r| {
                        vacc[r] = @mulAdd(Self.Vec, Self.vecLoad(q_ptrs[r] + i), vk, vacc[r]);
                    }
                }

                var acc: [N]f32 = undefined;
                inline for (0..N) |r| acc[r] = @reduce(.Add, vacc[r]);
                i = vec_end;
                while (i < k_dim) : (i += 1) {
                    const kv: f32 = k_ptr[i];
                    inline for (0..N) |r| acc[r] += q_ptrs[r][i] * kv;
                }

                inline for (0..N) |r| scores[(r0 + r) * score_stride + c] += acc[r];
            }
        }

        pub fn accumulateValuesF32(
            m_tile: usize,
            n_v: usize,
            dv: usize,
            scores: []f32,
            score_stride: usize,
            vs: []align(1) const f32,
            v_stride: usize,
            acc: []f32,
            acc_stride: usize,
        ) void {
            const MR: usize = tuning.mr_acc;
            var j: usize = 0;
            while (j < dv) : (j += lanes) {
                if (j + lanes > dv) {
                    var jj: usize = j;
                    while (jj < dv) : (jj += 1) {
                        var r: usize = 0;
                        while (r < m_tile) : (r += 1) {
                            const p_row: []f32 = scores[r * score_stride ..];
                            var s: f32 = acc[r * acc_stride + jj];
                            var k: usize = 0;
                            while (k < n_v) : (k += 1) {
                                s += p_row[k] * vs[k * v_stride + jj];
                            }
                            acc[r * acc_stride + jj] = s;
                        }
                    }
                    break;
                }

                var r0: usize = 0;
                while (r0 < m_tile) : (r0 += MR) {
                    const r1: usize = @min(r0 + MR, m_tile);

                    var vAcc: [MR]Self.Vec = undefined;
                    var r: usize = r0;
                    while (r < r1) : (r += 1) {
                        const ri: usize = r - r0;
                        vAcc[ri] = Self.vecLoad(acc[r * acc_stride + j ..].ptr);
                    }

                    var k: usize = 0;
                    while (k < n_v) : (k += 1) {
                        const vV: Self.Vec = Self.vecLoad(vs[k * v_stride + j ..].ptr);
                        r = r0;
                        while (r < r1) : (r += 1) {
                            const ri: usize = r - r0;
                            vAcc[ri] += @as(Self.Vec, @splat(scores[r * score_stride + k])) * vV;
                        }
                    }

                    r = r0;
                    while (r < r1) : (r += 1) {
                        const ri: usize = r - r0;
                        Self.vecStore(acc[r * acc_stride + j ..].ptr, vAcc[ri]);
                    }
                }
            }
        }
    };
}

pub const DefaultTuning = Tuning{ .mr = 6, .nr = 16, .mr_acc = 8, .vec_lanes = simd.lanesF32() };
pub const DefaultKernel = Kernel(DefaultTuning);

pub const simd_lanes: usize = DefaultKernel.simd_lanes;
pub const Vec = DefaultKernel.Vec;

pub inline fn vecLoad(ptr: [*]align(1) const f32) Vec {
    return DefaultKernel.vecLoad(ptr);
}

pub inline fn vecStore(ptr: [*]align(1) f32, v: Vec) void {
    DefaultKernel.vecStore(ptr, v);
}

pub fn expSoftmax(x: f32) f32 {
    return DefaultKernel.expSoftmax(x);
}

pub fn packQBlock(
    rows: usize,
    cols: usize,
    src: []align(1) const f32,
    src_stride: usize,
    dst: []f32,
) void {
    DefaultKernel.packQBlock(rows, cols, src, src_stride, dst);
}

pub fn packKBlock(
    rows: usize,
    cols: usize,
    src: []align(1) const f32,
    src_stride: usize,
    dst: []f32,
) void {
    DefaultKernel.packKBlock(rows, cols, src, src_stride, dst);
}

pub fn calcScoresF32(
    m_q: usize,
    n_k: usize,
    k_dim: usize,
    qs: []align(1) const f32,
    q_stride: usize,
    ks: []align(1) const f32,
    k_stride: usize,
    kt: []f32,
    qt: []f32,
    scores: []f32,
    score_stride: usize,
) void {
    DefaultKernel.calcScoresF32(m_q, n_k, k_dim, qs, q_stride, ks, k_stride, kt, qt, scores, score_stride);
}

pub fn accumulateValuesF32(
    m_tile: usize,
    n_v: usize,
    dv: usize,
    scores: []f32,
    score_stride: usize,
    vs: []align(1) const f32,
    v_stride: usize,
    acc: []f32,
    acc_stride: usize,
) void {
    DefaultKernel.accumulateValuesF32(m_tile, n_v, dv, scores, score_stride, vs, v_stride, acc, acc_stride);
}

// ---------------------------------------------------------------------------
// Ranged variants, for blocked (flash-style) attention.
//
// Each q row sees a CONTIGUOUS key range — causal, sliding-window and ring limits
// are all one interval — so a key block is masked by a per-row `[lo, hi)` rather
// than by an additive mask. Out-of-range entries are written as exact 0.0 after the
// exp, which keeps them out of both the row sum and the V accumulation; using -inf
// logits instead would leak `exp(-80)` per masked key into the denominator.
// ---------------------------------------------------------------------------

/// The softmax/score-postprocessing family, generic over the SIMD width so each
/// registry variant instantiates its own. These used to be file-scope functions that
/// baked in `comptime simd.lanesF32()` — the BUILD target's width — while the GEMMs
/// beside them came from the runtime-selected tier, so on a machine whose tier is wider
/// than the baseline the two halves of the same kernel ran at different widths.
/// Reachable as `Kernel(tuning).ranged` and through the registry.
fn RangedOps(comptime lanes: usize) type {
    return struct {
        /// Fold `scores[r, lo[r]..hi[r]]` into the RUNNING per-row max (seed `row_max`
        /// with the online-softmax state, or -inf). Empty ranges contribute nothing.
        pub fn rowMaxRangeF32(
            scores: []const f32,
            rows: usize,
            score_stride: usize,
            lo: []const u32,
            hi: []const u32,
            row_max: []f32,
        ) void {
            const VecT = @Vector(lanes, f32);

            var r: usize = 0;
            while (r < rows) : (r += 1) {
                const c0: usize = lo[r];
                const c1: usize = hi[r];
                if (c1 <= c0) continue;

                const base: [*]const f32 = scores.ptr + r * score_stride;
                var mx: f32 = row_max[r];
                var c: usize = c0;
                if (c1 - c0 >= lanes) {
                    var vmx: VecT = @splat(mx);
                    while (c + lanes <= c1) : (c += lanes) {
                        vmx = @max(vmx, @as(*align(1) const VecT, @ptrCast(base + c)).*);
                    }
                    mx = @reduce(.Max, vmx);
                }
                while (c < c1) : (c += 1) mx = @max(mx, base[c]);
                row_max[r] = mx;
            }
        }

        /// In-place `exp(scores - m[r])` inside `[lo[r], hi[r])` and exact 0.0 outside (over
        /// the full `tn` columns), adding each row's sum into `l_state[r]`.
        pub fn expNormalizeRangeF32(
            scores: []f32,
            rows: usize,
            score_stride: usize,
            tn: usize,
            lo: []const u32,
            hi: []const u32,
            m: []const f32,
            l_state: []f32,
        ) void {
            const VecT = @Vector(lanes, f32);

            var r: usize = 0;
            while (r < rows) : (r += 1) {
                const base: [*]f32 = scores.ptr + r * score_stride;
                const c0: usize = @min(lo[r], tn);
                const c1: usize = @min(hi[r], tn);

                @memset(base[0..c0], 0.0);
                @memset(base[c1..tn], 0.0);
                if (c1 <= c0) continue;

                const mr: f32 = m[r];
                const v_m: VecT = @splat(mr);
                var v_sum: VecT = @splat(0.0);
                var ssum: f32 = 0.0;

                var c: usize = c0;
                while (c + lanes <= c1) : (c += lanes) {
                    const ptr: [*]f32 = base + c;
                    const v_s: VecT = @as(*align(1) const VecT, @ptrCast(ptr)).*;
                    const v_p: VecT = fast_math.expApproxVecF32(lanes, fast_math.clampVecF32(lanes, v_s - v_m, -80.0, 0.0));
                    @as(*align(1) VecT, @ptrCast(ptr)).* = v_p;
                    v_sum += v_p;
                }
                ssum += @reduce(.Add, v_sum);
                while (c < c1) : (c += 1) {
                    const p: f32 = fast_math.expApproxF32(fast_math.clampF32(base[c] - mr, -80.0, 0.0));
                    base[c] = p;
                    ssum += p;
                }

                l_state[r] += ssum;
            }
        }

        /// `scores[r, lo[r]..hi[r]] = cap * tanh(scores / cap)` — Gemma-style logit soft cap,
        /// applied to the raw logits before the softmax.
        pub fn softCapRangeF32(
            scores: []f32,
            rows: usize,
            score_stride: usize,
            lo: []const u32,
            hi: []const u32,
            cap: f32,
        ) void {
            const VecT = @Vector(lanes, f32);
            const inv_cap: f32 = 1.0 / cap;

            var r: usize = 0;
            while (r < rows) : (r += 1) {
                const base: [*]f32 = scores.ptr + r * score_stride;
                const c0: usize = lo[r];
                const c1: usize = hi[r];
                if (c1 <= c0) continue;

                const v_cap: VecT = @splat(cap);
                const v_inv: VecT = @splat(inv_cap);
                var c: usize = c0;
                while (c + lanes <= c1) : (c += lanes) {
                    const ptr: [*]f32 = base + c;
                    const v_s: VecT = @as(*align(1) const VecT, @ptrCast(ptr)).*;
                    @as(*align(1) VecT, @ptrCast(ptr)).* = v_cap * fast_math.tanhApproxVecF32(lanes, v_s * v_inv);
                }
                while (c < c1) : (c += 1) base[c] = cap * fast_math.tanhApproxF32(base[c] * inv_cap);
            }
        }

        /// `acc[r, 0..dv] *= factors[r]` — the online-softmax rescale applied when a key
        /// block raises a row's running max.
        pub fn rescaleRowsF32(acc: []f32, rows: usize, dv: usize, acc_stride: usize, factors: []const f32) void {
            const VecT = @Vector(lanes, f32);

            var r: usize = 0;
            while (r < rows) : (r += 1) {
                const f: f32 = factors[r];
                if (f == 1.0) continue;
                const base: [*]f32 = acc.ptr + r * acc_stride;
                const vf: VecT = @splat(f);
                var j: usize = 0;
                while (j + lanes <= dv) : (j += lanes) {
                    const ptr: [*]f32 = base + j;
                    @as(*align(1) VecT, @ptrCast(ptr)).* = @as(*align(1) const VecT, @ptrCast(ptr)).* * vf;
                }
                while (j < dv) : (j += 1) base[j] *= f;
            }
        }

        pub fn rowMaxF32(scores: []const f32, m_tile: usize, tn: usize, row_max: []f32) void {
            const VecT = @Vector(lanes, f32);

            var r: usize = 0;
            while (r < m_tile) : (r += 1) {
                var mx: f32 = -std.math.inf(f32);
                var c: usize = 0;
                if (tn >= lanes) {
                    var vmx: VecT = @splat(-std.math.inf(f32));
                    while (c + lanes <= tn) : (c += lanes) {
                        const v: VecT = @as(*align(1) const VecT, @ptrCast(scores.ptr + r * tn + c)).*;
                        vmx = @max(vmx, v);
                    }
                    mx = @reduce(.Max, vmx);
                }
                while (c < tn) : (c += 1) {
                    mx = @max(mx, scores[r * tn + c]);
                }
                row_max[r] = mx;
            }
        }

        pub fn expNormalizeScoresF32(
            scores: []f32,
            m_tile: usize,
            tn: usize,
            m_new: []const f32,
            l_state: []f32,
        ) void {
            const VecT = @Vector(lanes, f32);

            var r: usize = 0;
            while (r < m_tile) : (r += 1) {
                const mn: f32 = m_new[r];
                var ssum: f32 = 0.0;

                var c: usize = 0;
                const v_mn: VecT = @splat(mn);
                var v_ssum: VecT = @splat(0.0);
                while (c + lanes <= tn) : (c += lanes) {
                    const ptr: [*]f32 = scores.ptr + r * tn + c;
                    const v_s: VecT = @as(*align(1) const VecT, @ptrCast(ptr)).*;
                    const v_diff: VecT = fast_math.clampVecF32(lanes, v_s - v_mn, -80.0, 0.0);
                    const v_p: VecT = fast_math.expApproxVecF32(lanes, v_diff);
                    @as(*align(1) VecT, @ptrCast(ptr)).* = v_p;
                    v_ssum += v_p;
                }
                ssum += @reduce(.Add, v_ssum);
                while (c < tn) : (c += 1) {
                    const idx: usize = r * tn + c;
                    const p: f32 = fast_math.expApproxF32(fast_math.clampF32(scores[idx] - mn, -80.0, 0.0));
                    scores[idx] = p;
                    ssum += p;
                }

                l_state[r] += ssum;
            }
        }
    };
}
