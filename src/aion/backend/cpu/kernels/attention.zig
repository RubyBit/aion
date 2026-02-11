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

pub fn applyScaleMaskF32(
    scores: []f32,
    m_tile: usize,
    tn: usize,
    head_n: usize,
    local_key_row0: usize,
    local_q_row0: usize,
    scale: f32,
    causal: bool,
) void {
    var r: usize = 0;
    while (r < m_tile) : (r += 1) {
        const q_idx: usize = local_q_row0 + r;
        if (local_key_row0 >= head_n) continue;
        const valid_cols: usize = @min(tn, head_n - local_key_row0);
        var c: usize = 0;
        while (c < valid_cols) : (c += 1) {
            const k_idx: usize = local_key_row0 + c;
            const idx: usize = r * tn + c;
            var v0: f32 = scores[idx] * scale;
            if (causal and k_idx > q_idx) v0 = -std.math.inf(f32);
            scores[idx] = v0;
        }
        if (valid_cols < tn) {
            @memset(scores[r * tn + valid_cols .. r * tn + tn], -std.math.inf(f32));
        }
    }
}

pub fn rowMaxF32(scores: []const f32, m_tile: usize, tn: usize, row_max: []f32) void {
    const lanes: usize = comptime simd.lanesF32();
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
    const lanes: usize = comptime simd.lanesF32();
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
