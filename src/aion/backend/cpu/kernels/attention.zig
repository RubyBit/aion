const std = @import("std");
const fast_math = @import("fast_math.zig");
const simd = @import("simd.zig");

pub const simd_lanes: usize = simd.lanesF32();
pub const Vec = @Vector(simd_lanes, f32);

pub inline fn vecLoad(ptr: [*]align(1) const f32) Vec {
    return @as(*align(1) const [simd_lanes]f32, @ptrCast(ptr)).*;
}

pub inline fn vecStore(ptr: [*]align(1) f32, v: Vec) void {
    @as(*align(1) [simd_lanes]f32, @ptrCast(ptr)).* = v;
}

pub fn expSoftmax(x: f32) f32 {
    if (!std.math.isFinite(x)) {
        if (x < 0.0) return 0.0;
        return std.math.inf(f32);
    }
    return fast_math.expApproxF32(fast_math.clampF32(x, -20.0, 0.0));
}

fn packQBlock(
    rows: usize,
    cols: usize,
    src: []align(1) const f32,
    src_stride: usize,
    dst: []f32,
) void {
    const MR: usize = 6;
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
    const NR: usize = 16;
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
    kt: []f32,
    qt: []f32,
    scores: []f32,
    score_stride: usize,
) void {
    const MR: usize = 6;
    const NR: usize = 16;

    packQBlock(m_q, k_dim, qs, q_stride, qt);

    var r0: usize = 0;
    while (r0 < m_q) : (r0 += MR) {
        const r_end: usize = @min(r0 + MR, m_q);
        var c0: usize = 0;
        while (c0 < n_k) : (c0 += NR) {
            if (c0 + NR > n_k or r0 + MR > m_q) {
                var r: usize = r0;
                while (r < r_end) : (r += 1) {
                    const q_p: []align(1) const f32 = qs[r * q_stride ..];
                    const s_p: []f32 = scores[r * score_stride ..];
                    _ = q_p;
                    _ = s_p;
                }
                continue;
            }

            var vScores0: [MR]Vec = undefined;
            var vScores1: [MR]Vec = undefined;
            var i: usize = 0;
            while (i < MR) : (i += 1) {
                vScores0[i] = @splat(0.0);
                vScores1[i] = @splat(0.0);
            }

            const q_ptr_base: [*]f32 = qt[(r0 / MR) * (k_dim * MR) ..].ptr;
            const k_ptr_base: [*]f32 = kt[(c0 / NR) * (k_dim * NR) ..].ptr;

            var k: usize = 0;
            while (k < k_dim) : (k += 1) {
                const vk_ptr: [*]f32 = k_ptr_base + k * NR;
                const vK0: Vec = vecLoad(vk_ptr);
                const vK1: Vec = vecLoad(vk_ptr + 8);

                const q_k_ptr: [*]f32 = q_ptr_base + k * MR;

                inline for (0..MR) |ii| {
                    const vQ: Vec = @as(Vec, @splat(q_k_ptr[ii]));
                    vScores0[ii] += vQ * vK0;
                    vScores1[ii] += vQ * vK1;
                }
            }

            var r: usize = r0;
            while (r < r_end) : (r += 1) {
                const ri: usize = r - r0;
                const ptr: [*]f32 = scores[r * score_stride + c0 ..].ptr;
                vecStore(ptr, vecLoad(ptr) + vScores0[ri]);
                vecStore(ptr + 8, vecLoad(ptr + 8) + vScores1[ri]);
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
    const MR: usize = 8;
    var j: usize = 0;
    while (j < dv) : (j += simd_lanes) {
        if (j + simd_lanes > dv) {
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

            var vAcc: [MR]Vec = undefined;
            var r: usize = r0;
            while (r < r1) : (r += 1) {
                const ri: usize = r - r0;
                vAcc[ri] = vecLoad(acc[r * acc_stride + j ..].ptr);
            }

            var k: usize = 0;
            while (k < n_v) : (k += 1) {
                const vV: Vec = vecLoad(vs[k * v_stride + j ..].ptr);
                r = r0;
                while (r < r1) : (r += 1) {
                    const ri: usize = r - r0;
                    vAcc[ri] += @as(Vec, @splat(scores[r * score_stride + k])) * vV;
                }
            }

            r = r0;
            while (r < r1) : (r += 1) {
                const ri: usize = r - r0;
                vecStore(acc[r * acc_stride + j ..].ptr, vAcc[ri]);
            }
        }
    }
}
