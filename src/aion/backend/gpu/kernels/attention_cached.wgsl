// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Cached grouped-query attention (the decode hot path):
//   out[b, l, hq, :] = softmax(scale * q[b, l, hq, :] @ K[b, hkv, t, :]^T) @ V[b, hkv, t, :]
// over the valid key range t in [lower, upper):
//   upper = end_index[b], clamped to positions[b, l] + 1 when causal;
//   lower = max(ring window start, sliding-window start).
// Optional logit soft cap: s = cap * tanh(s / cap).
//
// Layouts (all packed, single device buffer each):
//   q:   [tb, tl, th, dk] f32 — ONE out/q tile per dispatch (grid = th x tl x tb;
//        p.base_b / p.base_h give the tile's global offsets)
//   k/v: [B, H_kv, T, D] bound as array<u32> so f16 caches work WITHOUT the
//        shader-f16 extension: p.kv_f16 selects unpack2x16float (two elements
//        per word, D even) vs bitcast<f32> (one element per word).
//   positions: [tb, tl] i32 (tile-local), end_index: [B] i32 (global).
//
// Logical->physical time mapping runs in-kernel: identity (none/growable
// policies — the exec pre-touches growth before recording) or ring modulo.
//
// Work shape: one 256-thread workgroup per (b, l, hq) — same chunked online
// softmax as attention.wgsl (q row staged in shared, 256 keys scored per
// chunk, per-thread dv accumulators). Matches the CPU streaming executor,
// including writing zeros when the valid window is empty.

@group(0) @binding(0) var<storage, read>       q: array<f32>;
@group(0) @binding(1) var<storage, read>       kc: array<u32>;
@group(0) @binding(2) var<storage, read>       vc: array<u32>;
@group(0) @binding(3) var<storage, read>       pos: array<i32>;
@group(0) @binding(4) var<storage, read>       endi: array<i32>;
@group(0) @binding(5) var<storage, read_write> o: array<f32>;
@group(0) @binding(6) var<uniform>             p: Params;

struct Params {
    base_b: u32,
    base_h: u32,
    tl: u32, // q/out tile-local L count (grid.y extent)
    th: u32, // q/out tile-local head count (grid.x extent)
    dk: u32,
    dv: u32,
    t_cap: u32, // physical T of the caches
    h_kv: u32,
    gqa: u32, // query heads per kv head
    causal: u32,
    sliding: u32, // 0 = global attention
    ring: u32, // 0 = identity time map, 1 = ring
    ring_window: u32,
    kv_f16: u32,
    scale: f32,
    soft_cap: f32, // 0 = disabled
    segs: u32, // split-K segment count (mha_cached_split only)
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

const WG: u32 = 256u;
const MAX_DK: u32 = 512u;
const ACC: u32 = 4u; // dv <= ACC * WG
const FMIN: f32 = -3.4028235e38;

var<workgroup> q_s: array<f32, 512>;
var<workgroup> p_sh: array<f32, 256>;
var<workgroup> t_sh: array<u32, 256>; // physical time per scored key
var<workgroup> red: array<f32, 256>;

fn expApprox(x_in: f32) -> f32 {
    let xc = clamp(x_in, -80.0, 80.0);
    let yy = xc * 1.4426950408889634;
    let nn = i32(floor(yy + 0.5));
    let tt = (yy - f32(nn)) * 0.6931471805599453;
    let e2 = bitcast<f32>(u32(nn + 127) << 23u);
    return e2 * (1.0 + tt * (1.0 + tt * (0.5 + tt * (0.16666667 + tt * 0.041666668))));
}

fn tanhApprox(v: f32) -> f32 {
    var y: f32;
    let x2 = 2.0 * v;
    if (x2 >= 0.0) { y = 1.0 / (1.0 + expApprox(-x2)); } else { let e = expApprox(x2); y = e / (1.0 + e); }
    return clamp(2.0 * clamp(y, 0.0, 1.0) - 1.0, -1.0, 1.0);
}

fn wg_max(lidx: u32, x: f32) -> f32 {
    red[lidx] = x;
    workgroupBarrier();
    var s = WG / 2u;
    while (s > 0u) {
        if (lidx < s) { red[lidx] = max(red[lidx], red[lidx + s]); }
        workgroupBarrier();
        s = s / 2u;
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

fn wg_sum(lidx: u32, x: f32) -> f32 {
    red[lidx] = x;
    workgroupBarrier();
    var s = WG / 2u;
    while (s > 0u) {
        if (lidx < s) { red[lidx] = red[lidx] + red[lidx + s]; }
        workgroupBarrier();
        s = s / 2u;
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

@compute @workgroup_size(256)
fn mha_cached(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let b = p.base_b + wid.z;
    let hq = p.base_h + wid.x;
    let hkv = hq / p.gqa;

    let q_pos = u32(pos[wid.z * p.tl + wid.y]);
    let valid_end = u32(endi[b]);

    var upper = valid_end;
    if (p.causal != 0u) { upper = min(upper, q_pos + 1u); }
    var lower = 0u;
    if (p.ring != 0u && valid_end > p.ring_window) { lower = valid_end - p.ring_window; }
    if (p.sliding > 0u && q_pos + 1u > p.sliding) { lower = max(lower, q_pos + 1u - p.sliding); }

    // Stage the q row.
    let q_base = ((wid.z * p.tl + wid.y) * p.th + wid.x) * p.dk;
    for (var i = lidx; i < p.dk; i += WG) { q_s[i] = q[q_base + i]; }
    workgroupBarrier();

    var acc: array<f32, 4>;
    for (var i = 0u; i < ACC; i += 1u) { acc[i] = 0.0; }
    var m_state = FMIN;
    var l_state = 0.0;

    let kv_row = (b * p.h_kv + hkv) * p.t_cap;

    for (var t0 = lower; t0 < upper; t0 += WG) {
        let tj = t0 + lidx;
        var ok = tj < upper;

        var t_phys = 0u;
        if (ok) {
            t_phys = tj;
            if (p.ring != 0u) { t_phys = tj % p.ring_window; }
            ok = t_phys < p.t_cap;
        }

        var s = FMIN;
        if (ok) {
            let kb = (kv_row + t_phys) * p.dk;
            var dot = 0.0;
            if (p.kv_f16 != 0u) {
                // D even -> the row starts word-aligned; walk element pairs.
                let wb = kb / 2u;
                for (var i = 0u; i < p.dk; i += 2u) {
                    let both = unpack2x16float(kc[wb + i / 2u]);
                    dot += q_s[i] * both.x + q_s[i + 1u] * both.y;
                }
            } else {
                for (var i = 0u; i < p.dk; i += 1u) { dot += q_s[i] * bitcast<f32>(kc[kb + i]); }
            }
            s = dot * p.scale;
            if (p.soft_cap > 0.0) { s = p.soft_cap * tanhApprox(s / p.soft_cap); }
        }

        let m_new = max(m_state, wg_max(lidx, s));
        let rescale = expApprox(m_state - m_new);
        var prob = 0.0;
        if (ok) { prob = expApprox(s - m_new); }
        p_sh[lidx] = prob;
        t_sh[lidx] = t_phys;
        l_state = l_state * rescale + wg_sum(lidx, prob);
        m_state = m_new;
        for (var i = 0u; i < ACC; i += 1u) { acc[i] = acc[i] * rescale; }
        workgroupBarrier();

        let cnt = min(WG, upper - t0);
        for (var jj = 0u; jj < cnt; jj += 1u) {
            let pj = p_sh[jj];
            let vb = (kv_row + t_sh[jj]) * p.dv;
            if (p.kv_f16 != 0u) {
                let wb = vb / 2u;
                for (var i = 0u; i < ACC; i += 1u) {
                    let d = lidx + i * WG;
                    if (d < p.dv) {
                        let both = unpack2x16float(vc[wb + d / 2u]);
                        var ve = both.x;
                        if ((d & 1u) != 0u) { ve = both.y; }
                        acc[i] += pj * ve;
                    }
                }
            } else {
                for (var i = 0u; i < ACC; i += 1u) {
                    let d = lidx + i * WG;
                    if (d < p.dv) { acc[i] += pj * bitcast<f32>(vc[vb + d]); }
                }
            }
        }
        workgroupBarrier();
    }

    // Empty window (lower >= upper) leaves l_state == 0 -> write zeros, same as
    // the CPU executor's memset.
    var inv = 0.0;
    if (l_state > 0.0) { inv = 1.0 / l_state; }
    let o_base = ((wid.z * p.tl + wid.y) * p.th + wid.x) * p.dv;
    for (var i = 0u; i < ACC; i += 1u) {
        let d = lidx + i * WG;
        if (d < p.dv) { o[o_base + d] = acc[i] * inv; }
    }
}

// ---- split-K (flash-decoding) ------------------------------------------------
//
// Decode dispatches only B*1*H_q workgroups above — 8 workgroups scanning a
// 4 k-token cache leaves the GPU ~99% idle. The split path carves each row's
// valid key range into `p.segs` segments: grid = (th, tl, tb * segs), each
// workgroup runs the same online softmax over ITS segment and writes an
// UNNORMALIZED partial — dv accumulator values plus (m, l) — into `o` (bound
// to the backend scratch). `mha_cached_merge` (attention_merge.wgsl) then
// log-sum-exp-combines the segments per row. Entry stride is dv + 2 floats at
//   entry = ((b_local * tl + l) * th + h) * segs + seg.
@compute @workgroup_size(256)
fn mha_cached_split(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let b_local = wid.z / p.segs;
    let seg = wid.z % p.segs;
    let b = p.base_b + b_local;
    let hq = p.base_h + wid.x;
    let hkv = hq / p.gqa;

    let q_pos = u32(pos[b_local * p.tl + wid.y]);
    let valid_end = u32(endi[b]);

    var upper = valid_end;
    if (p.causal != 0u) { upper = min(upper, q_pos + 1u); }
    var lower = 0u;
    if (p.ring != 0u && valid_end > p.ring_window) { lower = valid_end - p.ring_window; }
    if (p.sliding > 0u && q_pos + 1u > p.sliding) { lower = max(lower, q_pos + 1u - p.sliding); }

    // This workgroup's slice of [lower, upper).
    var span = 0u;
    if (upper > lower) { span = upper - lower; }
    let seg_len = (span + p.segs - 1u) / p.segs;
    let s_lo = lower + seg * seg_len;
    let s_hi = min(s_lo + seg_len, upper);

    let q_base = ((b_local * p.tl + wid.y) * p.th + wid.x) * p.dk;
    for (var i = lidx; i < p.dk; i += WG) { q_s[i] = q[q_base + i]; }
    workgroupBarrier();

    var acc: array<f32, 4>;
    for (var i = 0u; i < ACC; i += 1u) { acc[i] = 0.0; }
    var m_state = FMIN;
    var l_state = 0.0;

    let kv_row = (b * p.h_kv + hkv) * p.t_cap;

    for (var t0 = s_lo; t0 < s_hi; t0 += WG) {
        let tj = t0 + lidx;
        var ok = tj < s_hi;

        var t_phys = 0u;
        if (ok) {
            t_phys = tj;
            if (p.ring != 0u) { t_phys = tj % p.ring_window; }
            ok = t_phys < p.t_cap;
        }

        var s = FMIN;
        if (ok) {
            let kb = (kv_row + t_phys) * p.dk;
            var dot = 0.0;
            if (p.kv_f16 != 0u) {
                let wb = kb / 2u;
                for (var i = 0u; i < p.dk; i += 2u) {
                    let both = unpack2x16float(kc[wb + i / 2u]);
                    dot += q_s[i] * both.x + q_s[i + 1u] * both.y;
                }
            } else {
                for (var i = 0u; i < p.dk; i += 1u) { dot += q_s[i] * bitcast<f32>(kc[kb + i]); }
            }
            s = dot * p.scale;
            if (p.soft_cap > 0.0) { s = p.soft_cap * tanhApprox(s / p.soft_cap); }
        }

        let m_new = max(m_state, wg_max(lidx, s));
        let rescale = expApprox(m_state - m_new);
        var prob = 0.0;
        if (ok) { prob = expApprox(s - m_new); }
        p_sh[lidx] = prob;
        t_sh[lidx] = t_phys;
        l_state = l_state * rescale + wg_sum(lidx, prob);
        m_state = m_new;
        for (var i = 0u; i < ACC; i += 1u) { acc[i] = acc[i] * rescale; }
        workgroupBarrier();

        let cnt = min(WG, s_hi - t0);
        for (var jj = 0u; jj < cnt; jj += 1u) {
            let pj = p_sh[jj];
            let vb = (kv_row + t_sh[jj]) * p.dv;
            if (p.kv_f16 != 0u) {
                let wb = vb / 2u;
                for (var i = 0u; i < ACC; i += 1u) {
                    let d = lidx + i * WG;
                    if (d < p.dv) {
                        let both = unpack2x16float(vc[wb + d / 2u]);
                        var ve = both.x;
                        if ((d & 1u) != 0u) { ve = both.y; }
                        acc[i] += pj * ve;
                    }
                }
            } else {
                for (var i = 0u; i < ACC; i += 1u) {
                    let d = lidx + i * WG;
                    if (d < p.dv) { acc[i] += pj * bitcast<f32>(vc[vb + d]); }
                }
            }
        }
        workgroupBarrier();
    }

    let stride = p.dv + 2u;
    let e = (((b_local * p.tl + wid.y) * p.th + wid.x) * p.segs + seg) * stride;
    for (var i = 0u; i < ACC; i += 1u) {
        let d = lidx + i * WG;
        if (d < p.dv) { o[e + d] = acc[i]; }
    }
    if (lidx == 0u) {
        o[e + p.dv] = m_state;
        o[e + p.dv + 1u] = l_state;
    }
}
