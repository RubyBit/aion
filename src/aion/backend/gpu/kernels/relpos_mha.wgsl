// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Relative-positional multi-head self-attention (Transformer-XL / Conformer),
// one (batch, head) slice per dispatch:
//   scores[i,j] = ((q[i]+u) . k[j] + (q[i]+v) . pos_emb[base_i + j]) * scale
//                 + mask[i,j]                       (base_i = (T_q-1) - i)
//   out[i] = softmax_j(scores[i,:]) @ v
//
// Same work shape as attention.wgsl: one 256-thread workgroup per query row,
// chunked online softmax, per-thread dv accumulators. The two biased q rows
// (q+u and q+v) are staged in shared memory once per row; the rel-shift is the
// contiguous pos_emb band [base_i .. base_i + T_kv) — no [T_q, P] matrix.
//
// Layouts (per the compile contract): q/k/v/out slices are contiguous [T, D]
// panels (the [B, T, H, D] tensors are tiled [1, T, 1, D]); pos_emb / biases
// are indexed via p.pe_base / p.u_base / p.v_base element offsets into their
// (per-head or whole) tiles; mask is a packed additive [T_q, T_kv] tile. When
// p.has_mask == 0 the mask binding is a dummy (the backend rebinds q).

@group(0) @binding(0) var<storage, read>       q: array<f32>;
@group(0) @binding(1) var<storage, read>       k: array<f32>;
@group(0) @binding(2) var<storage, read>       v: array<f32>;
@group(0) @binding(3) var<storage, read>       pe: array<f32>;
@group(0) @binding(4) var<storage, read>       bu: array<f32>;
@group(0) @binding(5) var<storage, read>       bv: array<f32>;
@group(0) @binding(6) var<storage, read>       mask: array<f32>;
@group(0) @binding(7) var<storage, read_write> o: array<f32>;
@group(0) @binding(8) var<uniform>             p: Params;

struct Params {
    t_q: u32,
    t_kv: u32,
    d: u32,
    pe_base: u32, // element offset of this head's [P, D] panel
    u_base: u32, // element offset of this head's [D] bias row
    v_base: u32,
    has_mask: u32,
    _pad: u32,
    scale: f32,
};

const WG: u32 = 256u;
const MAX_D: u32 = 512u;
const ACC: u32 = 4u; // d <= ACC * WG
const FMIN: f32 = -3.4028235e38;

var<workgroup> qu_s: array<f32, 512>; // q[i] + pos_bias_u
var<workgroup> qv_s: array<f32, 512>; // q[i] + pos_bias_v
var<workgroup> p_sh: array<f32, 256>;
var<workgroup> red: array<f32, 256>;

fn expApprox(x_in: f32) -> f32 {
    let xc = clamp(x_in, -80.0, 80.0);
    let yy = xc * 1.4426950408889634;
    let nn = i32(floor(yy + 0.5));
    let tt = (yy - f32(nn)) * 0.6931471805599453;
    let e2 = bitcast<f32>(u32(nn + 127) << 23u);
    return e2 * (1.0 + tt * (1.0 + tt * (0.5 + tt * (0.16666667 + tt * 0.041666668))));
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
fn relpos_mha_row(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let row = wid.x;
    let qb = row * p.d;
    for (var i = lidx; i < p.d; i += WG) {
        let qi = q[qb + i];
        qu_s[i] = qi + bu[p.u_base + i];
        qv_s[i] = qi + bv[p.v_base + i];
    }
    workgroupBarrier();

    let base_i = (p.t_q - 1u) - row; // rel-shift band start for this query row

    var acc: array<f32, 4>;
    for (var i = 0u; i < ACC; i += 1u) { acc[i] = 0.0; }
    var m_state = FMIN;
    var l_state = 0.0;

    for (var t0 = 0u; t0 < p.t_kv; t0 += WG) {
        let j = t0 + lidx;

        var s = FMIN;
        if (j < p.t_kv) {
            let kb = j * p.d;
            let peb = p.pe_base + (base_i + j) * p.d;
            var ac = 0.0;
            var bd = 0.0;
            for (var i = 0u; i < p.d; i += 1u) {
                ac += qu_s[i] * k[kb + i];
                bd += qv_s[i] * pe[peb + i];
            }
            s = (ac + bd) * p.scale;
            if (p.has_mask != 0u) { s += mask[row * p.t_kv + j]; }
        }

        let m_new = max(m_state, wg_max(lidx, s));
        let rescale = expApprox(m_state - m_new);
        var prob = 0.0;
        if (j < p.t_kv) { prob = expApprox(s - m_new); }
        p_sh[lidx] = prob;
        l_state = l_state * rescale + wg_sum(lidx, prob);
        m_state = m_new;
        for (var i = 0u; i < ACC; i += 1u) { acc[i] = acc[i] * rescale; }
        workgroupBarrier();

        let cnt = min(WG, p.t_kv - t0);
        for (var jj = 0u; jj < cnt; jj += 1u) {
            let pj = p_sh[jj];
            let vb = (t0 + jj) * p.d;
            for (var i = 0u; i < ACC; i += 1u) {
                let dd = lidx + i * WG;
                if (dd < p.d) { acc[i] += pj * v[vb + dd]; }
            }
        }
        workgroupBarrier();
    }

    let inv = 1.0 / l_state;
    let ob = row * p.d;
    for (var i = 0u; i < ACC; i += 1u) {
        let dd = lidx + i * WG;
        if (dd < p.d) { o[ob + dd] = acc[i] * inv; }
    }
}
