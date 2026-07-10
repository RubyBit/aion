// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Fused attention over one slice: out = softmax(scale * q @ k^T) @ v, with an
// optional causal mask (key j > query i masked). Serves both `Attention` and
// `MultiHeadAttention` — the backend dispatches once per lead-dim slice (batch/
// head), which the compiler tiles as size-1, so the kernel only ever sees
//   q: [m, dk]   k: [n, dk]   v: [n, dv]   out: [m, dv]     (f32, row strides
// in elements — tiles may pad rows).
//
// Work shape: one 256-thread workgroup PER QUERY ROW, streaming keys in chunks
// of 256 with an online (flash-style) softmax:
//   - the q row is staged in shared memory (dk <= 512),
//   - each chunk: thread j scores key t0+j (dot + scale + mask) -> shared,
//     the chunk max/sum fold into the running (m, l) state via tree reduces,
//   - probabilities accumulate into per-thread output registers strided over
//     dv (ACC * 256 lanes -> dv <= 1024),
//   - final pass normalizes by l and writes the row.
// Numerically this matches the CPU streaming-softmax executor: one pass over
// keys, exp(x - running_max), rescale-on-new-max.

@group(0) @binding(0) var<storage, read>       q: array<f32>;
@group(0) @binding(1) var<storage, read>       k: array<f32>;
@group(0) @binding(2) var<storage, read>       v: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@group(0) @binding(4) var<uniform>             p: Params;

struct Params {
    m: u32,
    n: u32,
    dk: u32,
    dv: u32,
    q_row: u32, // row strides in elements
    k_row: u32,
    v_row: u32,
    o_row: u32,
    scale: f32,
    causal: u32,
};

const WG: u32 = 256u;
const MAX_DK: u32 = 512u;
const ACC: u32 = 4u; // dv <= ACC * WG
const FMIN: f32 = -3.4028235e38;

var<workgroup> q_s: array<f32, 512>;
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
fn attn_row_f32(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let row = wid.x;
    // Stage the q row (all threads participate; rows are guaranteed < m).
    let qb = row * p.q_row;
    for (var i = lidx; i < p.dk; i += WG) { q_s[i] = q[qb + i]; }
    workgroupBarrier();

    var kmax = p.n;
    if (p.causal != 0u) { kmax = min(kmax, row + 1u); }

    var acc: array<f32, 4>;
    for (var i = 0u; i < ACC; i += 1u) { acc[i] = 0.0; }
    var m_state = FMIN;
    var l_state = 0.0;

    for (var t0 = 0u; t0 < kmax; t0 += WG) {
        let j = t0 + lidx;

        // Score one key per thread.
        var s = FMIN;
        if (j < kmax) {
            let kb = j * p.k_row;
            var dot = 0.0;
            for (var i = 0u; i < p.dk; i += 1u) { dot += q_s[i] * k[kb + i]; }
            s = dot * p.scale;
        }

        // Online softmax update. exp(FMIN - m_new) underflows to 0, which is
        // exactly the "no previous state" rescale.
        let m_new = max(m_state, wg_max(lidx, s));
        let rescale = expApprox(m_state - m_new);
        var prob = 0.0;
        if (j < kmax) { prob = expApprox(s - m_new); }
        p_sh[lidx] = prob;
        l_state = l_state * rescale + wg_sum(lidx, prob);
        m_state = m_new;
        for (var i = 0u; i < ACC; i += 1u) { acc[i] = acc[i] * rescale; }
        workgroupBarrier(); // p_sh visible to all lanes

        // out += P @ V for this chunk; threads stride the dv axis.
        let cnt = min(WG, kmax - t0);
        for (var jj = 0u; jj < cnt; jj += 1u) {
            let pj = p_sh[jj];
            let vb = (t0 + jj) * p.v_row;
            for (var i = 0u; i < ACC; i += 1u) {
                let d = lidx + i * WG;
                if (d < p.dv) { acc[i] += pj * v[vb + d]; }
            }
        }
        workgroupBarrier(); // before the next chunk overwrites p_sh/red
    }

    let inv = 1.0 / l_state;
    let ob = row * p.o_row;
    for (var i = 0u; i < ACC; i += 1u) {
        let d = lidx + i * WG;
        if (d < p.dv) { o[ob + d] = acc[i] * inv; }
    }
}
