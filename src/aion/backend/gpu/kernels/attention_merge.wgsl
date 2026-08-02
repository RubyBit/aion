// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Split-K merge for cached attention (see attn_split in
// attention.wgsl): per output row, log-sum-exp-combine the segments'
// unnormalized partials into the final normalized row.
//
// parts layout: entry (row * segs + seg) has stride (dv + 2) floats —
// dv accumulator values, then m (running max) and l (softmax denominator at m).
// A segment that saw no valid keys wrote m = -FLT_MAX, l = 0 and drops out of
// the combine; if EVERY segment is empty l_tot stays 0 and the row is zeroed
// (same contract as the single-stage kernel and the CPU executor).

@group(0) @binding(0) var<storage, read>       parts: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@group(0) @binding(2) var<uniform>             p: Params;

struct Params { rows: u32, segs: u32, dv: u32, stride: u32 };

const WG: u32 = 256u;
const FMIN: f32 = -3.4028235e38;

fn expApprox(x_in: f32) -> f32 {
    let xc = clamp(x_in, -80.0, 80.0);
    let yy = xc * 1.4426950408889634;
    let nn = i32(floor(yy + 0.5));
    let tt = (yy - f32(nn)) * 0.6931471805599453;
    let e2 = bitcast<f32>(u32(nn + 127) << 23u);
    return e2 * (1.0 + tt * (1.0 + tt * (0.5 + tt * (0.16666667 + tt * 0.041666668))));
}

@compute @workgroup_size(256)
fn attn_merge(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let r = wid.x;
    let base = r * p.segs * p.stride;

    // Scalar pass over the (m, l) tails — redundant per thread, but segs is
    // small (<= 64) and this avoids any cross-thread reduction.
    var m_tot = FMIN;
    for (var s = 0u; s < p.segs; s += 1u) {
        m_tot = max(m_tot, parts[base + s * p.stride + p.dv]);
    }
    var l_tot = 0.0;
    for (var s = 0u; s < p.segs; s += 1u) {
        let m_s = parts[base + s * p.stride + p.dv];
        let l_s = parts[base + s * p.stride + p.dv + 1u];
        l_tot += l_s * expApprox(m_s - m_tot);
    }
    var inv = 0.0;
    if (l_tot > 0.0) { inv = 1.0 / l_tot; }

    for (var d = lidx; d < p.dv; d += WG) {
        var acc = 0.0;
        for (var s = 0u; s < p.segs; s += 1u) {
            let m_s = parts[base + s * p.stride + p.dv];
            acc += parts[base + s * p.stride + d] * expApprox(m_s - m_tot);
        }
        o[r * p.dv + d] = acc * inv;
    }
}
