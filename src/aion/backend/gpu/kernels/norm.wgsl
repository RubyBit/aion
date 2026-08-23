// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Row-wise LayerNorm / RMSNorm over the trailing dim: one 256-thread workgroup
// per row, shared-memory reduction for the row statistics, then the scaled
// affine apply. Matches the CPU kernels:
//   layernorm: (x - mean) / sqrt(max(0, E[x^2] - mean^2) + eps) * gamma + beta
//   rmsnorm:    x         / sqrt(E[x^2] + eps)                 * gamma + beta
// The backend guarantees the whole normalized axis lives in this tile and that
// gamma/beta are single-tile vectors of `cols` elements.

enable f16;

@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       gm: array<f32>;
@group(0) @binding(2) var<storage, read>       bt: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
// f16 twins alias `array<f16>` onto the f32 bindings (legal per entry point; see
// the shared-binding test). Row statistics stay in f32 exactly as the CPU kernels
// keep them, so only the load widens and the store rounds.
@group(0) @binding(0) var<storage, read>       xh: array<f16>;
@group(0) @binding(1) var<storage, read>       gmh: array<f16>;
@group(0) @binding(2) var<storage, read>       bth: array<f16>;
@group(0) @binding(3) var<storage, read_write> oh: array<f16>;

@group(0) @binding(4) var<uniform>             p: Params;

struct Params { rows: u32, cols: u32, x_row: u32, o_row: u32, eps: f32 };

const WG: u32 = 256u;
var<workgroup> scratch: array<f32, 256>;

fn wg_reduce_sum(lidx: u32, v: f32) -> f32 {
    scratch[lidx] = v;
    workgroupBarrier();
    var s = WG / 2u;
    while (s > 0u) {
        if (lidx < s) { scratch[lidx] = scratch[lidx] + scratch[lidx + s]; }
        workgroupBarrier();
        s = s / 2u;
    }
    let r = scratch[0];
    workgroupBarrier(); // scratch is reused by the next reduction
    return r;
}

@compute @workgroup_size(256)
fn rmsnorm_row(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let xb = wid.x * p.x_row;
    let ob = wid.x * p.o_row;

    var ss = 0.0;
    for (var c = lidx; c < p.cols; c += WG) { let v = x[xb + c]; ss += v * v; }
    let sumsq = wg_reduce_sum(lidx, ss);
    let inv = 1.0 / sqrt(sumsq / f32(p.cols) + p.eps);

    for (var c = lidx; c < p.cols; c += WG) {
        o[ob + c] = x[xb + c] * inv * gm[c] + bt[c];
    }
}

@compute @workgroup_size(256)
fn layernorm_row(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let xb = wid.x * p.x_row;
    let ob = wid.x * p.o_row;

    var s = 0.0;
    var ss = 0.0;
    for (var c = lidx; c < p.cols; c += WG) {
        let v = x[xb + c];
        s += v;
        ss += v * v;
    }
    let mu = wg_reduce_sum(lidx, s) / f32(p.cols);
    let msq = wg_reduce_sum(lidx, ss) / f32(p.cols);
    let inv = 1.0 / sqrt(max(0.0, msq - mu * mu) + p.eps);

    for (var c = lidx; c < p.cols; c += WG) {
        o[ob + c] = (x[xb + c] - mu) * inv * gm[c] + bt[c];
    }
}

@compute @workgroup_size(256)
fn rmsnorm_row_f16(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let xb = wid.x * p.x_row;
    let ob = wid.x * p.o_row;

    var ss = 0.0;
    for (var c = lidx; c < p.cols; c += WG) { let v = f32(xh[xb + c]); ss += v * v; }
    let sumsq = wg_reduce_sum(lidx, ss);
    let inv = 1.0 / sqrt(sumsq / f32(p.cols) + p.eps);

    for (var c = lidx; c < p.cols; c += WG) {
        oh[ob + c] = f16(f32(xh[xb + c]) * inv * f32(gmh[c]) + f32(bth[c]));
    }
}

@compute @workgroup_size(256)
fn layernorm_row_f16(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let xb = wid.x * p.x_row;
    let ob = wid.x * p.o_row;

    var s = 0.0;
    var ss = 0.0;
    for (var c = lidx; c < p.cols; c += WG) {
        let v = f32(xh[xb + c]);
        s += v;
        ss += v * v;
    }
    let mu = wg_reduce_sum(lidx, s) / f32(p.cols);
    let msq = wg_reduce_sum(lidx, ss) / f32(p.cols);
    let inv = 1.0 / sqrt(max(0.0, msq - mu * mu) + p.eps);

    for (var c = lidx; c < p.cols; c += WG) {
        oh[ob + c] = f16((f32(xh[xb + c]) - mu) * inv * f32(gmh[c]) + f32(bth[c]));
    }
}
