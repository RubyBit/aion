// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Fused residual + RMSNorm over the trailing dim (the `AddRMSNormTiled` step):
//   o = addend + (rmsnorm(x) * gamma + beta)
// One 256-thread workgroup per row.
//
// The reduction structure is identical to norm.wgsl's `rmsnorm_row` — the fusion only
// adds the residual read to the apply loop — so the norm itself matches the standalone
// kernel bit-exactly. f32 addition commutes exactly, so it also does not matter which
// operand of the original elementwise add was the residual.
//
// Why fuse at all: the model runs 105 of these pairs per token (three per layer, the
// "sandwich norm" residuals), and each one was two dispatches on the serial residual
// chain. The saving is the second launch plus a round trip of the intermediate through
// memory, not arithmetic.

@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       gm: array<f32>;
@group(0) @binding(2) var<storage, read>       bt: array<f32>;
@group(0) @binding(3) var<storage, read>       addend: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
@group(0) @binding(5) var<uniform>             p: Params;

struct Params { rows: u32, cols: u32, x_row: u32, o_row: u32, a_row: u32, eps: f32 };

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
    workgroupBarrier();
    return r;
}

@compute @workgroup_size(256)
fn add_rmsnorm_row(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let xb = wid.x * p.x_row;
    let ob = wid.x * p.o_row;
    let ab = wid.x * p.a_row;

    var ss = 0.0;
    for (var c = lidx; c < p.cols; c += WG) { let v = x[xb + c]; ss += v * v; }
    let sumsq = wg_reduce_sum(lidx, ss);
    let inv = 1.0 / sqrt(sumsq / f32(p.cols) + p.eps);

    for (var c = lidx; c < p.cols; c += WG) {
        o[ob + c] = addend[ab + c] + (x[xb + c] * inv * gm[c] + bt[c]);
    }
}
