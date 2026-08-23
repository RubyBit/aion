// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Row-wise softmax over the last axis: one 256-thread workgroup per row, with
// shared-memory tree reductions for the row max and the exp-sum. Numerically
// matches the CPU kernel (exp(x - max) / sum). The backend guarantees the whole
// reduced axis lives in this tile (tile_counts[axis] == 1) and rows fit the
// dispatch limit, so a single dispatch handles one tile of `rows` rows.
//
// Three column sweeps per row (max, exp+sum, normalize); x is read twice, o is
// written then rescaled in place — same structure as the CPU three-pass kernel.

enable f16;

@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;

// f16 twin. NOTE the different pass structure: the f32 kernel parks exp(x - max)
// in `o` and rescales it in place, which an f16 output must not do — exp spans
// [2.06e-9, 1] after the -20 clamp, and f16 goes subnormal below 6.10e-5 and
// flushes to zero below 5.96e-8, so the un-normalized intermediate would be
// quantized before it could be divided. Instead the sum pass stores nothing and
// the final pass recomputes exp, so each probability is rounded to f16 exactly
// once, on its finished value. `sumExpF16` / `expNormalizeStoreF16` in
// cpu/kernels/softmax.zig do precisely the same, for the same reason.
@group(0) @binding(0) var<storage, read>       xh: array<f16>;
@group(0) @binding(1) var<storage, read_write> oh: array<f16>;

@group(0) @binding(2) var<uniform>             p: Params;

// Row strides are in elements (tiles may pad rows; the backend passes both).
struct Params { rows: u32, cols: u32, x_row: u32, o_row: u32 };

const WG: u32 = 256u;
var<workgroup> scratch: array<f32, 256>;

fn wg_reduce_max(lidx: u32, v: f32) -> f32 {
    scratch[lidx] = v;
    workgroupBarrier();
    var s = WG / 2u;
    while (s > 0u) {
        if (lidx < s) { scratch[lidx] = max(scratch[lidx], scratch[lidx + s]); }
        workgroupBarrier();
        s = s / 2u;
    }
    let r = scratch[0];
    workgroupBarrier(); // scratch is reused by the next reduction
    return r;
}

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

// Match the CPU's fast exp approximation (fast_math.expApproxF32) so GPU results
// track the CPU reference bit-closely — the exact vs approx gap is small per-op
// but biases downstream argmax/greedy decode (e.g. RNNT). See lstm.wgsl.
fn expApprox(x_in: f32) -> f32 {
    let x = clamp(x_in, -80.0, 80.0);
    let y = x * 1.4426950408889634;
    let n = i32(floor(y + 0.5));
    let t = (y - f32(n)) * 0.6931471805599453;
    let e2 = bitcast<f32>(u32(n + 127) << 23u);
    return e2 * (1.0 + t * (1.0 + t * (0.5 + t * (0.16666667 + t * 0.041666668))));
}

@compute @workgroup_size(256)
fn softmax_row(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let xb = wid.x * p.x_row;
    let ob = wid.x * p.o_row;

    var m = -3.4028235e38;
    for (var c = lidx; c < p.cols; c += WG) { m = max(m, x[xb + c]); }
    let row_max = wg_reduce_max(lidx, m);

    var s = 0.0;
    for (var c = lidx; c < p.cols; c += WG) {
        let e = expApprox(clamp(x[xb + c] - row_max, -20.0, 0.0));
        o[ob + c] = e;
        s += e;
    }
    let inv = 1.0 / wg_reduce_sum(lidx, s);

    for (var c = lidx; c < p.cols; c += WG) { o[ob + c] = o[ob + c] * inv; }
}

@compute @workgroup_size(256)
fn softmax_row_f16(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let xb = wid.x * p.x_row;
    let ob = wid.x * p.o_row;

    var m = -3.4028235e38;
    for (var c = lidx; c < p.cols; c += WG) { m = max(m, f32(xh[xb + c])); }
    let row_max = wg_reduce_max(lidx, m);

    var s = 0.0;
    for (var c = lidx; c < p.cols; c += WG) {
        s += expApprox(clamp(f32(xh[xb + c]) - row_max, -20.0, 0.0));
    }
    let inv = 1.0 / wg_reduce_sum(lidx, s);

    for (var c = lidx; c < p.cols; c += WG) {
        oh[ob + c] = f16(expApprox(clamp(f32(xh[xb + c]) - row_max, -20.0, 0.0)) * inv);
    }
}
