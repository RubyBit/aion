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

@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
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

@compute @workgroup_size(256)
fn softmax_row(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let xb = wid.x * p.x_row;
    let ob = wid.x * p.o_row;

    var m = -3.4028235e38;
    for (var c = lidx; c < p.cols; c += WG) { m = max(m, x[xb + c]); }
    let row_max = wg_reduce_max(lidx, m);

    var s = 0.0;
    for (var c = lidx; c < p.cols; c += WG) {
        let e = exp(x[xb + c] - row_max);
        o[ob + c] = e;
        s += e;
    }
    let inv = 1.0 / wg_reduce_sum(lidx, s);

    for (var c = lidx; c < p.cols; c += WG) { o[ob + c] = o[ob + c] * inv; }
}
