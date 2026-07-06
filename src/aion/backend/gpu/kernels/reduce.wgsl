// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Row-wise sum/mean reduction over the last axis: one 256-thread workgroup per
// row writes o[row]. The backend also uses these for whole-tensor ReduceAll by
// passing rows = 1, cols = n (a single workgroup strides the entire tensor).

@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@group(0) @binding(2) var<uniform>             p: Params;

struct Params { rows: u32, cols: u32, x_row: u32 };

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

fn row_sum(wid: vec3<u32>, lidx: u32) -> f32 {
    let xb = wid.x * p.x_row;
    var s = 0.0;
    for (var c = lidx; c < p.cols; c += WG) { s += x[xb + c]; }
    return wg_reduce_sum(lidx, s);
}

@compute @workgroup_size(256)
fn reduce_sum_row(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let total = row_sum(wid, lidx);
    if (lidx == 0u) { o[wid.x] = total; }
}

@compute @workgroup_size(256)
fn reduce_mean_row(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let total = row_sum(wid, lidx);
    if (lidx == 0u) { o[wid.x] = total / f32(p.cols); }
}

// ---- two-stage whole-tensor reduction --------------------------------------
//
// A single workgroup striding a multi-megabyte tensor leaves the GPU >95% idle
// (measured 5 GB/s on an RTX 4080). Stage 1 spreads the sweep over many
// workgroups, each summing a grid-strided slice into o[workgroup]; stage 2 runs
// `reduce_sum_row` (rows=1) over those partials. The backend sizes stage 1 so
// each thread handles a few dozen elements.

@compute @workgroup_size(256)
fn reduce_all_partial(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let step = nwg.x * WG;
    var s = 0.0;
    for (var i = wid.x * WG + lidx; i < p.cols; i += step) { s += x[i]; }
    let total = wg_reduce_sum(lidx, s);
    if (lidx == 0u) { o[wid.x] = total; }
}

// Stage-2 mean finish: sums the partials in x[0..cols) and divides by the
// ORIGINAL element count, which the backend passes in `x_row` (only row 0 is
// dispatched, so the stride field is otherwise unused — deliberate reuse).
@compute @workgroup_size(256)
fn reduce_mean_finish(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let total = row_sum(wid, lidx);
    if (lidx == 0u) { o[0] = total / f32(p.x_row); }
}
