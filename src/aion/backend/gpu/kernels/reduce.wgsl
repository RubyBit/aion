// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Row-wise sum/mean reduction over the last axis: one 256-thread workgroup per
// row writes o[row]. The backend also uses these for whole-tensor ReduceAll by
// passing rows = 1, cols = n (a single workgroup strides the entire tensor).

enable f16;

@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;

// f16 views of the same two bindings. A reduction has THREE dtype combinations,
// not one, because the staged paths write and then re-read f32 PARTIALS:
//   h2h  data -> result      (single-dispatch row reduce)
//   h2f  data -> f32 scratch (stage 1 of a staged reduce)
//   f2h  f32 scratch -> result (the fold)
// Partials are never rounded to f16: an accumulator that is read back and summed
// again must keep f32 precision, so only the DATA load and the RESULT store
// change type. Accumulation is f32 throughout, as on the CPU.
@group(0) @binding(0) var<storage, read>       xh: array<f16>;
@group(0) @binding(1) var<storage, read_write> oh: array<f16>;
@group(0) @binding(2) var<uniform>             p: Params;

struct Params { rows: u32, cols: u32, x_row: u32, o_base: u32 };

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
    if (lidx == 0u) { o[p.o_base + wid.x] = total; }
}

@compute @workgroup_size(256)
fn reduce_mean_row(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let total = row_sum(wid, lidx);
    if (lidx == 0u) { o[p.o_base + wid.x] = total / f32(p.cols); }
}

// Fold column-tile partials. Scratch is part-major:
// x[part * rows + row]. For mean, `o_base` carries the full logical row width;
// the final output tile itself always begins at element zero.
fn parts_sum(wid: vec3<u32>, lidx: u32) -> f32 {
    var value = 0.0;
    for (var part = lidx; part < p.cols; part += WG) {
        value += x[part * p.x_row + wid.x];
    }
    return wg_reduce_sum(lidx, value);
}

@compute @workgroup_size(256)
fn reduce_parts_sum(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let total = parts_sum(wid, lidx);
    if (lidx == 0u) { o[wid.x] = total; }
}

@compute @workgroup_size(256)
fn reduce_parts_mean(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let total = parts_sum(wid, lidx);
    if (lidx == 0u) { o[wid.x] = total / f32(p.o_base); }
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
    if (lidx == 0u) { o[p.o_base + wid.x] = total; }
}

// Stage-2 mean finish: sums the partials in x[0..cols) and divides by the
// ORIGINAL element count, which the backend passes in `x_row` (only row 0 is
// dispatched, so the stride field is otherwise unused — deliberate reuse).
@compute @workgroup_size(256)
fn reduce_mean_finish(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let total = row_sum(wid, lidx);
    if (lidx == 0u) { o[0] = total / f32(p.x_row); }
}

// ---- f16 reductions ---------------------------------------------------------

fn row_sum_h(wid: vec3<u32>, lidx: u32) -> f32 {
    let xb = wid.x * p.x_row;
    var s = 0.0;
    for (var c = lidx; c < p.cols; c += WG) { s += f32(xh[xb + c]); }
    return wg_reduce_sum(lidx, s);
}

@compute @workgroup_size(256)
fn reduce_sum_row_h2h(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let total = row_sum_h(wid, lidx);
    if (lidx == 0u) { oh[p.o_base + wid.x] = f16(total); }
}

@compute @workgroup_size(256)
fn reduce_mean_row_h2h(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let total = row_sum_h(wid, lidx);
    if (lidx == 0u) { oh[p.o_base + wid.x] = f16(total / f32(p.cols)); }
}

/// Stage 1 of a staged reduce: f16 data in, f32 partial out.
@compute @workgroup_size(256)
fn reduce_sum_row_h2f(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let total = row_sum_h(wid, lidx);
    if (lidx == 0u) { o[p.o_base + wid.x] = total; }
}

@compute @workgroup_size(256)
fn reduce_all_partial_h2f(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let step = nwg.x * WG;
    var s = 0.0;
    for (var i = wid.x * WG + lidx; i < p.cols; i += step) { s += f32(xh[i]); }
    let total = wg_reduce_sum(lidx, s);
    if (lidx == 0u) { o[p.o_base + wid.x] = total; }
}

/// Folds: f32 partials in, f16 result out.
@compute @workgroup_size(256)
fn reduce_sum_row_f2h(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let total = row_sum(wid, lidx);
    if (lidx == 0u) { oh[p.o_base + wid.x] = f16(total); }
}

@compute @workgroup_size(256)
fn reduce_mean_finish_f2h(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let total = row_sum(wid, lidx);
    if (lidx == 0u) { oh[0] = f16(total / f32(p.x_row)); }
}

@compute @workgroup_size(256)
fn reduce_parts_sum_f2h(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let total = parts_sum(wid, lidx);
    if (lidx == 0u) { oh[wid.x] = f16(total); }
}

@compute @workgroup_size(256)
fn reduce_parts_mean_f2h(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let total = parts_sum(wid, lidx);
    if (lidx == 0u) { oh[wid.x] = f16(total / f32(p.o_base)); }
}
