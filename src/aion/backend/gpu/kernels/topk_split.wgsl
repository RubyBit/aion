// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Top-k split stage 1. Grid is (segments, rows): workgroup (s, r) takes the top-k
// of row r's columns [s*seg, min((s+1)*seg, cols)) and writes k interleaved
// (value, GLOBAL column) pairs into scratch at entry (r*segments + s)*k. A segment
// with fewer than k entries pads with the NONE column, which the fold skips.
//
// The union of the segments' top-k contains the row's true top-k: at most k-1
// entries beat the k-th best globally, so at most k-1 beat it inside its own
// segment. Storing global columns keeps the fold's lowest-index tie-break exact.
//
// Separate module from topk.wgsl only because it binds two storage buffers rather
// than three, and the uniform always sits at the trailing binding. The helpers
// below are the same ones topk.wgsl declares.

enable f16;

@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(0) var<storage, read>       xh: array<f16>;
@group(0) @binding(1) var<storage, read_write> scratch: array<i32>;
@group(0) @binding(2) var<uniform>             p: Params;

struct Params {
    rows: u32,
    cols: u32,
    k: u32,
    largest: u32,
    x_row: u32,
    o_row: u32,
    groups_x: u32,
    seg: u32,
};

const WG: u32 = 256u;
const NONE: u32 = 0xffffffffu;

var<workgroup> vals: array<f32, 256>;
var<workgroup> idxs: array<u32, 256>;
var<workgroup> pick_v: f32;
var<workgroup> pick_i: u32;

fn is_better(av: f32, ai: u32, bv: f32, bi: u32) -> bool {
    if (ai == NONE) { return false; }
    if (bi == NONE) { return true; }
    if (av != bv) {
        if (p.largest != 0u) { return av > bv; }
        return av < bv;
    }
    return ai < bi;
}

fn commit_round(lidx: u32, bv: f32, bi: u32) {
    vals[lidx] = bv;
    idxs[lidx] = bi;
    workgroupBarrier();
    var s = WG / 2u;
    while (s > 0u) {
        if (lidx < s) {
            if (is_better(vals[lidx + s], idxs[lidx + s], vals[lidx], idxs[lidx])) {
                vals[lidx] = vals[lidx + s];
                idxs[lidx] = idxs[lidx + s];
            }
        }
        workgroupBarrier();
        s = s / 2u;
    }
    if (lidx == 0u) {
        pick_v = vals[0];
        pick_i = idxs[0];
    }
    workgroupBarrier();
}

@compute @workgroup_size(256)
fn topk_partial(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let r = wid.y;
    let xb = r * p.x_row;
    let c0 = wid.x * p.seg;
    let c1 = min(c0 + p.seg, p.cols);
    let base = (r * nwg.x + wid.x) * p.k;

    var last_v: f32 = 0.0;
    var last_i: u32 = NONE;
    var have_last = false;

    for (var t = 0u; t < p.k; t = t + 1u) {
        var bv: f32 = 0.0;
        var bi: u32 = NONE;
        for (var j = c0 + lidx; j < c1; j = j + WG) {
            let v = x[xb + j];
            if (have_last && !is_better(last_v, last_i, v, j)) { continue; }
            if (is_better(v, j, bv, bi)) { bv = v; bi = j; }
        }
        commit_round(lidx, bv, bi);
        if (lidx == 0u) {
            scratch[2u * (base + t)] = bitcast<i32>(pick_v);
            scratch[2u * (base + t) + 1u] = i32(pick_i);
        }
        last_v = pick_v;
        last_i = pick_i;
        have_last = true;
    }
}

@compute @workgroup_size(256)
fn topk_partial_f16(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let r = wid.y;
    let xb = r * p.x_row;
    let c0 = wid.x * p.seg;
    let c1 = min(c0 + p.seg, p.cols);
    let base = (r * nwg.x + wid.x) * p.k;

    var last_v: f32 = 0.0;
    var last_i: u32 = NONE;
    var have_last = false;

    for (var t = 0u; t < p.k; t = t + 1u) {
        var bv: f32 = 0.0;
        var bi: u32 = NONE;
        for (var j = c0 + lidx; j < c1; j = j + WG) {
            let v = f32(xh[xb + j]);
            if (have_last && !is_better(last_v, last_i, v, j)) { continue; }
            if (is_better(v, j, bv, bi)) { bv = v; bi = j; }
        }
        commit_round(lidx, bv, bi);
        if (lidx == 0u) {
            // The scratch is f32 regardless of the input dtype: the fold compares
            // widened values, and rounding happens once, on the final store.
            scratch[2u * (base + t)] = bitcast<i32>(pick_v);
            scratch[2u * (base + t) + 1u] = i32(pick_i);
        }
        last_v = pick_v;
        last_i = pick_i;
        have_last = true;
    }
}
