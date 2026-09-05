// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Row-wise top-k over the last axis: `k` values and their i32 indices per row,
// sorted best-first, ties to the LOWEST index — the same total order the CPU
// kernel uses, so the two cannot disagree about a tie.
//
// A round takes the best entry STRICTLY WORSE than the previous round's pick, which
// frees the rounds from carrying an "already chosen" set: the order is total, so
// "worse than the last pick" identifies exactly the remaining candidates in O(1)
// per element. The row stays hot in cache across rounds, which is why re-reading it
// beats spilling a per-thread heap.
//
// Two dispatch shapes, mirroring argmax:
//   - topk_row: one workgroup per row. Fine when there are many rows.
//   - topk_partial (topk_split.wgsl) + topk_finish: the decode shape is ONE row of
//     a whole vocab, where k rounds on a single workgroup leave the GPU idle. Stage
//     1 takes the top-k of each column segment in parallel; this module folds them.
//     Stage 1 stores GLOBAL columns, so the fold's tie-break still resolves to the
//     lowest index.

enable f16;

@group(0) @binding(0) var<storage, read>       x: array<f32>;
// f16 twin. Only the load differs: comparisons happen on the widened f32, which
// preserves f16 ordering, so both paths select identical indices. The two views
// alias one binding, so no entry point may name both.
@group(0) @binding(0) var<storage, read>       xh: array<f16>;
@group(0) @binding(1) var<storage, read_write> ov: array<f32>;
@group(0) @binding(1) var<storage, read_write> ovh: array<f16>;
@group(0) @binding(2) var<storage, read_write> oi: array<i32>;
@group(0) @binding(3) var<uniform>             p: Params;

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

// Rows can exceed WebGPU's 65535 workgroups on X, so the grid is 2-D and the row
// is recovered from both axes.
fn row_index(wid: vec3<u32>) -> u32 {
    return wid.x + wid.y * p.groups_x;
}

// Is (av, ai) better than (bv, bi)? NONE means "no candidate", worse than all.
fn is_better(av: f32, ai: u32, bv: f32, bi: u32) -> bool {
    if (ai == NONE) { return false; }
    if (bi == NONE) { return true; }
    if (av != bv) {
        if (p.largest != 0u) { return av > bv; }
        return av < bv;
    }
    return ai < bi;
}

// Reduce this round's per-thread bests to `pick_v`/`pick_i`. Every thread calls it
// — the barriers inside must stay uniform, which is why the out-of-range rows
// below are masked rather than returned early.
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
fn topk_row(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let want = row_index(wid);
    let in_range = want < p.rows;
    // A masked lane still walks a real row so every load stays in bounds; only its
    // stores are dropped.
    let row = select(0u, want, in_range);
    let xb = row * p.x_row;
    let ob = row * p.o_row;

    var last_v: f32 = 0.0;
    var last_i: u32 = NONE;
    var have_last = false;

    for (var r = 0u; r < p.k; r = r + 1u) {
        var bv: f32 = 0.0;
        var bi: u32 = NONE;
        for (var j = lidx; j < p.cols; j = j + WG) {
            let v = x[xb + j];
            // Anything at least as good as the previous pick was already emitted.
            if (have_last && !is_better(last_v, last_i, v, j)) { continue; }
            if (is_better(v, j, bv, bi)) { bv = v; bi = j; }
        }
        commit_round(lidx, bv, bi);
        if (lidx == 0u && in_range) {
            ov[ob + r] = pick_v;
            oi[ob + r] = i32(pick_i);
        }
        last_v = pick_v;
        last_i = pick_i;
        have_last = true;
    }
}

@compute @workgroup_size(256)
fn topk_row_f16(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let want = row_index(wid);
    let in_range = want < p.rows;
    let row = select(0u, want, in_range);
    let xb = row * p.x_row;
    let ob = row * p.o_row;

    var last_v: f32 = 0.0;
    var last_i: u32 = NONE;
    var have_last = false;

    for (var r = 0u; r < p.k; r = r + 1u) {
        var bv: f32 = 0.0;
        var bi: u32 = NONE;
        for (var j = lidx; j < p.cols; j = j + WG) {
            let v = f32(xh[xb + j]);
            if (have_last && !is_better(last_v, last_i, v, j)) { continue; }
            if (is_better(v, j, bv, bi)) { bv = v; bi = j; }
        }
        commit_round(lidx, bv, bi);
        if (lidx == 0u && in_range) {
            ovh[ob + r] = f16(pick_v);
            oi[ob + r] = i32(pick_i);
        }
        last_v = pick_v;
        last_i = pick_i;
        have_last = true;
    }
}

// Split stage 2: one workgroup per row folds that row's `p.cols` candidate pairs
// (x bound to the stage-1 scratch: slot 2e is the value bits, 2e+1 the global
// column) down to the final k.
@compute @workgroup_size(256)
fn topk_finish(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let want = row_index(wid);
    let in_range = want < p.rows;
    let row = select(0u, want, in_range);
    let base = row * p.cols;
    let ob = row * p.o_row;

    var last_v: f32 = 0.0;
    var last_i: u32 = NONE;
    var have_last = false;

    for (var t = 0u; t < p.k; t = t + 1u) {
        var bv: f32 = 0.0;
        var bi: u32 = NONE;
        for (var s = lidx; s < p.cols; s = s + WG) {
            let e = base + s;
            let v = x[2u * e];
            let i = bitcast<u32>(x[2u * e + 1u]);
            if (i == NONE) { continue; }
            if (have_last && !is_better(last_v, last_i, v, i)) { continue; }
            if (is_better(v, i, bv, bi)) { bv = v; bi = i; }
        }
        commit_round(lidx, bv, bi);
        if (lidx == 0u && in_range) {
            ov[ob + t] = pick_v;
            oi[ob + t] = i32(pick_i);
        }
        last_v = pick_v;
        last_i = pick_i;
        have_last = true;
    }
}

// Same fold, f16 store. The scratch is f32 either way, so only the output differs.
@compute @workgroup_size(256)
fn topk_finish_f16(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let want = row_index(wid);
    let in_range = want < p.rows;
    let row = select(0u, want, in_range);
    let base = row * p.cols;
    let ob = row * p.o_row;

    var last_v: f32 = 0.0;
    var last_i: u32 = NONE;
    var have_last = false;

    for (var t = 0u; t < p.k; t = t + 1u) {
        var bv: f32 = 0.0;
        var bi: u32 = NONE;
        for (var s = lidx; s < p.cols; s = s + WG) {
            let e = base + s;
            let i = bitcast<u32>(x[2u * e + 1u]);
            if (i == NONE) { continue; }
            let v = x[2u * e];
            if (have_last && !is_better(last_v, last_i, v, i)) { continue; }
            if (is_better(v, i, bv, bi)) { bv = v; bi = i; }
        }
        commit_round(lidx, bv, bi);
        if (lidx == 0u && in_range) {
            ovh[ob + t] = f16(pick_v);
            oi[ob + t] = i32(pick_i);
        }
        last_v = pick_v;
        last_i = pick_i;
        have_last = true;
    }
}
