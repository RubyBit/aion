// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Row-wise argmax over the last axis, i32 output index per row. Ties resolve
// to the LOWEST index (matching the CPU scan) in every path.
//
// Two shapes of dispatch:
//   - argmax_row: one 256-thread workgroup per row — fine when there are many
//     rows.
//   - argmax_partial + argmax_finish: the decode shape is ONE row of a whole
//     vocab (256k logits) — a single workgroup leaves the GPU idle. The split
//     path carves each row into contiguous column segments, one workgroup per
//     (segment, row) writing an interleaved (value, column) pair into scratch,
//     then one workgroup per row folds the partials. Global column indexes are
//     stored, so the fold's index tie-break preserves the lowest-index rule.

enable f16;

@group(0) @binding(0) var<storage, read>       x: array<f32>;

// f16 input twin. Only the DATA side changes: the output is i32 indices either
// way, and the split path keeps its partial VALUES in f32 scratch, so stage 2
// (`argmax_finish`) is shared verbatim. Comparing widened f32 preserves f16
// ordering and the lowest-index tie-break, so the chosen index is identical.
@group(0) @binding(0) var<storage, read>       xh: array<f16>;
@group(0) @binding(1) var<storage, read_write> o: array<i32>;
@group(0) @binding(2) var<uniform>             p: Params;

// Same layout as the reduce/softmax params. `o_row` carries the segment width
// for argmax_partial (deliberate field reuse); `cols` is the segment count for
// argmax_finish.
struct Params { rows: u32, cols: u32, x_row: u32, o_row: u32 };

const WG: u32 = 256u;
const FMIN: f32 = -3.4028235e38;

var<workgroup> vals: array<f32, 256>;
var<workgroup> idxs: array<u32, 256>;

// Tree-reduce the (vals, idxs) workgroup arrays down to slot 0 (max value,
// lowest index on ties). Callers seed their slot and this barriers first.
fn wg_argmax_reduce(lidx: u32) {
    workgroupBarrier();
    var s = WG / 2u;
    while (s > 0u) {
        if (lidx < s) {
            let v2 = vals[lidx + s];
            let i2 = idxs[lidx + s];
            if (v2 > vals[lidx] || (v2 == vals[lidx] && i2 < idxs[lidx])) {
                vals[lidx] = v2;
                idxs[lidx] = i2;
            }
        }
        workgroupBarrier();
        s = s / 2u;
    }
}

@compute @workgroup_size(256)
fn argmax_row(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let xb = wid.x * p.x_row;

    var best_v = FMIN;
    var best_i = 0xffffffffu; // any real column beats the sentinel on ties
    for (var j = lidx; j < p.cols; j += WG) {
        let v = x[xb + j];
        if (v > best_v || (v == best_v && j < best_i)) {
            best_v = v;
            best_i = j;
        }
    }
    vals[lidx] = best_v;
    idxs[lidx] = best_i;
    wg_argmax_reduce(lidx);

    if (lidx == 0u) { o[wid.x] = i32(idxs[0]); }
}

// Split stage 1: grid (segments, rows). Workgroup (s, r) scans columns
// [s*o_row, min((s+1)*o_row, cols)) of row r and writes its (value, column)
// into o (bound to scratch) as the interleaved pair at entry r*segments + s.
@compute @workgroup_size(256)
fn argmax_partial(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let r = wid.y;
    let xb = r * p.x_row;
    let c0 = wid.x * p.o_row;
    let c1 = min(c0 + p.o_row, p.cols);

    var best_v = FMIN;
    var best_i = 0xffffffffu;
    for (var j = c0 + lidx; j < c1; j += WG) {
        let v = x[xb + j];
        if (v > best_v || (v == best_v && j < best_i)) {
            best_v = v;
            best_i = j;
        }
    }
    vals[lidx] = best_v;
    idxs[lidx] = best_i;
    wg_argmax_reduce(lidx);

    if (lidx == 0u) {
        let e = r * nwg.x + wid.x;
        o[2u * e] = bitcast<i32>(vals[0]);
        o[2u * e + 1u] = i32(idxs[0]);
    }
}

// Split stage 2: one workgroup per row folds that row's `cols` partial pairs
// (x bound to scratch — the f32 read of slot 2e recovers the stage-1 value
// bits, slot 2e+1 the global column).
@compute @workgroup_size(256)
fn argmax_finish(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let r = wid.x;

    var best_v = FMIN;
    var best_i = 0xffffffffu;
    for (var s = lidx; s < p.cols; s += WG) {
        let e = r * p.cols + s;
        let v = x[2u * e];
        let i = bitcast<u32>(x[2u * e + 1u]);
        if (v > best_v || (v == best_v && i < best_i)) {
            best_v = v;
            best_i = i;
        }
    }
    vals[lidx] = best_v;
    idxs[lidx] = best_i;
    wg_argmax_reduce(lidx);

    if (lidx == 0u) { o[r] = i32(idxs[0]); }
}

@compute @workgroup_size(256)
fn argmax_row_f16(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let xb = wid.x * p.x_row;

    var best_v = FMIN;
    var best_i = 0xffffffffu;
    for (var j = lidx; j < p.cols; j += WG) {
        let v = f32(xh[xb + j]);
        if (v > best_v || (v == best_v && j < best_i)) {
            best_v = v;
            best_i = j;
        }
    }
    vals[lidx] = best_v;
    idxs[lidx] = best_i;
    wg_argmax_reduce(lidx);

    if (lidx == 0u) { o[wid.x] = i32(idxs[0]); }
}

@compute @workgroup_size(256)
fn argmax_partial_f16(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let r = wid.y;
    let xb = r * p.x_row;
    let c0 = wid.x * p.o_row;
    let c1 = min(c0 + p.o_row, p.cols);

    var best_v = FMIN;
    var best_i = 0xffffffffu;
    for (var j = c0 + lidx; j < c1; j += WG) {
        let v = f32(xh[xb + j]);
        if (v > best_v || (v == best_v && j < best_i)) {
            best_v = v;
            best_i = j;
        }
    }
    vals[lidx] = best_v;
    idxs[lidx] = best_i;
    wg_argmax_reduce(lidx);

    if (lidx == 0u) {
        let e = r * nwg.x + wid.x;
        o[2u * e] = bitcast<i32>(vals[0]);
        o[2u * e + 1u] = i32(idxs[0]);
    }
}
