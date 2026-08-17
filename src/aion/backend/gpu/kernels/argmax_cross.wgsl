// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<i32>;
@group(0) @binding(2) var<uniform> p: Params;

struct Params {
    rows: u32, cols: u32, x_row: u32, parts: u32,
    part: u32, col_base: u32, pad0: u32, pad1: u32,
};

const WG: u32 = 256u;
const FMIN: f32 = -3.4028235e38;
var<workgroup> vals: array<f32, 256>;
var<workgroup> idxs: array<u32, 256>;

fn reduce_argmax(lidx: u32) {
    workgroupBarrier();
    var width = WG / 2u;
    while (width > 0u) {
        if (lidx < width) {
            let other_value = vals[lidx + width];
            let other_index = idxs[lidx + width];
            if (other_value > vals[lidx] || (other_value == vals[lidx] && other_index < idxs[lidx])) {
                vals[lidx] = other_value;
                idxs[lidx] = other_index;
            }
        }
        workgroupBarrier();
        width /= 2u;
    }
}

@compute @workgroup_size(256)
fn argmax_tile_partial(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let base = wid.x * p.x_row;
    var best_value = FMIN;
    var best_index = 0xffffffffu;
    for (var col = lidx; col < p.cols; col += WG) {
        let value = x[base + col];
        let index = p.col_base + col;
        if (value > best_value || (value == best_value && index < best_index)) {
            best_value = value;
            best_index = index;
        }
    }
    vals[lidx] = best_value;
    idxs[lidx] = best_index;
    reduce_argmax(lidx);
    if (lidx == 0u) {
        let dst = (p.part * p.rows + wid.x) * 2u;
        o[dst] = bitcast<i32>(vals[0]);
        o[dst + 1u] = i32(idxs[0]);
    }
}

@compute @workgroup_size(256)
fn argmax_tile_finish(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var best_value = FMIN;
    var best_index = 0xffffffffu;
    for (var part = lidx; part < p.parts; part += WG) {
        let src = (part * p.rows + wid.x) * 2u;
        let value = x[src];
        let index = bitcast<u32>(x[src + 1u]);
        if (value > best_value || (value == best_value && index < best_index)) {
            best_value = value;
            best_index = index;
        }
    }
    vals[lidx] = best_value;
    idxs[lidx] = best_index;
    reduce_argmax(lidx);
    if (lidx == 0u) { o[wid.x] = i32(idxs[0]); }
}
