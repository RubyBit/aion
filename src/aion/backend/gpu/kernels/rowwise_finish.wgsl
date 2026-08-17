// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

@group(0) @binding(0) var<storage, read_write> stats: array<f32>;
@group(0) @binding(1) var<uniform> p: Params;

struct Params {
    rows: u32, cols: u32, x_row: u32, o_row: u32,
    parts: u32, part: u32, col_base: u32, full_cols: u32,
    stat_base: u32, mode: u32, eps: f32, pad: u32,
};

const WG: u32 = 256u;
var<workgroup> scratch: array<f32, 256>;

fn reduce_sum(lidx: u32, value: f32) -> f32 {
    scratch[lidx] = value;
    workgroupBarrier();
    var width = WG / 2u;
    while (width > 0u) {
        if (lidx < width) { scratch[lidx] += scratch[lidx + width]; }
        workgroupBarrier();
        width /= 2u;
    }
    let result = scratch[0];
    workgroupBarrier();
    return result;
}

fn reduce_max(lidx: u32, value: f32) -> f32 {
    scratch[lidx] = value;
    workgroupBarrier();
    var width = WG / 2u;
    while (width > 0u) {
        if (lidx < width) { scratch[lidx] = max(scratch[lidx], scratch[lidx + width]); }
        workgroupBarrier();
        width /= 2u;
    }
    let result = scratch[0];
    workgroupBarrier();
    return result;
}

@compute @workgroup_size(256)
fn softmax_max_finish(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var value = -3.4028235e38;
    for (var part = lidx; part < p.parts; part += WG) { value = max(value, stats[part * p.rows + wid.x]); }
    let result = reduce_max(lidx, value);
    if (lidx == 0u) { stats[p.stat_base + wid.x * 2u] = result; }
}

@compute @workgroup_size(256)
fn softmax_sum_finish(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var value = 0.0;
    for (var part = lidx; part < p.parts; part += WG) { value += stats[part * p.rows + wid.x]; }
    let result = reduce_sum(lidx, value);
    if (lidx == 0u) { stats[p.stat_base + wid.x * 2u + 1u] = result; }
}

@compute @workgroup_size(256)
fn norm_finish(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var sum = 0.0;
    var sumsq = 0.0;
    for (var part = lidx; part < p.parts; part += WG) {
        let src = (part * p.rows + wid.x) * 2u;
        sum += stats[src];
        sumsq += stats[src + 1u];
    }
    let total = reduce_sum(lidx, sum);
    let total_sq = reduce_sum(lidx, sumsq);
    if (lidx == 0u) {
        let mean = total / f32(p.full_cols);
        let mean_sq = total_sq / f32(p.full_cols);
        let variance = select(mean_sq, max(0.0, mean_sq - mean * mean), p.mode == 1u);
        let dst = p.stat_base + wid.x * 2u;
        stats[dst] = select(0.0, mean, p.mode == 1u);
        stats[dst + 1u] = inverseSqrt(variance + p.eps);
    }
}
