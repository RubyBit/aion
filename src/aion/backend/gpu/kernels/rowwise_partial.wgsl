// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

enable f16;

@group(0) @binding(0) var<storage, read> x: array<f32>;
// f16 twins of the entry points below. Only the ELEMENT buffers change type:
// `stats` stays f32, because a partial row statistic is an accumulator and must
// not be rounded between the reduce and the apply stage.
@group(0) @binding(0) var<storage, read> xh: array<f16>;
@group(0) @binding(1) var<storage, read_write> stats: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

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

fn expApprox(x_in: f32) -> f32 {
    let v = clamp(x_in, -80.0, 80.0);
    let y = v * 1.4426950408889634;
    let n = i32(floor(y + 0.5));
    let t = (y - f32(n)) * 0.6931471805599453;
    let e2 = bitcast<f32>(u32(n + 127) << 23u);
    return e2 * (1.0 + t * (1.0 + t * (0.5 + t * (0.16666667 + t * 0.041666668))));
}

@compute @workgroup_size(256)
fn softmax_max_partial(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var value = -3.4028235e38;
    let base = wid.x * p.x_row;
    for (var col = lidx; col < p.cols; col += WG) { value = max(value, x[base + col]); }
    let result = reduce_max(lidx, value);
    if (lidx == 0u) { stats[p.part * p.rows + wid.x] = result; }
}

@compute @workgroup_size(256)
fn softmax_exp_partial(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let row_max = stats[p.stat_base + wid.x * 2u];
    let base = wid.x * p.x_row;
    var value = 0.0;
    for (var col = lidx; col < p.cols; col += WG) {
        value += expApprox(clamp(x[base + col] - row_max, -20.0, 0.0));
    }
    let result = reduce_sum(lidx, value);
    if (lidx == 0u) { stats[p.part * p.rows + wid.x] = result; }
}

@compute @workgroup_size(256)
fn norm_moments_partial(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let base = wid.x * p.x_row;
    var sum = 0.0;
    var sumsq = 0.0;
    for (var col = lidx; col < p.cols; col += WG) {
        let value = x[base + col];
        sum += value;
        sumsq += value * value;
    }
    let total = reduce_sum(lidx, sum);
    let total_sq = reduce_sum(lidx, sumsq);
    if (lidx == 0u) {
        let dst = (p.part * p.rows + wid.x) * 2u;
        stats[dst] = total;
        stats[dst + 1u] = total_sq;
    }
}

@compute @workgroup_size(256)
fn softmax_max_partial_f16(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var value = -3.4028235e38;
    let base = wid.x * p.x_row;
    for (var col = lidx; col < p.cols; col += WG) { value = max(value, f32(xh[base + col])); }
    let result = reduce_max(lidx, value);
    if (lidx == 0u) { stats[p.part * p.rows + wid.x] = result; }
}

@compute @workgroup_size(256)
fn softmax_exp_partial_f16(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let row_max = stats[p.stat_base + wid.x * 2u];
    let base = wid.x * p.x_row;
    var value = 0.0;
    for (var col = lidx; col < p.cols; col += WG) {
        value += expApprox(clamp(f32(xh[base + col]) - row_max, -20.0, 0.0));
    }
    let result = reduce_sum(lidx, value);
    if (lidx == 0u) { stats[p.part * p.rows + wid.x] = result; }
}

@compute @workgroup_size(256)
fn norm_moments_partial_f16(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let base = wid.x * p.x_row;
    var sum = 0.0;
    var sumsq = 0.0;
    for (var col = lidx; col < p.cols; col += WG) {
        let value = f32(xh[base + col]);
        sum += value;
        sumsq += value * value;
    }
    let total = reduce_sum(lidx, sum);
    let total_sq = reduce_sum(lidx, sumsq);
    if (lidx == 0u) {
        let dst = (p.part * p.rows + wid.x) * 2u;
        stats[dst] = total;
        stats[dst + 1u] = total_sq;
    }
}
