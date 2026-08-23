// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

enable f16;

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read> stats: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
// f16 twins of the entry points below. Only the ELEMENT buffers change type:
// `stats` stays f32, because a partial row statistic is an accumulator and must
// not be rounded between the reduce and the apply stage.
@group(0) @binding(0) var<storage, read> xh: array<f16>;
@group(0) @binding(1) var<storage, read> gammah: array<f16>;
@group(0) @binding(2) var<storage, read> betah: array<f16>;
@group(0) @binding(4) var<storage, read_write> oh: array<f16>;
@group(0) @binding(5) var<uniform> p: Params;

struct Params {
    rows: u32, cols: u32, x_row: u32, o_row: u32,
    parts: u32, part: u32, col_base: u32, full_cols: u32,
    stat_base: u32, mode: u32, eps: f32, pad: u32,
};

const WG: u32 = 256u;

@compute @workgroup_size(256)
fn norm_apply(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let mean = stats[p.stat_base + wid.x * 2u];
    let inv = stats[p.stat_base + wid.x * 2u + 1u];
    let xb = wid.x * p.x_row;
    let ob = wid.x * p.o_row;
    for (var col = lidx; col < p.cols; col += WG) {
        o[ob + col] = (x[xb + col] - mean) * inv * gamma[p.col_base + col] + beta[p.col_base + col];
    }
}

@compute @workgroup_size(256)
fn norm_apply_f16(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let mean = stats[p.stat_base + wid.x * 2u];
    let inv = stats[p.stat_base + wid.x * 2u + 1u];
    let xb = wid.x * p.x_row;
    let ob = wid.x * p.o_row;
    for (var col = lidx; col < p.cols; col += WG) {
        oh[ob + col] = f16((f32(xh[xb + col]) - mean) * inv * f32(gammah[p.col_base + col]) + f32(betah[p.col_base + col]));
    }
}
