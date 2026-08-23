// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

enable f16;

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> stats: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
// f16 twins of the entry points below. Only the ELEMENT buffers change type:
// `stats` stays f32, because a partial row statistic is an accumulator and must
// not be rounded between the reduce and the apply stage.
@group(0) @binding(0) var<storage, read> xh: array<f16>;
@group(0) @binding(2) var<storage, read_write> oh: array<f16>;
@group(0) @binding(3) var<uniform> p: Params;

struct Params {
    rows: u32, cols: u32, x_row: u32, o_row: u32,
    parts: u32, part: u32, col_base: u32, full_cols: u32,
    stat_base: u32, mode: u32, eps: f32, pad: u32,
};

const WG: u32 = 256u;

fn expApprox(x_in: f32) -> f32 {
    let v = clamp(x_in, -80.0, 80.0);
    let y = v * 1.4426950408889634;
    let n = i32(floor(y + 0.5));
    let t = (y - f32(n)) * 0.6931471805599453;
    let e2 = bitcast<f32>(u32(n + 127) << 23u);
    return e2 * (1.0 + t * (1.0 + t * (0.5 + t * (0.16666667 + t * 0.041666668))));
}

@compute @workgroup_size(256)
fn softmax_apply(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let row_max = stats[p.stat_base + wid.x * 2u];
    let inv_sum = 1.0 / stats[p.stat_base + wid.x * 2u + 1u];
    let xb = wid.x * p.x_row;
    let ob = wid.x * p.o_row;
    for (var col = lidx; col < p.cols; col += WG) {
        o[ob + col] = expApprox(clamp(x[xb + col] - row_max, -20.0, 0.0)) * inv_sum;
    }
}

@compute @workgroup_size(256)
fn softmax_apply_f16(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let row_max = stats[p.stat_base + wid.x * 2u];
    let inv_sum = 1.0 / stats[p.stat_base + wid.x * 2u + 1u];
    let xb = wid.x * p.x_row;
    let ob = wid.x * p.o_row;
    for (var col = lidx; col < p.cols; col += WG) {
        oh[ob + col] = f16(expApprox(clamp(f32(xh[xb + col]) - row_max, -20.0, 0.0)) * inv_sum);
    }
}
