// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Row-wise i32 sum reduction over the last axis. Used for attention masks and
// other integer-derived lengths; integer mean is intentionally unsupported.

@group(0) @binding(0) var<storage, read>       x: array<i32>;
@group(0) @binding(1) var<storage, read_write> o: array<i32>;
@group(0) @binding(2) var<uniform>             p: Params;

struct Params { rows: u32, cols: u32, x_row: u32 };

const WG: u32 = 256u;
var<workgroup> scratch: array<i32, 256>;

@compute @workgroup_size(256)
fn reduce_sum_row(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let xb = wid.x * p.x_row;
    var total = 0;
    for (var c = lidx; c < p.cols; c += WG) {
        total += x[xb + c];
    }
    scratch[lidx] = total;
    workgroupBarrier();

    var stride = WG / 2u;
    while (stride > 0u) {
        if (lidx < stride) {
            scratch[lidx] += scratch[lidx + stride];
        }
        workgroupBarrier();
        stride /= 2u;
    }
    if (lidx == 0u) {
        o[wid.x] = scratch[0];
    }
}
