// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// The LSTM cell after its matmuls (exec/lstm.zig's GEMM route): `gates`
// [B, 4H] holds x @ w_ih + h_prev @ w_hh (gate order i, f, g, o), and one
// thread per (batch, hidden) element adds the biases and applies the same
// update as lstm_cell (lstm_act.wgsl), writing out[b] = [h_t | c_t]. f32 only.

@group(0) @binding(0) var<storage, read>       gates: array<f32>;
@group(0) @binding(1) var<storage, read>       c_prev: array<f32>;
@group(0) @binding(2) var<storage, read>       b_ih: array<f32>;
@group(0) @binding(3) var<storage, read>       b_hh: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
@group(0) @binding(5) var<uniform>             p: Params;

// Same layout as lstm.wgsl's (`LstmParams` in exec/lstm.zig).
struct Params {
    batch: u32,
    input_size: u32,
    hidden: u32,
    has_bias: u32,
    total: u32, // batch * hidden
};

const WG: u32 = 64u;

fn gate(row: u32, col: u32) -> f32 {
    var v = gates[row + col];
    if (p.has_bias != 0u) { v += b_ih[col] + b_hh[col]; }
    return v;
}

@compute @workgroup_size(64)
fn lstm_gates(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let h = p.hidden;
    for (var idx = gid.x; idx < p.total; idx += nwg.x * WG) {
        let b = idx / h;
        let j = idx % h;
        let row = b * 4u * h;
        let g = vec4<f32>(gate(row, j), gate(row, h + j), gate(row, 2u * h + j), gate(row, 3u * h + j));
        let hc = lstm_update(g, c_prev[b * h + j]);
        o[b * 2u * h + j] = hc.x;
        o[b * 2u * h + h + j] = hc.y;
    }
}
