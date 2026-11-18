// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Fused single-timestep LSTM cell, one thread per (batch, hidden) element:
//   gates[k] = b_ih[kH+j] + b_hh[kH+j] + x[b] @ w_ih[:, kH+j] + h_prev[b] @ w_hh[:, kH+j]
//   (gate order i, f, g, o — PyTorch chunks of H)
//   c_t = sigmoid(f) * c_prev + sigmoid(i) * tanh(g)
//   h_t = sigmoid(o) * tanh(c_t)
//   out[b] = [h_t | c_t]           ([batch, 2H])
//
// All tensors packed single buffers: x [B, I], h/c [B, H], w_ih [I, 4H],
// w_hh [H, 4H], biases [4H]. When p.has_bias == 0 the bias bindings are
// dummies (the backend rebinds w_ih). Weight reads are column-strided, which
// suits a few rows; a larger f32 batch takes the GEMM route (lstm_gates.wgsl).
// The activations and the update are lstm_act.wgsl, appended to this file.

enable f16;

@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       h_prev: array<f32>;
@group(0) @binding(2) var<storage, read>       c_prev: array<f32>;
@group(0) @binding(3) var<storage, read>       w_ih: array<f32>;
@group(0) @binding(4) var<storage, read>       w_hh: array<f32>;
@group(0) @binding(5) var<storage, read>       b_ih: array<f32>;
@group(0) @binding(6) var<storage, read>       b_hh: array<f32>;
@group(0) @binding(7) var<storage, read_write> o: array<f32>;

// f16 twin of every operand. The gate accumulators, the sigmoid/tanh and the
// cell update all stay f32, exactly like the CPU cell, which widens its
// operands and computes in f32 (cpu/exec/lstm.zig). So the two backends compute
// the same cell and differ only in when they round.
@group(0) @binding(0) var<storage, read>       xh: array<f16>;
@group(0) @binding(1) var<storage, read>       h_prev_h: array<f16>;
@group(0) @binding(2) var<storage, read>       c_prev_h: array<f16>;
@group(0) @binding(3) var<storage, read>       w_ih_h: array<f16>;
@group(0) @binding(4) var<storage, read>       w_hh_h: array<f16>;
@group(0) @binding(5) var<storage, read>       b_ih_h: array<f16>;
@group(0) @binding(6) var<storage, read>       b_hh_h: array<f16>;
@group(0) @binding(7) var<storage, read_write> oh: array<f16>;
@group(0) @binding(8) var<uniform>             p: Params;

struct Params {
    batch: u32,
    input_size: u32,
    hidden: u32,
    has_bias: u32,
    total: u32, // batch * hidden
};

const WG: u32 = 64u;

@compute @workgroup_size(64)
fn lstm_cell(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let stride = nwg.x * WG;
    let gate_dim = 4u * p.hidden;

    for (var idx = gid.x; idx < p.total; idx += stride) {
        let b = idx / p.hidden;
        let j = idx % p.hidden;

        var g: array<f32, 4>;
        for (var kk = 0u; kk < 4u; kk += 1u) {
            let col = kk * p.hidden + j;
            var acc = 0.0;
            if (p.has_bias != 0u) { acc = b_ih[col] + b_hh[col]; }
            let xb = b * p.input_size;
            for (var t = 0u; t < p.input_size; t += 1u) {
                acc += x[xb + t] * w_ih[t * gate_dim + col];
            }
            let hb = b * p.hidden;
            for (var t = 0u; t < p.hidden; t += 1u) {
                acc += h_prev[hb + t] * w_hh[t * gate_dim + col];
            }
            g[kk] = acc;
        }

        let hc = lstm_update(vec4<f32>(g[0], g[1], g[2], g[3]), c_prev[b * p.hidden + j]);
        let ob = b * 2u * p.hidden;
        o[ob + j] = hc.x;
        o[ob + p.hidden + j] = hc.y;
    }
}

@compute @workgroup_size(64)
fn lstm_cell_f16(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let stride = nwg.x * WG;
    let gate_dim = 4u * p.hidden;

    for (var idx = gid.x; idx < p.total; idx += stride) {
        let b = idx / p.hidden;
        let j = idx % p.hidden;

        var g: array<f32, 4>;
        for (var kk = 0u; kk < 4u; kk += 1u) {
            let col = kk * p.hidden + j;
            var acc = 0.0;
            if (p.has_bias != 0u) { acc = f32(b_ih_h[col]) + f32(b_hh_h[col]); }
            let xb = b * p.input_size;
            for (var t = 0u; t < p.input_size; t += 1u) {
                acc += f32(xh[xb + t]) * f32(w_ih_h[t * gate_dim + col]);
            }
            let hb = b * p.hidden;
            for (var t = 0u; t < p.hidden; t += 1u) {
                acc += f32(h_prev_h[hb + t]) * f32(w_hh_h[t * gate_dim + col]);
            }
            g[kk] = acc;
        }

        let hc = lstm_update(vec4<f32>(g[0], g[1], g[2], g[3]), f32(c_prev_h[b * p.hidden + j]));
        let ob = b * 2u * p.hidden;
        oh[ob + j] = f16(hc.x);
        oh[ob + p.hidden + j] = f16(hc.y);
    }
}
