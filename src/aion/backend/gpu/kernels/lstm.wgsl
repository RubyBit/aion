// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Fused single-timestep LSTM cell, one thread per (batch, hidden) element:
//   gates[k] = b_ih[kH+j] + b_hh[kH+j] + x[b] @ w_ih[:, kH+j] + h_prev[b] @ w_hh[:, kH+j]
//   (gate order i, f, g, o — PyTorch chunks of H)
//   c_t = sigmoid(f) * c_prev + sigmoid(i) * tanh(g)
//   h_t = sigmoid(o) * tanh(c_t)
//   out[b] = [h_t | c_t]           ([batch, 2H])
//
// All tensors packed single tiles: x [B, I], h/c [B, H], w_ih [I, 4H],
// w_hh [H, 4H], biases [4H]. When p.has_bias == 0 the bias bindings are
// dummies (the backend rebinds w_ih). Weight reads are column-strided —
// correctness-first; pre-transposed weights are the perf follow-up if LSTM
// models show up hot on GPU.

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
// cell update all stay f32 — exactly like the CPU cell, which reads and writes
// through f32 scratch and only knows the dtype in its four I/O helpers
// (cpu/exec/lstm.zig). So the two backends compute the same cell and differ only
// in when they round.
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

// Match the CPU LSTM's fast transcendental approximations (fast_math.zig) so the
// autoregressive RNNT greedy decode makes the SAME emit-vs-blank argmax decisions
// as the CPU reference — exact exp/tanh here diverge just enough to double tokens.
fn expApprox(x_in: f32) -> f32 {
    let x = clamp(x_in, -80.0, 80.0);
    let y = x * 1.4426950408889634;              // x / ln2
    let n = i32(floor(y + 0.5));
    let t = (y - f32(n)) * 0.6931471805599453;   // r * ln2, r in [-0.5, 0.5]
    let exp2i = bitcast<f32>(u32(n + 127) << 23u); // 2^n
    // 4th-order Taylor of exp(t) on t in [-0.35, 0.35].
    let poly = 1.0 + t * (1.0 + t * (0.5 + t * (0.16666667 + t * 0.041666668)));
    return exp2i * poly;
}

fn sigmoid(v: f32) -> f32 {
    var y: f32;
    if (v >= 0.0) {
        y = 1.0 / (1.0 + expApprox(-v));
    } else {
        let e = expApprox(v);
        y = e / (1.0 + e);
    }
    return clamp(y, 0.0, 1.0);
}

fn tanhApprox(v: f32) -> f32 {
    return clamp(2.0 * sigmoid(2.0 * v) - 1.0, -1.0, 1.0);
}

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

        let i_gate = sigmoid(g[0]);
        let f_gate = sigmoid(g[1]);
        let g_gate = tanhApprox(g[2]);
        let o_gate = sigmoid(g[3]);

        let c_t = f_gate * c_prev[b * p.hidden + j] + i_gate * g_gate;
        let h_t = o_gate * tanhApprox(c_t);

        let ob = b * 2u * p.hidden;
        o[ob + j] = h_t;
        o[ob + p.hidden + j] = c_t;
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

        let i_gate = sigmoid(g[0]);
        let f_gate = sigmoid(g[1]);
        let g_gate = tanhApprox(g[2]);
        let o_gate = sigmoid(g[3]);

        let c_t = f_gate * f32(c_prev_h[b * p.hidden + j]) + i_gate * g_gate;
        let h_t = o_gate * tanhApprox(c_t);

        let ob = b * 2u * p.hidden;
        oh[ob + j] = f16(h_t);
        oh[ob + p.hidden + j] = f16(c_t);
    }
}
