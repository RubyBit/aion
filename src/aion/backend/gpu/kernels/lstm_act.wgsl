// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// The LSTM cell's activations and state update, shared by lstm.wgsl and
// lstm_gates.wgsl (exec/lstm.zig appends this file to each).
//
// They match the CPU LSTM's fast transcendental approximations (fast_math.zig)
// so the autoregressive RNNT greedy decode makes the SAME emit-vs-blank argmax
// decisions as the CPU reference: exact exp/tanh here diverge just enough to
// double tokens.

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

// (h_t, c_t) from the gate pre-activations (i, f, g, o) and c_prev.
fn lstm_update(g: vec4<f32>, c_prev: f32) -> vec2<f32> {
    let c_t = sigmoid(g.y) * c_prev + sigmoid(g.x) * tanhApprox(g.z);
    return vec2<f32>(sigmoid(g.w) * tanhApprox(c_t), c_t);
}
