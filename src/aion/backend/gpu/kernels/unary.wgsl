// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Elementwise unary activations over f32 storage buffers. One entry point per
// op; the backend selects from `UnaryOp`. f32 only; dispatched once per tile.
// The element count comes from a uniform (binding 2) rather than `arrayLength`.
// Grid-stride loops (see elementwise.wgsl) keep dispatches under the 65535
// workgroups-per-dimension limit on large GPU-policy tiles.

@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@group(0) @binding(2) var<uniform>             p: Params;

struct Params { n: u32 };

// Match the CPU's fast transcendental approximations (fast_math.zig) so activation
// outputs track the CPU reference — the exact-vs-approx gap is tiny per element but
// biases downstream argmax / greedy decode. See lstm.wgsl / softmax.wgsl.
fn expApprox(x_in: f32) -> f32 {
    let x = clamp(x_in, -80.0, 80.0);
    let y = x * 1.4426950408889634;
    let n = i32(floor(y + 0.5));
    let t = (y - f32(n)) * 0.6931471805599453;
    let e2 = bitcast<f32>(u32(n + 127) << 23u);
    return e2 * (1.0 + t * (1.0 + t * (0.5 + t * (0.16666667 + t * 0.041666668))));
}

fn sigmoidf(v: f32) -> f32 {
    var y: f32;
    if (v >= 0.0) { y = 1.0 / (1.0 + expApprox(-v)); } else { let e = expApprox(v); y = e / (1.0 + e); }
    return clamp(y, 0.0, 1.0);
}

fn tanhApprox(v: f32) -> f32 { return clamp(2.0 * sigmoidf(2.0 * v) - 1.0, -1.0, 1.0); }

@compute @workgroup_size(64)
fn relu(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = max(x[i], 0.0); }
}

@compute @workgroup_size(64)
fn gelu(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) {
        let v = x[i];
        // tanh approximation (matches the CPU kernel).
        let inner = 0.7978845608028654 * (v + 0.044715 * v * v * v);
        o[i] = 0.5 * v * (1.0 + tanhApprox(inner));
    }
}

@compute @workgroup_size(64)
fn silu(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { let v = x[i]; o[i] = v * sigmoidf(v); }
}

@compute @workgroup_size(64)
fn sigmoid(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = sigmoidf(x[i]); }
}

@compute @workgroup_size(64)
fn tanh_(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = tanhApprox(x[i]); }
}

@compute @workgroup_size(64)
fn sqrt_(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = sqrt(x[i]); }
}

@compute @workgroup_size(64)
fn log_(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = log(x[i]); }
}
