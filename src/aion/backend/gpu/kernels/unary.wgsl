// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Elementwise unary activations over f32 storage buffers. One entry point per
// op; the backend selects from `UnaryOp`. f32 only; dispatched once per tile.
// The element count comes from a uniform (binding 2) rather than `arrayLength`.

@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@group(0) @binding(2) var<uniform>             p: Params;

struct Params { n: u32 };

fn sigmoidf(v: f32) -> f32 { return 1.0 / (1.0 + exp(-v)); }

@compute @workgroup_size(64)
fn relu(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i < p.n) { o[i] = max(x[i], 0.0); }
}

@compute @workgroup_size(64)
fn gelu(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i < p.n) {
        let v = x[i];
        // tanh approximation (matches the CPU kernel).
        let inner = 0.7978845608028654 * (v + 0.044715 * v * v * v);
        o[i] = 0.5 * v * (1.0 + tanh(inner));
    }
}

@compute @workgroup_size(64)
fn silu(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i < p.n) { let v = x[i]; o[i] = v * sigmoidf(v); }
}

@compute @workgroup_size(64)
fn sigmoid(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i < p.n) { o[i] = sigmoidf(x[i]); }
}

@compute @workgroup_size(64)
fn tanh_(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i < p.n) { o[i] = tanh(x[i]); }
}

@compute @workgroup_size(64)
fn sqrt_(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i < p.n) { o[i] = sqrt(x[i]); }
}

@compute @workgroup_size(64)
fn log_(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i < p.n) { o[i] = log(x[i]); }
}
