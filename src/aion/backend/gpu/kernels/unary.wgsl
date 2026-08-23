// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Elementwise unary activations over f32 or f16 storage buffers. One entry point
// per (op, dtype); the backend selects from `UnaryOp` plus the tile dtype.
// Dispatched once per tile. The element count comes from a uniform (binding 2)
// rather than `arrayLength`, so it counts ELEMENTS for both dtypes. Grid-stride
// loops (see elementwise.wgsl) keep dispatches under the 65535
// workgroups-per-dimension limit on large GPU-policy tiles.
//
// The f16 entry points alias `array<f16>` onto the same bindings the f32 ones
// declare as `array<f32>`. A resource interface is per entry point, not per
// module, so this is legal as long as no single entry point touches both views
// (asserted by "gpu backend: f32/f16 storage may share a binding across entry
// points" in test_gpu_backend.zig). `shader-f16` is a required device feature,
// so the 2-byte addressing is always available.
//
// Every op is defined ONCE, as an f32 -> f32 function; the f16 entry points widen
// on load and round on store. That is exactly what the CPU f16 kernels do
// (kernels/gelu.zig and friends compute `@floatCast(geluApproxF32Scalar(x))`), so
// the two backends agree to the last bit rather than to a tolerance.

enable f16;

@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@group(0) @binding(2) var<uniform>             p: Params;

@group(0) @binding(0) var<storage, read>       xh: array<f16>;
@group(0) @binding(1) var<storage, read_write> oh: array<f16>;

struct Params { n: u32 };

// Match the CPU's fast transcendental approximations (fast_math.zig) so activation
// outputs track the CPU reference — the exact-vs-approx gap is tiny per element but
// biases downstream argmax / greedy decode. See lstm.wgsl / softmax.wgsl.
fn expApprox(x_in: f32) -> f32 {
    let x_c = clamp(x_in, -80.0, 80.0);
    let y = x_c * 1.4426950408889634;
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

// ---- the ops themselves, one definition each ------------------------------

fn opRelu(v: f32) -> f32 { return max(v, 0.0); }

fn opGelu(v: f32) -> f32 {
    // tanh approximation (matches the CPU kernel).
    let inner = 0.7978845608028654 * (v + 0.044715 * v * v * v);
    return 0.5 * v * (1.0 + tanhApprox(inner));
}

fn opSilu(v: f32) -> f32 { return v * sigmoidf(v); }
fn opSigmoid(v: f32) -> f32 { return sigmoidf(v); }
fn opTanh(v: f32) -> f32 { return tanhApprox(v); }
fn opSqrt(v: f32) -> f32 { return sqrt(v); }
fn opLog(v: f32) -> f32 { return log(v); }

// ---- f32 entry points ------------------------------------------------------

@compute @workgroup_size(64)
fn relu(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = opRelu(x[i]); }
}

@compute @workgroup_size(64)
fn gelu(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = opGelu(x[i]); }
}

@compute @workgroup_size(64)
fn silu(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = opSilu(x[i]); }
}

@compute @workgroup_size(64)
fn sigmoid(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = opSigmoid(x[i]); }
}

@compute @workgroup_size(64)
fn tanh_(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = opTanh(x[i]); }
}

@compute @workgroup_size(64)
fn sqrt_(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = opSqrt(x[i]); }
}

@compute @workgroup_size(64)
fn log_(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = opLog(x[i]); }
}

// ---- f16 entry points ------------------------------------------------------

@compute @workgroup_size(64)
fn relu_f16(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { oh[i] = f16(opRelu(f32(xh[i]))); }
}

@compute @workgroup_size(64)
fn gelu_f16(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { oh[i] = f16(opGelu(f32(xh[i]))); }
}

@compute @workgroup_size(64)
fn silu_f16(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { oh[i] = f16(opSilu(f32(xh[i]))); }
}

@compute @workgroup_size(64)
fn sigmoid_f16(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { oh[i] = f16(opSigmoid(f32(xh[i]))); }
}

@compute @workgroup_size(64)
fn tanh__f16(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { oh[i] = f16(opTanh(f32(xh[i]))); }
}

@compute @workgroup_size(64)
fn sqrt__f16(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { oh[i] = f16(opSqrt(f32(xh[i]))); }
}

@compute @workgroup_size(64)
fn log__f16(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { oh[i] = f16(opLog(f32(xh[i]))); }
}
