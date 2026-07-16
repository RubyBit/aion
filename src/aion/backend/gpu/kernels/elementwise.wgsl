// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Elementwise binary ops over f32 storage buffers. One entry point per op; the
// backend selects the entry point from `ElemwiseBinaryOp`. f32 only; the backend
// dispatches once per tile (each tile is its own pair of input buffers + output
// buffer). The element count comes from a uniform (binding 3) rather than
// `arrayLength`, so a tile's logical length is explicit and not tied to the
// device buffer's allocated size.
//
// Grid-stride: WebGPU caps workgroups per dimension at 65535, and GPU-policy
// tiles reach 16M+ elements (262144 groups of 64). Each thread therefore loops
// with stride = total threads; the backend dispatches at most MAX_GROUPS_1D.

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform>             p: Params;

struct Params { n: u32 };

@compute @workgroup_size(64)
fn add(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = a[i] + b[i]; }
}

@compute @workgroup_size(64)
fn sub(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = a[i] - b[i]; }
}

@compute @workgroup_size(64)
fn mul(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = a[i] * b[i]; }
}

@compute @workgroup_size(64)
fn divide(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = a[i] / b[i]; }
}

// Fused GeGLU multiply: o = gelu(a) * b — the `GeluMulTiled` step, compiled by
// the `fuse_gpu_decode` graph pass from a gelu whose ONLY consumer is the
// multiply. One dispatch instead of two for every gated-FFN block. The
// approximations replicate unary.wgsl (which replicates the CPU fast_math), so
// the fused result is bit-identical to the two-dispatch sequence.

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

fn geluf(v: f32) -> f32 {
    let inner = 0.7978845608028654 * (v + 0.044715 * v * v * v);
    return 0.5 * v * (1.0 + tanhApprox(inner));
}

@compute @workgroup_size(64)
fn mul_gelu_a(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = geluf(a[i]) * b[i]; }
}
