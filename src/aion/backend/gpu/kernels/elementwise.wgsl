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

// The f16 entry points alias `array<f16>` onto the same bindings the f32 ones
// declare as `array<f32>` — legal because a resource interface is per entry
// point, not per module (see the shared-binding test in test_gpu_backend.zig).
// `shader-f16` is a required device feature, so native 2-byte addressing is
// always available and an odd element count needs no word packing.
//
// Arithmetic is done widened to f32 and rounded once on store, matching
// `elemwiseBinaryF16` in kernels/elemwise.zig ("load f16 vectors, widen to f32
// for arithmetic, narrow back"), so CPU and GPU agree bit for bit.

enable f16;

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;

@group(0) @binding(0) var<storage, read>       ah: array<f16>;
@group(0) @binding(1) var<storage, read>       bh: array<f16>;
@group(0) @binding(2) var<storage, read_write> oh: array<f16>;

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

// Gated activations: o = act(a) * b — `ElemwiseBinaryOp.gate`, one entry point per
// `UnaryOp`. GEGLU/SwiGLU/GLU/ReGLU are all this, so one dispatch replaces two for
// every gated-FFN block whichever activation the model uses. The approximations
// replicate unary.wgsl (which replicates the CPU fast_math), so a gate is
// bit-identical to the unary-then-multiply pair it stands in for.

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
fn gate_relu(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = max(a[i], 0.0) * b[i]; }
}

@compute @workgroup_size(64)
fn gate_gelu(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = geluf(a[i]) * b[i]; }
}

@compute @workgroup_size(64)
fn gate_silu(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = a[i] * sigmoidf(a[i]) * b[i]; }
}

@compute @workgroup_size(64)
fn gate_sigmoid(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = sigmoidf(a[i]) * b[i]; }
}

@compute @workgroup_size(64)
fn gate_tanh(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = tanhApprox(a[i]) * b[i]; }
}

@compute @workgroup_size(64)
fn gate_sqrt(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = sqrt(a[i]) * b[i]; }
}

@compute @workgroup_size(64)
fn gate_log(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = log(a[i]) * b[i]; }
}

// ---- f16 arithmetic (comparisons are i32-only; a gate is f32-only) --------

@compute @workgroup_size(64)
fn add_f16(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { oh[i] = f16(f32(ah[i]) + f32(bh[i])); }
}

@compute @workgroup_size(64)
fn sub_f16(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { oh[i] = f16(f32(ah[i]) - f32(bh[i])); }
}

@compute @workgroup_size(64)
fn mul_f16(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { oh[i] = f16(f32(ah[i]) * f32(bh[i])); }
}

@compute @workgroup_size(64)
fn divide_f16(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { oh[i] = f16(f32(ah[i]) / f32(bh[i])); }
}
