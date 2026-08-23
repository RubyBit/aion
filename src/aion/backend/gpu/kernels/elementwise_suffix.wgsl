// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Broadcast-last-dim binary ops: o[i] = a[i] OP b[i % cols], where `b` is the
// rank-1 vector broadcast across each row of `a`. Tiles are packed row-major
// (the backend verifies), so the flat index decomposes with one modulo. One
// entry point per op; f32 only; grid-stride (see elementwise.wgsl).

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

struct Params { n: u32, cols: u32 };

@compute @workgroup_size(64)
fn suffix_add(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = a[i] + b[i % p.cols]; }
}

@compute @workgroup_size(64)
fn suffix_sub(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = a[i] - b[i % p.cols]; }
}

@compute @workgroup_size(64)
fn suffix_mul(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = a[i] * b[i % p.cols]; }
}

@compute @workgroup_size(64)
fn suffix_div(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = a[i] / b[i % p.cols]; }
}

// ---- f16 arithmetic --------------------------------------------------------

@compute @workgroup_size(64)
fn suffix_add_f16(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { oh[i] = f16(f32(ah[i]) + f32(bh[i % p.cols])); }
}

@compute @workgroup_size(64)
fn suffix_sub_f16(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { oh[i] = f16(f32(ah[i]) - f32(bh[i % p.cols])); }
}

@compute @workgroup_size(64)
fn suffix_mul_f16(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { oh[i] = f16(f32(ah[i]) * f32(bh[i % p.cols])); }
}

@compute @workgroup_size(64)
fn suffix_div_f16(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { oh[i] = f16(f32(ah[i]) / f32(bh[i % p.cols])); }
}
