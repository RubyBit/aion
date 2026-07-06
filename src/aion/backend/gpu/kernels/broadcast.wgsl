// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Broadcast-last-dim binary ops: o[i] = a[i] OP b[i % cols], where `b` is the
// rank-1 vector broadcast across each row of `a`. Tiles are packed row-major
// (the backend verifies), so the flat index decomposes with one modulo. One
// entry point per op; f32 only; grid-stride (see elementwise.wgsl).

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform>             p: Params;

struct Params { n: u32, cols: u32 };

@compute @workgroup_size(64)
fn bcast_add(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = a[i] + b[i % p.cols]; }
}

@compute @workgroup_size(64)
fn bcast_sub(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = a[i] - b[i % p.cols]; }
}

@compute @workgroup_size(64)
fn bcast_mul(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = a[i] * b[i % p.cols]; }
}

@compute @workgroup_size(64)
fn bcast_div(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = a[i] / b[i % p.cols]; }
}
