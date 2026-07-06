// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Elementwise binary ops over i32 storage buffers: integer arithmetic plus the
// comparisons (which produce i32 {0,1} — the If/Loop predicate dtype). Same
// dispatch contract as elementwise.wgsl: one entry point per op, one dispatch
// per tile, element count from the uniform, grid-stride loop.
//
// Semantics mirror the CPU's elemwiseBinaryI32: div-by-zero yields 0 and
// division truncates toward zero (WGSL `/` on i32 already truncates).

@group(0) @binding(0) var<storage, read>       a: array<i32>;
@group(0) @binding(1) var<storage, read>       b: array<i32>;
@group(0) @binding(2) var<storage, read_write> o: array<i32>;
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
    // WGSL defines x/0 == x (no trap), so the raw quotient is safe to compute.
    for (var i = g.x; i < p.n; i += step) {
        o[i] = select(a[i] / b[i], 0, b[i] == 0);
    }
}

@compute @workgroup_size(64)
fn eq(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = i32(a[i] == b[i]); }
}

@compute @workgroup_size(64)
fn ne(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = i32(a[i] != b[i]); }
}

@compute @workgroup_size(64)
fn lt(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = i32(a[i] < b[i]); }
}

@compute @workgroup_size(64)
fn gt(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = i32(a[i] > b[i]); }
}

@compute @workgroup_size(64)
fn le(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = i32(a[i] <= b[i]); }
}

@compute @workgroup_size(64)
fn ge(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.n; i += step) { o[i] = i32(a[i] >= b[i]); }
}
