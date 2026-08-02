// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

@group(0) @binding(0) var<storage, read>       a: array<i32>;
@group(0) @binding(1) var<storage, read>       b: array<i32>;
@group(0) @binding(2) var<storage, read_write> o: array<i32>;
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
    for (var i = g.x; i < p.n; i += step) {
        let divisor = b[i % p.cols];
        if (divisor == 0) {
            o[i] = 0;
        } else {
            o[i] = a[i] / divisor;
        }
    }
}
