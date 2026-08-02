// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform>             p: Params;

struct Params {
    n: u32,
    rank: u32,
    _pad0: u32,
    _pad1: u32,
    shape0: vec4<u32>,
    shape1: vec4<u32>,
    a_stride0: vec4<u32>,
    a_stride1: vec4<u32>,
    b_stride0: vec4<u32>,
    b_stride1: vec4<u32>,
};

fn lane(lo: vec4<u32>, hi: vec4<u32>, i: u32) -> u32 {
    if (i < 4u) { return lo[i]; }
    return hi[i - 4u];
}

fn indices(linear: u32) -> vec2<u32> {
    var rem = linear;
    var ai = 0u;
    var bi = 0u;
    var rev = p.rank;
    loop {
        if (rev == 0u) { break; }
        rev -= 1u;
        let dim = lane(p.shape0, p.shape1, rev);
        let coord = rem % dim;
        rem /= dim;
        ai += coord * lane(p.a_stride0, p.a_stride1, rev);
        bi += coord * lane(p.b_stride0, p.b_stride1, rev);
    }
    return vec2<u32>(ai, bi);
}

@compute @workgroup_size(64)
fn add(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= p.n) { return; }
    let ix = indices(g.x); o[g.x] = a[ix.x] + b[ix.y];
}
@compute @workgroup_size(64)
fn sub(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= p.n) { return; }
    let ix = indices(g.x); o[g.x] = a[ix.x] - b[ix.y];
}
@compute @workgroup_size(64)
fn mul(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= p.n) { return; }
    let ix = indices(g.x); o[g.x] = a[ix.x] * b[ix.y];
}
@compute @workgroup_size(64)
fn divide(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= p.n) { return; }
    let ix = indices(g.x); o[g.x] = a[ix.x] / b[ix.y];
}
