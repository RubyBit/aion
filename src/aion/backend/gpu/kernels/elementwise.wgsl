// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Elementwise binary ops over f32 storage buffers. One entry point per op; the
// backend selects the entry point from `ElemwiseBinaryOp`. f32 only; the backend
// dispatches once per tile (each tile is its own pair of input buffers + output
// buffer). The element count comes from a uniform (binding 3) rather than
// `arrayLength`, so a tile's logical length is explicit and not tied to the
// device buffer's allocated size.

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform>             p: Params;

struct Params { n: u32 };

@compute @workgroup_size(64)
fn add(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i < p.n) { o[i] = a[i] + b[i]; }
}

@compute @workgroup_size(64)
fn sub(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i < p.n) { o[i] = a[i] - b[i]; }
}

@compute @workgroup_size(64)
fn mul(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i < p.n) { o[i] = a[i] * b[i]; }
}

@compute @workgroup_size(64)
fn divide(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i < p.n) { o[i] = a[i] / b[i]; }
}
