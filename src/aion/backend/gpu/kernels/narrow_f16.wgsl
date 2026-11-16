// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// An f32 buffer rounded into f16, two elements per word: what f16 matrix tiles
// read an f32 operand from (see exec/attention_mx16.zig). One invocation per
// word; the grid may wrap into y.

@group(0) @binding(0) var<storage, read>       a: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> o: array<u32>;
@group(0) @binding(2) var<uniform>             p: Params;

// Field order matches `NarrowParams` in exec/attention_mx16.zig.
struct Params { words: u32, _p0: u32, _p1: u32, _p2: u32 };

const WG: u32 = 256u; // `NARROW_WG` in exec/attention_mx16.zig

@compute @workgroup_size(WG)
fn narrow(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = g.x + g.y * nwg.x * WG;
    if (i < p.words) {
        o[i] = pack2x16float(a[i]);
    }
}
