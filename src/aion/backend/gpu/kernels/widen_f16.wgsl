// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Two f16 buffers, read as words of two halves, into one f32 buffer: `a`'s
// elements, then `b`'s. Matrix attention reads an f16 K/V cache this way (see
// exec/attention_mx.zig). One invocation per word; the grid may wrap into y.

@group(0) @binding(0) var<storage, read>       a: array<u32>;
@group(0) @binding(1) var<storage, read>       b: array<u32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform>             p: Params;

// Field order matches `WidenParams` in exec/attention_mx.zig.
struct Params { a_words: u32, b_words: u32, _p0: u32, _p1: u32 };

const WG: u32 = 256u; // `WIDEN_WG` in exec/attention_mx.zig

@compute @workgroup_size(WG)
fn widen(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = g.x + g.y * nwg.x * WG;
    var x: vec2<f32>;
    if (i < p.a_words) {
        x = unpack2x16float(a[i]);
    } else if (i < p.a_words + p.b_words) {
        x = unpack2x16float(b[i - p.a_words]);
    } else {
        return;
    }
    o[2u * i] = x.x;
    o[2u * i + 1u] = x.y;
}
