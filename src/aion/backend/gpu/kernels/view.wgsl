// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Strided gather for view materializations: each dst element pulls from
//   src[p.base + sum_d coord_d * p.sstride_d]
// where coord decodes the flat dst index against the packed dst shape. One
// parameterization covers Transpose2DScalar (base 0, strides [1, m]) and
// SliceNDScalar (base = flat offset of `starts`, strides = src row-major).
//
// Elements are moved as u32 words — dtype-agnostic for any 4-byte scalar
// (f32 / i32). Rank <= 8; shape/stride arrays ride in the uniform as two
// vec4<u32> (uniform arrays need 16-byte stride).

@group(0) @binding(0) var<storage, read>       x: array<u32>;
@group(0) @binding(1) var<storage, read_write> o: array<u32>;
@group(0) @binding(2) var<uniform>             p: Params;

struct Params {
    total: u32,
    rank: u32,
    base: u32,
    _pad: u32,
    dshape: array<vec4<u32>, 2>, // dst shape, dims [0..rank)
    sstride: array<vec4<u32>, 2>, // src strides in elements
};

const WG: u32 = 64u;

// Contiguous-run scatter for ConcatScalar: src is fully packed; run r of
// `run_len` elements lands at o[base + r * run_stride]. Replaces the
// one-encoder-copy-per-outer-index path, whose per-copy overhead (~1 µs)
// dominates once `outer` grows past a few dozen. Field reuse from Params:
// dshape[0].x = run_len, sstride[0].x = run_stride.
@compute @workgroup_size(64)
fn strided_copy_u32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let stride = nwg.x * WG;
    let run_len = p.dshape[0].x;
    let run_stride = p.sstride[0].x;
    for (var idx = gid.x; idx < p.total; idx += stride) {
        let r = idx / run_len;
        let off = idx % run_len;
        o[p.base + r * run_stride + off] = x[idx];
    }
}

@compute @workgroup_size(64)
fn gather_nd_u32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let stride = nwg.x * WG;
    for (var idx = gid.x; idx < p.total; idx += stride) {
        var rem = idx;
        var src = p.base;
        var d = p.rank;
        while (d > 0u) {
            d -= 1u;
            let dim = p.dshape[d / 4u][d % 4u];
            let coord = rem % dim;
            rem = rem / dim;
            src += coord * p.sstride[d / 4u][d % 4u];
        }
        o[idx] = x[src];
    }
}
