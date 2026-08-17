// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// View materializations for f16 at ELEMENT granularity — the twins of view.wgsl's
// u32-word kernels, taken when a word view of the innermost axis does not exist
// (odd extents, odd slice starts, odd tile boundaries). `shader-f16` gives 2-byte
// storage addressing, so an invocation owns one element and neighbours never share
// a destination word: no lane masking, no atomics, no tail handling.

enable f16;

@group(0) @binding(0) var<storage, read>       x: array<f16>;
@group(0) @binding(1) var<storage, read_write> o: array<f16>;
@group(0) @binding(2) var<uniform>             p: Params;

/// Same layout as view.wgsl's Params, but every field counts ELEMENTS.
struct Params {
    total: u32,
    rank: u32,
    base: u32,
    src_base: u32,
    dshape: array<vec4<u32>, 2>,
    sstride: array<vec4<u32>, 2>,
};

const WG: u32 = 64u;

/// Flat index into the strided side: `base + sum_d coord_d * sstride_d`, decoding
/// `flat` against the packed shape `dshape`.
fn stridedIndex(flat: u32) -> u32 {
    var rem = flat;
    var idx = p.base;
    var d = p.rank;
    while (d > 0u) {
        d -= 1u;
        let dim = p.dshape[d / 4u][d % 4u];
        idx += (rem % dim) * p.sstride[d / 4u][d % 4u];
        rem = rem / dim;
    }
    return idx;
}

// Contiguous-run copy for ConcatScalar: run r of `run_len` elements lands at
// o[base + r * run_stride]. Adjacent concat inputs may end and begin inside one
// u32 word; addressing elements means neither dispatch can disturb the other.
@compute @workgroup_size(64)
fn strided_copy_f16(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let stride = nwg.x * WG;
    let run_len = p.dshape[0].x;
    let run_stride = p.sstride[0].x;
    for (var idx = gid.x; idx < p.total; idx += stride) {
        o[p.base + (idx / run_len) * run_stride + (idx % run_len)] = x[p.src_base + idx];
    }
}

// Strided gather: each packed dst element pulls from its strided src offset.
// Covers SliceNDScalar, Transpose2DScalar (rank 2, strides swapped), and the
// packed -> tiled half of reshape/retile.
@compute @workgroup_size(64)
fn gather_nd_f16(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let stride = nwg.x * WG;
    for (var idx = gid.x; idx < p.total; idx += stride) {
        o[idx] = x[stridedIndex(idx)];
    }
}

// Strided scatter — the write-side twin. A tile's packed elements land at their
// flat offsets in the packed tensor; neighbouring tiles may own opposite halves of
// one word, which is exactly what element addressing makes a non-issue.
@compute @workgroup_size(64)
fn scatter_nd_f16(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let stride = nwg.x * WG;
    for (var idx = gid.x; idx < p.total; idx += stride) {
        o[stridedIndex(idx)] = x[idx];
    }
}
