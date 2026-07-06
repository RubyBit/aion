// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Matvec for MatMulNT (M == 1, the decode hot path):
//   C[n] = alpha * sum_k A[k] * B[n, k]  +  beta * C[n]
// with B either q8_0 [N, K] (ggml blocks: f16 scale + 32 i8, 34 bytes) or f32.
//
// Layout trick for q8_0: a single 34-byte block is only 2-byte aligned, so the
// kernel walks BLOCK PAIRS (68 bytes = 17 u32 words, always word-aligned when
// rows hold an even number of blocks — the backend requires K % 64 == 0):
//   w0        = d0 | qs0[0..1]<<16
//   w1..w7    = qs0[2..29]
//   w8        = qs0[30..31] | d1<<16
//   w9..w16   = qs1[0..31]          (word-aligned)
// Block 0's quant words are rebuilt with one shift-combine per word; block 1's
// are read directly.
//
// Work shape: 256-thread workgroups = 8 output rows x 32 lanes. Lanes stride
// the row's block pairs (f32: vec4 chunks), then a shared-memory tree reduce
// folds the 32 lane partials per row. Bandwidth-bound: B is read exactly once.

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       b: array<u32>;
@group(0) @binding(2) var<storage, read_write> cmat: array<f32>;
@group(0) @binding(3) var<uniform>             p: Params;

// b_wpr = u32 words per B row; k in elements; n = rows in this B/C tile.
struct Params { k: u32, n: u32, b_wpr: u32, _pad: u32, alpha: f32, beta: f32 };

const TPR: u32 = 32u; // lanes per output row
const RPW: u32 = 8u;  // output rows per workgroup

var<workgroup> partial: array<f32, 256>;

// Sign-extend 4 packed i8s to f32x4.
fn i8x4f(w: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(i32(w << 24u) >> 24u),
        f32(i32(w << 16u) >> 24u),
        f32(i32(w << 8u) >> 24u),
        f32(i32(w) >> 24u),
    );
}

// Reduce each row group's 32 lane partials; returns the row total to lane 0.
fn reduceRow(lidx: u32, lane: u32, acc: f32) -> f32 {
    partial[lidx] = acc;
    workgroupBarrier();
    var s = TPR / 2u;
    while (s > 0u) {
        if (lane < s) { partial[lidx] = partial[lidx] + partial[lidx + s]; }
        workgroupBarrier();
        s = s / 2u;
    }
    return partial[lidx - lane];
}

fn store(n: u32, total: f32) {
    let v = p.alpha * total;
    if (p.beta == 0.0) {
        cmat[n] = v;
    } else {
        cmat[n] = v + p.beta * cmat[n];
    }
}

@compute @workgroup_size(256)
fn gemv_q8(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let grp = lidx / TPR;
    let lane = lidx % TPR;
    let n = wid.x * RPW + grp;
    let in_bounds = n < p.n;

    var acc = 0.0;
    if (in_bounds) {
        let row_base = n * p.b_wpr;
        let pairs = p.k / 64u;
        for (var pi = lane; pi < pairs; pi += TPR) {
            let base = row_base + pi * 17u;
            let w0 = b[base];
            let d0 = unpack2x16float(w0).x;
            let d1 = unpack2x16float(b[base + 8u]).y;
            let a0 = pi * 16u; // vec4 index of k = pi*64

            var s0 = 0.0;
            var prev = w0;
            for (var j = 0u; j < 8u; j += 1u) {
                let cur = b[base + 1u + j];
                s0 += dot(i8x4f((prev >> 16u) | (cur << 16u)), a[a0 + j]);
                prev = cur;
            }
            var s1 = 0.0;
            for (var j = 0u; j < 8u; j += 1u) {
                s1 += dot(i8x4f(b[base + 9u + j]), a[a0 + 8u + j]);
            }
            acc += d0 * s0 + d1 * s1;
        }
    }

    let total = reduceRow(lidx, lane, acc);
    if (in_bounds && lane == 0u) { store(n, total); }
}

@compute @workgroup_size(256)
fn gemv_f32(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let grp = lidx / TPR;
    let lane = lidx % TPR;
    let n = wid.x * RPW + grp;
    let in_bounds = n < p.n;

    var acc = 0.0;
    if (in_bounds) {
        let row_base = n * p.b_wpr;
        let chunks = p.k / 4u;
        for (var wj = lane; wj < chunks; wj += TPR) {
            let o = row_base + wj * 4u;
            let bv = vec4<f32>(
                bitcast<f32>(b[o]),
                bitcast<f32>(b[o + 1u]),
                bitcast<f32>(b[o + 2u]),
                bitcast<f32>(b[o + 3u]),
            );
            acc += dot(bv, a[wj]);
        }
    }

    let total = reduceRow(lidx, lane, acc);
    if (in_bounds && lane == 0u) { store(n, total); }
}
