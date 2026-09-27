// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Matvec for MatMulNT (M == 1, the decode hot path):
//   C[n] = alpha * sum_k A[k] * B[n, k]  +  beta * C[n]
// with B either q8_0 [N, K] (blocks of an f16 scale + 32 i8, 34 bytes) or f32.
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

// b_wpr = u32 words per B row; k in elements; n = rows in this B chunk, whose
// outputs start at cmat[c_off] (a B past the binding limit is chunked along N).
struct Params { k: u32, n: u32, b_wpr: u32, c_off: u32, alpha: f32, beta: f32 };

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
    let i = p.c_off + n;
    if (p.beta == 0.0) {
        cmat[i] = v;
    } else {
        cmat[i] = v + p.beta * cmat[i];
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

// B in `lanes32x16` order (types.QuantBlockOrder): for each group of 32 rows and
// each block, the group's 32 f16 scales (16 words), then 2 chunks holding the 32
// rows' 16 quant bytes side by side. One thread owns one row and fetches its 32
// quants with two 16-byte loads; the 32 threads of a chunk read 512 consecutive
// bytes. Segments are 68 vec4s, so every load is aligned.
//
// K is split across `L_SLICES` slices of the workgroup; the slices' partials are
// summed in shared memory. More slices for narrow N keep the GPU occupied: its
// core count is not visible through WebGPU, so `gemv_q8_lanes32x16_wide` (32
// slices) is picked host-side when N alone yields few workgroups.
@group(0) @binding(1) var<storage, read> b4: array<vec4<u32>>;

const L_W: u32 = 32u;
const L_SEG4: u32 = 68u; // vec4s per (group, block) segment: 32 * 34 / 16

var<workgroup> lpart: array<f32, 1024>;

fn lanesRow(wid: u32, lane: u32, slice: u32, slices: u32) -> f32 {
    let blocks = p.k / 32u;
    let gbase = wid * blocks * L_SEG4;
    var acc = 0.0;
    for (var kb = slice; kb < blocks; kb += slices) {
        let seg = gbase + kb * L_SEG4;
        let sw = unpack2x16float(b4[seg + lane / 8u][(lane / 2u) % 4u]);
        let d = select(sw.x, sw.y, (lane & 1u) == 1u);
        let q0 = b4[seg + 4u + lane];
        let q1 = b4[seg + 4u + L_W + lane];
        let av = kb * 8u;
        let s = dot(i8x4f(q0.x), a[av]) + dot(i8x4f(q0.y), a[av + 1u]) + dot(i8x4f(q0.z), a[av + 2u]) + dot(i8x4f(q0.w), a[av + 3u])
              + dot(i8x4f(q1.x), a[av + 4u]) + dot(i8x4f(q1.y), a[av + 5u]) + dot(i8x4f(q1.z), a[av + 6u]) + dot(i8x4f(q1.w), a[av + 7u]);
        acc += d * s;
    }
    return acc;
}

fn lanesReduce(lidx: u32, lane: u32, slice: u32, slices: u32, acc: f32) -> f32 {
    lpart[lidx] = acc;
    workgroupBarrier();
    var s = slices / 2u;
    while (s > 0u) {
        if (slice < s) { lpart[lidx] = lpart[lidx] + lpart[lidx + s * L_W]; }
        workgroupBarrier();
        s = s / 2u;
    }
    return lpart[lane];
}

@compute @workgroup_size(256)
fn gemv_q8_lanes32x16(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let lane = lidx % L_W;
    let slice = lidx / L_W;
    let total = lanesReduce(lidx, lane, slice, 8u, lanesRow(wid.x, lane, slice, 8u));
    let n = wid.x * L_W + lane;
    if (slice == 0u && n < p.n) { store(n, total); }
}

@compute @workgroup_size(1024)
fn gemv_q8_lanes32x16_wide(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let lane = lidx % L_W;
    let slice = lidx / L_W;
    let total = lanesReduce(lidx, lane, slice, 32u, lanesRow(wid.x, lane, slice, 32u));
    let n = wid.x * L_W + lane;
    if (slice == 0u && n < p.n) { store(n, total); }
}
