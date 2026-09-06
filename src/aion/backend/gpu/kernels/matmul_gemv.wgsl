// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Fused matvec for the PLAIN MatMul (K-major B), M == 1 — the Gemma decode hot
// path. Computes one B tile's contribution:
//   C[n] = alpha * sum_k A[k] * B[k, n]  +  beta * C[n]
// with B in q8_0 quantized ALONG K (ggml MatMul-B convention): blocks tile a
// [K/32, N] grid, row-major, so consecutive blocks run along N. This is the
// layout `dequant.wgsl :: q8_kmajor_to_f32` materializes to scratch — here we
// fold the dequant INTO the dot product so B is read exactly once and no f32
// scratch is written/re-read (the previous path did both, ~8x the traffic).
//
// Work shape: a 1024-thread workgroup is COLS (=32) column-pairs × LANES (=32)
// K-lanes. `pair` is the FAST index (lidx % COLS) so a warp (32 consecutive
// lanes) reads 32 ADJACENT column-pairs at the SAME K-block-row — 32 contiguous
// 17-word blocks → fully coalesced. `klane` (lidx / COLS) K-splits the block-row
// reduction across 32 threads per column, a 32x thread multiplier over one
// thread per column, which is what gives the launch enough warps for occupancy
// (a plain thread-per-column GEMV bottoms out at N/512 workgroups).
//
// `WG` MUST equal the `@workgroup_size` below: `klane` is derived from it and the
// reduction strides by `LANES * COLS`, so a mismatch silently sums uninitialized
// shared memory (and still produces plausible-looking numbers).
//
// LANES=8/WG=256 was tried on 2026-08-19 — more block-rows per thread, a shallower
// reduce, 4 blocks/SM instead of 2. At matched temperature it measured the same:
// 13.91 ms/token vs 13.88 for this config. NOT a win either way, so this stays.
// (First read looked like a 1.2 ms regression; that was the GPU being ~1 ms slower
// hot than cold, which is larger than the effect. Compare only same-temperature
// runs, interleaved.)
//
// A single 34-byte q8_0 block is only 2-byte aligned, so a thread reads its two
// adjacent columns' blocks as a word-aligned 68-byte PAIR (17 u32) — the same
// trick as matmul_nt_gemv.wgsl. A is read straight from global (it is one small
// activation vector, resident in L2, and each warp broadcasts one A load across
// its 32 column-pairs).

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       b: array<u32>;
@group(0) @binding(2) var<storage, read_write> cmat: array<f32>;
@group(0) @binding(3) var<uniform>             p: Params;

// k/n: this B tile's K rows / N cols. K % 32 == 0 and N % 2 == 0 (checked host-side).
struct Params { k: u32, n: u32, _p0: u32, _p1: u32, alpha: f32, beta: f32 };

const COLS: u32 = 32u;   // column-pairs per workgroup (== warp width → coalesced)
const LANES: u32 = 32u;  // K-lanes reducing each column's block-rows
const WG: u32 = COLS * LANES; // 1024

var<workgroup> partA: array<f32, WG>;
var<workgroup> partB: array<f32, WG>;

// Sign-extend 4 packed i8s to f32x4.
fn i8x4f(w: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(i32(w << 24u) >> 24u),
        f32(i32(w << 16u) >> 24u),
        f32(i32(w << 8u) >> 24u),
        f32(i32(w) >> 24u),
    );
}

@compute @workgroup_size(1024)
fn gemv_q8_kmajor(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let pair_local = lidx % COLS;
    let klane = lidx / COLS;
    let pair = wid.x * COLS + pair_local; // this thread's column pair
    let n0 = pair * 2u;

    let pairs_total = p.n / 2u;
    let n_bk = p.k / 32u;   // total K block-rows
    let half_n = p.n / 2u;  // block-pairs per block-row
    let in_bounds = pair < pairs_total;

    var accA = 0.0;
    var accB = 0.0;
    if (in_bounds) {
        // Each K-lane strides the block-rows.
        var bk = klane;
        while (bk < n_bk) {
            let base = (bk * half_n + pair) * 17u;
            let w0 = b[base];
            let d0 = unpack2x16float(w0).x;         // scale for column n0
            let d1 = unpack2x16float(b[base + 8u]).y; // scale for column n0+1
            let av = bk * 8u;                        // vec4 index of k = bk*32

            var sA = 0.0;
            var prev = w0;
            // The shift-combine below (and its serial prev/cur carry) is the whole cost
            // a two-plane q8 layout — separate scale and quant planes, enabling
            // vec4 loads — would remove. Measured 2026-08-19 by replacing this loop
            // with plain aligned loads (numerically wrong, identical traffic): the
            // WHOLE token gained 0.29 ms of 13.09. So the 34-byte block's odd stride is
            // NOT what limits this kernel, and splitting the planes is not worth the
            // derived-weight machinery or the weight duplication it would cost.
            for (var j = 0u; j < 8u; j += 1u) {
                let cur = b[base + 1u + j];
                sA += dot(i8x4f((prev >> 16u) | (cur << 16u)), a[av + j]);
                prev = cur;
            }
            var sB = 0.0;
            for (var j = 0u; j < 8u; j += 1u) {
                sB += dot(i8x4f(b[base + 9u + j]), a[av + j]);
            }
            accA += d0 * sA;
            accB += d1 * sB;
            bk += LANES;
        }
    }

    // Reduce the LANES K-lane partials for each column pair (stride COLS).
    partA[lidx] = accA;
    partB[lidx] = accB;
    workgroupBarrier();
    var s = LANES / 2u;
    while (s > 0u) {
        if (klane < s) {
            partA[lidx] += partA[lidx + s * COLS];
            partB[lidx] += partB[lidx + s * COLS];
        }
        workgroupBarrier();
        s = s / 2u;
    }

    if (in_bounds && klane == 0u) {
        let vA = p.alpha * partA[pair_local];
        let vB = p.alpha * partB[pair_local];
        if (p.beta == 0.0) {
            cmat[n0] = vA;
            cmat[n0 + 1u] = vB;
        } else {
            cmat[n0] = vA + p.beta * cmat[n0];
            cmat[n0 + 1u] = vB + p.beta * cmat[n0 + 1u];
        }
    }
}

// Narrow variant: same algorithm, a QUARTER of the column-pairs per workgroup.
//
// The group count is `pairs / COLS`, so COLS alone decides how much of the GPU a
// launch can occupy — LANES only reshapes work INSIDE a group. The model-dim
// projections (N=1536 -> 24 groups of 1024) left most of a 58-SM GPU idle; at
// COLS_N=8 the same work is 96 groups of 256. Total threads are unchanged, and each
// B column is still read exactly once, so this buys occupancy at no extra traffic.
//
// Coalescing is weaker but not lost: a warp now covers 8 adjacent column-pairs at
// each of 4 K-block-rows, so four 544-byte contiguous runs instead of one 2176-byte
// run. Still far above a cache line.
//
// (The header's earlier LANES=8/WG=256 experiment kept COLS=32, so it changed the
// shape inside a group while leaving the group COUNT at 24 — a different axis.)
const COLS_N: u32 = 8u;
const LANES_N: u32 = 32u;
const WG_N: u32 = COLS_N * LANES_N; // 256

var<workgroup> partA_N: array<f32, 256>;
var<workgroup> partB_N: array<f32, 256>;

@compute @workgroup_size(256)
fn gemv_q8_kmajor_narrow(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let pair_local = lidx % COLS_N;
    let klane = lidx / COLS_N;
    let pair = wid.x * COLS_N + pair_local; // this thread's column pair
    let n0 = pair * 2u;

    let pairs_total = p.n / 2u;
    let n_bk = p.k / 32u;   // total K block-rows
    let half_n = p.n / 2u;  // block-pairs per block-row
    let in_bounds = pair < pairs_total;

    var accA = 0.0;
    var accB = 0.0;
    if (in_bounds) {
        // Each K-lane strides the block-rows.
        var bk = klane;
        while (bk < n_bk) {
            let base = (bk * half_n + pair) * 17u;
            let w0 = b[base];
            let d0 = unpack2x16float(w0).x;         // scale for column n0
            let d1 = unpack2x16float(b[base + 8u]).y; // scale for column n0+1
            let av = bk * 8u;                        // vec4 index of k = bk*32

            var sA = 0.0;
            var prev = w0;
            // The shift-combine below (and its serial prev/cur carry) is the whole cost
            // a two-plane q8 layout — separate scale and quant planes, enabling
            // vec4 loads — would remove. Measured 2026-08-19 by replacing this loop
            // with plain aligned loads (numerically wrong, identical traffic): the
            // WHOLE token gained 0.29 ms of 13.09. So the 34-byte block's odd stride is
            // NOT what limits this kernel, and splitting the planes is not worth the
            // derived-weight machinery or the weight duplication it would cost.
            for (var j = 0u; j < 8u; j += 1u) {
                let cur = b[base + 1u + j];
                sA += dot(i8x4f((prev >> 16u) | (cur << 16u)), a[av + j]);
                prev = cur;
            }
            var sB = 0.0;
            for (var j = 0u; j < 8u; j += 1u) {
                sB += dot(i8x4f(b[base + 9u + j]), a[av + j]);
            }
            accA += d0 * sA;
            accB += d1 * sB;
            bk += LANES_N;
        }
    }

    // Reduce the LANES_N K-lane partials for each column pair (stride COLS_N).
    partA_N[lidx] = accA;
    partB_N[lidx] = accB;
    workgroupBarrier();
    var s = LANES_N / 2u;
    while (s > 0u) {
        if (klane < s) {
            partA_N[lidx] += partA_N[lidx + s * COLS_N];
            partB_N[lidx] += partB_N[lidx + s * COLS_N];
        }
        workgroupBarrier();
        s = s / 2u;
    }

    if (in_bounds && klane == 0u) {
        let vA = p.alpha * partA_N[pair_local];
        let vB = p.alpha * partB_N[pair_local];
        if (p.beta == 0.0) {
            cmat[n0] = vA;
            cmat[n0 + 1u] = vB;
        } else {
            cmat[n0] = vA + p.beta * cmat[n0];
            cmat[n0 + 1u] = vB + p.beta * cmat[n0 + 1u];
        }
    }
}

// General per-COLUMN variant for ANY N (used when N is odd, so the column-pair
// layout above doesn't apply). One thread owns one output column and walks its K
// blocks directly. A block's 34 bytes sit at byte offset (bk*N + n)*34, which is
// 4-byte aligned when that block index is even and 2-byte (mid-word) aligned when
// odd — the two cases mirror the "block A"/"block B" halves of the paired kernel.
// Slower (no K-split, one thread per column) but N-agnostic; fine for the small
// odd-N projections (e.g. an RNNT joint's vocab+blank output).
@compute @workgroup_size(256)
fn gemv_q8_kmajor_odd(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let n = wid.x * 256u + lid;
    if (n >= p.n) { return; }
    let n_bk = p.k / 32u;

    var acc = 0.0;
    for (var bk = 0u; bk < n_bk; bk += 1u) {
        let boff = (bk * p.n + n) * 34u; // byte offset (always even)
        let w = boff / 4u;               // word index
        let mid = (boff & 3u) != 0u;     // true => scale/quants start mid-word (+2)
        let word0 = b[w];
        let av = bk * 8u;                // vec4 index of k = bk*32

        var s = 0.0;
        var scale = 0.0;
        if (mid) {
            // scale in high half of word0; the 32 quants are word-aligned at w+1.
            scale = unpack2x16float(word0).y;
            for (var j = 0u; j < 8u; j += 1u) {
                s += dot(i8x4f(b[w + 1u + j]), a[av + j]);
            }
        } else {
            // scale in low half of word0; quants start at byte 2 (shift-combine).
            scale = unpack2x16float(word0).x;
            var prev = word0;
            for (var j = 0u; j < 8u; j += 1u) {
                let cur = b[w + 1u + j];
                s += dot(i8x4f((prev >> 16u) | (cur << 16u)), a[av + j]);
                prev = cur;
            }
        }
        acc += scale * s;
    }

    let v = p.alpha * acc;
    if (p.beta == 0.0) {
        cmat[n] = v;
    } else {
        cmat[n] = v + p.beta * cmat[n];
    }
}
