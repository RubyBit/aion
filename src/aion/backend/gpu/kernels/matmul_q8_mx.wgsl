// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// q8_0 GEMM on 8x8 f32 matrix tiles, dequantizing as it goes:
//   C[m, c_off + n] = alpha * sum_k A[m, k] * B[n, k]  +  beta * C[m, c_off + n]
// A f32 [M, K]; B q8_0 [N, K] in `lanes32x16` order (types.QuantBlockOrder): per
// group of 32 rows and per block of 32 along K, a 68-vec4 segment holding the 32
// rows' f16 scales (4 vec4), then their quants as two 16-byte chunks per row (row
// r's first chunk at vec4 4 + r, its second at 4 + 32 + r).
//
// One workgroup per BM x BN block of C, each SIMD group an SGM x SGN share of it.
// Per block of K, the workgroup stages A (BM rows x 32, f32) and B (BN rows x one
// q8 block, dequantized to f32) into shared memory, and every SIMD group
// multiplies its tiles out of it. Measured on an M5 (64 x 64 blocks, 32 x 32
// shares) against dequantizing all of B into scratch first and multiplying that:
// never slower, and 14-26% faster at 128 rows of A, where the separate dequant
// pass over the whole weight costs as much as the product. Several SIMD groups
// sharing one staged block is what makes staging pay; a single one ran at ~1.5
// TFLOPS. A short block (few rows of A) reads each weight once for all its rows.
//
// Lines `// #name` are filled in by exec/matmul_q8_mx.zig (the shader compiler
// does not unroll loops, and tiles cannot be indexed). The last block each way
// slides back to stay whole (the host guarantees N >= BN), and writes only the
// elements no earlier block owns. With M < BM the block's spare rows repeat A's
// last row and are not written.
//
// TILE LAYOUT DEPENDS ON THE WGPU VERSION. This kernel relies on naga 29's Metal
// writer inverting the load/store layout: here `coopLoad`/`coopStore` are
// row-major and `coopLoadT` column-major. That is an upstream bug, fixed in wgpu
// 30 (gfx-rs/wgpu#9451): from then on `coopLoad` is column-major and `coopLoadT`
// row-major on every backend, as the spec says. Moving wgpu-native to >= 30 must
// swap every `coopLoad` <-> `coopLoadT` and `coopStore` <-> `coopStoreT` here
// and in exec/matmul_q8_mx.zig, checked on the M5: a wrong layout transposes
// the square tiles silently. (matmul_q8_mx16.wgsl already follows the spec.)

enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       b: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             p: Params;

// Field order matches `Params` in exec/matmul_q8_mx.zig.
struct Params { m: u32, n: u32, k: u32, c_row: u32, c_off: u32, alpha: f32, beta: f32, _p0: u32 };

// BM, BN: rows of A and of B per block (BN whole `lanes32x16` groups). SGM, SGN:
// one SIMD group's share. THREADS: (BM / SGM) * (BN / SGN) SIMD groups of 32.
// SH: floats of `sh`, the larger of its two uses below.
// #consts
const BK: u32 = 32u; // one q8 block
const B_AT: u32 = BM * BK; // where B's staging starts in `sh`

// A [BM][BK] then B [BN][BK] while multiplying; the finished C block [BM][BN] after.
var<workgroup> sh: array<f32, SH>;

// Sign-extend 4 packed i8s to f32x4.
fn i8x4(w: u32) -> vec4<f32> {
    return vec4<f32>(f32(i32(w << 24u) >> 24u), f32(i32(w << 16u) >> 24u), f32(i32(w << 8u) >> 24u), f32(i32(w) >> 24u));
}

@compute @workgroup_size(THREADS)
fn gemm_q8(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let own_m = wid.y * BM;
    let own_n = wid.x * BN;
    // With M < BM the one block's own_m is 0, whatever `p.m - BM` wraps to.
    let m0 = min(own_m, p.m - BM);
    let n0 = min(own_n, p.n - BN);
    let blocks = p.k / BK;
    // This SIMD group's SGM x SGN share of the block.
    let sg = t / 32u;
    let sm = (sg % (BM / SGM)) * SGM;
    let sn = (sg / (BM / SGM)) * SGN;

    // #acc_decl

    for (var kb = 0u; kb < blocks; kb += 1u) {
        // A: 4 floats of a row per step.
        for (var e = t; e < BM * BK / 4u; e += THREADS) {
            let row = e / (BK / 4u);
            let k4 = (e % (BK / 4u)) * 4u;
            let x = a[(min(m0 + row, p.m - 1u) * p.k + kb * BK + k4) / 4u];
            let at = row * BK + k4;
            sh[at] = x.x;
            sh[at + 1u] = x.y;
            sh[at + 2u] = x.z;
            sh[at + 3u] = x.w;
        }
        // B: one 16-quant half of a row's block per step.
        for (var e = t; e < BN * 2u; e += THREADS) {
            let bg = e / 64u; // which group of 32 rows
            let br = (e % 64u) / 2u; // row in the group
            let bh = e % 2u; // which half of the row's block
            let seg = ((n0 / 32u + bg) * blocks + kb) * 68u;
            let scales = unpack2x16float(b[seg + br / 8u][(br / 2u) % 4u]);
            let d = select(scales.x, scales.y, (br & 1u) == 1u);
            let q = b[seg + 4u + bh * 32u + br];
            let at = B_AT + (bg * 32u + br) * BK + bh * 16u;
            for (var w = 0u; w < 4u; w += 1u) {
                let v = i8x4(q[w]) * d;
                sh[at + 4u * w] = v.x;
                sh[at + 4u * w + 1u] = v.y;
                sh[at + 4u * w + 2u] = v.z;
                sh[at + 4u * w + 3u] = v.w;
            }
        }
        workgroupBarrier();
        // #tiles
        workgroupBarrier();
    }

    // #store_acc
    workgroupBarrier();
    for (var e = t; e < BM * BN; e += THREADS) {
        let mm = m0 + e / BN;
        let nn = n0 + e % BN;
        if (mm >= own_m && mm < p.m && nn >= own_n) {
            let i = mm * p.c_row + p.c_off + nn;
            let v = p.alpha * sh[e];
            if (p.beta == 0.0) {
                c[i] = v;
            } else {
                c[i] = v + p.beta * c[i];
            }
        }
    }
}
