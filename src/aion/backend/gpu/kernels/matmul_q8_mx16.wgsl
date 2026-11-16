// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// q8_0 GEMM on 16x16x16 f16 matrix tiles with f32 accumulation (NVIDIA's tensor
// cores through Vulkan), dequantizing as it goes:
//   C[m, c_off + n] = sum_k A[m, k] * B[n, k]
// A f32 [M, K]; B q8_0 [N, K] in `lanes32x16` order (types.QuantBlockOrder): per
// group of 32 rows and per block of 32 along K, a 68-vec4 segment holding the 32
// rows' f16 scales (4 vec4), then their quants as two 16-byte chunks per row (row
// r's first chunk at vec4 4 + r, its second at 4 + 32 + r).
//
// The sibling of matmul_q8_mx.wgsl (8x8 f32 tiles). Per q8 block of K, the
// workgroup stages A (BM rows x 32, rounded to f16) and B (BN rows x one q8 block,
// dequantized to f16) into shared memory; each SIMD group then multiplies its
// SGM x SGN share of the block out of it, two tiles deep. Staged rows are padded
// to `SK` halves so a tile's 16 rows do not share banks.
//
// The finished tiles go straight to C: the last block each way slides back to
// stay whole (the host guarantees M >= BM, N >= BN), and the elements it shares
// with the block before it are computed identically by both, so writing them twice
// is harmless. That needs C = A @ B^T exactly (alpha 1, beta 0), which the host
// checks.
//
// Lines `// #name` are filled in by exec/matmul_q8_mx16.zig (the shader compiler
// does not unroll loops, and tiles cannot be indexed).

enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read>       a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       b: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             p: Params;

// Field order matches `Params` in exec/matmul_q8_mx16.zig.
struct Params { m: u32, n: u32, k: u32, c_row: u32, c_off: u32, _p0: u32, _p1: u32, _p2: u32 };

// BM, BN: rows of A and of B per block. SGM, SGN: one SIMD group's share.
// THREADS: (BM / SGM) * (BN / SGN) SIMD groups of 32. SK: a staged row's stride.
// #consts
const BK: u32 = 32u; // one q8 block

var<workgroup> sa: array<f16, BM * SK>;
var<workgroup> sb: array<f16, BN * SK>;

// Sign-extend 4 packed i8s to f32x4.
fn i8x4(w: u32) -> vec4<f32> {
    return vec4<f32>(f32(i32(w << 24u) >> 24u), f32(i32(w << 16u) >> 24u), f32(i32(w << 8u) >> 24u), f32(i32(w) >> 24u));
}

@compute @workgroup_size(THREADS)
fn gemm_q8(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let m0 = min(wid.y * BM, p.m - BM);
    let n0 = min(wid.x * BN, p.n - BN);
    let blocks = p.k / BK;
    // `coopStore`'s stride must be a named value: naga 29's WGSL frontend emits a
    // uniform load passed to it after the store ("Expression is not cached",
    // gfx-rs/wgpu#10901, open). On Vulkan the stride must also be a multiple of
    // 16 bytes, or NVIDIA writes the wrong rows silently (`fits` checks `c_row`).
    let c_row = p.c_row;
    // This SIMD group's SGM x SGN share of the block.
    let sg = t / 32u;
    let sm = (sg % (BM / SGM)) * SGM;
    let sn = (sg / (BM / SGM)) * SGN;

    // #acc_decl

    for (var kb = 0u; kb < blocks; kb += 1u) {
        // A: one vec4 of a row per step.
        for (var e = t; e < BM * BK / 4u; e += THREADS) {
            let row = e / (BK / 4u);
            let k4 = (e % (BK / 4u)) * 4u;
            let x = vec4<f16>(a[((m0 + row) * p.k + kb * BK + k4) / 4u]);
            let at = row * SK + k4;
            sa[at] = x.x;
            sa[at + 1u] = x.y;
            sa[at + 2u] = x.z;
            sa[at + 3u] = x.w;
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
            let at = (bg * 32u + br) * SK + bh * 16u;
            for (var w = 0u; w < 4u; w += 1u) {
                let v = vec4<f16>(i8x4(q[w]) * d);
                sb[at + 4u * w] = v.x;
                sb[at + 4u * w + 1u] = v.y;
                sb[at + 4u * w + 2u] = v.z;
                sb[at + 4u * w + 3u] = v.w;
            }
        }
        workgroupBarrier();
        // #tiles
        workgroupBarrier();
    }

    // #store_acc
}
