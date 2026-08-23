// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Layout/dtype conversion passes that feed the f32 GEMM: they materialize a
// B tile into a transient f32 scratch buffer so the autotuned f32 kernel can
// consume weights stored in other dtypes/orientations (the M>1 MatMulNT path).
// One extra bandwidth pass over B — cheap next to the GEMM it unblocks; the
// frame's pass ordering serializes scratch reuse across tiles.
//
//   q8_nt_to_f32t : B q8_0 [N, K] (NT)  -> scratch f32 [K, N]  (dequant + transpose)
//   f32_nt_t      : B f32  [N, K] (NT)  -> scratch f32 [K, N]  (transpose)
//   f16_to_f32    : same-layout f16 -> f32 widen (for f16 GEMM / cast)
//
// q8_0 rows are walked in word-aligned BLOCK PAIRS (see matmul_nt_gemv.wgsl for
// the 17-word layout); the backend requires K % 64 == 0 for the q8_0 path.
// Grid-stride dispatch (see elementwise.wgsl).

enable f16;

@group(0) @binding(0) var<storage, read>       src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

// Native f16 views of the same two bindings. `shader-f16` is a required device
// feature, so a cast addresses ELEMENTS and no longer has to pack pairs into u32
// words — which is what forced the old even-element-count restriction on the
// tile. An odd tile now casts as-is.
@group(0) @binding(0) var<storage, read>       src_h: array<f16>;
@group(0) @binding(1) var<storage, read_write> dst_h: array<f16>;
@group(0) @binding(2) var<uniform>             p: Params;

// n/k: B tile rows/cols. src_wpr = u32 words per source row. dst_row = f32
// elements per dst row ( == n for the transposed layouts). count = total work
// items (pairs for q8, elements for the others).
struct Params { n: u32, k: u32, src_wpr: u32, dst_row: u32, count: u32 };

fn i8x4f(w: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(i32(w << 24u) >> 24u),
        f32(i32(w << 16u) >> 24u),
        f32(i32(w << 8u) >> 24u),
        f32(i32(w) >> 24u),
    );
}

// One work item = one block pair of row n: dequantize 64 elements k0..k0+63
// into the transposed scratch (dst[k * dst_row + n]).
@compute @workgroup_size(64)
fn q8_nt_to_f32t(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    let pairs = p.k / 64u;
    for (var i = g.x; i < p.count; i += step) {
        let n = i / pairs;
        let pi = i % pairs;
        let base = n * p.src_wpr + pi * 17u;
        let k0 = pi * 64u;

        let w0 = src[base];
        let d0 = unpack2x16float(w0).x;
        let d1 = unpack2x16float(src[base + 8u]).y;

        var prev = w0;
        for (var j = 0u; j < 8u; j += 1u) {
            let cur = src[base + 1u + j];
            let q = i8x4f((prev >> 16u) | (cur << 16u)) * d0;
            let kb = k0 + j * 4u;
            dst[kb * p.dst_row + n] = q.x;
            dst[(kb + 1u) * p.dst_row + n] = q.y;
            dst[(kb + 2u) * p.dst_row + n] = q.z;
            dst[(kb + 3u) * p.dst_row + n] = q.w;
            prev = cur;
        }
        for (var j = 0u; j < 8u; j += 1u) {
            let q = i8x4f(src[base + 9u + j]) * d1;
            let kb = k0 + 32u + j * 4u;
            dst[kb * p.dst_row + n] = q.x;
            dst[(kb + 1u) * p.dst_row + n] = q.y;
            dst[(kb + 2u) * p.dst_row + n] = q.z;
            dst[(kb + 3u) * p.dst_row + n] = q.w;
        }
    }
}

// One work item = one element: dst[k * dst_row + n] = B[n, k].
@compute @workgroup_size(64)
fn f32_nt_t(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.count; i += step) {
        let n = i / p.k;
        let k = i % p.k;
        dst[k * p.dst_row + n] = bitcast<f32>(src[n * p.src_wpr + k]);
    }
}

// Dequantize ONE contiguous q8_0 row (a gather-row fetch): the row's block
// pairs start at word offset `src_wpr` and land contiguously at f32 element
// offset `dst_row`. `count` = block pairs (row elems / 64). Field reuse of
// Params is deliberate — one uniform layout for the whole module.
@compute @workgroup_size(64)
fn q8_row_to_f32(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.count; i += step) {
        let base = p.src_wpr + i * 17u;
        let e0 = p.dst_row + i * 64u;

        let w0 = src[base];
        let d0 = unpack2x16float(w0).x;
        let d1 = unpack2x16float(src[base + 8u]).y;

        var prev = w0;
        for (var j = 0u; j < 8u; j += 1u) {
            let cur = src[base + 1u + j];
            let q = i8x4f((prev >> 16u) | (cur << 16u)) * d0;
            let e = e0 + j * 4u;
            dst[e] = q.x;
            dst[e + 1u] = q.y;
            dst[e + 2u] = q.z;
            dst[e + 3u] = q.w;
            prev = cur;
        }
        for (var j = 0u; j < 8u; j += 1u) {
            let q = i8x4f(src[base + 9u + j]) * d1;
            let e = e0 + 32u + j * 4u;
            dst[e] = q.x;
            dst[e + 1u] = q.y;
            dst[e + 2u] = q.z;
            dst[e + 3u] = q.w;
        }
    }
}

// q8_0 tile [K, N] quantized along K (ggml MatMul-B convention): blocks tile a
// [K/32, N] grid, row-major, so consecutive blocks run along N. -> f32 [K, N]
// (same logical layout) for a normal [K,N] GEMM B operand. One work item = one
// block PAIR (two adjacent N columns, same 32-K range) so the 68-byte pair is
// word-aligned (17 u32). Each block's 32 int8 scatter down a column at stride N.
// Requires K % 32 == 0 and N % 2 == 0.
//   p.k = K, p.n = N, p.dst_row = N, p.count = (K/32) * (N/2) block pairs.
@compute @workgroup_size(64)
fn q8_kmajor_to_f32(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    let pairs_per_bk = p.n / 2u; // column pairs per block-row
    for (var i = g.x; i < p.count; i += step) {
        let bk = i / pairs_per_bk; // which 32-K block-row
        let np = i % pairs_per_bk; // which column pair
        let n0 = np * 2u;
        let k0 = bk * 32u;
        // Linear block index of (bk, n0) in the [K/32, N] grid is bk*N + n0
        // (even, since N is even) → word base = (bk*N + n0)/2 * 17.
        let base = ((bk * p.n + n0) / 2u) * 17u;

        let w0 = src[base];
        let d0 = unpack2x16float(w0).x;        // scale of block (bk, n0)
        let d1 = unpack2x16float(src[base + 8u]).y; // scale of block (bk, n0+1)

        // Block A -> column n0.
        var prev = w0;
        for (var j = 0u; j < 8u; j += 1u) {
            let cur = src[base + 1u + j];
            let q = i8x4f((prev >> 16u) | (cur << 16u)) * d0;
            let kb = k0 + j * 4u;
            dst[kb * p.dst_row + n0] = q.x;
            dst[(kb + 1u) * p.dst_row + n0] = q.y;
            dst[(kb + 2u) * p.dst_row + n0] = q.z;
            dst[(kb + 3u) * p.dst_row + n0] = q.w;
            prev = cur;
        }
        // Block B -> column n0+1.
        for (var j = 0u; j < 8u; j += 1u) {
            let q = i8x4f(src[base + 9u + j]) * d1;
            let kb = k0 + j * 4u;
            dst[kb * p.dst_row + n0 + 1u] = q.x;
            dst[(kb + 1u) * p.dst_row + n0 + 1u] = q.y;
            dst[(kb + 2u) * p.dst_row + n0 + 1u] = q.z;
            dst[(kb + 3u) * p.dst_row + n0 + 1u] = q.w;
        }
    }
}

// One work item = one u32 word = two f16 values widened in place-order.
@compute @workgroup_size(64)
fn f16_to_f32(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.count; i += step) {
        let v = unpack2x16float(src[i]);
        dst[i * 2u] = v.x;
        dst[i * 2u + 1u] = v.y;
    }
}

// One work item = one element. Round half AWAY from zero to match the CPU's
// `@round` (WGSL's builtin round() is half-to-even, which differs at exact
// .5 values). The i32 result is smuggled through the f32 dst binding.
@compute @workgroup_size(64)
fn f32_to_i32(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.count; i += step) {
        let v = bitcast<f32>(src[i]);
        dst[i] = bitcast<f32>(i32(sign(v) * floor(abs(v) + 0.5)));
    }
}

// One work item = one element: widen i32 (read via the u32 src binding) to f32.
@compute @workgroup_size(64)
fn i32_to_f32(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.count; i += step) {
        dst[i] = f32(bitcast<i32>(src[i]));
    }
}

// One work item = one OUTPUT u32 word = two f32 values narrowed to f16. The
// packed word is smuggled through the f32 dst binding via bitcast (WGSL can't
// bind array<f16> without the shader-f16 extension).
@compute @workgroup_size(64)
fn f32_to_f16(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.count; i += step) {
        let v = vec2<f32>(bitcast<f32>(src[i * 2u]), bitcast<f32>(src[i * 2u + 1u]));
        dst[i] = bitcast<f32>(pack2x16float(v));
    }
}

// One work item = one ELEMENT (see the f16 bindings above). These supersede the
// word-pair `f16_to_f32` / `f32_to_f16` entry points, which are kept only for
// reference; the backend dispatches these.
@compute @workgroup_size(64)
fn f16_to_f32_elem(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.count; i += step) { dst[i] = f32(src_h[i]); }
}

@compute @workgroup_size(64)
fn f32_to_f16_elem(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var i = g.x; i < p.count; i += step) { dst_h[i] = f16(bitcast<f32>(src[i])); }
}
