// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Direct convolution (channel-last), f32 — the correctness-first base case:
// one thread per output element, grid-strided, looping the kernel window and
// the per-group input channels. No im2col / implicit GEMM yet; that's a
// planned optimization once profiles justify it.
//
// Layouts (packed):
//   conv1d: x [l_in, c_in] (one batch, p.x_base pre-offsets the batch row),
//           w [k, c_in_g, c_out], bias [c_out], out tile [l_cnt, c_cnt]
//   conv2d: x [h_in, w_in, c_in], w [kh, kw, c_in_g, c_out], bias [c_out],
//           out tile [oh_cnt, ow_cnt, c_cnt]
// The out tile may be an edge tile (base_* / *_cnt describe its window); w and
// bias index by GLOBAL output channel.
//
// Padding: zero (skip out-of-range taps) or reflect (mirror around the edges,
// matching the CPU reflectIndex1D: x < 0 -> -x, x >= L -> 2L-2-x, repeated).
// When `has_bias == 0`, the bias binding is a dummy (the backend rebinds w).

@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       w: array<f32>;
@group(0) @binding(2) var<storage, read>       bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@group(0) @binding(4) var<uniform>             p: Params;

struct Params {
    // Input geometry. Conv1d uses h_* for the length axis (w_in/kw/etc = 1).
    x_base: u32, // element offset of this batch's [h_in(, w_in), c_in] block
    h_in: u32,
    w_in: u32,
    c_in: u32,
    // Filter geometry.
    kh: u32,
    kw: u32,
    c_in_g: u32, // c_in / groups
    c_out_g: u32, // c_out / groups
    c_out: u32,
    // Strides / dilation / padding (leading edge only; trailing pad affects
    // only the output size, which is baked into the out shape).
    stride_h: u32,
    stride_w: u32,
    dil_h: u32,
    dil_w: u32,
    pad_top: u32,
    pad_left: u32,
    // Output tile window.
    base_h: u32,
    base_w: u32,
    base_c: u32,
    oh_cnt: u32,
    ow_cnt: u32,
    c_cnt: u32,
    total: u32, // oh_cnt * ow_cnt * c_cnt
    reflect: u32,
    has_bias: u32,
};

const WG: u32 = 64u;

// Mirror an out-of-range coordinate into [0, len) — CPU reflectIndex1D.
fn reflect_idx(idx: i32, len: i32) -> i32 {
    var v = idx;
    loop {
        if (v >= 0 && v < len) { break; }
        if (v < 0) { v = -v; } else { v = (2 * len - 2) - v; }
    }
    return v;
}

fn conv_at(oh: u32, ow: u32, co: u32) -> f32 {
    let g = co / p.c_out_g;
    let c0 = g * p.c_in_g;

    var acc = 0.0;
    if (p.has_bias != 0u) { acc = bias[co]; }

    let h0 = i32(oh * p.stride_h) - i32(p.pad_top);
    let w0 = i32(ow * p.stride_w) - i32(p.pad_left);

    for (var ki = 0u; ki < p.kh; ki += 1u) {
        var hi = h0 + i32(ki * p.dil_h);
        if (p.reflect != 0u) {
            hi = reflect_idx(hi, i32(p.h_in));
        } else if (hi < 0 || hi >= i32(p.h_in)) {
            continue;
        }
        for (var kj = 0u; kj < p.kw; kj += 1u) {
            var wi = w0 + i32(kj * p.dil_w);
            if (p.reflect != 0u) {
                wi = reflect_idx(wi, i32(p.w_in));
            } else if (wi < 0 || wi >= i32(p.w_in)) {
                continue;
            }
            let x_off = p.x_base + ((u32(hi) * p.w_in + u32(wi)) * p.c_in) + c0;
            let w_off = ((ki * p.kw + kj) * p.c_in_g) * p.c_out + co;
            for (var ci = 0u; ci < p.c_in_g; ci += 1u) {
                acc += x[x_off + ci] * w[w_off + ci * p.c_out];
            }
        }
    }
    return acc;
}

@compute @workgroup_size(64)
fn conv_f32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let stride = nwg.x * WG;
    for (var idx = gid.x; idx < p.total; idx += stride) {
        let cc = idx % p.c_cnt;
        let rest = idx / p.c_cnt;
        let owc = rest % p.ow_cnt;
        let ohc = rest / p.ow_cnt;
        o[idx] = conv_at(p.base_h + ohc, p.base_w + owc, p.base_c + cc);
    }
}

// vec4 contraction over the per-group input channels (requires c_in_g % 4 == 0).
// x channels within a group are contiguous -> 128-bit load; w strides by c_out
// across input channels, so 4 scalar reads combined via dot.
fn conv_at_vec4c(oh: u32, ow: u32, co: u32) -> f32 {
    let g = co / p.c_out_g;
    let c0 = g * p.c_in_g;

    var acc = 0.0;
    if (p.has_bias != 0u) { acc = bias[co]; }

    let h0 = i32(oh * p.stride_h) - i32(p.pad_top);
    let w0 = i32(ow * p.stride_w) - i32(p.pad_left);

    for (var ki = 0u; ki < p.kh; ki += 1u) {
        var hi = h0 + i32(ki * p.dil_h);
        if (p.reflect != 0u) {
            hi = reflect_idx(hi, i32(p.h_in));
        } else if (hi < 0 || hi >= i32(p.h_in)) {
            continue;
        }
        for (var kj = 0u; kj < p.kw; kj += 1u) {
            var wi = w0 + i32(kj * p.dil_w);
            if (p.reflect != 0u) {
                wi = reflect_idx(wi, i32(p.w_in));
            } else if (wi < 0 || wi >= i32(p.w_in)) {
                continue;
            }
            let x_off = p.x_base + ((u32(hi) * p.w_in + u32(wi)) * p.c_in) + c0;
            let w_off = ((ki * p.kw + kj) * p.c_in_g) * p.c_out + co;
            let cs = p.c_out;
            for (var ci = 0u; ci < p.c_in_g; ci += 4u) {
                let xo = x_off + ci;
                let xv = vec4<f32>(x[xo], x[xo + 1u], x[xo + 2u], x[xo + 3u]);
                let wb = w_off + ci * cs;
                let wv = vec4<f32>(w[wb], w[wb + cs], w[wb + 2u * cs], w[wb + 3u * cs]);
                acc += dot(xv, wv);
            }
        }
    }
    return acc;
}

@compute @workgroup_size(64)
fn conv_f32_vec4c(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let stride = nwg.x * WG;
    for (var idx = gid.x; idx < p.total; idx += stride) {
        let cc = idx % p.c_cnt;
        let rest = idx / p.c_cnt;
        let owc = rest % p.ow_cnt;
        let ohc = rest / p.ow_cnt;
        o[idx] = conv_at_vec4c(p.base_h + ohc, p.base_w + owc, p.base_c + cc);
    }
}

// ---------------------------------------------------------------------------
// Depthwise conv1d fast path (zero-pad only). Selected by the executor when
// c_in_g == 1 && groups == c_in == c_out (channel-last conv1d), c_out % 4 == 0,
// and the input span fits DW_SPAN_MAX.
//
// One workgroup owns [DW_NCG channel-groups (vec4) x DW_LT output positions].
// Each thread owns one vec4 channel-group and computes DW_LT length outputs,
// reusing a shared input halo (each input row read once) and per-tap weights
// held in a register across all DW_LT outputs. Because c_out % 4 == 0 and the
// channel tile base is 4-aligned, every channel-group is full (no per-lane tail).
// Causal/zero padding falls out for free: out-of-range halo rows load as zero.
// ---------------------------------------------------------------------------
const DW_NCG: u32 = 64u;   // channel-groups (of 4 channels) per workgroup -> 256 ch
const DW_LT: u32 = 16u;    // output length positions per workgroup (unrolled)
const DW_SPAN_MAX: u32 = 40u; // Xs = 40*64*16 = 40960 B shared (executor gates on limits)

var<workgroup> Xs: array<vec4<f32>, DW_SPAN_MAX * DW_NCG>;

fn store_dw(oh: u32, gc4: u32, acc: vec4<f32>) {
    if (oh >= p.oh_cnt) { return; }
    let idx = oh * p.c_cnt + gc4 * 4u;
    o[idx] = acc.x;
    o[idx + 1u] = acc.y;
    o[idx + 2u] = acc.z;
    o[idx + 3u] = acc.w;
}

@compute @workgroup_size(64)
fn conv_dw_f32(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let gc4 = wid.x * DW_NCG + lid;      // channel-group index within the out tile
    let lblk = wid.y * DW_LT;            // first tile-local length position
    let co = p.base_c + gc4 * 4u;        // absolute first output channel of this group

    let span = (DW_LT - 1u) * p.stride_h + (p.kh - 1u) * p.dil_h + 1u;
    let block_c0 = p.base_c + wid.x * DW_NCG * 4u;
    let gfirst = i32((p.base_h + lblk) * p.stride_h) - i32(p.pad_top);

    // Cooperative halo load: fill Xs[row * DW_NCG + cg] for this channel block.
    var s = lid;
    loop {
        if (s >= span * DW_NCG) { break; }
        let row = s / DW_NCG;
        let cg = s % DW_NCG;
        let gr = gfirst + i32(row);
        let cch = block_c0 + cg * 4u;
        var v = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        if (gr >= 0 && gr < i32(p.h_in) && cch + 3u < p.c_in) {
            let base = p.x_base + u32(gr) * p.c_in + cch;
            v = vec4<f32>(x[base], x[base + 1u], x[base + 2u], x[base + 3u]);
        }
        Xs[s] = v;
        s = s + DW_NCG;
    }
    workgroupBarrier();

    if (gc4 * 4u >= p.c_cnt) { return; }

    var bv = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if (p.has_bias != 0u) {
        bv = vec4<f32>(bias[co], bias[co + 1u], bias[co + 2u], bias[co + 3u]);
    }
    var a0 = bv; var a1 = bv; var a2 = bv; var a3 = bv;
    var a4 = bv; var a5 = bv; var a6 = bv; var a7 = bv;
    var a8 = bv; var a9 = bv; var a10 = bv; var a11 = bv;
    var a12 = bv; var a13 = bv; var a14 = bv; var a15 = bv;

    var ki = 0u;
    loop {
        if (ki >= p.kh) { break; }
        let woff = ki * p.c_out + co;
        let wv = vec4<f32>(w[woff], w[woff + 1u], w[woff + 2u], w[woff + 3u]);
        let base = ki * p.dil_h;
        a0 = a0 + Xs[(0u * p.stride_h + base) * DW_NCG + lid] * wv;
        a1 = a1 + Xs[(1u * p.stride_h + base) * DW_NCG + lid] * wv;
        a2 = a2 + Xs[(2u * p.stride_h + base) * DW_NCG + lid] * wv;
        a3 = a3 + Xs[(3u * p.stride_h + base) * DW_NCG + lid] * wv;
        a4 = a4 + Xs[(4u * p.stride_h + base) * DW_NCG + lid] * wv;
        a5 = a5 + Xs[(5u * p.stride_h + base) * DW_NCG + lid] * wv;
        a6 = a6 + Xs[(6u * p.stride_h + base) * DW_NCG + lid] * wv;
        a7 = a7 + Xs[(7u * p.stride_h + base) * DW_NCG + lid] * wv;
        a8 = a8 + Xs[(8u * p.stride_h + base) * DW_NCG + lid] * wv;
        a9 = a9 + Xs[(9u * p.stride_h + base) * DW_NCG + lid] * wv;
        a10 = a10 + Xs[(10u * p.stride_h + base) * DW_NCG + lid] * wv;
        a11 = a11 + Xs[(11u * p.stride_h + base) * DW_NCG + lid] * wv;
        a12 = a12 + Xs[(12u * p.stride_h + base) * DW_NCG + lid] * wv;
        a13 = a13 + Xs[(13u * p.stride_h + base) * DW_NCG + lid] * wv;
        a14 = a14 + Xs[(14u * p.stride_h + base) * DW_NCG + lid] * wv;
        a15 = a15 + Xs[(15u * p.stride_h + base) * DW_NCG + lid] * wv;
        ki = ki + 1u;
    }

    store_dw(lblk + 0u, gc4, a0);
    store_dw(lblk + 1u, gc4, a1);
    store_dw(lblk + 2u, gc4, a2);
    store_dw(lblk + 3u, gc4, a3);
    store_dw(lblk + 4u, gc4, a4);
    store_dw(lblk + 5u, gc4, a5);
    store_dw(lblk + 6u, gc4, a6);
    store_dw(lblk + 7u, gc4, a7);
    store_dw(lblk + 8u, gc4, a8);
    store_dw(lblk + 9u, gc4, a9);
    store_dw(lblk + 10u, gc4, a10);
    store_dw(lblk + 11u, gc4, a11);
    store_dw(lblk + 12u, gc4, a12);
    store_dw(lblk + 13u, gc4, a13);
    store_dw(lblk + 14u, gc4, a14);
    store_dw(lblk + 15u, gc4, a15);
}
