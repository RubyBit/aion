// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Prefill attention on 16x16x16 f16 matrix tiles with f32 accumulation (NVIDIA's
// tensor cores through Vulkan): the same math and `Params` as
// attention_block.wgsl, with q rounded to f16 first and an f16 K/V cache read in
// place. See exec/attention_mx16.zig for the work shape; this is
// attention_mx.wgsl (8x8 f32 tiles) at the tile size NVIDIA has.
//
// Scores and the softmax stay f32; the probabilities are rounded to f16 for
// P @ V, which accumulates in f32 again.
//
// Tile layouts follow the spec: `coopLoad` is column-major, `coopLoadT` row-major
// (what naga emits on SPIR-V; its Metal writer inverts them until wgpu 30,
// gfx-rs/wgpu#9451, which the `_mx` kernels rely on). So here: q, P and V, stored
// as rows, load with `coopLoadT`; K^T, read from K's rows, with `coopLoad`.
//
// Lines `// #name` are filled in by exec/attention_mx16.zig: every tile operation
// is written out there, because the shader compiler does not unroll loops and
// tiles cannot be indexed. The `// #shape` constants are set there too.

enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read>       qh: array<f16>; // q rounded to f16
@group(0) @binding(1) var<storage, read>       kh: array<f16>; // the K cache
@group(0) @binding(2) var<storage, read>       vh: array<f16>; // the V cache
@group(0) @binding(3) var<storage, read>       pos: array<i32>;
@group(0) @binding(4) var<storage, read>       endi: array<i32>;
@group(0) @binding(5) var<storage, read_write> o: array<f32>;
@group(0) @binding(6) var<uniform>             p: Params;

// Identical to attention.wgsl `Params` (the host fills one struct for all of them).
struct Params {
    tl: u32, th: u32, dk: u32, dv: u32, t_cap: u32, h_kv: u32, gqa: u32,
    win_left: u32, win_right: u32, win_chunk: u32, ring: u32, ring_modulus: u32,
    kv_f16: u32, scale: f32, soft_cap: f32, segs: u32, has_pos: u32, has_lengths: u32,
    rl: u32, rh: u32,
};

// #shape
const R: u32 = 16u; // query rows per block: one tile
const TH: u32 = 32u * NS; // invocations per workgroup
const KB: u32 = 64u; // keys per block: 4 tiles, KT of them scored by each SIMD group
const KC: u32 = 64u; // head dims per pass of the score loop: 4 tiles
const PART: u32 = TH / R; // invocations per row in the softmax
const PER: u32 = KB / PART; // scores per invocation in the softmax
const THRESH: f32 = 8.0; // how far a score may pass a row's reference max
const FMIN: f32 = -3.4028235e38;

var<workgroup> st: array<f32, R * DV>; // the accumulators, when rescaled or written out
var<workgroup> ps: array<f32, R * KB>; // [row][key] scores
var<workgroup> ph: array<f16, R * KB>; // [row][key] probabilities, for P @ V
var<workgroup> red: array<f32, TH>; // per-invocation partials of a row max or sum
var<workgroup> row_lo: array<u32, R>; // each row's key window
var<workgroup> row_hi: array<u32, R>;
var<workgroup> row_m: array<f32, R>; // reference max
var<workgroup> row_l: array<f32, R>; // sum of exp(s - reference)
var<workgroup> row_c: array<f32, R>; // this block's rescale of the row (1 = none)
var<workgroup> span: vec2<u32>; // the keys any row of the block sees
var<workgroup> moved: u32; // some row's reference moved this block

// The exp and tanh the other attention kernels and the CPU use, so all agree.
fn expApprox(x_in: f32) -> f32 {
    let xc = clamp(x_in, -80.0, 80.0);
    let yy = xc * 1.4426950408889634;
    let nn = i32(floor(yy + 0.5));
    let tt = (yy - f32(nn)) * 0.6931471805599453;
    let e2 = bitcast<f32>(u32(nn + 127) << 23u);
    return e2 * (1.0 + tt * (1.0 + tt * (0.5 + tt * (0.16666667 + tt * 0.041666668))));
}

fn capped(x: f32) -> f32 {
    if (p.soft_cap <= 0.0) { return x; }
    let e = expApprox(-2.0 * abs(x / p.soft_cap));
    let y = select(e / (1.0 + e), 1.0 / (1.0 + e), x >= 0.0);
    return p.soft_cap * clamp(2.0 * y - 1.0, -1.0, 1.0);
}

// The cache slot of logical key `t`. Past the cache only when every key of its
// tile is past every row's window: the last whole tile is read instead, and the
// mask drops it.
fn slot(t: u32) -> u32 {
    if (p.ring != 0u) { return t % p.ring_modulus; }
    return min(t, p.t_cap - R);
}

// The reduction of row `r`'s partials in `red` (PART of them).
fn rowMax(r: u32) -> f32 {
    var m = FMIN;
    for (var i = 0u; i < PART; i += 1u) { m = max(m, red[r * PART + i]); }
    return m;
}

fn rowSum(r: u32) -> f32 {
    var s = 0.0;
    for (var i = 0u; i < PART; i += 1u) { s += red[r * PART + i]; }
    return s;
}

// One workgroup per 16 consecutive positions of one head. SIMD group `sg` scores
// key tiles sg * KT .. of each block, and accumulates value dims sg * SD .. .
@compute @workgroup_size(TH)
fn attn_mx16(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let h = wid.x;
    let b = wid.z;
    let hkv = h / p.gqa;
    let sg = t / 32u;
    // The last block slides back to stay whole (the host guarantees tl >= R) and
    // writes only the rows no earlier block owns.
    let own_row = wid.y * R;
    let row0 = min(own_row, p.tl - R);

    // --- each row's key window, and the block's span ---
    var valid_end = p.t_cap;
    if (p.has_lengths != 0u) { valid_end = u32(endi[b]); }
    if (t < R) {
        let l = row0 + t;
        var q_pos = l;
        if (p.has_pos != 0u) { q_pos = u32(pos[b * p.tl + l]); }
        let w = window_keys(p.win_left, p.win_right, p.win_chunk, q_pos, valid_end);
        var lo = w.x;
        var hi = w.y;
        if (p.ring != 0u && valid_end > p.ring_modulus) { lo = max(lo, valid_end - p.ring_modulus); }
        if (p.ring == 0u) { hi = min(hi, p.t_cap); }
        row_lo[t] = lo;
        row_hi[t] = max(hi, lo);
        row_m[t] = FMIN;
        row_l[t] = 0.0;
    }
    workgroupBarrier();
    if (t == 0u) {
        var s = vec2(0xffffffffu, 0u);
        for (var r = 0u; r < R; r += 1u) {
            if (row_hi[r] > row_lo[r]) { s = vec2(min(s.x, row_lo[r]), max(s.y, row_hi[r])); }
        }
        // Blocks start on a multiple of 16, so a tile of keys is 16 consecutive slots.
        span = select(vec2(0u), vec2(s.x & ~(R - 1u), s.y), s.y > s.x);
    }
    // Tile operations may only run where every invocation runs alike, so the
    // span that bounds the key loop is made uniform.
    let sp = workgroupUniformLoad(&span);

    // Rows of the block are th * dk apart in q; keys are h_kv * dk apart in the
    // cache. Strides are named values: naga 29's WGSL frontend emits a uniform
    // load passed to `coopStore` after the store ("Expression is not cached",
    // gfx-rs/wgpu#10901, open).
    let qrow = p.th * p.dk;
    let qb = ((b * p.tl + row0) * p.th + h) * p.dk;
    let krow = p.h_kv * p.dk;
    let vrow = p.h_kv * p.dv;
    let kb = b * p.t_cap * krow + hkv * p.dk;
    let vb = b * p.t_cap * vrow + hkv * p.dv + sg * SD;
    let kt0 = sg * KT; // this SIMD group's first key tile
    let dk = p.dk;

    // #acc_decl

    for (var t0 = sp.x; t0 < sp.y; t0 += KB) {
        // --- scores: 16 rows x this SIMD group's KT tiles of 16 keys ---
        // #score_decl
        for (var d0 = 0u; d0 < dk; d0 += KC) {
            // #score_tiles
        }
        // #score_store
        workgroupBarrier();

        // --- softmax: an invocation owns PER keys of row t / PART ---
        let r = t / PART;
        let c0 = (t % PART) * PER;
        let lo = row_lo[r];
        let hi = row_hi[r];
        var mx = FMIN;
        for (var j = 0u; j < PER; j += 1u) {
            let k = c0 + j;
            let tk = t0 + k;
            var x = FMIN;
            if (tk >= lo && tk < hi) { x = capped(ps[r * KB + k] * p.scale); }
            ps[r * KB + k] = x;
            mx = max(mx, x);
        }
        red[t] = mx;
        workgroupBarrier();

        // The accumulated tiles cannot be rescaled in place, so a row keeps a
        // reference max and moves it only when a score passes it by THRESH:
        // until then exp(s - reference) <= e^THRESH, inside f16 too.
        if (t < R) {
            let top = rowMax(t);
            let ref_m = row_m[t];
            var c = 1.0;
            if (top > ref_m + THRESH) {
                c = select(0.0, expApprox(ref_m - top), ref_m > FMIN);
                row_m[t] = top;
                row_l[t] *= c;
            }
            row_c[t] = c;
        }
        workgroupBarrier();
        if (t == 0u) {
            var any_moved = 0u;
            for (var i = 0u; i < R; i += 1u) { any_moved |= u32(row_c[i] != 1.0); }
            moved = any_moved;
        }
        if (workgroupUniformLoad(&moved) != 0u) {
            // #store_acc
            workgroupBarrier();
            for (var e = t; e < R * DV; e += TH) { st[e] *= row_c[e / DV]; }
            workgroupBarrier();
            // #load_acc
            workgroupBarrier();
        }

        // --- probabilities, summed as rounded for P @ V ---
        let m = row_m[r];
        var sum = 0.0;
        for (var j = 0u; j < PER; j += 1u) {
            let k = r * KB + c0 + j;
            let x = ps[k];
            var e = f16(0.0);
            if (x > FMIN) { e = f16(expApprox(x - m)); }
            ph[k] = e;
            sum += f32(e);
        }
        red[t] = sum;
        workgroupBarrier();
        if (t < R) { row_l[t] += rowSum(t); }

        // --- out += P @ V, this SIMD group's value dims ---
        let v0 = vb + slot(t0) * vrow;
        let v1 = vb + slot(t0 + 16u) * vrow;
        let v2 = vb + slot(t0 + 32u) * vrow;
        let v3 = vb + slot(t0 + 48u) * vrow;
        // #value_tiles
        workgroupBarrier();
    }

    // --- normalize and write the rows this block owns ---
    // #store_acc
    workgroupBarrier();
    for (var e = t; e < R * DV; e += TH) {
        let rr = e / DV;
        let d = e % DV;
        let l = row0 + rr;
        if (l >= own_row) {
            let lsum = row_l[rr];
            o[((b * p.tl + l) * p.th + h) * p.dv + d] = select(0.0, st[e] / lsum, lsum > 0.0);
        }
    }
}
