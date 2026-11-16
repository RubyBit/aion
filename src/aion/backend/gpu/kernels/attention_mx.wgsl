// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Prefill attention on 8x8 f32 matrix tiles (wgpu's cooperative matrices): the
// same math, bindings and `Params` as attention_block.wgsl. See
// exec/attention_mx.zig for the work shape.
//
// Lines `// #name` are filled in by exec/attention_mx.zig: every tile operation
// is written out there, because the shader compiler does not unroll loops and
// tiles cannot be indexed. The `// #shape` constants are set there too.
//
// TILE LAYOUT DEPENDS ON THE WGPU VERSION. This kernel relies on naga 29's Metal
// writer inverting the load/store layout: here `coopLoad`/`coopStore` are
// row-major and `coopLoadT` column-major. That is an upstream bug, fixed in wgpu
// 30 (gfx-rs/wgpu#9451): from then on `coopLoad` is column-major and `coopLoadT`
// row-major on every backend, as the spec says. Moving wgpu-native to >= 30 must
// swap every `coopLoad` <-> `coopLoadT` and `coopStore` <-> `coopStoreT` here
// and in exec/attention_mx.zig, checked on the M5: a wrong layout transposes
// the square tiles silently. (attention_mx16.wgsl already follows the spec.)

enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read>       q: array<f32>;
@group(0) @binding(1) var<storage, read>       kf: array<f32>; // K, f32 (an f16 cache widened)
@group(0) @binding(2) var<storage, read>       vf: array<f32>; // V, f32 (after K, when widened)
@group(0) @binding(3) var<storage, read>       pos: array<i32>;
@group(0) @binding(4) var<storage, read>       endi: array<i32>;
@group(0) @binding(5) var<storage, read_write> o: array<f32>;
@group(0) @binding(6) var<uniform>             p: Params;

// Identical to attention.wgsl `Params` (the host fills one struct for all three).
struct Params {
    tl: u32, th: u32, dk: u32, dv: u32, t_cap: u32, h_kv: u32, gqa: u32,
    win_left: u32, win_right: u32, win_chunk: u32, ring: u32, ring_modulus: u32,
    kv_f16: u32, scale: f32, soft_cap: f32, segs: u32, has_pos: u32, has_lengths: u32,
    rl: u32, rh: u32,
};

// #shape
const TH: u32 = 32u * NS; // invocations per workgroup
const KB: u32 = 32u; // keys per block: 4 tiles, KT of them scored by each SIMD group
const KC: u32 = 64u; // head dims per pass of the score loop: 8 tiles
const PER: u32 = KB * 8u / TH; // scores per invocation in the softmax
const THRESH: f32 = 8.0; // how far a score may pass a row's reference max
const FMIN: f32 = -3.4028235e38;

var<workgroup> st: array<f32, 8u * DV>; // the accumulators, when rescaled or written out
var<workgroup> ps: array<f32, 8u * KB>; // [row][key]: scores, then probabilities
var<workgroup> red: array<f32, TH>; // per-invocation partials of a row max or sum
var<workgroup> row_lo: array<u32, 8>; // each row's key window
var<workgroup> row_hi: array<u32, 8>;
var<workgroup> row_m: array<f32, 8>; // reference max
var<workgroup> row_l: array<f32, 8>; // sum of exp(s - reference)
var<workgroup> row_c: array<f32, 8>; // this block's rescale of the row (1 = none)
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
    return min(t, p.t_cap - 8u);
}

// The reduction of row `r`'s partials in `red` (PER scores apart, TH / 8 of them).
fn rowMax(r: u32) -> f32 {
    var m = FMIN;
    for (var i = 0u; i < TH / 8u; i += 1u) { m = max(m, red[r * (TH / 8u) + i]); }
    return m;
}

fn rowSum(r: u32) -> f32 {
    var s = 0.0;
    for (var i = 0u; i < TH / 8u; i += 1u) { s += red[r * (TH / 8u) + i]; }
    return s;
}

// One workgroup per 8 consecutive positions of one head. SIMD group `sg` scores
// key tiles sg * KT .. of each block, and accumulates value dims sg * SD .. .
@compute @workgroup_size(TH)
fn attn_mx(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let h = wid.x;
    let b = wid.z;
    let hkv = h / p.gqa;
    let sg = t / 32u;
    // The last block slides back to stay whole (the host guarantees tl >= 8) and
    // writes only the rows no earlier block owns.
    let own_row = wid.y * 8u;
    let row0 = min(own_row, p.tl - 8u);

    // --- each row's key window, and the block's span ---
    var valid_end = p.t_cap;
    if (p.has_lengths != 0u) { valid_end = u32(endi[b]); }
    if (t < 8u) {
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
        for (var r = 0u; r < 8u; r += 1u) {
            if (row_hi[r] > row_lo[r]) { s = vec2(min(s.x, row_lo[r]), max(s.y, row_hi[r])); }
        }
        // Blocks start on a multiple of 8, so a tile of keys is 8 consecutive slots.
        span = select(vec2(0u), vec2(s.x & ~7u, s.y), s.y > s.x);
    }
    // Tile operations may only run where every invocation runs alike, so the
    // span that bounds the key loop is made uniform.
    let sp = workgroupUniformLoad(&span);

    // Rows of the block are th * dk apart in q; keys are h_kv * dk apart in the
    // cache. A widened cache holds V after every batch's K.
    let qrow = p.th * p.dk;
    let qb = ((b * p.tl + row0) * p.th + h) * p.dk;
    let krow = p.h_kv * p.dk;
    let vrow = p.h_kv * p.dv;
    let kb = b * p.t_cap * krow + hkv * p.dk;
    let vb = select(0u, nwg.z * p.t_cap * krow, p.kv_f16 != 0u) + b * p.t_cap * vrow + hkv * p.dv + sg * SD;
    let kt0 = sg * KT; // this SIMD group's first key tile

    // #acc_decl

    for (var t0 = sp.x; t0 < sp.y; t0 += KB) {
        // --- scores: 8 rows x this SIMD group's KT tiles of 8 keys ---
        // #score_decl
        for (var d0 = 0u; d0 < p.dk; d0 += KC) {
            // #score_tiles
        }
        // #score_store
        workgroupBarrier();

        // --- softmax: an invocation owns PER keys of row t / (TH / 8) ---
        let r = t / (TH / 8u);
        let c0 = (t % (TH / 8u)) * PER;
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
        // until then exp(s - reference) <= e^THRESH, well inside f32.
        if (t < 8u) {
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
            for (var i = 0u; i < 8u; i += 1u) { any_moved |= u32(row_c[i] != 1.0); }
            moved = any_moved;
        }
        if (workgroupUniformLoad(&moved) != 0u) {
            // #store_acc
            workgroupBarrier();
            for (var e = t; e < 8u * DV; e += TH) { st[e] *= row_c[e / DV]; }
            workgroupBarrier();
            // #load_acc
            workgroupBarrier();
        }

        // --- probabilities ---
        let m = row_m[r];
        var sum = 0.0;
        for (var j = 0u; j < PER; j += 1u) {
            let k = r * KB + c0 + j;
            let x = ps[k];
            var e = 0.0;
            if (x > FMIN) { e = expApprox(x - m); }
            ps[k] = e;
            sum += e;
        }
        red[t] = sum;
        workgroupBarrier();
        if (t < 8u) { row_l[t] += rowSum(t); }

        // --- out += P @ V, this SIMD group's value dims ---
        let v0 = vb + slot(t0) * vrow;
        let v1 = vb + slot(t0 + 8u) * vrow;
        let v2 = vb + slot(t0 + 16u) * vrow;
        let v3 = vb + slot(t0 + 24u) * vrow;
        // #value_tiles
        workgroupBarrier();
    }

    // --- normalize and write the rows this block owns ---
    // #store_acc
    workgroupBarrier();
    for (var e = t; e < 8u * DV; e += TH) {
        let rr = e / DV;
        let d = e % DV;
        let l = row0 + rr;
        if (l >= own_row) {
            let lsum = row_l[rr];
            o[((b * p.tl + l) * p.th + h) * p.dv + d] = select(0.0, st[e] / lsum, lsum > 0.0);
        }
    }
}
