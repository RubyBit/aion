// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Grouped-query attention over a block of query rows at once — the prefill twin of
// attention.wgsl, with the same math, bindings, and `Params`:
//   out[b, l, hq, :] = softmax(scale * q[b, l, hq, :] @ K[b, t, hkv, :]^T) @ V[b, t, hkv, :]
//
// WORK SHAPE — one 256-thread workgroup per block of R = 32 rows (`rl` positions x
// `rh` heads of one kv head), walking the block's key range in tiles of BK = 64:
//
//   * Scores are a register-tiled R x BK product: thread (tr, tk) = (tid / 16,
//     tid % 16) owns rows 2tr, 2tr+1 and keys 4tk..4tk+3, with q and k staged
//     through shared memory DC = 16 dims at a time.
//   * The online softmax runs per row in registers; row max and sum reduce through
//     shared partials, one slot per (row, tk).
//   * P goes through shared memory, and P @ V accumulates into registers. A
//     workgroup owns one DVS = 256-dim slice of the output (the grid's x carries
//     the slice), so thread (tr, tk) holds its two rows' dims tk + 16j in named
//     vec4 registers; recomputing the scores per slice is what keeps them there.
//
// Every K and V row is read once per block instead of once per 4 rows, and the
// products run out of registers — the decode kernel's per-row key scan is what
// capped prefill there. The host routes here for f32 q over a single k/v tile.

enable f16;

@group(0) @binding(0) var<storage, read>       q: array<f32>;
@group(0) @binding(1) var<storage, read>       kc: array<u32>;
@group(0) @binding(2) var<storage, read>       vc: array<u32>;
@group(0) @binding(3) var<storage, read>       pos: array<i32>;
@group(0) @binding(4) var<storage, read>       endi: array<i32>;
@group(0) @binding(5) var<storage, read_write> o: array<f32>;
@group(0) @binding(6) var<uniform>             p: Params;

// Identical to attention.wgsl `Params` (the host fills one struct for both).
struct Params {
    base_b: u32,
    base_h: u32,
    tl: u32,
    th: u32,
    dk: u32,
    dv: u32,
    t_cap: u32,
    h_kv: u32,
    gqa: u32,
    win_left: u32,
    win_right: u32,
    win_chunk: u32,
    ring: u32,
    ring_modulus: u32,
    kv_f16: u32,
    scale: f32,
    soft_cap: f32,
    segs: u32,
    base_l: u32,
    has_pos: u32,
    has_lengths: u32,
    rl: u32,
    rh: u32,
    kv_t0: u32,
    kv_tile_t: u32,
    seg_base: u32,
    segs_local: u32,
};

const R: u32 = 32u;
const BK: u32 = 64u;
const DC: u32 = 16u;
const DS: u32 = 17u; // staged row stride: DC + 1 keeps the 16 tk lanes off one bank
const VK: u32 = 32u; // keys per V staging pass
const VD: u32 = 64u; // dims per V staging pass
const DVS: u32 = 256u; // value dims per workgroup slice
const FMIN: f32 = -3.4028235e38;
const NO_KEY: u32 = 0xffffffffu;

var<workgroup> p_sh: array<f32, 2048>; // R x BK probabilities
// Staging, reused by phase: q (R x DS) + k (BK x DS) chunks, then the row
// max / sum partials (R x 16 each), then a V chunk (VK x VD).
var<workgroup> st: array<f32, 2048>;
var<workgroup> t_sh: array<u32, 64>; // physical time of each tile key, or NO_KEY
var<workgroup> lo_sh: array<u32, 32>;
var<workgroup> hi_sh: array<u32, 32>;

// The exp and tanh the decode kernel and the CPU use, so both kernels agree.
fn expApprox(x_in: f32) -> f32 {
    let xc = clamp(x_in, -80.0, 80.0);
    let yy = xc * 1.4426950408889634;
    let nn = i32(floor(yy + 0.5));
    let tt = (yy - f32(nn)) * 0.6931471805599453;
    let e2 = bitcast<f32>(u32(nn + 127) << 23u);
    return e2 * (1.0 + tt * (1.0 + tt * (0.5 + tt * (0.16666667 + tt * 0.041666668))));
}

fn tanhApprox(v: f32) -> f32 {
    var y: f32;
    let x2 = 2.0 * v;
    if (x2 >= 0.0) { y = 1.0 / (1.0 + expApprox(-x2)); } else { let e = expApprox(x2); y = e / (1.0 + e); }
    return clamp(2.0 * clamp(y, 0.0, 1.0) - 1.0, -1.0, 1.0);
}

// One k / v element: two per word for f16 caches, one bitcast word for f32.
fn kElem(idx: u32) -> f32 {
    if (p.kv_f16 != 0u) {
        let h = unpack2x16float(kc[idx / 2u]);
        return select(h.x, h.y, (idx & 1u) == 1u);
    }
    return bitcast<f32>(kc[idx]);
}

fn vElem(idx: u32) -> f32 {
    if (p.kv_f16 != 0u) {
        let h = unpack2x16float(vc[idx / 2u]);
        return select(h.x, h.y, (idx & 1u) == 1u);
    }
    return bitcast<f32>(vc[idx]);
}

// Four value dims of a staged V chunk at `kk`, the ones lane `tk` accumulates.
fn vLanes(kk: u32, tk: u32) -> vec4<f32> {
    let at = kk * VD + tk;
    return vec4<f32>(st[at], st[at + 16u], st[at + 32u], st[at + 48u]);
}

/// Stage keys `[k0, k0 + VK)` of the tile's V rows, dims `[v0, v0 + VD)`, into `st`.
fn stageV(tid: u32, b: u32, hkv: u32, k0: u32, v0: u32) {
    for (var e = tid; e < VK * VD; e += 256u) {
        let kk = e / VD;
        let dd = e % VD;
        let tp = t_sh[k0 + kk];
        var x = 0.0;
        if (tp != NO_KEY && v0 + dd < p.dv) {
            x = vElem(((b * p.t_cap + tp) * p.h_kv + hkv) * p.dv + v0 + dd);
        }
        st[e] = x;
    }
}

@compute @workgroup_size(256)
fn attn_block(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let slices = (p.dv + DVS - 1u) / DVS;
    let vs = (wid.x % slices) * DVS;
    let b_local = wid.z;
    let b = p.base_b + b_local;
    let blk_h = (wid.x / slices) * p.rh;
    let blk_l = wid.y * p.rl;
    let hkv = (p.base_h + blk_h) / p.gqa;
    let rows = p.rl * p.rh;

    var valid_end = p.t_cap;
    if (p.has_lengths != 0u) { valid_end = u32(endi[b]); }

    // --- each row's key window, published for the block's union ---
    if (tid < R) {
        var lo = 0u;
        var hi = 0u;
        let l_local = blk_l + tid / p.rh;
        let h_local = blk_h + tid % p.rh;
        if (tid < rows && l_local < p.tl && h_local < p.th) {
            var q_pos = p.base_l + l_local;
            if (p.has_pos != 0u) { q_pos = u32(pos[b_local * p.tl + l_local]); }
            let w = window_keys(p.win_left, p.win_right, p.win_chunk, q_pos, valid_end);
            lo = w.x;
            hi = w.y;
            if (p.ring != 0u && valid_end > p.ring_modulus) { lo = max(lo, valid_end - p.ring_modulus); }
            if (hi <= lo) { lo = 0u; hi = 0u; }
        }
        lo_sh[tid] = lo;
        hi_sh[tid] = hi;
    }
    workgroupBarrier();

    let tr = tid / 16u;
    let tk = tid % 16u;
    let r0 = 2u * tr;
    let r1 = r0 + 1u;
    var span_lo = 0xffffffffu;
    var span_hi = 0u;
    for (var r = 0u; r < R; r += 1u) {
        if (hi_sh[r] > lo_sh[r]) {
            span_lo = min(span_lo, lo_sh[r]);
            span_hi = max(span_hi, hi_sh[r]);
        }
    }
    let lo0 = lo_sh[r0];
    let hi0 = hi_sh[r0];
    let lo1 = lo_sh[r1];
    let hi1 = hi_sh[r1];

    // Rows r0 / r1; vec4 `c` holds dims vs + 64c + tk + 16 * (0..3).
    var a0 = array<vec4<f32>, 4>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    var a1 = array<vec4<f32>, 4>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    var m0 = FMIN;
    var m1 = FMIN;
    var l0 = 0.0;
    var l1 = 0.0;

    for (var t0 = span_lo; t0 < span_hi; t0 += BK) {
        // --- the tile's keys: logical t0 + kk, mapped to physical time ---
        workgroupBarrier(); // the previous tile is done with t_sh, st and p_sh
        if (tid < BK) {
            let tj = t0 + tid;
            var tp = NO_KEY;
            if (tj < span_hi) {
                var phys = tj;
                if (p.ring != 0u) { phys = tj % p.ring_modulus; }
                if (phys < p.t_cap) { tp = phys; }
            }
            t_sh[tid] = tp;
        }

        // --- scores: s0 / s1 = rows r0 / r1 against keys 4tk .. 4tk+3 ---
        var s0 = vec4<f32>(0.0);
        var s1 = vec4<f32>(0.0);
        for (var d0 = 0u; d0 < p.dk; d0 += DC) {
            workgroupBarrier();
            for (var e = tid; e < R * DC; e += 256u) {
                let r = e / DC;
                let dd = e % DC;
                let l_local = blk_l + r / p.rh;
                let h_local = blk_h + r % p.rh;
                var x = 0.0;
                if (r < rows && l_local < p.tl && h_local < p.th && d0 + dd < p.dk) {
                    x = q[((b_local * p.tl + l_local) * p.th + h_local) * p.dk + d0 + dd];
                }
                st[r * DS + dd] = x;
            }
            for (var e = tid; e < BK * DC; e += 256u) {
                let kk = e / DC;
                let dd = e % DC;
                let tp = t_sh[kk];
                var x = 0.0;
                if (tp != NO_KEY && d0 + dd < p.dk) {
                    x = kElem(((b * p.t_cap + tp) * p.h_kv + hkv) * p.dk + d0 + dd);
                }
                st[R * DS + kk * DS + dd] = x;
            }
            workgroupBarrier();
            let kb = R * DS + 4u * tk * DS;
            for (var dd = 0u; dd < DC; dd += 1u) {
                let kv = vec4<f32>(st[kb + dd], st[kb + DS + dd], st[kb + 2u * DS + dd], st[kb + 3u * DS + dd]);
                s0 += st[r0 * DS + dd] * kv;
                s1 += st[r1 * DS + dd] * kv;
            }
        }

        // --- mask, scale, cap; per-row tile max ---
        var pm0 = FMIN;
        var pm1 = FMIN;
        for (var j = 0u; j < 4u; j += 1u) {
            let kk = 4u * tk + j;
            let tj = t0 + kk;
            let live = t_sh[kk] != NO_KEY;
            var x0 = FMIN;
            var x1 = FMIN;
            if (live && tj >= lo0 && tj < hi0) { x0 = capped(s0[j] * p.scale); }
            if (live && tj >= lo1 && tj < hi1) { x1 = capped(s1[j] * p.scale); }
            s0[j] = x0;
            s1[j] = x1;
            pm0 = max(pm0, x0);
            pm1 = max(pm1, x1);
        }
        workgroupBarrier(); // scores are done reading the q/k staging
        st[r0 * 16u + tk] = pm0;
        st[r1 * 16u + tk] = pm1;
        workgroupBarrier();

        var mn0 = m0;
        var mn1 = m1;
        for (var x = 0u; x < 16u; x += 1u) {
            mn0 = max(mn0, st[r0 * 16u + x]);
            mn1 = max(mn1, st[r1 * 16u + x]);
        }
        var ps0 = 0.0;
        var ps1 = 0.0;
        for (var j = 0u; j < 4u; j += 1u) {
            var e0 = 0.0;
            var e1 = 0.0;
            if (s0[j] > FMIN) { e0 = expApprox(s0[j] - mn0); }
            if (s1[j] > FMIN) { e1 = expApprox(s1[j] - mn1); }
            p_sh[r0 * BK + 4u * tk + j] = e0;
            p_sh[r1 * BK + 4u * tk + j] = e1;
            ps0 += e0;
            ps1 += e1;
        }
        st[R * 16u + r0 * 16u + tk] = ps0;
        st[R * 16u + r1 * 16u + tk] = ps1;
        workgroupBarrier();

        var lt0 = 0.0;
        var lt1 = 0.0;
        for (var x = 0u; x < 16u; x += 1u) {
            lt0 += st[R * 16u + r0 * 16u + x];
            lt1 += st[R * 16u + r1 * 16u + x];
        }
        let c0 = select(0.0, expApprox(m0 - mn0), m0 > FMIN);
        let c1 = select(0.0, expApprox(m1 - mn1), m1 > FMIN);
        l0 = l0 * c0 + lt0;
        l1 = l1 * c1 + lt1;
        m0 = mn0;
        m1 = mn1;
        for (var c = 0u; c < 4u; c += 1u) {
            a0[c] *= c0;
            a1[c] *= c1;
        }

        // --- out[rows, slice] += P[rows, tile] @ V[tile, slice] ---
        for (var k0 = 0u; k0 < BK; k0 += VK) {
            for (var c = 0u; c < DVS / VD; c += 1u) {
                workgroupBarrier(); // partials / the previous V chunk are consumed
                stageV(tid, b, hkv, k0, vs + c * VD);
                workgroupBarrier();
                var u0 = vec4<f32>(0.0);
                var u1 = vec4<f32>(0.0);
                for (var kk = 0u; kk < VK; kk += 1u) {
                    let v = vLanes(kk, tk);
                    u0 += p_sh[r0 * BK + k0 + kk] * v;
                    u1 += p_sh[r1 * BK + k0 + kk] * v;
                }
                a0[c] += u0;
                a1[c] += u1;
            }
        }
    }

    // --- normalize and write the slice of the rows this block owns ---
    let inv0 = select(0.0, 1.0 / l0, l0 > 0.0);
    let inv1 = select(0.0, 1.0 / l1, l1 > 0.0);
    for (var c = 0u; c < 4u; c += 1u) {
        writeLanes(r0, blk_l, blk_h, b_local, vs + c * VD + tk, a0[c] * inv0);
        writeLanes(r1, blk_l, blk_h, b_local, vs + c * VD + tk, a1[c] * inv1);
    }
}

fn capped(x: f32) -> f32 {
    if (p.soft_cap > 0.0) { return p.soft_cap * tanhApprox(x / p.soft_cap); }
    return x;
}

/// Row `r`'s dims d0 + 16 * (0..3), the ones past `dv` dropped.
fn writeLanes(r: u32, blk_l: u32, blk_h: u32, b_local: u32, d0: u32, v: vec4<f32>) {
    let l_local = blk_l + r / p.rh;
    let h_local = blk_h + r % p.rh;
    if (r >= p.rl * p.rh || l_local >= p.tl || h_local >= p.th) { return; }
    let ob = ((b_local * p.tl + l_local) * p.th + h_local) * p.dv;
    for (var j = 0u; j < 4u; j += 1u) {
        if (d0 + 16u * j < p.dv) { o[ob + d0 + 16u * j] = v[j]; }
    }
}
