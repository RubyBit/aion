// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Grouped-query attention:
//   out[b, l, hq, :] = softmax(scale * q[b, l, hq, :] @ K[b, t, hkv, :]^T) @ V[b, t, hkv, :]
// over the valid key range t in [lower, upper):
//   the key range is `window_keys(...)` on the query's position, intersected with
//   kv_lengths[b] and the ring-cache start.
// Optional logit soft cap: s = cap * tanh(s / cap).
//
// Layouts (all packed, single device buffer each):
//   q:   [B, tl, th, dk] f32
//   k/v: [B, T, H_kv, D] bound as array<u32> so f16 caches work WITHOUT the
//        shader-f16 extension: p.kv_f16 selects unpack2x16float (two elements
//        per word, D even) vs bitcast<f32>(one element per word).
//   query_positions: [B, tl] i32, kv_lengths: [B] i32.
//
// Logical->physical time mapping runs in-kernel: identity (none/growable
// policies — the exec pre-touches growth before recording) or ring modulo.
//
// WORK SHAPE — one 256-thread workgroup per (batch, row block, key segment):
//
//   * A row block is `p.rl` query rows x `p.rh` query heads, and the heads never
//     cross a GQA group (the host guarantees `rh` divides both `gqa` and the head
//     count). K/V depend only on (batch, kv head), so every row in the block
//     reads the SAME K/V rows: each K element is loaded once and fanned into all
//     `rl * rh` dot products, and each V element once into all their accumulators.
//     One row per workgroup — the obvious mapping — re-reads the whole cache per
//     row instead, which is what made this kernel memory-bound at a small fraction
//     of the device's bandwidth.
//
//   * The score phase is KEY-parallel (thread i scores key t0+i for every row in
//     the block); the V phase is DIM-parallel (thread i owns value dim
//     `i % dv_eff` of key group `i / dv_eff`). Threads therefore read consecutive
//     dims of one V row — coalesced — and all 256 stay busy even at dv=64, where
//     the previous dim-strided mapping left 3 of 4 threads idle in a loop that ran
//     one key at a time. Per-key-group partials are combined once at the end, not
//     per chunk: the online-softmax rescale is a per-row scalar, so it commutes
//     with the split.
//
//   * The per-row workgroup reductions (max, then sum) share ONE barrier chain
//     across the block, so a block of rows costs the same barriers as a single row.
//
// All four entry points share `attnCore` (the `_qf16` pair only stages q from the
// f16 view); `attn_split*` differs only in writing an
// unnormalized (acc, m, l) partial for `attention_merge.wgsl` to combine.

enable f16;

@group(0) @binding(0) var<storage, read>       q: array<f32>;
@group(0) @binding(1) var<storage, read>       kc: array<u32>;
@group(0) @binding(2) var<storage, read>       vc: array<u32>;
@group(0) @binding(3) var<storage, read>       pos: array<i32>;
@group(0) @binding(4) var<storage, read>       endi: array<i32>;
@group(0) @binding(5) var<storage, read_write> o: array<f32>;
// f16 query alias of binding 0, read only by the `*_qf16` entry points. The CPU
// attention takes any mix of f16/f32 for q/k/v (exec/attention.zig), so this is
// what lets the GPU accept the same graphs instead of demanding a Cast on q.
@group(0) @binding(0) var<storage, read>       qh: array<f16>;
@group(0) @binding(6) var<uniform>             p: Params;

struct Params {
    tl: u32, // q/out L count
    th: u32, // q/out head count
    dk: u32,
    dv: u32,
    t_cap: u32, // physical T of the caches
    h_kv: u32,
    gqa: u32, // query heads per kv head
    win_left: u32,
    win_right: u32,
    win_chunk: u32,
    ring: u32, // 0 = identity time map, 1 = ring
    ring_modulus: u32,
    kv_f16: u32,
    scale: f32,
    soft_cap: f32, // 0 = disabled
    segs: u32, // split-K segment count (1 for attn_row)
    has_pos: u32,
    has_lengths: u32,
    rl: u32, // rows per block along L
    rh: u32, // heads per block (within one GQA group)
};

const WG: u32 = 256u;
const MAX_DK: u32 = 512u; // `MAX_DK` in exec/attention.zig; sizes q_s
const FMIN: f32 = -3.4028235e38;

// A block's rows (at most 4, `MAX_ROWS` in exec/attention.zig) are the lanes of a
// vec4 everywhere: q, scores, probabilities, the softmax state and the
// accumulators. Rows held in a per-row array instead are indexed inside loops the
// shader compiler does not unroll, which puts the array in thread memory rather
// than registers; every access is then a memory round trip, and the V phase, which
// touches the accumulators once per key, ran several times slower for it.
//
// 13 KiB in all, inside the 16 KiB WebGPU workgroup-storage floor. Both workgroup
// reductions run destructively in p_sh: the max before the probabilities are
// written, the sum after the V phase has read them.
var<workgroup> q_s: array<vec4<f32>, MAX_DK>; // [dim] -> the block's rows
var<workgroup> p_sh: array<vec4<f32>, WG>; // [key] -> scores, then probabilities
var<workgroup> t_sh: array<u32, WG>; // physical time per scored key

fn expApprox(x_in: vec4<f32>) -> vec4<f32> {
    let xc = clamp(x_in, vec4(-80.0), vec4(80.0));
    let yy = xc * 1.4426950408889634;
    let nn = vec4<i32>(floor(yy + 0.5));
    let tt = (yy - vec4<f32>(nn)) * 0.6931471805599453;
    let e2 = bitcast<vec4<f32>>(vec4<u32>(nn + 127) << vec4(23u));
    return e2 * (1.0 + tt * (1.0 + tt * (0.5 + tt * (0.16666667 + tt * 0.041666668))));
}

fn tanhApprox(v: vec4<f32>) -> vec4<f32> {
    // tanh(v) = 2 * sigmoid(2v) - 1, the exponent taken on -|2v| so it cannot overflow.
    let e = expApprox(-2.0 * abs(v));
    let y = select(e / (1.0 + e), 1.0 / (1.0 + e), v >= vec4(0.0));
    return clamp(2.0 * y - 1.0, vec4(-1.0), vec4(1.0));
}

/// Tree-reduce `p_sh[0..WG]` into `p_sh[0]`, every row at once. Destructive. The
/// caller has written its lane and issued a barrier; the result is readable after.
fn reduceMax(lidx: u32) {
    for (var s = WG / 2u; s > 0u; s /= 2u) {
        if (lidx < s) { p_sh[lidx] = max(p_sh[lidx], p_sh[lidx + s]); }
        workgroupBarrier();
    }
}

fn reduceSum(lidx: u32) {
    for (var s = WG / 2u; s > 0u; s /= 2u) {
        if (lidx < s) { p_sh[lidx] += p_sh[lidx + s]; }
        workgroupBarrier();
    }
}

// --- q staging -------------------------------------------------------------
//
// The block's q rows are copied into `q_s` (zero for rows past the block), and
// this is the ONLY place q is read. It lives in the entry points rather than in
// `attnCore` for a validation reason: a resource interface is per entry point, so
// `q` (f32) and `qh` (f16) may share binding 0 only while no single entry point
// names both. A runtime dtype flag branching inside the shared core names both and
// fails shader validation ("Entry point attn_row at Compute is invalid") — hence
// two staging functions and two entry points per dispatch shape rather than one
// flag.
//
// `qFlatIndex` is the shared half, so the addressing exists once.
fn qFlatIndex(b_local: u32, blk_l: u32, blk_h: u32, r: u32, i: u32) -> i32 {
    let l_local = blk_l + r / p.rh;
    let h_local = blk_h + r % p.rh;
    if (r >= p.rl * p.rh || l_local >= p.tl || h_local >= p.th) { return -1; }
    return i32(((b_local * p.tl + l_local) * p.th + h_local) * p.dk + i);
}

fn qF32(b_local: u32, blk_l: u32, blk_h: u32, r: u32, i: u32) -> f32 {
    let qi = qFlatIndex(b_local, blk_l, blk_h, r, i);
    if (qi < 0) { return 0.0; }
    return q[u32(qi)];
}

fn qF16(b_local: u32, blk_l: u32, blk_h: u32, r: u32, i: u32) -> f32 {
    let qi = qFlatIndex(b_local, blk_l, blk_h, r, i);
    if (qi < 0) { return 0.0; }
    return f32(qh[u32(qi)]);
}

fn stageQF32(b_local: u32, blk_l: u32, blk_h: u32, lidx: u32) {
    for (var i = lidx; i < p.dk; i += WG) {
        q_s[i] = vec4(qF32(b_local, blk_l, blk_h, 0u, i), qF32(b_local, blk_l, blk_h, 1u, i),
                      qF32(b_local, blk_l, blk_h, 2u, i), qF32(b_local, blk_l, blk_h, 3u, i));
    }
}

fn stageQF16(b_local: u32, blk_l: u32, blk_h: u32, lidx: u32) {
    for (var i = lidx; i < p.dk; i += WG) {
        q_s[i] = vec4(qF16(b_local, blk_l, blk_h, 0u, i), qF16(b_local, blk_l, blk_h, 1u, i),
                      qF16(b_local, blk_l, blk_h, 2u, i), qF16(b_local, blk_l, blk_h, 3u, i));
    }
}

/// Value dim `d` of the V row starting at element `vb`.
fn vAt(vb: u32, d: u32) -> f32 {
    if (p.kv_f16 != 0u) {
        let both = unpack2x16float(vc[vb / 2u + d / 2u]);
        return select(both.x, both.y, (d & 1u) != 0u);
    }
    return bitcast<f32>(vc[vb + d]);
}

/// The block's rows dotted with key row `kb`: one K read, fanned into every row.
fn scoreKey(kb: u32) -> vec4<f32> {
    var dots = vec4(0.0);
    if (p.kv_f16 != 0u) {
        // D even -> the row starts word-aligned; walk element pairs.
        //
        // Unrolled to 8 words = 32 bytes = one full L1 sector per thread per
        // iteration. This phase is KEY-parallel, so a thread walks its OWN key's
        // row and consecutive threads sit dk*2 bytes apart (512 for Gemma): every
        // load instruction touches 32 cache lines whatever its width, so consuming
        // a whole sector per thread is what stops 7/8 of each fetch being wasted.
        // Written out rather than looped: the compiler does not unroll loops, and
        // an inner 8-word loop measured twice as slow on 512-wide heads.
        let wb = kb / 2u;
        var i = 0u;
        for (; i + 16u <= p.dk; i += 16u) {
            let w = wb + i / 2u;
            let a0 = unpack2x16float(kc[w]);
            let a1 = unpack2x16float(kc[w + 1u]);
            let a2 = unpack2x16float(kc[w + 2u]);
            let a3 = unpack2x16float(kc[w + 3u]);
            let a4 = unpack2x16float(kc[w + 4u]);
            let a5 = unpack2x16float(kc[w + 5u]);
            let a6 = unpack2x16float(kc[w + 6u]);
            let a7 = unpack2x16float(kc[w + 7u]);
            dots += q_s[i] * a0.x + q_s[i + 1u] * a0.y + q_s[i + 2u] * a1.x + q_s[i + 3u] * a1.y
                  + q_s[i + 4u] * a2.x + q_s[i + 5u] * a2.y + q_s[i + 6u] * a3.x + q_s[i + 7u] * a3.y
                  + q_s[i + 8u] * a4.x + q_s[i + 9u] * a4.y + q_s[i + 10u] * a5.x + q_s[i + 11u] * a5.y
                  + q_s[i + 12u] * a6.x + q_s[i + 13u] * a6.y + q_s[i + 14u] * a7.x + q_s[i + 15u] * a7.y;
        }
        for (; i < p.dk; i += 2u) {
            let a = unpack2x16float(kc[wb + i / 2u]);
            dots += q_s[i] * a.x + q_s[i + 1u] * a.y;
        }
    } else {
        for (var i = 0u; i < p.dk; i += 1u) { dots += q_s[i] * bitcast<f32>(kc[kb + i]); }
    }
    return dots;
}

fn attnCore(b_local: u32, seg_local: u32, blk_l: u32, blk_h: u32, lidx: u32, partial: bool) {
    let b = b_local;
    // Every row in the block shares this kv head: `rh` divides `gqa` and the block's
    // head base is a multiple of `rh`, so the block never straddles a group.
    let hkv = blk_h / p.gqa;
    // The host guarantees rl * rh <= 4 (`MAX_ROWS` in exec/attention.zig): the grid
    // is sized from it, so a larger block would leave outputs unwritten.
    let rows = p.rl * p.rh;

    var valid_end = p.t_cap;
    if (p.has_lengths != 0u) { valid_end = u32(endi[b]); }

    // --- per-row key window, plus the block's union for the segment split ---
    // A lane past the block keeps the empty window [0, 0).
    var lo = vec4(0u);
    var hi = vec4(0u);
    var span_lo = 0xffffffffu;
    var span_hi = 0u;
    for (var r = 0u; r < rows; r += 1u) {
        let l_local = blk_l + r / p.rh;
        let h_local = blk_h + r % p.rh;
        if (l_local >= p.tl || h_local >= p.th) { continue; }

        var q_pos = l_local;
        if (p.has_pos != 0u) { q_pos = u32(pos[b_local * p.tl + l_local]); }

        let w = window_keys(p.win_left, p.win_right, p.win_chunk, q_pos, valid_end);
        var lower = w.x;
        if (p.ring != 0u && valid_end > p.ring_modulus) { lower = max(lower, valid_end - p.ring_modulus); }
        if (w.y <= lower) { continue; }

        lo[r] = lower;
        hi[r] = w.y;
        span_lo = min(span_lo, lower);
        span_hi = max(span_hi, w.y);
    }

    // --- this workgroup's slice of the block's key range ---
    var t_start = 0u;
    var t_end = 0u;
    if (span_hi > span_lo) {
        let seg_len = (span_hi - span_lo + p.segs - 1u) / p.segs;
        t_start = span_lo + seg_local * seg_len;
        t_end = min(t_start + seg_len, span_hi);
    }

    // q was staged into `q_s` by the entry point (see `stageQF32` / `stageQF16`)
    // and barriered there, so `attnCore` itself never names a q binding — which is
    // what lets the f32 and f16 queries alias binding 0.

    // V-phase thread mapping: `dv_eff` consecutive dims per key group, key groups
    // rounded down to a power of two so the tail reduction is a clean tree. A
    // thread owns dim d0 and, when dv > WG, the dims WG, 2 WG and 3 WG past it
    // (dv <= 4 * WG, `MAX_DV` in exec/attention.zig).
    let dv_eff = min(p.dv, WG);
    let kgc = 1u << firstLeadingBit(WG / dv_eff);
    let d0 = lidx % dv_eff;
    let kg = lidx / dv_eff;
    let has1 = d0 + WG < p.dv;
    let has2 = d0 + 2u * WG < p.dv;
    let has3 = d0 + 3u * WG < p.dv;

    var acc0 = vec4(0.0);
    var acc1 = vec4(0.0);
    var acc2 = vec4(0.0);
    var acc3 = vec4(0.0);
    var m_state = vec4(FMIN);
    var l_state = vec4(0.0);

    for (var t0 = t_start; t0 < t_end; t0 += WG) {
        let tj = t0 + lidx;
        var t_phys = 0u;
        var okk = tj < t_end;
        if (okk) {
            t_phys = tj;
            if (p.ring != 0u) { t_phys = tj % p.ring_modulus; }
            okk = t_phys < p.t_cap;
        }

        // --- scores ---
        let valid = vec4(okk) & (vec4(tj) >= lo) & (vec4(tj) < hi);
        var sv = vec4(FMIN);
        if (okk) {
            var sc = scoreKey(((b * p.t_cap + t_phys) * p.h_kv + hkv) * p.dk) * p.scale;
            if (p.soft_cap > 0.0) { sc = p.soft_cap * tanhApprox(sc / p.soft_cap); }
            sv = select(vec4(FMIN), sc, valid);
        }

        // --- running max (destructive reduce over the staged scores) ---
        p_sh[lidx] = sv;
        workgroupBarrier();
        reduceMax(lidx);
        let m_new = max(m_state, p_sh[0]);
        let resc = expApprox(m_state - m_new);
        m_state = m_new;
        acc0 *= resc;
        acc1 *= resc;
        acc2 *= resc;
        acc3 *= resc;
        workgroupBarrier();

        // --- probabilities ---
        p_sh[lidx] = select(vec4(0.0), expApprox(sv - m_new), valid);
        t_sh[lidx] = t_phys;
        workgroupBarrier();

        // --- V: each element loaded once, reused across the block's rows ---
        let cnt = min(WG, t_end - t0);
        if (kg < kgc) {
            for (var jj = kg; jj < cnt; jj += kgc) {
                let pj = p_sh[jj];
                let vb = ((b * p.t_cap + t_sh[jj]) * p.h_kv + hkv) * p.dv;
                acc0 += pj * vAt(vb, d0);
                if (has1) { acc1 += pj * vAt(vb, d0 + WG); }
                if (has2) { acc2 += pj * vAt(vb, d0 + 2u * WG); }
                if (has3) { acc3 += pj * vAt(vb, d0 + 3u * WG); }
            }
        }
        workgroupBarrier();

        // --- running sum: destructive, so it must follow the V phase ---
        reduceSum(lidx);
        l_state = l_state * resc + p_sh[0];
        workgroupBarrier();
    }

    // --- combine the key groups' partials (dv < WG, so only acc0 is in use) ---
    if (kgc > 1u) {
        p_sh[lidx] = acc0;
        workgroupBarrier();
        for (var half = kgc / 2u; half > 0u; half /= 2u) {
            if (kg < half) { p_sh[lidx] += p_sh[lidx + half * dv_eff]; }
            workgroupBarrier();
        }
        acc0 = p_sh[d0];
    }

    // --- write ---
    if (kg != 0u) { return; }
    // Empty window (lower >= upper) leaves l_state == 0 -> write zeros, same as the
    // CPU executor's memset.
    let inv = select(vec4(0.0), 1.0 / l_state, l_state > vec4(0.0));
    for (var r = 0u; r < rows; r += 1u) {
        let l_local = blk_l + r / p.rh;
        let h_local = blk_h + r % p.rh;
        if (l_local >= p.tl || h_local >= p.th) { continue; }
        let row = (b_local * p.tl + l_local) * p.th + h_local;

        var base = row * p.dv;
        var scale = inv[r];
        if (partial) {
            base = (row * p.segs + seg_local) * (p.dv + 2u);
            scale = 1.0;
            if (lidx == 0u) {
                o[base + p.dv] = m_state[r];
                o[base + p.dv + 1u] = l_state[r];
            }
        }
        o[base + d0] = acc0[r] * scale;
        if (has1) { o[base + d0 + WG] = acc1[r] * scale; }
        if (has2) { o[base + d0 + 2u * WG] = acc2[r] * scale; }
        if (has3) { o[base + d0 + 3u * WG] = acc3[r] * scale; }
    }
}

@compute @workgroup_size(256)
fn attn_row(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    stageQF32(wid.z, wid.y * p.rl, wid.x * p.rh, lidx);
    workgroupBarrier();
    attnCore(wid.z, 0u, wid.y * p.rl, wid.x * p.rh, lidx, false);
}

@compute @workgroup_size(256)
fn attn_row_qf16(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    stageQF16(wid.z, wid.y * p.rl, wid.x * p.rh, lidx);
    workgroupBarrier();
    attnCore(wid.z, 0u, wid.y * p.rl, wid.x * p.rh, lidx, false);
}

// ---- split-K (flash-decoding) ------------------------------------------------
//
// Decode dispatches only a handful of blocks above — a few rows scanning a
// thousands-of-tokens cache leaves the GPU idle. The split path carves each row
// block's key range into `p.segs` segments (grid.z = B * segs), each writing an
// UNNORMALIZED partial — dv accumulator values plus (m, l) — into `o` (bound to the
// backend scratch). `attn_merge` (attention_merge.wgsl) then log-sum-exp-combines
// the segments per row. Entry stride is dv + 2 floats at
//   entry = ((b_local * tl + l) * th + h) * segs + seg.
@compute @workgroup_size(256)
fn attn_split(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    stageQF32(wid.z / p.segs, wid.y * p.rl, wid.x * p.rh, lidx);
    workgroupBarrier();
    attnCore(wid.z / p.segs, wid.z % p.segs, wid.y * p.rl, wid.x * p.rh, lidx, true);
}

@compute @workgroup_size(256)
fn attn_split_qf16(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    stageQF16(wid.z / p.segs, wid.y * p.rl, wid.x * p.rh, lidx);
    workgroupBarrier();
    attnCore(wid.z / p.segs, wid.z % p.segs, wid.y * p.rl, wid.x * p.rh, lidx, true);
}

