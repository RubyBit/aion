// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Grouped-query attention:
//   out[b, l, hq, :] = softmax(scale * q[b, l, hq, :] @ K[b, t, hkv, :]^T) @ V[b, t, hkv, :]
// over the valid key range t in [lower, upper):
//   upper = kv_lengths[b], clamped to query_positions[b, l] + 1 when causal;
//   lower = max(ring window start, sliding-window start).
// Optional logit soft cap: s = cap * tanh(s / cap).
//
// Layouts (all packed, single device buffer each):
//   q:   [tb, tl, th, dk] f32 — ONE out/q tile per dispatch (p.base_b / p.base_h /
//        p.base_l give the tile's global offsets)
//   k/v: [B, T, H_kv, D] bound as array<u32> so f16 caches work WITHOUT the
//        shader-f16 extension: p.kv_f16 selects unpack2x16float (two elements
//        per word, D even) vs bitcast<f32>(one element per word).
//   query_positions: [tb, tl] i32 (tile-local), kv_lengths: [B] i32 (global).
//
// Logical->physical time mapping runs in-kernel: identity (none/growable
// policies — the exec pre-touches growth before recording) or ring modulo.
//
// WORK SHAPE — one 256-thread workgroup per (batch, row block, key segment):
//
//   * A row block is `p.rl` query rows x `p.rh` query heads, and the heads never
//     cross a GQA group (the host guarantees `rh` divides both `gqa` and the tile's
//     head count). K/V depend only on (batch, kv head), so every row in the block
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
// Both entry points share `attnCore`; `attn_split` differs only in writing an
// unnormalized (acc, m, l) partial for `attention_merge.wgsl` to combine.

@group(0) @binding(0) var<storage, read>       q: array<f32>;
@group(0) @binding(1) var<storage, read>       kc: array<u32>;
@group(0) @binding(2) var<storage, read>       vc: array<u32>;
@group(0) @binding(3) var<storage, read>       pos: array<i32>;
@group(0) @binding(4) var<storage, read>       endi: array<i32>;
@group(0) @binding(5) var<storage, read_write> o: array<f32>;
@group(0) @binding(6) var<uniform>             p: Params;

struct Params {
    base_b: u32,
    base_h: u32,
    tl: u32, // q/out tile-local L count
    th: u32, // q/out tile-local head count
    dk: u32,
    dv: u32,
    t_cap: u32, // physical T of the caches
    h_kv: u32,
    gqa: u32, // query heads per kv head
    causal: u32,
    sliding: u32, // 0 = global attention
    ring: u32, // 0 = identity time map, 1 = ring
    ring_window: u32,
    kv_f16: u32,
    scale: f32,
    soft_cap: f32, // 0 = disabled
    segs: u32, // split-K segment count (1 for attn_row)
    base_l: u32,
    has_pos: u32,
    has_lengths: u32,
    rl: u32, // rows per block along L
    rh: u32, // heads per block (within one GQA group)
};

const WG: u32 = 256u;
const RMAX: u32 = 4u; // max rows per block; bounds q_s / p_sh and the register arrays
const ACC: u32 = 4u; // dims per thread when dv > WG (dv <= ACC * WG)
const FMIN: f32 = -3.4028235e38;

// 9 KiB total, well inside the 16 KiB WebGPU workgroup-storage floor. There is no
// separate reduction scratch: both workgroup reductions run DESTRUCTIVELY in p_sh —
// the max before the probabilities are written, the sum after the V phase has
// consumed them.
// The 1024 here is `Q_STAGE_FLOATS` in exec/attention.zig, which enforces
// `rows * dk <= 1024`; WGSL array sizes must be literals, so the two are hand-kept.
var<workgroup> q_s: array<f32, 1024>; // [row][dk]
var<workgroup> p_sh: array<f32, 1024>; // [row][WG] scores -> probabilities
var<workgroup> t_sh: array<u32, 256>; // physical time per scored key

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

/// Tree-reduce `p_sh[r][0..WG]` into `p_sh[r][0]` for all `rows` in ONE barrier
/// chain, so a block of rows costs the same barriers as a single row. Destructive.
/// Caller has written its lane and issued a barrier; the result is readable after.
fn reduceMaxRows(lidx: u32, rows: u32) {
    var s = WG / 2u;
    while (s > 0u) {
        if (lidx < s) {
            for (var r = 0u; r < rows; r += 1u) {
                p_sh[r * WG + lidx] = max(p_sh[r * WG + lidx], p_sh[r * WG + lidx + s]);
            }
        }
        workgroupBarrier();
        s = s / 2u;
    }
}

fn reduceSumRows(lidx: u32, rows: u32) {
    var s = WG / 2u;
    while (s > 0u) {
        if (lidx < s) {
            for (var r = 0u; r < rows; r += 1u) {
                p_sh[r * WG + lidx] = p_sh[r * WG + lidx] + p_sh[r * WG + lidx + s];
            }
        }
        workgroupBarrier();
        s = s / 2u;
    }
}

fn attnCore(b_local: u32, seg: u32, blk_l: u32, blk_h: u32, lidx: u32, partial: bool) {
    let b = p.base_b + b_local;
    // Every row in the block shares this kv head: `rh` divides `gqa` and the tile's
    // head base is a multiple of `rh`, so the block never straddles a group.
    let hkv = (p.base_h + blk_h) / p.gqa;
    // NOT clamped to RMAX on purpose: the grid is sized from rl*rh, so silently
    // dropping rows here would leave outputs unwritten. The host guarantees
    // rl * rh <= RMAX (see MAX_ROWS in exec/attention.zig).
    let rows = p.rl * p.rh;

    var valid_end = p.t_cap;
    if (p.has_lengths != 0u) { valid_end = u32(endi[b]); }

    // --- per-row key window, plus the block's union for the segment split ---
    var lo_r: array<u32, 4>;
    var hi_r: array<u32, 4>;
    var span_lo = 0xffffffffu;
    var span_hi = 0u;
    for (var r = 0u; r < rows; r += 1u) {
        lo_r[r] = 0u;
        hi_r[r] = 0u;
        let l_local = blk_l + r / p.rh;
        let h_local = blk_h + r % p.rh;
        if (l_local >= p.tl || h_local >= p.th) { continue; }

        var q_pos = p.base_l + l_local;
        if (p.has_pos != 0u) { q_pos = u32(pos[b_local * p.tl + l_local]); }

        var upper = valid_end;
        if (p.causal != 0u) { upper = min(upper, q_pos + 1u); }
        var lower = 0u;
        if (p.ring != 0u && valid_end > p.ring_window) { lower = valid_end - p.ring_window; }
        if (p.sliding > 0u && q_pos + 1u > p.sliding) { lower = max(lower, q_pos + 1u - p.sliding); }
        if (upper <= lower) { continue; }

        lo_r[r] = lower;
        hi_r[r] = upper;
        span_lo = min(span_lo, lower);
        span_hi = max(span_hi, upper);
    }

    // --- this workgroup's slice of the block's key range ---
    var t_start = 0u;
    var t_end = 0u;
    if (span_hi > span_lo) {
        let seg_len = (span_hi - span_lo + p.segs - 1u) / p.segs;
        t_start = span_lo + seg * seg_len;
        t_end = min(t_start + seg_len, span_hi);
    }

    // --- stage the block's q rows (zero-filled for out-of-range rows) ---
    let rows_dk = rows * p.dk;
    for (var i = lidx; i < rows_dk; i += WG) {
        let r = i / p.dk;
        let l_local = blk_l + r / p.rh;
        let h_local = blk_h + r % p.rh;
        var val = 0.0;
        if (l_local < p.tl && h_local < p.th) {
            val = q[((b_local * p.tl + l_local) * p.th + h_local) * p.dk + (i % p.dk)];
        }
        q_s[i] = val;
    }
    workgroupBarrier();

    // V-phase thread mapping: `dv_eff` consecutive dims per key group, key groups
    // rounded down to a power of two so the tail reduction is a clean tree.
    let dv_eff = min(p.dv, WG);
    let kgc = 1u << firstLeadingBit(WG / dv_eff);
    let d0 = lidx % dv_eff;
    let kg = lidx / dv_eff;
    let dsteps = (p.dv + dv_eff - 1u) / dv_eff;

    var acc: array<f32, 16>; // [row][dim step], RMAX * ACC
    for (var i = 0u; i < 16u; i += 1u) { acc[i] = 0.0; }
    var m_state: array<f32, 4>;
    var l_state: array<f32, 4>;
    for (var r = 0u; r < RMAX; r += 1u) {
        m_state[r] = FMIN;
        l_state[r] = 0.0;
    }

    for (var t0 = t_start; t0 < t_end; t0 += WG) {
        let tj = t0 + lidx;
        var t_phys = 0u;
        var okk = tj < t_end;
        if (okk) {
            t_phys = tj;
            if (p.ring != 0u) { t_phys = tj % p.ring_window; }
            okk = t_phys < p.t_cap;
        }

        // --- scores: one K row read, fanned into every row of the block ---
        var sv: array<f32, 4>;
        var vr: array<bool, 4>;
        for (var r = 0u; r < rows; r += 1u) {
            sv[r] = FMIN;
            vr[r] = false;
        }
        if (okk) {
            let kb = ((b * p.t_cap + t_phys) * p.h_kv + hkv) * p.dk;
            var dots: array<f32, 4>;
            for (var r = 0u; r < rows; r += 1u) { dots[r] = 0.0; }
            if (p.kv_f16 != 0u) {
                // D even -> the row starts word-aligned; walk element pairs.
                let wb = kb / 2u;
                for (var i = 0u; i < p.dk; i += 2u) {
                    let both = unpack2x16float(kc[wb + i / 2u]);
                    for (var r = 0u; r < rows; r += 1u) {
                        dots[r] += q_s[r * p.dk + i] * both.x + q_s[r * p.dk + i + 1u] * both.y;
                    }
                }
            } else {
                for (var i = 0u; i < p.dk; i += 1u) {
                    let kv = bitcast<f32>(kc[kb + i]);
                    for (var r = 0u; r < rows; r += 1u) { dots[r] += q_s[r * p.dk + i] * kv; }
                }
            }
            for (var r = 0u; r < rows; r += 1u) {
                if (tj >= lo_r[r] && tj < hi_r[r]) {
                    var sc = dots[r] * p.scale;
                    if (p.soft_cap > 0.0) { sc = p.soft_cap * tanhApprox(sc / p.soft_cap); }
                    sv[r] = sc;
                    vr[r] = true;
                }
            }
        }

        // --- running max (destructive reduce over the staged scores) ---
        for (var r = 0u; r < rows; r += 1u) { p_sh[r * WG + lidx] = sv[r]; }
        workgroupBarrier();
        reduceMaxRows(lidx, rows);
        var m_new: array<f32, 4>;
        var resc: array<f32, 4>;
        for (var r = 0u; r < rows; r += 1u) {
            m_new[r] = max(m_state[r], p_sh[r * WG]);
            resc[r] = expApprox(m_state[r] - m_new[r]);
            m_state[r] = m_new[r];
            for (var i = 0u; i < dsteps; i += 1u) { acc[r * ACC + i] = acc[r * ACC + i] * resc[r]; }
        }
        workgroupBarrier();

        // --- probabilities ---
        for (var r = 0u; r < rows; r += 1u) {
            var pr = 0.0;
            if (vr[r]) { pr = expApprox(sv[r] - m_new[r]); }
            p_sh[r * WG + lidx] = pr;
        }
        t_sh[lidx] = t_phys;
        workgroupBarrier();

        // --- V: each element loaded once, reused across the block's rows ---
        let cnt = min(WG, t_end - t0);
        if (kg < kgc) {
            for (var jj = kg; jj < cnt; jj += kgc) {
                let vb = ((b * p.t_cap + t_sh[jj]) * p.h_kv + hkv) * p.dv;
                for (var i = 0u; i < dsteps; i += 1u) {
                    let d = d0 + i * dv_eff;
                    if (d < p.dv) {
                        var ve = 0.0;
                        if (p.kv_f16 != 0u) {
                            let both = unpack2x16float(vc[vb / 2u + d / 2u]);
                            ve = select(both.x, both.y, (d & 1u) != 0u);
                        } else {
                            ve = bitcast<f32>(vc[vb + d]);
                        }
                        for (var r = 0u; r < rows; r += 1u) {
                            acc[r * ACC + i] += p_sh[r * WG + jj] * ve;
                        }
                    }
                }
            }
        }
        workgroupBarrier();

        // --- running sum: destructive, so it must follow the V phase ---
        reduceSumRows(lidx, rows);
        for (var r = 0u; r < rows; r += 1u) {
            l_state[r] = l_state[r] * resc[r] + p_sh[r * WG];
        }
        workgroupBarrier();
    }

    // --- combine the key groups' partials (dv <= WG => dsteps == 1) ---
    if (kgc > 1u) {
        for (var r = 0u; r < rows; r += 1u) { p_sh[r * WG + lidx] = acc[r * ACC]; }
        workgroupBarrier();
        var half = kgc / 2u;
        while (half > 0u) {
            if (kg < half) {
                for (var r = 0u; r < rows; r += 1u) {
                    p_sh[r * WG + lidx] += p_sh[r * WG + lidx + half * dv_eff];
                }
            }
            workgroupBarrier();
            half = half / 2u;
        }
        for (var r = 0u; r < rows; r += 1u) { acc[r * ACC] = p_sh[r * WG + d0]; }
    }

    // --- write ---
    for (var r = 0u; r < rows; r += 1u) {
        let l_local = blk_l + r / p.rh;
        let h_local = blk_h + r % p.rh;
        if (l_local >= p.tl || h_local >= p.th) { continue; }
        let row = (b_local * p.tl + l_local) * p.th + h_local;

        if (partial) {
            let e = (row * p.segs + seg) * (p.dv + 2u);
            if (kg == 0u) {
                for (var i = 0u; i < dsteps; i += 1u) {
                    let d = d0 + i * dv_eff;
                    if (d < p.dv) { o[e + d] = acc[r * ACC + i]; }
                }
            }
            if (lidx == 0u) {
                o[e + p.dv] = m_state[r];
                o[e + p.dv + 1u] = l_state[r];
            }
        } else {
            // Empty window (lower >= upper) leaves l_state == 0 -> write zeros,
            // same as the CPU executor's memset.
            var inv = 0.0;
            if (l_state[r] > 0.0) { inv = 1.0 / l_state[r]; }
            let o_base = row * p.dv;
            if (kg == 0u) {
                for (var i = 0u; i < dsteps; i += 1u) {
                    let d = d0 + i * dv_eff;
                    if (d < p.dv) { o[o_base + d] = acc[r * ACC + i] * inv; }
                }
            }
        }
    }
}

@compute @workgroup_size(256)
fn attn_row(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    attnCore(wid.z, 0u, wid.y * p.rl, wid.x * p.rh, lidx, false);
}

// ---- split-K (flash-decoding) ------------------------------------------------
//
// Decode dispatches only a handful of blocks above — a few rows scanning a
// thousands-of-tokens cache leaves the GPU idle. The split path carves each row
// block's key range into `p.segs` segments (grid.z = tb * segs), each writing an
// UNNORMALIZED partial — dv accumulator values plus (m, l) — into `o` (bound to the
// backend scratch). `attn_merge` (attention_merge.wgsl) then log-sum-exp-combines
// the segments per row. Entry stride is dv + 2 floats at
//   entry = ((b_local * tl + l) * th + h) * segs + seg.
@compute @workgroup_size(256)
fn attn_split(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    attnCore(wid.z / p.segs, wid.z % p.segs, wid.y * p.rl, wid.x * p.rh, lidx, true);
}

