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
    base_b: u32,
    base_h: u32,
    tl: u32, // q/out tile-local L count
    th: u32, // q/out tile-local head count
    dk: u32,
    dv: u32,
    t_cap: u32, // physical T of the caches
    h_kv: u32,
    gqa: u32, // query heads per kv head
    win_left: u32,
    win_right: u32,
    win_chunk: u32,
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
    // A cache too large for one storage binding is split along time, and k/v bind
    // ONE tile: `[kv_t0, kv_t0 + kv_tile_t)` of the physical axis. Each tile is a
    // separate dispatch writing its own partial slots (`seg_base` .. +`segs_local`),
    // which `attn_merge` combines exactly as it combines ordinary split-K segments.
    // A single-tile cache is `kv_t0 = 0`, `kv_tile_t = t_cap`.
    kv_t0: u32,
    kv_tile_t: u32,
    seg_base: u32,
    segs_local: u32,
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

// --- q staging -------------------------------------------------------------
//
// The block's q rows are copied into the f32 shared array `q_s` (zero-filled for
// out-of-range rows), and this is the ONLY place q is read. It lives in the entry
// points rather than in `attnCore` for a validation reason: a resource interface
// is per entry point, so `q` (f32) and `qh` (f16) may share binding 0 only while
// no single entry point names both. A runtime dtype flag branching inside the
// shared core names both and fails shader validation ("Entry point attn_row at
// Compute is invalid") — hence two staging functions and two entry points per
// dispatch shape rather than one flag.
//
// `flatIndex` is the shared half, so the addressing exists once.
fn qFlatIndex(b_local: u32, blk_l: u32, blk_h: u32, i: u32) -> i32 {
    let r = i / p.dk;
    let l_local = blk_l + r / p.rh;
    let h_local = blk_h + r % p.rh;
    if (l_local >= p.tl || h_local >= p.th) { return -1; }
    return i32(((b_local * p.tl + l_local) * p.th + h_local) * p.dk + (i % p.dk));
}

fn stageQF32(b_local: u32, blk_l: u32, blk_h: u32, lidx: u32) {
    let rows_dk = p.rl * p.rh * p.dk;
    for (var i = lidx; i < rows_dk; i += WG) {
        let qi = qFlatIndex(b_local, blk_l, blk_h, i);
        if (qi < 0) { q_s[i] = 0.0; } else { q_s[i] = q[u32(qi)]; }
    }
}

fn stageQF16(b_local: u32, blk_l: u32, blk_h: u32, lidx: u32) {
    let rows_dk = p.rl * p.rh * p.dk;
    for (var i = lidx; i < rows_dk; i += WG) {
        let qi = qFlatIndex(b_local, blk_l, blk_h, i);
        if (qi < 0) { q_s[i] = 0.0; } else { q_s[i] = f32(qh[u32(qi)]); }
    }
}

fn attnCore(b_local: u32, seg_local: u32, blk_l: u32, blk_h: u32, lidx: u32, partial: bool) {
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

        let w = window_keys(p.win_left, p.win_right, p.win_chunk, q_pos, valid_end);
        let upper = w.y;
        var lower = w.x;
        if (p.ring != 0u && valid_end > p.ring_window) { lower = max(lower, valid_end - p.ring_window); }
        if (upper <= lower) { continue; }

        lo_r[r] = lower;
        hi_r[r] = upper;
        span_lo = min(span_lo, lower);
        span_hi = max(span_hi, upper);
    }

    // --- this workgroup's slice of the block's key range ---
    //
    // Under the identity time map a tile owns a CONTIGUOUS logical range, so the
    // scan is narrowed to it. A ring map scatters logical times across tiles, so
    // there the full span is scanned and the membership test below filters —
    // correct, at the cost of re-scoring keys other tiles own.
    var lo = span_lo;
    var hi = span_hi;
    if (p.ring == 0u) {
        lo = max(span_lo, p.kv_t0);
        hi = min(span_hi, p.kv_t0 + p.kv_tile_t);
    }
    var t_start = 0u;
    var t_end = 0u;
    if (hi > lo) {
        let seg_len = (hi - lo + p.segs_local - 1u) / p.segs_local;
        t_start = lo + seg_local * seg_len;
        t_end = min(t_start + seg_len, hi);
    }

    // q was staged into `q_s` by the entry point (see `stageQF32` / `stageQF16`)
    // and barriered there, so `attnCore` itself never names a q binding — which is
    // what lets the f32 and f16 queries alias binding 0.

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
            // Bound by the cache AND by the tile actually bound here.
            okk = t_phys < p.t_cap && t_phys >= p.kv_t0 && t_phys - p.kv_t0 < p.kv_tile_t;
        }

        // --- scores: one K row read, fanned into every row of the block ---
        var sv: array<f32, 4>;
        var vr: array<bool, 4>;
        for (var r = 0u; r < rows; r += 1u) {
            sv[r] = FMIN;
            vr[r] = false;
        }
        if (okk) {
            let kb = ((b * p.kv_tile_t + (t_phys - p.kv_t0)) * p.h_kv + hkv) * p.dk;
            var dots: array<f32, 4>;
            for (var r = 0u; r < rows; r += 1u) { dots[r] = 0.0; }
            if (p.kv_f16 != 0u) {
                // D even -> the row starts word-aligned; walk element pairs.
                let wb = kb / 2u;
                var i = 0u;
                // Unrolled to 8 words = 32 bytes = one full L1 sector per thread per
                // iteration. This phase is KEY-parallel, so a thread walks its OWN key's
                // row and consecutive threads sit dk*2 bytes apart (512 for Gemma): the
                // access is inherently uncoalesced and what costs is the NUMBER of
                // memory transactions. Every load instruction touches 32 cache lines
                // whatever its width, so consuming a whole sector per thread is what
                // stops 7/8 of each fetch being wasted.
                //
                // Measured: 1 word -> 4 words -> 8 words took local decode attention
                // 44.3 -> 36.4 -> 31.9 us/op, i.e. attention 2.52 -> 1.42 ms/token.
                // Going wider does not help (32 bytes is the sector), and making the
                // phase cooperative/coalesced instead gained only 0.05 ms more for a
                // large rewrite -- past this point the kernel is launch- and
                // barrier-bound, not memory-bound. Reducing BYTES does nothing either:
                // sharing the K read across all 8 query heads (rh=4) cuts traffic 8x and
                // measures SLOWER, because it costs workgroups.
                let d16 = p.dk & ~15u;
                for (; i < d16; i += 16u) {
                    let w = wb + i / 2u;
                    let a0 = unpack2x16float(kc[w]);
                    let a1 = unpack2x16float(kc[w + 1u]);
                    let a2 = unpack2x16float(kc[w + 2u]);
                    let a3 = unpack2x16float(kc[w + 3u]);
                    let a4 = unpack2x16float(kc[w + 4u]);
                    let a5 = unpack2x16float(kc[w + 5u]);
                    let a6 = unpack2x16float(kc[w + 6u]);
                    let a7 = unpack2x16float(kc[w + 7u]);
                    for (var r = 0u; r < rows; r += 1u) {
                        let qb = r * p.dk + i;
                        dots[r] += q_s[qb] * a0.x + q_s[qb + 1u] * a0.y +
                            q_s[qb + 2u] * a1.x + q_s[qb + 3u] * a1.y +
                            q_s[qb + 4u] * a2.x + q_s[qb + 5u] * a2.y +
                            q_s[qb + 6u] * a3.x + q_s[qb + 7u] * a3.y +
                            q_s[qb + 8u] * a4.x + q_s[qb + 9u] * a4.y +
                            q_s[qb + 10u] * a5.x + q_s[qb + 11u] * a5.y +
                            q_s[qb + 12u] * a6.x + q_s[qb + 13u] * a6.y +
                            q_s[qb + 14u] * a7.x + q_s[qb + 15u] * a7.y;
                    }
                }
                let d8 = p.dk & ~7u;
                for (; i < d8; i += 8u) {
                    let w = wb + i / 2u;
                    let a0 = unpack2x16float(kc[w]);
                    let a1 = unpack2x16float(kc[w + 1u]);
                    let a2 = unpack2x16float(kc[w + 2u]);
                    let a3 = unpack2x16float(kc[w + 3u]);
                    for (var r = 0u; r < rows; r += 1u) {
                        let qb = r * p.dk + i;
                        dots[r] += q_s[qb] * a0.x + q_s[qb + 1u] * a0.y +
                            q_s[qb + 2u] * a1.x + q_s[qb + 3u] * a1.y +
                            q_s[qb + 4u] * a2.x + q_s[qb + 5u] * a2.y +
                            q_s[qb + 6u] * a3.x + q_s[qb + 7u] * a3.y;
                    }
                }
                for (; i < p.dk; i += 2u) {
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
                let vb = ((b * p.kv_tile_t + (t_sh[jj] - p.kv_t0)) * p.h_kv + hkv) * p.dv;
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
            let e = (row * p.segs + p.seg_base + seg_local) * (p.dv + 2u);
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
// block's key range into `p.segs` segments (grid.z = tb * segs), each writing an
// UNNORMALIZED partial — dv accumulator values plus (m, l) — into `o` (bound to the
// backend scratch). `attn_merge` (attention_merge.wgsl) then log-sum-exp-combines
// the segments per row. Entry stride is dv + 2 floats at
//   entry = ((b_local * tl + l) * th + h) * segs + seg.
@compute @workgroup_size(256)
fn attn_split(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    stageQF32(wid.z / p.segs_local, wid.y * p.rl, wid.x * p.rh, lidx);
    workgroupBarrier();
    attnCore(wid.z / p.segs_local, wid.z % p.segs_local, wid.y * p.rl, wid.x * p.rh, lidx, true);
}

@compute @workgroup_size(256)
fn attn_split_qf16(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    stageQF16(wid.z / p.segs_local, wid.y * p.rl, wid.x * p.rh, lidx);
    workgroupBarrier();
    attnCore(wid.z / p.segs_local, wid.z % p.segs_local, wid.y * p.rl, wid.x * p.rh, lidx, true);
}

