// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Relative-positional multi-head self-attention (Transformer-XL / Conformer):
//   scores[i,j] = ((q[i]+u) . k[j] + (q[i]+v) . pos_emb[base_i + j]) * scale
//                 + mask[i,j]                       (base_i = (T_q-1) - i)
//   out[i] = softmax_j(scores[i,:]) @ v
// over the row's key window (chunked-limited attention; see p.chunk_size).
//
// Layouts (per the compile contract): q/k/v/out slices are contiguous [T, D]
// panels (the [B, T, H, D] tensors are tiled [1, T, 1, D]); pos_emb / biases are
// indexed via p.pe_base / p.u_base / p.v_base element offsets into their
// (per-head or whole) tiles; mask is a packed additive [T_q, T_kv] tile. When
// p.has_mask == 0 the mask binding is a dummy (the backend rebinds q).
//
// WORK SHAPE — one 256-thread workgroup per (batch, head, row block), mirroring
// attention.wgsl:
//
//   * A block is `p.rl` query rows. K and V rows are shared by every row in the
//     block, so each is loaded once and fanned into all `rl` dot products /
//     accumulators. `pos_emb` is NOT shared — row i reads the band at
//     `base_i + j`, which shifts by one per row — so the position term still costs
//     one row read per (row, key). That still moves the load:FMA ratio from 1.0 to
//     (1 + rl) / (2 * rl).
//
//   * The score phase is KEY-parallel; the V phase is DIM-parallel (thread i owns
//     value dim `i % dv_eff` of key group `i / dv_eff`), so V reads coalesce and all
//     256 threads stay busy. The previous dim-strided V loop ran one key at a time
//     with only d/256 threads active.
//
//   * Per-row reductions share ONE barrier chain and run destructively in `p_sh`
//     (max before the probabilities are written, sum after the V phase consumes
//     them), so there is no separate reduction scratch.
//
// Rows in a block may sit in different chunks and so have different windows; each
// row carries its own [lo, hi) and the block's key loop covers their union.

@group(0) @binding(0) var<storage, read>       q: array<f32>;
@group(0) @binding(1) var<storage, read>       k: array<f32>;
@group(0) @binding(2) var<storage, read>       v: array<f32>;
@group(0) @binding(3) var<storage, read>       pe: array<f32>;
@group(0) @binding(4) var<storage, read>       bu: array<f32>;
@group(0) @binding(5) var<storage, read>       bv: array<f32>;
@group(0) @binding(6) var<storage, read>       mask: array<f32>;
@group(0) @binding(7) var<storage, read_write> o: array<f32>;
@group(0) @binding(8) var<uniform>             p: Params;

struct Params {
    t_q: u32,
    t_kv: u32,
    d: u32,
    pe_base: u32, // element offset of this head's [P, D] panel
    u_base: u32, // element offset of this head's [D] bias row
    v_base: u32,
    has_mask: u32,
    // Chunked-limited window: 0 = attend to every key. Otherwise a query attends to
    // its own chunk of `chunk_size` keys plus `chunk_left` before that chunk's start
    // — one contiguous interval, so the key loop runs over the window instead of
    // scoring all t_kv keys and masking almost all of them away.
    chunk_size: u32,
    chunk_left: u32,
    rl: u32, // query rows per block
    scale: f32,
};

const WG: u32 = 256u;
const RMAX: u32 = 4u; // rows per block; bounds the staging arrays and register arrays
const ACC: u32 = 4u; // dims per thread when d > WG (d <= ACC * WG)
const FMIN: f32 = -3.4028235e38;

// 12 KiB, inside the 16 KiB WebGPU workgroup-storage floor. The 1024 is
// `Q_STAGE_FLOATS` in exec/attention.zig, which enforces `rl * d <= 1024`; WGSL array
// sizes must be literals, so the two are hand-kept.
var<workgroup> qu_s: array<f32, 1024>; // [row][d] q + pos_bias_u
var<workgroup> qv_s: array<f32, 1024>; // [row][d] q + pos_bias_v
var<workgroup> p_sh: array<f32, 1024>; // [row][WG] scores -> probabilities

fn expApprox(x_in: f32) -> f32 {
    let xc = clamp(x_in, -80.0, 80.0);
    let yy = xc * 1.4426950408889634;
    let nn = i32(floor(yy + 0.5));
    let tt = (yy - f32(nn)) * 0.6931471805599453;
    let e2 = bitcast<f32>(u32(nn + 127) << 23u);
    return e2 * (1.0 + tt * (1.0 + tt * (0.5 + tt * (0.16666667 + tt * 0.041666668))));
}

/// Tree-reduce `p_sh[r][0..WG]` into `p_sh[r][0]` for all `rows` in ONE barrier
/// chain. Destructive; caller has written its lane and issued a barrier.
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

@compute @workgroup_size(256)
fn relpos_mha_row(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let row0 = wid.x * p.rl;
    let rows = p.rl;

    // --- per-row key window + the block's union ---
    var lo_r: array<u32, 4>;
    var hi_r: array<u32, 4>;
    var span_lo = 0xffffffffu;
    var span_hi = 0u;
    for (var r = 0u; r < rows; r += 1u) {
        lo_r[r] = 0u;
        hi_r[r] = 0u;
        let row = row0 + r;
        if (row >= p.t_q) { continue; }
        var lo = 0u;
        var hi = p.t_kv;
        if (p.chunk_size > 0u) {
            let a = row + (p.t_kv - p.t_q);
            let cs = (a / p.chunk_size) * p.chunk_size;
            lo = cs - min(cs, p.chunk_left);
            hi = min(cs + p.chunk_size, p.t_kv);
        }
        lo_r[r] = lo;
        hi_r[r] = hi;
        span_lo = min(span_lo, lo);
        span_hi = max(span_hi, hi);
    }
    if (span_hi <= span_lo) { return; }

    // --- stage the block's biased q rows ---
    let rows_d = rows * p.d;
    for (var i = lidx; i < rows_d; i += WG) {
        let r = i / p.d;
        let dd = i % p.d;
        var qi = 0.0;
        if (row0 + r < p.t_q) { qi = q[(row0 + r) * p.d + dd]; }
        qu_s[i] = qi + bu[p.u_base + dd];
        qv_s[i] = qi + bv[p.v_base + dd];
    }
    workgroupBarrier();

    // V-phase mapping: `dv_eff` consecutive dims per key group, key groups rounded
    // down to a power of two so the tail reduction is a clean tree.
    let dv_eff = min(p.d, WG);
    let kgc = 1u << firstLeadingBit(WG / dv_eff);
    let d0 = lidx % dv_eff;
    let kg = lidx / dv_eff;
    let dsteps = (p.d + dv_eff - 1u) / dv_eff;

    var acc: array<f32, 16>; // [row][dim step]
    for (var i = 0u; i < 16u; i += 1u) { acc[i] = 0.0; }
    var m_state: array<f32, 4>;
    var l_state: array<f32, 4>;
    for (var r = 0u; r < RMAX; r += 1u) {
        m_state[r] = FMIN;
        l_state[r] = 0.0;
    }

    for (var t0 = span_lo; t0 < span_hi; t0 += WG) {
        let j = t0 + lidx;

        // --- scores: K row read ONCE for the block; pos_emb once per (row, key) ---
        var sv: array<f32, 4>;
        var vr: array<bool, 4>;
        for (var r = 0u; r < rows; r += 1u) {
            sv[r] = FMIN;
            vr[r] = false;
        }
        if (j < span_hi) {
            let kb = j * p.d;
            var ac: array<f32, 4>;
            var bd: array<f32, 4>;
            var peb: array<u32, 4>;
            for (var r = 0u; r < rows; r += 1u) {
                ac[r] = 0.0;
                bd[r] = 0.0;
                // base_{row0+r} + j, i.e. the band shifts down by one per row. Loop
                // invariant, so it is computed once per row rather than per element.
                peb[r] = p.pe_base + (((p.t_q - 1u) - (row0 + r)) + j) * p.d;
            }
            for (var i = 0u; i < p.d; i += 1u) {
                let kv = k[kb + i];
                for (var r = 0u; r < rows; r += 1u) {
                    ac[r] += qu_s[r * p.d + i] * kv;
                    bd[r] += qv_s[r * p.d + i] * pe[peb[r] + i];
                }
            }
            for (var r = 0u; r < rows; r += 1u) {
                if (j >= lo_r[r] && j < hi_r[r]) {
                    var sc = (ac[r] + bd[r]) * p.scale;
                    if (p.has_mask != 0u) { sc += mask[(row0 + r) * p.t_kv + j]; }
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
        workgroupBarrier();

        // --- V: each element loaded once, reused across the block's rows ---
        let cnt = min(WG, span_hi - t0);
        if (kg < kgc) {
            for (var jj = kg; jj < cnt; jj += kgc) {
                let vb = (t0 + jj) * p.d;
                for (var i = 0u; i < dsteps; i += 1u) {
                    let dd = d0 + i * dv_eff;
                    if (dd < p.d) {
                        let ve = v[vb + dd];
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

    // --- combine the key groups' partials (d <= WG => dsteps == 1) ---
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
        let row = row0 + r;
        if (row >= p.t_q) { continue; }
        // An all-masked row would leave l_state == 0; write zeros rather than inf,
        // matching the CPU executor.
        var inv = 0.0;
        if (l_state[r] > 0.0) { inv = 1.0 / l_state[r]; }
        if (kg == 0u) {
            for (var i = 0u; i < dsteps; i += 1u) {
                let dd = d0 + i * dv_eff;
                if (dd < p.d) { o[row * p.d + dd] = acc[r * ACC + i] * inv; }
            }
        }
    }
}
