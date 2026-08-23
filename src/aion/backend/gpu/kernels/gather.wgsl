// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Device-side row gather: o[r, :] = table[idx[r], :] for a single-buffer table.
// The row index resolves ON DEVICE (no host read of the indices → no forced
// sync when a GPU op produced them, e.g. an in-graph decode loop feeding an
// argmax / a reshaped carry into the gather). Replaces the record-time
// one-copy-per-row path (which costs ~1 µs of encoder overhead per row plus that
// host read) when the table is a single tile.
//
// Every entry point shares one binding layout so they coexist in this module:
// `table`, `idx` and `o` are bound as raw words and moved by bitcast, which is
// what lets the copying gathers serve f32/f16/i32 alike — only the q8_0 gather
// interprets bits, because it dequantizes. Out-of-range indices clamp into
// [0, v) — the device cannot report errors; the CPU backend still validates the
// same graphs at execute time.
//
// Operands too large for one storage binding are split into row-major tiles, and
// each binding holds ONE of them, so the caller dispatches per tile pair:
//   `p.p0`/`p.p1` = the bound table tile's `[row_begin, row_end)` in table rows;
//                   a work item whose index falls outside it skips.
//   `p.p2`        = the bound output tile's first row (batch, for the batched
//                   gather) in the full output, so `idx` still indexes globally
//                   while `o` is written tile-locally.
// Single-tile operands are just the cases `[0, v)` and `0`.

enable f16;

@group(0) @binding(0) var<storage, read>       table: array<u32>;
@group(0) @binding(1) var<storage, read>       idx: array<i32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;

// f16 element-addressed aliases of bindings 0 and 2 (`idx` is i32 either way).
//
// The word kernels above move 4 bytes per work item, which is the right unit for
// f32/i32 and for an f16 row of EVEN width. It cannot address an f16 row of odd
// width at all: row `v` of an f16 [V, 37] table starts at byte 74*v, which is
// 2 mod 4 on odd `v`, so it has no u32 word offset. `shader-f16` gives 2-byte
// addressing, so the twins below take that case by counting ELEMENTS where their
// word originals count words. They are chosen ONLY when the row is odd — an even
// f16 row keeps the word path and its half-as-many invocations.
@group(0) @binding(0) var<storage, read>       table_h: array<f16>;
@group(0) @binding(2) var<storage, read_write> oh: array<f16>;
@group(0) @binding(3) var<uniform>             p: Params;

struct Params {
    rows: u32,
    d: u32,
    v: u32,
    wpr: u32,
    total: u32,
    // gather_*: table tile row range [p0, p1). sequence_append_u32: ring window
    // and total words. Unused by an entry point that does not name them.
    p0: u32,
    p1: u32,
    p2: u32,
    /// gather_batched_words: first gathered row this output tile holds, so the
    /// output may also split along G. 0 for every other entry point.
    p3: u32,
    // Scalar pads, not a vec3: a vec3<u32> would force 16-byte alignment and
    // make this struct 64 bytes, silently disagreeing with the 48-byte Zig
    // `GatherParams` that fills it.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

const WG: u32 = 64u;

// Pure word copy, so it serves any non-quantized dtype whose row is a whole
// number of 4-byte words: `p.d` counts WORDS per row, not elements.
@compute @workgroup_size(64)
fn gather_rows_words(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let stride = nwg.x * WG;
    for (var i = gid.x; i < p.total; i += stride) {
        let r = i / p.d;
        let col = i % p.d;
        let src_row = u32(clamp(idx[p.p2 + r], 0, i32(p.v) - 1));
        if (src_row < p.p0 || src_row >= p.p1) { continue; } // row lives in another tile
        o[i] = bitcast<f32>(table[(src_row - p.p0) * p.d + col]);
    }
}

fn i8x4f(w: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(i32(w << 24u) >> 24u),
        f32(i32(w << 16u) >> 24u),
        f32(i32(w << 8u) >> 24u),
        f32(i32(w) >> 24u),
    );
}

// Device-side gather from a q8_0 single-buffer table:
//   o[r, :] = dequant(table[idx[r], :]).
// Same dequant math and 17-word block-pair layout as dequant.wgsl
// `q8_row_to_f32`; requires D % 64 == 0. `p.wpr` = u32 words per table row
// ( = (D/64)*17 ); `p.total` = block pairs across all output rows. One work item
// = one 64-element block pair of one output row.
@compute @workgroup_size(64)
fn gather_q8_rows_f32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let stride = nwg.x * WG;
    let pairs_per_row = p.d / 64u;
    for (var i = gid.x; i < p.total; i += stride) {
        let r = i / pairs_per_row;
        let pi = i % pairs_per_row;
        let src_row = u32(clamp(idx[p.p2 + r], 0, i32(p.v) - 1));
        if (src_row < p.p0 || src_row >= p.p1) { continue; } // row lives in another tile
        let base = (src_row - p.p0) * p.wpr + pi * 17u;
        let e0 = r * p.d + pi * 64u;

        let w0 = table[base];
        let d0 = unpack2x16float(w0).x;
        let d1 = unpack2x16float(table[base + 8u]).y;

        var prev = w0;
        for (var j = 0u; j < 8u; j += 1u) {
            let cur = table[base + 1u + j];
            let q = i8x4f((prev >> 16u) | (cur << 16u)) * d0;
            let e = e0 + j * 4u;
            o[e] = q.x;
            o[e + 1u] = q.y;
            o[e + 2u] = q.z;
            o[e + 3u] = q.w;
            prev = cur;
        }
        for (var j = 0u; j < 8u; j += 1u) {
            let q = i8x4f(table[base + 9u + j]) * d1;
            let e = e0 + 32u + j * 4u;
            o[e] = q.x;
            o[e + 1u] = q.y;
            o[e + 2u] = q.z;
            o[e + 3u] = q.w;
        }
    }
}

// Device-side row scatter: buf[idx[0], :] = src[:]. The destination row resolves
// ON DEVICE (the emit index of an in-graph decode loop) so no host read forces a
// control-flow sync. Reuses this module's bindings: `table` = the packed src row
// (u32 words), `idx` = the destination index, `o` = the destination buffer
// (written via bitcast so any 4-byte scalar — f32 / i32 — moves dtype-agnostically).
// Reuses Params fields: `p.d` = words per row, `p.v` = row count (clamp bound),
// `p.total` = words per row, `p.p0`/`p.p1` = the bound buf tile's row range.
@compute @workgroup_size(64)
fn scatter_row_u32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let stride = nwg.x * WG;
    let row = u32(clamp(idx[0], 0, i32(p.v) - 1));
    if (row < p.p0 || row >= p.p1) { return; } // row lives in another buf tile
    let dst0 = (row - p.p0) * p.d;
    for (var i = gid.x; i < p.total; i += stride) {
        o[dst0 + i] = bitcast<f32>(table[i]);
    }
}

// Device-side KV append for packed single-buffer caches. Bindings are reused as:
// `table` = new_kv words, `idx` = per-batch append offsets, `o` = cache words.
// Params are reinterpreted as {batch, tile_t, new_t, heads, row_words,
// ring_window, total_words, tile_t0}. A zero ring window selects ordinary fixed
// storage. A cache too large for one binding is split along TIME and `o` binds
// ONE tile: `tile_t0`/`tile_t` are its range, so each tile is a dispatch that
// writes only the appended rows landing inside it. A single-tile cache is
// `{0, cache_t}`.
@compute @workgroup_size(64)
fn sequence_append_u32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let batch = p.rows;
    let tile_t = p.d;
    let tile_t0 = p.p2;
    let new_t = p.v;
    let heads = p.wpr;
    let row_words = p.total;
    let ring_window = p.p0;
    let total_words = p.p1;
    let stride = nwg.x * WG;
    for (var i = gid.x; i < total_words; i += stride) {
        let word = i % row_words;
        let row = i / row_words;
        let h = row % heads;
        let l = (row / heads) % new_t;
        let b = row / (heads * new_t);
        if (b >= batch) { continue; }
        var dst_t = u32(max(idx[b], 0)) + l;
        if (ring_window != 0u) { dst_t = dst_t % ring_window; }
        // Tiles partition [0, cache_t), so this also drops a position past the end.
        if (dst_t >= tile_t0 && dst_t - tile_t0 < tile_t) {
            let dst = (((b * tile_t + (dst_t - tile_t0)) * heads + h) * row_words) + word;
            o[dst] = bitcast<f32>(table[i]);
        }
    }
}

// Batched row gather: o[b, g, :] = data[b, idx[b, g], :] — the `GatherTiled`
// lowering for axis 1 with one batch dim. Single-tile operands, so no row-range
// parameters: `p.rows` = B, `p.d` = row width, `p.v` = rows per batch,
// `p.wpr` = gathered rows per batch, `p.total` = B*G*W output elements.
// One work item = one output element.
@compute @workgroup_size(64)
fn gather_batched_words(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let stride = nwg.x * WG;
    let per_batch = p.total / p.rows; // words per batch in THIS tile
    for (var i = gid.x; i < p.total; i += stride) {
        let b = i / per_batch;
        let rem = i % per_batch;
        let g = rem / p.d;
        let w = rem % p.d;
        let gb = p.p2 + b;  // batch index in the full tensor
        let gg = p.p3 + g;  // gathered row in the full tensor
        if (gb < p.p0 || gb >= p.p1) { continue; } // batch lives in another data tile
        let src = u32(clamp(idx[gb * p.wpr + gg], 0, i32(p.v) - 1));
        o[i] = bitcast<f32>(table[((gb - p.p0) * p.v + src) * p.d + w]);
    }
}

// ---- f16 element-addressed twins -------------------------------------------
//
// Each is its f32/word original with `p.d` / `p.total` reinterpreted as ELEMENTS
// per row / total elements, and the bitcast move replaced by a direct f16 one.

@compute @workgroup_size(64)
fn gather_rows_f16(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let stride = nwg.x * WG;
    for (var i = gid.x; i < p.total; i += stride) {
        let r = i / p.d;
        let col = i % p.d;
        let src_row = u32(clamp(idx[p.p2 + r], 0, i32(p.v) - 1));
        if (src_row < p.p0 || src_row >= p.p1) { continue; } // row lives in another tile
        oh[i] = table_h[(src_row - p.p0) * p.d + col];
    }
}

@compute @workgroup_size(64)
fn scatter_row_f16(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let stride = nwg.x * WG;
    let row = u32(clamp(idx[0], 0, i32(p.v) - 1));
    if (row < p.p0 || row >= p.p1) { return; } // row lives in another buf tile
    let dst0 = (row - p.p0) * p.d;
    for (var i = gid.x; i < p.total; i += stride) {
        oh[dst0 + i] = table_h[i];
    }
}

@compute @workgroup_size(64)
fn sequence_append_f16(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let batch = p.rows;
    let tile_t = p.d;
    let tile_t0 = p.p2;
    let new_t = p.v;
    let heads = p.wpr;
    let row_elems = p.total;
    let ring_window = p.p0;
    let total_elems = p.p1;
    let stride = nwg.x * WG;
    for (var i = gid.x; i < total_elems; i += stride) {
        let elem = i % row_elems;
        let row = i / row_elems;
        let h = row % heads;
        let l = (row / heads) % new_t;
        let b = row / (heads * new_t);
        if (b >= batch) { continue; }
        var dst_t = u32(max(idx[b], 0)) + l;
        if (ring_window != 0u) { dst_t = dst_t % ring_window; }
        // Tiles partition [0, cache_t), so this also drops a position past the end.
        if (dst_t >= tile_t0 && dst_t - tile_t0 < tile_t) {
            let dst = (((b * tile_t + (dst_t - tile_t0)) * heads + h) * row_elems) + elem;
            oh[dst] = table_h[i];
        }
    }
}

@compute @workgroup_size(64)
fn gather_batched_f16(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let stride = nwg.x * WG;
    let per_batch = p.total / p.rows; // elements per batch in THIS tile
    for (var i = gid.x; i < p.total; i += stride) {
        let b = i / per_batch;
        let rem = i % per_batch;
        let g = rem / p.d;
        let w = rem % p.d;
        let gb = p.p2 + b;  // batch index in the full tensor
        let gg = p.p3 + g;  // gathered row in the full tensor
        if (gb < p.p0 || gb >= p.p1) { continue; } // batch lives in another data tile
        let src = u32(clamp(idx[gb * p.wpr + gg], 0, i32(p.v) - 1));
        oh[i] = table_h[((gb - p.p0) * p.v + src) * p.d + w];
    }
}
