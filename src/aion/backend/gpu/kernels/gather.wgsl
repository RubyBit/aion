// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Device-side row gather: o[r, :] = table[idx[r], :] for a single-buffer table.
// The row index resolves ON DEVICE (no host read of the indices → no forced
// sync when a GPU op produced them, e.g. an in-graph decode loop feeding an
// argmax / a reshaped carry into the gather). Replaces the record-time
// one-copy-per-row path (which costs ~1 µs of encoder overhead per row plus that
// host read) when the table is a single tile.
//
// Both entry points share one binding layout so they coexist in this module:
// `table` is bound as raw u32 words (f32 bits for the f32 gather, q8_0 bytes for
// the quantized gather). `p.wpr` (words per q8_0 table row) is unused by the f32
// path. Out-of-range indices clamp into [0, v) — the device cannot report errors;
// the CPU backend still validates the same graphs at execute time.

@group(0) @binding(0) var<storage, read>       table: array<u32>;
@group(0) @binding(1) var<storage, read>       idx: array<i32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform>             p: Params;

struct Params { rows: u32, d: u32, v: u32, wpr: u32, total: u32 };

const WG: u32 = 64u;

@compute @workgroup_size(64)
fn gather_rows_f32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let stride = nwg.x * WG;
    for (var i = gid.x; i < p.total; i += stride) {
        let r = i / p.d;
        let col = i % p.d;
        let src_row = u32(clamp(idx[r], 0, i32(p.v) - 1));
        o[i] = bitcast<f32>(table[src_row * p.d + col]);
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
        let src_row = u32(clamp(idx[r], 0, i32(p.v) - 1));
        let base = src_row * p.wpr + pi * 17u;
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
// `p.total` = words per row.
@compute @workgroup_size(64)
fn scatter_row_u32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let stride = nwg.x * WG;
    let row = u32(clamp(idx[0], 0, i32(p.v) - 1));
    let dst0 = row * p.d;
    for (var i = gid.x; i < p.total; i += stride) {
        o[dst0 + i] = bitcast<f32>(table[i]);
    }
}
