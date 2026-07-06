// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Device-side row gather: o[r, :] = table[idx[r], :] for a single-buffer f32
// table. One thread per output element, grid-strided. Replaces the record-time
// one-copy-per-row path when the table is a single tile — that path costs ~1 µs
// of encoder overhead per row AND a host read of the indices (a forced sync
// when a GPU op produced them, e.g. argmax feeding the next decode step).
//
// Out-of-range indices clamp into [0, v) — the device cannot report errors;
// the CPU backend still validates the same graphs at execute time.

@group(0) @binding(0) var<storage, read>       table: array<f32>;
@group(0) @binding(1) var<storage, read>       idx: array<i32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform>             p: Params;

struct Params { rows: u32, d: u32, v: u32, total: u32 };

const WG: u32 = 64u;

@compute @workgroup_size(64)
fn gather_rows_f32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let stride = nwg.x * WG;
    for (var i = gid.x; i < p.total; i += stride) {
        let r = i / p.d;
        let col = i % p.d;
        let src_row = u32(clamp(idx[r], 0, i32(p.v) - 1));
        o[i] = table[src_row * p.d + col];
    }
}
