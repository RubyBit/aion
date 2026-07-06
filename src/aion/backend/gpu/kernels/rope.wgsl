// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Rotary positional embedding (rotate-half layout), matching the CPU kernel
// (backend/cpu/kernels/rope.zig):
//   for pair i < rope_pairs, with (xl, xr) = (x[.., i], x[.., pairs_total + i]):
//     angle  = pos * scale_factor * freq_step^i
//     out_l  = xl*cos - xr*sin
//     out_r  = xl*sin + xr*cos
//   remaining elements copy through unchanged.
//
// One work item per OUTPUT ELEMENT of a packed [tb, tl, tn, th] tile (each
// element of a rotated pair recomputes sincos — cheap next to the loads).
// positions is the matching packed [tb, tl] i32 tile. Grid-stride dispatch.
// NOTE: the CPU uses a fast sincos approximation, so CPU-vs-GPU differences are
// bounded by that approximation (~1e-6 relative), not exact equality.

@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       pos: array<i32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform>             p: Params;

// count = tile elements; th = head dim; tn = heads in tile;
// pairs_total = th/2; rope_pairs = rotated pair count.
struct Params {
    count: u32,
    th: u32,
    tn: u32,
    pairs_total: u32,
    rope_pairs: u32,
    freq_step: f32,
    scale_factor: f32,
    _pad: u32,
};

@compute @workgroup_size(64)
fn rope_f32(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let step = nwg.x * 64u;
    for (var idx = g.x; idx < p.count; idx += step) {
        let h = idx % p.th;
        let row = idx / p.th;

        // Odd th leaves a trailing element beyond both halves; it copies through.
        if (h >= 2u * p.pairs_total) {
            o[idx] = x[idx];
            continue;
        }
        let i = h % p.pairs_total; // pair index (works for both halves)
        if (i >= p.rope_pairs) {
            o[idx] = x[idx];
            continue;
        }

        let l_index = row / p.tn; // (b, l) flat index into the positions tile
        let position = f32(pos[l_index]);
        let freq = p.scale_factor * pow(p.freq_step, f32(i));
        let angle = position * freq;
        let s = sin(angle);
        let cs = cos(angle);

        let base = row * p.th;
        let xl = x[base + i];
        let xr = x[base + p.pairs_total + i];
        if (h < p.pairs_total) {
            o[idx] = xl * cs - xr * s;
        } else {
            o[idx] = xl * s + xr * cs;
        }
    }
}
