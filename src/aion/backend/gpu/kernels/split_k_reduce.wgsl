// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// The second stage of a split-K product: sum the `splits` partial [M, N] blocks a
// GEMM wrote one after another, then scale and accumulate into C:
//   C[m, c_off + n] = alpha * sum_z part[z, m, n]  +  beta * C[m, c_off + n]
// Grid-strided over the M * N elements, a vec4 of N per step (N % 4 == 0).

@group(0) @binding(0) var<storage, read>       part: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> c: array<f32>;
@group(0) @binding(2) var<uniform>             p: Params;

// Field order matches `ReduceParams` in exec/matmul_q8_mx16.zig.
struct Params { m: u32, n: u32, splits: u32, c_row: u32, c_off: u32, alpha: f32, beta: f32, _p0: u32 };

const WG: u32 = 256u;

@compute @workgroup_size(WG)
fn split_k_reduce(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let quads = p.m * p.n / 4u;
    for (var e = gid.x; e < quads; e += nwg.x * WG) {
        var s = vec4<f32>();
        for (var z = 0u; z < p.splits; z += 1u) {
            s += part[z * quads + e];
        }
        let mm = (e * 4u) / p.n;
        let nn = (e * 4u) % p.n;
        let i = mm * p.c_row + p.c_off + nn;
        for (var j = 0u; j < 4u; j += 1u) {
            let v = p.alpha * s[j];
            if (p.beta == 0.0) {
                c[i + j] = v;
            } else {
                c[i + j] = v + p.beta * c[i + j];
            }
        }
    }
}
