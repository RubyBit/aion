// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Real FFT over the last dim — v1 is a NAIVE per-bin DFT (one thread per
// output bin, grid-strided):
//   X[k] = sum_t x[t] * exp(-2*pi*i * k*t / n)
// Output layout matches the CPU RFFT: last dim `n+2` holds the one-sided
// spectrum, real parts in [0..bins), imaginary in [bins..2*bins), bins=n/2+1.
//
// Twiddle angles reduce k*t modulo n in INTEGER math (incrementally: m += k,
// fold once), so cos/sin always see arguments in [0, 2*pi) — full f32
// precision regardless of k*t magnitude.
//
// O(n^2) per frame vs the CPU's O(n log n) plan: for the audio-front-end sizes
// this feeds (n_fft <= 1024) the GPU has heaps of headroom; a shared-memory
// radix-2 kernel is the perf follow-up if profiles ever show it.

@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@group(0) @binding(2) var<uniform>             p: Params;

// x: [rows, n] packed; o: [rows, 2*bins] packed.
struct Params { rows: u32, n: u32, bins: u32, total: u32 };

const TWO_PI: f32 = 6.283185307179586;
const WG: u32 = 64u;

@compute @workgroup_size(64)
fn rfft_dft(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let stride = nwg.x * WG;
    for (var idx = gid.x; idx < p.total; idx += stride) {
        let row = idx / p.bins;
        let k = idx % p.bins;
        let xb = row * p.n;

        var re = 0.0;
        var im = 0.0;
        var m = 0u; // (k*t) mod n, updated incrementally
        for (var t = 0u; t < p.n; t += 1u) {
            let theta = -TWO_PI * f32(m) / f32(p.n);
            let v = x[xb + t];
            re += v * cos(theta);
            im += v * sin(theta);
            m += k;
            if (m >= p.n) { m -= p.n; } // k < n, so one subtraction suffices
        }

        let ob = row * 2u * p.bins;
        o[ob + k] = re;
        o[ob + p.bins + k] = im;
    }
}

// ---------------------------------------------------------------------------
// Cooperative shared-memory radix-2 FFT (Stockham auto-sort), one workgroup per
// row. O(n log n) instead of the naive DFT's O(n^2). Full-size complex FFT on
// the real input (imag = 0), emitting the one-sided spectrum [0, n/2]. Output
// packing is identical to rfft_dft, so the executor/tests are unchanged.
// Selected by the executor when n <= MAX_N and the device fits the shared
// buffer + 256 invocations; otherwise it falls back to rfft_dft.
// ---------------------------------------------------------------------------
const PI: f32 = 3.141592653589793;
const WGC: u32 = 256u;
const MAX_N: u32 = 1024u;

// Two ping-pong complex buffers packed into one array (offset 0 or MAX_N), so
// the stage toggle is index arithmetic — no per-access uniform branch.
var<workgroup> fbuf: array<vec2<f32>, 2u * MAX_N>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Size-n Stockham FFT of data preloaded into fbuf[0..n]. Returns the base
// offset (0 or MAX_N) of the natural-order result.
fn fft_shared(n: u32, lidx: u32) -> u32 {
    let half = n >> 1u;
    var src = 0u;
    var p_stride = 1u;
    loop {
        if (p_stride >= n) { break; }
        let dst = MAX_N - src; // toggles 0 <-> MAX_N
        var j = lidx;
        loop {
            if (j >= half) { break; }
            let k = j & (p_stride - 1u);
            let u0 = fbuf[src + j];
            let u1 = fbuf[src + j + half];
            let angle = -PI * f32(k) / f32(p_stride);
            let wv = vec2<f32>(cos(angle), sin(angle));
            let v = cmul(wv, u1);
            let jo = ((j - k) << 1u) + k;
            fbuf[dst + jo] = u0 + v;
            fbuf[dst + jo + p_stride] = u0 - v;
            j = j + WGC;
        }
        workgroupBarrier();
        src = dst;
        p_stride = p_stride << 1u;
    }
    return src;
}

@compute @workgroup_size(256)
fn rfft_coop(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let row = wid.x;
    let xb = row * p.n;
    var t = lidx;
    loop {
        if (t >= p.n) { break; }
        fbuf[t] = vec2<f32>(x[xb + t], 0.0);
        t = t + WGC;
    }
    workgroupBarrier();

    let base = fft_shared(p.n, lidx);

    let ob = row * 2u * p.bins;
    var kk = lidx;
    loop {
        if (kk >= p.bins) { break; }
        let cc = fbuf[base + kk];
        o[ob + kk] = cc.x;
        o[ob + p.bins + kk] = cc.y;
        kk = kk + WGC;
    }
}
