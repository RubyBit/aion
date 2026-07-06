// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// STFT: framing + window + real DFT fused in one kernel — one thread per
// (batch, frame, bin), grid-strided; the overlapped framed signal is never
// materialized. Same naive-DFT + integer twiddle reduction and output packing
// as fft.wgsl (see there for the rationale and the radix-2 follow-up note).
//
// Framing: frame f starts at f*hop, shifted left by n_fft/2 with `center`;
// out-of-range samples reflect-pad exactly like the CPU's reflectIndex
// (period 2(S-1), fold m >= S to period - m — no edge repeat).

@group(0) @binding(0) var<storage, read>       sig: array<f32>;
@group(0) @binding(1) var<storage, read>       win: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform>             p: Params;

// sig: [batch, samples]; win: [n_fft]; o: [batch, frames, 2*bins]; all packed.
struct Params {
    batch: u32,
    samples: u32,
    frames: u32,
    n_fft: u32,
    hop: u32,
    bins: u32,
    center: u32,
    total: u32, // batch * frames * bins
};

const TWO_PI: f32 = 6.283185307179586;
const WG: u32 = 64u;

// torch/NeMo reflect padding (no edge repeat): fold i into [0, n). Reflection
// is an even function (x[-j] == x[j]), so fold |i| with an UNSIGNED modulo —
// signed `%` on negative operands miscompiled on NVIDIA Vulkan (observed on an
// RTX 4080, driver-level: the same WGSL was correct on Intel), so keep all the
// index math in u32.
fn reflect_sample(i: i32, n: u32) -> u32 {
    if (n <= 1u) { return 0u; }
    let period = 2u * (n - 1u);
    var m = u32(abs(i)) % period;
    if (m >= n) { m = period - m; }
    return m;
}

@compute @workgroup_size(64)
fn stft_dft(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let stride = nwg.x * WG;
    for (var idx = gid.x; idx < p.total; idx += stride) {
        let k = idx % p.bins;
        let rest = idx / p.bins;
        let f = rest % p.frames;
        let b = rest / p.frames;

        var start = i32(f * p.hop);
        if (p.center != 0u) { start -= i32(p.n_fft / 2u); }
        let sb = b * p.samples;

        var re = 0.0;
        var im = 0.0;
        var m = 0u;
        for (var t = 0u; t < p.n_fft; t += 1u) {
            let si = start + i32(t);
            var v = 0.0;
            if (p.center != 0u) {
                v = sig[sb + reflect_sample(si, p.samples)];
            } else if (si >= 0 && si < i32(p.samples)) {
                v = sig[sb + u32(si)];
            }
            v = v * win[t];
            let theta = -TWO_PI * f32(m) / f32(p.n_fft);
            re += v * cos(theta);
            im += v * sin(theta);
            m += k;
            if (m >= p.n_fft) { m -= p.n_fft; }
        }

        let ob = (b * p.frames + f) * 2u * p.bins;
        o[ob + k] = re;
        o[ob + p.bins + k] = im;
    }
}

// ---------------------------------------------------------------------------
// Cooperative shared-memory radix-2 STFT (Stockham auto-sort), one workgroup
// per (frame, batch). The workgroup frames + windows + reflect-pads into shared
// memory once (each input sample read once, not `bins` times), then runs an
// O(n log n) complex FFT. Framing math is identical to stft_dft; only the DFT
// is replaced. Output packing unchanged. Falls back to stft_dft when the device
// or n_fft don't fit (executor predicate). See fft.wgsl for the FFT rationale.
// ---------------------------------------------------------------------------
const PI: f32 = 3.141592653589793;
const WGC: u32 = 256u;
const MAX_N: u32 = 1024u;

var<workgroup> fbuf: array<vec2<f32>, 2u * MAX_N>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn fft_shared(n: u32, lidx: u32) -> u32 {
    let half = n >> 1u;
    var src = 0u;
    var p_stride = 1u;
    loop {
        if (p_stride >= n) { break; }
        let dst = MAX_N - src;
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
fn stft_coop(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let f = wid.x; // frame
    let b = wid.y; // batch

    var start = i32(f * p.hop);
    if (p.center != 0u) { start -= i32(p.n_fft / 2u); }
    let sb = b * p.samples;

    var t = lidx;
    loop {
        if (t >= p.n_fft) { break; }
        let si = start + i32(t);
        var v = 0.0;
        if (p.center != 0u) {
            v = sig[sb + reflect_sample(si, p.samples)];
        } else if (si >= 0 && si < i32(p.samples)) {
            v = sig[sb + u32(si)];
        }
        fbuf[t] = vec2<f32>(v * win[t], 0.0);
        t = t + WGC;
    }
    workgroupBarrier();

    let base = fft_shared(p.n_fft, lidx);

    let ob = (b * p.frames + f) * 2u * p.bins;
    var kk = lidx;
    loop {
        if (kk >= p.bins) { break; }
        let cc = fbuf[base + kk];
        o[ob + kk] = cc.x;
        o[ob + p.bins + kk] = cc.y;
        kk = kk + WGC;
    }
}
