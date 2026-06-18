// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Power-of-two real FFT (RFFT) core.
//
// Strategy:
//   * A real FFT of length `n_fft` is computed via a complex FFT of half size
//     `m = n_fft/2`: the real input is packed as `z[n] = x[2n] + i*x[2n+1]`,
//     run through an in-place iterative radix-2 DIT complex FFT, then a final
//     "split" recombination recovers the `n_fft/2 + 1` one-sided bins.
//   * SIMD is applied ACROSS frames, not within a single FFT. A group of up to
//     `lanes` independent frames is processed at once: each complex working
//     element holds a `@Vector(lanes, f32)` of real and imaginary parts, while
//     twiddle factors are scalars broadcast via `@splat`. This keeps the
//     butterflies free of intra-FFT shuffles and saturates the vector units.
//
// The `Plan` (bit-reversal table + twiddles) is lane-independent and shared by
// every lane-width variant. `Kernel(lanes)` produces the lane-specific group
// processor; `kernels(lanes)` wraps it in the `FftKernels` fn-pointer struct
// consumed by `registry/fft_registry.zig`.

const std = @import("std");
const fast_math = @import("fast_math.zig");

pub const FftError = error{
    NotPowerOfTwo,
    SizeTooSmall,
    InvalidArgument,
    OutOfMemory,
};

fn isPowerOfTwo(n: usize) bool {
    return n != 0 and (n & (n - 1)) == 0;
}

/// Lane-independent plan for a power-of-two real FFT of length `n_fft`.
///
/// Holds the bit-reversal permutation and precomputed twiddle tables for the
/// half-size complex FFT plus the split-step recombination. Build once (the
/// `n_fft` of a graph op is fixed), reuse across all frames and threads.
pub const Plan = struct {
    /// Real FFT length (power of two, >= 4).
    n_fft: usize,
    /// Half length `n_fft/2` — the size of the internal complex FFT.
    m: usize,
    /// Number of one-sided output bins, `n_fft/2 + 1`.
    bins: usize,

    /// Bit-reversal permutation for size `m`.
    bit_rev: []u32,
    /// Complex-FFT twiddles, length `m/2`: `cos(2*pi*t/m)` / `-sin(2*pi*t/m)`.
    cfft_cos: []f32,
    cfft_sin: []f32,
    /// Split-step twiddles, length `bins`: `cos(2*pi*k/n_fft)` / `-sin(2*pi*k/n_fft)`.
    split_cos: []f32,
    split_sin: []f32,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, n_fft: usize) FftError!Plan {
        if (!isPowerOfTwo(n_fft)) return FftError.NotPowerOfTwo;
        if (n_fft < 4) return FftError.SizeTooSmall;

        const m: usize = n_fft / 2;
        const bins: usize = n_fft / 2 + 1;

        const bit_rev = allocator.alloc(u32, m) catch return FftError.OutOfMemory;
        errdefer allocator.free(bit_rev);

        const half_m: usize = m / 2;
        const cfft_cos = allocator.alloc(f32, @max(half_m, 1)) catch return FftError.OutOfMemory;
        errdefer allocator.free(cfft_cos);
        const cfft_sin = allocator.alloc(f32, @max(half_m, 1)) catch return FftError.OutOfMemory;
        errdefer allocator.free(cfft_sin);

        const split_cos = allocator.alloc(f32, bins) catch return FftError.OutOfMemory;
        errdefer allocator.free(split_cos);
        const split_sin = allocator.alloc(f32, bins) catch return FftError.OutOfMemory;
        errdefer allocator.free(split_sin);

        // Bit-reversal permutation for size m.
        const log2_m: u6 = @intCast(std.math.log2_int(usize, m));
        for (0..m) |i| {
            var v: usize = i;
            var r: usize = 0;
            var b: u6 = 0;
            while (b < log2_m) : (b += 1) {
                r = (r << 1) | (v & 1);
                v >>= 1;
            }
            bit_rev[i] = @intCast(r);
        }

        // Complex-FFT twiddles: W_m^t = exp(-2*pi*i*t/m), t in [0, m/2).
        const m_f: f64 = @floatFromInt(m);
        for (0..half_m) |t| {
            const ang: f64 = -2.0 * std.math.pi * @as(f64, @floatFromInt(t)) / m_f;
            cfft_cos[t] = @floatCast(@cos(ang));
            cfft_sin[t] = @floatCast(@sin(ang));
        }

        // Split twiddles: W_n^k = exp(-2*pi*i*k/n_fft), k in [0, bins).
        const n_f: f64 = @floatFromInt(n_fft);
        for (0..bins) |k| {
            const ang: f64 = -2.0 * std.math.pi * @as(f64, @floatFromInt(k)) / n_f;
            split_cos[k] = @floatCast(@cos(ang));
            split_sin[k] = @floatCast(@sin(ang));
        }

        return .{
            .n_fft = n_fft,
            .m = m,
            .bins = bins,
            .bit_rev = bit_rev,
            .cfft_cos = cfft_cos,
            .cfft_sin = cfft_sin,
            .split_cos = split_cos,
            .split_sin = split_sin,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Plan) void {
        self.allocator.free(self.bit_rev);
        self.allocator.free(self.cfft_cos);
        self.allocator.free(self.cfft_sin);
        self.allocator.free(self.split_cos);
        self.allocator.free(self.split_sin);
        self.* = undefined;
    }
};

/// Scratch (per call/thread) needed by `Kernel(lanes).processGroup` for a plan.
/// Holds the `m` interleaved complex working elements for one lane group.
pub fn scratchBytes(plan: *const Plan, lanes: usize) usize {
    return plan.m * 2 * lanes * @sizeOf(f32);
}

/// Type-erased lane-specific group processor + its lane width.
pub const FftKernels = struct {
    lanes: usize,
    scratch_alignment: usize,
    process_group: *const fn (
        plan: *const Plan,
        in_group: []const f32,
        out_group: []f32,
        count: usize,
        scratch: []u8,
    ) FftError!void,
};

/// Comptime factory for a lane-width variant. `lanes` is the across-frame SIMD
/// vector width (typically 4 / 8 / 16).
pub fn Kernel(comptime lanes: usize) type {
    return struct {
        pub const LANES: usize = lanes;
        const V = @Vector(lanes, f32);
        const Cplx = struct { re: V, im: V };
        pub const ScratchAlignment: usize = @alignOf(Cplx);

        fn scratchAsCplx(plan: *const Plan, scratch: []u8) FftError![]Cplx {
            if ((@intFromPtr(scratch.ptr) & (ScratchAlignment - 1)) != 0) return FftError.InvalidArgument;
            const need: usize = plan.m * @sizeOf(Cplx);
            if (scratch.len < need) return FftError.InvalidArgument;
            const aligned: []align(ScratchAlignment) u8 = @alignCast(scratch[0..need]);
            return std.mem.bytesAsSlice(Cplx, aligned);
        }

        /// Process up to `lanes` frames at once.
        ///
        /// `in_group` is `count * n_fft` reals, frame-major (frame `f` occupies
        /// `in_group[f*n_fft ..][0..n_fft]`). `out_group` is `count * (2*bins)`
        /// reals, packed per frame as `[0..bins)` real then `[bins..2*bins)`
        /// imaginary. `count` must be in `[1, lanes]`.
        pub fn processGroup(
            plan: *const Plan,
            in_group: []const f32,
            out_group: []f32,
            count: usize,
            scratch: []u8,
        ) FftError!void {
            if (count == 0 or count > lanes) return FftError.InvalidArgument;
            const n_fft: usize = plan.n_fft;
            const m: usize = plan.m;
            const bins: usize = plan.bins;
            if (in_group.len < count * n_fft) return FftError.InvalidArgument;
            if (out_group.len < count * 2 * bins) return FftError.InvalidArgument;

            const z: []Cplx = try scratchAsCplx(plan, scratch);

            // Load + bit-reverse: z[bit_rev[n]] = (x[2n], x[2n+1]) packed across lanes.
            // Lanes >= count are zero-padded so their butterflies stay finite.
            // Marshal through arrays since @Vector lvalues can't be indexed at runtime.
            for (0..m) |n| {
                var re_arr: [lanes]f32 = .{0.0} ** lanes;
                var im_arr: [lanes]f32 = .{0.0} ** lanes;
                for (0..count) |l| {
                    const base: usize = l * n_fft + 2 * n;
                    re_arr[l] = in_group[base];
                    im_arr[l] = in_group[base + 1];
                }
                z[plan.bit_rev[n]] = .{ .re = re_arr, .im = im_arr };
            }

            // Iterative radix-2 DIT complex FFT of size m (in place).
            var len: usize = 2;
            while (len <= m) : (len <<= 1) {
                const half: usize = len >> 1;
                const step: usize = m / len; // twiddle table stride for this stage
                var i: usize = 0;
                while (i < m) : (i += len) {
                    var j: usize = 0;
                    while (j < half) : (j += 1) {
                        const t: usize = j * step;
                        const wr: V = @splat(plan.cfft_cos[t]);
                        const wi: V = @splat(plan.cfft_sin[t]);

                        const a: Cplx = z[i + j];
                        const b: Cplx = z[i + j + half];
                        // bt = b * w  (w = wr + i*wi)
                        const btr: V = b.re * wr - b.im * wi;
                        const bti: V = b.re * wi + b.im * wr;

                        z[i + j] = .{ .re = a.re + btr, .im = a.im + bti };
                        z[i + j + half] = .{ .re = a.re - btr, .im = a.im - bti };
                    }
                }
            }

            // Split recombination: recover one-sided bins X[0..m] from Z[0..m).
            const half_v: V = @splat(0.5);

            // k = 0 and k = m (Nyquist) are purely real.
            {
                const z0: Cplx = z[0];
                const x0r: [lanes]f32 = z0.re + z0.im; // sum of all samples
                const xnr: [lanes]f32 = z0.re - z0.im; // alternating sum
                for (0..count) |l| {
                    out_group[l * 2 * bins + 0] = x0r[l];
                    out_group[l * 2 * bins + 0 + bins] = 0.0;
                    out_group[l * 2 * bins + m] = xnr[l];
                    out_group[l * 2 * bins + m + bins] = 0.0;
                }
            }

            var k: usize = 1;
            while (k < m) : (k += 1) {
                const zk: Cplx = z[k];
                const zmk: Cplx = z[m - k];

                // Xe = (Z[k] + conj(Z[m-k])) / 2
                const xer: V = (zk.re + zmk.re) * half_v;
                const xei: V = (zk.im - zmk.im) * half_v;
                // A = Z[k] - conj(Z[m-k]);  Xo = A / (2i) = (Im(A)/2, -Re(A)/2)
                const ar: V = zk.re - zmk.re;
                const ai: V = zk.im + zmk.im;
                const xor_: V = ai * half_v;
                const xoi: V = -(ar * half_v);

                const wr: V = @splat(plan.split_cos[k]);
                const wi: V = @splat(plan.split_sin[k]);
                // X[k] = Xe + W * Xo
                const xr: [lanes]f32 = xer + (wr * xor_ - wi * xoi);
                const xi: [lanes]f32 = xei + (wr * xoi + wi * xor_);

                for (0..count) |l| {
                    out_group[l * 2 * bins + k] = xr[l];
                    out_group[l * 2 * bins + k + bins] = xi[l];
                }
            }
        }
    };
}

/// Wrap a lane-width variant in the type-erased `FftKernels` struct.
pub fn kernels(comptime lanes: usize) FftKernels {
    const K = Kernel(lanes);
    return .{
        .lanes = lanes,
        .scratch_alignment = K.ScratchAlignment,
        .process_group = K.processGroup,
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Naive one-sided DFT reference: out[k] = sum_n x[n] * exp(-2*pi*i*k*n/N).
fn referenceRfft(x: []const f32, out_re: []f32, out_im: []f32) void {
    const n: usize = x.len;
    const bins: usize = n / 2 + 1;
    const n_f: f64 = @floatFromInt(n);
    for (0..bins) |k| {
        var sr: f64 = 0;
        var si: f64 = 0;
        for (0..n) |nn| {
            const ang: f64 = -2.0 * std.math.pi * @as(f64, @floatFromInt(k)) *
                @as(f64, @floatFromInt(nn)) / n_f;
            sr += @as(f64, x[nn]) * @cos(ang);
            si += @as(f64, x[nn]) * @sin(ang);
        }
        out_re[k] = @floatCast(sr);
        out_im[k] = @floatCast(si);
    }
}

fn runRfftCase(comptime lanes: usize, allocator: std.mem.Allocator, n_fft: usize, count: usize, seed: u64) !void {
    var plan = try Plan.init(allocator, n_fft);
    defer plan.deinit();

    const bins: usize = plan.bins;
    const in_buf = try allocator.alloc(f32, count * n_fft);
    defer allocator.free(in_buf);
    const out_buf = try allocator.alloc(f32, count * 2 * bins);
    defer allocator.free(out_buf);

    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    for (in_buf) |*v| v.* = rnd.float(f32) * 2.0 - 1.0;

    const scratch = try allocator.alignedAlloc(u8, .@"64", scratchBytes(&plan, lanes));
    defer allocator.free(scratch);

    try Kernel(lanes).processGroup(&plan, in_buf, out_buf, count, scratch);

    const ref_re = try allocator.alloc(f32, bins);
    defer allocator.free(ref_re);
    const ref_im = try allocator.alloc(f32, bins);
    defer allocator.free(ref_im);

    for (0..count) |l| {
        referenceRfft(in_buf[l * n_fft ..][0..n_fft], ref_re, ref_im);
        const tol: f32 = 1e-3 * @as(f32, @floatFromInt(n_fft));
        for (0..bins) |k| {
            try std.testing.expectApproxEqAbs(ref_re[k], out_buf[l * 2 * bins + k], tol);
            try std.testing.expectApproxEqAbs(ref_im[k], out_buf[l * 2 * bins + k + bins], tol);
        }
    }
}

test "fft: rfft matches naive DFT across sizes" {
    const allocator = std.testing.allocator;
    inline for (.{ 8, 16, 256, 512 }) |n| {
        try runRfftCase(8, allocator, n, 3, 0x1234 + n);
    }
}

test "fft: lane-width variants agree" {
    const allocator = std.testing.allocator;
    inline for (.{ 4, 8, 16 }) |lanes| {
        try runRfftCase(lanes, allocator, 256, lanes, 0xABCD + lanes);
    }
}

test "fft: rejects non-power-of-two and tiny sizes" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(FftError.NotPowerOfTwo, Plan.init(allocator, 6));
    try std.testing.expectError(FftError.SizeTooSmall, Plan.init(allocator, 2));
}
