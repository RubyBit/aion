// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const types = @import("../../types.zig");
const simd = @import("simd.zig");
const matvec_registry = @import("../registry/matvec_registry.zig");

const BackendError = types.BackendError;
const MatMulParams = types.MatMulParams;
const Tuning = matvec_registry.Tuning;

/// A parameterized matvec kernel generator.
///
/// This targets transformer-style GEMV: M == 1.
/// We keep it fully comptime-specialized so the hot loop remains vectorized.
/// Widest column block the shared accumulator below serves. One block that
/// covers the whole row keeps the walk of B perfectly sequential, which is worth
/// far more than any blocking: 72 GB/s against 11 GB/s for a 256-wide block.
pub const NC_MAX: usize = 4096;

/// Shared by every kernel instantiation — one 16 KiB block per worker rather
/// than one per tuning, and too large to leave on the stack.
threadlocal var acc_buf: [NC_MAX]f32 align(64) = undefined;




pub fn Kernel(comptime t: Tuning) type {
    return struct {
        pub const LANES: usize = t.lanes;
        pub const NR: usize = t.nr;
        pub const NC: usize = @min(t.nc, NC_MAX);
        pub const PREFETCH_K_DIST: usize = t.prefetch_k_dist;

        comptime {
            if (LANES == 0) @compileError("LANES must be > 0");
            if (NR != 2 * LANES) @compileError("NR must equal 2*LANES");
            if (NC == 0) @compileError("NC must be > 0");
        }

        pub fn matvecF32(params: MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) BackendError!void {
            return matvecF32Range(params, 0, params.n, c_bytes, a_bytes, b_bytes);
        }

        pub fn matvecF16(params: MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) BackendError!void {
            return matvecF16Range(params, 0, params.n, c_bytes, a_bytes, b_bytes);
        }

        /// Column block wider than the register file: the running sums live in
        /// `acc` and are re-read each k.
        fn accumulateInMemory(acc: []f32, a_all: []align(1) const f32, b_all: []align(1) const f32, k: usize, n_total: usize, j0: usize, nc: usize) void {
            @setRuntimeSafety(false);
            const VecF = @Vector(LANES, f32);
            @memset(acc[0..nc], 0.0);
            var kk: usize = 0;
            while (kk < k) : (kk += 1) {
                const a_v: VecF = @splat(a_all[kk]);
                const b_ptr: [*]align(1) const f32 = b_all.ptr + kk * n_total + j0;

                // Only a block narrower than the row leaves a gap for the
                // hardware prefetcher to miss; on a straight walk this fights it.
                if (PREFETCH_K_DIST != 0 and nc < n_total and kk + PREFETCH_K_DIST < k) {
                    const pf: [*]align(1) const f32 = b_all.ptr + (kk + PREFETCH_K_DIST) * n_total + j0;
                    @prefetch(@as([*]const u8, @ptrCast(pf)), .{ .rw = .read, .locality = 3, .cache = .data });
                }

                var j: usize = 0;
                while (j + LANES <= nc) : (j += LANES) {
                    const b_v: VecF = @as(*align(1) const VecF, @ptrCast(b_ptr + j)).*;
                    acc[j..][0..LANES].* = @mulAdd(VecF, a_v, b_v, acc[j..][0..LANES].*);
                }
                while (j < nc) : (j += 1) acc[j] = @mulAdd(f32, a_all[kk], b_ptr[j], acc[j]);
            }
        }

        pub fn matvecF32Range(
            params: MatMulParams,
            col_start: usize,
            col_count: usize,
            c_bytes: []u8,
            a_bytes: []const u8,
            b_bytes: []const u8,
        ) BackendError!void {
            if (params.m != 1) return BackendError.InvalidArgument;
            const n_total: usize = params.n;
            const k: usize = params.k;
            if (col_start > n_total or col_count > n_total - col_start) return BackendError.InvalidArgument;

            const c_all: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, c_bytes);
            const a_all: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, a_bytes);
            const b_all: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, b_bytes);

            if (c_all.len < n_total) return BackendError.InvalidArgument;
            if (a_all.len < k) return BackendError.InvalidArgument;
            if (b_all.len < k * n_total) return BackendError.InvalidArgument;

            const alpha: f32 = params.alpha;
            const beta: f32 = params.beta;

            const VecF = @Vector(LANES, f32);
            const alpha_v: VecF = @splat(alpha);
            const beta_v: VecF = @splat(beta);

            // Accumulate a whole column block in one place and walk k on the
            // outside, so every k reads a CONTIGUOUS run of B. Holding the
            // accumulators in registers instead would mean a narrow column strip
            // per pass, and B's rows are `n_total` apart: that touches a fresh
            // cache line per k and uses a fraction of it, which measured ~21 GB/s
            // against the 80 GB/s one core can stream.
            const acc = acc_buf[0..NC];

            var jc: usize = 0;
            while (jc < col_count) : (jc += NC) {
                const nc: usize = @min(NC, col_count - jc);
                const j0: usize = col_start + jc;

                accumulateInMemory(acc, a_all, b_all, k, n_total, j0, nc);

                const c_ptr: [*]align(1) f32 = c_all.ptr + j0;
                var j: usize = 0;
                while (j + LANES <= nc) : (j += LANES) {
                    const a_acc: VecF = acc[j..][0..LANES].*;
                    const old: VecF = @as(*align(1) const VecF, @ptrCast(c_ptr + j)).*;
                    const res: VecF = if (beta == 0.0) (alpha_v * a_acc) else @mulAdd(VecF, alpha_v, a_acc, beta_v * old);
                    @as(*align(1) VecF, @ptrCast(c_ptr + j)).* = res;
                }
                while (j < nc) : (j += 1) {
                    const old: f32 = c_ptr[j];
                    c_ptr[j] = alpha * acc[j] + (if (beta == 0.0) 0.0 else beta * old);
                }
            }
        }

        /// f16 operands with f32 accumulation. Same contiguous-B walk as the f32
        /// kernel — see `matvecF32Range` for why k is on the outside.
        pub fn matvecF16Range(
            params: MatMulParams,
            col_start: usize,
            col_count: usize,
            c_bytes: []u8,
            a_bytes: []const u8,
            b_bytes: []const u8,
        ) BackendError!void {
            if (params.m != 1) return BackendError.InvalidArgument;
            const n_total: usize = params.n;
            const k: usize = params.k;
            if (col_start > n_total or col_count > n_total - col_start) return BackendError.InvalidArgument;

            const c_all: []align(1) f16 = simd.bytesAsSliceMutUnaligned(f16, c_bytes);
            const a_all: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, a_bytes);
            const b_all: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, b_bytes);

            if (c_all.len < n_total) return BackendError.InvalidArgument;
            if (a_all.len < k) return BackendError.InvalidArgument;
            if (b_all.len < k * n_total) return BackendError.InvalidArgument;

            const alpha: f32 = params.alpha;
            const beta: f32 = params.beta;

            const VecF = @Vector(LANES, f32);
            const VecH = @Vector(LANES, f16);
            const alpha_v: VecF = @splat(alpha);
            const beta_v: VecF = @splat(beta);

            const acc = acc_buf[0..NC];

            var jc: usize = 0;
            while (jc < col_count) : (jc += NC) {
                const nc: usize = @min(NC, col_count - jc);
                const j0: usize = col_start + jc;
                @memset(acc[0..nc], 0.0);

                @setRuntimeSafety(false);
                var kk: usize = 0;
                while (kk < k) : (kk += 1) {
                    const a_v: VecF = @splat(@as(f32, a_all[kk]));
                    const b_ptr: [*]align(1) const f16 = b_all.ptr + kk * n_total + j0;

                    if (PREFETCH_K_DIST != 0 and nc < n_total and kk + PREFETCH_K_DIST < k) {
                        const pf: [*]align(1) const f16 = b_all.ptr + (kk + PREFETCH_K_DIST) * n_total + j0;
                        @prefetch(@as([*]const u8, @ptrCast(pf)), .{ .rw = .read, .locality = 3, .cache = .data });
                    }

                    var j: usize = 0;
                    while (j + LANES <= nc) : (j += LANES) {
                        const b_h: VecH = @as(*align(1) const VecH, @ptrCast(b_ptr + j)).*;
                        const b_v: VecF = @floatCast(b_h);
                        const cur: VecF = acc[j..][0..LANES].*;
                        acc[j..][0..LANES].* = @mulAdd(VecF, a_v, b_v, cur);
                    }
                    while (j < nc) : (j += 1) acc[j] = @mulAdd(f32, @as(f32, a_all[kk]), @as(f32, b_ptr[j]), acc[j]);
                }
                @setRuntimeSafety(true);

                const c_ptr: [*]align(1) f16 = c_all.ptr + j0;
                var j: usize = 0;
                while (j + LANES <= nc) : (j += LANES) {
                    const a_acc: VecF = acc[j..][0..LANES].*;
                    const old_h: VecH = @as(*align(1) const VecH, @ptrCast(c_ptr + j)).*;
                    const old: VecF = @floatCast(old_h);
                    const res: VecF = if (beta == 0.0) (alpha_v * a_acc) else @mulAdd(VecF, alpha_v, a_acc, beta_v * old);
                    @as(*align(1) VecH, @ptrCast(c_ptr + j)).* = @floatCast(res);
                }
                while (j < nc) : (j += 1) {
                    const old: f32 = @floatCast(c_ptr[j]);
                    c_ptr[j] = @floatCast(alpha * acc[j] + (if (beta == 0.0) 0.0 else beta * old));
                }
            }
        }
    };
}

const std = @import("std");

/// `c[j] = alpha * sum_k a[k]*b[k*n+j] + beta*c[j]`, over `[col_start, +col_count)`.
fn refMatvec(comptime T: type, n: usize, k: usize, a: []const T, b: []const T, c: []T, col_start: usize, col_count: usize, alpha: f32, beta: f32) void {
    for (col_start..col_start + col_count) |j| {
        var acc: f32 = 0;
        for (0..k) |kk| acc += @as(f32, a[kk]) * @as(f32, b[kk * n + j]);
        const old: f32 = @floatCast(c[j]);
        c[j] = @floatCast(alpha * acc + beta * old);
    }
}

fn checkMatvec(comptime T: type, comptime K: type, n: usize, k: usize, col_start: usize, col_count: usize, alpha: f32, beta: f32, tol: f32) !void {
    const alloc = std.testing.allocator;
    const a = try alloc.alloc(T, k);
    defer alloc.free(a);
    const b = try alloc.alloc(T, k * n);
    defer alloc.free(b);
    const got = try alloc.alloc(T, n);
    defer alloc.free(got);
    const want = try alloc.alloc(T, n);
    defer alloc.free(want);

    var rng = std.Random.DefaultPrng.init(n * 7919 + k * 104729 + col_count);
    const r = rng.random();
    for (a) |*v| v.* = @floatCast(r.float(f32) - 0.5);
    for (b) |*v| v.* = @floatCast(r.float(f32) - 0.5);
    for (got, want, 0..) |*g, *w, i| {
        const seed: T = @floatCast(@as(f32, @floatFromInt(i % 5)) * 0.25);
        g.* = seed;
        w.* = seed;
    }

    refMatvec(T, n, k, a, b, want, col_start, col_count, alpha, beta);
    const params: MatMulParams = .{ .m = 1, .n = n, .k = k, .alpha = alpha, .beta = beta };
    if (T == f32) {
        try K.matvecF32Range(params, col_start, col_count, std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
    } else {
        try K.matvecF16Range(params, col_start, col_count, std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
    }
    for (got, want) |g, w| try std.testing.expect(@abs(@as(f32, g) - @as(f32, w)) <= tol);
}

test "matvec f32 and f16 match a reference across widths, ranges and alpha/beta" {
    const K = Kernel(.{ .nr = 8, .lanes = 4, .nc = 256, .prefetch_k_dist = 4 });
    // n spans the column block boundary and lands off a lane multiple; the
    // ranges exercise the partial-column entry points the threaded path uses.
    const cases = [_][4]usize{
        .{ 8, 4, 0, 8 },
        .{ 256, 32, 0, 256 }, // exactly one block
        .{ 300, 17, 0, 300 }, // more than one block, ragged tail
        .{ 300, 17, 256, 44 }, // the tail block alone
        .{ 130, 64, 8, 100 }, // offset range, not lane aligned
        .{ 512, 5, 128, 256 },
        .{ 7, 3, 0, 7 }, // narrower than one vector
    };
    for (cases) |c| {
        for ([_][2]f32{ .{ 1.0, 0.0 }, .{ 1.0, 1.0 }, .{ 0.5, -0.25 } }) |ab| {
            try checkMatvec(f32, K, c[0], c[1], c[2], c[3], ab[0], ab[1], 1e-4);
            try checkMatvec(f16, K, c[0], c[1], c[2], c[3], ab[0], ab[1], 2e-2);
        }
    }
}
