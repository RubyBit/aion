// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const types = @import("../../types.zig");
const simd = @import("simd.zig");

const BackendError = types.BackendError;
const MatMulParams = types.MatMulParams;

pub const Tuning = struct {
    /// SIMD lane width selected by `matmul_nt_registry`.
    lanes: usize,

    /// Preferred micro-tile along N for this lane group. The current f32 kernel
    /// only needs `lanes`, but keeping `nr` here makes this the shared NT tuning
    /// contract for future f16/q8/fused variants.
    nr: usize,
};

/// Compute `C = alpha * A @ B^T + beta * C` for one N tile, where A is `[m, k]` f32
/// (row-major over K) and B is `[n, k]` f32 (row-major over K, i.e. already transposed
/// w.r.t. a standard matmul). Directly streams B rows — no pack step, since B's natural
/// layout is the one we want.
///
/// See `matmul_nt_registry.MatMulNtF32Fn` for the parameter contract. Parallelism is the
/// caller's business (it splits N into tiles and hands each one its own slices). Inside
/// we SIMD-unroll over K with four independent FMA accumulators and prefetch the next B
/// row for latency hiding in the `m == 1` decode fast path.
fn matmulNtF32Impl(
    comptime LANES: comptime_int,
    params: MatMulParams,
    c_bytes: []u8,
    a_bytes: []const u8,
    b_bytes: []const u8,
) BackendError!void {
    const m: usize = params.m;
    const n: usize = params.n;
    const k: usize = params.k;
    const ldc: usize = if (params.ldc == 0) n else params.ldc;
    if (ldc < n) return BackendError.InvalidArgument;
    if (m == 0 or n == 0) return;

    const c_all: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, c_bytes);
    const a_all: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, a_bytes);
    const b_all: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, b_bytes);

    if (a_all.len < m * k) return BackendError.InvalidArgument;
    if (b_all.len < n * k) return BackendError.InvalidArgument;
    if (c_all.len < (m - 1) * ldc + n) return BackendError.InvalidArgument;

    const alpha: f32 = params.alpha;
    const beta: f32 = params.beta;

    const VF = @Vector(LANES, f32);
    const vec_k: usize = k - (k % (4 * LANES));

    const rowDot = struct {
        fn run(a_row: [*]align(1) const f32, b_row: [*]align(1) const f32, k_total: usize, k_vec: usize) f32 {
            var vacc0: VF = @splat(0.0);
            var vacc1: VF = @splat(0.0);
            var vacc2: VF = @splat(0.0);
            var vacc3: VF = @splat(0.0);

            var kk: usize = 0;
            while (kk < k_vec) : (kk += 4 * LANES) {
                const a0: VF = @as(*align(1) const VF, @ptrCast(a_row + kk + 0 * LANES)).*;
                const a1: VF = @as(*align(1) const VF, @ptrCast(a_row + kk + 1 * LANES)).*;
                const a2: VF = @as(*align(1) const VF, @ptrCast(a_row + kk + 2 * LANES)).*;
                const a3: VF = @as(*align(1) const VF, @ptrCast(a_row + kk + 3 * LANES)).*;

                const b0: VF = @as(*align(1) const VF, @ptrCast(b_row + kk + 0 * LANES)).*;
                const b1: VF = @as(*align(1) const VF, @ptrCast(b_row + kk + 1 * LANES)).*;
                const b2: VF = @as(*align(1) const VF, @ptrCast(b_row + kk + 2 * LANES)).*;
                const b3: VF = @as(*align(1) const VF, @ptrCast(b_row + kk + 3 * LANES)).*;

                vacc0 = @mulAdd(VF, a0, b0, vacc0);
                vacc1 = @mulAdd(VF, a1, b1, vacc1);
                vacc2 = @mulAdd(VF, a2, b2, vacc2);
                vacc3 = @mulAdd(VF, a3, b3, vacc3);
            }
            var acc: f32 = @reduce(.Add, vacc0) + @reduce(.Add, vacc1) + @reduce(.Add, vacc2) + @reduce(.Add, vacc3);
            while (kk < k_total) : (kk += 1) {
                acc += a_row[kk] * b_row[kk];
            }
            return acc;
        }
    }.run;

    if (m == 1) {
        var j: usize = 0;
        while (j < n) : (j += 1) {
            if (j + 1 < n) {
                @prefetch(b_all.ptr + (j + 1) * k, .{ .rw = .read, .locality = 3, .cache = .data });
            }

            const acc = rowDot(a_all.ptr, b_all.ptr + j * k, k, vec_k);
            c_all[j] = if (beta == 0.0) alpha * acc else alpha * acc + beta * c_all[j];
        }
        return;
    }

    var j: usize = 0;
    while (j < n) : (j += 1) {
        const b_row: [*]align(1) const f32 = b_all.ptr + j * k;

        var i: usize = 0;
        while (i < m) : (i += 1) {
            const acc = rowDot(a_all.ptr + i * k, b_row, k, vec_k);
            const c_idx: usize = i * ldc + j;
            c_all[c_idx] = if (beta == 0.0) alpha * acc else alpha * acc + beta * c_all[c_idx];
        }
    }
}

pub fn Kernel(comptime t: Tuning) type {
    return struct {
        pub fn matmulNtF32(
            params: MatMulParams,
            c_bytes: []u8,
            a_bytes: []const u8,
            b_bytes: []const u8,
        ) BackendError!void {
            return matmulNtF32Impl(t.lanes, params, c_bytes, a_bytes, b_bytes);
        }
    };
}
