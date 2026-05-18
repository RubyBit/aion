// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const types = @import("../../types.zig");

const BackendError = types.BackendError;

pub const Tuning = struct {
    /// SIMD lane width selected by `matmul_nt_registry`.
    lanes: usize,

    /// Preferred micro-tile along N for this lane group. The current f32 kernel
    /// only needs `lanes`, but keeping `nr` here makes this the shared NT tuning
    /// contract for future f16/q8/fused variants.
    nr: usize,
};

/// Compute `C[:, n_start..n_start+n_count] = alpha * A @ B[n_start..n_start+n_count, :]^T + beta*C[...]`
/// where A is M×K f32 (row-major over K) and B is N×K f32 (row-major over K, i.e. already transposed
/// w.r.t. a standard matmul). Directly streams B rows — no pack step, since B's natural layout is the
/// one we want.
///
/// Parallelism is expected to be driven by the caller (splitting over N tiles). Inside this function
/// we SIMD-unroll over K with four independent FMA accumulators and prefetch the next B row for
/// latency hiding in the `m_total == 1` decode fast path.
fn matmulNtF32Impl(
    comptime LANES: comptime_int,
    a_ptr: [*]align(1) const f32,
    b_ptr: [*]align(1) const f32,
    c_ptr: [*]align(1) f32,
    m_total: usize,
    k: usize,
    n_total: usize,
    n_start: usize,
    n_count: usize,
    alpha: f32,
    beta: f32,
) BackendError!void {
    _ = n_total;
    if (n_start != 0 and false) {
        // n_start only used for caller-side offsetting of b_ptr/c_ptr; kernel operates on [M, n_count].
    }

    const VF = @Vector(LANES, f32);

    const vec_k: usize = k - (k % (4 * LANES));

    if (m_total == 1) {
        var j: usize = 0;
        while (j < n_count) : (j += 1) {
            const b_row: [*]align(1) const f32 = b_ptr + j * k;

            if (j + 1 < n_count) {
                @prefetch(b_ptr + (j + 1) * k, .{ .rw = .read, .locality = 3, .cache = .data });
            }

            var vacc0: VF = @splat(0.0);
            var vacc1: VF = @splat(0.0);
            var vacc2: VF = @splat(0.0);
            var vacc3: VF = @splat(0.0);

            var kk: usize = 0;
            while (kk < vec_k) : (kk += 4 * LANES) {
                const a0: VF = @as(*align(1) const VF, @ptrCast(a_ptr + kk + 0 * LANES)).*;
                const a1: VF = @as(*align(1) const VF, @ptrCast(a_ptr + kk + 1 * LANES)).*;
                const a2: VF = @as(*align(1) const VF, @ptrCast(a_ptr + kk + 2 * LANES)).*;
                const a3: VF = @as(*align(1) const VF, @ptrCast(a_ptr + kk + 3 * LANES)).*;

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
            while (kk < k) : (kk += 1) {
                acc += a_ptr[kk] * b_row[kk];
            }

            if (beta == 0.0) {
                c_ptr[j] = alpha * acc;
            } else {
                c_ptr[j] = alpha * acc + beta * c_ptr[j];
            }
        }
        return;
    }

    var j: usize = 0;
    while (j < n_count) : (j += 1) {
        const b_row: [*]align(1) const f32 = b_ptr + j * k;

        var m: usize = 0;
        while (m < m_total) : (m += 1) {
            const a_row: [*]align(1) const f32 = a_ptr + m * k;

            var vacc0: VF = @splat(0.0);
            var vacc1: VF = @splat(0.0);
            var vacc2: VF = @splat(0.0);
            var vacc3: VF = @splat(0.0);

            var kk: usize = 0;
            while (kk < vec_k) : (kk += 4 * LANES) {
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
            while (kk < k) : (kk += 1) {
                acc += a_row[kk] * b_row[kk];
            }

            const c_idx: usize = m * n_count + j;
            if (beta == 0.0) {
                c_ptr[c_idx] = alpha * acc;
            } else {
                c_ptr[c_idx] = alpha * acc + beta * c_ptr[c_idx];
            }
        }
    }
}

pub fn Kernel(comptime t: Tuning) type {
    return struct {
        pub fn matmulNtF32(
            a_ptr: [*]align(1) const f32,
            b_ptr: [*]align(1) const f32,
            c_ptr: [*]align(1) f32,
            m_total: usize,
            k: usize,
            n_total: usize,
            n_start: usize,
            n_count: usize,
            alpha: f32,
            beta: f32,
        ) BackendError!void {
            return matmulNtF32Impl(t.lanes, a_ptr, b_ptr, c_ptr, m_total, k, n_total, n_start, n_count, alpha, beta);
        }
    };
}
