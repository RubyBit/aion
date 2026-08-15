// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const types = @import("../../types.zig");
const matmul_nt = @import("matmul_nt.zig");
const matmul_q = @import("matmul_q.zig");

const BackendError = types.BackendError;
const Q8_0_BLOCK_ELEMS: usize = matmul_q.Q8_0_BLOCK_ELEMS;
const Q8_0_BLOCK_BYTES: usize = matmul_q.Q8_0_BLOCK_BYTES;

/// Compute `C[:, n_start..n_start+n_count] = alpha * A @ B[n_start..n_start+n_count, :]^T + beta*C[...]`
/// where A is M×K f32 (row-major, contiguous over K) and B is N×K q8_0 with one contiguous
/// run of `K/32` blocks per row.
///
/// Unlike the tile-packed quant GEMM kernel, this one reads the on-disk Q8_0 layout directly.
fn matmulNtQ8_0Impl(
    comptime LANES: comptime_int,
    a_ptr: [*]align(1) const f32,
    b_ptr: [*]const u8,
    c_ptr: [*]align(1) f32,
    m_total: usize,
    k: usize,
    n_total: usize,
    n_start: usize,
    n_count: usize,
    alpha: f32,
    beta: f32,
) BackendError!void {
    const blocks_per_row: usize = k / Q8_0_BLOCK_ELEMS;
    const row_bytes: usize = blocks_per_row * Q8_0_BLOCK_BYTES;

    if ((k % Q8_0_BLOCK_ELEMS) != 0) return BackendError.InvalidArgument;
    if (n_start > n_total) return BackendError.InvalidArgument;
    if (n_count > (n_total - n_start)) return BackendError.InvalidArgument;

    comptime {
        if ((Q8_0_BLOCK_ELEMS % LANES) != 0) @compileError("q8 NT kernel requires Q8_0_BLOCK_ELEMS to be divisible by SIMD lanes");
    }

    const chunks_per_block: comptime_int = Q8_0_BLOCK_ELEMS / LANES;
    const VF = @Vector(LANES, f32);
    const VI8 = @Vector(LANES, i8);

    const rowDot = struct {
        fn run(b_row: [*]const u8, a_row: [*]align(1) const f32, block_count: usize) f32 {
            // Keep one independent vector chain per SIMD chunk across the whole K
            // row, then reduce only once.  A scalar reduction per q8 block creates
            // a long dependency chain (48 reductions at K=1536); lane positions
            // are freely additive across blocks, so no intermediate horizontal
            // sum is required.
            var acc: [chunks_per_block]VF = @splat(@as(VF, @splat(0.0)));
            var b: usize = 0;
            while (b < block_count) : (b += 1) {
                const block_ptr = b_row + b * Q8_0_BLOCK_BYTES;
                const scale_bits: u16 = @as(*align(1) const u16, @ptrCast(block_ptr)).*;
                const v_scale: VF = @splat(@as(f32, @as(f16, @bitCast(scale_bits))));
                inline for (0..chunks_per_block) |chunk| {
                    const lane_off: usize = chunk * LANES;
                    const qv: VI8 = @as(*align(1) const VI8, @ptrCast(block_ptr + 2 + lane_off)).*;
                    const av: VF = @as(*align(1) const VF, @ptrCast(a_row + b * Q8_0_BLOCK_ELEMS + lane_off)).*;
                    const qf: VF = @floatFromInt(qv);
                    acc[chunk] = @mulAdd(VF, qf, av * v_scale, acc[chunk]);
                }
            }
            var total: VF = acc[0];
            inline for (1..chunks_per_block) |chunk| total += acc[chunk];
            return @reduce(.Add, total);
        }
    }.run;

    if (m_total == 1) {
        const a_row: [*]align(1) const f32 = a_ptr;

        var j: usize = 0;
        while (j < n_count) : (j += 1) {
            const b_row_base: [*]const u8 = b_ptr + j * row_bytes;

            if (j + 1 < n_count) {
                @prefetch(b_ptr + (j + 1) * row_bytes, .{ .rw = .read, .locality = 3, .cache = .data });
            }

            const acc = rowDot(b_row_base, a_row, blocks_per_row);

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
        const b_row_base: [*]const u8 = b_ptr + j * row_bytes;

        var m: usize = 0;
        while (m < m_total) : (m += 1) {
            const a_row: [*]align(1) const f32 = a_ptr + m * k;

            const acc = rowDot(b_row_base, a_row, blocks_per_row);

            const c_idx: usize = m * n_count + j;
            if (beta == 0.0) {
                c_ptr[c_idx] = alpha * acc;
            } else {
                c_ptr[c_idx] = alpha * acc + beta * c_ptr[c_idx];
            }
        }
    }
}

pub fn Kernel(comptime t: matmul_nt.Tuning) type {
    return struct {
        pub fn matmulNtQ8_0(
            a_ptr: [*]align(1) const f32,
            b_ptr: [*]const u8,
            c_ptr: [*]align(1) f32,
            m_total: usize,
            k: usize,
            n_total: usize,
            n_start: usize,
            n_count: usize,
            alpha: f32,
            beta: f32,
        ) BackendError!void {
            return matmulNtQ8_0Impl(t.lanes, a_ptr, b_ptr, c_ptr, m_total, k, n_total, n_start, n_count, alpha, beta);
        }
    };
}
