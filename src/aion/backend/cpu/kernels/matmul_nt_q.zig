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

    const blockDot = struct {
        fn run(block_ptr: [*]const u8, a_base: [*]align(1) const f32) f32 {
            const scale_bits: u16 = @as(*align(1) const u16, @ptrCast(block_ptr)).*;
            const scale: f32 = @as(f32, @as(f16, @bitCast(scale_bits)));
            const v_scale: VF = @splat(scale);

            var acc: f32 = 0.0;
            inline for (0..chunks_per_block) |chunk| {
                const lane_off: usize = chunk * LANES;
                const qv: VI8 = @as(*align(1) const VI8, @ptrCast(block_ptr + 2 + lane_off)).*;
                const av: VF = @as(*align(1) const VF, @ptrCast(a_base + lane_off)).*;
                const qf: VF = @floatFromInt(qv);
                const prod: VF = qf * (av * v_scale);
                acc += @reduce(.Add, prod);
            }
            return acc;
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

            var acc: f32 = 0.0;
            var b: usize = 0;
            while (b < blocks_per_row) : (b += 1) {
                const block_off: usize = b * Q8_0_BLOCK_BYTES;
                acc += blockDot(b_row_base + block_off, a_row + b * Q8_0_BLOCK_ELEMS);
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
        const b_row_base: [*]const u8 = b_ptr + j * row_bytes;

        var m: usize = 0;
        while (m < m_total) : (m += 1) {
            const a_row: [*]align(1) const f32 = a_ptr + m * k;

            var acc: f32 = 0.0;
            var b: usize = 0;
            while (b < blocks_per_row) : (b += 1) {
                const block_off: usize = b * Q8_0_BLOCK_BYTES;
                acc += blockDot(b_row_base + block_off, a_row + b * Q8_0_BLOCK_ELEMS);
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
