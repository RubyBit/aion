// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const types = @import("../../types.zig");
const simd = @import("simd.zig");
const matmul_nt = @import("matmul_nt.zig");
const matmul_q = @import("matmul_q.zig");

const BackendError = types.BackendError;
const MatMulParams = types.MatMulParams;
const Q8_0_BLOCK_ELEMS: usize = matmul_q.Q8_0_BLOCK_ELEMS;
const Q8_0_BLOCK_BYTES: usize = matmul_q.Q8_0_BLOCK_BYTES;

/// Compute `C = alpha * A @ B^T + beta * C` for one N tile, where A is `[m, k]` f32
/// (row-major, contiguous over K) and B is `[n, k]` q8_0 with one contiguous run of
/// `k/32` blocks per row.
///
/// See `matmul_nt_registry.MatMulNtQ8_0Fn` for the parameter contract. Unlike the
/// tile-packed quant GEMM kernel, this one reads the on-disk q8_0 layout directly.
fn matmulNtQ8_0Impl(
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
    if ((k % Q8_0_BLOCK_ELEMS) != 0) return BackendError.InvalidArgument;
    if (m == 0 or n == 0) return;

    const blocks_per_row: usize = k / Q8_0_BLOCK_ELEMS;
    const row_bytes: usize = blocks_per_row * Q8_0_BLOCK_BYTES;

    const c_all: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, c_bytes);
    const a_all: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, a_bytes);

    if (a_all.len < m * k) return BackendError.InvalidArgument;
    if (b_bytes.len < n * row_bytes) return BackendError.InvalidArgument;
    if (c_all.len < (m - 1) * ldc + n) return BackendError.InvalidArgument;

    const alpha: f32 = params.alpha;
    const beta: f32 = params.beta;

    comptime {
        if ((Q8_0_BLOCK_ELEMS % LANES) != 0) @compileError("q8 NT kernel requires Q8_0_BLOCK_ELEMS to be divisible by SIMD lanes");
    }

    const chunks_per_block: comptime_int = Q8_0_BLOCK_ELEMS / LANES;
    const VF = @Vector(LANES, f32);
    const VI8 = @Vector(LANES, i8);

    const rowDot = struct {
        fn run(a_row: [*]align(1) const f32, b_row: [*]const u8, block_count: usize) f32 {
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

    if (m == 1) {
        var j: usize = 0;
        while (j < n) : (j += 1) {
            if (j + 1 < n) {
                @prefetch(b_bytes.ptr + (j + 1) * row_bytes, .{ .rw = .read, .locality = 3, .cache = .data });
            }

            const acc = rowDot(a_all.ptr, b_bytes.ptr + j * row_bytes, blocks_per_row);
            c_all[j] = if (beta == 0.0) alpha * acc else alpha * acc + beta * c_all[j];
        }
        return;
    }

    var j: usize = 0;
    while (j < n) : (j += 1) {
        const b_row: [*]const u8 = b_bytes.ptr + j * row_bytes;

        var i: usize = 0;
        while (i < m) : (i += 1) {
            const acc = rowDot(a_all.ptr + i * k, b_row, blocks_per_row);
            const c_idx: usize = i * ldc + j;
            c_all[c_idx] = if (beta == 0.0) alpha * acc else alpha * acc + beta * c_all[c_idx];
        }
    }
}

pub fn Kernel(comptime t: matmul_nt.Tuning) type {
    return struct {
        pub fn matmulNtQ8_0(
            params: MatMulParams,
            c_bytes: []u8,
            a_bytes: []const u8,
            b_bytes: []const u8,
        ) BackendError!void {
            return matmulNtQ8_0Impl(t.lanes, params, c_bytes, a_bytes, b_bytes);
        }
    };
}
