// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Quantized matrix-vector (small-M GEMV) kernels — the q8_0 counterpart to the
// f32 `matvec.zig`. Used for autoregressive decode (M = a few rows), where B is
// streamed once from memory rather than packed. Selected via `matvec_registry`.

const std = @import("std");
const types = @import("../../types.zig");
const simd = @import("simd.zig");
const matmul_q_i8 = @import("matmul_q_i8.zig");

const BackendError = types.BackendError;
const MatMulParams = types.MatMulParams;

/// q8_0 block: 32 elements stored as 2-byte f16 scale + 32 int8s.
pub const Q8_0_BLOCK_ELEMS: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 34;

pub const MatvecTuning = struct {
    /// SIMD lane width selected by `matvec_registry`.
    lanes: usize,
    /// The byte dot the prepared activation is multiplied with.
    dot_enc: matmul_q_i8.DotEnc,
};

fn matvecQ8_0KMajorDotAccumulateImpl(
    comptime enc: matmul_q_i8.DotEnc,
    params: MatMulParams,
    c_bytes: []u8,
    a_bytes: []const u8,
    b_bytes: []const u8,
    acc_bytes: []align(32) u8,
    prepared_a: []align(32) u8,
    prepare_a: bool,
    first_k_tile: bool,
    last_k_tile: bool,
) BackendError!void {
    if (params.m != 1 or (params.k % Q8_0_BLOCK_ELEMS) != 0) return BackendError.InvalidArgument;
    const n = params.n;
    const k = params.k;
    const blocks = k / Q8_0_BLOCK_ELEMS;
    // A folded block dot is four lanes wide (see `matmul_q_i8.dotI8Narrow`), and
    // the running sums that consume it follow.
    const VF = @Vector(4, f32);
    const VI = @Vector(4, i32);
    const scratch_need = n * @sizeOf(VF);
    const prepared_block_bytes = matmul_q_i8.PREP_BLOCK_BYTES;
    const prepared_need = blocks * prepared_block_bytes;
    if (acc_bytes.len < scratch_need) return BackendError.InvalidArgument;
    if (prepared_a.len < prepared_need) return BackendError.InvalidArgument;
    if (a_bytes.len < k * @sizeOf(f32) or b_bytes.len < blocks * n * Q8_0_BLOCK_BYTES) return BackendError.InvalidArgument;
    if (last_k_tile and c_bytes.len < n * @sizeOf(f32)) return BackendError.InvalidArgument;

    const a = simd.bytesAsSliceConstUnaligned(f32, a_bytes);
    const c = simd.bytesAsSliceMutUnaligned(f32, c_bytes);
    const acc: []align(32) VF = @alignCast(std.mem.bytesAsSlice(VF, acc_bytes[0..scratch_need]));

    // Quantise this K tile's activation once, then sweep columns in tiles small
    // enough to keep their running sums in registers. Accumulating straight into
    // `acc` instead costs a 32-byte read and write per column per block — 64
    // bytes of accumulator traffic for every 34 bytes of weight, which is more
    // than the weights themselves.
    if (prepare_a) matmul_q_i8.prepareARow(prepared_a.ptr, a.ptr, blocks);

    const NT: usize = 4;
    var jt: usize = 0;
    while (jt + NT <= n) : (jt += NT) {
        var reg: [NT]VF = undefined;
        inline for (0..NT) |jj| reg[jj] = if (first_k_tile) @splat(0.0) else acc[jt + jj];
        var kb: usize = 0;
        while (kb < blocks) : (kb += 1) {
            const slot = prepared_a[kb * prepared_block_bytes ..];
            const aqv: @Vector(32, i8) = @as(*align(1) const @Vector(32, i8), @ptrCast(slot.ptr)).*;
            const a_scale: f32 = @as(*align(1) const f32, @ptrCast(slot.ptr + Q8_0_BLOCK_ELEMS)).*;
            const correction = matmul_q_i8.prepBiasNarrow(enc, slot.ptr);
            const row = b_bytes.ptr + (kb * n + jt) * Q8_0_BLOCK_BYTES;
            inline for (0..NT) |jj| {
                const bp = row + jj * Q8_0_BLOCK_BYTES;
                const bits: u16 = @as(*align(1) const u16, @ptrCast(bp)).*;
                const qv: @Vector(32, i8) = @as(*align(1) const @Vector(32, i8), @ptrCast(bp + 2)).*;
                const dots = matmul_q_i8.dotI8Narrow(enc, @as(VI, @splat(0)), qv, aqv) - correction;
                reg[jj] += @as(VF, @floatFromInt(dots)) * @as(VF, @splat(a_scale * @as(f32, @as(f16, @bitCast(bits)))));
            }
        }
        inline for (0..NT) |jj| acc[jt + jj] = reg[jj];
    }

    while (jt < n) : (jt += 1) {
        var reg: VF = if (first_k_tile) @splat(0.0) else acc[jt];
        var kb: usize = 0;
        while (kb < blocks) : (kb += 1) {
            const slot = prepared_a[kb * prepared_block_bytes ..];
            const aqv: @Vector(32, i8) = @as(*align(1) const @Vector(32, i8), @ptrCast(slot.ptr)).*;
            const a_scale: f32 = @as(*align(1) const f32, @ptrCast(slot.ptr + Q8_0_BLOCK_ELEMS)).*;
            const bp = b_bytes.ptr + (kb * n + jt) * Q8_0_BLOCK_BYTES;
            const bits: u16 = @as(*align(1) const u16, @ptrCast(bp)).*;
            const qv: @Vector(32, i8) = @as(*align(1) const @Vector(32, i8), @ptrCast(bp + 2)).*;
            const dots = matmul_q_i8.dotI8Narrow(enc, @as(VI, @splat(0)), qv, aqv) - matmul_q_i8.prepBiasNarrow(enc, slot.ptr);
            reg += @as(VF, @floatFromInt(dots)) * @as(VF, @splat(a_scale * @as(f32, @as(f16, @bitCast(bits)))));
        }
        acc[jt] = reg;
    }

    if (last_k_tile) {
        var j: usize = 0;
        while (j < n) : (j += 1) {
            const dot = @reduce(.Add, acc[j]);
            c[j] = if (params.beta == 0.0) params.alpha * dot else params.alpha * dot + params.beta * c[j];
        }
    }
}

/// Deepest K a single-call dot matvec quantises its activation for up front.
const DOT_MAX_K: usize = 8192;
const PREP_BLOCK_BYTES: usize = matmul_q_i8.PREP_BLOCK_BYTES;

threadlocal var dot_prep: [(DOT_MAX_K / Q8_0_BLOCK_ELEMS) * PREP_BLOCK_BYTES]u8 align(32) = undefined;

inline fn storeDot(c: []align(1) f32, params: MatMulParams, idx: usize, dot: f32) void {
    c[idx] = if (params.beta == 0.0) params.alpha * dot else params.alpha * dot + params.beta * c[idx];
}

/// q8 matvec with integer dot products and register-resident sums; the form
/// every M and K takes in one call.
///
/// The accumulating form above parks one vector per column in memory and reads
/// and writes it for every block — 64 bytes of accumulator traffic per 34 bytes
/// of weight, which is what made it lose as the whole-call kernel.
///
/// A column tile narrow enough to stay in the register file fixes that, and the
/// activation is quantised once per call rather than once per column tile, so B
/// is the only thing streamed. Whole tiles run unconditionally: a `jj < nt` test
/// inside the unrolled body makes every accumulator write conditional, and the
/// sums then spill to memory, which is the whole point of tiling them.
///
/// Tiling columns does stride B by `n * 34`, and reading it straight through is
/// worth 40% on the loads alone. Both ways of holding the sums that a contiguous
/// sweep needs lose far more than that, measured on a 1B q8 decode against 32
/// tok/s for this kernel:
///
///   contiguous, a scalar sum per column   4.1 tok/s  reduces every block's dot
///   contiguous, a vector sum per column   4.3 tok/s  16 KiB of L1 traffic, 64
///                                                    bytes for every 34 of B
///
/// The sums' residency dominates the read pattern, and the register file holds
/// only a few columns' worth. Getting contiguous reads *and* register sums needs
/// B stored per output column, `[n][k_blocks]` rather than `[k_blocks][n]` — a
/// packing change, not a kernel one.
fn matvecQ8_0DotRegImpl(
    comptime enc: matmul_q_i8.DotEnc,
    comptime NT: usize,
    params: MatMulParams,
    c_bytes: []u8,
    a_bytes: []const u8,
    b_bytes: []const u8,
) BackendError!void {
    const m = params.m;
    const n = params.n;
    const k = params.k;
    const blocks = k / Q8_0_BLOCK_ELEMS;

    const a = simd.bytesAsSliceConstUnaligned(f32, a_bytes);
    const c = simd.bytesAsSliceMutUnaligned(f32, c_bytes);
    if (a.len < m * k or c.len < m * n) return BackendError.InvalidArgument;
    if (b_bytes.len < blocks * n * Q8_0_BLOCK_BYTES) return BackendError.InvalidArgument;

    // One row at a time, its activation quantised in chunks of K the buffer holds:
    // every M and K gets the same integer dot, rather than a deep or multi-row call
    // quietly falling back to f32. A row past the first re-reads B, which only the
    // shapes a decode never issues pay.
    const chunk_blocks: usize = DOT_MAX_K / Q8_0_BLOCK_ELEMS;
    for (0..m) |r| {
        const c_row = c[r * n ..][0..n];
        var kb0: usize = 0;
        while (kb0 < blocks) : (kb0 += chunk_blocks) {
            const nb = @min(chunk_blocks, blocks - kb0);
            matmul_q_i8.prepareARow(&dot_prep, a.ptr + r * k + kb0 * Q8_0_BLOCK_ELEMS, nb);
            const first = kb0 == 0;
            dotColumns(enc, NT, c_row, params, first, b_bytes.ptr + kb0 * n * Q8_0_BLOCK_BYTES, n, nb);
        }
    }
}

/// Every column of one row over `nb` K blocks of B starting at `b`, against the
/// prepared activation in `dot_prep`. The first chunk stores through alpha and
/// beta; later ones add their share.
fn dotColumns(
    comptime enc: matmul_q_i8.DotEnc,
    comptime NT: usize,
    c: []align(1) f32,
    params: MatMulParams,
    first: bool,
    b: [*]const u8,
    n: usize,
    nb: usize,
) void {
    // A folded block dot is four lanes wide (see `matmul_q_i8.dotI8Narrow`), and
    // the running sums that consume it follow.
    const VF = @Vector(4, f32);
    const VI = @Vector(4, i32);
    const store = struct {
        fn at(cc: []align(1) f32, p: MatMulParams, is_first: bool, idx: usize, dot: f32) void {
            if (is_first) storeDot(cc, p, idx, dot) else cc[idx] += p.alpha * dot;
        }
    }.at;

    var jt: usize = 0;
    while (jt + NT <= n) : (jt += NT) {
        var acc: [NT]VF = @splat(@as(VF, @splat(0.0)));
        var kb: usize = 0;
        while (kb < nb) : (kb += 1) {
            const slot = dot_prep[kb * PREP_BLOCK_BYTES ..];
            const aqv: @Vector(32, i8) = @as(*align(1) const @Vector(32, i8), @ptrCast(slot.ptr)).*;
            const a_scale: f32 = @as(*align(1) const f32, @ptrCast(slot.ptr + Q8_0_BLOCK_ELEMS)).*;
            const correction = matmul_q_i8.prepBiasNarrow(enc, slot.ptr);
            const row = b + (kb * n + jt) * Q8_0_BLOCK_BYTES;
            inline for (0..NT) |jj| {
                const bp = row + jj * Q8_0_BLOCK_BYTES;
                const bits: u16 = @as(*align(1) const u16, @ptrCast(bp)).*;
                const qv: @Vector(32, i8) = @as(*align(1) const @Vector(32, i8), @ptrCast(bp + 2)).*;
                const dots = matmul_q_i8.dotI8Narrow(enc, @as(VI, @splat(0)), qv, aqv) - correction;
                acc[jj] += @as(VF, @floatFromInt(dots)) * @as(VF, @splat(a_scale * @as(f32, @as(f16, @bitCast(bits)))));
            }
        }
        inline for (0..NT) |jj| store(c, params, first, jt + jj, @reduce(.Add, acc[jj]));
    }

    while (jt < n) : (jt += 1) {
        var acc: VF = @splat(0.0);
        var kb: usize = 0;
        while (kb < nb) : (kb += 1) {
            const slot = dot_prep[kb * PREP_BLOCK_BYTES ..];
            const aqv: @Vector(32, i8) = @as(*align(1) const @Vector(32, i8), @ptrCast(slot.ptr)).*;
            const a_scale: f32 = @as(*align(1) const f32, @ptrCast(slot.ptr + Q8_0_BLOCK_ELEMS)).*;
            const bp = b + (kb * n + jt) * Q8_0_BLOCK_BYTES;
            const bits: u16 = @as(*align(1) const u16, @ptrCast(bp)).*;
            const qv: @Vector(32, i8) = @as(*align(1) const @Vector(32, i8), @ptrCast(bp + 2)).*;
            const dots = matmul_q_i8.dotI8Narrow(enc, @as(VI, @splat(0)), qv, aqv) - matmul_q_i8.prepBiasNarrow(enc, slot.ptr);
            acc += @as(VF, @floatFromInt(dots)) * @as(VF, @splat(a_scale * @as(f32, @as(f16, @bitCast(bits)))));
        }
        store(c, params, first, jt, @reduce(.Add, acc));
    }
}

pub fn MatvecKernel(comptime t: MatvecTuning) type {
    return struct {
        pub fn matvecQ8_0KMajor(params: MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) BackendError!void {
            if (params.m == 0) return;
            if ((params.k % Q8_0_BLOCK_ELEMS) != 0) return BackendError.InvalidArgument;
            return matvecQ8_0DotRegImpl(t.dot_enc, 4, params, c_bytes, a_bytes, b_bytes);
        }

        pub fn matvecQ8_0KMajorAccumulate(params: MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8, acc_bytes: []align(32) u8, prepared_a: []align(32) u8, prepare_a: bool, first_k_tile: bool, last_k_tile: bool) BackendError!void {
            return matvecQ8_0KMajorDotAccumulateImpl(t.dot_enc, params, c_bytes, a_bytes, b_bytes, acc_bytes, prepared_a, prepare_a, first_k_tile, last_k_tile);
        }
    };
}

// Past `DOT_MAX_K` the activation is quantized in chunks; a chunk boundary sits on
// a block boundary, so it must change nothing about the integer dot.
test "matvec_q: int8 results hold past the prepared-activation chunk and above M = 1" {
    const testing = std.testing;
    const m: usize = 2;
    const n: usize = 4;
    const k: usize = DOT_MAX_K + 64;
    const blocks = k / Q8_0_BLOCK_ELEMS;

    const a = try testing.allocator.alloc(f32, m * k);
    defer testing.allocator.free(a);
    for (a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 37)) - 18)) * 0.01;
    const b = try testing.allocator.alloc(u8, blocks * n * Q8_0_BLOCK_BYTES);
    defer testing.allocator.free(b);
    for (0..blocks * n) |bi| {
        std.mem.writeInt(u16, b[bi * Q8_0_BLOCK_BYTES ..][0..2], @bitCast(@as(f16, 0.02)), .little);
        for (b[bi * Q8_0_BLOCK_BYTES + 2 ..][0..32], 0..) |*q, t| q.* = @truncate(bi *% 5 +% t *% 3);
    }

    var got: [m * n]f32 = undefined;
    const K = MatvecKernel(.{ .lanes = 4, .dot_enc = .portable });
    try K.matvecQ8_0KMajor(.{ .m = m, .n = n, .k = k }, std.mem.sliceAsBytes(&got), std.mem.sliceAsBytes(a), b);

    for (0..m) |r| {
        for (0..n) |c| {
            var want: f64 = 0;
            for (0..blocks) |kb| {
                var aq: [32]i8 = undefined;
                const a_scale = matmul_q_i8.quantizeABlock(a[r * k + kb * 32 ..].ptr, &aq);
                const bp = b[(kb * n + c) * Q8_0_BLOCK_BYTES ..];
                var dot: i64 = 0;
                for (0..32) |t| dot += @as(i64, aq[t]) * @as(i8, @bitCast(bp[2 + t]));
                const b_scale: f64 = @as(f16, @bitCast(std.mem.readInt(u16, bp[0..2], .little)));
                want += @as(f64, a_scale) * b_scale * @as(f64, @floatFromInt(dot));
            }
            try testing.expectApproxEqRel(want, @as(f64, got[r * n + c]), 1e-5);
        }
    }
}
