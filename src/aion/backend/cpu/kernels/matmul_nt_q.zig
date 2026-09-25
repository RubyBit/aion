// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! `C = alpha * A @ B^T + beta * C` against a q8_0 B stored `[n, k]`
//! with each row one run of `k / 32` blocks, in either `types.QuantBlockOrder`.
//!
//! A is quantized once per call (`prepareActivation`) and every sweep dots it with
//! the tier's byte dot, so every shape, block order and thread split multiplies alike.

const std = @import("std");
const builtin = @import("builtin");
const types = @import("../../types.zig");
const simd = @import("simd.zig");
const matmul_nt = @import("matmul_nt.zig");
const matmul_q_i8 = @import("matmul_q_i8.zig");

const BackendError = types.BackendError;
const MatMulParams = types.MatMulParams;
const Q8_0_BLOCK_ELEMS: usize = matmul_q_i8.Q8_0_BLOCK_ELEMS;
const Q8_0_BLOCK_BYTES: usize = matmul_q_i8.Q8_0_BLOCK_BYTES;
const PREP_BLOCK_BYTES: usize = matmul_q_i8.PREP_BLOCK_BYTES;

/// A folded block dot is four lanes wide (see `matmul_q_i8.dotI8Narrow`), and the
/// sums that consume it follow.
const VF = @Vector(4, f32);
const VI = @Vector(4, i32);
const VQ = @Vector(32, i8);

const DotEnc = matmul_q_i8.DotEnc;

/// One block of one prepared A row: its int8s, their scale, and the bias the dot's
/// encoding needs removed.
const ABlock = struct { q: VQ, scale: f32, correction: VI };

inline fn loadA(comptime enc: DotEnc, p: [*]const u8) ABlock {
    return .{
        .q = @as(*align(1) const VQ, @ptrCast(p)).*,
        .scale = @as(*align(1) const f32, @ptrCast(p + Q8_0_BLOCK_ELEMS)).*,
        .correction = matmul_q_i8.prepBiasNarrow(enc, p),
    };
}

/// `a` against the q8 block at `bp`, as a four-lane partial sum of their product.
inline fn blockDot(comptime enc: DotEnc, a: ABlock, bp: [*]const u8) VF {
    const bits: u16 = @as(*align(1) const u16, @ptrCast(bp)).*;
    const b_scale: f32 = @as(f16, @bitCast(bits));
    const bq: VQ = @as(*align(1) const VQ, @ptrCast(bp + 2)).*;
    const dots = matmul_q_i8.dotI8Narrow(enc, @as(VI, @splat(0)), bq, a.q) - a.correction;
    return @as(VF, @floatFromInt(dots)) * @as(VF, @splat(a.scale * b_scale));
}

/// Micro-tile: rows of C held in registers, and columns of C beside them.
///
/// Each column is its own sequential stream through B, so `NR` of them run `K/32
/// * 34` bytes apart — the wider the reduction, the further apart. Two streams
/// keep the pipeline fed without outrunning that locality; four fall off a cliff
/// as K grows. Measured on 1 GB of weights, one core, GB/s at NR = 1 / 2 / 4:
///
///   K=2048   62.8  67.1  60.3
///   K=5632   59.1  67.2  45.1
///   K=8192   57.4  62.8  31.8
const MR: usize = 2;
const NR: usize = 2;

/// One `[rows, cols]` block of C from a row-major B, swept over the whole of K.
///
/// B's rows are the output columns here, so each of `cols` walks an unbroken run
/// of `blocks` q8 blocks and the sums never leave registers — the two things the
/// block-major q8 matvec in `matvec_q.zig` has to choose between.
inline fn sweepBlock(
    comptime enc: DotEnc,
    comptime rows: usize,
    comptime cols: usize,
    a_rows: [*]const u8,
    a_stride: usize,
    b_col: [*]const u8,
    b_stride: usize,
    blocks: usize,
) [rows][cols]f32 {
    var acc: [rows][cols]VF = @splat(@splat(@as(VF, @splat(0.0))));
    var kb: usize = 0;
    while (kb < blocks) : (kb += 1) {
        var a: [rows]ABlock = undefined;
        inline for (0..rows) |ii| a[ii] = loadA(enc, a_rows + ii * a_stride + kb * PREP_BLOCK_BYTES);
        inline for (0..cols) |jj| {
            const bp = b_col + jj * b_stride + kb * Q8_0_BLOCK_BYTES;
            inline for (0..rows) |ii| acc[ii][jj] += blockDot(enc, a[ii], bp);
        }
    }
    return reduced(rows, cols, acc);
}

/// Rows the grouped kernel computes together, from the register file the tier
/// was compiled for. Each row keeps an f32 and an int32 accumulator (a vector
/// each) and its 32-byte A block live; four vectors stay free for B, its scales
/// and temporaries. On NEON that predicts 7, inside the measured plateau (flat
/// from 4 to 12 rows, spilling from 14).
fn groupTileRows(comptime lanes: usize) usize {
    const vector_bytes = lanes * 4;
    const registers: usize = if (builtin.cpu.arch.isAARCH64())
        32
    else if (builtin.cpu.arch.isX86() and std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f))
        32
    else
        16;
    const a_registers = (Q8_0_BLOCK_ELEMS + vector_bytes - 1) / vector_bytes;
    return @max(1, (registers - 4) / (2 + a_registers));
}

/// Independent int32 sums per row: a dot's result is not ready for several
/// cycles, so about four must be in flight, and a lone row cannot supply them
/// from its own lanes. Measured at one row: 48 GMAC/s with one sum, 117 with four.
fn chainsFor(comptime rows: usize) usize {
    return @max(1, 4 / rows);
}

/// `acc[l] += sum over t of b[4l + t] * a[4 * lane + t]`: row `l` of the group
/// against one 4-byte chunk of the activation. For the unsigned-by-signed x86
/// dot the result carries `128 * sum(b)` per lane, which `laneBias` removes.
inline fn laneDot(
    comptime enc: matmul_q_i8.DotEnc,
    comptime W: usize,
    acc: @Vector(W, i32),
    b: @Vector(4 * W, i8),
    a: @Vector(16, i8),
    comptime lane: usize,
) @Vector(W, i32) {
    if (comptime enc == .sdot and W == 4) {
        return asm (std.fmt.comptimePrint("sdot %[acc].4s, %[b].16b, %[a].4b[{d}]", .{lane})
            : [acc] "=w" (-> @Vector(4, i32)),
            : [b] "w" (b),
              [a] "w" (a),
              [acc0] "0" (acc),
        );
    }
    const words: @Vector(4, i32) = @bitCast(a);
    const spread: @Vector(4 * W, i8) = @bitCast(@as(@Vector(W, i32), @splat(words[lane])));
    if (comptime W % 8 == 0) {
        var out: [W]i32 = acc;
        const bs: [4 * W]i8 = b;
        const as: [4 * W]i8 = spread;
        inline for (0..W / 8) |h| {
            const part: @Vector(8, i32) = out[h * 8 ..][0..8].*;
            out[h * 8 ..][0..8].* = matmul_q_i8.dotI8(enc, part, as[h * 32 ..][0..32].*, bs[h * 32 ..][0..32].*);
        }
        return out;
    }
    var out = acc;
    inline for (0..W) |l| {
        inline for (0..4) |t| out[l] += @as(i32, b[4 * l + t]) * @as(i32, spread[4 * l + t]);
    }
    return out;
}

/// What `laneDot` adds on top of the signed dot for `b`: zero for signed dots.
inline fn laneBias(comptime enc: matmul_q_i8.DotEnc, comptime W: usize, b: @Vector(4 * W, i8)) @Vector(W, i32) {
    if (comptime !matmul_q_i8.encNeedsBias(enc)) return @splat(0);
    return laneDot(enc, W, @splat(0), b, @splat(0), 0);
}

/// `rows` rows of C for one group of `W` output columns, over all of K. One
/// lane-wise dot covers the group, so the scaling that follows each block is
/// paid once per row for all `W` columns.
inline fn sweepLanes(
    comptime enc: DotEnc,
    comptime W: usize,
    comptime rows: usize,
    a_rows: [*]const u8,
    a_stride: usize,
    group: [*]const u8,
    blocks: usize,
) [rows]@Vector(W, f32) {
    const VF_W = @Vector(W, f32);
    const VI_W = @Vector(W, i32);
    const VB = @Vector(4 * W, i8);
    const chunk_bytes = 4 * W;
    const segment_bytes = W * types.QuantBlockOrder.BLOCK_BYTES;

    var acc: [rows]VF_W = @splat(@splat(0.0));
    var kb: usize = 0;
    while (kb < blocks) : (kb += 1) {
        const seg = group + kb * segment_bytes;
        const b_scale: VF_W = @floatCast(@as(*align(1) const @Vector(W, f16), @ptrCast(seg)).*);
        const quants = seg + 2 * W;
        const CH = comptime chainsFor(rows);
        var sums: [rows][CH]VI_W = @splat(@splat(@splat(0)));
        var bias: VI_W = @splat(0);
        var a: [rows][2]@Vector(16, i8) = undefined;
        inline for (0..rows) |r| {
            const slot = a_rows + r * a_stride + kb * PREP_BLOCK_BYTES;
            inline for (0..2) |h| a[r][h] = @as(*align(1) const @Vector(16, i8), @ptrCast(slot + h * 16)).*;
        }
        inline for (0..Q8_0_BLOCK_ELEMS / 4) |j| {
            const b: VB = @as(*align(1) const VB, @ptrCast(quants + j * chunk_bytes)).*;
            bias += laneBias(enc, W, b);
            inline for (0..rows) |r| sums[r][j % CH] = laneDot(enc, W, sums[r][j % CH], b, a[r][j / 4], j % 4);
        }
        inline for (0..rows) |r| {
            var sum: VI_W = sums[r][0] - bias;
            inline for (1..CH) |c| sum += sums[r][c];
            const a_scale: f32 = @as(*align(1) const f32, @ptrCast(a_rows + r * a_stride + kb * PREP_BLOCK_BYTES + Q8_0_BLOCK_ELEMS)).*;
            acc[r] += @as(VF_W, @floatFromInt(sum)) * b_scale * @as(VF_W, @splat(a_scale));
        }
    }
    return acc;
}

inline fn reduced(comptime rows: usize, comptime cols: usize, acc: [rows][cols]VF) [rows][cols]f32 {
    var out: [rows][cols]f32 = undefined;
    inline for (0..rows) |ii| {
        inline for (0..cols) |jj| out[ii][jj] = @reduce(.Add, acc[ii][jj]);
    }
    return out;
}

inline fn storeLanes(
    comptime W: usize,
    comptime rows: usize,
    c: []align(1) f32,
    params: MatMulParams,
    ldc: usize,
    row0: usize,
    col0: usize,
    vals: [rows]@Vector(W, f32),
) void {
    inline for (0..rows) |r| {
        const lanes: [W]f32 = vals[r];
        inline for (0..W) |l| {
            const idx = (row0 + r) * ldc + col0 + l;
            c[idx] = if (params.beta == 0.0) params.alpha * lanes[l] else params.alpha * lanes[l] + params.beta * c[idx];
        }
    }
}

inline fn storeBlock(
    comptime rows: usize,
    comptime cols: usize,
    c: []align(1) f32,
    params: MatMulParams,
    ldc: usize,
    row0: usize,
    col0: usize,
    vals: [rows][cols]f32,
) void {
    inline for (0..rows) |ii| {
        inline for (0..cols) |jj| {
            const idx = (row0 + ii) * ldc + col0 + jj;
            c[idx] = if (params.beta == 0.0) params.alpha * vals[ii][jj] else params.alpha * vals[ii][jj] + params.beta * c[idx];
        }
    }
}

/// All of C for `rows` rows of A, in whichever block order B holds.
///
/// Columns — or groups of them — are the outer loop, so the rows read B once: a
/// column tile's blocks stay in L1 while every row consumes them. The rows are
/// what the caller's panel bounds; each extra panel is another stream of B.
inline fn sweepRows(
    comptime enc: DotEnc,
    comptime order: types.QuantBlockOrder,
    c: []align(1) f32,
    params: MatMulParams,
    ldc: usize,
    rows: usize,
    a_rows: [*]const u8,
    a_stride: usize,
    b: [*]const u8,
    b_stride: usize,
    blocks: usize,
) void {
    switch (order) {
        .row_major => {
            var j: usize = 0;
            while (j + NR <= params.n) : (j += NR) {
                sweepColumns(enc, NR, c, params, ldc, rows, a_rows, a_stride, b + j * b_stride, b_stride, blocks, j);
            }
            while (j < params.n) : (j += 1) {
                sweepColumns(enc, 1, c, params, ldc, rows, a_rows, a_stride, b + j * b_stride, b_stride, blocks, j);
            }
        },
        inline .lanes4, .lanes8, .lanes16, .lanes32 => |grouped| {
            const W = comptime grouped.groupRows();
            const R = comptime groupTileRows(W);
            const group_bytes: usize = blocks * W * types.QuantBlockOrder.BLOCK_BYTES;
            var g: usize = 0;
            while (g < params.n / W) : (g += 1) {
                const group = b + g * group_bytes;
                var i: usize = 0;
                while (i < rows) {
                    const left = rows - i;
                    if (left >= R) {
                        storeLanes(W, R, c, params, ldc, i, g * W, sweepLanes(enc, W, R, a_rows + i * a_stride, a_stride, group, blocks));
                        i += R;
                        continue;
                    }
                    inline for (1..R) |tail| {
                        if (left == tail) storeLanes(W, tail, c, params, ldc, i, g * W, sweepLanes(enc, W, tail, a_rows + i * a_stride, a_stride, group, blocks));
                    }
                    break;
                }
            }
        },
    }
}

/// Every row of A against `cols` adjacent row-major columns of B.
inline fn sweepColumns(
    comptime enc: DotEnc,
    comptime cols: usize,
    c: []align(1) f32,
    params: MatMulParams,
    ldc: usize,
    rows: usize,
    a_rows: [*]const u8,
    a_stride: usize,
    b_col: [*]const u8,
    b_stride: usize,
    blocks: usize,
    col0: usize,
) void {
    var i: usize = 0;
    while (i + MR <= rows) : (i += MR) {
        storeBlock(MR, cols, c, params, ldc, i, col0, sweepBlock(enc, MR, cols, a_rows + i * a_stride, a_stride, b_col, b_stride, blocks));
    }
    while (i < rows) : (i += 1) {
        storeBlock(1, cols, c, params, ldc, i, col0, sweepBlock(enc, 1, cols, a_rows + i * a_stride, a_stride, b_col, b_stride, blocks));
    }
}

/// One N tile. `a_bytes` is `m` rows prepared by `prepareActivation`, which every
/// N tile of one matmul shares. How many rows to hand over at once is the caller's
/// cache decision.
fn matmulNtQ8_0Impl(
    comptime enc: DotEnc,
    comptime order: types.QuantBlockOrder,
    params: MatMulParams,
    c_bytes: []u8,
    a_bytes: []const u8,
    b_bytes: []const u8,
) BackendError!void {
    // The micro-tile fully unrolls both axes over a portable byte dot's own loops.
    @setEvalBranchQuota(1_000_000);
    const g = try Geometry.of(order, params, c_bytes, a_bytes, b_bytes) orelse return;
    sweepRows(enc, order, g.c, params, g.ldc, params.m, a_bytes.ptr, g.a_stride, b_bytes.ptr, g.blocks * Q8_0_BLOCK_BYTES, g.blocks);
}

/// B laid out for another tier's or device's kernel: correct, not fast. Each block
/// is gathered back to row-major and contracted like one, so a tier compiles its
/// unrolled sweep for its own order only.
fn matmulNtQ8_0Foreign(
    comptime enc: DotEnc,
    order: types.QuantBlockOrder,
    params: MatMulParams,
    c_bytes: []u8,
    a_bytes: []const u8,
    b_bytes: []const u8,
) BackendError!void {
    const g = try Geometry.of(order, params, c_bytes, a_bytes, b_bytes) orelse return;
    var block: [Q8_0_BLOCK_BYTES]u8 = undefined;
    for (0..params.n) |col| {
        for (0..params.m) |r| {
            var acc: VF = @splat(0.0);
            for (0..g.blocks) |kb| {
                order.loadBlock(b_bytes, g.blocks, col, kb, &block);
                acc += blockDot(enc, loadA(enc, a_bytes.ptr + r * g.a_stride + kb * PREP_BLOCK_BYTES), &block);
            }
            storeBlock(1, 1, g.c, params, g.ldc, r, col, .{.{@reduce(.Add, acc)}});
        }
    }
}

/// The checked shape of one NT call; null when there is nothing to compute.
const Geometry = struct {
    blocks: usize,
    a_stride: usize,
    ldc: usize,
    c: []align(1) f32,

    fn of(order: types.QuantBlockOrder, params: MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) BackendError!?Geometry {
        const m = params.m;
        const n = params.n;
        const ldc: usize = if (params.ldc == 0) n else params.ldc;
        if (ldc < n) return BackendError.InvalidArgument;
        if ((params.k % Q8_0_BLOCK_ELEMS) != 0) return BackendError.InvalidArgument;
        if (m == 0 or n == 0) return null;

        const blocks = params.k / Q8_0_BLOCK_ELEMS;
        const a_stride = blocks * PREP_BLOCK_BYTES;
        const c: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, c_bytes);
        if (a_bytes.len < m * a_stride) return BackendError.InvalidArgument;
        if (b_bytes.len < n * blocks * Q8_0_BLOCK_BYTES) return BackendError.InvalidArgument;
        if (c.len < (m - 1) * ldc + n) return BackendError.InvalidArgument;
        if ((n % order.groupRows()) != 0) return BackendError.InvalidArgument;
        return .{ .blocks = blocks, .a_stride = a_stride, .ldc = ldc, .c = c };
    }
};

/// Bytes `prepareActivation` needs for `m` rows of `k`.
pub fn preparedBytes(m: usize, k: usize) usize {
    return m * (k / Q8_0_BLOCK_ELEMS) * PREP_BLOCK_BYTES;
}

/// Quantize `m` rows of A once, for every N tile of the same matmul to share.
pub fn prepareActivation(out: []u8, a_bytes: []const u8, m: usize, k: usize) BackendError!void {
    const blocks: usize = k / Q8_0_BLOCK_ELEMS;
    const a: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, a_bytes);
    if (a.len < m * k or out.len < preparedBytes(m, k)) return BackendError.InvalidArgument;
    for (0..m) |r| {
        matmul_q_i8.prepareARow(out[r * blocks * PREP_BLOCK_BYTES ..].ptr, a.ptr + r * k, blocks);
    }
}

pub fn Kernel(comptime t: matmul_nt.Tuning) type {
    const enc = t.dot_enc;
    return struct {
        /// The grouping this tier's kernel reads: its dot's lane count.
        pub const block_order: types.QuantBlockOrder = types.QuantBlockOrder.withGroup(t.lanes) orelse .row_major;

        /// The kernel for B stored in `order`: the unrolled sweep for row-major and
        /// this tier's own `block_order`, the gathering fallback for a weight laid
        /// out for another device.
        pub fn matmulNtQ8_0(comptime order: types.QuantBlockOrder) matmul_nt.MatMulNtQ8_0Fn {
            const fast = comptime order == .row_major or order == block_order;
            return struct {
                fn run(params: MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) BackendError!void {
                    if (fast) return matmulNtQ8_0Impl(enc, order, params, c_bytes, a_bytes, b_bytes);
                    return matmulNtQ8_0Foreign(enc, order, params, c_bytes, a_bytes, b_bytes);
                }
            }.run;
        }
    };
}

// Every grouping against a reference, through the unrolled sweep and the
// gathering fallback, at row counts below, at and past the
// kernel's row tile. The portable dot runs anywhere, so
// the widths other tiers use are checked on this machine too.
test "matmul_nt_q: grouped kernels match the reference at every width" {
    const testing = std.testing;
    const k: usize = 64;
    const blocks = k / Q8_0_BLOCK_ELEMS;

    inline for ([_]types.QuantBlockOrder{ .lanes4, .lanes8, .lanes16, .lanes32 }) |order| {
        const W = comptime order.groupRows();
        const n = 2 * W;
        var b_rows: [n * blocks * Q8_0_BLOCK_BYTES]u8 = undefined;
        var tile: [n * blocks * Q8_0_BLOCK_BYTES]u8 = undefined;
        for (0..n * blocks) |bi| {
            std.mem.writeInt(u16, b_rows[bi * 34 ..][0..2], @bitCast(@as(f16, @floatFromInt(1 + bi % 3)) * 0.01), .little);
            for (b_rows[bi * 34 + 2 ..][0..32], 0..) |*q, t| q.* = @truncate(bi *% 29 +% t *% 7);
        }
        for (0..n) |r| for (0..blocks) |kb| order.storeBlock(&tile, blocks, r, kb, b_rows[(r * blocks + kb) * 34 ..][0..34]);

        const R = comptime groupTileRows(W);
        const enc: DotEnc = .portable;
        for ([_]usize{ 1, R, R + 3 }) |m| {
            var a: [(R + 3) * k]f32 = undefined;
            for (a[0 .. m * k], 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 19)) - 9)) * 0.03;
            var prepared: [(R + 3) * blocks * PREP_BLOCK_BYTES]u8 align(32) = undefined;
            try prepareActivation(&prepared, std.mem.sliceAsBytes(a[0 .. m * k]), m, k);
            const a_bytes: []const u8 = prepared[0 .. m * blocks * PREP_BLOCK_BYTES];

            for ([_]bool{ true, false }) |fast| {
                var c: [(R + 3) * n]f32 = undefined;
                const params: MatMulParams = .{ .m = m, .n = n, .k = k };
                if (fast) {
                    try matmulNtQ8_0Impl(enc, order, params, std.mem.sliceAsBytes(c[0 .. m * n]), a_bytes, &tile);
                } else {
                    try matmulNtQ8_0Foreign(enc, order, params, std.mem.sliceAsBytes(c[0 .. m * n]), a_bytes, &tile);
                }

                for (0..m) |r| for (0..n) |col| {
                    var want: f64 = 0;
                    for (0..blocks) |kb| {
                        const blk = b_rows[(col * blocks + kb) * 34 ..];
                        const b_scale: f64 = @as(f16, @bitCast(std.mem.readInt(u16, blk[0..2], .little)));
                        var aq: [32]i8 = undefined;
                        const a_scale = matmul_q_i8.quantizeABlock(a[r * k + kb * 32 ..].ptr, &aq);
                        for (0..32) |t| {
                            const x: f64 = @as(f64, a_scale) * @as(f64, @floatFromInt(aq[t]));
                            want += x * b_scale * @as(f64, @floatFromInt(@as(i8, @bitCast(blk[2 + t]))));
                        }
                    }
                    try testing.expectApproxEqAbs(want, @as(f64, c[r * n + col]), 1e-4 * @max(1.0, @abs(want)));
                };
            }
        }
    }
}
