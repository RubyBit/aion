// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Prefill attention on the device's 16x16x16 f16 matrix tiles, accumulating in
//! f32 (kernels/attention_mx16.wgsl): the `matrix_f16` sibling of
//! `attention_mx.zig`, with the same `Params`, replacing attention_block.wgsl
//! wherever the device has the tiles and the shapes fit.
//!
//! WORK SHAPE — one workgroup per block of 16 consecutive query positions of one
//! head, all its value dims, split across 1, 2 or 4 SIMD groups of at most 128
//! dims (8 accumulator tiles) each:
//!
//!   * q is rounded to f16 into scratch first, once per op (kernels/narrow_f16.wgsl):
//!     a tile is loaded in its element type, and staging the block's rows in shared
//!     memory instead does not fit beside a 512-wide head's accumulators.
//!   * Keys are walked 64 at a time, as 4 tiles of 16; each SIMD group scores its
//!     share of the tiles, with q and the f16 K cache read in place: the block's
//!     rows are evenly strided in q, and 16 keys are 16 consecutive cache slots
//!     once a block starts on a multiple of 16.
//!   * The softmax runs in f32 on the scores in shared memory, all invocations at
//!     once, and leaves the probabilities there as f16.
//!   * Each SIMD group accumulates P @ V into its own value dims, V read in place.
//!
//! The shader's tile operations are written out here, one line each, into its
//! `// #name` lines: the shader compiler does not unroll loops, and tiles cannot
//! be indexed. There is one kernel per value width, since it sets how many SIMD
//! groups and accumulator tiles there are.

const std = @import("std");
const wgpu = @import("../wgpu.zig");
const pipelines = @import("../pipelines.zig");
const template = @import("../wgsl_template.zig");
const context = @import("../context.zig");
const backend_mod = @import("../../backend.zig");

const c = wgpu.c;
const Ctx = context.Ctx;
const Frame = @import("../frame.zig").Frame;
const KernelDesc = pipelines.KernelDesc;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

const TILE: usize = 16;
/// Query rows per block: one tile.
const ROWS: usize = TILE;
/// Head dims per pass of the score loop (`KC` in the shader): a head's dims must
/// divide into it.
const K_CHUNK: usize = 64;
/// Value widths there is a kernel for: multiples of this up to `MAX_DV`.
const DV_STEP: usize = 64;
const MAX_DV: usize = 512;
/// Most value dims one SIMD group accumulates (8 tiles); wider heads take more
/// SIMD groups per workgroup.
const MAX_SD: usize = 128;
/// Keys per block (`KB`): 4 tiles, split evenly over the SIMD groups.
const KEY_TILES: usize = 4;

/// SIMD groups per workgroup for heads `dv` wide: 1, 2 or 4, so each holds at
/// most `MAX_SD` dims (and the block's 4 key tiles split evenly).
fn simdGroups(dv: usize) usize {
    return if (dv <= MAX_SD) 1 else if (dv <= 2 * MAX_SD) 2 else 4;
}

/// Shared memory the kernel for `dv` uses, as attention_mx16.wgsl declares it.
fn sharedBytes(dv: usize) usize {
    const kb = KEY_TILES * TILE;
    const f32s = ROWS * dv + ROWS * kb + 32 * simdGroups(dv) + 5 * ROWS + 3;
    return f32s * 4 + ROWS * kb * 2;
}

/// Whether the matrix kernel computes this attention on a device with `limits`:
/// f32 q, an f16 cache whose rows tiles can address, head widths it has a kernel
/// for, at least one whole block of rows (the last one slides back), and a cache
/// of whole tiles, so a tile of keys never wraps around it.
pub fn fits(q_f16: bool, kv_f16: bool, rows: usize, dk: usize, dv: usize, t_cap: usize, limits: wgpu.Limits) bool {
    if (!limits.matrix_f16 or q_f16 or !kv_f16) return false;
    if (dk == 0 or dk % K_CHUNK != 0 or dv == 0 or dv % DV_STEP != 0 or dv > MAX_DV) return false;
    return rows >= ROWS and t_cap > 0 and t_cap % TILE == 0 and sharedBytes(dv) <= limits.max_shared_bytes;
}

/// The shape of a prefill attention, beyond what its `Params` hold.
pub const Shape = struct { heads: usize, rows: usize, batch: usize, dv: usize };

/// Record an attention that `fits`, with the bindings and `Params` bytes the
/// other attention kernels take: q, k, v, positions, lengths, out.
pub fn record(ctx: Ctx, frame: *Frame, params: []const u8, bufs: [6]c.WGPUBuffer, sizes: [6]u64, shape: Shape) ExecuteProgramError!void {
    // q into scratch as f16.
    const words = sizes[0] / 8; // two f32s per word
    const qh_bytes = words * 4;
    const qh = ctx.scratch.ensure(ctx.gpu, qh_bytes) catch return error.ExecutionFailed;
    const groups = std.math.divCeil(u64, words, NARROW_WG) catch unreachable;
    const gx = @min(groups, context.MAX_GROUPS_PER_DIM);
    const np: NarrowParams = .{ .words = @intCast(words) };
    const narrow = try ctx.pipes.get(narrow_kernel, "narrow");
    try frame.recordCompute(narrow, &.{ bufs[0], qh }, &.{ sizes[0], qh_bytes }, std.mem.asBytes(&np), .{
        @intCast(gx),
        @intCast(std.math.divCeil(u64, groups, gx) catch unreachable),
        1,
    });

    var b = bufs;
    var n = sizes;
    b[0] = qh;
    n[0] = qh_bytes;
    const built = try ctx.pipes.get(kernels[shape.dv / DV_STEP - 1], "attn_mx16");
    try frame.recordCompute(built, &b, &n, params, .{
        @intCast(shape.heads),
        @intCast(std.math.divCeil(usize, shape.rows, ROWS) catch unreachable),
        @intCast(shape.batch),
    });
}

/// Rounds an f32 buffer into f16 (kernels/narrow_f16.wgsl): bindings a, out,
/// then `NarrowParams`. Entry point `narrow`.
const narrow_kernel: KernelDesc = .{ .name = "narrow_f16", .wgsl = @embedFile("../kernels/narrow_f16.wgsl") };
/// Invocations per `narrow` workgroup (`WG` in the shader).
const NARROW_WG: u64 = 256;
/// Field order matches `struct Params` in narrow_f16.wgsl.
const NarrowParams = extern struct { words: u32, _p0: u32 = 0, _p1: u32 = 0, _p2: u32 = 0 };

const kernels = blk: {
    @setEvalBranchQuota(10_000_000);
    var out: [MAX_DV / DV_STEP]KernelDesc = undefined;
    for (&out, 1..) |*k, i| {
        const dv = i * DV_STEP;
        k.* = .{
            .name = std.fmt.comptimePrint("attention_mx16_{d}", .{dv}),
            .wgsl = source(dv) ++ @embedFile("../kernels/window.wgsl"),
        };
    }
    break :blk out;
};

fn source(comptime dv: usize) []const u8 {
    const ns = simdGroups(dv);
    const sd = dv / ns; // value dims per SIMD group
    const kt = KEY_TILES / ns; // key tiles each SIMD group scores per block
    const shape = std.fmt.comptimePrint(
        \\const NS: u32 = {d}u; // SIMD groups per workgroup
        \\const DV: u32 = {d}u; // value dims of a head
        \\const SD: u32 = {d}u; // value dims per SIMD group
        \\const KT: u32 = {d}u; // key tiles each SIMD group scores per block
    , .{ ns, dv, sd, kt });
    var s: []const u8 = @embedFile("../kernels/attention_mx16.wgsl");
    s = template.fill(s, "shape", &.{shape});
    s = template.fill(s, "acc_decl", template.each(sd / TILE, TILE, "var acc{0d} = coop_mat16x16<f32, C>(); // dims {1d}.. of this SIMD group's"));
    s = template.fill(s, "store_acc", template.each(sd / TILE, TILE, "coopStoreT(acc{0d}, &st[sg * SD + {1d}u], DV);"));
    s = template.fill(s, "load_acc", template.each(sd / TILE, TILE, "acc{0d} = coopLoadT<coop_mat16x16<f32, C>>(&st[sg * SD + {1d}u], DV);"));
    s = template.fill(s, "score_decl", template.each(kt, TILE, "let k{0d} = kb + slot(t0 + kt0 * 16u + {1d}u) * krow; var s{0d} = coop_mat16x16<f32, C>();"));
    s = template.fill(s, "score_tiles", scoreTiles(kt));
    s = template.fill(s, "score_store", template.each(kt, TILE, "coopStoreT(s{0d}, &ps[kt0 * 16u + {1d}u], KB);"));
    s = template.fill(s, "value_tiles", valueTiles(sd));
    return s;
}

/// One pass of the score loop: each of 4 tiles of q (16 dims) against this SIMD
/// group's `kt` tiles of keys (rows `k0`.. of K, read as K^T's columns).
fn scoreTiles(comptime kt: usize) []const []const u8 {
    var lines: []const []const u8 = &.{};
    for (0..K_CHUNK / TILE) |dd| {
        lines = lines ++ .{std.fmt.comptimePrint("let a{0d} = coopLoadT<coop_mat16x16<f16, A>>(&qh[qb + d0 + {1d}u], qrow);", .{ dd, dd * TILE })};
        for (0..kt) |i| {
            lines = lines ++ .{std.fmt.comptimePrint("s{0d} = coopMultiplyAdd(a{1d}, coopLoad<coop_mat16x16<f16, B>>(&kh[k{0d} + d0 + {2d}u], krow), s{0d});", .{ i, dd, dd * TILE })};
        }
    }
    return lines;
}

/// out += P @ V for this SIMD group's `sd` value dims: each of the block's 4
/// tiles of P against the matching rows (`v0`..`v3`) of V.
fn valueTiles(comptime sd: usize) []const []const u8 {
    var lines: []const []const u8 = &.{};
    for (0..KEY_TILES) |kt| {
        lines = lines ++ .{std.fmt.comptimePrint("let p{0d} = coopLoadT<coop_mat16x16<f16, A>>(&ph[{1d}u], KB);", .{ kt, kt * TILE })};
        for (0..sd / TILE) |j| {
            lines = lines ++ .{std.fmt.comptimePrint("acc{0d} = coopMultiplyAdd(p{1d}, coopLoadT<coop_mat16x16<f16, B>>(&vh[v{1d} + {2d}u], vrow), acc{0d});", .{ j, kt, j * TILE })};
        }
    }
    return lines;
}
