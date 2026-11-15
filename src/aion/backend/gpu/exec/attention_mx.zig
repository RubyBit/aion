// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Prefill attention on the device's matrix units (kernels/attention_mx.wgsl):
//! the same math, bindings and `Params` as attention_block.wgsl, which it
//! replaces wherever the device has 8x8 f32 tiles and the shapes fit.
//!
//! WORK SHAPE — one workgroup per block of 8 consecutive query positions of one
//! head, all its value dims, split across 1, 2 or 4 SIMD groups of at most 128
//! dims (16 accumulator tiles) each:
//!
//!   * Keys are walked 32 at a time, as 4 tiles of 8; each SIMD group scores its
//!     share of the tiles, with q and K read in place: the block's rows are
//!     evenly strided in q, and 8 keys are 8 consecutive cache slots once a block
//!     starts on a multiple of 8.
//!   * The softmax runs on the scores in shared memory, all invocations at once.
//!   * Each SIMD group accumulates P @ V into its own value dims, V read in place.
//!
//! Splitting the value dims across SIMD groups rather than across workgroups is
//! what computes each score once: with 256 dims per workgroup, a 512-wide head
//! scored every key twice and its layers ran at half the speed of the others.
//!
//! An f16 cache is first widened into f32 scratch (kernels/widen_f16.wgsl), once
//! per op. f16 tiles would read it in place, but accumulating in f16 is too
//! coarse and f16-in/f32-out tiles do not compile on Metal yet (see the TODO on
//! `wgpu.Limits.matrix_f32`); staging it through shared memory instead, block by
//! block and head by head, cost twice what the products did.
//!
//! The shader's tile operations are written out here, one line each, into its
//! `// #name` lines: the shader compiler does not unroll loops, and tiles cannot
//! be indexed. There is one kernel per value width, since it sets how many
//! SIMD groups and accumulator tiles there are.

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

/// Rows (query positions) per block: one tile.
const ROWS: usize = 8;
/// Head dims per pass of the score loop (`KC` in the shader): a head's dims must
/// divide into it.
const K_CHUNK: usize = 64;
/// Value widths there is a kernel for: multiples of this up to `MAX_DV`.
const DV_STEP: usize = 64;
const MAX_DV: usize = 512;
/// Most value dims one SIMD group accumulates (16 tiles); wider heads take more
/// SIMD groups per workgroup.
const MAX_SD: usize = 128;

/// SIMD groups per workgroup for heads `dv` wide: 1, 2 or 4, so each holds at
/// most `MAX_SD` dims (and the block's 4 key tiles split evenly).
fn simdGroups(dv: usize) usize {
    return if (dv <= MAX_SD) 1 else if (dv <= 2 * MAX_SD) 2 else 4;
}

/// Shared memory the kernel for `dv` uses, as attention_mx.wgsl declares it.
fn sharedBytes(dv: usize) usize {
    const floats = 8 * dv + 8 * 32 + 32 * simdGroups(dv) + 5 * 8 + 3;
    return floats * 4;
}

/// Whether the matrix kernel computes attention with these head widths over a
/// cache of `t_cap` slots on a device with `limits`. Tiles are read in place, so
/// a tile of 8 keys must never wrap around the cache.
pub fn fits(dk: usize, dv: usize, t_cap: usize, limits: wgpu.Limits) bool {
    if (!limits.matrix_f32) return false;
    if (dk == 0 or dk % K_CHUNK != 0 or dv == 0 or dv % DV_STEP != 0 or dv > MAX_DV) return false;
    return t_cap > 0 and t_cap % ROWS == 0 and sharedBytes(dv) <= limits.max_shared_bytes;
}

/// The shape of a prefill attention, beyond what its `Params` hold.
pub const Shape = struct { heads: usize, rows: usize, batch: usize, dv: usize, kv_f16: bool };

/// Record an attention that `fits`, with the bindings and `Params` bytes the
/// other attention kernels take: q, k, v, positions, lengths, out.
pub fn record(ctx: Ctx, frame: *Frame, params: []const u8, bufs: [6]c.WGPUBuffer, sizes: [6]u64, shape: Shape) ExecuteProgramError!void {
    var b = bufs;
    var n = sizes;
    if (shape.kv_f16) {
        // Both caches into scratch as f32, K then V; the kernel reads both from it.
        const k_words = sizes[1] / 4;
        const v_words = sizes[2] / 4;
        const wide_bytes = (k_words + v_words) * 2 * @sizeOf(f32);
        const wide = ctx.scratch.ensure(ctx.gpu, wide_bytes) catch return error.ExecutionFailed;
        const groups = std.math.divCeil(u64, k_words + v_words, WIDEN_WG) catch unreachable;
        const gx = @min(groups, context.MAX_GROUPS_PER_DIM);
        const wp: WidenParams = .{ .a_words = @intCast(k_words), .b_words = @intCast(v_words) };
        const widen = try ctx.pipes.get(widen_kernel, "widen");
        try frame.recordCompute(widen, &.{ bufs[1], bufs[2], wide }, &.{ sizes[1], sizes[2], wide_bytes }, std.mem.asBytes(&wp), .{
            @intCast(gx),
            @intCast(std.math.divCeil(u64, groups, gx) catch unreachable),
            1,
        });
        b[1] = wide;
        b[2] = wide;
        n[1] = wide_bytes;
        n[2] = wide_bytes;
    }
    const built = try ctx.pipes.get(kernels[shape.dv / DV_STEP - 1], "attn_mx");
    try frame.recordCompute(built, &b, &n, params, .{
        @intCast(shape.heads),
        @intCast(std.math.divCeil(usize, shape.rows, ROWS) catch unreachable),
        @intCast(shape.batch),
    });
}

/// Widens two f16 buffers into one f32 buffer (kernels/widen_f16.wgsl): bindings
/// a, b, out, then `WidenParams`. Entry point `widen`.
const widen_kernel: KernelDesc = .{ .name = "widen_f16", .wgsl = @embedFile("../kernels/widen_f16.wgsl") };
/// Invocations per `widen` workgroup (`WG` in the shader).
const WIDEN_WG: u64 = 256;
/// Field order matches `struct Params` in widen_f16.wgsl.
const WidenParams = extern struct { a_words: u32, b_words: u32, _p0: u32 = 0, _p1: u32 = 0 };

const kernels = blk: {
    @setEvalBranchQuota(10_000_000);
    var out: [MAX_DV / DV_STEP]KernelDesc = undefined;
    for (&out, 1..) |*k, i| {
        const dv = i * DV_STEP;
        k.* = .{
            .name = std.fmt.comptimePrint("attention_mx_{d}", .{dv}),
            .wgsl = source(dv) ++ @embedFile("../kernels/window.wgsl"),
        };
    }
    break :blk out;
};

fn source(comptime dv: usize) []const u8 {
    const ns = simdGroups(dv);
    const sd = dv / ns; // value dims per SIMD group
    const kt = 4 / ns; // key tiles each SIMD group scores per block
    const shape = std.fmt.comptimePrint(
        \\const NS: u32 = {d}u; // SIMD groups per workgroup
        \\const DV: u32 = {d}u; // value dims of a head
        \\const SD: u32 = {d}u; // value dims per SIMD group
        \\const KT: u32 = {d}u; // key tiles each SIMD group scores per block
    , .{ ns, dv, sd, kt });
    var s: []const u8 = @embedFile("../kernels/attention_mx.wgsl");
    s = template.fill(s, "shape", &.{shape});
    s = template.fill(s, "acc_decl", template.each(sd / 8, 8, "var acc{0d} = coop_mat8x8<f32, C>(); // dims {1d}.. of this SIMD group's"));
    s = template.fill(s, "store_acc", template.each(sd / 8, 8, "coopStore(acc{0d}, &st[sg * SD + {1d}u], DV);"));
    s = template.fill(s, "load_acc", template.each(sd / 8, 8, "acc{0d} = coopLoad<coop_mat8x8<f32, C>>(&st[sg * SD + {1d}u], DV);"));
    s = template.fill(s, "score_decl", template.each(kt, 8, "let k{0d} = kb + slot(t0 + kt0 * 8u + {1d}u) * krow; var s{0d} = coop_mat8x8<f32, C>();"));
    s = template.fill(s, "score_tiles", scoreTiles(kt));
    s = template.fill(s, "score_store", template.each(kt, 8, "coopStore(s{0d}, &ps[kt0 * 8u + {1d}u], KB);"));
    s = template.fill(s, "value_tiles", valueTiles(sd));
    return s;
}

/// One pass of the score loop: each of 8 tiles of q (8 dims) against this SIMD
/// group's `kt` tiles of keys (rows `k0`.. of K).
fn scoreTiles(comptime kt: usize) []const []const u8 {
    var lines: []const []const u8 = &.{};
    for (0..K_CHUNK / 8) |dd| {
        lines = lines ++ .{std.fmt.comptimePrint("let a{0d} = coopLoad<coop_mat8x8<f32, A>>(&q[qb + d0 + {1d}u], qrow);", .{ dd, dd * 8 })};
        for (0..kt) |i| {
            lines = lines ++ .{std.fmt.comptimePrint("s{0d} = coopMultiplyAdd(a{1d}, coopLoadT<coop_mat8x8<f32, B>>(&kf[k{0d} + d0 + {2d}u], krow), s{0d});", .{ i, dd, dd * 8 })};
        }
    }
    return lines;
}

/// out += P @ V for this SIMD group's `sd` value dims: each of the block's 4
/// tiles of P against the matching rows (`v0`..`v3`) of V.
fn valueTiles(comptime sd: usize) []const []const u8 {
    var lines: []const []const u8 = &.{};
    for (0..4) |kt| {
        lines = lines ++ .{std.fmt.comptimePrint("let p{0d} = coopLoad<coop_mat8x8<f32, A>>(&ps[{1d}u], KB);", .{ kt, kt * 8 })};
        for (0..sd / 8) |j| {
            lines = lines ++ .{std.fmt.comptimePrint("acc{0d} = coopMultiplyAdd(p{1d}, coopLoad<coop_mat8x8<f32, B>>(&vf[v{1d} + {2d}u], vrow), acc{0d});", .{ j, kt, j * 8 })};
        }
    }
    return lines;
}
