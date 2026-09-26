// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const conv_utils = @import("conv_utils.zig");
const conv2d_kernels = @import("../kernels/conv2d.zig");
const simd = @import("../kernels/simd.zig");
const matmul_registry = @import("../registry/matmul_registry.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");

const BackendError = conv_utils.BackendError;
const MatMulParams = conv_utils.MatMulParams;
const ExecuteProgramError = conv_utils.ExecuteProgramError;

const StepConv1D = executable.StepConv1D;

const elemCountFromShape = conv_utils.elemCountFromShape;

pub const ConvExecCtx = conv_utils.ConvExecCtx;
const PackedWeightKey = conv_utils.PackedWeightKey;
const PackedWeightEntry = conv_utils.PackedWeightEntry;

const getOrCreatePackedWeights = conv_utils.getOrCreatePackedWeights;
const scratchForTid = conv_utils.scratchForTid;
const fillWeightBlock = conv_utils.fillWeightBlock;
const addBiasRowsF32 = conv_utils.addBiasRowsF32;

fn bytesAsF32Const(bytes: []const u8) []align(1) const f32 {
    std.debug.assert((bytes.len % @sizeOf(f32)) == 0);
    const ptr: [*]align(1) const f32 = @ptrCast(bytes.ptr);
    return ptr[0 .. bytes.len / @sizeOf(f32)];
}

fn bytesAsF32Mut(bytes: []u8) []align(1) f32 {
    std.debug.assert((bytes.len % @sizeOf(f32)) == 0);
    const ptr: [*]align(1) f32 = @ptrCast(bytes.ptr);
    return ptr[0 .. bytes.len / @sizeOf(f32)];
}

inline fn reflectIndex1D(idx_nom: isize, len: usize) usize {
    const l: isize = @intCast(len);
    var x: isize = idx_nom;
    while (x < 0 or x >= l) {
        if (x < 0) {
            x = -x;
        } else {
            x = (2 * l - 2) - x;
        }
    }
    return @intCast(x);
}

fn tryExecConv1DSmallDirect(
    ctx: *ConvExecCtx,
    s: StepConv1D,
    out_meta: tensor_store.TensorMeta,
    x_meta: tensor_store.TensorMeta,
    w_meta: tensor_store.TensorMeta,
    store: tensor_store.TensorStore,
) ExecuteProgramError!bool {
    @setRuntimeSafety(false);

    const rank: usize = @as(usize, out_meta.rank);
    if (rank != 3) return false;
    if (s.groups != 1) return false;
    if (out_meta.dtype != .f32 or x_meta.dtype != .f32 or w_meta.dtype != .f32) return false;

    const batch: usize = out_meta.shape[0];
    const l_out: usize = out_meta.shape[1];
    const c_out: usize = out_meta.shape[2];
    const l_in: usize = x_meta.shape[1];
    const c_in: usize = x_meta.shape[2];
    const k_len: usize = w_meta.shape[0];

    // Core compute: assumes NLC packed buffers.
    const doDirect = struct {
        fn run(
            s2: StepConv1D,
            batch2: usize,
            l_out2: usize,
            c_out2: usize,
            l_in2: usize,
            c_in2: usize,
            k_len2: usize,
            x_flat2: [*]align(1) const f32,
            w_flat2: [*]align(1) const f32,
            out_flat2: [*]align(1) f32,
            bias_flat2: ?[*]align(1) const f32,
        ) void {
            const use_reflect2: bool = (s2.pad_mode == .reflect);
            const l_in_i2: isize = @intCast(l_in2);
            const stride2: usize = s2.stride;
            const dilation2: usize = s2.dilation;
            const pad_left2: usize = s2.pad_left;

            const Vec = @Vector(simd.lanesF32(), f32);
            const lanes2: usize = @typeInfo(Vec).vector.len;
            const vecs_per_block: usize = 4;
            const block_ch: usize = lanes2 * vecs_per_block;

            var b2: usize = 0;
            while (b2 < batch2) : (b2 += 1) {
                const x_base2: usize = b2 * l_in2 * c_in2;
                const out_base2: usize = b2 * l_out2 * c_out2;

                var lo2: usize = 0;
                while (lo2 < l_out2) : (lo2 += 1) {
                    const out_off2: usize = out_base2 + lo2 * c_out2;

                    // Block over output channels to keep accumulators in registers.
                    const full_blocks: usize = (c_out2 / block_ch) * block_ch;
                    var co0: usize = 0;
                    while (co0 < full_blocks) : (co0 += block_ch) {
                        var acc0: Vec = @splat(@as(f32, 0.0));
                        var acc1: Vec = @splat(@as(f32, 0.0));
                        var acc2: Vec = @splat(@as(f32, 0.0));
                        var acc3: Vec = @splat(@as(f32, 0.0));
                        if (bias_flat2) |bp| {
                            const b0: Vec = @as(*align(1) const Vec, @ptrCast(bp + co0)).*;
                            const b1: Vec = @as(*align(1) const Vec, @ptrCast(bp + co0 + lanes2)).*;
                            const b2v: Vec = @as(*align(1) const Vec, @ptrCast(bp + co0 + 2 * lanes2)).*;
                            const b3: Vec = @as(*align(1) const Vec, @ptrCast(bp + co0 + 3 * lanes2)).*;
                            acc0 = b0;
                            acc1 = b1;
                            acc2 = b2v;
                            acc3 = b3;
                        }

                        var kw2: usize = 0;
                        while (kw2 < k_len2) : (kw2 += 1) {
                            const in_nom2: isize = @as(isize, @intCast(lo2 * stride2 + kw2 * dilation2)) - @as(isize, @intCast(pad_left2));
                            const li2: usize = if (use_reflect2) reflectIndex1D(in_nom2, l_in2) else blk: {
                                if (in_nom2 < 0 or in_nom2 >= l_in_i2) continue;
                                break :blk @intCast(in_nom2);
                            };

                            const x_row2: usize = x_base2 + li2 * c_in2;
                            const w_kw2: usize = kw2 * c_in2 * c_out2;

                            var ic2: usize = 0;
                            while (ic2 < c_in2) : (ic2 += 1) {
                                const x_val2: f32 = x_flat2[x_row2 + ic2];
                                const xv2: Vec = @splat(x_val2);
                                const w_ic2: usize = w_kw2 + ic2 * c_out2 + co0;
                                const w0: Vec = @as(*align(1) const Vec, @ptrCast(w_flat2 + w_ic2)).*;
                                const w1: Vec = @as(*align(1) const Vec, @ptrCast(w_flat2 + w_ic2 + lanes2)).*;
                                const w2: Vec = @as(*align(1) const Vec, @ptrCast(w_flat2 + w_ic2 + 2 * lanes2)).*;
                                const w3: Vec = @as(*align(1) const Vec, @ptrCast(w_flat2 + w_ic2 + 3 * lanes2)).*;
                                acc0 = @mulAdd(Vec, xv2, w0, acc0);
                                acc1 = @mulAdd(Vec, xv2, w1, acc1);
                                acc2 = @mulAdd(Vec, xv2, w2, acc2);
                                acc3 = @mulAdd(Vec, xv2, w3, acc3);
                            }
                        }

                        const outp: [*]align(1) f32 = out_flat2 + out_off2 + co0;
                        @as(*align(1) Vec, @ptrCast(outp)).* = acc0;
                        @as(*align(1) Vec, @ptrCast(outp + lanes2)).* = acc1;
                        @as(*align(1) Vec, @ptrCast(outp + 2 * lanes2)).* = acc2;
                        @as(*align(1) Vec, @ptrCast(outp + 3 * lanes2)).* = acc3;
                    }

                    // Remaining full vectors.
                    var co2: usize = full_blocks;
                    while (co2 + lanes2 <= c_out2) : (co2 += lanes2) {
                        var accv: Vec = @splat(@as(f32, 0.0));
                        if (bias_flat2) |bp| {
                            accv = @as(*align(1) const Vec, @ptrCast(bp + co2)).*;
                        }

                        var kw2: usize = 0;
                        while (kw2 < k_len2) : (kw2 += 1) {
                            const in_nom2: isize = @as(isize, @intCast(lo2 * stride2 + kw2 * dilation2)) - @as(isize, @intCast(pad_left2));
                            const li2: usize = if (use_reflect2) reflectIndex1D(in_nom2, l_in2) else blk: {
                                if (in_nom2 < 0 or in_nom2 >= l_in_i2) continue;
                                break :blk @intCast(in_nom2);
                            };
                            const x_row2: usize = x_base2 + li2 * c_in2;
                            const w_kw2: usize = kw2 * c_in2 * c_out2;
                            var ic2: usize = 0;
                            while (ic2 < c_in2) : (ic2 += 1) {
                                const x_val2: f32 = x_flat2[x_row2 + ic2];
                                const xv2: Vec = @splat(x_val2);
                                const w_ic2: usize = w_kw2 + ic2 * c_out2 + co2;
                                const wv: Vec = @as(*align(1) const Vec, @ptrCast(w_flat2 + w_ic2)).*;
                                accv = @mulAdd(Vec, xv2, wv, accv);
                            }
                        }
                        @as(*align(1) Vec, @ptrCast(out_flat2 + out_off2 + co2)).* = accv;
                    }

                    // Scalar tail.
                    while (co2 < c_out2) : (co2 += 1) {
                        var accs: f32 = if (bias_flat2) |bp| bp[co2] else 0.0;
                        var kw2: usize = 0;
                        while (kw2 < k_len2) : (kw2 += 1) {
                            const in_nom2: isize = @as(isize, @intCast(lo2 * stride2 + kw2 * dilation2)) - @as(isize, @intCast(pad_left2));
                            const li2: usize = if (use_reflect2) reflectIndex1D(in_nom2, l_in2) else blk: {
                                if (in_nom2 < 0 or in_nom2 >= l_in_i2) continue;
                                break :blk @intCast(in_nom2);
                            };
                            const x_row2: usize = x_base2 + li2 * c_in2;
                            const w_kw2: usize = kw2 * c_in2 * c_out2;
                            var ic2: usize = 0;
                            while (ic2 < c_in2) : (ic2 += 1) {
                                const x_val2: f32 = x_flat2[x_row2 + ic2];
                                const w_idx2: usize = w_kw2 + ic2 * c_out2 + co2;
                                accs = @mulAdd(f32, x_val2, w_flat2[w_idx2], accs);
                            }
                        }
                        out_flat2[out_off2 + co2] = accs;
                    }
                }
            }
        }
    }.run;

    // Compute gate: this path is only intended for *very* small convolutions
    // where tiling + packing overhead dominates (e.g. feature extraction with
    // tiny sequence lengths). For larger problems, the implicit-GEMM kernels
    // are dramatically faster.
    const max_macs: u128 = 2_000_000;
    var macs: u128 = 1;
    macs = std.math.mul(u128, macs, @as(u128, batch)) catch return false;
    macs = std.math.mul(u128, macs, @as(u128, l_out)) catch return false;
    macs = std.math.mul(u128, macs, @as(u128, c_out)) catch return false;
    macs = std.math.mul(u128, macs, @as(u128, k_len)) catch return false;
    macs = std.math.mul(u128, macs, @as(u128, c_in)) catch return false;
    if (macs > max_macs) return false;

    // Size gate: only small workloads (each buffer < 1 MB).
    const x_n: usize = batch * l_in * c_in;
    const w_n: usize = k_len * c_in * c_out;
    const out_n: usize = batch * l_out * c_out;
    const limit: usize = 256 * 1024;
    if (x_n > limit or w_n > limit or out_n > limit) return false;

    const x = try store.acquireConst(s.x);
    defer store.releaseConst(x.token);
    const w = try store.acquireConst(s.w);
    defer store.releaseConst(w.token);
    const out = try store.acquireMut(s.out);
    defer store.releaseMut(out.token);
    const bias = if (s.bias) |b_id| try store.acquireConst(b_id) else null;
    defer if (bias) |bt| store.releaseConst(bt.token);

    doDirect(
        s,
        batch,
        l_out,
        c_out,
        l_in,
        c_in,
        k_len,
        bytesAsF32Const(x.bytes).ptr,
        bytesAsF32Const(w.bytes).ptr,
        bytesAsF32Mut(out.bytes).ptr,
        if (bias) |bt| bytesAsF32Const(bt.bytes).ptr else null,
    );
    _ = ctx;
    return true;
}

/// Depthwise Conv1D is depthwise Conv2D over a single input row, its length the width.
fn execDepthwise(ctx: *ConvExecCtx, s: StepConv1D, out_meta: tensor_store.TensorMeta, x_meta: tensor_store.TensorMeta, w_meta: tensor_store.TensorMeta, store: tensor_store.TensorStore) ExecuteProgramError!bool {
    const rank: usize = out_meta.rank;
    const c: usize = out_meta.shape[rank - 1];
    if (s.groups != c or x_meta.shape[rank - 1] != c or w_meta.shape[1] != 1 or w_meta.shape[2] != c) return false;
    if (w_meta.shape[0] > conv2d_kernels.MAX_TAPS) return false;
    var batch: usize = 1;
    for (out_meta.shape[0 .. rank - 2]) |d| batch *= d;

    const x = try store.acquireConst(s.x);
    defer store.releaseConst(x.token);
    const w = try store.acquireConst(s.w);
    defer store.releaseConst(w.token);
    const out = try store.acquireMut(s.out);
    defer store.releaseMut(out.token);
    const bias = if (s.bias) |b_id| try store.acquireConst(b_id) else null;
    defer if (bias) |b| store.releaseConst(b.token);

    var task: conv2d_kernels.DepthwiseConv2DTask = .{
        .p = .{
            .stride_h = 1,
            .stride_w = s.stride,
            .dilation_h = 1,
            .dilation_w = s.dilation,
            .pad_top = 0,
            .pad_left = s.pad_left,
            .reflect = s.pad_mode == .reflect,
        },
        .batch = batch,
        .h_in = 1,
        .w_in = x_meta.shape[rank - 2],
        .h_out = 1,
        .w_out = out_meta.shape[rank - 2],
        .c = c,
        .k_h = 1,
        .k_w = w_meta.shape[0],
        .x = bytesAsF32Const(x.bytes),
        .w = bytesAsF32Const(w.bytes),
        .bias = if (bias) |b| bytesAsF32Const(b.bytes) else &.{},
        .out = bytesAsF32Mut(out.bytes),
    };
    conv_utils.runDepthwise(ctx, &task);
    return true;
}

fn execConv1DImplicitGemm(
    ctx: *ConvExecCtx,
    s: StepConv1D,
    out_meta: tensor_store.TensorMeta,
    x_meta: tensor_store.TensorMeta,
    w_meta: tensor_store.TensorMeta,
    store: tensor_store.TensorStore,
) ExecuteProgramError!bool {
    // Small convolutions: a direct loop beats packing for the GEMM.
    if (try tryExecConv1DSmallDirect(ctx, s, out_meta, x_meta, w_meta, store)) {
        return true;
    }

    if (try execDepthwise(ctx, s, out_meta, x_meta, w_meta, store)) return true;

    const rank: usize = @as(usize, out_meta.rank);
    const c_out: usize = out_meta.shape[rank - 1];
    const l_in: usize = x_meta.shape[rank - 2];
    const c_in: usize = x_meta.shape[rank - 1];
    const k: usize = w_meta.shape[0];
    const c_in_g: usize = w_meta.shape[1];
    const l_out: usize = out_meta.shape[rank - 2];
    const batch: usize = if (rank == 2) 1 else blk: {
        var acc: usize = 1;
        var d: usize = 0;
        while (d + 2 < rank) : (d += 1) acc = std.math.mul(usize, acc, out_meta.shape[d]) catch return BackendError.InvalidArgument;
        break :blk acc;
    };

    const groups: usize = s.groups;
    if (groups == 0) return BackendError.InvalidArgument;
    if (c_in % groups != 0 or c_out % groups != 0) return BackendError.InvalidArgument;
    if (c_in_g * groups != c_in) return BackendError.InvalidArgument;
    const c_out_g: usize = c_out / groups;

    const rows_total: usize = batch * l_out;
    const k_dim_g: usize = k * c_in_g;
    const is_pointwise_unit: bool = (k == 1 and s.stride == 1 and s.dilation == 1 and s.pad_left == 0 and s.pad_right == 0 and l_out == l_in);
    const is_k3_same_regular: bool = (k == 3 and s.stride == 1 and s.dilation == 1 and s.pad_left == 1 and s.pad_right == 1 and l_out == l_in);

    const alloc: std.mem.Allocator = std.heap.page_allocator;

    // Only build K→(kw,ic) maps when we actually need per-tap indexing.
    // Pointwise unit conv gathers directly from contiguous x slices.
    const kw_map: []usize = if (!is_pointwise_unit) try alloc.alloc(usize, k_dim_g) else &[_]usize{};
    defer if (kw_map.len != 0) alloc.free(kw_map);
    const ic_map: []usize = if (!is_pointwise_unit) try alloc.alloc(usize, k_dim_g) else &[_]usize{};
    defer if (ic_map.len != 0) alloc.free(ic_map);

    if (!is_pointwise_unit) {
        var gk_init: usize = 0;
        while (gk_init < k_dim_g) : (gk_init += 1) {
            const kwv: usize = gk_init / c_in_g;
            const icv: usize = gk_init - kwv * c_in_g;
            kw_map[gk_init] = kwv;
            ic_map[gk_init] = icv;
        }
    }

    const is_stride1_dilation1: bool = (s.stride == 1 and s.dilation == 1);
    const is_stride1_contig_groups1: bool = (is_stride1_dilation1 and groups == 1 and c_in_g == c_in);
    const is_stride1_pad0_contig: bool = (is_stride1_contig_groups1 and s.pad_left == 0 and s.pad_right == 0);
    const x_offset_map: []usize = if (!is_pointwise_unit and is_stride1_dilation1 and !is_stride1_pad0_contig) try alloc.alloc(usize, k_dim_g) else &[_]usize{};
    defer if (x_offset_map.len != 0) alloc.free(x_offset_map);

    if (x_offset_map.len != 0) {
        var gk_off: usize = 0;
        while (gk_off < k_dim_g) : (gk_off += 1) {
            x_offset_map[gk_off] = kw_map[gk_off] * c_in + ic_map[gk_off];
        }
    }

    const x_ref = try store.acquireConst(s.x);
    defer store.releaseConst(x_ref.token);
    const out_ref = try store.acquireMut(s.out);
    defer store.releaseMut(out_ref.token);
    const bias_ref = if (s.bias) |b_id| try store.acquireConst(b_id) else null;
    defer if (bias_ref) |b| store.releaseConst(b.token);
    const x_packed: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, x_ref.bytes));
    const out_packed: []f32 = @alignCast(std.mem.bytesAsSlice(f32, out_ref.bytes));
    const bias_packed: []const f32 = if (bias_ref) |b| @alignCast(std.mem.bytesAsSlice(f32, b.bytes)) else &.{};
    if (bias_packed.len != 0 and bias_packed.len != c_out) return BackendError.InvalidArgument;

    if (c_out_g == 0 or ctx.matmul_f32.tuning.nc == 0) return BackendError.InvalidArgument;

    const matmul: matmul_registry.F32Kernels = ctx.matmul_f32;
    const use_local_scratch: bool = false;

    const kc: usize = matmul.tuning.kc;
    const m_cap: usize = matmul.tuning.mc;
    const oc_tile_max: usize = @min(c_out_g, matmul.tuning.nc);

    const w_ref = try store.acquireConst(s.w);
    defer store.releaseConst(w_ref.token);
    const w_packed: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, w_ref.bytes));

    const TileInfo = struct {
        oc_start: usize,
        oc_count: usize,
        ic_base: usize,
        packed_w: PackedWeightEntry,
    };

    const tiles_per_group: usize = (c_out_g + oc_tile_max - 1) / oc_tile_max;
    const total_tiles: usize = groups * tiles_per_group;
    const tile_infos: []TileInfo = try alloc.alloc(TileInfo, total_tiles);
    defer alloc.free(tile_infos);

    const w_block: []f32 = try alloc.alloc(f32, k_dim_g * oc_tile_max);
    defer alloc.free(w_block);

    var ti: usize = 0;
    var g: usize = 0;
    while (g < groups) : (g += 1) {
        const oc_base: usize = g * c_out_g;
        const ic_base: usize = g * c_in_g;

        var oc0: usize = 0;
        while (oc0 < c_out_g) : (oc0 += oc_tile_max) {
            const oc_count: usize = @min(oc_tile_max, c_out_g - oc0);
            const oc_start: usize = oc_base + oc0;

            try fillWeightBlock(w_block[0 .. k_dim_g * oc_count], w_packed, k_dim_g, c_out, oc_start, oc_count);
            const w_block_vals: []align(1) const f32 = w_block[0 .. k_dim_g * oc_count];

            const key_g: PackedWeightKey = .{
                    .w_id = s.w,
                .oc_start = oc_start,
                .k_dim = k_dim_g,
                .c_out = oc_count,
                .groups = s.groups,
                .kc = kc,
                .nc = ctx.matmul_f32.tuning.nc,
            };

            const packed_w_g: PackedWeightEntry = try getOrCreatePackedWeights(ctx.cache, matmul, key_g, w_block_vals);
            tile_infos[ti] = .{ .oc_start = oc_start, .oc_count = oc_count, .ic_base = ic_base, .packed_w = packed_w_g };
            ti += 1;
        }
    }
    std.debug.assert(ti == total_tiles);

    const Task = struct {
        ctx: *ConvExecCtx,
        s: StepConv1D,
        matmul: matmul_registry.F32Kernels,
        use_local_scratch: bool,
        params: struct {
            l_in: usize,
            l_out: usize,
            c_in: usize,
            c_out: usize,
            k: usize,
            c_in_g: usize,
            k_dim_g: usize,
        },
        x: []const f32,
        out: []f32,
        tile_infos: []const TileInfo,
        bias: []const f32,
        kw_map: []const usize,
        ic_map: []const usize,
        x_offset_map: []const usize,
        kc: usize,
        m_cap: usize,
        oc_tile_max: usize,
        groups: usize,
        tiles_per_group: usize,
        is_pointwise_unit: bool,
        is_stride1_dilation1: bool,
        is_k3_same_regular: bool,
        is_stride1_pad0_contig: bool,
        is_stride1_contig_groups1: bool,
        alloc: std.mem.Allocator,

        fn runRowsRange(t: *const @This(), scratch: []align(32) u8, start: usize, end: usize) ExecuteProgramError!void {
            const m_cap_local: usize = t.m_cap;
            const a_panel: []f32 = try t.alloc.alloc(f32, m_cap_local * t.kc);
            defer t.alloc.free(a_panel);
            const c_blocks: []f32 = try t.alloc.alloc(f32, t.tiles_per_group * m_cap_local * t.oc_tile_max);
            defer t.alloc.free(c_blocks);

            var local_scratch: []align(32) u8 = &[_]u8{};
            if (t.use_local_scratch) {
                local_scratch = try t.alloc.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), t.matmul.scratch_bytes);
                defer t.alloc.free(local_scratch);
            }
            const scratch_use: []align(32) u8 = if (t.use_local_scratch) local_scratch else scratch;

            // Index maps for the current M rows. Must scale with matmul tuning MC.
            const b_idx: []usize = try t.alloc.alloc(usize, m_cap_local);
            defer t.alloc.free(b_idx);
            const lo_idx: []usize = try t.alloc.alloc(usize, m_cap_local);
            defer t.alloc.free(lo_idx);
            const tile_stride: usize = m_cap_local * t.oc_tile_max;
            const bias_present: bool = (t.bias.len != 0);
            const use_reflect: bool = (t.s.pad_mode == .reflect);
            const l_in_i: isize = @as(isize, @intCast(t.params.l_in));

            var row0: usize = start;
            while (row0 < end) {
                const m_rows: usize = @min(m_cap_local, end - row0);

                var mr0: usize = 0;
                while (mr0 < m_rows) : (mr0 += 1) {
                    const row: usize = row0 + mr0;
                    const b: usize = row / t.params.l_out;
                    const lo: usize = row - b * t.params.l_out;
                    b_idx[mr0] = b;
                    lo_idx[mr0] = lo;
                }

                var gg: usize = 0;
                while (gg < t.groups) : (gg += 1) {
                    const ic_base: usize = gg * t.params.c_in_g;
                    const tile_base: usize = gg * t.tiles_per_group;
                    const block_count: usize = t.tile_infos[tile_base].packed_w.block_count;

                    var bi: usize = 0;
                    while (bi < block_count) : (bi += 1) {
                        const kk0: usize = bi * t.kc;
                        const k_sub: usize = @min(t.kc, t.params.k_dim_g - kk0);

                        if (t.is_pointwise_unit) {
                            var mr: usize = 0;
                            while (mr < m_rows) : (mr += 1) {
                                const b: usize = b_idx[mr];
                                const lo: usize = lo_idx[mr];
                                const x_base: usize = ((b * t.params.l_in + lo) * t.params.c_in) + ic_base + kk0;
                                @memcpy(a_panel[mr * k_sub .. mr * k_sub + k_sub], t.x[x_base .. x_base + k_sub]);
                            }
                        } else if (t.is_k3_same_regular) {
                            var mr: usize = 0;
                            while (mr < m_rows) : (mr += 1) {
                                const b: usize = b_idx[mr];
                                const lo: usize = lo_idx[mr];
                                const interior: bool = (lo > 0 and lo + 1 < t.params.l_in);

                                if (interior) {
                                    var rem: usize = k_sub;
                                    var gk: usize = kk0;
                                    var dst_off: usize = 0;
                                    while (rem > 0) {
                                        const kw: usize = gk / t.params.c_in_g;
                                        const ic0: usize = gk - kw * t.params.c_in_g;
                                        const seg: usize = @min(rem, t.params.c_in_g - ic0);
                                        const li: usize = lo + kw - 1;
                                        const src_base: usize = ((b * t.params.l_in + li) * t.params.c_in) + ic_base + ic0;
                                        @memcpy(a_panel[mr * k_sub + dst_off .. mr * k_sub + dst_off + seg], t.x[src_base .. src_base + seg]);
                                        gk += seg;
                                        dst_off += seg;
                                        rem -= seg;
                                    }
                                } else {
                                    var kl: usize = 0;
                                    while (kl < k_sub) : (kl += 1) {
                                        const gk: usize = kk0 + kl;
                                        const kw: usize = t.kw_map[gk];
                                        const ic: usize = t.ic_map[gk];

                                        const pos0: usize = lo * t.s.stride + kw * t.s.dilation;
                                        var xv: f32 = 0.0;
                                        const li_nom: isize = @as(isize, @intCast(pos0)) - @as(isize, @intCast(t.s.pad_left));
                                        if (use_reflect) {
                                            const li: usize = reflectIndex1D(li_nom, t.params.l_in);
                                            const x_idx: usize = ((b * t.params.l_in + li) * t.params.c_in) + ic_base + ic;
                                            xv = t.x[x_idx];
                                        } else if (li_nom >= 0 and li_nom < l_in_i) {
                                            const li: usize = @intCast(li_nom);
                                            const x_idx: usize = ((b * t.params.l_in + li) * t.params.c_in) + ic_base + ic;
                                            xv = t.x[x_idx];
                                        }
                                        a_panel[mr * k_sub + kl] = xv;
                                    }
                                }
                            }
                        } else if (t.is_stride1_dilation1) {
                            var mr: usize = 0;
                            while (mr < m_rows) : (mr += 1) {
                                const b: usize = b_idx[mr];
                                const lo: usize = lo_idx[mr];
                                if (t.is_stride1_pad0_contig) {
                                    const x_base: usize = ((b * t.params.l_in + lo) * t.params.c_in) + kk0;
                                    @memcpy(a_panel[mr * k_sub .. mr * k_sub + k_sub], t.x[x_base .. x_base + k_sub]);
                                } else {
                                    const interior: bool = (lo >= t.s.pad_left and (lo + t.params.k - 1) < t.params.l_in + t.s.pad_left);

                                    if (interior) {
                                        if (t.is_stride1_contig_groups1) {
                                            const lo_base: usize = lo - t.s.pad_left;
                                            const x_base: usize = ((b * t.params.l_in + lo_base) * t.params.c_in) + kk0;
                                            @memcpy(a_panel[mr * k_sub .. mr * k_sub + k_sub], t.x[x_base .. x_base + k_sub]);
                                        } else {
                                            const lo_base: usize = lo - t.s.pad_left;
                                            const row_base: usize = ((b * t.params.l_in + lo_base) * t.params.c_in) + ic_base;
                                            var rem: usize = k_sub;
                                            var gk: usize = kk0;
                                            var dst_off: usize = 0;
                                            while (rem > 0) {
                                                const ic0: usize = t.ic_map[gk];
                                                const seg: usize = @min(rem, t.params.c_in_g - ic0);
                                                const src_base: usize = row_base + t.x_offset_map[gk];
                                                @memcpy(a_panel[mr * k_sub + dst_off .. mr * k_sub + dst_off + seg], t.x[src_base .. src_base + seg]);
                                                gk += seg;
                                                dst_off += seg;
                                                rem -= seg;
                                            }
                                        }
                                    } else {
                                        var kl: usize = 0;
                                        while (kl < k_sub) : (kl += 1) {
                                            const gk: usize = kk0 + kl;
                                            const kw: usize = t.kw_map[gk];
                                            const ic: usize = t.ic_map[gk];

                                            const pos0: usize = lo + kw;
                                            var xv: f32 = 0.0;
                                            const li_nom: isize = @as(isize, @intCast(pos0)) - @as(isize, @intCast(t.s.pad_left));
                                            if (use_reflect) {
                                                const li: usize = reflectIndex1D(li_nom, t.params.l_in);
                                                const x_idx: usize = ((b * t.params.l_in + li) * t.params.c_in) + ic_base + ic;
                                                xv = t.x[x_idx];
                                            } else if (li_nom >= 0 and li_nom < l_in_i) {
                                                const li: usize = @intCast(li_nom);
                                                const x_idx: usize = ((b * t.params.l_in + li) * t.params.c_in) + ic_base + ic;
                                                xv = t.x[x_idx];
                                            }
                                            a_panel[mr * k_sub + kl] = xv;
                                        }
                                    }
                                }
                            }
                        } else {
                            var mr: usize = 0;
                            while (mr < m_rows) : (mr += 1) {
                                const b: usize = b_idx[mr];
                                const lo: usize = lo_idx[mr];

                                var kl: usize = 0;
                                while (kl < k_sub) : (kl += 1) {
                                    const gk: usize = kk0 + kl;
                                    const kw: usize = t.kw_map[gk];
                                    const ic: usize = t.ic_map[gk];

                                    const pos0: usize = lo * t.s.stride + kw * t.s.dilation;
                                    var xv: f32 = 0.0;
                                    const li_nom: isize = @as(isize, @intCast(pos0)) - @as(isize, @intCast(t.s.pad_left));
                                    if (use_reflect) {
                                        const li: usize = reflectIndex1D(li_nom, t.params.l_in);
                                        const x_idx: usize = ((b * t.params.l_in + li) * t.params.c_in) + ic_base + ic;
                                        xv = t.x[x_idx];
                                    } else if (li_nom >= 0 and li_nom < l_in_i) {
                                        const li: usize = @intCast(li_nom);
                                        const x_idx: usize = ((b * t.params.l_in + li) * t.params.c_in) + ic_base + ic;
                                        xv = t.x[x_idx];
                                    }
                                    a_panel[mr * k_sub + kl] = xv;
                                }
                            }
                        }

                        var ti1: usize = 0;
                        while (ti1 < t.tiles_per_group) : (ti1 += 1) {
                            const tile: TileInfo = t.tile_infos[tile_base + ti1];
                            const packed_w_g: PackedWeightEntry = tile.packed_w;
                            const block_elems: usize = packed_w_g.block_elems;
                            const pb_start: usize = bi * block_elems;
                            const packed_b_view: []align(32) const f32 = @alignCast(packed_w_g.blocks[pb_start .. pb_start + block_elems]);
                            const pp: MatMulParams = .{
                                .m = m_rows,
                                .n = tile.oc_count,
                                .k = k_sub,
                                .alpha = 1.0,
                                .beta = if (bi == 0) 0.0 else 1.0,
                            };
                            const c_block: []f32 = c_blocks[ti1 * tile_stride .. ti1 * tile_stride + m_rows * tile.oc_count];
                            try t.matmul.matmul_packed_b(
                                scratch_use,
                                packed_b_view,
                                pp,
                                std.mem.sliceAsBytes(c_block),
                                std.mem.sliceAsBytes(a_panel[0 .. m_rows * k_sub]),
                            );
                        }
                    }

                    var ti2: usize = 0;
                    while (ti2 < t.tiles_per_group) : (ti2 += 1) {
                        const tile2: TileInfo = t.tile_infos[tile_base + ti2];
                        const c_block2: []f32 = c_blocks[ti2 * tile_stride .. ti2 * tile_stride + m_rows * tile2.oc_count];
                        const bias_slice: []const f32 = if (bias_present) t.bias[tile2.oc_start .. tile2.oc_start + tile2.oc_count] else &[_]f32{};
                        var mr: usize = 0;
                        while (mr < m_rows) : (mr += 1) {
                            const dst_base: usize = (row0 + mr) * t.params.c_out + tile2.oc_start;
                            const src_base: usize = mr * tile2.oc_count;
                            if (bias_present) {
                                var oc: usize = 0;
                                while (oc < tile2.oc_count) : (oc += 1) {
                                    t.out[dst_base + oc] = c_block2[src_base + oc] + bias_slice[oc];
                                }
                            } else {
                                @memcpy(t.out[dst_base .. dst_base + tile2.oc_count], c_block2[src_base .. src_base + tile2.oc_count]);
                            }
                        }
                    }
                }

                row0 += m_rows;
            }
        }

        fn runRows(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) ExecuteProgramError!void {
            const t: *@This() = @ptrCast(@alignCast(ctx_any));
            const scratch: []align(32) u8 = t.ctx.matmul_scratch[tid];
            try t.runRowsRange(scratch, start, end);
        }
    };

    var task: Task = .{
        .ctx = ctx,
        .s = s,
        .matmul = matmul,
        .use_local_scratch = use_local_scratch,
        .params = .{ .l_in = l_in, .l_out = l_out, .c_in = c_in, .c_out = c_out, .k = k, .c_in_g = c_in_g, .k_dim_g = k_dim_g },
        .x = x_packed,
        .out = out_packed,
        .tile_infos = tile_infos,
        .bias = bias_packed,
        .kw_map = kw_map,
        .ic_map = ic_map,
        .x_offset_map = x_offset_map,
        .kc = kc,
        .m_cap = m_cap,
        .oc_tile_max = oc_tile_max,
        .groups = groups,
        .tiles_per_group = tiles_per_group,
        .is_pointwise_unit = is_pointwise_unit,
        .is_stride1_dilation1 = is_stride1_dilation1,
        .is_k3_same_regular = is_k3_same_regular,
        .is_stride1_pad0_contig = is_stride1_pad0_contig,
        .is_stride1_contig_groups1 = is_stride1_contig_groups1,
        .alloc = alloc,
    };

    if (ctx.pool) |p| {
        if (ctx.thread_count > 1 and rows_total >= 2 and ctx.matmul_scratch.len >= ctx.thread_count) {
            const grain: usize = @max(m_cap, @max(@as(usize, 1), rows_total / (ctx.thread_count * 4)));
            try p.parallelForFallible(ExecuteProgramError, @ptrCast(&task), rows_total, grain, Task.runRows);
            return true;
        }
    }

    const scratch: []align(32) u8 = try scratchForTid(ctx, 0);
    defer if (ctx.matmul_scratch.len == 0) ctx.allocator.free(scratch);
    try task.runRowsRange(scratch, 0, rows_total);
    return true;
}

pub fn execConv1D(ctx: *ConvExecCtx, s: StepConv1D, store: tensor_store.TensorStore) ExecuteProgramError!void {
    const out_meta: tensor_store.TensorMeta = try store.meta(s.out);
    const x_meta: tensor_store.TensorMeta = try store.meta(s.x);
    const w_meta: tensor_store.TensorMeta = try store.meta(s.w);

    std.debug.assert(out_meta.dtype == .f32 and x_meta.dtype == .f32 and w_meta.dtype == .f32);
    std.debug.assert(out_meta.rank == x_meta.rank and out_meta.rank >= 2);
    std.debug.assert(w_meta.rank == 3);
    std.debug.assert(s.groups > 0 and s.stride > 0 and s.dilation > 0);

    const rank: usize = @as(usize, out_meta.rank);
    const c_out: usize = out_meta.shape[rank - 1];
    const c_in: usize = x_meta.shape[rank - 1];

    const c_in_g: usize = w_meta.shape[1];

    std.debug.assert(c_in % s.groups == 0);
    std.debug.assert(c_out % s.groups == 0);
    std.debug.assert(c_in_g * s.groups == c_in);

    _ = try execConv1DImplicitGemm(ctx, s, out_meta, x_meta, w_meta, store);
    return;
}
