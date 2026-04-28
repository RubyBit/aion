// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const conv_utils = @import("conv_utils.zig");
const conv1d_kernels = @import("../kernels/conv1d.zig");
const simd = @import("../kernels/simd.zig");
const matmul_registry = @import("../registry/matmul_registry.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");

const BackendError = conv_utils.BackendError;
const MatMulParams = conv_utils.MatMulParams;
const ExecuteProgramError = conv_utils.ExecuteProgramError;

const StepConv1DTiled = executable.StepConv1DTiled;

const elemCountFromShape = conv_utils.elemCountFromShape;

pub const ConvExecCtx = conv_utils.ConvExecCtx;
const PackedWeightKey = conv_utils.PackedWeightKey;
const PackedWeightEntry = conv_utils.PackedWeightEntry;

const getOrCreatePackedWeights = conv_utils.getOrCreatePackedWeights;
const scratchForTid = conv_utils.scratchForTid;
const fillWeightBlock = conv_utils.fillWeightBlock;
const readTensorPackedF32 = conv_utils.readTensorPackedF32;
const writeTensorPackedF32 = conv_utils.writeTensorPackedF32;

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

fn tryExecConv1DDepthwiseTileNative(
    ctx: *ConvExecCtx,
    s: StepConv1DTiled,
    out_meta: tensor_store.TensorMeta,
    x_meta: tensor_store.TensorMeta,
    w_meta: tensor_store.TensorMeta,
    store: tensor_store.TensorStore,
) ExecuteProgramError!bool {
    // Setup/dispatch for the depthwise tile-native kernel.
    // The actual compute loop lives in kernels/conv1d.zig.

    const rank: usize = @as(usize, out_meta.rank);
    if (rank != 3) return false;
    if (out_meta.tile_counts.len != 3 or x_meta.tile_counts.len != 3 or w_meta.tile_counts.len != 3) return false;

    const batch: usize = out_meta.shape[0];
    const l_out: usize = out_meta.shape[1];
    const c_out: usize = out_meta.shape[2];

    const l_in: usize = x_meta.shape[1];
    const c_in: usize = x_meta.shape[2];

    const k: usize = w_meta.shape[0];
    const c_in_g: usize = w_meta.shape[1];

    // Depthwise check.
    if (s.groups == 0) return BackendError.InvalidArgument;
    if (s.groups != c_in or s.groups != c_out) return false;
    if (c_in != c_out) return false;
    if (c_in_g != 1) return false;
    if (w_meta.shape[2] != c_out) return false;
    if (l_out == 0 or c_out == 0 or l_in == 0 or k == 0) return BackendError.InvalidArgument;

    // Simple indexing requirement (matches other tile-native conv paths).
    if (x_meta.tile_shape[0] != 1 or out_meta.tile_shape[0] != 1) return false;

    // X and out channel tiling must align.
    if (x_meta.tile_counts[2] != out_meta.tile_counts[2]) return false;
    if (x_meta.tile_shape[2] != out_meta.tile_shape[2]) return false;

    // W is tiled only along K (dim0) and C_out (dim2); dim1 is size 1.
    if (w_meta.tile_counts[1] != 1 or w_meta.tile_shape[1] != 1) return false;
    if (w_meta.tile_counts[2] != out_meta.tile_counts[2]) return false;
    if (w_meta.tile_shape[2] != out_meta.tile_shape[2]) return false;

    // Bias (optional) is tiled along C_out to match output channel tiles.
    var bias_present: bool = false;
    if (s.bias) |b_id| {
        const bias_meta: tensor_store.TensorMeta = try store.meta(b_id);
        if (bias_meta.rank != 1 or bias_meta.shape[0] != c_out or bias_meta.tile_counts.len != 1) return false;
        if (bias_meta.tile_counts[0] != out_meta.tile_counts[2]) return false;
        if (bias_meta.tile_shape[0] != out_meta.tile_shape[2]) return false;
        bias_present = true;
    }

    const x_ltc: usize = x_meta.tile_counts[1];
    const x_ctc: usize = x_meta.tile_counts[2];
    const out_ltc: usize = out_meta.tile_counts[1];
    const out_ctc: usize = out_meta.tile_counts[2];
    const w_ktc: usize = w_meta.tile_counts[0];

    if (x_ltc == 0 or x_ctc == 0 or out_ltc == 0 or out_ctc == 0 or w_ktc == 0) return BackendError.InvalidArgument;

    const alloc: std.mem.Allocator = ctx.allocator;

    // Cache all X tiles.
    const XTile = conv1d_kernels.XTile;
    const x_tile_total: usize = batch * x_ltc * x_ctc;
    const x_tiles: []XTile = try alloc.alloc(XTile, x_tile_total);
    defer alloc.free(x_tiles);

    const x_tokens: []usize = try alloc.alloc(usize, x_tile_total);
    defer alloc.free(x_tokens);
    var x_acquired: usize = 0;
    errdefer {
        var i: usize = 0;
        while (i < x_acquired) : (i += 1) store.releaseConst(x_tokens[i]);
    }

    var x_coords_buf: [3]usize = undefined;
    var b: usize = 0;
    while (b < batch) : (b += 1) {
        var xlti: usize = 0;
        while (xlti < x_ltc) : (xlti += 1) {
            var xcti: usize = 0;
            while (xcti < x_ctc) : (xcti += 1) {
                x_coords_buf = .{ b, xlti, xcti };
                const x_tile_index: usize = try tensor_store.encodeTileIndex(x_meta, x_coords_buf[0..3]);
                const x_tile = try store.acquireTileConstLinear(s.x, x_tile_index);
                x_tokens[x_acquired] = x_tile.token;
                x_acquired += 1;

                const idx: usize = (b * x_ltc + xlti) * x_ctc + xcti;
                const xv_all: []align(1) const f32 = bytesAsF32Const(x_tile.bytes);
                const l_mem: usize = @as(usize, x_tile.shape_mem[1]);
                const c_mem: usize = @as(usize, x_tile.shape_mem[2]);
                const need: usize = l_mem * c_mem;
                if (xv_all.len < need) return BackendError.InvalidArgument;
                x_tiles[idx] = .{ .vals = xv_all[0..need], .l_mem = l_mem, .c_mem = c_mem, .row_stride = c_mem };
            }
        }
    }
    defer {
        var i: usize = 0;
        while (i < x_acquired) : (i += 1) store.releaseConst(x_tokens[i]);
    }

    // Cache W tiles.
    const WTile = conv1d_kernels.WTile;
    const w_tile_total: usize = out_ctc * w_ktc;
    const w_tiles: []WTile = try alloc.alloc(WTile, w_tile_total);
    defer alloc.free(w_tiles);

    const w_tokens: []usize = try alloc.alloc(usize, w_tile_total);
    defer alloc.free(w_tokens);
    var w_acquired: usize = 0;
    errdefer {
        var i: usize = 0;
        while (i < w_acquired) : (i += 1) store.releaseConst(w_tokens[i]);
    }

    var w_coords_buf: [3]usize = undefined;
    var out_cti: usize = 0;
    while (out_cti < out_ctc) : (out_cti += 1) {
        var wkti: usize = 0;
        while (wkti < w_ktc) : (wkti += 1) {
            w_coords_buf = .{ wkti, 0, out_cti };
            const w_tile_index: usize = try tensor_store.encodeTileIndex(w_meta, w_coords_buf[0..3]);
            const w_tile = try store.acquireTileConstLinear(s.w, w_tile_index);
            w_tokens[w_acquired] = w_tile.token;
            w_acquired += 1;

            const idx: usize = out_cti * w_ktc + wkti;
            const wv_all: []align(1) const f32 = bytesAsF32Const(w_tile.bytes);
            const k_mem: usize = @as(usize, w_tile.shape_mem[0]);
            const c_mem: usize = @as(usize, w_tile.shape_mem[2]);
            const need: usize = k_mem * c_mem;
            if (wv_all.len < need) return BackendError.InvalidArgument;
            w_tiles[idx] = .{ .vals = wv_all[0..need], .k_mem = k_mem, .c_mem = c_mem };
        }
    }
    defer {
        var i: usize = 0;
        while (i < w_acquired) : (i += 1) store.releaseConst(w_tokens[i]);
    }

    // Cache bias tiles (optional).
    const bias_slices: [][]align(1) const f32 = if (bias_present) try alloc.alloc([]align(1) const f32, out_ctc) else &[_][]align(1) const f32{};
    defer if (bias_present) alloc.free(bias_slices);

    const bias_tokens: []usize = if (bias_present) try alloc.alloc(usize, out_ctc) else &[_]usize{};
    defer if (bias_present) alloc.free(bias_tokens);

    var bias_acquired: usize = 0;
    errdefer {
        if (bias_present) {
            var i: usize = 0;
            while (i < bias_acquired) : (i += 1) store.releaseConst(bias_tokens[i]);
        }
    }

    if (bias_present) {
        const b_id: tensor_store.TensorId = s.bias.?;
        out_cti = 0;
        while (out_cti < out_ctc) : (out_cti += 1) {
            const b_tile = try store.acquireTileConstLinear(b_id, out_cti);
            bias_tokens[bias_acquired] = b_tile.token;
            bias_acquired += 1;

            const oc_count: usize = @as(usize, b_tile.shape_mem[0]);
            const b_all: []align(1) const f32 = bytesAsF32Const(b_tile.bytes);
            if (b_all.len < oc_count) return BackendError.InvalidArgument;
            bias_slices[out_cti] = b_all[0..oc_count];
        }
    }
    defer if (bias_present) {
        var i: usize = 0;
        while (i < bias_acquired) : (i += 1) store.releaseConst(bias_tokens[i]);
    };

    // Pre-acquire all output tiles.
    const out_tile_total: usize = batch * out_ltc * out_ctc;
    const out_tiles_all: [][]align(1) f32 = try alloc.alloc([]align(1) f32, out_tile_total);
    defer alloc.free(out_tiles_all);

    const out_tokens_all: []usize = try alloc.alloc(usize, out_tile_total);
    defer alloc.free(out_tokens_all);
    var out_acquired: usize = 0;
    errdefer {
        var i: usize = 0;
        while (i < out_acquired) : (i += 1) store.releaseMut(out_tokens_all[i]);
    }

    const out_l_mems: []usize = try alloc.alloc(usize, batch * out_ltc);
    defer alloc.free(out_l_mems);

    var out_coords_buf: [3]usize = undefined;
    b = 0;
    while (b < batch) : (b += 1) {
        var out_lti: usize = 0;
        while (out_lti < out_ltc) : (out_lti += 1) {
            var l_mem_ref: usize = 0;
            out_cti = 0;
            while (out_cti < out_ctc) : (out_cti += 1) {
                out_coords_buf = .{ b, out_lti, out_cti };
                const out_tile_index: usize = try tensor_store.encodeTileIndex(out_meta, out_coords_buf[0..3]);
                const out_tile = try store.acquireTileMutLinear(s.out, out_tile_index);
                out_tokens_all[out_acquired] = out_tile.token;
                out_acquired += 1;

                const idx: usize = (b * out_ltc + out_lti) * out_ctc + out_cti;

                const l_mem: usize = @as(usize, out_tile.shape_mem[1]);
                const c_mem: usize = @as(usize, out_tile.shape_mem[2]);
                if (out_cti == 0) {
                    l_mem_ref = l_mem;
                    out_l_mems[b * out_ltc + out_lti] = l_mem;
                } else if (l_mem != l_mem_ref) {
                    return BackendError.InvalidArgument;
                }

                const ov_all: []align(1) f32 = bytesAsF32Mut(out_tile.bytes);
                if (ov_all.len < l_mem * c_mem) return BackendError.InvalidArgument;
                out_tiles_all[idx] = ov_all[0 .. l_mem * c_mem];
            }
        }
    }
    defer {
        var i: usize = 0;
        while (i < out_acquired) : (i += 1) store.releaseMut(out_tokens_all[i]);
    }

    const out_tl: usize = out_meta.tile_shape[1];
    const out_tc: usize = out_meta.tile_shape[2];
    const x_tl: usize = x_meta.tile_shape[1];
    const w_tk: usize = w_meta.tile_shape[0];

    const fast_stride1_dil1: bool = (s.pad_mode != .reflect and s.stride == 1 and s.dilation == 1 and x_ltc == 1 and w_ktc == 1);

    var task: conv1d_kernels.DepthwiseConv1DTask = .{
        .p = .{ .stride = s.stride, .dilation = s.dilation, .pad_left = s.pad_left, .reflect = s.pad_mode == .reflect },
        .batch = batch,
        .l_in = l_in,
        .l_out = l_out,
        .c = c_out,
        .k = k,
        .out_ltc = out_ltc,
        .out_ctc = out_ctc,
        .out_tl = out_tl,
        .out_tc = out_tc,
        .out_l_mems = out_l_mems,
        .x_ltc = x_ltc,
        .x_ctc = x_ctc,
        .x_tl = x_tl,
        .x_tiles = x_tiles,
        .w_tiles = w_tiles,
        .w_ktc = w_ktc,
        .w_tk = w_tk,
        .out_tiles_all = out_tiles_all,
        .bias_slices = if (bias_present) bias_slices else &[_][]align(1) const f32{},
        .fast_stride1_dil1 = fast_stride1_dil1,
    };

    const work_items: usize = batch * out_ltc * out_ctc;
    if (ctx.pool) |p| {
        if (ctx.thread_count > 1 and work_items >= 2) {
            p.parallelForAny(@ptrCast(&task), work_items, 1, ctx.depthwise_conv1d.run_items);
            return true;
        }
    }

    ctx.depthwise_conv1d.run_item_range(&task, 0, work_items);
    return true;
}

fn tryExecConv1DImplicitGemmTileNative(
    ctx: *ConvExecCtx,
    s: StepConv1DTiled,
    out_meta: tensor_store.TensorMeta,
    x_meta: tensor_store.TensorMeta,
    w_meta: tensor_store.TensorMeta,
    store: tensor_store.TensorStore,
) ExecuteProgramError!bool {
    // Support reflect padding inside the same tile-native packing loop.
    // Tile-native fast path (Conv1D NLC-style channel-last).
    // Goals:
    // - avoid packed-scalar read/write of X/out
    // - avoid per-call heap allocations in the hot loop
    // Requirements (v1):
    // - rank-3 (batch, length, channel)
    // - groups == 1
    // - batch tile shape is 1
    // - X channel dim is not tiled (tile_counts[C_in] == 1)
    // - W is tiled only along C_out, and aligns with Out's C tile
    const rank: usize = @as(usize, out_meta.rank);
    if (rank != 3) return false;
    if (s.groups != 1) return false;

    if (out_meta.tile_counts.len != 3 or x_meta.tile_counts.len != 3 or w_meta.tile_counts.len != 3) return false;
    if (x_meta.tile_shape[0] != 1 or out_meta.tile_shape[0] != 1) return false;
    if (x_meta.tile_counts[2] != 1) return false;
    if (w_meta.tile_counts[0] != 1 or w_meta.tile_counts[1] != 1) return false;
    if (w_meta.tile_counts[2] != out_meta.tile_counts[2]) return false;
    if (w_meta.tile_shape[2] != out_meta.tile_shape[2]) return false;

    const batch: usize = out_meta.shape[0];
    const c_out: usize = out_meta.shape[2];
    const l_in: usize = x_meta.shape[1];
    const c_in: usize = x_meta.shape[2];

    const k: usize = w_meta.shape[0];
    const c_in_g: usize = w_meta.shape[1];
    if (c_in_g != c_in) return false;
    if (c_out != w_meta.shape[2]) return false;

    const k_dim_g: usize = k * c_in_g;

    const matmul_default: matmul_registry.F32Kernels = ctx.matmul_f32;
    const kc: usize = matmul_default.tuning.kc;
    const m_cap: usize = matmul_default.tuning.mc;
    if (kc == 0 or m_cap == 0) return BackendError.InvalidArgument;

    // Bias is tiled along C_out.
    var bias_present: bool = false;
    var bias_meta: tensor_store.TensorMeta = undefined;
    if (s.bias) |b_id| {
        bias_meta = try store.meta(b_id);
        if (bias_meta.rank != 1 or bias_meta.shape[0] != c_out or bias_meta.tile_counts.len != 1) return false;
        if (bias_meta.tile_counts[0] != out_meta.tile_counts[2]) return false;
        if (bias_meta.tile_shape[0] != out_meta.tile_shape[2]) return false;
        bias_present = true;
    }

    const XTile = struct {
        vals: []align(1) const f32,
        l_mem: usize,
        c_mem: usize,
        row_stride: usize,
    };

    const x_ltc: usize = x_meta.tile_counts[1];
    const out_ltc: usize = out_meta.tile_counts[1];
    if (x_ltc == 0 or out_ltc == 0) return BackendError.InvalidArgument;

    const alloc: std.mem.Allocator = ctx.allocator;

    // Cache all X tiles (const) for all batches.
    const x_tile_total: usize = batch * x_ltc;
    const x_tiles: []XTile = try alloc.alloc(XTile, x_tile_total);
    defer alloc.free(x_tiles);
    const x_tokens: []usize = try alloc.alloc(usize, x_tile_total);
    defer alloc.free(x_tokens);

    var x_coords_buf: [3]usize = undefined;
    var xb: usize = 0;
    while (xb < batch) : (xb += 1) {
        var xlti: usize = 0;
        while (xlti < x_ltc) : (xlti += 1) {
            x_coords_buf = .{ xb, xlti, 0 };
            const x_tile_index: usize = try tensor_store.encodeTileIndex(x_meta, x_coords_buf[0..3]);
            const x_tile = try store.acquireTileConstLinear(s.x, x_tile_index);
            const idx: usize = xb * x_ltc + xlti;
            x_tokens[idx] = x_tile.token;

            const xv_all: []align(1) const f32 = bytesAsF32Const(x_tile.bytes);
            const l_mem: usize = @as(usize, x_tile.shape_mem[1]);
            const c_mem: usize = @as(usize, x_tile.shape_mem[2]);
            const need: usize = l_mem * c_mem;
            if (xv_all.len < need) {
                store.releaseConst(x_tile.token);
                return BackendError.InvalidArgument;
            }
            x_tiles[idx] = .{ .vals = xv_all[0..need], .l_mem = l_mem, .c_mem = c_mem, .row_stride = c_mem };
        }
    }
    defer {
        var i: usize = 0;
        while (i < x_tile_total) : (i += 1) store.releaseConst(x_tokens[i]);
    }

    // Precompute packed weights per OC tile (and keep bias tiles acquired).
    const oc_tiles: usize = out_meta.tile_counts[2];
    if (oc_tiles == 0) return BackendError.InvalidArgument;

    // Allow per-OC-tile matmul selection (primarily narrow-N variants) while keeping
    // KC constant, because KC drives the K-blocking loop structure below.
    const oc_matmuls: []matmul_registry.F32Kernels = try alloc.alloc(matmul_registry.F32Kernels, oc_tiles);
    defer alloc.free(oc_matmuls);
    var max_nc: usize = matmul_default.tuning.nc;

    const packed_ws: []PackedWeightEntry = try alloc.alloc(PackedWeightEntry, oc_tiles);
    defer alloc.free(packed_ws);
    const oc_counts: []usize = try alloc.alloc(usize, oc_tiles);
    defer alloc.free(oc_counts);

    const bias_slices: [][]align(1) const f32 = if (bias_present) try alloc.alloc([]align(1) const f32, oc_tiles) else &[_][]align(1) const f32{};
    defer if (bias_present) alloc.free(bias_slices);
    const bias_tokens: []usize = if (bias_present) try alloc.alloc(usize, oc_tiles) else &[_]usize{};
    defer if (bias_present) alloc.free(bias_tokens);

    var w_coords_buf: [3]usize = undefined;
    var oc_ti: usize = 0;
    while (oc_ti < oc_tiles) : (oc_ti += 1) {
        w_coords_buf = .{ 0, 0, oc_ti };
        const w_tile_index: usize = try tensor_store.encodeTileIndex(w_meta, w_coords_buf[0..3]);
        const w_tile = try store.acquireTileConstLinear(s.w, w_tile_index);
        defer store.releaseConst(w_tile.token);
        const w_vals_all: []align(1) const f32 = bytesAsF32Const(w_tile.bytes);
        const oc_count: usize = @as(usize, w_tile.shape_mem[2]);
        oc_counts[oc_ti] = oc_count;
        if (w_vals_all.len < k_dim_g * oc_count) return BackendError.InvalidArgument;
        const w_vals: []align(1) const f32 = w_vals_all[0 .. k_dim_g * oc_count];

        // Select a matmul variant for this OC tile. We only accept variants that
        // preserve KC (and MR/NR) to keep the surrounding packing/loop structure stable.
        var matmul_oc: matmul_registry.F32Kernels = matmul_registry.selectForConvOcTile(matmul_default, oc_count);
        if (matmul_oc.tuning.kc != kc or matmul_oc.tuning.mr != matmul_default.tuning.mr or matmul_oc.tuning.nr != matmul_default.tuning.nr) {
            matmul_oc = matmul_default;
        }
        oc_matmuls[oc_ti] = matmul_oc;
        max_nc = @max(max_nc, matmul_oc.tuning.nc);

        const oc_start: usize = oc_ti * out_meta.tile_shape[2];
        const key: PackedWeightKey = .{
            .w_id = s.w,
            .oc_start = oc_start,
            .k_dim = k_dim_g,
            .c_out = oc_count,
            .groups = 1,
            .kc = kc,
            .nc = matmul_oc.tuning.nc,
        };
        packed_ws[oc_ti] = try getOrCreatePackedWeights(matmul_oc, key, w_vals);

        if (bias_present) {
            const b_id: tensor_store.TensorId = s.bias.?;
            const b_tile = try store.acquireTileConstLinear(b_id, oc_ti);
            bias_tokens[oc_ti] = b_tile.token;
            const b_all: []align(1) const f32 = bytesAsF32Const(b_tile.bytes);
            if (b_all.len < oc_count) return BackendError.InvalidArgument;
            bias_slices[oc_ti] = b_all[0..oc_count];
        }
    }
    defer if (bias_present) {
        var i: usize = 0;
        while (i < oc_tiles) : (i += 1) store.releaseConst(bias_tokens[i]);
    };

    const full_blocks: usize = k_dim_g / kc;
    const k_tail: usize = k_dim_g - full_blocks * kc;

    // Pre-acquire all output tiles once (avoids repeated acquire/release and enables global parallelism).
    const out_tile_total: usize = batch * out_ltc * oc_tiles;
    const out_tiles_all: [][]align(1) f32 = try alloc.alloc([]align(1) f32, out_tile_total);
    defer alloc.free(out_tiles_all);
    const out_tokens_all: []usize = try alloc.alloc(usize, out_tile_total);
    defer alloc.free(out_tokens_all);
    const out_l_mems: []usize = try alloc.alloc(usize, batch * out_ltc);
    defer alloc.free(out_l_mems);

    var out_coords_buf: [3]usize = undefined;
    var bb: usize = 0;
    while (bb < batch) : (bb += 1) {
        var lti: usize = 0;
        while (lti < out_ltc) : (lti += 1) {
            var l_mem_ref: usize = 0;
            var oc_ti2: usize = 0;
            while (oc_ti2 < oc_tiles) : (oc_ti2 += 1) {
                out_coords_buf = .{ bb, lti, oc_ti2 };
                const out_tile_index: usize = try tensor_store.encodeTileIndex(out_meta, out_coords_buf[0..3]);
                const out_tile = try store.acquireTileMutLinear(s.out, out_tile_index);

                const idx: usize = ((bb * out_ltc + lti) * oc_tiles) + oc_ti2;
                out_tokens_all[idx] = out_tile.token;

                const l_mem: usize = @as(usize, out_tile.shape_mem[1]);
                const c_mem: usize = @as(usize, out_tile.shape_mem[2]);
                if (c_mem != oc_counts[oc_ti2]) return BackendError.InvalidArgument;
                if (oc_ti2 == 0) {
                    l_mem_ref = l_mem;
                    out_l_mems[bb * out_ltc + lti] = l_mem;
                } else if (l_mem != l_mem_ref) {
                    return BackendError.InvalidArgument;
                }

                const ov_all: []align(1) f32 = bytesAsF32Mut(out_tile.bytes);
                if (ov_all.len < l_mem * c_mem) return BackendError.InvalidArgument;
                out_tiles_all[idx] = ov_all[0 .. l_mem * c_mem];
            }
        }
    }
    defer {
        var i: usize = 0;
        while (i < out_tile_total) : (i += 1) store.releaseMut(out_tokens_all[i]);
    }

    const blocks_per_tile: usize = (out_meta.tile_shape[1] + m_cap - 1) / m_cap;
    const work_items: usize = batch * out_ltc * blocks_per_tile;

    const Task = struct {
        ctx: *ConvExecCtx,
        s: StepConv1DTiled,
        // Matmul kernel selection per OC tile. All entries share KC/MR/NR.
        oc_matmuls: []const matmul_registry.F32Kernels,
        mr: usize,
        max_nc: usize,

        // Shapes.
        l_in: usize,
        c_in: usize,
        k: usize,
        batch: usize,
        out_ltc: usize,
        blocks_per_tile: usize,

        // Tiling.
        out_tl: usize,
        out_l_mems: []const usize,

        // X tiles.
        x_tiles: []const XTile,
        x_ltc: usize,
        x_tl: usize,

        // Output tiles for all (b,lti,oc_ti): indexed by ((b*out_ltc + lti)*oc_tiles + oc_ti).
        out_tiles_all: [][]align(1) f32,
        packed_ws: []const PackedWeightEntry,
        oc_counts: []const usize,
        bias_slices: [][]align(1) const f32,

        kc: usize,
        m_cap: usize,
        full_blocks: usize,
        k_tail: usize,

        inline fn reflectIndex(idx_nom: isize, len: isize) isize {
            // len must be > 1 (validated in infer/program for reflect)
            var x: isize = idx_nom;
            while (x < 0 or x >= len) {
                // Reflect padding semantics used by our reference tests.
                // Note: this mapping reflects around the edge including index 0.
                if (x < 0) x = -x else x = (2 * len - 2) - x;
            }
            return x;
        }

        fn runItemRange(t: *const @This(), scratch: []align(32) u8, start: usize, end: usize) ExecuteProgramError!void {
            @setRuntimeSafety(false);

            const pb_elems: usize = t.kc * t.max_nc;
            const pa_elems: usize = t.m_cap * t.kc;
            const scratch_f32: []align(32) f32 = @alignCast(std.mem.bytesAsSlice(f32, scratch));
            std.debug.assert(scratch_f32.len >= pb_elems + pa_elems);
            const packed_a_buf: []align(32) f32 = @alignCast(scratch_f32[pb_elems .. pb_elems + pa_elems]);

            const l_in_i: isize = @as(isize, @intCast(t.l_in));
            const use_reflect: bool = (t.s.pad_mode == .reflect);
            const max_l: isize = @as(isize, @intCast((t.k - 1) * t.s.dilation));
            const max_l_u: usize = @intCast(max_l);
            const dilation_u: usize = t.s.dilation;

            const bias_present_local: bool = (t.bias_slices.len != 0);
            const MR: usize = t.mr;
            const KC: usize = t.kc;

            var item0: usize = start;
            while (item0 < end) : (item0 += 1) {
                // Decode work item.
                const items_per_batch: usize = t.out_ltc * t.blocks_per_tile;
                const b: usize = item0 / items_per_batch;
                const rem0: usize = item0 - b * items_per_batch;
                const lti: usize = rem0 / t.blocks_per_tile;
                const bi: usize = rem0 - lti * t.blocks_per_tile;

                if (b >= t.batch) continue;
                const out_l_mem: usize = t.out_l_mems[b * t.out_ltc + lti];
                const row0: usize = bi * t.m_cap;
                if (row0 >= out_l_mem) continue;
                const row_end: usize = @min(out_l_mem, row0 + t.m_cap);
                const m_rows: usize = row_end - row0;

                const l_base: usize = lti * t.out_tl;
                const b_x_off: usize = b * t.x_ltc;
                const b_out_off: usize = (b * t.out_ltc + lti) * t.oc_counts.len;
                const x_single_len_tile: bool = (t.x_ltc == 1);
                const xt_only: XTile = if (x_single_len_tile) t.x_tiles[b_x_off] else undefined;

                if (t.full_blocks != 0) {
                    var bi_full: usize = 0;
                    while (bi_full < t.full_blocks) : (bi_full += 1) {
                        const kk0: usize = bi_full * t.kc;
                        const k_sub: usize = t.kc;
                        const panel_count: usize = (m_rows + MR - 1) / MR;

                        // Pack A once for this K-block.
                        var panel: usize = 0;
                        while (panel < panel_count) : (panel += 1) {
                            const base_pa: usize = panel * (MR * KC);
                            var r: usize = 0;
                            while (r < MR) : (r += 1) {
                                const mr: usize = panel * MR + r;
                                if (mr >= m_rows) break;

                                const lo_abs: usize = l_base + (row0 + mr);
                                const lo0: isize = @as(isize, @intCast(lo_abs)) * @as(isize, @intCast(t.s.stride)) - @as(isize, @intCast(t.s.pad_left));
                                const all_valid: bool = (lo0 >= 0 and (lo0 + max_l) < l_in_i);

                                const row_pa: []f32 = packed_a_buf[base_pa + r * KC .. base_pa + r * KC + KC];

                                // Fast-path when the entire convolution window stays within a single X length-tile.
                                var one_tile: bool = false;
                                var li_l0: usize = 0;
                                var xt0: XTile = undefined;
                                if (all_valid) {
                                    const li0_u: usize = @intCast(lo0);
                                    if (x_single_len_tile) {
                                        li_l0 = li0_u;
                                        one_tile = (li_l0 + max_l_u) < t.x_tl;
                                        if (one_tile) xt0 = xt_only;
                                    } else {
                                        const xlti0: usize = li0_u / t.x_tl;
                                        li_l0 = li0_u - xlti0 * t.x_tl;
                                        one_tile = (li_l0 + max_l_u) < t.x_tl;
                                        if (one_tile) {
                                            xt0 = t.x_tiles[b_x_off + xlti0];
                                        }
                                    }
                                }

                                var rem_k: usize = k_sub;
                                var gk: usize = kk0;
                                var a_off: usize = 0;
                                while (rem_k != 0) {
                                    const kw: usize = gk / t.c_in;
                                    const ic0: usize = gk - kw * t.c_in;
                                    const take: usize = @min(rem_k, t.c_in - ic0);
                                    const dst: []f32 = row_pa[a_off .. a_off + take];

                                    if (all_valid) {
                                        if (one_tile) {
                                            const li_l: usize = li_l0 + kw * dilation_u;
                                            const src0: usize = li_l * xt0.row_stride + ic0;
                                            @memcpy(dst, xt0.vals[src0 .. src0 + take]);
                                        } else {
                                            const li_u: usize = @intCast(lo0 + @as(isize, @intCast(kw * dilation_u)));
                                            const xlti: usize = li_u / t.x_tl;
                                            const li_l: usize = li_u - xlti * t.x_tl;
                                            const xt: XTile = t.x_tiles[b_x_off + xlti];
                                            const src0: usize = li_l * xt.row_stride + ic0;
                                            @memcpy(dst, xt.vals[src0 .. src0 + take]);
                                        }
                                    } else {
                                        // Edge / padding.
                                        const li_nom: isize = lo0 + @as(isize, @intCast(kw * t.s.dilation));
                                        if (use_reflect) {
                                            // Reflect always maps into range (given validated constraints).
                                            const li: isize = reflectIndex(li_nom, l_in_i);
                                            const li_u: usize = @intCast(li);
                                            const xlti: usize = li_u / t.x_tl;
                                            const li_l: usize = li_u - xlti * t.x_tl;
                                            const xt: XTile = t.x_tiles[b_x_off + xlti];
                                            const src0: usize = li_l * xt.row_stride + ic0;
                                            @memcpy(dst, xt.vals[src0 .. src0 + take]);
                                        } else {
                                            if (li_nom >= 0 and li_nom < l_in_i) {
                                                const li_u: usize = @intCast(li_nom);
                                                const xlti: usize = li_u / t.x_tl;
                                                if (xlti < t.x_ltc) {
                                                    const li_l: usize = li_u - xlti * t.x_tl;
                                                    const xt: XTile = t.x_tiles[b_x_off + xlti];
                                                    if (li_l < xt.l_mem and (ic0 + take) <= xt.c_mem) {
                                                        const src0: usize = li_l * xt.row_stride + ic0;
                                                        @memcpy(dst, xt.vals[src0 .. src0 + take]);
                                                    } else {
                                                        @memset(dst, 0.0);
                                                    }
                                                } else {
                                                    @memset(dst, 0.0);
                                                }
                                            } else {
                                                @memset(dst, 0.0);
                                            }
                                        }
                                    }

                                    gk += take;
                                    a_off += take;
                                    rem_k -= take;
                                }
                            }
                        }

                        // Compute all OC tiles using the same packed A.
                        const beta_eff: f32 = if (bi_full == 0) 0.0 else 1.0;
                        var oc_ti2: usize = 0;
                        while (oc_ti2 < t.oc_counts.len) : (oc_ti2 += 1) {
                            const oc_count: usize = t.oc_counts[oc_ti2];
                            const out: []align(1) f32 = t.out_tiles_all[b_out_off + oc_ti2];
                            const c_base: usize = row0 * oc_count;
                            const c_len: usize = m_rows * oc_count;
                            const c_slice: []align(1) f32 = out[c_base .. c_base + c_len];

                            const pw: PackedWeightEntry = t.packed_ws[oc_ti2];
                            const block_elems: usize = pw.block_elems;
                            const pb0: usize = bi_full * block_elems;
                            const packed_b_view: []align(32) const f32 = @alignCast(pw.blocks[pb0 .. pb0 + block_elems]);
                            const pp: MatMulParams = .{ .m = m_rows, .n = oc_count, .k = k_sub, .ldc = oc_count, .alpha = 1.0, .beta = beta_eff };
                            const mk: matmul_registry.F32Kernels = t.oc_matmuls[oc_ti2];
                            try mk.matmul_packed_ab(packed_a_buf, packed_b_view, pp, std.mem.sliceAsBytes(c_slice));
                        }
                    }
                }

                if (t.k_tail != 0) {
                    const kk0: usize = t.full_blocks * t.kc;
                    const k_sub: usize = t.k_tail;
                    const panel_count: usize = (m_rows + MR - 1) / MR;

                    var panel: usize = 0;
                    while (panel < panel_count) : (panel += 1) {
                        const base_pa: usize = panel * (MR * KC);
                        var r: usize = 0;
                        while (r < MR) : (r += 1) {
                            const mr: usize = panel * MR + r;
                            if (mr >= m_rows) break;

                            const lo_abs: usize = l_base + (row0 + mr);
                            const lo0: isize = @as(isize, @intCast(lo_abs)) * @as(isize, @intCast(t.s.stride)) - @as(isize, @intCast(t.s.pad_left));
                            const all_valid: bool = (lo0 >= 0 and (lo0 + max_l) < l_in_i);

                            const row_pa: []f32 = packed_a_buf[base_pa + r * KC .. base_pa + r * KC + KC];

                            var one_tile: bool = false;
                            var li_l0: usize = 0;
                            var xt0: XTile = undefined;
                            if (all_valid) {
                                const li0_u: usize = @intCast(lo0);
                                if (x_single_len_tile) {
                                    li_l0 = li0_u;
                                    one_tile = (li_l0 + max_l_u) < t.x_tl;
                                    if (one_tile) xt0 = xt_only;
                                } else {
                                    const xlti0: usize = li0_u / t.x_tl;
                                    li_l0 = li0_u - xlti0 * t.x_tl;
                                    one_tile = (li_l0 + max_l_u) < t.x_tl;
                                    if (one_tile) {
                                        xt0 = t.x_tiles[b_x_off + xlti0];
                                    }
                                }
                            }

                            var rem_k: usize = k_sub;
                            var gk: usize = kk0;
                            var a_off: usize = 0;
                            while (rem_k != 0) {
                                const kw: usize = gk / t.c_in;
                                const ic0: usize = gk - kw * t.c_in;
                                const take: usize = @min(rem_k, t.c_in - ic0);
                                const dst: []f32 = row_pa[a_off .. a_off + take];

                                if (all_valid) {
                                    if (one_tile) {
                                        const li_l: usize = li_l0 + kw * dilation_u;
                                        const src0: usize = li_l * xt0.row_stride + ic0;
                                        @memcpy(dst, xt0.vals[src0 .. src0 + take]);
                                    } else {
                                        const li_u: usize = @intCast(lo0 + @as(isize, @intCast(kw * dilation_u)));
                                        const xlti: usize = li_u / t.x_tl;
                                        const li_l: usize = li_u - xlti * t.x_tl;
                                        const xt: XTile = t.x_tiles[b_x_off + xlti];
                                        const src0: usize = li_l * xt.row_stride + ic0;
                                        @memcpy(dst, xt.vals[src0 .. src0 + take]);
                                    }
                                } else {
                                    const li_nom: isize = lo0 + @as(isize, @intCast(kw * t.s.dilation));
                                    if (use_reflect) {
                                        const li: isize = reflectIndex(li_nom, l_in_i);
                                        const li_u: usize = @intCast(li);
                                        const xlti: usize = li_u / t.x_tl;
                                        const li_l: usize = li_u - xlti * t.x_tl;
                                        const xt: XTile = t.x_tiles[b_x_off + xlti];
                                        const src0: usize = li_l * xt.row_stride + ic0;
                                        @memcpy(dst, xt.vals[src0 .. src0 + take]);
                                    } else {
                                        if (li_nom >= 0 and li_nom < l_in_i) {
                                            const li_u: usize = @intCast(li_nom);
                                            const xlti: usize = li_u / t.x_tl;
                                            if (xlti < t.x_ltc) {
                                                const li_l: usize = li_u - xlti * t.x_tl;
                                                const xt: XTile = t.x_tiles[b_x_off + xlti];
                                                if (li_l < xt.l_mem and (ic0 + take) <= xt.c_mem) {
                                                    const src0: usize = li_l * xt.row_stride + ic0;
                                                    @memcpy(dst, xt.vals[src0 .. src0 + take]);
                                                } else {
                                                    @memset(dst, 0.0);
                                                }
                                            } else {
                                                @memset(dst, 0.0);
                                            }
                                        } else {
                                            @memset(dst, 0.0);
                                        }
                                    }
                                }

                                gk += take;
                                a_off += take;
                                rem_k -= take;
                            }
                        }
                    }

                    const beta_eff: f32 = if (t.full_blocks == 0) 0.0 else 1.0;
                    var oc_ti2: usize = 0;
                    while (oc_ti2 < t.oc_counts.len) : (oc_ti2 += 1) {
                        const oc_count: usize = t.oc_counts[oc_ti2];
                        const out: []align(1) f32 = t.out_tiles_all[b_out_off + oc_ti2];
                        const c_base: usize = row0 * oc_count;
                        const c_len: usize = m_rows * oc_count;
                        const c_slice: []align(1) f32 = out[c_base .. c_base + c_len];

                        const pw: PackedWeightEntry = t.packed_ws[oc_ti2];
                        const block_elems: usize = pw.block_elems;
                        const pb0: usize = t.full_blocks * block_elems;
                        const packed_b_view: []align(32) const f32 = @alignCast(pw.blocks[pb0 .. pb0 + block_elems]);
                        const pp: MatMulParams = .{ .m = m_rows, .n = oc_count, .k = k_sub, .ldc = oc_count, .alpha = 1.0, .beta = beta_eff };
                        const mk: matmul_registry.F32Kernels = t.oc_matmuls[oc_ti2];
                        try mk.matmul_packed_ab(packed_a_buf, packed_b_view, pp, std.mem.sliceAsBytes(c_slice));
                    }
                }

                if (bias_present_local) {
                    const lanes: usize = 8;
                    const Vec = @Vector(lanes, f32);
                    var oc_ti2: usize = 0;
                    while (oc_ti2 < t.oc_counts.len) : (oc_ti2 += 1) {
                        const oc_count: usize = t.oc_counts[oc_ti2];
                        const out: []align(1) f32 = t.out_tiles_all[b_out_off + oc_ti2];
                        const bias: []align(1) const f32 = t.bias_slices[oc_ti2];

                        var oc: usize = 0;
                        while (oc + lanes <= oc_count) : (oc += lanes) {
                            const bias_v: Vec = @as(*align(1) const Vec, @ptrCast(bias.ptr + oc)).*;
                            var mr: usize = 0;
                            while (mr < m_rows) : (mr += 1) {
                                const row_base: usize = (row0 + mr) * oc_count + oc;
                                const c_ptr = out.ptr + row_base;
                                const c_v: Vec = @as(*align(1) const Vec, @ptrCast(c_ptr)).*;
                                @as(*align(1) Vec, @ptrCast(c_ptr)).* = c_v + bias_v;
                            }
                        }
                        while (oc < oc_count) : (oc += 1) {
                            var mr: usize = 0;
                            while (mr < m_rows) : (mr += 1) {
                                out[(row0 + mr) * oc_count + oc] += bias[oc];
                            }
                        }
                    }
                }
            }
        }

        fn runItems(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
            const t: *@This() = @ptrCast(@alignCast(ctx_any));
            if (tid >= t.ctx.matmul_scratch.len) return;
            const scratch_bytes: []align(32) u8 = t.ctx.matmul_scratch[tid];
            t.runItemRange(scratch_bytes, start, end) catch return;
        }
    };

    var task: Task = .{
        .ctx = ctx,
        .s = s,
        .oc_matmuls = oc_matmuls,
        .mr = matmul_default.tuning.mr,
        .max_nc = max_nc,
        .l_in = l_in,
        .c_in = c_in,
        .k = k,
        .batch = batch,
        .out_ltc = out_ltc,
        .blocks_per_tile = blocks_per_tile,
        .out_tl = out_meta.tile_shape[1],
        .out_l_mems = out_l_mems,
        .x_tiles = x_tiles,
        .x_ltc = x_ltc,
        .x_tl = x_meta.tile_shape[1],
        .out_tiles_all = out_tiles_all,
        .packed_ws = packed_ws,
        .oc_counts = oc_counts,
        .bias_slices = if (bias_present) bias_slices else &[_][]align(1) const f32{},
        .kc = kc,
        .m_cap = m_cap,
        .full_blocks = full_blocks,
        .k_tail = k_tail,
    };

    if (ctx.pool) |p| {
        if (ctx.thread_count > 1 and work_items >= 2 and ctx.matmul_scratch.len >= ctx.thread_count) {
            p.parallelForAny(@ptrCast(&task), work_items, 1, Task.runItems);
            return true;
        }
    }

    // Fallback single-thread.
    const scratch0: []align(32) u8 = try scratchForTid(ctx, 0);
    defer if (ctx.matmul_scratch.len == 0) ctx.allocator.free(scratch0);
    try task.runItemRange(scratch0, 0, work_items);

    return true;
}

/// Fast path for small convolutions (e.g. Silero VAD feature extraction).
///
/// Bypasses tiling infrastructure entirely: reads flat, convolves directly with
/// SIMD broadcast-accumulate, writes flat.  Only groups=1, rank-3, f32 and
/// small element counts.
fn tryExecConv1DSmallDirect(
    ctx: *ConvExecCtx,
    s: StepConv1DTiled,
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

    // Helper: compute tile_total.
    const calcTileTotal = struct {
        fn run(meta: tensor_store.TensorMeta) usize {
            var acc: usize = 1;
            for (meta.tile_counts) |tc| acc *= tc;
            return acc;
        }
    }.run;

    // Helper: verify a single-tile, packed-contiguous f32 layout and return the usable slice.
    const PackedTile = struct {
        vals: []align(1) const f32,
        token: usize,
    };
    const PackedTileMut = struct {
        vals: []align(1) f32,
        token: usize,
    };
    const tryAcquirePackedF32Const = struct {
        fn run(store2: tensor_store.TensorStore, meta: tensor_store.TensorMeta, id: tensor_store.TensorId) ExecuteProgramError!?PackedTile {
            if (meta.dtype != .f32) return null;
            const rank2: usize = @as(usize, meta.rank);
            if (rank2 == 0 or rank2 > 8) return null;
            if (calcTileTotal(meta) != 1) return null;
            const t = store2.acquireTileConstLinear(id, 0) catch return null;
            const view = t.bufferView();
            if (view.layout.rank != meta.rank) {
                store2.releaseConst(t.token);
                return null;
            }
            if (view.layout.shape.len != rank2 or meta.shape.len != rank2) {
                store2.releaseConst(t.token);
                return null;
            }
            var d: usize = 0;
            while (d < rank2) : (d += 1) {
                if (view.layout.shape[d] != meta.shape[d]) {
                    store2.releaseConst(t.token);
                    return null;
                }
            }
            // Require packed-contiguous row-major in bytes.
            var expect: isize = @intCast(@sizeOf(f32));
            var dd: usize = rank2;
            while (dd > 0) : (dd -= 1) {
                const i: usize = dd - 1;
                if (view.layout.strides_bytes[i] != expect) {
                    store2.releaseConst(t.token);
                    return null;
                }
                if (i > 0) {
                    const mulv: isize = @intCast(view.layout.shape[i]);
                    expect *= mulv;
                }
            }

            const total_elems: usize = elemCountFromShape(meta.shape) catch {
                store2.releaseConst(t.token);
                return null;
            };
            const need_bytes: usize = total_elems * @sizeOf(f32);
            if (t.bytes.len < need_bytes) {
                store2.releaseConst(t.token);
                return null;
            }
            const all_vals: []align(1) const f32 = bytesAsF32Const(t.bytes[0..need_bytes]);
            return .{ .vals = all_vals[0..total_elems], .token = t.token };
        }
    }.run;
    const tryAcquirePackedF32Mut = struct {
        fn run(store2: tensor_store.TensorStore, meta: tensor_store.TensorMeta, id: tensor_store.TensorId) ExecuteProgramError!?PackedTileMut {
            if (meta.dtype != .f32) return null;
            const rank2: usize = @as(usize, meta.rank);
            if (rank2 == 0 or rank2 > 8) return null;
            if (calcTileTotal(meta) != 1) return null;
            var t = store2.acquireTileMutLinear(id, 0) catch return null;
            const view = t.bufferView();
            if (view.layout.rank != meta.rank) {
                store2.releaseMut(t.token);
                return null;
            }
            if (view.layout.shape.len != rank2 or meta.shape.len != rank2) {
                store2.releaseMut(t.token);
                return null;
            }
            var d: usize = 0;
            while (d < rank2) : (d += 1) {
                if (view.layout.shape[d] != meta.shape[d]) {
                    store2.releaseMut(t.token);
                    return null;
                }
            }
            var expect: isize = @intCast(@sizeOf(f32));
            var dd: usize = rank2;
            while (dd > 0) : (dd -= 1) {
                const i: usize = dd - 1;
                if (view.layout.strides_bytes[i] != expect) {
                    store2.releaseMut(t.token);
                    return null;
                }
                if (i > 0) {
                    const mulv: isize = @intCast(view.layout.shape[i]);
                    expect *= mulv;
                }
            }

            const total_elems: usize = elemCountFromShape(meta.shape) catch {
                store2.releaseMut(t.token);
                return null;
            };
            const need_bytes: usize = total_elems * @sizeOf(f32);
            if (t.bytes.len < need_bytes) {
                store2.releaseMut(t.token);
                return null;
            }
            const all_vals: []align(1) f32 = bytesAsF32Mut(t.bytes[0..need_bytes]);
            return .{ .vals = all_vals[0..total_elems], .token = t.token };
        }
    }.run;

    // Core compute: assumes NLC packed buffers.
    const doDirect = struct {
        fn run(
            s2: StepConv1DTiled,
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

    // Fastest path: all tensors are a single packed f32 tile we can operate on directly.
    // This avoids heap allocations and pack/unpack copies in readTensorPackedF32/writeTensorPackedF32.
    if (try tryAcquirePackedF32Const(store, x_meta, s.x)) |x_tile| {
        defer store.releaseConst(x_tile.token);
        if (try tryAcquirePackedF32Const(store, w_meta, s.w)) |w_tile| {
            defer store.releaseConst(w_tile.token);
            if (try tryAcquirePackedF32Mut(store, out_meta, s.out)) |out_tile| {
                defer store.releaseMut(out_tile.token);

                var bias_ptr: ?[*]align(1) const f32 = null;
                var bias_token: usize = 0;
                if (s.bias) |b_id| {
                    const b_meta: tensor_store.TensorMeta = store.meta(b_id) catch {
                        return false;
                    };
                    if (try tryAcquirePackedF32Const(store, b_meta, b_id)) |b_tile| {
                        bias_ptr = @ptrCast(b_tile.vals.ptr);
                        bias_token = b_tile.token;
                    } else {
                        return false;
                    }
                }
                defer if (bias_ptr != null) store.releaseConst(bias_token);

                doDirect(
                    s,
                    batch,
                    l_out,
                    c_out,
                    l_in,
                    c_in,
                    k_len,
                    @ptrCast(x_tile.vals.ptr),
                    @ptrCast(w_tile.vals.ptr),
                    @ptrCast(out_tile.vals.ptr),
                    bias_ptr,
                );
                return true;
            }
        }
    }

    // Fallback: keep existing behavior for nontrivial tilings/layouts.
    // (Still uses the improved register-accumulating microkernel, but with packed copies.)
    const total_f32: usize = x_n + w_n + out_n + c_out;
    const scratch: []f32 = ctx.allocator.alloc(f32, total_f32) catch return false;
    defer ctx.allocator.free(scratch);

    try readTensorPackedF32(store, x_meta, s.x, scratch[0..x_n]);
    try readTensorPackedF32(store, w_meta, s.w, scratch[x_n .. x_n + w_n]);

    var bias_ptr2: ?[*]align(1) const f32 = null;
    if (s.bias) |b_id| {
        const b_meta: tensor_store.TensorMeta = try store.meta(b_id);
        try readTensorPackedF32(store, b_meta, b_id, scratch[x_n + w_n + out_n ..][0..c_out]);
        bias_ptr2 = @ptrCast(scratch.ptr + x_n + w_n + out_n);
    }

    doDirect(
        s,
        batch,
        l_out,
        c_out,
        l_in,
        c_in,
        k_len,
        @ptrCast(scratch.ptr),
        @ptrCast(scratch.ptr + x_n),
        @ptrCast(scratch.ptr + x_n + w_n),
        bias_ptr2,
    );

    try writeTensorPackedF32(store, out_meta, s.out, scratch[x_n + w_n ..][0..out_n]);
    return true;
}

fn tileCount3(meta: tensor_store.TensorMeta) usize {
    if (meta.tile_counts.len != 3) return 0;
    return meta.tile_counts[0] * meta.tile_counts[1] * meta.tile_counts[2];
}

fn execConv1DImplicitGemm(
    ctx: *ConvExecCtx,
    s: StepConv1DTiled,
    out_meta: tensor_store.TensorMeta,
    x_meta: tensor_store.TensorMeta,
    w_meta: tensor_store.TensorMeta,
    store: tensor_store.TensorStore,
) ExecuteProgramError!bool {
    // Fast path for small convolutions — bypasses all tiling overhead.
    if (try tryExecConv1DSmallDirect(ctx, s, out_meta, x_meta, w_meta, store)) {
        return true;
    }

    // Dedicated depthwise kernel (tile-native) when groups == C and W is [K,1,C].
    if (try tryExecConv1DDepthwiseTileNative(ctx, s, out_meta, x_meta, w_meta, store)) {
        return true;
    }

    // Prefer a tile-native path when possible (avoids large scalar pack/unpack copies
    // and eliminates per-call heap allocations in the hot loop).
    if (try tryExecConv1DImplicitGemmTileNative(ctx, s, out_meta, x_meta, w_meta, store)) {
        return true;
    }

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

    const x_count: usize = try elemCountFromShape(x_meta.shape);
    const out_count: usize = try elemCountFromShape(out_meta.shape);

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

    const x_packed: []f32 = try alloc.alloc(f32, x_count);
    defer alloc.free(x_packed);
    const out_packed: []f32 = try alloc.alloc(f32, out_count);
    defer alloc.free(out_packed);

    try readTensorPackedF32(store, x_meta, s.x, x_packed);

    var bias_packed: []f32 = &[_]f32{};
    defer if (bias_packed.len != 0) alloc.free(bias_packed);
    if (s.bias) |b_id| {
        const b_meta: tensor_store.TensorMeta = try store.meta(b_id);
        std.debug.assert(b_meta.dtype == .f32 and b_meta.rank == 1 and b_meta.shape[0] == c_out);
        bias_packed = try alloc.alloc(f32, c_out);
        try readTensorPackedF32(store, b_meta, b_id, bias_packed);
    }

    if (c_out_g == 0 or ctx.matmul_f32.tuning.nc == 0) return BackendError.InvalidArgument;

    const matmul: matmul_registry.F32Kernels = ctx.matmul_f32;
    const use_local_scratch: bool = false;

    const kc: usize = matmul.tuning.kc;
    const m_cap: usize = matmul.tuning.mc;
    const oc_tile_max: usize = @min(c_out_g, matmul.tuning.nc);

    const w_count: usize = try elemCountFromShape(w_meta.shape);
    const w_packed: []f32 = try alloc.alloc(f32, w_count);
    defer alloc.free(w_packed);
    try readTensorPackedF32(store, w_meta, s.w, w_packed);

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

            const packed_w_g: PackedWeightEntry = try getOrCreatePackedWeights(matmul, key_g, w_block_vals);
            tile_infos[ti] = .{ .oc_start = oc_start, .oc_count = oc_count, .ic_base = ic_base, .packed_w = packed_w_g };
            ti += 1;
        }
    }
    std.debug.assert(ti == total_tiles);

    const Task = struct {
        ctx: *ConvExecCtx,
        s: StepConv1DTiled,
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

        fn runRows(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
            const t: *@This() = @ptrCast(@alignCast(ctx_any));
            const scratch: []align(32) u8 = t.ctx.matmul_scratch[tid];
            t.runRowsRange(scratch, start, end) catch return;
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
            p.parallelForAny(@ptrCast(&task), rows_total, grain, Task.runRows);
            try writeTensorPackedF32(store, out_meta, s.out, out_packed);
            return true;
        }
    }

    const scratch: []align(32) u8 = try scratchForTid(ctx, 0);
    defer if (ctx.matmul_scratch.len == 0) ctx.allocator.free(scratch);
    try task.runRowsRange(scratch, 0, rows_total);

    try writeTensorPackedF32(store, out_meta, s.out, out_packed);
    return true;
}

pub fn execConv1DTiled(ctx: *ConvExecCtx, s: StepConv1DTiled, store: tensor_store.TensorStore) ExecuteProgramError!void {
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
