const std = @import("std");
const conv_utils = @import("conv_utils.zig");
const conv2d_kernels = @import("../kernels/conv2d.zig");
const matmul_registry = @import("../registry/matmul_registry.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");

const BackendError = conv_utils.BackendError;
const MatMulParams = conv_utils.MatMulParams;
const ExecuteProgramError = conv_utils.ExecuteProgramError;

const StepConv2DTiled = executable.StepConv2DTiled;

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

fn tryExecConv2DDepthwiseTileNative(
    ctx: *ConvExecCtx,
    s: StepConv2DTiled,
    out_meta: tensor_store.TensorMeta,
    x_meta: tensor_store.TensorMeta,
    w_meta: tensor_store.TensorMeta,
    store: tensor_store.TensorStore,
) ExecuteProgramError!bool {
    // Setup/dispatch for the depthwise tile-native kernel.
    // Inner compute loop lives in kernels/conv2d.zig.

    const rank: usize = @as(usize, out_meta.rank);
    if (rank != 4) return false;
    if (out_meta.tile_counts.len != 4 or x_meta.tile_counts.len != 4 or w_meta.tile_counts.len != 4) return false;

    const batch: usize = out_meta.shape[0];
    const h_out: usize = out_meta.shape[1];
    const w_out: usize = out_meta.shape[2];
    const c_out: usize = out_meta.shape[3];

    const h_in: usize = x_meta.shape[1];
    const w_in: usize = x_meta.shape[2];
    const c_in: usize = x_meta.shape[3];

    const k_h: usize = w_meta.shape[0];
    const k_w: usize = w_meta.shape[1];
    const c_in_g: usize = w_meta.shape[2];

    if (s.groups == 0) return BackendError.InvalidArgument;
    if (s.groups != c_in or s.groups != c_out) return false;
    if (c_in != c_out) return false;
    if (c_in_g != 1) return false;
    if (w_meta.shape[3] != c_out) return false;

    if (batch == 0 or h_out == 0 or w_out == 0 or c_out == 0) return BackendError.InvalidArgument;
    if (h_in == 0 or w_in == 0 or k_h == 0 or k_w == 0) return BackendError.InvalidArgument;

    // Match other tile-native paths.
    if (x_meta.tile_shape[0] != 1 or out_meta.tile_shape[0] != 1) return false;

    // X and out channel tiling must align.
    if (x_meta.tile_counts[3] != out_meta.tile_counts[3]) return false;
    if (x_meta.tile_shape[3] != out_meta.tile_shape[3]) return false;

    // W: [k_h, k_w, 1, c] (dim2 is size 1, not tiled), tiled along (k_h,k_w,c).
    if (w_meta.tile_counts[2] != 1 or w_meta.tile_shape[2] != 1) return false;
    if (w_meta.tile_counts[3] != out_meta.tile_counts[3]) return false;
    if (w_meta.tile_shape[3] != out_meta.tile_shape[3]) return false;

    // Bias (optional) is tiled along C_out to match output channel tiles.
    var bias_present: bool = false;
    if (s.bias) |b_id| {
        const bias_meta: tensor_store.TensorMeta = try store.meta(b_id);
        if (bias_meta.rank != 1 or bias_meta.shape[0] != c_out or bias_meta.tile_counts.len != 1) return false;
        if (bias_meta.tile_counts[0] != out_meta.tile_counts[3]) return false;
        if (bias_meta.tile_shape[0] != out_meta.tile_shape[3]) return false;
        bias_present = true;
    }

    const x_htc: usize = x_meta.tile_counts[1];
    const x_wtc: usize = x_meta.tile_counts[2];
    const x_ctc: usize = x_meta.tile_counts[3];

    const out_htc: usize = out_meta.tile_counts[1];
    const out_wtc: usize = out_meta.tile_counts[2];
    const out_ctc: usize = out_meta.tile_counts[3];

    const w_khtc: usize = w_meta.tile_counts[0];
    const w_kwtc: usize = w_meta.tile_counts[1];

    if (x_htc == 0 or x_wtc == 0 or x_ctc == 0) return BackendError.InvalidArgument;
    if (out_htc == 0 or out_wtc == 0 or out_ctc == 0) return BackendError.InvalidArgument;
    if (w_khtc == 0 or w_kwtc == 0) return BackendError.InvalidArgument;

    const alloc: std.mem.Allocator = ctx.allocator;

    // Cache all X tiles (b,hti,wti,cti).
    const XTile = conv2d_kernels.XTile;
    const x_tile_total: usize = batch * x_htc * x_wtc * x_ctc;
    const x_tiles: []XTile = try alloc.alloc(XTile, x_tile_total);
    defer alloc.free(x_tiles);

    const x_tokens: []usize = try alloc.alloc(usize, x_tile_total);
    defer alloc.free(x_tokens);
    var x_acquired: usize = 0;
    errdefer {
        var i: usize = 0;
        while (i < x_acquired) : (i += 1) store.releaseConst(x_tokens[i]);
    }

    var x_coords_buf: [4]usize = undefined;
    var b: usize = 0;
    while (b < batch) : (b += 1) {
        var xhti: usize = 0;
        while (xhti < x_htc) : (xhti += 1) {
            var xwti: usize = 0;
            while (xwti < x_wtc) : (xwti += 1) {
                var xcti: usize = 0;
                while (xcti < x_ctc) : (xcti += 1) {
                    x_coords_buf = .{ b, xhti, xwti, xcti };
                    const x_tile_index: usize = try tensor_store.encodeTileIndex(x_meta, x_coords_buf[0..4]);
                    const x_tile = try store.acquireTileConstLinear(s.x, x_tile_index);
                    x_tokens[x_acquired] = x_tile.token;
                    x_acquired += 1;

                    const idx: usize = (((b * x_htc + xhti) * x_wtc + xwti) * x_ctc) + xcti;
                    const xv_all: []align(1) const f32 = bytesAsF32Const(x_tile.bytes);
                    const h_mem: usize = @as(usize, x_tile.shape_mem[1]);
                    const w_mem: usize = @as(usize, x_tile.shape_mem[2]);
                    const c_mem: usize = @as(usize, x_tile.shape_mem[3]);
                    const need: usize = h_mem * w_mem * c_mem;
                    if (xv_all.len < need) return BackendError.InvalidArgument;
                    x_tiles[idx] = .{ .vals = xv_all[0..need], .h_mem = h_mem, .w_mem = w_mem, .c_mem = c_mem, .row_stride = w_mem * c_mem };
                }
            }
        }
    }
    defer {
        var i: usize = 0;
        while (i < x_acquired) : (i += 1) store.releaseConst(x_tokens[i]);
    }

    // Cache W tiles for each cti and each (khti,kwti).
    const WTile = conv2d_kernels.WTile;
    const w_tiles_per_ct: usize = w_khtc * w_kwtc;
    const w_tile_total: usize = out_ctc * w_tiles_per_ct;
    const w_tiles: []WTile = try alloc.alloc(WTile, w_tile_total);
    defer alloc.free(w_tiles);

    const w_tokens: []usize = try alloc.alloc(usize, w_tile_total);
    defer alloc.free(w_tokens);
    var w_acquired: usize = 0;
    errdefer {
        var i: usize = 0;
        while (i < w_acquired) : (i += 1) store.releaseConst(w_tokens[i]);
    }

    var w_coords_buf: [4]usize = undefined;
    var out_cti: usize = 0;
    while (out_cti < out_ctc) : (out_cti += 1) {
        var khti: usize = 0;
        while (khti < w_khtc) : (khti += 1) {
            var kwti: usize = 0;
            while (kwti < w_kwtc) : (kwti += 1) {
                w_coords_buf = .{ khti, kwti, 0, out_cti };
                const w_tile_index: usize = try tensor_store.encodeTileIndex(w_meta, w_coords_buf[0..4]);
                const w_tile = try store.acquireTileConstLinear(s.w, w_tile_index);
                w_tokens[w_acquired] = w_tile.token;
                w_acquired += 1;

                const idx: usize = out_cti * w_tiles_per_ct + (khti * w_kwtc + kwti);
                const wv_all: []align(1) const f32 = bytesAsF32Const(w_tile.bytes);
                const kh_mem: usize = @as(usize, w_tile.shape_mem[0]);
                const kw_mem: usize = @as(usize, w_tile.shape_mem[1]);
                const c_mem: usize = @as(usize, w_tile.shape_mem[3]);
                const need: usize = kh_mem * kw_mem * c_mem;
                if (wv_all.len < need) return BackendError.InvalidArgument;
                w_tiles[idx] = .{ .vals = wv_all[0..need], .kh_mem = kh_mem, .kw_mem = kw_mem, .c_mem = c_mem, .kh_stride = kw_mem * c_mem };
            }
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
    const out_tile_total: usize = batch * out_htc * out_wtc * out_ctc;
    const out_tiles_all: [][]align(1) f32 = try alloc.alloc([]align(1) f32, out_tile_total);
    defer alloc.free(out_tiles_all);

    const out_tokens_all: []usize = try alloc.alloc(usize, out_tile_total);
    defer alloc.free(out_tokens_all);
    var out_acquired: usize = 0;
    errdefer {
        var i: usize = 0;
        while (i < out_acquired) : (i += 1) store.releaseMut(out_tokens_all[i]);
    }

    const out_hw_total: usize = batch * out_htc * out_wtc;
    const out_h_mems: []usize = try alloc.alloc(usize, out_hw_total);
    defer alloc.free(out_h_mems);
    const out_w_mems: []usize = try alloc.alloc(usize, out_hw_total);
    defer alloc.free(out_w_mems);

    var out_coords_buf: [4]usize = undefined;
    b = 0;
    while (b < batch) : (b += 1) {
        var ohti: usize = 0;
        while (ohti < out_htc) : (ohti += 1) {
            var owti: usize = 0;
            while (owti < out_wtc) : (owti += 1) {
                var h_mem_ref: usize = 0;
                var w_mem_ref: usize = 0;
                out_cti = 0;
                while (out_cti < out_ctc) : (out_cti += 1) {
                    out_coords_buf = .{ b, ohti, owti, out_cti };
                    const out_tile_index: usize = try tensor_store.encodeTileIndex(out_meta, out_coords_buf[0..4]);
                    const out_tile = try store.acquireTileMutLinear(s.out, out_tile_index);
                    out_tokens_all[out_acquired] = out_tile.token;
                    out_acquired += 1;

                    const idx: usize = (((b * out_htc + ohti) * out_wtc + owti) * out_ctc) + out_cti;

                    const h_mem: usize = @as(usize, out_tile.shape_mem[1]);
                    const w_mem: usize = @as(usize, out_tile.shape_mem[2]);
                    const c_mem: usize = @as(usize, out_tile.shape_mem[3]);
                    if (out_cti == 0) {
                        h_mem_ref = h_mem;
                        w_mem_ref = w_mem;
                        const hw_idx: usize = (b * out_htc + ohti) * out_wtc + owti;
                        out_h_mems[hw_idx] = h_mem;
                        out_w_mems[hw_idx] = w_mem;
                    } else {
                        if (h_mem != h_mem_ref or w_mem != w_mem_ref) return BackendError.InvalidArgument;
                    }

                    const ov_all: []align(1) f32 = bytesAsF32Mut(out_tile.bytes);
                    const need: usize = h_mem * w_mem * c_mem;
                    if (ov_all.len < need) return BackendError.InvalidArgument;
                    out_tiles_all[idx] = ov_all[0..need];
                }
            }
        }
    }
    defer {
        var i: usize = 0;
        while (i < out_acquired) : (i += 1) store.releaseMut(out_tokens_all[i]);
    }

    const out_th: usize = out_meta.tile_shape[1];
    const out_tw: usize = out_meta.tile_shape[2];
    const out_tc: usize = out_meta.tile_shape[3];
    const x_th: usize = x_meta.tile_shape[1];
    const x_tw: usize = x_meta.tile_shape[2];
    const w_tkh: usize = w_meta.tile_shape[0];
    const w_tkw: usize = w_meta.tile_shape[1];

    const fast_stride1_dil1: bool = (s.pad_mode != .reflect and s.stride_h == 1 and s.stride_w == 1 and s.dilation_h == 1 and s.dilation_w == 1 and x_htc == 1 and x_wtc == 1 and w_khtc == 1 and w_kwtc == 1);

    // Precompute ih/iw -> x tile indices and local indices for stride=1,dilation=1.
    // This avoids division/mod in the hot loops when spatial dims are tiled.
    var use_stride1_maps: bool = (s.pad_mode != .reflect and s.stride_h == 1 and s.stride_w == 1 and s.dilation_h == 1 and s.dilation_w == 1);
    var ih_to_xhti: []u16 = &[_]u16{};
    var ih_to_ihl: []u16 = &[_]u16{};
    var iw_to_xwti: []u16 = &[_]u16{};
    var iw_to_iwl: []u16 = &[_]u16{};
    defer {
        if (ih_to_xhti.len != 0) alloc.free(ih_to_xhti);
        if (ih_to_ihl.len != 0) alloc.free(ih_to_ihl);
        if (iw_to_xwti.len != 0) alloc.free(iw_to_xwti);
        if (iw_to_iwl.len != 0) alloc.free(iw_to_iwl);
    }

    if (use_stride1_maps) {
        // Ensure the map values fit in u16.
        if (x_htc > std.math.maxInt(u16) or x_wtc > std.math.maxInt(u16) or x_th > std.math.maxInt(u16) or x_tw > std.math.maxInt(u16)) {
            use_stride1_maps = false;
        } else {
            ih_to_xhti = try alloc.alloc(u16, h_in);
            ih_to_ihl = try alloc.alloc(u16, h_in);
            iw_to_xwti = try alloc.alloc(u16, w_in);
            iw_to_iwl = try alloc.alloc(u16, w_in);

            var ih: usize = 0;
            while (ih < h_in) : (ih += 1) {
                const xhti: usize = ih / x_th;
                const ihl: usize = ih - xhti * x_th;
                ih_to_xhti[ih] = @intCast(xhti);
                ih_to_ihl[ih] = @intCast(ihl);
            }

            var iw: usize = 0;
            while (iw < w_in) : (iw += 1) {
                const xwti: usize = iw / x_tw;
                const iwl: usize = iw - xwti * x_tw;
                iw_to_xwti[iw] = @intCast(xwti);
                iw_to_iwl[iw] = @intCast(iwl);
            }
        }
    }

    var task: conv2d_kernels.DepthwiseConv2DTask = .{
        .p = .{ .stride_h = s.stride_h, .stride_w = s.stride_w, .dilation_h = s.dilation_h, .dilation_w = s.dilation_w, .pad_top = s.pad_top, .pad_left = s.pad_left, .reflect = s.pad_mode == .reflect },
        .batch = batch,
        .h_in = h_in,
        .w_in = w_in,
        .h_out = h_out,
        .w_out = w_out,
        .c = c_out,
        .k_h = k_h,
        .k_w = k_w,
        .out_htc = out_htc,
        .out_wtc = out_wtc,
        .out_ctc = out_ctc,
        .out_th = out_th,
        .out_tw = out_tw,
        .out_tc = out_tc,
        .out_h_mems = out_h_mems,
        .out_w_mems = out_w_mems,
        .x_htc = x_htc,
        .x_wtc = x_wtc,
        .x_ctc = x_ctc,
        .x_th = x_th,
        .x_tw = x_tw,
        .x_tiles = x_tiles,
        .w_tiles = w_tiles,
        .w_khtc = w_khtc,
        .w_kwtc = w_kwtc,
        .w_tkh = w_tkh,
        .w_tkw = w_tkw,
        .out_tiles_all = out_tiles_all,
        .bias_slices = if (bias_present) bias_slices else &[_][]align(1) const f32{},
        .fast_stride1_dil1 = fast_stride1_dil1,

        .use_stride1_maps = use_stride1_maps,
        .ih_to_xhti = ih_to_xhti,
        .ih_to_ihl = ih_to_ihl,
        .iw_to_xwti = iw_to_xwti,
        .iw_to_iwl = iw_to_iwl,
    };

    const work_items: usize = batch * out_htc * out_wtc * out_ctc;
    if (ctx.pool) |p| {
        if (ctx.thread_count > 1 and work_items >= 2) {
            p.parallelForAny(@ptrCast(&task), work_items, 1, ctx.depthwise_conv2d.run_items);
            return true;
        }
    }

    ctx.depthwise_conv2d.run_item_range(&task, 0, work_items);
    return true;
}

fn tryExecConv2DImplicitGemmTileNative(
    ctx: *ConvExecCtx,
    s: StepConv2DTiled,
    out_meta: tensor_store.TensorMeta,
    x_meta: tensor_store.TensorMeta,
    w_meta: tensor_store.TensorMeta,
    store: tensor_store.TensorStore,
) ExecuteProgramError!bool {
    // Reflect padding is supported by mapping out-of-bounds (ih,iw) through a reflect
    // index transform inside the existing pack loop.
    // Fast path (tile-native) that supports spatial tiling. The key goal is to avoid
    // full-tensor packed-scalar copies for large tensors.
    // Requirements (v1):
    // - rank-4 NHWC
    // - groups == 1
    // - batch tile shape is 1
    // - X channel dim is not tiled (tile_counts[C_in] == 1)
    // - W is tiled only along C_out, and aligns with Out's C tile
    const rank: usize = @as(usize, out_meta.rank);
    if (rank != 4) return false;
    if (s.groups != 1) return false;

    if (out_meta.tile_counts.len != 4 or x_meta.tile_counts.len != 4 or w_meta.tile_counts.len != 4) return false;
    if (x_meta.tile_shape[0] != 1 or out_meta.tile_shape[0] != 1) return false;
    if (x_meta.tile_counts[3] != 1) return false;
    if (w_meta.tile_counts[0] != 1 or w_meta.tile_counts[1] != 1 or w_meta.tile_counts[2] != 1) return false;
    if (w_meta.tile_counts[3] != out_meta.tile_counts[3]) return false;
    if (w_meta.tile_shape[3] != out_meta.tile_shape[3]) return false;

    const batch: usize = out_meta.shape[0];
    const h_out: usize = out_meta.shape[1];
    const w_out: usize = out_meta.shape[2];
    const c_out: usize = out_meta.shape[3];
    const h_in: usize = x_meta.shape[1];
    const w_in: usize = x_meta.shape[2];
    const c_in: usize = x_meta.shape[3];

    const k_h: usize = w_meta.shape[0];
    const k_w: usize = w_meta.shape[1];
    const c_in_g: usize = w_meta.shape[2];
    if (c_in_g != c_in) return false;
    if (c_out != w_meta.shape[3]) return false;

    const k_dim_g: usize = k_h * k_w * c_in_g;

    const matmul_default: matmul_registry.F32Kernels = ctx.matmul_f32;
    if (matmul_default.tuning.kc == 0 or matmul_default.tuning.mc == 0 or matmul_default.tuning.nc == 0) return BackendError.InvalidArgument;

    var bias_present: bool = false;
    var bias_meta: tensor_store.TensorMeta = undefined;
    if (s.bias) |b_id| {
        bias_meta = try store.meta(b_id);
        if (bias_meta.rank != 1 or bias_meta.shape[0] != c_out or bias_meta.tile_counts.len != 1) return false;
        if (bias_meta.tile_counts[0] != out_meta.tile_counts[3]) return false;
        if (bias_meta.tile_shape[0] != out_meta.tile_shape[3]) return false;
        bias_present = true;
    }

    const XTile = struct {
        vals: []align(1) const f32,
        h_mem: usize,
        w_mem: usize,
        c_mem: usize,
        row_stride: usize,
    };

    const x_htc: usize = x_meta.tile_counts[1];
    const x_wtc: usize = x_meta.tile_counts[2];
    const out_htc: usize = out_meta.tile_counts[1];
    const out_wtc: usize = out_meta.tile_counts[2];
    if (x_htc == 0 or x_wtc == 0 or out_htc == 0 or out_wtc == 0) return BackendError.InvalidArgument;

    const alloc: std.mem.Allocator = ctx.allocator;

    // Cache all X tiles (const) for all batches.
    const x_tile_total: usize = batch * x_htc * x_wtc;
    const x_tiles: []XTile = try alloc.alloc(XTile, x_tile_total);
    defer alloc.free(x_tiles);
    const x_tokens: []usize = try alloc.alloc(usize, x_tile_total);
    defer alloc.free(x_tokens);

    var x_coords_buf: [4]usize = undefined;
    var xb: usize = 0;
    while (xb < batch) : (xb += 1) {
        var xhti: usize = 0;
        while (xhti < x_htc) : (xhti += 1) {
            var xwti: usize = 0;
            while (xwti < x_wtc) : (xwti += 1) {
                x_coords_buf = .{ xb, xhti, xwti, 0 };
                const x_tile_index: usize = try tensor_store.encodeTileIndex(x_meta, x_coords_buf[0..4]);
                const x_tile = try store.acquireTileConstLinear(s.x, x_tile_index);
                const idx: usize = (xb * x_htc + xhti) * x_wtc + xwti;
                x_tokens[idx] = x_tile.token;

                const xv_all: []align(1) const f32 = bytesAsF32Const(x_tile.bytes);
                const h_mem: usize = @as(usize, x_tile.shape_mem[1]);
                const w_mem: usize = @as(usize, x_tile.shape_mem[2]);
                const c_mem: usize = @as(usize, x_tile.shape_mem[3]);
                const need: usize = h_mem * w_mem * c_mem;
                if (xv_all.len < need) {
                    store.releaseConst(x_tile.token);
                    return BackendError.InvalidArgument;
                }
                x_tiles[idx] = .{
                    .vals = xv_all[0..need],
                    .h_mem = h_mem,
                    .w_mem = w_mem,
                    .c_mem = c_mem,
                    .row_stride = w_mem * c_mem,
                };
            }
        }
    }
    defer {
        var i: usize = 0;
        while (i < x_tile_total) : (i += 1) {
            store.releaseConst(x_tokens[i]);
        }
    }

    var w_coords_buf: [4]usize = undefined;
    var out_coords_buf: [4]usize = undefined;

    var oc_ti: usize = 0;
    while (oc_ti < out_meta.tile_counts[3]) : (oc_ti += 1) {
        // Acquire weight tile for this output-channel tile.
        w_coords_buf = .{ 0, 0, 0, oc_ti };
        const w_tile_index: usize = try tensor_store.encodeTileIndex(w_meta, w_coords_buf[0..4]);
        const w_tile = try store.acquireTileConstLinear(s.w, w_tile_index);
        defer store.releaseConst(w_tile.token);
        const w_vals_all: []align(1) const f32 = bytesAsF32Const(w_tile.bytes);
        const oc_count: usize = @as(usize, w_tile.shape_mem[3]);
        if (w_vals_all.len < k_dim_g * oc_count) return BackendError.InvalidArgument;
        const w_vals: []align(1) const f32 = w_vals_all[0 .. k_dim_g * oc_count];

        // Pick a narrower NC for small-N cases *without* reducing KC.
        // This keeps K blocking (KC) large while shrinking packed-B and scratch footprint.
        const matmul: matmul_registry.F32Kernels = matmul_registry.selectForConvOcTile(matmul_default, oc_count);
        const kc: usize = matmul.tuning.kc;
        if (kc == 0) return BackendError.InvalidArgument;

        // Heuristic: when N (= oc_count) is relatively small, a very large MC can
        // hurt by inflating the per-thread packed-A working set (MC*KC) and reducing
        // cache locality. Use a smaller effective MC in that case.
        //
        // Safe because scratch is sized for matmul.tuning.mc; using less is always valid.
        // For N==128, halving MC improves cache locality without making per-task work too small.
        // Keep it a multiple of MR (6).
        const m_cap_eff: usize = if (matmul.tuning.mc > 144 and oc_count <= 128) 144 else matmul.tuning.mc;
        if (m_cap_eff == 0) return BackendError.InvalidArgument;

        const oc_start: usize = oc_ti * out_meta.tile_shape[3];
        const key: PackedWeightKey = .{
            .w_id = s.w,
            .oc_start = oc_start,
            .k_dim = k_dim_g,
            .c_out = oc_count,
            .groups = 1,
            .kc = kc,
            .nc = matmul.tuning.nc,
        };
        const packed_w: PackedWeightEntry = try getOrCreatePackedWeights(matmul, key, w_vals);

        // Acquire bias tile (optional).
        var bias_vals: []align(1) const f32 = &[_]f32{};
        var bias_token: usize = 0;
        if (bias_present) {
            const b_id: tensor_store.TensorId = s.bias.?;
            const b_tile = try store.acquireTileConstLinear(b_id, oc_ti);
            bias_token = b_tile.token;
            const b_all: []align(1) const f32 = bytesAsF32Const(b_tile.bytes);
            if (b_all.len < oc_count) {
                store.releaseConst(b_tile.token);
                return BackendError.InvalidArgument;
            }
            bias_vals = b_all[0..oc_count];
        }
        defer if (bias_present) store.releaseConst(bias_token);

        const full_blocks: usize = k_dim_g / kc;
        const k_tail: usize = k_dim_g - full_blocks * kc;

        const Task = struct {
            ctx: *ConvExecCtx,
            s: StepConv2DTiled,
            matmul: matmul_registry.F32Kernels,
            packed_w: PackedWeightEntry,

            // Shapes.
            h_in: usize,
            w_in: usize,
            c_in: usize,
            h_out: usize,
            w_out: usize,
            oc_count: usize,
            k_h: usize,
            k_w: usize,

            // Tile mapping.
            b: usize,
            oh_base: usize,
            ow_base: usize,
            out_tile_w: usize,

            x_tiles: []const XTile,
            x_htc: usize,
            x_wtc: usize,
            x_th: usize,
            x_tw: usize,

            out_tile: []align(1) f32,
            bias: []align(1) const f32,

            kc: usize,
            m_cap: usize,
            full_blocks: usize,
            k_tail: usize,

            inline fn reflectIndex(idx_nom: isize, len: isize) isize {
                var x: isize = idx_nom;
                while (x < 0 or x >= len) {
                    if (x < 0) x = -x else x = (2 * len - 2) - x;
                }
                return x;
            }

            fn runRowsRange(t: *const @This(), scratch: []align(32) u8, start: usize, end: usize) ExecuteProgramError!void {
                @setRuntimeSafety(false);

                const pb_elems: usize = t.kc * t.matmul.tuning.nc;
                const pa_elems: usize = t.m_cap * t.kc;
                const scratch_f32: []align(32) f32 = @alignCast(std.mem.bytesAsSlice(f32, scratch));
                std.debug.assert(scratch_f32.len >= pb_elems + pa_elems);
                const packed_a_buf: []align(32) f32 = @alignCast(scratch_f32[pb_elems .. pb_elems + pa_elems]);

                const h_in_i: isize = @as(isize, @intCast(t.h_in));
                const w_in_i: isize = @as(isize, @intCast(t.w_in));
                const use_reflect: bool = (t.s.pad_mode == .reflect);
                const max_h: isize = @as(isize, @intCast((t.k_h - 1) * t.s.dilation_h));
                const max_w: isize = @as(isize, @intCast((t.k_w - 1) * t.s.dilation_w));

                const bias_present_local: bool = (t.bias.len != 0);
                const MR: usize = t.matmul.tuning.mr;
                const KC: usize = t.kc;

                // Common (fast) case for the bench configs: X is a single full tile (no spatial tiling,
                // no channel tiling). In that case we can skip tile coordinate division and bounds checks
                // when the computed (ih,iw) is known in-range.
                const single_x_tile: bool = (t.x_htc == 1 and t.x_wtc == 1);
                const xt0: XTile = if (single_x_tile) t.x_tiles[(t.b * t.x_htc) * t.x_wtc] else undefined;
                const single_full_x_tile: bool = single_x_tile and (xt0.h_mem == t.h_in) and (xt0.w_mem == t.w_in) and (xt0.c_mem == t.c_in);

                const out: []align(1) f32 = t.out_tile;
                var row0: usize = start;
                while (row0 < end) {
                    const m_rows: usize = @min(t.m_cap, end - row0);

                    // Precompute per-row spatial bases once per M-chunk. These values do not
                    // depend on the K-block, so reusing them avoids repeating integer math in
                    // the hot gather/pack loop (especially noticeable when N is small).
                    //
                    // Use fixed-size buffers sized for the maximum MC across our matmul variants.
                    // This keeps the precompute stack allocation simple and fast.
                    var oh0_buf: [288]isize = undefined;
                    var ow0_buf: [288]isize = undefined;
                    var all_valid_buf: [288]bool = undefined;
                    {
                        var mrp: usize = 0;
                        while (mrp < m_rows) : (mrp += 1) {
                            const row_in_tile: usize = row0 + mrp;
                            const oh_local: usize = row_in_tile / t.out_tile_w;
                            const ow_local: usize = row_in_tile - oh_local * t.out_tile_w;
                            const oh_abs: usize = t.oh_base + oh_local;
                            const ow_abs: usize = t.ow_base + ow_local;

                            const oh0: isize = @as(isize, @intCast(oh_abs)) * @as(isize, @intCast(t.s.stride_h)) - @as(isize, @intCast(t.s.pad_top));
                            const ow0: isize = @as(isize, @intCast(ow_abs)) * @as(isize, @intCast(t.s.stride_w)) - @as(isize, @intCast(t.s.pad_left));
                            oh0_buf[mrp] = oh0;
                            ow0_buf[mrp] = ow0;
                            all_valid_buf[mrp] = (oh0 >= 0 and ow0 >= 0 and (oh0 + max_h) < h_in_i and (ow0 + max_w) < w_in_i);
                        }
                    }

                    const c_base: usize = row0 * t.oc_count;
                    const c_len: usize = m_rows * t.oc_count;
                    const c_slice: []align(1) f32 = out[c_base .. c_base + c_len];

                    if (t.full_blocks != 0) {
                        var bi_full: usize = 0;
                        while (bi_full < t.full_blocks) : (bi_full += 1) {
                            const kk0: usize = bi_full * t.kc;
                            const k_sub: usize = t.kc;
                            const panel_count: usize = (m_rows + MR - 1) / MR;

                            var panel: usize = 0;
                            while (panel < panel_count) : (panel += 1) {
                                const base_pa: usize = panel * (MR * KC);
                                var r: usize = 0;
                                while (r < MR) : (r += 1) {
                                    const mr: usize = panel * MR + r;
                                    if (mr >= m_rows) break;

                                    const row_in_tile: usize = row0 + mr;
                                    _ = row_in_tile;
                                    const oh0: isize = oh0_buf[mr];
                                    const ow0: isize = ow0_buf[mr];
                                    const all_valid: bool = all_valid_buf[mr];

                                    const row_pa: []f32 = packed_a_buf[base_pa + r * KC .. base_pa + r * KC + KC];

                                    var rem_k: usize = k_sub;
                                    var a_off: usize = 0;
                                    // Avoid div/mod in the hot packing loop by iterating (kh,kw,ic)
                                    // with carries. This is a big win when KC is small-ish relative to C.
                                    const pos: usize = kk0 / t.c_in;
                                    var ic: usize = kk0 - pos * t.c_in;
                                    var kh: usize = pos / t.k_w;
                                    var kw: usize = pos - kh * t.k_w;
                                    while (rem_k != 0) {
                                        const take: usize = @min(rem_k, t.c_in - ic);
                                        const ih: isize = oh0 + @as(isize, @intCast(kh * t.s.dilation_h));
                                        const iw: isize = ow0 + @as(isize, @intCast(kw * t.s.dilation_w));
                                        const dst: []f32 = row_pa[a_off .. a_off + take];

                                        if (all_valid) {
                                            // Interior: in-range by construction. Keep this path as branch-light
                                            // as possible.
                                            if (single_full_x_tile) {
                                                const ih_u: usize = @intCast(ih);
                                                const iw_u: usize = @intCast(iw);
                                                const src0: usize = (ih_u * xt0.row_stride) + (iw_u * xt0.c_mem) + ic;
                                                @memcpy(dst, xt0.vals[src0 .. src0 + take]);
                                            } else {
                                                const ih_u: usize = @intCast(ih);
                                                const iw_u: usize = @intCast(iw);
                                                const xhti: usize = ih_u / t.x_th;
                                                const xwti: usize = iw_u / t.x_tw;
                                                const ih_l: usize = ih_u - xhti * t.x_th;
                                                const iw_l: usize = iw_u - xwti * t.x_tw;
                                                const xti: usize = (t.b * t.x_htc + xhti) * t.x_wtc + xwti;
                                                const xt: XTile = t.x_tiles[xti];
                                                const src0: usize = (ih_l * xt.row_stride) + (iw_l * xt.c_mem) + ic;
                                                @memcpy(dst, xt.vals[src0 .. src0 + take]);
                                            }
                                        } else {
                                            // Edge / padding.
                                            if (use_reflect) {
                                                const ih_r: usize = @intCast(reflectIndex(ih, h_in_i));
                                                const iw_r: usize = @intCast(reflectIndex(iw, w_in_i));
                                                const xhti: usize = ih_r / t.x_th;
                                                const xwti: usize = iw_r / t.x_tw;
                                                const ih_l: usize = ih_r - xhti * t.x_th;
                                                const iw_l: usize = iw_r - xwti * t.x_tw;
                                                const xti: usize = (t.b * t.x_htc + xhti) * t.x_wtc + xwti;
                                                const xt: XTile = t.x_tiles[xti];
                                                const src0: usize = (ih_l * xt.row_stride) + (iw_l * xt.c_mem) + ic;
                                                @memcpy(dst, xt.vals[src0 .. src0 + take]);
                                            } else {
                                                if (ih >= 0 and iw >= 0 and ih < h_in_i and iw < w_in_i) {
                                                    const ih_u: usize = @intCast(ih);
                                                    const iw_u: usize = @intCast(iw);
                                                    const xhti: usize = ih_u / t.x_th;
                                                    const xwti: usize = iw_u / t.x_tw;
                                                    if (xhti < t.x_htc and xwti < t.x_wtc) {
                                                        const ih_l: usize = ih_u - xhti * t.x_th;
                                                        const iw_l: usize = iw_u - xwti * t.x_tw;
                                                        const xti: usize = (t.b * t.x_htc + xhti) * t.x_wtc + xwti;
                                                        const xt: XTile = t.x_tiles[xti];
                                                        if (ih_l < xt.h_mem and iw_l < xt.w_mem and (ic + take) <= xt.c_mem) {
                                                            const src0: usize = (ih_l * xt.row_stride) + (iw_l * xt.c_mem) + ic;
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

                                        a_off += take;
                                        rem_k -= take;
                                        ic += take;
                                        if (ic == t.c_in) {
                                            ic = 0;
                                            kw += 1;
                                            if (kw == t.k_w) {
                                                kw = 0;
                                                kh += 1;
                                            }
                                        }
                                    }
                                }
                            }

                            const block_elems: usize = t.packed_w.block_elems;
                            const pb0: usize = bi_full * block_elems;
                            const packed_b_view: []align(32) const f32 = @alignCast(t.packed_w.blocks[pb0 .. pb0 + block_elems]);
                            const pp: MatMulParams = .{ .m = m_rows, .n = t.oc_count, .k = k_sub, .ldc = t.oc_count, .alpha = 1.0, .beta = if (bi_full == 0) 0.0 else 1.0 };
                            try t.matmul.matmul_packed_ab(packed_a_buf, packed_b_view, pp, std.mem.sliceAsBytes(c_slice));
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

                                const row_in_tile: usize = row0 + mr;
                                _ = row_in_tile;
                                const oh0: isize = oh0_buf[mr];
                                const ow0: isize = ow0_buf[mr];
                                const all_valid: bool = all_valid_buf[mr];

                                const row_pa: []f32 = packed_a_buf[base_pa + r * KC .. base_pa + r * KC + KC];

                                var rem_k: usize = k_sub;
                                var a_off: usize = 0;
                                const pos: usize = kk0 / t.c_in;
                                var ic: usize = kk0 - pos * t.c_in;
                                var kh: usize = pos / t.k_w;
                                var kw: usize = pos - kh * t.k_w;
                                while (rem_k != 0) {
                                    const take: usize = @min(rem_k, t.c_in - ic);
                                    const ih: isize = oh0 + @as(isize, @intCast(kh * t.s.dilation_h));
                                    const iw: isize = ow0 + @as(isize, @intCast(kw * t.s.dilation_w));
                                    const dst: []f32 = row_pa[a_off .. a_off + take];

                                    if (all_valid) {
                                        if (single_full_x_tile) {
                                            const ih_u: usize = @intCast(ih);
                                            const iw_u: usize = @intCast(iw);
                                            const src0: usize = (ih_u * xt0.row_stride) + (iw_u * xt0.c_mem) + ic;
                                            @memcpy(dst, xt0.vals[src0 .. src0 + take]);
                                        } else {
                                            const ih_u: usize = @intCast(ih);
                                            const iw_u: usize = @intCast(iw);
                                            const xhti: usize = ih_u / t.x_th;
                                            const xwti: usize = iw_u / t.x_tw;
                                            const ih_l: usize = ih_u - xhti * t.x_th;
                                            const iw_l: usize = iw_u - xwti * t.x_tw;
                                            const xti: usize = (t.b * t.x_htc + xhti) * t.x_wtc + xwti;
                                            const xt: XTile = t.x_tiles[xti];
                                            const src0: usize = (ih_l * xt.row_stride) + (iw_l * xt.c_mem) + ic;
                                            @memcpy(dst, xt.vals[src0 .. src0 + take]);
                                        }
                                    } else {
                                        if (use_reflect) {
                                            const ih_r: usize = @intCast(reflectIndex(ih, h_in_i));
                                            const iw_r: usize = @intCast(reflectIndex(iw, w_in_i));
                                            const xhti: usize = ih_r / t.x_th;
                                            const xwti: usize = iw_r / t.x_tw;
                                            const ih_l: usize = ih_r - xhti * t.x_th;
                                            const iw_l: usize = iw_r - xwti * t.x_tw;
                                            const xti: usize = (t.b * t.x_htc + xhti) * t.x_wtc + xwti;
                                            const xt: XTile = t.x_tiles[xti];
                                            const src0: usize = (ih_l * xt.row_stride) + (iw_l * xt.c_mem) + ic;
                                            @memcpy(dst, xt.vals[src0 .. src0 + take]);
                                        } else {
                                            if (ih >= 0 and iw >= 0 and ih < h_in_i and iw < w_in_i) {
                                                const ih_u: usize = @intCast(ih);
                                                const iw_u: usize = @intCast(iw);
                                                const xhti: usize = ih_u / t.x_th;
                                                const xwti: usize = iw_u / t.x_tw;
                                                if (xhti < t.x_htc and xwti < t.x_wtc) {
                                                    const ih_l: usize = ih_u - xhti * t.x_th;
                                                    const iw_l: usize = iw_u - xwti * t.x_tw;
                                                    const xti: usize = (t.b * t.x_htc + xhti) * t.x_wtc + xwti;
                                                    const xt: XTile = t.x_tiles[xti];
                                                    if (ih_l < xt.h_mem and iw_l < xt.w_mem and (ic + take) <= xt.c_mem) {
                                                        const src0: usize = (ih_l * xt.row_stride) + (iw_l * xt.c_mem) + ic;
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

                                    a_off += take;
                                    rem_k -= take;
                                    ic += take;
                                    if (ic == t.c_in) {
                                        ic = 0;
                                        kw += 1;
                                        if (kw == t.k_w) {
                                            kw = 0;
                                            kh += 1;
                                        }
                                    }
                                }
                            }
                        }

                        const block_elems: usize = t.packed_w.block_elems;
                        const pb0: usize = t.full_blocks * block_elems;
                        const packed_b_view: []align(32) const f32 = @alignCast(t.packed_w.blocks[pb0 .. pb0 + block_elems]);
                        const pp: MatMulParams = .{ .m = m_rows, .n = t.oc_count, .k = k_sub, .ldc = t.oc_count, .alpha = 1.0, .beta = if (t.full_blocks == 0) 0.0 else 1.0 };
                        try t.matmul.matmul_packed_ab(packed_a_buf, packed_b_view, pp, std.mem.sliceAsBytes(c_slice));
                    }

                    if (bias_present_local) {
                        const lanes: usize = 8;
                        const Vec = @Vector(lanes, f32);
                        var oc: usize = 0;
                        while (oc + lanes <= t.oc_count) : (oc += lanes) {
                            const bias_v: Vec = @as(*align(1) const Vec, @ptrCast(t.bias.ptr + oc)).*;
                            var mr: usize = 0;
                            while (mr < m_rows) : (mr += 1) {
                                const row_base: usize = mr * t.oc_count + oc;
                                const c_ptr = c_slice.ptr + row_base;
                                const c_v: Vec = @as(*align(1) const Vec, @ptrCast(c_ptr)).*;
                                @as(*align(1) Vec, @ptrCast(c_ptr)).* = c_v + bias_v;
                            }
                        }
                        while (oc < t.oc_count) : (oc += 1) {
                            var mr: usize = 0;
                            while (mr < m_rows) : (mr += 1) {
                                const row_base: usize = mr * t.oc_count + oc;
                                c_slice[row_base] += t.bias[oc];
                            }
                        }
                    }

                    row0 += m_rows;
                }
            }

            fn runRows(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
                const t: *@This() = @ptrCast(@alignCast(ctx_any));
                if (tid >= t.ctx.matmul_scratch.len) return;
                const scratch_bytes: []align(32) u8 = t.ctx.matmul_scratch[tid];
                t.runRowsRange(scratch_bytes, start, end) catch return;
            }
        };

        var b: usize = 0;
        while (b < batch) : (b += 1) {
            var ohti: usize = 0;
            while (ohti < out_htc) : (ohti += 1) {
                var owti: usize = 0;
                while (owti < out_wtc) : (owti += 1) {
                    out_coords_buf = .{ b, ohti, owti, oc_ti };
                    const out_tile_index: usize = try tensor_store.encodeTileIndex(out_meta, out_coords_buf[0..4]);
                    const out_tile = try store.acquireTileMutLinear(s.out, out_tile_index);
                    defer store.releaseMut(out_tile.token);

                    const out_h_mem: usize = @as(usize, out_tile.shape_mem[1]);
                    const out_w_mem: usize = @as(usize, out_tile.shape_mem[2]);
                    const out_c_mem: usize = @as(usize, out_tile.shape_mem[3]);
                    if (out_c_mem != oc_count) return BackendError.InvalidArgument;

                    const tile_rows: usize = out_h_mem * out_w_mem;
                    const ov_all: []align(1) f32 = bytesAsF32Mut(out_tile.bytes);
                    if (ov_all.len < tile_rows * oc_count) return BackendError.InvalidArgument;
                    const out_vals: []align(1) f32 = ov_all[0 .. tile_rows * oc_count];

                    var task: Task = .{
                        .ctx = ctx,
                        .s = s,
                        .matmul = matmul,
                        .packed_w = packed_w,
                        .h_in = h_in,
                        .w_in = w_in,
                        .c_in = c_in,
                        .h_out = h_out,
                        .w_out = w_out,
                        .oc_count = oc_count,
                        .k_h = k_h,
                        .k_w = k_w,
                        .b = b,
                        .oh_base = ohti * out_meta.tile_shape[1],
                        .ow_base = owti * out_meta.tile_shape[2],
                        .out_tile_w = out_w_mem,
                        .x_tiles = x_tiles,
                        .x_htc = x_htc,
                        .x_wtc = x_wtc,
                        .x_th = x_meta.tile_shape[1],
                        .x_tw = x_meta.tile_shape[2],
                        .out_tile = out_vals,
                        .bias = if (bias_present) bias_vals else &[_]f32{},
                        .kc = kc,
                        .m_cap = m_cap_eff,
                        .full_blocks = full_blocks,
                        .k_tail = k_tail,
                    };

                    if (ctx.pool) |p| {
                        if (ctx.thread_count > 1 and tile_rows >= 2 and ctx.matmul_scratch.len >= ctx.thread_count) {
                            const grain: usize = @max(m_cap_eff, @max(@as(usize, 1), tile_rows / (ctx.thread_count * 4)));
                            p.parallelForAny(@ptrCast(&task), tile_rows, grain, Task.runRows);
                        } else {
                            const scratch0: []align(32) u8 = try scratchForTid(ctx, 0);
                            defer if (ctx.matmul_scratch.len == 0) ctx.allocator.free(scratch0);
                            try task.runRowsRange(scratch0, 0, tile_rows);
                        }
                    } else {
                        const scratch0: []align(32) u8 = try scratchForTid(ctx, 0);
                        defer if (ctx.matmul_scratch.len == 0) ctx.allocator.free(scratch0);
                        try task.runRowsRange(scratch0, 0, tile_rows);
                    }
                }
            }
        }
    }

    return true;
}

fn execConv2DImplicitGemm(
    ctx: *ConvExecCtx,
    s: StepConv2DTiled,
    out_meta: tensor_store.TensorMeta,
    x_meta: tensor_store.TensorMeta,
    w_meta: tensor_store.TensorMeta,
    store: tensor_store.TensorStore,
) ExecuteProgramError!bool {
    // Dedicated depthwise kernel (tile-native) when groups == C and W is [K_h, K_w, 1, C].
    if (try tryExecConv2DDepthwiseTileNative(ctx, s, out_meta, x_meta, w_meta, store)) {
        return true;
    }

    // Prefer a tile-native path when possible (avoids large scalar pack/unpack copies).
    if (try tryExecConv2DImplicitGemmTileNative(ctx, s, out_meta, x_meta, w_meta, store)) {
        return true;
    }

    const rank: usize = @as(usize, out_meta.rank);
    const c_out: usize = out_meta.shape[rank - 1];
    const h_out: usize = out_meta.shape[rank - 3];
    const w_out: usize = out_meta.shape[rank - 2];
    const h_in: usize = x_meta.shape[rank - 3];
    const w_in: usize = x_meta.shape[rank - 2];
    const c_in: usize = x_meta.shape[rank - 1];
    const k_h: usize = w_meta.shape[0];
    const k_w: usize = w_meta.shape[1];
    const c_in_g: usize = w_meta.shape[2];

    const batch: usize = if (rank == 3) 1 else blk: {
        var acc: usize = 1;
        var d: usize = 0;
        while (d + 3 < rank) : (d += 1) acc = std.math.mul(usize, acc, out_meta.shape[d]) catch return BackendError.InvalidArgument;
        break :blk acc;
    };

    const groups: usize = s.groups;
    if (groups == 0) return BackendError.InvalidArgument;
    if (c_in % groups != 0 or c_out % groups != 0) return BackendError.InvalidArgument;
    if (c_in_g * groups != c_in) return BackendError.InvalidArgument;
    const c_out_g: usize = c_out / groups;

    const rows_total: usize = batch * h_out * w_out;
    const k_dim_g: usize = k_h * k_w * c_in_g;

    const x_count: usize = try elemCountFromShape(x_meta.shape);
    const out_count: usize = try elemCountFromShape(out_meta.shape);

    const alloc: std.mem.Allocator = std.heap.page_allocator;

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
        s: StepConv2DTiled,
        matmul: matmul_registry.F32Kernels,
        use_local_scratch: bool,
        params: struct {
            h_in: usize,
            w_in: usize,
            h_out: usize,
            w_out: usize,
            c_in: usize,
            c_out: usize,
            k_h: usize,
            k_w: usize,
            c_in_g: usize,
            k_dim_g: usize,
        },
        x: []const f32,
        out: []f32,
        tile_infos: []const TileInfo,
        bias: []const f32,
        kc: usize,
        m_cap: usize,
        oc_tile_max: usize,
        groups: usize,
        tiles_per_group: usize,
        alloc: std.mem.Allocator,

        fn runRowsRange(t: *const @This(), scratch: []align(32) u8, start: usize, end: usize) ExecuteProgramError!void {
            @setRuntimeSafety(false);
            const m_cap_local: usize = t.m_cap;
            const full_blocks: usize = t.params.k_dim_g / t.kc;
            const k_tail: usize = t.params.k_dim_g - full_blocks * t.kc;

            var local_scratch: []align(32) u8 = &[_]u8{};
            if (t.use_local_scratch) {
                local_scratch = try t.alloc.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), t.matmul.scratch_bytes);
                defer t.alloc.free(local_scratch);
            }
            const scratch_use: []align(32) u8 = if (t.use_local_scratch) local_scratch else scratch;

            // Reuse the matmul scratch's packed-A region ("pa") to avoid allocating a packed-A buffer.
            // Layout matches matmul_tuned.Kernel.splitScratch(): pb = KC*NC, pa = MC*KC.
            const pb_elems: usize = t.kc * t.matmul.tuning.nc;
            const pa_elems: usize = t.m_cap * t.kc;
            const scratch_f32: []align(32) f32 = @alignCast(std.mem.bytesAsSlice(f32, scratch_use));
            std.debug.assert(scratch_f32.len >= pb_elems + pa_elems);
            const packed_a_buf: []align(32) f32 = @alignCast(scratch_f32[pb_elems .. pb_elems + pa_elems]);

            const h_in_i: isize = @as(isize, @intCast(t.params.h_in));
            const w_in_i: isize = @as(isize, @intCast(t.params.w_in));
            const max_h: isize = @as(isize, @intCast((t.params.k_h - 1) * t.s.dilation_h));
            const max_w: isize = @as(isize, @intCast((t.params.k_w - 1) * t.s.dilation_w));
            const use_reflect: bool = (t.s.pad_mode == .reflect);

            const b_idx: []usize = try t.alloc.alloc(usize, m_cap_local);
            defer t.alloc.free(b_idx);
            const oh_idx: []usize = try t.alloc.alloc(usize, m_cap_local);
            defer t.alloc.free(oh_idx);
            const ow_idx: []usize = try t.alloc.alloc(usize, m_cap_local);
            defer t.alloc.free(ow_idx);
            const hw_out: usize = t.params.h_out * t.params.w_out;
            const bias_present: bool = (t.bias.len != 0);

            var row0: usize = start;
            while (row0 < end) {
                const m_rows: usize = @min(m_cap_local, end - row0);

                var mr0: usize = 0;
                while (mr0 < m_rows) : (mr0 += 1) {
                    const row: usize = row0 + mr0;
                    const b: usize = row / hw_out;
                    const rem: usize = row - b * hw_out;
                    b_idx[mr0] = b;
                    oh_idx[mr0] = rem / t.params.w_out;
                    ow_idx[mr0] = rem - oh_idx[mr0] * t.params.w_out;
                }

                var gg: usize = 0;
                while (gg < t.groups) : (gg += 1) {
                    const ic_base: usize = gg * t.params.c_in_g;
                    const tile_base: usize = gg * t.tiles_per_group;

                    // Compute all K-blocks for this group, reusing the same gathered A panel
                    // across all output-channel tiles.

                    if (full_blocks != 0) {
                        var bi_full: usize = 0;
                        while (bi_full < full_blocks) : (bi_full += 1) {
                            const kk0: usize = bi_full * t.kc;
                            const k_sub: usize = t.kc;

                            // Gather directly into packed-A layout: panels of MR x KC.
                            const MR: usize = t.matmul.tuning.mr;
                            const KC: usize = t.kc;
                            const panel_count: usize = (m_rows + MR - 1) / MR;

                            var panel: usize = 0;
                            while (panel < panel_count) : (panel += 1) {
                                const base_pa: usize = panel * (MR * KC);
                                var r: usize = 0;
                                while (r < MR) : (r += 1) {
                                    const mr: usize = panel * MR + r;
                                    if (mr >= m_rows) break;

                                    const b: usize = b_idx[mr];
                                    const oh: usize = oh_idx[mr];
                                    const ow: usize = ow_idx[mr];

                                    const base_batch: usize = (b * t.params.h_in * t.params.w_in * t.params.c_in) + ic_base;
                                    const oh0: isize = @as(isize, @intCast(oh)) * @as(isize, @intCast(t.s.stride_h)) - @as(isize, @intCast(t.s.pad_top));
                                    const ow0: isize = @as(isize, @intCast(ow)) * @as(isize, @intCast(t.s.stride_w)) - @as(isize, @intCast(t.s.pad_left));

                                    const all_valid: bool = (oh0 >= 0 and ow0 >= 0 and (oh0 + max_h) < h_in_i and (ow0 + max_w) < w_in_i);

                                    const row_pa: []f32 = packed_a_buf[base_pa + r * KC .. base_pa + r * KC + KC];

                                    var rem_k: usize = k_sub;
                                    var gk: usize = kk0;
                                    var a_off: usize = 0;
                                    while (rem_k != 0) {
                                        const pos: usize = gk / t.params.c_in_g;
                                        const ic0: usize = gk - pos * t.params.c_in_g;
                                        const take: usize = @min(rem_k, t.params.c_in_g - ic0);
                                        const kh: usize = pos / t.params.k_w;
                                        const kw: usize = pos - kh * t.params.k_w;

                                        const ih: isize = oh0 + @as(isize, @intCast(kh * t.s.dilation_h));
                                        const iw: isize = ow0 + @as(isize, @intCast(kw * t.s.dilation_w));

                                        const dst: []f32 = row_pa[a_off .. a_off + take];
                                        if (all_valid) {
                                            const ih_u: usize = @intCast(ih);
                                            const iw_u: usize = @intCast(iw);
                                            const x_idx: usize = base_batch + ((ih_u * t.params.w_in + iw_u) * t.params.c_in) + ic0;
                                            @memcpy(dst, t.x[x_idx .. x_idx + take]);
                                        } else {
                                            if (use_reflect) {
                                                const ih_u: usize = reflectIndex1D(ih, t.params.h_in);
                                                const iw_u: usize = reflectIndex1D(iw, t.params.w_in);
                                                const x_idx: usize = base_batch + ((ih_u * t.params.w_in + iw_u) * t.params.c_in) + ic0;
                                                @memcpy(dst, t.x[x_idx .. x_idx + take]);
                                            } else if (ih >= 0 and iw >= 0 and ih < h_in_i and iw < w_in_i) {
                                                const ih_u: usize = @intCast(ih);
                                                const iw_u: usize = @intCast(iw);
                                                const x_idx: usize = base_batch + ((ih_u * t.params.w_in + iw_u) * t.params.c_in) + ic0;
                                                @memcpy(dst, t.x[x_idx .. x_idx + take]);
                                            } else {
                                                @memset(dst, 0.0);
                                            }
                                        }

                                        gk += take;
                                        a_off += take;
                                        rem_k -= take;
                                    }
                                }
                            }

                            const beta_eff: f32 = if (bi_full == 0) 0.0 else 1.0;

                            var ti1: usize = 0;
                            while (ti1 < t.tiles_per_group) : (ti1 += 1) {
                                const tile: TileInfo = t.tile_infos[tile_base + ti1];
                                const packed_w_g: PackedWeightEntry = tile.packed_w;
                                const block_elems: usize = packed_w_g.block_elems;

                                const dst_base: usize = row0 * t.params.c_out + tile.oc_start;
                                const c_len: usize = (m_rows - 1) * t.params.c_out + tile.oc_count;
                                const c_slice: []f32 = t.out[dst_base .. dst_base + c_len];

                                const pb0: usize = bi_full * block_elems;
                                const packed_b_view: []align(32) const f32 = @alignCast(packed_w_g.blocks[pb0 .. pb0 + block_elems]);
                                const pp: MatMulParams = .{
                                    .m = m_rows,
                                    .n = tile.oc_count,
                                    .k = k_sub,
                                    .ldc = t.params.c_out,
                                    .alpha = 1.0,
                                    .beta = beta_eff,
                                };
                                try t.matmul.matmul_packed_ab(
                                    packed_a_buf,
                                    packed_b_view,
                                    pp,
                                    std.mem.sliceAsBytes(c_slice),
                                );
                            }
                        }
                    }

                    if (k_tail != 0) {
                        const kk0: usize = full_blocks * t.kc;
                        const k_sub: usize = k_tail;

                        const MR: usize = t.matmul.tuning.mr;
                        const KC: usize = t.kc;
                        const panel_count: usize = (m_rows + MR - 1) / MR;

                        var panel: usize = 0;
                        while (panel < panel_count) : (panel += 1) {
                            const base_pa: usize = panel * (MR * KC);
                            var r: usize = 0;
                            while (r < MR) : (r += 1) {
                                const mr: usize = panel * MR + r;
                                if (mr >= m_rows) break;

                                const b: usize = b_idx[mr];
                                const oh: usize = oh_idx[mr];
                                const ow: usize = ow_idx[mr];

                                const base_batch: usize = (b * t.params.h_in * t.params.w_in * t.params.c_in) + ic_base;
                                const oh0: isize = @as(isize, @intCast(oh)) * @as(isize, @intCast(t.s.stride_h)) - @as(isize, @intCast(t.s.pad_top));
                                const ow0: isize = @as(isize, @intCast(ow)) * @as(isize, @intCast(t.s.stride_w)) - @as(isize, @intCast(t.s.pad_left));

                                const all_valid: bool = (oh0 >= 0 and ow0 >= 0 and (oh0 + max_h) < h_in_i and (ow0 + max_w) < w_in_i);

                                const row_pa: []f32 = packed_a_buf[base_pa + r * KC .. base_pa + r * KC + KC];

                                var rem_k: usize = k_sub;
                                var gk: usize = kk0;
                                var a_off: usize = 0;
                                while (rem_k != 0) {
                                    const pos: usize = gk / t.params.c_in_g;
                                    const ic0: usize = gk - pos * t.params.c_in_g;
                                    const take: usize = @min(rem_k, t.params.c_in_g - ic0);
                                    const kh: usize = pos / t.params.k_w;
                                    const kw: usize = pos - kh * t.params.k_w;

                                    const ih: isize = oh0 + @as(isize, @intCast(kh * t.s.dilation_h));
                                    const iw: isize = ow0 + @as(isize, @intCast(kw * t.s.dilation_w));

                                    const dst: []f32 = row_pa[a_off .. a_off + take];
                                    if (all_valid) {
                                        const ih_u: usize = @intCast(ih);
                                        const iw_u: usize = @intCast(iw);
                                        const x_idx: usize = base_batch + ((ih_u * t.params.w_in + iw_u) * t.params.c_in) + ic0;
                                        @memcpy(dst, t.x[x_idx .. x_idx + take]);
                                    } else {
                                        if (use_reflect) {
                                            const ih_u: usize = reflectIndex1D(ih, t.params.h_in);
                                            const iw_u: usize = reflectIndex1D(iw, t.params.w_in);
                                            const x_idx: usize = base_batch + ((ih_u * t.params.w_in + iw_u) * t.params.c_in) + ic0;
                                            @memcpy(dst, t.x[x_idx .. x_idx + take]);
                                        } else if (ih >= 0 and iw >= 0 and ih < h_in_i and iw < w_in_i) {
                                            const ih_u: usize = @intCast(ih);
                                            const iw_u: usize = @intCast(iw);
                                            const x_idx: usize = base_batch + ((ih_u * t.params.w_in + iw_u) * t.params.c_in) + ic0;
                                            @memcpy(dst, t.x[x_idx .. x_idx + take]);
                                        } else {
                                            @memset(dst, 0.0);
                                        }
                                    }

                                    gk += take;
                                    a_off += take;
                                    rem_k -= take;
                                }
                            }
                        }

                        const beta_eff: f32 = if (full_blocks == 0) 0.0 else 1.0;
                        var ti1: usize = 0;
                        while (ti1 < t.tiles_per_group) : (ti1 += 1) {
                            const tile: TileInfo = t.tile_infos[tile_base + ti1];
                            const packed_w_g: PackedWeightEntry = tile.packed_w;
                            const block_elems: usize = packed_w_g.block_elems;

                            const dst_base: usize = row0 * t.params.c_out + tile.oc_start;
                            const c_len: usize = (m_rows - 1) * t.params.c_out + tile.oc_count;
                            const c_slice: []f32 = t.out[dst_base .. dst_base + c_len];

                            const pb0: usize = full_blocks * block_elems;
                            const packed_b_view: []align(32) const f32 = @alignCast(packed_w_g.blocks[pb0 .. pb0 + block_elems]);
                            const pp_tail: MatMulParams = .{
                                .m = m_rows,
                                .n = tile.oc_count,
                                .k = k_sub,
                                .ldc = t.params.c_out,
                                .alpha = 1.0,
                                .beta = beta_eff,
                            };
                            try t.matmul.matmul_packed_ab(
                                packed_a_buf,
                                packed_b_view,
                                pp_tail,
                                std.mem.sliceAsBytes(c_slice),
                            );
                        }
                    }

                    if (bias_present) {
                        var ti1: usize = 0;
                        while (ti1 < t.tiles_per_group) : (ti1 += 1) {
                            const tile: TileInfo = t.tile_infos[tile_base + ti1];
                            const dst_base: usize = row0 * t.params.c_out + tile.oc_start;
                            const c_len: usize = (m_rows - 1) * t.params.c_out + tile.oc_count;
                            const c_slice: []f32 = t.out[dst_base .. dst_base + c_len];
                            const bias_slice: []const f32 = t.bias[tile.oc_start .. tile.oc_start + tile.oc_count];

                            const lanes: usize = 8;
                            const Vec = @Vector(lanes, f32);
                            var oc: usize = 0;
                            while (oc + lanes <= tile.oc_count) : (oc += lanes) {
                                const bias_v: Vec = @as(*align(1) const Vec, @ptrCast(bias_slice.ptr + oc)).*;
                                var mr: usize = 0;
                                while (mr < m_rows) : (mr += 1) {
                                    const row_base: usize = mr * t.params.c_out + oc;
                                    const c_ptr = c_slice.ptr + row_base;
                                    const c_v: Vec = @as(*align(1) const Vec, @ptrCast(c_ptr)).*;
                                    @as(*align(1) Vec, @ptrCast(c_ptr)).* = c_v + bias_v;
                                }
                            }
                            while (oc < tile.oc_count) : (oc += 1) {
                                var mr: usize = 0;
                                while (mr < m_rows) : (mr += 1) {
                                    const row_base: usize = mr * t.params.c_out + oc;
                                    c_slice[row_base] += bias_slice[oc];
                                }
                            }
                        }
                    }
                }

                row0 += m_rows;
            }
        }

        fn runRows(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
            const t: *@This() = @ptrCast(@alignCast(ctx_any));
            const scratch_bytes: []align(32) u8 = t.ctx.matmul_scratch[tid];
            t.runRowsRange(scratch_bytes, start, end) catch return;
        }
    };

    var task: Task = .{
        .ctx = ctx,
        .s = s,
        .matmul = matmul,
        .use_local_scratch = use_local_scratch,
        .params = .{
            .h_in = h_in,
            .w_in = w_in,
            .h_out = h_out,
            .w_out = w_out,
            .c_in = c_in,
            .c_out = c_out,
            .k_h = k_h,
            .k_w = k_w,
            .c_in_g = c_in_g,
            .k_dim_g = k_dim_g,
        },
        .x = x_packed,
        .out = out_packed,
        .tile_infos = tile_infos,
        .bias = bias_packed,
        .kc = kc,
        .m_cap = m_cap,
        .oc_tile_max = oc_tile_max,
        .groups = groups,
        .tiles_per_group = tiles_per_group,
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

pub fn execConv2DTiled(ctx: *ConvExecCtx, s: StepConv2DTiled, store: tensor_store.TensorStore) ExecuteProgramError!void {
    const out_meta: tensor_store.TensorMeta = try store.meta(s.out);
    const x_meta: tensor_store.TensorMeta = try store.meta(s.x);
    const w_meta: tensor_store.TensorMeta = try store.meta(s.w);

    std.debug.assert(out_meta.dtype == .f32 and x_meta.dtype == .f32 and w_meta.dtype == .f32);
    std.debug.assert(out_meta.rank == x_meta.rank and out_meta.rank >= 3);
    std.debug.assert(w_meta.rank == 4);
    std.debug.assert(s.groups > 0 and s.stride_h > 0 and s.stride_w > 0 and s.dilation_h > 0 and s.dilation_w > 0);

    const rank: usize = @as(usize, out_meta.rank);
    const c_out: usize = out_meta.shape[rank - 1];
    const c_in: usize = x_meta.shape[rank - 1];
    const c_in_g: usize = w_meta.shape[2];

    std.debug.assert(c_in % s.groups == 0);
    std.debug.assert(c_out % s.groups == 0);
    std.debug.assert(c_in_g * s.groups == c_in);

    _ = try execConv2DImplicitGemm(ctx, s, out_meta, x_meta, w_meta, store);
    return;
}
