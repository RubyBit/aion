// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const conv_utils = @import("conv_utils.zig");



/// Panels held in one group on the indirect path. A group shares a sweep of the
/// packed weights; rows-per-panel falls to the output width when that is under
/// the micro-kernel's 32, so a narrow layer needs more panels to fill a group.
const max_group_panels: usize = 128;

/// Rows of output one group accumulates before moving on.
///
/// The group's slice of C is read and written once per K block, so it has to
/// survive in L2. Within that, more rows is strictly better: the packed weights
/// are swept once per group, and on the early wide layers that sweep is the
/// GEMM's largest memory cost — going from 288 rows to a cache-sized group was
/// worth 1532 -> 1660 GFLOP/s on VGG-19.
fn indirectGroupRows(c_out: usize, mr: usize, l2_bytes: usize) usize {
    const default_l2: usize = 4 << 20;
    // A quarter of L2, so the weight panels streaming past keep their room.
    const budget: usize = (if (l2_bytes != 0) l2_bytes else default_l2) / 4;
    const rows: usize = budget / (c_out * @sizeOf(f32));
    return std.math.clamp(rows, mr, max_group_panels * mr);
}

/// Whether conv can read activations straight out of the channel-major image
/// instead of gathering them.
///
/// Unit stride is what keeps 32 consecutive output positions consecutive in the
/// image, zero padding is what lets the pad offsets cancel out of the index, and
/// the panels stop at row ends — so a width that rounds far past a whole
/// micro-kernel block would pay for more rows than it uses.
fn indirectApplies(matmul: matmul_registry.F32Kernels, s: StepConv2D, w_out: usize) bool {
    if (matmul.matmul_indirect == null) return false;
    if (s.stride_h != 1 or s.stride_w != 1 or s.pad_mode != .zero) return false;
    const panel_rows: usize = conv_utils.roundUpToMultiple(w_out, matmul.tuning.mr);
    return panel_rows * 4 <= w_out * 5;
}

/// Rewrites one batch item's rows channel-major; split across workers because it
/// is otherwise the serial head of every conv on the indirect path.
const ImageBuild = struct {
    dst: []f32,
    x: []const f32,
    geo: conv_utils.ImageGeometry,
    plane: usize,

    fn run(ctx_any: *anyopaque, first: usize, last: usize, tid: usize) ExecuteProgramError!void {
        _ = tid;
        const t: *@This() = @ptrCast(@alignCast(ctx_any));
        const rows: usize = t.geo.h_in;
        var i: usize = first;
        while (i < last) {
            const bi: usize = i / rows;
            const lo: usize = i - bi * rows;
            const hi: usize = @min(rows, lo + (last - i));
            conv_utils.buildChannelMajorRows(
                t.dst[bi * t.geo.c_in * t.plane ..],
                t.x[bi * rows * t.geo.w_in * t.geo.c_in ..],
                t.geo,
                lo,
                hi,
            );
            i += hi - lo;
        }
    }
};
const conv2d_kernels = @import("../kernels/conv2d.zig");
const matmul_registry = @import("../registry/matmul_registry.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");

const BackendError = conv_utils.BackendError;
const MatMulParams = conv_utils.MatMulParams;
const ExecuteProgramError = conv_utils.ExecuteProgramError;

const StepConv2D = executable.StepConv2D;

const elemCountFromShape = conv_utils.elemCountFromShape;

pub const ConvExecCtx = conv_utils.ConvExecCtx;
const PackedWeightKey = conv_utils.PackedWeightKey;
const PackedWeightEntry = conv_utils.PackedWeightEntry;

const getOrCreatePackedWeights = conv_utils.getOrCreatePackedWeights;
const scratchForTid = conv_utils.scratchForTid;
const fillWeightBlock = conv_utils.fillWeightBlock;
const addBiasRowsF32 = conv_utils.addBiasRowsF32;
const findPackedWeights = conv_utils.findPackedWeights;

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

/// One im2col run is `c_in` wide — often a single channel, where the call into
/// `memcpy` costs more than the move itself. Inline the short runs.
inline fn packRun(dst: []f32, src: anytype) void {
    if (dst.len > 16) return @memcpy(dst, src[0..dst.len]);
    for (dst, src[0..dst.len]) |*d, s| d.* = s;
}

inline fn packZero(dst: []f32) void {
    if (dst.len > 16) return @memset(dst, 0.0);
    for (dst) |*d| d.* = 0.0;
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

/// Depthwise convolution (`groups == C_in == C_out`, one input channel per filter),
/// straight over the flat tensors. False when the step is not depthwise or its
/// kernel has more taps than the direct kernel carries; the grouped GEMM takes those.
fn execDepthwise(ctx: *ConvExecCtx, s: StepConv2D, out_meta: tensor_store.TensorMeta, x_meta: tensor_store.TensorMeta, w_meta: tensor_store.TensorMeta, store: tensor_store.TensorStore) ExecuteProgramError!bool {
    const rank: usize = out_meta.rank;
    const c: usize = out_meta.shape[rank - 1];
    if (s.groups != c or x_meta.shape[rank - 1] != c or w_meta.shape[2] != 1 or w_meta.shape[3] != c) return false;
    if (w_meta.shape[0] * w_meta.shape[1] > conv2d_kernels.MAX_TAPS) return false;
    var batch: usize = 1;
    for (out_meta.shape[0 .. rank - 3]) |d| batch *= d;

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
            .stride_h = s.stride_h,
            .stride_w = s.stride_w,
            .dilation_h = s.dilation_h,
            .dilation_w = s.dilation_w,
            .pad_top = s.pad_top,
            .pad_left = s.pad_left,
            .reflect = s.pad_mode == .reflect,
        },
        .batch = batch,
        .h_in = x_meta.shape[rank - 3],
        .w_in = x_meta.shape[rank - 2],
        .h_out = out_meta.shape[rank - 3],
        .w_out = out_meta.shape[rank - 2],
        .c = c,
        .k_h = w_meta.shape[0],
        .k_w = w_meta.shape[1],
        .x = bytesAsF32Const(x.bytes),
        .w = bytesAsF32Const(w.bytes),
        .bias = if (bias) |b| bytesAsF32Const(b.bytes) else &.{},
        .out = bytesAsF32Mut(out.bytes),
    };
    conv_utils.runDepthwise(ctx, &task);
    return true;
}

fn execConv2DImplicitGemm(
    ctx: *ConvExecCtx,
    s: StepConv2D,
    out_meta: tensor_store.TensorMeta,
    x_meta: tensor_store.TensorMeta,
    w_meta: tensor_store.TensorMeta,
    store: tensor_store.TensorStore,
) ExecuteProgramError!bool {
    if (try execDepthwise(ctx, s, out_meta, x_meta, w_meta, store)) return true;

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

    const alloc: std.mem.Allocator = std.heap.page_allocator;

    const x_ref = try store.acquireConst(s.x);
    defer store.releaseConst(x_ref.token);
    const out_ref = try store.acquireMut(s.out);
    defer store.releaseMut(out_ref.token);
    const w_ref = try store.acquireConst(s.w);
    defer store.releaseConst(w_ref.token);
    const bias_ref = if (s.bias) |b_id| try store.acquireConst(b_id) else null;
    defer if (bias_ref) |b| store.releaseConst(b.token);

    const x_packed: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, x_ref.bytes));
    const out_packed: []f32 = @alignCast(std.mem.bytesAsSlice(f32, out_ref.bytes));
    const w_packed: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, w_ref.bytes));
    const bias_packed: []const f32 = if (bias_ref) |b| @alignCast(std.mem.bytesAsSlice(f32, b.bytes)) else &.{};
    if (bias_packed.len != 0 and bias_packed.len != c_out) return BackendError.InvalidArgument;

    if (c_out_g == 0 or ctx.matmul_f32.tuning.nc == 0) return BackendError.InvalidArgument;

    const matmul: matmul_registry.F32Kernels = ctx.matmul_f32;
    const use_local_scratch: bool = false;

    const kc: usize = matmul.tuning.kc;
    const m_cap: usize = conv_utils.roundUpToMultiple(matmul.tuning.mc, matmul.tuning.mr);
    const oc_tile_max: usize = @min(c_out_g, matmul.tuning.nc);


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

            const key_g: PackedWeightKey = .{
                    .w_id = s.w,
                .oc_start = oc_start,
                .k_dim = k_dim_g,
                .c_out = oc_count,
                .groups = s.groups,
                .kc = kc,
                .nc = ctx.matmul_f32.tuning.nc,
            };

            // A weight never changes, so gathering its block is worth doing only
            // when the pack cache has nothing for it.
            const packed_w_g: PackedWeightEntry = findPackedWeights(ctx.cache, key_g) orelse blk: {
                try fillWeightBlock(w_block[0 .. k_dim_g * oc_count], w_packed, k_dim_g, c_out, oc_start, oc_count);
                break :blk try getOrCreatePackedWeights(ctx.cache, matmul, key_g, w_block[0 .. k_dim_g * oc_count]);
            };
            tile_infos[ti] = .{ .oc_start = oc_start, .oc_count = oc_count, .ic_base = ic_base, .packed_w = packed_w_g };
            ti += 1;
        }
    }
    std.debug.assert(ti == total_tiles);

    const Task = struct {
        ctx: *ConvExecCtx,
        s: StepConv2D,
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

        /// Set when the activations were rewritten channel-major up front, which
        /// lets the GEMM read them through a pointer table instead of a gather.
        image: ?conv_utils.ChannelMajorImage,

        /// Row panels straight out of the channel-major image, in groups.
        ///
        /// Panels stop at output-row ends: only within a row are 32 consecutive
        /// output positions 32 consecutive floats for every reduction index. A
        /// group of them shares one sweep of the packed weights — a panel at a
        /// time re-reads the whole weight block every 32 rows, which on the deep
        /// layers costs more than the gather this path removes.
        fn runPanelsIndirect(t: *const @This(), img: conv_utils.ChannelMajorImage, table_pool: [][*]const f32, start: usize, end: usize) ExecuteProgramError!void {
            @setRuntimeSafety(false);
            const indirect = t.matmul.matmul_indirect.?;
            const MR: usize = t.matmul.tuning.mr;
            const full_blocks: usize = t.params.k_dim_g / t.kc;
            const k_tail: usize = t.params.k_dim_g - full_blocks * t.kc;
            const k_blocks: usize = full_blocks + @intFromBool(k_tail != 0);
            const hw_out: usize = t.params.h_out * t.params.w_out;
            const bias_present: bool = (t.bias.len != 0);
            const max_panels: usize = @min(max_group_panels, table_pool.len / t.kc);
            const group_rows: usize = indirectGroupRows(t.params.c_out, MR, t.ctx.l2_bytes);

            var panel_row: [max_group_panels]usize = undefined;
            var panel_len: [max_group_panels]usize = undefined;

            var row: usize = start;
            while (row < end) {
                var panels: usize = 0;
                var rows_acc: usize = 0;
                var r: usize = row;
                while (r < end and panels < max_panels and rows_acc < group_rows) {
                    const ow_r: usize = (r % hw_out) % t.params.w_out;
                    const mr: usize = @min(@min(MR, t.params.w_out - ow_r), end - r);
                    panel_row[panels] = r;
                    panel_len[panels] = mr;
                    panels += 1;
                    rows_acc += mr;
                    r += mr;
                }

                var gg: usize = 0;
                while (gg < t.groups) : (gg += 1) {
                    const ic_base: usize = gg * t.params.c_in_g;
                    const tile_base: usize = gg * t.tiles_per_group;

                    var bi: usize = 0;
                    while (bi < k_blocks) : (bi += 1) {
                        const k_sub: usize = if (bi < full_blocks) t.kc else k_tail;
                        for (0..panels) |pi| {
                            const rr: usize = panel_row[pi];
                            const b: usize = rr / hw_out;
                            const rem: usize = rr - b * hw_out;
                            const oh: usize = rem / t.params.w_out;
                            conv_utils.buildIndirectTable(
                                table_pool[pi * k_sub ..][0..k_sub],
                                img,
                                b * t.params.c_in + ic_base,
                                t.params.c_in_g,
                                t.params.k_w,
                                t.s.dilation_h,
                                t.s.dilation_w,
                                bi * t.kc,
                                oh,
                                rem - oh * t.params.w_out,
                            );
                        }
                        const beta: f32 = if (bi == 0) 0.0 else 1.0;

                        var ti1: usize = 0;
                        while (ti1 < t.tiles_per_group) : (ti1 += 1) {
                            const tile: TileInfo = t.tile_infos[tile_base + ti1];
                            const block_elems: usize = tile.packed_w.block_elems;
                            const pb0: usize = bi * block_elems;
                            const packed_b_view: []align(32) const f32 = @alignCast(tile.packed_w.blocks[pb0 .. pb0 + block_elems]);

                            for (0..panels) |pi| {
                                const rr: usize = panel_row[pi];
                                const mr: usize = panel_len[pi];
                                const dst_base: usize = rr * t.params.c_out + tile.oc_start;
                                const c_len: usize = (mr - 1) * t.params.c_out + tile.oc_count;
                                try indirect(
                                    table_pool[pi * k_sub ..][0..k_sub],
                                    packed_b_view,
                                    .{ .m = mr, .n = tile.oc_count, .k = k_sub, .ldc = t.params.c_out, .alpha = 1.0, .beta = beta },
                                    std.mem.sliceAsBytes(t.out[dst_base .. dst_base + c_len]),
                                );
                            }
                        }
                    }

                    if (bias_present) {
                        var ti1: usize = 0;
                        while (ti1 < t.tiles_per_group) : (ti1 += 1) {
                            const tile: TileInfo = t.tile_infos[tile_base + ti1];
                            const bias_slice: []const f32 = t.bias[tile.oc_start .. tile.oc_start + tile.oc_count];
                            for (0..panels) |pi| {
                                const rr: usize = panel_row[pi];
                                const mr: usize = panel_len[pi];
                                const dst_base: usize = rr * t.params.c_out + tile.oc_start;
                                const c_len: usize = (mr - 1) * t.params.c_out + tile.oc_count;
                                addBiasRowsF32(t.out[dst_base .. dst_base + c_len], 0, mr, t.params.c_out, tile.oc_count, bias_slice, t.matmul.tuning.lanes);
                            }
                        }
                    }
                }

                row = r;
            }
        }

        fn runRowsRange(t: *const @This(), scratch: []align(32) u8, start: usize, end: usize) ExecuteProgramError!void {
            @setRuntimeSafety(false);
            const m_cap_local: usize = t.m_cap;
            const full_blocks: usize = t.params.k_dim_g / t.kc;
            const k_tail: usize = t.params.k_dim_g - full_blocks * t.kc;

            // The free has to outlive the block that allocates, or every use
            // below reads freed memory.
            var local_scratch: []align(32) u8 = &[_]u8{};
            defer if (local_scratch.len != 0) t.alloc.free(local_scratch);
            if (t.use_local_scratch) {
                local_scratch = try t.alloc.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), t.matmul.scratch_bytes);
            }
            const scratch_use: []align(32) u8 = if (t.use_local_scratch) local_scratch else scratch;

            // Reuse the matmul scratch's packed-A region ("pa") to avoid allocating a packed-A buffer.
            // Layout matches matmul.Kernel.splitScratch(): pb = KC*NC, pa = MC*KC.
            const pb_elems: usize = t.kc * t.matmul.tuning.nc;
            const pa_elems: usize = conv_utils.packedAElems(t.matmul.tuning.mr, t.m_cap, t.kc);
            const scratch_f32: []align(32) f32 = @alignCast(std.mem.bytesAsSlice(f32, scratch_use));
            std.debug.assert(scratch_f32.len >= pb_elems + pa_elems);
            const packed_a_buf: []align(32) f32 = @alignCast(scratch_f32[pb_elems .. pb_elems + pa_elems]);

            // The packed-A region is dead when activations are read through the
            // image, so the pointer tables live there rather than in a buffer of
            // their own.
            if (t.image) |img| {
                const pool: [][*]const f32 = @alignCast(std.mem.bytesAsSlice([*]const f32, std.mem.sliceAsBytes(packed_a_buf)));
                if (pool.len >= t.kc) return t.runPanelsIndirect(img, pool, start, end);
            }

            const h_in_i: isize = @as(isize, @intCast(t.params.h_in));
            const w_in_i: isize = @as(isize, @intCast(t.params.w_in));
            const max_h: isize = @as(isize, @intCast((t.params.k_h - 1) * t.s.dilation_h));
            const max_w: isize = @as(isize, @intCast((t.params.k_w - 1) * t.s.dilation_w));
            const use_reflect: bool = (t.s.pad_mode == .reflect);

            // Packs one GEMM row of the im2col operand, shared by the full-KC
            // blocks and the K tail. Reduction indices run (kh, kw, ic).
            const pack = struct {
                t: @TypeOf(t),
                use_reflect: bool,
                h_in_i: isize,
                w_in_i: isize,
                wide: bool,

                /// Zero padding splits a kernel row into at most three runs: leading
                /// pad, one contiguous copy, trailing pad. Interior rows are one run.
                fn rowWide(p: @This(), row_pa: []f32, base_batch: usize, k_sub: usize, oh0: isize, ow0: isize, kh_in: usize, kw_in: usize, ic_in: usize) void {
                    const task = p.t;
                    const taps: usize = task.params.k_w;
                    const chans: usize = task.params.c_in_g;
                    const lo: usize = if (ow0 < 0) @min(taps, @as(usize, @intCast(-ow0))) else 0;
                    const hi: usize = if (ow0 >= p.w_in_i) 0 else @min(taps, @as(usize, @intCast(p.w_in_i - ow0)));
                    var kh: usize = kh_in;
                    var kw: usize = kw_in;
                    var ic0: usize = ic_in;
                    var rem_k: usize = k_sub;
                    var a_off: usize = 0;
                    while (rem_k != 0) {
                        const ih: isize = oh0 + @as(isize, @intCast(kh * task.s.dilation_h));
                        const in_row: bool = ih >= 0 and ih < p.h_in_i;
                        const copy: bool = in_row and kw >= lo and kw < hi;
                        const bound: usize = if (!in_row) taps else if (kw < lo) lo else if (kw < hi) hi else taps;
                        const take: usize = @min(rem_k, (bound - kw) * chans - ic0);
                        const dst: []f32 = row_pa[a_off .. a_off + take];
                        if (copy) {
                            const iw: usize = @intCast(ow0 + @as(isize, @intCast(kw)));
                            const x_idx: usize = base_batch + ((@as(usize, @intCast(ih)) * task.params.w_in + iw) * task.params.c_in) + ic0;
                            packRun(dst, task.x[x_idx .. x_idx + take]);
                        } else {
                            packZero(dst);
                        }
                        a_off += take;
                        rem_k -= take;
                        ic0 += take;
                        while (ic0 >= chans) {
                            ic0 -= chans;
                            kw += 1;
                            if (kw == taps) {
                                kw = 0;
                                kh += 1;
                            }
                        }
                    }
                }

                fn row(p: @This(), row_pa: []f32, base_batch: usize, kk0: usize, k_sub: usize, oh0: isize, ow0: isize, all_valid: bool) void {
                    const task = p.t;
                    const c_g: usize = task.params.c_in_g;
                    var rem_k: usize = k_sub;
                    var a_off: usize = 0;
                    const pos: usize = kk0 / c_g;
                    var ic0: usize = kk0 - pos * c_g;
                    var kh: usize = pos / task.params.k_w;
                    var kw: usize = pos - kh * task.params.k_w;
                    if (p.wide) return p.rowWide(row_pa, base_batch, k_sub, oh0, ow0, kh, kw, ic0);
                    while (rem_k != 0) {
                        const span: usize = c_g - ic0;
                        const take: usize = @min(rem_k, span);
                        const ih: isize = oh0 + @as(isize, @intCast(kh * task.s.dilation_h));
                        const iw: isize = ow0 + @as(isize, @intCast(kw * task.s.dilation_w));
                        const dst: []f32 = row_pa[a_off .. a_off + take];

                        if (all_valid or p.use_reflect) {
                            const ih_u: usize = if (all_valid) @intCast(ih) else reflectIndex1D(ih, task.params.h_in);
                            const iw_u: usize = if (all_valid) @intCast(iw) else reflectIndex1D(iw, task.params.w_in);
                            const x_idx: usize = base_batch + ((ih_u * task.params.w_in + iw_u) * task.params.c_in) + ic0;
                            packRun(dst, task.x[x_idx .. x_idx + take]);
                        } else if (ih >= 0 and iw >= 0 and ih < p.h_in_i and iw < p.w_in_i) {
                            const x_idx: usize = base_batch + ((@as(usize, @intCast(ih)) * task.params.w_in + @as(usize, @intCast(iw))) * task.params.c_in) + ic0;
                            packRun(dst, task.x[x_idx .. x_idx + take]);
                        } else {
                            packZero(dst);
                        }

                        a_off += take;
                        rem_k -= take;
                        ic0 += take;
                        while (ic0 >= c_g) {
                            ic0 -= c_g;
                            kw += 1;
                            if (kw == task.params.k_w) {
                                kw = 0;
                                kh += 1;
                            }
                        }
                    }
                }
            }{
                .t = t,
                .use_reflect = use_reflect,
                .h_in_i = h_in_i,
                .w_in_i = w_in_i,
                .wide = t.s.dilation_w == 1 and t.params.c_in_g == t.params.c_in and t.s.pad_mode != .reflect,
            };

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

                                    const kmaj = t.matmul.tuning.a_layout == .k_major;
                                    const row_pa: []f32 = if (kmaj)
                                        conv_utils.kMajorStage(KC)[(r % conv_utils.KMAJOR_GROUP) * KC ..][0..KC]
                                    else
                                        packed_a_buf[base_pa + r * KC .. base_pa + r * KC + KC];

                                    pack.row(row_pa, base_batch, kk0, k_sub, oh0, ow0, all_valid);
                                    if (kmaj and ((r % conv_utils.KMAJOR_GROUP) == conv_utils.KMAJOR_GROUP - 1 or r + 1 == MR or mr + 1 == m_rows)) {
                                        const g0: usize = r - (r % conv_utils.KMAJOR_GROUP);
                                        conv_utils.cornerTurnKMajor(packed_a_buf[base_pa..], conv_utils.kMajorStage(KC), MR, g0, r - g0 + 1, k_sub, KC);
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

                                const kmaj = t.matmul.tuning.a_layout == .k_major;
                                const row_pa: []f32 = if (kmaj)
                                    conv_utils.kMajorStage(KC)[(r % conv_utils.KMAJOR_GROUP) * KC ..][0..KC]
                                else
                                    packed_a_buf[base_pa + r * KC .. base_pa + r * KC + KC];

                                pack.row(row_pa, base_batch, kk0, k_sub, oh0, ow0, all_valid);
                                if (kmaj and ((r % conv_utils.KMAJOR_GROUP) == conv_utils.KMAJOR_GROUP - 1 or r + 1 == MR or mr + 1 == m_rows)) {
                                    const g0: usize = r - (r % conv_utils.KMAJOR_GROUP);
                                    conv_utils.cornerTurnKMajor(packed_a_buf[base_pa..], conv_utils.kMajorStage(KC), MR, g0, r - g0 + 1, k_sub, KC);
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

                            addBiasRowsF32(c_slice, 0, m_rows, t.params.c_out, tile.oc_count, bias_slice, t.matmul.tuning.lanes);
                        }
                    }
                }

                row0 += m_rows;
            }
        }

        fn runRows(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) ExecuteProgramError!void {
            const t: *@This() = @ptrCast(@alignCast(ctx_any));
            const scratch_bytes: []align(32) u8 = t.ctx.matmul_scratch[tid];
            try t.runRowsRange(scratch_bytes, start, end);
        }
    };

    // Rewriting the activations channel-major up front costs one pass over x and
    // removes the im2col gather outright: the GEMM then reads 32 output positions
    // straight out of the image. Only unit stride keeps those 32 consecutive, and
    // only zero padding lets the pad offsets cancel out of the index.
    //
    // Panels also stop at row ends, so each output row is rounded up to a whole
    // micro-kernel block: a width just past a multiple pays for rows it discards.
    // Once that passes a quarter of the work the gather is the cheaper of the two.
    var image: ?conv_utils.ChannelMajorImage = null;
    if (indirectApplies(matmul, s, w_out)) {
        const hp: usize = h_out + (k_h - 1) * s.dilation_h;
        const wp: usize = w_out + (k_w - 1) * s.dilation_w;
        if (s.pad_top + h_in <= hp and s.pad_left + w_in <= wp) {
            const plane: usize = hp * wp;
            const buf = try ctx.cache.scratch(.image, conv_utils.ChannelMajorImage.elems(batch * c_in, hp, wp));
            @memset(buf[batch * c_in * plane ..], 0);
            const geo: conv_utils.ImageGeometry = .{
                .h_in = h_in,
                .w_in = w_in,
                .c_in = c_in,
                .pad_top = s.pad_top,
                .pad_left = s.pad_left,
                .hp = hp,
                .wp = wp,
            };
            @memset(buf, 0);
            var build: ImageBuild = .{ .dst = buf, .x = x_packed, .geo = geo, .plane = plane };
            const img_rows: usize = batch * h_in;
            var built_parallel = false;
            if (ctx.pool) |pl| {
                if (ctx.thread_count > 1 and img_rows >= 2) {
                    const grain: usize = @max(@as(usize, 1), img_rows / (ctx.thread_count * 4));
                    try pl.parallelForFallible(ExecuteProgramError, @ptrCast(&build), img_rows, grain, ImageBuild.run);
                    built_parallel = true;
                }
            }
            if (!built_parallel) try ImageBuild.run(@ptrCast(&build), 0, img_rows, 0);
            image = .{ .data = buf, .wp = wp, .plane_elems = plane };
        }
    }

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
        .image = image,
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

pub fn execConv2D(ctx: *ConvExecCtx, s: StepConv2D, store: tensor_store.TensorStore) ExecuteProgramError!void {
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
