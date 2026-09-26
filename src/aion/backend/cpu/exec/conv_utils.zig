// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const matmul_registry = @import("../registry/matmul_registry.zig");
const conv2d_registry = @import("../registry/conv2d_registry.zig");
const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const exec_utils = @import("utils.zig");

pub const BackendError = types.BackendError;
pub const MatMulParams = types.MatMulParams;
pub const ExecuteProgramError = backend_mod.ExecuteProgramError;

pub const MAX_RANK: usize = 8;

pub const elemCountFromShape = exec_utils.elemCountFromShape;

pub const ConvExecCtx = struct {
    allocator: std.mem.Allocator,
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    matmul_f32: matmul_registry.F32Kernels,

    /// Shared L2 per core cluster, 0 when unknown. Conv sizes its row groups
    /// against this: the group's slice of C has to survive the K loop.
    l2_bytes: usize,

    // Depthwise direct conv kernels (selected by CPU registry).
    depthwise_conv2d: conv2d_registry.Kernels,

    // Per-thread scratch for matmul packing. May be empty in single-thread mode.
    matmul_scratch: [][]align(32) u8,

    /// Packed weights for this backend, reused across runs of the same model.
    cache: *ConvCache,
};

pub const PackedWeightKey = struct {
    w_id: tensor_store.TensorId,
    oc_start: usize,
    k_dim: usize,
    c_out: usize,
    groups: usize,
    kc: usize,
    nc: usize,
};

pub const PackedWeightEntry = struct {
    key: PackedWeightKey,
    // Concatenated packed-B panels, each of length kc*nc f32.
    blocks: []align(32) f32,
    block_count: usize,
    block_elems: usize,
};

fn addBiasRowsF32Vector(
    comptime lanes: usize,
    out: []align(1) f32,
    row_start: usize,
    rows: usize,
    row_stride: usize,
    channel_count: usize,
    bias: []align(1) const f32,
) void {
    const Vec = @Vector(lanes, f32);

    // Rows on the outside: an output row is contiguous in memory, so this walks
    // it straight through. Channel-outer instead strides by `row_stride` between
    // consecutive touches, which is one cache line per element and measured 5.5 ms
    // of VGG-19 at one thread against ~2 ms here.
    var mr: usize = 0;
    while (mr < rows) : (mr += 1) {
        const row_base: usize = (row_start + mr) * row_stride;
        var oc: usize = 0;
        while (oc + lanes <= channel_count) : (oc += lanes) {
            const bias_v: Vec = @as(*align(1) const Vec, @ptrCast(bias.ptr + oc)).*;
            const c_ptr = out.ptr + row_base + oc;
            const c_v: Vec = @as(*align(1) const Vec, @ptrCast(c_ptr)).*;
            @as(*align(1) Vec, @ptrCast(c_ptr)).* = c_v + bias_v;
        }
        while (oc < channel_count) : (oc += 1) out[row_base + oc] += bias[oc];
    }
}

pub fn addBiasRowsF32(
    out: []align(1) f32,
    row_start: usize,
    rows: usize,
    row_stride: usize,
    channel_count: usize,
    bias: []align(1) const f32,
    lanes: usize,
) void {
    switch (lanes) {
        16 => addBiasRowsF32Vector(16, out, row_start, rows, row_stride, channel_count, bias),
        8 => addBiasRowsF32Vector(8, out, row_start, rows, row_stride, channel_count, bias),
        4 => addBiasRowsF32Vector(4, out, row_start, rows, row_stride, channel_count, bias),
        else => addBiasRowsF32Vector(1, out, row_start, rows, row_stride, channel_count, bias),
    }
}

/// Packed conv weights, reused across runs of the same model.
///
/// Owned by the execution session, because `w_id` identifies a tensor only
/// inside one store and a session is bound to one store. A cache any wider than
/// that hands the second model built in a process the first one's weights.
/// Conv state that outlives a single run: packed weights, plus the buffers each
/// step stages through.
///
/// The buffers are held here rather than acquired per step because the page
/// allocator hands back fresh pages every time, and the kernel zero-fills them:
/// on VGG-19 that was tens of megabytes of page faults per inference, more than
/// the arithmetic they feed.
pub const ConvCache = struct {
    mutex: std.Io.Mutex = .init,
    map: std.AutoHashMapUnmanaged(PackedWeightKey, PackedWeightEntry) = .empty,
    allocator: std.mem.Allocator,
    buffers: [std.enums.values(Scratch).len][]align(64) f32 = @splat(&.{}),

    /// One reusable buffer per role. A step acquires each at most once, before
    /// any worker starts, so these need no lock of their own.
    pub const Scratch = enum { image };

    pub fn init(allocator: std.mem.Allocator) ConvCache {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *ConvCache) void {
        var it = self.map.valueIterator();
        while (it.next()) |entry| self.allocator.free(entry.blocks);
        self.map.deinit(self.allocator);
        for (self.buffers) |b| self.allocator.free(b);
    }

    /// Grows only; every conv in a model reuses the widest it has seen.
    pub fn scratch(self: *ConvCache, which: Scratch, n: usize) ![]align(64) f32 {
        const slot = &self.buffers[@intFromEnum(which)];
        if (slot.len < n) {
            self.allocator.free(slot.*);
            slot.* = try self.allocator.alignedAlloc(f32, .@"64", n);
        }
        return slot.*[0..n];
    }
};

/// A previously packed weight, without materializing anything to pack.
///
/// Gathering the weight block costs as much as the pack itself, so a caller that
/// asks this first pays neither once the cache is warm.
pub fn findPackedWeights(cache: *ConvCache, key: PackedWeightKey) ?PackedWeightEntry {
    std.Io.Threaded.mutexLock(&cache.mutex);
    defer std.Io.Threaded.mutexUnlock(&cache.mutex);
    return cache.map.get(key);
}

pub fn getOrCreatePackedWeights(
    cache: *ConvCache,
    matmul_f32: matmul_registry.F32Kernels,
    key: PackedWeightKey,
    w_matrix: ?[]align(1) const f32,
) ExecuteProgramError!PackedWeightEntry {
    std.Io.Threaded.mutexLock(&cache.mutex);
    defer std.Io.Threaded.mutexUnlock(&cache.mutex);

    if (cache.map.get(key)) |entry| return entry;

    const wv: []align(1) const f32 = w_matrix orelse return BackendError.InvalidArgument;
    if (wv.len < key.k_dim * key.c_out) return BackendError.InvalidArgument;

    const k_blocks: usize = (key.k_dim + key.kc - 1) / key.kc;
    // The packed-B kernel reads `ceil(c_out/NR)` panels of `kc*NR` each (panel stride
    // is `kc*NR`, independent of NC). Sizing to `kc*nc` over-allocated massively for
    // small `c_out` — e.g. a per-output-channel conv weight tile (c_out=1) got a full
    // `kc*nc` (~1MB) panel to hold a handful of floats, and these entries are cached
    // for the model's lifetime. Size to the panels actually packed/read.
    const nr: usize = matmul_f32.tuning.nr;
    if (nr == 0) return BackendError.InvalidArgument;
    const n_panels: usize = (key.c_out + nr - 1) / nr;
    const block_elems: usize = n_panels * key.kc * nr;
    const total_elems: usize = k_blocks * block_elems;

    const alloc: std.mem.Allocator = cache.allocator;
    const blocks: []align(32) f32 = try alloc.alignedAlloc(f32, std.mem.Alignment.fromByteUnits(32), total_elems);
    errdefer alloc.free(blocks);

    const scratch_tmp: []align(32) u8 = try alloc.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), matmul_f32.scratch_bytes);
    defer alloc.free(scratch_tmp);

    var bi: usize = 0;
    while (bi < k_blocks) : (bi += 1) {
        const kk0: usize = bi * key.kc;
        const k_sub: usize = @min(key.kc, key.k_dim - kk0);
        const b_off: usize = kk0 * key.c_out;
        const b_need: usize = k_sub * key.c_out;
        const b_bytes: []const u8 = std.mem.sliceAsBytes(wv[b_off .. b_off + b_need]);

        try matmul_f32.pack_b_tile(scratch_tmp, k_sub, key.c_out, 0, b_bytes);

        const dst_start: usize = bi * block_elems;
        const dst: []align(32) f32 = @alignCast(blocks[dst_start .. dst_start + block_elems]);
        const src: []align(32) const f32 = @alignCast(std.mem.bytesAsSlice(f32, scratch_tmp[0 .. block_elems * @sizeOf(f32)]));
        @memcpy(dst, src);
    }

    const entry: PackedWeightEntry = .{
        .key = key,
        .blocks = blocks,
        .block_count = k_blocks,
        .block_elems = block_elems,
    };
    cache.map.put(cache.allocator, key, entry) catch return error.OutOfMemory;
    return entry;
}

/// `x` rounded up to a whole multiple of `m` (at least `m`).
pub fn roundUpToMultiple(x: usize, m: usize) usize {
    if (m == 0) return x;
    return ((x + m - 1) / m) * m;
}

/// Rows staged at a time when gathering into a k-major panel.
///
/// 16 f32 is a cache line, so a group's corner turn writes whole lines, and the
/// group itself (`16 * kc`) stays in L1 — which is the point: the panel is then
/// written exactly once, instead of being written row-major and read back by a
/// separate transpose.
pub const KMAJOR_GROUP: usize = 16;

/// Deepest K block the staging below is sized for. Every shipped tuning's `kc`
/// is at or under this; a deeper one would need it raised.
pub const KMAJOR_KC_MAX: usize = 2048;

threadlocal var kmajor_stage: [KMAJOR_GROUP * KMAJOR_KC_MAX]f32 align(64) = undefined;

/// This worker's staging rows for a `kc`-deep gather. Thread-local because the
/// row loops that fill it run one per worker, and it is scratch either way.
pub fn kMajorStage(kc: usize) []f32 {
    std.debug.assert(kc <= KMAJOR_KC_MAX);
    return kmajor_stage[0 .. KMAJOR_GROUP * kc];
}

/// Move `rows` staged rows into a `[k][mr_total]` panel starting at row `r0`.
/// `stage` holds them row-major, `kc_stride` apart, `k_count` of each valid.
pub fn cornerTurnKMajor(
    dst: []f32,
    stage: []const f32,
    mr_total: usize,
    r0: usize,
    rows: usize,
    k_count: usize,
    kc_stride: usize,
) void {
    transposeF32(dst[r0..], mr_total, stage, kc_stride, rows, k_count);
}

/// `dst[c * dst_stride + r] = src[r * src_stride + c]` over `rows` x `cols`.
///
/// The 4x4 core is written as pair shuffles because the naive element form
/// compiles to lane inserts — a third more instructions for the same work.
pub fn transposeF32(dst: []f32, dst_stride: usize, src: []align(1) const f32, src_stride: usize, rows: usize, cols: usize) void {
    @setRuntimeSafety(false);
    const V4 = @Vector(4, f32);
    var c: usize = 0;
    while (c + 4 <= cols) : (c += 4) {
        var r: usize = 0;
        while (r + 4 <= rows) : (r += 4) {
            const a: V4 = src[(r + 0) * src_stride + c ..][0..4].*;
            const b: V4 = src[(r + 1) * src_stride + c ..][0..4].*;
            const d: V4 = src[(r + 2) * src_stride + c ..][0..4].*;
            const e: V4 = src[(r + 3) * src_stride + c ..][0..4].*;
            const t0 = @shuffle(f32, a, b, [4]i32{ 0, -1, 2, -3 });
            const t1 = @shuffle(f32, a, b, [4]i32{ 1, -2, 3, -4 });
            const t2 = @shuffle(f32, d, e, [4]i32{ 0, -1, 2, -3 });
            const t3 = @shuffle(f32, d, e, [4]i32{ 1, -2, 3, -4 });
            dst[(c + 0) * dst_stride + r ..][0..4].* = @shuffle(f32, t0, t2, [4]i32{ 0, 1, -1, -2 });
            dst[(c + 1) * dst_stride + r ..][0..4].* = @shuffle(f32, t1, t3, [4]i32{ 0, 1, -1, -2 });
            dst[(c + 2) * dst_stride + r ..][0..4].* = @shuffle(f32, t0, t2, [4]i32{ 2, 3, -3, -4 });
            dst[(c + 3) * dst_stride + r ..][0..4].* = @shuffle(f32, t1, t3, [4]i32{ 2, 3, -3, -4 });
        }
        while (r < rows) : (r += 1) {
            inline for (0..4) |i| dst[(c + i) * dst_stride + r] = src[r * src_stride + c + i];
        }
    }
    while (c < cols) : (c += 1) {
        for (0..rows) |r| dst[c * dst_stride + r] = src[r * src_stride + c];
    }
}

/// One NHWC image rewritten as zero-padded channel planes: channel `c` starts at
/// `data[c * plane_elems]` with row stride `wp`, and input `(ih, iw)` lands at
/// `(ih + pad_top) * wp + (iw + pad_left)`.
///
/// A unit-stride conv reads output position `(oh, ow)` at tap `(kh, kw)` from
/// `(oh + kh*dil_h, ow + kw*dil_w)` — the pads cancel — so a whole row of output
/// positions is a run of consecutive floats. That is what lets the GEMM take one
/// pointer per reduction index instead of a gathered panel.
pub const ChannelMajorImage = struct {
    data: []f32,
    wp: usize,
    plane_elems: usize,

    /// A partial panel reads up to `MR-1` columns past the last output column.
    /// The GEMM drops those rows, so they only have to stay in bounds.
    pub const overrun: usize = 32;

    pub fn elems(c_in: usize, hp: usize, wp: usize) usize {
        return c_in * hp * wp + overrun;
    }

    pub fn at(self: @This(), c: usize, row: usize, col: usize) [*]const f32 {
        return self.data.ptr + c * self.plane_elems + row * self.wp + col;
    }
};

pub const ImageGeometry = struct {
    h_in: usize,
    w_in: usize,
    c_in: usize,
    pad_top: usize,
    pad_left: usize,
    hp: usize,
    wp: usize,
};

/// Transpose input rows `first..last` of one NHWC image into `dst`.
///
/// The pad border is not written here: one `@memset` over the whole buffer is
/// cheaper than the tens of thousands of few-element memsets that zeroing each
/// row's left and right margin separately would take.
pub fn buildChannelMajorRows(dst: []f32, x: []align(1) const f32, g: ImageGeometry, first: usize, last: usize) void {
    @setRuntimeSafety(false);
    const plane: usize = g.hp * g.wp;
    for (first..last) |ih| {
        const src = x[ih * g.w_in * g.c_in ..][0 .. g.w_in * g.c_in];
        const base: usize = (g.pad_top + ih) * g.wp + g.pad_left;
        transposeF32(dst[base..], plane, src, g.c_in, g.w_in, g.c_in);
    }
}

threadlocal var indirect_table: [KMAJOR_KC_MAX][*]const f32 = undefined;

/// This worker's reduction-index pointer table. Thread-local for the same reason
/// as `kMajorStage`: one panel is built and consumed by one worker.
pub fn indirectTable(k_sub: usize) [][*]const f32 {
    std.debug.assert(k_sub <= KMAJOR_KC_MAX);
    return indirect_table[0..k_sub];
}

/// One pointer per reduction index, each to `MR` consecutive activations.
/// Reduction indices run (kh, kw, ic) — the order weights are packed in.
pub fn buildIndirectTable(
    table: [][*]const f32,
    img: ChannelMajorImage,
    plane0: usize,
    c_in: usize,
    k_w: usize,
    dil_h: usize,
    dil_w: usize,
    kk0: usize,
    oh: usize,
    ow: usize,
) void {
    @setRuntimeSafety(false);
    const pos: usize = kk0 / c_in;
    var ic: usize = kk0 - pos * c_in;
    var kh: usize = pos / k_w;
    var kw: usize = pos - kh * k_w;
    var row_ptr: [*]const f32 = img.at(plane0 + ic, oh + kh * dil_h, ow + kw * dil_w);
    for (table) |*e| {
        e.* = row_ptr;
        ic += 1;
        row_ptr += img.plane_elems;
        if (ic == c_in) {
            ic = 0;
            kw += 1;
            if (kw == k_w) {
                kw = 0;
                kh += 1;
            }
            row_ptr = img.at(plane0, oh + kh * dil_h, ow + kw * dil_w);
        }
    }
}

/// Elements the packed-A region must hold for `m_cap` rows.
///
/// Packing always writes WHOLE `mr`-row panels, so a cap that is not a multiple
/// of the kernel's `mr` still needs the rounded-up panel count. `m_cap * kc` only
/// happened to be right while every kernel's `mr` divided every tuning's `mc`.
pub fn packedAElems(mr: usize, m_cap: usize, kc: usize) usize {
    if (mr == 0) return m_cap * kc;
    return ((m_cap + mr - 1) / mr) * mr * kc;
}

pub fn scratchForTid(ctx: *const ConvExecCtx, tid: usize) ExecuteProgramError![]align(32) u8 {
    if (ctx.matmul_scratch.len != 0 and tid < ctx.matmul_scratch.len) return ctx.matmul_scratch[tid];
    if (ctx.matmul_scratch.len != 0) return ctx.matmul_scratch[0];
    const scratch_need: usize = matmul_registry.maxScratchBytes();
    return ctx.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), scratch_need) catch return BackendError.ExecutionFailed;
}

pub fn fillWeightBlock(
    dst: []f32,
    w_full: []const f32,
    k_dim: usize,
    c_out: usize,
    oc_start: usize,
    oc_count: usize,
) ExecuteProgramError!void {
    if (oc_count == 0) return BackendError.InvalidArgument;
    if (oc_start + oc_count > c_out) return BackendError.InvalidArgument;
    if (dst.len < k_dim * oc_count) return BackendError.InvalidArgument;
    if (w_full.len < k_dim * c_out) return BackendError.InvalidArgument;

    var k: usize = 0;
    while (k < k_dim) : (k += 1) {
        const src_off: usize = k * c_out + oc_start;
        const dst_off: usize = k * oc_count;
        @memcpy(dst[dst_off .. dst_off + oc_count], w_full[src_off .. src_off + oc_count]);
    }
}

/// Run a depthwise task across the pool: an item is a block of one output row.
pub fn runDepthwise(ctx: *ConvExecCtx, task: *@import("../kernels/conv2d.zig").DepthwiseConv2DTask) void {
    const items = task.items();
    if (ctx.pool) |p| {
        if (ctx.thread_count > 1 and items >= 2) return p.parallelForAny(@ptrCast(task), items, 1, ctx.depthwise_conv2d.run_items);
    }
    ctx.depthwise_conv2d.run_item_range(task, 0, items);
}
