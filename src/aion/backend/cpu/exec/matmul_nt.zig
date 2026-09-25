// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const thread_pool = @import("../../../runtime/thread_pool.zig");
const matmul_nt_q = @import("../kernels/matmul_nt_q.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");
const exec_utils = @import("utils.zig");

const matmul_nt_registry = @import("../registry/matmul_nt_registry.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;
const DType = types.DType;

/// Elements per q8_0 block (2-byte f16 scale + 32 i8 values on disk). Only the element
/// count matters here; the kernels own the byte layout.
const Q8_0_BLOCK_ELEMS: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 34;

/// Least weight a worker's column run should stream to be worth handing out.
/// In bytes, not columns: a column is 17x more weight at K=8192 than at K=512.
const MIN_CHUNK_BYTES: usize = 64 * 1024;

pub const MatMulNtExecCtx = struct {
    matmul_nt: matmul_nt_registry.Kernels,
    /// One shared staging buffer for the quantized activation. Shared, not
    /// per-thread: every N tile reads the same rows and none of them writes.
    scratch: []align(32) u8 = &[_]u8{},
    /// The detected L2, which sizes the activation panel; 0 when unknown.
    l2_bytes: usize = 0,
    /// The detected L1D, which sizes the rows one kernel call holds; 0 when unknown.
    l1d_bytes: usize = 0,
};

/// Rows of A a kernel call sweeps B's chunk with: as many as fill half of L1,
/// leaving the rest to the B blocks in flight. Without a detected L1D, 32 KiB is
/// under what any current core has.
fn blockRows(l1d_bytes: usize, row_bytes: usize) usize {
    const budget: usize = if (l1d_bytes != 0) l1d_bytes / 2 else 16 * 1024;
    return @max(@as(usize, 1), budget / row_bytes);
}

/// Quantize `rows` rows of A into `out`, across the pool when there are rows to
/// share: they are independent, and at prefill the whole activation otherwise
/// waits on one thread while the rest sit idle.
fn prepareRows(pool: ?*thread_pool.ThreadPool, threads: usize, out: []u8, a_rows: []const u8, rows: usize, k: usize) ExecuteProgramError!void {
    const p = pool orelse return matmul_nt_q.prepareActivation(out, a_rows, rows, k);
    if (threads <= 1 or rows < threads) return matmul_nt_q.prepareActivation(out, a_rows, rows, k);
    const Job = struct {
        out: []u8,
        a: []const u8,
        k: usize,

        fn run(ctx: *anyopaque, start: usize, end: usize, _: usize) BackendError!void {
            const j: *@This() = @ptrCast(@alignCast(ctx));
            const out_row = matmul_nt_q.preparedBytes(1, j.k);
            const a_row = j.k * @sizeOf(f32);
            try matmul_nt_q.prepareActivation(j.out[start * out_row ..], j.a[start * a_row ..], end - start, j.k);
        }
    };
    var job: Job = .{ .out = out, .a = a_rows, .k = k };
    try p.parallelForFallible(BackendError, @ptrCast(&job), rows, 0, Job.run);
}

/// Activation rows to quantize and sweep B with at once.
///
/// Every panel costs one full pass over B, so taller is fewer passes; but inside
/// a pass the panel is re-read for each group of B's rows, so past the cache it
/// comes from memory every time. Half of L2 leaves the other half to B. Without
/// a detected L2, 256 KiB is under what any current core has.
fn panelRows(l2_bytes: usize, room_bytes: usize, row_bytes: usize) usize {
    const budget: usize = if (l2_bytes != 0) l2_bytes / 2 else 256 * 1024;
    return @max(@as(usize, 1), @min(budget, room_bytes) / row_bytes);
}

/// C[m, n] = alpha * sum_k A[m, k] * B[n, k]  +  beta * C[m, n]
///
/// Layout:
/// - A: f32, trailing axis K. Leading axes collapse to a flat M = prod(A.shape[:-1]).
/// - B: q8_0 `[N, K]` (quant_axis == 1, one row = K/32 contiguous q8_0 blocks) or f32 `[N, K]`.
/// - C: f32, trailing axis N.
///
/// Actual compute is delegated to the NT matmul registry:
/// - Q8_0 → `matmul_nt_q.Kernel(...).matmulNtQ8_0` via `matmul_nt_registry.Kernels.matmul_q8_0`
/// - F32  → `matmul_nt.Kernel(...).matmulNtF32` via `matmul_nt_registry.Kernels.matmul_f32`
///
/// Parallelism here is over N tiles (B's axis-0 tiling must match C's last-axis tiling,
/// enforced at compile time). Each worker handles a contiguous range of N tiles.
pub fn execMatMulNTTiled(
    ctx: *const MatMulNtExecCtx,
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepMatMulNTTiled,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    const c_meta = try store.meta(s.c);
    const a_meta = try store.meta(s.a);
    const b_meta = try store.meta(s.b);

    if (b_meta.rank != 2) return BackendError.InvalidArgument;
    if (a_meta.rank != c_meta.rank) return BackendError.InvalidArgument;
    if (c_meta.dtype != .f32 or a_meta.dtype != .f32) return BackendError.InvalidArgument;
    if (b_meta.dtype != .q8_0 and b_meta.dtype != .f32) return BackendError.InvalidArgument;

    const k: usize = b_meta.shape[1];
    const n_total: usize = b_meta.shape[0];
    if (a_meta.shape[a_meta.rank - 1] != k) return BackendError.InvalidArgument;
    if (c_meta.shape[c_meta.rank - 1] != n_total) return BackendError.InvalidArgument;
    if (b_meta.dtype == .q8_0 and (k % Q8_0_BLOCK_ELEMS) != 0) return BackendError.InvalidArgument;

    // `m_total` is the flattened product of leading C/A axes.
    var m_total: usize = 1;
    var d: usize = 0;
    while (d + 1 < @as(usize, c_meta.rank)) : (d += 1) {
        if (a_meta.shape[d] != c_meta.shape[d]) return BackendError.InvalidArgument;
        m_total = std.math.mul(usize, m_total, c_meta.shape[d]) catch return BackendError.InvalidArgument;
    }

    // A is held in a single tile spanning all (batch,seq) rows × K. Acquire once.
    var a_tile = try store.acquireTileConstLinear(s.a, 0);
    defer store.releaseConst(a_tile.token);
    const a_view = a_tile.bufferView();

    // A must occupy a single tile containing all M×K f32 elements (the compile step
    // retiles to `[...leading_full, K]` which is a single tile).
    const expected_a_bytes: usize = m_total * k * @sizeOf(f32);
    if (a_view.bytes.len < expected_a_bytes) return BackendError.InvalidArgument;

    // C's last-axis tile size is aligned to B's axis-0 tile size by the compile step,
    // so B-tile `nt` covers C's N range [nt*tile_size .. min((nt+1)*tile_size, N)].
    if (b_meta.tile_counts[0] != c_meta.tile_counts[c_meta.rank - 1]) return BackendError.InvalidArgument;
    if (b_meta.tile_shape[0] != c_meta.tile_shape[c_meta.rank - 1]) return BackendError.InvalidArgument;

    const tile_total: usize = c_meta.tile_counts[c_meta.rank - 1];
    const n_tile_size: usize = c_meta.tile_shape[c_meta.rank - 1];

    const Runner = struct {
        const Self = @This();

        store: tensor_store.TensorStore,
        c: tensor_store.TensorId,
        b: tensor_store.TensorId,
        n_total: usize,
        n_tile_size: usize,
        chunks_per_tile: usize,
        b_row_bytes: usize,
        block_order: types.QuantBlockOrder,
        k: usize,
        alpha: f32,
        beta: f32,
        /// The panel this pass computes: C rows `[row0, row0 + rows)`.
        row0: usize = 0,
        rows: usize = 0,
        /// Rows a kernel call takes at once (see `blockRows`).
        block_rows: usize = std.math.maxInt(usize),
        a: union(enum) {
            /// f32 rows of A, starting at `row0`.
            f32: []const u8,
            /// The same rows prepared for the q8 kernel (`matmul_nt_q.prepareActivation`),
            /// for the kernel of B's block order.
            q8_0: struct { rows: []const u8, row_bytes: usize, kernel: matmul_nt_registry.MatMulNtQ8_0Fn },
        } = undefined,
        matmul_nt_f32: matmul_nt_registry.MatMulNtF32Fn,

        /// `[start, end)` indexes column chunks, `chunks_per_tile` to a tile. B's
        /// rows are its output columns, so a chunk is a contiguous run of them and
        /// needs no more than a slice; C keeps the tile's width as its row stride.
        fn runRange(self: *@This(), start: usize, end: usize) ExecuteProgramError!void {
            var idx: usize = start;
            while (idx < end) : (idx += 1) {
                const nt: usize = idx / self.chunks_per_tile;
                const chunk: usize = idx % self.chunks_per_tile;

                const n_start: usize = nt * self.n_tile_size;
                const n_count: usize = @min(self.n_tile_size, self.n_total - n_start);
                // Chunks start on a group boundary: a grouped tile shares a
                // block's bytes across its rows, so a split inside a group would
                // address the wrong ones.
                const group: usize = self.block_order.groupRows();
                const raw = std.math.divCeil(usize, n_count, self.chunks_per_tile) catch return BackendError.InvalidArgument;
                const per_chunk: usize = (std.math.divCeil(usize, raw, group) catch return BackendError.InvalidArgument) * group;
                const col_lo: usize = chunk * per_chunk;
                if (col_lo >= n_count) continue;
                const cols: usize = @min(per_chunk, n_count - col_lo);

                var c_tile = try self.store.acquireTileMutLinear(self.c, nt);
                defer self.store.releaseMut(c_tile.token);
                const c_view = c_tile.bufferView();
                if ((c_view.bytes.len % @sizeOf(f32)) != 0) return BackendError.InvalidArgument;

                var b_tile = try self.store.acquireTileConstLinear(self.b, nt);
                defer self.store.releaseConst(b_tile.token);
                const b_view = b_tile.bufferView();

                const b_off: usize = col_lo * self.b_row_bytes;
                if (b_off > b_view.bytes.len) return BackendError.InvalidArgument;
                const c_off: usize = (self.row0 * n_count + col_lo) * @sizeOf(f32);
                if (c_off > c_view.bytes.len) return BackendError.InvalidArgument;

                // The kernels take the chunk, not the whole matrix: `params.n` is its
                // column count and `ldc` keeps C addressed at the tile's full width.
                // They range-check the slices against these dims themselves.
                //
                // Rows go in blocks that fit L1: a kernel sweeps every column for
                // each row it holds, so a block re-reads this chunk of B from L2
                // once, where all the rows at once would re-read A per column group.
                var r0: usize = 0;
                while (r0 < self.rows) : (r0 += self.block_rows) {
                    const params: types.MatMulParams = .{
                        .m = @min(self.block_rows, self.rows - r0),
                        .n = cols,
                        .k = self.k,
                        .ldc = n_count,
                        .alpha = self.alpha,
                        .beta = self.beta,
                    };
                    const c_block = c_view.bytes[c_off + r0 * n_count * @sizeOf(f32) ..];
                    switch (self.a) {
                        .q8_0 => |q| try q.kernel(params, c_block, q.rows[r0 * q.row_bytes ..], b_view.bytes[b_off..]),
                        .f32 => |a_rows| try self.matmul_nt_f32(params, c_block, a_rows[r0 * self.k * @sizeOf(f32) ..], b_view.bytes[b_off..]),
                    }
                }
            }
        }

        fn run(self: *@This(), workers: ?*thread_pool.ThreadPool, threads: usize, work_total: usize, tile_bytes: usize) ExecuteProgramError!void {
            if (workers) |p| {
                // A tile costs the B panel it streams, not the C row it writes. At
                // decode C is one f32 per column against half a megabyte of weights,
                // so sizing the split by C alone swept every tile into one grain and
                // left the whole matmul on a single worker.
                const min_total_bytes: usize = 256 * 1024;
                if (exec_utils.shouldParallelTiles(threads, work_total, tile_bytes, min_total_bytes)) {
                    const Task = struct {
                        runner: *Self,
                        stop: std.atomic.Value(bool) = .init(false),
                        err_mutex: std.Io.Mutex = .init,
                        err_any: ?anyerror = null,

                        fn fail(t: *@This(), err: anyerror) void {
                            if (t.stop.swap(true, .acq_rel)) return;
                            std.Io.Threaded.mutexLock(&t.err_mutex);
                            defer std.Io.Threaded.mutexUnlock(&t.err_mutex);
                            if (t.err_any == null) t.err_any = err;
                        }

                        fn runTiles(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
                            _ = tid;
                            const t: *@This() = @ptrCast(@alignCast(ctx_any));
                            if (start >= end) return;
                            if (t.stop.load(.acquire)) return;
                            t.runner.runRange(start, end) catch |e| {
                                t.fail(e);
                                return;
                            };
                        }
                    };

                    var task: Task = .{ .runner = self };
                    // Uniform cost per chunk, non-uniform cores: let them claim on
                    // demand, one chunk at a time. A chunk is hundreds of kilobytes of
                    // weights, so the atomic that hands it out costs nothing beside it.
                    p.parallelForDynamic(@ptrCast(&task), work_total, 1, Task.runTiles);
                    if (task.err_any) |e| return @errorCast(e);
                    return;
                }
            }
            try self.runRange(0, work_total);
        }
    };

    // One tile per worker leaves most of them idle on the narrow projections: a
    // 2048-wide one is eight tiles against six threads, and a 256-wide KV
    // projection is a single tile. Splitting a tile's columns costs nothing —
    // they are separate rows of B — and only kicks in where tiles are too few.
    const b_row_bytes: usize = switch (b_meta.dtype) {
        .q8_0 => (k / Q8_0_BLOCK_ELEMS) * Q8_0_BLOCK_BYTES,
        .f32 => k * @sizeOf(f32),
        else => return BackendError.Unsupported,
    };
    // Units per thread, for claiming to even out cores of unequal speed. At
    // decode a unit is its stream of B and a few suffice; with rows to sweep it
    // is heavy, and the last wave idles the rest: prefill gained up to 16.
    const per_thread: usize = if (m_total > 1) 16 else 4;
    const chunks_per_tile: usize = blk: {
        if (thread_count <= 1 or tile_total >= thread_count * per_thread) break :blk 1;
        const want = std.math.divCeil(usize, thread_count * per_thread, tile_total) catch 1;
        const by_width = @max(@as(usize, 1), (n_tile_size * b_row_bytes) / MIN_CHUNK_BYTES);
        break :blk @max(@as(usize, 1), @min(want, by_width));
    };
    const work_total: usize = tile_total * chunks_per_tile;
    const tile_bytes: usize = (exec_utils.tileByteSize(b_meta) + exec_utils.tileByteSize(c_meta)) / chunks_per_tile;

    var runner: Runner = .{
        .store = store,
        .c = s.c,
        .b = s.b,
        .n_total = n_total,
        .n_tile_size = n_tile_size,
        .chunks_per_tile = chunks_per_tile,
        .b_row_bytes = b_row_bytes,
        .block_order = b_meta.block_order,
        .k = k,
        .alpha = s.alpha,
        .beta = s.beta,
        .matmul_nt_f32 = ctx.matmul_nt.matmul_f32,
    };

    if (b_meta.dtype == .f32) {
        runner.rows = m_total;
        runner.a = .{ .f32 = a_view.bytes };
        return runner.run(pool, thread_count, work_total, tile_bytes);
    }

    // A panel at a time, each quantized once for every N tile of it to share — a
    // decode step has hundreds of tiles per matmul and they all read the same
    // activation. Decode is a single panel.
    const kernel = ctx.matmul_nt.matmul_q8_0.get(b_meta.block_order);
    const a_row_bytes: usize = matmul_nt_q.preparedBytes(1, k);
    if (ctx.scratch.len < a_row_bytes) return BackendError.Unsupported;
    const row_f32_bytes: usize = k * @sizeOf(f32);
    runner.block_rows = blockRows(ctx.l1d_bytes, a_row_bytes);
    const per_panel: usize = panelRows(ctx.l2_bytes, ctx.scratch.len, a_row_bytes);
    var row0: usize = 0;
    while (row0 < m_total) : (row0 += per_panel) {
        const rows: usize = @min(per_panel, m_total - row0);
        const a_rows = a_view.bytes[row0 * row_f32_bytes ..][0 .. rows * row_f32_bytes];
        const panel = ctx.scratch[0..matmul_nt_q.preparedBytes(rows, k)];
        try prepareRows(pool, thread_count, panel, a_rows, rows, k);
        runner.row0 = row0;
        runner.rows = rows;
        runner.a = .{ .q8_0 = .{ .rows = panel, .row_bytes = a_row_bytes, .kernel = kernel } };
        try runner.run(pool, thread_count, work_total, tile_bytes);
    }
}

// Rows are independent — each is quantized and swept on its own — so how many
// share a panel may change speed but never a result.
test "matmul_nt: results do not depend on how A is split into panels" {
    const manager_mod = @import("../../../storage/manager.zig");
    const cpu_target = @import("../registry/cpu_target.zig");
    const testing = std.testing;

    const m: usize = 7;
    const k: usize = 128;
    const n: usize = 16;
    var sm = manager_mod.StorageManager.init(testing.allocator);
    defer sm.deinit();

    var a_vals: [m * k]f32 = undefined;
    for (&a_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 29)) - 14)) * 0.03;
    const a = try sm.createTiledTensor(.f32, &.{ m, k }, &.{ m, k }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(a, std.mem.sliceAsBytes(&a_vals));

    var b_bytes: [n * (k / 32) * 34]u8 = undefined;
    for (0..n * (k / 32)) |bi| {
        std.mem.writeInt(u16, b_bytes[bi * 34 ..][0..2], @bitCast(@as(f16, 0.01)), .little);
        for (b_bytes[bi * 34 + 2 ..][0..32], 0..) |*q, i| q.* = @truncate(bi *% 13 +% i *% 7);
    }
    const b = try sm.createTiledTensor(.q8_0, &.{ n, k }, &.{ n, k }, .{ .tile_alignment = 64, .quant_axis = 1 });
    try sm.writeFromPackedQuant(b, &b_bytes);

    const kernels = comptime matmul_nt_registry.selectForTarget(cpu_target.compiled).kernels;
    var scratch: [4096]u8 align(32) = undefined;
    var out: [2][m * n]f32 = undefined;
    // All seven rows in one panel, then a scratch that holds only two at a time.
    for ([_]usize{ scratch.len, 2 * matmul_nt_q.preparedBytes(1, k) }, 0..) |room, i| {
        const c = try sm.createTiledTensor(.f32, &.{ m, n }, &.{ m, n }, .{ .tile_alignment = 64 });
        const ctx: MatMulNtExecCtx = .{ .matmul_nt = kernels, .scratch = scratch[0..room] };
        try execMatMulNTTiled(&ctx, null, 1, .{ .c = c, .a = a, .b = b, .alpha = 1.0, .beta = 0.0 }, sm.tensorStore());
        try sm.readToPackedScalar(c, std.mem.sliceAsBytes(&out[i]));
    }
    try testing.expectEqualSlices(f32, &out[0], &out[1]);
}
