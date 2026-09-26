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
    /// per-thread: every run of B reads the same rows and none of them writes.
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
/// B's rows are C's columns, so work splits into runs of them: a run needs no more
/// of B than a slice, and C keeps its full width as its row stride.
pub fn execMatMulNT(
    ctx: *const MatMulNtExecCtx,
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepMatMulNT,
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
    const n: usize = b_meta.shape[0];
    if (a_meta.shape[a_meta.rank - 1] != k) return BackendError.InvalidArgument;
    if (c_meta.shape[c_meta.rank - 1] != n) return BackendError.InvalidArgument;
    if (b_meta.dtype == .q8_0 and (k % Q8_0_BLOCK_ELEMS) != 0) return BackendError.InvalidArgument;

    var m_total: usize = 1;
    for (0..c_meta.rank - 1) |d| {
        if (a_meta.shape[d] != c_meta.shape[d]) return BackendError.InvalidArgument;
        m_total = std.math.mul(usize, m_total, c_meta.shape[d]) catch return BackendError.InvalidArgument;
    }

    const a_view = try store.acquireConst(s.a);
    defer store.releaseConst(a_view.token);
    const b_view = try store.acquireConst(s.b);
    defer store.releaseConst(b_view.token);
    const c_view = try store.acquireMut(s.c);
    defer store.releaseMut(c_view.token);
    if (a_view.bytes.len < m_total * k * @sizeOf(f32) or c_view.bytes.len < m_total * n * @sizeOf(f32)) return BackendError.InvalidArgument;

    const b_row_bytes: usize = switch (b_meta.dtype) {
        .q8_0 => (k / Q8_0_BLOCK_ELEMS) * Q8_0_BLOCK_BYTES,
        .f32 => k * @sizeOf(f32),
        else => return BackendError.Unsupported,
    };
    // Runs per thread, for claiming to even out cores of unequal speed. At decode a
    // run is its stream of B and a few suffice; with rows to sweep it is heavy, and
    // the last wave idles the rest: prefill gained up to 16. Never a run so short its
    // stream of B is not worth handing out.
    const per_thread: usize = if (m_total > 1) 16 else 4;
    const by_width = @max(@as(usize, 1), (n * b_row_bytes) / MIN_CHUNK_BYTES);
    const chunks: usize = if (thread_count <= 1) 1 else @max(@as(usize, 1), @min(thread_count * per_thread, by_width));

    var runner: Runner = .{
        .n = n,
        .chunks = chunks,
        .b_row_bytes = b_row_bytes,
        .block_order = b_meta.block_order,
        .k = k,
        .alpha = s.alpha,
        .beta = s.beta,
        .b = b_view.bytes,
        .c = c_view.bytes,
        .matmul_nt_f32 = ctx.matmul_nt.matmul_f32,
    };

    if (b_meta.dtype == .f32) {
        runner.rows = m_total;
        runner.a = .{ .f32 = a_view.bytes };
        return runner.run(pool, thread_count);
    }

    // A panel at a time, each quantized once for every run of B to share — they all
    // read the same activation. Decode is a single panel.
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
        try runner.run(pool, thread_count);
    }
}

const Runner = struct {
    n: usize,
    chunks: usize,
    b_row_bytes: usize,
    block_order: types.QuantBlockOrder,
    k: usize,
    alpha: f32,
    beta: f32,
    b: []const u8,
    c: []u8,
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

    /// Column runs `[start, end)`. A run starts on a group boundary: a grouped B
    /// shares a block's bytes across its rows, so a split inside a group would
    /// address the wrong ones.
    fn runRange(self: *const Runner, start: usize, end: usize) ExecuteProgramError!void {
        const group: usize = self.block_order.groupRows();
        const raw = std.math.divCeil(usize, self.n, self.chunks) catch return BackendError.InvalidArgument;
        const per_chunk: usize = (std.math.divCeil(usize, raw, group) catch return BackendError.InvalidArgument) * group;
        for (start..end) |chunk| {
            const col_lo: usize = chunk * per_chunk;
            if (col_lo >= self.n) continue;
            const cols: usize = @min(per_chunk, self.n - col_lo);
            const b_run = self.b[col_lo * self.b_row_bytes ..];
            const c_off: usize = (self.row0 * self.n + col_lo) * @sizeOf(f32);

            // Rows go in blocks that fit L1: a kernel sweeps every column for each
            // row it holds, so a block re-reads this run of B from L2 once, where
            // all the rows at once would re-read A per column group.
            var r0: usize = 0;
            while (r0 < self.rows) : (r0 += self.block_rows) {
                const params: types.MatMulParams = .{
                    .m = @min(self.block_rows, self.rows - r0),
                    .n = cols,
                    .k = self.k,
                    .ldc = self.n,
                    .alpha = self.alpha,
                    .beta = self.beta,
                };
                const c_block = self.c[c_off + r0 * self.n * @sizeOf(f32) ..];
                switch (self.a) {
                    .q8_0 => |q| try q.kernel(params, c_block, q.rows[r0 * q.row_bytes ..], b_run),
                    .f32 => |a_rows| try self.matmul_nt_f32(params, c_block, a_rows[r0 * self.k * @sizeOf(f32) ..], b_run),
                }
            }
        }
    }

    fn run(self: *const Runner, pool: ?*thread_pool.ThreadPool, thread_count: usize) ExecuteProgramError!void {
        const p = pool orelse return self.runRange(0, self.chunks);
        if (thread_count <= 1 or self.chunks < 2) return self.runRange(0, self.chunks);
        const Task = struct {
            runner: *const Runner,
            failure: std.atomic.Value(u16) = .init(0),

            fn claim(ctx_any: *anyopaque, start: usize, end: usize, _: usize) void {
                const t: *@This() = @ptrCast(@alignCast(ctx_any));
                if (t.failure.load(.monotonic) != 0) return;
                t.runner.runRange(start, end) catch |e| {
                    _ = t.failure.cmpxchgStrong(0, @intFromError(e), .release, .monotonic);
                };
            }
        };
        // Uniform cost per run, non-uniform cores: let them claim on demand, one run
        // at a time. A run is hundreds of kilobytes of weights, so the atomic that
        // hands it out costs nothing beside it.
        var task: Task = .{ .runner = self };
        p.parallelForDynamic(@ptrCast(&task), self.chunks, 1, Task.claim);
        const failure = task.failure.load(.acquire);
        if (failure != 0) return @errorCast(@errorFromInt(failure));
    }
};

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
    const a = try sm.createTensor(.f32, &.{ m, k }, .{});
    try sm.writeFromPackedScalar(a, std.mem.sliceAsBytes(&a_vals));

    var b_bytes: [n * (k / 32) * 34]u8 = undefined;
    for (0..n * (k / 32)) |bi| {
        std.mem.writeInt(u16, b_bytes[bi * 34 ..][0..2], @bitCast(@as(f16, 0.01)), .little);
        for (b_bytes[bi * 34 + 2 ..][0..32], 0..) |*q, i| q.* = @truncate(bi *% 13 +% i *% 7);
    }
    const b = try sm.createTensor(.q8_0, &.{ n, k }, .{ .quant_axis = 1 });
    try sm.writeFromPackedQuant(b, &b_bytes);

    const kernels = comptime matmul_nt_registry.selectForTarget(cpu_target.compiled).kernels;
    var scratch: [4096]u8 align(32) = undefined;
    var out: [2][m * n]f32 = undefined;
    // All seven rows in one panel, then a scratch that holds only two at a time.
    for ([_]usize{ scratch.len, 2 * matmul_nt_q.preparedBytes(1, k) }, 0..) |room, i| {
        const c = try sm.createTensor(.f32, &.{ m, n }, .{});
        const ctx: MatMulNtExecCtx = .{ .matmul_nt = kernels, .scratch = scratch[0..room] };
        try execMatMulNT(&ctx, null, 1, .{ .c = c, .a = a, .b = b, .alpha = 1.0, .beta = 0.0 }, sm.tensorStore());
        try sm.readToPackedScalar(c, std.mem.sliceAsBytes(&out[i]));
    }
    try testing.expectEqualSlices(f32, &out[0], &out[1]);
}
