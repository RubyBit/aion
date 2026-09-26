// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");

const executable = @import("../../../runtime/executable.zig");
const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");

const simd = @import("../kernels/simd.zig");
const matmul_registry = @import("../registry/matmul_registry.zig");
const matmul_q_registry = @import("../registry/matmul_q_registry.zig");
const matvec_registry = @import("../registry/matvec_registry.zig");
const matmul_routing = @import("../registry/matmul_routing.zig");
const exec_utils = @import("utils.zig");
const backend_utils = @import("../../utils.zig");

const BackendError = types.BackendError;
const DType = types.DType;
const MatMulParams = types.MatMulParams;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

pub const MatMulExecCtx = struct {
    allocator: std.mem.Allocator,
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,

    matmul_f32: matmul_registry.F32Kernels,
    matmul_q: matmul_q_registry.Choice,
    matvec: matvec_registry.Kernels,

    // Per-thread scratch.
    matmul_scratch: [][]align(32) u8,
};

/// `C = alpha * A @ B + beta * C`. A rank-2 B is one weight shared by every row of
/// A, so A's and C's leading dims fold into M; a batched B runs one 2D matmul per
/// batch, its size-1 batch dims broadcast.
///
/// The kernels compute in f32. An f16 A or C is converted at the edges — A once up
/// front, C through an f32 copy — which is O(MK + MN) beside the O(MNK) product.
pub fn execMatMul(ctx: *MatMulExecCtx, s: executable.StepMatMul, store: tensor_store.TensorStore) ExecuteProgramError!void {
    const c_meta = try store.meta(s.c);
    const a_meta = try store.meta(s.a);
    const b_meta = try store.meta(s.b);
    switch (b_meta.dtype) {
        .f32, .f16, .q8_0, .q4_0 => {},
        else => return BackendError.Unsupported,
    }
    if (b_meta.rank != 2 and (b_meta.rank != c_meta.rank or a_meta.rank != c_meta.rank)) return BackendError.InvalidArgument;

    const a_view = try store.acquireConst(s.a);
    defer store.releaseConst(a_view.token);
    const b_view = try store.acquireConst(s.b);
    defer store.releaseConst(b_view.token);
    const c_view = try store.acquireMut(s.c);
    defer store.releaseMut(c_view.token);

    if (a_meta.dtype == .f32 and c_meta.dtype == .f32) return matmulFlat(ctx, s, c_meta, a_meta, b_meta, c_view.bytes, a_view.bytes, b_view.bytes);

    const a32 = try widen(ctx.allocator, a_meta.dtype, @constCast(a_view.bytes));
    defer if (a32.ptr != a_view.bytes.ptr) unwiden(ctx.allocator, a32);
    const c32 = try widen(ctx.allocator, c_meta.dtype, c_view.bytes);
    defer if (c32.ptr != c_view.bytes.ptr) unwiden(ctx.allocator, c32);
    try matmulFlat(ctx, s, c_meta, a_meta, b_meta, c32, a32, b_view.bytes);
    if (c_meta.dtype == .f16) {
        const dst: []align(1) f16 = std.mem.bytesAsSlice(f16, c_view.bytes);
        const src: []align(1) const f32 = std.mem.bytesAsSlice(f32, c32);
        for (dst, src) |*d, v| d.* = @floatCast(v);
    }
}

/// f32 bytes of a float tensor: the bytes themselves when already f32, a new f32
/// copy of an f16 one.
fn widen(allocator: std.mem.Allocator, dtype: DType, bytes: []u8) ExecuteProgramError![]u8 {
    return switch (dtype) {
        .f32 => bytes,
        .f16 => blk: {
            const src: []align(1) const f16 = std.mem.bytesAsSlice(f16, bytes);
            const out = allocator.alloc(f32, src.len) catch return BackendError.ExecutionFailed;
            for (out, src) |*d, v| d.* = v;
            break :blk std.mem.sliceAsBytes(out);
        },
        else => BackendError.Unsupported,
    };
}

/// Free a copy `widen` made.
fn unwiden(allocator: std.mem.Allocator, bytes: []u8) void {
    allocator.free(@as([]f32, @alignCast(std.mem.bytesAsSlice(f32, bytes))));
}

fn matmulFlat(ctx: *MatMulExecCtx, s: executable.StepMatMul, c_meta: tensor_store.TensorMeta, a_meta: tensor_store.TensorMeta, b_meta: tensor_store.TensorMeta, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) ExecuteProgramError!void {
    const cr: usize = c_meta.rank;
    const k: usize = a_meta.shape[a_meta.rank - 1];
    const n: usize = c_meta.shape[cr - 1];
    if (b_meta.rank == 2) {
        var m: usize = 1;
        for (c_meta.shape[0 .. cr - 1]) |d| m *= d;
        return matmul2D(ctx, s, m, n, k, c_bytes, a_bytes, b_bytes, b_meta.dtype);
    }

    const batched: Batched = .init(c_meta, a_meta, b_meta, c_bytes, a_bytes, b_bytes);
    const m: usize = c_meta.shape[cr - 2];
    const batches = batched.count();

    // A GEMM per batch: one grid over every batch's tasks, sized so the whole of it
    // is about a task per thread.
    if (!matvecRoute(m, b_meta.dtype)) {
        const threads_per_batch = std.math.divCeil(usize, @max(ctx.thread_count, 1), batches) catch unreachable;
        const plan = planGemm(ctx, s, m, n, k, b_meta.dtype, threads_per_batch);
        const Job = struct {
            plan: GemmPlan,
            batched: Batched,
            scratch: [][]align(32) u8,

            fn run(x: @This(), lo: usize, hi: usize, tid: usize) BackendError!void {
                for (lo..hi) |i| {
                    const off = x.batched.offsets(i / x.plan.tasks);
                    try gemmTask(x.plan, i % x.plan.tasks, x.scratch[tid], x.batched.c[off.c..], x.batched.a[off.a..], x.batched.b[off.b..]);
                }
            }
        };
        const job: Job = .{ .plan = plan, .batched = batched, .scratch = ctx.matmul_scratch };
        return exec_utils.parallelRange(BackendError, ctx.pool, ctx.thread_count, batches * plan.tasks, exec_utils.parallel_min_bytes, job, Job.run);
    }

    // Matvec-shaped batches: with a thread's worth of them, each thread takes whole
    // batches (alone, on its own scratch); otherwise the batches run in turn.
    const Job = struct {
        ctx: *MatMulExecCtx,
        s: executable.StepMatMul,
        batched: Batched,
        m: usize,
        n: usize,
        k: usize,
        serial: bool,

        fn run(x: @This(), lo: usize, hi: usize, tid: usize) ExecuteProgramError!void {
            var solo: MatMulExecCtx = x.ctx.*;
            if (x.serial) {
                solo.pool = null;
                solo.thread_count = 1;
                solo.matmul_scratch = x.ctx.matmul_scratch[tid .. tid + 1];
            }
            for (lo..hi) |bi| {
                const off = x.batched.offsets(bi);
                try matmul2D(&solo, x.s, x.m, x.n, x.k, x.batched.c[off.c..], x.batched.a[off.a..], x.batched.b[off.b..], x.batched.b_dtype);
            }
        }
    };
    const whole_batches = ctx.pool != null and batches >= ctx.thread_count;
    const job: Job = .{ .ctx = ctx, .s = s, .batched = batched, .m = m, .n = n, .k = k, .serial = whole_batches };
    if (!whole_batches) return job.run(0, batches, 0);
    return exec_utils.parallelRange(ExecuteProgramError, ctx.pool, ctx.thread_count, batches, m * n * k * @sizeOf(f32), job, Job.run);
}

/// The batch coordinates of a batched matmul, as byte offsets into A, B and C.
const Batched = struct {
    rank: usize,
    shape: [MAX_RANK]usize,
    /// Per batch axis: bytes between consecutive coordinates, 0 where the operand
    /// broadcasts over it.
    a_stride: [MAX_RANK]usize,
    b_stride: [MAX_RANK]usize,
    c_stride: [MAX_RANK]usize,
    b_dtype: DType,
    a: []const u8,
    b: []const u8,
    c: []u8,

    const MAX_RANK: usize = 8;

    fn init(c_meta: tensor_store.TensorMeta, a_meta: tensor_store.TensorMeta, b_meta: tensor_store.TensorMeta, c: []u8, a: []const u8, b: []const u8) Batched {
        const r: usize = c_meta.rank;
        var x: Batched = .{ .rank = r - 2, .shape = undefined, .a_stride = undefined, .b_stride = undefined, .c_stride = undefined, .b_dtype = b_meta.dtype, .a = a, .b = b, .c = c };
        // One matrix of each: A and C are f32, B any dtype (a quantized B is packed blocks).
        var a_run: usize = a_meta.shape[r - 2] * a_meta.shape[r - 1] * @sizeOf(f32);
        var b_run: usize = backend_utils.requiredBytesForElems(b_meta.dtype, b_meta.shape[r - 2] * b_meta.shape[r - 1]) catch 0;
        var c_run: usize = c_meta.shape[r - 2] * c_meta.shape[r - 1] * @sizeOf(f32);
        var axis = r - 2;
        while (axis > 0) {
            axis -= 1;
            x.shape[axis] = c_meta.shape[axis];
            x.a_stride[axis] = if (a_meta.shape[axis] == 1) 0 else a_run;
            x.b_stride[axis] = if (b_meta.shape[axis] == 1) 0 else b_run;
            x.c_stride[axis] = c_run;
            a_run *= a_meta.shape[axis];
            b_run *= b_meta.shape[axis];
            c_run *= c_meta.shape[axis];
        }
        return x;
    }

    fn count(x: Batched) usize {
        var n: usize = 1;
        for (x.shape[0..x.rank]) |d| n *= d;
        return n;
    }

    const Offsets = struct { a: usize = 0, b: usize = 0, c: usize = 0 };

    /// Batch `bi` of C, and the batch of A and of B it reads.
    fn offsets(x: Batched, bi: usize) Offsets {
        var rem = bi;
        var off: Offsets = .{};
        var axis: usize = x.rank;
        while (axis > 0) {
            axis -= 1;
            const coord = rem % x.shape[axis];
            rem /= x.shape[axis];
            off.a += coord * x.a_stride[axis];
            off.b += coord * x.b_stride[axis];
            off.c += coord * x.c_stride[axis];
        }
        return off;
    }
};

/// Columns a matvec thread takes at a time: a multiple of every kernel's column
/// step, and wide enough that a call's fixed cost stays small next to its reads.
const MATVEC_COLUMN_GROUP: usize = 16;

/// Whether `m` rows against a B of `b_dtype` go through a matvec kernel rather than
/// a packed GEMM: one row of floats, or up to `Q8_DIRECT_MAX_M` rows of q8.
fn matvecRoute(m: usize, b_dtype: DType) bool {
    return switch (b_dtype) {
        .f32, .f16 => m == 1,
        .q8_0 => matmul_routing.shouldUseQ8DirectMatvec(.{ .m = m, .n = 1, .k = 1 }),
        else => false,
    };
}

/// `C[m, n] = alpha * A[m, k] @ B[k, n] + beta * C` on flat row-major operands.
fn matmul2D(ctx: *MatMulExecCtx, s: executable.StepMatMul, m: usize, n: usize, k: usize, c: []u8, a: []const u8, b: []const u8, b_dtype: DType) ExecuteProgramError!void {
    if (m == 0 or n == 0) return;
    const groups = std.math.divCeil(usize, n, MATVEC_COLUMN_GROUP) catch unreachable;

    if (matvecRoute(m, b_dtype) and b_dtype != .q8_0) {
        // One row against a float B: the matvec kernels read column ranges of B in place.
        const Ctx = struct {
            mv: matvec_registry.Kernels,
            params: MatMulParams,
            f16: bool,
            c: []u8,
            a: []const u8,
            b: []const u8,

            fn run(x: @This(), lo: usize, hi: usize, _: usize) BackendError!void {
                const j0 = lo * MATVEC_COLUMN_GROUP;
                const cnt = @min(hi * MATVEC_COLUMN_GROUP, x.params.n) - j0;
                return if (x.f16) x.mv.matvec_f16_range(x.params, j0, cnt, x.c, x.a, x.b) else x.mv.matvec_f32_range(x.params, j0, cnt, x.c, x.a, x.b);
            }
        };
        const eb = b_dtype.info().block_bytes;
        const ctx2: Ctx = .{ .mv = ctx.matvec, .params = .{ .m = 1, .n = n, .k = k, .alpha = s.alpha, .beta = s.beta }, .f16 = b_dtype == .f16, .c = c, .a = a, .b = b };
        return exec_utils.parallelRange(BackendError, ctx.pool, ctx.thread_count, groups, MATVEC_COLUMN_GROUP * k * eb, ctx2, Ctx.run);
    }

    if (matvecRoute(m, b_dtype)) {
        // A few rows against a K-blocked q8 B: read B in place, column ranges per thread.
        const blk = b_dtype.info();
        const Ctx = struct {
            mv: matvec_registry.Kernels,
            m: usize,
            n: usize,
            k: usize,
            alpha: f32,
            beta: f32,
            block_bytes: usize,
            c: []u8,
            a: []const u8,
            b: []const u8,

            fn run(x: @This(), lo: usize, hi: usize, _: usize) BackendError!void {
                const j0 = lo * MATVEC_COLUMN_GROUP;
                const cnt = @min(hi * MATVEC_COLUMN_GROUP, x.n) - j0;
                const params: MatMulParams = .{ .m = x.m, .n = cnt, .k = x.k, .lda = x.k, .ldb = x.n, .ldc = x.n, .alpha = x.alpha, .beta = x.beta };
                return x.mv.matvec_q8_0_kmajor(params, x.c[j0 * @sizeOf(f32) ..], x.a, x.b[j0 * x.block_bytes ..]);
            }
        };
        const ctx2: Ctx = .{ .mv = ctx.matvec, .m = m, .n = n, .k = k, .alpha = s.alpha, .beta = s.beta, .block_bytes = blk.block_bytes, .c = c, .a = a, .b = b };
        const col_bytes = (k / blk.block_elems) * blk.block_bytes;
        return exec_utils.parallelRange(BackendError, ctx.pool, ctx.thread_count, groups, MATVEC_COLUMN_GROUP * col_bytes, ctx2, Ctx.run);
    }

    const plan = planGemm(ctx, s, m, n, k, b_dtype, ctx.thread_count);
    const Job = struct {
        plan: GemmPlan,
        scratch: [][]align(32) u8,
        c: []u8,
        a: []const u8,
        b: []const u8,

        fn run(x: @This(), lo: usize, hi: usize, tid: usize) BackendError!void {
            for (lo..hi) |task| try gemmTask(x.plan, task, x.scratch[tid], x.c, x.a, x.b);
        }
    };
    const job: Job = .{ .plan = plan, .scratch = ctx.matmul_scratch, .c = c, .a = a, .b = b };
    // Compute-heavy: a task is worth a thread as soon as there are two of them.
    return exec_utils.parallelRange(BackendError, ctx.pool, ctx.thread_count, plan.tasks, exec_utils.parallel_min_bytes, job, Job.run);
}

/// A blocked GEMM's grid: `row_groups x col_blocks` tasks, each a `rows x nc` block
/// of C that walks K in `kc` blocks.
const GemmPlan = struct {
    s: executable.StepMatMul,
    mk: matmul_registry.F32Kernels,
    qk: matmul_q_registry.QuantKernels,
    b_dtype: DType,
    m: usize,
    n: usize,
    k: usize,
    kc: usize,
    nc: usize,
    rows: usize,
    row_groups: usize,
    tasks: usize,
};

/// Lay out `threads` tasks over C. B is packed once per row group and A once per
/// column block, so the grid balances the two: `col_blocks ~ sqrt(threads * n * cost / m)`,
/// with `cost` what packing a B element costs next to copying an A element (a q4
/// block is unpacked, a q8 one reordered, floats only copied).
fn planGemm(ctx: *MatMulExecCtx, s: executable.StepMatMul, m: usize, n: usize, k: usize, b_dtype: DType, threads_in: usize) GemmPlan {
    const quant = b_dtype.info().is_quantized;
    const kc: usize = if (quant) ctx.matmul_q.default.tuning.kc else ctx.matmul_f32.tuning.kc;
    const nc_max: usize = if (quant) ctx.matmul_q.default.tuning.nc else ctx.matmul_f32.tuning.nc;
    const threads = @max(threads_in, 1);
    const cost: f64 = switch (b_dtype) {
        .q4_0 => 3.0,
        .q8_0 => 1.5,
        else => 1.0,
    };
    const ideal = @sqrt(@as(f64, @floatFromInt(threads)) * @as(f64, @floatFromInt(n)) * cost / @as(f64, @floatFromInt(m)));
    const max_blocks = std.math.divCeil(usize, n, MATVEC_COLUMN_GROUP) catch unreachable;
    const want_blocks = std.math.clamp(@as(usize, @intFromFloat(@round(ideal))), 1, max_blocks);
    const nc: usize = std.math.clamp(std.mem.alignForward(usize, std.math.divCeil(usize, n, want_blocks) catch unreachable, MATVEC_COLUMN_GROUP), MATVEC_COLUMN_GROUP, nc_max);
    const col_blocks = std.math.divCeil(usize, n, nc) catch unreachable;
    // Rows split only as far as the threads need, and never into slivers. Rounded
    // down: a grid a few tasks past a multiple of the threads idles most of them
    // through a second wave (36 tasks on 32 threads runs like 64).
    const min_rows: usize = 16;
    const want_groups = @max(@as(usize, 1), threads / col_blocks);
    const row_groups_max = @max(@as(usize, 1), m / min_rows);
    var rows = std.math.divCeil(usize, m, @min(want_groups, row_groups_max)) catch unreachable;
    rows = @max(rows, 1);
    const row_groups = std.math.divCeil(usize, m, rows) catch unreachable;
    return .{
        .s = s,
        .mk = ctx.matmul_f32,
        .qk = ctx.matmul_q.default,
        .b_dtype = b_dtype,
        .m = m,
        .n = n,
        .k = k,
        .kc = kc,
        .nc = nc,
        .rows = rows,
        .row_groups = row_groups,
        .tasks = col_blocks * row_groups,
    };
}

/// One task of `plan`: its `rows x nc` block of C, K walked in `kc` blocks -- each
/// B block packed straight from flat B, then the kernel run on strided A and C.
fn gemmTask(p: GemmPlan, task: usize, scratch: []align(32) u8, c: []u8, a: []const u8, b: []const u8) BackendError!void {
    const info = p.b_dtype.info();
    const j0 = (task / p.row_groups) * p.nc;
    const r0 = (task % p.row_groups) * p.rows;
    if (r0 >= p.m or j0 >= p.n) return;
    const nw = @min(p.nc, p.n - j0);
    const rw = @min(p.rows, p.m - r0);
    const c_blk = c[(r0 * p.n + j0) * @sizeOf(f32) ..];
    var pc: usize = 0;
    while (pc < p.k) : (pc += p.kc) {
        const kw = @min(p.kc, p.k - pc);
        const params: MatMulParams = .{ .m = rw, .n = nw, .k = kw, .lda = p.k, .ldc = p.n, .alpha = p.s.alpha, .beta = if (pc == 0) p.s.beta else 1.0 };
        const a_blk = a[(r0 * p.k + pc) * @sizeOf(f32) ..];
        switch (p.b_dtype) {
            .q8_0, .q4_0 => {
                // K-blocked B: block row `pc / block_elems`, N blocks apart.
                const b_blk = b[((pc / info.block_elems) * p.n + j0) * info.block_bytes ..];
                if (p.b_dtype == .q8_0) try p.qk.pack_b_tile_q8_0(scratch, kw, nw, p.n, b_blk) else try p.qk.pack_b_tile_q4_0(scratch, kw, nw, p.n, b_blk);
                const view: matmul_q_registry.PackedBView = @alignCast(scratch[0..p.qk.packed_b_bytes]);
                try p.qk.matmul_packed_b(scratch, view, params, c_blk, a_blk);
            },
            .f32, .f16 => {
                const pb_bytes = p.mk.tuning.kc * p.mk.tuning.nc * @sizeOf(f32);
                const b_blk = b[(pc * p.n + j0) * info.block_bytes ..];
                if (p.b_dtype == .f32) {
                    try p.mk.pack_b_tile(scratch, kw, nw, p.n, b_blk);
                } else {
                    const pb: []align(32) f32 = @alignCast(std.mem.bytesAsSlice(f32, scratch[0..pb_bytes]));
                    try p.mk.pack_b_tile_f16_to_packed_f32(pb, kw, nw, p.n, b_blk);
                }
                const view: []align(32) const f32 = @alignCast(std.mem.bytesAsSlice(f32, scratch[0..pb_bytes]));
                try p.mk.matmul_packed_b(scratch, view, params, c_blk, a_blk);
            },
            else => return BackendError.Unsupported,
        }
    }
}
