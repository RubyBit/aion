// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const reduce_k = @import("../kernels/reduce.zig");
const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");

const BackendError = types.BackendError;
const DType = types.DType;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

const MAX_RANK: usize = 8;

/// Bytes of work below which forking the pool costs more than it saves, and the
/// size of one chunk of work when it does fork.
pub const parallel_min_bytes: usize = 256 * 1024;

/// Run `body(ctx, start, end)` over `[0, units)` -- elements, rows, whatever unit the
/// caller splits by, each `unit_bytes` of work. The range is cut into one chunk per
/// ~256 KiB of work, and the pool wakes only as many workers as there are chunks:
/// forking costs far more than a light op on a small tensor (a 512 KiB row gather is
/// 0.008 ms inline, 0.25 ms across 32 threads), so work earns threads by volume, and
/// a few-KiB decode tensor never forks at all. Work enough for every thread is cut
/// several chunks per thread and claimed on demand, so fast cores are not left
/// waiting on slow ones (hybrid CPUs mix both).
/// `body` must be safe to run on disjoint ranges concurrently; its last argument is
/// the running thread's id, for per-thread scratch.
pub fn parallelRange(
    comptime E: type,
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    units: usize,
    unit_bytes: usize,
    ctx: anytype,
    comptime body: fn (@TypeOf(ctx), usize, usize, usize) E!void,
) E!void {
    const p = pool orelse return body(ctx, 0, units, 0);
    const total = std.math.mul(usize, units, unit_bytes) catch std.math.maxInt(usize);
    const by_volume = @min(units, total / parallel_min_bytes);
    if (@min(thread_count, by_volume) < 2) return body(ctx, 0, units, 0);
    const claim = by_volume >= thread_count;
    const chunks = if (claim) @min(by_volume, CLAIM_CHUNKS_PER_THREAD * thread_count) else by_volume;
    const Job = struct {
        ctx: @TypeOf(ctx),
        units: usize,
        chunks: usize,
        failure: std.atomic.Value(u16) = .init(0),

        fn run(raw: *anyopaque, start: usize, end: usize, tid: usize) E!void {
            const job: *@This() = @ptrCast(@alignCast(raw));
            return body(job.ctx, start * job.units / job.chunks, end * job.units / job.chunks, tid);
        }

        fn runClaimed(raw: *anyopaque, start: usize, end: usize, tid: usize) void {
            const job: *@This() = @ptrCast(@alignCast(raw));
            if (job.failure.load(.monotonic) != 0) return;
            run(raw, start, end, tid) catch |err| {
                _ = job.failure.cmpxchgStrong(0, @intFromError(err), .release, .monotonic);
            };
        }
    };
    var job: Job = .{ .ctx = ctx, .units = units, .chunks = chunks };
    if (!claim) return p.parallelForFallible(E, &job, chunks, 1, Job.run);
    p.parallelForDynamic(&job, chunks, 1, Job.runClaimed);
    const failure = job.failure.load(.acquire);
    if (failure != 0) return @errorCast(@errorFromInt(failure));
}

/// Chunks per thread `parallelRange` cuts work into when every thread has some.
const CLAIM_CHUNKS_PER_THREAD: usize = 4;

/// `v` restricted to `[lo, hi)` along `axis`; its shape is written into `shape`, which
/// must outlive the result. Works for const and mutable buffer views alike.
pub fn sliceAxis(v: anytype, axis: usize, lo: usize, hi: usize, shape: *[8]usize) @TypeOf(v) {
    const r: usize = v.layout.rank;
    @memcpy(shape[0..r], v.layout.shape);
    shape[axis] = hi - lo;
    const stride: usize = @intCast(v.layout.strides_bytes[axis]);
    var sub = v;
    sub.bytes = v.bytes[lo * stride ..];
    sub.layout.shape = shape[0..r];
    return sub;
}

pub fn elemCountFromView(view: anytype) usize {
    // view.layout.rank is u8; shape slice length matches rank.
    if (view.layout.rank == 0) return 1;
    var acc: usize = 1;
    var d: usize = 0;
    while (d < @as(usize, view.layout.rank)) : (d += 1) {
        acc *= view.layout.shape[d];
    }
    return acc;
}

fn scalarElemBytes(dtype: DType) ExecuteProgramError!usize {
    return switch (dtype) {
        .f32 => 4,
        .f16 => 2,
        .i8 => 1,
        .i32 => 4,
        else => return BackendError.InvalidArgument,
    };
}

pub fn elemCountFromShape(shape: []const usize) ExecuteProgramError!usize {
    if (shape.len == 0) return BackendError.InvalidArgument;
    var acc: usize = 1;
    for (shape) |d| {
        acc = std.math.mul(usize, acc, d) catch return BackendError.InvalidArgument;
    }
    return acc;
}

/// Side of the square blocks `transposeWhole` moves: both a block's source rows and
/// its destination rows stay in L1 while it is copied.
const TRANSPOSE_BLOCK: usize = 32;

/// `dst[rows, cols] = src[cols, rows]^T`, split across threads by
/// bands of destination rows and copied block by block.
fn transposeWhole(pool: ?*thread_pool.ThreadPool, thread_count: usize, store: tensor_store.TensorStore, dst_id: tensor_store.TensorId, src_id: tensor_store.TensorId, rows: usize, cols: usize, elem_bytes: usize) ExecuteProgramError!void {
    const dst_view = try store.acquireMut(dst_id);
    defer store.releaseMut(dst_view.token);
    const src_view = try store.acquireConst(src_id);
    defer store.releaseConst(src_view.token);
    const Ctx = struct {
        rows: usize,
        cols: usize,
        elem_bytes: usize,
        dst: []u8,
        src: []const u8,

        fn run(c: @This(), lo: usize, hi: usize, _: usize) BackendError!void {
            switch (c.elem_bytes) {
                1 => c.band(u8, lo, hi),
                2 => c.band(u16, lo, hi),
                4 => c.band(u32, lo, hi),
                else => return BackendError.InvalidArgument,
            }
        }

        /// Destination row bands `[lo, hi)`, in units of `TRANSPOSE_BLOCK` rows.
        fn band(c: @This(), comptime T: type, lo: usize, hi: usize) void {
            const dst: [*]align(1) T = @ptrCast(c.dst.ptr);
            const src: [*]align(1) const T = @ptrCast(c.src.ptr);
            for (lo..hi) |b| {
                const r0 = b * TRANSPOSE_BLOCK;
                const r1 = @min(r0 + TRANSPOSE_BLOCK, c.rows);
                var c0: usize = 0;
                while (c0 < c.cols) : (c0 += TRANSPOSE_BLOCK) {
                    const c1 = @min(c0 + TRANSPOSE_BLOCK, c.cols);
                    for (r0..r1) |r| {
                        for (c0..c1) |col| dst[r * c.cols + col] = src[col * c.rows + r];
                    }
                }
            }
        }
    };
    const ctx: Ctx = .{ .rows = rows, .cols = cols, .elem_bytes = elem_bytes, .dst = dst_view.bufferView().bytes, .src = src_view.bufferView().bytes };
    const bands = std.math.divCeil(usize, rows, TRANSPOSE_BLOCK) catch unreachable;
    return parallelRange(BackendError, pool, thread_count, bands, 2 * TRANSPOSE_BLOCK * cols * elem_bytes, ctx, Ctx.run);
}

/// Sum of a tensor: each chunk adds its range into its thread's slot
/// of `partials`, and the slots are summed in f64.
fn sumWhole(pool: ?*thread_pool.ThreadPool, thread_count: usize, partials: []f32, store: tensor_store.TensorStore, a_id: tensor_store.TensorId, dtype: DType) ExecuteProgramError!f64 {
    const lease = try store.acquireConst(a_id);
    defer store.releaseConst(lease.token);
    const v = lease.bufferView();
    const slots = partials[0..@max(thread_count, 1)];
    @memset(slots, 0.0);
    const Ctx = struct {
        dtype: DType,
        bytes: []const u8,
        slots: []f32,

        fn run(c: @This(), lo: usize, hi: usize, tid: usize) BackendError!void {
            c.slots[tid] += switch (c.dtype) {
                .f32 => try reduce_k.sumF32Range(c.bytes, lo, hi),
                .f16 => try reduce_k.sumF16RangeToF32(c.bytes, lo, hi),
                else => return BackendError.InvalidArgument,
            };
        }
    };
    const ctx: Ctx = .{ .dtype = dtype, .bytes = v.bytes, .slots = slots };
    try parallelRange(BackendError, pool, thread_count, elemCountFromView(v), try scalarElemBytes(dtype), ctx, Ctx.run);
    var sum: f64 = 0.0;
    for (slots) |part| sum += part;
    return sum;
}

/// Output elements `reduceAxisWhole` accumulates at once when the reduced axis
/// is not innermost.
const REDUCE_COLUMN_BLOCK: usize = 256;

/// The input is `[outer, axis, inner]` in one flat run and the output
/// `[outer, inner]`; output elements split across threads directly. An
/// innermost axis sums contiguous runs; otherwise a block of adjacent outputs
/// accumulates row by row, so every read is contiguous too.
fn reduceAxisWhole(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    op: types.ReduceOp,
    store: tensor_store.TensorStore,
    out_id: tensor_store.TensorId,
    a_id: tensor_store.TensorId,
    a_meta: tensor_store.TensorMeta,
    axis: usize,
) ExecuteProgramError!void {
    const rank: usize = a_meta.rank;
    const axis_len = a_meta.shape[axis];
    var inner: usize = 1;
    for (a_meta.shape[axis + 1 .. rank]) |d| inner *= d;
    var outer: usize = 1;
    for (a_meta.shape[0..axis]) |d| outer *= d;
    const dtype = a_meta.dtype;
    if (dtype != .f32 and dtype != .f16 and dtype != .i32) return BackendError.InvalidArgument;

    const out_view = try store.acquireMut(out_id);
    defer store.releaseMut(out_view.token);
    const a_view = try store.acquireConst(a_id);
    defer store.releaseConst(a_view.token);

    const Ctx = struct {
        op: types.ReduceOp,
        dtype: DType,
        axis_len: usize,
        inner: usize,
        out: []u8,
        a: []const u8,

        fn load(c: @This(), i: usize) f64 {
            return switch (c.dtype) {
                .f32 => @as([*]align(1) const f32, @ptrCast(c.a.ptr))[i],
                .f16 => @floatCast(@as([*]align(1) const f16, @ptrCast(c.a.ptr))[i]),
                .i32 => @floatFromInt(@as([*]align(1) const i32, @ptrCast(c.a.ptr))[i]),
                else => unreachable,
            };
        }

        fn put(c: @This(), i: usize, sum: f64) void {
            const v = if (c.op == .mean) sum / @as(f64, @floatFromInt(c.axis_len)) else sum;
            switch (c.dtype) {
                .f32 => @as([*]align(1) f32, @ptrCast(c.out.ptr))[i] = @floatCast(v),
                .f16 => @as([*]align(1) f16, @ptrCast(c.out.ptr))[i] = @floatCast(@as(f32, @floatCast(v))),
                .i32 => @as([*]align(1) i32, @ptrCast(c.out.ptr))[i] = @intFromFloat(v),
                else => unreachable,
            }
        }

        fn run(c: @This(), lo: usize, hi: usize, _: usize) BackendError!void {
            if (c.inner == 1) {
                for (lo..hi) |o| {
                    const start = o * c.axis_len;
                    const sum: f64 = switch (c.dtype) {
                        .f32 => try reduce_k.sumF32Range(c.a, start, start + c.axis_len),
                        .f16 => try reduce_k.sumF16RangeToF32(c.a, start, start + c.axis_len),
                        else => blk: {
                            var acc: f64 = 0.0;
                            for (start..start + c.axis_len) |i| acc += c.load(i);
                            break :blk acc;
                        },
                    };
                    c.put(o, sum);
                }
                return;
            }
            var acc: [REDUCE_COLUMN_BLOCK]f64 = undefined;
            var q = lo;
            while (q < hi) {
                // A run of outputs inside one outer row, at most a block wide.
                const o = q / c.inner;
                const j0 = q % c.inner;
                const n = @min(REDUCE_COLUMN_BLOCK, c.inner - j0, hi - q);
                @memset(acc[0..n], 0.0);
                const base = o * c.axis_len * c.inner + j0;
                for (0..c.axis_len) |r| {
                    const row = base + r * c.inner;
                    for (0..n) |j| acc[j] += c.load(row + j);
                }
                for (0..n) |j| c.put(q + j, acc[j]);
                q += n;
            }
        }
    };
    const ctx: Ctx = .{ .op = op, .dtype = dtype, .axis_len = axis_len, .inner = inner, .out = out_view.bufferView().bytes, .a = a_view.bufferView().bytes };
    return parallelRange(BackendError, pool, thread_count, outer * inner, axis_len * try scalarElemBytes(dtype), ctx, Ctx.run);
}

/// `dst` takes `src`'s elements in row-major order under a new shape: one copy.
pub fn reshapeCopyScalar(store: tensor_store.TensorStore, dst_id: tensor_store.TensorId, src_id: tensor_store.TensorId) ExecuteProgramError!void {
    const dst_meta = try store.meta(dst_id);
    const src_meta = try store.meta(src_id);
    if (dst_meta.dtype != src_meta.dtype) return BackendError.InvalidArgument;
    _ = try scalarElemBytes(dst_meta.dtype);
    if (try elemCountFromShape(src_meta.shape) != try elemCountFromShape(dst_meta.shape)) return BackendError.InvalidArgument;

    const src = try store.acquireConst(src_id);
    defer store.releaseConst(src.token);
    const dst = try store.acquireMut(dst_id);
    defer store.releaseMut(dst.token);
    if (dst.bytes.len != src.bytes.len) return BackendError.InvalidArgument;
    @memcpy(dst.bytes, src.bytes);
}

pub fn transpose2DCopyScalar(pool: ?*thread_pool.ThreadPool, thread_count: usize, store: tensor_store.TensorStore, dst_id: tensor_store.TensorId, src_id: tensor_store.TensorId) ExecuteProgramError!void {
    const dst_meta = try store.meta(dst_id);
    const src_meta = try store.meta(src_id);
    if (dst_meta.rank != 2 or src_meta.rank != 2 or dst_meta.dtype != src_meta.dtype) return BackendError.InvalidArgument;
    if (dst_meta.shape[0] != src_meta.shape[1] or dst_meta.shape[1] != src_meta.shape[0]) return BackendError.InvalidArgument;
    return transposeWhole(pool, thread_count, store, dst_id, src_id, dst_meta.shape[0], dst_meta.shape[1], try scalarElemBytes(dst_meta.dtype));
}

/// `dst = src[starts : starts + dst.shape]`. Trailing axes taken whole merge into
/// the innermost axis that is not, so every copy is one contiguous run of it.
pub fn sliceNDCopyScalar(store: tensor_store.TensorStore, dst_id: tensor_store.TensorId, src_id: tensor_store.TensorId, starts: []const usize) ExecuteProgramError!void {
    const dst_meta = try store.meta(dst_id);
    const src_meta = try store.meta(src_id);
    if (dst_meta.dtype != src_meta.dtype) return BackendError.InvalidArgument;
    const rank: usize = dst_meta.rank;
    if (rank == 0 or rank != src_meta.rank or rank > MAX_RANK or starts.len != rank) return BackendError.InvalidArgument;
    const elem: usize = try scalarElemBytes(dst_meta.dtype);
    for (0..rank) |d| {
        if (dst_meta.shape[d] == 0 or starts[d] + dst_meta.shape[d] > src_meta.shape[d]) return BackendError.InvalidArgument;
    }

    // `j`: the innermost axis not taken whole; everything inside it is one run.
    var j: usize = rank;
    var inner: usize = 1;
    while (j > 0 and starts[j - 1] == 0 and dst_meta.shape[j - 1] == src_meta.shape[j - 1]) : (j -= 1) inner *= dst_meta.shape[j - 1];

    const src = try store.acquireConst(src_id);
    defer store.releaseConst(src.token);
    const dst = try store.acquireMut(dst_id);
    defer store.releaseMut(dst.token);
    if (j == 0) return @memcpy(dst.bytes, src.bytes);

    const axis = j - 1;
    const run = dst_meta.shape[axis] * inner * elem;
    // Element strides of src.
    var src_stride: [MAX_RANK]usize = undefined;
    var acc: usize = 1;
    var d = rank;
    while (d > 0) {
        d -= 1;
        src_stride[d] = acc;
        acc *= src_meta.shape[d];
    }
    var outer: usize = 1;
    for (dst_meta.shape[0..axis]) |n| outer *= n;
    var coord: [MAX_RANK]usize = @splat(0);
    for (0..outer) |o| {
        var src_off: usize = starts[axis] * src_stride[axis];
        for (0..axis) |a| src_off += (starts[a] + coord[a]) * src_stride[a];
        @memcpy(dst.bytes[o * run ..][0..run], src.bytes[src_off * elem ..][0..run]);
        // Next outer coordinate.
        var a = axis;
        while (a > 0) {
            a -= 1;
            coord[a] += 1;
            if (coord[a] < dst_meta.shape[a]) break;
            coord[a] = 0;
        }
    }
}

/// Inputs joined along `axis`: with the output viewed as `[outer, axis_total * inner]`,
/// each input fills one contiguous run of every outer row.
pub fn concatScalar(step: executable.StepConcatScalar, store: tensor_store.TensorStore) ExecuteProgramError!void {
    const out_meta = try store.meta(step.out);
    const rank: usize = out_meta.rank;
    if (rank == 0 or rank > MAX_RANK or step.axis >= rank) return BackendError.InvalidArgument;
    const count: usize = step.input_count;
    if (count == 0) return BackendError.InvalidArgument;
    const elem: usize = try scalarElemBytes(out_meta.dtype);

    var outer: usize = 1;
    for (out_meta.shape[0..step.axis]) |n| outer *= n;
    var inner: usize = 1;
    for (out_meta.shape[step.axis + 1 .. rank]) |n| inner *= n;
    const out_row = out_meta.shape[step.axis] * inner * elem;

    const out = try store.acquireMut(step.out);
    defer store.releaseMut(out.token);

    var at: usize = 0;
    for (step.inputs[0..count]) |in_id| {
        const m = try store.meta(in_id);
        if (m.dtype != out_meta.dtype or m.rank != rank) return BackendError.InvalidArgument;
        for (0..rank) |d| if (d != step.axis and m.shape[d] != out_meta.shape[d]) return BackendError.InvalidArgument;
        const run = m.shape[step.axis] * inner * elem;
        if (at + run > out_row) return BackendError.InvalidArgument;
        const in = try store.acquireConst(in_id);
        defer store.releaseConst(in.token);
        for (0..outer) |o| @memcpy(out.bytes[o * out_row + at ..][0..run], in.bytes[o * run ..][0..run]);
        at += run;
    }
    if (at != out_row) return BackendError.InvalidArgument;
}

pub fn reduceAllScalar(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    scratch_f32: []f32,
    op: types.ReduceOp,
    out_id: tensor_store.TensorId,
    a_id: tensor_store.TensorId,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    const out_meta = try store.meta(out_id);
    const a_meta = try store.meta(a_id);
    if (out_meta.dtype != a_meta.dtype) return BackendError.InvalidArgument;
    if (scratch_f32.len < @max(thread_count, 1)) return BackendError.InvalidArgument;

    var result: f64 = try sumWhole(pool, thread_count, scratch_f32, store, a_id, a_meta.dtype);
    if (op == .mean) result /= @floatFromInt(try elemCountFromShape(a_meta.shape));

    const out = try store.acquireMut(out_id);
    defer store.releaseMut(out.token);
    switch (out_meta.dtype) {
        .f32 => @as(*align(1) f32, @ptrCast(out.bytes.ptr)).* = @floatCast(result),
        .f16 => @as(*align(1) f16, @ptrCast(out.bytes.ptr)).* = @floatCast(@as(f32, @floatCast(result))),
        else => return BackendError.InvalidArgument,
    }
}

pub fn reduceAxisScalar(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    op: types.ReduceOp,
    out_id: tensor_store.TensorId,
    a_id: tensor_store.TensorId,
    axis: usize,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    const out_meta = try store.meta(out_id);
    const a_meta = try store.meta(a_id);
    if (out_meta.dtype != a_meta.dtype) return BackendError.InvalidArgument;
    const in_rank: usize = a_meta.rank;
    if (in_rank == 0 or in_rank > MAX_RANK or axis >= in_rank) return BackendError.InvalidArgument;
    if (@as(usize, out_meta.rank) != (if (in_rank == 1) 1 else in_rank - 1)) return BackendError.InvalidArgument;
    if (a_meta.shape[axis] == 0) return BackendError.InvalidArgument;
    return reduceAxisWhole(pool, thread_count, op, store, out_id, a_id, a_meta, axis);
}

test "reshapeCopyScalar: preserves row-major order" {
    const manager_mod = @import("../../../storage/manager.zig");
    const testing = std.testing;

    var sm = manager_mod.StorageManager.init(testing.allocator);
    defer sm.deinit();

    const src_id = try sm.createTensor(.f32, &.{ 2, 4 }, .{});
    const dst_id = try sm.createTensor(.f32, &.{ 4, 2 }, .{});
    var vals: [8]f32 = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
    try sm.writeFromPackedScalar(src_id, std.mem.sliceAsBytes(vals[0..]));
    try reshapeCopyScalar(sm.tensorStore(), dst_id, src_id);

    var out: [8]f32 = undefined;
    try sm.readToPackedScalar(dst_id, std.mem.sliceAsBytes(out[0..]));
    try testing.expectEqualSlices(f32, vals[0..], out[0..]);
}
