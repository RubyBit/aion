// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Top-k over the last axis: for each of the `outer = prod(shape[:-1])` rows, the
// `k` best of its `n` values and the index each came from, sorted best-first.
//
// A bounded k-heap scanned in one pass rather than a sort of the row: sampling
// wants k in the tens out of a vocabulary in the hundreds of thousands, so the
// cost that matters is O(n log k) with no scratch proportional to `n`.
const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");

const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");
const simd = @import("../kernels/simd.zig");
const thread_pool = @import("../../../runtime/thread_pool.zig");
const exec_utils = @import("utils.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

/// k values that fit a worker's stack. Sampling k is in the tens; larger falls
/// back to one allocation per worker, not per row.
const stack_k: usize = 64;

/// One candidate. `value` is widened to f32, which is order-preserving for f16, so
/// both dtypes break ties on the same index.
const Entry = struct { value: f32, index: i32 };

/// The k-best ordering: better first, and on a tie the lower index. Every
/// comparison here goes through it, so "better" has exactly one definition and
/// CPU and GPU cannot disagree about a tie.
fn better(a: Entry, b: Entry, largest: bool) bool {
    if (a.value != b.value) return if (largest) a.value > b.value else a.value < b.value;
    return a.index < b.index;
}

fn lessThanBest(largest: bool, a: Entry, b: Entry) bool {
    return better(a, b, largest);
}

/// Sift `heap[root]` down so the WORST entry ends up at index 0 — that is the one
/// a new candidate must beat, which is what keeps the row scan O(n log k).
fn siftDown(heap: []Entry, root_start: usize, largest: bool) void {
    var root = root_start;
    while (true) {
        const left = 2 * root + 1;
        if (left >= heap.len) return;
        var worst = left;
        const right = left + 1;
        if (right < heap.len and better(heap[worst], heap[right], largest)) worst = right;
        if (!better(heap[root], heap[worst], largest)) return;
        std.mem.swap(Entry, &heap[root], &heap[worst]);
        root = worst;
    }
}

/// `best` (length k) <- the k best of `row`, sorted best-first.
fn topkRow(comptime T: type, row: []align(1) const T, best: []Entry, largest: bool) void {
    for (best, 0..) |*e, i| e.* = .{ .value = @floatCast(row[i]), .index = @intCast(i) };
    var i: usize = best.len / 2;
    while (i > 0) {
        i -= 1;
        siftDown(best, i, largest);
    }
    var j: usize = best.len;
    while (j < row.len) : (j += 1) {
        const cand: Entry = .{ .value = @floatCast(row[j]), .index = @intCast(j) };
        if (better(cand, best[0], largest)) {
            best[0] = cand;
            siftDown(best, 0, largest);
        }
    }
    // Heap order is not sorted order, and the contract is best-first.
    std.mem.sort(Entry, best, largest, lessThanBest);
}

fn Rows(comptime T: type) type {
    return struct {
        in_buf: []align(1) const T,
        val_buf: []align(1) T,
        idx_buf: []align(1) i32,
        n: usize,
        k: usize,
        largest: bool,
        allocator: std.mem.Allocator,
        stop: std.atomic.Value(bool) = .init(false),
        err_mutex: std.Io.Mutex = .init,
        err_any: ?anyerror = null,

        fn fail(t: *@This(), err: anyerror) void {
            if (t.stop.swap(true, .acq_rel)) return;
            std.Io.Threaded.mutexLock(&t.err_mutex);
            defer std.Io.Threaded.mutexUnlock(&t.err_mutex);
            if (t.err_any == null) t.err_any = err;
        }

        fn run(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
            _ = tid;
            const t: *@This() = @ptrCast(@alignCast(ctx_any));
            if (start >= end or t.stop.load(.acquire)) return;

            var stack_buf: [stack_k]Entry = undefined;
            var heap_buf: ?[]Entry = null;
            defer if (heap_buf) |b| t.allocator.free(b);
            const best: []Entry = if (t.k <= stack_k) stack_buf[0..t.k] else blk: {
                const b = t.allocator.alloc(Entry, t.k) catch |e| {
                    t.fail(e);
                    return;
                };
                heap_buf = b;
                break :blk b;
            };

            var r = start;
            while (r < end) : (r += 1) {
                if (t.stop.load(.acquire)) return;
                topkRow(T, t.in_buf[r * t.n ..][0..t.n], best, t.largest);
                const out_base = r * t.k;
                for (best, 0..) |e, i| {
                    t.val_buf[out_base + i] = @floatCast(e.value);
                    t.idx_buf[out_base + i] = e.index;
                }
            }
        }
    };
}

fn runRows(
    comptime T: type,
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    allocator: std.mem.Allocator,
    in_bytes: []const u8,
    val_bytes: []u8,
    idx_bytes: []u8,
    outer: usize,
    n: usize,
    k: usize,
    largest: bool,
) ExecuteProgramError!void {
    const Task = Rows(T);
    var task: Task = .{
        .in_buf = simd.bytesAsSliceConstUnaligned(T, in_bytes),
        .val_buf = simd.bytesAsSliceMutUnaligned(T, val_bytes),
        .idx_buf = simd.bytesAsSliceMutUnaligned(i32, idx_bytes),
        .n = n,
        .k = k,
        .largest = largest,
        .allocator = allocator,
    };
    if (task.in_buf.len < outer * n or task.val_buf.len < outer * k or task.idx_buf.len < outer * k) {
        return BackendError.InvalidArgument;
    }

    // A row is already a full scan of `n`, so rows are the only unit worth splitting.
    if (pool) |p| {
        if (exec_utils.shouldParallelTiles(thread_count, outer, n * @sizeOf(T), 256 * 1024)) {
            const grain: usize = @max(1, (64 * 1024) / @max(n, 1));
            p.parallelForAny(@ptrCast(&task), outer, grain, Task.run);
            if (task.err_any) |e| return @errorCast(e);
            return;
        }
    }
    Task.run(@ptrCast(&task), 0, outer, 0);
    if (task.err_any) |e| return @errorCast(e);
}

pub fn execTopK(
    allocator: std.mem.Allocator,
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepTopK,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    const in_meta = try store.meta(s.a);
    const val_meta = try store.meta(s.values);
    const idx_meta = try store.meta(s.indices);

    const rank: usize = @as(usize, in_meta.rank);
    if (rank == 0 or s.axis != rank - 1) return BackendError.InvalidArgument;
    if (in_meta.dtype != .f32 and in_meta.dtype != .f16) return BackendError.InvalidArgument;
    if (val_meta.dtype != in_meta.dtype or idx_meta.dtype != .i32) return BackendError.InvalidArgument;
    if (val_meta.rank != in_meta.rank or idx_meta.rank != in_meta.rank) return BackendError.InvalidArgument;

    const n: usize = in_meta.shape[rank - 1];
    const k: usize = s.k;
    if (k == 0 or k > n) return BackendError.InvalidArgument;

    var outer: usize = 1;
    for (in_meta.shape[0 .. rank - 1]) |dim| {
        outer = std.math.mul(usize, outer, dim) catch return BackendError.InvalidArgument;
    }
    for (0..rank) |d| {
        const want: usize = if (d == rank - 1) k else in_meta.shape[d];
        if (val_meta.shape[d] != want or idx_meta.shape[d] != want) return BackendError.InvalidArgument;
        if (in_meta.tile_counts[d] != 1 or val_meta.tile_counts[d] != 1 or idx_meta.tile_counts[d] != 1) {
            return BackendError.InvalidArgument;
        }
    }

    const in_tile = try store.acquireTileConstLinear(s.a, 0);
    defer store.releaseConst(in_tile.token);
    const val_tile = try store.acquireTileMutLinear(s.values, 0);
    defer store.releaseMut(val_tile.token);
    const idx_tile = try store.acquireTileMutLinear(s.indices, 0);
    defer store.releaseMut(idx_tile.token);

    const in_bytes = in_tile.bufferView().bytes;
    const val_bytes = val_tile.bufferView().bytes;
    const idx_bytes = idx_tile.bufferView().bytes;

    return switch (in_meta.dtype) {
        .f32 => runRows(f32, pool, thread_count, allocator, in_bytes, val_bytes, idx_bytes, outer, n, k, s.largest),
        .f16 => runRows(f16, pool, thread_count, allocator, in_bytes, val_bytes, idx_bytes, outer, n, k, s.largest),
        else => BackendError.InvalidArgument,
    };
}
