const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");

const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");

const exec_utils = @import("utils.zig");
const fast_math = @import("../kernels/fast_math.zig");
const simd = @import("../kernels/simd.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

fn expFastVec(comptime lanes: usize, x: @Vector(lanes, f32)) @Vector(lanes, f32) {
    // Clamp to keep exp approx stable and sums nonzero.
    return fast_math.expApproxVecF32(lanes, fast_math.clampVecF32(lanes, x, -20.0, 0.0));
}

fn expFast(x: f32) f32 {
    return fast_math.expApproxF32(fast_math.clampF32(x, -20.0, 0.0));
}

pub fn execSoftmaxTiled(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    scratch_f32: []f32,
    s: executable.StepSoftmaxTiled,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    const out_meta = try store.meta(s.out);
    if (out_meta.dtype != .f32) return BackendError.InvalidArgument;
    if (!(out_meta.rank == 1 or out_meta.rank == 2)) return BackendError.InvalidArgument;

    if (out_meta.rank == 1) {
        return softmaxRank1(pool, thread_count, scratch_f32, store, out_meta, s.out, s.a);
    }

    // Work units: tile rows.
    const ti0_total: usize = out_meta.tile_counts[0];
    const tile_bytes: usize = exec_utils.tileByteSize(out_meta);
    const min_total_bytes: usize = 256 * 1024;

    if (pool) |p| {
        const total_bytes: usize = tile_bytes * (out_meta.tile_counts[0] * out_meta.tile_counts[1]);
        if (thread_count > 1 and ti0_total >= 2 and total_bytes >= min_total_bytes) {
            const Task = struct {
                store: tensor_store.TensorStore,
                out_meta: tensor_store.TensorMeta,
                out: tensor_store.TensorId,
                a: tensor_store.TensorId,

                stop: std.atomic.Value(bool) = .init(false),
                err_mutex: std.Thread.Mutex = .{},
                err_any: ?anyerror = null,

                fn fail(t: *@This(), err: anyerror) void {
                    if (t.stop.swap(true, .acq_rel)) return;
                    t.err_mutex.lock();
                    defer t.err_mutex.unlock();
                    if (t.err_any == null) t.err_any = err;
                }

                fn runTi0(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
                    _ = tid;
                    const t: *@This() = @ptrCast(@alignCast(ctx_any));
                    if (start >= end) return;
                    if (t.stop.load(.acquire)) return;
                    var ti0: usize = start;
                    while (ti0 < end) : (ti0 += 1) {
                        if (t.stop.load(.acquire)) return;
                        softmaxTi0(t.store, t.out_meta, t.out, t.a, ti0) catch |e| {
                            t.fail(e);
                            return;
                        };
                    }
                }
            };

            var task: Task = .{ .store = store, .out_meta = out_meta, .out = s.out, .a = s.a };
            // One work item per tile-row.
            p.parallelForAny(@ptrCast(&task), ti0_total, 1, Task.runTi0);
            if (task.err_any) |e| return @errorCast(e);
            return;
        }
    }

    // Sequential fallback.
    var ti0: usize = 0;
    while (ti0 < ti0_total) : (ti0 += 1) {
        try softmaxTi0(store, out_meta, s.out, s.a, ti0);
    }
}

fn softmaxRank1(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    scratch_f32: []f32,
    store: tensor_store.TensorStore,
    out_meta: tensor_store.TensorMeta,
    out: tensor_store.TensorId,
    a: tensor_store.TensorId,
) ExecuteProgramError!void {
    std.debug.assert(out_meta.rank == 1);

    const ti0_total: usize = out_meta.tile_counts[0];
    const tc1: usize = out_meta.tile_counts[1];

    const tile_bytes: usize = exec_utils.tileByteSize(out_meta);
    const min_total_bytes: usize = 256 * 1024;
    const total_bytes: usize = tile_bytes * (out_meta.tile_counts[0] * out_meta.tile_counts[1]);

    // Parallel path: reduce max + sum across ti0 tiles.
    if (pool) |p| {
        if (thread_count > 1 and ti0_total >= 2 and total_bytes >= min_total_bytes and scratch_f32.len >= thread_count * 2) {
            const scratch_max: []f32 = scratch_f32[0..thread_count];
            const scratch_sum: []f32 = scratch_f32[thread_count .. thread_count * 2];

            @memset(scratch_max, -std.math.inf(f32));
            @memset(scratch_sum, 0.0);

            const TaskMax = struct {
                store: tensor_store.TensorStore,
                out_meta: tensor_store.TensorMeta,
                a: tensor_store.TensorId,
                scratch_max: []f32,

                stop: std.atomic.Value(bool) = .init(false),
                err_mutex: std.Thread.Mutex = .{},
                err_any: ?anyerror = null,

                fn fail(t: *@This(), err: anyerror) void {
                    if (t.stop.swap(true, .acq_rel)) return;
                    t.err_mutex.lock();
                    defer t.err_mutex.unlock();
                    if (t.err_any == null) t.err_any = err;
                }

                fn runTi0(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
                    const t: *@This() = @ptrCast(@alignCast(ctx_any));
                    if (tid < t.scratch_max.len) t.scratch_max[tid] = -std.math.inf(f32);
                    if (start >= end) return;
                    if (t.stop.load(.acquire)) return;

                    var max_buf: [1]f32 = .{-std.math.inf(f32)};
                    var ti0: usize = start;
                    while (ti0 < end) : (ti0 += 1) {
                        if (t.stop.load(.acquire)) return;
                        var ti1: usize = 0;
                        while (ti1 < t.out_meta.tile_counts[1]) : (ti1 += 1) {
                            if (ti1 + 1 < t.out_meta.tile_counts[1]) t.store.prefetch(t.a, ti0, ti1 + 1);
                            const in_tile = t.store.acquireTileConst(t.a, ti0, ti1) catch |e| {
                                t.fail(e);
                                return;
                            };
                            defer t.store.releaseConst(in_tile.token);
                            const in_view = in_tile.bufferView();
                            if (in_view.dtype != .f32) {
                                t.fail(BackendError.InvalidArgument);
                                return;
                            }
                            updateMaxF32(max_buf[0..1], in_view, 1);
                        }
                    }
                    if (tid < t.scratch_max.len) t.scratch_max[tid] = max_buf[0];
                }
            };

            var task_max: TaskMax = .{ .store = store, .out_meta = out_meta, .a = a, .scratch_max = scratch_max };
            p.parallelForAny(@ptrCast(&task_max), ti0_total, 1, TaskMax.runTi0);
            if (task_max.err_any) |e| return @errorCast(e);

            var global_max: f32 = -std.math.inf(f32);
            for (scratch_max) |m| global_max = @max(global_max, m);

            const TaskExpSum = struct {
                store: tensor_store.TensorStore,
                out_meta: tensor_store.TensorMeta,
                out: tensor_store.TensorId,
                a: tensor_store.TensorId,
                global_max: f32,
                scratch_sum: []f32,

                stop: std.atomic.Value(bool) = .init(false),
                err_mutex: std.Thread.Mutex = .{},
                err_any: ?anyerror = null,

                fn fail(t: *@This(), err: anyerror) void {
                    if (t.stop.swap(true, .acq_rel)) return;
                    t.err_mutex.lock();
                    defer t.err_mutex.unlock();
                    if (t.err_any == null) t.err_any = err;
                }

                fn runTi0(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
                    const t: *@This() = @ptrCast(@alignCast(ctx_any));
                    if (tid < t.scratch_sum.len) t.scratch_sum[tid] = 0.0;
                    if (start >= end) return;
                    if (t.stop.load(.acquire)) return;

                    var sum_buf: [1]f32 = .{0.0};
                    const max_buf: [1]f32 = .{t.global_max};

                    var ti0: usize = start;
                    while (ti0 < end) : (ti0 += 1) {
                        if (t.stop.load(.acquire)) return;
                        var ti1: usize = 0;
                        while (ti1 < t.out_meta.tile_counts[1]) : (ti1 += 1) {
                            if (ti1 + 1 < t.out_meta.tile_counts[1]) {
                                t.store.prefetch(t.a, ti0, ti1 + 1);
                                t.store.prefetch(t.out, ti0, ti1 + 1);
                            }
                            var out_tile = t.store.acquireTileMut(t.out, ti0, ti1) catch |e| {
                                t.fail(e);
                                return;
                            };
                            defer t.store.releaseMut(out_tile.token);
                            const in_tile = t.store.acquireTileConst(t.a, ti0, ti1) catch |e| {
                                t.fail(e);
                                return;
                            };
                            defer t.store.releaseConst(in_tile.token);

                            const out_view = out_tile.bufferView();
                            const in_view = in_tile.bufferView();
                            if (out_view.dtype != .f32 or in_view.dtype != .f32) {
                                t.fail(BackendError.InvalidArgument);
                                return;
                            }
                            expSumStoreF32(sum_buf[0..1], out_view, in_view, max_buf[0..1], 1);
                        }
                    }

                    if (tid < t.scratch_sum.len) t.scratch_sum[tid] = sum_buf[0];
                }
            };

            var task_sum: TaskExpSum = .{ .store = store, .out_meta = out_meta, .out = out, .a = a, .global_max = global_max, .scratch_sum = scratch_sum };
            p.parallelForAny(@ptrCast(&task_sum), ti0_total, 1, TaskExpSum.runTi0);
            if (task_sum.err_any) |e| return @errorCast(e);

            var global_sum: f32 = 0.0;
            for (scratch_sum) |v| global_sum += v;
            if (!(global_sum > 0.0) or !std.math.isFinite(global_sum)) return BackendError.InvalidArgument;

            const inv: f32 = 1.0 / global_sum;

            const TaskNorm = struct {
                store: tensor_store.TensorStore,
                out_meta: tensor_store.TensorMeta,
                out: tensor_store.TensorId,
                inv: f32,

                stop: std.atomic.Value(bool) = .init(false),
                err_mutex: std.Thread.Mutex = .{},
                err_any: ?anyerror = null,

                fn fail(t: *@This(), err: anyerror) void {
                    if (t.stop.swap(true, .acq_rel)) return;
                    t.err_mutex.lock();
                    defer t.err_mutex.unlock();
                    if (t.err_any == null) t.err_any = err;
                }

                fn runTi0(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
                    _ = tid;
                    const t: *@This() = @ptrCast(@alignCast(ctx_any));
                    if (start >= end) return;
                    if (t.stop.load(.acquire)) return;

                    const lanes: usize = comptime simd.lanesF32();
                    const Vec = @Vector(lanes, f32);
                    const inv_v: Vec = @splat(t.inv);

                    var ti0: usize = start;
                    while (ti0 < end) : (ti0 += 1) {
                        if (t.stop.load(.acquire)) return;
                        var ti1: usize = 0;
                        while (ti1 < t.out_meta.tile_counts[1]) : (ti1 += 1) {
                            if (ti1 + 1 < t.out_meta.tile_counts[1]) t.store.prefetch(t.out, ti0, ti1 + 1);
                            var out_tile = t.store.acquireTileMut(t.out, ti0, ti1) catch |e| {
                                t.fail(e);
                                return;
                            };
                            defer t.store.releaseMut(out_tile.token);
                            const out_view = out_tile.bufferView();
                            if (out_view.dtype != .f32) {
                                t.fail(BackendError.InvalidArgument);
                                return;
                            }
                            var out_slice: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, out_view.bytes);
                            var i: usize = 0;
                            const vec_end: usize = out_slice.len - (out_slice.len % lanes);
                            while (i < vec_end) : (i += lanes) {
                                const v: Vec = @as(*align(1) const Vec, @ptrCast(out_slice.ptr + i)).*;
                                @as(*align(1) Vec, @ptrCast(out_slice.ptr + i)).* = v * inv_v;
                            }
                            while (i < out_slice.len) : (i += 1) out_slice[i] *= t.inv;
                        }
                    }
                }
            };

            var task_norm: TaskNorm = .{ .store = store, .out_meta = out_meta, .out = out, .inv = inv };
            p.parallelForAny(@ptrCast(&task_norm), ti0_total, 1, TaskNorm.runTi0);
            if (task_norm.err_any) |e| return @errorCast(e);
            return;
        }
    }

    // Sequential fallback.
    var global_max: f32 = -std.math.inf(f32);
    var ti0: usize = 0;
    while (ti0 < ti0_total) : (ti0 += 1) {
        var ti1: usize = 0;
        while (ti1 < tc1) : (ti1 += 1) {
            if (ti1 + 1 < tc1) store.prefetch(a, ti0, ti1 + 1);
            const in_tile = try store.acquireTileConst(a, ti0, ti1);
            defer store.releaseConst(in_tile.token);
            const in_view = in_tile.bufferView();
            if (in_view.dtype != .f32) return BackendError.InvalidArgument;
            var max_buf: [1]f32 = .{global_max};
            updateMaxF32(max_buf[0..1], in_view, 1);
            global_max = max_buf[0];
        }
    }

    var global_sum: f32 = 0.0;
    ti0 = 0;
    while (ti0 < ti0_total) : (ti0 += 1) {
        var ti1: usize = 0;
        while (ti1 < tc1) : (ti1 += 1) {
            if (ti1 + 1 < tc1) {
                store.prefetch(a, ti0, ti1 + 1);
                store.prefetch(out, ti0, ti1 + 1);
            }
            var out_tile = try store.acquireTileMut(out, ti0, ti1);
            defer store.releaseMut(out_tile.token);
            const in_tile = try store.acquireTileConst(a, ti0, ti1);
            defer store.releaseConst(in_tile.token);

            const out_view = out_tile.bufferView();
            const in_view = in_tile.bufferView();
            if (out_view.dtype != .f32 or in_view.dtype != .f32) return BackendError.InvalidArgument;
            var sum_buf: [1]f32 = .{global_sum};
            const max_buf: [1]f32 = .{global_max};
            expSumStoreF32(sum_buf[0..1], out_view, in_view, max_buf[0..1], 1);
            global_sum = sum_buf[0];
        }
    }
    if (!(global_sum > 0.0) or !std.math.isFinite(global_sum)) return BackendError.InvalidArgument;

    const inv: f32 = 1.0 / global_sum;
    ti0 = 0;
    while (ti0 < ti0_total) : (ti0 += 1) {
        var ti1: usize = 0;
        while (ti1 < tc1) : (ti1 += 1) {
            if (ti1 + 1 < tc1) store.prefetch(out, ti0, ti1 + 1);
            var out_tile = try store.acquireTileMut(out, ti0, ti1);
            defer store.releaseMut(out_tile.token);
            const out_view = out_tile.bufferView();
            if (out_view.dtype != .f32) return BackendError.InvalidArgument;

            // Multiply by inv directly (normalizeF32 assumes per-row sums).
            var out_slice: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, out_view.bytes);
            const lanes: usize = comptime simd.lanesF32();
            const Vec = @Vector(lanes, f32);
            const inv_v: Vec = @splat(inv);
            var i: usize = 0;
            const vec_end: usize = out_slice.len - (out_slice.len % lanes);
            while (i < vec_end) : (i += lanes) {
                const v: Vec = @as(*align(1) const Vec, @ptrCast(out_slice.ptr + i)).*;
                @as(*align(1) Vec, @ptrCast(out_slice.ptr + i)).* = v * inv_v;
            }
            while (i < out_slice.len) : (i += 1) out_slice[i] *= inv;
        }
    }
}

fn softmaxTi0(
    store: tensor_store.TensorStore,
    out_meta: tensor_store.TensorMeta,
    out: tensor_store.TensorId,
    a: tensor_store.TensorId,
    ti0: usize,
) ExecuteProgramError!void {
    // v0 contract: tile_shape[0] <= 256 validated at compile time.
    const tm: usize = out_meta.tile_shape[0];
    if (tm == 0 or tm > 256) return BackendError.InvalidArgument;

    var max_buf: [256]f32 = undefined;
    var sum_buf: [256]f32 = undefined;
    @memset(max_buf[0..tm], -std.math.inf(f32));
    @memset(sum_buf[0..tm], 0.0);

    const tc1: usize = out_meta.tile_counts[1];

    // Pass 1: max per row.
    var ti1: usize = 0;
    while (ti1 < tc1) : (ti1 += 1) {
        if (ti1 + 1 < tc1) store.prefetch(a, ti0, ti1 + 1);
        const in_tile = try store.acquireTileConst(a, ti0, ti1);
        defer store.releaseConst(in_tile.token);
        const in_view = in_tile.bufferView();
        if (in_view.dtype != .f32) return BackendError.InvalidArgument;
        updateMaxF32(max_buf[0..tm], in_view, out_meta.rank);
    }

    // Pass 2: exp(x-max), store to out, accumulate sum.
    ti1 = 0;
    while (ti1 < tc1) : (ti1 += 1) {
        if (ti1 + 1 < tc1) {
            store.prefetch(a, ti0, ti1 + 1);
            store.prefetch(out, ti0, ti1 + 1);
        }
        var out_tile = try store.acquireTileMut(out, ti0, ti1);
        defer store.releaseMut(out_tile.token);
        const in_tile = try store.acquireTileConst(a, ti0, ti1);
        defer store.releaseConst(in_tile.token);

        const out_view = out_tile.bufferView();
        const in_view = in_tile.bufferView();
        if (out_view.dtype != .f32 or in_view.dtype != .f32) return BackendError.InvalidArgument;
        expSumStoreF32(sum_buf[0..tm], out_view, in_view, max_buf[0..tm], out_meta.rank);
    }

    // Pass 3: normalize.
    ti1 = 0;
    while (ti1 < tc1) : (ti1 += 1) {
        if (ti1 + 1 < tc1) store.prefetch(out, ti0, ti1 + 1);
        var out_tile = try store.acquireTileMut(out, ti0, ti1);
        defer store.releaseMut(out_tile.token);
        const out_view = out_tile.bufferView();
        if (out_view.dtype != .f32) return BackendError.InvalidArgument;
        normalizeF32(out_view, sum_buf[0..tm], out_meta.rank);
    }
}

pub fn updateMaxF32(max_buf: []f32, in_view: types.BufferViewConst, rank: usize) void {
    const in: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, in_view.bytes);
    if (rank == 1) {
        // Single row.
        var m: f32 = max_buf[0];
        for (in) |v| m = @max(m, v);
        max_buf[0] = m;
        return;
    }

    const m_tile: usize = in_view.layout.shape[0];
    const n_tile: usize = in_view.layout.shape[1];
    var r: usize = 0;
    while (r < m_tile) : (r += 1) {
        var m: f32 = max_buf[r];
        const off: usize = r * n_tile;
        var c: usize = 0;
        while (c < n_tile) : (c += 1) {
            m = @max(m, in[off + c]);
        }
        max_buf[r] = m;
    }
}

pub fn expSumStoreF32(sum_buf: []f32, out_view: types.BufferViewMut, in_view: types.BufferViewConst, max_buf: []const f32, rank: usize) void {
    var out: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, out_view.bytes);
    const in: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, in_view.bytes);

    const lanes: usize = comptime simd.lanesF32();
    const Vec = @Vector(lanes, f32);

    if (rank == 1) {
        const m: f32 = max_buf[0];
        var acc_v: Vec = @splat(@as(f32, 0.0));

        var i: usize = 0;
        const vec_end: usize = in.len - (in.len % lanes);
        while (i < vec_end) : (i += lanes) {
            const xv: Vec = @as(*align(1) const Vec, @ptrCast(in.ptr + i)).*;
            const ev: Vec = expFastVec(lanes, xv - @as(Vec, @splat(m)));
            @as(*align(1) Vec, @ptrCast(out.ptr + i)).* = ev;
            acc_v += ev;
        }
        var acc: f32 = @reduce(.Add, acc_v);
        while (i < in.len) : (i += 1) {
            const e: f32 = expFast(in[i] - m);
            out[i] = e;
            acc += e;
        }
        sum_buf[0] += acc;
        return;
    }

    const m_tile: usize = in_view.layout.shape[0];
    const n_tile: usize = in_view.layout.shape[1];
    var r: usize = 0;
    while (r < m_tile) : (r += 1) {
        const m: f32 = max_buf[r];
        const off: usize = r * n_tile;

        var acc_v: Vec = @splat(@as(f32, 0.0));
        var c: usize = 0;
        const vec_end: usize = n_tile - (n_tile % lanes);
        while (c < vec_end) : (c += lanes) {
            const xv: Vec = @as(*align(1) const Vec, @ptrCast(in.ptr + off + c)).*;
            const ev: Vec = expFastVec(lanes, xv - @as(Vec, @splat(m)));
            @as(*align(1) Vec, @ptrCast(out.ptr + off + c)).* = ev;
            acc_v += ev;
        }
        var acc: f32 = @reduce(.Add, acc_v);
        while (c < n_tile) : (c += 1) {
            const e: f32 = expFast(in[off + c] - m);
            out[off + c] = e;
            acc += e;
        }
        sum_buf[r] += acc;
    }
}

pub fn normalizeF32(out_view: types.BufferViewMut, sum_buf: []const f32, rank: usize) void {
    var out: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, out_view.bytes);
    const lanes: usize = comptime simd.lanesF32();
    const Vec = @Vector(lanes, f32);

    if (rank == 1) {
        const inv: f32 = 1.0 / sum_buf[0];
        const inv_v: Vec = @splat(inv);
        var i: usize = 0;
        const vec_end: usize = out.len - (out.len % lanes);
        while (i < vec_end) : (i += lanes) {
            const v: Vec = @as(*align(1) const Vec, @ptrCast(out.ptr + i)).*;
            @as(*align(1) Vec, @ptrCast(out.ptr + i)).* = v * inv_v;
        }
        while (i < out.len) : (i += 1) out[i] *= inv;
        return;
    }

    const m_tile: usize = out_view.layout.shape[0];
    const n_tile: usize = out_view.layout.shape[1];
    var r: usize = 0;
    while (r < m_tile) : (r += 1) {
        const inv: f32 = 1.0 / sum_buf[r];
        const inv_v: Vec = @splat(inv);
        const off: usize = r * n_tile;
        var c: usize = 0;
        const vec_end: usize = n_tile - (n_tile % lanes);
        while (c < vec_end) : (c += lanes) {
            const v: Vec = @as(*align(1) const Vec, @ptrCast(out.ptr + off + c)).*;
            @as(*align(1) Vec, @ptrCast(out.ptr + off + c)).* = v * inv_v;
        }
        while (c < n_tile) : (c += 1) out[off + c] *= inv;
    }
}
