const std = @import("std");
const backend_mod = @import("../backend.zig");
const types = @import("../types.zig");
const utils = @import("../utils.zig");
const elemwise = @import("kernels/elemwise.zig");
const matmul_k = @import("kernels/matmul.zig");
const reduce_k = @import("kernels/reduce.zig");
const simd = @import("kernels/simd.zig");
const thread_pool = @import("../../runtime/thread_pool.zig");
const executable = @import("../../runtime/executable.zig");
const tensor_store = @import("../../runtime/tensor_store.zig");

const Backend = backend_mod.Backend;
const BackendKind = types.BackendKind;
const BackendCaps = types.BackendCaps;
const BackendError = types.BackendError;
const BufferViewConst = types.BufferViewConst;
const BufferViewMut = types.BufferViewMut;
const ElemwiseBinaryOp = types.ElemwiseBinaryOp;
const MatMulParams = types.MatMulParams;
const DType = types.DType;
const Layout = types.Layout;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

/// CPU Backend implementation.
/// Owns threading strategy (v0: single-threaded, later: thread pool).
pub const CpuBackend = struct {
    allocator: std.mem.Allocator,

    pool: ?thread_pool.ThreadPool = null,
    thread_count: usize = 1,

    // Per-thread scratch for reductions (sum/mean). Size == thread_count.
    reduce_scratch_f32: []f32 = &[_]f32{},

    // Future: thread pool handle, SIMD feature flags, scratch buffers

    const Self = @This();

    pub const Options = struct {
        /// Total threads to use including the calling thread.
        /// Set to 1 to disable parallelism (default).
        thread_count: usize = 1,
    };

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{ .allocator = allocator, .pool = null, .thread_count = 1, .reduce_scratch_f32 = &[_]f32{} };
    }

    pub fn initWithOptions(allocator: std.mem.Allocator, opts: Options) !Self {
        if (opts.thread_count == 0) return error.InvalidArgument;

        var self: Self = .{ .allocator = allocator, .pool = null, .thread_count = 1, .reduce_scratch_f32 = &[_]f32{} };
        if (opts.thread_count > 1) {
            const p = try thread_pool.ThreadPool.init(allocator, .{ .thread_count = opts.thread_count });
            self.pool = p;
            self.thread_count = opts.thread_count;

            // Allocate reduction scratch once. No allocations during op execution.
            self.reduce_scratch_f32 = try allocator.alloc(f32, opts.thread_count);
        }
        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.pool) |*p| {
            p.deinit();
            self.pool = null;
        }

        if (self.reduce_scratch_f32.len != 0) {
            self.allocator.free(self.reduce_scratch_f32);
            self.reduce_scratch_f32 = &[_]f32{};
        }
        self.thread_count = 1;
    }

    pub fn backend(self: *Self) Backend {
        return .{
            .ctx = @ptrCast(self),
            .vtable = &vtable,
        };
    }

    const vtable = Backend.VTable{
        .kind = kindImpl,
        .name = nameImpl,
        .caps = capsImpl,
        .deinit = deinitImpl,
        .elemwiseBinary = elemwiseBinaryImpl,
        .broadcastLastDimBinary = broadcastLastDimBinaryImpl,
        .relu = reluImpl,
        .reduce = reduceImpl,
        .matmul = matmulImpl,
        .copy = copyImpl,
        .executeProgram = executeProgramImpl,
    };

    fn scalarElemBytes(dtype: DType) ExecuteProgramError!usize {
        return switch (dtype) {
            .f32 => 4,
            .f16 => 2,
            else => return BackendError.InvalidArgument,
        };
    }

    fn tileElemCount(meta: tensor_store.TensorMeta) usize {
        return switch (meta.rank) {
            0 => 1,
            1 => meta.tile_shape[0],
            else => meta.tile_shape[0] * meta.tile_shape[1],
        };
    }

    fn tileByteSize(meta: tensor_store.TensorMeta) usize {
        const elems: usize = tileElemCount(meta);
        return switch (meta.dtype) {
            .f32 => elems * 4,
            .f16 => elems * 2,
            .i8 => elems,
            .q4_0, .q8_0 => blk: {
                const info = meta.dtype.info();
                const blocks = std.math.divCeil(usize, elems, info.block_elems) catch return 0;
                break :blk blocks * info.block_bytes;
            },
        };
    }

    /// Heuristic: parallelize tiled ops when either (a) we have enough tiles to hand out
    /// work, or (b) the total byte volume is large enough to amortize scheduling overhead.
    fn shouldParallelTiles(thread_count: usize, tile_total: usize, tile_bytes: usize, min_total_bytes: usize) bool {
        if (thread_count <= 1) return false;
        if (tile_total == 0) return false;
        const total_bytes: usize = tile_bytes * tile_total;
        if (total_bytes >= min_total_bytes) return true;
        // Also allow when we have at least two tiles; matmul-like ops are compute heavy per tile.
        return tile_total >= 2;
    }

    fn elemCountFromTileView(view: anytype) usize {
        // view.layout.rank is u8; shape slice length matches rank.
        if (view.layout.rank == 1) return view.layout.shape[0];
        return view.layout.shape[0] * view.layout.shape[1];
    }

    fn reduceAllScalar(self: *Self, op: types.ReduceOp, out_id: tensor_store.TensorId, a_id: tensor_store.TensorId, store: tensor_store.TensorStore) ExecuteProgramError!void {
        const out_meta = try store.meta(out_id);
        const a_meta = try store.meta(a_id);

        // Compile-time validation guarantees scalar supported and dtype matches.
        if (out_meta.dtype != a_meta.dtype) return BackendError.InvalidArgument;

        const total_elems_u64: u64 = switch (a_meta.rank) {
            1 => @as(u64, @intCast(a_meta.shape[0])),
            2 => @as(u64, @intCast(a_meta.shape[0])) * @as(u64, @intCast(a_meta.shape[1])),
            else => return BackendError.InvalidArgument,
        };

        var sum_f64: f64 = 0.0;

        // Parallelize over tiles at program level. Errors are not expected in v0.
        if (self.pool) |*p| {
            const tile_total: usize = a_meta.tile_counts[0] * a_meta.tile_counts[1];

            // Reuse the existing f32 scratch as a partial accumulator buffer when possible.
            // (We store partial sums as f32 then widen; good enough for perf benchmark paths.)
            if (self.reduce_scratch_f32.len >= self.thread_count and self.thread_count > 1 and tile_total >= 2) {
                @memset(self.reduce_scratch_f32[0..self.thread_count], 0.0);

                const Task = struct {
                    store: tensor_store.TensorStore,
                    a_id: tensor_store.TensorId,
                    a_meta: tensor_store.TensorMeta,
                    dtype: DType,
                    partials: []f32,

                    fn runTiles(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
                        const t: *@This() = @ptrCast(@alignCast(ctx_any));
                        if (start >= end) return;
                        if (tid >= t.partials.len) return;

                        const tc1: usize = t.a_meta.tile_counts[1];
                        var acc: f32 = 0.0;
                        var i: usize = start;
                        while (i < end) : (i += 1) {
                            const ti0: usize = i / tc1;
                            const ti1: usize = i - ti0 * tc1;
                            const tile = t.store.acquireTileConst(t.a_id, ti0, ti1) catch continue;
                            defer t.store.releaseConst(tile.token);
                            const v = tile.bufferView();
                            const n: usize = if (v.layout.rank == 1) v.layout.shape[0] else (v.layout.shape[0] * v.layout.shape[1]);
                            const part: f32 = switch (t.dtype) {
                                .f32 => reduce_k.sumF32Range(v.bytes, 0, n) catch 0.0,
                                .f16 => reduce_k.sumF16RangeToF32(v.bytes, 0, n) catch 0.0,
                                else => 0.0,
                            };
                            acc += part;
                        }
                        t.partials[tid] += acc;
                    }
                };

                var task: Task = .{ .store = store, .a_id = a_id, .a_meta = a_meta, .dtype = a_meta.dtype, .partials = self.reduce_scratch_f32[0..self.thread_count] };
                // Grain size: aim for at least 256KiB of work per task to amortize overhead.
                const bytes_per_tile = tileByteSize(a_meta);
                var grain: usize = if (bytes_per_tile == 0) 32 else @max(@as(usize, 1), (256 * 1024) / bytes_per_tile);
                if (grain > tile_total) grain = tile_total;

                p.parallelForAny(@ptrCast(&task), tile_total, grain, Task.runTiles);

                var i: usize = 0;
                while (i < self.thread_count) : (i += 1) {
                    sum_f64 += @as(f64, self.reduce_scratch_f32[i]);
                }
            } else {
                // Fallback to sequential reduce below.
                sum_f64 = 0.0;
            }
        }

        if (sum_f64 == 0.0) {
            // Sequential fallback (also used when thread_count==1).
            var ti0: usize = 0;
            while (ti0 < a_meta.tile_counts[0]) : (ti0 += 1) {
                var ti1: usize = 0;
                while (ti1 < a_meta.tile_counts[1]) : (ti1 += 1) {
                    const tile = try store.acquireTileConst(a_id, ti0, ti1);
                    defer store.releaseConst(tile.token);
                    const v = tile.bufferView();
                    const n: usize = elemCountFromTileView(v);
                    const part: f32 = switch (a_meta.dtype) {
                        .f32 => try reduce_k.sumF32Range(v.bytes, 0, n),
                        .f16 => try reduce_k.sumF16RangeToF32(v.bytes, 0, n),
                        else => return BackendError.InvalidArgument,
                    };
                    sum_f64 += @as(f64, part);
                }
            }
        }

        var result_f64: f64 = sum_f64;
        if (op == .mean) {
            if (total_elems_u64 == 0) return BackendError.InvalidArgument;
            result_f64 /= @as(f64, @floatFromInt(total_elems_u64));
        }

        var out_tile = try store.acquireTileMut(out_id, 0, 0);
        defer store.releaseMut(out_tile.token);
        const out_view = out_tile.bufferView();

        switch (out_meta.dtype) {
            .f32 => @as(*align(1) f32, @ptrCast(out_view.bytes.ptr)).* = @floatCast(result_f64),
            .f16 => @as(*align(1) f16, @ptrCast(out_view.bytes.ptr)).* = @floatCast(@as(f32, @floatCast(result_f64))),
            else => return BackendError.InvalidArgument,
        }
    }

    const ScalarTileCacheConst = struct {
        valid: bool = false,
        ti0: usize = 0,
        ti1: usize = 0,
        tile: tensor_store.TileRefConst = undefined,
    };

    const ScalarTileCacheMut = struct {
        valid: bool = false,
        ti0: usize = 0,
        ti1: usize = 0,
        tile: tensor_store.TileRefMut = undefined,
    };

    fn ensureConstTile(cache: *ScalarTileCacheConst, store: tensor_store.TensorStore, id: tensor_store.TensorId, ti0: usize, ti1: usize) ExecuteProgramError!void {
        if (cache.valid and cache.ti0 == ti0 and cache.ti1 == ti1) return;
        if (cache.valid) store.releaseConst(cache.tile.token);
        cache.tile = try store.acquireTileConst(id, ti0, ti1);
        cache.ti0 = ti0;
        cache.ti1 = ti1;
        cache.valid = true;
    }

    fn ensureMutTile(cache: *ScalarTileCacheMut, store: tensor_store.TensorStore, id: tensor_store.TensorId, ti0: usize, ti1: usize) ExecuteProgramError!void {
        if (cache.valid and cache.ti0 == ti0 and cache.ti1 == ti1) return;
        if (cache.valid) store.releaseMut(cache.tile.token);
        cache.tile = try store.acquireTileMut(id, ti0, ti1);
        cache.ti0 = ti0;
        cache.ti1 = ti1;
        cache.valid = true;
    }

    fn readScalarBytesAt(
        store: tensor_store.TensorStore,
        meta: tensor_store.TensorMeta,
        id: tensor_store.TensorId,
        elem_bytes: usize,
        idx0: usize,
        idx1: usize,
        cache: *ScalarTileCacheConst,
        out: []u8,
    ) ExecuteProgramError!void {
        if (meta.rank == 1) {
            const ti: usize = idx0 / meta.tile_shape[0];
            const in_i: usize = idx0 - ti * meta.tile_shape[0];
            try ensureConstTile(cache, store, id, ti, 0);
            const off: usize = in_i * elem_bytes;
            @memcpy(out, cache.tile.bytes[off .. off + elem_bytes]);
            return;
        }

        const ti0: usize = idx0 / meta.tile_shape[0];
        const ti1: usize = idx1 / meta.tile_shape[1];
        const in0: usize = idx0 - ti0 * meta.tile_shape[0];
        const in1: usize = idx1 - ti1 * meta.tile_shape[1];

        try ensureConstTile(cache, store, id, ti0, ti1);
        const n_tile: usize = cache.tile.shape_mem[1];
        const off: usize = (in0 * n_tile + in1) * elem_bytes;
        @memcpy(out, cache.tile.bytes[off .. off + elem_bytes]);
    }

    fn writeScalarBytesAt(
        store: tensor_store.TensorStore,
        meta: tensor_store.TensorMeta,
        id: tensor_store.TensorId,
        elem_bytes: usize,
        idx0: usize,
        idx1: usize,
        cache: *ScalarTileCacheMut,
        bytes: []const u8,
    ) ExecuteProgramError!void {
        if (meta.rank == 1) {
            const ti: usize = idx0 / meta.tile_shape[0];
            const in_i: usize = idx0 - ti * meta.tile_shape[0];
            try ensureMutTile(cache, store, id, ti, 0);
            const off: usize = in_i * elem_bytes;
            @memcpy(cache.tile.bytes[off .. off + elem_bytes], bytes);
            return;
        }

        const ti0: usize = idx0 / meta.tile_shape[0];
        const ti1: usize = idx1 / meta.tile_shape[1];
        const in0: usize = idx0 - ti0 * meta.tile_shape[0];
        const in1: usize = idx1 - ti1 * meta.tile_shape[1];

        try ensureMutTile(cache, store, id, ti0, ti1);
        const n_tile: usize = cache.tile.shape_mem[1];
        const off: usize = (in0 * n_tile + in1) * elem_bytes;
        @memcpy(cache.tile.bytes[off .. off + elem_bytes], bytes);
    }

    fn retileCopyScalar(store: tensor_store.TensorStore, dst_id: tensor_store.TensorId, src_id: tensor_store.TensorId) ExecuteProgramError!void {
        const dst_meta = try store.meta(dst_id);
        const src_meta = try store.meta(src_id);
        const elem_bytes: usize = try scalarElemBytes(dst_meta.dtype);

        var src_cache: ScalarTileCacheConst = .{};
        defer if (src_cache.valid) store.releaseConst(src_cache.tile.token);
        var dst_cache: ScalarTileCacheMut = .{};
        defer if (dst_cache.valid) store.releaseMut(dst_cache.tile.token);

        var tmp: [4]u8 = .{ 0, 0, 0, 0 };
        const buf: []u8 = tmp[0..elem_bytes];

        if (dst_meta.rank == 1) {
            var i: usize = 0;
            while (i < dst_meta.shape[0]) : (i += 1) {
                try readScalarBytesAt(store, src_meta, src_id, elem_bytes, i, 0, &src_cache, buf);
                try writeScalarBytesAt(store, dst_meta, dst_id, elem_bytes, i, 0, &dst_cache, buf);
            }
            return;
        }

        var r0: usize = 0;
        while (r0 < dst_meta.shape[0]) : (r0 += 1) {
            var c0: usize = 0;
            while (c0 < dst_meta.shape[1]) : (c0 += 1) {
                try readScalarBytesAt(store, src_meta, src_id, elem_bytes, r0, c0, &src_cache, buf);
                try writeScalarBytesAt(store, dst_meta, dst_id, elem_bytes, r0, c0, &dst_cache, buf);
            }
        }
    }

    fn reshapeCopyScalar(store: tensor_store.TensorStore, dst_id: tensor_store.TensorId, src_id: tensor_store.TensorId) ExecuteProgramError!void {
        const dst_meta = try store.meta(dst_id);
        const src_meta = try store.meta(src_id);

        const elem_bytes: usize = try scalarElemBytes(dst_meta.dtype);
        const dst_elems: usize = if (dst_meta.rank == 1) dst_meta.shape[0] else (dst_meta.shape[0] * dst_meta.shape[1]);

        var src_cache: ScalarTileCacheConst = .{};
        defer if (src_cache.valid) store.releaseConst(src_cache.tile.token);
        var dst_cache: ScalarTileCacheMut = .{};
        defer if (dst_cache.valid) store.releaseMut(dst_cache.tile.token);

        var tmp: [4]u8 = .{ 0, 0, 0, 0 };
        const buf: []u8 = tmp[0..elem_bytes];

        var lin: usize = 0;
        while (lin < dst_elems) : (lin += 1) {
            const src_i0: usize = if (src_meta.rank == 1) lin else (lin / src_meta.shape[1]);
            const src_i1: usize = if (src_meta.rank == 1) 0 else (lin - src_i0 * src_meta.shape[1]);

            const dst_i0: usize = if (dst_meta.rank == 1) lin else (lin / dst_meta.shape[1]);
            const dst_i1: usize = if (dst_meta.rank == 1) 0 else (lin - dst_i0 * dst_meta.shape[1]);

            try readScalarBytesAt(store, src_meta, src_id, elem_bytes, src_i0, src_i1, &src_cache, buf);
            try writeScalarBytesAt(store, dst_meta, dst_id, elem_bytes, dst_i0, dst_i1, &dst_cache, buf);
        }
    }

    fn transpose2DCopyScalar(store: tensor_store.TensorStore, dst_id: tensor_store.TensorId, src_id: tensor_store.TensorId) ExecuteProgramError!void {
        const dst_meta = try store.meta(dst_id);
        const src_meta = try store.meta(src_id);

        const elem_bytes: usize = try scalarElemBytes(dst_meta.dtype);
        var src_cache: ScalarTileCacheConst = .{};
        defer if (src_cache.valid) store.releaseConst(src_cache.tile.token);
        var dst_cache: ScalarTileCacheMut = .{};
        defer if (dst_cache.valid) store.releaseMut(dst_cache.tile.token);

        var tmp: [4]u8 = .{ 0, 0, 0, 0 };
        const buf: []u8 = tmp[0..elem_bytes];

        var r0: usize = 0;
        while (r0 < dst_meta.shape[0]) : (r0 += 1) {
            var c0: usize = 0;
            while (c0 < dst_meta.shape[1]) : (c0 += 1) {
                try readScalarBytesAt(store, src_meta, src_id, elem_bytes, c0, r0, &src_cache, buf);
                try writeScalarBytesAt(store, dst_meta, dst_id, elem_bytes, r0, c0, &dst_cache, buf);
            }
        }
    }

    fn slice2DCopyScalar(store: tensor_store.TensorStore, dst_id: tensor_store.TensorId, src_id: tensor_store.TensorId, start0: usize, start1: usize) ExecuteProgramError!void {
        const dst_meta = try store.meta(dst_id);
        const src_meta = try store.meta(src_id);

        const elem_bytes: usize = try scalarElemBytes(dst_meta.dtype);
        var src_cache: ScalarTileCacheConst = .{};
        defer if (src_cache.valid) store.releaseConst(src_cache.tile.token);
        var dst_cache: ScalarTileCacheMut = .{};
        defer if (dst_cache.valid) store.releaseMut(dst_cache.tile.token);

        var tmp: [4]u8 = .{ 0, 0, 0, 0 };
        const buf: []u8 = tmp[0..elem_bytes];

        var r0: usize = 0;
        while (r0 < dst_meta.shape[0]) : (r0 += 1) {
            var c0: usize = 0;
            while (c0 < dst_meta.shape[1]) : (c0 += 1) {
                try readScalarBytesAt(store, src_meta, src_id, elem_bytes, start0 + r0, start1 + c0, &src_cache, buf);
                try writeScalarBytesAt(store, dst_meta, dst_id, elem_bytes, r0, c0, &dst_cache, buf);
            }
        }
    }

    fn executeProgramImpl(ctx: *anyopaque, prog: *const executable.ExecutableProgram, store: tensor_store.TensorStore) ExecuteProgramError!void {
        const self: *Self = @ptrCast(@alignCast(ctx));

        for (prog.steps) |step| {
            switch (step) {
                .MatMulTiled => |s| {
                    const c_meta = try store.meta(s.c);
                    const a_meta = try store.meta(s.a);

                    const tile_total: usize = c_meta.tile_counts[0] * c_meta.tile_counts[1];
                    if (self.pool) |*p| {
                        // Dynamic splitting: sub-divide tiles along M if we have low tile count relative to threads.
                        // Heuristic: aim for over-subscription (e.g. 4x threads) to balance load.
                        const target_tasks = self.thread_count * 4;
                        var split: usize = 1;
                        if (tile_total < target_tasks and tile_total > 0) {
                            split = (target_tasks + tile_total - 1) / tile_total;
                            // Clamp: ensure at least 32 rows per task to keep kernel efficiency high.
                            const m_tile_max = c_meta.tile_shape[0];
                            const max_split = @max(1, m_tile_max / 32);
                            split = @min(split, max_split);
                        }

                        const total_work: usize = tile_total * split;

                        // MatMul tiles are compute-heavy; parallelize if we have enough work.
                        if (self.thread_count > 1 and total_work >= 2) {
                            const Task = struct {
                                store: tensor_store.TensorStore,
                                c_meta: tensor_store.TensorMeta,
                                a_meta: tensor_store.TensorMeta,
                                s: @TypeOf(s),
                                split: usize,

                                stop: std.atomic.Value(bool) = .init(false),
                                err_mutex: std.Thread.Mutex = .{},
                                err_any: ?anyerror = null,

                                fn fail(t: *@This(), err: anyerror) void {
                                    if (t.stop.swap(true, .acq_rel)) return;
                                    t.err_mutex.lock();
                                    defer t.err_mutex.unlock();
                                    if (t.err_any == null) t.err_any = err;
                                }

                                fn runTiles(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
                                    _ = tid;
                                    const t: *@This() = @ptrCast(@alignCast(ctx_any));
                                    if (start >= end) return;

                                    if (t.stop.load(.acquire)) return;

                                    //const tc1: usize = t.c_meta.tile_counts[1];
                                    const tc0: usize = t.c_meta.tile_counts[0];
                                    const split_factor = t.split;

                                    var i: usize = start;
                                    while (i < end) : (i += 1) {
                                        if (t.stop.load(.acquire)) return;

                                        const tile_idx = i / split_factor;
                                        const split_idx = i % split_factor;

                                        // Column-major iteration: ti_m is inner, ti_n is outer.
                                        // This helps keep B tiles in L3 cache across M-steps.
                                        const ti_n: usize = tile_idx / tc0;
                                        const ti_m: usize = tile_idx - ti_n * tc0;

                                        var c_tile = t.store.acquireTileMut(t.s.c, ti_m, ti_n) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                        defer t.store.releaseMut(c_tile.token);
                                        const c_view0 = c_tile.bufferView();
                                        const m_tile: usize = c_view0.layout.shape[0];
                                        const n_tile: usize = c_view0.layout.shape[1];

                                        const rows_per_split = m_tile / split_factor;
                                        const r_start = split_idx * rows_per_split;
                                        const r_end = if (split_idx == split_factor - 1) m_tile else (split_idx + 1) * rows_per_split;
                                        if (r_start >= r_end) continue;
                                        const m_sub = r_end - r_start;

                                        const a_dtype = (t.store.meta(t.s.a) catch |e| {
                                            t.fail(e);
                                            return;
                                        }).dtype;
                                        const b_dtype = (t.store.meta(t.s.b) catch |e| {
                                            t.fail(e);
                                            return;
                                        }).dtype;
                                        const c_dtype = t.c_meta.dtype;
                                        const a_info = a_dtype.info();
                                        const c_info = c_dtype.info();

                                        var ti_k: usize = 0;
                                        while (ti_k < t.a_meta.tile_counts[1]) : (ti_k += 1) {
                                            if (ti_k + 1 < t.a_meta.tile_counts[1]) {
                                                t.store.prefetch(t.s.a, ti_m, ti_k + 1);
                                                t.store.prefetch(t.s.b, ti_k + 1, ti_n);
                                            }

                                            const a_tile = t.store.acquireTileConst(t.s.a, ti_m, ti_k) catch |e| {
                                                t.fail(e);
                                                return;
                                            };
                                            defer t.store.releaseConst(a_tile.token);
                                            const b_tile = t.store.acquireTileConst(t.s.b, ti_k, ti_n) catch |e| {
                                                t.fail(e);
                                                return;
                                            };
                                            defer t.store.releaseConst(b_tile.token);

                                            const a_view = a_tile.bufferView();
                                            const b_view = b_tile.bufferView();

                                            const k_tile: usize = a_view.layout.shape[1];
                                            const beta_tile: f32 = if (ti_k == 0) t.s.beta else 1.0;
                                            const params: MatMulParams = .{ .m = m_sub, .n = n_tile, .k = k_tile, .alpha = t.s.alpha, .beta = beta_tile };

                                            const a_row_bytes = k_tile * a_info.block_bytes;
                                            const c_row_bytes = n_tile * c_info.block_bytes;
                                            const a_off = r_start * a_row_bytes;
                                            const c_off = r_start * c_row_bytes;
                                            const a_sub_bytes = a_view.bytes[a_off..];
                                            const c_sub_bytes = c_view0.bytes[c_off..];

                                            // Single-threaded kernels (outer tiling provides parallelism).
                                            switch (b_dtype) {
                                                .f32 => {
                                                    if (a_dtype != .f32 or c_dtype != .f32) {
                                                        t.fail(BackendError.InvalidArgument);
                                                        return;
                                                    }
                                                    matmul_k.matmulF32(params, c_sub_bytes, a_sub_bytes, b_view.bytes) catch |e| {
                                                        t.fail(e);
                                                        return;
                                                    };
                                                },
                                                .f16 => {
                                                    if (a_dtype != .f16) {
                                                        t.fail(BackendError.InvalidArgument);
                                                        return;
                                                    }
                                                    if (c_dtype == .f32) {
                                                        matmul_k.matmulF16ToF32(params, c_sub_bytes, a_sub_bytes, b_view.bytes) catch |e| {
                                                            t.fail(e);
                                                            return;
                                                        };
                                                    } else if (c_dtype == .f16) {
                                                        matmul_k.matmulF16(params, c_sub_bytes, a_sub_bytes, b_view.bytes) catch |e| {
                                                            t.fail(e);
                                                            return;
                                                        };
                                                    } else {
                                                        t.fail(BackendError.InvalidArgument);
                                                        return;
                                                    }
                                                },
                                                .q4_0 => {
                                                    if (a_dtype != .f32 or c_dtype != .f32) {
                                                        t.fail(BackendError.InvalidArgument);
                                                        return;
                                                    }
                                                    matmul_k.matmulQ4_0(params, c_sub_bytes, a_sub_bytes, b_view.bytes) catch |e| {
                                                        t.fail(e);
                                                        return;
                                                    };
                                                },
                                                .q8_0 => {
                                                    if (a_dtype != .f32 or c_dtype != .f32) {
                                                        t.fail(BackendError.InvalidArgument);
                                                        return;
                                                    }
                                                    matmul_k.matmulQ8_0(params, c_sub_bytes, a_sub_bytes, b_view.bytes) catch |e| {
                                                        t.fail(e);
                                                        return;
                                                    };
                                                },
                                                else => {
                                                    t.fail(BackendError.Unsupported);
                                                    return;
                                                },
                                            }
                                        }
                                    }
                                }
                            };

                            var task: Task = .{ .store = store, .c_meta = c_meta, .a_meta = a_meta, .s = s, .split = split };
                            // One output tile (or sub-split) per chunk keeps work nicely balanced.
                            p.parallelForAny(@ptrCast(&task), total_work, 1, Task.runTiles);
                            if (task.err_any) |e| return @errorCast(e);
                            continue;
                        }
                    }

                    // Sequential fallback.
                    var ti_m: usize = 0;
                    while (ti_m < c_meta.tile_counts[0]) : (ti_m += 1) {
                        var ti_n: usize = 0;
                        while (ti_n < c_meta.tile_counts[1]) : (ti_n += 1) {
                            var c_tile = try store.acquireTileMut(s.c, ti_m, ti_n);
                            defer store.releaseMut(c_tile.token);

                            const c_view0 = c_tile.bufferView();
                            const m_tile: usize = c_view0.layout.shape[0];
                            const n_tile: usize = c_view0.layout.shape[1];

                            var ti_k: usize = 0;
                            while (ti_k < a_meta.tile_counts[1]) : (ti_k += 1) {
                                const a_tile = try store.acquireTileConst(s.a, ti_m, ti_k);
                                defer store.releaseConst(a_tile.token);
                                const b_tile = try store.acquireTileConst(s.b, ti_k, ti_n);
                                defer store.releaseConst(b_tile.token);

                                const a_view = a_tile.bufferView();
                                const b_view = b_tile.bufferView();
                                const c_view = c_tile.bufferView();

                                const a_dtype: DType = a_view.dtype;
                                const b_dtype: DType = b_view.dtype;
                                const c_dtype: DType = c_view.dtype;

                                const k_tile: usize = a_view.layout.shape[1];
                                const beta_tile: f32 = if (ti_k == 0) s.beta else 1.0;
                                const params: MatMulParams = .{ .m = m_tile, .n = n_tile, .k = k_tile, .alpha = s.alpha, .beta = beta_tile };

                                switch (b_dtype) {
                                    .f32 => {
                                        if (a_dtype != .f32 or c_dtype != .f32) return BackendError.InvalidArgument;
                                        try matmul_k.matmulF32(params, c_view.bytes, a_view.bytes, b_view.bytes);
                                    },
                                    .f16 => {
                                        if (a_dtype != .f16) return BackendError.InvalidArgument;
                                        if (c_dtype == .f32) {
                                            try matmul_k.matmulF16ToF32(params, c_view.bytes, a_view.bytes, b_view.bytes);
                                        } else if (c_dtype == .f16) {
                                            try matmul_k.matmulF16(params, c_view.bytes, a_view.bytes, b_view.bytes);
                                        } else {
                                            return BackendError.InvalidArgument;
                                        }
                                    },
                                    .q4_0 => {
                                        if (a_dtype != .f32 or c_dtype != .f32) return BackendError.InvalidArgument;
                                        try matmul_k.matmulQ4_0(params, c_view.bytes, a_view.bytes, b_view.bytes);
                                    },
                                    .q8_0 => {
                                        if (a_dtype != .f32 or c_dtype != .f32) return BackendError.InvalidArgument;
                                        try matmul_k.matmulQ8_0(params, c_view.bytes, a_view.bytes, b_view.bytes);
                                    },
                                    else => return BackendError.Unsupported,
                                }
                            }
                        }
                    }
                },

                .ElemwiseBinaryTiled => |s| {
                    const out_meta = try store.meta(s.out);
                    const tile_total: usize = out_meta.tile_counts[0] * out_meta.tile_counts[1];
                    const tile_bytes: usize = tileByteSize(out_meta);
                    const min_total_bytes: usize = 256 * 1024; // aim for at least 256KiB of work

                    if (self.pool) |*p| {
                        if (shouldParallelTiles(self.thread_count, tile_total, tile_bytes, min_total_bytes)) {
                            const Task = struct {
                                store: tensor_store.TensorStore,
                                out_meta: tensor_store.TensorMeta,
                                op: ElemwiseBinaryOp,
                                out: tensor_store.TensorId,
                                a: tensor_store.TensorId,
                                b: tensor_store.TensorId,

                                stop: std.atomic.Value(bool) = .init(false),
                                err_mutex: std.Thread.Mutex = .{},
                                err_any: ?anyerror = null,

                                fn fail(t: *@This(), err: anyerror) void {
                                    if (t.stop.swap(true, .acq_rel)) return;
                                    t.err_mutex.lock();
                                    defer t.err_mutex.unlock();
                                    if (t.err_any == null) t.err_any = err;
                                }

                                fn runTiles(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
                                    _ = tid;
                                    const t: *@This() = @ptrCast(@alignCast(ctx_any));
                                    if (start >= end) return;

                                    if (t.stop.load(.acquire)) return;

                                    const tc1: usize = t.out_meta.tile_counts[1];
                                    var i: usize = start;
                                    while (i < end) : (i += 1) {
                                        if (t.stop.load(.acquire)) return;
                                        const ti0: usize = i / tc1;
                                        const ti1: usize = i - ti0 * tc1;

                                        if (i + 1 < end) {
                                            const nti0 = (i + 1) / tc1;
                                            const nti1 = (i + 1) - nti0 * tc1;
                                            t.store.prefetch(t.a, nti0, nti1);
                                            t.store.prefetch(t.b, nti0, nti1);
                                        }

                                        var out_tile = t.store.acquireTileMut(t.out, ti0, ti1) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                        defer t.store.releaseMut(out_tile.token);
                                        const a_tile = t.store.acquireTileConst(t.a, ti0, ti1) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                        defer t.store.releaseConst(a_tile.token);
                                        const b_tile = t.store.acquireTileConst(t.b, ti0, ti1) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                        defer t.store.releaseConst(b_tile.token);

                                        const out_view = out_tile.bufferView();
                                        const a_view = a_tile.bufferView();
                                        const b_view = b_tile.bufferView();

                                        const n: usize = if (out_view.layout.rank == 1) out_view.layout.shape[0] else (out_view.layout.shape[0] * out_view.layout.shape[1]);
                                        switch (out_view.dtype) {
                                            .f32 => elemwise.elemwiseBinaryF32(t.op, out_view.bytes, a_view.bytes, b_view.bytes, n) catch |e| {
                                                t.fail(e);
                                                return;
                                            },
                                            .f16 => elemwise.elemwiseBinaryF16(t.op, out_view.bytes, a_view.bytes, b_view.bytes, n) catch |e| {
                                                t.fail(e);
                                                return;
                                            },
                                            else => {
                                                t.fail(BackendError.InvalidArgument);
                                                return;
                                            },
                                        }
                                    }
                                }
                            };

                            var task: Task = .{ .store = store, .out_meta = out_meta, .op = s.op, .out = s.out, .a = s.a, .b = s.b };
                            // Grain based on bytes-per-tile: target ~256KiB per chunk.
                            var grain: usize = if (tile_bytes == 0) 32 else @max(@as(usize, 1), min_total_bytes / tile_bytes);
                            if (grain > tile_total) grain = tile_total;
                            p.parallelForAny(@ptrCast(&task), tile_total, grain, Task.runTiles);
                            if (task.err_any) |e| return @errorCast(e);
                            continue;
                        }
                    }

                    // Sequential fallback.
                    var ti0: usize = 0;
                    while (ti0 < out_meta.tile_counts[0]) : (ti0 += 1) {
                        var ti1: usize = 0;
                        while (ti1 < out_meta.tile_counts[1]) : (ti1 += 1) {
                            var out_tile = try store.acquireTileMut(s.out, ti0, ti1);
                            defer store.releaseMut(out_tile.token);
                            const a_tile = try store.acquireTileConst(s.a, ti0, ti1);
                            defer store.releaseConst(a_tile.token);
                            const b_tile = try store.acquireTileConst(s.b, ti0, ti1);
                            defer store.releaseConst(b_tile.token);

                            const out_view = out_tile.bufferView();
                            const a_view = a_tile.bufferView();
                            const b_view = b_tile.bufferView();
                            const n: usize = elemCountFromTileView(out_view);
                            switch (out_view.dtype) {
                                .f32 => try elemwise.elemwiseBinaryF32(s.op, out_view.bytes, a_view.bytes, b_view.bytes, n),
                                .f16 => try elemwise.elemwiseBinaryF16(s.op, out_view.bytes, a_view.bytes, b_view.bytes, n),
                                else => return BackendError.InvalidArgument,
                            }
                        }
                    }
                },

                .BroadcastLastDimBinaryTiled => |s| {
                    const out_meta = try store.meta(s.out);
                    const tile_total: usize = out_meta.tile_counts[0] * out_meta.tile_counts[1];
                    const tile_bytes: usize = tileByteSize(out_meta);
                    const min_total_bytes: usize = 256 * 1024;

                    if (self.pool) |*p| {
                        if (shouldParallelTiles(self.thread_count, tile_total, tile_bytes, min_total_bytes)) {
                            const Task = struct {
                                store: tensor_store.TensorStore,
                                out_meta: tensor_store.TensorMeta,
                                op: ElemwiseBinaryOp,
                                out: tensor_store.TensorId,
                                a: tensor_store.TensorId,
                                b: tensor_store.TensorId,

                                stop: std.atomic.Value(bool) = .init(false),
                                err_mutex: std.Thread.Mutex = .{},
                                err_any: ?anyerror = null,

                                fn fail(t: *@This(), err: anyerror) void {
                                    if (t.stop.swap(true, .acq_rel)) return;
                                    t.err_mutex.lock();
                                    defer t.err_mutex.unlock();
                                    if (t.err_any == null) t.err_any = err;
                                }

                                fn runTiles(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
                                    _ = tid;
                                    const t: *@This() = @ptrCast(@alignCast(ctx_any));
                                    if (start >= end) return;

                                    if (t.stop.load(.acquire)) return;

                                    const tc1: usize = t.out_meta.tile_counts[1];
                                    var i: usize = start;
                                    while (i < end) : (i += 1) {
                                        if (t.stop.load(.acquire)) return;
                                        const ti0: usize = i / tc1;
                                        const ti1: usize = i - ti0 * tc1;

                                        var out_tile = t.store.acquireTileMut(t.out, ti0, ti1) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                        defer t.store.releaseMut(out_tile.token);
                                        const a_tile = t.store.acquireTileConst(t.a, ti0, ti1) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                        defer t.store.releaseConst(a_tile.token);
                                        const b_tile = t.store.acquireTileConst(t.b, ti1, 0) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                        defer t.store.releaseConst(b_tile.token);

                                        const out_view = out_tile.bufferView();
                                        const a_view = a_tile.bufferView();
                                        const b_view = b_tile.bufferView();
                                        const row_count: usize = if (out_view.layout.rank == 1) 0 else out_view.layout.shape[0];
                                        const col_count: usize = if (out_view.layout.rank == 1) out_view.layout.shape[0] else out_view.layout.shape[1];
                                        if (out_view.layout.rank != 2) {
                                            t.fail(BackendError.InvalidArgument);
                                            return;
                                        }

                                        switch (out_view.dtype) {
                                            .f32 => switch (t.op) {
                                                .add => elemwise.broadcastLastDimBinaryF32(.add, out_view.bytes, a_view.bytes, b_view.bytes, row_count, col_count) catch |e| {
                                                    t.fail(e);
                                                    return;
                                                },
                                                .sub => elemwise.broadcastLastDimBinaryF32(.sub, out_view.bytes, a_view.bytes, b_view.bytes, row_count, col_count) catch |e| {
                                                    t.fail(e);
                                                    return;
                                                },
                                                .mul => elemwise.broadcastLastDimBinaryF32(.mul, out_view.bytes, a_view.bytes, b_view.bytes, row_count, col_count) catch |e| {
                                                    t.fail(e);
                                                    return;
                                                },
                                                .div => elemwise.broadcastLastDimBinaryF32(.div, out_view.bytes, a_view.bytes, b_view.bytes, row_count, col_count) catch |e| {
                                                    t.fail(e);
                                                    return;
                                                },
                                            },
                                            .f16 => switch (t.op) {
                                                .add => elemwise.broadcastLastDimBinaryF16(.add, out_view.bytes, a_view.bytes, b_view.bytes, row_count, col_count) catch |e| {
                                                    t.fail(e);
                                                    return;
                                                },
                                                .sub => elemwise.broadcastLastDimBinaryF16(.sub, out_view.bytes, a_view.bytes, b_view.bytes, row_count, col_count) catch |e| {
                                                    t.fail(e);
                                                    return;
                                                },
                                                .mul => elemwise.broadcastLastDimBinaryF16(.mul, out_view.bytes, a_view.bytes, b_view.bytes, row_count, col_count) catch |e| {
                                                    t.fail(e);
                                                    return;
                                                },
                                                .div => elemwise.broadcastLastDimBinaryF16(.div, out_view.bytes, a_view.bytes, b_view.bytes, row_count, col_count) catch |e| {
                                                    t.fail(e);
                                                    return;
                                                },
                                            },
                                            else => {
                                                t.fail(BackendError.InvalidArgument);
                                                return;
                                            },
                                        }
                                    }
                                }
                            };

                            var task: Task = .{ .store = store, .out_meta = out_meta, .op = s.op, .out = s.out, .a = s.a, .b = s.b };
                            var grain: usize = if (tile_bytes == 0) 16 else @max(@as(usize, 1), min_total_bytes / tile_bytes);
                            if (grain > tile_total) grain = tile_total;
                            p.parallelForAny(@ptrCast(&task), tile_total, grain, Task.runTiles);
                            if (task.err_any) |e| return @errorCast(e);
                            continue;
                        }
                    }

                    // Sequential fallback.
                    var ti0: usize = 0;
                    while (ti0 < out_meta.tile_counts[0]) : (ti0 += 1) {
                        var ti1: usize = 0;
                        while (ti1 < out_meta.tile_counts[1]) : (ti1 += 1) {
                            var out_tile = try store.acquireTileMut(s.out, ti0, ti1);
                            defer store.releaseMut(out_tile.token);
                            const a_tile = try store.acquireTileConst(s.a, ti0, ti1);
                            defer store.releaseConst(a_tile.token);
                            const b_tile = try store.acquireTileConst(s.b, ti1, 0);
                            defer store.releaseConst(b_tile.token);

                            const out_view = out_tile.bufferView();
                            const a_view = a_tile.bufferView();
                            const b_view = b_tile.bufferView();
                            if (out_view.layout.rank != 2) return BackendError.InvalidArgument;
                            const row_count: usize = out_view.layout.shape[0];
                            const col_count: usize = out_view.layout.shape[1];

                            switch (out_view.dtype) {
                                .f32 => switch (s.op) {
                                    .add => try elemwise.broadcastLastDimBinaryF32(.add, out_view.bytes, a_view.bytes, b_view.bytes, row_count, col_count),
                                    .sub => try elemwise.broadcastLastDimBinaryF32(.sub, out_view.bytes, a_view.bytes, b_view.bytes, row_count, col_count),
                                    .mul => try elemwise.broadcastLastDimBinaryF32(.mul, out_view.bytes, a_view.bytes, b_view.bytes, row_count, col_count),
                                    .div => try elemwise.broadcastLastDimBinaryF32(.div, out_view.bytes, a_view.bytes, b_view.bytes, row_count, col_count),
                                },
                                .f16 => switch (s.op) {
                                    .add => try elemwise.broadcastLastDimBinaryF16(.add, out_view.bytes, a_view.bytes, b_view.bytes, row_count, col_count),
                                    .sub => try elemwise.broadcastLastDimBinaryF16(.sub, out_view.bytes, a_view.bytes, b_view.bytes, row_count, col_count),
                                    .mul => try elemwise.broadcastLastDimBinaryF16(.mul, out_view.bytes, a_view.bytes, b_view.bytes, row_count, col_count),
                                    .div => try elemwise.broadcastLastDimBinaryF16(.div, out_view.bytes, a_view.bytes, b_view.bytes, row_count, col_count),
                                },
                                else => return BackendError.InvalidArgument,
                            }
                        }
                    }
                },

                .ReluTiled => |s| {
                    const out_meta = try store.meta(s.out);
                    const tile_total: usize = out_meta.tile_counts[0] * out_meta.tile_counts[1];
                    const tile_bytes: usize = tileByteSize(out_meta);
                    const min_total_bytes: usize = 256 * 1024;
                    if (self.pool) |*p| {
                        if (shouldParallelTiles(self.thread_count, tile_total, tile_bytes, min_total_bytes)) {
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

                                fn runTiles(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
                                    _ = tid;
                                    const t: *@This() = @ptrCast(@alignCast(ctx_any));
                                    if (start >= end) return;
                                    if (t.stop.load(.acquire)) return;
                                    const tc1: usize = t.out_meta.tile_counts[1];
                                    var i: usize = start;
                                    while (i < end) : (i += 1) {
                                        if (t.stop.load(.acquire)) return;
                                        const ti0: usize = i / tc1;
                                        const ti1: usize = i - ti0 * tc1;

                                        var out_tile = t.store.acquireTileMut(t.out, ti0, ti1) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                        defer t.store.releaseMut(out_tile.token);
                                        const a_tile = t.store.acquireTileConst(t.a, ti0, ti1) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                        defer t.store.releaseConst(a_tile.token);

                                        const out_view = out_tile.bufferView();
                                        const a_view = a_tile.bufferView();
                                        const n: usize = if (out_view.layout.rank == 1) out_view.layout.shape[0] else (out_view.layout.shape[0] * out_view.layout.shape[1]);
                                        switch (out_view.dtype) {
                                            .f32 => elemwise.reluF32(out_view.bytes, a_view.bytes, n) catch |e| {
                                                t.fail(e);
                                                return;
                                            },
                                            .f16 => elemwise.reluF16(out_view.bytes, a_view.bytes, n) catch |e| {
                                                t.fail(e);
                                                return;
                                            },
                                            else => {
                                                t.fail(BackendError.InvalidArgument);
                                                return;
                                            },
                                        }
                                    }
                                }
                            };

                            var task: Task = .{ .store = store, .out_meta = out_meta, .out = s.out, .a = s.a };
                            var grain: usize = if (tile_bytes == 0) 32 else @max(@as(usize, 1), min_total_bytes / tile_bytes);
                            if (grain > tile_total) grain = tile_total;
                            p.parallelForAny(@ptrCast(&task), tile_total, grain, Task.runTiles);
                            if (task.err_any) |e| return @errorCast(e);
                            continue;
                        }
                    }

                    // Sequential fallback.
                    var ti0: usize = 0;
                    while (ti0 < out_meta.tile_counts[0]) : (ti0 += 1) {
                        var ti1: usize = 0;
                        while (ti1 < out_meta.tile_counts[1]) : (ti1 += 1) {
                            var out_tile = try store.acquireTileMut(s.out, ti0, ti1);
                            defer store.releaseMut(out_tile.token);
                            const a_tile = try store.acquireTileConst(s.a, ti0, ti1);
                            defer store.releaseConst(a_tile.token);
                            const out_view = out_tile.bufferView();
                            const a_view = a_tile.bufferView();
                            const n: usize = elemCountFromTileView(out_view);
                            switch (out_view.dtype) {
                                .f32 => try elemwise.reluF32(out_view.bytes, a_view.bytes, n),
                                .f16 => try elemwise.reluF16(out_view.bytes, a_view.bytes, n),
                                else => return BackendError.InvalidArgument,
                            }
                        }
                    }
                },

                .CopyTiled => |s| {
                    const dst_meta = try store.meta(s.dst);
                    const tile_total: usize = dst_meta.tile_counts[0] * dst_meta.tile_counts[1];
                    const tile_bytes: usize = tileByteSize(dst_meta);
                    const min_total_bytes: usize = 256 * 1024;

                    if (self.pool) |*p| {
                        if (shouldParallelTiles(self.thread_count, tile_total, tile_bytes, min_total_bytes)) {
                            const Task = struct {
                                store: tensor_store.TensorStore,
                                dst_meta: tensor_store.TensorMeta,
                                dst: tensor_store.TensorId,
                                src: tensor_store.TensorId,

                                stop: std.atomic.Value(bool) = .init(false),
                                err_mutex: std.Thread.Mutex = .{},
                                err_any: ?anyerror = null,

                                fn fail(t: *@This(), err: anyerror) void {
                                    if (t.stop.swap(true, .acq_rel)) return;
                                    t.err_mutex.lock();
                                    defer t.err_mutex.unlock();
                                    if (t.err_any == null) t.err_any = err;
                                }

                                fn runTiles(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
                                    _ = tid;
                                    const t: *@This() = @ptrCast(@alignCast(ctx_any));
                                    if (start >= end) return;
                                    if (t.stop.load(.acquire)) return;
                                    const tc1: usize = t.dst_meta.tile_counts[1];
                                    var i: usize = start;
                                    while (i < end) : (i += 1) {
                                        if (t.stop.load(.acquire)) return;
                                        const ti0: usize = i / tc1;
                                        const ti1: usize = i - ti0 * tc1;

                                        var dst_tile = t.store.acquireTileMut(t.dst, ti0, ti1) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                        defer t.store.releaseMut(dst_tile.token);
                                        const src_tile = t.store.acquireTileConst(t.src, ti0, ti1) catch |e| {
                                            t.fail(e);
                                            return;
                                        };
                                        defer t.store.releaseConst(src_tile.token);

                                        const n: usize = @min(dst_tile.bytes.len, src_tile.bytes.len);
                                        @memcpy(dst_tile.bytes[0..n], src_tile.bytes[0..n]);
                                    }
                                }
                            };

                            var task: Task = .{ .store = store, .dst_meta = dst_meta, .dst = s.dst, .src = s.src };
                            var grain: usize = if (tile_bytes == 0) 64 else @max(@as(usize, 1), min_total_bytes / tile_bytes);
                            if (grain > tile_total) grain = tile_total;
                            p.parallelForAny(@ptrCast(&task), tile_total, grain, Task.runTiles);
                            if (task.err_any) |e| return @errorCast(e);
                            continue;
                        }
                    }

                    // Sequential fallback.
                    var ti0: usize = 0;
                    while (ti0 < dst_meta.tile_counts[0]) : (ti0 += 1) {
                        var ti1: usize = 0;
                        while (ti1 < dst_meta.tile_counts[1]) : (ti1 += 1) {
                            var dst_tile = try store.acquireTileMut(s.dst, ti0, ti1);
                            defer store.releaseMut(dst_tile.token);
                            const src_tile = try store.acquireTileConst(s.src, ti0, ti1);
                            defer store.releaseConst(src_tile.token);
                            const n: usize = @min(dst_tile.bytes.len, src_tile.bytes.len);
                            @memcpy(dst_tile.bytes[0..n], src_tile.bytes[0..n]);
                        }
                    }
                },

                .ReduceAll => |s| {
                    try self.reduceAllScalar(s.op, s.out, s.a, store);
                },

                .ReTileCopyScalar => |s| {
                    try retileCopyScalar(store, s.dst, s.src);
                },
                .ReshapeScalar => |s| {
                    try reshapeCopyScalar(store, s.dst, s.src);
                },
                .Transpose2DScalar => |s| {
                    try transpose2DCopyScalar(store, s.dst, s.src);
                },
                .Slice2DScalar => |s| {
                    try slice2DCopyScalar(store, s.dst, s.src, s.start0, s.start1);
                },
            }
        }
    }

    fn kindImpl(_: *anyopaque) BackendKind {
        return .cpu;
    }

    fn nameImpl(_: *anyopaque) []const u8 {
        return "Aion CPU Backend";
    }

    fn capsImpl(ctx: *anyopaque) BackendCaps {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return .{
            .simd = true, // @Vector fast paths (compiler will scalarize if needed)
            .threads = (self.thread_count > 1),
            .fp16 = true,
            .int8 = true,
            .quant_q4 = true,
            .quant_q8 = true,
            .strided_copy = false, // v0: packed only
        };
    }

    fn deinitImpl(ctx: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.deinit();
    }

    fn elemwiseBinaryImpl(
        ctx: *anyopaque,
        op: ElemwiseBinaryOp,
        out: BufferViewMut,
        a: BufferViewConst,
        b: BufferViewConst,
    ) BackendError!void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        // Validation already done by Backend wrapper; types are scalar & packed.
        const elem_count: usize = utils.elemCount(out.layout.shape) catch return BackendError.InvalidArgument;

        const ElemwiseTask = struct {
            op: ElemwiseBinaryOp,
            dtype: DType,
            out_bytes: []u8,
            a_bytes: []const u8,
            b_bytes: []const u8,
            elem_count_total: usize,

            fn runChunk(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
                _ = tid;
                const task: *@This() = @ptrCast(@alignCast(ctx_any));
                if (start >= end) return;

                const count: usize = end - start;
                const elem_size: usize = switch (task.dtype) {
                    .f32 => 4,
                    .f16 => 2,
                    else => 0,
                };
                if (elem_size == 0) return;

                const off: usize = start * elem_size;
                // Best-effort safety: if the view is too small, do nothing.
                if (off > task.out_bytes.len or off > task.a_bytes.len or off > task.b_bytes.len) return;

                const out_s: []u8 = task.out_bytes[off..];
                const a_s: []const u8 = task.a_bytes[off..];
                const b_s: []const u8 = task.b_bytes[off..];

                // Ignore errors here: Backend wrapper already validated, and we do bounds checks above.
                // Any remaining mismatch should be extremely unlikely; worst-case a no-op chunk.
                switch (task.dtype) {
                    .f32 => elemwise.elemwiseBinaryF32(task.op, out_s, a_s, b_s, count) catch {},
                    .f16 => elemwise.elemwiseBinaryF16(task.op, out_s, a_s, b_s, count) catch {},
                    else => {},
                }
            }
        };

        switch (out.dtype) {
            .f32, .f16 => {
                if (self.pool) |*p| {
                    const elem_bytes: usize = switch (out.dtype) {
                        .f32 => @as(usize, 4),
                        .f16 => @as(usize, 2),
                        else => @as(usize, 0),
                    };
                    if (elem_bytes == 0) return BackendError.InvalidArgument;

                    // Heuristic: aim for ~64KiB contiguous chunks per thread.
                    // This is cache-friendly and amortizes thread pool overhead.
                    const target_chunk_bytes: usize = 64 * 1024;
                    var grain_elems: usize = target_chunk_bytes / elem_bytes;
                    if (grain_elems < 4096) grain_elems = 4096;

                    // Only parallelize if each thread gets at least one "chunk" worth of work.
                    const min_total_elems: usize = std.math.mul(usize, self.thread_count, grain_elems) catch elem_count;

                    if (elem_count >= min_total_elems) {
                        var task: ElemwiseTask = .{ .op = op, .dtype = out.dtype, .out_bytes = out.bytes, .a_bytes = a.bytes, .b_bytes = b.bytes, .elem_count_total = elem_count };
                        p.parallelForAny(@ptrCast(&task), elem_count, grain_elems, ElemwiseTask.runChunk);
                        return;
                    }
                }

                // Fallback single-thread.
                switch (out.dtype) {
                    .f32 => try elemwise.elemwiseBinaryF32(op, out.bytes, a.bytes, b.bytes, elem_count),
                    .f16 => try elemwise.elemwiseBinaryF16(op, out.bytes, a.bytes, b.bytes, elem_count),
                    else => unreachable,
                }
            },
            .i8 => return BackendError.Unsupported, // integer elemwise not implemented yet
            .q4_0, .q8_0 => return BackendError.Unsupported, // quant elemwise makes no sense
        }
    }

    fn matmulImpl(
        ctx: *anyopaque,
        params: MatMulParams,
        c: BufferViewMut,
        a: BufferViewConst,
        b: BufferViewConst,
    ) BackendError!void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        // C is always scalar, A is always scalar, B can be quant.
        const c_dtype = c.dtype;
        const a_dtype = a.dtype;
        const b_dtype = b.dtype;

        // Simple, deterministic parallelization over output rows (M).
        // This is contention-free: each chunk writes disjoint C row ranges.
        const do_parallel: bool = (self.thread_count > 1) and (params.m >= 2 * self.thread_count);
        if (do_parallel and self.pool != null) {
            const MatmulTask = struct {
                params: MatMulParams,
                c_dtype: DType,
                a_dtype: DType,
                b_dtype: DType,
                c_bytes: []u8,
                a_bytes: []const u8,
                b_bytes: []const u8,

                fn runRows(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
                    _ = tid;
                    const task: *@This() = @ptrCast(@alignCast(ctx_any));
                    if (start >= end) return;

                    const m_chunk: usize = end - start;
                    var p: MatMulParams = task.params;
                    p.m = m_chunk;

                    const n: usize = task.params.n;
                    const k: usize = task.params.k;

                    // Byte offsets for row-major packed matrices.
                    const a_elem_bytes: usize = switch (task.a_dtype) {
                        .f32 => @as(usize, 4),
                        .f16 => @as(usize, 2),
                        else => @as(usize, 0),
                    };
                    const c_elem_bytes: usize = switch (task.c_dtype) {
                        .f32 => @as(usize, 4),
                        .f16 => @as(usize, 2),
                        else => @as(usize, 0),
                    };
                    const a_row_bytes: usize = k * a_elem_bytes;
                    const c_row_bytes: usize = n * c_elem_bytes;
                    if (a_row_bytes == 0 or c_row_bytes == 0) return;

                    const a_off: usize = start * a_row_bytes;
                    const c_off: usize = start * c_row_bytes;
                    if (a_off > task.a_bytes.len or c_off > task.c_bytes.len) return;

                    const a_s: []const u8 = task.a_bytes[a_off..];
                    const c_s: []u8 = task.c_bytes[c_off..];

                    // Dispatch mirrors CpuBackend.matmulImpl, but operating on row-slices.
                    switch (task.b_dtype) {
                        .f32 => matmul_k.matmulF32(p, c_s, a_s, task.b_bytes) catch {},
                        .f16 => {
                            if (task.c_dtype == .f32) {
                                matmul_k.matmulF16ToF32(p, c_s, a_s, task.b_bytes) catch {};
                            } else if (task.c_dtype == .f16) {
                                matmul_k.matmulF16(p, c_s, a_s, task.b_bytes) catch {};
                            }
                        },
                        .q4_0 => matmul_k.matmulQ4_0(p, c_s, a_s, task.b_bytes) catch {},
                        .q8_0 => matmul_k.matmulQ8_0(p, c_s, a_s, task.b_bytes) catch {},
                        else => {},
                    }
                }
            };

            var task: MatmulTask = .{
                .params = params,
                .c_dtype = c_dtype,
                .a_dtype = a_dtype,
                .b_dtype = b_dtype,
                .c_bytes = c.bytes,
                .a_bytes = a.bytes,
                .b_bytes = b.bytes,
            };

            // We already validated dtype combinations below, but do it early so we can return errors.
            switch (b_dtype) {
                .f32 => {
                    if (a_dtype != .f32 or c_dtype != .f32) return BackendError.InvalidArgument;
                },
                .f16 => {
                    if (a_dtype != .f16) return BackendError.InvalidArgument;
                    if (!(c_dtype == .f32 or c_dtype == .f16)) return BackendError.InvalidArgument;
                },
                .q4_0, .q8_0 => {
                    if (a_dtype != .f32 or c_dtype != .f32) return BackendError.InvalidArgument;
                },
                else => {},
            }

            // Don't sub-chunk within each thread: we want each thread to call into the
            // matmul kernel once for its contiguous row range (reduces call overhead).
            self.pool.?.parallelForAny(@ptrCast(&task), params.m, 0, MatmulTask.runRows);
            return;
        }

        switch (b_dtype) {
            .f32 => {
                if (a_dtype != .f32 or c_dtype != .f32) return BackendError.InvalidArgument;
                try matmul_k.matmulF32(params, c.bytes, a.bytes, b.bytes);
            },
            .f16 => {
                // Supported:
                // - A=f16, B=f16, C=f16
                // - A=f16, B=f16, C=f32  (accumulate in f32)
                if (a_dtype != .f16) return BackendError.InvalidArgument;
                if (c_dtype == .f32) {
                    try matmul_k.matmulF16ToF32(params, c.bytes, a.bytes, b.bytes);
                } else if (c_dtype == .f16) {
                    try matmul_k.matmulF16(params, c.bytes, a.bytes, b.bytes);
                } else {
                    return BackendError.InvalidArgument;
                }
            },
            .q4_0 => {
                // A must be f32, C must be f32 (dequant on the fly)
                if (a_dtype != .f32 or c_dtype != .f32) return BackendError.InvalidArgument;
                try matmul_k.matmulQ4_0(params, c.bytes, a.bytes, b.bytes);
            },
            .q8_0 => {
                if (a_dtype != .f32 or c_dtype != .f32) return BackendError.InvalidArgument;
                try matmul_k.matmulQ8_0(params, c.bytes, a.bytes, b.bytes);
            },
            .i8 => return BackendError.Unsupported,
        }
    }

    fn broadcastLastDimBinaryImpl(
        ctx: *anyopaque,
        op: ElemwiseBinaryOp,
        out: BufferViewMut,
        a: BufferViewConst,
        b: BufferViewConst,
    ) BackendError!void {
        const self: *Self = @ptrCast(@alignCast(ctx));

        // Backend wrapper validated packedness and dtype matches.
        const rank: usize = @as(usize, out.layout.rank);
        if (rank == 0 or out.layout.shape.len == 0) return BackendError.InvalidArgument;

        const col_count: usize = out.layout.shape[out.layout.shape.len - 1];
        if (col_count == 0) return;

        const elem_count: usize = utils.elemCount(out.layout.shape) catch return BackendError.InvalidArgument;
        if (elem_count == 0) return;
        if (elem_count % col_count != 0) return BackendError.InvalidArgument;
        const row_count: usize = elem_count / col_count;

        switch (out.dtype) {
            .f32, .f16 => {
                if (self.pool) |*p| {
                    const elem_bytes: usize = switch (out.dtype) {
                        .f32 => @as(usize, 4),
                        .f16 => @as(usize, 2),
                        else => @as(usize, 0),
                    };
                    if (elem_bytes == 0) return BackendError.InvalidArgument;

                    const row_bytes: usize = col_count * elem_bytes;
                    // Aim for ~128KiB contiguous work per chunk.
                    const target_bytes: usize = 128 * 1024;
                    var grain_rows: usize = if (row_bytes == 0) 0 else (target_bytes / row_bytes);
                    if (grain_rows < 1) grain_rows = 1;

                    // Only parallelize if there is enough outer work.
                    const min_rows: usize = std.math.mul(usize, self.thread_count, 2) catch row_count;
                    if (row_count >= min_rows and self.thread_count > 1) {
                        const Task = struct {
                            op: ElemwiseBinaryOp,
                            dtype: DType,
                            out_bytes: []u8,
                            a_bytes: []const u8,
                            b_bytes: []const u8,
                            col_count: usize,

                            fn runRows(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
                                _ = tid;
                                const t: *@This() = @ptrCast(@alignCast(ctx_any));
                                if (start >= end) return;

                                // Slice the row range into a contiguous element range.
                                const row_count_local: usize = end - start;
                                const col: usize = t.col_count;
                                const elem_size: usize = switch (t.dtype) {
                                    .f32 => 4,
                                    .f16 => 2,
                                    else => 0,
                                };
                                if (elem_size == 0) return;

                                const off: usize = start * col * elem_size;
                                if (off > t.out_bytes.len or off > t.a_bytes.len) return;
                                const out_s: []u8 = t.out_bytes[off..];
                                const a_s: []const u8 = t.a_bytes[off..];

                                // b always starts at offset 0.
                                switch (t.dtype) {
                                    .f32 => switch (t.op) {
                                        .add => elemwise.broadcastLastDimBinaryF32(.add, out_s, a_s, t.b_bytes, row_count_local, col) catch {},
                                        .sub => elemwise.broadcastLastDimBinaryF32(.sub, out_s, a_s, t.b_bytes, row_count_local, col) catch {},
                                        .mul => elemwise.broadcastLastDimBinaryF32(.mul, out_s, a_s, t.b_bytes, row_count_local, col) catch {},
                                        .div => elemwise.broadcastLastDimBinaryF32(.div, out_s, a_s, t.b_bytes, row_count_local, col) catch {},
                                    },
                                    .f16 => switch (t.op) {
                                        .add => elemwise.broadcastLastDimBinaryF16(.add, out_s, a_s, t.b_bytes, row_count_local, col) catch {},
                                        .sub => elemwise.broadcastLastDimBinaryF16(.sub, out_s, a_s, t.b_bytes, row_count_local, col) catch {},
                                        .mul => elemwise.broadcastLastDimBinaryF16(.mul, out_s, a_s, t.b_bytes, row_count_local, col) catch {},
                                        .div => elemwise.broadcastLastDimBinaryF16(.div, out_s, a_s, t.b_bytes, row_count_local, col) catch {},
                                    },
                                    else => {},
                                }
                            }
                        };

                        var task: Task = .{ .op = op, .dtype = out.dtype, .out_bytes = out.bytes, .a_bytes = a.bytes, .b_bytes = b.bytes, .col_count = col_count };
                        p.parallelForAny(@ptrCast(&task), row_count, grain_rows, Task.runRows);
                        return;
                    }
                }

                // Fallback single-thread.
                switch (out.dtype) {
                    .f32 => switch (op) {
                        .add => try elemwise.broadcastLastDimBinaryF32(.add, out.bytes, a.bytes, b.bytes, row_count, col_count),
                        .sub => try elemwise.broadcastLastDimBinaryF32(.sub, out.bytes, a.bytes, b.bytes, row_count, col_count),
                        .mul => try elemwise.broadcastLastDimBinaryF32(.mul, out.bytes, a.bytes, b.bytes, row_count, col_count),
                        .div => try elemwise.broadcastLastDimBinaryF32(.div, out.bytes, a.bytes, b.bytes, row_count, col_count),
                    },
                    .f16 => switch (op) {
                        .add => try elemwise.broadcastLastDimBinaryF16(.add, out.bytes, a.bytes, b.bytes, row_count, col_count),
                        .sub => try elemwise.broadcastLastDimBinaryF16(.sub, out.bytes, a.bytes, b.bytes, row_count, col_count),
                        .mul => try elemwise.broadcastLastDimBinaryF16(.mul, out.bytes, a.bytes, b.bytes, row_count, col_count),
                        .div => try elemwise.broadcastLastDimBinaryF16(.div, out.bytes, a.bytes, b.bytes, row_count, col_count),
                    },
                    else => unreachable,
                }
            },
            else => return BackendError.Unsupported,
        }
    }

    fn reluImpl(
        ctx: *anyopaque,
        out: BufferViewMut,
        a: BufferViewConst,
    ) BackendError!void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const elem_count: usize = utils.elemCount(out.layout.shape) catch return BackendError.InvalidArgument;

        switch (out.dtype) {
            .f32, .f16 => {
                if (self.pool) |*p| {
                    const elem_bytes: usize = switch (out.dtype) {
                        .f32 => @as(usize, 4),
                        .f16 => @as(usize, 2),
                        else => @as(usize, 0),
                    };
                    if (elem_bytes == 0) return BackendError.InvalidArgument;

                    // Similar heuristic to elemwiseBinary.
                    const target_chunk_bytes: usize = 64 * 1024;
                    var grain_elems: usize = target_chunk_bytes / elem_bytes;
                    if (grain_elems < 4096) grain_elems = 4096;

                    const min_total_elems: usize = std.math.mul(usize, self.thread_count, grain_elems) catch elem_count;
                    if (elem_count >= min_total_elems) {
                        const Task = struct {
                            dtype: DType,
                            out_bytes: []u8,
                            a_bytes: []const u8,

                            fn runChunk(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
                                _ = tid;
                                const t: *@This() = @ptrCast(@alignCast(ctx_any));
                                if (start >= end) return;

                                const count: usize = end - start;
                                const elem_size: usize = switch (t.dtype) {
                                    .f32 => 4,
                                    .f16 => 2,
                                    else => 0,
                                };
                                if (elem_size == 0) return;

                                const off: usize = start * elem_size;
                                if (off > t.out_bytes.len or off > t.a_bytes.len) return;

                                const out_s: []u8 = t.out_bytes[off..];
                                const a_s: []const u8 = t.a_bytes[off..];

                                switch (t.dtype) {
                                    .f32 => elemwise.reluF32(out_s, a_s, count) catch {},
                                    .f16 => elemwise.reluF16(out_s, a_s, count) catch {},
                                    else => {},
                                }
                            }
                        };

                        var task: Task = .{ .dtype = out.dtype, .out_bytes = out.bytes, .a_bytes = a.bytes };
                        p.parallelForAny(@ptrCast(&task), elem_count, grain_elems, Task.runChunk);
                        return;
                    }
                }

                switch (out.dtype) {
                    .f32 => try elemwise.reluF32(out.bytes, a.bytes, elem_count),
                    .f16 => try elemwise.reluF16(out.bytes, a.bytes, elem_count),
                    else => unreachable,
                }
            },
            else => return BackendError.Unsupported,
        }
    }

    fn reduceImpl(
        ctx: *anyopaque,
        op: types.ReduceOp,
        out: BufferViewMut,
        a: BufferViewConst,
    ) BackendError!void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const elem_count: usize = utils.elemCount(a.layout.shape) catch return BackendError.InvalidArgument;

        if (op == .mean and elem_count == 0) return BackendError.InvalidArgument;

        // v0: only scalar dtypes, output must be f32 or f16.
        if (!(out.dtype == .f32 or out.dtype == .f16)) return BackendError.InvalidArgument;
        if (!(a.dtype == .f32 or a.dtype == .f16)) return BackendError.InvalidArgument;

        // Empty sum is defined as 0.
        if (elem_count == 0) {
            switch (out.dtype) {
                .f32 => {
                    const o: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, out.bytes);
                    if (o.len < 1) return BackendError.InvalidArgument;
                    o[0] = 0.0;
                },
                .f16 => {
                    const o: []align(1) f16 = simd.bytesAsSliceMutUnaligned(f16, out.bytes);
                    if (o.len < 1) return BackendError.InvalidArgument;
                    o[0] = 0.0;
                },
                else => unreachable,
            }
            return;
        }

        var sum: f32 = 0.0;

        const elem_bytes: usize = switch (a.dtype) {
            .f32 => @as(usize, 4),
            .f16 => @as(usize, 2),
            else => @as(usize, 0),
        };
        if (elem_bytes == 0) return BackendError.InvalidArgument;

        const do_parallel_initial: bool = (self.thread_count > 1) and (self.pool != null) and (self.reduce_scratch_f32.len >= self.thread_count);
        if (do_parallel_initial) {
            // Heuristic: ~256KiB of input per thread-chunk.
            // IMPORTANT: reduce() is also used by the tiled Program execution path.
            // If we parallelize reductions on tiny tiles, thread scheduling overhead dominates
            // and performance collapses (especially at higher thread counts).
            const target_chunk_bytes: usize = 256 * 1024;
            var grain_elems: usize = target_chunk_bytes / elem_bytes;
            if (grain_elems < 8192) grain_elems = 8192;

            const min_total_elems: usize = std.math.mul(usize, self.thread_count, grain_elems) catch elem_count;
            if (elem_count < min_total_elems) {
                // Not enough work to amortize parallel overhead.
                sum = switch (a.dtype) {
                    .f32 => try reduce_k.sumF32Range(a.bytes, 0, elem_count),
                    .f16 => try reduce_k.sumF16RangeToF32(a.bytes, 0, elem_count),
                    else => return BackendError.InvalidArgument,
                };
            } else {
                // Initialize per-thread partials.
                @memset(self.reduce_scratch_f32[0..self.thread_count], 0.0);

                const Task = struct {
                    dtype: DType,
                    a_bytes: []const u8,
                    partials: []f32,

                    fn runChunk(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
                        const t: *@This() = @ptrCast(@alignCast(ctx_any));
                        if (start >= end) return;
                        if (tid >= t.partials.len) return;

                        const part: f32 = switch (t.dtype) {
                            .f32 => reduce_k.sumF32Range(t.a_bytes, start, end) catch 0.0,
                            .f16 => reduce_k.sumF16RangeToF32(t.a_bytes, start, end) catch 0.0,
                            else => 0.0,
                        };
                        t.partials[tid] += part;
                    }
                };

                var task: Task = .{ .dtype = a.dtype, .a_bytes = a.bytes, .partials = self.reduce_scratch_f32[0..self.thread_count] };

                self.pool.?.parallelForAny(@ptrCast(&task), elem_count, grain_elems, Task.runChunk);

                var i: usize = 0;
                while (i < self.thread_count) : (i += 1) {
                    sum += self.reduce_scratch_f32[i];
                }
            }
        } else {
            sum = switch (a.dtype) {
                .f32 => try reduce_k.sumF32Range(a.bytes, 0, elem_count),
                .f16 => try reduce_k.sumF16RangeToF32(a.bytes, 0, elem_count),
                else => return BackendError.InvalidArgument,
            };
        }

        if (op == .mean) {
            sum *= 1.0 / @as(f32, @floatFromInt(elem_count));
        }

        // Store.
        switch (out.dtype) {
            .f32 => {
                const o: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, out.bytes);
                if (o.len < 1) return BackendError.InvalidArgument;
                o[0] = sum;
            },
            .f16 => {
                const o: []align(1) f16 = simd.bytesAsSliceMutUnaligned(f16, out.bytes);
                if (o.len < 1) return BackendError.InvalidArgument;
                o[0] = @floatCast(sum);
            },
            else => unreachable,
        }
    }

    fn copyImpl(
        _: *anyopaque,
        dst: BufferViewMut,
        src: BufferViewConst,
    ) BackendError!void {

        // Caller is expected to validate dtype/shape/packedness via Backend.copy(),
        // but keep this implementation robust if invoked directly.
        const need: usize = utils.requiredByteLen(dst.dtype, dst.layout) catch return BackendError.InvalidArgument;
        if (dst.bytes.len < need or src.bytes.len < need) return BackendError.InvalidArgument;
        @memcpy(dst.bytes[0..need], src.bytes[0..need]);
    }
};
