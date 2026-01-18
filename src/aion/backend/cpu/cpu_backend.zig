const std = @import("std");
const backend_mod = @import("../backend.zig");
const types = @import("../types.zig");
const utils = @import("../utils.zig");
const elemwise = @import("kernels/elemwise.zig");
const matmul_k = @import("kernels/matmul.zig");
const matmul_registry = @import("kernels/matmul_registry.zig");
const matvec_k = @import("kernels/matvecmul.zig");
const reduce_k = @import("kernels/reduce.zig");
const simd = @import("kernels/simd.zig");
const thread_pool = @import("../../runtime/thread_pool.zig");
const executable = @import("../../runtime/executable.zig");
const x86_cpuid = @import("tuning/x86_cpuid.zig");
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

    matmul_f32: matmul_registry.F32Kernels = matmul_registry.candidates[1].kernels,

    // Per-thread scratch for f32 matmul packing (A/B panels).
    // Size == thread_count. Avoids large per-call stack frames.
    matmul_scratch_f32: [][]align(32) u8 = &[_][]align(32) u8{},

    // Future: thread pool handle, SIMD feature flags, scratch buffers

    const Self = @This();

    pub const Options = struct {
        /// Total threads to use including the calling thread.
        /// Set to 1 to disable parallelism (default).
        thread_count: usize = 1,
    };

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{ .allocator = allocator, .pool = null, .thread_count = 1, .reduce_scratch_f32 = &[_]f32{}, .matmul_scratch_f32 = &[_][]align(32) u8{} };
    }

    pub fn initWithOptions(allocator: std.mem.Allocator, opts: Options) !Self {
        if (opts.thread_count == 0) return error.InvalidArgument;

        var self: Self = .{ .allocator = allocator, .pool = null, .thread_count = 1, .reduce_scratch_f32 = &[_]f32{}, .matmul_scratch_f32 = &[_][]align(32) u8{} };
        if (opts.thread_count > 1) {
            const p = try thread_pool.ThreadPool.init(allocator, .{ .thread_count = opts.thread_count });
            self.pool = p;
            self.thread_count = opts.thread_count;

            // Allocate reduction scratch once. No allocations during op execution.
            self.reduce_scratch_f32 = try allocator.alloc(f32, opts.thread_count);

            // Allocate matmul scratch once. No allocations during op execution.
            // Select a variant based on CPU cache sizes (when available).
            const cpu_info = x86_cpuid.detect();
            const mm_choice = matmul_registry.selectHeuristic(cpu_info.caches.l2_bytes);
            self.matmul_f32 = mm_choice.kernels;

            var mm: [][]align(32) u8 = try allocator.alloc([]align(32) u8, opts.thread_count);
            errdefer allocator.free(mm);
            var i: usize = 0;
            errdefer {
                var j: usize = 0;
                while (j < i) : (j += 1) allocator.free(mm[j]);
            }
            while (i < opts.thread_count) : (i += 1) {
                mm[i] = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), self.matmul_f32.scratch_bytes);
            }
            self.matmul_scratch_f32 = mm;
        }
        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.pool) |*p| {
            p.deinit();
            self.pool = null;
        }

        if (self.matmul_scratch_f32.len != 0) {
            for (self.matmul_scratch_f32) |buf| {
                self.allocator.free(buf);
            }
            self.allocator.free(self.matmul_scratch_f32);
            self.matmul_scratch_f32 = &[_][]align(32) u8{};
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
                    const b_meta = try store.meta(s.b);

                    const a_dtype: DType = a_meta.dtype;
                    const b_dtype: DType = b_meta.dtype;
                    const c_dtype: DType = c_meta.dtype;

                    const a_info = a_dtype.info();
                    const c_info = c_dtype.info();

                    const tile_total: usize = c_meta.tile_counts[0] * c_meta.tile_counts[1];
                    if (self.pool) |*p| {
                        // NOTE on correctness: we must not run multiple tasks that mutate the same
                        // output tile concurrently (data race). Therefore we currently parallelize
                        // at the granularity of whole output tiles only.
                        const total_work: usize = tile_total;

                        // MatMul tiles are compute-heavy; parallelize if we have enough work.
                        if (self.thread_count > 1 and total_work >= 2) {
                            const Task = struct {
                                store: tensor_store.TensorStore,
                                c_meta: tensor_store.TensorMeta,
                                a_meta: tensor_store.TensorMeta,
                                b_meta: tensor_store.TensorMeta,
                                a_dtype: DType,
                                b_dtype: DType,
                                c_dtype: DType,
                                a_info: types.DTypeInfo,
                                c_info: types.DTypeInfo,
                                s: @TypeOf(s),

                                scratch: [][]align(32) u8,
                                matmul_f32: matmul_registry.F32Kernels,

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
                                    const t: *@This() = @ptrCast(@alignCast(ctx_any));
                                    if (start >= end) return;

                                    if (tid >= t.scratch.len) return;

                                    if (t.stop.load(.acquire)) return;

                                    const tc0: usize = t.c_meta.tile_counts[0];
                                    const is_matvec: bool = (t.c_meta.shape[0] == 1);

                                    // Process tiles in column-major order. Within this range, tiles
                                    // with the same `ti_n` are contiguous, so we can amortize B packing:
                                    // pack B once per (ti_k, ti_n) and apply it across multiple `ti_m`.
                                    const k_tiles: usize = t.a_meta.tile_counts[1];

                                    var i: usize = start;
                                    while (i < end) {
                                        if (t.stop.load(.acquire)) return;

                                        const tile_idx0: usize = i;
                                        const ti_n: usize = tile_idx0 / tc0;
                                        const group_end: usize = @min(end, (ti_n + 1) * tc0);

                                        const ti_m0: usize = tile_idx0 - ti_n * tc0;
                                        const ti_m_end: usize = group_end - ti_n * tc0;

                                        var ti_k: usize = 0;
                                        while (ti_k < k_tiles) : (ti_k += 1) {
                                            if (t.stop.load(.acquire)) return;
                                            const beta_tile: f32 = if (ti_k == 0) t.s.beta else 1.0;

                                            const b_tile = t.store.acquireTileConst(t.s.b, ti_k, ti_n) catch |e| {
                                                t.fail(e);
                                                return;
                                            };
                                            defer t.store.releaseConst(b_tile.token);
                                            const b_view = b_tile.bufferView();

                                            // Fast path for matvec-shaped problems in tiled execution:
                                            // when there is exactly one M-tile, packing B for reuse is pointless.
                                            if (is_matvec and t.a_dtype == .f32 and t.c_dtype == .f32) {
                                                const k_tile: usize = b_view.layout.shape[0];
                                                const n_tile: usize = b_view.layout.shape[1];

                                                // Only one ti_m exists.
                                                const ti_m: usize = 0;
                                                var c_tile = t.store.acquireTileMut(t.s.c, ti_m, ti_n) catch |e| {
                                                    t.fail(e);
                                                    return;
                                                };
                                                defer t.store.releaseMut(c_tile.token);
                                                const c_view0 = c_tile.bufferView();

                                                const a_tile = t.store.acquireTileConst(t.s.a, ti_m, ti_k) catch |e| {
                                                    t.fail(e);
                                                    return;
                                                };
                                                defer t.store.releaseConst(a_tile.token);
                                                const a_view = a_tile.bufferView();

                                                const beta_eff: f32 = if (ti_k == 0) t.s.beta else 1.0;
                                                const params: MatMulParams = .{ .m = 1, .n = n_tile, .k = k_tile, .alpha = t.s.alpha, .beta = beta_eff };

                                                switch (t.b_dtype) {
                                                    .f32 => matvec_k.matvecF32(params, c_view0.bytes, a_view.bytes, b_view.bytes) catch |e| {
                                                        t.fail(e);
                                                        return;
                                                    },
                                                    .q4_0 => matvec_k.matvecQ4_0(params, c_view0.bytes, a_view.bytes, b_view.bytes) catch |e| {
                                                        t.fail(e);
                                                        return;
                                                    },
                                                    .q8_0 => matvec_k.matvecQ8_0(params, c_view0.bytes, a_view.bytes, b_view.bytes) catch |e| {
                                                        t.fail(e);
                                                        return;
                                                    },
                                                    else => {
                                                        t.fail(BackendError.Unsupported);
                                                        return;
                                                    },
                                                }

                                                // Done for this (ti_k, ti_n) group.
                                                continue;
                                            }

                                            if (t.b_dtype == .f32 and t.a_dtype == .f32 and t.c_dtype == .f32) {
                                                const k_tile: usize = b_view.layout.shape[0];
                                                const n_tile: usize = b_view.layout.shape[1];

                                                // If the planned tiles exceed the selected packing kernel's limits,
                                                // fall back to the un-packed kernel for correctness.
                                                if (k_tile > t.matmul_f32.tuning.kc or n_tile > t.matmul_f32.tuning.nc) {
                                                    var ti_m: usize = ti_m0;
                                                    while (ti_m < ti_m_end) : (ti_m += 1) {
                                                        if (t.stop.load(.acquire)) return;

                                                        var c_tile = t.store.acquireTileMut(t.s.c, ti_m, ti_n) catch |e| {
                                                            t.fail(e);
                                                            return;
                                                        };
                                                        defer t.store.releaseMut(c_tile.token);

                                                        const c_view0 = c_tile.bufferView();
                                                        const m_total: usize = c_view0.layout.shape[0];

                                                        const a_tile = t.store.acquireTileConst(t.s.a, ti_m, ti_k) catch |e| {
                                                            t.fail(e);
                                                            return;
                                                        };
                                                        defer t.store.releaseConst(a_tile.token);
                                                        const a_view = a_tile.bufferView();

                                                        // If the scheduled tile exceeds the tuned kernel limits, compute it
                                                        // in smaller tiles using the tuned kernel.
                                                        const kk_total: usize = k_tile;
                                                        const jj_total: usize = n_tile;

                                                        const pb_f32_len: usize = t.matmul_f32.tuning.kc * t.matmul_f32.tuning.nc;
                                                        const pb_bytes_len: usize = pb_f32_len * @sizeOf(f32);
                                                        const packed_b_view: []align(32) const f32 = @alignCast(std.mem.bytesAsSlice(f32, t.scratch[tid][0..pb_bytes_len]));

                                                        var jj: usize = 0;
                                                        while (jj < jj_total) : (jj += t.matmul_f32.tuning.nc) {
                                                            const n_sub: usize = @min(t.matmul_f32.tuning.nc, jj_total - jj);

                                                            var kk: usize = 0;
                                                            while (kk < kk_total) : (kk += t.matmul_f32.tuning.kc) {
                                                                const k_sub: usize = @min(t.matmul_f32.tuning.kc, kk_total - kk);
                                                                const beta_eff: f32 = if (kk == 0) beta_tile else 1.0;

                                                                const b_off: usize = (kk * jj_total + jj) * @sizeOf(f32);
                                                                const b_need: usize = k_sub * n_sub * @sizeOf(f32);
                                                                if (b_off + b_need > b_view.bytes.len) {
                                                                    t.fail(BackendError.InvalidArgument);
                                                                    return;
                                                                }
                                                                const b_sub_bytes: []const u8 = b_view.bytes[b_off .. b_off + b_need];
                                                                t.matmul_f32.pack_b_tile(t.scratch[tid], k_sub, n_sub, b_sub_bytes) catch |e| {
                                                                    t.fail(e);
                                                                    return;
                                                                };

                                                                // Compute C tile for this sub-panel.
                                                                // C and A are packed tiles already (row-major), so we can slice per-subtile.

                                                                // A tile is [m_tile, kk_total] contiguous.
                                                                const a_row_bytes: usize = kk_total * @sizeOf(f32);
                                                                const a_k_off: usize = kk * @sizeOf(f32);

                                                                // C tile is [m_tile, jj_total] contiguous.
                                                                const c_row_bytes: usize = jj_total * @sizeOf(f32);
                                                                const c_n_off: usize = jj * @sizeOf(f32);

                                                                var row: usize = 0;
                                                                while (row < m_total) : (row += 1) {
                                                                    const a_row: []const u8 = a_view.bytes[row * a_row_bytes .. (row + 1) * a_row_bytes];
                                                                    const c_row: []u8 = c_view0.bytes[row * c_row_bytes .. (row + 1) * c_row_bytes];

                                                                    const a_sub: []const u8 = a_row[a_k_off .. a_k_off + k_sub * @sizeOf(f32)];
                                                                    const c_sub: []u8 = c_row[c_n_off .. c_n_off + n_sub * @sizeOf(f32)];

                                                                    const pp: MatMulParams = .{ .m = 1, .n = n_sub, .k = k_sub, .alpha = t.s.alpha, .beta = beta_eff };
                                                                    t.matmul_f32.matmul_packed_b(t.scratch[tid], packed_b_view, pp, c_sub, a_sub) catch |e| {
                                                                        t.fail(e);
                                                                        return;
                                                                    };
                                                                }
                                                            }
                                                        }

                                                        continue;
                                                    }

                                                    continue;
                                                }

                                                t.matmul_f32.pack_b_tile(t.scratch[tid], k_tile, n_tile, b_view.bytes) catch |e| {
                                                    t.fail(e);
                                                    return;
                                                };

                                                var ti_m: usize = ti_m0;
                                                while (ti_m < ti_m_end) : (ti_m += 1) {
                                                    if (t.stop.load(.acquire)) return;

                                                    var c_tile = t.store.acquireTileMut(t.s.c, ti_m, ti_n) catch |e| {
                                                        t.fail(e);
                                                        return;
                                                    };
                                                    defer t.store.releaseMut(c_tile.token);
                                                    const c_view0 = c_tile.bufferView();
                                                    const m_tile: usize = c_view0.layout.shape[0];

                                                    const a_tile = t.store.acquireTileConst(t.s.a, ti_m, ti_k) catch |e| {
                                                        t.fail(e);
                                                        return;
                                                    };
                                                    defer t.store.releaseConst(a_tile.token);
                                                    const a_view = a_tile.bufferView();

                                                    const params: MatMulParams = .{ .m = m_tile, .n = n_tile, .k = k_tile, .alpha = t.s.alpha, .beta = beta_tile };

                                                    const pb_f32_len: usize = t.matmul_f32.tuning.kc * t.matmul_f32.tuning.nc;
                                                    const pb_bytes_len: usize = pb_f32_len * @sizeOf(f32);
                                                    const packed_b_view: []align(32) const f32 = @alignCast(std.mem.bytesAsSlice(f32, t.scratch[tid][0..pb_bytes_len]));
                                                    t.matmul_f32.matmul_packed_b(t.scratch[tid], packed_b_view, params, c_view0.bytes, a_view.bytes) catch |e| {
                                                        t.fail(e);
                                                        return;
                                                    };
                                                }
                                            } else {
                                                // Fallback: no packed-B reuse path for non-f32 yet.
                                                var ti_m: usize = ti_m0;
                                                while (ti_m < ti_m_end) : (ti_m += 1) {
                                                    if (t.stop.load(.acquire)) return;

                                                    var c_tile = t.store.acquireTileMut(t.s.c, ti_m, ti_n) catch |e| {
                                                        t.fail(e);
                                                        return;
                                                    };
                                                    defer t.store.releaseMut(c_tile.token);
                                                    const c_view0 = c_tile.bufferView();
                                                    const m_tile: usize = c_view0.layout.shape[0];
                                                    const n_tile: usize = c_view0.layout.shape[1];

                                                    const a_tile = t.store.acquireTileConst(t.s.a, ti_m, ti_k) catch |e| {
                                                        t.fail(e);
                                                        return;
                                                    };
                                                    defer t.store.releaseConst(a_tile.token);
                                                    const a_view = a_tile.bufferView();

                                                    const k_tile: usize = a_view.layout.shape[1];
                                                    const params: MatMulParams = .{ .m = m_tile, .n = n_tile, .k = k_tile, .alpha = t.s.alpha, .beta = beta_tile };

                                                    switch (t.b_dtype) {
                                                        .f16 => {
                                                            if (t.a_dtype != .f16) {
                                                                t.fail(BackendError.InvalidArgument);
                                                                return;
                                                            }
                                                            if (t.c_dtype == .f32) {
                                                                matmul_k.matmulF16ToF32(params, c_view0.bytes, a_view.bytes, b_view.bytes) catch |e| {
                                                                    t.fail(e);
                                                                    return;
                                                                };
                                                            } else if (t.c_dtype == .f16) {
                                                                matmul_k.matmulF16(params, c_view0.bytes, a_view.bytes, b_view.bytes) catch |e| {
                                                                    t.fail(e);
                                                                    return;
                                                                };
                                                            } else {
                                                                t.fail(BackendError.InvalidArgument);
                                                                return;
                                                            }
                                                        },
                                                        .q4_0 => {
                                                            if (t.a_dtype != .f32 or t.c_dtype != .f32) {
                                                                t.fail(BackendError.InvalidArgument);
                                                                return;
                                                            }
                                                            matmul_k.matmulQ4_0(params, c_view0.bytes, a_view.bytes, b_view.bytes) catch |e| {
                                                                t.fail(e);
                                                                return;
                                                            };
                                                        },
                                                        .q8_0 => {
                                                            if (t.a_dtype != .f32 or t.c_dtype != .f32) {
                                                                t.fail(BackendError.InvalidArgument);
                                                                return;
                                                            }
                                                            matmul_k.matmulQ8_0(params, c_view0.bytes, a_view.bytes, b_view.bytes) catch |e| {
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

                                        i = group_end;
                                    }
                                }
                            };

                            var task: Task = .{
                                .store = store,
                                .c_meta = c_meta,
                                .a_meta = a_meta,
                                .b_meta = b_meta,
                                .a_dtype = a_dtype,
                                .b_dtype = b_dtype,
                                .c_dtype = c_dtype,
                                .a_info = a_info,
                                .c_info = c_info,
                                .s = s,
                                .scratch = self.matmul_scratch_f32,
                                .matmul_f32 = self.matmul_f32,
                            };

                            // Reduce callback overhead while also enabling B-pack amortization.
                            // Our tile linearization is column-major (ti_n major, ti_m minor).
                            // If `grain` divides `tc0`, each callback receives a contiguous range
                            // that stays within a single `ti_n` group, allowing `runTiles` to
                            // pack B once and apply it to multiple `ti_m` tiles.
                            const threads_total: usize = self.thread_count;
                            const tc0: usize = c_meta.tile_counts[0];

                            // Keep at least ~1 task per thread to avoid underutilization.
                            // (For small problems, this will be 1.)
                            const max_grain_for_parallelism: usize = @max(@as(usize, 1), total_work / threads_total);

                            var grain: usize = @min(@min(tc0, max_grain_for_parallelism), @as(usize, 32));
                            while (grain > 1 and (tc0 % grain) != 0) : (grain -= 1) {}

                            p.parallelForAny(@ptrCast(&task), total_work, grain, Task.runTiles);
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

                                const a_dt: DType = a_view.dtype;
                                const b_dt: DType = b_view.dtype;
                                const c_dt: DType = c_view.dtype;

                                const k_tile: usize = a_view.layout.shape[1];
                                const beta_tile: f32 = if (ti_k == 0) s.beta else 1.0;
                                const params: MatMulParams = .{ .m = m_tile, .n = n_tile, .k = k_tile, .alpha = s.alpha, .beta = beta_tile };

                                switch (b_dt) {
                                    .f32 => {
                                        if (a_dt != .f32 or c_dt != .f32) return BackendError.InvalidArgument;
                                        if (m_tile == 1) {
                                            try matvec_k.matvecF32(params, c_view.bytes, a_view.bytes, b_view.bytes);
                                        } else {
                                            // Use tuned packed-B kernel with B packing per tile.
                                            // This keeps Program execution working without the generic f32 kernel.
                                            if (k_tile > self.matmul_f32.tuning.kc or n_tile > self.matmul_f32.tuning.nc) return BackendError.InvalidArgument;

                                            const tid: usize = 0;
                                            var scratch_buf: []align(32) u8 = if (self.matmul_scratch_f32.len != 0) self.matmul_scratch_f32[0] else blk: {
                                                const tmp = self.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), self.matmul_f32.scratch_bytes) catch return BackendError.ExecutionFailed;
                                                break :blk tmp;
                                            };
                                            defer if (self.matmul_scratch_f32.len == 0) self.allocator.free(scratch_buf);

                                            try self.matmul_f32.pack_b_tile(scratch_buf, k_tile, n_tile, b_view.bytes);
                                            const pb_f32_len: usize = self.matmul_f32.tuning.kc * self.matmul_f32.tuning.nc;
                                            const pb_bytes_len: usize = pb_f32_len * @sizeOf(f32);
                                            const packed_b_view: []align(32) const f32 = @alignCast(std.mem.bytesAsSlice(f32, scratch_buf[0..pb_bytes_len]));
                                            try self.matmul_f32.matmul_packed_b(scratch_buf, packed_b_view, params, c_view.bytes, a_view.bytes);

                                            _ = tid;
                                        }
                                    },
                                    .f16 => {
                                        if (a_dt != .f16) return BackendError.InvalidArgument;
                                        if (c_dt == .f32) {
                                            try matmul_k.matmulF16ToF32(params, c_view.bytes, a_view.bytes, b_view.bytes);
                                        } else if (c_dt == .f16) {
                                            try matmul_k.matmulF16(params, c_view.bytes, a_view.bytes, b_view.bytes);
                                        } else {
                                            return BackendError.InvalidArgument;
                                        }
                                    },
                                    .q4_0 => {
                                        if (a_dt != .f32 or c_dt != .f32) return BackendError.InvalidArgument;
                                        if (m_tile == 1) {
                                            try matvec_k.matvecQ4_0(params, c_view.bytes, a_view.bytes, b_view.bytes);
                                        } else {
                                            try matmul_k.matmulQ4_0(params, c_view.bytes, a_view.bytes, b_view.bytes);
                                        }
                                    },
                                    .q8_0 => {
                                        if (a_dt != .f32 or c_dt != .f32) return BackendError.InvalidArgument;
                                        if (m_tile == 1) {
                                            try matvec_k.matvecQ8_0(params, c_view.bytes, a_view.bytes, b_view.bytes);
                                        } else {
                                            try matmul_k.matmulQ8_0(params, c_view.bytes, a_view.bytes, b_view.bytes);
                                        }
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
};
