const std = @import("std");
const backend_mod = @import("../backend.zig");
const types = @import("../types.zig");
const utils = @import("../utils.zig");
const elemwise = @import("kernels/elemwise.zig");
const matmul_k = @import("kernels/matmul.zig");
const reduce_k = @import("kernels/reduce.zig");
const simd = @import("kernels/simd.zig");
const thread_pool = @import("../../runtime/thread_pool.zig");

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
    };

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

        const do_parallel: bool = (self.thread_count > 1) and (self.pool != null) and (self.reduce_scratch_f32.len >= self.thread_count);
        if (do_parallel) {
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

            // Heuristic: ~256KiB of input per thread-chunk.
            const elem_bytes: usize = switch (a.dtype) {
                .f32 => @as(usize, 4),
                .f16 => @as(usize, 2),
                else => @as(usize, 0),
            };
            if (elem_bytes == 0) return BackendError.InvalidArgument;
            const target_chunk_bytes: usize = 256 * 1024;
            var grain_elems: usize = target_chunk_bytes / elem_bytes;
            if (grain_elems < 8192) grain_elems = 8192;

            self.pool.?.parallelForAny(@ptrCast(&task), elem_count, grain_elems, Task.runChunk);

            var i: usize = 0;
            while (i < self.thread_count) : (i += 1) {
                sum += self.reduce_scratch_f32[i];
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
