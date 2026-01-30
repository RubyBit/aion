const std = @import("std");
const backend_mod = @import("../backend.zig");
const types = @import("../types.zig");
const matmul_registry = @import("kernels/matmul_registry.zig");
const quant_matmul_registry = @import("kernels/quant_matmul_registry.zig");
const matvec_registry = @import("kernels/matvec_registry.zig");
const exec_utils = @import("exec/utils.zig");
const exec_elemwise = @import("exec/elementwise.zig");
const exec_unary = @import("exec/unary.zig");
const exec_matmul = @import("exec/matmul.zig");
const exec_softmax = @import("exec/softmax.zig");
const thread_pool = @import("../../runtime/thread_pool.zig");
const executable = @import("../../runtime/executable.zig");
const cpuid = @import("tuning/cpuid.zig");
const tensor_store = @import("../../runtime/tensor_store.zig");

const Backend = backend_mod.Backend;
const BackendKind = types.BackendKind;
const BackendCaps = types.BackendCaps;
const BackendError = types.BackendError;
const DType = types.DType;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

/// CPU Backend implementation.
/// Owns threading strategy (v0: single-threaded, later: thread pool).
pub const CpuBackend = struct {
    allocator: std.mem.Allocator,

    pool: ?thread_pool.ThreadPool = null,
    thread_count: usize = 1,

    // Per-thread scratch for reductions (sum/mean). Size == thread_count.
    reduce_scratch_f32: []f32 = &[_]f32{},

    // Per-thread scratch for softmax reductions (max + sum). Size == 2*thread_count.
    softmax_scratch_f32: []f32 = &[_]f32{},

    matmul_f32: matmul_registry.F32Kernels = matmul_registry.candidates[1].kernels,

    matmul_qx0: quant_matmul_registry.QuantKernels = quant_matmul_registry.candidates[1].kernels,

    matvec: matvec_registry.Kernels = matvec_registry.candidates[0].kernels,

    // Per-thread scratch for matmul packing (A/B panels).
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
        return .{ .allocator = allocator, .pool = null, .thread_count = 1, .reduce_scratch_f32 = &[_]f32{}, .softmax_scratch_f32 = &[_]f32{}, .matmul_scratch_f32 = &[_][]align(32) u8{} };
    }

    pub fn initWithOptions(allocator: std.mem.Allocator, opts: Options) !Self {
        if (opts.thread_count == 0) return error.InvalidArgument;

        var self: Self = .{ .allocator = allocator, .pool = null, .thread_count = 1, .reduce_scratch_f32 = &[_]f32{}, .softmax_scratch_f32 = &[_]f32{}, .matmul_scratch_f32 = &[_][]align(32) u8{} };
        if (opts.thread_count > 1) {
            const p = try thread_pool.ThreadPool.init(allocator, .{ .thread_count = opts.thread_count });
            self.pool = p;
            self.thread_count = opts.thread_count;

            // Allocate reduction scratch once. No allocations during op execution.
            self.reduce_scratch_f32 = try allocator.alloc(f32, opts.thread_count);

            // Allocate softmax reduction scratch once. No allocations during op execution.
            self.softmax_scratch_f32 = try allocator.alloc(f32, opts.thread_count * 2);

            // Allocate matmul scratch once. No allocations during op execution.
            // Select a variant based on CPU cache sizes (when available).
            const cpu_info = cpuid.detect();
            const mm_choice = matmul_registry.selectHeuristic(cpu_info);
            self.matmul_f32 = mm_choice.kernels;

            const qm_choice = quant_matmul_registry.selectHeuristic(cpu_info);
            self.matmul_qx0 = qm_choice.kernels;

            const mv_choice = matvec_registry.selectHeuristic(cpu_info);
            self.matvec = mv_choice.kernels;

            var mm: [][]align(32) u8 = try allocator.alloc([]align(32) u8, opts.thread_count);
            errdefer allocator.free(mm);
            var i: usize = 0;
            errdefer {
                var j: usize = 0;
                while (j < i) : (j += 1) allocator.free(mm[j]);
            }
            const scratch_bytes: usize = @max(self.matmul_f32.scratch_bytes, quant_matmul_registry.maxScratchBytes());
            while (i < opts.thread_count) : (i += 1) {
                mm[i] = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), scratch_bytes);
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

        if (self.softmax_scratch_f32.len != 0) {
            self.allocator.free(self.softmax_scratch_f32);
            self.softmax_scratch_f32 = &[_]f32{};
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

    const ElemwiseExec = exec_elemwise;

    fn executeProgramImpl(ctx: *anyopaque, prog: *const executable.ExecutableProgram, store: tensor_store.TensorStore) ExecuteProgramError!void {
        const self: *Self = @ptrCast(@alignCast(ctx));

        for (prog.steps) |step| {
            switch (step) {
                .MatMulTiled => |s| {
                    const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                    var mm_ctx: exec_matmul.MatMulExecCtx = .{
                        .allocator = self.allocator,
                        .pool = pool_ptr,
                        .thread_count = self.thread_count,
                        .matmul_f32 = self.matmul_f32,
                        .matmul_qx0 = self.matmul_qx0,
                        .matvec = self.matvec,
                        .matmul_scratch = self.matmul_scratch_f32,
                    };
                    try exec_matmul.execMatMulTiled(&mm_ctx, s, store);
                },

                .ElemwiseBinaryTiled => |s| {
                    const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                    try ElemwiseExec.execElemwiseBinaryTiled(pool_ptr, self.thread_count, s, store);
                },

                .BroadcastLastDimBinaryTiled => |s| {
                    const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                    try ElemwiseExec.execBroadcastLastDimBinaryTiled(pool_ptr, self.thread_count, s, store);
                },

                .UnaryTiled => |s| {
                    const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                    try exec_unary.execUnaryTiled(pool_ptr, self.thread_count, s, store);
                },

                .SoftmaxTiled => |s| {
                    const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                    try exec_softmax.execSoftmaxTiled(pool_ptr, self.thread_count, self.softmax_scratch_f32, s, store);
                },

                .CopyTiled => |s| {
                    const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                    try ElemwiseExec.execCopyTiled(pool_ptr, self.thread_count, s, store);
                },

                .ReduceAll => |s| {
                    const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                    try exec_utils.reduceAllScalar(pool_ptr, self.thread_count, self.reduce_scratch_f32, s.op, s.out, s.a, store);
                },

                .ReTileCopyScalar => |s| {
                    try exec_utils.retileCopyScalar(store, s.dst, s.src);
                },
                .ReshapeScalar => |s| {
                    try exec_utils.reshapeCopyScalar(store, s.dst, s.src);
                },
                .Transpose2DScalar => |s| {
                    try exec_utils.transpose2DCopyScalar(store, s.dst, s.src);
                },
                .Slice2DScalar => |s| {
                    try exec_utils.slice2DCopyScalar(store, s.dst, s.src, s.start0, s.start1);
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
