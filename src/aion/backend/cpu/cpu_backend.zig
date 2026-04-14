const std = @import("std");
const backend_mod = @import("../backend.zig");
const types = @import("../types.zig");
const matmul_registry = @import("registry/matmul_registry.zig");
const quant_matmul_registry = @import("registry/quant_matmul_registry.zig");
const matvec_registry = @import("registry/matvec_registry.zig");
const attention_registry = @import("registry/attention_registry.zig");
const conv1d_registry = @import("registry/conv1d_registry.zig");
const conv2d_registry = @import("registry/conv2d_registry.zig");
const exec_utils = @import("exec/utils.zig");
const exec_elemwise = @import("exec/elementwise.zig");
const exec_unary = @import("exec/unary.zig");
const exec_matmul = @import("exec/matmul.zig");
const exec_softmax = @import("exec/softmax.zig");
const exec_conv = @import("exec/conv.zig");
const exec_layernorm = @import("exec/layernorm.zig");
const exec_attention = @import("exec/attention.zig");
const exec_lstm = @import("exec/lstm.zig");
const exec_complex_abs_mean = @import("exec/complex_abs_mean.zig");
const exec_gather = @import("exec/gather.zig");
const exec_rope = @import("exec/rope.zig");
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

    /// When enabled, print per-step execution timing from `executeProgram`.
    /// Intended for coarse profiling/debugging; adds overhead.
    profile_steps: bool = false,

    // Per-thread scratch for reductions (sum/mean). Size == thread_count.
    reduce_scratch_f32: []f32 = &[_]f32{},

    // Per-thread scratch for softmax reductions (max + sum). Size == 2*thread_count.
    softmax_scratch_f32: []f32 = &[_]f32{},

    matmul_f32: matmul_registry.F32Kernels = matmul_registry.candidates[1].kernels,

    matmul_qx0: quant_matmul_registry.QuantKernels = quant_matmul_registry.candidates[1].kernels,

    matvec: matvec_registry.Kernels = matvec_registry.candidates[0].kernels,

    attention_kernels: attention_registry.Kernels = attention_registry.candidates[0].kernels,

    depthwise_conv1d: conv1d_registry.Kernels = conv1d_registry.candidates[0].kernels,
    depthwise_conv2d: conv2d_registry.Kernels = conv2d_registry.candidates[0].kernels,

    // Per-thread scratch for matmul packing (A/B panels).
    // Size == thread_count. Avoids large per-call stack frames.
    matmul_scratch_f32: [][]align(32) u8 = &[_][]align(32) u8{},

    // Future: thread pool handle, SIMD feature flags, scratch buffers

    const Self = @This();

    pub const Options = struct {
        /// Total threads to use including the calling thread.
        /// Set to 1 to disable parallelism (default).
        thread_count: usize = 1,

        /// Print per-step timing from `executeProgram`.
        profile_steps: bool = false,
    };

    pub fn init(allocator: std.mem.Allocator) Self {
        const cpu_info = cpuid.detect();
        const mm_choice: matmul_registry.Candidate = matmul_registry.selectHeuristic(cpu_info);
        const qm_choice: quant_matmul_registry.Candidate = quant_matmul_registry.selectHeuristic(cpu_info);
        const mv_choice: matvec_registry.Candidate = matvec_registry.selectHeuristic(cpu_info);
        const attn_choice: attention_registry.Candidate = attention_registry.selectHeuristic(cpu_info);
        const conv1d_choice: conv1d_registry.Candidate = conv1d_registry.selectHeuristic(cpu_info);
        const conv2d_choice: conv2d_registry.Candidate = conv2d_registry.selectHeuristic(cpu_info);

        return .{
            .allocator = allocator,
            .pool = null,
            .thread_count = 1,
            .profile_steps = false,
            .reduce_scratch_f32 = &[_]f32{},
            .softmax_scratch_f32 = &[_]f32{},
            .matmul_f32 = mm_choice.kernels,
            .matmul_qx0 = qm_choice.kernels,
            .matvec = mv_choice.kernels,
            .attention_kernels = attn_choice.kernels,
            .depthwise_conv1d = conv1d_choice.kernels,
            .depthwise_conv2d = conv2d_choice.kernels,
            .matmul_scratch_f32 = &[_][]align(32) u8{},
        };
    }

    pub fn initWithOptions(allocator: std.mem.Allocator, opts: Options) !Self {
        if (opts.thread_count == 0) return error.InvalidArgument;

        var self: Self = .{ .allocator = allocator, .pool = null, .thread_count = 1, .profile_steps = opts.profile_steps, .reduce_scratch_f32 = &[_]f32{}, .softmax_scratch_f32 = &[_]f32{}, .matmul_scratch_f32 = &[_][]align(32) u8{} };
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

            const attn_choice = attention_registry.selectHeuristic(cpu_info);
            self.attention_kernels = attn_choice.kernels;

            const conv1d_choice = conv1d_registry.selectHeuristic(cpu_info);
            self.depthwise_conv1d = conv1d_choice.kernels;

            const conv2d_choice = conv2d_registry.selectHeuristic(cpu_info);
            self.depthwise_conv2d = conv2d_choice.kernels;

            var mm: [][]align(32) u8 = try allocator.alloc([]align(32) u8, opts.thread_count);
            errdefer allocator.free(mm);
            var i: usize = 0;
            errdefer {
                var j: usize = 0;
                while (j < i) : (j += 1) allocator.free(mm[j]);
            }
            const scratch_bytes: usize = @max(matmul_registry.maxScratchBytes(), quant_matmul_registry.maxScratchBytes());
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

        const do_profile: bool = self.profile_steps;
        if (do_profile) {
            std.debug.print("cpu_backend: program steps={}\n", .{prog.steps.len});
        }

        const nowNs = struct {
            fn f() u64 {
                const ts: std.Io.Timestamp = std.Io.Clock.awake.now(std.Options.debug_io);
                const ns: i96 = ts.toNanoseconds();
                if (ns <= 0) return 0;
                const max_u64_i96: i96 = @as(i96, std.math.maxInt(u64));
                return @intCast(@min(ns, max_u64_i96));
            }
        }.f;

        for (prog.steps, 0..) |step, step_i| {
            const t0: u64 = if (do_profile) nowNs() else 0;
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

                .Conv1DTiled => |s| {
                    const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                    var conv_ctx: exec_conv.ConvExecCtx = .{
                        .allocator = self.allocator,
                        .pool = pool_ptr,
                        .thread_count = self.thread_count,
                        .matmul_f32 = self.matmul_f32,
                        .depthwise_conv1d = self.depthwise_conv1d,
                        .depthwise_conv2d = self.depthwise_conv2d,
                        .matmul_scratch = self.matmul_scratch_f32,
                    };
                    try exec_conv.execConv1DTiled(&conv_ctx, s, store);
                },

                .Conv2DTiled => |s| {
                    const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                    var conv_ctx: exec_conv.ConvExecCtx = .{
                        .allocator = self.allocator,
                        .pool = pool_ptr,
                        .thread_count = self.thread_count,
                        .matmul_f32 = self.matmul_f32,
                        .depthwise_conv1d = self.depthwise_conv1d,
                        .depthwise_conv2d = self.depthwise_conv2d,
                        .matmul_scratch = self.matmul_scratch_f32,
                    };
                    try exec_conv.execConv2DTiled(&conv_ctx, s, store);
                },

                .LayerNormTiled => |s| {
                    const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                    try exec_layernorm.execLayerNormTiled(pool_ptr, self.thread_count, s, store);
                },

                .RMSNormTiled => |s| {
                    const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                    try exec_layernorm.execRMSNormTiled(pool_ptr, self.thread_count, s, store);
                },

                .AttentionTiled => |s| {
                    const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                    try exec_attention.execAttentionTiled(pool_ptr, self.thread_count, self.attention_kernels, s, store);
                },

                .MultiHeadAttentionTiled => |s| {
                    const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                    try exec_attention.execMultiHeadAttentionTiled(pool_ptr, self.thread_count, self.attention_kernels, s, store);
                },

                .CopyTiled => |s| {
                    const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                    try ElemwiseExec.execCopyTiled(pool_ptr, self.thread_count, s, store);
                },

                .LSTMCellFused => |s| {
                    const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                    try exec_lstm.execLSTMCellFused(pool_ptr, self.thread_count, s, store);
                },

                .ComplexAbsMean => |s| {
                    const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                    try exec_complex_abs_mean.execComplexAbsMean(pool_ptr, self.thread_count, s, store);
                },

                .ReduceAll => |s| {
                    const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                    try exec_utils.reduceAllScalar(pool_ptr, self.thread_count, self.reduce_scratch_f32, s.op, s.out, s.a, store);
                },

                .ReduceAxis => |s| {
                    const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                    try exec_utils.reduceAxisScalar(pool_ptr, self.thread_count, self.reduce_scratch_f32, s.op, s.out, s.a, s.axis, store);
                },

                .ConcatScalar => |s| {
                    try exec_utils.concatScalar(s, store);
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
                .SliceNDScalar => |s| {
                    const rank: usize = @as(usize, s.rank);
                    try exec_utils.sliceNDCopyScalar(store, s.dst, s.src, s.starts[0..rank]);
                },

                .GatherRowsTiled => |s| {
                    const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                    try exec_gather.execGatherRowsTiled(pool_ptr, self.thread_count, s, store);
                },

                .RoPE1DTiled => |s| {
                    const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                    try exec_rope.execRoPE1DTiled(pool_ptr, self.thread_count, s, store);
                },
            }

            if (do_profile) {
                const t1: u64 = nowNs();
                const dt_us: u64 = (t1 - t0) / 1000;
                std.debug.print("  step {d:3}: {s}  {d} us\n", .{ step_i, @tagName(step), dt_us });
            }
        }

        // Avoid spamming during benchmarks: if enabled, profile only the first execution.
        if (do_profile) self.profile_steps = false;
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
