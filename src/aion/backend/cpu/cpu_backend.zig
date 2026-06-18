// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const backend_mod = @import("../backend.zig");
const types = @import("../types.zig");
const matmul_registry = @import("registry/matmul_registry.zig");
const quant_matmul_registry = @import("registry/quant_matmul_registry.zig");
const cpu_target = @import("registry/cpu_target.zig");
const matmul_nt_registry = @import("registry/matmul_nt_registry.zig");
const matvec_registry = @import("registry/matvec_registry.zig");
const attention_registry = @import("registry/attention_registry.zig");
const conv1d_registry = @import("registry/conv1d_registry.zig");
const conv2d_registry = @import("registry/conv2d_registry.zig");
const fft_registry = @import("registry/fft_registry.zig");
const fft_kernels = @import("kernels/fft.zig");
const exec_utils = @import("exec/utils.zig");
const exec_elemwise = @import("exec/elementwise.zig");
const exec_unary = @import("exec/unary.zig");
const exec_matmul = @import("exec/matmul.zig");
const exec_softmax = @import("exec/softmax.zig");
const exec_conv = @import("exec/conv.zig");
const exec_layernorm = @import("exec/layernorm.zig");
const exec_attention = @import("exec/attention.zig");
const exec_attention_cached = @import("exec/attention_cached.zig");
const exec_lstm = @import("exec/lstm.zig");
const exec_rfft = @import("exec/rfft.zig");
const exec_stft = @import("exec/stft.zig");
const exec_gather = @import("exec/gather.zig");
const exec_rope = @import("exec/rope.zig");
const exec_kv_cache_append = @import("exec/kv_cache_append.zig");
const exec_cast = @import("exec/cast.zig");
const exec_matmul_nt = @import("exec/matmul_nt.zig");
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

    /// Cached env-controlled tracing/profile toggles (read once at init).
    trace_exec: bool = false,
    profile_steps_env: bool = false,

    // Per-thread scratch for reductions (sum/mean). Size == thread_count.
    reduce_scratch_f32: []f32 = &[_]f32{},

    // Per-thread scratch for softmax reductions (max + sum). Size == 2*thread_count.
    softmax_scratch_f32: []f32 = &[_]f32{},

    matmul_f32: matmul_registry.F32Kernels = matmul_registry.candidates[1].kernels,

    matmul_qx0: quant_matmul_registry.QuantKernels = quant_matmul_registry.candidates[1].kernels,

    matmul_nt: matmul_nt_registry.Kernels = matmul_nt_registry.candidates[0].kernels,

    matvec: matvec_registry.Kernels = matvec_registry.candidates[0].kernels,

    attention_kernels: attention_registry.Kernels = attention_registry.candidates[0].kernels,

    depthwise_conv1d: conv1d_registry.Kernels = conv1d_registry.candidates[0].kernels,
    depthwise_conv2d: conv2d_registry.Kernels = conv2d_registry.candidates[0].kernels,

    fft: fft_registry.FftKernels = fft_registry.candidates[1].kernels,

    // Cached real-FFT plan (bit-reversal + twiddles), reused across RFFT/STFT
    // steps with the same n_fft to keep plan build off the hot path.
    fft_plan: ?fft_kernels.Plan = null,

    // Per-thread scratch for matmul packing (A/B panels).
    // Size == thread_count. Avoids large per-call stack frames.
    matmul_scratch_f32: [][]align(32) u8 = &[_][]align(32) u8{},

    // Future: thread pool handle, SIMD feature flags, scratch buffers

    const Self = @This();

    fn traceEnabled() bool {
        return envFlagEnabled("AION_TRACE");
    }

    fn profileStepsEnabled() bool {
        return envFlagEnabled("AION_PROFILE_STEPS");
    }

    fn envFlagEnabled(name: []const u8) bool {
        const env: std.process.Environ = .{ .block = .global };
        const raw: []u8 = std.process.Environ.getAlloc(env, std.heap.page_allocator, name) catch return false;
        defer std.heap.page_allocator.free(raw);
        const trimmed: []const u8 = std.mem.trim(u8, raw, " \t\r\n");
        if (trimmed.len == 0) return false;
        if (std.mem.eql(u8, trimmed, "0")) return false;
        if (std.mem.eql(u8, trimmed, "false")) return false;
        return true;
    }

    pub const Options = struct {
        /// Total threads to use including the calling thread.
        /// Set to 1 to disable parallelism (default).
        thread_count: usize = 1,

        /// Print per-step timing from `executeProgram`.
        profile_steps: bool = false,
    };

    pub fn init(allocator: std.mem.Allocator) Self {
        return initWithOptions(allocator, .{ .thread_count = 1, .profile_steps = false }) catch |e| {
            std.debug.panic("CpuBackend.init failed: {s}", .{@errorName(e)});
        };
    }

    pub fn initWithOptions(allocator: std.mem.Allocator, opts: Options) !Self {
        if (opts.thread_count == 0) return error.InvalidArgument;

        const hw_threads_raw: usize = std.Thread.getCpuCount() catch opts.thread_count;
        const hw_threads: usize = if (hw_threads_raw == 0) opts.thread_count else hw_threads_raw;
        const topo_info = cpuid.detect();
        const target = cpu_target.fromCpuInfo(topo_info);
        // Respect user-requested thread count up to the system's logical CPUs.
        // For decode-heavy workloads on hybrid CPUs, SMT threads can still help,
        // and higher-level tile heuristics decide when to parallelize small ops.
        const effective_thread_count: usize = @max(@as(usize, 1), @min(opts.thread_count, hw_threads));

        var self: Self = .{
            .allocator = allocator,
            .pool = null,
            .thread_count = 1,
            .profile_steps = opts.profile_steps,
            .trace_exec = traceEnabled(),
            .profile_steps_env = profileStepsEnabled(),
            .reduce_scratch_f32 = &[_]f32{},
            .softmax_scratch_f32 = &[_]f32{},
            .matmul_scratch_f32 = &[_][]align(32) u8{},
        };

        // Kernel selection based on CPU (same for single- and multi-threaded).
        self.matmul_f32 = matmul_registry.selectForTarget(target).kernels;
        self.matmul_qx0 = quant_matmul_registry.selectForTarget(target).kernels;
        self.matmul_nt = matmul_nt_registry.selectForTarget(target).kernels;
        self.matvec = matvec_registry.selectForTarget(target).kernels;
        self.attention_kernels = attention_registry.selectForTarget(target).kernels;
        self.depthwise_conv1d = conv1d_registry.selectForTarget(target).kernels;
        self.depthwise_conv2d = conv2d_registry.selectForTarget(target).kernels;
        self.fft = fft_registry.selectForTarget(target).kernels;

        if (effective_thread_count > 1) {
            const p = try thread_pool.ThreadPool.init(allocator, .{ .thread_count = effective_thread_count });
            self.pool = p;
            self.thread_count = effective_thread_count;
        }

        // Allocate reduction + softmax scratch. Sized to thread_count; for single-thread
        // this is still 1 slot (not zero), so ops don't need a per-call fallback.
        self.reduce_scratch_f32 = try allocator.alloc(f32, effective_thread_count);
        errdefer allocator.free(self.reduce_scratch_f32);
        self.softmax_scratch_f32 = try allocator.alloc(f32, effective_thread_count * 2);
        errdefer allocator.free(self.softmax_scratch_f32);

        // Allocate matmul scratch once per thread. Avoids per-call alloc/free in the
        // sequential fallback (single-thread mode was previously hitting alignedAlloc/free
        // on every matmul call, a substantial overhead for Q8 decode).
        var mm: [][]align(32) u8 = try allocator.alloc([]align(32) u8, effective_thread_count);
        errdefer allocator.free(mm);
        var i: usize = 0;
        errdefer {
            var j: usize = 0;
            while (j < i) : (j += 1) allocator.free(mm[j]);
        }
        const scratch_bytes: usize = @max(matmul_registry.maxScratchBytes(), quant_matmul_registry.maxScratchBytes());
        while (i < effective_thread_count) : (i += 1) {
            mm[i] = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(32), scratch_bytes);
        }
        self.matmul_scratch_f32 = mm;

        return self;
    }

    /// Build-or-reuse the cached real-FFT plan for `n_fft`.
    fn fftPlanFor(self: *Self, n_fft: usize) ExecuteProgramError!*const fft_kernels.Plan {
        if (self.fft_plan) |*p| {
            if (p.n_fft == n_fft) return p;
            p.deinit();
            self.fft_plan = null;
        }
        self.fft_plan = fft_kernels.Plan.init(self.allocator, n_fft) catch |e| switch (e) {
            error.OutOfMemory => return BackendError.ExecutionFailed,
            else => return BackendError.InvalidArgument,
        };
        return &self.fft_plan.?;
    }

    pub fn deinit(self: *Self) void {
        if (self.pool) |*p| {
            p.deinit();
            self.pool = null;
        }

        if (self.fft_plan) |*p| {
            p.deinit();
            self.fft_plan = null;
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

    fn readI32Scalar(store: tensor_store.TensorStore, id: executable.TensorId) ExecuteProgramError!i32 {
        const meta: tensor_store.TensorMeta = try store.meta(id);
        if (meta.dtype != .i32 or meta.rank != 1 or meta.shape.len != 1 or meta.shape[0] != 1) return error.InvalidArgument;
        const tile: tensor_store.TileRefConst = try store.acquireTileConstLinear(id, 0);
        defer store.releaseConst(tile.token);
        if (tile.bytes.len < @sizeOf(i32)) return error.InvalidArgument;
        return std.mem.readInt(i32, tile.bytes[0..@sizeOf(i32)], .little);
    }

    fn readPredicate(store: tensor_store.TensorStore, id: executable.TensorId) ExecuteProgramError!bool {
        return (try readI32Scalar(store, id)) != 0;
    }

    fn readUsizeScalar(store: tensor_store.TensorStore, id: executable.TensorId) ExecuteProgramError!usize {
        const raw: i32 = try readI32Scalar(store, id);
        if (raw < 0) return error.InvalidArgument;
        return @intCast(raw);
    }

    fn totalTileCount(meta: tensor_store.TensorMeta) ExecuteProgramError!usize {
        var total: usize = 1;
        for (meta.tile_counts) |count| {
            total = std.math.mul(usize, total, count) catch return error.InvalidArgument;
        }
        return total;
    }

    fn copyTensorSameLayout(store: tensor_store.TensorStore, dst: executable.TensorId, src: executable.TensorId) ExecuteProgramError!void {
        const dst_meta: tensor_store.TensorMeta = try store.meta(dst);
        const src_meta: tensor_store.TensorMeta = try store.meta(src);
        if (dst_meta.dtype != src_meta.dtype or dst_meta.rank != src_meta.rank) return error.InvalidArgument;
        if (dst_meta.shape.len != src_meta.shape.len or dst_meta.tile_shape.len != src_meta.tile_shape.len or dst_meta.tile_counts.len != src_meta.tile_counts.len) return error.InvalidArgument;
        for (dst_meta.shape, 0..) |v, i| if (v != src_meta.shape[i]) return error.InvalidArgument;
        for (dst_meta.tile_shape, 0..) |v, i| if (v != src_meta.tile_shape[i]) return error.InvalidArgument;
        for (dst_meta.tile_counts, 0..) |v, i| if (v != src_meta.tile_counts[i]) return error.InvalidArgument;

        const tile_count: usize = try totalTileCount(dst_meta);
        var tile_index: usize = 0;
        while (tile_index < tile_count) : (tile_index += 1) {
            const src_tile: tensor_store.TileRefConst = try store.acquireTileConstLinear(src, tile_index);
            defer store.releaseConst(src_tile.token);
            const dst_tile: tensor_store.TileRefMut = try store.acquireTileMutLinear(dst, tile_index);
            defer store.releaseMut(dst_tile.token);
            if (dst_tile.bytes.len != src_tile.bytes.len) return error.InvalidArgument;
            @memcpy(dst_tile.bytes, src_tile.bytes);
        }
    }

    fn copyTensorLists(store: tensor_store.TensorStore, dst: []const executable.TensorId, src: []const executable.TensorId) ExecuteProgramError!void {
        if (dst.len != src.len) return error.InvalidArgument;
        for (dst, 0..) |dst_id, i| {
            try copyTensorSameLayout(store, dst_id, src[i]);
        }
    }

    fn swapTensorLists(store: tensor_store.TensorStore, a: []const executable.TensorId, b: []const executable.TensorId) ExecuteProgramError!void {
        if (a.len != b.len) return error.InvalidArgument;
        for (a, 0..) |a_id, i| {
            if (a_id == b[i]) continue;
            try store.swapTensors(a_id, b[i]);
        }
    }

    fn execBlock(self: *Self, prog: *const executable.ExecutableProgram, block_id: executable.BlockId, store: tensor_store.TensorStore) ExecuteProgramError!void {
        const idx: usize = @intCast(block_id);
        if (idx >= prog.blocks.len) return error.InvalidArgument;
        for (prog.blocks[idx].steps) |block_step| {
            try self.execStep(prog, block_step, store);
        }
    }

    fn execStep(self: *Self, prog: *const executable.ExecutableProgram, step: executable.Step, store: tensor_store.TensorStore) ExecuteProgramError!void {
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

            .MultiHeadAttentionCachedTiled => |s| {
                const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                try exec_attention_cached.execMultiHeadAttentionCachedTiled(pool_ptr, self.thread_count, self.attention_kernels, s, store);
            },

            .CopyTiled => |s| {
                const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                try ElemwiseExec.execCopyTiled(pool_ptr, self.thread_count, s, store);
            },

            .LSTMCellFused => |s| {
                const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                try exec_lstm.execLSTMCellFused(pool_ptr, self.thread_count, s, store);
            },

            .RFFT => |s| {
                const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                const plan = try self.fftPlanFor(s.n_fft);
                try exec_rfft.execRFFT(self.allocator, pool_ptr, self.thread_count, self.fft, plan, s, store);
            },

            .STFT => |s| {
                const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                const plan = try self.fftPlanFor(s.n_fft);
                try exec_stft.execSTFT(self.allocator, pool_ptr, self.thread_count, self.fft, plan, s, store);
            },

            .If => |s| {
                const take_then: bool = try readPredicate(store, s.cond);
                const count: usize = @intCast(s.output_count);
                if (count > executable.MAX_CONTROL_OUTPUTS) return error.InvalidArgument;
                if (take_then) {
                    try self.execBlock(prog, s.then_block, store);
                    try copyTensorLists(store, s.outputs[0..count], s.then_outputs[0..count]);
                } else {
                    try self.execBlock(prog, s.else_block, store);
                    try copyTensorLists(store, s.outputs[0..count], s.else_outputs[0..count]);
                }
            },

            .Loop => |s| {
                const carried_count: usize = @intCast(s.carried_count);
                if (carried_count > executable.MAX_LOOP_CARRIED) return error.InvalidArgument;
                const requested_iters: usize = if (s.trip_count) |trip_id| try readUsizeScalar(store, trip_id) else s.static_max_trip_count;
                const max_iters: usize = @min(requested_iters, s.static_max_trip_count);

                var iter: usize = 0;
                while (iter < max_iters) : (iter += 1) {
                    if (s.check_before) {
                        if (s.cond) |cond_id| {
                            if (!try readPredicate(store, cond_id)) break;
                        }
                    }

                    try self.execBlock(prog, s.body_block, store);
                    try swapTensorLists(store, s.carried[0..carried_count], s.body_carried_outputs[0..carried_count]);

                    if (!s.check_before) {
                        if (s.cond) |cond_id| {
                            if (!try readPredicate(store, cond_id)) break;
                        }
                    }
                }
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

            .KVCacheAppendTiled => |s| {
                const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                try exec_kv_cache_append.execKVCacheAppendTiled(pool_ptr, self.thread_count, s, store);
            },

            .CastTiled => |s| {
                const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                try exec_cast.execCastTiled(pool_ptr, self.thread_count, s, store);
            },

            .MatMulNTTiled => |s| {
                const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                const nt_ctx: exec_matmul_nt.MatMulNtExecCtx = .{
                    .matmul_nt = self.matmul_nt,
                };
                try exec_matmul_nt.execMatMulNTTiled(&nt_ctx, pool_ptr, self.thread_count, s, store);
            },
        }
    }

    fn executeProgramImpl(ctx: *anyopaque, prog: *const executable.ExecutableProgram, store: tensor_store.TensorStore) ExecuteProgramError!void {
        const self: *Self = @ptrCast(@alignCast(ctx));

        const do_profile: bool = self.profile_steps or self.profile_steps_env;

        const nowNs = struct {
            fn f() u64 {
                const ts: std.Io.Timestamp = std.Io.Clock.awake.now(std.Options.debug_io);
                const ns: i96 = ts.toNanoseconds();
                if (ns <= 0) return 0;
                const max_u64_i96: i96 = @as(i96, std.math.maxInt(u64));
                return @intCast(@min(ns, max_u64_i96));
            }
        }.f;

        // Per-step-kind aggregation: when profiling is on, we track cumulative time +
        // call count for each op kind across this one executeProgram call. That surfaces
        // the real cost drivers (e.g. a 400-call MatMul total) rather than drowning the
        // user in 1600 individual per-step printouts.
        const Step = executable.Step;
        const kind_count: usize = @typeInfo(Step).@"union".fields.len;
        var totals_ns: [kind_count]u64 = .{0} ** kind_count;
        var counts: [kind_count]u64 = .{0} ** kind_count;
        const trace_exec: bool = self.trace_exec;

        const run_t0: u64 = if (do_profile) nowNs() else 0;
        for (prog.steps, 0..) |step, step_i| {
            const t0: u64 = if (do_profile) nowNs() else 0;
            if (trace_exec) {
                std.debug.print("[aion][exec] step {d}/{d}: {s}\n", .{ step_i, prog.steps.len, @tagName(step) });
            }
            self.execStep(prog, step, store) catch |e| {
                if (trace_exec) {
                    std.debug.print("[aion][exec] step {d} failed: {s} err={s}\n", .{ step_i, @tagName(step), @errorName(e) });
                }
                return e;
            };

            if (do_profile) {
                const t1: u64 = nowNs();
                const dt_ns: u64 = t1 - t0;
                const idx: usize = @intFromEnum(std.meta.activeTag(step));
                if (idx < kind_count) {
                    totals_ns[idx] += dt_ns;
                    counts[idx] += 1;
                }
            }
        }

        if (do_profile) {
            const run_total_ns: u64 = nowNs() - run_t0;
            std.debug.print("\n[aion][profile] program steps={d}  total={d:.3} ms\n", .{ prog.steps.len, @as(f64, @floatFromInt(run_total_ns)) / 1.0e6 });
            std.debug.print("[aion][profile] per-op-kind (sorted by cumulative ms):\n", .{});

            // Sort indices by descending cumulative ns so the worst offenders print first.
            var order: [kind_count]usize = undefined;
            for (&order, 0..) |*o, i| o.* = i;
            // Simple insertion sort (kind_count is tiny).
            var i: usize = 1;
            while (i < kind_count) : (i += 1) {
                var j: usize = i;
                while (j > 0 and totals_ns[order[j]] > totals_ns[order[j - 1]]) : (j -= 1) {
                    const tmp = order[j];
                    order[j] = order[j - 1];
                    order[j - 1] = tmp;
                }
            }

            const field_names = comptime blk: {
                const fields = @typeInfo(Step).@"union".fields;
                var names: [fields.len][]const u8 = undefined;
                for (fields, 0..) |f, fi| names[fi] = f.name;
                break :blk names;
            };

            for (order) |idx| {
                if (counts[idx] == 0) continue;
                const ms: f64 = @as(f64, @floatFromInt(totals_ns[idx])) / 1.0e6;
                const avg_us: f64 = (@as(f64, @floatFromInt(totals_ns[idx])) / 1.0e3) / @as(f64, @floatFromInt(counts[idx]));
                std.debug.print("  {s:<32} {d:>8.3} ms  ({d:>6} calls, avg {d:>8.1} us)\n", .{ field_names[idx], ms, counts[idx], avg_us });
            }
        }

        // Avoid spamming across many runs: profile only the first execution per backend.
        if (self.profile_steps) self.profile_steps = false;
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
