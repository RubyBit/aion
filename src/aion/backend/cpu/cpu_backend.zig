// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const builtin = @import("builtin");
const backend_mod = @import("../backend.zig");
const types = @import("../types.zig");
const dispatch_table = @import("multiversion/table.zig");
const kernel_dispatch = @import("multiversion/dispatch.zig");
const env = @import("../../env.zig");
const matmul_registry = @import("registry/matmul_registry.zig");
const matmul_q_registry = @import("registry/matmul_q_registry.zig");
const cpu_target = @import("registry/cpu_target.zig");
const matmul_nt_registry = @import("registry/matmul_nt_registry.zig");
const matvec_registry = @import("registry/matvec_registry.zig");
const attention_registry = @import("registry/attention_registry.zig");
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
const exec_relpos_mha = @import("exec/relpos_mha.zig");
const exec_argmax = @import("exec/argmax.zig");
const exec_scatter = @import("exec/scatter.zig");
const exec_topk = @import("exec/topk.zig");
const exec_lstm = @import("exec/lstm.zig");
const exec_rfft = @import("exec/rfft.zig");
const exec_stft = @import("exec/stft.zig");
const exec_gather = @import("exec/gather.zig");
const exec_rope = @import("exec/rope.zig");
const exec_sequence_append = @import("exec/sequence_append.zig");
const exec_cast = @import("exec/cast.zig");
const exec_matmul_nt = @import("exec/matmul_nt.zig");
const thread_pool = @import("../../runtime/thread_pool.zig");
const executable = @import("../../runtime/executable.zig");
const diagnostic = @import("../../diagnostic.zig");
const cpuid = @import("tuning/cpuid.zig");
const tensor_store = @import("../../runtime/tensor_store.zig");
const profile = @import("../../profile.zig");

const Backend = backend_mod.Backend;
const Session = backend_mod.Session;
const BackendKind = types.BackendKind;
const BackendCaps = types.BackendCaps;
const BackendError = types.BackendError;
const DType = types.DType;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

/// True when built with `-Dmultiversion=true` (the default): the per-ISA tier
/// kernel objects are attached to the `aion` module by `build.zig`, every
/// artifact that imports the module links them transitively, and runtime CPUID
/// dispatch selects among them here.
///
/// When false (`-Dmultiversion=false`, or the raw `zig test` invocation, whose
/// build_options module is passed on the CLI with multiversion=false) the
/// in-module `selectForTarget` path is used and the `extern` tier accessors
/// are never referenced — so no tier objects need to be linked.
/// The single-target build's quantized kernels: those of the ISA it was compiled for.
const compiled_quantized = dispatch_table.quantizedFor(cpu_target.compiled);

const multiversion_enabled: bool = (builtin.cpu.arch.isX86() or builtin.cpu.arch.isAARCH64()) and
    @import("build_options").multiversion;

/// CPU Backend implementation.
/// Owns threading strategy (v0: single-threaded, later: thread pool).
pub const CpuBackend = struct {
    allocator: std.mem.Allocator,

    /// Shared L2 per core cluster as detected at init, 0 when unknown.
    l2_bytes: usize = 0,
    /// L1D as detected at init (the smallest core's), 0 when unknown.
    l1d_bytes: usize = 0,

    pool: ?thread_pool.ThreadPool = null,
    thread_count: usize = 1,

    /// Cached environment-controlled diagnostics (read once at init).
    trace_exec: bool = false,
    profile_config: profile.Config = .{},

    /// Internal counter used only to apply the profiler's skip/count window.
    profile_invocations: u64 = 0,

    // Per-thread scratch for reductions (sum/mean). Size == thread_count.
    reduce_scratch_f32: []f32 = &[_]f32{},

    matmul_f32: matmul_registry.F32Kernels = matmul_registry.candidates[1].kernels,

    /// The packed GEMM for quantized weights, and the family it falls back within.
    matmul_q: matmul_q_registry.Choice = .of(compiled_quantized.gemm, 0),

    matmul_nt: matmul_nt_registry.Kernels = compiled_quantized.nt,

    matvec: matvec_registry.Kernels = compiled_quantized.matvec,

    attention_kernels: attention_registry.Kernels = attention_registry.candidates[0].kernels,
    relpos_mha_kernels: attention_registry.Kernels = attention_registry.candidates[0].kernels,

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

    fn envFlagEnabled(name: [:0]const u8) bool {
        return env.flagEnabled(name);
    }

    pub const Options = struct {
        /// Total threads to use including the calling thread.
        /// Set to 1 to disable parallelism (default).
        thread_count: usize = 1,
    };

    pub fn init(allocator: std.mem.Allocator) Self {
        return initWithOptions(allocator, .{ .thread_count = 1 }) catch |e| {
            std.debug.panic("CpuBackend.init failed: {s}", .{@errorName(e)});
        };
    }

    /// The row grouping this backend's NT q8 kernel reads (see
    /// `types.QuantBlockOrder`): its byte dot's lane count.
    pub fn quantBlockOrder(self: *const Self) types.QuantBlockOrder {
        return types.QuantBlockOrder.withGroup(self.matmul_nt.tuning.lanes) orelse .row_major;
    }

    pub fn initWithOptions(allocator: std.mem.Allocator, opts: Options) !Self {
        if (opts.thread_count == 0) return error.InvalidArgument;

        const hw_threads_raw: usize = std.Thread.getCpuCount() catch opts.thread_count;
        const hw_threads: usize = if (hw_threads_raw == 0) opts.thread_count else hw_threads_raw;
        const topo_info = cpuid.detect();
        // Respect user-requested thread count up to the system's logical CPUs.
        // For decode-heavy workloads on hybrid CPUs, SMT threads can still help,
        // and higher-level view heuristics decide when to parallelize small ops.
        const effective_thread_count: usize = @max(@as(usize, 1), @min(opts.thread_count, hw_threads));

        var self: Self = .{
            .allocator = allocator,
            .pool = null,
            .thread_count = 1,
            .trace_exec = traceEnabled(),
            .profile_config = profile.Config.fromEnv(),
            .reduce_scratch_f32 = &[_]f32{},
            .matmul_scratch_f32 = &[_][]align(32) u8{},
        };

        // Kernel selection based on CPU (same for single- and multi-threaded).
        if (multiversion_enabled) {
            // Portable build: pick the linked tier object matching the runtime CPU,
            // then choose packed-GEMM tiles by L2 size. The kernel fn pointers come
            // from the per-ISA tier object; the main module holds none of them.
            const table = kernel_dispatch.selectTable(topo_info);
            if (table.abi_version != dispatch_table.ABI_VERSION) {
                @panic("aion: kernel dispatch ABI version mismatch (rebuild tier objects)");
            }
            const l2_bytes: usize = topo_info.caches.l2_bytes;
            self.l2_bytes = l2_bytes;
            self.l1d_bytes = topo_info.caches.l1d_bytes;
            self.matmul_f32 = kernel_dispatch.pickMatmul(table, l2_bytes);
            const quantized = table.quantized;
            self.matmul_q = .of(quantized.gemm, l2_bytes);
            self.matmul_nt = quantized.nt;
            self.matvec = quantized.matvec;
            self.attention_kernels = table.attention;
            self.relpos_mha_kernels = table.relpos_mha;
            self.depthwise_conv2d = table.conv2d;
            self.fft = table.fft;
        } else {
            const target = cpu_target.fromCpuInfo(topo_info);
            self.l2_bytes = target.caches.l2_bytes;
            self.l1d_bytes = target.caches.l1d_bytes;
            self.matmul_f32 = matmul_registry.selectForTarget(target).kernels;
            const quantized = compiled_quantized;
            self.matmul_q = .of(quantized.gemm, self.l2_bytes);
            self.matmul_nt = quantized.nt;
            self.matvec = quantized.matvec;
            self.attention_kernels = attention_registry.selectForTarget(target).kernels;
            self.relpos_mha_kernels = attention_registry.selectForTarget(target).kernels;
            self.depthwise_conv2d = conv2d_registry.selectForTarget(target).kernels;
            self.fft = fft_registry.selectForTarget(target).kernels;
        }

        if (effective_thread_count > 1) {
            const p = try thread_pool.ThreadPool.init(allocator, .{ .thread_count = effective_thread_count });
            self.pool = p;
            self.thread_count = effective_thread_count;
        }

        // Allocate reduction scratch. Sized to thread_count; for single-thread
        // this is still 1 slot (not zero), so ops don't need a per-call fallback.
        self.reduce_scratch_f32 = try allocator.alloc(f32, effective_thread_count);
        errdefer allocator.free(self.reduce_scratch_f32);

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
        // Cover the f32 registry maxima, the selected f32 kernel, and every member of
        // the quantized family a view may fall back to.
        const scratch_bytes: usize = @max(
            @max(matmul_registry.maxScratchBytes(), self.matmul_q.scratchBytes()),
            self.matmul_f32.scratch_bytes,
        );
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
        .createSession = createSessionImpl,
    };

    /// CPU has no per-store residency, so a session is just a `(backend, store)`
    /// pair. `execute` runs a program straight against the bound store.
    const CpuSession = struct {
        cpu: *Self,
        store: tensor_store.TensorStore,
        /// Per-store state, as the session contract says: conv weights packed
        /// for this store's tensor ids, reused across this model's runs.
        conv_cache: exec_conv.ConvCache,

        fn execute(ctx: *anyopaque, prog: *const executable.ExecutableProgram) ExecuteProgramError!void {
            const s: *CpuSession = @ptrCast(@alignCast(ctx));
            const r = s.cpu.runProgram(prog, s.store, &s.conv_cache);
            return r;
        }

        fn deinitSession(ctx: *anyopaque) void {
            const s: *CpuSession = @ptrCast(@alignCast(ctx));
            s.conv_cache.deinit();
            s.cpu.allocator.destroy(s);
        }

        fn retireResources(_: *anyopaque) void {}

        const session_vtable = Session.VTable{
            .execute = execute,
            .retireResources = retireResources,
            .deinit = deinitSession,
        };
    };

    fn createSessionImpl(ctx: *anyopaque, store: tensor_store.TensorStore) tensor_store.StoreError!Session {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const s = self.allocator.create(CpuSession) catch return error.OutOfMemory;
        s.* = .{ .cpu = self, .store = store, .conv_cache = exec_conv.ConvCache.init(self.allocator) };
        return .{ .ctx = @ptrCast(s), .vtable = &CpuSession.session_vtable };
    }

    const ElemwiseExec = exec_elemwise;

    fn readI32Scalar(store: tensor_store.TensorStore, id: executable.TensorId) ExecuteProgramError!i32 {
        const meta: tensor_store.TensorMeta = try store.meta(id);
        if (meta.dtype != .i32 or meta.rank != 1 or meta.shape.len != 1 or meta.shape[0] != 1) return error.InvalidArgument;
        const view: tensor_store.ViewConst = try store.acquireConst(id);
        defer store.releaseConst(view.token);
        if (view.bytes.len < @sizeOf(i32)) return error.InvalidArgument;
        return std.mem.readInt(i32, view.bytes[0..@sizeOf(i32)], .little);
    }

    fn readPredicate(store: tensor_store.TensorStore, id: executable.TensorId) ExecuteProgramError!bool {
        return (try readI32Scalar(store, id)) != 0;
    }

    fn readUsizeScalar(store: tensor_store.TensorStore, id: executable.TensorId) ExecuteProgramError!usize {
        const raw: i32 = try readI32Scalar(store, id);
        if (raw < 0) return error.InvalidArgument;
        return @intCast(raw);
    }

    fn copyTensorSameLayout(store: tensor_store.TensorStore, dst: executable.TensorId, src: executable.TensorId) ExecuteProgramError!void {
        const dst_meta: tensor_store.TensorMeta = try store.meta(dst);
        const src_meta: tensor_store.TensorMeta = try store.meta(src);
        if (dst_meta.dtype != src_meta.dtype or !std.mem.eql(usize, dst_meta.shape, src_meta.shape)) return error.InvalidArgument;

        const src_view = try store.acquireConst(src);
        defer store.releaseConst(src_view.token);
        const dst_view = try store.acquireMut(dst);
        defer store.releaseMut(dst_view.token);
        if (dst_view.bytes.len != src_view.bytes.len) return error.InvalidArgument;
        @memcpy(dst_view.bytes, src_view.bytes);
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

    fn execBlock(self: *Self, prog: *const executable.ExecutableProgram, block_id: executable.BlockId, store: tensor_store.TensorStore, cache: *exec_conv.ConvCache) ExecuteProgramError!void {
        const idx: usize = @intCast(block_id);
        if (idx >= prog.blocks.len) return error.InvalidArgument;
        for (prog.blocks[idx].steps) |block_step| {
            self.execStep(prog, block_step.op, store, cache) catch |err| {
                diagnostic.current().recordStep("cpu", block_step, err);
                return err;
            };
        }
    }

    fn execStep(self: *Self, prog: *const executable.ExecutableProgram, step: executable.Step, store: tensor_store.TensorStore, cache: *exec_conv.ConvCache) ExecuteProgramError!void {
        switch (step) {
            .MatMul => |s| {
                const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                var mm_ctx: exec_matmul.MatMulExecCtx = .{
                    .allocator = self.allocator,
                    .pool = pool_ptr,
                    .thread_count = self.thread_count,
                    .matmul_f32 = self.matmul_f32,
                    .matmul_q = self.matmul_q,
                    .matvec = self.matvec,
                    .matmul_scratch = self.matmul_scratch_f32,
                };
                try exec_matmul.execMatMul(&mm_ctx, s, store);
            },

            .ElemwiseBinary => |s| {
                const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                try ElemwiseExec.execElemwiseBinary(pool_ptr, self.thread_count, s, store);
            },

            .Unary => |s| {
                const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                try exec_unary.execUnary(pool_ptr, self.thread_count, s, store);
            },

            .Softmax => |s| {
                const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                try exec_softmax.execSoftmax(pool_ptr, self.thread_count, s, store);
            },

            .Conv1D => |s| {
                const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                var conv_ctx: exec_conv.ConvExecCtx = .{
                    .allocator = self.allocator,
                    .pool = pool_ptr,
                    .thread_count = self.thread_count,
                    .matmul_f32 = self.matmul_f32,
                    .depthwise_conv2d = self.depthwise_conv2d,
                    .matmul_scratch = self.matmul_scratch_f32,
                    .cache = cache,
                    .l2_bytes = self.l2_bytes,
                };
                try exec_conv.execConv1D(&conv_ctx, s, store);
            },

            .MaxPool2D => |s| try @import("exec/pool.zig").exec(
                self.allocator,
                if (self.pool) |*p| p else null,
                self.thread_count,
                s,
                store,
            ),
            .Conv2D => |s| {
                const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                var conv_ctx: exec_conv.ConvExecCtx = .{
                    .allocator = self.allocator,
                    .pool = pool_ptr,
                    .thread_count = self.thread_count,
                    .matmul_f32 = self.matmul_f32,
                    .depthwise_conv2d = self.depthwise_conv2d,
                    .matmul_scratch = self.matmul_scratch_f32,
                    .cache = cache,
                    .l2_bytes = self.l2_bytes,
                };
                try exec_conv.execConv2D(&conv_ctx, s, store);
            },

            .LayerNorm => |s| {
                const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                try exec_layernorm.execLayerNorm(pool_ptr, self.thread_count, s, store);
            },

            .RMSNorm => |s| {
                const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                try exec_layernorm.execRMSNorm(pool_ptr, self.thread_count, s, store);
            },

            .Attention => |s| {
                const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                try exec_attention.execAttention(self.allocator, pool_ptr, self.thread_count, self.attention_kernels, s, store);
            },

            .RelPosMHA => |s| {
                const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                try exec_relpos_mha.execRelPosMHA(self.allocator, pool_ptr, self.thread_count, self.relpos_mha_kernels, s, store);
            },

            .ArgMax => |s| {
                try exec_argmax.execArgMax(s, store);
            },

            .TopK => |s| {
                const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                try exec_topk.execTopK(self.allocator, pool_ptr, self.thread_count, s, store);
            },

            .ScatterRow => |s| {
                try exec_scatter.execScatterRow(s, store);
            },

            .Copy => |s| {
                const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                try ElemwiseExec.execCopy(pool_ptr, self.thread_count, s, store);
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
                    try self.execBlock(prog, s.then_block, store, cache);
                    try copyTensorLists(store, s.outputs[0..count], s.then_outputs[0..count]);
                } else {
                    try self.execBlock(prog, s.else_block, store, cache);
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

                    try self.execBlock(prog, s.body_block, store, cache);
                    try swapTensorLists(store, s.carried[0..carried_count], s.body_carried_outputs[0..carried_count]);

                    if (!s.check_before) {
                        if (s.cond) |cond_id| {
                            if (!try readPredicate(store, cond_id)) break;
                        }
                    }
                }
            },

            .Transfer => |s| {
                const dst = [_]tensor_store.TensorId{s.dst};
                const src = [_]tensor_store.TensorId{s.src};
                try copyTensorLists(store, &dst, &src);
            },

            .ReduceAll => |s| {
                const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                try exec_utils.reduceAllScalar(pool_ptr, self.thread_count, self.reduce_scratch_f32, s.op, s.out, s.a, store);
            },

            .ReduceAxis => |s| {
                const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                try exec_utils.reduceAxisScalar(pool_ptr, self.thread_count, s.op, s.out, s.a, s.axis, store);
            },

            .ConcatScalar => |s| {
                try exec_utils.concatScalar(s, store);
            },

            .ReshapeScalar => |s| {
                try exec_utils.reshapeCopyScalar(store, s.dst, s.src);
            },
            .Transpose2DScalar => |s| {
                const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                try exec_utils.transpose2DCopyScalar(pool_ptr, self.thread_count, store, s.dst, s.src);
            },
            .SliceNDScalar => |s| {
                const rank: usize = @as(usize, s.rank);
                try exec_utils.sliceNDCopyScalar(store, s.dst, s.src, s.starts[0..rank]);
            },

            .GatherRows => |s| {
                const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                try exec_gather.execGatherRows(pool_ptr, self.thread_count, s, store);
            },
            .GatherND => |s| try @import("exec/gather.zig").execGatherND(s, store),
            .Gather => |s| try exec_gather.execGather(s, store),

            .RoPE1D => |s| {
                const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                try exec_rope.execRoPE1D(pool_ptr, self.thread_count, s, store);
            },

            .SequenceAppend => |s| {
                const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                try exec_sequence_append.execSequenceAppend(pool_ptr, self.thread_count, s, store);
            },

            .Cast => |s| {
                const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                try exec_cast.execCast(pool_ptr, self.thread_count, s, store);
            },

            .MatMulNT => |s| {
                const pool_ptr: ?*thread_pool.ThreadPool = if (self.pool) |*p| p else null;
                const nt_ctx: exec_matmul_nt.MatMulNtExecCtx = .{
                    .matmul_nt = self.matmul_nt,
                    .scratch = if (self.matmul_scratch_f32.len != 0) self.matmul_scratch_f32[0] else &[_]u8{},
                    .l2_bytes = self.l2_bytes,
                    .l1d_bytes = self.l1d_bytes,
                };
                try exec_matmul_nt.execMatMulNT(&nt_ctx, pool_ptr, self.thread_count, s, store);
            },
        }
    }

    fn runProgram(self: *Self, prog: *const executable.ExecutableProgram, store: tensor_store.TensorStore, cache: *exec_conv.ConvCache) ExecuteProgramError!void {
        const invocation = self.profile_invocations;
        self.profile_invocations +|= 1;
        const config = self.profile_config;
        const do_profile = config.captures(invocation);
        var profiler: ?profile.Session = if (do_profile) profile.Session.init(self.allocator, config, "CPU program") else null;
        defer if (profiler) |*p| p.deinit();
        const cpu_track = if (profiler) |*p| p.addTrack("CPU", .host) else null;
        const trace_exec: bool = self.trace_exec;

        const run_t0: u64 = if (do_profile) profile.nowNs() else 0;
        for (prog.steps, 0..) |step, step_i| {
            const t0: u64 = if (do_profile) profile.nowNs() else 0;
            if (trace_exec) {
                std.debug.print("[aion][exec] step {d}/{d}: {s}\n", .{ step_i, prog.steps.len, @tagName(step.op) });
            }
            self.execStep(prog, step.op, store, cache) catch |e| {
                diagnostic.current().recordStep("cpu", step, e);
                if (trace_exec) {
                    std.debug.print("[aion][exec] step {d} failed: {s} err={s}\n", .{ step_i, @tagName(step.op), @errorName(e) });
                }
                return e;
            };

            if (do_profile) {
                const t1 = profile.nowNs();
                if (cpu_track) |track| profiler.?.recordSpan(track, .operation, @tagName(step.op), t0, t1);
            }
        }

        if (profiler) |*p| {
            if (cpu_track) |track| p.recordSpan(track, .phase, "program", run_t0, profile.nowNs());
            p.report();
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
