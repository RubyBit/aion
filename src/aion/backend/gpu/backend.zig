// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! `GpuBackend` -- the WebGPU (wgpu-native) implementation of `aion.backend.Backend`.
//!
//! In-tree but feature-gated: these files are part of the `aion` module, but
//! root.zig only `@import`s them when `build_options.enable_gpu` is true (set by
//! `-Dgpu`). When off, the gpu code is never analyzed, so wgpu isn't fetched or
//! linked -- Rust's `#[cfg(feature = "gpu")]` equivalent. The application builds
//! a `GpuBackend` (via `aion.gpu`) and drives it through its `.backend()` vtable
//! handle: `gb.backend().executeProgram(prog, store)`.
//!
//! Execution model:
//!   - Every tensor has one physical placement. Model preparation migrates GPU
//!     values before execution; the executor has no host-staging fallback.
//!   - Transfers are explicit executable steps. Host control reads can only see
//!     CPU mirrors produced by those steps, while kernels only see device tiles.
//!   - Each `executeProgram` records dispatches into a `Frame` and submits once
//!     except where an explicit control transfer requires synchronization.
//!   - WGSL kernels are cached via `Pipelines` (module per kernel, pipeline +
//!     bind-group layout per entry point -- reflected once, not per dispatch).
//!
//! Kernel coverage (f32, multi-tile): ElemwiseBinaryTiled, UnaryTiled,
//! broadcast-aware elementwise binary, CopyTiled (buffer-to-buffer, dtype-agnostic),
//! MatMulTiled (rank-2, autotuned register-blocked GEMM), MatMulNTTiled
//! (q8_0/f32 weights: GEMV for M==1, dequant + f32 GEMM for M>1), SoftmaxTiled /
//! RMSNormTiled / LayerNormTiled / ReduceAll / ReduceAxis (row-wise, last axis
//! intra-tile), GatherRows / RoPE1D / SequenceAppend / Cast (decode data
//! movement), AttentionTiled (GQA over f32/f16 k/v, positions/end read
//! on-device when present), Conv1DTiled /
//! Conv2DTiled (direct kernel). Everything else returns `error.Unsupported`.

const std = @import("std");

// In-tree: relative imports into the aion module (no `@import("aion")`), gated
// into the build only when `-Dgpu` is set (see root.zig). `wgpu` is the one
// genuine module import (the translate-c'd C bindings).
pub const wgpu = @import("wgpu.zig"); // re-exported so apps reach Gpu/Options/c
const wgpu_dm = @import("device_memory.zig");
const pipelines_mod = @import("pipelines.zig");
const context = @import("context.zig");
const simple_ops = @import("exec/simple_ops.zig");
const rowwise = @import("exec/rowwise.zig");
const decode_ops = @import("exec/decode_ops.zig");
const attention_exec = @import("exec/attention.zig");
const conv_exec = @import("exec/conv.zig");
const lstm_exec = @import("exec/lstm.zig");
const fft_ops = @import("exec/fft_ops.zig");
const view_ops = @import("exec/view_ops.zig");
const matmul_exec = @import("exec/matmul.zig");
const matmul_nt = @import("exec/matmul_nt.zig");
const frame_mod = @import("frame.zig");
const timestamp_profile = @import("timestamp_profile.zig");
const TimestampProfiler = timestamp_profile.TimestampProfiler;
const backend_mod = @import("../backend.zig");
const types = @import("../types.zig");
const executable = @import("../../runtime/executable.zig");
const tensor_store_mod = @import("../../runtime/tensor_store.zig");
const device_store_mod = @import("../../runtime/device_store.zig");
const thread_pool_mod = @import("../../runtime/thread_pool.zig");
const profile_mod = @import("../../profile.zig");
const env_util = @import("../../env.zig");

const c = wgpu.c;
const fns = wgpu.fns; // runtime wgpu dispatch table (functions)

const Backend = backend_mod.Backend;
const Session = backend_mod.Session;
const ExecuteProgramError = backend_mod.ExecuteProgramError;
const BackendKind = types.BackendKind;
const BackendCaps = types.BackendCaps;
const ExecutableProgram = executable.ExecutableProgram;
const TensorStore = tensor_store_mod.TensorStore;
const TensorId = tensor_store_mod.TensorId;
const TensorMeta = tensor_store_mod.TensorMeta;
const Frame = frame_mod.Frame;

pub const GpuBackend = struct {
    allocator: std.mem.Allocator,
    gpu: *wgpu.Gpu,
    devmem: wgpu_dm.WgpuDeviceMemory,
    pipes: pipelines_mod.Pipelines,

    /// Matmul executor owns the per-shape autotune cache.
    matmul: matmul_exec.Matmul,

    /// MatMulNT executor (q8_0/f32 weights): GEMV for M==1, dequant-to-scratch +
    /// the f32 GEMM pipelines above for M>1. Owns the pooled scratch buffer.
    nt: matmul_nt.MatmulNt = .{},

    /// Shared scratch for multi-stage kernels (two-stage reduce/argmax,
    /// split-K cached attention partials). See `context.ScratchPool`.
    scratch: context.ScratchPool = .{},

    /// Pool for parallelizing the D2H readback host memcpy (the mapped-staging ->
    /// host-tensor copy is ~half the readback cost on large outputs). Owned here;
    /// `devmem.pool` points at it. Null when only one hardware thread is available.
    pool: ?thread_pool_mod.ThreadPool = null,

    /// Reused per-dispatch uniform buffers (see `frame.UniformPool`) — cuts the
    /// ~700 create/release driver calls a decode step would otherwise pay.
    uniform_pool: frame_mod.UniformPool,

    /// Reused per-dispatch bind groups + uniforms (see `frame.BindGroupCache`) —
    /// the decode program is identical each step, so bind groups are built once
    /// and replayed, removing the dominant CPU record cost. Supersedes the pool.
    bind_cache: frame_mod.BindGroupCache,

    /// Internal counter used only to apply the profiler's skip/count window.
    profile_invocations: u64 = 0,
    profile_config: profile_mod.Config,

    const Self = @This();

    /// D2H memcpy is memory-bandwidth bound; a few threads saturate it. Cap low so
    /// we don't spin up a big pool the GPU backend otherwise doesn't need.
    const D2H_MAX_THREADS: usize = 4;

    pub fn init(allocator: std.mem.Allocator, gpu: *wgpu.Gpu) Self {
        var self: Self = .{
            .allocator = allocator,
            .gpu = gpu,
            .devmem = wgpu_dm.WgpuDeviceMemory.init(allocator, gpu),
            .pipes = pipelines_mod.Pipelines.init(allocator, gpu),
            .matmul = matmul_exec.Matmul.init(allocator),
            .uniform_pool = .{ .gpu = gpu, .allocator = allocator },
            .bind_cache = .{ .gpu = gpu, .allocator = allocator },
            .profile_config = profile_mod.Config.fromEnv(),
        };
        const hw = std.Thread.getCpuCount() catch 1;
        const n = @max(@as(usize, 1), @min(hw, D2H_MAX_THREADS));
        if (n > 1) {
            if (thread_pool_mod.ThreadPool.init(allocator, .{ .thread_count = n })) |p| {
                self.pool = p;
            } else |_| {}
        }
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.matmul.deinit();
        self.nt.deinit();
        self.scratch.deinit();
        self.pipes.deinit();
        self.devmem.deinit();
        self.uniform_pool.deinit();
        self.bind_cache.deinit();
        if (self.pool) |*p| p.deinit();
        self.* = undefined;
    }

    pub fn backend(self: *Self) Backend {
        return .{ .ctx = @ptrCast(self), .vtable = &vtable };
    }

    /// Block until all submitted GPU work has completed. Benchmarks call this
    /// once after a batch of un-read runs to time pure compute — `execute` never
    /// reads outputs back, so not calling `outputTensor` is what excludes D2H.
    pub fn sync(self: *Self) void {
        _ = fns.wgpuDevicePoll(self.gpu.device, 1, null);
    }

    /// Build a session over an already-placed store. GPU operation code receives
    /// only opaque device tiles; host values are reachable solely through
    /// explicit transfer/control paths.
    pub fn createSession(self: *Self, store: TensorStore) tensor_store_mod.StoreError!Session {
        // Wire the D2H memcpy pool now that `self` is at its final address (the
        // backend is passed by pointer through the vtable). Idempotent.
        self.devmem.pool = if (self.pool) |*p| p else null;
        const s = self.allocator.create(GpuSession) catch return error.OutOfMemory;
        s.* = .{
            .gb = self,
            .host = store,
        };
        return .{ .ctx = @ptrCast(s), .vtable = &GpuSession.session_vtable };
    }

    fn createSessionImpl(ctx: *anyopaque, store: TensorStore) tensor_store_mod.StoreError!Session {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return self.createSession(store);
    }

    /// A GPU execution session bound to one placed tensor store.
    pub const GpuSession = struct {
        gb: *Self,
        host: TensorStore,

        fn execute(ctx: *anyopaque, prog: *const ExecutableProgram) ExecuteProgramError!void {
            const s: *GpuSession = @ptrCast(@alignCast(ctx));
            return s.gb.runProgram(s, prog);
        }

        fn deviceMeta(ctx: *anyopaque, id: TensorId) tensor_store_mod.StoreError!TensorMeta {
            const s: *GpuSession = @ptrCast(@alignCast(ctx));
            return s.host.meta(id);
        }

        fn deviceAcquire(ctx: *anyopaque, id: TensorId, tile_index: usize) tensor_store_mod.StoreError!device_store_mod.TileRef {
            const s: *GpuSession = @ptrCast(@alignCast(ctx));
            const tile = (try s.host.deviceTile(id, tile_index)) orelse return error.InvalidArgument;
            return .{
                .handle = tile.handle,
                .offset = 0,
                .len = tile.len,
                .dtype = tile.dtype,
                .rank = tile.rank,
                .shape_mem = tile.shape_mem,
                .strides_mem = tile.strides_mem,
            };
        }

        fn deviceRelease(_: *anyopaque, _: usize) void {}

        fn sequenceCachePolicyInfo(ctx: *anyopaque, id: TensorId) tensor_store_mod.SequenceCachePolicyInfo {
            const s: *GpuSession = @ptrCast(@alignCast(ctx));
            return s.host.sequenceCachePolicyInfo(id);
        }

        fn mapSequenceStep(ctx: *anyopaque, id: TensorId, logical: usize, capacity: usize) tensor_store_mod.StoreError!usize {
            const s: *GpuSession = @ptrCast(@alignCast(ctx));
            return s.host.mapSequenceStep(id, logical, capacity);
        }

        fn deviceStore(s: *GpuSession) device_store_mod.DeviceStore {
            return .{ .ctx = @ptrCast(s), .vtable = &device_vtable };
        }

        fn deinitSession(ctx: *anyopaque) void {
            const s: *GpuSession = @ptrCast(@alignCast(ctx));
            s.gb.allocator.destroy(s);
        }

        fn retireResources(ctx: *anyopaque) void {
            const s: *GpuSession = @ptrCast(@alignCast(ctx));
            _ = fns.wgpuDevicePoll(s.gb.gpu.device, 1, null);
            s.gb.bind_cache.clear();
        }

        const device_vtable = device_store_mod.DeviceStore.VTable{
            .meta = deviceMeta,
            .acquireConst = deviceAcquire,
            .acquireMut = deviceAcquire,
            .releaseConst = deviceRelease,
            .releaseMut = deviceRelease,
            .sequenceCachePolicyInfo = sequenceCachePolicyInfo,
            .mapSequenceStep = mapSequenceStep,
        };

        const session_vtable = Session.VTable{
            .execute = execute,
            .retireResources = retireResources,
            .deinit = deinitSession,
        };
    };

    fn totalTiles(meta: TensorMeta) usize {
        var total: usize = 1;
        for (meta.tile_counts) |cnt| total *= cnt;
        return total;
    }

    // ---- vtable ----

    /// Per-`executeProgram` execution state. Most steps just record into the
    /// current frame; control flow (If/Loop) needs host-visible predicates, which
    /// the compiler has already placed on the CPU — a `Transfer` step pays the one
    /// device poll, and only when the device actually produced the value.
    /// `Frame.records` lets a fixed-trip Loop record its whole unrolled body into
    /// one frame and submit once, with zero syncs.
    const Runner = struct {
        gb: *Self,
        op_ctx: context.Ctx,
        prog: *const ExecutableProgram,
        frame: Frame,

        // Top-level operation scopes avoid double-counting nested If/Loop bodies.
        depth: u32 = 0,
        profiler: ?*profile_mod.Session = null,
        cpu_track: ?profile_mod.TrackId = null,

        /// Submit the pending frame WITHOUT waiting, then keep recording into a
        /// fresh encoder on the same frame (see `Frame.flushInPlace`). No-op when
        /// nothing is recorded. We deliberately do not poll here: queue
        /// submissions execute in order, so a later D2H readback (whose own poll
        /// waits for all prior work) or a later frame sees these results without a
        /// redundant stall. Callers that need the result on the host follow this
        /// with `readI32Scalar`, whose `copyD2H` performs the single poll.
        fn submitPending(r: *Runner) ExecuteProgramError!void {
            r.frame.flushInPlace() catch return error.ExecutionFailed;
        }

        /// Host-read an i32 control value (element 0 of tile 0). `id` is always
        /// CPU-placed — the compiler transferred it if the device wrote it — so
        /// this is a plain read with no submit and no device poll.
        fn readI32Scalar(r: *Runner, id: TensorId) ExecuteProgramError!i32 {
            const lease = try r.op_ctx.control.readI32(id);
            defer lease.release();
            if (lease.vals.len == 0) return error.ExecutionFailed;
            return lease.vals[0];
        }

        /// Device-side equivalent of the CPU's `copyTensorLists`: same-layout
        /// tensors, tile-for-tile buffer copies (stays in-frame, no sync).
        fn copyLists(r: *Runner, dst: []const TensorId, src: []const TensorId) ExecuteProgramError!void {
            const hs = r.op_ctx.store;
            for (dst, src) |dst_id, src_id| {
                if (dst_id == src_id) continue;
                const dst_meta = hs.meta(dst_id) catch return error.ExecutionFailed;
                const src_meta = hs.meta(src_id) catch return error.ExecutionFailed;
                if (dst_meta.dtype != src_meta.dtype or dst_meta.rank != src_meta.rank) return error.ExecutionFailed;
                for (dst_meta.shape, src_meta.shape) |a, b| if (a != b) return error.ExecutionFailed;
                for (dst_meta.tile_shape, src_meta.tile_shape) |a, b| if (a != b) return error.ExecutionFailed;
                for (dst_meta.tile_counts, src_meta.tile_counts) |a, b| if (a != b) return error.ExecutionFailed;

                const total = totalTiles(dst_meta);
                var ti: usize = 0;
                while (ti < total) : (ti += 1) {
                    const st = r.op_ctx.store.acquireTileDeviceConstLinear(src_id, ti) catch return error.ExecutionFailed;
                    const dt = r.op_ctx.store.acquireTileDeviceMutLinear(dst_id, ti) catch return error.ExecutionFailed;
                    defer {
                        hs.releaseConst(st.token);
                        hs.releaseMut(dt.token);
                    }
                    if (st.len != dt.len or st.len % 4 != 0) return error.Unsupported;
                    r.frame.recordCopy(
                        r.op_ctx.devmem.bufferFor(st.handle).?,
                        0,
                        r.op_ctx.devmem.bufferFor(dt.handle).?,
                        0,
                        st.len,
                    );
                }
            }
        }

        fn runBlock(r: *Runner, block_id: executable.BlockId) ExecuteProgramError!void {
            const idx: usize = @intCast(block_id);
            if (idx >= r.prog.blocks.len) return error.ExecutionFailed;
            r.depth += 1;
            defer r.depth -= 1;
            for (r.prog.blocks[idx].steps) |step| try r.runStep(step.op);
        }

        fn runStep(r: *Runner, step: executable.Step) ExecuteProgramError!void {
            const previous_op = r.frame.profile_op;
            r.frame.profile_op = @tagName(std.meta.activeTag(step));
            defer r.frame.profile_op = previous_op;
            const generic_profile = r.profiler != null and r.depth == 0;
            const t0: u64 = if (generic_profile) profile_mod.nowNs() else 0;
            r.dispatchStep(step) catch |e| {
                if (env_util.flagEnabled("AION_GPU_TRACE")) {
                    std.debug.print("[gpu] step {s} failed: {s}\n", .{ @tagName(std.meta.activeTag(step)), @errorName(e) });
                    switch (step) {
                        .ReduceAxis => |s| {
                            const input = r.op_ctx.store.meta(s.a) catch null;
                            const output = r.op_ctx.store.meta(s.out) catch null;
                            if (input) |m| std.debug.print(
                                "[gpu]   input id={} dtype={s} shape={any} tile_shape={any} tile_counts={any}\n",
                                .{ s.a, @tagName(m.dtype), m.shape, m.tile_shape, m.tile_counts },
                            );
                            if (output) |m| std.debug.print(
                                "[gpu]   output id={} dtype={s} shape={any} tile_shape={any} tile_counts={any} axis={} op={s}\n",
                                .{ s.out, @tagName(m.dtype), m.shape, m.tile_shape, m.tile_counts, s.axis, @tagName(s.op) },
                            );
                        },
                        else => {},
                    }
                }
                return e;
            };
            if (generic_profile) {
                if (r.cpu_track) |track| r.profiler.?.recordSpan(track, .operation, @tagName(step), t0, profile_mod.nowNs());
            }
        }

        fn runTransfer(r: *Runner, transfer: executable.StepTransfer) ExecuteProgramError!void {
            // A transfer is an ordering point. Source and destination each have
            // one physical placement; lowering chooses the concrete copy.
            try r.frame.flushInPlace();
            const hs = r.op_ctx.control.host;
            const src_meta = hs.meta(transfer.src) catch return error.ExecutionFailed;
            const dst_meta = hs.meta(transfer.dst) catch return error.ExecutionFailed;
            if (src_meta.dtype != dst_meta.dtype or totalTiles(src_meta) != totalTiles(dst_meta)) return error.ExecutionFailed;
            const count = totalTiles(src_meta);
            var i: usize = 0;
            while (i < count) : (i += 1) {
                if (transfer.source.kind == .webgpu and transfer.destination.kind == .cpu) {
                    const src = r.op_ctx.store.acquireTileDeviceConstLinear(transfer.src, i) catch return error.ExecutionFailed;
                    defer r.op_ctx.store.releaseConst(src.token);
                    const dst = hs.acquireTileMutLinear(transfer.dst, i) catch return error.ExecutionFailed;
                    defer hs.releaseMut(dst.token);
                    if (src.len != dst.bytes.len) return error.ExecutionFailed;
                    r.op_ctx.devmem.device().copyD2H(dst.bytes, src.handle, src.offset) catch return error.ExecutionFailed;
                } else if (transfer.source.kind == .cpu and transfer.destination.kind == .webgpu) {
                    const src = hs.acquireTileConstLinear(transfer.src, i) catch return error.ExecutionFailed;
                    defer hs.releaseConst(src.token);
                    const dst = r.op_ctx.store.acquireTileDeviceMutLinear(transfer.dst, i) catch return error.ExecutionFailed;
                    defer r.op_ctx.store.releaseMut(dst.token);
                    if (src.bytes.len != dst.len) return error.ExecutionFailed;
                    r.op_ctx.devmem.device().copyH2D(dst.handle, dst.offset, src.bytes) catch return error.ExecutionFailed;
                } else {
                    return error.Unsupported;
                }
            }
        }

        fn dispatchStep(r: *Runner, step: executable.Step) ExecuteProgramError!void {
            const op_ctx = r.op_ctx;
            const frame = &r.frame;
            switch (step) {
                .ElemwiseBinaryTiled => |s| try simple_ops.execElemwiseBinary(op_ctx, frame, s),
                .UnaryTiled => |s| try simple_ops.execUnary(op_ctx, frame, s),
                .CopyTiled => |s| try simple_ops.execCopy(op_ctx, frame, s),
                .CastTiled => |s| try simple_ops.execCast(op_ctx, frame, s),
                .MatMulTiled => |s| try r.gb.matmul.exec(op_ctx, frame, s),
                .MatMulNTTiled => |s| try r.gb.nt.exec(op_ctx, frame, s, r.gb.matmul.generated),
                .SoftmaxTiled => |s| try rowwise.execSoftmax(op_ctx, frame, s),
                // The residual is a configuration of the norm, so the kernel choice is
                // made here rather than by a step tag: `add_norm.wgsl` when one is
                // present, the plain rowwise/cross-tile paths otherwise.
                .RMSNormTiled => |s| if (s.residual != null)
                    try rowwise.execAddNorm(op_ctx, frame, s)
                else
                    try rowwise.execNorm(op_ctx, frame, .rmsnorm, s),
                .LayerNormTiled => |s| try rowwise.execNorm(op_ctx, frame, .layernorm, s),
                .ReduceAll => |s| try rowwise.execReduceAll(op_ctx, frame, s),
                .ReduceAxis => |s| try rowwise.execReduceAxis(op_ctx, frame, s),
                .GatherRowsTiled => |s| try decode_ops.execGatherRows(op_ctx, frame, s),
                .GatherTiled => |s| try decode_ops.execGather(op_ctx, frame, s),
                .RoPE1DTiled => |s| try decode_ops.execRoPE(op_ctx, frame, s),
                .SequenceAppendTiled => |s| try decode_ops.execSequenceAppend(op_ctx, frame, s),
                .AttentionTiled => |s| try attention_exec.execAttention(op_ctx, frame, s),
                .RelPosMHATiled => |s| try attention_exec.execRelPosMHA(op_ctx, frame, s),
                .Conv1DTiled => |s| try conv_exec.execConv1D(op_ctx, frame, s),
                .Conv2DTiled => |s| try conv_exec.execConv2D(op_ctx, frame, s),
                .LSTMCellFused => |s| try lstm_exec.execLSTMCell(op_ctx, frame, s),
                .RFFT => |s| try fft_ops.execRFFT(op_ctx, frame, s),
                .STFT => |s| try fft_ops.execSTFT(op_ctx, frame, s),
                .ArgMax => |s| try rowwise.execArgMax(op_ctx, frame, s),
                .ScatterRow => |s| try decode_ops.execScatterRow(op_ctx, frame, s),
                .ConcatScalar => |s| try view_ops.execConcat(op_ctx, frame, s),
                .ReshapeScalar => |s| try view_ops.execPackedCopy(op_ctx, frame, s.dst, s.src),
                .ReTileCopyScalar => |s| try view_ops.execPackedCopy(op_ctx, frame, s.dst, s.src),
                .Transpose2DScalar => |s| try view_ops.execTranspose2D(op_ctx, frame, s),
                .SliceNDScalar => |s| try view_ops.execSliceND(op_ctx, frame, s),
                .Transfer => |s| try r.runTransfer(s),

                .If => |s| {
                    const take_then = (try r.readI32Scalar(s.cond)) != 0;
                    const count: usize = @intCast(s.output_count);
                    if (count > executable.MAX_CONTROL_OUTPUTS) return error.ExecutionFailed;
                    if (take_then) {
                        try r.runBlock(s.then_block);
                        try r.copyLists(s.outputs[0..count], s.then_outputs[0..count]);
                    } else {
                        try r.runBlock(s.else_block);
                        try r.copyLists(s.outputs[0..count], s.else_outputs[0..count]);
                    }
                },

                .Loop => |s| {
                    const carried_count: usize = @intCast(s.carried_count);
                    if (carried_count > executable.MAX_LOOP_CARRIED) return error.ExecutionFailed;
                    var requested: usize = s.static_max_trip_count;
                    if (s.trip_count) |tid| {
                        const raw = try r.readI32Scalar(tid);
                        if (raw < 0) return error.ExecutionFailed;
                        requested = @intCast(raw);
                    }
                    const max_iters = @min(requested, s.static_max_trip_count);

                    var iter: usize = 0;
                    while (iter < max_iters) : (iter += 1) {
                        if (s.check_before) {
                            if (s.cond) |cid| {
                                const p = try r.readI32Scalar(cid);
                                if (p == 0) break;
                            }
                        }
                        try r.runBlock(s.body_block);
                        // A logical tensor keeps its backing and placement across
                        // the handoff. Swapping complete storage records here made
                        // caller-bound state change ownership after execution.
                        try r.copyLists(s.carried[0..carried_count], s.body_carried_outputs[0..carried_count]);
                        if (!s.check_before) {
                            if (s.cond) |cid| {
                                const p = try r.readI32Scalar(cid);
                                if (p == 0) break;
                            }
                        }
                    }
                },
            }
        }
    };

    fn runProgram(self: *Self, session: *GpuSession, prog: *const ExecutableProgram) ExecuteProgramError!void {
        // A program carries the placement it was compiled for, and its host reads
        // were made explicit against that placement. Executing one compiled for
        // somewhere else would silently reintroduce undeclared host reads, so
        // refuse rather than run it.
        if (prog.target.kind != .webgpu) {
            if (env_util.flagEnabled("AION_GPU_TRACE"))
                std.debug.print("[gpu] placement guard rejected target {s}\n", .{@tagName(prog.target.kind)});
            return error.Unsupported;
        }

        const invocation = self.profile_invocations;
        self.profile_invocations +|= 1;
        const profile_config = self.profile_config;
        const generic_profile = profile_config.captures(invocation);
        var profile_session: ?profile_mod.Session = if (generic_profile) profile_mod.Session.init(self.allocator, profile_config, "WebGPU program") else null;
        defer if (profile_session) |*p| p.deinit();
        const cpu_track = if (profile_session) |*p| p.addTrack("CPU", .host) else null;
        const gpu_track = if (profile_session) |*p| p.addTrack("GPU", .device) else null;

        var timestamp_profiler: ?TimestampProfiler = null;
        if (generic_profile and gpu_track != null) {
            if (self.gpu.timestamp_query) {
                const query_capacity: u32 = @intCast(@min(@as(usize, 4096), profile_config.event_capacity *| 2));
                const profile_gpu = env_util.getOwned(std.heap.page_allocator, "AION_PROFILE_GPU");
                defer if (profile_gpu) |raw| std.heap.page_allocator.free(raw);
                const timestamp_mode: timestamp_profile.Mode = if (profile_gpu) |raw|
                    if (std.mem.eql(u8, raw, "pass")) .pass else .dispatch
                else
                    .dispatch;
                timestamp_profiler = TimestampProfiler.init(self.allocator, self.gpu, query_capacity, &profile_session.?, gpu_track.?, timestamp_mode) catch null;
            } else {
                std.debug.print("[aion-profile] GPU track unavailable: adapter does not support timestamp-query\n", .{});
            }
        }
        defer if (timestamp_profiler) |*p| p.deinit();

        var runner: Runner = .{
            .gb = self,
            .op_ctx = .{
                .gpu = self.gpu,
                .devmem = &self.devmem,
                .pipes = &self.pipes,
                .allocator = self.allocator,
                .store = session.deviceStore(),
                .control = .{ .host = session.host },
                .scratch = &self.scratch,
            },
            .prog = prog,
            .frame = try Frame.init(self.allocator, self.gpu),
        };
        defer runner.frame.deinit();
        runner.profiler = if (profile_session) |*p| p else null;
        runner.cpu_track = cpu_track;
        if (timestamp_profiler) |*p| runner.frame.timestamps = p;

        // Reuse pooled per-dispatch uniforms + cached bind groups this run. Safe:
        // the prior run ended with a readback poll, so its GPU work (and all its
        // uniform/bind-group reads) are complete before we replay them.
        self.uniform_pool.reset();
        self.bind_cache.reset();
        runner.frame.uniform_pool = &self.uniform_pool;
        runner.frame.bind_cache = &self.bind_cache;

        const t_start: u64 = if (generic_profile) profile_mod.nowNs() else 0;

        // Chunked submit: flush the pending frame every SUBMIT_CHUNK steps
        // (submit WITHOUT polling — queue ordering + wgpu's cross-submit resource
        // tracking preserve correctness). The GPU starts executing early chunks
        // while the CPU is still recording later ones, overlapping the ~10ms of
        // per-dispatch record cost with the GPU execution instead of paying them
        // back-to-back. For a program with control flow the inner submitPending
        // calls already split frames; this just adds periodic flushes.
        //
        // 32 was swept on device against 8/16/64/128/unbounded: larger chunks lose the
        // record/execute overlap and smaller ones submit too often.
        const SUBMIT_CHUNK: usize = 32;
        var since_submit: usize = 0;
        for (prog.steps) |step| {
            try runner.runStep(step.op);
            since_submit += 1;
            if (since_submit >= SUBMIT_CHUNK) {
                const t_chunk_submit: u64 = if (generic_profile) profile_mod.nowNs() else 0;
                runner.submitPending() catch |e| {
                    if (env_util.flagEnabled("AION_GPU_TRACE"))
                        std.debug.print("[gpu] chunk submit failed: {s}\n", .{@errorName(e)});
                    return e;
                };
                if (generic_profile) {
                    if (cpu_track) |track| profile_session.?.recordSpan(track, .phase, "chunk submit", t_chunk_submit, profile_mod.nowNs());
                }
                since_submit = 0;
            }
        }

        const t_recorded: u64 = if (generic_profile) profile_mod.nowNs() else 0;

        runner.frame.submit();
        const t_final_submitted: u64 = if (generic_profile) profile_mod.nowNs() else 0;

        const t_outputs_finished: u64 = if (generic_profile) profile_mod.nowNs() else 0;

        if (timestamp_profiler) |*p| {
            p.readResults() catch |e| std.debug.print("[aion-profile] GPU timestamp read failed: {s}\n", .{@errorName(e)});
        }

        const t_profile_finished = if (generic_profile) profile_mod.nowNs() else 0;
        if (profile_session) |*p| {
            if (cpu_track) |track| {
                p.recordSpan(track, .phase, "record", t_start, t_recorded);
                p.recordSpan(track, .phase, "final submit", t_recorded, t_final_submitted);
                p.recordSpan(track, .wait, "output readback", t_final_submitted, t_outputs_finished);
                p.recordSpan(track, .phase, "profiler readback", t_outputs_finished, t_profile_finished);
                p.recordSpan(track, .phase, "program", t_start, t_outputs_finished);
            }
            p.report();
        }
    }

    fn kindImpl(_: *anyopaque) BackendKind {
        return .webgpu;
    }

    fn nameImpl(_: *anyopaque) []const u8 {
        return "Aion WebGPU Backend";
    }

    fn capsImpl(_: *anyopaque) BackendCaps {
        return .{}; // f32 compute only; refine as kernels land.
    }

    fn deinitImpl(_: *anyopaque) void {
        // No-op: the GpuBackend is owned by its creator (call GpuBackend.deinit
        // directly). Context's external-backend arm never calls this.
    }

    const vtable = Backend.VTable{
        .kind = kindImpl,
        .name = nameImpl,
        .caps = capsImpl,
        .deinit = deinitImpl,
        .createSession = createSessionImpl,
    };
};
