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
//!   - Device residency (`ResidentTensorStore` over `WgpuDeviceMemory`) lives on
//!     a `Session` (`GpuSession`) bound to one host store, NOT on the backend.
//!     Residency PERSISTS for the session's lifetime across `session.execute`
//!     calls: weights are never host-written, so they upload once and stay
//!     device-resident across decode steps; inputs re-upload when the host
//!     rewrites them. The session object is the residency's identity — no
//!     pointer-keyed cache. Device-global caches (pipelines, autotune, scratch)
//!     stay on the backend and are shared by all its sessions. (The v0 backend
//!     rebuilt residency every call, so every tensor re-uploaded each run.)
//!   - Each `executeProgram` records all its dispatches into a single `Frame`
//!     (one command encoder) and submits ONCE, then flushes program outputs back
//!     to the host store (D2H).
//!   - WGSL kernels are cached via `Pipelines` (module per kernel, pipeline +
//!     bind-group layout per entry point -- reflected once, not per dispatch).
//!
//! Kernel coverage (f32, multi-tile): ElemwiseBinaryTiled, UnaryTiled,
//! BroadcastLastDimBinaryTiled, CopyTiled (buffer-to-buffer, dtype-agnostic),
//! MatMulTiled (rank-2, autotuned register-blocked GEMM), MatMulNTTiled
//! (q8_0/f32 weights: GEMV for M==1, dequant + f32 GEMM for M>1), SoftmaxTiled /
//! RMSNormTiled / LayerNormTiled / ReduceAll / ReduceAxis (row-wise, last axis
//! intra-tile), GatherRows / RoPE1D / KVCacheAppend / Cast (decode data
//! movement), AttentionTiled / MultiHeadAttentionTiled (streaming-softmax
//! kernel, one workgroup per query row), MultiHeadAttentionCachedTiled (GQA
//! over f32/f16 KV caches, positions/end read on-device), Conv1DTiled /
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
const backend_mod = @import("../backend.zig");
const types = @import("../types.zig");
const executable = @import("../../runtime/executable.zig");
const tensor_store_mod = @import("../../runtime/tensor_store.zig");
const resident_mod = @import("../../runtime/residency/resident_store.zig");
const thread_pool_mod = @import("../../runtime/thread_pool.zig");

const c = wgpu.c;

const Backend = backend_mod.Backend;
const Session = backend_mod.Session;
const ExecuteProgramError = backend_mod.ExecuteProgramError;
const BackendKind = types.BackendKind;
const BackendCaps = types.BackendCaps;
const ExecutableProgram = executable.ExecutableProgram;
const TensorStore = tensor_store_mod.TensorStore;
const TensorId = tensor_store_mod.TensorId;
const TensorMeta = tensor_store_mod.TensorMeta;
const ResidentTensorStore = resident_mod.ResidentTensorStore;
const Frame = frame_mod.Frame;

pub const GpuBackend = struct {
    allocator: std.mem.Allocator,
    gpu: *wgpu.Gpu,
    devmem: wgpu_dm.WgpuDeviceMemory,
    pipes: pipelines_mod.Pipelines,

    /// When false, `executeProgram` records + submits the compute but skips the
    /// device->host flush of outputs. Used by benchmarks to time pure kernel
    /// throughput without paying a full-output D2H readback every iteration.
    flush_outputs: bool = true,

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
        if (self.pool) |*p| p.deinit();
        self.* = undefined;
    }

    pub fn backend(self: *Self) Backend {
        return .{ .ctx = @ptrCast(self), .vtable = &vtable };
    }

    /// Block until all submitted GPU work has completed. Benchmarks call this once
    /// after a batch of `flush_outputs == false` runs to time pure compute.
    pub fn sync(self: *Self) void {
        _ = c.wgpuDevicePoll(self.gpu.device, 1, null);
    }

    /// Build a `Session` bound to `store`: a `GpuSession` owning a fresh
    /// `ResidentTensorStore` over this backend's device memory. Residency lives
    /// and dies with the returned session (no pointer-keyed cache, no reset).
    pub fn createSession(self: *Self, store: TensorStore) tensor_store_mod.StoreError!Session {
        // Wire the D2H memcpy pool now that `self` is at its final address (the
        // backend is passed by pointer through the vtable). Idempotent.
        self.devmem.pool = if (self.pool) |*p| p else null;
        const s = self.allocator.create(GpuSession) catch return error.OutOfMemory;
        s.* = .{
            .gb = self,
            .resident = ResidentTensorStore.init(self.allocator, store, self.devmem.device()),
        };
        return .{ .ctx = @ptrCast(s), .vtable = &GpuSession.session_vtable };
    }

    fn createSessionImpl(ctx: *anyopaque, store: TensorStore) tensor_store_mod.StoreError!Session {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return self.createSession(store);
    }

    /// A GPU execution session: owns the device residency for one host store.
    /// `execute` records a whole program into one `Frame` and submits once,
    /// reusing the session's resident device buffers across calls.
    pub const GpuSession = struct {
        gb: *Self,
        resident: ResidentTensorStore,

        fn execute(ctx: *anyopaque, prog: *const ExecutableProgram) ExecuteProgramError!void {
            const s: *GpuSession = @ptrCast(@alignCast(ctx));
            return s.gb.runProgram(&s.resident, prog);
        }

        fn deinitSession(ctx: *anyopaque) void {
            const s: *GpuSession = @ptrCast(@alignCast(ctx));
            s.resident.deinit();
            s.gb.allocator.destroy(s);
        }

        const session_vtable = Session.VTable{
            .execute = execute,
            .deinit = deinitSession,
        };
    };

    fn totalTiles(meta: TensorMeta) usize {
        var total: usize = 1;
        for (meta.tile_counts) |cnt| total *= cnt;
        return total;
    }

    /// Force device->host flush of every tile of `id` so the host store holds the
    /// computed result (a host read through the resident store triggers D2H).
    fn flushToHost(rstore: *ResidentTensorStore, id: TensorId) ExecuteProgramError!void {
        const hs = rstore.tensorStore();
        const meta = hs.meta(id) catch return error.ExecutionFailed;
        const total = totalTiles(meta);
        var i: usize = 0;
        while (i < total) : (i += 1) {
            const t = hs.acquireTileConstLinear(id, i) catch return error.ExecutionFailed;
            hs.releaseConst(t.token);
        }
    }

    // ---- vtable ----

    /// Per-`executeProgram` execution state. Most steps just record into the
    /// current frame; control flow (If/Loop) needs host-visible predicates.
    /// Reading a predicate costs a CPU<->GPU round-trip ONLY when the value was
    /// produced on device (see `readPredicate`): a host-resident predicate is
    /// read directly with no stall, and a device-produced one pays exactly one
    /// device poll (inside the D2H readback), not two. `Frame.records` lets a
    /// fixed-trip Loop record its whole unrolled body into one frame and submit
    /// once, with zero syncs.
    const Runner = struct {
        gb: *Self,
        op_ctx: context.Ctx,
        prog: *const ExecutableProgram,
        frame: Frame,

        /// Submit the pending frame WITHOUT waiting, then start a fresh frame.
        /// No-op when nothing is recorded. We deliberately do not poll here:
        /// queue submissions execute in order, so a later D2H readback (whose own
        /// poll waits for all prior work) or a later frame sees these results
        /// without a redundant stall. Callers that need the result on the host
        /// follow this with `readI32Scalar`, whose `copyD2H` performs the single
        /// necessary poll.
        fn submitPending(r: *Runner) ExecuteProgramError!void {
            if (r.frame.records == 0) return;
            r.frame.submit();
            r.frame.deinit();
            r.frame = Frame.init(r.gb.allocator, r.gb.gpu) catch return error.ExecutionFailed;
        }

        /// Read an i32 control-flow predicate to the host with the minimum GPU
        /// stall. If the value was produced on device, submit the pending frame
        /// (so the producing dispatch — which may still sit unsubmitted in the
        /// current frame — is queued) and pull it down; the D2H readback polls
        /// exactly once. If it is host-resident, read it directly: no submit, no
        /// device poll, and the current frame keeps batching.
        fn readPredicate(r: *Runner, id: TensorId) ExecuteProgramError!i32 {
            if (r.op_ctx.rstore.tileDeviceDirty(id, 0)) try r.submitPending();
            return r.readI32Scalar(id);
        }

        /// Host-read an i32 scalar (element 0 of tile 0). When the tile is
        /// device-dirty this triggers a D2H readback (a single device poll);
        /// otherwise it reads current host bytes with no GPU work.
        fn readI32Scalar(r: *Runner, id: TensorId) ExecuteProgramError!i32 {
            const hs = r.op_ctx.rstore.tensorStore();
            const meta = hs.meta(id) catch return error.ExecutionFailed;
            if (meta.dtype != .i32) return error.ExecutionFailed;
            const tile = hs.acquireTileConstLinear(id, 0) catch return error.ExecutionFailed;
            defer hs.releaseConst(tile.token);
            if (tile.bytes.len < 4) return error.ExecutionFailed;
            const ptr: *align(1) const i32 = @ptrCast(tile.bytes.ptr);
            return ptr.*;
        }

        /// Device-side equivalent of the CPU's `copyTensorLists`: same-layout
        /// tensors, tile-for-tile buffer copies (stays in-frame, no sync).
        fn copyLists(r: *Runner, dst: []const TensorId, src: []const TensorId) ExecuteProgramError!void {
            const hs = r.op_ctx.rstore.tensorStore();
            for (dst, src) |dst_id, src_id| {
                const dst_meta = hs.meta(dst_id) catch return error.ExecutionFailed;
                const src_meta = hs.meta(src_id) catch return error.ExecutionFailed;
                if (dst_meta.dtype != src_meta.dtype or dst_meta.rank != src_meta.rank) return error.ExecutionFailed;
                for (dst_meta.shape, src_meta.shape) |a, b| if (a != b) return error.ExecutionFailed;
                for (dst_meta.tile_shape, src_meta.tile_shape) |a, b| if (a != b) return error.ExecutionFailed;
                for (dst_meta.tile_counts, src_meta.tile_counts) |a, b| if (a != b) return error.ExecutionFailed;

                const total = totalTiles(dst_meta);
                var ti: usize = 0;
                while (ti < total) : (ti += 1) {
                    const st = r.op_ctx.rstore.acquireTileDeviceConstLinear(src_id, ti) catch return error.ExecutionFailed;
                    const dt = r.op_ctx.rstore.acquireTileDeviceMutLinear(dst_id, ti) catch return error.ExecutionFailed;
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

        /// Store-level swaps (the resident store migrates device residency with
        /// the swap, so this is zero-copy and safe mid-frame: already-recorded
        /// commands hold their buffer references; future acquires resolve to
        /// the swapped buffers).
        fn swapLists(r: *Runner, a: []const TensorId, b: []const TensorId) ExecuteProgramError!void {
            const hs = r.op_ctx.rstore.tensorStore();
            for (a, b) |a_id, b_id| {
                if (a_id == b_id) continue;
                hs.swapTensors(a_id, b_id) catch return error.ExecutionFailed;
            }
        }

        fn runBlock(r: *Runner, block_id: executable.BlockId) ExecuteProgramError!void {
            const idx: usize = @intCast(block_id);
            if (idx >= r.prog.blocks.len) return error.ExecutionFailed;
            for (r.prog.blocks[idx].steps) |step| try r.runStep(step);
        }

        fn runStep(r: *Runner, step: executable.Step) ExecuteProgramError!void {
            const op_ctx = r.op_ctx;
            const frame = &r.frame;
            switch (step) {
                .ElemwiseBinaryTiled => |s| try simple_ops.execElemwiseBinary(op_ctx, frame, s),
                .BroadcastLastDimBinaryTiled => |s| try simple_ops.execBroadcastLastDim(op_ctx, frame, s),
                .UnaryTiled => |s| try simple_ops.execUnary(op_ctx, frame, s),
                .CopyTiled => |s| try simple_ops.execCopy(op_ctx, frame, s),
                .CastTiled => |s| try simple_ops.execCast(op_ctx, frame, s),
                .MatMulTiled => |s| try r.gb.matmul.exec(op_ctx, frame, s),
                .MatMulNTTiled => |s| try r.gb.nt.exec(op_ctx, frame, s, r.gb.matmul.generated),
                .SoftmaxTiled => |s| try rowwise.execSoftmax(op_ctx, frame, s),
                .RMSNormTiled => |s| try rowwise.execNorm(op_ctx, frame, .rmsnorm, s),
                .LayerNormTiled => |s| try rowwise.execNorm(op_ctx, frame, .layernorm, s),
                .ReduceAll => |s| try rowwise.execReduceAll(op_ctx, frame, s),
                .ReduceAxis => |s| try rowwise.execReduceAxis(op_ctx, frame, s),
                .GatherRowsTiled => |s| try decode_ops.execGatherRows(op_ctx, frame, s),
                .RoPE1DTiled => |s| try decode_ops.execRoPE(op_ctx, frame, s),
                .KVCacheAppendTiled => |s| try decode_ops.execKVCacheAppend(op_ctx, frame, s),
                .AttentionTiled => |s| try attention_exec.execAttention(op_ctx, frame, s),
                .MultiHeadAttentionTiled => |s| try attention_exec.execAttention(op_ctx, frame, s),
                .MultiHeadAttentionCachedTiled => |s| try attention_exec.execAttentionCached(op_ctx, frame, s),
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

                .If => |s| {
                    // Read the predicate to the host, stalling only if it was
                    // produced on device (see `readPredicate`).
                    const take_then = (try r.readPredicate(s.cond)) != 0;
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
                        const raw = try r.readPredicate(tid);
                        if (raw < 0) return error.ExecutionFailed;
                        requested = @intCast(raw);
                    }
                    const max_iters = @min(requested, s.static_max_trip_count);

                    var iter: usize = 0;
                    while (iter < max_iters) : (iter += 1) {
                        if (s.check_before) {
                            if (s.cond) |cid| {
                                if ((try r.readPredicate(cid)) == 0) break;
                            }
                        }
                        try r.runBlock(s.body_block);
                        try r.swapLists(s.carried[0..carried_count], s.body_carried_outputs[0..carried_count]);
                        if (!s.check_before) {
                            if (s.cond) |cid| {
                                if ((try r.readPredicate(cid)) == 0) break;
                            }
                        }
                    }
                },
            }
        }
    };

    fn runProgram(self: *Self, rstore: *ResidentTensorStore, prog: *const ExecutableProgram) ExecuteProgramError!void {
        var runner: Runner = .{
            .gb = self,
            .op_ctx = .{
                .gpu = self.gpu,
                .devmem = &self.devmem,
                .pipes = &self.pipes,
                .allocator = self.allocator,
                .rstore = rstore,
                .scratch = &self.scratch,
            },
            .prog = prog,
            .frame = try Frame.init(self.allocator, self.gpu),
        };
        defer runner.frame.deinit();

        for (prog.steps) |step| try runner.runStep(step);

        runner.frame.submit();

        if (self.flush_outputs) {
            for (prog.outputs) |oid| try flushToHost(rstore, oid);
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
