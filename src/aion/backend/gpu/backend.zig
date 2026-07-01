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
//!   - Device residency (`ResidentTensorStore` over `WgpuDeviceMemory`) is held
//!     on the backend and PERSISTS across `executeProgram` calls, keyed on the
//!     host store's identity. Weights are never host-written, so they upload once
//!     and stay device-resident across decode steps; inputs re-upload when the
//!     host rewrites them. (The v0 backend rebuilt residency every call, so every
//!     tensor re-uploaded each run.)
//!   - Each `executeProgram` records all its dispatches into a single `Frame`
//!     (one command encoder) and submits ONCE, then flushes program outputs back
//!     to the host store (D2H).
//!   - WGSL kernels are cached via `Pipelines` (module per kernel, pipeline +
//!     bind-group layout per entry point -- reflected once, not per dispatch).
//!
//! Kernel coverage: ElemwiseBinaryTiled + UnaryTiled (f32, multi-tile) and
//! MatMulTiled (f32, rank-2, shared-memory tiled GEMM). Everything else returns
//! `error.Unsupported`. Quantized paths, batched matmul, and the remaining decode
//! kernels (RMSNorm/RoPE/softmax/attention/KV-append) are planned follow-ups.

const std = @import("std");

// In-tree: relative imports into the aion module (no `@import("aion")`), gated
// into the build only when `-Dgpu` is set (see root.zig). `wgpu` is the one
// genuine module import (the translate-c'd C bindings).
pub const wgpu = @import("wgpu.zig"); // re-exported so apps reach Gpu/Options/c
const wgpu_dm = @import("device_memory.zig");
const pipelines_mod = @import("pipelines.zig");
const context = @import("context.zig");
const simple_ops = @import("simple_ops.zig");
const matmul_exec = @import("matmul/exec.zig");
const frame_mod = @import("frame.zig");
const backend_mod = @import("../backend.zig");
const types = @import("../types.zig");
const executable = @import("../../runtime/executable.zig");
const tensor_store_mod = @import("../../runtime/tensor_store.zig");
const resident_mod = @import("../../runtime/residency/resident_store.zig");
const thread_pool_mod = @import("../../runtime/thread_pool.zig");

const c = wgpu.c;

const Backend = backend_mod.Backend;
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

    /// Persistent device residency. Built lazily on first use and reused while the
    /// same host store is passed (so weights stay resident across decode steps).
    resident: ?ResidentTensorStore = null,
    /// Identity of the host store `resident` decorates (its `ctx` pointer). A
    /// different store means stale residency -- rebuild.
    resident_ctx: ?*anyopaque = null,

    /// When false, `executeProgram` records + submits the compute but skips the
    /// device->host flush of outputs. Used by benchmarks to time pure kernel
    /// throughput without paying a full-output D2H readback every iteration.
    flush_outputs: bool = true,

    /// Matmul executor owns the per-shape autotune cache.
    matmul: matmul_exec.Matmul,

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
        if (self.resident) |*r| r.deinit();
        self.matmul.deinit();
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

    /// Ensure `self.resident` decorates `store`, rebuilding if the store changed.
    /// Returns a pointer to the live resident store.
    fn ensureResident(self: *Self, store: TensorStore) *ResidentTensorStore {
        if (self.resident == null or self.resident_ctx != store.ctx) {
            if (self.resident) |*r| r.deinit();
            self.resident = ResidentTensorStore.init(self.allocator, store, self.devmem.device());
            self.resident_ctx = store.ctx;
        }
        return &self.resident.?;
    }

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

    fn executeProgramImpl(ctx: *anyopaque, prog: *const ExecutableProgram, store: TensorStore) ExecuteProgramError!void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        // Wire the D2H memcpy pool now that `self` is at its final address (init
        // returns by value, so a pointer taken there would dangle). Idempotent.
        self.devmem.pool = if (self.pool) |*p| p else null;
        const rstore = self.ensureResident(store);

        var frame = try Frame.init(self.allocator, self.gpu);
        defer frame.deinit();
        const op_ctx: context.Ctx = .{
            .gpu = self.gpu,
            .devmem = &self.devmem,
            .pipes = &self.pipes,
            .allocator = self.allocator,
            .rstore = rstore,
        };

        for (prog.steps) |step| {
            switch (step) {
                .ElemwiseBinaryTiled => |s| try simple_ops.execElemwiseBinary(op_ctx, &frame, s),
                .UnaryTiled => |s| try simple_ops.execUnary(op_ctx, &frame, s),
                .MatMulTiled => |s| try self.matmul.exec(op_ctx, &frame, s),
                else => return error.Unsupported,
            }
        }

        frame.submit();

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
        .executeProgram = executeProgramImpl,
    };
};
