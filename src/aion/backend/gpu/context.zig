// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! `Ctx` — the bundle of backend handles an op's exec code needs, passed by value
//! per `executeProgram`. Lets per-op modules (`exec/simple_ops.zig`, `exec/matmul.zig`)
//! stay decoupled from `GpuBackend` (no import of `backend.zig`, no circular dep):
//! the backend builds a `Ctx` from its fields and hands it to each op handler.

const std = @import("std");
const wgpu = @import("wgpu.zig");
const wgpu_dm = @import("device_memory.zig");
const pipelines = @import("pipelines.zig");
const device_store = @import("../../runtime/device_store.zig");
const tensor_store = @import("../../runtime/tensor_store.zig");
const Frame = @import("frame.zig").Frame;

pub const Ctx = struct {
    gpu: *wgpu.Gpu,
    devmem: *wgpu_dm.WgpuDeviceMemory,
    pipes: *pipelines.Pipelines,
    allocator: std.mem.Allocator,
    /// Device-only data capability. It cannot return host byte slices.
    store: device_store.DeviceStore,
    /// Explicit bridge for the few control values consumed while recording.
    control: ControlTransfers,
    /// Shared grow-only device scratch for multi-stage kernels (two-stage
    /// reductions, split-K attention partials). Safe to reuse across steps in
    /// one frame: dispatches in a pass are ordered, and each multi-stage op
    /// consumes its partials before the next op overwrites them.
    scratch: *ScratchPool,
};

pub const I32Lease = struct {
    vals: []align(1) const i32,
    token: usize,
    host: tensor_store.TensorStore,

    pub fn release(self: I32Lease) void {
        self.host.releaseConst(self.token);
    }
};

/// Narrow host-value capability. Host bytes never appear on `Ctx.store`, so an
/// operation that consumes one must name that crossing by calling this API.
///
/// There is no residency check here: the compiler has already placed every
/// operand read through this on the CPU, inserting a `Transfer` when the device
/// writes it. Reading is therefore just a read — if a value could be stale, the
/// program would not have compiled.
pub const ControlTransfers = struct {
    host: tensor_store.TensorStore,

    /// Whether `id` is host-placed. The gate for an op carrying both a device
    /// kernel and a host-index fallback: placement already chose which applies,
    /// so read that choice instead of re-deriving it from shapes — a CPU mirror
    /// has the source's exact geometry and would fool any such test.
    pub fn isHostPlaced(self: ControlTransfers, id: tensor_store.TensorId) bool {
        return (self.host.deviceChunk(id, 0) catch null) == null;
    }

    pub fn readI32(self: ControlTransfers, id: tensor_store.TensorId) error{ExecutionFailed}!I32Lease {
        // The compiler placed this on the host; a device backing here means a
        // record-time gate disagreed with the compile-time declaration that is
        // supposed to mirror it, and the bytes below would be stale. Cheap
        // enough to keep in release: control reads are already off the hot path.
        if (!self.isHostPlaced(id)) return error.ExecutionFailed;

        const view = self.host.acquireConst(id) catch return error.ExecutionFailed;
        if (view.dtype != .i32 or view.bytes.len % @sizeOf(i32) != 0) {
            self.host.releaseConst(view.token);
            return error.ExecutionFailed;
        }
        const ptr: [*]align(1) const i32 = @ptrCast(view.bytes.ptr);
        return .{ .vals = ptr[0 .. view.bytes.len / @sizeOf(i32)], .token = view.token, .host = self.host };
    }
};

/// Grow-only pooled device buffer (same pattern as MatmulNt's dequant scratch,
/// which stays separate because matmul interleaves scratch use across N chunks).
pub const ScratchPool = struct {
    buf: ?wgpu.c.WGPUBuffer = null,
    cap: u64 = 0,

    pub fn deinit(self: *ScratchPool) void {
        if (self.buf) |b| wgpu.fns.wgpuBufferRelease(b);
        self.* = undefined;
    }

    /// Get a storage buffer of at least `bytes` (MiB-rounded, monotonic). The
    /// previous buffer (if outgrown) is released — recorded bind groups keep it
    /// alive until their frame completes.
    pub fn ensure(self: *ScratchPool, gpu: *wgpu.Gpu, bytes: u64) error{ExecutionFailed}!wgpu.c.WGPUBuffer {
        if (self.buf) |b| {
            if (self.cap >= bytes) return b;
            wgpu.fns.wgpuBufferRelease(b);
            self.buf = null;
            self.cap = 0;
        }
        const MiB: u64 = 1024 * 1024;
        const cap = (bytes + MiB - 1) / MiB * MiB;
        // Storage for compute stages + copy src/dst so it can also serve as a
        // packed staging buffer for view materialization.
        const usage = wgpu.c.WGPUBufferUsage_Storage | wgpu.c.WGPUBufferUsage_CopySrc | wgpu.c.WGPUBufferUsage_CopyDst;
        const b = wgpu.createBuffer(gpu.device, cap, usage) catch return error.ExecutionFailed;
        self.buf = b;
        self.cap = cap;
        return b;
    }
};

/// Integer ceil-div (workgroup-count helper), shared by op handlers.
pub fn ceilDiv(a: u32, b: u32) u32 {
    return (a + b - 1) / b;
}

/// WebGPU caps workgroups per grid dimension at 65535. 1D elementwise kernels
/// are grid-strided, so we dispatch at most this many groups and let each
/// thread loop; row-per-workgroup kernels must check rows against the cap.
pub const MAX_GROUPS_1D: u32 = 32768;
pub const MAX_GROUPS_PER_DIM: u32 = 65535;

pub fn storageBindingFits(ctx: Ctx, bytes: usize) bool {
    const n = std.math.cast(u64, bytes) orelse return false;
    return n <= ctx.gpu.limits.max_storage_binding_bytes;
}
