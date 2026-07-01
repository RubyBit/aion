// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! `Ctx` — the bundle of backend handles an op's exec code needs, passed by value
//! per `executeProgram`. Lets per-op modules (`simple_ops.zig`, `matmul/exec.zig`)
//! stay decoupled from `GpuBackend` (no import of `backend.zig`, no circular dep):
//! the backend builds a `Ctx` from its fields and hands it to each op handler.

const std = @import("std");
const wgpu = @import("wgpu.zig");
const wgpu_dm = @import("device_memory.zig");
const pipelines = @import("pipelines.zig");
const resident = @import("../../runtime/residency/resident_store.zig");

pub const Ctx = struct {
    gpu: *wgpu.Gpu,
    devmem: *wgpu_dm.WgpuDeviceMemory,
    pipes: *pipelines.Pipelines,
    allocator: std.mem.Allocator,
    rstore: *resident.ResidentTensorStore,
};

/// Integer ceil-div (workgroup-count helper), shared by op handlers.
pub fn ceilDiv(a: u32, b: u32) u32 {
    return (a + b - 1) / b;
}

pub fn storageBindingFits(ctx: Ctx, bytes: usize) bool {
    const n = std.math.cast(u64, bytes) orelse return false;
    return n <= ctx.gpu.limits.max_storage_binding_bytes;
}
