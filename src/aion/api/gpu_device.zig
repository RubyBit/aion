// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! GPU device bundle for the public API.
//!
//! This file imports the wgpu-backed GPU backend, so it must ONLY be reached
//! under `build_options.enable_gpu` (see `context.zig`'s gated alias). When the
//! GPU feature is off it is never analyzed and wgpu is neither fetched nor linked.
const std = @import("std");

const gpu_backend = @import("../backend/gpu/backend.zig");
const plan_mod = @import("../graph/plan.zig");
const device = @import("device.zig");
const dm = @import("../runtime/device_memory.zig");

/// Re-export so callers can enumerate adapters (`gpu_device.wgpu.listAdapters`).
pub const wgpu = gpu_backend.wgpu;

pub const GpuCreateError = error{ OutOfMemory, BackendUnavailable };

/// A heap-pinned GPU device: owns the wgpu device/queue and the `GpuBackend`
/// (which stores a `*wgpu.Gpu` into this bundle, so the bundle must not move).
pub const GpuDevice = struct {
    gpu: wgpu.Gpu,
    backend: gpu_backend.GpuBackend,
    policy: plan_mod.TilePolicy,

    /// The device-memory interface used to migrate tensors onto this GPU. Its
    /// `ctx` points into this (pinned) bundle.
    pub fn deviceMemory(self: *GpuDevice) dm.DeviceMemory {
        return self.backend.devmem.device();
    }
};

fn mapOptions(opts: device.GpuOptions) wgpu.Options {
    return .{
        .power = switch (opts.power) {
            .default => .default,
            .low => .low,
            .high => .high,
        },
        .backend = switch (opts.backend) {
            .any => .any,
            .vulkan => .vulkan,
            .d3d12 => .d3d12,
            .metal => .metal,
            .gl => .gl,
        },
        .adapter_index = opts.adapter_index,
    };
}

/// Create a heap-pinned GPU device. Construction order is load-bearing: `gpu` is
/// built first so `GpuBackend.init(&bundle.gpu)` captures a stable pointer.
/// A missing adapter / device (headless machine, no driver) → `BackendUnavailable`.
pub fn create(allocator: std.mem.Allocator, opts: device.GpuOptions) GpuCreateError!*GpuDevice {
    const bundle = allocator.create(GpuDevice) catch return error.OutOfMemory;
    errdefer allocator.destroy(bundle);

    bundle.gpu = wgpu.Gpu.init(mapOptions(opts)) catch return error.BackendUnavailable;
    errdefer bundle.gpu.deinit();

    bundle.backend = gpu_backend.GpuBackend.init(allocator, &bundle.gpu);
    bundle.policy = plan_mod.tilePolicyForTarget(.{ .kind = .webgpu });
    // Cap tile sizes to what this device can actually bind (e.g. a multi-GB
    // quantized embedding table must be split along its row axis).
    bundle.policy.max_binding_bytes = bundle.gpu.limits.max_storage_binding_bytes;
    return bundle;
}

/// Tear down in reverse order: the backend frees device buffers (needs a live
/// wgpu device), then the wgpu device/queue is released, then the heap bundle.
pub fn destroy(allocator: std.mem.Allocator, bundle: *GpuDevice) void {
    bundle.backend.deinit();
    bundle.gpu.deinit();
    allocator.destroy(bundle);
}
