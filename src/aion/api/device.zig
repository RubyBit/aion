// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Device selection types for the public API.
//!
//! This module is dependency-light and ALWAYS compiled (no wgpu import), so
//! `Context.Options` / `LoadModelOptions` stay type-stable regardless of `-Dgpu`.
//! The GPU device bundle itself lives in `gpu_device.zig`, which is only reached
//! under `build_options.enable_gpu`.
const std = @import("std");

const backend_mod = @import("../backend/backend.zig");
const plan_mod = @import("../graph/plan.zig");
const storage_mod = @import("../storage/storage.zig");
const dm = @import("../runtime/residency/device_memory.zig");

pub const DeviceRef = storage_mod.DeviceRef;

/// Which device to place a model / tensor on. `.gpu = i` selects the i-th GPU
/// registered on the `Context` (via `Context.Options.gpus`). Multi-GPU machines
/// index distinct adapters.
pub const DeviceSelector = union(enum) {
    cpu,
    gpu: usize,
};

/// Aion-level GPU request, decoupled from `wgpu.Options` so the public `Options`
/// structs never name a GPU API. Mapped to `wgpu.Options` inside `gpu_device.zig`.
pub const GpuOptions = struct {
    power: enum { default, low, high } = .default,
    backend: enum { any, vulkan, d3d12, metal, gl } = .any,
    /// Pin a specific physical adapter (from `wgpu.listAdapters`). Overrides
    /// `power`/`backend` when set.
    adapter_index: ?usize = null,
};

/// A resolved, transient device handle: the backend to execute on, the tile
/// policy to compile/tile for, and (for non-cpu) the device memory for migration.
/// Computed on demand from a `Context` — never stored long-term.
pub const Device = struct {
    ref: DeviceRef,
    backend: backend_mod.Backend,
    policy: plan_mod.TilePolicy,
    device_memory: ?dm.DeviceMemory,
};
