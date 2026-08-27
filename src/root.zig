// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

//! Public entrypoint for the Aion library.
//!
//! Consumers typically do:
//!   const aion = @import("aion");

pub const backend = @import("aion/backend/backend.zig");
pub const cpu = struct {
    pub const CpuBackend = @import("aion/backend/cpu/cpu_backend.zig").CpuBackend;
    pub const cpuid = @import("aion/backend/cpu/tuning/cpuid.zig");
};
pub const storage = @import("aion/storage/storage.zig");
pub const storage_file = @import("aion/storage/aion_file.zig");
pub const storage_manager = @import("aion/storage/manager.zig");
pub const graph = @import("aion/graph/graph.zig");
pub const infer = @import("aion/graph/infer.zig");
pub const plan = @import("aion/graph/plan.zig");
pub const program = @import("aion/graph/program.zig");
pub const opt = @import("aion/graph/opt.zig");

/// High-level, user-friendly API (graph-first under the hood, but graph-hidden).
pub const api = @import("aion/api/api.zig");

/// Tile-shape choosers the model loader uses. Exposed so a benchmark can tile a
/// synthetic weight exactly the way a loaded model would, instead of re-deriving
/// (and drifting from) the rule.
pub const tiling = @import("aion/api/tiling.zig");

pub const runtime = @import("aion/runtime/executable.zig");
pub const tensor_store = @import("aion/runtime/tensor_store.zig");
pub const profile = @import("aion/profile.zig");

/// WebGPU GPU backend — feature-gated (`-Dgpu` → `build_options.enable_gpu`).
/// When disabled the backend source is never analyzed, so wgpu is not fetched
/// or linked (Rust `#[cfg(feature = "gpu")]` equivalent). When enabled,
/// `build.zig` gives the module the wgpu bindings and native link inputs, and
/// `aion.gpu.GpuBackend` + `aion.gpu.wgpu` (device selection) are available.
pub const gpu = if (@import("build_options").enable_gpu)
    @import("aion/backend/gpu/backend.zig")
else
    struct {};
/// Device-memory abstraction: the seam a GPU backend implements
/// (`DeviceMemory`) to allocate device buffers and transfer bytes across the
/// host boundary. Crossings themselves are explicit `Transfer` steps chosen by
/// the compiler's placement pass — there is no residency or staging layer.
pub const device_memory = @import("aion/runtime/device_memory.zig");

/// Publicly exposed backend types and helpers.
///
/// These are kept as separate modules so consumers (and our own tooling like
/// `src/bench/`, `bench_cpu.zig`) can access core ABI types without importing internal files
/// into multiple Zig modules (which Zig disallows).
pub const types = @import("aion/backend/types.zig");
pub const utils = @import("aion/backend/utils.zig");

/// C ABI entrypoints (for Python/Rust/etc.).
///
/// This module defines `pub export` functions with `callconv(.c)`.
pub const c_api = @import("aion/c_api.zig");

/// Mirrors `build_options.multiversion` for callers that want to inspect whether
/// this `aion` module was built with portable multi-ISA CPU dispatch enabled.
/// The CPU backend reads `build_options` directly; this is just public metadata.
pub const aion_multiversion: bool = @import("build_options").multiversion;

// Zig compilation is lazy; force the module to be analyzed so its exported
// symbols are emitted into the library even if the Zig-level `c_api` constant
// is never referenced by Zig callers.
comptime {
    _ = c_api;
}
