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

/// High-level, user-friendly API (graph-first under the hood, but graph-hidden).
pub const api = @import("aion/api/api.zig");

pub const runtime = @import("aion/runtime/executable.zig");
pub const tensor_store = @import("aion/runtime/tensor_store.zig");

/// Publicly exposed backend types and helpers.
///
/// These are kept as separate modules so consumers (and our own tooling like
/// `src/bench.zig`) can access core ABI types without importing internal files
/// into multiple Zig modules (which Zig disallows).
pub const types = @import("aion/backend/types.zig");
pub const utils = @import("aion/backend/utils.zig");

/// C ABI entrypoints (for Python/Rust/etc.).
///
/// This module defines `pub export` functions with `callconv(.c)`.
pub const c_api = @import("aion/c_api.zig");

// Zig compilation is lazy; force the module to be analyzed so its exported
// symbols are emitted into the library even if the Zig-level `c_api` constant
// is never referenced by Zig callers.
comptime {
    _ = c_api;
}
