//! Public entrypoint for the Aion library.
//!
//! Consumers typically do:
//!   const aion = @import("aion");

pub const backend = @import("aion/backend/backend.zig");
pub const cpu = @import("aion/backend/cpu/cpu_backend.zig");
pub const storage = @import("aion/storage/storage.zig");
pub const storage_manager = @import("aion/storage/manager.zig");
pub const graph = @import("aion/graph/graph.zig");
pub const infer = @import("aion/graph/infer.zig");
pub const plan = @import("aion/graph/plan.zig");
pub const program = @import("aion/graph/program.zig");

pub const runtime = @import("aion/runtime/executable.zig");
pub const tensor_store = @import("aion/runtime/tensor_store.zig");

/// Publicly exposed backend types and helpers.
///
/// These are kept as separate modules so consumers (and our own tooling like
/// `src/bench.zig`) can access core ABI types without importing internal files
/// into multiple Zig modules (which Zig disallows).
pub const types = @import("aion/backend/types.zig");
pub const utils = @import("aion/backend/utils.zig");
