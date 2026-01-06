//! Public entrypoint for the Aion library.
//!
//! Consumers typically do:
//!   const aion = @import("aion");

pub const backend = @import("aion/backend/backend.zig");
pub const cpu = @import("aion/backend/cpu/cpu_backend.zig");
pub const storage = @import("aion/storage/storage.zig");
