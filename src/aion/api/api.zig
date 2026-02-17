//! High-level public API for Aion.
//!
//! Design goals (v0):
//! - Graph-first execution model, but graph is hidden behind a builder API.
//! - Owned tensors only (RAM-backed via StorageManager).
//! - No allocations during execution of compiled models.

pub const Context = @import("context.zig").Context;
pub const TilePolicy = @import("context.zig").TilePolicy;
pub const Tensor = @import("tensor.zig").Tensor;
pub const Builder = @import("builder.zig").Builder;
pub const TensorRef = @import("builder.zig").TensorRef;
pub const Model = @import("model.zig").Model;

pub const nn = @import("nn.zig");

pub const ApiError = @import("errors.zig").ApiError;
pub const InitError = @import("errors.zig").InitError;
pub const CompileError = @import("errors.zig").CompileError;
pub const ExecuteError = @import("errors.zig").ExecuteError;
