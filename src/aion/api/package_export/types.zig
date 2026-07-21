// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const package_file = @import("../../storage/aion_file.zig");
const manager_mod = @import("../../storage/manager.zig");
const api_builder = @import("../builder.zig");

pub const Builder = api_builder.Builder;
pub const NamedTensorRef = api_builder.NamedTensorRef;
pub const TensorRef = api_builder.TensorRef;
pub const StorageManager = manager_mod.StorageManager;
pub const TensorId = manager_mod.TensorId;
pub const Package = package_file.Package;
pub const max_rank: usize = 8;

pub const Metadata = struct {
    key: []const u8,
    value: []const u8,
};

/// Declares that `output`'s value should be written back into `input`'s slot after
/// each run (recurrent-state carry — KV caches, LSTM h/c). Referenced by tensor, not
/// by name: `input` must be a graph input and `output` must be one of the compiled/
/// exported outputs.
pub const OutputAlias = struct {
    input: TensorRef,
    output: TensorRef,
};

/// Declares the semantic role of a graph input (see `package_file.InputRoleKind`).
/// Roles are persisted in the package and let the runtime auto-allocate caches and
/// auto-drive position/index inputs on load.
pub const InputRoleDecl = struct {
    input: TensorRef,
    kind: package_file.InputRoleKind,
    /// Capacity axis (sequence_cache) or sequence axis (tokens/positions).
    axis: ?u8 = null,
    /// Dim symbol NAME of a free capacity axis; must also be declared on the
    /// builder via `symbolicDim`.
    capacity_symbol: ?[]const u8 = null,
    zero_init: bool = true,
    allow_growable: bool = false,
    allow_ring: bool = false,
};

pub const ExportModelOptions = struct {
    metadata: []const Metadata = &.{},
    output_aliases: []const OutputAlias = &.{},
    input_roles: []const InputRoleDecl = &.{},
};
