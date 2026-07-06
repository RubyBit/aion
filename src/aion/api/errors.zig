// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const backend_mod = @import("../backend/backend.zig");
const package_file = @import("../storage/aion_file.zig");
const manager_mod = @import("../storage/manager.zig");
const graph_mod = @import("../graph/graph.zig");
const program_mod = @import("../graph/program.zig");

/// `BackendUnavailable`: a GPU device was requested (`Options.gpus`) but no
/// adapter/device is present, or the library was built without GPU support
/// (`-Dgpu=false`). No silent CPU fallback — the caller decides.
pub const InitError = error{ InvalidArgument, OutOfMemory, BackendUnavailable } || std.Thread.SpawnError;

pub const LoadError = package_file.PackageError || graph_mod.GraphError || manager_mod.StorageError || program_mod.CompileError;

pub const CompileError = program_mod.CompileError;

pub const ExecuteError = backend_mod.ExecuteProgramError || manager_mod.StorageError || program_mod.CompileError || graph_mod.GraphError || package_file.PackageError;

pub const ApiError = error{ InvalidArgument, OutOfMemory } || graph_mod.GraphError || manager_mod.StorageError || CompileError || ExecuteError || LoadError;
