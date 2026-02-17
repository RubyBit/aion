const std = @import("std");

const backend_mod = @import("../backend/backend.zig");
const manager_mod = @import("../storage/manager.zig");
const graph_mod = @import("../graph/graph.zig");
const program_mod = @import("../graph/program.zig");

pub const InitError = error{ InvalidArgument, OutOfMemory } || std.Thread.SpawnError;

pub const CompileError = program_mod.CompileError;

pub const ExecuteError = backend_mod.ExecuteProgramError || manager_mod.StorageError;

pub const ApiError = error{ InvalidArgument, OutOfMemory } || graph_mod.GraphError || manager_mod.StorageError || CompileError || ExecuteError;
