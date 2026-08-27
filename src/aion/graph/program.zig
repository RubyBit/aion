//! Public graph-program facade.
//!
//! The pipeline: `reachability` selects requested work, `compiler` orchestrates
//! lowering, `allocation` owns graph-value-to-logical-tensor mapping, `placement`
//! makes crossings explicit, and `workspace` assigns/materializes physical storage.
//! Optional rewrites at both ends live in `graph/opt.zig`.

const compiler = @import("program/compiler.zig");

pub const StorageError = compiler.StorageError;
pub const TiledTensor = compiler.TiledTensor;
pub const StorageManager = compiler.StorageManager;
pub const TensorId = compiler.TensorId;
pub const Step = compiler.Step;
pub const PlacedStep = compiler.PlacedStep;
pub const Program = compiler.Program;
pub const CompileError = compiler.CompileError;
pub const OptPolicy = compiler.OptPolicy;
pub const optDefaults = @import("opt.zig").defaults;

pub const compileGraph = compiler.compileGraph;
pub const Target = compiler.Target;
pub const materializePlacements = compiler.materializePlacements;
