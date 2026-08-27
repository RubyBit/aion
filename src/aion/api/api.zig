// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! High-level public API for Aion.
//!
//! Design goals (v0):
//! - Graph-first execution model, but graph is hidden behind a builder API.
//! - Owned tensors only (RAM-backed via StorageManager).
//! - No allocations during execution of compiled models.

pub const Context = @import("context.zig").Context;
pub const TilePolicy = @import("context.zig").TilePolicy;
pub const LoadedModel = @import("context.zig").LoadedModel;
pub const Weights = @import("context.zig").Weights;
pub const Tensor = @import("tensor.zig").Tensor;
pub const Builder = @import("builder.zig").Builder;
pub const TensorRef = @import("builder.zig").TensorRef;
pub const NamedTensorRef = @import("builder.zig").NamedTensorRef;
pub const ModuleDyn = @import("module.zig").ModuleDyn;
pub const isModuleType = @import("module.zig").isModuleType;
pub const assertModuleType = @import("module.zig").assertModuleType;
pub const isForwardModuleType = @import("module.zig").isForwardModuleType;
pub const assertForwardModuleType = @import("module.zig").assertForwardModuleType;
pub const moduleDynFrom = @import("module.zig").moduleDynFrom;
pub const moduleTypeName = @import("module.zig").moduleTypeName;
pub const beginModuleScope = @import("module.zig").beginModuleScope;
pub const endModuleScope = @import("module.zig").endModuleScope;
pub const withModuleScope = @import("module.zig").withModuleScope;
pub const Model = @import("context.zig").Model;
pub const LoadModelOptions = @import("context.zig").LoadModelOptions;
pub const CacheOptions = @import("context.zig").CacheOptions;
pub const CacheGrowth = @import("context.zig").CacheGrowth;
pub const ExportModelOptions = @import("context.zig").ExportModelOptions;
pub const CompileOptions = @import("context.zig").CompileOptions;
/// Which optional rewrite passes a compile runs (`LoadModelOptions.passes`,
/// `ExportModelOptions.passes`). Build one with `.empty`, `.full`, `.initOne(.<pass>)`.
pub const OptPolicy = @import("../graph/opt.zig").Policy;
pub const OptPass = @import("../graph/opt.zig").Pass;
pub const Target = @import("../graph/target.zig").Target;
pub const DimSymbol = @import("builder.zig").DimSymbol;
pub const ExportMetadata = @import("context.zig").ExportMetadata;
pub const OutputAlias = @import("context.zig").OutputAlias;
pub const InputRoleDecl = @import("context.zig").InputRoleDecl;
pub const InputRoleKind = @import("context.zig").InputRoleKind;
pub const CacheConfig = @import("context.zig").CacheConfig;
pub const CachePolicy = @import("context.zig").CachePolicy;
pub const SequenceCachePolicy = @import("context.zig").SequenceCachePolicy;
pub const GrowablePolicy = @import("context.zig").GrowablePolicy;
pub const RingPolicy = @import("context.zig").RingPolicy;

pub const DeviceSelector = @import("device.zig").DeviceSelector;
pub const GpuOptions = @import("device.zig").GpuOptions;

pub const nn = @import("nn.zig");

pub const ApiError = @import("errors.zig").ApiError;
pub const InitError = @import("errors.zig").InitError;
pub const LoadError = @import("errors.zig").LoadError;
pub const CompileError = @import("errors.zig").CompileError;
pub const ExecuteError = @import("errors.zig").ExecuteError;
