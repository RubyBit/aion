// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const manager_mod = @import("../../storage/manager.zig");
const package_file = @import("../../storage/aion_file.zig");
const plan_mod = @import("../../graph/plan.zig");
const opt_mod = @import("../../graph/opt.zig");
const target_mod = @import("../../graph/target.zig");
const api_tensor = @import("../tensor.zig");
const device_mod = @import("../device.zig");

pub const Tensor = api_tensor.Tensor;
pub const StorageManager = manager_mod.StorageManager;
pub const TensorId = manager_mod.TensorId;
pub const DType = package_file.DType;
pub const TilePolicy = plan_mod.TilePolicy;
pub const OptPolicy = opt_mod.Policy;
pub const Target = target_mod.Target;
pub const Package = package_file.Package;
pub const DeviceRef = device_mod.DeviceRef;
pub const SequenceCachePolicy = manager_mod.SequenceCachePolicy;

pub const SignatureInfo = struct {
    name: []const u8,
    value: u32,
    dtype: DType,
    rank: u8,
};

pub const IoAliasInfo = struct {
    input_name: []const u8,
    output_name: []const u8,
    input_index: usize,
    output_index: usize,
};

/// Growth schedule for growable role-declared caches (see `CacheOptions.growable`).
pub const CacheGrowth = struct {
    /// Tokens allocated up front; the runtime grows the slot on demand from here.
    initial_capacity_tokens: usize = 8,
    /// Geometric growth factor (matches the storage `GrowablePolicy` defaults).
    growth_numerator: usize = 3,
    growth_denominator: usize = 2,
};

/// One-shot sizing/policy for sequence-cache inputs declared via package input
/// roles. Applies to every cache whose capacity axis is a free dim symbol; caches
/// with fixed shapes need nothing (auto-init covers them).
pub const CacheOptions = struct {
    /// Capacity in tokens for role-declared caches with a free capacity symbol.
    /// 0 = don't auto-size them (the caller binds those caches explicitly).
    capacity_tokens: usize = 0,
    /// Non-null: allocate the caches small and grow on demand up to
    /// `capacity_tokens`. Honored only for roles flagged `allow_growable` with the
    /// runtime-supported rank-4/axis-1 layout; otherwise the cache is pre-allocated
    /// at full capacity (never a load error).
    growable: ?CacheGrowth = null,
};

pub const LoadModelOptions = struct {
    /// When true (the default), any graph input the caller does not bind before
    /// `run()` is auto-allocated and zero-initialized, with its shape inferred from
    /// the symbol bindings contributed by the inputs that *were* bound. Recurrent
    /// (io-aliased) state inputs additionally carry their contents across runs.
    ///
    /// When false, `run()` preserves strict binding: any unbound input is an error.
    ///
    /// Note: auto-init seeds zeros. Inputs that need a non-zero initial value
    /// (e.g. a "still active" flag or a blank token id) must still be bound
    /// explicitly by the caller.
    auto_init_inputs: bool = true,

    /// Which device to load/run this model on. Defaults to `.cpu` (byte-identical
    /// to the pre-device behavior). `.gpu = i` requires that GPU to have been
    /// registered on the `Context` via `Options.gpus`; weights are tiled for it.
    device: device_mod.DeviceSelector = .cpu,

    /// One-shot sizing/policy for role-declared sequence caches.
    cache: CacheOptions = .{},

    /// Auto-feed and advance role-declared position/index inputs
    /// (cache_write_index, cache_visible_end, positions) from the bound
    /// tokens-role input. Ignored when the package declares no such roles.
    /// A manual `bindInput` on a role input always overrides its auto value.
    auto_positions: bool = true,

    /// Maximum physical bytes retained by exact-shape compiled
    /// specializations. Eviction is LRU and an individual specialization may
    /// exceed the budget (in which case it is retained alone).
    plan_cache_budget_bytes: usize = 256 * 1024 * 1024,

    /// Which optional rewrite passes to run. Null takes `opt.defaults` for the target,
    /// which is what production wants; set it to bisect a pass on a real model.
    passes: ?OptPolicy = null,
};

pub const invalid_alias_index: u32 = std.math.maxInt(u32);
pub const invalid_tensor_id: TensorId = std.math.maxInt(TensorId);
