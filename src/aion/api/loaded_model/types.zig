// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const manager_mod = @import("../../storage/manager.zig");
const package_file = @import("../../storage/aion_file.zig");
const plan_mod = @import("../../graph/plan.zig");
const api_tensor = @import("../tensor.zig");
const device_mod = @import("../device.zig");

pub const Tensor = api_tensor.Tensor;
pub const StorageManager = manager_mod.StorageManager;
pub const TensorId = manager_mod.TensorId;
pub const DType = package_file.DType;
pub const TilePolicy = plan_mod.TilePolicy;
pub const Package = package_file.Package;

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
};

pub const invalid_alias_index: u32 = std.math.maxInt(u32);
pub const invalid_tensor_id: TensorId = std.math.maxInt(TensorId);
