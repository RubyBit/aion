// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const manager_mod = @import("../../storage/manager.zig");
const package_file = @import("../../storage/aion_file.zig");
const plan_mod = @import("../../graph/plan.zig");
const api_tensor = @import("../tensor.zig");

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

pub const LoadModelOptions = struct {};

pub const invalid_alias_index: u32 = std.math.maxInt(u32);
pub const invalid_tensor_id: TensorId = std.math.maxInt(TensorId);
