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

pub const DimensionSymbol = struct {
    tensor: TensorRef,
    axis: usize,
    name: []const u8,
};

pub const OutputAlias = struct {
    input_name: []const u8,
    output_name: []const u8,
};

pub const ExportModelOptions = struct {
    input_symbols: []const DimensionSymbol = &.{},
    metadata: []const Metadata = &.{},
    output_aliases: []const OutputAlias = &.{},
};
