// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const package_file = @import("../storage/aion_file.zig");

const api_errors = @import("errors.zig");

const types_mod = @import("loaded_model/types.zig");
const params_mod = @import("loaded_model/params.zig");

pub const Tensor = types_mod.Tensor;
pub const StorageManager = types_mod.StorageManager;
pub const TensorId = types_mod.TensorId;
pub const DType = types_mod.DType;
pub const TilePolicy = types_mod.TilePolicy;
pub const Package = types_mod.Package;
pub const Params = params_mod.Params;

/// Weights-only view over an AION package.
///
/// This is intended for the HuggingFace-like workflow:
/// - load a pre-trained backbone's weights (initializers)
/// - bind them into a predefined Zig module/backbone implementation
/// - attach a task-specific head (classification/QA/etc) and compile
///
/// Important: this does *not* use the package graph (nodes) at all.
pub const Weights = struct {
    allocator: std.mem.Allocator,
    store: *StorageManager,
    policy: TilePolicy,
    package: Package,
    params: Params,
    package_hash: u64,

    const Self = @This();

    pub fn deinit(self: *Self) void {
        self.params.deinit(self.allocator);
        self.package.deinit();
        self.* = undefined;
    }

    pub fn debugNames(self: *const Self) []const package_file.DebugName {
        return self.package.debug_names;
    }

    pub fn findValueByDebugName(self: *const Self, name: []const u8) ?u32 {
        for (self.package.debug_names) |entry| {
            if (std.mem.eql(u8, entry.name, name)) return entry.value;
        }
        return null;
    }

    pub fn initializerTensorByValue(self: *Self, value_index: u32) api_errors.ApiError!Tensor {
        const tid: TensorId = self.params.get(value_index) orelse return api_errors.ApiError.InvalidArgument;
        const meta = try self.store.getConst(tid);
        return .{ .store = self.store, .id = tid, .dtype = meta.dtype, .shape = meta.shape };
    }

    pub fn initializerTensorByDebugName(self: *Self, debug_name: []const u8) api_errors.ApiError!Tensor {
        const value_index: u32 = self.findValueByDebugName(debug_name) orelse return api_errors.ApiError.InvalidArgument;
        return self.initializerTensorByValue(value_index);
    }

    pub fn initLoaded(
        allocator: std.mem.Allocator,
        store: *StorageManager,
        policy: TilePolicy,
        package: Package,
        params: Params,
        package_hash: u64,
    ) Self {
        return .{
            .allocator = allocator,
            .store = store,
            .policy = policy,
            .package = package,
            .params = params,
            .package_hash = package_hash,
        };
    }
};
