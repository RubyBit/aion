// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

//! Logical tensor allocation during graph lowering.
//!
//! This maps inferred graph values to tensor ids and records which tensors the
//! compiled program owns. It deliberately knows nothing about executable
//! liveness or physical reuse: `workspace.zig` handles those concerns after all
//! steps have been emitted.

const std = @import("std");
const types = @import("../../backend/types.zig");
const manager_mod = @import("../../storage/manager.zig");
const plan = @import("../plan.zig");

pub const TensorId = manager_mod.TensorId;

pub const Error = error{ InvalidArgument, OutOfMemory } || manager_mod.StorageError;

pub const Context = struct {
    allocator: std.mem.Allocator,
    mgr: *manager_mod.StorageManager,
    policy: plan.TilePolicy,
    value_tensor: []TensorId,
    value_has_tensor: []bool,
    owned_tensors: *std.ArrayList(TensorId),

    pub fn allocTensor(self: *Context, dtype: types.DType, shape: []const usize, tile_shape: []const usize) Error!TensorId {
        const tid = try self.mgr.createTiledTensorMetadata(dtype, shape, tile_shape, .{ .tile_alignment = self.policy.tile_alignment });
        self.owned_tensors.append(self.allocator, tid) catch {
            self.mgr.releaseTensorData(tid) catch {};
            return error.OutOfMemory;
        };
        return tid;
    }

    pub fn ensureValueTensor(self: *Context, value_index: usize, dtype: types.DType, shape: []const usize, tile_shape: []const usize) Error!TensorId {
        if (value_index >= self.value_tensor.len or value_index >= self.value_has_tensor.len) return error.InvalidArgument;
        if (self.value_has_tensor[value_index]) return self.value_tensor[value_index];

        const tid = try self.allocTensor(dtype, shape, tile_shape);
        self.value_tensor[value_index] = tid;
        self.value_has_tensor[value_index] = true;
        return tid;
    }

    pub fn tensorForValue(self: *const Context, value_index: usize) Error!TensorId {
        if (value_index >= self.value_tensor.len or !self.value_has_tensor[value_index]) return error.InvalidArgument;
        return self.value_tensor[value_index];
    }
};
