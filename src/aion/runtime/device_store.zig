// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Device-only storage capability handed to GPU operation code.  Deliberately
//! exposes metadata and opaque device handles, never host byte slices.

const std = @import("std");
const tensor_store = @import("tensor_store.zig");
const types = @import("../backend/types.zig");

pub const TensorId = tensor_store.TensorId;
pub const StoreError = tensor_store.StoreError;
pub const TensorMeta = tensor_store.TensorMeta;
pub const SequenceCachePolicyInfo = tensor_store.SequenceCachePolicyInfo;

pub const TileRef = struct {
    handle: u64,
    offset: usize,
    len: usize,
    dtype: types.DType,
    rank: u8,
    shape_mem: [tensor_store.INLINE_RANK]usize,
    strides_mem: [tensor_store.INLINE_RANK]isize,
    token: usize = 0,
};

pub const DeviceStore = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        meta: *const fn (*anyopaque, TensorId) StoreError!TensorMeta,
        acquireConst: *const fn (*anyopaque, TensorId, usize) StoreError!TileRef,
        acquireMut: *const fn (*anyopaque, TensorId, usize) StoreError!TileRef,
        releaseConst: *const fn (*anyopaque, usize) void,
        releaseMut: *const fn (*anyopaque, usize) void,
        sequenceCachePolicyInfo: *const fn (*anyopaque, TensorId) SequenceCachePolicyInfo,
        mapSequenceStep: *const fn (*anyopaque, TensorId, usize, usize) StoreError!usize,
    };

    pub fn meta(self: DeviceStore, id: TensorId) StoreError!TensorMeta {
        return self.vtable.meta(self.ctx, id);
    }

    pub fn acquireTileDeviceConstLinear(self: DeviceStore, id: TensorId, tile: usize) StoreError!TileRef {
        return self.vtable.acquireConst(self.ctx, id, tile);
    }

    pub fn acquireTileDeviceMutLinear(self: DeviceStore, id: TensorId, tile: usize) StoreError!TileRef {
        return self.vtable.acquireMut(self.ctx, id, tile);
    }

    pub fn releaseConst(self: DeviceStore, token: usize) void {
        self.vtable.releaseConst(self.ctx, token);
    }

    pub fn releaseMut(self: DeviceStore, token: usize) void {
        self.vtable.releaseMut(self.ctx, token);
    }

    pub fn sequenceCachePolicyInfo(self: DeviceStore, id: TensorId) SequenceCachePolicyInfo {
        return self.vtable.sequenceCachePolicyInfo(self.ctx, id);
    }

    pub fn mapSequenceStep(self: DeviceStore, id: TensorId, logical: usize, capacity: usize) StoreError!usize {
        return self.vtable.mapSequenceStep(self.ctx, id, logical, capacity);
    }
};

/// Whether `T`, or anything reachable through it, is a slice of raw bytes.
fn exposesHostBytes(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .pointer => |p| p.size == .slice and p.child == u8,
        .optional => |o| exposesHostBytes(o.child),
        .error_union => |e| exposesHostBytes(e.payload),
        .@"struct" => |s| blk: {
            inline for (s.field_types) |field| {
                if (exposesHostBytes(field)) break :blk true;
            }
            break :blk false;
        },
        else => false,
    };
}

// The invariant behind splitting this store out of `TensorStore`: GPU operation
// code holds only device handles, so it cannot read a value the device is still
// computing. Asserted over the vtable itself rather than over method names, so
// adding a host-byte accessor fails here instead of quietly reopening the hole.
test "no device store method can return host bytes" {
    // The host tile ref is what must never appear here, and proves the detector
    // detects rather than passing vacuously.
    try std.testing.expect(exposesHostBytes(tensor_store.StoreError!tensor_store.TileRefConst));

    inline for (@typeInfo(DeviceStore.VTable).@"struct".field_types) |field| {
        const signature = @typeInfo(@typeInfo(field).pointer.child).@"fn";
        try std.testing.expect(!exposesHostBytes(signature.return_type.?));
    }
}
