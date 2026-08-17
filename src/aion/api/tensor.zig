// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const types = @import("../backend/types.zig");
const backend_utils = @import("../backend/utils.zig");
const manager_mod = @import("../storage/manager.zig");
const api_tiling = @import("tiling.zig");
const device_mod = @import("device.zig");

pub const DType = types.DType;
pub const TensorId = manager_mod.TensorId;
pub const StorageManager = manager_mod.StorageManager;
pub const StorageError = manager_mod.StorageError;
pub const DeviceRef = manager_mod.DeviceRef;
pub const DeviceSelector = device_mod.DeviceSelector;

/// User-visible owned tensor handle.
///
/// In v0 this is always backed by a `StorageManager`-owned `TiledTensor`.
pub const Tensor = struct {
    store: *StorageManager,
    id: TensorId,

    /// Cached metadata (borrowed from the underlying `TiledTensor`).
    dtype: DType,
    shape: []const usize,

    const Self = @This();

    pub fn tensorId(self: Self) TensorId {
        return self.id;
    }

    pub fn getDType(self: Self) DType {
        return self.dtype;
    }

    pub fn getShape(self: Self) []const usize {
        return self.shape;
    }

    /// Migrate this tensor to `sel` (move semantics: the source-device copy is
    /// freed), re-tiling to the target device's geometry. `.gpu = i` requires that
    /// GPU to be registered on the owning `Context`. Idempotent when already there.
    /// After the move, host `read`/`write` require migrating back with `.to(.cpu)`.
    pub fn to(self: *Self, sel: DeviceSelector) StorageError!void {
        const target: DeviceRef = switch (sel) {
            .cpu => .{ .kind = .cpu },
            .gpu => |idx| blk: {
                if (idx > 255) return StorageError.InvalidArgument;
                break :blk .{ .kind = .gpu, .index = @intCast(idx) };
            },
        };

        const t = try self.store.getConst(self.id);
        if (t.device.eql(target)) return;
        if (@as(usize, t.rank) > api_tiling.MAX_RANK) return StorageError.InvalidArgument;

        const policy = self.store.policyFor(target);
        const dev = self.store.deviceMemoryFor(target);
        if (target.kind != .cpu and dev == null) return StorageError.InvalidArgument; // GPU not registered

        var tile_mem: [api_tiling.MAX_RANK]usize = undefined;
        const tile_shape = tile_mem[0..@as(usize, t.rank)];
        api_tiling.chooseTileShapeForTensor(policy, t.dtype, t.shape, t.quant_axis, tile_shape) catch return StorageError.InvalidArgument;

        try self.store.moveTensor(self.id, target, dev, tile_shape, policy.tile_alignment);

        // The metadata slice may have been reallocated by the re-tile; refresh caches.
        const nt = try self.store.getConst(self.id);
        self.dtype = nt.dtype;
        self.shape = nt.shape;
    }

    /// The device this tensor is currently resident on.
    ///
    /// A tensor lives on exactly one device at a time; migrate with `to`.
    pub fn device(self: Self) StorageError!DeviceRef {
        return self.store.tensorDevice(self.id);
    }

    pub fn elemCount(self: Self) StorageError!usize {
        var count: usize = 1;
        for (self.shape) |d| {
            count = std.math.mul(usize, count, d) catch return StorageError.InvalidArgument;
        }
        return count;
    }

    pub fn dtypeOf(comptime T: type) ?DType {
        return switch (T) {
            f16 => .f16,
            f32 => .f32,
            i8 => .i8,
            i32 => .i32,
            else => null,
        };
    }

    pub fn sliceElemType(comptime SliceT: type) ?type {
        return switch (@typeInfo(SliceT)) {
            .pointer => |p| if (p.size == .slice) p.child else null,
            .array => |a| a.child,
            else => null,
        };
    }

    fn arrayInfoIfPointerToArray(comptime T: type) ?std.builtin.Type.Array {
        const pinfo = switch (@typeInfo(T)) {
            .pointer => |p| p,
            else => return null,
        };
        if (pinfo.size != .one) return null;
        return switch (@typeInfo(pinfo.child)) {
            .array => |a| a,
            else => null,
        };
    }

    pub fn writePackedScalar(self: Self, packed_bytes: []const u8) StorageError!void {
        return self.store.writeFromPackedScalar(self.id, packed_bytes);
    }

    pub fn readPackedScalar(self: Self, out: []u8) StorageError!void {
        return self.store.readToPackedScalar(self.id, out);
    }

    pub fn writePackedQuant(self: Self, packed_bytes: []const u8) StorageError!void {
        return self.store.writeFromPackedQuant(self.id, packed_bytes);
    }

    pub fn readPackedQuant(self: Self, out: []u8) StorageError!void {
        return self.store.readToPackedQuant(self.id, out);
    }

    pub fn packedByteLen(self: Self) StorageError!usize {
        const elems = try self.elemCount();
        return backend_utils.requiredBytesForElems(self.dtype, elems) catch StorageError.InvalidArgument;
    }

    pub fn copyFrom(self: Self, allocator: std.mem.Allocator, src: Self) StorageError!void {
        if (self.dtype != src.dtype) return StorageError.InvalidArgument;
        if (self.shape.len != src.shape.len) return StorageError.InvalidArgument;
        var i: usize = 0;
        while (i < self.shape.len) : (i += 1) {
            if (self.shape[i] != src.shape[i]) return StorageError.InvalidArgument;
        }

        const dst_tensor = try self.store.getMut(self.id);
        const src_tensor = try src.store.getConst(src.id);
        if (canRawCopyTiled(dst_tensor, src_tensor)) {
            @memcpy(dst_tensor.data, src_tensor.data);
            return;
        }

        const byte_len = try src.packedByteLen();
        // Hot-path optimization: avoid heap alloc/free for small copies
        // (e.g. recurrent state tensors copied each run).
        const stack_cap: usize = 4096;
        if (byte_len <= stack_cap) {
            var stack_buf: [stack_cap]u8 = undefined;
            const buf: []u8 = stack_buf[0..byte_len];
            if (self.dtype.info().is_quantized) {
                try src.readPackedQuant(buf);
                try self.writePackedQuant(buf);
            } else {
                try src.readPackedScalar(buf);
                try self.writePackedScalar(buf);
            }
            return;
        }

        const heap_buf = allocator.alloc(u8, byte_len) catch return StorageError.OutOfMemory;
        defer allocator.free(heap_buf);

        if (self.dtype.info().is_quantized) {
            try src.readPackedQuant(heap_buf);
            try self.writePackedQuant(heap_buf);
        } else {
            try src.readPackedScalar(heap_buf);
            try self.writePackedScalar(heap_buf);
        }
    }

    fn canRawCopyTiled(dst: *const manager_mod.TiledTensor, src: *const manager_mod.TiledTensor) bool {
        if (dst.dtype != src.dtype) return false;
        if (dst.rank != src.rank) return false;
        if (!std.mem.eql(usize, dst.shape, src.shape)) return false;
        if (!std.mem.eql(usize, dst.tile_shape, src.tile_shape)) return false;
        if (!std.mem.eql(usize, dst.tile_counts, src.tile_counts)) return false;
        if (!std.mem.eql(usize, dst.tile_offsets, src.tile_offsets)) return false;
        if (!std.mem.eql(usize, dst.tile_lens, src.tile_lens)) return false;
        if (dst.data.len != src.data.len) return false;
        return true;
    }

    /// Write typed scalar tensor data (inferred from `values` element type).
    ///
    /// Today this supports f16/f32 (and can be extended as more dtypes are added).
    pub fn write(self: Self, values: anytype) StorageError!void {
        const V: type = @TypeOf(values);

        // Accept: []T / []const T
        if (sliceElemType(V)) |Elem| {
            const dt: DType = dtypeOf(Elem) orelse return StorageError.InvalidArgument;
            if (self.dtype != dt) return StorageError.InvalidArgument;
            const want: usize = try self.elemCount();

            // If it's an array value, slice it; otherwise it must be a slice and have .len.
            switch (@typeInfo(V)) {
                .array => {
                    const s = values[0..];
                    if (s.len != want) return StorageError.InvalidArgument;
                    return self.writePackedScalar(std.mem.sliceAsBytes(s));
                },
                .pointer => {
                    if (values.len != want) return StorageError.InvalidArgument;
                    return self.writePackedScalar(std.mem.sliceAsBytes(values));
                },
                else => return StorageError.InvalidArgument,
            }
        }

        // Accept: *[N]T / *const [N]T
        if (arrayInfoIfPointerToArray(V)) |ainfo| {
            const Elem: type = ainfo.child;
            const dt: DType = dtypeOf(Elem) orelse return StorageError.InvalidArgument;
            if (self.dtype != dt) return StorageError.InvalidArgument;

            const want: usize = try self.elemCount();
            const n: usize = @as(usize, @intCast(ainfo.len));
            if (n != want) return StorageError.InvalidArgument;
            const s = values.*[0..n];
            return self.writePackedScalar(std.mem.sliceAsBytes(s));
        }

        return StorageError.InvalidArgument;
    }

    /// Read typed scalar tensor data into `out` (inferred from `out` element type).
    pub fn read(self: Self, out: anytype) StorageError!void {
        const OutT: type = @TypeOf(out);
        const want: usize = try self.elemCount();

        // Accept: []T (mutable)
        if (switch (@typeInfo(OutT)) {
            .pointer => |p| p.size == .slice,
            else => false,
        }) {
            const pinfo = @typeInfo(OutT).pointer;
            if (pinfo.attrs.@"const") return StorageError.InvalidArgument;
            const Elem: type = pinfo.child;
            const dt: DType = dtypeOf(Elem) orelse return StorageError.InvalidArgument;
            if (self.dtype != dt) return StorageError.InvalidArgument;
            if (out.len != want) return StorageError.InvalidArgument;
            return self.readPackedScalar(std.mem.sliceAsBytes(out));
        }

        // Accept: *[N]T (mutable)
        if (arrayInfoIfPointerToArray(OutT)) |ainfo| {
            const pinfo = @typeInfo(OutT).pointer;
            if (pinfo.attrs.@"const") return StorageError.InvalidArgument;

            const Elem: type = ainfo.child;
            const dt: DType = dtypeOf(Elem) orelse return StorageError.InvalidArgument;
            if (self.dtype != dt) return StorageError.InvalidArgument;

            const n: usize = @as(usize, @intCast(ainfo.len));
            if (n != want) return StorageError.InvalidArgument;
            const s = out.*[0..n];
            return self.readPackedScalar(std.mem.sliceAsBytes(s));
        }

        return StorageError.InvalidArgument;
    }

    /// Allocate and read tensor data as a typed slice.
    /// Caller owns the returned memory.
    pub fn readAlloc(self: Self, allocator: std.mem.Allocator, comptime T: type) StorageError![]T {
        const dt: DType = dtypeOf(T) orelse return StorageError.InvalidArgument;
        if (self.dtype != dt) return StorageError.InvalidArgument;
        const n: usize = try self.elemCount();
        const out: []T = allocator.alloc(T, n) catch return StorageError.OutOfMemory;
        errdefer allocator.free(out);
        try self.read(out);
        return out;
    }

    /// Read a scalar tensor (must contain exactly 1 element).
    pub fn readScalar(self: Self, comptime T: type) StorageError!T {
        const dt: DType = dtypeOf(T) orelse return StorageError.InvalidArgument;
        if (self.dtype != dt) return StorageError.InvalidArgument;

        const n: usize = try self.elemCount();
        if (n != 1) return StorageError.InvalidArgument;

        var tmp: [1]T = undefined;
        try self.read(&tmp);
        return tmp[0];
    }

    pub fn writeF32(self: Self, values: []const f32) StorageError!void {
        return self.write(values);
    }

    pub fn readF32(self: Self, out: []f32) StorageError!void {
        return self.read(out);
    }

    pub fn readF32Alloc(self: Self, allocator: std.mem.Allocator) StorageError![]f32 {
        return self.readAlloc(allocator, f32);
    }

    pub fn writeF16(self: Self, values: []const f16) StorageError!void {
        return self.write(values);
    }

    pub fn readF16(self: Self, out: []f16) StorageError!void {
        return self.read(out);
    }

    pub fn readF16Alloc(self: Self, allocator: std.mem.Allocator) StorageError![]f16 {
        return self.readAlloc(allocator, f16);
    }
};
