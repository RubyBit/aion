// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Runtime growth of sequence-cache capacity while preserving existing data.
//! Growth is driven by requests with host-known write positions; `cache.zig` owns policy.

const std = @import("std");

const manager_mod = @import("../manager.zig");
const cache_mod = @import("../cache.zig");
const storage_mod = @import("../storage.zig");
const dm = @import("../../runtime/device_memory.zig");

const StorageManager = manager_mod.StorageManager;
const TensorId = manager_mod.TensorId;
const StorageError = storage_mod.StorageError;
const TiledTensor = storage_mod.TiledTensor;
const DeviceRef = storage_mod.DeviceRef;

/// Zero `bytes` of `handle` from `offset`, in bounded chunks so zeroing a
/// large buffer never needs an equally large host staging allocation.
pub fn zeroDeviceRange(mgr: *StorageManager, dev: dm.DeviceMemory, handle: dm.DeviceHandle, offset: usize, bytes: usize) StorageError!void {
    if (bytes == 0) return;
    const chunk_cap: usize = @min(bytes, 4 << 20);
    const zeros: []u8 = mgr.allocator.alloc(u8, chunk_cap) catch return StorageError.OutOfMemory;
    defer mgr.allocator.free(zeros);
    @memset(zeros, 0);

    var done: usize = 0;
    while (done < bytes) {
        const n = @min(chunk_cap, bytes - done);
        dev.copyH2D(handle, offset + done, zeros[0..n]) catch return StorageError.InvalidArgument;
        done += n;
    }
}

/// Grow a single-tile, unpadded device tensor by zeroing new backing and copying
/// preserved runs with `copyD2D`. Returns false for layouts requiring a host fallback.
fn growAxisOnDevice(mgr: *StorageManager, id: TensorId, axis: usize, new_size: usize, dev: dm.DeviceMemory) StorageError!bool {
    const t: *TiledTensor = try mgr.getMut(id);
    if (t.dtype.info().is_quantized) return false;
    if (t.tile_handles.len != 1 or !std.mem.eql(usize, t.tile_shape, t.shape)) return false;

    const rank: usize = @intCast(t.rank);
    var new_shape_mem: [8]usize = undefined;
    @memcpy(new_shape_mem[0..rank], t.shape);
    new_shape_mem[axis] = new_size;
    const new_shape: []const usize = new_shape_mem[0..rank];

    // Bytes spanned by one index of `axis`, and the runs on each side of it.
    var trailing: usize = t.dtype.info().block_bytes;
    for (t.shape[axis + 1 ..]) |d| trailing = std.math.mul(usize, trailing, d) catch return StorageError.InvalidArgument;
    var outer: usize = 1;
    for (t.shape[0..axis]) |d| outer = std.math.mul(usize, outer, d) catch return StorageError.InvalidArgument;
    const old_run = std.math.mul(usize, t.shape[axis], trailing) catch return StorageError.InvalidArgument;
    const new_run = std.math.mul(usize, new_size, trailing) catch return StorageError.InvalidArgument;
    // Device copies move whole 4-byte words; odd runs (f16) take the host path.
    if (old_run % 4 != 0 or new_run % 4 != 0) return false;

    // Built host-backed for its geometry, then immediately released: the
    // bytes live on the device, same handoff `moveTensor` performs.
    var staging: TiledTensor = undefined;
    try staging.init(mgr.allocator, t.dtype, new_shape, new_shape, .{ .tile_alignment = t.tile_alignment });
    errdefer staging.deinit();
    const total = staging.tile_lens[0];
    staging.releaseData();

    const handle = dev.alloc(total, 64) catch return StorageError.OutOfMemory;
    errdefer dev.free(handle);
    try zeroDeviceRange(mgr, dev, handle, 0, total);
    var i: usize = 0;
    while (i < outer) : (i += 1) {
        dev.copyD2D(handle, i * new_run, t.tile_handles[0], i * old_run, old_run) catch return StorageError.InvalidArgument;
    }

    const handles: []dm.DeviceHandle = mgr.allocator.alloc(dm.DeviceHandle, 1) catch return StorageError.OutOfMemory;
    handles[0] = handle;
    staging.device = t.device;
    staging.tile_handles = handles;
    staging.dev = dev;
    staging.owns_data = false;

    // Frees the old device buffer, which the submitted copy above keeps
    // alive until it retires.
    t.deinit();
    t.* = staging;
    t.shape = t.shape_storage.constSlice();
    t.tile_shape = t.tile_shape_storage.constSlice();
    t.tile_counts = t.tile_counts_storage.constSlice();
    t.tile_strides = t.tile_strides_storage.constSlice();
    return true;
}

pub fn ensureTensorAxisCapacity(mgr: *StorageManager, id: TensorId, axis: usize, min_size: usize) StorageError!void {
    const t0: *const TiledTensor = try mgr.getConst(id);
    if (axis >= @as(usize, t0.rank)) return StorageError.InvalidArgument;
    if (t0.shape[axis] >= min_size) return;
    if (t0.device.kind == .cpu) {
        return (try mgr.getMut(id)).growAxisPreserveScalar(axis, min_size);
    }

    const target: DeviceRef = t0.device;
    if (t0.dev) |d| {
        if (try growAxisOnDevice(mgr, id, axis, min_size, d)) return;
    }

    // Unsupported device layouts round-trip through host growth and remigration.
    // Geometric growth amortizes this to O(final size) while preserving the tensor id.
    const dev: dm.DeviceMemory = t0.dev orelse return StorageError.InvalidArgument;
    const tile_align: usize = t0.tile_alignment;
    const rank: usize = @as(usize, t0.rank);
    var shape_buf: [8]usize = undefined;
    @memcpy(shape_buf[0..rank], t0.shape);

    try mgr.moveTensor(id, .{ .kind = .cpu }, null, shape_buf[0..rank], tile_align);
    try (try mgr.getMut(id)).growAxisPreserveScalar(axis, min_size);

    const grown: *const TiledTensor = try mgr.getConst(id);
    var grown_shape: [8]usize = undefined;
    @memcpy(grown_shape[0..rank], grown.shape);
    try mgr.moveTensor(id, target, dev, grown_shape[0..rank], tile_align);
}

/// Grow rolling storage without changing the logical sequence it represents.
/// Ordinary axis growth preserves physical indices, but modulo indices change
/// when capacity changes, so retained rows must be explicitly rehashed.
pub fn ensureRollingCacheCapacity(
    mgr: *StorageManager,
    id: TensorId,
    min_size: usize,
    retained_history: usize,
    logical_ends: []const usize,
) StorageError!void {
    const before = try mgr.getConst(id);
    if (before.rank != 4 or before.dtype.info().is_quantized) return StorageError.InvalidArgument;
    if (before.shape[0] != logical_ends.len or before.shape[1] == 0) return StorageError.InvalidArgument;
    const old_capacity = before.shape[1];
    if (old_capacity >= min_size) return;

    const old_bytes_len = try before.packedByteLen();
    const old_bytes = mgr.allocator.alloc(u8, old_bytes_len) catch return StorageError.OutOfMemory;
    defer mgr.allocator.free(old_bytes);
    try mgr.readPackedAtPlacement(id, old_bytes);

    try ensureTensorAxisCapacity(mgr, id, 1, min_size);
    const after = try mgr.getConst(id);
    const new_capacity = after.shape[1];
    const new_bytes_len = try after.packedByteLen();
    const new_bytes = mgr.allocator.alloc(u8, new_bytes_len) catch return StorageError.OutOfMemory;
    defer mgr.allocator.free(new_bytes);
    @memset(new_bytes, 0);

    var row_bytes: usize = after.dtype.info().block_bytes;
    row_bytes = std.math.mul(usize, row_bytes, after.shape[2]) catch return StorageError.InvalidArgument;
    row_bytes = std.math.mul(usize, row_bytes, after.shape[3]) catch return StorageError.InvalidArgument;
    for (logical_ends, 0..) |end, batch| {
        const kept = @min(@min(end, retained_history), old_capacity);
        const begin = end - kept;
        var logical = begin;
        while (logical < end) : (logical += 1) {
            const old_row = batch * old_capacity + logical % old_capacity;
            const new_row = batch * new_capacity + logical % new_capacity;
            const old_off = std.math.mul(usize, old_row, row_bytes) catch return StorageError.InvalidArgument;
            const new_off = std.math.mul(usize, new_row, row_bytes) catch return StorageError.InvalidArgument;
            @memcpy(new_bytes[new_off .. new_off + row_bytes], old_bytes[old_off .. old_off + row_bytes]);
        }
    }
    try mgr.writePackedAtPlacement(id, new_bytes);
}
