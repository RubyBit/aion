// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Growing a sequence cache's capacity axis at run time.
//!
//! `cache.zig` holds the policy — how much to grow by, and up to what ceiling. This is
//! the execution: reallocating the axis, preserving what was already written, and zeroing
//! the rest. It needs the tensor table, the device memory and the policy at once, so it
//! sits beside the manager rather than inside it (the same split as `fold.zig`).
//!
//! Capacity is the one piece of model state the compiler cannot own, which is why the
//! runtime drives this from a `GrowthRequest` list rather than from the schedule — see
//! `api.Model.prepareGrowableCaches`, and note the host must know the write position
//! before the frame is recorded or there is no size to grow to.

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

/// Grow a device-resident tensor without leaving the device: allocate the
/// new backing, zero it, and move the old contents across with `copyD2D`.
///
/// Applies to the single, unpadded tile that role-declared sequence caches
/// use — there the two backings differ only in the stride of `axis`, so the
/// move is `outer` contiguous runs. Returns false for any other layout,
/// leaving the caller its host round-trip.
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

    // Layouts device growth does not cover round-trip through the host (D2H
    // gather -> host grow -> H2D re-migrate), reusing `moveTensor`. Growth is
    // geometric, so this amortizes to O(final size) and never touches the
    // fixed/ring fast path. `moveTensor` frees the old device buffer only
    // after the re-upload, and the tensor id is preserved.
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
