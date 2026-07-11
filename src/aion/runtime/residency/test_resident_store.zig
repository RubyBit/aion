// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Exercises `ResidentTensorStore` + `MockDeviceMemory`: host<->device
//! round-trip, dirty-tracking elision of redundant transfers, device-write ->
//! host-read flush, and swapTensors carrying device residency. No GPU needed.

const std = @import("std");

const manager_mod = @import("../../storage/manager.zig");
const tensor_store = @import("../tensor_store.zig");
const device_memory = @import("device_memory.zig");
const resident = @import("resident_store.zig");

const TensorId = manager_mod.TensorId;

fn writeF32(mgr: *manager_mod.StorageManager, id: TensorId, vals: []const f32) !void {
    try mgr.writeFromPackedScalar(id, std.mem.sliceAsBytes(vals));
}

fn readF32(store: tensor_store.TensorStore, id: TensorId, out: []f32) !void {
    const tile = try store.acquireTileConstLinear(id, 0);
    defer store.releaseConst(tile.token);
    const src = std.mem.bytesAsSlice(f32, tile.bytes);
    @memcpy(out, src[0..out.len]);
}

test "resident store: H2D round-trip + dirty-skip" {
    const a = std.testing.allocator;
    var mgr = manager_mod.StorageManager.init(a);
    defer mgr.deinit();

    const id = try mgr.createTiledTensor(.f32, &[_]usize{4}, &[_]usize{4}, .{});
    try writeF32(&mgr, id, &[_]f32{ 1, 2, 3, 4 });

    var mock = device_memory.MockDeviceMemory.init(a);
    defer mock.deinit();
    var rstore = resident.ResidentTensorStore.init(a, mgr.tensorStore(), mock.device());
    defer rstore.deinit();
    const store = rstore.tensorStore();

    // First device read uploads once. Device-acquire is a concrete method on the
    // decorator; release rides the vtable (token-flag dispatch).
    const d0 = try rstore.acquireTileDeviceConstLinear(id, 0);
    store.releaseConst(d0.token);
    try std.testing.expectEqual(@as(usize, 1), mock.h2d_count);
    try std.testing.expect(d0.handle != 0);
    try std.testing.expectEqual(@as(usize, 16), d0.len);

    // Second device read with no intervening host write: no extra H2D.
    const d1 = try rstore.acquireTileDeviceConstLinear(id, 0);
    store.releaseConst(d1.token);
    try std.testing.expectEqual(@as(usize, 1), mock.h2d_count);

    // The device buffer holds the host data.
    var dev_view: [4]f32 = undefined;
    try mock.device().copyD2H(std.mem.sliceAsBytes(dev_view[0..]), d1.handle, 0);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 3, 4 }, &dev_view);
}

test "resident store: host write re-uploads on next device read" {
    const a = std.testing.allocator;
    var mgr = manager_mod.StorageManager.init(a);
    defer mgr.deinit();

    const id = try mgr.createTiledTensor(.f32, &[_]usize{4}, &[_]usize{4}, .{});
    try writeF32(&mgr, id, &[_]f32{ 1, 1, 1, 1 });

    var mock = device_memory.MockDeviceMemory.init(a);
    defer mock.deinit();
    var rstore = resident.ResidentTensorStore.init(a, mgr.tensorStore(), mock.device());
    defer rstore.deinit();
    const store = rstore.tensorStore();

    const d0 = try rstore.acquireTileDeviceConstLinear(id, 0);
    store.releaseConst(d0.token);
    try std.testing.expectEqual(@as(usize, 1), mock.h2d_count);

    // Host write through the decorator marks the tile dirty.
    {
        const mut = try store.acquireTileMutLinear(id, 0);
        const dst = std.mem.bytesAsSlice(f32, mut.bytes);
        for (dst) |*x| x.* = 9;
        store.releaseMut(mut.token);
    }

    // Next device read must re-upload the new host data.
    const d1 = try rstore.acquireTileDeviceConstLinear(id, 0);
    store.releaseConst(d1.token);
    try std.testing.expectEqual(@as(usize, 2), mock.h2d_count);

    var dev_view: [4]f32 = undefined;
    try mock.device().copyD2H(std.mem.sliceAsBytes(dev_view[0..]), d1.handle, 0);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 9, 9, 9, 9 }, &dev_view);
}

test "resident store: device write flushes to host on host read" {
    const a = std.testing.allocator;
    var mgr = manager_mod.StorageManager.init(a);
    defer mgr.deinit();

    const id = try mgr.createTiledTensor(.f32, &[_]usize{4}, &[_]usize{4}, .{});
    try writeF32(&mgr, id, &[_]f32{ 0, 0, 0, 0 });

    var mock = device_memory.MockDeviceMemory.init(a);
    defer mock.deinit();
    var rstore = resident.ResidentTensorStore.init(a, mgr.tensorStore(), mock.device());
    defer rstore.deinit();
    const store = rstore.tensorStore();

    // Device "kernel" writes the tile: acquire mut, write the device buffer, release.
    const dm = try rstore.acquireTileDeviceMutLinear(id, 0);
    try mock.device().copyH2D(dm.handle, 0, std.mem.sliceAsBytes(@constCast(&[_]f32{ 5, 6, 7, 8 })));
    store.releaseMut(dm.token);

    // Host read must observe the device write (one D2H flush).
    var host_view: [4]f32 = undefined;
    try readF32(store, id, host_view[0..]);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 5, 6, 7, 8 }, &host_view);
    try std.testing.expectEqual(@as(usize, 1), mock.d2h_count);

    // Second host read with no new device write: no extra D2H.
    try readF32(store, id, host_view[0..]);
    try std.testing.expectEqual(@as(usize, 1), mock.d2h_count);
}

test "resident store: unified memory does zero copies + stays coherent" {
    const a = std.testing.allocator;
    var mgr = manager_mod.StorageManager.init(a);
    defer mgr.deinit();

    const id = try mgr.createTiledTensor(.f32, &[_]usize{4}, &[_]usize{4}, .{});
    try writeF32(&mgr, id, &[_]f32{ 1, 2, 3, 4 });

    var mock = device_memory.MockDeviceMemory.initWithModel(a, .unified);
    defer mock.deinit();
    var rstore = resident.ResidentTensorStore.init(a, mgr.tensorStore(), mock.device());
    defer rstore.deinit();
    const store = rstore.tensorStore();

    // Device read: binds the host allocation directly — no upload.
    const d0 = try rstore.acquireTileDeviceConstLinear(id, 0);
    store.releaseConst(d0.token);
    try std.testing.expectEqual(@as(usize, 0), mock.h2d_count);

    // The device handle aliases the host bytes (same memory).
    const aliased = mock.bufferSlice(d0.handle).?;
    var alias_view: [4]f32 = undefined;
    @memcpy(std.mem.sliceAsBytes(alias_view[0..]), aliased);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 3, 4 }, &alias_view);

    // "Device kernel" writes through that shared memory (UMA: no transfer).
    const dm = try rstore.acquireTileDeviceMutLinear(id, 0);
    const dev_mem = mock.bufferSlice(dm.handle).?;
    for (std.mem.bytesAsSlice(f32, dev_mem)) |*x| x.* = 7;
    store.releaseMut(dm.token);

    // Host read observes the device write with NO device->host copy.
    var host_view: [4]f32 = undefined;
    try readF32(store, id, host_view[0..]);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 7, 7, 7, 7 }, &host_view);

    // The whole point: unified memory never copies.
    try std.testing.expectEqual(@as(usize, 0), mock.h2d_count);
    try std.testing.expectEqual(@as(usize, 0), mock.d2h_count);
}

test "resident store: swapTensors carries device residency" {
    const a = std.testing.allocator;
    var mgr = manager_mod.StorageManager.init(a);
    defer mgr.deinit();

    const ta = try mgr.createTiledTensor(.f32, &[_]usize{4}, &[_]usize{4}, .{});
    const tb = try mgr.createTiledTensor(.f32, &[_]usize{4}, &[_]usize{4}, .{});
    try writeF32(&mgr, ta, &[_]f32{ 1, 2, 3, 4 });
    try writeF32(&mgr, tb, &[_]f32{ 5, 6, 7, 8 });

    var mock = device_memory.MockDeviceMemory.init(a);
    defer mock.deinit();
    var rstore = resident.ResidentTensorStore.init(a, mgr.tensorStore(), mock.device());
    defer rstore.deinit();
    const store = rstore.tensorStore();

    // Make both tensors device-resident.
    store.releaseConst((try rstore.acquireTileDeviceConstLinear(ta, 0)).token);
    store.releaseConst((try rstore.acquireTileDeviceConstLinear(tb, 0)).token);

    try store.swapTensors(ta, tb);

    // After swap, ta's logical data is {5,6,7,8}; a host read must see it
    // (host buffers swapped, and device residency followed so no stale upload).
    var va: [4]f32 = undefined;
    var vb: [4]f32 = undefined;
    try readF32(store, ta, va[0..]);
    try readF32(store, tb, vb[0..]);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 5, 6, 7, 8 }, &va);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 3, 4 }, &vb);

    // A device read of ta must now serve {5,6,7,8} without a fresh upload
    // (it was already resident pre-swap; residency moved with the data).
    const h2d_before = mock.h2d_count;
    const da = try rstore.acquireTileDeviceConstLinear(ta, 0);
    store.releaseConst(da.token);
    try std.testing.expectEqual(h2d_before, mock.h2d_count);
    var dev_view: [4]f32 = undefined;
    try mock.device().copyD2H(std.mem.sliceAsBytes(dev_view[0..]), da.handle, 0);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 5, 6, 7, 8 }, &dev_view);
}

test "resident store: growable kv cache flushes and refreshes device residency" {
    const a = std.testing.allocator;
    var mgr = try manager_mod.StorageManager.initWithCache(a, .{ .ram_budget_bytes = 1 << 20 });
    defer mgr.deinit();

    const cache = try mgr.createTiledTensor(.f32, &[_]usize{ 1, 1, 2, 1 }, &[_]usize{ 1, 1, 2, 1 }, .{});
    try mgr.registerSequenceCachePolicy(cache, .{ .growable = .{
        .initial_capacity_tokens = 2,
        .growth_numerator = 2,
        .growth_denominator = 1,
    } });

    try writeF32(&mgr, cache, &[_]f32{ 1, 2 });

    var mock = device_memory.MockDeviceMemory.init(a);
    defer mock.deinit();
    var rstore = resident.ResidentTensorStore.init(a, mgr.tensorStore(), mock.device());
    defer rstore.deinit();
    const store = rstore.tensorStore();

    const d0 = try rstore.acquireTileDeviceMutLinear(cache, 0);
    try mock.device().copyH2D(d0.handle, 0, std.mem.sliceAsBytes(@constCast(&[_]f32{ 10, 20 })));
    store.releaseMut(d0.token);

    _ = try store.mapSequenceStep(cache, 3, 2);

    const meta = try store.meta(cache);
    try std.testing.expectEqual(@as(usize, 4), meta.shape[2]);

    var host: [4]f32 = undefined;
    try mgr.readToPackedScalar(cache, std.mem.sliceAsBytes(host[0..]));
    try std.testing.expectEqualSlices(f32, &[_]f32{ 10, 20, 0, 0 }, host[0..]);
    try std.testing.expectEqual(@as(usize, 1), mock.d2h_count);

    const d1 = try rstore.acquireTileDeviceConstLinear(cache, 0);
    store.releaseConst(d1.token);
    try std.testing.expectEqual(@as(usize, 16), d1.len);
    try std.testing.expect(d1.handle != d0.handle);
    try std.testing.expectEqual(@as(usize, 3), mock.h2d_count);
}
