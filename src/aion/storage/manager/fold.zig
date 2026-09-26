// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Read, write, unfold, and collect optimization-derived weights.
//! `derived.zig` stores provenance; this module applies it to managed tensor storage.

const std = @import("std");

const manager_mod = @import("../manager.zig");
const derived_mod = @import("../derived.zig");
const storage_mod = @import("../storage.zig");

const StorageManager = manager_mod.StorageManager;
const TensorId = manager_mod.TensorId;
const StorageError = storage_mod.StorageError;
const Tensor = storage_mod.Tensor;

const RegionDir = enum { into_derived, out_of_derived };

/// Copy one source's blocks between a derived weight's packed bytes and its own, one
/// block at a time: neither side keeps a column contiguous in the other's order, and
/// the derived side may split a block's bytes.
fn copyRegion(whole: []u8, part: []u8, view: derived_mod.View, dir: RegionDir) void {
    const bb = view.block_bytes;
    for (0..view.cols) |c| {
        for (0..view.blocks) |kb| {
            const block = part[view.sourceAt(c, kb) * bb ..][0..bb];
            switch (dir) {
                .into_derived => view.order.storeBlock(whole, view.blocks, c, kb, block),
                .out_of_derived => view.order.loadBlock(whole, view.blocks, c, kb, block),
            }
        }
    }
}

/// Copy `src` into `id`'s region of its canonical derived weight, allowing swaps
/// after the source's standalone buffer was reclaimed.
pub fn writeDerivedSource(mgr: *StorageManager, id: TensorId, src: TensorId) StorageError!void {
    var it = mgr.derived.locations(id);
    var found = false;
    while (it.next()) |at| {
        try requireSourceLayout(mgr, at, src);

        const whole = mgr.allocator.alloc(u8, at.view.bytes()) catch return StorageError.OutOfMemory;
        defer mgr.allocator.free(whole);
        const part = mgr.allocator.alloc(u8, at.view.bytes()) catch return StorageError.OutOfMemory;
        defer mgr.allocator.free(part);

        try mgr.readPackedAtPlacement(at.result, whole);
        try mgr.readPackedAtPlacement(src, part);
        copyRegion(whole, part, at.view, .into_derived);
        try mgr.writePackedAtPlacement(at.result, whole);
        found = true;
    }
    if (!found) return StorageError.InvalidArgument;
}

/// Read a folded-away weight's current bytes out of its derived weight into `dst`.
pub fn readDerivedSource(mgr: *StorageManager, id: TensorId, dst: TensorId) StorageError!void {
    const at = mgr.derivedLocate(id) orelse return StorageError.InvalidArgument;
    try requireSourceLayout(mgr, at, dst);

    const part = mgr.allocator.alloc(u8, at.view.bytes()) catch return StorageError.OutOfMemory;
    defer mgr.allocator.free(part);

    try readDerivedPacked(mgr, at, part);
    try mgr.writePackedAtPlacement(dst, part);
}

/// A source's bytes out of its derived weight, packed. The derived weight is never
/// itself derived (`Table.record` refuses stacking), so this recurses no further.
pub fn readDerivedPacked(mgr: *StorageManager, at: derived_mod.Located, out: []u8) StorageError!void {
    if (out.len != at.view.bytes()) return StorageError.InvalidArgument;
    const whole = mgr.allocator.alloc(u8, at.view.bytes()) catch return StorageError.OutOfMemory;
    defer mgr.allocator.free(whole);
    try mgr.readPackedAtPlacement(at.result, whole);
    copyRegion(whole, out, at.view, .out_of_derived);
}

/// Restore a folded source's backing from its derived weight when a program names it.
/// No-op if the tensor still has backing or was never folded.
pub fn unfoldTensor(mgr: *StorageManager, id: TensorId) StorageError!void {
    if (try mgr.tensorHasBacking(id)) return;
    if (mgr.derivedLocate(id) == null) return;
    try mgr.reserveHostBacking(id, try mgr.tensorLogicalBackingBytes(id));
    return mgr.readDerivedSource(id, id);
}

/// Collect storage after program reference counts change.
/// First remove an unreferenced derived weight when its source is whole, or when
/// nothing holds or reads its source any more; otherwise reclaim an unreferenced
/// source copy because the derived weight is canonical.
pub fn collectDerived(mgr: *StorageManager) void {
    var i: usize = 0;
    while (i < mgr.derived.entries.items.len) {
        const e = mgr.derived.entries.items[i];
        const source_whole = mgr.tensorHasBacking(e.source) catch false;
        if (mgr.tensorProgramRefs(e.result) == 0 and (source_whole or mgr.tensorUnreferenced(e.source))) {
            mgr.releaseTensorData(e.result) catch {};
            mgr.derived.remove(i);
            continue;
        }
        if (mgr.tensorProgramRefs(e.source) == 0) mgr.releaseTensorData(e.source) catch {};
        i += 1;
    }
}

/// The tensor standing in for a folded-away weight must match it byte for byte:
/// same dtype, and exactly the bytes the view describes.
fn requireSourceLayout(mgr: *const StorageManager, at: derived_mod.Located, other: TensorId) StorageError!void {
    const result = try mgr.getConst(at.result);
    const t = try mgr.getConst(other);
    if (t.dtype != result.dtype) return StorageError.InvalidArgument;
    if (try t.byteLen() != at.view.bytes()) return StorageError.InvalidArgument;
}
