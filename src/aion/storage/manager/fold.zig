// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Reading, writing and collecting weights an optimization pass folded together.
//!
//! `derived.zig` holds the provenance — which sources went into which result, and where
//! each source's bytes landed. These are the operations over it, which need both that
//! table and the tensors themselves, so they live beside the manager rather than in it:
//! the manager owns the tensor table and where bytes live, and everything that is a
//! POLICY over the table sits above it. `graph/program/lease.zig` is the same shape.

const std = @import("std");

const manager_mod = @import("../manager.zig");
const derived_mod = @import("../derived.zig");
const storage_mod = @import("../storage.zig");

const StorageManager = manager_mod.StorageManager;
const TensorId = manager_mod.TensorId;
const StorageError = storage_mod.StorageError;
const TiledTensor = storage_mod.TiledTensor;

const RegionDir = enum { into_derived, out_of_derived };

/// Copy one source region between a derived weight's packed bytes and its own.
fn copyRegion(whole: []u8, part: []u8, view: derived_mod.View, dir: RegionDir) void {
    const n = view.len * view.block_bytes;
    var r: usize = 0;
    while (r < view.rows) : (r += 1) {
        const at_whole = (r * view.row_stride + view.offset) * view.block_bytes;
        const at_part = r * n;
        switch (dir) {
            .into_derived => @memcpy(whole[at_whole..][0..n], part[at_part..][0..n]),
            .out_of_derived => @memcpy(part[at_part..][0..n], whole[at_whole..][0..n]),
        }
    }
}

/// Copy `src`'s bytes into the region of the derived weight that `id` occupies.
///
/// The derived weight is the canonical store — `id`'s own buffer may have been
/// reclaimed — so a swap lands here and takes effect on the next run with no
/// recompile.
pub fn writeDerivedSource(mgr: *StorageManager, id: TensorId, src: TensorId) StorageError!void {
    var it = mgr.derived.locations(id);
    var found = false;
    while (it.next()) |at| {
        try requireSourceLayout(mgr, at, src);

        const whole = mgr.allocator.alloc(u8, at.view.derivedBytes()) catch return StorageError.OutOfMemory;
        defer mgr.allocator.free(whole);
        const part = mgr.allocator.alloc(u8, at.view.sourceBytes()) catch return StorageError.OutOfMemory;
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

    const part = mgr.allocator.alloc(u8, at.view.sourceBytes()) catch return StorageError.OutOfMemory;
    defer mgr.allocator.free(part);

    try readDerivedPacked(mgr, at, part);
    try mgr.writePackedAtPlacement(dst, part);
}

/// One source's stripe of a derived weight, packed. The derived weight is never
/// itself derived (`Table.record` refuses stacking), so this recurses no further.
pub fn readDerivedPacked(mgr: *StorageManager, at: derived_mod.Located, out: []u8) StorageError!void {
    if (out.len != at.view.sourceBytes()) return StorageError.InvalidArgument;
    const whole = mgr.allocator.alloc(u8, at.view.derivedBytes()) catch return StorageError.OutOfMemory;
    defer mgr.allocator.free(whole);
    try mgr.readPackedAtPlacement(at.result, whole);
    copyRegion(whole, out, at.view, .out_of_derived);
}

/// Give a folded-away weight its own bytes back, out of the derived weight.
///
/// The inverse of `collectDerived`: folding leaves a source with metadata and no
/// backing, which is sound only while every program reads the derived weight instead,
/// so a program that names the source itself unfolds it first.
///
/// No-op for a tensor that still owns its bytes, and for one that was never folded —
/// a program names plenty of tensors with no backing yet (all of its workspace), and
/// this is called across the whole set.
pub fn unfoldTensor(mgr: *StorageManager, id: TensorId) StorageError!void {
    if (try mgr.tensorHasBacking(id)) return;
    if (mgr.derivedLocate(id) == null) return;
    try mgr.reserveHostBacking(id, try mgr.tensorLogicalBackingBytes(id));
    return mgr.readDerivedSource(id, id);
}

/// Drop what the derivations no longer earn. Run after any change to the program
/// reference counts (see `graph/program/lease.zig`).
///
/// Two rules, and the order between them is the whole design:
///  1. A derived weight no program names, whose sources all hold their own bytes
///     again, is redundant — free it and forget the derivation. This is what gives a
///     derived weight an owner at all: it outlives the compile that made it, so
///     nothing else is in a position to collect it.
///  2. Otherwise the derived weight is the canonical store, so a source no program
///     names directly does not need a second copy of its bytes.
///
/// Checking (1) first is what makes them agree: (2) is exactly the reason a source
/// would be missing bytes, so evaluating it first would make (1) unreachable.
pub fn collectDerived(mgr: *StorageManager) void {
    var i: usize = 0;
    while (i < mgr.derived.entries.items.len) {
        const e = mgr.derived.entries.items[i];
        if (mgr.tensorProgramRefs(e.result) == 0 and sourcesAreWhole(mgr, e)) {
            mgr.releaseTensorData(e.result) catch {};
            mgr.derived.remove(mgr.allocator, i);
            continue;
        }
        for (e.sources) |s| {
            if (mgr.tensorProgramRefs(s.tid) == 0) mgr.releaseTensorData(s.tid) catch {};
        }
        i += 1;
    }
}

/// Every source of `e` holds its own bytes, so the derived weight is not the only
/// copy of anything.
fn sourcesAreWhole(mgr: *const StorageManager, e: derived_mod.Entry) bool {
    for (e.sources) |s| {
        if (!(mgr.tensorHasBacking(s.tid) catch return false)) return false;
    }
    return true;
}

/// The tensor standing in for a folded-away weight must match it byte for byte:
/// same dtype, and exactly the bytes the view describes.
fn requireSourceLayout(mgr: *const StorageManager, at: derived_mod.Located, other: TensorId) StorageError!void {
    const result = try mgr.getConst(at.result);
    const t = try mgr.getConst(other);
    if (t.dtype != result.dtype) return StorageError.InvalidArgument;
    if (try t.packedByteLen() != at.view.sourceBytes()) return StorageError.InvalidArgument;
}
