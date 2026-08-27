// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Provenance for weights an optimization pass derived from other weights.
//!
//! A pass that rewrites weights (concatenating, folding, repacking) makes the derived
//! tensor the canonical store and lets the model layer reclaim the sources. Two things
//! then have to keep working: a weight swap must land where the program reads, and a read
//! must materialize a weight that no longer has storage of its own.
//!
//! Both are one mechanism here: the pass records WHERE each source's bytes went as a
//! `View`, and the swap/read paths are generic over it. Passes do not implement their own
//! inverse, and the model layer does not switch on which pass ran.
//!
//! `record` refuses a derivation whose source is itself derived: resolving a chain means
//! composing views, and getting that silently wrong writes bytes nothing reads, so
//! stacking is a loud error until a pass actually needs it. One source folded into
//! SEVERAL results is allowed — a weight is derived once per (tiling, device) a target
//! needed — and a swap then has to reach every copy, which is what `locations` is for.

const std = @import("std");

const storage = @import("storage.zig");

const StorageError = storage.StorageError;
const DeviceRef = storage.DeviceRef;
const TensorId = u32;

pub const Kind = enum { column_concat };

/// One source's bytes inside a derived tensor, in packed block space: each row of the
/// derived tensor holds `row_stride` blocks, of which this source owns
/// `[offset, offset+len)`.
pub const View = struct {
    rows: usize,
    row_stride: usize,
    offset: usize,
    len: usize,
    block_bytes: usize,

    pub fn sourceBytes(self: View) usize {
        return self.rows * self.len * self.block_bytes;
    }

    pub fn derivedBytes(self: View) usize {
        return self.rows * self.row_stride * self.block_bytes;
    }
};

pub const Source = struct { tid: TensorId, view: View };

pub const Entry = struct {
    kind: Kind,
    /// The tile shape the result was built for. The bytes are shape-independent but the
    /// LAYOUT is not: tiling is chosen per target, and a quantized weight cannot be
    /// retiled downstream, so a result built for one target must not be reused for
    /// another.
    tiles: []usize,
    /// The device the result was built for. Part of the identity because a tensor is
    /// resident on exactly one device: handing one model's result to a model on another
    /// device would migrate it away from the first, which is a move, not a share.
    device: DeviceRef,
    result: TensorId,
    sources: []Source,
};

pub const Located = struct { result: TensorId, view: View };

pub const Table = struct {
    entries: std.ArrayList(Entry) = .empty,

    pub fn deinit(self: *Table, gpa: std.mem.Allocator) void {
        for (self.entries.items) |e| {
            gpa.free(e.tiles);
            gpa.free(e.sources);
        }
        self.entries.deinit(gpa);
    }

    /// A previously derived result for exactly these sources, at this tiling, on this device.
    pub fn find(
        self: *const Table,
        kind: Kind,
        tiles: []const usize,
        device: DeviceRef,
        sources: []const TensorId,
    ) ?TensorId {
        for (self.entries.items) |e| {
            if (e.kind != kind or e.sources.len != sources.len) continue;
            if (!e.device.eql(device)) continue;
            if (!std.mem.eql(usize, e.tiles, tiles)) continue;
            for (e.sources, sources) |have, want| {
                if (have.tid != want) break;
            } else return e.result;
        }
        return null;
    }

    pub fn record(
        self: *Table,
        gpa: std.mem.Allocator,
        kind: Kind,
        tiles: []const usize,
        device: DeviceRef,
        result: TensorId,
        sources: []const Source,
    ) StorageError!void {
        for (sources) |s| {
            if (self.isDerived(s.tid)) return StorageError.InvalidArgument;
        }
        const tiles_copy = gpa.dupe(usize, tiles) catch return StorageError.OutOfMemory;
        errdefer gpa.free(tiles_copy);
        const src_copy = gpa.dupe(Source, sources) catch return StorageError.OutOfMemory;
        errdefer gpa.free(src_copy);
        self.entries.append(gpa, .{
            .kind = kind,
            .tiles = tiles_copy,
            .device = device,
            .result = result,
            .sources = src_copy,
        }) catch return StorageError.OutOfMemory;
    }

    /// Where `tid`'s bytes live now, if a pass folded it away. Every copy holds the
    /// same bytes, so a read can take the first; a write must use `locations`.
    pub fn locate(self: *const Table, tid: TensorId) ?Located {
        var it = self.locations(tid);
        return it.next();
    }

    pub fn locations(self: *const Table, tid: TensorId) Locations {
        return .{ .entries = self.entries.items, .tid = tid };
    }

    pub const Locations = struct {
        entries: []const Entry,
        tid: TensorId,
        at: usize = 0,

        pub fn next(self: *Locations) ?Located {
            while (self.at < self.entries.len) {
                const e = self.entries[self.at];
                self.at += 1;
                for (e.sources) |s| {
                    if (s.tid == self.tid) return .{ .result = e.result, .view = s.view };
                }
            }
            return null;
        }
    };

    pub fn isDerived(self: *const Table, tid: TensorId) bool {
        for (self.entries.items) |e| if (e.result == tid) return true;
        return false;
    }

    /// Forget derivation `index`. The caller has already established that the result is
    /// redundant — see `StorageManager.collectDerived`, which owns that rule.
    pub fn remove(self: *Table, gpa: std.mem.Allocator, index: usize) void {
        const e = self.entries.orderedRemove(index);
        gpa.free(e.tiles);
        gpa.free(e.sources);
    }
};
