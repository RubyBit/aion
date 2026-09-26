// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Provenance for weights an optimization pass derived from other weights.
//!
//! A pass that re-lays a weight for a kernel (`opt/weight_layout`) makes the derived
//! tensor the canonical store and lets the model layer reclaim the source. Two things
//! then have to keep working: a weight swap must land where the program reads, and a read
//! must materialize a weight that no longer has storage of its own.
//!
//! Both are one mechanism here: the pass records WHERE the source's bytes went as a
//! `View`, and the swap/read paths are generic over it. Passes do not implement their own
//! inverse, and the model layer does not switch on which pass ran.
//!
//! `record` refuses a derivation whose source is itself derived: resolving a chain means
//! composing views, and getting that silently wrong writes bytes nothing reads, so
//! stacking is a loud error until a pass actually needs it. One source derived into
//! SEVERAL results is allowed — a weight is derived once per (order, device) a target
//! needed — and a swap then has to reach every copy, which is what `locations` is for.

const std = @import("std");

const storage = @import("storage.zig");
const types = @import("../backend/types.zig");

const StorageError = storage.StorageError;
const DeviceRef = storage.DeviceRef;
const TensorId = u32;

/// How a source's blocks are arranged inside the derived tensor: the result is
/// `[cols, blocks]` blocks — row `c` holds all `blocks` blocks of output column `c` —
/// laid out in `order`.
pub const View = struct {
    /// Blocks per row of the result (the reduction length in blocks).
    blocks: usize,
    /// Rows of the result (output columns).
    cols: usize,
    block_bytes: usize,
    /// The source is block-major `[blocks, cols]` — its block `(kb, c)` is derived
    /// block `(c, kb)`. Otherwise it is already `[cols, blocks]`.
    transposed: bool,
    order: types.QuantBlockOrder,

    /// Bytes of the source, and equally of the result: a relayout moves blocks, it
    /// never adds any.
    pub fn bytes(self: View) usize {
        return self.blocks * self.cols * self.block_bytes;
    }

    /// Source block index of column `c`'s block `kb`.
    pub fn sourceAt(self: View, c: usize, kb: usize) usize {
        return if (self.transposed) kb * self.cols + c else c * self.blocks + kb;
    }
};

pub const Entry = struct {
    /// The block order the result was built in. Part of the identity because a
    /// quantized weight's order is fixed once laid out: a result built for one target's
    /// kernel must not be handed to another's.
    order: types.QuantBlockOrder,
    /// The device the result was built for. Part of the identity because a tensor is
    /// resident on exactly one device: handing one model's result to a model on another
    /// device would migrate it away from the first, which is a move, not a share.
    device: DeviceRef,
    result: TensorId,
    source: TensorId,
    view: View,
};

pub const Located = struct { result: TensorId, view: View };

pub const Table = struct {
    entries: std.ArrayList(Entry) = .empty,

    pub fn deinit(self: *Table, gpa: std.mem.Allocator) void {
        self.entries.deinit(gpa);
    }

    /// A previously derived result for exactly this source, in this order, on this device.
    pub fn find(self: *const Table, order: types.QuantBlockOrder, device: DeviceRef, source: TensorId) ?TensorId {
        for (self.entries.items) |e| {
            if (e.order == order and e.device.eql(device) and e.source == source) return e.result;
        }
        return null;
    }

    pub fn record(self: *Table, gpa: std.mem.Allocator, device: DeviceRef, result: TensorId, source: TensorId, view: View) StorageError!void {
        if (self.isDerived(source)) return StorageError.InvalidArgument;
        self.entries.append(gpa, .{
            .order = view.order,
            .device = device,
            .result = result,
            .source = source,
            .view = view,
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
                if (e.source == self.tid) return .{ .result = e.result, .view = e.view };
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
    pub fn remove(self: *Table, index: usize) void {
        _ = self.entries.orderedRemove(index);
    }
};
