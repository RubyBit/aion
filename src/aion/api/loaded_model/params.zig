// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! A model's parameters: which graph values are weights, and which tensor each one is.
//!
//! One representation for both model sources. A package's value records say which
//! values are initializers; a builder graph's values carry their tensor in `Value.external`.
//! Either way the answer is "tensor per graph value", and that is what everything above
//! the loader wants — binding a fresh specialization, swapping a weight by name, and
//! attributing memory.
//!
//! The file's separate initializer section, and the `initializer_index` that names a slot
//! in it, exist because weight BYTES want to be one contiguous streamable region. That is
//! a serialization concern and it stops at the importer: nothing above here can tell that
//! a package was ever involved.

const std = @import("std");

const manager_mod = @import("../../storage/manager.zig");

const TensorId = manager_mod.TensorId;

pub const invalid: TensorId = std.math.maxInt(TensorId);

pub const Params = struct {
    /// The tensor backing graph value `i`, or `invalid` when value `i` is not a
    /// parameter. Indexed by value so both sources fill it the same way.
    by_value: []TensorId,

    pub fn init(allocator: std.mem.Allocator, value_count: usize) error{OutOfMemory}!Params {
        const by_value = try allocator.alloc(TensorId, value_count);
        @memset(by_value, invalid);
        return .{ .by_value = by_value };
    }

    pub fn deinit(self: *Params, allocator: std.mem.Allocator) void {
        allocator.free(self.by_value);
        self.* = undefined;
    }

    pub fn set(self: *Params, value: u32, tid: TensorId) void {
        self.by_value[@intCast(value)] = tid;
    }

    /// The tensor for graph value `value`, or null if it is not a parameter.
    pub fn get(self: Params, value: u32) ?TensorId {
        const idx: usize = @intCast(value);
        if (idx >= self.by_value.len) return null;
        const tid = self.by_value[idx];
        return if (tid == invalid) null else tid;
    }

    /// Whether `tid` backs any parameter. Used for memory attribution, where the
    /// question is about the tensor rather than the value.
    pub fn contains(self: Params, tid: TensorId) bool {
        for (self.by_value) |id| if (id == tid) return true;
        return false;
    }
};
