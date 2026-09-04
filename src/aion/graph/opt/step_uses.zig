// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Per-tensor read/write facts; indices are local to one schedule or control-flow body.
//! Tensors touched by multiple lists are unconfined and excluded from local rewrites.

const std = @import("std");
const executable = @import("../../runtime/executable.zig");
const manager_mod = @import("../../storage/manager.zig");

const TensorId = manager_mod.TensorId;
const Program = executable.ExecutableProgram;
const PlacedStep = executable.PlacedStep;

pub const Error = error{OutOfMemory};

/// A step list: 0 is the top-level schedule, `i + 1` is block `i`'s body. The same
/// numbering as `opt.rewriter`'s node lists.
pub const List = usize;

/// Lists in a program: the top-level schedule plus one per block.
pub fn listCount(prog: *const Program) usize {
    return 1 + prog.blocks.len;
}

/// The steps of one list.
pub fn listSteps(prog: *const Program, list: List) []PlacedStep {
    if (list == 0) return prog.steps;
    return prog.blocks[list - 1].steps;
}

pub const Use = struct {
    writes: usize = 0,
    reads: usize = 0,
    first_touch: usize = std.math.maxInt(usize),
    last_touch: usize = 0,
    last_write: usize = 0,
    /// The sole list touching this tensor, or null for multiple lists.
    /// Counts and indices are meaningful only when this is non-null.
    list: ?List = null,
};

pub const Map = std.AutoHashMap(TensorId, Use);

/// Walk every list of the program once, recording each tensor's facts within its own.
pub fn collect(allocator: std.mem.Allocator, prog: *const Program) Error!Map {
    var uses: Map = .init(allocator);
    errdefer uses.deinit();
    var list: List = 0;
    while (list < listCount(prog)) : (list += 1) {
        try gather(&uses, listSteps(prog, list), list);
    }
    return uses;
}

fn gather(uses: *Map, steps: []PlacedStep, list: List) Error!void {
    for (steps, 0..) |*placed, i| {
        const walk = executable.tensorUses(&placed.op);
        for (walk.slice()) |use| {
            const gop = uses.getOrPut(use.id.*) catch return error.OutOfMemory;
            if (!gop.found_existing) gop.value_ptr.* = .{ .list = list };
            const u = gop.value_ptr;
            const own = u.list orelse continue; // already crosses lists
            if (own != list) {
                u.list = null;
                continue;
            }
            u.first_touch = @min(u.first_touch, i);
            u.last_touch = @max(u.last_touch, i);
            switch (use.access) {
                .read => u.reads += 1,
                .write => {
                    u.writes += 1;
                    u.last_write = @max(u.last_write, i);
                },
                .read_write => {
                    u.writes += 1;
                    u.reads += 1;
                    u.last_write = @max(u.last_write, i);
                },
            }
        }
    }
}

pub const OwnedSet = std.AutoHashMap(TensorId, void);

/// The tensors the compiler owns: workspace it may re-point, re-slot or release.
/// Inputs, parameters and model state are deliberately absent.
pub fn ownedSet(allocator: std.mem.Allocator, owned: []const TensorId) Error!OwnedSet {
    var set: OwnedSet = .init(allocator);
    errdefer set.deinit();
    for (owned) |id| set.put(id, {}) catch return error.OutOfMemory;
    return set;
}
