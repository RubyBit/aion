// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Per-tensor facts about a lowered step list, shared by the step-level passes.
//!
//! Two passes run between placement and workspace planning — `alias_views` and
//! `fuse_steps` — and both need the same question answered for a tensor id: who writes
//! it, who reads it, and when. Both also need the same hazard boundary, that anything a
//! control-flow body touches is off the table, because a block replays and can capture
//! outer values, so step indices are not comparable across domains.
//!
//! Deciding this at the STEP level rather than the value level is deliberate. The one
//! previous attempt at workspace aliasing in this repo was reverted because it reasoned
//! about graph values while the hazards live in the emitted schedule.

const std = @import("std");
const executable = @import("../../runtime/executable.zig");
const manager_mod = @import("../../storage/manager.zig");

const TensorId = manager_mod.TensorId;
const Program = executable.ExecutableProgram;
const PlacedStep = executable.PlacedStep;

pub const Error = error{OutOfMemory};

pub const Use = struct {
    writes: usize = 0,
    reads: usize = 0,
    first_touch: usize = std.math.maxInt(usize),
    last_touch: usize = 0,
    last_write: usize = 0,
    /// Touched inside an If/Loop body. Those replay and can capture outer values, so
    /// nothing about them is decided by index.
    in_block: bool = false,
};

pub const Map = std.AutoHashMap(TensorId, Use);

/// Walk the whole program once. Blocks come first, so anything a region body touches is
/// flagged before the top-level walk records comparable indices for it.
pub fn collect(allocator: std.mem.Allocator, prog: *const Program) Error!Map {
    var uses: Map = .init(allocator);
    errdefer uses.deinit();
    for (prog.blocks) |block| try gather(&uses, block.steps, true);
    try gather(&uses, prog.steps, false);
    return uses;
}

fn gather(uses: *Map, steps: []PlacedStep, in_block: bool) Error!void {
    for (steps, 0..) |*placed, i| {
        const walk = executable.tensorUses(&placed.op);
        for (walk.slice()) |use| {
            const gop = uses.getOrPut(use.id.*) catch return error.OutOfMemory;
            if (!gop.found_existing) gop.value_ptr.* = .{};
            const u = gop.value_ptr;
            u.in_block = u.in_block or in_block;
            if (in_block) continue; // indices are not comparable across domains
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
