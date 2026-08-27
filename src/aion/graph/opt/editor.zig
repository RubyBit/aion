// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Step-list editing: the facts a peephole needs, plus one compaction.
//!
//! A rule marks producers dead with `kill` and rewrites consumers in place through
//! `steps`. Collecting uses, the owned set and the compaction are here, so a rule is
//! only its match condition.
//!
//! One editor edits one step list — the top-level schedule or one control-flow body —
//! so a rule matching on adjacency never sees across the boundary, and `confined` is how
//! it refuses a tensor another list can also observe. `opt.stepPasses` walks the lists.

const std = @import("std");

const executable = @import("../../runtime/executable.zig");
const manager_mod = @import("../../storage/manager.zig");
const step_uses = @import("step_uses.zig");

const Program = executable.ExecutableProgram;
const PlacedStep = executable.PlacedStep;
const StorageManager = manager_mod.StorageManager;
const TensorId = manager_mod.TensorId;

pub const Error = error{ InvalidArgument, OutOfMemory };

pub const Editor = struct {
    gpa: std.mem.Allocator,
    mgr: *StorageManager,
    prog: *Program,
    /// The list being edited (see `step_uses.List`).
    list: step_uses.List,
    /// That list's steps. A rule reads and rewrites these, never `prog.steps`.
    steps: []PlacedStep,
    uses: step_uses.Map,
    owned: step_uses.OwnedSet,
    drop: []bool,
    dropped: usize = 0,

    pub fn init(
        gpa: std.mem.Allocator,
        mgr: *StorageManager,
        prog: *Program,
        owned: []const TensorId,
        list: step_uses.List,
    ) Error!Editor {
        var uses = try step_uses.collect(gpa, prog);
        errdefer uses.deinit();
        var set = try step_uses.ownedSet(gpa, owned);
        errdefer set.deinit();
        const steps = step_uses.listSteps(prog, list);
        const drop = gpa.alloc(bool, steps.len) catch return error.OutOfMemory;
        @memset(drop, false);
        return .{
            .gpa = gpa,
            .mgr = mgr,
            .prog = prog,
            .list = list,
            .steps = steps,
            .uses = uses,
            .owned = set,
            .drop = drop,
        };
    }

    pub fn deinit(self: *Editor) void {
        self.uses.deinit();
        self.owned.deinit();
        self.gpa.free(self.drop);
        self.* = undefined;
    }

    pub fn kill(self: *Editor, step: usize) void {
        if (self.drop[step]) return;
        self.drop[step] = true;
        self.dropped += 1;
    }

    pub fn use(self: *const Editor, v: TensorId) ?step_uses.Use {
        return self.uses.get(v);
    }

    pub fn isOwned(self: *const Editor, v: TensorId) bool {
        return self.owned.contains(v);
    }

    pub fn isOutput(self: *const Editor, v: TensorId) bool {
        for (self.prog.outputs) |id| if (id == v) return true;
        return false;
    }

    /// Every step touching `v` is in the list being edited, so its indices are
    /// comparable here and no other list can observe what this rule does to it.
    pub fn confined(self: *const Editor, v: TensorId) bool {
        const u = self.uses.get(v) orelse return false;
        return u.list == self.list;
    }

    /// The one step after `producer` that reads `v`, when `v` is private workspace
    /// written only there and read exactly once. Anything else — a second reader, a
    /// program output, a value another list touches — means the producer stays.
    pub fn soleReader(self: *const Editor, v: TensorId, producer: usize) ?usize {
        if (!self.isOwned(v) or self.isOutput(v)) return null;
        if (!self.confined(v)) return null;
        const u = self.use(v) orelse return null;
        if (u.writes != 1 or u.last_write != producer) return null;
        if (u.reads != 1 or u.first_touch != producer) return null;

        var j = producer + 1;
        while (j < self.steps.len) : (j += 1) {
            const walk = executable.tensorUses(&self.steps[j].op);
            for (walk.slice()) |it| if (it.id.* == v) return j;
        }
        return null;
    }

    /// Compact the edited list, dropping what was killed. Returns how many went.
    pub fn commit(self: *Editor) Error!usize {
        if (self.dropped == 0) return 0;
        const kept = self.gpa.alloc(PlacedStep, self.steps.len - self.dropped) catch return error.OutOfMemory;
        var w: usize = 0;
        for (self.steps, 0..) |step, i| {
            if (self.drop[i]) continue;
            kept[w] = step;
            w += 1;
        }
        self.gpa.free(self.steps);
        if (self.list == 0) {
            self.prog.steps = kept;
        } else {
            self.prog.blocks[self.list - 1].steps = kept;
        }
        self.steps = kept;
        return self.dropped;
    }
};
