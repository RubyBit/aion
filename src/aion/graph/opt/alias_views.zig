// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Elide reshape copies between byte-identical layouts while preserving distinct
//! destination metadata and aliasing its backing during workspace planning.

const std = @import("std");

const executable = @import("../../runtime/executable.zig");
const manager_mod = @import("../../storage/manager.zig");
const editor_mod = @import("editor.zig");

const Editor = editor_mod.Editor;
const PlacedStep = executable.PlacedStep;
const StorageManager = manager_mod.StorageManager;
const TensorId = manager_mod.TensorId;

pub const Error = editor_mod.Error;

/// destination -> source. The destination borrows the source's backing.
pub const AliasMap = std.AutoHashMap(TensorId, TensorId);

/// Whether two tensors hold byte-identical layouts: every tensor is packed row-major,
/// so a reshape changes only the shape its bytes are read with.
pub fn layoutsIdentical(mgr: *const StorageManager, a_id: TensorId, b_id: TensorId) bool {
    const a = mgr.getConst(a_id) catch return false;
    const b = mgr.getConst(b_id) catch return false;
    if (a.dtype != b.dtype or a.chunkCount() != 1 or b.chunkCount() != 1) return false;
    return (a.byteLen() catch return false) == (b.byteLen() catch return false);
}

fn viewPair(step: *const PlacedStep) ?struct { src: TensorId, dst: TensorId } {
    return switch (step.op) {
        .ReshapeScalar => |s| .{ .src = s.src, .dst = s.dst },
        else => null,
    };
}

/// Remove eligible view copies and record destination-to-source backing aliases.
/// The alias map spans lists, though each match remains list-local.
pub fn elide(ed: *Editor, map: *AliasMap) Error!void {
    for (ed.steps, 0..) |*step, i| {
        const pair = viewPair(step) orelse continue;
        if (!eligible(ed, map, pair.src, pair.dst, i)) continue;
        // Follow a chain: if the source is itself an elided destination, borrow what it
        // borrows, keeping `aliasTensorBacking`'s one-level rule intact.
        const root = map.get(pair.src) orelse pair.src;
        map.put(pair.dst, root) catch return error.OutOfMemory;
        ed.kill(i);
    }
}

/// Every condition here is a hazard that would otherwise be silent.
fn eligible(ed: *const Editor, map: *const AliasMap, src: TensorId, dst: TensorId, step_index: usize) bool {
    if (src == dst) return false;
    // Only compiler workspace may be re-pointed: inputs, parameters and model state have
    // backing this pass does not own. An output must hold its own bytes past the run.
    if (!ed.isOwned(src) or !ed.isOwned(dst)) return false;
    if (ed.isOutput(src) or ed.isOutput(dst)) return false;
    // A destination already borrowing, or a source already re-pointed as a destination,
    // would need more than one level of indirection.
    if (map.contains(dst)) return false;
    if (!layoutsIdentical(ed.mgr, src, dst)) return false;

    // Both must live only in the list being edited: a tensor another list can write or
    // read has no index comparable to `step_index` here.
    if (!ed.confined(src) or !ed.confined(dst)) return false;
    const su = ed.use(src) orelse return false;
    const du = ed.use(dst) orelse return false;

    // The destination must be written by exactly this step and never read before it:
    // sharing bytes is only sound if the copy was the destination's whole definition.
    if (du.writes != 1 or du.last_write != step_index) return false;
    if (du.first_touch < step_index) return false;

    // The source must not change after the copy, or the destination would observe the
    // new value instead of the copied one.
    if (su.writes > 0 and su.last_write > step_index) return false;

    // Keep the copy if a later dispatch binds both ids: sharing one backing across two
    // operands conflicts with workspace and wgpu binding rules.
    var j: usize = step_index + 1;
    while (j < ed.steps.len) : (j += 1) {
        const walk = executable.tensorUses(&ed.steps[j].op);
        var saw_src = false;
        var saw_dst = false;
        for (walk.slice()) |it| {
            if (it.id.* == src) saw_src = true;
            if (it.id.* == dst) saw_dst = true;
        }
        if (saw_src and saw_dst) return false;
    }

    return true;
}
