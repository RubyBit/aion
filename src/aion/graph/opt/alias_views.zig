// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Elide reshape/retile copies between byte-identical layouts while preserving distinct
//! destination metadata and aliasing its backing during workspace planning.

const std = @import("std");

const executable = @import("../../runtime/executable.zig");
const manager_mod = @import("../../storage/manager.zig");
const editor_mod = @import("editor.zig");

const Editor = editor_mod.Editor;
const PlacedStep = executable.PlacedStep;
const StorageManager = manager_mod.StorageManager;
const TiledTensor = manager_mod.TiledTensor;
const TensorId = manager_mod.TensorId;

pub const Error = editor_mod.Error;

/// destination -> source. The destination borrows the source's backing.
pub const AliasMap = std.AutoHashMap(TensorId, TensorId);

/// Whether tensors have matching tile lengths/offsets and flat row-major byte order.
/// Offsets alone are insufficient because tile-major layouts can reorder values.
pub fn layoutsIdentical(mgr: *const StorageManager, a_id: TensorId, b_id: TensorId) bool {
    const a = mgr.getConst(a_id) catch return false;
    const b = mgr.getConst(b_id) catch return false;
    if (a.dtype != b.dtype) return false;
    if (a.tile_lens.len != b.tile_lens.len) return false;
    if (a.tile_offsets.len != b.tile_offsets.len) return false;
    for (a.tile_lens, b.tile_lens) |x, y| if (x != y) return false;
    for (a.tile_offsets, b.tile_offsets) |x, y| if (x != y) return false;
    return rowMajorChunked(a) and rowMajorChunked(b);
}

/// Whether tile-coordinate order preserves flat row-major order.
/// Every axis before the last split axis must contribute one element per tile.
fn rowMajorChunked(t: *const TiledTensor) bool {
    var split: usize = t.rank;
    while (split > 0) {
        if (t.tile_shape[split - 1] < t.shape[split - 1]) break;
        split -= 1;
    }
    if (split == 0) return true; // one tile spans every axis
    for (t.tile_shape[0 .. split - 1]) |extent| {
        if (extent != 1) return false;
    }
    return true;
}

fn viewPair(step: *const PlacedStep) ?struct { src: TensorId, dst: TensorId } {
    return switch (step.op) {
        .ReshapeScalar => |s| .{ .src = s.src, .dst = s.dst },
        .ReTileCopyScalar => |s| .{ .src = s.src, .dst = s.dst },
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
