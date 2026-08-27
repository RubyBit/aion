// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Elide view steps that copy between byte-identical layouts.
//!
//! A `Reshape`/`ReTile` exists because two graph values are two tensors. When the two
//! tensors occupy the same bytes in the same arrangement — which a reshape between
//! `[1,1,8,256]` and `[1,1,2048]` does, and a retile does whenever the tile extents
//! merely clamp to the same shape — the step copies a buffer onto an identical one.
//!
//! Measured on a Gemma-4 E2B step: 136 of 172 view steps were such copies, and a step on
//! the serial residual path costs ~10-20 us of critical path no matter how little it
//! does (a 6 KB scalar multiply measured the same as a 6.4 MiB GEMV).
//!
//! The fix is NOT to rewrite consumers to read the source id: consumers take logical
//! shape from tensor metadata, and the two disagree there by construction. The
//! destination keeps its metadata and BORROWS the source's bytes, so this pass only
//! deletes steps and records pairs; `workspace.plan` aliases once slots are known.

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

/// True when two tensors occupy the same bytes in the same arrangement.
///
/// Per-tile lengths AND offsets must match one-for-one: on the device each tile is its
/// own buffer, so equal tiling means a tile-for-tile duplicate, and on the host the
/// offsets are what make the single allocation identical.
///
/// Equal offsets are NOT enough on their own, because tiles are stored tile-major: both
/// tilings must also lay their bytes out in flat row-major order (see `rowMajorChunked`),
/// or the two tensors agree on every tile boundary while ordering values differently.
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

/// True when a tensor's bytes run in flat row-major order.
///
/// Each tile is a contiguous run holding its own elements row-major, and tiles follow
/// tile-coordinate order — so a tiling preserves the flat order only if it CHUNKS it:
/// every axis before the last split one must contribute a single element per tile. On
/// `[4,4]`, `{2,4}` chunks the flat order and `{4,2}` interleaves it, while both produce
/// two 32-byte tiles at the same offsets.
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

/// Kill every eligible no-op view step in the edited list, recording the
/// destination->source pairs into `map`. The caller commits the editor and owns the map,
/// which spans every list: the pairs are about backings, not step order, so a chain may
/// cross a control-flow boundary even though a match may not.
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

    // No later step may bind BOTH as operands: sharing a backing between two operands of
    // one dispatch is what `workspace.validateAliases` refuses (and what wgpu reports as
    // conflicting buffer usages), so `add(x, reshape(x))` must keep its copy. Only steps
    // after the copy can see the destination.
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
