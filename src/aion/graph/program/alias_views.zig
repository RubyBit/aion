// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Elide view steps that copy between byte-identical layouts.
//!
//! A `Reshape`/`ReTile` exists because two graph values are two tensors. When the
//! two tensors happen to occupy the same bytes in the same arrangement — which a
//! reshape between `[1,1,8,256]` and `[1,1,2048]` does, and which a retile does
//! whenever the tile extents merely clamp to the same shape — the step copies a
//! buffer onto an identical buffer and computes nothing.
//!
//! That is not a rounding error at decode. Measured on a Gemma-4 E2B step: **136 of
//! 172 view steps** were such copies, and a step on the serial residual path costs
//! ~10-20 us of critical path no matter how little it does (a 6 KB scalar multiply
//! measured the same as a 6.4 MiB GEMV). Parallel dispatches are ~1 us; serial ones
//! are not, which is why removing *steps* on the chain is what buys time.
//!
//! The fix is NOT to rewrite consumers to read the source id: consumers take logical
//! shape from tensor metadata, and the two disagree there by construction. Instead the
//! destination keeps its own metadata and BORROWS the source's bytes, via
//! `StorageManager.aliasTensorBacking`. So this pass only deletes steps and records
//! pairs; `workspace.plan` performs the aliasing once slots are known.
//!
//! Eligibility is deliberately conservative — see `eligible`. The one previous attempt
//! at workspace aliasing in this repo was reverted because it reasoned at the *value*
//! level while the hazards live at the *step* level, so everything here is decided by
//! walking the emitted steps through `executable.tensorUses`.

const std = @import("std");
const executable = @import("../../runtime/executable.zig");
const manager_mod = @import("../../storage/manager.zig");
const step_uses = @import("step_uses.zig");

const StorageManager = manager_mod.StorageManager;
const TensorId = manager_mod.TensorId;
const Program = executable.ExecutableProgram;
const PlacedStep = executable.PlacedStep;

pub const Error = error{ InvalidArgument, OutOfMemory };

/// destination -> source. The destination borrows the source's backing.
pub const AliasMap = std.AutoHashMap(TensorId, TensorId);

/// True when two tensors occupy the same bytes in the same arrangement.
///
/// Per-tile lengths AND offsets must match one-for-one: on the device each tile is its
/// own buffer, so equal tiling means a tile-for-tile duplicate, and on the host the
/// offsets are what make the single allocation identical.
pub fn layoutsIdentical(mgr: *const StorageManager, a_id: TensorId, b_id: TensorId) bool {
    const a = mgr.getConst(a_id) catch return false;
    const b = mgr.getConst(b_id) catch return false;
    if (a.dtype != b.dtype) return false;
    if (a.tile_lens.len != b.tile_lens.len) return false;
    if (a.tile_offsets.len != b.tile_offsets.len) return false;
    for (a.tile_lens, b.tile_lens) |x, y| if (x != y) return false;
    for (a.tile_offsets, b.tile_offsets) |x, y| if (x != y) return false;
    return true;
}

fn viewPair(step: *const PlacedStep) ?struct { src: TensorId, dst: TensorId } {
    return switch (step.op) {
        .ReshapeScalar => |s| .{ .src = s.src, .dst = s.dst },
        .ReTileCopyScalar => |s| .{ .src = s.src, .dst = s.dst },
        else => null,
    };
}

/// Delete every eligible no-op view step and return the destination->source pairs.
/// `prog.steps` is reallocated when anything is removed. The caller owns the map.
pub fn elideNoopViews(
    allocator: std.mem.Allocator,
    mgr: *StorageManager,
    prog: *Program,
    owned: []const TensorId,
) Error!AliasMap {
    var map: AliasMap = .init(allocator);
    errdefer map.deinit();
    if (prog.steps.len == 0 or owned.len == 0) return map;
    var uses = try step_uses.collect(allocator, prog);
    defer uses.deinit();
    var is_owned = try step_uses.ownedSet(allocator, owned);
    defer is_owned.deinit();

    var drop = allocator.alloc(bool, prog.steps.len) catch return error.OutOfMemory;
    defer allocator.free(drop);
    @memset(drop, false);

    var dropped: usize = 0;
    for (prog.steps, 0..) |*step, i| {
        const pair = viewPair(step) orelse continue;
        if (!eligible(mgr, prog, &uses, &is_owned, &map, pair.src, pair.dst, i)) continue;
        // Follow a chain: if the source is itself an elided destination, borrow from
        // whatever it borrows, keeping `aliasTensorBacking`'s one-level rule intact.
        const root = map.get(pair.src) orelse pair.src;
        map.put(pair.dst, root) catch return error.OutOfMemory;
        drop[i] = true;
        dropped += 1;
    }
    if (dropped == 0) return map;

    const kept = allocator.alloc(PlacedStep, prog.steps.len - dropped) catch return error.OutOfMemory;
    var w: usize = 0;
    for (prog.steps, 0..) |step, i| {
        if (drop[i]) continue;
        kept[w] = step;
        w += 1;
    }
    allocator.free(prog.steps);
    prog.steps = kept;
    return map;
}

/// Every condition here is a hazard that would otherwise be silent.
fn eligible(
    mgr: *StorageManager,
    prog: *const Program,
    uses: *const step_uses.Map,
    is_owned: *const step_uses.OwnedSet,
    map: *const AliasMap,
    src: TensorId,
    dst: TensorId,
    step_index: usize,
) bool {
    if (src == dst) return false;
    // Only compiler workspace may be re-pointed: inputs, parameters and model state
    // have backing this pass does not own.
    if (!is_owned.contains(src) or !is_owned.contains(dst)) return false;
    // An output must hold its own bytes past the run.
    for (prog.outputs) |id| if (id == src or id == dst) return false;
    // A destination that is already borrowing, or a source already re-pointed as a
    // destination, would need more than one level of indirection.
    if (map.contains(dst)) return false;
    if (!layoutsIdentical(mgr, src, dst)) return false;

    const su = uses.get(src) orelse return false;
    const du = uses.get(dst) orelse return false;
    if (su.in_block or du.in_block) return false;

    // The destination must be written by exactly this step and never read before it:
    // sharing bytes is only sound if the copy was the destination's whole definition.
    if (du.writes != 1 or du.last_write != step_index) return false;
    if (du.first_touch < step_index) return false;

    // The source must not change after the copy, or the destination would silently
    // observe the new value instead of the copied one.
    if (su.writes > 0 and su.last_write > step_index) return false;

    // No later step may bind BOTH as operands. Sharing a backing between two
    // operands of one dispatch is what `workspace.validateAliases` refuses (and what
    // wgpu reports as conflicting buffer usages), so a graph like `add(x,
    // reshape(x))` must keep its copy. Checking here turns "compile fails" into
    // "this one view is not elided". Only steps after the copy can see the
    // destination, so the scan starts there.
    var j: usize = step_index + 1;
    while (j < prog.steps.len) : (j += 1) {
        const walk = executable.tensorUses(&@constCast(&prog.steps[j]).op);
        var saw_src = false;
        var saw_dst = false;
        for (walk.slice()) |use| {
            if (use.id.* == src) saw_src = true;
            if (use.id.* == dst) saw_dst = true;
        }
        if (saw_src and saw_dst) return false;
    }

    return true;
}
