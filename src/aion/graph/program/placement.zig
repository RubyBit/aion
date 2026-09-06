//! Placement pass for lowered executable programs.
//!
//! Assigns one placement per tensor, inserts explicit device-to-host transfers
//! for control-flow reads, and records sequence-cache growth sites. Physical
//! allocation is deliberately left to workspace/materialization.

const std = @import("std");
const executable = @import("../../runtime/executable.zig");
const manager_mod = @import("../../storage/manager.zig");
const plan_mod = @import("../plan.zig");

const StorageManager = manager_mod.StorageManager;
const TensorId = manager_mod.TensorId;
const DeviceRef = manager_mod.DeviceRef;
const Program = executable.ExecutableProgram;
const PlacedStep = executable.PlacedStep;

pub const Error = error{ InvalidArgument, OutOfMemory };

/// Assign placements and make every host read explicit.
///
/// Every non-transfer step executes at the compile target. A control operand
/// interpreted by the CPU receives a host mirror and an explicit Transfer.
pub fn place(
    allocator: std.mem.Allocator,
    mgr: *StorageManager,
    prog: *Program,
    owned: *std.ArrayList(TensorId),
    policy: plan_mod.TilePolicy,
) Error!void {
    const target: executable.Placement = .{ .kind = policy.target_kind };
    prog.target = target;
    for (prog.steps) |*placed| placed.placement = target;
    for (prog.blocks) |block| {
        for (block.steps) |*placed| placed.placement = target;
    }

    var mirrors: std.AutoHashMap(TensorId, TensorId) = .init(allocator);
    defer mirrors.deinit();
    var refreshes: std.ArrayList(LoopCondRefresh) = .empty;
    defer refreshes.deinit(allocator);

    // Collected before hoisting so it covers the program as the compiler produced it.
    var written: std.AutoHashMap(TensorId, void) = .init(allocator);
    defer written.deinit();
    try collectWritten(&written, prog);
    // Host operands read in place: the placement table has to say so, or the
    // verifier sees a host read of a device tensor.
    var host_inputs: std.AutoHashMap(TensorId, void) = .init(allocator);
    defer host_inputs.deinit();

    var ctx: HoistCtx = .{
        .allocator = allocator,
        .mgr = mgr,
        .owned = owned,
        .mirrors = &mirrors,
        .refreshes = &refreshes,
        .target = target,
        .written = &written,
        .host_inputs = &host_inputs,
    };
    for (prog.blocks) |*block| block.steps = try ctx.hoist(block.steps);
    prog.steps = try ctx.hoist(prog.steps);

    for (refreshes.items) |request| {
        const idx: usize = @intCast(request.block);
        if (idx >= prog.blocks.len) return error.InvalidArgument;
        var list = std.ArrayList(PlacedStep).fromOwnedSlice(prog.blocks[idx].steps);
        list.append(allocator, .{ .op = .{ .Transfer = request.transfer } }) catch return error.OutOfMemory;
        prog.blocks[idx].steps = list.toOwnedSlice(allocator) catch return error.OutOfMemory;
    }

    try collectPlacements(allocator, prog, &mirrors, &host_inputs, target);
}

/// Every tensor some step writes, across the top level and every block.
fn collectWritten(out: *std.AutoHashMap(TensorId, void), prog: *Program) Error!void {
    for (prog.steps) |*placed| try noteWritten(out, placed);
    for (prog.blocks) |block| {
        for (block.steps) |*placed| try noteWritten(out, placed);
    }
}

fn noteWritten(out: *std.AutoHashMap(TensorId, void), placed: *PlacedStep) Error!void {
    const uses = executable.tensorUses(&placed.op);
    for (uses.slice()) |use| {
        if (use.access == .read) continue;
        out.put(use.id.*, {}) catch return error.OutOfMemory;
    }
}

const LoopCondRefresh = struct {
    block: executable.BlockId,
    transfer: executable.StepTransfer,
};

const HoistCtx = struct {
    allocator: std.mem.Allocator,
    mgr: *StorageManager,
    owned: *std.ArrayList(TensorId),
    mirrors: *std.AutoHashMap(TensorId, TensorId),
    refreshes: *std.ArrayList(LoopCondRefresh),
    target: executable.Placement,
    /// Tensors some step writes. A host operand whose source is NOT in here is a
    /// pure input, so where it already lives is where it will stay.
    written: *const std.AutoHashMap(TensorId, void),
    /// Sources whose host-operand transfer was elided because they already live
    /// on the host; `collectPlacements` records them as host-placed.
    host_inputs: *std.AutoHashMap(TensorId, void),

    fn hoist(self: *HoistCtx, old_steps: []PlacedStep) Error![]PlacedStep {
        var out: std.ArrayList(PlacedStep) = .empty;
        errdefer out.deinit(self.allocator);

        for (old_steps) |original| {
            var placed = original;
            const mask = executable.controlHostOperands(&placed.op);
            if (mask != 0) {
                const uses = executable.tensorUses(&placed.op);
                const loop_cond: ?*const TensorId = if (placed.op == .Loop and placed.op.Loop.cond != null)
                    &placed.op.Loop.cond.?
                else
                    null;
                for (uses.slice(), 0..) |use, i| {
                    if (mask & (@as(u64, 1) << @intCast(i)) == 0) continue;
                    const source = use.id.*;
                    if (self.target.kind == .cpu) continue;
                    // A pure input that already lives on the host needs no copy: the
                    // operand reads it in place. That is what keeps a caller-supplied
                    // `If` predicate from costing an upload plus the pipeline flush and
                    // readback a mirror would need. Anything a step WRITES is excluded:
                    // workspace tensors are not materialized on the device yet at this
                    // point, so residency alone would wrongly match every one of them.
                    if (!self.written.contains(source)) {
                        const src_dev = self.mgr.tensorDevice(source) catch DeviceRef{};
                        if (src_dev.kind == .cpu) {
                            self.host_inputs.put(source, {}) catch return error.OutOfMemory;
                            continue;
                        }
                    }
                    const mirror = try self.transferToHost(&out, source);
                    use.id.* = mirror;
                    if (loop_cond) |field| if (use.id == field) {
                        self.refreshes.append(self.allocator, .{
                            .block = placed.op.Loop.body_block,
                            .transfer = .{ .dst = mirror, .src = source, .source = self.target, .destination = .{} },
                        }) catch return error.OutOfMemory;
                    };
                }
                placed.host_operands = mask;
            }
            out.append(self.allocator, placed) catch return error.OutOfMemory;
        }
        const new_steps = out.toOwnedSlice(self.allocator) catch return error.OutOfMemory;
        self.allocator.free(old_steps);
        return new_steps;
    }

    fn transferToHost(self: *HoistCtx, out: *std.ArrayList(PlacedStep), source: TensorId) Error!TensorId {
        const dst = self.mirrors.get(source) orelse blk: {
            const src = self.mgr.getConst(source) catch return error.InvalidArgument;
            const mirror = self.mgr.createTiledTensor(src.dtype, src.shape, src.tile_shape, .{
                .tile_alignment = src.tile_alignment,
                .quant_axis = src.quant_axis,
            }) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.InvalidArgument,
            };
            self.owned.append(self.allocator, mirror) catch {
                self.mgr.releaseTensorData(mirror) catch {};
                return error.OutOfMemory;
            };
            self.mirrors.put(source, mirror) catch return error.OutOfMemory;
            break :blk mirror;
        };
        out.append(self.allocator, .{
            .op = .{ .Transfer = .{ .dst = dst, .src = source, .source = self.target, .destination = .{} } },
        }) catch return error.OutOfMemory;
        return dst;
    }
};

fn collectPlacements(
    allocator: std.mem.Allocator,
    prog: *Program,
    mirrors: *const std.AutoHashMap(TensorId, TensorId),
    host_inputs: *const std.AutoHashMap(TensorId, void),
    target: executable.Placement,
) Error!void {
    var host_side: std.AutoHashMap(TensorId, void) = .init(allocator);
    defer host_side.deinit();
    var it_mirror = mirrors.valueIterator();
    while (it_mirror.next()) |mirror| host_side.put(mirror.*, {}) catch return error.OutOfMemory;
    var it_host = host_inputs.keyIterator();
    while (it_host.next()) |id| host_side.put(id.*, {}) catch return error.OutOfMemory;
    var seen: std.AutoHashMap(TensorId, void) = .init(allocator);
    defer seen.deinit();

    for (prog.steps) |*placed| try noteTensors(&seen, placed);
    for (prog.blocks) |block| {
        for (block.steps) |*placed| try noteTensors(&seen, placed);
    }

    const list = allocator.alloc(executable.TensorPlacement, seen.count()) catch return error.OutOfMemory;
    var i: usize = 0;
    var it = seen.keyIterator();
    while (it.next()) |id| : (i += 1) {
        list[i] = .{ .id = id.*, .placement = if (host_side.contains(id.*)) .{} else target };
    }
    std.mem.sort(executable.TensorPlacement, list, {}, struct {
        fn lt(_: void, a: executable.TensorPlacement, b: executable.TensorPlacement) bool {
            return a.id < b.id;
        }
    }.lt);
    prog.tensor_placements = list;
}

/// Drop placement entries for tensors no longer named by any step.
///
/// `tensor_placements` is a snapshot of the schedule taken when placement ran, and a
/// later pass that deletes steps can orphan an entry. That is not cosmetic:
/// `workspace.materializePlacements` requires every listed tensor to still have backing,
/// while `workspace.plan` releases anything with no remaining use, so an orphan turns
/// into `InvalidArgument` at materialize time. Any pass that removes steps between
/// `place` and `plan` must call this.
///
/// Filtering preserves the id ordering the binary search in `Program.placementOf`
/// depends on, and each surviving entry keeps the placement it was given.
pub fn pruneOrphaned(allocator: std.mem.Allocator, prog: *Program) Error!void {
    if (prog.tensor_placements.len == 0) return;

    var live: std.AutoHashMap(TensorId, void) = .init(allocator);
    defer live.deinit();
    for (prog.steps) |*placed| try noteTensors(&live, placed);
    for (prog.blocks) |block| {
        for (block.steps) |*placed| try noteTensors(&live, placed);
    }

    var keep: usize = 0;
    for (prog.tensor_placements) |entry| {
        if (live.contains(entry.id)) keep += 1;
    }
    if (keep == prog.tensor_placements.len) return;

    const list = allocator.alloc(executable.TensorPlacement, keep) catch return error.OutOfMemory;
    var w: usize = 0;
    for (prog.tensor_placements) |entry| {
        if (!live.contains(entry.id)) continue;
        list[w] = entry;
        w += 1;
    }
    allocator.free(prog.tensor_placements);
    prog.tensor_placements = list;
}

fn noteTensors(seen: *std.AutoHashMap(TensorId, void), placed: *PlacedStep) Error!void {
    const uses = executable.tensorUses(&placed.op);
    for (uses.slice()) |use| seen.put(use.id.*, {}) catch return error.OutOfMemory;
}
