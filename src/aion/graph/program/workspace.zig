//! Physical workspace planning for a lowered executable program.
//!
//! The graph compiler owns logical tensors and emits steps. This module begins
//! after lowering: it derives tensor lifetimes from those steps, assigns
//! non-overlapping logical tensors to capacity-sized physical slots, validates
//! the alias plan, and materializes the plan at the chosen placement.

const std = @import("std");
const env = @import("../../env.zig");
const executable = @import("../../runtime/executable.zig");
const device_memory = @import("../../runtime/device_memory.zig");
const manager_mod = @import("../../storage/manager.zig");
const alias_views = @import("../opt/alias_views.zig");

const StorageManager = manager_mod.StorageManager;
const TensorId = manager_mod.TensorId;
const Program = executable.ExecutableProgram;
const PlacedStep = executable.PlacedStep;

pub const PlanError = error{ InvalidArgument, OutOfMemory };

const Interval = struct {
    id: TensorId,
    first: usize = std.math.maxInt(usize),
    last: usize = 0,
    seen: bool = false,
    reusable: bool = true,
    preserve_contents: bool = false,
    /// Null is the enclosing program; a block id names one replayable Loop body.
    /// An interval observed in two domains becomes non-reusable.
    domain: ?executable.BlockId = null,
    domain_set: bool = false,
};

const PendingSlot = struct {
    owner: TensorId,
    members: std.ArrayList(TensorId) = .empty,
    capacities: std.ArrayList(usize) = .empty,
    host_bytes: usize,
    last_end: usize,
    placement: executable.Placement,
    reusable: bool,
    preserve_contents: bool,
    domain: ?executable.BlockId,
};

/// Assign compiler-owned logical tensors to physical slots after lowering and
/// placement. Reuse is capacity-based inside one execution domain. Every Loop
/// body is a separate domain because its static step order is replayed at run
/// time and therefore cannot be flattened into the enclosing interval line.
/// The executable remains expressed in logical tensor IDs; StorageManager
/// resolves each ID to the canonical backing chosen here.
pub fn plan(
    allocator: std.mem.Allocator,
    mgr: *StorageManager,
    prog: *Program,
    owned: []const TensorId,
    /// destination -> source pairs from `alias_views.elide`. Each
    /// destination borrows the source's backing instead of getting a slot, because
    /// the copy that would have filled it was removed as a no-op.
    alias: *const alias_views.AliasMap,
) PlanError!void {
    if (owned.len == 0) return;

    var by_id: std.AutoHashMap(TensorId, usize) = .init(allocator);
    defer by_id.deinit();
    const intervals = allocator.alloc(Interval, owned.len) catch return error.OutOfMemory;
    defer allocator.free(intervals);
    for (owned, 0..) |id, i| {
        const preserve_contents = mgr.tensorHasBacking(id) catch return error.InvalidArgument;
        intervals[i] = .{
            .id = id,
            .reusable = !preserve_contents,
            .preserve_contents = preserve_contents,
        };
        mgr.markWorkspaceTensor(id) catch return error.InvalidArgument;
        by_id.put(id, i) catch return error.OutOfMemory;
    }

    const Liveness = struct {
        prog: *Program,
        by_id: *const std.AutoHashMap(TensorId, usize),
        intervals: []Interval,
        visiting: []bool,
        tick: usize = 0,
        domain: ?executable.BlockId = null,

        fn exclude(self: *@This(), id: TensorId) void {
            if (self.by_id.get(id)) |i| self.intervals[i].reusable = false;
        }

        fn visitBlock(self: *@This(), block_id: executable.BlockId, loop_domain: bool) PlanError!void {
            const i: usize = @intCast(block_id);
            if (i >= self.prog.blocks.len or self.visiting[i]) return error.InvalidArgument;
            self.visiting[i] = true;
            defer self.visiting[i] = false;
            const outer_domain = self.domain;
            if (loop_domain) self.domain = block_id;
            defer self.domain = outer_domain;
            try self.visitSteps(self.prog.blocks[i].steps);
        }

        fn observeDomain(self: *@This(), interval: *Interval) void {
            if (!interval.domain_set) {
                interval.domain = self.domain;
                interval.domain_set = true;
            } else if (interval.domain != self.domain) {
                // A value crossing a loop-region boundary is persistent state or
                // an outer capture, not reusable scratch within either domain.
                interval.reusable = false;
            }
        }

        fn visitSteps(self: *@This(), steps: []PlacedStep) PlanError!void {
            for (steps) |*placed_const| {
                const placed = @constCast(placed_const);
                switch (placed.op) {
                    .If => |s| {
                        try self.visitBlock(s.then_block, false);
                        try self.visitBlock(s.else_block, false);
                        const n: usize = @intCast(s.output_count);
                        for (s.outputs[0..n]) |id| self.exclude(id);
                        for (s.then_outputs[0..n]) |id| self.exclude(id);
                        for (s.else_outputs[0..n]) |id| self.exclude(id);
                    },
                    .Loop => |s| {
                        try self.visitBlock(s.body_block, true);
                        const n: usize = @intCast(s.carried_count);
                        for (s.carried[0..n]) |id| self.exclude(id);
                        for (s.body_carried_outputs[0..n]) |id| self.exclude(id);
                    },
                    else => {},
                }

                const uses = executable.tensorUses(&placed.op);
                for (uses.slice()) |use| {
                    const i = self.by_id.get(use.id.*) orelse continue;
                    const interval = &self.intervals[i];
                    self.observeDomain(interval);
                    if (!interval.seen) {
                        interval.first = self.tick;
                        interval.seen = true;
                    }
                    // Reuse is safe only when the new lifetime begins with a
                    // complete write. First-read values need their own backing.
                    if (interval.first == self.tick and use.access != .write) interval.reusable = false;
                    interval.last = self.tick;
                }
                self.tick += 1;
            }
        }
    };

    const visiting = allocator.alloc(bool, prog.blocks.len) catch return error.OutOfMemory;
    defer allocator.free(visiting);
    @memset(visiting, false);
    var liveness: Liveness = .{ .prog = prog, .by_id = &by_id, .intervals = intervals, .visiting = visiting };
    try liveness.visitSteps(prog.steps);

    // Outputs outlive execution and therefore always own dedicated slots.
    for (prog.outputs) |id| {
        if (by_id.get(id)) |i| {
            intervals[i].reusable = false;
            if (!intervals[i].seen) {
                intervals[i].seen = true;
                intervals[i].first = 0;
                intervals[i].last = liveness.tick;
            }
        }
    }

    // An aliased destination reads bytes the source produced, and it reads them AFTER
    // the source's own last use. So the source's slot must stay alive that long and
    // must never be handed to another tensor -- otherwise a later reuse would
    // overwrite the bytes the destination is about to read. Extend and pin it here,
    // before slots are assigned.
    {
        var it = alias.iterator();
        while (it.next()) |entry| {
            const dst_i = by_id.get(entry.key_ptr.*) orelse continue;
            const src_i = by_id.get(entry.value_ptr.*) orelse continue;
            intervals[src_i].reusable = false;
            if (intervals[dst_i].seen) {
                intervals[src_i].last = @max(intervals[src_i].last, intervals[dst_i].last);
                if (!intervals[src_i].seen) {
                    intervals[src_i].seen = true;
                    intervals[src_i].first = intervals[dst_i].first;
                }
            }
            // The destination owns nothing; skip it in slot assignment below.
            intervals[dst_i].seen = false;
        }
    }

    std.mem.sort(Interval, intervals, {}, struct {
        fn lessThan(_: void, a: Interval, b: Interval) bool {
            if (a.first != b.first) return a.first < b.first;
            return a.id < b.id;
        }
    }.lessThan);

    var slots: std.ArrayList(PendingSlot) = .empty;
    defer {
        for (slots.items) |*slot| {
            slot.members.deinit(allocator);
            slot.capacities.deinit(allocator);
        }
        slots.deinit(allocator);
    }

    for (intervals) |interval| {
        if (!interval.seen) {
            mgr.releaseTensorData(interval.id) catch {};
            continue;
        }
        const tensor = mgr.getConst(interval.id) catch return error.InvalidArgument;
        const placement = prog.placementOf(interval.id) orelse prog.target;
        const host_bytes = mgr.tensorLogicalBackingBytes(interval.id) catch return error.InvalidArgument;

        var reusable_slot: ?usize = null;
        var best_growth: usize = std.math.maxInt(usize);
        if (interval.reusable) {
            for (slots.items, 0..) |slot, i| {
                if (!slot.reusable or slot.last_end >= interval.first or !slot.placement.eql(placement)) continue;
                if (slot.domain != interval.domain) continue;

                var growth: usize = if (placement.kind == .cpu and host_bytes > slot.host_bytes) host_bytes - slot.host_bytes else 0;
                if (placement.kind != .cpu) {
                    for (tensor.tile_lens, 0..) |len, tile| {
                        const old = if (tile < slot.capacities.items.len) slot.capacities.items[tile] else 0;
                        if (len > old) growth = std.math.add(usize, growth, len - old) catch return error.InvalidArgument;
                    }
                }
                if (growth < best_growth) {
                    reusable_slot = i;
                    best_growth = growth;
                }
            }
        }

        if (reusable_slot) |slot_index| {
            const slot = &slots.items[slot_index];
            slot.members.append(allocator, interval.id) catch return error.OutOfMemory;
            if (slot.capacities.items.len < tensor.tile_lens.len) {
                const old_len = slot.capacities.items.len;
                slot.capacities.resize(allocator, tensor.tile_lens.len) catch return error.OutOfMemory;
                @memset(slot.capacities.items[old_len..], 0);
            }
            for (tensor.tile_lens, 0..) |len, tile| slot.capacities.items[tile] = @max(slot.capacities.items[tile], len);
            slot.host_bytes = @max(slot.host_bytes, host_bytes);
            slot.last_end = interval.last;
        } else {
            var slot: PendingSlot = .{
                .owner = interval.id,
                .host_bytes = host_bytes,
                .last_end = interval.last,
                .placement = placement,
                .reusable = interval.reusable,
                .preserve_contents = interval.preserve_contents,
                .domain = interval.domain,
            };
            slot.members.append(allocator, interval.id) catch return error.OutOfMemory;
            slot.capacities.appendSlice(allocator, tensor.tile_lens) catch return error.OutOfMemory;
            slots.append(allocator, slot) catch return error.OutOfMemory;
        }
    }

    // Join each aliased destination to the slot that holds its source, as an ordinary
    // extra member. That reuses the existing member machinery — the loop below aliases
    // every non-owner member to the owner and `validateAliases` sees it — instead of
    // bolting on a second aliasing path.
    {
        var it = alias.iterator();
        while (it.next()) |entry| {
            const dst_id = entry.key_ptr.*;
            const src_id = entry.value_ptr.*;
            const slot_index = blk: {
                for (slots.items, 0..) |slot, i| {
                    for (slot.members.items) |member| if (member == src_id) break :blk i;
                }
                // The source never got a slot (not compiler workspace, or unused).
                // Leaving the destination unaliased would leave it with no bytes, and
                // its copy is already gone, so this must not happen.
                return error.InvalidArgument;
            };
            const slot = &slots.items[slot_index];
            const tensor = mgr.getConst(dst_id) catch return error.InvalidArgument;
            slot.members.append(allocator, dst_id) catch return error.OutOfMemory;
            if (slot.capacities.items.len < tensor.tile_lens.len) {
                const old_len = slot.capacities.items.len;
                slot.capacities.resize(allocator, tensor.tile_lens.len) catch return error.OutOfMemory;
                @memset(slot.capacities.items[old_len..], 0);
            }
            for (tensor.tile_lens, 0..) |len, tile| slot.capacities.items[tile] = @max(slot.capacities.items[tile], len);
        }
    }

    const final = allocator.alloc(executable.WorkspaceSlot, slots.items.len) catch return error.OutOfMemory;
    var built: usize = 0;
    errdefer {
        for (final[0..built]) |slot| {
            allocator.free(slot.members);
            allocator.free(slot.tile_capacities);
        }
        allocator.free(final);
    }
    var workspace_bytes: usize = 0;
    for (slots.items, 0..) |*slot, i| {
        for (slot.members.items[1..]) |id| mgr.aliasTensorBacking(id, slot.owner) catch return error.InvalidArgument;
        // Keep one slot-sized host allocation even for a GPU program. It allows
        // CPU reference execution before placement without restoring per-value
        // allocations; GPU materialization releases it.
        mgr.reserveHostBacking(slot.owner, slot.host_bytes) catch |e| return switch (e) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.InvalidArgument,
        };
        const physical_bytes = if (slot.placement.kind == .cpu) slot.host_bytes else blk: {
            var total: usize = 0;
            for (slot.capacities.items) |cap| total = std.math.add(usize, total, cap) catch return error.InvalidArgument;
            break :blk total;
        };
        workspace_bytes = std.math.add(usize, workspace_bytes, physical_bytes) catch return error.InvalidArgument;

        const members = allocator.dupe(TensorId, slot.members.items) catch return error.OutOfMemory;
        errdefer allocator.free(members);
        const tile_capacities = allocator.dupe(usize, slot.capacities.items) catch return error.OutOfMemory;
        errdefer allocator.free(tile_capacities);
        final[i] = .{
            .owner = slot.owner,
            .members = members,
            .tile_capacities = tile_capacities,
            .host_bytes = slot.host_bytes,
            .placement = slot.placement,
            .preserve_contents = slot.preserve_contents,
        };
        built += 1;
    }
    prog.workspace_slots = final;
    prog.workspace_bytes = workspace_bytes;

    if (env.flagEnabled("AION_TRACE")) {
        std.debug.print(
            "[aion][workspace] logical={d} slots={d} bytes={d}\n",
            .{ owned.len, final.len, workspace_bytes },
        );
    }
    try validateAliases(mgr, prog);
}

/// Materialize a compiler-planned program at its declared placement. Workspace
/// slots are created first; all remaining placed tensors are migrated normally.
pub fn materializePlacements(
    mgr: *StorageManager,
    prog: *const Program,
    target: manager_mod.DeviceRef,
    dev: device_memory.DeviceMemory,
) manager_mod.StorageError!void {
    for (prog.workspace_slots) |slot| {
        const slot_target: manager_mod.DeviceRef = if (slot.placement.kind == .cpu) .{} else target;
        try mgr.materializeWorkspaceSlot(
            slot.owner,
            slot_target,
            if (slot_target.kind == .cpu) null else dev,
            slot.tile_capacities,
            slot.host_bytes,
            slot.preserve_contents,
        );
    }
    if (target.kind == .cpu) return;

    for (prog.tensor_placements) |entry| {
        if (entry.placement.kind == .cpu) continue;
        const tensor = try mgr.getConst(entry.id);
        if ((try mgr.tensorDevice(entry.id)).eql(target)) continue;
        if (tensor.backing_owner != null) continue;
        if (!try mgr.tensorHasBacking(entry.id)) return error.InvalidArgument;

        var tile_shape: [8]usize = undefined;
        const rank: usize = @intCast(tensor.rank);
        @memcpy(tile_shape[0..rank], tensor.tile_shape);
        try mgr.moveTensor(entry.id, target, dev, tile_shape[0..rank], tensor.tile_alignment);
    }
}

fn validateAliases(mgr: *const StorageManager, prog: *Program) PlanError!void {
    const Validate = struct {
        fn steps(store: *const StorageManager, list: []PlacedStep) PlanError!void {
            for (list) |*placed_const| {
                const placed = @constCast(placed_const);
                const uses = executable.tensorUses(&placed.op);
                for (uses.slice(), 0..) |a, i| {
                    const a_backing = store.physicalBackingId(a.id.*) catch return error.InvalidArgument;
                    for (uses.slice()[0..i]) |b| {
                        if (a.id.* == b.id.*) continue;
                        const b_backing = store.physicalBackingId(b.id.*) catch return error.InvalidArgument;
                        if (a_backing == b_backing) return error.InvalidArgument;
                    }
                }
            }
        }
    };
    try Validate.steps(mgr, prog.steps);
    for (prog.blocks) |block| try Validate.steps(mgr, block.steps);
}

test "workspace planner grows capacity for disjoint lifetimes in one domain" {
    const allocator = std.testing.allocator;
    var mgr = StorageManager.init(allocator);
    defer mgr.deinit();

    const input = try mgr.createTiledTensor(.f32, &.{4}, &.{4}, .{});
    const a = try mgr.createTiledTensor(.f32, &.{4}, &.{4}, .{});
    const c = try mgr.createTiledTensor(.f32, &.{100}, &.{100}, .{});
    const b = try mgr.createTiledTensor(.f32, &.{4}, &.{4}, .{});
    const out = try mgr.createTiledTensor(.f32, &.{4}, &.{4}, .{});
    const owned = [_]TensorId{ a, c, b, out };
    for (owned) |id| try mgr.releaseTensorData(id);

    var steps = [_]PlacedStep{
        .{ .op = .{ .UnaryTiled = .{ .op = .relu, .out = a, .a = input } } },
        .{ .op = .{ .UnaryTiled = .{ .op = .relu, .out = c, .a = a } } },
        .{ .op = .{ .UnaryTiled = .{ .op = .relu, .out = b, .a = input } } },
        .{ .op = .{ .UnaryTiled = .{ .op = .relu, .out = out, .a = b } } },
    };
    var outputs = [_]TensorId{out};
    var placements = [_]executable.TensorPlacement{
        .{ .id = input, .placement = .{} },
        .{ .id = a, .placement = .{} },
        .{ .id = c, .placement = .{} },
        .{ .id = b, .placement = .{} },
        .{ .id = out, .placement = .{} },
    };
    var prog: Program = .{
        .allocator = allocator,
        .steps = &steps,
        .outputs = &outputs,
        .tensor_placements = &placements,
    };
    defer {
        for (prog.workspace_slots) |slot| {
            allocator.free(slot.members);
            allocator.free(slot.tile_capacities);
        }
        allocator.free(prog.workspace_slots);
    }

    var no_alias: alias_views.AliasMap = .init(allocator);
    defer no_alias.deinit();
    try plan(allocator, &mgr, &prog, &owned, &no_alias);
    try std.testing.expectEqual(@as(usize, 3), prog.workspace_slots.len);
    try std.testing.expect((try mgr.physicalBackingId(b)) != b);
    try std.testing.expect((try mgr.physicalBackingId(c)) != try mgr.physicalBackingId(b));
    try std.testing.expectEqual(out, try mgr.physicalBackingId(out));
}

test "workspace planner isolates loop-body reuse from the enclosing schedule" {
    const allocator = std.testing.allocator;
    var mgr = StorageManager.init(allocator);
    defer mgr.deinit();

    const input = try mgr.createTiledTensor(.f32, &.{4}, &.{4}, .{});
    const outer_tmp = try mgr.createTiledTensor(.f32, &.{4}, &.{4}, .{});
    const body_a = try mgr.createTiledTensor(.f32, &.{4}, &.{4}, .{});
    const body_b = try mgr.createTiledTensor(.f32, &.{100}, &.{100}, .{});
    const body_c = try mgr.createTiledTensor(.f32, &.{8}, &.{8}, .{});
    const owned = [_]TensorId{ outer_tmp, body_a, body_b, body_c };
    for (owned) |id| try mgr.releaseTensorData(id);

    var body_steps = [_]PlacedStep{
        .{ .op = .{ .UnaryTiled = .{ .op = .relu, .out = body_a, .a = input } } },
        .{ .op = .{ .UnaryTiled = .{ .op = .relu, .out = body_b, .a = body_a } } },
        .{ .op = .{ .UnaryTiled = .{ .op = .relu, .out = body_c, .a = input } } },
    };
    var blocks = [_]executable.Block{.{ .steps = &body_steps }};
    var steps = [_]PlacedStep{
        .{ .op = .{ .UnaryTiled = .{ .op = .relu, .out = outer_tmp, .a = input } } },
        .{ .op = .{ .Loop = .{
            .trip_count = null,
            .static_max_trip_count = 2,
            .cond = null,
            .check_before = false,
            .body_block = 0,
            .carried_count = 0,
            .carried = std.mem.zeroes([executable.MAX_LOOP_CARRIED]TensorId),
            .body_carried_outputs = std.mem.zeroes([executable.MAX_LOOP_CARRIED]TensorId),
        } } },
    };
    var placements = [_]executable.TensorPlacement{
        .{ .id = input, .placement = .{} },
        .{ .id = outer_tmp, .placement = .{} },
        .{ .id = body_a, .placement = .{} },
        .{ .id = body_b, .placement = .{} },
        .{ .id = body_c, .placement = .{} },
    };
    var prog: Program = .{
        .allocator = allocator,
        .steps = &steps,
        .blocks = &blocks,
        .outputs = &[_]TensorId{},
        .tensor_placements = &placements,
    };
    defer {
        for (prog.workspace_slots) |slot| {
            allocator.free(slot.members);
            allocator.free(slot.tile_capacities);
        }
        allocator.free(prog.workspace_slots);
    }

    var no_alias: alias_views.AliasMap = .init(allocator);
    defer no_alias.deinit();
    try plan(allocator, &mgr, &prog, &owned, &no_alias);
    try std.testing.expectEqual(@as(usize, 3), prog.workspace_slots.len);
    try std.testing.expect((try mgr.physicalBackingId(outer_tmp)) != try mgr.physicalBackingId(body_a));
    const c_backing = try mgr.physicalBackingId(body_c);
    try std.testing.expect(c_backing == try mgr.physicalBackingId(body_a) or c_backing == try mgr.physicalBackingId(body_b));
}
