const std = @import("std");
const types = @import("../backend/types.zig");
const storage = @import("../storage/storage.zig");
const executable = @import("../runtime/executable.zig");

const graph_mod = @import("graph.zig");
const infer_mod = @import("infer.zig");
const plan_mod = @import("plan.zig");
const manager_mod = @import("../storage/manager.zig");

const backend_utils = @import("../backend/utils.zig");

pub const StorageError = storage.StorageError;
pub const TiledTensor = storage.TiledTensor;

pub const StorageManager = manager_mod.StorageManager;
pub const TensorId = manager_mod.TensorId;

pub const Step = executable.Step;
pub const Program = executable.ExecutableProgram;

pub const CompileError = error{ InvalidArgument, OutOfMemory } || graph_mod.GraphError || infer_mod.InferError || StorageError;

fn compileRequire(cond: bool) CompileError!void {
    if (!cond) return CompileError.InvalidArgument;
}

fn elemCount(shape: []const usize) CompileError!usize {
    return backend_utils.elemCount(shape) catch return CompileError.InvalidArgument;
}

fn isScalarSupported(dtype: types.DType) bool {
    return switch (dtype) {
        .f32, .f16 => true,
        else => false,
    };
}

fn validateStep(mgr: *StorageManager, step: Step) CompileError!void {
    switch (step) {
        .MatMulTiled => |s| {
            const c: *const TiledTensor = mgr.getConst(s.c) catch return CompileError.InvalidArgument;
            const a: *const TiledTensor = mgr.getConst(s.a) catch return CompileError.InvalidArgument;
            const b: *const TiledTensor = mgr.getConst(s.b) catch return CompileError.InvalidArgument;

            try compileRequire(c.rank == 2 and a.rank == 2 and b.rank == 2);
            try compileRequire(!a.dtype.info().is_quantized);
            try compileRequire(!c.dtype.info().is_quantized);
            try compileRequire(isScalarSupported(a.dtype));
            try compileRequire(isScalarSupported(c.dtype));

            // Shapes.
            try compileRequire(a.shape[1] == b.shape[0]);
            try compileRequire(c.shape[0] == a.shape[0]);
            try compileRequire(c.shape[1] == b.shape[1]);

            // Canonical tiling geometry.
            try compileRequire(a.tile_shape[0] == c.tile_shape[0]);
            try compileRequire(b.tile_shape[1] == c.tile_shape[1]);
            try compileRequire(a.tile_shape[1] == b.tile_shape[0]);

            try compileRequire(a.tile_counts[0] == c.tile_counts[0]);
            try compileRequire(b.tile_counts[1] == c.tile_counts[1]);
            try compileRequire(a.tile_counts[1] == b.tile_counts[0]);

            if (b.dtype.info().is_quantized) {
                const be: usize = b.dtype.info().block_elems;
                try compileRequire(a.shape[1] % be == 0);
                try compileRequire(a.tile_shape[1] % be == 0);
                const rem: usize = a.shape[1] % a.tile_shape[1];
                if (rem != 0) try compileRequire(rem % be == 0);
            }
        },

        .ElemwiseBinaryTiled => |s| {
            const out: *const TiledTensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const a: *const TiledTensor = mgr.getConst(s.a) catch return CompileError.InvalidArgument;
            const b: *const TiledTensor = mgr.getConst(s.b) catch return CompileError.InvalidArgument;

            try compileRequire(isScalarSupported(out.dtype));
            try compileRequire(out.dtype == a.dtype and out.dtype == b.dtype);
            try compileRequire(out.rank == a.rank and out.rank == b.rank);
            if (out.rank == 1) {
                try compileRequire(out.shape[0] == a.shape[0] and out.shape[0] == b.shape[0]);
            } else {
                try compileRequire(out.shape[0] == a.shape[0] and out.shape[1] == a.shape[1]);
                try compileRequire(out.shape[0] == b.shape[0] and out.shape[1] == b.shape[1]);
            }
            try compileRequire(out.tile_shape[0] == a.tile_shape[0] and out.tile_shape[1] == a.tile_shape[1]);
            try compileRequire(out.tile_shape[0] == b.tile_shape[0] and out.tile_shape[1] == b.tile_shape[1]);
            try compileRequire(out.tile_counts[0] == a.tile_counts[0] and out.tile_counts[1] == a.tile_counts[1]);
            try compileRequire(out.tile_counts[0] == b.tile_counts[0] and out.tile_counts[1] == b.tile_counts[1]);
            _ = s.op;
        },

        .BroadcastLastDimBinaryTiled => |s| {
            const out: *const TiledTensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const a: *const TiledTensor = mgr.getConst(s.a) catch return CompileError.InvalidArgument;
            const b: *const TiledTensor = mgr.getConst(s.b) catch return CompileError.InvalidArgument;

            try compileRequire(out.rank == 2 and a.rank == 2 and b.rank == 1);
            try compileRequire(isScalarSupported(out.dtype));
            try compileRequire(out.dtype == a.dtype and out.dtype == b.dtype);
            try compileRequire(out.shape[0] == a.shape[0] and out.shape[1] == a.shape[1]);
            try compileRequire(b.shape[0] == out.shape[1]);

            try compileRequire(out.tile_shape[0] == a.tile_shape[0] and out.tile_shape[1] == a.tile_shape[1]);
            try compileRequire(out.tile_counts[0] == a.tile_counts[0] and out.tile_counts[1] == a.tile_counts[1]);

            // b is tiled along the last dimension.
            try compileRequire(b.tile_shape[0] == out.tile_shape[1]);
            try compileRequire(b.tile_counts[0] == out.tile_counts[1]);
            _ = s.op;
        },

        .ReluTiled => |s| {
            const out: *const TiledTensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const a: *const TiledTensor = mgr.getConst(s.a) catch return CompileError.InvalidArgument;
            try compileRequire(isScalarSupported(out.dtype));
            try compileRequire(out.dtype == a.dtype);
            try compileRequire(out.rank == a.rank);
            if (out.rank == 1) {
                try compileRequire(out.shape[0] == a.shape[0]);
            } else {
                try compileRequire(out.shape[0] == a.shape[0] and out.shape[1] == a.shape[1]);
            }
            try compileRequire(out.tile_shape[0] == a.tile_shape[0] and out.tile_shape[1] == a.tile_shape[1]);
            try compileRequire(out.tile_counts[0] == a.tile_counts[0] and out.tile_counts[1] == a.tile_counts[1]);
        },

        .CopyTiled => |s| {
            const dst: *const TiledTensor = mgr.getConst(s.dst) catch return CompileError.InvalidArgument;
            const src: *const TiledTensor = mgr.getConst(s.src) catch return CompileError.InvalidArgument;
            try compileRequire(dst.dtype == src.dtype);
            try compileRequire(dst.rank == src.rank);
            if (dst.rank == 1) {
                try compileRequire(dst.shape[0] == src.shape[0]);
            } else {
                try compileRequire(dst.shape[0] == src.shape[0] and dst.shape[1] == src.shape[1]);
            }
            try compileRequire(dst.tile_shape[0] == src.tile_shape[0] and dst.tile_shape[1] == src.tile_shape[1]);
            try compileRequire(dst.tile_counts[0] == src.tile_counts[0] and dst.tile_counts[1] == src.tile_counts[1]);
        },

        .ReduceAll => |s| {
            const out: *const TiledTensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const a: *const TiledTensor = mgr.getConst(s.a) catch return CompileError.InvalidArgument;
            try compileRequire(out.rank == 1 and out.shape[0] == 1);
            try compileRequire(isScalarSupported(out.dtype));
            try compileRequire(out.dtype == a.dtype);
            try compileRequire(isScalarSupported(a.dtype));
            _ = s.op;
        },

        .ReTileCopyScalar => |s| {
            const dst: *const TiledTensor = mgr.getConst(s.dst) catch return CompileError.InvalidArgument;
            const src: *const TiledTensor = mgr.getConst(s.src) catch return CompileError.InvalidArgument;
            try compileRequire(isScalarSupported(dst.dtype));
            try compileRequire(dst.dtype == src.dtype);
            try compileRequire(dst.rank == src.rank);
            if (dst.rank == 1) {
                try compileRequire(dst.shape[0] == src.shape[0]);
            } else {
                try compileRequire(dst.shape[0] == src.shape[0] and dst.shape[1] == src.shape[1]);
            }
        },

        .ReshapeScalar => |s| {
            const dst: *const TiledTensor = mgr.getConst(s.dst) catch return CompileError.InvalidArgument;
            const src: *const TiledTensor = mgr.getConst(s.src) catch return CompileError.InvalidArgument;
            try compileRequire(isScalarSupported(dst.dtype));
            try compileRequire(dst.dtype == src.dtype);
            const src_elems: usize = try elemCount(if (src.rank == 1) &[_]usize{src.shape[0]} else &[_]usize{ src.shape[0], src.shape[1] });
            const dst_elems: usize = try elemCount(if (dst.rank == 1) &[_]usize{dst.shape[0]} else &[_]usize{ dst.shape[0], dst.shape[1] });
            try compileRequire(src_elems == dst_elems);
        },

        .Transpose2DScalar => |s| {
            const dst: *const TiledTensor = mgr.getConst(s.dst) catch return CompileError.InvalidArgument;
            const src: *const TiledTensor = mgr.getConst(s.src) catch return CompileError.InvalidArgument;
            try compileRequire(isScalarSupported(dst.dtype));
            try compileRequire(dst.dtype == src.dtype);
            try compileRequire(dst.rank == 2 and src.rank == 2);
            try compileRequire(dst.shape[0] == src.shape[1] and dst.shape[1] == src.shape[0]);
        },

        .Slice2DScalar => |s| {
            const dst: *const TiledTensor = mgr.getConst(s.dst) catch return CompileError.InvalidArgument;
            const src: *const TiledTensor = mgr.getConst(s.src) catch return CompileError.InvalidArgument;
            try compileRequire(isScalarSupported(dst.dtype));
            try compileRequire(dst.dtype == src.dtype);
            try compileRequire(dst.rank == 2 and src.rank == 2);
            try compileRequire(s.start0 + dst.shape[0] <= src.shape[0]);
            try compileRequire(s.start1 + dst.shape[1] <= src.shape[1]);
        },
    }
}

fn appendStepChecked(
    allocator: std.mem.Allocator,
    mgr: *StorageManager,
    steps: *std.ArrayList(Step),
    step: Step,
) CompileError!void {
    try validateStep(mgr, step);
    steps.append(allocator, step) catch return CompileError.OutOfMemory;
}

fn tileShapeForValue(policy: plan_mod.TilePolicy, dtype: types.DType, shape: []const usize) [2]usize {
    // v0: rank 1 uses [t,1] in tiled storage. rank 2 uses square tiles by default.
    if (shape.len == 1) {
        const t1: [1]usize = plan_mod.chooseTileShape1D(policy, shape[0]);
        _ = dtype;
        return .{ t1[0], 1 };
    }
    const t2: [2]usize = plan_mod.chooseTileShape2DSquare(policy, shape[0], shape[1]);
    _ = dtype;
    return t2;
}

pub fn compileGraph(
    allocator: std.mem.Allocator,
    graph: *graph_mod.Graph,
    mgr: *StorageManager,
    policy: plan_mod.TilePolicy,
) CompileError!Program {
    try infer_mod.infer(graph);

    // Map graph values -> concrete tensors.
    const v_count: usize = graph.values.items.len;
    var value_tensor: []TensorId = try allocator.alloc(TensorId, v_count);
    defer allocator.free(value_tensor);
    @memset(value_tensor, @as(TensorId, 0));

    var value_has_tensor: []bool = try allocator.alloc(bool, v_count);
    defer allocator.free(value_has_tensor);
    @memset(value_has_tensor, false);

    // External bindings.
    for (graph.values.items, 0..) |v, i| {
        if (v.external) |ext| {
            const tid: TensorId = @intCast(ext);
            const t: *const TiledTensor = try mgr.getConst(tid);
            if (v.dtype.? != t.dtype) return CompileError.InvalidArgument;
            if (v.shape.len != t.rank) return CompileError.InvalidArgument;
            if (t.rank == 1) {
                if (v.shape[0] != t.shape[0]) return CompileError.InvalidArgument;
            } else {
                if (v.shape[0] != t.shape[0] or v.shape[1] != t.shape[1]) return CompileError.InvalidArgument;
            }

            value_tensor[i] = tid;
            value_has_tensor[i] = true;
        }
    }

    // Compile into a dynamic step list first.
    var steps = std.ArrayListUnmanaged(Step){};
    errdefer steps.deinit(allocator);

    // Helper: allocate tensor for a value (owned).
    const AllocCtx = struct {
        allocator: std.mem.Allocator,
        mgr: *StorageManager,
        policy: plan_mod.TilePolicy,
        value_tensor: []TensorId,
        value_has_tensor: []bool,

        fn ensureValueTensor(self: *@This(), value_index: usize, dtype: types.DType, shape: []const usize, tile_shape: [2]usize) CompileError!TensorId {
            if (self.value_has_tensor[value_index]) return self.value_tensor[value_index];

            const rank: usize = shape.len;
            const tile_shape_slice: []const usize = if (rank == 1) (&[_]usize{tile_shape[0]}) else (&[_]usize{ tile_shape[0], tile_shape[1] });
            const tid: TensorId = try self.mgr.createTiledTensor(dtype, shape, tile_shape_slice, .{ .tile_alignment = self.policy.tile_alignment });
            self.value_tensor[value_index] = tid;
            self.value_has_tensor[value_index] = true;
            return tid;
        }
    };

    var ctx: AllocCtx = .{ .allocator = allocator, .mgr = mgr, .policy = policy, .value_tensor = value_tensor, .value_has_tensor = value_has_tensor };

    // Lower nodes in order.
    for (graph.nodes.items) |node| {
        const out_idx: usize = @intCast(node.output);
        const out_v = graph.values.items[out_idx];
        const out_dt: types.DType = out_v.dtype.?;
        const out_shape: []const usize = out_v.shape;

        switch (node.op) {
            .MatMul => |mm| {
                const a_id: usize = @intCast(node.inputs[0]);
                const b_id: usize = @intCast(node.inputs[1]);
                const a_v = graph.values.items[a_id];
                const b_v = graph.values.items[b_id];
                const m: usize = a_v.shape[0];
                const k: usize = a_v.shape[1];
                const n: usize = b_v.shape[1];

                const tiles = plan_mod.chooseMatMulTiles(policy, m, n, k, b_v.dtype.?);

                const c_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, .{ tiles.tm, tiles.tn });

                // Ensure A and B have compatible tiling; if not, materialize into new tensors.
                const a_needed_tile: [2]usize = .{ tiles.tm, tiles.tk };
                const b_needed_tile: [2]usize = .{ tiles.tk, tiles.tn };

                const a_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, &steps, mgr, policy, &ctx, a_id, a_v.dtype.?, a_v.shape, a_needed_tile);
                const b_tid: TensorId = try ensureTilingMaybeRetile(allocator, &steps, mgr, policy, &ctx, b_id, b_v.dtype.?, b_v.shape, b_needed_tile);

                try appendStepChecked(allocator, mgr, &steps, .{ .MatMulTiled = .{ .c = c_tid, .a = a_tid, .b = b_tid, .alpha = mm.alpha, .beta = mm.beta } });
            },

            .ElemwiseBinary => |eb| {
                const a_id: usize = @intCast(node.inputs[0]);
                const b_id: usize = @intCast(node.inputs[1]);
                const a_v = graph.values.items[a_id];
                const tile: [2]usize = tileShapeForValue(policy, out_dt, out_shape);

                const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, tile);
                const a_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, &steps, mgr, policy, &ctx, a_id, a_v.dtype.?, a_v.shape, tile);
                const b_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, &steps, mgr, policy, &ctx, b_id, a_v.dtype.?, a_v.shape, tile);

                try appendStepChecked(allocator, mgr, &steps, .{ .ElemwiseBinaryTiled = .{ .op = eb.op, .out = out_tid, .a = a_tid, .b = b_tid } });
            },

            .BroadcastLastDimBinary => |bb| {
                const a_id: usize = @intCast(node.inputs[0]);
                const b_id: usize = @intCast(node.inputs[1]);
                const a_v = graph.values.items[a_id];
                const b_v = graph.values.items[b_id];

                const tile_out: [2]usize = tileShapeForValue(policy, out_dt, out_shape);
                const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, tile_out);
                const a_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, &steps, mgr, policy, &ctx, a_id, a_v.dtype.?, a_v.shape, tile_out);

                // b must be rank-1 tiled to match tile_out[1].
                const b_tile: [2]usize = .{ tile_out[1], 1 };
                const b_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, &steps, mgr, policy, &ctx, b_id, b_v.dtype.?, b_v.shape, b_tile);

                try appendStepChecked(allocator, mgr, &steps, .{ .BroadcastLastDimBinaryTiled = .{ .op = bb.op, .out = out_tid, .a = a_tid, .b = b_tid } });
            },

            .Relu => {
                const a_id: usize = @intCast(node.inputs[0]);
                const a_v = graph.values.items[a_id];
                const tile: [2]usize = tileShapeForValue(policy, out_dt, out_shape);
                const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, tile);
                const a_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, &steps, mgr, policy, &ctx, a_id, a_v.dtype.?, a_v.shape, tile);
                try appendStepChecked(allocator, mgr, &steps, .{ .ReluTiled = .{ .out = out_tid, .a = a_tid } });
            },

            .Reduce => |rr| {
                const a_id: usize = @intCast(node.inputs[0]);
                const a_v = graph.values.items[a_id];
                const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, .{ 1, 1 });
                // Reduce supports any tiling; no need to retile.
                const a_tid: TensorId = try ensureAnyTensor(&ctx, a_id);
                _ = a_v;
                try appendStepChecked(allocator, mgr, &steps, .{ .ReduceAll = .{ .op = rr.op, .out = out_tid, .a = a_tid } });
            },

            .Copy => {
                const a_id: usize = @intCast(node.inputs[0]);
                const a_v = graph.values.items[a_id];
                const tile: [2]usize = tileShapeForValue(policy, out_dt, out_shape);
                const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, tile);
                const a_tid: TensorId = try ensureTilingMaybeRetile(allocator, &steps, mgr, policy, &ctx, a_id, a_v.dtype.?, a_v.shape, tile);
                try appendStepChecked(allocator, mgr, &steps, .{ .CopyTiled = .{ .dst = out_tid, .src = a_tid } });
            },

            .ViewReshape => |_| {
                const a_id: usize = @intCast(node.inputs[0]);
                const a_v = graph.values.items[a_id];
                const out_tile: [2]usize = tileShapeForValue(policy, out_dt, out_shape);
                const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile);
                const a_tid: TensorId = try ensureAnyTensor(&ctx, a_id);
                // Materialize reshape (scalar-only for now).
                if (a_v.dtype.?.info().is_quantized) return CompileError.InvalidArgument;
                try appendStepChecked(allocator, mgr, &steps, .{ .ReshapeScalar = .{ .dst = out_tid, .src = a_tid } });
            },

            .ViewTranspose2D => {
                const a_id: usize = @intCast(node.inputs[0]);
                const a_v = graph.values.items[a_id];
                const out_tile: [2]usize = tileShapeForValue(policy, out_dt, out_shape);
                const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile);
                const a_tid: TensorId = try ensureAnyTensor(&ctx, a_id);
                if (a_v.dtype.?.info().is_quantized) return CompileError.InvalidArgument;
                try appendStepChecked(allocator, mgr, &steps, .{ .Transpose2DScalar = .{ .dst = out_tid, .src = a_tid } });
            },

            .ViewSlice2D => |sl| {
                const a_id: usize = @intCast(node.inputs[0]);
                const a_v = graph.values.items[a_id];
                const out_tile: [2]usize = tileShapeForValue(policy, out_dt, out_shape);
                const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile);
                const a_tid: TensorId = try ensureAnyTensor(&ctx, a_id);
                if (a_v.dtype.?.info().is_quantized) return CompileError.InvalidArgument;
                try appendStepChecked(allocator, mgr, &steps, .{ .Slice2DScalar = .{ .dst = out_tid, .src = a_tid, .start0 = sl.start0, .start1 = sl.start1 } });
            },
        }
    }

    // Collect outputs.
    const out_ids: []TensorId = try allocator.alloc(TensorId, graph.outputs.items.len);
    for (graph.outputs.items, 0..) |vid, i| {
        const idx: usize = @intCast(vid);
        if (!value_has_tensor[idx]) return CompileError.InvalidArgument;
        out_ids[i] = value_tensor[idx];
    }

    const steps_slice: []Step = try steps.toOwnedSlice(allocator);

    return .{ .allocator = allocator, .steps = steps_slice, .outputs = out_ids };
}

fn ensureAnyTensor(ctx: anytype, value_index: usize) CompileError!TensorId {
    if (!ctx.value_has_tensor[value_index]) return CompileError.InvalidArgument;
    return ctx.value_tensor[value_index];
}

fn ensureTilingMaybeRetile(
    allocator: std.mem.Allocator,
    steps: *std.ArrayListUnmanaged(Step),
    mgr: *StorageManager,
    policy: plan_mod.TilePolicy,
    ctx: anytype,
    value_index: usize,
    dtype: types.DType,
    shape: []const usize,
    want_tile: [2]usize,
) CompileError!TensorId {
    // Quant tensors can be re-tiled by raw block-copy only if shapes match.
    // We allow it here (it is still a scalar/quant packed view per tile), but we only have a scalar retile kernel.
    // For now, require quant tensors to already have the desired tiling.
    if (dtype.info().is_quantized) {
        const tid: TensorId = try ensureAnyTensor(ctx, value_index);
        const t: *const TiledTensor = try mgr.getConst(tid);
        if (t.tile_shape[0] != want_tile[0] or t.tile_shape[1] != want_tile[1]) return CompileError.InvalidArgument;
        return tid;
    }
    return ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, value_index, dtype, shape, want_tile);
}

fn ensureTilingScalarMaybeRetile(
    allocator: std.mem.Allocator,
    steps: *std.ArrayListUnmanaged(Step),
    mgr: *StorageManager,
    policy: plan_mod.TilePolicy,
    ctx: anytype,
    value_index: usize,
    dtype: types.DType,
    shape: []const usize,
    want_tile: [2]usize,
) CompileError!TensorId {
    const cur: TensorId = try ensureAnyTensor(ctx, value_index);
    const cur_t: *const TiledTensor = try mgr.getConst(cur);

    if (cur_t.tile_shape[0] == want_tile[0] and cur_t.tile_shape[1] == want_tile[1]) return cur;

    // Allocate new tensor with desired tiling and insert a scalar retile copy.
    const rank: usize = shape.len;
    const tile_shape_slice: []const usize = if (rank == 1) (&[_]usize{want_tile[0]}) else (&[_]usize{ want_tile[0], want_tile[1] });
    const new_tid: TensorId = try mgr.createTiledTensor(dtype, shape, tile_shape_slice, .{ .tile_alignment = policy.tile_alignment });
    try appendStepChecked(allocator, mgr, steps, .{ .ReTileCopyScalar = .{ .dst = new_tid, .src = cur } });
    ctx.value_tensor[value_index] = new_tid;
    ctx.value_has_tensor[value_index] = true;
    return new_tid;
}
