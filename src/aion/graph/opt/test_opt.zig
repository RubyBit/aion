// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Tests optional graph and step rewrite passes with explicit policies.
//! Step fusion also tests `compileGraph` to pin target defaults.

const std = @import("std");

const backend_mod = @import("../../backend/backend.zig");
const cpu_backend_mod = @import("../../backend/cpu/cpu_backend.zig");
const manager_mod = @import("../../storage/manager.zig");
const types = @import("../../backend/types.zig");
const graph_mod = @import("../graph.zig");
const opt = @import("../opt.zig");
const plan_mod = @import("../plan.zig");
const program = @import("../program.zig");

const alias_views = @import("alias_views.zig");
const horizontal_matmul = @import("horizontal_matmul.zig");

const Graph = graph_mod.Graph;
const Policy = opt.Policy;
const StorageManager = manager_mod.StorageManager;
const TensorId = manager_mod.TensorId;
const ValueId = graph_mod.ValueId;

const cpu_tiles: plan_mod.TilePolicy = .{ .tile_alignment = 64 };
const cpu_target: program.Target = .cpu(cpu_tiles);

/// The `DeviceRef` a backend kind executes on, for tests that sweep both.
fn deviceFor(kind: types.BackendKind) manager_mod.DeviceRef {
    return .{ .kind = if (kind == .cpu) .cpu else .gpu };
}

fn asF32(buf: []u8) []align(1) f32 {
    const ptr: [*]align(1) f32 = @ptrCast(buf.ptr);
    return ptr[0 .. buf.len / @sizeOf(f32)];
}

fn countOp(g: *const Graph, tag: std.meta.Tag(graph_mod.Op)) usize {
    var c: usize = 0;
    for (g.nodes.items) |node| {
        if (std.meta.activeTag(node.op) == tag) c += 1;
    }
    return c;
}

fn countRegionOp(g: *const Graph, region: graph_mod.RegionId, tag: std.meta.Tag(graph_mod.Op)) usize {
    var c: usize = 0;
    for (g.regions.items[@intCast(region)].nodes) |node| {
        if (std.meta.activeTag(node.op) == tag) c += 1;
    }
    return c;
}

fn countBlockStep(prog: *const program.Program, tag: std.meta.Tag(@TypeOf(prog.steps[0].op))) usize {
    var c: usize = 0;
    for (prog.blocks) |block| {
        for (block.steps) |step| {
            if (std.meta.activeTag(step.op) == tag) c += 1;
        }
    }
    return c;
}

fn countStep(prog: *const program.Program, tag: std.meta.Tag(@TypeOf(prog.steps[0].op))) usize {
    var c: usize = 0;
    for (prog.steps) |step| {
        if (std.meta.activeTag(step.op) == tag) c += 1;
    }
    return c;
}

fn f32Tensor(sm: *StorageManager, shape: []const usize, seed: usize) !TensorId {
    var n: usize = 1;
    for (shape) |d| n *= d;
    const vals = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(vals);
    for (vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i + seed) % 11)) - 5)) * 0.1;
    const tid = try sm.createTiledTensor(.f32, shape, shape, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(tid, std.mem.sliceAsBytes(vals));
    return tid;
}

fn maxAbsDiff(a: []u8, b: []u8) f32 {
    var m: f32 = 0;
    for (asF32(a), asF32(b)) |x, y| m = @max(m, @abs(x - y));
    return m;
}

// ---------------------------------------------------------------------------
// pointwise_conv
// ---------------------------------------------------------------------------

const ConvDims = struct { l: usize = 3, cin: usize = 8, cout: usize = 4 };

fn buildConvGraph(g: *Graph, x_tid: TensorId, w_tid: TensorId, bias_tid: ?TensorId, d: ConvDims) !void {
    const x = try g.addInput(.f32, &[_]usize{ 1, d.l, d.cin });
    try g.bindExternal(x, @intCast(x_tid));
    const w = try g.addInput(.f32, &[_]usize{ 1, d.cin, d.cout }); // [k=1, c_in, c_out]
    try g.bindExternal(w, @intCast(w_tid));
    var bias: ?ValueId = null;
    if (bias_tid) |bt| {
        const bv = try g.addInput(.f32, &[_]usize{d.cout});
        try g.bindExternal(bv, @intCast(bt));
        bias = bv;
    }
    const out = try g.addConv1D(x, w, bias, 1, 1, 0, 0, 1); // stride/dil 1, no pad, groups 1
    try g.setOutputs(&[_]ValueId{out});
}

fn convParity(with_bias: bool) !void {
    const allocator = std.testing.allocator;
    const d = ConvDims{};

    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const x_tid = try f32Tensor(&sm, &[_]usize{ 1, d.l, d.cin }, 1);
    const w_tid = try f32Tensor(&sm, &[_]usize{ 1, d.cin, d.cout }, 7);
    const bias_tid: ?TensorId = if (with_bias) try f32Tensor(&sm, &[_]usize{d.cout}, 3) else null;

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: backend_mod.Backend = cpu.backend();

    const bytes = d.l * d.cout * 4;
    const ref = try allocator.alloc(u8, bytes);
    defer allocator.free(ref);
    {
        var g = Graph.init(allocator);
        defer g.deinit();
        try buildConvGraph(&g, x_tid, w_tid, bias_tid, d);
        var prog = try program.compileGraph(allocator, &g, &sm, (cpu_target.withPasses(.empty)));
        defer prog.deinit();
        try backend.executeProgram(&prog, sm.tensorStore());
        try std.testing.expectEqual(@as(usize, 1), countOp(&g, .Conv1D));
        try std.testing.expectEqual(@as(usize, 0), countOp(&g, .MatMul));
        try sm.readToPackedScalar(prog.outputs[0], ref);
    }
    {
        var g = Graph.init(allocator);
        defer g.deinit();
        try buildConvGraph(&g, x_tid, w_tid, bias_tid, d);
        var prog = try program.compileGraph(allocator, &g, &sm, (cpu_target.withPasses(.initOne(.pointwise_conv))));
        defer prog.deinit();
        try backend.executeProgram(&prog, sm.tensorStore());

        try std.testing.expectEqual(@as(usize, 0), countOp(&g, .Conv1D));
        try std.testing.expectEqual(@as(usize, 1), countOp(&g, .MatMul));
        if (with_bias) try std.testing.expectEqual(@as(usize, 1), countOp(&g, .ElemwiseBinary));

        const got = try allocator.alloc(u8, bytes);
        defer allocator.free(got);
        try sm.readToPackedScalar(prog.outputs[0], got);
        try std.testing.expect(maxAbsDiff(got, ref) <= 1e-5);
    }
}

test "pointwise_conv: 1x1 conv1d lowers to matmul and matches conv output" {
    try convParity(false);
}

test "pointwise_conv: biased 1x1 conv1d lowers to matmul + broadcast add" {
    try convParity(true);
}

test "pointwise_conv: non-pointwise conv1d is left as Conv1D" {
    const allocator = std.testing.allocator;

    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const x_tid = try f32Tensor(&sm, &[_]usize{ 1, 6, 8 }, 1);
    const w_tid = try f32Tensor(&sm, &[_]usize{ 3, 8, 4 }, 7); // k=3

    var g = Graph.init(allocator);
    defer g.deinit();
    const x = try g.addInput(.f32, &[_]usize{ 1, 6, 8 });
    try g.bindExternal(x, @intCast(x_tid));
    const w = try g.addInput(.f32, &[_]usize{ 3, 8, 4 });
    try g.bindExternal(w, @intCast(w_tid));
    try g.setOutputs(&[_]ValueId{try g.addConv1D(x, w, null, 1, 1, 1, 1, 1)});

    var prog = try program.compileGraph(allocator, &g, &sm, (cpu_target.withPasses(.initOne(.pointwise_conv))));
    defer prog.deinit();
    try std.testing.expectEqual(@as(usize, 1), countOp(&g, .Conv1D));
    try std.testing.expectEqual(@as(usize, 0), countOp(&g, .MatMul));
}

// ---------------------------------------------------------------------------
// horizontal_matmul: the weight concat
// ---------------------------------------------------------------------------

/// A `[1, K, N]` quant tensor filled with a deterministic per-byte pattern (the concat is
/// byte-level, so the values don't matter), plus the packed bytes written.
fn quantWeight(
    allocator: std.mem.Allocator,
    sm: *StorageManager,
    dtype: types.DType,
    k: usize,
    n: usize,
    seed: u8,
) !struct { tid: TensorId, bytes: []u8 } {
    const info = dtype.info();
    const buf = try allocator.alloc(u8, (k / info.block_elems) * n * info.block_bytes);
    for (buf, 0..) |*b, i| b.* = @truncate(i *% 131 +% seed);

    const tid = try sm.createTiledTensor(dtype, &[_]usize{ 1, k, n }, &[_]usize{ 1, k, n }, .{
        .tile_alignment = 64,
        .quant_axis = 1,
    });
    try sm.writeFromPackedQuant(tid, buf);
    return .{ .tid = tid, .bytes = buf };
}

fn concatByteIdentity(dtype: types.DType) !void {
    const allocator = std.testing.allocator;
    const k: usize = 64; // 2 K-blocks
    const n0: usize = 3;
    const n1: usize = 5;
    const info = dtype.info();
    const bb = info.block_bytes;
    const kb = k / info.block_elems;

    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const w0 = try quantWeight(allocator, &sm, dtype, k, n0, 1);
    defer allocator.free(w0.bytes);
    const w1 = try quantWeight(allocator, &sm, dtype, k, n1, 200);
    defer allocator.free(w1.bytes);

    const out = try horizontal_matmul.concatColumns(allocator, &sm, cpu_target, &[_]TensorId{ w0.tid, w1.tid });

    // Per block-row, w0's n0 blocks then w1's n1 blocks.
    const sum_n = n0 + n1;
    const expected = try allocator.alloc(u8, kb * sum_n * bb);
    defer allocator.free(expected);
    for (0..kb) |r| {
        @memcpy(expected[(r * sum_n) * bb ..][0 .. n0 * bb], w0.bytes[(r * n0) * bb ..][0 .. n0 * bb]);
        @memcpy(expected[(r * sum_n + n0) * bb ..][0 .. n1 * bb], w1.bytes[(r * n1) * bb ..][0 .. n1 * bb]);
    }

    const got = try allocator.alloc(u8, kb * sum_n * bb);
    defer allocator.free(got);
    try sm.readToPackedQuant(out, got);
    try std.testing.expectEqualSlices(u8, expected, got);

    // Same sources at the same tiling reuse the result (no re-alloc, no leak).
    const again = try horizontal_matmul.concatColumns(allocator, &sm, cpu_target, &[_]TensorId{ w0.tid, w1.tid });
    try std.testing.expectEqual(out, again);
}

test "horizontal_matmul: concat is byte-identical and memoized (q8_0)" {
    try concatByteIdentity(.q8_0);
}

test "horizontal_matmul: concat is byte-identical and memoized (q4_0)" {
    try concatByteIdentity(.q4_0);
}

// The memo key includes the tiling, because tiling is chosen per target and a quantized
// weight cannot be retiled downstream. Keyed on sources alone, compiling one store for
// two targets would hand the second a layout built for the first.
test "horizontal_matmul: a different tiling derives its own weight" {
    const allocator = std.testing.allocator;
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const w0 = try quantWeight(allocator, &sm, .q8_0, 64, 32, 1);
    defer allocator.free(w0.bytes);
    const w1 = try quantWeight(allocator, &sm, .q8_0, 64, 32, 9);
    defer allocator.free(w1.bytes);
    const sources = [_]TensorId{ w0.tid, w1.tid };

    const narrow: plan_mod.TilePolicy = .{ .tile_alignment = 64, .base_square_2d = 32 };
    const wide = try horizontal_matmul.concatColumns(allocator, &sm, cpu_target, &sources);
    const thin = try horizontal_matmul.concatColumns(allocator, &sm, .cpu(narrow), &sources);

    try std.testing.expect(wide != thin);
    const a = try sm.getConst(wide);
    const b = try sm.getConst(thin);
    try std.testing.expect(!std.mem.eql(usize, a.tile_shape, b.tile_shape));
}

// Stacking is refused rather than silently mis-resolved: a chain would need composed
// views, so a swap of the original would land in a tensor nothing reads. Folding the same
// weight into two SEPARATE results stays legal — a swap reaches both.
test "horizontal_matmul: deriving from a derived weight is refused" {
    const allocator = std.testing.allocator;
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const w0 = try quantWeight(allocator, &sm, .q8_0, 64, 32, 1);
    defer allocator.free(w0.bytes);
    const w1 = try quantWeight(allocator, &sm, .q8_0, 64, 32, 9);
    defer allocator.free(w1.bytes);

    const fused = try horizontal_matmul.concatColumns(allocator, &sm, cpu_target, &[_]TensorId{ w0.tid, w1.tid });
    const w2 = try quantWeight(allocator, &sm, .q8_0, 64, 32, 21);
    defer allocator.free(w2.bytes);

    try std.testing.expectError(
        error.InvalidArgument,
        horizontal_matmul.concatColumns(allocator, &sm, cpu_target, &[_]TensorId{ fused, w2.tid }),
    );

    // A second, independent fold of w0 is fine, and a swap then updates both copies.
    const other = try horizontal_matmul.concatColumns(allocator, &sm, cpu_target, &[_]TensorId{ w0.tid, w2.tid });
    const replacement = try quantWeight(allocator, &sm, .q8_0, 64, 32, 77);
    defer allocator.free(replacement.bytes);
    try sm.writeDerivedSource(w0.tid, replacement.tid);

    const bytes = (64 / 32) * 32 * 34;
    const want = try allocator.alloc(u8, bytes);
    defer allocator.free(want);
    try sm.readToPackedQuant(replacement.tid, want);
    for ([_]TensorId{ fused, other }) |result| {
        const whole = try allocator.alloc(u8, (64 / 32) * 64 * 34);
        defer allocator.free(whole);
        try sm.readToPackedQuant(result, whole);
        // w0 is the first source, so it owns the leading 32 blocks of each block-row.
        for (0..64 / 32) |r| {
            try std.testing.expectEqualSlices(u8, want[(r * 32) * 34 ..][0 .. 32 * 34], whole[(r * 64) * 34 ..][0 .. 32 * 34]);
        }
    }
}

// ---------------------------------------------------------------------------
// horizontal_matmul: fusion parity and weight IO
// ---------------------------------------------------------------------------

/// Quantize a `[K, N]` f32 weight to q8_0 matmul-B packed bytes: blocks run along K
/// (the quant axis), block (kb, j) holds 32 K-values of column j.
fn packQ8MatmulB(allocator: std.mem.Allocator, vals: []const f32, k: usize, n: usize) ![]u8 {
    const kb = k / 32;
    const buf = try allocator.alloc(u8, kb * n * 34);
    for (0..kb) |b| {
        for (0..n) |j| {
            var absmax: f32 = 0;
            for (0..32) |t| absmax = @max(absmax, @abs(vals[(b * 32 + t) * n + j]));
            const scale: f32 = if (absmax == 0) 1 else absmax / 127.0;
            const inv: f32 = if (absmax == 0) 0 else 1.0 / scale;
            const off = (b * n + j) * 34;
            std.mem.writeInt(u16, buf[off .. off + 2][0..2], @bitCast(@as(f16, @floatCast(scale))), .little);
            for (0..32) |t| {
                var q: i32 = @intFromFloat(@round(vals[(b * 32 + t) * n + j] * inv));
                q = @max(@as(i32, -128), @min(@as(i32, 127), q));
                buf[off + 2 + t] = @bitCast(@as(i8, @intCast(q)));
            }
        }
    }
    return buf;
}

fn q8Weight(allocator: std.mem.Allocator, sm: *StorageManager, k: usize, n: usize, seed: usize) !TensorId {
    const vals = try allocator.alloc(f32, k * n);
    defer allocator.free(vals);
    for (vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i + seed) % 17)) - 8)) * 0.1;
    const buf = try packQ8MatmulB(allocator, vals, k, n);
    defer allocator.free(buf);

    // Tile exactly as the MatMul lowering tiles a quant B, so no retile is needed.
    const t = plan_mod.chooseMatMulTiles(cpu_tiles, plan_mod.matMulMHint(cpu_tiles), n, k, .q8_0);
    const tid = try sm.createTiledTensor(.q8_0, &[_]usize{ 1, k, n }, &[_]usize{ 1, t.tk, t.tn }, .{
        .tile_alignment = 64,
        .quant_axis = 1,
    });
    try sm.writeFromPackedQuant(tid, buf);
    return tid;
}

const ProjDims = struct { m: usize = 2, k: usize = 64, ns: [3]usize = .{ 32, 32, 64 } };

/// Outputs: o0, relu(o1), o2 — exercises both output-remap and consumer-remap.
fn buildProjGraph(g: *Graph, a_tid: TensorId, w: [3]TensorId, d: ProjDims) !void {
    const a_in = try g.addInput(.f32, &[_]usize{ 1, d.m, d.k });
    try g.bindExternal(a_in, @intCast(a_tid));
    var outs: [3]ValueId = undefined;
    for (0..3) |i| {
        const w_in = try g.addInput(.q8_0, &[_]usize{ 1, d.k, d.ns[i] });
        try g.bindExternal(w_in, @intCast(w[i]));
        outs[i] = try g.addMatMul(a_in, w_in, 1.0, 0.0);
    }
    try g.setOutputs(&[_]ValueId{ outs[0], try g.addRelu(outs[1]), outs[2] });
}

test "horizontal_matmul: fusion matches the unfused output" {
    const allocator = std.testing.allocator;
    const d = ProjDims{};

    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const a_vals = try allocator.alloc(f32, d.m * d.k);
    defer allocator.free(a_vals);
    for (a_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) * 0.05;
    const a_tid = try sm.createTiledTensor(.f32, &[_]usize{ 1, d.m, d.k }, &[_]usize{ 1, d.m, d.k }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(a_tid, std.mem.sliceAsBytes(a_vals));

    const w: [3]TensorId = .{
        try q8Weight(allocator, &sm, d.k, d.ns[0], 0),
        try q8Weight(allocator, &sm, d.k, d.ns[1], 5),
        try q8Weight(allocator, &sm, d.k, d.ns[2], 11),
    };

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: backend_mod.Backend = cpu.backend();

    const sizes = [_]usize{ d.m * d.ns[0], d.m * d.ns[1], d.m * d.ns[2] };
    var ref: [3][]u8 = undefined;
    {
        var g = Graph.init(allocator);
        defer g.deinit();
        try buildProjGraph(&g, a_tid, w, d);
        var prog = try program.compileGraph(allocator, &g, &sm, (cpu_target.withPasses(.empty)));
        defer prog.deinit();
        try backend.executeProgram(&prog, sm.tensorStore());
        try std.testing.expectEqual(@as(usize, 3), countOp(&g, .MatMul));
        for (0..3) |i| {
            ref[i] = try allocator.alloc(u8, sizes[i] * 4);
            try sm.readToPackedScalar(prog.outputs[i], ref[i]);
        }
    }
    defer for (ref) |b| allocator.free(b);

    // Twice: the second compile hits the concat memo, which must produce the same result.
    for (0..2) |_| {
        var g = Graph.init(allocator);
        defer g.deinit();
        try buildProjGraph(&g, a_tid, w, d);
        var prog = try program.compileGraph(allocator, &g, &sm, (cpu_target.withPasses(.initOne(.horizontal_matmul))));
        defer prog.deinit();
        try backend.executeProgram(&prog, sm.tensorStore());

        try std.testing.expectEqual(@as(usize, 1), countOp(&g, .MatMul));
        try std.testing.expectEqual(@as(usize, 3), countOp(&g, .ViewSliceND));

        for (0..3) |i| {
            const got = try allocator.alloc(u8, sizes[i] * 4);
            defer allocator.free(got);
            try sm.readToPackedScalar(prog.outputs[i], got);
            try std.testing.expect(maxAbsDiff(got, ref[i]) <= 1e-4);
        }
    }

    // The pass is pure: reclaiming the sources is the model layer's call, not its.
    for (w) |wt| try std.testing.expect((try sm.getConst(wt)).data.len != 0);
}

// A fused matmul feeding a loop body: the producer is dropped, so a pass that remapped
// only `graph.nodes` would leave the body reading a value nothing writes. The Rewriter
// remaps regions too, which is why this compiles at all.
test "horizontal_matmul: a fused output read inside a loop body still resolves" {
    const allocator = std.testing.allocator;
    const k: usize = 64;
    const n: usize = 32;

    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const a_tid = try f32Tensor(&sm, &[_]usize{ 1, 1, k }, 3);
    const w0 = try q8Weight(allocator, &sm, k, n, 1);
    const w1 = try q8Weight(allocator, &sm, k, n, 4);

    var g = Graph.init(allocator);
    defer g.deinit();

    const a = try g.addInput(.f32, &[_]usize{ 1, 1, k });
    try g.bindExternal(a, @intCast(a_tid));
    const bw0 = try g.addInput(.q8_0, &[_]usize{ 1, k, n });
    try g.bindExternal(bw0, @intCast(w0));
    const bw1 = try g.addInput(.q8_0, &[_]usize{ 1, k, n });
    try g.bindExternal(bw1, @intCast(w1));

    const p0 = try g.addMatMul(a, bw0, 1.0, 0.0);
    const p1 = try g.addMatMul(a, bw1, 1.0, 0.0);

    // A body that reads `p0` directly — an outer value, not routed through an operand.
    try g.beginRegion();
    const inner = try g.addRelu(p0);
    const body = try g.endRegion(&[_]ValueId{inner});

    try g.setOutputs(&[_]ValueId{try g.addLoop(p1, body, 2)});

    var prog = try program.compileGraph(allocator, &g, &sm, (cpu_target.withPasses(.initOne(.horizontal_matmul))));
    defer prog.deinit();
    try std.testing.expectEqual(@as(usize, 1), countOp(&g, .MatMul));
}

// Swapping a weight the pass folded away: the fused weight is the canonical store, so
// the write lands in its region and a read materializes the logical weight back out.
test "horizontal_matmul: a folded weight round-trips through its derived weight" {
    const allocator = std.testing.allocator;
    const k: usize = 64;
    const n: usize = 32;

    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const w0 = try q8Weight(allocator, &sm, k, n, 1);
    const w1 = try q8Weight(allocator, &sm, k, n, 7);
    _ = try horizontal_matmul.concatColumns(allocator, &sm, cpu_target, &[_]TensorId{ w0, w1 });

    const replacement = try q8Weight(allocator, &sm, k, n, 99);
    try sm.writeDerivedSource(w0, replacement);

    // Reading w0 back now yields the replacement, and w1's region is untouched.
    const scratch = try q8Weight(allocator, &sm, k, n, 0);
    const bytes = (k / 32) * n * 34;
    const want = try allocator.alloc(u8, bytes);
    defer allocator.free(want);
    const got = try allocator.alloc(u8, bytes);
    defer allocator.free(got);

    try sm.readDerivedSource(w0, scratch);
    try sm.readToPackedQuant(replacement, want);
    try sm.readToPackedQuant(scratch, got);
    try std.testing.expectEqualSlices(u8, want, got);

    try sm.readDerivedSource(w1, scratch);
    try sm.readToPackedQuant(w1, want);
    try sm.readToPackedQuant(scratch, got);
    try std.testing.expectEqualSlices(u8, want, got);
}

// ---------------------------------------------------------------------------
// fuse_steps (asserted through `opt.defaults`)
// ---------------------------------------------------------------------------

test "add_norm: residual + rmsnorm is one step on device and a pair on cpu" {
    const allocator = std.testing.allocator;

    for ([_]types.BackendKind{ .webgpu, .cpu }) |target| {
        var sm = StorageManager.init(allocator);
        defer sm.deinit();

        var g = Graph.init(allocator);
        defer g.deinit();

        const M = 2;
        const N = 8;
        const res = try g.addInput(.f32, &.{ M, N });
        try g.bindExternal(res, try sm.createTiledTensor(.f32, &.{ M, N }, &.{ M, N }, .{}));
        const x = try g.addInput(.f32, &.{ M, N });
        try g.bindExternal(x, try sm.createTiledTensor(.f32, &.{ M, N }, &.{ M, N }, .{}));
        const gamma = try g.addInput(.f32, &.{N});
        try g.bindExternal(gamma, try sm.createTiledTensor(.f32, &.{N}, &.{N}, .{}));
        const beta = try g.addInput(.f32, &.{N});
        try g.bindExternal(beta, try sm.createTiledTensor(.f32, &.{N}, &.{N}, .{}));

        const normed = try g.addRMSNorm(x, gamma, beta, 1e-6, &.{N});
        try g.setOutputs(&.{try g.addElemwiseBinary(.add, res, normed)});

        var prog = try program.compileGraph(allocator, &g, &sm, .init(deviceFor(target), .{ .target_kind = target }));
        defer prog.deinit();

        var norms: usize = 0;
        var adds: usize = 0;
        var fused: usize = 0;
        for (prog.steps) |step| switch (step.op) {
            // The residual is a field, not a tag, so assert on the operand being there.
            .RMSNormTiled => |s| if (s.residual != null) {
                fused += 1;
            } else {
                norms += 1;
            },
            .ElemwiseBinaryTiled => adds += 1,
            else => {},
        };

        const device = target != .cpu;
        try std.testing.expectEqual(@as(usize, if (device) 0 else 1), norms);
        try std.testing.expectEqual(@as(usize, if (device) 0 else 1), adds);
        try std.testing.expectEqual(@as(usize, if (device) 1 else 0), fused);

        // The pass runs after placement, so the entry for the intermediate it killed has
        // to be gone: `materializePlacements` demands backing for everything listed.
        try prog.validatePlacements();
        for (prog.tensor_placements) |entry| {
            try std.testing.expect(try sm.tensorHasBacking(entry.id));
        }
    }
}

test "gate: unary + mul becomes one gate step on device, stays a pair on cpu" {
    const allocator = std.testing.allocator;

    for ([_]types.BackendKind{ .webgpu, .cpu }) |target| {
        for ([_]types.UnaryOp{ .gelu, .silu, .relu }) |act| {
            var sm = StorageManager.init(allocator);
            defer sm.deinit();

            var g = Graph.init(allocator);
            defer g.deinit();

            const N = 8;
            const x = try g.addInput(.f32, &.{ 2, N });
            try g.bindExternal(x, try sm.createTiledTensor(.f32, &.{ 2, N }, &.{ 2, N }, .{}));
            const y = try g.addInput(.f32, &.{ 2, N });
            try g.bindExternal(y, try sm.createTiledTensor(.f32, &.{ 2, N }, &.{ 2, N }, .{}));

            try g.setOutputs(&.{try g.addElemwiseBinary(.mul, try g.addUnary(act, x), y)});

            var prog = try program.compileGraph(allocator, &g, &sm, .init(deviceFor(target), .{ .target_kind = target }));
            defer prog.deinit();

            var unaries: usize = 0;
            var muls: usize = 0;
            var gates: usize = 0;
            for (prog.steps) |step| switch (step.op) {
                .UnaryTiled => unaries += 1,
                .ElemwiseBinaryTiled => |s| if (s.op == .gate) {
                    gates += 1;
                    // The activation has to survive, or every gate would be a GEGLU
                    // regardless of what the graph asked for.
                    try std.testing.expectEqual(act, s.act);
                } else {
                    muls += 1;
                },
                else => {},
            };

            const device = target != .cpu;
            try std.testing.expectEqual(@as(usize, if (device) 0 else 1), unaries);
            try std.testing.expectEqual(@as(usize, if (device) 0 else 1), muls);
            try std.testing.expectEqual(@as(usize, if (device) 1 else 0), gates);
            try prog.validatePlacements();
        }
    }
}

// The shape a real gated FFN has: rank-3 [B, L, ffn], authored the way `nn.GatedMLP`
// authors it. Nothing else in the suite would notice if this stopped fusing.
test "gate: a rank-3 gated FFN fuses" {
    const allocator = std.testing.allocator;

    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    var g = Graph.init(allocator);
    defer g.deinit();

    const shape = [_]usize{ 1, 4, 16 };
    const gate_v = try g.addInput(.f32, &shape);
    try g.bindExternal(gate_v, try sm.createTiledTensor(.f32, &shape, &shape, .{}));
    const up_v = try g.addInput(.f32, &shape);
    try g.bindExternal(up_v, try sm.createTiledTensor(.f32, &shape, &shape, .{}));

    try g.setOutputs(&.{try g.addElemwiseBinary(.mul, try g.addUnary(.silu, gate_v), up_v)});

    var prog = try program.compileGraph(allocator, &g, &sm, .init(.{ .kind = .gpu }, .{ .target_kind = .webgpu }));
    defer prog.deinit();

    var unaries: usize = 0;
    var gates: usize = 0;
    for (prog.steps) |step| switch (step.op) {
        .UnaryTiled => unaries += 1,
        .ElemwiseBinaryTiled => |s| if (s.op == .gate) {
            gates += 1;
            try std.testing.expectEqual(types.UnaryOp.silu, s.act);
        },
        else => {},
    };
    try std.testing.expectEqual(@as(usize, 1), gates);
    try std.testing.expectEqual(@as(usize, 0), unaries);
}

// A dtype the fused kernel does not take is an unfused pair that runs, not a rejected
// graph: `act(a) * b` is well defined in f16 and both halves have f16 kernels.
test "gate: an f16 gate compiles and runs as an unfused pair" {
    const allocator = std.testing.allocator;

    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    var g = Graph.init(allocator);
    defer g.deinit();

    const shape = [_]usize{ 2, 8 };
    const a = try g.addInput(.f16, &shape);
    try g.bindExternal(a, try sm.createTiledTensor(.f16, &shape, &shape, .{}));
    const b = try g.addInput(.f16, &shape);
    try g.bindExternal(b, try sm.createTiledTensor(.f16, &shape, &shape, .{}));

    try g.setOutputs(&.{try g.addElemwiseBinary(.mul, try g.addUnary(.silu, a), b)});

    var prog = try program.compileGraph(allocator, &g, &sm, .init(.{ .kind = .gpu }, .{ .target_kind = .webgpu }));
    defer prog.deinit();

    var unaries: usize = 0;
    var muls: usize = 0;
    var gates: usize = 0;
    for (prog.steps) |step| switch (step.op) {
        .UnaryTiled => unaries += 1,
        .ElemwiseBinaryTiled => |s| if (s.op == .gate) {
            gates += 1;
        } else if (s.op == .mul) {
            muls += 1;
        },
        else => {},
    };
    try std.testing.expectEqual(@as(usize, 0), gates);
    try std.testing.expectEqual(@as(usize, 1), unaries);
    try std.testing.expectEqual(@as(usize, 1), muls);
}

// ---------------------------------------------------------------------------
// alias_views
// ---------------------------------------------------------------------------

// A reshape between byte-identical layouts computes nothing, so the step goes and the
// destination borrows the source's backing.
test "alias_views: a no-op reshape is elided" {
    const allocator = std.testing.allocator;

    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    var g = Graph.init(allocator);
    defer g.deinit();

    const x = try g.addInput(.f32, &.{ 2, 8 });
    try g.bindExternal(x, try sm.createTiledTensor(.f32, &.{ 2, 8 }, &.{ 2, 8 }, .{}));
    const scaled = try g.addUnary(.relu, x);
    try g.setOutputs(&.{try g.addUnary(.relu, try g.addViewReshape(scaled, &.{ 1, 2, 8 }))});

    var with = try program.compileGraph(allocator, &g, &sm, (cpu_target.withPasses(.initOne(.alias_views))));
    defer with.deinit();
    try std.testing.expectEqual(@as(usize, 0), countStep(&with, .ReshapeScalar));

    var without = try program.compileGraph(allocator, &g, &sm, (cpu_target.withPasses(.empty)));
    defer without.deinit();
    try std.testing.expectEqual(@as(usize, 1), countStep(&without, .ReshapeScalar));
}

// `add(x, reshape(x))` must keep its copy: sharing a backing between two operands of one
// dispatch is what `workspace.validateAliases` refuses, and wgpu reports as conflicting
// buffer usages. Checking it here turns "compile fails" into "this view is not elided".
test "alias_views: a view bound alongside its source keeps its copy" {
    const allocator = std.testing.allocator;

    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    var g = Graph.init(allocator);
    defer g.deinit();

    const x = try g.addInput(.f32, &.{ 1, 2, 8 });
    try g.bindExternal(x, try sm.createTiledTensor(.f32, &.{ 1, 2, 8 }, &.{ 1, 2, 8 }, .{}));
    const same = try g.addUnary(.relu, x);
    const viewed = try g.addViewReshape(same, &.{ 1, 2, 8 });
    try g.setOutputs(&.{try g.addElemwiseBinary(.add, same, viewed)});

    var prog = try program.compileGraph(allocator, &g, &sm, (cpu_target.withPasses(.initOne(.alias_views))));
    defer prog.deinit();
    try std.testing.expectEqual(@as(usize, 1), countStep(&prog, .ReshapeScalar));
    try prog.validatePlacements();
}

// A program output has to hold its own bytes past the run, so a view that produces one
// is never elided however identical the layouts are.
test "alias_views: a view producing a program output keeps its copy" {
    const allocator = std.testing.allocator;

    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    var g = Graph.init(allocator);
    defer g.deinit();

    const x = try g.addInput(.f32, &.{ 2, 8 });
    try g.bindExternal(x, try sm.createTiledTensor(.f32, &.{ 2, 8 }, &.{ 2, 8 }, .{}));
    const relu = try g.addUnary(.relu, x);
    try g.setOutputs(&.{try g.addViewReshape(relu, &.{ 1, 2, 8 })});

    var prog = try program.compileGraph(allocator, &g, &sm, (cpu_target.withPasses(.initOne(.alias_views))));
    defer prog.deinit();
    try std.testing.expectEqual(@as(usize, 1), countStep(&prog, .ReshapeScalar));
}

// Equal tile offsets and lengths can still order values differently: `{2,4}` chunks a
// `[4,4]` row-major tensor, while `{4,2}` interleaves it.
test "alias_views: equal tile offsets do not imply equal byte order" {
    const allocator = std.testing.allocator;

    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const shape = [_]usize{ 4, 4 };
    const chunked = try sm.createTiledTensor(.f32, &shape, &[_]usize{ 2, 4 }, .{});
    const interleaved = try sm.createTiledTensor(.f32, &shape, &[_]usize{ 4, 2 }, .{});

    // Same dtype, same tile count, same offsets and lengths.
    const a = try sm.getConst(chunked);
    const b = try sm.getConst(interleaved);
    try std.testing.expectEqualSlices(usize, a.tile_lens, b.tile_lens);
    try std.testing.expectEqualSlices(usize, a.tile_offsets, b.tile_offsets);

    // Same logical values written to both...
    var vals: [16]f32 = undefined;
    for (&vals, 0..) |*v, i| v.* = @floatFromInt(i);
    try sm.writeFromPackedScalar(chunked, std.mem.sliceAsBytes(vals[0..]));
    try sm.writeFromPackedScalar(interleaved, std.mem.sliceAsBytes(vals[0..]));

    // ...land at different bytes, so sharing one backing would corrupt the other.
    try std.testing.expect(!std.mem.eql(u8, a.data, b.data));
    try std.testing.expect(!alias_views.layoutsIdentical(&sm, chunked, interleaved));
}

// ---------------------------------------------------------------------------
// Control-flow bodies are node lists like any other
// ---------------------------------------------------------------------------

// Rewrites visit loop bodies as closed node lists; replayed nodes cannot cross the body
// boundary.
test "horizontal_matmul: fuses inside a control-flow body" {
    const allocator = std.testing.allocator;
    const k: usize = 64;
    const n: usize = 32;

    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const seed_tid = try f32Tensor(&sm, &[_]usize{ 1, 1, n }, 2);
    const a_tid = try f32Tensor(&sm, &[_]usize{ 1, 1, k }, 3);
    const w0 = try q8Weight(allocator, &sm, k, n, 1);
    const w1 = try q8Weight(allocator, &sm, k, n, 4);

    var g = Graph.init(allocator);
    defer g.deinit();

    const seed = try g.addInput(.f32, &[_]usize{ 1, 1, n });
    try g.bindExternal(seed, @intCast(seed_tid));
    const a = try g.addInput(.f32, &[_]usize{ 1, 1, k });
    try g.bindExternal(a, @intCast(a_tid));
    const bw0 = try g.addInput(.q8_0, &[_]usize{ 1, k, n });
    try g.bindExternal(bw0, @intCast(w0));
    const bw1 = try g.addInput(.q8_0, &[_]usize{ 1, k, n });
    try g.bindExternal(bw1, @intCast(w1));

    // Both projections live INSIDE the body, off the same outer activation.
    try g.beginRegion();
    const p0 = try g.addMatMul(a, bw0, 1.0, 0.0);
    const p1 = try g.addMatMul(a, bw1, 1.0, 0.0);
    const sum = try g.addElemwiseBinary(.add, p0, p1);
    const body = try g.endRegion(&[_]ValueId{sum});

    try g.setOutputs(&[_]ValueId{try g.addLoop(seed, body, 2)});

    var prog = try program.compileGraph(allocator, &g, &sm, (cpu_target.withPasses(.initOne(.horizontal_matmul))));
    defer prog.deinit();

    // One wide MatMul plus a slice per member, all still in the body.
    try std.testing.expectEqual(@as(usize, 1), countRegionOp(&g, body, .MatMul));
    try std.testing.expectEqual(@as(usize, 2), countRegionOp(&g, body, .ViewSliceND));
    try std.testing.expectEqual(@as(usize, 0), countOp(&g, .MatMul));
}

// The step-level rules had the same blind spot for the same reason, and their hazard
// boundary was "touched inside any block". It is now the list itself, so a body's own
// steps fuse while a tensor another list can observe is still refused.
test "add_norm + gate: fuse inside a control-flow body" {
    const allocator = std.testing.allocator;
    const N: usize = 8;

    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    var g = Graph.init(allocator);
    defer g.deinit();

    const seed = try g.addInput(.f32, &[_]usize{ 1, N });
    try g.bindExternal(seed, try sm.createTiledTensor(.f32, &[_]usize{ 1, N }, &[_]usize{ 1, N }, .{}));
    const gamma = try g.addInput(.f32, &[_]usize{N});
    try g.bindExternal(gamma, try sm.createTiledTensor(.f32, &[_]usize{N}, &[_]usize{N}, .{}));
    const beta = try g.addInput(.f32, &[_]usize{N});
    try g.bindExternal(beta, try sm.createTiledTensor(.f32, &[_]usize{N}, &[_]usize{N}, .{}));

    try g.beginRegion();
    // Everything the rules match on has to be body-local: the carry itself is touched by
    // the enclosing schedule too, and a tensor two lists observe is refused by design.
    const y = try g.addUnary(.relu, seed);
    // y + rmsnorm(y)  ->  one norm carrying the residual.
    const normed = try g.addRMSNorm(y, gamma, beta, 1e-6, &[_]usize{N});
    const residual = try g.addElemwiseBinary(.add, y, normed);
    // gelu(residual) * y  ->  one gate step.
    const gated = try g.addElemwiseBinary(.mul, try g.addUnary(.gelu, residual), y);
    const body = try g.endRegion(&[_]ValueId{gated});

    try g.setOutputs(&[_]ValueId{try g.addLoop(seed, body, 2)});

    var prog = try program.compileGraph(allocator, &g, &sm, .init(.{ .kind = .gpu }, .{ .target_kind = .webgpu }));
    defer prog.deinit();

    var fused_norms: usize = 0;
    var gates: usize = 0;
    for (prog.blocks) |block| {
        for (block.steps) |step| switch (step.op) {
            .RMSNormTiled => |st| if (st.residual != null) {
                fused_norms += 1;
            },
            .ElemwiseBinaryTiled => |st| if (st.op == .gate) {
                gates += 1;
            },
            else => {},
        };
    }
    try std.testing.expectEqual(@as(usize, 1), fused_norms);
    try std.testing.expectEqual(@as(usize, 1), gates);
    // The gate consumed the gelu; the leading relu is nobody's producer here and stays.
    try std.testing.expectEqual(@as(usize, 1), countBlockStep(&prog, .UnaryTiled));
    try std.testing.expectEqual(@as(usize, 0), countBlockStep(&prog, .RMSNormTiled) - fused_norms);
    try prog.validatePlacements();
    for (prog.tensor_placements) |entry| {
        try std.testing.expect(try sm.tensorHasBacking(entry.id));
    }
}

// ---------------------------------------------------------------------------
// Grouping is exact
// ---------------------------------------------------------------------------

/// A batched q8_0 matmul-B weight `[batch, k, n]`, every slab holding the same values.
fn q8WeightBatched(allocator: std.mem.Allocator, sm: *StorageManager, batch: usize, k: usize, n: usize, seed: usize) !TensorId {
    const vals = try allocator.alloc(f32, k * n);
    defer allocator.free(vals);
    for (vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i + seed) % 17)) - 8)) * 0.1;
    const slab = try packQ8MatmulB(allocator, vals, k, n);
    defer allocator.free(slab);

    const whole = try allocator.alloc(u8, slab.len * batch);
    defer allocator.free(whole);
    for (0..batch) |b| @memcpy(whole[b * slab.len ..][0..slab.len], slab);

    const t = plan_mod.chooseMatMulTiles(cpu_tiles, plan_mod.matMulMHint(cpu_tiles), n, k, .q8_0);
    const tid = try sm.createTiledTensor(.q8_0, &[_]usize{ batch, k, n }, &[_]usize{ 1, t.tk, t.tn }, .{
        .tile_alignment = 64,
        .quant_axis = 1,
    });
    try sm.writeFromPackedQuant(tid, whole);
    return tid;
}

// Fusion compares stored batch layouts, not broadcast-compatible graph outputs, so one
// incompatible weight cannot invalidate an otherwise valid concat group.
test "horizontal_matmul: an unconcatenatable weight does not poison its group" {
    const allocator = std.testing.allocator;
    const k: usize = 64;
    const n: usize = 32;
    const batch: usize = 2;

    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const a_tid = try f32Tensor(&sm, &[_]usize{ batch, 1, k }, 3);
    const w0 = try q8WeightBatched(allocator, &sm, batch, k, n, 1);
    const w1 = try q8WeightBatched(allocator, &sm, batch, k, n, 5);
    const broadcast = try q8Weight(allocator, &sm, k, n, 9); // `[1, k, n]`

    var g = Graph.init(allocator);
    defer g.deinit();
    const a = try g.addInput(.f32, &[_]usize{ batch, 1, k });
    try g.bindExternal(a, @intCast(a_tid));
    const b0 = try g.addInput(.q8_0, &[_]usize{ batch, k, n });
    try g.bindExternal(b0, @intCast(w0));
    const b1 = try g.addInput(.q8_0, &[_]usize{ batch, k, n });
    try g.bindExternal(b1, @intCast(w1));
    const b2 = try g.addInput(.q8_0, &[_]usize{ 1, k, n });
    try g.bindExternal(b2, @intCast(broadcast));

    const p0 = try g.addMatMul(a, b0, 1.0, 0.0);
    const p1 = try g.addMatMul(a, b1, 1.0, 0.0);
    const p2 = try g.addMatMul(a, b2, 1.0, 0.0);
    try g.setOutputs(&[_]ValueId{
        try g.addElemwiseBinary(.add, try g.addElemwiseBinary(.add, p0, p1), p2),
    });

    var prog = try program.compileGraph(allocator, &g, &sm, (cpu_target.withPasses(.initOne(.horizontal_matmul))));
    defer prog.deinit();

    // w0+w1 became one wide matmul plus a slice each; the broadcast weight kept its own.
    try std.testing.expectEqual(@as(usize, 2), countOp(&g, .MatMul));
    try std.testing.expectEqual(@as(usize, 2), countOp(&g, .ViewSliceND));
    try std.testing.expect(sm.derivedLocate(w0) != null);
    try std.testing.expect(sm.derivedLocate(w1) != null);
    try std.testing.expect(sm.derivedLocate(broadcast) == null);
}

// ---------------------------------------------------------------------------
// Derived-weight lifecycle
// ---------------------------------------------------------------------------

// Folding leaves a source with metadata and no bytes, which is only sound while every
// program reads the fused weight instead. A program that reads the source itself gets the
// bytes back out of the fused weight rather than executing against released memory.
test "derived: a folded weight's own bytes come back on demand" {
    const allocator = std.testing.allocator;
    const k: usize = 64;
    const n: usize = 32;

    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const w0 = try quantWeight(allocator, &sm, .q8_0, k, n, 1);
    defer allocator.free(w0.bytes);
    const w1 = try quantWeight(allocator, &sm, .q8_0, k, n, 9);
    defer allocator.free(w1.bytes);
    _ = try horizontal_matmul.concatColumns(allocator, &sm, cpu_target, &[_]TensorId{ w0.tid, w1.tid });

    // What the model layer does once nothing reads w0 directly any more.
    try sm.releaseTensorData(w0.tid);
    try std.testing.expect(!try sm.tensorHasBacking(w0.tid));

    // A read at placement finds them in the derived weight even before that.
    const read_through = try allocator.alloc(u8, w0.bytes.len);
    defer allocator.free(read_through);
    try sm.readPackedAtPlacement(w0.tid, read_through);
    try std.testing.expectEqualSlices(u8, w0.bytes, read_through);

    try sm.unfoldTensor(w0.tid);
    try std.testing.expect(try sm.tensorHasBacking(w0.tid));

    const got = try allocator.alloc(u8, w0.bytes.len);
    defer allocator.free(got);
    try sm.readToPackedQuant(w0.tid, got);
    try std.testing.expectEqualSlices(u8, w0.bytes, got);

    // Idempotent: a source that still owns its bytes is left alone.
    try sm.unfoldTensor(w0.tid);
    try sm.readToPackedQuant(w0.tid, got);
    try std.testing.expectEqualSlices(u8, w0.bytes, got);
}

// A derived weight outlives the compile that built it, so no compile can collect it. The
// store does, off the same reference counts reclaim uses — which is what finally gives it
// an owner. Walk the whole state machine.
test "derived: a derivation is collected once it is redundant and unused" {
    const allocator = std.testing.allocator;

    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const w0 = try quantWeight(allocator, &sm, .q8_0, 64, 32, 1);
    defer allocator.free(w0.bytes);
    const w1 = try quantWeight(allocator, &sm, .q8_0, 64, 32, 9);
    defer allocator.free(w1.bytes);
    const d = try horizontal_matmul.concatColumns(allocator, &sm, cpu_target, &[_]TensorId{ w0.tid, w1.tid });

    // A fused program names the derived weight: it is canonical, so the sources do not
    // need their own copies.
    sm.retainTensor(d);
    sm.collectDerived();
    try std.testing.expect(try sm.tensorHasBacking(d));
    try std.testing.expect(!try sm.tensorHasBacking(w0.tid));
    try std.testing.expect(!try sm.tensorHasBacking(w1.tid));

    // That program goes. The derived weight is now unused but still the ONLY copy of
    // those bytes, so it has to stay.
    sm.releaseTensor(d);
    sm.collectDerived();
    try std.testing.expect(try sm.tensorHasBacking(d));
    try std.testing.expect(sm.derivedLocate(w0.tid) != null);

    // An unfused program names the sources instead, so they get unfolded...
    try sm.unfoldTensor(w0.tid);
    try sm.unfoldTensor(w1.tid);
    sm.retainTensor(w0.tid);
    sm.retainTensor(w1.tid);

    // ...and now nothing needs the derivation at all: it goes, bytes and record.
    sm.collectDerived();
    try std.testing.expect(!try sm.tensorHasBacking(d));
    try std.testing.expect(sm.derivedLocate(w0.tid) == null);

    // The weights survived the round trip intact.
    const got = try allocator.alloc(u8, w0.bytes.len);
    defer allocator.free(got);
    try sm.readToPackedQuant(w0.tid, got);
    try std.testing.expectEqualSlices(u8, w0.bytes, got);
    try sm.readToPackedQuant(w1.tid, got);
    try std.testing.expectEqualSlices(u8, w1.bytes, got);
}

// A tensor is resident on exactly one device, so two models on two GPUs cannot share one
// derived weight: handing the second the first's result would migrate it away. The device
// is part of the memo key for that reason, alongside the tiling.
test "derived: a result is keyed by device" {
    const allocator = std.testing.allocator;

    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const w0 = try quantWeight(allocator, &sm, .q8_0, 64, 32, 1);
    defer allocator.free(w0.bytes);
    const w1 = try quantWeight(allocator, &sm, .q8_0, 64, 32, 9);
    defer allocator.free(w1.bytes);
    const sources = [_]TensorId{ w0.tid, w1.tid };

    const host = try horizontal_matmul.concatColumns(allocator, &sm, .cpu(cpu_tiles), &sources);
    const gpu0 = try horizontal_matmul.concatColumns(allocator, &sm, .init(.{ .kind = .gpu, .index = 0 }, cpu_tiles), &sources);
    const gpu1 = try horizontal_matmul.concatColumns(allocator, &sm, .init(.{ .kind = .gpu, .index = 1 }, cpu_tiles), &sources);

    try std.testing.expect(host != gpu0);
    try std.testing.expect(gpu0 != gpu1);
    // Same key twice is still one result.
    try std.testing.expectEqual(gpu0, try horizontal_matmul.concatColumns(allocator, &sm, .init(.{ .kind = .gpu, .index = 0 }, cpu_tiles), &sources));

    // A swap reaches every copy, so the extra results are not stale.
    const replacement = try quantWeight(allocator, &sm, .q8_0, 64, 32, 77);
    defer allocator.free(replacement.bytes);
    try sm.writeDerivedSource(w0.tid, replacement.tid);
    const scratch = try quantWeight(allocator, &sm, .q8_0, 64, 32, 0);
    defer allocator.free(scratch.bytes);
    const got = try allocator.alloc(u8, replacement.bytes.len);
    defer allocator.free(got);
    try sm.readDerivedSource(w0.tid, scratch.tid);
    try sm.readToPackedQuant(scratch.tid, got);
    try std.testing.expectEqualSlices(u8, replacement.bytes, got);
}

// Reclaiming a fused-away weight turns on "does any live program still read it", and a
// `Context` shares one store between models — so the count has to live with the tensor,
// not in the model that happened to compile last.
test "derived: program references are counted on the store" {
    const allocator = std.testing.allocator;

    var sm = StorageManager.init(allocator);
    defer sm.deinit();
    const w = try quantWeight(allocator, &sm, .q8_0, 64, 32, 1);
    defer allocator.free(w.bytes);

    try std.testing.expectEqual(@as(u32, 0), sm.tensorProgramRefs(w.tid));
    sm.retainTensor(w.tid);
    sm.retainTensor(w.tid);
    try std.testing.expectEqual(@as(u32, 2), sm.tensorProgramRefs(w.tid));
    sm.releaseTensor(w.tid);
    try std.testing.expectEqual(@as(u32, 1), sm.tensorProgramRefs(w.tid));
    sm.releaseTensor(w.tid);
    try std.testing.expectEqual(@as(u32, 0), sm.tensorProgramRefs(w.tid));
    // Saturating, so an unbalanced release cannot make a live weight look reclaimable.
    sm.releaseTensor(w.tid);
    try std.testing.expectEqual(@as(u32, 0), sm.tensorProgramRefs(w.tid));
}
