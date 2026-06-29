// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Tests for horizontal MatMul fusion: the quant weight concat (+cache) and the
//! end-to-end fusion-on vs fusion-off parity / graph-rewrite behavior.

const std = @import("std");

const types = @import("../../backend/types.zig");
const manager_mod = @import("../../storage/manager.zig");
const graph_mod = @import("../graph.zig");
const plan_mod = @import("../plan.zig");
const program = @import("../program.zig");
const cpu_backend_mod = @import("../../backend/cpu/cpu_backend.zig");
const backend_mod = @import("../../backend/backend.zig");
const fuse = @import("fuse_horizontal_matmul.zig");

const TensorId = manager_mod.TensorId;
const ValueId = graph_mod.ValueId;

fn asF32(buf: []u8) []align(1) f32 {
    const ptr: [*]align(1) f32 = @ptrCast(buf.ptr);
    return ptr[0 .. buf.len / @sizeOf(f32)];
}

// ---------------------------------------------------------------------------
// concatQuantMatmulB
// ---------------------------------------------------------------------------

/// Build a `[1, K, N]` quant tensor filled with a deterministic per-byte pattern
/// (the concat is byte-level, so the exact values don't matter), returning its id
/// and the packed bytes that were written.
fn makeQuantWeight(
    allocator: std.mem.Allocator,
    sm: *manager_mod.StorageManager,
    dtype: types.DType,
    k: usize,
    n: usize,
    seed: u8,
) !struct { tid: TensorId, bytes: []u8 } {
    const bb = dtype.info().block_bytes;
    const blk = dtype.info().block_elems;
    const bytes = (k / blk) * n * bb;
    const buf = try allocator.alloc(u8, bytes);
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
    const bb = dtype.info().block_bytes;
    const blk = dtype.info().block_elems;
    const kb = k / blk;

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const w0 = try makeQuantWeight(allocator, &sm, dtype, k, n0, 1);
    defer allocator.free(w0.bytes);
    const w1 = try makeQuantWeight(allocator, &sm, dtype, k, n1, 200);
    defer allocator.free(w1.bytes);

    const policy: plan_mod.TilePolicy = .{ .tile_alignment = 64 };
    const out = try fuse.concatQuantMatmulB(allocator, &sm, policy, &[_]TensorId{ w0.tid, w1.tid });

    // Expected: per block-row, w0's n0 blocks followed by w1's n1 blocks.
    const sum_n = n0 + n1;
    const expected = try allocator.alloc(u8, kb * sum_n * bb);
    defer allocator.free(expected);
    var r: usize = 0;
    while (r < kb) : (r += 1) {
        @memcpy(expected[(r * sum_n) * bb ..][0 .. n0 * bb], w0.bytes[(r * n0) * bb ..][0 .. n0 * bb]);
        @memcpy(expected[(r * sum_n + n0) * bb ..][0 .. n1 * bb], w1.bytes[(r * n1) * bb ..][0 .. n1 * bb]);
    }

    const got = try allocator.alloc(u8, kb * sum_n * bb);
    defer allocator.free(got);
    try sm.readToPackedQuant(out, got);
    try std.testing.expectEqualSlices(u8, expected, got);

    // Cache: same ordered sources return the same tensor id (no re-alloc/leak).
    const again = try fuse.concatQuantMatmulB(allocator, &sm, policy, &[_]TensorId{ w0.tid, w1.tid });
    try std.testing.expectEqual(out, again);
}

test "fuse: concatQuantMatmulB q8_0 byte-identity + cache" {
    try concatByteIdentity(.q8_0);
}

test "fuse: concatQuantMatmulB q4_0 byte-identity + cache" {
    try concatByteIdentity(.q4_0);
}

// ---------------------------------------------------------------------------
// End-to-end fusion parity + rewrite
// ---------------------------------------------------------------------------

/// Quantize a `[K, N]` f32 weight (row-major) to q8_0 matmul-B packed bytes:
/// blocks run along K (quant axis), block (kb, j) holds 32 K-values of column j.
fn packQ8MatmulB(allocator: std.mem.Allocator, vals: []const f32, k: usize, n: usize) ![]u8 {
    const kb = k / 32;
    const buf = try allocator.alloc(u8, kb * n * 34);
    for (0..kb) |b| {
        for (0..n) |j| {
            var absmax: f32 = 0;
            for (0..32) |t| absmax = @max(absmax, @abs(vals[(b * 32 + t) * n + j]));
            const scale: f32 = if (absmax == 0) 1 else absmax / 127.0;
            const inv: f32 = if (absmax == 0) 0 else 1.0 / scale;
            const scale_f16: f16 = @floatCast(scale);
            const off = (b * n + j) * 34;
            std.mem.writeInt(u16, buf[off .. off + 2][0..2], @bitCast(scale_f16), .little);
            for (0..32) |t| {
                var q: i32 = @intFromFloat(@round(vals[(b * 32 + t) * n + j] * inv));
                q = @max(@as(i32, -128), @min(@as(i32, 127), q));
                buf[off + 2 + t] = @bitCast(@as(i8, @intCast(q)));
            }
        }
    }
    return buf;
}

fn makeQ8Weight(allocator: std.mem.Allocator, sm: *manager_mod.StorageManager, policy: plan_mod.TilePolicy, k: usize, n: usize, seed: usize) !TensorId {
    const vals = try allocator.alloc(f32, k * n);
    defer allocator.free(vals);
    for (vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i + seed) % 17)) - 8)) * 0.1;
    const buf = try packQ8MatmulB(allocator, vals, k, n);
    defer allocator.free(buf);

    // Tile exactly as the MatMul lowering tiles a quant B, so no retile is needed.
    const m_hint: usize = @max(@as(usize, 1), policy.base_square_2d);
    const t = plan_mod.chooseMatMulTiles(policy, m_hint, n, k, .q8_0);
    const tid = try sm.createTiledTensor(.q8_0, &[_]usize{ 1, k, n }, &[_]usize{ 1, t.tk, t.tn }, .{ .tile_alignment = 64, .quant_axis = 1 });
    try sm.writeFromPackedQuant(tid, buf);
    return tid;
}

test "fuse: overwriteFusedColumns writes a source's region through to the fused weight" {
    const allocator = std.testing.allocator;
    const k: usize = 64;
    const n0: usize = 32;
    const n1: usize = 64;
    const policy: plan_mod.TilePolicy = .{ .tile_alignment = 64 };

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const w0 = try makeQ8Weight(allocator, &sm, policy, k, n0, 1);
    const w1 = try makeQ8Weight(allocator, &sm, policy, k, n1, 7);
    const fused = try fuse.concatQuantMatmulB(allocator, &sm, policy, &[_]TensorId{ w0, w1 });

    // A fresh replacement for w0, then write it through to the fused weight.
    const new0 = try makeQ8Weight(allocator, &sm, policy, k, n0, 99);
    const ref = sm.findDerivedWeightBySource(w0) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(usize, 0), ref.index);
    try fuse.overwriteFusedColumns(allocator, &sm, ref, new0);

    // The fused weight must now equal concat(new0, w1): columns [0:n0) == new0,
    // columns [n0:] == w1, block-for-block.
    const bb: usize = 34;
    const kb = k / 32;
    const new0_p = try allocator.alloc(u8, kb * n0 * bb);
    defer allocator.free(new0_p);
    const w1_p = try allocator.alloc(u8, kb * n1 * bb);
    defer allocator.free(w1_p);
    const fused_p = try allocator.alloc(u8, kb * (n0 + n1) * bb);
    defer allocator.free(fused_p);
    try sm.readToPackedQuant(new0, new0_p);
    try sm.readToPackedQuant(w1, w1_p);
    try sm.readToPackedQuant(fused, fused_p);

    var r: usize = 0;
    while (r < kb) : (r += 1) {
        try std.testing.expectEqualSlices(u8, new0_p[(r * n0) * bb ..][0 .. n0 * bb], fused_p[(r * (n0 + n1)) * bb ..][0 .. n0 * bb]);
        try std.testing.expectEqualSlices(u8, w1_p[(r * n1) * bb ..][0 .. n1 * bb], fused_p[(r * (n0 + n1) + n0) * bb ..][0 .. n1 * bb]);
    }
}

const Dims = struct { m: usize = 2, k: usize = 64, ns: [3]usize = .{ 32, 32, 64 } };

/// out values: o0, relu(o1), o2 — exercises both output-remap and consumer-remap.
fn buildProjGraph(g: *graph_mod.Graph, a_tid: TensorId, w: [3]TensorId, d: Dims) !void {
    const a_in = try g.addInput(.f32, &[_]usize{ 1, d.m, d.k });
    try g.bindExternal(a_in, @intCast(a_tid));
    var outs: [3]ValueId = undefined;
    for (0..3) |i| {
        const w_in = try g.addInput(.q8_0, &[_]usize{ 1, d.k, d.ns[i] });
        try g.bindExternal(w_in, @intCast(w[i]));
        outs[i] = try g.addMatMul(a_in, w_in, 1.0, 0.0);
    }
    const r1 = try g.addRelu(outs[1]);
    try g.setOutputs(&[_]ValueId{ outs[0], r1, outs[2] });
}

fn countOp(g: *const graph_mod.Graph, tag: std.meta.Tag(graph_mod.Op)) usize {
    var c: usize = 0;
    for (g.nodes.items) |node| {
        if (std.meta.activeTag(node.op) == tag) c += 1;
    }
    return c;
}

test "fuse: horizontal matmul fusion matches unfused output" {
    const allocator = std.testing.allocator;
    const d = Dims{};
    const policy: plan_mod.TilePolicy = .{ .tile_alignment = 64 };

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    // Shared activation A.
    const a_vals = try allocator.alloc(f32, d.m * d.k);
    defer allocator.free(a_vals);
    for (a_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) * 0.05;
    const a_tid = try sm.createTiledTensor(.f32, &[_]usize{ 1, d.m, d.k }, &[_]usize{ 1, d.m, d.k }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(a_tid, std.mem.sliceAsBytes(a_vals));

    const w: [3]TensorId = .{
        try makeQ8Weight(allocator, &sm, policy, d.k, d.ns[0], 0),
        try makeQ8Weight(allocator, &sm, policy, d.k, d.ns[1], 5),
        try makeQ8Weight(allocator, &sm, policy, d.k, d.ns[2], 11),
    };

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: backend_mod.Backend = cpu.backend();

    const out_sizes = [_]usize{ d.m * d.ns[0], d.m * d.ns[1], d.m * d.ns[2] };

    // Unfused reference.
    var off_bufs: [3][]u8 = undefined;
    {
        var g = graph_mod.Graph.init(allocator);
        defer g.deinit();
        try buildProjGraph(&g, a_tid, w, d);
        var prog = try program.compileGraphOpt(allocator, &g, &sm, policy, .{ .fuse_horizontal_matmul = false });
        defer prog.deinit();
        try backend.executeProgram(&prog, sm.tensorStore());

        try std.testing.expectEqual(@as(usize, 3), countOp(&g, .MatMul));
        for (0..3) |i| {
            off_bufs[i] = try allocator.alloc(u8, out_sizes[i] * 4);
            try sm.readToPackedScalar(prog.outputs[i], off_bufs[i]);
        }
    }
    defer for (off_bufs) |b| allocator.free(b);

    // Fused.
    {
        var g = graph_mod.Graph.init(allocator);
        defer g.deinit();
        try buildProjGraph(&g, a_tid, w, d);
        var prog = try program.compileGraphOpt(allocator, &g, &sm, policy, .{});
        defer prog.deinit();
        try backend.executeProgram(&prog, sm.tensorStore());

        // The three projections collapsed to one wide matmul + three slices.
        try std.testing.expectEqual(@as(usize, 1), countOp(&g, .MatMul));
        try std.testing.expectEqual(@as(usize, 3), countOp(&g, .ViewSliceND));

        for (0..3) |i| {
            const got = try allocator.alloc(u8, out_sizes[i] * 4);
            defer allocator.free(got);
            try sm.readToPackedScalar(prog.outputs[i], got);
            const gv = asF32(got);
            const rv = asF32(off_bufs[i]);
            var max_abs: f32 = 0;
            for (gv, rv) |x, y| max_abs = @max(max_abs, @abs(x - y));
            try std.testing.expect(max_abs <= 1e-4);
        }
    }

    // The pass itself is pure — it does not free source buffers (reclamation is a
    // model-layer lifecycle concern). The sources remain intact here.
    for (w) |wt| {
        const t = try sm.getConst(wt);
        try std.testing.expect(t.data.len != 0);
    }

    // Recompile on the same store still works: the concat cache hits and the result
    // is unchanged.
    {
        var g = graph_mod.Graph.init(allocator);
        defer g.deinit();
        try buildProjGraph(&g, a_tid, w, d);
        var prog = try program.compileGraphOpt(allocator, &g, &sm, policy, .{});
        defer prog.deinit();
        try backend.executeProgram(&prog, sm.tensorStore());
        try std.testing.expectEqual(@as(usize, 1), countOp(&g, .MatMul));
        for (0..3) |i| {
            const got = try allocator.alloc(u8, out_sizes[i] * 4);
            defer allocator.free(got);
            try sm.readToPackedScalar(prog.outputs[i], got);
            const gv = asF32(got);
            const rv = asF32(off_bufs[i]);
            var max_abs: f32 = 0;
            for (gv, rv) |x, y| max_abs = @max(max_abs, @abs(x - y));
            try std.testing.expect(max_abs <= 1e-4);
        }
    }
}
