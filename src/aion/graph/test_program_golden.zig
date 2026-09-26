// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Golden snapshot of compiled `ExecutableProgram` structure + tensor table.
//!
//! Purpose: a mechanical guarantee that backend-agnostic refactors (backend
//! selection, compile-`target` threading) do NOT change what the CPU compile
//! pipeline produces. We serialize, for a set of representative graphs, the
//! ordered step tags plus every tensor's dtype/shape,
//! and assert the dump is byte-identical to a checked-in golden string.
//!
//! If a change here is intentional, regenerate the goldens by flipping
//! `capture_mode` to true, running this file, and pasting the printed dumps.

const std = @import("std");

const graph_mod = @import("graph.zig");
const program = @import("program.zig");
const manager_mod = @import("../storage/manager.zig");
const types = @import("../backend/types.zig");

/// Set true to print fresh serializations to stderr instead of asserting.
const capture_mode = false;

const ValueId = graph_mod.ValueId;

fn app(list: *std.ArrayList(u8), a: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !void {
    const s = try std.fmt.allocPrint(a, fmt, args);
    defer a.free(s);
    try list.appendSlice(a, s);
}

fn appendDims(list: *std.ArrayList(u8), a: std.mem.Allocator, dims: []const usize) !void {
    try list.appendSlice(a, "[");
    for (dims, 0..) |d, i| {
        if (i != 0) try list.appendSlice(a, ",");
        try app(list, a, "{d}", .{d});
    }
    try list.appendSlice(a, "]");
}

/// Canonical, deterministic dump of a compiled program + the manager's tensor
/// table (every tensor created during compile).
fn serialize(a: std.mem.Allocator, prog: *const program.Program, mgr: *manager_mod.StorageManager) ![]u8 {
    var list: std.ArrayList(u8) = .empty;
    errdefer list.deinit(a);

    try app(&list, a, "steps={d}\n", .{prog.steps.len});
    for (prog.steps, 0..) |step, i| {
        try app(&list, a, "  [{d}] {s}\n", .{ i, @tagName(step.op) });
    }
    try app(&list, a, "blocks={d}\n", .{prog.blocks.len});
    for (prog.blocks, 0..) |block, bi| {
        try app(&list, a, " block {d} steps={d}\n", .{ bi, block.steps.len });
        for (block.steps, 0..) |step, i| {
            try app(&list, a, "   [{d}] {s}\n", .{ i, @tagName(step.op) });
        }
    }
    try app(&list, a, "outputs=[", .{});
    for (prog.outputs, 0..) |o, i| {
        if (i != 0) try list.appendSlice(a, ",");
        try app(&list, a, "{d}", .{o});
    }
    try list.appendSlice(a, "]\n");

    // Tensor table: iterate ids until out-of-range. Captures every tensor the
    // compiler created.
    try list.appendSlice(a, "tensors:\n");
    var id: manager_mod.TensorId = 0;
    while (true) : (id += 1) {
        const t = mgr.getConst(id) catch break;
        try app(&list, a, "  #{d} {s} rank={d} shape=", .{ id, @tagName(t.dtype), t.rank });
        try appendDims(&list, a, t.shape);
        try list.appendSlice(a, "\n");
    }

    return list.toOwnedSlice(a);
}

/// Build context handed to each graph builder: create + bind external input
/// tensors so the graph compiles.
const B = struct {
    g: *graph_mod.Graph,
    mgr: *manager_mod.StorageManager,

    fn input(self: *B, dtype: types.DType, shape: []const usize) !ValueId {
        const tid = try self.mgr.createTensor(dtype, shape, .{});
        const v = try self.g.addInput(dtype, shape);
        try self.g.bindExternal(v, tid);
        return v;
    }
};

/// Compile a graph built by `build`, serialize, and either print (capture) or
/// assert against `golden`.
fn checkGraph(
    name: []const u8,
    golden: []const u8,
    build: *const fn (b: *B) anyerror!ValueId,
) !void {
    const a = std.testing.allocator;

    var mgr = manager_mod.StorageManager.init(a);
    defer mgr.deinit();

    var g = graph_mod.Graph.init(a);
    defer g.deinit();

    var b: B = .{ .g = &g, .mgr = &mgr };
    const out = try build(&b);
    try g.setOutputs(&[_]ValueId{out});

    var prog = try program.compileGraph(a, &g, &mgr, .cpu());
    defer prog.deinit();

    const dump = try serialize(a, &prog, &mgr);
    defer a.free(dump);

    if (capture_mode) {
        std.debug.print("\n===BEGIN {s}===\n{s}===END {s}===\n", .{ name, dump, name });
        return error.CaptureMode; // force runner to surface the stderr dump
    } else {
        std.testing.expectEqualStrings(golden, dump) catch |e| {
            std.debug.print("\n[golden mismatch: {s}] actual dump:\n{s}\n", .{ name, dump });
            return e;
        };
    }
}

// ---- Representative graph builders ----

fn buildMatmulSquare(b: *B) anyerror!ValueId {
    const A = try b.input(.f32, &[_]usize{ 64, 128 });
    const W = try b.input(.f32, &[_]usize{ 128, 96 });
    return b.g.addMatMul(A, W, 1.0, 0.0);
}

fn buildMatvec(b: *B) anyerror!ValueId {
    // m == 1 wide-N matvec.
    const A = try b.input(.f32, &[_]usize{ 1, 256 });
    const W = try b.input(.f32, &[_]usize{ 256, 512 });
    return b.g.addMatMul(A, W, 1.0, 0.0);
}

fn buildSoftmax(b: *B) anyerror!ValueId {
    const X = try b.input(.f32, &[_]usize{ 32, 200 });
    return b.g.addSoftmax(X, -1);
}

fn buildRMSNorm(b: *B) anyerror!ValueId {
    const X = try b.input(.f32, &[_]usize{ 8, 128 });
    const gamma = try b.input(.f32, &[_]usize{128});
    const beta = try b.input(.f32, &[_]usize{128});
    return b.g.addRMSNorm(X, gamma, beta, 1e-5, &[_]usize{128});
}

fn buildElemwiseUnary(b: *B) anyerror!ValueId {
    const X = try b.input(.f32, &[_]usize{ 16, 64 });
    const Y = try b.input(.f32, &[_]usize{ 16, 64 });
    const s = try b.g.addElemwiseBinary(.add, X, Y);
    return b.g.addUnary(.silu, s);
}

fn buildAttention(b: *B) anyerror!ValueId {
    // Plain-sequence attention (no index operands): pins that the lowering emits
    // one Attention step.
    const q = try b.input(.f32, &[_]usize{ 1, 8, 2, 16 });
    const k = try b.input(.f32, &[_]usize{ 1, 8, 2, 16 });
    const v = try b.input(.f32, &[_]usize{ 1, 8, 2, 16 });
    return b.g.addAttention(q, k, v, null, null, 0.25, .full, 0.0);
}

fn buildDecodeChain(b: *B) anyerror!ValueId {
    // gather -> rmsnorm -> matmul -> softmax, a decode-shaped chain.
    const table = try b.input(.f32, &[_]usize{ 256, 128 });
    const idx = try b.input(.i32, &[_]usize{ 1, 1 });
    const emb = try b.g.addGather(table, idx, 0, 0); // [1,1,128]
    const x = try b.g.addViewReshape(emb, &[_]usize{ 1, 128 });
    const gamma = try b.input(.f32, &[_]usize{128});
    const beta = try b.input(.f32, &[_]usize{128});
    const normed = try b.g.addRMSNorm(x, gamma, beta, 1e-5, &[_]usize{128});
    const W = try b.input(.f32, &[_]usize{ 128, 256 });
    const logits = try b.g.addMatMul(normed, W, 1.0, 0.0);
    return b.g.addSoftmax(logits, -1);
}

test "golden: matmul square" {
    try checkGraph("matmul_square", golden_matmul_square, buildMatmulSquare);
}
test "golden: matvec" {
    try checkGraph("matvec", golden_matvec, buildMatvec);
}
test "golden: softmax" {
    try checkGraph("softmax", golden_softmax, buildSoftmax);
}
test "golden: rmsnorm" {
    try checkGraph("rmsnorm", golden_rmsnorm, buildRMSNorm);
}
test "golden: elemwise+unary" {
    try checkGraph("elemwise_unary", golden_elemwise_unary, buildElemwiseUnary);
}
test "golden: attention" {
    try checkGraph("attention", golden_attention, buildAttention);
}
test "golden: decode chain" {
    try checkGraph("decode_chain", golden_decode_chain, buildDecodeChain);
}

// ---- Golden strings (captured from the CPU compile pipeline) ----
// Regenerate by flipping `capture_mode` to true and running this file.

const golden_matmul_square =
    \\steps=1
    \\  [0] MatMul
    \\blocks=0
    \\outputs=[2]
    \\tensors:
    \\  #0 f32 rank=2 shape=[64,128]
    \\  #1 f32 rank=2 shape=[128,96]
    \\  #2 f32 rank=2 shape=[64,96]
    \\
;

const golden_matvec =
    \\steps=1
    \\  [0] MatMul
    \\blocks=0
    \\outputs=[2]
    \\tensors:
    \\  #0 f32 rank=2 shape=[1,256]
    \\  #1 f32 rank=2 shape=[256,512]
    \\  #2 f32 rank=2 shape=[1,512]
    \\
;

const golden_softmax =
    \\steps=1
    \\  [0] Softmax
    \\blocks=0
    \\outputs=[1]
    \\tensors:
    \\  #0 f32 rank=2 shape=[32,200]
    \\  #1 f32 rank=2 shape=[32,200]
    \\
;

const golden_rmsnorm =
    \\steps=1
    \\  [0] RMSNorm
    \\blocks=0
    \\outputs=[3]
    \\tensors:
    \\  #0 f32 rank=2 shape=[8,128]
    \\  #1 f32 rank=1 shape=[128]
    \\  #2 f32 rank=1 shape=[128]
    \\  #3 f32 rank=2 shape=[8,128]
    \\
;

const golden_elemwise_unary =
    \\steps=2
    \\  [0] ElemwiseBinary
    \\  [1] Unary
    \\blocks=0
    \\outputs=[3]
    \\tensors:
    \\  #0 f32 rank=2 shape=[16,64]
    \\  #1 f32 rank=2 shape=[16,64]
    \\  #2 f32 rank=2 shape=[16,64]
    \\  #3 f32 rank=2 shape=[16,64]
    \\
;

const golden_attention =
    \\steps=1
    \\  [0] Attention
    \\blocks=0
    \\outputs=[3]
    \\tensors:
    \\  #0 f32 rank=4 shape=[1,8,2,16]
    \\  #1 f32 rank=4 shape=[1,8,2,16]
    \\  #2 f32 rank=4 shape=[1,8,2,16]
    \\  #3 f32 rank=4 shape=[1,8,2,16]
    \\
;

// No ReshapeScalar step: `[1,1,128]` -> `[1,128]` is the same contiguous run of
// floats, so `alias_views.elideNoopViews` drops the copy and #6 borrows #5's backing. Both
// tensors still exist below — the destination keeps its own shape metadata, which is
// the whole reason the pass aliases instead of rewriting operands to the source id.
const golden_decode_chain =
    \\steps=4
    \\  [0] GatherRows
    \\  [1] RMSNorm
    \\  [2] MatMul
    \\  [3] Softmax
    \\blocks=0
    \\outputs=[9]
    \\tensors:
    \\  #0 f32 rank=2 shape=[256,128]
    \\  #1 i32 rank=2 shape=[1,1]
    \\  #2 f32 rank=1 shape=[128]
    \\  #3 f32 rank=1 shape=[128]
    \\  #4 f32 rank=2 shape=[128,256]
    \\  #5 f32 rank=3 shape=[1,1,128]
    \\  #6 f32 rank=2 shape=[1,128]
    \\  #7 f32 rank=2 shape=[1,128]
    \\  #8 f32 rank=2 shape=[1,256]
    \\  #9 f32 rank=2 shape=[1,256]
    \\
;
