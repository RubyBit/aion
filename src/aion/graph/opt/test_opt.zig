// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Tests optional graph and step rewrite passes with explicit policies.
//! Step fusion also tests `compileGraph` to pin target defaults.

const std = @import("std");

const backend_mod = @import("../../backend/backend.zig");
const cpu_backend_mod = @import("../../backend/cpu/cpu_backend.zig");
const matmul_q_i8 = @import("../../backend/cpu/kernels/matmul_q_i8.zig");
const derived = @import("../../storage/derived.zig");
const manager_mod = @import("../../storage/manager.zig");
const types = @import("../../backend/types.zig");
const graph_mod = @import("../graph.zig");
const opt = @import("../opt.zig");
const program = @import("../program.zig");

const alias_views = @import("alias_views.zig");

const Graph = graph_mod.Graph;
const Policy = opt.Policy;
const StorageManager = manager_mod.StorageManager;
const TensorId = manager_mod.TensorId;
const ValueId = graph_mod.ValueId;

const cpu_target: program.Target = .cpu();
const weight_layout = @import("weight_layout.zig");

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
    const tid = try sm.createTensor(.f32, shape, .{});
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
// q8 matmul-B packing
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

// ---------------------------------------------------------------------------
// fuse_steps (asserted through `opt.defaults`)
// ---------------------------------------------------------------------------

test "add_norm: residual + rmsnorm is one step on every target" {
    const allocator = std.testing.allocator;

    for ([_]types.BackendKind{ .webgpu, .cpu }) |target| {
        var sm = StorageManager.init(allocator);
        defer sm.deinit();

        var g = Graph.init(allocator);
        defer g.deinit();

        const M = 2;
        const N = 8;
        const res = try g.addInput(.f32, &.{ M, N });
        try g.bindExternal(res, try sm.createTensor(.f32, &.{ M, N }, .{}));
        const x = try g.addInput(.f32, &.{ M, N });
        try g.bindExternal(x, try sm.createTensor(.f32, &.{ M, N }, .{}));
        const gamma = try g.addInput(.f32, &.{N});
        try g.bindExternal(gamma, try sm.createTensor(.f32, &.{N}, .{}));
        const beta = try g.addInput(.f32, &.{N});
        try g.bindExternal(beta, try sm.createTensor(.f32, &.{N}, .{}));

        const normed = try g.addRMSNorm(x, gamma, beta, 1e-6, &.{N});
        try g.setOutputs(&.{try g.addElemwiseBinary(.add, res, normed)});

        var prog = try program.compileGraph(allocator, &g, &sm, .init(deviceFor(target), .row_major));
        defer prog.deinit();

        var norms: usize = 0;
        var adds: usize = 0;
        var fused: usize = 0;
        for (prog.steps) |step| switch (step.op) {
            // The residual is a field, not a tag, so assert on the operand being there.
            .RMSNorm => |s| if (s.residual != null) {
                fused += 1;
            } else {
                norms += 1;
            },
            .ElemwiseBinary => adds += 1,
            else => {},
        };

        try std.testing.expectEqual(@as(usize, 0), norms);
        try std.testing.expectEqual(@as(usize, 0), adds);
        try std.testing.expectEqual(@as(usize, 1), fused);

        // The pass runs after placement, so the entry for the intermediate it killed has
        // to be gone: `materializePlacements` demands backing for everything listed.
        try prog.validatePlacements();
        for (prog.tensor_placements) |entry| {
            try std.testing.expect(try sm.tensorHasBacking(entry.id));
        }
    }
}

test "gate: unary + mul becomes one gate step on every target" {
    const allocator = std.testing.allocator;

    for ([_]types.BackendKind{ .webgpu, .cpu }) |target| {
        for ([_]types.UnaryOp{ .gelu, .silu, .relu }) |act| {
            var sm = StorageManager.init(allocator);
            defer sm.deinit();

            var g = Graph.init(allocator);
            defer g.deinit();

            const N = 8;
            const x = try g.addInput(.f32, &.{ 2, N });
            try g.bindExternal(x, try sm.createTensor(.f32, &.{ 2, N }, .{}));
            const y = try g.addInput(.f32, &.{ 2, N });
            try g.bindExternal(y, try sm.createTensor(.f32, &.{ 2, N }, .{}));

            try g.setOutputs(&.{try g.addElemwiseBinary(.mul, try g.addUnary(act, x), y)});

            var prog = try program.compileGraph(allocator, &g, &sm, .init(deviceFor(target), .row_major));
            defer prog.deinit();

            var unaries: usize = 0;
            var muls: usize = 0;
            var gates: usize = 0;
            for (prog.steps) |step| switch (step.op) {
                .Unary => unaries += 1,
                .ElemwiseBinary => |s| if (s.op == .gate) {
                    gates += 1;
                    // The activation has to survive, or every gate would be a GEGLU
                    // regardless of what the graph asked for.
                    try std.testing.expectEqual(act, s.act);
                } else {
                    muls += 1;
                },
                else => {},
            };

            try std.testing.expectEqual(@as(usize, 0), unaries);
            try std.testing.expectEqual(@as(usize, 0), muls);
            try std.testing.expectEqual(@as(usize, 1), gates);
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
    try g.bindExternal(gate_v, try sm.createTensor(.f32, &shape, .{}));
    const up_v = try g.addInput(.f32, &shape);
    try g.bindExternal(up_v, try sm.createTensor(.f32, &shape, .{}));

    try g.setOutputs(&.{try g.addElemwiseBinary(.mul, try g.addUnary(.silu, gate_v), up_v)});

    var prog = try program.compileGraph(allocator, &g, &sm, .init(.{ .kind = .gpu }, .row_major));
    defer prog.deinit();

    var unaries: usize = 0;
    var gates: usize = 0;
    for (prog.steps) |step| switch (step.op) {
        .Unary => unaries += 1,
        .ElemwiseBinary => |s| if (s.op == .gate) {
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
    try g.bindExternal(a, try sm.createTensor(.f16, &shape, .{}));
    const b = try g.addInput(.f16, &shape);
    try g.bindExternal(b, try sm.createTensor(.f16, &shape, .{}));

    try g.setOutputs(&.{try g.addElemwiseBinary(.mul, try g.addUnary(.silu, a), b)});

    var prog = try program.compileGraph(allocator, &g, &sm, .init(.{ .kind = .gpu }, .row_major));
    defer prog.deinit();

    var unaries: usize = 0;
    var muls: usize = 0;
    var gates: usize = 0;
    for (prog.steps) |step| switch (step.op) {
        .Unary => unaries += 1,
        .ElemwiseBinary => |s| if (s.op == .gate) {
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
    try g.bindExternal(x, try sm.createTensor(.f32, &.{ 2, 8 }, .{}));
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
    try g.bindExternal(x, try sm.createTensor(.f32, &.{ 1, 2, 8 }, .{}));
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
    try g.bindExternal(x, try sm.createTensor(.f32, &.{ 2, 8 }, .{}));
    const relu = try g.addUnary(.relu, x);
    try g.setOutputs(&.{try g.addViewReshape(relu, &.{ 1, 2, 8 })});

    var prog = try program.compileGraph(allocator, &g, &sm, (cpu_target.withPasses(.initOne(.alias_views))));
    defer prog.deinit();
    try std.testing.expectEqual(@as(usize, 1), countStep(&prog, .ReshapeScalar));
}

// ---------------------------------------------------------------------------
// Control-flow bodies are node lists like any other
// ---------------------------------------------------------------------------

// Step rules treat a body as its own list: a body's own steps fuse while a tensor
// another list can observe is still refused.
test "add_norm + gate: fuse inside a control-flow body" {
    const allocator = std.testing.allocator;
    const N: usize = 8;

    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    var g = Graph.init(allocator);
    defer g.deinit();

    const seed = try g.addInput(.f32, &[_]usize{ 1, N });
    try g.bindExternal(seed, try sm.createTensor(.f32, &[_]usize{ 1, N }, .{}));
    const gamma = try g.addInput(.f32, &[_]usize{N});
    try g.bindExternal(gamma, try sm.createTensor(.f32, &[_]usize{N}, .{}));
    const beta = try g.addInput(.f32, &[_]usize{N});
    try g.bindExternal(beta, try sm.createTensor(.f32, &[_]usize{N}, .{}));

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

    var prog = try program.compileGraph(allocator, &g, &sm, .init(.{ .kind = .gpu }, .row_major));
    defer prog.deinit();

    var fused_norms: usize = 0;
    var gates: usize = 0;
    for (prog.blocks) |block| {
        for (block.steps) |step| switch (step.op) {
            .RMSNorm => |st| if (st.residual != null) {
                fused_norms += 1;
            },
            .ElemwiseBinary => |st| if (st.op == .gate) {
                gates += 1;
            },
            else => {},
        };
    }
    try std.testing.expectEqual(@as(usize, 1), fused_norms);
    try std.testing.expectEqual(@as(usize, 1), gates);
    // The gate consumed the gelu; the leading relu is nobody's producer here and stays.
    try std.testing.expectEqual(@as(usize, 1), countBlockStep(&prog, .Unary));
    try std.testing.expectEqual(@as(usize, 0), countBlockStep(&prog, .RMSNorm) - fused_norms);
    try prog.validatePlacements();
    for (prog.tensor_placements) |entry| {
        try std.testing.expect(try sm.tensorHasBacking(entry.id));
    }
}

// ---------------------------------------------------------------------------
// Derived-weight lifecycle
// ---------------------------------------------------------------------------

/// A q8 matmul-B the layout pass can derive from, plus its packed bytes.
fn derivable(allocator: std.mem.Allocator, sm: *StorageManager, seed: usize) !struct { tid: TensorId, bytes: []u8 } {
    const tid = try q8MatmulB(allocator, sm, 64, 32, seed);
    const bytes = try allocator.alloc(u8, (64 / 32) * 32 * 34);
    errdefer allocator.free(bytes);
    try sm.readToPackedQuant(tid, bytes);
    return .{ .tid = tid, .bytes = bytes };
}

// Stacking is refused rather than silently mis-resolved: a chain would need composed
// views, so a swap of the original would land in a tensor nothing reads.
test "derived: deriving from a derived weight is refused" {
    const allocator = std.testing.allocator;
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const w = try derivable(allocator, &sm, 1);
    defer allocator.free(w.bytes);
    const laid = try weight_layout.relayout(allocator, &sm, cpu_target, w.tid);
    const out = try sm.createTensor(.q8_0, &.{ 32, 64 }, .{ .quant_axis = 1 });
    const view: derived.View = .{ .blocks = 2, .cols = 32, .block_bytes = 34, .transposed = false, .order = .row_major };
    try std.testing.expectError(
        error.InvalidArgument,
        sm.derivedRecord(cpu_target.device, out, laid, view),
    );
}

// Deriving must not hold a weight twice: a source no program reads gives its bytes to
// the result at once, while one a live program still names keeps them.
test "derived: deriving releases a source no program reads" {
    const allocator = std.testing.allocator;
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const idle = try derivable(allocator, &sm, 1);
    defer allocator.free(idle.bytes);
    _ = try weight_layout.relayout(allocator, &sm, cpu_target, idle.tid);
    try std.testing.expect(!try sm.tensorHasBacking(idle.tid));

    const read = try derivable(allocator, &sm, 2);
    defer allocator.free(read.bytes);
    sm.retainTensor(read.tid);
    _ = try weight_layout.relayout(allocator, &sm, cpu_target, read.tid);
    try std.testing.expect(try sm.tensorHasBacking(read.tid));
}

// Folding leaves a source with metadata and no bytes, which is only sound while every
// program reads the derived weight instead. A program that reads the source itself gets
// the bytes back out of the derived weight rather than executing against released memory.
test "derived: a folded weight's own bytes come back on demand" {
    const allocator = std.testing.allocator;
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const w = try derivable(allocator, &sm, 1);
    defer allocator.free(w.bytes);
    _ = try weight_layout.relayout(allocator, &sm, cpu_target, w.tid);

    // What the model layer does once nothing reads w directly any more.
    try sm.releaseTensorData(w.tid);
    try std.testing.expect(!try sm.tensorHasBacking(w.tid));

    // A read at placement finds the bytes in the derived weight even before that.
    const read_through = try allocator.alloc(u8, w.bytes.len);
    defer allocator.free(read_through);
    try sm.readPackedAtPlacement(w.tid, read_through);
    try std.testing.expectEqualSlices(u8, w.bytes, read_through);

    try sm.unfoldTensor(w.tid);
    try std.testing.expect(try sm.tensorHasBacking(w.tid));

    const got = try allocator.alloc(u8, w.bytes.len);
    defer allocator.free(got);
    try sm.readToPackedQuant(w.tid, got);
    try std.testing.expectEqualSlices(u8, w.bytes, got);

    // Idempotent: a source that still owns its bytes is left alone.
    try sm.unfoldTensor(w.tid);
    try sm.readToPackedQuant(w.tid, got);
    try std.testing.expectEqualSlices(u8, w.bytes, got);
}

// A derived weight outlives the compile that built it, so no compile can collect it. The
// store does, off the same reference counts reclaim uses — which is what finally gives it
// an owner. Walk the whole state machine.
test "derived: a derivation is collected once it is redundant and unused" {
    const allocator = std.testing.allocator;
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const w = try derivable(allocator, &sm, 1);
    defer allocator.free(w.bytes);
    const d = try weight_layout.relayout(allocator, &sm, cpu_target, w.tid);

    // A re-laid program names the derived weight: it is canonical, so the source does
    // not need its own copy.
    sm.retainTensor(d);
    sm.collectDerived();
    try std.testing.expect(try sm.tensorHasBacking(d));
    try std.testing.expect(!try sm.tensorHasBacking(w.tid));

    // That program goes. The derived weight is now unused but still the ONLY copy of
    // those bytes, so it has to stay.
    sm.releaseTensor(d);
    sm.collectDerived();
    try std.testing.expect(try sm.tensorHasBacking(d));
    try std.testing.expect(sm.derivedLocate(w.tid) != null);

    // A plain program names the source instead, so it gets unfolded...
    try sm.unfoldTensor(w.tid);
    sm.retainTensor(w.tid);

    // ...and now nothing needs the derivation at all: it goes, bytes and record.
    sm.collectDerived();
    try std.testing.expect(!try sm.tensorHasBacking(d));
    try std.testing.expect(sm.derivedLocate(w.tid) == null);

    // The weight survived the round trip intact.
    const got = try allocator.alloc(u8, w.bytes.len);
    defer allocator.free(got);
    try sm.readToPackedQuant(w.tid, got);
    try std.testing.expectEqualSlices(u8, w.bytes, got);
}

// A tensor is resident on exactly one device, so two models on two GPUs cannot share one
// derived weight: handing the second the first's result would migrate it away. The device
// is part of the memo key for that reason, alongside the block order.
test "derived: a result is keyed by device" {
    const allocator = std.testing.allocator;
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const w = try derivable(allocator, &sm, 1);
    defer allocator.free(w.bytes);

    const host = try weight_layout.relayout(allocator, &sm, .cpu(), w.tid);
    const gpu0 = try weight_layout.relayout(allocator, &sm, .init(.{ .kind = .gpu, .index = 0 }, .row_major), w.tid);
    const gpu1 = try weight_layout.relayout(allocator, &sm, .init(.{ .kind = .gpu, .index = 1 }, .row_major), w.tid);

    try std.testing.expect(host != gpu0);
    try std.testing.expect(gpu0 != gpu1);
    // Same key twice is still one result.
    try std.testing.expectEqual(gpu0, try weight_layout.relayout(allocator, &sm, .init(.{ .kind = .gpu, .index = 0 }, .row_major), w.tid));

    // A swap reaches every copy, so the extra results are not stale.
    const replacement = try derivable(allocator, &sm, 77);
    defer allocator.free(replacement.bytes);
    try sm.writeDerivedSource(w.tid, replacement.tid);
    const scratch = try derivable(allocator, &sm, 0);
    defer allocator.free(scratch.bytes);
    const got = try allocator.alloc(u8, replacement.bytes.len);
    defer allocator.free(got);
    try sm.readDerivedSource(w.tid, scratch.tid);
    try sm.readToPackedQuant(scratch.tid, got);
    try std.testing.expectEqualSlices(u8, replacement.bytes, got);
}

// Reclaiming a derived-away weight turns on "does any live program still read it", and a
// `Context` shares one store between models — so the count has to live with the tensor,
// not in the model that happened to compile last.
test "derived: program references are counted on the store" {
    const allocator = std.testing.allocator;

    var sm = StorageManager.init(allocator);
    defer sm.deinit();
    const w = try derivable(allocator, &sm, 1);
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

// ---------------------------------------------------------------------------
// weight_layout
// ---------------------------------------------------------------------------

/// A rank-2 q8 matmul-B `[k, n]`.
fn q8MatmulB(allocator: std.mem.Allocator, sm: *StorageManager, k: usize, n: usize, seed: usize) !TensorId {
    const vals = try allocator.alloc(f32, k * n);
    defer allocator.free(vals);
    for (vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i + seed) % 23)) - 11)) * 0.07;
    const buf = try packQ8MatmulB(allocator, vals, k, n);
    defer allocator.free(buf);

    const tid = try sm.createTensor(.q8_0, &[_]usize{ k, n }, .{
        .quant_axis = 0,
    });
    try sm.writeFromPackedQuant(tid, buf);
    return tid;
}

test "weight_layout: a quantized matmul weight is re-laid and contracted row-wise" {
    const allocator = std.testing.allocator;
    const m: usize = 2;
    const k: usize = 64;
    const n: usize = 96;

    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const a_tid = try f32Tensor(&sm, &[_]usize{ m, k }, 3);
    const w = try q8MatmulB(allocator, &sm, k, n, 1);

    var g = Graph.init(allocator);
    defer g.deinit();
    const a = try g.addInput(.f32, &[_]usize{ m, k });
    try g.bindExternal(a, @intCast(a_tid));
    const b = try g.addInput(.q8_0, &[_]usize{ k, n });
    try g.bindExternal(b, @intCast(w));
    try g.setOutputs(&[_]ValueId{try g.addMatMul(a, b, 1.0, 0.0)});

    var prog = try program.compileGraph(allocator, &g, &sm, cpu_target.withPasses(.initOne(.weight_layout)));
    defer prog.deinit();

    try std.testing.expectEqual(@as(usize, 0), countOp(&g, .MatMul));
    try std.testing.expectEqual(@as(usize, 1), countOp(&g, .MatMulNT));

    // The weight now lives in its re-laid copy: `[n, k]`, blocked along its rows.
    const at = sm.derivedLocate(w) orelse return error.TestExpectedFolded;
    const laid = try sm.getConst(at.result);
    try std.testing.expectEqual(@as(u8, 1), laid.quant_axis);
    try std.testing.expectEqualSlices(usize, &[_]usize{ n, k }, laid.shape);
}

// The NT lowering runs q8 only, so the pass must leave every other quantized
// weight on the MatMul it came with — or a default compile emits an op the CPU
// backend rejects at run time.
test "weight_layout: a q4_0 matmul is left alone and runs under default passes" {
    const allocator = std.testing.allocator;
    const m: usize = 2;
    const k: usize = 64;
    const n: usize = 8;

    var sm = StorageManager.init(allocator);
    defer sm.deinit();
    const a_tid = try f32Tensor(&sm, &[_]usize{ m, k }, 3);

    // q4_0 blocks: an f16 scale of 1.0, then 16 bytes of nibbles.
    const blocks = (k / 32) * n;
    const buf = try allocator.alloc(u8, blocks * 18);
    defer allocator.free(buf);
    for (0..blocks) |bi| {
        std.mem.writeInt(u16, buf[bi * 18 ..][0..2], @bitCast(@as(f16, 1.0)), .little);
        for (buf[bi * 18 + 2 ..][0..16], 0..) |*q, i| q.* = @truncate(bi *% 37 +% i *% 11);
    }
    const w = try sm.createTensor(.q4_0, &[_]usize{ k, n }, .{ .quant_axis = 0 });
    try sm.writeFromPackedQuant(w, buf);

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    var out: [2][m * n]f32 = undefined;
    for ([_]program.Target{ cpu_target.withPasses(.empty), cpu_target }, 0..) |target, i| {
        var g = Graph.init(allocator);
        defer g.deinit();
        const a = try g.addInput(.f32, &[_]usize{ m, k });
        try g.bindExternal(a, @intCast(a_tid));
        const b = try g.addInput(.q4_0, &[_]usize{ k, n });
        try g.bindExternal(b, @intCast(w));
        try g.setOutputs(&[_]ValueId{try g.addMatMul(a, b, 1.0, 0.0)});

        var prog = try program.compileGraph(allocator, &g, &sm, target);
        defer prog.deinit();
        try std.testing.expectEqual(@as(usize, 0), countOp(&g, .MatMulNT));
        try cpu.backend().executeProgram(&prog, sm.tensorStore());
        try sm.readToPackedScalar(prog.outputs[0], std.mem.sliceAsBytes(&out[i]));
    }
    try std.testing.expectEqualSlices(f32, &out[0], &out[1]);
}

test "weight_layout: re-laying permutes bytes and requantizes nothing" {
    const allocator = std.testing.allocator;
    const k: usize = 64;
    const n: usize = 96;

    var sm = StorageManager.init(allocator);
    defer sm.deinit();
    const w = try q8MatmulB(allocator, &sm, k, n, 7);

    const before = try allocator.alloc(u8, (k / 32) * n * 34);
    defer allocator.free(before);
    try sm.readPackedAtPlacement(w, before);

    // For each grouping `W`, block (kb, col) lands in segment
    // `(col / W, kb)`: its scale at lane `col % W` of the segment's `W` scales,
    // then each 4-byte chunk `c` at lane `col % W` of the segment's chunk `c`.
    // Every byte arrives — nothing is requantized.
    const blocks = k / 32;
    for ([_]types.QuantBlockOrder{ .lanes4, .lanes8, .lanes16 }) |order| {
        const laid = try weight_layout.relayout(allocator, &sm, .init(.{}, order), w);
        try std.testing.expectEqual(order, (try sm.getConst(laid)).block_order);
        const after = try allocator.alloc(u8, blocks * n * 34);
        defer allocator.free(after);
        try sm.readPackedAtPlacement(laid, after);

        const W = order.groupRows();
        for (0..blocks) |kb| {
            for (0..n) |col| {
                const seg = ((col / W) * blocks + kb) * W * 34;
                const lane = col % W;
                const src = before[(kb * n + col) * 34 ..][0..34];
                try std.testing.expectEqualSlices(u8, src[0..2], after[seg + 2 * lane ..][0..2]);
                for (0..8) |c| {
                    try std.testing.expectEqualSlices(u8, src[2 + 4 * c ..][0..4], after[seg + 2 * W + (c * W + lane) * 4 ..][0..4]);
                }
            }
        }
    }

    // And the source's own bytes still come back out of it.
    const back = try allocator.alloc(u8, before.len);
    defer allocator.free(back);
    try sm.readPackedAtPlacement(w, back);
    try std.testing.expectEqualSlices(u8, before, back);
}

// The relayout splits its rows across the store's bulk threads. Rows write disjoint
// bytes, so the pooled result must equal the serial one byte for byte, for both
// source forms, with enough rows that the workers claim several ranges each.
test "weight_layout: a pooled relayout matches the serial one byte for byte" {
    const allocator = std.testing.allocator;
    const k: usize = 4096;
    const n: usize = 512;
    const blocks = k / 32;

    // An `[n, k]` source, already blocked along its rows. Only bytes move, so any
    // block contents do.
    const nt_bytes = try allocator.alloc(u8, n * blocks * 34);
    defer allocator.free(nt_bytes);
    for (nt_bytes, 0..) |*b, i| b.* = @truncate(i *% 131 +% (i >> 9));

    var results: [2][2][]u8 = undefined;
    for ([_]usize{ 1, 4 }, 0..) |threads, t| {
        var sm = StorageManager.init(allocator);
        defer sm.deinit();
        sm.bulk_threads = threads;

        const km = try q8MatmulB(allocator, &sm, k, n, 3);
        const nt = try sm.createTensor(.q8_0, &.{ n, k }, .{ .quant_axis = 1 });
        try sm.writeFromPackedQuant(nt, nt_bytes);

        for ([_]TensorId{ km, nt }, 0..) |src, s| {
            const laid = try weight_layout.relayout(allocator, &sm, .init(.{}, .lanes8), src);
            results[t][s] = try allocator.alloc(u8, n * blocks * 34);
            try sm.readPackedAtPlacement(laid, results[t][s]);
        }
    }
    defer for (results) |pair| for (pair) |r| allocator.free(r);
    for (0..2) |s| try std.testing.expectEqualSlices(u8, results[0][s], results[1][s]);
}

// A swap after the source was reclaimed goes through the recorded mapping, so the
// mapping must describe the order the pass actually chose. A width that does not
// split into whole groups stays row-major.
test "weight_layout: a re-laid weight round-trips a swap in every block order" {
    const allocator = std.testing.allocator;
    const k: usize = 64;

    const cases = [_]struct { n: usize, want: types.QuantBlockOrder, got: types.QuantBlockOrder }{
        .{ .n = 6, .want = .lanes4, .got = .row_major },
        .{ .n = 8, .want = .lanes4, .got = .lanes4 },
        .{ .n = 16, .want = .lanes8, .got = .lanes8 },
        .{ .n = 32, .want = .lanes16, .got = .lanes16 },
    };
    for (cases) |case| {
        const n = case.n;
        const order = case.got;
        var sm = StorageManager.init(allocator);
        defer sm.deinit();

        const w = try q8MatmulB(allocator, &sm, k, n, 3);
        const laid = try weight_layout.relayout(allocator, &sm, .init(.{}, case.want), w);
        try std.testing.expectEqual(order, (try sm.getConst(laid)).block_order);

        const replacement = try q8MatmulB(allocator, &sm, k, n, 41);
        try sm.writeDerivedSource(w, replacement);

        const bytes = (k / 32) * n * 34;
        const want = try allocator.alloc(u8, bytes);
        defer allocator.free(want);
        const got = try allocator.alloc(u8, bytes);
        defer allocator.free(got);
        const scratch = try q8MatmulB(allocator, &sm, k, n, 0);
        try sm.readDerivedSource(w, scratch);
        try sm.readToPackedQuant(replacement, want);
        try sm.readToPackedQuant(scratch, got);
        try std.testing.expectEqualSlices(u8, want, got);
    }
}

// A tied table — looked up by row and contracted by an NT matmul — is kept once:
// the lookup reads the re-laid copy, and the source is left unread and freed.
test "weight_layout: a lookup of a re-laid table reads the re-laid copy" {
    const allocator = std.testing.allocator;
    const n: usize = 64;
    const k: usize = 64;
    const m: usize = 2;
    const blocks = k / 32;

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    for ([_]types.QuantBlockOrder{ .lanes4, .lanes8, .lanes16, .lanes32 }) |order| {
        var sm = StorageManager.init(allocator);
        defer sm.deinit();

        const table_bytes = try allocator.alloc(u8, n * blocks * 34);
        defer allocator.free(table_bytes);
        for (0..n * blocks) |bi| {
            const blk = table_bytes[bi * 34 ..][0..34];
            std.mem.writeInt(u16, blk[0..2], @bitCast(@as(f16, @floatFromInt(bi % 5 + 1)) * 0.01), .little);
            for (blk[2..], 0..) |*q, i| q.* = @truncate(bi *% 29 +% i *% 7);
        }
        const table = try sm.createTensor(.q8_0, &.{ n, k }, .{ .quant_axis = 1 });
        try sm.writeFromPackedQuant(table, table_bytes);
        const a_tid = try f32Tensor(&sm, &.{ m, k }, 5);
        const ids = [_]i32{ 5, 0, 63 };
        const idx_tid = try sm.createTensor(.i32, &.{ 1, ids.len }, .{});
        try sm.writeFromPackedScalar(idx_tid, std.mem.sliceAsBytes(&ids));

        var rows: [2][ids.len * k]f32 = undefined;
        var prods: [2][m * n]f32 = undefined;
        for ([_]opt.Policy{ .empty, .initOne(.weight_layout) }, 0..) |policy, i| {
            var g = Graph.init(allocator);
            defer g.deinit();
            const t = try g.addInput(.q8_0, &.{ n, k });
            try g.bindExternal(t, @intCast(table));
            const idx = try g.addInput(.i32, &.{ 1, ids.len });
            try g.bindExternal(idx, @intCast(idx_tid));
            const a = try g.addInput(.f32, &.{ m, k });
            try g.bindExternal(a, @intCast(a_tid));
            try g.setOutputs(&.{ try g.addGather(t, idx, 0, 0), try g.addMatMulNT(a, t, 1.0, 0.0) });

            var prog = try program.compileGraph(allocator, &g, &sm, program.Target.init(.{}, order).withPasses(policy));
            defer prog.deinit();
            try cpu.backend().executeProgram(&prog, sm.tensorStore());
            try sm.readToPackedScalar(prog.outputs[0], std.mem.sliceAsBytes(&rows[i]));
            try sm.readToPackedScalar(prog.outputs[1], std.mem.sliceAsBytes(&prods[i]));

            if (i == 1) {
                for (prog.steps) |step| switch (step.op) {
                    .GatherRows => |gr| try std.testing.expectEqual(order, (try sm.getConst(gr.table)).block_order),
                    else => {},
                };
                try std.testing.expect(!try sm.tensorHasBacking(table));
            }
        }
        try std.testing.expectEqualSlices(f32, &rows[0], &rows[1]);
        try std.testing.expect(nmseOf(&prods[0], &prods[1]) < 1e-10);
    }
}

// What `opt.zig` asks of every pass — that dropping it costs only speed. The
// NT kernel is a different kernel, so the check is against arithmetic, not bits:
// with the pass and without, at every M, results match the int8 reference to f32
// rounding.
test "weight_layout: results follow the arithmetic, not the pass" {
    const allocator = std.testing.allocator;
    const k: usize = 128;
    const n: usize = 96;
    const blocks = k / 32;

    for ([_]usize{ 1, 2, 3, 7, 8, 17 }) |m| {
        var sm = StorageManager.init(allocator);
        defer sm.deinit();

        const a_vals = try allocator.alloc(f32, m * k);
        defer allocator.free(a_vals);
        var seed: u64 = 12345;
        for (a_vals) |*v| {
            seed = seed *% 6364136223846793005 +% 1442695040888963407;
            v.* = (@as(f32, @floatFromInt((seed >> 33) % 2000)) - 1000.0) * 0.0005;
        }
        const a_tid = try sm.createTensor(.f32, &[_]usize{ m, k }, .{});
        try sm.writeFromPackedScalar(a_tid, std.mem.sliceAsBytes(a_vals));
        const w = try q8MatmulB(allocator, &sm, k, n, 1);

        const packed_w = try allocator.alloc(u8, blocks * n * 34);
        defer allocator.free(packed_w);
        try sm.readPackedAtPlacement(w, packed_w);

        // The reference, in f64, with the activation as the library's block
        // quantizer rounds it.
        const want = try allocator.alloc(f32, m * n);
        defer allocator.free(want);
        for (0..m) |r| {
            for (0..n) |c| {
                var acc: f64 = 0;
                for (0..blocks) |kb| {
                    const a_blk = a_vals[r * k + kb * 32 ..][0..32];
                    var aq: [32]i8 = undefined;
                    const a_scale = matmul_q_i8.quantizeABlock(a_blk.ptr, &aq);
                    const off = (kb * n + c) * 34;
                    const b_scale: f64 = @as(f16, @bitCast(std.mem.readInt(u16, packed_w[off..][0..2], .little)));
                    for (0..32) |t| {
                        const q: f64 = @floatFromInt(@as(i8, @bitCast(packed_w[off + 2 + t])));
                        const x: f64 = @as(f64, a_scale) * @as(f64, @floatFromInt(aq[t]));
                        acc += x * q * b_scale;
                    }
                }
                want[r * n + c] = @floatCast(acc);
            }
        }

        var cpu = try cpu_backend_mod.CpuBackend.initWithOptions(allocator, .{});
        defer cpu.deinit();
        // Laid out the way this backend's kernel reads, as a context would.
        const target: program.Target = .init(.{}, cpu.quantBlockOrder());
        for ([_]opt.Policy{ .empty, .initOne(.weight_layout) }) |policy| {
            var g = Graph.init(allocator);
            defer g.deinit();
            const a = try g.addInput(.f32, &[_]usize{ m, k });
            try g.bindExternal(a, @intCast(a_tid));
            const b = try g.addInput(.q8_0, &[_]usize{ k, n });
            try g.bindExternal(b, @intCast(w));
            try g.setOutputs(&[_]ValueId{try g.addMatMul(a, b, 1.0, 0.0)});
            var prog = try program.compileGraph(allocator, &g, &sm, target.withPasses(policy));
            defer prog.deinit();
            try cpu.backend().executeProgram(&prog, sm.tensorStore());
            const got = try allocator.alloc(f32, m * n);
            defer allocator.free(got);
            try sm.readToPackedScalar(prog.outputs[0], std.mem.sliceAsBytes(got));
            try std.testing.expect(nmseOf(want, got) < 1e-10);
        }
    }
}

fn nmseOf(a: []const f32, b: []const f32) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (a, b) |av, bv| {
        const d: f64 = @as(f64, av) - @as(f64, bv);
        num += d * d;
        den += @as(f64, av) * @as(f64, av);
    }
    return if (den == 0) num else num / den;
}

// `add_norm` and `gate` are on for every target now, so the CPU runs its fused
// kernels by default. Nothing else pins them to the unfused forms they replace:
// the GPU comparison compiles once for the GPU and runs that program on both
// backends, so it never sees an unfused CPU schedule. This does.
test "add_norm + gate: the fused CPU kernels match the pair they replace" {
    const allocator = std.testing.allocator;
    const M = 3;
    const N = 64;

    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const res = try f32Tensor(&sm, &[_]usize{ M, N }, 2);
    const x = try f32Tensor(&sm, &[_]usize{ M, N }, 7);
    const gamma = try f32Tensor(&sm, &[_]usize{N}, 3);
    const beta = try f32Tensor(&sm, &[_]usize{N}, 5);
    const up = try f32Tensor(&sm, &[_]usize{ M, N }, 11);

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();

    var out: [2][]f32 = undefined;
    for ([_]opt.Policy{ .empty, .initMany(&.{ .add_norm, .gate }) }, 0..) |policy, i| {
        var g = Graph.init(allocator);
        defer g.deinit();
        const r = try g.addInput(.f32, &.{ M, N });
        try g.bindExternal(r, @intCast(res));
        const xi = try g.addInput(.f32, &.{ M, N });
        try g.bindExternal(xi, @intCast(x));
        const gi = try g.addInput(.f32, &.{N});
        try g.bindExternal(gi, @intCast(gamma));
        const bi = try g.addInput(.f32, &.{N});
        try g.bindExternal(bi, @intCast(beta));
        const ui = try g.addInput(.f32, &.{ M, N });
        try g.bindExternal(ui, @intCast(up));

        // `res + rmsnorm(x)` feeding `silu(.) * up` — one of each fusion.
        const normed = try g.addRMSNorm(xi, gi, bi, 1e-6, &.{N});
        const summed = try g.addElemwiseBinary(.add, r, normed);
        const gated = try g.addElemwiseBinary(.mul, try g.addUnary(.silu, summed), ui);
        try g.setOutputs(&.{gated});

        var prog = try program.compileGraph(allocator, &g, &sm, cpu_target.withPasses(policy));
        defer prog.deinit();
        try cpu.backend().executeProgram(&prog, sm.tensorStore());

        const buf = try allocator.alloc(u8, M * N * @sizeOf(f32));
        defer allocator.free(buf);
        try sm.readToPackedScalar(prog.outputs[0], buf);
        out[i] = try allocator.alloc(f32, M * N);
        @memcpy(out[i], asF32(buf));
    }
    defer for (out) |o| allocator.free(o);

    // Same arithmetic in a different order: f32 rounding only.
    for (out[0], out[1]) |unfused, fused| {
        try std.testing.expectApproxEqAbs(unfused, fused, 1e-5);
    }
}
