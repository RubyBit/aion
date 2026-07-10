// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! GPU backend end-to-end correctness test. Compiles a small graph
//! (`silu(a + b)`), runs it on both the CPU backend and the WebGPU `GpuBackend`,
//! and asserts the outputs match — proving the kernels + residency + dispatch
//! are correct end-to-end on real hardware.
//!
//! Built as its own test artifact (it links wgpu-native), wired into
//! `zig build test` / `test-fast` when `-Dgpu` is on. When no GPU adapter is
//! available (headless CI), the test skips itself rather than failing.
//!
//! Tests request the HIGH-POWER adapter so a discrete GPU (the primary deploy
//! target) is what gets validated — the default adapter is often an iGPU, and
//! at least one real driver-level miscompile (signed `%` on negative i32,
//! NVIDIA Vulkan) only reproduced on the discrete card.

const std = @import("std");
const aion = @import("aion");

const gpu = aion.gpu; // feature-gated GPU backend (root.zig exposes it when -Dgpu)
const wgpu = gpu.wgpu; // the wgpu helper, re-exported by the backend

const StorageManager = aion.storage_manager.StorageManager;
const Graph = aion.graph.Graph;
const TensorId = aion.storage_manager.TensorId;
const plan = aion.plan;

const M = 16;
const N = 64;
const COUNT = M * N;

/// Build `silu(a + b)` over two [M,N] f32 inputs; returns the compiled program
/// + the output tensor id. Inputs are single-tile (tile_shape == shape).
fn buildProgram(alloc: std.mem.Allocator, mgr: *StorageManager) !struct { prog: aion.program.Program, out: TensorId } {
    const shape = [_]usize{ M, N };
    const a_id = try mgr.createTiledTensor(.f32, &shape, &shape, .{});
    const b_id = try mgr.createTiledTensor(.f32, &shape, &shape, .{});

    var a_data: [COUNT]f32 = undefined;
    var b_data: [COUNT]f32 = undefined;
    for (0..COUNT) |i| {
        a_data[i] = @as(f32, @floatFromInt(i)) * 0.01 - 2.0;
        b_data[i] = @as(f32, @floatFromInt(i)) * -0.005 + 0.5;
    }
    try mgr.writeFromPackedScalar(a_id, std.mem.sliceAsBytes(a_data[0..]));
    try mgr.writeFromPackedScalar(b_id, std.mem.sliceAsBytes(b_data[0..]));

    var g = Graph.init(alloc);
    defer g.deinit();
    const av = try g.addInput(.f32, &shape);
    try g.bindExternal(av, a_id);
    const bv = try g.addInput(.f32, &shape);
    try g.bindExternal(bv, b_id);
    const sum = try g.addElemwiseBinary(.add, av, bv);
    const res = try g.addUnary(.silu, sum);
    try g.setOutputs(&[_]aion.graph.ValueId{res});

    const prog = try aion.program.compileGraph(alloc, &g, mgr, .{});
    return .{ .prog = prog, .out = prog.outputs[0] };
}

fn readOutput(mgr: *StorageManager, id: TensorId, dst: []f32) !void {
    try mgr.readToPackedScalar(id, std.mem.sliceAsBytes(dst));
}

test "gpu backend: silu(a+b) matches CPU reference" {
    const alloc = std.testing.allocator;

    // No adapter available (e.g. headless CI) → skip rather than fail.
    var device = wgpu.Gpu.init(.{ .power = .high }) catch return error.SkipZigTest;
    defer device.deinit();

    var gb = gpu.GpuBackend.init(alloc, &device);
    defer gb.deinit();

    // --- CPU reference ---
    var cpu_result: [COUNT]f32 = undefined;
    {
        var mgr = StorageManager.init(alloc);
        defer mgr.deinit();
        var built = try buildProgram(alloc, &mgr);
        defer built.prog.deinit();
        var cpu = aion.cpu.CpuBackend.init(alloc);
        defer cpu.deinit();
        try cpu.backend().executeProgram(&built.prog, mgr.tensorStore());
        try readOutput(&mgr, built.out, cpu_result[0..]);
    }

    // --- GPU ---
    var gpu_result: [COUNT]f32 = undefined;
    {
        var mgr = StorageManager.init(alloc);
        defer mgr.deinit();
        var built = try buildProgram(alloc, &mgr);
        defer built.prog.deinit();
        try gb.backend().executeProgram(&built.prog, mgr.tensorStore());
        try readOutput(&mgr, built.out, gpu_result[0..]);
    }

    // --- compare ---
    for (0..COUNT) |i| {
        try std.testing.expectApproxEqAbs(cpu_result[i], gpu_result[i], 1e-4);
    }
}

// Multi-tile matmul. Sizes chosen so M/K/N each span several tiles under the
// small policy below, exercising the GPU executor's ti_m × ti_n loop and the
// k-tile accumulation (beta on the first k-tile, 1.0 after).
const MM_M = 96;
const MM_K = 160;
const MM_N = 128;

/// Build `C = A @ B` over [MM_M,MM_K] @ [MM_K,MM_N] f32, tiled so the policy
/// forces multiple tiles in every dimension. Returns the compiled program + the
/// output tensor id.
fn buildMatMulProgram(alloc: std.mem.Allocator, mgr: *StorageManager) !struct { prog: aion.program.Program, out: TensorId } {
    const policy: plan.TilePolicy = .{ .base_square_2d = 32, .base_1d = 32, .tile_alignment = 64 };
    const tiles = plan.chooseMatMulTiles(policy, MM_M, MM_N, MM_K, .f32);

    const a_id = try mgr.createTiledTensor(.f32, &[_]usize{ MM_M, MM_K }, &[_]usize{ tiles.tm, tiles.tk }, .{ .tile_alignment = 64 });
    const b_id = try mgr.createTiledTensor(.f32, &[_]usize{ MM_K, MM_N }, &[_]usize{ tiles.tk, tiles.tn }, .{ .tile_alignment = 64 });

    const a_data = try alloc.alloc(f32, MM_M * MM_K);
    defer alloc.free(a_data);
    const b_data = try alloc.alloc(f32, MM_K * MM_N);
    defer alloc.free(b_data);
    for (a_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) * 0.1;
    for (b_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) * 0.2;
    try mgr.writeFromPackedScalar(a_id, std.mem.sliceAsBytes(a_data));
    try mgr.writeFromPackedScalar(b_id, std.mem.sliceAsBytes(b_data));

    var g = Graph.init(alloc);
    defer g.deinit();
    const av = try g.addInput(.f32, &[_]usize{ MM_M, MM_K });
    try g.bindExternal(av, a_id);
    const bv = try g.addInput(.f32, &[_]usize{ MM_K, MM_N });
    try g.bindExternal(bv, b_id);
    const cv = try g.addMatMul(av, bv, 1.0, 0.0);
    try g.setOutputs(&[_]aion.graph.ValueId{cv});

    const prog = try aion.program.compileGraph(alloc, &g, mgr, policy);
    return .{ .prog = prog, .out = prog.outputs[0] };
}

test "gpu backend: tiled matmul matches CPU reference" {
    const alloc = std.testing.allocator;

    var device = wgpu.Gpu.init(.{ .power = .high }) catch return error.SkipZigTest;
    defer device.deinit();

    var gb = gpu.GpuBackend.init(alloc, &device);
    defer gb.deinit();

    const cpu_result = try alloc.alloc(f32, MM_M * MM_N);
    defer alloc.free(cpu_result);
    const gpu_result = try alloc.alloc(f32, MM_M * MM_N);
    defer alloc.free(gpu_result);

    {
        var mgr = StorageManager.init(alloc);
        defer mgr.deinit();
        var built = try buildMatMulProgram(alloc, &mgr);
        defer built.prog.deinit();
        var cpu = aion.cpu.CpuBackend.init(alloc);
        defer cpu.deinit();
        try cpu.backend().executeProgram(&built.prog, mgr.tensorStore());
        try readOutput(&mgr, built.out, cpu_result);
    }

    {
        var mgr = StorageManager.init(alloc);
        defer mgr.deinit();
        var built = try buildMatMulProgram(alloc, &mgr);
        defer built.prog.deinit();
        try gb.backend().executeProgram(&built.prog, mgr.tensorStore());
        try readOutput(&mgr, built.out, gpu_result);
    }

    for (cpu_result, gpu_result) |cv, gv| {
        try std.testing.expectApproxEqAbs(cv, gv, 1e-3);
    }
}

// ---- shared CPU-vs-GPU runner for the simple/row-wise op tests --------------

const BuiltProg = struct { prog: aion.program.Program, out: TensorId };

/// Build the same program twice (fresh storage each time), run it on the CPU
/// backend and the GPU backend, and compare the packed f32 outputs.
fn expectGpuMatchesCpu(comptime buildFn: fn (std.mem.Allocator, *StorageManager) anyerror!BuiltProg, out_len: usize, tol: f32) !void {
    const alloc = std.testing.allocator;

    var device = wgpu.Gpu.init(.{ .power = .high }) catch return error.SkipZigTest;
    defer device.deinit();
    var gb = gpu.GpuBackend.init(alloc, &device);
    defer gb.deinit();

    const cpu_result = try alloc.alloc(f32, out_len);
    defer alloc.free(cpu_result);
    const gpu_result = try alloc.alloc(f32, out_len);
    defer alloc.free(gpu_result);

    {
        var mgr = StorageManager.init(alloc);
        defer mgr.deinit();
        var built = buildFn(alloc, &mgr) catch |e| {
            std.debug.print("build (cpu phase) failed: {s}\n", .{@errorName(e)});
            return e;
        };
        defer built.prog.deinit();
        var cpu = aion.cpu.CpuBackend.init(alloc);
        defer cpu.deinit();
        cpu.backend().executeProgram(&built.prog, mgr.tensorStore()) catch |e| {
            std.debug.print("cpu execute failed: {s}\n", .{@errorName(e)});
            return e;
        };
        try mgr.readToPackedScalar(built.out, std.mem.sliceAsBytes(cpu_result));
    }
    {
        var mgr = StorageManager.init(alloc);
        defer mgr.deinit();
        var built = buildFn(alloc, &mgr) catch |e| {
            std.debug.print("build (gpu phase) failed: {s}\n", .{@errorName(e)});
            return e;
        };
        defer built.prog.deinit();
        gb.backend().executeProgram(&built.prog, mgr.tensorStore()) catch |e| {
            std.debug.print("gpu execute failed: {s}\n", .{@errorName(e)});
            return e;
        };
        try mgr.readToPackedScalar(built.out, std.mem.sliceAsBytes(gpu_result));
    }

    for (cpu_result, gpu_result, 0..) |cv, gv, i| {
        std.testing.expectApproxEqAbs(cv, gv, tol) catch |e| {
            std.debug.print("mismatch at [{d}]: cpu={d} gpu={d}\n", .{ i, cv, gv });
            return e;
        };
    }
}

/// Like `expectGpuMatchesCpu` but for i32 outputs: negative i32 bit patterns
/// reinterpret as NaN through the f32 harness (NaN != NaN), so compare exactly
/// as integers instead.
fn expectGpuMatchesCpuI32(comptime buildFn: fn (std.mem.Allocator, *StorageManager) anyerror!BuiltProg, out_len: usize) !void {
    const alloc = std.testing.allocator;

    var device = wgpu.Gpu.init(.{ .power = .high }) catch return error.SkipZigTest;
    defer device.deinit();
    var gb = gpu.GpuBackend.init(alloc, &device);
    defer gb.deinit();

    const cpu_result = try alloc.alloc(i32, out_len);
    defer alloc.free(cpu_result);
    const gpu_result = try alloc.alloc(i32, out_len);
    defer alloc.free(gpu_result);

    {
        var mgr = StorageManager.init(alloc);
        defer mgr.deinit();
        var built = try buildFn(alloc, &mgr);
        defer built.prog.deinit();
        var cpu = aion.cpu.CpuBackend.init(alloc);
        defer cpu.deinit();
        try cpu.backend().executeProgram(&built.prog, mgr.tensorStore());
        try mgr.readToPackedScalar(built.out, std.mem.sliceAsBytes(cpu_result));
    }
    {
        var mgr = StorageManager.init(alloc);
        defer mgr.deinit();
        var built = try buildFn(alloc, &mgr);
        defer built.prog.deinit();
        try gb.backend().executeProgram(&built.prog, mgr.tensorStore());
        try mgr.readToPackedScalar(built.out, std.mem.sliceAsBytes(gpu_result));
    }

    for (cpu_result, gpu_result, 0..) |cv, gv, i| {
        std.testing.expectEqual(cv, gv) catch |e| {
            std.debug.print("i32 mismatch at [{d}]: cpu={d} gpu={d}\n", .{ i, cv, gv });
            return e;
        };
    }
}

/// Create an f32 input tensor with an explicit tile shape and a deterministic
/// value pattern, bound to a fresh graph input value.
fn makeInput(g: *Graph, mgr: *StorageManager, shape: []const usize, tile: []const usize, seed: u32) !aion.graph.ValueId {
    const id = try mgr.createTiledTensor(.f32, shape, tile, .{});
    var n: usize = 1;
    for (shape) |d| n *= d;
    const data = try mgr.allocator.alloc(f32, n);
    defer mgr.allocator.free(data);
    for (data, 0..) |*v, i| {
        const k: u32 = @intCast((i * 2654435761 + seed) % 1000);
        v.* = (@as(f32, @floatFromInt(k)) - 500.0) * 0.004; // [-2, 2)
    }
    try mgr.writeFromPackedScalar(id, std.mem.sliceAsBytes(data));
    const v = try g.addInput(.f32, shape);
    try g.bindExternal(v, id);
    return v;
}

fn finishProg(alloc: std.mem.Allocator, g: *Graph, mgr: *StorageManager, out_v: aion.graph.ValueId) !BuiltProg {
    try g.setOutputs(&[_]aion.graph.ValueId{out_v});
    const prog = try aion.program.compileGraph(alloc, g, mgr, .{});
    return .{ .prog = prog, .out = prog.outputs[0] };
}

// Row-wise op shapes: 48 rows tiled 16 high (3 row-tile dispatches), 64 cols in
// a single tile along the reduced axis — exercises the multi-tile loop AND the
// intra-tile workgroup reduction (cols > one 256-thread sweep is covered by the
// strided loops in the kernels regardless of size).
const RW_M = 48;
const RW_N = 64;

fn buildSoftmax(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    const av = try makeInput(&g, mgr, &.{ RW_M, RW_N }, &.{ 16, RW_N }, 1);
    const out = try g.addSoftmax(av, -1);
    return finishProg(alloc, &g, mgr, out);
}

test "gpu backend: softmax (last axis, row-tiled) matches CPU" {
    try expectGpuMatchesCpu(buildSoftmax, RW_M * RW_N, 1e-5);
}

fn buildRMSNorm(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    const xv = try makeInput(&g, mgr, &.{ RW_M, RW_N }, &.{ 16, RW_N }, 2);
    const gv = try makeInput(&g, mgr, &.{RW_N}, &.{RW_N}, 3);
    const bv = try makeInput(&g, mgr, &.{RW_N}, &.{RW_N}, 4);
    const out = try g.addRMSNorm(xv, gv, bv, 1e-5, &.{RW_N});
    return finishProg(alloc, &g, mgr, out);
}

test "gpu backend: rmsnorm matches CPU" {
    try expectGpuMatchesCpu(buildRMSNorm, RW_M * RW_N, 1e-4);
}

fn buildLayerNorm(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    const xv = try makeInput(&g, mgr, &.{ RW_M, RW_N }, &.{ 16, RW_N }, 5);
    const gv = try makeInput(&g, mgr, &.{RW_N}, &.{RW_N}, 6);
    const bv = try makeInput(&g, mgr, &.{RW_N}, &.{RW_N}, 7);
    const out = try g.addLayerNorm(xv, gv, bv, 1e-5, &.{RW_N});
    return finishProg(alloc, &g, mgr, out);
}

test "gpu backend: layernorm matches CPU" {
    try expectGpuMatchesCpu(buildLayerNorm, RW_M * RW_N, 1e-4);
}

// The RNNT decoder's state gate (`sel_row`): keep + (take-keep)*emit, done via
// reshape [1,H]->[H,1], bcast_mul by scalar emit [1], reshape back, add. This is
// the exact op chain the in-graph loop runs every iteration for h0/c0/h1/c1.
// `emit` (0 or 1) is baked into the b input's data seed.
fn buildSelRow(alloc: std.mem.Allocator, mgr: *StorageManager, emit: f32) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    const H: usize = 640;
    const keep = try makeInput(&g, mgr, &.{ 1, H }, &.{ 1, H }, 40);
    const take = try makeInput(&g, mgr, &.{ 1, H }, &.{ 1, H }, 41);
    const emit_id = try mgr.createTiledTensor(.f32, &.{1}, &.{1}, .{});
    try mgr.writeFromPackedScalar(emit_id, std.mem.sliceAsBytes(&[_]f32{emit}));
    const emit_f = try g.addInput(.f32, &.{1});
    try g.bindExternal(emit_f, emit_id);

    const diff = try g.addElemwiseBinary(.sub, take, keep); // [1,H]
    const diff_c = try g.addViewReshape(diff, &.{ H, 1 }); // [H,1]
    const scaled_c = try g.addBroadcastLastDimBinary(.mul, diff_c, emit_f); // [H,1]
    const scaled = try g.addViewReshape(scaled_c, &.{ 1, H }); // [1,H]
    const out = try g.addElemwiseBinary(.add, keep, scaled); // [1,H]
    return finishProgGpuTiled(alloc, &g, mgr, out);
}
fn buildSelRowEmit(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    return buildSelRow(alloc, mgr, 1.0);
}
fn buildSelRowKeep(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    return buildSelRow(alloc, mgr, 0.0);
}
test "gpu backend: sel_row state gate (emit=1) matches CPU" {
    try expectGpuMatchesCpu(buildSelRowEmit, 640, 1e-5);
}
test "gpu backend: sel_row state gate (emit=0) matches CPU" {
    try expectGpuMatchesCpu(buildSelRowKeep, 640, 1e-5);
}

fn buildBroadcast(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    // 32-wide column tiles: out tile (r, c) pairs with b tile c — exercises the
    // per-tile b lookup, not just a single broadcast vector.
    const av = try makeInput(&g, mgr, &.{ 96, 64 }, &.{ 32, 32 }, 8);
    const bv = try makeInput(&g, mgr, &.{64}, &.{32}, 9);
    const out = try g.addBroadcastLastDimBinary(.mul, av, bv);
    return finishProg(alloc, &g, mgr, out);
}

test "gpu backend: broadcast-last-dim binary matches CPU" {
    try expectGpuMatchesCpu(buildBroadcast, 96 * 64, 1e-6);
}

fn buildReduceAxis(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    const av = try makeInput(&g, mgr, &.{ 32, 48 }, &.{ 32, 48 }, 10);
    const out = try g.addReduceAxis(.mean, av, -1);
    return finishProg(alloc, &g, mgr, out);
}

test "gpu backend: reduce-axis mean matches CPU" {
    try expectGpuMatchesCpu(buildReduceAxis, 32, 1e-5);
}

fn buildReduceAll(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    const av = try makeInput(&g, mgr, &.{ 32, 48 }, &.{ 32, 48 }, 11);
    const out = try g.addReduce(.sum, av);
    return finishProg(alloc, &g, mgr, out);
}

test "gpu backend: reduce-all sum matches CPU" {
    try expectGpuMatchesCpu(buildReduceAll, 1, 1e-3);
}

fn buildCopy(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    const av = try makeInput(&g, mgr, &.{ 64, 32 }, &.{ 64, 32 }, 12);
    const relu = try g.addRelu(av);
    const out = try g.addCopy(relu);
    return finishProg(alloc, &g, mgr, out);
}

test "gpu backend: copy (buffer-to-buffer) matches CPU" {
    try expectGpuMatchesCpu(buildCopy, 64 * 32, 0.0);
}

// ---- MatMulNT (q8_0 / f32 weights) ------------------------------------------

/// Pack f32 rows [n, k] into ggml q8_0 blocks (f16 scale + 32 i8 per 32 elems).
fn packQ8(alloc: std.mem.Allocator, vals: []const f32, n: usize, k: usize) ![]u8 {
    const bpr = k / 32;
    const out = try alloc.alloc(u8, n * bpr * 34);
    for (0..n) |row| {
        for (0..bpr) |blk| {
            const src = vals[row * k + blk * 32 ..][0..32];
            var absmax: f32 = 0;
            for (src) |v| absmax = @max(absmax, @abs(v));
            const scale: f32 = if (absmax == 0) 1.0 else absmax / 127.0;
            const inv: f32 = if (absmax == 0) 0 else 1.0 / scale;
            const off = (row * bpr + blk) * 34;
            std.mem.writeInt(u16, out[off..][0..2], @bitCast(@as(f16, @floatCast(scale))), .little);
            for (src, 0..) |v, i| {
                const q: i32 = std.math.clamp(@as(i32, @intFromFloat(@round(v * inv))), -128, 127);
                out[off + 2 + i] = @bitCast(@as(i8, @intCast(q)));
            }
        }
    }
    return out;
}

/// Build `C = A @ B^T` with A f32 [m,k] (single tile) and B [n,k] tiled
/// `b_tile_rows` rows per N tile, in q8_0 or f32.
fn buildNt(alloc: std.mem.Allocator, mgr: *StorageManager, m: usize, k: usize, n: usize, b_tile_rows: usize, q8: bool) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();

    const av = try makeInput(&g, mgr, &.{ m, k }, &.{ m, k }, 20);

    const b_vals = try alloc.alloc(f32, n * k);
    defer alloc.free(b_vals);
    for (b_vals, 0..) |*v, i| {
        const p: u32 = @intCast((i * 2654435761 + 21) % 1000);
        v.* = (@as(f32, @floatFromInt(p)) - 500.0) * 0.004;
    }

    var bv: aion.graph.ValueId = undefined;
    if (q8) {
        const packed_b = try packQ8(alloc, b_vals, n, k);
        defer alloc.free(packed_b);
        const b_id = try mgr.createTiledTensor(.q8_0, &.{ n, k }, &.{ b_tile_rows, k }, .{ .tile_alignment = 64, .quant_axis = 1 });
        try mgr.writeFromPackedQuant(b_id, packed_b);
        bv = try g.addInput(.q8_0, &.{ n, k });
        try g.bindExternal(bv, b_id);
    } else {
        const b_id = try mgr.createTiledTensor(.f32, &.{ n, k }, &.{ b_tile_rows, k }, .{ .tile_alignment = 64 });
        try mgr.writeFromPackedScalar(b_id, std.mem.sliceAsBytes(b_vals));
        bv = try g.addInput(.f32, &.{ n, k });
        try g.bindExternal(bv, b_id);
    }

    const cv = try g.addMatMulNT(av, bv, 1.0, 0.0);
    return finishProg(alloc, &g, mgr, cv);
}

// M == 1 exercises the GEMV kernel; K = 128 → 2 block pairs/row for q8. B is
// N-tiled 32 rows/tile (n = 100 → 4 tiles, last one a 4-row edge).
fn buildNtQ8Gemv(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    return buildNt(alloc, mgr, 1, 128, 100, 32, true);
}
test "gpu backend: matmul NT q8_0 matvec (M=1) matches CPU" {
    try expectGpuMatchesCpu(buildNtQ8Gemv, 100, 2e-3);
}

// M > 1 exercises the dequant-to-scratch + f32 GEMM path (single edge-sized
// output block under the 128x128 bounds-checked config).
fn buildNtQ8Gemm(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    return buildNt(alloc, mgr, 24, 128, 100, 100, true);
}
test "gpu backend: matmul NT q8_0 GEMM (M=24) matches CPU" {
    try expectGpuMatchesCpu(buildNtQ8Gemm, 24 * 100, 2e-3);
}

fn buildNtF32Gemv(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    return buildNt(alloc, mgr, 1, 64, 50, 50, false);
}
test "gpu backend: matmul NT f32 matvec (M=1) matches CPU" {
    try expectGpuMatchesCpu(buildNtF32Gemv, 50, 1e-4);
}

fn buildNtF32Gemm(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    return buildNt(alloc, mgr, 16, 64, 50, 16, false);
}
test "gpu backend: matmul NT f32 GEMM (M=16, multi-N-tile) matches CPU" {
    try expectGpuMatchesCpu(buildNtF32Gemm, 16 * 50, 1e-4);
}

// ---- MatMul (plain, K-major q8_0 B) — the Gemma decode GEMV -----------------

/// Pack f32 [k, n] (row-major) into ggml q8_0 blocks quantized ALONG K: block
/// grid [k/32, n], row-major, block (bk, col) holds B[bk*32 .. +31, col]. This
/// is the layout `matmul_gemv.wgsl` / `q8_kmajor_to_f32` consume.
fn packQ8Kmajor(alloc: std.mem.Allocator, vals: []const f32, k: usize, n: usize) ![]u8 {
    const bpr = k / 32;
    const out = try alloc.alloc(u8, bpr * n * 34);
    for (0..bpr) |bk| {
        for (0..n) |col| {
            var absmax: f32 = 0;
            for (0..32) |kk| absmax = @max(absmax, @abs(vals[(bk * 32 + kk) * n + col]));
            const scale: f32 = if (absmax == 0) 1.0 else absmax / 127.0;
            const inv: f32 = if (absmax == 0) 0 else 1.0 / scale;
            const off = (bk * n + col) * 34;
            std.mem.writeInt(u16, out[off..][0..2], @bitCast(@as(f16, @floatCast(scale))), .little);
            for (0..32) |kk| {
                const v = vals[(bk * 32 + kk) * n + col];
                const q: i32 = std.math.clamp(@as(i32, @intFromFloat(@round(v * inv))), -128, 127);
                out[off + 2 + kk] = @bitCast(@as(i8, @intCast(q)));
            }
        }
    }
    return out;
}

// The fused GEMV computes `f32 activation · dequant(B)` — the exact same thing
// the (production-validated) dequant→GEMM path does. We validate it against a
// host-computed dequant reference rather than the CPU backend: for these small
// test shapes the CPU routes M==1 through an int8-GEMM that quantizes the
// activation too (a *different*, lossier algorithm), so a CPU cross-check would
// compare apples to oranges. On real Gemma shapes the CPU uses the f32-A path.
fn expectQ8KMajorGemv(alloc: std.mem.Allocator, k: usize, n: usize, tol: f32) !void {
    var device = wgpu.Gpu.init(.{ .power = .high }) catch return error.SkipZigTest;
    defer device.deinit();
    var gb = gpu.GpuBackend.init(alloc, &device);
    defer gb.deinit();

    const a = try alloc.alloc(f32, k);
    defer alloc.free(a);
    for (a, 0..) |*v, i| {
        const kk: u32 = @intCast((i * 2654435761 + 20) % 1000);
        v.* = (@as(f32, @floatFromInt(kk)) - 500.0) * 0.004;
    }
    const b = try alloc.alloc(f32, k * n);
    defer alloc.free(b);
    for (b, 0..) |*v, i| {
        const p: u32 = @intCast((i * 2654435761 + 21) % 1000);
        v.* = (@as(f32, @floatFromInt(p)) - 500.0) * 0.004;
    }
    const packed_b = try packQ8Kmajor(alloc, b, k, n);
    defer alloc.free(packed_b);

    // Host reference: f32 A · dequant(B), per output column.
    const ref = try alloc.alloc(f32, n);
    defer alloc.free(ref);
    for (0..n) |col| {
        var acc: f32 = 0;
        for (0..k / 32) |bk| {
            const off = (bk * n + col) * 34;
            const scale: f32 = @as(f16, @bitCast(std.mem.readInt(u16, packed_b[off..][0..2], .little)));
            for (0..32) |kk| {
                const q: i8 = @bitCast(packed_b[off + 2 + kk]);
                acc += a[bk * 32 + kk] * (@as(f32, @floatFromInt(q)) * scale);
            }
        }
        ref[col] = acc;
    }

    var mgr = StorageManager.init(alloc);
    defer mgr.deinit();
    var g = Graph.init(alloc);
    defer g.deinit();
    const a_id = try mgr.createTiledTensor(.f32, &.{ 1, k }, &.{ 1, k }, .{});
    try mgr.writeFromPackedScalar(a_id, std.mem.sliceAsBytes(a));
    const av = try g.addInput(.f32, &.{ 1, k });
    try g.bindExternal(av, a_id);
    const b_id = try mgr.createTiledTensor(.q8_0, &.{ k, n }, &.{ k, n }, .{ .tile_alignment = 64, .quant_axis = 0 });
    try mgr.writeFromPackedQuant(b_id, packed_b);
    const bv = try g.addInput(.q8_0, &.{ k, n });
    try g.bindExternal(bv, b_id);
    const cv = try g.addMatMul(av, bv, 1.0, 0.0);
    var built = try finishProgGpuTiled(alloc, &g, &mgr, cv);
    defer built.prog.deinit();

    try gb.backend().executeProgram(&built.prog, mgr.tensorStore());
    const out = try alloc.alloc(f32, n);
    defer alloc.free(out);
    try mgr.readToPackedScalar(built.out, std.mem.sliceAsBytes(out));

    for (ref, out, 0..) |r, o, i| {
        std.testing.expectApproxEqAbs(r, o, tol) catch |e| {
            std.debug.print("q8 gemv mismatch at [{d}]: ref={d} gpu={d}\n", .{ i, r, o });
            return e;
        };
    }
}

// K = 256 (8 blocks = one partial 16-block chunk), N = 128 (64 column pairs).
test "gpu backend: matmul q8_0 (K-major) matvec (M=1)" {
    try expectQ8KMajorGemv(std.testing.allocator, 256, 128, 1e-3);
}

// K = 1024 spans multiple 16-block chunks (32 block-rows / 16 = 2 chunks) to
// exercise the shared-A chunk loop and its tail.
test "gpu backend: matmul q8_0 (K-major) matvec (M=1, multi-chunk K)" {
    try expectQ8KMajorGemv(std.testing.allocator, 1024, 128, 2e-3);
}

// Odd N exercises the per-column `gemv_q8_kmajor_odd` path (mixed block
// alignment); N=65 → columns straddle both parities.
test "gpu backend: matmul q8_0 (K-major) matvec (M=1, odd N)" {
    try expectQ8KMajorGemv(std.testing.allocator, 256, 65, 2e-3);
}

// The RNNT joint's vocab+blank projection shape (K=640, N=1025) — the blank is
// the last (odd) column, so a subtle odd-path bug would bias emit-vs-blank.
test "gpu backend: matmul q8_0 (K-major) matvec (M=1, RNNT joint shape)" {
    try expectQ8KMajorGemv(std.testing.allocator, 640, 1025, 2e-3);
}

// ---- decode ops: gather / rope / kv-append -----------------------------------

fn makeInputI32(g: *Graph, mgr: *StorageManager, shape: []const usize, tile: []const usize, vals: []const i32) !aion.graph.ValueId {
    const id = try mgr.createTiledTensor(.i32, shape, tile, .{});
    try mgr.writeFromPackedScalar(id, std.mem.sliceAsBytes(vals));
    const v = try g.addInput(.i32, shape);
    try g.bindExternal(v, id);
    return v;
}

// Indices deliberately hop between table tiles (rows 0..63, tiles of 16) to
// exercise the record-time tile resolution.
const GATHER_IDX = [_]i32{ 3, 17, 62, 0, 33, 47, 5, 18, 40, 63 };

fn buildGatherF32(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    const table = try makeInput(&g, mgr, &.{ 64, 32 }, &.{ 16, 32 }, 40);
    const idx = try makeInputI32(&g, mgr, &.{ 2, 5 }, &.{ 2, 5 }, &GATHER_IDX);
    const out = try g.addGatherRows(table, idx);
    return finishProg(alloc, &g, mgr, out);
}

test "gpu backend: gather rows (f32 table, multi-tile) matches CPU" {
    try expectGpuMatchesCpu(buildGatherF32, 2 * 5 * 32, 0.0);
}

fn buildGatherF32SingleTile(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    // Single-tile table -> the device-side gather kernel (indices read on-GPU).
    const table = try makeInput(&g, mgr, &.{ 64, 32 }, &.{ 64, 32 }, 42);
    const idx = try makeInputI32(&g, mgr, &.{ 2, 5 }, &.{ 2, 5 }, &GATHER_IDX);
    const out = try g.addGatherRows(table, idx);
    return finishProg(alloc, &g, mgr, out);
}

test "gpu backend: gather rows (f32 single-tile table, device-side kernel) matches CPU" {
    try expectGpuMatchesCpu(buildGatherF32SingleTile, 2 * 5 * 32, 0.0);
}

fn buildGatherQ8(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();

    const v = 64;
    const d = 128; // % 64 == 0 for the word-aligned q8 row path
    const table_vals = try alloc.alloc(f32, v * d);
    defer alloc.free(table_vals);
    for (table_vals, 0..) |*val, i| {
        const p: u32 = @intCast((i * 2654435761 + 41) % 1000);
        val.* = (@as(f32, @floatFromInt(p)) - 500.0) * 0.004;
    }
    const packed_t = try packQ8(alloc, table_vals, v, d);
    defer alloc.free(packed_t);
    const t_id = try mgr.createTiledTensor(.q8_0, &.{ v, d }, &.{ 16, d }, .{ .tile_alignment = 64, .quant_axis = 1 });
    try mgr.writeFromPackedQuant(t_id, packed_t);
    const tv = try g.addInput(.q8_0, &.{ v, d });
    try g.bindExternal(tv, t_id);

    const idx = try makeInputI32(&g, mgr, &.{ 2, 5 }, &.{ 2, 5 }, &GATHER_IDX);
    const out = try g.addGatherRows(tv, idx);
    return finishProg(alloc, &g, mgr, out);
}

test "gpu backend: gather rows (q8_0 table) matches CPU" {
    try expectGpuMatchesCpu(buildGatherQ8, 2 * 5 * 128, 1e-6);
}

fn buildGatherQ8SingleTile(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();

    const v = 64;
    const d = 128; // % 64 == 0
    const table_vals = try alloc.alloc(f32, v * d);
    defer alloc.free(table_vals);
    for (table_vals, 0..) |*val, i| {
        const p: u32 = @intCast((i * 2654435761 + 41) % 1000);
        val.* = (@as(f32, @floatFromInt(p)) - 500.0) * 0.004;
    }
    const packed_t = try packQ8(alloc, table_vals, v, d);
    defer alloc.free(packed_t);
    // Single-tile table (tile == full shape) -> the device-side q8 gather kernel
    // (row index resolved on-GPU, no host read).
    const t_id = try mgr.createTiledTensor(.q8_0, &.{ v, d }, &.{ v, d }, .{ .tile_alignment = 64, .quant_axis = 1 });
    try mgr.writeFromPackedQuant(t_id, packed_t);
    const tv = try g.addInput(.q8_0, &.{ v, d });
    try g.bindExternal(tv, t_id);

    const idx = try makeInputI32(&g, mgr, &.{ 2, 5 }, &.{ 2, 5 }, &GATHER_IDX);
    const out = try g.addGatherRows(tv, idx);
    return finishProg(alloc, &g, mgr, out);
}

test "gpu backend: gather rows (q8_0 single-tile table, device-side kernel) matches CPU" {
    try expectGpuMatchesCpu(buildGatherQ8SingleTile, 2 * 5 * 128, 1e-6);
}

// Regression guard for the token-doubling class: the gather index is COMPUTED ON
// DEVICE (idx = base + 0, an i32 elementwise op) rather than host-bound. The
// device gather kernels read it on-GPU; the host-fallback path must submit the
// producing dispatch before its record-time read. Either way the freshly
// computed index — not a stale prior value — must feed the gather. Covers both a
// single-tile f32 table (device kernel) and a multi-tile table (host fallback).
fn buildGatherDeviceIdx(comptime multi_tile: bool) fn (std.mem.Allocator, *StorageManager) anyerror!BuiltProg {
    return struct {
        fn build(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
            var g = Graph.init(alloc);
            defer g.deinit();
            const table_tile: [2]usize = if (multi_tile) .{ 16, 32 } else .{ 64, 32 };
            const table = try makeInput(&g, mgr, &.{ 64, 32 }, &table_tile, 41);
            const base = try makeInputI32(&g, mgr, &.{ 2, 5 }, &.{ 2, 5 }, &GATHER_IDX);
            const zero = try makeInputI32(&g, mgr, &.{ 2, 5 }, &.{ 2, 5 }, &[_]i32{0} ** 10);
            const idx = try g.addElemwiseBinary(.add, base, zero); // device-computed index
            const out = try g.addGatherRows(table, idx);
            return finishProg(alloc, &g, mgr, out);
        }
    }.build;
}

test "gpu backend: gather rows with device-computed index (single-tile) matches CPU" {
    try expectGpuMatchesCpu(buildGatherDeviceIdx(false), 2 * 5 * 32, 0.0);
}

test "gpu backend: gather rows with device-computed index (multi-tile fallback) matches CPU" {
    try expectGpuMatchesCpu(buildGatherDeviceIdx(true), 2 * 5 * 32, 0.0);
}

fn buildRope(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    const x = try makeInput(&g, mgr, &.{ 1, 4, 2, 16 }, &.{ 1, 4, 2, 16 }, 42);
    const positions = try makeInputI32(&g, mgr, &.{ 1, 4 }, &.{ 1, 4 }, &.{ 0, 1, 2, 5 });
    const out = try g.addRoPE1D(x, positions, 10000.0, 1.0, 1.0);
    return finishProg(alloc, &g, mgr, out);
}

test "gpu backend: rope matches CPU" {
    // The CPU uses a fast sincos approximation; GPU hardware sin/cos differs
    // within that approximation's error.
    try expectGpuMatchesCpu(buildRope, 1 * 4 * 2 * 16, 1e-4);
}

fn buildKVAppend(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    const cache = try makeInput(&g, mgr, &.{ 1, 2, 8, 16 }, &.{ 1, 2, 8, 16 }, 43);
    const new_kv = try makeInput(&g, mgr, &.{ 1, 2, 3, 16 }, &.{ 1, 2, 3, 16 }, 44);
    const end = try makeInputI32(&g, mgr, &.{1}, &.{1}, &.{2});
    const out = try g.addKVCacheAppend(cache, new_kv, end);
    return finishProg(alloc, &g, mgr, out);
}

test "gpu backend: kv-cache append matches CPU" {
    try expectGpuMatchesCpu(buildKVAppend, 1 * 2 * 8 * 16, 0.0);
}

// f32 -> f16 -> f32 round trip: exercises both cast kernels; both backends
// apply the identical f16 rounding, so outputs must match closely.
fn buildCastRoundtrip(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    const av = try makeInput(&g, mgr, &.{ 32, 64 }, &.{ 32, 64 }, 30);
    const half = try g.addCast(av, .f16);
    const back = try g.addCast(half, .f32);
    return finishProg(alloc, &g, mgr, back);
}

test "gpu backend: cast f32->f16->f32 matches CPU" {
    try expectGpuMatchesCpu(buildCastRoundtrip, 32 * 64, 1e-6);
}

/// Like `finishProg` but compiled under the GPU tile policy, so attention
/// slices (and conv outputs) land in single tiles — the layout the GPU exec
/// requires. Shapes in these tests are kept within the CPU kernels' run-time
/// tile limits (q rows <= 256, key rows <= 128, v cols <= 64, dk tile <= 128)
/// so the CPU backend can execute the identical program as the reference.
fn finishProgGpuTiled(alloc: std.mem.Allocator, g: *Graph, mgr: *StorageManager, out_v: aion.graph.ValueId) !BuiltProg {
    try g.setOutputs(&[_]aion.graph.ValueId{out_v});
    const policy = plan.tilePolicyForTarget(.{ .kind = .webgpu });
    const prog = try aion.program.compileGraph(alloc, g, mgr, policy);
    return .{ .prog = prog, .out = prog.outputs[0] };
}

fn buildAttention(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    // Rank-3: 2 lead slices, m=24 queries, n=20 keys, dk=32, dv=24 (deliberately
    // not a multiple of 16 — exercises the exact-dim GPU tile chooser).
    const q = try makeInput(&g, mgr, &.{ 2, 24, 32 }, &.{ 1, 24, 32 }, 50);
    const k = try makeInput(&g, mgr, &.{ 2, 20, 32 }, &.{ 1, 20, 32 }, 51);
    const v = try makeInput(&g, mgr, &.{ 2, 20, 24 }, &.{ 1, 20, 24 }, 52);
    const out = try g.addAttention(q, k, v, 0.25, false);
    return finishProgGpuTiled(alloc, &g, mgr, out);
}

test "gpu backend: attention (non-causal) matches CPU" {
    try expectGpuMatchesCpu(buildAttention, 2 * 24 * 24, 2e-5);
}

fn buildAttentionCausal(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    const q = try makeInput(&g, mgr, &.{ 2, 24, 32 }, &.{ 1, 24, 32 }, 53);
    const k = try makeInput(&g, mgr, &.{ 2, 24, 32 }, &.{ 1, 24, 32 }, 54);
    const v = try makeInput(&g, mgr, &.{ 2, 24, 24 }, &.{ 1, 24, 24 }, 55);
    const out = try g.addAttention(q, k, v, 0.25, true);
    return finishProgGpuTiled(alloc, &g, mgr, out);
}

test "gpu backend: attention (causal) matches CPU" {
    try expectGpuMatchesCpu(buildAttentionCausal, 2 * 24 * 24, 2e-5);
}

fn buildMHA(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    // [B, H, T, D]: 2 batches x 3 heads, causal.
    const q = try makeInput(&g, mgr, &.{ 2, 3, 40, 16 }, &.{ 1, 1, 40, 16 }, 56);
    const k = try makeInput(&g, mgr, &.{ 2, 3, 24, 16 }, &.{ 1, 1, 24, 16 }, 57);
    const v = try makeInput(&g, mgr, &.{ 2, 3, 24, 16 }, &.{ 1, 1, 24, 16 }, 58);
    const out = try g.addMultiHeadAttention(q, k, v, 0.25, true, 3);
    return finishProgGpuTiled(alloc, &g, mgr, out);
}

test "gpu backend: multi-head attention matches CPU" {
    try expectGpuMatchesCpu(buildMHA, 2 * 3 * 40 * 16, 2e-5);
}

fn buildMHACached(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    // GQA decode-ish shape with a T large enough to force several 256-key
    // chunks through the online softmax (T=600, end=520 / 100 per batch).
    const q = try makeInput(&g, mgr, &.{ 2, 3, 4, 32 }, &.{ 1, 3, 4, 32 }, 60);
    const k = try makeInput(&g, mgr, &.{ 2, 2, 600, 32 }, &.{ 2, 2, 600, 32 }, 61);
    const v = try makeInput(&g, mgr, &.{ 2, 2, 600, 32 }, &.{ 2, 2, 600, 32 }, 62);
    const pos = try makeInputI32(&g, mgr, &.{ 2, 3 }, &.{ 1, 3 }, &.{ 517, 518, 519, 97, 98, 99 });
    const end = try makeInputI32(&g, mgr, &.{2}, &.{2}, &.{ 520, 100 });
    const out = try g.addMultiHeadAttentionCached(q, k, v, pos, end, 0.1767767, true, 0, 0.0);
    return finishProgGpuTiled(alloc, &g, mgr, out);
}

test "gpu backend: cached GQA attention (f32 caches, multi-chunk) matches CPU" {
    try expectGpuMatchesCpu(buildMHACached, 2 * 3 * 4 * 32, 2e-5);
}

/// f16 input tensor with the same deterministic pattern as `makeInput`.
fn makeInputF16(g: *Graph, mgr: *StorageManager, shape: []const usize, tile: []const usize, seed: u32) !aion.graph.ValueId {
    const id = try mgr.createTiledTensor(.f16, shape, tile, .{});
    var n: usize = 1;
    for (shape) |d| n *= d;
    const data = try mgr.allocator.alloc(f16, n);
    defer mgr.allocator.free(data);
    for (data, 0..) |*val, i| {
        const p: u32 = @intCast((i * 2654435761 + seed) % 1000);
        val.* = @floatCast((@as(f32, @floatFromInt(p)) - 500.0) * 0.004);
    }
    try mgr.writeFromPackedScalar(id, std.mem.sliceAsBytes(data));
    const v = try g.addInput(.f16, shape);
    try g.bindExternal(v, id);
    return v;
}

fn buildMHACachedF16(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    // f16 caches + sliding window + logit soft cap in one program.
    const q = try makeInput(&g, mgr, &.{ 1, 2, 2, 16 }, &.{ 1, 2, 2, 16 }, 63);
    const k = try makeInputF16(&g, mgr, &.{ 1, 1, 64, 16 }, &.{ 1, 1, 64, 16 }, 64);
    const v = try makeInputF16(&g, mgr, &.{ 1, 1, 64, 16 }, &.{ 1, 1, 64, 16 }, 65);
    const pos = try makeInputI32(&g, mgr, &.{ 1, 2 }, &.{ 1, 2 }, &.{ 48, 49 });
    const end = try makeInputI32(&g, mgr, &.{1}, &.{1}, &.{50});
    const out = try g.addMultiHeadAttentionCached(q, k, v, pos, end, 0.25, true, 20, 30.0);
    return finishProgGpuTiled(alloc, &g, mgr, out);
}

fn buildMHACachedSplitK(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    // Long cache (T=2048 >= the split-K threshold) with a decode-shaped q:
    // exercises the flash-decoding split + merge path end-to-end.
    const q = try makeInput(&g, mgr, &.{ 1, 2, 4, 32 }, &.{ 1, 2, 4, 32 }, 66);
    const k = try makeInput(&g, mgr, &.{ 1, 2, 2048, 32 }, &.{ 1, 2, 2048, 32 }, 67);
    const v = try makeInput(&g, mgr, &.{ 1, 2, 2048, 32 }, &.{ 1, 2, 2048, 32 }, 68);
    const pos = try makeInputI32(&g, mgr, &.{ 1, 2 }, &.{ 1, 2 }, &.{ 1990, 1991 });
    const end = try makeInputI32(&g, mgr, &.{1}, &.{1}, &.{1992});
    const out = try g.addMultiHeadAttentionCached(q, k, v, pos, end, 0.1767767, true, 0, 0.0);
    return finishProgGpuTiled(alloc, &g, mgr, out);
}

test "gpu backend: cached GQA attention (split-K long cache) matches CPU" {
    try expectGpuMatchesCpu(buildMHACachedSplitK, 1 * 2 * 4 * 32, 2e-5);
}

test "gpu backend: cached GQA attention (f16 caches, sliding window, soft cap) matches CPU" {
    // f16 cache rounding is identical on both sides, but the CPU soft cap uses
    // a tanh approximation while the GPU uses hardware tanh.
    try expectGpuMatchesCpu(buildMHACachedF16, 1 * 2 * 2 * 16, 5e-3);
}

// ---- conv ----

fn buildConv1D(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    // Grouped + strided + padded, with bias: x [2, 20, 6], w [5, 3, 8], groups=2
    // -> out [2, 10, 8].
    const x = try makeInput(&g, mgr, &.{ 2, 20, 6 }, &.{ 1, 20, 6 }, 70);
    const w = try makeInput(&g, mgr, &.{ 5, 3, 8 }, &.{ 5, 3, 8 }, 71);
    const b = try makeInput(&g, mgr, &.{8}, &.{8}, 72);
    const out = try g.addConv1D(x, w, b, 2, 1, 2, 1, 2);
    return finishProgGpuTiled(alloc, &g, mgr, out);
}

test "gpu backend: conv1d (grouped, strided, biased) matches CPU" {
    try expectGpuMatchesCpu(buildConv1D, 2 * 10 * 8, 1e-4);
}

fn buildConv1DReflect(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    // Reflect padding, dilation 2: x [1, 16, 4], w [3, 4, 4] -> out [1, 12, 4].
    const x = try makeInput(&g, mgr, &.{ 1, 16, 4 }, &.{ 1, 16, 4 }, 73);
    const w = try makeInput(&g, mgr, &.{ 3, 4, 4 }, &.{ 3, 4, 4 }, 74);
    const out = try g.addConv1DWithPadMode(x, w, null, 1, 2, 0, 0, .reflect, 1);
    return finishProgGpuTiled(alloc, &g, mgr, out);
}

test "gpu backend: conv1d (reflect pad, dilated, no bias) matches CPU" {
    try expectGpuMatchesCpu(buildConv1DReflect, 1 * 12 * 4, 1e-4);
}

fn buildConv1DDepthwise(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    // Depthwise causal (Nemotron Conformer shape): groups == c_in == c_out == 8,
    // k=3, pad_left=2, pad_right=0, stride 1 -> exercises conv_dw_f32.
    const x = try makeInput(&g, mgr, &.{ 1, 12, 8 }, &.{ 1, 12, 8 }, 78);
    const w = try makeInput(&g, mgr, &.{ 3, 1, 8 }, &.{ 3, 1, 8 }, 79);
    const b = try makeInput(&g, mgr, &.{8}, &.{8}, 80);
    const out = try g.addConv1D(x, w, b, 1, 1, 2, 0, 8);
    return finishProgGpuTiled(alloc, &g, mgr, out);
}

test "gpu backend: conv1d (depthwise causal) matches CPU" {
    try expectGpuMatchesCpu(buildConv1DDepthwise, 1 * 12 * 8, 1e-4);
}

fn buildConv1DDepthwiseStride2(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    // Depthwise, stride 2, batch 2, no bias: groups == c_in == c_out == 16, k=3,
    // pad_left=1 -> out [2, 10, 16]. Exercises the stride-2 span + x_base offset.
    const x = try makeInput(&g, mgr, &.{ 2, 20, 16 }, &.{ 1, 20, 16 }, 81);
    const w = try makeInput(&g, mgr, &.{ 3, 1, 16 }, &.{ 3, 1, 16 }, 82);
    const out = try g.addConv1D(x, w, null, 2, 1, 1, 0, 16);
    return finishProgGpuTiled(alloc, &g, mgr, out);
}

test "gpu backend: conv1d (depthwise stride 2, batched) matches CPU" {
    try expectGpuMatchesCpu(buildConv1DDepthwiseStride2, 2 * 10 * 16, 1e-4);
}

fn buildConv2D(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    // Asymmetric strides/dilations/pads with bias:
    // x [1, 10, 12, 3], w [3, 3, 3, 5] -> out [1, 5, 10, 5].
    const x = try makeInput(&g, mgr, &.{ 1, 10, 12, 3 }, &.{ 1, 10, 12, 3 }, 75);
    const w = try makeInput(&g, mgr, &.{ 3, 3, 3, 5 }, &.{ 3, 3, 3, 5 }, 76);
    const b = try makeInput(&g, mgr, &.{5}, &.{5}, 77);
    const out = try g.addConv2D(x, w, b, 2, 1, 1, 2, 1, 1, 2, 0, 1);
    return finishProgGpuTiled(alloc, &g, mgr, out);
}

test "gpu backend: conv2d matches CPU" {
    try expectGpuMatchesCpu(buildConv2D, 1 * 5 * 10 * 5, 1e-4);
}

fn buildArgMax(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    // Wide rows (> one 256-thread sweep) exercise the strided scan + tie-break.
    const x = try makeInput(&g, mgr, &.{ 6, 700 }, &.{ 6, 700 }, 80);
    const out = try g.addArgMax(x, -1);
    return finishProgGpuTiled(alloc, &g, mgr, out);
}

// RNNT joint shape: one row of 1025 (vocab+blank). Exercises argmax over an odd
// wide row where the winner can be the LAST index (the blank) — the decode's
// emit-vs-blank decision. `spike_at` (< 0 = none) forces a clear max there.
fn buildArgMax1025(alloc: std.mem.Allocator, mgr: *StorageManager, spike_at: i64) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    const n: usize = 1025;
    const id = try mgr.createTiledTensor(.f32, &.{ 1, n }, &.{ 1, n }, .{});
    const data = try mgr.allocator.alloc(f32, n);
    defer mgr.allocator.free(data);
    for (data, 0..) |*v, i| {
        const k: u32 = @intCast((i * 2654435761 + 88) % 1000);
        v.* = (@as(f32, @floatFromInt(k)) - 500.0) * 0.004;
    }
    if (spike_at >= 0) data[@intCast(spike_at)] = 99.0;
    try mgr.writeFromPackedScalar(id, std.mem.sliceAsBytes(data));
    const x = try g.addInput(.f32, &.{ 1, n });
    try g.bindExternal(x, id);
    const out = try g.addArgMax(x, -1);
    return finishProgGpuTiled(alloc, &g, mgr, out);
}
fn buildArgMax1025Plain(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    return buildArgMax1025(alloc, mgr, -1);
}
fn buildArgMax1025Blank(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    return buildArgMax1025(alloc, mgr, 1024); // max at the last (blank) index
}
test "gpu backend: argmax [1,1025] matches CPU" {
    try expectGpuMatchesCpuI32(buildArgMax1025Plain, 1);
}
test "gpu backend: argmax [1,1025] winner at last index matches CPU" {
    try expectGpuMatchesCpuI32(buildArgMax1025Blank, 1);
}

test "gpu backend: argmax (last axis) matches CPU" {
    // i32 outputs are compared as raw bits (identical or fail).
    try expectGpuMatchesCpu(buildArgMax, 6, 0.0);
}

fn buildScatterRow(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    const buf = try makeInput(&g, mgr, &.{ 8, 16 }, &.{ 8, 16 }, 81);
    const idx = try makeInputI32(&g, mgr, &.{1}, &.{1}, &.{5});
    const src = try makeInput(&g, mgr, &.{16}, &.{16}, 82);
    const out = try g.addScatterRow(buf, idx, src);
    return finishProgGpuTiled(alloc, &g, mgr, out);
}

test "gpu backend: scatter-row matches CPU" {
    try expectGpuMatchesCpu(buildScatterRow, 8 * 16, 0.0);
}

// Regression guard: the scatter destination index is COMPUTED ON DEVICE (the
// decode-loop emit case). The device scatter kernel reads it on-GPU, so the row
// must land at the freshly computed index, not a stale one.
fn buildScatterDeviceIdx(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    const buf = try makeInput(&g, mgr, &.{ 8, 16 }, &.{ 8, 16 }, 81);
    const base = try makeInputI32(&g, mgr, &.{1}, &.{1}, &.{5});
    const zero = try makeInputI32(&g, mgr, &.{1}, &.{1}, &.{0});
    const idx = try g.addElemwiseBinary(.add, base, zero); // device-computed index
    const src = try makeInput(&g, mgr, &.{16}, &.{16}, 82);
    const out = try g.addScatterRow(buf, idx, src);
    return finishProgGpuTiled(alloc, &g, mgr, out);
}

test "gpu backend: scatter-row with device-computed index matches CPU" {
    try expectGpuMatchesCpu(buildScatterDeviceIdx, 8 * 16, 0.0);
}

// ---- lstm / fft / stft / rel-pos mha ------------------------------------------

fn buildLSTM(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    // batch=4, input_size=8, hidden=16 -> state [4, 32].
    const x = try makeInput(&g, mgr, &.{ 4, 8 }, &.{ 4, 8 }, 90);
    const h = try makeInput(&g, mgr, &.{ 4, 16 }, &.{ 4, 16 }, 91);
    const cc = try makeInput(&g, mgr, &.{ 4, 16 }, &.{ 4, 16 }, 92);
    const w_ih = try makeInput(&g, mgr, &.{ 8, 64 }, &.{ 8, 64 }, 93);
    const w_hh = try makeInput(&g, mgr, &.{ 16, 64 }, &.{ 16, 64 }, 94);
    const b_ih = try makeInput(&g, mgr, &.{64}, &.{64}, 95);
    const b_hh = try makeInput(&g, mgr, &.{64}, &.{64}, 96);
    const out = try g.addLSTMCell(x, h, cc, w_ih, w_hh, b_ih, b_hh);
    return finishProgGpuTiled(alloc, &g, mgr, out);
}

test "gpu backend: lstm cell (fused, biased) matches CPU" {
    // CPU uses sigmoid/tanh fast approximations; GPU uses exact builtins.
    try expectGpuMatchesCpu(buildLSTM, 4 * 32, 2e-3);
}

fn buildRFFT(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    const x = try makeInput(&g, mgr, &.{ 3, 64 }, &.{ 3, 64 }, 100);
    const out = try g.addRFFT(x);
    return finishProgGpuTiled(alloc, &g, mgr, out);
}

test "gpu backend: rfft matches CPU" {
    // CPU runs an O(n log n) FFT plan, GPU a direct DFT — same math, different
    // rounding order.
    try expectGpuMatchesCpu(buildRFFT, 3 * 66, 1e-3);
}

fn buildSTFT(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    // batch=2, samples=256, n_fft=64, hop=32, center -> [2, 9, 66].
    const signal = try makeInput(&g, mgr, &.{ 2, 256 }, &.{ 2, 256 }, 101);
    const window = try makeInput(&g, mgr, &.{64}, &.{64}, 102);
    const out = try g.addSTFT(signal, window, 64, 32, true);
    return finishProgGpuTiled(alloc, &g, mgr, out);
}

test "gpu backend: stft (centered) matches CPU" {
    try expectGpuMatchesCpu(buildSTFT, 2 * 9 * 66, 1e-3);
}

fn buildSTFTAudio(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    // ASR front-end shape: 1s of 16 kHz audio, n_fft=512, hop=160 -> [1, 101, 514].
    const signal = try makeInput(&g, mgr, &.{ 1, 16000 }, &.{ 1, 16000 }, 110);
    const window = try makeInput(&g, mgr, &.{512}, &.{512}, 111);
    const out = try g.addSTFT(signal, window, 512, 160, true);
    return finishProgGpuTiled(alloc, &g, mgr, out);
}

test "gpu backend: stft (asr-scale, hop not dividing n_fft) matches CPU" {
    try expectGpuMatchesCpu(buildSTFTAudio, 1 * 101 * 514, 2e-3);
}


fn buildRelPosMHA(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    // [B, T, H, D] with per-head tiles (the layout the CPU exec uses too).
    // B=2, T=10, H=2, D=16; pos_emb [H, 2T-1, D]; additive mask [T, T].
    const q = try makeInput(&g, mgr, &.{ 2, 10, 2, 16 }, &.{ 1, 10, 1, 16 }, 103);
    const k = try makeInput(&g, mgr, &.{ 2, 10, 2, 16 }, &.{ 1, 10, 1, 16 }, 104);
    const v = try makeInput(&g, mgr, &.{ 2, 10, 2, 16 }, &.{ 1, 10, 1, 16 }, 105);
    const pe = try makeInput(&g, mgr, &.{ 2, 19, 16 }, &.{ 1, 19, 16 }, 106);
    const u = try makeInput(&g, mgr, &.{ 2, 16 }, &.{ 2, 16 }, 107);
    const vb = try makeInput(&g, mgr, &.{ 2, 16 }, &.{ 2, 16 }, 108);
    const mask = try makeInput(&g, mgr, &.{ 10, 10 }, &.{ 10, 10 }, 109);
    const out = try g.addRelPosMHA(q, k, v, pe, u, vb, mask, 0.25, 2);
    return finishProgGpuTiled(alloc, &g, mgr, out);
}

test "gpu backend: rel-pos mha (masked) matches CPU" {
    try expectGpuMatchesCpu(buildRelPosMHA, 2 * 10 * 2 * 16, 2e-3);
}

// ---- concat / views -------------------------------------------------------------

fn buildConcat(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    // Three inputs on the inner axis (outer=4): [4,4] ++ [4,8] ++ [4,4] -> [4,16].
    const a = try makeInput(&g, mgr, &.{ 4, 4 }, &.{ 4, 4 }, 120);
    const b = try makeInput(&g, mgr, &.{ 4, 8 }, &.{ 4, 8 }, 121);
    const cc = try makeInput(&g, mgr, &.{ 4, 4 }, &.{ 4, 4 }, 122);
    const out = try g.addConcat(&.{ a, b, cc }, 1);
    return finishProgGpuTiled(alloc, &g, mgr, out);
}

test "gpu backend: concat (inner axis, 3 inputs) matches CPU" {
    try expectGpuMatchesCpu(buildConcat, 4 * 16, 0.0);
}

fn buildViewChain(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    // unsqueeze -> squeeze -> reshape -> transpose2d -> slice: covers the
    // ReshapeScalar, Transpose2DScalar, and SliceNDScalar exec paths.
    const x = try makeInput(&g, mgr, &.{ 6, 8 }, &.{ 6, 8 }, 123);
    const un = try g.addViewUnsqueeze(x, 0); // [1, 6, 8]
    const sq = try g.addViewSqueeze(un, 0); // [6, 8]
    const rs = try g.addViewReshape(sq, &.{ 4, 12 });
    const tr = try g.addViewTranspose2D(rs); // [12, 4]
    const sl = try g.addViewSliceND(tr, &.{ 2, 1 }, &.{ 8, 2 }); // [8, 2]
    return finishProgGpuTiled(alloc, &g, mgr, sl);
}

test "gpu backend: view chain (reshape/transpose/slice) matches CPU" {
    try expectGpuMatchesCpu(buildViewChain, 8 * 2, 0.0);
}

// Contiguous-slab reshape fast path: a MULTI-tile input whose tiles are each a
// contiguous packed slab (row-tiled leading dim, whole trailing dims) reshaped
// through the packed representation. Mirrors the encoder's [1,T,1024] <-> per-time
// tiling that dominated GPU time before gatherScatterTiles used a buffer copy
// instead of the strided gather/scatter kernel. Both backends run the identical
// program, so the result must be byte-exact.
fn buildReshapeContiguousMultiTile(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    // Input tiled [1,6,8] -> 4 contiguous 48-element tiles along dim 0.
    const x = try makeInput(&g, mgr, &.{ 4, 6, 8 }, &.{ 1, 6, 8 }, 77);
    const rs = try g.addViewReshape(x, &.{ 4, 48 }); // regroup whole trailing dims
    const back = try g.addViewReshape(rs, &.{ 4, 6, 8 });
    const y = try g.addElemwiseBinary(.add, back, x);
    return finishProgGpuTiled(alloc, &g, mgr, y);
}

test "gpu backend: contiguous multi-tile reshape matches CPU" {
    try expectGpuMatchesCpu(buildReshapeContiguousMultiTile, 4 * 6 * 8, 0.0);
}

// Slice a MULTI-tile tensor along a dim OTHER than its split dim: input tiled by
// dim 2 (the "head" axis) but sliced along dim 1 (the "time" axis). The single-
// split-dim fast path can't express this, so it packs to a contiguous scratch and
// gathers the slice out. Mirrors the streaming attention-cache "keep last N frames"
// update ([1,84,8,128] tiled by head, sliced along time).
fn buildSliceNonSplitDim(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    // [1,6,4,8] tiled [1,6,1,8] -> 4 tiles split along dim 2.
    const x = try makeInput(&g, mgr, &.{ 1, 6, 4, 8 }, &.{ 1, 6, 1, 8 }, 88);
    const sl = try g.addViewSliceND(x, &.{ 0, 2, 0, 0 }, &.{ 1, 4, 4, 8 }); // slice dim 1: [2,6)
    return finishProgGpuTiled(alloc, &g, mgr, sl);
}

test "gpu backend: slice multi-tile along non-split dim matches CPU" {
    try expectGpuMatchesCpu(buildSliceNonSplitDim, 1 * 4 * 4 * 8, 0.0);
}

// The RNNT decoder splits its fused LSTM output [1, 2H] into h=[1,H] (offset 0)
// and c=[1,H] (offset H). Exercise the second-half slice specifically (H=640).
fn buildSliceSecondHalf(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    const x = try makeInput(&g, mgr, &.{ 1, 1280 }, &.{ 1, 1280 }, 30);
    const out = try g.addViewSliceND(x, &.{ 0, 640 }, &.{ 1, 640 });
    return finishProgGpuTiled(alloc, &g, mgr, out);
}
test "gpu backend: slice second half [1,1280]->[1,640] matches CPU" {
    try expectGpuMatchesCpu(buildSliceSecondHalf, 640, 0.0);
}

// The RNNT decoder's state gate multiplies a per-hidden delta [H,1] by the scalar
// emit flag [1] (BroadcastLastDim with a size-1 last dim on the left operand).
fn buildBcastMulColVec(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    const a = try makeInput(&g, mgr, &.{ 640, 1 }, &.{ 640, 1 }, 31);
    const b = try makeInput(&g, mgr, &.{1}, &.{1}, 32);
    const out = try g.addBroadcastLastDimBinary(.mul, a, b);
    return finishProgGpuTiled(alloc, &g, mgr, out);
}
test "gpu backend: bcast_mul [640,1]x[1] matches CPU" {
    try expectGpuMatchesCpu(buildBcastMulColVec, 640, 1e-5);
}

// ---- i32 elementwise / cast -------------------------------------------------------

/// i32 input with a deterministic pattern in [-500, 500), including zeros
/// (every 9th element) so div-by-zero handling is exercised.
fn makeInputI32Pattern(g: *Graph, mgr: *StorageManager, shape: []const usize, tile: []const usize, seed: u32) !aion.graph.ValueId {
    const id = try mgr.createTiledTensor(.i32, shape, tile, .{});
    var n: usize = 1;
    for (shape) |d| n *= d;
    const data = try mgr.allocator.alloc(i32, n);
    defer mgr.allocator.free(data);
    for (data, 0..) |*v, i| {
        const k: u32 = @intCast((i * 2654435761 + seed) % 1000);
        v.* = if (i % 9 == 0) 0 else @as(i32, @intCast(k)) - 500;
    }
    try mgr.writeFromPackedScalar(id, std.mem.sliceAsBytes(data));
    const v = try g.addInput(.i32, shape);
    try g.bindExternal(v, id);
    return v;
}

fn buildI32Arith(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    const a = try makeInput32x2(&g, mgr, 140);
    const b = try makeInput32x2(&g, mgr, 141);
    const cc = try makeInput32x2(&g, mgr, 142); // contains zeros -> div yields 0 there
    const sum = try g.addElemwiseBinary(.add, a, b);
    const prod = try g.addElemwiseBinary(.mul, sum, b);
    const diff = try g.addElemwiseBinary(.sub, prod, a);
    const quot = try g.addElemwiseBinary(.div, diff, cc);
    return finishProgGpuTiled(alloc, &g, mgr, quot);
}

fn makeInput32x2(g: *Graph, mgr: *StorageManager, seed: u32) !aion.graph.ValueId {
    return makeInputI32Pattern(g, mgr, &.{ 8, 32 }, &.{ 8, 32 }, seed);
}

test "gpu backend: i32 elementwise arithmetic (incl. div-by-zero) matches CPU" {
    try expectGpuMatchesCpuI32(buildI32Arith, 8 * 32);
}

fn buildI32Compare(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    const a = try makeInput32x2(&g, mgr, 143);
    const b = try makeInput32x2(&g, mgr, 144);
    const lt = try g.addElemwiseBinary(.lt, a, b);
    const le = try g.addElemwiseBinary(.le, a, b);
    const gt = try g.addElemwiseBinary(.gt, a, b);
    const ge = try g.addElemwiseBinary(.ge, a, b);
    const ne = try g.addElemwiseBinary(.ne, lt, gt);
    const both = try g.addElemwiseBinary(.mul, le, ge); // 1 iff a == b
    const eq = try g.addElemwiseBinary(.eq, ne, both);
    return finishProgGpuTiled(alloc, &g, mgr, eq);
}

test "gpu backend: i32 comparisons match CPU" {
    try expectGpuMatchesCpu(buildI32Compare, 8 * 32, 0.0);
}

fn buildCastI32Roundtrip(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    const x = try makeInput(&g, mgr, &.{ 8, 32 }, &.{ 8, 32 }, 145);
    const as_i32 = try g.addCast(x, .i32); // round-to-nearest
    const back = try g.addCast(as_i32, .f32);
    return finishProgGpuTiled(alloc, &g, mgr, back);
}

test "gpu backend: cast f32->i32->f32 (round-to-nearest) matches CPU" {
    try expectGpuMatchesCpu(buildCastI32Roundtrip, 8 * 32, 0.0);
}

// ---- control flow: If / Loop -----------------------------------------------------

fn makeInputF32(g: *Graph, mgr: *StorageManager, shape: []const usize, tile: []const usize, vals: []const f32) !aion.graph.ValueId {
    const id = try mgr.createTiledTensor(.f32, shape, tile, .{});
    try mgr.writeFromPackedScalar(id, std.mem.sliceAsBytes(vals));
    const v = try g.addInput(.f32, shape);
    try g.bindExternal(v, id);
    return v;
}

/// If with a device-computed predicate: argmax over a 2-vector yields the i32
/// cond (1 -> then, 0 -> else), exercising the submit/flush split + host read
/// of a GPU-produced scalar. Branch bodies do real compute (add vs mul).
fn buildIfCommon(alloc: std.mem.Allocator, mgr: *StorageManager, comptime take_then: bool) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    const a = try makeInput(&g, mgr, &.{ 4, 8 }, &.{ 4, 8 }, 130);
    const b = try makeInput(&g, mgr, &.{ 4, 8 }, &.{ 4, 8 }, 131);
    const sel_vals: [2]f32 = if (take_then) .{ 0.5, 1.5 } else .{ 1.5, 0.5 };
    const sel = try makeInputF32(&g, mgr, &.{2}, &.{2}, &sel_vals);
    const cond = try g.addArgMax(sel, -1);
    try g.beginRegion();
    const then_v = try g.addElemwiseBinary(.add, a, b);
    const then_r = try g.endRegion(&.{then_v});
    try g.beginRegion();
    const else_v = try g.addElemwiseBinary(.mul, a, b);
    const else_r = try g.endRegion(&.{else_v});
    const out = try g.addIf(cond, then_r, else_r);
    return finishProgGpuTiled(alloc, &g, mgr, out);
}

fn buildIfThen(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    return buildIfCommon(alloc, mgr, true);
}
fn buildIfElse(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    return buildIfCommon(alloc, mgr, false);
}

test "gpu backend: if (then branch, gpu-computed cond) matches CPU" {
    try expectGpuMatchesCpu(buildIfThen, 4 * 8, 1e-6);
}

test "gpu backend: if (else branch, gpu-computed cond) matches CPU" {
    try expectGpuMatchesCpu(buildIfElse, 4 * 8, 1e-6);
}

/// Fixed-trip loop, no cond: the whole unrolled body records into one frame
/// with zero mid-program syncs; carried state advances via tensor-id swaps.
fn buildLoopFixed(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    const carried = try makeInput(&g, mgr, &.{ 4, 8 }, &.{ 4, 8 }, 132);
    const inc = try makeInput(&g, mgr, &.{ 4, 8 }, &.{ 4, 8 }, 133);
    try g.beginRegion();
    const next = try g.addElemwiseBinary(.add, carried, inc);
    const body = try g.endRegion(&.{next});
    const out = try g.addLoop(carried, body, 4);
    return finishProgGpuTiled(alloc, &g, mgr, out);
}

test "gpu backend: loop (fixed trip count) matches CPU" {
    try expectGpuMatchesCpu(buildLoopFixed, 4 * 8, 1e-6);
}

/// Multi-carry loop with an early-exit predicate the GPU itself computes:
/// vec starts {0.0, 1.1} and gains {0.5, 0} per iteration, so argmax(vec)
/// flips from 1 to 0 on the third trip — the loop must stop there (not at
/// static_max=100). Exercises per-iteration flush + D2H predicate reads.
fn buildLoopEarlyExit(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    const vec = try makeInputF32(&g, mgr, &.{2}, &.{2}, &.{ 0.0, 1.1 });
    const delta = try makeInputF32(&g, mgr, &.{2}, &.{2}, &.{ 0.5, 0.0 });
    const active = try makeInputI32(&g, mgr, &.{1}, &.{1}, &.{1});
    const acc = try makeInput(&g, mgr, &.{ 4, 8 }, &.{ 4, 8 }, 134);
    const ten = try makeInput(&g, mgr, &.{ 4, 8 }, &.{ 4, 8 }, 135);
    try g.beginRegion();
    const vec_next = try g.addElemwiseBinary(.add, vec, delta);
    const active_next = try g.addArgMax(vec_next, -1);
    const acc_next = try g.addElemwiseBinary(.add, acc, ten);
    const body = try g.endRegion(&.{ vec_next, active_next, acc_next });
    const outs = try g.addLoopMulti(&.{ vec, active, acc }, body, 100, 1, true);
    return finishProgGpuTiled(alloc, &g, mgr, outs[2]); // final acc: init + 3 * ten
}

test "gpu backend: loop (multi-carry, gpu-computed early exit) matches CPU" {
    try expectGpuMatchesCpu(buildLoopEarlyExit, 4 * 8, 1e-6);
}

/// The canonical decode-loop shape: i32 counter incremented in-body, `lt`
/// comparison as the continue predicate — all computed by the GPU's i32
/// elementwise kernels (5 iterations, static_max=100).
fn buildLoopCounter(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    var g = Graph.init(alloc);
    defer g.deinit();
    const iv = try makeInputI32(&g, mgr, &.{1}, &.{1}, &.{0});
    const one = try makeInputI32(&g, mgr, &.{1}, &.{1}, &.{1});
    const limit = try makeInputI32(&g, mgr, &.{1}, &.{1}, &.{5});
    const active = try makeInputI32(&g, mgr, &.{1}, &.{1}, &.{1});
    const acc = try makeInput(&g, mgr, &.{ 4, 8 }, &.{ 4, 8 }, 150);
    const ten = try makeInput(&g, mgr, &.{ 4, 8 }, &.{ 4, 8 }, 151);
    try g.beginRegion();
    const i_next = try g.addElemwiseBinary(.add, iv, one);
    const active_next = try g.addElemwiseBinary(.lt, i_next, limit);
    const acc_next = try g.addElemwiseBinary(.add, acc, ten);
    const body = try g.endRegion(&.{ i_next, active_next, acc_next });
    const outs = try g.addLoopMulti(&.{ iv, active, acc }, body, 100, 1, true);
    return finishProgGpuTiled(alloc, &g, mgr, outs[2]); // final acc: init + 5 * ten
}

test "gpu backend: loop (i32 counter + lt predicate on gpu) matches CPU" {
    try expectGpuMatchesCpu(buildLoopCounter, 4 * 8, 1e-6);
}

test "gpu backend: residency persists across session.execute calls" {
    const alloc = std.testing.allocator;

    var device = wgpu.Gpu.init(.{ .power = .high }) catch return error.SkipZigTest;
    defer device.deinit();

    var gb = gpu.GpuBackend.init(alloc, &device);
    defer gb.deinit();

    // CPU reference for silu(a+b).
    var cpu_result: [COUNT]f32 = undefined;
    {
        var mgr = StorageManager.init(alloc);
        defer mgr.deinit();
        var built = try buildProgram(alloc, &mgr);
        defer built.prog.deinit();
        var cpu = aion.cpu.CpuBackend.init(alloc);
        defer cpu.deinit();
        try cpu.backend().executeProgram(&built.prog, mgr.tensorStore());
        try readOutput(&mgr, built.out, cpu_result[0..]);
    }

    // One Session executed twice. The second run reuses the session's persistent
    // ResidentTensorStore (weights/inputs stay device-resident across execute
    // calls); both runs must still match the CPU reference.
    var mgr = StorageManager.init(alloc);
    defer mgr.deinit();
    var built = try buildProgram(alloc, &mgr);
    defer built.prog.deinit();

    var session = try gb.backend().createSession(mgr.tensorStore());
    defer session.deinit();

    var run: usize = 0;
    while (run < 2) : (run += 1) {
        var gpu_result: [COUNT]f32 = undefined;
        try session.execute(&built.prog);
        try readOutput(&mgr, built.out, gpu_result[0..]);
        for (0..COUNT) |i| {
            try std.testing.expectApproxEqAbs(cpu_result[i], gpu_result[i], 1e-4);
        }
    }
}

// ---------------------------------------------------------------------------
// Control flow (If / Loop). Exercises the GPU predicate-read paths end-to-end
// against the CPU reference: a DEVICE-produced predicate (submit-without-poll
// then a single D2H poll), a HOST-resident predicate (read directly, no GPU
// stall, pending work stays batched), the zero-sync fixed-trip unrolled loop,
// and a per-iteration device predicate with loop-carried residency swaps.
// ---------------------------------------------------------------------------

fn writeI32Scalar(mgr: *StorageManager, id: TensorId, v: i32) !void {
    try mgr.writeFromPackedScalar(id, std.mem.asBytes(&v));
}
fn writeF32Scalar(mgr: *StorageManager, id: TensorId, v: f32) !void {
    try mgr.writeFromPackedScalar(id, std.mem.asBytes(&v));
}

/// If with a DEVICE-produced predicate: `cond = (x < y)` is computed on device,
/// so reading it drives the submit + single-poll readback path.
fn buildIfDeviceCond(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    const s1 = [_]usize{1};
    const x_id = try mgr.createTiledTensor(.i32, &s1, &s1, .{});
    const y_id = try mgr.createTiledTensor(.i32, &s1, &s1, .{});
    const a_id = try mgr.createTiledTensor(.f32, &s1, &s1, .{});
    const b_id = try mgr.createTiledTensor(.f32, &s1, &s1, .{});
    try writeI32Scalar(mgr, x_id, 1);
    try writeI32Scalar(mgr, y_id, 2); // 1 < 2 -> take `then`
    try writeF32Scalar(mgr, a_id, 3.0);
    try writeF32Scalar(mgr, b_id, 4.0);

    var g = Graph.init(alloc);
    defer g.deinit();
    const x = try g.addInput(.i32, &s1);
    try g.bindExternal(x, x_id);
    const y = try g.addInput(.i32, &s1);
    try g.bindExternal(y, y_id);
    const a = try g.addInput(.f32, &s1);
    try g.bindExternal(a, a_id);
    const b = try g.addInput(.f32, &s1);
    try g.bindExternal(b, b_id);
    const cond = try g.addElemwiseBinary(.lt, x, y); // device-produced i32[1]

    try g.beginRegion();
    const then_v = try g.addElemwiseBinary(.add, a, b);
    const then_region = try g.endRegion(&[_]aion.graph.ValueId{then_v});
    try g.beginRegion();
    const else_v = try g.addElemwiseBinary(.sub, a, b);
    const else_region = try g.endRegion(&[_]aion.graph.ValueId{else_v});
    const out = try g.addIf(cond, then_region, else_region);
    try g.setOutputs(&[_]aion.graph.ValueId{out});

    const prog = try aion.program.compileGraph(alloc, &g, mgr, .{});
    return .{ .prog = prog, .out = prog.outputs[0] };
}

test "gpu backend: if (device-produced predicate) matches CPU" {
    try expectGpuMatchesCpu(buildIfDeviceCond, 1, 1e-5);
}

/// If with a HOST-resident predicate and unrelated device work already pending
/// (`pre = silu(a)`): the predicate is read directly with no GPU stall while the
/// pending compute stays batched — the clean-skip path.
fn buildIfHostCond(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    const s1 = [_]usize{1};
    const cond_id = try mgr.createTiledTensor(.i32, &s1, &s1, .{});
    const a_id = try mgr.createTiledTensor(.f32, &s1, &s1, .{});
    const b_id = try mgr.createTiledTensor(.f32, &s1, &s1, .{});
    try writeI32Scalar(mgr, cond_id, 1); // host predicate -> take `then`
    try writeF32Scalar(mgr, a_id, 1.5);
    try writeF32Scalar(mgr, b_id, 0.25);

    var g = Graph.init(alloc);
    defer g.deinit();
    const cond = try g.addInput(.i32, &s1);
    try g.bindExternal(cond, cond_id);
    const a = try g.addInput(.f32, &s1);
    try g.bindExternal(a, a_id);
    const b = try g.addInput(.f32, &s1);
    try g.bindExternal(b, b_id);
    const pre = try g.addUnary(.silu, a); // device work pending before the If

    try g.beginRegion();
    const then_v = try g.addElemwiseBinary(.add, pre, b);
    const then_region = try g.endRegion(&[_]aion.graph.ValueId{then_v});
    try g.beginRegion();
    const else_v = try g.addElemwiseBinary(.sub, pre, b);
    const else_region = try g.endRegion(&[_]aion.graph.ValueId{else_v});
    const out = try g.addIf(cond, then_region, else_region);
    try g.setOutputs(&[_]aion.graph.ValueId{out});

    const prog = try aion.program.compileGraph(alloc, &g, mgr, .{});
    return .{ .prog = prog, .out = prog.outputs[0] };
}

test "gpu backend: if (host-resident predicate) matches CPU" {
    try expectGpuMatchesCpu(buildIfHostCond, 1, 1e-5);
}

/// Fixed-trip Loop (no cond): the whole unrolled body records into one frame and
/// submits once — zero predicate syncs — while carried state is swapped
/// device-side between iterations. Result: 1 + 2*4 = 9.
fn buildFixedTripLoop(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    const s1 = [_]usize{1};
    const carried_id = try mgr.createTiledTensor(.f32, &s1, &s1, .{});
    const inc_id = try mgr.createTiledTensor(.f32, &s1, &s1, .{});
    try writeF32Scalar(mgr, carried_id, 1.0);
    try writeF32Scalar(mgr, inc_id, 2.0);

    var g = Graph.init(alloc);
    defer g.deinit();
    const carried = try g.addInput(.f32, &s1);
    try g.bindExternal(carried, carried_id);
    const inc = try g.addInput(.f32, &s1);
    try g.bindExternal(inc, inc_id);
    try g.beginRegion();
    const next = try g.addElemwiseBinary(.add, carried, inc);
    const body = try g.endRegion(&[_]aion.graph.ValueId{next});
    const out = try g.addLoop(carried, body, 4);
    try g.setOutputs(&[_]aion.graph.ValueId{out});

    const prog = try aion.program.compileGraph(alloc, &g, mgr, .{});
    return .{ .prog = prog, .out = prog.outputs[0] };
}

test "gpu backend: fixed-trip loop matches CPU" {
    try expectGpuMatchesCpu(buildFixedTripLoop, 1, 1e-5);
}

/// Multi-carry Loop with a DEVICE-computed early-exit predicate re-evaluated each
/// iteration — the hot path: one submit + one D2H poll per iteration to read the
/// continue flag, carried state swapped device-side between iterations. Carries
/// i(i32)/acc(f32)/active(i32); stops once i reaches limit=3, so acc = 3*10 = 30.
fn buildEarlyExitLoop(alloc: std.mem.Allocator, mgr: *StorageManager) !BuiltProg {
    const s1 = [_]usize{1};
    const i_id = try mgr.createTiledTensor(.i32, &s1, &s1, .{});
    const acc_id = try mgr.createTiledTensor(.f32, &s1, &s1, .{});
    const active_id = try mgr.createTiledTensor(.i32, &s1, &s1, .{});
    const one_id = try mgr.createTiledTensor(.i32, &s1, &s1, .{});
    const ten_id = try mgr.createTiledTensor(.f32, &s1, &s1, .{});
    const limit_id = try mgr.createTiledTensor(.i32, &s1, &s1, .{});
    try writeI32Scalar(mgr, i_id, 0);
    try writeF32Scalar(mgr, acc_id, 0.0);
    try writeI32Scalar(mgr, active_id, 1);
    try writeI32Scalar(mgr, one_id, 1);
    try writeF32Scalar(mgr, ten_id, 10.0);
    try writeI32Scalar(mgr, limit_id, 3);

    var g = Graph.init(alloc);
    defer g.deinit();
    const i_in = try g.addInput(.i32, &s1);
    try g.bindExternal(i_in, i_id);
    const acc_in = try g.addInput(.f32, &s1);
    try g.bindExternal(acc_in, acc_id);
    const active_in = try g.addInput(.i32, &s1);
    try g.bindExternal(active_in, active_id);
    const one = try g.addInput(.i32, &s1);
    try g.bindExternal(one, one_id);
    const ten = try g.addInput(.f32, &s1);
    try g.bindExternal(ten, ten_id);
    const limit = try g.addInput(.i32, &s1);
    try g.bindExternal(limit, limit_id);

    try g.beginRegion();
    const i_next = try g.addElemwiseBinary(.add, i_in, one);
    const acc_next = try g.addElemwiseBinary(.add, acc_in, ten);
    const active_next = try g.addElemwiseBinary(.lt, i_next, limit); // device i32 {0,1}
    const body = try g.endRegion(&[_]aion.graph.ValueId{ i_next, acc_next, active_next });
    const outs = try g.addLoopMulti(
        &[_]aion.graph.ValueId{ i_in, acc_in, active_in },
        body,
        100,
        2, // cond carry = `active`
        true, // check before body
    );
    try g.setOutputs(&[_]aion.graph.ValueId{outs[1]}); // acc

    const prog = try aion.program.compileGraph(alloc, &g, mgr, .{});
    return .{ .prog = prog, .out = prog.outputs[0] };
}

test "gpu backend: early-exit loop (device predicate) matches CPU" {
    try expectGpuMatchesCpu(buildEarlyExitLoop, 1, 1e-5);
}
