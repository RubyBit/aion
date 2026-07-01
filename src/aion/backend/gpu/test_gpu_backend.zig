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
    var device = wgpu.Gpu.init(.{}) catch return error.SkipZigTest;
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

    var device = wgpu.Gpu.init(.{}) catch return error.SkipZigTest;
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

test "gpu backend: residency persists across executeProgram calls" {
    const alloc = std.testing.allocator;

    var device = wgpu.Gpu.init(.{}) catch return error.SkipZigTest;
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

    // Same backend + same store, executed twice. The second run reuses the
    // persistent ResidentTensorStore (weights/inputs stay device-resident); both
    // runs must still match the CPU reference.
    var mgr = StorageManager.init(alloc);
    defer mgr.deinit();
    var built = try buildProgram(alloc, &mgr);
    defer built.prog.deinit();
    const store = mgr.tensorStore();

    var run: usize = 0;
    while (run < 2) : (run += 1) {
        var gpu_result: [COUNT]f32 = undefined;
        try gb.backend().executeProgram(&built.prog, store);
        try readOutput(&mgr, built.out, gpu_result[0..]);
        for (0..COUNT) |i| {
            try std.testing.expectApproxEqAbs(cpu_result[i], gpu_result[i], 1e-4);
        }
    }
}
