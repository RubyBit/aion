// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! CPU-vs-GPU benchmark for the WebGPU backend. Compiles the same matmul (and a
//! couple of elementwise ops) with the GPU `TilePolicy` and times it on both the
//! CPU backend and the `GpuBackend`, reporting GFLOP/s, ms/iter, and the speedup.
//!
//! Built as its own artifact (it links wgpu-native), run via `zig build gpu-bench`.
//! Persistent residency means weights upload once, so iters >= 2 measure steady
//! state (end-to-end, including the output device->host readback).
//!
//! Run:
//!   zig build gpu-bench -Doptimize=ReleaseFast -- --m 1024 --n 1024 --k 1024 --iters 50

const std = @import("std");
const aion = @import("aion");

const gpu = aion.gpu; // feature-gated GPU backend (root.zig exposes it when -Dgpu)
const wgpu = gpu.wgpu;

const Backend = aion.backend.Backend;
const StorageManager = aion.storage_manager.StorageManager;
const Graph = aion.graph.Graph;
const TensorId = aion.storage_manager.TensorId;
const plan = aion.plan;

const out = std.debug;

const Args = struct {
    m: usize = 1024,
    n: usize = 1024,
    k: usize = 1024,
    iters: usize = 50,
    gpu_only: bool = false,
    // GPU selection. Defaults to high-power so the discrete GPU is benched (the
    // default adapter is often the weak integrated one). Override: --low,
    // --adapter=N, --backend=vulkan|d3d12|metal.
    opts: wgpu.Options = .{ .power = .high },
};

/// Accept both `--m 2048` (space-separated, like the main bench) and `--m=2048`.
/// `valueFor` returns the inline `=value` or pulls the next token from `it`.
fn valueFor(arg: []const u8, key: []const u8, it: anytype) ?[]const u8 {
    if (std.mem.eql(u8, arg, key)) return it.next();
    var buf: [32]u8 = undefined;
    const pfx = std.fmt.bufPrint(&buf, "{s}=", .{key}) catch return null;
    if (std.mem.startsWith(u8, arg, pfx)) return arg[pfx.len..];
    return null;
}

fn applyArg(a: *Args, arg: []const u8, it: anytype) void {
    if (valueFor(arg, "--m", it)) |v| a.m = std.fmt.parseInt(usize, v, 10) catch a.m;
    if (valueFor(arg, "--n", it)) |v| a.n = std.fmt.parseInt(usize, v, 10) catch a.n;
    if (valueFor(arg, "--k", it)) |v| a.k = std.fmt.parseInt(usize, v, 10) catch a.k;
    if (valueFor(arg, "--iters", it)) |v| a.iters = std.fmt.parseInt(usize, v, 10) catch a.iters;
    if (std.mem.eql(u8, arg, "--gpu-only")) a.gpu_only = true;
    if (std.mem.eql(u8, arg, "--high")) a.opts.power = .high;
    if (std.mem.eql(u8, arg, "--low")) a.opts.power = .low;
    if (valueFor(arg, "--adapter", it)) |v| a.opts.adapter_index = std.fmt.parseInt(usize, v, 10) catch null;
    if (valueFor(arg, "--backend", it)) |v| {
        a.opts.backend = if (std.mem.eql(u8, v, "vulkan")) .vulkan else if (std.mem.eql(u8, v, "d3d12")) .d3d12 else if (std.mem.eql(u8, v, "metal")) .metal else .any;
    }
}

const gpu_policy: plan.TilePolicy = plan.tilePolicyForTarget(.{ .kind = .webgpu });
// Representative CPU tiling (matches src/bench.zig's defaultTilePolicy): the CPU
// kernels are cache-blocked and reject GPU-sized tiles, so each backend is timed on
// its own appropriate tiling — the fair comparison.
const cpu_policy: plan.TilePolicy = .{ .base_square_2d = 256, .base_1d = 256, .quant_k_block = 32, .tile_alignment = 64 };

/// Build `C = A @ B` over [m,k] @ [k,n] f32 with the given tile policy. Returns the
/// compiled program + the output tensor id.
fn buildMatMul(alloc: std.mem.Allocator, mgr: *StorageManager, policy: plan.TilePolicy, m: usize, n: usize, k: usize) !struct { prog: aion.program.Program, out: TensorId } {
    const tiles = plan.chooseMatMulTiles(policy, m, n, k, .f32);
    const a_id = try mgr.createTiledTensor(.f32, &[_]usize{ m, k }, &[_]usize{ tiles.tm, tiles.tk }, .{ .tile_alignment = policy.tile_alignment });
    const b_id = try mgr.createTiledTensor(.f32, &[_]usize{ k, n }, &[_]usize{ tiles.tk, tiles.tn }, .{ .tile_alignment = policy.tile_alignment });

    const a_data = try alloc.alloc(f32, m * k);
    defer alloc.free(a_data);
    const b_data = try alloc.alloc(f32, k * n);
    defer alloc.free(b_data);
    for (a_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) * 0.05;
    for (b_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) * 0.07;
    try mgr.writeFromPackedScalar(a_id, std.mem.sliceAsBytes(a_data));
    try mgr.writeFromPackedScalar(b_id, std.mem.sliceAsBytes(b_data));

    var g = Graph.init(alloc);
    defer g.deinit();
    const av = try g.addInput(.f32, &[_]usize{ m, k });
    try g.bindExternal(av, a_id);
    const bv = try g.addInput(.f32, &[_]usize{ k, n });
    try g.bindExternal(bv, b_id);
    const cv = try g.addMatMul(av, bv, 1.0, 0.0);
    try g.setOutputs(&[_]aion.graph.ValueId{cv});

    const prog = try aion.program.compileGraph(alloc, &g, mgr, policy);
    return .{ .prog = prog, .out = prog.outputs[0] };
}

fn nowNs() u64 {
    const ts: std.Io.Timestamp = std.Io.Clock.awake.now(std.Options.debug_io);
    const ns: i96 = ts.toNanoseconds();
    if (ns <= 0) return 0;
    return @intCast(@min(ns, @as(i96, std.math.maxInt(u64))));
}

/// Time `iters` executions of `prog` on `be` (one warmup run excluded). Returns
/// total nanoseconds for the timed iterations. End-to-end: includes the per-call
/// output device->host readback for the GPU backend.
fn timeBackend(be: Backend, prog: *const aion.program.Program, store: aion.tensor_store.TensorStore, iters: usize) !u64 {
    try be.executeProgram(prog, store); // warmup (uploads weights; builds residency)
    const start = nowNs();
    var i: usize = 0;
    while (i < iters) : (i += 1) try be.executeProgram(prog, store);
    return nowNs() - start;
}

/// Pure GPU kernel time: submit the compute `iters` times with NO per-iteration
/// output readback, then block once on completion. Isolates kernel (+ CPU submit)
/// throughput from the full-output D2H readback that `timeBackend` includes.
fn timeGpuKernel(gb: *gpu.GpuBackend, prog: *const aion.program.Program, store: aion.tensor_store.TensorStore, iters: usize) !u64 {
    gb.flush_outputs = false;
    defer gb.flush_outputs = true;
    try gb.backend().executeProgram(prog, store); // warmup
    gb.sync();
    const start = nowNs();
    var i: usize = 0;
    while (i < iters) : (i += 1) try gb.backend().executeProgram(prog, store);
    gb.sync();
    return nowNs() - start;
}

fn reportMatMul(label: []const u8, m: usize, n: usize, k: usize, iters: usize, ns: u64) void {
    const flops: f64 = 2.0 * @as(f64, @floatFromInt(m * n * k)) * @as(f64, @floatFromInt(iters));
    const gflops: f64 = flops / @as(f64, @floatFromInt(ns)); // FLOP/ns == GFLOP/s
    const ms_iter: f64 = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(iters)) / 1.0e6;
    out.print("  {s:<8} {d:8.2} GFLOP/s   {d:8.3} ms/iter\n", .{ label, gflops, ms_iter });
}

fn wgpuLog(level: wgpu.c.WGPULogLevel, message: wgpu.c.WGPUStringView, _: ?*anyopaque) callconv(.c) void {
    out.print("[wgpu:{d}] {s}\n", .{ level, wgpu.fromStrv(message) });
}

fn run(args: std.process.Args) !void {
    const alloc = std.heap.page_allocator;

    // Surface wgpu-native validation/limit errors (otherwise swallowed as ExecutionFailed).
    wgpu.c.wgpuSetLogCallback(wgpuLog, null);
    wgpu.c.wgpuSetLogLevel(wgpu.c.WGPULogLevel_Warn);

    var it = try args.iterateAllocator(alloc);
    defer it.deinit();
    var a: Args = .{};
    _ = it.next();
    while (it.next()) |arg| applyArg(&a, arg, &it);

    const tiles = plan.chooseMatMulTiles(gpu_policy, a.m, a.n, a.k, .f32);
    out.print("matmul f32: M={d} N={d} K={d}, iters={d}\n", .{ a.m, a.n, a.k, a.iters });
    out.print("  gpu tiles: tm={d} tn={d} tk={d}  (output tiles: {d}x{d}, k-tiles: {d})\n", .{
        tiles.tm,                        tiles.tn,                        tiles.tk,
        (a.m + tiles.tm - 1) / tiles.tm, (a.n + tiles.tn - 1) / tiles.tn, (a.k + tiles.tk - 1) / tiles.tk,
    });

    var cpu_out: ?[]f32 = null;
    defer if (cpu_out) |buf| alloc.free(buf);
    var gpu_out: ?[]f32 = null;
    defer if (gpu_out) |buf| alloc.free(buf);
    if (!a.gpu_only) {
        cpu_out = try alloc.alloc(f32, a.m * a.n);
        gpu_out = try alloc.alloc(f32, a.m * a.n);
    }

    // --- CPU ---
    var cpu_ns: u64 = 0;
    if (!a.gpu_only) {
        var mgr = StorageManager.init(alloc);
        defer mgr.deinit();
        var built = try buildMatMul(alloc, &mgr, cpu_policy, a.m, a.n, a.k);
        defer built.prog.deinit();
        var cpu = aion.cpu.CpuBackend.init(alloc);
        defer cpu.deinit();
        cpu_ns = try timeBackend(cpu.backend(), &built.prog, mgr.tensorStore(), a.iters);
        try mgr.readToPackedScalar(built.out, std.mem.sliceAsBytes(cpu_out.?));
        reportMatMul("cpu", a.m, a.n, a.k, a.iters, cpu_ns);
    }

    // --- GPU ---
    var device = wgpu.Gpu.init(a.opts) catch |e| {
        out.print("  gpu:      unavailable ({s}) — skipped\n", .{@errorName(e)});
        return;
    };
    defer device.deinit();
    const d = device.describe();
    out.print("  gpu dev:  {s} ({s}) \"{s}\"\n", .{ d.backendName(), d.kindName(), d.nameSlice() });
    out.print("  limits:   shared={d}B storage_binding={d}MiB invocations={d} wg_x={d}\n", .{ device.limits.max_shared_bytes, device.limits.max_storage_binding_bytes / (1024 * 1024), device.limits.max_invocations, device.limits.max_workgroup_size_x });
    out.print("  subgroup: enabled={} size=[{d}..{d}] fixed={?d}\n", .{ device.limits.subgroups, device.limits.subgroup_min, device.limits.subgroup_max, device.limits.fixedSubgroupSize() });

    var gb = gpu.GpuBackend.init(alloc, &device);
    defer gb.deinit();
    var gpu_ns: u64 = 0;
    {
        var mgr = StorageManager.init(alloc);
        defer mgr.deinit();
        var built = try buildMatMul(alloc, &mgr, gpu_policy, a.m, a.n, a.k);
        defer built.prog.deinit();
        const kernel_ns = try timeGpuKernel(&gb, &built.prog, mgr.tensorStore(), a.iters);
        gpu_ns = try timeBackend(gb.backend(), &built.prog, mgr.tensorStore(), a.iters);
        if (!a.gpu_only) try mgr.readToPackedScalar(built.out, std.mem.sliceAsBytes(gpu_out.?));
        if (gb.matmul.lastChoiceEntry()) |entry| {
            out.print("  gpu cfg:  {s}  threads={d} shared={d}B\n", .{
                entry,
                gb.matmul.lastChoiceThreads() orelse 0,
                gb.matmul.lastChoiceSharedBytes() orelse 0,
            });
        }
        reportMatMul("gpu-knl", a.m, a.n, a.k, a.iters, kernel_ns); // pure kernel
        reportMatMul("gpu-e2e", a.m, a.n, a.k, a.iters, gpu_ns); // incl. output readback
    }

    // Correctness sanity + headline speedup.
    if (a.gpu_only) return;
    var max_abs: f32 = 0;
    for (cpu_out.?, gpu_out.?) |cv, gv| max_abs = @max(max_abs, @abs(cv - gv));
    const speedup: f64 = if (gpu_ns == 0) 0 else @as(f64, @floatFromInt(cpu_ns)) / @as(f64, @floatFromInt(gpu_ns));
    out.print("  gpu/cpu speedup: {d:.2}x   (max|cpu-gpu|={e:.2})\n", .{ speedup, max_abs });
}

pub fn main(minimal: std.process.Init.Minimal) void {
    run(minimal.args) catch |e| {
        out.print("[gpu-bench] error: {s}\n", .{@errorName(e)});
        std.process.exit(1);
    };
}
