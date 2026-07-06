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
//!   zig build gpu-bench -Doptimize=ReleaseFast -- --suite ops --m 4096 --n 4096
//!   zig build gpu-bench -Doptimize=ReleaseFast -- --suite kernels --iters 20
//! `--suite ops` sweeps the bandwidth-bound elementwise/row-wise kernels at
//! [--m, --n] reporting kernel GB/s, e2e ms, and a CPU correctness/speedup
//! column. `--suite kernels` sweeps every remaining kernel (attention, conv,
//! LSTM, FFT/STFT, gather/rope/kv, views, i32 ops, casts) at fixed
//! model-representative shapes.

const std = @import("std");
const aion = @import("aion");

const gpu = aion.gpu; // feature-gated GPU backend (root.zig exposes it when -Dgpu)
const wgpu = gpu.wgpu;
const bk = @import("bench_kernels"); // shared op list + unified reporter + timing

const Backend = aion.backend.Backend;
const StorageManager = aion.storage_manager.StorageManager;
const Graph = aion.graph.Graph;
const TensorId = aion.storage_manager.TensorId;
const plan = aion.plan;

// Shared with the CPU bench so ops + report format can't drift.
const KOp = bk.KOp;
const kInfo = bk.kInfo;
const KA = bk.KA;
const fillTensor = bk.fillTensor;
const timeBackend = bk.timeBackend;
const nowNs = bk.nowNs;

const out = std.debug;

const Suite = enum { matmul, ops, nt, kernels };

const Args = struct {
    m: usize = 1024,
    n: usize = 1024,
    k: usize = 1024,
    iters: usize = 50,
    gpu_only: bool = false,
    /// Optional label filter for the kernels suite (--op <label>).
    op_filter: ?[]const u8 = null,
    /// `matmul` (default) or `ops` (row-wise + elementwise kernel sweep).
    suite: Suite = .matmul,
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
    if (valueFor(arg, "--op", it)) |v| a.op_filter = v;
    if (valueFor(arg, "--suite", it)) |v| {
        a.suite = if (std.mem.eql(u8, v, "ops")) .ops else if (std.mem.eql(u8, v, "nt")) .nt else if (std.mem.eql(u8, v, "kernels")) .kernels else .matmul;
    }
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

// ---- ops suite (row-wise + elementwise kernels) -----------------------------

const OpKind = enum { add, silu, bcast_mul, softmax, rmsnorm, layernorm };

fn opLabel(kind: OpKind) []const u8 {
    return switch (kind) {
        .add => "add",
        .silu => "silu",
        .bcast_mul => "bcast",
        .softmax => "softmax",
        .rmsnorm => "rmsnorm",
        .layernorm => "layernorm",
    };
}

/// Logical bytes moved per iteration (reads + writes the op's definition needs,
/// not what the kernel might re-read) — the basis for the GB/s figure.
fn opBytes(kind: OpKind, m: usize, n: usize) f64 {
    const mn: f64 = @floatFromInt(m * n);
    const nn: f64 = @floatFromInt(n);
    return 4.0 * switch (kind) {
        .add => 3.0 * mn, // 2 reads + 1 write
        .silu => 2.0 * mn,
        .bcast_mul => 2.0 * mn + nn,
        .softmax => 5.0 * mn, // x read twice, o written twice + read once
        .rmsnorm, .layernorm => 3.0 * mn + 2.0 * nn, // x twice, o once, g+b
    };
}

/// Build one benchmark op over an [m,n] input with `policy`-appropriate tiling
/// (softmax/norm lowering inherits the bound input's tiles, so the input must be
/// created with the same chooser the compiler would use).
fn buildOp(alloc: std.mem.Allocator, mgr: *StorageManager, policy: plan.TilePolicy, kind: OpKind, m: usize, n: usize) !struct { prog: aion.program.Program, out: TensorId } {
    var g = Graph.init(alloc);
    defer g.deinit();

    const tile2: [2]usize = switch (kind) {
        .softmax => blk: {
            const t = plan.chooseSoftmaxTiles(policy, m, n);
            break :blk .{ t.tm, t.tn };
        },
        .rmsnorm, .layernorm => blk: {
            const t = plan.chooseNormTiles(policy, m, n);
            break :blk .{ t.tm, t.tn };
        },
        else => plan.chooseTileShape2DSquare(policy, m, n),
    };
    const x_id = try mgr.createTiledTensor(.f32, &[_]usize{ m, n }, &tile2, .{ .tile_alignment = policy.tile_alignment });
    try fillTensor(alloc, mgr, x_id, m * n, 1);
    const xv = try g.addInput(.f32, &[_]usize{ m, n });
    try g.bindExternal(xv, x_id);

    const out_v = switch (kind) {
        .add => blk: {
            const y_id = try mgr.createTiledTensor(.f32, &[_]usize{ m, n }, &tile2, .{ .tile_alignment = policy.tile_alignment });
            try fillTensor(alloc, mgr, y_id, m * n, 2);
            const yv = try g.addInput(.f32, &[_]usize{ m, n });
            try g.bindExternal(yv, y_id);
            break :blk try g.addElemwiseBinary(.add, xv, yv);
        },
        .silu => try g.addUnary(.silu, xv),
        .bcast_mul => blk: {
            const b_id = try mgr.createTiledTensor(.f32, &[_]usize{n}, &[_]usize{tile2[1]}, .{ .tile_alignment = policy.tile_alignment });
            try fillTensor(alloc, mgr, b_id, n, 3);
            const bv = try g.addInput(.f32, &[_]usize{n});
            try g.bindExternal(bv, b_id);
            break :blk try g.addBroadcastLastDimBinary(.mul, xv, bv);
        },
        .softmax => try g.addSoftmax(xv, -1),
        .rmsnorm, .layernorm => blk: {
            const t1 = plan.chooseTileShape1D(policy, n);
            const g_id = try mgr.createTiledTensor(.f32, &[_]usize{n}, &t1, .{ .tile_alignment = policy.tile_alignment });
            const b_id = try mgr.createTiledTensor(.f32, &[_]usize{n}, &t1, .{ .tile_alignment = policy.tile_alignment });
            try fillTensor(alloc, mgr, g_id, n, 4);
            try fillTensor(alloc, mgr, b_id, n, 5);
            const gv = try g.addInput(.f32, &[_]usize{n});
            try g.bindExternal(gv, g_id);
            const bv = try g.addInput(.f32, &[_]usize{n});
            try g.bindExternal(bv, b_id);
            break :blk if (kind == .rmsnorm)
                try g.addRMSNorm(xv, gv, bv, 1e-5, &[_]usize{n})
            else
                try g.addLayerNorm(xv, gv, bv, 1e-5, &[_]usize{n});
        },
    };
    try g.setOutputs(&[_]aion.graph.ValueId{out_v});

    const prog = try aion.program.compileGraph(alloc, &g, mgr, policy);
    return .{ .prog = prog, .out = prog.outputs[0] };
}

/// Bench one op: CPU (unless --gpu-only), GPU kernel-only, GPU end-to-end, plus
/// a correctness check (max |cpu-gpu| over the full output).
fn benchOp(alloc: std.mem.Allocator, gb: *gpu.GpuBackend, a: Args, kind: OpKind) !void {
    var cpu_ns: u64 = 0;
    var cpu_out: ?[]f32 = null;
    defer if (cpu_out) |buf| alloc.free(buf);

    if (!a.gpu_only) {
        cpu_out = try alloc.alloc(f32, a.m * a.n);
        var mgr = StorageManager.init(alloc);
        defer mgr.deinit();
        var built = try buildOp(alloc, &mgr, cpu_policy, kind, a.m, a.n);
        defer built.prog.deinit();
        var cpu = aion.cpu.CpuBackend.init(alloc);
        defer cpu.deinit();
        cpu_ns = try timeBackend(cpu.backend(), &built.prog, mgr.tensorStore(), a.iters);
        try mgr.readToPackedScalar(built.out, std.mem.sliceAsBytes(cpu_out.?));
    }

    // Each op gets a fresh StorageManager and its own Session; residency lives
    // and dies with the session, so there's no stale-pointer hazard to reset.
    var mgr = StorageManager.init(alloc);
    defer mgr.deinit();
    var built = try buildOp(alloc, &mgr, gpu_policy, kind, a.m, a.n);
    defer built.prog.deinit();
    var session = try gb.backend().createSession(mgr.tensorStore());
    defer session.deinit();
    const kernel_ns = try timeGpuKernel(gb, session, &built.prog, a.iters);
    const gpu_ns = try timeSession(session, &built.prog, a.iters);

    var sample: bk.Sample = .{
        .label = opLabel(kind),
        .iters = a.iters,
        .ns = kernel_ns,
        .bytes = opBytes(kind, a.m, a.n),
        .e2e_ns = gpu_ns,
    };
    if (cpu_out) |cref| {
        const gpu_out = try alloc.alloc(f32, a.m * a.n);
        defer alloc.free(gpu_out);
        try mgr.readToPackedScalar(built.out, std.mem.sliceAsBytes(gpu_out));
        var max_abs: f32 = 0;
        for (cref, gpu_out) |cv, gv| max_abs = @max(max_abs, @abs(cv - gv));
        sample.ref_ns = cpu_ns;
        sample.max_abs = max_abs;
    }
    bk.report(sample);
}

fn runOps(alloc: std.mem.Allocator, a: Args) !void {
    out.print("ops f32: M={d} N={d}, iters={d}\n", .{ a.m, a.n, a.iters });

    var device = wgpu.Gpu.init(a.opts) catch |e| {
        out.print("  gpu:      unavailable ({s}) — skipped\n", .{@errorName(e)});
        return;
    };
    defer device.deinit();
    const d = device.describe();
    out.print("  gpu dev:  {s} ({s}) \"{s}\"\n", .{ d.backendName(), d.kindName(), d.nameSlice() });

    var gb = gpu.GpuBackend.init(alloc, &device);
    defer gb.deinit();

    for (std.enums.values(OpKind)) |kind| {
        benchOp(alloc, &gb, a, kind) catch |e| {
            out.print("  {s:<10} failed: {s}\n", .{ opLabel(kind), @errorName(e) });
        };
    }
}

// ---- nt suite (MatMulNT: q8_0/f32 weights, GEMV + dequant-GEMM paths) -------

/// Pack f32 rows [n, k] into ggml q8_0 blocks (f16 scale + 32 i8 per 32 elems).
fn packQ8(alloc: std.mem.Allocator, vals: []const f32, n: usize, k: usize) ![]u8 {
    const bpr = k / 32;
    const out_bytes = try alloc.alloc(u8, n * bpr * 34);
    for (0..n) |row| {
        for (0..bpr) |blk| {
            const src = vals[row * k + blk * 32 ..][0..32];
            var absmax: f32 = 0;
            for (src) |v| absmax = @max(absmax, @abs(v));
            const scale: f32 = if (absmax == 0) 1.0 else absmax / 127.0;
            const inv: f32 = if (absmax == 0) 0 else 1.0 / scale;
            const off = (row * bpr + blk) * 34;
            std.mem.writeInt(u16, out_bytes[off..][0..2], @bitCast(@as(f16, @floatCast(scale))), .little);
            for (src, 0..) |v, i| {
                const q: i32 = std.math.clamp(@as(i32, @intFromFloat(@round(v * inv))), -128, 127);
                out_bytes[off + 2 + i] = @bitCast(@as(i8, @intCast(q)));
            }
        }
    }
    return out_bytes;
}

/// Build `C[m,n] = A[m,k] @ B[n,k]^T` with B in q8_0 or f32, N-tiled
/// `b_tile_rows` per tile (GPU: one tile → one dispatch; CPU: chunks for
/// thread-level parallelism — each backend gets its natural tiling).
fn buildNt(alloc: std.mem.Allocator, mgr: *StorageManager, q8: bool, m: usize, n: usize, k: usize, b_tile_rows: usize) !struct { prog: aion.program.Program, out: TensorId } {
    var g = Graph.init(alloc);
    defer g.deinit();

    const a_id = try mgr.createTiledTensor(.f32, &[_]usize{ m, k }, &[_]usize{ m, k }, .{ .tile_alignment = 64 });
    try fillTensor(alloc, mgr, a_id, m * k, 1);
    const av = try g.addInput(.f32, &[_]usize{ m, k });
    try g.bindExternal(av, a_id);

    const b_vals = try alloc.alloc(f32, n * k);
    defer alloc.free(b_vals);
    for (b_vals, 0..) |*v, i| {
        const p: usize = (i * 2654435761 + 77) % 1000;
        v.* = (@as(f32, @floatFromInt(p)) - 500.0) * 0.004;
    }

    var bv: aion.graph.ValueId = undefined;
    if (q8) {
        const packed_b = try packQ8(alloc, b_vals, n, k);
        defer alloc.free(packed_b);
        const b_id = try mgr.createTiledTensor(.q8_0, &[_]usize{ n, k }, &[_]usize{ b_tile_rows, k }, .{ .tile_alignment = 64, .quant_axis = 1 });
        try mgr.writeFromPackedQuant(b_id, packed_b);
        bv = try g.addInput(.q8_0, &[_]usize{ n, k });
        try g.bindExternal(bv, b_id);
    } else {
        const b_id = try mgr.createTiledTensor(.f32, &[_]usize{ n, k }, &[_]usize{ b_tile_rows, k }, .{ .tile_alignment = 64 });
        try mgr.writeFromPackedScalar(b_id, std.mem.sliceAsBytes(b_vals));
        bv = try g.addInput(.f32, &[_]usize{ n, k });
        try g.bindExternal(bv, b_id);
    }

    const cv = try g.addMatMulNT(av, bv, 1.0, 0.0);
    try g.setOutputs(&[_]aion.graph.ValueId{cv});
    const prog = try aion.program.compileGraph(alloc, &g, mgr, gpu_policy);
    return .{ .prog = prog, .out = prog.outputs[0] };
}

fn benchNt(alloc: std.mem.Allocator, gb: *gpu.GpuBackend, a: Args, q8: bool, m: usize, n: usize, k: usize) !void {
    const label: []const u8 = if (q8) "q8_0" else "f32";
    // B bytes read once per iteration — the bandwidth-bound GEMV figure.
    const b_bytes: f64 = if (q8)
        @as(f64, @floatFromInt(n * (k / 32) * 34))
    else
        @as(f64, @floatFromInt(n * k * 4));
    const flops: f64 = 2.0 * @as(f64, @floatFromInt(m * n * k));

    var cpu_ns: u64 = 0;
    var cpu_out: ?[]f32 = null;
    defer if (cpu_out) |buf| alloc.free(buf);
    if (!a.gpu_only) {
        cpu_out = try alloc.alloc(f32, m * n);
        var mgr = StorageManager.init(alloc);
        defer mgr.deinit();
        // CPU parallelizes over N tiles; give it row chunks.
        var built = try buildNt(alloc, &mgr, q8, m, n, k, @min(n, 256));
        defer built.prog.deinit();
        var cpu = aion.cpu.CpuBackend.init(alloc);
        defer cpu.deinit();
        cpu_ns = try timeBackend(cpu.backend(), &built.prog, mgr.tensorStore(), a.iters);
        try mgr.readToPackedScalar(built.out, std.mem.sliceAsBytes(cpu_out.?));
    }

    var mgr = StorageManager.init(alloc);
    defer mgr.deinit();
    var built = try buildNt(alloc, &mgr, q8, m, n, k, n); // GPU: single N tile
    defer built.prog.deinit();
    var session = try gb.backend().createSession(mgr.tensorStore());
    defer session.deinit();
    const kernel_ns = try timeGpuKernel(gb, session, &built.prog, a.iters);
    const gpu_ns = try timeSession(session, &built.prog, a.iters);

    var lbuf: [24]u8 = undefined;
    const row_label = std.fmt.bufPrint(&lbuf, "nt-{s} M={d}", .{ label, m }) catch label;

    var sample: bk.Sample = .{
        .label = row_label,
        .iters = a.iters,
        .ns = kernel_ns,
        .bytes = b_bytes,
        .flops = flops,
        .e2e_ns = gpu_ns,
    };
    if (cpu_out) |cref| {
        const gpu_out = try alloc.alloc(f32, m * n);
        defer alloc.free(gpu_out);
        try mgr.readToPackedScalar(built.out, std.mem.sliceAsBytes(gpu_out));
        var max_abs: f32 = 0;
        for (cref, gpu_out) |cv2, gv| max_abs = @max(max_abs, @abs(cv2 - gv));
        sample.ref_ns = cpu_ns;
        sample.max_abs = max_abs;
    }
    bk.report(sample);
}

fn runNt(alloc: std.mem.Allocator, a: Args) !void {
    out.print("matmul NT: N={d} K={d}, iters={d}  (C[M,N] = A[M,K] @ B[N,K]^T)\n", .{ a.n, a.k, a.iters });

    var device = wgpu.Gpu.init(a.opts) catch |e| {
        out.print("  gpu:      unavailable ({s}) — skipped\n", .{@errorName(e)});
        return;
    };
    defer device.deinit();
    const d = device.describe();
    out.print("  gpu dev:  {s} ({s}) \"{s}\"\n", .{ d.backendName(), d.kindName(), d.nameSlice() });

    var gb = gpu.GpuBackend.init(alloc, &device);
    defer gb.deinit();

    // M=1 hits the GEMV kernel (decode); M=a.m>1 hits dequant+GEMM (prefill).
    for ([_]bool{ true, false }) |q8| {
        try benchNt(alloc, &gb, a, q8, 1, a.n, a.k);
        if (a.m > 1) try benchNt(alloc, &gb, a, q8, a.m, a.n, a.k);
    }
}

fn benchKernel(alloc: std.mem.Allocator, gb: *gpu.GpuBackend, a: Args, op: KOp) !void {
    const info = kInfo(op);

    // CPU reference: same GPU-tiled program. Tolerated to fail (tile caps).
    var cpu_ns: u64 = 0;
    var cpu_out: ?[]u8 = null;
    defer if (cpu_out) |buf| alloc.free(buf);
    if (!a.gpu_only) cpu: {
        var mgr = StorageManager.init(alloc);
        defer mgr.deinit();
        var built = bk.buildK(alloc, &mgr, op, gpu_policy) catch break :cpu;
        defer built.prog.deinit();
        var cpu = aion.cpu.CpuBackend.init(alloc);
        defer cpu.deinit();
        cpu_ns = timeBackend(cpu.backend(), &built.prog, mgr.tensorStore(), a.iters) catch break :cpu;
        const buf = try alloc.alloc(u8, info.out_elems * 4);
        mgr.readToPackedScalar(built.out, buf) catch {
            alloc.free(buf);
            break :cpu;
        };
        cpu_out = buf;
    }

    var mgr = StorageManager.init(alloc);
    defer mgr.deinit();
    var built = bk.buildK(alloc, &mgr, op, gpu_policy) catch |e| {
        out.print("  {s:<16} build/compile failed: {s}\n", .{ info.label, @errorName(e) });
        return;
    };
    defer built.prog.deinit();
    var session = try gb.backend().createSession(mgr.tensorStore());
    defer session.deinit();
    const kernel_ns = timeGpuKernel(gb, session, &built.prog, a.iters) catch |e| {
        out.print("  {s:<16} gpu execute failed: {s}\n", .{ info.label, @errorName(e) });
        return;
    };
    const gpu_ns = try timeSession(session, &built.prog, a.iters);

    var sample: bk.Sample = .{
        .label = info.label,
        .iters = a.iters,
        .ns = kernel_ns,
        .bytes = info.bytes,
        .flops = info.flops,
        .e2e_ns = gpu_ns,
    };
    if (cpu_out) |cref| {
        const gpu_buf = try alloc.alloc(u8, info.out_elems * 4);
        defer alloc.free(gpu_buf);
        try mgr.readToPackedScalar(built.out, gpu_buf);
        sample.ref_ns = cpu_ns;
        if (info.out_i32) {
            const ci: []align(1) const i32 = @alignCast(std.mem.bytesAsSlice(i32, cref));
            const gi: []align(1) const i32 = @alignCast(std.mem.bytesAsSlice(i32, gpu_buf));
            var mismatches: usize = 0;
            for (ci, gi) |cv, gv| mismatches += @intFromBool(cv != gv);
            sample.i32_diffs = mismatches;
        } else {
            const cf: []align(1) const f32 = @alignCast(std.mem.bytesAsSlice(f32, cref));
            const gf: []align(1) const f32 = @alignCast(std.mem.bytesAsSlice(f32, gpu_buf));
            var max_abs: f32 = 0;
            for (cf, gf) |cv, gv| max_abs = @max(max_abs, @abs(cv - gv));
            sample.max_abs = max_abs;
        }
    }
    bk.report(sample);
}

fn runKernels(alloc: std.mem.Allocator, a: Args) !void {
    out.print("kernel sweep (fixed shapes), iters={d}\n", .{a.iters});

    var device = wgpu.Gpu.init(a.opts) catch |e| {
        out.print("  gpu:      unavailable ({s}) — skipped\n", .{@errorName(e)});
        return;
    };
    defer device.deinit();
    const d = device.describe();
    out.print("  gpu dev:  {s} ({s}) \"{s}\"\n", .{ d.backendName(), d.kindName(), d.nameSlice() });

    var gb = gpu.GpuBackend.init(alloc, &device);
    defer gb.deinit();

    for (std.enums.values(KOp)) |op| {
        if (a.op_filter) |f| {
            if (!std.mem.eql(u8, kInfo(op).label, f)) continue;
        }
        benchKernel(alloc, &gb, a, op) catch |e| {
            out.print("  {s:<12} failed: {s}\n", .{ kInfo(op).label, @errorName(e) });
        };
    }
}

/// Pure GPU kernel time: submit the compute `iters` times with NO per-iteration
/// output readback, then block once on completion. Isolates kernel (+ CPU submit)
/// throughput from the full-output D2H readback that `timeBackend` includes.
fn timeGpuKernel(gb: *gpu.GpuBackend, session: aion.backend.Session, prog: *const aion.program.Program, iters: usize) !u64 {
    gb.flush_outputs = false;
    defer gb.flush_outputs = true;
    try session.execute(prog); // warmup
    gb.sync();
    const start = nowNs();
    var i: usize = 0;
    while (i < iters) : (i += 1) try session.execute(prog);
    gb.sync();
    return nowNs() - start;
}

/// End-to-end warmup + timed loop over one persistent session (outputs flushed
/// to host each iteration). Session is created once by the caller, so residency
/// stays warm across the loop.
fn timeSession(session: aion.backend.Session, prog: *const aion.program.Program, iters: usize) !u64 {
    try session.execute(prog); // warmup
    const start = nowNs();
    var i: usize = 0;
    while (i < iters) : (i += 1) try session.execute(prog);
    return nowNs() - start;
}

fn reportMatMul(label: []const u8, m: usize, n: usize, k: usize, iters: usize, ns: u64) void {
    bk.report(.{
        .label = label,
        .iters = iters,
        .ns = ns,
        .flops = 2.0 * @as(f64, @floatFromInt(m * n * k)),
    });
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

    if (a.suite == .ops) return runOps(alloc, a);
    if (a.suite == .nt) return runNt(alloc, a);
    if (a.suite == .kernels) return runKernels(alloc, a);

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

    var gb = gpu.GpuBackend.init(alloc, &device);
    defer gb.deinit();
    var gpu_ns: u64 = 0;
    {
        var mgr = StorageManager.init(alloc);
        defer mgr.deinit();
        var built = try buildMatMul(alloc, &mgr, gpu_policy, a.m, a.n, a.k);
        defer built.prog.deinit();
        var session = try gb.backend().createSession(mgr.tensorStore());
        defer session.deinit();
        const kernel_ns = try timeGpuKernel(&gb, session, &built.prog, a.iters);
        gpu_ns = try timeSession(session, &built.prog, a.iters);
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
