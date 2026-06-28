// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

const std = @import("std");
const aion = @import("aion");

/// Mirror the library's multiversion switch so the CPU backend (which reads
/// `@import("root")`) dispatches to the linked per-ISA tier objects. `build.zig`
/// links those objects into this exe and supplies `build_options.multiversion`.
pub const aion_multiversion: bool = @import("build_options").multiversion;

const Backend = aion.backend.Backend;
const CpuBackend = aion.cpu.CpuBackend;

const StorageManager = aion.storage_manager.StorageManager;
const TensorId = aion.storage_manager.TensorId;

const graph_mod = aion.graph;
const plan_mod = aion.plan;
const program_mod = aion.program;

const types = aion.types;

const Layout = types.Layout;

const Q4_0_BLOCK_ELEMS: usize = types.DType.q4_0.info().block_elems;
const Q4_0_BLOCK_BYTES: usize = types.DType.q4_0.info().block_bytes;
const Q8_0_BLOCK_ELEMS: usize = types.DType.q8_0.info().block_elems;
const Q8_0_BLOCK_BYTES: usize = types.DType.q8_0.info().block_bytes;

const BenchSuite = enum {
    all,
    matmul,
    matmul_q,
    decode,
};

const BenchDTypeMode = enum {
    f32,
    f16,
    both,
};

const BenchOptions = struct {
    iters: usize = 50,
    threads: usize = 1,

    suite: BenchSuite = .all,

    // Batch size for batched matmul benchmarks.
    batch: usize = 4,

    // Multi-head attention heads (separate head dimension).
    heads: usize = 4,

    // Elementwise length (logical elements).
    n_elem: usize = 8 * 1024 * 1024,

    // Matmul sizes.
    m: usize = 512,
    n: usize = 512,
    k: usize = 512,

    // Also benchmark quantized matmul variants (requires k % 32 == 0).
    quant: bool = true,

    // Scalar dtype benchmark mode.
    dtype_mode: BenchDTypeMode = .f32,

    // Print cpu cache detection / tuning selection.
    print_cpuid: bool = false,

    // Conv benchmark sizes.
    conv_batch: usize = 2,
    conv_l: usize = 512,
    conv_h: usize = 64,
    conv_w: usize = 64,
    conv_cin: usize = 64,
    conv_cout: usize = 64,
    conv_k: usize = 3,

    // Reflect convolution benchmarks.
    reflect_conv: bool = true,
};

fn printUsage() void {
    std.debug.print(
        "Aion kernel benchmarks\n\n" ++
            "Usage:\n" ++
            "  zig build bench -- [options]\n\n" ++
            "Options:\n" ++
            "  --iters N        Iterations per benchmark (default: 50)\n" ++
            "  --threads N      CPU backend thread count (default: 1)\n" ++
            "  --suite NAME     Bench suite: all|matmul|quant-matmul|decode (default: all)\n" ++
            "  --batch N        Batch size for batched matmul (default: 4)\n" ++
            "  --heads N        Multi-head attention heads (default: 8)\n" ++
            "  --n-elem N       Elemwise/Reduce logical element count (default: 8388608)\n" ++
            "  --m N            MatMul M (default: 512)\n" ++
            "  --n N            MatMul N (default: 512)\n" ++
            "  --k N            MatMul K (default: 512)\n" ++
            "  --dtype MODE     Benchmark dtype mode: f32|f16|both (default: f32)\n" ++
            "  --no-quant       Skip quantized matmul benches\n" ++
            "  --print-cpuid    Print detected CPU caches\n" ++
            "  --conv-batch N   Conv batch size (default: 2)\n" ++
            "  --conv-l N       Conv1D input length (default: 512)\n" ++
            "  --conv-h N       Conv2D input height (default: 64)\n" ++
            "  --conv-w N       Conv2D input width (default: 64)\n" ++
            "  --conv-cin N     Conv input channels (default: 64)\n" ++
            "  --conv-cout N    Conv output channels (default: 64)\n" ++
            "  --conv-k N       Conv kernel size (default: 3)\n" ++
            "  --no-reflect-conv  Disable reflect-padding conv benches\n" ++
            "  -h, --help       Show help\n",
        .{},
    );
}

fn parseUsize(arg: []const u8) !usize {
    return std.fmt.parseInt(usize, arg, 10);
}

fn parseBenchDTypeMode(arg: []const u8) !BenchDTypeMode {
    if (std.mem.eql(u8, arg, "f32")) return .f32;
    if (std.mem.eql(u8, arg, "f16")) return .f16;
    if (std.mem.eql(u8, arg, "both")) return .both;
    return error.InvalidArgument;
}

fn parseBenchSuite(arg: []const u8) !BenchSuite {
    if (std.mem.eql(u8, arg, "all")) return .all;
    if (std.mem.eql(u8, arg, "matmul")) return .matmul;
    if (std.mem.eql(u8, arg, "quant-matmul")) return .matmul_q;
    if (std.mem.eql(u8, arg, "decode")) return .decode;
    return error.InvalidArgument;
}

fn parseArgs(args: std.process.Args, allocator: std.mem.Allocator) !BenchOptions {
    var opts: BenchOptions = .{};

    var it = try args.iterateAllocator(allocator);
    defer it.deinit();

    _ = it.next(); // skip argv[0] (program name)
    while (it.next()) |a| {
        if (std.mem.eql(u8, a, "-h") or std.mem.eql(u8, a, "--help")) {
            printUsage();
            std.process.exit(0);
        } else if (std.mem.eql(u8, a, "--no-quant")) {
            opts.quant = false;
        } else if (std.mem.eql(u8, a, "--print-cpuid")) {
            opts.print_cpuid = true;
        } else if (std.mem.eql(u8, a, "--suite")) {
            const v = it.next() orelse return error.InvalidArgument;
            opts.suite = try parseBenchSuite(v);
        } else if (std.mem.eql(u8, a, "--iters")) {
            const v = it.next() orelse return error.InvalidArgument;
            opts.iters = try parseUsize(v);
        } else if (std.mem.eql(u8, a, "--threads")) {
            const v = it.next() orelse return error.InvalidArgument;
            opts.threads = try parseUsize(v);
        } else if (std.mem.eql(u8, a, "--batch")) {
            const v = it.next() orelse return error.InvalidArgument;
            opts.batch = try parseUsize(v);
        } else if (std.mem.eql(u8, a, "--heads")) {
            const v = it.next() orelse return error.InvalidArgument;
            opts.heads = try parseUsize(v);
        } else if (std.mem.eql(u8, a, "--n-elem")) {
            const v = it.next() orelse return error.InvalidArgument;
            opts.n_elem = try parseUsize(v);
        } else if (std.mem.eql(u8, a, "--m")) {
            const v = it.next() orelse return error.InvalidArgument;
            opts.m = try parseUsize(v);
        } else if (std.mem.eql(u8, a, "--n")) {
            const v = it.next() orelse return error.InvalidArgument;
            opts.n = try parseUsize(v);
        } else if (std.mem.eql(u8, a, "--k")) {
            const v = it.next() orelse return error.InvalidArgument;
            opts.k = try parseUsize(v);
        } else if (std.mem.eql(u8, a, "--dtype")) {
            const v = it.next() orelse return error.InvalidArgument;
            opts.dtype_mode = try parseBenchDTypeMode(v);
        } else if (std.mem.eql(u8, a, "--conv-batch")) {
            const v = it.next() orelse return error.InvalidArgument;
            opts.conv_batch = try parseUsize(v);
        } else if (std.mem.eql(u8, a, "--conv-l")) {
            const v = it.next() orelse return error.InvalidArgument;
            opts.conv_l = try parseUsize(v);
        } else if (std.mem.eql(u8, a, "--conv-h")) {
            const v = it.next() orelse return error.InvalidArgument;
            opts.conv_h = try parseUsize(v);
        } else if (std.mem.eql(u8, a, "--conv-w")) {
            const v = it.next() orelse return error.InvalidArgument;
            opts.conv_w = try parseUsize(v);
        } else if (std.mem.eql(u8, a, "--conv-cin")) {
            const v = it.next() orelse return error.InvalidArgument;
            opts.conv_cin = try parseUsize(v);
        } else if (std.mem.eql(u8, a, "--conv-cout")) {
            const v = it.next() orelse return error.InvalidArgument;
            opts.conv_cout = try parseUsize(v);
        } else if (std.mem.eql(u8, a, "--conv-k")) {
            const v = it.next() orelse return error.InvalidArgument;
            opts.conv_k = try parseUsize(v);
        } else if (std.mem.eql(u8, a, "--no-reflect-conv")) {
            opts.reflect_conv = false;
        } else {
            return error.InvalidArgument;
        }
    }

    if (opts.iters == 0) return error.InvalidArgument;
    if (opts.threads == 0) return error.InvalidArgument;
    if (opts.heads == 0) return error.InvalidArgument;
    if (opts.batch == 0) return error.InvalidArgument;
    if (opts.n_elem == 0) return error.InvalidArgument;
    if (opts.m == 0 or opts.n == 0 or opts.k == 0) return error.InvalidArgument;
    if (opts.conv_batch == 0) return error.InvalidArgument;
    if (opts.conv_l == 0 or opts.conv_h == 0 or opts.conv_w == 0) return error.InvalidArgument;
    if (opts.conv_cin == 0 or opts.conv_cout == 0) return error.InvalidArgument;
    if (opts.conv_k == 0) return error.InvalidArgument;

    return opts;
}

fn benchProgramConv1D(
    allocator: std.mem.Allocator,
    rnd: std.Random,
    iters: usize,
    batch: usize,
    l_in: usize,
    c_in: usize,
    c_out: usize,
    kernel: usize,
    groups: usize,
    pad_mode: types.PadMode,
    label: []const u8,
    be: Backend,
) !void {
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    if (groups == 0) return error.InvalidArgument;
    if (c_in % groups != 0 or c_out % groups != 0) return error.InvalidArgument;

    const stride: usize = 1;
    const dilation: usize = 1;
    const pad_left: usize = kernel / 2;
    const pad_right: usize = kernel / 2;
    const c_in_g: usize = c_in / groups;

    const l_out: usize = ((l_in + pad_left + pad_right - ((kernel - 1) * dilation + 1)) / stride) + 1;

    const policy: plan_mod.TilePolicy = defaultTilePolicy();
    const ct = plan_mod.chooseConv1DTiles(policy, l_out, c_out);

    const x: []f32 = try allocator.alloc(f32, batch * l_in * c_in);
    defer allocator.free(x);
    const w: []f32 = try allocator.alloc(f32, kernel * c_in_g * c_out);
    defer allocator.free(w);
    const b: []f32 = try allocator.alloc(f32, c_out);
    defer allocator.free(b);
    fillRandomF32(rnd, x);
    fillRandomF32(rnd, w);
    fillRandomF32(rnd, b);

    const x_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{ batch, l_in, c_in }, &[_]usize{ 1, ct.tl, ct.tc }, .{ .tile_alignment = policy.tile_alignment });
    const w_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{ kernel, c_in_g, c_out }, &[_]usize{ kernel, c_in_g, ct.tc }, .{ .tile_alignment = policy.tile_alignment });
    const b_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{c_out}, &[_]usize{ct.tc}, .{ .tile_alignment = policy.tile_alignment });
    try sm.writeFromPackedScalar(x_tid, std.mem.sliceAsBytes(x));
    try sm.writeFromPackedScalar(w_tid, std.mem.sliceAsBytes(w));
    try sm.writeFromPackedScalar(b_tid, std.mem.sliceAsBytes(b));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f32, &[_]usize{ batch, l_in, c_in });
    const w_in = try g.addInput(.f32, &[_]usize{ kernel, c_in_g, c_out });
    const b_in = try g.addInput(.f32, &[_]usize{c_out});
    try g.bindExternal(x_in, @intCast(x_tid));
    try g.bindExternal(w_in, @intCast(w_tid));
    try g.bindExternal(b_in, @intCast(b_tid));
    const y = try g.addConv1DWithPadMode(x_in, w_in, b_in, stride, dilation, pad_left, pad_right, pad_mode, groups);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    var prog = try program_mod.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    const ns: u64 = try benchLoop(iters, struct {
        be: Backend,
        sm: *StorageManager,
        prog: *const program_mod.Program,
        fn run(self: @This()) !void {
            try self.be.executeProgram(self.prog, self.sm.tensorStore());
        }
    }{ .be = be, .sm = &sm, .prog = &prog });

    const flops_per: u64 = 2 * @as(u64, @intCast(batch)) * @as(u64, @intCast(l_out)) * @as(u64, @intCast(c_out)) *
        @as(u64, @intCast(kernel * c_in_g));
    const gflops: f64 = fmtRateGFLOPs(flops_per * @as(u64, @intCast(iters)), ns);

    const out_tid: TensorId = prog.outputs[0];
    const out_bytes_len: usize = batch * l_out * c_out * @sizeOf(f32);
    const out_buf: []u8 = try allocator.alloc(u8, out_bytes_len);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(out_tid, out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);
    const idx0: usize = 0;
    const idx1: usize = (batch / 2) * l_out * c_out + (l_out / 2) * c_out + (c_out / 2);
    const idx2: usize = (batch - 1) * l_out * c_out + (l_out - 1) * c_out + (c_out - 1);
    const chk: f32 = out_vals[idx0] + out_vals[idx1] + out_vals[idx2];

    std.debug.print("program conv1d {s}:      {d:.2} GFLOP/s (b={} l={} cin={} cout={} k={} g={}, chk={d:.4}, wall={d:.3} ms, iter={d:.3} ms)\n", .{
        label,
        gflops,
        batch,
        l_in,
        c_in,
        c_out,
        kernel,
        groups,
        chk,
        nsToMilliseconds(ns),
        nsToMillisecondsPerIter(ns, iters),
    });
}

fn benchProgramConv2D(
    allocator: std.mem.Allocator,
    rnd: std.Random,
    iters: usize,
    batch: usize,
    h_in: usize,
    w_in: usize,
    c_in: usize,
    c_out: usize,
    kernel: usize,
    groups: usize,
    pad_mode: types.PadMode,
    label: []const u8,
    be: Backend,
) !void {
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    if (groups == 0) return error.InvalidArgument;
    if (c_in % groups != 0 or c_out % groups != 0) return error.InvalidArgument;

    const stride_h: usize = 1;
    const stride_w: usize = 1;
    const dilation_h: usize = 1;
    const dilation_w: usize = 1;
    const pad_top: usize = kernel / 2;
    const pad_bottom: usize = kernel / 2;
    const pad_left: usize = kernel / 2;
    const pad_right: usize = kernel / 2;

    const k_h: usize = kernel;
    const k_w: usize = kernel;
    const c_in_g: usize = c_in / groups;

    const h_out: usize = ((h_in + pad_top + pad_bottom - ((k_h - 1) * dilation_h + 1)) / stride_h) + 1;
    const w_out: usize = ((w_in + pad_left + pad_right - ((k_w - 1) * dilation_w + 1)) / stride_w) + 1;

    const policy: plan_mod.TilePolicy = defaultTilePolicy();
    const ct = plan_mod.chooseConv2DTiles(policy, h_out, w_out, c_out);

    const x: []f32 = try allocator.alloc(f32, batch * h_in * w_in * c_in);
    defer allocator.free(x);
    const w: []f32 = try allocator.alloc(f32, k_h * k_w * c_in_g * c_out);
    defer allocator.free(w);
    const b: []f32 = try allocator.alloc(f32, c_out);
    defer allocator.free(b);
    fillRandomF32(rnd, x);
    fillRandomF32(rnd, w);
    fillRandomF32(rnd, b);

    const x_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{ batch, h_in, w_in, c_in }, &[_]usize{ 1, ct.th, ct.tw, ct.tc }, .{ .tile_alignment = policy.tile_alignment });
    const w_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{ k_h, k_w, c_in_g, c_out }, &[_]usize{ k_h, k_w, c_in_g, ct.tc }, .{ .tile_alignment = policy.tile_alignment });
    const b_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{c_out}, &[_]usize{ct.tc}, .{ .tile_alignment = policy.tile_alignment });
    try sm.writeFromPackedScalar(x_tid, std.mem.sliceAsBytes(x));
    try sm.writeFromPackedScalar(w_tid, std.mem.sliceAsBytes(w));
    try sm.writeFromPackedScalar(b_tid, std.mem.sliceAsBytes(b));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f32, &[_]usize{ batch, h_in, w_in, c_in });
    const w_val = try g.addInput(.f32, &[_]usize{ k_h, k_w, c_in_g, c_out });
    const b_in = try g.addInput(.f32, &[_]usize{c_out});
    try g.bindExternal(x_in, @intCast(x_tid));
    try g.bindExternal(w_val, @intCast(w_tid));
    try g.bindExternal(b_in, @intCast(b_tid));
    const y = try g.addConv2DWithPadMode(x_in, w_val, b_in, stride_h, stride_w, dilation_h, dilation_w, pad_top, pad_bottom, pad_left, pad_right, pad_mode, groups);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    var prog = try program_mod.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    const ns: u64 = try benchLoop(iters, struct {
        be: Backend,
        sm: *StorageManager,
        prog: *const program_mod.Program,
        fn run(self: @This()) !void {
            try self.be.executeProgram(self.prog, self.sm.tensorStore());
        }
    }{ .be = be, .sm = &sm, .prog = &prog });

    const flops_per: u64 = 2 * @as(u64, @intCast(batch)) * @as(u64, @intCast(h_out)) * @as(u64, @intCast(w_out)) *
        @as(u64, @intCast(c_out)) * @as(u64, @intCast(k_h * k_w * c_in_g));
    const gflops: f64 = fmtRateGFLOPs(flops_per * @as(u64, @intCast(iters)), ns);

    const out_tid: TensorId = prog.outputs[0];
    const out_bytes_len: usize = batch * h_out * w_out * c_out * @sizeOf(f32);
    const out_buf: []u8 = try allocator.alloc(u8, out_bytes_len);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(out_tid, out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);
    const idx0: usize = 0;
    const idx1: usize = ((batch / 2) * h_out * w_out * c_out) + ((h_out / 2) * w_out * c_out) + ((w_out / 2) * c_out) + (c_out / 2);
    const idx2: usize = ((batch - 1) * h_out * w_out * c_out) + ((h_out - 1) * w_out * c_out) + ((w_out - 1) * c_out) + (c_out - 1);
    const chk: f32 = out_vals[idx0] + out_vals[idx1] + out_vals[idx2];

    std.debug.print("program conv2d {s}:      {d:.2} GFLOP/s (b={} h={} w={} cin={} cout={} k={} g={}, chk={d:.4}, wall={d:.3} ms, iter={d:.3} ms)\n", .{
        label,
        gflops,
        batch,
        h_in,
        w_in,
        c_in,
        c_out,
        kernel,
        groups,
        chk,
        nsToMilliseconds(ns),
        nsToMillisecondsPerIter(ns, iters),
    });
}

fn readF32AtTiled(sm: *const StorageManager, id: TensorId, idx0: usize, idx1: usize) !f32 {
    const t = try sm.getConst(id);
    if (t.dtype != .f32) return error.InvalidArgument;

    const ti0: usize = idx0 / t.tile_shape[0];
    const ti1: usize = if (t.rank == 1) 0 else (idx1 / t.tile_shape[1]);
    const in0: usize = idx0 - ti0 * t.tile_shape[0];
    const in1: usize = if (t.rank == 1) 0 else (idx1 - ti1 * t.tile_shape[1]);

    const tile = try t.acquireTileConst(ti0, ti1);
    const n_tile: usize = tile.shape_mem[1];
    const off: usize = (in0 * n_tile + in1) * @sizeOf(f32);
    return @as(*align(1) const f32, @ptrCast(tile.bytes[off..][0..4].ptr)).*;
}

fn readF16AtTiled(sm: *const StorageManager, id: TensorId, idx0: usize, idx1: usize) !f16 {
    const t = try sm.getConst(id);
    if (t.dtype != .f16) return error.InvalidArgument;

    const ti0: usize = idx0 / t.tile_shape[0];
    const ti1: usize = if (t.rank == 1) 0 else (idx1 / t.tile_shape[1]);
    const in0: usize = idx0 - ti0 * t.tile_shape[0];
    const in1: usize = if (t.rank == 1) 0 else (idx1 - ti1 * t.tile_shape[1]);

    const tile = try t.acquireTileConst(ti0, ti1);
    const n_tile: usize = tile.shape_mem[1];
    const off: usize = (in0 * n_tile + in1) * @sizeOf(f16);
    return @as(*align(1) const f16, @ptrCast(tile.bytes[off..][0..2].ptr)).*;
}

fn asF32Slice(buf: []u8) []align(1) f32 {
    std.debug.assert((buf.len % @sizeOf(f32)) == 0);
    const ptr: [*]align(1) f32 = @ptrCast(buf.ptr);
    return ptr[0 .. buf.len / @sizeOf(f32)];
}

fn asF16Slice(buf: []u8) []align(1) f16 {
    std.debug.assert((buf.len % @sizeOf(f16)) == 0);
    const ptr: [*]align(1) f16 = @ptrCast(buf.ptr);
    return ptr[0 .. buf.len / @sizeOf(f16)];
}

fn defaultTilePolicy() plan_mod.TilePolicy {
    // Chosen to be representative for CPU cache-friendly tiling and quant block alignment.
    // Increased base_square_2d to 256 to allow kernels to use L2 blocking effectively.
    return .{ .base_square_2d = 256, .base_1d = 256, .quant_k_block = 32, .tile_alignment = 64 };
}

fn benchProgramElemwiseAdd(allocator: std.mem.Allocator, rnd: std.Random, iters: usize, n_elem: usize, be: Backend) !void {
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = defaultTilePolicy();
    const t1: [1]usize = plan_mod.chooseTileShape1D(policy, n_elem);

    const a: []f32 = try allocator.alloc(f32, n_elem);
    defer allocator.free(a);
    const b: []f32 = try allocator.alloc(f32, n_elem);
    defer allocator.free(b);
    fillRandomF32(rnd, a);
    fillRandomF32(rnd, b);

    const a_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{n_elem}, &[_]usize{t1[0]}, .{ .tile_alignment = policy.tile_alignment });
    const b_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{n_elem}, &[_]usize{t1[0]}, .{ .tile_alignment = policy.tile_alignment });
    try sm.writeFromPackedScalar(a_tid, std.mem.sliceAsBytes(a));
    try sm.writeFromPackedScalar(b_tid, std.mem.sliceAsBytes(b));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const a_in = try g.addInput(.f32, &[_]usize{n_elem});
    const b_in = try g.addInput(.f32, &[_]usize{n_elem});
    try g.bindExternal(a_in, @intCast(a_tid));
    try g.bindExternal(b_in, @intCast(b_tid));
    const out = try g.addElemwiseBinary(.add, a_in, b_in);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    var prog = try program_mod.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    const ns: u64 = try benchLoop(iters, struct {
        be: Backend,
        sm: *StorageManager,
        prog: *const program_mod.Program,
        fn run(self: @This()) !void {
            try self.be.executeProgram(self.prog, self.sm.tensorStore());
        }
    }{ .be = be, .sm = &sm, .prog = &prog });

    const bytes: u64 = @as(u64, @intCast(3 * n_elem * @sizeOf(f32))) * @as(u64, @intCast(iters));
    const gib_s: f64 = fmtRateGiBPerSec(bytes, ns);

    const out_tid: TensorId = prog.outputs[0];
    const chk: f32 = try readF32AtTiled(&sm, out_tid, 0, 0) +
        try readF32AtTiled(&sm, out_tid, n_elem / 2, 0) +
        try readF32AtTiled(&sm, out_tid, n_elem - 1, 0);
    std.debug.print("program elemwise add f32: {d:.3} GiB/s  (chk={d:.4}, wall={d:.3} ms, iter={d:.3} ms)\n", .{ gib_s, chk, nsToMilliseconds(ns), nsToMillisecondsPerIter(ns, iters) });
}

fn benchProgramRelu(allocator: std.mem.Allocator, rnd: std.Random, iters: usize, n_elem: usize, be: Backend) !void {
    return benchProgramUnary(allocator, rnd, iters, n_elem, .relu, "relu", be);
}

fn benchProgramUnary(
    allocator: std.mem.Allocator,
    rnd: std.Random,
    iters: usize,
    n_elem: usize,
    op: types.UnaryOp,
    label: []const u8,
    be: Backend,
) !void {
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = defaultTilePolicy();
    const t1: [1]usize = plan_mod.chooseTileShape1D(policy, n_elem);

    const a: []f32 = try allocator.alloc(f32, n_elem);
    defer allocator.free(a);
    fillRandomF32(rnd, a);

    const a_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{n_elem}, &[_]usize{t1[0]}, .{ .tile_alignment = policy.tile_alignment });
    try sm.writeFromPackedScalar(a_tid, std.mem.sliceAsBytes(a));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const a_in = try g.addInput(.f32, &[_]usize{n_elem});
    try g.bindExternal(a_in, @intCast(a_tid));
    const out = try g.addUnary(op, a_in);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    var prog = try program_mod.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    const ns: u64 = try benchLoop(iters, struct {
        be: Backend,
        sm: *StorageManager,
        prog: *const program_mod.Program,
        fn run(self: @This()) !void {
            try self.be.executeProgram(self.prog, self.sm.tensorStore());
        }
    }{ .be = be, .sm = &sm, .prog = &prog });

    // One input read + one output write.
    const bytes: u64 = @as(u64, @intCast(2 * n_elem * @sizeOf(f32))) * @as(u64, @intCast(iters));
    const gib_s: f64 = fmtRateGiBPerSec(bytes, ns);

    const out_tid: TensorId = prog.outputs[0];
    const chk: f32 = try readF32AtTiled(&sm, out_tid, 0, 0) +
        try readF32AtTiled(&sm, out_tid, n_elem / 2, 0) +
        try readF32AtTiled(&sm, out_tid, n_elem - 1, 0);
    std.debug.print("program unary {s} f32:    {d:.3} GiB/s  (chk={d:.4}, wall={d:.3} ms, iter={d:.3} ms)\n", .{ label, gib_s, chk, nsToMilliseconds(ns), nsToMillisecondsPerIter(ns, iters) });
}

fn benchProgramReduceSum(allocator: std.mem.Allocator, rnd: std.Random, iters: usize, n_elem: usize, be: Backend) !void {
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = defaultTilePolicy();
    const t1: [1]usize = plan_mod.chooseTileShape1D(policy, n_elem);

    const a: []f32 = try allocator.alloc(f32, n_elem);
    defer allocator.free(a);
    fillRandomF32(rnd, a);

    const a_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{n_elem}, &[_]usize{t1[0]}, .{ .tile_alignment = policy.tile_alignment });
    try sm.writeFromPackedScalar(a_tid, std.mem.sliceAsBytes(a));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const a_in = try g.addInput(.f32, &[_]usize{n_elem});
    try g.bindExternal(a_in, @intCast(a_tid));
    const out = try g.addReduce(.sum, a_in);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    var prog = try program_mod.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    const ns: u64 = try benchLoop(iters, struct {
        be: Backend,
        sm: *StorageManager,
        prog: *const program_mod.Program,
        fn run(self: @This()) !void {
            try self.be.executeProgram(self.prog, self.sm.tensorStore());
        }
    }{ .be = be, .sm = &sm, .prog = &prog });

    const bytes: u64 = @as(u64, @intCast(n_elem * @sizeOf(f32))) * @as(u64, @intCast(iters));
    const gib_s: f64 = fmtRateGiBPerSec(bytes, ns);

    const out_tid: TensorId = prog.outputs[0];
    const sum: f32 = try readF32AtTiled(&sm, out_tid, 0, 0);
    std.debug.print("program reduce sum f32:   {d:.3} GiB/s  (sum={d:.4}, wall={d:.3} ms, iter={d:.3} ms)\n", .{ gib_s, sum, nsToMilliseconds(ns), nsToMillisecondsPerIter(ns, iters) });
}

fn benchProgramReduceAxisF32(
    allocator: std.mem.Allocator,
    rnd: std.Random,
    iters: usize,
    m: usize,
    n: usize,
    axis: i32,
    label: []const u8,
    be: Backend,
) !void {
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = defaultTilePolicy();
    const t2: [2]usize = plan_mod.chooseTileShape2DSquare(policy, m, n);

    const a: []f32 = try allocator.alloc(f32, m * n);
    defer allocator.free(a);
    fillRandomF32(rnd, a);

    const a_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{ m, n }, &[_]usize{ t2[0], t2[1] }, .{ .tile_alignment = policy.tile_alignment });
    try sm.writeFromPackedScalar(a_tid, std.mem.sliceAsBytes(a));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const a_in = try g.addInput(.f32, &[_]usize{ m, n });
    try g.bindExternal(a_in, @intCast(a_tid));
    const out = try g.addReduceAxis(.mean, a_in, axis);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    var prog = try program_mod.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    const ns: u64 = try benchLoop(iters, struct {
        be: Backend,
        sm: *StorageManager,
        prog: *const program_mod.Program,
        fn run(self: @This()) !void {
            try self.be.executeProgram(self.prog, self.sm.tensorStore());
        }
    }{ .be = be, .sm = &sm, .prog = &prog });

    const out_tid: TensorId = prog.outputs[0];
    const out_t = try sm.getConst(out_tid);
    var out_elems: usize = 1;
    for (out_t.shape) |dim| {
        out_elems *= dim;
    }

    // Approx memory traffic: one full input read + one output write.
    const bytes_per_iter: usize = (m * n + out_elems) * @sizeOf(f32);
    const bytes: u64 = @as(u64, @intCast(bytes_per_iter)) * @as(u64, @intCast(iters));
    const gib_s: f64 = fmtRateGiBPerSec(bytes, ns);

    const out_bytes_len: usize = out_elems * @sizeOf(f32);
    const out_buf: []u8 = try allocator.alloc(u8, out_bytes_len);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(out_tid, out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    const idx0: usize = 0;
    const idx1: usize = out_elems / 2;
    const idx2: usize = out_elems - 1;
    const chk: f32 = out_vals[idx0] + out_vals[idx1] + out_vals[idx2];

    std.debug.print("program reduce axis mean f32 {s}: {d:.3} GiB/s  (m={} n={} axis={}, out={}, chk={d:.4}, wall={d:.3} ms, iter={d:.3} ms)\n", .{
        label,
        gib_s,
        m,
        n,
        axis,
        out_elems,
        chk,
        nsToMilliseconds(ns),
        nsToMillisecondsPerIter(ns, iters),
    });
}

fn benchProgramSoftmaxF32(allocator: std.mem.Allocator, rnd: std.Random, iters: usize, m: usize, n: usize, be: Backend) !void {
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = defaultTilePolicy();
    const st = plan_mod.chooseSoftmaxTiles(policy, m, n);

    const x: []f32 = try allocator.alloc(f32, m * n);
    defer allocator.free(x);
    fillRandomF32(rnd, x);

    const x_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{ m, n }, &[_]usize{ st.tm, st.tn }, .{ .tile_alignment = policy.tile_alignment });
    try sm.writeFromPackedScalar(x_tid, std.mem.sliceAsBytes(x));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f32, &[_]usize{ m, n });
    try g.bindExternal(x_in, @intCast(x_tid));
    const y = try g.addSoftmax(x_in, -1);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    var prog = try program_mod.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    const ns: u64 = try benchLoop(iters, struct {
        be: Backend,
        sm: *StorageManager,
        prog: *const program_mod.Program,
        fn run(self: @This()) !void {
            try self.be.executeProgram(self.prog, self.sm.tensorStore());
        }
    }{ .be = be, .sm = &sm, .prog = &prog });

    // Report as actual memory traffic (2 reads of input, 3 accesses to output).
    const bytes: u64 = @as(u64, @intCast(2 * m * n * @sizeOf(f32))) * @as(u64, @intCast(iters));
    const gib_s: f64 = fmtRateGiBPerSec(bytes, ns);

    const out_tid: TensorId = prog.outputs[0];
    const chk: f32 = try readF32AtTiled(&sm, out_tid, 0, 0) +
        try readF32AtTiled(&sm, out_tid, m / 2, n / 2) +
        try readF32AtTiled(&sm, out_tid, m - 1, n - 1);
    std.debug.print("program softmax f32:     {d:.3} GiB/s  (chk={d:.4}, wall={d:.3} ms, iter={d:.3} ms)\n", .{ gib_s, chk, nsToMilliseconds(ns), nsToMillisecondsPerIter(ns, iters) });
}

fn benchProgramLayerNormF32(allocator: std.mem.Allocator, rnd: std.Random, iters: usize, m: usize, n: usize, be: Backend) !void {
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = defaultTilePolicy();
    const st = plan_mod.chooseNormTiles(policy, m, n);

    const x: []f32 = try allocator.alloc(f32, m * n);
    defer allocator.free(x);
    const gamma: []f32 = try allocator.alloc(f32, n);
    defer allocator.free(gamma);
    const beta: []f32 = try allocator.alloc(f32, n);
    defer allocator.free(beta);
    fillRandomF32(rnd, x);
    fillRandomF32(rnd, gamma);
    fillRandomF32(rnd, beta);

    const x_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{ m, n }, &[_]usize{ st.tm, st.tn }, .{ .tile_alignment = policy.tile_alignment });
    const g_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{n}, &[_]usize{st.tn}, .{ .tile_alignment = policy.tile_alignment });
    const b_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{n}, &[_]usize{st.tn}, .{ .tile_alignment = policy.tile_alignment });
    try sm.writeFromPackedScalar(x_tid, std.mem.sliceAsBytes(x));
    try sm.writeFromPackedScalar(g_tid, std.mem.sliceAsBytes(gamma));
    try sm.writeFromPackedScalar(b_tid, std.mem.sliceAsBytes(beta));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f32, &[_]usize{ m, n });
    const gamma_in = try g.addInput(.f32, &[_]usize{n});
    const beta_in = try g.addInput(.f32, &[_]usize{n});
    try g.bindExternal(x_in, @intCast(x_tid));
    try g.bindExternal(gamma_in, @intCast(g_tid));
    try g.bindExternal(beta_in, @intCast(b_tid));
    const norm_shape: [1]usize = .{n};
    const y = try g.addLayerNorm(x_in, gamma_in, beta_in, 1e-5, norm_shape[0..]);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    var prog = try program_mod.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    const ns: u64 = try benchLoop(iters, struct {
        be: Backend,
        sm: *StorageManager,
        prog: *const program_mod.Program,
        fn run(self: @This()) !void {
            try self.be.executeProgram(self.prog, self.sm.tensorStore());
        }
    }{ .be = be, .sm = &sm, .prog = &prog });

    // Our implementation reads X twice (stats + apply), reads gamma/beta once, writes Y once.
    const bytes: u64 = @as(u64, @intCast(5 * m * n * @sizeOf(f32))) * @as(u64, @intCast(iters));
    const gib_s: f64 = fmtRateGiBPerSec(bytes, ns);

    const out_tid: TensorId = prog.outputs[0];
    const chk: f32 = try readF32AtTiled(&sm, out_tid, 0, 0) +
        try readF32AtTiled(&sm, out_tid, m / 2, n / 2) +
        try readF32AtTiled(&sm, out_tid, m - 1, n - 1);
    std.debug.print("program layernorm f32:   {d:.3} GiB/s  (chk={d:.4}, wall={d:.3} ms, iter={d:.3} ms)\n", .{ gib_s, chk, nsToMilliseconds(ns), nsToMillisecondsPerIter(ns, iters) });
}

fn benchProgramRMSNormF32(allocator: std.mem.Allocator, rnd: std.Random, iters: usize, m: usize, n: usize, be: Backend) !void {
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = defaultTilePolicy();
    const st = plan_mod.chooseNormTiles(policy, m, n);

    const x: []f32 = try allocator.alloc(f32, m * n);
    defer allocator.free(x);
    const gamma: []f32 = try allocator.alloc(f32, n);
    defer allocator.free(gamma);
    const beta: []f32 = try allocator.alloc(f32, n);
    defer allocator.free(beta);
    fillRandomF32(rnd, x);
    fillRandomF32(rnd, gamma);
    fillRandomF32(rnd, beta);

    const x_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{ m, n }, &[_]usize{ st.tm, st.tn }, .{ .tile_alignment = policy.tile_alignment });
    const g_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{n}, &[_]usize{st.tn}, .{ .tile_alignment = policy.tile_alignment });
    const b_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{n}, &[_]usize{st.tn}, .{ .tile_alignment = policy.tile_alignment });
    try sm.writeFromPackedScalar(x_tid, std.mem.sliceAsBytes(x));
    try sm.writeFromPackedScalar(g_tid, std.mem.sliceAsBytes(gamma));
    try sm.writeFromPackedScalar(b_tid, std.mem.sliceAsBytes(beta));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f32, &[_]usize{ m, n });
    const gamma_in = try g.addInput(.f32, &[_]usize{n});
    const beta_in = try g.addInput(.f32, &[_]usize{n});
    try g.bindExternal(x_in, @intCast(x_tid));
    try g.bindExternal(gamma_in, @intCast(g_tid));
    try g.bindExternal(beta_in, @intCast(b_tid));
    const norm_shape: [1]usize = .{n};
    const y = try g.addRMSNorm(x_in, gamma_in, beta_in, 1e-5, norm_shape[0..]);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    var prog = try program_mod.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    const ns: u64 = try benchLoop(iters, struct {
        be: Backend,
        sm: *StorageManager,
        prog: *const program_mod.Program,
        fn run(self: @This()) !void {
            try self.be.executeProgram(self.prog, self.sm.tensorStore());
        }
    }{ .be = be, .sm = &sm, .prog = &prog });

    const bytes: u64 = @as(u64, @intCast(5 * m * n * @sizeOf(f32))) * @as(u64, @intCast(iters));
    const gib_s: f64 = fmtRateGiBPerSec(bytes, ns);

    const out_tid: TensorId = prog.outputs[0];
    const chk: f32 = try readF32AtTiled(&sm, out_tid, 0, 0) +
        try readF32AtTiled(&sm, out_tid, m / 2, n / 2) +
        try readF32AtTiled(&sm, out_tid, m - 1, n - 1);
    std.debug.print("program rmsnorm f32:     {d:.3} GiB/s  (chk={d:.4}, wall={d:.3} ms, iter={d:.3} ms)\n", .{ gib_s, chk, nsToMilliseconds(ns), nsToMillisecondsPerIter(ns, iters) });
}

fn benchProgramAttentionF32(
    allocator: std.mem.Allocator,
    rnd: std.Random,
    iters: usize,
    m: usize,
    n: usize,
    dk: usize,
    dv: usize,
    causal: bool,
    be: Backend,
) !void {
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = defaultTilePolicy();
    const st = plan_mod.chooseAttentionTiles(policy, m, n, dk, dv);

    const q: []f32 = try allocator.alloc(f32, m * dk);
    defer allocator.free(q);
    const k: []f32 = try allocator.alloc(f32, n * dk);
    defer allocator.free(k);
    const v: []f32 = try allocator.alloc(f32, n * dv);
    defer allocator.free(v);
    fillRandomF32(rnd, q);
    fillRandomF32(rnd, k);
    fillRandomF32(rnd, v);

    const q_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{ m, dk }, &[_]usize{ st.tm, st.tk }, .{ .tile_alignment = policy.tile_alignment });
    const k_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{ n, dk }, &[_]usize{ st.tn, st.tk }, .{ .tile_alignment = policy.tile_alignment });
    const v_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{ n, dv }, &[_]usize{ st.tn, st.tv }, .{ .tile_alignment = policy.tile_alignment });
    try sm.writeFromPackedScalar(q_tid, std.mem.sliceAsBytes(q));
    try sm.writeFromPackedScalar(k_tid, std.mem.sliceAsBytes(k));
    try sm.writeFromPackedScalar(v_tid, std.mem.sliceAsBytes(v));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const q_in = try g.addInput(.f32, &[_]usize{ m, dk });
    const k_in = try g.addInput(.f32, &[_]usize{ n, dk });
    const v_in = try g.addInput(.f32, &[_]usize{ n, dv });
    try g.bindExternal(q_in, @intCast(q_tid));
    try g.bindExternal(k_in, @intCast(k_tid));
    try g.bindExternal(v_in, @intCast(v_tid));

    const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(dk)));
    const y = try g.addAttention(q_in, k_in, v_in, scale, causal);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    var prog = try program_mod.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    const ns: u64 = try benchLoop(iters, struct {
        be: Backend,
        sm: *StorageManager,
        prog: *const program_mod.Program,
        fn run(self: @This()) !void {
            try self.be.executeProgram(self.prog, self.sm.tensorStore());
        }
    }{ .be = be, .sm = &sm, .prog = &prog });

    // FLOPs (ignoring softmax exp overhead):
    // - QK^T: 2*m*n*dk
    // - P@V:  2*m*n*dv
    const flops_per: u64 = 2 * @as(u64, @intCast(m)) * @as(u64, @intCast(n)) * (@as(u64, @intCast(dk)) + @as(u64, @intCast(dv)));
    const gflops: f64 = fmtRateGFLOPs(flops_per * @as(u64, @intCast(iters)), ns);

    const out_tid: TensorId = prog.outputs[0];
    const chk: f32 = try readF32AtTiled(&sm, out_tid, 0, 0) +
        try readF32AtTiled(&sm, out_tid, m / 2, dv / 2) +
        try readF32AtTiled(&sm, out_tid, m - 1, dv - 1);

    const mode: []const u8 = if (causal) "causal" else "noncausal";
    std.debug.print(
        "program attention f32 {s}: {d:.2} GFLOP/s (m={} n={} dk={} dv={}, chk={d:.4}, wall={d:.3} ms, iter={d:.3} ms)\n",
        .{ mode, gflops, m, n, dk, dv, chk, nsToMilliseconds(ns), nsToMillisecondsPerIter(ns, iters) },
    );
}

fn benchProgramMultiHeadAttentionSeparateF32(
    allocator: std.mem.Allocator,
    rnd: std.Random,
    iters: usize,
    batch: usize,
    heads: usize,
    m_head: usize,
    n_head: usize,
    dk: usize,
    dv: usize,
    causal: bool,
    be: Backend,
) !void {
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = defaultTilePolicy();
    const st = plan_mod.chooseAttentionTiles(policy, m_head, n_head, dk, dv);

    const q: []f32 = try allocator.alloc(f32, batch * heads * m_head * dk);
    defer allocator.free(q);
    const k: []f32 = try allocator.alloc(f32, batch * heads * n_head * dk);
    defer allocator.free(k);
    const v: []f32 = try allocator.alloc(f32, batch * heads * n_head * dv);
    defer allocator.free(v);
    fillRandomF32(rnd, q);
    fillRandomF32(rnd, k);
    fillRandomF32(rnd, v);

    const q_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{ batch, heads, m_head, dk }, &[_]usize{ 1, 1, st.tm, st.tk }, .{ .tile_alignment = policy.tile_alignment });
    const k_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{ batch, heads, n_head, dk }, &[_]usize{ 1, 1, st.tn, st.tk }, .{ .tile_alignment = policy.tile_alignment });
    const v_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{ batch, heads, n_head, dv }, &[_]usize{ 1, 1, st.tn, st.tv }, .{ .tile_alignment = policy.tile_alignment });
    try sm.writeFromPackedScalar(q_tid, std.mem.sliceAsBytes(q));
    try sm.writeFromPackedScalar(k_tid, std.mem.sliceAsBytes(k));
    try sm.writeFromPackedScalar(v_tid, std.mem.sliceAsBytes(v));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const q_in = try g.addInput(.f32, &[_]usize{ batch, heads, m_head, dk });
    const k_in = try g.addInput(.f32, &[_]usize{ batch, heads, n_head, dk });
    const v_in = try g.addInput(.f32, &[_]usize{ batch, heads, n_head, dv });
    try g.bindExternal(q_in, @intCast(q_tid));
    try g.bindExternal(k_in, @intCast(k_tid));
    try g.bindExternal(v_in, @intCast(v_tid));

    const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(dk)));
    const y = try g.addMultiHeadAttention(q_in, k_in, v_in, scale, causal, heads);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    var prog = try program_mod.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    const ns: u64 = try benchLoop(iters, struct {
        be: Backend,
        sm: *StorageManager,
        prog: *const program_mod.Program,
        fn run(self: @This()) !void {
            try self.be.executeProgram(self.prog, self.sm.tensorStore());
        }
    }{ .be = be, .sm = &sm, .prog = &prog });

    const flops_per: u64 = 2 * @as(u64, @intCast(batch)) * @as(u64, @intCast(heads)) * @as(u64, @intCast(m_head)) * @as(u64, @intCast(n_head)) *
        (@as(u64, @intCast(dk)) + @as(u64, @intCast(dv)));
    const gflops: f64 = fmtRateGFLOPs(flops_per * @as(u64, @intCast(iters)), ns);

    const out_tid: TensorId = prog.outputs[0];
    const out_bytes_len: usize = batch * heads * m_head * dv * @sizeOf(f32);
    const out_buf: []u8 = try allocator.alloc(u8, out_bytes_len);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(out_tid, out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    const idx0: usize = 0;
    const idx1: usize = (((batch / 2) * heads + (heads / 2)) * m_head + (m_head / 2)) * dv + (dv / 2);
    const idx2: usize = (((batch - 1) * heads + (heads - 1)) * m_head + (m_head - 1)) * dv + (dv - 1);
    const chk: f32 = out_vals[idx0] + out_vals[idx1] + out_vals[idx2];

    const mode: []const u8 = if (causal) "causal" else "noncausal";
    std.debug.print(
        "program mha separate f32 {s}: {d:.2} GFLOP/s (b={} h={} m={} n={} dk={} dv={}, chk={d:.4}, wall={d:.3} ms, iter={d:.3} ms)\n",
        .{ mode, gflops, batch, heads, m_head, n_head, dk, dv, chk, nsToMilliseconds(ns), nsToMillisecondsPerIter(ns, iters) },
    );
}

fn benchProgramMultiHeadAttentionCachedF32(
    allocator: std.mem.Allocator,
    rnd: std.Random,
    iters: usize,
    batch: usize,
    h_q: usize,
    h_kv: usize,
    l_q: usize,
    t_cache: usize,
    dk: usize,
    dv: usize,
    causal: bool,
    sliding_window: usize,
    attn_logits_soft_cap: f32,
    label: []const u8,
    be: Backend,
) !void {
    if (batch == 0 or h_q == 0 or h_kv == 0 or l_q == 0 or t_cache == 0 or dk == 0 or dv == 0) return error.InvalidArgument;
    if ((h_q % h_kv) != 0) return error.InvalidArgument;
    if (!(attn_logits_soft_cap >= 0.0) or !std.math.isFinite(attn_logits_soft_cap)) return error.InvalidArgument;

    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = defaultTilePolicy();
    const st = plan_mod.chooseAttentionTiles(policy, l_q, t_cache, dk, dv);

    const q: []f32 = try allocator.alloc(f32, batch * l_q * h_q * dk);
    defer allocator.free(q);
    const k_cache: []f32 = try allocator.alloc(f32, batch * h_kv * t_cache * dk);
    defer allocator.free(k_cache);
    const v_cache: []f32 = try allocator.alloc(f32, batch * h_kv * t_cache * dv);
    defer allocator.free(v_cache);
    const positions: []i32 = try allocator.alloc(i32, batch * l_q);
    defer allocator.free(positions);
    const end_index: []i32 = try allocator.alloc(i32, batch);
    defer allocator.free(end_index);

    fillRandomF32(rnd, q);
    fillRandomF32(rnd, k_cache);
    fillRandomF32(rnd, v_cache);

    // Place query positions near the current cache tail so causal/local windows
    // exercise realistic decode-time slices.
    const base_pos: usize = if (t_cache > l_q) t_cache - l_q else 0;
    for (0..batch) |b0| {
        end_index[b0] = @intCast(t_cache);
        for (0..l_q) |l| {
            positions[b0 * l_q + l] = @intCast(base_pos + l);
        }
    }

    const q_tid: TensorId = try sm.createTiledTensor(
        .f32,
        &[_]usize{ batch, l_q, h_q, dk },
        &[_]usize{ 1, st.tm, 1, dk },
        .{ .tile_alignment = policy.tile_alignment },
    );
    const k_tid: TensorId = try sm.createTiledTensor(
        .f32,
        &[_]usize{ batch, h_kv, t_cache, dk },
        &[_]usize{ 1, 1, st.tn, dk },
        .{ .tile_alignment = policy.tile_alignment },
    );
    const v_tid: TensorId = try sm.createTiledTensor(
        .f32,
        &[_]usize{ batch, h_kv, t_cache, dv },
        &[_]usize{ 1, 1, st.tn, dv },
        .{ .tile_alignment = policy.tile_alignment },
    );
    const pos_tid: TensorId = try sm.createTiledTensor(
        .i32,
        &[_]usize{ batch, l_q },
        &[_]usize{ 1, st.tm },
        .{ .tile_alignment = policy.tile_alignment },
    );
    const end_tid: TensorId = try sm.createTiledTensor(
        .i32,
        &[_]usize{batch},
        &[_]usize{batch},
        .{ .tile_alignment = policy.tile_alignment },
    );

    try sm.writeFromPackedScalar(q_tid, std.mem.sliceAsBytes(q));
    try sm.writeFromPackedScalar(k_tid, std.mem.sliceAsBytes(k_cache));
    try sm.writeFromPackedScalar(v_tid, std.mem.sliceAsBytes(v_cache));
    try sm.writeFromPackedScalar(pos_tid, std.mem.sliceAsBytes(positions));
    try sm.writeFromPackedScalar(end_tid, std.mem.sliceAsBytes(end_index));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const q_in = try g.addInput(.f32, &[_]usize{ batch, l_q, h_q, dk });
    const k_in = try g.addInput(.f32, &[_]usize{ batch, h_kv, t_cache, dk });
    const v_in = try g.addInput(.f32, &[_]usize{ batch, h_kv, t_cache, dv });
    const p_in = try g.addInput(.i32, &[_]usize{ batch, l_q });
    const e_in = try g.addInput(.i32, &[_]usize{batch});
    try g.bindExternal(q_in, @intCast(q_tid));
    try g.bindExternal(k_in, @intCast(k_tid));
    try g.bindExternal(v_in, @intCast(v_tid));
    try g.bindExternal(p_in, @intCast(pos_tid));
    try g.bindExternal(e_in, @intCast(end_tid));

    const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(dk)));
    const y = try g.addMultiHeadAttentionCached(q_in, k_in, v_in, p_in, e_in, scale, causal, sliding_window, attn_logits_soft_cap);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    var prog = try program_mod.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    const ns: u64 = try benchLoop(iters, struct {
        be: Backend,
        sm: *StorageManager,
        prog: *const program_mod.Program,
        fn run(self: @This()) !void {
            try self.be.executeProgram(self.prog, self.sm.tensorStore());
        }
    }{ .be = be, .sm = &sm, .prog = &prog });

    // Approx FLOPs (ignoring exp/tanh overhead):
    // per (b,l,h): 2 * n_eff * (dk + dv)
    // n_eff is derived from the same range logic used in cached attention.
    var n_eff_sum: u64 = 0;
    for (0..l_q) |l| {
        const q_pos: usize = base_pos + l;
        var upper: usize = t_cache;
        if (causal) {
            const q_pos_next: usize = std.math.add(usize, q_pos, 1) catch return error.InvalidArgument;
            if (q_pos_next < upper) upper = q_pos_next;
        }

        var lower: usize = 0;
        if (sliding_window > 0) {
            const q_pos_next: usize = std.math.add(usize, q_pos, 1) catch return error.InvalidArgument;
            const sw_lower: usize = if (q_pos_next > sliding_window) q_pos_next - sliding_window else 0;
            if (sw_lower > lower) lower = sw_lower;
        }

        if (upper > lower) n_eff_sum += @intCast(upper - lower);
    }

    const flops_total_wide: u128 = @as(u128, 2) *
        @as(u128, batch) *
        @as(u128, h_q) *
        @as(u128, n_eff_sum) *
        (@as(u128, dk) + @as(u128, dv)) *
        @as(u128, iters);
    const flops_total: u64 = if (flops_total_wide > std.math.maxInt(u64)) std.math.maxInt(u64) else @intCast(flops_total_wide);
    const gflops: f64 = fmtRateGFLOPs(flops_total, ns);

    const out_tid: TensorId = prog.outputs[0];
    const out_bytes_len: usize = batch * l_q * h_q * dv * @sizeOf(f32);
    const out_buf: []u8 = try allocator.alloc(u8, out_bytes_len);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(out_tid, out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    const idx0: usize = 0;
    const idx1: usize = (((batch / 2) * l_q + (l_q / 2)) * h_q + (h_q / 2)) * dv + (dv / 2);
    const idx2: usize = (((batch - 1) * l_q + (l_q - 1)) * h_q + (h_q - 1)) * dv + (dv - 1);
    const chk: f32 = out_vals[idx0] + out_vals[idx1] + out_vals[idx2];

    const mode: []const u8 = if (causal) "causal" else "noncausal";
    std.debug.print(
        "program mha cached f32 {s}/{s}: {d:.2} GFLOP/s (b={} hq={} hkv={} lq={} t={} dk={} dv={} sw={} cap={d:.2}, chk={d:.4}, wall={d:.3} ms, iter={d:.3} ms)\n",
        .{ mode, label, gflops, batch, h_q, h_kv, l_q, t_cache, dk, dv, sliding_window, attn_logits_soft_cap, chk, nsToMilliseconds(ns), nsToMillisecondsPerIter(ns, iters) },
    );
}

fn benchProgramSTFTF32(
    allocator: std.mem.Allocator,
    rnd: std.Random,
    iters: usize,
    batch: usize,
    samples: usize,
    n_fft: usize,
    hop: usize,
    be: Backend,
) !void {
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = defaultTilePolicy();

    const sig: []f32 = try allocator.alloc(f32, batch * samples);
    defer allocator.free(sig);
    fillRandomF32(rnd, sig);

    const win: []f32 = try allocator.alloc(f32, n_fft);
    defer allocator.free(win);
    for (0..n_fft) |i| win[i] = 0.5 - 0.5 * @cos(2.0 * std.math.pi * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n_fft)));

    // Single-tile inputs keep setup simple; the op reads via scalar tile access.
    const sig_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{ batch, samples }, &[_]usize{ batch, samples }, .{ .tile_alignment = policy.tile_alignment });
    try sm.writeFromPackedScalar(sig_tid, std.mem.sliceAsBytes(sig));
    const win_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{n_fft}, &[_]usize{n_fft}, .{ .tile_alignment = policy.tile_alignment });
    try sm.writeFromPackedScalar(win_tid, std.mem.sliceAsBytes(win));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const sig_in = try g.addInput(.f32, &[_]usize{ batch, samples });
    try g.bindExternal(sig_in, @intCast(sig_tid));
    const win_in = try g.addInput(.f32, &[_]usize{n_fft});
    try g.bindExternal(win_in, @intCast(win_tid));
    const y = try g.addSTFT(sig_in, win_in, n_fft, hop, true);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    var prog = try program_mod.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    const ns: u64 = try benchLoop(iters, struct {
        be: Backend,
        sm: *StorageManager,
        prog: *const program_mod.Program,
        fn run(self: @This()) !void {
            try self.be.executeProgram(self.prog, self.sm.tensorStore());
        }
    }{ .be = be, .sm = &sm, .prog = &prog });

    const num_frames: usize = 1 + samples / hop;
    const out_tid: TensorId = prog.outputs[0];
    const out_elems: usize = batch * num_frames * (n_fft + 2);
    const out_buf: []f32 = try allocator.alloc(f32, out_elems);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(out_tid, std.mem.sliceAsBytes(out_buf));
    const chk: f32 = out_buf[0] + out_buf[@min(out_elems - 1, n_fft + 2)];

    // FLOP estimate: per frame, window multiply (~N) + real FFT (~2.5*N*log2 N),
    // over batch*num_frames frames.
    const log2n: f64 = std.math.log2(@as(f64, @floatFromInt(n_fft)));
    const flops_per_frame: f64 = @as(f64, @floatFromInt(n_fft)) + 2.5 * @as(f64, @floatFromInt(n_fft)) * log2n;
    const total_flops: f64 = flops_per_frame * @as(f64, @floatFromInt(batch * num_frames * iters));
    const gflops: f64 = total_flops / @as(f64, @floatFromInt(ns));
    std.debug.print("program stft f32 (nfft={d} hop={d} frames={d}): {d:.3} GFLOP/s  (chk={d:.4}, wall={d:.3} ms, iter={d:.3} ms)\n", .{ n_fft, hop, num_frames, gflops, chk, nsToMilliseconds(ns), nsToMillisecondsPerIter(ns, iters) });
}

fn benchProgramRoPE1DF32(
    allocator: std.mem.Allocator,
    rnd: std.Random,
    iters: usize,
    batch: usize,
    seq_len: usize,
    heads: usize,
    head_dim: usize,
    be: Backend,
) !void {
    if (head_dim == 0 or (head_dim % 2) != 0) return error.InvalidArgument;

    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = defaultTilePolicy();
    const t_l: [1]usize = plan_mod.chooseTileShape1D(policy, seq_len);

    const x_elems: usize = batch * seq_len * heads * head_dim;
    const p_elems: usize = batch * seq_len;

    const x: []f32 = try allocator.alloc(f32, x_elems);
    defer allocator.free(x);
    fillRandomF32(rnd, x);

    const positions: []i32 = try allocator.alloc(i32, p_elems);
    defer allocator.free(positions);
    for (0..batch) |b0| {
        for (0..seq_len) |t| {
            const idx: usize = b0 * seq_len + t;
            positions[idx] = @intCast(t + b0 * 7);
        }
    }

    const x_tid: TensorId = try sm.createTiledTensor(
        .f32,
        &[_]usize{ batch, seq_len, heads, head_dim },
        &[_]usize{ 1, t_l[0], 1, head_dim },
        .{ .tile_alignment = policy.tile_alignment },
    );
    const p_tid: TensorId = try sm.createTiledTensor(
        .i32,
        &[_]usize{ batch, seq_len },
        &[_]usize{ 1, t_l[0] },
        .{ .tile_alignment = policy.tile_alignment },
    );

    try sm.writeFromPackedScalar(x_tid, std.mem.sliceAsBytes(x));
    try sm.writeFromPackedScalar(p_tid, std.mem.sliceAsBytes(positions));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f32, &[_]usize{ batch, seq_len, heads, head_dim });
    const p_in = try g.addInput(.i32, &[_]usize{ batch, seq_len });
    try g.bindExternal(x_in, @intCast(x_tid));
    try g.bindExternal(p_in, @intCast(p_tid));

    const y = try g.addRoPE1D(x_in, p_in, 10000.0, 1.0, 1.0);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    var prog = try program_mod.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    const ns: u64 = try benchLoop(iters, struct {
        be: Backend,
        sm: *StorageManager,
        prog: *const program_mod.Program,
        fn run(self: @This()) !void {
            try self.be.executeProgram(self.prog, self.sm.tensorStore());
        }
    }{ .be = be, .sm = &sm, .prog = &prog });

    // Approx traffic per iter: read x + read positions + write out.
    const bytes_per_iter: usize = (2 * x_elems * @sizeOf(f32)) + (p_elems * @sizeOf(i32));
    const bytes: u64 = @as(u64, @intCast(bytes_per_iter)) * @as(u64, @intCast(iters));
    const gib_s: f64 = fmtRateGiBPerSec(bytes, ns);

    const out_buf: []u8 = try allocator.alloc(u8, x_elems * @sizeOf(f32));
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    const idx0: usize = 0;
    const idx1: usize = (((batch / 2) * seq_len + (seq_len / 2)) * heads + (heads / 2)) * head_dim + (head_dim / 2);
    const idx2: usize = (((batch - 1) * seq_len + (seq_len - 1)) * heads + (heads - 1)) * head_dim + (head_dim - 1);
    const chk: f32 = out_vals[idx0] + out_vals[idx1] + out_vals[idx2];

    std.debug.print(
        "program rope1d f32:      {d:.3} GiB/s  (b={} l={} h={} d={}, chk={d:.4}, wall={d:.3} ms, iter={d:.3} ms)\n",
        .{ gib_s, batch, seq_len, heads, head_dim, chk, nsToMilliseconds(ns), nsToMillisecondsPerIter(ns, iters) },
    );
}

fn benchProgramElemwiseAddF16(allocator: std.mem.Allocator, rnd: std.Random, iters: usize, n_elem: usize, be: Backend) !void {
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = defaultTilePolicy();
    const t1: [1]usize = plan_mod.chooseTileShape1D(policy, n_elem);

    const a: []f16 = try allocator.alloc(f16, n_elem);
    defer allocator.free(a);
    const b: []f16 = try allocator.alloc(f16, n_elem);
    defer allocator.free(b);
    fillRandomF16(rnd, a);
    fillRandomF16(rnd, b);

    const a_tid: TensorId = try sm.createTiledTensor(.f16, &[_]usize{n_elem}, &[_]usize{t1[0]}, .{ .tile_alignment = policy.tile_alignment });
    const b_tid: TensorId = try sm.createTiledTensor(.f16, &[_]usize{n_elem}, &[_]usize{t1[0]}, .{ .tile_alignment = policy.tile_alignment });
    try sm.writeFromPackedScalar(a_tid, std.mem.sliceAsBytes(a));
    try sm.writeFromPackedScalar(b_tid, std.mem.sliceAsBytes(b));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const a_in = try g.addInput(.f16, &[_]usize{n_elem});
    const b_in = try g.addInput(.f16, &[_]usize{n_elem});
    try g.bindExternal(a_in, @intCast(a_tid));
    try g.bindExternal(b_in, @intCast(b_tid));
    const out = try g.addElemwiseBinary(.add, a_in, b_in);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    var prog = try program_mod.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    const ns: u64 = try benchLoop(iters, struct {
        be: Backend,
        sm: *StorageManager,
        prog: *const program_mod.Program,
        fn run(self: @This()) !void {
            try self.be.executeProgram(self.prog, self.sm.tensorStore());
        }
    }{ .be = be, .sm = &sm, .prog = &prog });

    const bytes: u64 = @as(u64, @intCast(3 * n_elem * @sizeOf(f16))) * @as(u64, @intCast(iters));
    const gib_s: f64 = fmtRateGiBPerSec(bytes, ns);

    const out_tid: TensorId = prog.outputs[0];
    const chk: f32 = @as(f32, @floatCast(try readF16AtTiled(&sm, out_tid, 0, 0))) +
        @as(f32, @floatCast(try readF16AtTiled(&sm, out_tid, n_elem / 2, 0))) +
        @as(f32, @floatCast(try readF16AtTiled(&sm, out_tid, n_elem - 1, 0)));
    std.debug.print("program elemwise add f16: {d:.3} GiB/s  (chk={d:.4}, wall={d:.3} ms, iter={d:.3} ms)\n", .{ gib_s, chk, nsToMilliseconds(ns), nsToMillisecondsPerIter(ns, iters) });
}

fn benchProgramUnaryF16(
    allocator: std.mem.Allocator,
    rnd: std.Random,
    iters: usize,
    n_elem: usize,
    op: types.UnaryOp,
    label: []const u8,
    be: Backend,
) !void {
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = defaultTilePolicy();
    const t1: [1]usize = plan_mod.chooseTileShape1D(policy, n_elem);

    const a: []f16 = try allocator.alloc(f16, n_elem);
    defer allocator.free(a);
    fillRandomF16(rnd, a);

    const a_tid: TensorId = try sm.createTiledTensor(.f16, &[_]usize{n_elem}, &[_]usize{t1[0]}, .{ .tile_alignment = policy.tile_alignment });
    try sm.writeFromPackedScalar(a_tid, std.mem.sliceAsBytes(a));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const a_in = try g.addInput(.f16, &[_]usize{n_elem});
    try g.bindExternal(a_in, @intCast(a_tid));
    const out = try g.addUnary(op, a_in);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    var prog = try program_mod.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    const ns: u64 = try benchLoop(iters, struct {
        be: Backend,
        sm: *StorageManager,
        prog: *const program_mod.Program,
        fn run(self: @This()) !void {
            try self.be.executeProgram(self.prog, self.sm.tensorStore());
        }
    }{ .be = be, .sm = &sm, .prog = &prog });

    const bytes: u64 = @as(u64, @intCast(2 * n_elem * @sizeOf(f16))) * @as(u64, @intCast(iters));
    const gib_s: f64 = fmtRateGiBPerSec(bytes, ns);

    const out_tid: TensorId = prog.outputs[0];
    const chk: f32 = @as(f32, @floatCast(try readF16AtTiled(&sm, out_tid, 0, 0))) +
        @as(f32, @floatCast(try readF16AtTiled(&sm, out_tid, n_elem / 2, 0))) +
        @as(f32, @floatCast(try readF16AtTiled(&sm, out_tid, n_elem - 1, 0)));
    std.debug.print("program unary {s} f16:    {d:.3} GiB/s  (chk={d:.4}, wall={d:.3} ms, iter={d:.3} ms)\n", .{ label, gib_s, chk, nsToMilliseconds(ns), nsToMillisecondsPerIter(ns, iters) });
}

fn benchProgramReduceSumF16(allocator: std.mem.Allocator, rnd: std.Random, iters: usize, n_elem: usize, be: Backend) !void {
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = defaultTilePolicy();
    const t1: [1]usize = plan_mod.chooseTileShape1D(policy, n_elem);

    const a: []f16 = try allocator.alloc(f16, n_elem);
    defer allocator.free(a);
    fillRandomF16(rnd, a);

    const a_tid: TensorId = try sm.createTiledTensor(.f16, &[_]usize{n_elem}, &[_]usize{t1[0]}, .{ .tile_alignment = policy.tile_alignment });
    try sm.writeFromPackedScalar(a_tid, std.mem.sliceAsBytes(a));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const a_in = try g.addInput(.f16, &[_]usize{n_elem});
    try g.bindExternal(a_in, @intCast(a_tid));
    const out = try g.addReduce(.sum, a_in);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    var prog = try program_mod.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    const ns: u64 = try benchLoop(iters, struct {
        be: Backend,
        sm: *StorageManager,
        prog: *const program_mod.Program,
        fn run(self: @This()) !void {
            try self.be.executeProgram(self.prog, self.sm.tensorStore());
        }
    }{ .be = be, .sm = &sm, .prog = &prog });

    const bytes: u64 = @as(u64, @intCast(n_elem * @sizeOf(f16))) * @as(u64, @intCast(iters));
    const gib_s: f64 = fmtRateGiBPerSec(bytes, ns);

    const out_tid: TensorId = prog.outputs[0];
    const sum: f32 = @floatCast(try readF16AtTiled(&sm, out_tid, 0, 0));
    std.debug.print("program reduce sum f16:   {d:.3} GiB/s  (sum={d:.4}, wall={d:.3} ms, iter={d:.3} ms)\n", .{ gib_s, sum, nsToMilliseconds(ns), nsToMillisecondsPerIter(ns, iters) });
}

fn benchProgramReduceAxisF16(
    allocator: std.mem.Allocator,
    rnd: std.Random,
    iters: usize,
    m: usize,
    n: usize,
    axis: i32,
    label: []const u8,
    be: Backend,
) !void {
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = defaultTilePolicy();
    const t2: [2]usize = plan_mod.chooseTileShape2DSquare(policy, m, n);

    const a: []f16 = try allocator.alloc(f16, m * n);
    defer allocator.free(a);
    fillRandomF16(rnd, a);

    const a_tid: TensorId = try sm.createTiledTensor(.f16, &[_]usize{ m, n }, &[_]usize{ t2[0], t2[1] }, .{ .tile_alignment = policy.tile_alignment });
    try sm.writeFromPackedScalar(a_tid, std.mem.sliceAsBytes(a));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const a_in = try g.addInput(.f16, &[_]usize{ m, n });
    try g.bindExternal(a_in, @intCast(a_tid));
    const out = try g.addReduceAxis(.mean, a_in, axis);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    var prog = try program_mod.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    const ns: u64 = try benchLoop(iters, struct {
        be: Backend,
        sm: *StorageManager,
        prog: *const program_mod.Program,
        fn run(self: @This()) !void {
            try self.be.executeProgram(self.prog, self.sm.tensorStore());
        }
    }{ .be = be, .sm = &sm, .prog = &prog });

    const out_tid: TensorId = prog.outputs[0];
    const out_t = try sm.getConst(out_tid);
    var out_elems: usize = 1;
    for (out_t.shape) |dim| out_elems *= dim;

    const bytes_per_iter: usize = (m * n + out_elems) * @sizeOf(f16);
    const bytes: u64 = @as(u64, @intCast(bytes_per_iter)) * @as(u64, @intCast(iters));
    const gib_s: f64 = fmtRateGiBPerSec(bytes, ns);

    const out_bytes_len: usize = out_elems * @sizeOf(f16);
    const out_buf: []u8 = try allocator.alloc(u8, out_bytes_len);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(out_tid, out_buf);
    const out_vals: []align(1) f16 = asF16Slice(out_buf);

    const idx0: usize = 0;
    const idx1: usize = out_elems / 2;
    const idx2: usize = out_elems - 1;
    const chk: f32 = @as(f32, @floatCast(out_vals[idx0])) + @as(f32, @floatCast(out_vals[idx1])) + @as(f32, @floatCast(out_vals[idx2]));

    std.debug.print("program reduce axis mean f16 {s}: {d:.3} GiB/s  (m={} n={} axis={}, out={}, chk={d:.4}, wall={d:.3} ms, iter={d:.3} ms)\n", .{
        label,
        gib_s,
        m,
        n,
        axis,
        out_elems,
        chk,
        nsToMilliseconds(ns),
        nsToMillisecondsPerIter(ns, iters),
    });
}

fn benchProgramLayerNormF16(allocator: std.mem.Allocator, rnd: std.Random, iters: usize, m: usize, n: usize, be: Backend) !void {
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = defaultTilePolicy();
    const st = plan_mod.chooseNormTiles(policy, m, n);

    const x: []f16 = try allocator.alloc(f16, m * n);
    defer allocator.free(x);
    const gamma: []f16 = try allocator.alloc(f16, n);
    defer allocator.free(gamma);
    const beta: []f16 = try allocator.alloc(f16, n);
    defer allocator.free(beta);
    fillRandomF16(rnd, x);
    fillRandomF16(rnd, gamma);
    fillRandomF16(rnd, beta);

    const x_tid: TensorId = try sm.createTiledTensor(.f16, &[_]usize{ m, n }, &[_]usize{ st.tm, st.tn }, .{ .tile_alignment = policy.tile_alignment });
    const g_tid: TensorId = try sm.createTiledTensor(.f16, &[_]usize{n}, &[_]usize{st.tn}, .{ .tile_alignment = policy.tile_alignment });
    const b_tid: TensorId = try sm.createTiledTensor(.f16, &[_]usize{n}, &[_]usize{st.tn}, .{ .tile_alignment = policy.tile_alignment });
    try sm.writeFromPackedScalar(x_tid, std.mem.sliceAsBytes(x));
    try sm.writeFromPackedScalar(g_tid, std.mem.sliceAsBytes(gamma));
    try sm.writeFromPackedScalar(b_tid, std.mem.sliceAsBytes(beta));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f16, &[_]usize{ m, n });
    const gamma_in = try g.addInput(.f16, &[_]usize{n});
    const beta_in = try g.addInput(.f16, &[_]usize{n});
    try g.bindExternal(x_in, @intCast(x_tid));
    try g.bindExternal(gamma_in, @intCast(g_tid));
    try g.bindExternal(beta_in, @intCast(b_tid));
    const norm_shape: [1]usize = .{n};
    const y = try g.addLayerNorm(x_in, gamma_in, beta_in, 1e-5, norm_shape[0..]);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    var prog = try program_mod.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    const ns: u64 = try benchLoop(iters, struct {
        be: Backend,
        sm: *StorageManager,
        prog: *const program_mod.Program,
        fn run(self: @This()) !void {
            try self.be.executeProgram(self.prog, self.sm.tensorStore());
        }
    }{ .be = be, .sm = &sm, .prog = &prog });

    const bytes: u64 = @as(u64, @intCast(5 * m * n * @sizeOf(f16))) * @as(u64, @intCast(iters));
    const gib_s: f64 = fmtRateGiBPerSec(bytes, ns);

    const out_tid: TensorId = prog.outputs[0];
    const chk: f32 = @as(f32, @floatCast(try readF16AtTiled(&sm, out_tid, 0, 0))) +
        @as(f32, @floatCast(try readF16AtTiled(&sm, out_tid, m / 2, n / 2))) +
        @as(f32, @floatCast(try readF16AtTiled(&sm, out_tid, m - 1, n - 1)));
    std.debug.print("program layernorm f16:   {d:.3} GiB/s  (chk={d:.4}, wall={d:.3} ms, iter={d:.3} ms)\n", .{ gib_s, chk, nsToMilliseconds(ns), nsToMillisecondsPerIter(ns, iters) });
}

fn benchProgramRMSNormF16(allocator: std.mem.Allocator, rnd: std.Random, iters: usize, m: usize, n: usize, be: Backend) !void {
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = defaultTilePolicy();
    const st = plan_mod.chooseNormTiles(policy, m, n);

    const x: []f16 = try allocator.alloc(f16, m * n);
    defer allocator.free(x);
    const gamma: []f16 = try allocator.alloc(f16, n);
    defer allocator.free(gamma);
    const beta: []f16 = try allocator.alloc(f16, n);
    defer allocator.free(beta);
    fillRandomF16(rnd, x);
    fillRandomF16(rnd, gamma);
    fillRandomF16(rnd, beta);

    const x_tid: TensorId = try sm.createTiledTensor(.f16, &[_]usize{ m, n }, &[_]usize{ st.tm, st.tn }, .{ .tile_alignment = policy.tile_alignment });
    const g_tid: TensorId = try sm.createTiledTensor(.f16, &[_]usize{n}, &[_]usize{st.tn}, .{ .tile_alignment = policy.tile_alignment });
    const b_tid: TensorId = try sm.createTiledTensor(.f16, &[_]usize{n}, &[_]usize{st.tn}, .{ .tile_alignment = policy.tile_alignment });
    try sm.writeFromPackedScalar(x_tid, std.mem.sliceAsBytes(x));
    try sm.writeFromPackedScalar(g_tid, std.mem.sliceAsBytes(gamma));
    try sm.writeFromPackedScalar(b_tid, std.mem.sliceAsBytes(beta));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f16, &[_]usize{ m, n });
    const gamma_in = try g.addInput(.f16, &[_]usize{n});
    const beta_in = try g.addInput(.f16, &[_]usize{n});
    try g.bindExternal(x_in, @intCast(x_tid));
    try g.bindExternal(gamma_in, @intCast(g_tid));
    try g.bindExternal(beta_in, @intCast(b_tid));
    const norm_shape: [1]usize = .{n};
    const y = try g.addRMSNorm(x_in, gamma_in, beta_in, 1e-5, norm_shape[0..]);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    var prog = try program_mod.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    const ns: u64 = try benchLoop(iters, struct {
        be: Backend,
        sm: *StorageManager,
        prog: *const program_mod.Program,
        fn run(self: @This()) !void {
            try self.be.executeProgram(self.prog, self.sm.tensorStore());
        }
    }{ .be = be, .sm = &sm, .prog = &prog });

    const bytes: u64 = @as(u64, @intCast(5 * m * n * @sizeOf(f16))) * @as(u64, @intCast(iters));
    const gib_s: f64 = fmtRateGiBPerSec(bytes, ns);

    const out_tid: TensorId = prog.outputs[0];
    const chk: f32 = @as(f32, @floatCast(try readF16AtTiled(&sm, out_tid, 0, 0))) +
        @as(f32, @floatCast(try readF16AtTiled(&sm, out_tid, m / 2, n / 2))) +
        @as(f32, @floatCast(try readF16AtTiled(&sm, out_tid, m - 1, n - 1)));
    std.debug.print("program rmsnorm f16:     {d:.3} GiB/s  (chk={d:.4}, wall={d:.3} ms, iter={d:.3} ms)\n", .{ gib_s, chk, nsToMilliseconds(ns), nsToMillisecondsPerIter(ns, iters) });
}

fn benchProgramRoPE1DF16(
    allocator: std.mem.Allocator,
    rnd: std.Random,
    iters: usize,
    batch: usize,
    seq_len: usize,
    heads: usize,
    head_dim: usize,
    be: Backend,
) !void {
    if (head_dim == 0 or (head_dim % 2) != 0) return error.InvalidArgument;

    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = defaultTilePolicy();
    const t_l: [1]usize = plan_mod.chooseTileShape1D(policy, seq_len);

    const x_elems: usize = batch * seq_len * heads * head_dim;
    const p_elems: usize = batch * seq_len;

    const x: []f16 = try allocator.alloc(f16, x_elems);
    defer allocator.free(x);
    fillRandomF16(rnd, x);

    const positions: []i32 = try allocator.alloc(i32, p_elems);
    defer allocator.free(positions);
    for (0..batch) |b0| {
        for (0..seq_len) |t| {
            const idx: usize = b0 * seq_len + t;
            positions[idx] = @intCast(t + b0 * 7);
        }
    }

    const x_tid: TensorId = try sm.createTiledTensor(
        .f16,
        &[_]usize{ batch, seq_len, heads, head_dim },
        &[_]usize{ 1, t_l[0], 1, head_dim },
        .{ .tile_alignment = policy.tile_alignment },
    );
    const p_tid: TensorId = try sm.createTiledTensor(
        .i32,
        &[_]usize{ batch, seq_len },
        &[_]usize{ 1, t_l[0] },
        .{ .tile_alignment = policy.tile_alignment },
    );

    try sm.writeFromPackedScalar(x_tid, std.mem.sliceAsBytes(x));
    try sm.writeFromPackedScalar(p_tid, std.mem.sliceAsBytes(positions));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f16, &[_]usize{ batch, seq_len, heads, head_dim });
    const p_in = try g.addInput(.i32, &[_]usize{ batch, seq_len });
    try g.bindExternal(x_in, @intCast(x_tid));
    try g.bindExternal(p_in, @intCast(p_tid));

    const y = try g.addRoPE1D(x_in, p_in, 10000.0, 1.0, 1.0);
    try g.setOutputs(&[_]graph_mod.ValueId{y});

    var prog = try program_mod.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    const ns: u64 = try benchLoop(iters, struct {
        be: Backend,
        sm: *StorageManager,
        prog: *const program_mod.Program,
        fn run(self: @This()) !void {
            try self.be.executeProgram(self.prog, self.sm.tensorStore());
        }
    }{ .be = be, .sm = &sm, .prog = &prog });

    const bytes_per_iter: usize = (2 * x_elems * @sizeOf(f16)) + (p_elems * @sizeOf(i32));
    const bytes: u64 = @as(u64, @intCast(bytes_per_iter)) * @as(u64, @intCast(iters));
    const gib_s: f64 = fmtRateGiBPerSec(bytes, ns);

    const out_buf: []u8 = try allocator.alloc(u8, x_elems * @sizeOf(f16));
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const out_vals: []align(1) f16 = asF16Slice(out_buf);

    const idx0: usize = 0;
    const idx1: usize = (((batch / 2) * seq_len + (seq_len / 2)) * heads + (heads / 2)) * head_dim + (head_dim / 2);
    const idx2: usize = (((batch - 1) * seq_len + (seq_len - 1)) * heads + (heads - 1)) * head_dim + (head_dim - 1);
    const chk: f32 = @as(f32, @floatCast(out_vals[idx0])) + @as(f32, @floatCast(out_vals[idx1])) + @as(f32, @floatCast(out_vals[idx2]));

    std.debug.print(
        "program rope1d f16:      {d:.3} GiB/s  (b={} l={} h={} d={}, chk={d:.4}, wall={d:.3} ms, iter={d:.3} ms)\n",
        .{ gib_s, batch, seq_len, heads, head_dim, chk, nsToMilliseconds(ns), nsToMillisecondsPerIter(ns, iters) },
    );
}

fn benchProgramMatmulF16(allocator: std.mem.Allocator, rnd: std.Random, iters: usize, m: usize, n: usize, k: usize, be: Backend) !void {
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = defaultTilePolicy();
    const tiles = plan_mod.chooseMatMulTiles(policy, m, n, k, .f16);

    const a: []f16 = try allocator.alloc(f16, m * k);
    defer allocator.free(a);
    const b: []f16 = try allocator.alloc(f16, k * n);
    defer allocator.free(b);
    fillRandomF16(rnd, a);
    fillRandomF16(rnd, b);

    const a_tid: TensorId = try sm.createTiledTensor(.f16, &[_]usize{ m, k }, &[_]usize{ tiles.tm, tiles.tk }, .{ .tile_alignment = policy.tile_alignment });
    const b_tid: TensorId = try sm.createTiledTensor(.f16, &[_]usize{ k, n }, &[_]usize{ tiles.tk, tiles.tn }, .{ .tile_alignment = policy.tile_alignment });
    try sm.writeFromPackedScalar(a_tid, std.mem.sliceAsBytes(a));
    try sm.writeFromPackedScalar(b_tid, std.mem.sliceAsBytes(b));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const a_in = try g.addInput(.f16, &[_]usize{ m, k });
    const b_in = try g.addInput(.f16, &[_]usize{ k, n });
    try g.bindExternal(a_in, @intCast(a_tid));
    try g.bindExternal(b_in, @intCast(b_tid));
    const out = try g.addMatMul(a_in, b_in, 1.0, 0.0);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    var prog = try program_mod.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    const ns: u64 = try benchLoop(iters, struct {
        be: Backend,
        sm: *StorageManager,
        prog: *const program_mod.Program,
        fn run(self: @This()) !void {
            try self.be.executeProgram(self.prog, self.sm.tensorStore());
        }
    }{ .be = be, .sm = &sm, .prog = &prog });

    const flops_per: u64 = 2 * @as(u64, @intCast(m)) * @as(u64, @intCast(n)) * @as(u64, @intCast(k));
    const gflops: f64 = fmtRateGFLOPs(flops_per * @as(u64, @intCast(iters)), ns);

    const out_tid: TensorId = prog.outputs[0];
    const chk: f32 = @as(f32, @floatCast(try readF16AtTiled(&sm, out_tid, 0, 0))) +
        @as(f32, @floatCast(try readF16AtTiled(&sm, out_tid, m / 2, n / 2))) +
        @as(f32, @floatCast(try readF16AtTiled(&sm, out_tid, m - 1, n - 1)));
    std.debug.print("program matmul f16:       {d:.2} GFLOP/s (m={} n={} k={}, chk={d:.4}, wall={d:.3} ms, iter={d:.3} ms)\n", .{ gflops, m, n, k, chk, nsToMilliseconds(ns), nsToMillisecondsPerIter(ns, iters) });
}

fn benchProgramMatmulBatchedF16(allocator: std.mem.Allocator, rnd: std.Random, iters: usize, batch: usize, m: usize, n: usize, k: usize, be: Backend) !void {
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = defaultTilePolicy();
    const tiles = plan_mod.chooseMatMulTiles(policy, m, n, k, .f16);

    const a: []f16 = try allocator.alloc(f16, batch * m * k);
    defer allocator.free(a);
    const b: []f16 = try allocator.alloc(f16, batch * k * n);
    defer allocator.free(b);
    fillRandomF16(rnd, a);
    fillRandomF16(rnd, b);

    const a_tid: TensorId = try sm.createTiledTensor(.f16, &[_]usize{ batch, m, k }, &[_]usize{ 1, tiles.tm, tiles.tk }, .{ .tile_alignment = policy.tile_alignment });
    const b_tid: TensorId = try sm.createTiledTensor(.f16, &[_]usize{ batch, k, n }, &[_]usize{ 1, tiles.tk, tiles.tn }, .{ .tile_alignment = policy.tile_alignment });
    try sm.writeFromPackedScalar(a_tid, std.mem.sliceAsBytes(a));
    try sm.writeFromPackedScalar(b_tid, std.mem.sliceAsBytes(b));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const a_in = try g.addInput(.f16, &[_]usize{ batch, m, k });
    const b_in = try g.addInput(.f16, &[_]usize{ batch, k, n });
    try g.bindExternal(a_in, @intCast(a_tid));
    try g.bindExternal(b_in, @intCast(b_tid));
    const out = try g.addMatMul(a_in, b_in, 1.0, 0.0);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    var prog = try program_mod.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    const ns: u64 = try benchLoop(iters, struct {
        be: Backend,
        sm: *StorageManager,
        prog: *const program_mod.Program,
        fn run(self: @This()) !void {
            try self.be.executeProgram(self.prog, self.sm.tensorStore());
        }
    }{ .be = be, .sm = &sm, .prog = &prog });

    const flops_per: u64 = 2 * @as(u64, @intCast(batch)) * @as(u64, @intCast(m)) * @as(u64, @intCast(n)) * @as(u64, @intCast(k));
    const gflops: f64 = fmtRateGFLOPs(flops_per * @as(u64, @intCast(iters)), ns);

    const out_tid: TensorId = prog.outputs[0];
    const out_bytes_len: usize = batch * m * n * @sizeOf(f16);
    const out_buf: []u8 = try allocator.alloc(u8, out_bytes_len);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(out_tid, out_buf);
    const out_vals: []align(1) f16 = asF16Slice(out_buf);

    const idx0: usize = 0;
    const idx1: usize = (batch / 2) * m * n + (m / 2) * n + (n / 2);
    const idx2: usize = (batch - 1) * m * n + (m - 1) * n + (n - 1);
    const chk: f32 = @as(f32, @floatCast(out_vals[idx0])) + @as(f32, @floatCast(out_vals[idx1])) + @as(f32, @floatCast(out_vals[idx2]));

    std.debug.print("program matmul batched f16: {d:.2} GFLOP/s (b={} m={} n={} k={}, chk={d:.4}, wall={d:.3} ms, iter={d:.3} ms)\n", .{ gflops, batch, m, n, k, chk, nsToMilliseconds(ns), nsToMillisecondsPerIter(ns, iters) });
}

fn benchProgramMatmulF32(allocator: std.mem.Allocator, rnd: std.Random, iters: usize, m: usize, n: usize, k: usize, be: Backend) !void {
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = defaultTilePolicy();
    const tiles = plan_mod.chooseMatMulTiles(policy, m, n, k, .f32);

    const a: []f32 = try allocator.alloc(f32, m * k);
    defer allocator.free(a);
    const b: []f32 = try allocator.alloc(f32, k * n);
    defer allocator.free(b);
    fillRandomF32(rnd, a);
    fillRandomF32(rnd, b);

    const a_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{ m, k }, &[_]usize{ tiles.tm, tiles.tk }, .{ .tile_alignment = policy.tile_alignment });
    const b_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{ k, n }, &[_]usize{ tiles.tk, tiles.tn }, .{ .tile_alignment = policy.tile_alignment });
    try sm.writeFromPackedScalar(a_tid, std.mem.sliceAsBytes(a));
    try sm.writeFromPackedScalar(b_tid, std.mem.sliceAsBytes(b));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const a_in = try g.addInput(.f32, &[_]usize{ m, k });
    const b_in = try g.addInput(.f32, &[_]usize{ k, n });
    try g.bindExternal(a_in, @intCast(a_tid));
    try g.bindExternal(b_in, @intCast(b_tid));
    const out = try g.addMatMul(a_in, b_in, 1.0, 0.0);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    var prog = try program_mod.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    const ns: u64 = try benchLoop(iters, struct {
        be: Backend,
        sm: *StorageManager,
        prog: *const program_mod.Program,
        fn run(self: @This()) !void {
            try self.be.executeProgram(self.prog, self.sm.tensorStore());
        }
    }{ .be = be, .sm = &sm, .prog = &prog });

    const flops_per: u64 = 2 * @as(u64, @intCast(m)) * @as(u64, @intCast(n)) * @as(u64, @intCast(k));
    const gflops: f64 = fmtRateGFLOPs(flops_per * @as(u64, @intCast(iters)), ns);

    const out_tid: TensorId = prog.outputs[0];
    const chk: f32 = try readF32AtTiled(&sm, out_tid, 0, 0) +
        try readF32AtTiled(&sm, out_tid, m / 2, n / 2) +
        try readF32AtTiled(&sm, out_tid, m - 1, n - 1);
    std.debug.print("program matmul f32:       {d:.2} GFLOP/s (m={} n={} k={}, chk={d:.4}, wall={d:.3} ms, iter={d:.3} ms)\n", .{ gflops, m, n, k, chk, nsToMilliseconds(ns), nsToMillisecondsPerIter(ns, iters) });
}

fn benchProgramMatmulBatchedF32(allocator: std.mem.Allocator, rnd: std.Random, iters: usize, batch: usize, m: usize, n: usize, k: usize, be: Backend) !void {
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = defaultTilePolicy();
    const tiles = plan_mod.chooseMatMulTiles(policy, m, n, k, .f32);

    const a: []f32 = try allocator.alloc(f32, batch * m * k);
    defer allocator.free(a);
    const b: []f32 = try allocator.alloc(f32, batch * k * n);
    defer allocator.free(b);
    fillRandomF32(rnd, a);
    fillRandomF32(rnd, b);

    const a_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{ batch, m, k }, &[_]usize{ 1, tiles.tm, tiles.tk }, .{ .tile_alignment = policy.tile_alignment });
    const b_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{ batch, k, n }, &[_]usize{ 1, tiles.tk, tiles.tn }, .{ .tile_alignment = policy.tile_alignment });
    try sm.writeFromPackedScalar(a_tid, std.mem.sliceAsBytes(a));
    try sm.writeFromPackedScalar(b_tid, std.mem.sliceAsBytes(b));

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const a_in = try g.addInput(.f32, &[_]usize{ batch, m, k });
    const b_in = try g.addInput(.f32, &[_]usize{ batch, k, n });
    try g.bindExternal(a_in, @intCast(a_tid));
    try g.bindExternal(b_in, @intCast(b_tid));
    const out = try g.addMatMul(a_in, b_in, 1.0, 0.0);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    var prog = try program_mod.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    const ns: u64 = try benchLoop(iters, struct {
        be: Backend,
        sm: *StorageManager,
        prog: *const program_mod.Program,
        fn run(self: @This()) !void {
            try self.be.executeProgram(self.prog, self.sm.tensorStore());
        }
    }{ .be = be, .sm = &sm, .prog = &prog });

    const flops_per: u64 = 2 * @as(u64, @intCast(batch)) * @as(u64, @intCast(m)) * @as(u64, @intCast(n)) * @as(u64, @intCast(k));
    const gflops: f64 = fmtRateGFLOPs(flops_per * @as(u64, @intCast(iters)), ns);

    const out_tid: TensorId = prog.outputs[0];
    const out_bytes_len: usize = batch * m * n * @sizeOf(f32);
    const out_buf: []u8 = try allocator.alloc(u8, out_bytes_len);
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(out_tid, out_buf);
    const out_vals: []align(1) f32 = asF32Slice(out_buf);

    const idx0: usize = 0;
    const idx1: usize = (batch / 2) * m * n + (m / 2) * n + (n / 2);
    const idx2: usize = (batch - 1) * m * n + (m - 1) * n + (n - 1);
    const chk: f32 = out_vals[idx0] + out_vals[idx1] + out_vals[idx2];

    std.debug.print("program matmul batched f32: {d:.2} GFLOP/s (b={} m={} n={} k={}, chk={d:.4}, wall={d:.3} ms, iter={d:.3} ms)\n", .{ gflops, batch, m, n, k, chk, nsToMilliseconds(ns), nsToMillisecondsPerIter(ns, iters) });
}

fn benchProgramMatmulQuant(allocator: std.mem.Allocator, rnd: std.Random, iters: usize, m: usize, n: usize, k: usize, b_dtype: types.DType, be: Backend) !void {
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = defaultTilePolicy();
    const quant_m_hint: usize = @max(@as(usize, 1), policy.base_square_2d);
    const tiles = plan_mod.chooseMatMulTiles(policy, quant_m_hint, n, k, b_dtype);

    const a: []f32 = try allocator.alloc(f32, m * k);
    defer allocator.free(a);
    fillRandomF32(rnd, a);

    const a_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{ m, k }, &[_]usize{ tiles.tm, tiles.tk }, .{ .tile_alignment = policy.tile_alignment });
    try sm.writeFromPackedScalar(a_tid, std.mem.sliceAsBytes(a));

    const b_bytes: []u8 = switch (b_dtype) {
        .q8_0 => try buildQuantB_Q8_0(allocator, rnd, k, n),
        .q4_0 => try buildQuantB_Q4_0(allocator, rnd, k, n),
        else => return error.InvalidArgument,
    };
    defer allocator.free(b_bytes);

    const b_tid: TensorId = try sm.createTiledTensor(b_dtype, &[_]usize{ k, n }, &[_]usize{ tiles.tk, tiles.tn }, .{ .tile_alignment = policy.tile_alignment });
    try sm.writeFromPackedQuant(b_tid, b_bytes);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const a_in = try g.addInput(.f32, &[_]usize{ m, k });
    const b_in = try g.addInput(b_dtype, &[_]usize{ k, n });
    try g.bindExternal(a_in, @intCast(a_tid));
    try g.bindExternal(b_in, @intCast(b_tid));
    const out = try g.addMatMul(a_in, b_in, 1.0, 0.0);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    var prog = try program_mod.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    const ns: u64 = try benchLoop(iters, struct {
        be: Backend,
        sm: *StorageManager,
        prog: *const program_mod.Program,
        fn run(self: @This()) !void {
            try self.be.executeProgram(self.prog, self.sm.tensorStore());
        }
    }{ .be = be, .sm = &sm, .prog = &prog });

    const flops_per: u64 = 2 * @as(u64, @intCast(m)) * @as(u64, @intCast(n)) * @as(u64, @intCast(k));
    const gflops: f64 = fmtRateGFLOPs(flops_per * @as(u64, @intCast(iters)), ns);

    const out_tid: TensorId = prog.outputs[0];
    const chk: f32 = try readF32AtTiled(&sm, out_tid, 0, 0) +
        try readF32AtTiled(&sm, out_tid, m / 2, n / 2) +
        try readF32AtTiled(&sm, out_tid, m - 1, n - 1);

    const name: []const u8 = if (b_dtype == .q8_0) "q8_0" else "q4_0";
    std.debug.print("program matmul {s}:      {d:.2} GFLOP/s (m={} n={} k={}, chk={d:.4}, wall={d:.3} ms, iter={d:.3} ms)\n", .{ name, gflops, m, n, k, chk, nsToMilliseconds(ns), nsToMillisecondsPerIter(ns, iters) });
}

// ---------------------------------------------------------------------------
// Decode (autoregressive GEMV) suite.
//
// Reproduces the per-token decode shapes of a transformer (M=1, q8_0 weights)
// through the real graph/program/executor path, isolated from model load. Compares
// the two weight layouts the runtime can use:
//   * MatMul   : B stored `[1, K, N]` (block-major over K) — the projection path.
//   * MatMulNT : B stored `[N, K]`    (per-output-row contiguous K) — the tied-logits
//                path. `tn` is B's N-axis tile = how finely N parallelizes.
// Reported as GiB/s of q8_0 weight bytes streamed (the decode-relevant metric).
// ---------------------------------------------------------------------------

fn buildQuantB_Q8_0_NT(allocator: std.mem.Allocator, rnd: std.Random, n: usize, k: usize) ![]u8 {
    // NT B is `[N, K]` (quant_axis=1): per row j, K/32 contiguous blocks.
    // offset(j, kb) = (j*k_blocks + kb) * block_bytes.
    if (k % Q8_0_BLOCK_ELEMS != 0) return error.InvalidArgument;
    const k_blocks: usize = k / Q8_0_BLOCK_ELEMS;

    const b_f32: []f32 = try allocator.alloc(f32, n * k);
    defer allocator.free(b_f32);
    fillRandomF32(rnd, b_f32);

    const out: []u8 = try allocator.alloc(u8, n * k_blocks * Q8_0_BLOCK_BYTES);
    for (0..n) |j| {
        for (0..k_blocks) |kb| {
            var block_vals: [32]f32 = undefined;
            for (0..32) |t| block_vals[t] = b_f32[j * k + kb * 32 + t];
            const off: usize = (j * k_blocks + kb) * Q8_0_BLOCK_BYTES;
            quantizeQ8_0FromF32Block32(&block_vals, @ptrCast(out[off..][0..Q8_0_BLOCK_BYTES]));
        }
    }
    return out;
}

fn decodeWeightGiBs(k: usize, n: usize, iters: usize, ns: u64) f64 {
    const bytes: u64 = @as(u64, @intCast((k / Q8_0_BLOCK_ELEMS) * n * Q8_0_BLOCK_BYTES)) * @as(u64, @intCast(iters));
    return fmtRateGiBPerSec(bytes, ns);
}

fn benchDecodeMatMul(allocator: std.mem.Allocator, rnd: std.Random, iters: usize, k: usize, n: usize, label: []const u8, be: Backend) !void {
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    // Production-faithful policy (the model loads with `TilePolicy{}` defaults, i.e.
    // base_square_2d=64) so this reproduces the model's actual decode tiling — NOT
    // the bench-wide defaultTilePolicy() which bumps base_square_2d to 256.
    const policy: plan_mod.TilePolicy = .{};
    const m_hint: usize = @max(@as(usize, 1), policy.base_square_2d);
    const tiles = plan_mod.chooseMatMulTiles(policy, m_hint, n, k, .q8_0);

    const a: []f32 = try allocator.alloc(f32, k);
    defer allocator.free(a);
    fillRandomF32(rnd, a);

    const a_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{ 1, 1, k }, &[_]usize{ 1, tiles.tm, tiles.tk }, .{ .tile_alignment = policy.tile_alignment });
    try sm.writeFromPackedScalar(a_tid, std.mem.sliceAsBytes(a));

    const b_bytes: []u8 = try buildQuantB_Q8_0(allocator, rnd, k, n);
    defer allocator.free(b_bytes);
    const b_tid: TensorId = try sm.createTiledTensor(.q8_0, &[_]usize{ 1, k, n }, &[_]usize{ 1, tiles.tk, tiles.tn }, .{ .tile_alignment = policy.tile_alignment, .quant_axis = 1 });
    try sm.writeFromPackedQuant(b_tid, b_bytes);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();
    const a_in = try g.addInput(.f32, &[_]usize{ 1, 1, k });
    const b_in = try g.addInput(.q8_0, &[_]usize{ 1, k, n });
    try g.bindExternal(a_in, @intCast(a_tid));
    try g.bindExternal(b_in, @intCast(b_tid));
    const out = try g.addMatMul(a_in, b_in, 1.0, 0.0);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    var prog = try program_mod.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    const ns: u64 = try benchLoop(iters, struct {
        be: Backend,
        sm: *StorageManager,
        prog: *const program_mod.Program,
        fn run(self: @This()) !void {
            try self.be.executeProgram(self.prog, self.sm.tensorStore());
        }
    }{ .be = be, .sm = &sm, .prog = &prog });

    const out_buf: []u8 = try allocator.alloc(u8, n * @sizeOf(f32));
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const ov = asF32Slice(out_buf);
    const chk: f32 = ov[0] + ov[n / 2] + ov[n - 1];

    std.debug.print(
        "decode {s:<16} MatMul       K={d:<5} N={d:<6} tn={d:<4} tk={d:<4}: {d:6.2} GiB/s  ({d:.3} ms/iter, chk={d:.3})\n",
        .{ label, k, n, tiles.tn, tiles.tk, decodeWeightGiBs(k, n, iters, ns), nsToMillisecondsPerIter(ns, iters), chk },
    );
}

fn benchDecodeMatMulNT(allocator: std.mem.Allocator, rnd: std.Random, iters: usize, k: usize, n: usize, tn: usize, label: []const u8, be: Backend) !void {
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = defaultTilePolicy();
    const tn_eff: usize = @max(@as(usize, 1), @min(tn, n));

    const a: []f32 = try allocator.alloc(f32, k);
    defer allocator.free(a);
    fillRandomF32(rnd, a);

    const a_tid: TensorId = try sm.createTiledTensor(.f32, &[_]usize{ 1, 1, k }, &[_]usize{ 1, 1, k }, .{ .tile_alignment = policy.tile_alignment });
    try sm.writeFromPackedScalar(a_tid, std.mem.sliceAsBytes(a));

    const b_bytes: []u8 = try buildQuantB_Q8_0_NT(allocator, rnd, n, k);
    defer allocator.free(b_bytes);
    // NT B `[N, K]`, quant_axis=1, tile_shape[1] == K (full-K per row).
    const b_tid: TensorId = try sm.createTiledTensor(.q8_0, &[_]usize{ n, k }, &[_]usize{ tn_eff, k }, .{ .tile_alignment = policy.tile_alignment, .quant_axis = 1 });
    try sm.writeFromPackedQuant(b_tid, b_bytes);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();
    const a_in = try g.addInput(.f32, &[_]usize{ 1, 1, k });
    const b_in = try g.addInput(.q8_0, &[_]usize{ n, k });
    try g.bindExternal(a_in, @intCast(a_tid));
    try g.bindExternal(b_in, @intCast(b_tid));
    const out = try g.addMatMulNT(a_in, b_in, 1.0, 0.0);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    var prog = try program_mod.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    const ns: u64 = try benchLoop(iters, struct {
        be: Backend,
        sm: *StorageManager,
        prog: *const program_mod.Program,
        fn run(self: @This()) !void {
            try self.be.executeProgram(self.prog, self.sm.tensorStore());
        }
    }{ .be = be, .sm = &sm, .prog = &prog });

    const out_buf: []u8 = try allocator.alloc(u8, n * @sizeOf(f32));
    defer allocator.free(out_buf);
    try sm.readToPackedScalar(prog.outputs[0], out_buf);
    const ov = asF32Slice(out_buf);
    const chk: f32 = ov[0] + ov[n / 2] + ov[n - 1];

    std.debug.print(
        "decode {s:<16} MatMulNT     K={d:<5} N={d:<6} tn={d:<4}        : {d:6.2} GiB/s  ({d:.3} ms/iter, chk={d:.3})\n",
        .{ label, k, n, tn_eff, decodeWeightGiBs(k, n, iters, ns), nsToMillisecondsPerIter(ns, iters), chk },
    );
}

fn runDecodeSuite(allocator: std.mem.Allocator, rnd: std.Random, opts: BenchOptions, be: Backend) !void {
    const Shape = struct { label: []const u8, k: usize, n: usize };
    // Gemma 4 E2B per-token decode GEMV shapes.
    const shapes = [_]Shape{
        .{ .label = "ffn_gate_up", .k = 1536, .n = 6144 },
        .{ .label = "ffn_gateup_FUSED", .k = 1536, .n = 12288 }, // gate+up concatenated (1 op vs 2)
        .{ .label = "ffn_down", .k = 6144, .n = 1536 },
        .{ .label = "q_proj_local", .k = 1536, .n = 2048 },
        .{ .label = "o_proj_local", .k = 2048, .n = 1536 },
        .{ .label = "q_proj_global", .k = 1536, .n = 4096 },
        .{ .label = "qkv_FUSED_local", .k = 1536, .n = 2560 }, // q2048+k256+v256 (1 op vs 3)
        .{ .label = "qkv_FUSED_global", .k = 1536, .n = 5120 }, // q4096+k512+v512
    };
    const tn_sweep = [_]usize{ 64, 256, 1024 };

    for (shapes) |s| {
        try benchDecodeMatMul(allocator, rnd, opts.iters, s.k, s.n, s.label, be);
        for (tn_sweep) |tn| try benchDecodeMatMulNT(allocator, rnd, opts.iters, s.k, s.n, tn, s.label, be);
        std.debug.print("\n", .{});
    }
    // Tied-logits projection: NT only (the real model path), huge N.
    for (tn_sweep) |tn| try benchDecodeMatMulNT(allocator, rnd, opts.iters, 1536, 262144, tn, "logits_tied", be);
}

fn makeCpuBackend(allocator: std.mem.Allocator, threads: usize) !CpuBackend {
    if (threads == 1) {
        return CpuBackend.init(allocator);
    }
    return try CpuBackend.initWithOptions(allocator, .{ .thread_count = threads });
}

fn nsToSeconds(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / @as(f64, 1_000_000_000.0);
}

fn nsToMilliseconds(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / @as(f64, 1_000_000.0);
}

fn nsToMillisecondsPerIter(ns: u64, iters: usize) f64 {
    const denom: f64 = @as(f64, @floatFromInt(iters));
    if (denom == 0.0) return 0.0;
    return nsToMilliseconds(ns) / denom;
}

fn fmtRateGiBPerSec(bytes: u64, ns: u64) f64 {
    const sec: f64 = nsToSeconds(ns);
    if (sec == 0.0) return 0.0;
    const gib: f64 = @as(f64, @floatFromInt(bytes)) / @as(f64, 1024.0 * 1024.0 * 1024.0);
    return gib / sec;
}

fn fmtRateGFLOPs(flops: u64, ns: u64) f64 {
    const sec: f64 = nsToSeconds(ns);
    if (sec == 0.0) return 0.0;
    const gflop: f64 = @as(f64, @floatFromInt(flops)) / 1_000_000_000.0;
    return gflop / sec;
}

fn nowNs() u64 {
    const ts: std.Io.Timestamp = std.Io.Clock.awake.now(std.Options.debug_io);
    const ns: i96 = ts.toNanoseconds();
    if (ns <= 0) return 0;
    const max_u64_i96: i96 = @as(i96, std.math.maxInt(u64));
    return @intCast(@min(ns, max_u64_i96));
}

fn benchLoop(iters: usize, work_in: anytype) !u64 {
    // Work is passed as a tiny context object with a `run()` method.
    // Using a context avoids heap allocations and gives us capture-like behavior.
    var work = work_in;

    // Warm-up (helps with cache and first-touch page faults).
    try work.run();

    const start: u64 = nowNs();
    var i: usize = 0;
    while (i < iters) : (i += 1) {
        try work.run();
    }
    const end: u64 = nowNs();
    return end - start;
}

fn fillRandomF32(rnd: std.Random, data: []f32) void {
    for (data) |*x| {
        // Keep values in a tame range to avoid NaNs/denorm weirdness.
        x.* = (rnd.float(f32) - 0.5) * 2.0;
    }
}

fn fillRandomF16(rnd: std.Random, data: []f16) void {
    for (data) |*x| {
        const v: f32 = (rnd.float(f32) - 0.5) * 2.0;
        x.* = @floatCast(v);
    }
}

fn quantizeQ8_0FromF32Block32(vals: *const [32]f32, out: *[Q8_0_BLOCK_BYTES]u8) void {
    var max_abs: f32 = 0.0;
    for (vals.*) |v| {
        const a: f32 = @abs(v);
        if (a > max_abs) max_abs = a;
    }

    // ggml-ish: scale is per-block.
    const scale: f32 = if (max_abs == 0.0) 1.0 else (max_abs / 127.0);
    const scale_f16: f16 = @floatCast(scale);

    const scale_bits: [2]u8 = @bitCast(scale_f16);
    out[0] = scale_bits[0];
    out[1] = scale_bits[1];

    var i: usize = 0;
    while (i < 32) : (i += 1) {
        const qf: f32 = vals[i] / scale;
        const qi32: i32 = @intFromFloat(std.math.round(qf));
        const qi8: i8 = @intCast(std.math.clamp(qi32, -128, 127));
        out[2 + i] = @bitCast(qi8);
    }
}

fn quantizeQ4_0FromF32Block32(vals: *const [32]f32, out: *[Q4_0_BLOCK_BYTES]u8) void {
    var max_abs: f32 = 0.0;
    for (vals.*) |v| {
        const a: f32 = @abs(v);
        if (a > max_abs) max_abs = a;
    }

    const scale: f32 = if (max_abs == 0.0) 1.0 else (max_abs / 7.0);
    const scale_f16: f16 = @floatCast(scale);

    const scale_bits: [2]u8 = @bitCast(scale_f16);
    out[0] = scale_bits[0];
    out[1] = scale_bits[1];

    // Encode 32 values into 16 bytes of nibbles. Values are in [-8, 7].
    var bi: usize = 0;
    while (bi < 16) : (bi += 1) {
        const lo_i: usize = bi * 2;
        const hi_i: usize = bi * 2 + 1;

        const lo_q: i32 = @intFromFloat(std.math.round(vals[lo_i] / scale));
        const hi_q: i32 = @intFromFloat(std.math.round(vals[hi_i] / scale));

        const lo_i8: i8 = @intCast(std.math.clamp(lo_q, -8, 7));
        const hi_i8: i8 = @intCast(std.math.clamp(hi_q, -8, 7));

        const lo_u: u8 = @intCast(@as(i16, lo_i8) + 8);
        const hi_u: u8 = @intCast(@as(i16, hi_i8) + 8);

        out[2 + bi] = (lo_u & 0x0F) | ((hi_u & 0x0F) << 4);
    }
}

fn buildQuantB_Q8_0(allocator: std.mem.Allocator, rnd: std.Random, k: usize, n: usize) ![]u8 {
    // Storage layout expected by quant matmul:
    // blocks indexed by (kb, j) => offset = (kb*n + j)*block_bytes
    if (k % Q8_0_BLOCK_ELEMS != 0) return error.InvalidArgument;
    const k_blocks: usize = k / Q8_0_BLOCK_ELEMS;

    // Generate a reference f32 B[K,N] first, then quantize into block layout.
    const b_f32: []f32 = try allocator.alloc(f32, k * n);
    defer allocator.free(b_f32);
    fillRandomF32(rnd, b_f32);

    const total_bytes: usize = k_blocks * n * Q8_0_BLOCK_BYTES;
    var out: []u8 = try allocator.alloc(u8, total_bytes);

    for (0..k_blocks) |kb| {
        for (0..n) |j| {
            var block_vals: [32]f32 = undefined;
            for (0..32) |t| {
                const kk: usize = kb * 32 + t;
                block_vals[t] = b_f32[kk * n + j];
            }

            const off: usize = (kb * n + j) * Q8_0_BLOCK_BYTES;
            const dst: *[Q8_0_BLOCK_BYTES]u8 = @ptrCast(out[off..][0..Q8_0_BLOCK_BYTES]);
            quantizeQ8_0FromF32Block32(&block_vals, dst);
        }
    }

    return out;
}

fn buildQuantB_Q4_0(allocator: std.mem.Allocator, rnd: std.Random, k: usize, n: usize) ![]u8 {
    if (k % Q4_0_BLOCK_ELEMS != 0) return error.InvalidArgument;
    const k_blocks: usize = k / Q4_0_BLOCK_ELEMS;

    const b_f32: []f32 = try allocator.alloc(f32, k * n);
    defer allocator.free(b_f32);
    fillRandomF32(rnd, b_f32);

    const total_bytes: usize = k_blocks * n * Q4_0_BLOCK_BYTES;
    var out: []u8 = try allocator.alloc(u8, total_bytes);

    for (0..k_blocks) |kb| {
        for (0..n) |j| {
            var block_vals: [32]f32 = undefined;
            for (0..32) |t| {
                const kk: usize = kb * 32 + t;
                block_vals[t] = b_f32[kk * n + j];
            }

            const off: usize = (kb * n + j) * Q4_0_BLOCK_BYTES;
            const dst: *[Q4_0_BLOCK_BYTES]u8 = @ptrCast(out[off..][0..Q4_0_BLOCK_BYTES]);
            quantizeQ4_0FromF32Block32(&block_vals, dst);
        }
    }

    return out;
}

fn runF32Benches(allocator: std.mem.Allocator, rnd: std.Random, opts: BenchOptions, be: Backend) !void {
    if (opts.suite == .decode) {
        try runDecodeSuite(allocator, rnd, opts, be);
        return;
    }

    if (opts.suite == .matmul) {
        try benchProgramMatmulF32(allocator, rnd, opts.iters, opts.m, opts.n, opts.k, be);
        try benchProgramMatmulBatchedF32(allocator, rnd, opts.iters, opts.batch, opts.m, opts.n, opts.k, be);
        if (opts.quant) {
            if (opts.k % 32 != 0) {
                std.debug.print("(skipping quant matmul: k={} not divisible by 32)\n", .{opts.k});
            } else {
                try benchProgramMatmulQuant(allocator, rnd, opts.iters, opts.m, opts.n, opts.k, .q8_0, be);
                try benchProgramMatmulQuant(allocator, rnd, opts.iters, opts.m, opts.n, opts.k, .q4_0, be);
            }
        }
        return;
    }

    if (opts.suite == .matmul_q) {
        if (!opts.quant) {
            std.debug.print("(skipping quant matmul suite: --no-quant specified)\n", .{});
            return;
        }
        if (opts.k % 32 != 0) {
            std.debug.print("(skipping quant matmul: k={} not divisible by 32)\n", .{opts.k});
            return;
        }
        try benchProgramMatmulQuant(allocator, rnd, opts.iters, opts.m, opts.n, opts.k, .q8_0, be);
        try benchProgramMatmulQuant(allocator, rnd, opts.iters, opts.m, opts.n, opts.k, .q4_0, be);
        return;
    }

    try benchProgramElemwiseAdd(allocator, rnd, opts.iters, opts.n_elem, be);
    try benchProgramRelu(allocator, rnd, opts.iters, opts.n_elem, be);
    try benchProgramUnary(allocator, rnd, opts.iters, opts.n_elem, .gelu, "gelu", be);
    try benchProgramUnary(allocator, rnd, opts.iters, opts.n_elem, .silu, "silu", be);
    try benchProgramUnary(allocator, rnd, opts.iters, opts.n_elem, .sigmoid, "sigmoid", be);
    try benchProgramUnary(allocator, rnd, opts.iters, opts.n_elem, .tanh, "tanh", be);
    try benchProgramSoftmaxF32(allocator, rnd, opts.iters, opts.m, opts.n, be);
    try benchProgramLayerNormF32(allocator, rnd, opts.iters, opts.m, opts.n, be);
    try benchProgramRMSNormF32(allocator, rnd, opts.iters, opts.m, opts.n, be);
    try benchProgramAttentionF32(allocator, rnd, opts.iters, opts.m, opts.n, opts.k, opts.n, false, be);
    try benchProgramAttentionF32(allocator, rnd, opts.iters, opts.m, opts.n, opts.k, opts.n, true, be);
    try benchProgramMultiHeadAttentionSeparateF32(allocator, rnd, opts.iters, opts.batch, opts.heads, opts.m, opts.n, opts.k, opts.n, false, be);
    try benchProgramMultiHeadAttentionSeparateF32(allocator, rnd, opts.iters, opts.batch, opts.heads, opts.m, opts.n, opts.k, opts.n, true, be);
    const cached_h_kv: usize = if ((opts.heads % 2) == 0) @max(@as(usize, 1), opts.heads / 2) else 1;
    try benchProgramMultiHeadAttentionCachedF32(allocator, rnd, opts.iters, opts.batch, opts.heads, cached_h_kv, opts.m, opts.n, opts.k, opts.n, true, 0, 0.0, "global", be);
    const cached_sw: usize = @max(@as(usize, 1), opts.n / 2);
    if (cached_sw > 0 and cached_sw < opts.n) {
        try benchProgramMultiHeadAttentionCachedF32(allocator, rnd, opts.iters, opts.batch, opts.heads, cached_h_kv, opts.m, opts.n, opts.k, opts.n, true, cached_sw, 0.0, "window", be);
    }
    if ((opts.k % 2) == 0) {
        try benchProgramRoPE1DF32(allocator, rnd, opts.iters, opts.batch, opts.m, opts.heads, opts.k, be);
    } else {
        std.debug.print("(skipping rope1d f32: k/head_dim={} must be even)\n", .{opts.k});
    }

    // STFT front-end (ASR-style: 16 kHz, 32 ms window, 10 ms hop).
    try benchProgramSTFTF32(allocator, rnd, opts.iters, 1, 16000, 512, 160, be);

    try benchProgramConv1D(allocator, rnd, opts.iters, opts.conv_batch, opts.conv_l, opts.conv_cin, opts.conv_cout, opts.conv_k, 1, .zero, "regular", be);
    try benchProgramConv1D(allocator, rnd, opts.iters, opts.conv_batch, opts.conv_l, opts.conv_cin, opts.conv_cout, 1, 1, .zero, "pointwise", be);
    if (opts.conv_cout == opts.conv_cin) {
        try benchProgramConv1D(allocator, rnd, opts.iters, opts.conv_batch, opts.conv_l, opts.conv_cin, opts.conv_cout, opts.conv_k, opts.conv_cin, .zero, "depthwise", be);
    } else {
        std.debug.print("(skipping conv1d depthwise: conv-cout={} must equal conv-cin={})\n", .{ opts.conv_cout, opts.conv_cin });
    }

    if (opts.reflect_conv) {
        const conv1d_pad: usize = opts.conv_k / 2;
        if (opts.conv_l > 1 and conv1d_pad < opts.conv_l) {
            benchProgramConv1D(allocator, rnd, opts.iters, opts.conv_batch, opts.conv_l, opts.conv_cin, opts.conv_cout, opts.conv_k, 1, .reflect, "regular-reflect", be) catch |e| {
                std.debug.print("(conv1d reflect bench failed: {s})\n", .{@errorName(e)});
            };
        } else {
            std.debug.print("(skipping conv1d reflect: requires conv-l>1 and conv-k/2 < conv-l; got conv-l={} conv-k={})\n", .{ opts.conv_l, opts.conv_k });
        }
    }

    try benchProgramConv2D(allocator, rnd, opts.iters, opts.conv_batch, opts.conv_h, opts.conv_w, opts.conv_cin, opts.conv_cout, opts.conv_k, 1, .zero, "regular", be);
    try benchProgramConv2D(allocator, rnd, opts.iters, opts.conv_batch, opts.conv_h, opts.conv_w, opts.conv_cin, opts.conv_cout, 1, 1, .zero, "pointwise", be);
    if (opts.conv_cout == opts.conv_cin) {
        try benchProgramConv2D(allocator, rnd, opts.iters, opts.conv_batch, opts.conv_h, opts.conv_w, opts.conv_cin, opts.conv_cout, opts.conv_k, opts.conv_cin, .zero, "depthwise", be);
    } else {
        std.debug.print("(skipping conv2d depthwise: conv-cout={} must equal conv-cin={})\n", .{ opts.conv_cout, opts.conv_cin });
    }

    if (opts.reflect_conv) {
        const conv2d_pad: usize = opts.conv_k / 2;
        if (opts.conv_h > 1 and opts.conv_w > 1 and conv2d_pad < opts.conv_h and conv2d_pad < opts.conv_w) {
            benchProgramConv2D(allocator, rnd, opts.iters, opts.conv_batch, opts.conv_h, opts.conv_w, opts.conv_cin, opts.conv_cout, opts.conv_k, 1, .reflect, "regular-reflect", be) catch |e| {
                std.debug.print("(conv2d reflect bench failed: {s})\n", .{@errorName(e)});
            };
        } else {
            std.debug.print("(skipping conv2d reflect: requires conv-h>1, conv-w>1, conv-k/2<conv-h, conv-k/2<conv-w; got h={} w={} k={})\n", .{ opts.conv_h, opts.conv_w, opts.conv_k });
        }
    }

    try benchProgramReduceSum(allocator, rnd, opts.iters, opts.n_elem, be);
    try benchProgramReduceAxisF32(allocator, rnd, opts.iters, opts.m, opts.n, -1, "last", be);
    try benchProgramReduceAxisF32(allocator, rnd, opts.iters, opts.m, opts.n, 0, "axis0", be);
    try benchProgramMatmulF32(allocator, rnd, opts.iters, opts.m, opts.n, opts.k, be);
    try benchProgramMatmulBatchedF32(allocator, rnd, opts.iters, opts.batch, opts.m, opts.n, opts.k, be);
    if (opts.quant) {
        if (opts.k % 32 != 0) {
            std.debug.print("(skipping quant matmul: k={} not divisible by 32)\n", .{opts.k});
        } else {
            try benchProgramMatmulQuant(allocator, rnd, opts.iters, opts.m, opts.n, opts.k, .q8_0, be);
            try benchProgramMatmulQuant(allocator, rnd, opts.iters, opts.m, opts.n, opts.k, .q4_0, be);
        }
    }
}

fn runF16Benches(allocator: std.mem.Allocator, rnd: std.Random, opts: BenchOptions, be: Backend) !void {
    if (opts.suite == .matmul) {
        try benchProgramMatmulF16(allocator, rnd, opts.iters, opts.m, opts.n, opts.k, be);
        try benchProgramMatmulBatchedF16(allocator, rnd, opts.iters, opts.batch, opts.m, opts.n, opts.k, be);
        std.debug.print("(skipping quant matmul in f16 mode: quant benches currently use f32 activations)\n", .{});
        return;
    }
    if (opts.suite == .matmul_q) {
        std.debug.print("(skipping quant matmul suite in f16 mode: quant benches currently use f32 activations)\n", .{});
        return;
    }

    try benchProgramElemwiseAddF16(allocator, rnd, opts.iters, opts.n_elem, be);
    try benchProgramUnaryF16(allocator, rnd, opts.iters, opts.n_elem, .relu, "relu", be);
    try benchProgramUnaryF16(allocator, rnd, opts.iters, opts.n_elem, .gelu, "gelu", be);
    try benchProgramUnaryF16(allocator, rnd, opts.iters, opts.n_elem, .silu, "silu", be);
    try benchProgramUnaryF16(allocator, rnd, opts.iters, opts.n_elem, .sigmoid, "sigmoid", be);
    try benchProgramUnaryF16(allocator, rnd, opts.iters, opts.n_elem, .tanh, "tanh", be);

    std.debug.print("(skipping softmax f16: softmax kernel is currently f32-only)\n", .{});
    try benchProgramLayerNormF16(allocator, rnd, opts.iters, opts.m, opts.n, be);
    try benchProgramRMSNormF16(allocator, rnd, opts.iters, opts.m, opts.n, be);
    std.debug.print("(skipping attention/mha f16: attention kernels are currently f32-only)\n", .{});

    if ((opts.k % 2) == 0) {
        try benchProgramRoPE1DF16(allocator, rnd, opts.iters, opts.batch, opts.m, opts.heads, opts.k, be);
    } else {
        std.debug.print("(skipping rope1d f16: k/head_dim={} must be even)\n", .{opts.k});
    }

    std.debug.print("(skipping conv f16 benches: conv kernels are currently f32-only)\n", .{});

    try benchProgramReduceSumF16(allocator, rnd, opts.iters, opts.n_elem, be);
    try benchProgramReduceAxisF16(allocator, rnd, opts.iters, opts.m, opts.n, -1, "last", be);
    try benchProgramReduceAxisF16(allocator, rnd, opts.iters, opts.m, opts.n, 0, "axis0", be);
    try benchProgramMatmulF16(allocator, rnd, opts.iters, opts.m, opts.n, opts.k, be);
    try benchProgramMatmulBatchedF16(allocator, rnd, opts.iters, opts.batch, opts.m, opts.n, opts.k, be);
    std.debug.print("(skipping quant matmul in f16 mode: quant benches currently use f32 activations)\n", .{});
}

pub fn main(minimal: std.process.Init.Minimal) void {
    mainImpl(minimal.args) catch |e| {
        std.debug.print("fatal: {s}\n", .{@errorName(e)});
        std.process.exit(1);
    };
}

fn mainImpl(args: std.process.Args) !void {
    const allocator: std.mem.Allocator = std.heap.page_allocator;

    const opts: BenchOptions = parseArgs(args, allocator) catch |e| {
        std.debug.print("error: {s}\n\n", .{@errorName(e)});
        printUsage();
        return;
    };

    var prng = std.Random.DefaultPrng.init(0x1234_5678_9abc_def0);
    const rnd = prng.random();

    if (opts.print_cpuid) {
        const info = aion.cpu.cpuid.detect();
        std.debug.print(
            "cpu cpuid caches: l1d={} l2={} l3={}\n",
            .{ info.caches.l1d_bytes, info.caches.l2_bytes, info.caches.l3_bytes },
        );
    }

    var cpu: CpuBackend = try makeCpuBackend(allocator, opts.threads);
    defer cpu.deinit();
    const be: Backend = cpu.backend();

    std.debug.print("Aion bench (threads={}, iters={}, dtype={s})\n", .{ opts.threads, opts.iters, @tagName(opts.dtype_mode) });

    switch (opts.dtype_mode) {
        .f32 => try runF32Benches(allocator, rnd, opts, be),
        .f16 => try runF16Benches(allocator, rnd, opts, be),
        .both => {
            std.debug.print("\n=== f32 benches ===\n", .{});
            try runF32Benches(allocator, rnd, opts, be);
            std.debug.print("\n=== f16 benches ===\n", .{});
            try runF16Benches(allocator, rnd, opts, be);
        },
    }
}
