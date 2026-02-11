const std = @import("std");
const aion = @import("aion");

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

const BenchOptions = struct {
    iters: usize = 50,
    threads: usize = 1,

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

    // Print cpu cache detection / tuning selection.
    print_cpuid: bool = false,
};

fn printUsage() void {
    std.debug.print(
        "Aion kernel benchmarks\n\n" ++
            "Usage:\n" ++
            "  zig build bench -- [options]\n\n" ++
            "Options:\n" ++
            "  --iters N        Iterations per benchmark (default: 50)\n" ++
            "  --threads N      CPU backend thread count (default: 1)\n" ++
            "  --batch N        Batch size for batched matmul (default: 4)\n" ++
            "  --heads N        Multi-head attention heads (default: 8)\n" ++
            "  --n-elem N       Elemwise/Reduce logical element count (default: 8388608)\n" ++
            "  --m N            MatMul M (default: 512)\n" ++
            "  --n N            MatMul N (default: 512)\n" ++
            "  --k N            MatMul K (default: 512)\n" ++
            "  --no-quant       Skip quantized matmul benches\n" ++
            "  --print-cpuid    Print detected CPU caches\n" ++
            "  -h, --help       Show help\n",
        .{},
    );
}

fn parseUsize(arg: []const u8) !usize {
    return std.fmt.parseInt(usize, arg, 10);
}

fn parseArgs(allocator: std.mem.Allocator) !BenchOptions {
    var opts: BenchOptions = .{};

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "-h") or std.mem.eql(u8, a, "--help")) {
            printUsage();
            std.process.exit(0);
        } else if (std.mem.eql(u8, a, "--no-quant")) {
            opts.quant = false;
        } else if (std.mem.eql(u8, a, "--print-cpuid")) {
            opts.print_cpuid = true;
        } else if (std.mem.eql(u8, a, "--iters")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgument;
            opts.iters = try parseUsize(args[i]);
        } else if (std.mem.eql(u8, a, "--threads")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgument;
            opts.threads = try parseUsize(args[i]);
        } else if (std.mem.eql(u8, a, "--batch")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgument;
            opts.batch = try parseUsize(args[i]);
        } else if (std.mem.eql(u8, a, "--heads")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgument;
            opts.heads = try parseUsize(args[i]);
        } else if (std.mem.eql(u8, a, "--n-elem")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgument;
            opts.n_elem = try parseUsize(args[i]);
        } else if (std.mem.eql(u8, a, "--m")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgument;
            opts.m = try parseUsize(args[i]);
        } else if (std.mem.eql(u8, a, "--n")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgument;
            opts.n = try parseUsize(args[i]);
        } else if (std.mem.eql(u8, a, "--k")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgument;
            opts.k = try parseUsize(args[i]);
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

    return opts;
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

fn asF32Slice(buf: []u8) []align(1) f32 {
    std.debug.assert((buf.len % @sizeOf(f32)) == 0);
    const ptr: [*]align(1) f32 = @ptrCast(buf.ptr);
    return ptr[0 .. buf.len / @sizeOf(f32)];
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
    std.debug.print("program elemwise add f32: {d:.3} GiB/s  (chk={d:.4})\n", .{ gib_s, chk });
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
    std.debug.print("program unary {s} f32:    {d:.3} GiB/s  (chk={d:.4})\n", .{ label, gib_s, chk });
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
    std.debug.print("program reduce sum f32:   {d:.3} GiB/s  (sum={d:.4})\n", .{ gib_s, sum });
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
    std.debug.print("program softmax f32:     {d:.3} GiB/s  (chk={d:.4})\n", .{ gib_s, chk });
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
    std.debug.print("program layernorm f32:   {d:.3} GiB/s  (chk={d:.4})\n", .{ gib_s, chk });
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
    std.debug.print("program rmsnorm f32:     {d:.3} GiB/s  (chk={d:.4})\n", .{ gib_s, chk });
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
        "program attention f32 {s}: {d:.2} GFLOP/s (m={} n={} dk={} dv={}, chk={d:.4})\n",
        .{ mode, gflops, m, n, dk, dv, chk },
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
        "program mha separate f32 {s}: {d:.2} GFLOP/s (b={} h={} m={} n={} dk={} dv={}, chk={d:.4})\n",
        .{ mode, gflops, batch, heads, m_head, n_head, dk, dv, chk },
    );
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
    std.debug.print("program matmul f32:       {d:.2} GFLOP/s (m={} n={} k={}, chk={d:.4})\n", .{ gflops, m, n, k, chk });
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

    std.debug.print("program matmul batched f32: {d:.2} GFLOP/s (b={} m={} n={} k={}, chk={d:.4})\n", .{ gflops, batch, m, n, k, chk });
}

fn benchProgramMatmulQuant(allocator: std.mem.Allocator, rnd: std.Random, iters: usize, m: usize, n: usize, k: usize, b_dtype: types.DType, be: Backend) !void {
    var sm = StorageManager.init(allocator);
    defer sm.deinit();

    const policy: plan_mod.TilePolicy = defaultTilePolicy();
    const tiles = plan_mod.chooseMatMulTiles(policy, m, n, k, b_dtype);

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
    std.debug.print("program matmul {s}:      {d:.2} GFLOP/s (m={} n={} k={}, chk={d:.4})\n", .{ name, gflops, m, n, k, chk });
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

fn nowNs(timer: *std.time.Timer) u64 {
    return timer.read();
}

fn benchLoop(iters: usize, work_in: anytype) !u64 {
    // Work is passed as a tiny context object with a `run()` method.
    // Using a context avoids heap allocations and gives us capture-like behavior.
    var work = work_in;

    // Warm-up (helps with cache and first-touch page faults).
    try work.run();

    var t = try std.time.Timer.start();
    const start: u64 = nowNs(&t);
    var i: usize = 0;
    while (i < iters) : (i += 1) {
        try work.run();
    }
    const end: u64 = nowNs(&t);
    return end - start;
}

fn fillRandomF32(rnd: std.Random, data: []f32) void {
    for (data) |*x| {
        // Keep values in a tame range to avoid NaNs/denorm weirdness.
        x.* = (rnd.float(f32) - 0.5) * 2.0;
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
    var b_f32: []f32 = try allocator.alloc(f32, k * n);
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

    var b_f32: []f32 = try allocator.alloc(f32, k * n);
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

pub fn main() void {
    mainImpl() catch |e| {
        std.debug.print("fatal: {s}\n", .{@errorName(e)});
        std.process.exit(1);
    };
}

fn mainImpl() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator: std.mem.Allocator = gpa.allocator();

    const opts: BenchOptions = parseArgs(allocator) catch |e| {
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

    std.debug.print("Aion bench (threads={}, iters={})\n", .{ opts.threads, opts.iters });

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
    try benchProgramReduceSum(allocator, rnd, opts.iters, opts.n_elem, be);
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
