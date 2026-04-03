const std = @import("std");
const aion = @import("aion");

const api = aion.api;
const types = aion.types;
const nn = api.nn;
const Builder = api.Builder;
const Tensor = api.Tensor;
const TensorRef = api.TensorRef;

const ExampleOptions = struct {
    weights_path: ?[]u8 = null,
    chunks: usize = 8,
    num_samples: usize = 512,
    context_size: usize = 64,
    bench_iters: usize = 1,
    pad_mode: types.PadMode = .reflect,
    profile: bool = false,
    profile_steps: bool = false,

    /// Front-end choice:
    /// - true: use fused `complexAbsMean` op (single runtime step)
    /// - false: build the equivalent subgraph from primitive ops
    use_fused_complex_abs_mean: bool = true,

    pub fn deinit(self: *ExampleOptions, allocator: std.mem.Allocator) void {
        if (self.weights_path) |p| allocator.free(p);
        self.* = undefined;
    }
};

const ModelTensorNames = struct {
    pub const x: []const u8 = "x";
    pub const h: []const u8 = "h";
    pub const c: []const u8 = "c";
    pub const prob: []const u8 = "prob";
    pub const next_h: []const u8 = "next_h";
    pub const next_c: []const u8 = "next_c";
};

const TinySileroWeights = struct {
    stft_w: Tensor,

    conv1_w: Tensor,
    conv1_b: Tensor,
    conv2_w: Tensor,
    conv2_b: Tensor,
    conv3_w: Tensor,
    conv3_b: Tensor,
    conv4_w: Tensor,
    conv4_b: Tensor,

    lstm_w_ih: Tensor,
    lstm_w_hh: Tensor,
    lstm_b_ih: Tensor,
    lstm_b_hh: Tensor,

    final_w: Tensor,
    final_b: Tensor,
};

const TinySileroState = struct {
    h: TensorRef,
    c: TensorRef,
};

const TinySileroForward = struct {
    features: TensorRef,
    prob: TensorRef,
    h: TensorRef,
    c: TensorRef,
};

const TinySileroVAD = struct {
    n_fft: usize = 256,
    stride: usize = 128,
    pad: usize = 64,
    cutoff: usize = 129,

    stft_conv: nn.Conv1D,
    conv1: nn.Conv1D,
    conv2: nn.Conv1D,
    conv3: nn.Conv1D,
    conv4: nn.Conv1D,

    lstm_cell: nn.LSTMCell,
    final_conv: nn.Conv1D,

    const Self = @This();

    pub fn bind(bld: *Builder, weights: TinySileroWeights, pad_mode: types.PadMode) Builder.Error!Self {
        const stft_conv: nn.Conv1D = try nn.Conv1D.bind(bld, weights.stft_w, null, .{
            .stride = 128,
            .dilation = 1,
            .pad_left = 0,
            .pad_right = 64,
            .pad_mode = pad_mode,
            .groups = 1,
        });

        const conv1: nn.Conv1D = try nn.Conv1D.bind(bld, weights.conv1_w, weights.conv1_b, .{
            .stride = 1,
            .dilation = 1,
            .pad_left = 1,
            .pad_right = 1,
            .pad_mode = .zero,
            .groups = 1,
        });
        const conv2: nn.Conv1D = try nn.Conv1D.bind(bld, weights.conv2_w, weights.conv2_b, .{
            .stride = 2,
            .dilation = 1,
            .pad_left = 1,
            .pad_right = 1,
            .pad_mode = .zero,
            .groups = 1,
        });
        const conv3: nn.Conv1D = try nn.Conv1D.bind(bld, weights.conv3_w, weights.conv3_b, .{
            .stride = 2,
            .dilation = 1,
            .pad_left = 1,
            .pad_right = 1,
            .pad_mode = .zero,
            .groups = 1,
        });
        const conv4: nn.Conv1D = try nn.Conv1D.bind(bld, weights.conv4_w, weights.conv4_b, .{
            .stride = 1,
            .dilation = 1,
            .pad_left = 1,
            .pad_right = 1,
            .pad_mode = .zero,
            .groups = 1,
        });

        if (weights.lstm_w_ih.shape.len != 2 or weights.lstm_w_hh.shape.len != 2) return Builder.Error.InvalidArgument;
        const input_size: usize = weights.lstm_w_ih.shape[0];
        const gate_dim: usize = weights.lstm_w_ih.shape[1];
        if (input_size != 128 or gate_dim != 512) return Builder.Error.InvalidArgument;
        if (weights.lstm_w_hh.shape[0] != 128 or weights.lstm_w_hh.shape[1] != 512) return Builder.Error.InvalidArgument;
        if (weights.lstm_b_ih.shape.len != 1 or weights.lstm_b_ih.shape[0] != 512) return Builder.Error.InvalidArgument;
        if (weights.lstm_b_hh.shape.len != 1 or weights.lstm_b_hh.shape[0] != 512) return Builder.Error.InvalidArgument;

        const lstm_cell: nn.LSTMCell = try nn.LSTMCell.bind(
            bld,
            weights.lstm_w_ih,
            weights.lstm_w_hh,
            weights.lstm_b_ih,
            weights.lstm_b_hh,
        );

        const final_conv: nn.Conv1D = try nn.Conv1D.bind(bld, weights.final_w, weights.final_b, .{
            .stride = 1,
            .dilation = 1,
            .pad_left = 0,
            .pad_right = 0,
            .pad_mode = .zero,
            .groups = 1,
        });

        return .{
            .stft_conv = stft_conv,
            .conv1 = conv1,
            .conv2 = conv2,
            .conv3 = conv3,
            .conv4 = conv4,
            .lstm_cell = lstm_cell,
            .final_conv = final_conv,
        };
    }

    pub fn forward(self: Self, bld: *Builder, x: TensorRef, state: TinySileroState, use_fused_absmean: bool) Builder.Error!TinySileroForward {
        // x: [batch, time]
        const x_shape: []const usize = try bld.requireKnownShape(x);
        if (x_shape.len != 2) return Builder.Error.InvalidArgument;
        const batch: usize = x_shape[0];
        const in_len: usize = x_shape[1];
        if (in_len < self.n_fft) return Builder.Error.InvalidArgument;

        const padded_len: usize = std.math.add(usize, in_len, self.pad) catch return Builder.Error.InvalidArgument;
        if (padded_len < self.n_fft) return Builder.Error.InvalidArgument;
        const numer: usize = padded_len - self.n_fft;
        const t_steps: usize = (numer / self.stride) + 1;
        if (t_steps == 0) return Builder.Error.InvalidArgument;

        const x3: TensorRef = try bld.unsqueeze(x, 2); // [batch, time, 1]

        // stft_conv: [batch, t', 258]
        const stft: TensorRef = try self.stft_conv.forward(bld, x3);

        if (bld.knownShape(stft)) |stft_shape| {
            if (stft_shape.len != 3) return Builder.Error.InvalidArgument;
            if (stft_shape[0] != batch) return Builder.Error.InvalidArgument;
            if (stft_shape[1] != t_steps) return Builder.Error.InvalidArgument;
            if (stft_shape[2] != self.cutoff * 2) return Builder.Error.InvalidArgument;
        }

        const y4_2d: TensorRef = if (use_fused_absmean)
            // Fused front-end: abs_mean = mean_t sqrt(real^2 + imag^2)
            // and keep the first `hidden` channels for the LSTM input.
            try bld.complexAbsMean(stft, self.lstm_cell.hidden_size)
        else
            // Unfused reference: build the equivalent subgraph from primitive ops.
            try complexAbsMeanUnfused(bld, stft, batch, t_steps, self.cutoff, self.lstm_cell.hidden_size);

        const st: nn.LSTMState = try self.lstm_cell.forward(bld, y4_2d, state.h, state.c);

        const h3: TensorRef = try bld.unsqueeze(st.h, 1); // [batch, 1, 128]
        const act: TensorRef = try bld.relu(h3);
        const logits3: TensorRef = try self.final_conv.forward(bld, act); // [batch, 1, 1]
        const probs3: TensorRef = try bld.sigmoid(logits3);
        const probs2: TensorRef = try bld.squeeze(probs3, 2); // [batch, 1]
        const probs1: TensorRef = try bld.reduceAxis(.mean, probs2, 1); // [batch]
        const probs: TensorRef = try bld.unsqueeze(probs1, 1); // [batch, 1]

        return .{ .features = y4_2d, .prob = probs, .h = st.h, .c = st.c };
    }
};

fn parseArgs(args: std.process.Args, allocator: std.mem.Allocator) !ExampleOptions {
    var out: ExampleOptions = .{};

    var it = try args.iterateAllocator(allocator);
    defer it.deinit();

    _ = it.next(); // argv[0]
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--weights")) {
            const p: []const u8 = it.next() orelse return error.InvalidArgument;
            if (out.weights_path) |old| allocator.free(old);
            out.weights_path = try allocator.dupe(u8, p);
        } else if (std.mem.eql(u8, arg, "--chunks")) {
            const v: []const u8 = it.next() orelse return error.InvalidArgument;
            out.chunks = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, arg, "--num-samples")) {
            const v: []const u8 = it.next() orelse return error.InvalidArgument;
            out.num_samples = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, arg, "--context")) {
            const v: []const u8 = it.next() orelse return error.InvalidArgument;
            out.context_size = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, arg, "--bench-iters")) {
            const v: []const u8 = it.next() orelse return error.InvalidArgument;
            out.bench_iters = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, arg, "--pad-mode")) {
            const v: []const u8 = it.next() orelse return error.InvalidArgument;
            if (std.mem.eql(u8, v, "reflect")) {
                out.pad_mode = .reflect;
            } else if (std.mem.eql(u8, v, "zero")) {
                out.pad_mode = .zero;
            } else {
                return error.InvalidArgument;
            }
        } else if (std.mem.eql(u8, arg, "--profile")) {
            out.profile = true;
        } else if (std.mem.eql(u8, arg, "--profile-steps")) {
            out.profile_steps = true;
        } else if (std.mem.eql(u8, arg, "--no-fused-absmean")) {
            out.use_fused_complex_abs_mean = false;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printUsage();
            return error.HelpRequested;
        } else {
            return error.InvalidArgument;
        }
    }

    if (out.chunks == 0 or out.num_samples == 0 or out.context_size == 0 or out.bench_iters == 0) return error.InvalidArgument;
    if (out.context_size != 64) {
        std.debug.print("warning: TinySilero reference context is 64; got {}\n", .{out.context_size});
    }

    return out;
}

fn printUsage() void {
    std.debug.print(
        \\TinySileroVAD-style Aion integration example
        \\
        \\Usage:
        \\  zig build examples -- [options]
        \\  zig build bench-examples -- [options]
        \\
        \\Options:
        \\  --weights <path>       Optional .aion model package to load directly
        \\  --chunks <n>           Number of streaming chunks (default: 8)
        \\  --num-samples <n>      Samples per chunk (default: 512)
        \\  --context <n>          Context size prepended to chunk (default: 64)
        \\  --bench-iters <n>      Run the streaming loop n times for timing (default: 1)
        \\  --pad-mode <reflect|zero>  STFT pre-conv pad mode (default: reflect)
        \\  --profile              Print a rough per-chunk time breakdown (adds overhead)
        \\  --profile-steps        Print per-step timings for the first model.run() (adds overhead)
        \\  --no-fused-absmean     Build abs-mean front-end from primitive ops (no fused op)
        \\  -h, --help             Show this help
        \\
        \\If --weights is omitted, the example creates deterministic synthetic weights,
        \\exports a temporary `.aion` model package, reloads it, and runs streaming
        \\inference from that package rather than rebuilding the architecture for execution.
        \\
    , .{});
}

fn complexAbsMeanUnfused(
    bld: *Builder,
    stft: TensorRef,
    batch: usize,
    time: usize,
    cutoff: usize,
    out_channels: usize,
) Builder.Error!TensorRef {
    if (batch == 0 or time == 0 or cutoff == 0) return Builder.Error.InvalidArgument;
    if (out_channels == 0 or out_channels > cutoff) return Builder.Error.InvalidArgument;

    // stft is [batch, time, 2*cutoff]. Real is first half, Imag is second half.
    const real: TensorRef = try bld.slice(stft, &[_]usize{ 0, 0, 0 }, &[_]usize{ batch, time, out_channels });
    const imag: TensorRef = try bld.slice(stft, &[_]usize{ 0, 0, cutoff }, &[_]usize{ batch, time, out_channels });

    const r2: TensorRef = try bld.mul(real, real);
    const im2: TensorRef = try bld.mul(imag, imag);
    const mag2: TensorRef = try bld.add(r2, im2);
    const mag: TensorRef = try bld.sqrt(mag2);

    // Mean over time axis => [batch, out_channels]
    return bld.reduceAxis(.mean, mag, 1);
}

fn createDeterministicTensor(
    ctx: *api.Context,
    allocator: std.mem.Allocator,
    shape: []const usize,
    phase: f32,
    amp: f32,
) !Tensor {
    var n: usize = 1;
    for (shape) |d| {
        n = std.math.mul(usize, n, d) catch return error.InvalidArgument;
    }

    const vals: []f32 = try allocator.alloc(f32, n);
    defer allocator.free(vals);

    for (vals, 0..) |*v, i| {
        const x: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(i % 251)))) * 0.03125 + phase;
        v.* = amp * @sin(x);
    }

    return ctx.fromF32(shape, vals);
}

fn createSyntheticWeights(ctx: *api.Context, allocator: std.mem.Allocator) !TinySileroWeights {
    return .{
        .stft_w = try createDeterministicTensor(ctx, allocator, &[_]usize{ 256, 1, 258 }, 0.10, 0.025),

        .conv1_w = try createDeterministicTensor(ctx, allocator, &[_]usize{ 3, 129, 128 }, 0.20, 0.020),
        .conv1_b = try createDeterministicTensor(ctx, allocator, &[_]usize{128}, 0.30, 0.015),
        .conv2_w = try createDeterministicTensor(ctx, allocator, &[_]usize{ 3, 128, 64 }, 0.40, 0.020),
        .conv2_b = try createDeterministicTensor(ctx, allocator, &[_]usize{64}, 0.50, 0.015),
        .conv3_w = try createDeterministicTensor(ctx, allocator, &[_]usize{ 3, 64, 64 }, 0.60, 0.020),
        .conv3_b = try createDeterministicTensor(ctx, allocator, &[_]usize{64}, 0.70, 0.015),
        .conv4_w = try createDeterministicTensor(ctx, allocator, &[_]usize{ 3, 64, 128 }, 0.80, 0.020),
        .conv4_b = try createDeterministicTensor(ctx, allocator, &[_]usize{128}, 0.90, 0.015),

        .lstm_w_ih = try createDeterministicTensor(ctx, allocator, &[_]usize{ 128, 512 }, 1.00, 0.020),
        .lstm_w_hh = try createDeterministicTensor(ctx, allocator, &[_]usize{ 128, 512 }, 1.10, 0.020),
        .lstm_b_ih = try createDeterministicTensor(ctx, allocator, &[_]usize{512}, 1.20, 0.015),
        .lstm_b_hh = try createDeterministicTensor(ctx, allocator, &[_]usize{512}, 1.30, 0.015),

        .final_w = try createDeterministicTensor(ctx, allocator, &[_]usize{ 1, 128, 1 }, 1.40, 0.020),
        .final_b = try createDeterministicTensor(ctx, allocator, &[_]usize{1}, 1.50, 0.010),
    };
}

fn exportTinySileroPackage(
    ctx: *api.Context,
    allocator: std.mem.Allocator,
    weights: TinySileroWeights,
    chunk_input_len: usize,
    pad_mode: types.PadMode,
    use_fused_absmean: bool,
    file: std.Io.File,
) !void {
    var bld: Builder = api.Builder.init(allocator);
    defer bld.deinit();

    const x_ref: TensorRef = try bld.name(try bld.input(.f32, &[_]usize{ 1, chunk_input_len }), ModelTensorNames.x);
    const h_ref: TensorRef = try bld.name(try bld.input(.f32, &[_]usize{ 1, 128 }), ModelTensorNames.h);
    const c_ref: TensorRef = try bld.name(try bld.input(.f32, &[_]usize{ 1, 128 }), ModelTensorNames.c);

    const tiny: TinySileroVAD = try TinySileroVAD.bind(&bld, weights, pad_mode);
    const out: TinySileroForward = try tiny.forward(&bld, x_ref, .{ .h = h_ref, .c = c_ref }, use_fused_absmean);

    try ctx.exportModel(file, &bld, &[_]api.NamedTensorRef{
        .{ .name = ModelTensorNames.prob, .tensor = out.prob },
        .{ .name = ModelTensorNames.next_h, .tensor = out.h },
        .{ .name = ModelTensorNames.next_c, .tensor = out.c },
    }, .{
        .metadata = &[_]api.ExportMetadata{
            .{ .key = "arch", .value = "tiny-silero-vad" },
        },
        .input_symbols = &[_]api.DimensionSymbol{
            .{ .tensor = x_ref, .axis = 0, .name = "batch" },
            .{ .tensor = h_ref, .axis = 0, .name = "batch" },
            .{ .tensor = c_ref, .axis = 0, .name = "batch" },
        },
        .output_aliases = &[_]api.OutputAlias{
            .{ .input_name = ModelTensorNames.h, .output_name = ModelTensorNames.next_h },
            .{ .input_name = ModelTensorNames.c, .output_name = ModelTensorNames.next_c },
        },
    });
}

fn fillSyntheticAudio(signal: []f32, context_size: usize) void {
    @memset(signal[0..context_size], 0.0);

    var i: usize = context_size;
    while (i < signal.len) : (i += 1) {
        const t: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(i - context_size))));
        const s0: f32 = 0.35 * @sin(2.0 * std.math.pi * 220.0 * (t / 16_000.0));
        const s1: f32 = 0.20 * @sin(2.0 * std.math.pi * 440.0 * (t / 16_000.0));
        const n0: f32 = 0.02 * @sin(2.0 * std.math.pi * 37.0 * (t / 16_000.0));
        signal[i] = s0 + s1 + n0;
    }
}

fn nowNs() u64 {
    const ts: std.Io.Timestamp = std.Io.Clock.awake.now(std.Options.debug_io);
    const ns: i96 = ts.toNanoseconds();
    if (ns <= 0) return 0;
    const max_u64_i96: i96 = @as(i96, std.math.maxInt(u64));
    return @intCast(@min(ns, max_u64_i96));
}

pub fn main(minimal: std.process.Init.Minimal) void {
    mainImpl(minimal.args) catch |e| {
        if (e == error.HelpRequested) return;
        std.debug.print("fatal: {s}\n", .{@errorName(e)});
        std.process.exit(1);
    };
}

fn mainImpl(args: std.process.Args) !void {
    const allocator: std.mem.Allocator = std.heap.page_allocator;

    var opts: ExampleOptions = parseArgs(args, allocator) catch |e| {
        if (e == error.HelpRequested) return;
        std.debug.print("error: {s}\n\n", .{@errorName(e)});
        printUsage();
        return;
    };
    defer opts.deinit(allocator);

    const chunk_input_len: usize = opts.context_size + opts.num_samples;
    var ctx: api.Context = try api.Context.initCpu(allocator, .{ .thread_count = 1, .profile_steps = opts.profile_steps });
    defer ctx.deinit();

    var loaded_model: api.LoadedModel = if (opts.weights_path) |p| blk: {
        std.debug.print("package: loading model package from {s}\n", .{p});
        break :blk try ctx.loadModelPath(p, .{});
    } else blk: {
        std.debug.print("package: exporting deterministic synthetic model to a temporary .aion and reloading it\n", .{});

        var export_ctx: api.Context = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
        defer export_ctx.deinit();
        const weights = try createSyntheticWeights(&export_ctx, allocator);

        var io_backend: std.Io.Threaded = .init_single_threaded;
        const io = io_backend.io();
        const synthetic_path = ".zig-cache/tiny_silero_vad.synthetic.aion";
        const file = try std.Io.Dir.cwd().createFile(io, synthetic_path, .{ .read = true, .truncate = true });
        defer {
            file.close(io);
            std.Io.Dir.cwd().deleteFile(io, synthetic_path) catch {};
        }

        try exportTinySileroPackage(&export_ctx, allocator, weights, chunk_input_len, opts.pad_mode, opts.use_fused_complex_abs_mean, file);
        break :blk try ctx.loadModel(file, .{});
    };
    defer loaded_model.deinit();

    const x_shape: [2]usize = .{ 1, chunk_input_len };
    const h_shape: [2]usize = .{ 1, 128 };
    const c_shape: [2]usize = .{ 1, 128 };
    const x_tensor: Tensor = try ctx.tensor(.f32, x_shape[0..2]);
    const h_tensor: Tensor = try ctx.tensor(.f32, h_shape[0..2]);
    const c_tensor: Tensor = try ctx.tensor(.f32, c_shape[0..2]);

    var zeros_h: [128]f32 = .{0.0} ** 128;
    var zeros_c: [128]f32 = .{0.0} ** 128;
    try h_tensor.writeF32(zeros_h[0..]);
    try c_tensor.writeF32(zeros_c[0..]);
    try loaded_model.bindInput(ModelTensorNames.x, x_tensor);
    try loaded_model.bindInput(ModelTensorNames.h, h_tensor);
    try loaded_model.bindInput(ModelTensorNames.c, c_tensor);

    const total_len: usize = opts.context_size + opts.chunks * opts.num_samples;
    const signal: []f32 = try allocator.alloc(f32, total_len);
    defer allocator.free(signal);
    fillSyntheticAudio(signal, opts.context_size);

    const chunk_buf: []f32 = try allocator.alloc(f32, chunk_input_len);
    defer allocator.free(chunk_buf);

    var h_buf: [128]f32 = undefined;
    var c_buf: [128]f32 = undefined;

    const probs: []f32 = try allocator.alloc(f32, opts.chunks);
    defer allocator.free(probs);
    var prob_t_opt: ?Tensor = null;
    var next_h_t_opt: ?Tensor = null;
    var next_c_t_opt: ?Tensor = null;
    var t_memcpy_ns: u64 = 0;
    var t_write_in_ns: u64 = 0;
    var t_run_ns: u64 = 0;
    var t_read_out_ns: u64 = 0;

    const t0_ns: u64 = nowNs();
    for (0..opts.bench_iters) |_| {
        try h_tensor.writeF32(zeros_h[0..]);
        try c_tensor.writeF32(zeros_c[0..]);
        try loaded_model.bindInput(ModelTensorNames.h, h_tensor);
        try loaded_model.bindInput(ModelTensorNames.c, c_tensor);

        for (0..opts.chunks) |chunk_idx| {
            const start: usize = chunk_idx * opts.num_samples;
            const end: usize = start + chunk_input_len;

            const t_a: u64 = if (opts.profile) nowNs() else 0;
            @memcpy(chunk_buf, signal[start..end]);
            const t_b: u64 = if (opts.profile) nowNs() else 0;
            if (opts.profile) t_memcpy_ns += (t_b - t_a);

            const t_c: u64 = if (opts.profile) nowNs() else 0;
            x_tensor.writeF32(chunk_buf) catch |e| {
                std.debug.print("write input failed at chunk {}: {s}\n", .{ chunk_idx, @errorName(e) });
                return e;
            };
            const t_d: u64 = if (opts.profile) nowNs() else 0;
            if (opts.profile) t_write_in_ns += (t_d - t_c);

            const t_e: u64 = if (opts.profile) nowNs() else 0;
            loaded_model.run() catch |e| {
                std.debug.print("loaded model run failed at chunk {}: {s}\n", .{ chunk_idx, @errorName(e) });
                return e;
            };
            const t_f: u64 = if (opts.profile) nowNs() else 0;
            if (opts.profile) t_run_ns += (t_f - t_e);

            const t_g: u64 = if (opts.profile) nowNs() else 0;
            if (prob_t_opt == null) {
                prob_t_opt = try loaded_model.outputTensor(ModelTensorNames.prob);
                next_h_t_opt = try loaded_model.outputTensor(ModelTensorNames.next_h);
                next_c_t_opt = try loaded_model.outputTensor(ModelTensorNames.next_c);
            }
            const prob_t = prob_t_opt.?;
            var prob_tmp: [1]f32 = .{0.0};
            prob_t.readF32(prob_tmp[0..]) catch |e| {
                std.debug.print("read prob failed at chunk {}: {s}\n", .{ chunk_idx, @errorName(e) });
                return e;
            };
            probs[chunk_idx] = prob_tmp[0];

            const t_h: u64 = if (opts.profile) nowNs() else 0;
            if (opts.profile) t_read_out_ns += (t_h - t_g);
        }
    }
    const t1_ns: u64 = nowNs();

    // Snapshot final streaming state once (for a cheap correctness signal).
    try next_h_t_opt.?.readF32(h_buf[0..]);
    try next_c_t_opt.?.readF32(c_buf[0..]);

    const elapsed_ns_u: u64 = @max(@as(u64, 1), t1_ns - t0_ns);
    const total_chunks: usize = opts.bench_iters * opts.chunks;
    const chunks_per_sec: f64 = (@as(f64, @floatFromInt(total_chunks)) * 1_000_000_000.0) / @as(f64, @floatFromInt(elapsed_ns_u));
    const us_per_chunk: f64 = @as(f64, @floatFromInt(elapsed_ns_u)) / (@as(f64, @floatFromInt(total_chunks)) * 1_000.0);

    var p_sum: f32 = 0.0;
    var p_min: f32 = probs[0];
    var p_max: f32 = probs[0];
    for (probs) |p| {
        p_sum += p;
        p_min = @min(p_min, p);
        p_max = @max(p_max, p);
    }
    const p_mean: f32 = p_sum / @as(f32, @floatFromInt(opts.chunks));

    std.debug.print("tiny-silero example complete\n", .{});
    std.debug.print("  chunks={}  num_samples={}  context={}  pad_mode={s}  bench_iters={}\n", .{ opts.chunks, opts.num_samples, opts.context_size, @tagName(opts.pad_mode), opts.bench_iters });
    std.debug.print("  probs: first={d:.6} last={d:.6} min={d:.6} max={d:.6} mean={d:.6}\n", .{ probs[0], probs[opts.chunks - 1], p_min, p_max, p_mean });
    std.debug.print("  final_state checksum: h0={d:.6} c0={d:.6}\n", .{ h_buf[0], c_buf[0] });
    std.debug.print("  perf: total_chunks={}  chunks/s={d:.2}  us/chunk={d:.2}\n", .{ total_chunks, chunks_per_sec, us_per_chunk });

    if (opts.profile) {
        const denom: f64 = @as(f64, @floatFromInt(@max(@as(usize, 1), total_chunks)));
        const memcpy_us: f64 = @as(f64, @floatFromInt(t_memcpy_ns)) / (denom * 1_000.0);
        const write_us: f64 = @as(f64, @floatFromInt(t_write_in_ns)) / (denom * 1_000.0);
        const run_us: f64 = @as(f64, @floatFromInt(t_run_ns)) / (denom * 1_000.0);
        const read_us: f64 = @as(f64, @floatFromInt(t_read_out_ns)) / (denom * 1_000.0);
        std.debug.print("  profile (approx, adds overhead): memcpy={d:.2}us write_in={d:.2}us run={d:.2}us read_out={d:.2}us\n", .{ memcpy_us, write_us, run_us, read_us });
    }
}
