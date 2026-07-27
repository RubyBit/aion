// Copyright (c) 2026 Angelos-Ermis Mangos
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//
// A minimal multilayer-perceptron (MLP) classifier — the nominal Aion use case.
//
// This example shows the whole happy path with nothing extra:
//   1. bind weights into an `nn.Linear` stack,
//   2. describe the forward pass on a `Builder`,
//   3. `compile` it into a runnable `Model`,
//   4. write an input, `run`, and read the output.
//
// There is no streaming, recurrent state, or file I/O here — see
// `silero_vad.zig` for those. The weights are deterministic and synthetic so
// the example runs out of the box:
//
//   zig build examples -- --help
//   zig build examples
const std = @import("std");
const aion = @import("aion");

const api = aion.api;
const nn = api.nn;
const Builder = api.Builder;
const Tensor = api.Tensor;
const TensorRef = api.TensorRef;

const ExampleOptions = struct {
    in_dim: usize = 16,
    hidden_dim: usize = 32,
    num_classes: usize = 4,
    bench_iters: usize = 1,
};

// Stable names so inputs/outputs can be bound and read back by string.
const Names = struct {
    pub const x: []const u8 = "x";
    pub const logits: []const u8 = "logits";
    pub const probs: []const u8 = "probs";
};

/// A 3-layer perceptron: x -> relu(fc1) -> relu(fc2) -> fc_out.
///
/// `bind` registers the parameters on the builder once; `forward` wires the
/// activations. This is the same bind/forward split the `nn` layers use, so an
/// MLP composes into a larger model exactly like `nn.Linear` does.
const MLP = struct {
    fc1: nn.Linear,
    fc2: nn.Linear,
    fc_out: nn.Linear,

    const Self = @This();

    pub fn bind(bld: *Builder, weights: Weights) nn.BindError!Self {
        // A layer resolves its parameters from a source. The tensors are already in
        // hand here, so that source is a struct literal; the same layer would load
        // from a package by passing `pkg.at("fc1")` instead. Naming each layer makes
        // the exported parameter names readable (`fc1/weight`, `fc1/bias`, ...) and
        // stable, which is what lets a loaded model find and swap one layer later.
        return .{
            .fc1 = try nn.Linear.bind(bld, .{ .weight = weights.fc1_w, .bias = weights.fc1_b }, .{ .name = "fc1" }),
            .fc2 = try nn.Linear.bind(bld, .{ .weight = weights.fc2_w, .bias = weights.fc2_b }, .{ .name = "fc2" }),
            .fc_out = try nn.Linear.bind(bld, .{ .weight = weights.fc_out_w, .bias = weights.fc_out_b }, .{ .name = "fc_out" }),
        };
    }

    /// x: [batch, in_dim] -> logits: [batch, num_classes]
    pub fn forward(self: Self, bld: *Builder, x: TensorRef) Builder.Error!TensorRef {
        const h1: TensorRef = try bld.relu(try self.fc1.forward(bld, x));
        const h2: TensorRef = try bld.relu(try self.fc2.forward(bld, h1));
        return self.fc_out.forward(bld, h2);
    }
};

// Weight tensors, laid out for `y = x @ w + b`:
//   fcN_w: [in, out]   fcN_b: [out]
const Weights = struct {
    fc1_w: Tensor,
    fc1_b: Tensor,
    fc2_w: Tensor,
    fc2_b: Tensor,
    fc_out_w: Tensor,
    fc_out_b: Tensor,
};

fn buildModel(ctx: *api.Context, weights: Weights, in_dim: usize) !api.Model {
    var bld: Builder = api.Builder.init(ctx);
    defer bld.deinit();

    // A single public input. Batch is fixed at 1 here to keep the example small.
    const x_ref: TensorRef = try bld.name(try bld.input(.f32, &[_]usize{ 1, in_dim }), Names.x);

    const mlp: MLP = try MLP.bind(&bld, weights);
    const logits: TensorRef = try bld.name(try mlp.forward(&bld, x_ref), Names.logits);
    const probs: TensorRef = try bld.name(try bld.softmax(logits, -1), Names.probs);

    return ctx.compile(&bld, &[_]TensorRef{ logits, probs }, .{});
}

/// Deterministic pseudo-weights so the example is self-contained. In a real
/// program these tensors would come from `ctx.loadWeightsPath(...)` on an
/// exported `.aion` package (see `silero_vad.zig`).
fn createSyntheticWeights(ctx: *api.Context, allocator: std.mem.Allocator, opts: ExampleOptions) !Weights {
    return .{
        .fc1_w = try makeTensor(ctx, allocator, &[_]usize{ opts.in_dim, opts.hidden_dim }, 0.10, 0.20),
        .fc1_b = try makeTensor(ctx, allocator, &[_]usize{opts.hidden_dim}, 0.30, 0.05),
        .fc2_w = try makeTensor(ctx, allocator, &[_]usize{ opts.hidden_dim, opts.hidden_dim }, 0.40, 0.20),
        .fc2_b = try makeTensor(ctx, allocator, &[_]usize{opts.hidden_dim}, 0.50, 0.05),
        .fc_out_w = try makeTensor(ctx, allocator, &[_]usize{ opts.hidden_dim, opts.num_classes }, 0.60, 0.20),
        .fc_out_b = try makeTensor(ctx, allocator, &[_]usize{opts.num_classes}, 0.70, 0.05),
    };
}

fn makeTensor(ctx: *api.Context, allocator: std.mem.Allocator, shape: []const usize, phase: f32, amp: f32) !Tensor {
    var n: usize = 1;
    for (shape) |d| n = std.math.mul(usize, n, d) catch return error.InvalidArgument;

    const vals: []f32 = try allocator.alloc(f32, n);
    defer allocator.free(vals);
    for (vals, 0..) |*v, i| {
        const t: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(i % 97)))) * 0.031 + phase;
        v.* = amp * @sin(t);
    }
    return ctx.fromF32(shape, vals);
}

/// Deterministic input feature vector, so the printed output is reproducible.
fn fillInput(x: []f32) void {
    for (x, 0..) |*v, i| {
        const t: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(i)))) * 0.25;
        v.* = @sin(t) * 0.5;
    }
}

fn parseArgs(args: std.process.Args, allocator: std.mem.Allocator) !ExampleOptions {
    var out: ExampleOptions = .{};

    var it = try args.iterateAllocator(allocator);
    defer it.deinit();

    _ = it.next(); // argv[0]
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--in-dim")) {
            out.in_dim = try std.fmt.parseInt(usize, it.next() orelse return error.InvalidArgument, 10);
        } else if (std.mem.eql(u8, arg, "--hidden")) {
            out.hidden_dim = try std.fmt.parseInt(usize, it.next() orelse return error.InvalidArgument, 10);
        } else if (std.mem.eql(u8, arg, "--classes")) {
            out.num_classes = try std.fmt.parseInt(usize, it.next() orelse return error.InvalidArgument, 10);
        } else if (std.mem.eql(u8, arg, "--bench-iters")) {
            out.bench_iters = try std.fmt.parseInt(usize, it.next() orelse return error.InvalidArgument, 10);
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printUsage();
            return error.HelpRequested;
        } else {
            return error.InvalidArgument;
        }
    }

    if (out.in_dim == 0 or out.hidden_dim == 0 or out.num_classes == 0 or out.bench_iters == 0) {
        return error.InvalidArgument;
    }
    return out;
}

fn printUsage() void {
    std.debug.print(
        \\Minimal MLP classifier — the nominal Aion use case.
        \\
        \\Usage:
        \\  zig build examples -- [options]
        \\  zig build bench-examples -- [options]
        \\
        \\Options:
        \\  --in-dim <n>       Input feature count (default: 16)
        \\  --hidden <n>       Hidden layer width (default: 32)
        \\  --classes <n>      Output class count (default: 4)
        \\  --bench-iters <n>  Run the forward pass n times for timing (default: 1)
        \\  -h, --help         Show this help
        \\
        \\Weights are deterministic and synthetic, so no files are needed.
        \\
    , .{});
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

    const opts: ExampleOptions = parseArgs(args, allocator) catch |e| {
        if (e == error.HelpRequested) return;
        std.debug.print("error: {s}\n\n", .{@errorName(e)});
        printUsage();
        return;
    };

    var ctx: api.Context = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();

    const weights: Weights = try createSyntheticWeights(&ctx, allocator, opts);
    var model: api.Model = try buildModel(&ctx, weights, opts.in_dim);
    defer model.deinit();

    // One reusable input tensor, filled once.
    const x_tensor: Tensor = try ctx.tensor(.f32, &[_]usize{ 1, opts.in_dim });
    const x_host: []f32 = try allocator.alloc(f32, opts.in_dim);
    defer allocator.free(x_host);
    fillInput(x_host);
    try x_tensor.writeF32(x_host);
    try model.bindInput(Names.x, x_tensor);

    const probs: []f32 = try allocator.alloc(f32, opts.num_classes);
    defer allocator.free(probs);

    // Run (repeatedly, for timing). The bound input is reused each pass.
    const t0: u64 = nowNs();
    for (0..opts.bench_iters) |_| {
        try model.run();
    }
    const t1: u64 = nowNs();

    // Read the softmax probabilities back out and pick the winning class.
    const probs_t: Tensor = try model.outputTensor(Names.probs);
    try probs_t.readF32(probs);

    var best: usize = 0;
    for (probs, 0..) |p, i| {
        if (p > probs[best]) best = i;
    }

    const elapsed_ns: u64 = @max(@as(u64, 1), t1 - t0);
    const us_per_run: f64 = @as(f64, @floatFromInt(elapsed_ns)) / (@as(f64, @floatFromInt(opts.bench_iters)) * 1_000.0);

    std.debug.print("mlp example complete\n", .{});
    std.debug.print("  shape: in={} hidden={} classes={}\n", .{ opts.in_dim, opts.hidden_dim, opts.num_classes });
    std.debug.print("  probs: ", .{});
    for (probs, 0..) |p, i| std.debug.print("{s}{d:.4}", .{ if (i == 0) "" else " ", p });
    std.debug.print("\n", .{});
    std.debug.print("  argmax class = {}\n", .{best});
    std.debug.print("  perf: iters={} us/run={d:.2}\n", .{ opts.bench_iters, us_per_run });
}
