// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! API-level device-selection tests. Built as its own artifact against the `aion`
//! module (enable_gpu=true) — see `addApiGpuTest` in build.zig. Every test skips
//! cleanly (`error.SkipZigTest`) when no GPU adapter is present (headless CI).
const std = @import("std");
const aion = @import("aion");
const api = aion.api;

/// Build a small model `out = mean(relu(a @ b))` on `ctx` targeting `dev`, run it,
/// and return the scalar output. Small single-tile operands so CPU and GPU tilings
/// coincide (no cross-device retile needed at this foundation stage).
fn runSmallModel(
    ctx: *api.Context,
    dev: api.DeviceSelector,
    a_t: api.Tensor,
    b_t: api.Tensor,
) !f32 {
    var bld = api.Builder.init(ctx);
    defer bld.deinit();

    const A = try bld.param(a_t);
    const B = try bld.param(b_t);
    const C = try bld.matmul(A, B, 1.0, 0.0);
    const E = try bld.relu(C);
    const Out = try bld.reduce(.mean, E);

    var model = try ctx.compileOn(dev, &bld, &[_]api.TensorRef{Out}, .{});
    defer model.deinit();

    const out_t: api.Tensor = try model.runOutputTensor(0);
    return out_t.readScalar(f32);
}

test "api: gpu model output matches cpu model (compileOn)" {
    const alloc = std.testing.allocator;

    var ctx = api.Context.init(alloc, .{ .gpus = &.{.{ .power = .high }} }) catch |e| switch (e) {
        error.BackendUnavailable => return error.SkipZigTest, // no adapter / headless
        else => return e,
    };
    defer ctx.deinit();

    const m: usize = 4;
    const k: usize = 8;
    const n: usize = 4;

    var a_vals: [m * k]f32 = undefined;
    var b_vals: [k * n]f32 = undefined;
    for (&a_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 5)) * 0.25;
    for (&b_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) * 0.5;

    const a_t = try ctx.fromF32(&[_]usize{ m, k }, &a_vals);
    const b_t = try ctx.fromF32(&[_]usize{ k, n }, &b_vals);

    const cpu_out = try runSmallModel(&ctx, .cpu, a_t, b_t);
    const gpu_out = try runSmallModel(&ctx, .{ .gpu = 0 }, a_t, b_t);

    try std.testing.expect(std.math.isFinite(gpu_out));
    try std.testing.expectApproxEqAbs(cpu_out, gpu_out, 1e-4);
}

fn runBatchedMatmul(
    ctx: *api.Context,
    dev: api.DeviceSelector,
    a_t: api.Tensor,
    b_t: api.Tensor,
) ![24]f32 {
    var bld = api.Builder.init(ctx);
    defer bld.deinit();

    const A = try bld.param(a_t); // [batch=2, seq=3, k=8]
    const B = try bld.param(b_t); // [k=8, n=4], broadcast over batch/seq
    const C = try bld.matmul(A, B, 1.0, 0.0);

    var model = try ctx.compileOn(dev, &bld, &[_]api.TensorRef{C}, .{});
    defer model.deinit();

    const out = try model.runOutputTensor(0);
    var vals: [24]f32 = undefined;
    try out.read(&vals);
    return vals;
}

test "api: gpu batched matmul broadcasts rank-2 weight (matches cpu)" {
    const alloc = std.testing.allocator;

    var ctx = api.Context.init(alloc, .{ .gpus = &.{.{ .power = .high }} }) catch |e| switch (e) {
        error.BackendUnavailable => return error.SkipZigTest,
        else => return e,
    };
    defer ctx.deinit();

    var a_vals: [2 * 3 * 8]f32 = undefined;
    var b_vals: [8 * 4]f32 = undefined;
    for (&a_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) * 0.125;
    for (&b_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) * 0.25;

    const a_t = try ctx.fromF32(&.{ 2, 3, 8 }, &a_vals);
    const b_t = try ctx.fromF32(&.{ 8, 4 }, &b_vals);

    const cpu = try runBatchedMatmul(&ctx, .cpu, a_t, b_t);
    const gpu_out = try runBatchedMatmul(&ctx, .{ .gpu = 0 }, a_t, b_t);
    for (cpu, gpu_out) |want, got| try std.testing.expectApproxEqAbs(want, got, 1e-4);
}

/// Run a 2-step KV-cache "decode" on `dev`: an in-place `sequenceAppend` whose
/// output is io-aliased back to the `cache` input, so the cache is recurrent
/// state carried across runs. Returns the cache contents after both steps.
///
/// This is the path the device-resident-state change touches: on the GPU the
/// cache is a program output aliased in place to an input, so `execute` keeps it
/// device-resident across steps instead of flushing it each run, and reading it
/// back below performs an explicit D2H copy into a host mirror.
fn runKvCacheSteps(ctx: *api.Context, dev: api.DeviceSelector) ![8]f32 {
    var bld = api.Builder.init(ctx);
    defer bld.deinit();

    const Cache = try bld.name(try bld.input(.f32, &[_]usize{ 1, 4, 1, 2 }), "cache");
    const New = try bld.name(try bld.input(.f32, &[_]usize{ 1, 1, 1, 2 }), "new_kv");
    const End = try bld.name(try bld.input(.i32, &[_]usize{1}), "end");
    const Out = try bld.name(try bld.sequenceAppend(Cache, New, End), "cache_out");

    var model = try ctx.compileOn(dev, &bld, &[_]api.TensorRef{Out}, .{
        .output_aliases = &[_]api.OutputAlias{.{ .input = Cache, .output = Out }},
    });
    defer model.deinit();

    // Seed the cache once with zeros; the io-alias then carries it across runs.
    const cache0 = try ctx.fromF32(&[_]usize{ 1, 4, 1, 2 }, &@as([8]f32, @splat(0)));
    try model.bindInput("cache", cache0);

    const steps = [_]struct { end: i32, kv: [2]f32 }{
        .{ .end = 0, .kv = .{ 10.0, 11.0 } }, // -> cache[0]
        .{ .end = 1, .kv = .{ 20.0, 21.0 } }, // -> cache[1]
    };
    for (steps) |st| {
        const kv_t = try ctx.fromArray([1][1][1][2]f32{.{.{.{ st.kv[0], st.kv[1] }}}});
        const end_t = try ctx.fromArray([1]i32{st.end});
        try model.bindInput("new_kv", kv_t);
        try model.bindInput("end", end_t);
        try model.run();
    }

    // The cache slot is device-exclusive on a (discrete) GPU and host-backed on
    // CPU — i.e. migration actually happened, not a silent staged no-op.
    const on_gpu = std.meta.activeTag(dev) != .cpu;
    try std.testing.expectEqual(on_gpu, model.stateInputOnDevice("cache"));

    const out_t = try model.outputTensorAt(0);
    var vals: [8]f32 = undefined;
    try out_t.read(&vals);
    return vals;
}

test "api: gpu kv-cache decode carries device-resident state (matches cpu)" {
    const alloc = std.testing.allocator;

    var ctx = api.Context.init(alloc, .{ .gpus = &.{.{ .power = .high }} }) catch |e| switch (e) {
        error.BackendUnavailable => return error.SkipZigTest,
        else => return e,
    };
    defer ctx.deinit();

    const cpu = try runKvCacheSteps(&ctx, .cpu);
    const gpu = try runKvCacheSteps(&ctx, .{ .gpu = 0 });

    // Two tokens appended at positions 0 and 1; the rest of the cache stays zero.
    const want = [8]f32{ 10, 11, 20, 21, 0, 0, 0, 0 };
    try std.testing.expectEqualSlices(f32, &want, &cpu);
    try std.testing.expectEqualSlices(f32, &want, &gpu);
}

/// Exercise a recurrent alias whose output cannot reuse the input slot. Unlike
/// SequenceAppend, `state + delta` produces a distinct output tensor, so every
/// run must copy that result back into the device-resident state slot.
fn runNonInPlaceStateSteps(ctx: *api.Context, dev: api.DeviceSelector) ![4]f32 {
    var bld = api.Builder.init(ctx);
    defer bld.deinit();

    const State = try bld.name(try bld.input(.f32, &[_]usize{4}), "state");
    const Delta = try bld.name(try bld.input(.f32, &[_]usize{4}), "delta");
    const Next = try bld.name(try bld.add(State, Delta), "next_state");

    var model = try ctx.compileOn(dev, &bld, &[_]api.TensorRef{Next}, .{
        .output_aliases = &[_]api.OutputAlias{.{ .input = State, .output = Next }},
        .input_roles = &[_]api.InputRoleDecl{.{ .input = State, .kind = .state }},
    });
    defer model.deinit();

    // State is deliberately unbound: the runtime allocates and zeroes it once.
    const delta = try ctx.fromArray([4]f32{ 1, 2, 3, 4 });
    try model.bindInput("delta", delta);
    try model.run();
    try model.run();

    const on_gpu = std.meta.activeTag(dev) != .cpu;
    try std.testing.expectEqual(on_gpu, model.stateInputOnDevice("state"));

    const out = try model.outputTensorAt(0);
    var vals: [4]f32 = undefined;
    try out.read(&vals);
    return vals;
}

test "api: non-in-place recurrent state carries on gpu without host-only copy" {
    const alloc = std.testing.allocator;

    var ctx = api.Context.init(alloc, .{ .gpus = &.{.{ .power = .high }} }) catch |e| switch (e) {
        error.BackendUnavailable => return error.SkipZigTest,
        else => return e,
    };
    defer ctx.deinit();

    const cpu = try runNonInPlaceStateSteps(&ctx, .cpu);
    const gpu = try runNonInPlaceStateSteps(&ctx, .{ .gpu = 0 });
    const want = [4]f32{ 2, 4, 6, 8 };
    try std.testing.expectEqualSlices(f32, &want, &cpu);
    try std.testing.expectEqualSlices(f32, &want, &gpu);
}

/// Grow-on-demand decode on `dev`: cache starts at capacity 2 and the runtime
/// grows its slot as appends cross capacity (up to 8). Returns the final cache.
fn runGrowableDecode(ctx: *api.Context, dev: api.DeviceSelector) ![8]f32 {
    var bld = api.Builder.init(ctx);
    defer bld.deinit();

    const Cache = try bld.name(try bld.input(.f32, &[_]usize{ 1, 2, 1, 1 }), "cache");
    const New = try bld.name(try bld.input(.f32, &[_]usize{ 1, 1, 1, 1 }), "new");
    const End = try bld.name(try bld.input(.i32, &[_]usize{1}), "end");
    const Out = try bld.name(try bld.sequenceAppend(Cache, New, End), "cache_out");

    var model = try ctx.compileOn(dev, &bld, &[_]api.TensorRef{Out}, .{
        .output_aliases = &[_]api.OutputAlias{.{ .input = Cache, .output = Out }},
    });
    defer model.deinit();

    try model.setStateInputPolicy("cache", .{ .growable = .{
        .initial_capacity_tokens = 2,
        .growth_numerator = 2,
        .growth_denominator = 1,
        .max_capacity_tokens = 8,
    } });

    const cache0 = try ctx.fromF32(&[_]usize{ 1, 2, 1, 1 }, &@as([2]f32, @splat(0)));
    try model.bindInput("cache", cache0);

    var pos: i32 = 0;
    while (pos < 5) : (pos += 1) {
        const new_t = try ctx.fromArray([1][1][1][1]f32{.{.{.{@as(f32, @floatFromInt(pos + 1))}}}});
        const end_t = try ctx.fromArray([1]i32{pos});
        try model.bindInput("new", new_t);
        try model.bindInput("end", end_t);
        try model.run();
    }

    // Still device-exclusive after growth (the round-trip re-migrates to the device).
    const on_gpu = std.meta.activeTag(dev) != .cpu;
    try std.testing.expectEqual(on_gpu, model.stateInputOnDevice("cache"));

    const out = try model.outputTensorAt(0);
    var vals: [8]f32 = undefined;
    try out.read(&vals);
    return vals;
}

test "api: growable state grows device-resident on gpu (matches cpu)" {
    const alloc = std.testing.allocator;

    var ctx = api.Context.init(alloc, .{ .gpus = &.{.{ .power = .high }} }) catch |e| switch (e) {
        error.BackendUnavailable => return error.SkipZigTest,
        else => return e,
    };
    defer ctx.deinit();

    const cpu = try runGrowableDecode(&ctx, .cpu);
    const gpu = try runGrowableDecode(&ctx, .{ .gpu = 0 });

    const want = [8]f32{ 1, 2, 3, 4, 5, 0, 0, 0 };
    try std.testing.expectEqualSlices(f32, &want, &cpu);
    try std.testing.expectEqualSlices(f32, &want, &gpu);
}

test "api: device growth is frame-safe with an in-frame cache read" {
    // A cache-reading op (`pre = cache + cache`) is recorded into the SAME frame as
    // the growing `sequenceAppend`. Real decode has exactly this shape (attention
    // reads the cache, then the step appends). If growth frees the old device buffer
    // while `pre`'s recorded dispatch still references it, submitting the frame faults.
    // Running clean across the two growth steps (pos 2 and pos 4) exercises that path.
    const alloc = std.testing.allocator;

    var ctx = api.Context.init(alloc, .{ .gpus = &.{.{ .power = .high }} }) catch |e| switch (e) {
        error.BackendUnavailable => return error.SkipZigTest,
        else => return e,
    };
    defer ctx.deinit();

    var bld = api.Builder.init(&ctx);
    defer bld.deinit();

    const Cache = try bld.name(try bld.input(.f32, &[_]usize{ 1, 2, 1, 1 }), "cache");
    const New = try bld.name(try bld.input(.f32, &[_]usize{ 1, 1, 1, 1 }), "new");
    const End = try bld.name(try bld.input(.i32, &[_]usize{1}), "end");
    const Pre = try bld.name(try bld.add(Cache, Cache), "pre"); // reads the cache buffer in-frame
    const Out = try bld.name(try bld.sequenceAppend(Cache, New, End), "cache_out");

    var model = try ctx.compileOn(.{ .gpu = 0 }, &bld, &[_]api.TensorRef{ Out, Pre }, .{
        .output_aliases = &[_]api.OutputAlias{.{ .input = Cache, .output = Out }},
    });
    defer model.deinit();

    try model.setStateInputPolicy("cache", .{ .growable = .{
        .initial_capacity_tokens = 2,
        .growth_numerator = 2,
        .growth_denominator = 1,
        .max_capacity_tokens = 8,
    } });

    const cache0 = try ctx.fromF32(&[_]usize{ 1, 2, 1, 1 }, &@as([2]f32, @splat(0)));
    try model.bindInput("cache", cache0);

    var pos: i32 = 0;
    while (pos < 5) : (pos += 1) {
        const new_t = try ctx.fromArray([1][1][1][1]f32{.{.{.{@as(f32, @floatFromInt(pos + 1))}}}});
        const end_t = try ctx.fromArray([1]i32{pos});
        try model.bindInput("new", new_t);
        try model.bindInput("end", end_t);
        try model.run();
    }

    const out = try model.outputTensorAt(0); // cache_out
    var vals: [8]f32 = undefined;
    try out.read(&vals);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 3, 4, 5, 0, 0, 0 }, &vals);
}

test "api: tensor.to round-trips cpu -> gpu -> cpu (move semantics)" {
    const alloc = std.testing.allocator;

    var ctx = api.Context.init(alloc, .{ .gpus = &.{.{ .power = .high }} }) catch |e| switch (e) {
        error.BackendUnavailable => return error.SkipZigTest,
        else => return e,
    };
    defer ctx.deinit();

    const n: usize = 64;
    var vals: [n]f32 = undefined;
    for (&vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 30)) * 0.125;

    var t = try ctx.fromF32(&[_]usize{n}, &vals);

    // Migrate onto the GPU: the host copy is freed and bytes live in device buffers.
    try t.to(.{ .gpu = 0 });
    // Host reads are not allowed while device-resident.
    var scratch: [n]f32 = undefined;
    try std.testing.expectError(error.InvalidArgument, t.readF32(&scratch));

    // Migrate back and confirm the bytes survived the H2D + D2H round-trip.
    try t.to(.cpu);
    const got = try t.readF32Alloc(alloc);
    defer alloc.free(got);
    try std.testing.expectEqualSlices(f32, &vals, got);
}

/// A rolling KV cache driven through a wide prefill and a few decode steps on
/// `dev`, returning every attention output. Retained history is 7 and the cache
/// starts at 8 rows, so the 20-token prefill forces growth-with-rehash and the
/// decode steps then wrap repeatedly.
fn runRollingCacheSchedule(
    alloc: std.mem.Allocator,
    ctx: *api.Context,
    file: std.Io.File,
    dev: api.DeviceSelector,
    out: *std.ArrayList(f32),
) !void {
    const head_dim: usize = 4;
    var model = try ctx.loadModel(file, .{ .device = dev });
    defer model.deinit();

    const chunks = [_]usize{ 20, 1, 1, 1, 1, 1, 5, 1, 1 };
    var pos: usize = 0;
    for (chunks) |width| {
        const tokens = try alloc.alloc(i32, width);
        defer alloc.free(tokens);
        const qkv = try alloc.alloc(f32, width * head_dim);
        defer alloc.free(qkv);
        for (0..width) |s| {
            tokens[s] = @intCast((pos + s) % 97);
            for (0..head_dim) |d| {
                qkv[s * head_dim + d] = @as(f32, @floatFromInt(((pos + s) * 7 + d * 3) % 23)) * 0.05 - 0.5;
            }
        }
        try model.bindInput("tokens", try ctx.from(&.{ 1, width }, tokens));
        try model.bindInput("q", try ctx.fromF32(&.{ 1, width, 1, head_dim }, qkv));
        try model.bindInput("kv", try ctx.fromF32(&.{ 1, width, 1, head_dim }, qkv));
        try model.run();

        const got = try model.outputTensor("out");
        const values = try alloc.alloc(f32, width * head_dim);
        defer alloc.free(values);
        try got.read(values);
        try out.appendSlice(alloc, values);
        pos += width;
    }
}

test "api: a rolling kv cache decodes identically on gpu and cpu" {
    const alloc = std.testing.allocator;
    const head_dim: usize = 4;

    var ctx = api.Context.init(alloc, .{ .gpus = &.{.{ .power = .high }} }) catch |e| switch (e) {
        error.BackendUnavailable => return error.SkipZigTest,
        else => return e,
    };
    defer ctx.deinit();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const file = try tmp.dir.createFile(std.testing.io, "rolling.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);

    {
        var bld = api.Builder.init(&ctx);
        defer bld.deinit();
        const Tokens = try bld.name(try bld.input(.i32, &.{ 1, 1 }), "tokens");
        try bld.symbolicDim(Tokens, 1, "S");
        const Q = try bld.name(try bld.input(.f32, &.{ 1, 1, 1, head_dim }), "q");
        try bld.symbolicDim(Q, 1, "S");
        const Kv = try bld.name(try bld.input(.f32, &.{ 1, 1, 1, head_dim }), "kv");
        try bld.symbolicDim(Kv, 1, "S");
        const Positions = try bld.name(try bld.input(.i32, &.{ 1, 1 }), "positions");
        try bld.symbolicDim(Positions, 1, "S");
        const WriteIndex = try bld.name(try bld.input(.i32, &.{1}), "cache_write_index");
        const VisibleEnd = try bld.name(try bld.input(.i32, &.{1}), "cache_visible_end");
        const KCache = try bld.name(try bld.input(.f32, &.{ 1, 2, 1, head_dim }), "k_cache");
        try bld.symbolicDim(KCache, 1, "L");
        const VCache = try bld.name(try bld.input(.f32, &.{ 1, 2, 1, head_dim }), "v_cache");
        try bld.symbolicDim(VCache, 1, "L");

        const KNext = try bld.name(try bld.sequenceAppend(KCache, try bld.copy(Kv), WriteIndex), "k_next");
        const VNext = try bld.name(try bld.sequenceAppend(VCache, try bld.relu(Kv), WriteIndex), "v_next");
        const Out = try bld.name(
            try bld.attention(Q, KNext, VNext, Positions, VisibleEnd, 0.5, .sliding(7, 0), 0.0),
            "out",
        );

        const cache_role: api.InputRoleDecl = .{
            .input = KCache,
            .kind = .sequence_cache,
            .axis = 1,
            .capacity_symbol = "L",
            .retained_history_tokens = 7,
        };
        var v_role = cache_role;
        v_role.input = VCache;
        try ctx.exportModel(file, &bld, &.{
            .{ .name = "out", .tensor = Out },
            .{ .name = "k_next", .tensor = KNext },
            .{ .name = "v_next", .tensor = VNext },
        }, .{
            .output_aliases = &.{
                .{ .input = KCache, .output = KNext },
                .{ .input = VCache, .output = VNext },
            },
            .input_roles = &.{
                .{ .input = Tokens, .kind = .tokens, .axis = 1 },
                .{ .input = Positions, .kind = .positions, .axis = 1 },
                .{ .input = WriteIndex, .kind = .cache_write_index },
                .{ .input = VisibleEnd, .kind = .cache_visible_end },
                cache_role,
                v_role,
            },
        });
    }

    var cpu: std.ArrayList(f32) = .empty;
    defer cpu.deinit(alloc);
    var gpu: std.ArrayList(f32) = .empty;
    defer gpu.deinit(alloc);
    try runRollingCacheSchedule(alloc, &ctx, file, .cpu, &cpu);
    try runRollingCacheSchedule(alloc, &ctx, file, .{ .gpu = 0 }, &gpu);

    try std.testing.expectEqual(cpu.items.len, gpu.items.len);
    for (cpu.items, gpu.items) |c, g| try std.testing.expectApproxEqAbs(c, g, 1e-5);
}

/// Top-k of one wide row on `dev` — the decode shape, where the row is a whole
/// vocabulary and only a handful of entries are wanted.
const TopKOut = struct { v: [8]f32, i: [8]i32 };

/// Builds its own param: a tensor is moved to the device it is compiled for, so a
/// single one cannot be shared between the cpu and gpu runs being compared.
fn runTopK(ctx: *api.Context, dev: api.DeviceSelector, vals: []const f32, k: usize, largest: bool) !TopKOut {
    var bld = api.Builder.init(ctx);
    defer bld.deinit();
    const src = try ctx.fromF32(&[_]usize{ 1, vals.len }, vals);
    const top = try bld.topk(try bld.param(src), k, -1, largest);
    var model = try ctx.compileOn(dev, &bld, &[_]api.TensorRef{ top.values, top.indices }, .{});
    defer model.deinit();
    try model.run();

    var out: TopKOut = .{ .v = @splat(0), .i = @splat(0) };
    try (try model.outputTensorAt(0)).read(out.v[0..k]);
    try (try model.outputTensorAt(1)).read(out.i[0..k]);
    return out;
}

test "api: topk matches between gpu and cpu, values and indices" {
    const alloc = std.testing.allocator;

    var ctx = api.Context.init(alloc, .{ .gpus = &.{.{ .power = .high }} }) catch |e| switch (e) {
        error.BackendUnavailable => return error.SkipZigTest,
        else => return e,
    };
    defer ctx.deinit();

    // Wider than the split path's 4096-column segment, so this exercises
    // topk_partial + topk_finish rather than the single-workgroup kernel, with
    // deliberate duplicate maxima so the lowest-index tie-break has to agree
    // across devices rather than just the values.
    const n: usize = 40000;
    const vals = try alloc.alloc(f32, n);
    defer alloc.free(vals);
    for (vals, 0..) |*v, i| {
        const x: f32 = @floatFromInt((i * 7919) % 1000);
        v.* = x * 0.001;
    }
    vals[100] = 2.0;
    vals[9000] = 2.0; // ties with vals[100], and in a different 4096-column segment
    vals[42] = 3.0;

    for ([_]bool{ true, false }) |largest| {
        const cpu = try runTopK(&ctx, .cpu, vals, 6, largest);
        const gpu = try runTopK(&ctx, .{ .gpu = 0 }, vals, 6, largest);
        try std.testing.expectEqualSlices(f32, cpu.v[0..6], gpu.v[0..6]);
        try std.testing.expectEqualSlices(i32, cpu.i[0..6], gpu.i[0..6]);
    }

    // The known extremes, so this pins actual values rather than only agreement.
    const top = try runTopK(&ctx, .{ .gpu = 0 }, vals, 3, true);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 3.0, 2.0, 2.0 }, top.v[0..3]);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 42, 100, 9000 }, top.i[0..3]);
}
