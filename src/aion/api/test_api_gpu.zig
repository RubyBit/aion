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
    var bld = api.Builder.init(ctx.allocator);
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

/// Run a 2-step KV-cache "decode" on `dev`: an in-place `sequenceAppend` whose
/// output is io-aliased back to the `cache` input, so the cache is recurrent
/// state carried across runs. Returns the cache contents after both steps.
///
/// This is the path the device-resident-state change touches: on the GPU the
/// cache is a program output aliased in place to an input, so `execute` keeps it
/// device-resident across steps instead of flushing it each run, and reading it
/// back below goes through the on-demand `syncToHost`.
fn runKvCacheSteps(ctx: *api.Context, dev: api.DeviceSelector) ![8]f32 {
    var bld = api.Builder.init(ctx.allocator);
    defer bld.deinit();

    const Cache = try bld.name(try bld.input(.f32, &[_]usize{ 1, 1, 4, 2 }), "cache");
    const New = try bld.name(try bld.input(.f32, &[_]usize{ 1, 1, 1, 2 }), "new_kv");
    const End = try bld.name(try bld.input(.i32, &[_]usize{1}), "end");
    const Out = try bld.name(try bld.sequenceAppend(Cache, New, End), "cache_out");

    var model = try ctx.compileOn(dev, &bld, &[_]api.TensorRef{Out}, .{
        .output_aliases = &[_]api.OutputAlias{.{ .input = Cache, .output = Out }},
    });
    defer model.deinit();

    // Seed the cache once with zeros; the io-alias then carries it across runs.
    const cache0 = try ctx.fromArray([1][1][4][2]f32{.{.{ .{ 0, 0 }, .{ 0, 0 }, .{ 0, 0 }, .{ 0, 0 } }}});
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
    var bld = api.Builder.init(ctx.allocator);
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
    var bld = api.Builder.init(ctx.allocator);
    defer bld.deinit();

    const Cache = try bld.name(try bld.input(.f32, &[_]usize{ 1, 1, 2, 1 }), "cache");
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

    const cache0 = try ctx.fromArray([1][1][2][1]f32{.{.{ .{0}, .{0} }}});
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

    var bld = api.Builder.init(ctx.allocator);
    defer bld.deinit();

    const Cache = try bld.name(try bld.input(.f32, &[_]usize{ 1, 1, 2, 1 }), "cache");
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

    const cache0 = try ctx.fromArray([1][1][2][1]f32{.{.{ .{0}, .{0} }}});
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
