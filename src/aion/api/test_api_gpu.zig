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
