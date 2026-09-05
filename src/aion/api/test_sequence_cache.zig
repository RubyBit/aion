// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Sequence-cache capacity, as a differential property.
//!
//! Every case here runs the same graph twice: once over a cache sized for the
//! whole context, once over one that only retains what its attention window can
//! see. Retention and physical layout are meant to be invisible to the model, so
//! the two must agree bit for bit. That composition -- a multi-token append into a
//! wrapped cache, then attention over every appended query -- is what unit tests of
//! interval arithmetic and of the ring kernel each miss on their own.

const std = @import("std");

const api = @import("api.zig");
const AttentionWindow = api.Builder.AttentionWindow;

const head_dim: usize = 4;
const scale: f32 = 0.5;

const Spec = struct {
    /// Prior positions the cache must retain. Null retains the whole context.
    history: ?u32 = null,
    /// `LoadModelOptions.cache.capacity_tokens` for full-history caches.
    capacity: usize = 512,
    growable: ?api.CacheGrowth = null,
    /// Append the cache input value itself instead of a produced one. Produced is
    /// the interesting case: the width then has to come from the tokens role.
    direct_kv: bool = false,
};

fn createTestFile(dir: std.Io.Dir, sub_path: []const u8) !std.Io.File {
    return try dir.createFile(std.testing.io, sub_path, .{ .read = true, .truncate = true });
}

/// Deterministic per-(position, channel) data, so two runs that chunk the same
/// sequence differently still feed identical values at identical positions.
fn sample(pos: usize, channel: usize, salt: usize) f32 {
    var h: u64 = @as(u64, pos) *% 0x9E3779B97F4A7C15 +% @as(u64, channel) *% 0xBF58476D1CE4E5B9 +% @as(u64, salt);
    h ^= h >> 31;
    h *%= 0x94D049BB133111EB;
    h ^= h >> 29;
    return @as(f32, @floatFromInt(@as(i32, @intCast(h % 2001)))) * 0.001 - 1.0;
}

fn exportModel(ctx: *api.Context, file: std.Io.File, spec: Spec, window: AttentionWindow) !void {
    var bld = api.Builder.init(ctx);
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

    const NewK = if (spec.direct_kv) Kv else try bld.copy(Kv);
    const NewV = if (spec.direct_kv) Kv else try bld.relu(Kv);
    const KNext = try bld.name(try bld.sequenceAppend(KCache, NewK, WriteIndex), "k_next");
    const VNext = try bld.name(try bld.sequenceAppend(VCache, NewV, WriteIndex), "v_next");
    const Out = try bld.name(try bld.attention(Q, KNext, VNext, Positions, VisibleEnd, scale, window, 0.0), "out");

    const history: u32 = spec.history orelse 0;
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
            .{
                .input = KCache,
                .kind = .sequence_cache,
                .axis = 1,
                .capacity_symbol = "L",
                .allow_growable = spec.history == null,
                .retained_history_tokens = history,
            },
            .{
                .input = VCache,
                .kind = .sequence_cache,
                .axis = 1,
                .capacity_symbol = "L",
                .allow_growable = spec.history == null,
                .retained_history_tokens = history,
            },
        },
    });
}

/// Run `chunks` through a freshly loaded model, collecting every attention output.
fn runSchedule(
    allocator: std.mem.Allocator,
    file: std.Io.File,
    spec: Spec,
    chunks: []const usize,
    out: *std.ArrayList(f32),
) !void {
    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();
    var model = try ctx.loadModel(file, .{ .cache = .{
        .capacity_tokens = if (spec.history == null) spec.capacity else 0,
        .growable = spec.growable,
    } });
    defer model.deinit();

    var pos: usize = 0;
    for (chunks) |width| {
        const tokens = try allocator.alloc(i32, width);
        defer allocator.free(tokens);
        const q = try allocator.alloc(f32, width * head_dim);
        defer allocator.free(q);
        const kv = try allocator.alloc(f32, width * head_dim);
        defer allocator.free(kv);
        for (0..width) |s| {
            tokens[s] = @intCast((pos + s) % 97);
            for (0..head_dim) |d| {
                q[s * head_dim + d] = sample(pos + s, d, 1);
                kv[s * head_dim + d] = sample(pos + s, d, 2);
            }
        }
        try model.bindInput("tokens", try ctx.from(&.{ 1, width }, tokens));
        try model.bindInput("q", try ctx.fromF32(&.{ 1, width, 1, head_dim }, q));
        try model.bindInput("kv", try ctx.fromF32(&.{ 1, width, 1, head_dim }, kv));
        try model.run();

        const got = try model.outputTensor("out");
        const values = try allocator.alloc(f32, width * head_dim);
        defer allocator.free(values);
        try got.read(values);
        try out.appendSlice(allocator, values);
        pos += width;
    }
}

/// Both specs over the same schedule must produce identical attention outputs.
fn expectSchedulesAgree(
    allocator: std.mem.Allocator,
    window: AttentionWindow,
    reference: Spec,
    candidate: Spec,
    chunks: []const usize,
) !void {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var want: std.ArrayList(f32) = .empty;
    defer want.deinit(allocator);
    var got: std.ArrayList(f32) = .empty;
    defer got.deinit(allocator);

    var ctx = try api.Context.initCpu(allocator, .{ .thread_count = 1 });
    defer ctx.deinit();
    const ref_file = try createTestFile(tmp.dir, "reference.aion");
    defer ref_file.close(std.testing.io);
    const cand_file = try createTestFile(tmp.dir, "candidate.aion");
    defer cand_file.close(std.testing.io);
    try exportModel(&ctx, ref_file, reference, window);
    try exportModel(&ctx, cand_file, candidate, window);
    try runSchedule(allocator, ref_file, reference, chunks, &want);
    try runSchedule(allocator, cand_file, candidate, chunks, &got);

    // Not bit-exact on purpose: capacity changes how the attention kernel blocks
    // its key loop, so the same keys sum in a different order. A wrong key set
    // moves a windowed softmax far more than this.
    try std.testing.expectEqual(want.items.len, got.items.len);
    for (want.items, got.items) |w, g| try std.testing.expectApproxEqAbs(w, g, 1e-5);
}

const window_left: u32 = 7;
const window_span: usize = window_left + 1;

test "sequence cache: retention is invisible across every append width" {
    const allocator = std.testing.allocator;
    const window = AttentionWindow.sliding(window_left, 0);
    const reference: Spec = .{};
    const rolling: Spec = .{ .history = window_left };

    // Prefills narrower than, equal to, and wider than the window span, each
    // followed by steps that carry the cache past several wraps.
    const schedules = [_][]const usize{
        &.{ window_span - 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        &.{ window_span, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        &.{ window_span + 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        &.{ 3 * window_span, 1, 1, 1, 1, 1 },
        &.{ 2, 17, 5, window_span, window_span + 1, 2, 1, 1, 33 },
        &.{ 41, 2, 2, 2, 2, 2, 2, 2, 2 },
    };
    for (schedules) |chunks| {
        try expectSchedulesAgree(allocator, window, reference, rolling, chunks);
    }
}

test "sequence cache: retention is invisible for a chunked window" {
    const allocator = std.testing.allocator;
    // A chunked window reaches back `left` from the chunk start, so retention has
    // to cover `left + chunk - 1` -- the bound a converter declares.
    const window = AttentionWindow.chunked(4, 6);
    const reference: Spec = .{};
    const rolling: Spec = .{ .history = 6 + 4 - 1 };
    const schedules = [_][]const usize{
        &.{ 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        &.{ 4, 4, 4, 4, 4, 4, 4, 4 },
        &.{ 13, 3, 7, 1, 1, 5 },
        // Growth after the cache already holds retained rows, so they must be
        // rehashed onto the new modulus rather than merely preserved in place.
        &.{ 4, 4, 30, 4, 1, 1 },
    };
    for (schedules) |chunks| {
        try expectSchedulesAgree(allocator, window, reference, rolling, chunks);
    }
}

test "sequence cache: a growable cache matches a preallocated one" {
    // The prefill is wider than the initial allocation, so capacity has to settle
    // before the graph is specialized: a slot grown afterwards leaves the compiled
    // program describing a shape the store no longer has.
    const allocator = std.testing.allocator;
    const window = AttentionWindow.sliding(window_left, 0);
    const reference: Spec = .{};
    const growable: Spec = .{ .growable = .{
        .initial_capacity_tokens = 2,
        .growth_numerator = 2,
        .growth_denominator = 1,
    } };
    const schedules = [_][]const usize{
        &.{ 17, 1, 1, 1, 1 },
        &.{ 1, 1, 1, 40, 1 },
        &.{ 64, 64 },
    };
    for (schedules) |chunks| {
        try expectSchedulesAgree(allocator, window, reference, growable, chunks);
    }
}

test "sequence cache: retention is invisible when the appended value is an input" {
    const allocator = std.testing.allocator;
    const window = AttentionWindow.sliding(window_left, 0);
    const reference: Spec = .{ .direct_kv = true };
    const rolling: Spec = .{ .history = window_left, .direct_kv = true };
    try expectSchedulesAgree(allocator, window, reference, rolling, &.{ 11, 1, 1, 1, 1, 1, 1, 1, 1, 9, 1 });
}
