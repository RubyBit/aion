const std = @import("std");

const storage = @import("storage.zig");
const types = @import("../backend/types.zig");
const backend_utils = @import("../backend/utils.zig");
const simd = @import("../backend/cpu/kernels/simd.zig");

const DType = types.DType;

test "storage: scalar f32 roundtrip pack<->tiles" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    // A small odd shape to exercise boundary tiles.
    const rows: usize = 5;
    const cols: usize = 7;

    var tt: storage.TiledTensor = undefined;
    try tt.init(
        allocator,
        .f32,
        &[_]usize{ rows, cols },
        &[_]usize{ 2, 3 },
        .{ .tile_alignment = 64 },
    );
    defer tt.deinit();

    const total_elems: usize = rows * cols;
    const packed_vals: []f32 = try allocator.alloc(f32, total_elems);
    defer allocator.free(packed_vals);

    // Fill with a deterministic pattern.
    for (0..rows) |i| {
        for (0..cols) |j| {
            packed_vals[i * cols + j] = @as(f32, @floatFromInt(i * 1000 + j));
        }
    }

    const packed_bytes: []const u8 = std.mem.sliceAsBytes(packed_vals);
    try tt.writeFromPackedScalar(packed_bytes);

    // Verify each tile is a valid packed scalar view, and its contents match expectations.
    var ti0: usize = 0;
    while (ti0 < tt.tile_counts[0]) : (ti0 += 1) {
        var ti1: usize = 0;
        while (ti1 < tt.tile_counts[1]) : (ti1 += 1) {
            const tv = try tt.acquireTileConst(ti0, ti1);
            const v = tv.bufferView();
            try backend_utils.requirePackedScalar(v);

            const tile_rows: usize = v.layout.shape[0];
            const tile_cols: usize = v.layout.shape[1];

            const tile_vals: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, v.bytes);
            try std.testing.expect(tile_vals.len >= tile_rows * tile_cols);

            const row0: usize = ti0 * tt.tile_shape[0];
            const col0: usize = ti1 * tt.tile_shape[1];

            for (0..tile_rows) |r| {
                for (0..tile_cols) |c| {
                    const gi: usize = row0 + r;
                    const gj: usize = col0 + c;
                    try std.testing.expectEqual(packed_vals[gi * cols + gj], tile_vals[r * tile_cols + c]);
                }
            }
        }
    }

    // Unpack and compare.
    const out: []f32 = try allocator.alloc(f32, total_elems);
    defer allocator.free(out);
    @memset(out, 0);

    try tt.readToPackedScalar(std.mem.sliceAsBytes(out));
    try std.testing.expectEqualSlices(f32, packed_vals, out);
}

test "storage: quant q8_0 roundtrip pack<->tiles (bit exact)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    // Treat this as a [K,N] weight matrix with K multiple of 32.
    const k: usize = 96;
    const n: usize = 5;

    var tt: storage.TiledTensor = undefined;
    try tt.init(
        allocator,
        .q8_0,
        &[_]usize{ k, n },
        &[_]usize{ 64, 3 },
        .{ .tile_alignment = 64 },
    );
    defer tt.deinit();

    const total_elems: usize = k * n;
    const total_bytes: usize = try backend_utils.requiredBytesForElems(DType.q8_0, total_elems);

    const packed_bytes_q: []u8 = try allocator.alloc(u8, total_bytes);
    defer allocator.free(packed_bytes_q);

    // Fill with deterministic data (not necessarily a meaningful quant tensor, but valid size).
    for (packed_bytes_q, 0..) |*b, i| b.* = @intCast(i % 251);

    try tt.writeFromPackedQuant(packed_bytes_q);

    const out: []u8 = try allocator.alloc(u8, total_bytes);
    defer allocator.free(out);
    @memset(out, 0);

    try tt.readToPackedQuant(out);
    try std.testing.expectEqualSlices(u8, packed_bytes_q, out);
}

test "storage: scalar f32 roundtrip rank-3 pack<->tiles" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const d0: usize = 3;
    const d1: usize = 4;
    const d2: usize = 5;

    var tt: storage.TiledTensor = undefined;
    try tt.init(
        allocator,
        .f32,
        &[_]usize{ d0, d1, d2 },
        &[_]usize{ 2, 3, 2 },
        .{ .tile_alignment = 64 },
    );
    defer tt.deinit();

    const total_elems: usize = d0 * d1 * d2;
    const packed_vals: []f32 = try allocator.alloc(f32, total_elems);
    defer allocator.free(packed_vals);

    // Deterministic pattern: linear index + offset.
    for (packed_vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 7)) * 0.25;

    try tt.writeFromPackedScalar(std.mem.sliceAsBytes(packed_vals));

    const out: []f32 = try allocator.alloc(f32, total_elems);
    defer allocator.free(out);
    @memset(out, 0);

    try tt.readToPackedScalar(std.mem.sliceAsBytes(out));
    try std.testing.expectEqualSlices(f32, packed_vals, out);
}
