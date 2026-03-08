const std = @import("std");

const aion_file = @import("aion_file.zig");
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

test "storage file: write/parse tiled tensors roundtrip" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var t0: storage.TiledTensor = undefined;
    try t0.init(
        allocator,
        .f32,
        &[_]usize{ 5, 7 },
        &[_]usize{ 2, 3 },
        .{ .tile_alignment = 64 },
    );
    defer t0.deinit();

    const vals0: []f32 = try allocator.alloc(f32, 5 * 7);
    defer allocator.free(vals0);
    for (vals0, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 11)) * 0.5;
    try t0.writeFromPackedScalar(std.mem.sliceAsBytes(vals0));

    var t1: storage.TiledTensor = undefined;
    try t1.init(
        allocator,
        .q8_0,
        &[_]usize{ 96, 5 },
        &[_]usize{ 64, 3 },
        .{ .tile_alignment = 64 },
    );
    defer t1.deinit();

    const q_bytes_len: usize = try backend_utils.requiredBytesForElems(types.DType.q8_0, 96 * 5);
    const q_vals: []u8 = try allocator.alloc(u8, q_bytes_len);
    defer allocator.free(q_vals);
    for (q_vals, 0..) |*b, i| b.* = @intCast((i * 17 + 3) % 251);
    try t1.writeFromPackedQuant(q_vals);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    {
        const file = try tmp.dir.createFile("weights.aion", .{ .read = true, .truncate = true });
        defer file.close();

        const metadata = [_]aion_file.MetadataSource{
            aion_file.MetadataSource.string("arch", "tiny-test"),
        };
        const tensors = [_]aion_file.TensorSource{
            .{ .name = "dense.weight", .tensor = &t0 },
            .{ .name = "dense.weight_q8", .tensor = &t1 },
        };
        try aion_file.writeFile(file, metadata[0..], tensors[0..], .{});

        const bytes: []align(64) u8 = try aion_file.readAlloc(allocator, file);
        defer allocator.free(bytes);

        const view: aion_file.View = try aion_file.parse(bytes);
        try std.testing.expectEqual(@as(u64, 1), view.header.kv_count);
        try std.testing.expectEqual(@as(u64, 2), view.header.tensor_count);

        const kv0 = try view.metadata(0);
        try std.testing.expectEqual(aion_file.ValueType.string, kv0.value_type);
        try std.testing.expectEqualStrings("arch", kv0.key);
        try std.testing.expectEqualStrings("tiny-test", kv0.value);

        const d0_opt = try view.findTensor("dense.weight");
        try std.testing.expect(d0_opt != null);
        const d0 = d0_opt.?;
        try std.testing.expectEqual(types.DType.f32, d0.dtype);
        try std.testing.expectEqual(@as(u8, 2), d0.rank);
        try std.testing.expectEqual(@as(u64, @intCast(t0.tile_offsets.len)), d0.tile_count);
        try view.validateTensorCrc32(d0);
        const d0_data = try view.tensorDataBytes(d0);
        try std.testing.expectEqualSlices(u8, t0.data, d0_data);
        for (t0.tile_offsets, t0.tile_lens, 0..) |off, len, i| {
            try std.testing.expectEqual(@as(u64, @intCast(off)), try view.tileOffsetAt(d0, i));
            try std.testing.expectEqual(@as(u64, @intCast(len)), try view.tileLenAt(d0, i));
            const tile_bytes = try view.tileBytes(d0, i);
            try std.testing.expectEqualSlices(u8, t0.data[off .. off + len], tile_bytes);
        }

        const d1_opt = try view.findTensor("dense.weight_q8");
        try std.testing.expect(d1_opt != null);
        const d1 = d1_opt.?;
        try std.testing.expectEqual(types.DType.q8_0, d1.dtype);
        try std.testing.expectEqual(@as(u8, 2), d1.rank);
        try view.validateTensorCrc32(d1);
        const d1_data = try view.tensorDataBytes(d1);
        try std.testing.expectEqualSlices(u8, t1.data, d1_data);
    }
}

test "storage file: mapReadOnly parses without heap copy" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var tt: storage.TiledTensor = undefined;
    try tt.init(
        allocator,
        .f32,
        &[_]usize{ 4, 4 },
        &[_]usize{ 2, 2 },
        .{ .tile_alignment = 64 },
    );
    defer tt.deinit();

    const vals: [16]f32 = .{
        1.0,  2.0,  3.0,  4.0,
        5.0,  6.0,  7.0,  8.0,
        9.0,  10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    };
    try tt.writeFromPackedScalar(std.mem.sliceAsBytes(vals[0..]));

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try tmp.dir.createFile("mapped.aion", .{ .read = true, .truncate = true });
    defer file.close();

    const tensors = [_]aion_file.TensorSource{.{ .name = "mapped.weight", .tensor = &tt }};
    try aion_file.writeFile(file, &.{}, tensors[0..], .{});

    var mapped = try aion_file.MappedFile.mapReadOnly(file);
    defer mapped.deinit();

    const file_size: usize = @intCast(try file.getEndPos());
    const bytes = try mapped.logicalBytes(file_size);
    const view = try aion_file.parse(bytes);
    const desc = (try view.findTensor("mapped.weight")).?;
    try view.validateTensorCrc32(desc);
    const data = try view.tensorDataBytes(desc);
    try std.testing.expectEqualSlices(u8, tt.data, data);
}
