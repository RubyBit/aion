const std = @import("std");

const package_file = @import("aion_file.zig");
const manager_mod = @import("manager.zig");
const storage = @import("storage.zig");
const types = @import("../backend/types.zig");
const backend_utils = @import("../backend/utils.zig");
const simd = @import("../backend/cpu/kernels/simd.zig");

const DType = types.DType;

fn createTestFile(dir: std.Io.Dir, sub_path: []const u8, flags: std.Io.File.CreateFlags) !std.Io.File {
    return try dir.createFile(std.testing.io, sub_path, flags);
}

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

test "storage file: model package write/parse roundtrip" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const w_vals: [6]f32 = .{ 1.0, -2.0, 0.5, 3.0, 4.0, -1.5 };

    var pkg = package_file.Package{
        .allocator = allocator,
        .initializers = try allocator.alloc(package_file.Initializer, 1),
        .values = try allocator.alloc(package_file.ValueRecord, 3),
        .nodes = try allocator.alloc(package_file.NodeRecord, 1),
        .inputs = try allocator.alloc(package_file.NamedValue, 1),
        .outputs = try allocator.alloc(package_file.NamedValue, 1),
        .dim_symbols = try allocator.alloc(package_file.DimSymbol, 0),
        .dim_exprs = try allocator.alloc(package_file.DimExpr, 0),
        .metadata = try allocator.alloc(package_file.MetadataEntry, 1),
        .debug_names = try allocator.alloc(package_file.DebugName, 1),
        .io_aliases = try allocator.alloc(package_file.IoAlias, 1),
    };
    defer pkg.deinit();

    pkg.initializers[0] = .{
        .encoding = .{ .plain = .f32 },
        .data = try allocator.dupe(u8, std.mem.sliceAsBytes(w_vals[0..])),
    };

    pkg.values[0] = .{
        .dtype = .f32,
        .rank = 2,
        .source = .public_input,
        .shape_terms = try package_file.makeConstantShapeTerms(allocator, &[_]usize{ 1, 2 }),
    };
    pkg.values[1] = .{
        .dtype = .f32,
        .rank = 2,
        .source = .initializer,
        .shape_terms = try package_file.makeConstantShapeTerms(allocator, &[_]usize{ 2, 3 }),
        .initializer_index = 0,
    };
    pkg.values[2] = .{
        .dtype = .f32,
        .rank = 2,
        .source = .produced,
        .shape_terms = try allocator.alloc(package_file.ShapeTerm, 0),
    };

    pkg.nodes[0] = .{
        .inputs = try allocator.dupe(u32, &[_]u32{ 0, 1 }),
        .output = 2,
        .op = .{ .MatMul = .{ .alpha = 1.0, .beta = 0.0 } },
    };
    pkg.inputs[0] = .{ .name = try allocator.dupe(u8, "x"), .value = 0 };
    pkg.outputs[0] = .{ .name = try allocator.dupe(u8, "y"), .value = 2 };
    pkg.metadata[0] = .{
        .key = try allocator.dupe(u8, "arch"),
        .value = try allocator.dupe(u8, "tiny-matmul"),
    };
    pkg.debug_names[0] = .{
        .value = 2,
        .name = try allocator.dupe(u8, "matmul.out"),
    };
    pkg.io_aliases[0] = .{ .input = 0, .output = 0 };

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try createTestFile(tmp.dir, "model.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);
    try package_file.writeFile(file, &pkg);

    const bytes = try package_file.readAlloc(allocator, file);
    defer allocator.free(bytes);
    var parsed = try package_file.parse(allocator, bytes);
    defer parsed.deinit();

    try std.testing.expectEqual(@as(usize, 1), parsed.initializers.len);
    try std.testing.expectEqual(@as(usize, 3), parsed.values.len);
    try std.testing.expectEqual(@as(usize, 1), parsed.nodes.len);
    try std.testing.expectEqualStrings("x", parsed.inputs[0].name);
    try std.testing.expectEqualStrings("y", parsed.outputs[0].name);
    try std.testing.expectEqualStrings("arch", parsed.metadata[0].key);
    try std.testing.expectEqualStrings("tiny-matmul", parsed.metadata[0].value);
    try std.testing.expectEqual(@as(usize, 1), parsed.io_aliases.len);
    try std.testing.expectEqual(@as(u32, 0), parsed.io_aliases[0].input);
    try std.testing.expectEqual(@as(u32, 0), parsed.io_aliases[0].output);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(w_vals[0..]), parsed.initializers[0].data);
}

test "storage file: dim expressions evaluate after parse" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var pkg = package_file.Package{
        .allocator = allocator,
        .initializers = try allocator.alloc(package_file.Initializer, 0),
        .values = try allocator.alloc(package_file.ValueRecord, 1),
        .nodes = try allocator.alloc(package_file.NodeRecord, 0),
        .inputs = try allocator.alloc(package_file.NamedValue, 1),
        .outputs = try allocator.alloc(package_file.NamedValue, 1),
        .dim_symbols = try allocator.alloc(package_file.DimSymbol, 1),
        .dim_exprs = try allocator.alloc(package_file.DimExpr, 2),
        .metadata = try allocator.alloc(package_file.MetadataEntry, 0),
        .debug_names = try allocator.alloc(package_file.DebugName, 0),
        .io_aliases = try allocator.alloc(package_file.IoAlias, 0),
    };
    defer pkg.deinit();

    pkg.dim_symbols[0] = .{ .name = try allocator.dupe(u8, "batch") };
    pkg.dim_exprs[0] = .{ .symbol = 0 };
    pkg.dim_exprs[1] = .{ .add = .{ .lhs = .{ .expr = 0 }, .rhs = .{ .constant = 1 } } };
    pkg.values[0] = .{
        .dtype = .f32,
        .rank = 1,
        .source = .public_input,
        .shape_terms = try allocator.dupe(package_file.ShapeTerm, &[_]package_file.ShapeTerm{.{ .expr = 0 }}),
    };
    pkg.inputs[0] = .{ .name = try allocator.dupe(u8, "x"), .value = 0 };
    pkg.outputs[0] = .{ .name = try allocator.dupe(u8, "y"), .value = 0 };

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const file = try createTestFile(tmp.dir, "dyn.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);
    try package_file.writeFile(file, &pkg);

    const bytes = try package_file.readAlloc(allocator, file);
    defer allocator.free(bytes);
    var parsed = try package_file.parse(allocator, bytes);
    defer parsed.deinit();

    const symbol_values = [_]?u64{3};
    try std.testing.expectEqual(@as(u64, 4), try package_file.evaluateShapeTerm(&parsed, .{ .expr = 1 }, symbol_values[0..]));
}

test "storage file: invalid package is rejected" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var pkg = package_file.Package{
        .allocator = allocator,
        .initializers = try allocator.alloc(package_file.Initializer, 0),
        .values = try allocator.alloc(package_file.ValueRecord, 1),
        .nodes = try allocator.alloc(package_file.NodeRecord, 0),
        .inputs = try allocator.alloc(package_file.NamedValue, 1),
        .outputs = try allocator.alloc(package_file.NamedValue, 1),
        .dim_symbols = try allocator.alloc(package_file.DimSymbol, 0),
        .dim_exprs = try allocator.alloc(package_file.DimExpr, 0),
        .metadata = try allocator.alloc(package_file.MetadataEntry, 0),
        .debug_names = try allocator.alloc(package_file.DebugName, 0),
        .io_aliases = try allocator.alloc(package_file.IoAlias, 0),
    };
    defer pkg.deinit();

    pkg.values[0] = .{
        .dtype = .f32,
        .rank = 1,
        .source = .public_input,
        .shape_terms = try package_file.makeConstantShapeTerms(allocator, &[_]usize{4}),
    };
    pkg.inputs[0] = .{ .name = try allocator.dupe(u8, "dup"), .value = 0 };
    pkg.outputs[0] = .{ .name = try allocator.dupe(u8, "dup"), .value = 0 };

    try std.testing.expectError(package_file.PackageError.InvalidFormat, package_file.validate(&pkg));
}
