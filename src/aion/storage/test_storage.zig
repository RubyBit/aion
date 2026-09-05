// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const package_file = @import("aion_file.zig");
const manager_mod = @import("manager.zig");
const storage = @import("storage.zig");
const dm = @import("../runtime/device_memory.zig");
const tensor_store = @import("../runtime/tensor_store.zig");
const types = @import("../backend/types.zig");
const backend_utils = @import("../backend/utils.zig");
const simd = @import("../backend/cpu/kernels/simd.zig");

const DType = types.DType;

fn createTestFile(dir: std.Io.Dir, sub_path: []const u8, flags: std.Io.Dir.CreateFileOptions) !std.Io.File {
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

test "storage: device-aware copy/zero round-trip via mock device memory" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var mock = dm.MockDeviceMemory.init(allocator);
    defer mock.deinit();

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const shape = [_]usize{ 2, 3 };
    const vals = [_]f32{ 1, 2, 3, 4, 5, 6 };

    // Host source seeded with known values (single tile).
    const src = try sm.createTiledTensor(.f32, &shape, &shape, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(src, std.mem.sliceAsBytes(&vals));

    // Device-exclusive destination (migrated onto the mock device; host bytes freed).
    const dst = try sm.createTiledTensor(.f32, &shape, &shape, .{ .tile_alignment = 64 });
    try sm.moveTensor(dst, .{ .kind = .gpu, .index = 0 }, mock.device(), &shape, 64);
    try std.testing.expect((try sm.tensorDevice(dst)).kind == .gpu);

    // Seed device dst from host src (H2D scatter), then mirror back to a host
    // tensor (D2H gather) — the seed + host-read paths a device-exclusive KV slot uses.
    try sm.copyTensorData(dst, src);
    const mirror = try sm.createTiledTensor(.f32, &shape, &shape, .{ .tile_alignment = 64 });
    try sm.copyTensorData(mirror, dst);

    var got: [6]f32 = undefined;
    try sm.readToPackedScalar(mirror, std.mem.sliceAsBytes(&got));
    try std.testing.expectEqualSlices(f32, &vals, &got);
    try std.testing.expect(mock.h2d_count > 0 and mock.d2h_count > 0);

    // Device-aware zero (resetState path), then mirror again → all zeros.
    try sm.zeroTensorData(dst);
    try sm.copyTensorData(mirror, dst);
    try sm.readToPackedScalar(mirror, std.mem.sliceAsBytes(&got));
    try std.testing.expectEqualSlices(f32, &[_]f32{ 0, 0, 0, 0, 0, 0 }, &got);
}

// A multi-tile host tensor migrated to a device must arrive byte-identical whether
// the target tiling matches its own (the copy-free upload placement uses) or forces
// a re-tile (the gather/scatter path).
test "storage: multi-tile host->device migration preserves bytes, matched and re-tiled" {
    const allocator: std.mem.Allocator = std.testing.allocator;
    const shape = [_]usize{ 4, 6 };
    var vals: [24]f32 = undefined;
    for (&vals, 0..) |*v, i| v.* = @floatFromInt(i);

    for ([_][2]usize{ .{ 2, 6 }, .{ 4, 3 } }) |target_tile| {
        var mock = dm.MockDeviceMemory.init(allocator);
        defer mock.deinit();
        var sm = manager_mod.StorageManager.init(allocator);
        defer sm.deinit();

        const src_tile = [_]usize{ 2, 6 }; // 2 tiles along dim 0
        const t = try sm.createTiledTensor(.f32, &shape, &src_tile, .{ .tile_alignment = 64 });
        try sm.writeFromPackedScalar(t, std.mem.sliceAsBytes(&vals));
        try sm.moveTensor(t, .{ .kind = .gpu, .index = 0 }, mock.device(), &target_tile, 64);
        try std.testing.expect((try sm.tensorDevice(t)).kind == .gpu);
        try std.testing.expect(mock.h2d_count > 0);

        const mirror = try sm.createTiledTensor(.f32, &shape, &shape, .{ .tile_alignment = 64 });
        try sm.copyTensorData(mirror, t);
        var got: [24]f32 = undefined;
        try sm.readToPackedScalar(mirror, std.mem.sliceAsBytes(&got));
        try std.testing.expectEqualSlices(f32, &vals, &got);
    }
}

test "storage: swap carries heterogeneous host and device backings without copies" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var mock = dm.MockDeviceMemory.init(allocator);
    defer mock.deinit();
    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const shape = [_]usize{4};
    const host_vals = [_]f32{ 1, 2, 3, 4 };
    const device_vals = [_]f32{ 5, 6, 7, 8 };
    const host = try sm.createTiledTensor(.f32, &shape, &shape, .{});
    const device = try sm.createTiledTensor(.f32, &shape, &shape, .{});
    try sm.writeFromPackedScalar(host, std.mem.sliceAsBytes(&host_vals));
    try sm.writeFromPackedScalar(device, std.mem.sliceAsBytes(&device_vals));
    try sm.moveTensor(device, .{ .kind = .gpu, .index = 0 }, mock.device(), &shape, 64);

    const h2d_before = mock.h2d_count;
    const d2h_before = mock.d2h_count;
    try sm.tensorStore().swapTensors(host, device);
    try std.testing.expectEqual(h2d_before, mock.h2d_count);
    try std.testing.expectEqual(d2h_before, mock.d2h_count);
    try std.testing.expect((try sm.tensorDevice(host)).kind == .gpu);
    try std.testing.expect((try sm.tensorDevice(device)).kind == .cpu);

    var got_host: [4]f32 = undefined;
    try sm.readToPackedScalar(device, std.mem.sliceAsBytes(&got_host));
    try std.testing.expectEqualSlices(f32, &host_vals, &got_host);

    const mirror = try sm.createTiledTensor(.f32, &shape, &shape, .{});
    try sm.copyTensorData(mirror, host);
    var got_device: [4]f32 = undefined;
    try sm.readToPackedScalar(mirror, std.mem.sliceAsBytes(&got_device));
    try std.testing.expectEqualSlices(f32, &device_vals, &got_device);
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

test "storage: quant q8_0 roundtrip on quant_axis=1 (embedding table layout)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    // [V, D] embedding table with per-row quantization (blocks along axis 1).
    const v: usize = 10;
    const d: usize = 96; // multiple of 32

    var tt: storage.TiledTensor = undefined;
    try tt.init(
        allocator,
        .q8_0,
        &[_]usize{ v, d },
        &[_]usize{ 4, 64 },
        .{ .tile_alignment = 64, .quant_axis = 1 },
    );
    defer tt.deinit();

    const total_elems: usize = v * d;
    const total_bytes: usize = try backend_utils.requiredBytesForElems(DType.q8_0, total_elems);

    const packed_bytes: []u8 = try allocator.alloc(u8, total_bytes);
    defer allocator.free(packed_bytes);

    // Fill with a pattern that lets us verify block-boundary correctness.
    // Block (row v, block b) byte k => (v*7 + b*13 + k) mod 251.
    const d_blocks: usize = d / 32; // quant axis at 1, block_elems=32
    var row: usize = 0;
    while (row < v) : (row += 1) {
        var b: usize = 0;
        while (b < d_blocks) : (b += 1) {
            const block_off: usize = (row * d_blocks + b) * 34;
            var kk: usize = 0;
            while (kk < 34) : (kk += 1) {
                packed_bytes[block_off + kk] = @intCast((row * 7 + b * 13 + kk) % 251);
            }
        }
    }

    try tt.writeFromPackedQuant(packed_bytes);

    const out: []u8 = try allocator.alloc(u8, total_bytes);
    defer allocator.free(out);
    @memset(out, 0);

    try tt.readToPackedQuant(out);
    try std.testing.expectEqualSlices(u8, packed_bytes, out);
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

test "storage file: cast and matmul_nt nodes roundtrip through write/parse" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var pkg = package_file.Package{
        .allocator = allocator,
        .initializers = try allocator.alloc(package_file.Initializer, 0),
        .values = try allocator.alloc(package_file.ValueRecord, 5),
        .nodes = try allocator.alloc(package_file.NodeRecord, 2),
        .inputs = try allocator.alloc(package_file.NamedValue, 2),
        .outputs = try allocator.alloc(package_file.NamedValue, 2),
        .dim_symbols = try allocator.alloc(package_file.DimSymbol, 0),
        .dim_exprs = try allocator.alloc(package_file.DimExpr, 0),
        .metadata = try allocator.alloc(package_file.MetadataEntry, 0),
        .debug_names = try allocator.alloc(package_file.DebugName, 0),
        .io_aliases = try allocator.alloc(package_file.IoAlias, 0),
    };
    defer pkg.deinit();

    // Values: x[3,4] f32, w[5,4] q8_0, cast_out[3,4] f16, mm_out[3,5] f32, dummy public-input-2 f32 for two-input signature.
    pkg.values[0] = .{
        .dtype = .f32,
        .rank = 2,
        .source = .public_input,
        .shape_terms = try package_file.makeConstantShapeTerms(allocator, &[_]usize{ 3, 4 }),
    };
    pkg.values[1] = .{
        .dtype = .q8_0,
        .rank = 2,
        .source = .public_input,
        .shape_terms = try package_file.makeConstantShapeTerms(allocator, &[_]usize{ 5, 4 }),
    };
    pkg.values[2] = .{
        .dtype = .f16,
        .rank = 2,
        .source = .produced,
        .shape_terms = try allocator.alloc(package_file.ShapeTerm, 0),
    };
    pkg.values[3] = .{
        .dtype = .f32,
        .rank = 2,
        .source = .produced,
        .shape_terms = try allocator.alloc(package_file.ShapeTerm, 0),
    };
    pkg.values[4] = .{
        .dtype = .f32,
        .rank = 2,
        .source = .produced,
        .shape_terms = try allocator.alloc(package_file.ShapeTerm, 0),
    };

    pkg.nodes[0] = .{
        .inputs = try allocator.dupe(u32, &[_]u32{0}),
        .output = 2,
        .op = .{ .Cast = .{ .to_dtype = .f16 } },
    };
    pkg.nodes[1] = .{
        .inputs = try allocator.dupe(u32, &[_]u32{ 0, 1 }),
        .output = 3,
        .op = .{ .MatMulNT = .{ .alpha = 1.25, .beta = 0.5 } },
    };
    pkg.inputs[0] = .{ .name = try allocator.dupe(u8, "x"), .value = 0 };
    pkg.inputs[1] = .{ .name = try allocator.dupe(u8, "w"), .value = 1 };
    pkg.outputs[0] = .{ .name = try allocator.dupe(u8, "cast_out"), .value = 2 };
    pkg.outputs[1] = .{ .name = try allocator.dupe(u8, "mm_out"), .value = 3 };

    // Bypass Package.validate()'s strict topology check for value[4]: remove it by truncating.
    const full_values = pkg.values;
    pkg.values = full_values[0..4];

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try createTestFile(tmp.dir, "cast_mm.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);
    try package_file.writeFile(file, &pkg);

    const bytes = try package_file.readAlloc(allocator, file);
    defer allocator.free(bytes);
    var parsed = try package_file.parse(allocator, bytes);
    defer parsed.deinit();

    try std.testing.expectEqual(@as(usize, 2), parsed.nodes.len);
    switch (parsed.nodes[0].op) {
        .Cast => |ct| try std.testing.expectEqual(types.DType.f16, ct.to_dtype),
        else => return error.TestUnexpectedResult,
    }
    switch (parsed.nodes[1].op) {
        .MatMulNT => |mm| {
            try std.testing.expectEqual(@as(f32, 1.25), mm.alpha);
            try std.testing.expectEqual(@as(f32, 0.5), mm.beta);
        },
        else => return error.TestUnexpectedResult,
    }

    // Restore the truncated slot so deinit frees the full allocation.
    pkg.values = full_values;
    allocator.free(full_values[4].shape_terms);
}

test "storage file: if/loop control-flow nodes roundtrip through write/parse" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var pkg = package_file.Package{
        .allocator = allocator,
        .initializers = try allocator.alloc(package_file.Initializer, 0),
        .values = try allocator.alloc(package_file.ValueRecord, 8),
        .nodes = try allocator.alloc(package_file.NodeRecord, 2),
        .regions = try allocator.alloc(package_file.RegionRecord, 3),
        .inputs = try allocator.alloc(package_file.NamedValue, 5),
        .outputs = try allocator.alloc(package_file.NamedValue, 2),
        .dim_symbols = try allocator.alloc(package_file.DimSymbol, 0),
        .dim_exprs = try allocator.alloc(package_file.DimExpr, 0),
        .metadata = try allocator.alloc(package_file.MetadataEntry, 0),
        .debug_names = try allocator.alloc(package_file.DebugName, 0),
        .io_aliases = try allocator.alloc(package_file.IoAlias, 0),
    };
    defer pkg.deinit();

    // Values: 5 public inputs (cond/then_v/else_v/carried/inc), 3 produced (next, if_out, loop_out).
    pkg.values[0] = .{
        .dtype = .i32,
        .rank = 1,
        .source = .public_input,
        .shape_terms = try package_file.makeConstantShapeTerms(allocator, &[_]usize{1}),
    };
    pkg.values[1] = .{
        .dtype = .f32,
        .rank = 1,
        .source = .public_input,
        .shape_terms = try package_file.makeConstantShapeTerms(allocator, &[_]usize{1}),
    };
    pkg.values[2] = .{
        .dtype = .f32,
        .rank = 1,
        .source = .public_input,
        .shape_terms = try package_file.makeConstantShapeTerms(allocator, &[_]usize{1}),
    };
    pkg.values[3] = .{
        .dtype = .f32,
        .rank = 1,
        .source = .public_input,
        .shape_terms = try package_file.makeConstantShapeTerms(allocator, &[_]usize{1}),
    };
    pkg.values[4] = .{
        .dtype = .f32,
        .rank = 1,
        .source = .public_input,
        .shape_terms = try package_file.makeConstantShapeTerms(allocator, &[_]usize{1}),
    };
    pkg.values[5] = .{
        .dtype = .f32,
        .rank = 1,
        .source = .produced,
        .shape_terms = try allocator.alloc(package_file.ShapeTerm, 0),
    };
    pkg.values[6] = .{
        .dtype = .f32,
        .rank = 1,
        .source = .produced,
        .shape_terms = try allocator.alloc(package_file.ShapeTerm, 0),
    };
    pkg.values[7] = .{
        .dtype = .f32,
        .rank = 1,
        .source = .produced,
        .shape_terms = try allocator.alloc(package_file.ShapeTerm, 0),
    };

    // Region 0: then-branch returns then_v (value 1).
    const then_outputs = try allocator.alloc(u32, 1);
    then_outputs[0] = 1;
    pkg.regions[0] = .{
        .nodes = try allocator.alloc(package_file.NodeRecord, 0),
        .outputs = then_outputs,
    };

    // Region 1: else-branch returns else_v (value 2).
    const else_outputs = try allocator.alloc(u32, 1);
    else_outputs[0] = 2;
    pkg.regions[1] = .{
        .nodes = try allocator.alloc(package_file.NodeRecord, 0),
        .outputs = else_outputs,
    };

    // Region 2: loop body computes next = carried + inc.
    const body_nodes = try allocator.alloc(package_file.NodeRecord, 1);
    body_nodes[0] = .{
        .inputs = try allocator.dupe(u32, &[_]u32{ 3, 4 }),
        .output = 5,
        .op = .{ .ElemwiseBinary = .{ .op = .add } },
    };
    const body_outputs = try allocator.alloc(u32, 1);
    body_outputs[0] = 5;
    pkg.regions[2] = .{ .nodes = body_nodes, .outputs = body_outputs };

    pkg.nodes[0] = .{
        .inputs = try allocator.dupe(u32, &[_]u32{0}),
        .output = 6,
        .op = .{ .If = .{ .then_region = 0, .else_region = 1 } },
    };
    pkg.nodes[1] = .{
        .inputs = try allocator.dupe(u32, &[_]u32{3}),
        .output = 7,
        .op = .{ .Loop = .{ .body_region = 2, .static_max_trip_count = 4 } },
    };

    pkg.inputs[0] = .{ .name = try allocator.dupe(u8, "cond"), .value = 0 };
    pkg.inputs[1] = .{ .name = try allocator.dupe(u8, "then_v"), .value = 1 };
    pkg.inputs[2] = .{ .name = try allocator.dupe(u8, "else_v"), .value = 2 };
    pkg.inputs[3] = .{ .name = try allocator.dupe(u8, "carried"), .value = 3 };
    pkg.inputs[4] = .{ .name = try allocator.dupe(u8, "inc"), .value = 4 };
    pkg.outputs[0] = .{ .name = try allocator.dupe(u8, "if_out"), .value = 6 };
    pkg.outputs[1] = .{ .name = try allocator.dupe(u8, "loop_out"), .value = 7 };

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try createTestFile(tmp.dir, "ctrl_flow.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);
    try package_file.writeFile(file, &pkg);

    const bytes = try package_file.readAlloc(allocator, file);
    defer allocator.free(bytes);
    var parsed = try package_file.parse(allocator, bytes);
    defer parsed.deinit();

    try std.testing.expectEqual(@as(usize, 3), parsed.regions.len);
    try std.testing.expectEqual(@as(usize, 0), parsed.regions[0].nodes.len);
    try std.testing.expectEqual(@as(usize, 1), parsed.regions[0].outputs.len);
    try std.testing.expectEqual(@as(u32, 1), parsed.regions[0].outputs[0]);
    try std.testing.expectEqual(@as(usize, 0), parsed.regions[1].nodes.len);
    try std.testing.expectEqual(@as(u32, 2), parsed.regions[1].outputs[0]);

    try std.testing.expectEqual(@as(usize, 1), parsed.regions[2].nodes.len);
    try std.testing.expectEqual(@as(u32, 5), parsed.regions[2].outputs[0]);
    switch (parsed.regions[2].nodes[0].op) {
        .ElemwiseBinary => |eb| try std.testing.expectEqual(types.ElemwiseBinaryOp.add, eb.op),
        else => return error.TestUnexpectedResult,
    }
    try std.testing.expectEqualSlices(u32, &[_]u32{ 3, 4 }, parsed.regions[2].nodes[0].inputs);

    try std.testing.expectEqual(@as(usize, 2), parsed.nodes.len);
    switch (parsed.nodes[0].op) {
        .If => |iff| {
            try std.testing.expectEqual(@as(u32, 0), iff.then_region);
            try std.testing.expectEqual(@as(u32, 1), iff.else_region);
        },
        else => return error.TestUnexpectedResult,
    }
    switch (parsed.nodes[1].op) {
        .Loop => |lp| {
            try std.testing.expectEqual(@as(u32, 2), lp.body_region);
            try std.testing.expectEqual(@as(u64, 4), lp.static_max_trip_count);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "storage file: rejects invalid control-flow region id" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var pkg = package_file.Package{
        .allocator = allocator,
        .initializers = try allocator.alloc(package_file.Initializer, 0),
        .values = try allocator.alloc(package_file.ValueRecord, 3),
        .nodes = try allocator.alloc(package_file.NodeRecord, 1),
        .regions = try allocator.alloc(package_file.RegionRecord, 1),
        .inputs = try allocator.alloc(package_file.NamedValue, 2),
        .outputs = try allocator.alloc(package_file.NamedValue, 1),
        .dim_symbols = try allocator.alloc(package_file.DimSymbol, 0),
        .dim_exprs = try allocator.alloc(package_file.DimExpr, 0),
        .metadata = try allocator.alloc(package_file.MetadataEntry, 0),
        .debug_names = try allocator.alloc(package_file.DebugName, 0),
        .io_aliases = try allocator.alloc(package_file.IoAlias, 0),
    };
    defer pkg.deinit();

    pkg.values[0] = .{
        .dtype = .i32,
        .rank = 1,
        .source = .public_input,
        .shape_terms = try package_file.makeConstantShapeTerms(allocator, &[_]usize{1}),
    };
    pkg.values[1] = .{
        .dtype = .f32,
        .rank = 1,
        .source = .public_input,
        .shape_terms = try package_file.makeConstantShapeTerms(allocator, &[_]usize{1}),
    };
    pkg.values[2] = .{
        .dtype = .f32,
        .rank = 1,
        .source = .produced,
        .shape_terms = try allocator.alloc(package_file.ShapeTerm, 0),
    };

    const region0_outputs = try allocator.alloc(u32, 1);
    region0_outputs[0] = 1;
    pkg.regions[0] = .{
        .nodes = try allocator.alloc(package_file.NodeRecord, 0),
        .outputs = region0_outputs,
    };

    pkg.nodes[0] = .{
        .inputs = try allocator.dupe(u32, &[_]u32{0}),
        .output = 2,
        // else_region 999 is out of bounds.
        .op = .{ .If = .{ .then_region = 0, .else_region = 999 } },
    };

    pkg.inputs[0] = .{ .name = try allocator.dupe(u8, "cond"), .value = 0 };
    pkg.inputs[1] = .{ .name = try allocator.dupe(u8, "v"), .value = 1 };
    pkg.outputs[0] = .{ .name = try allocator.dupe(u8, "out"), .value = 2 };

    try std.testing.expectError(package_file.PackageError.InvalidFormat, package_file.validate(&pkg));
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

// Four op attributes can carry free axes, but only a reshape's and a slice's are
// reachable from the authoring APIs (`Builder.reshapeSym`/`sliceSym`, and the `reshape`
// and `slice` members of `AionOpAttr` — `norm` has no `_symbols` array). The two norms
// have the slot because the format stores a shape term per axis for all four, so a
// hand-written package can put one there and must round-trip unchanged.
test "storage file: a norm's normalized_shape can be free" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var pkg = package_file.Package{
        .allocator = allocator,
        .initializers = try allocator.alloc(package_file.Initializer, 0),
        .values = try allocator.alloc(package_file.ValueRecord, 4),
        .nodes = try allocator.alloc(package_file.NodeRecord, 1),
        .inputs = try allocator.alloc(package_file.NamedValue, 3),
        .outputs = try allocator.alloc(package_file.NamedValue, 1),
        .dim_symbols = try allocator.alloc(package_file.DimSymbol, 1),
        .dim_exprs = try allocator.alloc(package_file.DimExpr, 1),
        .metadata = try allocator.alloc(package_file.MetadataEntry, 0),
        .debug_names = try allocator.alloc(package_file.DebugName, 0),
        .io_aliases = try allocator.alloc(package_file.IoAlias, 0),
    };
    defer pkg.deinit();

    pkg.dim_symbols[0] = .{ .name = try allocator.dupe(u8, "width") };
    pkg.dim_exprs[0] = .{ .symbol = 0 };

    // x, gamma and beta all have a free trailing axis, which is what makes a free
    // `normalized_shape` coherent at all: inference requires the three to agree.
    for (0..3) |i| {
        const terms = try allocator.alloc(package_file.ShapeTerm, if (i == 0) 2 else 1);
        if (i == 0) {
            terms[0] = .{ .constant = 1 };
            terms[1] = .{ .expr = 0 };
        } else {
            terms[0] = .{ .expr = 0 };
        }
        pkg.values[i] = .{
            .dtype = .f32,
            .rank = @intCast(terms.len),
            .source = .public_input,
            .shape_terms = terms,
        };
    }
    pkg.values[3] = .{
        .dtype = .f32,
        .rank = 2,
        .source = .produced,
        .shape_terms = try allocator.alloc(package_file.ShapeTerm, 0),
    };

    const norm_shape = try allocator.alloc(usize, 1);
    norm_shape[0] = 0; // a free axis records no size
    const norm_free = try allocator.alloc(?u32, 1);
    norm_free[0] = 0; // ...it resolves through dim expression 0
    pkg.nodes[0] = .{
        .inputs = try allocator.dupe(u32, &[_]u32{ 0, 1, 2 }),
        .output = 3,
        .op = .{ .RMSNorm = .{ .eps = 1e-6, .normalized_shape = norm_shape, .free_dims = norm_free } },
    };
    pkg.inputs[0] = .{ .name = try allocator.dupe(u8, "x"), .value = 0 };
    pkg.inputs[1] = .{ .name = try allocator.dupe(u8, "gamma"), .value = 1 };
    pkg.inputs[2] = .{ .name = try allocator.dupe(u8, "beta"), .value = 2 };
    pkg.outputs[0] = .{ .name = try allocator.dupe(u8, "y"), .value = 3 };

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const file = try createTestFile(tmp.dir, "free_norm.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);
    try package_file.writeFile(file, &pkg);

    const bytes = try package_file.readAlloc(allocator, file);
    defer allocator.free(bytes);
    var parsed = try package_file.parse(allocator, bytes);
    defer parsed.deinit();

    // The free axis survived, and its size is still the "unresolved" 0.
    const rn = parsed.nodes[0].op.RMSNorm;
    try std.testing.expectEqual(@as(usize, 1), rn.normalized_shape.len);
    try std.testing.expectEqual(@as(usize, 0), rn.normalized_shape[0]);
    try std.testing.expectEqual(@as(usize, 1), rn.free_dims.len);
    try std.testing.expectEqual(@as(?u32, 0), rn.free_dims[0]);
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

fn buildRolesTestPackage(allocator: std.mem.Allocator) !package_file.Package {
    // Inputs: k_cache (f16 [1,1,G,8], io-aliased to output 0), write_idx (i32 [1]),
    // tokens (i32 [1,4]). Output 0 references the cache value (alias target).
    var pkg = package_file.Package{
        .allocator = allocator,
        .initializers = try allocator.alloc(package_file.Initializer, 0),
        .values = try allocator.alloc(package_file.ValueRecord, 3),
        .nodes = try allocator.alloc(package_file.NodeRecord, 0),
        .inputs = try allocator.alloc(package_file.NamedValue, 3),
        .outputs = try allocator.alloc(package_file.NamedValue, 1),
        .dim_symbols = try allocator.alloc(package_file.DimSymbol, 1),
        .dim_exprs = try allocator.alloc(package_file.DimExpr, 1),
        .metadata = try allocator.alloc(package_file.MetadataEntry, 0),
        .debug_names = try allocator.alloc(package_file.DebugName, 0),
        .io_aliases = try allocator.alloc(package_file.IoAlias, 1),
        .input_roles = try allocator.alloc(package_file.InputRole, 3),
    };
    errdefer pkg.deinit();

    pkg.dim_symbols[0] = .{ .name = try allocator.dupe(u8, "G") };
    pkg.dim_exprs[0] = .{ .symbol = 0 };

    const cache_terms = try allocator.alloc(package_file.ShapeTerm, 4);
    cache_terms[0] = .{ .constant = 1 };
    cache_terms[1] = .{ .expr = 0 };
    cache_terms[2] = .{ .constant = 1 };
    cache_terms[3] = .{ .constant = 8 };
    pkg.values[0] = .{ .dtype = .f16, .rank = 4, .source = .public_input, .shape_terms = cache_terms };
    pkg.values[1] = .{
        .dtype = .i32,
        .rank = 1,
        .source = .public_input,
        .shape_terms = try package_file.makeConstantShapeTerms(allocator, &[_]usize{1}),
    };
    pkg.values[2] = .{
        .dtype = .i32,
        .rank = 2,
        .source = .public_input,
        .shape_terms = try package_file.makeConstantShapeTerms(allocator, &[_]usize{ 1, 4 }),
    };

    pkg.inputs[0] = .{ .name = try allocator.dupe(u8, "k_cache"), .value = 0 };
    pkg.inputs[1] = .{ .name = try allocator.dupe(u8, "write_idx"), .value = 1 };
    pkg.inputs[2] = .{ .name = try allocator.dupe(u8, "tokens"), .value = 2 };
    pkg.outputs[0] = .{ .name = try allocator.dupe(u8, "next_k_cache"), .value = 0 };
    pkg.io_aliases[0] = .{ .input = 0, .output = 0 };

    pkg.input_roles[0] = .{
        .input = 0,
        .kind = .sequence_cache,
        .axis = 1,
        .flags = package_file.InputRoleFlags.zero_init,
        .capacity_symbol = 0,
        .retained_history_tokens = 511,
    };
    pkg.input_roles[1] = .{ .input = 1, .kind = .cache_write_index };
    pkg.input_roles[2] = .{ .input = 2, .kind = .tokens, .axis = 1 };
    return pkg;
}

test "storage file: input roles roundtrip through write/parse" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var pkg = try buildRolesTestPackage(allocator);
    defer pkg.deinit();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const file = try createTestFile(tmp.dir, "roles.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);
    try package_file.writeFile(file, &pkg);

    const bytes = try package_file.readAlloc(allocator, file);
    defer allocator.free(bytes);
    var parsed = try package_file.parse(allocator, bytes);
    defer parsed.deinit();

    try std.testing.expectEqual(@as(usize, 3), parsed.input_roles.len);
    try std.testing.expectEqualSlices(package_file.InputRole, pkg.input_roles, parsed.input_roles);
}

test "storage file: package without input roles parses to empty slice" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var pkg = try buildRolesTestPackage(allocator);
    defer pkg.deinit();
    allocator.free(pkg.input_roles);
    pkg.input_roles = &[_]package_file.InputRole{};

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const file = try createTestFile(tmp.dir, "noroles.aion", .{ .read = true, .truncate = true });
    defer file.close(std.testing.io);
    try package_file.writeFile(file, &pkg);

    const bytes = try package_file.readAlloc(allocator, file);
    defer allocator.free(bytes);
    var parsed = try package_file.parse(allocator, bytes);
    defer parsed.deinit();
    try std.testing.expectEqual(@as(usize, 0), parsed.input_roles.len);
}

test "storage file: invalid input roles are rejected" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var pkg = try buildRolesTestPackage(allocator);
    defer pkg.deinit();

    // Baseline is valid.
    try package_file.validate(&pkg);

    // Input index out of range.
    pkg.input_roles[1].input = 99;
    try std.testing.expectError(package_file.PackageError.InvalidFormat, package_file.validate(&pkg));
    pkg.input_roles[1].input = 1;

    // Duplicate role on one input.
    pkg.input_roles[1].input = 0;
    try std.testing.expectError(package_file.PackageError.InvalidFormat, package_file.validate(&pkg));
    pkg.input_roles[1].input = 1;

    // sequence_cache must be io-aliased.
    pkg.input_roles[0].input = 1;
    pkg.input_roles[1].input = 0;
    try std.testing.expectError(package_file.PackageError.InvalidFormat, package_file.validate(&pkg));
    pkg.input_roles[0].input = 0;
    pkg.input_roles[1].input = 1;

    // capacity_symbol out of range.
    pkg.input_roles[0].capacity_symbol = 7;
    try std.testing.expectError(package_file.PackageError.InvalidFormat, package_file.validate(&pkg));
    pkg.input_roles[0].capacity_symbol = 0;

    // Two `tokens` roles package-wide.
    pkg.input_roles[1] = .{ .input = 1, .kind = .tokens, .axis = 0 };
    try std.testing.expectError(package_file.PackageError.InvalidFormat, package_file.validate(&pkg));
    pkg.input_roles[1] = .{ .input = 1, .kind = .cache_write_index };

    // Control-role input must be i32 (retarget tokens role at the f16 cache... use
    // write_index on the cache input, which is also a duplicate-free check).
    pkg.input_roles[2] = .{ .input = 2, .kind = .tokens, .axis = 9 }; // axis out of range
    try std.testing.expectError(package_file.PackageError.InvalidFormat, package_file.validate(&pkg));
    pkg.input_roles[2] = .{ .input = 2, .kind = .tokens, .axis = 1 };

    // Unknown flag bits.
    pkg.input_roles[0].flags = 0x80;
    try std.testing.expectError(package_file.PackageError.InvalidFormat, package_file.validate(&pkg));
    pkg.input_roles[0].flags = package_file.InputRoleFlags.zero_init;

    try package_file.validate(&pkg);
}

test "storage cache: lease tokens and policy info" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var sm: manager_mod.StorageManager = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    try sm.configureCache(.{ .ram_budget_bytes = 1 << 20, .max_live_leases = 2 });

    const tid: manager_mod.TensorId = try sm.createTiledTensor(
        .f32,
        &[_]usize{8},
        &[_]usize{4},
        .{ .tile_alignment = 64 },
    );
    try sm.registerSequenceCachePolicy(tid, .{ .rolling = .{ .history_tokens = 4 } });

    const store: tensor_store.TensorStore = sm.tensorStore();
    const info: tensor_store.SequenceCachePolicyInfo = store.sequenceCachePolicyInfo(tid);
    try std.testing.expectEqual(tensor_store.SequenceCachePolicyKind.rolling, info.kind);
    try std.testing.expectEqual(@as(usize, 4), info.rolling_history_tokens);

    var t0: tensor_store.TileRefConst = try store.acquireTileConstLinear(tid, 0);
    defer store.releaseConst(t0.token);
    const t1: tensor_store.TileRefConst = try store.acquireTileConstLinear(tid, 1);
    defer store.releaseConst(t1.token);

    try std.testing.expect(t0.token != 0);
    try std.testing.expect(t1.token != 0);
    try std.testing.expect(t0.token != t1.token);

    try std.testing.expectError(tensor_store.StoreError.InvalidArgument, store.acquireTileConstLinear(tid, 0));

    store.releaseConst(t0.token);
    t0.token = 0;

    const t2: tensor_store.TileRefConst = try store.acquireTileConstLinear(tid, 0);
    defer store.releaseConst(t2.token);
    try std.testing.expect(t2.token != 0);

    const mapped_ring: usize = try store.mapSequenceStep(tid, 5, 4);
    try std.testing.expectEqual(@as(usize, 1), mapped_ring);

    const grow_tid: manager_mod.TensorId = try sm.createTiledTensor(
        .f32,
        &[_]usize{ 1, 1, 8, 1 },
        &[_]usize{ 1, 1, 4, 1 },
        .{ .tile_alignment = 64 },
    );
    try sm.registerSequenceCachePolicy(grow_tid, .{ .growable = .{ .initial_capacity_tokens = 2, .growth_numerator = 2, .growth_denominator = 1 } });
    const mapped_grow: usize = try store.mapSequenceStep(grow_tid, 3, 8);
    try std.testing.expectEqual(@as(usize, 3), mapped_grow);
    const mapped_grow_expand: usize = try store.mapSequenceStep(grow_tid, 8, 8);
    try std.testing.expectEqual(@as(usize, 8), mapped_grow_expand);
    const grow_meta: *const manager_mod.TiledTensor = try sm.getConst(grow_tid);
    try std.testing.expectEqual(@as(usize, 16), grow_meta.shape[1]);

    const plain_tid: manager_mod.TensorId = try sm.createTiledTensor(
        .f32,
        &[_]usize{8},
        &[_]usize{4},
        .{ .tile_alignment = 64 },
    );
    try std.testing.expectError(tensor_store.StoreError.InvalidArgument, store.mapSequenceStep(plain_tid, 9, 8));
}

test "storage cache: rolling growth rehashes retained logical rows" {
    const allocator = std.testing.allocator;
    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const tid = try sm.createTiledTensor(
        .f32,
        &.{ 1, 4, 1, 1 },
        &.{ 1, 4, 1, 1 },
        .{ .tile_alignment = 64 },
    );
    // At logical end 6, retained rows 3,4,5 occupy physical 3,0,1.
    try sm.writeFromPackedScalar(tid, std.mem.sliceAsBytes(&[_]f32{ 4, 5, 0, 3 }));
    try sm.ensureRollingCacheCapacity(tid, 7, 3, &.{6});

    var got: [7]f32 = undefined;
    try sm.readToPackedScalar(tid, std.mem.sliceAsBytes(&got));
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 3, 4, 5, 0 }, &got);
}
