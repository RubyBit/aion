const std = @import("std");

const backend_mod = @import("../backend/backend.zig");
const cpu_backend_mod = @import("../backend/cpu/cpu_backend.zig");
const types = @import("../backend/types.zig");
const backend_utils = @import("../backend/utils.zig");
const storage = @import("../storage/storage.zig");
const program = @import("program.zig");

const Backend = backend_mod.Backend;
const DType = types.DType;

const PackedLayout2 = struct {
    shape_mem: [2]usize,
    strides_mem: [2]isize,

    pub fn init(shape0: usize, shape1: usize, elem_bytes: usize) PackedLayout2 {
        return .{
            .shape_mem = .{ shape0, shape1 },
            .strides_mem = .{ @intCast(shape1 * elem_bytes), @intCast(elem_bytes) },
        };
    }

    pub fn layout(self: *const PackedLayout2) types.Layout {
        return .{
            .rank = 2,
            .shape = self.shape_mem[0..2],
            .strides_bytes = self.strides_mem[0..2],
        };
    }
};

const PackedLayout2Quant = struct {
    shape_mem: [2]usize,
    strides_mem: [2]isize,

    pub fn init(shape0: usize, shape1: usize) PackedLayout2Quant {
        return .{
            .shape_mem = .{ shape0, shape1 },
            .strides_mem = .{ 0, 0 },
        };
    }

    pub fn layout(self: *const PackedLayout2Quant) types.Layout {
        return .{
            .rank = 2,
            .shape = self.shape_mem[0..2],
            .strides_bytes = self.strides_mem[0..2],
        };
    }
};

test "program: tiled matmul matches packed (f32 x f32 -> f32)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const m: usize = 7;
    const k: usize = 13;
    const n: usize = 9;

    // Choose tile sizes that create boundary tiles.
    const tm: usize = 3;
    const tk: usize = 5;
    const tn: usize = 4;

    // Packed inputs.
    const a_elems: usize = m * k;
    const b_elems: usize = k * n;
    const c_elems: usize = m * n;

    const a_bytes_len: usize = a_elems * 4;
    const b_bytes_len: usize = b_elems * 4;
    const c_bytes_len: usize = c_elems * 4;

    const a_buf: []u8 = try allocator.alloc(u8, a_bytes_len);
    defer allocator.free(a_buf);
    const b_buf: []u8 = try allocator.alloc(u8, b_bytes_len);
    defer allocator.free(b_buf);
    const c_ref_buf: []u8 = try allocator.alloc(u8, c_bytes_len);
    defer allocator.free(c_ref_buf);

    // Fill A,B deterministically with small values.
    const a_vals: []align(1) f32 = @ptrCast(a_buf);
    const b_vals: []align(1) f32 = @ptrCast(b_buf);
    for (0..a_elems) |i| a_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) * 0.25;
    for (0..b_elems) |i| b_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i % 19)) - 9)) * 0.2;
    @memset(c_ref_buf, 0);

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: Backend = cpu.backend();

    // Reference: one packed matmul.
    var a_l = PackedLayout2.init(m, k, 4);
    var b_l = PackedLayout2.init(k, n, 4);
    var c_l = PackedLayout2.init(m, n, 4);

    const a_view: types.BufferViewConst = .{ .bytes = a_buf, .dtype = .f32, .layout = a_l.layout() };
    const b_view: types.BufferViewConst = .{ .bytes = b_buf, .dtype = .f32, .layout = b_l.layout() };
    const c_ref_view: types.BufferViewMut = .{ .bytes = c_ref_buf, .dtype = .f32, .layout = c_l.layout() };

    try backend.matmul(.{ .m = m, .n = n, .k = k, .alpha = 1.0, .beta = 0.0 }, c_ref_view, a_view, b_view);

    // Tiled tensors.
    var a_t: storage.TiledTensor = try storage.TiledTensor.init(allocator, .f32, &[_]usize{ m, k }, &[_]usize{ tm, tk }, .{ .tile_alignment = 64 });
    defer a_t.deinit();
    var b_t: storage.TiledTensor = try storage.TiledTensor.init(allocator, .f32, &[_]usize{ k, n }, &[_]usize{ tk, tn }, .{ .tile_alignment = 64 });
    defer b_t.deinit();
    var c_t: storage.TiledTensor = try storage.TiledTensor.init(allocator, .f32, &[_]usize{ m, n }, &[_]usize{ tm, tn }, .{ .tile_alignment = 64 });
    defer c_t.deinit();

    try a_t.writeFromPackedScalar(a_buf);
    try b_t.writeFromPackedScalar(b_buf);

    // Compute tiled.
    try program.matmulTiled(backend, &c_t, &a_t, &b_t, 1.0, 0.0);

    // Compare.
    const c_out_buf: []u8 = try allocator.alloc(u8, c_bytes_len);
    defer allocator.free(c_out_buf);
    @memset(c_out_buf, 0);
    try c_t.readToPackedScalar(c_out_buf);

    const c_ref: []align(1) const f32 = @ptrCast(c_ref_buf);
    const c_out: []align(1) const f32 = @ptrCast(c_out_buf);

    // Exact equality is expected for deterministic f32 math in same loop order? Not guaranteed.
    // Use a tight tolerance.
    for (0..c_elems) |i| {
        const diff: f32 = @abs(c_ref[i] - c_out[i]);
        try std.testing.expect(diff <= 1e-5);
    }
}

test "program: tiled matmul matches packed (f32 x q8_0 -> f32)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const m: usize = 4;
    const k: usize = 96; // multiple of 32
    const n: usize = 7;

    const tm: usize = 2;
    const tk: usize = 64; // multiple of 32, and leaves boundary k tile (32)
    const tn: usize = 3;

    // Packed A (f32).
    const a_elems: usize = m * k;
    const a_bytes_len: usize = a_elems * 4;
    const a_buf: []u8 = try allocator.alloc(u8, a_bytes_len);
    defer allocator.free(a_buf);
    const a_vals: []align(1) f32 = @ptrCast(a_buf);
    for (0..a_elems) |i| a_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) * 0.1;

    // Packed B (q8_0) per kernel convention: [k_blocks, n] blocks.
    const di = DType.q8_0.info();
    const k_blocks: usize = k / di.block_elems;
    const b_bytes_len: usize = k_blocks * n * di.block_bytes;
    var b_buf: []u8 = try allocator.alloc(u8, b_bytes_len);
    defer allocator.free(b_buf);

    // Fill each block: scale=f16(1.0), then 32 int8 values.
    const scale_f16: f16 = 1.0;
    const scale_bytes: [2]u8 = @bitCast(scale_f16);
    var kb: usize = 0;
    while (kb < k_blocks) : (kb += 1) {
        var j: usize = 0;
        while (j < n) : (j += 1) {
            const off: usize = (kb * n + j) * di.block_bytes;
            b_buf[off + 0] = scale_bytes[0];
            b_buf[off + 1] = scale_bytes[1];

            // Deterministic small int8 pattern.
            var qi: usize = 0;
            while (qi < 32) : (qi += 1) {
                const v: i8 = @intCast(@as(i32, @intCast((kb + j + qi) % 13)) - 6);
                b_buf[off + 2 + qi] = @bitCast(v);
            }
        }
    }

    const c_bytes_len: usize = (m * n) * 4;
    const c_ref_buf: []u8 = try allocator.alloc(u8, c_bytes_len);
    defer allocator.free(c_ref_buf);
    @memset(c_ref_buf, 0);

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: Backend = cpu.backend();

    var a_l = PackedLayout2.init(m, k, 4);
    var b_l = PackedLayout2Quant.init(k, n);
    var c_l = PackedLayout2.init(m, n, 4);

    const a_view: types.BufferViewConst = .{ .bytes = a_buf, .dtype = .f32, .layout = a_l.layout() };
    const b_view: types.BufferViewConst = .{ .bytes = b_buf, .dtype = .q8_0, .layout = b_l.layout() };
    const c_ref_view: types.BufferViewMut = .{ .bytes = c_ref_buf, .dtype = .f32, .layout = c_l.layout() };

    // Reference packed.
    try backend.matmul(.{ .m = m, .n = n, .k = k, .alpha = 1.0, .beta = 0.0 }, c_ref_view, a_view, b_view);

    // Tiled.
    var a_t: storage.TiledTensor = try storage.TiledTensor.init(allocator, .f32, &[_]usize{ m, k }, &[_]usize{ tm, tk }, .{ .tile_alignment = 64 });
    defer a_t.deinit();
    var b_t: storage.TiledTensor = try storage.TiledTensor.init(allocator, .q8_0, &[_]usize{ k, n }, &[_]usize{ tk, tn }, .{ .tile_alignment = 64 });
    defer b_t.deinit();
    var c_t: storage.TiledTensor = try storage.TiledTensor.init(allocator, .f32, &[_]usize{ m, n }, &[_]usize{ tm, tn }, .{ .tile_alignment = 64 });
    defer c_t.deinit();

    try a_t.writeFromPackedScalar(a_buf);
    try b_t.writeFromPackedQuant(b_buf);

    try program.matmulTiled(backend, &c_t, &a_t, &b_t, 1.0, 0.0);

    const c_out_buf: []u8 = try allocator.alloc(u8, c_bytes_len);
    defer allocator.free(c_out_buf);
    @memset(c_out_buf, 0);
    try c_t.readToPackedScalar(c_out_buf);

    const c_ref: []align(1) const f32 = @ptrCast(c_ref_buf);
    const c_out: []align(1) const f32 = @ptrCast(c_out_buf);

    for (0..m * n) |i| {
        const diff: f32 = @abs(c_ref[i] - c_out[i]);
        try std.testing.expect(diff <= 1e-5);
    }

    // Spot-check: tiled B tiles should be valid packed quant views.
    const tv = try b_t.acquireTileConst(0, 0);
    try backend_utils.requirePacked(tv.bufferView());
}
