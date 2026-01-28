const std = @import("std");

const backend_mod = @import("../backend/backend.zig");
const cpu_backend_mod = @import("../backend/cpu/cpu_backend.zig");
const types = @import("../backend/types.zig");
const manager_mod = @import("../storage/manager.zig");
const graph_mod = @import("graph.zig");
const infer_mod = @import("infer.zig");
const plan_mod = @import("plan.zig");
const program = @import("program.zig");

const Backend = backend_mod.Backend;

fn asF32Slice(buf: []u8) []align(1) f32 {
    std.debug.assert((buf.len % @sizeOf(f32)) == 0);
    const ptr: [*]align(1) f32 = @ptrCast(buf.ptr);
    return ptr[0 .. buf.len / @sizeOf(f32)];
}

fn asF16Slice(buf: []u8) []align(1) f16 {
    std.debug.assert((buf.len % @sizeOf(f16)) == 0);
    const ptr: [*]align(1) f16 = @ptrCast(buf.ptr);
    return ptr[0 .. buf.len / @sizeOf(f16)];
}

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

test "graph: compile+run covers matmul/broadcast/elemwise/relu/copy/reduce" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const m: usize = 2;
    const k: usize = 3;
    const n: usize = 4;

    // Packed inputs.
    const a_bytes_len: usize = m * k * 4;
    const b_bytes_len: usize = k * n * 4;
    const bias_bytes_len: usize = n * 4;

    const a_buf: []u8 = try allocator.alloc(u8, a_bytes_len);
    defer allocator.free(a_buf);
    const b_buf: []u8 = try allocator.alloc(u8, b_bytes_len);
    defer allocator.free(b_buf);
    const bias_buf: []u8 = try allocator.alloc(u8, bias_bytes_len);
    defer allocator.free(bias_buf);

    const a_vals: []align(1) f32 = asF32Slice(a_buf);
    const b_vals: []align(1) f32 = asF32Slice(b_buf);
    const bias_vals: []align(1) f32 = asF32Slice(bias_buf);

    for (0..m * k) |i| a_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 2)) * 0.5;
    for (0..k * n) |i| b_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 7))) - 3)) * 0.25;
    for (0..n) |i| bias_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 1)) * 0.1;

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: Backend = cpu.backend();

    // Reference (all packed).
    const c_bytes_len: usize = m * n * 4;
    const d_bytes_len: usize = c_bytes_len;
    const e_bytes_len: usize = c_bytes_len;
    const f_bytes_len: usize = c_bytes_len;
    const g_bytes_len: usize = c_bytes_len;
    const out_bytes_len: usize = 4;

    const c_ref_buf: []u8 = try allocator.alloc(u8, c_bytes_len);
    defer allocator.free(c_ref_buf);
    const d_ref_buf: []u8 = try allocator.alloc(u8, d_bytes_len);
    defer allocator.free(d_ref_buf);
    const e_ref_buf: []u8 = try allocator.alloc(u8, e_bytes_len);
    defer allocator.free(e_ref_buf);
    const f_ref_buf: []u8 = try allocator.alloc(u8, f_bytes_len);
    defer allocator.free(f_ref_buf);
    const g_ref_buf: []u8 = try allocator.alloc(u8, g_bytes_len);
    defer allocator.free(g_ref_buf);
    const out_ref_buf: []u8 = try allocator.alloc(u8, out_bytes_len);
    defer allocator.free(out_ref_buf);

    // Naive reference on packed inputs (computed in plain Zig).
    const c_ref: []align(1) f32 = asF32Slice(c_ref_buf);
    const d_ref: []align(1) f32 = asF32Slice(d_ref_buf);
    const e_ref: []align(1) f32 = asF32Slice(e_ref_buf);
    const f_ref: []align(1) f32 = asF32Slice(f_ref_buf);
    const g_ref: []align(1) f32 = asF32Slice(g_ref_buf);
    const out_ref: []align(1) f32 = asF32Slice(out_ref_buf);

    // c = a @ b
    for (0..m) |i| {
        for (0..n) |j| {
            var acc: f32 = 0.0;
            for (0..k) |kk| {
                acc += a_vals[i * k + kk] * b_vals[kk * n + j];
            }
            c_ref[i * n + j] = acc;
        }
    }
    // d = c + bias (broadcast over last dim)
    for (0..m) |i| {
        for (0..n) |j| {
            d_ref[i * n + j] = c_ref[i * n + j] + bias_vals[j];
        }
    }
    // e = relu(d)
    for (0..m * n) |idx| {
        const v: f32 = d_ref[idx];
        e_ref[idx] = if (v > 0) v else 0;
    }
    // f = copy(e)
    @memcpy(f_ref[0 .. m * n], e_ref[0 .. m * n]);
    // g = f * f
    for (0..m * n) |idx| {
        const v: f32 = f_ref[idx];
        g_ref[idx] = v * v;
    }
    // out = mean(g)
    var sum: f32 = 0.0;
    for (0..m * n) |idx| sum += g_ref[idx];
    out_ref[0] = sum / @as(f32, @floatFromInt(m * n));

    // Build tiled inputs and graph.
    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    // Intentionally use non-matmul-friendly tiling to exercise ReTileCopyScalar.
    const a_tid = try sm.createTiledTensor(.f32, &[_]usize{ m, k }, &[_]usize{ 1, 2 }, .{ .tile_alignment = 64 });
    const b_tid = try sm.createTiledTensor(.f32, &[_]usize{ k, n }, &[_]usize{ 2, 3 }, .{ .tile_alignment = 64 });
    const bias_tid = try sm.createTiledTensor(.f32, &[_]usize{n}, &[_]usize{3}, .{ .tile_alignment = 64 });

    try sm.writeFromPackedScalar(a_tid, a_buf);
    try sm.writeFromPackedScalar(b_tid, b_buf);
    try sm.writeFromPackedScalar(bias_tid, bias_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const a_in = try g.addInput(.f32, &[_]usize{ m, k });
    const b_in = try g.addInput(.f32, &[_]usize{ k, n });
    const bias_in = try g.addInput(.f32, &[_]usize{n});
    try g.bindExternal(a_in, @intCast(a_tid));
    try g.bindExternal(b_in, @intCast(b_tid));
    try g.bindExternal(bias_in, @intCast(bias_tid));

    const c = try g.addMatMul(a_in, b_in, 1.0, 0.0);
    const d = try g.addBroadcastLastDimBinary(.add, c, bias_in);
    const e = try g.addRelu(d);
    const f = try g.addCopy(e);
    const gg = try g.addElemwiseBinary(.mul, f, f);
    const out = try g.addReduce(.mean, gg);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 2, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    try backend.executeProgram(&prog, sm.tensorStore());

    // Read output.
    var out_buf: [4]u8 = .{ 0, 0, 0, 0 };
    try sm.readToPackedScalar(prog.outputs[0], out_buf[0..4]);
    const out_val: f32 = @as(*align(1) const f32, @ptrCast(out_buf[0..4].ptr)).*;
    try std.testing.expect(std.math.isFinite(out_val));
    // Keep this reasonably tight, but allow minor FP differences due to kernel accumulation order.
    try std.testing.expectApproxEqAbs(out_ref[0], out_val, 1e-5);
}

test "graph: matmul f16 allows promoted f32 output" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const m: usize = 2;
    const k: usize = 4;
    const n: usize = 3;

    const a_bytes_len: usize = m * k * 2;
    const b_bytes_len: usize = k * n * 2;

    const a_buf: []u8 = try allocator.alloc(u8, a_bytes_len);
    defer allocator.free(a_buf);
    const b_buf: []u8 = try allocator.alloc(u8, b_bytes_len);
    defer allocator.free(b_buf);

    const a_vals: []align(1) f16 = asF16Slice(a_buf);
    const b_vals: []align(1) f16 = asF16Slice(b_buf);

    for (0..m * k) |i| a_vals[i] = @as(f16, @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast(i)) - 3)) * 0.25));
    for (0..k * n) |i| b_vals[i] = @as(f16, @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i % 5))) - 2)) * 0.5));

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: Backend = cpu.backend();

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    // Intentionally choose non-matmul-friendly tiling to ensure retile logic is exercised.
    const a_tid = try sm.createTiledTensor(.f16, &[_]usize{ m, k }, &[_]usize{ 1, 2 }, .{ .tile_alignment = 64 });
    const b_tid = try sm.createTiledTensor(.f16, &[_]usize{ k, n }, &[_]usize{ 2, 1 }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(a_tid, a_buf);
    try sm.writeFromPackedScalar(b_tid, b_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const a_in = try g.addInput(.f16, &[_]usize{ m, k });
    const b_in = try g.addInput(.f16, &[_]usize{ k, n });
    try g.bindExternal(a_in, @intCast(a_tid));
    try g.bindExternal(b_in, @intCast(b_tid));

    const c = try g.addMatMul(a_in, b_in, 1.0, 0.0);
    // Request f16->f32 promotion by constraining the matmul output dtype.
    g.values.items[@intCast(c)].dtype = .f32;

    const out = try g.addReduce(.sum, c);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    // Naive reference: matmul in f32 then sum.
    var expected_sum: f32 = 0.0;
    for (0..m) |i| {
        for (0..n) |j| {
            var acc: f32 = 0.0;
            for (0..k) |kk| {
                const av: f32 = @floatCast(a_vals[i * k + kk]);
                const bv: f32 = @floatCast(b_vals[kk * n + j]);
                acc += av * bv;
            }
            expected_sum += acc;
        }
    }

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();

    try backend.executeProgram(&prog, sm.tensorStore());

    var out_buf: [4]u8 = .{ 0, 0, 0, 0 };
    try sm.readToPackedScalar(prog.outputs[0], out_buf[0..4]);
    const out_val: f32 = @as(*align(1) const f32, @ptrCast(out_buf[0..4].ptr)).*;
    try std.testing.expect(std.math.isFinite(out_val));
    // f16 inputs may accumulate slightly differently depending on kernel, but should be very close.
    try std.testing.expectApproxEqAbs(expected_sum, out_val, 5e-3);
}

test "graph: matmul rejects mismatched non-quant dtypes" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const m: usize = 2;
    const k: usize = 3;
    const n: usize = 4;

    const a_bytes_len: usize = m * k * 2;
    const b_bytes_len: usize = k * n * 4;

    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const a_tid = try sm.createTiledTensor(.f16, &[_]usize{ m, k }, &[_]usize{ 2, 2 }, .{ .tile_alignment = 64 });
    const b_tid = try sm.createTiledTensor(.f32, &[_]usize{ k, n }, &[_]usize{ 2, 2 }, .{ .tile_alignment = 64 });

    // Contents are irrelevant; compilation should fail before execution.
    var a_zero: [a_bytes_len]u8 = [_]u8{0} ** a_bytes_len;
    var b_zero: [b_bytes_len]u8 = [_]u8{0} ** b_bytes_len;
    try sm.writeFromPackedScalar(a_tid, a_zero[0..]);
    try sm.writeFromPackedScalar(b_tid, b_zero[0..]);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const a_in = try g.addInput(.f16, &[_]usize{ m, k });
    const b_in = try g.addInput(.f32, &[_]usize{ k, n });
    try g.bindExternal(a_in, @intCast(a_tid));
    try g.bindExternal(b_in, @intCast(b_tid));

    const c = try g.addMatMul(a_in, b_in, 1.0, 0.0);
    try g.setOutputs(&[_]graph_mod.ValueId{c});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 4, .tile_alignment = 64 };
    try std.testing.expectError(infer_mod.InferError.DTypeMismatch, program.compileGraph(allocator, &g, &sm, policy));
}

test "graph: view ops lower to materialization (transpose/slice/reshape)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    const rows: usize = 3;
    const cols: usize = 4;

    // Packed input.
    const x_bytes_len: usize = rows * cols * 4;
    const x_buf: []u8 = try allocator.alloc(u8, x_bytes_len);
    defer allocator.free(x_buf);

    const x_vals: []align(1) f32 = @ptrCast(x_buf);
    for (0..rows * cols) |i| x_vals[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 6)) * 0.25;

    var cpu = cpu_backend_mod.CpuBackend.init(allocator);
    defer cpu.deinit();
    const backend: Backend = cpu.backend();

    // Expected: transpose -> slice -> reshape -> relu -> sum.
    // Slice is on the transposed tensor: take rows [1..3) and cols [0..3).
    var expected_sum: f32 = 0.0;
    // This corresponds to x[row=cc, col=1+rr] for rr in 0..2, cc in 0..3? see derivation.
    // Here slice shape is [2,3] (len0=2,len1=3).
    {
        // Elements in reshape order: s[0,0..2], s[1,0..2]
        const coords = [_][2]usize{
            .{ 0, 1 }, .{ 1, 1 }, .{ 2, 1 },
            .{ 0, 2 }, .{ 1, 2 }, .{ 2, 2 },
        };
        for (coords) |rc| {
            const r: usize = rc[0];
            const c: usize = rc[1];
            const v: f32 = x_vals[r * cols + c];
            expected_sum += if (v > 0) v else 0;
        }
    }

    // Storage + graph.
    var sm = manager_mod.StorageManager.init(allocator);
    defer sm.deinit();

    const x_tid = try sm.createTiledTensor(.f32, &[_]usize{ rows, cols }, &[_]usize{ 2, 2 }, .{ .tile_alignment = 64 });
    try sm.writeFromPackedScalar(x_tid, x_buf);

    var g = graph_mod.Graph.init(allocator);
    defer g.deinit();

    const x_in = try g.addInput(.f32, &[_]usize{ rows, cols });
    try g.bindExternal(x_in, @intCast(x_tid));

    const t = try g.addViewTranspose2D(x_in);
    const s = try g.addViewSlice2D(t, 1, 2, 0, 3);
    const r = try g.addViewReshape(s, &[_]usize{6});
    const u = try g.addRelu(r);
    const out = try g.addReduce(.sum, u);
    try g.setOutputs(&[_]graph_mod.ValueId{out});

    const policy: plan_mod.TilePolicy = .{ .base_square_2d = 2, .base_1d = 4, .tile_alignment = 64 };
    var prog = try program.compileGraph(allocator, &g, &sm, policy);
    defer prog.deinit();
    try backend.executeProgram(&prog, sm.tensorStore());

    var out_buf: [4]u8 = .{ 0, 0, 0, 0 };
    try sm.readToPackedScalar(prog.outputs[0], out_buf[0..4]);
    const out_val: f32 = @as(*align(1) const f32, @ptrCast(out_buf[0..4].ptr)).*;
    try std.testing.expect(@abs(out_val - expected_sum) <= 1e-5);
}
