// ============================================================================
// Tests
// ============================================================================
const std = @import("std");
const backend_mod = @import("../backend.zig");
const types = @import("../types.zig");
const utils = @import("../utils.zig");
const CpuBackend = @import("cpu_backend.zig").CpuBackend;

const Backend = backend_mod.Backend;
const BackendKind = types.BackendKind;
const BackendCaps = types.BackendCaps;
const BackendError = types.BackendError;
const BufferViewConst = types.BufferViewConst;
const BufferViewMut = types.BufferViewMut;
const ElemwiseBinaryOp = types.ElemwiseBinaryOp;
const ReduceOp = types.ReduceOp;
const MatMulParams = types.MatMulParams;
const DType = types.DType;
const Layout = types.Layout;

// ============================================================================
// Test helpers
// ============================================================================

const Packed1D = struct {
    shape: [1]usize,
    strides: [1]isize,

    pub fn init(len: usize, elem_bytes: usize) Packed1D {
        var self: Packed1D = undefined;
        self.shape = .{len};
        self.strides = .{@intCast(elem_bytes)};
        return self;
    }

    pub fn asLayout(self: *const Packed1D) Layout {
        return .{ .rank = 1, .shape = self.shape[0..], .strides_bytes = self.strides[0..] };
    }
};

const Packed2D = struct {
    shape: [2]usize,
    strides: [2]isize,

    pub fn init(rows: usize, cols: usize, elem_bytes: usize) Packed2D {
        var self: Packed2D = undefined;
        self.shape = .{ rows, cols };
        self.strides = .{ @intCast(cols * elem_bytes), @intCast(elem_bytes) };
        return self;
    }

    pub fn asLayout(self: *const Packed2D) Layout {
        return .{ .rank = 2, .shape = self.shape[0..], .strides_bytes = self.strides[0..] };
    }
};

const PackedQuant2D = struct {
    shape: [2]usize,
    strides: [2]isize,

    /// Layout helper for quantized weight matrices in the v0 CPU backend.
    ///
    /// The backend currently treats quant tensors as a flat-packed array of
    /// blocks; strides are only required to be present and non-negative.
    pub fn init(k: usize, n: usize, block_bytes: usize) PackedQuant2D {
        var self: PackedQuant2D = undefined;
        self.shape = .{ k, n };
        self.strides = .{ @intCast(n * block_bytes), @intCast(block_bytes) };
        return self;
    }

    pub fn asLayout(self: *const PackedQuant2D) Layout {
        return .{ .rank = 2, .shape = self.shape[0..], .strides_bytes = self.strides[0..] };
    }
};

fn dtypeOf(comptime T: type) DType {
    return switch (T) {
        f32 => .f32,
        f16 => .f16,
        i8 => .i8,
        else => @compileError("No DType mapping for " ++ @typeName(T)),
    };
}

fn viewConstScalar(comptime T: type, data: []const T, layout: Layout) BufferViewConst {
    return .{ .bytes = std.mem.sliceAsBytes(data), .dtype = dtypeOf(T), .layout = layout };
}

fn viewMutScalar(comptime T: type, data: []T, layout: Layout) BufferViewMut {
    return .{ .bytes = std.mem.sliceAsBytes(data), .dtype = dtypeOf(T), .layout = layout };
}

fn viewConstBytes(dtype: DType, bytes: []const u8, layout: Layout) BufferViewConst {
    return .{ .bytes = bytes, .dtype = dtype, .layout = layout };
}

fn viewMutBytes(dtype: DType, bytes: []u8, layout: Layout) BufferViewMut {
    return .{ .bytes = bytes, .dtype = dtype, .layout = layout };
}

fn expectSliceApproxEqAbs(comptime T: type, expected: []const f32, got: []const T, tol: f32) !void {
    try std.testing.expectEqual(expected.len, got.len);
    var i: usize = 0;
    while (i < expected.len) : (i += 1) {
        try std.testing.expectApproxEqAbs(expected[i], @as(f32, got[i]), tol);
    }
}

test "cpu backend: elemwise add f32" {
    var cpu = CpuBackend.init(std.testing.allocator);
    defer cpu.deinit();
    const be = cpu.backend();

    var l: Packed1D = Packed1D.init(4, @sizeOf(f32));

    var a_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var b_data = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    var c_data = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    const a_view: BufferViewConst = viewConstScalar(f32, a_data[0..], l.asLayout());
    const b_view: BufferViewConst = viewConstScalar(f32, b_data[0..], l.asLayout());
    const c_view: BufferViewMut = viewMutScalar(f32, c_data[0..], l.asLayout());

    try be.elemwiseBinary(.add, c_view, a_view, b_view);

    const expected = [_]f32{ 6.0, 8.0, 10.0, 12.0 };
    try expectSliceApproxEqAbs(f32, expected[0..], c_data[0..], 1e-6);
}

test "cpu backend: elemwise mul f32" {
    var cpu: CpuBackend = CpuBackend.init(std.testing.allocator);
    defer cpu.deinit();
    const be: Backend = cpu.backend();

    var l: Packed1D = Packed1D.init(5, @sizeOf(f32));

    var a_data = [_]f32{ -2.0, -1.0, 0.5, 2.0, 3.0 };
    var b_data = [_]f32{ 10.0, -3.0, 4.0, 0.25, -2.0 };
    var c_data = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0 };

    const a_view: BufferViewConst = viewConstScalar(f32, a_data[0..], l.asLayout());
    const b_view: BufferViewConst = viewConstScalar(f32, b_data[0..], l.asLayout());
    const c_view: BufferViewMut = viewMutScalar(f32, c_data[0..], l.asLayout());

    try be.elemwiseBinary(.mul, c_view, a_view, b_view);

    const expected = [_]f32{ -20.0, 3.0, 2.0, 0.5, -6.0 };
    try expectSliceApproxEqAbs(f32, expected[0..], c_data[0..], 1e-6);
}

test "cpu backend: elemwise sub f32" {
    var cpu: CpuBackend = CpuBackend.init(std.testing.allocator);
    defer cpu.deinit();
    const be: Backend = cpu.backend();

    var l: Packed1D = Packed1D.init(4, @sizeOf(f32));

    var a_data = [_]f32{ 10.0, -1.0, 0.5, 2.0 };
    var b_data = [_]f32{ 3.0, 5.0, -1.5, 2.25 };
    var c_data = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    const a_view: BufferViewConst = viewConstScalar(f32, a_data[0..], l.asLayout());
    const b_view: BufferViewConst = viewConstScalar(f32, b_data[0..], l.asLayout());
    const c_view: BufferViewMut = viewMutScalar(f32, c_data[0..], l.asLayout());

    try be.elemwiseBinary(.sub, c_view, a_view, b_view);

    const expected = [_]f32{ 7.0, -6.0, 2.0, -0.25 };
    try expectSliceApproxEqAbs(f32, expected[0..], c_data[0..], 1e-6);
}

test "cpu backend: elemwise div f32" {
    var cpu: CpuBackend = CpuBackend.init(std.testing.allocator);
    defer cpu.deinit();
    const be: Backend = cpu.backend();

    var l: Packed1D = Packed1D.init(4, @sizeOf(f32));

    var a_data = [_]f32{ 8.0, -9.0, 0.5, 10.0 };
    var b_data = [_]f32{ 2.0, 3.0, -2.0, 4.0 };
    var c_data = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    const a_view: BufferViewConst = viewConstScalar(f32, a_data[0..], l.asLayout());
    const b_view: BufferViewConst = viewConstScalar(f32, b_data[0..], l.asLayout());
    const c_view: BufferViewMut = viewMutScalar(f32, c_data[0..], l.asLayout());

    try be.elemwiseBinary(.div, c_view, a_view, b_view);

    const expected = [_]f32{ 4.0, -3.0, -0.25, 2.5 };
    try expectSliceApproxEqAbs(f32, expected[0..], c_data[0..], 1e-6);
}

test "cpu backend: elemwise add f16" {
    var cpu: CpuBackend = CpuBackend.init(std.testing.allocator);
    defer cpu.deinit();
    const be: Backend = cpu.backend();

    var l: Packed1D = Packed1D.init(4, @sizeOf(f16));

    var a_data = [_]f16{ 1.0, -2.0, 3.5, 0.25 };
    var b_data = [_]f16{ 5.0, 6.0, -7.0, 8.0 };
    var c_data = [_]f16{ 0.0, 0.0, 0.0, 0.0 };

    const a_view: BufferViewConst = viewConstScalar(f16, a_data[0..], l.asLayout());
    const b_view: BufferViewConst = viewConstScalar(f16, b_data[0..], l.asLayout());
    const c_view: BufferViewMut = viewMutScalar(f16, c_data[0..], l.asLayout());

    try be.elemwiseBinary(.add, c_view, a_view, b_view);

    const expected = [_]f32{ 6.0, 4.0, -3.5, 8.25 };
    try expectSliceApproxEqAbs(f16, expected[0..], c_data[0..], 1e-3);
}

test "cpu backend: elemwise mul f16" {
    var cpu: CpuBackend = CpuBackend.init(std.testing.allocator);
    defer cpu.deinit();
    const be: Backend = cpu.backend();

    var l: Packed1D = Packed1D.init(4, @sizeOf(f16));

    var a_data = [_]f16{ -2.0, -1.0, 0.5, 2.0 };
    var b_data = [_]f16{ 10.0, -3.0, 4.0, 0.25 };
    var c_data = [_]f16{ 0.0, 0.0, 0.0, 0.0 };

    const a_view: BufferViewConst = viewConstScalar(f16, a_data[0..], l.asLayout());
    const b_view: BufferViewConst = viewConstScalar(f16, b_data[0..], l.asLayout());
    const c_view: BufferViewMut = viewMutScalar(f16, c_data[0..], l.asLayout());

    try be.elemwiseBinary(.mul, c_view, a_view, b_view);

    const expected = [_]f32{ -20.0, 3.0, 2.0, 0.5 };
    try expectSliceApproxEqAbs(f16, expected[0..], c_data[0..], 1e-3);
}

test "cpu backend: elemwise sub f16" {
    var cpu: CpuBackend = CpuBackend.init(std.testing.allocator);
    defer cpu.deinit();
    const be: Backend = cpu.backend();

    var l: Packed1D = Packed1D.init(4, @sizeOf(f16));

    var a_data = [_]f16{ 10.0, -1.0, 0.5, 2.0 };
    var b_data = [_]f16{ 3.0, 5.0, -1.5, 2.25 };
    var c_data = [_]f16{ 0.0, 0.0, 0.0, 0.0 };

    const a_view: BufferViewConst = viewConstScalar(f16, a_data[0..], l.asLayout());
    const b_view: BufferViewConst = viewConstScalar(f16, b_data[0..], l.asLayout());
    const c_view: BufferViewMut = viewMutScalar(f16, c_data[0..], l.asLayout());

    try be.elemwiseBinary(.sub, c_view, a_view, b_view);

    const expected = [_]f32{ 7.0, -6.0, 2.0, -0.25 };
    try expectSliceApproxEqAbs(f16, expected[0..], c_data[0..], 1e-3);
}

test "cpu backend: elemwise div f16" {
    var cpu: CpuBackend = CpuBackend.init(std.testing.allocator);
    defer cpu.deinit();
    const be: Backend = cpu.backend();

    var l: Packed1D = Packed1D.init(4, @sizeOf(f16));

    var a_data = [_]f16{ 8.0, -9.0, 0.5, 10.0 };
    var b_data = [_]f16{ 2.0, 3.0, -2.0, 4.0 };
    var c_data = [_]f16{ 0.0, 0.0, 0.0, 0.0 };

    const a_view: BufferViewConst = viewConstScalar(f16, a_data[0..], l.asLayout());
    const b_view: BufferViewConst = viewConstScalar(f16, b_data[0..], l.asLayout());
    const c_view: BufferViewMut = viewMutScalar(f16, c_data[0..], l.asLayout());

    try be.elemwiseBinary(.div, c_view, a_view, b_view);

    const expected = [_]f32{ 4.0, -3.0, -0.25, 2.5 };
    try expectSliceApproxEqAbs(f16, expected[0..], c_data[0..], 2e-3);
}

test "cpu backend: bias add f32 (broadcast last dim)" {
    var cpu: CpuBackend = CpuBackend.init(std.testing.allocator);
    defer cpu.deinit();
    const be: Backend = cpu.backend();

    const rows: usize = 3;
    const cols: usize = 4;

    var a_data = [_]f32{
        1,  2,  3,  4,
        10, 20, 30, 40,
        -1, -2, -3, -4,
    };
    var bias_data = [_]f32{ 0.5, -1.0, 2.0, 0.0 };
    var out_data = [_]f32{0} ** (rows * cols);

    var a_l: Packed2D = Packed2D.init(rows, cols, @sizeOf(f32));
    var out_l: Packed2D = Packed2D.init(rows, cols, @sizeOf(f32));
    var bias_l: Packed1D = Packed1D.init(cols, @sizeOf(f32));

    const a_view: BufferViewConst = viewConstScalar(f32, a_data[0..], a_l.asLayout());
    const bias_view: BufferViewConst = viewConstScalar(f32, bias_data[0..], bias_l.asLayout());
    const out_view: BufferViewMut = viewMutScalar(f32, out_data[0..], out_l.asLayout());

    try be.broadcastLastDimBinary(.add, out_view, a_view, bias_view);

    const expected = [_]f32{
        1.5,  1.0,  5.0,  4.0,
        10.5, 19.0, 32.0, 40.0,
        -0.5, -3.0, -1.0, -4.0,
    };
    try expectSliceApproxEqAbs(f32, expected[0..], out_data[0..], 1e-6);
}

test "cpu backend: bias add f16 (broadcast last dim)" {
    var cpu: CpuBackend = CpuBackend.init(std.testing.allocator);
    defer cpu.deinit();
    const be: Backend = cpu.backend();

    const rows: usize = 2;
    const cols: usize = 4;

    var a_data = [_]f16{ 1, 2, 3, 4, -1, -2, -3, -4 };
    var bias_data = [_]f16{ 0.5, -1.0, 2.0, 0.0 };
    var out_data = [_]f16{0} ** (rows * cols);

    var a_l: Packed2D = Packed2D.init(rows, cols, @sizeOf(f16));
    var out_l: Packed2D = Packed2D.init(rows, cols, @sizeOf(f16));
    var bias_l: Packed1D = Packed1D.init(cols, @sizeOf(f16));

    const a_view: BufferViewConst = viewConstScalar(f16, a_data[0..], a_l.asLayout());
    const bias_view: BufferViewConst = viewConstScalar(f16, bias_data[0..], bias_l.asLayout());
    const out_view: BufferViewMut = viewMutScalar(f16, out_data[0..], out_l.asLayout());

    try be.broadcastLastDimBinary(.add, out_view, a_view, bias_view);

    const expected = [_]f32{ 1.5, 1.0, 5.0, 4.0, -0.5, -3.0, -1.0, -4.0 };
    try expectSliceApproxEqAbs(f16, expected[0..], out_data[0..], 1e-3);
}

test "cpu backend: relu f32" {
    var cpu: CpuBackend = CpuBackend.init(std.testing.allocator);
    defer cpu.deinit();
    const be: Backend = cpu.backend();

    var l: Packed1D = Packed1D.init(6, @sizeOf(f32));
    var a_data = [_]f32{ -2.0, 0.0, 0.25, 3.0, -0.1, 5.5 };
    var out_data = [_]f32{0} ** 6;

    const a_view: BufferViewConst = viewConstScalar(f32, a_data[0..], l.asLayout());
    const out_view: BufferViewMut = viewMutScalar(f32, out_data[0..], l.asLayout());

    try be.relu(out_view, a_view);

    const expected = [_]f32{ 0.0, 0.0, 0.25, 3.0, 0.0, 5.5 };
    try expectSliceApproxEqAbs(f32, expected[0..], out_data[0..], 1e-6);
}

test "cpu backend: relu f16" {
    var cpu: CpuBackend = CpuBackend.init(std.testing.allocator);
    defer cpu.deinit();
    const be: Backend = cpu.backend();

    var l: Packed1D = Packed1D.init(6, @sizeOf(f16));
    var a_data = [_]f16{ -2.0, 0.0, 0.25, 3.0, -0.1, 5.5 };
    var out_data = [_]f16{0} ** 6;

    const a_view: BufferViewConst = viewConstScalar(f16, a_data[0..], l.asLayout());
    const out_view: BufferViewMut = viewMutScalar(f16, out_data[0..], l.asLayout());

    try be.relu(out_view, a_view);

    const expected = [_]f32{ 0.0, 0.0, 0.25, 3.0, 0.0, 5.5 };
    try expectSliceApproxEqAbs(f16, expected[0..], out_data[0..], 1e-3);
}

test "cpu backend: reduce sum/mean f32" {
    var cpu: CpuBackend = CpuBackend.init(std.testing.allocator);
    defer cpu.deinit();
    const be: Backend = cpu.backend();

    var in_l: Packed1D = Packed1D.init(5, @sizeOf(f32));
    var a_data = [_]f32{ 1.0, -2.0, 3.0, 4.0, 0.5 };

    const shape0 = [_]usize{};
    const strides0 = [_]isize{};
    const out_l: Layout = .{ .rank = 0, .shape = shape0[0..], .strides_bytes = strides0[0..] };

    var out_sum: [1]f32 = .{0.0};
    var out_mean: [1]f32 = .{0.0};

    const a_view: BufferViewConst = viewConstScalar(f32, a_data[0..], in_l.asLayout());
    const sum_view: BufferViewMut = viewMutScalar(f32, out_sum[0..], out_l);
    const mean_view: BufferViewMut = viewMutScalar(f32, out_mean[0..], out_l);

    try be.reduce(.sum, sum_view, a_view);
    try be.reduce(.mean, mean_view, a_view);

    try std.testing.expectApproxEqAbs(@as(f32, 6.5), out_sum[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.3), out_mean[0], 1e-6);
}

test "cpu backend: reduce sum/mean f16 -> f32" {
    var cpu: CpuBackend = CpuBackend.init(std.testing.allocator);
    defer cpu.deinit();
    const be: Backend = cpu.backend();

    var in_l: Packed1D = Packed1D.init(4, @sizeOf(f16));
    var a_data = [_]f16{ 1.0, -2.0, 3.0, 4.0 };

    const shape0 = [_]usize{};
    const strides0 = [_]isize{};
    const out_l: Layout = .{ .rank = 0, .shape = shape0[0..], .strides_bytes = strides0[0..] };

    var out_sum: [1]f32 = .{0.0};
    var out_mean: [1]f32 = .{0.0};

    const a_view: BufferViewConst = viewConstScalar(f16, a_data[0..], in_l.asLayout());
    const sum_view: BufferViewMut = viewMutScalar(f32, out_sum[0..], out_l);
    const mean_view: BufferViewMut = viewMutScalar(f32, out_mean[0..], out_l);

    try be.reduce(.sum, sum_view, a_view);
    try be.reduce(.mean, mean_view, a_view);

    try std.testing.expectApproxEqAbs(@as(f32, 6.0), out_sum[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), out_mean[0], 1e-3);
}

test "cpu backend: matmul f32 2x3 @ 3x2" {
    var cpu = CpuBackend.init(std.testing.allocator);
    defer cpu.deinit();
    const be = cpu.backend();

    // A[2,3], B[3,2], C[2,2]
    const m: usize = 2;
    const k_dim: usize = 3;
    const n: usize = 2;

    var a_data = [_]f32{ 1, 2, 3, 4, 5, 6 }; // 2x3
    var b_data = [_]f32{ 7, 8, 9, 10, 11, 12 }; // 3x2
    var c_data = [_]f32{ 0, 0, 0, 0 }; // 2x2

    var a_l: Packed2D = Packed2D.init(m, k_dim, @sizeOf(f32));
    var b_l: Packed2D = Packed2D.init(k_dim, n, @sizeOf(f32));
    var c_l: Packed2D = Packed2D.init(m, n, @sizeOf(f32));

    const a_view: BufferViewConst = viewConstScalar(f32, a_data[0..], a_l.asLayout());
    const b_view: BufferViewConst = viewConstScalar(f32, b_data[0..], b_l.asLayout());
    const c_view: BufferViewMut = viewMutScalar(f32, c_data[0..], c_l.asLayout());

    const params = MatMulParams{ .m = m, .n = n, .k = k_dim };
    try be.matmul(params, c_view, a_view, b_view);

    // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    // C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    // C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    // C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    try std.testing.expectApproxEqAbs(@as(f32, 58.0), c_data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), c_data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 139.0), c_data[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 154.0), c_data[3], 1e-5);
}

test "cpu backend: matmul f16 2x3 @ 3x2 -> f16" {
    var cpu: CpuBackend = CpuBackend.init(std.testing.allocator);
    defer cpu.deinit();
    const be: Backend = cpu.backend();

    const m: usize = 2;
    const k_dim: usize = 3;
    const n: usize = 2;

    var a_data = [_]f16{ 1, 2, 3, 4, 5, 6 }; // 2x3
    var b_data = [_]f16{ 7, 8, 9, 10, 11, 12 }; // 3x2
    var c_data = [_]f16{ 0, 0, 0, 0 }; // 2x2

    var a_l: Packed2D = Packed2D.init(m, k_dim, @sizeOf(f16));
    var b_l: Packed2D = Packed2D.init(k_dim, n, @sizeOf(f16));
    var c_l: Packed2D = Packed2D.init(m, n, @sizeOf(f16));

    const a_view: BufferViewConst = viewConstScalar(f16, a_data[0..], a_l.asLayout());
    const b_view: BufferViewConst = viewConstScalar(f16, b_data[0..], b_l.asLayout());
    const c_view: BufferViewMut = viewMutScalar(f16, c_data[0..], c_l.asLayout());

    const params: MatMulParams = MatMulParams{ .m = m, .n = n, .k = k_dim };
    try be.matmul(params, c_view, a_view, b_view);

    try std.testing.expectApproxEqAbs(@as(f32, 58.0), @as(f32, c_data[0]), 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), @as(f32, c_data[1]), 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 139.0), @as(f32, c_data[2]), 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 154.0), @as(f32, c_data[3]), 1e-2);
}

test "cpu backend: matmul f16 2x3 @ 3x2 -> f32 (mixed precision)" {
    var cpu: CpuBackend = CpuBackend.init(std.testing.allocator);
    defer cpu.deinit();
    const be: Backend = cpu.backend();

    const m: usize = 2;
    const k_dim: usize = 3;
    const n: usize = 2;

    var a_data = [_]f16{ 1, 2, 3, 4, 5, 6 }; // 2x3
    var b_data = [_]f16{ 7, 8, 9, 10, 11, 12 }; // 3x2
    var c_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 }; // non-zero to test beta

    var a_l: Packed2D = Packed2D.init(m, k_dim, @sizeOf(f16));
    var b_l: Packed2D = Packed2D.init(k_dim, n, @sizeOf(f16));
    var c_l: Packed2D = Packed2D.init(m, n, @sizeOf(f32));

    const a_view: BufferViewConst = viewConstScalar(f16, a_data[0..], a_l.asLayout());
    const b_view: BufferViewConst = viewConstScalar(f16, b_data[0..], b_l.asLayout());
    const c_view: BufferViewMut = viewMutScalar(f32, c_data[0..], c_l.asLayout());

    const params: MatMulParams = MatMulParams{ .m = m, .n = n, .k = k_dim, .alpha = 2.0, .beta = 0.5 };
    try be.matmul(params, c_view, a_view, b_view);

    // expected = 2 * (A@B) + 0.5 * cold
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 * 58.0 + 0.5 * 1.0), c_data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 * 64.0 + 0.5 * 2.0), c_data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 * 139.0 + 0.5 * 3.0), c_data[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 * 154.0 + 0.5 * 4.0), c_data[3], 1e-5);
}

fn packQ8_0Block(scale: f16, vals: *const [32]i8, out: *[34]u8) void {
    const scale_bits: [2]u8 = @bitCast(scale);
    out[0] = scale_bits[0];
    out[1] = scale_bits[1];
    var i: usize = 0;
    while (i < 32) : (i += 1) {
        out[2 + i] = @bitCast(vals[i]);
    }
}

fn packQ4_0Block(scale: f16, vals: *const [32]i8, out: *[18]u8) void {
    const scale_bits: [2]u8 = @bitCast(scale);
    out[0] = scale_bits[0];
    out[1] = scale_bits[1];
    // 16 bytes, each containing 2 nibbles: low then high.
    var bi: usize = 0;
    while (bi < 16) : (bi += 1) {
        const lo_i: usize = bi * 2;
        const hi_i: usize = bi * 2 + 1;

        // q4_0 stores (val + 8) in a nibble, so representable range is [-8, 7].
        const lo_s: i8 = vals[lo_i];
        const hi_s: i8 = vals[hi_i];

        const lo_u: u8 = @intCast(@as(i16, lo_s) + 8);
        const hi_u: u8 = @intCast(@as(i16, hi_s) + 8);

        out[2 + bi] = (lo_u & 0x0F) | ((hi_u & 0x0F) << 4);
    }
}

test "cpu backend: matmul f32@q8_0 -> f32" {
    var cpu: CpuBackend = CpuBackend.init(std.testing.allocator);
    defer cpu.deinit();
    const be: Backend = cpu.backend();

    const m: usize = 2;
    const n: usize = 3;
    const k_dim: usize = 32;

    var a_data: [m * k_dim]f32 = undefined;
    var i: usize = 0;
    while (i < a_data.len) : (i += 1) {
        a_data[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) * 0.25;
    }

    // Build quantized B with scale=1.0 and integer values.
    var b_vals: [k_dim * n]i8 = undefined;
    var kk: usize = 0;
    while (kk < k_dim) : (kk += 1) {
        var j: usize = 0;
        while (j < n) : (j += 1) {
            const v: i8 = @as(i8, @intCast(@as(i32, @intCast((kk * 7 + j * 3) % 17)) - 8));
            b_vals[kk * n + j] = v;
        }
    }

    var b_bytes: [n * 34]u8 = undefined;
    var j: usize = 0;
    while (j < n) : (j += 1) {
        var col: [32]i8 = undefined;
        kk = 0;
        while (kk < 32) : (kk += 1) {
            col[kk] = b_vals[kk * n + j];
        }
        packQ8_0Block(@as(f16, 1.0), &col, @as(*[34]u8, @ptrCast(b_bytes[j * 34 ..][0..34])));
    }

    var c_data: [m * n]f32 = .{0} ** (m * n);

    var a_l: Packed2D = Packed2D.init(m, k_dim, @sizeOf(f32));
    var b_l: PackedQuant2D = PackedQuant2D.init(k_dim, n, 34);
    var c_l: Packed2D = Packed2D.init(m, n, @sizeOf(f32));

    const a_view: BufferViewConst = viewConstScalar(f32, a_data[0..], a_l.asLayout());
    const b_view: BufferViewConst = viewConstBytes(.q8_0, b_bytes[0..], b_l.asLayout());
    const c_view: BufferViewMut = viewMutScalar(f32, c_data[0..], c_l.asLayout());

    const params: MatMulParams = MatMulParams{ .m = m, .n = n, .k = k_dim };
    try be.matmul(params, c_view, a_view, b_view);

    // Reference check: C = A @ B (B dequant with scale=1)
    var r_i: usize = 0;
    while (r_i < m) : (r_i += 1) {
        j = 0;
        while (j < n) : (j += 1) {
            var acc: f32 = 0.0;
            kk = 0;
            while (kk < k_dim) : (kk += 1) {
                acc += a_data[r_i * k_dim + kk] * @as(f32, @floatFromInt(b_vals[kk * n + j]));
            }
            try std.testing.expectApproxEqAbs(acc, c_data[r_i * n + j], 1e-4);
        }
    }
}

test "cpu backend: matmul f32@q4_0 -> f32" {
    var cpu: CpuBackend = CpuBackend.init(std.testing.allocator);
    defer cpu.deinit();
    const be: Backend = cpu.backend();

    const m: usize = 2;
    const n: usize = 2;
    const k_dim: usize = 32;

    var a_data: [m * k_dim]f32 = undefined;
    var i: usize = 0;
    while (i < a_data.len) : (i += 1) {
        a_data[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i % 9))) - 4)) * 0.5;
    }

    var b_vals: [k_dim * n]i8 = undefined;
    var kk: usize = 0;
    while (kk < k_dim) : (kk += 1) {
        var j: usize = 0;
        while (j < n) : (j += 1) {
            // Values in [-8, 7]
            const v: i8 = @as(i8, @intCast(@as(i32, @intCast((kk + j * 3) & 0x0F)) - 8));
            b_vals[kk * n + j] = v;
        }
    }

    var b_bytes: [n * 18]u8 = undefined;
    var j: usize = 0;
    while (j < n) : (j += 1) {
        var col: [32]i8 = undefined;
        kk = 0;
        while (kk < 32) : (kk += 1) {
            col[kk] = b_vals[kk * n + j];
        }
        packQ4_0Block(@as(f16, 1.0), &col, @as(*[18]u8, @ptrCast(b_bytes[j * 18 ..][0..18])));
    }

    var c_data: [m * n]f32 = .{0} ** (m * n);

    var a_l: Packed2D = Packed2D.init(m, k_dim, @sizeOf(f32));
    var b_l: PackedQuant2D = PackedQuant2D.init(k_dim, n, 18);
    var c_l: Packed2D = Packed2D.init(m, n, @sizeOf(f32));

    const a_view: BufferViewConst = viewConstScalar(f32, a_data[0..], a_l.asLayout());
    const b_view: BufferViewConst = viewConstBytes(.q4_0, b_bytes[0..], b_l.asLayout());
    const c_view: BufferViewMut = viewMutScalar(f32, c_data[0..], c_l.asLayout());

    const params: MatMulParams = MatMulParams{ .m = m, .n = n, .k = k_dim };
    try be.matmul(params, c_view, a_view, b_view);

    var r_i: usize = 0;
    while (r_i < m) : (r_i += 1) {
        j = 0;
        while (j < n) : (j += 1) {
            var acc: f32 = 0.0;
            kk = 0;
            while (kk < k_dim) : (kk += 1) {
                acc += a_data[r_i * k_dim + kk] * @as(f32, @floatFromInt(b_vals[kk * n + j]));
            }
            try std.testing.expectApproxEqAbs(acc, c_data[r_i * n + j], 1e-3);
        }
    }
}

test "cpu backend: copy works for f32, f16, q8_0" {
    var cpu: CpuBackend = CpuBackend.init(std.testing.allocator);
    defer cpu.deinit();
    const be: Backend = cpu.backend();

    // f32
    {
        var l: Packed1D = Packed1D.init(4, @sizeOf(f32));

        var src = [_]f32{ 1.25, -2.5, 3.75, 4.0 };
        var dst = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

        const src_view: BufferViewConst = viewConstScalar(f32, src[0..], l.asLayout());
        const dst_view: BufferViewMut = viewMutScalar(f32, dst[0..], l.asLayout());
        try be.copy(dst_view, src_view);

        try std.testing.expectEqualSlices(f32, &src, &dst);
    }

    // f16
    {
        var l: Packed1D = Packed1D.init(5, @sizeOf(f16));

        var src = [_]f16{ 1.0, -2.0, 3.5, 0.25, 100.0 };
        var dst = [_]f16{ 0.0, 0.0, 0.0, 0.0, 0.0 };

        const src_view: BufferViewConst = viewConstScalar(f16, src[0..], l.asLayout());
        const dst_view: BufferViewMut = viewMutScalar(f16, dst[0..], l.asLayout());
        try be.copy(dst_view, src_view);

        try std.testing.expectEqualSlices(f16, &src, &dst);
    }

    // q8_0: one block (32 logical elems)
    {
        // For quant tensors, strides are currently ignored by v0 packed checks.
        // Keep stride=1 and rely on requiredByteLen(dtype, shape).
        var l: Packed1D = Packed1D.init(32, 1);

        var src: [34]u8 = undefined;
        var dst: [34]u8 = undefined;

        // Fill with a recognizable pattern.
        var i: usize = 0;
        while (i < src.len) : (i += 1) {
            src[i] = @as(u8, @intCast((i * 37 + 11) & 0xFF));
            dst[i] = 0;
        }

        const src_view: BufferViewConst = viewConstBytes(.q8_0, src[0..], l.asLayout());
        const dst_view: BufferViewMut = viewMutBytes(.q8_0, dst[0..], l.asLayout());
        try be.copy(dst_view, src_view);

        try std.testing.expectEqualSlices(u8, &src, &dst);
    }
}

test "cpu backend: thread pool matches single-thread (elemwise + matmul)" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var cpu_st: CpuBackend = CpuBackend.init(allocator);
    defer cpu_st.deinit();
    const be_st: Backend = cpu_st.backend();

    var cpu_mt: CpuBackend = try CpuBackend.initWithOptions(allocator, .{ .thread_count = 4 });
    defer cpu_mt.deinit();
    const be_mt: Backend = cpu_mt.backend();

    // Make sure we're actually configured to use threads.
    try std.testing.expect(be_mt.caps().threads);

    // ------------------------------------------------------------------------
    // Elemwise: choose a large N to force the parallel path (see CpuBackend heuristic).
    // ------------------------------------------------------------------------
    const n_elem: usize = 20000;
    var a_e: []f32 = try allocator.alloc(f32, n_elem);
    defer allocator.free(a_e);
    var b_e: []f32 = try allocator.alloc(f32, n_elem);
    defer allocator.free(b_e);
    var c_st_e: []f32 = try allocator.alloc(f32, n_elem);
    defer allocator.free(c_st_e);
    var c_mt_e: []f32 = try allocator.alloc(f32, n_elem);
    defer allocator.free(c_mt_e);

    var i: usize = 0;
    while (i < n_elem) : (i += 1) {
        a_e[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i * 17) % 101)) - 50)) * 0.125;
        b_e[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i * 29) % 97)) - 48)) * 0.25;
        c_st_e[i] = 0.0;
        c_mt_e[i] = 0.0;
    }

    var l1: Packed1D = Packed1D.init(n_elem, @sizeOf(f32));
    const a_view_e: BufferViewConst = viewConstScalar(f32, a_e, l1.asLayout());
    const b_view_e: BufferViewConst = viewConstScalar(f32, b_e, l1.asLayout());
    const c_st_view_e: BufferViewMut = viewMutScalar(f32, c_st_e, l1.asLayout());
    const c_mt_view_e: BufferViewMut = viewMutScalar(f32, c_mt_e, l1.asLayout());

    try be_st.elemwiseBinary(.add, c_st_view_e, a_view_e, b_view_e);
    try be_mt.elemwiseBinary(.add, c_mt_view_e, a_view_e, b_view_e);

    i = 0;
    while (i < n_elem) : (i += 1) {
        try std.testing.expectEqual(c_st_e[i], c_mt_e[i]);
    }

    // ------------------------------------------------------------------------
    // ReLU: same buffer sizes should hit the parallel path.
    // ------------------------------------------------------------------------
    try be_st.relu(c_st_view_e, a_view_e);
    try be_mt.relu(c_mt_view_e, a_view_e);

    i = 0;
    while (i < n_elem) : (i += 1) {
        try std.testing.expectEqual(c_st_e[i], c_mt_e[i]);
    }

    // ------------------------------------------------------------------------
    // Bias add: broadcast over last dim.
    // ------------------------------------------------------------------------
    const br: usize = 256;
    const bc: usize = 256;
    var a_b: []f32 = try allocator.alloc(f32, br * bc);
    defer allocator.free(a_b);
    var bias_b: []f32 = try allocator.alloc(f32, bc);
    defer allocator.free(bias_b);
    var out_st_b: []f32 = try allocator.alloc(f32, br * bc);
    defer allocator.free(out_st_b);
    var out_mt_b: []f32 = try allocator.alloc(f32, br * bc);
    defer allocator.free(out_mt_b);

    i = 0;
    while (i < a_b.len) : (i += 1) {
        a_b[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i * 7) % 31)) - 15)) * 0.25;
        out_st_b[i] = 0.0;
        out_mt_b[i] = 0.0;
    }
    i = 0;
    while (i < bias_b.len) : (i += 1) {
        bias_b[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i * 11) % 29)) - 14)) * 0.1;
    }

    var a_bl: Packed2D = Packed2D.init(br, bc, @sizeOf(f32));
    var out_bl: Packed2D = Packed2D.init(br, bc, @sizeOf(f32));
    var bias_bl: Packed1D = Packed1D.init(bc, @sizeOf(f32));

    const a_view_b: BufferViewConst = viewConstScalar(f32, a_b, a_bl.asLayout());
    const bias_view_b: BufferViewConst = viewConstScalar(f32, bias_b, bias_bl.asLayout());
    const out_st_view_b: BufferViewMut = viewMutScalar(f32, out_st_b, out_bl.asLayout());
    const out_mt_view_b: BufferViewMut = viewMutScalar(f32, out_mt_b, out_bl.asLayout());

    try be_st.broadcastLastDimBinary(.add, out_st_view_b, a_view_b, bias_view_b);
    try be_mt.broadcastLastDimBinary(.add, out_mt_view_b, a_view_b, bias_view_b);

    i = 0;
    while (i < out_st_b.len) : (i += 1) {
        try std.testing.expectEqual(out_st_b[i], out_mt_b[i]);
    }

    // ------------------------------------------------------------------------
    // Reduction: deterministic partitioning, but FP association differs from single-thread.
    // Compare with a tight tolerance.
    // ------------------------------------------------------------------------
    const shape0 = [_]usize{};
    const strides0 = [_]isize{};
    const l0: Layout = .{ .rank = 0, .shape = shape0[0..], .strides_bytes = strides0[0..] };
    var rsum_st: [1]f32 = .{0.0};
    var rsum_mt: [1]f32 = .{0.0};
    const rsum_st_view: BufferViewMut = viewMutScalar(f32, rsum_st[0..], l0);
    const rsum_mt_view: BufferViewMut = viewMutScalar(f32, rsum_mt[0..], l0);
    try be_st.reduce(.sum, rsum_st_view, a_view_e);
    try be_mt.reduce(.sum, rsum_mt_view, a_view_e);
    try std.testing.expectApproxEqAbs(rsum_st[0], rsum_mt[0], 1e-5);

    // ------------------------------------------------------------------------
    // Matmul: choose M big enough to force row-parallel path.
    // ------------------------------------------------------------------------
    const m: usize = 16;
    const k_dim: usize = 64;
    const n: usize = 32;

    var a_m: []f32 = try allocator.alloc(f32, m * k_dim);
    defer allocator.free(a_m);
    var b_m: []f32 = try allocator.alloc(f32, k_dim * n);
    defer allocator.free(b_m);
    var c_st_m: []f32 = try allocator.alloc(f32, m * n);
    defer allocator.free(c_st_m);
    var c_mt_m: []f32 = try allocator.alloc(f32, m * n);
    defer allocator.free(c_mt_m);

    i = 0;
    while (i < a_m.len) : (i += 1) {
        a_m[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i * 13) % 37)) - 18)) * 0.1;
    }
    i = 0;
    while (i < b_m.len) : (i += 1) {
        b_m[i] = @as(f32, @floatFromInt(@as(i32, @intCast((i * 19) % 41)) - 20)) * 0.05;
    }
    i = 0;
    while (i < c_st_m.len) : (i += 1) {
        c_st_m[i] = 0.0;
        c_mt_m[i] = 0.0;
    }

    var a_l: Packed2D = Packed2D.init(m, k_dim, @sizeOf(f32));
    var b_l: Packed2D = Packed2D.init(k_dim, n, @sizeOf(f32));
    var c_l: Packed2D = Packed2D.init(m, n, @sizeOf(f32));

    const a_view_m: BufferViewConst = viewConstScalar(f32, a_m, a_l.asLayout());
    const b_view_m: BufferViewConst = viewConstScalar(f32, b_m, b_l.asLayout());
    const c_st_view_m: BufferViewMut = viewMutScalar(f32, c_st_m, c_l.asLayout());
    const c_mt_view_m: BufferViewMut = viewMutScalar(f32, c_mt_m, c_l.asLayout());

    const params: MatMulParams = MatMulParams{ .m = m, .n = n, .k = k_dim };
    try be_st.matmul(params, c_st_view_m, a_view_m, b_view_m);
    try be_mt.matmul(params, c_mt_view_m, a_view_m, b_view_m);

    i = 0;
    while (i < c_st_m.len) : (i += 1) {
        try std.testing.expectApproxEqAbs(c_st_m[i], c_mt_m[i], 1e-6);
    }
}
