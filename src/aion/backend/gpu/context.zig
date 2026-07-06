// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! `Ctx` — the bundle of backend handles an op's exec code needs, passed by value
//! per `executeProgram`. Lets per-op modules (`exec/simple_ops.zig`, `exec/matmul.zig`)
//! stay decoupled from `GpuBackend` (no import of `backend.zig`, no circular dep):
//! the backend builds a `Ctx` from its fields and hands it to each op handler.

const std = @import("std");
const wgpu = @import("wgpu.zig");
const wgpu_dm = @import("device_memory.zig");
const pipelines = @import("pipelines.zig");
const resident = @import("../../runtime/residency/resident_store.zig");
const tensor_store = @import("../../runtime/tensor_store.zig");

pub const Ctx = struct {
    gpu: *wgpu.Gpu,
    devmem: *wgpu_dm.WgpuDeviceMemory,
    pipes: *pipelines.Pipelines,
    allocator: std.mem.Allocator,
    rstore: *resident.ResidentTensorStore,
    /// Shared grow-only device scratch for multi-stage kernels (two-stage
    /// reductions, split-K attention partials). Safe to reuse across steps in
    /// one frame: dispatches in a pass are ordered, and each multi-stage op
    /// consumes its partials before the next op overwrites them.
    scratch: *ScratchPool,
};

/// Grow-only pooled device buffer (same pattern as MatmulNt's dequant scratch,
/// which stays separate because matmul interleaves scratch use across N tiles).
pub const ScratchPool = struct {
    buf: ?wgpu.c.WGPUBuffer = null,
    cap: u64 = 0,

    pub fn deinit(self: *ScratchPool) void {
        if (self.buf) |b| wgpu.c.wgpuBufferRelease(b);
        self.* = undefined;
    }

    /// Get a storage buffer of at least `bytes` (MiB-rounded, monotonic). The
    /// previous buffer (if outgrown) is released — recorded bind groups keep it
    /// alive until their frame completes.
    pub fn ensure(self: *ScratchPool, gpu: *wgpu.Gpu, bytes: u64) error{ExecutionFailed}!wgpu.c.WGPUBuffer {
        if (self.buf) |b| {
            if (self.cap >= bytes) return b;
            wgpu.c.wgpuBufferRelease(b);
            self.buf = null;
            self.cap = 0;
        }
        const MiB: u64 = 1024 * 1024;
        const cap = (bytes + MiB - 1) / MiB * MiB;
        const b = wgpu.createBuffer(gpu.device, cap, wgpu.c.WGPUBufferUsage_Storage) catch return error.ExecutionFailed;
        self.buf = b;
        self.cap = cap;
        return b;
    }
};

/// Integer ceil-div (workgroup-count helper), shared by op handlers.
pub fn ceilDiv(a: u32, b: u32) u32 {
    return (a + b - 1) / b;
}

/// WebGPU caps workgroups per grid dimension at 65535. 1D elementwise kernels
/// are grid-strided, so we dispatch at most this many groups and let each
/// thread loop; row-per-workgroup kernels must check rows against the cap.
pub const MAX_GROUPS_1D: u32 = 32768;
pub const MAX_GROUPS_PER_DIM: u32 = 65535;

pub fn storageBindingFits(ctx: Ctx, bytes: usize) bool {
    const n = std.math.cast(u64, bytes) orelse return false;
    return n <= ctx.gpu.limits.max_storage_binding_bytes;
}

pub fn totalTiles(meta: tensor_store.TensorMeta) usize {
    var total: usize = 1;
    for (meta.tile_counts) |cnt| total *= cnt;
    return total;
}

/// A device tile's memory seen as `rows` packed-leading rows of `cols` f32
/// elements, `row_stride` elements apart. Row-wise kernels (softmax, norms,
/// reductions) index `base = row * row_stride` and sweep `cols`.
pub const RowView = struct { rows: u32, cols: u32, row_stride: u32 };

/// Collapse a device tile view (rank/shape_mem/strides_mem from `TileRefDevice`)
/// into a `RowView`. Returns null when the layout can't be described that way:
/// non-f32-contiguous last dim, negative strides, or leading dims that aren't
/// row-contiguous (so a flat row index would not address them uniformly).
pub fn rowView(rank: u8, shape_mem: []const usize, strides_mem: []const isize) ?RowView {
    const r: usize = rank;
    if (r == 0 or r > shape_mem.len) return null;
    if (strides_mem[r - 1] != @sizeOf(f32)) return null;
    const cols = std.math.cast(u32, shape_mem[r - 1]) orelse return null;
    if (r == 1) return .{ .rows = 1, .cols = cols, .row_stride = 0 };

    const rs = strides_mem[r - 2];
    if (rs < 0 or @rem(rs, @sizeOf(f32)) != 0) return null;
    const row_stride = std.math.cast(u32, @divExact(@as(usize, @intCast(rs)), @sizeOf(f32))) orelse return null;

    var rows: usize = 1;
    var expect: isize = rs;
    var d: usize = r - 1;
    while (d > 0) : (d -= 1) {
        rows = std.math.mul(usize, rows, shape_mem[d - 1]) catch return null;
        if (d >= 2) {
            expect = std.math.mul(isize, expect, @intCast(shape_mem[d - 1])) catch return null;
            if (strides_mem[d - 2] != expect) return null;
        }
    }
    return .{
        .rows = std.math.cast(u32, rows) orelse return null,
        .cols = cols,
        .row_stride = row_stride,
    };
}

/// Like `rowView` but additionally requires the tile fully packed (rows are
/// exactly `cols` apart), returning the total element count. Kernels that use a
/// flat index (broadcast's `i % cols`, whole-tensor reduce) need this.
pub fn packedElems(rank: u8, shape_mem: []const usize, strides_mem: []const isize) ?u32 {
    const rv = rowView(rank, shape_mem, strides_mem) orelse return null;
    if (rv.rows > 1 and rv.row_stride != rv.cols) return null;
    return std.math.mul(u32, rv.rows, rv.cols) catch null;
}

/// `packedElems` for arbitrary scalar sizes (f16 caches, i32 indices): require
/// a fully packed row-major layout of `elem_bytes` scalars and return the total
/// element count.
pub fn packedElemsSized(rank: u8, shape_mem: []const usize, strides_mem: []const isize, elem_bytes: usize) ?usize {
    const r: usize = rank;
    if (r == 0 or r > shape_mem.len) return null;
    var expect: isize = @intCast(elem_bytes);
    var total: usize = 1;
    var d: usize = r;
    while (d > 0) : (d -= 1) {
        if (strides_mem[d - 1] != expect) return null;
        total = std.math.mul(usize, total, shape_mem[d - 1]) catch return null;
        expect = std.math.mul(isize, expect, @intCast(shape_mem[d - 1])) catch return null;
    }
    return total;
}

test "rowView collapses packed leading dims" {
    const shape = [_]usize{ 2, 3, 8 };
    const strides = [_]isize{ 96, 32, 4 };
    const rv = rowView(3, &shape, &strides).?;
    try std.testing.expectEqual(@as(u32, 6), rv.rows);
    try std.testing.expectEqual(@as(u32, 8), rv.cols);
    try std.testing.expectEqual(@as(u32, 8), rv.row_stride);
    try std.testing.expectEqual(@as(u32, 48), packedElems(3, &shape, &strides).?);

    // Padded rows: still a valid RowView, but not packed.
    const padded = [_]isize{ 128, 40, 4 };
    try std.testing.expect(rowView(3, &shape, &padded) == null); // leading dim not contiguous
    const shape2 = [_]usize{ 3, 8 };
    const padded2 = [_]isize{ 40, 4 };
    const rv2 = rowView(2, &shape2, &padded2).?;
    try std.testing.expectEqual(@as(u32, 10), rv2.row_stride);
    try std.testing.expect(packedElems(2, &shape2, &padded2) == null);
}
