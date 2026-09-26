// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");
const exec_utils = @import("utils.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;
const DType = types.DType;
const QuantBlockOrder = types.QuantBlockOrder;

/// q8_0 block: 2-byte f16 scale + 32 i8 values, 34 bytes total.
const Q8_0_BLOCK_ELEMS: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 34;

/// ONNX index rules: `[-len, len-1]` is valid and a negative index counts from
/// the end; anything else is an error rather than a clamp.
pub fn resolveIndex(raw: i32, len: usize) ?usize {
    const l: i64 = @intCast(len);
    var v: i64 = raw;
    if (v < 0) v += l;
    if (v < 0 or v >= l) return null;
    return @intCast(v);
}

fn copyRowVectorized(dst: []u8, src: []const u8) void {
    std.debug.assert(dst.len == src.len);

    const Vec64 = @Vector(64, u8);
    const Vec32 = @Vector(32, u8);
    const Vec16 = @Vector(16, u8);

    const n: usize = dst.len;
    var i: usize = 0;

    while (i + 64 <= n) : (i += 64) {
        const v: Vec64 = @as(*align(1) const Vec64, @ptrCast(src.ptr + i)).*;
        @as(*align(1) Vec64, @ptrCast(dst.ptr + i)).* = v;
    }
    while (i + 32 <= n) : (i += 32) {
        const v: Vec32 = @as(*align(1) const Vec32, @ptrCast(src.ptr + i)).*;
        @as(*align(1) Vec32, @ptrCast(dst.ptr + i)).* = v;
    }
    while (i + 16 <= n) : (i += 16) {
        const v: Vec16 = @as(*align(1) const Vec16, @ptrCast(src.ptr + i)).*;
        @as(*align(1) Vec16, @ptrCast(dst.ptr + i)).* = v;
    }
    while (i < n) : (i += 1) {
        dst[i] = src[i];
    }
}

/// Dequantize row `r` of a q8_0 table, `d` elements, into `dst` scalar bytes.
/// Each block is read through the table's block order, so a row-major and a
/// grouped table read the same way. `dst` holds `.f16` or `.f32`.
fn dequantQ8_0Row(dst: []u8, table: []const u8, order: QuantBlockOrder, r: usize, d: usize, out_dtype: DType) BackendError!void {
    if ((d % Q8_0_BLOCK_ELEMS) != 0) return BackendError.InvalidArgument;
    const blocks: usize = d / Q8_0_BLOCK_ELEMS;
    const g = order.groupRows();
    if (table.len < (r / g + 1) * g * blocks * Q8_0_BLOCK_BYTES) return BackendError.InvalidArgument;
    const elem_bytes: usize = switch (out_dtype) {
        .f16 => 2,
        .f32 => 4,
        else => return BackendError.InvalidArgument,
    };
    if (dst.len < d * elem_bytes) return BackendError.InvalidArgument;

    var block: [Q8_0_BLOCK_BYTES]u8 = undefined;
    for (0..blocks) |b| {
        order.loadBlock(table, blocks, r, b, &block);
        const scale: f32 = @as(f16, @bitCast(std.mem.readInt(u16, block[0..2], .little)));
        const q: *const [Q8_0_BLOCK_ELEMS]i8 = @ptrCast(block[2..]);
        for (q, 0..) |v, i| {
            // Through f32, so an f16 output rounds once rather than overflowing midway.
            const x: f32 = @as(f32, @floatFromInt(v)) * scale;
            const e = b * Q8_0_BLOCK_ELEMS + i;
            switch (out_dtype) {
                .f32 => std.mem.bytesAsValue(f32, dst[e * 4 ..][0..4]).* = x,
                else => std.mem.bytesAsValue(f16, dst[e * 2 ..][0..2]).* = @floatCast(x),
            }
        }
    }
}

pub fn execGatherRows(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepGatherRows,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    const out_meta: tensor_store.TensorMeta = try store.meta(s.out);
    const table_meta: tensor_store.TensorMeta = try store.meta(s.table);
    const indices_meta: tensor_store.TensorMeta = try store.meta(s.indices);

    // The compiler is responsible for full validation; keep runtime checks minimal.
    if (out_meta.rank != 3 or table_meta.rank != 2 or indices_meta.rank != 2) return BackendError.InvalidArgument;
    if (indices_meta.dtype != .i32) return BackendError.InvalidArgument;

    // Two legal table-dtype cases:
    //   (a) scalar table, same-dtype output: row is memcpy'd.
    //   (b) q8_0 table (quant_axis=1), output is f16 or f32: row is dequantized on read.
    const table_is_quant: bool = table_meta.dtype == .q8_0;
    if (table_is_quant) {
        if (!(out_meta.dtype == .f16 or out_meta.dtype == .f32)) return BackendError.InvalidArgument;
        if ((table_meta.shape[1] % Q8_0_BLOCK_ELEMS) != 0) return BackendError.InvalidArgument;
    } else {
        if (out_meta.dtype != table_meta.dtype) return BackendError.InvalidArgument;
        if (!(table_meta.dtype == .f16 or table_meta.dtype == .f32)) return BackendError.InvalidArgument;
    }

    const out_elem_bytes: usize = switch (out_meta.dtype) {
        .f16 => 2,
        .f32 => 4,
        else => return BackendError.InvalidArgument,
    };
    // Table-side row stride in bytes. For scalar tables it's D * elem_bytes; for q8_0
    // it's (D/32) * 34 bytes (one contiguous run of blocks per row).
    const table_row_bytes: usize = if (table_is_quant)
        (table_meta.shape[1] / Q8_0_BLOCK_ELEMS) * Q8_0_BLOCK_BYTES
    else
        table_meta.shape[1] * out_elem_bytes;

    const b_total: usize = indices_meta.shape[0];
    const l_total: usize = indices_meta.shape[1];
    const v_total: usize = table_meta.shape[0];

    var idx_v: tensor_store.ViewConst = try store.acquireConst(s.indices);
    defer store.releaseConst(idx_v.token);
    const idx_view = idx_v.bufferView();
    if (idx_view.layout.rank != 2) return BackendError.InvalidArgument;

    if ((idx_view.bytes.len % @sizeOf(i32)) != 0) return BackendError.InvalidArgument;
    const idx_ptr: [*]align(1) const i32 = @ptrCast(idx_view.bytes.ptr);
    const idx_vals: []align(1) const i32 = idx_ptr[0 .. idx_view.bytes.len / @sizeOf(i32)];
    if (idx_vals.len < b_total * l_total) return BackendError.InvalidArgument;

    return gatherRowsWhole(pool, thread_count, s, store, idx_vals, l_total, v_total, table_row_bytes, table_is_quant, table_meta.block_order);
}

/// Whole tensors: the output's `B*L` rows split across the pool, each copied (or
/// dequantized) straight from its table row.
fn gatherRowsWhole(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepGatherRows,
    store: tensor_store.TensorStore,
    idx_vals: []align(1) const i32,
    l_total: usize,
    v_total: usize,
    table_row_bytes: usize,
    table_is_quant: bool,
    block_order: types.QuantBlockOrder,
) ExecuteProgramError!void {
    const out_view = try store.acquireMut(s.out);
    defer store.releaseMut(out_view.token);
    const table_view = try store.acquireConst(s.table);
    defer store.releaseConst(table_view.token);
    const out = out_view.bufferView();
    const td: usize = out.layout.shape[2];
    const Ctx = struct {
        out: []u8,
        out_dtype: types.DType,
        table: []const u8,
        idx: []align(1) const i32,
        v_total: usize,
        td: usize,
        out_row_bytes: usize,
        table_row_bytes: usize,
        table_is_quant: bool,
        block_order: types.QuantBlockOrder,

        fn run(c: @This(), lo: usize, hi: usize, _: usize) BackendError!void {
            for (lo..hi) |r| {
                const row = resolveIndex(c.idx[r], c.v_total) orelse return BackendError.InvalidArgument;
                if (r + 1 < hi) {
                    if (resolveIndex(c.idx[r + 1], c.v_total)) |next| {
                        const pf = if (c.table_is_quant) c.block_order.scaleAt(c.td / Q8_0_BLOCK_ELEMS, next, 0) else next * c.table_row_bytes;
                        if (pf < c.table.len) @prefetch(c.table[pf..].ptr, .{ .rw = .read, .locality = 3, .cache = .data });
                    }
                }
                const dst = c.out[r * c.out_row_bytes ..][0..c.out_row_bytes];
                if (c.table_is_quant) {
                    try dequantQ8_0Row(dst, c.table, c.block_order, row, c.td, c.out_dtype);
                } else {
                    copyRowVectorized(dst, c.table[row * c.table_row_bytes ..][0..c.out_row_bytes]);
                }
            }
        }
    };
    const out_row_bytes = td * out.dtype.info().block_bytes;
    const ctx: Ctx = .{
        .out = out.bytes,
        .out_dtype = out.dtype,
        .table = table_view.bufferView().bytes,
        .idx = idx_vals,
        .v_total = v_total,
        .td = td,
        .out_row_bytes = out_row_bytes,
        .table_row_bytes = table_row_bytes,
        .table_is_quant = table_is_quant,
        .block_order = block_order,
    };
    const rows = out.layout.shape[0] * l_total;
    return exec_utils.parallelRange(BackendError, pool, thread_count, rows, out_row_bytes, ctx, Ctx.run);
}

/// Batched row gather (`axis == 1`, `batch_dims == 1`): the general gather covers it.
pub fn execGather(s: executable.StepGather, store: tensor_store.TensorStore) ExecuteProgramError!void {
    return execGatherND(.{ .out = s.out, .data = s.data, .indices = s.indices, .axis = s.axis, .batch_dims = s.batch_dims }, store);
}

/// General gather: any axis / batch_dims / rank.
///
/// Flattening `data` around `axis` makes the whole family one loop: `lead` walks
/// `data[:axis]`, `pick` walks the index tail, and each pair copies `inner`
/// contiguous elements. Indices repeat across the axes between `batch_dims` and
/// `axis`, which is what `lead / mid` recovers.
pub fn execGatherND(
    s: executable.StepGatherND,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    const out_meta = try store.meta(s.out);
    const data_meta = try store.meta(s.data);
    const idx_meta = try store.meta(s.indices);
    if (idx_meta.dtype != .i32 or out_meta.dtype != data_meta.dtype) return BackendError.InvalidArgument;
    const elem: usize = switch (out_meta.dtype) {
        .f16 => 2,
        .f32 => 4,
        else => return BackendError.InvalidArgument,
    };

    const axis: usize = s.axis;
    const bd: usize = s.batch_dims;
    const dr: usize = data_meta.rank;
    const ir: usize = idx_meta.rank;
    if (axis >= dr or bd > axis or bd > ir) return BackendError.InvalidArgument;

    var batch: usize = 1;
    for (data_meta.shape[0..bd]) |d| batch *= d;
    var lead_total: usize = 1;
    for (data_meta.shape[0..axis]) |d| lead_total *= d;
    var inner: usize = 1;
    for (data_meta.shape[axis + 1 .. dr]) |d| inner *= d;
    var picked: usize = 1;
    for (idx_meta.shape[bd..ir]) |d| picked *= d;
    const axis_len: usize = data_meta.shape[axis];
    if (batch == 0 or lead_total == 0 or axis_len == 0) return BackendError.InvalidArgument;
    const mid: usize = lead_total / batch;

    const idx_v = try store.acquireConst(s.indices);
    defer store.releaseConst(idx_v.token);
    const idx_bytes = idx_v.bufferView().bytes;
    const idx_ptr: [*]align(1) const i32 = @ptrCast(idx_bytes.ptr);
    const idx_vals = idx_ptr[0 .. idx_bytes.len / @sizeOf(i32)];
    if (idx_vals.len < batch * picked) return BackendError.InvalidArgument;

    const data_view = try store.acquireConst(s.data);
    defer store.releaseConst(data_view.token);
    const src_bytes = data_view.bufferView().bytes;
    const out_view = try store.acquireMut(s.out);
    defer store.releaseMut(out_view.token);
    const dst_bytes = out_view.bufferView().bytes;

    const run: usize = inner * elem;
    if (src_bytes.len < lead_total * axis_len * run or dst_bytes.len < lead_total * picked * run) {
        return BackendError.InvalidArgument;
    }

    for (0..lead_total) |lead| {
        const b = lead / mid;
        for (0..picked) |pick| {
            const row = resolveIndex(idx_vals[b * picked + pick], axis_len) orelse return BackendError.InvalidArgument;
            const src = ((lead * axis_len) + row) * run;
            const dst = ((lead * picked) + pick) * run;
            @memcpy(dst_bytes[dst..][0..run], src_bytes[src..][0..run]);
        }
    }
}
