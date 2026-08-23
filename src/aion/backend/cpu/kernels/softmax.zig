// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const fast_math = @import("fast_math.zig");
const simd = @import("simd.zig");
const types = @import("../../types.zig");

const BackendError = types.BackendError;

fn expFastVec(comptime lanes: usize, x: @Vector(lanes, f32)) @Vector(lanes, f32) {
    return fast_math.expApproxVecF32(lanes, fast_math.clampVecF32(lanes, x, -20.0, 0.0));
}

fn expFast(x: f32) f32 {
    return fast_math.expApproxF32(fast_math.clampF32(x, -20.0, 0.0));
}

pub fn updateMaxF32(max_buf: []f32, in_view: types.BufferViewConst, rank: usize) void {
    const in: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, in_view.bytes);
    if (rank == 1) {
        var m: f32 = max_buf[0];
        for (in) |v| m = @max(m, v);
        max_buf[0] = m;
        return;
    }

    const m_tile: usize = in_view.layout.shape[0];
    const n_tile: usize = in_view.layout.shape[1];
    var r: usize = 0;
    while (r < m_tile) : (r += 1) {
        var m: f32 = max_buf[r];
        const off: usize = r * n_tile;
        var c: usize = 0;
        while (c < n_tile) : (c += 1) {
            m = @max(m, in[off + c]);
        }
        max_buf[r] = m;
    }
}

pub fn expSumStoreF32(sum_buf: []f32, out_view: types.BufferViewMut, in_view: types.BufferViewConst, max_buf: []const f32, rank: usize) void {
    var out: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, out_view.bytes);
    const in: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, in_view.bytes);

    const lanes: usize = comptime simd.lanesF32();
    const Vec = @Vector(lanes, f32);

    if (rank == 1) {
        const m: f32 = max_buf[0];
        var acc_v: Vec = @splat(@as(f32, 0.0));

        var i: usize = 0;
        const vec_end: usize = in.len - (in.len % lanes);
        while (i < vec_end) : (i += lanes) {
            const xv: Vec = @as(*align(1) const Vec, @ptrCast(in.ptr + i)).*;
            const ev: Vec = expFastVec(lanes, xv - @as(Vec, @splat(m)));
            @as(*align(1) Vec, @ptrCast(out.ptr + i)).* = ev;
            acc_v += ev;
        }
        var acc: f32 = @reduce(.Add, acc_v);
        while (i < in.len) : (i += 1) {
            const e: f32 = expFast(in[i] - m);
            out[i] = e;
            acc += e;
        }
        sum_buf[0] += acc;
        return;
    }

    const m_tile: usize = in_view.layout.shape[0];
    const n_tile: usize = in_view.layout.shape[1];
    var r: usize = 0;
    while (r < m_tile) : (r += 1) {
        const m: f32 = max_buf[r];
        const off: usize = r * n_tile;

        var acc_v: Vec = @splat(@as(f32, 0.0));
        var c: usize = 0;
        const vec_end: usize = n_tile - (n_tile % lanes);
        while (c < vec_end) : (c += lanes) {
            const xv: Vec = @as(*align(1) const Vec, @ptrCast(in.ptr + off + c)).*;
            const ev: Vec = expFastVec(lanes, xv - @as(Vec, @splat(m)));
            @as(*align(1) Vec, @ptrCast(out.ptr + off + c)).* = ev;
            acc_v += ev;
        }
        var acc: f32 = @reduce(.Add, acc_v);
        while (c < n_tile) : (c += 1) {
            const e: f32 = expFast(in[off + c] - m);
            out[off + c] = e;
            acc += e;
        }
        sum_buf[r] += acc;
    }
}

pub fn normalizeF32(out_view: types.BufferViewMut, sum_buf: []const f32, rank: usize) void {
    var out: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, out_view.bytes);
    const lanes: usize = comptime simd.lanesF32();
    const Vec = @Vector(lanes, f32);

    if (rank == 1) {
        const inv: f32 = 1.0 / sum_buf[0];
        const inv_v: Vec = @splat(inv);
        var i: usize = 0;
        const vec_end: usize = out.len - (out.len % lanes);
        while (i < vec_end) : (i += lanes) {
            const v: Vec = @as(*align(1) const Vec, @ptrCast(out.ptr + i)).*;
            @as(*align(1) Vec, @ptrCast(out.ptr + i)).* = v * inv_v;
        }
        while (i < out.len) : (i += 1) out[i] *= inv;
        return;
    }

    const m_tile: usize = out_view.layout.shape[0];
    const n_tile: usize = out_view.layout.shape[1];
    var r: usize = 0;
    while (r < m_tile) : (r += 1) {
        const inv: f32 = 1.0 / sum_buf[r];
        const inv_v: Vec = @splat(inv);
        const off: usize = r * n_tile;
        var c: usize = 0;
        const vec_end: usize = n_tile - (n_tile % lanes);
        while (c < vec_end) : (c += lanes) {
            const v: Vec = @as(*align(1) const Vec, @ptrCast(out.ptr + off + c)).*;
            @as(*align(1) Vec, @ptrCast(out.ptr + off + c)).* = v * inv_v;
        }
        while (c < n_tile) : (c += 1) out[off + c] *= inv;
    }
}

pub fn updateMaxStridedRowsF32(
    max_buf: []f32,
    in_bytes: []const u8,
    axis_len: usize,
    axis_stride: usize,
    row_offsets: []const usize,
) BackendError!void {
    if (axis_len == 0) return BackendError.InvalidArgument;
    if (row_offsets.len < max_buf.len) return BackendError.InvalidArgument;

    var r: usize = 0;
    while (r < max_buf.len) : (r += 1) {
        const base_off: usize = row_offsets[r];
        var m: f32 = max_buf[r];
        var c: usize = 0;
        while (c < axis_len) : (c += 1) {
            const off: usize = base_off + c * axis_stride;
            const v: f32 = @as(*align(1) const f32, @ptrCast(in_bytes.ptr + off)).*;
            m = @max(m, v);
        }
        max_buf[r] = m;
    }
}

pub fn expSumStoreStridedRowsF32(
    sum_buf: []f32,
    out_bytes: []u8,
    in_bytes: []const u8,
    axis_len: usize,
    axis_stride_in: usize,
    axis_stride_out: usize,
    row_offsets_in: []const usize,
    row_offsets_out: []const usize,
    max_buf: []const f32,
) BackendError!void {
    if (axis_len == 0) return BackendError.InvalidArgument;
    if (sum_buf.len != max_buf.len) return BackendError.InvalidArgument;
    if (row_offsets_in.len < max_buf.len or row_offsets_out.len < max_buf.len) return BackendError.InvalidArgument;

    var r: usize = 0;
    while (r < max_buf.len) : (r += 1) {
        const base_in: usize = row_offsets_in[r];
        const base_out: usize = row_offsets_out[r];
        const m: f32 = max_buf[r];
        var c: usize = 0;
        while (c < axis_len) : (c += 1) {
            const off_in: usize = base_in + c * axis_stride_in;
            const off_out: usize = base_out + c * axis_stride_out;
            const x0: f32 = @as(*align(1) const f32, @ptrCast(in_bytes.ptr + off_in)).*;
            const e: f32 = expFast(x0 - m);
            @as(*align(1) f32, @ptrCast(out_bytes.ptr + off_out)).* = e;
            sum_buf[r] += e;
        }
    }
}

pub fn normalizeStridedRowsF32(
    out_bytes: []u8,
    axis_len: usize,
    axis_stride_out: usize,
    row_offsets_out: []const usize,
    sum_buf: []const f32,
) BackendError!void {
    if (axis_len == 0) return BackendError.InvalidArgument;
    if (row_offsets_out.len < sum_buf.len) return BackendError.InvalidArgument;

    var r: usize = 0;
    while (r < sum_buf.len) : (r += 1) {
        const base_out: usize = row_offsets_out[r];
        const inv: f32 = 1.0 / sum_buf[r];
        var c: usize = 0;
        while (c < axis_len) : (c += 1) {
            const off_out: usize = base_out + c * axis_stride_out;
            const v: f32 = @as(*align(1) const f32, @ptrCast(out_bytes.ptr + off_out)).*;
            @as(*align(1) f32, @ptrCast(out_bytes.ptr + off_out)).* = v * inv;
        }
    }
}

// ---------------------------------------------------------------------------
// f16 softmax
//
// The f32 trio above is a three-pass streaming softmax: max, then exp(x-max)
// STORED INTO THE OUTPUT while summing, then an in-place divide. Storing the
// intermediate is free for f32 because an f32 exp round-trips exactly.
//
// It is not free for f16. `expFast` clamps its argument to [-20, 0], so the
// intermediate spans [2.06e-9, 1] -- and f16 goes subnormal below 6.10e-5 and
// flushes to zero below 5.96e-8. Parking the UN-NORMALIZED exponential in f16
// would quantize (often annihilate) values that are perfectly representable
// once divided by the sum, and no later pass can recover them.
//
// So the f16 path pays one extra exp instead: pass 2 accumulates the sum
// without storing, and pass 3 recomputes exp and stores `exp * (1/sum)`. Every
// element is then rounded to f16 exactly once, on its final value, which is the
// best an f16 output can represent. Maxima and sums stay in f32 throughout, so
// there is no overflow path either: the intermediate never exceeds 1 and the
// sum is an f32 accumulator.
// ---------------------------------------------------------------------------

pub fn updateMaxF16(max_buf: []f32, in_view: types.BufferViewConst, rank: usize) void {
    const in: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, in_view.bytes);
    if (rank == 1) {
        var m: f32 = max_buf[0];
        for (in) |v| m = @max(m, @as(f32, @floatCast(v)));
        max_buf[0] = m;
        return;
    }

    const m_tile: usize = in_view.layout.shape[0];
    const n_tile: usize = in_view.layout.shape[1];
    var r: usize = 0;
    while (r < m_tile) : (r += 1) {
        var m: f32 = max_buf[r];
        const off: usize = r * n_tile;
        var c: usize = 0;
        while (c < n_tile) : (c += 1) m = @max(m, @as(f32, @floatCast(in[off + c])));
        max_buf[r] = m;
    }
}

/// Pass 2 for f16: accumulate sum(exp(x - max)) per row WITHOUT writing the
/// intermediate anywhere. Nothing is stored, so nothing is lost.
pub fn sumExpF16(sum_buf: []f32, in_view: types.BufferViewConst, max_buf: []const f32, rank: usize) void {
    const in: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, in_view.bytes);

    if (rank == 1) {
        const m: f32 = max_buf[0];
        var acc: f32 = 0.0;
        for (in) |v| acc += expFast(@as(f32, @floatCast(v)) - m);
        sum_buf[0] += acc;
        return;
    }

    const m_tile: usize = in_view.layout.shape[0];
    const n_tile: usize = in_view.layout.shape[1];
    var r: usize = 0;
    while (r < m_tile) : (r += 1) {
        const m: f32 = max_buf[r];
        const off: usize = r * n_tile;
        var acc: f32 = 0.0;
        var c: usize = 0;
        while (c < n_tile) : (c += 1) acc += expFast(@as(f32, @floatCast(in[off + c])) - m);
        sum_buf[r] += acc;
    }
}

/// Pass 3 for f16: recompute exp(x - max), scale by 1/sum in f32, and store the
/// finished probability. This is the only write, and the only rounding to f16.
pub fn expNormalizeStoreF16(
    out_view: types.BufferViewMut,
    in_view: types.BufferViewConst,
    max_buf: []const f32,
    sum_buf: []const f32,
    rank: usize,
) void {
    var out: []align(1) f16 = simd.bytesAsSliceMutUnaligned(f16, out_view.bytes);
    const in: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, in_view.bytes);

    if (rank == 1) {
        const m: f32 = max_buf[0];
        const inv: f32 = 1.0 / sum_buf[0];
        var i: usize = 0;
        while (i < in.len) : (i += 1) {
            out[i] = @floatCast(expFast(@as(f32, @floatCast(in[i])) - m) * inv);
        }
        return;
    }

    const m_tile: usize = out_view.layout.shape[0];
    const n_tile: usize = out_view.layout.shape[1];
    var r: usize = 0;
    while (r < m_tile) : (r += 1) {
        const m: f32 = max_buf[r];
        const inv: f32 = 1.0 / sum_buf[r];
        const off: usize = r * n_tile;
        var c: usize = 0;
        while (c < n_tile) : (c += 1) {
            out[off + c] = @floatCast(expFast(@as(f32, @floatCast(in[off + c])) - m) * inv);
        }
    }
}

pub fn updateMaxStridedRowsF16(
    max_buf: []f32,
    in_bytes: []const u8,
    axis_len: usize,
    axis_stride: usize,
    row_offsets: []const usize,
) BackendError!void {
    if (axis_len == 0) return BackendError.InvalidArgument;
    if (row_offsets.len < max_buf.len) return BackendError.InvalidArgument;

    var r: usize = 0;
    while (r < max_buf.len) : (r += 1) {
        const base_off: usize = row_offsets[r];
        var m: f32 = max_buf[r];
        var c: usize = 0;
        while (c < axis_len) : (c += 1) {
            const off: usize = base_off + c * axis_stride;
            const v: f16 = @as(*align(1) const f16, @ptrCast(in_bytes.ptr + off)).*;
            m = @max(m, @as(f32, @floatCast(v)));
        }
        max_buf[r] = m;
    }
}

pub fn sumExpStridedRowsF16(
    sum_buf: []f32,
    in_bytes: []const u8,
    axis_len: usize,
    axis_stride_in: usize,
    row_offsets_in: []const usize,
    max_buf: []const f32,
) BackendError!void {
    if (axis_len == 0) return BackendError.InvalidArgument;
    if (sum_buf.len != max_buf.len) return BackendError.InvalidArgument;
    if (row_offsets_in.len < max_buf.len) return BackendError.InvalidArgument;

    var r: usize = 0;
    while (r < max_buf.len) : (r += 1) {
        const base_in: usize = row_offsets_in[r];
        const m: f32 = max_buf[r];
        var acc: f32 = 0.0;
        var c: usize = 0;
        while (c < axis_len) : (c += 1) {
            const off_in: usize = base_in + c * axis_stride_in;
            const x0: f16 = @as(*align(1) const f16, @ptrCast(in_bytes.ptr + off_in)).*;
            acc += expFast(@as(f32, @floatCast(x0)) - m);
        }
        sum_buf[r] += acc;
    }
}

pub fn expNormalizeStoreStridedRowsF16(
    out_bytes: []u8,
    in_bytes: []const u8,
    axis_len: usize,
    axis_stride_in: usize,
    axis_stride_out: usize,
    row_offsets_in: []const usize,
    row_offsets_out: []const usize,
    max_buf: []const f32,
    sum_buf: []const f32,
) BackendError!void {
    if (axis_len == 0) return BackendError.InvalidArgument;
    if (sum_buf.len != max_buf.len) return BackendError.InvalidArgument;
    if (row_offsets_in.len < max_buf.len or row_offsets_out.len < max_buf.len) return BackendError.InvalidArgument;

    var r: usize = 0;
    while (r < max_buf.len) : (r += 1) {
        const base_in: usize = row_offsets_in[r];
        const base_out: usize = row_offsets_out[r];
        const m: f32 = max_buf[r];
        const inv: f32 = 1.0 / sum_buf[r];
        var c: usize = 0;
        while (c < axis_len) : (c += 1) {
            const off_in: usize = base_in + c * axis_stride_in;
            const off_out: usize = base_out + c * axis_stride_out;
            const x0: f16 = @as(*align(1) const f16, @ptrCast(in_bytes.ptr + off_in)).*;
            const e: f32 = expFast(@as(f32, @floatCast(x0)) - m) * inv;
            @as(*align(1) f16, @ptrCast(out_bytes.ptr + off_out)).* = @floatCast(e);
        }
    }
}
