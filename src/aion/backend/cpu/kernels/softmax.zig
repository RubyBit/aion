const fast_math = @import("fast_math.zig");
const simd = @import("simd.zig");
const types = @import("../../types.zig");

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
