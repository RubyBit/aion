const simd = @import("simd.zig");
const types = @import("../../types.zig");

pub const Mode = enum { layernorm, rmsnorm };

pub fn accumulateStats(sum: []f32, sumsq: []f32, xv: types.BufferViewConst) void {
    const m_tile: usize = xv.layout.shape[0];
    const n_tile: usize = xv.layout.shape[1];
    const lanes: usize = comptime simd.lanesF32();
    const VecF = @Vector(lanes, f32);

    const m_len: usize = @min(m_tile, sum.len);

    var r: usize = 0;
    while (r < m_len) : (r += 1) {
        var acc_sum_v: VecF = @splat(@as(f32, 0.0));
        var acc_sq_v: VecF = @splat(@as(f32, 0.0));

        var c: usize = 0;
        const vec_end: usize = n_tile - (n_tile % lanes);

        if (xv.dtype == .f32) {
            const xs: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, xv.bytes);
            const off: usize = r * n_tile;
            while (c < vec_end) : (c += lanes) {
                const v: VecF = @as(*align(1) const VecF, @ptrCast(xs.ptr + off + c)).*;
                acc_sum_v += v;
                acc_sq_v += v * v;
            }
            var acc_sum: f32 = @reduce(.Add, acc_sum_v);
            var acc_sq: f32 = @reduce(.Add, acc_sq_v);
            while (c < n_tile) : (c += 1) {
                const v: f32 = xs[off + c];
                acc_sum += v;
                acc_sq += v * v;
            }
            sum[r] += acc_sum;
            sumsq[r] += acc_sq;
        } else {
            const xs: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, xv.bytes);
            const VecH = @Vector(lanes, f16);
            const off: usize = r * n_tile;
            while (c < vec_end) : (c += lanes) {
                const vh: VecH = @as(*align(1) const VecH, @ptrCast(xs.ptr + off + c)).*;
                const v: VecF = @floatCast(vh);
                acc_sum_v += v;
                acc_sq_v += v * v;
            }
            var acc_sum: f32 = @reduce(.Add, acc_sum_v);
            var acc_sq: f32 = @reduce(.Add, acc_sq_v);
            while (c < n_tile) : (c += 1) {
                const v: f32 = @floatCast(xs[off + c]);
                acc_sum += v;
                acc_sq += v * v;
            }
            sum[r] += acc_sum;
            sumsq[r] += acc_sq;
        }
    }
}

pub fn applyNorm(
    mode: Mode,
    mean: []const f32,
    inv: []const f32,
    ov: types.BufferViewMut,
    xv: types.BufferViewConst,
    gv: types.BufferViewConst,
    bv: types.BufferViewConst,
) void {
    const m_tile: usize = ov.layout.shape[0];
    const n_tile: usize = ov.layout.shape[1];
    const lanes: usize = comptime simd.lanesF32();
    const VecF = @Vector(lanes, f32);

    const m_len: usize = @min(m_tile, @min(mean.len, inv.len));

    if (ov.dtype == .f32) {
        var out_s: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, ov.bytes);
        const x_s: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, xv.bytes);
        const g_s: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, gv.bytes);
        const b_s: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, bv.bytes);

        var rr: usize = 0;
        while (rr < m_len) : (rr += 1) {
            const mu: f32 = mean[rr];
            const inv0: f32 = inv[rr];
            const mu_v: VecF = @splat(mu);
            const inv_v: VecF = @splat(inv0);

            const off: usize = rr * n_tile;
            var c: usize = 0;
            const vec_end: usize = n_tile - (n_tile % lanes);
            while (c < vec_end) : (c += lanes) {
                const xv0: VecF = @as(*align(1) const VecF, @ptrCast(x_s.ptr + off + c)).*;
                const gv0: VecF = @as(*align(1) const VecF, @ptrCast(g_s.ptr + c)).*;
                const bv0: VecF = @as(*align(1) const VecF, @ptrCast(b_s.ptr + c)).*;
                const norm: VecF = if (mode == .layernorm) (xv0 - mu_v) * inv_v else xv0 * inv_v;
                @as(*align(1) VecF, @ptrCast(out_s.ptr + off + c)).* = (norm * gv0) + bv0;
            }
            while (c < n_tile) : (c += 1) {
                const x0: f32 = x_s[off + c];
                const norm: f32 = if (mode == .layernorm) (x0 - mu) * inv0 else x0 * inv0;
                out_s[off + c] = (norm * g_s[c]) + b_s[c];
            }
        }
    } else {
        var out_s: []align(1) f16 = simd.bytesAsSliceMutUnaligned(f16, ov.bytes);
        const x_s: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, xv.bytes);
        const g_s: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, gv.bytes);
        const b_s: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, bv.bytes);

        const VecH = @Vector(lanes, f16);

        var rr: usize = 0;
        while (rr < m_len) : (rr += 1) {
            const mu: f32 = mean[rr];
            const inv0: f32 = inv[rr];
            const mu_v: VecF = @splat(mu);
            const inv_v: VecF = @splat(inv0);

            const off: usize = rr * n_tile;
            var c: usize = 0;
            const vec_end: usize = n_tile - (n_tile % lanes);
            while (c < vec_end) : (c += lanes) {
                const xh: VecH = @as(*align(1) const VecH, @ptrCast(x_s.ptr + off + c)).*;
                const gh: VecH = @as(*align(1) const VecH, @ptrCast(g_s.ptr + c)).*;
                const bh: VecH = @as(*align(1) const VecH, @ptrCast(b_s.ptr + c)).*;

                const xv0: VecF = @floatCast(xh);
                const gv0: VecF = @floatCast(gh);
                const bv0: VecF = @floatCast(bh);

                const norm: VecF = if (mode == .layernorm) (xv0 - mu_v) * inv_v else xv0 * inv_v;
                const y: VecF = (norm * gv0) + bv0;
                @as(*align(1) VecH, @ptrCast(out_s.ptr + off + c)).* = @floatCast(y);
            }
            while (c < n_tile) : (c += 1) {
                const x0: f32 = @floatCast(x_s[off + c]);
                const norm: f32 = if (mode == .layernorm) (x0 - mu) * inv0 else x0 * inv0;
                const y: f32 = (norm * @as(f32, @floatCast(g_s[c]))) + @as(f32, @floatCast(b_s[c]));
                out_s[off + c] = @floatCast(y);
            }
        }
    }
}
