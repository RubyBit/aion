// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const fast_math = @import("fast_math.zig");
const simd = @import("simd.zig");
const types = @import("../../types.zig");

const BackendError = types.BackendError;

/// One position's rotation pairs, held while every head reuses them.
///
/// The angles depend on the position alone, so computing them per head repeated
/// a transcendental 36 times over for a 32-head layer. Chunked so the table is a
/// fixed stack cost whatever the head width.
const TABLE_PAIRS: usize = 64;

const Rotation = struct {
    cos: [TABLE_PAIRS]f32 = undefined,
    sin: [TABLE_PAIRS]f32 = undefined,

    /// Fills `n` pairs from `freq`, returning the frequency the next chunk starts at.
    fn fill(self: *Rotation, pos: f32, freq: f32, freq_step: f32, n: usize) f32 {
        var f: f32 = freq;
        for (0..n) |i| {
            const sc: fast_math.SinCosF32 = fast_math.sinCosFastF32(pos * f);
            self.cos[i] = sc.cos;
            self.sin[i] = sc.sin;
            f *= freq_step;
        }
        return f;
    }
};

pub fn runTileF32(
    out_view: types.BufferViewMut,
    x_view: types.BufferViewConst,
    pos_view: types.BufferViewConst,
    pairs_total: usize,
    rope_pairs: usize,
    freq_step: f32,
    scale_factor: f32,
) BackendError!void {
    if ((out_view.bytes.len % @sizeOf(f32)) != 0 or (x_view.bytes.len % @sizeOf(f32)) != 0) return BackendError.InvalidArgument;
    if ((pos_view.bytes.len % @sizeOf(i32)) != 0) return BackendError.InvalidArgument;

    var out_vals: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, out_view.bytes);
    const x_vals: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, x_view.bytes);
    const pos_vals: []align(1) const i32 = simd.bytesAsSliceConstUnaligned(i32, pos_view.bytes);

    if (x_vals.len != out_vals.len) return BackendError.InvalidArgument;

    const tb: usize = out_view.layout.shape[0];
    const tl: usize = out_view.layout.shape[1];
    const tn: usize = out_view.layout.shape[2];
    const th: usize = out_view.layout.shape[3];

    const pb: usize = pos_view.layout.shape[0];
    const pl: usize = pos_view.layout.shape[1];
    if (pb != tb or pl != tl) return BackendError.InvalidArgument;
    if (pos_vals.len < pb * pl) return BackendError.InvalidArgument;

    var lb: usize = 0;
    while (lb < tb) : (lb += 1) {
        var ll: usize = 0;
        while (ll < tl) : (ll += 1) {
            const pos: f32 = @floatFromInt(pos_vals[lb * pl + ll]);
            const base: usize = ((lb * tl + ll) * tn) * th;

            var ln: usize = 0;
            while (ln < tn) : (ln += 1) {
                const row_off: usize = base + ln * th;
                @memcpy(out_vals[row_off .. row_off + th], x_vals[row_off .. row_off + th]);
            }

            if (rope_pairs == 0) continue;

            var rot: Rotation = .{};
            var freq: f32 = scale_factor;
            var pair0: usize = 0;
            while (pair0 < rope_pairs) : (pair0 += TABLE_PAIRS) {
                const n: usize = @min(TABLE_PAIRS, rope_pairs - pair0);
                freq = rot.fill(pos, freq, freq_step, n);

                ln = 0;
                while (ln < tn) : (ln += 1) {
                    const row_off: usize = base + ln * th;
                    const x_row = x_vals[row_off .. row_off + th];
                    const out_row = out_vals[row_off .. row_off + th];
                    for (0..n) |i| {
                        const xl: f32 = @as(f32, x_row[pair0 + i]);
                        const xr: f32 = @as(f32, x_row[pairs_total + pair0 + i]);
                        out_row[pair0 + i] = @as(f32, xl * rot.cos[i] - xr * rot.sin[i]);
                        out_row[pairs_total + pair0 + i] = @as(f32, xl * rot.sin[i] + xr * rot.cos[i]);
                    }
                }
            }
        }
    }
}

pub fn runTileF16(
    out_view: types.BufferViewMut,
    x_view: types.BufferViewConst,
    pos_view: types.BufferViewConst,
    pairs_total: usize,
    rope_pairs: usize,
    freq_step: f32,
    scale_factor: f32,
) BackendError!void {
    if ((out_view.bytes.len % @sizeOf(f16)) != 0 or (x_view.bytes.len % @sizeOf(f16)) != 0) return BackendError.InvalidArgument;
    if ((pos_view.bytes.len % @sizeOf(i32)) != 0) return BackendError.InvalidArgument;

    var out_vals: []align(1) f16 = simd.bytesAsSliceMutUnaligned(f16, out_view.bytes);
    const x_vals: []align(1) const f16 = simd.bytesAsSliceConstUnaligned(f16, x_view.bytes);
    const pos_vals: []align(1) const i32 = simd.bytesAsSliceConstUnaligned(i32, pos_view.bytes);

    if (x_vals.len != out_vals.len) return BackendError.InvalidArgument;

    const tb: usize = out_view.layout.shape[0];
    const tl: usize = out_view.layout.shape[1];
    const tn: usize = out_view.layout.shape[2];
    const th: usize = out_view.layout.shape[3];

    const pb: usize = pos_view.layout.shape[0];
    const pl: usize = pos_view.layout.shape[1];
    if (pb != tb or pl != tl) return BackendError.InvalidArgument;
    if (pos_vals.len < pb * pl) return BackendError.InvalidArgument;

    var lb: usize = 0;
    while (lb < tb) : (lb += 1) {
        var ll: usize = 0;
        while (ll < tl) : (ll += 1) {
            const pos: f32 = @floatFromInt(pos_vals[lb * pl + ll]);
            const base: usize = ((lb * tl + ll) * tn) * th;

            var ln: usize = 0;
            while (ln < tn) : (ln += 1) {
                const row_off: usize = base + ln * th;
                @memcpy(out_vals[row_off .. row_off + th], x_vals[row_off .. row_off + th]);
            }

            if (rope_pairs == 0) continue;

            var rot: Rotation = .{};
            var freq: f32 = scale_factor;
            var pair0: usize = 0;
            while (pair0 < rope_pairs) : (pair0 += TABLE_PAIRS) {
                const n: usize = @min(TABLE_PAIRS, rope_pairs - pair0);
                freq = rot.fill(pos, freq, freq_step, n);

                ln = 0;
                while (ln < tn) : (ln += 1) {
                    const row_off: usize = base + ln * th;
                    const x_row = x_vals[row_off .. row_off + th];
                    const out_row = out_vals[row_off .. row_off + th];
                    for (0..n) |i| {
                        const xl: f32 = @floatCast(x_row[pair0 + i]);
                        const xr: f32 = @floatCast(x_row[pairs_total + pair0 + i]);
                        out_row[pair0 + i] = @floatCast(xl * rot.cos[i] - xr * rot.sin[i]);
                        out_row[pairs_total + pair0 + i] = @floatCast(xl * rot.sin[i] + xr * rot.cos[i]);
                    }
                }
            }
        }
    }
}
