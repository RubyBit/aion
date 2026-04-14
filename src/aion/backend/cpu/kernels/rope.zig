const fast_math = @import("fast_math.zig");
const simd = @import("simd.zig");
const types = @import("../../types.zig");

const BackendError = types.BackendError;

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

            var ln: usize = 0;
            while (ln < tn) : (ln += 1) {
                const row_off: usize = (((lb * tl + ll) * tn) + ln) * th;
                const x_row: []align(1) const f32 = x_vals[row_off .. row_off + th];
                const out_row: []align(1) f32 = out_vals[row_off .. row_off + th];

                @memcpy(out_row, x_row);

                if (rope_pairs == 0) continue;

                var i: usize = 0;
                var freq: f32 = scale_factor;
                while (i < rope_pairs) : (i += 1) {
                    const xl: f32 = x_row[i];
                    const xr: f32 = x_row[pairs_total + i];
                    const angle: f32 = pos * freq;
                    const sc: fast_math.SinCosF32 = fast_math.sinCosFastF32(angle);
                    out_row[i] = xl * sc.cos - xr * sc.sin;
                    out_row[pairs_total + i] = xl * sc.sin + xr * sc.cos;
                    freq *= freq_step;
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

            var ln: usize = 0;
            while (ln < tn) : (ln += 1) {
                const row_off: usize = (((lb * tl + ll) * tn) + ln) * th;
                const x_row: []align(1) const f16 = x_vals[row_off .. row_off + th];
                const out_row: []align(1) f16 = out_vals[row_off .. row_off + th];

                @memcpy(out_row, x_row);

                if (rope_pairs == 0) continue;

                var i: usize = 0;
                var freq: f32 = scale_factor;
                while (i < rope_pairs) : (i += 1) {
                    const xl: f32 = @floatCast(x_row[i]);
                    const xr: f32 = @floatCast(x_row[pairs_total + i]);
                    const angle: f32 = pos * freq;
                    const sc: fast_math.SinCosF32 = fast_math.sinCosFastF32(angle);
                    out_row[i] = @floatCast(xl * sc.cos - xr * sc.sin);
                    out_row[pairs_total + i] = @floatCast(xl * sc.sin + xr * sc.cos);
                    freq *= freq_step;
                }
            }
        }
    }
}
