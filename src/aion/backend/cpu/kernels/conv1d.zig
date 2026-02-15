const std = @import("std");
const simd = @import("simd.zig");

pub const DepthwiseConv1DParams = struct {
    stride: usize,
    dilation: usize,
    pad_left: usize,
};

pub const XTile = struct {
    vals: []align(1) const f32,
    l_mem: usize,
    c_mem: usize,
    row_stride: usize,
};

pub const WTile = struct {
    vals: []align(1) const f32,
    k_mem: usize,
    c_mem: usize,
};

inline fn xIndex(b: usize, lti: usize, cti: usize, x_ltc: usize, x_ctc: usize) usize {
    return (b * x_ltc + lti) * x_ctc + cti;
}

inline fn outIndex(b: usize, lti: usize, cti: usize, out_ltc: usize, out_ctc: usize) usize {
    return (b * out_ltc + lti) * out_ctc + cti;
}

/// Inner depthwise Conv1D kernel.
///
/// This is intentionally "setup-free": it assumes the caller has already validated
/// shapes/tiling and acquired/cached all required tiles.
pub const DepthwiseConv1DTask = struct {
    p: DepthwiseConv1DParams,

    // Shapes.
    batch: usize,
    l_in: usize,
    l_out: usize,
    c: usize,
    k: usize,

    // Tiles.
    out_ltc: usize,
    out_ctc: usize,
    out_tl: usize,
    out_tc: usize,
    out_l_mems: []const usize,

    x_ltc: usize,
    x_ctc: usize,
    x_tl: usize,

    x_tiles: []const XTile,
    w_tiles: []const WTile,
    w_ktc: usize,
    w_tk: usize,
    out_tiles_all: [][]align(1) f32,
    bias_slices: [][]align(1) const f32,

    fast_stride1_dil1: bool,

    const lanes: usize = simd.lanesF32();
    const Vec = @Vector(lanes, f32);

    pub fn runItemRange(t: *const @This(), start: usize, end: usize) void {
        @setRuntimeSafety(false);

        const bias_present_local: bool = (t.bias_slices.len != 0);

        var item: usize = start;
        while (item < end) : (item += 1) {
            // Decode item => (b, out_lti, out_cti)
            const items_per_b: usize = t.out_ltc * t.out_ctc;
            const bb: usize = item / items_per_b;
            const rem0: usize = item - bb * items_per_b;
            const out_lti: usize = rem0 / t.out_ctc;
            const cti: usize = rem0 - out_lti * t.out_ctc;
            if (bb >= t.batch) continue;

            const l_mem: usize = t.out_l_mems[bb * t.out_ltc + out_lti];
            if (l_mem == 0) continue;

            const out_vals: []align(1) f32 = t.out_tiles_all[outIndex(bb, out_lti, cti, t.out_ltc, t.out_ctc)];
            const c_mem: usize = out_vals.len / l_mem;

            const bias: []align(1) const f32 = if (bias_present_local) t.bias_slices[cti] else &[_]f32{};

            const l_base: usize = out_lti * t.out_tl;

            if (t.fast_stride1_dil1) {
                // Very hot case (bench): stride=1, dilation=1, X length not tiled, W K not tiled.
                // X tile and W tile are fixed for this (b, out_cti).
                const xt: XTile = t.x_tiles[xIndex(bb, 0, cti, t.x_ltc, t.x_ctc)];
                const wt: WTile = t.w_tiles[cti * t.w_ktc];

                var lo_local: usize = 0;
                while (lo_local < l_mem) : (lo_local += 1) {
                    const lo_abs: usize = l_base + lo_local;
                    if (lo_abs >= t.l_out) break;

                    const in0: usize = lo_abs; // in_nom for kw=0
                    // Interior when all taps in range => in_nom>=pad_left and li<l_in.
                    const interior: bool = (in0 >= t.p.pad_left and (in0 + (t.k - 1)) >= t.p.pad_left and ((in0 + (t.k - 1)) - t.p.pad_left) < t.l_in);

                    const out_row_off: usize = lo_local * c_mem;

                    var c: usize = 0;
                    while (c + lanes <= c_mem) : (c += lanes) {
                        var acc: Vec = @splat(@as(f32, 0.0));

                        if (interior) {
                            // li = (lo_abs + kw) - pad_left
                            const li0: usize = in0 - t.p.pad_left;

                            var kw: usize = 0;
                            while (kw < t.k) : (kw += 1) {
                                const li: usize = li0 + kw;
                                const xp: [*]align(1) const f32 = xt.vals.ptr + (li * xt.row_stride + c);
                                const wp: [*]align(1) const f32 = wt.vals.ptr + (kw * wt.c_mem + c);
                                const xv: Vec = @as(*align(1) const Vec, @ptrCast(xp)).*;
                                const wv: Vec = @as(*align(1) const Vec, @ptrCast(wp)).*;
                                acc += xv * wv;
                            }
                        } else {
                            var kw: usize = 0;
                            while (kw < t.k) : (kw += 1) {
                                const in_nom: usize = lo_abs + kw;
                                if (in_nom < t.p.pad_left) continue;
                                const li: usize = in_nom - t.p.pad_left;
                                if (li >= t.l_in) continue;
                                if (li >= xt.l_mem) continue;

                                const xp: [*]align(1) const f32 = xt.vals.ptr + (li * xt.row_stride + c);
                                const wp: [*]align(1) const f32 = wt.vals.ptr + (kw * wt.c_mem + c);
                                const xv: Vec = @as(*align(1) const Vec, @ptrCast(xp)).*;
                                const wv: Vec = @as(*align(1) const Vec, @ptrCast(wp)).*;
                                acc += xv * wv;
                            }
                        }

                        if (bias_present_local) {
                            const bp: [*]align(1) const f32 = bias.ptr + c;
                            const bv: Vec = @as(*align(1) const Vec, @ptrCast(bp)).*;
                            acc += bv;
                        }

                        const dstp: [*]align(1) f32 = out_vals.ptr + (out_row_off + c);
                        @as(*align(1) Vec, @ptrCast(dstp)).* = acc;
                    }

                    while (c < c_mem) : (c += 1) {
                        var accs: f32 = 0.0;
                        var kw: usize = 0;
                        while (kw < t.k) : (kw += 1) {
                            const in_nom: usize = lo_abs + kw;
                            if (in_nom < t.p.pad_left) continue;
                            const li: usize = in_nom - t.p.pad_left;
                            if (li >= t.l_in or li >= xt.l_mem) continue;

                            const x_idx: usize = li * xt.row_stride + c;
                            const w_idx: usize = kw * wt.c_mem + c;
                            accs += xt.vals[x_idx] * wt.vals[w_idx];
                        }
                        if (bias_present_local and c < bias.len) accs += bias[c];
                        out_vals[out_row_off + c] = accs;
                    }
                }

                continue;
            }

            // Generic path: supports length tiling, K tiling, stride/dilation.
            var lo_local: usize = 0;
            while (lo_local < l_mem) : (lo_local += 1) {
                const lo_abs: usize = l_base + lo_local;
                if (lo_abs >= t.l_out) break;

                const out_row_off: usize = lo_local * c_mem;

                var c: usize = 0;
                while (c + lanes <= c_mem) : (c += lanes) {
                    var acc: Vec = @splat(@as(f32, 0.0));

                    var kw: usize = 0;
                    while (kw < t.k) : (kw += 1) {
                        const in_nom: usize = lo_abs * t.p.stride + kw * t.p.dilation;
                        if (in_nom < t.p.pad_left) continue;
                        const li: usize = in_nom - t.p.pad_left;
                        if (li >= t.l_in) continue;

                        const xlti: usize = li / t.x_tl;
                        if (xlti >= t.x_ltc) continue;

                        const li_l: usize = li - xlti * t.x_tl;
                        const xt: XTile = t.x_tiles[xIndex(bb, xlti, cti, t.x_ltc, t.x_ctc)];
                        if (li_l >= xt.l_mem) continue;
                        if (c + lanes > xt.c_mem) continue;

                        // Weight tile lookup for this kw.
                        const wkti: usize = if (t.w_tk == 0) 0 else (kw / t.w_tk);
                        if (wkti >= t.w_ktc) continue;
                        const kwl: usize = kw - wkti * t.w_tk;
                        const wt: WTile = t.w_tiles[cti * t.w_ktc + wkti];
                        if (kwl >= wt.k_mem) continue;
                        if (c + lanes > wt.c_mem) continue;

                        const xp: [*]align(1) const f32 = xt.vals.ptr + (li_l * xt.row_stride + c);
                        const wp: [*]align(1) const f32 = wt.vals.ptr + (kwl * wt.c_mem + c);
                        const xv: Vec = @as(*align(1) const Vec, @ptrCast(xp)).*;
                        const wv: Vec = @as(*align(1) const Vec, @ptrCast(wp)).*;
                        acc += xv * wv;
                    }

                    if (bias_present_local) {
                        const bp: [*]align(1) const f32 = bias.ptr + c;
                        const bv: Vec = @as(*align(1) const Vec, @ptrCast(bp)).*;
                        acc += bv;
                    }

                    const dstp: [*]align(1) f32 = out_vals.ptr + (out_row_off + c);
                    @as(*align(1) Vec, @ptrCast(dstp)).* = acc;
                }

                while (c < c_mem) : (c += 1) {
                    var accs: f32 = 0.0;
                    var kw: usize = 0;
                    while (kw < t.k) : (kw += 1) {
                        const in_nom: usize = lo_abs * t.p.stride + kw * t.p.dilation;
                        if (in_nom < t.p.pad_left) continue;
                        const li: usize = in_nom - t.p.pad_left;
                        if (li >= t.l_in) continue;

                        const xlti: usize = li / t.x_tl;
                        if (xlti >= t.x_ltc) continue;
                        const li_l: usize = li - xlti * t.x_tl;
                        const xt: XTile = t.x_tiles[xIndex(bb, xlti, cti, t.x_ltc, t.x_ctc)];
                        if (li_l >= xt.l_mem or c >= xt.c_mem) continue;

                        const wkti: usize = if (t.w_tk == 0) 0 else (kw / t.w_tk);
                        if (wkti >= t.w_ktc) continue;
                        const kwl: usize = kw - wkti * t.w_tk;
                        const wt: WTile = t.w_tiles[cti * t.w_ktc + wkti];
                        if (kwl >= wt.k_mem or c >= wt.c_mem) continue;

                        accs += xt.vals[li_l * xt.row_stride + c] * wt.vals[kwl * wt.c_mem + c];
                    }
                    if (bias_present_local and c < bias.len) accs += bias[c];
                    out_vals[out_row_off + c] = accs;
                }
            }
        }
    }

    pub fn runItems(ctx_any: *anyopaque, start: usize, end: usize, _: usize) void {
        const t: *@This() = @ptrCast(@alignCast(ctx_any));
        t.runItemRange(start, end);
    }
};
