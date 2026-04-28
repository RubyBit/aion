// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const simd = @import("simd.zig");

pub const DepthwiseConv2DParams = struct {
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    pad_top: usize,
    pad_left: usize,
    reflect: bool = false,
};

inline fn reflectIndex(idx_nom: isize, len: isize) isize {
    var x: isize = idx_nom;
    while (x < 0 or x >= len) {
        if (x < 0) x = -x else x = (2 * len - 2) - x;
    }
    return x;
}

pub const XTile = struct {
    vals: []align(1) const f32,
    h_mem: usize,
    w_mem: usize,
    c_mem: usize,
    row_stride: usize, // w_mem * c_mem
};

pub const WTile = struct {
    vals: []align(1) const f32,
    kh_mem: usize,
    kw_mem: usize,
    c_mem: usize,
    kh_stride: usize, // kw_mem * c_mem
};

inline fn xIndex(b: usize, hti: usize, wti: usize, cti: usize, x_htc: usize, x_wtc: usize, x_ctc: usize) usize {
    return ((b * x_htc + hti) * x_wtc + wti) * x_ctc + cti;
}

inline fn outIndex(b: usize, ohti: usize, owti: usize, cti: usize, out_htc: usize, out_wtc: usize, out_ctc: usize) usize {
    return ((b * out_htc + ohti) * out_wtc + owti) * out_ctc + cti;
}

inline fn outHWIndex(b: usize, ohti: usize, owti: usize, out_htc: usize, out_wtc: usize) usize {
    return (b * out_htc + ohti) * out_wtc + owti;
}

/// Depthwise Conv2D kernel tuning knobs (compile-time).
pub const DepthwiseConv2DTuning = struct {
    /// If true, unroll the 3x3 inner loops in the fast stride=1,dilation=1 path.
    ///
    /// This is only used when `t.fast_stride1_dil1` and `t.k_h == 3` and `t.k_w == 3`.
    unroll_3x3: bool = false,
    lanes: usize = simd.lanesF32(),
};

/// Generates a depthwise Conv2D kernel implementation specialized by `tuning`.
pub fn Kernel(comptime tuning: DepthwiseConv2DTuning) type {
    return struct {
        pub fn runItemRange(t: *const DepthwiseConv2DTask, start: usize, end: usize) void {
            DepthwiseConv2DTask.runItemRangeImpl(tuning, t, start, end);
        }

        pub fn runItems(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
            _ = tid;
            const t: *const DepthwiseConv2DTask = @ptrCast(@alignCast(ctx_any));
            runItemRange(t, start, end);
        }
    };
}

/// Inner depthwise Conv2D kernel.
///
/// Setup-free: assumes the caller has already validated shapes/tiling and has
/// acquired all required tiles.
pub const DepthwiseConv2DTask = struct {
    p: DepthwiseConv2DParams,

    // Shapes.
    batch: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
    c: usize,
    k_h: usize,
    k_w: usize,

    // Output tiles.
    out_htc: usize,
    out_wtc: usize,
    out_ctc: usize,
    out_th: usize,
    out_tw: usize,
    out_tc: usize,
    out_h_mems: []const usize, // per (b,ohti,owti)
    out_w_mems: []const usize, // per (b,ohti,owti)

    // X tiling.
    x_htc: usize,
    x_wtc: usize,
    x_ctc: usize,
    x_th: usize,
    x_tw: usize,

    // Cached tiles.
    x_tiles: []const XTile,
    w_tiles: []const WTile, // indexed by (cti * (w_khtc*w_kwtc) + (khti*w_kwtc + kwti))
    w_khtc: usize,
    w_kwtc: usize,
    w_tkh: usize,
    w_tkw: usize,
    out_tiles_all: [][]align(1) f32,
    bias_slices: [][]align(1) const f32, // optional, indexed by cti

    // Optional precomputed maps to avoid division/mod in the hot loop.
    // Only valid for stride=1,dilation=1 paths.
    use_stride1_maps: bool,
    ih_to_xhti: []const u16,
    ih_to_ihl: []const u16,
    iw_to_xwti: []const u16,
    iw_to_iwl: []const u16,

    // Fast-path enable.
    fast_stride1_dil1: bool,

    fn runItemRangeImpl(comptime tuning: DepthwiseConv2DTuning, t: *const @This(), start: usize, end: usize) void {
        @setRuntimeSafety(false);

        const lanes: usize = tuning.lanes;
        const Vec = @Vector(lanes, f32);

        const bias_present_local: bool = (t.bias_slices.len != 0);
        const use_reflect: bool = t.p.reflect;
        const h_in_i: isize = @intCast(t.h_in);
        const w_in_i: isize = @intCast(t.w_in);

        var item: usize = start;
        while (item < end) : (item += 1) {
            // Decode item => (b, ohti, owti, cti)
            const items_per_b: usize = t.out_htc * t.out_wtc * t.out_ctc;
            const bb: usize = item / items_per_b;
            if (bb >= t.batch) continue;
            const rem0: usize = item - bb * items_per_b;

            const items_per_oh: usize = t.out_wtc * t.out_ctc;
            const ohti: usize = rem0 / items_per_oh;
            const rem1: usize = rem0 - ohti * items_per_oh;
            const owti: usize = rem1 / t.out_ctc;
            const cti: usize = rem1 - owti * t.out_ctc;

            if (ohti >= t.out_htc or owti >= t.out_wtc or cti >= t.out_ctc) continue;

            const hw_idx: usize = outHWIndex(bb, ohti, owti, t.out_htc, t.out_wtc);
            const h_mem: usize = t.out_h_mems[hw_idx];
            const w_mem: usize = t.out_w_mems[hw_idx];
            if (h_mem == 0 or w_mem == 0) continue;

            const out_vals: []align(1) f32 = t.out_tiles_all[outIndex(bb, ohti, owti, cti, t.out_htc, t.out_wtc, t.out_ctc)];
            const c_mem: usize = out_vals.len / (h_mem * w_mem);

            const bias: []align(1) const f32 = if (bias_present_local) t.bias_slices[cti] else &[_]f32{};

            const oh_base: usize = ohti * t.out_th;
            const ow_base: usize = owti * t.out_tw;

            if (t.fast_stride1_dil1) {
                // Hot case: stride=1, dilation=1, X spatial dims not tiled and W spatial dims not tiled.
                const xt: XTile = t.x_tiles[xIndex(bb, 0, 0, cti, t.x_htc, t.x_wtc, t.x_ctc)];
                const wt: WTile = t.w_tiles[cti * (t.w_khtc * t.w_kwtc)];

                var oh_local: usize = 0;
                while (oh_local < h_mem) : (oh_local += 1) {
                    const oh_abs: usize = oh_base + oh_local;
                    if (oh_abs >= t.h_out) break;
                    const ih0: isize = @as(isize, @intCast(oh_abs)) - @as(isize, @intCast(t.p.pad_top));

                    var ow_local: usize = 0;
                    while (ow_local < w_mem) : (ow_local += 1) {
                        const ow_abs: usize = ow_base + ow_local;
                        if (ow_abs >= t.w_out) break;
                        const iw0: isize = @as(isize, @intCast(ow_abs)) - @as(isize, @intCast(t.p.pad_left));

                        const interior: bool = (ih0 >= 0 and iw0 >= 0 and (ih0 + @as(isize, @intCast(t.k_h - 1))) < @as(isize, @intCast(t.h_in)) and (iw0 + @as(isize, @intCast(t.k_w - 1))) < @as(isize, @intCast(t.w_in)));

                        const out_off: usize = (oh_local * (w_mem * c_mem)) + (ow_local * c_mem);

                        var c: usize = 0;
                        while (c + lanes <= c_mem) : (c += lanes) {
                            var acc: Vec = @splat(@as(f32, 0.0));

                            if (interior) {
                                const ih0u: usize = @intCast(ih0);
                                const iw0u: usize = @intCast(iw0);

                                if (tuning.unroll_3x3 and t.k_h == 3 and t.k_w == 3) {
                                    const x_row0: usize = (ih0u + 0) * xt.row_stride;
                                    const x_row1: usize = (ih0u + 1) * xt.row_stride;
                                    const x_row2: usize = (ih0u + 2) * xt.row_stride;
                                    const w_row0: usize = 0 * wt.kh_stride;
                                    const w_row1: usize = 1 * wt.kh_stride;
                                    const w_row2: usize = 2 * wt.kh_stride;

                                    const xp00: [*]align(1) const f32 = xt.vals.ptr + (x_row0 + (iw0u + 0) * xt.c_mem + c);
                                    const xp01: [*]align(1) const f32 = xt.vals.ptr + (x_row0 + (iw0u + 1) * xt.c_mem + c);
                                    const xp02: [*]align(1) const f32 = xt.vals.ptr + (x_row0 + (iw0u + 2) * xt.c_mem + c);
                                    const xp10: [*]align(1) const f32 = xt.vals.ptr + (x_row1 + (iw0u + 0) * xt.c_mem + c);
                                    const xp11: [*]align(1) const f32 = xt.vals.ptr + (x_row1 + (iw0u + 1) * xt.c_mem + c);
                                    const xp12: [*]align(1) const f32 = xt.vals.ptr + (x_row1 + (iw0u + 2) * xt.c_mem + c);
                                    const xp20: [*]align(1) const f32 = xt.vals.ptr + (x_row2 + (iw0u + 0) * xt.c_mem + c);
                                    const xp21: [*]align(1) const f32 = xt.vals.ptr + (x_row2 + (iw0u + 1) * xt.c_mem + c);
                                    const xp22: [*]align(1) const f32 = xt.vals.ptr + (x_row2 + (iw0u + 2) * xt.c_mem + c);

                                    const wp00: [*]align(1) const f32 = wt.vals.ptr + (w_row0 + 0 * wt.c_mem + c);
                                    const wp01: [*]align(1) const f32 = wt.vals.ptr + (w_row0 + 1 * wt.c_mem + c);
                                    const wp02: [*]align(1) const f32 = wt.vals.ptr + (w_row0 + 2 * wt.c_mem + c);
                                    const wp10: [*]align(1) const f32 = wt.vals.ptr + (w_row1 + 0 * wt.c_mem + c);
                                    const wp11: [*]align(1) const f32 = wt.vals.ptr + (w_row1 + 1 * wt.c_mem + c);
                                    const wp12: [*]align(1) const f32 = wt.vals.ptr + (w_row1 + 2 * wt.c_mem + c);
                                    const wp20: [*]align(1) const f32 = wt.vals.ptr + (w_row2 + 0 * wt.c_mem + c);
                                    const wp21: [*]align(1) const f32 = wt.vals.ptr + (w_row2 + 1 * wt.c_mem + c);
                                    const wp22: [*]align(1) const f32 = wt.vals.ptr + (w_row2 + 2 * wt.c_mem + c);

                                    const xv00: Vec = @as(*align(1) const Vec, @ptrCast(xp00)).*;
                                    const xv01: Vec = @as(*align(1) const Vec, @ptrCast(xp01)).*;
                                    const xv02: Vec = @as(*align(1) const Vec, @ptrCast(xp02)).*;
                                    const xv10: Vec = @as(*align(1) const Vec, @ptrCast(xp10)).*;
                                    const xv11: Vec = @as(*align(1) const Vec, @ptrCast(xp11)).*;
                                    const xv12: Vec = @as(*align(1) const Vec, @ptrCast(xp12)).*;
                                    const xv20: Vec = @as(*align(1) const Vec, @ptrCast(xp20)).*;
                                    const xv21: Vec = @as(*align(1) const Vec, @ptrCast(xp21)).*;
                                    const xv22: Vec = @as(*align(1) const Vec, @ptrCast(xp22)).*;

                                    const wv00: Vec = @as(*align(1) const Vec, @ptrCast(wp00)).*;
                                    const wv01: Vec = @as(*align(1) const Vec, @ptrCast(wp01)).*;
                                    const wv02: Vec = @as(*align(1) const Vec, @ptrCast(wp02)).*;
                                    const wv10: Vec = @as(*align(1) const Vec, @ptrCast(wp10)).*;
                                    const wv11: Vec = @as(*align(1) const Vec, @ptrCast(wp11)).*;
                                    const wv12: Vec = @as(*align(1) const Vec, @ptrCast(wp12)).*;
                                    const wv20: Vec = @as(*align(1) const Vec, @ptrCast(wp20)).*;
                                    const wv21: Vec = @as(*align(1) const Vec, @ptrCast(wp21)).*;
                                    const wv22: Vec = @as(*align(1) const Vec, @ptrCast(wp22)).*;

                                    acc += xv00 * wv00;
                                    acc += xv01 * wv01;
                                    acc += xv02 * wv02;
                                    acc += xv10 * wv10;
                                    acc += xv11 * wv11;
                                    acc += xv12 * wv12;
                                    acc += xv20 * wv20;
                                    acc += xv21 * wv21;
                                    acc += xv22 * wv22;
                                } else {
                                    var kh: usize = 0;
                                    while (kh < t.k_h) : (kh += 1) {
                                        const ih: usize = ih0u + kh;
                                        const x_row_base: usize = ih * xt.row_stride;
                                        const w_row_base: usize = kh * wt.kh_stride;

                                        var kw: usize = 0;
                                        while (kw < t.k_w) : (kw += 1) {
                                            const iw: usize = iw0u + kw;
                                            const xp: [*]align(1) const f32 = xt.vals.ptr + (x_row_base + iw * xt.c_mem + c);
                                            const wp: [*]align(1) const f32 = wt.vals.ptr + (w_row_base + kw * wt.c_mem + c);
                                            const xv: Vec = @as(*align(1) const Vec, @ptrCast(xp)).*;
                                            const wv: Vec = @as(*align(1) const Vec, @ptrCast(wp)).*;
                                            acc += xv * wv;
                                        }
                                    }
                                }
                            } else {
                                var kh: usize = 0;
                                while (kh < t.k_h) : (kh += 1) {
                                    const ih: isize = ih0 + @as(isize, @intCast(kh));
                                    if (ih < 0 or ih >= @as(isize, @intCast(t.h_in))) continue;
                                    const ih_u: usize = @intCast(ih);
                                    if (ih_u >= xt.h_mem) continue;
                                    const x_row_base: usize = ih_u * xt.row_stride;
                                    const w_row_base: usize = kh * wt.kh_stride;

                                    var kw: usize = 0;
                                    while (kw < t.k_w) : (kw += 1) {
                                        const iw: isize = iw0 + @as(isize, @intCast(kw));
                                        if (iw < 0 or iw >= @as(isize, @intCast(t.w_in))) continue;
                                        const iw_u: usize = @intCast(iw);
                                        if (iw_u >= xt.w_mem) continue;

                                        const xp: [*]align(1) const f32 = xt.vals.ptr + (x_row_base + iw_u * xt.c_mem + c);
                                        const wp: [*]align(1) const f32 = wt.vals.ptr + (w_row_base + kw * wt.c_mem + c);
                                        const xv: Vec = @as(*align(1) const Vec, @ptrCast(xp)).*;
                                        const wv: Vec = @as(*align(1) const Vec, @ptrCast(wp)).*;
                                        acc += xv * wv;
                                    }
                                }
                            }

                            if (bias_present_local) {
                                const bp: [*]align(1) const f32 = bias.ptr + c;
                                const bv: Vec = @as(*align(1) const Vec, @ptrCast(bp)).*;
                                acc += bv;
                            }

                            const dstp: [*]align(1) f32 = out_vals.ptr + (out_off + c);
                            @as(*align(1) Vec, @ptrCast(dstp)).* = acc;
                        }

                        while (c < c_mem) : (c += 1) {
                            var accs: f32 = 0.0;

                            if (interior and tuning.unroll_3x3 and t.k_h == 3 and t.k_w == 3) {
                                const ih0u: usize = @intCast(ih0);
                                const iw0u: usize = @intCast(iw0);

                                const x00: f32 = xt.vals[((ih0u + 0) * xt.row_stride) + ((iw0u + 0) * xt.c_mem) + c];
                                const x01: f32 = xt.vals[((ih0u + 0) * xt.row_stride) + ((iw0u + 1) * xt.c_mem) + c];
                                const x02: f32 = xt.vals[((ih0u + 0) * xt.row_stride) + ((iw0u + 2) * xt.c_mem) + c];
                                const x10: f32 = xt.vals[((ih0u + 1) * xt.row_stride) + ((iw0u + 0) * xt.c_mem) + c];
                                const x11: f32 = xt.vals[((ih0u + 1) * xt.row_stride) + ((iw0u + 1) * xt.c_mem) + c];
                                const x12: f32 = xt.vals[((ih0u + 1) * xt.row_stride) + ((iw0u + 2) * xt.c_mem) + c];
                                const x20: f32 = xt.vals[((ih0u + 2) * xt.row_stride) + ((iw0u + 0) * xt.c_mem) + c];
                                const x21: f32 = xt.vals[((ih0u + 2) * xt.row_stride) + ((iw0u + 1) * xt.c_mem) + c];
                                const x22: f32 = xt.vals[((ih0u + 2) * xt.row_stride) + ((iw0u + 2) * xt.c_mem) + c];

                                const w00: f32 = wt.vals[(0 * wt.kh_stride) + (0 * wt.c_mem) + c];
                                const w01: f32 = wt.vals[(0 * wt.kh_stride) + (1 * wt.c_mem) + c];
                                const w02: f32 = wt.vals[(0 * wt.kh_stride) + (2 * wt.c_mem) + c];
                                const w10: f32 = wt.vals[(1 * wt.kh_stride) + (0 * wt.c_mem) + c];
                                const w11: f32 = wt.vals[(1 * wt.kh_stride) + (1 * wt.c_mem) + c];
                                const w12: f32 = wt.vals[(1 * wt.kh_stride) + (2 * wt.c_mem) + c];
                                const w20: f32 = wt.vals[(2 * wt.kh_stride) + (0 * wt.c_mem) + c];
                                const w21: f32 = wt.vals[(2 * wt.kh_stride) + (1 * wt.c_mem) + c];
                                const w22: f32 = wt.vals[(2 * wt.kh_stride) + (2 * wt.c_mem) + c];

                                accs = x00 * w00 + x01 * w01 + x02 * w02 + x10 * w10 + x11 * w11 + x12 * w12 + x20 * w20 + x21 * w21 + x22 * w22;
                            } else {
                                var kh: usize = 0;
                                while (kh < t.k_h) : (kh += 1) {
                                    const ih: isize = ih0 + @as(isize, @intCast(kh));
                                    if (ih < 0 or ih >= @as(isize, @intCast(t.h_in))) continue;
                                    const ih_u: usize = @intCast(ih);
                                    if (ih_u >= xt.h_mem) continue;
                                    const x_row_base: usize = ih_u * xt.row_stride;
                                    const w_row_base: usize = kh * wt.kh_stride;

                                    var kw: usize = 0;
                                    while (kw < t.k_w) : (kw += 1) {
                                        const iw: isize = iw0 + @as(isize, @intCast(kw));
                                        if (iw < 0 or iw >= @as(isize, @intCast(t.w_in))) continue;
                                        const iw_u: usize = @intCast(iw);
                                        if (iw_u >= xt.w_mem) continue;

                                        accs += xt.vals[x_row_base + iw_u * xt.c_mem + c] * wt.vals[w_row_base + kw * wt.c_mem + c];
                                    }
                                }
                            }

                            if (bias_present_local and c < bias.len) accs += bias[c];
                            out_vals[out_off + c] = accs;
                        }
                    }
                }

                continue;
            }

            if (t.use_stride1_maps) {
                // Stride=1, dilation=1 path that supports spatial tiling via precomputed ih/iw maps.
                // This avoids integer division/mod inside the kh/kw loops.
                const pad_top: usize = t.p.pad_top;
                const pad_left: usize = t.p.pad_left;
                const h_in_u: usize = t.h_in;
                const w_in_u: usize = t.w_in;
                const k_h_u: usize = t.k_h;
                const k_w_u: usize = t.k_w;

                var oh_local: usize = 0;
                while (oh_local < h_mem) : (oh_local += 1) {
                    const oh_abs: usize = oh_base + oh_local;
                    if (oh_abs >= t.h_out) break;

                    var ow_local: usize = 0;
                    while (ow_local < w_mem) : (ow_local += 1) {
                        const ow_abs: usize = ow_base + ow_local;
                        if (ow_abs >= t.w_out) break;

                        const interior: bool = (oh_abs >= pad_top and ow_abs >= pad_left and (oh_abs + (k_h_u - 1)) < (h_in_u + pad_top) and (ow_abs + (k_w_u - 1)) < (w_in_u + pad_left));

                        const out_off: usize = (oh_local * (w_mem * c_mem)) + (ow_local * c_mem);

                        var c: usize = 0;
                        while (c + lanes <= c_mem) : (c += lanes) {
                            var acc: Vec = @splat(@as(f32, 0.0));

                            if (interior) {
                                const ih0: usize = oh_abs - pad_top;
                                const iw0: usize = ow_abs - pad_left;

                                var kh: usize = 0;
                                while (kh < k_h_u) : (kh += 1) {
                                    const ih: usize = ih0 + kh;
                                    const xhti: usize = @as(usize, t.ih_to_xhti[ih]);
                                    const ih_l: usize = @as(usize, t.ih_to_ihl[ih]);

                                    var kw: usize = 0;
                                    while (kw < k_w_u) : (kw += 1) {
                                        const iw: usize = iw0 + kw;
                                        const xwti: usize = @as(usize, t.iw_to_xwti[iw]);
                                        const iw_l: usize = @as(usize, t.iw_to_iwl[iw]);

                                        const xt: XTile = t.x_tiles[xIndex(bb, xhti, xwti, cti, t.x_htc, t.x_wtc, t.x_ctc)];

                                        const khti: usize = if (t.w_tkh == 0) 0 else (kh / t.w_tkh);
                                        const kwti: usize = if (t.w_tkw == 0) 0 else (kw / t.w_tkw);
                                        const khl: usize = kh - khti * t.w_tkh;
                                        const kwl: usize = kw - kwti * t.w_tkw;
                                        const wt: WTile = t.w_tiles[cti * (t.w_khtc * t.w_kwtc) + (khti * t.w_kwtc + kwti)];

                                        const x_base: usize = (ih_l * xt.row_stride) + (iw_l * xt.c_mem);
                                        const w_base: usize = (khl * wt.kh_stride) + (kwl * wt.c_mem);

                                        const xp: [*]align(1) const f32 = xt.vals.ptr + (x_base + c);
                                        const wp: [*]align(1) const f32 = wt.vals.ptr + (w_base + c);
                                        const xv: Vec = @as(*align(1) const Vec, @ptrCast(xp)).*;
                                        const wv: Vec = @as(*align(1) const Vec, @ptrCast(wp)).*;
                                        acc += xv * wv;
                                    }
                                }
                            } else {
                                var kh: usize = 0;
                                while (kh < k_h_u) : (kh += 1) {
                                    const ih_nom: usize = oh_abs + kh;
                                    if (ih_nom < pad_top) continue;
                                    const ih: usize = ih_nom - pad_top;
                                    if (ih >= h_in_u) continue;

                                    const xhti: usize = @as(usize, t.ih_to_xhti[ih]);
                                    if (xhti >= t.x_htc) continue;
                                    const ih_l: usize = @as(usize, t.ih_to_ihl[ih]);

                                    var kw: usize = 0;
                                    while (kw < k_w_u) : (kw += 1) {
                                        const iw_nom: usize = ow_abs + kw;
                                        if (iw_nom < pad_left) continue;
                                        const iw: usize = iw_nom - pad_left;
                                        if (iw >= w_in_u) continue;

                                        const xwti: usize = @as(usize, t.iw_to_xwti[iw]);
                                        if (xwti >= t.x_wtc) continue;
                                        const iw_l: usize = @as(usize, t.iw_to_iwl[iw]);

                                        const xt: XTile = t.x_tiles[xIndex(bb, xhti, xwti, cti, t.x_htc, t.x_wtc, t.x_ctc)];
                                        if (ih_l >= xt.h_mem or iw_l >= xt.w_mem) continue;
                                        if (c + lanes > xt.c_mem) continue;

                                        const khti: usize = if (t.w_tkh == 0) 0 else (kh / t.w_tkh);
                                        const kwti: usize = if (t.w_tkw == 0) 0 else (kw / t.w_tkw);
                                        if (khti >= t.w_khtc or kwti >= t.w_kwtc) continue;
                                        const khl: usize = kh - khti * t.w_tkh;
                                        const kwl: usize = kw - kwti * t.w_tkw;

                                        const wt: WTile = t.w_tiles[cti * (t.w_khtc * t.w_kwtc) + (khti * t.w_kwtc + kwti)];
                                        if (khl >= wt.kh_mem or kwl >= wt.kw_mem) continue;
                                        if (c + lanes > wt.c_mem) continue;

                                        const x_base: usize = (ih_l * xt.row_stride) + (iw_l * xt.c_mem);
                                        const w_base: usize = (khl * wt.kh_stride) + (kwl * wt.c_mem);

                                        const xp: [*]align(1) const f32 = xt.vals.ptr + (x_base + c);
                                        const wp: [*]align(1) const f32 = wt.vals.ptr + (w_base + c);
                                        const xv: Vec = @as(*align(1) const Vec, @ptrCast(xp)).*;
                                        const wv: Vec = @as(*align(1) const Vec, @ptrCast(wp)).*;
                                        acc += xv * wv;
                                    }
                                }
                            }

                            if (bias_present_local) {
                                const bp: [*]align(1) const f32 = bias.ptr + c;
                                const bv: Vec = @as(*align(1) const Vec, @ptrCast(bp)).*;
                                acc += bv;
                            }

                            const dstp: [*]align(1) f32 = out_vals.ptr + (out_off + c);
                            @as(*align(1) Vec, @ptrCast(dstp)).* = acc;
                        }

                        while (c < c_mem) : (c += 1) {
                            var accs: f32 = 0.0;

                            if (interior) {
                                const ih0: usize = oh_abs - pad_top;
                                const iw0: usize = ow_abs - pad_left;

                                var kh: usize = 0;
                                while (kh < k_h_u) : (kh += 1) {
                                    const ih: usize = ih0 + kh;
                                    const xhti: usize = @as(usize, t.ih_to_xhti[ih]);
                                    const ih_l: usize = @as(usize, t.ih_to_ihl[ih]);

                                    var kw: usize = 0;
                                    while (kw < k_w_u) : (kw += 1) {
                                        const iw: usize = iw0 + kw;
                                        const xwti: usize = @as(usize, t.iw_to_xwti[iw]);
                                        const iw_l: usize = @as(usize, t.iw_to_iwl[iw]);

                                        const xt: XTile = t.x_tiles[xIndex(bb, xhti, xwti, cti, t.x_htc, t.x_wtc, t.x_ctc)];

                                        const khti: usize = if (t.w_tkh == 0) 0 else (kh / t.w_tkh);
                                        const kwti: usize = if (t.w_tkw == 0) 0 else (kw / t.w_tkw);
                                        const khl: usize = kh - khti * t.w_tkh;
                                        const kwl: usize = kw - kwti * t.w_tkw;
                                        const wt: WTile = t.w_tiles[cti * (t.w_khtc * t.w_kwtc) + (khti * t.w_kwtc + kwti)];

                                        accs += xt.vals[(ih_l * xt.row_stride) + (iw_l * xt.c_mem) + c] * wt.vals[(khl * wt.kh_stride) + (kwl * wt.c_mem) + c];
                                    }
                                }
                            } else {
                                var kh: usize = 0;
                                while (kh < k_h_u) : (kh += 1) {
                                    const ih_nom: usize = oh_abs + kh;
                                    if (ih_nom < pad_top) continue;
                                    const ih: usize = ih_nom - pad_top;
                                    if (ih >= h_in_u) continue;

                                    const xhti: usize = @as(usize, t.ih_to_xhti[ih]);
                                    if (xhti >= t.x_htc) continue;
                                    const ih_l: usize = @as(usize, t.ih_to_ihl[ih]);

                                    var kw: usize = 0;
                                    while (kw < k_w_u) : (kw += 1) {
                                        const iw_nom: usize = ow_abs + kw;
                                        if (iw_nom < pad_left) continue;
                                        const iw: usize = iw_nom - pad_left;
                                        if (iw >= w_in_u) continue;

                                        const xwti: usize = @as(usize, t.iw_to_xwti[iw]);
                                        if (xwti >= t.x_wtc) continue;
                                        const iw_l: usize = @as(usize, t.iw_to_iwl[iw]);

                                        const xt: XTile = t.x_tiles[xIndex(bb, xhti, xwti, cti, t.x_htc, t.x_wtc, t.x_ctc)];
                                        if (ih_l >= xt.h_mem or iw_l >= xt.w_mem or c >= xt.c_mem) continue;

                                        const khti: usize = if (t.w_tkh == 0) 0 else (kh / t.w_tkh);
                                        const kwti: usize = if (t.w_tkw == 0) 0 else (kw / t.w_tkw);
                                        if (khti >= t.w_khtc or kwti >= t.w_kwtc) continue;
                                        const khl: usize = kh - khti * t.w_tkh;
                                        const kwl: usize = kw - kwti * t.w_tkw;

                                        const wt: WTile = t.w_tiles[cti * (t.w_khtc * t.w_kwtc) + (khti * t.w_kwtc + kwti)];
                                        if (khl >= wt.kh_mem or kwl >= wt.kw_mem or c >= wt.c_mem) continue;

                                        accs += xt.vals[(ih_l * xt.row_stride) + (iw_l * xt.c_mem) + c] * wt.vals[(khl * wt.kh_stride) + (kwl * wt.c_mem) + c];
                                    }
                                }
                            }

                            if (bias_present_local and c < bias.len) accs += bias[c];
                            out_vals[out_off + c] = accs;
                        }
                    }
                }

                continue;
            }

            // Generic path: supports spatial tiling, K tiling, stride/dilation.
            var oh_local: usize = 0;
            while (oh_local < h_mem) : (oh_local += 1) {
                const oh_abs: usize = oh_base + oh_local;
                if (oh_abs >= t.h_out) break;

                var ow_local: usize = 0;
                while (ow_local < w_mem) : (ow_local += 1) {
                    const ow_abs: usize = ow_base + ow_local;
                    if (ow_abs >= t.w_out) break;

                    const out_off: usize = (oh_local * (w_mem * c_mem)) + (ow_local * c_mem);

                    var c: usize = 0;
                    while (c + lanes <= c_mem) : (c += lanes) {
                        var acc: Vec = @splat(@as(f32, 0.0));

                        var kh: usize = 0;
                        while (kh < t.k_h) : (kh += 1) {
                            const ih_nom: usize = oh_abs * t.p.stride_h + kh * t.p.dilation_h;
                            const ih_idx: isize = @as(isize, @intCast(ih_nom)) - @as(isize, @intCast(t.p.pad_top));
                            const ih: usize = if (use_reflect) @intCast(reflectIndex(ih_idx, h_in_i)) else blk: {
                                if (ih_idx < 0 or ih_idx >= h_in_i) continue;
                                break :blk @as(usize, @intCast(ih_idx));
                            };

                            const xhti: usize = ih / t.x_th;
                            if (xhti >= t.x_htc) continue;
                            const ih_l: usize = ih - xhti * t.x_th;

                            var kw: usize = 0;
                            while (kw < t.k_w) : (kw += 1) {
                                const iw_nom: usize = ow_abs * t.p.stride_w + kw * t.p.dilation_w;
                                const iw_idx: isize = @as(isize, @intCast(iw_nom)) - @as(isize, @intCast(t.p.pad_left));
                                const iw: usize = if (use_reflect) @intCast(reflectIndex(iw_idx, w_in_i)) else blk: {
                                    if (iw_idx < 0 or iw_idx >= w_in_i) continue;
                                    break :blk @as(usize, @intCast(iw_idx));
                                };

                                const xwti: usize = iw / t.x_tw;
                                if (xwti >= t.x_wtc) continue;
                                const iw_l: usize = iw - xwti * t.x_tw;

                                const xt: XTile = t.x_tiles[xIndex(bb, xhti, xwti, cti, t.x_htc, t.x_wtc, t.x_ctc)];
                                if (ih_l >= xt.h_mem or iw_l >= xt.w_mem) continue;
                                if (c + lanes > xt.c_mem) continue;

                                // Weight tile lookup.
                                const khti: usize = if (t.w_tkh == 0) 0 else (kh / t.w_tkh);
                                const kwti: usize = if (t.w_tkw == 0) 0 else (kw / t.w_tkw);
                                if (khti >= t.w_khtc or kwti >= t.w_kwtc) continue;
                                const khl: usize = kh - khti * t.w_tkh;
                                const kwl: usize = kw - kwti * t.w_tkw;

                                const wt: WTile = t.w_tiles[cti * (t.w_khtc * t.w_kwtc) + (khti * t.w_kwtc + kwti)];
                                if (khl >= wt.kh_mem or kwl >= wt.kw_mem) continue;
                                if (c + lanes > wt.c_mem) continue;

                                const xp: [*]align(1) const f32 = xt.vals.ptr + ((ih_l * xt.row_stride) + (iw_l * xt.c_mem) + c);
                                const wp: [*]align(1) const f32 = wt.vals.ptr + ((khl * wt.kh_stride) + (kwl * wt.c_mem) + c);
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

                        const dstp: [*]align(1) f32 = out_vals.ptr + (out_off + c);
                        @as(*align(1) Vec, @ptrCast(dstp)).* = acc;
                    }

                    while (c < c_mem) : (c += 1) {
                        var accs: f32 = 0.0;

                        var kh: usize = 0;
                        while (kh < t.k_h) : (kh += 1) {
                            const ih_nom: usize = oh_abs * t.p.stride_h + kh * t.p.dilation_h;
                            const ih_idx: isize = @as(isize, @intCast(ih_nom)) - @as(isize, @intCast(t.p.pad_top));
                            const ih: usize = if (use_reflect) @intCast(reflectIndex(ih_idx, h_in_i)) else blk: {
                                if (ih_idx < 0 or ih_idx >= h_in_i) continue;
                                break :blk @as(usize, @intCast(ih_idx));
                            };
                            const xhti: usize = ih / t.x_th;
                            if (xhti >= t.x_htc) continue;
                            const ih_l: usize = ih - xhti * t.x_th;

                            var kw: usize = 0;
                            while (kw < t.k_w) : (kw += 1) {
                                const iw_nom: usize = ow_abs * t.p.stride_w + kw * t.p.dilation_w;
                                const iw_idx: isize = @as(isize, @intCast(iw_nom)) - @as(isize, @intCast(t.p.pad_left));
                                const iw: usize = if (use_reflect) @intCast(reflectIndex(iw_idx, w_in_i)) else blk: {
                                    if (iw_idx < 0 or iw_idx >= w_in_i) continue;
                                    break :blk @as(usize, @intCast(iw_idx));
                                };
                                const xwti: usize = iw / t.x_tw;
                                if (xwti >= t.x_wtc) continue;
                                const iw_l: usize = iw - xwti * t.x_tw;

                                const xt: XTile = t.x_tiles[xIndex(bb, xhti, xwti, cti, t.x_htc, t.x_wtc, t.x_ctc)];
                                if (ih_l >= xt.h_mem or iw_l >= xt.w_mem or c >= xt.c_mem) continue;

                                const khti: usize = if (t.w_tkh == 0) 0 else (kh / t.w_tkh);
                                const kwti: usize = if (t.w_tkw == 0) 0 else (kw / t.w_tkw);
                                if (khti >= t.w_khtc or kwti >= t.w_kwtc) continue;
                                const khl: usize = kh - khti * t.w_tkh;
                                const kwl: usize = kw - kwti * t.w_tkw;

                                const wt: WTile = t.w_tiles[cti * (t.w_khtc * t.w_kwtc) + (khti * t.w_kwtc + kwti)];
                                if (khl >= wt.kh_mem or kwl >= wt.kw_mem or c >= wt.c_mem) continue;

                                accs += xt.vals[(ih_l * xt.row_stride) + (iw_l * xt.c_mem) + c] * wt.vals[(khl * wt.kh_stride) + (kwl * wt.c_mem) + c];
                            }
                        }

                        if (bias_present_local and c < bias.len) accs += bias[c];
                        out_vals[out_off + c] = accs;
                    }
                }
            }
        }
    }

    pub fn runItemRange(t: *const @This(), start: usize, end: usize) void {
        runItemRangeImpl(.{}, t, start, end);
    }

    pub fn runItems(ctx_any: *anyopaque, start: usize, end: usize, _: usize) void {
        const t: *@This() = @ptrCast(@alignCast(ctx_any));
        t.runItemRange(start, end);
    }
};
