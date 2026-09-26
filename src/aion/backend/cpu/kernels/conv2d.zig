// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Depthwise convolution over flat NHWC tensors. Conv1D's depthwise case is this
//! kernel with one input row: its length is the width axis.
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

/// Depthwise Conv2D kernel tuning knobs (compile-time).
pub const DepthwiseConv2DTuning = struct {
    lanes: usize = simd.lanesF32(),
};

/// Output pixels one work item computes along a row.
pub const PIXEL_BLOCK: usize = 16;

/// Most kernel taps a pixel can have; a larger kernel takes the grouped GEMM path.
pub const MAX_TAPS: usize = 256;

/// `out[b, oh, ow, c] = bias[c] + sum over taps of x[b, ih, iw, c] * w[kh, kw, 0, c]`.
///
/// A work item is one block of up to `PIXEL_BLOCK` output pixels of one row, so a
/// single long row (Conv1D) still spreads across threads. Each pixel resolves its
/// in-range taps once; the channel loop over them is the vector loop, reading one
/// contiguous run of channels per tap from x and w.
pub const DepthwiseConv2DTask = struct {
    p: DepthwiseConv2DParams,
    batch: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
    c: usize,
    k_h: usize,
    k_w: usize,
    /// `[batch, h_in, w_in, c]`.
    x: []align(1) const f32,
    /// `[k_h, k_w, 1, c]`.
    w: []align(1) const f32,
    /// `[c]`, or empty.
    bias: []align(1) const f32,
    /// `[batch, h_out, w_out, c]`.
    out: []align(1) f32,

    pub fn items(t: *const DepthwiseConv2DTask) usize {
        return t.batch * t.h_out * blocksPerRow(t);
    }

    fn blocksPerRow(t: *const DepthwiseConv2DTask) usize {
        return std.math.divCeil(usize, t.w_out, PIXEL_BLOCK) catch unreachable;
    }
};

/// Input index for nominal coordinate `nom - pad`, or null when it falls in zero
/// padding.
inline fn resolve(nom: usize, pad: usize, len: usize, reflect: bool) ?usize {
    const i: isize = @as(isize, @intCast(nom)) - @as(isize, @intCast(pad));
    const n: isize = @intCast(len);
    if (i >= 0 and i < n) return @intCast(i);
    if (!reflect) return null;
    var x = i;
    while (x < 0 or x >= n) x = if (x < 0) -x else (2 * n - 2) - x;
    return @intCast(x);
}

pub fn Kernel(comptime tuning: DepthwiseConv2DTuning) type {
    return struct {
        const lanes = tuning.lanes;
        const Vec = @Vector(lanes, f32);

        pub fn runItemRange(t: *const DepthwiseConv2DTask, start: usize, end: usize) void {
            @setRuntimeSafety(false);
            var x_off: [MAX_TAPS]usize = undefined;
            var w_off: [MAX_TAPS]usize = undefined;
            const c = t.c;
            const per_row = t.blocksPerRow();
            for (start..end) |item| {
                const row = item / per_row;
                const ow0 = (item % per_row) * PIXEL_BLOCK;
                const b = row / t.h_out;
                const oh = row % t.h_out;
                for (ow0..@min(ow0 + PIXEL_BLOCK, t.w_out)) |ow| {
                    var n: usize = 0;
                    for (0..t.k_h) |kh| {
                        const ih = resolve(oh * t.p.stride_h + kh * t.p.dilation_h, t.p.pad_top, t.h_in, t.p.reflect) orelse continue;
                        for (0..t.k_w) |kw| {
                            const iw = resolve(ow * t.p.stride_w + kw * t.p.dilation_w, t.p.pad_left, t.w_in, t.p.reflect) orelse continue;
                            x_off[n] = ((b * t.h_in + ih) * t.w_in + iw) * c;
                            w_off[n] = (kh * t.k_w + kw) * c;
                            n += 1;
                        }
                    }
                    const out = t.out[((b * t.h_out + oh) * t.w_out + ow) * c ..][0..c];
                    var ch: usize = 0;
                    while (ch + lanes <= c) : (ch += lanes) {
                        var acc: Vec = if (t.bias.len != 0) load(t.bias, ch) else @splat(0.0);
                        for (x_off[0..n], w_off[0..n]) |xo, wo| acc += load(t.x, xo + ch) * load(t.w, wo + ch);
                        @as(*align(1) Vec, @ptrCast(out.ptr + ch)).* = acc;
                    }
                    while (ch < c) : (ch += 1) {
                        var acc: f32 = if (t.bias.len != 0) t.bias[ch] else 0.0;
                        for (x_off[0..n], w_off[0..n]) |xo, wo| acc += t.x[xo + ch] * t.w[wo + ch];
                        out[ch] = acc;
                    }
                }
            }
        }

        pub fn runItems(ctx_any: *anyopaque, start: usize, end: usize, _: usize) void {
            const t: *const DepthwiseConv2DTask = @ptrCast(@alignCast(ctx_any));
            runItemRange(t, start, end);
        }

        inline fn load(s: []align(1) const f32, i: usize) Vec {
            return @as(*align(1) const Vec, @ptrCast(s.ptr + i)).*;
        }
    };
}
