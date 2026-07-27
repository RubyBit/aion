// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Host-side constant builders: tables and masks computed on the CPU at authoring
//! time and bound as parameters. They return `Tensor`, not `TensorRef`, so the
//! caller decides how to bind them.
const std = @import("std");

const layer_mod = @import("layer.zig");

pub const Builder = layer_mod.Builder;
pub const Tensor = layer_mod.Tensor;
pub const Context = layer_mod.Context;

pub const Error = Builder.Error;

/// Masked-out logit value. Large enough to zero a softmax lane in f32, small
/// enough not to produce NaN when added to a finite logit.
pub const mask_neg_inf: f32 = -1e9;

/// Sinusoidal relative-position table, `[2*length - 1, d_model]`.
///
/// Row 0 is the most positive relative offset (`+(length-1)`) and the last row the
/// most negative, which is the ordering the relative-position attention op indexes
/// with `(T_q - 1) - i + j`. Even feature columns are sine, odd are cosine.
pub fn sinusoidalRelPos(ctx: *Context, allocator: std.mem.Allocator, length: usize, d_model: usize) Error!Tensor {
    if (length == 0 or d_model == 0 or d_model % 2 != 0) return Error.InvalidArgument;

    const rows: usize = std.math.sub(usize, std.math.mul(usize, 2, length) catch return Error.InvalidArgument, 1) catch return Error.InvalidArgument;
    const total: usize = std.math.mul(usize, rows, d_model) catch return Error.InvalidArgument;

    const vals: []f32 = allocator.alloc(f32, total) catch return Error.OutOfMemory;
    defer allocator.free(vals);

    const log_base: f64 = @log(10000.0);
    const dm: f64 = @floatFromInt(d_model);

    var r: usize = 0;
    while (r < rows) : (r += 1) {
        // Positions run +(length-1) down to -(length-1).
        const pos: f64 = @as(f64, @floatFromInt(length - 1)) - @as(f64, @floatFromInt(r));
        var i: usize = 0;
        while (i < d_model / 2) : (i += 1) {
            const two_i: f64 = @floatFromInt(2 * i);
            const inv_freq: f64 = @exp(-(two_i * log_base / dm));
            const angle: f64 = pos * inv_freq;
            vals[r * d_model + 2 * i] = @floatCast(@sin(angle));
            vals[r * d_model + 2 * i + 1] = @floatCast(@cos(angle));
        }
    }

    return ctx.fromF32(&[_]usize{ rows, d_model }, vals) catch Error.OutOfMemory;
}

/// Additive causal mask, `[seq, seq]`: 0 where `j <= i`, `mask_neg_inf` above the
/// diagonal.
pub fn causalMask(ctx: *Context, allocator: std.mem.Allocator, seq: usize) Error!Tensor {
    return bandedMask(ctx, allocator, seq, seq, null, 0);
}

/// Additive sliding-window mask, `[seq, seq]`: position `i` may attend to
/// `[i - left, i + right]`, everything else masked.
pub fn slidingWindowMask(
    ctx: *Context,
    allocator: std.mem.Allocator,
    seq: usize,
    left: usize,
    right: usize,
) Error!Tensor {
    return bandedMask(ctx, allocator, seq, seq, left, right);
}

fn bandedMask(
    ctx: *Context,
    allocator: std.mem.Allocator,
    rows: usize,
    cols: usize,
    left: ?usize,
    right: usize,
) Error!Tensor {
    if (rows == 0 or cols == 0) return Error.InvalidArgument;
    const total: usize = std.math.mul(usize, rows, cols) catch return Error.InvalidArgument;

    const vals: []f32 = allocator.alloc(f32, total) catch return Error.OutOfMemory;
    defer allocator.free(vals);
    @memset(vals, mask_neg_inf);

    var i: usize = 0;
    while (i < rows) : (i += 1) {
        const lo: usize = if (left) |l| (if (i > l) i - l else 0) else 0;
        const hi: usize = @min(cols, i + right + 1);
        var j: usize = lo;
        while (j < hi) : (j += 1) vals[i * cols + j] = 0.0;
    }

    return ctx.fromF32(&[_]usize{ rows, cols }, vals) catch Error.OutOfMemory;
}

/// Additive chunked-limited mask, `[seq, seq]` — the cache-aware streaming pattern.
///
/// Frames are grouped into non-overlapping chunks of `right + 1`. Every frame in a
/// chunk attends to the whole chunk plus `left` frames before the chunk *start*, so
/// early frames in a chunk get up to `right` frames of lookahead while the chunk
/// boundary stays a hard cut. That last part is what makes it streamable, and it is
/// why this is not the same as a sliding window.
pub fn chunkedLimitedMask(
    ctx: *Context,
    allocator: std.mem.Allocator,
    seq: usize,
    left: usize,
    right: usize,
) Error!Tensor {
    if (seq == 0) return Error.InvalidArgument;
    const chunk: usize = std.math.add(usize, right, 1) catch return Error.InvalidArgument;
    const total: usize = std.math.mul(usize, seq, seq) catch return Error.InvalidArgument;

    const vals: []f32 = allocator.alloc(f32, total) catch return Error.OutOfMemory;
    defer allocator.free(vals);
    @memset(vals, mask_neg_inf);

    var i: usize = 0;
    while (i < seq) : (i += 1) {
        const chunk_start: usize = (i / chunk) * chunk;
        const chunk_end: usize = @min(chunk_start + chunk, seq);
        const lo: usize = if (chunk_start > left) chunk_start - left else 0;
        var j: usize = lo;
        while (j < chunk_end) : (j += 1) vals[i * seq + j] = 0.0;
    }

    return ctx.fromF32(&[_]usize{ seq, seq }, vals) catch Error.OutOfMemory;
}
