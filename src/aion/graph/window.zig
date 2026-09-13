// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

pub const Error = error{ RankMismatch, EmptyDimension, InvalidKernel, InvalidStride, InvalidDilation, InvalidPadding, ShapeOverflow, EmptyOutput };

/// NHWC pooling geometry shared by inference, lowering, and execution.
pub const Pool2D = struct {
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize = 1,
    stride_w: usize = 1,
    dilation_h: usize = 1,
    dilation_w: usize = 1,
    pad_top: usize = 0,
    pad_bottom: usize = 0,
    pad_left: usize = 0,
    pad_right: usize = 0,
    ceil_mode: bool = false,

    pub fn output(self: Pool2D, shape: []const usize) Error![4]usize {
        if (shape.len != 4) return error.RankMismatch;
        if (shape[0] == 0 or shape[3] == 0) return error.EmptyDimension;
        return .{ shape[0], try extent(shape[1], self.kernel_h, self.stride_h, self.dilation_h, self.pad_top, self.pad_bottom, self.ceil_mode), try extent(shape[2], self.kernel_w, self.stride_w, self.dilation_w, self.pad_left, self.pad_right, self.ceil_mode), shape[3] };
    }
};

pub fn extent(input: usize, kernel: usize, stride: usize, dilation: usize, before: usize, after: usize, ceil_mode: bool) Error!usize {
    if (input == 0) return error.EmptyDimension;
    if (kernel == 0) return error.InvalidKernel;
    if (stride == 0) return error.InvalidStride;
    if (dilation == 0) return error.InvalidDilation;
    const effective = std.math.add(usize, std.math.mul(usize, kernel - 1, dilation) catch return error.ShapeOverflow, 1) catch return error.ShapeOverflow;
    if (before >= effective or after >= effective) return error.InvalidPadding;
    const padded = std.math.add(usize, std.math.add(usize, input, before) catch return error.ShapeOverflow, after) catch return error.ShapeOverflow;
    // Signed arithmetic allows a final partial window in ceil mode.
    const delta = @as(i128, @intCast(padded)) - @as(i128, @intCast(effective));
    const st: i128 = @intCast(stride);
    var out = @divFloor(delta + (if (ceil_mode) st - 1 else 0), st) + 1;
    if (ceil_mode and out > 0 and (out - 1) * st >= input + before) out -= 1;
    if (out <= 0) return error.EmptyOutput;
    return std.math.cast(usize, out) orelse error.ShapeOverflow;
}

test "pooling window floor, ceil, dilation, padding, and invalid geometry" {
    try std.testing.expectEqual(@as(usize, 4), try extent(14, 3, 3, 1, 0, 0, false));
    try std.testing.expectEqual(@as(usize, 5), try extent(14, 3, 3, 1, 0, 0, true));
    try std.testing.expectEqual(@as(usize, 2), try extent(3, 2, 2, 1, 0, 1, true));
    try std.testing.expectEqual(@as(usize, 3), try extent(5, 2, 1, 2, 0, 0, false));
    try std.testing.expectError(error.InvalidKernel, extent(5, 0, 1, 1, 0, 0, false));
    try std.testing.expectError(error.InvalidStride, extent(5, 2, 0, 1, 0, 0, false));
    try std.testing.expectError(error.EmptyOutput, extent(1, 3, 1, 1, 0, 0, false));
}
