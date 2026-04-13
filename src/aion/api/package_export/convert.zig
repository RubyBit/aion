const package_file = @import("../../storage/aion_file.zig");
const graph_mod = @import("../../graph/graph.zig");

pub fn convertOp(allocator: anytype, op: graph_mod.Op) !package_file.NodeOp {
    const out: package_file.NodeOp = switch (op) {
        .MatMul => |mm| .{ .MatMul = .{ .alpha = mm.alpha, .beta = mm.beta } },
        .ElemwiseBinary => |eb| .{ .ElemwiseBinary = .{ .op = eb.op } },
        .BroadcastLastDimBinary => |eb| .{ .BroadcastLastDimBinary = .{ .op = eb.op } },
        .Unary => |u| .{ .Unary = .{ .op = u.op } },
        .Softmax => |s| .{ .Softmax = .{ .axis = s.axis } },
        .Conv1D => |cv| .{ .Conv1D = .{
            .stride = cv.stride,
            .dilation = cv.dilation,
            .pad_left = cv.pad_left,
            .pad_right = cv.pad_right,
            .pad_mode = cv.pad_mode,
            .groups = cv.groups,
        } },
        .Conv2D => |cv| .{ .Conv2D = .{
            .stride_h = cv.stride_h,
            .stride_w = cv.stride_w,
            .dilation_h = cv.dilation_h,
            .dilation_w = cv.dilation_w,
            .pad_top = cv.pad_top,
            .pad_bottom = cv.pad_bottom,
            .pad_left = cv.pad_left,
            .pad_right = cv.pad_right,
            .pad_mode = cv.pad_mode,
            .groups = cv.groups,
        } },
        .LayerNorm => |ln| .{ .LayerNorm = .{
            .eps = ln.eps,
            .normalized_shape = try package_file.makeConstantShapeTerms(allocator, ln.normalized_shape),
        } },
        .RMSNorm => |ln| .{ .RMSNorm = .{
            .eps = ln.eps,
            .normalized_shape = try package_file.makeConstantShapeTerms(allocator, ln.normalized_shape),
        } },
        .Attention => |attn| .{ .Attention = .{ .scale = attn.scale, .causal = attn.causal } },
        .MultiHeadAttention => |attn| .{ .MultiHeadAttention = .{ .scale = attn.scale, .causal = attn.causal, .heads = attn.heads } },
        .Reduce => |rr| .{ .Reduce = .{ .op = rr.op, .axis = rr.axis } },
        .Concat => |cc| .{ .Concat = .{ .axis = cc.axis } },
        .LSTMCell => |lc| .{ .LSTMCell = .{ .has_bias = lc.has_bias } },
        .ComplexAbsMean => |cm| .{ .ComplexAbsMean = .{ .out_channels = cm.out_channels } },
        .Copy => .Copy,
        .GatherRows => .GatherRows,
        .ViewReshape => |vr| .{ .ViewReshape = .{ .new_shape = try package_file.makeConstantShapeTerms(allocator, vr.new_shape) } },
        .ViewSqueeze => |vs| .{ .ViewSqueeze = .{ .axis = vs.axis } },
        .ViewUnsqueeze => |vu| .{ .ViewUnsqueeze = .{ .axis = vu.axis } },
        .ViewTranspose2D => .ViewTranspose2D,
        .ViewSliceND => |sl| blk: {
            const starts = try allocator.alloc(u64, sl.starts.len);
            errdefer allocator.free(starts);
            const lens = try package_file.makeConstantShapeTerms(allocator, sl.lens);
            for (sl.starts, 0..) |value, idx| starts[idx] = @intCast(value);
            break :blk .{ .ViewSliceND = .{ .starts = starts, .lens = lens } };
        },
    };
    return out;
}
