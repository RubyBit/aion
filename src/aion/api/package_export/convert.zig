// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
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
        .RelPosMHA => |attn| .{ .RelPosMHA = .{ .scale = attn.scale, .heads = attn.heads, .has_mask = attn.has_mask } },
        .ArgMax => |am| .{ .ArgMax = .{ .axis = am.axis } },
        .ScatterRow => .ScatterRow,
        .MultiHeadAttentionCached => |attn| .{ .MultiHeadAttentionCached = .{
            .scale = attn.scale,
            .causal = attn.causal,
            .sliding_window = attn.sliding_window,
            .attn_logits_soft_cap = attn.attn_logits_soft_cap,
        } },
        .Reduce => |rr| .{ .Reduce = .{ .op = rr.op, .axis = rr.axis } },
        .Concat => |cc| .{ .Concat = .{ .axis = cc.axis } },
        .LSTMCell => |lc| .{ .LSTMCell = .{ .has_bias = lc.has_bias } },
        .RFFT => .RFFT,
        .STFT => |st| .{ .STFT = .{ .n_fft = st.n_fft, .hop_length = st.hop_length, .center = st.center } },
        .Copy => .Copy,
        .GatherRows => .GatherRows,
        .RoPE1D => |rp| .{ .RoPE1D = .{
            .base_frequency = rp.base_frequency,
            .scale_factor = rp.scale_factor,
            .rope_proportion = rp.rope_proportion,
        } },
        .SequenceAppend => .SequenceAppend,
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
        .Cast => |ct| .{ .Cast = .{ .to_dtype = ct.to_dtype } },
        .MatMulNT => |mm| .{ .MatMulNT = .{ .alpha = mm.alpha, .beta = mm.beta } },
        .If => |iff| .{ .If = .{ .then_region = iff.then_region, .else_region = iff.else_region } },
        .Loop => |lp| .{ .Loop = .{ .body_region = lp.body_region, .static_max_trip_count = lp.static_max_trip_count } },
    };
    return out;
}
