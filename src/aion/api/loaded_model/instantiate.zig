// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const graph_mod = @import("../../graph/graph.zig");
const package_file = @import("../../storage/aion_file.zig");
const api_tiling = @import("../tiling.zig");
const api_errors = @import("../errors.zig");

pub fn optionalizeSymbols(allocator: std.mem.Allocator, values: []const u64) ![]?u64 {
    const out = try allocator.alloc(?u64, values.len);
    for (values, 0..) |value, idx| out[idx] = if (value == 0) null else value;
    return out;
}

pub fn instantiateNode(
    allocator: std.mem.Allocator,
    pkg: *const package_file.Package,
    symbol_values: []const u64,
    graph: *graph_mod.Graph,
    node: package_file.NodeRecord,
    mapped_inputs: []const graph_mod.ValueId,
    region_map: []const graph_mod.RegionId,
    /// Filled with the graph value ids of the node's extra outputs (multi-carry
    /// `Loop` only); its length must equal the record node's extra-output count.
    extra_out: []graph_mod.ValueId,
) api_errors.ExecuteError!graph_mod.ValueId {
    _ = allocator;

    const optional_symbols: []?u64 = optionalizeSymbols(graph.arenaAlloc(), symbol_values) catch return error.OutOfMemory;

    return switch (node.op) {
        .MatMul => |mm| try graph.addMatMul(mapped_inputs[0], mapped_inputs[1], mm.alpha, mm.beta),
        .ElemwiseBinary => |eb| try graph.addElemwiseBinary(eb.op, mapped_inputs[0], mapped_inputs[1]),
        .BroadcastLastDimBinary => |eb| try graph.addBroadcastLastDimBinary(eb.op, mapped_inputs[0], mapped_inputs[1]),
        .Unary => |u| try graph.addUnary(u.op, mapped_inputs[0]),
        .Softmax => |s| try graph.addSoftmax(mapped_inputs[0], s.axis),
        .Conv1D => |cv| try graph.addConv1DWithPadMode(
            mapped_inputs[0],
            mapped_inputs[1],
            if (mapped_inputs.len > 2) mapped_inputs[2] else null,
            @intCast(cv.stride),
            @intCast(cv.dilation),
            @intCast(cv.pad_left),
            @intCast(cv.pad_right),
            cv.pad_mode,
            @intCast(cv.groups),
        ),
        .Conv2D => |cv| try graph.addConv2DWithPadMode(
            mapped_inputs[0],
            mapped_inputs[1],
            if (mapped_inputs.len > 2) mapped_inputs[2] else null,
            @intCast(cv.stride_h),
            @intCast(cv.stride_w),
            @intCast(cv.dilation_h),
            @intCast(cv.dilation_w),
            @intCast(cv.pad_top),
            @intCast(cv.pad_bottom),
            @intCast(cv.pad_left),
            @intCast(cv.pad_right),
            cv.pad_mode,
            @intCast(cv.groups),
        ),
        .LayerNorm => |ln| blk: {
            const shape = try package_file.resolveShapeTerms(graph.arenaAlloc(), pkg, ln.normalized_shape, optional_symbols);
            break :blk try graph.addLayerNorm(mapped_inputs[0], mapped_inputs[1], mapped_inputs[2], ln.eps, shape);
        },
        .RMSNorm => |ln| blk: {
            const shape = try package_file.resolveShapeTerms(graph.arenaAlloc(), pkg, ln.normalized_shape, optional_symbols);
            break :blk try graph.addRMSNorm(mapped_inputs[0], mapped_inputs[1], mapped_inputs[2], ln.eps, shape);
        },
        .Attention => |attn| try graph.addAttention(mapped_inputs[0], mapped_inputs[1], mapped_inputs[2], attn.scale, attn.causal),
        .MultiHeadAttention => |attn| try graph.addMultiHeadAttention(mapped_inputs[0], mapped_inputs[1], mapped_inputs[2], attn.scale, attn.causal, @intCast(attn.heads)),
        .RelPosMHA => |attn| try graph.addRelPosMHA(
            mapped_inputs[0],
            mapped_inputs[1],
            mapped_inputs[2],
            mapped_inputs[3],
            mapped_inputs[4],
            mapped_inputs[5],
            if (attn.has_mask) mapped_inputs[6] else null,
            attn.scale,
            @intCast(attn.heads),
        ),
        .ArgMax => |am| try graph.addArgMax(mapped_inputs[0], am.axis),
        .ScatterRow => try graph.addScatterRow(mapped_inputs[0], mapped_inputs[1], mapped_inputs[2]),
        .MultiHeadAttentionCached => |attn| try graph.addMultiHeadAttentionCached(
            mapped_inputs[0],
            mapped_inputs[1],
            mapped_inputs[2],
            mapped_inputs[3],
            mapped_inputs[4],
            attn.scale,
            attn.causal,
            @intCast(attn.sliding_window),
            attn.attn_logits_soft_cap,
        ),
        .Reduce => |rr| if (rr.axis) |axis| try graph.addReduceAxis(rr.op, mapped_inputs[0], axis) else try graph.addReduce(rr.op, mapped_inputs[0]),
        .Concat => |cc| try graph.addConcat(mapped_inputs, cc.axis),
        .LSTMCell => |lc| try graph.addLSTMCell(
            mapped_inputs[0],
            mapped_inputs[1],
            mapped_inputs[2],
            mapped_inputs[3],
            mapped_inputs[4],
            if (lc.has_bias) mapped_inputs[5] else null,
            if (lc.has_bias) mapped_inputs[6] else null,
        ),
        .RFFT => try graph.addRFFT(mapped_inputs[0]),
        .STFT => |st| try graph.addSTFT(
            mapped_inputs[0],
            mapped_inputs[1],
            @intCast(st.n_fft),
            @intCast(st.hop_length),
            st.center,
        ),
        .Copy => try graph.addCopy(mapped_inputs[0]),
        .GatherRows => try graph.addGatherRows(mapped_inputs[0], mapped_inputs[1]),
        .RoPE1D => |rp| try graph.addRoPE1D(
            mapped_inputs[0],
            mapped_inputs[1],
            rp.base_frequency,
            rp.scale_factor,
            rp.rope_proportion,
        ),
        .KVCacheAppend => try graph.addKVCacheAppend(mapped_inputs[0], mapped_inputs[1], mapped_inputs[2]),
        .ViewReshape => |vr| blk: {
            const shape = try package_file.resolveShapeTerms(graph.arenaAlloc(), pkg, vr.new_shape, optional_symbols);
            break :blk try graph.addViewReshape(mapped_inputs[0], shape);
        },
        .ViewSqueeze => |vs| try graph.addViewSqueeze(mapped_inputs[0], vs.axis),
        .ViewUnsqueeze => |vu| try graph.addViewUnsqueeze(mapped_inputs[0], vu.axis),
        .ViewTranspose2D => try graph.addViewTranspose2D(mapped_inputs[0]),
        .ViewSliceND => |sl| blk: {
            const lens = try package_file.resolveShapeTerms(graph.arenaAlloc(), pkg, sl.lens, optional_symbols);
            var starts_mem: [api_tiling.MAX_RANK]usize = undefined;
            if (sl.starts.len > starts_mem.len) return error.InvalidArgument;
            for (sl.starts, 0..) |value, idx| starts_mem[idx] = std.math.cast(usize, value) orelse return error.InvalidArgument;
            break :blk try graph.addViewSliceND(mapped_inputs[0], starts_mem[0..sl.starts.len], lens);
        },
        .Cast => |ct| try graph.addCast(mapped_inputs[0], ct.to_dtype),
        .MatMulNT => |mm| try graph.addMatMulNT(mapped_inputs[0], mapped_inputs[1], mm.alpha, mm.beta),
        .If => |iff| blk: {
            if (iff.then_region >= region_map.len or iff.else_region >= region_map.len) return error.InvalidArgument;
            break :blk try graph.addIf(mapped_inputs[0], region_map[iff.then_region], region_map[iff.else_region]);
        },
        .Loop => |lp| blk: {
            if (lp.body_region >= region_map.len) return error.InvalidArgument;
            const outs = try graph.addLoopMulti(
                mapped_inputs,
                region_map[lp.body_region],
                @intCast(lp.static_max_trip_count),
                if (lp.cond_carry) |c| @as(usize, @intCast(c)) else null,
                lp.check_before,
            );
            if (outs.len != extra_out.len + 1) return error.InvalidArgument;
            for (extra_out, 0..) |*e, i| e.* = outs[i + 1];
            break :blk outs[0];
        },
    };
}
