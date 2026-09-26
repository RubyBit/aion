// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Graph-to-executable lowering and validate-on-emit.
const std = @import("std");
const env = @import("../../env.zig");
const types = @import("../../backend/types.zig");
const derived = @import("../../storage/derived.zig");
const storage = @import("../../storage/storage.zig");
const executable = @import("../../runtime/executable.zig");

const graph_mod = @import("../graph.zig");
const infer_mod = @import("../infer.zig");
const opt_mod = @import("../opt.zig");
const placement = @import("placement.zig");
const workspace = @import("workspace.zig");
const allocation = @import("allocation.zig");
const reachability = @import("reachability.zig");
const manager_mod = @import("../../storage/manager.zig");
const target_mod = @import("../target.zig");

const backend_utils = @import("../../backend/utils.zig");
const diagnostic = @import("../../diagnostic.zig");

pub const StorageError = storage.StorageError;
pub const Tensor = storage.Tensor;

pub const StorageManager = manager_mod.StorageManager;
pub const TensorId = manager_mod.TensorId;

pub const Step = executable.Step;
pub const PlacedStep = executable.PlacedStep;
pub const Program = executable.ExecutableProgram;
pub const materializePlacements = workspace.materializePlacements;

const MAX_RANK: usize = 8;

pub const CompileError = error{ InvalidArgument, OutOfMemory } || graph_mod.GraphError || infer_mod.InferError || StorageError;

fn traceEnabled() bool {
    // Keep tracing opt-in to avoid log spam in normal runs/tests.
    return env.flagEnabled("AION_TRACE");
}

fn compileRequire(cond: bool) CompileError!void {
    if (!cond) return CompileError.InvalidArgument;
}

/// Which nodes a compile actually has to lower: those reachable backwards from
/// `graph.outputs`.
///
/// Without this, `compile(outputs)` lowers *every* node the graph has ever held,
/// so anything authored and not asked for — a discarded branch, an expression
/// evaluated while exploring — is compiled into the program and paid for at run
/// time. Keying off the requested outputs makes the program depend on what was
/// asked for rather than on the order things were written.
///
/// A node stays if *any* of its outputs is live (`Loop`/`If` produce several), and
/// its inputs then become live in turn. Nodes are appended in dependency order, so
/// one reverse sweep settles it.
///
/// A live control-flow node also keeps whatever its regions read. A region body
/// references outer values directly — they are not routed through the `If`/`Loop`
/// node's own inputs — so walking only `node.inputs` would prune a value that a
/// loop body is the sole consumer of, which is how this first broke the Nemotron
/// ASR graph.
fn liveNodes(allocator: std.mem.Allocator, graph: *const graph_mod.Graph) CompileError![]bool {
    return reachability.findLiveNodes(allocator, graph);
}

fn normalizeAxis(axis: i32, rank: usize) CompileError!usize {
    if (rank == 0) return CompileError.InvalidArgument;
    const r_i32: i32 = @intCast(rank);
    var ax: i32 = axis;
    if (ax < 0) ax += r_i32;
    if (ax < 0 or ax >= r_i32) return CompileError.InvalidArgument;
    return @intCast(ax);
}

fn productUsize(vals: []const usize) CompileError!usize {
    if (vals.len == 0) return CompileError.InvalidArgument;
    var acc: usize = 1;
    for (vals) |v| {
        acc = std.math.mul(usize, acc, v) catch return CompileError.InvalidArgument;
    }
    return acc;
}

fn requireSameShape(a: []const usize, b: []const usize) CompileError!void {
    if (a.len != b.len) return CompileError.InvalidArgument;
    var i: usize = 0;
    while (i < a.len) : (i += 1) {
        try compileRequire(a[i] == b[i]);
    }
}

fn shapesEqual(a: []const usize, b: []const usize) bool {
    return a.len == b.len and std.mem.eql(usize, a, b);
}

fn alignedDim(shape: []const usize, output_rank: usize, output_axis: usize) usize {
    const off = output_rank - shape.len;
    return if (output_axis < off) 1 else shape[output_axis - off];
}

fn broadcastAxes(shape: []const usize, output_shape: []const usize) u8 {
    var mask: u8 = 0;
    for (output_shape, 0..) |out_dim, axis| {
        if (alignedDim(shape, output_shape.len, axis) == 1 and out_dim != 1) {
            mask |= @as(u8, 1) << @intCast(axis);
        }
    }
    return mask;
}

fn isContiguousSuffix(shape: []const usize, output_shape: []const usize) bool {
    var seen_non_broadcast = false;
    for (output_shape, 0..) |out_dim, axis| {
        const dim = alignedDim(shape, output_shape.len, axis);
        if (dim == out_dim and out_dim != 1) {
            seen_non_broadcast = true;
        } else if (dim == 1 and !seen_non_broadcast) {
            continue;
        } else if (dim != out_dim) {
            return false;
        }
    }
    return seen_non_broadcast;
}

fn makeElementwiseBroadcastPlan(
    a_shape: []const usize,
    b_shape: []const usize,
    output_shape: []const usize,
) CompileError!executable.ElementwiseBroadcastPlan {
    if (output_shape.len == 0 or output_shape.len > MAX_RANK) return CompileError.InvalidArgument;
    const a_count = try elemCount(a_shape);
    const b_count = try elemCount(b_shape);
    const kind: executable.ElementwiseBroadcastKind = if (shapesEqual(a_shape, output_shape) and shapesEqual(b_shape, output_shape))
        .identical
    else if (a_count == 1)
        .scalar_a
    else if (b_count == 1)
        .scalar_b
    else if (shapesEqual(b_shape, output_shape) and isContiguousSuffix(a_shape, output_shape))
        .contiguous_suffix_a
    else if (shapesEqual(a_shape, output_shape) and isContiguousSuffix(b_shape, output_shape))
        .contiguous_suffix_b
    else
        .generic;
    return .{
        .kind = kind,
        .output_rank = @intCast(output_shape.len),
        .a_broadcast_axes = broadcastAxes(a_shape, output_shape),
        .b_broadcast_axes = broadcastAxes(b_shape, output_shape),
    };
}

fn elemCount(shape: []const usize) CompileError!usize {
    return backend_utils.elemCount(shape) catch return CompileError.InvalidArgument;
}

fn convOutDim(in_len: usize, kernel: usize, stride: usize, dilation: usize, pad_before: usize, pad_after: usize) CompileError!usize {
    if (kernel == 0 or stride == 0 or dilation == 0) return CompileError.InvalidArgument;
    const eff_kernel_sub1: usize = std.math.mul(usize, dilation, kernel - 1) catch return CompileError.InvalidArgument;
    const eff_kernel: usize = std.math.add(usize, eff_kernel_sub1, 1) catch return CompileError.InvalidArgument;
    const padded: usize = std.math.add(usize, std.math.add(usize, in_len, pad_before) catch return CompileError.InvalidArgument, pad_after) catch return CompileError.InvalidArgument;
    if (padded < eff_kernel) return CompileError.InvalidArgument;
    const numer: usize = padded - eff_kernel;
    return (numer / stride) + 1;
}

fn isScalarSupported(dtype: types.DType) bool {
    return switch (dtype) {
        .f32, .f16, .i8, .i32 => true,
        else => false,
    };
}

fn validateStep(mgr: *StorageManager, step: Step) CompileError!void {
    switch (step) {
        .Transfer => |s| {
            const src: *const Tensor = mgr.getConst(s.src) catch return CompileError.InvalidArgument;
            const dst: *const Tensor = mgr.getConst(s.dst) catch return CompileError.InvalidArgument;
            try compileRequire(src.dtype == dst.dtype);
            try compileRequire(src.rank == dst.rank);
            try compileRequire(std.mem.eql(usize, src.shape, dst.shape));
        },
        .MatMul => |s| {
            const c: *const Tensor = mgr.getConst(s.c) catch return CompileError.InvalidArgument;
            const a: *const Tensor = mgr.getConst(s.a) catch return CompileError.InvalidArgument;
            const b: *const Tensor = mgr.getConst(s.b) catch return CompileError.InvalidArgument;

            try compileRequire(c.rank >= 2);
            try compileRequire(a.rank == c.rank);
            // B may carry fewer batch dims and broadcast into the rest, right-aligned
            try compileRequire(b.rank >= 2 and b.rank <= c.rank);
            try compileRequire(!a.dtype.info().is_quantized);
            try compileRequire(!c.dtype.info().is_quantized);
            try compileRequire(isScalarSupported(a.dtype));
            try compileRequire(isScalarSupported(c.dtype));

            const rank: usize = @as(usize, c.rank);
            const b_rank: usize = @as(usize, b.rank);
            const b_off: usize = rank - b_rank;
            var d: usize = 0;
            while (d + 2 < rank) : (d += 1) {
                const ad: usize = a.shape[d];
                // A dim B does not have is a broadcast dim, exactly like a size-1 one.
                const bd: usize = if (d >= b_off) b.shape[d - b_off] else 1;
                const cd: usize = c.shape[d];
                if (ad != bd and ad != 1 and bd != 1) return CompileError.InvalidArgument;
                try compileRequire(cd == @max(ad, bd));
            }

            // DType contract (v0):
            // - Quantized B: A and C must be f32.
            // - Non-quantized B: A and B must match.
            //   * f32: C must be f32
            //   * f16: C may be f16 or f32 (promotion)
            if (b.dtype.info().is_quantized) {
                try compileRequire(a.dtype == .f32);
                try compileRequire(c.dtype == .f32);
            } else {
                try compileRequire(isScalarSupported(b.dtype));
                try compileRequire(b.dtype == a.dtype);
                switch (b.dtype) {
                    .f32 => try compileRequire(c.dtype == .f32),
                    .f16 => try compileRequire(c.dtype == .f16 or c.dtype == .f32),
                    else => return CompileError.InvalidArgument,
                }
            }

            // Shapes (batched): [..., m, k] @ [..., k, n] -> [..., m, n].
            try compileRequire(a.shape[rank - 1] == b.shape[b_rank - 2]);
            try compileRequire(c.shape[rank - 2] == a.shape[rank - 2]);
            try compileRequire(c.shape[rank - 1] == b.shape[b_rank - 1]);
            if (b.dtype.info().is_quantized) try compileRequire(a.shape[rank - 1] % b.dtype.info().block_elems == 0);
        },

        .ElemwiseBinary => |s| {
            const out: *const Tensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const a: *const Tensor = mgr.getConst(s.a) catch return CompileError.InvalidArgument;
            const b: *const Tensor = mgr.getConst(s.b) catch return CompileError.InvalidArgument;

            try compileRequire(isScalarSupported(out.dtype));
            try compileRequire(out.dtype == a.dtype and out.dtype == b.dtype);
            try compileRequire(out.rank == s.broadcast.output_rank);
            try compileRequire(a.rank <= out.rank and b.rank <= out.rank);
            const rank: usize = out.rank;
            for (0..rank) |axis| {
                const ad = alignedDim(a.shape, rank, axis);
                const bd = alignedDim(b.shape, rank, axis);
                try compileRequire(ad == bd or ad == 1 or bd == 1);
                try compileRequire(out.shape[axis] == @max(ad, bd));
            }
            // A gate fuses an activation into the multiply, so it is one kernel over
            // three matching buffers: f32, identical shape, no broadcast.
            if (s.op == .gate) {
                try compileRequire(out.dtype == .f32);
                try compileRequire(s.broadcast.kind == .identical);
                try compileRequire(out.rank == a.rank and out.rank == b.rank);
                try requireSameShape(out.shape, a.shape);
                try requireSameShape(out.shape, b.shape);
            }
        },

        .Unary => |s| {
            const out: *const Tensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const a: *const Tensor = mgr.getConst(s.a) catch return CompileError.InvalidArgument;
            try compileRequire(isScalarSupported(out.dtype));
            try compileRequire(out.dtype == a.dtype);
            try compileRequire(out.rank == a.rank);
            try requireSameShape(out.shape, a.shape);
            _ = s.op;
        },

        .Cast => |s| {
            const out: *const Tensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const x: *const Tensor = mgr.getConst(s.x) catch return CompileError.InvalidArgument;
            try compileRequire(out.dtype == s.to_dtype);
            try compileRequire(!x.dtype.info().is_quantized);
            try compileRequire(!s.to_dtype.info().is_quantized);
            // Supported: f16<->f32, f32<->i32, and no-op same-dtype.
            const from = x.dtype;
            const to = s.to_dtype;
            const ok = (from == .f16 and to == .f32) or (from == .f32 and to == .f16) or
                (from == .f32 and to == .i32) or (from == .i32 and to == .f32) or (from == to);
            try compileRequire(ok);
            try compileRequire(out.rank == x.rank);
            try requireSameShape(out.shape, x.shape);
        },

        .If => |s| {
            const cond: *const Tensor = mgr.getConst(s.cond) catch return CompileError.InvalidArgument;
            try compileRequire(cond.dtype == .i32 and cond.rank == 1 and cond.shape[0] == 1);
            const count: usize = @intCast(s.output_count);
            try compileRequire(count <= executable.MAX_CONTROL_OUTPUTS);
            var i: usize = 0;
            while (i < count) : (i += 1) {
                const out: *const Tensor = mgr.getConst(s.outputs[i]) catch return CompileError.InvalidArgument;
                const then_out: *const Tensor = mgr.getConst(s.then_outputs[i]) catch return CompileError.InvalidArgument;
                const else_out: *const Tensor = mgr.getConst(s.else_outputs[i]) catch return CompileError.InvalidArgument;
                try compileRequire(out.dtype == then_out.dtype and out.dtype == else_out.dtype);
                try compileRequire(out.rank == then_out.rank and out.rank == else_out.rank);
                try requireSameShape(out.shape, then_out.shape);
                try requireSameShape(out.shape, else_out.shape);
            }
            _ = s.then_block;
            _ = s.else_block;
        },

        .Loop => |s| {
            if (s.trip_count) |trip_count| {
                const trip: *const Tensor = mgr.getConst(trip_count) catch return CompileError.InvalidArgument;
                try compileRequire(trip.dtype == .i32 and trip.rank == 1 and trip.shape[0] == 1);
            }
            if (s.cond) |cond_id| {
                const cond: *const Tensor = mgr.getConst(cond_id) catch return CompileError.InvalidArgument;
                try compileRequire(cond.dtype == .i32 and cond.rank == 1 and cond.shape[0] == 1);
            }
            try compileRequire(s.static_max_trip_count > 0);
            const count: usize = @intCast(s.carried_count);
            try compileRequire(count <= executable.MAX_LOOP_CARRIED);
            var i: usize = 0;
            while (i < count) : (i += 1) {
                const carried: *const Tensor = mgr.getConst(s.carried[i]) catch return CompileError.InvalidArgument;
                const next: *const Tensor = mgr.getConst(s.body_carried_outputs[i]) catch return CompileError.InvalidArgument;
                try compileRequire(carried.dtype == next.dtype);
                try compileRequire(carried.rank == next.rank);
                try requireSameShape(carried.shape, next.shape);
            }
            _ = s.check_before;
            _ = s.body_block;
        },

        .MatMulNT => |s| {
            const c: *const Tensor = mgr.getConst(s.c) catch return CompileError.InvalidArgument;
            const a: *const Tensor = mgr.getConst(s.a) catch return CompileError.InvalidArgument;
            const b: *const Tensor = mgr.getConst(s.b) catch return CompileError.InvalidArgument;
            try compileRequire(a.dtype == .f32 and c.dtype == .f32);
            // The CPU/GPU executors implement q8_0 (per-row blocks) and plain f32 B.
            try compileRequire(b.dtype == .q8_0 or b.dtype == .f32);
            try compileRequire(b.rank == 2);
            if (b.dtype.info().is_quantized) try compileRequire(b.quant_axis == 1);
            try compileRequire(a.rank == c.rank);
            // Trailing dims: A's last == B's K (= b.shape[1]); C's last == B's N (= b.shape[0]).
            try compileRequire(a.shape[a.rank - 1] == b.shape[1]);
            try compileRequire(c.shape[c.rank - 1] == b.shape[0]);
            // Leading dims share.
            var d: usize = 0;
            while (d + 1 < a.rank) : (d += 1) {
                try compileRequire(a.shape[d] == c.shape[d]);
            }
            _ = s.alpha;
            _ = s.beta;
        },

        .Softmax => |s| {
            const out: *const Tensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const a: *const Tensor = mgr.getConst(s.a) catch return CompileError.InvalidArgument;
            // v0: f32 only (fast + stable).
            try compileRequire((out.dtype == .f32 or out.dtype == .f16) and a.dtype == out.dtype);
            try compileRequire(out.rank == a.rank);
            const rank: usize = @as(usize, out.rank);
            try compileRequire(rank >= 1 and rank <= MAX_RANK);
            _ = try normalizeAxis(s.axis, rank);
            try requireSameShape(out.shape, a.shape);
        },

        .Conv1D => |s| {
            const out: *const Tensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const x: *const Tensor = mgr.getConst(s.x) catch return CompileError.InvalidArgument;
            const w: *const Tensor = mgr.getConst(s.w) catch return CompileError.InvalidArgument;

            try compileRequire(out.dtype == .f32 and x.dtype == .f32 and w.dtype == .f32);
            try compileRequire(out.rank == x.rank and out.rank >= 2);
            try compileRequire(w.rank == 3);
            try compileRequire(s.groups > 0);
            try compileRequire(s.stride > 0 and s.dilation > 0);

            const rank: usize = @as(usize, out.rank);
            const l_in: usize = x.shape[rank - 2];
            const c_in: usize = x.shape[rank - 1];
            const k: usize = w.shape[0];
            const c_in_g: usize = w.shape[1];
            const c_out: usize = w.shape[2];

            try compileRequire(c_in % s.groups == 0);
            try compileRequire(c_out % s.groups == 0);
            try compileRequire(c_in_g * s.groups == c_in);

            const l_out: usize = try convOutDim(l_in, k, s.stride, s.dilation, s.pad_left, s.pad_right);
            try compileRequire(out.shape[rank - 2] == l_out);
            try compileRequire(out.shape[rank - 1] == c_out);

            if (s.pad_mode == .reflect) {
                try compileRequire(l_in > 1);
                try compileRequire(s.pad_left < l_in and s.pad_right < l_in);
            }

            var d: usize = 0;
            while (d + 2 < rank) : (d += 1) {
                try compileRequire(out.shape[d] == x.shape[d]);
            }

            if (s.bias) |bias_id| {
                const b: *const Tensor = mgr.getConst(bias_id) catch return CompileError.InvalidArgument;
                try compileRequire(b.dtype == .f32);
                try compileRequire(b.rank == 1 and b.shape[0] == c_out);
            }
        },

        .MaxPool2D => |s| {
            const x = try mgr.getConst(s.x);
            const out = try mgr.getConst(s.out);
            const shape = try s.opts.output(x.shape);
            if (!std.mem.eql(usize, &shape, out.shape) or x.dtype != out.dtype or (x.dtype != .f32 and x.dtype != .f16)) return CompileError.InvalidArgument;
        },
        .Conv2D => |s| {
            const out: *const Tensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const x: *const Tensor = mgr.getConst(s.x) catch return CompileError.InvalidArgument;
            const w: *const Tensor = mgr.getConst(s.w) catch return CompileError.InvalidArgument;

            try compileRequire(out.dtype == .f32 and x.dtype == .f32 and w.dtype == .f32);
            try compileRequire(out.rank == x.rank and out.rank >= 3);
            try compileRequire(w.rank == 4);
            try compileRequire(s.groups > 0);
            try compileRequire(s.stride_h > 0 and s.stride_w > 0 and s.dilation_h > 0 and s.dilation_w > 0);

            const rank: usize = @as(usize, out.rank);
            const h_in: usize = x.shape[rank - 3];
            const w_in: usize = x.shape[rank - 2];
            const c_in: usize = x.shape[rank - 1];

            const k_h: usize = w.shape[0];
            const k_w: usize = w.shape[1];
            const c_in_g: usize = w.shape[2];
            const c_out: usize = w.shape[3];

            try compileRequire(c_in % s.groups == 0);
            try compileRequire(c_out % s.groups == 0);
            try compileRequire(c_in_g * s.groups == c_in);

            const h_out: usize = try convOutDim(h_in, k_h, s.stride_h, s.dilation_h, s.pad_top, s.pad_bottom);
            const w_out: usize = try convOutDim(w_in, k_w, s.stride_w, s.dilation_w, s.pad_left, s.pad_right);
            try compileRequire(out.shape[rank - 3] == h_out);
            try compileRequire(out.shape[rank - 2] == w_out);
            try compileRequire(out.shape[rank - 1] == c_out);

            if (s.pad_mode == .reflect) {
                try compileRequire(h_in > 1 and w_in > 1);
                try compileRequire(s.pad_top < h_in and s.pad_bottom < h_in);
                try compileRequire(s.pad_left < w_in and s.pad_right < w_in);
            }

            var d: usize = 0;
            while (d + 3 < rank) : (d += 1) {
                try compileRequire(out.shape[d] == x.shape[d]);
            }

            if (s.bias) |bias_id| {
                const b: *const Tensor = mgr.getConst(bias_id) catch return CompileError.InvalidArgument;
                try compileRequire(b.dtype == .f32);
                try compileRequire(b.rank == 1 and b.shape[0] == c_out);
            }
        },

        .LayerNorm => |s| {
            const out: *const Tensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const x: *const Tensor = mgr.getConst(s.x) catch return CompileError.InvalidArgument;
            const gamma: *const Tensor = mgr.getConst(s.gamma) catch return CompileError.InvalidArgument;
            const beta: *const Tensor = mgr.getConst(s.beta) catch return CompileError.InvalidArgument;

            try compileRequire(out.rank == x.rank);
            try compileRequire(out.rank >= 1 and out.rank <= MAX_RANK);
            try compileRequire(gamma.rank == beta.rank);
            try compileRequire(gamma.rank >= 1);
            try compileRequire(isScalarSupported(out.dtype));
            try compileRequire(out.dtype == x.dtype and out.dtype == gamma.dtype and out.dtype == beta.dtype);

            try requireSameShape(out.shape, x.shape);

            const rank: usize = @as(usize, out.rank);
            const norm_rank: usize = @as(usize, gamma.rank);
            try compileRequire(rank >= norm_rank);

            // normalized_shape matches trailing dims.
            var d: usize = 0;
            while (d < norm_rank) : (d += 1) {
                const od: usize = out.shape[rank - norm_rank + d];
                try compileRequire(gamma.shape[d] == od);
                try compileRequire(beta.shape[d] == od);
            }

            try compileRequire(s.eps > 0.0);
        },

        .RMSNorm => |s| {
            const out: *const Tensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const x: *const Tensor = mgr.getConst(s.x) catch return CompileError.InvalidArgument;
            const gamma: *const Tensor = mgr.getConst(s.gamma) catch return CompileError.InvalidArgument;
            const beta: *const Tensor = mgr.getConst(s.beta) catch return CompileError.InvalidArgument;

            try compileRequire(out.rank == x.rank);
            try compileRequire(out.rank >= 1 and out.rank <= MAX_RANK);
            try compileRequire(gamma.rank == beta.rank);
            try compileRequire(gamma.rank >= 1);
            try compileRequire(isScalarSupported(out.dtype));
            try compileRequire(out.dtype == x.dtype and out.dtype == gamma.dtype and out.dtype == beta.dtype);

            try requireSameShape(out.shape, x.shape);

            const rank: usize = @as(usize, out.rank);
            const norm_rank: usize = @as(usize, gamma.rank);
            try compileRequire(rank >= norm_rank);

            var d: usize = 0;
            while (d < norm_rank) : (d += 1) {
                const od: usize = out.shape[rank - norm_rank + d];
                try compileRequire(gamma.shape[d] == od);
                try compileRequire(beta.shape[d] == od);
            }

            try compileRequire(s.eps > 0.0);

            // A residual is a configuration of this step, so its contract is checked here
            // rather than by a tag of its own: the fused kernel is one workgroup per row,
            // so it normalizes one trailing axis, and the residual matches `out` exactly.
            if (s.residual) |res| {
                const r: *const Tensor = mgr.getConst(res) catch return CompileError.InvalidArgument;
                try compileRequire(out.dtype == .f32 and r.dtype == .f32);
                try compileRequire(r.rank == out.rank);
                try requireSameShape(out.shape, r.shape);
                try compileRequire(norm_rank == 1);
            }
        },

        .RelPosMHA => |s| {
            const out: *const Tensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const q: *const Tensor = mgr.getConst(s.q) catch return CompileError.InvalidArgument;
            const k: *const Tensor = mgr.getConst(s.k) catch return CompileError.InvalidArgument;
            const v: *const Tensor = mgr.getConst(s.v) catch return CompileError.InvalidArgument;
            const pe: *const Tensor = mgr.getConst(s.pos_emb) catch return CompileError.InvalidArgument;
            const bu: *const Tensor = mgr.getConst(s.pos_bias_u) catch return CompileError.InvalidArgument;
            const bv: *const Tensor = mgr.getConst(s.pos_bias_v) catch return CompileError.InvalidArgument;

            try compileRequire(s.scale > 0.0 and std.math.isFinite(s.scale));
            try compileRequire(std.math.isFinite(s.attn_logits_soft_cap) and s.attn_logits_soft_cap >= 0);

            // q,k,v,out:[B,H,T*,D]; pos_emb:[H,P,D]; pos_bias_u/_v:[H,D]
            try compileRequire(out.rank == 4 and q.rank == 4 and k.rank == 4 and v.rank == 4);
            try compileRequire(pe.rank == 3 and bu.rank == 2 and bv.rank == 2);
            try compileRequire(out.dtype == .f32 and q.dtype == .f32 and k.dtype == .f32 and v.dtype == .f32);
            try compileRequire(pe.dtype == .f32 and bu.dtype == .f32 and bv.dtype == .f32);

            // Layout [B, T*, H, D].
            const B: usize = q.shape[0];
            const t_q: usize = q.shape[1];
            const H: usize = q.shape[2];
            const d: usize = q.shape[3];
            const t_kv: usize = k.shape[1];
            const p_len: usize = pe.shape[1];

            try compileRequire(H > 0);
            try compileRequire(out.shape[0] == B and out.shape[1] == t_q and out.shape[2] == H and out.shape[3] == d);
            try compileRequire(k.shape[0] == B and k.shape[2] == H and k.shape[3] == d);
            try compileRequire(v.shape[0] == B and v.shape[1] == t_kv and v.shape[2] == H and v.shape[3] == d);
            try compileRequire(pe.shape[0] == H and pe.shape[2] == d);
            try compileRequire(bu.shape[0] == H and bu.shape[1] == d);
            try compileRequire(bv.shape[0] == H and bv.shape[1] == d);
            try compileRequire(t_kv > 0 and p_len > 0 and s.relative_zero_index < p_len);
            try compileRequire(t_q <= t_kv);

            if (s.mask) |mask_id| {
                const m: *const Tensor = mgr.getConst(mask_id) catch return CompileError.InvalidArgument;
                try compileRequire(m.rank == 2 and m.dtype == .f32);
                try compileRequire(m.shape[0] == t_q and m.shape[1] == t_kv);
            }
        },

        .ArgMax => |s| {
            const a: *const Tensor = mgr.getConst(s.a) catch return CompileError.InvalidArgument;
            const out: *const Tensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            try compileRequire((a.dtype == .f32 or a.dtype == .f16) and out.dtype == .i32);
            try compileRequire(@as(usize, a.rank) >= 1);
            try compileRequire(s.axis == @as(usize, a.rank) - 1);
        },

        .TopK => |s| {
            const a: *const Tensor = mgr.getConst(s.a) catch return CompileError.InvalidArgument;
            const values: *const Tensor = mgr.getConst(s.values) catch return CompileError.InvalidArgument;
            const indices: *const Tensor = mgr.getConst(s.indices) catch return CompileError.InvalidArgument;
            try compileRequire(a.dtype == .f32 or a.dtype == .f16);
            try compileRequire(values.dtype == a.dtype and indices.dtype == .i32);
            try compileRequire(@as(usize, a.rank) >= 1 and s.axis == @as(usize, a.rank) - 1);
            try compileRequire(values.rank == a.rank and indices.rank == a.rank);
            try compileRequire(s.k != 0 and s.k <= a.shape[s.axis]);
            var d: usize = 0;
            while (d < @as(usize, a.rank)) : (d += 1) {
                const want: usize = if (d == s.axis) s.k else a.shape[d];
                try compileRequire(values.shape[d] == want and indices.shape[d] == want);
            }
        },

        .ScatterRow => |s| {
            const buf: *const Tensor = mgr.getConst(s.buf) catch return CompileError.InvalidArgument;
            const idx: *const Tensor = mgr.getConst(s.idx) catch return CompileError.InvalidArgument;
            const src: *const Tensor = mgr.getConst(s.src) catch return CompileError.InvalidArgument;
            try compileRequire(!buf.dtype.info().is_quantized);
            try compileRequire(idx.dtype == .i32 and src.dtype == buf.dtype);
            try compileRequire(@as(usize, buf.rank) >= 1);
        },

        .Attention => |s| {
            const out: *const Tensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const q: *const Tensor = mgr.getConst(s.q) catch return CompileError.InvalidArgument;
            const k: *const Tensor = mgr.getConst(s.k) catch return CompileError.InvalidArgument;
            const v: *const Tensor = mgr.getConst(s.v) catch return CompileError.InvalidArgument;

            // Attention accepts q/k/v in {f16,f32} and accumulates in f32.
            try compileRequire(out.dtype == .f32);
            try compileRequire((q.dtype == .f16 or q.dtype == .f32));
            try compileRequire((k.dtype == .f16 or k.dtype == .f32));
            try compileRequire((v.dtype == .f16 or v.dtype == .f32));

            try compileRequire(out.rank == 4 and q.rank == 4 and k.rank == 4 and v.rank == 4);

            // q:[B,L_q,H_q,D_k], k/v:[B,T,H_kv,D_*], out:[B,L_q,H_q,D_v]
            try compileRequire(q.shape[0] == k.shape[0] and q.shape[0] == v.shape[0]);
            try compileRequire(q.shape[1] == out.shape[1]);
            try compileRequire(q.shape[2] == out.shape[2]);
            try compileRequire(k.shape[1] == v.shape[1]);
            try compileRequire(k.shape[2] == v.shape[2]);
            try compileRequire(q.shape[3] == k.shape[3]);
            try compileRequire(out.shape[3] == v.shape[3]);
            try compileRequire(out.shape[0] == q.shape[0]);

            // GQA constraint.
            try compileRequire(k.shape[2] > 0 and q.shape[2] > 0);
            try compileRequire((q.shape[2] % k.shape[2]) == 0);

            if (s.query_positions) |pos_tid| {
                const positions: *const Tensor = mgr.getConst(pos_tid) catch return CompileError.InvalidArgument;
                try compileRequire(positions.dtype == .i32 and positions.rank == 2);
                try compileRequire(positions.shape[0] == q.shape[0]);
                try compileRequire(positions.shape[1] == q.shape[1]);
            }
            if (s.kv_lengths) |lengths_tid| {
                const lengths: *const Tensor = mgr.getConst(lengths_tid) catch return CompileError.InvalidArgument;
                try compileRequire(lengths.dtype == .i32 and lengths.rank == 1);
                try compileRequire(lengths.shape[0] == q.shape[0]);
            }

            try compileRequire(s.scale > 0.0 and std.math.isFinite(s.scale));
            try compileRequire(std.math.isFinite(s.attn_logits_soft_cap));
            try compileRequire(s.attn_logits_soft_cap >= 0.0);
            _ = s.window;
        },

        .Copy => |s| {
            const dst: *const Tensor = mgr.getConst(s.dst) catch return CompileError.InvalidArgument;
            const src: *const Tensor = mgr.getConst(s.src) catch return CompileError.InvalidArgument;
            try compileRequire(dst.dtype == src.dtype);
            try compileRequire(dst.rank == src.rank);
            try requireSameShape(dst.shape, src.shape);
        },

        .GatherRows => |s| {
            const out: *const Tensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const table: *const Tensor = mgr.getConst(s.table) catch return CompileError.InvalidArgument;
            const indices: *const Tensor = mgr.getConst(s.indices) catch return CompileError.InvalidArgument;

            try compileRequire(out.rank == 3);
            try compileRequire(table.rank == 2);
            try compileRequire(indices.rank == 2);

            try compileRequire(!out.dtype.info().is_quantized);
            try compileRequire(!indices.dtype.info().is_quantized);

            // Table dtype / output dtype contract:
            //   - scalar table (f16/f32): output matches table exactly.
            //   - q8_0 table (per-row quantized, `quant_axis == 1`): output is f32 or f16,
            //     and the kernel dequantizes rows on read.
            switch (table.dtype) {
                .f16, .f32 => {
                    try compileRequire(out.dtype == table.dtype);
                },
                .q8_0 => {
                    try compileRequire(out.dtype == .f32 or out.dtype == .f16);
                    try compileRequire(table.quant_axis == 1);
                    try compileRequire((table.shape[1] % 32) == 0);
                },
                else => return CompileError.InvalidArgument,
            }

            // Indices: i32.
            try compileRequire(indices.dtype == .i32);

            // Shape contract.
            try compileRequire(out.shape[0] == indices.shape[0]);
            try compileRequire(out.shape[1] == indices.shape[1]);
            try compileRequire(out.shape[2] == table.shape[1]);
        },

        .GatherND => |s| {
            const out = try mgr.getConst(s.out);
            const data = try mgr.getConst(s.data);
            const idx = try mgr.getConst(s.indices);
            if (idx.dtype != .i32 or out.dtype != data.dtype) return CompileError.InvalidArgument;
            if (s.axis >= data.rank or s.batch_dims > s.axis or s.batch_dims > idx.rank) return CompileError.InvalidArgument;
        },
        .Gather => |s| {
            const out = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const data = mgr.getConst(s.data) catch return CompileError.InvalidArgument;
            const indices = mgr.getConst(s.indices) catch return CompileError.InvalidArgument;
            try compileRequire(s.axis == 1 and s.batch_dims == 1);
            try compileRequire(data.rank == 3 and indices.rank == 2 and out.rank == 3);
            try compileRequire(indices.dtype == .i32);
            try compileRequire(data.dtype == out.dtype and (data.dtype == .f16 or data.dtype == .f32));
            try compileRequire(data.shape[0] == indices.shape[0]);
            try compileRequire(out.shape[0] == indices.shape[0]);
            try compileRequire(out.shape[1] == indices.shape[1]);
            try compileRequire(out.shape[2] == data.shape[2]);
        },

        .RoPE1D => |s| {
            const out: *const Tensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const x: *const Tensor = mgr.getConst(s.x) catch return CompileError.InvalidArgument;
            const positions: *const Tensor = mgr.getConst(s.positions) catch return CompileError.InvalidArgument;

            try compileRequire(out.rank == 4);
            try compileRequire(x.rank == 4);
            try compileRequire(positions.rank == 2);

            try compileRequire(!out.dtype.info().is_quantized);
            try compileRequire(!x.dtype.info().is_quantized);
            try compileRequire(!positions.dtype.info().is_quantized);

            try compileRequire(out.dtype == x.dtype);
            try compileRequire(out.dtype == .f16 or out.dtype == .f32);
            try compileRequire(positions.dtype == .i32);

            try requireSameShape(out.shape, x.shape);

            try compileRequire(positions.shape[0] == out.shape[0]);
            try compileRequire(positions.shape[1] == out.shape[1]);

            try compileRequire(s.base_frequency > 0.0 and std.math.isFinite(s.base_frequency));
            try compileRequire(s.scale_factor > 0.0 and std.math.isFinite(s.scale_factor));
            try compileRequire(std.math.isFinite(s.rope_proportion));
            try compileRequire(s.rope_proportion >= 0.0 and s.rope_proportion <= 1.0);
        },

        .SequenceAppend => |s| {
            const cache: *const Tensor = mgr.getConst(s.cache) catch return CompileError.InvalidArgument;
            const new_kv: *const Tensor = mgr.getConst(s.new_kv) catch return CompileError.InvalidArgument;
            const end_idx: *const Tensor = mgr.getConst(s.end_index) catch return CompileError.InvalidArgument;

            try compileRequire(cache.rank == 4 and new_kv.rank == 4 and end_idx.rank == 1);

            try compileRequire(!cache.dtype.info().is_quantized);
            try compileRequire(!new_kv.dtype.info().is_quantized);
            try compileRequire(!end_idx.dtype.info().is_quantized);

            try compileRequire(cache.dtype == new_kv.dtype);
            try compileRequire(cache.dtype == .f16 or cache.dtype == .f32);
            try compileRequire(end_idx.dtype == .i32);

            // Shape contract.
            try compileRequire(cache.shape[0] == new_kv.shape[0]); // B
            try compileRequire(cache.shape[2] == new_kv.shape[2]); // H_kv
            try compileRequire(cache.shape[3] == new_kv.shape[3]); // D
            try compileRequire(end_idx.shape[0] == cache.shape[0]);
        },

        .LSTMCellFused => |s| {
            const out_state: *const Tensor = mgr.getConst(s.out_state) catch return CompileError.InvalidArgument;
            const x: *const Tensor = mgr.getConst(s.x) catch return CompileError.InvalidArgument;
            const h_prev: *const Tensor = mgr.getConst(s.h_prev) catch return CompileError.InvalidArgument;
            const c_prev: *const Tensor = mgr.getConst(s.c_prev) catch return CompileError.InvalidArgument;
            const w_ih: *const Tensor = mgr.getConst(s.w_ih) catch return CompileError.InvalidArgument;
            const w_hh: *const Tensor = mgr.getConst(s.w_hh) catch return CompileError.InvalidArgument;

            try compileRequire(isScalarSupported(out_state.dtype));
            try compileRequire(out_state.dtype == x.dtype);
            try compileRequire(out_state.dtype == h_prev.dtype);
            try compileRequire(out_state.dtype == c_prev.dtype);
            try compileRequire(out_state.dtype == w_ih.dtype);
            try compileRequire(out_state.dtype == w_hh.dtype);
            try compileRequire(!out_state.dtype.info().is_quantized);

            try compileRequire(out_state.rank == 2);
            try compileRequire(x.rank == 2);
            try compileRequire(h_prev.rank == 2);
            try compileRequire(c_prev.rank == 2);
            try compileRequire(w_ih.rank == 2);
            try compileRequire(w_hh.rank == 2);

            const batch: usize = x.shape[0];
            const input_size: usize = x.shape[1];
            const hidden: usize = h_prev.shape[1];
            try compileRequire(batch != 0 and input_size != 0 and hidden != 0);

            try compileRequire(h_prev.shape[0] == batch and c_prev.shape[0] == batch);
            try compileRequire(c_prev.shape[1] == hidden);

            const gate_dim: usize = std.math.mul(usize, hidden, 4) catch return CompileError.InvalidArgument;
            try compileRequire(w_ih.shape[0] == input_size and w_ih.shape[1] == gate_dim);
            try compileRequire(w_hh.shape[0] == hidden and w_hh.shape[1] == gate_dim);

            const out_dim: usize = std.math.mul(usize, hidden, 2) catch return CompileError.InvalidArgument;
            try compileRequire(out_state.shape[0] == batch and out_state.shape[1] == out_dim);

            if (s.b_ih) |bid| {
                const b_ih: *const Tensor = mgr.getConst(bid) catch return CompileError.InvalidArgument;
                try compileRequire(b_ih.dtype == out_state.dtype);
                try compileRequire(b_ih.rank == 1);
                try compileRequire(b_ih.shape[0] == gate_dim);
            } else {
                try compileRequire(s.b_hh == null);
            }

            if (s.b_hh) |bid| {
                const b_hh: *const Tensor = mgr.getConst(bid) catch return CompileError.InvalidArgument;
                try compileRequire(b_hh.dtype == out_state.dtype);
                try compileRequire(b_hh.rank == 1);
                try compileRequire(b_hh.shape[0] == gate_dim);
            }
        },

        .RFFT => |s| {
            const out: *const Tensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const x: *const Tensor = mgr.getConst(s.x) catch return CompileError.InvalidArgument;

            try compileRequire(out.dtype == .f32 and x.dtype == .f32);
            try compileRequire(x.rank >= 1 and out.rank == x.rank);

            const n_fft: usize = x.shape[@as(usize, x.rank) - 1];
            try compileRequire(n_fft == s.n_fft);
            try compileRequire(n_fft >= 4 and (n_fft & (n_fft - 1)) == 0);

            var d: usize = 0;
            while (d + 1 < @as(usize, x.rank)) : (d += 1) {
                try compileRequire(out.shape[d] == x.shape[d]);
            }
            try compileRequire(out.shape[@as(usize, out.rank) - 1] == n_fft + 2);
        },

        .STFT => |s| {
            const out: *const Tensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const signal: *const Tensor = mgr.getConst(s.signal) catch return CompileError.InvalidArgument;
            const window: *const Tensor = mgr.getConst(s.window) catch return CompileError.InvalidArgument;

            try compileRequire(out.dtype == .f32 and signal.dtype == .f32 and window.dtype == .f32);
            try compileRequire(signal.rank == 2 and window.rank == 1 and out.rank == 3);

            const n_fft: usize = s.n_fft;
            try compileRequire(n_fft >= 4 and (n_fft & (n_fft - 1)) == 0);
            try compileRequire(s.hop_length != 0);
            try compileRequire(window.shape[0] == n_fft);

            try compileRequire(out.shape[0] == signal.shape[0]);
            try compileRequire(out.shape[1] == s.num_frames);
            try compileRequire(out.shape[2] == n_fft + 2);
        },

        .ReduceAll => |s| {
            const out: *const Tensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const a: *const Tensor = mgr.getConst(s.a) catch return CompileError.InvalidArgument;
            try compileRequire(out.rank == 1 and out.shape[0] == 1);
            try compileRequire(isScalarSupported(out.dtype));
            try compileRequire(out.dtype == a.dtype);
            try compileRequire(isScalarSupported(a.dtype));
            _ = s.op;
        },

        .ReduceAxis => |s| {
            const out: *const Tensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const a: *const Tensor = mgr.getConst(s.a) catch return CompileError.InvalidArgument;
            try compileRequire(isScalarSupported(out.dtype));
            try compileRequire(out.dtype == a.dtype);
            try compileRequire(isScalarSupported(a.dtype));
            try compileRequire(a.dtype != .i32 or s.op == .sum);
            try compileRequire(a.rank >= 1 and a.rank <= MAX_RANK);

            const rank: usize = @as(usize, a.rank);
            try compileRequire(s.axis < rank);

            if (rank == 1) {
                try compileRequire(out.rank == 1 and out.shape[0] == 1);
            } else {
                try compileRequire(out.rank == rank - 1);
                var src_d: usize = 0;
                var dst_d: usize = 0;
                while (src_d < rank) : (src_d += 1) {
                    if (src_d == s.axis) continue;
                    try compileRequire(out.shape[dst_d] == a.shape[src_d]);
                    dst_d += 1;
                }
            }

            _ = s.op;
        },

        .ConcatScalar => |s| {
            const out: *const Tensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            try compileRequire(isScalarSupported(out.dtype));

            const count: usize = @as(usize, s.input_count);
            try compileRequire(count >= 1);
            const rank: usize = @as(usize, out.rank);
            try compileRequire(rank >= 1 and rank <= MAX_RANK);
            try compileRequire(s.axis < rank);

            var axis_sum: usize = 0;
            var i: usize = 0;
            while (i < count) : (i += 1) {
                const in_t: *const Tensor = mgr.getConst(s.inputs[i]) catch return CompileError.InvalidArgument;
                try compileRequire(in_t.dtype == out.dtype);
                try compileRequire(in_t.rank == out.rank);

                var d: usize = 0;
                while (d < rank) : (d += 1) {
                    if (d == s.axis) continue;
                    try compileRequire(in_t.shape[d] == out.shape[d]);
                }

                axis_sum = std.math.add(usize, axis_sum, in_t.shape[s.axis]) catch return CompileError.InvalidArgument;
            }

            try compileRequire(axis_sum == out.shape[s.axis]);
        },

        .ReshapeScalar => |s| {
            const dst: *const Tensor = mgr.getConst(s.dst) catch return CompileError.InvalidArgument;
            const src: *const Tensor = mgr.getConst(s.src) catch return CompileError.InvalidArgument;
            try compileRequire(isScalarSupported(dst.dtype));
            try compileRequire(dst.dtype == src.dtype);
            const src_elems: usize = try elemCount(src.shape);
            const dst_elems: usize = try elemCount(dst.shape);
            try compileRequire(src_elems == dst_elems);
        },

        .Transpose2DScalar => |s| {
            const dst: *const Tensor = mgr.getConst(s.dst) catch return CompileError.InvalidArgument;
            const src: *const Tensor = mgr.getConst(s.src) catch return CompileError.InvalidArgument;
            try compileRequire(isScalarSupported(dst.dtype));
            try compileRequire(dst.dtype == src.dtype);
            try compileRequire(dst.rank == 2 and src.rank == 2);
            try compileRequire(dst.shape[0] == src.shape[1] and dst.shape[1] == src.shape[0]);
        },

        .SliceNDScalar => |s| {
            const dst: *const Tensor = mgr.getConst(s.dst) catch return CompileError.InvalidArgument;
            const src: *const Tensor = mgr.getConst(s.src) catch return CompileError.InvalidArgument;
            try compileRequire(isScalarSupported(dst.dtype));
            try compileRequire(dst.dtype == src.dtype);
            try compileRequire(dst.rank == src.rank);

            const rank: usize = @as(usize, s.rank);
            try compileRequire(rank == @as(usize, dst.rank));
            try compileRequire(rank >= 1 and rank <= MAX_RANK);

            var d: usize = 0;
            while (d < rank) : (d += 1) {
                try compileRequire(s.starts[d] + dst.shape[d] <= src.shape[d]);
            }
        },
    }
}

fn appendStepChecked(
    allocator: std.mem.Allocator,
    mgr: *StorageManager,
    steps: *std.ArrayList(PlacedStep),
    step: Step,
) CompileError!void {
    validateStep(mgr, step) catch |e| {
        if (traceEnabled()) {
            std.debug.print("[aion][compile] validateStep failed: step={s} err={s}\n", .{ @tagName(step), @errorName(e) });
            debugDumpStep(mgr, step);
        }
        return e;
    };
    steps.append(allocator, .{ .op = step }) catch return CompileError.OutOfMemory;
}

fn debugDumpTensorMeta(mgr: *StorageManager, tid: TensorId, label: []const u8) void {
    const t: *const Tensor = mgr.getConst(tid) catch {
        std.debug.print("  {s}: <invalid tensor id {d}>\n", .{ label, tid });
        return;
    };
    std.debug.print(
        "  {s}: id={d} dtype={s} rank={d} shape={any} quant_axis={d}\n",
        .{ label, tid, @tagName(t.dtype), t.rank, t.shape, t.quant_axis },
    );
}

fn debugDumpStep(mgr: *StorageManager, step: Step) void {
    switch (step) {
        .SequenceAppend => |s| {
            debugDumpTensorMeta(mgr, s.cache, "cache");
            debugDumpTensorMeta(mgr, s.new_kv, "new_kv");
            debugDumpTensorMeta(mgr, s.end_index, "end_index");
        },
        .Attention => |s| {
            debugDumpTensorMeta(mgr, s.q, "q");
            debugDumpTensorMeta(mgr, s.k, "k");
            debugDumpTensorMeta(mgr, s.v, "v");
            if (s.query_positions) |t| debugDumpTensorMeta(mgr, t, "query_positions");
            if (s.kv_lengths) |t| debugDumpTensorMeta(mgr, t, "kv_lengths");
            debugDumpTensorMeta(mgr, s.out, "out");
        },
        .RoPE1D => |s| {
            debugDumpTensorMeta(mgr, s.x, "x");
            debugDumpTensorMeta(mgr, s.positions, "positions");
            debugDumpTensorMeta(mgr, s.out, "out");
        },
        .GatherRows => |s| {
            debugDumpTensorMeta(mgr, s.table, "table");
            debugDumpTensorMeta(mgr, s.indices, "indices");
            debugDumpTensorMeta(mgr, s.out, "out");
        },
        .GatherND => |s| {
            debugDumpTensorMeta(mgr, s.data, "data");
            debugDumpTensorMeta(mgr, s.indices, "indices");
            debugDumpTensorMeta(mgr, s.out, "out");
        },
        .Gather => |s| {
            debugDumpTensorMeta(mgr, s.data, "data");
            debugDumpTensorMeta(mgr, s.indices, "indices");
            debugDumpTensorMeta(mgr, s.out, "out");
        },
        .MatMul => |s| {
            debugDumpTensorMeta(mgr, s.a, "a");
            debugDumpTensorMeta(mgr, s.b, "b");
            debugDumpTensorMeta(mgr, s.c, "c");
        },
        .MatMulNT => |s| {
            debugDumpTensorMeta(mgr, s.a, "a");
            debugDumpTensorMeta(mgr, s.b, "b");
            debugDumpTensorMeta(mgr, s.c, "c");
        },
        else => {},
    }
}

pub const OptPolicy = opt_mod.Policy;
pub const Target = target_mod.Target;

pub fn compileGraph(
    allocator: std.mem.Allocator,
    graph: *graph_mod.Graph,
    mgr: *StorageManager,
    target: Target,
) CompileError!Program {
    diagnostic.current().clear();
    infer_mod.infer(graph) catch |e| {
        if (traceEnabled()) std.debug.print("[aion][compile] infer failed: {s}\n", .{@errorName(e)});
        return e;
    };

    const opt_ctx: opt_mod.Ctx = .{ .gpa = allocator, .mgr = mgr, .target = target };
    try opt_mod.graphPasses(opt_ctx, graph);

    // Map graph values -> concrete tensors.
    const v_count: usize = graph.values.items.len;
    var value_tensor: []TensorId = try allocator.alloc(TensorId, v_count);
    defer allocator.free(value_tensor);
    @memset(value_tensor, @as(TensorId, 0));

    var value_has_tensor: []bool = try allocator.alloc(bool, v_count);
    defer allocator.free(value_has_tensor);
    @memset(value_has_tensor, false);

    const value_is_param: []bool = try allocator.alloc(bool, v_count);
    defer allocator.free(value_is_param);
    @memset(value_is_param, false);

    var owned_tensors: std.ArrayList(TensorId) = .empty;
    errdefer {
        for (owned_tensors.items) |tid| mgr.releaseTensorData(tid) catch {};
        owned_tensors.deinit(allocator);
    }

    // External bindings.
    for (graph.values.items, 0..) |v, i| {
        if (v.external) |ext| {
            const tid: TensorId = @intCast(ext);
            const t: *const Tensor = try mgr.getConst(tid);
            if (v.dtype.? != t.dtype) return CompileError.InvalidArgument;
            if (v.shape.len != t.rank) return CompileError.InvalidArgument;
            var d: usize = 0;
            while (d < v.shape.len) : (d += 1) {
                if (v.shape[d] != t.shape[d]) return CompileError.InvalidArgument;
            }

            value_tensor[i] = tid;
            value_has_tensor[i] = true;
            value_is_param[i] = v.external_is_param;
        }
    }

    // Compile into a dynamic step list first.
    var steps: std.ArrayList(PlacedStep) = .empty;
    errdefer steps.deinit(allocator);

    var blocks: std.ArrayList(executable.Block) = .empty;
    errdefer {
        for (blocks.items) |block| allocator.free(block.steps);
        blocks.deinit(allocator);
    }

    var ctx: allocation.Context = .{ .allocator = allocator, .mgr = mgr, .device = target.device, .value_tensor = value_tensor, .value_has_tensor = value_has_tensor, .value_is_param = value_is_param, .owned_tensors = &owned_tensors };

    // Lower nodes in order, skipping any whose results nothing asked for.
    const live: []bool = try liveNodes(allocator, graph);
    defer allocator.free(live);

    for (graph.nodes.items, 0..) |node, idx| {
        if (!live[idx]) continue;
        try lowerTraced(allocator, graph, node, mgr, &ctx, &steps, &blocks);
    }

    var compiled: Program = .{
        .allocator = allocator,
        .steps = try steps.toOwnedSlice(allocator),
        .outputs = &.{},
    };
    errdefer {
        for (compiled.owned_tensors) |tid| mgr.releaseTensorData(tid) catch {};
        compiled.deinit();
    }
    compiled.blocks = try blocks.toOwnedSlice(allocator);

    compiled.outputs = try allocator.alloc(TensorId, graph.outputs.items.len);
    for (graph.outputs.items, 0..) |vid, i| {
        const idx: usize = @intCast(vid);
        if (!value_has_tensor[idx]) return CompileError.InvalidArgument;
        compiled.outputs[i] = value_tensor[idx];
    }

    try placement.place(allocator, mgr, &compiled, &owned_tensors, target.backendKind());
    var view_alias = try opt_mod.stepPasses(opt_ctx, &compiled, owned_tensors.items);
    defer view_alias.deinit();
    try workspace.plan(allocator, mgr, &compiled, owned_tensors.items, &view_alias);
    compiled.owned_tensors = try owned_tensors.toOwnedSlice(allocator);
    compiled.validatePlacements() catch return CompileError.InvalidArgument;
    return compiled;
}

/// Assign placements and make every host read explicit.
///
/// Every step executes at the compile target, and so does every tensor it
/// touches — except the CPU mirrors created here.
///
/// An operand the executor resolves on the CPU (a control-flow predicate the
/// runtime interprets, or an index the target declares its kernels cannot
/// consume) gets an inserted `Transfer` into a CPU-placed mirror, and the step
/// is rewritten to read that mirror. Placement is per-tensor, not per-run: device
/// residency outlives a single `execute`, so a value the device wrote in an
/// earlier run is still device-newer here even if this program only reads it.
/// Transferring on every declared host read is what keeps that sound.
///
/// Nothing here branches on the target: on a CPU target every value is already
/// CPU-placed, so every transfer elides and the schedule comes out unchanged.
fn ensureAnyTensor(ctx: anytype, value_index: usize) CompileError!TensorId {
    if (!ctx.value_has_tensor[value_index]) return CompileError.InvalidArgument;
    return ctx.value_tensor[value_index];
}

/// Lower a control-flow region's body into its own executable block. Mutually
/// recursive with `lowerNode` (so nested If/Loop work). Module-level so its
/// params don't shadow `compileGraph` locals (Zig forbids that for nested fns).
fn lowerRegionBlock(
    allocator: std.mem.Allocator,
    graph: *graph_mod.Graph,
    region: graph_mod.Region,
    mgr: *StorageManager,
    ctx: anytype,
    blocks: *std.ArrayList(executable.Block),
) CompileError!executable.BlockId {
    var region_steps: std.ArrayList(PlacedStep) = .empty;
    errdefer region_steps.deinit(allocator);

    for (region.nodes) |rnode| {
        try lowerTraced(allocator, graph, rnode, mgr, ctx, &region_steps, blocks);
    }

    const block_steps: []PlacedStep = try region_steps.toOwnedSlice(allocator);
    errdefer allocator.free(block_steps);
    const id: executable.BlockId = @intCast(blocks.items.len);
    blocks.append(allocator, .{ .steps = block_steps }) catch return CompileError.OutOfMemory;
    return id;
}

/// `lowerNode`, reporting which op refused. A rejected `compileRequire` is
/// otherwise an `InvalidArgument` with no hint which of ~30 ops raised it; nested
/// regions print innermost-first.
fn lowerTraced(
    allocator: std.mem.Allocator,
    graph: *graph_mod.Graph,
    node: graph_mod.Node,
    mgr: *StorageManager,
    ctx: anytype,
    steps: *std.ArrayList(PlacedStep),
    blocks: *std.ArrayList(executable.Block),
) CompileError!void {
    const first_step = steps.items.len;
    lowerNode(allocator, graph, node, mgr, ctx, steps, blocks) catch |e| {
        diagnostic.current().recordGraph(.lowering, graph, node, e);
        if (traceEnabled()) {
            std.debug.print(
                "[aion][compile] lowerNode failed: op={s} err={s}\n",
                .{ @tagName(node.op), @errorName(e) },
            );
            for (node.inputs, 0..) |in_id, i| {
                const v = graph.values.items[@intCast(in_id)];
                std.debug.print("  in[{d}]: dtype={?s} shape={any}\n", .{ i, if (v.dtype) |d| @tagName(d) else null, v.shape });
            }
            const ov = graph.values.items[@intCast(node.output)];
            std.debug.print("  out:   dtype={?s} shape={any}\n", .{ if (ov.dtype) |d| @tagName(d) else null, ov.shape });
        }
        return e;
    };
    for (steps.items[first_step..]) |*step| {
        if (step.origin == null) step.origin = .{ .output = node.output, .operation = @tagName(node.op) };
    }
}

/// The tensor holding input `i` of `node`.
fn input(ctx: anytype, node: graph_mod.Node, i: usize) CompileError!TensorId {
    return ensureAnyTensor(ctx, @intCast(node.inputs[i]));
}

/// Lower a single graph node into executable steps. Shared by the top-level node
/// loop and `lowerRegionBlock` (control-flow region bodies), so both paths get
/// identical op support. `ctx`/`steps`/`blocks` are pointers.
///
/// Every tensor is one flat buffer, so lowering only picks the step: an output is
/// created at its shape, and inputs are used where they are.
fn lowerNode(
    allocator: std.mem.Allocator,
    graph: *graph_mod.Graph,
    node: graph_mod.Node,
    mgr: *StorageManager,
    ctx: anytype,
    steps: *std.ArrayList(PlacedStep),
    blocks: *std.ArrayList(executable.Block),
) CompileError!void {
    if (!graph_mod.opInputCountValid(node.op, node.inputs.len)) return CompileError.InvalidArgument;
    const out_idx: usize = @intCast(node.output);
    const out_v = graph.values.items[out_idx];
    const out_dt: types.DType = out_v.dtype.?;
    const out_shape: []const usize = out_v.shape;
    const values = graph.values.items;

    switch (node.op) {
        .MatMul => |mm| {
            if (values[@intCast(node.inputs[0])].dtype.?.info().is_quantized) return CompileError.InvalidArgument;
            const c = try ctx.ensureValueTensor(out_idx, out_dt, out_shape);
            try appendStepChecked(allocator, mgr, steps, .{ .MatMul = .{ .c = c, .a = try input(ctx, node, 0), .b = try input(ctx, node, 1), .alpha = mm.alpha, .beta = mm.beta } });
        },

        .ElemwiseBinary => |eb| {
            const a_v = values[@intCast(node.inputs[0])];
            const b_v = values[@intCast(node.inputs[1])];
            const out = try ctx.ensureValueTensor(out_idx, out_dt, out_shape);
            try appendStepChecked(allocator, mgr, steps, .{ .ElemwiseBinary = .{
                .op = eb.op,
                .out = out,
                .a = try input(ctx, node, 0),
                .b = try input(ctx, node, 1),
                .broadcast = try makeElementwiseBroadcastPlan(a_v.shape, b_v.shape, out_shape),
            } });
        },

        .Unary => |u| {
            const out = try ctx.ensureValueTensor(out_idx, out_dt, out_shape);
            try appendStepChecked(allocator, mgr, steps, .{ .Unary = .{ .op = u.op, .out = out, .a = try input(ctx, node, 0) } });
        },

        .Softmax => |sm| {
            // Scalar floats only; the executors accumulate max/sum in f32 for both.
            if (out_dt != .f32 and out_dt != .f16) return CompileError.InvalidArgument;
            _ = try normalizeAxis(sm.axis, out_shape.len);
            const out = try ctx.ensureValueTensor(out_idx, out_dt, out_shape);
            try appendStepChecked(allocator, mgr, steps, .{ .Softmax = .{ .out = out, .a = try input(ctx, node, 0), .axis = sm.axis } });
        },

        .Conv1D => |cv| {
            if (out_shape.len < 2) return CompileError.InvalidArgument;
            const out = try ctx.ensureValueTensor(out_idx, out_dt, out_shape);
            try appendStepChecked(allocator, mgr, steps, .{ .Conv1D = .{
                .out = out,
                .x = try input(ctx, node, 0),
                .w = try input(ctx, node, 1),
                .bias = if (node.inputs.len == 3) try input(ctx, node, 2) else null,
                .stride = cv.stride,
                .dilation = cv.dilation,
                .pad_left = cv.pad_left,
                .pad_right = cv.pad_right,
                .pad_mode = cv.pad_mode,
                .groups = cv.groups,
            } });
        },

        .MaxPool2D => |opts| {
            const out = try ctx.ensureValueTensor(out_idx, out_dt, out_shape);
            try appendStepChecked(allocator, mgr, steps, .{ .MaxPool2D = .{ .x = try input(ctx, node, 0), .out = out, .opts = opts } });
        },

        .Conv2D => |cv| {
            if (out_shape.len < 3) return CompileError.InvalidArgument;
            const out = try ctx.ensureValueTensor(out_idx, out_dt, out_shape);
            try appendStepChecked(allocator, mgr, steps, .{ .Conv2D = .{
                .out = out,
                .x = try input(ctx, node, 0),
                .w = try input(ctx, node, 1),
                .bias = if (node.inputs.len == 3) try input(ctx, node, 2) else null,
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
            } });
        },

        .LayerNorm => |ln| {
            if (out_dt != .f32 and out_dt != .f16) return CompileError.InvalidArgument;
            if (ln.normalized_shape.len == 0 or out_shape.len < ln.normalized_shape.len) return CompileError.InvalidArgument;
            const out = try ctx.ensureValueTensor(out_idx, out_dt, out_shape);
            try appendStepChecked(allocator, mgr, steps, .{ .LayerNorm = .{ .out = out, .x = try input(ctx, node, 0), .gamma = try input(ctx, node, 1), .beta = try input(ctx, node, 2), .eps = ln.eps } });
        },

        .RMSNorm => |rn| {
            if (out_dt != .f32 and out_dt != .f16) return CompileError.InvalidArgument;
            if (rn.normalized_shape.len == 0 or out_shape.len < rn.normalized_shape.len) return CompileError.InvalidArgument;
            const out = try ctx.ensureValueTensor(out_idx, out_dt, out_shape);
            try appendStepChecked(allocator, mgr, steps, .{ .RMSNorm = .{ .out = out, .x = try input(ctx, node, 0), .gamma = try input(ctx, node, 1), .beta = try input(ctx, node, 2), .eps = rn.eps } });
        },

        .RelPosMHA => |attn| {
            if (out_shape.len != 4 or out_dt != .f32) return CompileError.InvalidArgument;
            const out = try ctx.ensureValueTensor(out_idx, out_dt, out_shape);
            try appendStepChecked(allocator, mgr, steps, .{ .RelPosMHA = .{
                .out = out,
                .q = try input(ctx, node, 0),
                .k = try input(ctx, node, 1),
                .v = try input(ctx, node, 2),
                .pos_emb = try input(ctx, node, 3),
                .pos_bias_u = try input(ctx, node, 4),
                .pos_bias_v = try input(ctx, node, 5),
                .mask = if (attn.has_mask) try input(ctx, node, 6) else null,
                .scale = attn.scale,
                .window = attn.window,
                .relative_zero_index = attn.relative_zero_index,
                .attn_logits_soft_cap = attn.attn_logits_soft_cap,
            } });
        },

        .Attention => |attn| {
            var control: usize = 3;
            const pos_i: ?usize = if (attn.has_query_positions) blk: {
                defer control += 1;
                break :blk control;
            } else null;
            const lengths_i: ?usize = if (attn.has_kv_lengths) control else null;

            if (out_shape.len != 4 or out_dt != .f32) return CompileError.InvalidArgument;
            for (node.inputs[0..3]) |id| {
                const dt = values[@intCast(id)].dtype.?;
                if (dt != .f16 and dt != .f32) return CompileError.InvalidArgument;
            }
            inline for (.{ pos_i, lengths_i }) |maybe| if (maybe) |i| {
                if (values[@intCast(node.inputs[i])].dtype.? != .i32) return CompileError.InvalidArgument;
            };

            const out = try ctx.ensureValueTensor(out_idx, out_dt, out_shape);
            try appendStepChecked(allocator, mgr, steps, .{ .Attention = .{
                .out = out,
                .q = try input(ctx, node, 0),
                .k = try input(ctx, node, 1),
                .v = try input(ctx, node, 2),
                .query_positions = if (pos_i) |i| try input(ctx, node, i) else null,
                .kv_lengths = if (lengths_i) |i| try input(ctx, node, i) else null,
                .scale = attn.scale,
                .window = attn.window,
                .attn_logits_soft_cap = attn.attn_logits_soft_cap,
            } });
        },

        .ArgMax => |am| {
            const rank: usize = values[@intCast(node.inputs[0])].shape.len;
            const axis: usize = try normalizeAxis(am.axis, rank);
            if (axis != rank - 1) return CompileError.InvalidArgument; // v1: last axis
            if (out_dt != .i32) return CompileError.InvalidArgument;
            const out = try ctx.ensureValueTensor(out_idx, out_dt, out_shape);
            try appendStepChecked(allocator, mgr, steps, .{ .ArgMax = .{ .out = out, .a = try input(ctx, node, 0), .axis = axis } });
        },

        .TopK => |tk| {
            const rank: usize = values[@intCast(node.inputs[0])].shape.len;
            const axis: usize = try normalizeAxis(tk.axis, rank);
            // Transpose to top-k a different axis; the kernels walk contiguous rows.
            if (axis != rank - 1) return CompileError.InvalidArgument;
            if (node.extra_outputs.len != 1) return CompileError.InvalidArgument;
            const values_tid = try ctx.ensureValueTensor(out_idx, out_dt, out_shape);
            const indices_tid = try ctx.ensureValueTensor(@intCast(node.extra_outputs[0]), .i32, out_shape);
            try appendStepChecked(allocator, mgr, steps, .{ .TopK = .{
                .values = values_tid,
                .indices = indices_tid,
                .a = try input(ctx, node, 0),
                .k = tk.k,
                .axis = axis,
                .largest = tk.largest,
            } });
        },

        .ScatterRow => {
            const buf = try input(ctx, node, 0);
            // In-place: output aliases buf storage.
            ctx.value_tensor[out_idx] = buf;
            ctx.value_has_tensor[out_idx] = true;
            try appendStepChecked(allocator, mgr, steps, .{ .ScatterRow = .{ .buf = buf, .idx = try input(ctx, node, 1), .src = try input(ctx, node, 2) } });
        },

        .Reduce => |rr| {
            const a = try input(ctx, node, 0);
            const out = try ctx.ensureValueTensor(out_idx, out_dt, out_shape);
            if (rr.axis) |axis_raw| {
                const axis: usize = try normalizeAxis(axis_raw, values[@intCast(node.inputs[0])].shape.len);
                try appendStepChecked(allocator, mgr, steps, .{ .ReduceAxis = .{ .op = rr.op, .out = out, .a = a, .axis = axis } });
            } else {
                try appendStepChecked(allocator, mgr, steps, .{ .ReduceAll = .{ .op = rr.op, .out = out, .a = a } });
            }
        },

        .Concat => |cc| {
            const axis: usize = try normalizeAxis(cc.axis, out_shape.len);
            var in_ids: [16]TensorId = @splat(0);
            if (node.inputs.len > in_ids.len) return CompileError.InvalidArgument;
            for (node.inputs, 0..) |id, i| {
                if (values[@intCast(id)].dtype.?.info().is_quantized) return CompileError.InvalidArgument;
                in_ids[i] = try input(ctx, node, i);
            }
            const out = try ctx.ensureValueTensor(out_idx, out_dt, out_shape);
            try appendStepChecked(allocator, mgr, steps, .{ .ConcatScalar = .{ .out = out, .axis = axis, .input_count = @intCast(node.inputs.len), .inputs = in_ids } });
        },

        .RFFT => {
            const out = try ctx.ensureValueTensor(out_idx, out_dt, out_shape);
            try appendStepChecked(allocator, mgr, steps, .{ .RFFT = .{ .out = out, .x = try input(ctx, node, 0), .n_fft = out_shape[out_shape.len - 1] - 2 } });
        },

        .STFT => |st| {
            const out = try ctx.ensureValueTensor(out_idx, out_dt, out_shape);
            try appendStepChecked(allocator, mgr, steps, .{ .STFT = .{
                .out = out,
                .signal = try input(ctx, node, 0),
                .window = try input(ctx, node, 1),
                .n_fft = st.n_fft,
                .hop_length = st.hop_length,
                .center = st.center,
                .num_frames = out_shape[1],
            } });
        },

        .LSTMCell => |lc| {
            const out = try ctx.ensureValueTensor(out_idx, out_dt, out_shape);
            try appendStepChecked(allocator, mgr, steps, .{ .LSTMCellFused = .{
                .out_state = out,
                .x = try input(ctx, node, 0),
                .h_prev = try input(ctx, node, 1),
                .c_prev = try input(ctx, node, 2),
                .w_ih = try input(ctx, node, 3),
                .w_hh = try input(ctx, node, 4),
                .b_ih = if (lc.has_bias) try input(ctx, node, 5) else null,
                .b_hh = if (lc.has_bias) try input(ctx, node, 6) else null,
            } });
        },

        .Dim, .Iota => {
            // Specialization-time constants, written once here rather than computed
            // by a runtime kernel.
            const input_v = values[@intCast(node.inputs[0])];
            const out = try ctx.ensureValueTensor(out_idx, out_dt, out_shape);
            const count = try elemCount(out_shape);
            const ints = allocator.alloc(i32, count) catch return CompileError.OutOfMemory;
            defer allocator.free(ints);
            switch (node.op) {
                .Dim => |dd| {
                    const axis = try normalizeAxis(dd.axis, input_v.shape.len);
                    ints[0] = std.math.cast(i32, input_v.shape[axis]) orelse return CompileError.InvalidArgument;
                },
                .Iota => |io| {
                    const axis = try normalizeAxis(io.axis, input_v.shape.len);
                    var stride: usize = 1;
                    for (input_v.shape[axis + 1 ..]) |d| {
                        stride = std.math.mul(usize, stride, d) catch return CompileError.InvalidArgument;
                    }
                    for (ints, 0..) |*v, linear| {
                        const coordinate = (linear / stride) % input_v.shape[axis];
                        v.* = std.math.cast(i32, coordinate) orelse return CompileError.InvalidArgument;
                    }
                },
                else => unreachable,
            }
            try mgr.writeFromPackedScalar(out, std.mem.sliceAsBytes(ints));
        },

        .Gather => |gg| {
            const data_v = values[@intCast(node.inputs[0])];
            const indices_v = values[@intCast(node.inputs[1])];
            const axis = try normalizeAxis(gg.axis, data_v.shape.len);
            if (indices_v.dtype.? != .i32) return CompileError.InvalidArgument;

            // A specialized step's preconditions SELECT it; they must not reject,
            // or a shape inference accepted would have nowhere to run.
            const quantized = data_v.dtype.?.info().is_quantized;
            const embedding_shaped = axis == 0 and gg.batch_dims == 0 and
                data_v.shape.len == 2 and indices_v.shape.len == 2 and out_shape.len == 3;
            const batched_shaped = axis == 1 and gg.batch_dims == 1 and
                data_v.shape.len == 3 and indices_v.shape.len == 2 and out_shape.len == 3;

            const data = try input(ctx, node, 0);
            const indices = try input(ctx, node, 1);
            if (embedding_shaped) {
                switch (data_v.dtype.?) {
                    .f16, .f32 => {},
                    // A q8 table is looked up by rows, so it must be blocked along them.
                    .q8_0 => if ((try mgr.getConst(data)).quant_axis != 1) return CompileError.InvalidArgument,
                    else => return CompileError.InvalidArgument,
                }
                const out = try ctx.ensureValueTensor(out_idx, out_dt, out_shape);
                try appendStepChecked(allocator, mgr, steps, .{ .GatherRows = .{ .out = out, .table = data, .indices = indices } });
            } else if (batched_shaped and !quantized) {
                const out = try ctx.ensureValueTensor(out_idx, out_dt, out_shape);
                try appendStepChecked(allocator, mgr, steps, .{ .Gather = .{ .out = out, .data = data, .indices = indices, .axis = 1, .batch_dims = 1 } });
            } else {
                // Everything the two specialized steps decline: any remaining axis /
                // batch_dims / rank. Only the embedding step reads a quantized table,
                // which is why inference confines q8_0 to it.
                if (quantized) return CompileError.InvalidArgument;
                const out = try ctx.ensureValueTensor(out_idx, out_dt, out_shape);
                try appendStepChecked(allocator, mgr, steps, .{ .GatherND = .{
                    .out = out,
                    .data = data,
                    .indices = indices,
                    .axis = @intCast(axis),
                    .batch_dims = @intCast(gg.batch_dims),
                } });
            }
        },

        .RoPE1D => |rp| {
            const x_v = values[@intCast(node.inputs[0])];
            const pos_v = values[@intCast(node.inputs[1])];
            if (x_v.shape.len != 4 or pos_v.shape.len != 2) return CompileError.InvalidArgument;
            if (!(x_v.dtype.? == .f16 or x_v.dtype.? == .f32)) return CompileError.InvalidArgument;
            if (pos_v.dtype.? != .i32) return CompileError.InvalidArgument;
            const out = try ctx.ensureValueTensor(out_idx, out_dt, out_shape);
            try appendStepChecked(allocator, mgr, steps, .{ .RoPE1D = .{
                .out = out,
                .x = try input(ctx, node, 0),
                .positions = try input(ctx, node, 1),
                .base_frequency = rp.base_frequency,
                .scale_factor = rp.scale_factor,
                .rope_proportion = rp.rope_proportion,
            } });
        },

        .SequenceAppend => {
            const cache = try input(ctx, node, 0);
            // In-place semantics: output aliases cache storage.
            ctx.value_tensor[out_idx] = cache;
            ctx.value_has_tensor[out_idx] = true;
            try appendStepChecked(allocator, mgr, steps, .{ .SequenceAppend = .{ .cache = cache, .new_kv = try input(ctx, node, 1), .end_index = try input(ctx, node, 2) } });
        },

        .If => |iff| {
            const then_block = try lowerRegionBlock(allocator, graph, graph.regions.items[@intCast(iff.then_region)], mgr, ctx, blocks);
            const else_block = try lowerRegionBlock(allocator, graph, graph.regions.items[@intCast(iff.else_region)], mgr, ctx, blocks);
            const out = try ctx.ensureValueTensor(out_idx, out_dt, out_shape);

            var outputs_arr: [executable.MAX_CONTROL_OUTPUTS]TensorId = @splat(0);
            var then_arr: [executable.MAX_CONTROL_OUTPUTS]TensorId = @splat(0);
            var else_arr: [executable.MAX_CONTROL_OUTPUTS]TensorId = @splat(0);
            outputs_arr[0] = out;
            then_arr[0] = try input(ctx, node, 1);
            else_arr[0] = try input(ctx, node, 2);
            try appendStepChecked(allocator, mgr, steps, .{ .If = .{
                .cond = try input(ctx, node, 0),
                .then_block = then_block,
                .else_block = else_block,
                .output_count = 1,
                .outputs = outputs_arr,
                .then_outputs = then_arr,
                .else_outputs = else_arr,
            } });
        },

        .Loop => |lp| {
            const body_region: graph_mod.Region = graph.regions.items[@intCast(lp.body_region)];
            const n: usize = node.inputs.len;
            if (n == 0 or n > executable.MAX_LOOP_CARRIED) return CompileError.InvalidArgument;
            if (node.extra_outputs.len + 1 != n or body_region.outputs.len != n) return CompileError.InvalidArgument;

            var carried_arr: [executable.MAX_LOOP_CARRIED]TensorId = @splat(0);
            for (0..n) |i| carried_arr[i] = try input(ctx, node, i);

            const body_block = try lowerRegionBlock(allocator, graph, body_region, mgr, ctx, blocks);

            var body_arr: [executable.MAX_LOOP_CARRIED]TensorId = @splat(0);
            for (0..n) |i| {
                body_arr[i] = try ensureAnyTensor(ctx, @intCast(body_region.outputs[i]));
                // After the loop the carried tensors hold the final state; map
                // each loop output (primary + extras) onto its carry buffer.
                const out_value: usize = if (i == 0) out_idx else @intCast(node.extra_outputs[i - 1]);
                ctx.value_tensor[out_value] = carried_arr[i];
                ctx.value_has_tensor[out_value] = true;
            }

            try appendStepChecked(allocator, mgr, steps, .{ .Loop = .{
                .trip_count = null,
                .static_max_trip_count = lp.static_max_trip_count,
                .cond = if (lp.cond_carry) |ci| carried_arr[ci] else null,
                .check_before = lp.check_before,
                .body_block = body_block,
                .carried_count = @intCast(n),
                .carried = carried_arr,
                .body_carried_outputs = body_arr,
            } });
        },

        .Cast => |ct| {
            const out = try ctx.ensureValueTensor(out_idx, out_dt, out_shape);
            try appendStepChecked(allocator, mgr, steps, .{ .Cast = .{ .out = out, .x = try input(ctx, node, 0), .to_dtype = ct.to_dtype } });
        },

        .MatMulNT => |mm| {
            if (values[@intCast(node.inputs[1])].shape.len != 2) return CompileError.InvalidArgument;
            const b = try input(ctx, node, 1);
            const b_t = try mgr.getConst(b);
            // A quantized `[N, K]` B holds each row as one run of blocks.
            if (b_t.dtype.info().is_quantized and b_t.quant_axis != 1) return CompileError.InvalidArgument;
            const c = try ctx.ensureValueTensor(out_idx, out_dt, out_shape);
            try appendStepChecked(allocator, mgr, steps, .{ .MatMulNT = .{ .c = c, .a = try input(ctx, node, 0), .b = b, .alpha = mm.alpha, .beta = mm.beta } });
        },

        .Copy => {
            const out = try ctx.ensureValueTensor(out_idx, out_dt, out_shape);
            try appendStepChecked(allocator, mgr, steps, .{ .Copy = .{ .dst = out, .src = try input(ctx, node, 0) } });
        },

        .ViewReshape, .ViewSqueeze, .ViewUnsqueeze, .ViewTranspose2D, .ViewSliceND => {
            // A view cannot reinterpret a block-quantized layout. Reject before
            // allocating: creating the output first surfaces this as an unrelated
            // quant-axis alignment error from storage, far from the real cause.
            if (values[@intCast(node.inputs[0])].dtype.?.info().is_quantized) return CompileError.InvalidArgument;
            const out = try ctx.ensureValueTensor(out_idx, out_dt, out_shape);
            const src = try input(ctx, node, 0);
            const step: Step = switch (node.op) {
                .ViewReshape, .ViewSqueeze, .ViewUnsqueeze => .{ .ReshapeScalar = .{ .dst = out, .src = src } },
                .ViewTranspose2D => .{ .Transpose2DScalar = .{ .dst = out, .src = src } },
                .ViewSliceND => |sl| blk: {
                    if (sl.starts.len == 0 or sl.starts.len > MAX_RANK) return CompileError.InvalidArgument;
                    var starts: [MAX_RANK]usize = @splat(0);
                    @memcpy(starts[0..sl.starts.len], sl.starts);
                    break :blk .{ .SliceNDScalar = .{ .dst = out, .src = src, .rank = @intCast(sl.starts.len), .starts = starts } };
                },
                else => unreachable,
            };
            try appendStepChecked(allocator, mgr, steps, step);
        },
    }
}
