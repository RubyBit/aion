// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const env = @import("../env.zig");
const types = @import("../backend/types.zig");
const storage = @import("../storage/storage.zig");
const executable = @import("../runtime/executable.zig");

const graph_mod = @import("graph.zig");
const infer_mod = @import("infer.zig");
const plan_mod = @import("plan.zig");
const optimize_mod = @import("optimize.zig");
const manager_mod = @import("../storage/manager.zig");

const backend_utils = @import("../backend/utils.zig");

pub const StorageError = storage.StorageError;
pub const TiledTensor = storage.TiledTensor;

pub const StorageManager = manager_mod.StorageManager;
pub const TensorId = manager_mod.TensorId;

pub const Step = executable.Step;
pub const Program = executable.ExecutableProgram;

const MAX_RANK: usize = 8;

pub const CompileError = error{ InvalidArgument, OutOfMemory } || graph_mod.GraphError || infer_mod.InferError || StorageError;

fn traceEnabled() bool {
    // Keep tracing opt-in to avoid log spam in normal runs/tests.
    return env.flagEnabled("AION_TRACE");
}

fn compileRequire(cond: bool) CompileError!void {
    if (!cond) return CompileError.InvalidArgument;
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

fn requireSameTileShape(a: *const TiledTensor, b: *const TiledTensor) CompileError!void {
    if (a.tile_shape.len != b.tile_shape.len) return CompileError.InvalidArgument;
    var i: usize = 0;
    while (i < a.tile_shape.len) : (i += 1) {
        try compileRequire(a.tile_shape[i] == b.tile_shape[i]);
    }
}

fn requireSameTileCounts(a: *const TiledTensor, b: *const TiledTensor) CompileError!void {
    if (a.tile_counts.len != b.tile_counts.len) return CompileError.InvalidArgument;
    var i: usize = 0;
    while (i < a.tile_counts.len) : (i += 1) {
        try compileRequire(a.tile_counts[i] == b.tile_counts[i]);
    }
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

// CPU-kernel scratch limits used by the validation guard below. These mirror
// the tiling caps in `TilePolicy` (softmax_row_cap / attn_*), but apply to the
// *kernel's* bounded (stack) scratch rather than the tiling heuristic. A GPU
// backend lowers through its own path with its own (much larger) limits; these
// are the CPU contract. Kept as named constants — not magic numbers — so the
// coupling to `plan.TilePolicy` defaults is explicit and auditable.
const CPU_SOFTMAX_ROW_SCRATCH_MAX: usize = 256; // == plan.TilePolicy.softmax_row_cap
// Attention tile caps are policy-driven (see `TilePolicy.attn_max_q_rows` /
// `attn_k_tile_cap` / `attn_v_tile_cap`): CPU defaults reproduce the historical
// 256/128/64 stack-scratch limits, GPU targets relax them to the macro tile cap
// (the CPU exec still re-checks its own limits at run time).

fn validateStep(mgr: *StorageManager, policy: plan_mod.TilePolicy, step: Step) CompileError!void {
    switch (step) {
        .MatMulTiled => |s| {
            const c: *const TiledTensor = mgr.getConst(s.c) catch return CompileError.InvalidArgument;
            const a: *const TiledTensor = mgr.getConst(s.a) catch return CompileError.InvalidArgument;
            const b: *const TiledTensor = mgr.getConst(s.b) catch return CompileError.InvalidArgument;

            try compileRequire(c.rank >= 2);
            try compileRequire(a.rank == c.rank and b.rank == c.rank);
            try compileRequire(!a.dtype.info().is_quantized);
            try compileRequire(!c.dtype.info().is_quantized);
            try compileRequire(isScalarSupported(a.dtype));
            try compileRequire(isScalarSupported(c.dtype));

            const rank: usize = @as(usize, c.rank);
            var d: usize = 0;
            while (d + 2 < rank) : (d += 1) {
                const ad: usize = a.shape[d];
                const bd: usize = b.shape[d];
                const cd: usize = c.shape[d];
                if (ad != bd and ad != 1 and bd != 1) return CompileError.InvalidArgument;
                try compileRequire(cd == @max(ad, bd));

                // Batch dims must be tiled as size-1 so each tile is a single batch slice.
                try compileRequire(c.tile_shape[d] == 1);

                if (ad == 1) {
                    try compileRequire(a.tile_shape[d] == 1);
                    try compileRequire(a.tile_counts[d] == 1);
                } else {
                    try compileRequire(a.tile_shape[d] == c.tile_shape[d]);
                    try compileRequire(a.tile_counts[d] == c.tile_counts[d]);
                }

                if (bd == 1) {
                    try compileRequire(b.tile_shape[d] == 1);
                    try compileRequire(b.tile_counts[d] == 1);
                } else {
                    try compileRequire(b.tile_shape[d] == c.tile_shape[d]);
                    try compileRequire(b.tile_counts[d] == c.tile_counts[d]);
                }
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
            try compileRequire(a.shape[rank - 1] == b.shape[rank - 2]);
            try compileRequire(c.shape[rank - 2] == a.shape[rank - 2]);
            try compileRequire(c.shape[rank - 1] == b.shape[rank - 1]);

            // Canonical tiling geometry for last two dims.
            try compileRequire(a.tile_shape[rank - 2] == c.tile_shape[rank - 2]);
            try compileRequire(b.tile_shape[rank - 1] == c.tile_shape[rank - 1]);
            try compileRequire(a.tile_shape[rank - 1] == b.tile_shape[rank - 2]);

            try compileRequire(a.tile_counts[rank - 2] == c.tile_counts[rank - 2]);
            try compileRequire(b.tile_counts[rank - 1] == c.tile_counts[rank - 1]);
            try compileRequire(a.tile_counts[rank - 1] == b.tile_counts[rank - 2]);

            if (b.dtype.info().is_quantized) {
                const be: usize = b.dtype.info().block_elems;
                try compileRequire(a.shape[rank - 1] % be == 0);
                try compileRequire(a.tile_shape[rank - 1] % be == 0);
                const rem: usize = a.shape[rank - 1] % a.tile_shape[rank - 1];
                if (rem != 0) try compileRequire(rem % be == 0);
            }
        },

        .ElemwiseBinaryTiled => |s| {
            const out: *const TiledTensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const a: *const TiledTensor = mgr.getConst(s.a) catch return CompileError.InvalidArgument;
            const b: *const TiledTensor = mgr.getConst(s.b) catch return CompileError.InvalidArgument;

            try compileRequire(isScalarSupported(out.dtype));
            try compileRequire(out.dtype == a.dtype and out.dtype == b.dtype);
            try compileRequire(out.rank == a.rank and out.rank == b.rank);
            try requireSameShape(out.shape, a.shape);
            try requireSameShape(out.shape, b.shape);
            try requireSameTileShape(out, a);
            try requireSameTileShape(out, b);
            try requireSameTileCounts(out, a);
            try requireSameTileCounts(out, b);
            _ = s.op;
        },

        .BroadcastLastDimBinaryTiled => |s| {
            const out: *const TiledTensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const a: *const TiledTensor = mgr.getConst(s.a) catch return CompileError.InvalidArgument;
            const b: *const TiledTensor = mgr.getConst(s.b) catch return CompileError.InvalidArgument;

            try compileRequire(out.rank >= 2 and a.rank == out.rank and b.rank == 1);
            try compileRequire(isScalarSupported(out.dtype));
            try compileRequire(out.dtype == a.dtype and out.dtype == b.dtype);
            try requireSameShape(out.shape, a.shape);
            try requireSameTileShape(out, a);
            try requireSameTileCounts(out, a);

            const last: usize = @as(usize, out.rank) - 1;
            try compileRequire(b.shape[0] == out.shape[last]);
            // b is tiled along the last dimension.
            try compileRequire(b.tile_shape[0] == out.tile_shape[last]);
            try compileRequire(b.tile_counts[0] == out.tile_counts[last]);
            _ = s.op;
        },

        .UnaryTiled => |s| {
            const out: *const TiledTensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const a: *const TiledTensor = mgr.getConst(s.a) catch return CompileError.InvalidArgument;
            try compileRequire(isScalarSupported(out.dtype));
            try compileRequire(out.dtype == a.dtype);
            try compileRequire(out.rank == a.rank);
            try requireSameShape(out.shape, a.shape);
            try requireSameTileShape(out, a);
            try requireSameTileCounts(out, a);
            _ = s.op;
        },

        .CastTiled => |s| {
            const out: *const TiledTensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const x: *const TiledTensor = mgr.getConst(s.x) catch return CompileError.InvalidArgument;
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
            try requireSameTileShape(out, x);
            try requireSameTileCounts(out, x);
        },

        .If => |s| {
            const cond: *const TiledTensor = mgr.getConst(s.cond) catch return CompileError.InvalidArgument;
            try compileRequire(cond.dtype == .i32 and cond.rank == 1 and cond.shape[0] == 1);
            const count: usize = @intCast(s.output_count);
            try compileRequire(count <= executable.MAX_CONTROL_OUTPUTS);
            var i: usize = 0;
            while (i < count) : (i += 1) {
                const out: *const TiledTensor = mgr.getConst(s.outputs[i]) catch return CompileError.InvalidArgument;
                const then_out: *const TiledTensor = mgr.getConst(s.then_outputs[i]) catch return CompileError.InvalidArgument;
                const else_out: *const TiledTensor = mgr.getConst(s.else_outputs[i]) catch return CompileError.InvalidArgument;
                try compileRequire(out.dtype == then_out.dtype and out.dtype == else_out.dtype);
                try compileRequire(out.rank == then_out.rank and out.rank == else_out.rank);
                try requireSameShape(out.shape, then_out.shape);
                try requireSameShape(out.shape, else_out.shape);
                try requireSameTileShape(out, then_out);
                try requireSameTileShape(out, else_out);
                try requireSameTileCounts(out, then_out);
                try requireSameTileCounts(out, else_out);
            }
            _ = s.then_block;
            _ = s.else_block;
        },

        .Loop => |s| {
            if (s.trip_count) |trip_count| {
                const trip: *const TiledTensor = mgr.getConst(trip_count) catch return CompileError.InvalidArgument;
                try compileRequire(trip.dtype == .i32 and trip.rank == 1 and trip.shape[0] == 1);
            }
            if (s.cond) |cond_id| {
                const cond: *const TiledTensor = mgr.getConst(cond_id) catch return CompileError.InvalidArgument;
                try compileRequire(cond.dtype == .i32 and cond.rank == 1 and cond.shape[0] == 1);
            }
            try compileRequire(s.static_max_trip_count > 0);
            const count: usize = @intCast(s.carried_count);
            try compileRequire(count <= executable.MAX_LOOP_CARRIED);
            var i: usize = 0;
            while (i < count) : (i += 1) {
                const carried: *const TiledTensor = mgr.getConst(s.carried[i]) catch return CompileError.InvalidArgument;
                const next: *const TiledTensor = mgr.getConst(s.body_carried_outputs[i]) catch return CompileError.InvalidArgument;
                try compileRequire(carried.dtype == next.dtype);
                try compileRequire(carried.rank == next.rank);
                try requireSameShape(carried.shape, next.shape);
                try requireSameTileShape(carried, next);
                try requireSameTileCounts(carried, next);
            }
            _ = s.check_before;
            _ = s.body_block;
        },

        .MatMulNTTiled => |s| {
            const c: *const TiledTensor = mgr.getConst(s.c) catch return CompileError.InvalidArgument;
            const a: *const TiledTensor = mgr.getConst(s.a) catch return CompileError.InvalidArgument;
            const b: *const TiledTensor = mgr.getConst(s.b) catch return CompileError.InvalidArgument;
            try compileRequire(a.dtype == .f32 and c.dtype == .f32);
            // The CPU/GPU executors implement q8_0 (per-row blocks) and plain f32 B.
            try compileRequire(b.dtype == .q8_0 or b.dtype == .f32);
            try compileRequire(b.rank == 2);
            if (b.dtype.info().is_quantized) try compileRequire(b.quant_axis == 1);
            try compileRequire(b.tile_shape[1] == b.shape[1]);
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

        .SoftmaxTiled => |s| {
            const out: *const TiledTensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const a: *const TiledTensor = mgr.getConst(s.a) catch return CompileError.InvalidArgument;
            // v0: f32 only (fast + stable).
            try compileRequire(out.dtype == .f32 and a.dtype == .f32);
            try compileRequire(out.rank == a.rank);
            const rank: usize = @as(usize, out.rank);
            try compileRequire(rank >= 1 and rank <= MAX_RANK);
            const axis: usize = try normalizeAxis(s.axis, rank);

            try requireSameShape(out.shape, a.shape);
            try requireSameTileShape(out, a);
            try requireSameTileCounts(out, a);

            // Per-row scratch in exec uses stack arrays sized by tile_shape[0].
            if (rank == 1) {
                try compileRequire(out.tile_shape[0] <= CPU_SOFTMAX_ROW_SCRATCH_MAX);
            } else {
                var row_elems: usize = 1;
                var d: usize = 0;
                while (d < rank) : (d += 1) {
                    if (d == axis) continue;
                    row_elems = std.math.mul(usize, row_elems, out.tile_shape[d]) catch return CompileError.InvalidArgument;
                }
                try compileRequire(row_elems <= CPU_SOFTMAX_ROW_SCRATCH_MAX);
            }
        },

        .Conv1DTiled => |s| {
            const out: *const TiledTensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const x: *const TiledTensor = mgr.getConst(s.x) catch return CompileError.InvalidArgument;
            const w: *const TiledTensor = mgr.getConst(s.w) catch return CompileError.InvalidArgument;

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
                try compileRequire(out.tile_shape[d] == x.tile_shape[d]);
                try compileRequire(out.tile_counts[d] == x.tile_counts[d]);
            }
            try compileRequire(out.tile_shape[rank - 2] == x.tile_shape[rank - 2]);

            if (s.bias) |bias_id| {
                const b: *const TiledTensor = mgr.getConst(bias_id) catch return CompileError.InvalidArgument;
                try compileRequire(b.dtype == .f32);
                try compileRequire(b.rank == 1 and b.shape[0] == c_out);
            }
        },

        .Conv2DTiled => |s| {
            const out: *const TiledTensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const x: *const TiledTensor = mgr.getConst(s.x) catch return CompileError.InvalidArgument;
            const w: *const TiledTensor = mgr.getConst(s.w) catch return CompileError.InvalidArgument;

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
                try compileRequire(out.tile_shape[d] == x.tile_shape[d]);
                try compileRequire(out.tile_counts[d] == x.tile_counts[d]);
            }
            try compileRequire(out.tile_shape[rank - 3] == x.tile_shape[rank - 3]);
            try compileRequire(out.tile_shape[rank - 2] == x.tile_shape[rank - 2]);

            if (s.bias) |bias_id| {
                const b: *const TiledTensor = mgr.getConst(bias_id) catch return CompileError.InvalidArgument;
                try compileRequire(b.dtype == .f32);
                try compileRequire(b.rank == 1 and b.shape[0] == c_out);
            }
        },

        .LayerNormTiled => |s| {
            const out: *const TiledTensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const x: *const TiledTensor = mgr.getConst(s.x) catch return CompileError.InvalidArgument;
            const gamma: *const TiledTensor = mgr.getConst(s.gamma) catch return CompileError.InvalidArgument;
            const beta: *const TiledTensor = mgr.getConst(s.beta) catch return CompileError.InvalidArgument;

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

            // Tiling for out/x must match.
            try requireSameTileShape(out, x);
            try requireSameTileCounts(out, x);

            // gamma/beta tiled along normalized dims only.
            d = 0;
            while (d < norm_rank) : (d += 1) {
                try compileRequire(gamma.tile_shape[d] == out.tile_shape[rank - norm_rank + d]);
                try compileRequire(beta.tile_shape[d] == out.tile_shape[rank - norm_rank + d]);
                try compileRequire(gamma.tile_counts[d] == out.tile_counts[rank - norm_rank + d]);
                try compileRequire(beta.tile_counts[d] == out.tile_counts[rank - norm_rank + d]);
            }

            try compileRequire(s.eps > 0.0);
        },

        .RMSNormTiled => |s| {
            const out: *const TiledTensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const x: *const TiledTensor = mgr.getConst(s.x) catch return CompileError.InvalidArgument;
            const gamma: *const TiledTensor = mgr.getConst(s.gamma) catch return CompileError.InvalidArgument;
            const beta: *const TiledTensor = mgr.getConst(s.beta) catch return CompileError.InvalidArgument;

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

            try requireSameTileShape(out, x);
            try requireSameTileCounts(out, x);

            d = 0;
            while (d < norm_rank) : (d += 1) {
                try compileRequire(gamma.tile_shape[d] == out.tile_shape[rank - norm_rank + d]);
                try compileRequire(beta.tile_shape[d] == out.tile_shape[rank - norm_rank + d]);
                try compileRequire(gamma.tile_counts[d] == out.tile_counts[rank - norm_rank + d]);
                try compileRequire(beta.tile_counts[d] == out.tile_counts[rank - norm_rank + d]);
            }
            try compileRequire(s.eps > 0.0);
        },

        .AttentionTiled => |s| {
            const out: *const TiledTensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const q: *const TiledTensor = mgr.getConst(s.q) catch return CompileError.InvalidArgument;
            const k: *const TiledTensor = mgr.getConst(s.k) catch return CompileError.InvalidArgument;
            const v: *const TiledTensor = mgr.getConst(s.v) catch return CompileError.InvalidArgument;

            try compileRequire(out.rank >= 2 and q.rank == out.rank and k.rank == out.rank and v.rank == out.rank);
            try compileRequire(isScalarSupported(out.dtype));
            try compileRequire(out.dtype == q.dtype and out.dtype == k.dtype and out.dtype == v.dtype);

            const rank: usize = @as(usize, out.rank);
            const lead_dims: usize = rank - 2;

            // Shapes: q:[...,m,dk], k:[...,n,dk], v:[...,n,dv], out:[...,m,dv]
            var d: usize = 0;
            while (d < lead_dims) : (d += 1) {
                try compileRequire(q.shape[d] == k.shape[d] and q.shape[d] == v.shape[d]);
                try compileRequire(out.shape[d] == q.shape[d]);
            }
            try compileRequire(q.shape[rank - 1] == k.shape[rank - 1]);
            try compileRequire(k.shape[rank - 2] == v.shape[rank - 2]);
            try compileRequire(out.shape[rank - 2] == q.shape[rank - 2]);
            try compileRequire(out.shape[rank - 1] == v.shape[rank - 1]);

            // Batch/head dims must be tiled as size-1 so each tile is a single slice.
            d = 0;
            while (d < lead_dims) : (d += 1) {
                try compileRequire(out.tile_shape[d] == 1);
                try compileRequire(q.tile_shape[d] == 1);
                try compileRequire(k.tile_shape[d] == 1);
                try compileRequire(v.tile_shape[d] == 1);
                try compileRequire(out.tile_counts[d] == q.tile_counts[d]);
                try compileRequire(out.tile_counts[d] == k.tile_counts[d]);
                try compileRequire(out.tile_counts[d] == v.tile_counts[d]);
            }

            // Tiling contract for last two dims:
            // - out tile rows match q tile rows.
            // - v tile cols match out tile cols.
            // - k/v tile rows match (key blocks).
            // - q/k tile cols match (dk blocks).
            try compileRequire(out.tile_shape[rank - 2] == q.tile_shape[rank - 2]);
            try compileRequire(out.tile_counts[rank - 2] == q.tile_counts[rank - 2]);

            try compileRequire(out.tile_shape[rank - 1] == v.tile_shape[rank - 1]);
            try compileRequire(out.tile_counts[rank - 1] == v.tile_counts[rank - 1]);

            try compileRequire(k.tile_shape[rank - 2] == v.tile_shape[rank - 2]);
            try compileRequire(k.tile_counts[rank - 2] == v.tile_counts[rank - 2]);

            try compileRequire(q.tile_shape[rank - 1] == k.tile_shape[rank - 1]);
            try compileRequire(q.tile_counts[rank - 1] == k.tile_counts[rank - 1]);

            // Scratch limits (v0): keep per-tile row scratch bounded.
            try compileRequire(out.tile_shape[rank - 2] <= policy.attn_max_q_rows);
            try compileRequire(k.tile_shape[rank - 2] <= policy.attn_k_tile_cap);
            try compileRequire(out.tile_shape[rank - 1] <= policy.attn_v_tile_cap);
            try compileRequire(q.tile_shape[rank - 1] > 0);

            try compileRequire(s.scale > 0.0);
            try compileRequire(std.math.isFinite(s.scale));
            _ = s.causal;
        },

        .MultiHeadAttentionTiled => |s| {
            const out: *const TiledTensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const q: *const TiledTensor = mgr.getConst(s.q) catch return CompileError.InvalidArgument;
            const k: *const TiledTensor = mgr.getConst(s.k) catch return CompileError.InvalidArgument;
            const v: *const TiledTensor = mgr.getConst(s.v) catch return CompileError.InvalidArgument;

            try compileRequire(s.heads > 0);

            try compileRequire(out.rank >= 3 and q.rank == out.rank and k.rank == out.rank and v.rank == out.rank);
            try compileRequire(isScalarSupported(out.dtype));
            try compileRequire(out.dtype == q.dtype and out.dtype == k.dtype and out.dtype == v.dtype);

            const rank: usize = @as(usize, out.rank);
            const lead_dims: usize = rank - 3;
            const head_dim: usize = rank - 3;

            // Shapes: q:[...,h,m,dk], k:[...,h,n,dk], v:[...,h,n,dv], out:[...,h,m,dv]
            var d: usize = 0;
            while (d < lead_dims) : (d += 1) {
                try compileRequire(q.shape[d] == k.shape[d] and q.shape[d] == v.shape[d]);
                try compileRequire(out.shape[d] == q.shape[d]);
            }
            try compileRequire(q.shape[head_dim] == s.heads);
            try compileRequire(k.shape[head_dim] == s.heads and v.shape[head_dim] == s.heads);
            try compileRequire(out.shape[head_dim] == s.heads);
            try compileRequire(q.shape[rank - 1] == k.shape[rank - 1]);
            try compileRequire(k.shape[rank - 2] == v.shape[rank - 2]);
            try compileRequire(out.shape[rank - 2] == q.shape[rank - 2]);
            try compileRequire(out.shape[rank - 1] == v.shape[rank - 1]);

            // Batch/head dims must be tiled as size-1 so each tile is a single slice.
            d = 0;
            while (d < rank - 2) : (d += 1) {
                try compileRequire(out.tile_shape[d] == 1);
                try compileRequire(q.tile_shape[d] == 1);
                try compileRequire(k.tile_shape[d] == 1);
                try compileRequire(v.tile_shape[d] == 1);
                try compileRequire(out.tile_counts[d] == q.tile_counts[d]);
                try compileRequire(out.tile_counts[d] == k.tile_counts[d]);
                try compileRequire(out.tile_counts[d] == v.tile_counts[d]);
            }

            // Tiling contract for last two dims (same as attention).
            try compileRequire(out.tile_shape[rank - 2] == q.tile_shape[rank - 2]);
            try compileRequire(out.tile_counts[rank - 2] == q.tile_counts[rank - 2]);

            try compileRequire(out.tile_shape[rank - 1] == v.tile_shape[rank - 1]);
            try compileRequire(out.tile_counts[rank - 1] == v.tile_counts[rank - 1]);

            try compileRequire(k.tile_shape[rank - 2] == v.tile_shape[rank - 2]);
            try compileRequire(k.tile_counts[rank - 2] == v.tile_counts[rank - 2]);

            try compileRequire(q.tile_shape[rank - 1] == k.tile_shape[rank - 1]);
            try compileRequire(q.tile_counts[rank - 1] == k.tile_counts[rank - 1]);

            // Scratch limits (v0): keep per-tile row scratch bounded.
            try compileRequire(out.tile_shape[rank - 2] <= policy.attn_max_q_rows);
            try compileRequire(k.tile_shape[rank - 2] <= policy.attn_k_tile_cap);
            try compileRequire(out.tile_shape[rank - 1] <= policy.attn_v_tile_cap);
            try compileRequire(q.tile_shape[rank - 1] > 0);

            try compileRequire(s.scale > 0.0);
            try compileRequire(std.math.isFinite(s.scale));
            _ = s.causal;
        },

        .RelPosMHATiled => |s| {
            const out: *const TiledTensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const q: *const TiledTensor = mgr.getConst(s.q) catch return CompileError.InvalidArgument;
            const k: *const TiledTensor = mgr.getConst(s.k) catch return CompileError.InvalidArgument;
            const v: *const TiledTensor = mgr.getConst(s.v) catch return CompileError.InvalidArgument;
            const pe: *const TiledTensor = mgr.getConst(s.pos_emb) catch return CompileError.InvalidArgument;
            const bu: *const TiledTensor = mgr.getConst(s.pos_bias_u) catch return CompileError.InvalidArgument;
            const bv: *const TiledTensor = mgr.getConst(s.pos_bias_v) catch return CompileError.InvalidArgument;

            try compileRequire(s.heads > 0);
            try compileRequire(s.scale > 0.0 and std.math.isFinite(s.scale));

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

            try compileRequire(H == s.heads);
            try compileRequire(out.shape[0] == B and out.shape[1] == t_q and out.shape[2] == H and out.shape[3] == d);
            try compileRequire(k.shape[0] == B and k.shape[2] == H and k.shape[3] == d);
            try compileRequire(v.shape[0] == B and v.shape[1] == t_kv and v.shape[2] == H and v.shape[3] == d);
            try compileRequire(pe.shape[0] == H and pe.shape[2] == d);
            try compileRequire(bu.shape[0] == H and bu.shape[1] == d);
            try compileRequire(bv.shape[0] == H and bv.shape[1] == d);
            try compileRequire(t_kv > 0 and p_len == 2 * t_kv - 1);
            try compileRequire(t_q <= t_kv);

            // [T, D] dims (1 and 3) must be a single tile; [B, H] (0 and 2) size-1 tiles.
            try compileRequire(out.tile_shape[0] == 1 and out.tile_shape[2] == 1);
            try compileRequire(out.tile_shape[1] == t_q and out.tile_shape[3] == d);
            try compileRequire(q.tile_shape[1] == t_q and q.tile_shape[3] == d);
            try compileRequire(k.tile_shape[1] == t_kv and k.tile_shape[3] == d);
            try compileRequire(v.tile_shape[1] == t_kv and v.tile_shape[3] == d);
            try compileRequire(pe.tile_shape[1] == p_len and pe.tile_shape[2] == d);

            if (s.mask) |mask_id| {
                const m: *const TiledTensor = mgr.getConst(mask_id) catch return CompileError.InvalidArgument;
                try compileRequire(m.rank == 2 and m.dtype == .f32);
                try compileRequire(m.shape[0] == t_q and m.shape[1] == t_kv);
                try compileRequire(m.tile_shape[0] == t_q and m.tile_shape[1] == t_kv);
            }
        },

        .ArgMax => |s| {
            const a: *const TiledTensor = mgr.getConst(s.a) catch return CompileError.InvalidArgument;
            const out: *const TiledTensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            try compileRequire(a.dtype == .f32 and out.dtype == .i32);
            try compileRequire(@as(usize, a.rank) >= 1);
            try compileRequire(s.axis == @as(usize, a.rank) - 1);
            var d: usize = 0;
            while (d < @as(usize, a.rank)) : (d += 1) try compileRequire(a.tile_counts[d] == 1);
            d = 0;
            while (d < @as(usize, out.rank)) : (d += 1) try compileRequire(out.tile_counts[d] == 1);
        },

        .ScatterRow => |s| {
            const buf: *const TiledTensor = mgr.getConst(s.buf) catch return CompileError.InvalidArgument;
            const idx: *const TiledTensor = mgr.getConst(s.idx) catch return CompileError.InvalidArgument;
            const src: *const TiledTensor = mgr.getConst(s.src) catch return CompileError.InvalidArgument;
            try compileRequire(!buf.dtype.info().is_quantized);
            try compileRequire(idx.dtype == .i32 and src.dtype == buf.dtype);
            try compileRequire(@as(usize, buf.rank) >= 1);
            var d: usize = 0;
            while (d < @as(usize, buf.rank)) : (d += 1) try compileRequire(buf.tile_counts[d] == 1);
        },

        .MultiHeadAttentionCachedTiled => |s| {
            const out: *const TiledTensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const q: *const TiledTensor = mgr.getConst(s.q) catch return CompileError.InvalidArgument;
            const k_cache: *const TiledTensor = mgr.getConst(s.k_cache) catch return CompileError.InvalidArgument;
            const v_cache: *const TiledTensor = mgr.getConst(s.v_cache) catch return CompileError.InvalidArgument;
            const positions: *const TiledTensor = mgr.getConst(s.positions) catch return CompileError.InvalidArgument;
            const end_idx: *const TiledTensor = mgr.getConst(s.end_index) catch return CompileError.InvalidArgument;

            // v2 cached attention accepts q/k/v in {f16,f32} and accumulates in f32.
            try compileRequire(out.dtype == .f32);
            try compileRequire((q.dtype == .f16 or q.dtype == .f32));
            try compileRequire((k_cache.dtype == .f16 or k_cache.dtype == .f32));
            try compileRequire((v_cache.dtype == .f16 or v_cache.dtype == .f32));
            try compileRequire(positions.dtype == .i32 and end_idx.dtype == .i32);

            try compileRequire(out.rank == 4 and q.rank == 4 and k_cache.rank == 4 and v_cache.rank == 4);
            try compileRequire(positions.rank == 2 and end_idx.rank == 1);

            // q:[B,L_q,H_q,D_k], k_cache/v_cache:[B,H_kv,T,D_*], out:[B,L_q,H_q,D_v]
            try compileRequire(q.shape[0] == k_cache.shape[0] and q.shape[0] == v_cache.shape[0]);
            try compileRequire(q.shape[1] == out.shape[1]);
            try compileRequire(q.shape[2] == out.shape[2]);
            try compileRequire(k_cache.shape[1] == v_cache.shape[1]);
            try compileRequire(k_cache.shape[2] == v_cache.shape[2]);
            try compileRequire(q.shape[3] == k_cache.shape[3]);
            try compileRequire(out.shape[3] == v_cache.shape[3]);
            try compileRequire(out.shape[0] == q.shape[0]);

            // GQA constraint.
            try compileRequire(k_cache.shape[1] > 0 and q.shape[2] > 0);
            try compileRequire((q.shape[2] % k_cache.shape[1]) == 0);

            // positions:[B,L_q], end_index:[B]
            try compileRequire(positions.shape[0] == q.shape[0]);
            try compileRequire(positions.shape[1] == q.shape[1]);
            try compileRequire(end_idx.shape[0] == q.shape[0]);

            // Current kernel assumes full D vectors within one tile for direct row access.
            try compileRequire(q.tile_counts[3] == 1 and q.tile_shape[3] == q.shape[3]);
            try compileRequire(k_cache.tile_counts[3] == 1 and k_cache.tile_shape[3] == k_cache.shape[3]);
            try compileRequire(v_cache.tile_counts[3] == 1 and v_cache.tile_shape[3] == v_cache.shape[3]);
            try compileRequire(out.tile_counts[3] == 1 and out.tile_shape[3] == out.shape[3]);

            try compileRequire(s.scale > 0.0 and std.math.isFinite(s.scale));
            try compileRequire(std.math.isFinite(s.attn_logits_soft_cap));
            try compileRequire(s.attn_logits_soft_cap >= 0.0);
            _ = s.causal;
            _ = s.sliding_window;
        },

        .CopyTiled => |s| {
            const dst: *const TiledTensor = mgr.getConst(s.dst) catch return CompileError.InvalidArgument;
            const src: *const TiledTensor = mgr.getConst(s.src) catch return CompileError.InvalidArgument;
            try compileRequire(dst.dtype == src.dtype);
            try compileRequire(dst.rank == src.rank);
            try requireSameShape(dst.shape, src.shape);
            try requireSameTileShape(dst, src);
            try requireSameTileCounts(dst, src);
        },

        .GatherRowsTiled => |s| {
            const out: *const TiledTensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const table: *const TiledTensor = mgr.getConst(s.table) catch return CompileError.InvalidArgument;
            const indices: *const TiledTensor = mgr.getConst(s.indices) catch return CompileError.InvalidArgument;

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
                    // Tiling invariant: the full feature axis must fit in one tile so a row is
                    // a contiguous run of blocks.
                    try compileRequire(table.tile_shape[1] == table.shape[1]);
                },
                else => return CompileError.InvalidArgument,
            }

            // Indices: i32.
            try compileRequire(indices.dtype == .i32);

            // Shape contract.
            try compileRequire(out.shape[0] == indices.shape[0]);
            try compileRequire(out.shape[1] == indices.shape[1]);
            try compileRequire(out.shape[2] == table.shape[1]);

            // Tiling contract:
            // - indices: single tile (rank-2).
            // - table: feature dim D is within one tile.
            // - out: feature dim D is within one tile.
            try compileRequire(indices.tile_counts[0] == 1 and indices.tile_counts[1] == 1);
            try compileRequire(table.tile_counts[1] == 1);
            try compileRequire(out.tile_counts[2] == 1);
        },

        .RoPE1DTiled => |s| {
            const out: *const TiledTensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const x: *const TiledTensor = mgr.getConst(s.x) catch return CompileError.InvalidArgument;
            const positions: *const TiledTensor = mgr.getConst(s.positions) catch return CompileError.InvalidArgument;

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
            try requireSameTileShape(out, x);
            try requireSameTileCounts(out, x);

            try compileRequire(positions.shape[0] == out.shape[0]);
            try compileRequire(positions.shape[1] == out.shape[1]);

            // RoPE v1 contract: full head dimension in one tile.
            try compileRequire(out.tile_counts[3] == 1);
            try compileRequire(x.tile_counts[3] == 1);

            // Positions must be tiled identically to [B, L] dimensions of x/out.
            try compileRequire(positions.tile_shape[0] == out.tile_shape[0]);
            try compileRequire(positions.tile_shape[1] == out.tile_shape[1]);
            try compileRequire(positions.tile_counts[0] == out.tile_counts[0]);
            try compileRequire(positions.tile_counts[1] == out.tile_counts[1]);

            try compileRequire(s.base_frequency > 0.0 and std.math.isFinite(s.base_frequency));
            try compileRequire(s.scale_factor > 0.0 and std.math.isFinite(s.scale_factor));
            try compileRequire(std.math.isFinite(s.rope_proportion));
            try compileRequire(s.rope_proportion >= 0.0 and s.rope_proportion <= 1.0);
        },

        .SequenceAppendTiled => |s| {
            const cache: *const TiledTensor = mgr.getConst(s.cache) catch return CompileError.InvalidArgument;
            const new_kv: *const TiledTensor = mgr.getConst(s.new_kv) catch return CompileError.InvalidArgument;
            const end_idx: *const TiledTensor = mgr.getConst(s.end_index) catch return CompileError.InvalidArgument;

            try compileRequire(cache.rank == 4 and new_kv.rank == 4 and end_idx.rank == 1);

            try compileRequire(!cache.dtype.info().is_quantized);
            try compileRequire(!new_kv.dtype.info().is_quantized);
            try compileRequire(!end_idx.dtype.info().is_quantized);

            try compileRequire(cache.dtype == new_kv.dtype);
            try compileRequire(cache.dtype == .f16 or cache.dtype == .f32);
            try compileRequire(end_idx.dtype == .i32);

            // Shape contract.
            try compileRequire(cache.shape[0] == new_kv.shape[0]); // B
            try compileRequire(cache.shape[1] == new_kv.shape[1]); // H_kv
            try compileRequire(cache.shape[3] == new_kv.shape[3]); // D
            try compileRequire(end_idx.shape[0] == cache.shape[0]);

            // Tiling contract:
            // - full head_dim contiguous in one tile (kernel writes a row per (b,h,t)
            //   as one memcpy, so head_dim must not span tile boundaries)
            // - single tile for end_index vector (kernel reads it via tile 0)
            // Batch/heads/time can be arbitrarily tiled — the kernel maps logical
            // (b, h, t) to tile coords by integer division (sequence_append.zig:116-124).
            try compileRequire(cache.tile_counts[3] == 1 and new_kv.tile_counts[3] == 1);
            try compileRequire(cache.tile_shape[3] == cache.shape[3]);
            try compileRequire(new_kv.tile_shape[3] == new_kv.shape[3]);
            try compileRequire(end_idx.tile_counts[0] == 1);
        },

        .LSTMCellFused => |s| {
            const out_state: *const TiledTensor = mgr.getConst(s.out_state) catch return CompileError.InvalidArgument;
            const x: *const TiledTensor = mgr.getConst(s.x) catch return CompileError.InvalidArgument;
            const h_prev: *const TiledTensor = mgr.getConst(s.h_prev) catch return CompileError.InvalidArgument;
            const c_prev: *const TiledTensor = mgr.getConst(s.c_prev) catch return CompileError.InvalidArgument;
            const w_ih: *const TiledTensor = mgr.getConst(s.w_ih) catch return CompileError.InvalidArgument;
            const w_hh: *const TiledTensor = mgr.getConst(s.w_hh) catch return CompileError.InvalidArgument;

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
                const b_ih: *const TiledTensor = mgr.getConst(bid) catch return CompileError.InvalidArgument;
                try compileRequire(b_ih.dtype == out_state.dtype);
                try compileRequire(b_ih.rank == 1);
                try compileRequire(b_ih.shape[0] == gate_dim);
            } else {
                try compileRequire(s.b_hh == null);
            }

            if (s.b_hh) |bid| {
                const b_hh: *const TiledTensor = mgr.getConst(bid) catch return CompileError.InvalidArgument;
                try compileRequire(b_hh.dtype == out_state.dtype);
                try compileRequire(b_hh.rank == 1);
                try compileRequire(b_hh.shape[0] == gate_dim);
            }
        },

        .RFFT => |s| {
            const out: *const TiledTensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const x: *const TiledTensor = mgr.getConst(s.x) catch return CompileError.InvalidArgument;

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
            const out: *const TiledTensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const signal: *const TiledTensor = mgr.getConst(s.signal) catch return CompileError.InvalidArgument;
            const window: *const TiledTensor = mgr.getConst(s.window) catch return CompileError.InvalidArgument;

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
            const out: *const TiledTensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const a: *const TiledTensor = mgr.getConst(s.a) catch return CompileError.InvalidArgument;
            try compileRequire(out.rank == 1 and out.shape[0] == 1);
            try compileRequire(isScalarSupported(out.dtype));
            try compileRequire(out.dtype == a.dtype);
            try compileRequire(isScalarSupported(a.dtype));
            _ = s.op;
        },

        .ReduceAxis => |s| {
            const out: *const TiledTensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const a: *const TiledTensor = mgr.getConst(s.a) catch return CompileError.InvalidArgument;
            try compileRequire(isScalarSupported(out.dtype));
            try compileRequire(out.dtype == a.dtype);
            try compileRequire(isScalarSupported(a.dtype));
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
            const out: *const TiledTensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            try compileRequire(isScalarSupported(out.dtype));

            const count: usize = @as(usize, s.input_count);
            try compileRequire(count >= 1);
            const rank: usize = @as(usize, out.rank);
            try compileRequire(rank >= 1 and rank <= MAX_RANK);
            try compileRequire(s.axis < rank);

            var axis_sum: usize = 0;
            var i: usize = 0;
            while (i < count) : (i += 1) {
                const in_t: *const TiledTensor = mgr.getConst(s.inputs[i]) catch return CompileError.InvalidArgument;
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

        .ReTileCopyScalar => |s| {
            const dst: *const TiledTensor = mgr.getConst(s.dst) catch return CompileError.InvalidArgument;
            const src: *const TiledTensor = mgr.getConst(s.src) catch return CompileError.InvalidArgument;
            try compileRequire(isScalarSupported(dst.dtype));
            try compileRequire(dst.dtype == src.dtype);
            try compileRequire(dst.rank == src.rank);
            try requireSameShape(dst.shape, src.shape);
        },

        .ReshapeScalar => |s| {
            const dst: *const TiledTensor = mgr.getConst(s.dst) catch return CompileError.InvalidArgument;
            const src: *const TiledTensor = mgr.getConst(s.src) catch return CompileError.InvalidArgument;
            try compileRequire(isScalarSupported(dst.dtype));
            try compileRequire(dst.dtype == src.dtype);
            const src_elems: usize = try elemCount(src.shape);
            const dst_elems: usize = try elemCount(dst.shape);
            try compileRequire(src_elems == dst_elems);
        },

        .Transpose2DScalar => |s| {
            const dst: *const TiledTensor = mgr.getConst(s.dst) catch return CompileError.InvalidArgument;
            const src: *const TiledTensor = mgr.getConst(s.src) catch return CompileError.InvalidArgument;
            try compileRequire(isScalarSupported(dst.dtype));
            try compileRequire(dst.dtype == src.dtype);
            try compileRequire(dst.rank == 2 and src.rank == 2);
            try compileRequire(dst.shape[0] == src.shape[1] and dst.shape[1] == src.shape[0]);
        },

        .SliceNDScalar => |s| {
            const dst: *const TiledTensor = mgr.getConst(s.dst) catch return CompileError.InvalidArgument;
            const src: *const TiledTensor = mgr.getConst(s.src) catch return CompileError.InvalidArgument;
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
    policy: plan_mod.TilePolicy,
    steps: *std.ArrayList(Step),
    step: Step,
) CompileError!void {
    validateStep(mgr, policy, step) catch |e| {
        if (traceEnabled()) {
            std.debug.print("[aion][compile] validateStep failed: step={s} err={s}\n", .{ @tagName(step), @errorName(e) });
            debugDumpStep(mgr, step);
        }
        return e;
    };
    steps.append(allocator, step) catch return CompileError.OutOfMemory;
}

fn debugDumpTensorMeta(mgr: *StorageManager, tid: TensorId, label: []const u8) void {
    const t: *const TiledTensor = mgr.getConst(tid) catch {
        std.debug.print("  {s}: <invalid tensor id {d}>\n", .{ label, tid });
        return;
    };
    std.debug.print(
        "  {s}: id={d} dtype={s} rank={d} shape={any} tile_shape={any} tile_counts={any} quant_axis={d}\n",
        .{ label, tid, @tagName(t.dtype), t.rank, t.shape, t.tile_shape, t.tile_counts, t.quant_axis },
    );
}

fn debugDumpStep(mgr: *StorageManager, step: Step) void {
    switch (step) {
        .SequenceAppendTiled => |s| {
            debugDumpTensorMeta(mgr, s.cache, "cache");
            debugDumpTensorMeta(mgr, s.new_kv, "new_kv");
            debugDumpTensorMeta(mgr, s.end_index, "end_index");
        },
        .MultiHeadAttentionCachedTiled => |s| {
            debugDumpTensorMeta(mgr, s.q, "q");
            debugDumpTensorMeta(mgr, s.k_cache, "k_cache");
            debugDumpTensorMeta(mgr, s.v_cache, "v_cache");
            debugDumpTensorMeta(mgr, s.positions, "positions");
            debugDumpTensorMeta(mgr, s.end_index, "end_index");
            debugDumpTensorMeta(mgr, s.out, "out");
        },
        .RoPE1DTiled => |s| {
            debugDumpTensorMeta(mgr, s.x, "x");
            debugDumpTensorMeta(mgr, s.positions, "positions");
            debugDumpTensorMeta(mgr, s.out, "out");
        },
        .GatherRowsTiled => |s| {
            debugDumpTensorMeta(mgr, s.table, "table");
            debugDumpTensorMeta(mgr, s.indices, "indices");
            debugDumpTensorMeta(mgr, s.out, "out");
        },
        .MatMulTiled => |s| {
            debugDumpTensorMeta(mgr, s.a, "a");
            debugDumpTensorMeta(mgr, s.b, "b");
            debugDumpTensorMeta(mgr, s.c, "c");
        },
        .MatMulNTTiled => |s| {
            debugDumpTensorMeta(mgr, s.a, "a");
            debugDumpTensorMeta(mgr, s.b, "b");
            debugDumpTensorMeta(mgr, s.c, "c");
        },
        else => {},
    }
}

/// Like `fillTileShapeDefault`, but non-CPU targets get one full-shape tile:
/// the v1 GPU execs for RFFT/STFT/LSTM bind exactly one buffer per operand and
/// reject multi-tile tensors, so their outputs must never be split by the
/// square-tile heuristic (which kicks in past `small_tensor_threshold`).
fn fillTileShapeSingleOnGpu(policy: plan_mod.TilePolicy, dtype: types.DType, shape: []const usize, out: []usize) CompileError!void {
    if (policy.target_kind != .cpu) {
        if (out.len != shape.len or shape.len == 0) return CompileError.InvalidArgument;
        @memcpy(out, shape);
        return;
    }
    return fillTileShapeDefault(policy, dtype, shape, out);
}

fn fillTileShapeDefault(policy: plan_mod.TilePolicy, dtype: types.DType, shape: []const usize, out: []usize) CompileError!void {
    if (out.len != shape.len) return CompileError.InvalidArgument;
    if (shape.len == 0) return CompileError.InvalidArgument;

    // Small tensors: use full dimensions as a single tile.  Avoids per-tile
    // overhead (acquire/release, multi-tile element-by-element read/write)
    // that dominates for small workloads.
    var total: usize = 1;
    for (shape) |dim| total = std.math.mul(usize, total, dim) catch break;
    if (total <= policy.small_tensor_threshold) {
        @memcpy(out, shape);
        _ = dtype;
        return;
    }

    if (shape.len == 1) {
        const t1: [1]usize = plan_mod.chooseTileShape1D(policy, shape[0]);
        out[0] = t1[0];
        _ = dtype;
        return;
    }

    var d: usize = 0;
    while (d + 2 < shape.len) : (d += 1) {
        out[d] = 1;
    }

    const m: usize = shape[shape.len - 2];
    const n: usize = shape[shape.len - 1];
    const t2: [2]usize = plan_mod.chooseTileShape2DSquare(policy, m, n);
    out[shape.len - 2] = t2[0];
    out[shape.len - 1] = t2[1];
    _ = dtype;
}

pub const OptPolicy = optimize_mod.OptPolicy;

pub fn compileGraph(
    allocator: std.mem.Allocator,
    graph: *graph_mod.Graph,
    mgr: *StorageManager,
    policy: plan_mod.TilePolicy,
) CompileError!Program {
    return compileGraphOpt(allocator, graph, mgr, policy, .{});
}

pub fn compileGraphOpt(
    allocator: std.mem.Allocator,
    graph: *graph_mod.Graph,
    mgr: *StorageManager,
    policy: plan_mod.TilePolicy,
    opt: OptPolicy,
) CompileError!Program {
    try infer_mod.infer(graph);

    // Graph-rewrite optimization passes (e.g. horizontal MatMul fusion). Runs
    // after inference (so rewrites see concrete shapes) and before the
    // value→tensor map below is sized (so appended values compile normally).
    try optimize_mod.run(allocator, graph, mgr, policy, opt);

    // Map graph values -> concrete tensors.
    const v_count: usize = graph.values.items.len;
    var value_tensor: []TensorId = try allocator.alloc(TensorId, v_count);
    defer allocator.free(value_tensor);
    @memset(value_tensor, @as(TensorId, 0));

    var value_has_tensor: []bool = try allocator.alloc(bool, v_count);
    defer allocator.free(value_has_tensor);
    @memset(value_has_tensor, false);

    // External bindings.
    for (graph.values.items, 0..) |v, i| {
        if (v.external) |ext| {
            const tid: TensorId = @intCast(ext);
            const t: *const TiledTensor = try mgr.getConst(tid);
            if (v.dtype.? != t.dtype) return CompileError.InvalidArgument;
            if (v.shape.len != t.rank) return CompileError.InvalidArgument;
            var d: usize = 0;
            while (d < v.shape.len) : (d += 1) {
                if (v.shape[d] != t.shape[d]) return CompileError.InvalidArgument;
            }

            value_tensor[i] = tid;
            value_has_tensor[i] = true;
        }
    }

    // Compile into a dynamic step list first.
    var steps: std.ArrayList(Step) = .empty;
    errdefer steps.deinit(allocator);

    var blocks: std.ArrayList(executable.Block) = .empty;
    errdefer {
        for (blocks.items) |block| allocator.free(block.steps);
        blocks.deinit(allocator);
    }

    // Helper: allocate tensor for a value (owned).
    const AllocCtx = struct {
        allocator: std.mem.Allocator,
        mgr: *StorageManager,
        policy: plan_mod.TilePolicy,
        value_tensor: []TensorId,
        value_has_tensor: []bool,

        fn ensureValueTensor(self: *@This(), value_index: usize, dtype: types.DType, shape: []const usize, tile_shape: []const usize) CompileError!TensorId {
            if (self.value_has_tensor[value_index]) return self.value_tensor[value_index];

            const tid: TensorId = try self.mgr.createTiledTensor(dtype, shape, tile_shape, .{ .tile_alignment = self.policy.tile_alignment });
            self.value_tensor[value_index] = tid;
            self.value_has_tensor[value_index] = true;
            return tid;
        }
    };

    var ctx: AllocCtx = .{ .allocator = allocator, .mgr = mgr, .policy = policy, .value_tensor = value_tensor, .value_has_tensor = value_has_tensor };

    // Lower nodes in order.
    for (graph.nodes.items) |node| {
        try lowerNode(allocator, graph, node, mgr, policy, &ctx, &steps, &blocks);
    }

    // Collect outputs.
    const out_ids: []TensorId = try allocator.alloc(TensorId, graph.outputs.items.len);
    for (graph.outputs.items, 0..) |vid, i| {
        const idx: usize = @intCast(vid);
        if (!value_has_tensor[idx]) return CompileError.InvalidArgument;
        out_ids[i] = value_tensor[idx];
    }

    const steps_slice: []Step = try steps.toOwnedSlice(allocator);
    const blocks_slice: []executable.Block = try blocks.toOwnedSlice(allocator);

    return .{ .allocator = allocator, .steps = steps_slice, .blocks = blocks_slice, .outputs = out_ids };
}

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
    policy: plan_mod.TilePolicy,
    ctx: anytype,
    blocks: *std.ArrayList(executable.Block),
    /// For loop bodies: the carry-in tensor for each region output (len ==
    /// `region.outputs.len`). Each carried output is reconciled to match its
    /// carry-in's tiling. Pass `null` for non-loop regions (e.g. `If` branches).
    carry_match: ?[]const TensorId,
) CompileError!executable.BlockId {
    var region_steps: std.ArrayList(Step) = .empty;
    errdefer region_steps.deinit(allocator);
    for (region.nodes) |rnode| {
        try lowerNode(allocator, graph, rnode, mgr, policy, ctx, &region_steps, blocks);
    }
    // A loop's executor writes each carried output back into the carry buffer, so
    // the two must be tiled identically. Body ops may choose a different tiling
    // (e.g. a small matmul output tiled {1,1} feeding a unary that inherits it),
    // so insert an in-body retile whenever they diverge. A matching tiling makes
    // `ensureTilingScalarMaybeRetile` a no-op.
    if (carry_match) |carried| {
        for (region.outputs, 0..) |out_vid, i| {
            const idx: usize = @intCast(out_vid);
            const want: *const TiledTensor = try mgr.getConst(carried[i]);
            var want_buf: [MAX_RANK]usize = undefined;
            const want_tile: []usize = want_buf[0..want.tile_shape.len];
            @memcpy(want_tile, want.tile_shape);
            const v = graph.values.items[idx];
            _ = try ensureTilingScalarMaybeRetile(allocator, &region_steps, mgr, policy, ctx, idx, v.dtype.?, v.shape, want_tile);
        }
    }
    const block_steps: []Step = try region_steps.toOwnedSlice(allocator);
    errdefer allocator.free(block_steps);
    const id: executable.BlockId = @intCast(blocks.items.len);
    blocks.append(allocator, .{ .steps = block_steps }) catch return CompileError.OutOfMemory;
    return id;
}

/// Lower a single graph node into tiled executable steps. Shared by the
/// top-level node loop and `lowerRegionBlock` (control-flow region bodies),
/// so both paths get identical op support. `ctx`/`steps`/`blocks` are pointers.
fn lowerNode(
    allocator: std.mem.Allocator,
    graph: *graph_mod.Graph,
    node: graph_mod.Node,
    mgr: *StorageManager,
    policy: plan_mod.TilePolicy,
    ctx: anytype,
    steps: *std.ArrayList(Step),
    blocks: *std.ArrayList(executable.Block),
) CompileError!void {
    if (!graph_mod.opInputCountValid(node.op, node.inputs.len)) return CompileError.InvalidArgument;
    const out_idx: usize = @intCast(node.output);
    const out_v = graph.values.items[out_idx];
    const out_dt: types.DType = out_v.dtype.?;
    const out_shape: []const usize = out_v.shape;

    switch (node.op) {
        .MatMul => |mm| {
            const a_id: usize = @intCast(node.inputs[0]);
            const b_id: usize = @intCast(node.inputs[1]);
            const a_v = graph.values.items[a_id];
            const b_v = graph.values.items[b_id];
            const rank: usize = a_v.shape.len;
            const m: usize = a_v.shape[rank - 2];
            const k: usize = a_v.shape[rank - 1];
            const b_dtype: types.DType = b_v.dtype.?;
            const n: usize = b_v.shape[rank - 1];

            // Quantized B tensors cannot be re-tiled today (we only have scalar
            // retile kernels), so their tiling must be stable across dynamic M.
            // Use a fixed M-hint for quantized B so `MatMul` works for seq==1 and
            // longer prefill runs without demanding different weight tile shapes.
            const tiles = if (b_dtype.info().is_quantized) blk: {
                const m_hint: usize = @max(@as(usize, 1), policy.base_square_2d);
                break :blk plan_mod.chooseMatMulTiles(policy, m_hint, n, k, b_dtype);
            } else plan_mod.chooseMatMulTiles(policy, m, n, k, b_dtype);

            var c_tile_buf: [MAX_RANK]usize = undefined;
            var a_tile_buf: [MAX_RANK]usize = undefined;
            var b_tile_buf: [MAX_RANK]usize = undefined;
            const c_tile: []usize = c_tile_buf[0..rank];
            const a_tile: []usize = a_tile_buf[0..rank];
            const b_tile: []usize = b_tile_buf[0..rank];

            var d: usize = 0;
            while (d + 2 < rank) : (d += 1) {
                c_tile[d] = 1;
                a_tile[d] = 1;
                b_tile[d] = 1;
            }
            c_tile[rank - 2] = tiles.tm;
            c_tile[rank - 1] = tiles.tn;
            a_tile[rank - 2] = tiles.tm;
            a_tile[rank - 1] = tiles.tk;
            b_tile[rank - 2] = tiles.tk;
            b_tile[rank - 1] = tiles.tn;

            const c_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, c_tile);

            // Ensure A and B have compatible tiling; if not, materialize into new tensors.
            const a_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, a_id, a_v.dtype.?, a_v.shape, a_tile);
            const b_tid: TensorId = try ensureTilingMaybeRetile(allocator, steps, mgr, policy, ctx, b_id, b_v.dtype.?, b_v.shape, b_tile);

            try appendStepChecked(allocator, mgr, policy, steps, .{ .MatMulTiled = .{ .c = c_tid, .a = a_tid, .b = b_tid, .alpha = mm.alpha, .beta = mm.beta } });
        },

        .ElemwiseBinary => |eb| {
            const a_id: usize = @intCast(node.inputs[0]);
            const b_id: usize = @intCast(node.inputs[1]);
            const a_v = graph.values.items[a_id];
            const b_v = graph.values.items[b_id];
            var tile_buf: [MAX_RANK]usize = undefined;
            const tile: []usize = tile_buf[0..out_shape.len];

            // Prefer inheriting tiling from an existing input tensor to avoid
            // retile failures for large activations (e.g. quant matmul outputs
            // use a fixed tm/tn, which may differ from the default tiler for
            // short-and-wide matrices like [14, 12288]).
            if (ctx.value_has_tensor[a_id]) {
                const a_t: *const TiledTensor = try mgr.getConst(ctx.value_tensor[a_id]);
                if (a_t.tile_shape.len == out_shape.len) {
                    @memcpy(tile, a_t.tile_shape);
                } else {
                    try fillTileShapeDefault(policy, out_dt, out_shape, tile);
                }
            } else if (ctx.value_has_tensor[b_id]) {
                const b_t: *const TiledTensor = try mgr.getConst(ctx.value_tensor[b_id]);
                if (b_t.tile_shape.len == out_shape.len) {
                    @memcpy(tile, b_t.tile_shape);
                } else {
                    try fillTileShapeDefault(policy, out_dt, out_shape, tile);
                }
            } else {
                try fillTileShapeDefault(policy, out_dt, out_shape, tile);
            }

            const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, tile);
            const a_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, a_id, a_v.dtype.?, a_v.shape, tile);
            const b_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, b_id, b_v.dtype.?, b_v.shape, tile);

            try appendStepChecked(allocator, mgr, policy, steps, .{ .ElemwiseBinaryTiled = .{ .op = eb.op, .out = out_tid, .a = a_tid, .b = b_tid } });
        },

        .BroadcastLastDimBinary => |bb| {
            const a_id: usize = @intCast(node.inputs[0]);
            const b_id: usize = @intCast(node.inputs[1]);
            const a_v = graph.values.items[a_id];
            const b_v = graph.values.items[b_id];

            var tile_out_buf: [MAX_RANK]usize = undefined;
            const tile_out: []usize = tile_out_buf[0..out_shape.len];

            // Prefer inheriting tiling from A when possible (broadcast is
            // elementwise over the full output shape).
            if (ctx.value_has_tensor[a_id]) {
                const a_t: *const TiledTensor = try mgr.getConst(ctx.value_tensor[a_id]);
                if (a_t.tile_shape.len == out_shape.len) {
                    @memcpy(tile_out, a_t.tile_shape);
                } else {
                    try fillTileShapeDefault(policy, out_dt, out_shape, tile_out);
                }
            } else {
                try fillTileShapeDefault(policy, out_dt, out_shape, tile_out);
            }
            const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, tile_out);
            const a_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, a_id, a_v.dtype.?, a_v.shape, tile_out);

            const last: usize = out_shape.len - 1;
            // b must be rank-1 tiled to match the last-dim tile.
            const b_tile: [1]usize = .{tile_out[last]};
            const b_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, b_id, b_v.dtype.?, b_v.shape, b_tile[0..]);

            try appendStepChecked(allocator, mgr, policy, steps, .{ .BroadcastLastDimBinaryTiled = .{ .op = bb.op, .out = out_tid, .a = a_tid, .b = b_tid } });
        },

        .Unary => |u| {
            const a_id: usize = @intCast(node.inputs[0]);
            const a_v = graph.values.items[a_id];
            var tile_buf: [MAX_RANK]usize = undefined;
            const tile: []usize = tile_buf[0..out_shape.len];

            // Inherit input tiling when available to avoid retile failures
            // (e.g. Conv1D output uses chooseConv1DTiles, which may differ
            // from the default chooseTileShape2DSquare).
            if (ctx.value_has_tensor[a_id]) {
                const a_t: *const TiledTensor = try mgr.getConst(ctx.value_tensor[a_id]);
                if (a_t.tile_shape.len == out_shape.len) {
                    @memcpy(tile, a_t.tile_shape);
                } else {
                    try fillTileShapeDefault(policy, out_dt, out_shape, tile);
                }
            } else {
                try fillTileShapeDefault(policy, out_dt, out_shape, tile);
            }

            const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, tile);
            const a_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, a_id, a_v.dtype.?, a_v.shape, tile);
            try appendStepChecked(allocator, mgr, policy, steps, .{ .UnaryTiled = .{ .op = u.op, .out = out_tid, .a = a_tid } });
        },

        .Softmax => |sm| {
            const a_id: usize = @intCast(node.inputs[0]);
            const a_v = graph.values.items[a_id];

            // v0: softmax only for f32.
            if (a_v.dtype.? != .f32) return CompileError.InvalidArgument;
            const rank: usize = out_shape.len;
            _ = try normalizeAxis(sm.axis, rank);

            var tile_buf: [MAX_RANK]usize = undefined;
            const tile: []usize = tile_buf[0..out_shape.len];

            if (ctx.value_has_tensor[a_id]) {
                const a_tid_existing: TensorId = ctx.value_tensor[a_id];
                const a_t_existing: *const TiledTensor = try mgr.getConst(a_tid_existing);
                if (a_t_existing.tile_shape.len != out_shape.len) return CompileError.InvalidArgument;
                var d0: usize = 0;
                while (d0 < out_shape.len) : (d0 += 1) {
                    tile[d0] = a_t_existing.tile_shape[d0];
                }
            } else if (out_shape.len == 1) {
                const t1: [1]usize = plan_mod.chooseTileShape1D(policy, out_shape[0]);
                // Keep tm <= 256 for per-row scratch.
                tile[0] = @min(@as(usize, 256), t1[0]);
            } else {
                // Default softmax tiling: tile last two dims for reasonable locality.
                var d: usize = 0;
                while (d + 2 < rank) : (d += 1) {
                    tile[d] = 1;
                }
                const m: usize = out_shape[rank - 2];
                const n: usize = out_shape[rank - 1];
                const st = plan_mod.chooseSoftmaxTiles(policy, m, n);
                tile[rank - 2] = st.tm;
                tile[rank - 1] = st.tn;
            }

            const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, tile);
            const a_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, a_id, a_v.dtype.?, a_v.shape, tile);
            try appendStepChecked(allocator, mgr, policy, steps, .{ .SoftmaxTiled = .{ .out = out_tid, .a = a_tid, .axis = sm.axis } });
        },

        .Conv1D => |cv| {
            const x_id: usize = @intCast(node.inputs[0]);
            const w_id: usize = @intCast(node.inputs[1]);

            const x_v = graph.values.items[x_id];
            const rank: usize = out_shape.len;
            if (rank < 2) return CompileError.InvalidArgument;

            var out_tile_buf: [MAX_RANK]usize = undefined;
            const out_tile: []usize = out_tile_buf[0..rank];
            @memset(out_tile, 1);

            // Keep leading + length tiling aligned with X when available.
            if (ctx.value_has_tensor[x_id]) {
                const x_tid_existing: TensorId = ctx.value_tensor[x_id];
                const x_t_existing: *const TiledTensor = try mgr.getConst(x_tid_existing);
                var d: usize = 0;
                while (d < rank - 1) : (d += 1) {
                    out_tile[d] = x_t_existing.tile_shape[d];
                }
            } else {
                const ct = plan_mod.chooseConv1DTiles(policy, out_shape[rank - 2], out_shape[rank - 1]);
                if (rank > 2) @memset(out_tile[0 .. rank - 2], 1);
                out_tile[rank - 2] = ct.tl;
            }
            const ct = plan_mod.chooseConv1DTiles(policy, out_shape[rank - 2], out_shape[rank - 1]);
            out_tile[rank - 1] = ct.tc;

            // Depthwise conv: keep C tiling aligned with X when possible.
            // This preserves eligibility for the depthwise tile-native kernel
            // (including reflect padding), even when the generic Conv1D tile
            // chooser would collapse small outputs into a single C tile.
            if (cv.groups == x_v.shape[rank - 1] and cv.groups == out_shape[rank - 1] and ctx.value_has_tensor[x_id]) {
                const x_tid_existing: TensorId = ctx.value_tensor[x_id];
                const x_t_existing: *const TiledTensor = try mgr.getConst(x_tid_existing);
                if (x_t_existing.tile_shape.len == rank) {
                    const tc_x: usize = x_t_existing.tile_shape[rank - 1];
                    if (tc_x != 0 and tc_x <= out_shape[rank - 1]) {
                        out_tile[rank - 1] = tc_x;
                    }
                }
            }

            const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile);
            const x_tid: TensorId = try ensureAnyTensor(ctx, x_id);
            const w_tid: TensorId = try ensureAnyTensor(ctx, w_id);

            var bias_tid: ?TensorId = null;
            if (node.inputs.len == 3) {
                const b_id: usize = @intCast(node.inputs[2]);
                bias_tid = try ensureAnyTensor(ctx, b_id);
            }

            try appendStepChecked(allocator, mgr, policy, steps, .{ .Conv1DTiled = .{
                .out = out_tid,
                .x = x_tid,
                .w = w_tid,
                .bias = bias_tid,
                .stride = cv.stride,
                .dilation = cv.dilation,
                .pad_left = cv.pad_left,
                .pad_right = cv.pad_right,
                .pad_mode = cv.pad_mode,
                .groups = cv.groups,
            } });
        },

        .Conv2D => |cv| {
            const x_id: usize = @intCast(node.inputs[0]);
            const w_id: usize = @intCast(node.inputs[1]);
            const rank: usize = out_shape.len;
            if (rank < 3) return CompileError.InvalidArgument;

            var out_tile_buf: [MAX_RANK]usize = undefined;
            const out_tile: []usize = out_tile_buf[0..rank];
            @memset(out_tile, 1);

            // Keep leading + spatial tiling aligned with X when available.
            if (ctx.value_has_tensor[x_id]) {
                const x_tid_existing: TensorId = ctx.value_tensor[x_id];
                const x_t_existing: *const TiledTensor = try mgr.getConst(x_tid_existing);
                var d: usize = 0;
                while (d < rank - 1) : (d += 1) {
                    out_tile[d] = x_t_existing.tile_shape[d];
                }
            } else {
                const ct = plan_mod.chooseConv2DTiles(policy, out_shape[rank - 3], out_shape[rank - 2], out_shape[rank - 1]);
                if (rank > 3) @memset(out_tile[0 .. rank - 3], 1);
                out_tile[rank - 3] = ct.th;
                out_tile[rank - 2] = ct.tw;
            }
            const ct = plan_mod.chooseConv2DTiles(policy, out_shape[rank - 3], out_shape[rank - 2], out_shape[rank - 1]);
            out_tile[rank - 1] = ct.tc;

            const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile);
            const x_tid: TensorId = try ensureAnyTensor(ctx, x_id);
            const w_tid: TensorId = try ensureAnyTensor(ctx, w_id);

            var bias_tid: ?TensorId = null;
            if (node.inputs.len == 3) {
                const b_id: usize = @intCast(node.inputs[2]);
                bias_tid = try ensureAnyTensor(ctx, b_id);
            }

            try appendStepChecked(allocator, mgr, policy, steps, .{ .Conv2DTiled = .{
                .out = out_tid,
                .x = x_tid,
                .w = w_tid,
                .bias = bias_tid,
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
            const x_id: usize = @intCast(node.inputs[0]);
            const g_id: usize = @intCast(node.inputs[1]);
            const b_id: usize = @intCast(node.inputs[2]);
            const x_v = graph.values.items[x_id];
            const g_v = graph.values.items[g_id];
            const b_v = graph.values.items[b_id];

            if (out_dt != .f32 and out_dt != .f16) return CompileError.InvalidArgument;
            if (ln.normalized_shape.len == 0) return CompileError.InvalidArgument;
            if (out_shape.len < ln.normalized_shape.len) return CompileError.InvalidArgument;

            const rank: usize = out_shape.len;
            const norm_rank: usize = ln.normalized_shape.len;

            var tile_buf: [MAX_RANK]usize = undefined;
            const tile: []usize = tile_buf[0..out_shape.len];

            if (ctx.value_has_tensor[x_id]) {
                const x_tid_existing: TensorId = ctx.value_tensor[x_id];
                const x_t_existing: *const TiledTensor = try mgr.getConst(x_tid_existing);
                if (x_t_existing.tile_shape.len != out_shape.len) return CompileError.InvalidArgument;
                var d0: usize = 0;
                while (d0 < out_shape.len) : (d0 += 1) {
                    tile[d0] = x_t_existing.tile_shape[d0];
                }
            } else {
                @memset(tile, 1);

                const last_dim: usize = out_shape[rank - 1];
                const m_ref: usize = if (norm_rank >= 2) ln.normalized_shape[norm_rank - 2] else if (rank >= 2) out_shape[rank - 2] else 1;
                const st = plan_mod.chooseNormTiles(policy, m_ref, last_dim);
                if (rank >= 2) tile[rank - 2] = st.tm;
                tile[rank - 1] = st.tn;
            }

            const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, tile);
            const x_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, x_id, x_v.dtype.?, x_v.shape, tile);

            var bcast_tile_buf: [MAX_RANK]usize = undefined;
            const bcast_tile: []usize = bcast_tile_buf[0..norm_rank];
            var d: usize = 0;
            while (d < norm_rank) : (d += 1) {
                bcast_tile[d] = tile[rank - norm_rank + d];
            }
            const gamma_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, g_id, g_v.dtype.?, g_v.shape, bcast_tile);
            const beta_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, b_id, b_v.dtype.?, b_v.shape, bcast_tile);

            try appendStepChecked(allocator, mgr, policy, steps, .{ .LayerNormTiled = .{ .out = out_tid, .x = x_tid, .gamma = gamma_tid, .beta = beta_tid, .eps = ln.eps } });
        },

        .RMSNorm => |rn| {
            const x_id: usize = @intCast(node.inputs[0]);
            const g_id: usize = @intCast(node.inputs[1]);
            const b_id: usize = @intCast(node.inputs[2]);
            const x_v = graph.values.items[x_id];
            const g_v = graph.values.items[g_id];
            const b_v = graph.values.items[b_id];

            if (out_dt != .f32 and out_dt != .f16) return CompileError.InvalidArgument;
            if (rn.normalized_shape.len == 0) return CompileError.InvalidArgument;
            if (out_shape.len < rn.normalized_shape.len) return CompileError.InvalidArgument;

            const rank: usize = out_shape.len;
            const norm_rank: usize = rn.normalized_shape.len;

            var tile_buf: [MAX_RANK]usize = undefined;
            const tile: []usize = tile_buf[0..out_shape.len];

            if (ctx.value_has_tensor[x_id]) {
                const x_tid_existing: TensorId = ctx.value_tensor[x_id];
                const x_t_existing: *const TiledTensor = try mgr.getConst(x_tid_existing);
                if (x_t_existing.tile_shape.len != out_shape.len) return CompileError.InvalidArgument;
                var d0: usize = 0;
                while (d0 < out_shape.len) : (d0 += 1) {
                    tile[d0] = x_t_existing.tile_shape[d0];
                }
            } else {
                @memset(tile, 1);

                const last_dim: usize = out_shape[rank - 1];
                const m_ref: usize = if (norm_rank >= 2) rn.normalized_shape[norm_rank - 2] else if (rank >= 2) out_shape[rank - 2] else 1;
                const st = plan_mod.chooseNormTiles(policy, m_ref, last_dim);
                if (rank >= 2) tile[rank - 2] = st.tm;
                tile[rank - 1] = st.tn;
            }

            const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, tile);
            const x_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, x_id, x_v.dtype.?, x_v.shape, tile);

            var bcast_tile_buf: [MAX_RANK]usize = undefined;
            const bcast_tile: []usize = bcast_tile_buf[0..norm_rank];
            var d: usize = 0;
            while (d < norm_rank) : (d += 1) {
                bcast_tile[d] = tile[rank - norm_rank + d];
            }
            const gamma_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, g_id, g_v.dtype.?, g_v.shape, bcast_tile);
            const beta_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, b_id, b_v.dtype.?, b_v.shape, bcast_tile);

            try appendStepChecked(allocator, mgr, policy, steps, .{ .RMSNormTiled = .{ .out = out_tid, .x = x_tid, .gamma = gamma_tid, .beta = beta_tid, .eps = rn.eps } });
        },

        .Attention => |attn| {
            const q_id: usize = @intCast(node.inputs[0]);
            const k_id: usize = @intCast(node.inputs[1]);
            const v_id: usize = @intCast(node.inputs[2]);

            const q_v = graph.values.items[q_id];
            const k_v = graph.values.items[k_id];
            const v_v = graph.values.items[v_id];

            if (out_shape.len < 2) return CompileError.InvalidArgument;
            if (!isScalarSupported(out_dt)) return CompileError.InvalidArgument;

            const rank: usize = out_shape.len;
            const m: usize = q_v.shape[rank - 2];
            const n: usize = k_v.shape[rank - 2];
            const dk: usize = q_v.shape[rank - 1];
            const dv: usize = v_v.shape[rank - 1];

            const st = plan_mod.chooseAttentionTiles(policy, m, n, dk, dv);

            var out_tile_buf: [MAX_RANK]usize = undefined;
            const out_tile_full: []usize = out_tile_buf[0..rank];
            @memset(out_tile_full, 1);
            out_tile_full[rank - 2] = st.tm;
            out_tile_full[rank - 1] = st.tv;
            const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile_full);

            var q_tile_buf: [MAX_RANK]usize = undefined;
            var k_tile_buf: [MAX_RANK]usize = undefined;
            var v_tile_buf: [MAX_RANK]usize = undefined;
            const q_tile: []usize = q_tile_buf[0..rank];
            const k_tile: []usize = k_tile_buf[0..rank];
            const v_tile: []usize = v_tile_buf[0..rank];
            @memset(q_tile, 1);
            @memset(k_tile, 1);
            @memset(v_tile, 1);
            q_tile[rank - 2] = st.tm;
            q_tile[rank - 1] = st.tk;
            k_tile[rank - 2] = st.tn;
            k_tile[rank - 1] = st.tk;
            v_tile[rank - 2] = st.tn;
            v_tile[rank - 1] = st.tv;

            const q_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, q_id, q_v.dtype.?, q_v.shape, q_tile);
            const k_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, k_id, k_v.dtype.?, k_v.shape, k_tile);
            const v_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, v_id, v_v.dtype.?, v_v.shape, v_tile);

            try appendStepChecked(allocator, mgr, policy, steps, .{ .AttentionTiled = .{ .out = out_tid, .q = q_tid, .k = k_tid, .v = v_tid, .scale = attn.scale, .causal = attn.causal } });
        },

        .MultiHeadAttention => |attn| {
            const q_id: usize = @intCast(node.inputs[0]);
            const k_id: usize = @intCast(node.inputs[1]);
            const v_id: usize = @intCast(node.inputs[2]);

            const q_v = graph.values.items[q_id];
            const k_v = graph.values.items[k_id];
            const v_v = graph.values.items[v_id];

            if (out_shape.len < 3) return CompileError.InvalidArgument;
            if (!isScalarSupported(out_dt)) return CompileError.InvalidArgument;
            if (attn.heads == 0) return CompileError.InvalidArgument;
            const rank: usize = out_shape.len;
            const m: usize = q_v.shape[rank - 2];
            const n: usize = k_v.shape[rank - 2];
            const dk: usize = q_v.shape[rank - 1];
            const dv: usize = v_v.shape[rank - 1];
            const st = plan_mod.chooseAttentionTiles(policy, m, n, dk, dv);

            var out_tile_buf: [MAX_RANK]usize = undefined;
            const out_tile_full: []usize = out_tile_buf[0..rank];
            @memset(out_tile_full, 1);
            out_tile_full[rank - 2] = st.tm;
            out_tile_full[rank - 1] = st.tv;
            const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile_full);

            var q_tile_buf: [MAX_RANK]usize = undefined;
            var k_tile_buf: [MAX_RANK]usize = undefined;
            var v_tile_buf: [MAX_RANK]usize = undefined;
            const q_tile: []usize = q_tile_buf[0..rank];
            const k_tile: []usize = k_tile_buf[0..rank];
            const v_tile: []usize = v_tile_buf[0..rank];
            @memset(q_tile, 1);
            @memset(k_tile, 1);
            @memset(v_tile, 1);
            q_tile[rank - 2] = st.tm;
            q_tile[rank - 1] = st.tk;
            k_tile[rank - 2] = st.tn;
            k_tile[rank - 1] = st.tk;
            v_tile[rank - 2] = st.tn;
            v_tile[rank - 1] = st.tv;

            const q_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, q_id, q_v.dtype.?, q_v.shape, q_tile);
            const k_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, k_id, k_v.dtype.?, k_v.shape, k_tile);
            const v_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, v_id, v_v.dtype.?, v_v.shape, v_tile);

            try appendStepChecked(allocator, mgr, policy, steps, .{ .MultiHeadAttentionTiled = .{ .out = out_tid, .q = q_tid, .k = k_tid, .v = v_tid, .scale = attn.scale, .causal = attn.causal, .heads = attn.heads } });
        },

        .RelPosMHA => |attn| {
            const q_id: usize = @intCast(node.inputs[0]);
            const k_id: usize = @intCast(node.inputs[1]);
            const v_id: usize = @intCast(node.inputs[2]);
            const pe_id: usize = @intCast(node.inputs[3]);
            const bu_id: usize = @intCast(node.inputs[4]);
            const bv_id: usize = @intCast(node.inputs[5]);

            const q_v = graph.values.items[q_id];
            const k_v = graph.values.items[k_id];
            const v_v = graph.values.items[v_id];
            const pe_v = graph.values.items[pe_id];
            const bu_v = graph.values.items[bu_id];
            const bv_v = graph.values.items[bv_id];

            if (out_shape.len != 4) return CompileError.InvalidArgument;
            if (out_dt != .f32) return CompileError.InvalidArgument;
            if (attn.heads == 0) return CompileError.InvalidArgument;

            // Layout [B, T*, H, D].
            const t_q: usize = q_v.shape[1];
            const d: usize = q_v.shape[3];
            const t_kv: usize = k_v.shape[1];
            const p_len: usize = pe_v.shape[1];

            // Single tile over the [T, D] dims (1 and 3); [B, H] (0 and 2) are size-1 tiles.
            const out_tile: [4]usize = .{ 1, t_q, 1, d };
            const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile[0..]);

            const q_tile: [4]usize = .{ 1, t_q, 1, d };
            const k_tile: [4]usize = .{ 1, t_kv, 1, d };
            const v_tile: [4]usize = .{ 1, t_kv, 1, d };
            const pe_tile: [3]usize = .{ 1, p_len, d };
            const bu_tile: [2]usize = .{ bu_v.shape[0], bu_v.shape[1] };
            const bv_tile: [2]usize = .{ bv_v.shape[0], bv_v.shape[1] };

            const q_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, q_id, q_v.dtype.?, q_v.shape, q_tile[0..]);
            const k_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, k_id, k_v.dtype.?, k_v.shape, k_tile[0..]);
            const v_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, v_id, v_v.dtype.?, v_v.shape, v_tile[0..]);
            const pe_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, pe_id, pe_v.dtype.?, pe_v.shape, pe_tile[0..]);
            const bu_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, bu_id, bu_v.dtype.?, bu_v.shape, bu_tile[0..]);
            const bv_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, bv_id, bv_v.dtype.?, bv_v.shape, bv_tile[0..]);

            var mask_tid: ?TensorId = null;
            if (attn.has_mask) {
                const m_id: usize = @intCast(node.inputs[6]);
                const m_v = graph.values.items[m_id];
                const m_tile: [2]usize = .{ m_v.shape[0], m_v.shape[1] };
                mask_tid = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, m_id, m_v.dtype.?, m_v.shape, m_tile[0..]);
            }

            try appendStepChecked(allocator, mgr, policy, steps, .{ .RelPosMHATiled = .{
                .out = out_tid,
                .q = q_tid,
                .k = k_tid,
                .v = v_tid,
                .pos_emb = pe_tid,
                .pos_bias_u = bu_tid,
                .pos_bias_v = bv_tid,
                .mask = mask_tid,
                .scale = attn.scale,
                .heads = attn.heads,
            } });
        },

        .MultiHeadAttentionCached => |attn| {
            const q_id: usize = @intCast(node.inputs[0]);
            const k_id: usize = @intCast(node.inputs[1]);
            const v_id: usize = @intCast(node.inputs[2]);
            const pos_id: usize = @intCast(node.inputs[3]);
            const end_id: usize = @intCast(node.inputs[4]);

            const q_v = graph.values.items[q_id];
            const k_v = graph.values.items[k_id];
            const v_v = graph.values.items[v_id];
            const pos_v = graph.values.items[pos_id];
            const end_v = graph.values.items[end_id];

            if (out_shape.len != 4) return CompileError.InvalidArgument;
            if (out_dt != .f32) return CompileError.InvalidArgument;
            if (!(q_v.dtype.? == .f16 or q_v.dtype.? == .f32)) return CompileError.InvalidArgument;
            if (!(k_v.dtype.? == .f16 or k_v.dtype.? == .f32)) return CompileError.InvalidArgument;
            if (!(v_v.dtype.? == .f16 or v_v.dtype.? == .f32)) return CompileError.InvalidArgument;
            if (pos_v.dtype.? != .i32 or end_v.dtype.? != .i32) return CompileError.InvalidArgument;

            var q_tid: TensorId = try ensureAnyTensor(ctx, q_id);
            var k_tid: TensorId = try ensureAnyTensor(ctx, k_id);
            var v_tid: TensorId = try ensureAnyTensor(ctx, v_id);

            // Ensure full D vectors in one tile (required by current cached kernel row access).
            {
                const q_cur: *const TiledTensor = try mgr.getConst(q_tid);
                if (@as(usize, q_cur.rank) != q_v.shape.len) return CompileError.InvalidArgument;
                var q_want_buf: [MAX_RANK]usize = undefined;
                const q_want: []usize = q_want_buf[0..q_v.shape.len];
                var d: usize = 0;
                while (d < q_want.len) : (d += 1) q_want[d] = q_cur.tile_shape[d];
                q_want[q_want.len - 1] = q_v.shape[q_v.shape.len - 1];
                q_tid = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, q_id, q_v.dtype.?, q_v.shape, q_want);
            }
            {
                const k_cur: *const TiledTensor = try mgr.getConst(k_tid);
                if (@as(usize, k_cur.rank) != k_v.shape.len) return CompileError.InvalidArgument;
                var k_want_buf: [MAX_RANK]usize = undefined;
                const k_want: []usize = k_want_buf[0..k_v.shape.len];
                var d: usize = 0;
                while (d < k_want.len) : (d += 1) k_want[d] = k_cur.tile_shape[d];
                k_want[k_want.len - 1] = k_v.shape[k_v.shape.len - 1];
                k_tid = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, k_id, k_v.dtype.?, k_v.shape, k_want);
            }
            {
                const v_cur: *const TiledTensor = try mgr.getConst(v_tid);
                if (@as(usize, v_cur.rank) != v_v.shape.len) return CompileError.InvalidArgument;
                var v_want_buf: [MAX_RANK]usize = undefined;
                const v_want: []usize = v_want_buf[0..v_v.shape.len];
                var d: usize = 0;
                while (d < v_want.len) : (d += 1) v_want[d] = v_cur.tile_shape[d];
                v_want[v_want.len - 1] = v_v.shape[v_v.shape.len - 1];
                v_tid = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, v_id, v_v.dtype.?, v_v.shape, v_want);
            }

            const q_t: *const TiledTensor = try mgr.getConst(q_tid);
            const v_t: *const TiledTensor = try mgr.getConst(v_tid);

            const out_tile: [4]usize = .{
                q_t.tile_shape[0],
                q_t.tile_shape[1],
                q_t.tile_shape[2],
                v_t.shape[3],
            };
            const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile[0..]);

            var pos_tid: TensorId = try ensureAnyTensor(ctx, pos_id);
            {
                const pos_cur: *const TiledTensor = try mgr.getConst(pos_tid);
                if (pos_cur.rank != 2) return CompileError.InvalidArgument;
                if (pos_cur.tile_shape[0] != q_t.tile_shape[0] or pos_cur.tile_shape[1] != q_t.tile_shape[1]) {
                    const pos_want: [2]usize = .{ q_t.tile_shape[0], q_t.tile_shape[1] };
                    pos_tid = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, pos_id, pos_v.dtype.?, pos_v.shape, pos_want[0..]);
                }
            }

            const end_tid: TensorId = try ensureAnyTensor(ctx, end_id);

            try appendStepChecked(allocator, mgr, policy, steps, .{ .MultiHeadAttentionCachedTiled = .{
                .out = out_tid,
                .q = q_tid,
                .k_cache = k_tid,
                .v_cache = v_tid,
                .positions = pos_tid,
                .end_index = end_tid,
                .scale = attn.scale,
                .causal = attn.causal,
                .sliding_window = attn.sliding_window,
                .attn_logits_soft_cap = attn.attn_logits_soft_cap,
            } });
        },

        .ArgMax => |am| {
            const a_id: usize = @intCast(node.inputs[0]);
            const a_v = graph.values.items[a_id];
            const rank: usize = a_v.shape.len;
            const axis: usize = try normalizeAxis(am.axis, rank);
            if (axis != rank - 1) return CompileError.InvalidArgument;  // v1: last axis
            if (out_dt != .i32) return CompileError.InvalidArgument;

            // Force a single tile on input and output (v1 ArgMax contract).
            var in_tile_buf: [MAX_RANK]usize = undefined;
            const in_tile: []usize = in_tile_buf[0..rank];
            var d: usize = 0;
            while (d < rank) : (d += 1) in_tile[d] = a_v.shape[d];
            const a_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, a_id, a_v.dtype.?, a_v.shape, in_tile);

            var out_tile_buf: [MAX_RANK]usize = undefined;
            const out_tile: []usize = out_tile_buf[0..out_shape.len];
            d = 0;
            while (d < out_shape.len) : (d += 1) out_tile[d] = out_shape[d];
            const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile);
            try appendStepChecked(allocator, mgr, policy, steps, .{ .ArgMax = .{ .out = out_tid, .a = a_tid, .axis = axis } });
        },

        .ScatterRow => {
            const buf_id: usize = @intCast(node.inputs[0]);
            const idx_id: usize = @intCast(node.inputs[1]);
            const src_id: usize = @intCast(node.inputs[2]);
            const buf_v = graph.values.items[buf_id];
            const idx_v = graph.values.items[idx_id];
            const src_v = graph.values.items[src_id];
            var buf_tile_buf: [MAX_RANK]usize = undefined;
            const buf_tile: []usize = buf_tile_buf[0..buf_v.shape.len];
            var d: usize = 0;
            while (d < buf_v.shape.len) : (d += 1) buf_tile[d] = buf_v.shape[d];
            const buf_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, buf_id, buf_v.dtype.?, buf_v.shape, buf_tile);
            const idx_tile: [1]usize = .{1};
            const idx_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, idx_id, idx_v.dtype.?, idx_v.shape, idx_tile[0..]);
            var src_tile_buf: [MAX_RANK]usize = undefined;
            const src_tile: []usize = src_tile_buf[0..src_v.shape.len];
            d = 0;
            while (d < src_v.shape.len) : (d += 1) src_tile[d] = src_v.shape[d];
            const src_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, src_id, src_v.dtype.?, src_v.shape, src_tile);
            // In-place: output aliases buf storage.
            ctx.value_tensor[out_idx] = buf_tid;
            ctx.value_has_tensor[out_idx] = true;
            try appendStepChecked(allocator, mgr, policy, steps, .{ .ScatterRow = .{ .buf = buf_tid, .idx = idx_tid, .src = src_tid } });
        },

        .Reduce => |rr| {
            const a_id: usize = @intCast(node.inputs[0]);
            const a_tid: TensorId = try ensureAnyTensor(ctx, a_id);
            const a_v = graph.values.items[a_id];

            if (rr.axis) |axis_raw| {
                const rank: usize = a_v.shape.len;
                const axis: usize = try normalizeAxis(axis_raw, rank);

                var out_tile_buf: [MAX_RANK]usize = undefined;
                const out_tile: []usize = out_tile_buf[0..out_shape.len];

                if (out_shape.len == 1) {
                    const t1: [1]usize = plan_mod.chooseTileShape1D(policy, out_shape[0]);
                    out_tile[0] = t1[0];
                } else {
                    try fillTileShapeDefault(policy, out_dt, out_shape, out_tile);
                }

                const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile);
                try appendStepChecked(allocator, mgr, policy, steps, .{ .ReduceAxis = .{ .op = rr.op, .out = out_tid, .a = a_tid, .axis = axis } });
            } else {
                const out_tile: [1]usize = .{1};
                const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile[0..]);
                try appendStepChecked(allocator, mgr, policy, steps, .{ .ReduceAll = .{ .op = rr.op, .out = out_tid, .a = a_tid } });
            }
        },

        .Concat => |cc| {
            const axis: usize = try normalizeAxis(cc.axis, out_shape.len);

            var out_tile_buf: [MAX_RANK]usize = undefined;
            const out_tile: []usize = out_tile_buf[0..out_shape.len];
            try fillTileShapeDefault(policy, out_dt, out_shape, out_tile);
            const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile);

            var in_ids: [16]TensorId = .{0} ** 16;
            if (node.inputs.len > in_ids.len) return CompileError.InvalidArgument;

            var i: usize = 0;
            while (i < node.inputs.len) : (i += 1) {
                const vidx: usize = @intCast(node.inputs[i]);
                const in_v = graph.values.items[vidx];
                if (in_v.dtype.?.info().is_quantized) return CompileError.InvalidArgument;
                in_ids[i] = try ensureAnyTensor(ctx, vidx);
            }

            try appendStepChecked(allocator, mgr, policy, steps, .{ .ConcatScalar = .{ .out = out_tid, .axis = axis, .input_count = @intCast(node.inputs.len), .inputs = in_ids } });
        },

        .RFFT => {
            const x_id: usize = @intCast(node.inputs[0]);

            var out_tile_buf: [MAX_RANK]usize = undefined;
            const out_tile: []usize = out_tile_buf[0..out_shape.len];
            try fillTileShapeSingleOnGpu(policy, out_dt, out_shape, out_tile);
            const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile);

            const x_tid: TensorId = try ensureAnyTensor(ctx, x_id);

            const n_fft: usize = out_shape[out_shape.len - 1] - 2;
            try appendStepChecked(allocator, mgr, policy, steps, .{ .RFFT = .{
                .out = out_tid,
                .x = x_tid,
                .n_fft = n_fft,
            } });
        },

        .STFT => |st| {
            const signal_id: usize = @intCast(node.inputs[0]);
            const window_id: usize = @intCast(node.inputs[1]);

            var out_tile_buf: [MAX_RANK]usize = undefined;
            const out_tile: []usize = out_tile_buf[0..out_shape.len];
            try fillTileShapeSingleOnGpu(policy, out_dt, out_shape, out_tile);
            const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile);

            const signal_tid: TensorId = try ensureAnyTensor(ctx, signal_id);
            const window_tid: TensorId = try ensureAnyTensor(ctx, window_id);

            try appendStepChecked(allocator, mgr, policy, steps, .{ .STFT = .{
                .out = out_tid,
                .signal = signal_tid,
                .window = window_tid,
                .n_fft = st.n_fft,
                .hop_length = st.hop_length,
                .center = st.center,
                .num_frames = out_shape[1],
            } });
        },

        .LSTMCell => |lc| {
            const x_id: usize = @intCast(node.inputs[0]);
            const h_id: usize = @intCast(node.inputs[1]);
            const c_id: usize = @intCast(node.inputs[2]);
            const wih_id: usize = @intCast(node.inputs[3]);
            const whh_id: usize = @intCast(node.inputs[4]);

            var out_tile_buf: [MAX_RANK]usize = undefined;
            const out_tile: []usize = out_tile_buf[0..out_shape.len];
            try fillTileShapeSingleOnGpu(policy, out_dt, out_shape, out_tile);
            const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile);

            const x_tid: TensorId = try ensureAnyTensor(ctx, x_id);
            const h_tid: TensorId = try ensureAnyTensor(ctx, h_id);
            const c_tid: TensorId = try ensureAnyTensor(ctx, c_id);
            const wih_tid: TensorId = try ensureAnyTensor(ctx, wih_id);
            const whh_tid: TensorId = try ensureAnyTensor(ctx, whh_id);

            var b_ih_tid: ?TensorId = null;
            var b_hh_tid: ?TensorId = null;
            if (lc.has_bias) {
                const bih_id: usize = @intCast(node.inputs[5]);
                const bhh_id: usize = @intCast(node.inputs[6]);
                b_ih_tid = try ensureAnyTensor(ctx, bih_id);
                b_hh_tid = try ensureAnyTensor(ctx, bhh_id);
            }

            try appendStepChecked(allocator, mgr, policy, steps, .{ .LSTMCellFused = .{
                .out_state = out_tid,
                .x = x_tid,
                .h_prev = h_tid,
                .c_prev = c_tid,
                .w_ih = wih_tid,
                .w_hh = whh_tid,
                .b_ih = b_ih_tid,
                .b_hh = b_hh_tid,
            } });
        },

        .GatherRows => {
            const table_id: usize = @intCast(node.inputs[0]);
            const indices_id: usize = @intCast(node.inputs[1]);

            const table_v = graph.values.items[table_id];
            const indices_v = graph.values.items[indices_id];

            if (table_v.shape.len != 2 or indices_v.shape.len != 2) return CompileError.InvalidArgument;
            if (indices_v.dtype.? != .i32) return CompileError.InvalidArgument;
            const table_dtype = table_v.dtype.?;
            switch (table_dtype) {
                .f16, .f32, .q8_0 => {},
                else => return CompileError.InvalidArgument,
            }
            if (out_shape.len != 3) return CompileError.InvalidArgument;

            const b: usize = indices_v.shape[0];
            const l: usize = indices_v.shape[1];
            const v_rows: usize = table_v.shape[0];
            const d_embed: usize = table_v.shape[1];

            // Output tiling: tile B as size-1 slices, tile L modestly, keep D contiguous.
            var out_tile_buf: [MAX_RANK]usize = undefined;
            const out_tile: []usize = out_tile_buf[0..3];
            out_tile[0] = 1;
            out_tile[1] = plan_mod.chooseTileShape1D(policy, l)[0];
            out_tile[2] = d_embed;
            const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile);

            // Indices: force single tile (for cheap indexed access).
            const idx_tile: [2]usize = .{ b, l };
            const indices_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, indices_id, indices_v.dtype.?, indices_v.shape, idx_tile[0..]);

            // Table tiling:
            // - Scalar tables: retile only if D isn't already within one tile (cheap on hit).
            // - q8_0 tables: layout invariant is `quant_axis == 1` and `tile_shape[1] == D`.
            //   `TiledTensor` retiling for quant tensors isn't implemented here, so the bound
            //   storage must already satisfy the invariant. The compile-step validator
            //   double-checks below.
            var table_tid: TensorId = try ensureAnyTensor(ctx, table_id);
            const table_cur: *const TiledTensor = mgr.getConst(table_tid) catch return CompileError.InvalidArgument;
            if (table_dtype == .q8_0) {
                if (table_cur.quant_axis != 1) return CompileError.InvalidArgument;
                if (table_cur.tile_shape[1] != d_embed) return CompileError.InvalidArgument;
            } else if (table_cur.tile_counts[1] != 1) {
                const tv: usize = plan_mod.chooseTileShape1D(policy, v_rows)[0];
                const table_tile: [2]usize = .{ tv, d_embed };
                table_tid = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, table_id, table_dtype, table_v.shape, table_tile[0..]);
            }

            try appendStepChecked(allocator, mgr, policy, steps, .{ .GatherRowsTiled = .{ .out = out_tid, .table = table_tid, .indices = indices_tid } });
        },

        .RoPE1D => |rp| {
            const x_id: usize = @intCast(node.inputs[0]);
            const pos_id: usize = @intCast(node.inputs[1]);

            const x_v = graph.values.items[x_id];
            const pos_v = graph.values.items[pos_id];

            if (x_v.shape.len != 4 or pos_v.shape.len != 2) return CompileError.InvalidArgument;
            if (!(x_v.dtype.? == .f16 or x_v.dtype.? == .f32)) return CompileError.InvalidArgument;
            if (pos_v.dtype.? != .i32) return CompileError.InvalidArgument;

            // RoPE v1 requires full head-dim tiles on input.
            const x_existing_tid: TensorId = try ensureAnyTensor(ctx, x_id);
            const x_existing: *const TiledTensor = mgr.getConst(x_existing_tid) catch return CompileError.InvalidArgument;
            if (x_existing.rank != 4) return CompileError.InvalidArgument;
            if (x_existing.tile_counts[3] != 1) return CompileError.InvalidArgument;

            var out_tile_buf: [MAX_RANK]usize = undefined;
            const out_tile: []usize = out_tile_buf[0..4];
            var d: usize = 0;
            while (d < 4) : (d += 1) {
                out_tile[d] = x_existing.tile_shape[d];
            }
            const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile);

            const x_tid: TensorId = try ensureTilingScalarMaybeRetile(
                allocator,
                steps,
                mgr,
                policy,
                ctx,
                x_id,
                x_v.dtype.?,
                x_v.shape,
                out_tile,
            );

            // Align positions tiling to [B, L] tile grid of x/out.
            const pos_tile: [2]usize = .{ out_tile[0], out_tile[1] };
            const pos_tid: TensorId = try ensureTilingScalarMaybeRetile(
                allocator,
                steps,
                mgr,
                policy,
                ctx,
                pos_id,
                pos_v.dtype.?,
                pos_v.shape,
                pos_tile[0..],
            );

            try appendStepChecked(allocator, mgr, policy, steps, .{ .RoPE1DTiled = .{
                .out = out_tid,
                .x = x_tid,
                .positions = pos_tid,
                .base_frequency = rp.base_frequency,
                .scale_factor = rp.scale_factor,
                .rope_proportion = rp.rope_proportion,
            } });
        },

        .SequenceAppend => {
            const cache_id: usize = @intCast(node.inputs[0]);
            const new_kv_id: usize = @intCast(node.inputs[1]);
            const end_idx_id: usize = @intCast(node.inputs[2]);

            const cache_v = graph.values.items[cache_id];
            const new_kv_v = graph.values.items[new_kv_id];

            // Coerce both cache and new_kv to have the full head_dim in a single
            // tile (the kernel does one memcpy per (b,h,t) row). Other axes keep
            // whatever tiling they came in with i.e the kernel handles arbitrary
            // tile_counts on batch/heads/time via integer division.
            var cache_tid: TensorId = try ensureAnyTensor(ctx, cache_id);
            { // not sure if necessary
                const cache_cur: *const TiledTensor = try mgr.getConst(cache_tid);
                if (@as(usize, cache_cur.rank) != cache_v.shape.len) return CompileError.InvalidArgument;
                var want_buf: [MAX_RANK]usize = undefined;
                const want: []usize = want_buf[0..cache_v.shape.len];
                var d: usize = 0;
                while (d < want.len) : (d += 1) want[d] = cache_cur.tile_shape[d];
                want[want.len - 1] = cache_v.shape[cache_v.shape.len - 1];
                cache_tid = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, cache_id, cache_v.dtype.?, cache_v.shape, want);
            }

            var new_kv_tid: TensorId = try ensureAnyTensor(ctx, new_kv_id);
            {
                const nkv_cur: *const TiledTensor = try mgr.getConst(new_kv_tid);
                if (@as(usize, nkv_cur.rank) != new_kv_v.shape.len) return CompileError.InvalidArgument;
                var want_buf: [MAX_RANK]usize = undefined;
                const want: []usize = want_buf[0..new_kv_v.shape.len];
                var d: usize = 0;
                while (d < want.len) : (d += 1) want[d] = nkv_cur.tile_shape[d];
                want[want.len - 1] = new_kv_v.shape[new_kv_v.shape.len - 1];
                new_kv_tid = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, new_kv_id, new_kv_v.dtype.?, new_kv_v.shape, want);
            }

            const end_idx_tid: TensorId = try ensureAnyTensor(ctx, end_idx_id);

            // In-place semantics: output aliases (post-retile) cache storage.
            ctx.value_tensor[out_idx] = cache_tid;
            ctx.value_has_tensor[out_idx] = true;

            try appendStepChecked(allocator, mgr, policy, steps, .{ .SequenceAppendTiled = .{
                .cache = cache_tid,
                .new_kv = new_kv_tid,
                .end_index = end_idx_tid,
            } });
        },

        .If => |iff| {
            const cond_id: usize = @intCast(node.inputs[0]);
            const then_value_id: usize = @intCast(node.inputs[1]);
            const else_value_id: usize = @intCast(node.inputs[2]);

            const then_region: graph_mod.Region = graph.regions.items[@intCast(iff.then_region)];
            const else_region: graph_mod.Region = graph.regions.items[@intCast(iff.else_region)];
            const then_block: executable.BlockId = try lowerRegionBlock(allocator, graph, then_region, mgr, policy, ctx, blocks, null);
            const else_block: executable.BlockId = try lowerRegionBlock(allocator, graph, else_region, mgr, policy, ctx, blocks, null);

            const cond_tid: TensorId = try ensureAnyTensor(ctx, cond_id);
            const then_tid: TensorId = try ensureAnyTensor(ctx, then_value_id);
            const else_tid: TensorId = try ensureAnyTensor(ctx, else_value_id);
            const then_t: *const TiledTensor = try mgr.getConst(then_tid);
            const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, then_t.tile_shape);

            var outputs_arr: [executable.MAX_CONTROL_OUTPUTS]TensorId = .{0} ** executable.MAX_CONTROL_OUTPUTS;
            var then_arr: [executable.MAX_CONTROL_OUTPUTS]TensorId = .{0} ** executable.MAX_CONTROL_OUTPUTS;
            var else_arr: [executable.MAX_CONTROL_OUTPUTS]TensorId = .{0} ** executable.MAX_CONTROL_OUTPUTS;
            outputs_arr[0] = out_tid;
            then_arr[0] = then_tid;
            else_arr[0] = else_tid;
            try appendStepChecked(allocator, mgr, policy, steps, .{ .If = .{
                .cond = cond_tid,
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

            // Gather carry-in tensors before lowering the body so it can reconcile
            // each carried output's tiling to the matching carry buffer.
            var carried_arr: [executable.MAX_LOOP_CARRIED]TensorId = .{0} ** executable.MAX_LOOP_CARRIED;
            var i: usize = 0;
            while (i < n) : (i += 1) carried_arr[i] = try ensureAnyTensor(ctx, @intCast(node.inputs[i]));

            const body_block: executable.BlockId = try lowerRegionBlock(allocator, graph, body_region, mgr, policy, ctx, blocks, carried_arr[0..n]);

            var body_arr: [executable.MAX_LOOP_CARRIED]TensorId = .{0} ** executable.MAX_LOOP_CARRIED;
            i = 0;
            while (i < n) : (i += 1) {
                body_arr[i] = try ensureAnyTensor(ctx, @intCast(body_region.outputs[i]));
                // After the loop the carried tensors hold the final state; map
                // each loop output (primary + extras) onto its carry buffer.
                const out_value: usize = if (i == 0) out_idx else @intCast(node.extra_outputs[i - 1]);
                ctx.value_tensor[out_value] = carried_arr[i];
                ctx.value_has_tensor[out_value] = true;
            }

            const cond_tid: ?TensorId = if (lp.cond_carry) |ci| carried_arr[ci] else null;
            try appendStepChecked(allocator, mgr, policy, steps, .{ .Loop = .{
                .trip_count = null,
                .static_max_trip_count = lp.static_max_trip_count,
                .cond = cond_tid,
                .check_before = lp.check_before,
                .body_block = body_block,
                .carried_count = @intCast(n),
                .carried = carried_arr,
                .body_carried_outputs = body_arr,
            } });
        },

        .Cast => |ct| {
            const x_id: usize = @intCast(node.inputs[0]);
            const x_v = graph.values.items[x_id];
            var tile_buf: [MAX_RANK]usize = undefined;
            const tile: []usize = tile_buf[0..out_shape.len];

            // Inherit input tiling when available — Cast is shape/layout-preserving.
            if (ctx.value_has_tensor[x_id]) {
                const x_t: *const TiledTensor = try mgr.getConst(ctx.value_tensor[x_id]);
                if (x_t.tile_shape.len == out_shape.len) {
                    @memcpy(tile, x_t.tile_shape);
                } else {
                    try fillTileShapeDefault(policy, out_dt, out_shape, tile);
                }
            } else {
                try fillTileShapeDefault(policy, out_dt, out_shape, tile);
            }

            const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, tile);
            const x_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, x_id, x_v.dtype.?, x_v.shape, tile);
            try appendStepChecked(allocator, mgr, policy, steps, .{ .CastTiled = .{
                .out = out_tid,
                .x = x_tid,
                .to_dtype = ct.to_dtype,
            } });
        },

        .MatMulNT => |mm| {
            const a_id: usize = @intCast(node.inputs[0]);
            const b_id: usize = @intCast(node.inputs[1]);
            const a_v = graph.values.items[a_id];
            const b_v = graph.values.items[b_id];
            const rank: usize = a_v.shape.len;
            if (b_v.shape.len != 2) return CompileError.InvalidArgument;

            // B keeps its `[N, K]` `quant_axis == 1` layout with `tile_shape[1] == K`
            // (per-row contiguous blocks). B's N-axis is typically tiled into row
            // chunks by the default quant-embedding tiler.
            const b_tid: TensorId = try ensureAnyTensor(ctx, b_id);
            const b_cur: *const TiledTensor = try mgr.getConst(b_tid);
            if (b_cur.dtype.info().is_quantized and b_cur.quant_axis != 1) return CompileError.InvalidArgument;
            if (b_cur.tile_shape[1] != b_cur.shape[1]) return CompileError.InvalidArgument;

            // Tiling invariant for the v1 kernel: A occupies a single tile spanning
            // `[...leading, K]`, and C is tiled on its trailing N axis with the *same*
            // N-chunk size as B's axis-0 tiling. That 1:1 alignment lets the kernel
            // walk `tile_index = 0..tile_count` on B and C in lockstep without random
            // access across B tiles.
            var c_tile_buf: [MAX_RANK]usize = undefined;
            const c_tile: []usize = c_tile_buf[0..rank];
            var d: usize = 0;
            while (d + 1 < rank) : (d += 1) c_tile[d] = out_shape[d];
            c_tile[rank - 1] = b_cur.tile_shape[0];

            const c_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, c_tile);

            var a_tile_buf: [MAX_RANK]usize = undefined;
            const a_tile: []usize = a_tile_buf[0..rank];
            d = 0;
            while (d + 1 < rank) : (d += 1) a_tile[d] = a_v.shape[d];
            a_tile[rank - 1] = a_v.shape[rank - 1];
            const a_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, a_id, a_v.dtype.?, a_v.shape, a_tile);

            try appendStepChecked(allocator, mgr, policy, steps, .{ .MatMulNTTiled = .{
                .c = c_tid,
                .a = a_tid,
                .b = b_tid,
                .alpha = mm.alpha,
                .beta = mm.beta,
            } });
        },

        .Copy => {
            const a_id: usize = @intCast(node.inputs[0]);
            const a_v = graph.values.items[a_id];
            var tile_buf: [MAX_RANK]usize = undefined;
            const tile: []usize = tile_buf[0..out_shape.len];
            try fillTileShapeDefault(policy, out_dt, out_shape, tile);
            const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, tile);
            const a_tid: TensorId = try ensureTilingMaybeRetile(allocator, steps, mgr, policy, ctx, a_id, a_v.dtype.?, a_v.shape, tile);
            try appendStepChecked(allocator, mgr, policy, steps, .{ .CopyTiled = .{ .dst = out_tid, .src = a_tid } });
        },

        .ViewReshape => {
            const a_id: usize = @intCast(node.inputs[0]);
            const a_v = graph.values.items[a_id];
            var out_tile_buf: [MAX_RANK]usize = undefined;
            const out_tile: []usize = out_tile_buf[0..out_shape.len];
            try fillTileShapeDefault(policy, out_dt, out_shape, out_tile);
            const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile);
            const a_tid: TensorId = try ensureAnyTensor(ctx, a_id);
            // Materialize reshape (scalar-only for now).
            if (a_v.dtype.?.info().is_quantized) return CompileError.InvalidArgument;
            try appendStepChecked(allocator, mgr, policy, steps, .{ .ReshapeScalar = .{ .dst = out_tid, .src = a_tid } });
        },

        .ViewSqueeze => {
            const a_id: usize = @intCast(node.inputs[0]);
            const a_v = graph.values.items[a_id];
            var out_tile_buf: [MAX_RANK]usize = undefined;
            const out_tile: []usize = out_tile_buf[0..out_shape.len];
            try fillTileShapeDefault(policy, out_dt, out_shape, out_tile);
            const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile);
            const a_tid: TensorId = try ensureAnyTensor(ctx, a_id);
            if (a_v.dtype.?.info().is_quantized) return CompileError.InvalidArgument;
            try appendStepChecked(allocator, mgr, policy, steps, .{ .ReshapeScalar = .{ .dst = out_tid, .src = a_tid } });
        },

        .ViewUnsqueeze => {
            const a_id: usize = @intCast(node.inputs[0]);
            const a_v = graph.values.items[a_id];
            var out_tile_buf: [MAX_RANK]usize = undefined;
            const out_tile: []usize = out_tile_buf[0..out_shape.len];
            try fillTileShapeDefault(policy, out_dt, out_shape, out_tile);
            const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile);
            const a_tid: TensorId = try ensureAnyTensor(ctx, a_id);
            if (a_v.dtype.?.info().is_quantized) return CompileError.InvalidArgument;
            try appendStepChecked(allocator, mgr, policy, steps, .{ .ReshapeScalar = .{ .dst = out_tid, .src = a_tid } });
        },

        .ViewTranspose2D => {
            const a_id: usize = @intCast(node.inputs[0]);
            const a_v = graph.values.items[a_id];
            var out_tile_buf: [MAX_RANK]usize = undefined;
            const out_tile: []usize = out_tile_buf[0..out_shape.len];
            try fillTileShapeDefault(policy, out_dt, out_shape, out_tile);
            const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile);
            const a_tid: TensorId = try ensureAnyTensor(ctx, a_id);
            if (a_v.dtype.?.info().is_quantized) return CompileError.InvalidArgument;
            try appendStepChecked(allocator, mgr, policy, steps, .{ .Transpose2DScalar = .{ .dst = out_tid, .src = a_tid } });
        },

        .ViewSliceND => |sl| {
            const a_id: usize = @intCast(node.inputs[0]);
            const a_v = graph.values.items[a_id];
            var out_tile_buf: [MAX_RANK]usize = undefined;
            const out_tile: []usize = out_tile_buf[0..out_shape.len];
            try fillTileShapeDefault(policy, out_dt, out_shape, out_tile);
            const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile);
            const a_tid: TensorId = try ensureAnyTensor(ctx, a_id);
            if (a_v.dtype.?.info().is_quantized) return CompileError.InvalidArgument;

            var starts_buf: [MAX_RANK]usize = .{0} ** MAX_RANK;
            if (sl.starts.len == 0 or sl.starts.len > MAX_RANK) return CompileError.InvalidArgument;
            const rank: usize = sl.starts.len;
            var d: usize = 0;
            while (d < rank) : (d += 1) {
                starts_buf[d] = sl.starts[d];
            }

            try appendStepChecked(allocator, mgr, policy, steps, .{ .SliceNDScalar = .{ .dst = out_tid, .src = a_tid, .rank = @intCast(rank), .starts = starts_buf } });
        },
    }
}

fn ensureTilingMaybeRetile(
    allocator: std.mem.Allocator,
    steps: *std.ArrayList(Step),
    mgr: *StorageManager,
    policy: plan_mod.TilePolicy,
    ctx: anytype,
    value_index: usize,
    dtype: types.DType,
    shape: []const usize,
    want_tile: []const usize,
) CompileError!TensorId {
    // Quant tensors can be re-tiled by raw block-copy only if shapes match.
    // We allow it here (it is still a scalar/quant packed view per tile), but we only have a scalar retile kernel.
    // For now, require quant tensors to already have the desired tiling.
    if (dtype.info().is_quantized) {
        const tid: TensorId = try ensureAnyTensor(ctx, value_index);
        const t: *const TiledTensor = try mgr.getConst(tid);
        if (t.tile_shape.len != want_tile.len) return CompileError.InvalidArgument;
        var i: usize = 0;
        while (i < want_tile.len) : (i += 1) {
            if (t.tile_shape[i] != want_tile[i]) return CompileError.InvalidArgument;
        }
        return tid;
    }
    return ensureTilingScalarMaybeRetile(allocator, steps, mgr, policy, ctx, value_index, dtype, shape, want_tile);
}

fn ensureTilingScalarMaybeRetile(
    allocator: std.mem.Allocator,
    steps: *std.ArrayList(Step),
    mgr: *StorageManager,
    policy: plan_mod.TilePolicy,
    ctx: anytype,
    value_index: usize,
    dtype: types.DType,
    shape: []const usize,
    want_tile: []const usize,
) CompileError!TensorId {
    const cur: TensorId = try ensureAnyTensor(ctx, value_index);
    const cur_t: *const TiledTensor = try mgr.getConst(cur);

    if (cur_t.tile_shape.len != want_tile.len) return CompileError.InvalidArgument;
    var i: usize = 0;
    while (i < want_tile.len) : (i += 1) {
        if (cur_t.tile_shape[i] != want_tile[i]) break;
    }
    if (i == want_tile.len) return cur;

    if (shape.len > 2) {
        // Performance guard: for large rank>2 tensors, only allow batch retile
        // when the last two dims' tiling is unchanged and the batch tile grid is
        // small. For small tensors, retiling is cheap and we allow changing the
        // last two dims to avoid compile-time tiling dead-ends.
        const total_elems: usize = try elemCount(shape);
        const rank: usize = shape.len;

        // Even for rank>2 tensors, we sometimes *must* retile across the last
        // dims to bridge between ops with different canonical tilings (common in
        // matmul chains). Bound this by a dedicated threshold so we don't have to
        // over-inflate `small_tensor_threshold` (which would also change default
        // storage tiling for many tensors).
        const allow_last2d_change: bool = total_elems <= policy.retile_last2d_change_max_elems;

        // For very large tensors, changing the last two dims can turn a cheap
        // retile into a massive element-wise shuffle *and* increases peak memory
        // use (new backing buffer). Reject unless the tensor is within the policy
        // bound above.
        if (!allow_last2d_change and total_elems > policy.small_tensor_threshold) {
            if (cur_t.tile_shape[rank - 1] != want_tile[rank - 1]) {
                if (traceEnabled()) {
                    std.debug.print(
                        "[aion][compile] retile rejected: large tensor last-dim tile change (dtype={s} shape={any} cur_tile={any} want_tile={any})\n",
                        .{ @tagName(dtype), shape, cur_t.tile_shape, want_tile },
                    );
                }
                return CompileError.InvalidArgument;
            }
            if (cur_t.tile_shape[rank - 2] != want_tile[rank - 2]) {
                if (traceEnabled()) {
                    std.debug.print(
                        "[aion][compile] retile rejected: large tensor 2nd-last-dim tile change (dtype={s} shape={any} cur_tile={any} want_tile={any})\n",
                        .{ @tagName(dtype), shape, cur_t.tile_shape, want_tile },
                    );
                }
                return CompileError.InvalidArgument;
            }
        }

        // Always keep the batch tile grid bounded (even for small tensors), since
        // a tiny per-tile footprint can still imply thousands of tiles.
        var batch_tiles: usize = 1;
        var d: usize = 0;
        while (d + 2 < rank) : (d += 1) {
            batch_tiles = std.math.mul(usize, batch_tiles, cur_t.tile_counts[d]) catch return CompileError.InvalidArgument;
        }
        if (batch_tiles > policy.batch_retile_max_tiles) {
            if (traceEnabled()) {
                std.debug.print(
                    "[aion][compile] retile rejected: batch tile grid too large (tiles={d} max={d}) dtype={s} shape={any} cur_tile={any} want_tile={any}\n",
                    .{ batch_tiles, policy.batch_retile_max_tiles, @tagName(dtype), shape, cur_t.tile_shape, want_tile },
                );
            }
            return CompileError.InvalidArgument;
        }
    }

    // Allocate new tensor with desired tiling and insert a scalar retile copy.
    const new_tid: TensorId = try mgr.createTiledTensor(dtype, shape, want_tile, .{ .tile_alignment = policy.tile_alignment });
    try appendStepChecked(allocator, mgr, policy, steps, .{ .ReTileCopyScalar = .{ .dst = new_tid, .src = cur } });
    ctx.value_tensor[value_index] = new_tid;
    ctx.value_has_tensor[value_index] = true;
    return new_tid;
}
