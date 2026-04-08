const std = @import("std");
const types = @import("../backend/types.zig");
const storage = @import("../storage/storage.zig");
const executable = @import("../runtime/executable.zig");

const graph_mod = @import("graph.zig");
const infer_mod = @import("infer.zig");
const plan_mod = @import("plan.zig");
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
        .f32, .f16 => true,
        else => false,
    };
}

fn validateStep(mgr: *StorageManager, step: Step) CompileError!void {
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
                try compileRequire(out.tile_shape[0] <= 256);
            } else {
                var row_elems: usize = 1;
                var d: usize = 0;
                while (d < rank) : (d += 1) {
                    if (d == axis) continue;
                    row_elems = std.math.mul(usize, row_elems, out.tile_shape[d]) catch return CompileError.InvalidArgument;
                }
                try compileRequire(row_elems <= 256);
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

            // Per-row scratch arrays in exec: rows are leading dims.
            if (rank > norm_rank) {
                const row_elems: usize = try productUsize(out.tile_shape[0..(rank - norm_rank)]);
                try compileRequire(row_elems <= 256);
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

            if (rank > norm_rank) {
                const row_elems: usize = try productUsize(out.tile_shape[0..(rank - norm_rank)]);
                try compileRequire(row_elems <= 256);
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
            try compileRequire(out.tile_shape[rank - 2] <= 256);
            try compileRequire(k.tile_shape[rank - 2] <= 128);
            try compileRequire(out.tile_shape[rank - 1] <= 64);
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
            try compileRequire(out.tile_shape[rank - 2] <= 256);
            try compileRequire(k.tile_shape[rank - 2] <= 128);
            try compileRequire(out.tile_shape[rank - 1] <= 64);
            try compileRequire(q.tile_shape[rank - 1] > 0);

            try compileRequire(s.scale > 0.0);
            try compileRequire(std.math.isFinite(s.scale));
            _ = s.causal;
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

        .ComplexAbsMean => |s| {
            const out: *const TiledTensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const stft: *const TiledTensor = mgr.getConst(s.stft) catch return CompileError.InvalidArgument;

            try compileRequire(isScalarSupported(out.dtype));
            try compileRequire(out.dtype == stft.dtype);
            try compileRequire(!out.dtype.info().is_quantized);

            try compileRequire(stft.rank == 3);
            try compileRequire(out.rank == 2);

            const batch: usize = stft.shape[0];
            const time: usize = stft.shape[1];
            const chans2: usize = stft.shape[2];
            try compileRequire(batch != 0 and time != 0 and chans2 != 0);
            try compileRequire(chans2 % 2 == 0);

            const cutoff: usize = chans2 / 2;
            try compileRequire(s.out_channels >= 1 and s.out_channels <= cutoff);

            try compileRequire(out.shape[0] == batch);
            try compileRequire(out.shape[1] == s.out_channels);
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
    steps: *std.ArrayList(Step),
    step: Step,
) CompileError!void {
    try validateStep(mgr, step);
    steps.append(allocator, step) catch return CompileError.OutOfMemory;
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

pub fn compileGraph(
    allocator: std.mem.Allocator,
    graph: *graph_mod.Graph,
    mgr: *StorageManager,
    policy: plan_mod.TilePolicy,
) CompileError!Program {
    try infer_mod.infer(graph);

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
                const n: usize = b_v.shape[rank - 1];

                const tiles = plan_mod.chooseMatMulTiles(policy, m, n, k, b_v.dtype.?);

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
                const a_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, &steps, mgr, policy, &ctx, a_id, a_v.dtype.?, a_v.shape, a_tile);
                const b_tid: TensorId = try ensureTilingMaybeRetile(allocator, &steps, mgr, policy, &ctx, b_id, b_v.dtype.?, b_v.shape, b_tile);

                try appendStepChecked(allocator, mgr, &steps, .{ .MatMulTiled = .{ .c = c_tid, .a = a_tid, .b = b_tid, .alpha = mm.alpha, .beta = mm.beta } });
            },

            .ElemwiseBinary => |eb| {
                const a_id: usize = @intCast(node.inputs[0]);
                const b_id: usize = @intCast(node.inputs[1]);
                const a_v = graph.values.items[a_id];
                var tile_buf: [MAX_RANK]usize = undefined;
                const tile: []usize = tile_buf[0..out_shape.len];
                try fillTileShapeDefault(policy, out_dt, out_shape, tile);

                const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, tile);
                const a_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, &steps, mgr, policy, &ctx, a_id, a_v.dtype.?, a_v.shape, tile);
                const b_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, &steps, mgr, policy, &ctx, b_id, a_v.dtype.?, a_v.shape, tile);

                try appendStepChecked(allocator, mgr, &steps, .{ .ElemwiseBinaryTiled = .{ .op = eb.op, .out = out_tid, .a = a_tid, .b = b_tid } });
            },

            .BroadcastLastDimBinary => |bb| {
                const a_id: usize = @intCast(node.inputs[0]);
                const b_id: usize = @intCast(node.inputs[1]);
                const a_v = graph.values.items[a_id];
                const b_v = graph.values.items[b_id];

                var tile_out_buf: [MAX_RANK]usize = undefined;
                const tile_out: []usize = tile_out_buf[0..out_shape.len];
                try fillTileShapeDefault(policy, out_dt, out_shape, tile_out);
                const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, tile_out);
                const a_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, &steps, mgr, policy, &ctx, a_id, a_v.dtype.?, a_v.shape, tile_out);

                const last: usize = out_shape.len - 1;
                // b must be rank-1 tiled to match the last-dim tile.
                const b_tile: [1]usize = .{tile_out[last]};
                const b_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, &steps, mgr, policy, &ctx, b_id, b_v.dtype.?, b_v.shape, b_tile[0..]);

                try appendStepChecked(allocator, mgr, &steps, .{ .BroadcastLastDimBinaryTiled = .{ .op = bb.op, .out = out_tid, .a = a_tid, .b = b_tid } });
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
                const a_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, &steps, mgr, policy, &ctx, a_id, a_v.dtype.?, a_v.shape, tile);
                try appendStepChecked(allocator, mgr, &steps, .{ .UnaryTiled = .{ .op = u.op, .out = out_tid, .a = a_tid } });
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
                const a_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, &steps, mgr, policy, &ctx, a_id, a_v.dtype.?, a_v.shape, tile);
                try appendStepChecked(allocator, mgr, &steps, .{ .SoftmaxTiled = .{ .out = out_tid, .a = a_tid, .axis = sm.axis } });
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
                const x_tid: TensorId = try ensureAnyTensor(&ctx, x_id);
                const w_tid: TensorId = try ensureAnyTensor(&ctx, w_id);

                var bias_tid: ?TensorId = null;
                if (node.inputs.len == 3) {
                    const b_id: usize = @intCast(node.inputs[2]);
                    bias_tid = try ensureAnyTensor(&ctx, b_id);
                }

                try appendStepChecked(allocator, mgr, &steps, .{ .Conv1DTiled = .{
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
                const x_tid: TensorId = try ensureAnyTensor(&ctx, x_id);
                const w_tid: TensorId = try ensureAnyTensor(&ctx, w_id);

                var bias_tid: ?TensorId = null;
                if (node.inputs.len == 3) {
                    const b_id: usize = @intCast(node.inputs[2]);
                    bias_tid = try ensureAnyTensor(&ctx, b_id);
                }

                try appendStepChecked(allocator, mgr, &steps, .{ .Conv2DTiled = .{
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
                const x_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, &steps, mgr, policy, &ctx, x_id, x_v.dtype.?, x_v.shape, tile);

                var bcast_tile_buf: [MAX_RANK]usize = undefined;
                const bcast_tile: []usize = bcast_tile_buf[0..norm_rank];
                var d: usize = 0;
                while (d < norm_rank) : (d += 1) {
                    bcast_tile[d] = tile[rank - norm_rank + d];
                }
                const gamma_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, &steps, mgr, policy, &ctx, g_id, g_v.dtype.?, g_v.shape, bcast_tile);
                const beta_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, &steps, mgr, policy, &ctx, b_id, b_v.dtype.?, b_v.shape, bcast_tile);

                try appendStepChecked(allocator, mgr, &steps, .{ .LayerNormTiled = .{ .out = out_tid, .x = x_tid, .gamma = gamma_tid, .beta = beta_tid, .eps = ln.eps } });
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
                const x_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, &steps, mgr, policy, &ctx, x_id, x_v.dtype.?, x_v.shape, tile);

                var bcast_tile_buf: [MAX_RANK]usize = undefined;
                const bcast_tile: []usize = bcast_tile_buf[0..norm_rank];
                var d: usize = 0;
                while (d < norm_rank) : (d += 1) {
                    bcast_tile[d] = tile[rank - norm_rank + d];
                }
                const gamma_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, &steps, mgr, policy, &ctx, g_id, g_v.dtype.?, g_v.shape, bcast_tile);
                const beta_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, &steps, mgr, policy, &ctx, b_id, b_v.dtype.?, b_v.shape, bcast_tile);

                try appendStepChecked(allocator, mgr, &steps, .{ .RMSNormTiled = .{ .out = out_tid, .x = x_tid, .gamma = gamma_tid, .beta = beta_tid, .eps = rn.eps } });
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

                const q_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, &steps, mgr, policy, &ctx, q_id, q_v.dtype.?, q_v.shape, q_tile);
                const k_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, &steps, mgr, policy, &ctx, k_id, k_v.dtype.?, k_v.shape, k_tile);
                const v_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, &steps, mgr, policy, &ctx, v_id, v_v.dtype.?, v_v.shape, v_tile);

                try appendStepChecked(allocator, mgr, &steps, .{ .AttentionTiled = .{ .out = out_tid, .q = q_tid, .k = k_tid, .v = v_tid, .scale = attn.scale, .causal = attn.causal } });
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

                const q_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, &steps, mgr, policy, &ctx, q_id, q_v.dtype.?, q_v.shape, q_tile);
                const k_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, &steps, mgr, policy, &ctx, k_id, k_v.dtype.?, k_v.shape, k_tile);
                const v_tid: TensorId = try ensureTilingScalarMaybeRetile(allocator, &steps, mgr, policy, &ctx, v_id, v_v.dtype.?, v_v.shape, v_tile);

                try appendStepChecked(allocator, mgr, &steps, .{ .MultiHeadAttentionTiled = .{ .out = out_tid, .q = q_tid, .k = k_tid, .v = v_tid, .scale = attn.scale, .causal = attn.causal, .heads = attn.heads } });
            },

            .Reduce => |rr| {
                const a_id: usize = @intCast(node.inputs[0]);
                const a_tid: TensorId = try ensureAnyTensor(&ctx, a_id);
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
                    try appendStepChecked(allocator, mgr, &steps, .{ .ReduceAxis = .{ .op = rr.op, .out = out_tid, .a = a_tid, .axis = axis } });
                } else {
                    const out_tile: [1]usize = .{1};
                    const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile[0..]);
                    try appendStepChecked(allocator, mgr, &steps, .{ .ReduceAll = .{ .op = rr.op, .out = out_tid, .a = a_tid } });
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
                    in_ids[i] = try ensureAnyTensor(&ctx, vidx);
                }

                try appendStepChecked(allocator, mgr, &steps, .{ .ConcatScalar = .{ .out = out_tid, .axis = axis, .input_count = @intCast(node.inputs.len), .inputs = in_ids } });
            },

            .ComplexAbsMean => |cm| {
                const stft_id: usize = @intCast(node.inputs[0]);

                var out_tile_buf: [MAX_RANK]usize = undefined;
                const out_tile: []usize = out_tile_buf[0..out_shape.len];
                try fillTileShapeDefault(policy, out_dt, out_shape, out_tile);
                const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile);

                const stft_tid: TensorId = try ensureAnyTensor(&ctx, stft_id);

                try appendStepChecked(allocator, mgr, &steps, .{ .ComplexAbsMean = .{
                    .out = out_tid,
                    .stft = stft_tid,
                    .out_channels = cm.out_channels,
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
                try fillTileShapeDefault(policy, out_dt, out_shape, out_tile);
                const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile);

                const x_tid: TensorId = try ensureAnyTensor(&ctx, x_id);
                const h_tid: TensorId = try ensureAnyTensor(&ctx, h_id);
                const c_tid: TensorId = try ensureAnyTensor(&ctx, c_id);
                const wih_tid: TensorId = try ensureAnyTensor(&ctx, wih_id);
                const whh_tid: TensorId = try ensureAnyTensor(&ctx, whh_id);

                var b_ih_tid: ?TensorId = null;
                var b_hh_tid: ?TensorId = null;
                if (lc.has_bias) {
                    const bih_id: usize = @intCast(node.inputs[5]);
                    const bhh_id: usize = @intCast(node.inputs[6]);
                    b_ih_tid = try ensureAnyTensor(&ctx, bih_id);
                    b_hh_tid = try ensureAnyTensor(&ctx, bhh_id);
                }

                try appendStepChecked(allocator, mgr, &steps, .{ .LSTMCellFused = .{
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

            .Copy => {
                const a_id: usize = @intCast(node.inputs[0]);
                const a_v = graph.values.items[a_id];
                var tile_buf: [MAX_RANK]usize = undefined;
                const tile: []usize = tile_buf[0..out_shape.len];
                try fillTileShapeDefault(policy, out_dt, out_shape, tile);
                const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, tile);
                const a_tid: TensorId = try ensureTilingMaybeRetile(allocator, &steps, mgr, policy, &ctx, a_id, a_v.dtype.?, a_v.shape, tile);
                try appendStepChecked(allocator, mgr, &steps, .{ .CopyTiled = .{ .dst = out_tid, .src = a_tid } });
            },

            .ViewReshape => {
                const a_id: usize = @intCast(node.inputs[0]);
                const a_v = graph.values.items[a_id];
                var out_tile_buf: [MAX_RANK]usize = undefined;
                const out_tile: []usize = out_tile_buf[0..out_shape.len];
                try fillTileShapeDefault(policy, out_dt, out_shape, out_tile);
                const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile);
                const a_tid: TensorId = try ensureAnyTensor(&ctx, a_id);
                // Materialize reshape (scalar-only for now).
                if (a_v.dtype.?.info().is_quantized) return CompileError.InvalidArgument;
                try appendStepChecked(allocator, mgr, &steps, .{ .ReshapeScalar = .{ .dst = out_tid, .src = a_tid } });
            },

            .ViewSqueeze => {
                const a_id: usize = @intCast(node.inputs[0]);
                const a_v = graph.values.items[a_id];
                var out_tile_buf: [MAX_RANK]usize = undefined;
                const out_tile: []usize = out_tile_buf[0..out_shape.len];
                try fillTileShapeDefault(policy, out_dt, out_shape, out_tile);
                const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile);
                const a_tid: TensorId = try ensureAnyTensor(&ctx, a_id);
                if (a_v.dtype.?.info().is_quantized) return CompileError.InvalidArgument;
                try appendStepChecked(allocator, mgr, &steps, .{ .ReshapeScalar = .{ .dst = out_tid, .src = a_tid } });
            },

            .ViewUnsqueeze => {
                const a_id: usize = @intCast(node.inputs[0]);
                const a_v = graph.values.items[a_id];
                var out_tile_buf: [MAX_RANK]usize = undefined;
                const out_tile: []usize = out_tile_buf[0..out_shape.len];
                try fillTileShapeDefault(policy, out_dt, out_shape, out_tile);
                const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile);
                const a_tid: TensorId = try ensureAnyTensor(&ctx, a_id);
                if (a_v.dtype.?.info().is_quantized) return CompileError.InvalidArgument;
                try appendStepChecked(allocator, mgr, &steps, .{ .ReshapeScalar = .{ .dst = out_tid, .src = a_tid } });
            },

            .ViewTranspose2D => {
                const a_id: usize = @intCast(node.inputs[0]);
                const a_v = graph.values.items[a_id];
                var out_tile_buf: [MAX_RANK]usize = undefined;
                const out_tile: []usize = out_tile_buf[0..out_shape.len];
                try fillTileShapeDefault(policy, out_dt, out_shape, out_tile);
                const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile);
                const a_tid: TensorId = try ensureAnyTensor(&ctx, a_id);
                if (a_v.dtype.?.info().is_quantized) return CompileError.InvalidArgument;
                try appendStepChecked(allocator, mgr, &steps, .{ .Transpose2DScalar = .{ .dst = out_tid, .src = a_tid } });
            },

            .ViewSliceND => |sl| {
                const a_id: usize = @intCast(node.inputs[0]);
                const a_v = graph.values.items[a_id];
                var out_tile_buf: [MAX_RANK]usize = undefined;
                const out_tile: []usize = out_tile_buf[0..out_shape.len];
                try fillTileShapeDefault(policy, out_dt, out_shape, out_tile);
                const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile);
                const a_tid: TensorId = try ensureAnyTensor(&ctx, a_id);
                if (a_v.dtype.?.info().is_quantized) return CompileError.InvalidArgument;

                var starts_buf: [MAX_RANK]usize = .{0} ** MAX_RANK;
                if (sl.starts.len == 0 or sl.starts.len > MAX_RANK) return CompileError.InvalidArgument;
                const rank: usize = sl.starts.len;
                var d: usize = 0;
                while (d < rank) : (d += 1) {
                    starts_buf[d] = sl.starts[d];
                }

                try appendStepChecked(allocator, mgr, &steps, .{ .SliceNDScalar = .{ .dst = out_tid, .src = a_tid, .rank = @intCast(rank), .starts = starts_buf } });
            },
        }
    }

    // Collect outputs.
    const out_ids: []TensorId = try allocator.alloc(TensorId, graph.outputs.items.len);
    for (graph.outputs.items, 0..) |vid, i| {
        const idx: usize = @intCast(vid);
        if (!value_has_tensor[idx]) return CompileError.InvalidArgument;
        out_ids[i] = value_tensor[idx];
    }

    const steps_slice: []Step = try steps.toOwnedSlice(allocator);

    return .{ .allocator = allocator, .steps = steps_slice, .outputs = out_ids };
}

fn ensureAnyTensor(ctx: anytype, value_index: usize) CompileError!TensorId {
    if (!ctx.value_has_tensor[value_index]) return CompileError.InvalidArgument;
    return ctx.value_tensor[value_index];
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

        // For large tensors, changing the last two dims can turn a cheap retile into
        // a massive element-wise shuffle; reject it.
        if (total_elems > policy.small_tensor_threshold) {
            if (cur_t.tile_shape[rank - 1] != want_tile[rank - 1]) return CompileError.InvalidArgument;
            if (cur_t.tile_shape[rank - 2] != want_tile[rank - 2]) return CompileError.InvalidArgument;
        }

        // Always keep the batch tile grid bounded (even for small tensors), since
        // a tiny per-tile footprint can still imply thousands of tiles.
        var batch_tiles: usize = 1;
        var d: usize = 0;
        while (d + 2 < rank) : (d += 1) {
            batch_tiles = std.math.mul(usize, batch_tiles, cur_t.tile_counts[d]) catch return CompileError.InvalidArgument;
        }
        if (batch_tiles > policy.batch_retile_max_tiles) return CompileError.InvalidArgument;
    }

    // Allocate new tensor with desired tiling and insert a scalar retile copy.
    const new_tid: TensorId = try mgr.createTiledTensor(dtype, shape, want_tile, .{ .tile_alignment = policy.tile_alignment });
    try appendStepChecked(allocator, mgr, steps, .{ .ReTileCopyScalar = .{ .dst = new_tid, .src = cur } });
    ctx.value_tensor[value_index] = new_tid;
    ctx.value_has_tensor[value_index] = true;
    return new_tid;
}
