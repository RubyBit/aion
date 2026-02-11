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

        .ReduceAll => |s| {
            const out: *const TiledTensor = mgr.getConst(s.out) catch return CompileError.InvalidArgument;
            const a: *const TiledTensor = mgr.getConst(s.a) catch return CompileError.InvalidArgument;
            try compileRequire(out.rank == 1 and out.shape[0] == 1);
            try compileRequire(isScalarSupported(out.dtype));
            try compileRequire(out.dtype == a.dtype);
            try compileRequire(isScalarSupported(a.dtype));
            _ = s.op;
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

        .Slice2DScalar => |s| {
            const dst: *const TiledTensor = mgr.getConst(s.dst) catch return CompileError.InvalidArgument;
            const src: *const TiledTensor = mgr.getConst(s.src) catch return CompileError.InvalidArgument;
            try compileRequire(isScalarSupported(dst.dtype));
            try compileRequire(dst.dtype == src.dtype);
            try compileRequire(dst.rank == 2 and src.rank == 2);
            try compileRequire(s.start0 + dst.shape[0] <= src.shape[0]);
            try compileRequire(s.start1 + dst.shape[1] <= src.shape[1]);
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
    var steps = std.ArrayListUnmanaged(Step){};
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
                try fillTileShapeDefault(policy, out_dt, out_shape, tile);
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
                const a_v = graph.values.items[a_id];
                const out_tile: [1]usize = .{1};
                const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile[0..]);
                // Reduce supports any tiling; no need to retile.
                const a_tid: TensorId = try ensureAnyTensor(&ctx, a_id);
                _ = a_v;
                try appendStepChecked(allocator, mgr, &steps, .{ .ReduceAll = .{ .op = rr.op, .out = out_tid, .a = a_tid } });
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

            .ViewReshape => |_| {
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

            .ViewSlice2D => |sl| {
                const a_id: usize = @intCast(node.inputs[0]);
                const a_v = graph.values.items[a_id];
                var out_tile_buf: [MAX_RANK]usize = undefined;
                const out_tile: []usize = out_tile_buf[0..out_shape.len];
                try fillTileShapeDefault(policy, out_dt, out_shape, out_tile);
                const out_tid: TensorId = try ctx.ensureValueTensor(out_idx, out_dt, out_shape, out_tile);
                const a_tid: TensorId = try ensureAnyTensor(&ctx, a_id);
                if (a_v.dtype.?.info().is_quantized) return CompileError.InvalidArgument;
                try appendStepChecked(allocator, mgr, &steps, .{ .Slice2DScalar = .{ .dst = out_tid, .src = a_tid, .start0 = sl.start0, .start1 = sl.start1 } });
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
    steps: *std.ArrayListUnmanaged(Step),
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
    steps: *std.ArrayListUnmanaged(Step),
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
        // Performance guard: only allow batch retile when M/N/K tiles are unchanged
        // and batch tile count is small.
        const rank: usize = shape.len;
        if (cur_t.tile_shape[rank - 1] != want_tile[rank - 1]) return CompileError.InvalidArgument;
        if (cur_t.tile_shape[rank - 2] != want_tile[rank - 2]) return CompileError.InvalidArgument;

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
