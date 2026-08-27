// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const types = @import("../backend/types.zig");

pub const DType = types.DType;
pub const ElemwiseBinaryOp = types.ElemwiseBinaryOp;
pub const UnaryOp = types.UnaryOp;
pub const ReduceOp = types.ReduceOp;
pub const PadMode = types.PadMode;
const MAX_CONCAT_INPUTS: usize = 16;

pub const ValueId = u32;
pub const NodeId = u32;
pub const RegionId = u32;

/// External binding id (typically a `storage.manager.TensorId`).
///
/// This is kept as a plain integer to avoid importing storage modules here.
pub const ExternalId = u32;

pub const Value = struct {
    /// If null, inference will fill this in.
    dtype: ?DType = null,
    /// Shape owned by the graph arena. Empty => unknown.
    shape: []const usize = &[_]usize{},

    /// Per-axis dimension symbol, or null on axes with a fixed size. Empty means
    /// "no axis is symbolic"; otherwise it has one entry per axis of `shape`.
    ///
    /// A symbol says the size in `shape` is only the size this graph was authored
    /// at — the axis is free, and one compile serves any size on it. Declared once
    /// where a tensor is defined (`Builder.symbolicDim`) and propagated by
    /// inference, so a layer can ask its input which of its axes are free instead
    /// of having to be told. Strings are arena-owned, like `shape`.
    ///
    /// This matters at *authoring* time. At runtime shapes are re-inferred from
    /// the real input shapes, so nothing downstream reads these — except the two
    /// places an author bakes a size into the graph, a view's `new_shape` and a
    /// slice's `lens`, which is what `Builder`'s view dim symbols record for export.
    dim_symbols: []const ?[]const u8 = &.{},

    /// Producing node if any.
    producer: ?NodeId = null,

    /// Optional binding to an externally managed tensor.
    external: ?ExternalId = null,
};

/// Stable op ids shared by graph ops and serialized package node ops.
pub const OpTag = enum(u8) {
    MatMul = 0,
    ElemwiseBinary = 1,
    Unary = 2,
    Softmax = 3,
    Conv1D = 4,
    Conv2D = 5,
    LayerNorm = 6,
    RMSNorm = 7,
    Reduce = 8,
    Concat = 9,
    LSTMCell = 10,
    Copy = 11,
    ViewReshape = 12,
    ViewSqueeze = 13,
    ViewUnsqueeze = 14,
    ViewTranspose2D = 15,
    ViewSliceND = 16,

    /// Rotary positional embedding over 1D positions.
    RoPE1D = 17,

    /// In-place KV cache append (scatter-write into existing cache tensor).
    SequenceAppend = 18,

    /// Grouped-query attention, over a KV cache or over a plain sequence.
    Attention = 19,

    /// Elementwise scalar-dtype cast (shape- and layout-preserving).
    ///
    /// Supported pairs: f32<->f16. Primary use case: bridging the f32 output of
    /// quantized matmul to f16 KV caches before `SequenceAppend`.
    ///
    Cast = 20,

    /// Matmul with the right operand conceptually transposed: C[m,n] = sum_k A[m,k] * B[n,k].
    ///
    /// Unlike the standard `MatMul` (which expects B shaped `[K, N]` with axis-0 blocks
    /// for quantized B), `MatMulNT` expects B shaped `[N, K]` with per-row blocks
    /// (q8_0 `quant_axis == 1`). This is the layout an embedding table already has on
    /// disk, so tied logits can reuse the token embedding without duplicating it.
    ///
    MatMulNT = 21,

    /// Single-output conditional region.
    If = 22,

    /// Single-carried-value loop region.
    Loop = 23,

    /// Real FFT over the last dimension (power-of-two length).
    RFFT = 24,

    /// Short-time Fourier transform (framing + window + real FFT).
    STFT = 25,

    /// Relative-positional (Transformer-XL / Conformer) multi-head self-attention.
    RelPosMHA = 26,

    /// Index of the maximum value along an axis (v1: last axis). Output dtype i32.
    ArgMax = 27,

    /// In-place scatter of one row: buf[idx] = src. Output aliases buf.
    ScatterRow = 28,
    // 29 was GeluMul, retired: a gated activation is `ElemwiseBinary{ .op = .gate }`
    // parameterized by its `UnaryOp`, which covers GEGLU/SwiGLU/GLU/ReGLU instead of
    // one tag for one of them. Ids are stable on disk, so 29 stays unused.
    /// General gather with TensorFlow-style `axis` / `batch_dims` semantics.
    Gather = 30,
    /// One input extent reified as an i32 one-element tensor.
    Dim = 31,
    /// Coordinate tensor with the input's shape, increasing along one axis.
    Iota = 32,
};

pub const Op = union(OpTag) {
    /// out = alpha * (a @ b) + beta * out
    MatMul: struct { alpha: f32 = 1.0, beta: f32 = 0.0 },

    /// `ElemwiseBinaryOp.gate` is NOT graph vocabulary — it is a step-level schedule
    /// that `opt/fuse_steps.zig` produces from a `Unary` feeding a `mul`. A gated
    /// activation is authored as exactly that pair, so no `act`
    /// parameter is carried here or serialized.
    ElemwiseBinary: struct { op: ElemwiseBinaryOp },
    Unary: struct { op: UnaryOp },
    Softmax: struct { axis: i32 },

    /// 1D convolution (NLC-style channel-last).
    ///
    /// Shapes:
    /// - x: [..., l_in, c_in]
    /// - w: [k, c_in/groups, c_out]
    /// - bias (optional): [c_out]
    /// - out: [..., l_out, c_out]
    Conv1D: struct {
        stride: usize = 1,
        dilation: usize = 1,
        pad_left: usize = 0,
        pad_right: usize = 0,
        pad_mode: PadMode = .zero,
        groups: usize = 1,
    },

    /// 2D convolution (NHWC-style channel-last).
    ///
    /// Shapes:
    /// - x: [..., h_in, w_in, c_in]
    /// - w: [k_h, k_w, c_in/groups, c_out]
    /// - bias (optional): [c_out]
    /// - out: [..., h_out, w_out, c_out]
    Conv2D: struct {
        stride_h: usize = 1,
        stride_w: usize = 1,
        dilation_h: usize = 1,
        dilation_w: usize = 1,
        pad_top: usize = 0,
        pad_bottom: usize = 0,
        pad_left: usize = 0,
        pad_right: usize = 0,
        pad_mode: PadMode = .zero,
        groups: usize = 1,
    },

    /// Normalize over the last dimensions (rank>=1; normalized_shape is required).
    /// out = ((x - mean) / sqrt(var + eps)) * gamma + beta
    LayerNorm: struct {
        eps: f32,
        normalized_shape: []const usize,
        /// Free axes: `free_dims[i]`, when non-null, is the index of the dim EXPRESSION
        /// that gives axis `i`'s size at runtime. Empty when the attribute is fully
        /// fixed, which is the common case.
        ///
        /// An expression index, not a symbol index — for a `Builder` graph the two
        /// coincide (one expression per declared symbol, in declaration order), but a
        /// package may point an axis at an arithmetic tree over several symbols.
        ///
        /// A free axis makes the size above a PLACEHOLDER: the authoring size while a
        /// `Builder` graph is being inferred eagerly, and 0 for one parsed from a file
        /// (which records no size for a free axis). `Template.specialize` overwrites it
        /// before inference runs, and a 0 that escaped would fail there rather than
        /// quietly become an empty axis.
        free_dims: []const ?u32 = &.{},
    },

    /// Normalize over the last dimensions (rank>=1; normalized_shape is required).
    /// out = (x / sqrt(mean(x^2) + eps)) * gamma + beta
    RMSNorm: struct {
        eps: f32,
        normalized_shape: []const usize,
        /// Free axes: `free_dims[i]`, when non-null, is the index of the dim EXPRESSION
        /// that gives axis `i`'s size at runtime. Empty when the attribute is fully
        /// fixed, which is the common case.
        ///
        /// An expression index, not a symbol index — for a `Builder` graph the two
        /// coincide (one expression per declared symbol, in declaration order), but a
        /// package may point an axis at an arithmetic tree over several symbols.
        ///
        /// A free axis makes the size above a PLACEHOLDER: the authoring size while a
        /// `Builder` graph is being inferred eagerly, and 0 for one parsed from a file
        /// (which records no size for a free axis). `Template.specialize` overwrites it
        /// before inference runs, and a 0 that escaped would fail there rather than
        /// quietly become an empty axis.
        free_dims: []const ?u32 = &.{},
    },

    Reduce: struct { op: ReduceOp, axis: ?i32 = null },
    Concat: struct { axis: i32 },

    /// Fused single-timestep LSTM cell.
    ///
    /// Inputs (N=rank-2):
    /// - x:      [batch, input_size]
    /// - h_prev: [batch, hidden]
    /// - c_prev: [batch, hidden]
    /// - w_ih:   [input_size, 4*hidden]
    /// - w_hh:   [hidden, 4*hidden]
    /// - b_ih:   [4*hidden] (optional)
    /// - b_hh:   [4*hidden] (optional)
    ///
    /// Output:
    /// - state: [batch, 2*hidden] where state[:,0:h]=h_t and state[:,h:2h]=c_t
    LSTMCell: struct { has_bias: bool },

    Copy: void,

    /// View ops (lowered into materialization steps in v0).
    ViewReshape: struct {
        new_shape: []const usize,
        /// Free axes: `free_dims[i]`, when non-null, is the index of the dim EXPRESSION
        /// that gives axis `i`'s size at runtime. Empty when the attribute is fully
        /// fixed, which is the common case.
        ///
        /// An expression index, not a symbol index — for a `Builder` graph the two
        /// coincide (one expression per declared symbol, in declaration order), but a
        /// package may point an axis at an arithmetic tree over several symbols.
        ///
        /// A free axis makes the size above a PLACEHOLDER: the authoring size while a
        /// `Builder` graph is being inferred eagerly, and 0 for one parsed from a file
        /// (which records no size for a free axis). `Template.specialize` overwrites it
        /// before inference runs, and a 0 that escaped would fail there rather than
        /// quietly become an empty axis.
        free_dims: []const ?u32 = &.{},
    },
    ViewSqueeze: struct { axis: ?i32 = null },
    ViewUnsqueeze: struct { axis: i32 },
    ViewTranspose2D: void,
    ViewSliceND: struct {
        starts: []const usize,
        /// Only `lens` may be free; a slice's START is a position, not an extent.
        lens: []const usize,
        /// Free axes: `free_dims[i]`, when non-null, is the index of the dim EXPRESSION
        /// that gives axis `i`'s size at runtime. Empty when the attribute is fully
        /// fixed, which is the common case.
        ///
        /// An expression index, not a symbol index — for a `Builder` graph the two
        /// coincide (one expression per declared symbol, in declaration order), but a
        /// package may point an axis at an arithmetic tree over several symbols.
        ///
        /// A free axis makes the size above a PLACEHOLDER: the authoring size while a
        /// `Builder` graph is being inferred eagerly, and 0 for one parsed from a file
        /// (which records no size for a free axis). `Template.specialize` overwrites it
        /// before inference runs, and a 0 that escaped would fail there rather than
        /// quietly become an empty axis.
        free_dims: []const ?u32 = &.{},
    },

    /// Rotary positional embedding over head dimension using chunked-halves pairing.
    ///
    /// Inputs:
    /// - x:         [B, L, N, H] (f16/f32)
    /// - positions: [B, L] (i32)
    ///
    /// Output:
    /// - out:       [B, L, N, H]
    ///
    /// Semantics match:
    /// - pairs_total = floor(H/2)
    /// - split x into left=x[..., :pairs_total], right=x[..., pairs_total:2*pairs_total], pass=x[..., 2*pairs_total:]
    /// - rotate first floor(rope_proportion * pairs_total) pairs using
    ///   theta_i = positions * (scale_factor * base_frequency^(-2*i/H))
    RoPE1D: struct {
        base_frequency: f32,
        scale_factor: f32,
        rope_proportion: f32,
    },

    /// In-place KV cache append.
    ///
    /// Inputs:
    /// - cache:     [B, T, H_kv, D] (f16/f32)
    /// - new_kv:    [B, new_len, H_kv, D] (same dtype as cache)
    /// - end_index: [B] (i32)
    ///
    /// Output:
    /// - out:       [B, T, H_kv, D] (aliased mutation semantics)
    SequenceAppend: void,

    /// Grouped-query attention (GQA) over time-major K/V, dense or cached.
    ///
    /// Inputs:
    /// - q:         [B, L_q, H_q, D_k]
    /// - k:               [B, T, H_kv, D_k]
    /// - v:               [B, T, H_kv, D_v]
    /// - query_positions: [B, L_q] (i32) -- optional
    /// - kv_lengths:      [B] (i32)      -- optional
    ///
    /// Output:
    /// - out:       [B, L_q, H_q, D_v]
    ///
    /// The controls are independent. Query positions default to row indices; K/V
    /// lengths default to T. Position-dependent attention with L_q != T must supply
    /// explicit query positions.
    ///
    /// Semantics:
    /// - valid keys are in logical range [0, kv_lengths[b]), or [0, T) when absent
    /// - causal mask uses absolute query positions from `query_positions`, or the query's
    ///   row index when absent
    /// - sliding_window == 0 means global attention
    /// - attn_logits_soft_cap == 0 means disabled
    /// - requires H_q % H_kv == 0
    Attention: struct {
        scale: f32,
        causal: bool,
        sliding_window: usize,
        attn_logits_soft_cap: f32,
        has_query_positions: bool,
        has_kv_lengths: bool,
    },

    /// Elementwise scalar-dtype cast.
    Cast: struct { to_dtype: DType },

    /// Matmul with B conceptually transposed; see `OpTag.MatMulNT` doc.
    MatMulNT: struct { alpha: f32 = 1.0, beta: f32 = 0.0 },

    /// Single-output conditional. Inputs: cond, then_value, else_value.
    If: struct { then_region: RegionId, else_region: RegionId },

    /// Loop region. Inputs: N carried inits (one per body-region output). The
    /// node produces N outputs (`output` + `extra_outputs`), each the final value
    /// of the corresponding carry. `cond_carry`, if set, names a carried index
    /// whose i32 [1] value is the continue predicate (early exit); `check_before`
    /// selects top-of-loop vs end-of-loop evaluation.
    Loop: struct {
        body_region: RegionId,
        static_max_trip_count: usize,
        cond_carry: ?usize = null,
        check_before: bool = true,
    },

    /// Real FFT over the last (power-of-two) dimension.
    ///
    /// Input:
    /// - x: [.., n_fft] real (f32), n_fft a power of two >= 4.
    ///
    /// Output:
    /// - out: [.., n_fft + 2] packed complex — one-sided bins `n_fft/2 + 1`
    ///   with real parts in `[0..bins)` and imaginary parts in `[bins..2*bins)`.
    RFFT: void,

    /// Short-time Fourier transform: frame the signal, apply the window, and
    /// run a real FFT per frame.
    ///
    /// Inputs:
    /// - signal: [batch, samples] real (f32)
    /// - window: [n_fft] real (f32) — already padded to n_fft by the caller
    ///
    /// Output:
    /// - out: [batch, num_frames, n_fft + 2] packed complex (real/imag halves,
    ///   same layout as RFFT).
    ///
    /// Framing: with `center`, the signal is reflect-padded by `n_fft/2` on each
    /// end (torch/NeMo default) and `num_frames = 1 + samples/hop_length`;
    /// otherwise `num_frames = 1 + (samples - n_fft)/hop_length`.
    STFT: struct {
        n_fft: usize,
        hop_length: usize,
        center: bool,
    },

    /// Relative-positional multi-head self-attention.
    ///
    /// Inputs (head-second layout):
    /// - q, k, v:        [B, H, T*, D] (q rows T_q; k/v rows T_kv)
    /// - pos_emb:        [H, P, D]  (P = 2*T_kv - 1; already projected by linear_pos)
    /// - pos_bias_u/_v:  [H, D]
    /// - mask (optional, when has_mask): [T_q, T_kv] additive
    ///
    /// Output:
    /// - out: [B, H, T_q, D]
    ///
    /// scores[i,j] = ((q[i]+u)·k[j] + (q[i]+v)·pos_emb[(T_q-1)-i+j]) * scale + mask[i,j]
    /// Relative-positional MHA. `chunk_size > 0` selects chunked-limited attention
    /// (NeMo `att_context_style="chunked_limited"`): the keys are cut into
    /// non-overlapping chunks of `chunk_size`, and a query attends to its own chunk
    /// plus `chunk_left` frames before that chunk's start — a contiguous interval,
    /// so it costs two integers instead of an additive [T_q, T_kv] mask tensor.
    /// `chunk_size == 0` means attend to every key (subject to `mask`).
    RelPosMHA: struct {
        scale: f32,
        has_mask: bool,
        chunk_size: usize = 0,
        chunk_left: usize = 0,
    },

    /// Index of max value along `axis` (v1: must be the last axis). Output i32,
    /// shape = input shape with `axis` removed (rank R-1; rank-1 input -> [1]).
    ArgMax: struct { axis: i32 },

    /// In-place row scatter: buf[idx] = src. Inputs (buf, idx[1] i32, src). The
    /// "row" is buf[1:] flattened (scalar for rank-1 buf). Output aliases buf.
    ScatterRow: void,
    Gather: struct { axis: i32, batch_dims: usize },
    Dim: struct { axis: i32 },
    Iota: struct { axis: i32 },
};

pub const InputArity = union(enum) {
    exact: usize,
    range: struct {
        min: usize,
        max: usize,
    },
    at_least: usize,

    pub fn allows(self: InputArity, input_count: usize) bool {
        return switch (self) {
            .exact => |v| input_count == v,
            .range => |v| input_count >= v.min and input_count <= v.max,
            .at_least => |v| input_count >= v,
        };
    }
};

pub fn opInputArity(op: Op) InputArity {
    return switch (op) {
        .MatMul => .{ .exact = 2 },
        .ElemwiseBinary => .{ .exact = 2 },
        .Unary => .{ .exact = 1 },
        .Softmax => .{ .exact = 1 },
        .Conv1D => .{ .range = .{ .min = 2, .max = 3 } },
        .Conv2D => .{ .range = .{ .min = 2, .max = 3 } },
        .LayerNorm => .{ .exact = 3 },
        .RMSNorm => .{ .exact = 3 },
        // q, k, v, [query_positions], [kv_lengths]
        .Attention => |a| .{ .exact = 3 + @as(usize, @intFromBool(a.has_query_positions)) + @as(usize, @intFromBool(a.has_kv_lengths)) },
        .Reduce => .{ .exact = 1 },
        .Concat => .{ .at_least = 1 },
        .LSTMCell => |lc| .{ .exact = if (lc.has_bias) 7 else 5 },
        .Copy => .{ .exact = 1 },
        .ViewReshape => .{ .exact = 1 },
        .ViewSqueeze => .{ .exact = 1 },
        .ViewUnsqueeze => .{ .exact = 1 },
        .ViewTranspose2D => .{ .exact = 1 },
        .ViewSliceND => .{ .exact = 1 },
        .RoPE1D => .{ .exact = 2 },
        .SequenceAppend => .{ .exact = 3 },
        .Cast => .{ .exact = 1 },
        .MatMulNT => .{ .exact = 2 },
        .If => .{ .exact = 3 },
        .Loop => .{ .at_least = 1 },
        .RFFT => .{ .exact = 1 },
        .STFT => .{ .exact = 2 },
        // q, k, v, pos_emb, pos_bias_u, pos_bias_v, [mask]
        .RelPosMHA => |r| .{ .exact = if (r.has_mask) 7 else 6 },
        .ArgMax => .{ .exact = 1 },
        .ScatterRow => .{ .exact = 3 },
        .Gather => .{ .exact = 2 },
        .Dim => .{ .exact = 1 },
        .Iota => .{ .exact = 1 },
    };
}

pub fn opInputCountValid(op: Op, input_count: usize) bool {
    const arity: InputArity = opInputArity(op);
    return arity.allows(input_count);
}

pub fn opTag(op: Op) OpTag {
    return std.meta.activeTag(op);
}

pub fn opId(op: Op) u8 {
    return @intFromEnum(opTag(op));
}

pub const Node = struct {
    op: Op,
    inputs: []ValueId,
    output: ValueId,
    /// Additional outputs beyond the primary `output`. Empty for all ops except
    /// a multi-carry `Loop` (whose i-th final carried tensor is exposed here as
    /// output i+1). Kept as a separate field so the 73 single-output call sites
    /// are unaffected.
    extra_outputs: []const ValueId = &[_]ValueId{},
};

pub const Region = struct {
    nodes: []Node,
    outputs: []ValueId,
};

pub const GraphError = error{ InvalidArgument, OutOfMemory };

/// An op attribute that can carry free axes: the sizes, and the symbol per axis.
///
/// Four ops have one, and they are the only ops that STATE a shape rather than deriving
/// it: the two norms' `normalized_shape`, a reshape's `new_shape`, a slice's `lens`.
/// Everywhere else a free axis flows through inference on its own — a matmul's output
/// inherits its operand's free dim with nothing to annotate — which is why this list is
/// four and not thirty-two.
///
/// Only the two view ops can be AUTHORED with free axes (`Builder.reshapeSym`/`sliceSym`);
/// the norms carry the field because the file format stores a shape term per axis there
/// and a package that has one must round-trip. Everything that treats the four uniformly
/// — resolving a template, reading a package, writing one — goes through here.
pub const SymbolicAttr = struct {
    sizes: []usize,
    free_dims: []const ?u32,
};

/// The op's free-axis attribute, or null when it has none.
pub fn symbolicAttr(op: *Op) ?SymbolicAttr {
    return switch (op.*) {
        .LayerNorm => |*a| .{ .sizes = @constCast(a.normalized_shape), .free_dims = a.free_dims },
        .RMSNorm => |*a| .{ .sizes = @constCast(a.normalized_shape), .free_dims = a.free_dims },
        .ViewReshape => |*a| .{ .sizes = @constCast(a.new_shape), .free_dims = a.free_dims },
        .ViewSliceND => |*a| .{ .sizes = @constCast(a.lens), .free_dims = a.free_dims },
        else => null,
    };
}

pub const Graph = struct {
    arena: std.heap.ArenaAllocator,
    allocator: std.mem.Allocator,

    values: std.ArrayList(Value) = .empty,
    nodes: std.ArrayList(Node) = .empty,
    outputs: std.ArrayList(ValueId) = .empty,
    regions: std.ArrayList(Region) = .empty,
    active_region_nodes: std.ArrayList(Node) = .empty,
    active_region: bool = false,

    const Self = @This();

    /// Maximum tensor rank the graph (and the tiling/exec layers) support.
    pub const MAX_RANK: usize = 8;

    pub fn init(allocator: std.mem.Allocator) Self {
        const arena = std.heap.ArenaAllocator.init(allocator);
        return .{ .arena = arena, .allocator = allocator };
    }

    pub fn deinit(self: *Self) void {
        self.values.deinit(self.allocator);
        self.nodes.deinit(self.allocator);
        self.outputs.deinit(self.allocator);
        for (self.regions.items) |region| {
            self.allocator.free(region.nodes);
            self.allocator.free(region.outputs);
        }
        self.regions.deinit(self.allocator);
        self.active_region_nodes.deinit(self.allocator);
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn dupeFreeDims(self: *Self, free_dims: []const ?u32) GraphError![]const ?u32 {
        if (free_dims.len == 0) return &.{};
        const out: []?u32 = self.arenaAlloc().alloc(?u32, free_dims.len) catch return GraphError.OutOfMemory;
        @memcpy(out, free_dims);
        return out;
    }

    pub fn arenaAlloc(self: *Self) std.mem.Allocator {
        return self.arena.allocator();
    }

    pub fn dupeShape(self: *Self, shape: []const usize) GraphError![]const usize {
        if (shape.len == 0 or shape.len > MAX_RANK) return GraphError.InvalidArgument;
        const out: []usize = self.arenaAlloc().alloc(usize, shape.len) catch return GraphError.OutOfMemory;
        @memcpy(out, shape);
        return out;
    }

    /// Deep-copy this graph into a fresh, independent `Graph` owned by `allocator`.
    ///
    /// Structure (values, nodes, regions, shapes, op attributes) is duplicated into
    /// the clone's own arena; external bindings (weight `TensorId`s) are copied by
    /// value, so the clone shares the source's weights rather than duplicating them.
    /// Value ids are preserved (same indices), so a caller's retained value ids stay
    /// valid against the clone. Used to compile a pristine authored template for a
    /// concrete shape without mutating the template.
    pub fn clone(self: *const Self, allocator: std.mem.Allocator) GraphError!Self {
        var out = Self.init(allocator);
        errdefer out.deinit();
        const aa = out.arenaAlloc();

        for (self.values.items) |v| {
            const sh: []const usize = if (v.shape.len == 0) &[_]usize{} else try out.dupeShape(v.shape);
            out.values.append(allocator, .{
                .dtype = v.dtype,
                .shape = sh,
                .dim_symbols = try dupeDimSymbols(aa, v.dim_symbols),
                .producer = v.producer,
                .external = v.external,
            }) catch return GraphError.OutOfMemory;
        }
        for (self.nodes.items) |n| {
            out.nodes.append(allocator, try cloneNode(aa, n)) catch return GraphError.OutOfMemory;
        }
        out.outputs.appendSlice(allocator, self.outputs.items) catch return GraphError.OutOfMemory;
        for (self.regions.items) |r| {
            const nodes = allocator.alloc(Node, r.nodes.len) catch return GraphError.OutOfMemory;
            errdefer allocator.free(nodes);
            for (r.nodes, 0..) |rn, i| nodes[i] = try cloneNode(aa, rn);
            const outs = allocator.dupe(ValueId, r.outputs) catch return GraphError.OutOfMemory;
            errdefer allocator.free(outs);
            out.regions.append(allocator, .{ .nodes = nodes, .outputs = outs }) catch return GraphError.OutOfMemory;
        }
        return out;
    }

    /// Symbol names are interned in the source graph's arena, so a clone that may
    /// outlive it needs its own copies.
    fn dupeDimSymbols(aa: std.mem.Allocator, syms: []const ?[]const u8) GraphError![]const ?[]const u8 {
        if (syms.len == 0) return &.{};
        const out = aa.alloc(?[]const u8, syms.len) catch return GraphError.OutOfMemory;
        for (syms, 0..) |maybe_name, i| {
            out[i] = if (maybe_name) |name| (aa.dupe(u8, name) catch return GraphError.OutOfMemory) else null;
        }
        return out;
    }

    fn cloneNode(aa: std.mem.Allocator, n: Node) GraphError!Node {
        return .{
            .op = try cloneOp(aa, n.op),
            .inputs = aa.dupe(ValueId, n.inputs) catch return GraphError.OutOfMemory,
            .output = n.output,
            .extra_outputs = aa.dupe(ValueId, n.extra_outputs) catch return GraphError.OutOfMemory,
        };
    }

    /// Copy an op, deep-copying the only attribute fields that hold slices
    /// (everything else is POD and copies by value).
    fn cloneOp(aa: std.mem.Allocator, op: Op) GraphError!Op {
        const dup = struct {
            fn f(a: std.mem.Allocator, s: []const usize) GraphError![]const usize {
                const d = a.alloc(usize, s.len) catch return GraphError.OutOfMemory;
                @memcpy(d, s);
                return d;
            }
        }.f;
        const dupSyms = struct {
            fn f(a: std.mem.Allocator, s: []const ?u32) GraphError![]const ?u32 {
                if (s.len == 0) return &.{};
                const d = a.alloc(?u32, s.len) catch return GraphError.OutOfMemory;
                @memcpy(d, s);
                return d;
            }
        }.f;
        // The four ops with slice attributes; everything else is copied by value. Same
        // list as `symbolicAttr`, which is why both live next to each other.
        return switch (op) {
            .LayerNorm => |x| .{ .LayerNorm = .{
                .eps = x.eps,
                .normalized_shape = try dup(aa, x.normalized_shape),
                .free_dims = try dupSyms(aa, x.free_dims),
            } },
            .RMSNorm => |x| .{ .RMSNorm = .{
                .eps = x.eps,
                .normalized_shape = try dup(aa, x.normalized_shape),
                .free_dims = try dupSyms(aa, x.free_dims),
            } },
            .ViewReshape => |x| .{ .ViewReshape = .{
                .new_shape = try dup(aa, x.new_shape),
                .free_dims = try dupSyms(aa, x.free_dims),
            } },
            .ViewSliceND => |x| .{ .ViewSliceND = .{
                .starts = try dup(aa, x.starts),
                .lens = try dup(aa, x.lens),
                .free_dims = try dupSyms(aa, x.free_dims),
            } },
            else => op,
        };
    }

    pub fn addValue(self: *Self) GraphError!ValueId {
        const id: ValueId = @intCast(self.values.items.len);
        self.values.append(self.allocator, .{}) catch return GraphError.OutOfMemory;
        return id;
    }

    pub fn addInput(self: *Self, dtype: DType, shape: []const usize) GraphError!ValueId {
        const id: ValueId = try self.addValue();
        const sh: []const usize = try self.dupeShape(shape);
        self.values.items[@intCast(id)] = .{ .dtype = dtype, .shape = sh };
        return id;
    }

    pub fn bindExternal(self: *Self, value: ValueId, external: ExternalId) GraphError!void {
        const idx: usize = @intCast(value);
        if (idx >= self.values.items.len) return GraphError.InvalidArgument;
        self.values.items[idx].external = external;
    }

    fn addNodeInternal(self: *Self, op: Op, inputs: []const ValueId) GraphError!ValueId {
        const out_id: ValueId = try self.addValue();

        const inputs_copy: []ValueId = self.arenaAlloc().alloc(ValueId, inputs.len) catch return GraphError.OutOfMemory;
        @memcpy(inputs_copy, inputs);

        const node: Node = .{ .op = op, .inputs = inputs_copy, .output = out_id };
        if (self.active_region) {
            self.active_region_nodes.append(self.allocator, node) catch return GraphError.OutOfMemory;
            // Mark the value as produced. The sentinel keeps package-export and
            // input-collection from misclassifying region-internal values as
            // public inputs; the actual node lives in active_region_nodes, not
            // in self.nodes, so the id isn't dereferenceable.
            self.values.items[@intCast(out_id)].producer = std.math.maxInt(NodeId);
        } else {
            const node_id: NodeId = @intCast(self.nodes.items.len);
            self.nodes.append(self.allocator, node) catch return GraphError.OutOfMemory;
            self.values.items[@intCast(out_id)].producer = node_id;
        }
        return out_id;
    }

    /// Like `addNodeInternal` but the node produces `n_outputs` values. Returns
    /// the (arena-owned) output id slice; `node.output` is outs[0] and the rest
    /// land in `node.extra_outputs`.
    fn addNodeMulti(self: *Self, op: Op, inputs: []const ValueId, n_outputs: usize) GraphError![]const ValueId {
        std.debug.assert(n_outputs >= 1);
        const outs: []ValueId = self.arenaAlloc().alloc(ValueId, n_outputs) catch return GraphError.OutOfMemory;
        for (outs) |*o| o.* = try self.addValue();

        const inputs_copy: []ValueId = self.arenaAlloc().alloc(ValueId, inputs.len) catch return GraphError.OutOfMemory;
        @memcpy(inputs_copy, inputs);

        const node: Node = .{ .op = op, .inputs = inputs_copy, .output = outs[0], .extra_outputs = outs[1..] };
        if (self.active_region) {
            self.active_region_nodes.append(self.allocator, node) catch return GraphError.OutOfMemory;
            for (outs) |o| self.values.items[@intCast(o)].producer = std.math.maxInt(NodeId);
        } else {
            const node_id: NodeId = @intCast(self.nodes.items.len);
            self.nodes.append(self.allocator, node) catch return GraphError.OutOfMemory;
            for (outs) |o| self.values.items[@intCast(o)].producer = node_id;
        }
        return outs;
    }

    /// Append a node whose output values already exist.
    ///
    /// The one op-agnostic way in: a caller that materializes a whole graph at once
    /// already knows every value id, so it has nothing to allocate. `Template.specialize`
    /// is that caller, and this is what keeps it from needing a case per op. The
    /// authoring `add*` helpers allocate their outputs as they go and cannot use it.
    pub fn appendNode(
        self: *Self,
        op: Op,
        inputs: []const ValueId,
        output: ValueId,
        extra_outputs: []const ValueId,
    ) GraphError!void {
        const inputs_copy: []ValueId = self.arenaAlloc().alloc(ValueId, inputs.len) catch return GraphError.OutOfMemory;
        @memcpy(inputs_copy, inputs);
        const extra_copy: []ValueId = self.arenaAlloc().alloc(ValueId, extra_outputs.len) catch return GraphError.OutOfMemory;
        @memcpy(extra_copy, extra_outputs);

        const node: Node = .{ .op = op, .inputs = inputs_copy, .output = output, .extra_outputs = extra_copy };
        if (self.active_region) {
            self.active_region_nodes.append(self.allocator, node) catch return GraphError.OutOfMemory;
            // See `addNodeInternal`: a region's nodes do not live in `self.nodes`, so the
            // producer id is a sentinel that only says "not a public input".
            self.values.items[@intCast(output)].producer = std.math.maxInt(NodeId);
            for (extra_copy) |o| self.values.items[@intCast(o)].producer = std.math.maxInt(NodeId);
        } else {
            const node_id: NodeId = @intCast(self.nodes.items.len);
            self.nodes.append(self.allocator, node) catch return GraphError.OutOfMemory;
            self.values.items[@intCast(output)].producer = node_id;
            for (extra_copy) |o| self.values.items[@intCast(o)].producer = node_id;
        }
    }

    /// The node that produced `value`, mutable, when it is in scope.
    ///
    /// The authoring `Builder` needs it to finish an op whose attribute it could only
    /// resolve after the op existed. Null for a value with no producer, and for one
    /// produced inside a region that is no longer open.
    pub fn mutableNodeFor(self: *Self, value: ValueId) ?*Node {
        const idx: usize = @intCast(value);
        if (idx >= self.values.items.len) return null;
        const producer = self.values.items[idx].producer orelse return null;
        if (producer == std.math.maxInt(NodeId)) {
            if (!self.active_region or self.active_region_nodes.items.len == 0) return null;
            const last = &self.active_region_nodes.items[self.active_region_nodes.items.len - 1];
            return if (last.output == value) last else null;
        }
        if (producer >= self.nodes.items.len) return null;
        return &self.nodes.items[@intCast(producer)];
    }

    /// The most recently appended node — the active region's last node while a
    /// region is open, otherwise the graph's last node. Lets the authoring
    /// `Builder` infer the node it just added. Null if none exists in scope.
    pub fn lastNode(self: *const Self) ?Node {
        if (self.active_region) {
            if (self.active_region_nodes.items.len == 0) return null;
            return self.active_region_nodes.items[self.active_region_nodes.items.len - 1];
        }
        if (self.nodes.items.len == 0) return null;
        return self.nodes.items[self.nodes.items.len - 1];
    }

    pub fn beginRegion(self: *Self) GraphError!void {
        if (self.active_region) return GraphError.InvalidArgument;
        self.active_region_nodes.clearRetainingCapacity();
        self.active_region = true;
    }

    pub fn endRegion(self: *Self, outputs_in: []const ValueId) GraphError!RegionId {
        if (!self.active_region) return GraphError.InvalidArgument;
        if (outputs_in.len == 0) return GraphError.InvalidArgument;

        const nodes_slice: []Node = self.active_region_nodes.toOwnedSlice(self.allocator) catch return GraphError.OutOfMemory;
        errdefer self.allocator.free(nodes_slice);
        const outputs_slice: []ValueId = self.allocator.dupe(ValueId, outputs_in) catch return GraphError.OutOfMemory;
        errdefer self.allocator.free(outputs_slice);

        const id: RegionId = @intCast(self.regions.items.len);
        self.regions.append(self.allocator, .{ .nodes = nodes_slice, .outputs = outputs_slice }) catch return GraphError.OutOfMemory;
        self.active_region = false;
        return id;
    }

    pub fn addIf(self: *Self, cond: ValueId, then_region: RegionId, else_region: RegionId) GraphError!ValueId {
        const then_idx: usize = @intCast(then_region);
        const else_idx: usize = @intCast(else_region);
        if (then_idx >= self.regions.items.len or else_idx >= self.regions.items.len) return GraphError.InvalidArgument;
        if (self.regions.items[then_idx].outputs.len != 1 or self.regions.items[else_idx].outputs.len != 1) return GraphError.InvalidArgument;
        return self.addNodeInternal(
            .{ .If = .{ .then_region = then_region, .else_region = else_region } },
            &[_]ValueId{ cond, self.regions.items[then_idx].outputs[0], self.regions.items[else_idx].outputs[0] },
        );
    }

    pub fn addLoop(self: *Self, carried_init: ValueId, body_region: RegionId, static_max_trip_count: usize) GraphError!ValueId {
        const outs = try self.addLoopMulti(&[_]ValueId{carried_init}, body_region, static_max_trip_count, null, true);
        return outs[0];
    }

    /// Multi-carry loop. `carried_inits[i]` pairs with body-region output i; the
    /// returned slice holds the N final carried values. `cond_carry`, if set,
    /// names the carry index used as the i32 [1] continue predicate.
    pub fn addLoopMulti(
        self: *Self,
        carried_inits: []const ValueId,
        body_region: RegionId,
        static_max_trip_count: usize,
        cond_carry: ?usize,
        check_before: bool,
    ) GraphError![]const ValueId {
        const body_idx: usize = @intCast(body_region);
        if (body_idx >= self.regions.items.len) return GraphError.InvalidArgument;
        const n = carried_inits.len;
        if (n == 0 or self.regions.items[body_idx].outputs.len != n or static_max_trip_count == 0) return GraphError.InvalidArgument;
        if (cond_carry) |ci| {
            if (ci >= n) return GraphError.InvalidArgument;
        }
        return self.addNodeMulti(.{ .Loop = .{
            .body_region = body_region,
            .static_max_trip_count = static_max_trip_count,
            .cond_carry = cond_carry,
            .check_before = check_before,
        } }, carried_inits, n);
    }

    pub fn addMatMul(self: *Self, a: ValueId, b: ValueId, alpha: f32, beta: f32) GraphError!ValueId {
        return self.addNodeInternal(.{ .MatMul = .{ .alpha = alpha, .beta = beta } }, &[_]ValueId{ a, b });
    }

    pub fn addElemwiseBinary(self: *Self, op: ElemwiseBinaryOp, a: ValueId, b: ValueId) GraphError!ValueId {
        return self.addNodeInternal(.{ .ElemwiseBinary = .{ .op = op } }, &[_]ValueId{ a, b });
    }

    pub fn addUnary(self: *Self, op: UnaryOp, a: ValueId) GraphError!ValueId {
        return self.addNodeInternal(.{ .Unary = .{ .op = op } }, &[_]ValueId{a});
    }

    /// Softmax over the specified axis (negative axes are allowed).
    /// - axis == -1 => last dimension
    pub fn addSoftmax(self: *Self, a: ValueId, axis: i32) GraphError!ValueId {
        return self.addNodeInternal(.{ .Softmax = .{ .axis = axis } }, &[_]ValueId{a});
    }

    pub fn addConv1D(
        self: *Self,
        x: ValueId,
        w: ValueId,
        bias: ?ValueId,
        stride: usize,
        dilation: usize,
        pad_left: usize,
        pad_right: usize,
        groups: usize,
    ) GraphError!ValueId {
        return self.addConv1DWithPadMode(x, w, bias, stride, dilation, pad_left, pad_right, .zero, groups);
    }

    pub fn addConv1DWithPadMode(
        self: *Self,
        x: ValueId,
        w: ValueId,
        bias: ?ValueId,
        stride: usize,
        dilation: usize,
        pad_left: usize,
        pad_right: usize,
        pad_mode: PadMode,
        groups: usize,
    ) GraphError!ValueId {
        const op: Op = .{ .Conv1D = .{
            .stride = stride,
            .dilation = dilation,
            .pad_left = pad_left,
            .pad_right = pad_right,
            .pad_mode = pad_mode,
            .groups = groups,
        } };
        if (bias) |b| {
            return self.addNodeInternal(op, &[_]ValueId{ x, w, b });
        }
        return self.addNodeInternal(op, &[_]ValueId{ x, w });
    }

    pub fn addConv2D(
        self: *Self,
        x: ValueId,
        w: ValueId,
        bias: ?ValueId,
        stride_h: usize,
        stride_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        pad_top: usize,
        pad_bottom: usize,
        pad_left: usize,
        pad_right: usize,
        groups: usize,
    ) GraphError!ValueId {
        return self.addConv2DWithPadMode(x, w, bias, stride_h, stride_w, dilation_h, dilation_w, pad_top, pad_bottom, pad_left, pad_right, .zero, groups);
    }

    pub fn addConv2DWithPadMode(
        self: *Self,
        x: ValueId,
        w: ValueId,
        bias: ?ValueId,
        stride_h: usize,
        stride_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        pad_top: usize,
        pad_bottom: usize,
        pad_left: usize,
        pad_right: usize,
        pad_mode: PadMode,
        groups: usize,
    ) GraphError!ValueId {
        const op: Op = .{ .Conv2D = .{
            .stride_h = stride_h,
            .stride_w = stride_w,
            .dilation_h = dilation_h,
            .dilation_w = dilation_w,
            .pad_top = pad_top,
            .pad_bottom = pad_bottom,
            .pad_left = pad_left,
            .pad_right = pad_right,
            .pad_mode = pad_mode,
            .groups = groups,
        } };
        if (bias) |b| {
            return self.addNodeInternal(op, &[_]ValueId{ x, w, b });
        }
        return self.addNodeInternal(op, &[_]ValueId{ x, w });
    }

    pub fn addLayerNorm(self: *Self, x: ValueId, gamma: ValueId, beta: ValueId, eps: f32, normalized_shape: []const usize) GraphError!ValueId {
        const ns: []const usize = try self.dupeShape(normalized_shape);
        return self.addNodeInternal(.{ .LayerNorm = .{ .eps = eps, .normalized_shape = ns } }, &[_]ValueId{ x, gamma, beta });
    }

    pub fn addRMSNorm(self: *Self, x: ValueId, gamma: ValueId, beta: ValueId, eps: f32, normalized_shape: []const usize) GraphError!ValueId {
        const ns: []const usize = try self.dupeShape(normalized_shape);
        return self.addNodeInternal(.{ .RMSNorm = .{ .eps = eps, .normalized_shape = ns } }, &[_]ValueId{ x, gamma, beta });
    }

    /// Relative-positional multi-head self-attention (Conformer / Transformer-XL).
    /// `pos_emb` is [H, P, D] (P = 2*T_kv-1, already projected); biases are [H, D];
    /// `mask` (optional) is an additive [T_q, T_kv] tensor.
    ///
    /// `chunk_size`/`chunk_left` express chunked-limited attention structurally (see
    /// `Op.RelPosMHA`); pass 0 for full attention. They compose with `mask`, which
    /// then only has to carry what an interval cannot — e.g. streaming padding.
    /// There is no `heads` parameter or attribute: the head count is `q`'s dim 2, and
    /// every consumer reads it from the shape. (Carrying it separately is the exact
    /// redundancy that got `MultiHeadAttention.heads` retired.)
    pub fn addRelPosMHA(
        self: *Self,
        q: ValueId,
        k: ValueId,
        v: ValueId,
        pos_emb: ValueId,
        pos_bias_u: ValueId,
        pos_bias_v: ValueId,
        mask: ?ValueId,
        scale: f32,
        chunk_size: usize,
        chunk_left: usize,
    ) GraphError!ValueId {
        const op: Op = .{ .RelPosMHA = .{
            .scale = scale,
            .has_mask = (mask != null),
            .chunk_size = chunk_size,
            .chunk_left = chunk_left,
        } };
        if (mask) |m| {
            return self.addNodeInternal(op, &[_]ValueId{ q, k, v, pos_emb, pos_bias_u, pos_bias_v, m });
        }
        return self.addNodeInternal(op, &[_]ValueId{ q, k, v, pos_emb, pos_bias_u, pos_bias_v });
    }

    /// Index of the max value along `axis` (v1: last axis). Output dtype i32.
    pub fn addArgMax(self: *Self, x: ValueId, axis: i32) GraphError!ValueId {
        return self.addNodeInternal(.{ .ArgMax = .{ .axis = axis } }, &[_]ValueId{x});
    }

    /// In-place row scatter: buf[idx] = src. Output aliases buf.
    pub fn addScatterRow(self: *Self, buf: ValueId, idx: ValueId, src: ValueId) GraphError!ValueId {
        return self.addNodeInternal(.{ .ScatterRow = {} }, &[_]ValueId{ buf, idx, src });
    }

    /// Grouped-query attention; see `Op.Attention`. Optional query positions and
    /// K/V lengths are independent and appended to the input list in that order.
    pub fn addAttention(
        self: *Self,
        q: ValueId,
        k: ValueId,
        v: ValueId,
        query_positions: ?ValueId,
        kv_lengths: ?ValueId,
        scale: f32,
        causal: bool,
        sliding_window: usize,
        attn_logits_soft_cap: f32,
    ) GraphError!ValueId {
        const op: Op = .{ .Attention = .{
            .scale = scale,
            .causal = causal,
            .sliding_window = sliding_window,
            .attn_logits_soft_cap = attn_logits_soft_cap,
            .has_query_positions = (query_positions != null),
            .has_kv_lengths = (kv_lengths != null),
        } };
        if (query_positions) |p| {
            if (kv_lengths) |lens| return self.addNodeInternal(op, &[_]ValueId{ q, k, v, p, lens });
            return self.addNodeInternal(op, &[_]ValueId{ q, k, v, p });
        }
        if (kv_lengths) |lens| return self.addNodeInternal(op, &[_]ValueId{ q, k, v, lens });
        return self.addNodeInternal(op, &[_]ValueId{ q, k, v });
    }

    pub fn addRelu(self: *Self, a: ValueId) GraphError!ValueId {
        return self.addUnary(.relu, a);
    }

    pub fn addReduce(self: *Self, op: ReduceOp, a: ValueId) GraphError!ValueId {
        return self.addNodeInternal(.{ .Reduce = .{ .op = op, .axis = null } }, &[_]ValueId{a});
    }

    pub fn addReduceAxis(self: *Self, op: ReduceOp, a: ValueId, axis: i32) GraphError!ValueId {
        return self.addNodeInternal(.{ .Reduce = .{ .op = op, .axis = axis } }, &[_]ValueId{a});
    }

    pub fn addConcat(self: *Self, inputs: []const ValueId, axis: i32) GraphError!ValueId {
        if (inputs.len == 0 or inputs.len > MAX_CONCAT_INPUTS) return GraphError.InvalidArgument;
        return self.addNodeInternal(.{ .Concat = .{ .axis = axis } }, inputs);
    }

    pub fn addCopy(self: *Self, a: ValueId) GraphError!ValueId {
        return self.addNodeInternal(.Copy, &[_]ValueId{a});
    }

    pub fn addGather(self: *Self, data: ValueId, indices: ValueId, axis: i32, batch_dims: usize) GraphError!ValueId {
        return self.addNodeInternal(.{ .Gather = .{ .axis = axis, .batch_dims = batch_dims } }, &[_]ValueId{ data, indices });
    }

    pub fn addDim(self: *Self, input: ValueId, axis: i32) GraphError!ValueId {
        return self.addNodeInternal(.{ .Dim = .{ .axis = axis } }, &[_]ValueId{input});
    }

    pub fn addIota(self: *Self, shape_like: ValueId, axis: i32) GraphError!ValueId {
        return self.addNodeInternal(.{ .Iota = .{ .axis = axis } }, &[_]ValueId{shape_like});
    }

    pub fn addRoPE1D(
        self: *Self,
        x: ValueId,
        positions: ValueId,
        base_frequency: f32,
        scale_factor: f32,
        rope_proportion: f32,
    ) GraphError!ValueId {
        return self.addNodeInternal(
            .{ .RoPE1D = .{
                .base_frequency = base_frequency,
                .scale_factor = scale_factor,
                .rope_proportion = rope_proportion,
            } },
            &[_]ValueId{ x, positions },
        );
    }

    pub fn addSequenceAppend(
        self: *Self,
        cache: ValueId,
        new_kv: ValueId,
        end_index: ValueId,
    ) GraphError!ValueId {
        return self.addNodeInternal(
            .SequenceAppend,
            &[_]ValueId{ cache, new_kv, end_index },
        );
    }

    pub fn addCast(self: *Self, x: ValueId, to_dtype: DType) GraphError!ValueId {
        return self.addNodeInternal(.{ .Cast = .{ .to_dtype = to_dtype } }, &[_]ValueId{x});
    }

    pub fn addMatMulNT(self: *Self, a: ValueId, b: ValueId, alpha: f32, beta: f32) GraphError!ValueId {
        return self.addNodeInternal(
            .{ .MatMulNT = .{ .alpha = alpha, .beta = beta } },
            &[_]ValueId{ a, b },
        );
    }

    pub fn addLSTMCell(
        self: *Self,
        x: ValueId,
        h_prev: ValueId,
        c_prev: ValueId,
        w_ih: ValueId,
        w_hh: ValueId,
        b_ih: ?ValueId,
        b_hh: ?ValueId,
    ) GraphError!ValueId {
        // Bias policy: either both provided or both omitted.
        if ((b_ih != null) != (b_hh != null)) return GraphError.InvalidArgument;

        const op: Op = .{ .LSTMCell = .{ .has_bias = (b_ih != null) } };
        if (b_ih) |b0| {
            return self.addNodeInternal(op, &[_]ValueId{ x, h_prev, c_prev, w_ih, w_hh, b0, b_hh.? });
        }
        return self.addNodeInternal(op, &[_]ValueId{ x, h_prev, c_prev, w_ih, w_hh });
    }

    pub fn addRFFT(self: *Self, x: ValueId) GraphError!ValueId {
        return self.addNodeInternal(.RFFT, &[_]ValueId{x});
    }

    pub fn addSTFT(
        self: *Self,
        signal: ValueId,
        window: ValueId,
        n_fft: usize,
        hop_length: usize,
        center: bool,
    ) GraphError!ValueId {
        if (n_fft == 0 or hop_length == 0) return GraphError.InvalidArgument;
        return self.addNodeInternal(
            .{ .STFT = .{ .n_fft = n_fft, .hop_length = hop_length, .center = center } },
            &[_]ValueId{ signal, window },
        );
    }

    pub fn addViewReshape(self: *Self, a: ValueId, new_shape: []const usize) GraphError!ValueId {
        return self.addViewReshapeSym(a, new_shape, &.{});
    }

    /// Reshape with free axes: `free_dims[i]` is the dim expression giving axis `i`'s
    /// runtime size, `new_shape[i]` being its authoring placeholder. `&.{}` for none.
    pub fn addViewReshapeSym(
        self: *Self,
        a: ValueId,
        new_shape: []const usize,
        free_dims: []const ?u32,
    ) GraphError!ValueId {
        const sh: []const usize = try self.dupeShape(new_shape);
        return self.addNodeInternal(
            .{ .ViewReshape = .{ .new_shape = sh, .free_dims = try self.dupeFreeDims(free_dims) } },
            &[_]ValueId{a},
        );
    }

    pub fn addViewSqueeze(self: *Self, a: ValueId, axis: ?i32) GraphError!ValueId {
        return self.addNodeInternal(.{ .ViewSqueeze = .{ .axis = axis } }, &[_]ValueId{a});
    }

    pub fn addViewUnsqueeze(self: *Self, a: ValueId, axis: i32) GraphError!ValueId {
        return self.addNodeInternal(.{ .ViewUnsqueeze = .{ .axis = axis } }, &[_]ValueId{a});
    }

    pub fn addViewTranspose2D(self: *Self, a: ValueId) GraphError!ValueId {
        return self.addNodeInternal(.ViewTranspose2D, &[_]ValueId{a});
    }

    pub fn addViewSliceND(self: *Self, a: ValueId, starts: []const usize, lens: []const usize) GraphError!ValueId {
        return self.addViewSliceNDSym(a, starts, lens, &.{});
    }

    /// Slice with free extents. See `addViewReshapeSym`; only `lens` may be free, since a
    /// slice's start is a position rather than an extent.
    pub fn addViewSliceNDSym(
        self: *Self,
        a: ValueId,
        starts: []const usize,
        lens: []const usize,
        free_dims: []const ?u32,
    ) GraphError!ValueId {
        if (starts.len == 0 or starts.len != lens.len or starts.len > MAX_RANK) return GraphError.InvalidArgument;
        const starts_copy: []const usize = try self.dupeShape(starts);
        const lens_copy: []const usize = try self.dupeShape(lens);
        return self.addNodeInternal(
            .{ .ViewSliceND = .{ .starts = starts_copy, .lens = lens_copy, .free_dims = try self.dupeFreeDims(free_dims) } },
            &[_]ValueId{a},
        );
    }

    pub fn addViewSlice2D(self: *Self, a: ValueId, start0: usize, len0: usize, start1: usize, len1: usize) GraphError!ValueId {
        return self.addViewSliceND(a, &[_]usize{ start0, start1 }, &[_]usize{ len0, len1 });
    }

    pub fn setOutputs(self: *Self, outs: []const ValueId) GraphError!void {
        self.outputs.clearRetainingCapacity();
        self.outputs.appendSlice(self.allocator, outs) catch return GraphError.OutOfMemory;
    }
};
