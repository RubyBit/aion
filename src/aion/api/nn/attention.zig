// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const types = @import("../../backend/types.zig");
const package_export = @import("../package_export.zig");
const layer_mod = @import("layer.zig");
const state_mod = @import("state.zig");
const linear_mod = @import("linear.zig");

pub const Builder = layer_mod.Builder;
pub const TensorRef = layer_mod.TensorRef;
pub const Tensor = layer_mod.Tensor;
pub const LayerName = layer_mod.LayerName;
pub const Params = state_mod.Params;
pub const BindError = state_mod.BindError;
pub const Linear = linear_mod.Linear;
pub const OutputAlias = package_export.OutputAlias;
pub const InputRoleDecl = package_export.InputRoleDecl;

/// Rotary position embedding over `x: [batch, seq, heads, head_dim]`, driven by an
/// i32 `positions: [batch, seq]`.
///
/// Stateless — the rotation is derived from `positions`, not from a parameter — so
/// it exists to name the hyperparameters rather than to hold weights.
pub const RoPE = struct {
    opts: Options = .{},

    const Self = @This();

    pub const Options = struct {
        /// Rotation base (`theta`). 10000 is the usual default; long-context models
        /// raise it, and Gemma uses different bases for local vs global layers.
        base_frequency: f32 = 10000.0,
        /// Linear position scaling for context extension (1.0 = none).
        scale_factor: f32 = 1.0,
        /// Fraction of each head's pairs to rotate (1.0 = all of them).
        rope_proportion: f32 = 1.0,
    };

    pub fn init(opts: Options) Self {
        return .{ .opts = opts };
    }

    pub fn bind(bld: *Builder, opts: Options) Builder.Error!Self {
        _ = bld;
        return .{ .opts = opts };
    }

    pub fn forward(self: Self, bld: *Builder, x: TensorRef, positions: TensorRef) Builder.Error!TensorRef {
        return bld.rope1d(x, positions, self.opts.base_frequency, self.opts.scale_factor, self.opts.rope_proportion);
    }
};

/// Independent 1-D RoPE per spatial axis, over consecutive head-dim slices.
///
/// A 2-D image patch rotates its first half by the patch's x position and its
/// second half by its y. `head_dim` splits evenly across `positions`, and each
/// slice is one ordinary `rope1d` — the axes never mix.
pub const AxialRoPE = struct {
    opts: Options = .{},

    const Self = @This();

    pub const Options = struct {
        base_frequency: f32 = 10000.0,
    };

    pub fn init(opts: Options) Self {
        return .{ .opts = opts };
    }

    pub fn forward(self: Self, bld: *Builder, x: TensorRef, positions: []const TensorRef) Builder.Error!TensorRef {
        if (positions.len == 0 or positions.len > 4) return Builder.Error.InvalidArgument;
        const shape = bld.knownShape(x) orelse return Builder.Error.InvalidArgument;
        const head_dim: usize = shape[shape.len - 1];
        if (head_dim % positions.len != 0) return Builder.Error.InvalidArgument;
        const width: usize = head_dim / positions.len;

        var parts: [4]TensorRef = undefined;
        for (positions, 0..) |pos, i| {
            const part = try bld.sliceLastDim(x, i * width, width);
            parts[i] = try bld.rope1d(part, pos, self.opts.base_frequency, 1.0, 1.0);
        }
        return bld.concat(parts[0..positions.len], -1);
    }
};

/// A `[batch, capacity, kv_heads, head_dim]` key/value cache that the runtime
/// allocates, zero-fills, and carries across runs.
///
/// This bundles the three things a cache needs, which every converter wires by
/// hand and which are easy to get subtly wrong:
///  1. the cache *input* (a public input the runtime owns),
///  2. the `sequenceAppend` that writes this step's K/V at `write_index`,
///  3. the role declaration + output alias that make it persistent.
///
/// Roles and aliases are consumed by `compile`/`export`, not by the `Builder`, so
/// `roleDecl()` and `alias()` hand them back for the caller to pass along. That is
/// the API shape; hiding it would mean hiding `compile`'s options.
pub const KVCache = struct {
    input: TensorRef,
    current: TensorRef,
    /// Element type of the cache store. f16 halves cache footprint and is what the
    /// cached-attention op expects for large contexts.
    dtype: types.DType,
    axis: u8,
    capacity_symbol: ?[]const u8,
    growable: bool,

    const Self = @This();

    pub const Options = struct {
        /// Name for the cache input, so it is addressable from the loaded model.
        name: ?[]const u8 = null,
        dtype: types.DType = .f16,
        /// Fixed capacity along the time axis. Ignored when `capacity_symbol` is set.
        capacity: usize = 0,
        /// Free (symbolic) capacity axis; the loader sizes it. Must also be declared
        /// on the builder via `symbolicDim`, which `bind` does for you.
        capacity_symbol: ?[]const u8 = null,
        /// Let the runtime grow the cache on demand up to its bound.
        growable: bool = false,
    };

    /// Declare the cache input. `shape` is `[batch, capacity, kv_heads, head_dim]`;
    /// the capacity entry is replaced by `opts.capacity`/`opts.capacity_symbol`.
    pub fn bind(
        bld: *Builder,
        batch: usize,
        kv_heads: usize,
        head_dim: usize,
        opts: Options,
    ) Builder.Error!Self {
        if (batch == 0 or kv_heads == 0 or head_dim == 0) return Builder.Error.InvalidArgument;

        // A symbolic axis still needs a concrete authoring placeholder for eager
        // shape inference; 1 is enough and the loader overrides it.
        const capacity: usize = if (opts.capacity_symbol != null)
            @max(@as(usize, 1), opts.capacity)
        else if (opts.capacity == 0)
            return Builder.Error.InvalidArgument
        else
            opts.capacity;

        const shape = [_]usize{ batch, capacity, kv_heads, head_dim };
        const input: TensorRef = try bld.input(opts.dtype, &shape);
        if (opts.name) |n| _ = try bld.name(input, n);
        if (opts.capacity_symbol) |sym| try bld.symbolicDim(input, 1, sym);

        return .{
            .input = input,
            .current = input,
            .dtype = opts.dtype,
            .axis = 1,
            .capacity_symbol = opts.capacity_symbol,
            .growable = opts.growable,
        };
    }

    /// Append this step's K or V, casting to the cache dtype.
    ///
    /// `kv` is `[batch, seq, kv_heads, head_dim]` — what `Attention.project`
    /// returns and the cache stores. Keeping the projection and cache layouts the
    /// same makes multi-KV-head append a direct operation.
    pub fn append(self: *Self, bld: *Builder, kv: TensorRef, write_index: TensorRef) Builder.Error!TensorRef {
        const dims: []const usize = bld.knownShape(kv) orelse return Builder.Error.InvalidArgument;
        const store: []const usize = bld.knownShape(self.input) orelse return Builder.Error.InvalidArgument;
        if (dims.len != 4 or store.len != 4) return Builder.Error.InvalidArgument;

        const kv_heads: usize = store[2];
        if (dims[2] != kv_heads or dims[3] != store[3]) return Builder.Error.ShapeMismatch;

        const cast_kv: TensorRef = if (bld.dtypeOf(kv)) |dt|
            (if (dt == self.dtype) kv else try bld.cast(kv, self.dtype))
        else
            try bld.cast(kv, self.dtype);

        self.current = try bld.sequenceAppend(self.current, cast_kv, write_index);
        return self.current;
    }

    /// Hand to `compile`/`export` so the runtime allocates and zero-fills the cache.
    pub fn roleDecl(self: Self) InputRoleDecl {
        return .{
            .input = self.input,
            .kind = .sequence_cache,
            .axis = self.axis,
            .capacity_symbol = self.capacity_symbol,
            .zero_init = true,
            .allow_growable = self.growable,
        };
    }

    /// The updated cache after the last `append`.
    ///
    /// This must be listed among `compile`/`export` outputs: an output alias is
    /// resolved by *output index*, so a cache that is not an output has nothing to
    /// alias to and `compile` rejects it.
    pub fn outputRef(self: Self) TensorRef {
        return self.current;
    }

    /// Hand to `compile`/`export` so the updated cache is written back over the
    /// input, making it persist across runs. Pair with `outputRef()` in the output
    /// list.
    pub fn alias(self: Self) OutputAlias {
        return .{ .input = self.input, .output = self.current };
    }
};

/// Grouped-query attention — over a KV cache (decode) or a plain sequence
/// (prefill, encoders); `attend` takes the cache indices or omits them.
///
/// Holds the four projections and the attention hyperparameters. Q, K and V are
/// separate, the way checkpoints ship them; concatenating them into one wide
/// projection is a fusion the compiler performs (`opt/horizontal_matmul`),
/// not a weight layout this API asks you to pick.
///
/// Grouped-query attention falls out of `kv_heads < heads`; the op requires
/// `heads % kv_heads == 0`.
pub const Attention = struct {
    q_proj: Linear,
    /// Null when this layer reads a cache another layer writes (see `Weights`).
    k_proj: ?Linear,
    v_proj: ?Linear,
    o_proj: Linear,

    cfg: Config,
    id: LayerName = .{},

    const Self = @This();

    /// `k_proj`/`v_proj` are optional, all-or-nothing: omitting them means this
    /// layer does not write a cache, it only reads one another layer produced.
    ///
    /// That is a real architecture, not a packing choice — Gemma 4 shares one KV
    /// cache across a run of layers, where only the first projects K/V and the rest
    /// attend over it. `attend` already takes the caches as arguments, so the shared
    /// case needs nothing else.
    pub const Weights = struct {
        q_proj: Linear.Weights,
        k_proj: ?Linear.Weights = null,
        v_proj: ?Linear.Weights = null,
        o_proj: Linear.Weights,
    };

    pub const Config = struct {
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        /// Logit scale. Pass `1 / sqrt(head_dim)` for textbook attention; some
        /// models fold the scale into a Q-norm instead and pass 1.0.
        scale: f32,
        window: Builder.AttentionWindow = .causal,
        /// `cap * tanh(logits / cap)` on the attention logits (0 = disabled).
        attn_logits_soft_cap: f32 = 0.0,
    };

    pub const Options = struct {
        name: ?[]const u8 = null,
    };

    pub fn bind(bld: *Builder, params: anytype, cfg: Config, opts: Options) BindError!Self {
        try validate(cfg);

        var src = state_mod.binding(Weights, params);
        const p: Params = src.params();

        const id, const scope = try LayerName.open(Self, bld, p.name orelse opts.name);
        defer bld.endScope(scope);

        // K and V go together: a layer with one but not the other cannot fill a
        // cache, and would fail later with a confusing shape error instead.
        const has_k: bool = try p.hasNested(bld, Weights, .k_proj, Linear.Weights, .weight);
        const has_v: bool = try p.hasNested(bld, Weights, .v_proj, Linear.Weights, .weight);
        if (has_k != has_v) return error.InvalidArgument;

        return .{
            .q_proj = try Linear.bind(bld, p.child(Weights, .q_proj), .{}),
            .k_proj = if (has_k) try Linear.bind(bld, p.child(Weights, .k_proj), .{}) else null,
            .v_proj = if (has_v) try Linear.bind(bld, p.child(Weights, .v_proj), .{}) else null,
            .o_proj = try Linear.bind(bld, p.child(Weights, .o_proj), .{}),
            .cfg = cfg,
            .id = id,
        };
    }

    fn validate(cfg: Config) BindError!void {
        if (cfg.heads == 0 or cfg.kv_heads == 0 or cfg.head_dim == 0) return Builder.Error.InvalidArgument;
        if (cfg.heads % cfg.kv_heads != 0) return Builder.Error.InvalidArgument;
        if (!std.math.isFinite(cfg.scale)) return Builder.Error.InvalidArgument;
    }

    /// Q/K/V for this step, all reshaped to `[batch, seq, heads, head_dim]` (K/V
    /// with `kv_heads`).
    ///
    /// One layout for all three, shared by projection, norms, RoPE, cache append,
    /// and attention.
    ///
    /// `k`/`v` are null on a layer that only reads a shared cache — there is
    /// nothing to append, and `attend` is handed the owning layer's cache instead.
    pub const Projected = struct {
        q: TensorRef,
        k: ?TensorRef,
        v: ?TensorRef,
    };

    /// Run the projections and reshape into attention layout, without touching the
    /// cache. Split out so a caller can insert Q/K norms or RoPE — which every
    /// modern transformer does — between projection and attention.
    pub fn project(self: Self, bld: *Builder, x: TensorRef) Builder.Error!Projected {
        const q_flat: TensorRef = try self.q_proj.forward(bld, x);

        const rank: usize = (bld.knownShape(q_flat) orelse return Builder.Error.InvalidArgument).len;
        if (rank < 2) return Builder.Error.InvalidArgument;
        // Batch and sequence are carried over as they are, so an axis the caller
        // declared free stays free through the reshape. Reading them off the value
        // rather than taking them as configuration is the whole point: the layer
        // never has to be told which of its axes vary.
        const batch: Builder.Dim = try bld.dimAt(q_flat, 0);
        const seq: Builder.Dim = if (rank >= 3) try bld.dimAt(q_flat, 1) else .{ .size = 1 };

        const heads: Builder.Dim = .{ .size = self.cfg.heads };
        const kv_heads: Builder.Dim = .{ .size = self.cfg.kv_heads };
        const head_dim: Builder.Dim = .{ .size = self.cfg.head_dim };

        const kv_dims = [_]Builder.Dim{ batch, seq, kv_heads, head_dim };

        return .{
            .q = try bld.reshapeDims(q_flat, &[_]Builder.Dim{ batch, seq, heads, head_dim }),
            .k = if (self.k_proj) |k_lin| try bld.reshapeDims(try k_lin.forward(bld, x), &kv_dims) else null,
            .v = if (self.v_proj) |v_lin| try bld.reshapeDims(try v_lin.forward(bld, x), &kv_dims) else null,
        };
    }

    /// Attend `q` against k/v, then apply the output projection.
    ///
    /// `q` is `[batch, seq, heads, head_dim]`; `k`/`v` are `[batch, t, kv_heads,
    /// head_dim]` — the refs `KVCache.append` returns, or this layer's own
    /// whole-sequence K/V when there is no cache.
    ///
    /// The controls are independent. `query_positions` defaults to each query's
    /// row index and `kv_lengths` defaults to `t`. Causal attention with unequal
    /// query/key lengths therefore requires explicit query positions.
    pub fn attend(
        self: Self,
        bld: *Builder,
        q: TensorRef,
        k: TensorRef,
        v: TensorRef,
        query_positions: ?TensorRef,
        kv_lengths: ?TensorRef,
    ) Builder.Error!TensorRef {
        const scope = try self.id.enter(bld);
        defer bld.endScope(scope);

        const attn: TensorRef = try bld.attention(
            q,
            k,
            v,
            query_positions,
            kv_lengths,
            self.cfg.scale,
            self.cfg.window,
            self.cfg.attn_logits_soft_cap,
        );

        const dims: []const usize = bld.knownShape(attn) orelse return Builder.Error.InvalidArgument;
        if (dims.len != 4) return Builder.Error.InvalidArgument;
        // Heads are merged back into one feature axis; batch and sequence carry over.
        const flat: TensorRef = try bld.reshapeDims(attn, &[_]Builder.Dim{
            try bld.dimAt(attn, 0),
            try bld.dimAt(attn, 1),
            .{ .size = dims[2] * dims[3] },
        });
        return self.o_proj.forward(bld, flat);
    }
};

/// Relative-position multi-head self-attention (Transformer-XL / Conformer style).
///
/// Inputs are head-second `[batch, heads, time, head_dim]`. `pos_emb` is the
/// already-projected `[heads, 2*T-1, head_dim]` table (see `initializers.relPosTable`),
/// and `pos_bias_u`/`pos_bias_v` are the two `[heads, head_dim]` learned biases.
pub const RelPosSelfAttention = struct {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,

    pos_emb: TensorRef,
    pos_bias_u: TensorRef,
    pos_bias_v: TensorRef,
    mask: ?TensorRef,

    heads: usize,
    head_dim: usize,
    scale: f32,
    window: Builder.AttentionWindow = .full,
    relative_zero_index: usize,
    attn_logits_soft_cap: f32 = 0,
    id: LayerName = .{},

    const Self = @This();

    /// A tree mixing sub-layers and plain parameters: the four projections are
    /// `Linear`s, the position table and biases are tensors of this layer's own.
    pub const Weights = struct {
        linear_q: Linear.Weights,
        linear_k: Linear.Weights,
        linear_v: Linear.Weights,
        linear_out: Linear.Weights,
        pos_emb: Tensor,
        pos_bias_u: Tensor,
        pos_bias_v: Tensor,
        /// An additive mask is a parameter like any other, so it simply may or may
        /// not be in the source.
        mask: ?Tensor = null,
    };

    /// No `heads`: it is `pos_emb`'s dim 0, like `head_dim` is its dim 2.
    pub const Config = struct {
        /// Logit scale, usually `1 / sqrt(head_dim)`.
        scale: f32,
        window: Builder.AttentionWindow = .full,
        /// Null selects the middle row, suitable for symmetric full tables.
        relative_zero_index: ?usize = null,
        attn_logits_soft_cap: f32 = 0,
    };

    pub const Options = struct {
        name: ?[]const u8 = null,
    };

    pub fn bind(bld: *Builder, params: anytype, cfg: Config, opts: Options) BindError!Self {
        if (!std.math.isFinite(cfg.scale) or !std.math.isFinite(cfg.attn_logits_soft_cap) or cfg.attn_logits_soft_cap < 0) return error.InvalidArgument;

        var src = state_mod.binding(Weights, params);
        const p: Params = src.params();

        const id, const scope = try LayerName.open(Self, bld, p.name orelse opts.name);
        defer bld.endScope(scope);

        const pos_emb = try p.get(bld, Weights, .pos_emb);
        const pos_bias_u = try p.get(bld, Weights, .pos_bias_u);
        const pos_bias_v = try p.get(bld, Weights, .pos_bias_v);

        // `heads` and `head_dim` come off `pos_emb` ([heads, P, head_dim]); the biases
        // must agree with it. Nothing declares the head count separately, so nothing
        // can disagree with the weights.
        const pe = bld.knownShape(pos_emb) orelse return error.InvalidArgument;
        if (pe.len != 3 or pe[0] == 0 or pe[2] == 0) return error.InvalidArgument;
        for ([_]TensorRef{ pos_bias_u, pos_bias_v }) |bias| {
            const bs = bld.knownShape(bias) orelse return error.InvalidArgument;
            if (bs.len != 2 or bs[0] != pe[0] or bs[1] != pe[2]) return error.InvalidArgument;
        }

        return .{
            .q_proj = try Linear.bind(bld, p.child(Weights, .linear_q), .{}),
            .k_proj = try Linear.bind(bld, p.child(Weights, .linear_k), .{}),
            .v_proj = try Linear.bind(bld, p.child(Weights, .linear_v), .{}),
            .o_proj = try Linear.bind(bld, p.child(Weights, .linear_out), .{}),
            .pos_emb = pos_emb,
            .pos_bias_u = pos_bias_u,
            .pos_bias_v = pos_bias_v,
            .mask = try p.getOpt(bld, Weights, .mask),
            .heads = pe[0],
            .head_dim = pe[2],
            .scale = cfg.scale,
            .window = cfg.window,
            .relative_zero_index = cfg.relative_zero_index orelse pe[1] / 2,
            .attn_logits_soft_cap = cfg.attn_logits_soft_cap,
            .id = id,
        };
    }

    /// Q/K/V for this step, each `[batch, time, heads, head_dim]`.
    pub const Projected = struct { q: TensorRef, k: TensorRef, v: TensorRef };

    /// Run the three projections and split them into heads, without attending.
    ///
    /// Split out for the same reason `Attention.project` is: a streaming
    /// encoder prepends cached K/V from earlier chunks between projection and
    /// attention, so the two cannot be one step.
    pub fn project(self: Self, bld: *Builder, x: TensorRef) Builder.Error!Projected {
        const scope = try self.id.enter(bld);
        defer bld.endScope(scope);

        const dims: []const usize = bld.knownShape(x) orelse return Builder.Error.InvalidArgument;
        if (dims.len != 3) return Builder.Error.InvalidArgument;

        // Batch and time carry whatever freedom `x` has; see `Attention.project`.
        const head_shape = [_]Builder.Dim{
            try bld.dimAt(x, 0),
            try bld.dimAt(x, 1),
            .{ .size = self.heads },
            .{ .size = self.head_dim },
        };
        return .{
            .q = try bld.reshapeDims(try self.q_proj.forward(bld, x), &head_shape),
            .k = try bld.reshapeDims(try self.k_proj.forward(bld, x), &head_shape),
            .v = try bld.reshapeDims(try self.v_proj.forward(bld, x), &head_shape),
        };
    }

    /// Attend `q` against `k`/`v`, then apply the output projection. `k`/`v` may be
    /// longer than `q` along time — that is what attending over cached history is.
    pub fn attend(self: Self, bld: *Builder, q: TensorRef, k: TensorRef, v: TensorRef) Builder.Error!TensorRef {
        const scope = try self.id.enter(bld);
        defer bld.endScope(scope);

        const attn: TensorRef = try bld.relPosMHA(
            q,
            k,
            v,
            self.pos_emb,
            self.pos_bias_u,
            self.pos_bias_v,
            self.mask,
            self.scale,
            self.window,
            self.relative_zero_index,
            self.attn_logits_soft_cap,
        );
        const flat: TensorRef = try bld.reshapeDims(attn, &[_]Builder.Dim{
            try bld.dimAt(attn, 0),
            try bld.dimAt(attn, 1),
            .{ .size = self.heads * self.head_dim },
        });
        return self.o_proj.forward(bld, flat);
    }

    /// `x` is `[batch, time, model_dim]`; returns the same shape.
    pub fn forward(self: Self, bld: *Builder, x: TensorRef) Builder.Error!TensorRef {
        const p = try self.project(bld, x);
        return self.attend(bld, p.q, p.k, p.v);
    }
};
