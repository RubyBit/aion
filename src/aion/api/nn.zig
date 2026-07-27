// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Neural-network layers built on top of `api.Builder`.
//!
//! Design goals:
//! - Keep the core API graph-hidden and tensor-first.
//! - `bind` registers parameters once; `forward` wires activations and allocates
//!   nothing at runtime.
//! - Parameters get semantic, hierarchical debug names (`Block#0/attn#0/weight`),
//!   which is what makes load/swap-by-name usable on a real model.
//!
//! Everything takes the `Builder`. It carries the `Context`, so a layer can
//! synthesize the constants some ops require as operands — the identity `gamma`/
//! `beta` of a scale-only norm, or the one-element vector a scalar multiply needs —
//! and the `Builder` caches them, so a 35-layer model binds each one once.
//!
//! Weight layouts are Aion's, not any framework's: dense weights are matmul-B
//! ordered (`[in, out]`), conv1d is `[k, c_in/groups, c_out]` (NLC), conv2d is
//! `[k_h, k_w, c_in/groups, c_out]` (NHWC). Converting a foreign checkpoint into
//! that order is the converter's job.

const layer_mod = @import("nn/layer.zig");
const linear_mod = @import("nn/linear.zig");
const norm_mod = @import("nn/norm.zig");
const conv_mod = @import("nn/conv.zig");
const recurrent_mod = @import("nn/recurrent.zig");
const activation_mod = @import("nn/activation.zig");
const container_mod = @import("nn/container.zig");
const mlp_mod = @import("nn/mlp.zig");
const attention_mod = @import("nn/attention.zig");
const functional_mod = @import("nn/functional.zig");
const initializers_mod = @import("nn/initializers.zig");
const state_mod = @import("nn/state.zig");

// --- shared handles ---------------------------------------------------------
pub const Builder = layer_mod.Builder;
pub const TensorRef = layer_mod.TensorRef;
pub const Tensor = layer_mod.Tensor;
pub const Context = layer_mod.Context;
pub const LayerName = layer_mod.LayerName;

// --- projections / lookup ---------------------------------------------------
pub const Linear = linear_mod.Linear;
pub const Embedding = linear_mod.Embedding;

// --- normalization ----------------------------------------------------------
pub const LayerNorm = norm_mod.LayerNorm;
pub const RMSNorm = norm_mod.RMSNorm;
pub const NormOptions = norm_mod.Options;

// --- convolution ------------------------------------------------------------
pub const Conv1D = conv_mod.Conv1D;
pub const Conv2D = conv_mod.Conv2D;
pub const depthwise = conv_mod.depthwise;
pub const ResBlock2D = conv_mod.ResBlock2D;
pub const PadMode = conv_mod.PadMode;

// --- recurrent --------------------------------------------------------------
pub const LSTMCell = recurrent_mod.LSTMCell;
pub const LSTMState = recurrent_mod.LSTMState;

// --- activations ------------------------------------------------------------
pub const Activation = activation_mod.Activation;
pub const ActivationKind = activation_mod.Kind;
pub const relu = activation_mod.relu;
pub const gelu = activation_mod.gelu;
pub const silu = activation_mod.silu;
pub const sigmoid = activation_mod.sigmoid;
pub const tanh = activation_mod.tanh;
pub const log = activation_mod.log;
pub const softmaxLastDim = activation_mod.softmaxLastDim;

// --- containers -------------------------------------------------------------
pub const Sequential = container_mod.Sequential;
pub const Residual = container_mod.Residual;

// --- feed-forward blocks ----------------------------------------------------
pub const FeedForward = mlp_mod.FeedForward;
pub const GatedMLP = mlp_mod.GatedMLP;
pub const GLU = mlp_mod.GLU;

// --- attention --------------------------------------------------------------
pub const RoPE = attention_mod.RoPE;
pub const KVCache = attention_mod.KVCache;
pub const CachedAttention = attention_mod.CachedAttention;
pub const RelPosSelfAttention = attention_mod.RelPosSelfAttention;

// --- constant-bearing helpers ----------------------------------------------
pub const scale = functional_mod.scale;
pub const shift = functional_mod.shift;
pub const softcap = functional_mod.softcap;

// --- parameter sources / introspection ---------------------------------------
// Every layer has exactly one constructor, `bind(bld, params, opts)`, and declares
// its parameters as a type: `pub const Weights`, where a `Tensor` field is a
// parameter and a struct field is a sub-layer. So the call site is a typed literal
// the compiler checks — no path strings anywhere:
//
//     const fc = try nn.Linear.bind(&bld, .{ .weight = w, .bias = b }, .{});
//
//     const mlp = try nn.GatedMLP.bind(&bld, .{
//         .gate_up_proj = .{ .weight = gu },
//         .down_proj    = .{ .weight = d },
//     }, .{ .act = .silu });
//
// A required weight has no default, so omitting one does not compile; a misspelled
// field is a compile error rather than a bind-time failure; and `bind` reaches its
// own children through the same type (`p.child(Weights, .down_proj)`), so the field
// name and the graph path cannot drift.
//
// The other source is a loaded package, which resolves those same paths:
//
//     var pkg = nn.Package.init(&weights);
//     const mlp = try nn.GatedMLP.bind(&bld, pkg.at("mlp"), .{});
//
// A layer therefore cannot support one source and not the other. `Source(W)` is a
// weights tree flattened once and shared across several layers — a whole model's,
// say; a single layer never needs one, since `bind` builds it on its own frame.
// `Params` is the type-erased form all three arrive as, and writing a `resolveFn`
// is how you plug in a source of your own.
pub const Params = state_mod.Params;
pub const Source = state_mod.Source;
pub const Package = state_mod.Package;
/// What a layer's `bind` calls to accept all three spellings. A model of your own
/// composed out of `nn` layers uses it the same way they do.
pub const binding = state_mod.binding;
pub const BindError = state_mod.BindError;
pub const forEachParam = state_mod.forEachParam;
pub const collectParamNames = state_mod.collectParamNames;

// --- host-side constant builders (tables, masks) ----------------------------
pub const initializers = initializers_mod;
pub const sinusoidalRelPos = initializers_mod.sinusoidalRelPos;
pub const causalMask = initializers_mod.causalMask;
pub const slidingWindowMask = initializers_mod.slidingWindowMask;
pub const chunkedLimitedMask = initializers_mod.chunkedLimitedMask;
