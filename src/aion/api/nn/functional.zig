// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Stateless graph helpers that materialize a constant operand via the Builder.
const std = @import("std");

const layer_mod = @import("layer.zig");

pub const Builder = layer_mod.Builder;
pub const TensorRef = layer_mod.TensorRef;

/// `x * k`. The scalar is bound as a **one-element** operand to the broadcast op,
/// so its cost is independent of `x`'s size (and the constant is shared across
/// every use of the same `k` in the model).
///
/// Requires rank >= 2, which the broadcast op enforces.
pub fn scale(bld: *Builder, x: TensorRef, k: f32) Builder.Error!TensorRef {
    if (k == 1.0) return x;
    return bld.broadcastLastDim(.mul, x, try bld.constant(k));
}

/// `x + k`, same one-element-constant treatment as `scale`.
pub fn shift(bld: *Builder, x: TensorRef, k: f32) Builder.Error!TensorRef {
    if (k == 0.0) return x;
    return bld.broadcastLastDim(.add, x, try bld.constant(k));
}

/// `cap * tanh(x / cap)` — a smooth clamp to `(-cap, cap)`.
///
/// Gemma applies this to attention logits and to the final logits; `cap <= 0`
/// means "disabled" and returns `x` untouched, matching how the attention op reads
/// its own soft-cap attribute.
pub fn softcap(bld: *Builder, x: TensorRef, cap: f32) Builder.Error!TensorRef {
    if (!(cap > 0.0)) return x;
    if (!std.math.isFinite(cap)) return Builder.Error.InvalidArgument;

    const scaled: TensorRef = try scale(bld, x, 1.0 / cap);
    const squashed: TensorRef = try bld.tanh(scaled);
    return scale(bld, squashed, cap);
}
