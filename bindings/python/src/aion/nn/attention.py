# SPDX-License-Identifier: Apache-2.0
"""Attention layers, mirroring `src/aion/api/nn/attention.zig`."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from ..dtype import float16, float32
from ..types import AttentionWindow, DTypeLike
from ..builder import Builder, TensorRef, WeightData
from ..enums import AionInputRoleKind
from .layers import Linear, _shape_of
from .module import Module, Parameter, builder_of


class RoPE(Module):
    """Rotary position embedding over `[batch, seq, heads, head_dim]`.

    Stateless: the rotation comes from `positions`, not a parameter, so this
    exists to name the hyperparameters rather than to hold weights.
    """

    def __init__(
        self,
        *,
        base_frequency: float = 10000.0,
        scale_factor: float = 1.0,
        rope_proportion: float = 1.0,
    ) -> None:
        self.base_frequency = float(base_frequency)
        self.scale_factor = float(scale_factor)
        self.rope_proportion = float(rope_proportion)

    def forward(self, x: TensorRef, positions: TensorRef) -> TensorRef:
        return builder_of(x).rope1d(
            x, positions,
            base_frequency=self.base_frequency,
            scale_factor=self.scale_factor,
            rope_proportion=self.rope_proportion,
        )


class AxialRoPE(Module):
    """Independent 1-D RoPE per spatial axis, over consecutive head-dim slices.

    A 2-D image patch rotates its first half by the patch's x position and its
    second half by its y. `head_dim` splits evenly across `positions`, and each
    slice is one ordinary `rope1d` — the axes never mix.
    """

    def __init__(self, *, base_frequency: float = 10000.0) -> None:
        self.base_frequency = float(base_frequency)

    def forward(self, x: TensorRef, positions: Sequence[TensorRef]) -> TensorRef:
        if not positions:
            raise ValueError("AxialRoPE needs at least one position tensor")
        b = builder_of(x)
        head_dim = int(x.shape[-1])
        if head_dim % len(positions) != 0:
            raise ValueError(
                f"head_dim {head_dim} does not split evenly across {len(positions)} axes"
            )
        width = head_dim // len(positions)
        return b.concat(
            [
                b.rope1d(b.slice_last_dim(x, i * width, width), pos,
                         base_frequency=self.base_frequency)
                for i, pos in enumerate(positions)
            ],
            axis=-1,
        )


class KVCache:
    """A `[batch, capacity, kv_heads, head_dim]` cache the runtime carries.

    Bundles the three things a cache needs, which converters wire by hand and
    which are easy to get subtly wrong:
      1. the cache *input* (a public input the runtime owns),
      2. the `sequence_append` writing this step's K/V at `write_index`,
      3. the role declaration + output alias that make it persistent.

    Roles and aliases are consumed by compile/export, not by the Builder, so
    `role`/`alias` hand them back for the caller to apply. Not a `Module`: it owns
    a graph input, not parameters.
    """

    def __init__(
        self,
        b: Builder,
        batch: int,
        kv_heads: int,
        head_dim: int,
        *,
        name: Optional[str] = None,
        dtype: DTypeLike = float16,
        capacity: int = 0,
        capacity_symbol: Optional[str] = None,
        growable: bool = False,
    ) -> None:
        if min(batch, kv_heads, head_dim) <= 0:
            raise ValueError("batch, kv_heads and head_dim must be > 0")
        # A symbolic axis still needs a concrete authoring placeholder for eager
        # shape inference; the loader overrides it.
        if capacity_symbol is None and capacity <= 0:
            raise ValueError("give a fixed capacity or a capacity_symbol")
        cap = max(1, capacity) if capacity_symbol is not None else capacity

        self._b = b
        self.dtype: DTypeLike = dtype
        self.capacity_symbol = capacity_symbol
        self.growable = bool(growable)

        dyn = {1: capacity_symbol} if capacity_symbol else None
        self.input = b.input((batch, cap, kv_heads, head_dim), dtype=dtype, dynamic=dyn)
        if name:
            b.name(self.input, name)
        self.current = self.input

    def append(self, kv: TensorRef, write_index: TensorRef) -> TensorRef:
        """Append this step's K or V, casting to the cache dtype first.

        `kv` is `[batch, seq, kv_heads, head_dim]`, like the cache.
        Advances `current`, so repeated calls chain.
        """
        from ..dtype import normalize_dtype

        want = normalize_dtype(self.dtype)
        cast_kv = kv if kv.dtype == want else self._b.cast(kv, want)
        self.current = self._b.sequence_append(self.current, cast_kv, write_index)
        return self.current

    def role(self) -> dict:
        """kwargs for `Builder.add_input_role`, so the runtime allocates it."""
        return {
            "value": self.input,
            "kind": AionInputRoleKind.AION_ROLE_SEQUENCE_CACHE,
            "axis": 1,
            "capacity_symbol": self.capacity_symbol,
            "zero_init": True,
            "growable": self.growable,
        }

    def declare(self) -> None:
        """Apply the role and the write-back alias to this cache's builder.

        The alias makes the updated cache persist across runs; `current` must also
        be marked as an output, since an alias is resolved by output index.
        """
        self._b.add_input_role(**self.role())
        self._b.add_output_alias(self.input, self.current)


class Attention(Module):
    """Grouped-query attention — over a KV cache (decode) or a plain sequence
    (prefill, encoders); `attend` takes the cache indices or omits them.

    Q, K and V are separate projections, the way checkpoints ship them;
    concatenating them into one wide projection is a fusion the compiler performs
    (`opt/fuse_horizontal_matmul`), not a weight layout to pick here.

    `k_proj`/`v_proj` are optional, all-or-nothing: omitting them means this layer
    does not write a cache, it only reads one another layer produced. Gemma 4 shares
    one KV cache across a run of layers, where the first projects K/V and the rest
    attend over it — `attend` takes the caches as arguments, so sharing needs nothing
    else. A reader layer's `project` returns `(q, None, None)`.

    Grouped-query attention falls out of `kv_heads < heads`; the op requires
    `heads % kv_heads == 0`.
    """

    def __init__(
        self,
        q_proj: WeightData,
        o_proj: WeightData,
        *,
        k_proj: Optional[WeightData] = None,
        v_proj: Optional[WeightData] = None,
        heads: int,
        kv_heads: int,
        head_dim: int,
        scale: float,
        window: AttentionWindow = AttentionWindow.CAUSAL,
        attn_logits_soft_cap: float = 0.0,
        name: Optional[str] = None,
        dtype: DTypeLike = float32,
    ) -> None:
        if heads <= 0 or kv_heads <= 0 or head_dim <= 0:
            raise ValueError("heads, kv_heads and head_dim must be > 0")
        if heads % kv_heads != 0:
            raise ValueError(f"heads ({heads}) must be a multiple of kv_heads ({kv_heads})")

        self._layer_name = name
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.scale = float(scale)
        self.window = window
        self.attn_logits_soft_cap = float(attn_logits_soft_cap)

        # K and V go together: a layer with one but not the other cannot fill a
        # cache, and would fail later with a confusing shape error instead.
        if (k_proj is None) != (v_proj is None):
            raise ValueError("pass both k_proj and v_proj, or neither")

        self.q_proj = Linear(q_proj, dtype=dtype)
        self.k_proj = Linear(k_proj, dtype=dtype) if k_proj is not None else None
        self.v_proj = Linear(v_proj, dtype=dtype) if v_proj is not None else None
        self.o_proj = Linear(o_proj, dtype=dtype)

    def project(
        self, x: TensorRef
    ) -> tuple[TensorRef, Optional[TensorRef], Optional[TensorRef]]:
        """Project and reshape to attention layout, without touching the cache.

        Split out because every modern transformer inserts Q/K norms or RoPE
        between projection and attention — so all three come back in the layout
        those operate on, `[b, s, h, d]` (k/v with `kv_heads`). This is also the
        cache and attention layout.

        k/v are `None` on a layer that only reads a shared cache.
        """
        b = builder_of(x)
        with self._scoped(b):
            q = self.q_proj(x)
            k = self.k_proj(x) if self.k_proj is not None else None
            v = self.v_proj(x) if self.v_proj is not None else None

            # `dims`, not `shape`: batch and sequence are carried over as authored,
            # so an axis the caller declared free stays free through the reshape
            # instead of freezing at its placeholder size. The layer is never told
            # which axes those are — it reads them off its input.
            dims = q.dims
            batch = dims[0]
            seq = dims[1] if len(dims) >= 3 else 1
            kv_shape = (batch, seq, self.kv_heads, self.head_dim)
            return (
                b.reshape(q, (batch, seq, self.heads, self.head_dim)),
                b.reshape(k, kv_shape) if k is not None else None,
                b.reshape(v, kv_shape) if v is not None else None,
            )

    def attend(
        self,
        q: TensorRef,
        k: TensorRef,
        v: TensorRef,
        query_positions: Optional[TensorRef] = None,
        kv_lengths: Optional[TensorRef] = None,
    ) -> TensorRef:
        """Attend `q` against `k`/`v`, then apply the output projection.

        The controls are independent. `query_positions` defaults to each query's
        row index and `kv_lengths` defaults to `t`. Causal attention with unequal
        query/key lengths therefore requires explicit query positions. K/V are
        always `[batch, t, kv_heads, head_dim]`.
        """
        b = builder_of(q)
        with self._scoped(b):
            opts = {
                "scale": self.scale,
                "window": self.window,
                "attn_logits_soft_cap": self.attn_logits_soft_cap,
            }
            attn = b.attention(
                q, k, v,
                query_positions=query_positions,
                kv_lengths=kv_lengths,
                **opts,
            )
            # Heads merge back into one feature axis; batch and time carry over.
            d = attn.dims
            sizes = attn.shape
            return self.o_proj(
                b.reshape(attn, (d[0], d[1], sizes[2] * sizes[3]))
            )


class RelPosSelfAttention(Module):
    """Relative-position multi-head self-attention (Transformer-XL / Conformer).

    `pos_emb` is the already-projected `[heads, P, head_dim]` table — projecting
    the sinusoidal table through `linear_pos` is a fold the converter does once, on
    the host — and `pos_bias_u`/`pos_bias_v` are the two `[heads, head_dim]` learned
    biases. `head_dim` comes off `pos_emb`, so it cannot disagree with it.

    `mask` is a value rather than a weight: it is geometry, identical for every layer,
    so the caller binds it once and hands the same one to all of them.
    """

    def __init__(
        self,
        q: WeightData,
        k: WeightData,
        v: WeightData,
        o: WeightData,
        pos_emb: WeightData,
        pos_bias_u: WeightData,
        pos_bias_v: WeightData,
        *,
        scale: float,
        mask: Optional[TensorRef] = None,
        window: AttentionWindow = AttentionWindow.FULL,
        relative_zero_index: Optional[int] = None,
        attn_logits_soft_cap: float = 0.0,
        name: Optional[str] = None,
        dtype: DTypeLike = float32,
    ) -> None:
        self._layer_name = name

        # `heads` and `head_dim` are pos_emb's dims 0 and 2; neither is declared
        # separately, so neither can disagree with the weights.
        pe = _shape_of(pos_emb)
        if len(pe) != 3:
            raise ValueError(f"pos_emb must be [heads, P, head_dim], got {pe}")

        self.heads = int(pe[0])
        self.head_dim = int(pe[2])
        self.scale = float(scale)
        self.mask = mask
        self.window = window
        self.relative_zero_index = int(pe[1] // 2 if relative_zero_index is None else relative_zero_index)
        self.attn_logits_soft_cap = float(attn_logits_soft_cap)

        self.q_proj = Linear(q, dtype=dtype)
        self.k_proj = Linear(k, dtype=dtype)
        self.v_proj = Linear(v, dtype=dtype)
        self.o_proj = Linear(o, dtype=dtype)
        self.pos_emb = Parameter(pos_emb, dtype=float32)
        self.pos_bias_u = Parameter(pos_bias_u, dtype=float32)
        self.pos_bias_v = Parameter(pos_bias_v, dtype=float32)

    def project(self, x: TensorRef) -> tuple[TensorRef, TensorRef, TensorRef]:
        """Q/K/V split into heads, `[batch, time, heads, head_dim]` each.

        Split from `attend` for the same reason `Attention.project` is: a
        streaming encoder prepends cached K/V from earlier chunks in between, so the
        two cannot be one step.
        """
        b = builder_of(x)
        with self._scoped(b):
            # As authored, so a free time axis stays free across the head split.
            head_shape = (*x.dims[:2], self.heads, self.head_dim)
            return (
                b.reshape(self.q_proj(x), head_shape),
                b.reshape(self.k_proj(x), head_shape),
                b.reshape(self.v_proj(x), head_shape),
            )

    def attend(self, q: TensorRef, k: TensorRef, v: TensorRef) -> TensorRef:
        """Attend and project out. `k`/`v` may be longer than `q` along time — that
        is what attending over cached history is."""
        b = builder_of(q)
        with self._scoped(b):
            att = b.relpos_mha(
                q, k, v,
                self.pos_emb.value(b, "pos_emb"),
                self.pos_bias_u.value(b, "pos_bias_u"),
                self.pos_bias_v.value(b, "pos_bias_v"),
                self.mask,
                scale=self.scale,
                window=self.window,
                relative_zero_index=self.relative_zero_index,
                attn_logits_soft_cap=self.attn_logits_soft_cap,
            )
            return self.o_proj(
                b.reshape(att, (*att.dims[:2], self.heads * self.head_dim))
            )

    def forward(self, x: TensorRef) -> TensorRef:
        """`x` is `[batch, time, heads * head_dim]`; returns the same shape."""
        return self.attend(*self.project(x))
