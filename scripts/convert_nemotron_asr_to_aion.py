# Copyright (c) 2026 Angelos-Ermis Mangos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Convert NVIDIA Nemotron streaming ASR (.nemo) to a self-contained q8_0 `.aion`.

Both build modes emit ONE full model — in-graph log-mel front-end + FastConformer
encoder + in-graph greedy RNNT decode — that takes raw 16 kHz audio and returns
token ids. They differ only in how audio is fed:

  (default)    offline whole-clip model: one fixed audio window (--frames mel
               frames) per run. Full attention unless an attention config is set.
  --streaming  cache-aware streaming model: one small audio window per chunk;
               all state (per-layer K/V + conv caches, RNNT decode state) is
               threaded inside the runtime via IoAlias, so a trivial driver
               streams indefinitely.

The attention config is [--att-left, --att-right] (chunk = att_right + 1); the
checkpoint was trained on right contexts 13/6/1/0 with left context 70.

The tokenizer (`nemotron_tokenizer.model`) is extracted next to the output, and
all runtime parameters (chunk, window/buffer sizes, ...) are recorded in package
metadata — the example scripts auto-configure from it.

Weight/quant policy (mirrors convert_gemma4_e2b_to_aion.py):
- dense matmul weights (FF, q/k/v/out proj, subsampling `out`) -> q8_0 (B `[1,K,N]`).
- LayerNorm gamma/beta, conv weights/biases, pos_bias_u/_v, pos_emb -> f32.

Authored via the C-ABI `aion.Builder`; the Zig core serializes through
`Builder.export()` (no pure-Python writer).

Run:  uv run --project bindings/python \
        python scripts/convert_nemotron_asr_to_aion.py <model.nemo> <out.aion> [--streaming]
"""

from __future__ import annotations

import argparse
import math
import os
import tarfile
import tempfile
from typing import IO, Dict, List, Optional, Sequence, Tuple

import numpy as np

import aion
from aion import Builder, TensorRef, nn
from aion.enums import AionInputRoleKind as RK
from aion.nn import functional as F

# ------------------------------- architecture --------------------------------

D_MODEL = 1024
N_LAYERS = 24
N_HEADS = 8
HEAD_DIM = 128  # D_MODEL // N_HEADS
D_FF = 4096
CONV_K = 9
SUB_CH = 256
FEAT_IN = 128
FREQ_OUT = 17  # 128 -> 65 -> 33 -> 17 under total-3 padding
LN_EPS = 1.0e-5
ATT_SCALE = 1.0 / math.sqrt(HEAD_DIM)

# Log-mel front-end (NeMo AudioToMelSpectrogramPreprocessor, normalize="NA"=none).
SAMPLE_RATE = 16000
N_FFT = 512
WIN_LEN = 400
HOP_LEN = 160
PREEMPH = 0.97
LOG_GUARD = 2.0 ** -24


def mel_samples(t_mel: int) -> int:
    """Audio sample count whose center-STFT yields exactly t_mel frames."""
    return (t_mel - 1) * HOP_LEN


Shape = Tuple[int, ...]


def calc_length(length: int) -> int:
    """NeMo subsampling output length: floor(L/2)+1 applied 3x (factor 8, total pad 3)."""
    for _ in range(3):
        length = (length // 2) + 1
    return length


# ------------------------------- weight loader -------------------------------


def _tar_member(tf: tarfile.TarFile, member: str) -> IO[bytes]:
    """`extractfile` returns None for non-regular members; we only ask for files."""
    f = tf.extractfile(member)
    if f is None:
        raise FileNotFoundError(f"{member}: not a regular file in the archive")
    return f


class _Loader:
    def __init__(self, nemo_path: str) -> None:
        self._tmp = tempfile.mkdtemp(prefix="nemotron_")
        with tarfile.open(nemo_path, "r") as tf:
            names = tf.getnames()
            ck = next(m for m in names if m.endswith("model_weights.ckpt"))
            ckpt_path = os.path.join(self._tmp, "model_weights.ckpt")
            with _tar_member(tf, ck) as src, open(ckpt_path, "wb") as dst:
                while True:
                    chunk = src.read(1 << 24)
                    if not chunk:
                        break
                    dst.write(chunk)
            self.tokenizer_path: Optional[str] = None
            for m in names:
                if os.path.basename(m).endswith("tokenizer.model"):
                    self.tokenizer_path = os.path.join(self._tmp, "tokenizer.model")
                    with _tar_member(tf, m) as src, open(self.tokenizer_path, "wb") as dst:
                        dst.write(src.read())
        import torch  # lazy: only needed for conversion

        obj = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self._sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj

    def get(self, name: str) -> np.ndarray:
        return self._sd[name].float().numpy().astype(np.float32, copy=False)


# ----------------------- C-ABI Builder authoring helpers ----------------------
# The converter drives the C-ABI `aion.Builder`; these small helpers cover the
# two weight-packing idioms (q8_0 matmul-B and q8_0 embedding) and layer norm.


def q8b(b: Builder, w_torch: np.ndarray) -> TensorRef:
    """PyTorch linear `[out, in]` -> a q8_0 matmul-B param bound as `[1, in, out]`.

    The encoder's weights go through `nn` (see `mm_b`), which stores them at their
    natural `[K, N]` rank. This remains for the RNN-T decode loop, which is written
    directly on the Builder — control flow is the low-level escape hatch — and whose
    graph was authored against the rank-3 form.
    """
    w_t = np.ascontiguousarray(w_torch.T)  # [K, N]
    k, n = int(w_t.shape[0]), int(w_t.shape[1])
    return b.param(w_t.reshape(1, k, n), dtype=aion.q8_0, shape=(1, k, n))


def q8_embed(b: Builder, table: np.ndarray) -> TensorRef:
    """Embedding table `[V, D]` -> q8_0 param blocked along the feature dim D."""
    v, d = int(table.shape[0]), int(table.shape[1])
    return b.param(
        np.ascontiguousarray(table),
        dtype=aion.q8_0,
        shape=(v, d),
        quant_axis=1,
    )


def mm_b(w_torch: np.ndarray) -> np.ndarray:
    """A PyTorch linear's `[out, in]` weight in Aion's matmul-B layout `[in, out]`.

    Adapting someone else's layout is a converter's job. Rank 2 is all it needs —
    MatMul broadcasts a `[K, N]` weight into a rank-3 `[B, S, K]` activation — and
    `nn` does the q8_0 quantization, blocking along K.
    """
    return np.ascontiguousarray(w_torch.T)


def ln(loader: _Loader, base: str, name: str) -> nn.LayerNorm:
    """A LayerNorm from the checkpoint, under this model's epsilon."""
    return nn.LayerNorm(
        loader.get(f"{base}.weight"), loader.get(f"{base}.bias"),
        eps=LN_EPS, name=name,
    )


class ConformerLayer(nn.Module):
    """One FastConformer block: half-scaled FF, rel-pos MHSA, conv module, half-scaled
    FF, output norm — the macaron layout.

    Held as a module so the batch encoder and the cache-aware streaming encoder share
    one definition. They differ only in what happens *between* projection and
    attention (streaming prepends cached K/V) and around the depthwise conv (streaming
    prepends a cached left context), which is why those two steps are the caller's and
    everything else lives here.
    """

    def __init__(self, loader: _Loader, li: int, *, t_kv: int,
                 mask: Optional[TensorRef] = None, conv_pad_left: int = 0,
                 window: aion.AttentionWindow = aion.AttentionWindow.FULL) -> None:
        # The caller opens a `layers.{li}` scope around the body, so every parameter
        # below lands under it without this type having to know its own index.
        pf = f"encoder.layers.{li}"
        get = loader.get

        # The sinusoidal table projected through `linear_pos` is fixed, so it is
        # folded on the host once per layer rather than emitted as two ops.
        p_len = 2 * t_kv - 1
        pe_proj = ((sinusoidal_pe(t_kv, D_MODEL) @ get(f"{pf}.self_attn.linear_pos.weight").T)
                   .reshape(p_len, N_HEADS, HEAD_DIM).transpose(1, 0, 2))

        self.norm_feed_forward1 = ln(loader, f"{pf}.norm_feed_forward1", "norm_feed_forward1")
        self.feed_forward1 = nn.FeedForward(
            mm_b(get(f"{pf}.feed_forward1.linear1.weight")),
            mm_b(get(f"{pf}.feed_forward1.linear2.weight")),
                act="silu", dtype=aion.q8_0, name="feed_forward1")

        self.norm_self_att = ln(loader, f"{pf}.norm_self_att", "norm_self_att")
        self.self_attn = nn.RelPosSelfAttention(
            mm_b(get(f"{pf}.self_attn.linear_q.weight")),
            mm_b(get(f"{pf}.self_attn.linear_k.weight")),
            mm_b(get(f"{pf}.self_attn.linear_v.weight")),
            mm_b(get(f"{pf}.self_attn.linear_out.weight")),
            np.ascontiguousarray(pe_proj),
            get(f"{pf}.self_attn.pos_bias_u"),
            get(f"{pf}.self_attn.pos_bias_v"),
            scale=ATT_SCALE, mask=mask, window=window,
                dtype=aion.q8_0, name="self_attn")

        # Pointwise convs are k=1, i.e. matmuls, so they are stored q8_0 and take the
        # fast matmul path; only the depthwise conv stays f32.
        self.norm_conv = ln(loader, f"{pf}.norm_conv", "norm_conv")
        self.conv_glu = nn.GLU(mm_b(get(f"{pf}.conv.pointwise_conv1.weight")[:, :, 0]),
                dtype=aion.q8_0, name="conv/pointwise_conv1")
        # Causal padding in the batch path; the streaming path pads with a cached
        # left context instead, so it asks for none.
        self.conv_dw = nn.DepthwiseConv1D(
            np.transpose(get(f"{pf}.conv.depthwise_conv.weight"), (2, 1, 0)),
            pad_left=conv_pad_left, pad_right=0, name="conv/depthwise_conv")
        # A LayerNorm occupies the batch-norm slot in this checkpoint.
        self.conv_norm = ln(loader, f"{pf}.conv.batch_norm", "conv/batch_norm")
        self.conv_out = nn.Linear(mm_b(get(f"{pf}.conv.pointwise_conv2.weight")[:, :, 0]),
                dtype=aion.q8_0, name="conv/pointwise_conv2")

        self.norm_feed_forward2 = ln(loader, f"{pf}.norm_feed_forward2", "norm_feed_forward2")
        self.feed_forward2 = nn.FeedForward(
            mm_b(get(f"{pf}.feed_forward2.linear1.weight")),
            mm_b(get(f"{pf}.feed_forward2.linear2.weight")),
                act="silu", dtype=aion.q8_0, name="feed_forward2")

        self.norm_out = ln(loader, f"{pf}.norm_out", "norm_out")

    def ff1(self, h: TensorRef) -> TensorRef:
        """The macaron half-step: `h + 0.5 * FF(norm(h))`."""
        return h + F.scale(self.feed_forward1(self.norm_feed_forward1(h)), 0.5)

    def ff2_and_out(self, h: TensorRef) -> TensorRef:
        h = h + F.scale(self.feed_forward2(self.norm_feed_forward2(h)), 0.5)
        return self.norm_out(h)


# ------------------------------ positional emb -------------------------------


def sinusoidal_pe(length: int, d_model: int) -> np.ndarray:
    """NeMo RelPositionalEncoding pe of shape [2*length-1, d_model] (idx 0 = +(length-1))."""
    pos = np.arange(length - 1, -length, -1, dtype=np.float32)[:, None]  # [2L-1, 1]
    div = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model))
    pe = np.zeros((2 * length - 1, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe


# --------------------------------- encoder -----------------------------------


def build_logmel(loader: _Loader, b: Builder, t_mel: int) -> TensorRef:
    """In-graph NeMo log-mel: audio [n,1] -> feat [1, t_mel, 128].

    Matches AudioToMelSpectrogramPreprocessor (normalize="NA"=none): preemphasis
    0.97, center STFT (n_fft 512, win 400 hann padded to 512, hop 160), power,
    mel filterbank, log(x + 2^-24). Verified vs the torch reference at rel ~3e-7.
    """
    n = mel_samples(t_mel)
    bins = N_FFT // 2 + 1
    win400 = loader.get("preprocessor.featurizer.window").astype(np.float32)  # [400] hann
    win512 = np.zeros(N_FFT, np.float32)
    off = (N_FFT - WIN_LEN) // 2
    win512[off:off + WIN_LEN] = win400                       # torch pads win_length->n_fft centered
    fb = np.ascontiguousarray(loader.get("preprocessor.featurizer.fb")[0].astype(np.float32))  # [128,257]

    audio = b.input((n, 1)).rename("audio")
    win_id = b.param(win512)
    fb_b = b.param(np.ascontiguousarray(fb.T)[None])  # f32 B [1,257,128] (K=257 not /32)
    guard = b.param(np.full((FEAT_IN,), LOG_GUARD, np.float32))
    coef = b.param(np.array([PREEMPH], np.float32))

    # Preemphasis y[0]=x[0]; y[t]=x[t]-0.97*x[t-1] (column [n,1] so the 0.97 scalar
    # broadcasts over the trailing dim of size 1).
    x_main = b.slice(audio, (1, 0), (n - 1, 1))
    x_prev = b.slice(audio, (0, 0), (n - 1, 1))
    diff = b.sub(x_main, b.mul(x_prev, coef))      # [n-1,1]
    y = b.concat([b.slice(audio, (0, 0), (1, 1)), diff], axis=0)  # [n,1]
    sig = b.reshape(y, (1, n))                               # [1,n]
    spec = b.stft(sig, win_id, n_fft=N_FFT, hop_length=HOP_LEN, center=True)  # [1,t_mel,514]
    re = b.slice(spec, (0, 0, 0), (1, t_mel, bins))
    im = b.slice(spec, (0, 0, bins), (1, t_mel, bins))
    power = b.add(re * re, im * im)             # [1,t_mel,257]
    mel = power @ fb_b                             # [1,t_mel,128]
    return b.unary("log", b.add(mel, guard))     # [1,t_mel,128]


def build_subsample(loader: _Loader, b: Builder, feat: TensorRef, t_mel: int) -> TensorRef:
    """dw_striding ×8 subsampling: feat [1, t_mel, 128] -> h [1, t_out, 1024]."""
    t_out = calc_length(t_mel)

    def conv2d_w(name):  # torch [cout,cin/g,kh,kw] -> aion NHWC [kh,kw,cin/g,cout]
        return b.param(np.transpose(loader.get(name), (2, 3, 1, 0)))

    pe = "encoder.pre_encode"
    w0 = conv2d_w(f"{pe}.conv.0.weight"); b0 = b.param(loader.get(f"{pe}.conv.0.bias"))
    w2 = conv2d_w(f"{pe}.conv.2.weight"); b2 = b.param(loader.get(f"{pe}.conv.2.bias"))
    w3 = conv2d_w(f"{pe}.conv.3.weight"); b3 = b.param(loader.get(f"{pe}.conv.3.bias"))
    w5 = conv2d_w(f"{pe}.conv.5.weight"); b5 = b.param(loader.get(f"{pe}.conv.5.bias"))
    w6 = conv2d_w(f"{pe}.conv.6.weight"); b6 = b.param(loader.get(f"{pe}.conv.6.bias"))
    out_w = loader.get(f"{pe}.out.weight").reshape(D_MODEL, SUB_CH, FREQ_OUT)             # [out,c,f]
    out_w = np.ascontiguousarray(out_w.transpose(0, 2, 1).reshape(D_MODEL, FREQ_OUT * SUB_CH))  # [out,f*c]
    out_wq = b.param(mm_b(out_w), dtype=aion.q8_0)
    out_b = b.param(loader.get(f"{pe}.out.bias"))

    relu = lambda x: x.relu()
    x = b.reshape(feat, (1, t_mel, FEAT_IN, 1))

    # CausalConv2D pads both spatial dims (top/left=k-1=2, bottom/right=stride-1=1).
    def c2(x, w, bias, groups, k3):
        if k3:
            return b.conv2d(x, w, bias, stride_h=2, stride_w=2, pad_top=2, pad_bottom=1, pad_left=2, pad_right=1, groups=groups)
        return b.conv2d(x, w, bias, stride_h=1, stride_w=1, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, groups=groups)

    x = relu(c2(x, w0, b0, 1, True))
    x = c2(x, w2, b2, SUB_CH, True)
    x = relu(c2(x, w3, b3, 1, False))
    x = c2(x, w5, b5, SUB_CH, True)
    x = relu(c2(x, w6, b6, 1, False))
    x = b.reshape(x, (1, t_out, FREQ_OUT * SUB_CH))  # NHWC flatten (f*256+c)
    return b.add(x @ out_wq, out_b)  # [1, t_out, 1024]


def build_encoder(loader: _Loader, b: Builder, t_mel: int,
                  att_left: Optional[int] = None, att_right: int = 0,
                  feat: Optional[TensorRef] = None) -> TensorRef:
    t_out = calc_length(t_mel)
    p_len = 2 * t_out - 1

    # Optional chunked_limited attention (cache-aware streaming). Frames are grouped
    # into non-overlapping chunks of size att_right+1; every frame in a chunk attends
    # to the whole chunk plus att_left frames before the chunk start (so early-chunk
    # frames get up to att_right lookahead). This matches NeMo
    # att_context_style="chunked_limited", att_context_size=[att_left, att_right].
    #
    # It goes on the op as a window, NOT as an additive [t_out, t_out] mask: the
    # pattern is one contiguous key interval per row, so the kernels can restrict the
    # work to the window (~att_left + chunk keys) instead of scoring all t_out keys and
    # adding -1e9 to nearly all of them. At t_out = 3750 that mask was also a 56 MB
    # parameter in the file.
    att_window = (aion.AttentionWindow.chunked(att_right + 1, att_left)
                  if att_left is not None else aion.AttentionWindow.FULL)

    # ---- in-graph log-mel -> dw_striding subsampling -> h [1, T_out, 1024] ----
    # `feat` lets a caller supply `[1, T_mel, 128]` itself, which is how
    # verify_nemotron_encoder.py checks the encoder against a mel-fed reference.
    if feat is None:
        feat = build_logmel(loader, b, t_mel)
    h = build_subsample(loader, b, feat, t_mel)  # [1, T_out, 1024]

    # ---- 24 conformer layers ----
    for li in range(N_LAYERS):
        with b.scope(f"layers.{li}"):
            layer = ConformerLayer(loader, li, t_kv=t_out, window=att_window,
                                   conv_pad_left=CONV_K - 1)
            h = layer.ff1(h)

            # Q/K/V are three separate matmuls sharing `norm_self_att(h)`; the compiler
            # fuses them into one wide matmul plus slices.
            h = h + layer.self_attn(layer.norm_self_att(h))

            c = layer.conv_glu(layer.norm_conv(h))
            c = layer.conv_norm(layer.conv_dw(c)).silu()
            h = h + layer.conv_out(c)

            h = layer.ff2_and_out(h)

    return h


PRED_HIDDEN = 640
VOCAB = 1024
MAX_SYMBOLS = 10
BLANK = VOCAB  # 1024


def _ingraph_decode(loader: _Loader, b: Builder, enc: TensorRef, t_out: int, max_out: int) -> None:
    """In-graph greedy RNNT decode over `enc [1, t_out, 1024]`. Adds outputs
    out_count/out_tokens + persistent decode state (last/h0/c0/h1/c1) with their
    IoAliases. Named multi-carry Loop, early-exit via `active = frame < t_out`."""
    H = PRED_HIDDEN
    enc2d = b.reshape(enc, (t_out, D_MODEL))     # [t_out, 1024] gather table

    dp = "decoder.prediction"
    embed = q8_embed(b, loader.get(f"{dp}.embed.weight"))

    def lstm_w(li):
        return (
            b.param(loader.get(f"{dp}.dec_rnn.lstm.weight_ih_l{li}").T),
            b.param(loader.get(f"{dp}.dec_rnn.lstm.weight_hh_l{li}").T),
            b.param(loader.get(f"{dp}.dec_rnn.lstm.bias_ih_l{li}")),
            b.param(loader.get(f"{dp}.dec_rnn.lstm.bias_hh_l{li}")),
        )

    w0 = lstm_w(0)
    w1 = lstm_w(1)
    enc_w = q8b(b, loader.get("joint.enc.weight"))
    enc_b = b.param(loader.get("joint.enc.bias"))
    pred_w = q8b(b, loader.get("joint.pred.weight"))
    pred_b = b.param(loader.get("joint.pred.bias"))
    jn_w = q8b(b, loader.get("joint.joint_net.2.weight"))
    jn_b = b.param(loader.get("joint.joint_net.2.bias"))

    one_i = b.param(np.array([1], np.int32), dtype=aion.int32)       # [1]
    blank_i = b.param(np.array([BLANK], np.int32), dtype=aion.int32)
    maxsym_i = b.param(np.array([MAX_SYMBOLS], np.int32), dtype=aion.int32)
    T_i = b.param(np.array([t_out], np.int32), dtype=aion.int32)

    # Decode state as named, separately-typed carries (no packing). Persistent
    # cross-chunk state (last_token, LSTM h/c) is IoAlias'd output->input; the
    # per-run loop bookkeeping (frame/sym/count/active) and the output token
    # buffer are bound fresh each run.
    frame_in = b.input((1,), dtype=aion.int32).rename("frame_in")   # init 0
    sym_in = b.input((1,), dtype=aion.int32).rename("sym_in")       # init 0
    count_in = b.input((1,), dtype=aion.int32).rename("count_in")   # init 0
    active_in = b.input((1,), dtype=aion.int32).rename("active_in") # init 1
    last_in = b.input((1,), dtype=aion.int32).rename("last_in")     # init BLANK
    h0_in = b.input((1, H)).rename("h0_in")
    c0_in = b.input((1, H)).rename("c0_in")
    h1_in = b.input((1, H)).rename("h1_in")
    c1_in = b.input((1, H)).rename("c1_in")
    # ScatterRow treats a rank-1 buffer as a sequence of scalar rows, which is
    # exactly the token-buffer contract. Keep it rank-1 from input to output.
    toks_in = b.input((max_out,), dtype=aion.int32).rename("toks_in")  # init zeros

    b.begin_region()
    # `active` (carry 3) is the continue predicate; the loop only runs the body
    # while frame < t_out, so the gather frame is always in-range (no clamp needed).
    # Gather indices stay rank-2 so lookup results retain batch/time axes.
    enc_frame = b.gather(enc2d, b.reshape(frame_in, (1, 1)), axis=0)    # [1,1,1024]
    emb = b.reshape(b.gather(embed, b.reshape(last_in, (1, 1)), axis=0), (1, H))  # [1,640]

    st0 = b.lstm_cell(emb, h0_in, c0_in, *w0)
    h0o = b.slice(st0, (0, 0), (1, H))
    c0o = b.slice(st0, (0, H), (1, H))
    st1 = b.lstm_cell(h0o, h1_in, c1_in, *w1)
    h1o = b.slice(st1, (0, 0), (1, H))
    c1o = b.slice(st1, (0, H), (1, H))

    fproj = b.add(enc_frame @ enc_w, enc_b)      # [1,1,640]
    gproj = b.add(b.matmul(b.reshape(h1o, (1, 1, H)), pred_w), pred_b)
    sj = b.unary("relu", fproj + gproj)                      # [1,1,640]
    logits = b.add(sj @ jn_w, jn_b)             # [1,1,1025]
    k = b.reshape(b.argmax(logits, axis=2), (1,))                  # [1] i32

    # advance = (k==blank) or (sym>=max_symbols);  emit = 1 - advance.
    advance = b.compare("ge", b.add(b.compare("eq", k, blank_i),
                                    b.compare("ge", sym_in, maxsym_i)), one_i)  # i32 [1]
    emit = b.sub(one_i, advance)                                    # i32 [1]
    emit_f = b.reshape(b.cast(emit, aion.float32), (1,))           # f32 [1]

    def sel_row(keep, take):  # emit ? take : keep, over [1,H]
        diff_c = b.reshape(b.sub(take, keep), (H, 1))             # [H,1]
        scaled = b.reshape(b.mul(diff_c, emit_f), (1, H))  # [1,H]
        return keep + scaled

    # Next state (arithmetic select on named carries).
    frame_next = frame_in + advance                          # +1 on advance
    sym_next = b.mul(sym_in + one_i, emit)                   # sym+1 on emit else 0
    count_next = count_in + emit                             # +1 on emit
    last_next = b.add(last_in, b.mul(b.sub(k, last_in), emit))     # k on emit
    active_next = b.compare("lt", frame_next, T_i)                 # i32 [1] {0,1}
    h0_next, c0_next = sel_row(h0_in, h0o), sel_row(c0_in, c0o)
    h1_next, c1_next = sel_row(h1_in, h1o), sel_row(c1_in, c1o)
    # Always write k at index `count`; only emit advances `count`, so blank
    # writes land at a slot the next emit overwrites and sit outside [:count].
    toks_next = b.scatter_row(toks_in, count_in, k)                    # [max_out] i32

    body = b.end_region([frame_next, sym_next, count_next, active_next, last_next,
                         h0_next, c0_next, h1_next, c1_next, toks_next])
    carries = [frame_in, sym_in, count_in, active_in, last_in,
               h0_in, c0_in, h1_in, c1_in, toks_in]
    outs = b.loop(carries, body, t_out * (MAX_SYMBOLS + 1), cond_carry=3)
    _, _, count_out, _, last_out, h0_out, c0_out, h1_out, c1_out, toks_out = outs

    count_o = count_out
    b.mark_output(count_o, "out_count")                            # i32 [1]
    b.mark_output(toks_out, "out_tokens")                          # i32 [max_out]
    b.mark_output(last_out, "last_out")
    b.mark_output(h0_out, "h0_out")
    b.mark_output(c0_out, "c0_out")
    b.mark_output(h1_out, "h1_out")
    b.mark_output(c1_out, "c1_out")

    # Persistent decode state, io-aliased output->input.
    b.add_output_alias(last_in, last_out)
    b.add_output_alias(h0_in, h0_out)
    b.add_output_alias(c0_in, c0_out)
    b.add_output_alias(h1_in, h1_out)
    b.add_output_alias(c1_in, c1_out)

    # Roles: the LSTM state is generic zero-init recurrent state, so the runtime
    # auto-zeros + carries it. `last_in` is deliberately NOT tagged — it seeds at
    # the (non-zero) blank SOS token and stays caller-bound.
    for v in (h0_in, c0_in, h1_in, c1_in):
        b.add_input_role(v, RK.AION_ROLE_STATE, zero_init=True)


def build_single(loader: _Loader, b: Builder, t_mel: int, max_out: int,
                 att_left: Optional[int] = None, att_right: int = 0) -> int:
    """Encoder + in-graph greedy RNNT decode in one graph. Returns max_out."""
    T = calc_length(t_mel)
    enc = build_encoder(loader, b, t_mel, att_left=att_left, att_right=att_right)  # [1, T, 1024]
    b.mark_output(enc, "encoder_out")
    _ingraph_decode(loader, b, enc, T, max_out)
    return max_out


def _stream_conformer_layers(loader, b, h, att_left, chunk, t_kv, cache_mask, k_in, v_in, conv_in):
    """24 cache-aware conformer layers over chunk `h [1,chunk,1024]`. Returns
    (enc, k_out, v_out, conv_out)."""
    conv_left = CONV_K - 1

    k_out, v_out, conv_out = [], [], []
    for li in range(N_LAYERS):
        # The same layer definition the batch encoder uses. Only the two steps that
        # consult a cache are written out here.
        with b.scope(f"layers.{li}"):
            layer = ConformerLayer(loader, li, t_kv=t_kv, mask=cache_mask)
            h = layer.ff1(h)

            # Project only the new chunk's frames, then prepend the cached K/V. Q/K/V
            # are separate matmuls over a shared operand, which the compiler fuses
            # into one wide matmul — so this no longer pre-concatenates the weights.
            q, k_c, v_c = layer.self_attn.project(layer.norm_self_att(h))
            k = b.concat([k_in[li], k_c], axis=1)                   # [1,t_kv,H,D]
            v = b.concat([v_in[li], v_c], axis=1)
            h = h + layer.self_attn.attend(q, k, v)
            k_out.append(b.slice(k, (0, chunk, 0, 0), (1, att_left, N_HEADS, HEAD_DIM)))  # last att_left
            v_out.append(b.slice(v, (0, chunk, 0, 0), (1, att_left, N_HEADS, HEAD_DIM)))

            # The depthwise conv's left context comes from the cache rather than from
            # padding, so the GLU output is spliced before the conv and re-cached after.
            glu_full = b.concat([conv_in[li], layer.conv_glu(layer.norm_conv(h))], axis=1)
            c = layer.conv_norm(layer.conv_dw(glu_full)).silu()
            h = h + layer.conv_out(c)
            conv_out.append(b.slice(glu_full, (0, chunk, 0), (1, conv_left, D_MODEL)))

            h = layer.ff2_and_out(h)
    return h, k_out, v_out, conv_out


def _stream_cache_io(b, att_left, chunk):
    """Create per-layer K/V (projected) + conv cache inputs + a cache_mask input.
    Returns (cache_mask, k_in, v_in, conv_in)."""
    conv_left = CONV_K - 1
    t_kv = att_left + chunk
    cache_mask = b.input((chunk, t_kv)).rename("cache_mask")
    k_in = [b.input((1, att_left, N_HEADS, HEAD_DIM)).rename(f"kcache_{li}_in") for li in range(N_LAYERS)]
    v_in = [b.input((1, att_left, N_HEADS, HEAD_DIM)).rename(f"vcache_{li}_in") for li in range(N_LAYERS)]
    conv_in = [b.input((1, conv_left, D_MODEL)).rename(f"conv_cache_{li}_in") for li in range(N_LAYERS)]
    return cache_mask, k_in, v_in, conv_in


def _stream_cache_alias(b, k_in, v_in, conv_in, k_out, v_out, conv_out):
    """Mark per-layer cache outputs, io-alias them output->input, and tag the
    caches as zero-init recurrent state. `cache_mask` stays caller-bound."""
    for li in range(N_LAYERS):
        b.mark_output(k_out[li], f"kcache_{li}_out")
        b.mark_output(v_out[li], f"vcache_{li}_out")
        b.mark_output(conv_out[li], f"conv_cache_{li}_out")
    for li in range(N_LAYERS):
        b.add_output_alias(k_in[li], k_out[li])
        b.add_output_alias(v_in[li], v_out[li])
        b.add_output_alias(conv_in[li], conv_out[li])
    for li in range(N_LAYERS):
        for v in (k_in[li], v_in[li], conv_in[li]):
            b.add_input_role(v, RK.AION_ROLE_STATE, zero_init=True)


# Window (mel frames) fed per streaming step: in-graph log-mel + subsampling of the
# window yields calc_length(WMEL) encoder frames; window frame j maps to global encoder
# frame (m_start//8 + j) for j>=STREAM_SUB_DROP, so win[DROP:DROP+chunk] are the `chunk`
# new frames (verified bit-exact). The driver slices overlapping audio windows so there
# is NO length cap. The window is sized to the chunk (smallest mel window yielding
# DROP+chunk+1 enc frames, min 40 for the subsampling receptive field).
STREAM_SUB_DROP = 2


def stream_window_mel(chunk: int) -> int:
    need = STREAM_SUB_DROP + chunk + 1
    w = 40
    while calc_length(w) < need:
        w += 8
    return w


def build_stream_full(loader: _Loader, b: Builder, att_left: int = 70, att_right: int = 13,
                      max_out: int = 64):
    """Self-contained cache-aware streaming model: one audio window per chunk ->
    in-graph log-mel + subsampling -> cached conformer -> in-graph RNNT decode."""
    chunk = att_right + 1
    t_kv = att_left + chunk
    wmel = stream_window_mel(chunk)
    feat = build_logmel(loader, b, wmel)                    # audio window -> [1,wmel,128]
    h_win = build_subsample(loader, b, feat, wmel)          # [1, calc_length(wmel), 1024]
    h = b.slice(h_win, (0, STREAM_SUB_DROP, 0), (1, chunk, D_MODEL))  # the `chunk` new frames
    cache_mask, k_in, v_in, conv_in = _stream_cache_io(b, att_left, chunk)
    enc, k_out, v_out, conv_out = _stream_conformer_layers(loader, b, h, att_left, chunk, t_kv, cache_mask, k_in, v_in, conv_in)
    _stream_cache_alias(b, k_in, v_in, conv_in, k_out, v_out, conv_out)
    _ingraph_decode(loader, b, enc, chunk, max_out)         # adds out_count/out_tokens + decode-state IoAlias
    return chunk, t_kv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nemo_path")
    ap.add_argument("out_aion")
    ap.add_argument("--streaming", action="store_true",
                    help="build the cache-aware streaming model (default: offline whole-clip)")
    ap.add_argument("--att-left", type=int, default=None,
                    help="left attention context in encoder frames (streaming default 70; "
                         "offline default: full attention)")
    ap.add_argument("--att-right", type=int, default=None,
                    help="right attention context; chunk = att_right+1; the checkpoint was "
                         "trained on 13/6/1/0 (streaming default 13; offline default: full attention)")
    ap.add_argument("--frames", type=int, default=2885,
                    help="offline only: fixed audio window as mel frame count (T_mel); "
                         "default 2885 (~28.8 s)")
    ap.add_argument("--max-out", type=int, default=512,
                    help="output token buffer size (max emitted tokens per run)")
    args = ap.parse_args()

    loader = _Loader(args.nemo_path)
    common_md = [
        ("quant", "q8_0-weights"),
        ("max_out", str(args.max_out)),
        ("blank_id", str(VOCAB)),
        ("max_symbols", str(MAX_SYMBOLS)),
    ]
    with aion.Context(thread_count=1) as ctx:
        with Builder(ctx) as b:
            if args.streaming:
                al = 70 if args.att_left is None else args.att_left
                ar = 13 if args.att_right is None else args.att_right
                chunk, t_kv = build_stream_full(loader, b, att_left=al, att_right=ar, max_out=args.max_out)
                md = common_md + [
                    ("arch", "nemotron-fastconformer-rnnt-stream-full"),
                    ("chunk", str(chunk)),
                    ("att_left", str(al)),
                    ("t_kv", str(t_kv)),
                    ("window_mel", str(stream_window_mel(chunk))),
                    ("sub_drop", str(STREAM_SUB_DROP)),
                ]
                desc = f"streaming; chunk={chunk}, att=[{al},{ar}], window_mel={stream_window_mel(chunk)}"
            else:
                ar = 0 if args.att_right is None else args.att_right
                build_single(loader, b, args.frames, args.max_out, att_left=args.att_left, att_right=ar)
                md = common_md + [
                    ("arch", "nemotron-fastconformer-rnnt-single"),
                    ("t_mel", str(args.frames)),
                    ("t_out", str(calc_length(args.frames))),
                    ("att_left", "full" if args.att_left is None else str(args.att_left)),
                    ("decode_state", "named-carries"),
                ]
                att = "full" if args.att_left is None else f"[{args.att_left},{ar}]"
                desc = f"offline; T_mel={args.frames} -> T_out={calc_length(args.frames)}, att={att}"
            for key, value in md:
                b.add_metadata(key, value)
            b.export(os.path.abspath(args.out_aion), None)
    print(f"wrote {args.out_aion}  ({desc}, max_out={args.max_out})")

    if loader.tokenizer_path:
        import shutil
        tok = os.path.join(os.path.dirname(os.path.abspath(args.out_aion)) or ".", "nemotron_tokenizer.model")
        shutil.copyfile(loader.tokenizer_path, tok)
        print(f"wrote {tok}")


if __name__ == "__main__":
    main()
