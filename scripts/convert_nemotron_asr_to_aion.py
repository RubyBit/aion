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

Run:  uv run --project bindings/python \
        python scripts/convert_nemotron_asr_to_aion.py <model.nemo> <out.aion> [--streaming]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tarfile
import tempfile
from typing import IO, Dict, List, Optional, Sequence, Tuple

import numpy as np

from aion._writer import Builder, format as aw

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


# The graph-builder layer lives in `aion._writer.builder.Builder`; this
# converter only parameterizes it (LayerNorm epsilon).


def _make_builder() -> Builder:
    return Builder(ln_eps=LN_EPS)


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


def build_logmel(loader: _Loader, b: Builder, t_mel: int) -> int:
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

    audio = b.add_input("audio", aw.DType.f32, (n, 1))
    win_id = b.f32("preprocessor.window512", win512)
    fb_b = b.f32("preprocessor.mel_fb", np.ascontiguousarray(fb.T)[None])  # f32 B [1,257,128] (K=257 not /32)
    guard = b.f32("preprocessor.log_guard", np.full((FEAT_IN,), LOG_GUARD, np.float32))
    coef = b.f32("preprocessor.preemph", np.array([PREEMPH], np.float32))

    sub, add = aw.ElemwiseBinaryOp.sub, aw.ElemwiseBinaryOp.add
    # Preemphasis y[0]=x[0]; y[t]=x[t]-0.97*x[t-1] (column [n,1] so the 0.97 scalar
    # broadcasts over the trailing dim of size 1).
    x_main = b.slice_nd(audio, (1, 0), (n - 1, 1))
    x_prev = b.slice_nd(audio, (0, 0), (n - 1, 1))
    diff = b.ew(sub, x_main, b.bcast_mul(x_prev, coef))      # [n-1,1]
    y = b.concat([b.slice_nd(audio, (0, 0), (1, 1)), diff], axis=0, out_rank=2)  # [n,1]
    sig = b.reshape(y, (1, n))                               # [1,n]
    spec = b.stft(sig, win_id, N_FFT, HOP_LEN, True)         # [1,t_mel,514] = [re(257)|im(257)]
    re = b.slice_nd(spec, (0, 0, 0), (1, t_mel, bins))
    im = b.slice_nd(spec, (0, 0, bins), (1, t_mel, bins))
    power = b.ew(add, b.mul(re, re), b.mul(im, im))          # [1,t_mel,257]
    mel = b.matmul(power, fb_b)                              # [1,t_mel,128]
    return b.unary(aw.UnaryOp.log, b.bcast_add(mel, guard))  # [1,t_mel,128]


def build_att_mask(b: Builder, t_out: int, att_left: int, att_right: int) -> int:
    """Additive [t_out, t_out] chunked_limited mask (0 allowed, -1e9 masked)."""
    chunk = att_right + 1
    m = np.full((t_out, t_out), -1e9, np.float32)
    for i in range(t_out):
        cs = (i // chunk) * chunk
        ce = min(cs + chunk, t_out)
        lo = max(0, cs - att_left)
        m[i, lo:ce] = 0.0
    return b.f32("encoder.att_mask", m)


def build_subsample(loader: _Loader, b: Builder, feat: int, t_mel: int) -> int:
    """dw_striding ×8 subsampling: feat [1, t_mel, 128] -> h [1, t_out, 1024]."""
    t_out = calc_length(t_mel)

    def conv2d_w(name):  # torch [cout,cin/g,kh,kw] -> aion NHWC [kh,kw,cin/g,cout]
        return b.f32(name, np.transpose(loader.get(name), (2, 3, 1, 0)))

    pe = "encoder.pre_encode"
    w0 = conv2d_w(f"{pe}.conv.0.weight"); b0 = b.f32(f"{pe}.conv.0.bias", loader.get(f"{pe}.conv.0.bias"))
    w2 = conv2d_w(f"{pe}.conv.2.weight"); b2 = b.f32(f"{pe}.conv.2.bias", loader.get(f"{pe}.conv.2.bias"))
    w3 = conv2d_w(f"{pe}.conv.3.weight"); b3 = b.f32(f"{pe}.conv.3.bias", loader.get(f"{pe}.conv.3.bias"))
    w5 = conv2d_w(f"{pe}.conv.5.weight"); b5 = b.f32(f"{pe}.conv.5.bias", loader.get(f"{pe}.conv.5.bias"))
    w6 = conv2d_w(f"{pe}.conv.6.weight"); b6 = b.f32(f"{pe}.conv.6.bias", loader.get(f"{pe}.conv.6.bias"))
    out_w = loader.get(f"{pe}.out.weight").reshape(D_MODEL, SUB_CH, FREQ_OUT)             # [out,c,f]
    out_w = np.ascontiguousarray(out_w.transpose(0, 2, 1).reshape(D_MODEL, FREQ_OUT * SUB_CH))  # [out,f*c]
    out_wq = b.q8_matmul_b(f"{pe}.out.weight", out_w)
    out_b = b.f32(f"{pe}.out.bias", loader.get(f"{pe}.out.bias"))

    relu = lambda x: b.unary(aw.UnaryOp.relu, x)
    x = b.reshape(feat, (1, t_mel, FEAT_IN, 1))

    # CausalConv2D pads both spatial dims (top/left=k-1=2, bottom/right=stride-1=1).
    def c2(x, w, bias, groups, k3):
        if k3:
            return b.conv2d(x, w, bias, stride=2, pad_top=2, pad_bottom=1, pad_left=2, pad_right=1, groups=groups)
        return b.conv2d(x, w, bias, stride=1, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, groups=groups)

    x = relu(c2(x, w0, b0, 1, True))
    x = c2(x, w2, b2, SUB_CH, True)
    x = relu(c2(x, w3, b3, 1, False))
    x = c2(x, w5, b5, SUB_CH, True)
    x = relu(c2(x, w6, b6, 1, False))
    x = b.reshape(x, (1, t_out, FREQ_OUT * SUB_CH))  # NHWC flatten (f*256+c)
    return b.bcast_add(b.matmul(x, out_wq), out_b)   # [1, t_out, 1024]


def build_encoder(loader: _Loader, b: Builder, t_mel: int,
                  att_left: Optional[int] = None, att_right: int = 0) -> int:
    t_out = calc_length(t_mel)
    p_len = 2 * t_out - 1

    # Optional chunked_limited attention mask (cache-aware streaming). Frames are
    # grouped into non-overlapping chunks of size att_right+1; every frame in a
    # chunk attends to the whole chunk plus att_left frames before the chunk start
    # (so early-chunk frames get up to att_right lookahead). This matches NeMo
    # att_context_style="chunked_limited", att_context_size=[att_left, att_right].
    att_mask = build_att_mask(b, t_out, att_left, att_right) if att_left is not None else None

    # ---- in-graph log-mel -> dw_striding subsampling -> h [1, T_out, 1024] ----
    feat = build_logmel(loader, b, t_mel)        # [1, T_mel, 128]
    h = build_subsample(loader, b, feat, t_mel)  # [1, T_out, 1024]

    # ---- 24 conformer layers ----
    for li in range(N_LAYERS):
        pf = f"encoder.layers.{li}"

        def ln(x, base, nd):
            g = b.f32(f"{base}.weight", loader.get(f"{base}.weight"))
            be = b.f32(f"{base}.bias", loader.get(f"{base}.bias"))
            return b.layernorm(x, g, be, nd)

        def ffn(x, ff_pref):
            l1 = b.q8_matmul_b(f"{ff_pref}.linear1.weight", loader.get(f"{ff_pref}.linear1.weight"))
            l2 = b.q8_matmul_b(f"{ff_pref}.linear2.weight", loader.get(f"{ff_pref}.linear2.weight"))
            a = b.matmul(x, l1)
            a = b.unary(aw.UnaryOp.silu, a)
            return b.matmul(a, l2)

        # FF1 (x0.5)
        hn = ln(h, f"{pf}.norm_feed_forward1", (D_MODEL,))
        ff1 = ffn(hn, f"{pf}.feed_forward1")
        half = b.f32(f"{pf}._half_ff1", np.full((D_MODEL,), 0.5, dtype=np.float32))
        h = b.add(h, b.bcast_mul(ff1, half))

        # MHSA (rel-pos)
        hn = ln(h, f"{pf}.norm_self_att", (D_MODEL,))
        q = b.reshape(b.matmul(hn, b.q8_matmul_b(f"{pf}.self_attn.linear_q.weight", loader.get(f"{pf}.self_attn.linear_q.weight"))),
                      (1, t_out, N_HEADS, HEAD_DIM))
        k = b.reshape(b.matmul(hn, b.q8_matmul_b(f"{pf}.self_attn.linear_k.weight", loader.get(f"{pf}.self_attn.linear_k.weight"))),
                      (1, t_out, N_HEADS, HEAD_DIM))
        v = b.reshape(b.matmul(hn, b.q8_matmul_b(f"{pf}.self_attn.linear_v.weight", loader.get(f"{pf}.self_attn.linear_v.weight"))),
                      (1, t_out, N_HEADS, HEAD_DIM))
        # pos_emb projected per layer: linear_pos @ sinusoidal -> [P, 1024] -> [H, P, HD]
        pe = sinusoidal_pe(t_out, D_MODEL)  # [P, 1024]
        lin_pos = loader.get(f"{pf}.self_attn.linear_pos.weight")  # [1024, 1024]
        pe_proj = (pe @ lin_pos.T).reshape(p_len, N_HEADS, HEAD_DIM).transpose(1, 0, 2)  # [H, P, HD]
        pe_id = b.f32(f"{pf}.self_attn.pos_emb", np.ascontiguousarray(pe_proj))
        bu = b.f32(f"{pf}.self_attn.pos_bias_u", loader.get(f"{pf}.self_attn.pos_bias_u"))  # [H, HD]
        bv = b.f32(f"{pf}.self_attn.pos_bias_v", loader.get(f"{pf}.self_attn.pos_bias_v"))
        att = b.relpos_mha(q, k, v, pe_id, bu, bv, att_mask, ATT_SCALE, N_HEADS)  # [1,T,H,HD]
        att = b.reshape(att, (1, t_out, D_MODEL))
        att = b.matmul(att, b.q8_matmul_b(f"{pf}.self_attn.linear_out.weight", loader.get(f"{pf}.self_attn.linear_out.weight")))
        h = b.add(h, att)

        # Conv module. Pointwise convs are k=1 == matmul; store them q8 and route
        # through the fast matmul path (matches _stream_conformer_layers). Only the
        # depthwise conv stays f32 (tiny, precision-sensitive — not a matmul).
        hn = ln(h, f"{pf}.norm_conv", (D_MODEL,))
        pw1 = b.q8_matmul_b(f"{pf}.conv.pointwise_conv1.weight",
                            loader.get(f"{pf}.conv.pointwise_conv1.weight")[:, :, 0])  # [2048,1024]
        c = b.matmul(hn, pw1)  # [1,T,2048]
        a = b.slice_nd(c, (0, 0, 0), (1, t_out, D_MODEL))
        g = b.slice_nd(c, (0, 0, D_MODEL), (1, t_out, D_MODEL))
        c = b.mul(a, b.unary(aw.UnaryOp.sigmoid, g))  # GLU
        dw = b.f32(f"{pf}.conv.depthwise_conv.weight",
                   np.transpose(loader.get(f"{pf}.conv.depthwise_conv.weight"), (2, 1, 0)))  # [9,1,1024]
        c = b.conv1d(c, dw, None, stride=1, dilation=1, pad_left=CONV_K - 1, pad_right=0, groups=D_MODEL)
        c = ln(c, f"{pf}.conv.batch_norm", (D_MODEL,))  # LayerNorm slot
        c = b.unary(aw.UnaryOp.silu, c)
        pw2 = b.q8_matmul_b(f"{pf}.conv.pointwise_conv2.weight",
                            loader.get(f"{pf}.conv.pointwise_conv2.weight")[:, :, 0])  # [1024,1024]
        c = b.matmul(c, pw2)  # [1,T,1024]
        h = b.add(h, c)

        # FF2 (x0.5)
        hn = ln(h, f"{pf}.norm_feed_forward2", (D_MODEL,))
        ff2 = ffn(hn, f"{pf}.feed_forward2")
        half2 = b.f32(f"{pf}._half_ff2", np.full((D_MODEL,), 0.5, dtype=np.float32))
        h = b.add(h, b.bcast_mul(ff2, half2))

        # final norm
        h = ln(h, f"{pf}.norm_out", (D_MODEL,))

    return h


PRED_HIDDEN = 640
VOCAB = 1024
MAX_SYMBOLS = 10
BLANK = VOCAB  # 1024


def _ingraph_decode(loader: _Loader, b: Builder, enc: int, t_out: int, max_out: int) -> None:
    """In-graph greedy RNNT decode over `enc [1, t_out, 1024]`. Adds outputs
    out_count/out_tokens + persistent decode state (last/h0/c0/h1/c1) with their
    IoAliases. Named multi-carry Loop, early-exit via `active = frame < t_out`.
    Shared by build_single (t_out = full encoder length) and build_stream_full
    (t_out = chunk, state threaded across chunks)."""
    H = PRED_HIDDEN
    enc2d = b.reshape(enc, (t_out, D_MODEL))     # [t_out, 1024]  (GatherRows table)

    dp = "decoder.prediction"
    embed = b.q8_embedding(f"{dp}.embed.weight", loader.get(f"{dp}.embed.weight"))

    def lstm_w(li):
        return (
            b.f32(f"{dp}.dec_rnn.lstm.weight_ih_l{li}", loader.get(f"{dp}.dec_rnn.lstm.weight_ih_l{li}").T),
            b.f32(f"{dp}.dec_rnn.lstm.weight_hh_l{li}", loader.get(f"{dp}.dec_rnn.lstm.weight_hh_l{li}").T),
            b.f32(f"{dp}.dec_rnn.lstm.bias_ih_l{li}", loader.get(f"{dp}.dec_rnn.lstm.bias_ih_l{li}")),
            b.f32(f"{dp}.dec_rnn.lstm.bias_hh_l{li}", loader.get(f"{dp}.dec_rnn.lstm.bias_hh_l{li}")),
        )

    w0 = lstm_w(0)
    w1 = lstm_w(1)
    enc_w = b.q8_matmul_b("joint.enc.weight", loader.get("joint.enc.weight"))
    enc_b = b.f32("joint.enc.bias", loader.get("joint.enc.bias"))
    pred_w = b.q8_matmul_b("joint.pred.weight", loader.get("joint.pred.weight"))
    pred_b = b.f32("joint.pred.bias", loader.get("joint.pred.bias"))
    jn_w = b.q8_matmul_b("joint.joint_net.2.weight", loader.get("joint.joint_net.2.weight"))
    jn_b = b.f32("joint.joint_net.2.bias", loader.get("joint.joint_net.2.bias"))

    add, sub, mul = aw.ElemwiseBinaryOp.add, aw.ElemwiseBinaryOp.sub, aw.ElemwiseBinaryOp.mul
    eq, ge, lt = aw.ElemwiseBinaryOp.eq, aw.ElemwiseBinaryOp.ge, aw.ElemwiseBinaryOp.lt
    one_i = b.i32_const("_one_i", np.array([1], np.int32))           # [1]
    blank_i = b.i32_const("_blank_i", np.array([BLANK], np.int32))
    maxsym_i = b.i32_const("_maxsym_i", np.array([MAX_SYMBOLS], np.int32))
    T_i = b.i32_const("_T_i", np.array([t_out], np.int32))

    # Decode state as named, separately-typed carries (no packing). Persistent
    # cross-chunk state (last_token, LSTM h/c) is IoAlias'd output->input; the
    # per-run loop bookkeeping (frame/sym/count/active) and the output token
    # buffer are bound fresh each run.
    frame_in = b.add_input("frame_in", aw.DType.i32, (1,))           # init 0
    sym_in = b.add_input("sym_in", aw.DType.i32, (1,))               # init 0
    count_in = b.add_input("count_in", aw.DType.i32, (1,))           # init 0
    active_in = b.add_input("active_in", aw.DType.i32, (1,))         # init 1
    last_in = b.add_input("last_in", aw.DType.i32, (1,))             # init BLANK
    h0_in = b.add_input("h0_in", aw.DType.f32, (1, H))
    c0_in = b.add_input("c0_in", aw.DType.f32, (1, H))
    h1_in = b.add_input("h1_in", aw.DType.f32, (1, H))
    c1_in = b.add_input("c1_in", aw.DType.f32, (1, H))
    toks_in = b.add_input("toks_in", aw.DType.i32, (max_out, 1))     # init zeros

    b.begin_region()
    # `active` (carry 3) is the continue predicate; the loop only runs the body
    # while frame < t_out, so the gather frame is always in-range (no clamp needed).
    # GatherRows wants rank-2 indices, so reshape the [1] carries to [1,1].
    enc_frame = b.gather_rows(enc2d, b.reshape(frame_in, (1, 1)))    # [1,1,1024]
    emb = b.reshape(b.gather_rows(embed, b.reshape(last_in, (1, 1))), (1, H))  # [1,640]

    st0 = b.lstm_cell(emb, h0_in, c0_in, *w0)
    h0o = b.slice_nd(st0, (0, 0), (1, H))
    c0o = b.slice_nd(st0, (0, H), (1, H))
    st1 = b.lstm_cell(h0o, h1_in, c1_in, *w1)
    h1o = b.slice_nd(st1, (0, 0), (1, H))
    c1o = b.slice_nd(st1, (0, H), (1, H))

    fproj = b.bcast_add(b.matmul(enc_frame, enc_w), enc_b)           # [1,1,640]
    gproj = b.bcast_add(b.matmul(b.reshape(h1o, (1, 1, H)), pred_w), pred_b)
    sj = b.unary(aw.UnaryOp.relu, b.add(fproj, gproj))              # [1,1,640]
    logits = b.bcast_add(b.matmul(sj, jn_w), jn_b)                  # [1,1,1025]
    k = b.reshape(b.argmax(logits, axis=2, out_rank=2), (1,))       # [1] i32

    # advance = (k==blank) or (sym>=max_symbols);  emit = 1 - advance.
    advance = b.compare(ge, b.ew(add, b.compare(eq, k, blank_i),
                                 b.compare(ge, sym_in, maxsym_i)), one_i)  # i32 [1]
    emit = b.ew(sub, one_i, advance)                                # i32 [1]
    emit_f = b.reshape(b.cast(emit, aw.DType.f32), (1,))           # f32 [1]

    def sel_row(keep, take):  # emit ? take : keep, over [1,H]
        diff_c = b.reshape(b.ew(sub, take, keep), (H, 1))          # [H,1]
        scaled = b.reshape(b.bcast_mul(diff_c, emit_f), (1, H))    # [1,H]
        return b.ew(add, keep, scaled)

    # Next state (arithmetic select on named carries).
    frame_next = b.ew(add, frame_in, advance)                      # +1 on advance
    sym_next = b.ew(mul, b.ew(add, sym_in, one_i), emit)           # sym+1 on emit else 0
    count_next = b.ew(add, count_in, emit)                         # +1 on emit
    last_next = b.ew(add, last_in, b.ew(mul, b.ew(sub, k, last_in), emit))  # k on emit
    active_next = b.compare(lt, frame_next, T_i)                   # i32 [1] {0,1}
    h0_next, c0_next = sel_row(h0_in, h0o), sel_row(c0_in, c0o)
    h1_next, c1_next = sel_row(h1_in, h1o), sel_row(c1_in, c1o)
    # Always write k at index `count`; only emit advances `count`, so blank
    # writes land at a slot the next emit overwrites and sit outside [:count].
    toks_next = b.scatter_row(toks_in, count_in, k)               # [max_out,1] i32

    body = b.end_region([frame_next, sym_next, count_next, active_next, last_next,
                         h0_next, c0_next, h1_next, c1_next, toks_next])
    carries = [frame_in, sym_in, count_in, active_in, last_in,
               h0_in, c0_in, h1_in, c1_in, toks_in]
    outs = b.loop(carries, body, t_out * (MAX_SYMBOLS + 1), cond_carry=3)
    _, _, count_out, _, last_out, h0_out, c0_out, h1_out, c1_out, toks_out = outs

    b.add_output("out_count", count_out)                           # i32 [1]
    b.add_output("out_tokens", b.reshape(toks_out, (max_out,)))    # i32 [max_out]
    b.add_output("last_out", last_out)
    b.add_output("h0_out", h0_out)
    b.add_output("c0_out", c0_out)
    b.add_output("h1_out", h1_out)
    b.add_output("c1_out", c1_out)

    def alias(in_name, out_name):
        ii = next(i for i, nv in enumerate(b.pkg.inputs) if nv.name == in_name)
        oi = next(i for i, nv in enumerate(b.pkg.outputs) if nv.name == out_name)
        b.add_io_alias(ii, oi)
    for nm in ("last", "h0", "c0", "h1", "c1"):  # persistent decode state
        alias(f"{nm}_in", f"{nm}_out")

    # Roles: the LSTM state is generic zero-init recurrent state, so the runtime
    # auto-zeros + carries it. `last_in` is deliberately NOT tagged — it seeds at
    # the (non-zero) blank SOS token and stays caller-bound.
    for nm in ("h0", "c0", "h1", "c1"):
        ii = next(i for i, nv in enumerate(b.pkg.inputs) if nv.name == f"{nm}_in")
        b.pkg.input_roles.append(aw.InputRole(input_index=ii, kind=aw.RoleKind.state))


def build_single(loader: _Loader, b: Builder, t_mel: int, max_out: int,
                 att_left: Optional[int] = None, att_right: int = 0) -> int:
    """Encoder + in-graph greedy RNNT decode in one graph. Returns max_out.

    Decode state is carried as named, separately-typed Loop carries (frame, sym,
    count, active, last_token, LSTM h0/c0/h1/c1, output token buffer) rather than
    a single packed tensor. The loop early-exits via the i32 `active` predicate
    (frame < T), so it runs ~T+emitted iterations instead of the static cap.
    Persistent state (last_token + LSTM h/c) is IoAlias'd output->input so chunks
    can be streamed; per-run bookkeeping and the token buffer are bound fresh.
    """
    T = calc_length(t_mel)
    enc = build_encoder(loader, b, t_mel, att_left=att_left, att_right=att_right)  # [1, T, 1024]
    b.add_output("encoder_out", enc)
    _ingraph_decode(loader, b, enc, T, max_out)
    return max_out


def _stream_conformer_layers(loader, b, h, att_left, chunk, t_kv, cache_mask, k_in, v_in, conv_in):
    """24 cache-aware conformer layers over chunk `h [1,chunk,1024]`. Per layer the
    attention uses a projected-K/V cache (standard KV-cache): only the `chunk` new
    frames are projected (M=chunk), then Concat'd onto the cached K/V — instead of
    re-projecting all att_left cached frames every step. The causal depthwise conv
    prepends conv_cache. Returns (enc, k_out, v_out, conv_out)."""
    conv_left = CONV_K - 1
    p_len = 2 * t_kv - 1

    def ln(x, base):
        return b.layernorm(x, b.f32(f"{base}.weight", loader.get(f"{base}.weight")),
                           b.f32(f"{base}.bias", loader.get(f"{base}.bias")), (D_MODEL,))

    def ffn(x, pref):
        l1 = b.q8_matmul_b(f"{pref}.linear1.weight", loader.get(f"{pref}.linear1.weight"))
        l2 = b.q8_matmul_b(f"{pref}.linear2.weight", loader.get(f"{pref}.linear2.weight"))
        return b.matmul(b.unary(aw.UnaryOp.silu, b.matmul(x, l1)), l2)

    def q8(name):
        return b.q8_matmul_b(name, loader.get(name))

    k_out, v_out, conv_out = [], [], []
    for li in range(N_LAYERS):
        pf = f"encoder.layers.{li}"
        hn = ln(h, f"{pf}.norm_feed_forward1")
        h = b.add(h, b.bcast_mul(ffn(hn, f"{pf}.feed_forward1"),
                                 b.f32(f"{pf}._half_ff1", np.full((D_MODEL,), 0.5, np.float32))))
        hn = ln(h, f"{pf}.norm_self_att")                        # [1,chunk,1024]
        # Fused QKV projection of ONLY the new chunk frames (M=chunk): one matmul
        # instead of three (fewer op dispatches), then prepend the cached K/V.
        wqkv = np.concatenate([loader.get(f"{pf}.self_attn.linear_q.weight"),
                               loader.get(f"{pf}.self_attn.linear_k.weight"),
                               loader.get(f"{pf}.self_attn.linear_v.weight")], axis=0)  # [3072,1024]
        qkv = b.matmul(hn, b.q8_matmul_b(f"{pf}.self_attn.qkv.weight", wqkv))           # [1,chunk,3072]
        q = b.reshape(b.slice_nd(qkv, (0, 0, 0), (1, chunk, D_MODEL)), (1, chunk, N_HEADS, HEAD_DIM))
        k_c = b.reshape(b.slice_nd(qkv, (0, 0, D_MODEL), (1, chunk, D_MODEL)), (1, chunk, N_HEADS, HEAD_DIM))
        v_c = b.reshape(b.slice_nd(qkv, (0, 0, 2 * D_MODEL), (1, chunk, D_MODEL)), (1, chunk, N_HEADS, HEAD_DIM))
        k = b.concat([k_in[li], k_c], axis=1, out_rank=4)        # [1,t_kv,H,D]
        v = b.concat([v_in[li], v_c], axis=1, out_rank=4)
        pe = sinusoidal_pe(t_kv, D_MODEL)
        lin_pos = loader.get(f"{pf}.self_attn.linear_pos.weight")
        pe_proj = (pe @ lin_pos.T).reshape(p_len, N_HEADS, HEAD_DIM).transpose(1, 0, 2)
        pe_id = b.f32(f"{pf}.self_attn.pos_emb", np.ascontiguousarray(pe_proj))
        bu = b.f32(f"{pf}.self_attn.pos_bias_u", loader.get(f"{pf}.self_attn.pos_bias_u"))
        bv = b.f32(f"{pf}.self_attn.pos_bias_v", loader.get(f"{pf}.self_attn.pos_bias_v"))
        att = b.relpos_mha(q, k, v, pe_id, bu, bv, cache_mask, ATT_SCALE, N_HEADS)
        att = b.matmul(b.reshape(att, (1, chunk, D_MODEL)), q8(f"{pf}.self_attn.linear_out.weight"))
        h = b.add(h, att)
        k_out.append(b.slice_nd(k, (0, chunk, 0, 0), (1, att_left, N_HEADS, HEAD_DIM)))  # last att_left
        v_out.append(b.slice_nd(v, (0, chunk, 0, 0), (1, att_left, N_HEADS, HEAD_DIM)))

        hn = ln(h, f"{pf}.norm_conv")
        # Pointwise convs are k=1 == matmul; route through the q8 fast (matvec) path
        # instead of f32 Conv1D (which misses it and repacks f32 weights per chunk).
        pw1 = b.q8_matmul_b(f"{pf}.conv.pointwise_conv1.weight",
                            loader.get(f"{pf}.conv.pointwise_conv1.weight")[:, :, 0])  # [2048,1024]
        c = b.matmul(hn, pw1)                                                          # [1,chunk,2048]
        glu = b.mul(b.slice_nd(c, (0, 0, 0), (1, chunk, D_MODEL)),
                    b.unary(aw.UnaryOp.sigmoid, b.slice_nd(c, (0, 0, D_MODEL), (1, chunk, D_MODEL))))
        glu_full = b.concat([conv_in[li], glu], axis=1, out_rank=3)
        dw = b.f32(f"{pf}.conv.depthwise_conv.weight",
                   np.transpose(loader.get(f"{pf}.conv.depthwise_conv.weight"), (2, 1, 0)))
        c = b.conv1d(glu_full, dw, None, stride=1, dilation=1, pad_left=0, pad_right=0, groups=D_MODEL)
        c = b.unary(aw.UnaryOp.silu, ln(c, f"{pf}.conv.batch_norm"))
        pw2 = b.q8_matmul_b(f"{pf}.conv.pointwise_conv2.weight",
                            loader.get(f"{pf}.conv.pointwise_conv2.weight")[:, :, 0])  # [1024,1024]
        c = b.matmul(c, pw2)                                                           # [1,chunk,1024]
        h = b.add(h, c)
        conv_out.append(b.slice_nd(glu_full, (0, chunk, 0), (1, conv_left, D_MODEL)))

        hn = ln(h, f"{pf}.norm_feed_forward2")
        h = b.add(h, b.bcast_mul(ffn(hn, f"{pf}.feed_forward2"),
                                 b.f32(f"{pf}._half_ff2", np.full((D_MODEL,), 0.5, np.float32))))
        h = ln(h, f"{pf}.norm_out")
    return h, k_out, v_out, conv_out


def _stream_cache_io(b, att_left, chunk):
    """Create per-layer K/V (projected) + conv cache inputs + a cache_mask input.
    Returns (cache_mask, k_in, v_in, conv_in)."""
    conv_left = CONV_K - 1
    t_kv = att_left + chunk
    cache_mask = b.add_input("cache_mask", aw.DType.f32, (chunk, t_kv))
    k_in = [b.add_input(f"kcache_{li}_in", aw.DType.f32, (1, att_left, N_HEADS, HEAD_DIM)) for li in range(N_LAYERS)]
    v_in = [b.add_input(f"vcache_{li}_in", aw.DType.f32, (1, att_left, N_HEADS, HEAD_DIM)) for li in range(N_LAYERS)]
    conv_in = [b.add_input(f"conv_cache_{li}_in", aw.DType.f32, (1, conv_left, D_MODEL)) for li in range(N_LAYERS)]
    return cache_mask, k_in, v_in, conv_in


def _stream_cache_alias(b, k_out, v_out, conv_out):
    """Add per-layer cache outputs and IoAlias them output->input."""
    for li in range(N_LAYERS):
        b.add_output(f"kcache_{li}_out", k_out[li])
        b.add_output(f"vcache_{li}_out", v_out[li])
        b.add_output(f"conv_cache_{li}_out", conv_out[li])

    def alias(in_name, out_name):
        ii = next(i for i, nv in enumerate(b.pkg.inputs) if nv.name == in_name)
        oi = next(i for i, nv in enumerate(b.pkg.outputs) if nv.name == out_name)
        b.add_io_alias(ii, oi)
    for li in range(N_LAYERS):
        alias(f"kcache_{li}_in", f"kcache_{li}_out")
        alias(f"vcache_{li}_in", f"vcache_{li}_out")
        alias(f"conv_cache_{li}_in", f"conv_cache_{li}_out")

    # Roles: these are fixed-shape shift-window buffers (not append-style
    # sequence caches), i.e. plain zero-init recurrent state the runtime can
    # auto-allocate and carry. `cache_mask` stays caller-bound (per-chunk values).
    for li in range(N_LAYERS):
        for nm in (f"kcache_{li}_in", f"vcache_{li}_in", f"conv_cache_{li}_in"):
            ii = next(i for i, nv in enumerate(b.pkg.inputs) if nv.name == nm)
            b.pkg.input_roles.append(aw.InputRole(input_index=ii, kind=aw.RoleKind.state))


# Window (mel frames) fed per streaming step: in-graph log-mel + subsampling of the
# window yields calc_length(WMEL) encoder frames; window frame j maps to global encoder
# frame (m_start//8 + j) for j>=STREAM_SUB_DROP, so win[DROP:DROP+chunk] are the `chunk`
# new frames (verified bit-exact). The driver slices overlapping audio windows
# (m_start = chunk*8*c - DROP*8 global mel), so there is NO length cap. The window is
# sized to the chunk (smallest mel window yielding DROP+chunk+1 enc frames, min 40 for
# the subsampling receptive field) so small/low-latency chunks don't subsample 160 mel
# frames to extract a handful — keeps per-chunk subsampling cheap.
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
    in-graph log-mel + subsampling -> cached conformer -> in-graph RNNT decode.
    All conformer caches AND decode state are IoAlias'd, so a trivial driver
    ("slice next window -> run -> read tokens") streams indefinitely (no cap)."""
    chunk = att_right + 1
    t_kv = att_left + chunk
    wmel = stream_window_mel(chunk)
    feat = build_logmel(loader, b, wmel)                    # audio window -> [1,wmel,128]
    h_win = build_subsample(loader, b, feat, wmel)          # [1, calc_length(wmel), 1024]
    h = b.slice_nd(h_win, (0, STREAM_SUB_DROP, 0), (1, chunk, D_MODEL))  # the `chunk` new frames
    cache_mask, k_in, v_in, conv_in = _stream_cache_io(b, att_left, chunk)
    enc, k_out, v_out, conv_out = _stream_conformer_layers(loader, b, h, att_left, chunk, t_kv, cache_mask, k_in, v_in, conv_in)
    _stream_cache_alias(b, k_out, v_out, conv_out)
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
    b = _make_builder()
    common_md = [
        aw.MetadataEntry(key="quant", value="q8_0-weights"),
        aw.MetadataEntry(key="max_out", value=str(args.max_out)),
        aw.MetadataEntry(key="blank_id", value=str(VOCAB)),
        aw.MetadataEntry(key="max_symbols", value=str(MAX_SYMBOLS)),
    ]
    if args.streaming:
        al = 70 if args.att_left is None else args.att_left
        ar = 13 if args.att_right is None else args.att_right
        chunk, t_kv = build_stream_full(loader, b, att_left=al, att_right=ar, max_out=args.max_out)
        b.pkg.metadata.extend(common_md + [
            aw.MetadataEntry(key="arch", value="nemotron-fastconformer-rnnt-stream-full"),
            aw.MetadataEntry(key="chunk", value=str(chunk)),
            aw.MetadataEntry(key="att_left", value=str(al)),
            aw.MetadataEntry(key="t_kv", value=str(t_kv)),
            aw.MetadataEntry(key="window_mel", value=str(stream_window_mel(chunk))),
            aw.MetadataEntry(key="sub_drop", value=str(STREAM_SUB_DROP)),
        ])
        desc = f"streaming; chunk={chunk}, att=[{al},{ar}], window_mel={stream_window_mel(chunk)}"
    else:
        ar = 0 if args.att_right is None else args.att_right
        build_single(loader, b, args.frames, args.max_out, att_left=args.att_left, att_right=ar)
        b.pkg.metadata.extend(common_md + [
            aw.MetadataEntry(key="arch", value="nemotron-fastconformer-rnnt-single"),
            aw.MetadataEntry(key="t_mel", value=str(args.frames)),
            aw.MetadataEntry(key="t_out", value=str(calc_length(args.frames))),
            aw.MetadataEntry(key="att_left", value="full" if args.att_left is None else str(args.att_left)),
            aw.MetadataEntry(key="decode_state", value="named-carries"),
        ])
        att = "full" if args.att_left is None else f"[{args.att_left},{ar}]"
        desc = f"offline; T_mel={args.frames} -> T_out={calc_length(args.frames)}, att={att}"
    b.write(args.out_aion)
    print(f"wrote {args.out_aion}  ({desc}, max_out={args.max_out})")

    if loader.tokenizer_path:
        import shutil
        tok = os.path.join(os.path.dirname(os.path.abspath(args.out_aion)) or ".", "nemotron_tokenizer.model")
        shutil.copyfile(loader.tokenizer_path, tok)
        print(f"wrote {tok}")


if __name__ == "__main__":
    main()
