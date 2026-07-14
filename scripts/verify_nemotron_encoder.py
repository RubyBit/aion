# SPDX-License-Identifier: Apache-2.0
"""Numerically verify the Aion Nemotron encoder against a faithful torch reference.

The torch reference reimplements NeMo's FastConformer encoder forward from the same
checkpoint weights, so a close match confirms the `.aion` graph (subsampling, RelPosMHA,
conv module, FFNs) is constructed correctly. Run:

  uv run --project bindings/python --extra dev \
    python scripts/verify_nemotron_encoder.py <model.nemo> <encoder.aion> [--frames N]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tarfile
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert_nemotron_asr_to_aion as conv  # reuse constants/calc_length

D, L, H, HD, K = conv.D_MODEL, conv.N_LAYERS, conv.N_HEADS, conv.HEAD_DIM, conv.CONV_K
LN_EPS = conv.LN_EPS


def load_sd(nemo_path):
    tmp = tempfile.mkdtemp()
    with tarfile.open(nemo_path, "r") as tf:
        ck = next(m for m in tf.getnames() if m.endswith("model_weights.ckpt"))
        p = os.path.join(tmp, "w.ckpt")
        with conv._tar_member(tf, ck) as s, open(p, "wb") as d:
            while True:
                c = s.read(1 << 24)
                if not c:
                    break
                d.write(c)
    return torch.load(p, map_location="cpu", weights_only=True)


def ln(x, w, b):
    return F.layer_norm(x, (x.shape[-1],), w, b, eps=LN_EPS)


def rel_shift(x):  # x: [H, T, 2T-1]
    h, t, p = x.shape
    x = F.pad(x, (1, 0))           # [H,T,2T]
    x = x.reshape(h, p + 1, t)     # [H,2T,T]
    x = x[:, 1:].reshape(h, t, p)  # [H,T,2T-1]
    return x[:, :, :t]


def sinusoidal_pe(length, d):
    return torch.from_numpy(conv.sinusoidal_pe(length, d))


def encoder_ref(sd, mel):  # mel [T_mel,128] -> [T,1024]
    g = lambda n: sd[n].float()
    x = mel.unsqueeze(0).unsqueeze(0)  # [1,1,T_mel,128]
    pe = "encoder.pre_encode"

    def c2(x, w, b, groups, k3):
        x = F.pad(x, (2, 1, 2, 1)) if k3 else x  # (freq_l,freq_r,time_t,time_b)
        s = 2 if k3 else 1
        return F.conv2d(x, w, b, stride=s, padding=0, groups=groups)

    x = F.relu(c2(x, g(f"{pe}.conv.0.weight"), g(f"{pe}.conv.0.bias"), 1, True))
    x = c2(x, g(f"{pe}.conv.2.weight"), g(f"{pe}.conv.2.bias"), 256, True)
    x = F.relu(c2(x, g(f"{pe}.conv.3.weight"), g(f"{pe}.conv.3.bias"), 1, False))
    x = c2(x, g(f"{pe}.conv.5.weight"), g(f"{pe}.conv.5.bias"), 256, True)
    x = F.relu(c2(x, g(f"{pe}.conv.6.weight"), g(f"{pe}.conv.6.bias"), 1, False))
    b1, c1, t1, f1 = x.shape
    x = x.transpose(1, 2).reshape(b1, t1, c1 * f1)  # NeMo flatten c*F+f
    h = F.linear(x, g(f"{pe}.out.weight"), g(f"{pe}.out.bias"))[0]  # [T,1024]
    T = h.shape[0]

    pe_sin = sinusoidal_pe(T, D)  # [2T-1, 1024]
    scale = 1.0 / math.sqrt(HD)
    for li in range(L):
        pf = f"encoder.layers.{li}"
        # FF1
        a = ln(h, g(f"{pf}.norm_feed_forward1.weight"), g(f"{pf}.norm_feed_forward1.bias"))
        a = F.linear(F.silu(F.linear(a, g(f"{pf}.feed_forward1.linear1.weight"))), g(f"{pf}.feed_forward1.linear2.weight"))
        h = h + 0.5 * a
        # MHSA rel-pos
        hn = ln(h, g(f"{pf}.norm_self_att.weight"), g(f"{pf}.norm_self_att.bias"))
        q = F.linear(hn, g(f"{pf}.self_attn.linear_q.weight")).view(T, H, HD).transpose(0, 1)  # [H,T,HD]
        k = F.linear(hn, g(f"{pf}.self_attn.linear_k.weight")).view(T, H, HD).transpose(0, 1)
        v = F.linear(hn, g(f"{pf}.self_attn.linear_v.weight")).view(T, H, HD).transpose(0, 1)
        p = F.linear(pe_sin, g(f"{pf}.self_attn.linear_pos.weight")).view(-1, H, HD).transpose(0, 1)  # [H,2T-1,HD]
        u = g(f"{pf}.self_attn.pos_bias_u").unsqueeze(1)  # [H,1,HD]
        vb = g(f"{pf}.self_attn.pos_bias_v").unsqueeze(1)
        ac = torch.matmul(q + u, k.transpose(-2, -1))      # [H,T,T]
        bd = rel_shift(torch.matmul(q + vb, p.transpose(-2, -1)))  # [H,T,T]
        att = torch.softmax((ac + bd) * scale, dim=-1)
        o = torch.matmul(att, v).transpose(0, 1).reshape(T, D)  # [T,1024]
        o = F.linear(o, g(f"{pf}.self_attn.linear_out.weight"))
        h = h + o
        # Conv module
        c = ln(h, g(f"{pf}.norm_conv.weight"), g(f"{pf}.norm_conv.bias")).transpose(0, 1).unsqueeze(0)  # [1,1024,T]
        c = F.conv1d(c, g(f"{pf}.conv.pointwise_conv1.weight"))  # [1,2048,T]
        c = F.glu(c, dim=1)  # [1,1024,T]
        c = F.conv1d(F.pad(c, (K - 1, 0)), g(f"{pf}.conv.depthwise_conv.weight"), groups=D)
        c = c.squeeze(0).transpose(0, 1)  # [T,1024]
        c = ln(c, g(f"{pf}.conv.batch_norm.weight"), g(f"{pf}.conv.batch_norm.bias"))
        c = F.silu(c).transpose(0, 1).unsqueeze(0)  # [1,1024,T]
        c = F.conv1d(c, g(f"{pf}.conv.pointwise_conv2.weight")).squeeze(0).transpose(0, 1)  # [T,1024]
        h = h + c
        # FF2
        a = ln(h, g(f"{pf}.norm_feed_forward2.weight"), g(f"{pf}.norm_feed_forward2.bias"))
        a = F.linear(F.silu(F.linear(a, g(f"{pf}.feed_forward2.linear1.weight"))), g(f"{pf}.feed_forward2.linear2.weight"))
        h = h + 0.5 * a
        # norm_out
        h = ln(h, g(f"{pf}.norm_out.weight"), g(f"{pf}.norm_out.bias"))
    return h


def main():
    import aion

    ap = argparse.ArgumentParser()
    ap.add_argument("nemo_path")
    ap.add_argument("aion_path")
    ap.add_argument("--frames", type=int, default=200)
    args = ap.parse_args()

    sd = load_sd(args.nemo_path)
    rng = np.random.RandomState(0)
    mel_np = (rng.randn(args.frames, 128).astype(np.float32)) * 0.5

    ref = encoder_ref(sd, torch.from_numpy(mel_np)).detach().numpy()

    m = aion.load_model(args.aion_path, thread_count=4)
    got = m.run_numpy({"mel": mel_np[None]}, outputs=["encoder_out"])["encoder_out"][0]

    print("ref", ref.shape, "got", got.shape)
    diff = np.abs(ref - got)
    denom = np.abs(ref).mean() + 1e-6
    print(f"max_abs={diff.max():.4e}  mean_abs={diff.mean():.4e}  rel={diff.mean()/denom:.4e}")
    print(f"ref mean/std {ref.mean():.4f}/{ref.std():.4f}   got mean/std {got.mean():.4f}/{got.std():.4f}")
    # q8_0 weights -> expect small but nonzero error.
    ok = diff.max() < 0.15 and (diff.mean() / denom) < 0.05
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
