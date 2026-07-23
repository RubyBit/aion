# SPDX-License-Identifier: Apache-2.0
"""Transcribe a 16 kHz wav with the Nemotron FastConformer-Transducer on Aion.

Takes ONE self-contained .aion — in-graph log-mel + encoder + in-graph greedy
RNNT decode, raw audio in / token ids out — built by
scripts/convert_nemotron_asr_to_aion.py. The model kind and all runtime
parameters (chunk, window/buffer sizes) are auto-detected from the package
metadata:

  offline   (default converter build): the whole clip in one run.
  streaming (converter --streaming build): chunked; prints the transcript as
            it grows, threading all state inside the runtime via IoAlias.

Run:
  uv run --project bindings/python python \
    bindings/python/examples/nemotron_asr_streaming.py \
      models/nemotron-asr/nemotron.aion --wav models/nemotron-asr/test_audio.wav

`--tokenizer` defaults to `nemotron_tokenizer.model` next to the model (the
converter drops it there).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import aion

from _common import add_device_args, load_model_with_device, read_aion_metadata

SAMPLE_RATE = 16000
HOP = 160


def _i32(ctx, arr) -> "aion.Tensor":
    """Bind an i32 tensor."""
    return aion.Tensor(np.asarray(arr, np.int32).tolist(), ctx=ctx, dtype=aion.AionDType.AION_DTYPE_I32)


def _fit_samples(audio: np.ndarray, n: int) -> np.ndarray:
    if len(audio) < n:
        return np.concatenate([audio, np.zeros(n - len(audio), np.float32)])
    return audio[:n]


def run_offline(model_path: str, md: dict, audio: np.ndarray, threads, args):
    """Run the offline whole-clip model once; returns emitted token ids.

    The fixed audio window (t_mel) comes from package metadata; audio is
    padded/truncated to it. Decode state is exposed as named Loop carries; only
    the inputs with a non-zero initial value are bound (audio, the active flag,
    blank-SOS last_token) — the runtime auto-initializes every unbound input to
    zeros (LSTM state, frame/sym/count bookkeeping, the token buffer).
    """
    t_mel = int(md["t_mel"])
    blank = int(md["blank_id"])
    m, dev = load_model_with_device(model_path, args, thread_count=threads)
    print(f"device: {dev}")
    ctx = m.context
    n = (t_mel - 1) * HOP  # samples whose center-STFT yields t_mel frames
    a = _fit_samples(audio, n).astype(np.float32)
    m.bind_input("audio", aion.Tensor(a.reshape(n, 1), ctx=ctx))
    m.bind_input("active_in", _i32(ctx, [1]))
    m.bind_input("last_in", _i32(ctx, [blank]))
    m.run()
    oc = m.output_tensor("out_count")
    count = int(oc.item())
    oc.close()
    ot = m.output_tensor("out_tokens")
    toks = [int(x) for x in ot.tolist()]
    ot.close()
    return toks[:count]


def _calc_len(length):
    for _ in range(3):
        length = length // 2 + 1
    return length


def run_streaming(model_path: str, md: dict, audio: np.ndarray, sp, threads, args):
    """Cache-aware streaming: one run per chunk, printing the transcript as it grows.

    Per chunk (= att_right+1 encoder frames) it slides a left-context audio
    window and threads ALL state — per-layer projected-K/V + conv caches and the
    decode state (last_token + LSTM h/c) — via IoAlias, so it streams
    indefinitely. All streaming state lives *inside the runtime*: the IoAlias'd
    inputs start at zero, so they are left UNBOUND and the runtime auto-zeros
    each once and carries it across runs (LoadModelOptions.auto_init_inputs).
    Only a few inputs are bound: audio/cache_mask (rewritten in place each
    chunk) and the per-run decode bookkeeping (frame/sym/count/active/toks),
    Loop carries re-seeded to their start values each chunk. `last_in` also
    needs an explicit seed because it starts at the (non-zero) blank SOS token.
    """
    chunk = int(md["chunk"])
    att_left = int(md["att_left"])
    t_kv = int(md["t_kv"])
    wmel = int(md["window_mel"])
    max_out = int(md["max_out"])
    blank = int(md["blank_id"])
    wsamp = (wmel - 1) * HOP
    step = chunk * 8 * HOP                    # new audio samples per chunk
    pad = int(md["sub_drop"]) * 8 * HOP       # front-pad: chunk c window = buf[c*step : +wsamp]
    m, dev = load_model_with_device(model_path, args, thread_count=threads)
    print(f"device: {dev}")
    ctx = m.context
    fn = lambda a: aion.Tensor(a, ctx=ctx)

    # Per-run inputs (rewritten in place each chunk) — bound once.
    t_audio = fn(np.zeros((wsamp, 1), np.float32))
    t_mask = fn(np.zeros((chunk, t_kv), np.float32))
    t_frame, t_sym = _i32(ctx, [0]), _i32(ctx, [0])
    t_count, t_active = _i32(ctx, [0]), _i32(ctx, [1])
    t_toks = _i32(ctx, np.zeros((max_out, 1), np.int32))
    m.bind_input("audio", t_audio); m.bind_input("cache_mask", t_mask)
    m.bind_input("frame_in", t_frame); m.bind_input("sym_in", t_sym)
    m.bind_input("count_in", t_count); m.bind_input("active_in", t_active)
    m.bind_input("toks_in", t_toks)
    m.bind_input("last_in", _i32(ctx, [blank]))

    tokens = []
    buf = np.concatenate([np.zeros(pad, np.float32), audio.astype(np.float32)])
    t_out = _calc_len((len(buf) - pad) // HOP + 1)
    nchunks = (t_out + chunk - 1) // chunk
    for c in range(nchunks):
        w = buf[c * step:c * step + wsamp]
        if len(w) < wsamp:
            w = np.concatenate([w, np.zeros(wsamp - len(w), np.float32)])
        real = min(chunk, t_out - c * chunk)
        thr = max(0, att_left - c * chunk)
        mask = np.full((chunk, t_kv), -1e9, np.float32)
        mask[:, [j for j in range(t_kv) if thr <= j < att_left + real]] = 0.0
        t_audio.copy_from(w.reshape(wsamp, 1))
        t_mask.copy_from(mask)
        t_frame.copy_from([0]); t_sym.copy_from([0])
        t_count.copy_from([0]); t_active.copy_from([1])
        t_toks.copy_from([0] * max_out)
        m.run()
        oc = m.output_tensor("out_count"); cnt = int(oc.item()); oc.close()
        ot = m.output_tensor("out_tokens"); tokens += [int(x) for x in ot.tolist()][:cnt]; ot.close()
        t_ms = (c + 1) * step * 1000 // SAMPLE_RATE
        print(f"[{t_ms:6d}ms] {sp.DecodeIds([int(t) for t in tokens])}")
    return tokens


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="self-contained .aion from convert_nemotron_asr_to_aion.py")
    ap.add_argument("--wav", required=True, help="16 kHz wav to transcribe")
    ap.add_argument("--tokenizer", default=None,
                    help="sentencepiece model (default: nemotron_tokenizer.model next to the model)")
    ap.add_argument("--threads", type=int, default=None)
    add_device_args(ap)
    args = ap.parse_args()

    import soundfile as sf
    import sentencepiece as spm

    audio, sr = sf.read(args.wav)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    assert sr == SAMPLE_RATE, f"expected 16kHz, got {sr}"
    print(f"audio {len(audio)/sr:.1f}s ({len(audio)} samples)")

    tok_path = args.tokenizer or os.path.join(os.path.dirname(os.path.abspath(args.model)),
                                              "nemotron_tokenizer.model")
    sp = spm.SentencePieceProcessor()
    sp.Load(tok_path)

    md = read_aion_metadata(args.model)
    arch = md.get("arch", "")
    if arch == "nemotron-fastconformer-rnnt-stream-full":
        tokens = run_streaming(args.model, md, audio, sp, args.threads, args)
        print(f"\n[streaming, {len(tokens)} tokens] done\n")
    elif arch == "nemotron-fastconformer-rnnt-single":
        tokens = run_offline(args.model, md, audio, args.threads, args)
        text = sp.DecodeIds([int(t) for t in tokens])
        print(f"\n[offline, {len(tokens)} tokens] transcript:\n{text}\n")
    else:
        raise SystemExit(f"{args.model}: unrecognized arch {arch!r} — rebuild with "
                         "scripts/convert_nemotron_asr_to_aion.py")


if __name__ == "__main__":
    main()
