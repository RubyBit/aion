# SPDX-License-Identifier: Apache-2.0
"""Transcribe speech with the Nemotron FastConformer-Transducer on Aion.

Log-mel is computed IN-GRAPH (no Python preprocessing); models take raw 16 kHz
audio. Three modes:

  --single    one self-contained .aion: in-graph log-mel + encoder + in-graph
              greedy RNNT decode. Whole-clip.
  --stream    cache-aware streaming: per-layer attention(70)/conv(8) caches +
              RNNT decode state threaded across chunks (chunk = att_right+1 enc
              frames). The recurrent conformer (bit-exact vs the offline
              [70,13] encoder) and decode are streamed; subsampling runs up-front.
  (default)   two-graph: audio-input encoder.aion + per-step decoder.aion + a
              Python greedy loop.

Models are produced by scripts/convert_nemotron_asr_to_aion.py --component
{single,stream,subsample,encoder,decoder,assets}.

Run (single, fully self-contained):
  uv run --project bindings/python --extra dev python \
    bindings/python/examples/nemotron_asr_streaming.py \
      --single models/nemotron-asr/nemotron_single_2885.aion \
      --tokenizer models/nemotron-asr/nemotron_tokenizer.model \
      --wav models/nemotron-asr/test_audio.wav

Run (cache-aware streaming):
  ... --stream --subsample models/nemotron-asr/nemotron_subsample_2885.aion \
      --stream-conf models/nemotron-asr/nemotron_stream_conf.aion \
      --decoder models/nemotron-asr/nemotron_decoder.aion \
      --tokenizer ... --wav ...
"""
from __future__ import annotations

import argparse

import numpy as np

import aion

SAMPLE_RATE = 16000
N_FFT = 512
WIN = 400
HOP = 160
N_MELS = 128
PREEMPH = 0.97
LOG_GUARD = 2.0 ** -24
BLANK = 1024
D_MODEL = 1024
PRED_HIDDEN = 640


def log_mel(audio: np.ndarray, window: np.ndarray, fb: np.ndarray) -> np.ndarray:
    """NeMo AudioToMelSpectrogramPreprocessor (normalize='NA' => no normalization)."""
    import torch

    x = torch.from_numpy(audio.astype(np.float32))
    # preemphasis
    x = torch.cat([x[:1], x[1:] - PREEMPH * x[:-1]])
    win = torch.from_numpy(window.astype(np.float32))
    spec = torch.stft(x, n_fft=N_FFT, hop_length=HOP, win_length=WIN, window=win,
                      center=True, pad_mode="reflect", return_complex=True)  # [257, frames]
    power = (spec.real ** 2 + spec.imag ** 2)  # mag_power=2.0
    mel = torch.from_numpy(fb.astype(np.float32)) @ power  # [128, frames]
    feat = torch.log(mel + LOG_GUARD)
    return feat.transpose(0, 1).contiguous().numpy()  # [frames, 128]


def greedy_decode(decoder, enc: np.ndarray, max_symbols: int = 10):
    """RNNT greedy decode. enc: [T, D_MODEL]. Returns list of emitted token ids."""
    ctx = decoder.context
    T = enc.shape[0]
    z = lambda: np.zeros((1, PRED_HIDDEN), np.float32)
    h0, c0, h1, c1 = z(), z(), z(), z()
    last = BLANK
    tokens = []
    t = 0
    while t < T:
        emitted = 0
        while emitted < max_symbols:
            tok = aion.Tensor([[int(last)]], ctx=ctx, dtype=aion.AionDType.AION_DTYPE_I32)
            res = decoder.run_numpy(
                {
                    "enc_frame": enc[t:t + 1][None],  # [1,1,D]
                    "token": tok,
                    "h0_in": h0, "c0_in": c0, "h1_in": h1, "c1_in": c1,
                },
                outputs=["logits", "h0_out", "c0_out", "h1_out", "c1_out"],
            )
            k = int(np.argmax(res["logits"][0, 0]))
            if k == BLANK:
                break
            tokens.append(k)
            last = k
            h0, c0, h1, c1 = res["h0_out"], res["c0_out"], res["h1_out"], res["c1_out"]
            emitted += 1
        t += 1
    return tokens


def _fit_window(feats: np.ndarray, t_window: int) -> np.ndarray:
    if feats.shape[0] < t_window:
        return np.concatenate([feats, np.zeros((t_window - feats.shape[0], N_MELS), np.float32)], axis=0)
    return feats[:t_window]


def _i32(ctx, arr) -> "aion.Tensor":
    """Bind an i32 tensor (from_numpy defaults to f32, so build i32 explicitly)."""
    return aion.Tensor(np.asarray(arr, np.int32).tolist(), ctx=ctx, dtype=aion.AionDType.AION_DTYPE_I32)


def _fit_samples(audio: np.ndarray, n: int) -> np.ndarray:
    if len(audio) < n:
        return np.concatenate([audio, np.zeros(n - len(audio), np.float32)])
    return audio[:n]


def run_single_file(model_path: str, audio: np.ndarray, t_mel: int, max_out: int, threads):
    """Run the all-in-one .aion (in-graph log-mel + encoder + in-graph RNNT decode).

    Takes raw 16 kHz audio — mel is computed inside the graph. Decode state is
    exposed as named carries; the initial state is fresh bookkeeping
    (frame/sym/count = 0, active = 1) plus blank-SOS last_token and zero LSTM
    state. (For streaming, last_out + h*/c*_out are IoAlias'd back.)
    """
    m = aion.load_model(model_path, thread_count=threads)
    ctx = m.context
    n = (t_mel - 1) * HOP  # samples whose center-STFT yields t_mel frames
    a = _fit_samples(audio, n).astype(np.float32)
    m.bind_input("audio", aion.Tensor.from_numpy(ctx, a.reshape(n, 1)))
    m.bind_input("frame_in", _i32(ctx, [0]))
    m.bind_input("sym_in", _i32(ctx, [0]))
    m.bind_input("count_in", _i32(ctx, [0]))
    m.bind_input("active_in", _i32(ctx, [1]))
    m.bind_input("last_in", _i32(ctx, [BLANK]))
    zero = np.zeros((1, PRED_HIDDEN), np.float32)
    for nm in ("h0_in", "c0_in", "h1_in", "c1_in"):
        m.bind_input(nm, aion.Tensor.from_numpy(ctx, zero))
    m.bind_input("toks_in", _i32(ctx, np.zeros((max_out, 1), np.int32)))
    m.run()
    oc = m.output_tensor("out_count")
    count = int(oc.read_i32()[0])
    oc.close()
    ot = m.output_tensor("out_tokens")
    toks = [int(x) for x in ot.read_i32()]
    ot.close()
    return toks[:count]


def _calc_len(length):
    for _ in range(3):
        length = length // 2 + 1
    return length


def run_streaming(stream_full_path, audio, tokenizer, threads, att_left=70, att_right=6):
    """Cache-aware streaming with ONE self-contained model (in-graph mel + subsample
    + cached conformer + in-graph RNNT decode). Per chunk (= att_right+1 encoder
    frames) it slides a left-context audio window and threads ALL state — per-layer
    projected-K/V + conv caches and the decode state (last_token + LSTM h/c) — via
    IoAlias, so it streams indefinitely. Prints the transcript as it grows.

    att_right must match the model (trained latencies: 13/6/1/0).

    All streaming state is threaded *inside the runtime*: the per-layer K/V + conv
    caches and the persistent decode state (last_token + LSTM h/c) are IoAlias'd
    output->input, and each decode-Loop carry leaves its final value in its own
    input slot. So those inputs are bound ONCE (zeros) and never rebound — the
    runtime carries them across runs. Tensor storage is owned by the Context for
    its whole lifetime, so creating fresh inputs per chunk would leak ~15 MB/step
    (the 24 K/V caches dominate). Bind-once + write-in-place => zero per-step alloc.
    Only the per-run inputs change: audio/cache_mask (rewritten in place) and the
    per-run decode bookkeeping (frame/sym/count/active/toks), which are Loop carries
    re-seeded to their start values each chunk."""
    import sentencepiece as spm
    chunk = att_right + 1
    t_kv = att_left + chunk
    nl, d, head, hd = 24, 1024, 8, 128
    need = 2 + chunk + 1; wmel = 40   # mel window sized to chunk (matches converter)
    while _calc_len(wmel) < need:
        wmel += 8
    wsamp = (wmel - 1) * HOP
    step = chunk * 8 * HOP            # new audio samples per chunk
    pad = 16 * HOP                   # front-pad: chunk c window = buf[c*step : +wsamp]
    max_out = 64
    m = aion.load_model(stream_full_path, thread_count=threads)
    ctx = m.context
    fn = lambda a: aion.Tensor.from_numpy(ctx, a)
    sp = spm.SentencePieceProcessor(); sp.Load(tokenizer)

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
    # Persistent state (IoAlias'd / self-carried) — bound once, threaded by runtime.
    m.bind_input("last_in", _i32(ctx, [BLANK]))
    for nm in ("h0_in", "c0_in", "h1_in", "c1_in"):
        m.bind_input(nm, fn(np.zeros((1, PRED_HIDDEN), np.float32)))
    for li in range(nl):
        m.bind_input(f"kcache_{li}_in", fn(np.zeros((1, att_left, head, hd), np.float32)))
        m.bind_input(f"vcache_{li}_in", fn(np.zeros((1, att_left, head, hd), np.float32)))
        m.bind_input(f"conv_cache_{li}_in", fn(np.zeros((1, 8, d), np.float32)))

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
        t_audio.write_from_numpy(w.reshape(wsamp, 1))
        t_mask.write_from_numpy(mask)
        t_frame.write_i32([0]); t_sym.write_i32([0])
        t_count.write_i32([0]); t_active.write_i32([1])
        t_toks.write_i32([0] * max_out)
        m.run()
        oc = m.output_tensor("out_count"); cnt = int(oc.read_i32()[0]); oc.close()
        ot = m.output_tensor("out_tokens"); tokens += [int(x) for x in ot.read_i32()][:cnt]; ot.close()
        t_ms = (c + 1) * step * 1000 // SAMPLE_RATE
        print(f"[{t_ms:6d}ms] {sp.DecodeIds([int(t) for t in tokens])}")
    return tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", default=None, help="all-in-one .aion (encoder + in-graph decode)")
    ap.add_argument("--stream", action="store_true", help="cache-aware streaming (needs --stream-full)")
    ap.add_argument("--stream-full", default=None, help="self-contained streaming .aion")
    ap.add_argument("--att-right", type=int, default=6, help="right context the --stream-full model was built with (13/6/1/0)")
    ap.add_argument("--encoder", default=None, help="encoder .aion (two-graph mode)")
    ap.add_argument("--decoder", default=None, help="decoder .aion (two-graph mode)")
    ap.add_argument("--assets", default=None, help="(only the two-graph path needs window/fb)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--t-mel", type=int, default=None, help="fixed mel window (else inferred from filename)")
    ap.add_argument("--max-out", type=int, default=512, help="single-file: output token buffer size")
    ap.add_argument("--max-symbols", type=int, default=10)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    import re
    import soundfile as sf
    import sentencepiece as spm

    audio, sr = sf.read(args.wav)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    assert sr == SAMPLE_RATE, f"expected 16kHz, got {sr}"
    print(f"audio {len(audio)/sr:.1f}s ({len(audio)} samples)")

    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)

    # Both the single model and the (audio-input) encoder compute log-mel in-graph,
    # so a fixed window is set by the model's audio length, T_mel = filename suffix.
    def infer_t_mel(path: str) -> int:
        mm = re.search(r"_(\d+)\.aion$", path)
        if mm:
            return int(mm.group(1))
        return 1 + len(audio) // HOP  # center-STFT frame count

    if args.stream:
        assert args.stream_full, "streaming needs --stream-full"
        tokens = run_streaming(args.stream_full, audio, args.tokenizer, args.threads,
                               att_right=args.att_right)
        print(f"\n[streaming, {len(tokens)} tokens] done\n")
        return

    if args.single is not None:
        t_mel = args.t_mel or infer_t_mel(args.single)
        tokens = run_single_file(args.single, audio, t_mel, args.max_out, args.threads)
        text = sp.DecodeIds([int(t) for t in tokens])
        print(f"\n[single-file, {len(tokens)} tokens] transcript:\n{text}\n")
        return

    assert args.encoder and args.decoder, "two-graph mode needs --encoder and --decoder"
    enc_model = aion.load_model(args.encoder, thread_count=args.threads)
    t_mel = args.t_mel or infer_t_mel(args.encoder)
    n = (t_mel - 1) * HOP
    a = _fit_samples(audio, n).astype(np.float32)
    enc = enc_model.run_numpy({"audio": a.reshape(n, 1)}, outputs=["encoder_out"])["encoder_out"][0]
    print(f"encoder out {enc.shape}")

    decoder = aion.load_model(args.decoder, thread_count=args.threads)
    tokens = greedy_decode(decoder, enc, max_symbols=args.max_symbols)
    text = sp.DecodeIds([int(t) for t in tokens])
    print(f"\n[{len(tokens)} tokens] transcript:\n{text}\n")


if __name__ == "__main__":
    main()
