# SPDX-License-Identifier: Apache-2.0
"""Transcribe from the microphone with the Nemotron FastConformer on Aion.

Two modes:

  (default) record-then-transcribe: capture audio until you press Enter (or for
            --seconds), then run the self-contained single model once.

  --live    cache-aware streaming with ONE self-contained model
            (nemotron_stream_full.aion): in-graph log-mel + subsampling + cached
            conformer + in-graph RNNT decode. The driver just slices an overlapping
            audio window per ~1.1 s step and threads all state via IoAlias, so it
            streams indefinitely (no length cap).

`--wav PATH` replays a 16 kHz wav through the same path instead of the mic.

Examples:
  # record 10s then transcribe:
  uv run --project bindings/python --extra dev python \
    bindings/python/examples/nemotron_asr_mic.py \
      --single models/nemotron-asr/nemotron_single_2885.aion \
      --tokenizer models/nemotron-asr/nemotron_tokenizer.model --seconds 10

  # live (one model):
  uv run --project bindings/python --extra dev python \
    bindings/python/examples/nemotron_asr_mic.py --live \
      --stream-full models/nemotron-asr/nemotron_stream_full.aion \
      --tokenizer models/nemotron-asr/nemotron_tokenizer.model
"""
from __future__ import annotations

import argparse
import threading

import numpy as np

import aion

from _common import add_device_args, load_model_with_device

SR = 16000
HOP = 160
BLANK = 1024
PRED_HIDDEN = 640
CONV_LEFT = 8
N_LAYERS = 24
D = 1024
N_HEADS = 8
HEAD_DIM = 128
PAD = 16 * HOP                   # front-pad: chunk c window = buf[c*step : c*step+wsamp] (sub-drop=2 frames)
T_MEL_CAP = 2885                 # for record-then-transcribe single model only
# chunk size / left context are read from --att-right/--att-left (must match the built model).


def calc_length(length: int) -> int:
    for _ in range(3):
        length = length // 2 + 1
    return length


SUB_DROP = 2


def window_mel(chunk: int) -> int:
    """Mel-window size the stream_full model uses for `chunk` (must match the
    converter's stream_window_mel)."""
    need = SUB_DROP + chunk + 1
    w = 40
    while calc_length(w) < need:
        w += 8
    return w


def _i32(ctx, v, shape):
    return aion.Tensor(np.full(shape, v, np.int32).tolist(), ctx=ctx, dtype=aion.AionDType.AION_DTYPE_I32)


# ---------------- record-then-transcribe (single model) ----------------

def record_until_enter(seconds=None) -> np.ndarray:
    import sounddevice as sd
    buf = []
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        callback=lambda i, f, t, s: buf.append(i[:, 0].copy())):
        if seconds:
            print(f"recording {seconds}s ...", flush=True)
            sd.sleep(int(seconds * 1000))
        else:
            print("recording... press Enter to stop.", flush=True)
            input()
    return np.concatenate(buf).astype(np.float32) if buf else np.zeros(0, np.float32)


def transcribe_single(model_path, audio, tokenizer, threads, args):
    import sentencepiece as spm
    n = (T_MEL_CAP - 1) * HOP
    a = (np.concatenate([audio, np.zeros(n - len(audio), np.float32)]) if len(audio) < n else audio[:n])
    m, dev = load_model_with_device(model_path, args, thread_count=threads)
    print(f"device: {dev}")
    ctx = m.context
    m.bind_input("audio", aion.Tensor.from_numpy(ctx, a.reshape(n, 1)))
    m.bind_input("frame_in", _i32(ctx, 0, (1,))); m.bind_input("sym_in", _i32(ctx, 0, (1,)))
    m.bind_input("count_in", _i32(ctx, 0, (1,))); m.bind_input("active_in", _i32(ctx, 1, (1,)))
    m.bind_input("last_in", _i32(ctx, BLANK, (1,)))
    for nm in ("h0_in", "c0_in", "h1_in", "c1_in"):
        m.bind_input(nm, aion.Tensor.from_numpy(ctx, np.zeros((1, PRED_HIDDEN), np.float32)))
    m.bind_input("toks_in", _i32(ctx, 0, (512, 1)))
    m.run()
    oc = m.output_tensor("out_count"); count = int(oc.read_i32()[0]); oc.close()
    ot = m.output_tensor("out_tokens"); toks = [int(x) for x in ot.read_i32()][:count]; ot.close()
    sp = spm.SentencePieceProcessor(); sp.Load(tokenizer)
    return sp.DecodeIds(toks)


# ---------------- live streaming (one self-contained model) ----------------

class StreamFull:
    """Drives nemotron_stream_full.aion: one .aion run per ~1.1s window.

    Streaming state is threaded *inside the runtime*, not in Python. The 24 attn
    + 24 conv caches and the persistent decode state (last token + LSTM h/c) are
    IoAlias'd output->input in the graph, and the decode Loop leaves each carry's
    final value in its own input slot. So those inputs are bound ONCE (zeros) and
    never rebound — the runtime carries them across runs for free.

    This matters for memory: tensor storage is owned by the Context for its whole
    lifetime (destroying a tensor handle frees only the handle, not the backing
    store). Creating fresh input tensors every chunk would therefore leak ~15 MB
    per step (the 24 k/v caches dominate) and climb to multiple GB within a minute.
    Bind-once + write-in-place keeps per-step allocation at zero.

    Only the per-run inputs change each step: the audio window and chunked mask
    (rewritten in place), and the per-run decode bookkeeping (frame/sym/count/
    active/toks) which are Loop carries that must be re-seeded to their start
    values each chunk."""

    def __init__(self, model_path, tokenizer, threads, args, att_left=70, att_right=13):
        import sentencepiece as spm
        self.m, dev = load_model_with_device(model_path, args, thread_count=threads)
        print(f"device: {dev}")
        self.ctx = self.m.context
        self.sp = spm.SentencePieceProcessor(); self.sp.Load(tokenizer)
        self.att_left = att_left
        self.chunk = att_right + 1
        self.t_kv = att_left + self.chunk
        self.wsamp = (window_mel(self.chunk) - 1) * HOP
        self.max_out = 64
        self.tokens = []
        self.c = 0  # chunk index
        ctx, m = self.ctx, self.m
        fn = lambda a: aion.Tensor.from_numpy(ctx, a)

        # Per-run inputs (rewritten in place each step) — bound once.
        self.t_audio = fn(np.zeros((self.wsamp, 1), np.float32))
        self.t_mask = fn(np.zeros((self.chunk, self.t_kv), np.float32))
        self.t_frame = _i32(ctx, 0, (1,)); self.t_sym = _i32(ctx, 0, (1,))
        self.t_count = _i32(ctx, 0, (1,)); self.t_active = _i32(ctx, 1, (1,))
        self.t_toks = _i32(ctx, 0, (self.max_out, 1))
        m.bind_input("audio", self.t_audio); m.bind_input("cache_mask", self.t_mask)
        m.bind_input("frame_in", self.t_frame); m.bind_input("sym_in", self.t_sym)
        m.bind_input("count_in", self.t_count); m.bind_input("active_in", self.t_active)
        m.bind_input("toks_in", self.t_toks)

        # Persistent state (IoAlias'd / self-carried) — bound once to seed values,
        # then threaded by the runtime across runs. Never rebound or rewritten.
        m.bind_input("last_in", _i32(ctx, BLANK, (1,)))
        for nm in ("h0_in", "c0_in", "h1_in", "c1_in"):
            m.bind_input(nm, fn(np.zeros((1, PRED_HIDDEN), np.float32)))
        for li in range(N_LAYERS):
            m.bind_input(f"kcache_{li}_in", fn(np.zeros((1, att_left, N_HEADS, HEAD_DIM), np.float32)))
            m.bind_input(f"vcache_{li}_in", fn(np.zeros((1, att_left, N_HEADS, HEAD_DIM), np.float32)))
            m.bind_input(f"conv_cache_{li}_in", fn(np.zeros((1, CONV_LEFT, D), np.float32)))

    def step(self, window: np.ndarray, real: int):
        """Process one window; `real` = valid new frames (<=chunk)."""
        m = self.m
        chunk, t_kv, att_left = self.chunk, self.t_kv, self.att_left
        thr = max(0, att_left - self.c * chunk)
        mask = np.full((chunk, t_kv), -1e9, np.float32)
        mask[:, [j for j in range(t_kv) if thr <= j < att_left + real]] = 0.0
        # Rewrite the per-run inputs in place (no new tensors -> no leak).
        self.t_audio.write_from_numpy(window.reshape(self.wsamp, 1))
        self.t_mask.write_from_numpy(mask)
        # Re-seed the per-run decode carries (the Loop leaves final values here).
        self.t_frame.write_i32([0]); self.t_sym.write_i32([0])
        self.t_count.write_i32([0]); self.t_active.write_i32([1])
        self.t_toks.write_i32([0] * self.max_out)
        m.run()
        oc = m.output_tensor("out_count"); cnt = int(oc.read_i32()[0]); oc.close()
        ot = m.output_tensor("out_tokens"); self.tokens += [int(x) for x in ot.read_i32()][:cnt]; ot.close()
        self.c += 1

    def text(self):
        return self.sp.DecodeIds([int(t) for t in self.tokens])


def run_live(model_path, tokenizer, threads, audio_source, args, att_left=70, att_right=13):
    """audio_source yields float32 mono 16k blocks. `buf` is front-padded by PAD so
    chunk c's window is buf[c*STEP : c*STEP+WSAMP] (== global mel m_start = chunk*8*c - 16)."""
    drv = StreamFull(model_path, tokenizer, threads, args, att_left, att_right)
    chunk = att_right + 1
    step = chunk * 8 * HOP   # new audio samples per chunk
    wsamp = drv.wsamp
    buf = np.zeros(PAD, np.float32)
    done = 0  # chunks processed
    for block in audio_source:
        buf = np.concatenate([buf, block])
        # Process full windows whose `chunk` frames are all backed by real audio.
        while (done * step + wsamp) <= len(buf):
            drv.step(buf[done * step: done * step + wsamp], chunk)
            done += 1
            print("\r" + drv.text()[-110:], end="", flush=True)
    # Final flush: process remaining frames (partial last chunk), zero-padding the window.
    t_out = calc_length((len(buf) - PAD) // HOP + 1)
    while done * chunk < t_out:
        w = buf[done * step: done * step + wsamp]
        if len(w) < wsamp:
            w = np.concatenate([w, np.zeros(wsamp - len(w), np.float32)])
        drv.step(w, min(chunk, t_out - done * chunk))
        done += 1
        print("\r" + drv.text()[-110:], end="", flush=True)
    print("\n\nFINAL:", drv.text())


def mic_blocks(block):
    import sounddevice as sd
    import queue
    q = queue.Queue()
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=block,
                        callback=lambda i, f, t, s: q.put(i[:, 0].copy())):
        print("live transcribing... Ctrl+C to stop.", flush=True)
        try:
            while True:
                yield q.get()
        except KeyboardInterrupt:
            return


def wav_blocks(path, block):
    import soundfile as sf
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(1)
    audio = audio.astype(np.float32)
    assert sr == SR, f"expected 16kHz, got {sr}"
    for i in range(0, len(audio), block):
        yield audio[i:i + block]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--single", default=None, help="single model (record-then-transcribe mode)")
    ap.add_argument("--live", action="store_true", help="live streaming mode (needs --stream-full)")
    ap.add_argument("--stream-full", default=None, help="self-contained streaming .aion")
    ap.add_argument("--att-left", type=int, default=70, help="left context the model was built with")
    ap.add_argument("--att-right", type=int, default=13, help="right context the model was built with (chunk=att_right+1)")
    ap.add_argument("--seconds", type=float, default=None, help="record duration (default: until Enter)")
    ap.add_argument("--wav", default=None, help="replay a 16kHz wav instead of the mic")
    ap.add_argument("--threads", type=int, default=None)
    add_device_args(ap)
    args = ap.parse_args()

    if args.live:
        assert args.stream_full, "--live needs --stream-full"
        block = (args.att_right + 1) * 8 * HOP  # feed one chunk's worth of audio at a time
        src = wav_blocks(args.wav, block) if args.wav else mic_blocks(block)
        run_live(args.stream_full, args.tokenizer, args.threads, src, args, args.att_left, args.att_right)
        return

    assert args.single, "record-then-transcribe needs --single"
    if args.wav:
        import soundfile as sf
        audio, sr = sf.read(args.wav)
        if audio.ndim > 1:
            audio = audio.mean(1)
        audio = audio.astype(np.float32)
    else:
        audio = record_until_enter(args.seconds)
    print(f"captured {len(audio)/SR:.1f}s")
    print("\ntranscript:\n" + transcribe_single(args.single, audio, args.tokenizer, args.threads, args))


if __name__ == "__main__":
    main()
