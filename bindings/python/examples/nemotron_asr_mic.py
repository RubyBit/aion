# SPDX-License-Identifier: Apache-2.0
"""Transcribe from the microphone with the Nemotron FastConformer on Aion.

Takes ONE self-contained .aion (built by scripts/convert_nemotron_asr_to_aion.py);
the mode follows the model kind, auto-detected from package metadata:

  offline   (default converter build): record until Enter (or --seconds), then
            transcribe the clip in one run.
  streaming (converter --streaming build): live transcription — one run per
            chunk (~1.1 s at the default [70,13] attention config), threading
            all state inside the runtime via IoAlias, so it streams
            indefinitely (no length cap).

`--wav PATH` replays a 16 kHz wav through the same path instead of the mic.
`--tokenizer` defaults to `nemotron_tokenizer.model` next to the model.

Examples:
  # record 10s then transcribe (offline model):
  uv run --project bindings/python python \
    bindings/python/examples/nemotron_asr_mic.py \
      models/nemotron-asr/nemotron.aion --seconds 10

  # live (streaming model):
  uv run --project bindings/python python \
    bindings/python/examples/nemotron_asr_mic.py \
      models/nemotron-asr/nemotron_stream.aion
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import aion

from _common import add_device_args, load_model_with_device, read_aion_metadata

SR = 16000
HOP = 160


def _i32(ctx, v, shape):
    return aion.Tensor(np.full(shape, v, np.int32).tolist(), ctx=ctx, dtype=aion.AionDType.AION_DTYPE_I32)


# ---------------- record-then-transcribe (offline model) ----------------

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


def transcribe_offline(model_path, md, audio, sp, threads, args):
    t_mel = int(md["t_mel"])  # fixed audio window baked into the model
    blank = int(md["blank_id"])
    n = (t_mel - 1) * HOP
    a = (np.concatenate([audio, np.zeros(n - len(audio), np.float32)]) if len(audio) < n else audio[:n])
    m, dev = load_model_with_device(model_path, args, thread_count=threads)
    print(f"device: {dev}")
    ctx = m.context
    # Only non-zero-seed inputs need binding: the runtime auto-initializes every
    # unbound input to zeros (LSTM state, frame/sym/count bookkeeping, toks_in).
    m.bind_input("audio", aion.Tensor(a.reshape(n, 1), ctx=ctx))
    m.bind_input("active_in", _i32(ctx, 1, (1,)))
    m.bind_input("last_in", _i32(ctx, blank, (1,)))
    m.run()
    oc = m.output_tensor("out_count"); count = int(oc.item()); oc.close()
    ot = m.output_tensor("out_tokens"); toks = [int(x) for x in ot.tolist()][:count]; ot.close()
    return sp.DecodeIds(toks)


# ---------------- live streaming (one self-contained model) ----------------

class StreamFull:
    """Drives the streaming .aion: one run per audio window (chunk).

    Streaming state is threaded *inside the runtime*, not in Python. The 24 attn
    + 24 conv caches and the persistent decode state (last token + LSTM h/c) are
    IoAlias'd output->input in the graph and tagged with `state` roles, so they
    are left UNBOUND: the runtime auto-allocates each once, zeros it, and carries
    it across runs. Only `last_in` needs an explicit bind (non-zero blank-SOS seed).

    Only the per-run inputs change each step: the audio window and chunked mask
    (rewritten in place), and the per-run decode bookkeeping (frame/sym/count/
    active/toks) which are Loop carries that must be re-seeded to their start
    values each chunk. Those are bound once and rewritten in place — per-step
    allocation stays zero (Context-owned tensor storage is never freed by handle
    close, so fresh per-chunk tensors would leak)."""

    def __init__(self, model_path, md, sp, threads, args):
        self.m, dev = load_model_with_device(model_path, args, thread_count=threads)
        print(f"device: {dev}")
        self.ctx = self.m.context
        self.sp = sp
        self.att_left = int(md["att_left"])
        self.chunk = int(md["chunk"])
        self.t_kv = int(md["t_kv"])
        self.wsamp = (int(md["window_mel"]) - 1) * HOP
        self.max_out = int(md["max_out"])
        blank = int(md["blank_id"])
        self.tokens = []
        self.c = 0  # chunk index
        ctx, m = self.ctx, self.m
        fn = lambda a: aion.Tensor(a, ctx=ctx)

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

        # Persistent state (IoAlias'd, zero-init) — the caches and LSTM h/c are
        # left UNBOUND; the runtime auto-zeros each once and carries it across
        # runs. Only the non-zero blank-SOS seed needs an explicit bind.
        m.bind_input("last_in", _i32(ctx, blank, (1,)))

    def step(self, window: np.ndarray, real: int):
        """Process one window; `real` = valid new frames (<=chunk)."""
        m = self.m
        chunk, t_kv, att_left = self.chunk, self.t_kv, self.att_left
        thr = max(0, att_left - self.c * chunk)
        mask = np.full((chunk, t_kv), -1e9, np.float32)
        mask[:, [j for j in range(t_kv) if thr <= j < att_left + real]] = 0.0
        # Rewrite the per-run inputs in place (no new tensors -> no leak).
        self.t_audio.copy_from(window.reshape(self.wsamp, 1))
        self.t_mask.copy_from(mask)
        # Re-seed the per-run decode carries (the Loop leaves final values here).
        self.t_frame.copy_from([0]); self.t_sym.copy_from([0])
        self.t_count.copy_from([0]); self.t_active.copy_from([1])
        self.t_toks.copy_from([0] * self.max_out)
        m.run()
        oc = m.output_tensor("out_count"); cnt = int(oc.item()); oc.close()
        ot = m.output_tensor("out_tokens"); self.tokens += [int(x) for x in ot.tolist()][:cnt]; ot.close()
        self.c += 1

    def text(self):
        return self.sp.DecodeIds([int(t) for t in self.tokens])


def calc_length(length: int) -> int:
    for _ in range(3):
        length = length // 2 + 1
    return length


def run_live(model_path, md, sp, threads, audio_source, args):
    """audio_source yields float32 mono 16k blocks. `buf` is front-padded by `pad`
    so chunk c's window is buf[c*step : c*step+wsamp]."""
    drv = StreamFull(model_path, md, sp, threads, args)
    chunk = drv.chunk
    step = chunk * 8 * HOP                # new audio samples per chunk
    pad = int(md["sub_drop"]) * 8 * HOP   # front-pad matching the model's window layout
    wsamp = drv.wsamp
    buf = np.zeros(pad, np.float32)
    done = 0  # chunks processed
    for block in audio_source:
        buf = np.concatenate([buf, block])
        # Process full windows whose `chunk` frames are all backed by real audio.
        while (done * step + wsamp) <= len(buf):
            drv.step(buf[done * step: done * step + wsamp], chunk)
            done += 1
            print("\r" + drv.text()[-110:], end="", flush=True)
    # Final flush: process remaining frames (partial last chunk), zero-padding the window.
    t_out = calc_length((len(buf) - pad) // HOP + 1)
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="self-contained .aion from convert_nemotron_asr_to_aion.py")
    ap.add_argument("--tokenizer", default=None,
                    help="sentencepiece model (default: nemotron_tokenizer.model next to the model)")
    ap.add_argument("--seconds", type=float, default=None,
                    help="offline model: record duration (default: until Enter)")
    ap.add_argument("--wav", default=None, help="replay a 16kHz wav instead of the mic")
    ap.add_argument("--threads", type=int, default=None)
    add_device_args(ap)
    args = ap.parse_args()

    import sentencepiece as spm
    tok_path = args.tokenizer or os.path.join(os.path.dirname(os.path.abspath(args.model)),
                                              "nemotron_tokenizer.model")
    sp = spm.SentencePieceProcessor()
    sp.Load(tok_path)

    md = read_aion_metadata(args.model)
    arch = md.get("arch", "")
    if arch == "nemotron-fastconformer-rnnt-stream-full":
        block = int(md["chunk"]) * 8 * HOP  # feed one chunk's worth of audio at a time
        src = wav_blocks(args.wav, block) if args.wav else mic_blocks(block)
        run_live(args.model, md, sp, args.threads, src, args)
        return

    if arch != "nemotron-fastconformer-rnnt-single":
        raise SystemExit(f"{args.model}: unrecognized arch {arch!r} — rebuild with "
                         "scripts/convert_nemotron_asr_to_aion.py")
    if args.wav:
        import soundfile as sf
        audio, sr = sf.read(args.wav)
        if audio.ndim > 1:
            audio = audio.mean(1)
        audio = audio.astype(np.float32)
    else:
        audio = record_until_enter(args.seconds)
    print(f"captured {len(audio)/SR:.1f}s")
    print("\ntranscript:\n" + transcribe_offline(args.model, md, audio, sp, args.threads, args))


if __name__ == "__main__":
    main()
