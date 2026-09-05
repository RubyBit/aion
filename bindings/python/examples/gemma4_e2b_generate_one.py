# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Iterable, List, Optional, Sequence

import numpy as np

import aion

from _common import add_device_args, load_model_with_device


def _read_next_token(model) -> int:
    """Greedy token id from the model's device-side `next_token` output.

    The graph selects the final sequence row before the tied head, so this is
    `[B, 1]` for both prefill and decode: the only row is the one to sample.
    """
    t = model.output_tensor("next_token")
    try:
        return int(t.numpy()[0, -1])
    finally:
        t.close()


class Sampler:
    """Temperature / top-k / top-p sampling over the model's device-side top-k.

    The graph hands back the 64 best logits and their ids, so a step reads ~512
    bytes instead of the 1 MiB the full vocabulary row would cost. Policy stays
    here: the model contains no RNG, and the same seed reproduces a run exactly.

    `temperature == 0` means greedy, which is just the first entry — the top-k
    output is already sorted best-first.
    """

    # Markers that only ever appear in a PROMPT: the modality blocks the processor
    # splices in, and pad. The model will happily emit `<|image>` as its first reply
    # token and then loop on it, so a serving layer has to take them off the table.
    SUPPRESSED = frozenset({
        0,       # <pad>
        255999,  # <|image>
        256000,  # <|audio>
        258880,  # <|image|>
        258881,  # <|audio|>
        258882,  # <image|>
        258883,  # <audio|>
    })

    def __init__(self, *, temperature: float, top_k: int, top_p: float, seed: int):
        if temperature < 0.0:
            raise ValueError("temperature must be >= 0")
        if not (0.0 < top_p <= 1.0):
            raise ValueError("top-p must be in (0, 1]")
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.rng = np.random.default_rng(seed)

    def __call__(self, model) -> int:
        vt = model.output_tensor("topk_values")
        it = model.output_tensor("topk_indices")
        try:
            values = np.asarray(vt.numpy(), dtype=np.float64).reshape(-1)
            ids = np.asarray(it.numpy()).reshape(-1)
        finally:
            vt.close()
            it.close()

        keep = np.array([int(i) not in self.SUPPRESSED for i in ids], dtype=bool)
        if keep.any():
            values, ids = values[keep], ids[keep]

        if self.temperature == 0.0:
            return int(ids[0])

        if 0 < self.top_k < values.size:
            values = values[: self.top_k]
            ids = ids[: self.top_k]

        # Subtracting the max before exp is what keeps a large logit finite.
        probs = np.exp((values - values[0]) / self.temperature)
        probs /= probs.sum()

        if self.top_p < 1.0:
            # Values arrive sorted, so the nucleus is a prefix: keep entries up to
            # and including the one that crosses `top_p`.
            cutoff = int(np.searchsorted(np.cumsum(probs), self.top_p)) + 1
            probs = probs[:cutoff]
            ids = ids[:cutoff]
            probs /= probs.sum()

        return int(self.rng.choice(ids, p=probs))


def default_model_path() -> Path:
    # bindings/python/examples -> repo root
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "models" / "gemma" / "gemma4_e2b_q8.aion"


def default_tokenizer_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "models" / "gemma" / "tokenizer.json"


class _Tokenizer:
    """Tiny wrapper to normalize multiple tokenizer backends."""

    # Gemma4 special token IDs (from the JAX reference / upstream tokenizer wrapper).
    PAD: int = 0
    EOS: int = 1
    BOS: int = 2
    UNK: int = 3
    MASK: int = 4
    THINK: int = 98
    START_OF_TURN: int = 105
    END_OF_TURN: int = 106
    BEGIN_OF_IMAGE: int = 255999
    BEGIN_OF_AUDIO: int = 256000
    IMAGE: int = 258880
    AUDIO: int = 258881
    END_OF_IMAGE: int = 258882
    END_OF_AUDIO: int = 258883

    def __init__(self, kind: str, backend: object):
        self.kind = kind
        self.backend = backend

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        if self.kind == "hf_tokenizers_json":
            # Match upstream Gemma4Tokenizer: do not implicitly add special tokens.
            enc = self.backend.encode(text, add_special_tokens=False) # type: ignore
            ids = list(enc.ids)
            out = [int(x) for x in ids]
        elif self.kind == "sentencepiece":
            ids = self.backend.EncodeAsIds(text) # type: ignore (TODO: Test this path)
            out = [int(x) for x in ids]
        else:
            raise RuntimeError(f"unknown tokenizer kind: {self.kind}")

        if add_bos:
            out.insert(0, int(self.BOS))
        if add_eos:
            out.append(int(self.EOS))
        return out

    def decode(self, ids: Sequence[int]) -> str:
        if self.kind == "hf_tokenizers_json":
            return str(self.backend.decode(list(ids), skip_special_tokens=False)) # type: ignore
        if self.kind == "sentencepiece":
            return str(self.backend.DecodeIds([int(x) for x in ids])) # type: ignore (TODO: Test this path)
        raise RuntimeError(f"unknown tokenizer kind: {self.kind}")


def _load_tokenizer(path: str) -> _Tokenizer:
    """Load a tokenizer from a local file or directory.

    Supported inputs:
      - tokenizer.json (Hugging Face `tokenizers`)
      - tokenizer.model (SentencePiece)
      - a directory containing either of the above
    """

    def load_json(p: Path) -> _Tokenizer:
        try:
            from tokenizers import Tokenizer
        except Exception as e:  # pragma: no cover
            raise SystemExit(
                "Hugging Face tokenizers is required for tokenizer.json; install via: uv pip install tokenizers"
            ) from e
        return _Tokenizer("hf_tokenizers_json", Tokenizer.from_file(str(p)))

    def load_spm(p: Path) -> _Tokenizer:
        try:
            import sentencepiece as spm
        except Exception as e:  # pragma: no cover
            raise SystemExit(
                "sentencepiece is required for tokenizer.model; install via: uv pip install sentencepiece"
            ) from e
        proc = spm.SentencePieceProcessor()
        if not proc.Load(str(p)):
            raise SystemExit(f"failed to load sentencepiece model: {p}")
        return _Tokenizer("sentencepiece", proc)

    p = Path(path).expanduser().resolve()

    if p.is_dir():
        json_p = p / "tokenizer.json"
        spm_p = p / "tokenizer.model"
        if json_p.is_file():
            return load_json(json_p)
        if spm_p.is_file():
            return load_spm(spm_p)
        raise SystemExit(f"tokenizer directory has no tokenizer.json or tokenizer.model: {p}")

    if not p.is_file():
        raise SystemExit(f"tokenizer not found: {p}")

    suf = p.suffix.lower()
    if suf == ".json":
        return load_json(p)
    if suf in {".model", ".spm"}:
        return load_spm(p)
    raise SystemExit(f"unsupported tokenizer file extension {p.suffix!r}: {p}")


def _encode_prompt(tokenizer: Optional[_Tokenizer], prompt: str, *, add_bos: bool, add_eos: bool) -> List[int]:
    if tokenizer is None:
        # Fallback: a single token id so we can at least validate end-to-end execution.
        # (This is NOT a real text encoding.)
        return [0]

    ids = tokenizer.encode(prompt, add_bos=add_bos, add_eos=add_eos)
    if len(ids) == 0:
        raise ValueError("tokenizer produced an empty id list")
    return ids


def _decode_ids(tokenizer: Optional[_Tokenizer], ids: Iterable[int]) -> str:
    if tokenizer is None:
        return ""
    return str(tokenizer.decode(list(ids)))


def _ns_to_ms(ns: int) -> float:
    return float(ns) / 1.0e6


def _gemma_chat_ids(
    tokenizer: _Tokenizer,
    content_ids: Sequence[int],
    prompt: str,
    *,
    system: Optional[str] = None,
    think: bool = False,
) -> List[int]:
    """One user turn in Gemma 4's canonical chat format.

    Mirrors `chat_template.jinja` from the checkpoint for the single-message case:

        <bos>[<|turn>system\n[<|think|>\n][system]<turn|>\n]<|turn>user\n
        {content}<turn|>\n<|turn>model\n

    The system turn appears only when there is a system message or thinking is on,
    exactly as the template gates it. It is worth passing one: without it this
    checkpoint tends to continue the conversation itself instead of answering.
    """
    ids = [tokenizer.BOS]
    if think or system:
        ids += [tokenizer.START_OF_TURN]
        ids += tokenizer.encode("system\n")
        if think:
            ids += [tokenizer.THINK]
            ids += tokenizer.encode("\n")
        if system:
            ids += tokenizer.encode(system.strip())
        ids += [tokenizer.END_OF_TURN]
        ids += tokenizer.encode("\n")
    ids += [tokenizer.START_OF_TURN]
    ids += tokenizer.encode("user\n")
    ids += [int(x) for x in content_ids]
    ids += tokenizer.encode(prompt)
    ids += [tokenizer.END_OF_TURN]
    ids += tokenizer.encode("\n")
    ids += [tokenizer.START_OF_TURN]
    ids += tokenizer.encode("model\n")
    return ids


def _mel_filter_bank() -> np.ndarray:
    """Gemma's 257-bin -> 128-bin HTK mel filter bank."""
    fft_freqs = np.linspace(0.0, 8000.0, 257)
    min_mel = 2595.0 * np.log10(1.0)
    max_mel = 2595.0 * np.log10(1.0 + 8000.0 / 700.0)
    mel_freqs = np.linspace(min_mel, max_mel, 130)
    filter_freqs = 700.0 * (np.power(10.0, mel_freqs / 2595.0) - 1.0)
    diffs = np.diff(filter_freqs)
    slopes = filter_freqs[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / diffs[:-1]
    up = slopes[:, 2:] / diffs[1:]
    return np.maximum(0.0, np.minimum(down, up))


def _prepare_audio(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read 16 kHz audio and reproduce Gemma's local log-mel frontend."""
    try:
        import soundfile as sf
    except Exception as exc:  # pragma: no cover
        raise SystemExit("--audio requires soundfile; install via: uv pip install soundfile") from exc

    samples, rate = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim == 2:
        samples = samples.mean(axis=1, dtype=np.float32)
    if samples.ndim != 1:
        raise SystemExit(f"unsupported audio shape: {samples.shape}")
    if int(rate) != 16_000:
        raise SystemExit(f"Gemma audio must be 16 kHz; got {rate} Hz")

    # The reference accepts at most 30 seconds and pads the waveform length to a
    # multiple of 128 before applying semicausal (160-sample left) padding.
    samples = np.asarray(samples[:480_000], dtype=np.float32)
    original_length = samples.shape[0]
    padded_length = ((max(original_length, 1) + 127) // 128) * 128
    waveform = np.pad(samples, (160, padded_length - original_length))
    sample_mask = np.pad(
        np.ones(original_length, dtype=np.bool_),
        (160, padded_length - original_length),
    )

    # Frames contain 321 samples because the last is reserved for the optional
    # pre-emphasis operation. Gemma has pre-emphasis disabled, so use the first 320.
    frame_count = (waveform.shape[0] - 321) // 160 + 1
    if frame_count <= 0:
        raise SystemExit("audio is too short to produce a feature frame")
    frames = np.lib.stride_tricks.as_strided(
        waveform,
        shape=(frame_count, 321),
        strides=(waveform.strides[0] * 160, waveform.strides[0]),
    )[:, :320]
    n = np.arange(320, dtype=np.float32)
    window = (0.5 - 0.5 * np.cos(2.0 * np.pi * n / 320.0)).astype(np.float32)
    magnitude = np.abs(np.fft.rfft(frames * window, n=512, axis=-1))
    features = np.log(magnitude @ _mel_filter_bank() + np.float64(1.0e-3))

    frame_ends = np.arange(frame_count) * 160 + 320
    mask = sample_mask[frame_ends]
    features = (features * mask[:, None]).astype(np.float32)
    return features[None, ...], mask[None, ...].astype(np.float32)


def _image_target_size(height: int, width: int) -> tuple[int, int]:
    max_patches = 280 * 9
    patch_size = 16
    side_multiple = 3 * patch_size
    factor = math.sqrt(max_patches * patch_size**2 / (height * width))
    target_h = int(math.floor(factor * height / side_multiple)) * side_multiple
    target_w = int(math.floor(factor * width / side_multiple)) * side_multiple
    max_side = (max_patches // 9) * side_multiple
    if target_h == 0 and target_w == 0:
        raise SystemExit("image aspect ratio is too extreme for Gemma's patch budget")
    if target_h == 0:
        target_h = side_multiple
        target_w = min(int(math.floor(width / height)) * side_multiple, max_side)
    elif target_w == 0:
        target_w = side_multiple
        target_h = min(int(math.floor(height / width)) * side_multiple, max_side)
    return target_h, target_w


def _prepare_image(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Resize, patchify, and position one image in Gemma's expected layout."""
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise SystemExit("--image requires pillow; install via: uv pip install pillow") from exc

    with Image.open(path) as source:
        image = source.convert("RGB")
        target_h, target_w = _image_target_size(image.height, image.width)
        image = image.resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
        pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0

    patch_h, patch_w = target_h // 16, target_w // 16
    patches = pixels.reshape(3, patch_h, 16, patch_w, 16)
    patches = patches.transpose(1, 3, 2, 4, 0).reshape(patch_h * patch_w, 768)
    yy, xx = np.meshgrid(np.arange(patch_h), np.arange(patch_w), indexing="ij")
    positions = np.stack((xx, yy), axis=-1).reshape(-1, 2).astype(np.int32)
    real_patches = patches.shape[0]
    patches = np.pad(patches, ((0, 2520 - real_patches), (0, 0)))
    positions = np.pad(positions, ((0, 2520 - real_patches), (0, 0)), constant_values=-1)
    return patches[None, ...].astype(np.float32), positions[None, ...], real_patches // 9


def _pool_matrix(position_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact Gemma 4 position-driven 3x3 average-pooling geometry."""
    if position_ids.shape[0] != 1:
        raise SystemExit("this example currently accepts one image per prompt")
    positions = position_ids[0]
    p = positions.shape[0]
    out_len = p // 9
    valid_patch = np.all(positions != -1, axis=-1)
    clamped = np.maximum(positions, 0)
    max_x = int(clamped[valid_patch][:, 0].max()) + 1
    groups = clamped[:, 0] // 3 + (max_x // 3) * (clamped[:, 1] // 3)
    pool = np.zeros((1, out_len, p), dtype=np.float32)
    pool[0, groups, np.arange(p)] = 1.0 / 9.0
    # A padding patch has position (-1, -1), which clamps into group 0 -- so every
    # one of them would otherwise be averaged into the image's first soft token
    # (216 extra patches at 1/9 each, i.e. 25x too large). The reference zeroes the
    # padded rows before pooling; dropping their weights here is the same thing,
    # and keeps the divisor at the fixed k^2 the reference uses.
    pool[0][:, ~valid_patch] = 0.0
    valid_groups = np.any(pool[0] != 0, axis=1)
    return pool, valid_groups


def _prepare_multimodal(args) -> tuple[List[int], _Tokenizer, dict[str, np.ndarray]]:
    tokenizer = _load_tokenizer(args.tokenizer)
    content_ids: List[int] = []
    pixels = pos = features = mask0 = None
    if args.image is not None:
        pixels, pos, image_tokens = _prepare_image(args.image)
        content_ids += [tokenizer.BEGIN_OF_IMAGE]
        content_ids += [tokenizer.IMAGE] * image_tokens
        content_ids += [tokenizer.END_OF_IMAGE]
    if args.audio is not None:
        features, mask0 = _prepare_audio(args.audio)
        mask1 = mask0[:, ::2]
        mask2 = mask1[:, ::2]
        audio_tokens = int((mask2[0] != 0).sum())
        content_ids += [tokenizer.BEGIN_OF_AUDIO]
        content_ids += [tokenizer.AUDIO] * audio_tokens
        content_ids += [tokenizer.END_OF_AUDIO]

    original_ids = np.asarray(
        [_gemma_chat_ids(tokenizer, content_ids, args.prompt, system=args.system, think=args.think)], dtype=np.int32
    )
    seq = original_ids.shape[1]
    image_slots = np.flatnonzero(original_ids[0] == tokenizer.IMAGE)
    audio_slots = np.flatnonzero(original_ids[0] == tokenizer.AUDIO)
    multimodal = (original_ids == tokenizer.IMAGE) | (original_ids == tokenizer.AUDIO)
    llm_ids = original_ids.copy()
    llm_ids[multimodal] = 0

    bindings: dict[str, np.ndarray] = {
        "has_image": np.array([int(args.image is not None)], np.int32),
        "has_audio": np.array([int(args.audio is not None)], np.int32),
        "text_embedding_mask": (~multimodal)[..., None].astype(np.float32),
    }

    if args.image is not None:
        assert pixels is not None and pos is not None
        pool, valid_groups = _pool_matrix(pos)
        valid_count = int(valid_groups.sum())
        if valid_count != len(image_slots):
            raise SystemExit(f"image soft-token mismatch: prompt={len(image_slots)}, tower={valid_count}")
        place = np.zeros((1, seq, pool.shape[1]), np.float32)
        place[0, image_slots, np.flatnonzero(valid_groups)] = 1.0
        bindings.update({
            "pixel_values": pixels,
            "image_position_x": np.maximum(pos[..., 0], 0),
            "image_position_y": np.maximum(pos[..., 1], 0),
            "vision_lengths": np.array([np.all(pos != -1, axis=-1).sum()], np.int32),
            "image_pool_map": pool,
            "image_embed_map": place,
        })
    else:
        bindings.update({
            "pixel_values": np.zeros((1, 9, 768), np.float32),
            "image_position_x": np.zeros((1, 9), np.int32),
            "image_position_y": np.zeros((1, 9), np.int32),
            "vision_lengths": np.array([1], np.int32),
            "image_pool_map": np.zeros((1, 1, 9), np.float32),
            "image_embed_map": np.zeros((1, seq, 1), np.float32),
        })

    if args.audio is not None:
        assert features is not None and mask0 is not None
        mask1 = mask0[:, ::2]
        mask2 = mask1[:, ::2]
        valid = mask2[0] != 0
        if int(valid.sum()) != len(audio_slots):
            raise SystemExit(f"audio soft-token mismatch: prompt={len(audio_slots)}, tower={int(valid.sum())}")
        place = np.zeros((1, seq, mask2.shape[1]), np.float32)
        place[0, audio_slots, np.flatnonzero(valid)] = 1.0
        additive = np.where(valid[None, :], 0.0, -1.0e9).astype(np.float32)
        bindings.update({
            "input_features": features,
            "input_features_mask": mask0[..., None],
            "audio_subsample1_mask": mask1[..., None, None],
            "audio_attention_mask": np.broadcast_to(additive, (mask2.shape[1], mask2.shape[1])).copy(),
            "audio_embed_map": place,
        })
    else:
        bindings.update({
            "input_features": np.zeros((1, 4, 128), np.float32),
            "input_features_mask": np.ones((1, 4, 1), np.float32),
            "audio_subsample1_mask": np.ones((1, 2, 1, 1), np.float32),
            "audio_attention_mask": np.zeros((1, 1), np.float32),
            "audio_embed_map": np.zeros((1, seq, 1), np.float32),
        })
    return llm_ids[0].tolist(), tokenizer, bindings


def main() -> None:
    p = argparse.ArgumentParser(description="Gemma 4 E2B: prefill + generate 1+ token(s) using aion")
    p.add_argument("--model", type=Path, default=default_model_path(), help="Path to gemma4_e2b_q8.aion")
    p.add_argument(
        "--tokenizer",
        type=str,
        default=str(default_tokenizer_path()),
        help=(
            "Local tokenizer.json/tokenizer.model path, or a directory containing one. "
            "If omitted, uses a dummy token id (useful for kernel debugging but not real text)."
        ),
    )
    # Back-compat flag.
    p.add_argument(
        "--tokenizer-json",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument("--prompt", type=str, default="Hello")
    p.add_argument("--image", type=Path, default=None, help="Optional image for a multimodal prompt")
    p.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Optional audio file for a multimodal prompt (16 kHz wav). Note: this "
        "checkpoint does not usefully transcribe -- the same input through the "
        "reference Gemma4Processor + f32 model returns whitespace too. The path is "
        "numerically faithful; the model is the limit.",
    )
    p.add_argument("--add-bos", action="store_true", help="Prepend BOS token (requires tokenizer)")
    p.add_argument("--add-eos", action="store_true", help="Append EOS token (requires tokenizer)")
    p.add_argument(
        "--chat",
        action="store_true",
        help=(
            "Use Gemma chat prompt template: <BOS><START_OF_TURN>user\\n...<END_OF_TURN>\\n"
            "<START_OF_TURN>model\\n"
        ),
    )
    p.add_argument(
        "--stop-on-eos",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Stop early when EOS is generated (default: on for --chat)",
    )
    p.add_argument(
        "--stop-on-end-of-turn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Stop early when END_OF_TURN is generated (default: on for --chat)",
    )
    p.add_argument("--max-new-tokens", type=int, default=1)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; 0 (default) is greedy. Greedy repeats itself on "
        "this checkpoint, so a chat-style run wants ~1.0.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=32,
        help="Keep only the k best candidates, capped at the number the model "
        "returns (TOPK_OUT in the converter, currently 32).",
    )
    p.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus cutoff over the kept candidates [default: 0.95]",
    )
    p.add_argument("--seed", type=int, default=0, help="Sampling seed (reproducible).")
    p.add_argument(
        "--system",
        type=str,
        default=None,
        help="System message for --chat. The template emits a system turn only when "
        "one is given; without it this checkpoint tends to answer its own question.",
    )
    p.add_argument(
        "--think",
        action="store_true",
        help="Open the template's thinking channel (<|think|>) in the system turn.",
    )
    p.add_argument("--thread-count", type=int, default=1, help="Context thread count (default: 1)")
    add_device_args(p)
    p.add_argument("--timing", action="store_true", help="Print per-stage timing and throughput")
    p.add_argument("--timing-json", action="store_true", help="Emit one structured aion.model_timing.v1 JSON line")
    p.add_argument(
        "--profile-backend",
        action="store_true",
        help="Enable the backend-neutral text profiler (sets AION_PROFILE=summary)",
    )
    p.add_argument(
        "--trace-backend",
        action="store_true",
        help="Enable verbose backend tracing (sets AION_TRACE=1)",
    )
    p.add_argument(
        "--global-cache-capacity",
        type=int,
        default=2048,
        help="Capacity G for global KV caches (layers 4/9/14) [default: 2048]",
    )
    p.add_argument(
        "--growable",
        action="store_true",
        help=(
            "Allocate KV caches small and let the runtime grow them on demand "
            "(device-resident growth) up to their capacity, instead of pre-allocating."
        ),
    )
    p.add_argument(
        "--initial-cache-capacity",
        type=int,
        default=8,
        help="Initial KV cache capacity in tokens when --growable [default: 8]",
    )
    args = p.parse_args()

    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be > 0")
    if args.global_cache_capacity <= 0:
        raise SystemExit("--global-cache-capacity must be > 0")

    model_path = args.model.resolve()
    if not model_path.is_file():
        raise SystemExit(f"model not found: {model_path}")

    if args.trace_backend:
        os.environ["AION_TRACE"] = "1"
    if args.profile_backend:
        os.environ["AION_PROFILE"] = "summary"

    tokenizer = None
    tok_src: Optional[str] = args.tokenizer
    if tok_src is None and args.tokenizer_json is not None:
        tok_src = args.tokenizer_json
    if tok_src is not None and args.image is None and args.audio is None:
        tokenizer = _load_tokenizer(tok_src)

    t0_all_ns = time.perf_counter_ns()

    t0_tok_ns = time.perf_counter_ns()
    mm_bindings: dict[str, np.ndarray] = {}
    if args.image is not None or args.audio is not None:
        prompt_ids, tokenizer, mm_bindings = _prepare_multimodal(args)
    elif args.chat:
        if tokenizer is None:
            raise SystemExit("--chat requires a tokenizer")
        # One builder for the chat format, shared with the multimodal path.
        prompt_ids = _gemma_chat_ids(
            tokenizer, [], args.prompt, system=args.system, think=args.think
        )
    else:
        if (args.add_bos or args.add_eos) and tokenizer is None:
            raise SystemExit("--add-bos/--add-eos require a tokenizer")
        prompt_ids = _encode_prompt(tokenizer, args.prompt, add_bos=bool(args.add_bos), add_eos=bool(args.add_eos))
    if args.image is None and args.audio is None:
        seq = len(prompt_ids)
        mm_bindings = {
            "has_image": np.array([0], np.int32),
            "has_audio": np.array([0], np.int32),
            "text_embedding_mask": np.ones((1, seq, 1), np.float32),
            "pixel_values": np.zeros((1, 9, 768), np.float32),
            "image_position_x": np.zeros((1, 9), np.int32),
            "image_position_y": np.zeros((1, 9), np.int32),
            "vision_lengths": np.array([1], np.int32),
            "image_pool_map": np.zeros((1, 1, 9), np.float32),
            "image_embed_map": np.zeros((1, seq, 1), np.float32),
            "input_features": np.zeros((1, 4, 128), np.float32),
            "input_features_mask": np.ones((1, 4, 1), np.float32),
            "audio_subsample1_mask": np.ones((1, 2, 1, 1), np.float32),
            "audio_attention_mask": np.zeros((1, 1), np.float32),
            "audio_embed_map": np.zeros((1, seq, 1), np.float32),
        }
    tok_ns = time.perf_counter_ns() - t0_tok_ns
    s0 = len(prompt_ids)

    sampler = Sampler(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )

    # Stop tokens. The multimodal paths always build a chat turn (the model has to
    # be told where the image/audio block sits), so they need the turn's stops too —
    # gating these on `--chat` alone let `<eos>` through mid-reply.
    templated: bool = bool(args.chat or args.image is not None or args.audio is not None)
    stop_on_eos: bool = bool(args.stop_on_eos) if args.stop_on_eos is not None else bool(templated and tokenizer is not None)
    stop_on_eot: bool = (
        bool(args.stop_on_end_of_turn) if args.stop_on_end_of_turn is not None else bool(templated and tokenizer is not None)
    )
    stop_tokens: set[int] = set()
    if tokenizer is not None:
        if stop_on_eos:
            stop_tokens.add(int(tokenizer.EOS))
        if stop_on_eot:
            stop_tokens.add(int(tokenizer.END_OF_TURN))

    if (s0 + args.max_new_tokens) > args.global_cache_capacity:
        raise SystemExit(
            "global cache capacity is too small: "
            f"need >= prompt_len({s0}) + max_new_tokens({args.max_new_tokens}) = {s0 + args.max_new_tokens}, "
            f"got {args.global_cache_capacity}."
        )

    # The model declares input roles, so loading takes over ALL state plumbing:
    # KV caches (io-aliased f16, capacity `G` for global layers) are allocated
    # and zeroed by the runtime — sized by `cache_capacity`, optionally starting
    # small and growing on demand with `growable` — and the index/position
    # inputs (cache_write_index / cache_visible_end / positions) are fed from
    # the runtime-tracked position, which advances by the bound `tokens`
    # sequence length each run. The only input this example binds is `tokens`.
    model, device_str = load_model_with_device(
        str(model_path),
        args,
        thread_count=args.thread_count,
        cache_capacity=args.global_cache_capacity,
        growable=bool(args.growable),
        initial_cache_capacity=args.initial_cache_capacity,
    )
    print(f"device: {device_str}")
    with model:
        ctx = model.context

        # Prefill tokens [1, S]; decode-step tensor [1, 1] rewritten in place.
        tokens_prefill = aion.Tensor([prompt_ids], ctx=ctx, dtype=aion.int32)
        tokens_step = aion.Tensor([[0]], ctx=ctx, dtype=aion.int32)
        modality_tensors = {
            name: aion.Tensor(value, ctx=ctx, dtype=aion.int32 if value.dtype == np.int32 else aion.float32)
            for name, value in mm_bindings.items()
        }

        try:
            # ---- Prefill ----
            model.bind_input("tokens", tokens_prefill)
            for name, tensor in modality_tensors.items():
                model.bind_input(name, tensor)

            print(f"prefill: tokens={s0}")
            t0_prefill_run_ns = time.perf_counter_ns()
            model.run()
            prefill_run_ns = time.perf_counter_ns() - t0_prefill_run_ns

            # No cache bookkeeping: the shared aliased slots already hold prefill's
            # K/V and carry it into the decode loop automatically.

            # Greedy next token from the last position, picked on the execution
            # device. Outputs are mirrored to the host lazily, so leaving the
            # [1, S, vocab] logits unread is what skips copying them back.
            t0_argmax_ns = time.perf_counter_ns()
            next_id = sampler(model)
            argmax_ns = time.perf_counter_ns() - t0_argmax_ns

            # Generated token ids (excluding stop tokens for readability).
            generated: List[int] = []
            stopped_by: Optional[int] = None
            if tokenizer is not None and (next_id in stop_tokens):
                stopped_by = next_id
            else:
                generated.append(next_id)

            print(f"next_token_id={next_id}")
            if tokenizer is not None:
                print(f"next_token_text={_decode_ids(tokenizer, [next_id])!r}")
            if stopped_by is not None:
                print(f"stop_token_id={stopped_by}")

            # Decode-only view for readability (prompt + first token).
            if tokenizer is not None:
                try:
                    print(f"prompt_text={args.prompt!r}")
                    print(f"prompt_tokens={s0}  decoded_prompt={_decode_ids(tokenizer, prompt_ids)!r}")
                except Exception:
                    # Keep example resilient if tokenizer backend can't roundtrip perfectly.
                    pass

            # ---- Decode loop (optional) ----
            # Switch bindings once: decode steps use a [1,1] tokens tensor. The
            # runtime keeps feeding indices/positions from its tracked position.
            model.bind_input("tokens", tokens_step)
            modality_tensors["has_image"].copy_from(np.array([0], np.int32))
            modality_tensors["has_audio"].copy_from(np.array([0], np.int32))
            decode_bindings = {
                "text_embedding_mask": np.ones((1, 1, 1), np.float32),
                "image_embed_map": np.zeros((1, 1, mm_bindings["image_embed_map"].shape[2]), np.float32),
                "audio_embed_map": np.zeros((1, 1, mm_bindings["audio_embed_map"].shape[2]), np.float32),
            }
            for name, value in decode_bindings.items():
                previous = modality_tensors[name]
                replacement = aion.Tensor(value, ctx=ctx, dtype=aion.float32)
                modality_tensors[name] = replacement
                model.bind_input(name, replacement)
                previous.close()

            decode_total_run_ns: int = 0
            decode_total_argmax_ns: int = 0
            decode_steps_executed: int = 0
            for _ in range(args.max_new_tokens - 1):
                if stopped_by is not None:
                    break

                tokens_step.copy_from(next_id)

                t0_step_run_ns = time.perf_counter_ns()
                model.run()
                decode_steps_executed += 1
                decode_total_run_ns += time.perf_counter_ns() - t0_step_run_ns

                t0_step_argmax_ns = time.perf_counter_ns()
                next_id = sampler(model)
                decode_total_argmax_ns += time.perf_counter_ns() - t0_step_argmax_ns

                if tokenizer is not None and (next_id in stop_tokens):
                    stopped_by = next_id
                    break

                generated.append(next_id)

            if tokenizer is not None:
                print(f"generated_text={_decode_ids(tokenizer, generated)!r}")
            else:
                print(f"generated_ids={generated}")

            if args.timing or args.timing_json:
                t_all_ns = time.perf_counter_ns() - t0_all_ns
                prefill_new: int = 1
                decode_new: int = int(decode_steps_executed)
                total_new: int = int(prefill_new + decode_new)
                kept_new: int = int(len(generated))

                decode_tps = 0.0
                if decode_steps_executed > 0 and decode_total_run_ns > 0:
                    decode_tps = decode_steps_executed / (float(decode_total_run_ns) / 1.0e9)

                # `model.run()` submits without waiting, so on a device backend the
                # token is not ready when it returns — reading it is where the GPU
                # wait lands. Tokens/s therefore has to count run + read, or it
                # reports the rate at which steps are ISSUED, not tokens produced.
                decode_e2e_ns = decode_total_run_ns + decode_total_argmax_ns
                decode_e2e_tps = 0.0
                if decode_steps_executed > 0 and decode_e2e_ns > 0:
                    decode_e2e_tps = decode_steps_executed / (float(decode_e2e_ns) / 1.0e9)

                if args.timing:
                    print("\n--- timing ---")
                    print(f"tokenize:          {_ns_to_ms(tok_ns):9.3f} ms  (prompt_tokens={s0})")
                    print(f"prefill.run:       {_ns_to_ms(prefill_run_ns):9.3f} ms")
                    print(f"prefill.argmax:    {_ns_to_ms(argmax_ns):9.3f} ms")
                    print(f"generated kept:    {kept_new:9d} tok")
                    if stopped_by is not None:
                        print(f"stopped by token:  {stopped_by:9d}")

                    if decode_steps_executed > 0:
                        print(f"decode.run total:  {_ns_to_ms(decode_total_run_ns):9.3f} ms  ({decode_steps_executed} steps)")
                        print(f"decode.argmax tot: {_ns_to_ms(decode_total_argmax_ns):9.3f} ms")
                        print(f"decode throughput: {decode_tps:9.2f} tok/s (model.run only)")
                        print(f"decode e2e:        {decode_e2e_tps:9.2f} tok/s (run + token read)")
                    print(f"total wall:        {_ns_to_ms(t_all_ns):9.3f} ms")

                if args.timing_json:
                    timing_record = {
                        "schema": "aion.model_timing.v1",
                        "model": str(model_path),
                        "device": device_str,
                        "prompt_tokens": s0,
                        "generated_tokens": kept_new,
                        "decode_steps": decode_steps_executed,
                        "tokenize_ns": tok_ns,
                        "prefill_run_ns": prefill_run_ns,
                        "prefill_argmax_ns": argmax_ns,
                        "decode_run_ns": decode_total_run_ns,
                        "decode_argmax_ns": decode_total_argmax_ns,
                        "decode_tokens_per_second": decode_tps,
                        "decode_e2e_tokens_per_second": decode_e2e_tps,
                        "total_wall_ns": t_all_ns,
                        "instrumentation": [
                            name
                            for name in ("AION_PROFILE",)
                            if os.environ.get(name)
                        ],
                    }
                    print(f"[aion-perf-json] {json.dumps(timing_record, separators=(',', ':'))}")
        finally:
            # Close tensors (best-effort).
            for t in [tokens_prefill, tokens_step, *modality_tensors.values()]:
                try:
                    t.close()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
