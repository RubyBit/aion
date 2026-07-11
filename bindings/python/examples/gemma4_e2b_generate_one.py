# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
import urllib.request
from typing import Iterable, List, Optional, Sequence

import numpy as np

import aion

from _common import add_device_args, load_model_with_device


def default_model_path() -> Path:
    # bindings/python/examples -> repo root
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "models" / "gemma" / "gemma4_e2b_q8.aion"


def default_tokenizer_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "models" / "gemma" / "tokenizer.json"


def default_sp_model_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "models" / "gemma" / "tokenizer.model"


def is_global_layer(layer: int) -> bool:
    return (layer % 5) == 4


def _download(url: str, dst: Path) -> None: # TODO: Test download path
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "aion-gemma-example"})
    with urllib.request.urlopen(req, timeout=60.0) as resp:
        dst.write_bytes(resp.read())

class _Tokenizer:
    """Tiny wrapper to normalize multiple tokenizer backends."""

    # Gemma4 special token IDs (from the JAX reference / upstream tokenizer wrapper).
    PAD: int = 0
    EOS: int = 1
    BOS: int = 2
    UNK: int = 3
    MASK: int = 4
    START_OF_TURN: int = 105
    END_OF_TURN: int = 106

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


def _load_tokenizer(path_or_url: str, *, cache_path: Path) -> _Tokenizer:
    """Load a tokenizer from a file path, directory, or URL.

    Supported inputs:
      - tokenizer.json (Hugging Face `tokenizers`)
      - tokenizer.model (SentencePiece)
      - a directory containing either of the above
    """

    def load_json(p: Path) -> _Tokenizer:
        try:
            from tokenizers import Tokenizer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise SystemExit(
                "Hugging Face tokenizers is required for tokenizer.json; install via: uv pip install tokenizers"
            ) from e
        return _Tokenizer("hf_tokenizers_json", Tokenizer.from_file(str(p)))

    def load_spm(p: Path) -> _Tokenizer:
        try:
            import sentencepiece as spm  # type: ignore
        except Exception as e:  # pragma: no cover
            raise SystemExit(
                "sentencepiece is required for tokenizer.model; install via: uv pip install sentencepiece"
            ) from e
        proc = spm.SentencePieceProcessor()
        if not proc.Load(str(p)):
            raise SystemExit(f"failed to load sentencepiece model: {p}")
        return _Tokenizer("sentencepiece", proc)

    # URL: download to cache_path and then load based on extension.
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        # Preserve extension when the cache_path is a directory.
        if cache_path.suffix == "":
            cache_path = cache_path / Path(path_or_url).name
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not cache_path.is_file():
            print(f"downloading tokenizer -> {cache_path}")
            _download(path_or_url, cache_path)
        p = cache_path
    else:
        p = Path(path_or_url).expanduser().resolve()

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


def main() -> None:
    p = argparse.ArgumentParser(description="Gemma 4 E2B: prefill + generate 1+ token(s) using aion")
    p.add_argument("--model", type=Path, default=default_model_path(), help="Path to gemma4_e2b_q8.aion")
    p.add_argument(
        "--tokenizer",
        type=str,
        default=str(default_tokenizer_path()),
        help=(
            "Path/URL to tokenizer.json or tokenizer.model, or a directory containing them. "
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
    p.add_argument("--thread-count", type=int, default=1, help="Context thread count (default: 1)")
    add_device_args(p)
    p.add_argument("--timing", action="store_true", help="Print per-stage timing and throughput")
    p.add_argument(
        "--profile-backend",
        action="store_true",
        help="Enable backend per-op-kind profiling (sets AION_PROFILE_STEPS=1)",
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
        os.environ["AION_PROFILE_STEPS"] = "1"

    tokenizer = None
    tok_src: Optional[str] = args.tokenizer
    if tok_src is None and args.tokenizer_json is not None:
        tok_src = args.tokenizer_json
    if tok_src is not None:
        # If the user provided a URL, cache into models/gemma/{tokenizer.json|tokenizer.model}.
        cache_path = default_tokenizer_path() if tok_src.endswith(".json") else default_sp_model_path()
        tokenizer = _load_tokenizer(tok_src, cache_path=cache_path)

    t0_all_ns = time.perf_counter_ns()

    t0_tok_ns = time.perf_counter_ns()
    if args.chat:
        if tokenizer is None:
            raise SystemExit("--chat requires a tokenizer")
        # Follow upstream Gemma4 chat formatting (see gemma4_pt_claude README).
        prompt_ids = [tokenizer.BOS, tokenizer.START_OF_TURN]
        prompt_ids += tokenizer.encode(f"user\n{args.prompt}")
        prompt_ids += [tokenizer.END_OF_TURN]
        prompt_ids += tokenizer.encode("\n")
        prompt_ids += [tokenizer.START_OF_TURN]
        prompt_ids += tokenizer.encode("model\n")
    else:
        if (args.add_bos or args.add_eos) and tokenizer is None:
            raise SystemExit("--add-bos/--add-eos require a tokenizer")
        prompt_ids = _encode_prompt(tokenizer, args.prompt, add_bos=bool(args.add_bos), add_eos=bool(args.add_eos))
    tok_ns = time.perf_counter_ns() - t0_tok_ns
    s0 = len(prompt_ids)

    # Stop tokens (useful for chat templates).
    stop_on_eos: bool = bool(args.stop_on_eos) if args.stop_on_eos is not None else bool(args.chat and tokenizer is not None)
    stop_on_eot: bool = (
        bool(args.stop_on_end_of_turn) if args.stop_on_end_of_turn is not None else bool(args.chat and tokenizer is not None)
    )
    stop_tokens: set[int] = set()
    if tokenizer is not None:
        if stop_on_eos:
            stop_tokens.add(int(tokenizer.EOS))
        if stop_on_eot:
            stop_tokens.add(int(tokenizer.END_OF_TURN))

    if s0 > 512:
        raise SystemExit(
            f"prompt is too long for the local sliding window (512): tokens={s0}. "
            "Try a shorter prompt."
        )
    if (s0 + args.max_new_tokens) > args.global_cache_capacity:
        raise SystemExit(
            "global cache capacity is too small: "
            f"need >= prompt_len({s0}) + max_new_tokens({args.max_new_tokens}) = {s0 + args.max_new_tokens}, "
            f"got {args.global_cache_capacity}."
        )

    # Model interface expects caches for layers 0..14 (15 source layers).
    source_layers = list(range(15))

    model, device_str = load_model_with_device(str(model_path), args, thread_count=args.thread_count)
    print(f"device: {device_str}")
    with model:
        ctx = model.context

        # The KV caches are io-aliased to their `next_*_cache.layerN` outputs, and
        # the runtime backs each aliased input with a model-level state slot shared
        # across cache entries. So even though prefill (seq=S) and decode (seq=1)
        # compile to two different entries, the appended K/V carries across them
        # automatically: we bind the caches ONCE and never swap outputs into inputs.
        # We still allocate them explicitly (rather than letting auto-init zero them)
        # because the global layers' time axis is a free capacity symbol `G` the
        # caller chooses — no bound input lets the runtime infer it.
        # With --growable we allocate a cache at a small initial capacity and let the
        # runtime grow its device-resident slot on demand up to `t_cap`; without it we
        # pre-allocate the full `t_cap` (the original behavior).
        #
        # Only GLOBAL layers have a free (symbolic) capacity axis `G`; local layers are
        # fixed at 512 in the model, so they must be bound at exactly 512 and cannot
        # start small. So grow-on-demand applies to the global caches (the large ones).
        def is_growable_layer(layer: int) -> bool:
            return bool(args.growable) and is_global_layer(layer)

        k_cache = {}
        v_cache = {}
        for layer in source_layers:
            head_dim = 512 if is_global_layer(layer) else 256
            t_cap = args.global_cache_capacity if is_global_layer(layer) else 512
            alloc_cap = min(int(args.initial_cache_capacity), int(t_cap)) if is_growable_layer(layer) else int(t_cap)
            shape = (1, 1, int(alloc_cap), int(head_dim))

            k = aion.Tensor.empty(ctx, shape, dtype=aion.AionDType.AION_DTYPE_F16)
            v = aion.Tensor.empty(ctx, shape, dtype=aion.AionDType.AION_DTYPE_F16)
            k.zero()
            v.zero()
            k_cache[layer] = k
            v_cache[layer] = v

        # Scalars (rank-1, shape [B]).
        cache_write_index = aion.Tensor([0], ctx=ctx, dtype=aion.AionDType.AION_DTYPE_I32)
        cache_visible_end = aion.Tensor([s0], ctx=ctx, dtype=aion.AionDType.AION_DTYPE_I32)

        # Prefill inputs.
        tokens_prefill = aion.Tensor([prompt_ids], ctx=ctx, dtype=aion.AionDType.AION_DTYPE_I32)
        positions_prefill = aion.Tensor([list(range(s0))], ctx=ctx, dtype=aion.AionDType.AION_DTYPE_I32)

        # Decode-step tensors (shape [1,1]) reused for each additional token.
        tokens_step = aion.Tensor([[0]], ctx=ctx, dtype=aion.AionDType.AION_DTYPE_I32)
        positions_step = aion.Tensor([[0]], ctx=ctx, dtype=aion.AionDType.AION_DTYPE_I32)

        # Bind the KV caches once. The shared aliased state slots carry the appended
        # K/V across every run (prefill and each decode step) with no rebinding.
        for layer in source_layers:
            model.bind_input(f"k_cache.layer{layer}", k_cache[layer])
            model.bind_input(f"v_cache.layer{layer}", v_cache[layer])
            if is_growable_layer(layer):
                t_cap = args.global_cache_capacity
                init_cap = min(int(args.initial_cache_capacity), int(t_cap))
                for nm in (f"k_cache.layer{layer}", f"v_cache.layer{layer}"):
                    model.set_state_input_growable(nm, initial_capacity=init_cap, max_capacity=int(t_cap))

        try:
            # ---- Prefill ----
            model.bind_input("tokens", tokens_prefill)
            model.bind_input("positions", positions_prefill)
            model.bind_input("cache_write_index", cache_write_index)
            model.bind_input("cache_visible_end", cache_visible_end)

            print(f"prefill: tokens={s0}")
            t0_prefill_run_ns = time.perf_counter_ns()
            model.run()
            prefill_run_ns = time.perf_counter_ns() - t0_prefill_run_ns

            # No cache bookkeeping: the shared aliased slots already hold prefill's
            # K/V and carry it into the decode loop automatically.

            t0_logits_ns = time.perf_counter_ns()
            logits_t = model.output_tensor("logits")
            try:
                logits = logits_t.numpy()  # [1, S, V]
            finally:
                logits_t.close()
            logits_copy_ns = time.perf_counter_ns() - t0_logits_ns

            # Greedy next token from last position.
            t0_argmax_ns = time.perf_counter_ns()
            next_id = int(np.argmax(logits[0, -1]))
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
            # Switch bindings once: decode step uses [1,1] tokens/positions.
            cur_pos = s0
            model.bind_input("tokens", tokens_step)
            model.bind_input("positions", positions_step)
            model.bind_input("cache_write_index", cache_write_index)
            model.bind_input("cache_visible_end", cache_visible_end)

            decode_total_run_ns: int = 0
            decode_total_logits_ns: int = 0
            decode_total_argmax_ns: int = 0
            decode_steps_executed: int = 0
            for _ in range(args.max_new_tokens - 1):
                if stopped_by is not None:
                    break

                # Update scalar indices.
                cache_write_index.write_i32([cur_pos])
                cache_visible_end.write_i32([cur_pos + 1])

                tokens_step.write_i32([next_id])
                positions_step.write_i32([cur_pos])

                t0_step_run_ns = time.perf_counter_ns()
                model.run()
                decode_steps_executed += 1
                decode_total_run_ns += time.perf_counter_ns() - t0_step_run_ns

                t0_step_logits_ns = time.perf_counter_ns()
                logits_t = model.output_tensor("logits")
                try:
                    logits = logits_t.numpy()  # [1, 1, V]
                finally:
                    logits_t.close()
                decode_total_logits_ns += time.perf_counter_ns() - t0_step_logits_ns

                t0_step_argmax_ns = time.perf_counter_ns()
                next_id = int(np.argmax(logits[0, 0]))
                decode_total_argmax_ns += time.perf_counter_ns() - t0_step_argmax_ns

                if tokenizer is not None and (next_id in stop_tokens):
                    stopped_by = next_id
                    break

                generated.append(next_id)
                cur_pos += 1

            if tokenizer is not None:
                print(f"generated_text={_decode_ids(tokenizer, generated)!r}")
            else:
                print(f"generated_ids={generated}")

            if args.timing:
                t_all_ns = time.perf_counter_ns() - t0_all_ns
                prefill_new: int = 1
                decode_new: int = int(decode_steps_executed)
                total_new: int = int(prefill_new + decode_new)
                kept_new: int = int(len(generated))

                print("\n--- timing ---")
                print(f"tokenize:          {_ns_to_ms(tok_ns):9.3f} ms  (prompt_tokens={s0})")
                print(f"prefill.run:       {_ns_to_ms(prefill_run_ns):9.3f} ms")
                print(f"prefill.logits:    {_ns_to_ms(logits_copy_ns):9.3f} ms  (numpy copy)")
                print(f"prefill.argmax:    {_ns_to_ms(argmax_ns):9.3f} ms")
                print(f"generated kept:    {kept_new:9d} tok")
                if stopped_by is not None:
                    print(f"stopped by token:  {stopped_by:9d}")

                if decode_steps_executed > 0:
                    print(f"decode.run total:  {_ns_to_ms(decode_total_run_ns):9.3f} ms  ({decode_steps_executed} steps)")
                    print(f"decode.logits tot: {_ns_to_ms(decode_total_logits_ns):9.3f} ms")
                    print(f"decode.argmax tot: {_ns_to_ms(decode_total_argmax_ns):9.3f} ms")
                    dt_s = float(decode_total_run_ns) / 1.0e9
                    if dt_s > 0:
                        print(f"decode throughput: {decode_steps_executed / dt_s:9.2f} tok/s (model.run only)")
                print(f"total wall:        {_ns_to_ms(t_all_ns):9.3f} ms")
        finally:
            # Close tensors (best-effort).
            for t in [tokens_prefill, positions_prefill, tokens_step, positions_step, cache_write_index, cache_visible_end]:
                try:
                    t.close()
                except Exception:
                    pass
            for layer in source_layers:
                try:
                    k_cache[layer].close()
                except Exception:
                    pass
                try:
                    v_cache[layer].close()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
