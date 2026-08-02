# SPDX-License-Identifier: Apache-2.0
"""Embed text with harrier-oss-v1-270m and rank documents against a query.

The `.aion` encoder is stateless and batched. The driver right-pads token ids and
supplies an attention mask; positions and valid lengths are derived in-graph.
The returned embeddings are already L2-normalized, so cosine similarity is a plain dot product.

Asymmetric by design: the *query* gets a task instruction prefix (which one is
recorded in the package's `prompt.*` metadata), documents get none.

Run:  python examples/harrier_embed.py --query "..." --document "..." --document "..."
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import numpy as np

import aion

from _common import add_device_args, load_model_with_device, read_aion_metadata


def default_model_path() -> Path:
    # bindings/python/examples -> repo root
    return Path(__file__).resolve().parents[3] / "models" / "harrier" / "harrier_oss_v1_q8.aion"


def default_tokenizer_path() -> Path:
    return Path(__file__).resolve().parents[3] / "models" / "harrier" / "tokenizer.json"


DEMO_QUERY = "What is the capital of France?"
DEMO_DOCUMENTS = [
    "Paris is the capital and most populous city of France.",
    "Berlin is the capital of Germany.",
    "The mitochondrion is the powerhouse of the cell.",
    "France is a country in Western Europe whose largest city is Paris.",
]


def encode(
    model: aion.LoadedModel,
    rows: Sequence[Sequence[int]],
    *,
    pad_id: int,
) -> np.ndarray:
    """A right-padded batch -> unit-norm embeddings `[batch, 640]`."""
    if not rows or any(not row for row in rows):
        raise ValueError("encode requires a non-empty batch of non-empty token rows")
    width = max(len(row) for row in rows)
    tokens = np.full((len(rows), width), pad_id, dtype=np.int32)
    attention_mask = np.zeros((len(rows), width), dtype=np.int32)
    for batch_index, row in enumerate(rows):
        length = len(row)
        tokens[batch_index, :length] = row
        attention_mask[batch_index, :length] = 1

    out = model.run_numpy({
        "tokens": tokens,
        "attention_mask": attention_mask,
    }, outputs=["embedding"])
    return out["embedding"]


def main() -> None:
    p = argparse.ArgumentParser(description="harrier-oss-v1-270m text embeddings via aion")
    p.add_argument("--model", type=Path, default=default_model_path())
    p.add_argument("--tokenizer", type=Path, default=default_tokenizer_path(),
                   help="Path to tokenizer.json (adds <bos>/<eos> itself)")
    p.add_argument("--query", type=str, default=DEMO_QUERY)
    p.add_argument("--document", action="append", default=None,
                   help="A document to rank (repeatable). Defaults to a demo set.")
    p.add_argument("--task", type=str, default="web_search_query",
                   help="Which prompt.* prefix from the package metadata to prepend "
                        "to the query ('none' to skip)")
    p.add_argument("--thread-count", type=int, default=8)
    add_device_args(p)
    args = p.parse_args()

    model_path = args.model.resolve()
    if not model_path.is_file():
        raise SystemExit(f"model not found: {model_path}")
    if not args.tokenizer.is_file():
        raise SystemExit(f"tokenizer not found: {args.tokenizer}")

    try:
        from tokenizers import Tokenizer
    except ImportError as e:
        raise SystemExit("this example needs `tokenizers`: uv pip install tokenizers") from e

    meta = read_aion_metadata(str(model_path))
    prefix = ""
    if args.task != "none":
        key = f"prompt.{args.task}"
        if key not in meta:
            available = sorted(k[len("prompt."):] for k in meta if k.startswith("prompt."))
            raise SystemExit(f"unknown task {args.task!r}; package offers: {available}")
        prefix = meta[key]

    tok = Tokenizer.from_file(str(args.tokenizer))
    documents: List[str] = args.document if args.document else list(DEMO_DOCUMENTS)
    pad_id = int(meta.get("pad_token_id", "0"))

    model, device_str = load_model_with_device(
        str(model_path), args, thread_count=args.thread_count
    )
    print(f"device: {device_str}  pooling: {meta.get('pooling')}  "
          f"normalize: {meta.get('normalize')}  quant: {meta.get('quant')}")
    print(f"query prefix: {prefix!r}\n")

    with model:
        encoded = tok.encode_batch([prefix + args.query, *documents])
        vectors = encode(model, [item.ids for item in encoded], pad_id=pad_id)
        q_vec, d_vecs = vectors[0], vectors[1:]

    # Both sides are already unit-norm, so the dot product IS the cosine.
    scores = [float(q_vec @ d) for d in d_vecs]
    print(f"query: {args.query!r}")
    for score, doc in sorted(zip(scores, documents), reverse=True):
        print(f"  {score:+.4f}  {doc}")


if __name__ == "__main__":
    main()
