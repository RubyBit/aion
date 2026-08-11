# aion-engine

Python bindings for [Aion](https://github.com/RubyBit/aion) — a graph-IR tensor
runtime with a Zig core, CPU and GPU. The distribution is `aion-engine`; the
import name is `aion`.

## Install

```bash
pip install aion-engine          # CPU
pip install aion-engine[gpu]     # + the GPU (wgpu) runtime
```

Prebuilt wheels for CPython 3.12/3.13 on Windows x64, Linux x86_64/aarch64 and
macOS arm64/x86_64 — no Zig or C toolchain needed. The Aion runtime is
**static-linked** into the extension module, so there is no separate
`aion.dll`/`libaion.so` to ship or locate.

The GPU backend is compiled into those same wheels but links no wgpu — it
`dlopen`s wgpu-native at runtime. `[gpu]` adds the companion
[`aion-wgpu`](https://github.com/RubyBit/aion/tree/main/bindings/python/aion-wgpu)
wheel, which ships that library for every platform the runtime wheels target.
Without it, GPU calls raise `aion.AionError` and CPU is unaffected.

NumPy is optional (`[numpy]` extra); without it use `tensor.tolist()` /
`tensor.item()`.

## Run a model

A `.aion` file carries graph, weights and metadata, so this is the whole setup:

```python
import aion, numpy as np

model = aion.load_model("model.aion")              # device="gpu" to run on GPU
out = model.run_numpy({"x": np.zeros((1, 4), "f4")})["y"]
```

In hot loops, bind tensors once (`model.bind_input(name, t)` + `model.run()`)
instead of creating temporaries per step. KV caches and streaming state are
managed by the runtime from the model's input roles — you don't thread them
through Python (`load_model(..., cache_capacity=, growable=)` tunes it).

## Build a model

Two handles, and they don't mix. `aion.Tensor` is **data**: what you write inputs
into, read outputs from, and hand a layer as a weight. `aion.TensorRef` is a
**value in a graph**: what ops and `aion.nn` layers take and return.

```python
import aion, numpy as np
from aion import nn

ctx = aion.Context()
w = aion.tensor(np.random.randn(4, 3).astype("f4"))   # data

with aion.Builder(ctx) as b:
    x = b.input((1, 4)).rename("x")                   # TensorRef
    y = nn.Linear(w)(x).relu().rename("y")
    model = b.compile([y])

out = model.run_numpy({"x": np.ones((1, 4), "f4")})["y"]
```

For a reusable module, `aion.compile(MyModule(), aion.spec((None, 4)))` traces
`forward` once (`None` = dynamic axis) and `aion.export(..., "mlp.aion")` writes
the package. `compile` works on a copy of the graph, so authoring survives it:
you can compile twice, and `print(y)` / `y.numpy()` evaluates just that value's
cone while you keep building.

`dtype` takes Aion constants (`aion.float32`, `float16`, `int8`, `int32`,
`q8_0`, `q4_0`) or NumPy dtypes — not strings.

## GPU

```python
ctx = aion.Context.gpu()                                  # first discrete adapter
model = aion.LoadedModel.load(ctx, "model.aion", device="gpu")
```

Keep feeding CPU input tensors: the model migrates them on `run()` and flushes
outputs back for reading. A standalone `tensor.to("gpu")` is a *move* — the host
copy is freed, so `to("cpu")` before reading it. `device="gpu:1"` (or
`adapter_index=`) picks a physical adapter, `power="low"`/`"high"` picks
integrated/discrete, and `Context(gpus=[...])` registers several.

## Examples

Runnable scripts in
[`bindings/python/examples/`](https://github.com/RubyBit/aion/tree/main/bindings/python/examples),
all taking `--device {cpu,gpu,auto}`:

| | |
|---|---|
| `silero_vad_simple.py` | chunked VAD, reports throughput |
| `harrier_embed.py` | text embeddings, ranks documents against a query |
| `nemotron_asr_streaming.py`, `nemotron_asr_mic.py` | streaming ASR from a wav or the mic |
| `gemma4_e2b_generate_one.py` | LLM token generation |

Models are produced by the converters in
[`scripts/`](https://github.com/RubyBit/aion/tree/main/scripts).

## Developing from a checkout

```bash
uv sync        # environment + native build (dev group: numpy, torch, pytest, aion-wgpu)
uv run pytest
```

`uv run` rebuilds and relinks the extension when the Zig core changes — the uv
cache key covers `src/**/*.zig`, the public headers and the build files.
`uv sync --reinstall-package aion-engine` forces a rebuild.

Build knobs: `AION_PY_OPTIMIZE` (default `ReleaseFast`), `AION_PY_CPU`
(`native`), `AION_PY_TARGET`, `AION_PY_GPU` (overrides `[tool.aion] gpu` in
`pyproject.toml`). Building from source needs Zig on `PATH` plus a C toolchain.

Notes: prefer `with` for deterministic cleanup (`Context.close()` closes live
tensors/models first); `AION_DEFAULT_THREAD_COUNT` sets the default context's
thread count, and `aion.reset_default_context()` re-reads it.
