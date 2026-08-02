# aion (Python)

Python bindings for the Aion runtime, implemented on top of Aion’s stable C ABI (`include/aion.h`).

## Install

```bash
pip install aion         # CPU (prebuilt wheels; no Zig/toolchain needed)
pip install aion[gpu]    # + optional GPU runtime (see "GPU support")
```

Wheels are published for CPython 3.12/3.13 on Windows (x64), Linux (x86_64 &
aarch64), and macOS (arm64 & x86_64). Other platforms build from source (below).

## Build requirements (from source)

Building from source requires:
- **Zig** available on `PATH` (used to build the Aion static library)
- A working C toolchain suitable for building Python extensions
- Python build deps: `cffi`, `setuptools`, `wheel`

Wheels are intended to be provided for common Windows/Linux targets so end users do not need Zig.

## Quick start (developer, repo checkout)

This subproject is set up to work well with **uv**.

From `bindings/python/`:

- Create/sync the environment and build the extension: `uv sync`
  - The default `dev` group includes NumPy, tokenizers, torch, pytest, and the
    `aion-wgpu` runtime on supported platforms.

- For a leaner env with just the test deps: `uv sync --group test`
- For a runtime-only env (no dev/test tooling): `uv sync --no-default-groups`

- Run tests:
  - `uv run pytest`

> **Changed the Zig core? Just use `uv run`.** The uv cache key includes
> `../../src/**/*.zig`, the public headers, and the Zig build files, so the next
> command rebuilds and relinks the editable package before it runs:
>
> ```bash
> uv run pytest
> uv run python examples/silero_vad_simple.py --device gpu
> ```
>
> Two lower-level staleness traps are also covered:
>
> - `tools/build_zig.py` always invokes `zig build install` (never skips on a
>   config stamp, which cannot see source edits). Zig's own content cache makes a
>   no-op rebuild ~1s.
> - setuptools decides whether to relink the extension by comparing it against the
>   cffi-generated C file, which is byte-identical on every build — so a changed
>   `aion.lib` alone would *not* trigger a relink. `setup.py` therefore treats the
>   archive as the staleness input: it builds Aion first, forces a relink when the
>   archive is newer than the extension, and fails the build if the extension is
>   somehow still older afterwards.
>
> To force a rebuild independently of the cache key, use
> `uv sync --reinstall-package aion`.

Build tuning env vars (for native extension build):

- `AION_PY_OPTIMIZE` (default: `ReleaseFast`)
- `AION_PY_CPU` (default: `native`)
- `AION_PY_TARGET` (Windows default: `x86_64-windows-msvc`; set only when you need an explicit triple)

On Windows, the default target stays MSVC-compatible so setuptools/cffi can link cleanly.

### Simple streaming example (Silero-style)

A minimal end-to-end example is provided at:

- `bindings/python/examples/silero_vad_simple.py`

It loads `models/silero/silero_vad_16k.aion` (I'll host this), runs chunked inference, and reports:

- `chunks/s`
- `us/chunk`

For best throughput, the example reuses pre-bound tensors (`bind_input` + `run`) rather than
creating temporary tensors via `run_numpy(...)` each chunk.

Run it from `bindings/python/`:

- `uv run python examples/silero_vad_simple.py`

### NumPy interop (optional)

NumPy is an **optional runtime** dependency (it ships in the default `dev` group, so
`uv sync` already installs it). For a runtime-only install that still wants NumPy
interop, use the `numpy` extra: `uv pip install "aion[numpy]"`.

With NumPy present you can move data between Aion tensors and NumPy efficiently:

- `t = aion.Tensor([1.0, 2.0, 3.0])`
- `t2 = aion.Tensor([[1.0, 2.0], [3.0, 4.0]])`
- `t3 = aion.Tensor(numpy_array, ctx=ctx)`
- `numpy_array = t.numpy()`
- `t.copy_from(numpy_array)`

Without NumPy, use `tensor.tolist()` for nested Python data or
`tensor.item()` for a one-element tensor.

Tensor helpers for common initialization/update patterns:

- `t = aion.Tensor.zeros((1, 128))`
- `t.zero()` (in-place)
- `t.fill(0.5)` (in-place)

Default-context thread count (used by `Tensor(...)` when no explicit `Context` is passed)
can be configured via environment variable before first use:

- `AION_DEFAULT_THREAD_COUNT` (preferred)
- `AION_THREAD_COUNT` (fallback)

If you change env vars mid-process, call `aion.reset_default_context()` so the next implicit
tensor/context creation picks up the new value.

`aion.load_model(path)` also uses the default context when `thread_count` is omitted.
If you pass `thread_count`, it creates a dedicated context for that model.

`LoadedModel.run_numpy(...)` is a convenience wrapper that accepts Tensor *or* array-like inputs and
returns NumPy arrays for outputs.

### Building models

There is one graph representation. `aion.Builder` holds it; its ops and every
`aion.nn` layer take and return `aion.TensorRef`, a handle on a value in that graph.
`TensorRef` carries the operators and fluent forms, so composing reads the way you
would expect, and it knows its own builder — layers need no ambient state. This
is the same split the Zig API makes between `TensorRef` and `api.Tensor`.

`aion.Tensor` is the **data** handle: what you write inputs into, read outputs
out of, and hand a layer as a weight. It is not a graph node.

```python
import aion, numpy as np
from aion import nn

ctx = aion.Context()
w = aion.tensor(np.random.randn(4, 3).astype("f4"))    # data

with aion.Builder(ctx) as b:
    x = b.input((1, 4)).rename("x")                    # TensorRef
    y = nn.Linear(w)(x).relu().rename("y")             # TensorRef
    model = b.compile([y])

out = model.run_numpy({"x": np.ones((1, 4), "f4")})["y"]
```

For a reusable module, `aion.compile` traces `forward` once against inputs built
from the specs — the same `TensorRef`s, so a module composes exactly the ops a raw
`Builder` would:

```python
model = aion.compile(MyModule(), aion.spec((None, 4)))   # None = dynamic axis
aion.export(MyModule(), aion.spec((None, 4)), "mlp.aion")
```

`dtype` accepts Aion constants (`aion.float32`, `aion.float16`, `aion.int8`,
`aion.int32`, and `aion.q8_0`) plus NumPy dtype classes/objects. Dtype strings
and raw enum ordinals are intentionally unsupported. `aion.tensor(np.array(...,
np.int32))` infers `aion.int32`.

**Looking at a value while you build.** `compile` hands the model a *copy* of the
graph, so authoring survives it — you can compile twice, compile different outputs,
and keep building afterwards. That also means a value can be computed just to look
at it:

```python
with aion.Builder(ctx) as b:
    y = (b.param(a) @ b.param(w)).relu()
    print(y)              # TensorRef(shape=(1, 2), dtype=..., value=[[1.0, 0.0]])
    print(y.numpy())      # same result, memoized
    z = y + y             # still building on y
```

Compilation is pruned to the value's cone, so this costs what the value needs
rather than what the whole graph holds. Evaluating something that depends on a
public input raises and names it: the runtime would otherwise zero-fill the input
and hand back a plausible-looking answer.

**Exploring without naming a Builder.** `Tensor.ref()` binds data into a graph and
hands back the handle — using the Context's scratch builder unless you pass one:

```python
x = aion.tensor(np_x, ctx=ctx)
w = aion.tensor(np_w, ctx=ctx)

y = (x.ref() @ w.ref()).relu()      # no `with aion.Builder(...)`
print(y)
print(nn.Linear(w)(y).numpy())      # nn composes into the same graph
```

Safe to share because compile prunes: values you explored and did not ask for never
reach a compiled model. The scratch graph does grow across a long session — a fresh
`Context` clears it. Author models on an explicit `Builder` so each owns its graph.

### GPU 

With `aion[gpu]` installed and a discrete GPU present:

```python
import aion

# Register one GPU as gpu:0 (or Context(gpus=[aion.GpuOptions(power="high")])).
ctx = aion.Context.gpu()

# Standalone tensors: move onto the GPU (move semantics — the host copy is freed).
t = aion.Tensor([1.0, 2.0, 3.0], ctx=ctx, device="gpu")
assert t.device() == "gpu:0"
t.to("cpu")                     # migrate back before host read/write
print(t.tolist())

# Models: load onto the GPU. Keep feeding CPU input tensors — the runtime
# migrates them on run() and flushes outputs back to host for reading.
m = aion.LoadedModel.load(ctx, "model.aion", device="gpu")
```

Notes:
- `tensor.to("gpu")` frees the host copy; host conversion/mutation then raises
  until you `to("cpu")`. Do **not** bind a device-resident tensor as a model
  input — bind CPU tensors and let the model's runtime migrate them.
- `aion.load_model(path, device="gpu")` creates a dedicated single-GPU context.
  The GPU index selects the **physical adapter** (integrated vs discrete):
  `device="gpu"` picks by power preference (default discrete), `device="gpu:1"`
  (or `adapter_index=1`) picks adapter 1. For multiple GPUs on one context, build
  `Context(gpus=[...])` and use `LoadedModel.load(ctx, path, device="gpu:i")`
  (registry index).
- Example scripts accept `--device {cpu,gpu,auto}` plus `--gpu-index N` /
  `--gpu-power {low,high}` (default `cpu`). `low`=integrated, `high`=discrete.

### Windows note

On Windows, we aim to **static-link** the Aion runtime into the compiled extension module.
This avoids shipping a separate `aion.dll` and simplifies imports.

## Notes

- Preferred lifecycle style is context managers (`with`) for deterministic cleanup.
- If child tensors/models are still live, `Context.close()` now closes them automatically before destroying the native context.
- GC/finalizers are available as a fallback, but `with` is still the recommended pattern for predictable release timing.
