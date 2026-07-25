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

- Create/sync the environment and build the extension. The `dev` dependency
  group (NumPy, tokenizers, torch, pytest, …) is installed by default:
  - Windows (recommended): `uv sync --no-editable`
  - Linux: `uv sync`

- For a leaner env with just the test deps: `uv sync --group test`
- For a runtime-only env (no dev/test tooling): `uv sync --no-default-groups`

- Run tests:
  - `uv run pytest`

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

### Building models (Tensor-first)

`aion.Tensor` — created with `aion.tensor(...)` — is the one high-level value you
use. It is an **immutable lazy value**: operators (`@`, `+`, `-`, `*`, `/`,
`.relu()`, `.softmax()`, …) build a small graph, and `.numpy()` / `.realize()`
compiles and runs it. There are no builder lifecycles to manage — tensors
combine freely, materialize any time, and re-materialize at will.

Compute on concrete data reads like NumPy — no explicit `Builder` needed:

```python
import aion, numpy as np

a = aion.tensor(np.random.randn(2, 4).astype("f4"))   # concrete
w = aion.tensor(np.random.randn(4, 3).astype("f4"))   # concrete
y = (a @ w).relu()                                     # lazy
print(y.shape)                                         # (2, 3) — inferred, no run
print(y.numpy())                                       # compiles + runs
```

`aion.realize([y1, y2])` materializes several results in a single compile + run
(a shared subgraph is emitted once); it is an efficiency tool, never required.

Author a reusable model from symbolic **free inputs**, then compile or export it:

```python
x = aion.tensor(shape=(1, 4), name="x")               # symbolic free input
w = aion.tensor(np.random.randn(4, 3).astype("f4"))
model = aion.compile({"y": (x @ w).relu()}, inputs=[x])
out = model.run_numpy({"x": np.ones((1, 4), "f4")})["y"]

aion.export({"y": x @ w}, "mlp.aion", inputs=[x])      # or serialize
```

`dtype` accepts a single vocabulary everywhere — `"f32"`, `"i32"`, `aion.float32`,
a NumPy dtype, or the `AionDType` enum — and `aion.tensor(np.array(..., np.int32))`
infers `i32`. For the higher-level PyTorch-style module API, see `aion.nn` with
`aion.compile(module, aion.spec(...))`.

**Low-level escape hatch.** `aion.Builder` is the imperative graph-construction
API (control flow, in-place ops, KV-cache roles/aliases) used by the model
converters; its ops return `aion.Value`, a builder-bound handle. The high-level
`Tensor` lowers to it at compile time. Reach for `Builder`/`Value` only when you
need that low-level control; otherwise `Tensor` is the front door.

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
