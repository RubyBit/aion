# aion (Python)

Python bindings for the Aion runtime, implemented on top of Aion’s stable C ABI (`include/aion.h`).

## Build requirements (from source)

Building from source requires:
- **Zig** available on `PATH` (used to build the Aion static library)
- A working C toolchain suitable for building Python extensions
- Python build deps: `cffi`, `setuptools`, `wheel`

Wheels are intended to be provided for common Windows/Linux targets so end users do not need Zig.

## Quick start (developer, repo checkout)

This subproject is set up to work well with **uv**.

From `bindings/python/`:

- Create/sync the environment (installs `cffi` + test deps, and builds the extension):
  - Windows (recommended): `uv sync --extra test --no-editable`
  - Linux: `uv sync --extra test`

If you're developing locally (and want NumPy + tests available), use:

- `uv sync --extra dev --no-editable`

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

NumPy is an **optional** dependency. If you want fast array interop and better typing for arrays,
install the extra:

- `uv sync --extra numpy --extra test --no-editable`

Then you can move data between Aion tensors and NumPy efficiently (currently f32 only):

- `Tensor.from_numpy(ctx, array)`
- `tensor.numpy()`
- `tensor.write_from_numpy(array)`

You can also use the simple constructor style (PyTorch-like):

- `t = aion.Tensor([1.0, 2.0, 3.0])`
- `t2 = aion.Tensor([[1.0, 2.0], [3.0, 4.0]])`

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

### Windows note

On Windows, we aim to **static-link** the Aion runtime into the compiled extension module.
This avoids shipping a separate `aion.dll` and simplifies imports.

## Notes

- Preferred lifecycle style is context managers (`with`) for deterministic cleanup.
- If child tensors/models are still live, `Context.close()` now closes them automatically before destroying the native context.
- GC/finalizers are available as a fallback, but `with` is still the recommended pattern for predictable release timing.
