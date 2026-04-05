# Bindings

This folder contains **downstream language bindings** built on top of Aion’s stable C ABI.

Aion’s core runtime and kernels live in Zig under `src/`. The C ABI surface is defined in:
- `include/aion.h`
- `src/aion/c_api.zig`

## Subprojects

- `python/` — Python package that wraps `aion.h` via cffi.

## Design goals

- Keep the C ABI small and stable.
- Provide deterministic errors via status codes and per-context last-error strings.
- Prefer thin, explicit wrappers in downstream languages.
