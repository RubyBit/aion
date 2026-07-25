# aion-wgpu

Prebuilt [wgpu-native](https://github.com/gfx-rs/wgpu-native) runtime library for
the **optional GPU backend** of [`aion`](https://pypi.org/project/aion/).

You almost never install this directly — it is pulled in by:

```bash
pip install aion[gpu]
```

## What it is

`aion`'s published wheels already contain the GPU backend code, but link **no**
wgpu: the wgpu-native library is loaded at runtime via `dlopen`. This package is
the delivery vehicle for that library — a single platform-specific shared object
(`wgpu_native.dll` / `libwgpu_native.so` / `libwgpu_native.dylib`) and a
`library_path()` helper. On import, `aion` finds it and points its loader at it.

Keeping the (multi-MiB) runtime in a separate package means the default
`pip install aion` stays small and CPU-only, and works on every platform —
including `linux/aarch64`, where no wgpu-native prebuilt exists and `aion[gpu]`
therefore stays CPU-only.

## Versioning

The version tracks the wgpu-native release it wraps (pinned in the repo's
`build.zig.zon`, the single source of truth for the library's URL and hash).

## Licensing

The packaging code here is Apache-2.0. The bundled `wgpu_native` library is
distributed by the wgpu project under its own terms (Apache-2.0 OR MIT).
