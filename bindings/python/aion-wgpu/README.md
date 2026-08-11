# aion-wgpu

The prebuilt [wgpu-native](https://github.com/gfx-rs/wgpu-native) runtime for the
optional GPU backend of [`aion-engine`](https://pypi.org/project/aion-engine/).
You don't install it directly:

```bash
pip install aion-engine[gpu]
```

`aion-engine`'s wheels already contain the GPU backend code but link **no** wgpu:
the library is loaded at runtime via `dlopen`. This package is the delivery
vehicle — one platform-specific shared library (`wgpu_native.dll` /
`libwgpu_native.so` / `libwgpu_native.dylib`) plus a `library_path()` helper that
`aion` calls on import to point its loader at it.

Keeping the multi-MiB runtime in its own package keeps `pip install aion-engine`
small and CPU-only by default. One wheel is published per platform the runtime
wheels target: Windows x64, Linux x86_64/aarch64, macOS arm64/x86_64.

The version tracks the wgpu-native release it wraps (pinned in the repo's
`build.zig.zon`, the single source of truth for that URL and hash).

Licensing: the packaging code here is Apache-2.0; the bundled `wgpu_native`
library ships under the wgpu project's own terms (Apache-2.0 OR MIT).
