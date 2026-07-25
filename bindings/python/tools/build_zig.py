# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import platform
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ZigBuildArtifacts:
    """Result of building Aion into an install `prefix`.

    The Python bindings always *static-link* Aion into the cffi extension module
    (`_aion_cffi.pyd` / `.so`), so there is no separate shared library to ship.
    `link_lib_path` is the static archive fed to the extension's linker.

    GPU builds (`AION_PY_GPU=1`) add nothing here: wgpu-native is neither linked
    nor bundled — the archive resolves it at runtime via dlopen (see
    `src/aion/backend/gpu/wgpu.zig`). The wgpu runtime ships in the separate
    `aion-wgpu` package, which `aion[gpu]` pulls in; the loader finds it via the
    `AION_WGPU_LIB` path that `aion`'s package `__init__` sets from it.
    """

    repo_root: Path
    prefix: Path
    include_dir: Path
    header_path: Path
    # Static archive linked into the extension module:
    #   Windows -> aion.lib   ·   Linux/macOS -> libaion.a
    link_lib_path: Path


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for p in (cur, *cur.parents):
        if (p / "build.zig").is_file():
            return p
    raise RuntimeError(f"Could not locate repo root (build.zig) from: {start}")


def _static_lib_name(system: str) -> str:
    # `zig build` names a static library "aion" as `aion.lib` on Windows and
    # `libaion.a` everywhere else. This is the single source of truth for the
    # archive filename — no probing, no globbing.
    return "aion.lib" if system == "windows" else "libaion.a"


def _run(cmd: list[str], cwd: Path) -> None:
    # Keep output visible for build logs (pip/cibuildwheel), but do not spam by default.
    subprocess.check_call(cmd, cwd=str(cwd))


def _truthy(v: str) -> bool:
    return v.strip() in ("1", "true", "True", "yes", "Yes", "on", "On")


def _resolve_gpu() -> bool:
    """Whether to build the wgpu GPU backend into the extension.

    Precedence: the `AION_PY_GPU` env var (if set, wins — CI can force `0`),
    else the persistent `[tool.aion] gpu` switch in `pyproject.toml`, else off.
    The pyproject switch matters under `uv`, whose build cache is env-blind: it
    keys on the source tree (which includes pyproject), so flipping it there
    actually triggers a rebuild, whereas an env var alone would be ignored.
    """

    env = os.environ.get("AION_PY_GPU")
    if env is not None and env.strip() != "":
        return _truthy(env)

    try:
        import tomllib  # py3.11+

        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        return bool(data.get("tool", {}).get("aion", {}).get("gpu", False))
    except Exception:
        return False


def build_aion(prefix: Path | None = None) -> ZigBuildArtifacts:
    """Build and install the Aion static library + header into a dedicated prefix.

    Aion is always built as a static archive and linked into the extension module
    (see `ZigBuildArtifacts`). The build is parameterized only by deterministic
    environment knobs:

      - AION_ZIG_EXE: zig executable (default: 'zig')
      - AION_PY_OPTIMIZE: Zig optimize mode (default: 'ReleaseFast')
      - AION_PY_CPU: Zig CPU model/features via -Dcpu (default: 'native')
      - AION_PY_TARGET: Zig target triple via -Dtarget
            (default: x86_64-windows-msvc on Windows so the ABI matches MSVC/cffi;
             native elsewhere)
      - AION_PY_PIC: build with -Dpic=true (default: on for non-Windows, needed to
            link the static archive into a shared extension module)
      - AION_PY_MULTIVERSION: '1' to build the portable multi-ISA kernels with
            runtime CPUID dispatch (x86_64_v3 floor + v3/v3_vnni/v4 tiers). This is
            the portable-wheel build; off by default so local builds stay native.
      - AION_PY_GPU: '1' to build with GPU support (`-Dgpu`), linking wgpu-native;
            '0' to force CPU-only. Overrides the persistent `[tool.aion] gpu`
            switch in pyproject.toml (the recommended way to enable GPU under uv,
            whose build cache is env-blind). Off by default; GPU device calls
            return AION_UNSUPPORTED on a CPU-only build.
      - AION_PY_BUILD_PREFIX: install prefix directory (default: a fresh temp dir)
    """

    # File layout: bindings/python/tools/build_zig.py -> repo root is the dir
    # containing build.zig, above bindings/python/...
    repo_root = _find_repo_root(Path(__file__).resolve().parent)
    system = platform.system().lower()
    is_windows = system.startswith("win")

    zig_exe = os.environ.get("AION_ZIG_EXE", "zig")
    optimize = os.environ.get("AION_PY_OPTIMIZE", "ReleaseFast")
    cpu_opt = os.environ.get("AION_PY_CPU", "native")

    # On Windows, cffi/setuptools links with MSVC, so default to the MSVC ABI
    # target unless the caller overrides it.
    target_opt = os.environ.get("AION_PY_TARGET")
    if target_opt is None and is_windows:
        target_opt = "x86_64-windows-msvc"

    # PIC defaults on for non-Windows (required to link a static archive into a
    # shared extension module); meaningless on Windows.
    pic_env = os.environ.get("AION_PY_PIC")
    if pic_env is None:
        pic = not is_windows
    else:
        pic = pic_env.strip() not in ("", "0", "false", "False", "no", "No")

    multiversion = os.environ.get("AION_PY_MULTIVERSION", "").strip() in (
        "1", "true", "True", "yes", "Yes",
    )

    enable_gpu = _resolve_gpu()

    if prefix is None:
        env_prefix = os.environ.get("AION_PY_BUILD_PREFIX")
        prefix = Path(env_prefix) if env_prefix else Path(tempfile.mkdtemp(prefix="aion-zig-"))

    # Absolute prefix so Zig and our post-build checks agree even when callers
    # pass a relative build_temp.
    prefix = prefix.resolve()
    prefix.mkdir(parents=True, exist_ok=True)

    cmd = [
        zig_exe,
        "build",
        "install",
        f"-Doptimize={optimize}",
        "-Dlinkage=static",
        f"-Dmultiversion={'true' if multiversion else 'false'}",
        "--prefix",
        str(prefix),
    ]
    if target_opt:
        cmd.append(f"-Dtarget={target_opt}")
    if cpu_opt:
        cmd.append(f"-Dcpu={cpu_opt}")
    if pic:
        cmd.append("-Dpic=true")
    # Pass -Dgpu explicitly: the Zig default is ON, but the Python package is
    # CPU-only unless the caller opts in (AION_PY_GPU=1 / `pip install .[gpu]`).
    cmd.append(f"-Dgpu={'true' if enable_gpu else 'false'}")

    # The one artifact the build must produce, computed up front (no searching).
    include_dir = prefix / "include"
    header_path = include_dir / "aion.h"
    link_lib_path = prefix / "lib" / _static_lib_name(system)

    # Reuse an existing install only when it was built with the exact same config.
    config_stamp_path = prefix / ".aion-build-config.json"
    build_config = {
        "optimize": optimize,
        "target": target_opt or "",
        "cpu": cpu_opt,
        "pic": pic,
        "multiversion": multiversion,
        "gpu": enable_gpu,
        "platform": system,
    }

    def prefix_is_ready() -> bool:
        if not (header_path.is_file() and link_lib_path.is_file()):
            return False
        try:
            stamp = json.loads(config_stamp_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return stamp == build_config

    if not prefix_is_ready():
        _run(cmd, cwd=repo_root)
        config_stamp_path.write_text(
            json.dumps(build_config, sort_keys=True, indent=2), encoding="utf-8"
        )

    if not header_path.is_file():
        raise RuntimeError(f"Expected installed header not found: {header_path}")
    if not link_lib_path.is_file():
        raise RuntimeError(f"Expected installed static library not found: {link_lib_path}")

    # GPU builds add no link inputs and no bundled runtime: the archive references
    # zero wgpu symbols (resolved at runtime via dlopen). `enable_gpu` only decided
    # whether the seam was compiled in, via `-Dgpu` above.
    return ZigBuildArtifacts(
        repo_root=repo_root,
        prefix=prefix,
        include_dir=include_dir,
        header_path=header_path,
        link_lib_path=link_lib_path,
    )
