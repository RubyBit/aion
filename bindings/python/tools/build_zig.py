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
    repo_root: Path
    prefix: Path
    include_dir: Path
    header_path: Path
    # Library used for linking the Python extension:
    # - Windows dynamic: import library (aion.lib)
    # - Windows static: static library (aion.lib)
    # - Linux static: libaion.a
    link_lib_path: Path

    # Optional shared library (for packaging/runtime loading):
    # - Windows dynamic: aion.dll
    # - Linux dynamic: libaion.so
    shared_lib_path: Path | None = None


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for p in (cur, *cur.parents):
        if (p / "build.zig").is_file():
            return p
    raise RuntimeError(f"Could not locate repo root (build.zig) from: {start}")


def _run(cmd: list[str], cwd: Path) -> None:
    # Keep output visible for build logs (pip/cibuildwheel), but do not spam by default.
    subprocess.check_call(cmd, cwd=str(cwd))


def build_aion(prefix: Path | None = None) -> ZigBuildArtifacts:
    """Build and install the Aion static library + header into a dedicated prefix.

    Environment variables:
      - AION_ZIG_EXE: override zig executable (default: 'zig')
      - AION_PY_OPTIMIZE: Zig optimize mode (default: 'ReleaseFast')
            - AION_PY_CPU: Zig CPU model/features (default: 'native')
      - AION_PY_PIC: '1' to build with -Dpic=true (default: enabled on non-Windows)
      - AION_PY_BUILD_PREFIX: optional install prefix directory
            - AION_PY_LINKAGE: 'static' or 'dynamic' (default: dynamic on Windows, static elsewhere)
            - AION_PY_TARGET: Zig target triple override
                (default: x86_64-windows-msvc on Windows, native otherwise)
    """

    # File layout:
    #   bindings/python/tools/build_zig.py
    #   -> repo root is above bindings/python/...
    tools_dir = Path(__file__).resolve().parent
    repo_root = _find_repo_root(tools_dir)

    zig_exe = os.environ.get("AION_ZIG_EXE", "zig")
    optimize = os.environ.get("AION_PY_OPTIMIZE", "ReleaseFast")
    cpu_opt = os.environ.get("AION_PY_CPU", "native")
    is_windows = platform.system().lower().startswith("win")

    # On Windows, cffi/setuptools uses MSVC by default, so keep the library ABI
    # aligned unless explicitly overridden.
    target_opt = os.environ.get("AION_PY_TARGET")
    if target_opt is None and is_windows:
        target_opt = "x86_64-windows-msvc"

    if prefix is None:
        env_prefix = os.environ.get("AION_PY_BUILD_PREFIX")
        if env_prefix:
            prefix = Path(env_prefix)
        else:
            prefix = Path(tempfile.mkdtemp(prefix="aion-zig-"))

    # Use an absolute prefix so Zig and our post-build checks agree even when
    # callers provide a relative `build_temp`.
    prefix = prefix.resolve()
    prefix.mkdir(parents=True, exist_ok=True)

    linkage_env = os.environ.get("AION_PY_LINKAGE")
    if linkage_env is None:
        linkage = "static"
    else:
        linkage = linkage_env.strip().lower()
        if linkage not in ("static", "dynamic"):
            raise ValueError("AION_PY_LINKAGE must be 'static' or 'dynamic'")

    # Default: PIC on non-Windows (needed to link static lib into a shared extension).
    pic_env = os.environ.get("AION_PY_PIC")
    if pic_env is None:
        pic = not is_windows
    else:
        pic = pic_env.strip() not in ("", "0", "false", "False", "no", "No")

    cmd = [
        zig_exe,
        "build",
        "install",
        f"-Doptimize={optimize}",
        f"-Dlinkage={linkage}",
        "--prefix",
        str(prefix),
    ]

    if target_opt:
        cmd.append(f"-Dtarget={target_opt}")

    if cpu_opt:
        cmd.append(f"-Dcpu={cpu_opt}")

    if pic:
        cmd.append("-Dpic=true")

    config_stamp_path = prefix / ".aion-build-config.json"
    build_config = {
        "optimize": optimize,
        "linkage": linkage,
        "target": target_opt or "",
        "cpu": cpu_opt,
        "pic": pic,
        "platform": platform.system().lower(),
    }

    # If the prefix already contains installed artifacts, reuse them.
    include_dir = prefix / "include"
    header_path = include_dir / "aion.h"
    lib_dir = prefix / "lib"
    bin_dir = prefix / "bin"

    def prefix_is_ready() -> bool:
        if not config_stamp_path.is_file():
            return False
        try:
            stamp = json.loads(config_stamp_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if stamp != build_config:
            return False

        if not header_path.is_file():
            return False
        if not lib_dir.is_dir():
            return False
        if is_windows and not (lib_dir / "aion.lib").is_file():
            return False
        if is_windows and linkage == "dynamic" and not (bin_dir / "aion.dll").is_file():
            return False
        if (
            (not is_windows)
            and linkage == "static"
            and not (
                (lib_dir / "libaion.a").is_file() or (lib_dir / "aion.a").is_file()
            )
        ):
            return False
        return True

    if not prefix_is_ready():
        _run(cmd, cwd=repo_root)
        config_stamp_path.write_text(json.dumps(build_config, sort_keys=True, indent=2), encoding="utf-8")

    # Recompute after build.
    include_dir = prefix / "include"
    header_path = include_dir / "aion.h"
    if not header_path.is_file():
        raise RuntimeError(f"Expected installed header not found: {header_path}")

    lib_dir = prefix / "lib"
    if not lib_dir.is_dir():
        raise RuntimeError(f"Expected installed lib dir not found: {lib_dir}")

    link_candidates: list[Path] = []
    shared_candidates: list[Path] = []

    if is_windows:
        # Zig uses `aion.dll` + `aion.lib` for dynamic builds, and `aion.lib` for static.
        link_candidates = [lib_dir / "aion.lib"]
        if linkage == "dynamic":
            # Zig installs the DLL under `bin/` and the import library under `lib/`.
            shared_candidates = [prefix / "bin" / "aion.dll", lib_dir / "aion.dll"]
    else:
        if linkage == "static":
            link_candidates = [lib_dir / "libaion.a", lib_dir / "aion.a"]
        else:
            # Not used today, but keep it supported.
            link_candidates = [lib_dir / "libaion.so", lib_dir / "libaion.dylib"]
            shared_candidates = link_candidates.copy()

    link_lib_path: Path | None = None
    for c in link_candidates:
        if c.is_file():
            link_lib_path = c
            break

    if link_lib_path is None:
        libs = sorted(lib_dir.glob("*.a")) + sorted(lib_dir.glob("*.lib")) + sorted(lib_dir.glob("*.dll")) + sorted(lib_dir.glob("*.so")) + sorted(lib_dir.glob("*.dylib"))
        raise RuntimeError(
            "Aion library not found under prefix. "
            f"Tried: {link_candidates}. Found: {libs}"
        )

    shared_lib_path: Path | None = None
    for c in shared_candidates:
        if c.is_file():
            shared_lib_path = c
            break

    return ZigBuildArtifacts(
        repo_root=repo_root,
        prefix=prefix,
        include_dir=include_dir,
        header_path=header_path,
        link_lib_path=link_lib_path,
        shared_lib_path=shared_lib_path,
    )
