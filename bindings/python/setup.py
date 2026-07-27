# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext

sys.path.insert(0, str(Path(__file__).parent / "tools"))
import build_zig  # noqa: E402


class build_ext(_build_ext):
    """Build Aion, then link it into the cffi extension module.

    Aion is always built as a static archive (`tools/build_zig.py`) and linked
    into `_aion_cffi`; the cffi builder at `src/aion/_ffi/build.py` names that
    archive as the extension's library.

    The one thing this hook exists for: **setuptools cannot see Zig sources.** Its
    staleness check compares the extension against the cffi-generated C file,
    which is identical on every build, so an edit anywhere under `src/aion/`
    leaves the previously linked archive in place and the install silently ships
    stale bytes — indistinguishable from a working rebuild, and the kind of thing
    you only notice after debugging the wrong binary for an hour. So the archive
    itself is the staleness input: build it first, and force a relink when it is
    newer than the extension.

    GPU builds ship no extra runtime here: wgpu-native is loaded at runtime via
    dlopen and delivered by the separate `aion-wgpu` package (see the README).
    """

    def run(self):
        prefix = Path(self.build_temp).resolve() / "aion-zig-prefix"
        prefix.mkdir(parents=True, exist_ok=True)
        os.environ["AION_PY_BUILD_PREFIX"] = str(prefix)

        # Native CPU tuning for local builds unless explicitly overridden.
        os.environ.setdefault("AION_PY_CPU", "native")

        # Build Aion up front. `build_aion` memoizes per prefix, so the call the
        # cffi builder makes during `super().run()` is free.
        archive = build_zig.build_aion(prefix).link_lib_path
        if self._needs_relink(archive):
            self.force = True

        super().run()

        self._verify_fresh(archive)

    def _outputs(self) -> list[Path]:
        return [Path(self.get_ext_fullpath(ext.name)) for ext in (self.extensions or [])]

    def _needs_relink(self, archive: Path) -> bool:
        """True when any extension is missing or older than the Aion archive."""
        if not archive.is_file():
            return True
        archive_mtime = archive.stat().st_mtime
        for out in self._outputs():
            if not out.is_file() or out.stat().st_mtime < archive_mtime:
                return True
        return False

    def _verify_fresh(self, archive: Path) -> None:
        """Fail loudly if an extension is still older than the archive.

        Belt to the braces above: shipping a stale binary is worse than failing to
        build, because it looks like success.
        """
        if not archive.is_file():
            return
        archive_mtime = archive.stat().st_mtime
        for out in self._outputs():
            if out.is_file() and out.stat().st_mtime < archive_mtime:
                raise SystemExit(
                    f"aion build: {out.name} is older than {archive.name} — the "
                    "extension was not relinked against the current Aion build. "
                    "Remove the `build/` directory and rebuild."
                )


# cffi will import the builder referenced below during the build.
setup(
    cffi_modules=["src/aion/_ffi/build.py:ffibuilder"],
    cmdclass={"build_ext": build_ext},
)
