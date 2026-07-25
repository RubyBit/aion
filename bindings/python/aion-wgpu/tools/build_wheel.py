# SPDX-License-Identifier: Apache-2.0
"""Build the `aion-wgpu` wheel for the **native** platform.

Run once per platform on its own runner (see .github/workflows/publish.yml). The
wheel carries only the prebuilt wgpu-native library, which `setup.py` stages via
`zig build wgpu-runtime` (from the repo's pinned build.zig.zon). On Linux the
wheel is retagged to a manylinux policy by auditwheel (which verifies the
library's glibc requirement); Windows/macOS wheels need no repair.

    python tools/build_wheel.py --output-dir dist
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]  # bindings/python/aion-wgpu


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="dist")
    args = ap.parse_args()
    out_dir = (Path.cwd() / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    staging = PKG_ROOT / "build" / "wheel"
    if staging.exists():
        shutil.rmtree(staging)

    subprocess.check_call(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(staging)],
        cwd=str(PKG_ROOT),
    )
    wheels = list(staging.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one wheel, got {wheels}")
    wheel = wheels[0]

    if sys.platform.startswith("linux"):
        # Let auditwheel pick the minimal manylinux policy the library satisfies.
        subprocess.check_call(
            [sys.executable, "-m", "auditwheel", "repair", str(wheel), "-w", str(out_dir)]
        )
    else:
        shutil.copy2(wheel, out_dir / wheel.name)

    print(f"\naion-wgpu wheel(s) in {out_dir}:")
    for w in sorted(out_dir.glob("aion_wgpu-*.whl")):
        print("  " + w.name)


if __name__ == "__main__":
    main()
