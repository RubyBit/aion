"""Run with Python unittest; requires setuptools and wheel, but not Zig."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile


class WheelLayoutTest(unittest.TestCase):
    def test_native_library_is_not_installed_to_purelib(self):
        source = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            project.mkdir()
            for name in ("setup.py", "pyproject.toml", "README.md"):
                shutil.copy2(source / name, project / name)
            package = project / "src" / "aion_wgpu"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("")
            # Only archive layout is under test; no GPU or real ELF is needed.
            (package / "libwgpu_native.so").write_bytes(b"layout fixture")
            result = subprocess.run(
                [sys.executable, "setup.py", "bdist_wheel", "--plat-name", "linux_x86_64"],
                cwd=project,
                env={**os.environ, "AION_WGPU_STAGED": "1"},
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            wheels = list((project / "dist").glob("*.whl"))
            self.assertEqual(len(wheels), 1)
            self.assertTrue(wheels[0].name.endswith("-py3-none-linux_x86_64.whl"))
            with zipfile.ZipFile(wheels[0]) as wheel:
                names = wheel.namelist()
                self.assertIn("aion_wgpu/libwgpu_native.so", names)
                self.assertFalse(any(".data/purelib/" in name for name in names))
                metadata = next(name for name in names if name.endswith(".dist-info/WHEEL"))
                self.assertIn("Root-Is-Purelib: false", wheel.read(metadata).decode())


if __name__ == "__main__":
    unittest.main()
