import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BenchLauncherContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = (ROOT / "bench.sh").read_text()

    def test_launcher_is_repository_relative_and_uses_module_entrypoint(self):
        self.assertIn("BASH_SOURCE[0]", self.script)
        self.assertIn('RUNTIME_ROOT="${BENCHMARK_RUNTIME_ROOT:-$ROOT/.runtime}"', self.script)
        self.assertIn('VENV="$RUNTIME_ROOT/venv"', self.script)
        self.assertIn('"$VENV/bin/python"', self.script)
        self.assertIn('-m benchmark_engine', self.script)
        self.assertIn(
            'export PYTHONPATH="$ROOT/src:$SGLANG_ROOT/python:$AITER_ROOT"',
            self.script,
        )
        self.assertLess(self.script.index('cd "$ROOT"'), self.script.index('exec "$VENV/bin/python"'))

    def test_launcher_sets_all_repository_local_caches(self):
        expected = {
            "UV_CACHE_DIR": "$CACHE_ROOT/uv",
            "TORCH_EXTENSIONS_DIR": "$CACHE_ROOT/torch_extensions",
            "FLASHINFER_WORKSPACE_BASE": "$CACHE_ROOT/flashinfer",
            "XDG_CACHE_HOME": "$CACHE_ROOT/xdg",
        }
        for variable, relative_path in expected.items():
            with self.subTest(variable=variable):
                self.assertIn(f"export {variable}=", self.script)
                self.assertIn(relative_path, self.script)
        self.assertIn('mkdir -p "$UV_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"', self.script)

    @unittest.skipUnless(shutil.which("bash"), "bash is required")
    def test_missing_runtime_points_to_bootstrap(self):
        with tempfile.TemporaryDirectory() as temporary:
            launcher = Path(temporary) / "bench.sh"
            launcher.write_text(self.script)
            launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)
            completed = subprocess.run(
                [str(launcher), "--help"],
                text=True,
                capture_output=True,
                check=False,
                env={**os.environ, "PYTHONPATH": "must-be-cleared"},
            )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("./bootstrap.sh", completed.stderr)

    @unittest.skipUnless(shutil.which("bash"), "bash is required")
    def test_repository_launcher_exposes_help(self):
        completed = subprocess.run(
            [str(ROOT / "bench.sh"), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("usage: bench", completed.stdout)


if __name__ == "__main__":
    unittest.main()
