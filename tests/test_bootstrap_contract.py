import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BootstrapContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = (ROOT / "bootstrap.sh").read_text()
        cls.lock = json.loads(
            (ROOT / "requirements" / "benchmark-lock.json").read_text()
        )

    def test_is_strict_repository_relative_and_uses_shared_configuration(self):
        self.assertIn("set -euo pipefail", self.script)
        self.assertIn("BASH_SOURCE[0]", self.script)
        self.assertIn('unset PYTHONPATH', self.script)
        self.assertIn('$HOME/.config/agent4kernel/env.sh', self.script)
        self.assertIn('source "$ENV_FILE"', self.script)
        for name in ("GPU_VENV", "SGLANG_ROOT", "AITER_ROOT"):
            self.assertIn(name, self.script)

    def test_reuses_gpu_venv_through_a_safe_local_symlink(self):
        self.assertIn('VENV="$RUNTIME/venv"', self.script)
        self.assertIn('PYTHON="$GPU_VENV/bin/python"', self.script)
        self.assertIn('[[ -e "$VENV" && ! -L "$VENV" ]]', self.script)
        self.assertIn("refusing to replace non-symlink runtime", self.script)
        self.assertIn('ln -sfn "$GPU_VENV" "$VENV"', self.script)

    def test_builds_only_ignored_repository_local_state(self):
        for name in (
            "UV_CACHE_DIR",
            "TORCH_EXTENSIONS_DIR",
            "FLASHINFER_WORKSPACE_BASE",
            "XDG_CACHE_HOME",
            "TRITON_CACHE_DIR",
        ):
            self.assertIn(name, self.script)
        self.assertIn('RUNTIME="${BENCHMARK_RUNTIME_ROOT:-$ROOT/.runtime}"', self.script)
        self.assertIn('CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$RUNTIME/cache}"', self.script)
        self.assertIn('VENV="$RUNTIME/venv"', self.script)
        self.assertIn('LOG="$RUNTIME/logs/bootstrap.log"', self.script)
        self.assertIn(".runtime/", (ROOT / ".gitignore").read_text().splitlines())

    def test_lock_pins_the_live_rocm_and_source_fingerprints(self):
        self.assertEqual(self.lock["schema_version"], 2)
        self.assertEqual(self.lock["python"], "3.11")
        self.assertEqual(
            self.lock["accelerator"],
            {"backend": "rocm", "runtime_prefix": "7.0"},
        )
        self.assertEqual(self.lock["gpu"]["arch_prefix"], "gfx942")
        self.assertEqual(
            self.lock["packages"]["torch"]["version"], "2.10.0+rocm7.0"
        )
        self.assertEqual(
            self.lock["source"]["sglang"],
            {
                "root_env": "SGLANG_ROOT",
                "commit": "85fd90072d1a9f2432842b03588f63b745e524e4",
                "dirty": True,
                "diff_sha256": "be81f7d070f7d17e38196d2c12ea85353b51529ce1af825e598179c86c0b9278",
            },
        )
        self.assertEqual(
            self.lock["source"]["aiter"],
            {
                "root_env": "AITER_ROOT",
                "commit": "6d0304e8ce88bddef4ec875bdd844804fd631089",
                "dirty": True,
                "diff_sha256": "23f8dba6cd44e09543dabc37e3b0aac220167b00bfd0afa874892d799ab97806",
                "submodules": {
                    "3rdparty/composable_kernel": "af7118e342580ecd3f71edce7b1d0ba465012ecf"
                },
            },
        )

    def test_declares_direct_engine_dependency_in_project_and_lock(self):
        self.assertEqual(
            self.lock["packages"]["pyyaml"],
            {"version": "6.0.3", "module": "yaml"},
        )
        project = (ROOT / "pyproject.toml").read_text()
        self.assertIn('requires-python = ">=3.11,<3.12"', project)
        self.assertIn('"PyYAML==6.0.3"', project)

    def test_explicit_python_path_does_not_inherit_the_callers_value(self):
        expected = 'export PYTHONPATH="$ROOT/src:$SGLANG_ROOT/python:$AITER_ROOT"'
        self.assertIn(expected, self.script)
        for entrypoint in ("run.sh", "bench.sh"):
            text = (ROOT / entrypoint).read_text()
            self.assertIn(expected, text)
            self.assertIn('export AITER_META_DIR="$AITER_ROOT"', text)
            self.assertIn("export SGLANG_OPT_SWIGLU_CLAMP_FUSION=0", text)

    def test_does_not_download_or_install_gpu_dependencies(self):
        for forbidden in (
            "git clone",
            "pip install",
            "uv pip",
            "curl ",
            "wget ",
            "/mnt/",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.script)

    def test_validates_environment_and_import_before_writing_marker(self):
        environment_check = '"$PYTHON" -m benchmark_environment --check'
        import_check = '"$PYTHON" -c \'import benchmark_engine\''
        marker_write = "printf '%s %s\\n' \"$LOCK_HASH\" \"$FINGERPRINT\" > \"$MARKER\""
        for fragment in (environment_check, import_check, marker_write):
            self.assertIn(fragment, self.script)
        self.assertLess(self.script.index(environment_check), self.script.index(marker_write))
        self.assertLess(self.script.index(import_check), self.script.index(marker_write))


if __name__ == "__main__":
    unittest.main()
