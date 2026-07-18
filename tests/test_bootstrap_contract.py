import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BootstrapContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = (ROOT / "bootstrap.sh").read_text()

    def test_is_strict_and_repository_relative(self):
        self.assertIn("set -euo pipefail", self.script)
        self.assertIn("BASH_SOURCE[0]", self.script)
        self.assertIn('export PYTHONPATH="$ROOT/src"', self.script)

    def test_builds_an_ignored_local_runtime(self):
        self.assertIn('RUNTIME="${BENCHMARK_RUNTIME_ROOT:-$ROOT/.runtime}"', self.script)
        self.assertIn('VENV="$RUNTIME/venv"', self.script)
        self.assertIn('LOG="$RUNTIME/logs/bootstrap.log"', self.script)
        self.assertIn("UV_CACHE_DIR", self.script)
        self.assertIn(".runtime/", (ROOT / ".gitignore").read_text().splitlines())

    def test_installs_the_immutable_sglang_revision(self):
        lock = json.loads((ROOT / "requirements/benchmark-lock.json").read_text())
        source = lock["source"]["sglang"]
        self.assertEqual(
            source["commit"], "19593359971ebc3582a74f000bf285488d993362"
        )
        self.assertEqual(source["url"], "https://github.com/sgl-project/sglang.git")
        self.assertEqual(source["subdirectory"], "python")
        self.assertIn("source['commit']", self.script)
        self.assertIn("source['subdirectory']", self.script)

    def test_declares_direct_engine_dependency_in_project_and_lock(self):
        lock = json.loads((ROOT / "requirements/benchmark-lock.json").read_text())
        self.assertEqual(
            lock["packages"]["pyyaml"],
            {"version": "6.0.3", "module": "yaml"},
        )
        self.assertIn('"PyYAML==6.0.3"', (ROOT / "pyproject.toml").read_text())

    def test_clears_stale_marker_before_installing(self):
        marker_clear = 'rm -f "$MARKER"'
        editable_install = 'pip install --no-deps -e "$ROOT"'
        self.assertIn(marker_clear, self.script)
        self.assertLess(self.script.index(marker_clear), self.script.index(editable_install))

    def test_never_reuses_reference_environment_or_external_source(self):
        self.assertNotIn(".venv-sglang0515", self.script)
        self.assertNotIn("/home/claude-lsh", self.script)
        self.assertNotIn("/mnt/", self.script)

    def test_validates_before_marking_install_complete(self):
        validation = '"$VENV/bin/python" -m benchmark_environment --check'
        self.assertIn(validation, self.script)
        marker_write = "printf '%s\\n' \"$LOCK_HASH\" > \"$MARKER\""
        self.assertIn(marker_write, self.script)
        self.assertLess(self.script.index(validation), self.script.index(marker_write))

    def test_installs_editable_package_after_locked_dependencies(self):
        dependency_install = 'echo "Installing locked runtime dependencies"'
        editable_install = (
            'VIRTUAL_ENV="$VENV" "$UV" pip install --no-deps -e "$ROOT"'
        )
        validation = '"$VENV/bin/python" -m benchmark_environment --check'
        self.assertIn(editable_install, self.script)
        self.assertLess(self.script.index(dependency_install), self.script.index(editable_install))
        self.assertLess(self.script.index(editable_install), self.script.rindex(validation))

    def test_import_is_required_on_fast_and_install_paths(self):
        import_check = '"$VENV/bin/python" -c \'import benchmark_engine\''
        self.assertGreaterEqual(self.script.count(import_check), 2)
        marker_write = "printf '%s\\n' \"$LOCK_HASH\" > \"$MARKER\""
        self.assertLess(self.script.rindex(import_check), self.script.rindex(marker_write))


if __name__ == "__main__":
    unittest.main()
