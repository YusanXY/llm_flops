from __future__ import annotations

import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from benchmark_engine.cli import main
from benchmark_engine.registry import FilesystemRegistry


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = "example_cpu_add"
EXAMPLE_CANDIDATE = "quickstart__20260716T120000Z__4279e756"


def make_task(root: Path) -> Path:
    task = root / "model__kernel_interface"
    shutil.copytree(ROOT / "operators" / "references" / EXAMPLE, task / "baseline")
    shutil.copytree(
        ROOT / "operators" / "candidates" / EXAMPLE / EXAMPLE_CANDIDATE,
        task / "solution" / EXAMPLE_CANDIDATE,
    )
    (task / "bench").mkdir()
    return task


class KdaTaskRegistryTest(unittest.TestCase):
    def test_task_layout_discovers_baseline_and_versioned_solution_in_place(self):
        with tempfile.TemporaryDirectory() as temporary:
            task = make_task(Path(temporary))
            snapshot = FilesystemRegistry.for_kda_task(task).discover()
            self.assertTrue(snapshot.is_valid, snapshot.issues)
            self.assertEqual(snapshot.references[EXAMPLE].root, task / "baseline")
            candidate = snapshot.candidates[EXAMPLE][0]
            self.assertEqual(candidate.implementation_id, EXAMPLE_CANDIDATE)
            self.assertEqual(candidate.root, task / "solution" / EXAMPLE_CANDIDATE)

    def test_flat_solution_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            task = make_task(Path(temporary))
            (task / "solution" / "implementation.py").write_text(
                "def operator(left, right): return left + right\n", encoding="utf-8"
            )
            issues = FilesystemRegistry.for_kda_task(task).discover().issues
            self.assertIn("kda.flat_solution", {issue.code for issue in issues})

    def test_cli_dry_run_defaults_to_task_bench_and_has_no_artifact_side_effect(self):
        with tempfile.TemporaryDirectory() as temporary:
            task = make_task(Path(temporary))
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = main(
                    (
                        "run",
                        "--task-root",
                        str(task),
                        "--suite",
                        "kda_task",
                        "--candidate",
                        EXAMPLE_CANDIDATE,
                        "--dry-run",
                    ),
                    repository_root=ROOT,
                )
            self.assertEqual(code, 0, stderr.getvalue())
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["jobs"])
            for job in payload["jobs"]:
                output = Path(job["output_dir"])
                self.assertEqual(output.parents[2], task / "bench")
            self.assertEqual(list((task / "bench").iterdir()), [])


class KdaInstallerTest(unittest.TestCase):
    def test_mapping_covers_every_non_tutorial_reference_and_installs_contracts(self):
        mapping_path = ROOT / "integrations" / "kda-pilot" / "operators.json"
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))["operators"]
        references = {
            path.name
            for path in (ROOT / "operators" / "references").iterdir()
            if path.is_dir() and path.name != EXAMPLE
        }
        self.assertEqual(set(mapping), references)
        self.assertEqual(len({slug.casefold() for slug in mapping.values()}), len(mapping))

        operator_id = "deepseek_v4_topk_transform"
        with tempfile.TemporaryDirectory() as temporary:
            llm_root = Path(temporary) / "llm"
            result = subprocess.run(
                (
                    sys.executable,
                    str(ROOT / "tools" / "install_kda_workload.py"),
                    "--llm-root",
                    str(llm_root),
                    "--operator",
                    operator_id,
                    "--refresh-baseline",
                    "--rewrite-task-guides",
                ),
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("verified_tasks\t1", result.stdout)
            task = llm_root / mapping[operator_id]
            self.assertTrue((task / "baseline" / "operator.yaml").is_file())
            self.assertTrue(any(path.is_dir() for path in (task / "solution").iterdir()))
            prompt = (task / "prompt.md").read_text(encoding="utf-8")
            self.assertIn("MUST NOT", prompt)
            self.assertIn("../../../llm_flops/docs/kda-pilot-integration.md", prompt)
            self.assertIn("ranking_eligible", (task / "bench" / "README.md").read_text(encoding="utf-8"))
            self.assertTrue(FilesystemRegistry.for_kda_task(task).discover().is_valid)


if __name__ == "__main__":
    unittest.main()
