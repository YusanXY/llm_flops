import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class KdaDocumentationContractTest(unittest.TestCase):
    def test_agent_entrypoints_and_ownership_are_documented(self):
        documents = {
            "README.md": ("kda-bench.sh", "baseline/", "solution/<candidate_id>/", "bench/"),
            "docs/kda-pilot-integration.md": (
                "--task-root",
                "BENCHMARK_RUNTIME_ROOT",
                "BENCHMARK_CACHE_ROOT",
                "ranking_eligible=true",
                "must not",
            ),
            "docs/cli.md": ("--task-root", "kda_task", "kda-bench.sh"),
            "docs/result-layout.md": ("solution/<candidate_id>/", "bench/<operator_id>"),
            "docs/csv-schema.md": ("KDA task `bench/`", "hand-edited"),
        }
        for relative, required in documents.items():
            with self.subTest(document=relative):
                text = (ROOT / relative).read_text(encoding="utf-8")
                for fragment in required:
                    self.assertIn(fragment, text)

    def test_mapping_is_strict_and_has_no_tutorial_task(self):
        payload = json.loads(
            (ROOT / "integrations" / "kda-pilot" / "operators.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(set(payload), {"schema_version", "operators"})
        self.assertEqual(payload["schema_version"], 1)
        self.assertNotIn("example_cpu_add", payload["operators"])


if __name__ == "__main__":
    unittest.main()
