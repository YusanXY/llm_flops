from __future__ import annotations

import importlib.util
import unittest
from collections import Counter
from pathlib import Path

import yaml

from benchmark_engine.registry import FilesystemRegistry


ROOT = Path(__file__).resolve().parents[1]
HIGH_PRIORITY = (
    "deepseek_v4_aiter_block_fp8_gemm",
    "deepseek_v4_fused_qk_norm_rope_store",
    "deepseek_v4_c4_c128_compressor",
    "deepseek_v4_aiter_c4_paged_mqa_logits",
    "deepseek_v4_tilelang_sparse_attention",
    "deepseek_v4_aiter_fp8_fused_moe",
)


def load_spec(operator_id):
    path = ROOT / "operators" / "references" / operator_id / "spec.py"
    spec = importlib.util.spec_from_file_location(
        f"test_{operator_id}_spec", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.SPEC


class DeepSeekV4FlashMi300xContractTests(unittest.TestCase):
    def test_registry_has_six_references_and_mirrored_control_candidates(self):
        snapshot = FilesystemRegistry(ROOT).discover()
        self.assertFalse(snapshot.issues)
        for operator_id in HIGH_PRIORITY:
            with self.subTest(operator_id=operator_id):
                self.assertIn(operator_id, snapshot.references)
                candidates = snapshot.candidates.get(operator_id, ())
                self.assertEqual(len(candidates), 1)
                self.assertTrue(
                    candidates[0].implementation_id.startswith("reference_control__")
                )

    def test_specs_are_controller_safe_and_cover_both_phases(self):
        expected_counts = {
            "deepseek_v4_aiter_block_fp8_gemm": 4,
            "deepseek_v4_fused_qk_norm_rope_store": 4,
            "deepseek_v4_c4_c128_compressor": 8,
            "deepseek_v4_aiter_c4_paged_mqa_logits": 7,
            "deepseek_v4_tilelang_sparse_attention": 16,
            "deepseek_v4_aiter_fp8_fused_moe": 8,
        }
        for operator_id in HIGH_PRIORITY:
            with self.subTest(operator_id=operator_id):
                spec = load_spec(operator_id)
                cases = spec.cases()
                self.assertEqual(len(cases), expected_counts[operator_id])
                phases = Counter(case.symbols["phase"] for case in cases)
                self.assertEqual(set(phases), {"prefill", "decode"})
                self.assertTrue(any("oracle" in case.tags for case in cases))
                self.assertTrue(any("representative" in case.tags for case in cases))
                for case in cases:
                    cost = spec.cost_model(case)
                    self.assertGreater(cost["flops"], 0)
                    self.assertGreater(cost["estimated_bytes"], 0)
                    self.assertGreater(cost["throughput_units"], 0)
                    self.assertTrue(spec.layout_contract(case)["backend"])

    def test_suites_keep_only_confirmed_high_priority_operators(self):
        for phase in ("prefill", "decode"):
            path = ROOT / "suites" / f"deepseek_v4_{phase}.yaml"
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(tuple(data["operators"]["include"]), HIGH_PRIORITY)
            self.assertEqual(data["cases"]["tags"], [f"deepseek_v4_{phase}"])
            self.assertEqual(data["performance"]["warmup"], 5)
            self.assertEqual(data["performance"]["samples"], 30)
            self.assertEqual(data["performance"]["inner_iterations"], 20)
            self.assertNotIn("timer", data["performance"])

    def test_moe_disables_unsafe_hip_graph_probe(self):
        snapshot = FilesystemRegistry(ROOT).discover()
        moe = snapshot.operator_manifests[
            "deepseek_v4_aiter_fp8_fused_moe"
        ]
        self.assertEqual(moe.performance.timer, "cuda_event")
        self.assertEqual(moe.performance.graph_mode, "disabled")
        sparse = snapshot.operator_manifests[
            "deepseek_v4_tilelang_sparse_attention"
        ]
        self.assertEqual(sparse.performance.timer, "cuda_event")
        self.assertEqual(sparse.performance.graph_mode, "disabled")

    def test_flash_prefill_unified_suite_has_seven_logical_rows(self):
        path = ROOT / "suites" / "deepseek_v4_flash_prefill_unified.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(
            tuple(data["operators"]["include"]),
            (
                "deepseek_v4_q_rmsnorm_wqb",
                "deepseek_v4_wo_a_grouped_bf16",
                "deepseek_v4_aiter_fp8_linear",
                "deepseek_v4_aiter_c4_paged_mqa_logits",
                "deepseek_v4_tilelang_sparse_attention",
                "deepseek_v4_aiter_fp8_fused_moe",
            ),
        )
        self.assertEqual(
            data["cases"]["tags"],
            ["deepseek_v4_flash_prefill_unified"],
        )
        self.assertEqual(data["performance"]["warmup"], 20)
        self.assertEqual(data["performance"]["samples"], 100)
        self.assertNotIn("inner_iterations", data["performance"])

        expected = {
            "deepseek_v4_q_rmsnorm_wqb": 3,
            "deepseek_v4_wo_a_grouped_bf16": 3,
            "deepseek_v4_aiter_fp8_linear": 6,
            "deepseek_v4_aiter_c4_paged_mqa_logits": 3,
            "deepseek_v4_tilelang_sparse_attention": 6,
            "deepseek_v4_aiter_fp8_fused_moe": 3,
        }
        for operator_id, count in expected.items():
            cases = [
                case
                for case in load_spec(operator_id).cases()
                if "deepseek_v4_flash_prefill_unified" in case.tags
            ]
            self.assertEqual(len(cases), count)
            self.assertEqual(
                {case.symbols["model_input"] for case in cases},
                {1024, 2048, 4096},
            )
            self.assertEqual(
                {case.symbols["raw_context"] for case in cases},
                {65536},
            )

    def test_flash_decode_unified_suite_has_one_common_configuration(self):
        path = ROOT / "suites" / "deepseek_v4_flash_decode_unified.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(
            tuple(data["operators"]["include"]),
            (
                "deepseek_v4_q_rmsnorm_wqb",
                "deepseek_v4_wo_a_grouped_bf16",
                "deepseek_v4_aiter_fp8_linear",
                "deepseek_v4_aiter_fp8_fused_moe",
            ),
        )
        self.assertEqual(
            data["cases"]["tags"],
            ["deepseek_v4_flash_decode_unified"],
        )
        self.assertEqual(data["performance"]["warmup"], 20)
        self.assertEqual(data["performance"]["samples"], 100)
        self.assertNotIn("inner_iterations", data["performance"])

        expected = {
            "deepseek_v4_q_rmsnorm_wqb": 2,
            "deepseek_v4_wo_a_grouped_bf16": 2,
            "deepseek_v4_aiter_fp8_linear": 6,
            "deepseek_v4_aiter_fp8_fused_moe": 2,
        }
        for operator_id, count in expected.items():
            cases = [
                case
                for case in load_spec(operator_id).cases()
                if "deepseek_v4_flash_decode_unified" in case.tags
            ]
            self.assertEqual(len(cases), count)
            self.assertEqual(
                {case.symbols["model_input"] for case in cases},
                {16, 32},
            )
            self.assertEqual(
                {case.symbols["raw_context"] for case in cases},
                {65536},
            )

    def test_attention_ratios_and_full_moe_geometry_match_v4_flash(self):
        attention = load_spec("deepseek_v4_tilelang_sparse_attention")
        for phase in ("prefill", "decode"):
            ratios = {
                int(case.symbols["compression_ratio"])
                for case in attention.cases()
                if case.symbols["phase"] == phase
            }
            self.assertEqual(ratios, {0, 4, 128})
        moe = load_spec("deepseek_v4_aiter_fp8_fused_moe")
        representative = next(
            case
            for case in moe.cases()
            if case.symbols["phase"] == "decode"
            and "representative" in case.tags
        )
        self.assertEqual(
            {
                key: representative.symbols[key]
                for key in ("hidden", "intermediate", "experts", "topk")
            },
            {"hidden": 4096, "intermediate": 2048, "experts": 256, "topk": 6},
        )

    def test_launchers_isolate_aiter_merged_config_cache(self):
        expected_roots = {
            "bootstrap.sh": "$CACHE_ROOT/aiter",
            "bench.sh": "$CACHE_ROOT/aiter",
            "run.sh": "$ROOT/.runtime/cache/aiter",
        }
        for filename, expected in expected_roots.items():
            source = (ROOT / filename).read_text(encoding="utf-8")
            self.assertIn("AITER_CONFIG_DIR", source)
            self.assertIn(expected, source)
        bootstrap = (ROOT / "bootstrap.sh").read_text(encoding="utf-8")
        self.assertIn('export AITER_META_DIR="$AITER_ROOT"', bootstrap)
        self.assertIn("SGLANG_OPT_SWIGLU_CLAMP_FUSION=0", bootstrap)

    def test_moe_calls_aiter_fused_kernel_without_sglang_runner(self):
        reference = (
            ROOT
            / "operators"
            / "references"
            / "deepseek_v4_aiter_fp8_fused_moe"
            / "implementation.py"
        ).read_text(encoding="utf-8")
        self.assertIn("from aiter.fused_moe import fused_moe", reference)
        self.assertIn("quant_type=quant_type.per_128x128", reference)
        self.assertIn("gate_mode=gate_mode.INTERLEAVE.value", reference)
        self.assertIn("@lru_cache(maxsize=1)", reference)
        self.assertNotIn("MoeRunner", reference)
        self.assertNotIn("StandardDispatchOutput", reference)
        self.assertNotIn("AiterMoeQuantInfo", reference)
        self.assertNotIn("_triton_fallback", reference)
        self.assertNotIn("LLM_FLOPS_DSV4_USE_AITER_MOE", reference)


if __name__ == "__main__":
    unittest.main()
