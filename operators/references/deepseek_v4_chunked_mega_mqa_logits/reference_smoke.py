"""Audit one formal full-KV-reuse case against both independent oracles."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch

from benchmark_engine.correctness.inputs import make_generator_context


ROOT = Path("/home/gjy/data/agent4kernel/mega_mqa_logits/llm_flops")
REFERENCE = ROOT / "operators/references/deepseek_v4_chunked_mega_mqa_logits"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1024, choices=(1024, 2048, 4096))
    parser.add_argument("--samples", type=int, default=10)
    args_cli = parser.parse_args()

    spec_module = load("full_reuse_spec", REFERENCE / "spec.py")
    implementation = load("full_reuse_impl", REFERENCE / "implementation.py")
    case = next(
        case for case in spec_module.SPEC.cases()
        if case.symbols["m"] == args_cli.m
    )
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    context = make_generator_context(case.seed, cuda_devices=("cuda:0",))

    construct_start = time.perf_counter()
    canonical = spec_module.SPEC.make_inputs(case, context)
    torch.cuda.synchronize()
    construct_seconds = time.perf_counter() - construct_start

    clone_start = time.perf_counter()
    inputs = spec_module.SPEC.clone_inputs(canonical)
    torch.cuda.synchronize()
    clone_seconds = time.perf_counter() - clone_start

    (
        history_hidden,
        wkv_gate,
        ape,
        norm_weight,
        rope_cos,
        rope_sin,
        q_fp8,
        weights,
        kv_fused,
        c4_seq_lens,
        page_table,
        _,
        raw_context,
        recompute_start,
        rms_eps,
    ) = inputs.args
    cache_before = kv_fused.clone()

    # Exercise the candidate-side reconstruction helper outside baseline time.
    rebuild_start = time.perf_counter()
    rebuilt_k, rebuilt_scale = implementation.rebuild_full_indexer_cache(
        history_hidden,
        wkv_gate,
        ape,
        norm_weight,
        rope_cos,
        rope_sin,
        rms_eps,
    )
    torch.cuda.synchronize()
    rebuild_seconds = time.perf_counter() - rebuild_start
    page_bytes = 64 * (128 + 4)
    cache_2d = kv_fused.view(kv_fused.shape[0], page_bytes)
    cache_k = (
        cache_2d[:, : 64 * 128]
        .view(torch.float8_e4m3fn)
        .reshape(-1, 128)
    )
    cache_scale = (
        cache_2d[:, 64 * 128 :].view(torch.float32).reshape(-1, 1)
    )
    history_cache_codes_equal = bool(torch.equal(rebuilt_k, cache_k))
    history_cache_scale_max_abs = float(
        (rebuilt_scale - cache_scale).abs().max().item()
    )

    # Warm and time only the reference's full-cache-reuse path.
    for _ in range(3):
        implementation.operator(*inputs.args)
    torch.cuda.synchronize()
    timings = []
    for _ in range(args_cli.samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = implementation.operator(*inputs.args)
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))

    sample_indices = spec_module._sample_valid_output_indices(
        torch, output.shape[0], output.device, 256
    )
    sampled = output.reshape(-1)[sample_indices].float()
    oracle = inputs.observed_state["semantic_oracle"].float()
    output_oracle_max_abs = float((sampled - oracle).abs().max().item())
    output_oracle_bound = float(1e-6 + 1e-5 * oracle.abs().max().item())
    cache_unchanged = bool(torch.equal(kv_fused, cache_before))

    raw_prefix = raw_context - args_cli.m
    prefix_c4 = raw_prefix // 4
    causal_c4_elements = sum(
        (raw_prefix + index + 1) // 4 for index in range(args_cli.m)
    )
    mean_ms = statistics.fmean(timings)
    stdev_ms = statistics.pstdev(timings)
    report = {
        "case_id": case.case_id,
        "symbols": case.symbols,
        "baseline_path": "complete_physical_c4_kv_reuse",
        "candidate_alternative": (
            "recompute any C4 suffix from history_hidden+wkv_gate+"
            "APE+RMSNorm+RoPE+FWHT"
        ),
        "construct_seconds_outside_timing": construct_seconds,
        "clone_seconds_outside_timing": clone_seconds,
        "full_history_rebuild_seconds_outside_timing": rebuild_seconds,
        "operator_ms": {
            "samples": timings,
            "mean": mean_ms,
            "median": statistics.median(timings),
            "p95": percentile(timings, 0.95),
            "cv": 0.0 if mean_ms == 0.0 else stdev_ms / mean_ms,
        },
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "sampled_valid_output_finite": bool(torch.isfinite(sampled).all().item()),
        "sampled_valid_output_absmax": float(sampled.abs().max().item()),
        "output_oracle_max_abs": output_oracle_max_abs,
        "output_oracle_bound": output_oracle_bound,
        "history_rebuild_fp8_codes_equal_full_cache": history_cache_codes_equal,
        "history_rebuild_scale_max_abs": history_cache_scale_max_abs,
        "physical_cache_unchanged_bitwise": cache_unchanged,
        "history_hidden_bytes": int(history_hidden.numel() * history_hidden.element_size()),
        "physical_cache_bytes": int(kv_fused.numel() * kv_fused.element_size()),
        "aggregate_causal_k_read_bytes": int(causal_c4_elements * 132),
        "recompute_eligible_raw_start": recompute_start,
        "all_queries_share_one_request_page_table": bool(
            torch.all(page_table == page_table[:1]).item()
        ),
        "causal_c4_lens_first_four": c4_seq_lens[:4, 0].tolist(),
        "causal_c4_len_last": int(c4_seq_lens[-1, 0].item()),
        "first_extension_k_excluded_for_first_query": bool(
            prefix_c4 >= int(c4_seq_lens[0, 0].item())
        ),
        "first_extension_k_visible_for_fourth_query": bool(
            prefix_c4 < int(c4_seq_lens[3, 0].item())
        ),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }

    if report["output_shape"] != [args_cli.m, 16384]:
        raise AssertionError(report)
    if report["output_dtype"] != "torch.float32":
        raise AssertionError(report)
    if not report["sampled_valid_output_finite"]:
        raise AssertionError(report)
    if output_oracle_max_abs > output_oracle_bound:
        raise AssertionError(report)
    if not history_cache_codes_equal or history_cache_scale_max_abs != 0.0:
        raise AssertionError(report)
    if not cache_unchanged:
        raise AssertionError(report)
    if report["causal_c4_lens_first_four"] != [
        prefix_c4,
        prefix_c4,
        prefix_c4,
        prefix_c4 + 1,
    ]:
        raise AssertionError(report)
    if report["causal_c4_len_last"] != 16384:
        raise AssertionError(report)
    for key in (
        "all_queries_share_one_request_page_table",
        "first_extension_k_excluded_for_first_query",
        "first_extension_k_visible_for_fourth_query",
    ):
        if not report[key]:
            raise AssertionError(report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
