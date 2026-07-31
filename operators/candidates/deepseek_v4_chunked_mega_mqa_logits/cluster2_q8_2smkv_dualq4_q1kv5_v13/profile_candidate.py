"""Single-call NCU/NSYS harness for the V13 2-SM KV-gather kernel."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch

from benchmark_engine.correctness.inputs import make_generator_context


ROOT = Path("/home/gjy/data/agent4kernel/mega_mqa_logits/llm_flops")
REFERENCE = ROOT / "operators/references/deepseek_v4_chunked_mega_mqa_logits"
CANDIDATE = (
    ROOT
    / "operators/candidates/deepseek_v4_chunked_mega_mqa_logits/cluster2_q8_2smkv_dualq4_q1kv5_v13"
)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True, choices=(1024, 2048, 4096))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--capture", action="store_true")
    args_cli = parser.parse_args()

    spec_module = load("v13_single_request_profile_spec", REFERENCE / "spec.py")
    implementation = load(
        "v13_single_request_profile_impl", CANDIDATE / "implementation.py"
    )
    case = next(
        case for case in spec_module.SPEC.cases()
        if case.symbols["m"] == args_cli.m
    )
    torch.cuda.set_device(0)
    context = make_generator_context(case.seed, cuda_devices=("cuda:0",))
    construct_start = time.perf_counter()
    canonical = spec_module.SPEC.make_inputs(case, context)
    inputs = spec_module.SPEC.clone_inputs(canonical)
    torch.cuda.synchronize()
    construct_seconds = time.perf_counter() - construct_start

    for _ in range(args_cli.warmup):
        implementation.operator(*inputs.args)
    torch.cuda.synchronize()

    cache = inputs.args[8]
    cache_before = cache.clone()
    if args_cli.capture:
        torch.cuda.profiler.start()
        torch.cuda.nvtx.range_push(
            f"cluster2_q8_2smkv_dualq4_q1kv5_v13_m{args_cli.m}_ctx65536"
        )
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = implementation.operator(*inputs.args)
    end.record()
    end.synchronize()
    if args_cli.capture:
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()

    sample_indices = spec_module._sample_valid_output_indices(
        torch, args_cli.m, output.device, 256
    )
    sampled = output.reshape(-1)[sample_indices].float()
    oracle = inputs.observed_state["semantic_oracle"].float()
    error = float((sampled - oracle).abs().max().item())
    bound = float(1e-6 + 1e-5 * oracle.abs().max().item())
    cache_unchanged = bool(torch.equal(cache, cache_before))
    report = {
        "case_id": case.case_id,
        "capture": args_cli.capture,
        "construct_seconds_outside_capture": construct_seconds,
        "operator_cuda_event_ms": float(start.elapsed_time(end)),
        "output_oracle_max_abs": error,
        "output_oracle_bound": bound,
        "physical_cache_unchanged_bitwise": cache_unchanged,
        "profile_range": f"cluster2_q8_2smkv_dualq4_q1kv5_v13_m{args_cli.m}_ctx65536",
    }
    if error > bound or not cache_unchanged:
        raise AssertionError(report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
