"""Capture one warmed-up contract-v2 single-request baseline with NCU."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path

import torch

from benchmark_engine.correctness.inputs import make_generator_context


def load_module(name: str, path: Path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, choices=(1024, 2048, 4096), default=4096)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--metadata-out", type=Path, required=True)
    parser.add_argument(
        "--nvtx-range",
        default="baseline_control__v2_single_request/formal_m4096/operator",
    )
    args = parser.parse_args()

    spec_module = load_module("mega_mqa_spec", args.reference_dir / "spec.py")
    candidate_module = load_module("mega_mqa_candidate", args.candidate)
    case = next(case for case in spec_module.SPEC.cases() if case.symbols["m"] == args.m)

    torch.cuda.set_device(0)
    context = make_generator_context(case.seed, cuda_devices=("cuda:0",))
    canonical = spec_module.SPEC.make_inputs(case, context)
    profile_inputs = spec_module.SPEC.clone_inputs(canonical)

    # Exclude input generation, cloning, lazy CUDA work, and DeepGEMM JIT.
    warmup_output = candidate_module.operator(*profile_inputs.args)
    torch.cuda.synchronize()
    del warmup_output
    gc.collect()

    before_free, before_total = torch.cuda.mem_get_info()
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(args.nvtx_range)
    output = candidate_module.operator(*profile_inputs.args)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    after_free, after_total = torch.cuda.mem_get_info()

    metadata = {
        "case_id": case.case_id,
        "m": args.m,
        "profile_scope": "one complete warmed-up contract-v2 operator call",
        "request_count": 1,
        "query_tokens": args.m,
        "physical_pages": 256,
        "new_c4_k": args.m // 4,
        "nvtx_range": args.nvtx_range,
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "valid_output_probe": [
            float(output[0, (65536 - args.m + 1) // 4 - 1].item()),
            float(output[-1, 16383].item()),
        ],
        "free_before_profile": before_free,
        "free_after_profile": after_free,
        "total_memory": before_total,
        "candidate": str(args.candidate.resolve()),
        "reference_dir": str(args.reference_dir.resolve()),
    }
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_out.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
