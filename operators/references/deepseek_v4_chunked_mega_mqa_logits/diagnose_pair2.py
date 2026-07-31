"""Check whether stock DeepGEMM can legally group two shared-cache queries."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

from benchmark_engine.correctness.inputs import make_generator_context


ROOT = Path("/home/gjy/data/agent4kernel/mega_mqa_logits/llm_flops")
REFERENCE = ROOT / "operators/references/deepseek_v4_chunked_mega_mqa_logits"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    import deep_gemm

    spec_module = load("pair2_spec", REFERENCE / "spec.py")
    case = next(case for case in spec_module.SPEC.cases() if case.symbols["m"] == 1024)
    torch.cuda.set_device(0)
    context = make_generator_context(case.seed, cuda_devices=("cuda:0",))
    inputs = spec_module.SPEC.clone_inputs(spec_module.SPEC.make_inputs(case, context))
    args = inputs.args
    q, weights, cache, lens, table = args[6:11]
    m = q.shape[0]

    q_pair2 = q.reshape(m // 2, 2, 64, 128)
    lens_pair2 = lens.reshape(m // 2, 2)
    table_pair2 = table[: m // 2]
    schedule_pair2 = deep_gemm.get_paged_mqa_logits_metadata(
        lens_pair2, 64, deep_gemm.get_num_sms()
    )
    for _ in range(3):
        out = deep_gemm.fp8_paged_mqa_logits(
            q_pair2,
            cache,
            weights,
            lens_pair2,
            table_pair2,
            schedule_pair2,
            16384,
            False,
        )
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    out = deep_gemm.fp8_paged_mqa_logits(
        q_pair2,
        cache,
        weights,
        lens_pair2,
        table_pair2,
        schedule_pair2,
        16384,
        False,
    )
    end.record()
    end.synchronize()

    sample_indices = spec_module._sample_valid_output_indices(
        torch, m, out.device, 256
    )
    sampled = out.reshape(-1)[sample_indices].float()
    oracle = inputs.observed_state["semantic_oracle"].float()
    max_abs = float((sampled - oracle).abs().max().item())
    bound = float(1e-6 + 1e-5 * oracle.abs().max().item())
    print(
        json.dumps(
            {
                "output_shape": list(out.shape),
                "schedule_shape": list(schedule_pair2.shape),
                "kernel_ms": float(start.elapsed_time(end)),
                "max_abs": max_abs,
                "bound": bound,
                "passed": max_abs <= bound,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
