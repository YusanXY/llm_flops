"""Probe DeepGEMM's true one-request ABI against the expanded-row baseline."""

from __future__ import annotations

import argparse
import importlib.util
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
    parser.add_argument("--m", type=int, choices=(1024, 2048, 4096), default=1024)
    parser.add_argument("--reference-dir", type=Path, required=True)
    args = parser.parse_args()

    spec = load_module("mega_spec_probe", args.reference_dir / "spec.py")
    impl = load_module("mega_impl_probe", args.reference_dir / "implementation.py")
    case = next(c for c in spec.SPEC.cases() if c.symbols["m"] == args.m)

    torch.cuda.set_device(0)
    canonical = spec.SPEC.make_inputs(
        case, make_generator_context(case.seed, cuda_devices=("cuda:0",))
    )
    expanded_inputs = spec.SPEC.clone_inputs(canonical)
    single_inputs = spec.SPEC.clone_inputs(canonical)

    expanded = impl.operator(*expanded_inputs.args)
    # Populate the second clone's physical cache with the same producer path.
    impl.operator(*single_inputs.args)

    (
        _hidden_states,
        _wkv_gate,
        _c4_state,
        _ape,
        _norm_weight,
        _rope_cos,
        _rope_sin,
        q_fp8,
        weights,
        kv_fused,
        context_lens,
        page_table,
        _schedule,
        _cache_write_locs,
        _raw_context,
        compressed_context,
        _rms_eps,
    ) = single_inputs.args

    import deep_gemm

    q_single = q_fp8.reshape(1, args.m, spec.HEADS, spec.HEAD_DIM)
    # DeepGEMM flattens (batch, next_n) for the weight rows even though Q and
    # context lengths retain the explicit two-dimensional request geometry.
    weights_single = weights
    context_single = context_lens.reshape(1, args.m)
    page_single = page_table[:1].contiguous()
    schedule_single = deep_gemm.get_paged_mqa_logits_metadata(
        context_single, spec.PAGE_SIZE, deep_gemm.get_num_sms()
    )
    single = deep_gemm.fp8_paged_mqa_logits(
        q_single,
        kv_fused,
        weights_single,
        context_single,
        page_single,
        schedule_single,
        compressed_context,
        False,
    )
    torch.cuda.synchronize()

    single_flat = single.reshape(args.m, compressed_context)
    rows = torch.arange(args.m, device="cuda", dtype=torch.long)
    valid_lens = context_lens[:, 0].long()
    cols = torch.remainder(rows * 2654435761, valid_lens)
    diff = (expanded[rows, cols] - single_flat[rows, cols]).abs()
    print(
        {
            "m": args.m,
            "expanded_q": tuple(q_fp8.shape),
            "single_q": tuple(q_single.shape),
            "expanded_context": tuple(context_lens.shape),
            "single_context": tuple(context_single.shape),
            "expanded_pages": tuple(page_table.shape),
            "single_pages": tuple(page_single.shape),
            "single_output": tuple(single.shape),
            "max_probe_abs_diff": float(diff.max().item()),
            "mean_probe_abs_diff": float(diff.mean().item()),
            "finite": bool(torch.isfinite(single_flat[rows, cols]).all().item()),
        }
    )


if __name__ == "__main__":
    main()
