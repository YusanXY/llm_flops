"""Locate V13 errors by query row and logical KV position."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
from benchmark_engine.correctness.inputs import make_generator_context


ROOT = Path("/home/gjy/data/agent4kernel/mega_mqa_logits/llm_flops")
REFERENCE = ROOT / "operators/references/deepseek_v4_chunked_mega_mqa_logits"
HERE = Path(__file__).resolve().parent


def load(name: str, path: Path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def main() -> None:
    spec_module = load("v13_diag_spec", REFERENCE / "spec.py")
    candidate = load("v13_diag_candidate", HERE / "implementation.py")
    case = next(c for c in spec_module.SPEC.cases() if c.symbols["m"] == 1024)
    torch.cuda.set_device(0)
    context = make_generator_context(case.seed, cuda_devices=("cuda:0",))
    inputs = spec_module.SPEC.make_inputs(case, context)
    output = candidate.operator(*inputs.args)
    torch.cuda.synchronize()

    indices = spec_module._sample_valid_output_indices(torch, 1024, output.device, 256)
    sampled = output.reshape(-1)[indices].float()
    oracle = inputs.observed_state["semantic_oracle"].float()
    errors = (sampled - oracle).abs()
    bad = torch.nonzero(errors > (1e-6 + 1e-5 * oracle.abs()), as_tuple=False).flatten()
    rows = torch.div(indices, spec_module.COMPRESSED_CONTEXT, rounding_mode="floor")
    positions = torch.remainder(indices, spec_module.COMPRESSED_CONTEXT)
    records = [
        {
            "ordinal": int(i),
            "query_row": int(rows[i]),
            "query_mod8": int(rows[i] % 8),
            "kv_position": int(positions[i]),
            "kv_mod256": int(positions[i] % 256),
            "candidate": float(sampled[i]),
            "oracle": float(oracle[i]),
            "abs_error": float(errors[i]),
        }
        for i in bad.cpu().tolist()
    ]
    q_fp8, weights, cache, lens, table = inputs.args[6:11]
    grid_rows = torch.tensor(
        list(range(8))
        + list(range(256, 264))
        + list(range(512, 520))
        + list(range(768, 776))
        + list(range(1016, 1024)),
        device=output.device,
        dtype=torch.long,
    )
    grid_positions = torch.tensor(
        [0, 127, 128, 255, 256, 4095, 8191, 12287, 12288, 13000, 15000, 16000],
        device=output.device,
        dtype=torch.long,
    )
    row_grid = grid_rows[:, None].expand(-1, grid_positions.numel()).reshape(-1)
    pos_grid = grid_positions[None, :].expand(grid_rows.numel(), -1).reshape(-1)
    page_columns = torch.div(pos_grid, 64, rounding_mode="floor")
    pages = table[0, page_columns].long()
    slots = torch.remainder(pos_grid, 64).long()
    cache_2d = cache.view(cache.shape[0], 64 * (128 + 4))
    cache_k = cache_2d[:, : 64 * 128].view(torch.float8_e4m3fn)
    columns = slots[:, None] * 128 + torch.arange(128, device=output.device)[None, :]
    k = cache_k[pages[:, None], columns].float()
    scale = cache_2d[:, 64 * 128 :].view(torch.float32)[pages, slots]
    q = q_fp8[0, row_grid].float()
    per_head = torch.relu((q * k[:, None, :]).sum(dim=-1))
    grid_oracle = (per_head * weights[row_grid]).sum(dim=-1) * scale
    grid_candidate = output[row_grid, pos_grid].float()
    grid_error = (grid_candidate - grid_oracle).abs().reshape(
        grid_rows.numel(), grid_positions.numel()
    )
    grid_records = {
        str(int(row)): [round(float(v), 6) for v in grid_error[index].cpu()]
        for index, row in enumerate(grid_rows.cpu())
    }
    print(
        json.dumps(
            {
                "bad_count": len(records),
                "records": records,
                "grid_positions": grid_positions.cpu().tolist(),
                "grid_abs_error_by_query": grid_records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
