"""Materialize and audit V13 cluster metadata coverage."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent


def load_candidate():
    module_spec = importlib.util.spec_from_file_location(
        "v13_schedule_candidate", HERE / "implementation.py"
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def main() -> None:
    candidate = load_candidate()
    m = 1024
    raw_prefix = 65536 - m
    lengths = torch.div(
        raw_prefix + torch.arange(1, m + 1, device="cuda", dtype=torch.int32),
        4,
        rounding_mode="floor",
    ).reshape(1, m)
    schedule = candidate._cluster2_schedule(lengths, m).cpu().tolist()

    num_requests = m // 8
    splits_per_request = [
        (int(lengths[0, request * 8 + 7]) + 255) // 256
        for request in range(num_requests)
    ]
    coverage = [[0] * splits_per_request[r] for r in range(num_requests)]

    for cluster in range(len(schedule) - 1):
        start_token, start_split = schedule[cluster]
        end_token, end_split = schedule[cluster + 1]
        request = start_token // 8
        split = start_split
        while request < num_requests:
            upper = end_split if request * 8 == end_token else splits_per_request[request]
            for kv_split in range(split, upper):
                coverage[request][kv_split] += 1
            if request * 8 == end_token:
                break
            request += 1
            split = 0

    gaps = []
    duplicates = []
    for request, row in enumerate(coverage):
        for split, count in enumerate(row):
            if count == 0:
                gaps.append((request, split))
            elif count != 1:
                duplicates.append((request, split, count))
    print(
        json.dumps(
            {
                "schedule_shape": [len(schedule), 2],
                "schedule_head": schedule[:5],
                "schedule_tail": schedule[-5:],
                "request_split_range": [min(splits_per_request), max(splits_per_request)],
                "gaps": gaps[:32],
                "gap_count": len(gaps),
                "duplicates": duplicates[:32],
                "duplicate_count": len(duplicates),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
