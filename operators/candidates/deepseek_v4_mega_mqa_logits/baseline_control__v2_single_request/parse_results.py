from __future__ import annotations

import csv
import glob
import os


ROOT = (
    "/home/gjy/data/agent4kernel/mega_mqa_logits/llm_flops/results/"
    "deepseek_v4_mega_mqa_logits/baseline_control__v2_single_request"
)


for path in sorted(glob.glob(f"{ROOT}/20260731T09*/results.csv")):
    if "093741" in path:
        continue
    with open(path, newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["status"] == "passed"]
    if not rows:
        continue
    row = rows[0]
    print(
        os.path.basename(os.path.dirname(path)),
        row["case_id"],
        {
            "correctness_rows": len(rows),
            "reference_mean_ms": float(row["reference_mean_ms"]),
            "reference_median_ms": float(row["reference_median_ms"]),
            "reference_cv": float(row["reference_cv"]),
            "candidate_mean_ms": float(row["candidate_mean_ms"]),
            "candidate_median_ms": float(row["candidate_median_ms"]),
            "candidate_cv": float(row["candidate_cv"]),
            "speedup": float(row["speedup"]),
            "other_compute": row["other_compute_processes_detected"],
        },
    )
