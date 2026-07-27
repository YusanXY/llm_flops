#!/usr/bin/env python3
"""Print compact diagnostics for one DeepSeek V4 Flash benchmark output root."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = (
    "operator_id",
    "case_id",
    "status",
    "correctness_status",
    "performance_status",
    "reference_median_ms",
    "candidate_median_ms",
    "reference_cv",
    "candidate_cv",
    "performance_gate_reasons",
    "error_type",
    "error_message",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    arguments = parser.parse_args()
    rows = []
    for path in arguments.output_root.rglob("results.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    print("\t".join(FIELDS))
    for row in sorted(rows, key=lambda item: (item["operator_id"], item["case_id"])):
        values = []
        for field in FIELDS:
            value = (row.get(field) or "").replace("\n", " ").replace("\t", " ")
            values.append(value[:240])
        print("\t".join(values))
    print(f"rows\t{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
