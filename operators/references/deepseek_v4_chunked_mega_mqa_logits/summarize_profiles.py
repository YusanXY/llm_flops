"""Create compact machine-readable summaries from NCU and nsys artifacts."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from pathlib import Path


ROOT = Path(
    "/home/gjy/data/agent4kernel/mega_mqa_logits/llm_flops/results/"
    "deepseek_v4_chunked_mega_mqa_logits/full_kv_reuse_baseline"
)
NCU = ROOT / "ncu"
NSYS = ROOT / "nsys"


def load_ncu_raw():
    with (NCU / "m4096_raw_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise RuntimeError("NCU raw CSV must contain units and values")
    units, values = rows[-2], rows[-1]
    keep_exact = {
        "gpu__time_duration.sum",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "dram__bytes_read.sum.per_second",
        "dram__bytes_write.sum.per_second",
        "l1tex__m_xbar2l1tex_read_bytes.sum",
        "l1tex__m_xbar2l1tex_read_bytes.sum.per_second",
        "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum",
        "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum.per_second",
        "l1tex__m_l1tex2xbar_write_bytes.sum",
        "l1tex__t_sector_hit_rate.pct",
        "lts__t_sector_hit_rate.pct",
        "lts__t_sectors.sum",
        "lts__average_t_sector_hit_rate.pct",
        "inst_executed",
    }
    prefixes = (
        "launch__",
        "sm__pipe_tc_cycles_active",
        "sm__pipe_tensor_cycles_active",
        "sm__pipe_tma_cycles_active",
        "sm__mem_tensor_cycles_active",
        "smsp__warps_issue_stalled_",
    )
    selected = {}
    for name, value in values.items():
        if not value:
            continue
        if name in keep_exact or name.startswith(prefixes):
            selected[name] = {"unit": units.get(name, ""), "value": value}
    return values, units, selected


def to_float(values, name):
    value = values.get(name, "")
    return None if value == "" else float(value)


def ncu_derived(values):
    duration_us = to_float(values, "gpu__time_duration.sum")
    dram_read_mb = to_float(values, "dram__bytes_read.sum")
    dram_write_mb = to_float(values, "dram__bytes_write.sum")
    l1_read_gb = to_float(values, "l1tex__m_xbar2l1tex_read_bytes.sum")
    tma_read_gb = to_float(
        values, "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum"
    )
    l2_sectors = to_float(values, "lts__t_sectors.sum")
    return {
        "duration_us": duration_us,
        "dram_read_mb": dram_read_mb,
        "dram_write_mb": dram_write_mb,
        "l2_total_sector_bytes_gb": (
            None if l2_sectors is None else l2_sectors * 32 / 1e9
        ),
        "l2_to_l1_read_gb": l1_read_gb,
        "tma_global_to_shared_read_gb": tma_read_gb,
        "dram_read_fraction_of_l2_to_l1": (
            None
            if None in (dram_read_mb, l1_read_gb)
            else (dram_read_mb / 1000) / l1_read_gb
        ),
        "l1_hit_rate_pct": to_float(values, "l1tex__t_sector_hit_rate.pct"),
        "l2_hit_rate_pct": to_float(values, "lts__t_sector_hit_rate.pct"),
        "dram_read_effective_gbps": to_float(
            values, "dram__bytes_read.sum.per_second"
        ),
        "dram_write_effective_gbps": to_float(
            values, "dram__bytes_write.sum.per_second"
        ),
        "l2_to_l1_effective_tbps": to_float(
            values, "l1tex__m_xbar2l1tex_read_bytes.sum.per_second"
        ),
    }


def sass_summary():
    text = (NCU / "m4096_sass.txt").read_text(errors="replace")
    mnemonics = (
        "UTCQMMA",
        "UTMALDG.2D",
        "UTMALDG.3D",
        "UTCBAR",
        "UTMACCTL.PF",
        "UTMALD",
        "UTMAST",
        "UTMALD",
    )
    return {
        mnemonic: len(re.findall(rf"\b{re.escape(mnemonic)}\b", text))
        for mnemonic in mnemonics
    }


def nsys_summary():
    path = NSYS / "m4096_full_kv_reuse.sqlite"
    connection = sqlite3.connect(path)
    tables = [
        row[0]
        for row in connection.execute(
            "select name from sqlite_master where type='table' order by name"
        )
    ]
    metric_tables = [
        name
        for name in tables
        if "METRIC" in name.upper() or "GPU" in name.upper()
    ]
    cuda_kernel_tables = [
        name
        for name in tables
        if "CUPTI_ACTIVITY_KIND_KERNEL" in name.upper()
    ]
    result = {
        "tables_with_gpu_or_metric": metric_tables,
        "cuda_kernel_tables": cuda_kernel_tables,
    }
    for table in metric_tables:
        columns = [
            row[1]
            for row in connection.execute(f'pragma table_info("{table}")')
        ]
        result[f"{table}_columns"] = columns
        result[f"{table}_rows"] = connection.execute(
            f'select count(*) from "{table}"'
        ).fetchone()[0]
        sample = connection.execute(
            f'select * from "{table}" limit 3'
        ).fetchall()
        result[f"{table}_sample"] = [
            {
                column: (
                    value.hex()
                    if isinstance(value, bytes)
                    else value
                )
                for column, value in zip(columns, row)
            }
            for row in sample
        ]
    for table in cuda_kernel_tables:
        columns = [
            row[1]
            for row in connection.execute(f'pragma table_info("{table}")')
        ]
        result[f"{table}_columns"] = columns
        result[f"{table}_rows"] = connection.execute(
            f'select count(*) from "{table}"'
        ).fetchone()[0]
    connection.close()
    return result


def main():
    values, _, selected = load_ncu_raw()
    summary = {
        "ncu_set": "full",
        "ncu_selected_metrics": selected,
        "ncu_derived": ncu_derived(values),
        "sass_static_instruction_rows": sass_summary(),
        "ptx_available_in_ncu_report": False,
        "ptx_note": (
            "DeepGEMM runtime JIT report embeds SASS/cubin but NCU reports "
            "that PTX source is unavailable."
        ),
        "nsys": nsys_summary(),
    }
    target = ROOT / "profile_summary.json"
    target.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(target)
    print(json.dumps(summary["ncu_derived"], indent=2, sort_keys=True))
    print(json.dumps(summary["sass_static_instruction_rows"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
