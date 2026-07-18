#!/usr/bin/env python3
"""Install llm_flops contracts into KDA-Pilot task-native directories.

This is an administrative, one-time migration helper.  Normal Agent benchmark
runs consume baseline/ and solution/<version>/ in place and never stage source
back into the llm_flops checkout.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING = REPOSITORY_ROOT / "integrations" / "kda-pilot" / "operators.json"
REFERENCE_FILENAMES = ("README.md", "implementation.py", "operator.yaml", "spec.py")
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from benchmark_engine.registry import FilesystemRegistry  # noqa: E402
from benchmark_engine.registry.validation import compute_source_hash  # noqa: E402


def _digest_tree(root: Path) -> str:
    return compute_source_hash(root)


def _load_mapping(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or set(payload) != {"schema_version", "operators"}:
        raise ValueError("KDA operator mapping must use strict schema version 1")
    operators = payload["operators"]
    if not isinstance(operators, dict) or not operators:
        raise ValueError("KDA operator mapping must contain operators")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in operators.items()):
        raise TypeError("KDA operator mapping keys and values must be strings")
    if len({value.casefold() for value in operators.values()}) != len(operators):
        raise ValueError("KDA task slugs must be case-insensitively unique")
    return dict(sorted(operators.items()))


def _copy_reference(source: Path, destination: Path, *, refresh: bool, dry_run: bool) -> None:
    existing_manifest = destination / "operator.yaml"
    if existing_manifest.exists() and not refresh:
        source_digest = _digest_tree(source)
        existing_digest = _digest_tree(destination)
        if source_digest != existing_digest:
            raise FileExistsError(
                f"baseline already exists and differs: {destination}; use --refresh-baseline administratively"
            )
        return
    if dry_run:
        return
    destination.mkdir(parents=True, exist_ok=True)
    for filename in REFERENCE_FILENAMES:
        source_file = source / filename
        if not source_file.is_file():
            raise FileNotFoundError(f"reference contract is incomplete: {source_file}")
        shutil.copy2(source_file, destination / filename)
    placeholder = destination / ".gitkeep"
    if placeholder.exists():
        placeholder.unlink()


def _copy_candidates(source: Path, destination: Path, *, dry_run: bool) -> None:
    if not source.is_dir():
        return
    for candidate in sorted(path for path in source.iterdir() if path.is_dir()):
        target = destination / candidate.name
        if target.exists():
            if not target.is_dir() or _digest_tree(candidate) != _digest_tree(target):
                raise FileExistsError(f"solution version already exists and differs: {target}")
            continue
        if not dry_run:
            shutil.copytree(candidate, target)


def _task_prompt(operator_id: str, task_slug: str) -> str:
    return f"""# KDA kernel optimization task: {task_slug}

Optimize llm_flops operator `{operator_id}` on one idle target GPU.

The files in `baseline/` are the immutable reference contract. Agents MUST NOT
edit, replace, format, patch, or generate files in that directory. Read
`baseline/README.md`, `baseline/operator.yaml`, and `baseline/spec.py` before
implementing a candidate.

Create every candidate as an independent version directory:

```text
solution/<candidate_id>/implementation.py
```

The candidate must export the entrypoint declared by its optional
`candidate.yaml`, or `implementation:operator` by default. Candidate code may
not change cases, tolerances, comparators, timing policy, or the reference.

Run the complete correctness and performance gate from this task directory:

```bash
CUDA_VISIBLE_DEVICES=0 ../../../llm_flops/kda-bench.sh . <candidate_id>
```

Results are written to `bench/{operator_id}/<candidate_id>/<evaluation_id>/`.
Read `../../../llm_flops/docs/kda-pilot-integration.md` for the source and artifact
contracts. Do not stage or copy the candidate into the llm_flops repository.
"""


def _task_config(operator_id: str, task_slug: str) -> str:
    return f"""[task]
slug = \"{task_slug}\"
agent_enabled = true
operator_id = \"{operator_id}\"

[evaluation]
owner = \"llm_flops\"
layout = \"kda_task_v1\"
baseline = \"baseline/\"
solutions = \"solution/<candidate_id>/\"
results = \"bench/{operator_id}/<candidate_id>/<evaluation_id>/\"

[benchmark]
command = \"CUDA_VISIBLE_DEVICES=0 ../../../llm_flops/kda-bench.sh . <candidate_id>\"
correctness_required = true
single_gpu = true
"""


def _bench_readme(operator_id: str) -> str:
    return f"""# Benchmark artifacts

This directory is generated by llm_flops for `{operator_id}`. Do not place
benchmark source, a second oracle, or hand-edited result rows here.

```text
bench/{operator_id}/<candidate_id>/<evaluation_id>/
```

See `../../../../llm_flops/docs/result-layout.md` and
`../../../../llm_flops/docs/csv-schema.md`. `order_index` records interleaved
execution order; it is not a rank. Only rows with passed correctness and
`ranking_eligible=true` may be ranked.
"""


def _contract_readme(operator_id: str) -> str:
    return f"""# llm_flops task-native contract

Formal operator: `{operator_id}`.

`baseline/` is the immutable reference and semantic owner. Each immediate
directory under `solution/` is one candidate version. llm_flops loads both in
place and writes all generated artifacts below `bench/`; no staging adapter or
copy into `operators/candidates/` is used.

Use `../../../llm_flops/kda-bench.sh . <candidate_id>` from the task root. See
`../../../../llm_flops/docs/kda-pilot-integration.md` for the complete protocol.
"""


def install(
    llm_root: Path,
    mapping: dict[str, str],
    *,
    selected: frozenset[str],
    refresh_baseline: bool,
    include_candidates: bool,
    rewrite_task_guides: bool,
    dry_run: bool,
) -> tuple[str, ...]:
    installed: list[str] = []
    for operator_id, task_slug in mapping.items():
        if selected and operator_id not in selected:
            continue
        reference = REPOSITORY_ROOT / "operators" / "references" / operator_id
        if not reference.is_dir():
            raise FileNotFoundError(f"mapped reference does not exist: {reference}")
        task = llm_root / task_slug
        baseline = task / "baseline"
        solution = task / "solution"
        for directory in (baseline, solution, task / "bench", task / "docs", task / "profile", task / "ncu", task / "tests"):
            if not dry_run:
                directory.mkdir(parents=True, exist_ok=True)
        _copy_reference(reference, baseline, refresh=refresh_baseline, dry_run=dry_run)
        if include_candidates:
            _copy_candidates(
                REPOSITORY_ROOT / "operators" / "candidates" / operator_id,
                solution,
                dry_run=dry_run,
            )
        if not dry_run:
            (task / "bench" / "README.md").write_text(_bench_readme(operator_id), encoding="utf-8")
            (task / "docs" / "llm_flops_contract.md").write_text(_contract_readme(operator_id), encoding="utf-8")
            if rewrite_task_guides or not (task / "prompt.md").exists():
                (task / "prompt.md").write_text(_task_prompt(operator_id, task_slug), encoding="utf-8")
            if rewrite_task_guides or not (task / "config.toml").exists():
                (task / "config.toml").write_text(_task_config(operator_id, task_slug), encoding="utf-8")
        installed.append(f"{operator_id}\t{task_slug}")
    missing = selected.difference(mapping)
    if missing:
        raise KeyError("unmapped operator(s): " + ", ".join(sorted(missing)))
    return tuple(installed)


def verify(llm_root: Path, mapping: dict[str, str], selected: frozenset[str]) -> int:
    checked = 0
    failures: list[str] = []
    for operator_id, task_slug in mapping.items():
        if selected and operator_id not in selected:
            continue
        task = llm_root / task_slug
        snapshot = FilesystemRegistry.for_kda_task(task).discover()
        if not snapshot.is_valid:
            failures.extend(f"{operator_id}: {issue}" for issue in snapshot.issues)
            continue
        if tuple(snapshot.references) != (operator_id,):
            failures.append(f"{operator_id}: task declares the wrong reference identity")
            continue
        if snapshot.references[operator_id].root != task / "baseline":
            failures.append(f"{operator_id}: reference is not loaded from baseline/")
            continue
        if any(candidate.root.parent != task / "solution" for candidate in snapshot.candidates[operator_id]):
            failures.append(f"{operator_id}: candidate is not loaded from solution/<candidate_id>/")
            continue
        checked += 1
    if failures:
        raise ValueError("KDA workload verification failed:\n" + "\n".join(failures))
    return checked


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-root", required=True, type=Path)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--operator", action="append", default=[])
    parser.add_argument("--refresh-baseline", action="store_true")
    parser.add_argument("--without-candidates", action="store_true")
    parser.add_argument("--rewrite-task-guides", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    llm_root = arguments.llm_root.resolve()
    mapping = _load_mapping(arguments.mapping.resolve())
    selected = frozenset(arguments.operator)
    rows = () if arguments.verify_only else install(
        llm_root,
        mapping,
        selected=frozenset(arguments.operator),
        refresh_baseline=arguments.refresh_baseline,
        include_candidates=not arguments.without_candidates,
        rewrite_task_guides=arguments.rewrite_task_guides,
        dry_run=arguments.dry_run,
    )
    print("operator_id\ttask_slug")
    print("\n".join(rows))
    if arguments.verify_only or not arguments.dry_run:
        print(f"verified_tasks\t{verify(llm_root, mapping, selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
