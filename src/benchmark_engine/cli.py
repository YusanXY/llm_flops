"""Registry, environment, and Phase-3 dry-run planning commands."""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .config import DEFAULT_OUTPUT_ROOT
from .engine import (
    build_dry_run_plan,
    build_execution_plan,
    build_resume_plan,
    execute_plan,
)
from .environment import collect_report
from .registry import (
    FilesystemRegistry,
    RegistryIssue,
    RegistrySnapshot,
    selected_registry_issues,
)
from .selectors import Selectors
from .reporting import (ArtifactError, ResumeMismatchError, summarize_evaluation, summarize_run,
                        CompareCompatibilityError, compare_artifacts)
from .reporting.csv_writer import CsvContractError
from .ids import evaluation_result_path


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 3 CLI parser."""

    parser = argparse.ArgumentParser(
        prog="bench",
        description="Reproducible operator benchmark engine.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    commands = parser.add_subparsers(dest="command")

    list_parser = commands.add_parser(
        "list", help="list statically discovered operators and candidates"
    )
    list_parser.add_argument(
        "--operator", default="*", metavar="GLOB", help="operator ID glob"
    )
    list_parser.add_argument(
        "--candidate", default="*", metavar="GLOB", help="candidate ID glob"
    )
    list_parser.add_argument(
        "--task-root", type=Path, help="KDA task containing baseline/ and solution/"
    )

    validate_parser = commands.add_parser(
        "validate", help="validate operator and candidate manifests"
    )
    validate_parser.add_argument(
        "--operator", default="*", metavar="GLOB", help="operator ID glob"
    )
    validate_parser.add_argument(
        "--task-root", type=Path, help="KDA task containing baseline/ and solution/"
    )

    env_parser = commands.add_parser("env", help="collect the benchmark environment")
    env_parser.add_argument("--json", action="store_true", help="emit stable JSON")

    run_parser = commands.add_parser("run", help="run correctness and/or performance")
    run_parser.add_argument(
        "--task-root", type=Path, help="KDA task containing baseline/ and solution/"
    )
    run_parser.add_argument("--suite", metavar="ID")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--operator", action="append", default=[], metavar="GLOB")
    run_parser.add_argument("--candidate", action="append", default=[], metavar="GLOB")
    run_parser.add_argument("--case", action="append", default=[], metavar="GLOB")
    run_parser.add_argument("--tag", action="append", default=[], metavar="TAG")
    run_parser.add_argument(
        "--exclude-operator", action="append", default=[], metavar="GLOB"
    )
    run_parser.add_argument("--seed", action="append", default=[], type=int)
    run_parser.add_argument(
        "--mode", choices=("all", "correctness", "performance")
    )
    run_parser.add_argument("--output-root", type=Path)
    run_parser.add_argument("--evaluation-id")
    run_parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        help="resume unfinished results associated with RUN_ID",
    )
    run_parser.add_argument("--fail-fast", action="store_true")
    run_parser.add_argument("--timeout-s", type=float)
    run_parser.add_argument(
        "--timer",
        choices=("auto", "cuda_event", "cuda_graph", "wall_clock"),
        help="steady-state timer (CLI overrides suite/operator defaults)",
    )
    run_parser.add_argument("--warmup", type=int, help="performance warmup calls")
    run_parser.add_argument("--samples", type=int, help="raw performance samples")
    run_parser.add_argument(
        "--inner-iterations", type=int, help="calls represented by each raw sample"
    )
    run_parser.add_argument("--perf-on-correctness-fail", action="store_true", default=None,
                            help="measure failed candidates as permanently non-formal")
    run_parser.add_argument("--max-slowdown-pct", type=float)
    run_parser.add_argument("--min-speedup", type=float)
    run_parser.add_argument("--max-candidate-median-ms", type=float)
    run_parser.add_argument("--max-cv", type=float)
    run_parser.add_argument("--max-memory-bytes", type=int)
    run_parser.add_argument("--unsupported-policy", choices=("fail", "allow"))
    run_parser.add_argument("--gpu-lock-timeout-s", type=float)

    summary_parser = commands.add_parser(
        "summarize", help="summarize existing artifacts without executing code"
    )
    summary_parser.add_argument("target", nargs="?", type=Path)
    summary_parser.add_argument("--operator")
    summary_parser.add_argument("--candidate")
    summary_parser.add_argument("--evaluation")
    summary_parser.add_argument("--run", dest="summary_run")
    summary_parser.add_argument("--task-root", type=Path)
    summary_parser.add_argument("--output-root", type=Path)
    compare_parser = commands.add_parser("compare", help="compare compatible performance artifacts")
    current = compare_parser.add_mutually_exclusive_group(required=True)
    current.add_argument("--result", metavar="EVALUATION")
    current.add_argument("--run", metavar="RUN_ID")
    compare_parser.add_argument("--baseline-result", metavar="EVALUATION")
    compare_parser.add_argument("--baseline-run", metavar="RUN_ID")
    compare_parser.add_argument("--task-root", type=Path)
    compare_parser.add_argument("--output-root", type=Path)
    return parser


def _print_issues(issues: Sequence[RegistryIssue]) -> None:
    for issue in issues:
        print(issue, file=sys.stderr)


def _selected_operators(snapshot: RegistrySnapshot, pattern: str) -> tuple[str, ...]:
    return tuple(
        operator_id
        for operator_id in snapshot.discovered_operator_ids
        if fnmatch.fnmatchcase(operator_id, pattern)
    )


def _resolve_task_root(arguments: argparse.Namespace, root: Path) -> Path | None:
    value = getattr(arguments, "task_root", None)
    if value is None:
        return None
    return (value if value.is_absolute() else root / value).resolve()


def _resolve_output_root(
    arguments: argparse.Namespace, root: Path, task_root: Path | None
) -> Path:
    value = getattr(arguments, "output_root", None)
    if value is None:
        return (task_root / "bench" if task_root is not None else root / DEFAULT_OUTPUT_ROOT).resolve()
    return (value if value.is_absolute() else root / value).resolve()


def _run_list(
    snapshot: RegistrySnapshot,
    repository_root: Path,
    operator_glob: str,
    candidate_glob: str,
) -> int:
    selected = _selected_operators(snapshot, operator_glob)
    if not selected:
        print(f"selector matched no operators: {operator_glob}", file=sys.stderr)
        return 2
    issues = selected_registry_issues(
        snapshot,
        selected,
        candidate_patterns=(candidate_glob,),
        repository_root=repository_root,
    )
    if issues:
        _print_issues(issues)
        return 2

    rows: list[tuple[str, str, str]] = []
    for operator_id in selected:
        matching = tuple(
            candidate
            for candidate in snapshot.candidates.get(operator_id, ())
            if fnmatch.fnmatchcase(candidate.implementation_id, candidate_glob)
        )
        rows.extend(
            (operator_id, candidate.implementation_id, candidate.source_hash)
            for candidate in matching
        )
        if candidate_glob == "*" and not matching:
            rows.append((operator_id, "-", "-"))
    if not rows:
        print(f"selector matched no candidates: {candidate_glob}", file=sys.stderr)
        return 2

    print("operator_id\tcandidate_id\tsource_hash")
    for row in sorted(rows):
        print("\t".join(row))
    return 0


def _run_validate(
    snapshot: RegistrySnapshot, repository_root: Path, operator_glob: str
) -> int:
    selected = _selected_operators(snapshot, operator_glob)
    if not selected:
        print(f"selector matched no operators: {operator_glob}", file=sys.stderr)
        return 2
    issues = selected_registry_issues(
        snapshot, selected, repository_root=repository_root
    )
    if issues:
        _print_issues(issues)
        return 2
    candidate_count = sum(len(snapshot.candidates.get(item, ())) for item in selected)
    print(
        f"registry valid: {len(selected)} operator(s), "
        f"{candidate_count} candidate(s)"
    )
    return 0


def main(
    argv: Sequence[str] | None = None, *, repository_root: Path | None = None
) -> int:
    """Run an import-free registry command."""

    arguments = build_parser().parse_args(argv)
    if arguments.command is None:
        return 0
    root = Path(repository_root or Path.cwd()).resolve()
    task_root = _resolve_task_root(arguments, root)
    if arguments.command == "env":
        try:
            report = collect_report(root / "requirements" / "benchmark-lock.json")
        except (OSError, KeyError, TypeError, ValueError) as error:
            print(f"environment error: {error}", file=sys.stderr)
            return 2
        if arguments.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            observed = report["environment"]
            assert isinstance(observed, dict)
            gpu = observed.get("gpu", {})
            assert isinstance(gpu, dict)
            print(f"Benchmark environment: {report['fingerprint']}")
            print(f"Python {observed.get('python')}  CUDA {observed.get('cuda')}")
            print(f"GPU {gpu.get('name')}  capability={gpu.get('capability')}")
            for error in report["errors"]:
                print(f"ERROR: {error}")
        return 0
    if arguments.command == "summarize":
        identity_values = (
            arguments.operator,
            arguments.candidate,
            arguments.evaluation,
        )
        if arguments.target is not None and (any(identity_values) or arguments.summary_run):
            print("summarize target and identity selectors are mutually exclusive", file=sys.stderr)
            return 2
        if arguments.summary_run and any(identity_values):
            print("summarize --run and evaluation identity selectors are mutually exclusive", file=sys.stderr)
            return 2
        if arguments.target is None and not arguments.summary_run and not all(identity_values):
            print(
                "summarize requires a target or --operator/--candidate/--evaluation",
                file=sys.stderr,
            )
            return 2
        output_root = _resolve_output_root(arguments, root, task_root)
        if arguments.summary_run:
            try:
                print(summarize_run(output_root, arguments.summary_run), end="")
            except (OSError, CsvContractError, ValueError) as error:
                print(f"summary error: {error}", file=sys.stderr)
                return 2
            return 0
        target = arguments.target
        if target is None:
            target = evaluation_result_path(
                output_root,
                arguments.operator,
                arguments.candidate,
                arguments.evaluation,
            )
        elif not target.is_absolute():
            target = (task_root / "bench" if task_root is not None else root) / target
        try:
            print(summarize_evaluation(target), end="")
        except (OSError, CsvContractError, ValueError) as error:
            print(f"summary error: {error}", file=sys.stderr)
            return 3 if Path(target).exists() else 2
        return 0
    if arguments.command == "compare":
        run_mode = arguments.run is not None
        if run_mode != (arguments.baseline_run is not None):
            print("compare requires --run with --baseline-run", file=sys.stderr)
            return 2
        if not run_mode and arguments.baseline_result is None:
            print("compare requires --result with --baseline-result", file=sys.stderr)
            return 2
        if arguments.baseline_run is not None and arguments.baseline_result is not None:
            print("compare baseline forms are mutually exclusive", file=sys.stderr)
            return 2
        output_root = _resolve_output_root(arguments, root, task_root)
        try:
            print(compare_artifacts(output_root,
                arguments.run if run_mode else arguments.result,
                arguments.baseline_run if run_mode else arguments.baseline_result,
                run=run_mode), end="")
        except (CompareCompatibilityError, CsvContractError, OSError, ValueError) as error:
            print(f"compare error: {error}", file=sys.stderr)
            return 2
        return 0
    if arguments.command == "run":
        output_root = _resolve_output_root(arguments, root, task_root)
        selectors = Selectors(
            operators=tuple(arguments.operator),
            candidates=tuple(arguments.candidate),
            cases=tuple(arguments.case),
            tags=tuple(arguments.tag),
            exclude_operators=tuple(arguments.exclude_operator),
        )
        try:
            if arguments.dry_run:
                if arguments.resume is not None:
                    raise ValueError("--dry-run and --resume are mutually exclusive")
                plan = build_dry_run_plan(
                    root,
                    arguments.suite or "smoke",
                    selectors,
                    output_root=output_root,
                    mode=arguments.mode,
                    seeds=tuple(arguments.seed),
                    evaluation_id=arguments.evaluation_id,
                    performance_timer=arguments.timer,
                    performance_warmup=arguments.warmup,
                    performance_samples=arguments.samples,
                    performance_inner_iterations=arguments.inner_iterations,
                    performance_max_slowdown_pct=arguments.max_slowdown_pct,
                    perf_on_correctness_fail=arguments.perf_on_correctness_fail,
                    performance_min_speedup=arguments.min_speedup,
                    performance_max_candidate_median_ms=arguments.max_candidate_median_ms,
                    performance_max_cv=arguments.max_cv,
                    performance_max_memory_bytes=arguments.max_memory_bytes,
                    performance_unsupported_policy=arguments.unsupported_policy,
                    gpu_lock_timeout_s=arguments.gpu_lock_timeout_s,
                    task_root=task_root,
                )
                print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
                return 0
            if arguments.resume is not None:
                if any(
                    (
                        arguments.operator,
                        arguments.candidate,
                        arguments.case,
                        arguments.tag,
                        arguments.exclude_operator,
                        arguments.seed,
                        arguments.evaluation_id,
                        arguments.suite,
                        arguments.mode,
                        arguments.timer,
                        arguments.warmup is not None,
                        arguments.samples is not None,
                        arguments.inner_iterations is not None,
                        arguments.perf_on_correctness_fail is not None,
                        arguments.max_slowdown_pct is not None,
                        arguments.min_speedup is not None,
                        arguments.max_candidate_median_ms is not None,
                        arguments.max_cv is not None,
                        arguments.max_memory_bytes is not None,
                        arguments.unsupported_policy is not None,
                        arguments.gpu_lock_timeout_s is not None,
                    )
                ):
                    raise ValueError("--resume cannot be combined with selectors or identity overrides")
                plan, snapshot, environment = build_resume_plan(
                    root,
                    arguments.resume,
                    output_root=output_root,
                    task_root=task_root,
                )
            else:
                plan, snapshot, environment = build_execution_plan(
                    root,
                    arguments.suite or "smoke",
                    selectors,
                    output_root=output_root,
                    mode=arguments.mode,
                    seeds=tuple(arguments.seed),
                    evaluation_id=arguments.evaluation_id,
                    performance_timer=arguments.timer,
                    performance_warmup=arguments.warmup,
                    performance_samples=arguments.samples,
                    performance_inner_iterations=arguments.inner_iterations,
                    performance_max_slowdown_pct=arguments.max_slowdown_pct,
                    perf_on_correctness_fail=arguments.perf_on_correctness_fail,
                    performance_min_speedup=arguments.min_speedup,
                    performance_max_candidate_median_ms=arguments.max_candidate_median_ms,
                    performance_max_cv=arguments.max_cv,
                    performance_max_memory_bytes=arguments.max_memory_bytes,
                    performance_unsupported_policy=arguments.unsupported_policy,
                    gpu_lock_timeout_s=arguments.gpu_lock_timeout_s,
                    task_root=task_root,
                )
            outcome = execute_plan(
                plan,
                snapshot,
                environment,
                output_root=output_root,
                original_command=("bench", *(argv or sys.argv[1:])),
                resume=arguments.resume is not None,
                fail_fast=arguments.fail_fast,
                timeout_s=arguments.timeout_s,
                lock_root=root / ".runtime" / "locks",
            )
        except KeyboardInterrupt:
            print("interrupted by user", file=sys.stderr)
            return 130
        except ResumeMismatchError as error:
            print(str(error), file=sys.stderr)
            return 2
        except ArtifactError as error:
            print(str(error), file=sys.stderr)
            return 3
        except OSError as error:
            print(f"infrastructure error: {error}", file=sys.stderr)
            return 3
        except (KeyError, TypeError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return 2
        print(f"run_id: {outcome.run_id}")
        print(
            f"results: {outcome.passed} passed, {outcome.failed} failed; "
            f"infrastructure failures: {outcome.infrastructure_failures}"
        )
        for path in outcome.evaluation_paths:
            print(f"result: {path}")
        return outcome.exit_code

    registry = (
        FilesystemRegistry(root)
        if task_root is None
        else FilesystemRegistry.for_kda_task(task_root)
    )
    snapshot = registry.discover()
    if arguments.command == "list":
        return _run_list(
            snapshot,
            registry.repository_root,
            arguments.operator,
            arguments.candidate,
        )
    if arguments.command == "validate":
        return _run_validate(snapshot, registry.repository_root, arguments.operator)
    return 0
