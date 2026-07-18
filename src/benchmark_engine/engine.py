"""Discovery-to-artifact orchestration for the correctness MVP."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from .environment import collect_environment, environment_fingerprint, load_lock, planning_fingerprint
from .execution import StageTimeouts, WorkerController, WorkerOutcome, WorkerStage
from .models import CorrectnessStatus, EvaluationJob, EvaluationPlan, PerformanceStatus, ResultStatus
from .operator_spec import load_operator_cases
from .planning import PlanBuilder
from .registry import FilesystemRegistry, RegistrySnapshot
from .reporting import (
    ArtifactWriter,
    EvaluationManifest,
    EvaluationState,
    ResumeMismatchError,
    ResumeReader,
)
from .reporting.csv_writer import (
    AtomicCsvTable,
    CORRECTNESS_OUTPUTS_SCHEMA,
    MODEL_PROJECTION_SCHEMA,
    PERFORMANCE_SAMPLES_SCHEMA,
    RESULTS_SCHEMA,
    atomic_write_text,
)
from .selectors import Selectors, select_cases
from .suite import load_suite
from .ids import generate_result_id
from .performance import PerformanceGateConfig, evaluate_performance_gate
from .execution.gpu_lock import GpuLock, resolve_gpu_identity
from .projection import projection_for_case


PERFORMANCE_SKIP_REASON = "performance_not_implemented"


def _projection_status(correctness, performance, error_message, performance_reason):
    if correctness is CorrectnessStatus.UNSUPPORTED:
        return "unsupported", str(error_message or "correctness_unsupported")
    if correctness is CorrectnessStatus.FAILED:
        return "correctness_failed", correctness.value
    if correctness is not CorrectnessStatus.PASSED:
        return "unavailable", str(error_message or correctness.value)
    if performance is PerformanceStatus.UNSUPPORTED:
        return "unsupported", str(performance_reason or "unsupported")
    if performance in {PerformanceStatus.PASSED, PerformanceStatus.UNSTABLE}:
        return "measured", None
    if performance is PerformanceStatus.SKIPPED:
        return "not_measured", str(performance_reason or "not_measured")
    return "unavailable", str(error_message or performance.value)


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    evaluation_paths: tuple[Path, ...]
    passed: int
    failed: int
    infrastructure_failures: int
    interrupted: bool = False

    @property
    def exit_code(self) -> int:
        if self.interrupted:
            return 130
        if self.infrastructure_failures:
            return 3
        return 1 if self.failed else 0


def _suite(repository_root: Path, suite_id: str):
    if re.fullmatch(r"[a-z][a-z0-9_-]{1,79}", suite_id) is None:
        raise ValueError("suite ID is invalid")
    suite = load_suite(repository_root / "suites" / f"{suite_id}.yaml")
    if suite.suite_id != suite_id:
        raise ValueError(
            f"suite_id {suite.suite_id!r} does not match requested {suite_id!r}"
        )
    return suite


def _registry(repository_root: Path, task_root: Path | None) -> FilesystemRegistry:
    """Select the repository or KDA task-native source layout."""

    if task_root is None:
        return FilesystemRegistry(repository_root)
    return FilesystemRegistry.for_kda_task(Path(task_root).resolve())


def build_dry_run_plan(
    repository_root: Path,
    suite_id: str,
    selectors: Selectors,
    *,
    output_root: Path,
    mode: str | None = None,
    seeds: tuple[int, ...] = (),
    evaluation_id: str | None = None,
    resume: bool = False,
    plan_builder: PlanBuilder | None = None,
    performance_timer: str | None = None,
    performance_warmup: int | None = None,
    performance_samples: int | None = None,
    performance_inner_iterations: int | None = None,
    performance_max_slowdown_pct: float | None = None,
    perf_on_correctness_fail: bool | None = None,
    performance_min_speedup: float | None = None,
    performance_max_candidate_median_ms: float | None = None,
    performance_max_cv: float | None = None,
    performance_max_memory_bytes: int | None = None,
    performance_unsupported_policy: str | None = None,
    gpu_lock_timeout_s: float | None = None,
    task_root: Path | None = None,
):
    """Build an import-free-of-candidates, artifact-free plan."""

    root = Path(repository_root).resolve()
    suite = _suite(root, suite_id)
    snapshot = _registry(root, task_root).discover()
    lock = load_lock(root / "requirements" / "benchmark-lock.json")
    fingerprint = planning_fingerprint(lock)
    return (plan_builder or PlanBuilder()).build(
        snapshot,
        suite,
        selectors,
        environment_fingerprint=fingerprint,
        output_root=output_root,
        mode=mode,
        seeds=seeds,
        evaluation_id=evaluation_id,
        resume=resume,
        performance_timer=performance_timer,
        performance_warmup=performance_warmup,
        performance_samples=performance_samples,
        performance_inner_iterations=performance_inner_iterations,
        performance_max_slowdown_pct=performance_max_slowdown_pct,
        perf_on_correctness_fail=perf_on_correctness_fail,
        performance_min_speedup=performance_min_speedup,
        performance_max_candidate_median_ms=performance_max_candidate_median_ms,
        performance_max_cv=performance_max_cv,
        performance_max_memory_bytes=performance_max_memory_bytes,
        performance_unsupported_policy=performance_unsupported_policy,
        gpu_lock_timeout_s=gpu_lock_timeout_s,
    )


def _runtime_environment(
    root: Path, *, include_cuda: bool = False
) -> tuple[dict[str, object], str]:
    lock = load_lock(root / "requirements" / "benchmark-lock.json")
    # Correctness-only CPU runs avoid CUDA initialization.  Performance/all
    # identities include the CUDA runtime and device metadata they measure.
    observed = collect_environment(lock, include_cuda=include_cuda)
    return observed, environment_fingerprint(observed)


def build_execution_plan(
    repository_root: Path,
    suite_id: str,
    selectors: Selectors,
    *,
    output_root: Path,
    mode: str | None = None,
    seeds: tuple[int, ...] = (),
    evaluation_id: str | None = None,
    performance_timer: str | None = None,
    performance_warmup: int | None = None,
    performance_samples: int | None = None,
    performance_inner_iterations: int | None = None,
    performance_max_slowdown_pct: float | None = None,
    perf_on_correctness_fail: bool | None = None,
    performance_min_speedup: float | None = None,
    performance_max_candidate_median_ms: float | None = None,
    performance_max_cv: float | None = None,
    performance_max_memory_bytes: int | None = None,
    performance_unsupported_policy: str | None = None,
    gpu_lock_timeout_s: float | None = None,
    task_root: Path | None = None,
) -> tuple[EvaluationPlan, RegistrySnapshot, Mapping[str, object]]:
    root = Path(repository_root).resolve()
    suite = _suite(root, suite_id)
    snapshot = _registry(root, task_root).discover()
    resolved_mode = mode or suite.mode
    environment, fingerprint = _runtime_environment(
        root, include_cuda=resolved_mode in {"all", "performance"}
    )
    plan = PlanBuilder().build(
        snapshot,
        suite,
        selectors,
        environment_fingerprint=fingerprint,
        output_root=output_root,
        mode=mode,
        seeds=seeds,
        evaluation_id=evaluation_id,
        performance_timer=performance_timer,
        performance_warmup=performance_warmup,
        performance_samples=performance_samples,
        performance_inner_iterations=performance_inner_iterations,
        performance_max_slowdown_pct=performance_max_slowdown_pct,
        perf_on_correctness_fail=perf_on_correctness_fail,
        performance_min_speedup=performance_min_speedup,
        performance_max_candidate_median_ms=performance_max_candidate_median_ms,
        performance_max_cv=performance_max_cv,
        performance_max_memory_bytes=performance_max_memory_bytes,
        performance_unsupported_policy=performance_unsupported_policy,
        gpu_lock_timeout_s=gpu_lock_timeout_s,
    )
    return replace(plan, fingerprint_kind="runtime"), snapshot, environment


def _candidate(snapshot: RegistrySnapshot, operator_id: str, candidate_id: str):
    for candidate in snapshot.candidates.get(operator_id, ()):
        if candidate.implementation_id == candidate_id:
            return candidate
    raise ResumeMismatchError(
        f"resume candidate is not present in registry: {operator_id}/{candidate_id}"
    )


def _resume_selectors(command: Sequence[str]) -> Selectors:
    """Recover case/tag narrowing from the immutable original command."""

    cases: list[str] = []
    tags: list[str] = []
    tokens = tuple(command[1:] if command and command[0] == "bench" else command)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        destination = None
        if token in {"--case", "--tag"} and index + 1 < len(tokens):
            destination = cases if token == "--case" else tags
            destination.append(tokens[index + 1])
            index += 2
            continue
        for option, destination in (("--case=", cases), ("--tag=", tags)):
            if token.startswith(option):
                destination.append(token[len(option) :])
                break
        index += 1
    return Selectors(cases=tuple(cases), tags=tuple(tags))


def build_resume_plan(
    repository_root: Path,
    run_id: str,
    *,
    output_root: Path,
    task_root: Path | None = None,
) -> tuple[EvaluationPlan, RegistrySnapshot, Mapping[str, object]]:
    root = Path(repository_root).resolve()
    snapshot = _registry(root, task_root).discover()
    states = ResumeReader(output_root).for_run(run_id)
    jobs: list[EvaluationJob] = []
    suite_ids = {state.manifest.suite_id for state in states}
    modes = {state.manifest.mode for state in states}
    if len(suite_ids) != 1 or len(modes) != 1:
        raise ResumeMismatchError("run contains incompatible suite or mode values")
    suite_id = next(iter(suite_ids))
    environment, fingerprint = _runtime_environment(
        root, include_cuda=next(iter(modes)) in {"all", "performance"}
    )
    suite = _suite(root, suite_id)
    for state in states:
        manifest = state.manifest
        identity = manifest.identity
        reference = snapshot.references.get(identity.operator_id)
        if reference is None:
            raise ResumeMismatchError(
                f"resume reference is not present: {identity.operator_id}"
            )
        candidate = _candidate(snapshot, identity.operator_id, identity.candidate_id)
        mismatches = []
        if reference.source_hash != manifest.reference_source_hash:
            mismatches.append("reference source")
        if candidate.source_hash != manifest.candidate_source_hash:
            mismatches.append("candidate source")
        if fingerprint != manifest.environment_fingerprint:
            mismatches.append("environment fingerprint")
        if mismatches:
            raise ResumeMismatchError("resume is incompatible: " + ", ".join(mismatches))
        cases = select_cases(
            load_operator_cases(snapshot, identity.operator_id),
            suite,
            _resume_selectors(manifest.original_command),
        )
        for case in cases:
            for seed in manifest.resolved_config.correctness_seeds:
                job_case = replace(case, seed=seed)
                jobs.append(
                    EvaluationJob(
                        identity=identity,
                        reference=reference,
                        candidate=candidate,
                        case=job_case,
                        mode=manifest.mode,
                        output_dir=Path(output_root).resolve()
                        / identity.operator_id
                        / identity.candidate_id
                        / identity.evaluation_id,
                        result_id=generate_result_id(
                            identity.operator_id,
                            identity.candidate_id,
                            identity.evaluation_id,
                            case.case_id,
                            seed,
                        ),
                        resolved_config=manifest.resolved_config,
                    )
                )
    jobs.sort(
        key=lambda job: (
            job.identity.operator_id,
            job.identity.candidate_id,
            job.case.case_id,
            job.case.seed,
        )
    )
    if not jobs:
        raise ResumeMismatchError("resume run has no expected jobs")
    return (
        EvaluationPlan(
            run_id=run_id,
            mode=next(iter(modes)),
            jobs=tuple(jobs),
            environment_fingerprint=fingerprint,
            suite_id=suite_id,
            fingerprint_kind="runtime",
        ),
        snapshot,
        environment,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _metric(metrics: Mapping[str, object], name: str) -> object | None:
    direct = metrics.get(name)
    if isinstance(direct, (int, float)) and not isinstance(direct, bool):
        return direct
    values = [
        item[name]
        for item in metrics.values()
        if isinstance(item, Mapping)
        and isinstance(item.get(name), (int, float))
        and not isinstance(item.get(name), bool)
    ]
    return max(values) if values else None


def _correctness_status(raw: str) -> CorrectnessStatus:
    return {
        "pass": CorrectnessStatus.PASSED,
        "fail": CorrectnessStatus.FAILED,
        "nondeterministic": CorrectnessStatus.FAILED,
        "error": CorrectnessStatus.ERROR,
        "timeout": CorrectnessStatus.TIMEOUT,
        "oom": CorrectnessStatus.OOM,
        "unsupported": CorrectnessStatus.UNSUPPORTED,
        "crashed": CorrectnessStatus.CRASHED,
    }[raw]


def _result_status(status: CorrectnessStatus) -> ResultStatus:
    return {
        CorrectnessStatus.PASSED: ResultStatus.PASSED,
        CorrectnessStatus.FAILED: ResultStatus.FAILED,
        CorrectnessStatus.ERROR: ResultStatus.ERROR,
        CorrectnessStatus.TIMEOUT: ResultStatus.TIMEOUT,
        CorrectnessStatus.OOM: ResultStatus.OOM,
        CorrectnessStatus.UNSUPPORTED: ResultStatus.UNSUPPORTED,
        CorrectnessStatus.CRASHED: ResultStatus.CRASHED,
    }[status]


def _performance_status(raw: str) -> PerformanceStatus:
    try:
        return {
            "pass": PerformanceStatus.PASSED,
            "unstable": PerformanceStatus.UNSTABLE,
            "skipped": PerformanceStatus.SKIPPED,
            "unsupported": PerformanceStatus.UNSUPPORTED,
            "error": PerformanceStatus.ERROR,
            "timeout": PerformanceStatus.TIMEOUT,
            "oom": PerformanceStatus.OOM,
            "crashed": PerformanceStatus.CRASHED,
            "failed": PerformanceStatus.FAILED,
        }[raw]
    except KeyError as error:
        raise ValueError(f"unknown performance status {raw!r}") from error


def _overall_status(
    correctness: CorrectnessStatus, performance: PerformanceStatus
) -> ResultStatus:
    if correctness is not CorrectnessStatus.PASSED:
        return _result_status(correctness)
    return {
        PerformanceStatus.PASSED: ResultStatus.PASSED,
        PerformanceStatus.UNSTABLE: ResultStatus.PASSED,
        PerformanceStatus.SKIPPED: ResultStatus.PASSED,
        PerformanceStatus.FAILED: ResultStatus.FAILED,
        PerformanceStatus.UNSUPPORTED: ResultStatus.UNSUPPORTED,
        PerformanceStatus.ERROR: ResultStatus.ERROR,
        PerformanceStatus.TIMEOUT: ResultStatus.TIMEOUT,
        PerformanceStatus.OOM: ResultStatus.OOM,
        PerformanceStatus.CRASHED: ResultStatus.CRASHED,
        PerformanceStatus.PLANNED: ResultStatus.ERROR,
    }[performance]


def _manifest_device(device_types: Sequence[str]) -> str:
    """Project declared device support without guessing from a timer backend.

    A one-device operator records exactly ``cpu`` or ``cuda``.  Until device
    selection becomes an explicit planning dimension, a multi-device manifest
    records its canonical sorted support set (for example ``cpu+cuda``).
    """

    devices = tuple(sorted({device.strip().lower() for device in device_types}))
    if not devices or any(not device for device in devices):
        raise ValueError("operator manifest device_types must not be empty")
    return "+".join(devices)


def _manifest_cuda_devices(device_types: Sequence[str]) -> tuple[str, ...]:
    """Return deterministic generator devices for the current single worker."""

    return (
        ("cuda:0",)
        if any(device.strip().lower().startswith("cuda") for device in device_types)
        else ()
    )


def _environment_package_version(
    environment_snapshot: Mapping[str, object], package: str
) -> str | None:
    """Read a collected package version without inventing legacy flat keys."""

    packages = environment_snapshot.get("packages")
    if not isinstance(packages, Mapping):
        return None
    metadata = packages.get(package)
    if not isinstance(metadata, Mapping):
        return None
    version = metadata.get("version")
    return version if isinstance(version, str) and version else None


def _payload_for_worker_response(response) -> tuple[dict[str, object] | None, bool]:
    if response.result_payload is not None:
        return dict(response.result_payload), False
    if response.stage is WorkerStage.CORRECTNESS and response.outcome in {
        WorkerOutcome.TIMEOUT,
        WorkerOutcome.OOM,
        WorkerOutcome.UNSUPPORTED,
    }:
        return {
            "status": response.outcome.value,
            "case_hash": "unavailable",
            "input_summary": {},
            "comparison": None,
            "diagnostic": {
                "kind": response.outcome.value,
                "message": response.error_message,
            },
            "performance": {
                "status": "skipped",
                "reason": "correctness_gate_failed",
            },
        }, False
    if response.stage in {WorkerStage.WARMUP, WorkerStage.SAMPLING} and response.outcome in {
        WorkerOutcome.ERROR,
        WorkerOutcome.TIMEOUT,
        WorkerOutcome.OOM,
        WorkerOutcome.UNSUPPORTED,
        WorkerOutcome.CRASHED,
    }:
        # Reaching either performance stage proves that the trusted worker
        # completed the correctness gate.  A controller-side kill/crash has no
        # result payload, so synthesize only the durable terminal status--never
        # samples, statistics, cost, or comparison metrics.
        return {
            "status": "pass",
            "case_hash": "unavailable",
            "input_summary": {},
            "comparison": {"metrics": {}},
            "output_contracts": {},
            "performance": {
                "status": response.outcome.value,
                "reason": response.error_message or response.outcome.value,
            },
        }, False
    return None, True


def _result_error_fields(
    raw_status: str,
    diagnostic: Mapping[str, object] | None,
    response,
) -> tuple[str | None, str | None]:
    """Project bounded, stable error columns from a structured diagnostic."""

    if response.error_type or response.error_message:
        return response.error_type, response.error_message
    if raw_status in {"pass", "fail", "nondeterministic"}:
        return None, None
    error_type: str | None = raw_status
    error_message: str | None = None
    if diagnostic is not None:
        exception_type = diagnostic.get("exception_type")
        message = diagnostic.get("message")
        if isinstance(exception_type, str) and exception_type:
            error_type = exception_type
        if isinstance(message, str) and message:
            error_message = message
    return (
        None if error_type is None else error_type[:128],
        None if error_message is None else error_message[:512],
    )


def _performance_sample_contract_fields(
    output_rows: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    """Encode every normalized output contract without multiplying samples.

    A performance sample covers the whole operator invocation rather than one
    output leaf.  The four CSV columns therefore contain compact JSON objects
    keyed by normalized ``output_path``.  This remains unambiguous for
    operators that return tuples, mappings, or other nested structures.
    """

    encoded: dict[str, str] = {}
    for field in (
        "reference_dtype",
        "candidate_dtype",
        "reference_shape",
        "candidate_shape",
    ):
        values: dict[str, object] = {}
        for row in output_rows:
            output_path = str(row["output_path"])
            value = row[field]
            if field.endswith("_shape"):
                value = json.loads(str(value))
            values[output_path] = value
        encoded[field] = json.dumps(
            values, sort_keys=True, separators=(",", ":")
        )
    return encoded


def _append_result(
    job: EvaluationJob,
    snapshot: RegistrySnapshot,
    plan: EvaluationPlan,
    response,
    environment_snapshot: Mapping[str, object],
) -> tuple[bool, bool]:
    payload, infrastructure = _payload_for_worker_response(response)
    if infrastructure or payload is None:
        return False, True
    raw_status = payload.get("status")
    if not isinstance(raw_status, str) or raw_status not in {
        "pass", "fail", "nondeterministic", "error", "timeout", "oom", "unsupported", "crashed"
    }:
        return False, True
    correctness = _correctness_status(raw_status)
    comparison = payload.get("comparison")
    comparison = comparison if isinstance(comparison, Mapping) else {}
    metrics = comparison.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    raw_diagnostic = payload.get("diagnostic")
    diagnostic = raw_diagnostic if isinstance(raw_diagnostic, Mapping) else None
    error_type, error_message = _result_error_fields(
        raw_status, diagnostic, response
    )
    manifest = snapshot.operator_manifests[job.identity.operator_id]
    output_contracts = payload.get("output_contracts")
    output_contracts = output_contracts if isinstance(output_contracts, Mapping) else {}
    output_rows: list[dict[str, object]] = []
    for output_path, raw_contract in sorted(output_contracts.items()):
        if not isinstance(output_path, str) or not isinstance(raw_contract, Mapping):
            continue
        path_metrics = metrics.get(output_path)
        path_metrics = path_metrics if isinstance(path_metrics, Mapping) else metrics
        mismatch_count = _metric(path_metrics, "mismatch_count")
        path_passed = correctness is CorrectnessStatus.PASSED or mismatch_count == 0
        output_rows.append(
            {
                "result_id": job.result_id,
                "output_path": output_path,
                "comparator": str(comparison.get("comparator") or "unknown"),
                "reference_dtype": str(raw_contract.get("reference_dtype") or "unknown"),
                "candidate_dtype": str(raw_contract.get("candidate_dtype") or "unknown"),
                "reference_shape": json.dumps(raw_contract.get("reference_shape", [])),
                "candidate_shape": json.dumps(raw_contract.get("candidate_shape", [])),
                "passed": path_passed,
                "rtol": _metric(path_metrics, "rtol"),
                "atol": _metric(path_metrics, "atol"),
                "max_abs_error": _metric(path_metrics, "max_abs_error"),
                "mean_abs_error": _metric(path_metrics, "mean_abs_error"),
                "p95_abs_error": _metric(path_metrics, "p95_abs_error"),
                "max_rel_error": _metric(path_metrics, "max_rel_error"),
                "rmse": _metric(path_metrics, "rmse"),
                "rel_l2": _metric(path_metrics, "relative_l2"),
                "cosine_similarity": _metric(path_metrics, "cosine_similarity"),
                "mismatch_count": mismatch_count,
                "mismatch_rate": _metric(path_metrics, "mismatch_rate"),
                "reference_nan_count": _metric(path_metrics, "reference_nan_count"),
                "candidate_nan_count": _metric(path_metrics, "candidate_nan_count"),
                "diagnostic_path": response.diagnostic_path,
            }
        )
    failed_output_count = sum(not bool(output["passed"]) for output in output_rows)
    sample_contract_fields = _performance_sample_contract_fields(output_rows)
    if correctness is not CorrectnessStatus.PASSED and not failed_output_count:
        failed_output_count = 1
    # The trusted correctness gate dominates every worker-provided performance
    # field.  A malformed/malicious candidate cannot publish samples for a
    # failed correctness result.
    trusted_opt_in = job.resolved_config.perf_on_correctness_fail
    raw_performance = payload.get("performance") if (
        correctness is CorrectnessStatus.PASSED or trusted_opt_in
    ) else {"status": "skipped", "reason": "correctness_gate_failed"}
    if not isinstance(raw_performance, Mapping):
        return False, True
    raw_performance_status = raw_performance.get("status")
    if not isinstance(raw_performance_status, str):
        return False, True
    performance = _performance_status(raw_performance_status)
    measurements = raw_performance.get("measurements")
    measurements = measurements if isinstance(measurements, Mapping) else {}
    reference_measurement = measurements.get("reference")
    candidate_measurement = measurements.get("candidate")
    reference_measurement = (
        reference_measurement if isinstance(reference_measurement, Mapping) else {}
    )
    candidate_measurement = (
        candidate_measurement if isinstance(candidate_measurement, Mapping) else {}
    )

    def _selection(measurement: Mapping[str, object]) -> Mapping[str, object]:
        value = measurement.get("selection")
        return value if isinstance(value, Mapping) else {}

    def _statistics(measurement: Mapping[str, object]) -> Mapping[str, object]:
        value = measurement.get("statistics")
        return value if isinstance(value, Mapping) else {}

    reference_selection = _selection(reference_measurement)
    candidate_selection = _selection(candidate_measurement)
    reference_statistics = _statistics(reference_measurement)
    candidate_statistics = _statistics(candidate_measurement)
    sample_rows: list[dict[str, object]] = []
    if performance in {PerformanceStatus.PASSED, PerformanceStatus.UNSTABLE}:
        role_indices: dict[str, list[int]] = {}
        all_orders: list[int] = []
        for role, measurement, selection in (
            ("reference", reference_measurement, reference_selection),
            ("candidate", candidate_measurement, candidate_selection),
        ):
            raw_samples = measurement.get("samples")
            if not isinstance(raw_samples, list) or not raw_samples:
                return False, True
            requested_timer = selection.get("requested_timer")
            effective_timer = selection.get("effective_timer")
            fallback_reason = selection.get("fallback_reason")
            if not isinstance(requested_timer, str) or not isinstance(
                effective_timer, str
            ):
                return False, True
            role_indices[role] = []
            for raw_sample in raw_samples:
                if not isinstance(raw_sample, Mapping):
                    return False, True
                sample_index = raw_sample.get("sample_index")
                if isinstance(sample_index, bool) or not isinstance(sample_index, int):
                    return False, True
                order_index = raw_sample.get("order_index")
                if isinstance(order_index, bool) or not isinstance(order_index, int):
                    return False, True
                role_indices[role].append(sample_index)
                all_orders.append(order_index)
                sample_rows.append(
                    {
                        "result_id": job.result_id,
                        "implementation_role": role,
                        **sample_contract_fields,
                        "sample_index": sample_index,
                        "inner_iterations": raw_sample.get("inner_iterations"),
                        "elapsed_ms": raw_sample.get("elapsed_ms"),
                        "per_call_ms": raw_sample.get("per_call_ms"),
                        "order_index": order_index,
                        "requested_timer": requested_timer,
                        "effective_timer": effective_timer,
                        "fallback_reason": fallback_reason,
                    }
                )
        counts = {role: len(indices) for role, indices in role_indices.items()}
        if set(counts) != {"reference", "candidate"} or len(set(counts.values())) != 1:
            return False, True
        if any(indices != list(range(len(indices))) for indices in role_indices.values()):
            return False, True
        if sorted(all_orders) != list(range(len(all_orders))):
            return False, True
    skip_reason = (
        str(raw_performance.get("reason") or "performance_skipped")
        if performance is PerformanceStatus.SKIPPED
        else None
    )
    cost = raw_performance.get("cost")
    cost = cost if isinstance(cost, Mapping) else {}
    gpu = environment_snapshot.get("gpu")
    gpu = gpu if isinstance(gpu, Mapping) else {}
    gpu_runtime = raw_performance.get("gpu_runtime")
    gpu_runtime = gpu_runtime if isinstance(gpu_runtime, Mapping) else {}
    gate = evaluate_performance_gate(
        raw_performance,
        PerformanceGateConfig(
            max_slowdown_pct=job.resolved_config.performance_regression_threshold_pct,
            min_speedup=job.resolved_config.performance_min_speedup,
            max_candidate_median_ms=job.resolved_config.performance_max_candidate_median_ms,
            max_cv=job.resolved_config.performance_max_cv,
            max_memory_bytes=job.resolved_config.performance_max_memory_bytes,
            unsupported_policy=job.resolved_config.performance_unsupported_policy,
        ),
        correctness_pass=correctness is CorrectnessStatus.PASSED,
        perf_on_correctness_fail=correctness is not CorrectnessStatus.PASSED and trusted_opt_in,
    )
    overall = _overall_status(correctness, performance)
    if correctness is CorrectnessStatus.PASSED and job.mode != "correctness":
        if (
            performance in {PerformanceStatus.PASSED, PerformanceStatus.UNSTABLE}
            and gate.status == "failed"
        ):
            overall = ResultStatus.FAILED
        elif (
            performance is PerformanceStatus.UNSUPPORTED
            and gate.status == "passed"
        ):
            # unsupported_policy=allow is a successful gate but remains
            # visibly unsupported and permanently ineligible for ranking.
            overall = ResultStatus.PASSED
    row = {
        "run_id": job.identity.run_id,
        "evaluation_id": job.identity.evaluation_id,
        "timestamp_utc": _utc_now(),
        "suite_id": plan.suite_id,
        "mode": job.mode,
        "result_id": job.result_id,
        "operator_id": job.identity.operator_id,
        "contract_version": manifest.contract_version,
        "candidate_id": job.identity.candidate_id,
        "reference_id": job.reference.implementation_id,
        "candidate_source_hash": job.candidate.source_hash,
        "reference_source_hash": job.reference.source_hash,
        "imported_legacy": False,
        "environment_fingerprint": plan.environment_fingerprint,
        "device": _manifest_device(manifest.device_types),
        "gpu_name": gpu.get("name"),
        "gpu_uuid": gpu_runtime.get("gpu_uuid"),
        "logical_device": gpu_runtime.get("logical_device"),
        "visible_device": gpu_runtime.get("visible_device"),
        "cuda_visible_devices": gpu_runtime.get("cuda_visible_devices"),
        "driver_version": gpu_runtime.get("driver_version"),
        "other_compute_processes_detected": gpu_runtime.get("other_compute_processes_detected"),
        "telemetry_error": gpu_runtime.get("telemetry_error"),
        "cuda_version": environment_snapshot.get("cuda"),
        "torch_version": _environment_package_version(environment_snapshot, "torch"),
        "case_id": job.case.case_id,
        "case_hash": str(payload.get("case_hash") or "unavailable"),
        "seed": job.case.seed,
        "tags": json.dumps(sorted(job.case.tags)),
        "input_summary": json.dumps(payload.get("input_summary", {}), sort_keys=True),
        "status": overall.value,
        "correctness_status": correctness.value,
        "performance_status": performance.value,
        "skip_reason": skip_reason,
        "correctness_pass": correctness is CorrectnessStatus.PASSED,
        "failed_output_count": failed_output_count,
        "max_abs_error": _metric(metrics, "max_abs_error"),
        "max_rel_error": _metric(metrics, "max_rel_error"),
        "rmse": _metric(metrics, "rmse"),
        "rel_l2": _metric(metrics, "relative_l2"),
        "cosine_similarity": _metric(metrics, "cosine_similarity"),
        "mismatch_count": _metric(metrics, "mismatch_count"),
        "mismatch_rate": _metric(metrics, "mismatch_rate"),
        "timer": candidate_selection.get("effective_timer"),
        "requested_timer": candidate_selection.get("requested_timer"),
        "effective_timer": candidate_selection.get("effective_timer"),
        "timer_fallback_reason": candidate_selection.get("fallback_reason"),
        "reference_requested_timer": reference_selection.get("requested_timer"),
        "reference_effective_timer": reference_selection.get("effective_timer"),
        "reference_timer_fallback_reason": reference_selection.get("fallback_reason"),
        "import_ms": response.stage_elapsed_s.get("import", 0.0) * 1000.0,
        "build_ms": response.stage_elapsed_s.get("build", 0.0) * 1000.0,
        "first_call_ms": candidate_measurement.get("first_call_ms"),
        "warmup_ms": candidate_measurement.get("warmup_ms"),
        "graph_capture_ms": candidate_measurement.get("graph_capture_ms"),
        "steady_state_ms": candidate_measurement.get("steady_state_ms"),
        "reference_first_call_ms": reference_measurement.get("first_call_ms"),
        "reference_warmup_ms": reference_measurement.get("warmup_ms"),
        "reference_graph_capture_ms": reference_measurement.get("graph_capture_ms"),
        "reference_steady_state_ms": reference_measurement.get("steady_state_ms"),
        "reference_mean_ms": reference_statistics.get("mean_ms"),
        "reference_median_ms": reference_statistics.get("median_ms"),
        "reference_min_ms": reference_statistics.get("min_ms"),
        "reference_max_ms": reference_statistics.get("max_ms"),
        "reference_stddev_ms": reference_statistics.get("stddev_ms"),
        "reference_cv": reference_statistics.get("cv"),
        "reference_p50_ms": reference_statistics.get("p50_ms"),
        "reference_p90_ms": reference_statistics.get("p90_ms"),
        "reference_p95_ms": reference_statistics.get("p95_ms"),
        "reference_p99_ms": reference_statistics.get("p99_ms"),
        "reference_unstable": reference_statistics.get("unstable"),
        "candidate_mean_ms": candidate_statistics.get("mean_ms"),
        "candidate_median_ms": candidate_statistics.get("median_ms"),
        "candidate_min_ms": candidate_statistics.get("min_ms"),
        "candidate_max_ms": candidate_statistics.get("max_ms"),
        "candidate_p50_ms": candidate_statistics.get("p50_ms"),
        "candidate_p90_ms": candidate_statistics.get("p90_ms"),
        "candidate_p95_ms": candidate_statistics.get("p95_ms"),
        "candidate_p99_ms": candidate_statistics.get("p99_ms"),
        "candidate_stddev_ms": candidate_statistics.get("stddev_ms"),
        "candidate_cv": candidate_statistics.get("cv"),
        "candidate_unstable": candidate_statistics.get("unstable"),
        "instability_reason": candidate_statistics.get("instability_reason"),
        "speedup": raw_performance.get("speedup"),
        "slowdown_pct": raw_performance.get("slowdown_pct"),
        "latency_delta_ms": raw_performance.get("latency_delta_ms"),
        "performance_formal": gate.formal,
        "ranking_eligible": gate.ranking_eligible,
        "performance_gate_status": gate.status,
        "performance_gate_reasons": json.dumps(gate.reasons),
        "perf_on_correctness_fail": trusted_opt_in,
        "gate_max_slowdown_pct": job.resolved_config.performance_regression_threshold_pct,
        "gate_min_speedup": job.resolved_config.performance_min_speedup,
        "gate_max_candidate_median_ms": job.resolved_config.performance_max_candidate_median_ms,
        "gate_max_cv": job.resolved_config.performance_max_cv,
        "gate_max_memory_bytes": job.resolved_config.performance_max_memory_bytes,
        "gate_unsupported_policy": job.resolved_config.performance_unsupported_policy,
        "tflops": cost.get("tflops"),
        "effective_bandwidth_gbps": cost.get("effective_bandwidth_gbps"),
        "flops": cost.get("flops"),
        "estimated_bytes": cost.get("estimated_bytes"),
        "arithmetic_intensity": cost.get("arithmetic_intensity"),
        "throughput": cost.get("throughput"),
        "peak_memory_bytes": raw_performance.get("peak_memory_allocated_bytes"),
        "peak_reserved_memory_bytes": raw_performance.get("peak_memory_reserved_bytes"),
        "workspace_bytes": raw_performance.get("workspace_bytes"),
        "error_type": error_type,
        "error_message": error_message,
        "diagnostic_path": response.diagnostic_path,
        "stdout_path": "logs/stdout.log",
        "stderr_path": "logs/stderr.log",
    }
    projection, projection_mapping = projection_for_case(
        job.identity.operator_id, job.case
    )
    projection_rows: list[dict[str, object]] = []
    if projection_mapping is not None:
        symbols = job.case.symbols
        projection_status, projection_reason = _projection_status(
            correctness, performance, error_message, raw_performance.get("reason")
        )
        for role, statistics in (("reference", reference_statistics), ("candidate", candidate_statistics)):
            raw_per_call = statistics.get("median_ms") if projection_status == "measured" else None
            per_call = raw_per_call if isinstance(raw_per_call, (int, float)) and not isinstance(raw_per_call, bool) else None
            projection_rows.append({
                "run_id": job.identity.run_id, "evaluation_id": job.identity.evaluation_id,
                "result_id": job.result_id, "suite_id": plan.suite_id,
                "projection_id": projection.projection_id,
                "phase": str(symbols["phase"]), "quant_profile": str(symbols["quant_profile"]),
                "model_input": int(symbols["model_input"]), "raw_context": int(symbols["raw_context"]),
                "operator_id": job.identity.operator_id, "candidate_id": job.identity.candidate_id,
                "case_id": job.case.case_id, "adapter_id": projection_mapping.adapter_id,
                "display_name": projection_mapping.display_name, "backend": projection_mapping.backend,
                "kind": projection_mapping.kind, "legacy_shape": json.dumps(projection_mapping.shape),
                "instances": projection_mapping.instances,
                "implementation_role": role, "per_call_ms": per_call,
                "projected_model_ms": None if per_call is None else per_call * projection_mapping.instances,
                "status": projection_status, "reason": projection_reason,
            })
    # Per-output/projection details are durable before results.csv acts as the completion
    # marker consumed by resume.
    if output_rows:
        AtomicCsvTable(
            job.output_dir / CORRECTNESS_OUTPUTS_SCHEMA.filename,
            CORRECTNESS_OUTPUTS_SCHEMA,
        ).replace_partitions(output_rows, partition_fields=("result_id",))
    if sample_rows:
        AtomicCsvTable(
            job.output_dir / PERFORMANCE_SAMPLES_SCHEMA.filename,
            PERFORMANCE_SAMPLES_SCHEMA,
        ).replace_partitions(sample_rows, partition_fields=("result_id",))
    if projection_rows:
        AtomicCsvTable(
            job.output_dir / MODEL_PROJECTION_SCHEMA.filename,
            MODEL_PROJECTION_SCHEMA,
        ).replace_partitions(projection_rows, partition_fields=("result_id",))
    AtomicCsvTable(job.output_dir / RESULTS_SCHEMA.filename, RESULTS_SCHEMA).append(row)
    return overall is ResultStatus.PASSED, False


def execute_plan(
    plan: EvaluationPlan,
    snapshot: RegistrySnapshot,
    environment_snapshot: Mapping[str, object],
    *,
    output_root: Path,
    original_command: Sequence[str],
    resume: bool = False,
    fail_fast: bool = False,
    timeout_s: float | None = None,
    controller: WorkerController | None = None,
    lock_root: Path | None = None,
) -> RunOutcome:
    writer = ArtifactWriter(output_root)
    grouped: dict[object, list[EvaluationJob]] = {}
    for job in plan.jobs:
        grouped.setdefault(job.identity, []).append(job)
    paths: list[Path] = []
    completed: dict[object, frozenset[str]] = {}
    active: set[object] = set()
    for identity, jobs in grouped.items():
        paths.append(writer.evaluation_dir(identity))
        if resume:
            existing = writer.read_manifest(identity)
            state = writer.resume_state(existing)
            if existing.status is EvaluationState.INTERRUPTED:
                writer.update_status(identity, EvaluationState.RUNNING)
                active.add(identity)
            elif existing.status is EvaluationState.RUNNING:
                active.add(identity)
            elif existing.status is EvaluationState.COMPLETE:
                pass
            else:
                raise ResumeMismatchError(
                    f"evaluation status cannot be resumed: {existing.status.value}"
                )
        else:
            first = jobs[0]
            manifest = EvaluationManifest.create(
                identity=identity,
                original_command=original_command,
                resolved_config=first.resolved_config,
                reference_source_hash=first.reference.source_hash,
                candidate_source_hash=first.candidate.source_hash,
                environment_snapshot=environment_snapshot,
                environment_fingerprint=plan.environment_fingerprint,
                suite_id=plan.suite_id,
            )
            state = writer.initialize(manifest)
            writer.update_status(identity, EvaluationState.RUNNING)
            active.add(identity)
        completed[identity] = state.completed_result_ids

    worker_controller = controller or WorkerController(artifact_writer=writer)
    passed = failed = infrastructure = 0
    if resume:
        for identity in grouped:
            for existing_row in AtomicCsvTable(
                writer.evaluation_dir(identity) / RESULTS_SCHEMA.filename,
                RESULTS_SCHEMA,
            ).read_rows():
                if existing_row["status"] == ResultStatus.PASSED.value:
                    passed += 1
                else:
                    failed += 1
    infrastructure_identities: set[object] = set()
    interrupted = False
    stopped = False
    for job in plan.jobs:
        if stopped or job.result_id in completed[job.identity]:
            continue
        metadata = snapshot.operator_specs[job.identity.operator_id]
        operator_manifest = snapshot.operator_manifests[job.identity.operator_id]
        candidate_manifest = snapshot.candidate_manifests.get(
            (job.identity.operator_id, job.identity.candidate_id)
        )
        build = None if candidate_manifest is None else candidate_manifest.build
        limits = StageTimeouts(
            import_s=60,
            build_s=float(build.timeout_s if build is not None else 600),
            correctness_s=float(timeout_s or job.case.timeout_s or 300),
            performance_s=float(job.resolved_config.performance_timeout_s),
        )
        try:
            cuda_devices = _manifest_cuda_devices(operator_manifest.device_types)
            gpu_lock = None
            if cuda_devices and job.mode in {"all", "performance"}:
                identity = resolve_gpu_identity(0, allow_torch_fallback=False)
                gpu_lock = GpuLock(
                    (
                        Path(lock_root).resolve()
                        if lock_root is not None
                        else job.reference.root.parents[2] / ".runtime" / "locks"
                    ),
                    identity,
                    run_id=job.identity.run_id,
                    timeout_s=job.resolved_config.gpu_lock_timeout_s,
                ).acquire()
            try:
                response = worker_controller.run(
                    job,
                    spec_entrypoint=metadata.entrypoint,
                    build_argv=() if build is None else build.command,
                    cuda_devices=cuda_devices,
                    timeouts=limits,
                )
            finally:
                if gpu_lock is not None:
                    gpu_lock.release()
        except KeyboardInterrupt:
            interrupted = True
            stopped = True
            break
        except Exception as error:
            infrastructure += 1
            infrastructure_identities.add(job.identity)
            stopped = True
            atomic_write_text(
                job.output_dir / "diagnostics" / "engine-error.txt",
                f"{type(error).__name__}: {error}\n",
            )
            break
        if response.outcome is WorkerOutcome.INTERRUPTED:
            interrupted = True
            stopped = True
            break
        try:
            did_pass, infra = _append_result(
                job, snapshot, plan, response, environment_snapshot
            )
        except Exception as error:
            infrastructure += 1
            infrastructure_identities.add(job.identity)
            stopped = True
            atomic_write_text(
                job.output_dir / "diagnostics" / "engine-error.txt",
                f"{type(error).__name__}: {error}\n",
            )
            continue
        if infra:
            infrastructure += 1
            infrastructure_identities.add(job.identity)
            stopped = True
        elif did_pass:
            passed += 1
        else:
            failed += 1
            if fail_fast:
                stopped = True

    for identity, jobs in grouped.items():
        if identity not in active:
            continue
        current_ids = {
            row["result_id"]
            for row in AtomicCsvTable(
                writer.evaluation_dir(identity) / RESULTS_SCHEMA.filename,
                RESULTS_SCHEMA,
            ).read_rows()
        }
        expected = {job.result_id for job in jobs}
        if current_ids == expected:
            writer.update_status(identity, EvaluationState.COMPLETE)
        elif interrupted:
            writer.update_status(
                identity, EvaluationState.INTERRUPTED, terminal_reason="user_interrupt"
            )
        elif identity in infrastructure_identities:
            writer.update_status(
                identity, EvaluationState.FAILED, terminal_reason="infrastructure_failure"
            )
        elif infrastructure:
            writer.update_status(
                identity,
                EvaluationState.INTERRUPTED,
                terminal_reason="stopped_after_other_infrastructure_failure",
            )
        else:
            writer.update_status(
                identity, EvaluationState.FAILED, terminal_reason="fail_fast"
            )
    return RunOutcome(
        plan.run_id,
        tuple(paths),
        passed,
        failed,
        infrastructure,
        interrupted,
    )


__all__ = [
    "PERFORMANCE_SKIP_REASON",
    "RunOutcome",
    "build_dry_run_plan",
    "build_execution_plan",
    "build_resume_plan",
    "execute_plan",
]
