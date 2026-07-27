"""Reference/candidate staged performance evaluation inside one worker."""

from __future__ import annotations

import gc
import math
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from benchmark_engine.correctness.inputs import assert_input_isolation, make_generator_context
from benchmark_engine.correctness.models import InputBundle
from benchmark_engine.correctness.evaluator import synchronize_cuda
from benchmark_engine.models import CaseSpec

from .cost_model import CostMetrics, evaluate_cost_model
from .statistics import SampleStatistics, compute_statistics
from .timers import RawSample, Timer, TimerConfig, TimerSelection, select_timer


@dataclass(frozen=True)
class PerformanceConfig:
    requested_timer: str = "auto"
    warmup: int = 5
    samples: int = 30
    inner_iterations: int = 20
    minimum_stable_samples: int = 5
    maximum_cv: float = 0.1

    def __post_init__(self) -> None:
        if self.requested_timer not in {"auto", "cuda_event", "cuda_graph", "wall_clock"}:
            raise ValueError(
                "requested_timer must be one of: auto, cuda_event, cuda_graph, wall_clock"
            )
        for name in ("warmup", "samples", "inner_iterations", "minimum_stable_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if name == "warmup":
                if value < 0:
                    raise ValueError("warmup must be non-negative")
            elif value <= 0:
                raise ValueError(f"{name} must be positive")
        if isinstance(self.maximum_cv, bool) or not isinstance(self.maximum_cv, (int, float)):
            raise TypeError("maximum_cv must be a number")
        if not math.isfinite(float(self.maximum_cv)) or self.maximum_cv < 0:
            raise ValueError("maximum_cv must be finite and non-negative")


@dataclass(frozen=True)
class ImplementationMeasurement:
    role: str
    selection: TimerSelection
    first_call_ms: float
    warmup_ms: float
    graph_capture_ms: float
    steady_state_ms: float
    samples: tuple[RawSample, ...]
    statistics: SampleStatistics

    def __post_init__(self) -> None:
        if self.role not in {"reference", "candidate"}:
            raise ValueError("role must be reference or candidate")

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "selection": self.selection.to_dict(),
            "first_call_ms": self.first_call_ms,
            "warmup_ms": self.warmup_ms,
            "graph_capture_ms": self.graph_capture_ms,
            "steady_state_ms": self.steady_state_ms,
            "samples": [sample.to_dict() for sample in self.samples],
            "statistics": self.statistics.to_dict(),
        }


@dataclass(frozen=True)
class PerformanceResult:
    status: str
    reference: ImplementationMeasurement
    candidate: ImplementationMeasurement
    cost: CostMetrics
    formal: bool = True
    speedup: float | None = None
    latency_delta_ms: float | None = None
    slowdown_pct: float | None = None
    peak_memory_allocated_bytes: int | None = None
    peak_memory_reserved_bytes: int | None = None
    workspace_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.status not in {"pass", "unstable"}:
            raise ValueError("performance status must be pass or unstable")

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "measurements": {
                "reference": self.reference.to_dict(),
                "candidate": self.candidate.to_dict(),
            },
            "cost": self.cost.to_dict(),
            "formal": self.formal,
            "speedup": self.speedup,
            "latency_delta_ms": self.latency_delta_ms,
            "slowdown_pct": self.slowdown_pct,
            "peak_memory_allocated_bytes": self.peak_memory_allocated_bytes,
            "peak_memory_reserved_bytes": self.peak_memory_reserved_bytes,
            "workspace_bytes": self.workspace_bytes,
        }


@dataclass
class _PreparedImplementation:
    role: str
    timer: Timer
    invoke: Callable[[], object]
    sampling: TimerConfig
    first_call_ms: float
    warmup_ms: float
    graph_capture_ms: float


@dataclass
class PreparedPerformance:
    """Runtime-only boundary between preparation and steady sampling."""

    reference: _PreparedImplementation
    candidate: _PreparedImplementation
    spec: object
    case: CaseSpec
    config: PerformanceConfig
    track_cuda_memory: bool = False


class PerformanceEvaluator:
    """Prepare independently, then fairly interleave steady-state measurements."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.perf_counter,
        synchronizer: Callable[[object, InputBundle], None] = synchronize_cuda,
        timer_factory: Callable[[str, Callable[[], None]], Timer] | None = None,
        reset_peak_memory: Callable[[], bool] | None = None,
        read_peak_memory: Callable[[], tuple[int | None, int | None]] | None = None,
    ) -> None:
        self._clock = clock
        self._synchronizer = synchronizer
        self._timer_factory = timer_factory or (
            lambda requested, sync: select_timer(
                requested, clock=self._clock, synchronizer=sync
            )
        )
        self._reset_peak_memory = reset_peak_memory or _reset_cuda_peak_memory
        self._read_peak_memory = read_peak_memory or _read_cuda_peak_memory

    def _prepare_implementation(
        self,
        role: str,
        fn: Callable[..., object],
        inputs: InputBundle,
        config: PerformanceConfig,
    ) -> _PreparedImplementation:
        # Keep every asynchronous output in the current inner-iteration group
        # alive until the timer closes that group.  Dropping the previous
        # output at each Python call can release storage still used by a GPU
        # kernel (AITER fused MoE is one concrete example).  The bounded deque
        # drops the previous group's outputs only after its end event has
        # synchronized.
        latest: deque[object] = deque(maxlen=config.inner_iterations)

        def invoke() -> object:
            output = fn(*inputs.args, **inputs.kwargs)
            latest.append(output)
            return output

        def synchronize() -> None:
            self._synchronizer(latest[-1] if latest else None, inputs)

        sampling = TimerConfig(config.samples, config.inner_iterations)
        timer = self._timer_factory(config.requested_timer, synchronize)

        synchronize()
        started = self._clock()
        invoke()
        synchronize()
        first_call_ms = (self._clock() - started) * 1000.0

        synchronize()
        started = self._clock()
        for _ in range(config.warmup):
            invoke()
        synchronize()
        warmup_ms = (self._clock() - started) * 1000.0

        graph_capture_ms = timer.prepare(invoke, sampling)
        for name, value in (
            ("first_call_ms", first_call_ms),
            ("warmup_ms", warmup_ms),
            ("graph_capture_ms", graph_capture_ms),
        ):
            if not math.isfinite(value) or value < 0:
                raise RuntimeError(f"{name} is non-finite or negative")
        return _PreparedImplementation(
            role=role,
            timer=timer,
            invoke=invoke,
            sampling=sampling,
            first_call_ms=first_call_ms,
            warmup_ms=warmup_ms,
            graph_capture_ms=graph_capture_ms,
        )

    def _measurement(
        self, prepared: _PreparedImplementation, config: PerformanceConfig,
        samples: tuple[RawSample, ...],
    ) -> ImplementationMeasurement:
        steady_state_ms = sum(sample.elapsed_ms for sample in samples)
        statistics = compute_statistics(
            (sample.per_call_ms for sample in samples),
            minimum_stable_samples=config.minimum_stable_samples,
            maximum_cv=config.maximum_cv,
        )
        if not math.isfinite(steady_state_ms) or steady_state_ms < 0:
            raise RuntimeError("steady_state_ms is non-finite or negative")
        return ImplementationMeasurement(
            role=prepared.role,
            selection=prepared.timer.selection,
            first_call_ms=prepared.first_call_ms,
            warmup_ms=prepared.warmup_ms,
            graph_capture_ms=prepared.graph_capture_ms,
            steady_state_ms=steady_state_ms,
            samples=samples,
            statistics=statistics,
        )

    def prepare(
        self,
        *,
        spec: object,
        reference: Callable[..., object],
        candidate: Callable[..., object],
        case: CaseSpec,
        config: PerformanceConfig,
        cuda_devices: tuple[str, ...] = (),
    ) -> PreparedPerformance:
        context = make_generator_context(case.seed, cuda_devices)
        canonical = spec.make_inputs(case, context)
        if not isinstance(canonical, InputBundle):
            raise TypeError("OperatorSpec.make_inputs() must return correctness.InputBundle")
        reference_inputs = spec.clone_inputs(canonical)
        candidate_inputs = spec.clone_inputs(canonical)
        if not isinstance(reference_inputs, InputBundle) or not isinstance(candidate_inputs, InputBundle):
            raise TypeError("OperatorSpec.clone_inputs() must return correctness.InputBundle")
        assert_input_isolation(canonical, reference_inputs)
        assert_input_isolation(canonical, candidate_inputs)
        assert_input_isolation(reference_inputs, candidate_inputs)

        reference_prepared = self._prepare_implementation(
            "reference", reference, reference_inputs, config
        )
        candidate_prepared = self._prepare_implementation(
            "candidate", candidate, candidate_inputs, config
        )
        return PreparedPerformance(
            reference_prepared, candidate_prepared, spec, case, config,
            bool(cuda_devices),
        )

    def sample(self, prepared: PreparedPerformance) -> PerformanceResult:
        buckets: dict[str, list[RawSample]] = {"reference": [], "candidate": []}
        order_index = 0
        by_role = {"reference": prepared.reference, "candidate": prepared.candidate}
        schedule = ("reference", "candidate", "candidate", "reference")
        peak_allocated: int | None = None
        peak_reserved: int | None = None
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            # Cyclic collection can pause Python between asynchronous launches.
            # GPU events then include an artificial idle gap even though no
            # kernel work changed. Collect before the window and restore GC
            # immediately after all steady-state samples.
            gc.collect()
            gc.disable()
        try:
            while any(len(values) < prepared.config.samples for values in buckets.values()):
                for role in schedule:
                    bucket = buckets[role]
                    if len(bucket) >= prepared.config.samples:
                        continue
                    implementation = by_role[role]
                    memory_tracking = role == "candidate" and prepared.track_cuda_memory and self._reset_peak_memory()
                    one = implementation.timer.sample(
                        implementation.invoke,
                        TimerConfig(1, implementation.sampling.inner_iterations),
                    )[0]
                    if memory_tracking:
                        allocated, reserved = self._read_peak_memory()
                        if allocated is not None:
                            peak_allocated = allocated if peak_allocated is None else max(peak_allocated, allocated)
                        if reserved is not None:
                            peak_reserved = reserved if peak_reserved is None else max(peak_reserved, reserved)
                    bucket.append(RawSample(
                        len(bucket), one.inner_iterations, one.elapsed_ms,
                        one.per_call_ms, order_index,
                    ))
                    order_index += 1
        finally:
            if gc_was_enabled:
                gc.enable()
        reference_measurement = self._measurement(
            prepared.reference, prepared.config, tuple(buckets["reference"])
        )
        candidate_measurement = self._measurement(
            prepared.candidate, prepared.config, tuple(buckets["candidate"])
        )
        model_value = prepared.spec.cost_model(prepared.case)
        cost = evaluate_cost_model(
            model_value, candidate_measurement.statistics.median_ms
        )
        workspace_bytes = None
        workspace_hook = getattr(prepared.spec, "workspace_bytes", None)
        if callable(workspace_hook):
            value = workspace_hook(prepared.case)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise TypeError("OperatorSpec.workspace_bytes(case) must return a non-negative integer or None")
                workspace_bytes = value
        unstable = (
            reference_measurement.statistics.unstable
            or candidate_measurement.statistics.unstable
        )
        timers_match = reference_measurement.selection.effective_timer == candidate_measurement.selection.effective_timer
        reference_median = reference_measurement.statistics.median_ms
        candidate_median = candidate_measurement.statistics.median_ms
        speedup = latency_delta_ms = slowdown_pct = None
        if reference_median > 0 and candidate_median > 0:
            values = (reference_median / candidate_median,
                      candidate_median - reference_median,
                      (candidate_median / reference_median - 1.0) * 100.0)
            if all(math.isfinite(value) for value in values):
                speedup, latency_delta_ms, slowdown_pct = values
        return PerformanceResult(
            "unstable" if unstable else "pass",
            reference_measurement,
            candidate_measurement,
            cost,
            formal=not unstable and timers_match and speedup is not None,
            speedup=speedup,
            latency_delta_ms=latency_delta_ms,
            slowdown_pct=slowdown_pct,
            peak_memory_allocated_bytes=peak_allocated,
            peak_memory_reserved_bytes=peak_reserved,
            workspace_bytes=workspace_bytes,
        )

    def evaluate(
        self,
        *,
        spec: object,
        reference: Callable[..., object],
        candidate: Callable[..., object],
        case: CaseSpec,
        config: PerformanceConfig,
        cuda_devices: tuple[str, ...] = (),
    ) -> PerformanceResult:
        prepared = self.prepare(
            spec=spec,
            reference=reference,
            candidate=candidate,
            case=case,
            config=config,
            cuda_devices=cuda_devices,
        )
        return self.sample(prepared)


def _reset_cuda_peak_memory() -> bool:
    try:
        import torch
        if not torch.cuda.is_available(): return False
        torch.cuda.reset_peak_memory_stats()
        return True
    except (ImportError, RuntimeError):
        return False


def _read_cuda_peak_memory() -> tuple[int | None, int | None]:
    try:
        import torch
        if not torch.cuda.is_available(): return None, None
        return int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())
    except (ImportError, RuntimeError):
        return None, None


__all__ = [
    "ImplementationMeasurement",
    "PerformanceConfig",
    "PerformanceEvaluator",
    "PerformanceResult",
    "PreparedPerformance",
]
