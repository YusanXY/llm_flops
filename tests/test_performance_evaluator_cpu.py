import gc
import unittest
import weakref

from benchmark_engine.correctness import InputBundle, clone_input_bundle
from benchmark_engine.models import CaseSpec
from benchmark_engine.performance import PerformanceConfig, PerformanceEvaluator
from benchmark_engine.performance.timers import RawSample, TimerSelection


class StepClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        current = self.value
        self.value += 0.001
        return current


class Spec:
    def make_inputs(self, case, context):
        return InputBundle(args=([1.0, 2.0], [3.0, 4.0]))

    def clone_inputs(self, inputs):
        return clone_input_bundle(inputs)

    def cost_model(self, case):
        return {"flops": 2, "bytes": 32}


class PerformanceEvaluatorCpuTests(unittest.TestCase):
    def test_cyclic_gc_is_disabled_only_during_steady_sampling(self):
        states = []
        original_state = gc.isenabled()
        gc.enable()
        try:
            evaluator = PerformanceEvaluator(
                clock=StepClock(), synchronizer=lambda output, inputs: None
            )

            def operator(left, right):
                states.append(gc.isenabled())
                return left

            evaluator.evaluate(
                spec=Spec(),
                reference=operator,
                candidate=operator,
                case=CaseSpec("case", {}, 0, frozenset()),
                config=PerformanceConfig(
                    requested_timer="wall_clock",
                    warmup=0,
                    samples=1,
                    inner_iterations=1,
                    minimum_stable_samples=1,
                    maximum_cv=1.0,
                ),
            )
            self.assertEqual(states, [True, True, False, False])
            self.assertTrue(gc.isenabled())
        finally:
            if original_state:
                gc.enable()
            else:
                gc.disable()

    def test_inner_iteration_outputs_remain_alive_until_sample_boundary(self):
        class Output:
            pass

        class LifetimeTimer:
            selection = TimerSelection("cuda_event", "cuda_event")

            def prepare(self, fn, config):
                return 0.0

            def sample(self, fn, config):
                references = []
                for _ in range(config.inner_iterations):
                    output = fn()
                    references.append(weakref.ref(output))
                    del output
                if not all(reference() is not None for reference in references):
                    raise AssertionError("an inner-iteration output was released early")
                return (RawSample(0, config.inner_iterations, 1.0, 0.25),)

        evaluator = PerformanceEvaluator(
            synchronizer=lambda output, inputs: None,
            timer_factory=lambda requested, sync: LifetimeTimer(),
        )
        result = evaluator.evaluate(
            spec=Spec(),
            reference=lambda a, b: Output(),
            candidate=lambda a, b: Output(),
            case=CaseSpec("case", {}, 0, frozenset()),
            config=PerformanceConfig(
                requested_timer="cuda_event",
                warmup=0,
                samples=1,
                inner_iterations=4,
                minimum_stable_samples=1,
                maximum_cv=1.0,
            ),
        )
        self.assertEqual(result.status, "pass")

    def test_candidate_peak_memory_and_optional_workspace_are_outside_timing(self):
        class WorkspaceSpec(Spec):
            def workspace_bytes(self, case): return 4096
        resets = []
        reads = iter(((100, 200), (150, 250)))
        evaluator = PerformanceEvaluator(
            clock=StepClock(), synchronizer=lambda output, inputs: None,
            reset_peak_memory=lambda: resets.append(True) or True,
            read_peak_memory=lambda: next(reads),
        )
        result = evaluator.evaluate(
            spec=WorkspaceSpec(), reference=lambda a, b: a,
            candidate=lambda a, b: a, case=CaseSpec("case", {}, 0, frozenset()),
            config=PerformanceConfig(requested_timer="wall_clock", warmup=0,
                                     samples=2, inner_iterations=1,
                                     minimum_stable_samples=2, maximum_cv=1),
            cuda_devices=("cuda:0",),
        )
        self.assertEqual(len(resets), 2)
        self.assertEqual(result.peak_memory_allocated_bytes, 150)
        self.assertEqual(result.peak_memory_reserved_bytes, 250)
        self.assertEqual(result.workspace_bytes, 4096)

    def test_unknown_memory_is_none_not_zero(self):
        result = PerformanceEvaluator(clock=StepClock(), synchronizer=lambda o, i: None).evaluate(
            spec=Spec(), reference=lambda a, b: a, candidate=lambda a, b: a,
            case=CaseSpec("case", {}, 0, frozenset()),
            config=PerformanceConfig(requested_timer="wall_clock", warmup=0,
                                     samples=1, inner_iterations=1),
        )
        self.assertIsNone(result.peak_memory_allocated_bytes)
        self.assertIsNone(result.peak_memory_reserved_bytes)
    def test_first_warmup_and_sampling_are_separate_for_both_roles(self):
        calls = {"reference": 0, "candidate": 0}

        def reference(left, right):
            calls["reference"] += 1
            return [a + b for a, b in zip(left, right)]

        def candidate(left, right):
            calls["candidate"] += 1
            return [a + b for a, b in zip(left, right)]

        clock = StepClock()
        result = PerformanceEvaluator(
            clock=clock, synchronizer=lambda output, inputs: None
        ).evaluate(
            spec=Spec(),
            reference=reference,
            candidate=candidate,
            case=CaseSpec("case", {}, 0, frozenset()),
            config=PerformanceConfig(
                requested_timer="wall_clock",
                warmup=1,
                samples=2,
                inner_iterations=2,
                minimum_stable_samples=2,
                maximum_cv=1.0,
            ),
        )
        # Per role: one first call, one warmup, and 2*2 steady calls.
        self.assertEqual(calls, {"reference": 6, "candidate": 6})
        for measurement in (result.reference, result.candidate):
            self.assertAlmostEqual(measurement.first_call_ms, 1.0)
            self.assertAlmostEqual(measurement.warmup_ms, 1.0)
            self.assertEqual(len(measurement.samples), 2)
            self.assertEqual(
                [round(sample.per_call_ms, 6) for sample in measurement.samples],
                [0.5, 0.5],
            )
            self.assertAlmostEqual(measurement.steady_state_ms, 2.0)
        self.assertEqual(result.status, "pass")
        self.assertTrue(result.cost.available)

    def test_prepare_does_not_create_steady_state_samples(self):
        calls = 0

        def operator(left, right):
            nonlocal calls
            calls += 1
            return left

        evaluator = PerformanceEvaluator(
            clock=StepClock(), synchronizer=lambda output, inputs: None
        )
        prepared = evaluator.prepare(
            spec=Spec(),
            reference=operator,
            candidate=operator,
            case=CaseSpec("case", {}, 0, frozenset()),
            config=PerformanceConfig(
                requested_timer="wall_clock", warmup=0, samples=3, inner_iterations=2
            ),
        )
        self.assertEqual(calls, 2)
        evaluator.sample(prepared)
        self.assertEqual(calls, 14)


if __name__ == "__main__":
    unittest.main()
