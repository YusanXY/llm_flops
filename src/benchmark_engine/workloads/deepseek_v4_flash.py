"""Small, controller-safe helpers for DeepSeek V4 Flash reference contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from benchmark_engine.correctness.models import (
    ComparisonResult,
    OutputBundle,
    OutputLeaf,
)


def sample_indices(torch, total: int, device, limit: int = 256):
    """Return deterministic, evenly spaced indices without synchronizing the GPU."""

    total = int(total)
    count = min(total, int(limit))
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if count == 1:
        return torch.zeros(1, dtype=torch.long, device=device)
    positions = torch.arange(count, dtype=torch.long, device=device)
    span = total - 1
    intervals = count - 1
    return (
        positions * (span // intervals)
        + positions * (span % intervals) // intervals
    )


def sample_tensor(tensor, *, limit: int = 256):
    """Create a bounded one-dimensional sample that remains a live tensor view."""

    torch = __import__("torch")
    flat = tensor.detach().reshape(-1)
    indices = sample_indices(torch, flat.numel(), flat.device, limit)
    return flat[indices]


def normalize_named_tensors(
    output,
    names: Sequence[str] = ("output",),
    *,
    limit: int = 256,
) -> OutputBundle:
    """Normalize one tensor or a tensor tuple into bounded output leaves."""

    tensors = output if isinstance(output, (tuple, list)) else (output,)
    if len(tensors) != len(names):
        raise ValueError(
            f"expected {len(names)} output tensors, received {len(tensors)}"
        )
    leaves = []
    for name, tensor in zip(names, tensors):
        sampled = sample_tensor(tensor, limit=limit)
        values = tuple(sampled.float().cpu().tolist())
        leaves.append(
            OutputLeaf(
                name,
                values,
                str(tensor.dtype),
                tuple(tensor.shape),
                tuple(tensor.stride()),
                "strided",
                str(tensor.device),
            )
        )
    return OutputBundle(tuple(leaves))


@dataclass(frozen=True)
class NumericPath:
    """Comparison rule for a normalized output/state leaf."""

    path: str
    atol: float
    rtol: float = 0.0
    oracle_path: str | None = None
    oracle_atol: float | None = None
    oracle_rtol: float | None = None


def _numeric_error(left, right, *, atol: float, rtol: float):
    if len(left) != len(right):
        return float("inf"), False
    maximum = 0.0
    passed = True
    for lhs, rhs in zip(left, right):
        lhs_f, rhs_f = float(lhs), float(rhs)
        error = abs(lhs_f - rhs_f)
        maximum = max(maximum, error)
        if error > atol + rtol * abs(rhs_f):
            passed = False
    return maximum, passed


class OracleStateComparator:
    """Compare reference/candidate outputs, semantic oracles, and live state."""

    def __init__(
        self,
        *,
        numeric_paths: Sequence[NumericPath],
        immutable_paths: Sequence[str] = (),
        require_oracle: bool = True,
        name: str = "deepseek_v4_oracle_state",
    ):
        self.numeric_paths = tuple(numeric_paths)
        self.immutable_paths = tuple(immutable_paths)
        self.require_oracle = bool(require_oracle)
        self.name = name

    def compare(self, reference, candidate, **_):
        failures = []
        metrics: dict[str, float] = {}
        reference_leaves = reference.by_path()
        candidate_leaves = candidate.by_path()

        for rule in self.numeric_paths:
            ref_leaf = reference_leaves.get(rule.path)
            cand_leaf = candidate_leaves.get(rule.path)
            if ref_leaf is None or cand_leaf is None:
                failures.append(
                    {"path": rule.path, "error": "missing reference/candidate leaf"}
                )
                continue
            maximum, passed = _numeric_error(
                ref_leaf.value,
                cand_leaf.value,
                atol=rule.atol,
                rtol=rule.rtol,
            )
            metrics[f"{rule.path}.reference_candidate_max_abs"] = maximum
            if not passed:
                failures.append(
                    {
                        "path": rule.path,
                        "error": "reference/candidate mismatch",
                        "max_abs_error": maximum,
                    }
                )

            if not self.require_oracle or rule.oracle_path is None:
                continue
            for role, leaves, leaf in (
                ("reference", reference_leaves, ref_leaf),
                ("candidate", candidate_leaves, cand_leaf),
            ):
                oracle = leaves.get(rule.oracle_path)
                if oracle is None:
                    failures.append(
                        {
                            "path": rule.oracle_path,
                            "role": role,
                            "error": "missing semantic oracle",
                        }
                    )
                    continue
                maximum, passed = _numeric_error(
                    leaf.value,
                    oracle.value,
                    atol=rule.atol if rule.oracle_atol is None else rule.oracle_atol,
                    rtol=rule.rtol if rule.oracle_rtol is None else rule.oracle_rtol,
                )
                metrics[f"{rule.path}.{role}_oracle_max_abs"] = maximum
                if not passed:
                    failures.append(
                        {
                            "path": rule.path,
                            "role": role,
                            "error": "semantic oracle mismatch",
                            "max_abs_error": maximum,
                        }
                    )

        for path in self.immutable_paths:
            left = reference_leaves.get(path)
            right = candidate_leaves.get(path)
            if left is None or right is None:
                failures.append({"path": path, "error": "missing immutable state"})
            elif left.contract() != right.contract() or left.value != right.value:
                failures.append({"path": path, "error": "immutable state changed"})

        return ComparisonResult(
            passed=not failures,
            comparator=self.name,
            metrics={"max_abs_error": metrics},
            diagnostics=tuple(failures[:16]),
            failed_path=None if not failures else failures[0].get("path"),
        )


def estimated_tensor_bytes(shape: Sequence[int], element_size: int) -> int:
    total = int(element_size)
    for dimension in shape:
        total *= int(dimension)
    return total


def clone_observed_state(state: Mapping[str, object]):
    """Clone tensor values in an observed-state mapping."""

    cloned = {}
    for key, value in state.items():
        clone = getattr(value, "clone", None)
        cloned[key] = clone() if callable(clone) else value
    return cloned
