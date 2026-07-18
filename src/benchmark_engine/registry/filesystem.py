"""Filesystem-backed, import-free operator registry."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from benchmark_engine.ids import (
    IdentifierError,
    validate_candidate_id,
    validate_operator_id,
)
from benchmark_engine.models import ImplementationSpec

from .base import (
    CandidateManifest,
    OperatorManifest,
    OperatorSpecMetadata,
    RegistryIssue,
    RegistrySnapshot,
)
from .validation import (
    ManifestValidationError,
    SourceHashError,
    compute_source_hash,
    parse_candidate_manifest,
    parse_operator_manifest,
    validate_entrypoint,
)


def _issue(code: str, path: Path, field: str, message: str) -> RegistryIssue:
    return RegistryIssue(code, path, field, message)


def _directories(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.iterdir() if path.is_dir() or path.is_symlink()),
        key=lambda path: (path.name.casefold(), path.name),
    )


def _case_collisions(paths: list[Path], kind: str) -> list[RegistryIssue]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        groups[path.name.casefold()].append(path)
    issues: list[RegistryIssue] = []
    for colliding in groups.values():
        if len(colliding) < 2:
            continue
        names = ", ".join(sorted(path.name for path in colliding))
        for path in colliding:
            issues.append(
                _issue(
                    "registry.case_collision",
                    path,
                    kind,
                    f"case-insensitive identifier collision: {names}",
                )
            )
    return issues


class FilesystemRegistry:
    """Discover references and candidates rooted at a repository directory."""

    def __init__(self, repository_root: Path) -> None:
        self.repository_root = Path(repository_root).resolve()
        self.references_root = self.repository_root / "operators" / "references"
        self.candidates_root = self.repository_root / "operators" / "candidates"
        self._kda_task_layout = False
        self._snapshot: RegistrySnapshot | None = None

    @classmethod
    def for_kda_task(cls, task_root: Path) -> "FilesystemRegistry":
        """Discover one KDA task without copying sources into this repository.

        ``baseline/`` is the reference root and every immediate directory below
        ``solution/`` is an independent candidate version.  Generated artifacts
        are deliberately outside the registry and belong under ``bench/``.
        """

        registry = cls(task_root)
        registry.references_root = registry.repository_root / "baseline"
        registry.candidates_root = registry.repository_root / "solution"
        registry._kda_task_layout = True
        return registry

    def discover(self) -> RegistrySnapshot:
        references: dict[str, ImplementationSpec] = {}
        candidates: dict[str, list[ImplementationSpec]] = defaultdict(list)
        operator_manifests: dict[str, OperatorManifest] = {}
        candidate_manifests: dict[tuple[str, str], CandidateManifest] = {}
        operator_specs: dict[str, OperatorSpecMetadata] = {}
        issues: list[RegistryIssue] = []
        discovered_operator_ids: set[str] = set()
        declared_references: dict[str, Path] = {}

        task_operator_id: str | None = None
        if self._kda_task_layout:
            reference_paths = (
                [self.references_root]
                if self.references_root.exists() or self.references_root.is_symlink()
                else []
            )
            candidate_operator_paths = (
                [self.candidates_root]
                if self.candidates_root.exists() or self.candidates_root.is_symlink()
                else []
            )
            if not reference_paths:
                issues.append(
                    _issue(
                        "kda.missing_baseline",
                        self.references_root,
                        "baseline",
                        "KDA task must contain a baseline directory",
                    )
                )
            if not candidate_operator_paths:
                issues.append(
                    _issue(
                        "kda.missing_solution",
                        self.candidates_root,
                        "solution",
                        "KDA task must contain a solution directory",
                    )
                )
            for filename in ("implementation.py", "candidate.yaml"):
                flat_path = self.candidates_root / filename
                if flat_path.exists() or flat_path.is_symlink():
                    issues.append(
                        _issue(
                            "kda.flat_solution",
                            flat_path,
                            "solution",
                            "candidate files must live under solution/<candidate_id>/",
                        )
                    )
        else:
            reference_paths = _directories(self.references_root)
            candidate_operator_paths = _directories(self.candidates_root)
        issues.extend(_case_collisions(reference_paths, "operator_id"))
        issues.extend(_case_collisions(candidate_operator_paths, "operator_id"))

        for reference_root in reference_paths:
            operator_id = reference_root.name
            before = len(issues)
            if reference_root.is_symlink():
                issues.append(
                    _issue(
                        "registry.symlink",
                        reference_root,
                        "operator_id",
                        "reference directory must not be a symlink",
                    )
                )
                continue
            if not self._kda_task_layout:
                discovered_operator_ids.add(operator_id)
                try:
                    validate_operator_id(operator_id)
                except IdentifierError as error:
                    issues.append(
                        _issue("id.operator", reference_root, "operator_id", str(error))
                    )
                    continue

            manifest_path = reference_root / "operator.yaml"
            if manifest_path.is_symlink():
                issues.append(
                    _issue(
                        "registry.symlink",
                        manifest_path,
                        "$",
                        "operator manifest must not be a symlink",
                    )
                )
                continue
            try:
                manifest = parse_operator_manifest(manifest_path)
            except ManifestValidationError as error:
                issues.append(error.as_issue())
                continue
            if self._kda_task_layout:
                operator_id = manifest.operator_id
                task_operator_id = operator_id
                discovered_operator_ids.add(operator_id)
                try:
                    validate_operator_id(operator_id)
                except IdentifierError as error:
                    issues.append(
                        _issue("id.operator", manifest_path, "operator_id", str(error))
                    )
                    continue
            previous_reference = declared_references.get(manifest.operator_id)
            if previous_reference is not None:
                issues.append(
                    _issue(
                        "registry.duplicate_reference",
                        manifest_path,
                        "operator_id",
                        f"operator_id {manifest.operator_id!r} is also declared by "
                        f"{previous_reference}",
                    )
                )
            else:
                declared_references[manifest.operator_id] = manifest_path
            if not self._kda_task_layout and manifest.operator_id != operator_id:
                issues.append(
                    _issue(
                        "operator.id_mismatch",
                        manifest_path,
                        "operator_id",
                        f"manifest {manifest.operator_id!r} does not match directory "
                        f"{operator_id!r}",
                    )
                )
            for field, entrypoint in (
                ("reference_entrypoint", manifest.reference_entrypoint),
                ("spec_entrypoint", manifest.spec_entrypoint),
                ("cost_model", manifest.cost_model),
            ):
                try:
                    validate_entrypoint(reference_root, entrypoint, manifest_path, field)
                except ManifestValidationError as error:
                    issues.append(error.as_issue())
            try:
                source_hash = compute_source_hash(reference_root)
            except SourceHashError as error:
                issues.append(
                    _issue(error.code, error.path, "$", str(error))
                )
                continue
            if len(issues) != before:
                continue

            implementation = ImplementationSpec(
                operator_id=operator_id,
                implementation_id="reference",
                role="reference",
                root=reference_root,
                entrypoint=manifest.reference_entrypoint,
                source_hash=source_hash,
                manifest_version=manifest.schema_version,
            )
            if operator_id in references:
                issues.append(
                    _issue(
                        "registry.duplicate_reference",
                        reference_root,
                        "operator_id",
                        f"duplicate reference for {operator_id!r}",
                    )
                )
                continue
            references[operator_id] = implementation
            operator_manifests[operator_id] = manifest
            operator_specs[operator_id] = OperatorSpecMetadata(
                operator_id=operator_id,
                root=reference_root,
                entrypoint=manifest.spec_entrypoint,
                source_hash=source_hash,
                manifest=manifest,
            )

        for operator_root in candidate_operator_paths:
            operator_id = task_operator_id or operator_root.name
            discovered_operator_ids.add(operator_id)
            if operator_root.is_symlink():
                issues.append(
                    _issue(
                        "registry.symlink",
                        operator_root,
                        "operator_id",
                        (
                            "solution directory must not be a symlink"
                            if self._kda_task_layout
                            else "candidate operator directory must not be a symlink"
                        ),
                    )
                )
                continue
            try:
                validate_operator_id(operator_id)
            except IdentifierError as error:
                issues.append(
                    _issue("id.operator", operator_root, "operator_id", str(error))
                )
                continue
            candidate_paths = _directories(operator_root)
            issues.extend(_case_collisions(candidate_paths, "candidate_id"))
            if operator_id not in references:
                issues.append(
                    _issue(
                        "registry.orphan_candidate",
                        operator_root,
                        "operator_id",
                        f"candidate directory has no valid reference for {operator_id!r}",
                    )
                )

            seen: set[str] = set()
            declared_candidates: dict[str, Path] = {}
            for candidate_root in candidate_paths:
                candidate_id = candidate_root.name
                before = len(issues)
                if candidate_root.is_symlink():
                    issues.append(
                        _issue(
                            "registry.symlink",
                            candidate_root,
                            "candidate_id",
                            "candidate directory must not be a symlink",
                        )
                    )
                    continue
                try:
                    validate_candidate_id(candidate_id)
                except IdentifierError as error:
                    issues.append(
                        _issue(
                            "id.candidate", candidate_root, "candidate_id", str(error)
                        )
                    )
                    continue
                if candidate_id in seen:
                    issues.append(
                        _issue(
                            "registry.duplicate_candidate",
                            candidate_root,
                            "candidate_id",
                            f"duplicate candidate {candidate_id!r}",
                        )
                    )
                    continue
                seen.add(candidate_id)

                manifest_path = candidate_root / "candidate.yaml"
                if manifest_path.is_symlink():
                    issues.append(
                        _issue(
                            "registry.symlink",
                            manifest_path,
                            "$",
                            "candidate manifest must not be a symlink",
                        )
                    )
                    continue
                if manifest_path.exists():
                    try:
                        candidate_manifest = parse_candidate_manifest(manifest_path)
                    except ManifestValidationError as error:
                        issues.append(error.as_issue())
                        continue
                    if candidate_manifest.candidate_id != candidate_id:
                        issues.append(
                            _issue(
                                "candidate.id_mismatch",
                                manifest_path,
                                "candidate_id",
                                f"manifest {candidate_manifest.candidate_id!r} does not "
                                f"match directory {candidate_id!r}",
                            )
                        )
                    if (
                        candidate_manifest.operator_id is not None
                        and candidate_manifest.operator_id != operator_id
                    ):
                        issues.append(
                            _issue(
                                "candidate.operator_mismatch",
                                manifest_path,
                                "operator_id",
                                f"manifest {candidate_manifest.operator_id!r} does not "
                                f"match directory {operator_id!r}",
                            )
                        )
                else:
                    candidate_manifest = CandidateManifest(
                        schema_version=1,
                        candidate_id=candidate_id,
                    )
                previous_candidate = declared_candidates.get(
                    candidate_manifest.candidate_id
                )
                if previous_candidate is not None:
                    issues.append(
                        _issue(
                            "registry.duplicate_candidate",
                            manifest_path if manifest_path.exists() else candidate_root,
                            "candidate_id",
                            f"candidate_id {candidate_manifest.candidate_id!r} is also "
                            f"declared by {previous_candidate}",
                        )
                    )
                else:
                    declared_candidates[candidate_manifest.candidate_id] = (
                        manifest_path if manifest_path.exists() else candidate_root
                    )

                try:
                    validate_entrypoint(
                        candidate_root,
                        candidate_manifest.entrypoint,
                        manifest_path,
                        "entrypoint",
                    )
                except ManifestValidationError as error:
                    issues.append(error.as_issue())
                try:
                    source_hash = compute_source_hash(candidate_root)
                except SourceHashError as error:
                    issues.append(_issue(error.code, error.path, "$", str(error)))
                    continue
                if len(issues) != before or operator_id not in references:
                    continue

                implementation = ImplementationSpec(
                    operator_id=operator_id,
                    implementation_id=candidate_id,
                    role="candidate",
                    root=candidate_root,
                    entrypoint=candidate_manifest.entrypoint,
                    source_hash=source_hash,
                    manifest_version=candidate_manifest.schema_version,
                )
                candidates[operator_id].append(implementation)
                candidate_manifests[(operator_id, candidate_id)] = candidate_manifest

        sorted_candidates = {
            operator_id: tuple(
                sorted(values, key=lambda candidate: candidate.implementation_id)
            )
            for operator_id, values in sorted(candidates.items())
        }
        for operator_id in references:
            sorted_candidates.setdefault(operator_id, ())
        snapshot = RegistrySnapshot(
            repository_root=self.repository_root,
            references=dict(sorted(references.items())),
            candidates=dict(sorted(sorted_candidates.items())),
            operator_manifests=dict(sorted(operator_manifests.items())),
            candidate_manifests=dict(sorted(candidate_manifests.items())),
            operator_specs=dict(sorted(operator_specs.items())),
            discovered_operator_ids=tuple(
                sorted(discovered_operator_ids, key=lambda value: (value.casefold(), value))
            ),
            issues=tuple(
                sorted(
                    issues,
                    key=lambda item: (
                        item.path.as_posix().casefold(),
                        item.path.as_posix(),
                        item.code,
                        item.field,
                        item.message,
                    ),
                )
            ),
        )
        self._snapshot = snapshot
        return snapshot

    def _current(self) -> RegistrySnapshot:
        return self._snapshot if self._snapshot is not None else self.discover()

    def get_reference(self, operator_id: str) -> ImplementationSpec:
        return self._current().references[operator_id]

    def get_candidates(self, operator_id: str) -> tuple[ImplementationSpec, ...]:
        return self._current().candidates.get(operator_id, ())

    def get_operator_spec(self, operator_id: str) -> OperatorSpecMetadata:
        """Return spec location metadata without importing ``spec.py``."""

        return self._current().operator_specs[operator_id]

    def validate(self) -> tuple[RegistryIssue, ...]:
        return self.discover().issues
