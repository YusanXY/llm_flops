"""Validate and fingerprint the MI300X/ROCm benchmark runtime."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_LOCK = ROOT / "requirements" / "benchmark-lock.json"


def load_lock(path: Path = DEFAULT_LOCK) -> dict[str, Any]:
    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    if lock.get("schema_version") != 2:
        raise ValueError(f"unsupported schema_version: {lock.get('schema_version')}")
    return lock


def _has_symbol(name: str) -> bool:
    parts = name.split(".")
    module = None
    remainder: list[str] = []
    for split_at in range(len(parts), 0, -1):
        candidate = ".".join(parts[:split_at])
        try:
            module = importlib.import_module(candidate)
            remainder = parts[split_at:]
            break
        except ModuleNotFoundError as error:
            if error.name != candidate and not candidate.startswith(f"{error.name}."):
                return False
        except Exception:
            return False
    if module is None:
        return False
    value: Any = module
    for attribute in remainder:
        if not hasattr(value, attribute):
            return False
        value = getattr(value, attribute)
    return True


def _run_git(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", "-C", str(root), *args),
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.decode("utf-8", errors="replace").strip()


def _source_state(source: dict[str, Any]) -> dict[str, object]:
    root_env = source["root_env"]
    root_value = os.environ.get(root_env)
    if not root_value:
        return {
            "commit": None,
            "dirty": None,
            "diff_sha256": None,
            "root_env": root_env,
        }
    root = Path(root_value).expanduser().resolve()
    commit = _run_git(root, "rev-parse", "HEAD")
    status = _run_git(root, "status", "--porcelain", "--untracked-files=no")
    try:
        diff = subprocess.run(
            ("git", "-C", str(root), "diff", "--binary"),
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        diff_sha256 = None
    else:
        diff_sha256 = (
            hashlib.sha256(diff.stdout).hexdigest()
            if diff.returncode == 0
            else None
        )
    submodules: dict[str, str | None] = {}
    for relative in source.get("submodules", {}):
        submodule_root = (root / relative).resolve()
        top_level = _run_git(submodule_root, "rev-parse", "--show-toplevel")
        if top_level is None or Path(top_level).resolve() != submodule_root:
            submodules[relative] = None
        else:
            submodules[relative] = _run_git(submodule_root, "rev-parse", "HEAD")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "diff_sha256": diff_sha256,
        "root_env": root_env,
        "submodules": submodules,
    }


def collect_environment(
    lock: dict[str, Any], include_cuda: bool = True
) -> dict[str, Any]:
    packages: dict[str, dict[str, str | None]] = {}
    import_paths: dict[str, str | None] = {}
    for distribution, package in lock["packages"].items():
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = None
        module_name = package["module"]
        try:
            module = importlib.import_module(module_name)
            import_path = getattr(module, "__file__", None)
        except Exception:
            import_path = None
        packages[distribution] = {"version": version, "module": module_name}
        import_paths[module_name] = import_path

    observed: dict[str, Any] = {
        "python": platform.python_version(),
        "accelerator": {
            "backend": None,
            "runtime": None,
            "arch": None,
        },
        # Retained for compatibility with pre-ROCm result readers.
        "cuda": None,
        "hip": None,
        "gpu": {
            "available": False,
            "count": 0,
            "capability": None,
            "name": None,
            "arch": None,
        },
        "packages": packages,
        "source": {
            name: _source_state(source)
            for name, source in lock.get("source", {}).items()
        },
        "symbols": {
            name: _has_symbol(name) for name in lock.get("required_symbols", ())
        },
        "import_paths": import_paths,
    }
    if include_cuda:
        try:
            import torch

            hip = getattr(getattr(torch, "version", None), "hip", None)
            cuda = getattr(getattr(torch, "version", None), "cuda", None)
            available = bool(torch.cuda.is_available())
            observed["cuda"] = cuda
            observed["hip"] = hip
            observed["accelerator"]["backend"] = "rocm" if hip else "cuda" if cuda else None
            observed["accelerator"]["runtime"] = hip or cuda
            observed["gpu"]["available"] = available
            observed["gpu"]["count"] = int(torch.cuda.device_count()) if available else 0
            if available:
                properties = torch.cuda.get_device_properties(0)
                arch = getattr(properties, "gcnArchName", None)
                observed["gpu"]["capability"] = list(
                    torch.cuda.get_device_capability(0)
                )
                observed["gpu"]["name"] = torch.cuda.get_device_name(0)
                observed["gpu"]["arch"] = arch
                observed["accelerator"]["arch"] = arch
        except Exception:
            pass
    return observed


def validate_environment(lock: dict[str, Any], observed: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not observed.get("python", "").startswith(f"{lock['python']}."):
        errors.append(
            f"Python mismatch: expected {lock['python']}.x, observed {observed.get('python')}"
        )

    expected_accelerator = lock["accelerator"]
    accelerator = observed.get("accelerator", {})
    if accelerator.get("backend") != expected_accelerator["backend"]:
        errors.append(
            "accelerator backend mismatch: expected "
            f"{expected_accelerator['backend']}, observed {accelerator.get('backend')}"
        )
    runtime = str(accelerator.get("runtime") or "")
    if not runtime.startswith(expected_accelerator["runtime_prefix"]):
        errors.append(
            "accelerator runtime mismatch: expected "
            f"{expected_accelerator['runtime_prefix']}*, observed {runtime or None}"
        )

    expected_gpu = lock["gpu"]
    observed_gpu = observed.get("gpu", {})
    arch = str(observed_gpu.get("arch") or "")
    if not arch.startswith(expected_gpu["arch_prefix"]):
        errors.append(
            f"GPU arch mismatch: expected {expected_gpu['arch_prefix']}*, "
            f"observed {arch or None}"
        )
    if expected_gpu["name_contains"] not in (observed_gpu.get("name") or ""):
        errors.append(
            f"GPU name mismatch: expected *{expected_gpu['name_contains']}*, "
            f"observed {observed_gpu.get('name')}"
        )
    if observed_gpu.get("count") != expected_gpu["count"]:
        errors.append(
            f"GPU count mismatch: expected {expected_gpu['count']}, "
            f"observed {observed_gpu.get('count')}"
        )

    for distribution, expected in lock["packages"].items():
        actual = observed.get("packages", {}).get(distribution, {}).get("version")
        if expected["version"] is not None and actual != expected["version"]:
            errors.append(
                f"package {distribution} mismatch: expected {expected['version']}, "
                f"observed {actual}"
            )

    for name, expected in lock.get("source", {}).items():
        actual = observed.get("source", {}).get(name, {})
        for field in ("commit", "diff_sha256"):
            if actual.get(field) != expected[field]:
                errors.append(
                    f"source {name} {field} mismatch: expected {expected[field]}, "
                    f"observed {actual.get(field)}"
                )
        if bool(actual.get("dirty")) != bool(expected["dirty"]):
            errors.append(
                f"source {name} dirty mismatch: expected {expected['dirty']}, "
                f"observed {actual.get('dirty')}"
            )
        for path, commit in expected.get("submodules", {}).items():
            actual_commit = actual.get("submodules", {}).get(path)
            if actual_commit != commit:
                errors.append(
                    f"source {name} submodule {path} mismatch: "
                    f"expected {commit}, observed {actual_commit}"
                )

    for symbol in lock.get("required_symbols", ()):
        if not observed.get("symbols", {}).get(symbol, False):
            errors.append(f"required symbol unavailable: {symbol}")
    return errors


def environment_fingerprint(observed: dict[str, Any]) -> str:
    stable = copy.deepcopy(
        {key: value for key, value in observed.items() if key != "import_paths"}
    )
    encoded = json.dumps(
        stable,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    lock = load_lock(args.lock)
    observed = collect_environment(lock)
    errors = validate_environment(lock, observed)
    report = {
        "fingerprint": environment_fingerprint(observed),
        "valid": not errors,
        "errors": errors,
        "environment": observed,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        accelerator = observed["accelerator"]
        gpu = observed["gpu"]
        print(f"Benchmark environment: {report['fingerprint']}")
        print(
            f"Python {observed['python']}  "
            f"{str(accelerator.get('backend') or 'accelerator').upper()} "
            f"{accelerator.get('runtime')}"
        )
        print(
            f"GPU {gpu.get('name')}  arch={gpu.get('arch')}  count={gpu.get('count')}"
        )
        for error in errors:
            print(f"ERROR: {error}")
    return 1 if args.check and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
