#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="${BENCHMARK_RUNTIME_ROOT:-$ROOT/.runtime}"
CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$RUNTIME/cache}"
VENV="$RUNTIME/venv"
LOG="$RUNTIME/logs/bootstrap.log"
LOCK="$ROOT/requirements/benchmark-lock.json"
MARKER="$RUNTIME/installed.lock"
PYTHON="${PYTHON:-/usr/bin/python3.12}"

export PYTHONPATH="$ROOT/src"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export FLASHINFER_WORKSPACE_BASE="$CACHE_ROOT/flashinfer"

mkdir -p "$RUNTIME/logs" "$UV_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" \
  "$FLASHINFER_WORKSPACE_BASE"
touch "$LOG"
exec > >(tee -a "$LOG") 2>&1

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python 3.12 is required at $PYTHON" >&2
  exit 2
fi

UV="${UV:-$(command -v uv || true)}"
if [[ -z "$UV" ]]; then
  echo "ERROR: uv is required; install uv or set UV=/path/to/uv" >&2
  exit 2
fi

LOCK_HASH="$($PYTHON - "$LOCK" <<'PY'
import hashlib
import pathlib
import sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"

if [[ -x "$VENV/bin/python" ]] \
  && (cd "$ROOT" && "$VENV/bin/python" -m benchmark_environment --check) \
  && (cd "$ROOT" && "$VENV/bin/python" -c 'import benchmark_engine'); then
  if [[ ! -f "$MARKER" || "$(<"$MARKER")" != "$LOCK_HASH" ]]; then
    printf '%s\n' "$LOCK_HASH" > "$MARKER"
  fi
  echo "Benchmark runtime already satisfies lock $LOCK_HASH"
  exit 0
fi

rm -f "$MARKER"
echo "Creating benchmark runtime under $RUNTIME"
"$UV" venv --clear --python "$PYTHON" "$VENV"

mapfile -t PACKAGE_SPECS < <("$PYTHON" - "$LOCK" <<'PY'
import json
import sys
lock = json.load(open(sys.argv[1]))
for name, package in sorted(lock["packages"].items()):
    if name != "sglang":
        print(f"{name}=={package['version']}")
PY
)

SGLANG_SPEC="$($PYTHON - "$LOCK" <<'PY'
import json
import sys
source = json.load(open(sys.argv[1]))["source"]["sglang"]
print(
    "sglang @ git+"
    f"{source['url']}@{source['commit']}"
    f"#subdirectory={source['subdirectory']}"
)
PY
)"

echo "Installing locked runtime dependencies"
"$UV" pip install --python "$VENV/bin/python" \
  "${PACKAGE_SPECS[@]}" "$SGLANG_SPEC"

echo "Installing benchmark engine"
VIRTUAL_ENV="$VENV" "$UV" pip install --no-deps -e "$ROOT"

cd "$ROOT"
"$VENV/bin/python" -m benchmark_environment --check
"$VENV/bin/python" -c 'import benchmark_engine'
printf '%s\n' "$LOCK_HASH" > "$MARKER"
echo "Benchmark runtime ready: $VENV"
