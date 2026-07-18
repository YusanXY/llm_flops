#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="${BENCHMARK_RUNTIME_ROOT:-$ROOT/.runtime}"
CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$ROOT/.runtime/cache}"
VENV="$RUNTIME_ROOT/venv"

export PYTHONPATH="$ROOT/src"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export FLASHINFER_WORKSPACE_BASE="$CACHE_ROOT/flashinfer"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

mkdir -p "$UV_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" \
  "$FLASHINFER_WORKSPACE_BASE" "$XDG_CACHE_HOME"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "ERROR: benchmark runtime is missing at $VENV; run ./bootstrap.sh or set BENCHMARK_RUNTIME_ROOT" >&2
  exit 2
fi

cd "$ROOT"
exec "$VENV/bin/python" -m benchmark_engine "$@"
