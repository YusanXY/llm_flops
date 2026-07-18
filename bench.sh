#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="${BENCHMARK_RUNTIME_ROOT:-$ROOT/.runtime}"
CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$RUNTIME_ROOT/cache}"
VENV="$RUNTIME_ROOT/venv"
ENV_FILE="${AGENT4KERNEL_ENV:-$HOME/.config/agent4kernel/env.sh}"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "ERROR: benchmark runtime is missing at $VENV; run ./bootstrap.sh or set BENCHMARK_RUNTIME_ROOT" >&2
  exit 2
fi
if [[ ! -r "$ENV_FILE" ]]; then
  echo "ERROR: MI300X environment file is not readable: $ENV_FILE" >&2
  exit 2
fi

unset PYTHONPATH
# shellcheck source=/dev/null
source "$ENV_FILE"
export PYTHONPATH="$ROOT/src:$SGLANG_ROOT/python:$AITER_ROOT"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export FLASHINFER_WORKSPACE_BASE="$CACHE_ROOT/flashinfer"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export AITER_CONFIG_DIR="$CACHE_ROOT/aiter"
export AITER_META_DIR="$AITER_ROOT"
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=0

mkdir -p "$UV_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" \
  "$FLASHINFER_WORKSPACE_BASE" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" \
  "$AITER_CONFIG_DIR"

cd "$ROOT"
exec "$VENV/bin/python" -m benchmark_engine "$@"
