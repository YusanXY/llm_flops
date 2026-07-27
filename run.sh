#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/.runtime/venv/bin/python"
ENV_FILE="${AGENT4KERNEL_ENV:-$HOME/.config/agent4kernel/env.sh}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: benchmark runtime is missing; run ./bootstrap.sh first" >&2
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
export UV_CACHE_DIR="$ROOT/.runtime/cache/uv"
export TORCH_EXTENSIONS_DIR="$ROOT/.runtime/cache/torch_extensions"
export FLASHINFER_WORKSPACE_BASE="$ROOT/.runtime/cache/flashinfer"
export XDG_CACHE_HOME="$ROOT/.runtime/cache/xdg"
export TRITON_CACHE_DIR="$ROOT/.runtime/cache/triton"
export AITER_CONFIG_DIR="$ROOT/.runtime/cache/aiter"
export AITER_META_DIR="$AITER_ROOT"
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=0

cd "$ROOT"
exec "$PYTHON" "$ROOT/benchmark_cli.py" "$@"
