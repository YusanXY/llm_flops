#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="${BENCHMARK_RUNTIME_ROOT:-$ROOT/.runtime}"
CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$RUNTIME/cache}"
VENV="$RUNTIME/venv"
LOG="$RUNTIME/logs/bootstrap.log"
LOCK="$ROOT/requirements/benchmark-lock.json"
MARKER="$RUNTIME/installed.lock"
ENV_FILE="${AGENT4KERNEL_ENV:-$HOME/.config/agent4kernel/env.sh}"

if [[ ! -r "$ENV_FILE" ]]; then
  echo "ERROR: MI300X environment file is not readable: $ENV_FILE" >&2
  exit 2
fi

unset PYTHONPATH
# shellcheck source=/dev/null
source "$ENV_FILE"

for name in GPU_VENV SGLANG_ROOT AITER_ROOT; do
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: $name is not defined by $ENV_FILE" >&2
    exit 2
  fi
done

PYTHON="$GPU_VENV/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: ROCm Python is unavailable: $PYTHON" >&2
  exit 2
fi

export PYTHONPATH="$ROOT/src:$SGLANG_ROOT/python:$AITER_ROOT"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export FLASHINFER_WORKSPACE_BASE="$CACHE_ROOT/flashinfer"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export AITER_CONFIG_DIR="$CACHE_ROOT/aiter"
export AITER_META_DIR="$AITER_ROOT"
# SGLang's fused clamp epilogue is CUDA-only. AITER does not consume this
# setting, while the independent HIP Triton oracle must explicitly disable it.
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=0

mkdir -p "$RUNTIME/logs" "$UV_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" \
  "$FLASHINFER_WORKSPACE_BASE" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" \
  "$AITER_CONFIG_DIR"
touch "$LOG"
exec > >(tee -a "$LOG") 2>&1

if [[ -e "$VENV" && ! -L "$VENV" ]]; then
  echo "ERROR: refusing to replace non-symlink runtime: $VENV" >&2
  exit 2
fi
ln -sfn "$GPU_VENV" "$VENV"

LOCK_HASH="$("$PYTHON" - "$LOCK" <<'PY'
import hashlib
import pathlib
import sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"

cd "$ROOT"
"$PYTHON" -m benchmark_environment --check
"$PYTHON" -c 'import benchmark_engine'
FINGERPRINT="$("$PYTHON" -m benchmark_environment --json | "$PYTHON" -c \
  'import json,sys; print(json.load(sys.stdin)["fingerprint"])')"
printf '%s %s\n' "$LOCK_HASH" "$FINGERPRINT" > "$MARKER"
echo "MI300X benchmark runtime ready: $VENV -> $GPU_VENV"
