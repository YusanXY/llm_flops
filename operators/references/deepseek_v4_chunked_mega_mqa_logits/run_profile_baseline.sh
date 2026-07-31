#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/gjy/data/agent4kernel/mega_mqa_logits/llm_flops
REFERENCE="$ROOT/operators/references/deepseek_v4_chunked_mega_mqa_logits"
ART="$ROOT/results/deepseek_v4_chunked_mega_mqa_logits/full_kv_reuse_baseline"
PY="$ROOT/.runtime/venv/bin/python"
M="${1:-4096}"
MODE="${2:-nsys}"

cd "$ROOT"
mkdir -p "$ART/source_snapshot" "$ART/nsys" "$ART/ncu"
cp "$REFERENCE/implementation.py" "$ART/source_snapshot/"
cp "$REFERENCE/spec.py" "$ART/source_snapshot/"
cp "$REFERENCE/README.md" "$ART/source_snapshot/"
cp "$REFERENCE/profile_baseline.py" "$ART/source_snapshot/"

unset PYTHONPATH
export UV_CACHE_DIR="$ROOT/.runtime/cache/uv"
export TORCH_EXTENSIONS_DIR="$ROOT/.runtime/cache/torch_extensions"
export FLASHINFER_WORKSPACE_BASE="$ROOT/.runtime/cache/flashinfer"
export XDG_CACHE_HOME="$ROOT/.runtime/cache/xdg"

if [[ "$MODE" == "nsys" ]]; then
  exec nsys profile \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt,cublas \
    --sample=none \
    --cpuctxsw=none \
    --cuda-memory-usage=true \
    --gpu-metrics-devices=3 \
    --gpu-metrics-frequency=10000 \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --output="$ART/nsys/m${M}_full_kv_reuse" \
    "$PY" "$REFERENCE/profile_baseline.py" --m "$M" --capture
fi

if [[ "$MODE" == "ncu" ]]; then
  exec /usr/local/cuda/bin/ncu \
    --set full \
    --target-processes all \
    --profile-from-start off \
    --replay-mode kernel \
    --force-overwrite \
    --export "$ART/ncu/m${M}_full_kv_reuse_set_full" \
    "$PY" "$REFERENCE/profile_baseline.py" --m "$M" --capture
fi

echo "unsupported mode: $MODE" >&2
exit 2
