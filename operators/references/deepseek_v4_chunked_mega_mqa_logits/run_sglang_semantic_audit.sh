#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/gjy/data/agent4kernel/mega_mqa_logits/llm_flops
REFERENCE="$ROOT/operators/references/deepseek_v4_chunked_mega_mqa_logits"
ART="$ROOT/results/deepseek_v4_chunked_mega_mqa_logits/semantic_audit"
PY="$ROOT/.runtime/venv/bin/python"

cd "$ROOT"
mkdir -p "$ART"
unset PYTHONPATH
export UV_CACHE_DIR="$ROOT/.runtime/cache/uv"
export TORCH_EXTENSIONS_DIR="$ROOT/.runtime/cache/torch_extensions"
export FLASHINFER_WORKSPACE_BASE="$ROOT/.runtime/cache/flashinfer"
export XDG_CACHE_HOME="$ROOT/.runtime/cache/xdg"

set +e
flock -w 7200 /tmp/mega-mqa-gpu3.lock env CUDA_VISIBLE_DEVICES=3 \
  "$PY" "$REFERENCE/sglang_semantic_audit.py" >"$ART/sglang_crosscheck.log" 2>&1
rc=$?
set -e
echo "$rc" >"$ART/sglang_crosscheck.exit"
exit "$rc"
