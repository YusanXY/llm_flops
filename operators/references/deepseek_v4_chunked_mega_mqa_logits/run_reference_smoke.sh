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
flock -w 7200 /tmp/mega-mqa-gpu3.lock bash -lc "
  export CUDA_VISIBLE_DEVICES=3
  rc=0
  for m in 1024 2048 4096; do
    \"$PY\" \"$REFERENCE/reference_smoke.py\" --m \"\$m\" \
      >\"$ART/m\${m}.log\" 2>&1
    case_rc=\$?
    echo \"\$case_rc\" >\"$ART/m\${m}.exit\"
    if [[ \$case_rc -ne 0 ]]; then
      rc=\$case_rc
      break
    fi
  done
  exit \$rc
"
rc=$?
set -e
exit "$rc"
