#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/gjy/data/agent4kernel/mega_mqa_logits/llm_flops
REFERENCE="$ROOT/operators/references/deepseek_v4_chunked_mega_mqa_logits"
ART="$ROOT/results/deepseek_v4_chunked_mega_mqa_logits/single_request_full_kv_reuse_baseline_20260731"
PY="$ROOT/.runtime/venv/bin/python"
M="${1:-4096}"
MODE="${2:-ncu}"
LOG="$ART/${MODE}_m${M}.log"

mkdir -p "$ART/source_snapshot" "$ART/nsys" "$ART/ncu"
exec > >(tee "$LOG") 2>&1

for i in 1 2 3 4; do
    pids="$(nvidia-smi -i 3 --query-compute-apps=pid --format=csv,noheader,nounits)"
    util="$(nvidia-smi -i 3 --query-gpu=utilization.gpu --format=csv,noheader,nounits)"
    printf 'GPU3 idle check %d: pids=%s util=%s\n' "$i" "${pids:-none}" "$util"
    test -z "$pids"
    test "$util" -eq 0
    sleep 1
done

flock -w7200 /tmp/mega-mqa-gpu3.lock bash -lc "
    set -euo pipefail
    source /home/gjy/data/agent4kernel/kda-pilot/.env.sh
    cd '$ROOT'
    export CUDA_VISIBLE_DEVICES=3
    cp '$REFERENCE/implementation.py' '$ART/source_snapshot/'
    cp '$REFERENCE/spec.py' '$ART/source_snapshot/'
    cp '$REFERENCE/README.md' '$ART/source_snapshot/'
    cp '$REFERENCE/profile_baseline.py' '$ART/source_snapshot/'

    unset PYTHONPATH
    export UV_CACHE_DIR='$ROOT/.runtime/cache/uv'
    export TORCH_EXTENSIONS_DIR='$ROOT/.runtime/cache/torch_extensions'
    export FLASHINFER_WORKSPACE_BASE='$ROOT/.runtime/cache/flashinfer'
    export XDG_CACHE_HOME='$ROOT/.runtime/cache/xdg'

    if [[ '$MODE' == ncu ]]; then
      exec /usr/local/cuda/bin/ncu \
        --set full \
        --target-processes all \
        --profile-from-start off \
        --replay-mode kernel \
        --force-overwrite \
        --export '$ART/ncu/m${M}_single_request_set_full' \
        '$PY' '$REFERENCE/profile_baseline.py' --m '$M' --capture
    fi

    if [[ '$MODE' == nsys ]]; then
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
        --output='$ART/nsys/m${M}_single_request' \
        '$PY' '$REFERENCE/profile_baseline.py' --m '$M' --capture
    fi

    echo 'unsupported mode: $MODE' >&2
    exit 2
"
