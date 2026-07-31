#!/usr/bin/env bash
set -euo pipefail

REPO=/home/gjy/data/agent4kernel/mega_mqa_logits/llm_flops
CANDIDATE="$REPO/operators/candidates/deepseek_v4_mega_mqa_logits/baseline_control__v2_single_request"
REFERENCE="$REPO/operators/references/deepseek_v4_mega_mqa_logits"
OUT="$REPO/results/deepseek_v4_mega_mqa_logits/baseline_control__v2_single_request/nsys/m4096_contract_v2_timeline"
PY="$REPO/.runtime/venv/bin/python"
NAME=baseline_control__v2_single_request__m4096__timeline

mkdir -p "$OUT/source_snapshot"

for _ in 1 2 3 4; do
    test -z "$(nvidia-smi -i 3 --query-compute-apps=pid --format=csv,noheader,nounits)"
    nvidia-smi -i 3 --query-gpu=utilization.gpu --format=csv,noheader,nounits | grep -qx 0
    sleep 1
done

flock -w7200 /tmp/mega-mqa-gpu3.lock bash -lc "
    set -euo pipefail
    source /home/gjy/data/agent4kernel/kda-pilot/.env.sh
    export CUDA_VISIBLE_DEVICES=3
    export TORCH_CUDA_ARCH_LIST=10.0a
    cd '$REPO'

    /usr/local/cuda/bin/nsys profile \
      --trace=cuda,nvtx,osrt \
      --sample=none \
      --cpuctxsw=none \
      --capture-range=cudaProfilerApi \
      --capture-range-end=stop \
      --force-overwrite=true \
      --output='$OUT/$NAME' \
      '$PY' '$CANDIDATE/profile_timeline.py' \
        --m 4096 \
        --reference-dir '$REFERENCE' \
        --candidate '$CANDIDATE/implementation.py' \
        --metadata-out '$OUT/profile_metadata.json' \
        --nvtx-range 'baseline_control__v2_single_request/formal_m4096/operator' \
      2>&1 | tee '$OUT/nsys_stdout_stderr.log'

    /usr/local/cuda/bin/nsys export \
      --type=sqlite \
      --force-overwrite=true \
      --output='$OUT/$NAME.sqlite' \
      '$OUT/$NAME.nsys-rep' \
      > '$OUT/sqlite_export.stdout' 2> '$OUT/sqlite_export.stderr'

    for report in cuda_gpu_kern_sum cuda_gpu_trace cuda_kern_exec_sum cuda_kern_exec_trace cuda_api_sum cuda_api_trace nvtx_gpu_proj_sum nvtx_gpu_proj_trace; do
      /usr/local/cuda/bin/nsys stats \
        --report \"\$report\" \
        --format csv \
        --output '$OUT/'\"\$report\" \
        --force-overwrite true \
        '$OUT/$NAME.nsys-rep' \
        > '$OUT/'\"\$report\"'.stdout' 2> '$OUT/'\"\$report\"'.stderr' || true
    done

    cp '$REFERENCE/spec.py' '$REFERENCE/implementation.py' \
       '$REFERENCE/README.md' '$CANDIDATE/profile_timeline.py' \
       '$OUT/source_snapshot/'
    cp '$CANDIDATE/implementation.py' \
       '$OUT/source_snapshot/candidate_implementation.py'
    sha256sum '$OUT/$NAME.nsys-rep' '$OUT/$NAME.sqlite' \
      '$OUT/profile_metadata.json' '$OUT/nsys_stdout_stderr.log' \
      > '$OUT/report.sha256'
"
