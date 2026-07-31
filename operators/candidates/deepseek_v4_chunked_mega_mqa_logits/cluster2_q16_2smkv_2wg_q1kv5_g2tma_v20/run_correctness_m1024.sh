#!/usr/bin/env bash
set -euo pipefail

REPO=/home/gjy/data/agent4kernel/mega_mqa_logits/llm_flops
CANDIDATE="$REPO/operators/candidates/deepseek_v4_chunked_mega_mqa_logits/cluster2_q16_2smkv_2wg_q1kv5_g2tma_v20"
LOG="$CANDIDATE/correctness_m1024.log"

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
    cd '$REPO'
    export CUDA_VISIBLE_DEVICES=3
    export TORCH_CUDA_ARCH_LIST=10.0a
    export MEGA_V20_VERBOSE_BUILD=1
    ./bench.sh run \
      --suite full \
      --mode correctness \
      --operator deepseek_v4_chunked_mega_mqa_logits \
      --candidate cluster2_q16_2smkv_2wg_q1kv5_g2tma_v20 \
      --case formal__full_kv_reuse_mqa__m1024__ctx65536 \
      --seed 401 \
      --fail-fast \
      --timeout-s 3600
"
