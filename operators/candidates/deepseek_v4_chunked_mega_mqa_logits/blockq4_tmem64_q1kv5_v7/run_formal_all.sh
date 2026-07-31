#!/usr/bin/env bash
set -euo pipefail

REPO=/home/gjy/data/agent4kernel/mega_mqa_logits/llm_flops
CANDIDATE="$REPO/operators/candidates/deepseek_v4_chunked_mega_mqa_logits/blockq4_tmem64_q1kv5_v7"
LOG="$CANDIDATE/formal_all.log"

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
    ./bench.sh run \
      --suite full \
      --mode all \
      --operator deepseek_v4_chunked_mega_mqa_logits \
      --candidate blockq4_tmem64_q1kv5_v7 \
      --fail-fast \
      --timeout-s 3600
"
