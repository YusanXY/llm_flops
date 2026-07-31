#!/usr/bin/env bash
set -euo pipefail

for i in 1 2 3 4; do
    pids="$(nvidia-smi -i 3 --query-compute-apps=pid --format=csv,noheader,nounits)"
    util="$(nvidia-smi -i 3 --query-gpu=utilization.gpu --format=csv,noheader,nounits)"
    printf 'GPU3 idle check %d: pids=%s util=%s\n' "$i" "${pids:-none}" "$util"
    test -z "$pids"
    test "$util" -eq 0
    sleep 1
done

flock -w7200 /tmp/mega-mqa-gpu3.lock bash -lc '
    set -euo pipefail
    source /home/gjy/data/agent4kernel/kda-pilot/.env.sh
    cd /home/gjy/data/agent4kernel/mega_mqa_logits/llm_flops
    export CUDA_VISIBLE_DEVICES=3
    .runtime/venv/bin/python \
      operators/candidates/deepseek_v4_chunked_mega_mqa_logits/cluster2_q8_2smkv_dualq4_q1kv5_v13/inspect_schedule.py
'
