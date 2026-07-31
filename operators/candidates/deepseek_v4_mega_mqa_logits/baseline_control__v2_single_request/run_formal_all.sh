#!/usr/bin/env bash
set -euo pipefail

REPO=/home/gjy/data/agent4kernel/mega_mqa_logits/llm_flops
CANDIDATE=baseline_control__v2_single_request
CANDIDATE_DIR="$REPO/operators/candidates/deepseek_v4_mega_mqa_logits/$CANDIDATE"
LOG="$CANDIDATE_DIR/formal_all.log"

exec > >(tee "$LOG") 2>&1

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

    for m in 1024 2048 4096; do
        ./bench.sh run \
          --suite full --mode all \
          --operator deepseek_v4_mega_mqa_logits \
          --candidate '$CANDIDATE' \
          --case formal__mega_mqa_logits__m\${m}__ctx65536 \
          --timer cuda_event --warmup 4 --samples 30 \
          --inner-iterations 1 --max-cv 0.10
    done
"
