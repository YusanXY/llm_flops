#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/gjy/data/agent4kernel/mega_mqa_logits/llm_flops

before=$(nvidia-smi -i 3 --query-compute-apps=pid --format=csv,noheader | sed '/^$/d')
sleep 4
after=$(nvidia-smi -i 3 --query-compute-apps=pid --format=csv,noheader | sed '/^$/d')
if [[ -n "${before}${after}" ]]; then
  echo "GPU3_BUSY"
  nvidia-smi -i 3
  exit 42
fi

cd "${ROOT}"
source /home/gjy/data/agent4kernel/kda-pilot/.env.sh
flock -w7200 /tmp/mega-mqa-gpu3.lock \
  env CUDA_VISIBLE_DEVICES=3 \
  .runtime/venv/bin/python \
  operators/candidates/deepseek_v4_mega_mqa_logits/baseline_control__v2_single_request/probe_single_request_abi.py \
  --m "${1:-1024}" \
  --reference-dir operators/references/deepseek_v4_mega_mqa_logits
