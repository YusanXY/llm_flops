#!/usr/bin/env bash
set -euo pipefail

REPO=/home/gjy/data/agent4kernel/mega_mqa_logits/llm_flops
CANDIDATE="$REPO/operators/candidates/deepseek_v4_mega_mqa_logits/baseline_control__v2_single_request"
REFERENCE="$REPO/operators/references/deepseek_v4_mega_mqa_logits"
OUT="$REPO/results/deepseek_v4_mega_mqa_logits/baseline_control__v2_single_request/ncu/m4096_contract_v2_full"
PY="$REPO/.runtime/venv/bin/python"
NAME=baseline_control__v2_single_request__m4096__set_full

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

    /opt/nvidia/nsight-compute/2026.1.1/ncu \
      --set full \
      --section PmSampling \
      --section PmSampling_WarpStates \
      --target-processes all \
      --replay-mode kernel \
      --profile-from-start off \
      --force-overwrite \
      --export '$OUT/$NAME' \
      '$PY' '$CANDIDATE/profile_full.py' \
        --m 4096 \
        --reference-dir '$REFERENCE' \
        --candidate '$CANDIDATE/implementation.py' \
        --metadata-out '$OUT/profile_metadata.json' \
        --nvtx-range 'baseline_control__v2_single_request/formal_m4096/operator' \
      2>&1 | tee '$OUT/ncu_stdout_stderr.log'

    /opt/nvidia/nsight-compute/2026.1.1/ncu --import '$OUT/$NAME.ncu-rep' \
      --page details --csv > '$OUT/details.csv' 2> '$OUT/details_export.stderr'
    /opt/nvidia/nsight-compute/2026.1.1/ncu --import '$OUT/$NAME.ncu-rep' \
      --page raw --csv > '$OUT/raw_metrics.csv' 2> '$OUT/raw_export.stderr'

    cp '$REFERENCE/spec.py' '$REFERENCE/implementation.py' \
       '$REFERENCE/README.md' '$REFERENCE/CUDA_PROVENANCE.md' \
       '$CANDIDATE/profile_full.py' '$OUT/source_snapshot/'
    cp '$CANDIDATE/implementation.py' \
       '$OUT/source_snapshot/candidate_implementation.py'
    sha256sum '$OUT/$NAME.ncu-rep' '$OUT/details.csv' \
      '$OUT/raw_metrics.csv' '$OUT/profile_metadata.json' \
      '$OUT/ncu_stdout_stderr.log' > '$OUT/report.sha256'
"
