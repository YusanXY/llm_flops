#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: kda-bench.sh TASK_ROOT CANDIDATE_ID [bench run options]

Validate and benchmark one KDA-Pilot solution version in place.  The task must
contain baseline/, solution/<candidate_id>/, and bench/.  Results are written
under TASK_ROOT/bench/<operator_id>/<candidate_id>/<evaluation_id>/.

Example:
  CUDA_VISIBLE_DEVICES=0 ../../../llm_flops/kda-bench.sh . my_optimized_v1
EOF
}

if [[ $# -lt 2 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  [[ $# -ge 1 ]] && exit 0
  exit 2
fi

TASK_ROOT="$(cd -- "$1" && pwd)"
CANDIDATE_ID="$2"
shift 2

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-$TASK_ROOT/.cache/llm-flops}"

"$ROOT/bench.sh" validate --task-root "$TASK_ROOT"
exec "$ROOT/bench.sh" run \
  --task-root "$TASK_ROOT" \
  --suite kda_task \
  --candidate "$CANDIDATE_ID" \
  "$@"
