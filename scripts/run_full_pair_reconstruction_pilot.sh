#!/usr/bin/env bash
set -euo pipefail

ROOT="${CLAIMARC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

if [[ -f env.sh ]]; then
  # shellcheck disable=SC1091
  source env.sh >/dev/null
fi

if [[ -z "${MATPOOL_API_KEY:-}" ]]; then
  echo "[claimarc] MATPOOL_API_KEY is not set; aborting without running LLM/VLM calls." >&2
  exit 2
fi

export PYTHONPATH="${PYTHONPATH:-$ROOT/src}"

PRIORITY="${1:-P0}"
LIMIT="${2:-20}"
CONCURRENCY="${3:-2}"
MODEL="${4:-Qwen3-VL-Plus}"
QUEUE="${5:-data/final/repaired_v1/full_pair_llm_pilot_queue_v1_20260614.jsonl}"

echo "[claimarc] full-pair reconstruction pilot"
echo "[claimarc] queue=$QUEUE priority=$PRIORITY limit=$LIMIT concurrency=$CONCURRENCY model=$MODEL"

python -m data_quality.llm_full_pair_reconstruct_v1 \
  --queue "$QUEUE" \
  --priority "$PRIORITY" \
  --limit "$LIMIT" \
  --concurrency "$CONCURRENCY" \
  --model "$MODEL"

python -m data_quality.build_full_pair_promoted_dataset_v1

echo "[claimarc] pilot complete"
