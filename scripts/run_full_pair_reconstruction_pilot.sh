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
REVIEWS="${REVIEWS:-data/final/repaired_v1/full_pair_reconstruction_llm_v1_20260614.jsonl}"
LLM_REPORT="${LLM_REPORT:-data/final/repaired_v1/full_pair_reconstruction_llm_v1_20260614.report.json}"
AUDIT_REPORT="${AUDIT_REPORT:-data/final/repaired_v1/full_pair_reconstruction_llm_audit_v1_20260614.report.json}"
AUDIT_FLAGS="${AUDIT_FLAGS:-data/final/repaired_v1/full_pair_reconstruction_llm_audit_flags_v1_20260614.jsonl}"

echo "[claimarc] full-pair reconstruction pilot"
echo "[claimarc] queue=$QUEUE priority=$PRIORITY limit=$LIMIT concurrency=$CONCURRENCY model=$MODEL"

python -m data_quality.llm_full_pair_reconstruct_v1 \
  --queue "$QUEUE" \
  --out "$REVIEWS" \
  --report "$LLM_REPORT" \
  --priority "$PRIORITY" \
  --limit "$LIMIT" \
  --concurrency "$CONCURRENCY" \
  --model "$MODEL"

python -m data_quality.audit_full_pair_llm_reviews_v1 \
  --queue "$QUEUE" \
  --reviews "$REVIEWS" \
  --out "$AUDIT_REPORT" \
  --flagged "$AUDIT_FLAGS"

python - "$AUDIT_REPORT" <<'PY'
import json
import sys

report_path = sys.argv[1]
with open(report_path, "r", encoding="utf-8") as f:
    report = json.load(f)

missing = int(report.get("missing_reviews", 0) or 0)
high = int(report.get("high_flag_rows", 0) or 0)
if missing or high:
    print(
        f"[claimarc] audit blocked promotion: missing_reviews={missing}, "
        f"high_flag_rows={high}. Review flagged rows before promotion.",
        file=sys.stderr,
    )
    sys.exit(3)
PY

python -m data_quality.build_full_pair_promoted_dataset_v1 \
  --queue "$QUEUE" \
  --reviews "$REVIEWS"

echo "[claimarc] pilot complete"
