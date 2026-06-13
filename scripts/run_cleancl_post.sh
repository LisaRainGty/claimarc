#!/usr/bin/env bash
set -euo pipefail

ROOT="${CLAIMARC_ROOT:-/mnt/gty/claimarc_active}"
PY="${CLAIMARC_PY:-/root/miniconda3/bin/python}"
cd "$ROOT"

export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUT_DIR="data/final/cleancl"
mkdir -p "$OUT_DIR"

while true; do
  n=$(grep -c '^RESULT ' data/final/cleancl_nohup.log 2>/dev/null || true)
  echo "[post] cleancl results=${n}/20"
  if [ "$n" -ge 20 ]; then
    break
  fi
  sleep 30
done

for tag in small_e1_c00 small_e1_c10 small_e1_c15 small_e3_c10; do
  "$PY" -m models.fusion_eval \
    --dataset data/final/dataset_verify_faithful.jsonl \
    --cm "$OUT_DIR/${tag}"_s0.pt "$OUT_DIR/${tag}"_s1.pt "$OUT_DIR/${tag}"_s2.pt "$OUT_DIR/${tag}"_s3.pt "$OUT_DIR/${tag}"_s4.pt \
    --out "$OUT_DIR/cmp_${tag}_5seed.json"
done

"$PY" -m models.cv_eval \
  --dataset data/final/dataset_verify_faithful.jsonl \
  --folds 5 --cm_seeds 0 1 2 \
  --baselines bge_lr \
  --encoder_name BAAI/bge-small-zh-v1.5 \
  --n_fusion 1 --lora_rank 8 \
  --warmup 1 --cl_epochs 2 \
  --bs 8 --accum 4 \
  --cl_c_min 0.10 --cl_neg_c_min 0.10 \
  --tmpdir "$OUT_DIR/cv_tmp_small_e3_c10" \
  --out "$OUT_DIR/cv_small_e3_c10.json"

echo "[post] done"
