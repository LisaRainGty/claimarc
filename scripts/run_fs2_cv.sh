#!/usr/bin/env bash
set -euo pipefail

ROOT="${CLAIMARC_ROOT:-/mnt/gty/claimarc_active}"
PY="${CLAIMARC_PY:-/root/miniconda3/bin/python}"
cd "$ROOT"

export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUT="data/final/cleancl"
mkdir -p "$OUT"

COMMON=(
  --folds 5
  --fold_seed 2
  --cm_seeds 0
  --baselines bge_lr
  --encoder_name BAAI/bge-small-zh-v1.5
  --n_fusion 1
  --lora_rank 8
  --warmup 1
  --cl_epochs 2
  --bs 8
  --accum 4
  --cl_c_min 0.10
  --cl_neg_c_min 0.10
)

echo "[fs2] start noargs $(date '+%F %T')"
"$PY" -m models.cv_eval \
  --dataset data/final/dataset_verify_faithful.jsonl \
  "${COMMON[@]}" \
  --tmpdir "$OUT/cv_tmp_noargs_small_e3_c10_fs2_s0" \
  --out "$OUT/cv_noargs_small_e3_c10_fs2_s0.json"

echo "[fs2] start args $(date '+%F %T')"
"$PY" -m models.cv_eval \
  --dataset data/final/dataset_verify_faithful_args.jsonl \
  "${COMMON[@]}" \
  --tmpdir "$OUT/cv_tmp_args_small_e3_c10_det_fairbase_fs2_s0" \
  --out "$OUT/cv_args_small_e3_c10_det_fairbase_fs2_s0.json"

echo "[fs2] done $(date '+%F %T')"
