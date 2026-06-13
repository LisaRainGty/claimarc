#!/usr/bin/env bash
set -euo pipefail

ROOT="${CLAIMARC_ROOT:-/mnt/gty/claimarc_active}"
PY="${CLAIMARC_PY:-/root/miniconda3/bin/python}"
cd "$ROOT"

export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUT_DIR="data/final/cleancl"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/sweep_$(date +%Y%m%d_%H%M%S).log"
SUMMARY="$OUT_DIR/results.jsonl"

run_one() {
  local tag="$1"
  local cl_c_min="$2"
  local cl_neg_c_min="$3"
  local warmup="$4"
  local cl_epochs="$5"
  local seed="$6"
  local out="$OUT_DIR/${tag}_s${seed}.pt"
  echo "[run] tag=${tag} seed=${seed} cl_c_min=${cl_c_min} warmup=${warmup} cl_epochs=${cl_epochs}" | tee -a "$LOG"
  "$PY" -m models.train \
    --dataset data/final/dataset_verify_faithful.jsonl \
    --encoder_name BAAI/bge-small-zh-v1.5 \
    --bs 8 --accum 4 \
    --warmup "$warmup" --cl_epochs "$cl_epochs" \
    --n_fusion 1 --lora_rank 8 \
    --lambda_cl 0.5 --tau 0.07 \
    --cl_c_min "$cl_c_min" --cl_neg_c_min "$cl_neg_c_min" \
    --tag "$tag" --seed "$seed" --save_emb "$out" 2>&1 | tee -a "$LOG"
}

for seed in 0 1 2 3 4; do
  run_one small_e1_c00 0.00 0.00 0 1 "$seed"
  run_one small_e1_c10 0.10 0.10 0 1 "$seed"
  run_one small_e1_c15 0.15 0.15 0 1 "$seed"
  run_one small_e3_c10 0.10 0.10 1 2 "$seed"
done

grep '^RESULT ' "$LOG" | sed 's/^RESULT //' > "$SUMMARY"
echo "[summary] $SUMMARY" | tee -a "$LOG"
