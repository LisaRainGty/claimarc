#!/usr/bin/env bash
set -euo pipefail

cd /mnt/gty/claimarc_active
export PYTHONPATH=src

DATA=data/final/dataset_verify_faithful.jsonl
OUTDIR=data/final/cleancl2
mkdir -p "$OUTDIR"
LOG="$OUTDIR/sweep_$(date +%Y%m%d_%H%M%S).log"

run_one() {
  local tag=$1
  local seed=$2
  local cmin=$3
  local lambda=$4
  local out="$OUTDIR/${tag}_s${seed}.pt"
  if [[ -f "$out" ]]; then
    echo "[skip] $out" | tee -a "$LOG"
    return
  fi
  echo "[run] tag=$tag seed=$seed cmin=$cmin lambda=$lambda" | tee -a "$LOG"
  /root/miniconda3/bin/python -m models.train \
    --dataset "$DATA" \
    --encoder_name BAAI/bge-small-zh-v1.5 \
    --bs 8 --accum 4 --warmup 1 --cl_epochs 2 --n_fusion 1 --lora_rank 8 \
    --lambda_cl "$lambda" --tau 0.07 --cl_c_min "$cmin" --cl_neg_c_min "$cmin" \
    --tag "$tag" --seed "$seed" --save_emb "$out" 2>&1 | tee -a "$LOG"
}

for seed in 0 1 2 3 4; do
  run_one small_e3_c05 "$seed" 0.05 0.5
  run_one small_e3_c20 "$seed" 0.20 0.5
  run_one small_e3_lam03_c10 "$seed" 0.10 0.3
done

for tag in small_e3_c05 small_e3_c20 small_e3_lam03_c10; do
  /root/miniconda3/bin/python -m models.fusion_eval \
    --dataset "$DATA" \
    --cm "$OUTDIR/${tag}_s0.pt" "$OUTDIR/${tag}_s1.pt" "$OUTDIR/${tag}_s2.pt" "$OUTDIR/${tag}_s3.pt" "$OUTDIR/${tag}_s4.pt" \
    --out "$OUTDIR/cmp_${tag}_5seed.json" 2>&1 | tee -a "$LOG"
done

echo "[summary] $OUTDIR" | tee -a "$LOG"
