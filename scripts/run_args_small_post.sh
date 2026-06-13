#!/usr/bin/env bash
set -euo pipefail

cd /mnt/gty/claimarc_active
export PYTHONPATH=src

DATA=data/final/dataset_verify_faithful_args.jsonl
OUTDIR=data/final/cleancl
LOG=data/final/args_small_post_inner.log
mkdir -p "$OUTDIR"

need=1694
while true; do
  have=0
  if [[ -f "$DATA" ]]; then
    have=$(wc -l < "$DATA")
  fi
  echo "[args-post] argument rows=$have/$need" | tee -a "$LOG"
  if [[ "$have" -ge "$need" ]]; then
    break
  fi
  sleep 180
done

for seed in 0 1 2; do
  out="$OUTDIR/args_small_e3_c10_s${seed}.pt"
  if [[ -f "$out" ]]; then
    echo "[args-post] skip existing $out" | tee -a "$LOG"
    continue
  fi
  echo "[args-post] train seed=$seed" | tee -a "$LOG"
  /root/miniconda3/bin/python -m models.train \
    --dataset "$DATA" \
    --encoder_name BAAI/bge-small-zh-v1.5 \
    --bs 8 --accum 4 --warmup 1 --cl_epochs 2 --n_fusion 1 --lora_rank 8 \
    --lambda_cl 0.5 --tau 0.07 --cl_c_min 0.10 --cl_neg_c_min 0.10 \
    --tag args_small_e3_c10 --seed "$seed" --save_emb "$out" 2>&1 | tee -a "$LOG"
done

/root/miniconda3/bin/python -m models.fusion_eval \
  --dataset "$DATA" \
  --cm "$OUTDIR/args_small_e3_c10_s0.pt" "$OUTDIR/args_small_e3_c10_s1.pt" "$OUTDIR/args_small_e3_c10_s2.pt" \
  --out "$OUTDIR/cmp_args_small_e3_c10_3seed.json" 2>&1 | tee -a "$LOG"

echo "[args-post] done" | tee -a "$LOG"
