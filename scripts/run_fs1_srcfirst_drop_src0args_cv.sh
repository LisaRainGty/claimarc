#!/usr/bin/env bash
set -euo pipefail

cd /mnt/gty/claimarc_active
export PYTHONPATH=src

/root/miniconda3/bin/python -m models.cv_eval \
  --dataset data/final/dataset_verify_faithful_args_srcfirst_a120_drop_src0args.jsonl \
  --folds 5 \
  --fold_seed 1 \
  --cm_seeds 0 \
  --baselines bge_lr \
  --encoder_name BAAI/bge-small-zh-v1.5 \
  --n_fusion 1 \
  --lora_rank 8 \
  --warmup 1 \
  --cl_epochs 2 \
  --bs 8 \
  --accum 4 \
  --cl_c_min 0.10 \
  --cl_neg_c_min 0.10 \
  --tmpdir data/final/cleancl/cv_tmp_args_srcfirst_a120_drop_src0args_small_e3_c10_fs1_s0 \
  --out data/final/cleancl/cv_args_srcfirst_a120_drop_src0args_small_e3_c10_fs1_s0.json
