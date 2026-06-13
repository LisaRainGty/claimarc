#!/usr/bin/env bash
set -euo pipefail

cd /mnt/gty/claimarc_active
export PYTHONPATH=src

PY=${PY:-/root/miniconda3/bin/python}
DATA=${DATA:-data/final/dataset_verify_faithful_args_srcfirst_a120.jsonl}
OUTDIR=${OUTDIR:-data/final/cleancl}
FS=${FS:-1}
POLICIES=${POLICIES:-"ocr_only params_only args_only"}
N_BOOT=${N_BOOT:-0}

for policy in ${POLICIES}; do
  tmp="${OUTDIR}/cv_tmp_sourcepolicy_${policy}_small_e3_c10_fs${FS}_s0"
  out="${OUTDIR}/cv_sourcepolicy_${policy}_small_e3_c10_fs${FS}_s0.json"
  dump="${OUTDIR}/oof_sourcepolicy_${policy}_small_e3_c10_fs${FS}_s0.npz"

  echo "[source-policy] policy=${policy} fold_seed=${FS}"
  "${PY}" -m models.cv_eval \
    --dataset "${DATA}" \
    --folds 5 \
    --fold_seed "${FS}" \
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
    --evidence_policy "${policy}" \
    --tmpdir "${tmp}" \
    --out "${out}" \
    --dump_oof "${dump}" \
    --n_boot "${N_BOOT}"
done
