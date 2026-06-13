# GPU CV: stateful v2 neg200 + next500full l200

Date: 2026-06-14

## Dataset

- Supervised file: `data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_next200_neg200_next500full_l200_supervised_20260614.jsonl`
- Reviewed rows: 664
- Supervised rows: 327
- Contrastive rows: 146
- Label mix in supervised rows: 223 positive / 104 negative
- Fold protocol: room-grouped 5-fold CV, `fold_seed=14`, `cm_seeds=[0]`

## Run

Remote GPU run:

```bash
PYTHONPATH=src python -m models.cv_eval \
  --dataset data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_next200_neg200_next500full_l200_supervised_20260614.jsonl \
  --folds 5 --fold_seed 14 --cm_seeds 0 \
  --baselines bge_lr \
  --tmpdir data/final/repaired_v1/gpu_cv/tmp_l8_t10_lam10 \
  --out data/final/repaired_v1/gpu_cv/cv_l8_t10_lam10_neg200_next500full_l200.json \
  --dump_oof data/final/repaired_v1/gpu_cv/cv_l8_t10_lam10_neg200_next500full_l200_oof.npz \
  --n_boot 0 --warmup 3 --cl_epochs 6 --lora_rank 8 \
  --tau 0.10 --lambda_cl 1.0 --bs 8 --accum 2
```

## Pooled Results

| Method | AUPRC | AUROC | Macro-F1 | wF1 | n |
|---|---:|---:|---:|---:|---:|
| CLAIMARC_pcls | 0.7109 | 0.5529 | 0.5440 | 0.5507 | 327 |
| CLAIMARC_selectiveRKC | 0.7112 | 0.5540 | 0.5479 | 0.5550 | 327 |
| CLAIMARC_v2 | 0.7225 | 0.5672 | 0.5450 | 0.5546 | 327 |
| bge_lr | 0.7665 | 0.5878 | 0.5484 | 0.5479 | 327 |

## Threshold Diagnosis

Post-hoc oracle thresholds on the same OOF probabilities show only small recoverable F1 gaps:

| Method | Saved Macro-F1 | Oracle Macro-F1 | Gap | Saved AUROC |
|---|---:|---:|---:|---:|
| CLAIMARC_pcls | 0.5440 | 0.5677 | +0.0238 | 0.5529 |
| CLAIMARC_selectiveRKC | 0.5479 | 0.5677 | +0.0198 | 0.5540 |
| CLAIMARC_v2 | 0.5450 | 0.5702 | +0.0252 | 0.5672 |
| bge_lr | 0.5484 | 0.5532 | +0.0048 | 0.5878 |

## Interpretation

This run is not sufficient for a paper claim. The strongest proposal-faithful CLAIMARC variant does not yet beat the BGE-LR baseline on ranking, and the threshold diagnosis shows that the main bottleneck is not simply validation-threshold drift. Selective RKC makes only a tiny difference, which suggests the current contrastive pool is still too small or too weakly structured.

The next data work should not delete hard samples or chase artificially high AUROC. It should recover proposal-valid rows from the full `(product, attribute)` universe by improving:

1. claim extraction coverage from streamer speech,
2. aligned evidence extraction from product detail images/text,
3. comment-vs-claim contradiction labeling only after a claim is actually recovered,
4. strict same-attribute positive/negative contrastive neighborhoods.

The next model work should stay compact: lower contrastive weight screens, prior-stable threshold checks, and a source/evidence sufficiency guard are plausible, but the main path remains data recovery.
