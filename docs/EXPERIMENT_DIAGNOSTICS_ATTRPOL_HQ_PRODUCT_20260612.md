# CLAIMARC AttrPol HQ Product Diagnostics

Date: 2026-06-12

## Dataset

Main repaired dataset:
`data/final/repaired_v1/dataset_attrpol_hq_product_v1.jsonl`

- n = 2364
- positives = 966
- negatives = 1398
- room-level split leakage = 0
- source0 = 727, with labels nearly balanced: 372 positive / 355 negative

## Small CLAIMARC 5-Fold CV

Remote output:
`data/final/cleancl/remote_results_20260612/cv_attrpol_hq_product_f5_srcproto_small_e3_c10_allbase_prior1_20260612.json`

| method | AUPRC | AUROC | Macro-F1 | wF1 |
|---|---:|---:|---:|---:|
| CLAIMARC_pcls | 0.7385 | 0.8531 | 0.7748 | 0.6884 |
| CLAIMARC_selectiveRKC | 0.7382 | 0.8526 | 0.7748 | 0.6884 |
| CLAIMARC_v2 | 0.7380 | 0.8518 | 0.7721 | 0.6886 |
| BGE-LR | 0.7983 | 0.8751 | 0.7916 | 0.7058 |
| BERT-CLS | 0.8160 | 0.8732 | 0.7764 | 0.6781 |
| RoBERTa-CLS | 0.7942 | 0.8533 | 0.7719 | 0.6937 |

Interpretation: the repaired data makes the benchmark substantially cleaner and
harder. CLAIMARC-small is competitive in Macro-F1 but not superior; BGE-LR and
BERT have better ranking metrics. This is not publishable yet as a main-result
claim.

## Where CLAIMARC Loses

Grouped OOF diagnostics:

- `source0/none`: CLAIMARC AP 0.562 vs BERT 0.625 vs RoBERTa 0.639.
- `PO` evidence: CLAIMARC AP 0.741 vs BGE-LR 0.809 vs BERT 0.841.
- `medium` confidence: CLAIMARC AP 0.758 vs BERT 0.853.
- weak categories:
  - baby/kids/pets: CLAIMARC AP 0.690 vs BERT 0.836.
  - beauty/personal-care: CLAIMARC AP 0.625 vs BERT 0.754.
  - shoes/bags: CLAIMARC AP 0.760 vs BERT 0.868.
  - general: CLAIMARC AP 0.784 vs BERT 0.885.

Where CLAIMARC is healthy:

- Sourceful low-confidence pairs: Macro-F1 0.889, close to BERT 0.885.
- Params-only evidence: Macro-F1 0.925, close to BGE-LR 0.927.
- Food and jewelry categories are strong but small.

## Next Experiments

1. BGE-large CLAIMARC-only CV is queued/running:
   `cv_attrpol_hq_product_large_f5_srcproto_e4_claimarc_only_20260612`.
2. If large does not close the AUPRC gap, prioritize:
   - source0 enrichment with an explicit `raw_product_context` source type,
     not ordinary PARAM evidence;
   - full argument regeneration for the repaired dataset;
   - source/evidence-type ranking calibration rather than only Macro-F1
     threshold tuning;
   - category-aware or source-aware retrieval negatives, especially for
     baby/kids/pets, beauty, shoes/bags, and general.

