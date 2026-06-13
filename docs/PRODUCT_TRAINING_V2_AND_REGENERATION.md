# Product Training v2

## Outputs
- training dataset: `data/final/repaired_v1/dataset_attrpol_product_train_v2.jsonl`
- regeneration manifest: `data/final/repaired_v1/regeneration_manifest_v1.jsonl`

## Summary
- `train_n`: `3103`
- `train_labels`: `{0: 2137, 1: 966}`
- `train_split`: `{'train': 2116, 'test': 672, 'val': 315}`
- `train_split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
- `train_quality_bucket`: `{'neg_context_sourceful': 954, 'pos_silver': 560, 'pos_core': 266, 'neg_weak': 645, 'pos_weak': 140, 'neg_core': 47, 'neg_suspect_fake': 428, 'neg_silver_sourceful': 42, 'neg_silver_comment_only': 21}`
- `train_confidence`: `{'low': 1234, 'absent': 1192, 'medium': 648, 'high': 29}`
- `train_source0`: `1192`
- `train_needs_regeneration`: `2122`
- `train_attribute_noise`: `136`
- `manifest_n`: `3836`
- `manifest_priority`: `{1: 2086, 2: 1170, 3: 580}`
- `manifest_actions`: `{'rerun_claim_extraction': 1714, 'llm_claim_comment_adjudication': 2927, 'rerun_product_evidence': 1192, 'evidence_sufficiency_check': 1840, 'search_missing_counterevidence': 372, 'schema_repair_review': 136, 'negative_label_verification': 1073}`

## Learnability
- lightweight grouped diagnostic on `dataset_attrpol_product_train_v2.jsonl`:
  AUPRC `0.5817`, AUROC `0.7475`, Macro-F1 `0.6807`.
- this is lower than the clean HQ product set, confirming that v2 is an
  augmentation/regeneration pool rather than a clean benchmark.

## Interpretation
This is a training-augmentation pool, not the clean evaluation benchmark.
Rows with absent evidence, weak labels, or suspected fake positive comments are kept for scale but down-weighted and queued for regeneration.
The clean main benchmark should remain the HQ product set until these manifest items are re-adjudicated.
