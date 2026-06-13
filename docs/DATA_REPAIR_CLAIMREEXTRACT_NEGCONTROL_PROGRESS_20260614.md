# Data Repair Progress: Claim Re-Extraction + Natural Negatives

This note records the 2026-06-14 repair step after the proposal-alignment correction.
The goal is to recover complete `(claim, product evidence, consumer relation)` triplets, not to remove hard samples or inflate separability.

## Claim Re-Extraction Batch

- source queue: `data/final/repaired_v1/full_pair_claim_repair_queue_full_p0_strongweak_next1000_after500_v1_20260614.jsonl`
- claim re-extract output: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next1000_after500_v1_20260614.jsonl`
- claim-reextract result: 820/1000 pairs recovered at least one SRT claim candidate.
- joint review queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next1000_after500_fewerclaims_joint_review_queue_v1_20260614.jsonl`
- reviewed subset: first 300 fewer-claim rows.
- audit summary: 48 `main_positive_refute`, 2 `main_negative_support`, 63 evidence-incomplete silver positives, and 171 repair/unobserved rows.

Interpretation: claim recall improved, but positive-risk repair rows still need product evidence completion, especially when no-image review cannot inspect detail images directly.

## Schema Guard Update

The promotion and audit gates now route schema/meta/evaluative attributes to silver instead of strict main training. Examples include `视频内容`, `购买意图`, `宣传内容`, `假货`, `与事实不符`, `实物图描述`, `社交效果`, and `价格透明度`.

This preserves the rows for audit/mechanism analysis while preventing consumer-conclusion strings from becoming product attributes in the main contrastive task.

## Natural Negative Control Batch

- queue: `data/final/repaired_v1/full_pair_negative_control_queue_n0_500_v1_20260614.jsonl`
- reviewed subset: first 300 rows using the same full-pair reviewer.
- LLM review summary: 285 `new_y=0`, 15 `new_y=1`, 261 `claim_found`, 281 `product_evidence_found`.
- audit summary: 82 `main_negative_support`, 9 `main_positive_refute`, 3 evidence-incomplete/silver positives, and 178 repair/unobserved rows.

Interpretation: the negative-control queue is a valid source of proposal-faithful natural negatives because it starts from claim-present, evidence-present pairs and still rebuilds labels through the same claim-comment comparison.

## Combined Stateful View

- combined reviewed queue: `data/final/repaired_v1/full_pair_joint_review_queue_claimreextract300_plus_neg300_reviewed_v1_20260614.jsonl`
- stateful dataset: `data/final/repaired_v1/stateful_proposal_dataset_v2_claimreextract300_plus_neg300_noimg_20260614.jsonl`
- supervised observed rows: 251, with labels 139 positive / 112 negative.
- strict contrastive rows: 138.
- main rows: 141, with 57 `main_positive_refute` / 84 `main_negative_support`.
- repair/unobserved rows: 349.
- split leakage: 0 leaky rooms.

The combined view is much healthier than the earlier positive-heavy repair view, but it remains small and category-imbalanced. It should be used as a calibration/diagnostic set rather than the final paper-scale dataset.

## Lightweight Learnability Diagnostic

Dataset: `data/final/repaired_v1/stateful_proposal_dataset_v2_claimreextract300_plus_neg300_noimg_supervised_20260614.jsonl`

Simple character TF-IDF + logistic regression, 5-fold room-grouped OOF:

- AUPRC: 0.7213
- AUROC: 0.6984
- Macro-F1: 0.6498

This is only a data-health diagnostic, not a paper baseline. The fold variance remains large, so the current data is learnable but not suspiciously easy.

## Next Actions

1. Continue reviewing the remaining claim-reextract queue with offset-based batches.
2. Review the remaining 200 natural-negative candidates, then rebuild the combined stateful view.
3. Add image/VLM evidence refresh for rows currently blocked by `silver_refute_missing_product_evidence` or `silver_refute_insufficient_product_evidence`.
4. Recompute old stateful batches with the new schema guard before merging them into paper-scale experiments.
