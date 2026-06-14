# Data Repair Progress: Claim Re-Extraction + Natural Negatives

This note records the 2026-06-14 repair step after the proposal-alignment correction.
The goal is to recover complete `(claim, product evidence, consumer relation)` triplets, not to remove hard samples or inflate separability.

## Claim Re-Extraction Batch

- source queue: `data/final/repaired_v1/full_pair_claim_repair_queue_full_p0_strongweak_next1000_after500_v1_20260614.jsonl`
- claim re-extract output: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next1000_after500_v1_20260614.jsonl`
- claim-reextract result: 820/1000 pairs recovered at least one SRT claim candidate.
- joint review queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next1000_after500_fewerclaims_joint_review_queue_v1_20260614.jsonl`
- full reviewed set: 820/820 fewer-claim rows.
- audit summary: 132 `main_positive_refute`, 13 `main_negative_support`, 151 evidence-incomplete silver positives, and 471 repair/unobserved rows.
- reconstruction summary: 573/820 `claim_found`, 487/820 `product_evidence_found`, and 320 positive / 500 negative raw rebuilt labels.

Interpretation: claim recall improved, but positive-risk repair rows still need product evidence completion, especially when no-image review cannot inspect detail images directly.

## Schema Guard Update

The promotion and audit gates now route schema/meta/evaluative attributes to silver instead of strict main training. Examples include `视频内容`, `购买意图`, `宣传内容`, `假货`, `与事实不符`, `实物图描述`, `社交效果`, and `价格透明度`.

This preserves the rows for audit/mechanism analysis while preventing consumer-conclusion strings from becoming product attributes in the main contrastive task.

## Natural Negative Control Batch

- queue: `data/final/repaired_v1/full_pair_negative_control_queue_n0_500_v1_20260614.jsonl`
- reviewed aligned subset: all 377 old `label_negative_claim_aligned_nonneg` rows.
- LLM review summary: 359 `new_y=0`, 18 `new_y=1`, 314 `claim_found`, 351 `product_evidence_found`.
- audit summary: 104 `main_negative_support`, 11 `main_positive_refute`, 3 evidence-incomplete/silver positives, and 222 repair/unobserved rows.

Interpretation: the negative-control queue is a valid source of proposal-faithful natural negatives because it starts from claim-present, evidence-present pairs and still rebuilds labels through the same claim-comment comparison.
The remaining 123 `label_negative_no_aligned_review` rows are kept as low-information candidates rather than forced into main negatives.

## Combined Stateful View

- combined reviewed queue: `data/final/repaired_v1/full_pair_joint_review_queue_claimreextract820_plus_negaligned377_reviewed_v1_20260614.jsonl`
- no-image stateful dataset: `data/final/repaired_v1/stateful_proposal_dataset_v2_claimreextract820_plus_negaligned377_noimg_20260614.jsonl`
- supervised observed rows: 504, with labels 338 positive / 166 negative.
- strict contrastive rows: 248.
- main rows: 260, with 143 `main_positive_refute` / 117 `main_negative_support`.
- repair/unobserved rows: 692.
- split leakage: 0 leaky rooms.

The combined view is much healthier than the earlier positive-heavy repair view, but it remains category-imbalanced and still has many evidence-incomplete positives.

## VLM Evidence Repair

- evidence repair queue: `data/final/repaired_v1/full_pair_evidence_repair_queue_claimreextract820_plus_negaligned377_v1_20260614.jsonl`
- guarded attributes skipped before VLM: 8 schema/meta, 6 commercial-promise, 1 subjective/evaluative.
- VLM-reviewed rows: 120.
- VLM audit summary: 22 `main_positive_refute`, 31 `silver_refute_missing_product_evidence`, 26 `silver_refute_insufficient_product_evidence`, and 23 `repair_missing_claim`.
- VLM-overridden stateful dataset: `data/final/repaired_v1/stateful_proposal_dataset_v2_claimreextract820_plus_negaligned377_vlm120_20260614.jsonl`
- supervised observed rows: 469, with labels 303 positive / 166 negative.
- strict contrastive rows: 266.
- main rows: 282, with 165 `main_positive_refute` / 117 `main_negative_support`.
- evidence-incomplete silver rows: 91.

Interpretation: VLM repair does not simply add rows. It promotes some evidence-complete positives while demoting some claim/evidence-unstable rows back to repair. This is preferable for proposal-faithful data quality.

## Lightweight Learnability Diagnostic

No-image dataset: `data/final/repaired_v1/stateful_proposal_dataset_v2_claimreextract820_plus_negaligned377_noimg_supervised_20260614.jsonl`

Simple character TF-IDF + logistic regression, 5-fold room-grouped OOF:

- AUPRC: 0.8000
- AUROC: 0.6436
- Macro-F1: 0.5452

VLM120-overridden dataset: `data/final/repaired_v1/stateful_proposal_dataset_v2_claimreextract820_plus_negaligned377_vlm120_supervised_20260614.jsonl`

- AUPRC: 0.7938
- AUROC: 0.6845
- Macro-F1: 0.6112

This is only a data-health diagnostic, not a paper baseline. The fold variance remains large, so the current data is learnable but not suspiciously easy.

## Next Actions

1. Continue VLM evidence repair beyond the top 120 evidence-incomplete rows.
2. Build a new aligned-support negative queue from raw comments instead of using `label_negative_no_aligned_review` as main negatives.
3. Recompute old stateful batches with the new schema guard before merging them into paper-scale experiments.
4. Run GPU CV on the VLM120-overridden dataset as a small calibration benchmark before scaling to a larger reconstructed dataset.
