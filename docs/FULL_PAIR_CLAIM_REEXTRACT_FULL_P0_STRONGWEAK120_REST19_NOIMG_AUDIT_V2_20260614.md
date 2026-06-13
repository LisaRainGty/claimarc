# Full Pair LLM Review Audit v1

This report audits LLM/VLM reconstruction reviews before promotion.
It checks label-definition consistency rather than benchmark separability.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak120_rest19_joint_review_queue_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_rest19_noimg_v1_20260614.jsonl`

## Outputs

- report json: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_rest19_noimg_audit_v2_20260614.report.json`
- flagged rows: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_rest19_noimg_audit_flags_v2_20260614.jsonl`

## Summary

- `queue_rows`: `19`
- `review_rows`: `19`
- `matched_reviews`: `19`
- `missing_reviews`: `0`
- `duplicate_review_pairs`: `0`
- `duplicate_review_events`: `0`
- `flagged_rows`: `11`
- `high_flag_rows`: `0`
- `medium_or_high_flag_rows`: `11`
- `flag_severity`: `{'medium': 18}`
- `flag_code`: `{'conflicting_comment_relation_requires_silver': 6, 'promote_action_but_not_main_ready': 6, 'attribute_semantic_drift_requires_silver': 1, 'claim_not_in_top_srt_prefilter': 2, 'positive_label_missing_product_evidence_for_main': 2, 'identity_attribute_claim_lacks_value': 1}`
- `promotion_state`: `{'silver_conflicting_comment_relation': 4, 'silver_refute_insufficient_product_evidence': 3, 'silver_attribute_semantic_drift': 1, 'repair_missing_claim': 5, 'main_positive_refute': 3, 'lowinfo_no_aligned_comment': 1, 'silver_refute_missing_product_evidence': 2}`
- `new_y`: `{'1': 13, '0': 6}`
- `confidence`: `{'high': 19}`
- `category`: `{'baby_kids_and_pets': 5, 'food_and_beverages': 10, 'general': 2, 'sports_and_outdoor': 2}`
- `queue`: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak120_rest19_joint_review_queue_v1_20260614.jsonl`
- `reviews`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_rest19_noimg_v1_20260614.jsonl`
- `flagged`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_rest19_noimg_audit_flags_v2_20260614.jsonl`
- `out`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_rest19_noimg_audit_v2_20260614.report.json`

## Gate Interpretation

- `high` flags block main promotion until rerun or manual repair.
- `medium` flags require manual sampling or silver routing.
- `mechanism_contradiction_without_consumer_refute` is not a positive label by itself.
- Missing reviews are expected before the LLM pilot has been executed.
