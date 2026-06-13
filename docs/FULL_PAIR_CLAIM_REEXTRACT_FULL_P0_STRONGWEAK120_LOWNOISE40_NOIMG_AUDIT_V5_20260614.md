# Full Pair LLM Review Audit v1

This report audits LLM/VLM reconstruction reviews before promotion.
It checks label-definition consistency rather than benchmark separability.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak120_lownoise40_joint_review_queue_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_lownoise40_noimg_v1_20260614.jsonl`

## Outputs

- report json: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_lownoise40_noimg_audit_v5_20260614.report.json`
- flagged rows: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_lownoise40_noimg_audit_flags_v5_20260614.jsonl`

## Summary

- `queue_rows`: `40`
- `review_rows`: `40`
- `matched_reviews`: `40`
- `missing_reviews`: `0`
- `duplicate_review_pairs`: `0`
- `duplicate_review_events`: `0`
- `flagged_rows`: `16`
- `high_flag_rows`: `4`
- `medium_or_high_flag_rows`: `16`
- `flag_severity`: `{'medium': 28, 'high': 4, 'info': 1}`
- `flag_code`: `{'attribute_semantic_drift_requires_silver': 2, 'promote_action_but_not_main_ready': 8, 'conflicting_comment_relation_requires_silver': 5, 'enumeration_claim_evidence_extra_values': 1, 'identity_attribute_claim_lacks_value': 3, 'mechanism_contradiction_without_consumer_refute': 1, 'numeric_value_judgment_used_as_refute': 6, 'price_value_not_direct_refute_requires_silver': 3, 'positive_label_missing_product_evidence_for_main': 4}`
- `promotion_state`: `{'repair_missing_claim': 11, 'silver_attribute_semantic_drift': 1, 'silver_refute_insufficient_product_evidence': 8, 'main_positive_refute': 6, 'repair_identity_claim_value': 2, 'silver_refute_missing_product_evidence': 4, 'repair_numeric_value_judgment': 2, 'lowinfo_no_aligned_comment': 2, 'repair_insufficient_product_evidence': 2, 'silver_commercial_promise_attribute': 1, 'main_negative_support': 1}`
- `new_y`: `{'0': 18, '1': 22}`
- `confidence`: `{'high': 39, 'low': 1}`
- `category`: `{'baby_kids_and_pets': 9, 'beauty_and_personal_care': 2, 'food_and_beverages': 10, 'general': 10, 'shoes_and_bags': 3, 'smart_home': 6}`
- `queue`: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak120_lownoise40_joint_review_queue_v1_20260614.jsonl`
- `reviews`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_lownoise40_noimg_v1_20260614.jsonl`
- `flagged`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_lownoise40_noimg_audit_flags_v5_20260614.jsonl`
- `out`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_lownoise40_noimg_audit_v5_20260614.report.json`

## Gate Interpretation

- `high` flags block main promotion until rerun or manual repair.
- `medium` flags require manual sampling or silver routing.
- `mechanism_contradiction_without_consumer_refute` is not a positive label by itself.
- Missing reviews are expected before the LLM pilot has been executed.
