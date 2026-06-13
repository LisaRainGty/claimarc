# Full Pair LLM Review Audit v1

This report audits LLM/VLM reconstruction reviews before promotion.
It checks label-definition consistency rather than benchmark separability.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak120_lownoise_next40_joint_review_queue_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_lownoise_next40_noimg_v1_20260614.jsonl`

## Outputs

- report json: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_lownoise_next40_noimg_audit_v4_20260614.report.json`
- flagged rows: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_lownoise_next40_noimg_audit_flags_v4_20260614.jsonl`

## Summary

- `queue_rows`: `40`
- `review_rows`: `40`
- `matched_reviews`: `40`
- `missing_reviews`: `0`
- `duplicate_review_pairs`: `0`
- `duplicate_review_events`: `0`
- `flagged_rows`: `18`
- `high_flag_rows`: `2`
- `medium_or_high_flag_rows`: `18`
- `flag_severity`: `{'medium': 26, 'high': 2, 'info': 1}`
- `flag_code`: `{'attribute_semantic_drift_requires_silver': 1, 'promote_action_but_not_main_ready': 7, 'claim_not_in_top_srt_prefilter': 6, 'consumer_expectation_mismatch_requires_silver': 1, 'conflicting_comment_relation_requires_silver': 1, 'identity_attribute_claim_lacks_value': 3, 'numeric_value_judgment_used_as_refute': 1, 'positive_label_missing_product_evidence_for_main': 5, 'subjective_eval_attribute_requires_silver': 2, 'mechanism_contradiction_without_consumer_refute': 1, 'price_value_not_direct_refute_requires_silver': 1}`
- `promotion_state`: `{'repair_missing_claim': 12, 'silver_refute_insufficient_product_evidence': 6, 'main_positive_refute': 5, 'silver_attribute_semantic_drift': 1, 'lowinfo_no_aligned_comment': 1, 'silver_consumer_expectation_mismatch': 1, 'repair_identity_claim_value': 2, 'repair_insufficient_product_evidence': 3, 'silver_refute_missing_product_evidence': 5, 'silver_subjective_eval_attribute': 2, 'main_negative_support': 2}`
- `new_y`: `{'0': 20, '1': 20}`
- `confidence`: `{'high': 40}`
- `category`: `{'food_and_beverages': 14, 'general': 3, 'smart_home': 2, 'baby_kids_and_pets': 8, 'beauty_and_personal_care': 5, 'digital_and_electronics': 3, 'shoes_and_bags': 2, 'apparel_and_underwear': 2, 'sports_and_outdoor': 1}`
- `queue`: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak120_lownoise_next40_joint_review_queue_v1_20260614.jsonl`
- `reviews`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_lownoise_next40_noimg_v1_20260614.jsonl`
- `flagged`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_lownoise_next40_noimg_audit_flags_v4_20260614.jsonl`
- `out`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_lownoise_next40_noimg_audit_v4_20260614.report.json`

## Gate Interpretation

- `high` flags block main promotion until rerun or manual repair.
- `medium` flags require manual sampling or silver routing.
- `mechanism_contradiction_without_consumer_refute` is not a positive label by itself.
- Missing reviews are expected before the LLM pilot has been executed.
