# Full Pair LLM Review Audit v1

This report audits LLM/VLM reconstruction reviews before promotion.
It checks label-definition consistency rather than benchmark separability.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_weak_next1000_after1500_pilot300_fewerclaims_joint_review_queue_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_weak_next1000_after1500_pilot300_fewerclaims194_noimg_flash_v1_20260614.jsonl`

## Outputs

- report json: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_weak_next1000_after1500_pilot300_fewerclaims194_noimg_flash_v1_audit_20260614.json`
- flagged rows: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_weak_next1000_after1500_pilot300_fewerclaims194_noimg_flash_v1_flags_20260614.jsonl`

## Summary

- `queue_rows`: `194`
- `review_rows`: `194`
- `matched_reviews`: `194`
- `missing_reviews`: `0`
- `duplicate_review_pairs`: `0`
- `duplicate_review_events`: `0`
- `flagged_rows`: `61`
- `high_flag_rows`: `4`
- `medium_or_high_flag_rows`: `57`
- `flag_severity`: `{'info': 5, 'medium': 69, 'low': 5, 'high': 4}`
- `flag_code`: `{'mechanism_contradiction_without_consumer_refute': 5, 'promote_action_but_not_main_ready': 46, 'claim_not_in_top_srt_prefilter': 8, 'conflicting_comment_relation_requires_silver': 7, 'raw_llm_y_overridden_by_clean_rule': 5, 'positive_label_missing_product_evidence_for_main': 4, 'identity_attribute_claim_lacks_value': 2, 'schema_meta_attribute_requires_silver': 2, 'numeric_value_judgment_used_as_refute': 2, 'subjective_eval_attribute_requires_silver': 1, 'attribute_semantic_drift_requires_silver': 1}`
- `promotion_state`: `{'lowinfo_no_aligned_comment': 36, 'silver_conflicting_comment_relation': 5, 'main_positive_refute': 37, 'repair_insufficient_product_evidence': 13, 'repair_missing_claim': 82, 'silver_refute_missing_product_evidence': 4, 'repair_identity_claim_value': 2, 'repair_missing_evidence': 5, 'silver_schema_meta_attribute': 1, 'silver_refute_insufficient_product_evidence': 4, 'main_negative_support': 3, 'repair_numeric_value_judgment': 2}`
- `new_y`: `{'0': 142, '1': 52}`
- `confidence`: `{'high': 191, 'low': 3}`
- `category`: `{'apparel_and_underwear': 20, 'baby_kids_and_pets': 29, 'beauty_and_personal_care': 34, 'digital_and_electronics': 5, 'food_and_beverages': 30, 'general': 38, 'shoes_and_bags': 15, 'smart_home': 10, 'sports_and_outdoor': 13}`
- `queue`: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_weak_next1000_after1500_pilot300_fewerclaims_joint_review_queue_v1_20260614.jsonl`
- `reviews`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_weak_next1000_after1500_pilot300_fewerclaims194_noimg_flash_v1_20260614.jsonl`
- `flagged`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_weak_next1000_after1500_pilot300_fewerclaims194_noimg_flash_v1_flags_20260614.jsonl`
- `out`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_weak_next1000_after1500_pilot300_fewerclaims194_noimg_flash_v1_audit_20260614.json`

## Gate Interpretation

- `high` flags block main promotion until rerun or manual repair.
- `medium` flags require manual sampling or silver routing.
- `mechanism_contradiction_without_consumer_refute` is not a positive label by itself.
- Missing reviews are expected before the LLM pilot has been executed.
