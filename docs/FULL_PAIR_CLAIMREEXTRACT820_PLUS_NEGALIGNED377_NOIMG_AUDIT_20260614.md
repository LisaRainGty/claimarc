# Full Pair LLM Review Audit v1

This report audits LLM/VLM reconstruction reviews before promotion.
It checks label-definition consistency rather than benchmark separability.

## Inputs

- queue: `data/final/repaired_v1/full_pair_joint_review_queue_claimreextract820_plus_negaligned377_reviewed_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract820_plus_negaligned377_noimg_flash_v1_20260614.jsonl`

## Outputs

- report json: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract820_plus_negaligned377_noimg_flash_v1_audit_20260614.json`
- flagged rows: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract820_plus_negaligned377_noimg_flash_v1_flags_20260614.jsonl`

## Summary

- `queue_rows`: `1196`
- `review_rows`: `1196`
- `matched_reviews`: `1196`
- `missing_reviews`: `0`
- `duplicate_review_pairs`: `0`
- `duplicate_review_events`: `0`
- `flagged_rows`: `437`
- `high_flag_rows`: `11`
- `medium_or_high_flag_rows`: `406`
- `flag_severity`: `{'info': 100, 'medium': 482, 'high': 11, 'low': 9}`
- `flag_code`: `{'mechanism_contradiction_without_consumer_refute': 100, 'promote_action_but_not_main_ready': 264, 'conflicting_comment_relation_requires_silver': 32, 'schema_meta_attribute_requires_silver': 27, 'positive_label_missing_product_evidence_for_main': 82, 'claim_not_in_top_srt_prefilter': 48, 'numeric_value_judgment_used_as_refute': 6, 'subjective_eval_attribute_requires_silver': 21, 'identity_attribute_claim_lacks_value': 9, 'raw_llm_y_overridden_by_clean_rule': 9, 'attribute_semantic_drift_requires_silver': 1, 'enumeration_claim_evidence_extra_values': 3}`
- `promotion_state`: `{'repair_missing_claim': 310, 'main_positive_refute': 143, 'lowinfo_no_aligned_comment': 204, 'repair_missing_evidence': 73, 'silver_refute_insufficient_product_evidence': 72, 'silver_conflicting_comment_relation': 22, 'silver_refute_missing_product_evidence': 82, 'repair_insufficient_product_evidence': 133, 'repair_numeric_value_judgment': 3, 'silver_subjective_eval_attribute': 11, 'silver_schema_meta_attribute': 9, 'main_negative_support': 117, 'repair_identity_claim_value': 6, 'silver_commercial_promise_attribute': 7, 'silver_attribute_semantic_drift': 1, 'silver_enumeration_evidence_extra_values': 3}`
- `new_y`: `{'0': 858, '1': 338}`
- `confidence`: `{'high': 1188, 'medium': 1, 'low': 7}`
- `category`: `{'apparel_and_underwear': 192, 'baby_kids_and_pets': 213, 'beauty_and_personal_care': 58, 'digital_and_electronics': 60, 'food_and_beverages': 242, 'general': 174, 'jewelry_and_collectibles': 16, 'shoes_and_bags': 76, 'smart_home': 93, 'sports_and_outdoor': 72}`
- `queue`: `data/final/repaired_v1/full_pair_joint_review_queue_claimreextract820_plus_negaligned377_reviewed_v1_20260614.jsonl`
- `reviews`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract820_plus_negaligned377_noimg_flash_v1_20260614.jsonl`
- `flagged`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract820_plus_negaligned377_noimg_flash_v1_flags_20260614.jsonl`
- `out`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract820_plus_negaligned377_noimg_flash_v1_audit_20260614.json`

## Gate Interpretation

- `high` flags block main promotion until rerun or manual repair.
- `medium` flags require manual sampling or silver routing.
- `mechanism_contradiction_without_consumer_refute` is not a positive label by itself.
- Missing reviews are expected before the LLM pilot has been executed.
