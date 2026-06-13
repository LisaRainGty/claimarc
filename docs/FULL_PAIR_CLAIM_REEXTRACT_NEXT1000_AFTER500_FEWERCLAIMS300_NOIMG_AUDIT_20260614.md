# Full Pair LLM Review Audit v1

This report audits LLM/VLM reconstruction reviews before promotion.
It checks label-definition consistency rather than benchmark separability.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next1000_after500_fewerclaims_joint_review_queue_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_next1000_after500_fewerclaims300_noimg_flash_v1_20260614.jsonl`

## Outputs

- report json: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_next1000_after500_fewerclaims300_noimg_flash_v1_audit_20260614.json`
- flagged rows: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_next1000_after500_fewerclaims300_noimg_flash_v1_flags_20260614.jsonl`

## Summary

- `queue_rows`: `820`
- `review_rows`: `300`
- `matched_reviews`: `300`
- `missing_reviews`: `520`
- `duplicate_review_pairs`: `0`
- `duplicate_review_events`: `0`
- `flagged_rows`: `621`
- `high_flag_rows`: `3`
- `medium_or_high_flag_rows`: `99`
- `flag_severity`: `{'info': 10, 'medium': 123, 'high': 3, 'low': 2, 'missing': 520}`
- `flag_code`: `{'mechanism_contradiction_without_consumer_refute': 10, 'promote_action_but_not_main_ready': 52, 'conflicting_comment_relation_requires_silver': 10, 'schema_meta_attribute_requires_silver': 6, 'positive_label_missing_product_evidence_for_main': 35, 'claim_not_in_top_srt_prefilter': 9, 'numeric_value_judgment_used_as_refute': 3, 'subjective_eval_attribute_requires_silver': 9, 'identity_attribute_claim_lacks_value': 2, 'raw_llm_y_overridden_by_clean_rule': 2, 'missing_review': 520}`
- `promotion_state`: `{'repair_missing_claim': 88, 'main_positive_refute': 48, 'lowinfo_no_aligned_comment': 33, 'repair_missing_evidence': 31, 'silver_refute_insufficient_product_evidence': 25, 'silver_conflicting_comment_relation': 7, 'silver_refute_missing_product_evidence': 35, 'repair_insufficient_product_evidence': 19, 'repair_numeric_value_judgment': 1, 'silver_subjective_eval_attribute': 6, 'silver_schema_meta_attribute': 3, 'main_negative_support': 2, 'repair_identity_claim_value': 2, 'missing_review': 520}`
- `new_y`: `{'0': 176, '1': 124, 'None': 520}`
- `confidence`: `{'high': 299, 'medium': 1}`
- `category`: `{'apparel_and_underwear': 83, 'baby_kids_and_pets': 141, 'beauty_and_personal_care': 29, 'digital_and_electronics': 24, 'food_and_beverages': 227, 'general': 139, 'jewelry_and_collectibles': 7, 'shoes_and_bags': 18, 'smart_home': 87, 'sports_and_outdoor': 65}`
- `queue`: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next1000_after500_fewerclaims_joint_review_queue_v1_20260614.jsonl`
- `reviews`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_next1000_after500_fewerclaims300_noimg_flash_v1_20260614.jsonl`
- `flagged`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_next1000_after500_fewerclaims300_noimg_flash_v1_flags_20260614.jsonl`
- `out`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_next1000_after500_fewerclaims300_noimg_flash_v1_audit_20260614.json`

## Gate Interpretation

- `high` flags block main promotion until rerun or manual repair.
- `medium` flags require manual sampling or silver routing.
- `mechanism_contradiction_without_consumer_refute` is not a positive label by itself.
- Missing reviews are expected before the LLM pilot has been executed.
