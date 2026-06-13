# Full Pair LLM Review Audit v1

This report audits LLM/VLM reconstruction reviews before promotion.
It checks label-definition consistency rather than benchmark separability.

## Inputs

- queue: `data/final/repaired_v1/full_pair_negative_control_queue_n0_500_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_negative_control_n0_300_noimg_flash_v1_20260614.jsonl`

## Outputs

- report json: `data/final/repaired_v1/full_pair_reconstruction_llm_negative_control_n0_300_noimg_flash_v1_audit_20260614.json`
- flagged rows: `data/final/repaired_v1/full_pair_reconstruction_llm_negative_control_n0_300_noimg_flash_v1_flags_20260614.jsonl`

## Summary

- `queue_rows`: `500`
- `review_rows`: `300`
- `matched_reviews`: `300`
- `missing_reviews`: `200`
- `duplicate_review_pairs`: `0`
- `duplicate_review_events`: `0`
- `flagged_rows`: `325`
- `high_flag_rows`: `1`
- `medium_or_high_flag_rows`: `106`
- `flag_severity`: `{'medium': 109, 'info': 65, 'low': 1, 'high': 1, 'missing': 200}`
- `flag_code`: `{'promote_action_but_not_main_ready': 100, 'identity_attribute_claim_lacks_value': 3, 'mechanism_contradiction_without_consumer_refute': 65, 'raw_llm_y_overridden_by_clean_rule': 1, 'conflicting_comment_relation_requires_silver': 4, 'positive_label_missing_product_evidence_for_main': 2, 'enumeration_claim_evidence_extra_values': 1, 'missing_review': 200}`
- `promotion_state`: `{'lowinfo_no_aligned_comment': 96, 'repair_insufficient_product_evidence': 55, 'repair_missing_claim': 39, 'main_negative_support': 82, 'main_positive_refute': 9, 'repair_missing_evidence': 11, 'repair_identity_claim_value': 1, 'silver_conflicting_comment_relation': 3, 'silver_refute_missing_product_evidence': 2, 'silver_refute_insufficient_product_evidence': 1, 'silver_enumeration_evidence_extra_values': 1, 'missing_review': 200}`
- `new_y`: `{'0': 285, '1': 15, 'None': 200}`
- `confidence`: `{'high': 297, 'low': 3}`
- `category`: `{'food_and_beverages': 21, 'beauty_and_personal_care': 39, 'baby_kids_and_pets': 97, 'shoes_and_bags': 64, 'apparel_and_underwear': 158, 'digital_and_electronics': 54, 'general': 42, 'jewelry_and_collectibles': 12, 'sports_and_outdoor': 7, 'smart_home': 6}`
- `queue`: `data/final/repaired_v1/full_pair_negative_control_queue_n0_500_v1_20260614.jsonl`
- `reviews`: `data/final/repaired_v1/full_pair_reconstruction_llm_negative_control_n0_300_noimg_flash_v1_20260614.jsonl`
- `flagged`: `data/final/repaired_v1/full_pair_reconstruction_llm_negative_control_n0_300_noimg_flash_v1_flags_20260614.jsonl`
- `out`: `data/final/repaired_v1/full_pair_reconstruction_llm_negative_control_n0_300_noimg_flash_v1_audit_20260614.json`

## Gate Interpretation

- `high` flags block main promotion until rerun or manual repair.
- `medium` flags require manual sampling or silver routing.
- `mechanism_contradiction_without_consumer_refute` is not a positive label by itself.
- Missing reviews are expected before the LLM pilot has been executed.
