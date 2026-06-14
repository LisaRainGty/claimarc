# Full Pair LLM Review Audit v1

This report audits LLM/VLM reconstruction reviews before promotion.
It checks label-definition consistency rather than benchmark separability.

## Inputs

- queue: `data/final/repaired_v1/full_pair_evidence_repair_queue_claimreextract820_plus_negaligned377_all_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_evidence_repair_claimreextract820_plus_negaligned377_vlm139_v1_20260614.jsonl`

## Outputs

- report json: `data/final/repaired_v1/full_pair_reconstruction_llm_evidence_repair_claimreextract820_plus_negaligned377_vlm139_v1_audit_20260614.json`
- flagged rows: `data/final/repaired_v1/full_pair_reconstruction_llm_evidence_repair_claimreextract820_plus_negaligned377_vlm139_v1_flags_20260614.jsonl`

## Summary

- `queue_rows`: `139`
- `review_rows`: `139`
- `matched_reviews`: `139`
- `missing_reviews`: `0`
- `duplicate_review_pairs`: `0`
- `duplicate_review_events`: `0`
- `flagged_rows`: `57`
- `high_flag_rows`: `1`
- `medium_or_high_flag_rows`: `56`
- `flag_severity`: `{'medium': 67, 'high': 1, 'low': 2, 'info': 1}`
- `flag_code`: `{'positive_label_missing_product_evidence_for_main': 32, 'claim_not_in_top_srt_prefilter': 14, 'conflicting_comment_relation_requires_silver': 12, 'promote_action_but_not_main_ready': 8, 'numeric_value_judgment_used_as_refute': 1, 'enumeration_claim_evidence_extra_values': 1, 'raw_llm_y_overridden_by_clean_rule': 2, 'mechanism_contradiction_without_consumer_refute': 1}`
- `promotion_state`: `{'repair_missing_claim': 26, 'silver_refute_missing_product_evidence': 32, 'main_positive_refute': 24, 'silver_refute_insufficient_product_evidence': 35, 'silver_conflicting_comment_relation': 5, 'lowinfo_no_aligned_comment': 2, 'repair_missing_evidence': 5, 'repair_insufficient_product_evidence': 9, 'repair_numeric_value_judgment': 1}`
- `new_y`: `{'0': 42, '1': 97}`
- `confidence`: `{'high': 135, 'medium': 4}`
- `category`: `{'baby_kids_and_pets': 20, 'smart_home': 19, 'food_and_beverages': 39, 'general': 17, 'sports_and_outdoor': 22, 'apparel_and_underwear': 13, 'shoes_and_bags': 4, 'beauty_and_personal_care': 3, 'jewelry_and_collectibles': 1, 'digital_and_electronics': 1}`
- `queue`: `data/final/repaired_v1/full_pair_evidence_repair_queue_claimreextract820_plus_negaligned377_all_v1_20260614.jsonl`
- `reviews`: `data/final/repaired_v1/full_pair_reconstruction_llm_evidence_repair_claimreextract820_plus_negaligned377_vlm139_v1_20260614.jsonl`
- `flagged`: `data/final/repaired_v1/full_pair_reconstruction_llm_evidence_repair_claimreextract820_plus_negaligned377_vlm139_v1_flags_20260614.jsonl`
- `out`: `data/final/repaired_v1/full_pair_reconstruction_llm_evidence_repair_claimreextract820_plus_negaligned377_vlm139_v1_audit_20260614.json`

## Gate Interpretation

- `high` flags block main promotion until rerun or manual repair.
- `medium` flags require manual sampling or silver routing.
- `mechanism_contradiction_without_consumer_refute` is not a positive label by itself.
- Missing reviews are expected before the LLM pilot has been executed.
