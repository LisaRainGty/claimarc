# Full Pair LLM Review Audit v1

This report audits LLM/VLM reconstruction reviews before promotion.
It checks label-definition consistency rather than benchmark separability.

## Inputs

- queue: `data/final/repaired_v1/full_pair_llm_pilot_queue_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_pilot20_noimg_v1_20260614.jsonl`

## Outputs

- report json: `data/final/repaired_v1/full_pair_reconstruction_llm_pilot20_noimg_audit_v2_20260614.report.json`
- flagged rows: `data/final/repaired_v1/full_pair_reconstruction_llm_pilot20_noimg_audit_flags_v2_20260614.jsonl`

## Summary

- `queue_rows`: `72`
- `review_rows`: `20`
- `matched_reviews`: `20`
- `missing_reviews`: `52`
- `duplicate_review_pairs`: `0`
- `duplicate_review_events`: `0`
- `flagged_rows`: `53`
- `high_flag_rows`: `1`
- `medium_or_high_flag_rows`: `1`
- `flag_severity`: `{'high': 1, 'medium': 1, 'missing': 52}`
- `flag_code`: `{'promote_with_insufficient_claim_evidence_relation': 1, 'promote_action_but_not_main_ready': 1, 'missing_review': 52}`
- `promotion_state`: `{'repair_missing_claim': 12, 'main_positive_refute': 5, 'silver_refute_insufficient_product_evidence': 1, 'main_negative_support': 2, 'missing_review': 52}`
- `new_y`: `{'0': 14, '1': 6, 'None': 52}`
- `confidence`: `{'high': 20}`
- `category`: `{'apparel_and_underwear': 11, 'digital_and_electronics': 11, 'shoes_and_bags': 7, 'baby_kids_and_pets': 8, 'beauty_and_personal_care': 8, 'food_and_beverages': 6, 'general': 6, 'jewelry_and_collectibles': 5, 'smart_home': 5, 'sports_and_outdoor': 5}`
- `queue`: `data/final/repaired_v1/full_pair_llm_pilot_queue_v1_20260614.jsonl`
- `reviews`: `data/final/repaired_v1/full_pair_reconstruction_llm_pilot20_noimg_v1_20260614.jsonl`
- `flagged`: `data/final/repaired_v1/full_pair_reconstruction_llm_pilot20_noimg_audit_flags_v2_20260614.jsonl`
- `out`: `data/final/repaired_v1/full_pair_reconstruction_llm_pilot20_noimg_audit_v2_20260614.report.json`

## Gate Interpretation

- `high` flags block main promotion until rerun or manual repair.
- `medium` flags require manual sampling or silver routing.
- `mechanism_contradiction_without_consumer_refute` is not a positive label by itself.
- Missing reviews are expected before the LLM pilot has been executed.
