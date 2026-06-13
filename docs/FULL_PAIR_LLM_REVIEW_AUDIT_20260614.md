# Full Pair LLM Review Audit v1

This report audits LLM/VLM reconstruction reviews before promotion.
It checks label-definition consistency rather than benchmark separability.

## Inputs

- queue: `data/final/repaired_v1/full_pair_llm_pilot_queue_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_v1_20260614.jsonl`

## Outputs

- report json: `data/final/repaired_v1/full_pair_reconstruction_llm_audit_v1_20260614.report.json`
- flagged rows: `data/final/repaired_v1/full_pair_reconstruction_llm_audit_flags_v1_20260614.jsonl`

## Summary

- `queue_rows`: `72`
- `review_rows`: `0`
- `matched_reviews`: `0`
- `missing_reviews`: `72`
- `duplicate_review_pairs`: `0`
- `duplicate_review_events`: `0`
- `flagged_rows`: `72`
- `high_flag_rows`: `0`
- `medium_or_high_flag_rows`: `0`
- `flag_severity`: `{'missing': 72}`
- `flag_code`: `{'missing_review': 72}`
- `promotion_state`: `{'missing_review': 72}`
- `new_y`: `{'None': 72}`
- `confidence`: `{}`
- `category`: `{'apparel_and_underwear': 11, 'digital_and_electronics': 11, 'shoes_and_bags': 7, 'baby_kids_and_pets': 8, 'beauty_and_personal_care': 8, 'food_and_beverages': 6, 'general': 6, 'jewelry_and_collectibles': 5, 'smart_home': 5, 'sports_and_outdoor': 5}`
- `queue`: `data/final/repaired_v1/full_pair_llm_pilot_queue_v1_20260614.jsonl`
- `reviews`: `data/final/repaired_v1/full_pair_reconstruction_llm_v1_20260614.jsonl`
- `flagged`: `data/final/repaired_v1/full_pair_reconstruction_llm_audit_flags_v1_20260614.jsonl`
- `out`: `data/final/repaired_v1/full_pair_reconstruction_llm_audit_v1_20260614.report.json`

## Gate Interpretation

- `high` flags block main promotion until rerun or manual repair.
- `medium` flags require manual sampling or silver routing.
- `mechanism_contradiction_without_consumer_refute` is not a positive label by itself.
- Missing reviews are expected before the LLM pilot has been executed.
