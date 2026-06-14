# Full Pair LLM Review Audit v1

This report audits LLM/VLM reconstruction reviews before promotion.
It checks label-definition consistency rather than benchmark separability.

## Inputs

- queue: `data/final/repaired_v1/full_pair_evidence_repair_queue_weak194_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_evidence_repair_weak194_vlm_v1_20260614.jsonl`

## Outputs

- report json: `data/final/repaired_v1/full_pair_reconstruction_llm_evidence_repair_weak194_vlm_v1_audit_20260614.json`
- flagged rows: `data/final/repaired_v1/full_pair_reconstruction_llm_evidence_repair_weak194_vlm_v1_flags_20260614.jsonl`

## Summary

- `queue_rows`: `7`
- `review_rows`: `7`
- `matched_reviews`: `7`
- `missing_reviews`: `0`
- `duplicate_review_pairs`: `0`
- `duplicate_review_events`: `0`
- `flagged_rows`: `2`
- `high_flag_rows`: `0`
- `medium_or_high_flag_rows`: `2`
- `flag_severity`: `{'medium': 3}`
- `flag_code`: `{'conflicting_comment_relation_requires_silver': 2, 'positive_label_missing_product_evidence_for_main': 1}`
- `promotion_state`: `{'repair_missing_claim': 2, 'silver_refute_insufficient_product_evidence': 3, 'silver_refute_missing_product_evidence': 1, 'main_positive_refute': 1}`
- `new_y`: `{'0': 2, '1': 5}`
- `confidence`: `{'high': 7}`
- `category`: `{'shoes_and_bags': 2, 'beauty_and_personal_care': 2, 'apparel_and_underwear': 1, 'smart_home': 1, 'sports_and_outdoor': 1}`
- `queue`: `data/final/repaired_v1/full_pair_evidence_repair_queue_weak194_v1_20260614.jsonl`
- `reviews`: `data/final/repaired_v1/full_pair_reconstruction_llm_evidence_repair_weak194_vlm_v1_20260614.jsonl`
- `flagged`: `data/final/repaired_v1/full_pair_reconstruction_llm_evidence_repair_weak194_vlm_v1_flags_20260614.jsonl`
- `out`: `data/final/repaired_v1/full_pair_reconstruction_llm_evidence_repair_weak194_vlm_v1_audit_20260614.json`

## Gate Interpretation

- `high` flags block main promotion until rerun or manual repair.
- `medium` flags require manual sampling or silver routing.
- `mechanism_contradiction_without_consumer_refute` is not a positive label by itself.
- Missing reviews are expected before the LLM pilot has been executed.
