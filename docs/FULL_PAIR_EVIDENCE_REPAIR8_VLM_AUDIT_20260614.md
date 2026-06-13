# Full Pair LLM Review Audit v1

This report audits LLM/VLM reconstruction reviews before promotion.
It checks label-definition consistency rather than benchmark separability.

## Inputs

- queue: `data/final/repaired_v1/full_pair_evidence_repair_queue_pilot72_silverpos_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_evidence_repair8_vlm_v1_20260614.jsonl`

## Outputs

- report json: `data/final/repaired_v1/full_pair_reconstruction_llm_evidence_repair8_vlm_audit_v1_20260614.report.json`
- flagged rows: `data/final/repaired_v1/full_pair_reconstruction_llm_evidence_repair8_vlm_audit_flags_v1_20260614.jsonl`

## Summary

- `queue_rows`: `8`
- `review_rows`: `8`
- `matched_reviews`: `8`
- `missing_reviews`: `0`
- `duplicate_review_pairs`: `0`
- `duplicate_review_events`: `0`
- `flagged_rows`: `4`
- `high_flag_rows`: `0`
- `medium_or_high_flag_rows`: `4`
- `flag_severity`: `{'medium': 4}`
- `flag_code`: `{'positive_label_missing_product_evidence_for_main': 3, 'claim_not_in_top_srt_prefilter': 1}`
- `promotion_state`: `{'main_positive_refute': 3, 'silver_refute_missing_product_evidence': 3, 'silver_refute_insufficient_product_evidence': 2}`
- `new_y`: `{'1': 8}`
- `confidence`: `{'high': 8}`
- `category`: `{'baby_kids_and_pets': 1, 'general': 1, 'apparel_and_underwear': 2, 'digital_and_electronics': 2, 'food_and_beverages': 1, 'sports_and_outdoor': 1}`
- `queue`: `data/final/repaired_v1/full_pair_evidence_repair_queue_pilot72_silverpos_v1_20260614.jsonl`
- `reviews`: `data/final/repaired_v1/full_pair_reconstruction_llm_evidence_repair8_vlm_v1_20260614.jsonl`
- `flagged`: `data/final/repaired_v1/full_pair_reconstruction_llm_evidence_repair8_vlm_audit_flags_v1_20260614.jsonl`
- `out`: `data/final/repaired_v1/full_pair_reconstruction_llm_evidence_repair8_vlm_audit_v1_20260614.report.json`

## Gate Interpretation

- `high` flags block main promotion until rerun or manual repair.
- `medium` flags require manual sampling or silver routing.
- `mechanism_contradiction_without_consumer_refute` is not a positive label by itself.
- Missing reviews are expected before the LLM pilot has been executed.
