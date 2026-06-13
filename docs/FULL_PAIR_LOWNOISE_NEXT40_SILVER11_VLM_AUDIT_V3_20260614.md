# Full Pair LLM Review Audit v1

This report audits LLM/VLM reconstruction reviews before promotion.
It checks label-definition consistency rather than benchmark separability.

## Inputs

- queue: `data/final/repaired_v1/full_pair_evidence_repair_queue_lownoise_next40_silver11_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_lownoise_next40_silver11_vlm_v1_20260614.jsonl`

## Outputs

- report json: `data/final/repaired_v1/full_pair_reconstruction_llm_lownoise_next40_silver11_vlm_audit_v3_20260614.report.json`
- flagged rows: `data/final/repaired_v1/full_pair_reconstruction_llm_lownoise_next40_silver11_vlm_audit_flags_v3_20260614.jsonl`

## Summary

- `queue_rows`: `11`
- `review_rows`: `11`
- `matched_reviews`: `11`
- `missing_reviews`: `0`
- `duplicate_review_pairs`: `0`
- `duplicate_review_events`: `0`
- `flagged_rows`: `6`
- `high_flag_rows`: `0`
- `medium_or_high_flag_rows`: `6`
- `flag_severity`: `{'medium': 8}`
- `flag_code`: `{'positive_label_missing_product_evidence_for_main': 4, 'claim_not_in_top_srt_prefilter': 2, 'price_value_not_direct_refute_requires_silver': 1, 'promote_action_but_not_main_ready': 1}`
- `promotion_state`: `{'silver_refute_missing_product_evidence': 4, 'silver_price_value_not_direct_refute': 1, 'repair_missing_evidence': 1, 'repair_missing_claim': 1, 'main_positive_refute': 3, 'silver_refute_insufficient_product_evidence': 1}`
- `new_y`: `{'1': 9, '0': 2}`
- `confidence`: `{'high': 11}`
- `category`: `{'food_and_beverages': 7, 'baby_kids_and_pets': 3, 'general': 1}`
- `queue`: `data/final/repaired_v1/full_pair_evidence_repair_queue_lownoise_next40_silver11_v1_20260614.jsonl`
- `reviews`: `data/final/repaired_v1/full_pair_reconstruction_llm_lownoise_next40_silver11_vlm_v1_20260614.jsonl`
- `flagged`: `data/final/repaired_v1/full_pair_reconstruction_llm_lownoise_next40_silver11_vlm_audit_flags_v3_20260614.jsonl`
- `out`: `data/final/repaired_v1/full_pair_reconstruction_llm_lownoise_next40_silver11_vlm_audit_v3_20260614.report.json`

## Gate Interpretation

- `high` flags block main promotion until rerun or manual repair.
- `medium` flags require manual sampling or silver routing.
- `mechanism_contradiction_without_consumer_refute` is not a positive label by itself.
- Missing reviews are expected before the LLM pilot has been executed.
