# Full Pair LLM Review Audit v1

This report audits LLM/VLM reconstruction reviews before promotion.
It checks label-definition consistency rather than benchmark separability.

## Inputs

- queue: `data/final/repaired_v1/full_pair_evidence_repair_queue_lownoise40_silver12_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_lownoise40_silver12_vlm_v1_20260614.jsonl`

## Outputs

- report json: `data/final/repaired_v1/full_pair_reconstruction_llm_lownoise40_silver12_vlm_audit_v2_20260614.report.json`
- flagged rows: `data/final/repaired_v1/full_pair_reconstruction_llm_lownoise40_silver12_vlm_audit_flags_v2_20260614.jsonl`

## Summary

- `queue_rows`: `12`
- `review_rows`: `12`
- `matched_reviews`: `12`
- `missing_reviews`: `0`
- `duplicate_review_pairs`: `0`
- `duplicate_review_events`: `0`
- `flagged_rows`: `4`
- `high_flag_rows`: `1`
- `medium_or_high_flag_rows`: `4`
- `flag_severity`: `{'medium': 4, 'high': 1}`
- `flag_code`: `{'positive_label_missing_product_evidence_for_main': 2, 'enumeration_claim_evidence_extra_values': 1, 'promote_action_but_not_main_ready': 1, 'identity_attribute_claim_lacks_value': 1}`
- `promotion_state`: `{'repair_missing_evidence': 1, 'silver_refute_missing_product_evidence': 2, 'repair_missing_claim': 2, 'main_positive_refute': 2, 'silver_refute_insufficient_product_evidence': 3, 'repair_insufficient_product_evidence': 1, 'silver_enumeration_evidence_extra_values': 1}`
- `new_y`: `{'0': 4, '1': 8}`
- `confidence`: `{'high': 12}`
- `category`: `{'food_and_beverages': 4, 'smart_home': 5, 'general': 1, 'baby_kids_and_pets': 2}`
- `queue`: `data/final/repaired_v1/full_pair_evidence_repair_queue_lownoise40_silver12_v1_20260614.jsonl`
- `reviews`: `data/final/repaired_v1/full_pair_reconstruction_llm_lownoise40_silver12_vlm_v1_20260614.jsonl`
- `flagged`: `data/final/repaired_v1/full_pair_reconstruction_llm_lownoise40_silver12_vlm_audit_flags_v2_20260614.jsonl`
- `out`: `data/final/repaired_v1/full_pair_reconstruction_llm_lownoise40_silver12_vlm_audit_v2_20260614.report.json`

## Gate Interpretation

- `high` flags block main promotion until rerun or manual repair.
- `medium` flags require manual sampling or silver routing.
- `mechanism_contradiction_without_consumer_refute` is not a positive label by itself.
- Missing reviews are expected before the LLM pilot has been executed.
